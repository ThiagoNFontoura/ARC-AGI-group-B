import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from .llm_handler import GemmaHandler
from .prompt import build_prompt
from .strategy import AugmentedInference
from .transforms import get_transformations

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_arc_task(payload: Any) -> bool:
    """Return whether a decoded payload has ARC train and test arrays."""
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


def _extract_colors_from_grid(grid: Any) -> set[int]:
    """Extract all distinct integer colors from a 2D grid."""
    colors = set()
    if isinstance(grid, list):
        for row in grid:
            if isinstance(row, list):
                for cell in row:
                    if isinstance(cell, int):
                        colors.add(cell)
    return colors


def _extract_task_context_colors(task: dict[str, Any]) -> set[int]:
    """Collect all colors present in train inputs/outputs and test inputs."""
    colors = set()
    for case in task.get("train", []):
        if isinstance(case, dict):
            colors |= _extract_colors_from_grid(case.get("input"))
            colors |= _extract_colors_from_grid(case.get("output"))
    for case in task.get("test", []):
        if isinstance(case, dict):
            colors |= _extract_colors_from_grid(case.get("input"))
    return colors


def _load_tasks(tasks_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid ARC task files and record files that must be skipped."""
    tasks: list[dict[str, Any]] = []
    skipped: list[str] = []

    for file_path in sorted(tasks_dir.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            skipped.append(f"{file_path.name}: invalid JSON")
            continue

        if not _is_arc_task(payload):
            skipped.append(f"{file_path.name}: missing ARC train/test arrays")
            continue

        task_name = payload.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            task_name = file_path.stem

        tasks.append(
            {
                "task_name": task_name,
                "source_file": file_path.name,
                "train": payload.get("train", []),
                "test": payload.get("test", []),
            }
        )

    return tasks, skipped


def _evaluate_correctness(
    task: dict[str, Any], predicted_outputs: Any
) -> tuple[str, str, dict[str, Any]]:
    """Compare predictions with test outputs and compute ARC metrics (shape, pixel accuracy, color palette)."""
    test_cases = task.get("test", [])

    has_ground_truth_for_all = True
    ground_truth_outputs: list[Any] = []
    for case in test_cases:
        if isinstance(case, dict) and "output" in case:
            ground_truth_outputs.append(case["output"])
        else:
            has_ground_truth_for_all = False

    context_colors = _extract_task_context_colors(task)
    pred_colors = set()
    if isinstance(predicted_outputs, list):
        for grid in predicted_outputs:
            pred_colors |= _extract_colors_from_grid(grid)

    gt_colors = set()
    for grid in ground_truth_outputs:
        gt_colors |= _extract_colors_from_grid(grid)

    arc_metrics: dict[str, Any] = {
        "shape_match": False,
        "pixel_accuracy_percentage": 0.0,
        "total_test_cases": len(test_cases),
        "test_case_details": [],
        "color_evaluation": {
            "predicted_colors": sorted(pred_colors),
            "ground_truth_colors": sorted(gt_colors),
            "context_colors": sorted(context_colors),
            "palette_match": (pred_colors == gt_colors) if has_ground_truth_for_all else False,
            "palette_preserved": pred_colors.issubset(context_colors),
            "unseen_colors_introduced": sorted(pred_colors - context_colors),
            "missing_expected_colors": (
                sorted(gt_colors - pred_colors) if has_ground_truth_for_all else []
            ),
        },
    }

    if not has_ground_truth_for_all:
        return "unknown", "No full ground truth in test cases.", arc_metrics

    if not isinstance(predicted_outputs, list):
        return "incorrect", "predicted_test_outputs is not a list.", arc_metrics

    if len(predicted_outputs) != len(ground_truth_outputs):
        return "incorrect", "Prediction count does not match test case count.", arc_metrics

    total_gt_pixels = 0
    total_matching_pixels = 0
    all_shapes_match = True

    for case_idx, (gt_grid, pred_grid) in enumerate(
        zip(ground_truth_outputs, predicted_outputs)
    ):
        is_gt_2d = isinstance(gt_grid, list) and all(isinstance(r, list) for r in gt_grid)
        is_pred_2d = isinstance(pred_grid, list) and all(
            isinstance(r, list) for r in pred_grid
        )

        gt_h = len(gt_grid) if is_gt_2d else 0
        gt_w = len(gt_grid[0]) if (is_gt_2d and gt_h > 0) else 0

        pred_h = len(pred_grid) if is_pred_2d else 0
        pred_w = len(pred_grid[0]) if (is_pred_2d and pred_h > 0) else 0

        shape_match = is_gt_2d and is_pred_2d and gt_h == pred_h and gt_w == pred_w
        if not shape_match:
            all_shapes_match = False

        case_gt_pixels = gt_h * gt_w
        case_matching_pixels = 0

        if is_gt_2d and is_pred_2d:
            overlap_h = min(gt_h, pred_h)
            overlap_w = min(gt_w, pred_w)
            for r in range(overlap_h):
                for c in range(overlap_w):
                    if pred_grid[r][c] == gt_grid[r][c]:
                        case_matching_pixels += 1

        case_pixel_acc = (
            round((case_matching_pixels / case_gt_pixels) * 100, 2)
            if case_gt_pixels > 0
            else 0.0
        )

        total_gt_pixels += case_gt_pixels
        total_matching_pixels += case_matching_pixels

        arc_metrics["test_case_details"].append(
            {
                "test_case_index": case_idx + 1,
                "ground_truth_shape": [gt_h, gt_w],
                "predicted_shape": [pred_h, pred_w] if is_pred_2d else None,
                "shape_match": shape_match,
                "matching_pixels": case_matching_pixels,
                "total_pixels": case_gt_pixels,
                "pixel_accuracy_percentage": case_pixel_acc,
                "exact_match": (pred_grid == gt_grid),
            }
        )

    overall_pixel_acc = (
        round((total_matching_pixels / total_gt_pixels) * 100, 2)
        if total_gt_pixels > 0
        else 0.0
    )

    arc_metrics["shape_match"] = all_shapes_match
    arc_metrics["pixel_accuracy_percentage"] = overall_pixel_acc

    if predicted_outputs == ground_truth_outputs:
        return "correct", "All predicted test outputs match ground truth.", arc_metrics

    return "incorrect", "Predicted outputs differ from ground truth.", arc_metrics


def run_inference(
    tasks: list[dict[str, Any]],
    skipped: list[str],
    tasks_folder_name: str,
    transform_names: list[str] | None,
    output_path: Path,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    model_name = "gemma-4-31b-it"
    by_name: dict[str, dict[str, Any]] = {}
    task_errors_by_name: dict[str, str] = {}
    try:
        handler = GemmaHandler()
        model_name = handler.model
        augmented_inference = AugmentedInference(
            solve_prompt=handler.solve,
            build_prompt=build_prompt,
            transformations=get_transformations(transform_names),
        )

        total_tasks = len(tasks)
        for prompt_index, task in enumerate(tasks, start=1):
            t_name = task["task_name"]
            try:
                result = augmented_inference.solve_task(task, prompt_index=prompt_index)
                if "predicted_test_outputs" in result:
                    by_name[t_name] = result

                    ens = result.get("ensemble_details", {})
                    summary = ens.get("summary", {})
                    telemetry = ens.get("telemetry", {})
                    latency = telemetry.get("total_latency_seconds", 0.0)
                    consensus = summary.get("consensus_type", "done")
                    votes = summary.get("winner_vote_ratio", "")
                    total_views = summary.get("total_views_attempted", 1)

                    status, _, arc_eval = _evaluate_correctness(
                        task, result.get("predicted_test_outputs")
                    )
                    status_label = (
                        "CORRECT"
                        if status == "correct"
                        else ("INCORRECT" if status == "incorrect" else "COMPLETED")
                    )
                    px_acc = arc_eval.get("pixel_accuracy_percentage", 0.0)

                    if total_views > 1:
                        details_str = f"{latency}s | votes: {votes} ({consensus}) | px acc: {px_acc}%"
                    else:
                        details_str = f"{latency}s | px acc: {px_acc}%"

                    print(f"  [{prompt_index:02d}/{total_tasks:02d}] Task '{t_name}' -> {status_label} ({details_str})")
                else:
                    err_msg = "; ".join(
                        result.get("_augmentation_errors", ["No prediction returned."])
                    )
                    task_errors_by_name[t_name] = err_msg
                    print(f"  [{prompt_index:02d}/{total_tasks:02d}] Task '{t_name}' -> FAILED ({err_msg})")
            except Exception as exc:
                task_errors_by_name[t_name] = f"Augmented inference failed: {exc}"
                print(f"  [{prompt_index:02d}/{total_tasks:02d}] Task '{t_name}' -> ERROR: {exc}")
    except Exception as exc:
        error_text = f"LLM setup failed: {exc}"
        for task in tasks:
            task_errors_by_name[task["task_name"]] = error_text
        print(f"  ERROR during LLM setup: {exc}")

    output_tasks: list[dict[str, Any]] = []
    correct_count = 0
    incorrect_count = 0
    unknown_count = 0
    consensus_counts: dict[str, int] = defaultdict(int)
    transform_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "attempted": 0,
            "correct": 0,
            "incorrect": 0,
            "voted_winner": 0,
            "pixel_accuracies": [],
        }
    )

    pixel_acc_list: list[float] = []
    shape_match_count = 0
    color_preserved_count = 0
    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_tokens = 0

    for task in tasks:
        model_task = by_name.get(task["task_name"], {})
        logic_explanation = model_task.get("logic_explanation", "")
        if not isinstance(logic_explanation, str) or not logic_explanation.strip():
            logic_explanation = "No explanation returned by model."

        predicted_outputs = model_task.get("predicted_test_outputs", [])

        if task["task_name"] in task_errors_by_name:
            status, details, arc_metrics = (
                "unknown",
                task_errors_by_name[task["task_name"]],
                {"shape_match": False, "pixel_accuracy_percentage": 0.0},
            )
        elif task["task_name"] not in by_name:
            status, details, arc_metrics = (
                "unknown",
                "Task missing in model response.",
                {"shape_match": False, "pixel_accuracy_percentage": 0.0},
            )
        else:
            status, details, arc_metrics = _evaluate_correctness(task, predicted_outputs)

        if status == "correct":
            correct_count += 1
        elif status == "incorrect":
            incorrect_count += 1
        else:
            unknown_count += 1

        if status in ("correct", "incorrect"):
            pixel_acc_list.append(arc_metrics.get("pixel_accuracy_percentage", 0.0))
            if arc_metrics.get("shape_match"):
                shape_match_count += 1
            if arc_metrics.get("color_evaluation", {}).get("palette_preserved"):
                color_preserved_count += 1

        ensemble_details = model_task.get("ensemble_details")
        if ensemble_details:
            consensus_type = ensemble_details.get("summary", {}).get(
                "consensus_type", "unknown"
            )
            consensus_counts[consensus_type] += 1

            # Accumulate task tokens if present
            task_telemetry = ensemble_details.get("telemetry", {})
            if task_telemetry.get("prompt_tokens"):
                total_prompt_tokens += task_telemetry["prompt_tokens"]
            if task_telemetry.get("candidates_tokens"):
                total_candidates_tokens += task_telemetry["candidates_tokens"]
            if task_telemetry.get("total_tokens"):
                total_tokens += task_telemetry["total_tokens"]

            # Evaluate each individual view's prediction
            for view in ensemble_details.get("individual_views", []):
                t_name = view.get("transform", "unknown")
                transform_stats[t_name]["attempted"] += 1

                if view.get("status") == "success":
                    view_status, view_details, view_arc_metrics = _evaluate_correctness(
                        task, view.get("predicted_test_outputs")
                    )
                    view["correctness_result"] = {
                        "status": view_status,
                        "details": view_details,
                    }
                    view["arc_metrics"] = view_arc_metrics

                    if view_status == "correct":
                        transform_stats[t_name]["correct"] += 1
                    elif view_status == "incorrect":
                        transform_stats[t_name]["incorrect"] += 1

                    if view_status in ("correct", "incorrect"):
                        transform_stats[t_name]["pixel_accuracies"].append(
                            view_arc_metrics.get("pixel_accuracy_percentage", 0.0)
                        )

                    winner_transforms = ensemble_details.get("summary", {}).get(
                        "winner_transforms", []
                    )
                    if t_name in winner_transforms:
                        transform_stats[t_name]["voted_winner"] += 1

        task_entry: dict[str, Any] = {
            "task_name": task["task_name"],
            "logic_explanation": logic_explanation,
            "correctness_result": {
                "status": status,
                "details": details,
            },
            "arc_metrics": arc_metrics,
            "predicted_test_outputs": predicted_outputs,
            "source_file": task["source_file"],
        }
        if ensemble_details is not None:
            task_entry["ensemble_details"] = ensemble_details

        output_tasks.append(task_entry)

    total_evaluated = correct_count + incorrect_count
    accuracy = round((correct_count / total_evaluated * 100), 2) if total_evaluated > 0 else 0.0
    avg_pixel_acc = (
        round(sum(pixel_acc_list) / len(pixel_acc_list), 2) if pixel_acc_list else 0.0
    )
    shape_match_pct = (
        round((shape_match_count / total_evaluated) * 100, 2) if total_evaluated > 0 else 0.0
    )
    color_pres_pct = (
        round((color_preserved_count / total_evaluated) * 100, 2)
        if total_evaluated > 0
        else 0.0
    )
    total_duration = round(time.perf_counter() - start_time, 2)

    transform_performance: dict[str, Any] = {}
    for t_name, stats in sorted(transform_stats.items()):
        t_eval = stats["correct"] + stats["incorrect"]
        t_acc = round((stats["correct"] / t_eval * 100), 2) if t_eval > 0 else 0.0
        t_px_list = stats["pixel_accuracies"]
        t_avg_px = round(sum(t_px_list) / len(t_px_list), 2) if t_px_list else 0.0
        transform_performance[t_name] = {
            "attempted": stats["attempted"],
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "accuracy_percentage": t_acc,
            "average_pixel_accuracy_percentage": t_avg_px,
            "times_voted_winner": stats["voted_winner"],
        }

    output = {
        "prompt_index": 1,
        "prompt_chunks": len(tasks),
        "tasks_folder": tasks_folder_name,
        "augmentation_transforms": [
            transform.name for transform in get_transformations(transform_names)
        ],
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_files": skipped,
        "summary": {
            "total_tasks": len(tasks),
            "correct_tasks": correct_count,
            "incorrect_tasks": incorrect_count,
            "unknown_tasks": unknown_count,
            "accuracy_percentage": accuracy,
            "average_pixel_accuracy_percentage": avg_pixel_acc,
            "shape_match_percentage": shape_match_pct,
            "color_preservation_percentage": color_pres_pct,
            "total_duration_seconds": total_duration,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens if total_prompt_tokens > 0 else None,
                "candidates_tokens": (
                    total_candidates_tokens if total_candidates_tokens > 0 else None
                ),
                "total_tokens": total_tokens if total_tokens > 0 else None,
            },
            "consensus_distribution": dict(consensus_counts),
            "transforms_performance": transform_performance,
        },
        "tasks": output_tasks,
    }

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Processed tasks: {len(tasks)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Accuracy: {accuracy}% ({correct_count}/{total_evaluated} correct)")
    print(f"Avg Pixel Accuracy: {avg_pixel_acc}% | Shape Match: {shape_match_pct}%")
    print(f"Duration: {total_duration}s")
    print(f"Output saved to: {output_path}")

    return output


def generate_comparison_report(
    baseline_result: dict[str, Any],
    augmented_result: dict[str, Any],
    tasks_folder_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Compare baseline vs augmented inference results and generate a structured report."""
    baseline_tasks = {t["task_name"]: t for t in baseline_result.get("tasks", [])}
    augmented_tasks = {t["task_name"]: t for t in augmented_result.get("tasks", [])}

    all_task_names = sorted(set(baseline_tasks.keys()) | set(augmented_tasks.keys()))

    fixed_by_augmentation: list[dict[str, Any]] = []
    regressed_by_augmentation: list[dict[str, Any]] = []
    both_correct: list[dict[str, Any]] = []
    both_incorrect: list[dict[str, Any]] = []
    other_cases: list[dict[str, Any]] = []

    for task_name in all_task_names:
        b_task = baseline_tasks.get(task_name)
        a_task = augmented_tasks.get(task_name)

        b_status = (
            b_task.get("correctness_result", {}).get("status", "unknown")
            if b_task
            else "missing"
        )
        a_status = (
            a_task.get("correctness_result", {}).get("status", "unknown")
            if a_task
            else "missing"
        )

        b_arc = (b_task or {}).get("arc_metrics", {})
        a_arc = (a_task or {}).get("arc_metrics", {})
        ens_summary = a_task.get("ensemble_details", {}).get("summary", {}) if a_task else {}

        task_comparison = {
            "task_name": task_name,
            "source_file": (a_task or b_task or {}).get("source_file", ""),
            "baseline_status": b_status,
            "augmented_status": a_status,
            "baseline_pixel_accuracy": b_arc.get("pixel_accuracy_percentage", 0.0),
            "augmented_pixel_accuracy": a_arc.get("pixel_accuracy_percentage", 0.0),
            "baseline_shape_match": b_arc.get("shape_match", False),
            "augmented_shape_match": a_arc.get("shape_match", False),
            "augmented_consensus_type": ens_summary.get("consensus_type", "unknown"),
            "augmented_winner_votes": ens_summary.get("winner_vote_ratio", "N/A"),
            "augmented_winner_transforms": ens_summary.get("winner_transforms", []),
        }

        if b_status != "correct" and a_status == "correct":
            fixed_by_augmentation.append(task_comparison)
        elif b_status == "correct" and a_status != "correct":
            regressed_by_augmentation.append(task_comparison)
        elif b_status == "correct" and a_status == "correct":
            both_correct.append(task_comparison)
        elif b_status == "incorrect" and a_status == "incorrect":
            both_incorrect.append(task_comparison)
        else:
            other_cases.append(task_comparison)

    b_summary = baseline_result.get("summary", {})
    a_summary = augmented_result.get("summary", {})

    b_acc = b_summary.get("accuracy_percentage", 0.0)
    a_acc = a_summary.get("accuracy_percentage", 0.0)
    acc_delta = round(a_acc - b_acc, 2)

    b_px_acc = b_summary.get("average_pixel_accuracy_percentage", 0.0)
    a_px_acc = a_summary.get("average_pixel_accuracy_percentage", 0.0)
    px_acc_delta = round(a_px_acc - b_px_acc, 2)

    b_shape = b_summary.get("shape_match_percentage", 0.0)
    a_shape = a_summary.get("shape_match_percentage", 0.0)
    shape_delta = round(a_shape - b_shape, 2)

    net_gained = len(fixed_by_augmentation) - len(regressed_by_augmentation)

    report = {
        "tasks_folder": tasks_folder_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": augmented_result.get("model", "unknown"),
        "summary": {
            "total_tasks_evaluated": len(all_task_names),
            "baseline_accuracy_percentage": b_acc,
            "augmented_accuracy_percentage": a_acc,
            "accuracy_delta_percentage": acc_delta,
            "baseline_average_pixel_accuracy_percentage": b_px_acc,
            "augmented_average_pixel_accuracy_percentage": a_px_acc,
            "pixel_accuracy_delta_percentage": px_acc_delta,
            "baseline_shape_match_percentage": b_shape,
            "augmented_shape_match_percentage": a_shape,
            "shape_match_delta_percentage": shape_delta,
            "baseline_duration_seconds": b_summary.get("total_duration_seconds", 0.0),
            "augmented_duration_seconds": a_summary.get("total_duration_seconds", 0.0),
            "baseline_tokens": b_summary.get("token_usage", {}),
            "augmented_tokens": a_summary.get("token_usage", {}),
            "net_tasks_gained": net_gained,
            "fixed_by_augmentation_count": len(fixed_by_augmentation),
            "regressed_by_augmentation_count": len(regressed_by_augmentation),
            "both_correct_count": len(both_correct),
            "both_incorrect_count": len(both_incorrect),
            "other_count": len(other_cases),
        },
        "fixed_by_augmentation": fixed_by_augmentation,
        "regressed_by_augmentation": regressed_by_augmentation,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "other_cases": other_cases,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fixed_names = ", ".join(t["task_name"] for t in fixed_by_augmentation) or "none"
    regressed_names = ", ".join(t["task_name"] for t in regressed_by_augmentation) or "none"
    delta_sign = "+" if acc_delta > 0 else ""
    px_sign = "+" if px_acc_delta > 0 else ""
    shape_sign = "+" if shape_delta > 0 else ""

    print("\n" + "=" * 70)
    print("                 BASELINE vs. AUGMENTATION COMPARISON")
    print("=" * 70)
    print(f"Tasks Evaluated:         {len(all_task_names)}")
    print(
        f"Exact Task Accuracy:     Baseline: {b_acc}% | Augmented: {a_acc}% (Delta: {delta_sign}{acc_delta}%)"
    )
    print(
        f"Avg Pixel Accuracy:      Baseline: {b_px_acc}% | Augmented: {a_px_acc}% (Delta: {px_sign}{px_acc_delta}%)"
    )
    print(
        f"Shape Match Rate:        Baseline: {b_shape}% | Augmented: {a_shape}% (Delta: {shape_sign}{shape_delta}%)"
    )
    print(
        f"Total Duration:          Baseline: {b_summary.get('total_duration_seconds', 0)}s | Augmented: {a_summary.get('total_duration_seconds', 0)}s"
    )
    print("-" * 70)
    print(f"Fixed by Augmentation:   {len(fixed_by_augmentation)} ({fixed_names})")
    print(f"Regressed by Augment.:   {len(regressed_by_augmentation)} ({regressed_names})")
    print(f"Net Gained Tasks:        {net_gained:+d}")
    print(f"Both Correct:            {len(both_correct)}")
    print(f"Both Incorrect:          {len(both_incorrect)}")
    print("-" * 70)
    print(f"Comparison report saved to: {output_path}")
    print("=" * 70 + "\n")

    return report


def main() -> None:
    """Run the solver and save one normalized result file for the task folder."""
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver with a single Gemma prompt.")
    parser.add_argument(
        "tasks_folder_name",
        help="Folder name under data/ that contains one or more ARC task JSON files.",
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=None,
        help="Transformations to use (default: identity flip_horizontal flip_vertical transpose).",
    )
    parser.add_argument(
        "--run-both",
        action="store_true",
        help="Run both with and without data augmentation and generate a comparison report.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    data_dir = PROJECT_ROOT / "data"
    tasks_dir = data_dir / args.tasks_folder_name

    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise SystemExit(f"Tasks folder not found: {tasks_dir}")

    tasks, skipped = _load_tasks(tasks_dir)
    if not tasks:
        raise SystemExit(
            "No ARC task files found. Ensure at least one JSON file with train/test arrays exists."
        )

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_result = None
    if args.run_both:
        print("--- Running WITHOUT data augmentation ---")
        baseline_path = output_dir / f"{args.tasks_folder_name}_baseline_output.json"
        baseline_result = run_inference(
            tasks=tasks,
            skipped=skipped,
            tasks_folder_name=args.tasks_folder_name,
            transform_names=["identity"],
            output_path=baseline_path,
        )
        print("\n--- Running WITH data augmentation ---")
    else:
        print("--- Running WITH data augmentation ---")

    augmented_path = output_dir / f"{args.tasks_folder_name}_augmented_output.json"
    augmented_result = run_inference(
        tasks=tasks,
        skipped=skipped,
        tasks_folder_name=args.tasks_folder_name,
        transform_names=args.transforms,
        output_path=augmented_path,
    )

    if baseline_result is not None:
        report_path = output_dir / f"{args.tasks_folder_name}_comparison_report.json"
        generate_comparison_report(
            baseline_result=baseline_result,
            augmented_result=augmented_result,
            tasks_folder_name=args.tasks_folder_name,
            output_path=report_path,
        )


if __name__ == "__main__":
    main()
