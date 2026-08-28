import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from models.image_baseline_model.llm_handler import GemmaHandler
from models.image_baseline_model.prompt import (
    build_image_prompt,
    build_image_validation_prompt,
    build_json_validation_prompt,
    build_prompt,
)
from models.image_baseline_model.task_image_renderer import (
    default_render_style,
    render_grid_image,
    render_tasks_folder_parallel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_arc_task(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


def _load_tasks(tasks_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    tasks: list[dict[str, Any]] = []
    skipped: list[str] = []

    for file_path in sorted(tasks_dir.glob("*.json")):
        if re.fullmatch(r"\d+-(?:json|image)\.json", file_path.name):
            continue
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


def _next_prompt_index(tasks_dir: Path) -> int:
    max_index = 0
    for file_path in tasks_dir.glob("*.json"):
        match = re.fullmatch(r"(\d+)(?:-(?:json|image))?", file_path.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _evaluate_correctness(task: dict[str, Any], predicted_outputs: Any) -> tuple[str, str]:
    test_cases = task.get("test", [])

    has_ground_truth_for_all = True
    ground_truth_outputs: list[Any] = []
    for case in test_cases:
        if isinstance(case, dict) and "output" in case:
            ground_truth_outputs.append(case["output"])
        else:
            has_ground_truth_for_all = False

    if not has_ground_truth_for_all:
        return "unknown", "No full ground truth in test cases."

    if not isinstance(predicted_outputs, list):
        return "incorrect", "predicted_test_outputs is not a list."

    if len(predicted_outputs) != len(ground_truth_outputs):
        return "incorrect", "Prediction count does not match test case count."

    if predicted_outputs == ground_truth_outputs:
        return "correct", "All predicted test outputs match ground truth."

    return "incorrect", "Predicted outputs differ from ground truth."


def _normalize_validation_status(status: Any) -> str:
    if not isinstance(status, str):
        return "unknown"
    value = status.strip().lower()
    if value in {"correct", "incorrect", "unknown"}:
        return value
    if value in {"match", "matches", "true"}:
        return "correct"
    if value in {"mismatch", "false"}:
        return "incorrect"
    return "unknown"


def _extract_ground_truth_outputs(task: dict[str, Any]) -> tuple[list[Any], bool]:
    test_cases = task.get("test", [])
    has_ground_truth_for_all = True
    ground_truth_outputs: list[Any] = []

    if not isinstance(test_cases, list):
        return [], False

    for case in test_cases:
        if isinstance(case, dict) and "output" in case:
            ground_truth_outputs.append(case["output"])
        else:
            has_ground_truth_for_all = False

    return ground_truth_outputs, has_ground_truth_for_all


def _call_with_timeout(callable_fn: Any, timeout_seconds: int) -> tuple[Any | None, bool]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callable_fn)
        try:
            return future.result(timeout=timeout_seconds), False
        except FutureTimeoutError:
            future.cancel()
            return None, True


def _solve_with_deadline(
    handler: GemmaHandler,
    prompt: str,
    timeout_seconds: int,
    force_response_timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str | None]:
    result, timed_out = _call_with_timeout(lambda: handler.solve(prompt), timeout_seconds)
    if not timed_out:
        return result, None

    force_prompt = (
        "Time limit reached. Return your best final answer now as valid JSON only.\n\n"
        + prompt
    )
    forced_result, forced_timed_out = _call_with_timeout(
        lambda: handler.solve(force_prompt),
        force_response_timeout_seconds,
    )
    if forced_timed_out:
        return None, "response_not_informed"
    return forced_result, "forced_response"


def _solve_with_images_deadline(
    handler: GemmaHandler,
    prompt: str,
    image_inputs: list[tuple[str, Path]],
    timeout_seconds: int,
    force_response_timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str | None]:
    result, timed_out = _call_with_timeout(
        lambda: handler.solve_with_images(prompt, image_inputs),
        timeout_seconds,
    )
    if not timed_out:
        return result, None

    force_prompt = (
        "Time limit reached. Return your best final answer now as valid JSON only.\n\n"
        + prompt
    )
    forced_result, forced_timed_out = _call_with_timeout(
        lambda: handler.solve_with_images(force_prompt, image_inputs),
        force_response_timeout_seconds,
    )
    if forced_timed_out:
        return None, "response_not_informed"
    return forced_result, "forced_response"


def _safe_task_name(task_name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", task_name).strip("_")
    return clean or "task"


def _output_prefix(tasks_folder_name: str, prompt_index: int) -> str:
    root = re.sub(r"[^A-Za-z0-9_.-]", "_", tasks_folder_name).strip("_") or "tasks"
    return f"{root}_{prompt_index:03d}"


def _output_dir_for(tasks_folder_name: str, prompt_index: int) -> Path:
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / _output_prefix(tasks_folder_name, prompt_index)
    output_base.mkdir(parents=True, exist_ok=True)
    return output_base


def _build_image_inputs(tasks: list[dict[str, Any]], images_dir: Path) -> list[tuple[str, Path]]:
    labeled_images: list[tuple[str, Path]] = []
    for task in tasks:
        task_name = task["task_name"]
        task_dir = images_dir / _safe_task_name(task_name)
        for split in ("train", "test"):
            cases = task.get(split, [])
            if not isinstance(cases, list):
                continue
            for index, case in enumerate(cases):
                input_path = task_dir / f"{split}_{index:02d}_input.png"
                if input_path.exists():
                    labeled_images.append(
                        (f"Task {task_name}, {split} case {index}, input image.", input_path)
                    )
                if split == "train":
                    output_path = task_dir / f"{split}_{index:02d}_output.png"
                    if output_path.exists():
                        labeled_images.append(
                            (f"Task {task_name}, {split} case {index}, output image.", output_path)
                        )
    return labeled_images


def _is_int_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) and row for row in grid):
        return False
    cols = len(grid[0])
    if not all(len(row) == cols for row in grid):
        return False
    for row in grid:
        for value in row:
            if not isinstance(value, int):
                return False
    return True


def _build_validation_image_inputs(
    task: dict[str, Any],
    predicted_outputs: Any,
    images_dir: Path,
) -> tuple[list[tuple[str, Path]], list[Any]]:
    task_name = task["task_name"]
    task_dir = images_dir / _safe_task_name(task_name)
    validation_dir = images_dir / "_validation" / _safe_task_name(task_name)
    validation_dir.mkdir(parents=True, exist_ok=True)

    style = default_render_style()
    labeled_images: list[tuple[str, Path]] = []
    ground_truth_outputs: list[Any] = []
    test_cases = task.get("test", []) if isinstance(task.get("test", []), list) else []

    for idx, case in enumerate(test_cases):
        if not isinstance(case, dict) or "output" not in case:
            continue

        ground_truth = case.get("output")
        ground_truth_outputs.append(ground_truth)

        input_path = task_dir / f"test_{idx:02d}_input.png"
        if input_path.exists():
            labeled_images.append(
                (f"Task {task_name}, test case {idx}, input image.", input_path)
            )

        if isinstance(predicted_outputs, list) and idx < len(predicted_outputs):
            predicted_grid = predicted_outputs[idx]
            if _is_int_grid(predicted_grid):
                predicted_path = validation_dir / f"test_{idx:02d}_predicted_output.png"
                render_grid_image(predicted_grid, style).save(predicted_path)
                labeled_images.append(
                    (
                        f"Task {task_name}, test case {idx}, predicted output image.",
                        predicted_path,
                    )
                )

        if _is_int_grid(ground_truth):
            ground_truth_path = validation_dir / f"test_{idx:02d}_ground_truth_output.png"
            render_grid_image(ground_truth, style).save(ground_truth_path)
            labeled_images.append(
                (
                    f"Task {task_name}, test case {idx}, ground-truth output image.",
                    ground_truth_path,
                )
            )

    return labeled_images, ground_truth_outputs


def _validate_image_prediction_with_strong_model(
    handler: GemmaHandler,
    task: dict[str, Any],
    predicted_outputs: Any,
    images_dir: Path,
    validator_model: str,
) -> dict[str, Any]:
    labeled_images, ground_truth_outputs = _build_validation_image_inputs(
        task, predicted_outputs, images_dir
    )

    if not ground_truth_outputs:
        return {
            "validation_status": "unknown",
            "notes": "No test ground-truth outputs available for validation.",
        }

    prompt = build_image_validation_prompt(
        task_name=task["task_name"],
        predicted_test_outputs=predicted_outputs,
        ground_truth_test_outputs=ground_truth_outputs,
    )
    result = handler.solve_with_images(prompt, labeled_images, model=validator_model)
    if not isinstance(result, dict):
        raise ValueError("Strong-model validation did not return a JSON object.")
    return result


def _validate_json_prediction_with_strong_model(
    handler: GemmaHandler,
    task: dict[str, Any],
    predicted_outputs: Any,
    validator_model: str,
) -> dict[str, Any]:
    ground_truth_outputs, has_ground_truth = _extract_ground_truth_outputs(task)
    if not has_ground_truth:
        return {
            "validation_status": "unknown",
            "notes": "No full ground truth in test cases.",
        }

    prompt = build_json_validation_prompt(
        task_name=task["task_name"],
        predicted_test_outputs=predicted_outputs,
        ground_truth_test_outputs=ground_truth_outputs,
    )
    result = handler.solve(prompt, model=validator_model)
    if not isinstance(result, dict):
        raise ValueError("Strong-model JSON validation did not return a JSON object.")
    return result


def _build_report(
    tasks: list[dict[str, Any]],
    skipped: list[str],
    model_name: str,
    prompt_index: int,
    model_result: dict[str, Any],
    task_errors: dict[str, str],
    validation_results: dict[str, dict[str, Any]] | None = None,
    token_usage: dict[str, int] | None = None,
    total_duration_seconds: float = 0.0,
) -> dict[str, Any]:
    returned_tasks = model_result.get("tasks", []) if isinstance(model_result, dict) else []
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(returned_tasks, list):
        for item in returned_tasks:
            if isinstance(item, dict) and isinstance(item.get("task_name"), str):
                by_name[item["task_name"]] = item

    output_tasks: list[dict[str, Any]] = []
    for task in tasks:
        model_task = by_name.get(task["task_name"], {})
        explanation = model_task.get("logic_explanation", "")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "No explanation returned by model."

        predicted_outputs = model_task.get("predicted_test_outputs", [])
        if task["task_name"] in task_errors:
            if task_errors[task["task_name"]] == "response_not_informed":
                status, details = "incorrect", "Resposta nao informada dentro do tempo limite."
            else:
                status, details = "unknown", f"LLM call failed: {task_errors[task['task_name']]}"
        elif task["task_name"] not in by_name:
            status, details = "unknown", "Task missing in model response."
        elif validation_results and task["task_name"] in validation_results:
            validation = validation_results[task["task_name"]]
            status = _normalize_validation_status(validation.get("validation_status"))
            notes = validation.get("notes", "")
            if isinstance(notes, str) and notes.strip():
                details = notes.strip()
            else:
                details = "Validated by second-stage model."
        else:
            status, details = _evaluate_correctness(task, predicted_outputs)

        output_tasks.append(
            {
                "task_name": task["task_name"],
                "logic_explanation": explanation,
                "correctness_result": {"status": status, "details": details},
                "strong_model_validation": (
                    validation_results.get(task["task_name"], {}) if validation_results else {}
                ),
                "source_file": task["source_file"],
            }
        )

    correct_count = sum(
        item["correctness_result"]["status"] == "correct" for item in output_tasks
    )
    incorrect_count = sum(
        item["correctness_result"]["status"] == "incorrect" for item in output_tasks
    )
    unknown_count = len(output_tasks) - correct_count - incorrect_count
    evaluated_count = correct_count + incorrect_count
    accuracy_percentage = (
        round(correct_count / evaluated_count * 100, 2) if evaluated_count else 0.0
    )

    return {
        "prompt_index": prompt_index,
        "tasks_folder": "",
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_files": skipped,
        "summary": {
            "total_tasks": len(tasks),
            "correct_tasks": correct_count,
            "incorrect_tasks": incorrect_count,
            "unknown_tasks": unknown_count,
            "accuracy_percentage": accuracy_percentage,
            "total_duration_seconds": total_duration_seconds,
            "token_usage": token_usage or {
                "prompt_tokens": 0,
                "candidates_tokens": 0,
                "total_tokens": 0,
            },
        },
        "tasks": output_tasks,
    }


def _solve_json_tasks(
    handler: GemmaHandler,
    tasks: list[dict[str, Any]],
    prompt_index: int,
    validator_model: str,
    run_strong_validation: bool,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    combined_result: dict[str, Any] = {"tasks": []}
    task_errors: dict[str, str] = {}
    validation_results: dict[str, dict[str, Any]] = {}

    for offset, task in enumerate(tasks):
        task_name = task["task_name"]
        try:
            result = handler.solve(build_prompt([task], prompt_index + offset))
            returned_tasks = result.get("tasks", []) if isinstance(result, dict) else []
            if not isinstance(returned_tasks, list) or not returned_tasks:
                raise ValueError("Model returned no task result.")
            item = returned_tasks[0]
            if not isinstance(item, dict):
                raise ValueError("Model returned an invalid task result.")
            item = dict(item)
            item["task_name"] = task_name
            combined_result["tasks"].append(item)

            if run_strong_validation:
                predicted_outputs = item.get("predicted_test_outputs", [])
                try:
                    validation_results[task_name] = _validate_json_prediction_with_strong_model(
                        handler=handler,
                        task=task,
                        predicted_outputs=predicted_outputs,
                        validator_model=validator_model,
                    )
                except Exception as exc:
                    validation_results[task_name] = {
                        "validation_status": "unknown",
                        "notes": f"Strong-model JSON validation failed: {exc}",
                    }
        except Exception as exc:
            task_errors[task_name] = str(exc)

    return combined_result, task_errors, validation_results


def _solve_image_tasks(
    handler: GemmaHandler,
    tasks: list[dict[str, Any]],
    images_dir: Path,
    prompt_index: int,
    timeout_seconds: int,
    force_response_timeout_seconds: int,
    run_strong_validation: bool = False,
    validator_model: str | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    combined_result: dict[str, Any] = {"tasks": []}
    task_errors: dict[str, str] = {}
    validation_results: dict[str, dict[str, Any]] = {}

    for offset, task in enumerate(tasks):
        task_name = task["task_name"]
        try:
            image_inputs = _build_image_inputs([task], images_dir)
            result, state = _solve_with_images_deadline(
                handler=handler,
                prompt=build_image_prompt([task], prompt_index + offset),
                image_inputs=image_inputs,
                timeout_seconds=timeout_seconds,
                force_response_timeout_seconds=force_response_timeout_seconds,
            )
            if result is None:
                task_errors[task_name] = state or "response_not_informed"
                continue
            returned_tasks = result.get("tasks", []) if isinstance(result, dict) else []
            if not isinstance(returned_tasks, list) or not returned_tasks:
                raise ValueError("Model returned no task result.")
            item = returned_tasks[0]
            if not isinstance(item, dict):
                raise ValueError("Model returned an invalid task result.")
            item = dict(item)
            item["task_name"] = task_name
            combined_result["tasks"].append(item)

            if run_strong_validation and validator_model:
                predicted_outputs = item.get("predicted_test_outputs", [])
                try:
                    validation_results[task_name] = _validate_image_prediction_with_strong_model(
                        handler=handler,
                        task=task,
                        predicted_outputs=predicted_outputs,
                        images_dir=images_dir,
                        validator_model=validator_model,
                    )
                except Exception as exc:
                    validation_results[task_name] = {
                        "validation_status": "unknown",
                        "notes": f"Strong-model validation failed: {exc}",
                    }
        except Exception as exc:
            task_errors[task_name] = str(exc)

    return combined_result, task_errors, validation_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver with a single Gemma prompt.")
    parser.add_argument(
        "tasks_folder_name",
        help="Folder name under data/ that contains one or more ARC task JSON files.",
    )
    parser.add_argument(
        "--task-file",
        help="Process only this JSON file from the selected tasks folder.",
    )
    parser.add_argument(
        "--render-images",
        action="store_true",
        help="Generate task images in parallel under data/<tasks_folder_name>/images.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only generate images and skip the LLM solve step.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=None,
        help="Max workers for parallel image rendering. Default: Python executor default.",
    )
    parser.add_argument(
        "--images-dir-name",
        default="images",
        help="Directory name (inside selected task folder) where rendered images are saved.",
    )
    parser.add_argument(
        "--no-strong-validate",
        action="store_false",
        dest="strong_validate",
        help="Disable second-stage validation/correction call with validator model.",
    )
    parser.add_argument(
        "--validator-model",
        default=os.getenv("GEMMA_VALIDATOR_MODEL", "gemini-3.5-flash-lite"),
        help="Model name for second-stage image validation/correction.",
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=int,
        default=int(os.getenv("TASK_TIMEOUT_SECONDS", "90")),
        help="Soft timeout in seconds per task call to the main model.",
    )
    parser.add_argument(
        "--force-response-timeout-seconds",
        type=int,
        default=int(os.getenv("FORCE_RESPONSE_TIMEOUT_SECONDS", "30")),
        help="Extra timeout for forced-response retry after main timeout.",
    )
    parser.set_defaults(strong_validate=True)
    args = parser.parse_args()

    load_dotenv()

    data_dir = Path("data")
    tasks_dir = data_dir / args.tasks_folder_name

    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise SystemExit(f"Tasks folder not found: {tasks_dir}")

    tasks, skipped = _load_tasks(tasks_dir)
    if args.task_file:
        tasks = [task for task in tasks if task["source_file"] == args.task_file]
        if not tasks:
            raise SystemExit(f"Task file not found in selected folder: {args.task_file}")
    if not tasks:
        raise SystemExit(
            "No ARC task files found. Ensure at least one JSON file with train/test arrays exists."
        )

    if args.render_only:
        prompt_index = _next_prompt_index(tasks_dir)
        output_base = _output_dir_for(args.tasks_folder_name, prompt_index)
        images_dir = output_base / args.images_dir_name
        render_results = render_tasks_folder_parallel(
            tasks=tasks,
            output_root=images_dir,
            max_workers=args.render_workers,
        )
        total_rendered = sum(item.get("rendered_images", 0) for item in render_results)
        total_skipped = sum(item.get("skipped_items", 0) for item in render_results)
        print(f"Rendered images: {total_rendered}")
        print(f"Skipped render items: {total_skipped}")
        print(f"Images folder: {images_dir}")

        return

    prompt_index = _next_prompt_index(tasks_dir)
    model_name = os.getenv("GEMMA_MODEL", "gemini-3.5-flash-lite")
    output_base = _output_dir_for(args.tasks_folder_name, prompt_index)
    json_result: dict[str, Any] = {"tasks": []}
    image_result: dict[str, Any] = {"tasks": []}
    json_errors: dict[str, str] = {}
    image_errors: dict[str, str] = {}
    json_validation_results: dict[str, dict[str, Any]] = {}
    image_validation_results: dict[str, dict[str, Any]] = {}
    json_start_time = time.perf_counter()
    json_token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}

    try:
        handler = GemmaHandler()
        model_name = handler.model
        json_result, json_errors, json_validation_results = _solve_json_tasks(
            handler=handler,
            tasks=tasks,
            prompt_index=prompt_index,
            validator_model=args.validator_model,
            run_strong_validation=args.strong_validate,
        )
    except Exception as exc:
        for task in tasks:
            json_errors[task["task_name"]] = str(exc)
    json_token_usage = handler.token_usage if "handler" in locals() else json_token_usage

    json_report = _build_report(
        tasks,
        skipped,
        model_name,
        prompt_index,
        json_result,
        json_errors,
        validation_results=json_validation_results,
        token_usage=json_token_usage,
        total_duration_seconds=round(time.perf_counter() - json_start_time, 2),
    )
    json_report["tasks_folder"] = args.tasks_folder_name
    json_path = output_base / f"{output_base.name}-json.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(f"JSON report saved to: {json_path}")

    if args.render_images:
        images_dir = output_base / "images"
        render_results = render_tasks_folder_parallel(
            tasks=tasks,
            output_root=images_dir,
            max_workers=args.render_workers,
        )
        total_rendered = sum(item.get("rendered_images", 0) for item in render_results)
        total_skipped = sum(item.get("skipped_items", 0) for item in render_results)
        print(f"Rendered images: {total_rendered}")
        print(f"Skipped render items: {total_skipped}")
        print(f"Images folder: {images_dir}")

        try:
            image_start_time = time.perf_counter()
            image_token_usage = {"prompt_tokens": 0, "candidates_tokens": 0, "total_tokens": 0}
            handler = GemmaHandler()
            model_name = handler.model
            image_result, image_errors, image_validation_results = _solve_image_tasks(
                handler,
                tasks,
                images_dir,
                prompt_index,
                timeout_seconds=args.task_timeout_seconds,
                force_response_timeout_seconds=args.force_response_timeout_seconds,
                run_strong_validation=args.strong_validate,
                validator_model=args.validator_model,
            )
        except Exception as exc:
            for task in tasks:
                image_errors[task["task_name"]] = str(exc)
        image_token_usage = handler.token_usage if "handler" in locals() else image_token_usage

        image_report = _build_report(
            tasks,
            skipped,
            model_name,
            prompt_index,
            image_result,
            image_errors,
            validation_results=image_validation_results,
            token_usage=image_token_usage,
            total_duration_seconds=round(time.perf_counter() - image_start_time, 2),
        )
        image_report["tasks_folder"] = args.tasks_folder_name
        image_path = output_base / f"{output_base.name}-image.json"
        image_path.write_text(json.dumps(image_report, indent=2), encoding="utf-8")
        print(f"Image report saved to: {image_path}")

    print(f"Processed tasks: {len(tasks)}")
    print(f"Skipped files: {len(skipped)}")


if __name__ == "__main__":
    main()
