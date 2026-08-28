import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from models.baseline_model.llm_handler import GemmaHandler
from models.baseline_model.prompt import build_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_arc_task(payload: Any) -> bool:
    """Return whether a decoded payload has ARC train and test arrays."""
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


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


def _evaluate_correctness(task: dict[str, Any], predicted_outputs: Any) -> tuple[str, str]:
    """Compare predictions with available test outputs and explain the status."""
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


def main() -> None:
    """Run the solver and save one normalized result file for the task folder."""
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver with a single Gemma prompt.")
    parser.add_argument(
        "tasks_folder_name",
        help="Folder name under data/ that contains one or more ARC task JSON files.",
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

    model_name = "gemma-3-27b-it"
    by_name: dict[str, dict[str, Any]] = {}
    task_errors_by_name: dict[str, str] = {}
    start_time = time.perf_counter()
    token_usage = {
        "prompt_tokens": 0,
        "candidates_tokens": 0,
        "total_tokens": 0,
    }
    try:
        handler = GemmaHandler()
        model_name = handler.model

        for prompt_index, task in enumerate(tasks, start=1):
            prompt = build_prompt([task], prompt_index=prompt_index)
            try:
                model_result = handler.solve(prompt)
                returned_tasks = model_result.get("tasks", []) if isinstance(model_result, dict) else []
                found = False
                if isinstance(returned_tasks, list):
                    for item in returned_tasks:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("task_name")
                        if isinstance(name, str) and name.strip() == task["task_name"]:
                            by_name[name] = item
                            found = True
                            break
                if not found and isinstance(returned_tasks, list) and returned_tasks:
                    first_item = returned_tasks[0]
                    if isinstance(first_item, dict):
                        by_name[task["task_name"]] = first_item
            except Exception as exc:
                task_errors_by_name[task["task_name"]] = f"LLM task call failed: {exc}"
        token_usage = handler.token_usage
    except Exception as exc:
        error_text = f"LLM setup failed: {exc}"
        for task in tasks:
            task_errors_by_name[task["task_name"]] = error_text

    output_tasks: list[dict[str, Any]] = []
    for task in tasks:
        model_task = by_name.get(task["task_name"], {})
        logic_explanation = model_task.get("logic_explanation", "")
        if not isinstance(logic_explanation, str) or not logic_explanation.strip():
            logic_explanation = "No explanation returned by model."

        predicted_outputs = model_task.get("predicted_test_outputs", [])

        if task["task_name"] in task_errors_by_name:
            status, details = "unknown", task_errors_by_name[task["task_name"]]
        elif task["task_name"] not in by_name:
            status, details = "unknown", "Task missing in model response."
        else:
            status, details = _evaluate_correctness(task, predicted_outputs)

        output_tasks.append(
            {
                "task_name": task["task_name"],
                "logic_explanation": logic_explanation,
                "correctness_result": {
                    "status": status,
                    "details": details,
                },
                "predicted_test_outputs": predicted_outputs,
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
    total_duration_seconds = round(time.perf_counter() - start_time, 2)

    output = {
        "prompt_index": 1,
        "prompt_chunks": len(tasks),
        "tasks_folder": args.tasks_folder_name,
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
            "token_usage": token_usage,
        },
        "tasks": output_tasks,
    }

    output_path = output_dir / f"{args.tasks_folder_name}_baseline_output.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Processed tasks: {len(tasks)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Accuracy: {accuracy_percentage}% ({correct_count}/{evaluated_count} correct)")
    print(f"Duration: {total_duration_seconds}s")
    print(f"Total tokens: {token_usage['total_tokens']}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
