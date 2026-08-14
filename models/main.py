import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_handler import GemmaHandler
from prompt import build_prompt


def _is_arc_task(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


def _load_tasks(tasks_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
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


def _next_prompt_index(tasks_dir: Path) -> int:
    max_index = 0
    for file_path in tasks_dir.glob("*.json"):
        stem = file_path.stem
        if stem.isdigit():
            max_index = max(max_index, int(stem))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver with a single Gemma prompt.")
    parser.add_argument(
        "tasks_folder_name",
        help="Folder name under data/ that contains one or more ARC task JSON files.",
    )
    args = parser.parse_args()

    load_dotenv()

    data_dir = Path("data")
    tasks_dir = data_dir / args.tasks_folder_name

    if not tasks_dir.exists() or not tasks_dir.is_dir():
        raise SystemExit(f"Tasks folder not found: {tasks_dir}")

    tasks, skipped = _load_tasks(tasks_dir)
    if not tasks:
        raise SystemExit(
            "No ARC task files found. Ensure at least one JSON file with train/test arrays exists."
        )

    prompt_index = _next_prompt_index(tasks_dir)
    prompt = build_prompt(tasks, prompt_index)

    model_name = "gemma-3-27b-it"
    llm_error: str | None = None
    model_result: dict[str, Any] = {"tasks": []}

    try:
        handler = GemmaHandler()
        model_name = handler.model
        model_result = handler.solve(prompt)
    except Exception as exc:
        llm_error = str(exc)

    returned_tasks = model_result.get("tasks", []) if isinstance(model_result, dict) else []
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(returned_tasks, list):
        for item in returned_tasks:
            if not isinstance(item, dict):
                continue
            name = item.get("task_name")
            if isinstance(name, str) and name.strip():
                by_name[name] = item

    output_tasks: list[dict[str, Any]] = []
    for task in tasks:
        model_task = by_name.get(task["task_name"], {})
        logic_explanation = model_task.get("logic_explanation", "")
        if not isinstance(logic_explanation, str) or not logic_explanation.strip():
            logic_explanation = "No explanation returned by model."

        predicted_outputs = model_task.get("predicted_test_outputs", [])

        if llm_error:
            status, details = "unknown", f"LLM call failed: {llm_error}"
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

    output = {
        "prompt_index": prompt_index,
        "tasks_folder": args.tasks_folder_name,
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_files": skipped,
        "tasks": output_tasks,
    }

    output_path = tasks_dir / f"{prompt_index}.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Processed tasks: {len(tasks)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
