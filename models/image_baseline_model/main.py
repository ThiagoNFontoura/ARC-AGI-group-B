import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from models.image_baseline_model.llm_handler import GemmaHandler
from models.image_baseline_model.prompt import build_image_prompt, build_prompt
from models.image_baseline_model.task_image_renderer import render_tasks_folder_parallel


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


def _safe_task_name(task_name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]", "_", task_name).strip("_")
    return clean or "task"


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


def _build_report(
    tasks: list[dict[str, Any]],
    skipped: list[str],
    model_name: str,
    prompt_index: int,
    model_result: dict[str, Any],
    task_errors: dict[str, str],
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
            status, details = "unknown", f"LLM call failed: {task_errors[task['task_name']]}"
        elif task["task_name"] not in by_name:
            status, details = "unknown", "Task missing in model response."
        else:
            status, details = _evaluate_correctness(task, predicted_outputs)

        output_tasks.append(
            {
                "task_name": task["task_name"],
                "logic_explanation": explanation,
                "correctness_result": {"status": status, "details": details},
                "source_file": task["source_file"],
            }
        )

    return {
        "prompt_index": prompt_index,
        "tasks_folder": "",
        "model": model_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "skipped_files": skipped,
        "tasks": output_tasks,
    }


def _solve_json_tasks(
    handler: GemmaHandler,
    tasks: list[dict[str, Any]],
    prompt_index: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    combined_result: dict[str, Any] = {"tasks": []}
    task_errors: dict[str, str] = {}

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
        except Exception as exc:
            task_errors[task_name] = str(exc)

    return combined_result, task_errors


def _solve_image_tasks(
    handler: GemmaHandler,
    tasks: list[dict[str, Any]],
    images_dir: Path,
    prompt_index: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    combined_result: dict[str, Any] = {"tasks": []}
    task_errors: dict[str, str] = {}

    for offset, task in enumerate(tasks):
        task_name = task["task_name"]
        try:
            image_inputs = _build_image_inputs([task], images_dir)
            result = handler.solve_with_images(
                build_image_prompt([task], prompt_index + offset), image_inputs
            )
            returned_tasks = result.get("tasks", []) if isinstance(result, dict) else []
            if not isinstance(returned_tasks, list) or not returned_tasks:
                raise ValueError("Model returned no task result.")
            item = returned_tasks[0]
            if not isinstance(item, dict):
                raise ValueError("Model returned an invalid task result.")
            item = dict(item)
            item["task_name"] = task_name
            combined_result["tasks"].append(item)
        except Exception as exc:
            task_errors[task_name] = str(exc)

    return combined_result, task_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC-AGI solver with a single Gemma prompt.")
    parser.add_argument(
        "tasks_folder_name",
        help="Folder name under data/ that contains one or more ARC task JSON files.",
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

    if args.render_only:
        images_dir = tasks_dir / args.images_dir_name
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
    model_name = os.getenv("GEMMA_MODEL", "gemma-4-31b-it")
    json_result: dict[str, Any] = {"tasks": []}
    image_result: dict[str, Any] = {"tasks": []}
    json_errors: dict[str, str] = {}
    image_errors: dict[str, str] = {}

    try:
        handler = GemmaHandler()
        model_name = handler.model
        json_result, json_errors = _solve_json_tasks(handler, tasks, prompt_index)
    except Exception as exc:
        for task in tasks:
            json_errors[task["task_name"]] = str(exc)

    json_report = _build_report(
        tasks, skipped, model_name, prompt_index, json_result, json_errors
    )
    json_report["tasks_folder"] = args.tasks_folder_name
    json_path = tasks_dir / f"{prompt_index}-json.json"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    print(f"JSON report saved to: {json_path}")

    if args.render_images:
        images_dir = tasks_dir / args.images_dir_name
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
            handler = GemmaHandler()
            model_name = handler.model
            image_result, image_errors = _solve_image_tasks(
                handler, tasks, images_dir, prompt_index
            )
        except Exception as exc:
            for task in tasks:
                image_errors[task["task_name"]] = str(exc)

        image_report = _build_report(
            tasks, skipped, model_name, prompt_index, image_result, image_errors
        )
        image_report["tasks_folder"] = args.tasks_folder_name
        image_path = tasks_dir / f"{prompt_index}-image.json"
        image_path.write_text(json.dumps(image_report, indent=2), encoding="utf-8")
        print(f"Image report saved to: {image_path}")

    print(f"Processed tasks: {len(tasks)}")
    print(f"Skipped files: {len(skipped)}")


if __name__ == "__main__":
    main()
