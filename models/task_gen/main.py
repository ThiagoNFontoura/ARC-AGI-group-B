import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from models.baseline_model.llm_handler import GemmaHandler
from models.task_gen.prompt import build_prompt

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "task_gen_config.json"


def _is_arc_task(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"generated_examples": 2, "output_dir": "output/task-gen"}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    return payload


def _load_task(task_path: Path) -> dict[str, Any]:
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if not _is_arc_task(payload):
        raise ValueError(f"Not an ARC task (missing train/test arrays): {task_path}")
    task = dict(payload)
    task["task_name"] = task.get("task_name") or task_path.stem
    return task


def _without_outputs(test_cases: Any) -> list[Any]:
    if not isinstance(test_cases, list):
        return []
    return [
        {key: value for key, value in case.items() if key != "output"}
        if isinstance(case, dict)
        else case
        for case in test_cases
    ]


def _normalise_generated_examples(value: Any, expected: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Model returned generated_train that is not a list.")
    examples = []
    for example in value:
        if not isinstance(example, dict) or not isinstance(example.get("input"), list) or not isinstance(
            example.get("output"), list
        ):
            raise ValueError("Each generated example must have input and output grids.")
        examples.append({"input": example["input"], "output": example["output"]})
    if len(examples) != expected:
        raise ValueError(f"Model returned {len(examples)} examples; expected {expected}.")
    return examples


def _build_plus_task(
    task: dict[str, Any], result: dict[str, Any], generated_examples: int, source_file: str
) -> dict[str, Any]:
    explanation = result.get("logic_explanation", "")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Model did not return logic_explanation.")
    return {
        "task_name": task["task_name"],
        "logic_explanation": explanation.strip(),
        "train": task.get("train", [])[:3] + _normalise_generated_examples(
            result.get("generated_train"), generated_examples
        ),
        "test": _without_outputs(task.get("test", [])),
        "source_file": source_file,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _is_transient_api_error(error: Exception) -> bool:
    error_code = getattr(error, "code", None)
    error_text = str(error).upper()
    return error_code in (429, 500, 502, 503, 504) or any(
        marker in error_text
        for marker in ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
    )


def _solve_with_retry(
    handler: GemmaHandler,
    prompt: str,
    retry_attempts: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    for attempt in range(retry_attempts + 1):
        try:
            result = handler.solve(prompt)
            if not isinstance(result, dict):
                raise ValueError("Model returned an invalid object.")
            return result
        except Exception as exc:
            if attempt >= retry_attempts or not _is_transient_api_error(exc):
                raise
            delay = retry_delay_seconds * (2**attempt)
            print(
                f"Transient API error ({exc}); retrying in {delay:g}s "
                f"({attempt + 1}/{retry_attempts})."
            )
            time.sleep(delay)
    raise RuntimeError("Unreachable retry state.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ARC task examples with one API call per task.")
    parser.add_argument(
        "task",
        type=Path,
        help="Task JSON path, or a folder under data/ containing task JSON files.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--generated-examples", type=int, help="Override generated_examples from config.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = _load_config(args.config)
    count = args.generated_examples if args.generated_examples is not None else config.get("generated_examples", 2)
    if not isinstance(count, int) or count < 0:
        raise SystemExit("generated_examples must be a non-negative integer")

    task_path = args.task if args.task.is_absolute() else PROJECT_ROOT / args.task
    paths = sorted(task_path.glob("*.json")) if task_path.is_dir() else [task_path]
    paths = [path for path in paths if not path.name.endswith("-plus.json")]
    if not paths:
        raise SystemExit(f"No task JSON files found: {task_path}")

    retry_attempts = config.get("transient_retry_attempts", 2)
    retry_delay_seconds = config.get("transient_retry_delay_seconds", 5)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        raise SystemExit("transient_retry_attempts must be a non-negative integer")
    if not isinstance(retry_delay_seconds, (int, float)) or retry_delay_seconds < 0:
        raise SystemExit("transient_retry_delay_seconds must be a non-negative number")

    output_dir_value = config.get("output_dir", "output/task-gen")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = GemmaHandler(model=os.getenv("TASK_GEN_MODEL") or config.get("model"))
    for path in paths:
        task = _load_task(path)
        try:
            result = _solve_with_retry(
                handler,
                build_prompt(task, count),
                retry_attempts,
                retry_delay_seconds,
            )
        except Exception as exc:
            print(f"Failed: {path.name}: {exc}")
            continue
        plus_task = _build_plus_task(task, result, count, path.name)
        output_path = output_dir / f"{path.stem}-plus.json"
        output_path.write_text(json.dumps(plus_task, indent=2), encoding="utf-8")
        print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()