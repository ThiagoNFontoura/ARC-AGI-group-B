import argparse
import ast
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
from models.example_gen.invariants import analyze_task_invariants, validate_grid_invariants
from models.example_gen.prompt import build_prompt

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "example_gen_config.json"


def _is_arc_task(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("train"), list) and isinstance(
        payload.get("test"), list
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"generated_examples": 2, "output_dir": "output/example-gen"}
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


def _normalise_generated_examples(
    value: Any, expected: int, invariant_analysis: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[bool]]:
    if not isinstance(value, list):
        raise ValueError("Model returned generated_train that is not a list.")
    examples = []
    structural_flags = []
    for example in value:
        if not isinstance(example, dict):
            examples.append({"input": None, "output": None})
            structural_flags.append(False)
            continue
        input_grid = example.get("input")
        output_grid = example.get("output")
        examples.append({"input": input_grid, "output": output_grid})
        structural_flags.append(
            validate_grid_invariants(input_grid, invariant_analysis["inputs"])
            and validate_grid_invariants(output_grid, invariant_analysis["outputs"])
        )
    if len(examples) != expected:
        raise ValueError(f"Model returned {len(examples)} examples; expected {expected}.")
    return examples, structural_flags


def _compile_transformation(source: Any) -> Any:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Model did not return transformation_function.")
    tree = ast.parse(source, mode="exec")
    allowed_nodes = {
        ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
        ast.AnnAssign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple,
        ast.Dict, ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.If, ast.For, ast.comprehension, ast.ListComp, ast.DictComp, ast.SetComp,
        ast.Call, ast.keyword, ast.Attribute, ast.AugAssign, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.Not, ast.And, ast.Or, ast.Eq,
        ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.IfExp,
    }
    unsupported_nodes = sorted(
        {type(node).__name__ for node in ast.walk(tree) if type(node) not in allowed_nodes}
    )
    if unsupported_nodes:
        raise ValueError(
            "transformation_function contains unsupported Python syntax: "
            + ", ".join(unsupported_nodes)
        )
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "transform"]
    if len(functions) != 1 or len(functions[0].args.args) != 1:
        raise ValueError("transformation_function must define transform(grid).")
    forbidden_names = {"eval", "exec", "open", "__import__", "compile", "globals", "locals"}
    if any(isinstance(node, ast.Name) and node.id in forbidden_names for node in ast.walk(tree)):
        raise ValueError("transformation_function uses a forbidden name.")
    allowed_builtins = {
        name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
        for name in ("abs", "all", "any", "enumerate", "len", "list", "max", "min", "range", "reversed", "set", "sorted", "sum", "zip")
    }
    namespace = {"__builtins__": allowed_builtins}
    exec(compile(tree, "<transformation_function>", "exec"), namespace)
    return namespace["transform"]


def _validate_with_transformation(
    transform: Any, examples: list[dict[str, Any]]
) -> tuple[list[bool], list[str]]:
    flags = []
    reasons = []
    for index, example in enumerate(examples):
        try:
            predicted = transform(example.get("input"))
            passed = predicted == example.get("output")
            flags.append(passed)
            if not passed:
                reasons.append(f"Transformation output mismatch at example {index}.")
        except Exception as exc:
            flags.append(False)
            reasons.append(f"Transformation failed at example {index}: {exc}")
    return flags, reasons


def _build_plus_task(
    task: dict[str, Any],
    result: dict[str, Any],
    generated_examples: int,
    source_file: str,
) -> dict[str, Any]:
    explanation = result.get("logic_explanation", "")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Model did not return logic_explanation.")
    source_train = task.get("train", []) if isinstance(task.get("train", []), list) else []
    transform = _compile_transformation(result.get("transformation_function"))
    invariant_analysis = analyze_task_invariants(source_train)
    generated, structural_flags = _normalise_generated_examples(
        result.get("generated_train"), generated_examples, invariant_analysis
    )
    original_flags, validation_reasons = _validate_with_transformation(transform, source_train)
    original_validation_passed = all(original_flags)
    generated_function_flags: list[bool] = []
    if original_validation_passed:
        generated_function_flags, generated_reasons = _validate_with_transformation(transform, generated)
        validation_reasons.extend(
            f"Generated example: {reason}" for reason in generated_reasons
        )
    generated_validation_flags = [
        function_flag and structural_flag
        for function_flag, structural_flag in zip(generated_function_flags, structural_flags)
    ]
    has_invariant_violation = any(not flag for flag in structural_flags)
    generated_flags = (
        generated_validation_flags
        if original_validation_passed and not has_invariant_violation
        else []
    )
    accepted_generated = [
        example for example, flag in zip(generated, generated_flags) if flag
    ]
    annotated_originals = []
    for index, example in enumerate(source_train):
        item = dict(example) if isinstance(example, dict) else {"input": None, "output": None}
        item["validation_passed"] = original_flags[index] if index < len(original_flags) else False
        item["valid"] = original_flags[index] if index < len(original_flags) else False
        annotated_originals.append(item)
    annotated_generated = []
    for index, example in enumerate(accepted_generated):
        item = dict(example)
        item["validation_passed"] = True
        item["valid"] = True
        annotated_generated.append(item)
    return {
        "task_name": task["task_name"],
        "logic_explanation": explanation.strip(),
        "transformation_function": result["transformation_function"],
        "train": annotated_originals + annotated_generated,
        "test": _without_outputs(task.get("test", [])),
        "invariants": invariant_analysis,
        "validation_summary": {
            "original_train_total": len(source_train),
            "original_train_valid": sum(original_flags),
            "original_validation_passed": original_validation_passed,
            "generated_train_total": len(generated),
            "generated_train_valid": len(accepted_generated),
            "generated_train_invalid": len(generated) - len(accepted_generated),
            "generated_train_discarded": len(generated) - len(accepted_generated),
            "all_generated_invalid_due_to_invariant": has_invariant_violation,
            "all_generated_invalid_due_to_threshold": False,
            "validation_reasons": validation_reasons,
        },
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


def _is_retryable_response_error(error: Exception) -> bool:
    message = str(error)
    return (
        isinstance(error, (TypeError, ValueError, json.JSONDecodeError))
        and (
            "transformation_function" in message
            or "generated_train" in message
            or "logic_explanation" in message
            or "invalid object" in message
            or "JSON" in message
        )
    )


def _function_correction_prompt(
    prompt: str, previous_function: Any, error: Exception
) -> str:
    return (
        prompt
        + "\n\nIMPORTANT CORRECTION REQUEST: the previous transformation_function "
        + "was rejected by the local validator. The previous function source was:\n"
        + str(previous_function)
        + "\nThe validator error was:\n"
        + str(error)
        + "\nReturn a new complete STRICT JSON response, repeating every required field. "
        + "Do not reuse the invalid function unchanged. The transformation_function must "
        + "contain only one top-level def transform(grid): function, with local assignments, "
        + "if/for, comprehensions, indexing, basic built-ins, and return. It must have no "
        + "imports, aliases, docstrings, standalone expressions, lambda, while, break, "
        + "continue, try, raise, classes, nested functions, decorators, or generator "
        + "expressions."
    )


def _generate_plus_task_with_retry(
    handler: GemmaHandler,
    prompt: str,
    task: dict[str, Any],
    generated_examples: int,
    source_file: str,
    retry_attempts: int,
    retry_delay_seconds: float,
    function_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_prompt = prompt
    previous_function: Any = "<none>"
    previous_error: Exception | None = None
    for function_attempt in range(function_attempts):
        if previous_error is not None:
            current_prompt = _function_correction_prompt(
                prompt, previous_function, previous_error
            )
        for retry_attempt in range(retry_attempts + 1):
            result: dict[str, Any] | None = None
            try:
                result = handler.solve(current_prompt)
                if not isinstance(result, dict):
                    raise ValueError("Model returned an invalid object.")
                plus_task = _build_plus_task(
                    task,
                    result,
                    generated_examples,
                    source_file,
                )
                if not plus_task["validation_summary"]["original_validation_passed"]:
                    raise ValueError(
                        "transformation_function failed validation on the original training examples. "
                        + "; ".join(
                            plus_task["validation_summary"]["validation_reasons"]
                        )
                    )
                return plus_task, result
            except Exception as exc:
                retryable_transient = _is_transient_api_error(exc)
                retryable_response = _is_retryable_response_error(exc)
                if retryable_response:
                    previous_function = (
                        result.get("transformation_function", "<missing>")
                        if isinstance(result, dict)
                        else "<missing>"
                    )
                    previous_error = exc
                if retryable_transient and retry_attempt < retry_attempts:
                    delay = retry_delay_seconds * (2**retry_attempt)
                    print(
                        f"Transient API error ({exc}); retrying in {delay:g}s "
                        f"({retry_attempt + 1}/{retry_attempts})."
                    )
                    time.sleep(delay)
                    continue
                if retryable_response:
                    break
                raise
        if function_attempt + 1 < function_attempts:
            print(
                f"Function attempt {function_attempt + 1}/{function_attempts} failed; "
                "requesting a new function from the API."
            )
            continue
        if previous_error is not None:
            raise previous_error
    raise RuntimeError("Unreachable retry state.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ARC examples with one API call per task.")
    parser.add_argument(
        "task",
        type=Path,
        help="Task JSON path, or a folder under data/ containing task JSON files.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--generated-examples", type=int, help="Override generated_examples from config.")
    parser.add_argument("--retry-attempts", type=int, help="Override transient retry attempts for this run.")
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

    retry_attempts = args.retry_attempts if args.retry_attempts is not None else config.get("transient_retry_attempts", 0)
    function_attempts = config.get("function_attempts", 3)
    retry_delay_seconds = config.get("transient_retry_delay_seconds", 5)
    task_interval_seconds = config.get("interval_between_tasks_seconds", 0)
    if not isinstance(retry_attempts, int) or retry_attempts < 0:
        raise SystemExit("transient_retry_attempts must be a non-negative integer")
    if not isinstance(function_attempts, int) or function_attempts < 1:
        raise SystemExit("function_attempts must be a positive integer")
    if not isinstance(retry_delay_seconds, (int, float)) or retry_delay_seconds < 0:
        raise SystemExit("transient_retry_delay_seconds must be a non-negative number")
    if not isinstance(task_interval_seconds, (int, float)) or task_interval_seconds < 0:
        raise SystemExit("interval_between_tasks_seconds must be a non-negative number")

    output_dir_value = config.get("output_dir", "output/example-gen")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_id = task_path.stem if task_path.is_file() else task_path.name
    report_tasks: list[dict[str, Any]] = []
    handler = GemmaHandler(
        model=os.getenv("EXAMPLE_GEN_MODEL") or config.get("model"),
        thinking_level=config.get("thinking_level"),
    )
    for index, path in enumerate(paths):
        if index > 0 and task_interval_seconds:
            print(f"Waiting {task_interval_seconds:g}s before the next task...")
            time.sleep(task_interval_seconds)
        task = _load_task(path)
        try:
            plus_task, _ = _generate_plus_task_with_retry(
                handler,
                build_prompt(task, count),
                task,
                count,
                path.name,
                retry_attempts,
                retry_delay_seconds,
                function_attempts,
            )
            output_path = output_dir / f"{path.stem}-plus.json"
            output_path.write_text(json.dumps(plus_task, indent=2), encoding="utf-8")
            report_tasks.append({
                "source_file": path.name,
                "status": "success",
                "output_file": output_path.name,
                "validation_summary": plus_task["validation_summary"],
            })
            print(f"Generated: {output_path}")
        except Exception as exc:
            print(f"Failed: {path.name}: {exc}")
            report_tasks.append({"source_file": path.name, "status": "invalid_response", "error": str(exc)})

    report = {
        "report_id": report_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": handler.model,
        "thinking_level": handler.thinking_level,
        "generated_examples_requested": count,
        "retry_attempts": retry_attempts,
        "function_attempts": function_attempts,
        "retry_delay_seconds": retry_delay_seconds,
        "interval_between_tasks_seconds": task_interval_seconds,
        "summary": {
            "total_tasks": len(paths),
            "successful_tasks": sum(item["status"] == "success" for item in report_tasks),
            "failed_tasks": sum(item["status"] != "success" for item in report_tasks),
        },
        "tasks": report_tasks,
    }
    report_path = output_dir / f"relatorio[{report_id}].json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()