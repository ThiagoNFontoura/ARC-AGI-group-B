import json
from typing import Any


def _without_test_outputs(task: dict[str, Any]) -> list[Any]:
    test_cases: list[Any] = []
    for case in task.get("test", []):
        if isinstance(case, dict):
            test_cases.append({key: value for key, value in case.items() if key != "output"})
        else:
            test_cases.append(case)
    return test_cases


def build_prompt(task: dict[str, Any], generated_examples: int) -> str:
    source_train = task.get("train", [])
    if not isinstance(source_train, list):
        source_train = []

    task_payload = {
        "task_name": task["task_name"],
        "train": source_train[:3],
        "test": _without_test_outputs(task),
    }
    task_json = json.dumps(task_payload, separators=(",", ":"))

    return (
        "You are an ARC-AGI task generator. Infer the exact transformation rule from "
        "the three labeled training examples, then create new valid examples that "
        "follow the same rule. Return STRICT JSON only, with no markdown.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "logic_explanation": "brief rule explanation",\n'
        '  "generated_train": [{"input": [[...]], "output": [[...]]}],\n'
        '  "predicted_test_outputs": [[[...]]]\n'
        "}\n\n"
        f"Generate exactly {generated_examples} additional training examples.\n"
        "Rules:\n"
        "1) Every generated example must contain only integer rectangular grids.\n"
        "2) Generated outputs must be correct applications of the inferred rule.\n"
        "3) Keep logic_explanation to 1-3 concise sentences.\n"
        "4) Return one predicted output for each test input, in order.\n"
        "5) Carefully compare grid heights and widths across all source examples. "
        "If the grid dimensions are constant, every generated input and output must "
        "keep those same dimensions. Do not resize a grid unless the examples clearly "
        "demonstrate that resizing is part of the rule.\n"
        "6) Do not copy a source example unless it is unavoidable.\n\n"
        f"Task JSON: {task_json}"
    )