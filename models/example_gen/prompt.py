import json
from typing import Any

from models.example_gen.invariants import analyze_task_invariants, format_invariants_for_prompt


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

    invariant_analysis = analyze_task_invariants(source_train)

    task_payload = {
        "task_name": task["task_name"],
        "train": source_train,
        "test": _without_test_outputs(task),
    }
    task_json = json.dumps(task_payload, separators=(",", ":"))

    return (
        "You are an ARC-AGI example generator. Infer the exact transformation rule from "
        "all labeled training examples, then create new valid examples that "
        "follow the same rule. Return STRICT JSON only, with no markdown.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "logic_explanation": "brief rule explanation",\n'
        '  "generated_train": [{"input": [[...]], "output": [[...]]}],\n'
        '  "predicted_test_outputs": [[[...]]],\n'
        '  "validation": {\n'
        '    "original_train": [{"index": 0, "passed": true, "reason": "..."}],\n'
        '    "generated_train": [{"index": 0, "passed": true, "reason": "..."}]\n'
        "  }\n"
        "}\n\n"
        f"Generate exactly {generated_examples} additional training examples.\n"
        "Before generating, analyze input and output properties separately. "
        "Preserve every property marked CONSTANT for inputs in each generated input, "
        "and every property marked CONSTANT for outputs in each generated output. "
        "Do not assume a property is constant across input and output together.\n"
        f"Observed invariant analysis:\n{format_invariants_for_prompt(invariant_analysis)}\n"
        "Rules:\n"
        "1) Every generated example must contain only integer rectangular grids.\n"
        "2) Generated outputs must be correct applications of the inferred rule.\n"
        "3) Keep logic_explanation to 1-3 concise sentences.\n"
        "4) Return one predicted output for each test input, in order.\n"
        "5) Carefully compare grid heights and widths across all source examples. "
        "If the grid dimensions are constant, every generated input and output must "
        "keep those same dimensions. Do not resize a grid unless the examples clearly "
        "demonstrate that resizing is part of the rule.\n"
        "6) Do not copy a source example unless it is unavoidable.\n"
        "7) Explain which non-constant properties vary and how that variation follows "
        "from the inferred rule.\n"
        "8) Construct the inferred input-to-output rule and test it on every original "
        "and generated training pair. Return one validation record for every pair. "
        "Set passed to false when the rule fails, and briefly explain why.\n\n"
        f"Task JSON: {task_json}"
    )