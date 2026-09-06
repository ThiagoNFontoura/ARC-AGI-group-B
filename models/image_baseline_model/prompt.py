import json
from typing import Any


def _strip_test_outputs(task: dict[str, Any]) -> dict[str, Any]:
    sanitized_test: list[Any] = []
    for case in task.get("test", []):
        if isinstance(case, dict):
            case_copy = {k: v for k, v in case.items() if k != "output"}
            sanitized_test.append(case_copy)
        else:
            sanitized_test.append(case)

    return {
        "task_name": task["task_name"],
        "train": task.get("train", []),
        "test": sanitized_test,
    }


def build_prompt(tasks: list[dict[str, Any]], prompt_index: int) -> str:
    """Build one shared prompt that asks for predictions for all tasks at once."""
    compact_tasks = [_strip_test_outputs(task) for task in tasks]

    tasks_json = json.dumps(compact_tasks, separators=(",", ":"))

    return (
        "You are solving ARC-AGI tasks. "
        "Infer each task rule from train examples and predict all test outputs. "
        "Return STRICT JSON only (no markdown fences, no extra text).\\n\\n"
        "Required JSON schema:\\n"
        "{\\n"
        "  \"tasks\": [\\n"
        "    {\\n"
        "      \"task_name\": \"string\",\\n"
        "      \"logic_explanation\": \"brief rule explanation\",\\n"
        "      \"predicted_test_outputs\": [grid, grid, ...]\\n"
        "    }\\n"
        "  ]\\n"
        "}\\n\\n"
        "Rules:\\n"
        "1) Keep logic_explanation concise (1-3 short sentences).\\n"
        "2) predicted_test_outputs must have one grid per test input in order.\\n"
        "3) Grid values are integers.\\n"
        "4) Include every task exactly once.\\n\\n"
        f"Prompt index: {prompt_index}.\\n"
        f"Tasks JSON: {tasks_json}"
    )


def build_image_prompt(tasks: list[dict[str, Any]], prompt_index: int) -> str:
    task_names = [task["task_name"] for task in tasks]
    return (
        "You are solving ARC-AGI tasks from the attached images. "
        "For each task, infer the rule from its labeled training images and predict "
        "the outputs for its test input images. Return STRICT JSON only.\n\n"
        "The image attachments are grouped by task. Each attachment label identifies "
        "the task, split, case index, and whether it is an input or training output. "
        "Test output images are never attached.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"tasks\": [\n"
        "    {\n"
        "      \"task_name\": \"string\",\n"
        "      \"logic_explanation\": \"brief rule explanation\",\n"
        "      \"predicted_test_outputs\": [grid, grid, ...]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "1) Include every task exactly once.\n"
        "2) Keep logic_explanation concise (1-3 short sentences).\n"
        "3) Return one integer grid per test input, in order.\n"
        "4) Do not include markdown or extra text.\n\n"
        f"Prompt index: {prompt_index}.\n"
        f"Tasks in attachment order: {json.dumps(task_names)}"
    )


def build_image_validation_prompt(
    task_name: str,
    predicted_test_outputs: Any,
    ground_truth_test_outputs: Any,
) -> str:
    return (
        "You are validating ARC-AGI predicted test outputs against ground truth. "
        "Use attached images and provided grids to compare and correct results. "
        "Return STRICT JSON only.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"task_name\": \"string\",\n"
        "  \"validation_status\": \"correct|incorrect|unknown\",\n"
        "  \"notes\": \"brief explanation\",\n"
        "  \"corrected_test_outputs\": [grid, grid, ...]\n"
        "}\n\n"
        "Rules:\n"
        "1) If any mismatch exists, set validation_status to incorrect.\n"
        "2) corrected_test_outputs must be the final correct outputs.\n"
        "3) Keep notes concise (1-3 short sentences).\n"
        "4) Do not include markdown or extra text.\n\n"
        f"Task: {task_name}.\n"
        f"Predicted test outputs (raw): {json.dumps(predicted_test_outputs, separators=(",", ":"))}\n"
        f"Ground truth test outputs (raw): {json.dumps(ground_truth_test_outputs, separators=(",", ":"))}"
    )


def build_json_validation_prompt(
    task_name: str,
    predicted_test_outputs: Any,
    ground_truth_test_outputs: Any,
) -> str:
    return (
        "You are validating ARC-AGI predicted test outputs against ground truth. "
        "Return STRICT JSON only.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"task_name\": \"string\",\n"
        "  \"validation_status\": \"correct|incorrect|unknown\",\n"
        "  \"notes\": \"brief explanation\",\n"
        "  \"corrected_test_outputs\": [grid, grid, ...]\n"
        "}\n\n"
        "Rules:\n"
        "1) If prediction equals ground truth for all test cases, use correct.\n"
        "2) If any mismatch exists, use incorrect.\n"
        "3) If inputs are insufficient, use unknown.\n"
        "4) corrected_test_outputs must contain final correct outputs when available.\n"
        "5) Keep notes concise (1-3 short sentences).\n"
        "6) Do not include markdown or extra text.\n\n"
        f"Task: {task_name}.\n"
        f"Predicted test outputs (raw): {json.dumps(predicted_test_outputs, separators=(",", ":"))}\n"
        f"Ground truth test outputs (raw): {json.dumps(ground_truth_test_outputs, separators=(",", ":"))}"
    )
