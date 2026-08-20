import json
from typing import Any


def build_prompt(tasks: list[dict[str, Any]], prompt_index: int) -> str:
    """Build one shared prompt that asks for predictions for all tasks at once."""
    compact_tasks = []
    for task in tasks:
        compact_tasks.append(
            {
                "task_name": task["task_name"],
                "train": task.get("train", []),
                "test": task.get("test", []),
            }
        )

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
