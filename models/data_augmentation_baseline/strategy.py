"""Model-independent augmented inference and voting strategy."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Callable

from .transforms import TransformSpec, get_transformations, transform_outputs, transform_task

PromptBuilder = Callable[[list[dict[str, Any]], int], str]
PromptSolver = Callable[[str], dict[str, Any]]


class AugmentedInference:
    """Run one model behind multiple geometric views and vote on normalized outputs.

    The model is injected as a callable, so this strategy can be reused with any
    client that accepts a prompt and returns the baseline JSON response.
    """

    def __init__(
        self,
        solve_prompt: PromptSolver,
        build_prompt: PromptBuilder,
        transformations: list[TransformSpec] | None = None,
    ) -> None:
        self.solve_prompt = solve_prompt
        self.build_prompt = build_prompt
        self.transformations = transformations or get_transformations()

    def solve_task(self, task: dict[str, Any], prompt_index: int) -> dict[str, Any]:
        """Return the winning baseline-shaped task response plus inference errors."""
        candidates: list[dict[str, Any]] = []
        errors: list[str] = []

        for transform_index, spec in enumerate(self.transformations):
            try:
                augmented_task = transform_task(task, spec.apply)
                prompt = self.build_prompt([augmented_task], prompt_index=prompt_index)
                response = self.solve_prompt(prompt)
                model_task = _find_task(response, task["task_name"])
                if model_task is None:
                    raise ValueError("model response did not contain a task")

                outputs = transform_outputs(
                    model_task.get("predicted_test_outputs"), spec.inverse
                )
                if not isinstance(outputs, list):
                    raise ValueError("predicted_test_outputs is not a list")

                candidates.append(
                    {
                        "outputs": outputs,
                        "logic_explanation": model_task.get("logic_explanation", ""),
                        "transform": spec.name,
                        "transform_index": transform_index,
                    }
                )
            except Exception as exc:
                errors.append(f"{spec.name}: {exc}")

        if not candidates:
            return {"_augmentation_errors": errors}

        winner = _rank_candidates(candidates)[0]
        result = {
            "task_name": task["task_name"],
            "logic_explanation": winner["logic_explanation"],
            "predicted_test_outputs": winner["outputs"],
        }
        if errors:
            result["_augmentation_errors"] = errors
        return result


def _find_task(response: Any, task_name: str) -> dict[str, Any] | None:
    if not isinstance(response, dict) or not isinstance(response.get("tasks"), list):
        return None

    tasks = [item for item in response["tasks"] if isinstance(item, dict)]
    for item in tasks:
        if item.get("task_name") == task_name:
            return item
    return tasks[0] if tasks else None


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"votes": 0, "has_identity": False, "first_index": len(candidates)}
    )
    for candidate in candidates:
        key = json.dumps(candidate["outputs"], separators=(",", ":"), sort_keys=True)
        group = groups[key]
        group["votes"] += 1
        group["has_identity"] |= candidate["transform"] == "identity"
        group["first_index"] = min(group["first_index"], candidate["transform_index"])
        group.setdefault("candidate", candidate)

    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            -group["votes"],
            -int(group["has_identity"]),
            group["first_index"],
        ),
    )
    ranked: list[dict[str, Any]] = []
    for group in ranked_groups:
        candidate = dict(group["candidate"])
        candidate["votes"] = group["votes"]
        ranked.append(candidate)
    return ranked
