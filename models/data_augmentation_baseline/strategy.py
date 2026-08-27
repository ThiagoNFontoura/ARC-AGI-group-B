"""Model-independent augmented inference and voting strategy."""

from __future__ import annotations

import json
import time
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
        """Return the winning baseline-shaped task response plus rich ensemble details."""
        candidates: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for transform_index, spec in enumerate(self.transformations):
            view_start = time.perf_counter()
            try:
                augmented_task = transform_task(task, spec.apply)
                prompt = self.build_prompt([augmented_task], prompt_index=prompt_index)
                response = self.solve_prompt(prompt)
                view_latency = round(time.perf_counter() - view_start, 3)

                model_task = _find_task(response, task["task_name"])
                if model_task is None:
                    raise ValueError("model response did not contain a task")

                raw_outputs = model_task.get("predicted_test_outputs")
                outputs = transform_outputs(raw_outputs, spec.inverse)
                if not isinstance(outputs, list):
                    raise ValueError("predicted_test_outputs is not a list")

                usage = response.get("_usage_metadata") if isinstance(response, dict) else None

                candidates.append(
                    {
                        "transform": spec.name,
                        "transform_index": transform_index,
                        "canonical_outputs": outputs,
                        "raw_outputs": raw_outputs,
                        "logic_explanation": model_task.get("logic_explanation", ""),
                        "latency_seconds": view_latency,
                        "token_usage": usage,
                    }
                )
            except Exception as exc:
                view_latency = round(time.perf_counter() - view_start, 3)
                errors.append(
                    {
                        "transform": spec.name,
                        "error": str(exc),
                        "latency_seconds": view_latency,
                    }
                )

        total_latency = round(
            sum(c.get("latency_seconds", 0.0) for c in candidates)
            + sum(e.get("latency_seconds", 0.0) for e in errors),
            3,
        )

        prompt_tokens_list = [
            c["token_usage"]["prompt_tokens"]
            for c in candidates
            if c.get("token_usage") and c["token_usage"].get("prompt_tokens") is not None
        ]
        candidates_tokens_list = [
            c["token_usage"]["candidates_tokens"]
            for c in candidates
            if c.get("token_usage") and c["token_usage"].get("candidates_tokens") is not None
        ]
        total_tokens_list = [
            c["token_usage"]["total_tokens"]
            for c in candidates
            if c.get("token_usage") and c["token_usage"].get("total_tokens") is not None
        ]

        task_telemetry = {
            "total_latency_seconds": total_latency,
            "prompt_tokens": sum(prompt_tokens_list) if prompt_tokens_list else None,
            "candidates_tokens": sum(candidates_tokens_list) if candidates_tokens_list else None,
            "total_tokens": sum(total_tokens_list) if total_tokens_list else None,
        }

        if not candidates:
            return {
                "_augmentation_errors": [f"{e['transform']}: {e['error']}" for e in errors],
                "ensemble_details": {
                    "summary": {
                        "total_views_attempted": len(self.transformations),
                        "valid_votes": 0,
                        "failed_views": len(errors),
                        "consensus_type": "all_failed",
                        "winner_votes": 0,
                        "winner_vote_ratio": "0/0",
                        "winner_vote_percentage": 0.0,
                        "winner_transforms": [],
                        "unique_candidate_count": 0,
                    },
                    "telemetry": task_telemetry,
                    "voting_distribution": [],
                    "individual_views": [
                        {
                            "transform": e["transform"],
                            "status": "error",
                            "error": e["error"],
                            "latency_seconds": e.get("latency_seconds"),
                        }
                        for e in errors
                    ],
                },
            }

        ranked_groups, consensus_type = _analyze_and_rank_candidates(candidates)
        winner_group = ranked_groups[0]
        winner_candidate = winner_group["representative_candidate"]

        ensemble_summary = {
            "total_views_attempted": len(self.transformations),
            "valid_votes": len(candidates),
            "failed_views": len(errors),
            "consensus_type": consensus_type,
            "winner_votes": winner_group["votes"],
            "winner_vote_ratio": f"{winner_group['votes']}/{len(candidates)}",
            "winner_vote_percentage": round((winner_group["votes"] / len(candidates)) * 100, 1),
            "winner_transforms": winner_group["voted_by_transforms"],
            "unique_candidate_count": len(ranked_groups),
        }

        voting_distribution = [
            {
                "vote_count": group["votes"],
                "vote_percentage": round((group["votes"] / len(candidates)) * 100, 1),
                "voted_by": group["voted_by_transforms"],
                "is_winner": (idx == 0),
                "predicted_test_outputs": group["canonical_outputs"],
            }
            for idx, group in enumerate(ranked_groups)
        ]

        individual_views = [
            {
                "transform": c["transform"],
                "status": "success",
                "predicted_test_outputs": c["canonical_outputs"],
                "raw_transformed_outputs": c["raw_outputs"],
                "logic_explanation": c["logic_explanation"],
                "latency_seconds": c.get("latency_seconds"),
                "token_usage": c.get("token_usage"),
            }
            for c in candidates
        ] + [
            {
                "transform": e["transform"],
                "status": "error",
                "error": e["error"],
                "latency_seconds": e.get("latency_seconds"),
            }
            for e in errors
        ]

        result = {
            "task_name": task["task_name"],
            "logic_explanation": winner_candidate["logic_explanation"],
            "predicted_test_outputs": winner_group["canonical_outputs"],
            "ensemble_details": {
                "summary": ensemble_summary,
                "telemetry": task_telemetry,
                "voting_distribution": voting_distribution,
                "individual_views": individual_views,
            },
        }

        if errors:
            result["_augmentation_errors"] = [f"{e['transform']}: {e['error']}" for e in errors]

        return result


def _find_task(response: Any, task_name: str) -> dict[str, Any] | None:
    if not isinstance(response, dict) or not isinstance(response.get("tasks"), list):
        return None

    tasks = [item for item in response["tasks"] if isinstance(item, dict)]
    for item in tasks:
        if item.get("task_name") == task_name:
            return item
    return tasks[0] if tasks else None


def _analyze_and_rank_candidates(
    candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Group candidates by canonical output, rank them, and determine consensus type."""
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "votes": 0,
            "has_identity": False,
            "first_index": len(candidates),
            "voted_by_transforms": [],
            "canonical_outputs": None,
            "representative_candidate": None,
        }
    )

    for candidate in candidates:
        key = json.dumps(candidate["canonical_outputs"], separators=(",", ":"), sort_keys=True)
        group = groups[key]
        group["votes"] += 1
        group["has_identity"] |= candidate["transform"] == "identity"
        group["first_index"] = min(group["first_index"], candidate["transform_index"])
        group["voted_by_transforms"].append(candidate["transform"])
        if group["canonical_outputs"] is None:
            group["canonical_outputs"] = candidate["canonical_outputs"]
            group["representative_candidate"] = candidate

    ranked_groups = sorted(
        groups.values(),
        key=lambda group: (
            -group["votes"],
            -int(group["has_identity"]),
            group["first_index"],
        ),
    )

    total_valid = len(candidates)
    winner = ranked_groups[0]
    winner_votes = winner["votes"]

    if total_valid == 1:
        consensus_type = "single_view"
    elif winner_votes == total_valid:
        consensus_type = "unanimous"
    elif winner_votes > total_valid / 2:
        consensus_type = "majority"
    else:
        # Check if there was a tie for top vote counts
        tied_top = [g for g in ranked_groups if g["votes"] == winner_votes]
        if len(tied_top) > 1:
            if winner["has_identity"] and not all(g["has_identity"] for g in tied_top):
                consensus_type = "tie_broken_by_identity"
            else:
                consensus_type = "tie_broken_by_transform_order"
        else:
            consensus_type = "plurality"

    return ranked_groups, consensus_type

