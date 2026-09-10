"""Build a compact comparison report from three mini-ARC TTT evaluations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _final_stage(report: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if report.get("refinement_rounds", 0) > 0 and "refined_metrics" in report:
        return "refined", "refined_exact", report["refined_metrics"]
    return "ttt", "ttt_exact", report["ttt_metrics"]


def _solved_tasks(report: dict[str, Any], exact_field: str) -> set[str]:
    solved = set()
    for task_id, queries in report.get("predictions", {}).items():
        if queries and all(query.get(exact_field) is True for query in queries):
            solved.add(task_id)
    return solved


def _adaptation_summary(report: dict[str, Any]) -> dict[str, int | float]:
    task_records = [
        queries[0]
        for queries in report.get("predictions", {}).values()
        if isinstance(queries, list) and queries
    ]
    item_counts = [int(record.get("ttt_examples", 0)) for record in task_records]
    candidate_counts = [
        int(record.get("ttt_candidate_examples", record.get("ttt_examples", 0)))
        for record in task_records
    ]
    pool_sizes = [int(record.get("adaptation_pool_size", 0)) for record in task_records]
    return {
        "tasks": len(task_records),
        "derived_ttt_items_total": sum(item_counts),
        "derived_ttt_candidates_before_cap_total": sum(candidate_counts),
        "derived_ttt_items_mean": sum(item_counts) / len(item_counts) if item_counts else 0.0,
        "adaptation_pool_size_mean": sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0.0,
        "adaptation_pool_size_max": max(pool_sizes, default=0),
    }


def _atomic_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary_name, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def compare(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checkpoints = {report.get("checkpoint") for report in reports.values()}
    if len(checkpoints) != 1:
        raise ValueError("all runs must use the same checkpoint")

    stages = {name: _final_stage(report) for name, report in reports.items()}
    stage_names = {stage[0] for stage in stages.values()}
    if len(stage_names) != 1:
        raise ValueError("all runs must use the same final prediction stage")
    solved = {
        name: _solved_tasks(reports[name], exact_field)
        for name, (_, exact_field, _) in stages.items()
    }
    baseline_metrics = stages["baseline"][2]
    runs = {}
    for name, report in reports.items():
        final_stage, _, metrics = stages[name]
        runs[name] = {
            "final_stage": final_stage,
            "final_metrics": metrics,
            "ttt_metrics": report["ttt_metrics"],
            "refined_metrics": report.get("refined_metrics"),
            "score_delta_vs_baseline": metrics["score"] - baseline_metrics["score"],
            "cell_accuracy_delta_vs_baseline": (
                metrics["cell_accuracy"] - baseline_metrics["cell_accuracy"]
            ),
            "augmentation_transforms": report["ttt"]["augmentation_transforms"],
            "augmentation_strategy": report["ttt"].get(
                "augmentation_strategy", "legacy_geometric_exact_vote"
            ),
            "strong_augmentation": report["ttt"].get("strong_augmentation"),
            "extra_examples_loaded": report["ttt"]["extra_examples_loaded"],
            "extra_examples_rejected": report["ttt"]["extra_examples_rejected"],
            "max_examples_per_task": report["ttt"]["max_examples_per_task"],
            "adaptation": _adaptation_summary(report),
            "solved_tasks": sorted(solved[name]),
        }

    return {
        "checkpoint": checkpoints.pop(),
        "runs": runs,
        "task_differences": {
            "augmentation_fixed_vs_baseline": sorted(solved["augmentation"] - solved["baseline"]),
            "augmentation_regressed_vs_baseline": sorted(
                solved["baseline"] - solved["augmentation"]
            ),
            "cheat_fixed_vs_baseline": sorted(solved["cheat"] - solved["baseline"]),
            "cheat_regressed_vs_baseline": sorted(solved["baseline"] - solved["cheat"]),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--augmentation", required=True, type=Path)
    parser.add_argument("--cheat", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare(
        {
            "baseline": _load(args.baseline),
            "augmentation": _load(args.augmentation),
            "cheat": _load(args.cheat),
        }
    )
    _atomic_dump(report, args.output)
    print(json.dumps(report["runs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
