"""Evaluate a mini-arc-v12 checkpoint on ARC-AGI tasks, with optional TTT."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from arc_prize.data import ARCDatasetParams, pad_and_mask_grid
from arc_prize.model import ARCTransformerEncoderDecoderParams, ARCVisionEncoder
from models.data_augmentation_baseline.strong_augmentation import (
    D4_GEOMETRIES,
    TrainingVariantStats,
    make_views,
    select_inference_orders,
    select_training_variants,
)
from models.data_augmentation_baseline.transforms import (
    TransformSpec,
    get_transformations,
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _valid_grid(grid: Any, grid_dim: int) -> bool:
    return (
        isinstance(grid, list)
        and 0 < len(grid) <= grid_dim
        and all(
            isinstance(row, list)
            and len(row) == len(grid[0])
            and 0 < len(row) <= grid_dim
            and all(isinstance(color, int) and 0 <= color <= 9 for color in row)
            for row in grid
        )
    )


def _task_is_eligible(task: dict[str, Any], grid_dim: int) -> tuple[bool, str]:
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, list) or not isinstance(test, list) or not train or not test:
        return False, "missing train/test examples"
    for example in [*train, *test]:
        if not isinstance(example, dict) or not _valid_grid(example.get("input"), grid_dim):
            return False, "invalid or oversized input grid"
        if "output" in example and not _valid_grid(example["output"], grid_dim):
            return False, "invalid or oversized output grid"
    return True, ""


def _prompt(
    demonstrations: list[dict[str, list[list[int]]]],
    query: list[list[int]],
    config: ARCDatasetParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_count = 2 * config.max_train_grids + 1
    grids = torch.zeros((grid_count, config.max_grid_size, config.max_grid_size), dtype=torch.int)
    masks = torch.zeros_like(grids, dtype=torch.bool)
    for index, example in enumerate(demonstrations):
        grids[2 * index], masks[2 * index] = pad_and_mask_grid(example["input"], config)
        grids[2 * index + 1], masks[2 * index + 1] = pad_and_mask_grid(example["output"], config)
    grids[-1], masks[-1] = pad_and_mask_grid(query, config)
    return grids, masks


def _target(grid: list[list[int]], config: ARCDatasetParams) -> torch.Tensor:
    return pad_and_mask_grid(grid, config)[0]


def _ttt_candidate_count(pool_size: int, maximum_group_size: int, view_count: int) -> int:
    upper_bound = min(pool_size, maximum_group_size)
    return view_count * sum(math.perm(pool_size, length) for length in range(3, upper_bound + 1))


def _ttt_examples(
    train_examples: list[dict[str, list[list[int]]]],
    config: ARCDatasetParams,
    transformations: list[TransformSpec] | None = None,
    *,
    max_examples: int | None = None,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Use every supported ordering and geometric view as an adaptation item.

    The final element of each ordering is held out as its target; its input is
    the query and the preceding elements are demonstrations. This mirrors the
    original mini-arc fine-tuning construction. Group size is capped at the
    model's training-pair count so a larger synthetic pool remains compatible
    with the checkpoint architecture.
    """
    ordered_views: list[
        tuple[TransformSpec, tuple[dict[str, list[list[int]]], ...]]
    ] = []
    specs = transformations or get_transformations(["identity"])
    maximum_group_size = min(len(train_examples), config.max_train_grids)
    for spec in specs:
        for length in range(3, maximum_group_size + 1):
            for combination in itertools.combinations(train_examples, length):
                for ordering in itertools.permutations(combination):
                    ordered_views.append((spec, ordering))
    if max_examples is not None and len(ordered_views) > max_examples:
        ordered_views = random.Random(seed).sample(ordered_views, max_examples)

    examples = []
    for spec, ordering in ordered_views:
        transformed_ordering = [_transform_example(example, spec) for example in ordering]
        held_out = transformed_ordering[-1]
        grids, masks = _prompt(transformed_ordering[:-1], held_out["input"], config)
        examples.append((grids, masks, _target(held_out["output"], config)))
    return examples


def _transform_example(
    example: dict[str, list[list[int]]], spec: TransformSpec
) -> dict[str, list[list[int]]]:
    return {
        "input": spec.apply(example["input"]),
        "output": spec.apply(example["output"]),
    }


def _strong_ttt_examples(
    train_examples: list[dict[str, list[list[int]]]],
    config: ARCDatasetParams,
    *,
    max_examples: int,
    color_permutations: int,
    identity_fraction: float,
    preserve_zero: bool,
    seed: int,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    TrainingVariantStats,
]:
    variants, stats = select_training_variants(
        len(train_examples),
        config.max_train_grids,
        max_examples=max_examples,
        color_permutations=color_permutations,
        identity_fraction=identity_fraction,
        seed=seed,
        preserve_zero=preserve_zero,
        minimum_group_size=2 if len(train_examples) == 2 else 3,
    )
    examples = []
    for variant in variants:
        ordering = [train_examples[index] for index in variant.ordering]
        transformed = [
            {
                "input": variant.view.apply(example["input"]),
                "output": variant.view.apply(example["output"]),
            }
            for example in ordering
        ]
        held_out = transformed[-1]
        grids, masks = _prompt(transformed[:-1], held_out["input"], config)
        examples.append((grids, masks, _target(held_out["output"], config)))
    return examples, stats


def _adapt_model(
    base_model: ARCVisionEncoder,
    train_examples: list[dict[str, list[list[int]]]],
    config: ARCDatasetParams,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    accuracy_cutoff: float,
    transformations: list[TransformSpec] | None = None,
    max_examples: int | None = None,
    seed: int = 42,
    prepared_examples: (
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
    ) = None,
) -> tuple[ARCVisionEncoder, int, int]:
    model = copy.deepcopy(base_model).to(device)
    examples = (
        prepared_examples
        if prepared_examples is not None
        else _ttt_examples(
            train_examples,
            config,
            transformations,
            max_examples=max_examples,
            seed=seed,
        )
    )
    if not examples or epochs == 0:
        return model.eval(), 0, len(examples)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = torch.ones(model.num_classes, device=device)
    weights[0] = 0.2
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    model.train()
    epochs_run = 0
    for _ in range(epochs):
        correct = 0
        cells = 0
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), batch_size):
            chunk = [examples[index] for index in order[start : start + batch_size]]
            grids = torch.stack([item[0] for item in chunk]).to(device, non_blocking=True)
            masks = torch.stack([item[1] for item in chunk]).to(device, non_blocking=True)
            targets = torch.stack([item[2] for item in chunk]).to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(grids, masks)[0]
                loss = criterion(logits.reshape(-1, model.num_classes), targets.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            cells += targets.numel()
        epochs_run += 1
        if correct / cells >= accuracy_cutoff:
            break
    return model.eval(), epochs_run, len(examples)


def _predict(
    model: ARCVisionEncoder,
    grids: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    *,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad(), autocast(device_type=device.type, enabled=device.type == "cuda"):
        target_batch = target.unsqueeze(0).to(device) if target is not None else None
        return model.generate(
            grids.unsqueeze(0).to(device),
            masks.unsqueeze(0).to(device),
            tgt=target_batch,
        )[0][0].cpu()


def _crop_prediction(prediction: torch.Tensor) -> list[list[int]]:
    """Convert 0=padding, 1..10=ARC colors into a JSON ARC grid."""
    occupied = prediction != 0
    if not occupied.any():
        return [[0]]
    rows = occupied.any(dim=1).nonzero(as_tuple=True)[0]
    cols = occupied.any(dim=0).nonzero(as_tuple=True)[0]
    cropped = prediction[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] - 1
    return cropped.clamp_min(0).tolist()


def _vote_predictions(
    candidates: list[tuple[TransformSpec, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Vote like data_augmentation_baseline, including its deterministic ties."""
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "votes": 0,
            "has_identity": False,
            "first_index": len(candidates),
            "transforms": [],
            "prediction": None,
        }
    )
    for transform_index, (spec, prediction) in enumerate(candidates):
        key = json.dumps(prediction.tolist(), separators=(",", ":"))
        group = groups[key]
        group["votes"] += 1
        group["has_identity"] |= spec.name == "identity"
        group["first_index"] = min(group["first_index"], transform_index)
        group["transforms"].append(spec.name)
        if group["prediction"] is None:
            group["prediction"] = prediction
    ranked = sorted(
        groups.values(),
        key=lambda group: (-group["votes"], -int(group["has_identity"]), group["first_index"]),
    )
    winner = ranked[0]
    return winner["prediction"], {
        "valid_votes": len(candidates),
        "winner_votes": winner["votes"],
        "winner_transforms": winner["transforms"],
        "unique_candidate_count": len(ranked),
        "distribution": [
            {
                "votes": group["votes"],
                "transforms": group["transforms"],
                "prediction": _crop_prediction(group["prediction"]),
            }
            for group in ranked
        ],
    }


def _predict_with_augmentation(
    model: ARCVisionEncoder,
    demonstrations: list[dict[str, list[list[int]]]],
    query: list[list[int]],
    config: ARCDatasetParams,
    device: torch.device,
    transformations: list[TransformSpec],
    *,
    refinement_rounds: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Predict each geometric view, invert it, and vote in canonical space."""
    if len(transformations) == 1 and transformations[0].name == "identity":
        grids, masks = _prompt(demonstrations, query, config)
        tuned = _predict(model, grids, masks, device)
        refined = tuned
        for _ in range(refinement_rounds):
            refined = _predict(model, grids, masks, device, target=refined)
        return tuned, refined, {
            "valid_votes": 1,
            "winner_votes": 1,
            "winner_transforms": ["identity"],
            "unique_candidate_count": 1,
        }

    tuned_candidates: list[tuple[TransformSpec, torch.Tensor]] = []
    refined_candidates: list[tuple[TransformSpec, torch.Tensor]] = []
    individual_views: list[dict[str, Any]] = []
    for spec in transformations:
        transformed_demonstrations = [
            _transform_example(example, spec) for example in demonstrations
        ]
        grids, masks = _prompt(transformed_demonstrations, spec.apply(query), config)
        tuned_view = _predict(model, grids, masks, device)
        refined_view = tuned_view
        for _ in range(refinement_rounds):
            refined_view = _predict(model, grids, masks, device, target=refined_view)

        tuned_raw = spec.inverse(_crop_prediction(tuned_view))
        refined_raw = spec.inverse(_crop_prediction(refined_view))
        tuned_canonical = _target(tuned_raw, config)
        refined_canonical = _target(refined_raw, config)
        tuned_candidates.append((spec, tuned_canonical))
        refined_candidates.append((spec, refined_canonical))
        individual_views.append(
            {
                "transform": spec.name,
                "ttt_prediction": tuned_raw,
                "refined_prediction": refined_raw if refinement_rounds > 0 else None,
            }
        )

    tuned, tuned_vote = _vote_predictions(tuned_candidates)
    refined, refined_vote = _vote_predictions(refined_candidates)
    return tuned, refined, {
        **tuned_vote,
        "individual_views": individual_views,
        "refined_vote": refined_vote if refinement_rounds > 0 else None,
    }


def _row_majority(predictions: list[torch.Tensor]) -> torch.Tensor:
    rows = []
    for row_index in range(predictions[0].shape[0]):
        counts: dict[tuple[int, ...], int] = {}
        first_seen: dict[tuple[int, ...], int] = {}
        for index, prediction in enumerate(predictions):
            key = tuple(int(value) for value in prediction[row_index].tolist())
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, index)
        winner = min(counts, key=lambda key: (-counts[key], first_seen[key]))
        rows.append(torch.tensor(winner, dtype=predictions[0].dtype))
    return torch.stack(rows)


def _column_majority(predictions: list[torch.Tensor]) -> torch.Tensor:
    columns = []
    for column_index in range(predictions[0].shape[1]):
        counts: dict[tuple[int, ...], int] = {}
        first_seen: dict[tuple[int, ...], int] = {}
        for index, prediction in enumerate(predictions):
            key = tuple(int(value) for value in prediction[:, column_index].tolist())
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, index)
        winner = min(counts, key=lambda key: (-counts[key], first_seen[key]))
        columns.append(torch.tensor(winner, dtype=predictions[0].dtype))
    return torch.stack(columns, dim=1)


def _hierarchical_vote_predictions(
    candidates: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Vote within each D4 geometry, then vote across geometry candidates."""
    by_geometry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_geometry[candidate["geometry"]].append(candidate)

    stage_one: list[dict[str, Any]] = []
    stage_one_summary: dict[str, list[dict[str, Any]]] = {}
    for geometry in [spec.name for spec in D4_GEOMETRIES]:
        geometry_candidates = by_geometry.get(geometry, [])
        if not geometry_candidates:
            continue
        groups: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(geometry_candidates):
            key = json.dumps(candidate["prediction"].tolist(), separators=(",", ":"))
            group = groups.setdefault(
                key,
                {
                    "prediction": candidate["prediction"],
                    "votes": 0,
                    "has_identity_color": False,
                    "has_original_order": False,
                    "first_index": index,
                    "sources": [],
                },
            )
            group["votes"] += 1
            group["has_identity_color"] |= candidate["color"] == "identity"
            group["has_original_order"] |= candidate["order_index"] == 0
            group["sources"].append(
                {
                    "color": candidate["color"],
                    "order_index": candidate["order_index"],
                }
            )
        ranked = sorted(
            groups.values(),
            key=lambda group: (
                -group["votes"],
                -int(group["has_identity_color"]),
                -int(group["has_original_order"]),
                group["first_index"],
            ),
        )

        selected = ranked[:3]
        existing_keys = {
            json.dumps(group["prediction"].tolist(), separators=(",", ":"))
            for group in selected
        }
        raw_predictions = [candidate["prediction"] for candidate in geometry_candidates]
        for source, prediction in (
            ("row_majority", _row_majority(raw_predictions)),
            ("column_majority", _column_majority(raw_predictions)),
        ):
            if len(selected) >= 3:
                break
            key = json.dumps(prediction.tolist(), separators=(",", ":"))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            selected.append(
                {
                    "prediction": prediction,
                    "votes": 0,
                    "has_identity_color": False,
                    "has_original_order": False,
                    "first_index": len(geometry_candidates),
                    "sources": [{"synthetic": source}],
                }
            )

        stage_one_summary[geometry] = [
            {
                "intra_votes": group["votes"],
                "sources": group["sources"],
                "prediction": _crop_prediction(group["prediction"]),
            }
            for group in selected
        ]
        stage_one.extend(
            {
                **group,
                "geometry": geometry,
            }
            for group in selected
        )

    global_groups: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(stage_one):
        key = json.dumps(candidate["prediction"].tolist(), separators=(",", ":"))
        group = global_groups.setdefault(
            key,
            {
                "prediction": candidate["prediction"],
                "votes": 0,
                "has_identity_geometry": False,
                "first_index": index,
                "geometries": [],
            },
        )
        group["votes"] += 1
        group["has_identity_geometry"] |= candidate["geometry"] == "identity"
        group["geometries"].append(candidate["geometry"])
    ranked_global = sorted(
        global_groups.values(),
        key=lambda group: (
            -group["votes"],
            -int(group["has_identity_geometry"]),
            group["first_index"],
        ),
    )
    winner = ranked_global[0]
    return winner["prediction"], {
        "strategy": "hierarchical",
        "valid_votes": len(candidates),
        "geometry_groups": len(by_geometry),
        "stage_one_candidates": len(stage_one),
        "winner_votes": winner["votes"],
        "winner_geometries": winner["geometries"],
        "unique_candidate_count": len(ranked_global),
        "distribution": [
            {
                "votes": group["votes"],
                "geometries": group["geometries"],
                "prediction": _crop_prediction(group["prediction"]),
            }
            for group in ranked_global
        ],
        "intra_geometry": stage_one_summary,
    }


def _predict_with_strong_augmentation(
    model: ARCVisionEncoder,
    demonstrations: list[dict[str, list[list[int]]]],
    query: list[list[int]],
    config: ARCDatasetParams,
    device: torch.device,
    *,
    refinement_rounds: int,
    color_permutations: int,
    inference_orders: int,
    preserve_zero: bool,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    views = make_views(
        color_permutations=color_permutations,
        seed=seed,
        preserve_zero=preserve_zero,
    )
    orders = select_inference_orders(
        len(demonstrations),
        inference_orders,
        seed=seed + 1,
    )
    tuned_candidates: list[dict[str, Any]] = []
    refined_candidates: list[dict[str, Any]] = []
    individual_views: list[dict[str, Any]] = []
    for view in views:
        for order_index, ordering in enumerate(orders):
            ordered_demonstrations = [demonstrations[index] for index in ordering]
            transformed_demonstrations = [
                {
                    "input": view.apply(example["input"]),
                    "output": view.apply(example["output"]),
                }
                for example in ordered_demonstrations
            ]
            grids, masks = _prompt(
                transformed_demonstrations,
                view.apply(query),
                config,
            )
            tuned_view = _predict(model, grids, masks, device)
            refined_view = tuned_view
            for _ in range(refinement_rounds):
                refined_view = _predict(model, grids, masks, device, target=refined_view)

            tuned_raw = view.inverse(_crop_prediction(tuned_view))
            refined_raw = view.inverse(_crop_prediction(refined_view))
            common = {
                "geometry": view.geometry.name,
                "color": view.colors.name,
                "order_index": order_index,
            }
            tuned_candidates.append(
                {
                    **common,
                    "prediction": _target(tuned_raw, config),
                }
            )
            refined_candidates.append(
                {
                    **common,
                    "prediction": _target(refined_raw, config),
                }
            )
            individual_views.append(
                {
                    **common,
                    "ordering": list(ordering),
                    "ttt_prediction": tuned_raw,
                    "refined_prediction": refined_raw if refinement_rounds > 0 else None,
                }
            )

    tuned, tuned_vote = _hierarchical_vote_predictions(tuned_candidates)
    refined, refined_vote = _hierarchical_vote_predictions(refined_candidates)
    return tuned, refined, {
        **tuned_vote,
        "individual_views": individual_views,
        "refined_vote": refined_vote if refinement_rounds > 0 else None,
    }


def _load_extra_examples(
    directory: Path | None,
    task_id: str,
    grid_dim: int,
) -> tuple[list[dict[str, list[list[int]]]], dict[str, Any]]:
    """Load ARC-GEN pairs, excluding fallback task copies and oversized grids."""
    if directory is None:
        return [], {"status": "disabled", "loaded": 0, "rejected": 0}
    path = directory / f"{task_id}.json"
    if not path.is_file():
        return [], {"status": "missing", "loaded": 0, "rejected": 0}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        # ARC-GEN stored the original ARC task for families it could not generate.
        # Those pairs are not additional data and must not be duplicated.
        return [], {"status": "original_task_fallback", "loaded": 0, "rejected": 0}
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of generated pairs in {path}")

    valid = []
    rejected = 0
    for pair in payload:
        if (
            isinstance(pair, dict)
            and _valid_grid(pair.get("input"), grid_dim)
            and _valid_grid(pair.get("output"), grid_dim)
        ):
            valid.append({"input": pair["input"], "output": pair["output"]})
        else:
            rejected += 1
    return valid, {"status": "loaded", "loaded": len(valid), "rejected": rejected}


def _metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    task_ids: list[str],
) -> dict[str, float | int]:
    if not targets:
        return {
            "puzzles_scored": 0,
            "queries_scored": 0,
            "score": 0,
            "score_percent": 0.0,
            "accuracy": 0.0,
            "closeness": 0,
            "closeness_percent": 0.0,
            "cell_accuracy": 0.0,
            "exact_grid_accuracy": 0.0,
        }
    if not (len(predictions) == len(targets) == len(task_ids)):
        raise ValueError("predictions, targets, and task IDs must have equal lengths")
    cells = sum(target.numel() for target in targets)
    correct = sum((prediction == target).sum().item() for prediction, target in zip(predictions, targets))
    exact_queries = sum(torch.equal(prediction, target) for prediction, target in zip(predictions, targets))
    task_cells: dict[str, int] = {}
    task_correct: dict[str, int] = {}
    task_exact: dict[str, bool] = {}
    for task_id, prediction, target in zip(task_ids, predictions, targets):
        task_cells[task_id] = task_cells.get(task_id, 0) + target.numel()
        task_correct[task_id] = task_correct.get(task_id, 0) + (prediction == target).sum().item()
        task_exact[task_id] = task_exact.get(task_id, True) and torch.equal(prediction, target)
    puzzle_count = len(task_cells)
    score = sum(task_exact.values())
    closeness = sum(task_correct[task_id] / task_cells[task_id] >= 0.95 for task_id in task_cells)
    return {
        "puzzles_scored": puzzle_count,
        "queries_scored": len(targets),
        "score": score,
        "score_percent": score / puzzle_count,
        "accuracy": correct / cells,
        "closeness": closeness,
        "closeness_percent": closeness / puzzle_count,
        "cell_accuracy": correct / cells,
        "exact_grid_accuracy": exact_queries / len(targets),
    }


def _load_task_ids(path: Path | None, challenges: dict[str, Any]) -> list[str]:
    if path is None:
        return list(challenges)
    task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task ID in {path}")
    missing = [task_id for task_id in task_ids if task_id not in challenges]
    if missing:
        raise ValueError(f"{len(missing)} requested task IDs are absent from challenges: {missing[:5]}")
    return task_ids


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "vision_encoder":
        raise ValueError("checkpoint is not a vision_encoder checkpoint")
    trained_refinement_ratio = checkpoint.get("train_config", {}).get("refinement_ratio", 0.0)
    if args.refinement_rounds > 0 and trained_refinement_ratio <= 0.0:
        raise ValueError(
            "refinement was requested, but this checkpoint never trained its refinement branch"
        )
    params = ARCTransformerEncoderDecoderParams(**checkpoint["model_params"])
    model = ARCVisionEncoder(params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = ARCDatasetParams(max_grid_size=params.grid_dim, max_train_grids=params.num_train_pairs, color_offset=1)
    challenges = _load_json(args.challenges)
    solutions = _load_json(args.solutions) if args.solutions else {}
    requested_task_ids = _load_task_ids(args.task_ids_file, challenges)
    augmentation_transforms = get_transformations(args.augmentation_transforms)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    direct_predictions: list[torch.Tensor] = []
    tuned_predictions: list[torch.Tensor] = []
    refined_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    scored_task_ids: list[str] = []
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    extra_example_stats: dict[str, dict[str, Any]] = {}
    strong_augmentation_stats: dict[str, dict[str, int]] = {}
    total_extra_examples = 0
    total_rejected_extra_examples = 0
    eligible = 0
    for task_index, task_id in enumerate(requested_task_ids):
        if args.max_tasks is not None and task_index >= args.max_tasks:
            break
        task = challenges[task_id]
        eligible_task, reason = _task_is_eligible(task, params.grid_dim)
        task_solutions = solutions.get(task_id, [])
        if eligible_task and task_solutions:
            if len(task_solutions) != len(task["test"]):
                eligible_task = False
                reason = "solution count does not match test query count"
            elif any(not _valid_grid(solution, params.grid_dim) for solution in task_solutions):
                eligible_task = False
                reason = "invalid or oversized known solution grid"
        if not eligible_task:
            skipped[task_id] = reason
            continue
        eligible += 1
        all_train_examples = task["train"]
        train_examples = all_train_examples[: params.num_train_pairs]
        extra_examples, task_extra_stats = _load_extra_examples(
            args.ttt_extra_examples_dir,
            task_id,
            params.grid_dim,
        )
        extra_example_stats[task_id] = task_extra_stats
        total_extra_examples += task_extra_stats["loaded"]
        total_rejected_extra_examples += task_extra_stats["rejected"]
        adaptation_examples = [*train_examples, *extra_examples]
        task_seed = args.seed + task_index
        prepared_examples = None
        strong_stats = None
        if args.strong_augmentation:
            prepared_examples, strong_stats = _strong_ttt_examples(
                adaptation_examples,
                config,
                max_examples=args.strong_ttt_max_examples,
                color_permutations=args.strong_training_color_permutations,
                identity_fraction=args.strong_identity_fraction,
                preserve_zero=not args.strong_permute_zero,
                seed=task_seed,
            )
            strong_augmentation_stats[task_id] = asdict(strong_stats)
        tuned_model, ttt_epochs_run, ttt_examples = (
            _adapt_model(
                model,
                adaptation_examples,
                config,
                device,
                epochs=args.ttt_epochs,
                learning_rate=args.ttt_learning_rate,
                weight_decay=args.ttt_weight_decay,
                batch_size=args.ttt_batch_size,
                accuracy_cutoff=args.ttt_accuracy_cutoff,
                transformations=augmentation_transforms,
                max_examples=args.ttt_max_examples,
                seed=task_seed,
                prepared_examples=prepared_examples,
            )
            if args.ttt_epochs > 0
            else (model, 0, 0)
        )
        task_results = []
        task_queries = task["test"][:1] if args.first_query_only else task["test"]
        for query_index, query in enumerate(task_queries):
            grids, masks = _prompt(train_examples, query["input"], config)
            direct = _predict(model, grids, masks, device)
            if args.strong_augmentation:
                tuned, refined, augmentation_vote = _predict_with_strong_augmentation(
                    tuned_model,
                    train_examples,
                    query["input"],
                    config,
                    device,
                    refinement_rounds=args.refinement_rounds,
                    color_permutations=args.strong_inference_color_permutations,
                    inference_orders=args.strong_inference_orders,
                    preserve_zero=not args.strong_permute_zero,
                    seed=task_seed + 100_000,
                )
            else:
                tuned, refined, augmentation_vote = _predict_with_augmentation(
                    tuned_model,
                    train_examples,
                    query["input"],
                    config,
                    device,
                    augmentation_transforms,
                    refinement_rounds=args.refinement_rounds,
                )
            record: dict[str, Any] = {
                "direct_prediction": _crop_prediction(direct),
                "ttt_prediction": _crop_prediction(tuned),
                "demonstrations_available": len(all_train_examples),
                "demonstrations_used": len(train_examples),
                "extra_examples_used": len(extra_examples),
                "adaptation_pool_size": len(adaptation_examples),
                "ttt_candidate_examples": (
                    strong_stats.candidate_variants
                    if strong_stats is not None
                    else _ttt_candidate_count(
                        len(adaptation_examples),
                        config.max_train_grids,
                        len(augmentation_transforms),
                    )
                ),
                "ttt_examples": ttt_examples,
                "ttt_epochs_run": ttt_epochs_run,
                "augmentation_vote": augmentation_vote,
            }
            if args.refinement_rounds > 0:
                record["refined_prediction"] = _crop_prediction(refined)
            if query_index < len(task_solutions):
                target = _target(task_solutions[query_index], config)
                direct_predictions.append(direct)
                tuned_predictions.append(tuned)
                refined_predictions.append(refined)
                targets.append(target)
                scored_task_ids.append(task_id)
                record["target"] = task_solutions[query_index]
                record["direct_exact"] = bool(torch.equal(direct, target))
                record["ttt_exact"] = bool(torch.equal(tuned, target))
                if args.refinement_rounds > 0:
                    record["refined_exact"] = bool(torch.equal(refined, target))
            task_results.append(record)
        results[task_id] = task_results
        del tuned_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "model_params": asdict(params),
        "ttt": {
            "epochs": args.ttt_epochs,
            "learning_rate": args.ttt_learning_rate,
            "weight_decay": args.ttt_weight_decay,
            "batch_size": args.ttt_batch_size,
            "accuracy_cutoff": args.ttt_accuracy_cutoff,
            "seed": args.seed,
            "augmentation_transforms": (
                [spec.name for spec in D4_GEOMETRIES]
                if args.strong_augmentation
                else [spec.name for spec in augmentation_transforms]
            ),
            "augmentation_strategy": (
                "strong_d4_color_order_hierarchical"
                if args.strong_augmentation
                else "legacy_geometric_exact_vote"
            ),
            "extra_examples_directory": (
                str(args.ttt_extra_examples_dir.resolve())
                if args.ttt_extra_examples_dir is not None
                else None
            ),
            "extra_examples_loaded": total_extra_examples,
            "extra_examples_rejected": total_rejected_extra_examples,
            "extra_examples_by_task": extra_example_stats,
            "max_examples_per_task": (
                args.strong_ttt_max_examples
                if args.strong_augmentation
                else args.ttt_max_examples
            ),
            "strong_augmentation": (
                {
                    "training_color_permutations": args.strong_training_color_permutations,
                    "inference_color_permutations": args.strong_inference_color_permutations,
                    "inference_orders": args.strong_inference_orders,
                    "identity_fraction": args.strong_identity_fraction,
                    "preserve_zero": not args.strong_permute_zero,
                    "training_variants_by_task": strong_augmentation_stats,
                }
                if args.strong_augmentation
                else None
            ),
        },
        "refinement_rounds": args.refinement_rounds,
        "first_query_only": args.first_query_only,
        "tasks_requested": (
            len(requested_task_ids)
            if args.max_tasks is None
            else min(args.max_tasks, len(requested_task_ids))
        ),
        "tasks_eligible": eligible,
        "tasks_skipped": skipped,
        "direct_metrics": _metrics(direct_predictions, targets, scored_task_ids),
        "ttt_metrics": _metrics(tuned_predictions, targets, scored_task_ids),
        "predictions": results,
    }
    if args.refinement_rounds > 0:
        report["refined_metrics"] = _metrics(refined_predictions, targets, scored_task_ids)
    _atomic_json_dump(report, Path(args.output))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ttt-epochs", type=int, default=15)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ttt-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ttt-batch-size", type=int, default=4)
    parser.add_argument("--ttt-accuracy-cutoff", type=float, default=0.995)
    parser.add_argument(
        "--augmentation-transforms",
        nargs="+",
        default=["identity"],
        help=(
            "geometric views used both to augment TTT data and to vote over "
            "predictions (default: identity only)"
        ),
    )
    parser.add_argument(
        "--strong-augmentation",
        action="store_true",
        help=(
            "use the separate D4 + colour + demonstration-order TTT strategy "
            "with hierarchical inference voting"
        ),
    )
    parser.add_argument(
        "--strong-ttt-max-examples",
        type=int,
        default=256,
        help="maximum strong-augmentation TTT items per task (default: 256)",
    )
    parser.add_argument(
        "--strong-training-color-permutations",
        type=int,
        default=4,
        help="identity plus seeded colour mappings used for strong TTT (default: 4)",
    )
    parser.add_argument(
        "--strong-inference-color-permutations",
        type=int,
        default=2,
        help="colour mappings per D4 geometry at inference (default: 2)",
    )
    parser.add_argument(
        "--strong-inference-orders",
        type=int,
        default=2,
        help="demonstration orders per geometry/colour inference view (default: 2)",
    )
    parser.add_argument(
        "--strong-identity-fraction",
        type=float,
        default=0.25,
        help="target share of original identity items in capped strong TTT data",
    )
    parser.add_argument(
        "--strong-permute-zero",
        action="store_true",
        help="allow strong colour permutations to remap ARC colour 0",
    )
    parser.add_argument(
        "--ttt-extra-examples-dir",
        type=Path,
        help="directory containing <task-id>.json lists of extra supervised pairs",
    )
    parser.add_argument(
        "--ttt-max-examples",
        type=int,
        help="deterministically sample at most this many derived TTT items per task",
    )
    parser.add_argument("--refinement-rounds", type=int, default=0)
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument(
        "--first-query-only",
        action="store_true",
        help="score one query per puzzle, matching the original paper's evaluation artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.ttt_epochs < 0 or args.refinement_rounds < 0 or args.ttt_batch_size < 1:
        parser.error("TTT epochs must be non-negative and batch size must be positive")
    if args.ttt_max_examples is not None and args.ttt_max_examples < 1:
        parser.error("TTT max examples must be positive")
    if args.strong_ttt_max_examples < 1:
        parser.error("strong TTT max examples must be positive")
    if (
        args.strong_training_color_permutations < 1
        or args.strong_inference_color_permutations < 1
        or args.strong_inference_orders < 1
    ):
        parser.error("strong augmentation colour/order counts must be positive")
    if not 0.0 <= args.strong_identity_fraction <= 1.0:
        parser.error("strong identity fraction must be in [0, 1]")
    if not 0.0 < args.ttt_accuracy_cutoff <= 1.0:
        parser.error("TTT accuracy cutoff must be in (0, 1]")
    try:
        get_transformations(args.augmentation_transforms)
    except ValueError as exc:
        parser.error(str(exc))
    if args.strong_augmentation and args.augmentation_transforms != ["identity"]:
        parser.error(
            "strong augmentation is separate from --augmentation-transforms; "
            "leave the legacy transform list as identity"
        )
    if args.ttt_extra_examples_dir is not None and not args.ttt_extra_examples_dir.is_dir():
        parser.error(f"extra examples directory not found: {args.ttt_extra_examples_dir}")
    return args


def main() -> int:
    report = evaluate(_parse_args())
    summary_keys = ["tasks_eligible", "tasks_skipped", "direct_metrics", "ttt_metrics"]
    if "refined_metrics" in report:
        summary_keys.append("refined_metrics")
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
