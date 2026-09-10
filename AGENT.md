# mini-arc-v12 agent handoff

## Current project state (2026-09-09)

This is a UFRGS class-project reproduction and intervention study based on Paul
Fletcher-Hill's Mini-ARC. The public-facing narrative and current numerical
results are in `README.md`; do not reintroduce the old reservation diary or
describe the paper-aligned result as unknown.

The preferred evaluation checkpoint is
`$HOME/arc-checkpoints/mini-arc-v12-full-refinement/best.pt`. It has the full
Mini-ARC-v12 architecture and a refinement branch trained with noisy partial
targets. The consolidated result currently available in the workspace is
`results/comparison.json`.

## Research claim and limits

The local intervention asks whether strong invariance-based augmentation during
TTT and inference improves the same checkpoint relative to identity-only TTT.
The controlled comparison holds checkpoint, 114 task IDs, first four
demonstrations, first query, seed, TTT hyperparameters, and two refinement
rounds constant.

Final post-refinement paper metrics are:

| Scenario | Score | Accuracy | Closeness |
|---|---:|---:|---:|
| baseline | 5/114 (4.39%) | 89.80% | 40/114 (35.09%) |
| augmentation | 11/114 (9.65%) | 91.40% | 44/114 (38.60%) |
| cheat upper bound | 17/114 (14.91%) | 92.90% | 62/114 (54.39%) |

The defensible conclusion is that augmentation improved this checkpoint by 6
exact solutions (11 versus 5) on the selected 114 tasks. Do not claim an exact
reproduction of the original paper or generalize the effect to all ARC tasks
without another evaluation.

The `cheat` run loaded 1,063 task-associated ARC-GEN pairs and rejected 17. It
is a privileged-data upper bound/positive control, not a fair generalization
result and not evidence suitable for a leaderboard claim.

TTT-only Scores were baseline 5, augmentation 9, and cheat 20. Refinement
changed them to 5, 11, and 17 respectively. Do not assume refinement is
monotonic; in this run it helped augmentation by 2 and hurt cheat by 3.

## Relationship to the original paper

The original paper reports Mini-ARC-v12 TTT + Refined Score of 20/114 (17.5%).
The local baseline is 5/114 (4.39%). The most concrete gap is pretraining data:

- original: 830,648 training puzzles from RE-ARC, BARC Heavy, and ARC-HTML;
- local: 186,556 retained examples across 391 families from reduced RE-ARC.

The local corpus is about 4.5 times smaller and substantially less diverse.
The original training used 4–8 A100 GPUs over multiple days and at least
150,000 steps. The local full-refinement launcher targets 150,000 steps, but
`results/comparison.json` does not store checkpoint global step or history.
Treat reduced realized compute as a plausible but unverified explanation unless
checkpoint metadata or logs are available. Different data balancing, split,
checkpoint selection, and random trajectory may also matter.

This absolute reproduction gap does not invalidate the within-checkpoint
intervention because the three local runs are paired. It does limit the claim's
scope to this checkpoint, configuration, and task subset.

## Metric semantics

- `score`: number of completely correct puzzle outputs; primary solved-task
  metric.
- `accuracy` / `cell_accuracy`: accuracy across padded 12×12 cells.
- `closeness`: puzzles with at least 95% cell accuracy.
- `exact_grid_accuracy` and `score_percent`: `score / puzzles_scored`.

Padding can make cell Accuracy look high even when Score is low. For example,
the baseline has 89.80% Accuracy but solves only 5 puzzles. Always report Score
alongside Accuracy and Closeness.

`ttt_metrics` measure the first prediction from the task-adapted model.
`refined_metrics` measure the result after feeding that prediction back through
the same adapted model for the configured refinement rounds. Refinement is an
inference pass, not additional optimizer training.

## Scenario implementation

`scripts/eval_mini_arc_v12_ttt_comparison_oar.sh` and its PCAD counterpart run:

1. `baseline`: original demonstrations, identity transformation, legacy exact
   vote;
2. `augmentation`: eight D4 geometries, seeded colour permutations,
   demonstration orders, a 256-item TTT cap, and hierarchical canonical-space
   voting;
3. `cheat`: identity transformation plus compatible pairs from
   `to-solve/ARC-GEN/tasks`, with a default 256-item derived TTT cap.

All default to 15 TTT epochs, learning rate `1e-5`, weight decay `1e-5`, batch
size 4, 99.5% cutoff, seed 42, and two refinement rounds. The comparison uses
`data/mini_arc_v12_evaluation_ids.txt`, the first four demonstrations, and the
first query for each of the 114 paper tasks.

The strong augmentation run samples at most 256 identity-anchored items per
task from geometry × colour × order combinations. At inference it defaults to
32 candidates and maps transformed outputs back to canonical space before
hierarchical voting.

## Data and training

Prepare the deterministic, family-balanced reduced RE-ARC dataset with:

```bash
python3 -m arc_prize.rearc_manifest \
  --source ../re-arc/re_arc_5k_12x12 \
  --output data/re_arc_5k_12x12_balanced
```

Expected current output:

- 391 eligible families;
- 186,556 retained examples;
- fingerprint
  `f4c9714c940a05771462d1ee55c4f82fabbd6d56d3d7cd782299de8ce8522483`;
- `manifest.json` and `examples.sqlite3`.

Validation targets are held out within each family and must never be used as
demonstrations. `arc_prize/rearc_dataset.py` implements this property, and
`arc_prize/train_rearc.py` is the authoritative trainer.

Architecture profiles are checkpoint-incompatible:

- `reduced`: 4 layers, 4 heads, `d_model=128`, `d_ff=512`;
- `full`: 16 layers, 16 heads, `d_model=512`, `d_ff=3072`, direct output only;
- `full-refinement`: the same 67,343,755-parameter full architecture with
  noisy-target refinement on 25% of training steps.

Never request refinement for `mini-arc-v12-full/best.pt`; its `tgt_embedding`
path was not trained, and the evaluator correctly rejects it. Use the
independent `full-refinement` checkpoint when `REFINEMENT_ROUNDS>0`.

Train/resume the evaluated profile on Grid'5000 with:

```bash
./scripts/train_mini_arc_v12_full_refinement_oar.sh
```

Defaults are 8 GPUs, batch 16 per rank, 1,000 training steps and 100 validation
steps per epoch, 150 epochs, patience 150, and `REFINEMENT_RATIO=0.25`.
`latest.pt` is the complete resumable state; `best.pt` is selected by lowest
held-out validation loss. Both include model, optimizer, scaler, schedulers,
progress, history, dataset fingerprint, and RNG state. Never replace them with
model weights alone.

The launcher stages source, data, image, and checkpoint under a job-specific
`/tmp/$USER/mini-arc-v12-$OAR_JOB_ID` directory, trains locally, and atomically
mirrors checkpoints to `$HOME`. Do not change it to train directly from NFS.
Do not use `FORCE_RESTART=1` unless the user explicitly wants a new run; the
trainer archives old checkpoints rather than deleting them.

## Evaluation and reproduction

Grid'5000 comparison:

```bash
RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-refinement-ttt-comparison" \
./scripts/eval_mini_arc_v12_ttt_comparison_oar.sh
```

ARM64/PCAD comparison:

```bash
SCRATCH=<scratch-path> PYTHON_BIN=<cuda-python> \
RESULTS_DIR=<results-path> \
./scripts/eval_mini_arc_v12_ttt_comparison_pcad.sh
```

Use `MAX_TASKS=1 TTT_EPOCHS=1` for a smoke test. A complete run writes
`baseline.json`, `augmentation.json`, `cheat.json`, and `comparison.json`.
Preserve the raw reports: unlike the consolidated report, they include
checkpoint/model identity, per-task predictions, TTT epoch counts, and vote
details. They still do not record the pretrained checkpoint's global step or
full training history.

For direct-only paper-aligned evaluation, use
`scripts/eval_mini_arc_v12_full_oar.sh`. Explicitly set
`REFINEMENT_ROUNDS=0` for a direct-only checkpoint.

## Authoritative files

- `README.md`: experiment narrative, results, limitations, short reproduction.
- `results/comparison.json`: consolidated numerical result currently available
  in the workspace.
- `arc_prize/eval_arc_agi.py`: direct, TTT, augmentation, refinement, metrics.
- `arc_prize/compare_ttt_runs.py`: three-run aggregation and task differences.
- `models/data_augmentation_baseline/strong_augmentation.py`: strong policy.
- `arc_prize/train_rearc.py`: v12 training and checkpoint semantics.
- `TRAINING_MINI_ARC_V12.md`: detailed data, Apptainer, and cluster procedure.
- `paper/mini-arc.tex`: local source copy of the original paper.

## Safety and maintenance boundaries

- Do not commit generated datasets, SIF images, checkpoints, cluster logs, or
  job scratch files.
- Do not delete checkpoints or use destructive Git cleanup for generated
  artifacts.
- Preserve old notebooks, Modal paths, and editor code unless a task explicitly
  targets them.
- Keep baseline, augmentation, and cheat labels semantically distinct.
- If publishing a new comparison, record checkpoint global step, dataset
  fingerprint, seed, package versions, launcher environment overrides, and raw
  reports so the absolute result can be audited.
