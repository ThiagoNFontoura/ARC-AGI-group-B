# mini-arc-v12 — UFRGS class project

## Credits

This repository is based on the original **mini-arc / ARC Prize** project created
by **Paul Fletcher-Hill**. The original project and paper are available at
[mini-arc.pdf](https://www.paulfletcherhill.com/mini-arc.pdf).

The `mini-arc-v12` training path is work for a class project at the
**Federal University of Rio Grande do Sul (UFRGS)**. It adds a reduced 12×12
patch-based model, balanced RE-ARC preparation, portable Apptainer execution,
and restart-safe training on Grid'5000.

## What this project trains

`mini-arc-v12` is a fresh `vision_encoder` model with:

- 12×12 input/output grids;
- four ARC demonstration pairs plus one query grid;
- 2×2 patch embeddings;
- four encoder layers, four attention heads, `d_model=128`, and `d_ff=512`.

The default `reduced` profile is the validated baseline. The separate `full`
profile reproduces the creator's architecture: 16 layers, 16 heads,
`d_model=512`, and `d_ff=3072`. It uses independent checkpoints and is launched
with `scripts/train_mini_arc_v12_full_oar.sh`.

The completed `full` profile uses direct-output pre-training. Test-time training
(TTT) is performed separately during ARC-AGI evaluation, on a temporary copy of
the model for each puzzle. Its optional `tgt_embedding` path is untrained, so
refinement must not be requested for that checkpoint. The independent
`full-refinement` profile trains the same 67.3-million-parameter architecture
from scratch with noisy partial targets on 25% of steps, matching the paper's
refinement setup without overwriting the direct-only checkpoint.

Training uses the reduced RE-ARC dataset. Examples are sampled uniformly by
generator family, so large families do not dominate the objective. Validation
targets are held out per family and never appear as demonstrations.

## Project files

- `arc_prize/rearc_manifest.py` validates raw RE-ARC JSON and produces the
  prepared dataset.
- `arc_prize/rearc_dataset.py` creates deterministic, balanced training and
  validation tasks.
- `arc_prize/train_rearc.py` implements DDP training, metrics, checkpointing,
  signal handling, and resume.
- `arc_prize/eval_arc_agi.py` evaluates a checkpoint on ARC-AGI JSON tasks,
  reports direct and TTT metrics when solutions are available, and writes ARC
  predictions for tasks without solutions.
- `containers/mini-arc-v12.def` defines the PyTorch/Apptainer image.
- `scripts/train_mini_arc_v12_oar.sh` stages a Grid'5000 job into `/tmp` and
  mirrors checkpoints back to `$HOME`.
- `scripts/train_mini_arc_v12_full_oar.sh` selects the original full profile
  and `$HOME/arc-checkpoints/mini-arc-v12-full`.
- `scripts/train_mini_arc_v12_full_refinement_oar.sh` trains 150,000 steps with
  25% refinement in `$HOME/arc-checkpoints/mini-arc-v12-full-refinement` and
  saves after every approximately two-minute epoch.
- `scripts/train_mini_arc_v12_full_sirius_night_resume.sh` is the one-night,
  eight-A100 resume wrapper for the full checkpoint.

## Quick workflow

1. Prepare the dataset without a GPU:

   ```bash
   python3 -m arc_prize.rearc_manifest \
     --source ../re-arc/re_arc_5k_12x12 \
     --output data/re_arc_5k_12x12_balanced
   ```

2. Build the container on a Linux/amd64 machine with Apptainer:

   ```bash
   ./scripts/build_mini_arc_v12_container.sh \
     ./mini-arc-v12-pytorch2.4.1-cuda12.1.sif
   ```

3. Copy the prepared dataset, SIF, and source code to persistent Grid'5000
   `$HOME` storage.

4. Inside an OAR GPU allocation, verify the image with `apptainer exec --nv`,
   run a two-step smoke test, then start the launcher.

See [TRAINING_MINI_ARC_V12.md](TRAINING_MINI_ARC_V12.md) for exact transfer,
smoke-test, OAR, checkpoint, and resume commands. See [AGENT.md](AGENT.md) for
an implementation-oriented handoff.

## Paper-aligned ARC evaluation

The evaluator defaults to the paper's important TTT settings: all permutations
of all combinations containing at least three demonstration pairs, up to 15
epochs, and a 99.5% training-accuracy cutoff. The launcher evaluates the 114
paper task IDs, uses the first four demonstrations and first test query as the
original experiment artifacts did, and reports the paper's Score, Accuracy,
and Closeness metrics. Accuracy includes centered 12×12 padding, so Score is the
metric for completely solved puzzles.

Run the direct-only checkpoint with TTT:

```bash
RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-paper-ttt" \
./scripts/eval_mini_arc_v12_full_oar.sh
```

To compare the best `full-refinement` checkpoint under standard TTT, four-view
geometric augmentation, and the ARC-GEN extra-example upper bound, run:

```bash
RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-refinement-ttt-comparison" \
./scripts/eval_mini_arc_v12_ttt_comparison_oar.sh
```

All three runs use identical TTT hyperparameters, task IDs, seed, checkpoint,
and test queries. It defaults to
`$HOME/arc-checkpoints/mini-arc-v12-full-refinement/best.pt` and uses two
refinement rounds. The augmentation run uses `identity`, horizontal and
vertical flips, and transpose both for TTT examples and prediction voting. The
cheat run adds compatible pairs from `to-solve/ARC-GEN/tasks`; copied fallback
tasks and generated grids larger than 12x12 are excluded and counted in its
report. The derived cheat permutations are deterministically capped at 256 per
task while all compatible raw pairs remain in its pool; set
`CHEAT_MAX_TTT_EXAMPLES=0` to enumerate them all. The three raw reports and
`comparison.json` are written under `RESULTS_DIR`. Use `MAX_TASKS=1
TTT_EPOCHS=1` for a quick smoke test.

To run the old direct-only checkpoint instead, explicitly set
`CHECKPOINT_PATH="$HOME/arc-checkpoints/mini-arc-v12-full/best.pt"` and
`REFINEMENT_ROUNDS=0`.

On an ARM64 CUDA cluster without Apptainer, use
`scripts/eval_mini_arc_v12_ttt_comparison_pcad.sh`. It stages all transient
inputs beneath the required `$SCRATCH` directory and runs a CUDA-enabled native
Python environment selected with `PYTHON_BIN`.
