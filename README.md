# mini-arc-v12 — UFRGS class project

## Credits and project context

This repository is based on the original **Mini-ARC** project created by
**Paul Fletcher-Hill** for the ARC Prize. The original code is available at
[pfletcherhill/mini-arc](https://github.com/pfletcherhill/mini-arc), and the
paper is reproduced in [`paper/mini-arc.tex`](paper/mini-arc.tex) and published
as [mini-arc.pdf](https://www.paulfletcherhill.com/mini-arc.pdf).

The work in this repository is a class project at the **Federal University of
Rio Grande do Sul (UFRGS)**. It independently trains Mini-ARC-v12 on a reduced
dataset and evaluates a specific intervention: stronger data augmentation
during test-time training and inference.

## Research hypothesis

The original paper asks whether a small, ARC-specific visual Transformer can
solve abstraction and reasoning puzzles without a language model, search, or
program synthesis. Its proposed system combines three ingredients:

1. a 67-million-parameter Transformer trained only on ARC-like grids;
2. test-time training (TTT), which temporarily adapts a copy of the model to
   the demonstrations of each test puzzle;
3. refinement, which feeds a predicted output back to the adapted model for
   two additional correction passes.

The hypothesis tested by this class project is narrower:

> If ARC transformations are invariant to geometry, colour relabelling, and
> demonstration order, exposing those equivalent views during TTT and
> aggregating their predictions should improve performance over standard TTT
> on the same Mini-ARC-v12 checkpoint.

The comparison also includes a deliberately privileged `cheat` condition. It
adds generated input/output pairs associated with each evaluation task and is
used as an upper bound on the value of better task-specific data. It is not a
fair generalization result.

## Experimental setup

All three scenarios use the same `mini-arc-v12-full-refinement` checkpoint,
the same 114 puzzles used by the paper, the first four demonstrations, the
first test query, the same seed, and the same TTT hyperparameters. TTT runs for
up to 15 epochs with a 99.5% adaptation-accuracy cutoff; every final prediction
then receives two refinement rounds.

| Scenario | Task-specific training data and inference |
|---|---|
| **Baseline** | Original demonstrations and identity view only. |
| **Data augmentation** | Up to 256 identity-anchored TTT items per task, sampled from all eight D4 symmetries, seeded colour permutations, and demonstration orders. At inference, transformed predictions are mapped back to canonical space and combined by hierarchical voting. |
| **Cheat upper bound** | Baseline procedure plus compatible ARC-GEN pairs for the evaluation tasks, capped at 256 derived TTT items per task. |

The baseline used 2,952 derived TTT items in total. Data augmentation used
24,512 items from 94,912 candidates before the per-task cap. The cheat run
loaded 1,063 extra pairs, rejected 17 incompatible pairs, and used 27,818
derived items from 2,308,860 candidates before capping.

### Metrics from the paper

- **Score** is the number of puzzles whose complete output grid is exactly
  correct. This is the primary measure of solved tasks.
- **Accuracy** is cell accuracy over the centered, padded 12×12 output. Padding
  makes this number optimistic for small grids, so it must not be read as
  puzzle-solving accuracy.
- **Closeness** is the number of puzzles with at least 95% cell accuracy.

## Results

The comparison report selects the post-refinement prediction as the final
stage, matching the paper's **TTT + Refined** setting.

| Scenario | Score | Accuracy | Closeness |
|---|---:|---:|---:|
| **Baseline** | 5/114 (4.39%) | 89.80% | 40/114 (35.09%) |
| **Data augmentation** | **11/114 (9.65%)** | **91.40%** | **44/114 (38.60%)** |
| **Cheat upper bound** | 17/114 (14.91%) | 92.90% | 62/114 (54.39%) |

Relative to the baseline, data augmentation gains 6 exact solutions, 1.60
percentage points of cell accuracy, and 4 close solutions. It fixes 8 tasks
that the baseline misses and loses 2 that the baseline solves, for a net gain
of 6. This is a 2.2× increase in Score (11 versus 5) under the controlled
comparison.

The cheat condition gains 12 exact solutions over the baseline, but its 1,063
task-associated extra examples give it information unavailable to the other
runs. It is evidence that the model benefits from stronger task-specific
supervision, not an unbiased estimate of performance on unseen ARC tasks.

Refinement does not help every scenario:

| Scenario | TTT Score | TTT + Refined Score | Change |
|---|---:|---:|---:|
| Baseline | 5 | 5 | 0 |
| Data augmentation | 9 | 11 | +2 |
| Cheat upper bound | 20 | 17 | -3 |

Thus, the augmentation result supports the project's intervention, while the
cheat result also shows that refinement can overwrite correct TTT outputs. A
future evaluation should treat the number of refinement rounds as a validation
choice rather than assuming that two rounds always improve Score.

The complete machine-readable result is in
[`results/comparison.json`](results/comparison.json).

## Why our baseline is below the paper's 17.5%

The paper reports **20/114 (17.5%)** for Mini-ARC-v12 with TTT and refinement;
our independently trained baseline obtains **5/114 (4.39%)**. The two numbers
use the same metric and evaluation subset, but they do not come from the same
pretrained model or training corpus.

The largest documented reproduction gap is the training data:

| | Original Mini-ARC-v12 | This project |
|---|---:|---:|
| Training examples | 830,648 | 186,556 |
| Data sources | RE-ARC, BARC Heavy, ARC-HTML | Reduced RE-ARC only |
| Training diversity | RE-ARC patterns plus BARC and ARC-HTML generators | 391 eligible RE-ARC generator families |

The local corpus is about 4.5× smaller and, more importantly, omits the BARC
Heavy and ARC-HTML sources. The original paper trained on 4–8 A100 GPUs over
multiple days for at least 150,000 steps with varying effective batch sizes.
Our full-refinement launcher is configured for 150,000 steps on eight GPUs,
but `comparison.json` does not record the selected checkpoint's global step or
training history. A shorter realized run, different best-checkpoint selection,
optimizer trajectory, data balancing, and random initialization may therefore
also contribute, but the current report cannot quantify them.

There are further implementation differences. This project rebuilds the data
pipeline with family-balanced sampling and held-out examples per generator,
whereas the paper describes a different mixed synthetic dataset and split.
These choices improve traceability but change the training distribution.
Finally, Accuracy includes easy-to-predict padding cells; Score is the safer
number for comparing actual solutions.

The lower absolute baseline does **not** invalidate the augmentation
intervention. The intervention is evaluated as a paired comparison: checkpoint,
tasks, seed, TTT schedule, query selection, and refinement count are held
constant, while the augmentation policy changes. It supports the conclusion
that strong augmentation improves this checkpoint on this 114-task subset. It
does not establish that the repository reproduces the paper's absolute 17.5%,
nor that the same gain will necessarily transfer to another checkpoint or the
full ARC benchmark.

## Model and implementation

The evaluated profile keeps the original Mini-ARC-v12 scale:

- 12×12 input and output grids;
- four demonstration pairs and one query;
- 2×2 patch embeddings;
- 16 encoder layers, 16 attention heads, `d_model=512`, and `d_ff=3072`;
- 67,343,755 parameters;
- noisy partial targets on 25% of pretraining steps, enabling refinement.

`arc_prize/eval_arc_agi.py` creates and discards a separately adapted model copy
for every puzzle, so evaluation never changes the base checkpoint. The direct
`full` profile has an untrained refinement input and must be evaluated with
`REFINEMENT_ROUNDS=0`; use the independently trained `full-refinement` profile
for the experiment reported above.

## Reproducibility

The detailed cluster guide is in
[`TRAINING_MINI_ARC_V12.md`](TRAINING_MINI_ARC_V12.md). The short workflow is:

1. Prepare the reduced RE-ARC dataset:

   ```bash
   python3 -m arc_prize.rearc_manifest \
     --source ../re-arc/re_arc_5k_12x12 \
     --output data/re_arc_5k_12x12_balanced
   ```

   The current source should retain 186,556 examples across 391 families and
   produce `manifest.json` plus `examples.sqlite3`.

2. Build the reproducible CUDA/Apptainer image:

   ```bash
   ./scripts/build_mini_arc_v12_container.sh \
     ./mini-arc-v12-pytorch2.4.1-cuda12.1.sif
   ```

3. Train or resume the full model with its refinement branch:

   ```bash
   ./scripts/train_mini_arc_v12_full_refinement_oar.sh
   ```

   By default this runs 150 epochs of 1,000 steps, uses refinement targets on
   25% of steps, and stores resumable `latest.pt` and validation-selected
   `best.pt` checkpoints under
   `$HOME/arc-checkpoints/mini-arc-v12-full-refinement`.

4. Reproduce the three-scenario comparison on Grid'5000:

   ```bash
   RESULTS_DIR="$HOME/arc-results/mini-arc-v12-full-refinement-ttt-comparison" \
   ./scripts/eval_mini_arc_v12_ttt_comparison_oar.sh
   ```

   On the ARM64 PCAD environment, use
   `scripts/eval_mini_arc_v12_ttt_comparison_pcad.sh` and set `PYTHON_BIN` and
   `SCRATCH` as required by that host. A quick smoke test can set
   `MAX_TASKS=1 TTT_EPOCHS=1`.

The evaluation writes `baseline.json`, `augmentation.json`, `cheat.json`, and
the consolidated `comparison.json` beneath `RESULTS_DIR`. Keep the checkpoint,
dataset fingerprint, seed, and all evaluation environment variables with any
new result; `comparison.json` alone does not capture the full training
provenance.
