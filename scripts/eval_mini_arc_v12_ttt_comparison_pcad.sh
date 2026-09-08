#!/usr/bin/env bash
# Compare baseline TTT, strong augmentation, and ARC-GEN extra-data TTT on
# an ARM64 CUDA host using a native Python environment (no Apptainer required).
set -euo pipefail

ARC_WORKSPACE="${ARC_WORKSPACE:-$HOME/arc-agi}"
MINI_ARC_SOURCE="${MINI_ARC_SOURCE:-$ARC_WORKSPACE/mini-arc}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/arc-checkpoints/mini-arc-v12-full-refinement/best.pt}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/arc-results/mini-arc-v12-full-refinement-ttt-comparison}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHALLENGES_PATH="${CHALLENGES_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_challenges.json}"
SOLUTIONS_PATH="${SOLUTIONS_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_solutions.json}"
TASK_IDS_PATH="${TASK_IDS_PATH:-$MINI_ARC_SOURCE/data/mini_arc_v12_evaluation_ids.txt}"
EXTRA_EXAMPLES_DIR="${EXTRA_EXAMPLES_DIR:-$MINI_ARC_SOURCE/to-solve/ARC-GEN/tasks}"
SCRATCH_BASE="${SCRATCH:?SCRATCH must point to PCAD scratch storage}"
JOB_ID="${SLURM_JOB_ID:-${PBS_JOBID:-manual-$$}}"
JOB_ROOT="$SCRATCH_BASE/mini-arc-v12-ttt-comparison-$JOB_ID"
LOCAL_CHECKPOINT="$JOB_ROOT/checkpoint.pt"
LOCAL_RESULTS="$JOB_ROOT/results"

case "$JOB_ROOT" in
    "$SCRATCH_BASE"/mini-arc-v12-ttt-comparison-*) ;;
    *) echo "Refusing unsafe job directory: $JOB_ROOT" >&2; exit 2 ;;
esac

required_files=(
    "$MINI_ARC_SOURCE/arc_prize/eval_arc_agi.py"
    "$MINI_ARC_SOURCE/arc_prize/compare_ttt_runs.py"
    "$MINI_ARC_SOURCE/models/data_augmentation_baseline/strong_augmentation.py"
    "$CHECKPOINT_PATH"
    "$CHALLENGES_PATH"
    "$SOLUTIONS_PATH"
    "$TASK_IDS_PATH"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 2
    fi
done
if [[ ! -d "$EXTRA_EXAMPLES_DIR" ]]; then
    echo "Extra examples directory not found: $EXTRA_EXAMPLES_DIR" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 2
fi

GPU_COUNT="$("$PYTHON_BIN" -c 'import numpy; import platform, torch; print(torch.cuda.device_count()); assert platform.machine() == "aarch64", platform.machine(); assert torch.cuda.is_available(), "CUDA is unavailable"')"
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "No CUDA GPU is visible to $PYTHON_BIN" >&2
    exit 3
fi

mkdir -p "$JOB_ROOT/mini-arc" "$JOB_ROOT/data/extra-examples" "$LOCAL_RESULTS" "$RESULTS_DIR"
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='data/re_arc_5k_12x12/' \
    --exclude='data/re_arc_5k_12x12_balanced/' \
    --exclude='to-solve/' \
    "$MINI_ARC_SOURCE/" "$JOB_ROOT/mini-arc/"
rsync -a "$CHECKPOINT_PATH" "$LOCAL_CHECKPOINT"
rsync -a "$CHALLENGES_PATH" "$JOB_ROOT/data/challenges.json"
rsync -a "$SOLUTIONS_PATH" "$JOB_ROOT/data/solutions.json"
rsync -a "$TASK_IDS_PATH" "$JOB_ROOT/data/task_ids.txt"
rsync -a "$EXTRA_EXAMPLES_DIR/" "$JOB_ROOT/data/extra-examples/"

echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Scratch staging: $JOB_ROOT"
echo "Python: $PYTHON_BIN"
echo "Visible GPUs: $GPU_COUNT (the comparison uses cuda:0 only)"
"$PYTHON_BIN" -c 'import platform, torch; print("architecture:", platform.machine()); print("torch:", torch.__version__); print("GPU:", torch.cuda.get_device_name(0))'
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

cd "$JOB_ROOT/mini-arc"
COMMON_ARGS=(
    --checkpoint "$LOCAL_CHECKPOINT"
    --challenges "$JOB_ROOT/data/challenges.json"
    --solutions "$JOB_ROOT/data/solutions.json"
    --task-ids-file "$JOB_ROOT/data/task_ids.txt"
    --first-query-only
    --ttt-epochs "${TTT_EPOCHS:-15}"
    --ttt-learning-rate "${TTT_LEARNING_RATE:-0.00001}"
    --ttt-weight-decay "${TTT_WEIGHT_DECAY:-0.00001}"
    --ttt-batch-size "${TTT_BATCH_SIZE:-4}"
    --ttt-accuracy-cutoff "${TTT_ACCURACY_CUTOFF:-0.995}"
    --refinement-rounds "${REFINEMENT_ROUNDS:-2}"
    --seed "${SEED:-42}"
)
if [[ -n "${MAX_TASKS:-}" ]]; then
    COMMON_ARGS+=(--max-tasks "$MAX_TASKS")
fi

echo "Run 1/3: baseline TTT"
"$PYTHON_BIN" -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    --augmentation-transforms identity \
    --output "$LOCAL_RESULTS/baseline.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

echo "Run 2/3: strong D4 + colour + order augmentation"
"$PYTHON_BIN" -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    --strong-augmentation \
    --strong-ttt-max-examples "${STRONG_TTT_MAX_EXAMPLES:-256}" \
    --strong-training-color-permutations "${STRONG_TRAIN_COLOR_PERMUTATIONS:-4}" \
    --strong-inference-color-permutations "${STRONG_INFERENCE_COLOR_PERMUTATIONS:-2}" \
    --strong-inference-orders "${STRONG_INFERENCE_ORDERS:-2}" \
    --strong-identity-fraction "${STRONG_IDENTITY_FRACTION:-0.25}" \
    --output "$LOCAL_RESULTS/augmentation.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

echo "Run 3/3: ARC-GEN extra-example upper bound"
CHEAT_ARGS=(
    --augmentation-transforms identity
    --ttt-extra-examples-dir "$JOB_ROOT/data/extra-examples"
)
if [[ "${CHEAT_MAX_TTT_EXAMPLES:-256}" != "0" ]]; then
    CHEAT_ARGS+=(--ttt-max-examples "${CHEAT_MAX_TTT_EXAMPLES:-256}")
fi
"$PYTHON_BIN" -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    "${CHEAT_ARGS[@]}" \
    --output "$LOCAL_RESULTS/cheat.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

"$PYTHON_BIN" -m arc_prize.compare_ttt_runs \
    --baseline "$LOCAL_RESULTS/baseline.json" \
    --augmentation "$LOCAL_RESULTS/augmentation.json" \
    --cheat "$LOCAL_RESULTS/cheat.json" \
    --output "$LOCAL_RESULTS/comparison.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

if [[ "${KEEP_SCRATCH:-0}" != "1" ]]; then
    rm -rf -- "$JOB_ROOT"
fi
echo "Comparison finished; reports saved under: $RESULTS_DIR"
