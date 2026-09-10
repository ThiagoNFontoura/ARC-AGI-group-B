#!/usr/bin/env bash
#OAR -n mini-arc-v12-ttt-comparison
#OAR -O mini-arc-v12-ttt-comparison-%jobid%.out
#OAR -E mini-arc-v12-ttt-comparison-%jobid%.err

# Compare standard TTT, strong augmentation, and ARC-GEN upper-bound data.
# Run this from an allocation with one modern CUDA-capable GPU.
set -euo pipefail

ARC_WORKSPACE="${ARC_WORKSPACE:-$HOME/arc-agi}"
MINI_ARC_SOURCE="${MINI_ARC_SOURCE:-$ARC_WORKSPACE/mini-arc}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif}"
# The full-refinement profile is the best available model. Its refinement path
# was trained, so two inference refinement rounds are valid and enabled below.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/arc-checkpoints/mini-arc-v12-full-refinement/best.pt}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/arc-results/mini-arc-v12-full-refinement-ttt-comparison}"
CHALLENGES_PATH="${CHALLENGES_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_challenges.json}"
SOLUTIONS_PATH="${SOLUTIONS_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_solutions.json}"
TASK_IDS_PATH="${TASK_IDS_PATH:-$MINI_ARC_SOURCE/data/mini_arc_v12_evaluation_ids.txt}"
EXTRA_EXAMPLES_DIR="${EXTRA_EXAMPLES_DIR:-$MINI_ARC_SOURCE/to-solve/ARC-GEN/tasks}"
JOB_ID="${OAR_JOB_ID:-manual-$$}"
JOB_ROOT="/tmp/$USER/mini-arc-v12-ttt-comparison-$JOB_ID"
LOCAL_IMAGE="$JOB_ROOT/mini-arc-v12.sif"
LOCAL_CHECKPOINT="$JOB_ROOT/checkpoint.pt"
LOCAL_RESULTS="$JOB_ROOT/results"

case "$JOB_ROOT" in
    "/tmp/$USER/mini-arc-v12-ttt-comparison-"*) ;;
    *) echo "Refusing unsafe job directory: $JOB_ROOT" >&2; exit 2 ;;
esac

required_files=(
    "$MINI_ARC_SOURCE/arc_prize/eval_arc_agi.py"
    "$MINI_ARC_SOURCE/arc_prize/compare_ttt_runs.py"
    "$MINI_ARC_SOURCE/models/data_augmentation_baseline/strong_augmentation.py"
    "$APPTAINER_IMAGE"
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
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on this node" >&2
    exit 2
fi

mkdir -p "$JOB_ROOT/mini-arc" "$JOB_ROOT/data/extra-examples" "$LOCAL_RESULTS" "$RESULTS_DIR"
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='data/re_arc_5k_12x12/' \
    --exclude='data/re_arc_5k_12x12_balanced/' \
    --exclude='to-solve/' \
    "$MINI_ARC_SOURCE/" "$JOB_ROOT/mini-arc/"
rsync -a "$APPTAINER_IMAGE" "$LOCAL_IMAGE"
rsync -a "$CHECKPOINT_PATH" "$LOCAL_CHECKPOINT"
rsync -a "$CHALLENGES_PATH" "$JOB_ROOT/data/challenges.json"
rsync -a "$SOLUTIONS_PATH" "$JOB_ROOT/data/solutions.json"
rsync -a "$TASK_IDS_PATH" "$JOB_ROOT/data/task_ids.txt"
rsync -a "$EXTRA_EXAMPLES_DIR/" "$JOB_ROOT/data/extra-examples/"

CONTAINER=(
    apptainer exec --nv
    --bind "$JOB_ROOT:$JOB_ROOT"
    --bind "$HOME:$HOME"
    "$LOCAL_IMAGE"
)

GPU_COUNT="$("${CONTAINER[@]}" python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "No CUDA device is visible inside the allocation" >&2
    exit 3
fi

echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Visible GPUs: $GPU_COUNT (the comparison uses cuda:0 only)"
echo "Checkpoint shared by all runs: $CHECKPOINT_PATH"
echo "Extra examples: $EXTRA_EXAMPLES_DIR"
nvidia-smi

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

echo "Run 1/3: baseline TTT (original examples, identity view)"
"${CONTAINER[@]}" python -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    --augmentation-transforms identity \
    --output "$LOCAL_RESULTS/baseline.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

echo "Run 2/3: strong D4 + colour + order augmentation"
"${CONTAINER[@]}" python -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    --strong-augmentation \
    --strong-ttt-max-examples "${STRONG_TTT_MAX_EXAMPLES:-256}" \
    --strong-training-color-permutations "${STRONG_TRAIN_COLOR_PERMUTATIONS:-4}" \
    --strong-inference-color-permutations "${STRONG_INFERENCE_COLOR_PERMUTATIONS:-2}" \
    --strong-inference-orders "${STRONG_INFERENCE_ORDERS:-2}" \
    --strong-identity-fraction "${STRONG_IDENTITY_FRACTION:-0.25}" \
    --output "$LOCAL_RESULTS/augmentation.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

echo "Run 3/3: cheat upper bound (original plus all compatible ARC-GEN examples)"
CHEAT_ARGS=(
    --augmentation-transforms identity
    --ttt-extra-examples-dir "$JOB_ROOT/data/extra-examples"
)
# All raw ARC-GEN pairs enter the pool. Limiting only their derived permutations
# keeps this upper-bound run tractable; set 0 to enumerate every permutation.
if [[ "${CHEAT_MAX_TTT_EXAMPLES:-256}" != "0" ]]; then
    CHEAT_ARGS+=(--ttt-max-examples "${CHEAT_MAX_TTT_EXAMPLES:-256}")
fi
"${CONTAINER[@]}" python -m arc_prize.eval_arc_agi \
    "${COMMON_ARGS[@]}" \
    "${CHEAT_ARGS[@]}" \
    --output "$LOCAL_RESULTS/cheat.json"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"

"${CONTAINER[@]}" python -m arc_prize.compare_ttt_runs \
    --baseline "$LOCAL_RESULTS/baseline.json" \
    --augmentation "$LOCAL_RESULTS/augmentation.json" \
    --cheat "$LOCAL_RESULTS/cheat.json" \
    --output "$LOCAL_RESULTS/comparison.json"

rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"
rm -rf -- "$JOB_ROOT"
echo "Comparison finished; reports saved under: $RESULTS_DIR"
