#!/usr/bin/env bash
# Generic real-data entry point. Supply a private run environment with:
# sbatch --export=ALL,HEXIF_RUN_ENV=/absolute/path/run.env this-file
#SBATCH --job-name=hexif-cell-train
#SBATCH --account=andrewg
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_var() {
    local name="$1"
    [[ -n "${!name:-}" ]] || fail "required variable $name is unset"
}

require_var HEXIF_RUN_ENV
[[ -r "$HEXIF_RUN_ENV" ]] || fail "run environment is not readable: $HEXIF_RUN_ENV"
# shellcheck disable=SC1090
source "$HEXIF_RUN_ENV"

if [[ -n "${HEXIF_PYTHON_MODULE:-}" ]]; then
    module load "$HEXIF_PYTHON_MODULE"
fi
if [[ -n "${HEXIF_HF_TOKEN_PATH:-}" ]]; then
    [[ -s "$HEXIF_HF_TOKEN_PATH" ]] || fail "Hugging Face token file is missing or empty"
    export HF_TOKEN_PATH="$HEXIF_HF_TOKEN_PATH"
fi
if [[ -n "${HEXIF_HF_HOME:-}" ]]; then
    mkdir -p "$HEXIF_HF_HOME"
    export HF_HOME="$HEXIF_HF_HOME"
fi

for name in HEXIF_REPO HEXIF_PYTHON HEXIF_TRAIN_TABLE HEXIF_VAL_TABLE \
    HEXIF_PAIRS_DIR HEXIF_OUTPUT_DIR HEXIF_TRAINER HEXIF_ENCODER HEXIF_LABEL_SET; do
    require_var "$name"
done

[[ "$HEXIF_REPO" = /* ]] || fail "HEXIF_REPO must be absolute"
[[ "$HEXIF_TRAIN_TABLE" = /* ]] || fail "HEXIF_TRAIN_TABLE must be absolute"
[[ "$HEXIF_VAL_TABLE" = /* ]] || fail "HEXIF_VAL_TABLE must be absolute"
[[ "$HEXIF_PAIRS_DIR" = /* ]] || fail "HEXIF_PAIRS_DIR must be absolute"
[[ "$HEXIF_PYTHON" = /* ]] || fail "HEXIF_PYTHON must be absolute"
[[ "$HEXIF_OUTPUT_DIR" = /* ]] || fail "HEXIF_OUTPUT_DIR must be absolute"
[[ -x "$HEXIF_PYTHON" ]] || fail "Python is not executable: $HEXIF_PYTHON"
[[ -f "$HEXIF_TRAIN_TABLE" ]] || fail "missing train table: $HEXIF_TRAIN_TABLE"
[[ -f "$HEXIF_VAL_TABLE" ]] || fail "missing validation table: $HEXIF_VAL_TABLE"
[[ -d "$HEXIF_PAIRS_DIR" ]] || fail "missing pairs directory: $HEXIF_PAIRS_DIR"
[[ "$HEXIF_TRAINER" = v4 || "$HEXIF_TRAINER" = v6 ]] || fail "trainer must be v4 or v6"
[[ "$HEXIF_LABEL_SET" = gmm || "$HEXIF_LABEL_SET" = consensus || "$HEXIF_LABEL_SET" = spacec ]] || \
    fail "label set must be gmm, consensus, or spacec"

cd "$HEXIF_REPO"
[[ -z "$(git status --porcelain)" ]] || fail "repository is dirty"
commit="$(git rev-parse HEAD)"
if [[ -n "${HEXIF_COMMIT:-}" && "$commit" != "$HEXIF_COMMIT" ]]; then
    fail "checked-out commit $commit does not match HEXIF_COMMIT=$HEXIF_COMMIT"
fi
case "$HEXIF_OUTPUT_DIR" in
    "$HEXIF_REPO"|"$HEXIF_REPO"/*) fail "outputs must be outside the Git checkout" ;;
esac

mkdir -p "$HEXIF_OUTPUT_DIR"
exec > >(tee -a "$HEXIF_OUTPUT_DIR/job.out") 2> >(tee -a "$HEXIF_OUTPUT_DIR/job.err" >&2)
printf 'run_started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
printf 'host=%s\n' "$(hostname -f)"
printf 'git_commit=%s\n' "$commit"
printf 'trainer=%s encoder=%s label_set=%s\n' \
    "$HEXIF_TRAINER" "$HEXIF_ENCODER" "$HEXIF_LABEL_SET"
"$HEXIF_PYTHON" --version
nvidia-smi

export PYTHONPATH="$HEXIF_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

preflight="$HEXIF_OUTPUT_DIR/input_validation.json"
"$HEXIF_PYTHON" scripts/validate_cell_training_inputs.py \
    --train-table "$HEXIF_TRAIN_TABLE" \
    --val-table "$HEXIF_VAL_TABLE" \
    --pairs-dir "$HEXIF_PAIRS_DIR" \
    --label-set "$HEXIF_LABEL_SET" \
    --output "$preflight"

gpu_log="$HEXIF_OUTPUT_DIR/gpu.csv"
nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,power.draw \
    --format=csv -l 30 > "$gpu_log" &
monitor_pid=$!
cleanup() {
    kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

trainer_script="experiments/train/train_cell_phenotype_${HEXIF_TRAINER}.py"
command=(
    "$HEXIF_PYTHON" "$trainer_script"
    --train_cell_table "$HEXIF_TRAIN_TABLE"
    --val_cell_table "$HEXIF_VAL_TABLE"
    --pairs_dir "$HEXIF_PAIRS_DIR"
    --output_dir "$HEXIF_OUTPUT_DIR/model"
    --label_set "$HEXIF_LABEL_SET"
    --encoder "$HEXIF_ENCODER"
    --seed "${HEXIF_SEED:-42}"
    --batch "${HEXIF_BATCH:-32}"
    --grad_accum_steps "${HEXIF_GRAD_ACCUM_STEPS:-4}"
    --num_workers "${HEXIF_NUM_WORKERS:-4}"
    --epochs "${HEXIF_EPOCHS:-30}"
    --early_stop_patience "${HEXIF_EARLY_STOP_PATIENCE:-5}"
    --max_train_cells_per_epoch "${HEXIF_MAX_TRAIN_CELLS_PER_EPOCH:-120000}"
)
if [[ "${HEXIF_SMOKE:-0}" = 1 ]]; then
    command+=(--smoke)
fi
if [[ "${HEXIF_RESTART:-0}" = 1 ]]; then
    command+=(--restart)
fi
if [[ -n "${HEXIF_EXTRA_ARGS:-}" ]]; then
    fail "HEXIF_EXTRA_ARGS is unsupported; add explicit reviewed variables to this script"
fi

printf 'command='
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"

checkpoint="$HEXIF_OUTPUT_DIR/model/best.pt"
metrics="$HEXIF_OUTPUT_DIR/model/metrics.json"
[[ -f "$checkpoint" ]] || fail "training completed without best.pt"
[[ -f "$metrics" ]] || fail "training completed without metrics.json"
"$HEXIF_PYTHON" scripts/capture_run_provenance.py \
    --output "$HEXIF_OUTPUT_DIR/provenance.json" \
    --command "$(printf '%q ' "${command[@]}")" \
    --artifact "train_table=$HEXIF_TRAIN_TABLE" \
    --artifact "val_table=$HEXIF_VAL_TABLE" \
    --artifact "input_validation=$preflight" \
    --artifact "checkpoint=$checkpoint" \
    --artifact "metrics=$metrics"
printf 'run_completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
