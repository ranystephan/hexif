#!/usr/bin/env bash
# Submit CPU input validation and a dependent GPU smoke or production job.

set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s /absolute/private/path/run.env\n' "$0" >&2
    exit 2
fi
run_env="$1"
[[ "$run_env" = /* ]] || { printf 'run environment path must be absolute\n' >&2; exit 2; }
[[ -r "$run_env" ]] || { printf 'run environment is not readable: %s\n' "$run_env" >&2; exit 2; }
# shellcheck disable=SC1090
source "$run_env"
: "${HEXIF_REPO:?required in run environment}"
: "${HEXIF_OUTPUT_DIR:?required in run environment}"
cd "$HEXIF_REPO"
[[ -z "$(git status --porcelain)" ]] || { printf 'repository is dirty\n' >&2; exit 1; }
[[ ! -e "$HEXIF_OUTPUT_DIR" ]] || {
    printf 'output path already exists; use a new run directory: %s\n' "$HEXIF_OUTPUT_DIR" >&2
    exit 1
}
mkdir -p "$HEXIF_OUTPUT_DIR"

preflight_id="$(sbatch --parsable \
    --output="$HEXIF_OUTPUT_DIR/preflight-slurm-%j.out" \
    --error="$HEXIF_OUTPUT_DIR/preflight-slurm-%j.err" \
    --export=ALL,HEXIF_RUN_ENV="$run_env" \
    slurm/preflight_cell_training.sbatch)"
if ! training_id="$(sbatch --parsable \
    --dependency="afterok:$preflight_id" \
    --output="$HEXIF_OUTPUT_DIR/training-slurm-%j.out" \
    --error="$HEXIF_OUTPUT_DIR/training-slurm-%j.err" \
    --export=ALL,HEXIF_RUN_ENV="$run_env" \
    slurm/train_cell_phenotype.sbatch)"; then
    scancel "$preflight_id"
    printf 'GPU submission failed; canceled preflight job %s\n' "$preflight_id" >&2
    exit 1
fi
printf 'preflight_job_id=%s\n' "$preflight_id"
printf 'training_job_id=%s\n' "$training_id"
printf 'The GPU job can run only after successful real-input validation.\n'
