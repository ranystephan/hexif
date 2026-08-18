#!/usr/bin/env bash
#SBATCH --job-name=hexif-cell-preflight
#SBATCH --account=andrewg
#SBATCH --partition=owners
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -n "${HEXIF_RUN_ENV:-}" ]] || fail "HEXIF_RUN_ENV is unset"
[[ -r "$HEXIF_RUN_ENV" ]] || fail "run environment is not readable: $HEXIF_RUN_ENV"
# shellcheck disable=SC1090
source "$HEXIF_RUN_ENV"
for name in HEXIF_REPO HEXIF_PYTHON HEXIF_TRAIN_TABLE HEXIF_VAL_TABLE \
    HEXIF_PAIRS_DIR HEXIF_OUTPUT_DIR HEXIF_LABEL_SET; do
    [[ -n "${!name:-}" ]] || fail "required variable $name is unset"
done
if [[ -n "${HEXIF_PYTHON_MODULE:-}" ]]; then
    module load "$HEXIF_PYTHON_MODULE"
fi
[[ -x "$HEXIF_PYTHON" ]] || fail "Python is not executable: $HEXIF_PYTHON"
cd "$HEXIF_REPO"
[[ -z "$(git status --porcelain)" ]] || fail "repository is dirty"
commit="$(git rev-parse HEAD)"
if [[ -n "${HEXIF_COMMIT:-}" && "$commit" != "$HEXIF_COMMIT" ]]; then
    fail "checked-out commit $commit does not match HEXIF_COMMIT=$HEXIF_COMMIT"
fi
mkdir -p "$HEXIF_OUTPUT_DIR"
exec > >(tee -a "$HEXIF_OUTPUT_DIR/preflight.out") \
    2> >(tee -a "$HEXIF_OUTPUT_DIR/preflight.err" >&2)
export PYTHONPATH="$HEXIF_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$HEXIF_PYTHON" scripts/validate_cell_training_inputs.py \
    --train-table "$HEXIF_TRAIN_TABLE" \
    --val-table "$HEXIF_VAL_TABLE" \
    --pairs-dir "$HEXIF_PAIRS_DIR" \
    --label-set "$HEXIF_LABEL_SET" \
    --output "$HEXIF_OUTPUT_DIR/input_validation.json"
