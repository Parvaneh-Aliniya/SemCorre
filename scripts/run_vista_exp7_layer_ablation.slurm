#!/bin/bash
# Exp7: layer ablation on Exp1 CC↔MLO pairs — wide stack (2–10) vs production baseline (7–10).
# Two run tags under batch_experiments/ for side-by-side metrics.
#SBATCH -J semcorre_e7
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 16:00:00
#SBATCH -o logs/semcorre_e7_ablation_%j.out
#SBATCH -e logs/semcorre_e7_ablation_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

SK_ENV_DEFAULT="${SK_ENV:-/scratch/11364/paliniya/stablekeypoints/sk_env}"
if [[ -z "${SEMCRE_VENV:-}" && -f "$PROJECT/.semcorre_venv_path" ]]; then
  SEMCRE_VENV=$(tr -d '\r\n' < "$PROJECT/.semcorre_venv_path")
fi
if [[ -z "${SEMCRE_VENV:-}" || ! -x "${SEMCRE_VENV}/bin/python" ]]; then
  if [[ -x "$SK_ENV_DEFAULT/bin/python" ]]; then
    SEMCRE_VENV="$SK_ENV_DEFAULT"
  elif [[ -x "$PROJECT/venv/bin/python" ]]; then
    SEMCRE_VENV="$PROJECT/venv"
  fi
fi
export SEMCRE_VENV
VENV_DIR="${SEMCRE_VENV:-$SK_ENV_DEFAULT}"

module load gcc cuda 2>/dev/null || true
module load python3 2>/dev/null || true

export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
export EMBED_DATA_DIR="${EMBED_DATA_DIR:-$SCRATCH/embed_dataset/tables}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache_semcorre}"
export SEMCRE_OUT="${SEMCRE_OUT:-$SCRATCH/semcorre_batch_outputs}"
mkdir -p "$HF_HOME" "$SEMCRE_OUT/batch_experiments"
export PYTHONUNBUFFERED=1

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  export PATH="$VENV_DIR/bin:$PATH"
else
  echo "ERROR: invalid venv: $VENV_DIR"
  exit 1
fi

unset PACK
PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_exp1_views}"
SET_JSON="${SET_JSON:-$PROJECT/data/exp_sets/exp7_layer_ablation.json}"
test -f "$SET_JSON" || { echo "ERROR: missing $SET_JSON"; exit 1; }
test -f "$PACK/roi_coords.csv" || { echo "ERROR: missing exp1 views pack"; exit 1; }

COMMON=(
  --pack-dir "$PACK"
  --out-dir "$SEMCRE_OUT/batch_experiments"
  --run-tag-exact
  --skip-existing
  --experiments exp7
  --set-json "$SET_JSON"
  --gt-compare
  --num_opt_iterations 3
  --num_iterations 10
  --with-tps-warp
  --tps-modes roi
  --no-semcorre-after-tps
  --device cuda:0
)

run_one() {
  local tag=$1
  shift
  echo "=== Exp7 run_tag=$tag layers=$* === $(date -u)"
  python -u scripts/batch_mammo_correspondence.py \
    "${COMMON[@]}" \
    --run-tag "$tag" \
    --layers "$@"
}

nvidia-smi -L || { echo "ERROR: no GPU"; exit 1; }

echo "=== Exp7 layer ablation job start === $(hostname) $(date -u)"
echo "PACK=$PACK  SET_JSON=$SET_JSON"

run_one vista_exp7_layers_2_10 2 3 4 5 6 7 8 9 10
run_one vista_exp7_layers_7_10 7 8 9 10

echo "Done:"
echo "  $SEMCRE_OUT/batch_experiments/vista_exp7_layers_2_10/"
echo "  $SEMCRE_OUT/batch_experiments/vista_exp7_layers_7_10/"
