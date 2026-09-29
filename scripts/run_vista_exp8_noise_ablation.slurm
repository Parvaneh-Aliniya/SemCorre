#!/bin/bash
# Exp8 full Exp1 views: noise_level 4 vs 8 vs 12; layers 7–10.
#SBATCH -J semcorre_e8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/semcorre_e8_noise_%j.out
#SBATCH -e logs/semcorre_e8_noise_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh

set -euo pipefail

case "$(hostname -s)" in
  login*) echo "ERROR: use sbatch only"; exit 1 ;;
esac

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

SK_ENV_DEFAULT="${SK_ENV:-/scratch/11364/paliniya/stablekeypoints/sk_env}"
if [[ -z "${SEMCRE_VENV:-}" && -f "$PROJECT/.semcorre_venv_path" ]]; then
  SEMCRE_VENV=$(tr -d '\r\n' < "$PROJECT/.semcorre_venv_path")
fi
if [[ -z "${SEMCRE_VENV:-}" || ! -x "${SEMCRE_VENV}/bin/python" ]]; then
  SEMCRE_VENV="${SK_ENV_DEFAULT}"
fi
VENV_DIR="${SEMCRE_VENV:-$SK_ENV_DEFAULT}"

module load gcc cuda 2>/dev/null || true
module load python3 2>/dev/null || true

export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache_semcorre}"
export SEMCRE_OUT="${SEMCRE_OUT:-$SCRATCH/semcorre_batch_outputs}"
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate" 2>/dev/null || export PATH="$VENV_DIR/bin:$PATH"

unset PACK
PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_exp1_views}"
SET_JSON="$PROJECT/data/exp_sets/exp8_noise_ablation.json"

COMMON=(
  --pack-dir "$PACK"
  --out-dir "$SEMCRE_OUT/batch_experiments"
  --run-tag-exact
  --skip-existing
  --experiments exp1
  --set-json "$SET_JSON"
  --layers 7 8 9 10
  --gt-compare
  --num_opt_iterations 3
  --num_iterations 10
  --with-tps-warp
  --tps-modes roi
  --no-semcorre-after-tps
  --device cuda:0
)

run_noise() {
  python -u scripts/batch_mammo_correspondence.py \
    "${COMMON[@]}" --run-tag "$1" --noise_level "$2"
}

nvidia-smi -L
run_noise vista_exp8_noise_4 4
run_noise vista_exp8_noise_8 8
run_noise vista_exp8_noise_12 12
