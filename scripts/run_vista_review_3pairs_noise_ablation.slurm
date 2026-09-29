#!/bin/bash
# Exp8: same 3 review CC↔MLO pairs; noise_level 4 vs 8 (baseline) vs 12; layers 7–10.
#
#SBATCH -J semcorre_r3_noise
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/semcorre_review_3pairs_noise_%j.out
#SBATCH -e logs/semcorre_review_3pairs_noise_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh

set -euo pipefail

case "$(hostname -s)" in
  login*) echo "ERROR: use sbatch, not bash on $(hostname). Example: sbatch $0"; exit 1 ;;
esac

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs
echo "=== Exp8 noise ablation | $(hostname -s) $(date -u) ==="

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
SET_JSON="${SET_JSON:-$PROJECT/data/exp_sets/exp8_noise_ablation_3pairs.json}"
test -f "$SET_JSON" || { echo "ERROR: missing $SET_JSON (git pull)"; exit 1; }
test -f "$PACK/roi_coords.csv" || { echo "ERROR: missing $PACK/roi_coords.csv"; exit 1; }

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
  local tag=$1
  local nl=$2
  echo "=== run_tag=$tag noise_level=$nl === $(date -u)"
  python -u scripts/batch_mammo_correspondence.py \
    "${COMMON[@]}" \
    --run-tag "$tag" \
    --noise_level "$nl"
}

nvidia-smi -L || { echo "ERROR: no GPU — sbatch -p gh"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('cuda OK')"

run_noise vista_review_3pairs_noise_4 4
run_noise vista_review_3pairs_noise_8 8
run_noise vista_review_3pairs_noise_12 12

echo "Done:"
echo "  $SEMCRE_OUT/batch_experiments/vista_review_3pairs_noise_{4,8,12}/"
