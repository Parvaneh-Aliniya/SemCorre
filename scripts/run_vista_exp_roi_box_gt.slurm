#!/bin/bash
# 3-pair smoke test: binary ROI-box GT instead of Gaussian around the center.
#   exp1 58936031 R CC↔MLO (2 pairs)
#   exp5 58936031 R CC, 1 hop (1 pair)
#
#SBATCH -J semcorre_roi_box
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o logs/semcorre_roi_box_%j.out
#SBATCH -e logs/semcorre_roi_box_%j.err
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
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache_semcorre}"
export SEMCRE_OUT="${SEMCRE_OUT:-$SCRATCH/semcorre_batch_outputs}"
mkdir -p "$HF_HOME" "$SEMCRE_OUT/batch_experiments"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  export PATH="$VENV_DIR/bin:$PATH"
fi

PACK="${PACK:-/scratch/11364/paliniya/sk_review/roi_overlays_small2}"
RUN_TAG="${RUN_TAG:-vista_roi_box_gt}"

echo "=== Job start === $(hostname) $(date -u)"
echo "PACK=$PACK  RUN_TAG=$RUN_TAG  gt_mode=roi_box"
nvidia-smi -L
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$SEMCRE_OUT/batch_experiments" \
  --run-tag "$RUN_TAG" \
  --run-tag-exact \
  --experiments exp1,exp5 \
  --exp12-patient 58936031 \
  --exp1-patient 58936031 \
  --exp5-views 58936031:R:CC \
  --exp5-max-steps 1 \
  --gt-compare \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --device cuda:0

echo "Done: $SEMCRE_OUT/batch_experiments/$RUN_TAG/"
