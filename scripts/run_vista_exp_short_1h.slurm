#!/bin/bash
# Short ~1h batch: exp5 → exp3 → exp2 → exp1 (no exp4).
# Smallest valid ROI per experiment (pack roi_coords + index.json):
#   exp1: 11513410 L CC+MLO (smallest both-view boxes, 108k / 112k)
#   exp2: 62877247 L CC ↔ R CC (only bilateral ROI pair; 190k)
#   exp3: 90615275 → 11513410 L CC + L MLO (smallest matching two-patient pair)
#   exp5: 11513410 L MLO, full 8-exam chain (smallest ROI, 3+ exams)
#
#SBATCH -J semcorre_short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#SBATCH -o logs/semcorre_short_%j.out
#SBATCH -e logs/semcorre_short_%j.err
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
mkdir -p "$SCRATCH"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache_semcorre}"
export SEMCRE_OUT="${SEMCRE_OUT:-$SCRATCH/semcorre_batch_outputs}"
mkdir -p "$HF_HOME" "$SEMCRE_OUT/batch_experiments"
export PYTHONUNBUFFERED=1

echo "=== Job start ==="
echo "Host: $(hostname)  Date: $(date -u)"
echo "VENV_DIR=$VENV_DIR  HF_HOME=$HF_HOME  SEMCRE_OUT=$SEMCRE_OUT"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  export PATH="$VENV_DIR/bin:$PATH"
else
  echo "ERROR: invalid venv: $VENV_DIR"
  exit 1
fi

RUN_TAG="${RUN_TAG:-vista_short_1h}"
PACK="${PACK:-$PROJECT/data/roi_overlays_cancer5}"

echo "GPU on compute node:"
nvidia-smi -L || { echo "ERROR: no GPU — use #SBATCH -p gh"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$SEMCRE_OUT/batch_experiments" \
  --run-tag "$RUN_TAG" \
  --run-tag-exact \
  --skip-existing \
  --experiments exp5,exp3,exp2,exp1 \
  --exp1-patient 11513410 \
  --exp2-patient 62877247 \
  --anchor-patient 11513410 \
  --exp3-source-patient 90615275 \
  --exp3-max-sources 2 \
  --exp5-views 11513410:L:MLO \
  --exp5-max-steps 0 \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --device cuda:0

echo "Done: $SEMCRE_OUT/batch_experiments/$RUN_TAG/"
echo "Download: scp -r vista:$SEMCRE_OUT/batch_experiments/$RUN_TAG ./"
