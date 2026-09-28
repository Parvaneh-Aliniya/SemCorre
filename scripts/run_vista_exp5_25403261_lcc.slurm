#!/bin/bash
# One exp5 chain: 25403261 left CC, full exam timeline.
# Pack is built on the node if roi_coords.csv is missing (do not use --roi-only).
#
#SBATCH -J semcorre_e5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o logs/semcorre_e5_25403261_%j.out
#SBATCH -e logs/semcorre_e5_25403261_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
SKP=/home1/11364/paliniya/projects/StableKeypointsPlus
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

PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_new5}"
RUN_TAG="${RUN_TAG:-vista_exp5_25403261_lcc}"

echo "=== Job start === $(hostname) $(date -u)"
echo "PACK=$PACK  RUN_TAG=$RUN_TAG  VENV_DIR=$VENV_DIR"

if [[ ! -f "$PACK/roi_coords.csv" ]]; then
  echo "Building new5 pack (all exams, keep every ROI)..."
  python -u "$SKP/draw_embed_roi_overlays.py" \
    --dataset-loc "$SCRATCH/sk_png_dataset_full" \
    --out "$PACK" \
    --patient 25403261 \
    --patient 25468247 \
    --patient 35223111 \
    --patient 51147121 \
    --patient 65041852 \
    --keep-all-rois
fi
test -f "$PACK/roi_coords.csv" || { echo "ERROR: no $PACK/roi_coords.csv"; exit 1; }
echo "Pack L CC ROI rows:"
python -c "
import pandas as pd
df=pd.read_csv('$PACK/roi_coords.csv', dtype=str)
m=(df['patient_id'].astype(str)=='25403261') & (df['laterality'].str.upper().str[:1]=='L') & (df['view'].str.upper()=='CC')
print(df.loc[m, ['exam_date','laterality','view','image_path']].to_string(index=False))
"

nvidia-smi -L || { echo "ERROR: no GPU — use #SBATCH -p gh"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$SEMCRE_OUT/batch_experiments" \
  --run-tag "$RUN_TAG" \
  --run-tag-exact \
  --skip-existing \
  --experiments exp5 \
  --exp5-views 25403261:L:CC \
  --gt-compare \
  --exp5-max-steps 0 \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --device cuda:0

echo "Done: $SEMCRE_OUT/batch_experiments/$RUN_TAG/"
