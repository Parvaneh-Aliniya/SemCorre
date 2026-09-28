#!/bin/bash
# Exp5 ROI-box GT only (same chains as run_vista_exp5_temporal.slurm).
# Writes two trees (does not overwrite the Gaussian run):
#   vista_exp5_temporal_transfer_roi_box
#   vista_exp5_temporal_own_roi_roi_box
#
#SBATCH -J semcorre_e5r
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -o logs/semcorre_e5_temporal_roi_%j.out
#SBATCH -e logs/semcorre_e5_temporal_roi_%j.err
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

PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_exp5_temporal}"
SET_JSON="${SET_JSON:-$PROJECT/data/exp_sets/exp5_temporal.json}"
test -f "$SET_JSON" || { echo "ERROR: missing $SET_JSON"; exit 1; }

echo "=== Job start === $(hostname) $(date -u)"
echo "PACK=$PACK  SET_JSON=$SET_JSON  gt_mode=roi_box"

if [[ ! -f "$PACK/roi_coords.csv" ]]; then
  echo "Building exp5 temporal pack (all exams of tagged patients)..."
  mapfile -t PIDS < <(python -c "import json; print('\\n'.join(sorted(set(json.load(open('$SET_JSON'))['patient_ids']))))")
  args=()
  for pid in "${PIDS[@]}"; do
    args+=(--patient "$pid")
  done
  python -u "$SKP/draw_embed_roi_overlays.py" \
    --dataset-loc "$SCRATCH/sk_png_dataset_full" \
    --out "$PACK" \
    --keep-all-rois \
    "${args[@]}"
fi
test -f "$PACK/roi_coords.csv" || { echo "ERROR: no $PACK/roi_coords.csv"; exit 1; }

nvidia-smi -L || { echo "ERROR: no GPU — use #SBATCH -p gh"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

run_one() {
  local tag="$1"
  local mode="$2"
  python -u scripts/batch_mammo_correspondence.py \
    --pack-dir "$PACK" \
    --out-dir "$SEMCRE_OUT/batch_experiments" \
    --run-tag "$tag" \
    --run-tag-exact \
    --skip-existing \
    --experiments exp5 \
    --set-json "$SET_JSON" \
    --gt-mode roi_box \
    --exp5-source-mode "$mode" \
    --exp5-max-steps 0 \
    --num_opt_iterations 3 \
    --num_iterations 10 \
    --with-tps-warp \
    --tps-modes roi \
    --no-semcorre-after-tps \
    --device cuda:0
  echo "Done: $SEMCRE_OUT/batch_experiments/$tag/"
}

run_one "${RUN_TAG_TRANSFER:-vista_exp5_temporal_transfer_roi_box}" transfer
run_one "${RUN_TAG_OWNROI:-vista_exp5_temporal_own_roi_roi_box}" own_roi
