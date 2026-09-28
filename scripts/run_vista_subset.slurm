#!/bin/bash
# Truncated SemCorre subset ONLY (not full exp1-5):
#   exp1+2: patient 11513410 | exp3: 3 sources, anchor 11513410 | exp4: off
#   exp5: 62877247:L:CC + 45209155:R:MLO | TPS on | 3x10 iters
#SBATCH -J semcorre_subset
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o logs/semcorre_%j.out
#SBATCH -e logs/semcorre_%j.err
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p YOUR_GPU_PARTITION

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

module load python3 cuda 2>/dev/null || true

source venv/bin/activate
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"
mkdir -p "$HF_HOME"

# Optional overrides (export before sbatch):
#   export EXP5_VIEWS="62877247:L:CC,45209155:R:MLO"
#   export RUN_TAG=vista_subset_rerun
EXP5_VIEWS="${EXP5_VIEWS:-62877247:L:CC,45209155:R:MLO}"
RUN_TAG="${RUN_TAG:-vista_subset}"

python scripts/batch_mammo_correspondence.py \
  --pack-dir "$PROJECT/data/roi_overlays_cancer5" \
  --out-dir "$PROJECT/outputs/batch_experiments" \
  --experiments exp1,exp2,exp3,exp5 \
  --exp12-patient 11513410 \
  --exp3-max-sources 3 \
  --anchor-patient 11513410 \
  --exp5-views "$EXP5_VIEWS" \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --run-tag "$RUN_TAG" \
  --device cuda:0

echo "Done."
echo "  SemCorre + TPS: $PROJECT/outputs/batch_experiments/$RUN_TAG/"
echo "  Per pair: *_correspondences_estimated.png, *_tps_roi_warp_before_after.png"
echo "  exp5: */*_chain_overview.png"
