#!/bin/bash
#SBATCH -J semcorre_exp5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o logs/semcorre_exp5_%j.out
#SBATCH -e logs/semcorre_exp5_%j.err
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p YOUR_GPU_PARTITION

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

module load python3 cuda 2>/dev/null || true
source venv/bin/activate
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"

python scripts/batch_mammo_correspondence.py \
  --pack-dir "$PROJECT/data/roi_overlays_cancer5" \
  --out-dir "$PROJECT/outputs/batch_experiments" \
  --experiments exp5 \
  --exp5-views "62877247:L:CC,45209155:R:MLO" \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --run-tag vista_exp5_rmlo \
  --device cuda:0

echo "Done: $PROJECT/outputs/batch_experiments/vista_exp5_rmlo/"
