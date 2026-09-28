#!/bin/bash
#SBATCH -J semcorre_full
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o logs/semcorre_full_%j.out
#SBATCH -e logs/semcorre_full_%j.err
#SBATCH -A YOUR_ALLOCATION
#SBATCH -p YOUR_GPU_PARTITION

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

module load python3 cuda 2>/dev/null || true
source venv/bin/activate
export HF_HOME="${HF_HOME:-$PROJECT/hf_cache}"

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PROJECT/data/roi_overlays_cancer5" \
  --out-dir "$PROJECT/outputs/batch_experiments" \
  --experiments exp1,exp2,exp3,exp4,exp5 \
  --anchor-patient all \
  --exp5-max-steps 0 \
  --num_opt_iterations 5 \
  --num_iterations 20 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --run-tag "vista_full_${SLURM_JOB_ID}" \
  --device cuda:0

echo "Done. See $PROJECT/outputs/batch_experiments/"
