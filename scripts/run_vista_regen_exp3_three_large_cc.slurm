#!/bin/bash
# CPU-only: regen three exp3 cross CC figures from pack (no LDM).
#SBATCH -J exp3_regen3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -o logs/exp3_regen3_%j.out
#SBATCH -e logs/exp3_regen3_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh-dev

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

module load gcc python3 2>/dev/null || true
export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
export SEMCRE_VENV="${SEMCRE_VENV:-/scratch/11364/paliniya/stablekeypoints/sk_env}"

bash scripts/vista_regen_exp3_three_large_cc.sh
