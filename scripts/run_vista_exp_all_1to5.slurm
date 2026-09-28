#!/bin/bash
# Full batch: exp1–exp5, all patients in roi_overlays_cancer5.
# Vista: MUST use GPU queue gh (Hopper). Queue gg is CPU-only (Grace–Grace).
# Resume: re-sbatch; uses --run-tag-exact + --skip-existing on RUN_TAG folder.
#
#SBATCH -J semcorre_1to5
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -o logs/semcorre_1to5_%j.out
#SBATCH -e logs/semcorre_1to5_%j.err
#SBATCH -A ASC26012
#SBATCH -p gh

set -euo pipefail

PROJECT=/home1/11364/paliniya/projects/semcorre
cd "$PROJECT"
mkdir -p logs

# Vista has no `conda` on PATH. Use the existing SK venv on scratch:
#   echo /scratch/11364/paliniya/stablekeypoints/sk_env > .semcorre_venv_path
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

# Keep heavy artifacts off small $HOME (use scratch when available).
export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
mkdir -p "$SCRATCH"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache_semcorre}"
export SEMCRE_OUT="${SEMCRE_OUT:-$SCRATCH/semcorre_batch_outputs}"
mkdir -p "$HF_HOME" "$SEMCRE_OUT/batch_experiments"
export PYTHONUNBUFFERED=1
echo "SEMCRE_VENV=${SEMCRE_VENV:-"(unset)"}"
echo "VENV_DIR=$VENV_DIR"
if [[ -n "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
  echo "Venv python: $("$VENV_DIR/bin/python" -c 'import sys; print(sys.executable)')"
else
  echo "WARN: no executable python at VENV_DIR (preflight will fail)"
fi

echo "=== Job start ==="
echo "Host: $(hostname)  Date: $(date -u)"
echo "PROJECT=$PROJECT"
echo "HF_HOME=$HF_HOME"
echo "SEMCRE_OUT=$SEMCRE_OUT"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

if ! bash scripts/vista_job_preflight.sh; then
  echo "Preflight failed — fix issues on login node, then re-submit."
  exit 1
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -x "$VENV_DIR/bin/python" ]]; then
  export PATH="$VENV_DIR/bin:$PATH"
else
  echo "ERROR: invalid SEMCRE_VENV / .semcorre_venv_path: $VENV_DIR"
  exit 1
fi

RUN_TAG="${RUN_TAG:-vista_exp_all_1to5}"
PACK="${PACK:-$PROJECT/data/roi_overlays_cancer5}"

echo "GPU on compute node:"
nvidia-smi -L || { echo "ERROR: no GPU on this node — wrong queue? Use #SBATCH -p gh"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$SEMCRE_OUT/batch_experiments" \
  --run-tag "$RUN_TAG" \
  --run-tag-exact \
  --skip-existing \
  --experiments exp1,exp2,exp3,exp4,exp5 \
  --anchor-patient all \
  --exp5-max-steps 0 \
  --num_opt_iterations 5 \
  --num_iterations 20 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --device cuda:0

echo "Done: $SEMCRE_OUT/batch_experiments/$RUN_TAG/"
echo "Copy results: scp -r vista:$SEMCRE_OUT/batch_experiments/$RUN_TAG ./"
