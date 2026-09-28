#!/bin/bash
# One-time SemCorre venv on TACC Vista (run on login node, ideally after idev -p gh-dev).
#   cd /home1/11364/paliniya/projects/semcorre
#   bash scripts/vista_setup.sh

set -euo pipefail

PROJECT="${SEMCRE_ROOT:-/home1/11364/paliniya/projects/semcorre}"
cd "$PROJECT"

module load gcc cuda 2>/dev/null || module load gcc/15.1.0 cuda 2>/dev/null || true
module load python3 2>/dev/null || module load python3/3.11.8 2>/dev/null || true

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -U pip wheel

# Vista docs: cu129 wheels on Hopper; cu124 fallback if install fails.
if ! pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129; then
  echo "cu129 wheels failed; trying cu124..."
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

pip install diffusers transformers accelerate huggingface_hub
pip install pandas openpyxl pillow matplotlib scipy scikit-image pynvml tqdm

export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
mkdir -p "$SCRATCH" "${SCRATCH}/hf_cache_semcorre" logs outputs/batch_experiments

echo "OK: venv at $PROJECT/venv"
echo "Set on jobs: export HF_HOME=$SCRATCH/hf_cache_semcorre"
echo "Next: huggingface-cli login"
echo "Test GPU (gh-dev node): idev -p gh-dev -N 1 -n 1 -t 0:30:00"
echo "      source venv/bin/activate && python -c \"import torch; print(torch.cuda.is_available())\""
echo "Preflight: bash scripts/vista_job_preflight.sh"
