#!/bin/bash
# Vista readiness check for SemCorre exp1–5 batch (run on login node or at job start).
#   cd /home1/11364/paliniya/projects/semcorre
#   bash scripts/vista_job_preflight.sh

set -euo pipefail

PROJECT="${SEMCRE_ROOT:-/home1/11364/paliniya/projects/semcorre}"
cd "$PROJECT"

# Vista has no `conda` on PATH. sk_env is a venv on scratch (same as StableKeypointsPlus).
SK_ENV_DEFAULT="${SK_ENV:-/scratch/11364/paliniya/stablekeypoints/sk_env}"

_semcre_venv_dir() {
  local p cand
  if [[ -n "${SEMCRE_VENV:-}" && -x "${SEMCRE_VENV}/bin/python" ]]; then
    echo "$SEMCRE_VENV"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "$VIRTUAL_ENV"
    return 0
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "$CONDA_PREFIX"
    return 0
  fi
  if [[ -f "$PROJECT/.semcorre_venv_path" ]]; then
    p=$(tr -d '\r\n' < "$PROJECT/.semcorre_venv_path" || true)
    if [[ -n "$p" && -x "$p/bin/python" ]]; then
      echo "$p"
      return 0
    fi
  fi
  for cand in "$SK_ENV_DEFAULT" "$PROJECT/venv"; do
    if [[ -n "$cand" && -x "$cand/bin/python" ]]; then
      echo "$cand"
      return 0
    fi
  done
  return 0
}

FAIL=0
warn() { echo "WARN: $*"; }
err() { echo "ERROR: $*"; FAIL=1; }
ok() { echo "OK: $*"; }

echo "=== Vista SemCorre preflight ==="
echo "PROJECT=$PROJECT"
echo "HOST=$(hostname)  DATE=$(date -u)"

# --- SLURM script sanity ---
SLURM=scripts/run_vista_exp_all_1to5.slurm
if [[ ! -f "$SLURM" ]]; then
  err "Missing $SLURM"
else
  if grep -q $'\r' "$SLURM" 2>/dev/null || file "$SLURM" | grep -q CRLF; then
    err "$SLURM has DOS line breaks — run: sed -i 's/\\r$//' $SLURM"
  else
    ok "$SLURM line endings (LF)"
  fi
  if grep -q 'YOUR_ALLOCATION\|YOUR_GPU_PARTITION' "$SLURM"; then
    err "$SLURM still contains YOUR_ALLOCATION or YOUR_GPU_PARTITION placeholders"
  fi
  N_A=$(grep -c '^#SBATCH -A ' "$SLURM" || true)
  N_P=$(grep -c '^#SBATCH -p ' "$SLURM" || true)
  if [[ "$N_A" -ne 1 || "$N_P" -ne 1 ]]; then
    err "$SLURM must have exactly one #SBATCH -A and one #SBATCH -p (found A=$N_A p=$N_P)"
  else
    ok "SBATCH -A / -p (single each): $(grep '^#SBATCH -A ' "$SLURM") $(grep '^#SBATCH -p ' "$SLURM")"
  fi
  PART=$(grep '^#SBATCH -p ' "$SLURM" | awk '{print $NF}')
  if [[ "$PART" == "gg" ]]; then
    err "Queue is gg (Grace–Grace CPU). SemCorre needs GPU queue gh (Hopper). sed -i 's/^#SBATCH -p gg\$/#SBATCH -p gh/' $SLURM"
  elif [[ "$PART" == "gh" || "$PART" == "gh-dev" ]]; then
    ok "GPU queue: $PART"
  else
    warn "Unknown partition '$PART' — confirm with: sinfo"
  fi
fi

# --- Code tree ---
for d in scripts utils eval; do
  [[ -d "$d" ]] && ok "dir $d" || err "missing directory $d"
done
for f in scripts/batch_mammo_correspondence.py scripts/interactive_correspondence.py utils/embed_roi.py; do
  [[ -f "$f" ]] && ok "$f" || err "missing $f"
done

# --- Data pack ---
PACK="${PACK:-$PROJECT/data/roi_overlays_cancer5}"
if [[ ! -d "$PACK" ]]; then
  err "Mammogram pack missing: $PACK (scp from Windows to data/roi_overlays_cancer5)"
else
  ok "pack dir $PACK"
  for need in roi_coords.csv roi_coords.xlsx; do
    if [[ -f "$PACK/$need" ]]; then
      ok "  $need"
      break
    fi
  done
  if ! ls "$PACK"/patient_* >/dev/null 2>&1; then
    err "  no patient_* folders under pack"
  else
    NP=$(ls -d "$PACK"/patient_* 2>/dev/null | wc -l)
    ok "  $NP patient_* folder(s)"
  fi
fi

# --- Disk (home is often small; HF + outputs should use SCRATCH) ---
echo "--- Quota (df) ---"
df -h "$HOME" 2>/dev/null | tail -1 || true
if [[ -n "${SCRATCH:-}" && -d "$SCRATCH" ]]; then
  df -h "$SCRATCH" 2>/dev/null | tail -1 || true
  ok "SCRATCH=$SCRATCH (recommended for HF_HOME and large outputs)"
else
  warn "SCRATCH unset — HF cache and outputs will use PROJECT (may fill home quota)"
fi

# --- Python env (venv on scratch — do not use conda activate) ---
VENV_DIR=$(_semcre_venv_dir)
if [[ -z "$VENV_DIR" ]]; then
  err "No Python env found. On Vista, sk_env is a venv (conda is not installed):"
  echo "  echo $SK_ENV_DEFAULT > .semcorre_venv_path"
  echo "  test -x $SK_ENV_DEFAULT/bin/python && echo OK_PYTHON"
  if [[ -f "$PROJECT/.semcorre_venv_path" ]]; then
    echo "  current .semcorre_venv_path=[$(tr -d '\r\n' < "$PROJECT/.semcorre_venv_path")]"
  fi
else
  ok "Python env: $VENV_DIR"
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    set +e
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    act_rc=$?
    set -e
    if [[ "$act_rc" -ne 0 ]]; then
      warn "source activate failed (rc=$act_rc); using PATH=$VENV_DIR/bin"
      export PATH="$VENV_DIR/bin:$PATH"
    fi
  else
    export PATH="$VENV_DIR/bin:$PATH"
  fi
  module load gcc cuda 2>/dev/null || true
  module load python3 2>/dev/null || true
  if python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null; then
    ok "torch + CUDA in venv ($(python -c 'import torch; print(torch.__version__)'))"
  else
    warn "torch CUDA not available on login node (expected). Testing import only..."
    python -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)" || err "torch import failed"
  fi
  for mod in diffusers transformers pandas PIL scipy; do
    python -c "import $mod" 2>/dev/null && ok "import $mod" || err "pip install missing: $mod"
  done
fi

# --- Hugging Face ---
HF="${HF_HOME:-${SCRATCH:-$PROJECT}/hf_cache_semcorre}"
if [[ -d "$HF" ]] && ls "$HF"/hub/models--* >/dev/null 2>&1; then
  ok "HF cache has models under $HF"
else
  warn "SD 1.4 not cached yet at $HF — first job run will download (~4GB+). Run: huggingface-cli login"
fi

echo "=== Summary ==="
if [[ "$FAIL" -ne 0 ]]; then
  echo "PREFLIGHT FAILED — fix errors above before sbatch"
  exit 1
fi
echo "PREFLIGHT PASSED — safe to: sbatch scripts/run_vista_exp_all_1to5.slurm"
exit 0
