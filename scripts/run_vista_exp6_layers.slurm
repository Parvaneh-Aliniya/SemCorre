#!/bin/bash
# Exp6: same CC↔MLO pairs as Exp1 (exp1_views pack), DHPF layers 2–6 combined.
#SBATCH -J semcorre_e6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o logs/semcorre_e6_layers_%j.out
#SBATCH -e logs/semcorre_e6_layers_%j.err
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

unset PACK
PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_exp1_views}"
RUN_TAG="${RUN_TAG:-vista_exp6_layers_2_6}"
SET_JSON="${SET_JSON:-$PROJECT/data/exp_sets/exp6_layers_2_6.json}"
LAYERS=(2 3 4 5 6)
test -f "$SET_JSON" || { echo "ERROR: missing $SET_JSON"; exit 1; }

echo "=== Exp6 layers 2-6 === $(hostname) $(date -u)"
echo "PACK=$PACK  RUN_TAG=$RUN_TAG  SET_JSON=$SET_JSON  LAYERS=${LAYERS[*]}"

test -f "$PACK/roi_coords.csv" || { echo "ERROR: build exp1 views pack first (run_vista_exp1_views or draw pack)"; exit 1; }

nvidia-smi -L || { echo "ERROR: no GPU"; exit 1; }

python -u scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$SEMCRE_OUT/batch_experiments" \
  --run-tag "$RUN_TAG" \
  --run-tag-exact \
  --skip-existing \
  --experiments exp6 \
  --set-json "$SET_JSON" \
  --layers "${LAYERS[@]}" \
  --gt-compare \
  --num_opt_iterations 3 \
  --num_iterations 10 \
  --with-tps-warp \
  --tps-modes roi \
  --no-semcorre-after-tps \
  --device cuda:0

echo "Done: $SEMCRE_OUT/batch_experiments/$RUN_TAG/"
