#!/usr/bin/env bash
# Redraw pair PNGs + exp3 group grids from saved JSON/PT (no GPU). Run on Vista login or compute after batch.
#
#   export SCRATCH=/scratch/11364/paliniya
#   export PACK=$SCRATCH/sk_review/roi_overlays_exp1_views   # per experiment
#   bash scripts/vista_regen_figures.sh vista_exp1_views
#
# Optional second arg: pack dir (default tries common scratch paths).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${1:?usage: vista_regen_figures.sh RUN_TAG [PACK_DIR]}"
PACK="${2:-${PACK:-}}"

SEMCRE_OUT="${SEMCRE_OUT:-${SCRATCH:-}/semcorre_batch_outputs}"
RUN_ROOT="$SEMCRE_OUT/batch_experiments/$TAG"

if [[ ! -d "$RUN_ROOT" ]]; then
  echo "Missing run root: $RUN_ROOT"
  exit 1
fi

if [[ -z "$PACK" ]]; then
  case "$TAG" in
    *exp1*) PACK="${SCRATCH}/sk_review/roi_overlays_exp1_views" ;;
    *exp3*) PACK="${SCRATCH}/sk_review/roi_overlays_exp3_cross" ;;
    *exp2*) PACK="${SCRATCH}/sk_review/roi_overlays_exp2_lateral" ;;
    *exp5*) PACK="${SCRATCH}/sk_review/roi_overlays_exp5_temporal" ;;
    *) PACK="${SCRATCH}/sk_review/roi_overlays_cancer5" ;;
  esac
fi

pick_python() {
  if [[ -n "${SEMCRE_VENV:-}" && -x "${SEMCRE_VENV}/bin/python" ]]; then
    echo "${SEMCRE_VENV}/bin/python"
    return
  fi
  if [[ -f "$ROOT/.semcorre_venv_path" ]]; then
    local v
    v=$(tr -d '\r\n' < "$ROOT/.semcorre_venv_path")
    if [[ -x "$v/bin/python" ]]; then
      echo "$v/bin/python"
      return
    fi
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return
  fi
  local d="/scratch/11364/paliniya/stablekeypoints/sk_env"
  if [[ -x "$d/bin/python" ]]; then
    echo "$d/bin/python"
    return
  fi
  command -v python3 || command -v python
}
PYTHON="$(pick_python)"
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "ERROR: $PYTHON has no torch. Use: source /scratch/.../sk_env/bin/activate"
  echo "  or: export SEMCRE_VENV=/scratch/11364/paliniya/stablekeypoints/sk_env"
  exit 1
fi

echo "Run root: $RUN_ROOT"
echo "Pack:     $PACK"
echo "Python:   $PYTHON"

"$PYTHON" scripts/regenerate_pair_figures.py \
  --run-root "$RUN_ROOT" \
  --pack-dir "$PACK"

if [[ "$TAG" == *exp3* ]]; then
  "$PYTHON" scripts/draw_exp3_group_compare.py \
    --run-root "$RUN_ROOT" \
    --set-json data/exp_sets/exp3_cross_patient.json \
    --pack-dir "$PACK"
fi

if [[ "$TAG" == *exp5* ]]; then
  echo "Rebuilding exp5 chain overview PNGs from *_chain.json (no GPU)..."
  "$PYTHON" scripts/batch_mammo_correspondence.py \
    --rebuild-exp5-overviews "$RUN_ROOT" \
    --pack-dir "$PACK"
fi

echo "Done regen for $TAG"
