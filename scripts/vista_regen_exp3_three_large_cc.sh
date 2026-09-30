#!/usr/bin/env bash
# Redraw bidirectional pair figures for the three large-bucket CC → target 55757641 pairs.
# Uses roi_overlays_exp3_cross pack PNGs (not cropped figures from download).
#
# On Vista login (or gh-dev if login has no torch):
#   cd ~/projects/semcorre
#   bash scripts/vista_git_pull.sh    # needs --no-direction-arrows support on regenerate_pair_figures.py
#   bash scripts/vista_regen_exp3_three_large_cc.sh
#
# Optional:
#   RUN=$SCRATCH/semcorre_batch_outputs/batch_experiments/vista_exp3_cross_roi_box bash ...
#   NO_ARROWS=0 bash ...              # keep orange/pink arrows
#   sbatch scripts/run_vista_regen_exp3_three_large_cc.slurm

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SCRATCH="${SCRATCH:-/scratch/11364/paliniya}"
RUN_TAG="${RUN_TAG:-vista_exp3_cross}"
RUN_ROOT="${RUN:-$SCRATCH/semcorre_batch_outputs/batch_experiments/$RUN_TAG}"
PACK="${PACK:-$SCRATCH/sk_review/roi_overlays_exp3_cross}"
NO_ARROWS="${NO_ARROWS:-1}"
OUT_DIR="${OUT_DIR:-$SCRATCH/sk_review/exp3_three_large_cc_figures}"

SEMCRE_VENV="${SEMCRE_VENV:-/scratch/11364/paliniya/stablekeypoints/sk_env}"
if [[ -f "$ROOT/.semcorre_venv_path" ]]; then
  SEMCRE_VENV="$(tr -d '\r\n' < "$ROOT/.semcorre_venv_path")"
fi
# shellcheck disable=SC1091
source "$SEMCRE_VENV/bin/activate"
python -c "import torch; print('torch OK', torch.__version__)"

BASE="$RUN_ROOT/exp3_cross_patient"
test -d "$BASE" || { echo "Missing $BASE"; exit 1; }
test -f "$PACK/roi_coords.csv" || { echo "Missing pack $PACK/roi_coords.csv"; exit 1; }

PAIRS=(
  src_p16640764_2014_10_05_trg_p55757641_2015_04_10_r_cc_to_r
  src_p46377688_2017_04_14_trg_p55757641_2015_04_10_l_cc_to_r
  src_p96414778_2012_11_21_trg_p55757641_2015_04_10_r_cc_to_r
)

EXTRA=()
if [[ "$NO_ARROWS" == 1 ]]; then
  EXTRA+=(--no-direction-arrows)
fi

echo "RUN_ROOT=$RUN_ROOT"
echo "PACK=$PACK"
echo "OUT_DIR=$OUT_DIR"
echo "NO_ARROWS=$NO_ARROWS"

for d in "${PAIRS[@]}"; do
  pair_dir="$BASE/$d"
  test -f "$pair_dir/${d}_pair.json" || { echo "Missing pair: $pair_dir"; exit 1; }
  python scripts/regenerate_pair_figures.py \
    --run-root "$RUN_ROOT" \
    --pack-dir "$PACK" \
    --pair-dir "$pair_dir" \
    "${EXTRA[@]}"
done

mkdir -p "$OUT_DIR"
for d in "${PAIRS[@]}"; do
  cp "$BASE/$d/${d}_bidirectional_pair.png" "$OUT_DIR/${d}_bidirectional_pair.png"
done

tar -czf "$OUT_DIR.tgz" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "Done. Figures: $OUT_DIR"
echo "Bundle:       $OUT_DIR.tgz"
echo "scp: scp paliniya@vista.tacc.utexas.edu:$OUT_DIR.tgz ."
