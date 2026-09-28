#!/usr/bin/env bash
# Exp5 — default: FULL backward chain (all older exams). Quick one-hop: EXP5_MAX_STEPS=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PACK="${PACK_DIR:-../StableKeypointsPlus/local/data/packs/roi_overlays_cancer5}"
OUT="${OUT_DIR:-outputs/batch_experiments}"
# Prefix only — batch appends _YYYY-MM-DD_HHMMSS so reruns do not overwrite (set RUN_TAG_EXACT=1 to fix path)
TAG="${RUN_TAG:-exp5_62877247_lcc}"
EXP5_VIEWS="${EXP5_VIEWS:-62877247:L:CC}"
EXP5_MAX_STEPS="${EXP5_MAX_STEPS:-0}"
TPS_MODES="${TPS_MODES:-roi}"

TPS_EXTRA=(--no-semcorre-after-tps)
if [[ "${SEMCORRE_AFTER_TPS:-0}" == "1" ]]; then
  TPS_EXTRA=(--semcorre-after-tps)
fi

echo "Exp5: $EXP5_VIEWS | max_steps=$EXP5_MAX_STEPS (0 = full chain)"
echo "TPS modes: $TPS_MODES | post-TPS SemCorre: ${SEMCORRE_AFTER_TPS:-0}"
echo "Output prefix: $OUT/${TAG}_<UTC-date-time>/"
echo ""

RUN_TAG_EXTRA=()
if [[ "${RUN_TAG_EXACT:-0}" == "1" ]]; then
  RUN_TAG_EXTRA=(--run-tag-exact)
fi

python scripts/batch_mammo_correspondence.py \
  --pack-dir "$PACK" \
  --out-dir "$OUT" \
  --experiments exp5 \
  --exp5-views "$EXP5_VIEWS" \
  --exp5-max-steps "$EXP5_MAX_STEPS" \
  --num_opt_iterations "${NUM_OPT_ITERATIONS:-3}" \
  --num_iterations "${NUM_ITERATIONS:-10}" \
  --with-tps-warp \
  --tps-modes "$TPS_MODES" \
  "${TPS_EXTRA[@]}" \
  --run-tag "$TAG" \
  "${RUN_TAG_EXTRA[@]}" \
  --device "${DEVICE:-cuda:0}"
