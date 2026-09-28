#!/usr/bin/env bash
# Pack (optional) + SCP SemCorre code + mammogram pack to TACC Vista.
# Usage from repo root (Git Bash on Windows):
#   bash scripts/vista_scp_to_semcorre.sh
#
# Env overrides:
#   VISTA=paliniya@vista.tacc.utexas.edu
#   REMOTE=/home1/11364/paliniya/projects/semcorre
#   SKIP_PACK=1          — do not rebuild tarball
#   SKIP_DATA=1          — code only (no mammogram PNGs)
#   CODE_ARCHIVE=path    — use existing .tar.gz instead of packing

set -euo pipefail

VISTA="${VISTA:-paliniya@vista.tacc.utexas.edu}"
REMOTE="${REMOTE:-/home1/11364/paliniya/projects/semcorre}"

LOCAL_CODE="/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence"
LOCAL_PACK="/c/Users/paliniya/Desktop/apply/projects in progress/StableKeypointsPlus/local/data/packs/roi_overlays_cancer5"

cd "$LOCAL_CODE"

if [[ "${SKIP_PACK:-0}" != "1" && -z "${CODE_ARCHIVE:-}" ]]; then
  bash scripts/vista_pack_for_transfer.sh
fi
CODE_ARCHIVE="${CODE_ARCHIVE:-$(ls -t deploy/vista_semcorre_code_*.tar.gz 2>/dev/null | head -1)}"

if [[ -z "$CODE_ARCHIVE" || ! -f "$CODE_ARCHIVE" ]]; then
  echo "No code archive. Run: bash scripts/vista_pack_for_transfer.sh"
  exit 1
fi

echo "Remote: $VISTA:$REMOTE"
echo "Test login first: ssh $VISTA   (TACC MFA token if prompted)"
ssh "$VISTA" "mkdir -p $REMOTE/deploy $REMOTE/data $REMOTE/logs $REMOTE/outputs/batch_experiments"

echo "Upload code archive: $CODE_ARCHIVE"
scp "$CODE_ARCHIVE" "$VISTA:$REMOTE/deploy/"
MANIFEST="${CODE_ARCHIVE%.tar.gz}.manifest.txt"
[[ -f "$MANIFEST" ]] && scp "$MANIFEST" "$VISTA:$REMOTE/deploy/" || true

if [[ "${SKIP_DATA:-0}" != "1" ]]; then
  if [[ ! -d "$LOCAL_PACK" ]]; then
    echo "WARNING: pack not found at $LOCAL_PACK — skip data (set SKIP_DATA=1 to silence)"
  else
    echo "Upload mammogram pack (large; may take a long time)..."
    scp -r "$LOCAL_PACK" "$VISTA:$REMOTE/data/"
  fi
else
  echo "SKIP_DATA=1 — not uploading roi_overlays_cancer5"
fi

echo ""
echo "On Vista login node:"
echo "  ssh $VISTA"
echo "  cd $REMOTE"
echo "  bash scripts/vista_unpack_on_login.sh deploy/$(basename "$CODE_ARCHIVE")"
echo "  bash scripts/vista_setup.sh && source venv/bin/activate"
echo "  huggingface-cli login"
echo "  nano scripts/run_vista_exp_all_1to5.slurm   # #SBATCH -A and -p"
echo "  sbatch scripts/run_vista_exp_all_1to5.slurm"
echo "  squeue -u \$USER"
echo "  tail -f logs/semcorre_1to5_*.out"
