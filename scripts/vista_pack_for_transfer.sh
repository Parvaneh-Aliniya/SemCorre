#!/usr/bin/env bash
# LEGACY: tarball upload when git pull on Vista is not available.
# Preferred sync: git push (PC) + bash scripts/vista_git_pull.sh (Vista) — see docs/VISTA_GIT_SYNC.md
#
#   bash scripts/vista_pack_for_transfer.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p deploy

STAMP="$(date +%Y-%m-%d_%H%M%S)"
ARCHIVE="deploy/vista_semcorre_code_${STAMP}.tar.gz"
MANIFEST="deploy/vista_semcorre_code_${STAMP}.manifest.txt"

echo "Packing code from: $ROOT"
echo "Normalizing LF in scripts/*.sh and scripts/*.slurm ..."
python - <<'PY'
from pathlib import Path
root = Path("scripts")
for p in list(root.glob("*.sh")) + list(root.glob("*.slurm")):
    b = p.read_bytes()
    if b"\r" in b:
        p.write_bytes(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        print("  LF:", p)
PY

tar czf "$ARCHIVE" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  scripts \
  utils \
  eval \
  docs \
  data/exp_sets \
  requirements-pip.txt \
  README.md 2>/dev/null || tar czf "$ARCHIVE" \
  scripts utils eval docs data/exp_sets requirements-pip.txt

# Checklist copied into deploy/ for upload alongside tarball
CHECKLIST="deploy/VISTA_FINAL_RUN_CHECKLIST.md"
if [[ -f docs/VISTA_FINAL_RUN.md ]]; then
  cp -f docs/VISTA_FINAL_RUN.md "$CHECKLIST"
fi

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "archive=$ARCHIVE"
  echo "contents:"
  tar tzf "$ARCHIVE" | head -200
  echo "..."
  echo "file_count=$(tar tzf "$ARCHIVE" | wc -l)"
} > "$MANIFEST"

echo ""
echo "Created:"
echo "  $ARCHIVE"
echo "  $MANIFEST"
echo "  $CHECKLIST"
echo ""
echo "Upload tarball + checklist:"
echo "  scp \"$ARCHIVE\" \"$CHECKLIST\" paliniya@vista.tacc.utexas.edu:~/projects/semcorre/deploy/"
echo ""
echo "On Vista login: tar xzf deploy/vista_semcorre_code_*.tar.gz -C .  (see $CHECKLIST)"
echo ""
echo "Mammogram packs stay on SCRATCH (not in tarball): sk_review/roi_overlays_*"
