#!/usr/bin/env bash
# Pull latest SemCorre code on Vista login (use after git push from PC).
#
#   cd ~/projects/semcorre
#   bash scripts/vista_git_pull.sh
#
# Optional: VISTA_GIT_BRANCH=main bash scripts/vista_git_pull.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .git ]]; then
  echo "ERROR: $ROOT is not a git repository."
  echo "  Clone: git clone https://github.com/Parvaneh-Aliniya/SemCorre.git $ROOT"
  exit 1
fi

BRANCH="${VISTA_GIT_BRANCH:-stats/benchmarks}"

echo "=== git pull ($BRANCH) in $ROOT ==="
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git pull origin "$BRANCH"

echo "=== normalize CRLF on slurm/shell ==="
for f in scripts/*.slurm scripts/*.sh; do
  [[ -f "$f" ]] && sed -i 's/\r$//' "$f"
done

echo "=== HEAD ==="
git log -1 --oneline
echo "Done. Run batch or: module load python3; source \$SEMCRE_VENV/bin/activate; python scripts/regenerate_pair_figures.py ..."
