#!/usr/bin/env bash
# Run on Vista login node after SCP:
#   cd /home1/11364/paliniya/projects/semcorre
#   bash scripts/vista_unpack_on_login.sh deploy/vista_semcorre_code_*.tar.gz

set -euo pipefail

PROJECT="${SEMCRE_ROOT:-/home1/11364/paliniya/projects/semcorre}"
ARCHIVE="${1:-}"

cd "$PROJECT"
mkdir -p deploy logs outputs/batch_experiments data

if [[ -z "$ARCHIVE" ]]; then
  ARCHIVE="$(ls -t deploy/vista_semcorre_code_*.tar.gz 2>/dev/null | head -1)"
fi
if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "Usage: bash scripts/vista_unpack_on_login.sh [path/to/one.tar.gz]"
  echo "  (do not pass a glob with multiple .tar.gz files to tar xzf)"
  exit 1
fi

echo "Extracting $ARCHIVE into $PROJECT ..."
tar xzf "$ARCHIVE" -C "$PROJECT"

echo "Fixing CRLF -> LF for sbatch (Windows tarballs)..."
for f in scripts/*.slurm scripts/*.sh; do
  [[ -f "$f" ]] && sed -i 's/\r$//' "$f"
done

echo "OK. Next:"
echo "  bash scripts/vista_setup.sh"
echo "  source venv/bin/activate && huggingface-cli login"
echo "  nano scripts/run_vista_exp_all_1to5.slurm   # set #SBATCH -A and -p"
echo "  sbatch scripts/run_vista_exp_all_1to5.slurm"
