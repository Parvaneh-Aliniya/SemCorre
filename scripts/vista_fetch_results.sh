#!/usr/bin/env bash
# Pull Vista batch results to local Windows machine (Git Bash).
# Usage:
#   bash scripts/vista_fetch_results.sh vista_exp_all_1to5

set -euo pipefail

VISTA="${VISTA:-paliniya@vista.tacc.utexas.edu}"
REMOTE="${REMOTE:-/home1/11364/paliniya/projects/semcorre}"
RUN_TAG="${1:-vista_exp_all_1to5}"
# Full batch writes to scratch (see run_vista_exp_all_1to5.slurm)
REMOTE_OUT="${REMOTE_OUT:-/scratch/11364/paliniya/semcorre_batch_outputs/batch_experiments}"

LOCAL_ROOT="/c/Users/paliniya/Desktop/apply/projects in progress/semantic_correspondence"
LOCAL_OUT="$LOCAL_ROOT/outputs/batch_experiments"

mkdir -p "$LOCAL_OUT"

echo "SCP from $VISTA:$REMOTE_OUT/$RUN_TAG"
scp -r "$VISTA:$REMOTE_OUT/$RUN_TAG" "$LOCAL_OUT/" || \
  scp -r "$VISTA:$REMOTE/outputs/batch_experiments/$RUN_TAG" "$LOCAL_OUT/"

echo "Also fetching latest SLURM log..."
mkdir -p "$LOCAL_ROOT/deploy/vista_logs"
scp "$VISTA:$REMOTE/logs/semcorre_1to5_"*.out "$LOCAL_ROOT/deploy/vista_logs/" 2>/dev/null || true
scp "$VISTA:$REMOTE/logs/semcorre_1to5_"*.err "$LOCAL_ROOT/deploy/vista_logs/" 2>/dev/null || true

echo "Local: $LOCAL_OUT/$RUN_TAG"
