#!/bin/bash
#SBATCH --job-name=na-squash
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

# Single-writer squash consolidation: pack tars -> shots.sqsh. Never an array.
set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

# Sweep stale staging orphans (preempted tasks: bash skips EXIT traps on
# unhandled SIGTERM, so dirs accumulate and eventually ENOSPC the node).
for _d in /scratch /tmp; do
  find "$_d" -maxdepth 1 -user "$USER" \( -name "na-stream-*" -o -name "na-split-*" -o -name "na-squash-*" \) -mmin +240 -exec rm -rf {} + 2>/dev/null || true
done
STAGING="${SLURM_TMPDIR:-/tmp}/na-squash-${SLURM_JOB_ID:-$$}"
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT
trap 'rm -rf "$STAGING"; exit 143' INT TERM

"$COMFY_PY" tools/consolidate_squash.py \
  --shots-dir "$SAKUGA_ROOT/shots" \
  --staging-dir "$STAGING" \
  --batch-size "${BATCH_SIZE:-20}"
