#!/bin/bash
#SBATCH --job-name=na-split-shots
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-63
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Sweep stale staging orphans (preempted tasks: bash skips EXIT traps on
# unhandled SIGTERM, so dirs accumulate and eventually ENOSPC the node).
for _d in /scratch /tmp; do
  find "$_d" -maxdepth 1 -user "$USER" \( -name "na-stream-*" -o -name "na-split-*" -o -name "na-squash-*" \) -mmin +240 -exec rm -rf {} + 2>/dev/null || true
done
STAGING="${SLURM_TMPDIR:-/tmp}/na-split-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$STAGING"
trap 'rm -rf "$STAGING"' EXIT
trap 'rm -rf "$STAGING"; exit 143' INT TERM

"$COMFY_PY" tools/split_shots.py \
  --clips-dir "$SAKUGA_ROOT/clips" \
  --out-dir "$SAKUGA_ROOT/shots" \
  --config configs/stage0.yaml \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards 64 \
  --pack-shots --staging-dir "$STAGING"
