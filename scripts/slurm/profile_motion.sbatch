#!/bin/bash
#SBATCH --job-name=na-profile
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
source "$REPO_ROOT/scripts/slurm/lib/mount_shots.sh"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$COMFY_PY" tools/profile_motion.py \
  --shots-dir "$SAKUGA_ROOT/shots" \
  --out-dir "$SAKUGA_ROOT/profiles" \
  --config configs/stage0.yaml \
  --shard "$SLURM_ARRAY_TASK_ID"
