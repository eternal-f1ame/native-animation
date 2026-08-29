#!/bin/bash
#SBATCH --job-name=na-profile
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-63
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python tools/profile_motion.py \
  --shots-dir "$WORKSPACE_ROOT/data/sakugabooru/shots" \
  --out-dir "$WORKSPACE_ROOT/data/sakugabooru/profiles" \
  --config configs/stage0.yaml \
  --shard "$SLURM_ARRAY_TASK_ID"
