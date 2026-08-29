#!/bin/bash
#SBATCH --job-name=na-annotate
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gmem48|gmem80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-63%8
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%A_%a.out
#SBATCH --error=experiments/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$MODELS_ROOT/hf-cache"

/home/aeternum/anaconda3/envs/anno/bin/python tools/annotate_clips.py \
  --shots-dir "$WORKSPACE_ROOT/data/sakugabooru/shots" \
  --clips-dir "$WORKSPACE_ROOT/data/sakugabooru/clips" \
  --out-dir "$WORKSPACE_ROOT/data/sakugabooru/captions" \
  --config configs/stage0.yaml \
  --shard "$SLURM_ARRAY_TASK_ID"
