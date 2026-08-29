#!/bin/bash
#SBATCH --job-name=na-anno-smoke
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gmem48|gmem80
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$MODELS_ROOT/hf-cache"
export PYTHONPATH="$MODELS_ROOT/anno-overlay:$PYTHONPATH"
echo "host=$(hostname)"; date
time "$COMFY_PY" tools/annotate_clips.py \
  --shots-dir "$SAKUGA_ROOT/shots" \
  --clips-dir "$SAKUGA_ROOT/clips" \
  --out-dir "$REPO_ROOT/experiments/smoke/annotation" \
  --config configs/stage0.yaml \
  --shard 0 --limit 30 --backend transformers
date
