#!/bin/bash
#SBATCH --job-name=na-base-demo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --constraint=gmem48|gmem80
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs" "$EXPERIMENTS_ROOT/demo/base"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "HOSTNAME=$(hostname)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
"$COMFY_PY" -V
nvidia-smi

"$COMFY_PY" -m native_animation.inference.run_baseline \
  --input-csv "$METADATA_DIR/metadata_test.csv" \
  --dataset-base-path "$DATA_ROOT" \
  --output-dir "$EXPERIMENTS_ROOT/demo/base" \
  --limit 2 \
  --unique-series \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --num-inference-steps 30 \
  --cfg-scale 5.0 \
  --seed-base 1234 \
  --fps 15 \
  --quality 5 \
  --tiled
