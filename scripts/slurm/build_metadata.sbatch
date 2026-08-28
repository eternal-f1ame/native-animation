#!/bin/bash
#SBATCH --job-name=na-build-meta
#SBATCH --partition=short
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

mkdir -p "$EXPERIMENTS_ROOT/logs" "$METADATA_DIR"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "PYTHON=$(which python)"
python -V

python -m native_animation.data.build_metadata \
  --input-root "$DATA_ROOT" \
  --output-dir "$METADATA_DIR" \
  --seed 42 \
  --val-ratio 0.1 \
  --test-ratio 0.1
