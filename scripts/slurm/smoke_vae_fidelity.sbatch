#!/bin/bash
#SBATCH --job-name=na-smoke-vae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=gmem48|gmem80
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python tools/vae_fidelity_check.py \
  --metadata-csv "$METADATA_DIR/metadata_test.csv" \
  --dataset-base-path "$DATA_ROOT" \
  --out-dir experiments/smoke/vae_fidelity \
  --num-images 8
