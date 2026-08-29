#!/bin/bash
#SBATCH --job-name=na-train-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gmem80
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

STAGE_CONFIG=${STAGE_CONFIG:-configs/ct_a.yaml}
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
echo "stage=$STAGE_CONFIG gpus=$NUM_GPUS host=$(hostname)"
nvidia-smi | head -10

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes "$NUM_GPUS" \
  src/native_animation/training/stages/train_stage.py \
  --config "$STAGE_CONFIG" \
  --resume auto \
  ${SMOKE_STEPS:+--smoke-steps $SMOKE_STEPS} \
  ${DATASET_LIMIT:+--dataset-limit $DATASET_LIMIT}
