#!/bin/bash
#SBATCH --job-name=na-smoke-mem
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --constraint=gmem80
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "host=$(hostname)"; nvidia-smi | head -12

accelerate launch \
  --config_file configs/accelerate_fsdp.yaml \
  --num_processes "${SLURM_GPUS_ON_NODE:-4}" \
  src/native_animation/training/stages/train_stage.py \
  --config configs/smoke_v1data.yaml \
  --smoke-steps 3 --dataset-limit 8

python - <<'EOF'
import torch
print("smoke complete; per-rank peak memory printed above by torch if any")
EOF
