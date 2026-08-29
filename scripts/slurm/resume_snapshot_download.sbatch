#!/bin/bash
#SBATCH --job-name=na-snap-dl
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
set -euo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
"$COMFY_PY" - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='trojblue/sakugabooru2025', repo_type='dataset',
                      local_dir='/share/datasets/native-animation/sakugabooru/snapshot-2025',
                      max_workers=8)
print('DOWNLOAD COMPLETE:', p)
PY
