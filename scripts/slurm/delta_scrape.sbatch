#!/bin/bash
#SBATCH --job-name=na-delta-scrape
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --requeue
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

MAX_STEM=$(ls "$SAKUGA_ROOT/snapshot-2025/train" | sed 's/\.tar$//' | sort -n | tail -1)
MAX_ID=$(( MAX_STEM * 1000 + 999 ))
echo "delta scrape from post id $((MAX_ID+1))"
"$COMFY_PY" tools/scrape_sakugabooru.py --anime "id:$((MAX_ID+1))..400000" --limit 999999 --min-score 0
