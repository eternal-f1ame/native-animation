#!/bin/bash
#SBATCH --job-name=na-delta-scrape
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -euo pipefail

REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"

MAX_ID=$(python -c "import json;print(json.load(open('$SAKUGA_ROOT/extraction-summary.json'))['max_post_id'])")
echo "delta scrape from post id $((MAX_ID+1))"
"$COMFY_PY" tools/scrape_sakugabooru.py --anime "id:$((MAX_ID+1))..400000" --limit 999999 --min-score 0
