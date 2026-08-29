#!/bin/bash
#SBATCH --job-name=na-accounting
#SBATCH --partition=short
#SBATCH --time=30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err
set -uo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
source "$REPO_ROOT/configs/paths.env"
S="$SAKUGA_ROOT/shots"
{
echo "════════ STAGE-0 CORPUS ACCOUNTING $(date) ════════"
echo "total shot files: $(find "$S" -name '*.mp4' | wc -l)"
echo "tars done-marked: $(ls -a "$S/manifests" | grep -c "^\.done_")"
echo "manifest records: $(cat "$S"/manifests/shard_*.jsonl 2>/dev/null | wc -l)"
echo "curation pass:    $(cat "$S"/manifests/shard_*.jsonl 2>/dev/null | grep -o '"pass": true' | wc -l)"
echo "shots size GB:    $(du -s --block-size=1G "$S" | cut -f1)"
echo "profiles:         $(cat "$SAKUGA_ROOT"/profiles/shard_*.jsonl 2>/dev/null | wc -l)"
echo "series dirs:      $(find "$S" -maxdepth 1 -type d | wc -l)"
} | tee "$REPO_ROOT/experiments/logs/corpus-accounting-final.txt"
