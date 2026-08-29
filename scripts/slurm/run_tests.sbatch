#!/bin/bash
#SBATCH --job-name=na-tests
#SBATCH --partition=short
#SBATCH --time=30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --export=ALL
#SBATCH --output=experiments/logs/%x-%j.out
#SBATCH --error=experiments/logs/%x-%j.err

set -uo pipefail
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
/home/aeternum/anaconda3/envs/comfy/bin/python -m pytest ${TEST_TARGETS:-tests} -q
