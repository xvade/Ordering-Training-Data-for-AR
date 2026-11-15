#!/bin/bash
# Generic single-GPU job: runs the given command line inside the project environment.
#
#   sbatch --job-name=NAME --time=HH:MM:SS scripts/slurm/run.slurm python -m arorder.train --run-dir ...
#
# "python" at the start of the command is replaced by the project interpreter.
# Submit from the repository root (logs go to logs/slurm/, which must exist).
#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
echo "host=$(hostname -s) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date '+%F %T %Z')"
echo "cmd: $*"
if [ "$1" = "python" ]; then shift; set -- "$PY" "$@"; fi
"$@"
echo "end=$(date '+%F %T %Z')"
