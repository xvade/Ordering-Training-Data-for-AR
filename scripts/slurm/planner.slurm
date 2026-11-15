#!/bin/bash
# Run a headless Claude Code planning agent inside a Slurm allocation (CPU only).
#
#   sbatch scripts/slurm/planner.slurm docs/plans/<brief>.md
#
# Agent fan-out must not run on a login node (see docs/KLONE.md section 8); this
# gives the agent its own compute-node allocation. The brief is passed as the
# prompt; the agent's final message goes to logs/slurm/planner-<job>.out.
#SBATCH --job-name=planner
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/env.sh
echo "host=$(hostname -s) job=$SLURM_JOB_ID start=$(date '+%F %T %Z') brief=$1"
timeout 8400 claude -p "$(cat "$1")" --model claude-fable-5-2 --effort max \
    --allowedTools "Read" "Glob" "Grep" "Bash" "Write" "Edit"
echo "end=$(date '+%F %T %Z')"
