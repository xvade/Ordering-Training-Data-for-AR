# Source this inside a Slurm job to get the project environment.
# See docs/KLONE.md for why the paths are what they are.
export AR_VENV=/gscratch/amath/sgvtc/ordering-ar/venv
export AR_DATA=/gscratch/scrubbed/sgvtc/ordering-ar-data
export AR_RUNS=/gscratch/scrubbed/sgvtc/ordering-ar-runs
export UV_CACHE_DIR=/gscratch/scrubbed/sgvtc/uv-cache UV_LINK_MODE=copy
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_MAX_THREADS=$OMP_NUM_THREADS
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$AR_VENV/bin/python"
