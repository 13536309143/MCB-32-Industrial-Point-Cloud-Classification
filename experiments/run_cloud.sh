#!/usr/bin/env bash
# Linux GPU server entry point; respects the existing, isolated cloud environment.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-key6}"
case "$profile" in quick|smoke|full|key6) ;; *) echo 'Usage: bash experiments/run_cloud.sh [key6|smoke|quick|full]'; exit 1 ;; esac
ulimit -n 65536
runtime="${MCB_CLOUD_RUNTIME:-$project_root/.cloud-runtime}"
python="${MCB_CLOUD_PYTHON:-$runtime/venv/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$runtime/venv/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_JOBS="${MAX_JOBS:-2}"
cd "$project_root"
test -x "$python" || { echo "Missing cloud Python: $python"; exit 1; }
test -x "$CUDA_HOME/bin/nvcc" || { echo "Missing matching CUDA compiler: $CUDA_HOME/bin/nvcc"; exit 1; }
mkdir -p "$runtime/ops" "results/$profile" remote_logs
"$python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
"$python" -u experiments/build_ops.py --output "$runtime/ops"
export PYTHONPATH="$runtime/ops:$project_root${PYTHONPATH:+:$PYTHONPATH}"
"$python" -u -m experiments.mcb.preflight
"$python" -m pip freeze > "results/$profile/pip-freeze.txt"
cp "$runtime/ops/build_metadata.json" "results/$profile/cuda_build.json"
"$python" -u -m experiments.mcb.suite --profile "$profile" --no-pack
