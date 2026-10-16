#!/usr/bin/env bash
set -euo pipefail
project_root="${1:?}"
profile="${2:?}"
runtime="${3:?}"
reconverted="${4:-0}"
cd "$project_root"
uv="$runtime/uv"
python="$runtime/venv/bin/python"
"$uv" --no-config --system-certs python install cpython-3.11.13-linux-x86_64-gnu --no-bin
if [[ ! -x "$python" ]]; then
  "$uv" --no-config --system-certs venv "$runtime/venv" --python cpython-3.11.13-linux-x86_64-gnu --managed-python
fi
if ! "$python" -c 'import torch; assert torch.__version__.split("+")[0] == "2.8.0" and torch.version.cuda == "12.8"' 2>/dev/null; then
  "$uv" --no-config --system-certs pip install --python "$python" 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
fi
"$uv" --no-config --system-certs pip install --python "$python" -r experiments/requirements.txt
"$python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0), "VRAM GiB:", torch.cuda.get_device_properties(0).total_memory/2**30, "compute:", torch.cuda.get_device_capability(0)); assert torch.cuda.get_device_capability(0) >= (7, 0), "The pinned cu128 build requires a supported NVIDIA GPU"'
"$uv" --no-config pip freeze --python "$python" > "$runtime/pip-freeze.txt"
ops="$runtime/ops"
mkdir -p "$ops"
"$python" -u experiments/build_ops.py --output "$ops"
export PYTHONPATH="$ops:$project_root${PYTHONPATH:+:$PYTHONPATH}"
"$python" -u -m experiments.mcb.preflight
mkdir -p "results/$profile"
cp "$runtime/pip-freeze.txt" "results/$profile/pip-freeze.txt"
cp "$ops/build_metadata.json" "results/$profile/cuda_build.json"
suite_args=(--profile "$profile")
if [[ "$reconverted" == 1 ]]; then suite_args+=(--accept-reconverted-data); fi
"$python" -u -m experiments.mcb.suite "${suite_args[@]}"
