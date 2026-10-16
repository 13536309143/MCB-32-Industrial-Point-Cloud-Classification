#!/usr/bin/env bash
# Invoked as root inside the named Ubuntu WSL2 distribution by bootstrap_windows.ps1.
set -euo pipefail
project_root="${1:?Project path is required}"
profile="${2:-full}"
reconverted="${3:-0}"
case "$profile" in full|quick|smoke) ;; *) echo "Unknown profile: $profile"; exit 1 ;; esac
if [[ $(id -u) != 0 ]]; then echo 'WSL bootstrap must be started by the BAT as root.'; exit 1; fi
export DEBIAN_FRONTEND=noninteractive
export PATH="/usr/lib/wsl/lib:$PATH"
if ! nvidia-smi; then
  echo 'Windows NVIDIA driver / WSL CUDA is unavailable. Update the NVIDIA Windows driver, run wsl --update, and retry.'
  echo 'Do not install a Linux NVIDIA display driver inside WSL.'
  exit 1
fi
if ! getent passwd mcb-runner >/dev/null; then
  useradd --create-home --shell /bin/bash mcb-runner
fi
runtime=/home/mcb-runner/.cache/mcb-pointnext/runtime-v1
mkdir -p "$runtime"
chown mcb-runner:mcb-runner "$runtime" /home/mcb-runner/.cache /home/mcb-runner/.cache/mcb-pointnext
exec 9>"$runtime/bootstrap.lock"
flock -n 9 || { echo 'Another MCB bootstrap/suite is running in this Ubuntu distribution.'; exit 1; }
if ! command -v g++ >/dev/null || ! command -v curl >/dev/null || ! command -v git >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl build-essential git python3-dev unzip
fi
if [[ ! -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  echo 'Installing CUDA Toolkit 12.8 compiler/runtime only (no Linux display driver)...'
  keyring="$runtime/cuda-keyring.deb"
  curl --fail --location --retry 3 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb' -o "$keyring"
  dpkg -i "$keyring"
  rm "$keyring"
  apt-get update
  apt-get install -y --no-install-recommends cuda-toolkit-12-8
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4
if [[ ! -x "$runtime/uv" ]]; then
  archive="$runtime/uv.tar.gz"
  curl --fail --location --retry 3 'https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz' -o "$archive"
  echo "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154  $archive" | sha256sum --check -
  tar -xzf "$archive" -C "$runtime" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
  rm "$archive"
  chown mcb-runner:mcb-runner "$runtime/uv"
fi
export UV_CACHE_DIR="$runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$runtime/python"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# All experiment code, model builds and Python packages run as an ordinary Linux user.
runuser -u mcb-runner -- env CUDA_HOME="$CUDA_HOME" PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
  UV_CACHE_DIR="$UV_CACHE_DIR" UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
  PYTHONUTF8=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 bash "$project_root/experiments/run_linux.sh" "$project_root" "$profile" "$runtime" "$reconverted"
