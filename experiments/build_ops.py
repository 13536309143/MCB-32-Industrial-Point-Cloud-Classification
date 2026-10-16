"""Build the official classification kernels on Windows or Linux, without editing upstream."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import torch
from torch.utils.cpp_extension import load, CUDA_HOME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install/update NVIDIA Windows driver and use a CUDA PyTorch wheel.')
    if CUDA_HOME is None:
        raise RuntimeError('CUDA Toolkit with nvcc is required (12.8 for the pinned cu128 environment).')
    root = Path(__file__).resolve().parents[1]
    source = root / 'PointNeXt/openpoints/cpp/pointnet2_batch/src'
    output = Path(args.output).resolve()
    build = output / 'build'
    copied = output / 'source'
    build.mkdir(parents=True, exist_ok=True)
    copied.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for path in sorted(source.iterdir()):
        if path.suffix not in ('.h', '.cpp', '.cu'):
            continue
        raw = path.read_bytes()
        inventory[path.name] = hashlib.sha256(raw).hexdigest()
        text = raw.decode('utf-8')
        # Compatible modern tensor API; retain every CUDA kernel/algorithm unchanged.
        text = re.sub(r'(\w+)\.type\(\)\.is_cuda\(\)', r'\1.is_cuda()', text)
        text = re.sub(r'\.data<(float|int)>\(\)', r'.data_ptr<\1>()', text)
        target = copied / path.name
        if not target.exists() or target.read_text(encoding='utf-8') != text:
            target.write_text(text, encoding='utf-8', newline='\n')
    capability = torch.cuda.get_device_capability(0)
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{capability[0]}.{capability[1]}'
    os.environ.setdefault('MAX_JOBS', '2')
    # Minimal cloud toolkits may omit cuBLAS/cuSPARSE/cuSOLVER development headers.
    # CUDA PyTorch wheels already provide matching headers in their NVIDIA packages.
    nvidia_packages = Path(torch.__file__).resolve().parent.parent / 'nvidia'
    wheel_headers = [str(p) for p in sorted(nvidia_packages.glob('*/include')) if p.is_dir()]
    module = load(name='mcb_pointnet2_cuda', sources=[str(p) for p in sorted(copied.iterdir())
                                                    if p.suffix in ('.cpp', '.cu')],
                  build_directory=str(build), extra_cflags=['/O2'] if os.name == 'nt' else ['-O2'],
                  extra_cuda_cflags=['-O2'], extra_include_paths=wheel_headers, verbose=True)
    artifact = Path(module.__file__)
    # Make import independent of torch's JIT cache and this machine's source location.
    import shutil
    shutil.copy2(artifact, output / artifact.name)
    (output / 'build_metadata.json').write_text(json.dumps(dict(
        torch=torch.__version__, torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        capability=capability, cuda_home=CUDA_HOME, source_sha256=inventory,
        pytorch_wheel_header_paths=wheel_headers,
        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        adaptations=['Tensor.is_cuda(), Tensor.data_ptr()', 'MSVC /O2 flag on Windows']), indent=2), encoding='utf-8')
    print('Official CUDA kernels built:', output / artifact.name)


if __name__ == '__main__':
    main()
