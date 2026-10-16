# MCB-32: Industrial Point Cloud Classification

**PointNeXt-S/B/L/XL · PointNet · PointNet++ · Nine paired point cloud representations**

[English](README.md) | [简体中文](README.zh-CN.md) · [Experiment report](docs/reports/KEY6.md) · [Results & audit files](reports/key6/) · [Dataset card](huggingface/README.md)

A reproducible research workflow for classifying complete industrial components from XYZ point clouds. It combines a reorganized **32-class, 50,531-object MCB collection**, three surface sampling methods at three resolutions, and six point-based classifiers. The repository preserves the complete official PointNeXt/OpenPoints sources through pinned submodules and adds dataset conversion, CUDA compatibility, sequential training, checkpoint recovery, evaluation and structured result collection.

> **Measured release:** six completed training runs, 300 epochs each, and 192 full-test-set evaluations. Every model was trained on **area-random 2,048-point** inputs with seed 42. The other eight representations are evaluated without retraining. The larger nine-dataset and hyperparameter studies are executable plans, **not completed results**.

![Classification accuracy and inference latency](docs/assets/classification_and_latency.png)

## Results at a glance

Scores below use the same 10,107 held-out objects and the checkpoint selected by validation macro F1. OA is overall accuracy; mAcc is mean class recall.

| Model | OA ↑ | mAcc ↑ | Macro F1 ↑ | Parameters | Train + val | Batch-1 forward p50 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PointNeXt-S** | **97.71%** | 81.70% | **83.03%** | 1.371 M | 6.39 h | 3.531 ms |
| PointNeXt-B | 96.96% | 80.09% | 80.75% | 3.879 M | 3.64 h | 5.572 ms |
| PointNeXt-L | 97.02% | 80.25% | 80.65% | 7.176 M | 5.91 h | 8.653 ms |
| PointNeXt-XL | 97.23% | **81.72%** | 82.03% | 40.840 M | 17.58 h | 5.727 ms |
| PointNet | 95.72% | 74.98% | 75.40% | 3.477 M | 2.57 h | **1.515 ms** |
| PointNet++ | 97.39% | 80.77% | 81.39% | 1.470 M | 6.37 h | 2.293 ms |

Measured on **one RTX 4070 12 GB**, Linux, PyTorch 2.7.1/cu126 and BF16. Training time sums epoch training and validation; it excludes compilation, checkpoint writes, testing and interruption time. Latency measures GPU-resident forward passes, not end-to-end deployment. Single-seed results do not estimate training variance.

**What the experiment shows.** S has the strongest clean OA and macro F1 under this recipe. PointNet++ offers lower single-object latency and better performance under light outliers and heavy occlusion. Larger PointNeXt variants do not improve clean OA here. All models remain sensitive to arbitrary rotations and outliers, and rare-class performance is substantially below aggregate accuracy. See the [full report](docs/reports/KEY6.md) for effect sizes, learning curves, per-class results and limitations.

## Dataset

The derivative collection merges and reorganizes [MCB](https://github.com/stnoah1/mcb) assets into component-type labels. These labels reflect both shape and mechanical function; this is **whole-object classification**, not point-wise segmentation, material recognition or defect detection.

| Property | Value |
| --- | --- |
| Unique objects / classes | 50,531 / 32 |
| Distributed train / test | 40,424 / 10,107 |
| Experiment train / validation / test | 36,380 / 4,044 / 10,107 |
| Sampling | Area-weighted random, greedy FPS, Poisson-disk-style sample elimination |
| Point counts | 2,048 · 4,096 · 8,192 |
| File representation | NumPy `.npy`, contiguous `float32`, shape `(N, 3)`, XYZ only |
| Correspondence | Identical object IDs, class labels and partitions across all nine variants |
| Coordinate frame | Source mesh bounding-box center; unit maximum vertex radius; original orientation retained |
| Point cloud payload | 454,779 files; 26,137,058,688 bytes before archive framing |

These are nine views of the **same objects**, not 454,779 independent examples. Keep object identities in one partition across all representations. Point counts are nested as sets within each method; saved row order is independently shuffled. Point cloud samples are not re-centered or re-scaled after surface sampling.

The Hugging Face release is prepared as checksummed TAR shards to preserve the original NPY files without creating hundreds of thousands of repository entries. The [dataset card](huggingface/README.md) documents the exact sampling implementation, all 32 classes, layout and licensing scope. Dataset repository: [GeorgeWJJ/PointNext](https://huggingface.co/datasets/GeorgeWJJ/PointNext). Archive availability and content hashes are recorded in its release manifest.

```text
data/MCB_pointcloud/
  area_random_2048/  area_random_4096/  area_random_8192/
  fps_2048/          fps_4096/          fps_8192/
  poisson_disk_2048/ poisson_disk_4096/ poisson_disk_8192/
    train/<class>/<model_id>.npy
    test/<class>/<model_id>.npy
```

```python
from pathlib import Path
import numpy as np

path = next(Path("data/MCB_pointcloud/area_random_2048/train").glob("*/*.npy"))
xyz = np.load(path, allow_pickle=False)  # float32, (2048, 3)
label_name = path.parent.name
```

For the completed run, validation is derived only from the original training partition using split seed `20261013`. Model selection never uses test scores. Exact fingerprints are versioned in [data_binding.json](reports/key6/protocol/data_binding.json).

## Get the complete project

```bash
git -c url.https://github.com/.insteadOf=git@github.com: clone --recurse-submodules https://github.com/13536309143/pointnext.git
cd pointnext
```

The temporary URL rewrite handles the upstream OpenPoints SSH submodule URL without requiring an SSH key. Downloading GitHub's source ZIP does not populate submodules.

| Dependency | Recorded commit |
| --- | --- |
| PointNeXt | `b7ec10fd8b9acdb1c3b18e557f0d3465eb919164` |
| OpenPoints | `db31e0d94ede35bc672ef06d1b6e4073a9c269d5` |

S follows the upstream ModelNet classification encoder. B/L/XL retain their complete upstream S3DIS encoder depth/width/downsampling schedules and attach an object classification head. These are explicit classification adaptations, not official B/L/XL classification checkpoints. PointNet retains its input/feature transforms; PointNet++ uses the original SSG encoder. Detailed specs are in the [recorded artifact archive](reports/key6/metrics-and-protocol.zip).

## Run on Linux with CUDA

A supported NVIDIA driver, CUDA toolkit/compiler matching the PyTorch CUDA build, and a compatible C++ compiler are required. The **measured environment** used Python 3.11, PyTorch 2.7.1/cu126 and CUDA toolkit 12.6. Do not replace the NVIDIA driver supplied by a managed GPU service just to match this recipe.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r scripts/requirements-cloud.txt

export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/.cloud-runtime/ops:$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -n 65536

python experiments/build_ops.py --output .cloud-runtime/ops
python -m experiments.mcb.preflight
python -m experiments.mcb.suite --profile smoke --no-pack
# Full-data six-model comparison, 300 epochs per model:
python -m experiments.mcb.suite --profile key6 --no-pack
```

Put all nine extracted variants under `data/MCB_pointcloud/` first. The suite validates identity/label/split consistency, array shape, dtype, finiteness and content fingerprints. GPU forward/backward preflight checks cover all six models. `--no-pack` avoids a second full copy of the dataset. A cached Linux cloud launcher is also available; see [CLOUD.md](CLOUD.md).

BF16 is preferred when supported, with FP16 otherwise. Physical batch is probed against available GPU memory and gradients are accumulated to the requested effective batch. The completed run used physical/effective batch 32 for every model; no model or point-count reduction was applied.

## Run from Windows

The prepared launcher uses **Windows → WSL2 Ubuntu → CUDA**. Start with:

```bat
run_experiments.bat smoke
```

The first run can install WSL2 and require administrator access or a reboot. The Windows bootstrap targets an RTX 4090-class setup with PyTorch 2.8.0/cu128; it is a **prepared workflow**, whereas the published measurements were obtained on Linux/RTX 4070. Native Windows CUDA training and the WSL bootstrap were not hardware-validated in this release.

The BAT currently accepts `smoke`, `quick`, and `full`. For `key6`, use the Linux commands above inside a prepared WSL CUDA environment. See the [detailed Chinese operations guide](docs/OPERATIONS.zh-CN.md).

## Experiment profiles

| Profile | Training runs | Test cases | Status / purpose |
| --- | ---: | ---: | --- |
| `smoke` | 6 | 192 | One epoch and limited objects; pipeline check |
| **`key6`** | **6** | **192** | **Completed:** common-recipe full-data comparison |
| `quick` | 54 | 1,728 | Planned: six models ×nine training representations |
| `full` | 1,296 | 11,898 | Planned: seeds, parameters, interactions, data efficiency and robustness |

`full` schedules 399,600 training epochs and requires substantial compute/storage. Both `quick` and `full` use the entire selected training data; the name `quick` refers to a smaller experiment grid, not a toy dataset. Review the plan before launching:

```bash
python -m experiments.mcb.plan --profile full --output /tmp/mcb-full-plan.json
```

## Recovery, artifacts and inference

Runs are sequential and isolated. Each epoch atomically saves its metrics and a resumable `last.pt`; `best.pt` is selected using validation only. Restart the **same command with the same source/configuration/data** to resume. Completed artifacts are checked against their manifests. CUDA atomic operations can still prevent bitwise identical repeated GPU runs.

The training loader is reused with persistent workers; short-lived validation/test loaders release their workers, preventing the file-descriptor leak found and fixed during this experiment. Source/config/data bindings guard against silently mixing incompatible runs.

```text
results/key6/
  plan.json  status.json  results.csv  environment_initial.json
  _data/                     # class map, exact splits, inventory and hashes
  runs/<run_id>/
    config.json  model_spec.json  history.csv  epochs/
    best.pt  last.pt  best_checkpoint.json  training_complete.json
    benchmark.json  tests/<case>.json  tests/<case>.npz  complete.json
```

The public [results snapshot](reports/key6/) includes complete metric JSONs and histories; checkpoint/probability binaries remain in the local backup and are **not** hosted in this Git repository. After training or restoring your own trusted run folder:

```bash
python -m experiments.mcb.predict   --run-dir results/key6/runs/pointnext_s__area_random_2048__s42__648d97b274ce   --input path/to/prepared_2048_point_cloud.npy   --output predictions.csv
```

Use the same CUDA operator environment and keep `results/key6/_data/index.json`. Standalone inference currently requires exactly `(2048,3)` for this S run. Cross-resolution evaluation is implemented in the suite. New objects must follow the documented source-mesh normalization and sampling convention.

## Reports and reproducibility

- [English experimental report](docs/reports/KEY6.md) / [中文实验报告](docs/reports/KEY6.zh-CN.md).
- [All 192 evaluations](reports/key6/tables/all_192_tests.csv), [per-class scores](reports/key6/tables/per_class_clean.csv), [nine-representation tests](reports/key6/tables/cross_representation_54.csv).
- [Learning curves](docs/assets/training_curves.png), [robustness heatmap](docs/assets/robustness_oa.png), [per-class F1](docs/assets/per_class_f1.png); PNG and SVG sources are included.
- [Validation evidence](reports/key6/protocol/backup_verification.json): 2,283 backup files matched server SHA-256; 1,940,544 saved predictions were checked against their recorded metrics.

The published CSV/JSON archive can reproduce the metric tables and learning curves without a GPU. The figure builder and archive-integrity checker are in `scripts/build_publication.py`.

## Limits and intended use

This is a **single-seed CAD classification baseline**, not a state-of-the-art claim. Important limits include severe class imbalance, one test object each for flange/valve, potential family-level geometric similarity not ruled out by ID-based deduplication, classification adaptations for B/L/XL, and no real-sensor domain validation. Cross-representation tests preserve held-out identities and do not establish unseen-dataset generalization.

The most useful next steps are rare-class data collection, a nine-training-representation comparison, multiple training seeds, rotation/noise/outlier augmentation ablations, and a genuinely independent scanned-object test set. These additional experiments have not been run in this release.

## Attribution and citation

This project builds on [MCB](https://github.com/stnoah1/mcb), [PointNeXt](https://github.com/guochengqian/PointNeXt), [PointNet](https://arxiv.org/abs/1612.00593) and [PointNet++](https://arxiv.org/abs/1706.02413). See [CITATION.cff](CITATION.cff), [references](docs/references.bib) and [third-party notices](THIRD_PARTY_NOTICES.md).

Original harness and documentation: [MIT](LICENSE). Upstream submodules retain their own notices. **The source CAD assets and point clouds are not relicensed under the code's MIT license.** Original asset/source terms apply; the dataset card does not assert a newly verified dataset-wide license.
