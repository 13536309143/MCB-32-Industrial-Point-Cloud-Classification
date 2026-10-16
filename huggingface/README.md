---
viewer: false
pretty_name: MCB-32 Paired Industrial Point Clouds
language:
- en
- zh
size_categories:
- 10K<n<100K
task_categories:
- other
tags:
- point-cloud
- 3d-classification
- mechanical-components
- cad
- pointnext
- mcb
---
# MCB-32 Paired Industrial Point Clouds

[English](README.md) | [简体中文](README.zh-CN.md) · [Code and reports](https://github.com/13536309143/MCB-32-Industrial-Point-Cloud-Classification)

## Dataset summary

A research preparation of Mechanical Components Benchmark (MCB) assets for **whole-object industrial component classification**: **50,531 unique objects, 32 component types, nine paired point cloud representations**. The nine versions are the same objects sampled differently; the 454,779 NPY files are not independent examples. This is a reorganized derivative, not an official MCB benchmark release.

The release contains area-weighted random, greedy farthest-point and Poisson-disk-style samples at **2,048, 4,096 and 8,192 points**. Each file preserves a contiguous NumPy `float32` array of shape `(N,3)`, XYZ only. No color, normals, physical units, sensor measurements or point-wise labels are included. Labels describe component shape and function, not defects or materials.

All **50,531 prepared source OBJ meshes** are also available as an optional, separately compressed `meshes/` collection. They preserve the same IDs, classes and partitions. Conversion sources and the Windows one-click wrapper are mirrored under `conversion/`; see [conversion instructions](https://huggingface.co/datasets/GeorgeWJJ/PointNext/blob/main/conversion/README.md).

![Nine paired representations of one object](assets/sampling_preview.png)

The same held-out gear is shown with a fixed camera and coordinate frame. This is a data illustration, not a classification success example. The tabular dataset viewer is disabled because this release contains original NPY files in TAR shards; use the verified downloader below.

## Files and partitions

| Quantity | Value |
| --- | --- |
| Unique objects | 50,531 |
| Classes | 32 |
| Original train / test | 40,424 / 10,107 |
| Recorded experiment train / val / test | 36,380 / 4,044 / 10,107 |
| Point cloud files | 454,779 |
| NPY payload bytes | 26,137,058,688 (24.34 GiB) |
| Representations | Three methods ×three point counts |
| Optional prepared OBJ meshes | 50,531 files; 8,909,142,490 bytes (8.30 GiB) |
| Mesh download | 18 gzip TAR shards; 2,654,025,959 bytes (2.47 GiB) |

Files are delivered as approximately 512 MiB **uncompressed TAR shards** to avoid hundreds of thousands of small Hub entries. Sharding preserves every NPY byte and path; TAR headers/padding add some overhead. `release_manifest.json` lists archive sizes, SHA-256 hashes, variant, original partition and file counts. `SHA256SUMS` also covers metadata and documentation. This is a file-based NumPy dataset, not a preconfigured Hugging Face `load_dataset()` table.

```text
archives/<variant>/train-00000.tar
archives/<variant>/test-00000.tar
metadata/class_to_idx.json
metadata/class_counts.json
metadata/experiment_splits.csv.gz
metadata/file_manifest.csv.gz
release_manifest.json
SHA256SUMS
meshes/<train-or-test>-<shard>.tar.gz  # optional original OBJ meshes
mesh_release_manifest.json
metadata/mesh_file_manifest.csv.gz
conversion/                          # converter, C++ source, Windows DLL and BAT
```

Each archive restores paths of the form:

```text
MCB_pointcloud/<variant>/train/<class>/<model_id>.npy
MCB_pointcloud/<variant>/test/<class>/<model_id>.npy
```

Variants: `area_random_2048`, `area_random_4096`, `area_random_8192`, `fps_2048`, `fps_4096`, `fps_8192`, `poisson_disk_2048`, `poisson_disk_4096`, `poisson_disk_8192`.

All nine representations share object IDs, labels and original partitions. The `model_id` is a stable 64-character hexadecimal identifier inherited from mesh preparation; it is **not** the SHA-256 of the present NPY file. Use the file manifest for content hashes.

The recorded experiment creates validation data **only from the original train partition**, stratified by class with split seed `20261013`; test objects are unchanged. The metadata CSV records each object's experiment `split` and original file `relative` path. Validation files remain physically under their original `train/` directory. Never repartition independently by representation, and never count different representations of one object as independent train/test examples. ID separation alone does not exclude near-duplicate component families.

## Download and read

From a recursive clone of the experiment repository:

```bash
python -m pip install huggingface_hub numpy
python scripts/download_dataset.py --repo-id GeorgeWJJ/PointNext --output data
```

The default Hub download cache retains the TAR shards in addition to the extracted NPY files; allow approximately 55 GiB of free disk for the full download/extraction, plus space for training outputs.

For a single representation, add `--variant area_random_2048`. The full experiment suite requires all nine. Supply `--revision <dataset-commit>` to pin the exact version. The downloader verifies archive SHA-256, rejects unsafe archive paths and restores the original directory structure. The default downloads only point clouds. Add `--include-meshes` to also restore the original `MCB_merged/`, or use `--meshes-only` to download just the OBJ collection. Mesh shards use lossless gzip compression; `mesh_release_manifest.json` and `metadata/mesh_file_manifest.csv.gz` provide archive and individual OBJ hashes. Allow extra cache and extraction space for meshes (about 8.3 GiB extracted, plus compressed downloads).

```python
from pathlib import Path
import numpy as np

path = next(Path("data/MCB_pointcloud/area_random_2048/train").glob("*/*.npy"))
xyz = np.load(path, allow_pickle=False)
assert xyz.dtype == np.float32 and xyz.shape == (2048, 3)
class_name = path.parent.name
```

## Geometry, sampling and reproducibility

The source mesh is centered by its **bounding-box center** and scaled by its largest vertex radius to fit a unit sphere, retaining its source orientation. Samples inherit that coordinate frame. They are not re-centered or re-scaled using their own finite sampled points; their bounding-box center need not be exactly zero, and maximum sampled radius need not be exactly one.

Each object first generates **40,960 area-weighted surface candidates**. Triangles are drawn proportional to area, then sampled using square-root barycentric weights. Intermediate calculations use float64; candidate uniqueness and saved coordinates use float32.

- **Area random:** take 8,192 candidates, with 4,096/2,048-point nested subsets. Surface density follows mesh area and may have local clusters.
- **FPS:** exact greedy Euclidean farthest-point selection from the shared candidate pool, starting at candidate 0. Select 8,192 points; prefixes form the smaller sets. A KD tree and indexed max-heap accelerate nearest-distance updates without changing the greedy rule.
- **Poisson disk:** Yuksel-style weighted sample elimination, producing approximately even spacing/blue noise. Eliminate 40,960→8,192→4,096→2,048 using updated target radii. This is not a guarantee of a strict global minimum separation. Parameters are alpha=8, beta=0.65, gamma=1.5; the full formula and implementation are in the linked source documentation.

For each method, the point sets satisfy `2048 ⊂ 4096 ⊂ 8192`. Each saved file has its own deterministic row shuffle, so row positions are not cross-resolution correspondences. Base seed 42 is combined with the original relative OBJ path and stream name through SHA-256; NumPy `default_rng` draws the candidates. Original generation used NumPy 2.5.4. See the code repository's `data/README.md` and `data/_pointcloud_converter/` for formulas and exact implementation. Other implementations with seed 42 need not generate identical points.

Before packaging, every NPY is checked against the **content hash inventory from the completed experiment**, and shape/dtype/finiteness are verified again. The original conversion also checked unique coordinates, source bounds, unit-sphere bounds and representative point-to-triangle surface distances. See the experiment repository for evidence and the limits of representative geometry checks.

## Classes and imbalance

| ID | Class | Original train | Test | Total |
| --- | --- | ---: | ---: | ---: |
| 0 | `bearing` | 1,064 | 266 | 1,330 |
| 1 | `bushing` | 567 | 142 | 709 |
| 2 | `castors_and_wheels` | 289 | 72 | 361 |
| 3 | `chain_drive` | 78 | 19 | 97 |
| 4 | `clamp` | 126 | 31 | 157 |
| 5 | `collar` | 38 | 10 | 48 |
| 6 | `disc` | 82 | 21 | 103 |
| 7 | `fitting` | 1,309 | 327 | 1,636 |
| 8 | `flange` | 5 | 1 | 6 |
| 9 | `gear` | 386 | 97 | 483 |
| 10 | `handles` | 1,291 | 323 | 1,614 |
| 11 | `hinge` | 49 | 12 | 61 |
| 12 | `hook` | 97 | 24 | 121 |
| 13 | `joint` | 1,342 | 336 | 1,678 |
| 14 | `key_and_spline` | 3,408 | 852 | 4,260 |
| 15 | `linear_guide` | 39 | 10 | 49 |
| 16 | `motor` | 594 | 148 | 742 |
| 17 | `nut` | 1,696 | 424 | 2,120 |
| 18 | `pin` | 5,979 | 1,495 | 7,474 |
| 19 | `plate` | 240 | 60 | 300 |
| 20 | `plug` | 135 | 34 | 169 |
| 21 | `pulley` | 245 | 61 | 306 |
| 22 | `ring` | 711 | 178 | 889 |
| 23 | `rivet` | 1,854 | 464 | 2,318 |
| 24 | `rotor` | 375 | 94 | 469 |
| 25 | `screws_and_bolts` | 13,224 | 3,306 | 16,530 |
| 26 | `spacer` | 87 | 22 | 109 |
| 27 | `spring` | 278 | 69 | 347 |
| 28 | `stud` | 2,542 | 636 | 3,178 |
| 29 | `switch` | 143 | 36 | 179 |
| 30 | `valve` | 6 | 1 | 7 |
| 31 | `washer` | 2,145 | 536 | 2,681 |

The largest class, `screws_and_bolts`, contains 16,530 objects (32.71%). `flange` and `valve` contain only 6 and 7 objects, with **one test object each**. Report macro F1, mean class recall, class support and confusion matrices together with overall accuracy. A rare-class score based on one object is not a stable estimate.

## Intended use and limitations

Suitable for CAD surface classification, paired sampling/resolution studies and controlled synthetic corruption experiments. Inputs are ideal complete surfaces, not real scans. The release does not establish robustness to arbitrary poses, outliers, occlusion, sensor artifacts or unseen CAD sources. Labels merge several fine subtypes into broad component categories. The 32-class taxonomy and retained counts are specific to this preparation.

The associated **completed** experiment trains six classifiers on `area_random_2048` only; the remaining variants are cross-representation tests on the same held-out objects. It is not a nine-training-dataset optimization study and has only one training seed. The [English report](https://github.com/13536309143/MCB-32-Industrial-Point-Cloud-Classification/blob/main/docs/reports/KEY6.md) and [Chinese report](https://github.com/13536309143/MCB-32-Industrial-Point-Cloud-Classification/blob/main/docs/reports/KEY6.zh-CN.md) provide measured results.

## Provenance, rights and citation

Original source: [stnoah1/mcb](https://github.com/stnoah1/mcb), introduced by Kim et al., ECCV 2020. MCB tooling collects assets from online CAD repositories. This release reorganizes labels and derives point samples from prepared meshes; it does not claim authorship of the original CAD designs.

**No new dataset-wide license is asserted.** The upstream MCB tooling has an MIT software license; that alone does not establish an MIT license for every CAD asset. Source asset rights and distribution terms continue to apply to derived point clouds. The experiment code's MIT license must not be interpreted as a blanket commercial or redistribution permission for this dataset. Users should check the applicable original terms for their intended use. See `THIRD_PARTY_NOTICES.md`.

```bibtex
@inproceedings{sangpil2020large,
  title={A Large-scale Annotated Mechanical Components Benchmark for Classification and Retrieval Tasks with Deep Neural Networks},
  author={Kim, Sangpil and Chi, Hyung-gun and Hu, Xiao and Huang, Qixing and Ramani, Karthik},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

Dataset preparation and experiment code: [13536309143/MCB-32-Industrial-Point-Cloud-Classification](https://github.com/13536309143/MCB-32-Industrial-Point-Cloud-Classification). This Hub repository is a derivative research distribution, not an endorsement by the original MCB or PointNeXt authors.
