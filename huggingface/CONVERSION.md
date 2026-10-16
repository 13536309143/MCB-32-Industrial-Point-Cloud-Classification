# From source meshes to the nine point cloud representations

The optional `meshes/` archives preserve all 50,531 prepared OBJ files, paired with the point clouds by original partition, class and model ID. They restore `MCB_merged/train/<class>/<id>.obj` and `MCB_merged/test/<class>/<id>.obj`. The meshes are already normalized; they are the prepared conversion inputs, not the unprocessed upstream A/B download. `metadata/mesh_file_manifest.csv.gz` records every original OBJ SHA-256.

The complete experiment repository is https://github.com/13536309143/MCB-32-Industrial-Point-Cloud-Classification . Its `data/convert_mcb_9sets.bat` and `data/_pointcloud_converter/` are also mirrored in this `conversion/` directory. The conversion software license is provided in `conversion/LICENSE`; it applies to the software, not the dataset. Source CAD rights remain subject to the notices at the dataset root.

Download meshes with `python scripts/download_dataset.py --repo-id GeorgeWJJ/PointNext --meshes-only --output data` from a clone of the experiment repository. To download both point clouds and meshes, use `--include-meshes` instead. Add `--revision <dataset-commit>` for reproducibility.

On Windows x64, arrange these entries beside each other, then double-click the BAT:

```text
data/
  MCB_merged/train/...
  MCB_merged/test/...
  convert_mcb_9sets.bat
  _pointcloud_converter/
    bootstrap.ps1
    convert_mcb.py
    sampling.cpp
    sampling_windows_x64.dll
```

The BAT prepares a local Python environment, checks prerequisites and calls the converter. The script writes `MCB_pointcloud/<method>_<count>/<train-or-test>/<class>/<id>.npy`, with area random, FPS and Poisson-disk-style sampling at 2,048/4,096/8,192 points, XYZ float32. It supports resumable checkpoints and does not change the source OBJ files. For converter options, run the Python converter with `--help`; the double-click BAT uses the default full conversion. The Windows wrapper is prepared but has not been validated on Windows hardware in the reported experiment.

The C++ sampling source is provided alongside the Windows DLL. On macOS/Linux, build the native library and specify it explicitly, for example:

```bash
python -m pip install numpy==2.5.4
# Linux; on macOS replace g++ with clang++ and -shared with -dynamiclib.
g++ -O3 -std=c++17 -shared -fPIC -ffp-contract=off data/_pointcloud_converter/sampling.cpp -o data/_pointcloud_converter/libsampling.so
python data/_pointcloud_converter/convert_mcb.py --source data/MCB_merged --output data/MCB_pointcloud --library data/_pointcloud_converter/libsampling.so --limit 1
# Remove --limit 1 to convert the full collection after checking the small run.
```

The exact protocol, fixed seed streams, class counts and geometry checks are described in `README.zh-CN.md` and the repository's `data/README.md`. Cross-platform floating-point/compiler differences can affect exact sampled coordinates. Download the published NPYs when byte identity with the completed experiment is required.
