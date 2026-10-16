# MCB-32：工业零部件点云分类

**PointNeXt-S/B/L/XL · PointNet · PointNet++ · 九种配对点云表示**

[English](README.md) | [简体中文](README.zh-CN.md) · [完整实验报告](docs/reports/KEY6.zh-CN.md) · [指标与核验记录](reports/key6/) · [数据集说明](huggingface/README.zh-CN.md)

本项目面向工业零部件的整件三维类型识别，以整理后的 **32 类、50,531 个 MCB 模型**为基础，构建三种表面采样方法、三档点数的九套点云表示，并比较六种点云分类网络。项目通过固定提交的 Git 子模块保留完整 PointNeXt/OpenPoints，提供网格转换、CUDA 适配、顺序训练、中断恢复、鲁棒性测试和结构化结果记录。

> **已完成的实验：**6 次完整数据训练，每次 300 轮，共 192 项测试。六个模型均使用**面积随机采样 2,048 点、seed=42**训练；其他八种点云只用于跨表示测试。九套训练数据、多种子与参数搜索是已编写的计划，**尚未完整运行**。

![分类指标与推理延迟](docs/assets/classification_and_latency.png)

## 主要结果

以下指标使用相同的 10,107 个独立留出模型，检查点依据验证集 Macro F1 选择。OA 为总体准确率，mAcc 为各类召回率的平均。

| 模型 | OA ↑ | mAcc ↑ | Macro F1 ↑ | 参数量 | 训练＋验证 | 单模型前向中位延迟 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **PointNeXt-S** | **97.71%** | 81.70% | **83.03%** | 1.371 M | 6.39 小时 | 3.531 ms |
| PointNeXt-B | 96.96% | 80.09% | 80.75% | 3.879 M | 3.64 小时 | 5.572 ms |
| PointNeXt-L | 97.02% | 80.25% | 80.65% | 7.176 M | 5.91 小时 | 8.653 ms |
| PointNeXt-XL | 97.23% | **81.72%** | 82.03% | 40.840 M | 17.58 小时 | 5.727 ms |
| PointNet | 95.72% | 74.98% | 75.40% | 3.477 M | 2.57 小时 | **1.515 ms** |
| PointNet++ | 97.39% | 80.77% | 81.39% | 1.470 M | 6.37 小时 | 2.293 ms |

实测环境为 **RTX 4070 12 GB、Linux、PyTorch 2.7.1/cu126、BF16**。训练时间为各轮训练和验证时间之和，不含编译、检查点写入、测试和中断等待；延迟只测 GPU 内驻留输入的前向传播，不代表部署端到端耗时。仅一个种子，未估计训练波动。

**如何解读：**当前配置下，S 的干净 OA 和 Macro F1 最高；PointNet++ 单模型推理更快，对轻度离群点和严重遮挡更稳。更大的 PointNeXt 没有带来更高的干净 OA。所有网络在任意姿态和离群点下都有明显下降，小类别表现也显著低于总体准确率。详见[完整报告](docs/reports/KEY6.zh-CN.md)。

## 数据集

数据由 [MCB](https://github.com/stnoah1/mcb) 模型合并、去重并整理为零部件类型标签。类别同时涉及形状和机械用途；任务是**整件分类**，不包含逐点分割、材质识别或缺陷检测。

| 项目 | 内容 |
| --- | --- |
| 独立模型 / 类别 | 50,531 / 32 |
| 分发版 train / test | 40,424 / 10,107 |
| 实验 train / val / test | 36,380 / 4,044 / 10,107 |
| 采样方法 | 按面积随机、贪心 FPS、泊松盘式样本消除 |
| 点数 | 2,048、4,096、8,192 |
| 文件 | `.npy`，连续存储 `float32`，`(N,3)`，仅 XYZ |
| 配对关系 | 九套版本的模型 ID、类别和原始划分一致 |
| 坐标 | 源网格包围盒中心归零、最大顶点半径为 1，保留原方向 |
| 点云总量 | 454,779 个 NPY，逻辑大小 26,137,058,688 字节 |

九套点云属于**同一批模型的不同表示**，不能当作 454,779 个独立样本重新划分。同一 ID 的所有表示必须留在同一集合。同一采样方法的三档点数满足集合嵌套关系，但保存时独立打乱行顺序；采样后不再单独居中或缩放。

Hugging Face 发布准备采用带校验和的 TAR 分片，保留原始 NPY 字节，避免创建数十万个仓库条目。[数据卡](huggingface/README.zh-CN.md)说明类别、采样算法、文件结构和许可范围。目标数据仓库为 [GeorgeWJJ/PointNext](https://huggingface.co/datasets/GeorgeWJJ/PointNext)，分片可用性和哈希以仓库发布清单为准。

```bash
python -m pip install huggingface_hub numpy
python scripts/download_dataset.py --repo-id GeorgeWJJ/PointNext --output data
```

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

验证集仅从原训练集中按类划出，split seed 为 `20261013`，测试集不参与模型选择。数据及划分指纹见 [data_binding.json](reports/key6/protocol/data_binding.json)。

![同一留出零件的九套配对点云](docs/assets/sampling_preview.png)

图中使用同一个留出 gear 模型和固定视角，行表示采样方法，列表示点数；这里只展示数据对应关系，不代表分类结果。下载数据后可用 `python scripts/plot_sampling_preview.py` 重建。

## 获取完整项目

```bash
git -c url.https://github.com/.insteadOf=git@github.com: clone --recurse-submodules https://github.com/13536309143/pointnext.git
cd pointnext
```

临时 URL 重写用于处理上游 OpenPoints 的 SSH 子模块地址，无需配置 SSH 密钥。GitHub 下载 ZIP 不会自动包含子模块。

| 依赖 | 固定提交 |
| --- | --- |
| PointNeXt | `b7ec10fd8b9acdb1c3b18e557f0d3465eb919164` |
| OpenPoints | `db31e0d94ede35bc672ef06d1b6e4073a9c269d5` |

S 沿用官方 ModelNet 分类编码器。B/L/XL 保留官方 S3DIS 编码器层数、宽度与下采样结构，再接分类头；它们是本项目的分类适配，不能当作官方分类权重。PointNet 保留输入和特征变换，PointNet++ 使用原始 SSG 编码器。完整模型规格包含在[指标归档](reports/key6/metrics-and-protocol.zip)中。

## Linux / CUDA 运行

需要可用 NVIDIA 驱动、与 PyTorch CUDA 版本相匹配的 CUDA Toolkit 编译器，以及兼容的 C++ 编译器。实测版本为 Python 3.11、PyTorch 2.7.1/cu126、CUDA Toolkit 12.6。托管 GPU 环境无需为了这套配置替换已有驱动。

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
# 六模型、完整数据、每个 300 轮：
python -m experiments.mcb.suite --profile key6 --no-pack
```

先将全部九套点云解压到 `data/MCB_pointcloud/`。套件检查 ID/类别/划分、形状、类型、有限数值和内容指纹；CUDA 预检覆盖六种模型的前向、反向。`--no-pack` 避免重复创建整套点云缓存。[CLOUD.md](CLOUD.md)另有已有环境的云端启动方法。

支持 BF16 时优先使用 BF16，否则使用 FP16；自动探测实际 batch 并用梯度累积维持有效 batch。本次六个模型的实际和有效 batch 均为 32，没有缩小模型或输入点数。

## Windows 运行

预备流程为 **Windows → WSL2 Ubuntu → CUDA**，先运行：

```bat
run_experiments.bat smoke
```

首次安装可能需要管理员权限或重启。Windows 脚本为 RTX 4090 等设备准备 PyTorch 2.8.0/cu128；**本次结果来自 Linux/RTX 4070**，不能当作 Windows 的实测结果。原生 Windows CUDA 和 WSL 安装流程在本次发布中未经过实际硬件验证。

BAT 当前接受 `smoke`、`quick`、`full`。运行 `key6` 时，在准备好的 WSL CUDA 环境使用上面的 Linux 命令。详细操作见[原操作指南](docs/OPERATIONS.zh-CN.md)。

## 实验计划与完成范围

| 模式 | 训练次数 | 测试项数 | 状态与用途 |
| --- | ---: | ---: | --- |
| `smoke` | 6 | 192 | 每模型 1 轮、限制样本，检查流程 |
| **`key6`** | **6** | **192** | **已完成**：统一配置、完整数据 |
| `quick` | 54 | 1,728 | 待运行：六模型 ×九套训练表示 |
| `full` | 1,296 | 11,898 | 待运行：多种子、参数、交互、数据效率与鲁棒性 |

完整计划共 399,600 轮，需要大量算力和存储。`quick` 只是实验网格较小，仍使用完整选定训练数据。启动前可先生成并检查计划：

```bash
python -m experiments.mcb.plan --profile full --output /tmp/mcb-full-plan.json
```

## 中断恢复、结果与预测

实验依次在独立进程运行，每轮原子保存指标与 `last.pt`，`best.pt` 只依验证指标选择。使用**相同代码、配置和数据重新运行原命令**可恢复。完成的制品通过清单校验；CUDA 原子操作仍可能导致多次运行无法逐位相同。

训练加载器只创建并复用一组常驻工作进程；验证与测试进程结束后释放，已修复本实验发现的文件句柄泄漏。配置和数据绑定用于避免混用不同实验。

```text
results/key6/
  plan.json  status.json  results.csv  environment_initial.json
  _data/                         # 类别映射、准确划分、清单及指纹
  runs/<run_id>/
    config.json  model_spec.json  history.csv  epochs/
    best.pt  last.pt  best_checkpoint.json  training_complete.json
    benchmark.json  tests/<case>.json  tests/<case>.npz  complete.json
```

[公开结果快照](reports/key6/)包含完整指标 JSON 和训练历史；权重、逐模型概率 NPZ 保留在本地完整备份，**未托管于本 Git 仓库**。训练完成或恢复可信运行目录后，可预测自己的标准点云：

```bash
python -m experiments.mcb.predict --run-dir results/key6/runs/pointnext_s__area_random_2048__s42__648d97b274ce --input path/to/prepared_2048_point_cloud.npy --output predictions.csv
```

需要同一 CUDA 算子环境，并保留 `results/key6/_data/index.json`。当前单文件预测接口要求该 S 模型输入为 `(2048,3)`；跨分辨率评估由套件实现。新模型应遵循相同源网格归一化与采样约定。

## 报告与复现材料

- [中文实验报告](docs/reports/KEY6.zh-CN.md) / [English report](docs/reports/KEY6.md)。
- [192 项测试](reports/key6/tables/all_192_tests.csv)、[逐类指标](reports/key6/tables/per_class_clean.csv)、[九种输入表示测试](reports/key6/tables/cross_representation_54.csv)。
- [学习曲线](docs/assets/training_curves.png)、[鲁棒性](docs/assets/robustness_oa.png)、[各类 F1](docs/assets/per_class_f1.png)，同时提供 PNG/SVG。
- [核验记录](reports/key6/protocol/backup_verification.json)：2,283 个备份文件与服务器 SHA-256 一致，重新核对 1,940,544 条已保存预测。

公开的 CSV/JSON 归档可在无 GPU 条件下重建指标表和学习曲线。图表构建及归档校验程序为 `scripts/build_publication.py`。

## 适用边界

本项目是**单种子 CAD 分类基线**，不宣称达到最先进水平。限制包括类别不平衡、法兰和阀各仅一个测试对象、ID 去重不能排除同系列近似几何、B/L/XL 的分类适配，以及缺少真实传感器域评估。跨表示测试不等于跨数据集泛化。

后续应优先补充小类、比较九种训练输入、增加训练种子、做旋转/噪声/离群点增强消融，并使用独立真实扫描模型评估。这些额外实验尚未在本次发布中运行。

## 来源与引用

基于 [MCB](https://github.com/stnoah1/mcb)、[PointNeXt](https://github.com/guochengqian/PointNeXt)、[PointNet](https://arxiv.org/abs/1612.00593) 和 [PointNet++](https://arxiv.org/abs/1706.02413)。参见 [CITATION.cff](CITATION.cff)、[参考文献](docs/references.bib)和[第三方声明](THIRD_PARTY_NOTICES.md)。

本项目原创代码和文档采用 [MIT](LICENSE)。上游代码保留各自许可。**原始 CAD 和派生点云不适用本项目代码的 MIT 重新授权**，应遵守原始资产及来源条款；数据卡不声称已经确认统一的数据集许可。
