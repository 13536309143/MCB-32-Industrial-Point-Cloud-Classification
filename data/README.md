# MCB 机械零部件点云数据集

数据版本：2026-10-13。本地数据包含 **32 个机械零部件粗类别、50,531 个模型，以及 9 套点云采样版本**，适合 PointNeXt 等模型的整件点云分类和形状检索。

数据来源：[MCB 项目仓库](https://github.com/stnoah1/mcb)、[Kim 等人的 ECCV 2020 论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630171.pdf)。当前类别、模型数量和划分属于本地整理版。

## 1. 可用数据

- `MCB_merged/`：50,531 个已整理的源 OBJ 网格。
- `MCB_pointcloud/`：三种采样方法 × 三种点数，共 9 套 NPY 点云版本。
- 每套点云保留相同的 32 个类别、模型编号和 train/test 划分。
- 每套都有 **训练集 40,424 个、测试集 10,107 个模型**。
- 共 **454,779 个 NPY 文件**，逻辑大小约 **24.34 GiB**。

9 套点云是同一批模型的不同采样表示，类别数量始终为 32。它们不应作为九批独立模型重新随机划分；实验应沿用现有划分，避免同一零件的不同表示跨训练与测试。

## 2. 九套点云规格

所有 NPY 文件都保存为 **`float32`、`(N, 3)`、仅含 XYZ 坐标**，无额外的标签列、法线或颜色列。类别由父目录给出。

| 版本目录 | 采样方法 | 数组形状 | train | test | NPY 逻辑大小 |
| --- | --- | --- | ---: | ---: | ---: |
| `area_random_2048` | 按面积随机采样 | `2,048 × 3` | 40,424 | 10,107 | 1.16 GiB |
| `area_random_4096` | 按面积随机采样 | `4,096 × 3` | 40,424 | 10,107 | 2.32 GiB |
| `area_random_8192` | 按面积随机采样 | `8,192 × 3` | 40,424 | 10,107 | 4.63 GiB |
| `fps_2048` | 最远点采样 FPS | `2,048 × 3` | 40,424 | 10,107 | 1.16 GiB |
| `fps_4096` | 最远点采样 FPS | `4,096 × 3` | 40,424 | 10,107 | 2.32 GiB |
| `fps_8192` | 最远点采样 FPS | `8,192 × 3` | 40,424 | 10,107 | 4.63 GiB |
| `poisson_disk_2048` | 泊松盘采样 | `2,048 × 3` | 40,424 | 10,107 | 1.16 GiB |
| `poisson_disk_4096` | 泊松盘采样 | `4,096 × 3` | 40,424 | 10,107 | 2.32 GiB |
| `poisson_disk_8192` | 泊松盘采样 | `8,192 × 3` | 40,424 | 10,107 | 4.63 GiB |
| **合计** | **9 套版本** | — | **363,816** | **90,963** | **24.34 GiB** |

文件系统显示的占用空间可能略大于逻辑大小。源 OBJ 网格另外占约 8.30 GiB。

每个 NPY 数组的一行表示一个点：

```text
[[x₁, y₁, z₁],
 [x₂, y₂, z₂],
 ...,
 [xₙ, yₙ, zₙ]]
```

## 3. 目录与样本对应关系

```text
data/
├── README.md
├── convert_mcb_9sets.bat
├── _pointcloud_converter/          # 一键转换程序及 Windows 运行库
├── MCB_merged/
│   ├── train/<class>/<model_id>.obj
│   └── test/<class>/<model_id>.obj
└── MCB_pointcloud/
    ├── area_random_2048/
    ├── area_random_4096/
    ├── area_random_8192/
    ├── fps_2048/
    ├── fps_4096/
    ├── fps_8192/
    ├── poisson_disk_2048/
    ├── poisson_disk_4096/
    └── poisson_disk_8192/
        ├── train/<class>/<model_id>.npy
        └── test/<class>/<model_id>.npy
```

上述 9 个版本目录均具有相同的 `train/<class>/`、`test/<class>/` 结构。对应样本只改变版本目录和文件后缀，例如：

```text
MCB_merged/train/bearing/<model_id>.obj
MCB_pointcloud/area_random_2048/train/bearing/<model_id>.npy
MCB_pointcloud/fps_4096/train/bearing/<model_id>.npy
MCB_pointcloud/poisson_disk_8192/train/bearing/<model_id>.npy
```

`model_id` 是源网格整理时采用的 64 个十六进制字符构成的稳定标识符，不代表当前 OBJ 或 NPY 文件内容的哈希。

## 4. 类别编号和数量

所有版本的类别编号按英文目录名字典序排列，从 0 到 31。下表数量是**每套版本**的模型数量。

| ID | 类别目录 | 中文说明 | 训练集 | 测试集 | 合计 |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | `bearing` | 轴承及其附件 | 1,064 | 266 | 1,330 |
| 1 | `bushing` | 衬套 | 567 | 142 | 709 |
| 2 | `castors_and_wheels` | 脚轮与轮 | 289 | 72 | 361 |
| 3 | `chain_drive` | 链传动件 | 78 | 19 | 97 |
| 4 | `clamp` | 夹具 | 126 | 31 | 157 |
| 5 | `collar` | 轴环 | 38 | 10 | 48 |
| 6 | `disc` | 圆盘 | 82 | 21 | 103 |
| 7 | `fitting` | 管接头与喷嘴 | 1,309 | 327 | 1,636 |
| 8 | `flange` | 法兰 | 5 | 1 | 6 |
| 9 | `gear` | 齿轮与齿轮传动件 | 386 | 97 | 483 |
| 10 | `handles` | 手柄、旋钮与操纵杆 | 1,291 | 323 | 1,614 |
| 11 | `hinge` | 铰链 | 49 | 12 | 61 |
| 12 | `hook` | 钩 | 97 | 24 | 121 |
| 13 | `joint` | 铰接接头 | 1,342 | 336 | 1,678 |
| 14 | `key_and_spline` | 键、键槽与花键 | 3,408 | 852 | 4,260 |
| 15 | `linear_guide` | 直线导向件 | 39 | 10 | 49 |
| 16 | `motor` | 电机与减速电机 | 594 | 148 | 742 |
| 17 | `nut` | 螺母 | 1,696 | 424 | 2,120 |
| 18 | `pin` | 销 | 5,979 | 1,495 | 7,474 |
| 19 | `plate` | 板件 | 240 | 60 | 300 |
| 20 | `plug` | 堵头 | 135 | 34 | 169 |
| 21 | `pulley` | 带轮 | 245 | 61 | 306 |
| 22 | `ring` | 环与挡圈 | 711 | 178 | 889 |
| 23 | `rivet` | 铆钉 | 1,854 | 464 | 2,318 |
| 24 | `rotor` | 转动叶片件与转子 | 375 | 94 | 469 |
| 25 | `screws_and_bolts` | 螺钉与螺栓 | 13,224 | 3,306 | 16,530 |
| 26 | `spacer` | 间隔件 | 87 | 22 | 109 |
| 27 | `spring` | 弹簧 | 278 | 69 | 347 |
| 28 | `stud` | 螺柱与螺纹杆 | 2,542 | 636 | 3,178 |
| 29 | `switch` | 开关 | 143 | 36 | 179 |
| 30 | `valve` | 阀 | 6 | 1 | 7 |
| 31 | `washer` | 垫圈 | 2,145 | 536 | 2,681 |
| — | **合计** | **32 类** | **40,424** | **10,107** | **50,531** |

部分粗类别包含多个子类型：`bearing` 包含轴承及附件；`handles` 包含手柄、旋钮和操纵杆；`fitting` 包含管接头和喷嘴；`rotor` 包含转子、风扇、叶轮和涡轮；`stud` 包含螺柱与螺纹杆。类别表达零部件类型，同时涉及几何结构和用途。

类别分布不均衡。`screws_and_bolts` 有 16,530 个模型，约占 32.71%；前五大类别合计约占 67.53%。`flange` 和 `valve` 分别只有 6 个、7 个模型，测试集各有 1 个样本。评估时应结合各类测试数量解读结果。

## 5. 三种采样方法

### 5.1 共同的表面候选点

每个模型先生成 **40,960 个候选点**，即最高点数 8,192 的 5 倍。三种方法共享同一候选点池。

首先按三角面面积选择面，再用重心坐标在选中的三角面内部均匀采样：

```text
选择三角面 i 的概率 = 面积(i) / 所有三角面面积之和
s = sqrt(u)，其中 u、v 独立且均匀分布于 [0, 1)
p = (1-s)·a + s·(1-v)·b + s·v·c
```

`a、b、c` 为三角面的三个顶点。计算采用 `float64`，保存时转为 `float32`。候选点按最终 `float32` 坐标检查重复，必要时补充表面样本，使候选池保持 40,960 个可区分点。

所有方法均从真实网格表面采样点中选取，沿用源网格的几何结构。

### 5.2 按面积随机采样：`area_random`

使用候选点池的前 8,192 个点，分别取前 2,048、4,096、8,192 个形成三个版本。它们是按面积生成的随机表面样本，局部可能出现聚集和空隙。

该方法适合标准随机采样基线，表面采样规则与 [Trimesh 的面积加权采样定义](https://trimesh.org/trimesh.sample.html) 一致。

### 5.3 最远点采样：`fps`

在 40,960 个候选点上执行精确贪心 FPS：

1. 以候选池第 0 个点作为起点。
2. 维护每个剩余点到当前已选点集的最近欧氏距离。
3. 每轮选择最近距离最大的点，更新剩余点的最近距离。
4. 连续选取 8,192 个点，取选择序列的前 2,048、4,096、8,192 个形成三个版本。

距离计算在三维欧氏空间中进行，使用平方距离比较。距离相同时按候选点索引确定顺序。计算通过 KD 树与索引最大堆加速，保持与全量距离计算的贪心选择规则一致。

FPS 有助于分散覆盖候选点池，但仍受初始表面候选点的覆盖程度限制。方法定义可参考 [Open3D 的 FPS 接口](https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.farthest_point_down_sample)。

### 5.4 泊松盘采样：`poisson_disk`

采用 Yuksel 样本消除方法生成近似蓝噪声分布：根据邻域拥挤程度不断删除点，直到达到目标点数。这类方法也用于 [Open3D 泊松盘采样](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.sample_points_poisson_disk)。

本地实现使用 KD 树和索引最大堆；权重参数为 `alpha=8`、`beta=0.65`、`gamma=1.5`。设表面积为 `A`、当前候选数为 `M`、目标点数为 `N`：

```text
r_max = 2 × sqrt(A / (2 × sqrt(3) × N))
r_min = 0.65 × r_max × (1 - (N / M)^1.5)
w(d) = max(0, 1 - max(d, r_min) / r_max)^8
```

每个点的拥挤权重为周围其他点的 `w(d)` 之和。逐次删除权重最大的点，并更新其邻居权重。

三个版本按以下顺序生成：

```text
40,960 个表面候选点 → 8,192 点 → 4,096 点 → 2,048 点
```

各次消除使用对应目标点数重新计算半径和权重。4,096、2,048 点版本复用上一级点集作为候选。这是近似均匀间距的样本消除方案，不保证每一对点具有严格相同或统一下限的间距。

### 5.5 三档点数的关系和随机种子

每种方法下，同一模型的点集都满足：

```text
2048 点集 ⊂ 4096 点集 ⊂ 8192 点集
```

这是点集成员关系；保存前每个版本分别打乱行顺序，不能用行号直接对应不同版本中的点。

基础种子为 **42**。每个模型使用来源相对路径及流名称，通过 SHA-256 派生固定种子，避免并行任务调度改变结果：

```python
import hashlib

relative = "train/bearing/<model_id>.obj"
stream = "surface_candidates"
seed = int.from_bytes(
    hashlib.sha256(
        f"MCB_POINTCLOUD_V1|42|{relative}|{stream}".encode("utf-8")
    ).digest()[:8],
    "big",
)
```

候选点使用 `numpy.random.default_rng(seed)`。行顺序使用流名称 `order:<method>:<N>` 独立派生种子。生成环境的 NumPy 版本为 **2.5.4**。相同程序和环境下结果可重复；其他采样实现即使采用同一基础种子，也不保证逐点相同。

## 6. 坐标与数据质量

源 OBJ 网格以包围盒中心为原点，最大顶点半径为 1，保留原有方向。点云沿用源网格坐标，没有按各自采样点再次居中、缩放或旋转。因此不同方法、不同点数的表示可以直接在同一坐标系对齐。

采样点位于单位球内；由于只取表面有限个点，点云自身的包围盒中心不一定恰好为零，最大半径也不一定恰好为 1。坐标无真实物理尺寸单位。

生成阶段对全部点云检查了：

- 点数与数组形状符合对应版本。
- 数据类型为 `float32`，XYZ 为有限数值。
- 每个点云内部无完全相同的 `float32` 坐标行。
- 坐标位于源网格包围盒及单位球范围内，数值容差 `2e-7`。

独立全量校验还核对了每个 NPY 文件的读取结果与内容校验和，以及源 OBJ 内容保持一致。32 个类别的代表模型另做点到三角面检查，共检查 **1,376,256 个点**，全部在 `2e-7` 的表面距离容差内。

点数增加能够更密集地表达现有网格；原网格缺失的螺纹、孔洞或曲面细节无法通过增加采样点恢复。当前点云表示完整零件的理想表面，不包含模拟扫描遮挡、噪声或法线。

## 7. 读取 NPY 数据

从项目根目录运行：

```python
from pathlib import Path
import numpy as np

root = Path("data/MCB_pointcloud/fps_4096")
classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
class_to_idx = {name: index for index, name in enumerate(classes)}
train_samples = [
    (path, class_to_idx[path.parent.name])
    for path in sorted((root / "train").glob("*/*.npy"))
]
test_samples = [
    (path, class_to_idx[path.parent.name])
    for path in sorted((root / "test").glob("*/*.npy"))
]

path, label = train_samples[0]
points = np.load(path, allow_pickle=False)
assert points.shape == (4096, 3)
assert points.dtype == np.float32
assert len(classes) == 32
assert len(train_samples) == 40424
assert len(test_samples) == 10107
```

将 `fps_4096` 替换为其他版本目录，并调整点数检查，即可读取相应数据。训练、验证和测试共享上表的类别编号；验证集可从现有训练集中按类别再划分。

## 8. 用于 PointNeXt

NPY 是保存格式，接入 PointNeXt/OpenPoints 时需要一个读取数组并根据目录取得标签的数据加载器。XYZ 基线可参考 [PointNeXt 官方 ModelNet40 示例](https://guochengqian.github.io/PointNeXt/examples/modelnet/)。

| 设置 | 当前数据对应值 |
| --- | --- |
| 每样本坐标 | `(N, 3)` |
| 坐标类型 | `float32` |
| 输入特征 | XYZ |
| 编码器 `in_channels` | 3 |
| 分类输出数 | 32 |
| 标签范围 | 0～31 |

如果比较这 9 个版本，数据加载器和训练配置的实际输入点数应与所选版本相符。官方示例中的默认 1,024 点设置需要按实验调整；将所有版本都再次采成 1,024 点，会改变“比较三档输入点数”的实验含义。

固定类别划分、模型结构、训练预算、种子与增强策略，只改变待比较的采样方法或点数。增加点数会增加邻域计算和显存开销，可根据硬件调整 batch size，并记录该调整。

分类评估同时报告整体准确率、宏平均 F1、各类召回率和混淆矩阵。类别权重仅依据训练集计算。检索实验应明确查询集、图库和粗标签相关性定义；精确零件匹配需要更细的实例标准。

## 9. 源网格信息

`MCB_merged/` 保留模型的三角面连接关系，可用于进一步生成其他采样或渲染表示。

| 指标 | 最小值 | 中位数 | 最大值 |
| --- | ---: | ---: | ---: |
| 每模型顶点数 | 4 | 549 | 1,747,648 |
| 每模型三角面数 | 4 | 1,104 | 3,539,800 |

源网格总顶点数为 113,254,011，总三角面数为 232,721,348。OBJ 中 `v` 为顶点、`f` 为三角面，面索引从 1 开始。数据提供整件类别标签，没有逐点分割标签、材料、公差或 CAD 构造历史。

实验引用保留原 MCB 项目与论文，并注明本地 32 类版本、采样方法、点数及数据版本日期。


## 10. 在其他 Windows 设备一键生成这九套数据

入口为 [convert_mcb_9sets.bat](convert_mcb_9sets.bat)，支持 **Windows 10/11 64 位 x64**。第一次启动需要联网下载隔离的 Python 3.12.15 与 NumPy 2.5.4；无需提前安装 Python、C++ 编译器，也不修改系统 PATH 或 Python 注册表。

将以下三项一起复制到目标设备的同一目录，例如 `D:\MCB\data\`：

```text
MCB_merged/                 # 必须包含 train/ 与 test/ 下的 OBJ
convert_mcb_9sets.bat
_pointcloud_converter/      # 整个文件夹一起复制
```

双击 BAT，会自动在同一目录生成 `MCB_pointcloud/`，包含上述 9 个版本。完整转换建议预留至少 **30 GiB** 可用空间。程序自动根据 CPU 与可用内存选择并行数，上限为 8；可编辑 BAT 中的 `MCB_WORKERS=0`，例如改成 `MCB_WORKERS=4`。

转换程序沿用 40,960 点候选池、基础种子 42、三种采样规则、嵌套点集及每版本固定行顺序。输出为 `float32 (N, 3)`，不修改源 OBJ 或重新划分类别。不同操作系统、编译器的浮点运算可能产生微小差异，不保证与当前 macOS 生成的文件逐字节相同。

关闭窗口或按 Ctrl+C 中断后，再次双击可继续：已经完成的模型先核验来源和输出校验和；缺失或被修改的已记录输出会重新生成。每个模型完成九个文件后写入断点，文件采用临时文件加原子替换方式保存。结束前还会进行全量文件校验。

环境、断点与日志位于 `%LOCALAPPDATA%\MCBPointcloudConverter\`，不放入点云类别目录。后续运行在依赖完整时可以离线执行；日志位置会显示在控制台。

默认保护未经本程序记录且内容不同的已有输出。复制时只需源 OBJ、BAT 和辅助文件夹；如目标设备已有其他程序生成的 `MCB_pointcloud/`，可选择新的输出目录，或明确使用 `--overwrite` 重建对应文件。

高级命令示例，在 `data` 文件夹中执行（首次先运行 BAT 准备环境）：

```bat
"%LOCALAPPDATA%\MCBPointcloudConverter\runtime-v1\venv\Scripts\python.exe" "_pointcloud_converter\convert_mcb.py" --output "D:\MCB_pointcloud_new" --workers 4
```

Python 程序还支持 `--limit 3` 做小样本转换、`--verify-only` 只校验已记录输出，以及 `--overwrite` 重建内容不同的已有输出。Windows DLL 已包含采样计算核心；[sampling.cpp](_pointcloud_converter/sampling.cpp) 为相应的可审阅源码。

依赖引导使用固定版本的 [uv](https://docs.astral.sh/uv/guides/install-python/)，下载时核对官方发布包的 SHA-256。首次联网需要访问 GitHub/Astral 与 Python 包下载源；下载失败时窗口会保留错误信息，再次运行可重试。

## Windows CUDA 分类实验

使用这些点云训练 PointNeXt-S/B/L/XL、PointNet 和 PointNet++，请参见项目根目录的 [实验说明](../README.md) 和 [run_experiments.bat](../run_experiments.bat)。训练通过 Windows 上的 WSL2 使用 NVIDIA CUDA，保留相同模型 ID 的统一训练/验证/测试划分，原始点云不修改。
