# MCB 工业三维模型类型识别实验

使用现有 MCB 的 **32 类、50,531 个工业三维模型**，对三种采样方法 × 三种点数生成的 **9 套相同模型点云**进行分类实验。训练和测试使用 XYZ `float32`，不需要法线、颜色或在线网格转换。

项目为你的 **RTX 4090（24 GB）、i5-14600KF、64 GB 内存、Windows** 准备。运行方式是 **Windows BAT → WSL2 Ubuntu 24.04 → NVIDIA CUDA → PyTorch → 官方 OpenPoints 编码器**。训练使用 Windows 上的 4090；WSL2 提供 Linux 编译与运行环境。

## 复制与启动

将整个项目复制到 Windows 本地 SSD，例如 `D:\point`。必须保留 `PointNeXt/openpoints/` 子模块内容、`experiments/`、根目录 BAT 和 `data/MCB_pointcloud/` 的 9 套 NPY。训练已有 NPY 时不需要复制原始 OBJ 或 `MCB_pointcloud.zip`；需要重新转换时再携带 `MCB_merged/` 和现有转换工具。

先运行小规模试跑：

```bat
run_experiments.bat smoke
```

然后双击根目录 **`run_experiments.bat`**，或执行：

```bat
run_experiments.bat full
```

| 模式 | 训练任务 | 测试任务 | 用途 |
|---|---:|---:|---|
| `smoke` | 6 | 192 | 每个模型 1 轮，每个划分最多 64 个模型，检查完整流程 |
| `key6`，云服务器当前计划 | 6 | 192 | 六种算法统一使用面积随机采样 2048 点完整数据，各训练 300 轮 |
| `quick` | 54 | 1,728 | 六种算法 × 九套数据 × 一个种子，完整数据、300 轮 |
| `full`，默认 | **1,296** | **11,898** | 多种子、参数、交互、数据规模、鲁棒性及泛化实验 |

完整模式共 **399,600 个训练轮次**，需要长期运行；实际耗时由首次真实训练记录决定。它不是几小时内完成的任务。建议为环境、点云缓存、所有检查点和原始预测记录预留 **至少约 400 GiB 空闲磁盘**；脚本按权重、优化器状态、预测文件及缓存估算，完整计划的启动预算约 **354 GiB**，并在恢复时扣除已有缓存、计入已有结果。此数为容量规划估算。WSL 缓存所在的 Windows 分区也需有足够空间。默认全部保存 `best.pt` 和可恢复训练的 `last.pt`，不会自动删除训练记录。不要将试跑结果当作分类性能结论。

### 首次准备环境

BAT 会检查 64 位 Windows、项目完整性和 WSL2，必要时安装 Ubuntu 24.04。首次启用 WSL2 可能触发 Windows 管理员授权及重启：根据系统提示重启，再运行同一个 BAT；普通训练不需要一直以 Windows 管理员身份运行。

需要 Windows 10 21H2（build 19044）或更新的系统 / Windows 11、支持 WSL CUDA 的 NVIDIA Windows 驱动，以及 BIOS 中启用的 CPU 虚拟化。脚本检查 GPU；驱动异常时给出提示，由你安装或更新 Windows NVIDIA 驱动。**WSL 内安装的是 `cuda-toolkit-12-8`，不会安装 Linux NVIDIA 显示驱动。** GPU 运行条件参见 [NVIDIA 的 WSL CUDA 文档](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 和 [微软的 WSL 安装文档](https://learn.microsoft.com/windows/wsl/install)。

脚本自动准备编译器、Git、CUDA Toolkit 12.8、uv 0.13.0、隔离 Python 3.11.13、PyTorch **2.8.0/cu128** 及固定版本 Python 依赖；首次需要网络，会下载较大的 CUDA/PyTorch 软件包。环境位于 Ubuntu 的 `/home/mcb-runner/.cache/mcb-pointnext/runtime-v1/`，不依赖 Mac 虚拟环境。环境准备的系统操作以 WSL root 执行；模型编译与实验以普通 Linux 用户 `mcb-runner` 执行。

GPU 架构由构建脚本实际检测，4090 为 `sm_89`。只编译本实验需要的官方 PointNet2 批处理 CUDA 算子，不安装分割、重建等任务的额外扩展。编译后先检查 CUDA 分组运算的值与梯度，再检查六个模型的 FP32/AMP 前向和反向。实际实验不允许自动退回 CPU。

如果九套点云目录缺失，BAT 会先调用现有 Windows 转换器。不同 CPU/操作系统重新转换可能产生细微浮点差异：成功重转换后，套件校验模型 ID、类别及全部坐标，登记新点云指纹，并在之后的模式中使用该指纹。直接复制现有 NPY 时严格匹配 Mac 数据指纹。已开始的实验不会与改变后的数据混用。

Ubuntu 默认使用的 WSL 内存上限通常足够完成此项目；点云缓存通过 mmap 读取。首次数据核验会读取全部 **454,779** 个 NPY，核对 ID、类别、划分、形状、类型、有限值、归一化边界并计算 SHA-256。随后把同一套点云按模型顺序打包成连续 NPY，缓存到 WSL 的 ext4 文件系统，以减少 NTFS 上小文件访问开销；原始点云与点顺序不修改。九套打包缓存额外约 **24.34 GiB**。之后按文件大小和纳秒修改时间复用已验证缓存；强制重新核验可删除结果目录 `_data/inventory.sqlite3`。

## 模型来源与分类适配

`PointNeXt/` 是完整克隆，含 Git 历史、全部官方配置、示例、文档和完整 `openpoints` 子模块。没有删除其他官方文件或修改模型源码。来源：[官方 PointNeXt](https://github.com/guochengqian/PointNeXt)。

| 实验模型 | 编码器宽度 | 各阶段块数 | 本项目 32 类分类参数量 |
|---|---:|---|---:|
| PointNeXt-S | 32 | `[1,1,1,1,1,1]` | 1,371,456 |
| PointNeXt-B | 32 | `[1,2,3,2,2]` | 3,879,200 |
| PointNeXt-L | 32 | `[1,3,5,3,3]` | 7,176,416 |
| PointNeXt-XL | 64 | `[1,4,7,4,4]` | 40,839,968 |
| PointNet | 官方编码器，含输入与特征 T-Net | — | 3,477,225 |
| PointNet++ | 官方 original SSG 配置 | 三个 SA 阶段 | 1,470,496 |

上表是基础配置的实际实例化计数；改变分类头池化方式会改变参数量。S 使用官方 ModelNet40 分类编码器；B/L/XL 使用官方 S3DIS 配置中的**完整编码器**，保留原始宽度、块数、残差与倒残差结构，末端全局池化后连接官方 `ClsHead` 的 `[512,256]`、BatchNorm、Dropout 和 32 类输出。B/L/XL 是本项目的分类适配，不能把工业分类结果表述为官方 S3DIS 分割实验的复现。基础实验统一初始邻域半径为 0.15；S 保留官方的半径倍率，其余保留各自编码器默认倍率。每次实验的确切模型结构写入 `model_spec.json`。

PointNet++ 原始 SSG 编码器、PointNet 的两个 T-Net 均保留，PointNet 训练增加系数 0.001 的特征变换正交正则。旧算法实验以简明对照为主。

`models.py` 选择性加载官方分类所需模块，避免包初始化引入无关扩展；官方 `ClsHead` 完整类从源码 AST 加载。`ops.py` 仅更新张量分配与 AMP 包装，调用官方 CUDA 核函数。构建时在环境缓存内复制 C++ 源码，适配 `is_cuda()/data_ptr()` API，**不修改仓库中的原始源文件或 CUDA 采样/查询算法**。CPU 参考运算只用于 Mac 开发测试。

## 数据与实验设计

九套数据为 `area_random`、`fps`、`poisson_disk` × `2048/4096/8192`。每个训练输入使用该套数据的**全部点数**，8192 点不会为节省显存悄悄改为 1024 或 2048 点。坐标沿用转换时的网格归一化，不对不同采样点云分别重新居中或缩放。

保留原始 10,107 个测试模型，只从原始 40,424 个训练模型中按类别划分约 10% 验证模型，固定划分种子 `20261013`，得到 **36,380 个训练、4,044 个验证、10,107 个测试模型**。所有采样方法、点数、模型、训练种子使用**相同模型 ID 划分**。每类至少保留一个训练模型和一个验证模型。模型选择按验证集 macro F1，依次以 OA、loss 打破平局；测试集不参与训练、调参、检查点选择。

类别分布不均衡，flange、valve 的测试集各只有一个模型。保存每类 support、precision、recall、F1 和完整混淆矩阵，后续分析应同时查看这些样本量；不可只凭这两类的单次准确率下结论。

默认基础训练配置：300 轮、AdamW、学习率 0.001、weight decay 0.05、有效 batch 32、10 轮学习率预热、余弦衰减、label smoothing 0.1、梯度裁剪 1、分类头 dropout 0.5、AMP、4 个加载进程。基础增强为点顺序打乱、XYZ 分轴缩放 `[0.8,1.25]`、平移 `[-0.1,0.1]`。默认不假定工业模型具有某个统一重力方向。各参数的完整清单位于 [`experiments/experiment_plan.full.json`](../experiments/experiment_plan.full.json)。

AMP 的 `amp_dtype=auto` 优先使用支持设备上的 BF16，其他设备使用 FP16。BF16 不需要 loss scaling；FP16 使用 GradScaler。实际精度写入 `training_complete.json` 和原始结果 CSV，点云文件始终保存 float32 XYZ。云服务器的运行状态和部署方式见 [`CLOUD.md`](../CLOUD.md)。

| 实验组 | 训练次数 | 内容 |
|---|---:|---|
| `core` | 162 | 6 种模型 × 9 套数据 × 3 个种子（42、123、2026），使用共同训练配置 |
| `pointnext_sweep` | 864 | 4 个 PointNeXt × 9 套数据 × 24 种参数改动，种子 42 |
| `interaction_grid` | 144 | 学习率 × weight decay × batch 的交互组合，自动去除与前面完全相同的配置 |
| `data_efficiency` | 108 | 四个 PointNeXt、三种 4096 点采样、训练集的 10% / 25% / 50%、三个种子 |
| `baseline_classic` | 18 | PointNet/PointNet++ × 9 套数据，补充 Adam、阶梯学习率、零 smoothing、旋转和抖动的简明传统配置 |

24 种 PointNeXt 参数改动采用单因素对照：学习率 `0.0003/0.003`；weight decay `0/0.001/0.01`；batch `16/64`；初始半径 `0.075/0.3`；邻居数 `16/64`；smoothing `0/0.2`；dropout `0.2`；max+avg 池化；无增强、SO(3) 增强、强增强；类别加权 CE、focal loss；SGD；阶梯学习率；关闭 AMP；训练 600 轮。交互网格为学习率 `0.001/0.003`、weight decay `0.01/0.05`、batch `32/64`。

数据规模实验以模型 ID 做分层、嵌套子集，所有方法/算法/种子使用同一个子集，每类至少一个模型；小类别的实际保留比例因此可能超过名义比例。加权 CE 权重为训练类别频数平方根的倒数，再归一化到平均 1；loss 按样本平均。focal 是 smoothing CE 乘 `(1-p_true)^2`，gamma=2。配置和实际训练样本摘要均被记录。

显存探测涵盖前向、反向、优化器状态。物理 batch 会在不改变模型和点数的前提下自动选取，最低为 2，结合梯度累积维持设定的有效 batch。BatchNorm 仍按物理 batch 统计，梯度累积不等价于完整大 batch 的 BatchNorm；所有物理/有效 batch 都保存。每轮末尾不足一个物理 batch 的样本被丢弃并计数，末尾不足一个有效 batch 的累积按实际样本数归一化。模型随机种子与按模型 ID 的增强随机数都固定，但官方 CUDA backward 的原子加法不保证逐位可复现。

### 鲁棒性与泛化测试

每个 core 模型，以及 SO(3)/强增强和数据规模训练模型，使用同一个最佳验证检查点执行以下 32 项测试；其他参数对照执行常规干净测试。

| 测试 | 设置 |
|---|---|
| 干净测试 | 训练所用采样方法及点数 |
| 跨采样、跨点数 | 其余 8 套数据，不重新训练；测试模型 ID 完全相同 |
| 高斯噪声 | sigma `0.005/0.01/0.02`，截断 ±3 sigma |
| 旋转 | 随机轴旋转 `15°/45°/90°` |
| 缩放 | 各向同性倍率 `1.15/1.3/1.5` 或倒数 |
| 平移 | 随机方向长度 `0.05/0.1/0.2` |
| 随机缺点 | 删除 `10%/30%/50%`，以剩余点重复补齐原点数 |
| 离群点 | 将 `1%/5%/10%` 的点替换为 `[-1,1]^3` 中随机点 |
| 遮挡近似 | 随机平面方向裁掉 `10%/30%/50%`，补齐原点数 |
| 点顺序 | 随机置换 |
| 任意方向 | Haar 均匀 SO(3) 旋转 |

扰动以样本 ID、训练种子、扰动类型及严重度确定，跨算法可复现。跨采样/点数衡量同一批未参加训练的测试模型的表示变化；它们不是九批独立模型。平面裁切是遮挡近似，不是物理传感器渲染。当前数据为 CAD 完整表面采样，没有真实扫描设备数据，因此这些实验不提供真实扫描域泛化结论。

## 结果保存与恢复

默认结果写在项目下 `results/full/`；不同模式分开保存。BAT 按计划依次运行每个实验，没有自动排名、挑选最优配置、计算跨实验均值或生成结论。

```text
results/full/
  plan.json                    实际执行清单和计划哈希
  environment_initial.json     Python/依赖/GPU/驱动/源码哈希/仓库提交
  environment_latest.json      最近启动的环境记录
  cuda_build.json               原始算子源码哈希与编译产物哈希
  pip-freeze.txt
  _data/
    index.json                  模型 ID、类别、划分、数据指纹
    class_to_idx.json
    splits.csv
    inventory.sqlite3
  status.json                   每项任务状态、退出码、启动/结束时间
  execution_summary.json        执行数量、失败列表、当前会话耗时
  results.csv                   所有已完成测试的原始指标逐行汇总
  runs/<run_id>/
    config.json / model_spec.json / binding.json
    console.log
    epochs/0000.json ...         逐轮完整训练/验证指标
    history.csv                 逐轮标量指标
    last.pt                     最近完成轮次的模型/优化器/AMP 状态
    best.pt                     验证集选择的模型
    best_checkpoint.json
    training_complete.json      训练计时、参数量、实际 batch、训练样本摘要
    benchmark.json              batch 1/8、10 次预热、30 次计时的原始耗时
    tests/<case>.json            指标、每类指标、混淆矩阵、测试耗时和文件哈希
    tests/<case>.npz             model_id/y_true/y_pred/32 类 probabilities
    complete.json               完成标记与结果文件指纹
results/setup/                  环境准备及套件控制台日志
```

指标保存 loss、OA、mAcc、macro F1、weighted F1、macro precision、top-3 OA、NLL、Brier、15-bin ECE、每类 support/precision/recall/F1、混淆矩阵。OA 等比例用 `[0,1]` 保存。mAcc 是固定类别的平均 recall，macro F1 也是固定类别平均，零预测类别按 0 处理。loss 是该实验设定的分类目标，PointNet 训练 loss 另外包含变换正则；不同损失之间还可看统一的 NLL。

时间记录训练每轮墙钟时间、CUDA event 计算时间、验证/测试墙钟时间、测试前向+loss 计算时间、模型推理 batch 1/8 的原始 30 次耗时、p50/p95、吞吐和显存峰值、整个实验进程耗时。推理计时同步 CUDA，不含读取/传输数据；测试墙钟包含数据加载及预测结果转回 CPU。CSV 是原始结果的展开，比较时应区分训练目标、精度模式、点数、有效和物理 batch。

关闭或 Ctrl+C 中断后，再启动同一个模式即可恢复：完成的实验跳过，未完成训练从最近完成轮次重跑后续轮次，已完成测试复用，缺失或损坏的测试预测重算。默认某项失败后记录日志并继续剩余任务；整套存在失败时 BAT 返回非零。失败后再次启动会重试。若数据、代码、PyTorch 或计划改变，脚本拒绝混合旧结果，需要选择新的结果目录。只加载本项目自己生成的 PyTorch 检查点。

进一步定制：修改 `experiments/mcb/plan.py` 并重新生成计划，或在已准备好的 WSL 环境中使用 `python -m experiments.mcb.suite --plan <自定义JSON> --output <新目录>`。套件还支持 `--fail-fast` 和 `--no-pack`。计划 JSON 必须提供与自动清单相同的 `runs/config/evaluations` 结构。

## 使用训练后的模型识别

在已准备的 Ubuntu 环境中，进入项目路径，设置算子路径后执行：

```bash
runtime=/home/mcb-runner/.cache/mcb-pointnext/runtime-v1
export PYTHONPATH="$runtime/ops:$PWD"
"$runtime/venv/bin/python" -m experiments.mcb.predict \
  --run-dir "results/full/runs/某个实验ID" \
  --input "data/MCB_pointcloud/area_random_2048/test/bearing" \
  --output "predictions.csv"
```

输入必须是相同采样定义与网格归一化下的 `float32 (N,3)` NPY，点数与该训练实验一致。输出包括预测类型、最大概率以及全部类别概率。新模型的网格转点云方法参见 [`data/README.md`](../data/README.md)。

## 当前验证范围

在 Mac 开发环境以 PyTorch 2.8.0 CPU 验证模型构造与完整前向/反向、数据一致性、指标、扰动、训练/测试、结果记录、恢复和损坏预测修复，14 项开发测试通过；Bash 脚本已做语法检查。454,779 个原始点云的全量校验通过，并在 `experiments/data_reference.json` 保存数据指纹，Windows 启动时会比较指纹以检查复制是否完整。Mac 没有 NVIDIA CUDA，也没有 Windows/WSL2，**尚未真实执行 Windows 安装、CUDA 编译和 RTX 4090 训练**；首次 `smoke` 会在你的目标设备上完成这些检查。开发测试不构成工业分类性能报告。

依赖与上游许可保留在官方仓库中；原始数据来源与使用条件见数据 README。训练脚本从随机初始化开始，未下载官方预训练权重。

开发验证记录、上游提交与数据指纹见 [`experiments/verification.json`](../experiments/verification.json)。
