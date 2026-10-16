# GPU 云服务器运行

**最新状态（2026-10-16）：key6 六个实验均已完成，每个 300 轮，共 192 项测试。全部结果已下载到本地 [本地备份说明](reports/key6/README.md)，逐文件 SHA-256 校验一致。完整结果分析见 [中文分析报告](docs/reports/KEY6.zh-CN.md) / [English report](docs/reports/KEY6.md)。服务器原件仍保留，实例未自动关闭。**

此次服务器部署目录为 `/root/point`。实际硬件是 RTX 4070（12 GB）、8 个 vCPU、约 16 GB 内存、100 GB 系统盘。按最新选择，将原来的 54 项训练缩减为 **key6：6 次训练，每次 300 轮，192 项测试**。

六项分别为 PointNeXt-S、B、L、XL、PointNet 和 PointNet++。统一使用面积随机采样的 2,048 点完整数据、种子 42、同一划分和基础训练配置，用于公平比较算法。每项保留原来的 32 项测试，包括其余八套点云的跨采样/点数测试及鲁棒性测试；这些测试不增加训练次数。现有 S 检查点继续使用。

云服务器直接运行 Linux，不需要 WSL 或 Windows BAT。为了节省磁盘，使用 `--no-pack` 读取原始 NPY，不建立重复点云缓存。结果保存在 `/root/point/results/key6/`，记录内容与 Windows 方案相同；不进行自动结果分析。

独立环境在 `/root/point/.cloud-runtime/venv/`，复用机器已有的 PyTorch 2.7.1/cu126，不修改原有 Python 软件包；新增匹配的 CUDA 12.6 编译组件。实际环境、算子构建、GPU 和依赖版本均写入结果。与之前准备的 Windows PyTorch 2.8.0/cu128 环境不同，后续比较性能时应查看环境记录。

4070 上优先使用 BF16 混合精度，避免 XL 的 FP16 梯度溢出；XYZ 文件仍是 float32。六个模型均已通过 FP32/BF16 的 CUDA 前向、反向及有限梯度检查。

九套点云现已全部上传。首次部署已对 `area_random_2048` 的 50,531 个文件逐一核对 Mac 基准 SHA-256。S 训练完成后，标准套件全量核验九套数据，复用其检查点，补齐跨采样/点数及鲁棒性测试，然后运行剩余 5 项。部署状态在 `remote_logs/job_state.json`。

数据上传已经完成，服务器实验不依赖 Mac 开机或联网。原来的上传日志在本地项目 `remote_logs/upload_handoff.log`。

运行或恢复：

```bash
cd /root/point
bash experiments/run_cloud.sh key6
```

后台运行时使用 tmux，会话名 `mcb-key6`：

```bash
tmux attach -t mcb-key6
```

按 `Ctrl+B` 再按 `D` 可以离开会话而保持训练；不要在训练窗口按 Ctrl+C，除非需要中断训练。正常中断后，重新执行相同命令可以恢复最近完成的轮次。

查看日志与 GPU：

```bash
tail -f /root/point/remote_logs/key6.log
nvidia-smi
```

本文件不会保存 SSH 密码。SSH 断开不影响 tmux 中的训练。服务器不会自动关机；租用费用由平台计费规则决定。

2026-10-14 的第 118 轮后发生过文件句柄耗尽：验证加载器重复创建常驻工作进程，导致管道句柄累积。现已改为仅复用训练加载器的常驻进程，验证与测试进程在结束后释放，并提高当前实验进程的文件句柄上限。第 118 轮检查点和旧日志备份在 `results/quick/_recovery/20261014_fd_exhaustion/`，S 从第 119 轮继续，已有训练不会重做。
