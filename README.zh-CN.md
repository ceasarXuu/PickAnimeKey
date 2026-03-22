# 动漫关键帧提取工具

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-可选-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 中文文档

一个基于 GPU 加速的动漫视频关键帧提取工具集。本工具使用深度学习进行场景检测，并提供直观的人工审核界面用于帧标注和筛选。

## 功能特点

- **GPU 加速镜头切分**：使用 PyTorch 进行高效的镜头边界检测，支持 GPU 批量处理
- **智能候选帧选择**：基于时间轴校准的采样，支持自定义步长和边距
- **直观的人工审核界面**：键盘驱动的标注界面，支持多标签（人脸数、遮挡、重复、文字）
- **批量处理**：支持大规模视频文件处理，带进度跟踪
- **自适应阈值**：根据视频内容自动调整检测阈值
- **运动感知检测**：抑制相机平移和移动导致的误报

## 项目结构

```
anime_key/
├── sbd_gpu_simple_v1.py    # GPU 加速镜头边界检测
├── select_candidates.py    # 带时间校准的候选帧选择
├── manual_review.py        # 人工审核和标注工具
├── requirements.txt        # Python 依赖
├── .gitignore             # Git 忽略规则
├── README.md              # 英文文档
├── README.zh-CN.md        # 本文档（中文）
├── 测试命令.md             # 测试命令示例
└── source/                # 模型文件目录
    └── frozen_east_text_detection.pb  # EAST 文本检测模型
```

## 环境要求

- Python 3.7 或更高版本
- 支持 CUDA 的 GPU（推荐，处理更快）
- 16GB+ 内存（用于处理大视频文件）
- 支持 FFmpeg 的 OpenCV

## 安装

```bash
# 克隆仓库
git clone https://github.com/ceasarXuu/PickAnimeKey.git
cd PickAnimeKey

# 安装依赖
pip install -r requirements.txt
```

### 依赖包

| 包名 | 版本 | 用途 |
|------|------|------|
| opencv-python | >=4.5.0 | 计算机视觉和视频处理 |
| torch | >=1.9.0 | 深度学习框架 |
| torchvision | >=0.10.0 | PyTorch 视觉工具 |
| numpy | >=1.20.0 | 数值计算 |
| tqdm | >=4.60.0 | 进度条显示 |

## 快速开始

### 1. 镜头切分

检测场景变化并导出每个场景的首帧：

```bash
python sbd_gpu_simple_v1.py \
    --video your_video.mp4 \
    --out output/scenes \
    --device cuda:0
```

### 2. 候选帧选择

从检测到的场景中采样候选帧：

```bash
python select_candidates.py \
    --video your_video.mp4 \
    --scenes output/scenes/scenes.jsonl \
    --out output/candidates
```

### 3. 人工审核

交互式审核和标注候选帧：

```bash
python manual_review.py \
    --candidates output/candidates/candidates.jsonl \
    --out output/reviewed
```

## 详细用法

### 镜头切分 (`sbd_gpu_simple_v1.py`)

使用 GPU 加速特征提取和自适应阈值检测场景变化。

```bash
python sbd_gpu_simple_v1.py \
    --video input.mp4 \
    --out output/scenes \
    --device cuda:0 \
    --sample-every 3 \
    --min-scene-sec 1.2 \
    --nms-sec 1.2 \
    --pan-ssim 0.97 \
    --center-hist-cut 0.10 \
    --center-edge-cut 0.20 \
    --jpeg-quality 95
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | *必需* | 输入视频文件路径 |
| `--out` | *必需* | 场景帧和元数据输出目录 |
| `--device` | `cuda:0` | 计算设备 (`cuda:0`, `cuda:1`, `cpu`) |
| `--sample-every` | `3` | 特征提取采样间隔（每 N 帧） |
| `--min-scene-sec` | `1.2` | 最短镜头时长（秒） |
| `--nms-sec` | `1.2` | NMS 窗口时长（窗口内只保留最强切点） |
| `--global-sec` | `60.0` | 全局自适应阈值窗口大小 |
| `--local-sec` | `2.0` | 局部自适应阈值窗口大小 |
| `--p` | `94.0` | 基础阈值计算的百分位数 |
| `--k` | `1.8` | 全局阈值的标准差倍数 |
| `--k-local` | `1.6` | 局部阈值的标准差倍数 |
| `--peak-win` | `1` | 局部峰值检测窗口大小 |
| `--pan-ssim` | `0.97` | 平移/移动抑制的 SSIM 阈值 |
| `--center-hist-cut` | `0.10` | 主体替换检测的中心直方图差阈值 |
| `--center-edge-cut` | `0.20` | 主体替换检测的中心边缘差阈值 |
| `--downscale-longest` | `640` | 预处理缩放长边尺寸（0 禁用） |
| `--smooth` | `1` | 差分曲线平滑窗口大小 |
| `--batch-size` | `256` | GPU 特征提取批大小 |
| `--jpeg-quality` | `95` | 导出帧的 JPEG 质量（1-100） |

#### 参数调优指南

- **提高检测敏感度**：降低 `--min-scene-sec`，减小 `--pan-ssim`
- **减少误报**：增加 `--pan-ssim`，增加 `--k` 和 `--k-local`
- **检测更多主体变化**：降低 `--center-hist-cut` 和 `--center-edge-cut`
- **加快处理速度**：增加 `--sample-every`，减小 `--downscale-longest`
- **提高质量**：减小 `--sample-every`，增加 `--downscale-longest`

### 候选帧选择 (`select_candidates.py`)

从检测到的场景中带时间轴校准采样帧。

```bash
python select_candidates.py \
    --video input.mp4 \
    --scenes output/scenes/scenes.jsonl \
    --out output/candidates \
    --step-sec 0.8 \
    --margin-sec 0.35 \
    --max-per-shot 6 \
    --jpeg-quality 95
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | *必需* | 输入视频文件路径 |
| `--scenes` | *必需* | 镜头检测生成的场景元数据 JSONL |
| `--out` | *必需* | 候选帧输出目录 |
| `--step-sec` | `0.8` | 采样间隔（秒） |
| `--margin-sec` | `0.35` | 距离场景开始/结束的边距（秒） |
| `--max-per-shot` | `6` | 每个场景最大候选帧数 |
| `--jpeg-quality` | `95` | 导出帧的 JPEG 质量 |

### 人工审核 (`manual_review.py`)

交互式键盘驱动的审核和标注界面。

```bash
python manual_review.py \
    --candidates output/candidates/candidates.jsonl \
    --out output/reviewed \
    --log output/reviewed/review_log.jsonl \
    --jpg-quality 95 \
    --scale 0.9
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--candidates` | *必需* | 候选帧元数据 JSONL 文件 |
| `--out` | *必需* | 通过审核的帧输出目录 |
| `--log` | `<out>/review_log.jsonl` | 审核日志文件路径 |
| `--jpg-quality` | `95` | 导出帧的 JPEG 质量 |
| `--scale` | `0.9` | 显示窗口缩放比例（0.0-1.0） |

#### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `Y` / `K` | 通过并保存帧 |
| `N` / `J` | 跳过帧（不保存） |
| `B` | 回退到上一帧 |
| `0-9` | 设置人脸数量（0-9） |
| `O` | 切换遮挡/裁剪状态 |
| `R` | 切换重复状态 |
| `T` | 切换文字存在状态 |
| `S` | 立即保存进度 |
| `H` | 显示/隐藏帮助覆盖层 |
| `Q` | 退出并保存进度 |

#### 输出文件名格式

通过审核的帧使用描述性文件名保存：
```
场景{场景序号}_第{n}候选帧_{人脸数}人脸_遮挡{是/否}_{重复/不重复}_{有/无}文字.jpg
```

示例：`场景5_第2候选帧_3人脸_遮挡否_不重复_有文字.jpg`

## 输出文件

### 镜头切分

- `scene####_#####.##s.jpg` - 每个场景的首帧
- `scenes.jsonl` - 场景元数据（scene_id, start_sec, end_sec, file）

### 候选帧选择

- `{scene_id}_{nth}_{time}s.jpg` - 候选帧图像
- `candidates.jsonl` - 候选帧元数据（scene_id, n_in_scene, time_sec, file）

### 人工审核

- 带描述性文件名的标注帧图像
- `review_log.jsonl` - 完整审核历史记录

## 性能优化建议

1. **GPU 内存**：确保足够的 GPU 内存（建议 8GB+）。如遇到 OOM，减小 `--batch-size`
2. **大视频文件**：对于非常大的文件，建议先分割成片段处理
3. **CPU 模式**：如果没有 GPU，使用 `--device cpu`（较慢但可用）
4. **进度保存**：审核过程中按 `S` 键保存进度。工具支持中断后恢复
5. **批量审核**：对于大型数据集，使用一致的标注标准

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA 内存不足 | 减小 `--batch-size` 或 `--downscale-longest` |
| 漏检场景切换 | 降低 `--min-scene-sec`，减小 `--pan-ssim` |
| 误报太多 | 增加 `--pan-ssim`，增加 `--k` |
| 视频无法打开 | 确保已安装 FFmpeg 且 OpenCV 支持 FFmpeg |
| 处理速度慢 | 增加 `--sample-every`，使用 GPU 代替 CPU |

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 致谢

- EAST 文本检测模型用于文本区域检测
- PyTorch 团队提供优秀的深度学习框架
- OpenCV 社区提供计算机视觉工具

## 更新日志

### v1.0.0
- 初始版本发布
- GPU 加速镜头边界检测
- 时间轴校准候选帧选择
- 交互式人工审核界面
