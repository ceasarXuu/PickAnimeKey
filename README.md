# 动漫关键帧提取工具

一个基于 GPU 加速的动漫视频关键帧提取和人工审核工具集，用于从动漫视频中智能提取高质量的关键帧。

## 功能特点

- **GPU 加速镜头切分**：使用 PyTorch 进行高效的视频镜头切分
- **智能候选帧选择**：基于时间轴校准的候选帧抽样
- **人工审核界面**：直观的键盘操作界面，支持多标签打标
- **批量处理**：支持大规模视频文件的批量处理

## 项目结构

```
anime_key/
├── sbd_gpu_simple_v1.py    # GPU 加速镜头切分脚本
├── select_candidates.py    # 候选帧选择脚本
├── manual_review.py       # 人工审核工具
├── requirements.txt       # 项目依赖
├── .gitignore            # Git 忽略文件
├── 测试命令.md           # 测试命令文档
└── source/               # 模型文件存放目录
```

## 环境要求

- Python 3.7+
- CUDA 支持的 GPU（推荐）
- 16GB+ RAM（处理大视频文件）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 主要依赖

- **opencv-python**: 计算机视觉和视频处理
- **torch**: PyTorch 深度学习框架
- **torchvision**: PyTorch 计算机视觉扩展
- **numpy**: 数值计算
- **tqdm**: 进度条显示

## 使用流程

### 1. 镜头切分

```bash
python sbd_gpu_simple_v1.py \
    --video your_video.mp4 \
    --out output/scenes \
    --device cuda:0
```

### 2. 候选帧选择

```bash
python select_candidates.py \
    --video your_video.mp4 \
    --scenes output/scenes/scenes.jsonl \
    --out output/candidates
```

### 3. 人工审核

```bash
python manual_review.py \
    --candidates output/candidates/candidates.jsonl \
    --out output/reviewed
```

## 人工审核快捷键

- **Y/K**: 通过并保存
- **N/J**: 跳过（不保存）
- **B**: 回退上一张
- **0-9**: 设置人脸数量
- **O**: 切换遮挡/裁剪状态
- **R**: 切换重复/不重复
- **T**: 切换有文字/无文字
- **S**: 立即保存进度
- **H**: 显示/隐藏帮助
- **Q**: 退出并保存

## 输出说明

- 镜头切分会生成场景首帧和元数据文件
- 候选帧选择会生成按时间间隔抽样的候选帧
- 人工审核会生成重命名后的高质量关键帧和审核日志

## 参数调优

### 镜头切分参数

- `--min-scene-sec`: 最短镜头时长（默认1.2秒）
- `--nms-sec`: NMS窗口时长（默认1.2秒）
- `--pan-ssim`: 平移抑制阈值（默认0.97）

### 候选帧选择参数

- `--step-sec`: 抽样时间间隔（默认0.8秒）
- `--margin-sec`: 镜头边缘裕量（默认0.35秒）
- `--max-per-shot`: 每镜头最大候选数（默认6张）

## 注意事项

1. 确保有足够的 GPU 内存（建议 8GB+）
2. 大视频文件建议分段处理
3. 定期保存审核进度（按 S 键）
4. `test_video/` 和 `out/` 文件夹已被 Git 忽略

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！