# Anime Keyframe Extractor

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[中文文档](README.zh-CN.md) | English

A GPU-accelerated toolset for extracting high-quality keyframes from anime videos. This tool uses deep learning-based scene detection and provides an intuitive manual review interface for labeling and filtering frames.

## Features

- **GPU-Accelerated Scene Detection**: Efficient shot boundary detection using PyTorch with GPU batch processing
- **Smart Candidate Frame Selection**: Time-axis calibrated sampling with configurable step and margin
- **Intuitive Manual Review Interface**: Keyboard-driven labeling with multiple tags (face count, occlusion, duplicates, text)
- **Batch Processing**: Supports large-scale video file processing with progress tracking
- **Adaptive Thresholds**: Automatically adjusts detection thresholds based on video content
- **Motion-Aware Detection**: Suppresses false positives from camera panning and translation

## Project Structure

```
anime_key/
├── sbd_gpu_simple_v1.py    # GPU-accelerated shot boundary detection
├── select_candidates.py    # Candidate frame selection with time calibration
├── manual_review.py        # Manual review and labeling tool
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file (English)
├── README.zh-CN.md        # Chinese documentation
├── 测试命令.md             # Test command examples (Chinese)
└── source/                # Model files directory
    └── frozen_east_text_detection.pb  # EAST text detection model
```

## Requirements

- Python 3.7 or higher
- CUDA-capable GPU (recommended for faster processing)
- 16GB+ RAM (for processing large video files)
- OpenCV with FFmpeg support

## Installation

```bash
# Clone the repository
git clone https://github.com/ceasarXuu/PickAnimeKey.git
cd PickAnimeKey

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | >=4.5.0 | Computer vision and video processing |
| torch | >=1.9.0 | Deep learning framework |
| torchvision | >=0.10.0 | PyTorch vision utilities |
| numpy | >=1.20.0 | Numerical computation |
| tqdm | >=4.60.0 | Progress bar display |

## Quick Start

### 1. Shot Boundary Detection

Detect scene changes and export the first frame of each scene:

```bash
python sbd_gpu_simple_v1.py \
    --video your_video.mp4 \
    --out output/scenes \
    --device cuda:0
```

### 2. Candidate Frame Selection

Sample candidate frames from each detected scene:

```bash
python select_candidates.py \
    --video your_video.mp4 \
    --scenes output/scenes/scenes.jsonl \
    --out output/candidates
```

### 3. Manual Review

Review and label candidate frames interactively:

```bash
python manual_review.py \
    --candidates output/candidates/candidates.jsonl \
    --out output/reviewed
```

## Detailed Usage

### Shot Boundary Detection (`sbd_gpu_simple_v1.py`)

Detects scene changes using GPU-accelerated feature extraction with adaptive thresholds.

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

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | *required* | Input video file path |
| `--out` | *required* | Output directory for scene frames and metadata |
| `--device` | `cuda:0` | Computing device (`cuda:0`, `cuda:1`, `cpu`) |
| `--sample-every` | `3` | Sample every N frames for feature extraction |
| `--min-scene-sec` | `1.2` | Minimum scene duration in seconds |
| `--nms-sec` | `1.2` | NMS window duration (keep strongest cut in window) |
| `--global-sec` | `60.0` | Global adaptive threshold window size |
| `--local-sec` | `2.0` | Local adaptive threshold window size |
| `--p` | `94.0` | Percentile for base threshold calculation |
| `--k` | `1.8` | Standard deviation multiplier for global threshold |
| `--k-local` | `1.6` | Standard deviation multiplier for local threshold |
| `--peak-win` | `1` | Local peak detection window size |
| `--pan-ssim` | `0.97` | SSIM threshold for panning/translation suppression |
| `--center-hist-cut` | `0.10` | Center histogram difference threshold for subject change |
| `--center-edge-cut` | `0.20` | Center edge difference threshold for subject change |
| `--downscale-longest` | `640` | Resize longest edge to this value (0 to disable) |
| `--smooth` | `1` | Smoothing window size for difference curve |
| `--batch-size` | `256` | GPU batch size for feature extraction |
| `--jpeg-quality` | `95` | JPEG quality for exported frames (1-100) |

#### Parameter Tuning Guide

- **Increase detection sensitivity**: Lower `--min-scene-sec`, decrease `--pan-ssim`
- **Reduce false positives**: Increase `--pan-ssim`, increase `--k` and `--k-local`
- **Detect more subject changes**: Lower `--center-hist-cut` and `--center-edge-cut`
- **Faster processing**: Increase `--sample-every`, reduce `--downscale-longest`
- **Better quality**: Decrease `--sample-every`, increase `--downscale-longest`

### Candidate Frame Selection (`select_candidates.py`)

Samples frames from detected scenes with time-axis calibration.

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

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--video` | *required* | Input video file path |
| `--scenes` | *required* | Scene metadata JSONL from shot detection |
| `--out` | *required* | Output directory for candidate frames |
| `--step-sec` | `0.8` | Sampling interval in seconds |
| `--margin-sec` | `0.35` | Margin from scene start/end in seconds |
| `--max-per-shot` | `6` | Maximum candidates per scene |
| `--jpeg-quality` | `95` | JPEG quality for exported frames |

### Manual Review (`manual_review.py`)

Interactive keyboard-driven interface for reviewing and labeling frames.

```bash
python manual_review.py \
    --candidates output/candidates/candidates.jsonl \
    --out output/reviewed \
    --log output/reviewed/review_log.jsonl \
    --jpg-quality 95 \
    --scale 0.9
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--candidates` | *required* | Candidate metadata JSONL file |
| `--out` | *required* | Output directory for approved frames |
| `--log` | `<out>/review_log.jsonl` | Review log file path |
| `--jpg-quality` | `95` | JPEG quality for exported frames |
| `--scale` | `0.9` | Display window scale (0.0-1.0) |

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Y` / `K` | Approve and save frame |
| `N` / `J` | Skip frame (don't save) |
| `B` | Go back to previous frame |
| `0-9` | Set face count (0-9) |
| `O` | Toggle occlusion/cropping status |
| `R` | Toggle duplicate status |
| `T` | Toggle text presence status |
| `S` | Save progress immediately |
| `H` | Show/hide help overlay |
| `Q` | Quit and save progress |

#### Output Filename Format

Approved frames are saved with descriptive filenames:
```
Scene{scene_id}_{nth}thCandidate_{face_count}Faces_Occluded{Yes/No}_{Duplicate/Unique}_{With/Without}Text.jpg
```

Example: `Scene5_2ndCandidate_3Faces_OccludedNo_Unique_WithText.jpg`

## Output Files

### Shot Boundary Detection

- `scene####_#####.##s.jpg` - First frame of each scene
- `scenes.jsonl` - Scene metadata (scene_id, start_sec, end_sec, file)

### Candidate Selection

- `{scene_id}_{nth}_{time}s.jpg` - Candidate frame images
- `candidates.jsonl` - Candidate metadata (scene_id, n_in_scene, time_sec, file)

### Manual Review

- Labeled frame images with descriptive filenames
- `review_log.jsonl` - Complete review history with all labels

## Performance Tips

1. **GPU Memory**: Ensure sufficient GPU memory (8GB+ recommended). Reduce `--batch-size` if OOM occurs.
2. **Large Videos**: For very large files, consider splitting into segments first.
3. **CPU Mode**: Use `--device cpu` if GPU is unavailable (slower but works).
4. **Progress Saving**: Press `S` during review to save progress. The tool supports resume from interruption.
5. **Batch Review**: Use consistent labeling criteria for large datasets.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `--batch-size` or `--downscale-longest` |
| Missed scene cuts | Lower `--min-scene-sec`, decrease `--pan-ssim` |
| Too many false positives | Increase `--pan-ssim`, increase `--k` |
| Video won't open | Ensure FFmpeg is installed and OpenCV has FFmpeg support |
| Slow processing | Increase `--sample-every`, use GPU instead of CPU |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- EAST text detection model for text region detection
- PyTorch team for the excellent deep learning framework
- OpenCV community for computer vision tools

## Changelog

### v1.0.0
- Initial release
- GPU-accelerated shot boundary detection
- Time-axis calibrated candidate selection
- Interactive manual review interface
