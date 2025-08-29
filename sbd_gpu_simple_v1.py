#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-only 抽样镜头切分 v4
变化点：
- 增加窗口 NMS（非极大值抑制）：在 nms-sec 窗口内只保留得分最高的切点，避免密集切碎
- 默认更保守：min-scene-sec=1.2, pan-ssim=0.97, center-hist/edge 略调高
其余：顺序抽样、GPU 批量特征（HS直方图/SSIM/边缘），平移对齐后中心SSIM抑制，主体替换由中心直方图/边缘触发
"""
import os, sys, math, json, argparse
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------- utils ----------
def ts(): return datetime.now().strftime("%H:%M:%S")
def log(msg): print(f"[{ts()}] {msg}", flush=True)

def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size == 0: return x
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, k, mode='same')

def sliding_mean_std(x: np.ndarray, win: int):
    if win <= 1 or x.size == 0:
        return x.copy(), np.zeros_like(x)
    k = np.ones(win, dtype=np.float64)
    s1 = np.convolve(x, k, mode='same')
    s2 = np.convolve(x*x, k, mode='same')
    mean = s1 / win
    var = np.maximum(s2 / win - mean*mean, 0.0)
    return mean, np.sqrt(var)

def adaptive_thr(arr: np.ndarray, p: float, k: float) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return 1.0
    return max(float(np.percentile(arr, p)), float(arr.mean() + k * arr.std(ddof=0)))

def is_local_peak(arr: np.ndarray, i: int, w: int) -> bool:
    l = max(0, i - w); r = min(len(arr), i + w + 1)
    return arr[i] >= np.max(arr[l:r])

# ---------- GPU feature helpers ----------
@torch.inference_mode()
def rgb_to_hsv_torch(x: torch.Tensor, eps=1e-6):
    r,g,b = x[:,0], x[:,1], x[:,2]
    maxc,_ = torch.max(x, dim=1); minc,_ = torch.min(x, dim=1)
    v = maxc; d = maxc - minc + eps
    rc = (g-b)/d; gc = (b-r)/d; bc = (r-g)/d
    h = torch.zeros_like(v)
    h[maxc==r] = rc[maxc==r]; h[maxc==g] = 2.0 + gc[maxc==g]; h[maxc==b] = 4.0 + bc[maxc==b]
    h = (h/6.0) % 1.0
    s = d / (v + eps); s[maxc<=eps] = 0.0
    return torch.stack([h,s,v], dim=1)

@torch.inference_mode()
def hs_hist_batch(hs: torch.Tensor, bins_h=32, bins_s=32):
    N = hs.shape[0]
    H = torch.clamp((hs[:,0]*(bins_h-1)).long(), 0, bins_h-1)
    S = torch.clamp((hs[:,1]*(bins_s-1)).long(), 0, bins_s-1)
    idx = H*bins_s + S
    B = bins_h * bins_s
    fid = torch.arange(N, device=hs.device, dtype=torch.long).view(N,1,1)
    flat = (idx + fid*B).reshape(-1)
    cnt = torch.bincount(flat, minlength=N*B).float().reshape(N,B)
    hist = cnt / (cnt.sum(dim=1, keepdim=True) + 1e-8)
    hist = hist / (hist.norm(dim=1, keepdim=True) + 1e-8)
    return hist

@torch.inference_mode()
def to_gray_small(x: torch.Tensor, target=128):
    g = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
    n,c,h,w = g.shape; m = max(h,w)
    if m > target:
        s = target/float(m)
        nh, nw = max(1,int(round(h*s))), max(1,int(round(w*s)))
        g = F.interpolate(g, size=(nh,nw), mode='area', align_corners=None)
    return g

@torch.inference_mode()
def sobel_mag(x: torch.Tensor):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1); gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy) + 1e-8
    mag = mag / (mag.flatten(1).norm(dim=1, keepdim=True).view(-1,1,1,1) + 1e-8)
    return mag

@torch.inference_mode()
def ssim_torch(x: torch.Tensor, y: torch.Tensor, C1=0.01**2, C2=0.03**2):
    k = torch.tensor([[1.,4.,7.,4.,1.],
                      [4.,16.,26.,16.,4.],
                      [7.,26.,41.,26.,7.],
                      [4.,16.,26.,16.,4.],
                      [1.,4.,7.,4.,1.]], device=x.device, dtype=x.dtype)
    k = (k/k.sum()).view(1,1,5,5)
    mu_x = F.conv2d(x, k, padding=2)
    mu_y = F.conv2d(y, k, padding=2)
    sigma_x = F.conv2d(x*x, k, padding=2) - mu_x*mu_x
    sigma_y = F.conv2d(y*y, k, padding=2) - mu_y*mu_y
    sigma_xy = F.conv2d(x*y, k, padding=2) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8
    return (num/den).mean(dim=[1,2,3])

def center_crop(t: torch.Tensor, ratio=0.6):
    H = t.shape[-2]; W = t.shape[-1]
    h = int(round(H * ratio)); w = int(round(W * ratio))
    y0 = (H - h) // 2; x0 = (W - w) // 2
    return t[..., y0:y0+h, x0:x0+w]

@torch.inference_mode()
def phase_shift_xy(gray_a: torch.Tensor, gray_b: torch.Tensor):
    h, w = gray_a.shape[-2:]
    Fa = torch.fft.rfft2(gray_a); Fb = torch.fft.rfft2(gray_b)
    R = Fa * torch.conj(Fb); R = R / (torch.abs(R) + 1e-8)
    corr = torch.fft.irfft2(R, s=(h,w))
    idx = torch.argmax(corr.view(-1))
    iy = (idx // w).item(); ix = (idx % w).item()
    if ix > w//2: ix -= w
    if iy > h//2: iy -= h
    return float(ix), float(iy)

@torch.inference_mode()
def warp_by_shift(gray: torch.Tensor, dx: float, dy: float):
    n, c, h, w = gray.shape
    theta = torch.tensor([[1,0, 2*dx/max(w-1,1)], [0,1, 2*dy/max(h-1,1)]],
                         device=gray.device, dtype=gray.dtype).view(1,2,3)
    grid = F.affine_grid(theta, size=gray.size(), align_corners=False)
    out  = F.grid_sample(gray, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out

# ---------- Pass1：抽样+GPU特征 ----------
def pass1_gpu(video_path: str, device: str, sample_every: int,
              longest: int, batch_size: int, smooth: int):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    num_samples_est = 1 + max(0, (total_frames - 1)//sample_every)

    # 结构主导
    w_hist, w_luma, w_ssim = 0.45, 0.10, 0.45

    diffs = []
    c_ssim_aligned = []   # 对齐后的中心SSIM
    c_histdiffs = []      # 中心直方图差
    c_edgediffs = []      # 中心边缘差

    luma_cache = []

    torch.backends.cudnn.benchmark = True
    bar = tqdm(total=num_samples_est, desc="Pass1: 抽样&特征(GPU)", ncols=100, leave=False)

    def resize_longest_cpu(img_bgr, longest_):
        if longest_ <= 0: return img_bgr
        h, w = img_bgr.shape[:2]; m = max(h,w)
        if m <= longest_: return img_bgr
        s = float(longest_) / m
        return cv2.resize(img_bgr, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)

    batch_rgb = []
    times = []
    prev_hist_last = prev_luma_last = prev_gray_last = None
    prev_hist_c_last = prev_gray_c_last = None

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return np.array([]), np.array([]), float(fps), 0, np.array([]), np.array([]), np.array([]), np.array([])
    frame_idx = 0
    times.append(frame_idx / fps)
    frame = resize_longest_cpu(frame, longest)
    rgb = frame[..., ::-1].copy()
    t = torch.from_numpy(rgb).to(device).float().permute(2,0,1).unsqueeze(0) / 255.0
    batch_rgb.append(t); bar.update(1)

    while True:
        ok = True
        for _ in range(max(0, sample_every - 1)):
            if not cap.grab(): ok = False; break
            frame_idx += 1
        if not ok: break
        ret, frame = cap.retrieve()
        if not ret or frame is None: break
        frame_idx += 1
        times.append(frame_idx / fps)

        frame = resize_longest_cpu(frame, longest)
        rgb = frame[..., ::-1].copy()
        t = torch.from_numpy(rgb).to(device).float().permute(2,0,1).unsqueeze(0) / 255.0
        batch_rgb.append(t); bar.update(1)

        if len(batch_rgb) >= batch_size:
            x = torch.cat(batch_rgb, dim=0)  # [B,3,H,W]
            hsv = rgb_to_hsv_torch(x); hs = hsv[:, :2]
            hist = hs_hist_batch(hs, 32, 32)
            luma = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).mean(dim=[1,2])
            gray = to_gray_small(x, target=128)
            luma_cache.append(luma.detach().float().cpu().numpy())

            hsv_c = center_crop(hsv, 0.6); hs_c = hsv_c[:, :2]; hist_c = hs_hist_batch(hs_c, 32, 32)
            gray_c = center_crop(gray, 0.6); edge_c = sobel_mag(gray_c)

            if prev_hist_last is not None:
                cos_h0 = float((prev_hist_last * hist[0]).sum().clamp(-1,1).item())
                d_hist0 = 1.0 - cos_h0
                d_luma0 = float(torch.abs(prev_luma_last - luma[0]).item())
                ssim0   = float(ssim_torch(prev_gray_last, gray[0:1]).clamp(0,1).item())
                d_ssim0 = 1.0 - ssim0
                diffs.append(w_hist*d_hist0 + w_luma*d_luma0 + w_ssim*d_ssim0)

                dx, dy = phase_shift_xy(prev_gray_last, gray[0:1])
                prev_aligned = warp_by_shift(prev_gray_last, dx, dy)
                ssim_ca = float(ssim_torch(center_crop(prev_aligned,0.6), gray_c[0:1]).clamp(0,1).item())
                c_ssim_aligned.append(ssim_ca)

                cos_hc0 = float((prev_hist_c_last * hist_c[0]).sum().clamp(-1,1).item())
                c_histdiffs.append(1.0 - cos_hc0)
                v1 = edge_c[0:1].flatten(1); v2 = sobel_mag(prev_gray_c_last).flatten(1)
                cos_e0 = float((v1*v2).sum(dim=1).clamp(-1,1).item())
                c_edgediffs.append(1.0 - cos_e0)

            B = hist.shape[0]
            if B >= 2:
                cos_h = (hist[:-1] * hist[1:]).sum(dim=1).clamp(-1,1)
                d_hist = (1.0 - cos_h)
                d_luma = torch.abs(luma[1:] - luma[:-1])
                ssim_vals = ssim_torch(gray[:-1], gray[1:]).clamp(0,1)
                d_ssim = (1.0 - ssim_vals)
                d_all = (w_hist*d_hist + w_luma*d_luma + w_ssim*d_ssim).detach().cpu().numpy()
                diffs.extend(d_all.tolist())

                # 对齐后的中心SSIM
                for i in range(B-1):
                    dx, dy = phase_shift_xy(gray[i:i+1], gray[i+1:i+2])
                    prev_aligned = warp_by_shift(gray[i:i+1], dx, dy)
                    ssim_ca = float(ssim_torch(center_crop(prev_aligned,0.6), gray_c[i+1:i+2]).clamp(0,1).item())
                    c_ssim_aligned.append(ssim_ca)

                cos_hc = (hist_c[:-1] * hist_c[1:]).sum(dim=1).clamp(-1,1).detach().cpu().numpy()
                c_histdiffs.extend((1.0 - cos_hc).tolist())
                e1 = edge_c[:-1].flatten(1); e2 = edge_c[1:].flatten(1)
                cos_edge = (e1*e2).sum(dim=1).clamp(-1,1).detach().cpu().numpy()
                c_edgediffs.extend((1.0 - cos_edge).tolist())

            prev_hist_last = hist[-1]; prev_luma_last = luma[-1]; prev_gray_last = gray[-1:]
            prev_hist_c_last = hist_c[-1]; prev_gray_c_last = gray_c[-1:]
            batch_rgb = []

    if batch_rgb:
        x = torch.cat(batch_rgb, dim=0)
        hsv = rgb_to_hsv_torch(x); hs = hsv[:, :2]
        hist = hs_hist_batch(hs, 32, 32)
        luma = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).mean(dim=[1,2])
        gray = to_gray_small(x, target=128)
        luma_cache.append(luma.detach().float().cpu().numpy())
        hsv_c = center_crop(hsv, 0.6); hs_c = hsv_c[:, :2]; hist_c = hs_hist_batch(hs_c, 32, 32)
        gray_c = center_crop(gray, 0.6); edge_c = sobel_mag(gray_c)

        if prev_hist_last is not None:
            cos_h0 = float((prev_hist_last * hist[0]).sum().clamp(-1,1).item())
            d_hist0 = 1.0 - cos_h0
            d_luma0 = float(torch.abs(prev_luma_last - luma[0]).item())
            ssim0   = float(ssim_torch(prev_gray_last, gray[0:1]).clamp(0,1).item())
            d_ssim0 = 1.0 - ssim0
            diffs.append(w_hist*d_hist0 + w_luma*d_luma0 + w_ssim*d_ssim0)

            dx, dy = phase_shift_xy(prev_gray_last, gray[0:1])
            prev_aligned = warp_by_shift(prev_gray_last, dx, dy)
            ssim_ca = float(ssim_torch(center_crop(prev_aligned,0.6), gray_c[0:1]).clamp(0,1).item())
            c_ssim_aligned.append(ssim_ca)

            cos_hc0 = float((prev_hist_c_last * hist_c[0]).sum().clamp(-1,1).item())
            c_histdiffs.append(1.0 - cos_hc0)
            v1 = edge_c[0:1].flatten(1); v2 = sobel_mag(prev_gray_c_last).flatten(1)
            cos_e0 = float((v1*v2).sum(dim=1).clamp(-1,1).item())
            c_edgediffs.append(1.0 - cos_e0)

        B = hist.shape[0]
        if B >= 2:
            cos_h = (hist[:-1] * hist[1:]).sum(dim=1).clamp(-1,1)
            d_hist = (1.0 - cos_h)
            d_luma = torch.abs(luma[1:] - luma[:-1])
            ssim_vals = ssim_torch(gray[:-1], gray[1:]).clamp(0,1)
            d_ssim = (1.0 - ssim_vals)
            d_all = (w_hist*d_hist + w_luma*d_luma + w_ssim*d_ssim).detach().cpu().numpy()
            diffs.extend(d_all.tolist())

            for i in range(B-1):
                dx, dy = phase_shift_xy(gray[i:i+1], gray[i+1:i+2])
                prev_aligned = warp_by_shift(gray[i:i+1], dx, dy)
                ssim_ca = float(ssim_torch(center_crop(prev_aligned,0.6), gray_c[i+1:i+2]).clamp(0,1).item())
                c_ssim_aligned.append(ssim_ca)

            cos_hc = (hist_c[:-1] * hist_c[1:]).sum(dim=1).clamp(-1,1).detach().cpu().numpy()
            c_histdiffs.extend((1.0 - cos_hc).tolist())
            e1 = edge_c[:-1].flatten(1); e2 = edge_c[1:].flatten(1)
            cos_edge = (e1*e2).sum(dim=1).clamp(-1,1).detach().cpu().numpy()
            c_edgediffs.extend((1.0 - cos_edge).tolist())

    bar.close(); cap.release()

    diffs = np.asarray(diffs, dtype=np.float32)
    diffs_s = moving_avg(diffs, smooth)
    times = np.asarray(times, dtype=np.float32)
    lumas = np.concatenate(luma_cache, axis=0).astype(np.float32) if luma_cache else np.array([], dtype=np.float32)
    c_ssim_aligned = np.asarray(c_ssim_aligned, dtype=np.float32)
    c_histdiffs = np.asarray(c_histdiffs, dtype=np.float32)
    c_edgediffs = np.asarray(c_edgediffs, dtype=np.float32)
    return diffs_s, times, float(fps), len(times), lumas, c_ssim_aligned, c_histdiffs, c_edgediffs

# ---------- NMS ----------
def nms_on_cuts(cand_idx, score, w_samp):
    """cand_idx: 候选切点样本索引(>=1); score: 与 diffs 对齐的分数(取 diffs[i-1]); w_samp: NMS半窗"""
    if not cand_idx: return []
    # 按分数从高到低
    order = sorted(cand_idx, key=lambda i: float(score[i-1]), reverse=True)
    kept = []
    for i in order:
        if all(abs(i - j) > w_samp for j in kept):
            kept.append(i)
    kept.sort()
    return kept

# ---------- detect ----------
def detect_scenes(diffs_s, times, fps, sample_every, min_scene_sec,
                  p, k, local_sec, k_local, peak_win,
                  c_ssim_aligned, c_histdiffs, c_edgediffs,
                  pan_ssim_thr, center_hist_thr, center_edge_thr,
                  nms_sec, global_sec):
    n = times.size
    if n <= 1: return []
    dt = float(sample_every)/max(fps,1e-6)
    min_samp = max(1, int(math.ceil(min_scene_sec/dt)))
    nms_w = max(1, int(round(nms_sec/dt)))

    # —— 长窗“全局”阈值（随时间自适应）——
    win_g = max(3, int(round(global_sec/dt)))
    mu_g, sd_g = sliding_mean_std(diffs_s, win_g)
    thr_g_arr = mu_g + k * sd_g

    # —— 短窗“局部”阈值（对突发敏感）——
    win_l = max(3, int(round(local_sec/dt)))
    mu_l, sd_l = sliding_mean_std(diffs_s, win_l)
    thr_loc = mu_l + k_local * sd_l

    # 最终阈值（逐点取较大者）
    thr_arr = np.maximum(thr_g_arr, thr_loc)

    # 收集候选切点
    cand = []
    for i in range(1, n):  # diffs_s[i-1] 对应样本 i
        val = diffs_s[i-1]
        trigger = (val > thr_arr[i-1]) or \
                  (c_histdiffs[i-1] >= center_hist_thr) or \
                  (c_edgediffs[i-1] >= center_edge_thr)
        if not trigger:
            continue
        # 平移/摇移抑制：对齐后中心SSIM很高 -> 不切
        if c_ssim_aligned[i-1] >= pan_ssim_thr:
            continue
        if is_local_peak(diffs_s, i-1, peak_win):
            cand.append(i)

    # 最短镜头时长
    cand2, last = [], 0
    for i in cand:
        if (i - last) >= min_samp:
            cand2.append(i); last = i

    # 窗口 NMS：在 nms-sec 内只保留最高分
    kept = nms_on_cuts(cand2, diffs_s, nms_w)

    # 生成场景
    scenes, start_idx = [], 0
    for ci in kept:
        s = float(times[start_idx]); e = float(times[ci])
        if e > s: scenes.append((s, e, start_idx)); start_idx = ci
    if float(times[-1]) > float(times[start_idx]):
        scenes.append((float(times[start_idx]), float(times[-1]), start_idx))
    return scenes

# ---------- export ----------
def export_first_frames(video_path, scenes, fps, out_dir, jpeg_quality, sample_every):
    """
    按“采样索引”导出每段首帧：
    - 不再用 s*fps 映射帧号（避免 VFR 偏差）
    - 完全复现检测阶段的采样节奏：首帧=样本0，然后每隔 sample_every 帧 retrieve 一帧
    - 只向前顺序读取，不 seek
    """
    import os, json
    from tqdm import tqdm
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 目标采样索引 -> (scene_id, start_sec)
    targets = []
    for i, (s, e, start_idx) in enumerate(scenes, start=1):
        targets.append((int(start_idx), i, float(s)))
    # 去重并排序
    tmap = {}
    for si, sid, s in targets:
        if si not in tmap or sid < tmap[si][0]:
            tmap[si] = (sid, s)
    targets = sorted([(si, sid, s) for si, (sid, s) in tmap.items()], key=lambda x: x[0])

    log(f"导出模式: 按采样索引，不seek；目标数={len(targets)}")

    # 顺序采样并导出
    recs, misses = [], []
    bar = tqdm(total=len(targets), desc="Export: 导出首帧(按采样索引)", ncols=100, leave=False)

    # 读取第一帧 -> 样本索引 0
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return recs

    sample_idx = 0
    t_idx = 0

    # 命中当前样本？
    while t_idx < len(targets) and targets[t_idx][0] == sample_idx:
        _, scene_id, start_sec = targets[t_idx]
        fname = f"scene{scene_id:04d}_{start_sec:06.2f}s.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        recs.append({"scene_id": scene_id, "start_sec": round(start_sec, 3),
                     "end_sec": None, "file": fpath})
        t_idx += 1
        bar.update(1)

    # 继续以“每隔 sample_every 帧取一帧”的节奏顺序前进
    while t_idx < len(targets):
        # 跳过 sample_every-1 帧（grab 不解码）
        for _ in range(max(0, sample_every - 1)):
            if not cap.grab():
                break
        # 取样本帧
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            # 片源结束或异常，后续全部 miss
            while t_idx < len(targets):
                misses.append(targets[t_idx][1])
                t_idx += 1
                bar.update(1)
            break

        sample_idx += 1

        # 命中这一个或连着的多个目标样本索引
        while t_idx < len(targets) and targets[t_idx][0] == sample_idx:
            _, scene_id, start_sec = targets[t_idx]
            fname = f"scene{scene_id:04d}_{start_sec:06.2f}s.jpg"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            recs.append({"scene_id": scene_id, "start_sec": round(start_sec, 3),
                         "end_sec": None, "file": fpath})
            t_idx += 1
            bar.update(1)

    bar.close()
    cap.release()

    # 回填 end_sec
    sid2end = {i+1: scenes[i][1] for i in range(len(scenes))}
    for r in recs:
        r["end_sec"] = round(float(sid2end.get(r["scene_id"], r["start_sec"])), 3)

    # 写元数据
    with open(os.path.join(out_dir, "scenes.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if misses:
        log(f"警告: 有 {len(misses)} 个场景未命中（超出可读取样本范围？）示例: {misses[:10]}")
    return recs




# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="GPU-only 抽样镜头切分 v4（含窗口NMS）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--min-scene-sec", type=float, default=1.20)
    ap.add_argument("--nms-sec", type=float, default=1.20, help="窗口NMS时长（只保留窗口内最强切点）")
    ap.add_argument("--p", type=float, default=94.0)
    ap.add_argument("--k", type=float, default=1.8)
    ap.add_argument("--local-sec", type=float, default=2.0)
    ap.add_argument("--k-local", type=float, default=1.6)
    ap.add_argument("--peak-win", type=int, default=1)
    ap.add_argument("--downscale-longest", type=int, default=640)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pan-ssim", type=float, default=0.97, help="对齐后中心SSIM≥此值则抑制切分（平移/摇移）")
    ap.add_argument("--center-hist-cut", type=float, default=0.10, help="中心HS直方图差≥此值时强制切分（主体替换）")
    ap.add_argument("--center-edge-cut", type=float, default=0.20, help="中心边缘差≥此值时强制切分（主体替换）")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--global-sec", type=float, default=60.0,
                help="长窗全局阈值窗口秒（随时间自适应）")
    args = ap.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        print("需要 CUDA 环境且 --device 形如 cuda:0。", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(args.video):
        print(f"找不到视频: {args.video}", file=sys.stderr); sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    log(f"读取: {args.video}")
    log(f"参数: sample_every={args.sample_every}, min_scene_sec={args.min_scene_sec}, nms_sec={args.nms_sec}, "
        f"P={args.p}, K={args.k}, local_sec={args.local_sec}, k_local={args.k_local}, "
        f"peak_win={args.peak_win}, longest={args.downscale_longest}, smooth={args.smooth}, "
        f"batch={args.batch_size}")
    log(f"阈值: pan_ssim={args.pan_ssim}, "
        f"center_hist_cut={args.center_hist_cut}, center_edge_cut={args.center_edge_cut}")


    diffs_s, times, fps, n_samples, lumas, c_ssimA, c_histD, c_edgeD = pass1_gpu(
        args.video, args.device, args.sample_every, args.downscale_longest,
        args.batch_size, args.smooth
    )
    log(f"样本数: {n_samples} | FPS: {fps:.3f}")
    if n_samples <= 1:
        print("样本不足或读取失败。", file=sys.stderr); sys.exit(2)

    scenes = detect_scenes(
    diffs_s, times, fps, args.sample_every, args.min_scene_sec,
    args.p, args.k, args.local_sec, args.k_local, args.peak_win,
    c_ssimA, c_histD, c_edgeD,
    args.pan_ssim, args.center_hist_cut, args.center_edge_cut,
    args.nms_sec, args.global_sec
    )

    log(f"镜头数: {len(scenes)}")

    recs = export_first_frames(args.video, scenes, fps, args.out, args.jpeg_quality, args.sample_every)
    log(f"导出完成: {len(recs)} 张首帧，元数据写入 {os.path.join(args.out,'scenes.jsonl')}")

if __name__ == "__main__":
    main()
