#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从镜头切分结果中“只做候选帧抽样”（不做任何筛选）
改动：时间轴自校准 + 校准阶段带进度条，避免长时间无输出
- 输入: --video + --scenes(jsonl: scene_id, start_sec, end_sec)
- 输出: [镜头序号_该镜头的第N帧_视频秒].jpg
- 实现: 先顺序扫片获得实际总时长 T_actual；alpha=T_actual/max_end_sec；
       将 (start_sec,end_sec) 乘以 alpha，再顺序读取（不 seek），命中即保存
"""

import os, json, argparse
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm

def read_scenes(jsonl_path: str) -> List[Tuple[int, float, float]]:
    scenes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = int(obj["scene_id"])
            s = float(obj["start_sec"])
            e = float(obj["end_sec"])
            scenes.append((sid, s, e))
    scenes.sort(key=lambda x: x[1])
    return scenes
def estimate_duration_sec(video_path: str) -> float:
    """尽量不用全量扫描，优先用元数据/快路径估算时长（秒）；必要时才兜底扫描。"""
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 方案A：容器时长（某些 OpenCV 构建没有该属性）
    dur_prop = getattr(cv2, "CAP_PROP_DURATION", None)
    if dur_prop is not None:
        try:
            dur_ms = cap.get(dur_prop)
            if dur_ms and dur_ms > 0:
                cap.release()
                return float(dur_ms) / 1000.0
        except Exception:
            pass  # 安全回退

    # 方案B：总帧数 / FPS
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    if 1.0 < fps < 120.0 and n_frames > 0:
        cap.release()
        return n_frames / fps

    # 方案C：跳到结尾读 POS_MSEC（部分容器可用）
    try:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)  # 1 表示末尾
        t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_sec and t_sec > 0:
            cap.release()
            return float(t_sec)
    except Exception:
        pass

    cap.release()

    # 方案D：兜底——顺序扫描
    return quick_actual_duration(video_path)


def quick_actual_duration(video_path: str) -> float:
    """
    顺序扫片得到实际总时长(秒)，带进度条。
    注：为保证准确，这里用 read()，这样 POS_MSEC 会稳定更新。
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    show_bar = total_frames > 0
    last_sec = 0.0

    if show_bar:
        bar = tqdm(total=total_frames, desc="校准: 扫描以获取真实总时长", ncols=100)
    else:
        bar = tqdm(desc="校准: 扫描以获取真实总时长", ncols=100)

    cur = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        cur += 1
        last_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if show_bar:
            bar.update(1)
        else:
            # 无法获知总帧数时，适当刷新进度展示
            if cur % 300 == 0:
                bar.set_postfix_str(f"t≈{last_sec:.2f}s")

    bar.close()
    cap.release()
    return float(last_sec)

def build_targets_step(scenes, step_sec: float, margin_sec: float, max_per_shot: int):
    """
    固定步长抽样：
    每个镜头从 [start+margin, end-margin] 开始，每隔 step_sec 取一帧，最多 max_per_shot 张。
    返回: [(t_sec, scene_id, nth_in_scene)], 全局按时间升序
    """
    targets = []
    for sid, s, e in scenes:
        t0 = s + margin_sec
        t1 = e - margin_sec
        if t1 <= t0:
            t_mid = 0.5 * (s + e)
            targets.append((float(t_mid), int(sid), 1))
            continue
        cur = t0
        n = 0
        step = max(step_sec, 1e-6)
        while cur <= t1 and n < max_per_shot:
            n += 1
            targets.append((float(cur), int(sid), n))
            cur += step
    targets.sort(key=lambda x: x[0])
    return targets

def save_jpg(path: str, img, q=95):
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])

def main():
    ap = argparse.ArgumentParser(description="镜头候选帧抽样（无筛选，含时间轴自校准）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step-sec", type=float, default=0.8)
    ap.add_argument("--margin-sec", type=float, default=0.35)
    ap.add_argument("--max-per-shot", type=int, default=6)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"找不到视频: {args.video}")
    if not os.path.isfile(args.scenes):
        raise FileNotFoundError(f"找不到 scenes: {args.scenes}")
    os.makedirs(args.out, exist_ok=True)

    # 读取镜头时间
    scenes = read_scenes(args.scenes)
    if not scenes:
        print("scenes 为空，退出。", flush=True)
        return
    max_end = max(e for _, _, e in scenes)

    # --- 时间轴自校准 ---
    T_actual = estimate_duration_sec(args.video)
    alpha = (T_actual / max_end) if max_end > 0 else 1.0
    print(f"[校准] scenes.max_end={max_end:.3f}s | video.actual={T_actual:.3f}s | alpha={alpha:.6f}（未裁剪）", flush=True)



    scenes_scaled = [(sid, s * alpha, e * alpha) for (sid, s, e) in scenes]

    # 构造目标时刻
    targets = build_targets_step(scenes_scaled, args.step_sec, args.margin_sec, args.max_per_shot)
    print(f"镜头数={len(scenes)} | 目标候选帧数={len(targets)}", flush=True)

    # 顺序读取，不 seek；命中目标即保存
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    meta_path = os.path.join(args.out, "candidates.jsonl")
    mf = open(meta_path, "w", encoding="utf-8")

    t_idx = 0
    saved = 0
    misses = 0
    tol_sec = 1e-3
    pbar = tqdm(total=len(targets), desc="抽样导出", ncols=100)

    while t_idx < len(targets):
        ret, frame = cap.read()
        if not ret or frame is None:
            rem = len(targets) - t_idx
            misses += rem
            for _ in range(rem):
                pbar.update(1)
            break

        t_cur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 命中一个或多个目标时刻（允许同一帧满足多个）
        while t_idx < len(targets) and (t_cur + tol_sec) >= targets[t_idx][0]:
            t_sec, sid, nth = targets[t_idx]
            fname = f"{sid}_{nth}_{t_sec:06.2f}s.jpg"
            fpath = os.path.join(args.out, fname)
            save_jpg(fpath, frame, q=args.jpeg_quality)

            rec = {"scene_id": sid, "n_in_scene": nth, "time_sec": round(float(t_sec), 3), "file": fpath}
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            saved += 1
            t_idx += 1
            pbar.update(1)

    pbar.close()
    mf.close()
    cap.release()

    print(f"完成: 目标 {len(targets)} | 保存 {saved} | 未命中 {misses}", flush=True)
    print(f"元数据: {meta_path}", flush=True)

if __name__ == "__main__":
    main()
