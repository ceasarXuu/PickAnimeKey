#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, shutil, time
from typing import List, Dict
import cv2
from tqdm import tqdm

HELP_TEXT = [
    "快捷键：",
    "  Y/K  通过并保存到 --out",
    "  N/J  跳过（不保存）",
    "  B    回退上一张",
    "  0..9 设置人脸数",
    "  O    切换 遮挡/裁剪 是/否",
    "  R    切换 重复/不重复",
    "  T    切换 有文字/无文字",
    "  S    立即保存进度",
    "  H    显示/隐藏帮助",
    "  Q    退出（自动保存）",
]

def read_candidates(jsonl_path: str) -> List[dict]:
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    # 有序：场景、候选序、时间
    items.sort(key=lambda o: (o.get("scene_id",0), o.get("n_in_scene",0), o.get("time_sec",0.0)))
    return items

def load_log(log_path: str) -> Dict[str, dict]:
    if not os.path.isfile(log_path):
        return {}
    done = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 用 file_in 作为键，便于去重/断点续传
            done[rec["file_in"]] = rec
    return done

def append_log(log_path: str, rec: dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def draw_overlay(bgr, info: dict, show_help: bool):
    vis = bgr.copy()
    H, W = vis.shape[:2]
    # 半透明底
    overlay = vis.copy()
    cv2.rectangle(overlay, (0,0), (W, 92 if not show_help else 92+18*len(HELP_TEXT)), (0,0,0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0, vis)

    # 基本信息行
    line1 = f"[{info['idx']}/{info['total']}] 场景:{info['sid']} 候选:{info['nth']}  时间:{info['tsec']:.2f}s"
    cv2.putText(vis, line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # 标签行
    face = info['face_cnt']
    occ  = "是" if info['occ'] else "否"
    dup  = "重复" if info['dup'] else "不重复"
    txt  = "有文字" if info['txt'] else "无文字"
    line2 = f"人脸数:{face} | 遮挡/裁剪:{occ} | {dup} | {txt}"
    cv2.putText(vis, line2, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    # 提示
    cv2.putText(vis, "H 顯示/隱藏幫助 | Q 退出", (16, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

    if show_help:
        y0 = 100
        for i, t in enumerate(HELP_TEXT):
            col = (200,200,200) if i==0 else (180,180,255)
            cv2.putText(vis, t, (16, y0 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

    return vis

def safe_resize(bgr, scale=0.9, max_w=1920, max_h=1080):
    H, W = bgr.shape[:2]
    s = min(scale, max_w/float(W), max_h/float(H), 1.0)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    return bgr

def build_output_name(sid:int, nth:int, face_cnt:int, occ:bool, dup:bool, txt:bool):
    occ_tag = "是" if occ else "否"
    dup_tag = "重复" if dup else "不重复"
    txt_tag = "有文字" if txt else "无文字"
    return f"场景序号{sid}_第{nth}候选帧_人脸数:{face_cnt}_遮挡裁剪:{occ_tag}_{dup_tag}_{txt_tag}.jpg"

def main():
    ap = argparse.ArgumentParser(description="候选帧人工审核器（键盘打标 + 重命名导出）")
    ap.add_argument("--candidates", required=True, help="select_candidates.py 的 candidates.jsonl")
    ap.add_argument("--out", required=True, help="保存“通过”的重命名图片目录")
    ap.add_argument("--log", default=None, help="审核记录(jsonl)，默认 <out>/review_log.jsonl")
    ap.add_argument("--jpg-quality", type=int, default=95, help="导出 JPEG 质量")
    ap.add_argument("--scale", type=float, default=0.90, help="窗口缩放比例 (0~1)")
    args = ap.parse_args()

    items = read_candidates(args.candidates)
    if not items:
        print("candidates.jsonl 为空或路径错误。")
        return

    os.makedirs(args.out, exist_ok=True)
    log_path = args.log or os.path.join(args.out, "review_log.jsonl")
    done = load_log(log_path)

    total = len(items)
    idx = 0
    # 找断点
    for i, it in enumerate(items):
        if it["file"] not in done:
            idx = i
            break
    else:
        print("所有候选帧均已在日志中，直接退出。")
        return

    cv2.namedWindow("review", cv2.WINDOW_NORMAL)
    show_help = False
    history = []  # 支持单步回退

    pbar = tqdm(total=total, initial=idx, desc="进度", ncols=100)
    while idx < total:
        it = items[idx]
        img_path = it["file"]
        sid = int(it.get("scene_id", 0))
        nth = int(it.get("n_in_scene", 0))
        tsec = float(it.get("time_sec", 0.0))

        # 如果已存在日志，取回默认标签；否则初始化
        rec0 = done.get(img_path, None)
        state = {
            "face_cnt": rec0.get("face_count", 0) if rec0 else 0,
            "occ": bool(rec0.get("occluded_or_cropped", False)) if rec0 else False,
            "dup": bool(rec0.get("is_duplicate", False)) if rec0 else False,
            "txt": bool(rec0.get("has_text", False)) if rec0 else False,
            "decision": rec0.get("decision", None) if rec0 else None,  # "keep" | "skip"
        }

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            # 读图失败，标记跳过并前进
            rec = {
                "scene_id": sid, "n_in_scene": nth, "time_sec": round(tsec,3),
                "file_in": img_path, "error": "read_fail"
            }
            append_log(log_path, rec)
            done[img_path] = rec
            idx += 1
            pbar.update(1)
            continue

        disp = safe_resize(bgr, args.scale)
        while True:
            vis = draw_overlay(disp, {
                "idx": idx+1, "total": total,
                "sid": sid, "nth": nth, "tsec": tsec,
                "face_cnt": state["face_cnt"],
                "occ": state["occ"], "dup": state["dup"], "txt": state["txt"]
            }, show_help)
            cv2.imshow("review", vis)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord('h'), ord('H')):
                show_help = not show_help
                continue
            if key in (ord('q'), ord('Q')):
                # 保存当前进度（不记录当前未决定的项）
                pbar.close()
                print("\n退出并保存进度。")
                cv2.destroyAllWindows()
                return
            if key in (ord('s'), ord('S')):
                # 保存已完成项（当前项未决，不写日志）
                print("已保存进度。")
                continue
            if key in (ord('b'), ord('B')):
                # 单步回退：仅当有历史项可退
                if history:
                    idx = max(0, history.pop() - 1)
                    pbar.n = idx
                    pbar.refresh()
                    break
                else:
                    # 没历史，忽略
                    continue
            if key in (ord('y'), ord('Y'), ord('k'), ord('K')):
                # 决定：通过并保存
                out_name = build_output_name(sid, nth, state["face_cnt"], state["occ"], state["dup"], state["txt"])
                out_path = os.path.join(args.out, out_name)
                if img_path != out_path:
                    # 拷贝并重命名
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
                    else:
                        shutil.copy2(img_path, out_path)
                rec = {
                    "scene_id": sid, "n_in_scene": nth, "time_sec": round(tsec,3),
                    "file_in": img_path, "file_out": out_path,
                    "face_count": int(state["face_cnt"]),
                    "occluded_or_cropped": bool(state["occ"]),
                    "is_duplicate": bool(state["dup"]),
                    "has_text": bool(state["txt"]),
                    "decision": "keep",
                    "ts": int(time.time())
                }
                append_log(log_path, rec)
                done[img_path] = rec
                history.append(idx+1)
                idx += 1
                pbar.update(1)
                break
            if key in (ord('n'), ord('N'), ord('j'), ord('J')):
                # 决定：跳过
                rec = {
                    "scene_id": sid, "n_in_scene": nth, "time_sec": round(tsec,3),
                    "file_in": img_path,
                    "face_count": int(state["face_cnt"]),
                    "occluded_or_cropped": bool(state["occ"]),
                    "is_duplicate": bool(state["dup"]),
                    "has_text": bool(state["txt"]),
                    "decision": "skip",
                    "ts": int(time.time())
                }
                append_log(log_path, rec)
                done[img_path] = rec
                history.append(idx+1)
                idx += 1
                pbar.update(1)
                break

            # 标注切换
            if key in [ord(str(d)) for d in "0123456789"]:
                state["face_cnt"] = int(chr(key))
                continue
            if key in (ord('o'), ord('O')):
                state["occ"] = not state["occ"]; continue
            if key in (ord('r'), ord('R')):
                state["dup"] = not state["dup"]; continue
            if key in (ord('t'), ord('T')):
                state["txt"] = not state["txt"]; continue

    pbar.close()
    cv2.destroyAllWindows()
    print("全部审核完成。日志：", log_path)
    print("通过的图片已保存到：", args.out)

if __name__ == "__main__":
    main()
