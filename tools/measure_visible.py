#!/usr/bin/env python3
"""独立测量源视频里的「可见身高」（人头顶→脚底）与髋中点屏幕位置。

背景是静止的：用逐像素中位数做背景模型，前景 = 与背景差异超过阈值的像素，
取最大连通域的包围盒。这样得到的可见身高不依赖 MediaPipe 的关键点，
可以直接检验 videoBox 的耳→发顶 / 踝→鞋底标定常数。

用法: tools/.venv/bin/python tools/measure_visible.py <视频> [--stride 1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--work-w", type=int, default=360)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        nh = round(h * args.work_w / w)
        frames.append(cv2.resize(f, (args.work_w, nh), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise SystemExit("视频无帧")
    H, W = frames[0].shape[:2]
    print(f"{len(frames)} 帧 @ {fps:.3f}fps  {W}x{H}", flush=True)

    bg = np.median(np.stack(frames[:: max(1, len(frames) // 40)]).astype(np.float32), axis=0)

    rows = []
    for i, f in enumerate(frames):
        d = np.abs(f.astype(np.float32) - bg).max(axis=2)
        mask = (d > 28).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            rows.append(None)
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[k]
        if area < 0.02 * W * H:
            rows.append(None)
            continue
        rows.append({"i": i, "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                     "cx": (x + w / 2) / W, "top": y / H, "bot": (y + h) / H,
                     "frac": h / H, "area": int(area)})

    good = [r for r in rows if r]
    print(f"检出 {len(good)}/{len(rows)} 帧")
    if good:
        fr = np.array([r["frac"] for r in good])
        cx = np.array([r["cx"] for r in good])
        top = np.array([r["top"] for r in good])
        bot = np.array([r["bot"] for r in good])
        print(f"可见身高占比 中位={np.median(fr):.4f} 均值={fr.mean():.4f} "
              f"p10={np.percentile(fr,10):.4f} p90={np.percentile(fr,90):.4f}")
        print(f"头顶 y 中位={np.median(top):.4f}  脚底 y 中位={np.median(bot):.4f}")
        print(f"水平中心 中位={np.median(cx):.4f}")
    for t in [0, 2.5, 5, 6.5, 7.21, 10]:
        fi = min(len(rows) - 1, int(round(t * fps)))
        r = rows[fi]
        print(f"  t={t:>5}  " + (json.dumps(r) if r else "—"))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"fps": fps, "w": W, "h": H, "rows": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
