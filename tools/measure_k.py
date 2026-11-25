#!/usr/bin/env python3
"""测量源视频里表演者的「可见身高」（发顶→鞋底）并换算成「躯干长」的倍数 K。

videoBox 的可见身高靠两个手工常数标定（耳→发顶 0.73 躯干、踝→鞋底 0.38 躯干），
实测这两个常数在不同帧上误差 4%~11%，而 K（身高/躯干长）是**人物常量**，
用前景剪影逐帧测量后取中位数，比逐帧手工标定稳得多。

前景提取：局部时域中位数背景（容忍缓慢的推拉摇移）+ 最大连通域。
用法: tools/.venv/bin/python tools/measure_k.py <视频> <motion.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    video, mo_path = Path(sys.argv[1]), Path(sys.argv[2])
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    fps, vb = mo["fps"], mo["videoBox"]
    W, H = vb["w"], vb["h"]

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    cap.release()
    n = len(frames)
    arr = np.stack(frames).astype(np.float32)
    print(f"{n} 帧 {W}x{H}", flush=True)

    rows = []
    R = 14
    for i in range(n):
        lo, hi = max(0, i - R), min(n, i + R + 1)
        bg = np.median(arr[lo:hi], axis=0)
        d = np.abs(arr[i] - bg).max(axis=2)
        mask = (d > 26).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best = None
        for k in range(1, cnt):
            x, y, w, h, area = stats[k]
            if area < 0.01 * W * H or h < 0.15 * H or w > 0.9 * W:
                continue
            if best is None or area > best[4]:
                best = (x, y, w, h, area)
        if best is None:
            rows.append(None)
            continue
        x, y, w, h, area = best
        rows.append({"top": y / H, "bot": (y + h) / H, "h": h / H, "cx": (x + w / 2) / W,
                     "w": w / W, "area": area / (W * H)})

    tp = mo["videoBox"]["torsoPx"]
    ks = []
    for i, r in enumerate(rows):
        if not r or i >= len(tp) or not tp[i]:
            continue
        ks.append({"i": i, "t": i / fps, "K": r["h"] * H / tp[i], "h": r["h"],
                   "top": r["top"], "bot": r["bot"], "cx": r["cx"], "w": r["w"]})
    if not ks:
        print("未检出前景")
        return 1
    K = np.array([r["K"] for r in ks])
    print(f"K(可见身高/躯干) 中位={np.median(K):.3f} 均值={K.mean():.3f} "
          f"p10={np.percentile(K,10):.3f} p90={np.percentile(K,90):.3f} n={len(K)}")
    hs = np.array([r["h"] for r in ks])
    thr = np.percentile(hs, 40)
    sel = [r for r in ks if r["h"] <= thr]
    if sel:
        K2 = np.array([r["K"] for r in sel])
        print(f"紧凑姿态子集（h≤p40）中位 K={np.median(K2):.3f} n={len(sel)}")
    for t in [0, 2.5, 5, 6.5, 7.21, 10]:
        fi = min(n - 1, int(round(t * fps)))
        r = next((x for x in ks if x["i"] == fi), None)
        print(f"  t={t:>5} " + (json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in r.items()}) if r else "—"))
    Path("/tmp/measure_k.json").write_text(json.dumps({"K": float(np.median(K)), "rows": ks},
                                                       ensure_ascii=False), encoding="utf-8")
    print("→ /tmp/measure_k.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
