#!/usr/bin/env python3
"""在原始分辨率上量出表演者的可见身高（发顶→鞋底），换算成躯干长的倍数 K。

videoBox 现有两个手工常数（耳→发顶 0.73、踝→鞋底 0.38）逐帧标定，实测误差 4%~11%，
而 K（可见身高/躯干长）是**人物常量**。这里用「局部时域中位数背景 + 以髋点为种子的
连通域」在原始分辨率上量剪影：种子定位保证取到人而不是观众/灯。

用法: tools/.venv/bin/python tools/measure_k2.py <视频> <motion.json> [--native]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_motion import IDX  # noqa: E402


def main() -> int:
    video, mo_path = Path(sys.argv[1]), Path(sys.argv[2])
    native = "--native" in sys.argv
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    fps, vb = mo["fps"], mo["videoBox"]
    W, H = (vb["w"], vb["h"]) if not native else (720, 1280)
    sx, sy = W / vb["w"], H / vb["h"]

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) if not native else f)
    cap.release()
    n = len(frames)
    arr = np.stack(frames).astype(np.float32)

    rows = []
    R = 16
    for i in range(n):
        fi = i
        fr = mo["videoBox"]["frac"][fi] if fi < len(mo["videoBox"]["frac"]) else None
        if not fr:
            rows.append(None)
            continue
        seed_x, seed_y = int(fr[1] * vb["w"] * sx), int((1 - fr[2]) * vb["h"] * sy)
        lo, hi = max(0, i - R), min(n, i + R + 1)
        bg = np.median(arr[lo:hi], axis=0)
        d = np.abs(arr[i] - bg).max(axis=2)
        mask = (d > 24).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        k = int(lab[seed_y, seed_x]) if 0 <= seed_y < H and 0 <= seed_x < W else 0
        if k == 0:
            rows.append(None)
            continue
        x, y, w, h, area = stats[k]
        if h < 0.10 * H or area < 0.005 * W * H:
            rows.append(None)
            continue
        rows.append({"i": i, "t": i / fps, "top": y, "bot": y + h, "h": h,
                     "x": x, "w": w, "area": area, "K": h / mo["videoBox"]["torsoPx"][fi] * (vb["h"] / H)})

    good = [r for r in rows if r]
    if not good:
        print("未检出")
        return 1
    K = np.array([r["K"] for r in good])
    hs = np.array([r["h"] for r in good])
    print(f"检出 {len(good)}/{n}")
    print(f"K 中位={np.median(K):.4f} 均值={K.mean():.4f} p25={np.percentile(K,25):.4f} "
          f"p75={np.percentile(K,75):.4f}")
    for p in [20, 30, 40, 50, 60, 70]:
        sel = [r for r in good if r["h"] <= np.percentile(hs, p)]
        print(f"  h≤p{p}: K 中位={np.median([r['K'] for r in sel]):.4f} n={len(sel)}")
    for t in [0, 2.5, 5, 6.5, 7.21, 10, 12]:
        fi = min(n - 1, int(round(t * fps)))
        r = next((x for x in good if x["i"] == fi), None)
        print(f"  t={t:>5} " + (f"top={r['top']} bot={r['bot']} h={r['h']} K={r['K']:.4f}" if r else "—"))
    Path("/tmp/measure_k2.json").write_text(json.dumps({"rows": good}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
