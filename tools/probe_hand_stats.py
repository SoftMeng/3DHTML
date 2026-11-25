#!/usr/bin/env python3
"""手部可观测性探针（决策用）：全片统计 MediaPipe Hands 的可检出性与几何质量。

判据：
  · ROI 由「肘→腕」方向外推 0.18 前臂长、半径 0.9 前臂长（手在腕外侧）
  · 指派：ROI 内检出候选中，取腕点离 ROI 中心最近者
  · 可信门限：|腕→中指尖| ∈ [0.10, 0.20] m（世界系为米制）
  · 对齐门限：世界系 x/y 与图像系同向余弦 > 0.8（世界系与相机系同向）

用法: tools/.venv/bin/python tools/probe_hand_stats.py <视频> <motion.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "tools" / "models" / "hand_landmarker.task"
IDX = {"L": {"elbow": 8, "wrist": 9, "shoulder": 5},
       "R": {"elbow": 7, "wrist": 10, "shoulder": 6}}
SCALE = 2


def px_of(mo, fi, idx):
    vb = mo["videoBox"]
    p = mo["poses"][fi]
    if not p or not p[idx]:
        return None
    tp = vb["torsoPx"][fi]
    hx = vb["frac"][fi][1] * vb["w"]
    hy = (1 - vb["frac"][fi][2]) * vb["h"]
    return np.array([(hx + p[idx][0] * tp) * SCALE, (hy - p[idx][1] * tp) * SCALE])


def main() -> int:
    video, mo_path = Path(sys.argv[1]), Path(sys.argv[2])
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=3,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2)
    lm = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video))
    n = 0
    stat = {s: {"det": 0, "size_ok": 0, "align_ok": 0, "good": 0, "tot": 0,
                 "sz": [], "cos": [], "palmz": []} for s in ("L", "R")}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n >= len(mo["poses"]):
            break
        for side in ("L", "R"):
            m = IDX[side]
            w = px_of(mo, n, m["wrist"])
            e = px_of(mo, n, m["elbow"])
            if w is None or e is None:
                continue
            st = stat[side]
            st["tot"] += 1
            fdir = w - e
            L = float(np.linalg.norm(fdir))
            if L < 8:
                continue
            fdir /= L
            c = w + fdir * (L * 0.18)
            half = int(max(45, L * 0.9))
            x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
            x1, y1 = int(min(fr.shape[1], c[0] + half)), int(min(fr.shape[0], c[1] + half))
            crop = fr[y0:y1, x0:x1]
            if crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            sc = 256 / max(crop.shape[0], crop.shape[1])
            up = cv2.resize(crop, (max(32, int(crop.shape[1] * sc)), max(32, int(crop.shape[0] * sc))))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            res = lm.detect(img)
            if not res.hand_landmarks:
                continue
            st["det"] += 1
            best = None
            for h, wl, hw in zip(res.handedness, res.hand_landmarks, res.hand_world_landmarks):
                W = np.array([[q.x, q.y, q.z] for q in wl])
                X = np.array([[q.x, q.y, q.z] for q in hw])
                ax = (W[0][0] * up.shape[1] - up.shape[1] / 2) / sc + x0
                ay = (W[0][1] * up.shape[0] - up.shape[0] / 2) / sc + y0
                d = float(np.hypot(ax - c[0], ay - c[1]))
                if best is None or d < best[0]:
                    best = (d, ax, ay, X, W)
            d, ax, ay, X, W = best
            if d > L * 0.7:
                continue
            size = float(np.linalg.norm(X[12] - X[0]))
            fwd = X[9] - X[0]
            across = X[5] - X[17]
            fwd2 = W[9] - W[0]
            across2 = W[5] - W[17]
            def cos2(a, b):
                na, nb = float(np.linalg.norm(a[:2])), float(np.linalg.norm(b[:2]))
                if na < 1e-9 or nb < 1e-9:
                    return 0.0
                return float(np.dot(a[:2], b[:2]) / (na * nb))
            c1, c2 = cos2(fwd, fwd2), cos2(across, across2)
            palm = np.cross(fwd, across)
            st["sz"].append(size)
            st["cos"].append(min(c1, c2))
            st["palmz"].append(float(palm[2]))
            if 0.10 <= size <= 0.20:
                st["size_ok"] += 1
                if min(c1, c2) > 0.8:
                    st["align_ok"] += 1
                    st["good"] += 1
        n += 1
        if n % 40 == 0:
            print(f"  ... {n}/{len(mo['poses'])}", flush=True)
    cap.release()
    lm.close()
    print(f"\n共 {n} 帧")
    for side in ("L", "R"):
        st = stat[side]
        sz = np.array(st["sz"]) if st["sz"] else np.array([0.0])
        print(f"[{side}] 有腕点 {st['tot']}  检出 {st['det']}  尺寸合理 {st['size_ok']}  "
              f"对齐合格 {st['align_ok']}  综合可信 {st['good']}")
        print(f"     尺寸 中位 {np.median(sz):.3f} p10 {np.percentile(sz,10):.3f} p90 {np.percentile(sz,90):.3f}")
        if st["cos"]:
            print(f"     对齐余弦 中位 {np.median(st['cos']):.3f}")
        if st["palmz"]:
            pz = np.array(st["palmz"])
            print(f"     palm.z 正 {int((pz>0).sum())} 负 {int((pz<0).sum())}  中位 {np.median(pz):+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
