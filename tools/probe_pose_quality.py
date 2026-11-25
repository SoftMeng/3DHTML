#!/usr/bin/env python3
"""姿态质量对比探针：整帧检测 vs 裁剪复检 × lite/full 模型，输出骨架叠加图。

用法: tools/.venv/bin/python tools/probe_pose_quality.py <视频> [帧号...]
输出: screenshots/_probe_pose_<variant>.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"lite": ROOT / "tools/models/pose_landmarker_lite.task",
          "full": ROOT / "tools/models/pose_landmarker_full.task"}
CONN = [(0, 1), (1, 2), (0, 3), (0, 4), (3, 5), (4, 6), (5, 6), (5, 7), (6, 8), (7, 9),
        (8, 10), (9, 10), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 14),
        (13, 15), (14, 16), (15, 16)]
VIS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def make(model_path):
    base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
    return mp.tasks.vision.PoseLandmarker.create_from_options(opts)


def draw(fr, lm, tag):
    c = fr.copy()
    pts = {}
    for i in VIS:
        p = lm[i]
        if p is None or (p.visibility is not None and p.visibility < 0.2):
            continue
        pts[i] = (int(p.x * fr.shape[1]), int(p.y * fr.shape[0]))
    for a, b in CONN:
        if a in pts and b in pts:
            cv2.line(c, pts[a], pts[b], (0, 255, 255), 2)
    for i, q in pts.items():
        cv2.circle(c, q, 5, (0, 0, 255) if i in (9, 10, 15, 16) else (0, 255, 0), -1)
    cv2.putText(c, tag, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return c


def main() -> int:
    video = Path(sys.argv[1])
    frames = [int(x) for x in sys.argv[2:]] or [48, 96, 144, 288]
    cap = cv2.VideoCapture(str(video))
    imgs = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in frames:
            imgs[i] = fr.copy()
        i += 1
        if i > max(frames):
            break
    cap.release()

    for tag, path in MODELS.items():
        lm_task = make(path)
        for mode in ("frame", "crop"):
            tiles = []
            for fi, fr in imgs.items():
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                res = lm_task.detect(img)
                if not res.pose_landmarks:
                    continue
                lm = res.pose_landmarks[0]
                if mode == "crop":
                    xs = [p.x for p in lm if p is not None]
                    ys = [p.y for p in lm if p is not None]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    halfw = max(x1 - x0, (y1 - y0) * 0.42) * 0.72
                    halfh = max(y1 - y0, (x1 - x0) * 1.6) * 0.66
                    ax0 = int(max(0, (cx - halfw) * fr.shape[1]))
                    ax1 = int(min(fr.shape[1], (cx + halfw) * fr.shape[1]))
                    ay0 = int(max(0, (cy - halfh) * fr.shape[0]))
                    ay1 = int(min(fr.shape[0], (cy + halfh) * fr.shape[0]))
                    crop = fr[ay0:ay1, ax0:ax1]
                    if crop.size == 0:
                        continue
                    sc = 480 / max(crop.shape[0], crop.shape[1])
                    up = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)))
                    img2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                    res2 = lm_task.detect(img2)
                    if not res2.pose_landmarks:
                        continue
                    lm2 = res2.pose_landmarks[0]
                    class P:
                        def __init__(self, p, ox, oy, sc, w, h):
                            self.x = (p.x * up.shape[1] / sc + ox) / w
                            self.y = (p.y * up.shape[0] / sc + oy) / h
                            self.z = p.z
                            self.visibility = p.visibility
                    lm = [P(p, ax0, ay0, sc, fr.shape[1], fr.shape[0]) for p in lm2]
                tiles.append(cv2.resize(draw(fr, lm, f"f{fi} {tag}/{mode}"), (300, 533)))
        lm_task.close()
        rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
        out = ROOT / f"screenshots/_probe_pose_{tag}.jpg"
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("→", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
