#!/usr/bin/env python3
"""ROI 手部探针：把腕点附近裁剪放大后再送 HandLandmarker，检验检出率与几何。

坐标约定（已核验）：videoBox.w/h = 360/640（缩略空间），源视频帧为 720x1280 → 像素需 ×2。
腕点像素 = ((frac.x*w + pose.x*torsoPx) * 2, ((1-frac.y)*h - pose.y*torsoPx) * 2)

用法: tools/.venv/bin/python tools/probe_hand_roi.py <视频> <motion.json> [帧号...]
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
IDX = {"left_wrist": 9, "right_wrist": 10}
SCALE = 2          # videoBox 缩略空间 → 源帧像素


def wrist_px(mo, fi, side):
    vb = mo["videoBox"]
    p = mo["poses"][fi]
    q = p[IDX["left_wrist" if side == "L" else "right_wrist"]] if p else None
    if not q:
        return None
    tp = vb["torsoPx"][fi]
    hx = vb["frac"][fi][1] * vb["w"]
    hy = (1 - vb["frac"][fi][2]) * vb["h"]
    return ((hx + q[0] * tp) * SCALE, (hy - q[1] * tp) * SCALE, q[2])


def main() -> int:
    video, mo_path = Path(sys.argv[1]), Path(sys.argv[2])
    frames = [int(x) for x in sys.argv[3:]] or [96, 173, 240, 300]
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2)
    lm = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video))
    want = set(frames)
    i = 0
    hits = 0
    tot = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            tp = mo["videoBox"]["torsoPx"][i]
            print(f"--- frame {i} t={i/24:.2f}s torsoPx={tp:.1f}")
            for side in ("L", "R"):
                w = wrist_px(mo, i, side)
                if not w:
                    print(f"  {side}: 无腕点")
                    continue
                cx, cy = int(round(w[0])), int(round(w[1]))
                half = max(70, int(tp * SCALE * 0.8))
                x0, y0 = max(0, cx - half), max(0, cy - half)
                x1, y1 = min(fr.shape[1], cx + half), min(fr.shape[0], cy + half)
                crop = fr[y0:y1, x0:x1]
                if crop.size == 0:
                    print(f"  {side}: 空裁剪")
                    continue
                sc = 320 / max(crop.shape[0], crop.shape[1])
                up = cv2.resize(crop, (max(32, int(crop.shape[1] * sc)), max(32, int(crop.shape[0] * sc))))
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                res = lm.detect(img)
                tot += 1
                if not res.hand_landmarks:
                    print(f"  {side}: 未检出 (crop {crop.shape[1]}x{crop.shape[0]})")
                    continue
                hits += 1
                for h, wl, hw in zip(res.handedness, res.hand_landmarks, res.hand_world_landmarks):
                    W = np.array([[p.x, p.y, p.z] for p in wl])
                    X = np.array([[p.x, p.y, p.z] for p in hw])
                    wr = W[0]
                    dx = (wr[0] * up.shape[1] - up.shape[1] / 2) / sc + x0
                    dy = (wr[1] * up.shape[0] - up.shape[0] / 2) / sc + y0
                    fwd = X[9] - X[0]
                    across = X[5] - X[17]
                    palm = np.cross(fwd, across)
                    print(f"  {side}: 检出 {h[0].category_name}({h[0].score:.2f}) 腕误差 {np.hypot(dx-w[0], dy-w[1]):.0f}px "
                          f"|fwd|={np.linalg.norm(fwd):.3f} |across|={np.linalg.norm(across):.3f} "
                          f"fwd={np.round(fwd,3)} across={np.round(across,3)} palm={np.round(palm,3)}")
        i += 1
    cap.release()
    lm.close()
    print(f"检出率 {hits}/{tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
