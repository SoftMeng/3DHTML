#!/usr/bin/env python3
"""手部关键点探针：确定 MediaPipe Hands 的 world_landmarks 坐标系约定与检出率。

用法: tools/.venv/bin/python tools/probe_hand.py <视频> [帧号...]
输出: 每帧的检出数、handedness、世界系与图像系的方向一致性、掌法向在世界系里的取值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "tools" / "models" / "hand_landmarker.task"


def main() -> int:
    video = Path(sys.argv[1])
    frames = [int(x) for x in sys.argv[2:]] or [0, 96, 156, 240, 300]
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    lm = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video))
    want = set(frames)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            ts = int(i * 1000 / 24)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, ts)
            print(f"--- frame {i} (t={i/24:.2f}s) hands={len(res.hand_landmarks)}")
            for h, wl, hw in zip(res.handedness, res.hand_landmarks, res.hand_world_landmarks):
                label = h[0].category_name
                score = h[0].score
                W = np.array([[p.x, p.y, p.z] for p in wl])
                X = np.array([[p.x, p.y, p.z] for p in hw])
                wrist = X[0]
                fwd = X[9] - wrist                 # 腕 → 中指根（沿手指）
                across = X[5] - X[17]              # 小指根 → 食指根（掌横向）
                palm = np.cross(fwd, across)
                # 图像系同向向量（y 向下）
                fwd2 = W[9] - W[0]
                across2 = W[5] - W[17]
                def cos(a, b):
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na < 1e-9 or nb < 1e-9:
                        return float("nan")
                    return float(np.dot(a, b) / (na * nb))
                print(f"  {label}({score:.2f}) |fwd|={np.linalg.norm(fwd):.3f}m "
                      f"cos(world_fwd,img_fwd)={cos(fwd[:2], fwd2[:2]):+.3f} "
                      f"cos(world_across,img_across)={cos(across[:2], across2[:2]):+.3f} "
                      f"world_fwd={np.round(fwd,3)} world_across={np.round(across,3)} "
                      f"palm={np.round(palm,3)}")
                print(f"     img fwd={np.round(fwd2,3)} across={np.round(across2,3)} "
                      f"world wrist={np.round(wrist,3)}")
        i += 1
    cap.release()
    lm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
