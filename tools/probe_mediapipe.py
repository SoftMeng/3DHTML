"""探针：MediaPipe Pose Landmarker 的 3D/2D 质量与耗时（不入库，仅评估）。"""
import time, sys, os
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'tools/models/pose_landmarker_lite.task')
VIDEO = os.path.join(ROOT, '从该原始视频中提取到的JSON.mp4')

opts = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
lm = vision.PoseLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
# MediaPipe BlazePose 33 点：11/12 肩、13/14 肘、15/16 腕、23/24 髋、25/26 膝、27/28 踝
NAMES = {0:'nose',1:'left_eye',2:'right_eye',3:'left_ear',4:'right_ear',
         11:'left_shoulder',12:'right_shoulder',13:'left_elbow',14:'right_elbow',
         15:'left_wrist',16:'right_wrist',23:'left_hip',24:'right_hip',
         25:'left_knee',26:'right_knee',27:'left_ankle',28:'right_ankle'}
idx = {n: i for i, n in NAMES.items()}

t0 = time.time(); frames = 0; worst = 0.0
samples = []
while True:
    ok, img = cap.read()
    if not ok: break
    ts = int(frames * 1000 / fps)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t1 = time.time()
    res = lm.detect_for_video(img, ts)
    dt = time.time() - t1
    worst = max(worst, dt); frames += 1
    if frames in (1, 40, 80, 120, 150, 160, 170, 200, 260, 320):
        if res.pose_landmarks:
            w = res.pose_landmarks[0]
            wl = res.pose_world_landmarks[0]
            samples.append((frames, frames/fps,
                            [(w[idx[n]].x, w[idx[n]].y) for n in ('left_shoulder','right_shoulder')],
                            [(wl[idx[n]].x, wl[idx[n]].y, wl[idx[n]].z) for n in ('left_shoulder','right_shoulder','left_elbow','left_wrist','right_elbow','right_wrist')]))
        else:
            samples.append((frames, frames/fps, None, None))
    if frames >= 346: break
cap.release(); lm.close()
tot = time.time() - t0
print(f'帧数={frames} 总耗时={tot:.1f}s 单帧均值={tot/frames*1000:.1f}ms 最差={worst*1000:.0f}ms')
for f, t, w, wl in samples:
    if w is None:
        print(f'  f{f} t={t:.2f}s 无人'); continue
    shw = abs(w[0][0]-w[1][0])
    print(f'  f{f} t={t:.2f}s 肩宽(图像)=', round(shw, 3))
    if wl:
        ls, rs = wl[0], wl[1]
        print(f'      3D 肩L=({ls[0]:+.3f},{ls[1]:+.3f},{ls[2]:+.3f}) 肩R=({rs[0]:+.3f},{rs[1]:+.3f},{rs[2]:+.3f})')
        print(f'      3D 肘L={tuple(round(v,3) for v in wl[3])} 腕L={tuple(round(v,3) for v in wl[4])}')
