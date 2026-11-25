#!/usr/bin/env python3
"""从源视频离线提取人体姿态，生成 *.motion.json 供查看器「模仿源视频」使用。

用法:
  python3 tools/extract_motion.py 从该原始视频中提取到的JSON.mp4 --spec 示例.json

输出: 与规格同名 → 示例.motion.json
坐标约定: 原点=髋中点, x+=主体左侧, y+=向上, 长度单位=肩中点到髋中点的躯干长度。
依赖: numpy opencv-python onnxruntime（模型首次运行自动下载到 tools/models/）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_pipeline as MP
import depth_reconstruct as DR
import pose_repair as PR

MODEL_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "movenet_singlepose_lightning.onnx"
KEYS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
IDX = {k: i for i, k in enumerate(KEYS)}
CORE = [IDX[k] for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                         "left_ankle", "right_ankle")]
WORK_W = 360
# 表演者身高 / 躯干长（肩中点→髋中点）—— 人物常量，由四帧站姿在原生分辨率上量取。
# 画面基准的可见身高占比 = K × 躯干像素长，与姿态无关（见 video_box 注释）。
VISIBLE_K = 3.62


def load_model() -> "ort.InferenceSession":
    import onnxruntime as ort
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载姿态模型 → {MODEL_PATH.name}（仅首次）…", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"已缓存 {MODEL_PATH.stat().st_size//1024} KiB", file=sys.stderr)
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def decode(video: Path, work_w: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        work_h = round(h * work_w / w)
        frames.append(cv2.cvtColor(cv2.resize(f, (work_w, work_h), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit("视频无帧")
    return frames, fps


def infer_crop(sess, img, box):
    """在 box（正方形扩张）内裁剪检测；返回图像坐标点与置信度。"""
    H, W = img.shape[:2]
    if box is None:
        crop, ox, oy = img, 0, 0
    else:
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(40.0, max(x1 - x0, y1 - y0) / 2 * 1.15)
        cx = min(max(cx, half), W - half)
        cy = min(max(cy, half), H - half)
        ox, oy = int(cx - half), int(cy - half)
        crop = img[oy:oy + int(half * 2), ox:ox + int(half * 2)]
    if crop.size == 0:
        return None, None
    r = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_AREA)
    out = sess.run(None, {"input": r[None].astype(np.int32)})[0].reshape(-1, 3)
    sx, sy = float(crop.shape[1]), float(crop.shape[0])   # 输出为 0-1 归一化坐标
    pts = np.stack([out[:, 1] * sx + ox, out[:, 0] * sy + oy], axis=1)
    return pts, out[:, 2]


def track_video(sess, frames):
    """逐帧跟踪式检测：上一帧人体框 → 裁剪推理；失败回退全图。

    首帧只能整帧检测，人体在画面里很小 → 定位很粗（腿抬起的姿态会被判成双脚着地）。
    首帧出框后再用该框裁剪复检一次，精度显著提升。
    """
    results, box = [], None
    for img in frames:
        pts, sc = infer_crop(sess, img, box)
        conf = float(np.mean([sc[k] for k in CORE])) if sc is not None else 0.0
        if conf < 0.25:
            pts2, sc2 = infer_crop(sess, img, None)
            if pts2 is not None:
                c2 = float(np.mean([sc2[k] for k in CORE]))
                if c2 > conf:
                    pts, sc, conf = pts2, sc2, c2
        if pts is not None and box is None:          # 首帧：用检测框复检一次
            xs, ys = pts[CORE, 0], pts[CORE, 1]
            refine = (max(0, xs.min()-40), max(0, ys.min()-40),
                      min(img.shape[1], xs.max()+40), min(img.shape[0], ys.max()+40))
            pts3, sc3 = infer_crop(sess, img, refine)
            if pts3 is not None:
                c3 = float(np.mean([sc3[k] for k in CORE]))
                if c3 > conf:
                    pts, sc, conf = pts3, sc3, c3
        if pts is None or conf < 0.12:
            results.append(None)
            box = None
            continue
        xs, ys = pts[CORE, 0], pts[CORE, 1]
        raw = (max(0, xs.min() - 30), max(0, ys.min() - 30),
               min(img.shape[1], xs.max() + 30), min(img.shape[0], ys.max() + 30))
        box = raw if box is None else tuple(0.6 * raw[i] + 0.4 * box[i] for i in range(4))
        results.append((pts, sc, conf))
    return results


def median_filter(results, k=3):
    """3 帧中值滤波去单帧抖动（保序不保形，仅用于坐标）。"""
    out = []
    n = len(results)
    for i, r in enumerate(results):
        if r is None:
            out.append(None)
            continue
        pts, sc, conf = r
        pts = pts.copy()
        for j in range(len(KEYS)):
            cand = []
            for d in range(-(k // 2), k // 2 + 1):
                q = results[min(max(i + d, 0), n - 1)]
                if q is not None and q[1][j] >= 0.2:
                    cand.append(q[0][j])
            if len(cand) >= 2:
                pts[j] = np.median(np.array(cand), axis=0)
        out.append((pts, sc, conf))
    return out


# ---------- v2：修复 → 逐帧归一化 → 去噪 → 语义通道 ----------
GENERIC_JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                  "left_hip", "right_hip", "left_knee", "right_knee"]
CHAINS = [("left_wrist", "left_elbow", "left_shoulder"),
          ("right_wrist", "right_elbow", "right_shoulder"),
          ("left_ankle", "left_knee", "left_hip"),
          ("right_ankle", "right_knee", "right_hip")]


def repair_pixels(pix):
    """像素域置信度修复：一般关节线性插值，末端关节（腕/踝）链空间重建。

    入参：逐帧的像素系关键点 [[x,y,score]×17]（缺检帧为 None）。
    """
    pix = [[[float(q[0]), float(q[1]), float(q[2])] if q else None for q in frame] if frame else None
           for frame in pix]
    for k in GENERIC_JOINTS:
        MP.repair_joint(pix, IDX[k])
    for end, mid, root in CHAINS:
        MP.repair_end_joint(pix, IDX[end], IDX[mid], IDX[root])
    return pix


def torso_px_series(pix):
    """每帧的躯干像素长（肩中点→髋中点）——把归一化姿态换算回画面比例用。"""
    out = []
    for p in pix:
        if p is None:
            out.append(0.0); continue
        ls, rs = p[IDX["left_shoulder"]], p[IDX["right_shoulder"]]
        lh, rh = p[IDX["left_hip"]], p[IDX["right_hip"]]
        if not (ls and rs and lh and rh):
            out.append(0.0); continue
        out.append(round(float(np.hypot((ls[0]+rs[0])/2-(lh[0]+rh[0])/2,
                                        (ls[1]+rs[1])/2-(lh[1]+rh[1])/2)), 2))
    return out


def video_box(pix, frame_w, frame_h, torso_px):
    """每帧的画面基准（供查看器「对照机位」复现视频取景）：
    [主体像素高占比, 髋中点屏幕x, 髋中点屏幕y]（屏幕坐标：y 向上为正）。

    水平基准取**髋中点**而不是关键点包围盒中心：伸出手臂会把包围盒中心拉偏，
    导致 3D 人偶与视频表演者出现系统性错位。

    可见身高（占比第 0 列）= **K × 躯干像素长**，K 是「表演者身高/躯干长」——
    人物常量，与姿态无关，因此画面基准不会随抬腿/下蹲抖动。
    旧实现按「耳→发顶 0.73 躯干、踝→鞋底 0.38 躯干」逐帧换算，原生分辨率目视
    实测发顶被算到头顶上方 67px、鞋底被算到脚下 119px，可见身高高估 13%，
    「对照机位」据此把相机推远，人偶比视频里的表演者明显偏小。
    K 由四帧站姿实测（发顶/鞋底在 720×1280 原生帧上量取）：
      t=4.04 → 3.67   t=7.21 → 3.46   t=8.17 → 3.68   t=8.75 → 3.56
    与人体解剖一致（肩关节→大转子 ≈ 身高 27% → 身高/躯干 ≈ 3.7），取 3.62。
    """
    out = []
    for i, p in enumerate(pix):
        if p is None:
            out.append([0.0, 0.5, 0.5])
            continue
        xs = [q[0] for q in p if q and q[2] >= .2]
        ys = [q[1] for q in p if q and q[2] >= .2]
        if not xs:
            out.append([0.0, 0.5, 0.5])
            continue
        tp = torso_px[i] if i < len(torso_px) else 0.0
        fh = VISIBLE_K * tp / max(1, frame_h) if tp > 0 else 0.0
        lh, rh = p[IDX["left_hip"]], p[IDX["right_hip"]]
        if lh and rh:
            hx = (lh[0] + rh[0]) / 2
            hy = (lh[1] + rh[1]) / 2
        else:
            hx = (max(xs) + min(xs)) / 2
            hy = (max(ys) + min(ys)) / 2
        out.append([round(fh, 4), round(hx / max(1, frame_w), 4), round(1 - hy / max(1, frame_h), 4)])
    return out


def assemble_v2(pix, fps, frame_w, frame_h, video_name, video_bytes, model, poses3d=None,
                height_m=None, hfov_deg=None, img_w=None, img_h=None):
    """产出动作文件：干净姿态 + 语义通道 + 视频取景 + （可选）透视缩短深度重构。

    pix 已是修复后的像素系关键点；后端无关 —— MoveNet 与 MediaPipe 共用同一条管线。
    给定人物身高与源视频水平 FOV 时升级为 v3：新增 pose3d（相机系三维骨架）。
    """
    poses = MP.normalize_per_frame(pix, IDX)
    poses = MP.denoise_poses(poses, fps)
    # 左右归属翻转修复：检测器逐帧独立猜左右，转身/遮脸时会整段标反
    poses, flip_stats = PR.repair_lateral_flip(poses, KEYS)
    torso_px = MP.median_filter(torso_px_series(pix), 5)
    chan = {
        "yaw": [round(v, 4) for v in MP.derive_yaw(poses, IDX)],
        "roll": [round(v, 4) for v in MP.derive_roll(poses, IDX)],
        "root": MP.derive_root(poses, IDX),
        "contact": MP.derive_contact(poses, IDX),
        "conf": MP.frame_conf(poses, IDX, CORE),
    }
    body = MP.body_stats(poses, IDX)
    quality = statistics.median([c for c in chan["conf"] if c > 0]) if chan["conf"] else 0.0
    vbox = {"w": frame_w, "h": frame_h, "K": VISIBLE_K,
            "frac": MP.median_filter2(video_box(pix, frame_w, frame_h, torso_px), 5),
            "torsoPx": torso_px}
    payload = {
        "version": 2,
        "video": video_name,
        "videoBytes": video_bytes,
        "fps": round(fps, 4),
        "frames": len(poses),
        "duration": round(len(poses) / fps, 4),
        "model": model,
        "frame": "origin=hip-mid, x+=subject-left, y+=up, unit=torso(per-frame); "
                 "poses=repaired+denoised+flip-fixed; chan={yaw,roll,root,contact,conf}",
        "keys": KEYS,
        "quality": round(quality, 4),
        "videoBox": vbox,
        "body": body,
        "poses": poses,
        "chan": chan,
    }
    if poses3d is not None:
        payload["poses3d"] = poses3d        # MediaPipe 原生世界坐标（已知噪声大，已弃用）
    # v3：透视缩短深度重构 —— 2D + 解剖骨长 → 相机系三维骨架（米）
    if height_m and hfov_deg and img_w and img_h:
        rec = DR.reconstruct(poses, KEYS, vbox, height_m, hfov_deg, img_w, img_h, chan["yaw"])
        payload["version"] = 3
        payload["pose3d"] = rec["pose3d"]
        payload["pose3dConf"] = rec["conf"]
        payload["pose3dMeta"] = {
            "frame": rec["frame"], "depth": rec["depth"], "quality": rec["quality"],
            "height_m": height_m, "hfov_deg": hfov_deg, "img_w": img_w, "img_h": img_h}
        payload["frame"] += ("; pose3d: 米/髋中点原点/相机系(x=画面右,y=上,z=远离相机), "
                             "由透视缩短|Δz|=√(L²−d²)重构")
    return payload



# ---------- MediaPipe 后端：2D + 真 3D 世界坐标 + 根轨迹 ----------
# pose 检测器可选 lite / full / heavy：lite 快但腕/肘偶发贴到头发或衣袖上
# （t=2.5 右腕被检到肩位、左右归属整段标反都是 lite 的实际观测），full 显著更稳。
def mp_model_paths(variant: str = "lite") -> tuple:
    p = Path(__file__).resolve().parent / "models" / f"pose_landmarker_{variant}.task"
    u = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
         f"pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task")
    return p, u
MP_MODEL, MP_URL = mp_model_paths("lite")
# BlazePose 33 点布局 → 本项目的 17 点键名
MP_MAP = {0: "nose", 3: "left_ear", 4: "right_ear", 11: "left_shoulder", 12: "right_shoulder",
          13: "left_elbow", 14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
          23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
          27: "left_ankle", 28: "right_ankle"}
MP_NAME_IDX = {v: k for k, v in MP_MAP.items()}   # 键名 → MediaPipe 点位


def _download(url, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"下载模型 {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, path)


def extract_mediapipe(frames, fps, every=1, variant="lite"):
    """返回 (pix, poses3d, roots, quality)。

    pix: 每帧 17×[x,y,score] 的**像素系**原始关键点（未修复/未归一化），无检出的帧为 None。
    poses3d: 每帧 17×[x,y,z]，米，髋中点原点，x+=主体左，y+=上，z+=远离相机。
    roots: 每帧 [髋中点x, 髋中点y, 躯干像素长]（像素系）。
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path, model_url = mp_model_paths(variant)
    _download(model_url, model_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    lm = vision.PoseLandmarker.create_from_options(opts)

    h, w = frames[0].shape[:2]
    n = len(frames)
    pix = [None] * n            # 像素系关键点
    raw3d = [None] * n          # 米制世界坐标
    roots = [None] * n          # 髋中点像素位置与躯干像素尺度
    for i, img in enumerate(frames):
        ts = int(i * 1000 / fps)
        mimg = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(mimg, ts)
        if not res.pose_landmarks:
            continue
        lm2d, lm3d = res.pose_landmarks[0], res.pose_world_landmarks[0]
        p2, p3 = [], []
        for k in KEYS:
            j = MP_NAME_IDX.get(k)
            if j is None:
                # 缺失的关键点必须写成 None：写成 [0,0,0] 会被当成「图像左上角的有效点」，
                # 画面基准（videoBox）会据此把头顶算到画外 → 3D 人偶被放大
                p2.append(None); p3.append(None); continue
            a, b = lm2d[j], lm3d[j]
            p2.append([float(a.x) * w, float(a.y) * h, float(getattr(a, "visibility", 1.0))])
            p3.append([round(float(b.x), 4), round(-float(b.y), 4), round(-float(b.z), 4)])
        pix[i] = p2
        raw3d[i] = p3
        hipx = (p2[IDX["left_hip"]][0] + p2[IDX["right_hip"]][0]) / 2
        hipy = (p2[IDX["left_hip"]][1] + p2[IDX["right_hip"]][1]) / 2
        shx = (p2[IDX["left_shoulder"]][0] + p2[IDX["right_shoulder"]][0]) / 2
        shy = (p2[IDX["left_shoulder"]][1] + p2[IDX["right_shoulder"]][1]) / 2
        roots[i] = [hipx, hipy, float(np.hypot(shx - hipx, shy - hipy))]
    lm.close()

    valid = [i for i in range(n) if pix[i]]
    if len(valid) < n * 0.5:
        raise SystemExit(f"MediaPipe 仅检出 {len(valid)}/{n} 帧")
    conf = statistics.mean([statistics.mean([q[2] for q in pix[i] if q]) for i in valid])
    return pix, raw3d, roots, float(conf)


def fill_missing_frames(pix):
    """整帧缺检时按邻帧线性插值补齐（关节级修复之前先要有帧）。"""
    n = len(pix)
    out = [list(p) if p else None for p in pix]
    idx = [i for i in range(n) if out[i]]
    if not idx:
        raise SystemExit("无有效帧")
    for i in range(n):
        if out[i]:
            continue
        lo = max([k for k in idx if k < i], default=idx[0])
        hi = min([k for k in idx if k > i], default=idx[-1])
        a, b = out[lo], out[hi]
        t = 0.0 if hi == lo else (i - lo) / (hi - lo)
        frame = []
        for j in range(len(a)):
            if a[j] is None or b[j] is None:
                frame.append(None)
                continue
            frame.append([round(a[j][0] + (b[j][0] - a[j][0]) * t, 3),
                          round(a[j][1] + (b[j][1] - a[j][1]) * t, 3),
                          round(min(a[j][2], b[j][2]), 3)])
        out[i] = frame
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="源视频 → 姿态动作文件")
    ap.add_argument("video", type=Path)
    ap.add_argument("--spec", type=Path, default=None, help="配对规格（决定输出文件名）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=1, help="每 N 帧采样")
    ap.add_argument("--preview", type=Path, default=None, help="输出骨架预览图")
    ap.add_argument("--backend", choices=["auto", "movenet", "mp"], default="auto",
                    help="auto: 有 mediapipe 就用（含 3D），否则 MoveNet")
    ap.add_argument("--pose-model", choices=["lite", "full", "heavy"], default="lite",
                    help="MediaPipe pose landmarker 档位（full 比 lite 稳，代价是提取更慢）")
    args = ap.parse_args()

    if args.out:
        out = args.out
    elif args.spec:
        out = args.spec.with_suffix("").name + ".motion.json"
    else:
        out = args.video.with_suffix("").name + ".motion.json"

    t0 = time.time()

    def spec_calib():
        """v3 深度重构需要：人物身高（米）+ 源视频水平 FOV（度）+ 原生分辨率。"""
        if not args.spec:
            return {}
        try:
            sp = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        except Exception:
            return {}
        prim = next((s for s in sp.get("subjects", [])
                     if "person" in str(s.get("class", ""))), None)
        cam = sp.get("camera", {})
        cap = cv2.VideoCapture(str(args.video))
        iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        out = {}
        if prim and prim.get("size_m"):
            out["height_m"] = float(prim["size_m"][2])
        if cam.get("fov_deg"):
            out["hfov_deg"] = float(cam["fov_deg"])
        if iw > 0 and ih > 0:
            out["img_w"], out["img_h"] = iw, ih
        return out

    calib = spec_calib()
    backend = args.backend
    if backend == "auto":
        try:
            import mediapipe  # noqa: F401
            backend = "mp"
        except Exception:
            backend = "movenet"
    if backend == "mp":
        frames, fps = decode(args.video, WORK_W)
        if args.every > 1:
            frames = frames[::args.every]
            fps /= args.every
        print(f"解码 {len(frames)} 帧 @ {fps:.3f} fps（{args.video.name}）· backend=mediapipe", file=sys.stderr)
        pix, poses3d, roots, quality = extract_mediapipe(frames, fps, args.every, args.pose_model)
        pix = repair_pixels(fill_missing_frames(pix))
        payload = assemble_v2(pix, fps, frames[0].shape[1], frames[0].shape[0],
                              args.video.name, args.video.stat().st_size,
                              f"mediapipe-pose-landmarker-{args.pose_model}", poses3d=poses3d, **calib)
        payload["root"] = [[round(r[0], 2), round(r[1], 2), round(r[2], 2)] if r else None for r in roots]
        Path(out).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        print(f"✓ {out}  {payload['frames']} 帧 · conf {quality:.3f} · 3D · {time.time()-t0:.1f}s", file=sys.stderr)
        return 0

    sess = load_model()
    frames, fps = decode(args.video, WORK_W)
    if args.every > 1:
        frames = frames[::args.every]
        fps /= args.every
    print(f"解码 {len(frames)} 帧 @ {fps:.3f} fps（{args.video.name}）· backend=movenet", file=sys.stderr)

    results = track_video(sess, frames)
    results = median_filter(results)
    valid = [r for r in results if r]
    if len(valid) < len(frames) * 0.5:
        print(f"警告: 仅 {len(valid)}/{len(frames)} 帧检测到人体", file=sys.stderr)

    payload = assemble_v2(repair_pixels([[ [float(r[0][j][0]), float(r[0][j][1]), float(r[1][j])]
                                          for j in range(len(KEYS))] for r in results]),
                          fps, frames[0].shape[1], frames[0].shape[0],
                          args.video.name, args.video.stat().st_size,
                          "movenet-singlepose-lightning", **calib)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"✓ {out}  {payload['frames']} 帧 · conf {payload['quality']:.3f} · v2 · {time.time()-t0:.1f}s",
          file=sys.stderr)

    if args.preview:
        idx = list(range(0, len(frames), max(1, len(frames) // 12)))[:12]
        tiles = []
        edges = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)]
        for i in idx:
            vis = frames[i].copy()
            r = results[i]
            if r:
                pts, sc = r[0], r[1]
                for a, b in edges:
                    if sc[a] > .2 and sc[b] > .2:
                        cv2.line(vis, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                                 (0, 255, 255), 2)
                for j in range(len(KEYS)):
                    if sc[j] > .2:
                        cv2.circle(vis, tuple(pts[j].astype(int)), 3, (0, 255, 0), -1)
                cv2.putText(vis, f"c={r[2]:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 2)
            tiles.append(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(args.preview), np.vstack(rows))
        print(f"预览 → {args.preview}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
