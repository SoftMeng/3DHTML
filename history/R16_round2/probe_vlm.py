#!/usr/bin/env python3
"""多模态大模型能力基线测量：回答哪些问题、答得准不准、稳不稳定。

问题电池的每个问题都有**独立真值**（来自 MediaPipe 2D 与本项目重构的深度），
用来量化模型的边界，而不是听模型自述。

真值来源与镜像约定：
  源视频未镜像（人正面朝镜头）⇒ 图像左侧 = 主体右侧。
  归一化姿态 x+ = 图像右（= 主体左，因为主体朝镜头时左肩出现在画面右侧）。

用法:
  tools/.venv/bin/python tools/probe_vlm.py --n 18 --out tools/vlm_baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm import VLM, VLMError  # noqa: E402

KEYS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]

PROMPT = """You are annotating one frame of a dance video. Answer strictly with a JSON object.

Rules:
- "left"/"right" always mean the PERSON'S OWN left/right (their left hand, their right arm).
- Use only these exact words where a fixed vocabulary is given.
- If you cannot tell, answer "unknown".

Questions:
1. "handUpL": is the person's own LEFT hand raised ABOVE their LEFT shoulder? ("yes"/"no"/"unknown")
2. "handUpR": is the person's own RIGHT hand raised ABOVE their RIGHT shoulder? ("yes"/"no"/"unknown")
3. "elbowL": is the person's own LEFT arm bent at the elbow? ("bent"/"straight"/"unknown")
4. "elbowR": is the person's own RIGHT arm bent at the elbow? ("bent"/"straight"/"unknown")
5. "facing": where does the person face? ("toward_camera"/"side"/"away"/"unknown")
6. "handNear": which of the person's own hands is CLOSER to the camera? ("left"/"right"/"same"/"unknown")
7. "palmL": which way does the person's own LEFT palm face?
   ("toward_camera"/"away_from_camera"/"toward_body"/"toward_other_hand"/"sideways"/"not_visible"/"unknown")
8. "palmR": same for the person's own RIGHT palm.
9. "action": describe what the person is doing, at most 8 words.

Return only the JSON object with exactly these 9 keys."""


def elbow_angle(a, b, c) -> float | None:
    if not (a and b and c):
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    d = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def build_truth(motion: dict) -> list[dict]:
    """逐帧真值。全部用**主体自身左右**（与 Schema 语义一致，镜像视频天然正确）。"""
    keys = motion["keys"]
    idx = {k: keys.index(k) for k in KEYS if k in keys}
    poses = motion.get("posesD") or motion.get("poses") or []
    p3 = motion.get("pose3d") or []
    yaw = (motion.get("chan") or {}).get("yaw") or []
    out = []
    for i, p in enumerate(poses):
        if not p:
            continue
        g = lambda n: (p[idx[n]] if n in idx else None)
        ls, rs = g("left_shoulder"), g("right_shoulder")
        le, re = g("left_elbow"), g("right_elbow")
        lw, rw = g("left_wrist"), g("right_wrist")
        if not all([ls, rs, le, re, lw, rw]):
            continue
        upL = lw[1] > ls[1]
        upR = rw[1] > rs[1]
        aL = elbow_angle(ls, le, lw)
        aR = elbow_angle(rs, re, rw)
        bent = lambda a: None if a is None else ("bent" if a < 150.0 else "straight")
        y = yaw[i] if i < len(yaw) else 0.0
        facing = "toward_camera" if abs(y) < 0.30 else ("away" if abs(y) > 2.0 else "side")
        near = "unknown"
        if i < len(p3) and p3[i]:
            row = p3[i]
            zw = row[idx["left_wrist"]][2] if idx["left_wrist"] < len(row) and row[idx["left_wrist"]] else None
            zr = row[idx["right_wrist"]][2] if idx["right_wrist"] < len(row) and row[idx["right_wrist"]] else None
            if zw is not None and zr is not None:
                d = zw - zr
                near = "left" if d < -0.02 else ("right" if d > 0.02 else "same")
        out.append(dict(frame=i, handUpL=upL, handUpR=upR,
                        elbowL=bent(aL), elbowR=bent(aR),
                        facing=facing, handNear=near))
    return out


def stratified(truth: list[dict], n: int) -> list[dict]:
    """按真值位掩码分桶取样：保证抽到有区分度的帧（抬手/直臂/转身/深度不一致）。"""
    buckets: dict[tuple, list[dict]] = {}
    for t in truth:
        key = (t["handUpL"], t["handUpR"], t["elbowL"], t["elbowR"],
               t["facing"], t["handNear"])
        buckets.setdefault(key, []).append(t)
    order = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    picked, i = [], 0
    while len(picked) < n and any(b for _, b in order):
        for _, b in order:
            if not b:
                continue
            picked.append(b.pop(0))
            if len(picked) >= n:
                break
        i += 1
        if i > 500:
            break
    return sorted(picked, key=lambda t: t["frame"])


def person_crop(frame, motion, i, width=512, margin=0.18):
    """按归一化姿态 + videoBox 还原像素坐标，裁出主体区域。"""
    vb = motion["videoBox"]
    frac, tp = vb["frac"][i], vb["torsoPx"][i]
    H, W = frame.shape[:2]
    sc = tp * (W / vb["w"])
    idx = {k: motion["keys"].index(k) for k in KEYS if k in motion["keys"]}
    p = (motion.get("posesD") or motion["poses"])[i]
    xs, ys = [], []
    for name in KEYS:
        v = p[idx[name]] if name in idx and p[idx[name]] else None
        if not v:
            continue
        x = frac[1] * W + v[0] * sc
        y = (1 - frac[2]) * H - v[1] * sc
        xs.append(x)
        ys.append(y)
    if not xs:
        return None, None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * margin; x1 += bw * margin
    y0 -= bh * margin; y1 += bh * margin
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W, int(x1)); y1 = min(H, int(y1))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    h = max(1, int(round(crop.shape[0] * width / crop.shape[1])))
    return cv2.resize(crop, (width, h), interpolation=cv2.INTER_AREA), (x0, y0, x1, y1)


def enc(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise VLMError("JPEG 编码失败")
    return buf.tobytes()


REQ_KEYS = ["handUpL", "handUpR", "elbowL", "elbowR", "facing", "handNear", "palmL", "palmR", "action"]


def ask(vlm: VLM, img, tag: str, rec: dict) -> None:
    try:
        rec[tag] = vlm.ask_json(PROMPT, image=enc(img), max_tokens=420, required=REQ_KEYS)
    except VLMError as e:
        rec[tag] = {"_error": str(e)}
        print(f"[probe] {tag} 失败：{e}")


def _yn(v) -> str | None:
    """真值归一化：布尔 → yes/no。"""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return v


def score(records: list[dict], tag: str) -> dict:
    """逐题准确率。分母 = 真值不是 unknown 的帧数。"""
    fields = ["handUpL", "handUpR", "elbowL", "elbowR", "facing", "handNear"]
    out = {}
    for f in fields:
        hit = tot = skip = 0
        for r in records:
            t = _yn(r["truth"].get(f))
            a = (r.get(tag) or {}).get(f)
            if t is None or a is None or t == "unknown" or a == "unknown":
                skip += 1
                continue
            tot += 1
            hit += int(a == t)
        out[f] = dict(acc=round(hit / tot, 3) if tot else None, n=tot, skipped=skip)
    palm = []
    for r in records:
        for f in ("palmL", "palmR"):
            v = (r.get(tag) or {}).get(f)
            if v:
                palm.append(v)
    dist = {}
    for v in palm:
        dist[v] = dist.get(v, 0) + 1
    out["palmLabels"] = dict(n=len(palm), dist=dist)
    return out


def write_doc(res: dict, path: str) -> None:
    """基线报告：模型的**实测能力边界**（不是假设）。"""
    L = []
    L.append("# 多模态大模型能力边界（实测）\n")
    L.append(f"- 模型：`{res.get('model')}`　抽帧 {res.get('n')}　真值来自 MediaPipe 2D 与本项目的 yaw/pose3d")
    L.append("- 提问一律用**主体自身左右**（镜像视频天然正确）。\n")
    L.append("## 逐题准确率\n")
    L.append("| 题 | 真值来源 | 整帧 | 主体裁剪 |")
    L.append("|---|---|---|---|")
    src = {"handUpL": "MediaPipe 腕点高于肩点", "handUpR": "同上", "elbowL": "MediaPipe 肘角<150°",
           "elbowR": "同上", "facing": "chan.yaw 转身角", "handNear": "pose3d 深度序（**非独立真值**）"}
    for q in src:
        a = (res.get("scoreFull") or {}).get(q) or {}
        b = (res.get("scoreCrop") or {}).get(q) or {}
        fa = "—" if a.get("acc") is None else f"{a['acc']*100:.0f}%（n={a['n']}）"
        fb = "—" if b.get("acc") is None else f"{b['acc']*100:.0f}%（n={b['n']}）"
        L.append(f"| `{q}` | {src[q]} | {fa} | {fb} |")
    L.append("\n## 重复提问的翻转率（噪声地板）\n")
    rr = res.get("repeatFlipRate")
    if rr and rr.get("rate") is not None:
        L.append(f"- 同帧同题重复提问的标签翻转率：**{rr['rate']*100:.0f}%**（n={rr['n']}）")
    else:
        L.append("- 未测")
    L.append("\n## 掌面标签分布\n")
    for tag, key in (("整帧", "scoreFull"), ("主体裁剪", "scoreCrop")):
        d = ((res.get(key) or {}).get("palmLabels") or {}).get("dist") or {}
        if d:
            L.append(f"- {tag}：" + "，".join(f"{k}×{v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
    L.append("\n## 结论（决定模型在管线里的角色）\n")
    L.append(res.get("conclusion", "（见 tools/vlm_baseline.json）"))
    L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


def build_result(args, records: list[dict]) -> dict:
    res = dict(
        motion=args.motion, n=len(records),
        model=os.environ.get("MY_OPENAI_MODEL_NAME", "?"),
        frames=[r["frame"] for r in records],
        scoreFull=score(records, "full"),
        scoreCrop=score(records, "crop"),
        records=records,
    )
    if args.repeat:
        res["scoreCropRepeat"] = score(records, "crop2")
        flips = tot = 0
        for r in records:
            a, b = r.get("crop") or {}, r.get("crop2") or {}
            for f in ("palmL", "palmR", "handUpL", "handUpR", "facing", "handNear"):
                if a.get(f) and b.get(f):
                    tot += 1
                    flips += int(a[f] != b[f])
        res["repeatFlipRate"] = dict(rate=round(flips / tot, 3) if tot else None, n=tot)
    return res


def dump(args, records: list[dict]) -> None:
    res = build_result(args, records)
    tmp = args.out + ".tmp"
    json.dump(res, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="示例.motion.json")
    ap.add_argument("--video", default="从该原始视频中提取到的JSON.mp4")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--out", default="tools/vlm_baseline.json")
    ap.add_argument("--doc", default="docs/constraint/vlm_boundary.md")
    ap.add_argument("--doc-only", action="store_true", help="不测量，只用已有结果重写报告")
    ap.add_argument("--repeat", action="store_true", help="对 crop 变体重复提问，测噪声地板")
    args = ap.parse_args()

    if args.doc_only:
        res = json.load(open(args.out, encoding="utf-8"))
        write_doc(res, args.doc)
        print(f"[probe] 报告重写 {args.doc}")
        return 0

    motion = json.load(open(args.motion, encoding="utf-8"))
    truth = build_truth(motion)
    picked = stratified(truth, args.n)
    print(f"[probe] 真值帧 {len(truth)}，抽帧 {len(picked)}：{[t['frame'] for t in picked]}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    want = {t["frame"] for t in picked}
    grabbed: dict[int, "object"] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            grabbed[i] = frame.copy()
        i += 1
        if len(grabbed) == len(want):
            break
    cap.release()

    vlm = VLM()
    records = []
    out_path = args.out
    for t in picked:
        idx = t["frame"]
        frame = grabbed.get(idx)
        if frame is None:
            print(f"[probe] 帧 {idx} 读取失败")
            continue
        full = cv2.resize(frame, (480, int(round(frame.shape[0] * 480 / frame.shape[1]))),
                          interpolation=cv2.INTER_AREA)
        crop, box = person_crop(frame, motion, idx)
        rec = dict(frame=idx, t=round(idx / fps, 3), truth=t, cropBox=box)
        ask(vlm, full, "full", rec)
        if crop is not None:
            ask(vlm, crop, "crop", rec)
            if args.repeat:
                ask(vlm, crop, "crop2", rec)
        records.append(rec)
        print(f"[probe] 帧 {idx} done  full={list((rec.get('full') or {}).keys())[:3]}", flush=True)
        dump(args, records)          # 逐帧落盘：中途失败也不丢已测数据
    cap.release()

    res = build_result(args, records)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_doc(res, args.doc)
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, ensure_ascii=False, indent=1))
    print(f"[probe] 写入 {args.out} · {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
