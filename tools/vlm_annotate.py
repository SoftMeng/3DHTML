#!/usr/bin/env python3
"""用多模态模型给动作文件补语义通道：场景描述 + 动作短语（Schema 的多模态产出）。

产出写回动作文件（version → 4）：
  chan.semantic.scene    场景描述（一次调用）
  chan.semantic.facing   面向（toward_camera / side / away）
  chan.semantic.action   稀疏的动作短语 [{t, text}]
  chan.semantic.model    产出该通道的模型名（可追溯）

**不标掌向**：实测同帧重复提问的掌向标签翻转率 34.8%，不足以驱动渲染
（见 docs/constraint/vlm_boundary.md）。需要时加 --palm（仅作参考，不进渲染管线）。

结果缓存到 tools/vlm_cache.json：离线重跑不再调用 API。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm import VLM, VLMError  # noqa: E402
from probe_vlm import build_truth, enc, person_crop  # noqa: E402

ACTION_PROMPT = """You are annotating one frame of a person's performance video. Answer with a JSON object only.

Give:
- "action": at most 6 words describing what the person is doing in this frame (English).
- "facing": where the person faces — "toward_camera", "side", or "away".
- "visible_hands": are both hands visible outside the sleeves? "both"/"one"/"none".

Return only the JSON object."""

SCENE_PROMPT = """Describe this video frame in one sentence for a 3D-motion dataset:
what the person is doing, where they are, and what they are wearing.
Answer with a JSON object only, keys: "scene", "facing"."""

PALM_PROMPT = """Answer with a JSON object only.
For the person's own LEFT and RIGHT hand, say which way the palm faces:
"toward_camera" / "away_from_camera" / "toward_body" / "toward_other_hand" / "sideways" /
"not_visible" / "unknown".
Keys: "palmL", "palmR", "confidence" (0-1), "action" (at most 6 words)."""


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(path: str, cache: dict) -> None:
    tmp = path + ".tmp"
    json.dump(cache, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def as_dict(r):
    """模型偶尔返回数组或标量 → 归一成 dict（字段缺失即不可用）。"""
    return r if isinstance(r, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="示例.motion.json")
    ap.add_argument("--video", default="从该原始视频中提取到的JSON.mp4")
    ap.add_argument("--step", type=int, default=16, help="每隔几帧标注一次")
    ap.add_argument("--cache", default="tools/vlm_cache.json")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--palm", action="store_true",
                    help="一并标注掌向（默认关闭：实测翻转率 34.8%，不进渲染管线）")
    args = ap.parse_args()

    motion = json.load(open(args.motion, encoding="utf-8"))
    n = len(motion.get("poses") or [])
    fps = motion.get("fps") or 24.0
    cache = load_cache(args.cache)
    vlm = VLM()

    cap = cv2.VideoCapture(args.video)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    samples = list(range(0, len(frames), args.step))
    print(f"[annotate] 视频 {len(frames)} 帧，步长 {args.step} → {len(samples)} 次调用")

    actions = []
    done = 0
    for idx in samples:
        key = f"act:{idx}"
        if key in cache:
            r = as_dict(cache[key])
            if r.get("action"):
                actions.append({"t": round(idx/fps, 3), "text": str(r["action"])[:80]})
            continue
        crop, _ = person_crop(frames[idx], motion, idx)
        if crop is None:
            continue
        try:
            r = as_dict(vlm.ask_json(ACTION_PROMPT, image=enc(crop), max_tokens=200))
            cache[key] = r
            done += 1
            if r.get("action"):
                actions.append({"t": round(idx/fps, 3), "text": str(r["action"])[:80]})
            print(f"[annotate] {done}/{len(samples)} 帧 {idx}: {r.get('action')}")
        except VLMError as e:
            print(f"[annotate] 帧 {idx} 失败：{e}")
        if done % 8 == 0:
            save_cache(args.cache, cache)
    save_cache(args.cache, cache)

    scene = cache.get("scene")
    if not scene and not args.no_scene and frames:
        mid = frames[len(frames)//2]
        try:
            scene = as_dict(vlm.ask_json(SCENE_PROMPT,
                                         image=enc(cv2.resize(mid, (480, int(mid.shape[0]*480/mid.shape[1])))),
                                         max_tokens=200))
            cache["scene"] = scene
            save_cache(args.cache, cache)
            print(f"[annotate] scene: {scene}")
        except VLMError as e:
            print(f"[annotate] scene 失败：{e}")

    palm = None
    if args.palm:
        palm = {"L": [None]*n, "R": [None]*n}
        for idx in samples:
            key = f"palm:{idx}"
            if key not in cache:
                crop, _ = person_crop(frames[idx], motion, idx)
                if crop is None:
                    continue
                try:
                    cache[key] = as_dict(vlm.ask_json(PALM_PROMPT, image=enc(crop), max_tokens=220))
                except VLMError as e:
                    print(f"[annotate] 掌向帧 {idx} 失败：{e}")
            r = as_dict(cache.get(key) or {})
            if not r:
                continue
            conf = float(r.get("confidence", 0.0) or 0.0)
            for tag, k in (("L", "palmL"), ("R", "palmR")):
                palm[tag][idx] = {"face": str(r.get(k, "unknown")), "conf": round(conf, 3)}
        for tag in ("L", "R"):
            known = [i for i, v in enumerate(palm[tag]) if v]
            for i in range(n):
                if palm[tag][i] is None and known:
                    j = min(known, key=lambda k: abs(k-i))
                    palm[tag][i] = dict(palm[tag][j])

    chan = motion.setdefault("chan", {})
    if palm:
        chan["palm"] = palm
    sem = {"model": os.environ.get("MY_OPENAI_MODEL_NAME", "?"),
           "scene": (scene or {}).get("scene", ""),
           "facing": (scene or {}).get("facing", ""),
           "action": actions}
    if not args.palm:
        sem["note"] = ("掌向未由多模态模型标注：实测同帧重复提问的标签翻转率 34.8%，"
                       "不足以驱动渲染（docs/constraint/vlm_boundary.md）")
    chan["semantic"] = sem

    motion["version"] = 4
    motion["frame"] = (motion.get("frame", "") +
                       "; chan.semantic: 场景/动作短语（多模态模型产出）").strip("; ")
    json.dump(motion, open(args.motion, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[annotate] 写入 {args.motion}（version 4），动作短语 {len(actions)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
