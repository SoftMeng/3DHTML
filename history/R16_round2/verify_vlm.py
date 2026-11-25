#!/usr/bin/env python3
"""轨迹一致性交叉校验（难点四：准确性校验）。

同一个多模态模型，分别看**源视频帧**与**3D 渲染帧**，回答同一组问题；
两次回答的一致率 = 轨迹与视频的一致性得分。

它不依赖本项目的解析式，也不依赖渲染层自报的量 —— 是一条**独立**的证据链。
局限：模型的答案本身有噪声，一致率是相对量不是绝对准确率；
      只用于横向比较不同版本，不作为发布门槛。

用法:
  node tools/probe_render.mjs --view=match --times=2.5,7.21,10,12.5 --tag=r16
  tools/.venv/bin/python tools/verify_vlm.py --tag=r16 --times=2.5,7.21,10,12.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm import VLM, VLMError  # noqa: E402
from probe_vlm import enc  # noqa: E402

QUESTIONS = {
    "handHigher": "Which of the person's own hands is higher? (left/right/same)",
    "elbowBent": "Are the person's elbows bent? (both/one/none)",
    "palmFacing": "Which way do the person's palms face? (toward_camera/away_from_camera/"
                  "toward_each_other/toward_body/sideways/not_visible)",
    "facing": "Where does the person face? (toward_camera/side/away)",
}

PROMPT = ("You are annotating one frame of a person's performance. Answer with a JSON object only.\n"
          "Use the person's own left/right. If you cannot tell, answer \"unknown\".\n"
          "Questions:\n"
          + "\n".join(f'{i+1}. "{k}": {v}' for i, (k, v) in enumerate(QUESTIONS.items()))
          + '\nAlso give "action": at most 6 words describing what the person is doing.\n'
          'Return only the JSON object.')


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


def fit(img, width=420):
    h = max(1, int(round(img.shape[0] * width / img.shape[1])))
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="从该原始视频中提取到的JSON.mp4")
    ap.add_argument("--tag", default="r16")
    ap.add_argument("--times", default="2.5,7.21,10,12.5")
    ap.add_argument("--shots", default="screenshots")
    ap.add_argument("--cache", default="tools/vlm_verify_cache.json")
    ap.add_argument("--out", default="tools/vlm_consistency.json")
    args = ap.parse_args()

    times = [float(x) for x in args.times.split(",") if x.strip()]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    vlm = VLM()
    cache = load_cache(args.cache)
    rows = []
    for t in times:
        # 文件名与探针一致：探针用 CLI 里的原始数字串（"10"），不是 float 的 "10.0"
        tag_t = ('%g' % t).replace('.', '_')
        shot = os.path.join(args.shots, f"probe_{args.tag}_{tag_t}.png")
        if not os.path.exists(shot):
            print(f"[verify] 缺渲染帧 {shot}，跳过 t={t}")
            continue
        fi = min(len(frames) - 1, int(round(t * fps)))
        vid = fit(frames[fi])
        ren = fit(cv2.imread(shot))
        got = {}
        for side, img in (("video", vid), ("render", ren)):
            key = f"{args.tag}:{t}:{side}"
            if key not in cache:
                try:
                    cache[key] = vlm.ask_json(PROMPT, image=enc(img), max_tokens=220)
                except VLMError as e:
                    print(f"[verify] {key} 失败：{e}")
                    continue
                save_cache(args.cache, cache)
            got[side] = cache[key]
        if len(got) < 2:
            continue
        row = {"t": t, "agree": {}, "video": got["video"], "render": got["render"]}
        for q in QUESTIONS:
            a = str(got["video"].get(q, "")).strip()
            b = str(got["render"].get(q, "")).strip()
            row["agree"][q] = (a == b) if (a and b and a != "unknown" and b != "unknown") else None
        rows.append(row)
        print(f"[verify] t={t}: " + " ".join(f"{q}={'=' if v else ('×' if v is False else '?')}"
                                          for q, v in row["agree"].items()))

    pairs = [v for r in rows for v in r["agree"].values() if v is not None]
    rate = sum(1 for v in pairs if v) / len(pairs) if pairs else None
    per_q = {}
    for q in QUESTIONS:
        vals = [r["agree"][q] for r in rows if r["agree"].get(q) is not None]
        per_q[q] = round(sum(1 for v in vals if v) / len(vals), 3) if vals else None
    out = {"tag": args.tag, "times": times, "overall": None if rate is None else round(rate, 3),
           "perQuestion": per_q, "n": len(pairs), "rows": rows}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[verify] 一致率 {out['overall']}（{len(pairs)} 个可比对）逐题={per_q}")
    print(f"[verify] 写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
