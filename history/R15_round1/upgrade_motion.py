#!/usr/bin/env python3
"""把 v2 动作文件原地升级为 v3：附加透视缩短深度重构的 pose3d。

用法: tools/.venv/bin/python tools/upgrade_motion.py <motion.json> [--spec 示例.json] [--video <源视频>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_reconstruct as DR

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", type=Path)
    ap.add_argument("--spec", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None)
    args = ap.parse_args()
    mo = json.loads(args.motion.read_text(encoding="utf-8"))
    if mo.get("version", 0) >= 3 and mo.get("pose3d"):
        print("已是 v3，跳过")
        return 0
    spec_path = args.spec or ROOT / (args.motion.stem.replace(".motion", "") + ".json")
    if not spec_path.exists() and (ROOT / "示例.json").exists():
        spec_path = ROOT / "示例.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None
    video = args.video
    if video is None:
        for cand in ROOT.glob(mo.get("video", "*")):
            video = cand
            break
    if video is None or not Path(video).exists():
        print("找不到源视频，无法取原生分辨率（可用 --video 指定）", file=sys.stderr)
        return 1
    prim = None
    if spec:
        prim = next((s for s in spec.get("subjects", []) if "person" in str(s.get("class", ""))), None)
    cap = cv2.VideoCapture(str(video))
    iw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ih = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    height_m = float(prim["size_m"][2]) if prim and prim.get("size_m") else 1.6
    hfov = float(spec["camera"]["fov_deg"]) if spec else 50.0
    rec = DR.reconstruct(mo["poses"], mo["keys"], mo["videoBox"], height_m, hfov, iw, ih,
                         mo.get("chan", {}).get("yaw"))
    mo["version"] = 3
    mo["pose3d"] = rec["pose3d"]
    mo["pose3dConf"] = rec["conf"]
    mo["pose3dMeta"] = {"frame": rec["frame"], "depth": rec["depth"], "quality": rec["quality"],
                        "height_m": height_m, "hfov_deg": hfov, "img_w": iw, "img_h": ih}
    mo.setdefault("frame", "")
    mo["frame"] += ("; pose3d: 米/髋中点原点/相机系(x=画面右,y=上,z=远离相机), "
                    "由透视缩短|Δz|=√(L²−d²)重构")
    args.motion.write_text(json.dumps(mo, ensure_ascii=False, separators=(",", ":")),
                           encoding="utf-8")
    print(f"✓ {args.motion} → v3（{mo['frames']} 帧 · 深度跳变 p95 "
          f"{rec['quality']['depth_jump_p95']*1000:.0f}mm）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
