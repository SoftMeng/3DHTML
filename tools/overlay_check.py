#!/usr/bin/env python3
"""把 MediaPipe 关键点 + videoBox 的可见身高标定画到源视频帧上，人工核对标定常数。

用法: tools/.venv/bin/python tools/overlay_check.py <视频> <motion.json> [t1 t2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_motion import IDX, KEYS  # noqa: E402


def main() -> int:
    video, mo_path = Path(sys.argv[1]), Path(sys.argv[2])
    times = [float(x) for x in sys.argv[3:]] or [0, 5, 7.21, 10]
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    fps = mo["fps"]
    vb = mo["videoBox"]
    cap = cv2.VideoCapture(str(video))
    want = {int(round(t * fps)): t for t in times}
    out = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            t = want[i]
            fi = min(len(mo["poses"]) - 1, int(round(t * fps)))
            p = mo["poses"][fi]
            if p:
                f = cv2.resize(f, (vb["w"], vb["h"]))
                tp = vb["torsoPx"][fi]
                fr = vb["frac"][fi]
                hxp = fr[1] * vb["w"]
                hyp = (1 - fr[2]) * vb["h"]

                def px(q):
                    return (int(hxp + q[0] * tp), int(hyp - q[1] * tp))

                for k in KEYS:
                    q = p[IDX[k]]
                    if not q:
                        continue
                    x, y = px(q)
                    cv2.circle(f, (x, y), 4, (0, 255, 0), -1)
                    cv2.putText(f, k[:4], (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 0), 1)
                cv2.circle(f, (int(hxp), int(hyp)), 6, (255, 0, 255), -1)
                cv2.putText(f, "hip", (int(hxp) + 8, int(hyp)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 0, 255), 1)

                ears = [p[IDX[k]] for k in ("left_ear", "right_ear") if p[IDX[k]]]
                ey = hyp - (float(np.mean([q[1] for q in ears])) * tp if ears else 0)
                head_top = ey - 0.73 * tp
                ankles = [p[IDX[k]] for k in ("left_ankle", "right_ankle") if p[IDX[k]]]
                ay = hyp - (min(q[1] for q in ankles) * tp if ankles else 0)
                foot_bot = ay + 0.38 * tp
                cv2.line(f, (0, int(head_top)), (f.shape[1], int(head_top)), (0, 0, 255), 1)
                cv2.line(f, (0, int(foot_bot)), (f.shape[1], int(foot_bot)), (255, 0, 0), 1)
                cv2.putText(f, f"t={t} torsoPx={tp:.1f} headTop={head_top:.0f} footBot={foot_bot:.0f} "
                            f"vis={(foot_bot - head_top) / f.shape[0]:.3f}", (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
                out.append(f)
        i += 1
        if len(out) == len(times):
            break
    cap.release()
    if out:
        Path("/tmp/overlay_check.png").write_bytes(cv2.imencode(".png", np.hstack(out))[1].tobytes())
        print("→ /tmp/overlay_check.png", out[0].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
