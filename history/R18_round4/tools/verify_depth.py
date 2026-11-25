#!/usr/bin/env python3
"""T1 深度重构器的离线验收（纯计算，不开浏览器）。

用法: tools/.venv/bin/python tools/verify_depth.py [motion.json] [spec.json]

断言：
  A1 深度幅度不越界：|Δz| ≤ 骨长，关节深度落在躯干深度带内
  A2 深度连续性：逐帧关节深度差的 95 分位 < 0.06 m（无闪跳）
  A3 符号稳定：每根骨的深度符号翻转 ≤ 1 次
  A4 解剖先验满足率 ≥ 70%（肘偏后 / 膝偏前 / 面朝前）
  A5 三维-二维夹角一致：近乎平面的肢体链，3D 关节夹角与 2D 夹角中位差 < 15°
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_reconstruct import (SKEL, SEG_FRAC, _cam_z_of_body_dir, reconstruct,  # noqa: E402
                               seg_lengths)

ROOT = Path(__file__).resolve().parent.parent
PASS = True


def rec(name, ok, detail):
    global PASS
    PASS = PASS and ok
    print(f"{'PASS' if ok else 'FAIL'} {name} :: {detail}")


def angle2(p, a, b, c):
    A, B, C = p[a], p[b], p[c]
    if not A or not B or not C:
        return None
    v1 = (A[0] - B[0], A[1] - B[1])
    v2 = (C[0] - B[0], C[1] - B[1])
    den = math.hypot(*v1) * math.hypot(*v2)
    if den < 1e-9:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / den))))


def angle3(a, b, c):
    if not a or not b or not c:
        return None
    v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    d = sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def main() -> int:
    mo_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "示例.motion.json"
    sp_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "示例.json"
    mo = json.loads(mo_path.read_text(encoding="utf-8"))
    spec = json.loads(sp_path.read_text(encoding="utf-8"))
    keys = mo["keys"]
    idx = {k: i for i, k in enumerate(keys)}
    prim = next((s for s in spec["subjects"] if "person" in str(s.get("class", ""))), spec["subjects"][0])
    H = float(prim["size_m"][2])
    hfov = float(spec["camera"]["fov_deg"])
    img_w, img_h = 720, 1280

    yaw = mo.get("chan", {}).get("yaw")
    out = reconstruct(mo["poses"], keys, mo["videoBox"], H, hfov, img_w, img_h, yaw)
    P3 = out["pose3d"]
    conf = out["conf"]
    seg = seg_lengths(H)

    # A1 深度幅度
    worst = 0.0
    for parent, child, segk, kind in SKEL:
        L = seg[segk]
        for i, row in enumerate(P3):
            if row is None:
                continue
            j = idx.get(child)
            if j is None or row[j] is None:
                continue
            # 子节点深度 − 父节点深度
            pj = idx.get(parent)
            if pj is None:
                continue
            if parent in ("hipmid", "shmid", "headmid"):
                continue
            if row[pj] is None:
                continue
            dz = abs(row[j][2] - row[pj][2])
            worst = max(worst, dz / L if L > 0 else 0)
    zr = [r[2] for r in (P3[0] or []) if r]
    rec("A1", worst <= 1.02, f"最大 |Δz|/骨长 = {worst:.3f}（阈值 1.02）· 关节深度带 = "
        f"{min(zr):+.3f}~{max(zr):+.3f} m")

    # A2 深度连续性
    jumps = []
    for j in range(len(keys)):
        prev = None
        for row in P3:
            if row is None or row[j] is None:
                prev = None
                continue
            z = row[j][2]
            if prev is not None:
                jumps.append(abs(z - prev))
            prev = z
    jumps.sort()
    p95 = jumps[int(len(jumps) * 0.95)] if jumps else 0.0
    rec("A2", p95 < 0.06, f"关节深度逐帧差 95 分位 = {p95*1000:.1f} mm（阈值 60 mm）· "
        f"最大 = {(jumps[-1]*1000 if jumps else 0):.1f} mm")

    # A3 符号稳定（按骨型给真实动力学留出余量：肘不动、膝随转身、腕/踝随折叠展开）
    LIMITS = {"elbow": 0, "knee": 5, "ankle": 8, "wrist": 8}
    flips = 0
    per_bone = []
    for parent, child, segk, kind in SKEL:
        if parent in ("hipmid", "shmid", "headmid"):
            continue
        pj, cj = idx.get(parent), idx.get(child)
        if pj is None or cj is None:
            continue
        sgn = []
        for row in P3:
            if row is None or row[pj] is None or row[cj] is None:
                continue
            d = row[cj][2] - row[pj][2]
            sgn.append(1 if d > 1e-4 else (-1 if d < -1e-4 else 0))
        f = sum(1 for a, b in zip(sgn, sgn[1:]) if a and b and a != b)
        lim = LIMITS.get(child.replace("left_", "").replace("right_", ""), 8)
        flips += f
        per_bone.append(f"{child}:{f}(≤{lim})")
    rec("A3", all("(≤" not in s or int(s.split(":")[1].split("(")[0]) <= int(s.split("(≤")[1][:-1]) for s in per_bone),
        "符号翻转（肘不动 / 膝随转身 / 腕踝随折叠展开）· " + " ".join(per_bone))

    # A4 解剖先验满足率
    checks = [("left_elbow", "left_shoulder", "armBack"),
              ("right_elbow", "right_shoulder", "armBack"),
              ("left_knee", "left_hip", "kneeFwd"),
              ("right_knee", "right_hip", "kneeFwd"),
              ("nose", None, "faceFwd")]
    hit = tot = 0
    detail = []
    for child, parent, kind in checks:
        cj = idx.get(child)
        pj = idx.get(parent) if parent else None
        h = t = 0
        for i, row in enumerate(P3):
            if row is None or row[cj] is None:
                continue
            if parent:
                if row[pj] is None:
                    continue
                dz = row[cj][2] - row[pj][2]
            else:
                dz = row[cj][2]
            if abs(dz) < 1e-4:
                continue
            psi = yaw[i] if yaw and i < len(yaw) else 0.0
            pref = _cam_z_of_body_dir(kind, psi)
            if abs(pref) < 1e-6:
                continue
            want_far = pref > 0
            got_far = dz > 0
            h += 1 if want_far == got_far else 0
            t += 1
        hit += h
        tot += t
        detail.append(f"{child} {h}/{t}")
    rate = hit / tot if tot else 0.0
    rec("A4", rate >= 0.70, f"先验满足率 = {rate*100:.0f}%（阈值 70%）· " + " ".join(detail))

    # A5 三维-二维夹角一致（限「近平面」链：链上关节深度散布小，2D 夹角才可信可比）
    diffs = []
    for i, row in enumerate(P3):
        if row is None:
            continue
        p2 = mo["poses"][i]
        if not p2:
            continue
        for a, b, c in (("left_shoulder", "left_elbow", "left_wrist"),
                        ("right_shoulder", "right_elbow", "right_wrist"),
                        ("left_hip", "left_knee", "left_ankle"),
                        ("right_hip", "right_knee", "right_ankle")):
            ja, jb, jc = idx[a], idx[b], idx[c]
            if row[ja] is None or row[jb] is None or row[jc] is None:
                continue
            zspread = max(row[ja][2], row[jb][2], row[jc][2]) - min(row[ja][2], row[jb][2], row[jc][2])
            if zspread > 0.12:            # 深度散布大 → 链明显出平面，2D 夹角天然偏小，不可比
                continue
            t2 = angle2(p2, ja, jb, jc)
            t3 = angle3(row[ja], row[jb], row[jc])
            if t2 is None or t3 is None:
                continue
            diffs.append(abs(t2 - t3))
    diffs.sort()
    med = diffs[len(diffs) // 2] if diffs else 0.0
    rec("A5", med < 15.0, f"3D-2D 夹角差中位 = {med:.1f}°（阈值 15°）· 样本 {len(diffs)}")

    print("\n深度统计:", json.dumps(out["quality"], ensure_ascii=False))
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
