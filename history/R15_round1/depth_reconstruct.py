#!/usr/bin/env python3
"""单目深度重构：2D 关键点 → 视频相机系三维骨架。

原理
----
单目下每根骨的深度差由**透视缩短**唯一确定：

    |Δz| = √(L² − d²)

L = 该骨的解剖长度，d = 该骨在画面上的投影长度。骨在画面上越短，它越是朝向镜头，
Δz 越大。剩下的歧义只有**符号**（骨在视线的前方还是后方），由解剖先验 + 帧间
连续性用 2 态 Viterbi 裁决 —— 逐帧独立判决会让深度闪跳。

坐标系
------
相机系：x = 画面右，y = 画面上，z = 沿视线远离相机（右手系，相机看向 +z）。
原点取髋中点。输出 pose3d[frame][joint] = [x, y, z]，单位米。

输入的 poses 是**图像系**（motion_pipeline.normalize_per_frame 只做减髋中点、
除躯干长、y 翻转，不按偏航旋转），因此 2D 观测可直接当作相机系的切向坐标 u、v。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

DEG = math.pi / 180.0

# ── 解剖比例：骨长占「可见身高」（颅顶→鞋底）的比例 ──────────────────────────
# 源视频人物实测（E15）：腿 50.7% / 躯干 27.6% / 头颈 21.7%；其余取人体测量学常用值。
SEG_FRAC: Dict[str, float] = {
    "hipHalf": 0.055,     # 髋中点 → 股骨头
    "torso":   0.276,     # 髋中点 → 肩中点
    "shHalf":  0.105,     # 肩中点 → 肩关节
    "neck":    0.070,     # 肩中点 → 头中点（耳中点）
    "head":    0.085,     # 头中点 → 鼻尖
    "earHalf": 0.075,     # 头中点 → 耳
    "eye":     0.070,     # 头中点 → 眼
    "upperArm": 0.186,    # 肩关节 → 肘
    "foreArm":  0.146,    # 肘 → 腕
    "thigh":    0.245,    # 股骨头 → 膝
    "shin":     0.223,    # 膝 → 踝
}

# ── 骨架树：父 → 子 ──────────────────────────────────────────────────────────
# kind 决定深度符号的解剖先验（相机系，+z = 远离相机）：
#   lateral   横向骨（髋/肩）—— 视角正对时投影≈全长，Δz≈0，无需先验
#   vertical  竖直骨（躯干/颈）—— 同上
#   armBack   上臂 —— 肘偏后
#   kneeFwd   大腿 —— 膝偏前
#   ankleBack 小腿 —— 踝在膝后
#   faceFwd   鼻/眼 —— 面朝前
#   none      前臂 —— 无强先验，交由帧间连续性裁决
#   yawL/yawR 绕身体轴的侧向骨：深度偏移 = ±L·sin ψ（ψ 已被平滑限速，比 2D 反推干净得多）
#   yawFwd    朝向骨：深度偏移 = −L·cos ψ
#   headL/headR/headFwd 同上，但用**头**的偏航（鼻−耳中点偏移）
SKEL: List[Tuple[str, str, str, str]] = [
    ("hipmid", "left_hip",      "hipHalf",  "yawL"),
    ("hipmid", "right_hip",     "hipHalf",  "yawR"),
    ("hipmid", "shmid",         "torso",    "vertical"),
    ("shmid",  "left_shoulder", "shHalf",   "yawL"),
    ("shmid",  "right_shoulder", "shHalf",  "yawR"),
    ("shmid",  "headmid",       "neck",     "vertical"),
    ("left_shoulder",  "left_elbow",  "upperArm", "armBack"),
    ("left_elbow",     "left_wrist",  "foreArm",  "none"),
    ("right_shoulder", "right_elbow", "upperArm", "armBack"),
    ("right_elbow",    "right_wrist", "foreArm",  "none"),
    ("left_hip",   "left_knee",   "thigh", "kneeFwd"),
    ("left_knee",  "left_ankle",  "shin",  "ankleBack"),
    ("right_hip",  "right_knee",  "thigh", "kneeFwd"),
    ("right_knee", "right_ankle", "shin",  "ankleBack"),
    ("headmid", "nose",      "head",    "headFwd"),
    ("headmid", "left_ear",  "earHalf", "headL"),
    ("headmid", "right_ear", "earHalf", "headR"),
    ("headmid", "left_eye",  "eye",     "headFwd"),
    ("headmid", "right_eye", "eye",     "headFwd"),
]

# yaw 驱动的骨：深度偏移 = L · f(ψ)，符号由 ψ 直接给出，无需 2 歧义裁决
YAW_KINDS = {"yawL": ("body", 1.0), "yawR": ("body", -1.0),
             "headL": ("head", 1.0), "headR": ("head", -1.0),
             "headFwd": ("head", None), "yawFwd": ("body", None)}

# 观测点 → 虚拟节点的构造
DERIVED = {
    "hipmid":  ("left_hip", "right_hip"),
    "shmid":   ("left_shoulder", "right_shoulder"),
    "headmid": ("left_ear", "right_ear"),
}

# 深度符号先验：先验方向 = 该骨在**身体系**里的偏向，经偏航旋转到相机系
PRIOR_DIR: Dict[str, Tuple[float, float, float]] = {
    "armBack":   (0.0, 0.0, -1.0),    # 肘偏后（身体 -Z）
    "kneeFwd":   (0.0, 0.0, 1.0),     # 膝偏前（身体 +Z）
    "ankleBack": (0.0, 0.0, -1.0),    # 踝在膝后
    "faceFwd":   (0.0, 0.0, 1.0),     # 面朝前
}

# Viterbi 权重
W_PRIOR = 0.80        # 切换符号的代价（等效于一次 0.09 m 的深度跳变）
W_TEMP = 1.0 / 0.05 ** 2   # 帧间深度连续性：0.05 m 跳变代价 1.0
W_MAG = 1.0 / 0.15 ** 2    # 深度幅度越界惩罚（超过骨长的 55% 即开始罚）
MEDIAN_K = 7          # |Δz| 的时域中值窗口：透视缩短在 d→L 处病态，噪声被放大成 10cm 级跳变
RATE_LIMIT = 0.11     # 关节深度每帧变化上限（米）：≈2.6 m/s，快舞的真实上限，超出即检测伪影

# 投影长度可信区间（相对解剖长度）：超出即判定为检测错误
TRUST_LO, TRUST_HI = 0.30, 1.02


def _median_run(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """时域中值（去单帧离群、保阶跃）—— 与动作管线的去噪口径一致。"""
    out: List[Optional[float]] = []
    h = MEDIAN_K // 2
    for i in range(len(vals)):
        win = [vals[j] for j in range(i - h, i + h + 1)
               if 0 <= j < len(vals) and vals[j] is not None]
        out.append(sorted(win)[len(win) // 2] if win else None)
    return out


def seg_lengths(height_m: float) -> Dict[str, float]:
    """解剖骨长（米）。"""
    return {k: v * height_m for k, v in SEG_FRAC.items()}


def _median(xs: Sequence[Optional[float]], default: float = 0.0) -> float:
    v = sorted(x for x in xs if x is not None and math.isfinite(x))
    return v[len(v) // 2] if v else default


def _mid(p: Optional[Sequence[float]], a: Optional[Sequence[float]],
         b: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if a and b:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    if a:
        return (a[0], a[1])
    if b:
        return (b[0], b[1])
    return None


def _body_to_cam(psi: float) -> List[List[float]]:
    """身体系 → 相机系 的 3x3 矩阵。

    ψ = 躯干转身角（弧度，正 = 主体向自身左转 = 脸朝画面右）。
    身体系：+X 自身左、+Y 上、+Z 面朝。
    相机系：+X 画面右、+Y 上、+Z 远离相机。
    主体正对相机时（ψ=0）：body_x→cam_x，body_y→cam_y，body_z→cam(−z)。
    """
    c, s = math.cos(psi), math.sin(psi)
    # M = R_y(-ψ) · diag(1,1,-1)
    return [
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, -c],
    ]


def _cam_z_of_body_dir(kind: str, psi: float) -> float:
    """先验方向经偏航后在相机系的 z 分量（>0 表示偏向远离相机）。"""
    d = PRIOR_DIR.get(kind)
    if d is None:
        return 0.0
    m = _body_to_cam(psi)
    return m[2][0] * d[0] + m[2][1] * d[1] + m[2][2] * d[2]


class _Frame:
    """单帧的 2D 观测（相机系切向坐标 u、v，单位=躯干长）。"""

    __slots__ = ("uv", "valid", "s", "z0")

    def __init__(self) -> None:
        self.uv: Dict[str, Tuple[float, float]] = {}
        self.valid: Dict[str, bool] = {}
        self.s = 0.0     # 躯干长的角尺寸（tan 单位）：torsoPx_true / f
        self.z0 = 0.0    # 相机到髋中点的距离（米）


def reconstruct(poses: Sequence[Optional[Sequence[Optional[Sequence[float]]]]],
                keys: Sequence[str],
                vbox: dict,
                height_m: float,
                hfov_deg: float,
                img_w: int,
                img_h: int,
                yaw: Optional[Sequence[float]] = None,
                ) -> dict:
    """返回 {'pose3d', 'conf', 'depth', 'quality'}。

    pose3d[frame][joint] = [x, y, z]（米，相机系，髋中点原点）；不可信时为 null。
    conf[frame][joint] = 0..1（该关节所在骨的可信度）。
    depth[frame] = 相机到髋中点的距离（米）。
    """
    T = len(poses)
    idx = {k: i for i, k in enumerate(keys)}
    seg = seg_lengths(height_m)

    # ── 相机内参（等效）：水平 FOV → 焦距（像素） ────────────────────────────
    f_px = (img_w / 2.0) / math.tan(hfov_deg * DEG / 2.0)
    # videoBox 以缩略空间记录（w/h = 360/640），源帧是其 2 倍
    scale = img_w / max(1.0, float(vbox.get("w") or img_w))

    frames: List[_Frame] = []
    for i in range(T):
        fr = _Frame()
        p = poses[i]
        if p:
            for k, j in idx.items():
                q = p[j] if j < len(p) else None
                if q and q[2] is not None and q[2] >= 0.25:
                    fr.uv[k] = (float(q[0]), float(q[1]))
                    fr.valid[k] = True
                else:
                    fr.valid[k] = False
            fr.uv["hipmid"] = _mid(None, fr.uv.get("left_hip"), fr.uv.get("right_hip"))
            fr.uv["shmid"] = _mid(None, fr.uv.get("left_shoulder"), fr.uv.get("right_shoulder"))
            fr.uv["headmid"] = _mid(None, fr.uv.get("left_ear"), fr.uv.get("right_ear"))
        frac = (vbox.get("frac") or [[0.0, 0.0, 0.0]] * T)[min(i, len(vbox.get("frac") or []) - 1)]
        tp = (vbox.get("torsoPx") or [0.0] * T)[min(i, len(vbox.get("torsoPx") or []) - 1)]
        fr.s = (tp * scale) / f_px                       # 躯干长的角尺寸
        fr.z0 = (height_m * f_px / (frac[0] * img_h)) if frac and frac[0] > 0 else 0.0
        frames.append(fr)

    # ── 头的偏航（鼻−耳中点偏移）：头骨的深度偏移用它，比 2D 反推干净 ─────────
    head_yaw: List[float] = []
    for i in range(T):
        fr = frames[i]
        nose, le, re = fr.uv.get("nose"), fr.uv.get("left_ear"), fr.uv.get("right_ear")
        if nose and le and re:
            half = max(0.08, abs(le[0] - re[0]) / 2.0)
            head_yaw.append(math.asin(max(-1.0, min(1.0, (nose[0] - (le[0] + re[0]) / 2.0) / half))))
        else:
            head_yaw.append(0.0)

    # ── 逐骨求深度：分层处理（父层先于子层） ────────────────────────────────
    children: Dict[str, List[Tuple[str, str, str]]] = {}
    for parent, child, segk, kind in SKEL:
        children.setdefault(parent, []).append((child, segk, kind))
    order: List[str] = []
    seen = set()

    def visit(node: str) -> None:
        if node in seen:
            return
        seen.add(node)
        order.append(node)              # 前序：父层先于子层（深度与切向坐标都沿树自上而下累加）
        for child, _, _ in children.get(node, []):
            visit(child)

    for root in ("hipmid",):
        visit(root)

    # 每骨：逐帧的深度候选与可信度
    depth: Dict[str, List[Optional[float]]] = {"hipmid": [f.z0 for f in frames]}
    trust: Dict[str, List[bool]] = {"hipmid": [True] * T}
    dsign: Dict[str, List[int]] = {"hipmid": [0] * T}
    yaw_dz: Dict[str, List[Optional[float]]] = {}
    for node in order:
        for child, segk, kind in children.get(node, []):
            L = seg[segk]
            # ψ 驱动的骨：深度偏移由偏航精确给出（L·sin ψ / −L·cos ψ），不靠 2D 反推
            if kind in YAW_KINDS:
                src, sgn = YAW_KINDS[kind]
                zs: List[Optional[float]] = []
                oks: List[bool] = []
                dzs: List[Optional[float]] = []
                for i in range(T):
                    pz = depth[node][i]
                    if pz is None:
                        zs.append(None)
                        oks.append(False)
                        dzs.append(None)
                        continue
                    psi = (yaw[i] if (yaw and i < len(yaw) and yaw[i] is not None) else 0.0) \
                        if src == "body" else head_yaw[i]
                    if sgn is None:
                        dz = -L * math.cos(psi)
                    else:
                        dz = sgn * L * math.sin(psi)
                    dzs.append(dz)
                    zs.append(pz + dz)
                    oks.append(True)
                depth[child] = zs
                trust[child] = oks
                yaw_dz[child] = dzs
                continue
            cand_z: List[List[Optional[float]]] = []
            cand_ok: List[bool] = []
            raw_dz: List[Optional[float]] = []
            for i in range(T):
                fr = frames[i]
                pu = fr.uv.get(node)
                cu = fr.uv.get(child)
                if pu is None or cu is None:
                    raw_dz.append(None)
                    cand_z.append([None, None])
                    cand_ok.append(False)
                    continue
                d_uv = math.hypot(cu[0] - pu[0], cu[1] - pu[1])
                d_m = d_uv * fr.s * fr.z0            # 投影长度（米，父深度处）
                ratio = d_m / L if L > 0 else 0.0
                ok = TRUST_LO <= ratio <= TRUST_HI
                dz = math.sqrt(max(0.0, L * L - d_m * d_m))
                raw_dz.append(dz)
                pz = depth[node][i]
                if pz is None:
                    cand_z.append([None, None])
                    cand_ok.append(False)
                    continue
                cand_z.append([pz + dz, pz - dz])
                cand_ok.append(ok)
            # |Δz| 时域中值：透视缩短在 d→L 处病态（dz/dd = -d/dz），噪声被放大成 10cm 级跳变
            dz_s = _median_run(raw_dz)
            for i in range(T):
                if cand_z[i][0] is None or dz_s[i] is None:
                    continue
                pz = depth[node][i]
                cand_z[i] = [pz + dz_s[i], pz - dz_s[i]]
            # 2 态 Viterbi：状态 0 = Δz 取 +，状态 1 = 取 −
            psi = yaw[i] if (yaw and i < len(yaw) and yaw[i] is not None) else 0.0
            pref = _cam_z_of_body_dir(kind, psi)      # >0 偏好远离相机
            emis = [[0.0, 0.0] for _ in range(T)]
            for i in range(T):
                if cand_z[i][0] is None:
                    continue
                for s in (0, 1):
                    z = cand_z[i][s]
                    dz = abs(z - (depth[node][i] or 0.0))
                    e = W_MAG * max(0.0, dz - 0.55 * L) ** 2
                    if abs(pref) > 1e-6:
                        # 状态的符号：0 → Δz>0（远离），1 → Δz<0（靠近）
                        want_far = pref > 0
                        if (s == 0) != want_far:
                            e += W_PRIOR
                    emis[i][s] = e
            dp = [[float("inf"), float("inf")] for _ in range(T)]
            bp = [[-1, -1] for _ in range(T)]
            for i in range(T):
                if cand_z[i][0] is None:
                    dp[i] = [float("inf"), float("inf")]
                    continue
                if i == 0:
                    dp[i] = list(emis[i])
                    continue
                for s in (0, 1):
                    best, arg = float("inf"), 0
                    for sp in (0, 1):
                        if dp[i - 1][sp] == float("inf") or cand_z[i - 1][sp] is None:
                            continue
                        prev = cand_z[i - 1][sp]
                        cur = cand_z[i][s]
                        tr = W_TEMP * (cur - prev) ** 2
                        v = dp[i - 1][sp] + tr + emis[i][s]
                        if v < best:
                            best, arg = v, sp
                    dp[i][s] = best
                    bp[i][s] = arg
            zsel = [None] * T
            oksel = [False] * T
            ssel = [0] * T
            if T:
                s = 0 if dp[T - 1][0] <= dp[T - 1][1] else 1
                for i in range(T - 1, -1, -1):
                    if cand_z[i][s] is None:
                        s = bp[i][s] if bp[i][s] >= 0 else s
                        zsel[i] = None
                        oksel[i] = False
                        continue
                    zsel[i] = cand_z[i][s]
                    oksel[i] = cand_ok[i]
                    ssel[i] = 1 if s == 0 else -1
                    s = bp[i][s] if bp[i][s] >= 0 else s
            depth[child] = zsel
            trust[child] = oksel
            dsign[child] = ssel

    # ── 组装三维坐标 ─────────────────────────────────────────────────────────
    # 沿树累加切向坐标 u、v（相对髋中点），再乘各自深度 → 透视一致的屏幕位置
    uv: Dict[str, List[Optional[Tuple[float, float]]]] = {"hipmid": [(0.0, 0.0)] * T}
    for node in order:
        for child, _, _ in children.get(node, []):
            arr: List[Optional[Tuple[float, float]]] = []
            for i in range(T):
                pu = uv[node][i]
                cu = frames[i].uv.get(child)
                if pu is None or cu is None:
                    arr.append(None)
                    continue
                arr.append((pu[0] + cu[0] - frames[i].uv[node][0],
                            pu[1] + cu[1] - frames[i].uv[node][1]))
            uv[child] = arr

    # 关节深度限速：深度对 2D 噪声敏感，末端（腕/踝）尤甚；超物理速率的变化必是伪影。
    # 限速施加在**每根骨的 Δz** 上（而非累积深度），保证 |Δz| ≤ 骨长不被破坏。
    dz_limited: Dict[str, List[Optional[float]]] = {}
    for parent, child, segk, kind in SKEL:
        if kind in YAW_KINDS:
            dz_limited[child] = yaw_dz.get(child)
            continue
        arr = depth.get(child)
        parr = depth.get(parent)
        if not arr or not parr:
            continue
        dz = [None if (z is None or p is None) else z - p for z, p in zip(arr, parr)]
        prev = None
        for i in range(T):
            v = dz[i]
            if v is None:
                prev = None
                continue
            if prev is not None:
                d = v - prev
                if d > RATE_LIMIT:
                    v = prev + RATE_LIMIT
                elif d < -RATE_LIMIT:
                    v = prev - RATE_LIMIT
                dz[i] = v
            prev = v
        dz_limited[child] = dz
    # 逐层重累积：Δz 一律相对父节点，父的最终深度先定，子孙依次落上去
    for node in order:
        parr = depth.get(node)
        if not parr:
            continue
        for child, _, kind in children.get(node, []):
            dz = dz_limited.get(child)
            arr = depth.get(child)
            if not dz or not arr or not parr:
                continue
            for i in range(T):
                if dz[i] is not None and parr[i] is not None:
                    arr[i] = parr[i] + dz[i]
        arr = depth[k]
        prev = None
        for i in range(T):
            z = arr[i]
            if z is None:
                prev = None
                continue
            if prev is not None:
                d = z - prev
                lim = RATE_LIMIT
                if d > lim:
                    z = prev + lim
                elif d < -lim:
                    z = prev - lim
                arr[i] = z
            prev = z

    pose3d: List[List[Optional[List[float]]]] = []
    conf: List[List[float]] = []
    for i in range(T):
        row: List[Optional[List[float]]] = []
        crow: List[float] = []
        z0 = frames[i].z0
        for k in keys:
            node = k
            z = depth.get(node, [None] * T)[i]
            uvw = uv.get(node, [None] * T)[i]
            if z is None or uvw is None:
                row.append(None)
                crow.append(0.0)
                continue
            # uv 是躯干单位的切向坐标 → 乘 s（躯干角尺寸）化为相机切向坐标，再乘深度得米
            row.append([round(uvw[0] * fr.s * z, 5), round(uvw[1] * fr.s * z, 5),
                        round(z - z0, 5)])
            crow.append(1.0 if trust.get(node, [False] * T)[i] else 0.35)
        pose3d.append(row)
        conf.append(crow)

    # 深度时域中值去离群：腕/踝的 |Δz| 在 d→L 处病态，2D 袖口/裤脚噪声被放大成
    # ±15cm 摆动（离线只做去离群，不做低通 —— 低通带滞后，滞后比噪声更伤一致性）。
    for j in range(len(keys)):
        zser = [row[j][2] if row[j] is not None else None for row in pose3d]
        zs = _median_run(zser)
        for i in range(T):
            if pose3d[i] is not None and pose3d[i][j] is not None and zs[i] is not None:
                pose3d[i][j][2] = round(zs[i], 5)
    # 可信度 5 帧中值：孤立的单帧越界会让查看器在两条路径间跳变（目标跳 0.5m）
    for j in range(len(keys)):
        cser = [row[j] for row in conf]
        cs = _median_run(cser)
        for i in range(T):
            conf[i][j] = 1.0 if (cs[i] is not None and cs[i] >= 0.9) else 0.35

    # ── 质量统计 ─────────────────────────────────────────────────────────────
    flips = 0
    dz_all: List[float] = []
    for node, arr in depth.items():
        if node == "hipmid":
            continue
        prev = None
        for z in arr:
            if z is None:
                prev = None
                continue
            if prev is not None:
                dz_all.append(abs(z - prev))
            prev = z
    for child, dzs in dz_limited.items():
        if not dzs:
            continue
        prev = 0
        for v in dzs:
            if v is None or abs(v) < 1e-6:
                prev = 0
                continue
            s = 1 if v > 0 else -1
            if prev and s != prev:
                flips += 1
            prev = s
    dz_all.sort()
    quality = {
        "frames": T,
        "sign_flips": flips,
        "depth_jump_p95": (dz_all[int(len(dz_all) * 0.95)] if dz_all else 0.0),
        "depth_jump_max": (dz_all[-1] if dz_all else 0.0),
        "focal_px": round(f_px, 2),
    }
    return {
        "pose3d": pose3d,
        "conf": conf,
        "depth": [round(f.z0, 4) for f in frames],
        "quality": quality,
        "frame": "camera(x=right,y=up,z=away); hip-origin; meters",
    }
