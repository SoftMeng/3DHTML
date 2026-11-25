"""姿态动作管线（纯计算，无 I/O）：置信度修复 → 逐帧归一化 → 去噪 → 语义通道。

前提：单目逐帧检测给出的是**带噪测量**，不是动作。直接播放 = 把噪声当动作。
- 修复：遮挡帧（交叉手臂、低置信腕/踝）用**链空间重建**补回 —— 保留上段朝向，
  用邻帧的弯折角还原末端位置；比线性插值更接近真���（线性插值会把弯臂拉直）。
- 归一化：单位取**逐帧躯干长**（肩中点→髋中点）。躯干长不随转身缩短，
  但随景深变化 → 逐帧归一化后姿态与景深无关（人走近镜头不该看起来在蹲）。
- 去噪：在归一化域做（高斯去离群 + One-Euro 低通）。
- 通道：所有「从 2D 猜 3D」的歧义（转身、侧倾、重心、接触）在离线消解，
  以显式语义写回动作文件 v2 —— 查看器退化为纯重定向 + 物理层。

与 index.html 的运行时推导同源：v1 动作文件走运行时推导（兼容路径），
v2 直接消费本模块产出的通道。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

CONF_MIN = 0.25          # 低于此置信度的关键点视为不可用
DEG = math.pi / 180.0

Point = Optional[Sequence[float]]        # [x, y, score]
Frame = Optional[List[Point]]            # 17 点


# ---------- 滤波基元 ----------
def gauss_smooth(vals: Sequence[Optional[float]], w: int = 3) -> List[Optional[float]]:
    """高斯平滑（保序去离群）：窗口内加权均值，边缘按可用样本归一。"""
    h = max(1.0, w / 2.0)
    k = [math.exp(-(i * i) / (2 * h * h)) for i in range(-w, w + 1)]
    out: List[Optional[float]] = []
    for i in range(len(vals)):
        a = c = 0.0
        for j in range(-w, w + 1):
            idx = i + j
            if idx < 0 or idx >= len(vals) or vals[idx] is None:
                continue
            a += vals[idx] * k[j + w]
            c += k[j + w]
        out.append(a / c if c else None)
    return out


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One-Euro 自适应滤波（与 index.html 的实现逐位对齐）。"""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0):
        self.mc = min_cutoff
        self.beta = beta
        self.dc = d_cutoff
        self.x: Optional[float] = None
        self.dx = 0.0

    def filter(self, v: float, dt: float) -> Optional[float]:
        if not math.isfinite(v):
            return self.x
        if self.x is None or not dt > 0:
            self.x = v
            return v
        dv = (v - self.x) / dt
        ad = _alpha(self.dc, dt)
        self.dx = ad * dv + (1 - ad) * self.dx
        a = _alpha(self.mc + self.beta * abs(self.dx), dt)
        self.x = a * v + (1 - a) * self.x
        return self.x


# ---------- 修复 ----------
def _ok(pose: Frame, j: int) -> bool:
    return pose is not None and pose[j] is not None and pose[j][2] >= CONF_MIN


def _bad_runs(poses: Sequence[Frame], j: int) -> List[tuple]:
    runs, start = [], None
    for i, p in enumerate(poses):
        if not _ok(p, j):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(poses) - 1))
    return runs


def _lerp_pt(a: Point, b: Point, t: float, score: float) -> List[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, score]


def _unit(v: Sequence[float]) -> Optional[tuple]:
    l = math.hypot(v[0], v[1])
    return (v[0] / l, v[1] / l) if l > 1e-9 else None


def _chain_geom(pose: Frame, j_root: int, j_mid: int, j_end: int):
    """末端链几何：上段方向 u、弯折角 θ（u→下段）、下段长 L。"""
    R, M, E = pose[j_root], pose[j_mid], pose[j_end]
    if R is None or M is None or E is None:
        return None
    u = _unit((M[0] - R[0], M[1] - R[1]))
    if u is None:
        return None
    v = (E[0] - M[0], E[1] - M[1])
    L = math.hypot(v[0], v[1])
    if L < 1e-6:
        return None
    vn = (v[0] / L, v[1] / L)
    cross = u[0] * vn[1] - u[1] * vn[0]
    dot = u[0] * vn[0] + u[1] * vn[1]
    return u, math.atan2(cross, dot), L


def repair_end_joint(poses: List[Frame], j_end: int, j_mid: int, j_root: int) -> None:
    """末端关节（腕/踝）的置信度修复：链空间重建，失败退化为线性插值。

    链空间重建 = 用上段朝向 u（当前帧，可靠）+ 邻帧的弯折角 θ 与下段长 L
    还原末端位置：E = M + L·R(θ)·u。手臂被遮挡时弯肘依然成立，
    而线性插值会把弯臂拉直。
    """
    n = len(poses)
    for a, b in _bad_runs(poses, j_end):
        pa = a - 1
        while pa >= 0 and not _ok(poses[pa], j_end):
            pa -= 1
        nb = b + 1
        while nb < n and not _ok(poses[nb], j_end):
            nb += 1
        if pa < 0 and nb >= n:
            continue
        if pa < 0 or nb >= n:                      # 只有一侧可靠 → 保持该侧
            src = nb if pa < 0 else pa
            for f in range(a, b + 1):
                poses[f][j_end] = list(poses[src][j_end])
            continue
        ga = _chain_geom(poses[pa], j_root, j_mid, j_end)
        gb = _chain_geom(poses[nb], j_root, j_mid, j_end)
        score = min(poses[pa][j_end][2], poses[nb][j_end][2])
        span = max(1, nb - pa)
        if ga and gb:
            th_a, th_b = ga[1], gb[1]
            delta = (th_b - th_a + math.pi) % (2 * math.pi) - math.pi   # 最短角路径
            La, Lb = ga[2], gb[2]
            for f in range(a, b + 1):
                t = (f - pa) / span
                M = poses[f][j_mid]
                R = poses[f][j_root]
                u = _unit((M[0] - R[0], M[1] - R[1])) if (M and R) else None
                if u is None:
                    poses[f][j_end] = _lerp_pt(poses[pa][j_end], poses[nb][j_end], t, score)
                    continue
                th = th_a + delta * t
                L = La + (Lb - La) * t
                ct, st = math.cos(th), math.sin(th)
                poses[f][j_end] = [M[0] + L * (u[0] * ct - u[1] * st),
                                   M[1] + L * (u[0] * st + u[1] * ct), score]
        else:
            for f in range(a, b + 1):
                t = (f - pa) / span
                poses[f][j_end] = _lerp_pt(poses[pa][j_end], poses[nb][j_end], t, score)


def repair_joint(poses: List[Frame], j: int) -> None:
    """一般关节的置信度修复：低置信段用邻帧线性插值（端点保持）。"""
    n = len(poses)
    for a, b in _bad_runs(poses, j):
        pa = a - 1
        while pa >= 0 and not _ok(poses[pa], j):
            pa -= 1
        nb = b + 1
        while nb < n and not _ok(poses[nb], j):
            nb += 1
        if pa < 0 and nb >= n:
            continue
        if pa < 0:
            for f in range(a, b + 1):
                poses[f][j] = list(poses[nb][j])
        elif nb >= n:
            for f in range(a, b + 1):
                poses[f][j] = list(poses[pa][j])
        else:
            score = min(poses[pa][j][2], poses[nb][j][2])
            span = max(1, nb - pa)
            for f in range(a, b + 1):
                poses[f][j] = _lerp_pt(poses[pa][j], poses[nb][j], (f - pa) / span, score)


# ---------- 归一化 / 去噪 ----------
def torso_length(pose: Frame, i_lsh: int, i_rsh: int, i_lhp: int, i_rhp: int) -> Optional[float]:
    """躯干长 = 肩中点→髋中点（像素）。不随转身缩短 → 可作尺度基准。"""
    if pose is None:
        return None
    ls, rs, lh, rh = pose[i_lsh], pose[i_rsh], pose[i_lhp], pose[i_rhp]
    if ls is None or rs is None or lh is None or rh is None:
        return None
    sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
    return math.hypot(sx - hx, sy - hy)


def _median5(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """5 帧中值：把逐帧尺度的抖动压平（尺度变化本身是缓慢的）。"""
    out = []
    for i in range(len(vals)):
        win = [v for v in vals[max(0, i - 2):i + 3] if v is not None and v > 1e-6]
        out.append(sorted(win)[len(win) // 2] if win else None)
    return out


def normalize_per_frame(poses: Sequence[Frame], idx: dict, flip_y: bool = True) -> List[Frame]:
    """图像坐标 → 髋中点原点 / 逐帧躯干长单位 / x+=主体左。

    逐帧单位消除景深变化（人走近镜头 → 表观变大 → 归一化后姿态不变）。
    """
    lens = [torso_length(p, idx["left_shoulder"], idx["right_shoulder"],
                         idx["left_hip"], idx["right_hip"]) for p in poses]
    smooth = _median5(lens)
    valid = sorted(v for v in smooth if v)
    fallback = valid[len(valid) // 2] if valid else 1.0
    out: List[Frame] = []
    for i, p in enumerate(poses):
        unit = smooth[i] or fallback
        if p is None:
            out.append(None)
            continue
        lh, rh = p[idx["left_hip"]], p[idx["right_hip"]]
        if lh is None or rh is None:
            out.append(None)
            continue
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        frame = []
        for j in range(len(p)):
            q = p[j]
            if q is None:
                frame.append(None)
                continue
            x = (q[0] - hx) / unit
            y = ((hy - q[1]) / unit) if flip_y else ((q[1] - hy) / unit)
            frame.append([round(x, 4), round(y, 4), round(q[2], 3)])
        out.append(frame)
    return out


def median3(vals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """3 帧中值：去单帧离群但**保留阶跃**。

    高斯（±3 帧）会把「一帧之内的快速动作」抹平 7 帧 —— 24fps 的舞蹈里
    手臂从体侧甩到锁骨只要 1 帧，高斯后整整 7 帧都是过渡段，人偶明显落后。
    中值滤波保阶跃，去离群同样有效。
    """
    out: List[Optional[float]] = []
    n = len(vals)
    for i in range(n):
        win = [vals[k] for k in (i - 1, i, i + 1) if 0 <= k < n and vals[k] is not None]
        out.append(sorted(win)[len(win) // 2] if win else None)
    return out


def denoise_poses(poses: Sequence[Frame], fps: float = 24.0) -> List[Frame]:
    """逐关节去离群：3 帧中值（保阶跃）。

    这里**不做低通**：低通（One-Euro）必然带来滞后，24fps 的舞蹈里一帧的
    快速动作会被抹掉半个身位，人偶明显落后于源视频 —— 滞后比噪声更伤一致性。
    检测噪声交给查看器运行时按目标平滑（那里的低通是逐目标、随运动自适应）。
    """
    out: List[Frame] = [[list(j) if j else None for j in p] if p else None for p in poses]
    n = len(poses)
    jn = len(poses[0]) if poses and poses[0] else 0
    for j in range(jn):
        xs = [(p[j][0] if (p and p[j]) else None) for p in poses]
        ys = [(p[j][1] if (p and p[j]) else None) for p in poses]
        gx = median3(xs)
        gy = median3(ys)
        for i in range(n):
            if not (poses[i] and poses[i][j]):
                continue
            out[i][j][0] = gx[i]
            out[i][j][1] = gy[i]
    return out


# ---------- 语义通道 ----------
def median_filter(vals: Sequence[float], k: int = 5) -> List[float]:
    """窗口中值：去掉孤立尖峰（保留持续的真实变化）。"""
    h = k // 2
    out = []
    for i in range(len(vals)):
        win = sorted(vals[max(0, i - h):i + h + 1])
        out.append(win[len(win) // 2])
    return out


def median_filter2(rows: Sequence[Sequence[float]], k: int = 5) -> List[List[float]]:
    """逐列中值滤波：去掉检测失败导致的取景占比离群帧。"""
    if not rows: return []
    cols = len(rows[0])
    out = []
    for i in range(len(rows)):
        line = []
        for c in range(cols):
            h = k // 2
            win = sorted(rows[j][c] for j in range(max(0, i - h), min(len(rows), i + h + 1)))
            line.append(win[len(win) // 2])
        out.append(line)
    return out


def rate_limit(vals: Sequence[float], max_rate_deg: float, fps: float) -> List[float]:
    """速率上限：人体关节角速度有上限，单帧突变必是检测伪影（转身/侧倾通道用）。"""
    lim = max_rate_deg * DEG / max(1.0, fps)
    out, last = [], None
    for v in vals:
        if last is None:
            out.append(v); last = v; continue
        d = v - last
        d = lim if d > lim else (-lim if d < -lim else d)
        last = last + d
        out.append(last)
    return out


def derive_yaw(poses: Sequence[Frame], idx: dict) -> List[float]:
    """躯干转身角（度）：幅度=肩宽透视收缩，方向=鼻−耳中点偏移。

    参考肩宽自校准：取「头正对镜头」帧的肩宽中位。
    """
    T = len(poses)
    sh_w: List[Optional[float]] = [None] * T
    head_yaw: List[Optional[float]] = [None] * T
    for i, p in enumerate(poses):
        if p is None:
            continue
        ls, rs = p[idx["left_shoulder"]], p[idx["right_shoulder"]]
        if ls and rs:
            sh_w[i] = abs(ls[0] - rs[0])
        n, le, re = p[idx["nose"]], p[idx["left_ear"]], p[idx["right_ear"]]
        if n and le and re:
            half = max(0.08, abs(le[0] - re[0]) / 2)
            head_yaw[i] = math.asin(max(-1.0, min(1.0, (n[0] - (le[0] + re[0]) / 2) / half)))
    calm = sorted(sh_w[i] for i in range(T)
                  if head_yaw[i] is not None and abs(head_yaw[i]) < 0.20 and sh_w[i] is not None)
    if calm:
        ref = calm[len(calm) // 2]
    else:
        rest = sorted(v for v in sh_w if v is not None)
        ref = rest[len(rest) // 2] if rest else 1.0
    yaw: List[float] = []
    for i in range(T):
        if sh_w[i] is None or head_yaw[i] is None:
            yaw.append(0.0)
            continue
        sgn = (1.0 if head_yaw[i] > 0 else -1.0) if abs(head_yaw[i]) > 0.14 else 0.0
        yaw.append(sgn * math.acos(max(0.0, min(1.0, sh_w[i] / ref))))
    # 5 帧中值：孤立的 1-2 帧「假转身」（肩宽被检测压缩）是伪影，必须先去掉，
    # 否则死区会把 0 突变成 30°，转身通道带着整条身体横向弹一下。
    yaw = median_filter(yaw, 5)
    f = OneEuro(2.0, 0.04)
    sm, last = [], None
    for i, v in enumerate(yaw):
        d = 0 if last is None else (i - last)
        last = i
        sm.append(f.filter(v, d))
    sm = [0.0 if abs(v) < 30 * DEG else v for v in sm]     # 死区在前
    # 速率上限在后：真实转身仍能跟得上
    return rate_limit(sm, 250.0, 24.0)


def derive_roll(poses: Sequence[Frame], idx: dict, limit: float = 0.6) -> List[float]:
    """躯干侧倾（弧度）：躯干轴（髋中点→肩中点）相对画面垂直的倾角。

    比「肩线倾角」更干净：肩线倾角会随转身角 cos ψ 衰减，躯干轴不会。
    中位数扣掉相机滚转偏置（手持素材的全局倾斜）。
    正值 = 躯干上端向主体左倾。
    """
    T = len(poses)
    tilt: List[Optional[float]] = [None] * T
    for i, p in enumerate(poses):
        if p is None:
            continue
        ls, rs = p[idx["left_shoulder"]], p[idx["right_shoulder"]]
        lh, rh = p[idx["left_hip"]], p[idx["right_hip"]]
        if not (ls and rs and lh and rh):
            continue
        dx = (ls[0] + rs[0]) / 2 - (lh[0] + rh[0]) / 2
        dy = (ls[1] + rs[1]) / 2 - (lh[1] + rh[1]) / 2
        if dy < 1e-6:
            continue
        tilt[i] = math.atan2(dx, dy)
    valid = sorted(v for v in tilt if v is not None)
    bias = valid[len(valid) // 2] if valid else 0.0
    out = []
    for v in tilt:
        if v is None:
            out.append(0.0)
        else:
            out.append(max(-limit, min(limit, v - bias)))
    return rate_limit(median_filter(out, 5), 90.0, 24.0)


def derive_root(poses: Sequence[Frame], idx: dict) -> List[List[float]]:
    """重心通道：[横移, 髋高]（单位=躯干长）。

    横移 = 髋中点 x − 双脚中心 x（髋中点即原点 → 即 −双脚中心 x）；
    髋高 = 髋中点到双脚中心的垂直距离（1.0 ≈ 站直，0.8 ≈ 蹲）。
    """
    out = []
    for p in poses:
        if p is None:
            out.append([0.0, 0.0])
            continue
        la, ra = p[idx["left_ankle"]], p[idx["right_ankle"]]
        if not (la and ra):
            out.append([0.0, 0.0])
            continue
        bx = (la[0] + ra[0]) / 2
        by = (la[1] + ra[1]) / 2
        out.append([round(-bx, 4), round(-by, 4)])
    return out


def derive_contact(poses: Sequence[Frame], idx: dict, vel_th: float = 0.025) -> List[List[bool]]:
    """接触门控：更低的脚=支撑脚，其水平速度低于阈值且连续 2 帧 → 踩实。"""
    contact = []
    for i, p in enumerate(poses):
        c = [False, False]
        if p is not None:
            L, R = p[idx["left_ankle"]], p[idx["right_ankle"]]
            pL = poses[i - 1][idx["left_ankle"]] if i > 0 and poses[i - 1] else None
            pR = poses[i - 1][idx["right_ankle"]] if i > 0 and poses[i - 1] else None
            if L and R:
                lower = 0 if L[1] <= R[1] else 1
                cur = L if lower == 0 else R
                prev = pL if lower == 0 else pR
                v = abs(cur[0] - prev[0]) if prev else 1.0
                c[lower] = v < vel_th
        contact.append(c)
    raw = [list(c) for c in contact]
    for i in range(len(contact)):
        for k in (0, 1):
            if raw[i][k] and not (i > 0 and raw[i - 1][k]):
                contact[i][k] = False
    return contact


def body_stats(poses: Sequence[Frame], idx: dict) -> dict:
    """骨段长度中位统计（归一化域）：查看器按此重建骨架比例。"""
    good = [p for p in poses if p]

    def mean_xy(j):
        vals = [p[j] for p in good if p[j]]
        if not vals:
            return [0.0, 0.0]
        return [round(statistics_median([v[0] for v in vals]), 4),
                round(statistics_median([v[1] for v in vals]), 4)]

    def seg(a, b):
        vals = []
        for p in good:
            if p[a] and p[b]:
                vals.append(math.hypot(p[b][0] - p[a][0], p[b][1] - p[a][1]))
        return round(statistics_median(vals), 4) if vals else 0.0

    return {
        "shoulderL": mean_xy(idx["left_shoulder"]), "shoulderR": mean_xy(idx["right_shoulder"]),
        "hipL": mean_xy(idx["left_hip"]), "hipR": mean_xy(idx["right_hip"]),
        "shoulderHipUnit": 1.0,
        "armUpperL": seg(idx["left_shoulder"], idx["left_elbow"]),
        "armForeL": seg(idx["left_elbow"], idx["left_wrist"]),
        "armUpperR": seg(idx["right_shoulder"], idx["right_elbow"]),
        "armForeR": seg(idx["right_elbow"], idx["right_wrist"]),
        "legThighL": seg(idx["left_hip"], idx["left_knee"]),
        "legShinL": seg(idx["left_knee"], idx["left_ankle"]),
        "legThighR": seg(idx["right_hip"], idx["right_knee"]),
        "legShinR": seg(idx["right_knee"], idx["right_ankle"]),
    }


def statistics_median(vals: Sequence[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


def frame_conf(poses: Sequence[Frame], idx: dict, core: Sequence[int]) -> List[float]:
    """逐帧整体置信度（关键躯干点的均值）。"""
    out = []
    for p in poses:
        if p is None:
            out.append(0.0)
            continue
        vals = [p[j][2] for j in core if p[j]]
        out.append(round(sum(vals) / len(vals), 3) if vals else 0.0)
    return out
