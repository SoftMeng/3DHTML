#!/usr/bin/env python3
"""左右标注翻转修复。

MediaPipe 的左右归属是**逐帧独立**的假设：人转身、手遮脸、快速甩臂时，
它会把某一段上半身（或下半身）的左右标反 —— 画面上左右互换。
这不是"抖一下"，而是**几何性错误**：肩肘腕接到对面的肩上，
手臂画到身体另一侧，3D 判据（穿模/弯折侧向）随之全部失真。

判据：翻转 ⇔ 该组关键点在画面上左右互换 ⇒ 与时间连续性强烈冲突
（横向跳变幅度 ≈ 该对的间距，而真实运动是连续的）。
用 2 态 Viterbi 在「翻转/不翻转」间取全局最平滑解；
切换有代价（防止把真实快速动作误判成翻转）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Frame = List[Optional[List[float]]]

# 分组：同一组的关键点来自同一次左右归属假设，共享翻转状态
GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("upper", [("left_shoulder", "right_shoulder"),
               ("left_elbow", "right_elbow"),
               ("left_wrist", "right_wrist")]),
    ("lower", [("left_hip", "right_hip"),
               ("left_knee", "right_knee"),
               ("left_ankle", "right_ankle")]),
]

INF = float("inf")


def _group_cost(prev: Frame, cur: Frame, idx: Dict[str, int],
                pairs: Sequence[Tuple[str, str]], sp: int, s: int) -> float:
    """状态 sp→s 的横向运动代价（修复后标注下的时间连续性）。

    状态 0 = 不翻转（左接左、右接右）；状态 1 = 该组左右互换。
    翻转之所以被选中，是因为互换后轨迹变平滑 —— 这正是翻转的签名
    （左手的轨迹接上了右手上一帧的位置），真实运动不会这样。
    """
    total = 0.0
    for a, b in pairs:
        pa, pb = prev[idx[a]], prev[idx[b]]
        ca, cb = cur[idx[a]], cur[idx[b]]
        if not (pa and pb and ca and cb):
            continue
        pl = pa if sp == 0 else pb
        pr = pb if sp == 0 else pa
        cl = ca if s == 0 else cb
        cr = cb if s == 0 else ca
        total += abs(cl[0] - pl[0]) + abs(cr[0] - pr[0])
    return total


def repair_lateral_flip(poses: Sequence[Frame], keys: Sequence[str],
                        groups: Sequence[Tuple[str, List[Tuple[str, str]]]] = GROUPS
                        ) -> Tuple[List[Frame], Dict[str, dict]]:
    """返回 (修复后的 poses, 每组的统计)。"""
    idx = {k: i for i, k in enumerate(keys)}
    out: List[Frame] = [[list(j) if j else None for j in p] if p else None for p in poses]
    stats: Dict[str, dict] = {}
    n = len(poses)
    for name, pairs in groups:
        if any(a not in idx or b not in idx for a, b in pairs):
            continue
        seps: List[float] = []
        for p in poses:
            if not p:
                continue
            for a, b in pairs:
                pa, pb = p[idx[a]], p[idx[b]]
                if pa and pb:
                    seps.append(abs(pa[0] - pb[0]))
        seps.sort()
        ref = seps[len(seps) // 2] if seps else 0.1
        lam = 0.25 * ref                       # 翻转代价/帧：短 glitch 划算，长段落不划算
        dp = [0.0, lam]                      # 首帧：翻转需付费
        back: List[List[int]] = [[0, 0]]
        for i in range(1, n):
            ndp = [INF, INF]
            nb = [0, 0]
            prev, cur = poses[i - 1], poses[i]
            for s in (0, 1):
                if not prev or not cur:
                    continue
                for sp in (0, 1):
                    c = dp[sp] + _group_cost(prev, cur, idx, pairs, sp, s) + (lam if s else 0.0)
                    if c < ndp[s]:
                        ndp[s] = c
                        nb[s] = sp
            dp = ndp
            back.append(nb)
        # 回溯状态序列
        seq = [0] * n
        st = 0
        for i in range(n - 1, 0, -1):
            seq[i] = st
            st = back[i][st]
        flipped = [i for i, v in enumerate(seq) if v]
        for i in flipped:
            p = out[i]
            if not p:
                continue
            for a, b in pairs:
                ja, jb = p[idx[a]], p[idx[b]]
                if ja and jb:
                    p[idx[a]], p[idx[b]] = jb, ja
        stats[name] = {"frames_flipped": len(flipped), "frames": flipped, "lambda": round(lam, 4)}
    return out, stats
