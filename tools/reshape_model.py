"""按源视频人物的比例重塑 Mixamo 模型（离线，零依赖除 numpy）。

用法:
    python3 tools/reshape_model.py [src.glb] [dst.glb] [--leg 0.507 --torso 0.276 --head 0.217]

为什么：Mixamo 的女模是「时装插画」比例（腿占可见身高 58%、躯干 21%），
源视频里的表演者是常规人体比例（腿 51%、躯干 27.6%）。比例不符会让
「对照机位」的相机反解无法同时满足「可见身高占比」与「髋中点屏幕位置」
两个约束 —— 这正是 E13 剪影残差的来源。

怎么改：
  1. 骨偏移：腿链（膝/踝）按 legF 缩短，脊柱链按需加长 —— 骨长即 IK 的两连杆长度。
  2. 网格顶点：腿区按权重做 Y 向压缩（绕髋），鞋做刚体平移 —— 保持截面宽度不变。
  3. inverseBindMatrices 全部重算（骨骼静止姿势变了，旧矩阵不再成立）。

坐标系（已实测核验）：网格顶点坐标就是模型静止世界系（米，Y 向上），
骨骼节点平移在各自骨局部系里、长度单位是 Mixamo 的 100 单位（= 1cm），
inverseBindMatrices = inverse(骨世界矩阵)（乘积实测为单位阵）。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

import numpy as np


def q_mul(a, b):
    """四元数相乘（three.js 顺序：x,y,z,w）。"""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def q_axis_x(deg):
    h = math.radians(deg) / 2
    return [math.sin(h), 0.0, 0.0, math.cos(h)]

# 骨分组：腿（压缩）/ 鞋（刚体平移）/ 其余（不变）
LEG_BONES = ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg",
             "mixamorig:RightUpLeg", "mixamorig:RightLeg")
FOOT_BONES = ("mixamorig:LeftFoot", "mixamorig:LeftToeBase", "mixamorig:LeftToe_End",
              "mixamorig:RightFoot", "mixamorig:RightToeBase", "mixamorig:RightToe_End")
HEAD_BONES = ("mixamorig:Head", "mixamorig:HeadTop_End")
# 脊柱链：加长只发生在这一段，颈/头偏移不动 ⇒ 头相对肩的高度自动保持
SPINE_BONES = ("mixamorig:Spine", "mixamorig:Spine1", "mixamorig:Spine2")


# ---------- GLB 读写 ----------
def load_glb(path: Path):
    b = path.read_bytes()
    off, chunks = 12, {}
    while off < len(b):
        ln, ty = struct.unpack_from("<II", b, off)
        off += 8
        chunks[ty] = (b[off:off + ln], off)
        off += ln
    j = json.loads(chunks[0x4E4F534A][0].decode("utf-8"))
    return j, chunks[0x004E4942][0]


def pack_glb(j: dict, bin_: bytes) -> bytes:
    js = json.dumps(j, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bn = bin_ + b"\0" * ((4 - len(bin_) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(bn)
    out = b"glTF" + struct.pack("<II", 2, total)
    out += struct.pack("<I", len(js)) + b"JSON" + js
    out += struct.pack("<I", len(bn)) + b"BIN\0" + bn
    return out


# ---------- 节点矩阵（列主序，与 three.js 一致） ----------
def node_matrix(n):
    if n.get("matrix"):
        return list(n["matrix"])
    t = n.get("translation", [0, 0, 0])
    r = n.get("rotation", [0, 0, 0, 1])
    s = n.get("scale", [1, 1, 1])
    x, y, z, w = r
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return [
        (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
        (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
        (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
        t[0], t[1], t[2], 1,
    ]


def mat_mul(a, b):
    o = [0.0] * 16
    for c in range(4):
        for r in range(4):
            s = 0.0
            for k in range(4):
                s += a[k * 4 + r] * b[c * 4 + k]
            o[c * 4 + r] = s
    return o


def mat_inv(m):
    """列主序 4x4 求逆，输出仍是列主序（glTF 的 inverseBindMatrices 存法）。"""
    a = np.array(m, dtype=np.float64).reshape(4, 4, order="F")
    return np.linalg.inv(a).reshape(-1, order="F").tolist()


def world_matrices(j):
    nodes = j["nodes"]
    parent = [-1] * len(nodes)
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i
    world = [None] * len(nodes)

    def wof(i):
        if world[i] is None:
            p = parent[i]
            local = node_matrix(nodes[i])
            world[i] = mat_mul(wof(p), local) if p >= 0 else local
        return world[i]

    for i in range(len(nodes)):
        wof(i)
    return world, parent


def acc_read(j, bin_, idx):
    """按 accessor 的 componentType 读顶点属性（JOINTS_0 是 ubyte，不能当 float 读）。"""
    a = j["accessors"][idx]
    bv = j["bufferViews"][a["bufferView"]]
    base = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
    dt = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]]
    stride = bv.get("byteStride") or np.dtype(dt).itemsize * nc
    n = a["count"]
    if stride == np.dtype(dt).itemsize * nc:
        arr = np.frombuffer(bin_, dtype=dt, count=n * nc, offset=base).reshape(n, nc).copy()
    else:
        arr = np.stack([np.frombuffer(bin_, dtype=dt, count=n, offset=base + k * stride)
                        for k in range(nc)], axis=1).copy()
    return arr


# ---------- 主流程 ----------
def main() -> int:
    ap = argparse.ArgumentParser(description="按源视频人物比例重塑 Mixamo 模型")
    ap.add_argument("src", nargs="?", default="models/human.glb")
    ap.add_argument("dst", nargs="?", default="models/human_reshape.glb")
    ap.add_argument("--leg", type=float, default=0.507, help="腿(髋→鞋底)占可见身高目标")
    ap.add_argument("--torso", type=float, default=0.276, help="躯干(肩→髋)占可见身高目标")
    ap.add_argument("--head", type=float, default=0.217, help="头颈(肩→颅顶)占可见身高目标")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    j, bin_ = load_glb(src)
    bin_ = bytearray(bin_)
    nodes = j["nodes"]
    by_name = {n["name"]: i for i, n in enumerate(nodes) if "name" in n}
    world, parent = world_matrices(j)
    prim = j["meshes"][0]["primitives"][0]

    pos = acc_read(j, bin_, prim["attributes"]["POSITION"])
    joints = acc_read(j, bin_, prim["attributes"]["JOINTS_0"]).astype(np.int32)
    weights = acc_read(j, bin_, prim["attributes"]["WEIGHTS_0"]).astype(np.float64)

    def wy(name):
        m = world[by_name[name]]
        return m[13]

    hipY = (wy("mixamorig:LeftUpLeg") + wy("mixamorig:RightUpLeg")) / 2
    shoulderY = (wy("mixamorig:LeftArm") + wy("mixamorig:RightArm")) / 2
    ankleY = (wy("mixamorig:LeftFoot") + wy("mixamorig:RightFoot")) / 2
    soleY = float(pos[:, 1].min())
    crownY = float(pos[:, 1].max())
    V0 = crownY - soleY

    # ---- 解参数（闭式）：V' 由「头颈高度不变」定，其余两个占比反解系数 ----
    headH = crownY - shoulderY
    V1 = headH / a.head
    legF = (a.leg * V1 - (ankleY - soleY)) / (hipY - ankleY)
    torsoT = a.torso * V1 / (shoulderY - hipY)
    spineGrowth = (torsoT - 1.0) * (shoulderY - hipY)      # 肩需上移量（米）

    # ---- 骨偏移（骨局部系，100 单位 = 1cm）----
    # 腿链：膝(LeftLeg.t.y) 与 踝(LeftFoot.t.y) 的偏移按 legF 缩短；脚掌(ToeBase)不动
    for side in ("Left", "Right"):
        knee = by_name[f"mixamorig:{side}Leg"]
        nodes[knee]["translation"][1] *= legF
        ankle = by_name[f"mixamorig:{side}Foot"]
        nodes[ankle]["translation"][1] *= legF
    # 脊柱链：整体加长 spineGrowth（米 → 100 单位），按链长等比放大各节
    chain_len = sum(nodes[by_name[b]]["translation"][1] for b in SPINE_BONES)
    spine_factor = (chain_len + spineGrowth * 100.0) / chain_len
    for b in SPINE_BONES:
        nodes[by_name[b]]["translation"][1] *= spine_factor

    # ---- 网格顶点变形 ----
    joint_names = [nodes[i]["name"] for i in j["skins"][0]["joints"]]
    cat = np.zeros(len(joint_names), dtype=np.int8)       # 0=上半身 1=腿 2=鞋 3=头
    for k, nm in enumerate(joint_names):
        if nm in LEG_BONES:
            cat[k] = 1
        elif nm in FOOT_BONES:
            cat[k] = 2
        elif nm in HEAD_BONES:
            cat[k] = 3
    c = cat[joints]                                        # (n,4)
    leg_w = (weights * (c == 1)).sum(axis=1)
    foot_w = (weights * (c == 2)).sum(axis=1)
    head_w = (weights * (c == 3)).sum(axis=1)
    up_w = 1.0 - leg_w - foot_w                          # 骨盆不动；头随脊柱上移（只改 X/Z）

    d_ankle = (1.0 - legF) * (hipY - ankleY)          # 踝上移量（米）
    new = pos.copy()
    y = pos[:, 1]
    y_up = y + spineGrowth                             # 上半身随脊柱加长整体上移
    y_leg = hipY + (y - hipY) * legF                  # 腿区绕髋压缩（截面宽度不变）
    y_foot = y + d_ankle                               # 鞋整体平移到新踝位
    new[:, 1] = up_w * y_up + leg_w * y_leg + foot_w * y_foot

    # 头部收窄：Mixamo 的头（含发壳）横向 ±10cm，比真人（头宽 15~16cm）宽两成，
    # 渲染出来是「大头」。按头骨轴心做平滑径向压缩，超出 7.5cm 的部分收拢到 8.5cm。
    AXIS_Z = 0.02
    rr = np.hypot(pos[:, 0], pos[:, 2] - AXIS_Z)
    kk = np.minimum(0.075 + (rr - 0.075) * 0.35, 0.085)
    sc = np.where(rr > 0.075, kk / np.maximum(rr, 1e-6), 1.0)
    new[:, 0] = head_w * (pos[:, 0] * sc) + (1 - head_w) * pos[:, 0]
    new[:, 2] = head_w * (AXIS_Z + (pos[:, 2] - AXIS_Z) * sc) + (1 - head_w) * pos[:, 2]

    # 手指收拢：Mixamo 手指在 T 姿势里完全摊开，渲染出来像耙子。
    # 每节绕自身局部 X 轴向掌心弯（掌心朝 +Z ⇒ 弯向 −Z，角度取负），
    # 掌指关节再整体缩 0.88（手指长度回到真人尺度）。
    for i, n in enumerate(nodes):
        nm = n.get("name", "")
        if not nm.startswith("mixamorig:"):
            continue
        m = re.search(r"(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)(\d)$", nm)
        if not m:
            continue
        kind, seg = m.group(2), int(m.group(3))
        ang = {"Thumb": -12.0, "Index": -22.0, "Middle": -22.0, "Ring": -20.0, "Pinky": -18.0}[kind]
        if kind != "Thumb" and seg == 4:
            ang = -8.0
        q = n.get("rotation", [0, 0, 0, 1])
        n["rotation"] = list(q_mul(q, q_axis_x(ang)))
        if seg == 1:
            s = n.get("scale", [1, 1, 1])
            n["scale"] = [v * 0.88 for v in s]

    # ---- 写回：POSITION / IBM / 骨平移 / accessor min-max ----
    acc_p = j["accessors"][prim["attributes"]["POSITION"]]
    bv_p = j["bufferViews"][acc_p["bufferView"]]
    base_p = bv_p.get("byteOffset", 0) + acc_p.get("byteOffset", 0)
    bin_[base_p:base_p + len(pos) * 12] = new.astype("<f4").tobytes()
    acc_p["min"] = [float(v) for v in new.min(axis=0)]
    acc_p["max"] = [float(v) for v in new.max(axis=0)]

    world2, _ = world_matrices(j)
    acc_i = j["accessors"][j["skins"][0]["inverseBindMatrices"]]
    bv_i = j["bufferViews"][acc_i["bufferView"]]
    base_i = bv_i.get("byteOffset", 0) + acc_i.get("byteOffset", 0)
    ibm = []
    for node_idx in j["skins"][0]["joints"]:
        ibm.extend(mat_inv(world2[node_idx]))
    bin_[base_i:base_i + len(ibm) * 4] = np.array(ibm, dtype="<f4").tobytes()

    dst.write_bytes(pack_glb(j, bytes(bin_)))

    # ---- 报告 ----
    sole1 = float(new[:, 1].min())
    crown1 = float(new[:, 1].max())
    V2 = crown1 - sole1
    print(f"目标占比：腿 {a.leg:.3f} / 躯干 {a.torso:.3f} / 头颈 {a.head:.3f}")
    print(f"解得：legF={legF:.4f}  躯干系数={torsoT:.4f}  脊柱链系数={spine_factor:.4f}")
    print(f"踝上移 {d_ankle*100:.2f}cm，可见高度 {V0:.4f} → {V2:.4f} m")
    print(f"落地：髋 {hipY:.4f}→{hipY:.4f}  肩 {shoulderY:.4f}→{shoulderY+spineGrowth:.4f}  "
          f"鞋底 {soleY:.4f}→{sole1:.4f}  颅顶 {crownY:.4f}→{crown1:.4f}")
    print(f"新占比：腿 {(hipY-sole1)/V2:.4f}  躯干 {(shoulderY+spineGrowth-hipY)/V2:.4f}  "
          f"头颈 {(crown1-shoulderY-spineGrowth)/V2:.4f}")
    print(f"✓ {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
