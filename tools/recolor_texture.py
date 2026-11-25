"""按几何 + UV 岛给人体模型贴图换装（对齐源视频的服装配色）。

用法: python3 tools/recolor_texture.py models/human.glb models/human_recolor.png [--report]

判定的两个维度：
  · **UV 岛**：图集里每件衣物是独立的岛，岛的颜色统计决定它属于哪件衣物。
  · **主导骨骼组**：同一件衣物的几何可能跨骨（袖子与手掌在同一岛），
    此时必须按顶点的主导骨骼再切一刀。

为什么两者都要：只按颜色判定会把护目镜当头发、只按岛判定会把袖子连同手套一起涂成连帽衫。
判定必须引入几何（绑定姿势高度 + 骨骼组），纯色相判定解决不了四件事：
  · 头部的护目镜是深色件，按颜色会当成头发留下（源视频表演者没戴护目镜）；
  · 袖子与手掌在同一 UV 岛，必须按骨拆开（袖子藏青、手是裸手）；
  · 腰腹（露出的白 T 恤下摆）与胸口同属躯干，只有靠高度才能区分；
  · 白色袜口与鞋面同在下肢，只有靠骨骼组 + 明度才能区分。
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

# 服装配色（对齐源视频：深藏青连帽衫 + 白色内搭下摆 + 浅蓝牛仔裤 + 白鞋）
# 色相用「度」（0-360 的常规 HSV）；写入 OpenCV 的 uint8 HSV 时要除以 2，
# 否则超过 179 的色相会回绕（235° 写成 215 会变成 70° 的绿色）。
OUTFIT = {
    "denim": dict(hue=205, ks=0.55, kv=0.82, label="裤装→浅蓝牛仔"),
    "skin": dict(hue=28, ks=0.75, kv=1.55, label="肤色→浅肤色"),
    # 藏青用「绝对饱和度」：乘算会把原本低饱和的面料压成灰黑，读不出颜色
    "hoodie": dict(hue=232, s=140, kv=0.55, label="躯干/袖子→深藏青连帽衫"),
    "hem": dict(hue=0, ks=0.06, kv=1.30, label="内搭下摆→白"),
    "shoe": dict(hue=0, ks=0.10, kv=1.35, vmin=168, label="鞋面→白"),
    "face": dict(hue=None, label="头部彩色件→肤色"),
}
KEEP = dict(hsv=None, label="保持原样")
GROUPS = ["head", "torso", "arm", "hand", "leg", "shoe", "other"]


def load_gltf(glb: Path):
    b = glb.read_bytes()
    off, chunks = 12, {}
    while off < len(b):
        ln, ty = struct.unpack_from("<II", b, off)
        off += 8
        chunks[ty] = b[off:off + ln]
        off += ln
    j = json.loads(chunks[0x4E4F534A].decode("utf-8"))
    return j, chunks[0x004E4942]


CT = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(j, bin_, i):
    acc = j["accessors"][i]
    bv = j["bufferViews"][acc["bufferView"]]
    n = NC[acc["type"]]
    dt = np.dtype(CT[acc["componentType"]])
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * n
    count = acc["count"]
    if stride == dt.itemsize * n:
        a = np.frombuffer(bin_, dtype=dt, count=count * n, offset=base).reshape(count, n)
    else:
        a = np.stack([np.frombuffer(bin_, dtype=dt, count=n, offset=base + k * stride) for k in range(n)])
    return a.astype(np.float32)


def vertex_bones(j, bin_, prim):
    """每顶点的蒙皮主导骨骼名（用于判断它属于头/躯干/袖/手等部位）。"""
    nodes = j["nodes"]
    if "JOINTS_0" not in prim["attributes"]:
        return None
    joints = read_accessor(j, bin_, prim["attributes"]["JOINTS_0"]).reshape(-1, 4).astype(np.int32)
    weights = read_accessor(j, bin_, prim["attributes"]["WEIGHTS_0"]).reshape(-1, 4)
    skin = j["skins"][prim.get("skin", 0)] if "skins" in j else None
    jmap = skin["joints"] if skin else list(range(len(nodes)))
    out = []
    for k in range(len(joints)):
        w = weights[k]
        if w.max() <= 0:
            out.append("")
            continue
        ni = int(joints[k][int(np.argmax(w))])
        out.append(nodes[jmap[ni]].get("name", "") if ni < len(jmap) else "")
    return out


def bone_group(bone):
    """按蒙皮主导骨骼判定衣物归属。

    锁骨（LeftShoulder/RightShoulder）必须归到袖子：Mixamo 的锁骨骨名里没有
    "arm" 子串，落到 other 分支就会保留模型原本的黄绿色袖子。
    """
    b = (bone or "").lower()
    if "head" in b or "neck" in b:
        return "head"
    if "spine" in b or "hips" in b:
        return "torso"
    if "hand" in b:
        return "hand"
    if "arm" in b or "shoulder" in b:
        return "arm"
    if "foot" in b or "toe" in b:
        return "shoe"
    if "upleg" in b or "leg" in b:
        return "leg"
    return "other"


def _tri_uv_points(uv, tris, size):
    px = np.clip((uv[:, 0] * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip((uv[:, 1] * (size - 1)).astype(np.int32), 0, size - 1)
    return np.stack([px, py], axis=1)


def uv_islands(j, bin_, size=512):
    """UV 图集上的连通岛：每件衣物一个岛。"""
    mesh = j["meshes"][0]
    prim = mesh["primitives"][0]
    uv = read_accessor(j, bin_, prim["attributes"]["TEXCOORD_0"])
    if "indices" in prim:
        idx = read_accessor(j, bin_, prim["indices"]).reshape(-1).astype(np.int32)
    else:
        idx = np.arange(len(uv), dtype=np.int32)
    tris = idx.reshape(-1, 3)
    tpts = _tri_uv_points(uv, tris, size)[tris]
    label = np.zeros((size, size), np.int32)
    for t in range(len(tris)):
        cv2.fillPoly(label, [tpts[t]], t + 1)
    return label, tris


def group_map(j, bin_, prim, size):
    """逐像素的「主导骨骼组」标签：每个三角形按三顶点组的多数票落格。

    袖子与手掌在同一个 UV 岛里，只有按骨切开才能分别涂成藏青与肤色。
    """
    uv = read_accessor(j, bin_, prim["attributes"]["TEXCOORD_0"])
    idx = read_accessor(j, bin_, prim["indices"]).reshape(-1).astype(np.int32)
    tris = idx.reshape(-1, 3)
    vb = vertex_bones(j, bin_, prim)
    gidx = np.array([GROUPS.index(bone_group(b)) for b in vb], dtype=np.int32)
    tpts = _tri_uv_points(uv, tris, size)[tris]
    label = np.zeros((size, size), np.int32)
    for t, tri in enumerate(tris):
        c = int(np.bincount(gidx[tri], minlength=len(GROUPS)).argmax())
        cv2.fillPoly(label, [tpts[t]], c + 1)
    return label, gidx


def vertex_islands(label, uv):
    """顶点 → 岛号（按 UV 采样图集标签）。"""
    H, W = label.shape
    px = np.clip((uv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    py = np.clip((uv[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    return label[py, px]


def sub_regions(j, bin_, prim, lab_full, grp_full):
    """把图集切成「岛 × 骨骼组」的子区域，并给出每个子区域的顶点几何统计。"""
    uv = read_accessor(j, bin_, prim["attributes"]["TEXCOORD_0"])
    pos = read_accessor(j, bin_, prim["attributes"]["POSITION"])
    vlab = vertex_islands(lab_full, uv)
    _, gidx = group_map(j, bin_, prim, min(lab_full.shape))
    regions = []
    for cid in np.unique(vlab):
        if cid == 0:
            continue
        for gi, g in enumerate(GROUPS):
            vsel = (vlab == cid) & (gidx == gi)
            if int(vsel.sum()) < 8:
                continue
            mask = (lab_full == cid) & (grp_full == gi + 1)
            if mask.sum() < 8:
                continue
            p = pos[vsel]
            regions.append(dict(cid=int(cid), group=g, mask=mask, vsel=vsel,
                                n=int(vsel.sum()), px=int(mask.sum()),
                                y=float(np.median(p[:, 1])), x=float(np.median(p[:, 0])),
                                z=float(np.median(p[:, 2])),
                                ymin=float(p[:, 1].min()), ymax=float(p[:, 1].max())))
    return regions


def region_color(img, hsv, mask):
    """子区域的颜色统计：均值 RGB + 中位 HSV。"""
    mr = img[mask].mean(axis=0)[::-1]
    mh = np.median(hsv[mask].reshape(-1, 3), axis=0)
    return mr, mh


def classify(st, belly_y):
    """按「骨骼组 + 几何高度 + 色相」判定衣物。"""
    group, (h, s, v) = st["group"], st["hsv"]
    if group == "head":
        # 用「岛的中位饱和度」而不是像素占比：岛边界膨胀会吃到邻岛的红色镜圈，
        # 像素占比会把头发误判成彩色配件。头发=低饱和深色 → 保留；
        # 护目镜（红圈/镜片）= 高饱和深色 → 涂肤色（源视频表演者没戴护目镜，
        # 而且 tools/trim_buns.py 会把镜框顶点收成一点，贴图若留深色，
        # 收拢处会把深色 UV 拖成脸上的一团黑斑，像戴了护目镜）。
        if v < 90:
            return ("face", OUTFIT["face"]) if s >= 70 else ("keep", KEEP)
        return "face", OUTFIT["face"]
    if group == "torso":
        # 躯干整体按连帽衫：源视频的宽大连帽衫盖住腰腹，
        # 露出的白 T 恤下摆由查看器里的白边几何承担（贴图上腰腹是一整块）
        return "hoodie", OUTFIT["hoodie"]
    if group == "arm":
        return "hoodie", OUTFIT["hoodie"]
    if group == "hand":
        return "skin", OUTFIT["skin"]       # 裸手
    if group == "leg":
        return "denim", OUTFIT["denim"]    # 裤装 + 白色袜口 → 浅蓝牛仔
    if group == "shoe":
        # 源视频表演者穿白色运动鞋 → 整双鞋（含深色鞋底/鞋面配件）都刷白。
        # 早先只刷 v>150 的亮面，深色鞋块保留原色（贴图上是彩色拼块），缩略图里
        # 看起来像一双花鞋。
        return "shoe", OUTFIT["shoe"]
    if v < 45 and s < 70:
        return "keep", KEEP
    if 5 <= h <= 17 and s >= 90 and v >= 60:
        return "skin", OUTFIT["skin"]
    return "keep", KEEP


def main() -> int:
    ap = argparse.ArgumentParser(description="人体模型换装贴图")
    ap.add_argument("glb", nargs="?", default="models/human.glb")
    ap.add_argument("out", nargs="?", default="models/human_recolor.png")
    ap.add_argument("--report", action="store_true", help="只打印逐区域判定表")
    args = ap.parse_args()

    glb, out = Path(args.glb), Path(args.out)
    j, bin_ = load_gltf(glb)
    mesh = j["meshes"][0]
    prim = mesh["primitives"][0]
    ti = prim["material"]
    tex = j["materials"][ti]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    img_spec = j["images"][j["textures"][tex]["source"]]
    bv = j["bufferViews"][img_spec["bufferView"]]
    raw = bin_[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    tmp = out.with_suffix(".src.png")
    tmp.write_bytes(raw)
    img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    label, tris = uv_islands(j, bin_, size=min(W, H))
    # 同一件衣物的三角形在图集上相邻 → 用连通域把三角形标签合并成「岛」
    n_cc, cc = cv2.connectedComponents((label > 0).astype(np.uint8), connectivity=8)
    remap = np.zeros(n_cc, np.int32)
    for cid in range(1, n_cc):
        vals = np.unique(label[cc == cid])
        remap[cid] = vals[vals > 0][0] if (vals > 0).any() else 0
    label = np.where(cc > 0, remap[cc], 0).astype(np.int32)
    lab_full = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST) if label.max() > 0 \
        else np.zeros((H, W), np.int32)
    grp_full, _ = group_map(j, bin_, prim, min(W, H))
    grp_full = cv2.resize(grp_full, (W, H), interpolation=cv2.INTER_NEAREST)

    regions = sub_regions(j, bin_, prim, lab_full, grp_full)
    for st in regions:
        st["rgb"], st["hsv"] = region_color(img, hsv, st["mask"])
    # 躯干：按高度排序，最低的子区域 = 腰腹（露出的内搭下摆）
    torso = [st for st in regions if st["group"] == "torso"]
    torso.sort(key=lambda st: st["y"])
    belly_y = torso[0]["y"] if torso else -1e9

    if args.report:
        print(f"{'岛':>6} {'组':>6} {'顶点':>6} {'像素':>7} {'y':>7} {'x':>7} {'z':>7} "
              f"{'H':>4} {'S':>4} {'V':>4}  判定")
        for st in sorted(regions, key=lambda s: -s["px"]):
            kind, _ = classify(st, belly_y)
            print(f"{st['cid']:6d} {st['group']:>6} {st['n']:6d} {st['px']:7d} {st['y']:7.3f} "
                  f"{st['x']:7.3f} {st['z']:7.3f} {st['hsv'][0]:4.0f} {st['hsv'][1]:4.0f} "
                  f"{st['hsv'][2]:4.0f}  {kind}")
        print(f"\n腰腹分界 y = {belly_y:.3f}")
        return 0

    counts = {}
    for st in sorted(regions, key=lambda s: -s["px"]):
        kind, spec = classify(st, belly_y)
        counts[kind] = counts.get(kind, 0) + st["px"]
        if kind == "keep":
            continue
        m = cv2.dilate(st["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if kind == "face":
            # 护目镜等彩色件：整块涂肤色（只涂饱和像素会留下黑色镜筒）
            out_rgb = img.astype(np.float32).copy()
            lowsel = m & (hsv[:, :, 1] < 70)
            fill = np.median(img[lowsel].reshape(-1, 3), axis=0) if lowsel.sum() >= 30 \
                else np.array([145, 175, 215], np.float32)
            out_rgb[m] = fill
            img = np.clip(out_rgb, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            continue
        s_ = hsv[:, :, 1].astype(np.int16)
        v_ = hsv[:, :, 2].astype(np.int16)
        nh, kv = round(spec["hue"] / 2), spec["kv"]   # OpenCV uint8 HSV 的色相范围 0-179
        s_new = spec["s"] if "s" in spec else np.clip(s_ * spec["ks"], 0, 255)
        hsv[:, :, 0] = np.where(m, nh, hsv[:, :, 0])
        hsv[:, :, 1] = np.where(m, s_new, hsv[:, :, 1])
        v_new = np.clip(v_ * kv, 0, 255)
        if spec.get("vmin"):
            v_new = np.maximum(v_new, spec["vmin"])
        hsv[:, :, 2] = np.where(m, v_new, hsv[:, :, 2])
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)   # 写回，后续区域基于此图

    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {kind}: {n} 像素 ({n / (W * H) * 100:.1f}%)")
    cv2.imwrite(str(out), img)
    tmp.unlink(missing_ok=True)
    print(f"✓ {out} ({out.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
