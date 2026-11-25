"""把 Mixamo 模型的「太空髻」收进颅内，得到干净的后脑勺剪影。

用法: python3 tools/trim_buns.py models/human.glb

为什么：源视频的表演者是及腰直发，Mixamo 女模的头顶是两组外扩的髻球
（x≈±0.2m、半径 6cm），套任何发片都盖不住 —— 只能从几何上收掉。
做法：按绑定姿势的几何挑出髻部顶点（头部骨、|x| 大、位置高），
沿 x 向颅轴压缩并向头心回收，然后原地写回 POSITION 数据
（字节数不变 → 缓冲视图偏移全部保持有效）。

判据要严：只动 |x|>阈值 且 y 高于阈值 的顶点，
颅顶的刘海与后脑的头发（|x| 小）保持原样。
"""
from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recolor_texture as RT


def source_texture(j, bin_):
    """GLB 内嵌的原始贴图（PNG 字节）——换装前的原色，用来识别护目镜镜框。"""
    prim = j["meshes"][0]["primitives"][0]
    tex = j["materials"][prim["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    spec = j["images"][j["textures"][tex]["source"]]
    bv = j["bufferViews"][spec["bufferView"]]
    raw = bin_[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    import cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def load_gltf(glb: Path):
    b = glb.read_bytes()
    off, chunks = 12, {}
    while off < len(b):
        ln, ty = struct.unpack_from("<II", b, off)
        off += 8
        chunks[ty] = (b[off:off + ln], off)
        off += ln
    j = __import__("json").loads(chunks[0x4E4F534A][0].decode("utf-8"))
    return j, chunks, bytearray(b)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/human.glb")
    j, chunks, buf = load_gltf(path)
    prim = j["meshes"][0]["primitives"][0]
    acc = j["accessors"][prim["attributes"]["POSITION"]]
    bv = j["bufferViews"][acc["bufferView"]]
    bin_chunk, bin_off = chunks[0x004E4942]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    n = acc["count"]
    pos = np.frombuffer(bin_chunk, dtype="<f4", count=n * 3, offset=base).reshape(n, 3).copy()

    # 头部横向外扩的部分：外扩发髻 + 两侧短发的鬓发（源视频是及腰直发，
    # 模型的齐耳短发外扩 24cm，比真实头宽大一倍 —— 任何发片都盖不住）。
    # 只动**头骨蒙皮**的顶点：按全体顶点的 y 筛选会误伤张开的双臂。
    vbones = np.array(RT.vertex_bones(j, chunks[0x004E4942][0], prim), dtype=object)
    head_sel = vbones == "mixamorig:Head"
    # 头发壳整体收拢：Mixamo 女的头（含发）横向 ±15cm、顶到 1.635，
    # 比真人（头+发 ≈ ±9cm、颅顶 1.586）大了一圈 —— 套发片也盖不住，
    # 渲染出来是一颗戴桶的怪头。
    # 做法：对头骨蒙皮顶点做**平滑径向压缩** —— 颅缘（8.2cm）以内的顶点
    # （脸颊/下颌/鼻梁）完全不动，超出的按比例收拢并夹到 10.5cm；
    # 发量还在，但贴合头形。
    EDGE, SHRINK, CAP_R, Z0 = 0.082, 0.30, 0.105, 0.02
    r = np.hypot(pos[:, 0], pos[:, 2] - Z0)
    shell = head_sel & (r > EDGE)
    k = np.minimum(EDGE + (r - EDGE) * SHRINK, CAP_R)
    scale = np.where(shell, k / np.maximum(r, 1e-6), 1.0)
    pos[:, 0] *= scale
    pos[:, 2] = Z0 + (pos[:, 2] - Z0) * scale
    # 颅顶之上的发际：真人头顶只比颅顶高 2~4cm（视频里是毛线帽）。
    # 只压「颅缘之外」的那圈顶点（发壳），颅顶正中与颅面不受影响 ——
    # 否则会把颅骨也压扁。
    top = head_sel & (r > EDGE) & (pos[:, 1] > 1.55)
    pos[top, 1] = 1.55 + (pos[top, 1] - 1.55) * 0.50
    sel = shell
    x0 = float(np.abs(pos[head_sel, 0]).max()) if head_sel.any() else 0.0

    # 护目镜镜框：模型戴着红色护目镜，源视频表演者没有。
    # 按原贴图里的红色环定位（UV 落在红环上的顶点），连同镜片一起收缩成一点
    # （退化三角形不占像素；比删三角形安全 —— 索引缓冲与文件大小都不变）。
    hsv = source_texture(j, chunks[0x004E4942][0])
    # 纯红镜圈：H<6 且高饱和。放宽到 S>140 会把偏红的肤色一起吃掉（脸就没了）
    red = (hsv[:, :, 0] < 6) & (hsv[:, :, 1] > 180) & (hsv[:, :, 2] > 120)
    red = cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    uv = RT.read_accessor(j, chunks[0x004E4942][0], prim["attributes"]["TEXCOORD_0"])
    vbones = np.array(RT.vertex_bones(j, chunks[0x004E4942][0], prim), dtype=object)
    H, W = hsv.shape[:2]
    px = np.clip((uv[:, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    py = np.clip((uv[:, 1] * (H - 1)).astype(np.int32), 0, H - 1)
    on_red = red[py, px] & (vbones == "mixamorig:Head")
    if on_red.any():
        pos[on_red] = np.array([0.0, 1.44, 0.02], np.float32)

    # 原地写回：字节数不变 → 缓冲视图偏移与文件大小全部保持有效
    at = bin_off + base
    buf[at:at + n * 12] = pos.astype("<f4").tobytes()
    path.write_bytes(bytes(buf))
    print(f"✓ {path}  收拢 {int(sel.sum())} 个发壳顶点（头宽 ±{x0:.3f}m → ±{CAP_R:.3f}m），"
          f"收缩 {int(on_red.sum())} 个护目镜顶点")
    return 0

if __name__ == "__main__":
    sys.exit(main())
