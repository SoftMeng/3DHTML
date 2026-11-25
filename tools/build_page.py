#!/usr/bin/env python3
"""渲染层构建：把 src/ 的 ES 模块内联成**单文件** index.html。

为什么需要它：交付形态必须是**一个 HTML 文件**（file:// 直开零 CORS，见验收 B1b），
而开发期源码要按职责分模块（src/core 纯几何内核 + src/app 应用层）。
本脚本只做一件事：按清单顺序拼接模块、去掉 import/export 关键字、塞进模板。

用法:
    tools/.venv/bin/python tools/build_page.py            # 生成 index.html
    tools/.venv/bin/python tools/build_page.py --check    # 只校验：重建结果 == 签入产物
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TEMPLATE = SRC / "page.template.html"
OUT = ROOT / "index.html"
MARKER = "/*@BUNDLE@*/"

# 拼接顺序 = 原始单文件的文本顺序（保证重构前后行为等价）
# 内核在前（新代码，位置无关）；main.js（常量与 three.js 载入）随后，其余按原顺序。
MANIFEST = [
    "core/vec.js",
    "core/geom.js",
    "core/ik.js",
    "core/clip.js",
    "main.js",
    "app/state.js",
    "app/scene.js",
    "app/figure.js",
    "app/solver.js",
    "app/shell.js",
]

IMPORT_RE = re.compile(r"^\s*import\s+.*?from\s+['\"].*?['\"];?\s*$", re.M)
EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{[^}]*\}\s*;?\s*$", re.M)
EXPORT_DECL_RE = re.compile(r"^(\s*)export\s+(function|const|let|class|async function)\b", re.M)
DECL_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)")
VAR_RE = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+([^=;\n]*)")


def top_level_names(src: str):
    """模块的顶层声明名（内联后进同一作用域，重名会互相覆盖）。"""
    names = []
    for ln in src.split("\n"):
        m = DECL_RE.match(ln)
        if m:
            names.append(m.group(1))
            continue
        m = VAR_RE.match(ln)
        if m:
            for part in m.group(1).split(","):
                nm = part.split("=")[0].strip()
                if re.match(r"^[A-Za-z_$][\w$]*$", nm):
                    names.append(nm)
    return names


def check_collisions(mods):
    seen = {}
    dup = []
    for rel, src in mods:
        for n in top_level_names(src):
            if n in seen:
                dup.append(f"{n}（{seen[n]} 与 {rel}）")
            else:
                seen[n] = rel
    if dup:
        raise SystemExit("模块间顶层重名（内联后互相覆盖）：" + "; ".join(dup))


def strip_module(src: str) -> str:
    """去掉模块语法：import 行、export 声明前缀、export{...} 列表。"""
    out = IMPORT_RE.sub("", src)
    out = EXPORT_LIST_RE.sub("", out)
    out = EXPORT_DECL_RE.sub(r"\1\2", out)
    return out


def bundle() -> str:
    mods = []
    for rel in MANIFEST:
        p = SRC / rel
        if not p.exists():
            raise SystemExit(f"缺少模块 {rel}")
        mods.append((rel, p.read_text(encoding="utf-8")))
    check_collisions(mods)
    parts = []
    for rel, src in mods:
        parts.append(f"/* ===== {rel} ===== */\n" + strip_module(src).strip() + "\n")
    return "\n".join(parts)


def build() -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    if MARKER not in tpl:
        raise SystemExit(f"模板缺少占位符 {MARKER}")
    return tpl.replace(MARKER, bundle())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验重建结果与签入产物一致")
    args = ap.parse_args()
    out = build()
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != out:
            print("✗ index.html 与 src/ 不同步：跑 tools/build_page.py 重新生成", file=sys.stderr)
            return 1
        print("✓ index.html 与 src/ 同步（重建逐字节一致）")
        return 0
    OUT.write_text(out, encoding="utf-8")
    print(f"✓ {OUT.name} ← {len(MANIFEST)} 个模块（{len(out.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
