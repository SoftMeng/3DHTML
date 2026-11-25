#!/usr/bin/env python3
"""本地静态服务器：为 3D 轨迹查看器提供规格文件自动枚举。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
}
SPEC_SUFFIXES = {".json"}
MEDIA_SUFFIXES = {".mp4", ".webm", ".mov"}
EXCLUDED_NAMES = {"specs.json", "media.json"}


def _list_files(base_dir: Path, suffixes: set[str]) -> list[dict]:
    """按扩展名列出目录文件，按文件名排序。"""
    items = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_file() or path.name in EXCLUDED_NAMES or path.name.startswith("."):
            continue
        if path.suffix.lower() not in suffixes:
            continue
        items.append({"name": path.name, "bytes": path.stat().st_size})
    return items


def list_specs(base_dir: Path) -> list[dict]:
    """目录内的 JSON 规格清单，排除清单文件自身与隐藏项。"""
    return _list_files(base_dir, SPEC_SUFFIXES)


def list_media(base_dir: Path) -> list[dict]:
    """目录内的视频文件清单。"""
    return _list_files(base_dir, MEDIA_SUFFIXES)


class TrajectoryHandler(BaseHTTPRequestHandler):
    server_version = "TrajectoryViewer/1.0"

    def __init__(self, *args, base_dir: Path, **kwargs):
        self.base_dir = base_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/specs":
            self._send_json(list_specs(self.base_dir))
            return
        if path == "/api/media":
            self._send_json(list_media(self.base_dir))
            return
        self._serve_file(path)

    def _serve_file(self, url_path: str) -> None:
        rel = url_path.lstrip("/") or "index.html"
        target = (self.base_dir / rel).resolve()
        try:
            target.relative_to(self.base_dir)
        except ValueError:
            self._send_error(HTTPStatus.FORBIDDEN, "路径越界")
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"不存在: {url_path}")
            return
        data = target.read_bytes()
        ctype = MIME_TYPES.get(target.suffix.lower(), "application/octet-stream")
        span = self._parse_range(len(data))
        if span:
            start, end = span
            chunk = data[start:end + 1]
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self._send_bytes(data, ctype)

    def _parse_range(self, total: int) -> tuple[int, int] | None:
        header = self.headers.get("Range")
        if not header:
            return None
        match = re.match(r"bytes=(\d*)-(\d*)", header.strip())
        if not match:
            return None
        first, last = match.group(1), match.group(2)
        if first == "" and last == "":
            return None
        if first == "":
            length = int(last)
            start = max(0, total - length)
            return start, total - 1
        start = int(first)
        end = int(last) if last else total - 1
        start = max(0, min(start, total - 1))
        end = max(start, min(end, total - 1))
        return start, end

    def _send_json(self, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(body, MIME_TYPES[".json"])

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"error": status.phrase, "detail": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", MIME_TYPES[".json"])
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D 轨迹查看器本地服务器")
    parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent, help="服务目录，默认脚本所在目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.dir.resolve()
    if not base_dir.is_dir():
        print(f"服务目录不存在: {base_dir}", file=sys.stderr)
        return 2
    handler = partial(TrajectoryHandler, base_dir=base_dir)
    try:
        server = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        print(f"无法绑定 {args.host}:{args.port} —— {exc}", file=sys.stderr)
        return 1
    print(f"轨迹查看器: http://{args.host}:{args.port}/  (Ctrl+C 停止)")
    print(f"服务目录: {base_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
