# BUG0036 ffmpeg 抽帧不可用（libhvf.dylib 缺失）→ 改用 venv 的 cv2

- 现象：`ffmpeg -i … -vframes 1 out.png` 直接崩溃退出 134：`dyld: Library not loaded: /usr/lib/libhvf.dylib`（homebrew 的 freetype 依赖 macOS 内部库，视频工具链在当前系统上不可用）。
- 根因：brew 装的 ffmpeg 链接了系统私有库，系统版本升级后缺失。
- 修法：抽帧一律用 `tools/.venv/bin/python` 的 cv2（已验证可用，5.1.0），不再依赖 ffmpeg。
- 影响：工具链约定 —— 新脚本不要调 ffmpeg；媒体解码走 cv2 / mediapipe。

- 回归：`tools/.venv/bin/python -c "import cv2; …"` 可解码并写帧
