# TASK0007 MediaPipe Pose 后端 + 逐帧归一化 + 语义通道

- 状态：已完成 · 2026-11-24
- 产出：`tools/extract_motion.py:extract_mediapipe()` · `tools/motion_pipeline.py`

1. 换 BlazePose：置信 0.68 → 0.91（MoveNet 在宽大连帽衫上把双腕判到髋部）。
2. 逐帧归一化（单位=该帧躯干长）→ 姿态与景深解耦；旧全局单位下躯干长极差 0.78 → 现 0.09。
3. 3 帧中值去离群（保阶跃）；**不做低通**：低通带滞后 125ms，滞后比噪声更伤一致性。
4. 通道：yaw（肩宽透视收缩定幅度、鼻−耳定方向）/ roll（躯干轴倾角，扣相机滚转）/ root / contact / conf。

- 验收：E10 · E11 · E12 通道物理可信
