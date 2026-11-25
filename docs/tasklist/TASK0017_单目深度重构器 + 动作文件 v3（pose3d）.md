# TASK0017 单目深度重构器 + 动作文件 v3（pose3d）

- 状态：已完成 · 2026-11-25
- 产出：`tools/depth_reconstruct.py` · `tools/upgrade_motion.py` · `tools/extract_motion.py`

1. 单目下每根骨的深度差由**透视缩短**唯一确定：|Δz|=√(L²−d²)（L=解剖骨长，d=该骨画面投影长度）。
2. 符号由解剖先验（肘偏后/膝偏前/面朝前）+ 帧间连续性用 2 态 Viterbi 裁决；**横向骨（肩/髋/耳）的深度直接由偏航给出 ±L·sin ψ**。
3. 输出 pose3d（相机系，米，髋中点原点）+ pose3dConf（逐骨可信度）；旧 poses3d 保留但弃用。
4. v1/v2 读取路径不动：无 pose3d 时四肢退回 2D 路径。

- 验收：深度离线验收 5 条断言（tools/verify_depth.py）全绿
