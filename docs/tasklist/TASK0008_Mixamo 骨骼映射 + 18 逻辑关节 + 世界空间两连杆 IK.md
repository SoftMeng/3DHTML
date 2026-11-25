# TASK0008 Mixamo 骨骼映射 + 18 逻辑关节 + 世界空间两连杆 IK

- 状态：已完成 · 2026-11-23
- 产出：`index.html:BONE_MAP` · `index.html:solveChain()` · `index.html:captureRig()`

1. 65 骨 → 18 逻辑关节；上臂映射到 LeftArm（LeftShoulder 是锁骨）。
2. IK 全程世界空间：方向取世界坐标差（无量纲）、长度取世界距离。
3. 绝对姿态赋值：`bone.quaternion = q_delta · initQuat`（逐帧 multiply 增量会累积甩飞）。

- 验收：E1 骨骼映射 · E2 IK 命中精度 ≤0.25m
