# BUG0019 把锁骨当上臂，IK 目标指向孙骨致骨骼乱甩

- 状态：已修 · 2026-11-23
- 现象 / 根因：Mixamo 骨链 LeftShoulder→LeftArm→LeftForeArm→LeftHand。
- 修法：上臂映射到 LeftArm。
- 回归：E1
