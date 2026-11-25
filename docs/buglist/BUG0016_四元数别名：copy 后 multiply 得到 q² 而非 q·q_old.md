# BUG0016 四元数别名：copy 后 multiply 得到 q² 而非 q·q_old

- 状态：已修 · 2026-11-23
- 现象 / 根因：IK 误差从 4mm 变 0.7m。
- 修法：先 const old = bone.quaternion.clone() 再相乘。
- 回归：E2
