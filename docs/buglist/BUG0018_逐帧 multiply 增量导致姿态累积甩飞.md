# BUG0018 逐帧 multiply 增量导致姿态累积甩飞

- 状态：已修 · 2026-11-23
- 现象 / 根因：每帧 −0.013rad，70 帧后 90°。
- 修法：用初始姿势作基准的绝对姿态：bone.quaternion = q_delta · initQuat。
- 回归：E6
