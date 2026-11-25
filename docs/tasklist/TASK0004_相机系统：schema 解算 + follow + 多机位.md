# TASK0004 相机系统：schema 解算 + follow + 多机位

- 状态：已完成 · 2026-11-23
- 产出：`index.html:updateCamera()` · `index.html:yawOf()`

1. 首帧相机在世界原点朝 +Z；此时屏幕右 = 世界 −X → **规格 x 入料必须取负**。
2. 机位组：规格 / 对照机位 / 叠加对照 / 观众席 / 近景 / 俯视。
3. yawOf(f)=atan2(−cos f, sin f)（曾用 atan2(−cos,−sin) 差 180°，facing 270 变成背对）。

- 验收：A2 相机解算 · D2 朝向 · D3 镜像方向
