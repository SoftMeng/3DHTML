# TASK0001 规格 Schema v1 与示例规格

- 状态：已完成 · 2026-11-23
- 产出：`示例.json` · `index.html:normalize()`

1. 规格 v1 = 轨迹与机位的唯一真相源；origin=首帧相机正下方地面点，+x=相机右，+y=相机初始朝向。
2. facing_deg 从 +x 逆时针（90=背对相机，270=面向相机）；framing r=可见高/画面高。
3. camera.fov_deg 是**水平** FOV，three.js 需换算垂直：2·atan(tan(hfov/2)/aspect)。

- 验收：A1 逐点复算一致 · A2 相机解算与 framing 分档
