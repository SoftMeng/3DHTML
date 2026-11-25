# BUG0006 facing 270 被算成背对相机

- 状态：已修 · 2026-11-23
- 现象 / 根因：yawOf 用 atan2(−cos,−sin)，与 atan2(−cos, sin) 差 180°。
- 修法：yawOf(f)=atan2(−cos f, sin f)。
- 回归：D2 朝向
