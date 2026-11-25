# BUG0021 共享 wrapper 累积挂件，draw call 线性增长

- 状态：已修 · 2026-11-24
- 现象 / 根因：换规格重建时每次往头骨挂新头发。
- 修法：挂件缓存复用（hairGroup 单例）。
- 回归：C2 DrawCall ≤40
