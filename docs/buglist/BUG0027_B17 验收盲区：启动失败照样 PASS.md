# BUG0027 B17 验收盲区：启动失败照样 PASS

- 状态：已修 · 2026-11-25
- 现象 / 根因：B17 导航的正是会崩溃的深链 URL，但 waitReady() 返回值被丢弃、断言只查 playhead/spec/url —— 三项在抛错点之前已赋值。
- 修法：断言补 ready、错误面板、aspect、view 四项。
- 回归：B17
