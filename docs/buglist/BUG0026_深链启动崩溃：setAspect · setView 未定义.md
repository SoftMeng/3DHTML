# BUG0026 深链启动崩溃：setAspect / setView 未定义

- 状态：已修 · 2026-11-25
- 现象 / 根因：两函数只作为 window.__app 的对象方法存在，loadSpec 的深链还原块以自由函数调用 → ReferenceError: setAspect is not defined，页面「启动失败」。
- 修法：按既有 setPlaying 范式上提到模块作用域，__app 改用简写引用。
- 回归：B17 深链还原
