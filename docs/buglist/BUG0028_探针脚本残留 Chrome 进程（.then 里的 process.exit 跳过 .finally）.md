# BUG0028 探针脚本残留 Chrome 进程（.then 里的 process.exit 跳过 .finally）

- 状态：已修 · 2026-11-25
- 现象 / 根因：main().then(()=>process.exit(0)).finally(cleanup) → 立即终止进程，清理不执行，实测残留 13 个 Chrome。
- 修法：用 process.exitCode，在 finally 内 setTimeout 后再退出；结束时 ps 自查。
- 回归：残留浏览器进程 = 0
