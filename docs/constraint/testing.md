# Testing

## 三层

1. **纯计算断言**（秒级，零浏览器）：深度重构性质、动作文件格式与语义、模型骨长与比例。优先写在这一层。
2. **视觉探针**（单浏览器实例，一次多时刻）：把要看的画面合并到**一次**里取完，产出 PNG + 关键量 JSON。用于判断穿模、掌向、稻草人手臂等「看不数字」的问题。
3. **有界验收**（每轮一次）：`node tools/verify.mjs`。

## 浏览器纪律

- 同一时刻只允许 **1 个** Chrome 实例。
- 每步硬超时（单步 ≤15s，整体 ≤120s）；`try/finally` 保证异常路径也 kill。
- 结束 `ps` 自查残留进程数必须为 0。
- 启动参数固定：`--headless=new --enable-unsafe-swiftshader --use-angle=swiftshader --force-device-scale-factor=1 --window-size=360,640 --autoplay-policy=no-user-gesture-required`。
- 页面加载后立刻停连续渲染（`__app.setLoop(false)`），只在要截图时同步渲染一帧。
- 截图用 `canvas.toDataURL()`（headless 下 `Page.captureScreenshot` 只有黑图）。
- 启动用 `detached:true` + `process.kill(-pid)` 杀整个进程组（杀父进程会留下渲染子进程）。
- Node22 的全局 `fetch` 收大响应偶发 raw-throw → 探针一律用 `node:http`；注册 `uncaughtException` 做**同步**清理后 `process.exit(1)`。

## 规则

- 能用 node/python 纯计算断言的，绝不开浏览器。
- 一次迭代只跑一次浏览器验收；视觉探针与验收分开计数。
- 阈值来自实测分布；改动断言阈值必须说明依据。
- 断言失败先查断言本身是否写错（E7 类判据缺陷已发生过）。
