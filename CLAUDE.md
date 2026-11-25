@AGENTS.md

# Claude Code 专属补充

## 会话起步

1. 读 `AGENTS.md`（已由本文件引入）与 `docs/constraint/gotchas.md`。
2. 查 `docs/buglist/` 是否有同类根因的历史记录；查 `docs/tasklist/` 了解已实装能力与边界。
3. 需要外部资料时用 `agent-reach` skill；需要同类实现时查 `project-references/`（只读）。

## 执行顺序（Claude Code 适配）

- 改 `index.html` 后**先**跑 `node --check`（语法），再跑纯计算断言，最后才是浏览器。
- 浏览器每轮只跑一次有界验收；要看画面用 `tools/probe_render.mjs`（单实例、一次多时刻）。
- 不要为了"看一眼"反复启浏览器——用户机器曾因多实例无头 Chrome 被打满。

## 记录义务（Claude Code 强制）

- 实装 task → `docs/tasklist/TASKNNNN_*.md`（下一个可用编号）。
- 发现 bug → `docs/buglist/BUGNNNN_*.md`（先于修复）。
- 两者与代码**同一次提交**落盘；只写结论，不写过程。

## 交互约定

- 需要决策点（换模型、换管线、改 Schema、抬阈值）时先给选项与取舍，不擅自扩大范围。
- 报告结果用实测数字（偏差、断言数、耗时），不用"应该没问题"。
- 改动完成后说明改了哪些文件、跑了哪些门禁、结果如何；未跑的必须说明。
