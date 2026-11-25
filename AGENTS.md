# AGENTS.md

## 项目简介

**行为轨迹 3D 查看器**：把原始视频解析成标准 Schema 的动作 JSON，再渲染成等长的可运行 3D 行为轨迹（可导出视频）。
技术栈：前端 three.js r164（ESM + importmap）+ Python 离线提取管线 + Node 有界验收。
交付形态：**单文件 index.html**（源码在 `src/`，由 `tools/build_page.py` 内联；运行零构建、file:// 可直开）。
管线：`原始视频 → 多模态提取 → 动作 JSON（唯一契约）→ 3D 渲染 → 视频导出`。

## 项目目标

- **输入**：原始视频
- **输出**：同时长的可运行 3D 行为轨迹，可导出为视频
- **中间处理**：多模态模型 + 深度分析 + 线性分析，把视频解析成标准 Schema 的 JSON，再由渲染层驱动骨骼
- **参考源码**：`project-references/`（只读，可借鉴实现，不得引入其代码到生产路径）

### 四大难点（所有技术决策的出发点）

1. **多模态模型有边界**：能力必须**实测验证**，不可假设。用实测数字替代"应该可以"。
2. **2D → 3D 是核心难点**：深度是唯一缺失量。任何"从 2D 推 3D"的新逻辑必须先回答"深度从哪来"。
3. **视频信息杂乱**：Schema 必须承载**语义通道**（转身/侧倾/重心/接触/置信度），不能只堆像素点。
4. **准确性难校验**：断言必须量**渲染结果**，不能量解析式（自证无效）。

## 铁律

- 【铁律1】**深度不可臆测**。末端点的深度只能来自：透视缩短反解（`|Δz|=√(L²−d²)`）、偏航角给出的横向偏移、或解剖先验+帧间连续性的裁决。三者之外一律不猜。
- 【铁律2】**离线不做低通**。离线只做去离群（3/5 帧中值，保阶跃）；低通带滞后（实测 125ms），滞后比噪声更伤一致性。平滑只在运行时按目标做。
- 【铁律3】**断言量渲染结果**。禁止用解析式自证（相机反解与 videoBox 同源，比对必然通过）。阈值必须来自实测分布。
- 【铁律4】**启动类失败必须查 `ready` + 错误面板**。只查 `playhead`/`spec`/`url` 会让「启动失败」照样 PASS。
- 【铁律5】**每实装 1 个 task 写 1 篇 `docs/tasklist/TASKNNNN_*.md`；每检测到 1 个 bug 写 1 篇 `docs/buglist/BUGNNNN_*.md`**。先于修复写。编号单调递增，不复用。
- 【铁律6】**浏览器纪律**：单实例；每轮迭代最多 1 次有界验收；结束必 kill + `ps` 自查残留为 0；能用纯计算断言的绝不开浏览器。
- 【铁律7】**单一职责 + 整洁之道**：提取层不碰渲染决策，渲染层不重新检测；文档与代码只写结论，不写过程与版本信息。
- 【铁律8】**不敷衍**：禁止打补丁、捏造、臆测、留给下次。每个任务从粗到细分阶段完成，验收以断言为准。
- 【铁律9】**不可观测量要显式登记**。单目的固有限制（肢体深度、绕竖轴真实扭转、腾空高度、手指逐节）写进「已知不可观测」清单，不反复尝试修。
- 【铁律10】**多模态模型的能力边界必须实测，不得假设**。它的角色由 `docs/constraint/vlm_boundary.md` 的逐题准确率与噪声地板界定：不做度量估计（深度/角度/像素），只做序关系与粗分类；实测不可靠的通道（如掌向）不进渲染管线。
- 【铁律11】**测量代码不得污染生产状态**。临时播放动画/改写骨骼的代码必须完整还原（位置+旋转+缩放），否则症状会落在别的断言上。
- 【铁律12】**度量必须自检**。判据本身要被已知点检验（体内点量出正深度、体外点量出 ≤0）；度量出错时不会报错，只会安静地给出好看的数（求交符号写反 ⇒ 穿模量恒为 0 被读成「没有穿模」）。
- 【铁律13】**循环内只认自己声明的变量**。跨循环复用的变量名会让「用另一根骨的长度钳位」「只对最后一只脚限速」这类错误无声发生。
- 【铁律14】**失败先量后修，不许动阈值**。断言挂了先加诊断量出哪一步翻的，按根因修；把阈值放宽到刚好盖住残差 = 把判据迁就残差。
- 【铁律15】**检测器的输出是假设不是事实**。左右归属、低置信关节、遮挡处的位置都要先过修复/门控再进渲染；宁可显式登记为不可观测，不可静默当真。

## 架构与边界

- **中间产物是唯一契约**：渲染层只认动作 JSON，不认视频。换视频只改动作文件。
- **坐标系三套不可混用**：规格系（轨迹/机位）· 相机系（深度重构，x=画面右/y=上/z=远离相机）· 躯干系（驱动骨骼，+X=自身左/+Y=上/+Z=面朝）。换算见 @/docs/constraint/architecture.md。
- **动作文件向后兼容**：v1/v2 必须仍可加载；新字段只用新名字（`pose3d` 不用 `poses3d`），旧字段标注弃用。
- **规格语义变更**必须同步 README、示例与断言。
- 细节：@/docs/constraint/architecture.md

## 思维链

- **通用**：澄清目标 → 定位最小改动面 → 实现 → 语法检查/离线断言 → 视觉探针 → 有界验收 → 写 tasklist/buglist
- **新能力**：定契约 → 定验收判据 → 实现 → 离线断言 → 探针 → 验收 → 记录
- **缺陷**：复现 → 量化症状 → **最小根因**（症状与根因常不在一处）→ 修 → 回归断言 → 记录
- **姿态/算法改动**：先判定**可观测性** → 能观测走观测，不能走先验+连续性 → 纯计算离线验收 → 再进渲染层
- **换模型**：量比例与骨名 → 判断命名约定与手指骨 → 评估换装/几何修补链重做量 → 再决定
- 详情：@/docs/constraint/cots.md

## 命令

```bash
# 服务
python3 serve.py --port 8000

# 渲染层：改 src/ 后重建产物（index.html 是产物，不是源码）
tools/.venv/bin/python tools/build_page.py

# 质量门控（顺序即成本递增）
node tools/test_core.mjs                                   # 几何内核单测（秒级，不开浏览器）
node -e "const fs=require('fs'),s=fs.readFileSync('index.html','utf8'),a=s.indexOf('<script type=\"module\">'),b=s.indexOf('</script>',a);fs.writeFileSync('/tmp/_m.mjs',s.slice(a+22,b))" && node --check /tmp/_m.mjs
tools/.venv/bin/python tools/verify_depth.py
node tools/probe_render.mjs --view=overlay --times=0,4,7.21,10,12.5 --tag=x
node tools/verify.mjs

# 动作数据
tools/.venv/bin/python tools/extract_motion.py 从该原始视频中提取到的JSON.mp4 --spec 示例.json --backend mp
tools/.venv/bin/python tools/upgrade_motion.py 示例.motion.json

# 模型
node tools/glb_bones.mjs models/human.glb
node tools/glb_inspect.mjs models/human.glb
```

细节：@/docs/constraint/commands.md

## 代码规范

- 注释只写结论与边界，不写推理过程与原因说明。
- 角度单位：内部统一**弧度**，仅展示层转度。
- 长度单位：动作文件与运行时用**米**；Mixamo 骨骼局部单位是 100 单位=1m，必须换算。
- 命名：`camelCase`（JS/Python 同风格），常量 `UPPER_SNAKE`，模块级单例 `lowerCamel`。
- 缺失数据写 `null`，禁止用 `[0,0,0]` 或 0 代替。
- 异常不静默吞；失败要可见。

## Gotchas（动手前必读）

@/docs/constraint/gotchas.md —— 坐标系符号、Mixamo 量纲与骨链、四元数与姿态累乘、深度病态与时间滤波、验收盲区、浏览器纪律。

## 目录结构

```
index.html                渲染层**产物**（单文件；改代码改 src/，跑 tools/build_page.py 重建）
src/                      渲染层源码（core 纯几何内核 / app 应用层 / main 入口 / page 模板）
serve.py                  规格/媒体 API
示例.json / 示例.motion.json  规格 / 动作数据（v3）
models/                   人体模型与换装贴图
tools/                    提取·重构·验收·模型工具（py + mjs）
docs/constraint/          约束文档
docs/tasklist/            TASKNNNN_*.md
docs/buglist/             BUGNNNN_*.md
project-references/       外部参考源码（只读）
history/                  历史快照与回滚点
```
