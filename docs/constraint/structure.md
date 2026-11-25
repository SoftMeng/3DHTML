# Structure

```
3DHTML/
├── index.html                 渲染层产物（单文件，three.js r164 ESM + importmap，运行零构建）
├── src/                       渲染层源码
│   ├── core/                 纯几何内核（无 DOM/无 three.js/无全局状态，Node 可单测）
│   ├── app/                  应用层（state/scene/figure/solver/shell）
│   ├── main.js               入口（常量、three.js 载入、内核边界转换）
│   └── page.template.html    HTML+CSS 模板（含内联插槽）
├── serve.py                   规格/媒体 API + 静态服务（HTTP Range）
├── 示例.json                  规格（schema v1）
├── 示例.motion.json           动作数据（v3，含 pose3d）
├── 从该原始视频中提取到的JSON.mp4   源视频
├── models/                    human.glb · human_reshape.glb · human_recolor.png
├── tools/
│   ├── extract_motion.py      视频 → 动作数据（v1/v2/v3 打包；--pose-model 选检测档位）
│   ├── motion_pipeline.py     纯计算：置信度修复 → 逐帧归一化 → 去噪 → 语义通道
│   ├── pose_repair.py         纯计算：左右归属翻转修复（2 态 Viterbi）
│   ├── depth_reconstruct.py   纯计算：2D + 透视缩短 → 相机系三维骨架
│   ├── upgrade_motion.py      既有 v2 文件原地升级 v3（含翻转修复 + 重算深度）
│   ├── vlm.py                 多模态大模型客户端（凭据只读环境变量）
│   ├── probe_vlm.py           多模态能力基线测量（逐题准确率 + 噪声地板）
│   ├── vlm_annotate.py        多模态语义标注 → chan.palm / chan.semantic（v4）
│   ├── verify_vlm.py          轨迹一致性交叉校验（视频帧 vs 渲染帧，同题同答）
│   ├── verify_depth.py        深度重构离线验收（5 条断言）
│   ├── verify.mjs             有界验收（47 断言，单浏览器实例）
│   ├── probe_render.mjs       视觉探针（单实例，多时刻快照 + 关键量导出）
│   ├── reshape_model.py       按视频人物比例重塑 GLB
│   ├── recolor_texture.py     UV 岛 × 骨骼组 换装贴图
│   ├── trim_buns.py           几何修剪（发髻/护目镜）
│   ├── glb_bones.mjs          骨长与比例实测
│   ├── glb_inspect.mjs        GLB 结构清单
│   ├── probe_*.py / measure_*.py / overlay_check.py   一次性诊断探针（研究用，可重跑）
│   └── models/                离线模型缓存
├── docs/
│   ├── constraint/            约束文档（AGENTS.md 引用）
│   ├── tasklist/              TASKNNNN_*.md
│   └── buglist/               BUGNNNN_*.md
├── project-references/        外部参考源码（只读，不参与运行）
├── history/                   历史快照与回滚点
└── screenshots/               验收产出的截图
```

## 约定

- `index.html` 是渲染层唯一文件；新增能力优先加函数，不做结构大改。
- `tools/*.py` 是纯计算模块，不 import 渲染层；`tools/*.mjs` 是 node 脚本。
- 参考源码在 `project-references/`：**只读**，不得引入其代码到生产路径。
