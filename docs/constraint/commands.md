# Commands

## 环境

- 依赖：`tools/.venv`（mediapipe / opencv / numpy / onnxruntime）
- 模型缓存：`tools/models/`（`pose_landmarker_lite.task`、`hand_landmarker.task`、`movenet_singlepose_lightning.onnx`）

## 服务

- 启动：`python3 serve.py --port 8000` → http://127.0.0.1:8000/
- 停止：`pkill -f "serve.py --port 8000"`

## 质量

- 重建渲染层产物（改 `src/` 后必跑；`index.html` 是产物）：
  `tools/.venv/bin/python tools/build_page.py`
  校验产物与源码同步（也在验收里查）：`tools/.venv/bin/python tools/build_page.py --check`
- 几何内核单测（秒级，纯计算，不开浏览器）：
  `node tools/test_core.mjs`
- 验收（**每轮迭代仅一次**，单浏览器实例，≤120s）：
  `node tools/verify.mjs`
- 深度重构离线验收（秒级，纯计算，不开浏览器）：
  `tools/.venv/bin/python tools/verify_depth.py`
- 视觉探针（单实例，一次多时刻快照 + 关键量导出）：
  `node tools/probe_render.mjs --view=overlay --times=0,4,7.21,10,12.5 --tag=x`
- 语法检查（改完 index.html 必跑）：
  `node -e "const fs=require('fs'),s=fs.readFileSync('index.html','utf8'),a=s.indexOf('<script type=\"module\">'),b=s.indexOf('</script>',a);fs.writeFileSync('/tmp/_m.mjs',s.slice(a+22,b))" && node --check /tmp/_m.mjs`

## 动作数据

- 全量重提取：`tools/.venv/bin/python tools/extract_motion.py 从该原始视频中提取到的JSON.mp4 --spec 示例.json --backend mp`
- v2 → v3 升级（附加 pose3d，不重跑提取）：`tools/.venv/bin/python tools/upgrade_motion.py 示例.motion.json`
- 换视频后必须复核断言阈值（v1→v2 会改人体比例基线）

## 模型

- 骨长与比例实测：`node tools/glb_bones.mjs models/human.glb`
- 节点/蒙皮/动画清单：`node tools/glb_inspect.mjs <glb>`
- 比例重塑：`tools/.venv/bin/python tools/reshape_model.py models/human.glb models/human_reshape.glb`
- 换装贴图：`tools/.venv/bin/python tools/recolor_texture.py models/human.glb models/human_recolor.png`
- 几何修剪：`tools/.venv/bin/python tools/trim_buns.py models/human.glb`

## 规则

- 完成改动前执行语法检查 + 相关离线断言；浏览器验收每轮仅一次。
- 未执行不得声称通过。
- 优先纯计算断言，避免无必要的全量验收。
