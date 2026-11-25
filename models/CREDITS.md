# 人体模型来源与署名

`human.glb` = **Michelle.glb**，取自 three.js 官方示例仓库：

- 来源：https://github.com/mrdoob/three.js/tree/r164/examples/models/gltf/Michelle.glb
- 原始角色：Mixamo（Adobe）角色 "Michelle"，骨骼为标准 Mixamo 命名（`mixamorig:*`）
- 许可：three.js 仓库为 MIT；Mixamo 角色可免费用于个人与商业项目（需具备 Mixamo 账号获得原始资源），本项目仅做本地查看用途
- 规格：3.13 MB / 28106 三角面 / 65 关节 / 4 张贴图

## 换装贴图 `human_recolor.png`

由 `tools/recolor_texture.py` 从 `human.glb` 内嵌贴图生成（不改动原 GLB）：按 **UV 岛**判定衣物归属后整岛换色，
使配色与源视频表演者一致（深色上衣 + 浅蓝牛仔 + 白鞋 + 浅肤色），并抹掉角色自带的运动护目镜。
生成命令：`tools/.venv/bin/python tools/recolor_texture.py models/human.glb models/human_recolor.png`

## 离线模型缓存 `tools/models/`

| 文件 | 来源 | 许可 |
|---|---|---|
| `movenet_singlepose_lightning.onnx` | https://huggingface.co/Xenova/movenet-singlepose-lightning | Apache-2.0 |
| `pose_landmarker_lite.task` | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task | Apache-2.0 |

## 参考资源 `project-references/`

外部下载的对比用模型（只读，不参与运行）：three.js 示例的 `Xbot.glb` / `Soldier.glb`（MIT 仓库内，角色本体为 Mixamo）。

如需替换为其它角色：保持 Mixamo 骨骼命名（`mixamorig:Hips / Spine2 / Head / LeftShoulder / LeftArm / LeftForeArm / LeftHand / LeftUpLeg / LeftLeg / LeftFoot / LeftToeBase` 及右侧同名）即可被查看器直接驱动；`index.html` 中的 `BONE_MAP` / `MODEL_URL` 是唯一的耦合点。
