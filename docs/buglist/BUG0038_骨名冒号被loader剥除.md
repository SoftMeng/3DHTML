# BUG0038 three.js 的 GLTFLoader 会剥掉节点名里的冒号 → 骨名查找全部落空

- 现象：碰撞代理的半径全部缺失（`proxyR` 只有袖筒两项），`rig.palmAxis` 为空 → 穿模判定与掌向都不生效，且推出量异常。
- 根因：GLB 里的骨名是 `mixamorig:LeftHand`，但 three.js 的 GLTFLoader 用 `PropertyBinding.sanitizeNodeName` **删除**保留字符 → 运行时骨名是 `mixamorigLeftHand`。按原名查找必然落空。
- 修法：统一用规范化名查找（`findBoneByName`：小写 + 去掉非字母数字），与项目里已有的 `norm()` 约定一致。
- 影响面：`measureProxyRadii` / `measurePalmAxes` / `bodyTriangles` / `armVertices` 的全部骨名索引。

- 回归：探针 `__app.proxyInfo()` 的 `radii` 九项齐全、`caps` 非空
