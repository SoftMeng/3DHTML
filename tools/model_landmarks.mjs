/* 离线测量 Mixamo 人体模型的骨架 landmark 与真实几何包围盒（比例标定的事实来源）。
   用法: node tools/model_landmarks.mjs [models/human.glb]
   纯计算，不开浏览器。

   坐标系（已实测核验）：网格顶点坐标就是模型静止世界系（米，Y 向上）——
   蒙皮后的静止位置恒等于原始顶点（M(骨)·IBM = 单位阵），所以网格包围盒直接取
   POSITION 的 min/max，不能再乘节点的世界矩阵（Mixamo 导出把 mesh 挂在
   scale=0.01 的 Character 节点下，乘了会把范围算小 100 倍）。 */
import fs from 'fs';

const path = process.argv[2] || 'models/human.glb';
const buf = fs.readFileSync(path);
let off = 12, json = null, bin = null;
while (off < buf.length) {
  const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4), start = off + 8;
  if (type === 0x4e4f534a) json = JSON.parse(buf.slice(start, start + len).toString('utf8'));
  else if (type === 0x004e4942) bin = buf.slice(start, start + len);
  off = start + len;
}
const nodes = json.nodes;
const parent = new Array(nodes.length).fill(-1);
nodes.forEach((n, i) => (n.children || []).forEach(c => { parent[c] = i; }));

function nodeMatrix(n) {
  if (n.matrix) return n.matrix;
  const t = n.translation || [0, 0, 0], r = n.rotation || [0, 0, 0, 1], s = n.scale || [1, 1, 1];
  const [x, y, z, w] = r;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ];
}
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
const world = [];
function worldOf(i) {
  if (world[i]) return world[i];
  const p = parent[i];
  world[i] = p >= 0 ? mul(worldOf(p), nodeMatrix(nodes[i])) : nodeMatrix(nodes[i]);
  return world[i];
}
const pos = i => { const m = worldOf(i); return [m[12], m[13], m[14]]; };
const byName = {};
nodes.forEach((n, i) => { if (n.name) byName[n.name] = i; });
const P = n => { const i = byName[n]; return i === undefined ? null : pos(i); };
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];

/* 网格真实范围：直接取 POSITION 的 min/max（网格顶点即静止世界系坐标） */
function geomBox(file = path) {
  const acc = json.accessors[json.meshes[0].primitives[0].attributes.POSITION];
  const bv = json.bufferViews[acc.bufferView];
  const base = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const n = acc.count;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = bin.readFloatLE(base + i * 12 + k * 4);
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max, width: max[0] - min[0] };
}

export function measure(file = path) {
  const box = geomBox(file);
  const femurMid = mid(P('mixamorig:LeftUpLeg'), P('mixamorig:RightUpLeg'));
  const shMid = mid(P('mixamorig:LeftArm'), P('mixamorig:RightArm'));
  const ankleMid = mid(P('mixamorig:LeftFoot'), P('mixamorig:RightFoot'));
  const headBone = P('mixamorig:Head');
  const V = box.max[1] - box.min[1];                 // 可见身高 = 颅顶→鞋底
  const torso = shMid[1] - femurMid[1];
  const leg = femurMid[1] - box.min[1];
  const head = box.max[1] - shMid[1];
  return {
    visibleHeight: +V.toFixed(4),
    soleY: +box.min[1].toFixed(4),
    crownY: +box.max[1].toFixed(4),
    hipY: +femurMid[1].toFixed(4),
    shoulderY: +shMid[1].toFixed(4),
    ankleY: +ankleMid[1].toFixed(4),
    headBoneY: +headBone[1].toFixed(4),
    width: +box.width.toFixed(4),
    /* 比例分数：占可见身高的百分比（源视频人物：腿 50.7 / 躯干 27.6 / 头颈 21.7） */
    fracLeg: +(leg / V).toFixed(4),
    fracTorso: +(torso / V).toFixed(4),
    fracHead: +(head / V).toFixed(4),
    fracHip: +((femurMid[1] - box.min[1]) / V).toFixed(4),
    fracShoulder: +((shMid[1] - box.min[1]) / V).toFixed(4),
    boneLen: {
      thigh: +dist(P('mixamorig:LeftUpLeg'), P('mixamorig:LeftLeg')).toFixed(4),
      shin: +dist(P('mixamorig:LeftLeg'), P('mixamorig:LeftFoot')).toFixed(4),
      armUpper: +dist(P('mixamorig:LeftArm'), P('mixamorig:LeftForeArm')).toFixed(4),
      armFore: +dist(P('mixamorig:LeftForeArm'), P('mixamorig:LeftHand')).toFixed(4),
    },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  console.log(JSON.stringify(measure(), null, 1));
}
