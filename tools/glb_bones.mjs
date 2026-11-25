import fs from 'fs';

const buf = fs.readFileSync(process.argv[2] || 'models/human.glb');
if (buf.readUInt32LE(0) !== 0x46546c67) throw new Error('not glb');
let off = 12, json = null, bin = null;
while (off < buf.length) {
  const len = buf.readUInt32LE(off), type = buf.readUInt32LE(off + 4);
  const start = off + 8;
  if (type === 0x4e4f534a) json = JSON.parse(buf.slice(start, start + len).toString('utf8'));
  else if (type === 0x004e4942) bin = buf.slice(start, start + len);
  off = start + len;
}
const nodes = json.nodes;
const parent = new Array(nodes.length).fill(-1);
nodes.forEach((n, i) => (n.children || []).forEach(c => { parent[c] = i; }));
const idxOf = {}; json.nodes.forEach((n, i) => { if (n.name) idxOf[n.name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()] = i; });

// world transform (node hierarchy) — TRS with column-major matrices
function nodeMatrix(n) {
  if (n.matrix) return n.matrix;                     // 16 floats col-major
  const t = n.translation || [0, 0, 0], r = n.rotation || [0, 0, 0, 1], s = n.scale || [1, 1, 1];
  const [x, y, z, w] = r;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1
  ];
}
function mul(a, b) {                 // a*b, column-major
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
const world = [];
function worldOf(i) {
  if (world[i]) return world[i];
  const p = parent[i];
  const local = nodeMatrix(nodes[i]);
  world[i] = p >= 0 ? mul(worldOf(p), local) : local;
  return world[i];
}
const pos = i => { const m = worldOf(i); return [m[12], m[13], m[14]]; };
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

const B = n => {                       /* 兼容两种命名：mixamorig:LeftArm 与 LeftArm */
  const k = n.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
  if (idxOf[k] !== undefined) return idxOf[k];
  const bare = k.replace(/^mixamorig/, '');
  return idxOf[bare];
};
const hips = B('mixamorig:Hips'), head = B('mixamorig:Head');
const lArm = B('mixamorig:LeftArm'), lFore = B('mixamorig:LeftForeArm'), lHand = B('mixamorig:LeftHand');
const rArm = B('mixamorig:RightArm'), rFore = B('mixamorig:RightForeArm'), rHand = B('mixamorig:RightHand');
const lUL = B('mixamorig:LeftUpLeg'), lLeg = B('mixamorig:LeftLeg'), lFoot = B('mixamorig:LeftFoot'), lToe = B('mixamorig:LeftToeBase');
const rUL = B('mixamorig:RightUpLeg'), rLeg = B('mixamorig:RightLeg'), rFoot = B('mixamorig:RightFoot'), rToe = B('mixamorig:RightToeBase');

// mesh bbox (accessor min/max) to get model height like the viewer does
let minY = Infinity, maxY = -Infinity;
for (const m of json.meshes || []) for (const p of m.primitives) {
  const a = json.accessors[p.attributes.POSITION];
  if (!a || !a.min || !a.max) continue;
  minY = Math.min(minY, a.min[1]); maxY = Math.max(maxY, a.max[1]);
}
const modelH = maxY - minY;
const shMid = [(pos(lArm)[0] + pos(rArm)[0]) / 2, (pos(lArm)[1] + pos(rArm)[1]) / 2, (pos(lArm)[2] + pos(rArm)[2]) / 2];
const out = {
  modelHeightRaw: +modelH.toFixed(4),
  torso: +dist(shMid, pos(hips)).toFixed(4),
  armUpper: +dist(pos(lArm), pos(lFore)).toFixed(4),
  armFore: +dist(pos(lFore), pos(lHand)).toFixed(4),
  legThigh: +dist(pos(lUL), pos(lLeg)).toFixed(4),
  legShin: +dist(pos(lLeg), pos(lFoot)).toFixed(4),
  legShinToe: +dist(pos(lFoot), pos(lToe)).toFixed(4),
  hipsY: +pos(hips)[1].toFixed(4), headY: +pos(head)[1].toFixed(4),
  shoulderY: +shMid[1].toFixed(4),
};
out.armTotal = +(out.armUpper + out.armFore).toFixed(4);
out.legTotal = +(out.legThigh + out.legShin).toFixed(4);
out.torsoFrac = +(out.torso / modelH).toFixed(4);
out.armFrac = +(out.armTotal / modelH).toFixed(4);
out.legFrac = +(out.legTotal / modelH).toFixed(4);
console.log(JSON.stringify(out, null, 1));
