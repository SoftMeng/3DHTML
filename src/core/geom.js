/* 几何内核：射线求交与「视线穿透区间」内外判定。
   零依赖（无 DOM / 无 three.js / 无全局状态）⇒ 可被 Node 直接 import 做单测。

   视线穿透区间：点相对躯干网格的**真实**穿模深度（米；正 = 在体内，负 = 在体外，
   量级 = 到较近表面的距离）。
   内外判定既不依赖三角形法向（Mixamo 躯干三角形法向朝内，用法向定符号会全判反 ——
   见 BUG0041），也不依赖奇偶判定（网格非封闭）：相机在体外 ⇒ 沿视线方向第一个交点
   是前表面、最后一个是后表面，点落在两者之间即在体内。 */
import { Vec3 } from './vec.js';

/* Möller–Trumbore（标量式）：返回射线参数 t（>0），未命中返回 null */
export function rayTri(ox, oy, oz, dx, dy, dz, ax, ay, az, bx, by, bz, cx, cy, cz){
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if(Math.abs(det) < 1e-12) return null;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if(u < 0 || u > 1) return null;
  const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if(v < 0 || u + v > 1) return null;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > 1e-6 ? t : null;
}

/* 射线是否命中轴对齐盒（bb = [minx,miny,minz,maxx,maxy,maxz]） */
export function rayHitsBox(o, d, bb){
  let t0 = 0, t1 = Infinity;
  for(let k = 0; k < 3; k++){
    const inv = 1 / (d[k] || 1e-12);
    let a = (bb[k] - o[k]) * inv, b = (bb[k + 3] - o[k]) * inv;
    if(a > b){ const t = a; a = b; b = t; }
    if(a > t0) t0 = a;
    if(b < t1) t1 = b;
    if(t0 > t1) return false;
  }
  return true;
}

/* 点相对身体网格的穿模深度。
   body = {w: Float32Array(顶点), groups: {部位: {tri: [顶点索引×3…], bb: [min…,max…]}}}
   diag 非空时回填诊断（各部位的前后交点、命中数）。 */
export function intervalDepth(p, body, camPos, diag){
  const dx = p.x - camPos.x, dy = p.y - camPos.y, dz = p.z - camPos.z;
  const tp = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if(tp < 1e-6) return 0;
  const ix = dx / tp, iy = dy / tp, iz = dz / tp;
  let hits = 0;
  const w = body.w;
  const o = [camPos.x, camPos.y, camPos.z], d = [ix, iy, iz];
  /* 每个部位各自是一个**实体**：两腿之间的空隙不属于任何一个部位，
     若把所有部位合成一个区间，空隙会被判成"在体内"（实测误报 9cm）。 */
  let inside = -Infinity, outside = -Infinity;
  for(const k in body.groups){
    const G = body.groups[k];
    if(G.bb && !rayHitsBox(o, d, G.bb)) continue;
    const tri = G.tri;
    let t0 = Infinity, t1 = -Infinity;
    for(let i = 0; i < tri.length; i += 3){
      const a = tri[i] * 3, b = tri[i + 1] * 3, c = tri[i + 2] * 3;
      const t = rayTri(o[0], o[1], o[2], d[0], d[1], d[2],
                       w[a], w[a + 1], w[a + 2], w[b], w[b + 1], w[b + 2], w[c], w[c + 1], w[c + 2]);
      /* 近乎平行的三角形会让 t 爆到天上：只接受点附近合理窗口内的交点 */
      if(t === null || t > tp + 5) continue;
      hits++;
      if(t < t0) t0 = t;
      if(t > t1) t1 = t;
    }
    if(diag){ diag.g = diag.g || {}; diag.g[k] = {t0: t0 === Infinity ? null : +t0.toFixed(4), t1: t1 === -Infinity ? null : +t1.toFixed(4)}; }
    if(t0 === Infinity) continue;               /* 射线没碰到这个部位 */
    /* 只擦到一层表面（射线掠过网格边缘/开口）：点必在体外。
       距离取绝对值 —— 原式 -(tp-t0) 在 tp<t0 时是正值，会把"擦过"判成"深入体内"。 */
    if(t1 - t0 < 1e-6){ outside = Math.max(outside, -Math.abs(tp - t0)); continue; }
    if(tp <= t0){ outside = Math.max(outside, -(t0 - tp)); continue; }   /* 在前表面之前 */
    if(tp >= t1){ outside = Math.max(outside, -(tp - t1)); continue; }   /* 在后表面之后 */
    if(diag){ diag.in = k; }
    inside = Math.max(inside, Math.min(tp - t0, t1 - tp));              /* 体内 */
  }
  if(diag){ diag.tp = tp; diag.hits = hits; }
  if(inside > -Infinity) return inside;        /* 在某个部位内 */
  if(outside > -Infinity) return outside;      /* 在所有部位外 */
  return 0;                                     /* 射线没碰到身体 → 在体外 */
}

/* 点到射线的最近距离（用于把点投到视线轴上） */
export function closestOnRay(o, d, p){
  const t = p.clone().sub(o).dot(d);
  return o.addScaledVector(d, t);
}

export { Vec3 };
