#!/usr/bin/env node
/* 几何内核单元测试（纯计算，不开浏览器）。
   用法：node tools/test_core.mjs

   每条断言对应一类真实历史缺陷：
   · rayTri/intervalDepth —— BUG0044（求交符号写反 ⇒ 度量恒 0）、BUG0049（掠面距离符号）
   · solveTwoBone        —— 余弦定理与可达边界的自洽
   · transportPole       —— BUG0052（绝对提示点泄漏 ⇒ 弯折面翻转）
   · bendOffset          —— E7 同量纲的符号
   · chainTruthPen       —— BUG0050（半径符号 + 可见几何） */
import test from 'node:test';
import assert from 'node:assert/strict';
import { Vec3, Quat } from '../src/core/vec.js';
import { rayTri, rayHitsBox, intervalDepth } from '../src/core/geom.js';
import { solveTwoBone, transportPole, bendOffset } from '../src/core/ik.js';
import { truthPen } from '../src/core/clip.js';

const EPS = 1e-9;

/* 轴对齐盒的身体网格（8 顶点 / 12 三角） */
function boxBody(cx = 0, cy = 0, cz = 0, hx = 0.2, hy = 0.3, hz = 0.15){
  const v = [];
  for(const s of [-1, 1]) for(const t of [-1, 1]) for(const u of [-1, 1]) v.push(cx + s * hx, cy + t * hy, cz + u * hz);
  /* 顶点序：i = s*4 + t*2 + u */
  const q = [
    [0, 2, 6], [0, 6, 4],   /* -x 面 */
    [1, 5, 7], [1, 7, 3],   /* +x 面 */
    [0, 4, 5], [0, 5, 1],   /* -y 面 */
    [2, 3, 7], [2, 7, 6],   /* +y 面 */
    [0, 1, 3], [0, 3, 2],   /* -z 面 */
    [4, 6, 7], [4, 7, 5],   /* +z 面 */
  ];
  const tri = [];
  for(const f of q) tri.push(f[0], f[1], f[2]);
  return {
    w: new Float32Array(v),
    groups: {torso: {tri, bb: [cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz]}},
  };
}

test('vec: 与 three.js 同款的最小旋转（setFromUnitVectors 同式）', () => {
  const a = new Vec3(1, 0, 0), b = new Vec3(0, 0, 1);
  const q = Quat.fromUnitVectors(a, b);
  const r = a.clone().applyQuat(q);
  assert.ok(Math.abs(r.x - b.x) < 1e-9 && Math.abs(r.y - b.y) < 1e-9 && Math.abs(r.z - b.z) < 1e-9,
            '单位向量应被旋转到目标向量');
  const anti = new Vec3(1, 0, 0), antiT = new Vec3(-1, 0, 0);
  const q2 = Quat.fromUnitVectors(anti, antiT);
  const r2 = anti.clone().applyQuat(q2);
  assert.ok(Math.abs(r2.x + 1) < 1e-6, '反向情形不得产生 NaN');
});

test('rayTri: 三角形内点命中且命中点重合；退化/出界/平行不误命中', () => {
  const o = [0, 0, 1], d = [0, 0, -1];
  const t = rayTri(o[0], o[1], o[2], d[0], d[1], d[2], -1, -1, 0, 1, -1, 0, 0, 1, 0);
  assert.ok(t !== null && Math.abs(t - 1) < EPS, `命中参数应为 1，实际 ${t}`);
  assert.equal(rayTri(0, 0, 1, 0, 0, -1, -1, -1, 0, 0, -1, 0, 1, -1, 0), null, '共线（退化）三角形不命中');
  assert.equal(rayTri(2, 0, 1, 0, 0, -1, -1, -1, 0, 1, -1, 0, 0, 1, 0), null, '投影在三角形外不命中');
  assert.equal(rayTri(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1), null, '平行射线不命中');
});

test('rayHitsBox: 命中与未命中', () => {
  const bb = [-0.2, -0.3, -0.15, 0.2, 0.3, 0.15];
  assert.equal(rayHitsBox([0, 0, 2], [0, 0, -1], bb), true, '正对盒体应命中');
  assert.equal(rayHitsBox([1.5, 0, 2], [0, 0, -1], bb), false, '偏离盒体不应命中');
});

test('intervalDepth: 体内点为正、四个体外点非正（E19 的内核版）', () => {
  const body = boxBody();
  const cam = new Vec3(0, 0, 2);
  const inside = intervalDepth(new Vec3(0, 0, 0), body, cam);
  assert.ok(inside > 0, `体内点应为正深度，实际 ${inside}`);
  assert.ok(Math.abs(inside - 0.15) < 1e-6, `体内深度应等于到较近表面的距离，实际 ${inside}`);
  for(const [name, p] of [['身前 30cm', new Vec3(0, 0, 0.45)],
                          ['身后 30cm', new Vec3(0, 0, -0.45)],
                          ['侧方 35cm', new Vec3(0.35, 0, 0)],
                          ['上方 3m', new Vec3(0, 3, 0)]]){
    const d = intervalDepth(p, body, cam);
    assert.ok(d <= 0, `${name} 应判为体外，实际 ${d}`);
  }
});

test('intervalDepth: 沿视线推进时符号连续（抓「掠面被读成深入体内」这类符号错）', () => {
  const body = boxBody();
  const cam = new Vec3(0, 0, 2);
  let prev = intervalDepth(new Vec3(0, 0, 0.6), body, cam);
  assert.ok(prev <= 0, '起点在体外');
  for(let z = 0.6; z >= -0.6; z -= 0.02){
    const d = intervalDepth(new Vec3(0, 0, z), body, cam);
    assert.ok(d <= 0 || d >= 0, '深度必须是有限数');
    if(Math.abs(d) > 1e-6) assert.ok(Math.sign(d) === Math.sign(prev) || Math.abs(d) < 0.2,
      `z=${z.toFixed(2)} 处符号突变：${prev} → ${d}`);
    prev = d;
  }
  /* 体内段必须全为正，体外段必须全为负 */
  for(let z = -0.1; z <= 0.1; z += 0.02) assert.ok(intervalDepth(new Vec3(0, 0, z), body, cam) > 0, `z=${z} 应在体内`);
  assert.ok(intervalDepth(new Vec3(0, 0, 0.4), body, cam) < 0, 'z=0.4 应在体外');
  assert.ok(intervalDepth(new Vec3(0, 0, -0.4), body, cam) < 0, 'z=-0.4 应在体外');
});

test('solveTwoBone: 骨长守恒、极向量定侧、越界夹取且无 NaN', () => {
  const root = new Vec3(0, 0, 0), target = new Vec3(0.3, 0, 0);
  const L1 = 0.2, L2 = 0.2;
  const pole = new Vec3(0.15, 0.13, 0);
  const s = solveTwoBone(root, target, L1, L2, pole);
  assert.ok(s, '应给出解');
  assert.ok(Math.abs(s.elbow.distanceTo(root) - L1) < 1e-9, '上段骨长必须守恒');
  assert.ok(Math.abs(s.target.distanceTo(s.elbow) - L2) < 1e-9, '下段骨长必须守恒');
  assert.ok(s.elbow.y > 0, '极向量决定肘在弦的哪一侧');
  const s2 = solveTwoBone(root, target, L1, L2, new Vec3(0.15, -0.13, 0));
  assert.ok(s2.elbow.y < 0, '反向极向量给出反向的肘');
  const far = solveTwoBone(root, new Vec3(1, 0, 0), L1, L2, pole);
  assert.ok(far && Number.isFinite(far.elbow.x), '越界目标不得产生 NaN');
  assert.ok(Math.abs(far.target.distanceTo(root) - (L1 + L2) * 0.995) < 1e-9, '越界目标被夹到可达边界');
  assert.equal(solveTwoBone(root, new Vec3(0, 0, 0), L1, L2, pole), null, '退化目标返回 null');
});

test('transportPole: 弯折面随弦刚体搬运（构型整体旋转、侧向不翻）', () => {
  const root = new Vec3(0, 0, 0);
  const hand = new Vec3(0.4, 0, 0);
  const elbow = new Vec3(0.15, Math.sqrt(0.2 * 0.2 - 0.15 * 0.15), 0);
  const L1 = elbow.distanceTo(root), L2 = hand.distanceTo(elbow);
  /* 新弦：旧弦绕 z 轴转 45° */
  const ang = Math.PI / 4;
  const dirNew = new Vec3(Math.cos(ang), Math.sin(ang), 0);
  const newTarget = root.clone().addScaledVector(dirNew, 0.4);
  const pole = transportPole(root, elbow, hand, newTarget);
  assert.ok(pole, '应给出搬运后的极向量');
  const perpNew = pole.clone().sub(root);
  assert.ok(Math.abs(perpNew.dot(dirNew)) < 1e-9, '搬运后的偏置方向必须垂直于新弦');
  /* 整个弯折构型随弦刚体旋转：解出的新肘应等于旧肘被同一旋转搬过去 */
  const rotated = elbow.clone().sub(root).applyQuat(Quat.axisAngle(new Vec3(0, 0, 1), ang)).add(root);
  const solved = solveTwoBone(root, newTarget, L1, L2, pole);
  assert.ok(solved.elbow.distanceTo(rotated) < 1e-9,
            `搬运后构型必须整体旋转：${solved.elbow} vs ${rotated}`);
  /* 关键性质：弦转 90° 以内时，弯折侧向（前向分量）不翻面 */
  const fwd = new Vec3(0, 1, 0);
  const before = bendOffset(root, elbow, hand, fwd);
  const after = bendOffset(root, solved.elbow, newTarget, fwd);
  assert.ok(before > 0 && after > 0, `侧向不应翻转：${before} → ${after}`);
});

test('bendOffset: 符号正确、弦退化返回 null', () => {
  const root = new Vec3(0, 0, 0);
  const end = new Vec3(1, 0, 0);
  const fwd = new Vec3(0, 0, 1);
  const behind = bendOffset(root, new Vec3(0.5, 0, -0.2), end, fwd);
  const front = bendOffset(root, new Vec3(0.5, 0, 0.2), end, fwd);
  assert.ok(behind < 0 && front > 0, '前向分量的符号必须相反');
  assert.equal(bendOffset(root, new Vec3(0, 0, 0), root, fwd), null, '弦长为零返回 null');
});

test('chainTruthPen: 可见几何的表面穿透（含半径）；不可见采样不参与', () => {
  const body = boxBody();
  const cam = new Vec3(0, 0, 2);
  const inside = truthPen([{p: new Vec3(0, 0, 0), r: 0.01, vis: true}], body, cam);
  assert.ok(inside.pen > 0, '体内采样应报穿透');
  assert.ok(Math.abs(inside.pen - (0.15 + 0.01)) < 1e-6, `表面穿透 = 中心深度 + 半径，实际 ${inside.pen}`);
  const outside = truthPen([{p: new Vec3(0, 0, 0.5), r: 0.01, vis: true}], body, cam);
  assert.ok(outside.pen <= 0, '体外采样不应报穿透');
  const hidden = truthPen([{p: new Vec3(0, 0, 0), r: 0.01, vis: false}], body, cam);
  assert.equal(hidden.pen, 0, '不可见几何不参与判据');
  assert.equal(truthPen([], body, cam).pen, 0, '空采样为 0');
});
