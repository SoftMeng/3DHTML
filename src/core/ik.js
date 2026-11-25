/* 几何内核：2 骨 IK 的几何解、弯折面随弦搬运、弯折侧向偏移。
   零依赖（无 DOM / 无 three.js / 无全局状态）⇒ 可被 Node 直接 import 做单测。

   2 骨 IK（余弦定理）：给定根点、末端点、两骨长与极向量（pole），
   求出中间关节（肘/膝）的位置与两根骨的方向。
   极向量给的是**弯折面**（肘在弦的哪一侧）——它的自由度不可观测，
   绝对提示点会在弦被转动后泄漏（BUG0052），所以只搬运方向。 */
import { Vec3, Quat } from './vec.js';

/* 两骨 IK 几何解。
   root      根点（肩/髋）
   target    末端点（手/踝）目标
   L1,L2     两骨长
   pole      极向量提示点（肘的期望位置）；缺省时用 bendAxis 兜底
   bendAxis  兜底弯折面法向（弦平面内的旋转轴）
   tilt      绕弦的额外旋转（让位用）
   返回 null 表示目标与根重合（退化）。 */
export function solveTwoBone(root, target, L1, L2, pole, bendAxis, tilt){
  let d = target.clone().sub(root);
  let r = d.length();
  if(r < 1e-5) return null;
  const maxR = (L1 + L2) * 0.995;
  if(r > maxR){ d = d.scale(maxR / r); r = maxR; }
  const dir = d.normalize();
  const a = (r * r + L1 * L1 - L2 * L2) / (2 * r);       /* 弦足到根的距离 */
  const h = Math.sqrt(Math.max(0, L1 * L1 - a * a));      /* 肘到弦的垂距 */
  let n = null;
  if(pole){
    const chord = root.clone().addScaledVector(dir, a);
    n = pole.clone().sub(chord);
    n = n.addScaledVector(dir, -n.dot(dir));
    if(n.lengthSq() < 1e-10) n = null;                   /* 提示点在弦上：方位不可观测 */
  }
  if(!n){
    n = bendAxis ? bendAxis.clone() : new Vec3(0, 0, 1).cross(dir);
    n = n.addScaledVector(dir, -n.dot(dir));
    if(n.lengthSq() < 1e-8) n = new Vec3(0, 0, 1).cross(dir);
    if(n.lengthSq() < 1e-8) n = new Vec3(1, 0, 0);
  }
  n = n.normalize();
  if(tilt) n = n.applyQuat(Quat.axisAngle(dir, tilt));
  const clamped = root.clone().addScaledVector(dir, r);
  const elbow = root.clone().addScaledVector(dir, a).addScaledVector(n, h);
  return {
    root, target: clamped, elbow, dir, n, a, h,
    dirU: elbow.clone().sub(root).normalize(),
    dirF: clamped.clone().sub(elbow).normalize(),
  };
}

/* 弯折面随弦搬运（平行移动）：把「弯折点相对弦的偏置方向」随弦刚体搬运，
   偏置方向与弦保持原夹角 ⇒ 弦被转动后弯折点仍在同一侧（BUG0052）。
   root/elbow/hand 为当前构型；newTarget 为新的末端目标。
   返回新的极向量提示点（与 oldPole 同量级），退化时返回 null。 */
export function transportPole(root, elbow, hand, newTarget){
  let dOld = hand.clone().sub(root);
  const l0 = dOld.length();
  if(l0 < 1e-5) return null;
  dOld = dOld.scale(1 / l0);
  const e = elbow.clone().sub(root);
  const perp = e.addScaledVector(dOld, -e.dot(dOld));
  if(perp.lengthSq() < 1e-10) return null;               /* 弯折点在弦上：方位不可观测 */
  let dNew = newTarget.clone().sub(root);
  const l1 = dNew.length();
  if(l1 < 1e-5) return null;
  dNew = dNew.scale(1 / l1);
  const q = Quat.fromUnitVectors(dOld, dNew);
  const n = perp.normalize().applyQuat(q);               /* 刚体搬运：n ⟂ dNew */
  return root.clone().addScaledVector(n, 0.05);
}

/* 弯折侧向偏移：弯折点相对「根→末端」弦线在 fwd（身体前向）上的投影。
   正 = 弯折点在弦的前侧（手臂应为负 ⇒ 肘在弦后；腿应为正 ⇒ 膝在弦前）。 */
export function bendOffset(root, mid, end, fwd){
  let d = end.clone().sub(root);
  const len = d.length();
  if(len < 1e-6) return null;
  d = d.scale(1 / len);
  const e = mid.clone().sub(root);
  const perp = e.addScaledVector(d, -e.dot(d));
  return perp.dot(fwd);
}
