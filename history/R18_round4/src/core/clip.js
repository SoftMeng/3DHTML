/* 几何内核：可见手部几何相对躯干网格的穿透量。
   零依赖（无 DOM / 无 three.js / 无全局状态）⇒ 可被 Node 直接 import 做单测。

   判据只看**可见**的手部几何（裸露的手指）：掌背藏在卫衣里、袖筒与躯干是同一件衣服，
   它们的互穿在画面上不可见，不构成可见穿模。
   表面穿透 = 中心在体内的深度 **加上** 该处半径（朝身体方向的偏移是**更深**）。 */
import { intervalDepth } from './geom.js';

/* samples: [{p: Vec3, r: 半径, vis: 是否可见几何}]
   body/camPos 与 intervalDepth 同义。返回 {pen 最深穿透, idx 最深采样点, sum 穿透总和, center 最深中心深度}。 */
export function truthPen(samples, body, camPos){
  let m = 0, worstIdx = -1, sum = 0, center = 0;
  for(let i = 0; i < samples.length; i++){
    const s = samples[i];
    if(s.vis === false) continue;                 /* 只有手指是可见的手部几何 */
    const d = intervalDepth(s.p, body, camPos);
    if(d > center) center = d;
    const surf = d + s.r;
    if(surf > 0) sum += surf;
    if(surf > m){ m = surf; worstIdx = i; }
  }
  return {pen: m, idx: worstIdx, sum, center};
}
