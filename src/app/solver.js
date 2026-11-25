import {buildHumanFigure, rig} from './figure.js';
import {DEG, clamp, lerp, samplePath, toThree, toVec} from '../main.js';
import {rigHipJointPoint, rigHipJointY, updateScene} from './shell.js';
import {THREE, groups, motion, motionTargets, motionTargets3, playhead, spec} from './state.js';

/* ---------- 模仿驱动：动作文件 → 骨骼 IK ---------- */
export const MKEY = {nose:0, leye:1, reye:2, lear:3, rear:4, lsh:5, rsh:6, lel:7, rel:8, lwr:9, rwr:10,
              lhp:11, rhp:12, lkn:13, rkn:14, lan:15, ran:16};

/* ---------- One-Euro 自适应滤波：静止时强平滑（去抖），快速动作时低延迟 ---------- */
class OneEuro{
  constructor(minCutoff=1.2, beta=0.02, dCutoff=1.0){ this.mc=minCutoff; this.beta=beta; this.dc=dCutoff; this.x=null; this.dx=0; }
  static alpha(cutoff, dt){ const tau=1/(2*Math.PI*cutoff); return 1/(1+tau/dt); }
  filter(v, dt){
    if(!Number.isFinite(v)) return this.x;
    if(this.x===null || !(dt>0)){ this.x=v; return v; }
    const dv=(v-this.x)/dt;
    const ad=OneEuro.alpha(this.dc, dt);
    this.dx = ad*dv + (1-ad)*this.dx;
    const a=OneEuro.alpha(this.mc + this.beta*Math.abs(this.dx), dt);
    this.x = a*v + (1-a)*this.x;
    return this.x;
  }
}
let lastPoseT = null, poseNeutral = null;
/* 高斯平滑：处理逐帧独立噪声的离群点 */
function gaussSmooth(arr, w){
  const h = Math.max(1, w/2), k = [];
  for(let i=-w;i<=w;i++) k.push(Math.exp(-(i*i)/(2*h*h)));
  const out = new Array(arr.length).fill(null);
  for(let i=0;i<arr.length;i++){
    let a = 0, c = 0;
    for(let j=-w;j<=w;j++){
      const idx = i+j;
      if(idx<0 || idx>=arr.length || arr[idx]==null) continue;
      a += arr[idx]*k[j+w]; c += k[j+w];
    }
    out[i] = c ? a/c : null;
  }
  return out;
}
/* 3 帧中值：去单帧离群但保留阶跃（高斯 ±3 帧会把一帧内的快动作抹平 7 帧） */
function medianSmooth(arr){
  const out = new Array(arr.length).fill(null);
  for(let i=0;i<arr.length;i++){
    const win = [arr[i-1], arr[i], arr[i+1]].filter(v => v!=null && Number.isFinite(v));
    out[i] = win.length ? win.slice().sort((a,b)=>a-b)[Math.floor(win.length/2)] : null;
  }
  return out;
}
/* 源姿态去噪：单目逐帧检测的定位噪声是独立噪声，忠实播放等于把噪声当动作。
   加载时一次完成（中值去离群 + One-Euro 低通），运行时只做插值 —— 与帧率无关。 */
export function denoiseMotion(mo){
  const T = mo.poses.length, J = mo.poses[0]?.length || 17;
  const dt = 1/Math.max(1, mo.fps || 24);
  const out = mo.poses.map(p => p ? p.map(j => j ? [j[0], j[1], j[2]] : null) : null);
  for(let j=0;j<J;j++){
    const xs = mo.poses.map(p => (p && p[j]) ? p[j][0] : null);
    const ys = mo.poses.map(p => (p && p[j]) ? p[j][1] : null);
    const gx = medianSmooth(xs), gy = medianSmooth(ys);
    const fx = new OneEuro(2.5, 0.03), fy = new OneEuro(2.5, 0.03);
    for(let i=0;i<T;i++){
      if(!out[i] || !out[i][j]) continue;
      out[i][j][0] = fx.filter(gx[i], dt);
      out[i][j][1] = fy.filter(gy[i], dt);
    }
  }
  mo.posesD = out;
  return mo;
}
/* 躯干转身通道：单目下绕竖轴的旋转只能靠「肩宽透视收缩」估幅度、「鼻−耳中点偏移」定方向。
   参考宽度自校准：取「头正对镜头」的帧的肩宽中位（此时躯干多半也正对）。 */
export function deriveMotionChannels(mo){
  const D = mo.posesD || mo.poses;
  const T = D.length;
  const shW = new Array(T).fill(null), headYaw = new Array(T).fill(null);
  for(let i=0;i<T;i++){
    const p = D[i]; if(!p) continue;
    const L=p[MKEY.lsh], R=p[MKEY.rsh];
    if(L&&R) shW[i] = Math.abs(L[0]-R[0]);
    const n=p[MKEY.nose], le=p[MKEY.lear], re=p[MKEY.rear];
    if(n&&le&&re){ const half=Math.max(.08, Math.abs(le[0]-re[0])/2);
      headYaw[i]=Math.asin(clamp((n[0]-(le[0]+re[0])/2)/half,-1,1)); }
  }
  const calm=[];
  for(let i=0;i<T;i++) if(headYaw[i]!=null && Math.abs(headYaw[i])<.20 && shW[i]!=null) calm.push(shW[i]);
  calm.sort((a,b)=>a-b);
  const ref = calm.length ? calm[Math.floor(calm.length/2)]
    : (shW.filter(v=>v!=null).sort((a,b)=>a-b)[Math.floor(T/2)] || 1);
  let yaw = shW.map((w,i)=>{
    if(w==null || headYaw[i]==null) return 0;
    const sgn = Math.abs(headYaw[i])>.14 ? Math.sign(headYaw[i]) : 0;
    return sgn*Math.acos(clamp(w/ref,0,1));
  });
  yaw = yaw.map((v,i)=>{ if(i===0||i===yaw.length-1) return v;
    const s=[yaw[i-1],yaw[i],yaw[i+1]].sort((a,b)=>a-b); return s[1]; });
  const f = new OneEuro(2.0,.04), sm=[]; let last=null;
  for(let i=0;i<yaw.length;i++){ const d=last===null?0:i-last; last=i; sm.push(f.filter(yaw[i],d)); }
  mo.chan = {yaw: sm.map(v=>Math.abs(v)<30*DEG?0:v)};
  /* 接触通道：更低的脚=支撑脚；其水平速度低于阈值且持续 → 踩实（用于锁脚，防滑步） */
  const ank = (i,n)=>{ const p=D[i]; const q=p&&p[MKEY[n]]; return q?[q[0],q[1]]:null; };
  const contact=[];
  for(let i=0;i<T;i++){
    const L=ank(i,'lan'), R=ank(i,'ran'), pL=ank(i-1,'lan'), pR=ank(i-1,'ran');
    const c=[false,false];
    if(L&&R){
      const lower = L[1]<=R[1] ? 0 : 1;
      const cur = lower===0?L:R, prev = lower===0?pL:pR;
      const v = prev ? Math.abs(cur[0]-prev[0]) : 1;
      c[lower] = v < .025;
    }
    contact.push(c);
  }
  const rawContact = contact.map(c=>[c[0],c[1]]);   /* 需连续 2 帧，避免单帧误判 */
  for(let i=0;i<T;i++){
    for(const k of [0,1]){
      if(rawContact[i][k] && !(i>0 && rawContact[i-1][k])) contact[i][k]=false;
    }
  }
  mo.chan.contact = contact;
  return mo;
}
/* 脚是否踩实（接触门控）：踩实则保持上一目标，锁住脚，消除滑步 */
export function footPlanted(tag, t){
  const mo = motion;
  if(!mo || !mo.chan || !mo.chan.contact) return false;
  const k = tag==='L' ? 0 : 1;
  const fi = clamp(Math.round((t - spec.shot.start)*(mo.fps||24)), 0, mo.chan.contact.length-1);
  return !!mo.chan.contact[fi][k];
}
/* 通道取样：相邻帧线性插值 */
export function chanYaw(t){
  const mo = motion;
  if(!mo || !mo.chan || !mo.chan.yaw.length) return 0;
  const P = mo.chan.yaw;
  const ft = (t - spec.shot.start)*(mo.fps||24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1), i1 = Math.min(P.length-1, i0+1);
  return lerp(P[i0], P[i1], clamp(ft-i0, 0, 1));
}
/* 躯干侧倾通道（弧度，正=上端向主体左倾）：躯干轴相对画面垂直的倾角，已扣相机滚转 */
function chanRoll(t){
  const mo = motion;
  if(!mo || !mo.chan || !mo.chan.roll || !mo.chan.roll.length) return 0;
  const P = mo.chan.roll;
  const ft = (t - spec.shot.start)*(mo.fps||24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1), i1 = Math.min(P.length-1, i0+1);
  return lerp(P[i0], P[i1], clamp(ft-i0, 0, 1));
}
/* 源视频取景：主体像素包围盒占画面高的比例（对照机位用） */
export function videoBoxFrac(t){
  const mo = motion;
  if(!mo || !mo.videoBox || !mo.videoBox.frac || !mo.videoBox.frac.length) return null;
  const P = mo.videoBox.frac;
  const ft = (t - spec.shot.start)*(mo.fps||24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1), i1 = Math.min(P.length-1, i0+1);
  const f = clamp(ft-i0, 0, 1);
  const out = [];
  for(let k=0;k<P[i0].length;k++) out.push(lerp(P[i0][k], P[i1][k], f));
  return out;
}
/* 取样：相邻帧线性插值 —— 消除「最近帧」取样造成的阶跃，渲染出连续过程 */

export function poseAt(t){
  if(!motion) return null;
  const P = motion.posesD || motion.poses;
  if(!P.length) return null;
  const ft = (t - spec.shot.start)*(motion.fps || 24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1);
  const i1 = Math.min(P.length-1, i0+1);
  const f = clamp(ft - i0, 0, 1);
  const a = P[i0], b = P[i1];
  if(!a) return b || null;
  if(!b || f <= 0) return a;
  return a.map((p,j)=>{
    const q = b[j];
    if(!p) return q;
    if(!q) return p;
    return [lerp(p[0], q[0], f), lerp(p[1], q[1], f), Math.max(p[2], q[2])];
  });
}
/* 去噪已在加载时完成；这里只做时间连续性与跳转检测 */
let poseJump = false;
/* v3 透视缩短深度重构的三维骨架（相机系：x=画面右 / y=上 / z=远离相机，米，髋中点原点）。
   相邻帧线性插值；无 pose3d（v1/v2）返回 null，四肢走旧 2D 路径。 */
function pose3dAt(t){
  if(!motion || !motion.pose3d || !motion.pose3d.length) return null;
  const P = motion.pose3d;
  const ft = (t - spec.shot.start)*(motion.fps || 24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1);
  const i1 = Math.min(P.length-1, i0+1);
  const f = clamp(ft - i0, 0, 1);
  const a = P[i0], b = P[i1];
  if(!a) return b || null;
  if(!b || f <= 0) return a;
  return a.map((p,j)=>{
    const q = b[j];
    if(!p) return q;
    if(!q) return p;
    return [lerp(p[0], q[0], f), lerp(p[1], q[1], f), lerp(p[2], q[2], f)];
  });
}
/* v3 骨段可信度（1=可信 / 0.35=投影长度越界，深度不可信）：
   相邻帧逐关节取较小者 —— 链上任一骨不可信就整链回退 2D 路径（深度会放大 2D 噪声）。 */
function conf3dAt(t){
  if(!motion || !motion.pose3dConf || !motion.pose3dConf.length) return null;
  const P = motion.pose3dConf;
  const ft = (t - spec.shot.start)*(motion.fps || 24);
  const i0 = clamp(Math.floor(ft), 0, P.length-1);
  const i1 = Math.min(P.length-1, i0+1);
  const a = P[i0] || [], b = P[i1] || [];
  const out = [];
  for(let j=0;j<(a.length || b.length);j++) out.push(Math.min(a[j] ?? 0, b[j] ?? 0));
  return out;
}
/* 相机系 → 躯干系（ψ = 躯干转身角，正 = 主体向自身左转）：
   body_x = cosψ·cam_x + sinψ·cam_z；body_y = cam_y；body_z = sinψ·cam_x − cosψ·cam_z */
function camToBodyVec(v3, psi, out){
  const c = Math.cos(psi), s = Math.sin(psi);
  return (out || new THREE.Vector3()).set(c*v3[0] + s*v3[2], v3[1], s*v3[0] - c*v3[2]);
}
/* 躯干系向量 → 世界（基向量来自 rigAxes） */
function bodyVecToWorld(b, ax, out){
  return (out || new THREE.Vector3()).set(0,0,0)
    .addScaledVector(ax.left, b.x).addScaledVector(ax.up, b.y).addScaledVector(ax.fwd, b.z);
}
function filteredPose(t){
  const raw = poseAt(t);
  if(!raw) return null;
  poseJump = (lastPoseT===null || Math.abs(t-lastPoseT) > 0.3);
  if(poseJump) limbAngleCache.clear();
  lastPoseT = t;
  return raw;
}
/* 中性姿态基准（用于把绝对偏移转成变化量）：鼻−耳中点的偏移、肩宽、髋宽 */
function poseNeutralRef(){
  if(poseNeutral || !motion) return poseNeutral;
  const good = (motion.posesD || motion.poses).filter(Boolean);
  const med = arr => { const s=[...arr].sort((a,b)=>a-b); return s.length? s[Math.floor(s.length/2)] : 0; };
  const noseDx = [], earDy = [], shW = [], hpW = [];
  for(const p of good){
    const nose=p[MKEY.nose], le=p[MKEY.lear], re=p[MKEY.rear], ls=p[MKEY.lsh], rs=p[MKEY.rsh],
          lh=p[MKEY.lhp], rh=p[MKEY.rhp];
    if(nose&&le&&re&&le[2]>.3&&re[2]>.3&&nose[2]>.3){
      noseDx.push(nose[0]-(le[0]+re[0])/2);
      earDy.push(nose[1]-(le[1]+re[1])/2);
    }
    if(ls&&rs&&ls[2]>.3&&rs[2]>.3) shW.push(Math.abs(ls[0]-rs[0]));
    if(lh&&rh&&lh[2]>.3&&rh[2]>.3) hpW.push(Math.abs(lh[0]-rh[0]));
  }
  poseNeutral = {noseDx:med(noseDx), noseDy:med(earDy), shW:med(shW)||.5, hpW:med(hpW)||.3};
  return poseNeutral;
}
/* 视频端的参考伸展长度：肩→腕 / 髋→踝 的 90 分位（近伸直状态） */
let reachRef = null;
/* 源视频躯干单位 → 世界米：规格身高 × (源视频躯干像素 / 可见身高像素)。
   模型躯干比例与真人不同（Mixamo 躯干偏短、腿偏长），直接拿模型躯干长换算
   会把手臂缩短 36%；源视频的躯干像素长在动作文件的 videoBox.torsoPx 里。 */
function sourceTorsoScale(){
  const mo = motion;
  if(!mo || !mo.videoBox || !mo.videoBox.torsoPx || !mo.videoBox.frac) return (rig && rig.torso) || .35;
  const ft = (playhead - spec.shot.start)*(mo.fps||24);
  const i0 = clamp(Math.floor(ft), 0, mo.videoBox.torsoPx.length-1);
  const i1 = Math.min(mo.videoBox.torsoPx.length-1, i0+1);
  const f = clamp(ft-i0, 0, 1);
  const tp = lerp(mo.videoBox.torsoPx[i0], mo.videoBox.torsoPx[i1], f);
  const fr = lerp(mo.videoBox.frac[i0][0], mo.videoBox.frac[i1][0], f);
  const H = (spec.primary && spec.primary.size && spec.primary.size[2]) || 1.6;
  const visiblePx = fr * (mo.videoBox.h || 640);
  if(!(visiblePx > 1) || !(tp > 0)) return (rig && rig.torso) || .35;
  return H * tp / visiblePx;
}

function reachRefs(){
  if(reachRef) return reachRef;
  const arm=[], leg=[];
  for(const p of (motion?.posesD || motion?.poses || [])){
    if(!p) continue;
    const sm2 = joint2(p, MKEY.lsh, MKEY.rsh);
    const lm = joint2(p, MKEY.lhp, MKEY.rhp);
    if(sm2){ for(const w of [MKEY.lwr, MKEY.rwr]){ const W=p[w]; if(W&&W[2]>=.25) arm.push(Math.hypot(W[0]-sm2[0], W[1]-sm2[1])); } }
    if(lm){ for(const a of [MKEY.lan, MKEY.ran]){ const A=p[a]; if(A&&A[2]>=.25) leg.push(Math.hypot(A[0]-lm[0], A[1]-lm[1])); } }
  }
  const q = (arr,f)=>{ const s2=[...arr].sort((a,b)=>a-b); return s2.length? s2[Math.min(s2.length-1, Math.floor(s2.length*f))] : 1; };
  reachRef = {arm:q(arm,.9)||1, leg:q(leg,.9)||1};
  return reachRef;
}
export function joint2(pose, a, b){
  if(!pose) return null;
  const A = pose[a], B = pose[b];
  const good = x => x && x[2] >= .22;
  if(good(A) && good(B)) return [(A[0]+B[0])/2, (A[1]+B[1])/2];
  if(good(A)) return [A[0], A[1]];
  if(good(B)) return [B[0], B[1]];
  return null;
}
/* 目标跟随：时间常数化（一阶低通），与帧率无关 —— 固定 alpha 会让高刷屏上抖动被放大 */
export function sm(id, v, tau, dt){
  const cur = motionTargets.get(id);
  const next = (cur === null || cur === undefined || v === null)
    ? (cur ?? v)
    : (()=>{ const a = 1 - Math.exp(-(dt||1/30)/tau); return [lerp(cur[0], v[0], a), lerp(cur[1], v[1], a)]; })();
  motionTargets.set(id, next);
  return next;
}
/* 末端伸展量平滑：肘/膝角的噪声经余弦定理会被放大成位置抖动（d伸展/d角度≈L/2）。
   用固定时间常数低通（伸展量本身变化慢，40ms 滞后可忽略）——自适应滤波在快速动作时
   会放开噪声，反而把抖动原样传下去。 */
function smoothReach(tag, v, dt, jump){
  if(!rig.reachFilt) rig.reachFilt = new Map();
  let st = rig.reachFilt.get(tag);
  if(!st){ st = {x:null}; rig.reachFilt.set(tag, st); }
  if(jump || st.x === null || !Number.isFinite(v)){ st.x = v; return v; }
  const a = 1 - Math.exp(-(dt||1/30)/.04);
  st.x += (v - st.x)*a;
  return st.x;
}
/* 两连杆 IK：全程世界空间 —— 方向无量纲，长度用米制，与模型内部坐标/缩放解耦 */
function applyWorldDelta(bone, deltaWorld){
  const P = bone.parent;
  const qP = P.getWorldQuaternion(new THREE.Quaternion());
  const dLocal = qP.clone().invert().multiply(deltaWorld).multiply(qP);
  const old = bone.quaternion.clone();
  bone.quaternion.copy(dLocal).multiply(old);
}
/* 绝对姿态：绕世界轴的旋转 × 初始姿势（不累乘） */
function poseBoneWorld(name, qWorld){
  const j = rig.nodes.get(name);
  if(!j || !j.parent) return;
  const qP = j.parent.getWorldQuaternion(new THREE.Quaternion());
  const qLocal = qP.clone().invert().multiply(qWorld).multiply(qP);
  const q0 = rig.initQuat.get(name) || j.quaternion;
  j.quaternion.copy(qLocal).multiply(q0);
}
/* 目标跟随（三维）：与 sm() 同构的一阶低通，用于 pose3d 偏移 ——
   深度对 2D 噪声敏感（|Δz|=√(L²−d²) 在 d→L 处病态），必须时域滤波。 */
function sm1(id, v, tau, dt){
  /* 标量版一阶低通（深度分量用）；时间跳转必须丢弃旧状态 */
  if(poseJump){ motionTargets3.set(id, v); return v; }
  const cur = motionTargets3.get(id);
  if(cur === undefined || cur === null){ motionTargets3.set(id, v); return v; }
  const a = 1 - Math.exp(-(dt||1/30)/tau);
  const out = cur + (v - cur)*a;
  motionTargets3.set(id, out);
  return out;
}
function sm3(id, v, tau, dt){
  /* 时间跳转：滤波器状态属于上一时刻，硬 seek 后必须丢弃，否则读到混合值 */
  if(poseJump){ motionTargets3.set(id, v ? v.clone() : null); return v ? v.clone() : null; }
  let cur = motionTargets3.get(id);
  if(!cur){ motionTargets3.set(id, v ? v.clone() : null); return v ? v.clone() : null; }
  if(v === null) return cur.clone();
  if(!cur){ motionTargets3.set(id, v.clone()); return v.clone(); }
  const a = 1 - Math.exp(-(dt||1/30)/tau);
  cur.lerp(v, a);
  return cur.clone();
}
/* conf 门控滞回：可信度逐帧翻转（单帧越界）会让目标在两条路径间跳 0.5 m。
   与 resolveBend 的 pend 同构：连续 3 帧反向才切换。 */
function gate3d(id, ok){
  if(!rig.p3gate) rig.p3gate = {};
  const g = rig.p3gate[id] || (rig.p3gate[id] = {use: !!ok, pend: 0});
  if(ok === g.use){ g.pend = 0; }
  else if(++g.pend >= 4){ g.use = !g.use; g.pend = 0; }
  return g.use;
}
/* 两连杆 IK：全程世界空间。elbowWorld（可选）= 观测到的中段关节位置 ——
   只用它定「肘/膝在弦的哪一侧」（弯折面法向 n），肘/膝到弦的垂距仍由骨长决定；
   末端位置始终精确命中 targetWorld。弯折面深度是 2D 不可观测量，
   与其用常数先验不如用 v3 重构出的观测值。
   tilt（可选）= 绕弦轴旋转弯折面法向的角度（弧度）：肘/膝在弦平面内的方位不可观测，
   穿模时用它把肘/膝让出躯干。 */
function solveChain(chain, targetWorld, bendAxisWorld, elbowWorld, tilt){
  const [uName,lName,eName] = chain;
  const up = rig.nodes.get(uName), lo = rig.nodes.get(lName), en = rig.nodes.get(eName);
  const mu = rig.meta.get(uName), ml = rig.meta.get(lName);
  if(!up || !lo || !en || !mu || !ml || !up.parent) return;
  const L1 = mu.worldLen, L2 = ml.worldLen;
  up.updateWorldMatrix(true, false);
  const boneW = up.getWorldPosition(new THREE.Vector3());
  /* 几何解在内核（纯函数，可单测）：根点/末端/骨长/极向量 → 肘位与两根骨方向。
     期望肘位 → 法向：投影到「垂直于弦」的平面，方向即观测，垂距由骨长强制。 */
  const g = solveTwoBone(toVec(boneW), toVec(targetWorld), L1, L2,
                         elbowWorld ? toVec(elbowWorld) : null,
                         bendAxisWorld ? toVec(bendAxisWorld) : null, tilt);
  if(!g) return;
  const K = toThree(g.elbow);
  const curW = lo.getWorldPosition(new THREE.Vector3()).sub(boneW).normalize();
  const v1 = K.clone().sub(boneW).normalize();
  applyWorldDelta(up, new THREE.Quaternion().setFromUnitVectors(curW, v1));
  lo.updateWorldMatrix(true, false);
  const loW = lo.getWorldPosition(new THREE.Vector3());
  const cur2 = en.getWorldPosition(new THREE.Vector3()).sub(loW).normalize();
  const v2 = targetWorld.clone().sub(loW).normalize();
  applyWorldDelta(lo, new THREE.Quaternion().setFromUnitVectors(cur2, v2));
}

/* 弯折面随弦搬运（平行移动）：
   穿模推出会**转弦** —— 源视频里手臂折叠得很紧（弦只有 15cm），推 20cm 就能把弦转 60°+。
   绝对提示点（肘的 3D 位置）里**沿弦的分量**在弦转动时泄漏到垂直方向，
   肘于是翻到弦的另一侧（E7 判成反关节）。
   修法：只搬运「肘相对弦的偏置方向」，方向与弦保持原夹角 —— 推出前后肘始终同侧。 */
function transportedPole(chain, newTarget){
  const up = rig.nodes.get(chain[0]), lo = rig.nodes.get(chain[1]), hd = rig.nodes.get(chain[2]);
  if(!up || !lo || !hd) return null;
  const p = transportPole(toVec(up.getWorldPosition(new THREE.Vector3())),
                          toVec(lo.getWorldPosition(new THREE.Vector3())),
                          toVec(hd.getWorldPosition(new THREE.Vector3())),
                          toVec(newTarget));
  return p ? toThree(p) : null;
}
/* 弯折侧向偏移（E7 同量纲）：弯折点相对「根→末端」弦线在**身体前向**上的投影 */
function chainBendOff(chain){
  const a = rig.nodes.get(chain[0]), b = rig.nodes.get(chain[1]), c = rig.nodes.get(chain[2]);
  if(!a || !b || !c) return null;
  const fwd = new THREE.Vector3(0,0,1).applyQuaternion(rig.root.getWorldQuaternion(new THREE.Quaternion()));
  const off = bendOffset(toVec(a.getWorldPosition(new THREE.Vector3())),
                         toVec(b.getWorldPosition(new THREE.Vector3())),
                         toVec(c.getWorldPosition(new THREE.Vector3())), toVec(fwd));
  return off === null ? null : +off.toFixed(4);
}
/* ---------- 身体碰撞代理：穿模兜底 ---------- */
/* 代理半径由蒙皮网格实测：顶点按「主骨骼权重」分组，取到骨轴垂距的 95 分位。
   取 95 而不是 90：代理必须包住网格，90 分位会让 10% 的表面露在代理外，
   穿模判定会漏判（参考项目结论：胶囊代理偏乐观，只有网格说了算）。
   袖筒是外挂几何（不在蒙皮网格里），半径直接取已知的袖筒半径。 */
const PROXY_BONE_MAP = {
  pelvis:'mixamorig:Hips', torso:'mixamorig:Spine2', head:'mixamorig:Head',
  thighL:'mixamorig:LeftUpLeg', thighR:'mixamorig:RightUpLeg',
  shinL:'mixamorig:LeftLeg', shinR:'mixamorig:RightLeg',
  handL:'mixamorig:LeftHand', handR:'mixamorig:RightHand'
};
/* 袖筒半径（Mixamo 导出单位，1 单位 = 1cm），与 buildHumanFigure 里建袖筒时一致 */
export const SLEEVE_R = {upper:6.2, fore:5.2};
/* 骨名查找：three.js 的 GLTFLoader 会剥掉节点名里的保留字符
   （'mixamorig:Head' → 'mixamorigHead'），统一按规范化名查找。 */
export function findBoneByName(root, name){
  const key = String(name).toLowerCase().replace(/[^a-z0-9]/g, '');
  let found = null;
  root.traverse(o => { if(o.isBone && o.name.toLowerCase().replace(/[^a-z0-9]/g, '') === key) found = o; });
  return found;
}
/* 实测代理半径：单次遍历顶点，按主骨骼分组，算到骨轴垂距 */
export function measureProxyRadii(mesh, skeleton, axes){
  const out = {};
  const si = mesh.geometry.attributes.skinIndex, sw = mesh.geometry.attributes.skinWeight;
  if(!si || !sw) return out;
  const boneIdx = new Map();
  skeleton.bones.forEach((b,i) => boneIdx.set(b.name.toLowerCase().replace(/[^a-z0-9]/g,''), i));
  const buckets = new Map();
  for(const [key, boneName] of Object.entries(PROXY_BONE_MAP)){
    const bi = boneIdx.get(boneName.toLowerCase().replace(/[^a-z0-9]/g,''));
    const ax = axes[key];
    if(bi == null || !ax) continue;
    buckets.set(bi, {key, a:ax.a, d:ax.d, dists:[]});
  }
  if(!buckets.size) return out;
  const v = new THREE.Vector3();
  const n = mesh.geometry.attributes.position.count;
  for(let i=0;i<n;i++){
    const ids = [si.getX(i), si.getY(i), si.getZ(i), si.getW(i)];
    const ws  = [sw.getX(i), sw.getY(i), sw.getZ(i), sw.getW(i)];
    let best = -1, bw = 0;
    for(let k=0;k<4;k++) if(ws[k] > bw){ bw = ws[k]; best = ids[k]; }
    const b = buckets.get(best);
    if(!b) continue;
    mesh.getVertexPosition(i, v);
    v.applyMatrix4(mesh.matrixWorld);      /* getVertexPosition 返回网格局部坐标 */
    const r = v.clone().sub(b.a);
    b.dists.push(Math.sqrt(Math.max(0, r.lengthSq() - Math.pow(r.dot(b.d), 2))));
  }
  for(const b of buckets.values()){
    if(!b.dists.length) continue;
    b.dists.sort((x,y) => x-y);
    const p95 = b.dists[Math.min(b.dists.length-1, Math.floor(b.dists.length*0.95))];
    out[b.key] = {p95:+p95.toFixed(4), max:+b.dists[b.dists.length-1].toFixed(4)};
  }
  return out;
}
/* 躯干横截面剖面（按高度分箱实测）。
   每箱记录**横向/前后最大半宽**（而不是一个圆柱半径）：
   圆柱半径取全身最粗处 → 腰侧的手臂被误判成深插胸口；只按前后深度取半径，
   又会漏掉胸腔的横向鼓出。OBB 盒（lat × up × fwd）既紧又保守。 */
const TORSO_PROFILE_BINS = 8;
export function measureTorsoProfile(mesh, skeleton, axisA, axisDir, latDir, fwdDir){
  const si = mesh.geometry.attributes.skinIndex, sw = mesh.geometry.attributes.skinWeight;
  if(!si || !sw) return null;
  const boneIdx = new Map();
  skeleton.bones.forEach((b,i) => boneIdx.set(b.name.toLowerCase().replace(/[^a-z0-9]/g,''), i));
  const want = new Set();
  for(const n of ['mixamorig:Hips','mixamorig:Spine','mixamorig:Spine1','mixamorig:Spine2']){
    const i = boneIdx.get(n.toLowerCase().replace(/[^a-z0-9]/g,''));
    if(i != null) want.add(i);
  }
  const pts = [];
  const v = new THREE.Vector3();
  const n = mesh.geometry.attributes.position.count;
  for(let i=0;i<n;i++){
    const ids=[si.getX(i),si.getY(i),si.getZ(i),si.getW(i)];
    const ws=[sw.getX(i),sw.getY(i),sw.getZ(i),sw.getW(i)];
    let best=-1, bw=0;
    for(let k=0;k<4;k++) if(ws[k]>bw){bw=ws[k];best=ids[k];}
    if(!want.has(best)) continue;
    mesh.getVertexPosition(i, v); v.applyMatrix4(mesh.matrixWorld);
    const r = v.clone().sub(axisA);
    pts.push([r.dot(axisDir), r.dot(latDir), r.dot(fwdDir)]);
  }
  if(pts.length < 50) return null;
  let t0 = Infinity, t1 = -Infinity;
  for(const p of pts){ if(p[0]<t0) t0=p[0]; if(p[0]>t1) t1=p[0]; }
  if(t1 - t0 < 1e-3) return null;
  const bins = [];
  for(let i=0;i<TORSO_PROFILE_BINS;i++) bins.push({lat:0, dep:0});
  for(const [t,la,de] of pts){
    const b = clamp(Math.floor((t-t0)/(t1-t0)*TORSO_PROFILE_BINS), 0, TORSO_PROFILE_BINS-1);
    if(Math.abs(la) > bins[b].lat) bins[b].lat = Math.abs(la);
    if(Math.abs(de) > bins[b].dep) bins[b].dep = Math.abs(de);
  }
  return {t0, t1, bins};
}
/* 碰撞代理统一为「点 → 在代理内部的深度」（正 = 在内部，越大越深）。
   形状：躯干 = 按高度分箱的 OBB（紧且保守），头/四肢 = 胶囊（近似圆柱）。 */
function capProxy(name, a, b, r){
  const ab = b.clone().sub(a), L2 = ab.lengthSq();
  return {name, kind:'cap', a, b, r,
    pen(p){
      const t = L2 > 1e-12 ? clamp(p.clone().sub(a).dot(ab)/L2, 0, 1) : 0;
      return r - p.distanceTo(a.clone().addScaledVector(ab, t));
    }};
}
/* 躯干横截面用**椭圆**而不是方盒：方盒的角会伸到体外，
   手在盒角处会被判成"在体内 6cm"而代理其实没碰到。 */
function ellProxy(name, c, up, lat, fwd, hLat, hDep, hUp){
  return {name, kind:'ell', c, up, lat, fwd, hLat, hDep, hUp,
    pen(p){
      const d = p.clone().sub(c);
      const y = Math.abs(d.dot(up))/this.hUp;
      if(y >= 1) return -(y-1)*this.hUp;
      const x = Math.abs(d.dot(lat))/this.hLat, z = Math.abs(d.dot(fwd))/this.hDep;
      const q = Math.sqrt(x*x + z*z);
      return (1 - q) * Math.min(this.hLat, this.hDep);
    }};
}
/* 每帧的身体代理：躯干（OBB 剖面）+ 头/腿/臂（胶囊） */
export function bodyProxies(skip, ax){
  if(!rig || !rig.nodes || !rig.proxyR) return null;
  const R = rig.proxyR;
  const wp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
  const list = [];
  const add = p => { if(p) list.push(p); };
  const hips = wp('hips'), chest = wp('chest'), head = wp('head');
  const hipMid = rigHipJointPoint() || hips;
  if(rig.torsoProfile && hipMid && chest && ax){
    const up = chest.clone().sub(hipMid);
    const h = up.length();
    if(h > 1e-3){
      up.divideScalar(h);
      const pr = rig.torsoProfile;
      const lat = ax.left.clone(), fwd = ax.fwd.clone();
      for(let i=0;i<pr.bins.length;i++){
        const b = pr.bins[i];
        if(b.lat <= 0.005) continue;
        const t0 = pr.t0 + (pr.t1-pr.t0)*i/pr.bins.length;
        const t1 = pr.t0 + (pr.t1-pr.t0)*(i+1)/pr.bins.length;
        const c = hipMid.clone().addScaledVector(up, (t0+t1)/2);
        add(ellProxy('torso'+i, c, up, lat, fwd, b.lat, b.dep, (t1-t0)/2));
      }
    }
  }
  add(capProxy('head', head, head, R.head));
  for(const [tag,hk,kk,ak] of [['L','hipL','kneeL','ankleL'],['R','hipR','kneeR','ankleR']]){
    add(capProxy('thigh'+tag, wp(hk), wp(kk), R['thigh'+tag]));
    add(capProxy('shin'+tag,  wp(kk), wp(ak), R['shin'+tag]));
  }
  for(const [tag,sk,ek,wk] of [['L','shoulderL','elbowL','handL'],['R','shoulderR','elbowR','handR']]){
    add(capProxy('upperArm'+tag, wp(sk), wp(ek), R['upperArm'+tag]));
    add(capProxy('foreArm'+tag, wp(ek), wp(wk), R['foreArm'+tag]));
    add(capProxy('hand'+tag,    wp(wk), wp(wk), R['hand'+tag]));
  }
  return skip ? list.filter(c => !skip.includes(c.name)) : list;
}
/* 链的采样点（根关节除外：肩/髋本来就长在躯干里，恒在包络内 → 穿透量恒为正，
   推出与让位就永远"推不动"。只测上臂/前臂/腿/手的内部与末端。
   手臂链的末端是**指尖**而不是腕点：手沿手指方向伸出，掌心/指尖可能整片插进躯干
   而腕点还在外面（半径只管宽度不管长度）。 */
function chainSamples(chain, tag){
  const isArm = chain[0].startsWith('shoulder');
  const wp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
  const p0 = wp(chain[0]), p1 = wp(chain[1]), p2 = wp(chain[2]);
  if(!p0 || !p1 || !p2) return [];
  const R = rig.proxyR || {};
  const r0 = (isArm ? R['upperArm'+tag] : R['thigh'+tag]) || .06;
  const r1 = (isArm ? R['foreArm'+tag] : R['shin'+tag]) || .05;
  const r2 = isArm ? (R['hand'+tag] || .04) : (R['shin'+tag] || .05);
  const out = [[p0.clone().lerp(p1,.5), r0], [p1, r1], [p1.clone().lerp(p2,.5), r1], [p2, r2]];
  if(isArm){
    const tipB = rig.fingerTip && rig.fingerTip[tag];
    const tip = tipB ? tipB.getWorldPosition(new THREE.Vector3()) : null;
    if(tip && tip.distanceTo(p2) > 1e-4){
      /* 指尖是个点，不能按手的宽度再减一次半径（否则永远判为"在外面"） */
      out.push([p2.clone().lerp(tip,.5), r2*.5]);
      out.push([tip, 0.004]);
    }
    /* 指根与五指指尖：模型的手指是**张开**的，骨骼位置不代表几何范围 ——
       只测中指会漏掉外侧手指（实测外侧手指顶点插进躯干 7cm、中指却在体外）。
       指根在体内而指尖在体外也是常见形态（手背贴着身体）。 */
    /* 半径取手指的粗细（≈1.2~1.5cm）：判据量的是**表面**的穿透，
       中心在体外而手指表面在体内是常见形态。 */
    const roots = rig.fingerRoot && rig.fingerRoot[tag];
    if(roots) for(const b of roots) out.push([b.getWorldPosition(new THREE.Vector3()), 0.015, true]);
    const mids = rig.fingerMid && rig.fingerMid[tag];
    if(mids) for(const b of mids) out.push([b.getWorldPosition(new THREE.Vector3()), 0.014, true]);
    const tips = rig.fingerTips && rig.fingerTips[tag];
    if(tips) for(const b of tips) out.push([b.getWorldPosition(new THREE.Vector3()), 0.010, true]);
  }
  return out;
}
/* 链（肩→肘→手 / 髋→膝→踝）相对身体代理的穿透量：
   沿链采样若干点，每点带该段半径；代理深度减去半径才是真正插进去的量。 */
export function chainPenetration(chain, tag, ax){
  /* 世界矩阵要先刷新：IK 刚改完骨骼，指尖/子骨的矩阵还是上一帧的 */
  if(rig && rig.root) rig.root.updateMatrixWorld(true);
  const isArm = chain[0].startsWith('shoulder');
  const skip = isArm ? ['upperArm'+tag, 'foreArm'+tag, 'hand'+tag]
                     : ['thigh'+tag, 'shin'+tag, 'hand'+tag];
  const caps = bodyProxies(skip, ax);
  if(!caps || !caps.length) return 0;
  const samples = chainSamples(chain, tag);
  let m = 0;
  for(const [p, r] of samples)
    for(const c of caps){ const v = c.pen(p) - r; if(v > m) m = v; }
  return m;
}
/* 链采样点相对躯干网格的**真值**穿透（正 = 在网格内，量级 = 到较近表面）。
   代理是包络：既会偏大（包络比实体大）也会偏小（实测腕部代理判"在外"、网格判"在内"4cm），
   量级必须以网格为准。 */

/* 可见手部几何相对躯干网格的穿透量（内核实现，core/clip.js truthPen）：
   判据只看裸露的手指（掌背藏在卫衣里、袖筒与躯干同衣，不是可见穿模）；
   表面穿透 = 中心深度 + 该处半径。 */
export function chainTruthPen(chain, tag, truth){
  if(rig && rig.root) rig.root.updateMatrixWorld(true);
  const samples = chainSamples(chain, tag);
  const cs = samples.map(s2 => ({p: toVec(s2[0]), r: s2[1], vis: s2[2] === true}));
  return truthPen(cs, truth.body, toVec(truth.camPos));
}
/* 推出身体：沿候选方向找最小推出量。
   首选「相机射线」方向 —— 针孔投影下射线上的点投影相同，沿射线平移**不改变画面位置**，
   穿模修复对视频一致性零代价；相机射线由规格+视频基准解算，与当前视图无关（换镜头不改动作）。 */
function pushOutOfBody(p, caps, dirs){
  const pen = maxPenetration(p, caps);
  if(pen <= 0) return {point:p, moved:0};
  for(const dir of dirs){
    if(!dir || dir.lengthSq() < 1e-9) continue;
    for(const mul of [1.05, 1.35, 1.9, 2.8, 4.2]){
      const q = p.clone().addScaledVector(dir, pen*mul);
      if(maxPenetration(q, caps) <= 0) return {point:q, moved:pen*mul};
    }
  }
  return {point:p, moved:0};
}
/* 手（含手指）推出躯干：手的胶囊 = 腕点 + 前臂方向延伸 8.5cm。
   沿相机射线推：屏幕位置不变 → 视频一致性零代价。 */
const HAND_SPAN = .085;
const MAX_PUSH = .30;          /* 推出量上限：超过说明代理不可信，宁可不推 */
/* 可见手部穿透的硬上限（= E18 的验收阈值 20mm）：超过它就必须动结构，
   否则宁可选解剖正确（弯折不翻面）而留一点穿透。 */
const CLIP_CAP = .020;
/* 链上中段（肘/膝）穿躯干时，绕弦轴微调弯折面 —— 肘/膝在弦平面内的方位不可观测，
   让出躯干的代价为零。tilt 上限 ±35°，取穿透最小的一侧。
   两段都要测：上臂/大腿的袖筒也会插进胸口（手被推出去了，胳膊还在里面）。 */
function resolveChainCollision(chain, target, bend, hint, tag, ax, truth){
  /* truth = {body, camPos} 时用真值（网格）判据评估候选，否则用保守代理 */
  const t0 = truth ? chainTruthPen(chain, tag, truth) : {pen:chainPenetration(chain, tag, ax), idx:-1};
  const pen0 = t0.pen;
  if(pen0 <= 0.004) return pen0;
  const isArm = chain[0].startsWith('shoulder');
  const wp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
  const p0 = wp(chain[0]), mid = wp(chain[1]);
  const meta = rig.meta;
  const L1 = (meta.get(chain[0]) || {}).worldLen || 0, L2 = (meta.get(chain[1]) || {}).worldLen || 0;
  if(!p0 || !mid || !L1 || !L2) return pen0;
  const bones = chain.map(n => rig.nodes.get(n)).filter(Boolean);
  if(bones.length < 3) return pen0;
  const save = bones.map(b => b.quaternion.clone());
  const restore = () => bones.forEach((b,i) => b.quaternion.copy(save[i]));
  let best = {pen:pen0, score:truth ? t0.sum : pen0, tilt:0};
  const baseScore = best.score;
  const side0 = chainBendOff(chain);        /* 弯折侧向（正 = 弯折点在弦的前侧） */
  /* 候选角：① 细扫 ±60°（肘/膝在弦平面内的方位不可观测，可自由选取）；
     ② **把肘往相机方向推**（视线轴方向）—— 上臂横穿躯干时唯一能动的就是肘的方位，
        沿视线轴移动肘不改变它的屏幕位置（针孔投影下射线上的点投影相同）→ 一致性零代价。
        做法：把视线方向投影到弦的法平面，得到目标法向 n_target；
        当前法向 n_cur（肘相对弦的偏移方向）与 n_target 的有符号夹角就是候选 tilt。 */
  const tilts = truth ? [0.45,-0.45,0.9,-0.9] : [0.2,-0.2,0.45,-0.45,0.7,-0.7,0.9,-0.9,1.05,-1.05,1.25,-1.25];
  if(isArm){
    const cam = videoCamState();
    const wrist = wp(chain[2]);
    if(cam && wrist){
      const chord = wrist.clone().sub(p0);
      const r = chord.length();
      if(r > 1e-6){
        chord.divideScalar(r);
        const a = (r*r + L1*L1 - L2*L2) / (2*r);      /* 弦足到肩的距离 */
        const foot = p0.clone().addScaledVector(chord, a);
        const nCur = mid.clone().sub(foot);
        if(nCur.lengthSq() > 1e-8){
          nCur.normalize();
          const toCam = cam.pos.clone().sub(mid).normalize();
          const nTgt = toCam.clone().addScaledVector(chord, -toCam.dot(chord));
          if(nTgt.lengthSq() > 1e-8){
            nTgt.normalize();
            const ang = Math.atan2(nCur.clone().cross(nTgt).dot(chord), clamp(nCur.dot(nTgt), -1, 1));
            if(Number.isFinite(ang) && Math.abs(ang) > 1e-3) tilts.push(ang);
          }
        }
      }
    }
  }
  let sameBest = null, anyBest = null;
  for(const tilt of tilts){
    restore();
    solveChain(chain, target, bend, hint, tilt);
    const t = truth ? chainTruthPen(chain, tag, truth) : {pen:chainPenetration(chain, tag, ax), sum:0};
    const score = truth ? t.sum : t.pen;
    const cand = {pen:t.pen, score, tilt};
    if(!anyBest || score < anyBest.score - 1e-4) anyBest = cand;
    /* 弯折侧向不变优先：肘翻到弦的另一侧 = 反关节（解剖错误），
       代价远大于几厘米的穿透 → 同侧候选优先，同侧都无改善才允许翻面。 */
    const off = chainBendOff(chain);
    const same = (side0 === null || off === null) ? true : (Math.sign(off) === Math.sign(side0));
    if(same && (!sameBest || score < sameBest.score - 1e-4)) sameBest = cand;
  }
  restore();
  /* 同侧候选优先；只有同侧根本压不下去（穿透超过验收阈值）才允许翻面 ——
     翻面 = 反关节（解剖错误），代价远大于几厘米的可见穿透。 */
  const chosen = (sameBest && sameBest.score < baseScore - 1e-4) ? sameBest
               : (anyBest && anyBest.pen > CLIP_CAP && anyBest.score < baseScore - 1e-4) ? anyBest : best;
  solveChain(chain, target, bend, hint, chosen.tilt);
  return chosen.pen;
}
/* ---------- 网格级穿透实测（穿模判据，独立于运行时代理） ---------- */
/* 运行时用胶囊代理解算（便宜但偏乐观），判定必须量**真实网格**：
   手/袖筒的顶点 vs 躯干网格的真实三角面 —— 代理是包络，网格才是事实。 */
const BODY_BONES = ['mixamorig:Hips','mixamorig:Spine','mixamorig:Spine1','mixamorig:Spine2',
  'mixamorig:Head','mixamorig:LeftUpLeg','mixamorig:RightUpLeg',
  'mixamorig:LeftLeg','mixamorig:RightLeg','mixamorig:LeftFoot','mixamorig:RightFoot'];
/* 蒙皮网格的顶点 → 世界坐标 + 主骨骼索引 */
export function skinVertexData(sm){
  const geo = sm.geometry, pos = geo.attributes.position;
  const si = geo.attributes.skinIndex, sw = geo.attributes.skinWeight;
  const n = pos.count;
  const world = new Float32Array(n*3), dom = new Int16Array(n), boneName = [];
  const v = new THREE.Vector3();
  for(let i=0;i<n;i++){
    sm.getVertexPosition(i, v);
    v.applyMatrix4(sm.matrixWorld);      /* getVertexPosition 返回网格局部坐标 */
    world[i*3]=v.x; world[i*3+1]=v.y; world[i*3+2]=v.z;
    const ids=[si.getX(i),si.getY(i),si.getZ(i),si.getW(i)];
    const ws=[sw.getX(i),sw.getY(i),sw.getZ(i),sw.getW(i)];
    let b=-1, bw=0;
    for(let k=0;k<4;k++) if(ws[k] > bw){ bw=ws[k]; b=ids[k]; }
    dom[i]=b;
    boneName[i] = b>=0 ? (sm.skeleton.bones[b] ? sm.skeleton.bones[b].name : '?') : '?';
  }
  return {world, dom, count:n, boneName};
}
/* 身体网格的拓扑（哪些三角形属于身体）—— 静态，只算一次。
   逐帧只需刷新这些顶点的世界坐标（蒙皮顶点比全部顶点少一半）。 */
const bodyTopoCache = {sm:null, groups:null, verts:null};
export function bodyTopology(sm){
  if(bodyTopoCache.sm === sm && bodyTopoCache.groups) return bodyTopoCache;
  const idx = sm.geometry.index;
  if(!idx) return null;
  const boneIdx = new Map();
  sm.skeleton.bones.forEach((b,i)=>boneIdx.set(b.name.toLowerCase().replace(/[^a-z0-9]/g,''), i));
  const si = sm.geometry.attributes.skinIndex, sw = sm.geometry.attributes.skinWeight;
  const dom = new Int16Array(sm.geometry.attributes.position.count);
  for(let i=0;i<dom.length;i++){
    const ids=[si.getX(i),si.getY(i),si.getZ(i),si.getW(i)];
    const ws=[sw.getX(i),sw.getY(i),sw.getZ(i),sw.getW(i)];
    let b=-1, bw=0;
    for(let k=0;k<4;k++) if(ws[k]>bw){ bw=ws[k]; b=ids[k]; }
    dom[i]=b;
  }
  const groups = {};
  for(const g in BODY_GROUPS){
    const set = new Set();
    for(const n of BODY_GROUPS[g]){
      const i = boneIdx.get(n.toLowerCase().replace(/[^a-z0-9]/g,''));
      if(i != null) set.add(i);
    }
    groups[g] = set;
  }
  const out = {};
  for(const g in groups) out[g] = [];
  for(let t=0;t<idx.count;t+=3){
    const a=idx.getX(t), b=idx.getX(t+1), c=idx.getX(t+2);
    for(const g in groups){
      const set = groups[g];
      if(set.has(dom[a]) && set.has(dom[b]) && set.has(dom[c])) out[g].push(a,b,c);
    }
  }
  /* 去重顶点 + 紧凑索引 */
  const used = new Set();
  for(const g in out) for(const v of out[g]) used.add(v);
  const verts = Array.from(used).sort((x,y)=>x-y);
  const remap = new Map(verts.map((v,i)=>[v,i]));
  const groupsOut = {};
  for(const g in out){
    const tri = Int32Array.from(out[g].map(v=>remap.get(v)));
    const vset = new Set(tri);
    groupsOut[g] = {tri, vlist: Int32Array.from(vset)};
  }
  bodyTopoCache.sm = sm;
  bodyTopoCache.groups = groupsOut;
  bodyTopoCache.verts = Int32Array.from(verts);
  return bodyTopoCache;
}
/* 身体网格的逐帧世界坐标（只算身体用到的顶点） */
const bodyWorldCache = {w:null, n:0};
export function bodyWorldNow(sm, topo){
  const n = topo.verts.length;
  if(!bodyWorldCache.w || bodyWorldCache.n !== n) bodyWorldCache.w = new Float32Array(n*3);
  const W = bodyWorldCache.w;
  const v = new THREE.Vector3();
  for(let k=0;k<n;k++){
    const i = topo.verts[k];
    sm.getVertexPosition(i, v);
    v.applyMatrix4(sm.matrixWorld);
    W[k*3]=v.x; W[k*3+1]=v.y; W[k*3+2]=v.z;
  }
  bodyWorldCache.n = n;
  /* 各部位的包围盒：射线粗筛（腿部位占 2 万面，不筛就是每点 0.24ms） */
  for(const g in topo.groups){
    const vl = topo.groups[g].vlist;
    let x0=Infinity,y0=Infinity,z0=Infinity,x1=-Infinity,y1=-Infinity,z1=-Infinity;
    for(let i=0;i<vl.length;i++){
      const o=vl[i]*3, px=W[o], py=W[o+1], pz=W[o+2];
      if(px<x0)x0=px; if(py<y0)y0=py; if(pz<z0)z0=pz;
      if(px>x1)x1=px; if(py>y1)y1=py; if(pz>z1)z1=pz;
    }
    topo.groups[g].bb = [x0,y0,z0,x1,y1,z1];
  }
  return {w:W, groups:topo.groups};
}
/* 射线与轴对齐包围盒是否可能相交（slab 法） */
/* 身体网格的真实三角面（世界坐标），按部位分组 + 各自重心 */
export function bodyTriangles(sm, skin){
  const idx = sm.geometry.index;
  if(!idx) return null;
  const boneIdx = new Map();
  sm.skeleton.bones.forEach((b,i) => boneIdx.set(b.name.toLowerCase().replace(/[^a-z0-9]/g,''), i));
  const sets = {};
  for(const g in BODY_GROUPS){
    const set = new Set();
    for(const n of BODY_GROUPS[g]){
      const i = boneIdx.get(n.toLowerCase().replace(/[^a-z0-9]/g,''));
      if(i != null) set.add(i);
    }
    sets[g] = set;
  }
  const dom = skin.dom;
  const groups = {};
  for(const g in BODY_GROUPS) groups[g] = [];
  for(let t=0;t<idx.count;t+=3){
    const a = idx.getX(t), b = idx.getX(t+1), c = idx.getX(t+2);
    for(const g in BODY_GROUPS){
      const set = sets[g];
      if(set.has(dom[a]) && set.has(dom[b]) && set.has(dom[c])) groups[g].push(a,b,c);
    }
  }
  const out = {w:skin.world, groups:{}};
  for(const g in BODY_GROUPS) out.groups[g] = {tri:Int32Array.from(groups[g])};
  return out;
}
/* 点到三角形的最近点 */
function triClosest(px,py,pz, ax,ay,az, bx,by,bz, cx,cy,cz, o){
  const abx=bx-ax, aby=by-ay, abz=bz-az, acx=cx-ax, acy=cy-ay, acz=cz-az;
  const apx=px-ax, apy=py-ay, apz=pz-az;
  const d1=abx*apx+aby*apy+abz*apz, d2=acx*apx+acy*apy+acz*apz;
  if(d1<=0 && d2<=0){ o[0]=ax; o[1]=ay; o[2]=az; return; }
  const bpx=px-bx, bpy=py-by, bpz=pz-bz;
  const d3=abx*bpx+aby*bpy+abz*bpz, d4=acx*bpx+acy*bpy+acz*bpz;
  if(d3>=0 && d4<=d3){ o[0]=bx; o[1]=by; o[2]=bz; return; }
  const vc=d1*d4-d3*d2;
  if(vc<=0 && d1>=0 && d3<=0){ const v=d1/(d1-d3); o[0]=ax+abx*v; o[1]=ay+aby*v; o[2]=az+abz*v; return; }
  const cpx=px-cx, cpy=py-cy, cpz=pz-cz;
  const d5=abx*cpx+aby*cpy+abz*cpz, d6=acx*cpx+acy*cpy+acz*cpz;
  if(d6>=0 && d5<=d6){ o[0]=cx; o[1]=cy; o[2]=cz; return; }
  const vb=d5*d2-d1*d6;
  if(vb<=0 && d2>=0 && d6<=0){ const w=d2/(d2-d6); o[0]=ax+acx*w; o[1]=ay+acy*w; o[2]=az+acz*w; return; }
  const va=d3*d6-d5*d4;
  if(va<=0 && (d4-d3)>=0 && (d5-d6)>=0){
    const w=(d4-d3)/((d4-d3)+(d5-d6));
    o[0]=bx+(cx-bx)*w; o[1]=by+(cy-by)*w; o[2]=bz+(cz-bz)*w; return;
  }
  const den=1/(va+vb+vc), v=vb*den, w=vc*den;
  o[0]=ax+abx*v+acx*w; o[1]=ay+aby*v+acy*w; o[2]=az+abz*v+acz*w;
}
/* 穿模量 = **径向判据**（绕序无关、也不依赖网格封闭）：
   对每个身体部位分组（躯干/头/双腿）取其顶点重心 C，
   沿 C→点 的方向求出该方向的表面半径 R（射线首次命中的距离）：
     |点−C| > R → 在体外；否则在体内，深度 = R − |点−C|。
   躯干/头/腿对各自重心近似星形 ✓；三角形绕序朝内朝外都不影响判定
   （Mixamo 导出的躯干三角形法向朝内，用叉积定符号会把身后 17cm 的点
   误判成穿进胸口 10cm）。 */
const BODY_GROUPS = {
  torso:['mixamorig:Hips','mixamorig:Spine','mixamorig:Spine1','mixamorig:Spine2'],
  head:['mixamorig:Head'],
  legL:['mixamorig:LeftUpLeg','mixamorig:LeftLeg','mixamorig:LeftFoot'],
  legR:['mixamorig:RightUpLeg','mixamorig:RightLeg','mixamorig:RightFoot']
};
/* 射线首次命中三角面的距离 */
/* 射线-三角形求交（Möller–Trumbore）：返回命中距离 t，无命中返回 null。
   写法用向量积显式表达，避免符号错误 —— 符号一错，命中判据 u<0 会把**所有**
   合法命中判成非法，穿模判据就永远返回 0（静默失效）。 */
/* 视线穿透区间：点相对躯干网格的**真实**穿模深度（米；正 = 在体内，负 = 在体外，
   量级 = 到较近表面的距离）。
   内外判定既不依赖三角形法向（Mixamo 躯干三角形法向朝内，用法向定符号会全判反 ——
   见 BUG0041），也不依赖奇偶判定（网格非封闭）：相机在体外 ⇒ 沿视线方向第一个交点
   是前表面、最后一个是后表面，点落在两者之间即在体内。
   保守代理只能当**筛子**（代理外一定在网格外），量级必须由网格给。 */

/* 点相对身体网格的穿模深度（内核实现，core/geom.js intervalDepth）：
   相机在体外 ⇒ 沿视线首交点=前表面、末交点=后表面，点在两者之间即在体内。
   内外判定不依赖三角形法向（Mixamo 躯干法向朝内），也不依赖奇偶（网格非封闭）。 */
export function bodyIntervalDepth(p, body, camPos, diag){
  return intervalDepth(toVec(p), body, toVec(camPos), diag);
}
/* 手/前臂/袖筒的顶点集合（世界坐标）。
   bare=true 时只取**裸露的手**（Hand 骨）：上臂/前臂的蒙皮被袖筒罩住、袖筒与躯干
   属同一件卫衣 —— 它们互穿在画面上看不见；只有手（皮肤）对躯干/腿/头的互穿才是可见穿模。 */
function armVertices(sm, skin, tag, mode){
  const boneIdx = new Map();
  sm.skeleton.bones.forEach((b,i)=>boneIdx.set(b.name.toLowerCase().replace(/[^a-z0-9]/g,''), i));
  const want = new Set();
  const side = tag==='L' ? 'Left' : 'Right';
  const names = mode === 'fingers'
    ? ['Index','Middle','Ring','Pinky','Thumb'].flatMap(f=>[1,2,3].map(k=>'mixamorig:'+side+'Hand'+f+k))
    : mode === 'hand' ? ['mixamorig:'+side+'Hand']
    : ['mixamorig:'+side+'Hand', 'mixamorig:'+side+'ForeArm'];
  for(const n of names){
    const i = boneIdx.get(n.toLowerCase().replace(/[^a-z0-9]/g,'')); if(i != null) want.add(i);
  }
  const out = [];
  const w = skin.world, dom = skin.dom;
  for(let i=0;i<skin.count;i++) if(want.has(dom[i])) out.push(i);
  /* 同侧袖筒（外挂几何，不在蒙皮网格里）*/
  const armBone = rig && rig.nodes ? rig.nodes.get(tag==='L' ? 'shoulderL' : 'shoulderR') : null;
  const foreBone = rig && rig.nodes ? rig.nodes.get(tag==='L' ? 'elbowL' : 'elbowR') : null;
  const sleeves = [];
  for(const b of [armBone, foreBone]){
    if(!b) continue;
    b.traverse(o=>{ if(o.isMesh && o.name === 'sleeve') sleeves.push(o); });
  }
  /* 袖筒起点在肩/肘关节处 —— 那里本来就长在躯干里，套筒的端帽必然在体内。
     判定穿模要排除贴根的那一截（否则量到的永远是「肩关节在身体里」）。 */
  const v = new THREE.Vector3();
  const sleevePts = [];
  for(const s of sleeves){
    const p = s.geometry.attributes.position;
    const root = s.parent && s.parent.getWorldPosition ? s.parent.getWorldPosition(new THREE.Vector3()) : null;
    const tip  = s.getWorldPosition(new THREE.Vector3());
    let axis = null;
    if(root && tip){ axis = tip.clone().sub(root); const L = axis.length(); if(L > 1e-4) axis.divideScalar(L); }
    for(let i=0;i<p.count;i++){
      v.fromBufferAttribute(p, i).applyMatrix4(s.matrixWorld);
      if(axis && root){
        const t = v.clone().sub(root).dot(axis);
        if(t < 0.30) continue;          /* 贴根的 30% 不计入穿模判定 */
      }
      sleevePts.push(v.clone());
    }
  }
  return {skinned:out, skin, sleevePts};
}
/* 网格级穿透实测：返回每侧最深穿透（米，正 = 已在网格内部） */
export function meshPenetration(){
  export const sm = figGroup && figGroup.getObjectByProperty('isSkinnedMesh', true);
  if(!sm || !sm.geometry.index) return null;
  /* 必须先刷新世界矩阵与骨架：updateScene 只改了骨骼姿态，
     没刷新的话量到的是**上一帧**的顶点位置（推出/摆动都看不见）。 */
  if(figGroup) figGroup.updateMatrixWorld(true);
  const skin = skinVertexData(sm);
  const body = bodyTriangles(sm, skin);
  if(!body || !body.groups.torso.tri.length) return null;
  const c = [0,0,0];
  const res = {};
  let worst = 0;
  /* 两级判定：先用胶囊代理筛（代理是包络，代理外一定在网格外），
     只对嫌疑顶点做精确的点到三角面距离 —— 全量精确判定太慢。 */
  const proxyCaps = bodyProxies(null, rigAxes());
  /* 判据用的相机：视线方向决定穿透区间的采样方向，用渲染机位（体内外判定对凸体
     与方向无关，用视频机位则与当前视图解耦）。 */
  const vc = videoCamState();
  const camPos = (vc && vc.pos) ? vc.pos : camera.position;
  for(const tag of ['L','R']){
    const av = armVertices(sm, skin, tag, 'fingers');
    const mine = ['upperArm'+tag, 'foreArm'+tag, 'hand'+tag];
    const caps = proxyCaps ? proxyCaps.filter(c2 => !mine.includes(c2.name)) : [];
    const penAt = (x,y,z) => { const q = new THREE.Vector3(x,y,z); let m = -1e9;
      for(const c2 of caps){ const v = c2.pen(q); if(v > m) m = v; } return m; };
    let maxPen = 0, where = null, srcBest = null;
    let envMax = -1e9, envSkin = -1e9;
    const test = (x,y,z,src) => {
      const env = caps.length ? penAt(x,y,z) : -1;
      if(env > envMax) envMax = env;
      if(caps.length && env < -0.03) return;   /* 代理外 3cm → 必在网格外 */
      const pen = bodyIntervalDepth(new THREE.Vector3(x,y,z), body, camPos);
      if(pen > maxPen){ maxPen = pen; where = [x,y,z]; srcBest = src; }
    };
    for(const i of av.skinned){
      const bn = skin.boneName && skin.boneName[i];
      const x = skin.world[i*3], y = skin.world[i*3+1], z = skin.world[i*3+2];
      if(caps.length){ const e = penAt(x,y,z); if(e > envSkin) envSkin = e; }
      test(x, y, z, 'skin:'+(bn||'?'));
    }
    /* 袖筒不参与：它与躯干是同一件卫衣，互穿在画面上不可见。 */
    res[tag] = {pen: +maxPen.toFixed(5), at: where ? where.map(v=>+v.toFixed(3)) : null, src: srcBest,
                 env: +envMax.toFixed(4), envSkin: +envSkin.toFixed(4)};
    if(maxPen > worst) worst = maxPen;
  }
  res.max = +worst.toFixed(5);
  res.tris = body.groups.torso.tri.length/3;
  return res;
}

/* ---------- 掌面法线测量 ---------- */
/* 掌面法线（手骨局部系）：手指屈曲方向 = 掌面朝向（参考项目结论：
   位置无法承载扭转 —— 掌向是纯扭转自由度，只能单独驱动）。
   测量法：用模型自带动画找手指弯曲最大的时刻，此时远节指骨相对近节的
   偏转方向就是掌面法线，再换算回手骨局部系。无可用动画时退回解剖约定
   （Mixamo 静息 T-Pose 掌心朝下）。 */
const PALM_FINGER = {
  L:['mixamorig:LeftHandMiddle1','mixamorig:LeftHandMiddle2','mixamorig:LeftHandMiddle3'],
  R:['mixamorig:RightHandMiddle1','mixamorig:RightHandMiddle2','mixamorig:RightHandMiddle3']
};
const PALM_HAND = {L:'mixamorig:LeftHand', R:'mixamorig:RightHand'};
export function measurePalmAxes(gltf, wrapper){
  const out = {};
  const boneOf = name => findBoneByName(wrapper, name);
  /* 动画会同时改骨骼的**位置与旋转**（Mixamo 的根骨有位置轨道）。
     只还原四元数会静默污染绑定姿势 —— 实测模型的腿长被改掉，
     人偶的脚陷进地面 22cm（E13/E3/E7 全部带偏）。位置/缩放一并还原。 */
  const save = [];
  wrapper.traverse(o => { if(o.isBone) save.push([o, o.quaternion.clone(), o.position.clone(), o.scale.clone()]); });
  const restore = () => { for(const [b,q,p,sc] of save){ b.quaternion.copy(q); b.position.copy(p); b.scale.copy(sc); } };
  const clips = (gltf && gltf.animations) ? gltf.animations : [];
  const clip = clips.find(a => !/tpose/i.test(a.name || '')) || clips[0];
  let mixer = null;
  if(clip){
    try{
      mixer = new THREE.AnimationMixer(wrapper);
      mixer.clipAction(clip).play();
    }catch(e){ mixer = null; }
  }
  const best = {};
  if(mixer){
    const N = 32;
    for(let k=0;k<=N;k++){
      mixer.setTime(clip.duration*k/N);
      wrapper.updateMatrixWorld(true);
      for(const tag of ['L','R']){
        const [n1,n2,n3] = PALM_FINGER[tag];
        const b1 = boneOf(n1), b2 = boneOf(n2), b3 = boneOf(n3), hb = boneOf(PALM_HAND[tag]);
        if(!b1 || !b2 || !b3 || !hb) continue;
        const p1 = b1.getWorldPosition(new THREE.Vector3()), p2 = b2.getWorldPosition(new THREE.Vector3()),
              p3 = b3.getWorldPosition(new THREE.Vector3());
        const A = p2.clone().sub(p1).normalize(), B = p3.clone().sub(p2).normalize();
        const ang = Math.acos(clamp(A.dot(B), -1, 1));
        if(!best[tag] || ang > best[tag].ang){
          const tan = B.clone().addScaledVector(A, -B.dot(A));
          if(tan.lengthSq() > 1e-8)
            best[tag] = {ang, tan: tan.normalize(),
                         q: hb.getWorldQuaternion(new THREE.Quaternion()).clone()};
        }
      }
    }
  }
  restore();
  for(const tag of ['L','R']){
    const hb = boneOf(PALM_HAND[tag]);
    if(!hb) continue;
    if(best[tag] && best[tag].ang > 0.35){
      const n = best[tag].tan.clone().applyQuaternion(best[tag].q.clone().invert()).normalize();
      out[tag] = {axis:n, source:'measured', flex:+best[tag].ang.toFixed(3)};
    }else{
      /* 退回解剖约定：静息姿势掌心朝模型下方 */
      const q = hb.getWorldQuaternion(new THREE.Quaternion());
      const n = new THREE.Vector3(0,-1,0).applyQuaternion(q.clone().invert()).normalize();
      out[tag] = {axis:n, source:'convention', flex: best[tag] ? +best[tag].ang.toFixed(3) : 0};
    }
  }
  return out;
}

/* 源视频机位（规格 + 视频基准解算）：与当前渲染视图无关，保证动作与机位解耦 */function videoCamState(){
  const H = spec && spec.primary ? spec.primary.size[2] : 1.6;
  const hfov = spec && spec.camera ? spec.camera.fov_deg : 50;
  const vb = motion && motion.videoBox;
  if(!vb || !vb.w || !vb.h) return null;      /* 无源视频基准 → 没有规范机位 */
  const refAspect = vb.w/vb.h;
  const halfV = Math.atan(Math.tan(hfov/2*DEG)/refAspect);
  const fovV = 2*halfV;
  const box = videoBoxFrac(playhead);
  const fh = box ? clamp(box[0], .10, .95) : .45;
  const sx = box && box.length>1 ? box[1] : .5;
  const sy = box && box.length>2 ? box[2] : .5;
  const dist = clamp(H/Math.max(.05, 2*Math.tan(halfV)*fh), .8, 80);
  const camH = spec && spec.camera ? spec.camera.height_m : 1.2;
  const hipY = rigHipJointY() ?? ((rig && rig.restHipY) || H*.55);
  const halfH = Math.atan(Math.tan(halfV)*refAspect);
  const yaw = Math.atan((sx-.5)*2*Math.tan(halfH));
  const pit = Math.atan2(hipY - camH, dist) - (sy-.5)*fovV;
  const p = spec && spec.primary ? samplePath(spec.primary.path, playhead) : {x:0, d:0};
  return {pos:new THREE.Vector3(p.x, camH, p.d - dist),
          dir:new THREE.Vector3(Math.sin(yaw)*Math.cos(pit), Math.sin(pit), Math.cos(yaw)*Math.cos(pit))};
}
/* 视频端 2D 关节夹角（顶点 b）：尺度无关，是最忠实的姿态信号 */
function angle2(pose, a, b, c){
  const A=pose[a], B=pose[b], C=pose[c];
  if(!A||!B||!C || A[2]<.25||B[2]<.25||C[2]<.25) return null;
  const v1x=A[0]-B[0], v1y=A[1]-B[1], v2x=C[0]-B[0], v2y=C[1]-B[1];
  const den = Math.hypot(v1x,v1y)*Math.hypot(v2x,v2y);
  if(den < 1e-6) return null;
  return Math.acos(clamp((v1x*v2x+v1y*v2y)/den, -1, 1));
}
/* 由目标夹角反解骨根到末端的距离（余弦定理） */
function reachFor(L1, L2, theta){
  const c = Math.cos(theta);
  return Math.sqrt(Math.max(1e-6, L1*L1 + L2*L2 - 2*L1*L2*c));
}
/* 弯折平面观测：视频 2D 的「根→末端」与「中点」相对关系（方位 + 置信度）。
   手臂接近伸直时垂直偏移趋零、符号由噪声决定 —— 只作弱观测，不直接驱动骨骼。 */
function bendObs2D(pose, iRoot, iMid, iEnd){
  const R = pose[iRoot], M = pose[iMid], E = pose[iEnd];
  const ok = x => x && x[2] >= .25;
  if(!ok(R) || !ok(M) || !ok(E)) return null;
  const ux = E[0]-R[0], uy = E[1]-R[1];
  const ul = Math.hypot(ux,uy);
  if(ul < 1e-4) return null;
  const nx = ux/ul, ny = uy/ul;
  const ex = M[0]-R[0], ey = M[1]-R[1];
  const dot = ex*nx + ey*ny;
  const bx = ex - nx*dot, by = ey - ny*dot;
  const bl = Math.hypot(bx,by);
  if(bl < 0.03) return null;
  return {bx:bx/bl, by:by/bl, conf:bl};
}
/* 弯折方位状态机：去抖（短时翻转不计）+ 限速（每帧最多转 7°）+ 解剖先验。
   手臂接近伸直时垂直偏移趋零、方向由噪声决定 —— 观测只作平面内的小幅倾角，
   深度方向用解剖先验（肘向前、膝向后），任何情况下方位都连续演化，不会闪跳。 */
function resolveBend(key, obs, ax, depthPrior, dt){
  if(!rig.bendState) rig.bendState = new Map();
  let st = rig.bendState.get(key);
  if(!st){ st = {ang:0, goal:0, side:1, pend:0, init:false, jump:0}; rig.bendState.set(key, st); }
  if(obs && obs.conf > 0.04){
    const target = Math.atan2(obs.by, obs.bx);
    if(!st.init){ st.ang = st.goal = target; st.side = 1; st.init = true; }
    else{
      const side = Math.cos(target - st.ang) >= 0 ? 1 : -1;
      if(side !== st.side){
        st.pend++;
        if(st.pend >= 3){ st.goal = target; st.side = side; st.pend = 0; }
      }else{
        st.pend = 0;
        st.goal = target;
      }
    }
    let d = st.goal - st.ang;
    while(d > Math.PI) d -= 2*Math.PI;
    while(d < -Math.PI) d += 2*Math.PI;
    const maxTurn = .12*Math.max(1, dt*30);          /* 每帧最多 7° */
    st.ang += Math.abs(d) > maxTurn ? Math.sign(d)*maxTurn : d;
  }
  const magT = obs ? Math.min(1, obs.conf/.12)*.55 : 0;
  st.mag = st.mag === undefined ? magT : st.mag + (magT - st.mag)*Math.min(1, dt*6);
  const target = new THREE.Vector3()
    .addScaledVector(ax.left, Math.cos(st.ang)*st.mag)
    .addScaledVector(ax.up, Math.sin(st.ang)*st.mag)
    .addScaledVector(ax.fwd, depthPrior)
    .normalize();
  const maxStep = .12*Math.max(1, dt*30);          /* 每帧最多 7° */
  if(!st.dir) st.dir = target.clone();
  else{
    const ang = st.dir.angleTo(target);
    if(ang > maxStep) st.dir.lerp(target, maxStep/ang).normalize();
    else st.dir.copy(target);
  }
  if(st.last){ if(st.last.angleTo(st.dir) > .45) st.jump++; }   /* 单帧突变守卫 */
  st.last = st.dir.clone();
  return st.dir;
}
export function rigAxes(){
  const rootQ = (rig.yawNode || rig.root).getWorldQuaternion(new THREE.Quaternion());
  return {
    left: rig.basis.left.clone().applyQuaternion(rootQ),
    up: rig.basis.up.clone().applyQuaternion(rootQ),
    fwd: rig.basis.fwd.clone().applyQuaternion(rootQ)
  };
}
const headState = {yaw:0, pitch:0, vy:0, vp:0};
export let lastChestRoll = 0, lastHipRoll = 0;
const limbAngleCache = new Map();     /* 每肢：上次有效夹角 + 滤波器 */
function limbAngle(key, pose, a, b, c, dt, jump){
  let st = limbAngleCache.get(key);
  if(!st){ st = {v:null, f:new OneEuro(3.0,.015)}; limbAngleCache.set(key, st); }
  if(jump){ st.v = null; st.f.x = null; st.f.dx = 0; }
  const raw = angle2(pose, a, b, c);
  if(raw === null){ return st.v; }          /* 掉帧：沿用上次有效值，不切回距离回退 */
  st.v = st.f.filter(raw, dt);
  return st.v;
}
export function applyMotion(t){
  if(!rig || !figGroup) return;
  const dt = Math.max(1e-3, Math.min(.1, Math.abs((lastPoseT ?? t) - t) || 1/30));
  const pose = filteredPose(t);
  const pose3d = pose3dAt(t);
  const conf3d = conf3dAt(t);
  /* 跳转后旧的锁脚目标失效（否则拖时间轴会把脚拖到上一时刻的位置） */
  if(poseJump && rig.lastTarget){ rig.lastTarget.ankleL = null; rig.lastTarget.ankleR = null; }
  if(!motionTargets) motionTargets = new Map();
  if(!motionTargets3) motionTargets3 = new Map();
  rig.pushOut = {};                    /* 推出量统计（逐帧重置） */
  /* 躯干转身：绕角色竖轴的整体旋转（幅度=肩宽透视收缩，方向=头部朝向） */
  const bodyYaw = chanYaw(t);
  const yawQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0), bodyYaw);
  if(rig.yawNode) rig.yawNode.rotation.y = bodyYaw;
  rig.root.updateMatrixWorld(true);
  const ax = rigAxes();
  const hips = rig.nodes.get('hips');
  const hipW = hips.getWorldPosition(new THREE.Vector3());
  /* 髋高：由支撑腿的竖直伸展决定（模型骨长 × 膝角），不用「图像踝高 × 单位比例」。
     图像单位是躯干长，而模型躯干偏短、腿偏长（Mixamo 角色比例），按躯干比例换算
     会让人物陷进深蹲；改用腿的几何量 → 与模型自洽，也与图像景深无关。 */
  const ankL = pose ? pose[MKEY.lan] : null, ankR = pose ? pose[MKEY.ran] : null;
  const kneeAngL = pose ? limbAngle('kneeL', pose, MKEY.lhp, MKEY.lkn, MKEY.lan, dt, poseJump) : null;
  const kneeAngR = pose ? limbAngle('kneeR', pose, MKEY.rhp, MKEY.rkn, MKEY.ran, dt, poseJump) : null;
  const plantL = footPlanted('L', t), plantR = footPlanted('R', t);
  let supSide = null;
  if(plantL !== plantR) supSide = plantL ? 'L' : 'R';
  else if(ankL && ankR) supSide = (ankL[1] <= ankR[1]) ? 'L' : 'R';
  const supKnee = supSide === 'R' ? kneeAngR : kneeAngL;
  const supThigh = rig.meta.get(supSide === 'R' ? 'hipR' : 'hipL')?.worldLen || .3;
  const supShin = rig.meta.get(supSide === 'R' ? 'kneeR' : 'kneeL')?.worldLen || .3;
  if(supKnee != null){
    const bend = clamp((Math.PI - supKnee)/2, 0, 85*DEG);   /* 对称弯曲近似（膝角为弧度） */
    /* 髋高基准取**静止站姿的髋关节高度**（股骨头），不是骨盆根节点：
       Mixamo 的骨盆根比股骨头高 8.6cm，混用会让整条腿凭空短一截（人偶永久下蹲）。
       竖直伸展用实测竖直跨距（髋关节高 − 踝关节离地高），不是骨长之和
       （静止姿势的膝是前屈的，骨长之和比竖直跨距大）。 */
    const want = (rig.vertReach || (supThigh + supShin)) * Math.cos(bend) + (rig.ankleGroundY || 0);
    const hist = rig.hipHist || (rig.hipHist = []);
    if(poseJump) hist.length = 0;
    hist.push(want);
    if(hist.length > 3) hist.shift();
    const w = [...hist].sort((a,b)=>a-b)[Math.floor(hist.length/2)];
    const delta = clamp(w - (rig.hipJointY ?? rig.restHipY), -0.5, 0);   /* 不高于自然站高（腾空不可观测） */
    rig.hipWant = w; rig.hipDelta = delta;
    /* 骨骼局部单位 ≠ 米（Mixamo 导出 100 单位 = 1m）→ 换算到父级局部系再位移 */
    const pScale = hips.parent.getWorldScale(new THREE.Vector3()).length()/Math.sqrt(3);
    const upLocal = ax.up.clone().applyQuaternion(hips.parent.getWorldQuaternion(new THREE.Quaternion()).invert()).normalize();
    hips.position.copy(rig.hipsRest).addScaledVector(upLocal, delta/Math.max(1e-6, pScale));
  }
  /* 躯干侧倾：v2 取离线通道（躯干轴倾角，扣相机滚转，不随转身衰减）；
     v1 用肩线倾角 + 偏航补偿（tan(倾角)·cos ψ 反解）。 */
  const shL = pose ? pose[MKEY.lsh] : null, shR = pose ? pose[MKEY.rsh] : null;
  const shMid = joint2(pose, MKEY.lsh, MKEY.rsh);
  let chestRoll = 0, hipRoll = 0;
  if(motion && motion.chanV2){
    const r = -chanRoll(t);                 /* 通道正=上端向主体左倾；绕前向轴正转=向右倾 → 反号 */
    chestRoll = r*.6; hipRoll = r*.4;       /* 骨盆承担 4 成，脊柱三节共 6 成 */
  }else{
    if(shL && shR && shL[2] >= .25 && shR[2] >= .25){
      const dx = shL[0]-shR[0], dy = shL[1]-shR[1];
      const len = Math.hypot(dx, dy);
      if(dx > 0 && len > 1e-4) chestRoll = clamp(Math.asin(clamp(dy/len,-1,1)), -.6, .6);  /* 仰角：偏航无关 */
    }
    const hpL = pose ? pose[MKEY.lhp] : null, hpR = pose ? pose[MKEY.rhp] : null;
    if(hpL && hpR && hpL[2] >= .25 && hpR[2] >= .25){
      const dx = hpL[0]-hpR[0], dy = hpL[1]-hpR[1];
      const len = Math.hypot(dx, dy);
      hipRoll = (dx > 0 && len > 1e-4) ? clamp(Math.asin(clamp(dy/len,-1,1))*.5, -.35, .35) : 0;
    }
    /* 侧倾分配到脊柱三段：躯干逐节弯曲而不是整块硬板。
       视频看到的是「肩线在画面里的倾角」，它由身体绕前向轴的侧倾产生，
       但灵敏度随转身角 cos(ψ) 衰减 —— 按 tan(倾角)·cos(ψ) 反解，才能在转身段也复现同一倾角。 */
    const kYaw = (()=>{ const c=Math.cos(bodyYaw); return Math.abs(c)<.03 ? (c<0?-.03:.03) : c; })();
    chestRoll = clamp(Math.atan(Math.tan(chestRoll)*kYaw), -.6, .6);
    hipRoll = clamp(Math.atan(Math.tan(hipRoll)*kYaw), -.35, .35);
  }
  /* 胸腔跟随（二阶弹簧）：侧倾不是硬板 —— 胸腔滞后于髋部，停下时有落定回弹 */
  const cs = rig.chestSpring || (rig.chestSpring = {x:0, v:0});
  if(poseJump){ cs.x = chestRoll; cs.v = 0; }     /* 时间跳转：直接落到目标，避免拖动时间轴时的假滞后 */
  const kS = 150, dS = 14;
  cs.v += ((chestRoll - cs.x)*kS - cs.v*dS)*dt;
  cs.x += cs.v*dt;
  const rollC = clamp(cs.x, -.6, .6), rollH = clamp(hipRoll, -.35, .35);
  lastChestRoll = rollC; lastHipRoll = rollH;
  poseBoneWorld('hips', new THREE.Quaternion().setFromAxisAngle(ax.fwd, rollH).multiply(yawQ));
  /* 脊柱：链式增量，每节 1/3（绕身体前向轴，换算到各自父级局部系）。
     绝对姿态赋值会在链上叠加各节分数（1/3+2/3+1=2 → 侧倾翻倍），必须用增量。 */
  for(const nm of ['spine1','spine2','chest']){
    const j = rig.nodes.get(nm); if(!j) continue;
    const q0 = rig.initQuat.get(nm) || j.quaternion;
    const parentQ = j.parent.getWorldQuaternion(new THREE.Quaternion());
    const fwdLocal = ax.fwd.clone().applyQuaternion(parentQ.invert()).normalize();
    j.quaternion.copy(new THREE.Quaternion().setFromAxisAngle(fwdLocal, rollC/3)).multiply(q0);
  }
  /* 头部：yaw（鼻−耳中点偏移）+ pitch；二阶弹簧产生跟随滞后 */
  const nr = poseNeutralRef();
  const nose = pose ? pose[MKEY.nose] : null, le = pose ? pose[MKEY.lear] : null, re = pose ? pose[MKEY.rear] : null;
  if(nose && le && re && nose[2] >= .25 && le[2] >= .25 && re[2] >= .25){
    const earDx = (le[0]+re[0])/2, earDy = (le[1]+re[1])/2;
    const half = Math.max(.08, Math.abs(le[0]-re[0])/2);
    const yawAbs = clamp(Math.asin(clamp((nose[0]-earDx-nr.noseDx)/half, -1, 1)), -1.6, 1.6);
    const yaw = clamp(yawAbs - bodyYaw, -.8, .8);       /* 头相对身体 */
    headState.yawAbs = yawAbs;
    /* 俯仰：耳朵常被头发遮住，定位不稳 → 只取小幅度，避免出现「仰头看天」的假动作 */
    const pitch = clamp(Math.asin(clamp(-(nose[1]-earDy-nr.noseDy)/(half*1.6), -1, 1))*.55, -.22, .22);
    const k = 220, damp = 17;
    if(poseJump){ /* 时间跳转：直接落到目标（与胸腔弹簧同口径），否则同一时刻的测量随历史漂移 */
      headState.yaw = yaw; headState.pitch = pitch; headState.vy = headState.vp = 0;
    }else{
      headState.vy += ((yaw - headState.yaw)*k - headState.vy*damp)*dt;
      headState.vp += ((pitch - headState.pitch)*k - headState.vp*damp)*dt;
      headState.yaw += headState.vy*dt;
      headState.pitch += headState.vp*dt;
    }
  }
  poseBoneWorld('head', new THREE.Quaternion().setFromAxisAngle(ax.up, headState.yaw)
    .multiply(new THREE.Quaternion().setFromAxisAngle(ax.left, headState.pitch)).multiply(yawQ));
  /* 发帘跟随：头部位移驱动发梢摆动（发骨不存在的替代，二阶弹簧近似软体）。
     只驱动发帘组 hairSwing，头皮罩与刘海刚性跟随头骨。 */
  const headBone = rig.nodes.get('head');
  const hairNode = headBone && headBone.getObjectByName('hairSwing');
  if(hairNode){
    const hwp = headBone.getWorldPosition(new THREE.Vector3());
    const st = rig.hairState || (rig.hairState = {px:0, pz:0, x:0, z:0, vx:0, vz:0, init:false});
    if(!st.init){ st.px = hwp.x; st.pz = hwp.z; st.init = true; }
    const hv = dt > 0 ? {x:(hwp.x-st.px)/dt, z:(hwp.z-st.pz)/dt} : {x:0,z:0};
    st.px = hwp.x; st.pz = hwp.z;
    const tx = clamp(-hv.z*.045, -.38, .38), tz = clamp(-hv.x*.045, -.38, .38);
    st.vx += ((tx - st.x)*95 - st.vx*13)*dt; st.x += st.vx*dt;
    st.vz += ((tz - st.z)*95 - st.vz*13)*dt; st.z += st.vz*dt;
    hairNode.rotation.set(st.x, 0, st.z);
  }
  /* 四肢目标：方向重定向（关节方向 × 模型肢体长度）；弯折平面由视频肘/膝给出。
     有 3D 世界坐标时（MediaPipe）：肘/膝的真实弯折角与弯折面直接取自 3D ——
     2D 投影无法还原「朝镜头的弯折」（双手举在锁骨时 2D 肘角 90°，真实弯折朝前），
     用余弦定理反解会把手臂拉长一截。 */
  for(const [tag, wi, ei, chain, anchorName] of [
    ['L', MKEY.lwr, MKEY.lel, ['shoulderL','elbowL','handL'], 'shoulderL'],
    ['R', MKEY.rwr, MKEY.rel, ['shoulderR','elbowR','handR'], 'shoulderR']]){
    const w = pose ? pose[wi] : null;
    const t = sm('hand'+tag, (w && w[2] >= .22) ? [w[0], w[1]] : null, .06, dt);
    if(!t || !shMid) continue;
    const anchor = rig.nodes.get(anchorName);
    const mu = rig.meta.get(chain[0]), ml = rig.meta.get(chain[1]);
    if(!anchor || !mu || !ml) continue;
    const anchorW = anchor.getWorldPosition(new THREE.Vector3());
    let target = null, bend = null, elbowHint = null;
    /* v3 路径：透视缩短重构的三维偏移（相机系→躯干系→世界）。
       手落在视频的手上（x/y 即观测），深度来自重构；肘的观测位只定弯折面法向。
       链上三骨（肩/肘/腕）任一不可信（投影长度越解剖区间）→ 深度会放大 2D 噪声：
       此时**保持上一次可信深度**（两路径收敛，不产生半米级跳变）。 */
    const p3ok = conf3d !== null && conf3d[wi] >= 1 && conf3d[ei] >= 1 && conf3d[tag==='L'?MKEY.lsh:MKEY.rsh] >= 1;
    const p3 = pose3d;
    if(p3 && p3[tag==='L'?MKEY.lsh:MKEY.rsh] && p3[ei] && p3[wi]){
      const sh3 = p3[tag==='L'?MKEY.lsh:MKEY.rsh], el3 = p3[ei], wr3 = p3[wi];
      const s = sourceTorsoScale();
      const comp = (rig.torsoHip && s > rig.torsoHip) ? (s - rig.torsoHip) : 0;
      const trusted = gate3d('arm'+tag, p3ok);
      const hold = rig.zhold || (rig.zhold = {});
      const hk = 'arm'+tag;
      const zWraw = wr3[2]-sh3[2], zEraw = el3[2]-sh3[2];
      if(trusted) hold[hk] = {w: zWraw, e: zEraw};
      const hh = hold[hk] || {w: 0, e: 0};
      const zW = sm1('armZ'+tag, trusted ? zWraw : hh.w, .10, dt);
      const zE = sm1('armZe'+tag, trusted ? zEraw : hh.e, .10, dt);
      /* 偏移先过一阶低通：深度对 2D 噪声敏感（√(L²−d²) 在 d→L 病态） */
      const offW = sm3('armOff'+tag, bodyVecToWorld(camToBodyVec(
        [wr3[0]-sh3[0], wr3[1]-sh3[1], zW], bodyYaw), ax), .10, dt);
      offW.addScaledVector(ax.up, comp);           /* 肩高补偿：模型的肩比视频低一截 */
      const maxR = (mu.worldLen + ml.worldLen)*.98;
      const len = offW.length();
      if(len > 1e-6){
        const reach = smoothReach('arm'+tag, clamp(len, mu.worldLen*.15, maxR), dt, poseJump);
        target = anchorW.clone().addScaledVector(offW.clone().multiplyScalar(1/len), reach);
        const offE = sm3('armElb'+tag, bodyVecToWorld(camToBodyVec(
          [el3[0]-sh3[0], el3[1]-sh3[1], zE], bodyYaw), ax), .10, dt);
        offE.addScaledVector(ax.up, comp);
        elbowHint = anchorW.clone().add(offE);
        bend = null;
      }
    }
    if(!target){
      /* 末端位置用**画面上的肩→腕距离**：手臂朝镜头弯（双手举在锁骨那种）时，
         投影肘角约 90°，余弦定理会把手臂拉长一截；直接用画面距离才能让人偶的手
         落在视频里手的位置。2D 单位=躯干长，1 躯干长 = sourceTorsoScale() 米。
         叠加**肩高补偿**：模型的肩关节比源视频的肩关键点低（Mixamo 骨盆偏高、
         躯干偏短），差 = 源肩高 − 模型肩→髋关节距离；不补偿手会整体偏低一截。 */
      const shRef = pose ? pose[tag==='L'?MKEY.lsh:MKEY.rsh] : null;
      const ox = shRef ? t[0]-shRef[0] : t[0]-shMid[0];
      const oy = shRef ? t[1]-shRef[1] : t[1]-shMid[1];
      const maxR = (mu.worldLen + ml.worldLen)*.98;
      const s = sourceTorsoScale();
      const off = new THREE.Vector3().addScaledVector(ax.left, ox*s).addScaledVector(ax.up, oy*s);
      if(rig.torsoHip && s > rig.torsoHip) off.addScaledVector(ax.up, s - rig.torsoHip);
      const len = off.length();
      if(len < 1e-6) continue;
      const reach = smoothReach('arm'+tag, clamp(len, mu.worldLen*.15, maxR), dt, poseJump);
      target = anchorW.clone().addScaledVector(off.multiplyScalar(1/len), reach);
      bend = resolveBend('arm'+tag, pose ? bendObs2D(pose, tag==='L'?MKEY.lsh:MKEY.rsh, ei, wi) : null, ax, -.25, dt);
    }
    if(!rig.lastTarget) rig.lastTarget = {};
    /* 穿模兜底：整条手臂推出躯干。沿**相机射线**推 —— 针孔投影下射线上的点
       投影相同，画面位置零变化 → 视频一致性不受影响；射线由规格+视频基准解算，
       与当前渲染视图无关（换镜头不改变动作）。 */
    /* 手臂先按视频解算；穿模收敛（让位 + 推出）在两腿解完之后做第二遍 ——
       第一遍时躯干/髋的位置还没定稿，代理位置是上一帧的，推出量会算错。 */
    solveChain(chain, target, bend, elbowHint);
    rig.armState = rig.armState || {};
    rig.armState[tag] = {bend, elbowHint, target};
    rig.lastTarget['hand'+tag] = target;
  }
  for(const [tag, ai, ki, chain, anchorName] of [
    ['L', MKEY.lan, MKEY.lkn, ['hipL','kneeL','ankleL'], 'hipL'],
    ['R', MKEY.ran, MKEY.rkn, ['hipR','kneeR','ankleR'], 'hipR']]){
    const a = pose ? pose[ai] : null;
    const planted = footPlanted(tag, t);
    const ft = sm('foot'+tag, (a && a[2] >= .18 && !planted) ? [a[0], a[1]] : null, .09, dt);
    if(!ft) continue;
    const anchor = rig.nodes.get(anchorName);
    const mu = rig.meta.get(chain[0]), ml = rig.meta.get(chain[1]);
    if(!anchor || !mu || !ml) continue;
    if(!rig.lastTarget) rig.lastTarget = {};
    const prev = rig.lastTarget['ankle'+tag];
    /* 踩实：世界坐标锁定（只锁方向仍会随膝角沿射线滑动） */
    let target = null, bend = null, kneeHint = null;
    if(!rig.footState) rig.footState = {};
    const fstate = rig.footState[tag] || (rig.footState[tag] = {lock:null, blend:0});
    if(planted && prev){
      /* 锁定的脚若因身体移动而超出腿长，钳到可达球面（否则 IK 够不着 → 腿飘） */
      const anchorW0 = anchor.getWorldPosition(new THREE.Vector3());
      const d = prev.clone().sub(anchorW0);
      const maxR0 = (mu.worldLen + ml.worldLen)*.995;
      target = d.length() > maxR0 ? anchorW0.clone().addScaledVector(d.normalize(), maxR0) : prev.clone();
      fstate.lock = target.clone(); fstate.blend = 0;
    }
    if(!target){
      const anchorW = anchor.getWorldPosition(new THREE.Vector3());
      let ikTarget = null;
      /* v3 路径：透视缩短重构的三维偏移（含深度）—— 膝角与视频的 3D 膝角一致，
         末端落点即观测；踝深由重构给出（膝前/踝后）。深度不可信时保持上次可信值，
         偏移过一阶低通。 */
      const p3lok = conf3d !== null && conf3d[ai] >= 1 && conf3d[ki] >= 1 && conf3d[tag==='L'?MKEY.lhp:MKEY.rhp] >= 1;
      if(pose3d && pose3d[tag==='L'?MKEY.lhp:MKEY.rhp] && pose3d[ki] && pose3d[ai]){
        const hp3 = pose3d[tag==='L'?MKEY.lhp:MKEY.rhp], kn3 = pose3d[ki], an3 = pose3d[ai];
        const trusted = gate3d('leg'+tag, p3lok);
        const hold = rig.zhold || (rig.zhold = {});
        const hk = 'leg'+tag;
        const zAraw = an3[2]-hp3[2], zKraw = kn3[2]-hp3[2];
        if(trusted) hold[hk] = {a: zAraw, k: zKraw};
        const hh = hold[hk] || {a: 0, k: 0};
        const zA = sm1('legZ'+tag, trusted ? zAraw : hh.a, .10, dt);
        const zK = sm1('legZk'+tag, trusted ? zKraw : hh.k, .10, dt);
        const offA = sm3('legOff'+tag, bodyVecToWorld(camToBodyVec(
          [an3[0]-hp3[0], an3[1]-hp3[1], zA], bodyYaw), ax), .10, dt);
        const maxR = (mu.worldLen + ml.worldLen)*.995;
        const len = offA.length();
        if(len > 1e-6){
          const reach = smoothReach('leg'+tag, clamp(len, mu.worldLen*.15, maxR), dt, poseJump);
          ikTarget = anchorW.clone().addScaledVector(offA.clone().multiplyScalar(1/len), reach);
          const offK = sm3('legKne'+tag, bodyVecToWorld(camToBodyVec(
            [kn3[0]-hp3[0], kn3[1]-hp3[1], zK], bodyYaw), ax), .10, dt);
          kneeHint = anchorW.clone().add(offK);
          bend = null;
        }
      }
      if(!ikTarget){
        /* 末端距离用**膝角**反解（余弦定理 + 模型骨长），不用画面上的髋→踝距离：
           画面距离是投影，必然短于真实腿长（相机俯视、腿离画面平面），
           直接当伸展量会把腿压成深蹲（膝角偏离视频 60° 以上）；
           用膝角则膝关节夹角与视频一致、腿保持站直，髋高也自洽。 */
        const hipRef = pose ? pose[tag==='L'?MKEY.lhp:MKEY.rhp] : null;
        const L1 = mu.worldLen, L2 = ml.worldLen;
        const th = tag==='L' ? kneeAngL : kneeAngR;
        const reach = th != null
          ? Math.sqrt(Math.max(1e-4, L1*L1 + L2*L2 - 2*L1*L2*Math.cos(th)))
          : Math.hypot(hipRef ? ft[0]-hipRef[0] : ft[0], hipRef ? ft[1]-hipRef[1] : ft[1])*sourceTorsoScale();
        const maxR = (L1 + L2)*.995;
        const dirW = new THREE.Vector3().addScaledVector(ax.left, ft[0]).addScaledVector(ax.up, ft[1]);
        if(dirW.lengthSq() < 1e-8) continue;
        ikTarget = anchorW.clone().addScaledVector(dirW.normalize(),
          clamp(reach, mu.worldLen*.15, maxR));
        bend = resolveBend('leg'+tag, pose ? bendObs2D(pose, tag==='L'?MKEY.lhp:MKEY.rhp, ki, ai) : null, ax, .35, dt);
      }else{
        smoothReach('leg'+tag, ikTarget.distanceTo(anchorW), dt, poseJump);
      }
      /* 刚解除踩实：从锁定位置平滑过渡到 IK 目标（直接切会产生十几厘米的跳变） */
      if(fstate.lock){
        fstate.blend = Math.min(1, fstate.blend + dt/.12);
        target = fstate.lock.clone().lerp(ikTarget, fstate.blend);
        if(fstate.blend >= 1) fstate.lock = null;
      }else target = ikTarget;
    }
    rig.lastTarget['ankle'+tag] = target;
    if(!bend && !kneeHint) bend = resolveBend('leg'+tag, pose ? bendObs2D(pose, tag==='L'?MKEY.lhp:MKEY.rhp, ki, ai) : null, ax, .35, dt);
    solveChain(chain, target, bend, kneeHint);
    /* 抬膝顶到躯干时让出 —— **暂不启用**：弯折面旋转会改变膝的落点，
       实测把 E3（膝夹角保真）与 E7（弯折侧向）带偏 26°/4cm。
       膝与躯干几乎不互穿（视频里腿不交叉），收益远小于代价；
       需要时只对**确实穿透**的帧生效（阈值收紧到 >2cm 才让位）。 */
  }
  /* 掌向扭转（第一次）：在穿模收敛**之前**扭一次 ——
     手指朝向决定指尖会不会插进躯干/头部，不先扭好，穿模判据量的是没扭的手 ✗。
     扭转是幂等的（按当前法线与目标的夹角做增量旋转），收敛后再扭一次即可复原。 */
  twistPalms(ax);
  /* ---------- 穿模收敛（第二遍）：躯干与腿的位置已定稿 ---------- */
  /* 两个自由度交替收敛：
     ① 弯折面让位（肘在弦平面内的方位不可观测 → 绕弦轴旋转把上臂/前臂让出躯干）；
     ② 整条臂沿相机射线推出（针孔投影下射线上的点投影相同 → 画面位置零变化）。 */
  if(rig.armState){
    /* 身体的网格真值（逐帧一次）：代理是包络，既可能偏大也可能偏小
       （实测腕部代理判"在外"、网格判"在内"4.5cm），量级必须以网格为准。 */
    const smBody = figGroup && figGroup.getObjectByProperty('isSkinnedMesh', true);
    let bodyTruth = null, camPos = null;
    if(smBody && smBody.geometry.index){
      if(figGroup) figGroup.updateMatrixWorld(true);
      const topo = bodyTopology(smBody);
      if(topo) bodyTruth = bodyWorldNow(smBody, topo);
      const vc = videoCamState();
      camPos = (vc && vc.pos) ? vc.pos : camera.position;
    }
    for(const tag of ['L','R']){
      const st = rig.armState[tag];
      if(!st || !st.target) continue;
      const chain = tag==='L' ? ['shoulderL','elbowL','handL'] : ['shoulderR','elbowR','handR'];
      solveChain(chain, st.target, st.bend, st.elbowHint);
      /* 穿模收敛（真值优先）：
         ① 弯折面让位 —— 肘在弦平面内的方位**不可观测**（正视图看不出），
            把它转开是零代价的；候选用真值判据挑（代理会漏判）。
         ② 沿相机射线推出 —— 针孔投影下射线上的点投影相同，画面位置零变化。
            只在最深点位于**手**这一端时有效（推手才动得了手；肘/上臂深陷要靠①）。 */
      const truth = bodyTruth ? {body:bodyTruth, camPos} : null;
      const wp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
      const diag = (rig.pushDiag = rig.pushDiag || {})[tag] = {off0: chainBendOff(chain)};
      if(truth){
        for(let iter=0; iter<3; iter++){
          const t0 = chainTruthPen(chain, tag, truth);
          rig.pushTruth = rig.pushTruth || {};
          rig.pushTruth[tag] = {worst:+t0.pen.toFixed(4), idx:t0.idx, iter};
          if(t0.pen <= 0.002) break;
          /* ① 沿相机射线推手：肘/前臂也会跟着前移，深陷在肘/前臂时同样有效。
             深度符号本身不可观测（手可能在躯干前，也可能在躯干后）⇒ **两个方向都试**，
             取真正让穿透变小的那一个。 */
          const dir = camPos.clone().sub(st.target).normalize();
          /* 推出量受**手臂可达范围**约束：目标够不到时 IK 只能拉直手臂，
             肘会离开弦线很远（弯折几何自相矛盾）。 */
          const L1 = (rig.meta.get(chain[0])||{}).worldLen || 0, L2 = (rig.meta.get(chain[1])||{}).worldLen || 0;
          const sh = wp(chain[0]);
          const reach = (L1 + L2) * 0.995;
          const cur = sh ? st.target.distanceTo(sh) : 0;
          const need0 = Math.min(MAX_PUSH, t0.pen + 0.02);   /* 表面穿透量 + 余量 */
          const prev = st.target.clone();
          const side0 = chainBendOff(chain);    /* 本轮推出前的弯折侧向（正 = 弯折点在弦前） */
          let best = null;
          for(const sgn of [1, -1]){
            let need = need0;
            if(sh && reach > 0 && cur + need > reach) need = Math.max(0, reach - cur);
            if(need <= 1e-4) continue;
            const cand = prev.clone().addScaledVector(dir, need*sgn);
            /* 推出时弦会转 → 弯折面必须**随弦搬运**，否则肘翻到弦的另一侧（E7） */
            const pole = transportedPole(chain, cand);
            solveChain(chain, cand, st.bend, pole || st.elbowHint);
            /* 搬运是刚体的：弦转过 90° 以上时侧向仍会翻 —— 这种推出直接放弃，
               宁可留一点穿透（可见手部几何）也不画成反关节。 */
            const off = chainBendOff(chain);
            if(side0 !== null && off !== null && Math.sign(off) !== Math.sign(side0)) continue;
            const pen = chainTruthPen(chain, tag, truth).pen;
            if(!best || pen < best.pen - 1e-4) best = {cand, pole, pen, need};
          }
          if(best){
            solveChain(chain, best.cand, st.bend, best.pole || st.elbowHint);
            diag.push = +((diag.push||0) + best.need).toFixed(4);
            diag.offPush = chainBendOff(chain);
            /* 推出必须真的让穿透变小，否则回退（否则会一路推到底，把手甩到身前 30cm） */
            if(best.pen < t0.pen - 1e-4){
              st.target = best.cand;
              rig.lastTarget['hand'+tag] = best.cand;
              rig.pushOut['hand'+tag] = +(rig.pushOut['hand'+tag]||0) + best.need;
              continue;
            }
          }
          solveChain(chain, prev, st.bend, st.elbowHint);
          diag.offRevert = chainBendOff(chain);
          /* ② 让位：弯折面转开（肘在弦平面内的方位不可观测 → 零代价）。
             弯折面用**当前面**（随弦搬运后的极向量），不用绝对提示点 ——
             绝对提示点在弦转过之后会泄漏，肘会翻到弦的另一侧（E7 反关节）。 */
          const before = t0.pen;
          const poleNow = transportedPole(chain, st.target);
          resolveChainCollision(chain, st.target, st.bend, poleNow || st.elbowHint, tag, ax, truth);
          let after = chainTruthPen(chain, tag, truth).pen;
          diag.offTilt = chainBendOff(chain);
          if(after >= before - 1e-4) break;
        }
      }else{
        const poleNow = transportedPole(chain, st.target);
        resolveChainCollision(chain, st.target, st.bend, poleNow || st.elbowHint, tag, ax);
        diag.offTilt = chainBendOff(chain);
      }
    }
  }
  /* 掌向扭转（第二次）：收敛重解了手臂链，手骨四元数被覆写，按最终腕点重扭一次。
     掌心相对时需要另一只手的位置作参照，故两侧一起做。 */
  twistPalms(ax);
}

/* 掌向扭转（双臂）：掌心相对时需要另一只手的位置作参照，故两侧一起做 */
function twistPalms(ax){
  const gp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
  const wL = gp('handL'), wR = gp('handR'), eL = gp('elbowL'), eR = gp('elbowR');
  if(wL && eL) applyPalmTwist('L', palmDesiredNormal('L', wL, eL, ax, wR));
  if(wR && eR) applyPalmTwist('R', palmDesiredNormal('R', wR, eR, ax, wL));
}
/* ---------- 掌向：绕前臂轴扭转手腕 ---------- */
/* 掌面法线 ⟂ 前臂轴 ⇒ 掌向的自由度是**纯扭转**（参考项目结论：位置无法承载扭转）。
   世界轴旋转换算到手骨局部四元数：L' = P⁻¹·R·P·L（P = 前臂世界朝向）。 */
function applyPalmTwist(tag, desiredNormal){
  const pa = rig.palmAxis && rig.palmAxis[tag];
  if(!pa || !desiredNormal) return 0;
  const handB = rig.nodes.get(tag==='L' ? 'handL' : 'handR');
  const elbowB = rig.nodes.get(tag==='L' ? 'elbowL' : 'elbowR');
  if(!handB || !elbowB) return 0;
  const wrist = handB.getWorldPosition(new THREE.Vector3());
  const elbow = elbowB.getWorldPosition(new THREE.Vector3());
  const F = elbow.clone().sub(wrist).normalize();          /* 前臂轴（腕→肘） */
  const N = pa.axis.clone().applyQuaternion(handB.getWorldQuaternion(new THREE.Quaternion())).normalize();
  const Np = N.clone().addScaledVector(F, -N.dot(F));
  const Dp = desiredNormal.clone().addScaledVector(F, -desiredNormal.dot(F));
  if(Np.lengthSq() < 1e-6 || Dp.lengthSq() < 1e-6) return 0;
  Np.normalize(); Dp.normalize();
  const ang = Math.atan2(Np.clone().cross(Dp).dot(F), clamp(Np.dot(Dp), -1, 1));
  if(!Number.isFinite(ang) || Math.abs(ang) < 1e-4) return 0;
  const P = elbowB.getWorldQuaternion(new THREE.Quaternion());
  const R = new THREE.Quaternion().setFromAxisAngle(F, ang);
  handB.quaternion.copy(P.clone().invert().multiply(R).multiply(P).multiply(handB.quaternion.clone()));
  return ang;
}
/* 动作文件里的掌向标签（多模态模型产出）：缺项/低置信 → 退回解剖先验 */
export function palmLabelAt(t, tag){
  const mo = motion;
  if(!mo || !mo.chan || !mo.chan.palm) return null;
  const arr = mo.chan.palm[tag];
  if(!arr || !arr.length) return null;
  const ft = (t - spec.shot.start)*(mo.fps || 24);
  const i0 = clamp(Math.floor(ft), 0, arr.length-1);
  const i1 = Math.min(arr.length-1, i0+1);
  const f = clamp(ft - i0, 0, 1);
  const p = f < 0.5 ? arr[i0] : arr[i1];
  if(!p || !p.face || p.face === 'none' || p.face === 'unknown') return null;
  if((p.conf ?? 1) < 0.5) return null;
  return p;
}
/* 目标掌面法线：标签优先，先验兜底。
   先验：双手靠近且在胸腹高 → 掌心相对；前臂朝前 → 掌心朝前；否则掌心朝身体中线。 */
export function palmDesiredNormal(tag, W, E, ax, otherW){
  const cam = videoCamState();
  const toCam = cam ? cam.pos.clone().sub(W).normalize() : ax.fwd.clone();
  const inward = (tag === 'L' ? ax.left.clone().negate() : ax.left.clone());
  const lab = palmLabelAt(playhead, tag);
  if(lab){
    if(lab.face === 'fwd') return toCam;
    if(lab.face === 'away') return toCam.clone().negate();
    if(lab.face === 'body') return inward;
    if(lab.face === 'other' && otherW) return otherW.clone().sub(W);
  }
  const F = E.clone().sub(W).normalize();
  if(otherW && W.distanceTo(otherW) < .22) return otherW.clone().sub(W);
  if(F.dot(ax.fwd) > .45) return ax.fwd.clone();
  return inward;
}

/* 无动作文件时的站姿：双臂自然下垂、双腿站直（同一套 IK 驱动） */
function applyRestPose(){
  if(!rig || !figGroup) return;
  rig.root.updateMatrixWorld(true);
  const ax = rigAxes();
  const hips = rig.nodes.get('hips');
  const hipW = hips.getWorldPosition(new THREE.Vector3());
  const thighLen = rig.meta.get('hipL')?.worldLen || .3, shinLen = rig.meta.get('kneeL')?.worldLen || .3;
  const armLen = (rig.meta.get('shoulderL')?.worldLen || .2) + (rig.meta.get('elbowL')?.worldLen || .2);
  const down = hipW.clone().addScaledVector(ax.up, -(thighLen+shinLen));
  solveChain(['hipL','kneeL','ankleL'], down.clone().addScaledVector(ax.left, .015), ax.fwd);
  solveChain(['hipR','kneeR','ankleR'], down.clone().addScaledVector(ax.left, -.015), ax.fwd);
  const hand = hipW.clone().addScaledVector(ax.up, -armLen*.92).addScaledVector(ax.left, .02);
  solveChain(['shoulderL','elbowL','handL'], hand, ax.fwd);
  const handR = hipW.clone().addScaledVector(ax.up, -armLen*.92).addScaledVector(ax.left, -.02);
  solveChain(['shoulderR','elbowR','handR'], handR, ax.fwd);
}
export function resetFigurePose(){
  if(!rig || !figGroup) return;
  motionTargets = new Map();
  motionTargets3 = new Map();
  rig.footState = {L:{lock:null,blend:0}, R:{lock:null,blend:0}};
  if(rig.bendState) rig.bendState.clear();
  rig.hipHist = [];
  rig.chestSpring = {x:0, v:0};
  headState.yaw = headState.pitch = headState.vy = headState.vp = 0;
  for(const [k,o] of rig.nodes){
    const q = rig.initQuat.get(k);
    if(q) o.quaternion.copy(q);
    if(k === 'hips') o.position.copy(rig.hipsRest);
  }
  applyRestPose();
}
