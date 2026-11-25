import {THREE, spec} from './app/state.js';

export const $ = s => document.querySelector(s);
/* 内核（纯几何）与 three.js 的边界转换 —— 唯一的耦合点 */
export const toVec = v => new Vec3(v.x, v.y, v.z);
export const toThree = v => new THREE.Vector3(v.x, v.y, v.z);
export const clamp = (v,a,b) => Math.min(b, Math.max(a,v));
export const lerp = (a,b,r) => a + (b-a)*r;
export const DEG = Math.PI/180;
export const rand = (a,b) => a + Math.random()*(b-a);
export const pick = arr => arr[Math.floor(Math.random()*arr.length)];

/* ---------- 错误可见化 ---------- */
export function showError(title, msg, detail){
  $('#errTitle').textContent = title;
  $('#errMsg').textContent = msg || '';
  $('#errDetail').textContent = detail || '';
  $('#err').hidden = false;
}
window.addEventListener('error', e => showError('运行时错误', e.message, `${e.filename}:${e.lineno}`));
window.addEventListener('unhandledrejection', e => showError('异步异常', String(e.reason?.message || e.reason), e.reason?.stack || ''));
$('#errRetry').onclick = () => location.reload();
$('#errAlt').onclick = () => { const u = new URL(location.href); u.searchParams.set('cdn','unpkg'); location.href = u.toString(); };

/* ---------- 人偶造型常量（无动作文件时的回退比例；取自源视频帧） ---------- */
export const FIG = {
  hipY:0.52, chestY:0.80, headY:0.925, headR:0.062,
  sweaterTop:0.825, sweaterBottom:0.44, sleeveR:0.085,
  upperArm:0.165, foreArm:0.145, armOut:50*DEG, armFore:95*DEG, armFwd:8*DEG,
  thigh:0.25, shin:0.23, footH:0.055, legR:0.055, hairBottom:0.72,
  liftThigh:80*DEG, liftShin:30*DEG, headTilt:6*DEG,
  sweater:0x26262e, jeans:0x9fb0c8, skin:0xe8c3a8, hair:0x141419
};
/* 源视频测得的画面轴向相对规格 pitch_deg 的偏移（用于"对齐视频构图"） */
export const VIDEO_PITCH_ADJ = 12;

/* ---------- 规格归一化 ---------- */
const PART_RULES = [
  {kind:'head', h:0.93, k:['head','face','hair','skull','头','脸','发']},
  {kind:'arm',  h:0.78, k:['arm','hand','elbow','forearm','shoulder','肩','手','臂','肘']},
  {kind:'torso',h:0.55, k:['torso','chest','waist','hip','pelvis','body','腰','胯','躯干','胸']},
  {kind:'leg',  h:null, k:['leg','腿','foot','shoe','脚','鞋']},
  {kind:'leg',  h:0.35, k:['thigh','大腿']},
  {kind:'leg',  h:0.25, k:['knee','膝']},
  {kind:'leg',  h:0.15, k:['shin','小腿']}
];
export const TRAIL_COLORS = [0x4dd0e1,0xffb74d,0xaed581,0xf06292,0x9575cd,0x4fc3f7,0xff8a65,0x81c784];

function normSubject(s, i){
  const path = (s.path||[]).map(p => ({
    t: Number(p.t)||0,
    /* schema +x = 相机右；相机朝 +Z 时屏幕右 = 世界 −X，故取负镜像入世界系 */
    x: -(Number(p.pos_m?.[0])||0),
    d: Number(p.pos_m?.[1])||0,
    facing: p.facing_deg === undefined ? null : Number(p.facing_deg),
    posture: p.posture || s.posture || 'stand'
  })).sort((a,b) => a.t-b.t);
  return {
    id: s.id ?? `S${i+1}`,
    cls: String(s.class||'').toLowerCase(),
    desc: String(s.desc||'').toLowerCase(),
    size: (s.size_m && s.size_m.length===3) ? s.size_m.map(Number) : [0.4,0.4,1],
    posture: s.posture || 'stand',
    path
  };
}
function normSet(s){
  return {
    cls: String(s.class||'other'),
    x: -(Number(s.pos_m?.[0])||0),
    d: Number(s.pos_m?.[1])||0,
    size: (s.size_m && s.size_m.length===3) ? s.size_m.map(Number) : [1,1,1],
    base: s.base_m === undefined ? 0 : Number(s.base_m),
    yaw: Number(s.yaw_deg)||0
  };
}
function pickPrimary(subs){
  const person = subs.find(s => s.cls.includes('person'));
  if(person) return person;
  return subs.reduce((a,b) => (!a || b.size[2] > a.size[2]) ? b : a, null);
}
export function normalize(raw){
  const spec = {
    shot:{start:0, end:0, ...(raw.shot||{})},
    camera:{height_m:1.2, angle:'eye', pitch_deg:0, fov_deg:50, move:'static', framing:'full', subject:null, ...(raw.camera||{})},
    subjects:(raw.subjects||[]).map(normSubject),
    set:(raw.set||[]).map(normSet),
    events:Array.isArray(raw.events) ? raw.events : [],
    environment:{type:'indoor', ground:'floor', ceiling:false, sky:false, ...(raw.environment||{})}
  };
  spec.shot.start = Number(spec.shot.start)||0;
  const maxT = spec.subjects.reduce((m,s) => s.path.reduce((a,p) => Math.max(a,p.t), m), spec.shot.start);
  const end = Number(spec.shot.end);
  spec.shot.end = Number.isFinite(end) && end > spec.shot.start ? end : maxT;
  spec.camera.height_m = Number(spec.camera.height_m)||0;
  spec.camera.pitch_deg = Number(spec.camera.pitch_deg)||0;
  spec.camera.fov_deg = Number(spec.camera.fov_deg)||50;
  spec.primary = pickPrimary(spec.subjects);
  spec.partKind = {}; spec.partHeight = {};
  const H = spec.primary ? spec.primary.size[2] : 1;
  spec.subjects.forEach(s => {
    if(s === spec.primary){ spec.partKind[s.id]='body'; spec.partHeight[s.id]=0; return; }
    const text = `${s.cls} ${s.desc}`;
    const rule = PART_RULES.find(r => r.k.some(k => text.includes(k)));
    spec.partKind[s.id] = rule ? rule.kind : 'prop';
    spec.partHeight[s.id] = rule ? (rule.h === null ? null : rule.h*H) : null;
  });
  spec.parts = spec.subjects.filter(s => s !== spec.primary && spec.partKind[s.id] !== 'prop');
  return spec;
}

/* ---------- 采样 ---------- */
function lerpAngle(a,b,r){ const d = (((b-a)%360)+540)%360-180; return a + d*r; }
export function samplePath(path, t){
  if(!path.length) return {t,x:0,d:0,facing:null,posture:'stand'};
  if(t <= path[0].t) return {...path[0]};
  const last = path[path.length-1];
  if(t >= last.t) return {...last};
  for(let i=0;i<path.length-1;i++){
    const a=path[i], b=path[i+1];
    if(t >= a.t && t <= b.t){
      const r = (b.t-a.t) === 0 ? 0 : (t-a.t)/(b.t-a.t);
      return {t, x:lerp(a.x,b.x,r), d:lerp(a.d,b.d,r),
        facing:(a.facing===null||b.facing===null) ? (a.facing ?? b.facing) : lerpAngle(a.facing,b.facing,r),
        posture: r<0.5 ? a.posture : b.posture};
    }
  }
  return {...last};
}

/* ---------- 坐标约定 ---------- */
/* 世界系：相机首帧在原点朝 +Z（屏幕右 = 世界 −X），schema x 已在入料时镜像。
   人偶局部：+Z=面朝方向，+X=自身左侧（面向 +Z、up +Y 时右手在 −X）。 */
export const yawOf = deg => Math.atan2(-Math.cos(deg*DEG), Math.sin(deg*DEG));

/* ---------- three.js 加载（备用 CDN） ---------- */
export const CDN_BASES = [
  window.__CDN,
  {three:'https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js', addons:'https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/'},
  {three:'https://unpkg.com/three@0.164.1/build/three.module.js', addons:'https://unpkg.com/three@0.164.1/examples/jsm/'}
].filter((b,i,a) => a.findIndex(x => x.three===b.three) === i);
async function loadControls(base){
  try{ return (await import(base.addons+'controls/OrbitControls.js')).OrbitControls; }
  catch(e){
    const res = await fetch(base.addons+'controls/OrbitControls.js');
    if(!res.ok) throw new Error('OrbitControls HTTP '+res.status);
    const src = await res.text();
    const url = URL.createObjectURL(new Blob([src.replace(/(['"])three\1/g, `'${base.three}'`)], {type:'text/javascript'}));
    try{ return (await import(url)).OrbitControls; } finally{ URL.revokeObjectURL(url); }
  }
}
export async function loadThree(){
  const errs = [];
  for(const base of CDN_BASES){
    try{
      const THREE = await import(base.three);
      return {THREE, OrbitControls: await loadControls(base)};
    }catch(e){ errs.push(`${base.three}\n  ${e?.message||e}`); }
  }
  throw new Error('three.js CDN 全部不可达：\n'+errs.join('\n'));
}
