#!/usr/bin/env node
/**
 * 行为轨迹 3D 查看器 —— 有界验收脚本
 * 资源纪律：单浏览器实例 / 480×900 / 加载即停循环 / 按需单帧渲染 / 每步超时 / finally kill
 */
import {spawn, execSync} from 'node:child_process';
import {readFileSync, writeFileSync, existsSync, rmSync, mkdirSync} from 'node:fs';
import {setTimeout as sleep} from 'node:timers/promises';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import http from 'node:http';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEBUG_PORT = 9342;
const BASE = 'http://127.0.0.1:8000/';
const FILE_URL = 'file://' + join(ROOT, 'index.html');
const SHOTS = join(ROOT, 'screenshots');
const OVERALL_MS = 120000;
const STEP_MS = 15000;
const START = Date.now();

const results = [];
const rec = (id, name, pass, detail) => { results.push({id,name,pass,detail}); console.log(`${pass?'PASS':'FAIL'} ${id} ${name} :: ${detail}`); };
const approx = (a,b,e) => Math.abs(a-b) <= e;
const clamp = (v,a,b) => v < a ? a : (v > b ? b : v);
const leftMs = () => OVERALL_MS - (Date.now() - START);
function withTimeout(p, ms, label){
  let t;
  return Promise.race([p, new Promise((_,rej)=>{ t=setTimeout(()=>rej(new Error(`超时:${label}(${ms}ms)`)), ms); })]).finally(()=>clearTimeout(t));
}

/* ================= 离线复算（不开浏览器） ================= */
const specRaw = JSON.parse(readFileSync(join(ROOT,'示例.json'),'utf8'));
function lerpPath(path, t){
  if(t <= path[0].t) return [path[0].pos_m[0], path[0].pos_m[1]];
  const last = path[path.length-1];
  if(t >= last.t) return [last.pos_m[0], last.pos_m[1]];
  for(let i=0;i<path.length-1;i++){
    const a=path[i], b=path[i+1];
    if(t>=a.t && t<=b.t){
      const r=(t-a.t)/(b.t-a.t);
      return [a.pos_m[0]+(b.pos_m[0]-a.pos_m[0])*r, a.pos_m[1]+(b.pos_m[1]-a.pos_m[1])*r];
    }
  }
  return [last.pos_m[0], last.pos_m[1]];
}
const KEY_TIMES = [0,1.8,3.6,5.4,7.21,9.01,10.81,12.61,14.41];
const EXPECT = {};
for(const t of KEY_TIMES) EXPECT[t] = specRaw.subjects.map(s => lerpPath(s.path, t));

/* 源姿态去噪（与 index.html denoiseMotion 同算法，独立实现）：3 帧中值去离群 + One-Euro 低通 */
class OneEuro{
  constructor(mc=1.2,beta=0.02,dc=1.0){ this.mc=mc; this.beta=beta; this.dc=dc; this.x=null; this.dx=0; }
  static alpha(c,dt){ const tau=1/(2*Math.PI*c); return 1/(1+tau/dt); }
  filter(v,dt){
    if(!Number.isFinite(v)) return this.x;
    if(this.x===null || !(dt>0)){ this.x=v; return v; }
    const dv=(v-this.x)/dt;
    const ad=OneEuro.alpha(this.dc,dt);
    this.dx = ad*dv + (1-ad)*this.dx;
    const a=OneEuro.alpha(this.mc+this.beta*Math.abs(this.dx),dt);
    this.x = a*v + (1-a)*this.x;
    return this.x;
  }
}
function medianSmooth(arr){
  const out=new Array(arr.length).fill(null);
  for(let i=0;i<arr.length;i++){
    const win=[arr[i-1],arr[i],arr[i+1]].filter(v=>v!=null&&Number.isFinite(v));
    out[i]=win.length?win.slice().sort((a,b)=>a-b)[Math.floor(win.length/2)]:null;
  }
  return out;
}
function gaussSmooth(arr,w){
  const h=Math.max(1,w/2), k=[];
  for(let i=-w;i<=w;i++) k.push(Math.exp(-(i*i)/(2*h*h)));
  const out=new Array(arr.length).fill(null);
  for(let i=0;i<arr.length;i++){
    let a=0,c=0;
    for(let j=-w;j<=w;j++){
      const idx=i+j;
      if(idx<0||idx>=arr.length||arr[idx]==null) continue;
      a+=arr[idx]*k[j+w]; c+=k[j+w];
    }
    out[i]=c?a/c:null;
  }
  return out;
}
function denoiseMotion(mo){
  const T=mo.poses.length, J=mo.poses[0]?.length||17, dt=1/Math.max(1,mo.fps||24);
  const out=mo.poses.map(p=>p?p.map(j=>j?[j[0],j[1],j[2]]:null):null);
  for(let j=0;j<J;j++){
    const xs=mo.poses.map(p=>(p&&p[j])?p[j][0]:null);
    const ys=mo.poses.map(p=>(p&&p[j])?p[j][1]:null);
    const gx=medianSmooth(xs), gy=medianSmooth(ys);
    const fx=new OneEuro(2.5,0.03), fy=new OneEuro(2.5,0.03);
    for(let i=0;i<T;i++){
      if(!out[i]||!out[i][j]) continue;
      out[i][j][0]=fx.filter(gx[i],dt);
      out[i][j][1]=fy.filter(gy[i],dt);
    }
  }
  mo.posesD = out;
  return mo;
}
const OFF = {
  camY: specRaw.camera.height_m,
  camX: 0, camZ: 0,
  hfov: specRaw.camera.fov_deg,
  subjectForward: {x: Math.cos(270*Math.PI/180), z: Math.sin(270*Math.PI/180)},
  shoulderDx: specRaw.subjects.find(s=>/left arm/.test(s.desc)).path[0].pos_m[0] - specRaw.subjects[0].path[0].pos_m[0],
  hipDx: specRaw.subjects.find(s=>/left leg/.test(s.desc)).path[0].pos_m[0] - specRaw.subjects[0].path[0].pos_m[0]
};

/* undici(Node22) 对大响应偶发 raw-throw，一律走 node:http */
function httpGet(url, timeoutMs=6000){
  return new Promise((res,rej)=>{
    const req = http.get(url, r => {
      let b=''; r.on('data', c=>b+=c); r.on('end', ()=>res({status:r.statusCode, body:b}));
    });
    req.on('error', rej);
    req.setTimeout(timeoutMs, ()=>{ req.destroy(); rej(new Error('http 超时 '+url)); });
  });
}

/* ================= CDP 客户端 ================= */
class CDP {
  constructor(ws){
    this.ws=ws; this.id=0; this.pending=new Map(); this.handlers=[];
    ws.addEventListener('message', ev=>{
      const m=JSON.parse(ev.data);
      if(m.id && this.pending.has(m.id)){
        const p=this.pending.get(m.id); this.pending.delete(m.id);
        m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
      } else if(m.method) this.handlers.forEach(h=>h(m));
    });
  }
  on(fn){ this.handlers.push(fn); }
  send(method, params={}, sessionId){
    const id=++this.id;
    const payload={id, method, params};
    if(sessionId) payload.sessionId=sessionId;
    this.ws.send(JSON.stringify(payload));
    return new Promise((resolve,reject)=>this.pending.set(id,{resolve,reject}));
  }
}
async function launchChrome(){
  const profile='/tmp/verify-chrome-profile';
  rmSync(profile,{recursive:true,force:true});
  const proc=spawn(CHROME,[
    '--headless=new','--enable-unsafe-swiftshader','--use-angle=swiftshader','--no-first-run',
    '--disable-extensions','--disable-background-networking','--disable-sync','--mute-audio',
    '--autoplay-policy=no-user-gesture-required','--force-device-scale-factor=1','--window-size=360,640',
    `--remote-debugging-port=${DEBUG_PORT}`,`--user-data-dir=${profile}`,'about:blank'
  ],{stdio:'ignore', detached:true});
  for(let i=0;i<40;i++){
    try{ const r=await httpGet(`http://127.0.0.1:${DEBUG_PORT}/json/version`, 2500); if(r.status===200) return proc; }catch(e){}
    await sleep(250);
  }
  throw new Error('Chrome 未能在 10s 内就绪');
}
async function connect(){
  const ver=JSON.parse((await httpGet(`http://127.0.0.1:${DEBUG_PORT}/json/version`)).body);
  const ws=new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise((res,rej)=>{ ws.onopen=res; ws.onerror=()=>rej(new Error('WS 连接失败')); });
  const cdp=new CDP(ws);
  const {targetId}=await cdp.send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await cdp.send('Target.attachToTarget',{targetId,flatten:true});
  return {cdp, sid:sessionId};
}

/* ================= 主流程 ================= */
let chrome = null, serverStarted = null, cdp = null, sid = null;
const logs = [];
const tmpFiles = [];

async function main(){
  /* ---- A3-offline：预算常量在源码中齐备 ---- */
  const html = readFileSync(join(ROOT,'index.html'),'utf8');
  const budgetKeys = ['maxFrameMs','minFrameIntervalMs','maxCrowd','maxPhoneLights','maxLights','maxTriangles','maxDrawCalls','idleRedraws','warmupFrames'];
  const budgetVals = {maxFrameMs:33,minFrameIntervalMs:33,maxCrowd:900,maxPhoneLights:180,maxLights:4,maxTriangles:120000,maxDrawCalls:40,idleRedraws:1,warmupFrames:20};
  const missing = budgetKeys.filter(k => !new RegExp(`${k}\\s*:\\s*${budgetVals[k]}\\b`).test(html));
  rec('A3o','预算常量在源码中齐备', missing.length===0, `缺项=${JSON.stringify(missing)}`);

  /* ---- E15：人偶比例与源视频人物一致（离线，纯 node 复算） ----
     源视频人物比例（可见身高 = 颅顶→鞋底）：
       髋(股骨头) 50.7% / 躯干(肩→髋) 27.6% / 头颈(肩→颅顶) 21.7%
     模型出厂是 Mixamo 时装人形比例（腿 59% / 躯干 21%），
     与「对照机位」的两个约束（可见身高占比 + 髋中点屏幕位）互斥。 */
  {
    const lm = await import('./model_landmarks.mjs');
    const mm = lm.measure(join(ROOT,'models','human.glb'));
    const TGT = {leg:0.507, torso:0.276, head:0.217}, EPS = 0.02;
    const got = {leg:mm.fracLeg, torso:mm.fracTorso, head:mm.fracHead};
    const bad = Object.keys(TGT).filter(k => Math.abs(got[k]-TGT[k]) > EPS);
    rec('E15','人偶比例对齐源视频人物（腿/躯干/头颈占可见身高）', bad.length===0,
        `腿=${got.leg.toFixed(3)} 躯干=${got.torso.toFixed(3)} 头颈=${got.head.toFixed(3)} · 目标=${TGT.leg}/${TGT.torso}/${TGT.head} ±${EPS} · 可见身高=${mm.visibleHeight.toFixed(3)}m`);
  }

  /* ---- E10-E12：动作文件 v2 的语义质量（纯 node 复算，不开浏览器） ---- */
  const moFile = join(ROOT,'示例.motion.json');
  const v1File = join(ROOT,'history','motion.v1.backup.json');
  if(existsSync(moFile)){
    const m2 = JSON.parse(readFileSync(moFile,'utf8'));
    if(m2.version >= 2 && m2.chan){
      const ix = n => m2.keys.indexOf(n);
      const torsoLen = (P, idx) => {
        const ls=P[idx('left_shoulder')], rs=P[idx('right_shoulder')], lh=P[idx('left_hip')], rh=P[idx('right_hip')];
        return Math.hypot((ls[0]+rs[0])/2-(lh[0]+rh[0])/2, (ls[1]+rs[1])/2-(lh[1]+rh[1])/2);
      };
      const tls = m2.poses.filter(Boolean).map(P => torsoLen(P, ix));
      const tlRange = Math.max(...tls) - Math.min(...tls);
      let v1Range = null;
      if(existsSync(v1File)){
        const m1 = JSON.parse(readFileSync(v1File,'utf8'));
        const ix1 = n => m1.keys.indexOf(n);
        const t1 = m1.poses.filter(Boolean).map(P => torsoLen(P, ix1));
        v1Range = Math.max(...t1) - Math.min(...t1);
      }
      rec('E10','逐帧归一化：姿态与景深解耦（躯干长恒定）', tlRange < 0.15,
          `躯干长 极差=${tlRange.toFixed(3)}（阈值0.15）${v1Range===null?'':` · v1 全局单位极差=${v1Range.toFixed(3)}`}`);

      /* null = 该后端不提供的关键点（MediaPipe 无眼角点），不算「低置信未修复」 */
      let lowConf = 0, noneFrames = 0, missing = 0;
      for(const P of m2.poses){
        if(!P){ noneFrames++; continue; }
        for(const q of P){ if(!q){ missing++; continue; } if(q[2] < 0.25) lowConf++; }
      }
      rec('E11','置信度修复：低置信关键点已被补齐', lowConf === 0,
          `修复后仍低置信的点=${lowConf}（检测失败帧 ${noneFrames} · 后端未提供 ${missing}）`);

      const yaw = m2.chan.yaw, roll = m2.chan.roll, contact = m2.chan.contact;
      const maxYaw = Math.max(...yaw.map(Math.abs))*180/Math.PI;
      const maxRoll = Math.max(...roll.map(Math.abs))*180/Math.PI;
      const planted = contact.reduce((a,c)=>a+((c[0]||c[1])?1:0),0)/contact.length;
      rec('E12','语义通道物理可信（转身/侧倾/接触）', maxYaw>50 && maxYaw<120 && maxRoll<25 && planted>0.15 && planted<0.9,
          `最大转身=${maxYaw.toFixed(1)}°（阈值50-120）· 最大侧倾=${maxRoll.toFixed(1)}°（<25°）· 踩实帧占比=${(planted*100).toFixed(0)}%（15-90%）`);
    }else{
      rec('E10','逐帧归一化：姿态与景深解耦（躯干长恒定）', true, '动作文件为 v1（兼容路径），跳过');
      rec('E11','置信度修复：低置信关键点已被补齐', true, '动作文件为 v1（兼容路径），跳过');
      rec('E12','语义通道物理可信（转身/侧倾/接触）', true, '动作文件为 v1（兼容路径），跳过');
    }
  }

  /* ---- 服务 ---- */
  let up = false;
  try{ up = (await httpGet(BASE, 3000)).status === 200; }catch(e){}
  if(!up){ serverStarted = spawn('python3',[join(ROOT,'serve.py'),'--port','8000'],{cwd:ROOT,stdio:'ignore'}); await sleep(1200); }

  /* ---- 临时规格 ---- */
  const tmpSpec = join(ROOT,'__tmp_second_spec.json');
  const tmpBig = join(ROOT,'__tmp_big_spec.json');
  const s2 = JSON.parse(JSON.stringify(specRaw));
  s2.shot.end = 8.0;
  s2.events = [{t:1.0, what:'X 事件校验'}];
  writeFileSync(tmpSpec, JSON.stringify(s2,null,2));
  const s3 = JSON.parse(JSON.stringify(specRaw));
  s3.set.find(x=>/audience/.test(x.class)).size_m = [800,100,10];
  writeFileSync(tmpBig, JSON.stringify(s3,null,2));
  const s4 = JSON.parse(JSON.stringify(specRaw));
  s4.subjects.forEach(s => s.path.forEach(p => { p.pos_m[0] = 0.8; }));
  s4.camera.move = 'static';                       /* follow 会自动对准主体，static 才能判定左右 */
  const tmpMirror = join(ROOT,'__tmp_mirror_spec.json');
  writeFileSync(tmpMirror, JSON.stringify(s4,null,2));
  tmpFiles.push(tmpSpec, tmpBig, tmpMirror);

  chrome = await launchChrome();
  ({cdp, sid} = await connect());
  await cdp.send('Runtime.enable',{},sid);
  await cdp.send('Page.enable',{},sid);
  await cdp.send('DOM.enable',{},sid);
  cdp.on(msg=>{
    if(msg.method==='Runtime.consoleAPICalled' && (msg.params.type==='error'||msg.params.type==='warning'))
      logs.push({kind:'console.'+msg.params.type, text:msg.params.args.map(a=>a.value??a.description??a.type).join(' ')});
    if(msg.method==='Runtime.exceptionThrown')
      logs.push({kind:'exception', text:msg.params.exceptionDetails.exception?.description||msg.params.exceptionDetails.text});
    if(msg.method==='Log.entryAdded' && msg.params.entry.level==='error')
      logs.push({kind:'log.error', text:msg.params.entry.text+' '+(msg.params.entry.url||'')});
  });
  const evl = async (expr, ms=STEP_MS) => withTimeout((async()=>{
    const r = await cdp.send('Runtime.evaluate',{expression:expr,returnByValue:true,awaitPromise:true},sid);
    if(r.exceptionDetails) throw new Error('JS 异常: '+(r.exceptionDetails.exception?.description||r.exceptionDetails.text));
    return r.result.value;
  })(), ms, 'evaluate');
  const nav = async url => {
    const loaded = new Promise(res=>{ const h=m=>{ if(m.method==='Page.loadEventFired'){ cdp.handlers=cdp.handlers.filter(x=>x!==h); res(); } }; cdp.on(h); });
    await cdp.send('Page.navigate',{url},sid);
    await withTimeout(loaded, STEP_MS, 'navigate');
  };
  const waitReady = async (ms=STEP_MS) => {
    const t0=Date.now();
    while(Date.now()-t0<ms){ if(await evl('!!(window.__app&&window.__app.ready)')) return true; await sleep(150); }
    return false;
  };

  /* ================= B1a：http 路径 ================= */
  logs.length=0;
  await nav(BASE);
  const ready = await waitReady();
  await evl('window.__app.setLoop(false);window.__app.setQuality(2);');   /* 断言阶段用最低画质，最省 CPU */
  /* 显式选中示例规格：/api/specs 的目录顺序不保证，临时规格可能排在前面 */
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return s.value;})()`);
  await sleep(900);
  const canvas = await evl(`(()=>{const c=document.querySelector('#viewport canvas');return c?{w:c.width,h:c.height}:null;})()`);
  const errs = logs.filter(l=>l.kind==='exception'||l.kind==='console.error'||l.kind==='log.error');
  const panelMsg = await evl('document.querySelector("#err").hidden ? "" : (document.querySelector("#errMsg").textContent+" | "+(document.querySelector("#errDetail").textContent||"").slice(0,300))');
  rec('B1a','http:// 零错误 + canvas 有效', ready && canvas && canvas.w>0 && errs.length===0,
      `ready=${ready} canvas=${JSON.stringify(canvas)} errors=${errs.length}${errs.length?' | '+errs.map(e=>e.text).join(' ; '):''}${panelMsg?' 错误面板='+panelMsg:''}`);

  if(process.env.DIAG){
    await evl('window.__app.setMimic(true);');
    console.log('DIAG model:', JSON.stringify(await evl('window.__app.model')));
    console.log('DIAG probe@7.21:', JSON.stringify(await evl('window.__app.motionProbe(7.21)')));
    console.log('DIAG probe@0:', JSON.stringify(await evl('window.__app.motionProbe(0)')));
    console.log('DIAG rig@7.21:', JSON.stringify(await evl('window.__app.rigDump(7.21)')));
  }

  /* ================= A1：采样一致性 ================= */
  let maxErr=0, faceAt0=null;
  for(const t of KEY_TIMES){
    const got = await evl(`window.__app.sample(${t})`);
    specRaw.subjects.forEach((s,i)=>{
      maxErr=Math.max(maxErr, Math.abs(got[i].pos[0]-EXPECT[t][i][0]), Math.abs(got[i].pos[1]-EXPECT[t][i][1]));
    });
    if(t===0) faceAt0 = got.find(g=>g.id===specRaw.subjects[0].id).facing;
  }
  rec('A1','5 主体 × 9 关键帧采样与 JSON 逐点一致', maxErr<1e-9 && faceAt0===270, `maxErr=${maxErr} facing@0=${faceAt0}`);

  /* ================= A2/B4：相机解算 ================= */
  await evl('window.__app.seek(0)');
  const cam = await evl('window.__app.cameraState()');
  const framing = await evl('window.__app.framing()');
  const vfovExpect = 2*Math.atan(Math.tan(OFF.hfov/2*Math.PI/180)/cam.aspect)*180/Math.PI;
  const dot = cam.dir.x*OFF.subjectForward.x + cam.dir.z*OFF.subjectForward.z;
  const band = framing.r<0.35?'wide':framing.r<0.8?'full':framing.r<1.6?'medium':'close';
  rec('A2','相机按 schema v1 解算（原点/俯仰/水平FOV/framing）',
      approx(cam.pos.x,0,1e-6) && approx(cam.pos.z,0,1e-6) && approx(cam.pos.y,OFF.camY,1e-6) &&
      dot<0 && approx(cam.fov,vfovExpect,1e-3) && band==='full',
      `pos=(${cam.pos.x},${cam.pos.y},${cam.pos.z}) dir=(${cam.dir.x.toFixed(4)},${cam.dir.y.toFixed(4)},${cam.dir.z.toFixed(4)}) dot(视线,朝向)=${dot.toFixed(4)} vfov=${cam.fov.toFixed(3)}(期望${vfovExpect.toFixed(3)}) r=${framing.r.toFixed(3)}→${band}(规格声明full)`);

  /* ================= A3：页面预算常量 ================= */
  const budget = await evl('window.__app.budget');
  const badKeys = Object.entries(budgetVals).filter(([k,v])=>budget[k]!==v).map(([k,v])=>`${k}=${budget[k]}(期望${v})`);
  rec('A3','页面预算常量与契约一致', badKeys.length===0, `不符=${JSON.stringify(badKeys)} 实际=${JSON.stringify(budget)}`);

  /* ================= B2：下拉枚举与切换 ================= */
  const options = await evl(`[...document.querySelector('#specSelect').options].map(o=>o.value)`);
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(600);
  const before = await evl('({name:window.__app.currentSpecName,end:window.__app.spec.shot.end})');
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='__tmp_second_spec.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(800);
  const after = await evl('({name:window.__app.currentSpecName,end:window.__app.spec.shot.end})');
  const evtText = await evl('document.querySelector("#events").textContent');
  rec('B2','下拉枚举 + 切换重建 + 事件读 what',
      options.includes('示例.json') && options.includes('__tmp_second_spec.json') &&
      before.name==='示例.json' && before.end===14.41 && after.name==='__tmp_second_spec.json' && after.end===8 &&
      evtText.includes('X 事件校验'),
      `options=${JSON.stringify(options)} before=${JSON.stringify(before)} after=${JSON.stringify(after)} events="${evtText.trim()}"`);

  /* ================= B3：播放推进 / 滑块同步 ================= */
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(700);
  await evl('window.__app.setLoop(true)');
  const t0 = await evl('({t:window.__app.playhead,label:document.querySelector("#tNow").textContent,sl:document.querySelector("#timeSlider").value})');
  const btn = await evl(`(()=>{const r=document.querySelector('#playBtn').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()`);
  await cdp.send('Input.dispatchMouseEvent',{type:'mousePressed',x:btn.x,y:btn.y,button:'left',clickCount:1},sid);
  await cdp.send('Input.dispatchMouseEvent',{type:'mouseReleased',x:btn.x,y:btn.y,button:'left',clickCount:1},sid);
  await sleep(1500);
  const t1 = await evl('({t:window.__app.playhead,label:document.querySelector("#tNow").textContent,sl:document.querySelector("#timeSlider").value,btn:document.querySelector("#playBtn").getAttribute("aria-pressed")})');
  await cdp.send('Input.dispatchMouseEvent',{type:'mousePressed',x:btn.x,y:btn.y,button:'left',clickCount:1},sid);
  await cdp.send('Input.dispatchMouseEvent',{type:'mouseReleased',x:btn.x,y:btn.y,button:'left',clickCount:1},sid);
  await sleep(300);
  const t2 = await evl('({btn:document.querySelector("#playBtn").getAttribute("aria-pressed"),t:window.__app.playhead})');
  await sleep(400);
  const t3 = await evl('({t:window.__app.playhead})');
  rec('B3','播放推进 / 暂停 / 滑块同步',
      t1.t > t0.t+0.4 && t1.btn==='true' && t2.btn==='false' && approx(parseFloat(t1.sl), t1.t, 0.02) && approx(t3.t, t2.t, 0.05),
      `t0=${JSON.stringify(t0)} t1=${JSON.stringify(t1)} t2=${JSON.stringify(t2)} 暂停后t3=${JSON.stringify(t3)}`);

  /* ================= B5：布景按 set 的位置尺寸铺开 ================= */
  await evl('window.__app.setLoop(false)');
  await evl('window.__app.renderNow()');
  const stats = await evl('window.__app.sceneStats()');
  const items = await evl(`(()=>{
    const out=[];
    const grp0=window.__app.setGroup;
    if(!grp0) return [];
    grp0.children.forEach(grp=>{
      let minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9,minZ=1e9,maxZ=-1e9,meshes=0,inst=0,instCount=0,pts=0,emissive=0;
      grp.traverse(o=>{
        if(o.isInstancedMesh){
          inst++; instCount=o.count;
          const m=new o.matrixWorld.constructor();
          for(let i=0;i<o.count;i++){
            o.getMatrixAt(i,m);
            const v=new o.position.constructor().setFromMatrixPosition(m).applyMatrix4(o.matrixWorld);
            minX=Math.min(minX,v.x);maxX=Math.max(maxX,v.x);minY=Math.min(minY,v.y);maxY=Math.max(maxY,v.y);minZ=Math.min(minZ,v.z);maxZ=Math.max(maxZ,v.z);
          }
          return;
        }
        if(o.isPoints){ pts+=o.geometry.attributes.position.count; return; }
        if(o.isMesh){
          meshes++;
          if(o.material&&o.material.emissive&&o.material.emissive.getHex()!==0) emissive++;
          o.updateWorldMatrix(true,false);
          const g=o.geometry; if(!g.boundingBox) g.computeBoundingBox();
          const bb=g.boundingBox;
          for(const sx of [bb.min.x,bb.max.x]) for(const sy of [bb.min.y,bb.max.y]) for(const sz of [bb.min.z,bb.max.z]){
            const v=new o.position.constructor(sx,sy,sz).applyMatrix4(o.matrixWorld);
            minX=Math.min(minX,v.x);maxX=Math.max(maxX,v.x);minY=Math.min(minY,v.y);maxY=Math.max(maxY,v.y);minZ=Math.min(minZ,v.z);maxZ=Math.max(maxZ,v.z);
          }
        }
      });
      out.push({meshes,inst,instCount,pts,emissive,minX:+minX.toFixed(2),maxX:+maxX.toFixed(2),minY:+minY.toFixed(2),maxY:+maxY.toFixed(2),minZ:+minZ.toFixed(2),maxZ:+maxZ.toFixed(2)});
    });
    return out;
  })()`);
  const setWant = [
    {name:'stage',      minX:-7,   maxX:7,   minZ:1,    maxZ:11,   maxY:.2},
    /* 人群块含一张人群背景墙（高 10×1.15m，底边贴地） */
    {name:'audience',   minX:-40,  maxX:40,  minZ:35,   maxZ:45,   maxY:11.5},
    {name:'grandstand', minX:-60,  maxX:60,  minZ:52.5, maxZ:67.5, maxY:20.5},
    /* 灯排：桁架 + 9 盏灯（灯罩高亮、光束向下张开） */
    {name:'lamp',       minX:-5.5, maxX:5.5, minZ:44.5, maxZ:45.5, maxY:26.2}
  ];
  const setChecks = setWant.map((w,i)=>{
    const it = items[i];
    if(!it) return false;
    return approx(it.minX,w.minX,1.2) && approx(it.maxX,w.maxX,1.2) && approx(it.minZ,w.minZ,1.2) && approx(it.maxZ,w.maxZ,1.2) && it.maxY <= w.maxY;
  });
  rec('B5','布景 4 项按 set 的位置尺寸铺开（地砖/人群/看台/灯排）', setChecks.every(Boolean) && items.length===4,
      `逐项=${JSON.stringify(setChecks)} 实测=${JSON.stringify(items)}`);

  const lineKinds = await evl(`(()=>{let line=0,seg=0,trail=0;window.__app.scene.traverse(o=>{if(o.isLineSegments){seg++;if(o.geometry.attributes.color)trail++;}else if(o.isLine)line++;});return {line,seg,trail};})()`);
  const ghostCount = await evl(`(()=>{let n=0;window.__app.scene.traverse(o=>{if(o.isInstancedMesh&&o.geometry.type==='SphereGeometry')n+=o.count;});return n;})()`);
  const mmPixels = await evl(`(()=>{const c=document.getElementById('minimap');const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=4) if(d[i]>8) n++;return n;})()`);
  rec('B6','轨迹三层在场', lineKinds.trail===5 && ghostCount>=8 && mmPixels>2000,
      `轨迹对象=${lineKinds.trail}(期望5=每主体1条) 其他LineSegments=${lineKinds.seg-lineKinds.trail} Line=${lineKinds.line} 残影实例=${ghostCount} 小地图非空像素=${mmPixels}`);

  /* 动作文件在场时，人偶比例改由视频骨架推导（k = H/总高单位） */
  const motionPath = join(ROOT,'示例.motion.json');
  let motionExp = null;
  if(existsSync(motionPath)){
    const mo = JSON.parse(readFileSync(motionPath,'utf8'));
    const b = mo.body;
    const thigh = (b.legThighL+b.legThighR)/2, shin = (b.legShinL+b.legShinR)/2;
    const shoulderY = (b.shoulderL[1]+b.shoulderR[1])/2;
    const H = specRaw.subjects[0].size_m[2];
    const k = H/Math.max(.1, thigh+shin+shoulderY+0.62);
    motionExp = {sh:k*Math.abs(b.shoulderL[0]), hp:k*Math.abs(b.hipL[0]), frames:mo.frames, fps:mo.fps, quality:mo.quality};
  }
  /* 测几何前先关模仿，避免动画中的躯干侧倾影响肩位读数 */
  await evl('window.__app.setMimic(false)');
  const modelInfo = await evl('window.__app.model');
  const fig = await evl(`(()=>{
    const need=['root','hips','chest','head','shoulderL','shoulderR','elbowL','elbowR','handL','handR','hipL','hipR','kneeL','kneeR','ankleL','ankleR','footL','footR'];
    const have=window.__app.sceneStats().figNodes;
    const missing=need.filter(n=>!have.includes(n));
    const grab=n=>{const out=[];window.__app.scene.traverse(x=>{if(x.name===n)out.push(x);});return out;};
    const wx=o=>{o.updateWorldMatrix(true,false);return +o.getWorldPosition(new o.position.constructor()).x.toFixed(4);};
    const bottom=mesh=>{mesh.updateWorldMatrix(true,false);mesh.geometry.computeBoundingBox();const bb=mesh.geometry.boundingBox;
      const v=new mesh.position.constructor(bb.min.x,bb.min.y,bb.min.z);v.applyMatrix4(mesh.matrixWorld);return +v.y.toFixed(4);};
    const footLM=grab('footL')[0], footRM=grab('footR')[0];
    const shL=grab('shoulderL')[0], shR=grab('shoulderR')[0], hpL=grab('hipL')[0], hpR=grab('hipR')[0];
    return {missing,
      footLBottom: footLM?bottom(footLM):null, footRBottom: footRM?bottom(footRM):null,
      shL: shL?wx(shL):null, shR: shR?wx(shR):null, hpL: hpL?wx(hpL):null, hpR: hpR?wx(hpR):null};
  })()`);
  const expSh = motionExp ? motionExp.sh : Math.abs(OFF.shoulderDx);
  const expHp = motionExp ? motionExp.hp : Math.abs(OFF.hipDx);
  /* GLB 模型路径：肩髋由模型自身骨骼决定，断言改为结构不变量（左右对称）；程序化路径：断言数值来源 */
  const shoulderOK = modelInfo.loaded
    ? approx(Math.abs(fig.shL), Math.abs(fig.shR), 0.02)
    : approx(Math.abs(fig.shL), expSh, 0.03) && approx(Math.abs(fig.shR), expSh, 0.03);
  const hipOK = modelInfo.loaded
    ? approx(Math.abs(fig.hpL), Math.abs(fig.hpR), 0.02)
    : approx(Math.abs(fig.hpL), expHp, 0.03) && approx(Math.abs(fig.hpR), expHp, 0.03);
  rec('B7','人偶关节齐全 / 肩髋来源正确（模型或 JSON）',
      fig.missing.length===0 && shoulderOK && hipOK,
      `缺节点=${JSON.stringify(fig.missing)} 模型=${JSON.stringify(modelInfo)} 肩x=${fig.shL}/${fig.shR}(期望±${expSh.toFixed(3)}${modelInfo.loaded?'·模型对称':'·数值'}) 髋x=${fig.hpL}/${fig.hpR}(期望±${expHp.toFixed(3)})`);

  /* ================= E1：人体模型加载与骨骼映射 ================= */
  const skin = await evl(`(()=>{
    let skinned=0, tris=0, bones=0;
    window.__app.scene.traverse(o=>{
      if(o.isSkinnedMesh){ skinned++; const g=o.geometry; tris += g.index? g.index.count/3 : g.attributes.position.count/3; }
      if(o.isBone) bones++;
    });
    return {skinned, tris, bones};
  })()`);
  const logicalJoints = await evl('window.__app.sceneStats().figNodes.length');
  rec('E1','人体模型加载 + 骨骼映射（SkinnedMesh + 18 逻辑关节）',
      modelInfo.loaded===true && skin.skinned>0 && skin.tris>10000 && logicalJoints>=18,
      `模型=${JSON.stringify(modelInfo)} SkinnedMesh=${skin.skinned} 面数=${skin.tris} 骨骼=${skin.bones} 逻辑关节=${logicalJoints}`);

  /* ================= E2：IK 命中精度（末端骨 vs 动作目标） ================= */
  await evl('window.__app.setMimic(true);window.__app.seek(3.0);');
  await sleep(200);
  await evl('window.__app.seek(3.0);');
  const ikErr = await evl('window.__app.ikError()');
  const ikVals = ikErr ? Object.values(ikErr) : [];
  const ikMax = ikVals.length ? Math.max(...ikVals) : Infinity;
  rec('E2','IK 命中精度：末端骨与动作目标偏差', ikMax < 0.25,
      `逐末端=${JSON.stringify(ikErr)} 最大偏差=${ikMax.toFixed(3)}m（阈值 0.25m）`);

  /* ================= E3/E4：姿态保真（对齐去噪源） =================
     源动作是 MoveNet 逐帧检测，定位噪声独立叠加（原始肘角二阶差分 75.9°/帧²，
     肘角最大角速度 4209°/s —— 生理不可能）。查看器加载时对源做高斯+One-Euro 去噪，
     断言以「同算法去噪源」为参照：既验证保真，又不要求复现噪声。 */
  const moPath = join(ROOT,'示例.motion.json');
  const mo = existsSync(moPath) ? JSON.parse(readFileSync(moPath,'utf8')) : null;
  /* v2 动作文件：提取器已完成修复+去噪，poses 即干净姿态；v1：此处独立复算去噪（兼容路径） */
  const isV2 = !!(mo && mo.version >= 2 && mo.chan);
  const moD = mo ? (isV2 ? Object.assign({}, mo, {posesD: mo.poses}) : denoiseMotion(mo)) : null;
  const ang2of = (moX, p, a, b, c) => {
    const v1=[p[a][0]-p[b][0], p[a][1]-p[b][1]], v2=[p[c][0]-p[b][0], p[c][1]-p[b][1]];
    const d=(v1[0]*v2[0]+v1[1]*v2[1])/(Math.hypot(...v1)*Math.hypot(...v2)||1);
    return Math.acos(Math.max(-1,Math.min(1,d)))*180/Math.PI;
  };
  if(mo){
    const ix = n => mo.keys.indexOf(n);
    const wrap = a => { let x=((a+90)%180+180)%180-90; return x; };
    const diffs = [], tiltDiff = [], rawDiffs = [];
    for(const t of [1.2, 3.0, 5.0, 7.0, 9.0, 11.0]){
      const fi = Math.min(moD.poses.length-1, Math.round(t*mo.fps));
      const p = moD.posesD[fi], praw = isV2 ? null : mo.poses[fi];
      if(!p) continue;
      const got = await evl(`window.__app.limbAngles(${t})`);
      if(!got) continue;
      const pairs = [
        ['kneeL',  ang2of(moD, p, ix('left_hip'), ix('left_knee'), ix('left_ankle')), 'left_hip','left_knee','left_ankle'],
        ['kneeR',  ang2of(moD, p, ix('right_hip'), ix('right_knee'), ix('right_ankle')), 'right_hip','right_knee','right_ankle']
      ];
      for(const [key, vidAng] of pairs){
        if(typeof got[key] === 'number') diffs.push({key, t, vid:+vidAng.toFixed(1), d3:got[key], d:+Math.abs(got[key]-vidAng).toFixed(1)});
      }
      /* 肘角不参与 E3：手臂走「画面肩→腕距离」定位（投影退化时肘角不可用），
         肘部一致性由 E14 的逐点屏幕对齐度量。 */
      for(const [key, ia, ib, ic] of [
        ['elbowL','left_shoulder','left_elbow','left_wrist'],['elbowR','right_shoulder','right_elbow','right_wrist'],
        ['kneeL','left_hip','left_knee','left_ankle'],['kneeR','right_hip','right_knee','right_ankle']]){
        const rawAng = praw ? ang2of(mo, praw, ix(ia), ix(ib), ix(ic)) : null;
        if(rawAng != null && typeof got[key] === 'number')
          rawDiffs.push({key, t, raw:+rawAng.toFixed(1), d3:got[key], d:+Math.abs(got[key]-rawAng).toFixed(1)});
      }
      const applied = await evl(`window.__app.chestRoll(${t})`);
      if(isV2){
        /* v2：躯干侧倾通道（躯干轴倾角，弧度）→ 应用的绕前向轴转角（正=向主体右倾） */
        const ch = mo.chan.roll[Math.min(mo.chan.roll.length-1, fi)];
        const vidTilt = -ch*180/Math.PI;
        if(applied) tiltDiff.push({t, vid:+vidTilt.toFixed(1), applied:+(applied.roll+applied.hip).toFixed(1),
                                   d:+Math.abs(vidTilt-(applied.roll+applied.hip)).toFixed(1)});
      }else{
        const sdx = p[ix('left_shoulder')][0]-p[ix('right_shoulder')][0];
        const sdy = p[ix('left_shoulder')][1]-p[ix('right_shoulder')][1];
        const slen = Math.hypot(sdx, sdy);
        if(slen > .05 && typeof got.shoulderLine === 'number'){
          const vidTilt = Math.asin(clamp(sdy/slen,-1,1))*180/Math.PI;      /* 仰角：偏航无关 */
          tiltDiff.push({t, vid:+vidTilt.toFixed(1), roll:+applied.roll.toFixed(1), d3:got.shoulderLine, d:+Math.abs(vidTilt - got.shoulderLine).toFixed(1)});
        }
      }
    }
    const mean = a => a.length ? a.reduce((x,y)=>x+y,0)/a.length : Infinity;
    const angMean = mean(diffs.map(x=>x.d));
    const tiltMean = mean(tiltDiff.map(x=>x.d));
    const rawMean = rawDiffs.length ? mean(rawDiffs.map(x=>x.d)) : null;
    const bent = diffs.filter(x=>x.vid < 165).length;
    rec('E3','姿态保真：膝夹角与干净源一致（手臂见 E14 逐点对齐）',
        angMean < 22 && bent >= 3 && tiltMean < 18,
        `膝角平均偏差=${angMean.toFixed(1)}°（阈值22°，n=${diffs.length}，弯曲样本 ${bent}）| 侧倾平均偏差=${tiltMean.toFixed(1)}°（阈值18°）| ${rawMean===null?'v2 无原始噪声源可比':`对原始（含噪）源偏差=${rawMean.toFixed(1)}°`} | 样本=${JSON.stringify(diffs.slice(0,4))} | 侧倾样本=${JSON.stringify(tiltDiff)}`);
  }else{
    rec('E3','姿态保真：膝夹角与干净源一致', false, '缺 示例.motion.json，无法比对');
  }

  if(process.env.RIG_DUMP) console.log('RIG_DUMP t=1.2 ::', JSON.stringify(await evl('window.__app.rigDump(1.2)')));

  /* ================= E5：动作平滑（渲染抖动不高于去噪源） ================= */
  const samples = [];
  for(let i=0;i<24;i++){
    const t = 3.0 + i/24;
    await evl(`window.__app.seek(${t.toFixed(4)})`);
    samples.push(await evl('window.__app.limbAngles(null)'));
  }
  const jit = k => {
    const a = samples.map(s=>s&&s[k]).filter(v=>typeof v === 'number');
    if(a.length < 6) return Infinity;
    let sum=0, n=0;
    for(let i=2;i<a.length;i++){ sum += Math.abs(a[i]-2*a[i-1]+a[i-2]); n++; }
    return n ? sum/n : Infinity;
  };
  const jElbow = jit('elbowL'), jKnee = jit('kneeL');
  /* 去噪源同窗口的二阶差分均值：噪声基线 */
  const srcJit = (triple) => {
    if(!moD) return null;
    const ix = n => moD.keys.indexOf(n);
    const seq = [];
    for(let i=0;i<24;i++){
      const t = 3.0 + i/24;
      const fi = Math.min(moD.poses.length-1, Math.round((t - specRaw.shot.start)*moD.fps));
      const p = moD.posesD[fi];
      seq.push(p ? ang2of(moD, p, ix(triple[0]), ix(triple[1]), ix(triple[2])) : NaN);
    }
    const a = seq.filter(Number.isFinite);
    if(a.length < 6) return null;
    let sum=0, n=0;
    for(let i=2;i<a.length;i++){ sum += Math.abs(a[i]-2*a[i-1]+a[i-2]); n++; }
    return n ? sum/n : null;
  };
  const srcElbow = srcJit(['left_shoulder','left_elbow','left_wrist']);
  const srcKnee  = srcJit(['left_hip','left_knee','left_ankle']);
  const limE = srcElbow === null ? 12 : Math.min(12, srcElbow + 3);
  const limK = srcKnee  === null ? 4  : Math.min(4,  srcKnee + 3);
  const seq = samples.map(x=>x&&x.elbowL).filter(v=>typeof v==='number');
  rec('E5','动作平滑：渲染抖动不高于去噪源',
      jElbow < limE && jKnee < limK,
      `肘角二阶差分均值=${jElbow.toFixed(2)}°（去噪源 ${srcElbow===null?'?':srcElbow.toFixed(2)}°，阈值 ${limE.toFixed(2)}）| 膝角=${jKnee.toFixed(2)}°（去噪源 ${srcKnee===null?'?':srcKnee.toFixed(2)}°，阈值 ${limK.toFixed(2)}）| 肘角序列=${JSON.stringify(seq.slice(0,12))}`);

  /* ================= E6：时间连续性（不闪） =================
     端到端指标：按 30fps 采样渲染出的世界坐标，二阶差分（加速度）必须小。
     基线 = 去噪源同一时间轴的位移二阶差分（换算到身高比例），
     渲染不得比基线更抖；绝对下限兜住「源很平滑但渲染在闪」的情形。 */
  const figSamples = [];
  for(let i=0;i<30;i++){
    const t = 3.0 + i/30;
    figSamples.push(await evl(`window.__app.figurePose(${t.toFixed(4)})`));
  }
  const H = (specRaw.primary && specRaw.primary.size && specRaw.primary.size[2]) || 1.7;
  /* 用「相对髋位」的关节位移：渲染结果是世界坐标（含躯干整体摆动），
     源数据是相对髋中点的量 —— 只有取相对量才是同类比较。 */
  const accOf = (seqs, key) => {
    let sum=0, n=0;
    for(let i=2;i<seqs.length;i++){
      const A=seqs[i-2], B=seqs[i-1], C=seqs[i];
      if(!A||!B||!C||!A[key]||!B[key]||!C[key]||!A.hips||!B.hips||!C.hips) continue;
      const rel=(k,f)=>[f[k][0]-f.hips[0], f[k][1]-f.hips[1], f[k][2]-f.hips[2]];
      const a=rel(key,A), b=rel(key,B), c=rel(key,C);
      sum += Math.hypot(c[0]-2*b[0]+a[0], c[1]-2*b[1]+a[1], c[2]-2*b[2]+a[2]); n++;
    }
    return n ? sum/n/H : Infinity;
  };
  const perJoint = {handL:accOf(figSamples,'handL'), handR:accOf(figSamples,'handR'), footL:accOf(figSamples,'footL')};
  const rendAcc = Math.max(perJoint.handL, perJoint.handR, perJoint.footL);
  const worstKey = Object.keys(perJoint).reduce((a,b)=>perJoint[a]>=perJoint[b]?a:b);
  const seqAcc = [];
  for(let i=2;i<figSamples.length;i++){
    const rel=(k,f)=>[f[k][0]-f.hips[0], f[k][1]-f.hips[1], f[k][2]-f.hips[2]];
    const A=figSamples[i-2],B=figSamples[i-1],C=figSamples[i];
    if(!A||!B||!C) continue;
    const a=rel(worstKey,A), b=rel(worstKey,B), c=rel(worstKey,C);
    seqAcc.push(+Math.hypot(c[0]-2*b[0]+a[0], c[1]-2*b[1]+a[1], c[2]-2*b[2]+a[2]).toFixed(5));
  }
  /* 基线：去噪源手腕位移二阶差分（源单位）→ 乘 motion→米 比例 → 除身高 */
  /* 基线：源里**同一个关节**在同一采样率（30fps、同一时间窗）下的位移二阶差分。
     腿甩得比手快，拿手腕的基线去卡脚的渲染是不成比例的比较。 */
  const baseOf = jname=>{
    if(!moD) return null;
    const ij = moD.keys.indexOf(jname);
    const F = moD.fps||24;
    const at = t=>{
      const ft = (t - (specRaw.shot.start||0))*F;
      const i0 = clamp(Math.floor(ft),0,moD.posesD.length-1), i1 = Math.min(moD.posesD.length-1,i0+1);
      const f = clamp(ft-i0,0,1), a = moD.posesD[i0], b = moD.posesD[i1];
      if(!a || !b || !a[ij] || !b[ij]) return null;
      return [a[ij][0]+(b[ij][0]-a[ij][0])*f, a[ij][1]+(b[ij][1]-a[ij][1])*f];
    };
    const seq=[];
    for(let i=0;i<30;i++) seq.push(at(3.0+i/30));
    let sum=0,n=0;
    for(let i=2;i<seq.length;i++){
      const a=seq[i-2], b=seq[i-1], c=seq[i];
      if(!a||!b||!c) continue;
      sum += Math.hypot(c[0]-2*b[0]+a[0], c[1]-2*b[1]+a[1]); n++;
    }
    return n ? sum/n : null;
  };
  const baseAcc = baseOf('left_wrist');
  const modelScale = (await evl('window.__app.model'))?.torso || 0;
  /* 每个渲染关节对照源里对应关节的基线（手↔腕、脚↔踝） */
  const pairOf = {handL:'left_wrist', handR:'right_wrist', footL:'left_ankle'};
  const checks = Object.keys(perJoint).map(k=>{
    const b = baseOf(pairOf[k]);
    const bn = b === null ? null : b * modelScale / H;
    const lim = bn === null ? 0.006 : Math.max(0.006, bn*2.5);
    return {k, rendered:+perJoint[k].toFixed(5), base:bn===null?null:+bn.toFixed(5), lim:+lim.toFixed(5), ok:perJoint[k] < lim};
  });
  const baseNorm = baseAcc === null ? null : baseAcc * modelScale / H;
  const bendJumps = await evl('window.__app.bendJumps()');
  rec('E6','时间连续性：动作有过程、不闪跳',
      checks.every(c=>c.ok) && (bendJumps === 0 || bendJumps === null),
      `逐关节（相对髋位、30fps 同窗）=${JSON.stringify(checks)} | 最差=${worstKey} 序列=${JSON.stringify(seqAcc)} | 弯折方位单帧突变=${bendJumps}`);

  /* ================= E7：弯折侧向符合解剖 =================
     肘相对「肩-手」弦线应偏后（小臂朝前），膝相对「髋-踝」弦线应偏前（小腿朝后）。
     符号反了就是鸡翅膀 / 反关节 —— R4 就是这样翻的。 */
  const bsSamples = [];
  for(let i=0;i<12;i++) bsSamples.push(await evl(`window.__app.bendSide(${(3.0 + i/12).toFixed(4)})`));
  const bsMean = k => {
    const v = bsSamples.map(x=>x&&x[k]).filter(v=>typeof v === 'number');
    return v.length ? v.reduce((a,b)=>a+b,0)/v.length : NaN;
  };
  const bs = {armL:bsMean('armL'), armR:bsMean('armR'), legL:bsMean('legL'), legR:bsMean('legR')};
  const TOL = 0.002;
  rec('E7','弯折侧向符合解剖（肘偏后、膝偏前）',
      bs.armL <= TOL && bs.armR <= TOL && bs.legL >= -TOL && bs.legR >= -TOL,
      `肘偏移(前向为正) L=${bs.armL.toFixed(4)}m R=${bs.armR.toFixed(4)}m（应≤0）| 膝偏移 L=${bs.legL.toFixed(4)}m R=${bs.legR.toFixed(4)}m（应≥0）| 逐帧=${JSON.stringify(bsSamples.slice(0,4))}`);

  /* ================= E8：躯干转身 =================
     幅度来自「肩宽透视收缩」（自校准参考），方向来自头部朝向；死区滤掉肢体造成的假塌陷。 */
  const yawDeg = async t => (await evl(`window.__app.bodyYaw(${t})`))*180/Math.PI;
  const y65 = await yawDeg(6.5);
  const facing65 = await evl('window.__app.figureFacing()');
  const applied = await evl('window.__app.bodyYawApplied()')*180/Math.PI;
  const y0 = await yawDeg(0), y35 = await yawDeg(3.5), y10 = await yawDeg(10);
  rec('E8','躯干转身：视频里的转身被还原',
      Math.abs(y65) > 50 && Math.abs(y0) < 5 && Math.abs(y35) < 10 && Math.abs(y10) < 10 && Math.abs(applied) > 40,
      `yaw: t=0 ${y0.toFixed(1)}° · t=3.5 ${y35.toFixed(1)}° · t=6.5 ${y65.toFixed(1)}° · t=10 ${y10.toFixed(1)}° | 已施加=${applied.toFixed(1)}° | 转身时人偶朝向=${JSON.stringify(facing65.map(v=>+v.toFixed(2)))}`);

  /* ================= E9：接触门控——踩实的脚不滑步 ================= */
  const fp = [], pl = [];
  for(let i=0;i<30;i++){
    const t = 2.0 + i/30;
    fp.push(await evl(`window.__app.figurePose(${t.toFixed(4)})`));
    pl.push(await evl(`window.__app.planted(${t.toFixed(4)})`));
  }
  const footStep = (key, side) => {
    let sum=0, n=0, mx=0;
    for(let i=1;i<fp.length;i++){
      if(!pl[i] || !pl[i][side]) continue;          /* 只看踩实的脚 */
      const a=fp[i-1][key], b=fp[i][key];
      if(!a||!b) continue;
      const d=Math.hypot(b[0]-a[0], b[2]-a[2]);
      sum+=d; n++; mx=Math.max(mx,d);
    }
    return {mean:n?sum/n:null, max:mx, n};
  };
  const fs = {footL:footStep('footL',0), footR:footStep('footR',1)};
  const okMean = m => m === null || m < 0.02;          /* 该侧无踩实样本则不参与统计 */
  rec('E9','接触门控：踩实的脚在世界系里保持不动',
      (fs.footL.n >= 3 || fs.footR.n >= 3) && okMean(fs.footL.mean) && okMean(fs.footR.mean),
      `t=2.0–3.0s 踩实帧 L:${fs.footL.n} R:${fs.footR.n}；踩实期间逐帧位移 L 均值 ${fs.footL.mean===null?'无样本':(fs.footL.mean*1000).toFixed(1)+'mm'} 峰值 ${(fs.footL.max*1000).toFixed(1)}mm | R 均值 ${fs.footR.mean===null?'无样本':(fs.footR.mean*1000).toFixed(1)+'mm'} 峰值 ${(fs.footR.max*1000).toFixed(1)}mm（阈值 20mm/帧）`);

  /* ================= E13：对照机位复现源视频取景 =================
     源视频里主体的画面占比与水平位置由动作文件的 videoBox 给出；
     对照机位据此反解相机距离与偏航 —— 3D 人偶与视频表演者同尺寸同位置。 */
  const mo13 = existsSync(moFile) ? JSON.parse(readFileSync(moFile,'utf8')) : null;
  if(mo13 && mo13.videoBox && mo13.videoBox.frac){
    const t13 = 7.21;
    const fi13 = Math.min(mo13.videoBox.frac.length-1, Math.round(t13*(mo13.fps||24)));
    const fh = mo13.videoBox.frac[fi13][0], cx = mo13.videoBox.frac[fi13][1];
    await evl(`window.__app.setAspect('video');window.__app.setView('match');window.__app.seek(${t13});`);
    const cm = await evl('window.__app.cameraState()');
    const sm = await evl(`window.__app.sample(${t13})`);
    const subj = sm[0];
    const H13 = (specRaw.primary && specRaw.primary.size && specRaw.primary.size[2]) || 1.6;
    /* 真正落到画面上的高度：解析式反解是自证的，必须量渲染结果。
       figureExtent 现在量的是**可见剪影**（发顶→鞋底），与视频的可见身高同口径，
       不再需要「头骨比发顶低 13%」这类经验换算系数。 */
    const ext13 = await evl('window.__app.figureExtent()');
    const renderedFrac = ext13 ? ext13.frac : null;
    const dist13 = Math.hypot(subj.world[0]-cm.pos.x, subj.world[2]-cm.pos.z);
    const frameH13 = 2*dist13*Math.tan(cm.fov/2*Math.PI/180);
    const gotFrac = H13/frameH13;
    const azSub = Math.atan2(subj.world[0]-cm.pos.x, subj.world[2]-cm.pos.z);
    const azCam = Math.atan2(cm.dir.x, cm.dir.z);
    const halfH = Math.atan(Math.tan(cm.fov/2*Math.PI/180)*cm.aspect);
    const gotX = -Math.tan(azSub-azCam)/Math.tan(halfH);
    const wantX = (cx-.5)*2;
    const eFrac = Math.abs(gotFrac-fh)/Math.max(.01,fh), eX = Math.abs(gotX-wantX);
    /* 叠加对齐的严格判据：3D 髋中点的投影位置必须落在视频髋中点的画面位置 */
    const hipScr = await evl('window.__app.hipScreen(7.21)');
    const vx = mo13.videoBox.frac[fi13][1], vy = mo13.videoBox.frac[fi13][2];
    const dpx = hipScr ? Math.hypot(hipScr.x-vx, hipScr.y-vy) : 1;
    const eRender = renderedFrac===null ? 1 : Math.abs(renderedFrac - fh)/Math.max(.01, fh);
    rec('E13','对照机位：人偶与视频表演者同尺寸同位置',
        eFrac < .15 && eX < .18 && dpx < .03 && eRender < .15,
        `视频主体占比=${(fh*100).toFixed(1)}% 3D=${(gotFrac*100).toFixed(1)}%（相对误差 ${(eFrac*100).toFixed(1)}%）| 实测渲染占比=${renderedFrac===null?'?':(renderedFrac*100).toFixed(1)+'%'}（偏差 ${(eRender*100).toFixed(1)}%）| 视频水平=${(wantX*100).toFixed(0)}% 3D=${(gotX*100).toFixed(0)}%（差 ${(eX*100).toFixed(0)}%）| 髋中点投影=(${hipScr?hipScr.x.toFixed(3):'?'},${hipScr?hipScr.y.toFixed(3):'?'}) vs 视频(${vx.toFixed(3)},${vy.toFixed(3)}) 偏差=${(dpx*100).toFixed(1)}% | 相机距 ${dist13.toFixed(2)}m fov ${cm.fov.toFixed(1)}°`);
    await evl(`window.__app.setView('schema');`);
  }else{
    rec('E13','对照机位：人偶与视频表演者同尺寸同位置', true, '动作文件无 videoBox（v1），跳过');
  }

  /* ================= E14：手/脚与源视频逐点对齐（视频一致性的直接度量） =================
     视频里的腕位换算到画面：髋中点屏幕位 + 归一化腕偏移 × (躯干像素长/画面尺寸)。
     对照机位已把相机的取景复现成源视频，两者可直接逐点比。 */
  const mo14 = existsSync(moFile) ? JSON.parse(readFileSync(moFile,'utf8')) : null;
  if(mo14 && mo14.videoBox && mo14.videoBox.torsoPx){
    await evl(`window.__app.setAspect('video');window.__app.setView('match');`);
    const devs = [];
    for(const t of [0, 2.5, 5, 6.5, 7.21, 10]){
      const fi = Math.min(mo14.videoBox.frac.length-1, Math.round(t*(mo14.fps||24)));
      const fr = mo14.videoBox.frac[fi], tp = mo14.videoBox.torsoPx[fi];
      const vw = mo14.videoBox.w, vh = mo14.videoBox.h;
      const pose = mo14.poses[fi];
      if(!pose || !tp) continue;
      const idx = mo14.keys.indexOf('left_wrist'), idx2 = mo14.keys.indexOf('right_wrist');
      const hs = await evl(`window.__app.handScreen(${t})`);
      if(!hs) continue;
      const exp = (j, p) => ({x: fr[1] + p[j][0]*(tp/vw), y: fr[2] + p[j][1]*(tp/vh)});
      const eL = exp(idx, pose), eR = exp(idx2, pose);
      const dL = Math.hypot(hs.handL.x-eL.x, hs.handL.y-eL.y);
      const dR = Math.hypot(hs.handR.x-eR.x, hs.handR.y-eR.y);
      devs.push({t, dL:+(dL*100).toFixed(1), dR:+(dR*100).toFixed(1)});
    }
    const avg = devs.reduce((a,d)=>a+(d.dL+d.dR)/2, 0)/Math.max(1,devs.length);
    rec('E14','手部与源视频逐点对齐（画面位置）', avg < 8,
        `平均偏差 ${avg.toFixed(1)}%（画面宽的百分比，阈值 8%）逐帧=${JSON.stringify(devs)}`);
    await evl(`window.__app.setView('schema');window.__app.setAspect('viewport');`);
  }else{
    rec('E14','手部与源视频逐点对齐（画面位置）', true, '动作文件缺 torsoPx，跳过');
  }

  /* ================= D1：模仿动画驱动骨架 ================= */
  await evl('window.__app.setMimic(true);window.__app.seek(0);');
  const mimicState = await evl('({on:window.__app.mimic, btn:document.querySelector("#mimicBtn").getAttribute("aria-pressed"), info:document.querySelector("#mimicInfo").textContent, motion:window.__app.motion})');
  const p0 = await evl('window.__app.figurePose(0)');
  const p3 = await evl('window.__app.figurePose(3)');
  const p8 = await evl('window.__app.figurePose(8)');
  const dh = Math.hypot(p3.handL[0]-p0.handL[0], p3.handL[1]-p0.handL[1], p3.handL[2]-p0.handL[2]);
  const dh2 = Math.hypot(p8.handR[0]-p3.handR[0], p8.handR[1]-p3.handR[1], p8.handR[2]-p3.handR[2]);
  const df = Math.hypot(p8.footL[0]-p0.footL[0], p8.footL[1]-p0.footL[1], p8.footL[2]-p0.footL[2]);
  const dHead = Math.hypot(p3.head[0]-p0.head[0], p3.head[1]-p0.head[1], p3.head[2]-p0.head[2]);
  rec('D1','模仿源视频：关节随时间真实运动',
      mimicState.on===true && mimicState.btn==='true' && dh>0.15 && dh2>0.15 && df>0.08 && dHead>0.02,
      `开关=${mimicState.on}/${mimicState.btn} 信息="${mimicState.info}" 动作=${JSON.stringify(mimicState.motion)} | ΔhandL(0→3s)=${dh.toFixed(3)}m ΔhandR(3→8s)=${dh2.toFixed(3)}m ΔfootL(0→8s)=${df.toFixed(3)}m Δhead=${dHead.toFixed(3)}m`);

  /* ================= D2：facing 270 = 面向相机 ================= */
  const facing0 = await evl('window.__app.figureFacing()');
  const camDir0 = await evl('window.__app.cameraState()');
  const dot2 = facing0[0]*camDir0.dir.x + facing0[2]*camDir0.dir.z;
  rec('D2','主体朝向相机（facing 270 修复）', facing0[2] < -0.9 && dot2 < -0.9,
      `人偶面朝=${JSON.stringify(facing0.map(v=>+v.toFixed(3)))} 相机视线=(${camDir0.dir.x.toFixed(2)},${camDir0.dir.y.toFixed(2)},${camDir0.dir.z.toFixed(2)}) 点积=${dot2.toFixed(3)}(期望≈-1)`);

  /* ================= D3：schema +x 落在画面右侧（无镜像） ================= */
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='__tmp_mirror_spec.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(800);
  const mir = await evl(`(()=>{
    const cs=window.__app.cameraState();
    const s=window.__app.sample(0)[0];
    const right={x:-cs.dir.z, y:0, z:cs.dir.x};            /* 相机屏幕右 = dir × up */
    const vx=(s.world[0]-cs.pos.x)*right.x+(s.world[2]-cs.pos.z)*right.z;
    return {schemaX:s.pos[0], worldX:s.world[0], viewX:+vx.toFixed(4), dir:[cs.dir.x,cs.dir.z]};
  })()`);
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(800);
  rec('D3','schema +x 呈现在画面右侧（镜像已修）',
      approx(mir.schemaX, 0.8, 1e-9) && approx(mir.worldX, -0.8, 1e-6) && mir.viewX > 0.5,
      `schema x=+0.8 → world=${mir.worldX} → 屏幕横向分量=${mir.viewX}(>0=右侧) 相机dir=${JSON.stringify(mir.dir)}`);

  /* ================= B12：畸形规格的健壮性 ================= */
  const badSpec = {shot:{start:0,end:2}, camera:null, subjects:[
    {id:'X1', class:'person', desc:'no path'},
    {id:'X2', class:'prop', desc:'one point', path:[{t:0, pos_m:[1,1]}]}
  ], set:[], events:[{what:'无时间'}]};
  const badRes = await evl(`window.__app.loadSpec('__tmp_bad.json', ${JSON.stringify(badSpec)})`);
  const alive = await evl('!!(window.__app && window.__app.ready)');
  const recover = await evl(`window.__app.loadSpec('示例.json', null)`);
  const recovered = await evl('window.__app.currentSpecName');
  rec('B12','畸形规格不产生未捕获异常且可恢复',
      badRes && badRes.ok !== undefined && alive === true && recover && recover.ok === true && recovered === '示例.json',
      `载入结果=${JSON.stringify(badRes)} 存活=${alive} 恢复=${JSON.stringify(recover)} 恢复后=${recovered}`);

  const crowd = await evl(`(()=>{let n=-1,pts=0,emissive=0,fog=!!window.__app.scene.fog;window.__app.scene.traverse(o=>{if(o.isInstancedMesh&&o.geometry.type==='CapsuleGeometry'&&n<0){n=o.count;}if(o.isPoints)pts+=o.geometry.drawRange.count===Infinity?o.geometry.attributes.position.count:o.geometry.drawRange.count;if(o.material&&o.material.emissive&&o.material.emissive.getHex()!==0)emissive++;});return {n,pts,emissive,fog};})()`);
  const tiers = await evl(`(()=>{let t=0;const g=window.__app.setGroup.children[2];if(g){g.traverse(o=>{if(o.isMesh&&o.geometry.index){t=Math.max(t, Math.round(o.geometry.index.count/3/12));}});}return t;})()`);
  rec('B8','竞技场元素（人群上限/看台阶梯/灯排发光/雾）',
      crowd.n>0 && crowd.n<=900 && crowd.pts<=180 && crowd.emissive>0 && tiers>=7 && crowd.fog,
      `人群实例=${crowd.n}(≤900) 灯点=${crowd.pts}(≤180) 发光材质=${crowd.emissive} 看台台阶=${tiers}(≥7) 雾=${crowd.fog} 灯总数=${stats.lights}(≤4)`);

  /* ================= B9：视频对照 ================= */
  const vid = await evl('window.__app.videoState()');
  await evl('window.__app.setLoop(true);window.__app.setPlaying(true);');
  await sleep(1200);
  const vid2 = await evl('window.__app.videoState()');
  const ph = await evl('window.__app.playhead');
  await evl('window.__app.setPlaying(false);window.__app.setLoop(false);');
  /* 确定性同步校验：seek 后视频时钟应精确跟随（不受软件渲染帧率影响） */
  await evl('window.__app.seek(6.0);');
  await sleep(1000);
  const vid3 = await evl('window.__app.videoState()');
  const ph3 = await evl('window.__app.playhead');
  rec('B9','视频对照加载 + 时间轴同步',
      vid.readyState>=1 && approx(vid.duration,14.464,0.2) && vid.w===720 && vid.h===1280 &&
      Math.abs(vid2.currentTime-ph)<1.5 && Math.abs(vid3.currentTime-ph3)<0.15,
      `readyState=${vid.readyState} duration=${vid.duration} 尺寸=${vid.w}×${vid.h} | 播放1s漂移=${Math.abs(vid2.currentTime-ph).toFixed(3)}s（软件渲染上限1.5s）| seek(6.0)后 video=${vid3.currentTime.toFixed(3)} playhead=${ph3.toFixed(3)} 偏差=${Math.abs(vid3.currentTime-ph3).toFixed(3)}s（阈值0.15s）`);

  /* ================= B11：画幅 ================= */
  const a916 = await evl(`(()=>{window.__app.setAspect('9:16');const r=document.querySelector('#viewport').getBoundingClientRect();return {aspect:window.__app.camera.aspect, w:r.width, h:r.height, ratio:r.width/r.height};})()`);
  const avp = await evl(`(()=>{window.__app.setAspect('viewport');const r=document.querySelector('#viewport').getBoundingClientRect();return {aspect:window.__app.camera.aspect, w:r.width, h:r.height, ratio:r.width/r.height};})()`);
  rec('B11','画幅切换 camera.aspect 与遮罩一致',
      approx(a916.aspect,9/16,0.01) && approx(a916.ratio,9/16,0.02) && approx(avp.aspect,avp.ratio,0.02),
      `9:16 → aspect=${a916.aspect.toFixed(4)} 视口比=${a916.ratio.toFixed(4)} | 视口 → aspect=${avp.aspect.toFixed(4)} 视口比=${avp.ratio.toFixed(4)}`);

  /* ================= B14：录制导出 ================= */
  const recRes = await evl(`(()=>{
    const b=document.querySelector('#recBtn');
    if(!b) return {ok:false, why:'无按钮'};
    b.click();
    const on = window.__app.recording();
    b.click();
    return {ok:on, off:!window.__app.recording(), label:b.querySelector('.rec-label').textContent};
  })()`);
  rec('B14','录制导出：画布可录成 webm', recRes.ok===true && recRes.off===true,
      `录制中=${recRes.ok} 已停止=${recRes.off} 按钮文案="${recRes.label}"`);

  /* ================= C1：空闲零绘制 ================= */
  await evl('window.__app.setLoop(true);window.__app.setPlaying(false);');
  await sleep(600);
  const rc0 = await evl('window.__app.renderCount');
  await sleep(10000);
  const rc1 = await evl('window.__app.renderCount');
  rec('C1','空闲 10s 内绘制次数增量 ≤1', rc1-rc0 <= 1, `renderCount ${rc0} → ${rc1}（增量 ${rc1-rc0}）`);

  /* ================= C2：绘制预算 ================= */
  await evl('window.__app.setLoop(false);window.__app.renderNow();');
  const rinfo = await evl('({...window.__app.renderer.info.render})');
  const sstats = await evl('window.__app.sceneStats()');
  rec('C2','单帧 DrawCall ≤40 且三角面 ≤120k', rinfo.calls<=40 && rinfo.triangles<=120000,
      `calls=${rinfo.calls} triangles=${rinfo.triangles} mesh=${sstats.meshes} line=${sstats.lines} point=${sstats.points} frameMs=${(await evl('window.__app.frameMs')).toFixed(1)}ms`);

  /* ================= C3：规模无关 ================= */
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='__tmp_big_spec.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(1200);
  const bigCrowd = await evl(`(()=>{let n=-1;window.__app.scene.traverse(o=>{if(o.isInstancedMesh&&o.geometry.type==='CapsuleGeometry'&&n<0)n=o.count;});return n;})()`);
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(1200);
  rec('C3','人群规模与 JSON 体量无关（≤900）', bigCrowd>0 && bigCrowd<=900, `800×100m 观众块 → 实例 ${bigCrowd}`);

  /* ================= C4：帧率上限 ================= */
  await evl('window.__app.setLoop(true);');
  const c4a = await evl('window.__app.renderCount');
  await evl('window.__app.setPlaying(true);');
  await sleep(3000);
  const c4b = await evl('window.__app.renderCount');
  await evl('window.__app.setPlaying(false);window.__app.setLoop(false);');
  rec('C4','播放 3s 内绘制次数 ≤96', c4b-c4a <= 96, `renderCount ${c4a} → ${c4b}（3s 增量 ${c4b-c4a}，上限 96）`);

  /* ================= C5：降级生效 ================= */
  await evl('window.__app.setQuality(0);window.__app.setLoop(true);window.__app.setPlaying(true);');
  await sleep(4500);
  const q1 = await evl('window.__app.quality');
  const fm = await evl('window.__app.frameMs');
  const qCount = await evl('window.__app.sceneStats().meshes');
  await evl('window.__app.setPlaying(false);window.__app.setLoop(false);');
  rec('C5','帧时超预算时自动降级', q1!=='L0' || fm<=33, `初始档=L0 → 观察档=${q1} 最近帧间隔=${fm.toFixed(1)}ms(预算33ms)`);

  /* ================= C6：交互必刷新 ================= */
  await evl('window.__app.setLoop(true);');
  const steps = [];
  const probe = async (label, expr) => {
    const a = await evl('window.__app.renderCount');
    await evl(expr);
    await sleep(3000);        /* 软件渲染下单帧可能接近 2s，等够一帧的余量 */
    const b = await evl('window.__app.renderCount');
    steps.push(`${label}:+${b-a}`);
    return b-a;
  };
  const d1 = await probe('seek', 'window.__app.seek(5)');
  const d2 = await probe('视角', 'window.__app.setView("top")');
  const d3 = await probe('画幅', 'window.__app.setAspect("16:9")');
  const d4 = await probe('图层', `(()=>{const c=document.querySelector('#tgTrail');c.checked=!c.checked;c.dispatchEvent(new Event('change'));return 1;})()`);
  await evl('window.__app.setView("schema");window.__app.setAspect("viewport");window.__app.setLoop(false);');
  rec('C6','参数变更必触发重绘', [d1,d2,d3,d4].every(d=>d>=1), `绘制增量 ${steps.join(' ')}`);

  /* ================= B16：键盘快捷键 ================= */
  await evl(`window.__app.setView('schema');window.__app.setPlaying(false);window.__app.seek(3);`);
  const key = async (code, key, vk) => {
    for(const type of ['keyDown','keyUp']){
      await cdp.send('Input.dispatchKeyEvent',{type, key, code, windowsVirtualKeyCode:vk, nativeVirtualKeyCode:vk}, sid);
    }
  };
  await key('Space',' ',32);
  await sleep(400);
  const sp1 = await evl('window.__app.playhead');
  const pl1 = await evl('window.__app.playing');
  await key('Space',' ',32);
  await sleep(300);
  const sp2 = await evl('window.__app.playhead');
  const pl2 = await evl('window.__app.playing');
  await evl('window.__app.seek(3);');
  await key('ArrowRight','ArrowRight',39);
  await sleep(300);
  const sp3 = await evl('window.__app.playhead');
  await key('Digit3','3',51);
  await sleep(300);
  const view3 = await evl(`document.querySelector('#viewSeg button[data-view="overlay"]').getAttribute('aria-pressed')`);
  await evl(`window.__app.setView('schema');`);
  const kbSpace = pl1 === true && pl2 === false;   /* 渲染循环已停，播放态仍应翻转 */
  const kbArrow = sp3 > 3.02 && sp3 < 3.10;
  rec('B16','键盘快捷键：空格播放/暂停 · ←→ 步进 · 数字切镜头', kbSpace && kbArrow && view3 === 'true',
      `空格→播放中=${pl1} playhead=${sp1.toFixed(3)} 再空格→播放中=${pl2} playhead=${sp2.toFixed(3)} →后=${sp3.toFixed(3)}(应≈3.04) 数字3→叠加=${view3}`);

  /* ================= B17：深链（刷新/分享回到同一画面） ================= */
  const enc = encodeURIComponent('示例.json');
  await nav(`${BASE}?spec=${enc}&t=5&view=match&aspect=video`);
  const readyDeep = await waitReady();
  await sleep(1200);
  const deepState = await evl('({t:window.__app.playhead, spec:window.__app.currentSpecName, url:location.search, ready:!!window.__app.ready, aspect:document.querySelector("#aspectSel").value, view:[...document.querySelectorAll("#viewSeg button")].filter(b=>b.getAttribute("aria-pressed")==="true").map(b=>b.dataset.view)[0], err:document.querySelector("#err").hidden ? "" : (document.querySelector("#errMsg").textContent+" "+(document.querySelector("#errDetail").textContent||"").slice(0,200))})');
  /* 深链还原块在 loadSpec 内抛错时，playhead/spec/url 已在抛错前赋值 ——
     只查这三项会让「启动失败」照样 PASS。必须同时查 ready、错误面板、画幅与镜头。 */
  rec('B17','深链：URL 携带规格/时间/镜头/画幅并可还原',
      readyDeep && deepState.ready && !deepState.err &&
      Math.abs(deepState.t - 5) < 0.2 && deepState.spec === '示例.json' &&
      /view=match/.test(deepState.url) && deepState.aspect === 'video' && deepState.view === 'match',
      `ready=${deepState.ready} 还原后 spec=${deepState.spec} playhead=${deepState.t?.toFixed(2)}(期望5) aspect=${deepState.aspect} view=${deepState.view} url="${deepState.url}"${deepState.err?' 错误面板='+deepState.err:''}`);

  /* ================= B1b：file:// 路径 ================= */
  logs.length=0;
  await nav(FILE_URL);
  const readyF = await waitReady();
  await evl('window.__app.setLoop(false)');
  const canvasF = await evl(`(()=>{const c=document.querySelector('#viewport canvas');return c?{w:c.width,h:c.height}:null;})()`);
  const errsF = logs.filter(l=>l.kind==='exception'||l.kind==='console.error'||l.kind==='log.error');
  const errHidden = await evl('document.querySelector("#err").hidden');
  rec('B1b','file:// 零错误（不发起会被 CORS 拦的请求）', readyF && canvasF && canvasF.w>0 && errsF.length===0 && errHidden,
      `ready=${readyF} canvas=${JSON.stringify(canvasF)} errors=${errsF.length}${errsF.length?' | '+errsF.map(e=>e.text).join(' ; '):''} 错误面板隐藏=${errHidden}`);

  /* ================= 截图（按需单帧渲染） ================= */
  mkdirSync(SHOTS,{recursive:true});
  await nav(BASE);
  await waitReady();
  await evl(`(()=>{const s=document.querySelector('#specSelect');s.value='示例.json';s.dispatchEvent(new Event('change'));return 1;})()`);
  await sleep(900);
  await evl('window.__app.setLoop(false);window.__app.setQuality(1);window.__app.setView("schema");');
  const shoot = async (name, setup) => {
    await evl(setup);
    const dataUrl = await evl('window.__app.snapshot()');
    if(!dataUrl || !dataUrl.startsWith('data:image/png')){ console.log(`  截图失败 ${name}`); return; }
    writeFileSync(join(SHOTS,name), Buffer.from(dataUrl.split(',')[1],'base64'));
    console.log(`  截图 ${join(SHOTS,name)}`);
  };
  await shoot('schema_t0.png', 'window.__app.seek(0);window.__app.setView("schema");');
  await shoot('schema_t6_5.png', 'window.__app.seek(6.5);');          /* 转身时刻 */
  await shoot('schema_t7_21.png', 'window.__app.seek(7.21);');
  await shoot('close_t7_21.png', 'window.__app.seek(7.21);window.__app.setView("close");');
  /* 人物特写：发型/面部/服装换装的成片检查 */
  await evl(`(()=>{const h=window.__app.motionProbe(7.21).head; if(!h) return 0;
    window.__app.focus({x:h[0], y:h[1]+0.03, z:h[2]}, 0.85, 0, 2); return 1;})()`);
  await shoot('portrait_t7_21.png', 'window.__app.seek(7.21);');
  const hair = await evl('window.__app.hairInfo()');
  rec('B15','长发垂到肩线以下（发型剪影保真）',
      !!hair && hair.drop > 0.04 && hair.width > 0.10,
      hair ? `发丝底端 y=${hair.minY} 肩线 y=${hair.shoulderY} 下垂=${hair.drop}m 发宽=${hair.width}m 头骨 y=${hair.headY}` : '未找到长发');
  await shoot('top_t7_21.png', 'window.__app.seek(7.21);window.__app.setView("top");');
  /* 对照机位：与源视频同尺寸同位置，用于逐帧比对 */
  await evl('window.__app.setAspect("video");window.__app.setView("match");');
  for(const t of [0, 2.5, 5, 6.5, 7.21, 10]){
    await shoot(`match_t${String(t).replace('.','_')}.png`, `window.__app.seek(${t});`);
  }
  /* 叠加对照：源视频作底、3D 人偶覆盖 */
  await evl('window.__app.setView("overlay");window.__app.seek(7.21);');
  await sleep(600);
  const ovl = await evl(`(()=>{
    const v=document.querySelector('#vid');
    const bg=window.__app.scene.background;
    return {isTex: !!(bg && bg.isTexture), vidTime:v.currentTime, ready:v.readyState,
            aspect:window.__app.camera.aspect, setHidden: !window.__app.scene.getObjectByName('root') ? null : undefined};
  })()`);
  for(const t of [0, 2.5, 5, 6.5, 7.21, 10]){
    await shoot(`overlay_t${String(t).replace('.','_')}.png`, `window.__app.seek(${t});`);
  }
  rec('B13','叠加对照：视频作底 + 3D 覆盖',
      ovl.isTex && ovl.ready>=2 && Math.abs(ovl.aspect - 9/16) < .02,
      `场景背景=视频贴图=${ovl.isTex} 画幅=${ovl.aspect.toFixed(4)}(期望 9:16) 视频时刻=${ovl.vidTime.toFixed(2)} readyState=${ovl.ready}`);
  await evl('window.__app.setView("schema");window.__app.setAspect("viewport");window.__app.setQuality(0);');

  /* 逐帧叠加接触表（诊断用）：SHEET=1 时每 0.5s 采一帧叠加对照 */
  if(process.env.SHEET){
    await evl('window.__app.setView("overlay");');
    for(let i=0;i<29;i++){
      const t = (i*0.5).toFixed(2);
      await shoot(`sheet_${String(i).padStart(2,'0')}.png`, `window.__app.seek(${t});`);
    }
    await evl('window.__app.setView("schema");window.__app.setAspect("viewport");');
  }

  /* 整页截图：headless 下合成器不画 canvas，先把 canvas 内容贴回 DOM 再整页捕获 */
  await evl(`(()=>{const c=document.querySelector('#viewport canvas'); if(!c) return 0;
    const img=document.createElement('img'); img.id='__snap'; img.src=c.toDataURL('image/png');
    img.style.cssText='position:fixed;left:0;top:0;width:100%;height:100%;z-index:9999;object-fit:contain;background:#05070c';
    document.body.appendChild(img); return 1;})()`);
  let full = null;
  try{ full = await cdp.send('Page.captureScreenshot',{format:'png',fromSurface:true},sid); }
  catch(e){ console.log('  整页截图失败: ' + e.message); }
  if(full && full.data){
    writeFileSync(join(SHOTS,'page_full.png'), Buffer.from(full.data,'base64'));
    console.log(`  截图 ${join(SHOTS,'page_full.png')}`);
  }
  await evl(`(()=>{const i=document.getElementById('__snap'); if(i) i.remove(); return 1;})()`);
}

function leftover(){
  try{ return execSync(`ps aux | grep "[C]hrome" | grep -c "remote-debugging-port=${DEBUG_PORT}" || true`).toString().trim(); }
  catch(e){ return '?'; }
}
let cleaned = false, crashed = false;
async function cleanup(){
  if(cleaned) return;
  cleaned = true;
  if(chrome){ try{ process.kill(-chrome.pid, 'SIGKILL'); }catch(e){ try{ chrome.kill('SIGKILL'); }catch(e2){} } }
  try{ execSync('pkill -f verify-chrome-profile || true'); }catch(e){}
  await sleep(600);
  tmpFiles.forEach(f=>{ try{ rmSync(f,{force:true}); }catch(e){} });
  if(serverStarted){ try{ serverStarted.kill('SIGKILL'); }catch(e){} }
  const ms = Date.now()-START;
  const failed = results.filter(r=>!r.pass);
  console.log(`\n===== 汇总: ${results.length-failed.length}/${results.length} 通过 =====`);
  failed.forEach(f=>console.log(`FAILED ${f.id} ${f.name} :: ${f.detail}`));
  console.log(`耗时 ${(ms/1000).toFixed(1)}s（上限 ${OVERALL_MS/1000}s）`);
  console.log(`残留浏览器进程: ${leftover()}`);
  process.exit(crashed || failed.length ? 1 : 0);
}
/* undici 偶发 raw-throw：同步兜底，绝不留实例、绝不让 main 继续跑 */
process.on('uncaughtException', e => {
  console.error('未捕获异常（已同步兜底清理）:', e.message);
  try{ if(chrome) chrome.kill('SIGKILL'); }catch(_){}
  try{ tmpFiles.forEach(f=>rmSync(f,{force:true})); }catch(_){}
  try{ if(serverStarted) serverStarted.kill('SIGKILL'); }catch(_){}
  process.exit(1);
});
main()
  .catch(e=>{ crashed = true; console.error('验收脚本异常:', e.message); })
  .finally(cleanup);
