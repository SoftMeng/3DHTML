import {MODEL_URL, buildFigure, geometryBox, humanAsset, loadHumanModel, rig} from './figure.js';
import {$, DEG, FIG, TRAIL_COLORS, VIDEO_PITCH_ADJ, clamp, loadThree, normalize, samplePath, showError, yawOf} from '../main.js';
import {adaptQuality, ambLight, buildEnvironment, buildLights, buildSet, greyBox, markDirty, setQuality} from './scene.js';
import {MKEY, applyMotion, bodyIntervalDepth, bodyProxies, bodyTopology, bodyTriangles, bodyWorldNow, chainPenetration, chainTruthPen, chanYaw, denoiseMotion, deriveMotionChannels, footPlanted, joint2, lastChestRoll, meshPenetration, palmDesiredNormal, palmLabelAt, poseAt, resetFigurePose, rigAxes, skinVertexData, sm, videoBoxFrac} from './solver.js';
import {BUDGET, GHOST_COUNT, MM_SIZE, QUALITY_LEVELS, THREE, camState, clearScene, crowdMesh, dirty, groups, headRing, isCompareView, lastRenderAt, lastURLSync, lightBeams, mediaList, mm, motion, playhead, sceneRoot, setStateAspectMode, setStateCamState, setStateCamera, setStateControls, setStateDirty, setStateEnvGroup, setStateFrameMs, setStateFreeCam, setStateHeadRing, setStateLastRenderAt, setStateLastTickAt, setStateMediaList, setStateMimic, setStateMotion, setStatePlayhead, setStatePlaying, setStateRenderer, setStateScene, setStateSceneRoot, setStateSetGroup, setStateSpec, setStateSpecName, setStateSubmitMsLast, setStateTHREE, setStateVideoAspect, setStateVideoName, setStateView, spec, spotLight, view} from './state.js';

/* ---------- 其他主体 ---------- */
function buildSubjects(){
  spec.subjects.forEach(s => {
    if(s === spec.primary) return;
    if(spec.partKind[s.id] !== 'prop') return;
    const grp = new THREE.Group();
    const node = greyBox(s.cls, s.size);
    node.position.y = s.size[2]/2;
    grp.add(node);
    sceneRoot.add(grp);
    groups.set(s.id, grp);
  });
}
function trailHeight(s){
  const m = spec.partHeight[s.id];
  return m === null ? s.size[2]/2 : m;
}
function buildTrails(){
  spec.subjects.forEach((s,i) => {
    if(s.path.length < 2) return;
    const y = trailHeight(s);
    const col = new THREE.Color(TRAIL_COLORS[i % TRAIL_COLORS.length]);
    const positions = [], indices = [], colors = [];
    for(let k=0;k<s.path.length-1;k++){
      const a = s.path[k], b = s.path[k+1];
      positions.push(a.x,y,a.d, b.x,y,b.d);
      colors.push(col.r,col.g,col.b, col.r,col.g,col.b);
      indices.push(k*2, k*2+1);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors,3));
    geo.setIndex(indices);
    const line = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:.26}));
  line.visible = !isCompareView();
    line.frustumCulled = false;
    sceneRoot.add(line);
    trails.set(s.id, {line, geo, colors, color:col, segs:s.path.length-1, count:s.path.length});
  });
}
function updateTrailColors(s, tr, t){
  const arr = tr.geo.attributes.color.array;
  for(let k=0;k<tr.segs;k++){
    const done = s.path[k+1].t <= t + 1e-9;
    const f = done ? 1 : 0.22;
    arr[k*6+0] = tr.color.r*f; arr[k*6+1] = tr.color.g*f; arr[k*6+2] = tr.color.b*f;
    arr[k*6+3] = arr[k*6+0]; arr[k*6+4] = arr[k*6+1]; arr[k*6+5] = arr[k*6+2];
  }
  tr.geo.attributes.color.needsUpdate = true;
}
function buildTrailBits(){
  const p = spec.primary;
  if(!p) return;
  const H = p.size[2];
  const geo = new THREE.SphereGeometry(H*.045, 12, 10);
  const mat = new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:.34, depthWrite:false});
  const inst = new THREE.InstancedMesh(geo, mat, GHOST_COUNT);
  const dummy = new THREE.Object3D();
  for(let i=0;i<GHOST_COUNT;i++){
    dummy.position.set(0,-99,0); dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
    inst.setColorAt(i, new THREE.Color(0xffc38a));
  }
  inst.instanceMatrix.needsUpdate = true;
  if(inst.instanceColor) inst.instanceColor.needsUpdate = true;
  inst.frustumCulled = false;
  sceneRoot.add(inst);
  ghosts.push({inst, dummy});
  const rg = new THREE.RingGeometry(H*.14, H*.19, 48);
  setStateHeadRing(new THREE.Mesh(rg, new THREE.MeshBasicMaterial({color:0xffc38a, transparent:true, opacity:.9, side:THREE.DoubleSide})));
  headRing.rotation.x = -Math.PI/2;
  sceneRoot.add(headRing);
}
function buildEvents(){
  const p = spec.primary;
  for(const ev of spec.events){
    const t = Number(ev.t ?? ev.time ?? ev.start);
    if(!Number.isFinite(t)) continue;
    const base = p ? samplePath(p.path, t) : {x:0,d:0};
    const px = ev.pos_m ? -(Number(ev.pos_m[0])) : base.x;
    const pd = ev.pos_m ? Number(ev.pos_m[1]) : base.d;
    const pin = new THREE.Mesh(new THREE.ConeGeometry(.16,.55,12), new THREE.MeshBasicMaterial({color:0xff6b6b}));
    pin.position.set(px, 1.7, pd);
    pin.rotation.x = Math.PI;
    pin.userData.t = t;
    sceneRoot.add(pin);
    eventMarks.push(pin);
  }
}

/* ---------- 场景装配 ---------- */
function buildScene(){
  clearScene();
  setStateSceneRoot(new THREE.Group());
  scene.add(sceneRoot);
  setStateEnvGroup(buildEnvironment());
  sceneRoot.add(envGroup);
  setStateSetGroup(buildSet());
  sceneRoot.add(setGroup);
  sceneRoot.add(buildLights());
  buildFigure();
  buildSubjects();
  buildTrails();
  buildTrailBits();
  buildEvents();
  setStatePlayhead(spec.shot.start);
  syncTimeUI();
  updateScene();
  updateCamera();
  updateFramingInfo();
}

/* ---------- 相机（schema v1 语义） ---------- */
function setCameraOrientation(pos, dir, up){
  const zAxis = new THREE.Vector3(-dir.x,-dir.y,-dir.z).normalize();
  const xAxis = new THREE.Vector3().crossVectors(up, zAxis).normalize();
  const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis);
  camera.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(xAxis,yAxis,zAxis));
  camera.position.set(pos.x,pos.y,pos.z);
}
/* 脚本机位必须同步 controls.target：否则 controls.update() 会把相机拽回它自己的内部状态 */
function applyCamera(pos, dir, up, lookDist){
  camera.up.set(0,1,0);
  setCameraOrientation(pos, dir, up);
  controls.target.set(pos.x+dir.x*lookDist, pos.y+dir.y*lookDist, pos.z+dir.z*lookDist);
  controls.update();
}
/* 髋关节（股骨头）当前世界高度 —— 对照机位的相机基准、视频对齐校验都用它 */
export function rigHipJointY(){
  if(!rig || !rig.nodes) return null;
  const a = rig.nodes.get('hipL'), b = rig.nodes.get('hipR');
  if(!a || !b) return rig.hipJointY ?? null;
  return (a.getWorldPosition(new THREE.Vector3()).y + b.getWorldPosition(new THREE.Vector3()).y)/2;
}
/* 髋关节中点（世界坐标）—— 与源视频的髋关键点同一解剖位置 */
export function rigHipJointPoint(){
  if(!rig || !rig.nodes) return null;
  const a = rig.nodes.get('hipL'), b = rig.nodes.get('hipR');
  if(!a || !b) return null;
  return a.getWorldPosition(new THREE.Vector3()).add(b.getWorldPosition(new THREE.Vector3())).multiplyScalar(.5);
}
function updateCamera(){
  if(!spec || !spec.primary) return;
  const cam = spec.camera;
  const p = samplePath(spec.primary.path, playhead);
  const p0 = samplePath(spec.primary.path, spec.shot.start);
  const h = cam.height_m;
  const pitch = (cam.pitch_deg + pitchAdj)*DEG;
  const dx = p.x - p0.x, dz = p.d - p0.d;
  let pos = {x:0, y:h, z:0};
  let yaw = 0, pit = pitch;
  const aimYaw = () => Math.atan2(p.x-pos.x, p.d-pos.z);

  if(view === 'audience'){
    const aud = spec.set.find(s => /audience|grandstand|观众|看台/i.test(s.cls));
    const H = spec.primary.size[2];
    const ax = aud ? aud.x : p.x;
    const ad = aud ? aud.d - aud.size[1]/2 - 1.5 : p.d + 15;
    pos = {x:ax, y:2.4, z:ad};
    const v = {x:p.x-pos.x, y:H*.5-pos.y, z:p.d-pos.z};
    const len = Math.hypot(v.x,v.y,v.z)||1;
    const dir = {x:v.x/len, y:v.y/len, z:v.z/len};
    applyCamera(pos, dir, {x:0,y:1,z:0}, len);
    setStateCamState({pos:{...camera.position}, dir});
    return;
  }
  if(view === 'match' || view === 'overlay'){
    /* 对照机位：按源视频里主体的画面占比与髋中点屏幕位置反解相机，
       基准取髋中点（动作数据的原点）：关键点包围盒中心会随手臂伸展偏移。
       使 3D 人偶与视频里的表演者同尺寸同位置 —— 逐帧对照的前提。 */
    const H = spec.primary.size[2];
    const p = samplePath(spec.primary.path, playhead);
    const box = videoBoxFrac(playhead);
    const fh = box ? clamp(box[0], .10, .95) : .45;
    const sx = box && box.length>1 ? box[1] : .5;
    const sy = box && box.length>2 ? box[2] : .5;
    const halfV = camera.fov/2*DEG;
    const dist = clamp(H/Math.max(.05, 2*Math.tan(halfV)*fh), .8, 80);
    const camH = cam.height_m;
    /* 相机基准高度取**髋关节**（股骨头）当前高度：源视频的髋关键点也在股骨头位置，
       用骨盆根节点会整体差 8.6cm（人偶与视频表演者错位）。 */
    const hipY = rigHipJointY() ?? ((rig && rig.restHipY) || H*.55);
    const halfH = Math.atan(Math.tan(halfV)*camera.aspect);
    const yaw = Math.atan((sx-.5)*2*Math.tan(halfH));
    const pit = Math.atan2(hipY - camH, dist) - (sy-.5)*camera.fov*DEG;
    pos = {x:p.x, y:camH, z:p.d - dist};
    const dir = {x:Math.sin(yaw)*Math.cos(pit), y:Math.sin(pit), z:Math.cos(yaw)*Math.cos(pit)};
    applyCamera(pos, dir, {x:0,y:1,z:0}, dist);
    setStateCamState({pos:{...camera.position}, dir});
    return;
  }
  if(view === 'close'){
    const H = spec.primary.size[2];
    const p = samplePath(spec.primary.path, playhead);
    const dist = H*0.9, ang = -0.30;
    pos = {x: p.x + Math.sin(ang)*dist, y: H*0.58, z: p.d - Math.cos(ang)*dist};
    const v = {x:p.x-pos.x, y:H*0.52-pos.y, z:p.d-pos.z};
    const len = Math.hypot(v.x,v.y,v.z)||1;
    const dir = {x:v.x/len, y:v.y/len, z:v.z/len};
    applyCamera(pos, dir, {x:0,y:1,z:0}, len);
    setStateCamState({pos:{...camera.position}, dir});
    return;
  }
  if(view === 'top'){
    const b = worldBounds();
    const span = Math.max(b.maxX-b.minX, b.maxD-b.minD) + .6;
    const need = span/2/Math.tan(cam.fov_deg/2*DEG)*1.2;
    pos = {x:(b.minX+b.maxX)/2, y:b.maxY+need, z:(b.minD+b.maxD)/2};
    const dir = {x:0, y:-1, z:0};
    applyCamera(pos, dir, {x:0,y:0,z:1}, need);
    setStateCamState({pos:{...camera.position}, dir});
    return;
  }
  if(freeCam) return;

  switch(cam.move){
    case 'follow':
      pos.x += dx; pos.z += dz; yaw = aimYaw(); break;
    case 'orbit': {
      const R = Math.hypot(p0.x, p0.d) || 6;
      const base = Math.atan2(p0.x, p0.d);
      const th = base + 40*DEG*((playhead - spec.shot.start)/Math.max(.001, spec.shot.end-spec.shot.start));
      pos.x = p.x - Math.sin(th)*R; pos.z = p.d - Math.cos(th)*R;
      yaw = aimYaw(); break;
    }
    case 'push_in':  pos.z += dz; break;
    case 'pull_out': pos.z -= dz; break;
    case 'truck':    pos.x += dx; break;
    case 'crane':    pos.y += 0; break;
    case 'pan':      yaw = Math.atan2(p.x-p0.x, p.d-p0.d)*.35; break;
    case 'tilt':     pit = pitch + Math.atan2(p.d-p0.d, Math.max(.001,Math.hypot(p.x-p0.x,p.d-p0.d)))*.5; break;
    default: break;
  }
  const dir = {x:Math.sin(yaw)*Math.cos(pit), y:Math.sin(pit), z:Math.cos(yaw)*Math.cos(pit)};
  applyCamera(pos, dir, {x:0,y:1,z:0}, 3);
  setStateCamState({pos:{...camera.position}, dir});
}

/* ---------- 每帧更新 ---------- */
export function updateScene(){
  if(!spec) return;
  const prim = spec.primary;
  spec.subjects.forEach(s => {
    const p = samplePath(s.path, playhead);
    if(s === prim){
      if(figGroup){
        const facing = p.facing !== null ? p.facing : 0;
        figGroup.position.set(p.x, 0, p.d);
        figGroup.rotation.y = yawOf(facing);
      }
    }else{
      const grp = groups.get(s.id);
      if(grp){ grp.position.set(p.x, 0, p.d); grp.rotation.y = yawOf(p.facing ?? 0); }
    }
    const tr = trails.get(s.id);
    if(tr) updateTrailColors(s, tr, playhead);
  });
  if(mimic && motion && prim) applyMotion(playhead);
  if(spotBeam) spotBeam.visible = !isCompareView();   /* 对照机位：相机常在光锥内，锥面刻面会露馅 */
  lightBeams.forEach(bm => { bm.visible = !isCompareView(); });
  trails.forEach(tr => tr.line.visible = !isCompareView());
  if(prim && headRing){
    const p = samplePath(prim.path, playhead);
    headRing.position.set(p.x, .03, p.d);
    const span = Math.max(.5, spec.shot.end - spec.shot.start);
    const gi = ghosts[0];
    if(gi){
      gi.inst.visible = !isCompareView();          /* 对照机位：隐藏残影，避免干扰比对 */
      ghosts.forEach((g,i)=>{
        const gt = playhead - (i+1)*span/26;
        const vis = gt >= spec.shot.start;
        g.dummy.position.set(0,-99,0);
        g.dummy.scale.setScalar(1);
        if(vis){
          const gs = samplePath(prim.path, gt);
          g.dummy.position.set(gs.x, prim.size[2]*FIG.headY, gs.d);
          g.dummy.scale.setScalar(1 - i/ghosts.length*.75);
        }
        g.dummy.updateMatrix();
        g.inst.setMatrixAt(i, g.dummy.matrix);
        g.inst.setColorAt(i, new THREE.Color(0xffc38a).multiplyScalar(vis ? (1 - i/ghosts.length) : 0));
      });
      gi.inst.instanceMatrix.needsUpdate = true;
      if(gi.inst.instanceColor) gi.inst.instanceColor.needsUpdate = true;
    }
  }
  if(lightPool && prim){
    const p = samplePath(prim.path, playhead);
    const mx = Math.max(0, (poolOrigin?.halfW ?? 7) - 3.5), mz = Math.max(0, (poolOrigin?.halfD ?? 7) - 3.5);
    lightPool.position.x = clamp(p.x - (poolOrigin?.x || 0), -mx, mx);
    lightPool.position.z = clamp(p.d - (poolOrigin?.z || 0), -mz, mz);
  }
  if(spotLight && prim){
    const p = samplePath(prim.path, playhead);
    spotLight.position.set(p.x, 9, p.d - 4.2);
    spotTarget.position.set(p.x, prim.size[2]*.5, p.d);
    spotBeam.position.set(p.x, 4.9, p.d - 2.2);
    spotBeam.lookAt(p.x, 0, p.d);
    spotBeam.rotateX(-Math.PI/2);
  }
  eventMarks.forEach(m => { m.visible = Math.abs(playhead - m.userData.t) < .6; });
}

/* ---------- 取景比例自检 ---------- */
function framingRatio(){
  if(!spec || !spec.primary) return null;
  const p = samplePath(spec.primary.path, spec.shot.start);
  const H = spec.primary.size[2];
  const dist = Math.hypot(p.x-camState.pos.x, p.d-camState.pos.z) || 1;
  const halfV = camera.fov/2*DEG;
  const frameH = 2*dist*Math.tan(halfV);
  const hCam = camState.pos.y;
  const angTop = Math.atan2(H-hCam, dist), angBot = Math.atan2(-hCam, dist);
  const vis = Math.abs(angTop-angBot);
  return {r: vis/(2*halfV), dist, frameH};
}
function framingBand(f){
  if(f < .35) return 'wide';
  if(f < .8) return 'full';
  if(f < 1.6) return 'medium';
  return 'close';
}
function updateFramingInfo(){
  if(!spec) return;
  const f = framingRatio();
  if(!f) return;
  const band = framingBand(f.r);
  const ok = band === spec.camera.framing;
  const el = $('#framingInfo');
  if(!el) return;
  const html = `取景 r=<b>${f.r.toFixed(3)}</b> → <b>${band}</b>（规格 ${spec.camera.framing}${ok?' ✓':' ≠'}）· 水平FOV ${spec.camera.fov_deg}° / 垂直 ${camera.fov.toFixed(1)}°`;
  if(el.innerHTML !== html) el.innerHTML = html;
}

/* ---------- 小地图（schema 视向：+x 右 / +d 上） ---------- */
function worldBounds(){
  let minX=Infinity,maxX=-Infinity,minD=Infinity,maxD=-Infinity,maxY=0;
  spec.subjects.forEach(s => s.path.forEach(p => {
    minX=Math.min(minX,p.x); maxX=Math.max(maxX,p.x);
    minD=Math.min(minD,p.d); maxD=Math.max(maxD,p.d);
    maxY=Math.max(maxY, spec.partHeight[s.id] ?? s.size[2]/2);
  }));
  if(!Number.isFinite(minX)){ minX=maxX=minD=maxD=0; }
  return {minX,maxX,minD,maxD,maxY};
}
function drawMinimap(){
  if(!$('#tgMap').checked || !spec) return;
  const W = MM_SIZE, H = MM_SIZE, pad = 14;
  mmCtx.clearRect(0,0,W,H);
  const b = worldBounds();
  const spanX = Math.max(1, b.maxX-b.minX), spanD = Math.max(1, b.maxD-b.minD);
  const s = Math.min((W-pad*2)/spanX, (H-pad*2)/spanD, 200);
  const cx = (b.minX+b.maxX)/2, cd = (b.minD+b.maxD)/2;
  /* 世界 x 已镜像，画布按 schema +x 向右 */
  const toMap = (x,d) => ({px: W/2-(x-cx)*s, py: H/2-(d-cd)*s});
  mmCtx.strokeStyle = 'rgba(255,255,255,.08)'; mmCtx.lineWidth = 1;
  for(let i=0;i<=4;i++){
    const gx = pad+(W-pad*2)*i/4, gy = pad+(H-pad*2)*i/4;
    mmCtx.beginPath(); mmCtx.moveTo(gx,pad); mmCtx.lineTo(gx,H-pad); mmCtx.stroke();
    mmCtx.beginPath(); mmCtx.moveTo(pad,gy); mmCtx.lineTo(W-pad,gy); mmCtx.stroke();
  }
  spec.subjects.forEach((sub,i) => {
    if(sub.path.length < 2) return;
    mmCtx.strokeStyle = '#'+TRAIL_COLORS[i%TRAIL_COLORS.length].toString(16).padStart(6,'0');
    mmCtx.lineWidth = 1.6; mmCtx.globalAlpha = .85;
    mmCtx.beginPath();
    sub.path.forEach((p,k) => { const m = toMap(p.x,p.d); k ? mmCtx.lineTo(m.px,m.py) : mmCtx.moveTo(m.px,m.py); });
    mmCtx.stroke(); mmCtx.globalAlpha = 1;
  });
  if(spec.primary){
    const p = samplePath(spec.primary.path, playhead);
    const m = toMap(p.x,p.d);
    mmCtx.fillStyle = '#fff';
    mmCtx.beginPath(); mmCtx.arc(m.px,m.py,4,0,Math.PI*2); mmCtx.fill();
  }
  const cam = toMap(camState.pos.x, camState.pos.z);
  const half = spec.camera.fov_deg/2*DEG;
  const ang = Math.atan2(-camState.dir.z, -camState.dir.x);
  mmCtx.fillStyle = 'rgba(232,180,92,.2)';
  mmCtx.beginPath(); mmCtx.moveTo(cam.px,cam.py);
  mmCtx.arc(cam.px,cam.py,26,ang-half,ang+half); mmCtx.closePath(); mmCtx.fill();
  mmCtx.fillStyle = '#e8b45c';
  mmCtx.beginPath(); mmCtx.arc(cam.px,cam.py,3.2,0,Math.PI*2); mmCtx.fill();
}

/* ---------- 录制：把画布（含叠加对照的视频底）录成 webm ---------- */
let recorder = null, recChunks = [];
function toggleRecording(){
  const btn = $('#recBtn');
  if(recorder && recorder.state !== 'inactive'){
    recorder.stop();
    recorder = null;
    btn.setAttribute('aria-pressed','false');
    btn.querySelector('.rec-label').textContent = '录制';
    return;
  }
  if(typeof MediaRecorder === 'undefined' || !renderer.domElement.captureStream){
    btn.querySelector('.rec-label').textContent = '不支持';
    return;
  }
  const canvas = renderer.domElement;
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9'
              : (MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '');
  recChunks = [];
  recorder = new MediaRecorder(stream, mime ? {mimeType:mime, videoBitsPerSecond: 8e6} : undefined);
  recorder.ondataavailable = e => { if(e.data && e.data.size) recChunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(recChunks, {type:'video/webm'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `叠加对照_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.webm`;
    a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href), 8000);
    recChunks = [];
  };
  recorder.start();
  btn.setAttribute('aria-pressed','true');
  btn.querySelector('.rec-label').textContent = '停止';
}
$('#recBtn').onclick = toggleRecording;

/* ---------- 叠加对照：源视频作底、3D 覆盖 ---------- */
let overlayOn = false, baseBg = 0x070810, vidTex = null;
/* 视频贴图直接当场景背景：合成发生在 WebGL 画布内部，
   截图/导出拿到的是「视频 + 人偶」合成结果，而不是只有人偶。 */
function videoTexture(){
  if(!vidTex){
    const v = $('#vid');
    vidTex = new THREE.VideoTexture(v);
    vidTex.colorSpace = THREE.SRGBColorSpace;
    vidTex.minFilter = THREE.LinearFilter;
    vidTex.magFilter = THREE.LinearFilter;
  }
  return vidTex;
}
function setOverlay(on){
  if(on === overlayOn) return;
  overlayOn = on;
  if(on){
    if(scene){ baseBg = scene.background && scene.background.getHex ? scene.background.getHex() : baseBg; }
    if(setGroup) setGroup.visible = false;
    if(envGroup) envGroup.visible = false;
    if(aspectMode !== 'video'){ aspectMode = 'video'; $('#aspectSel').value = 'video'; }
    layoutFrame();
    if(scene) scene.background = videoTexture();
    /* 叠加时人偶要「像被舞台灯打亮」，否则是黑剪影，对照效果失真 */
    if(ambLight) ambLight.intensity = .9;
    if(spotMain) spotMain.intensity = 340;
  }else{
    if(scene) scene.background = new THREE.Color(baseBg);
    if(setGroup) setGroup.visible = true;
    if(envGroup) envGroup.visible = true;
    if(ambLight) ambLight.intensity = .5;
    if(spotMain) spotMain.intensity = 220;
    layoutFrame();
  }
  markDirty();
}

/* ---------- 镜头切换（深链还原与调试钩子共用同一实现） ---------- */
function setView(v){
  setStateView(v);
  freeCam = false;
  document.querySelectorAll('#viewSeg button').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.view===v)));
  setOverlay(v === 'overlay');
  updateCamera(); markDirty(); syncURL();
}

/* ---------- 画幅 / letterbox ---------- */
function currentAspect(){
  if(aspectMode === '9:16') return 9/16;
  if(aspectMode === '16:9') return 16/9;
  if(aspectMode === 'video') return videoAspect;
  return null;
}
function layoutFrame(){
  const W = innerWidth, H = innerHeight;
  const a = currentAspect();
  let fw = W, fh = H;
  if(a && isFinite(a)){ if(W/H > a) fw = H*a; else fh = W/a; }
  const vp = $('#viewport');
  vp.style.left = ((W-fw)/2)+'px'; vp.style.top = ((H-fh)/2)+'px';
  vp.style.width = fw+'px'; vp.style.height = fh+'px';
  if(overlayOn){
    const v = $('#vid');
    v.style.left = ((W-fw)/2)+'px'; v.style.top = ((H-fh)/2)+'px';
    v.style.width = fw+'px'; v.style.height = fh+'px';
  }
  if(renderer) renderer.setSize(fw, fh, false);
  if(camera){
    camera.aspect = fw/fh;
    const hfov = spec ? spec.camera.fov_deg : 50;
    camera.fov = 2*Math.atan(Math.tan(hfov/2*DEG)/camera.aspect)/DEG;
    camera.updateProjectionMatrix();
  }
  updateFramingInfo();
  markDirty();
}
/* 画幅切换（深链还原与调试钩子共用） */
function setAspect(m){
  setStateAspectMode(m);
  $('#aspectSel').value = m; layoutFrame(); markDirty(); syncURL();
}

/* ---------- 时间与播放 ---------- */
function syncTimeUI(){
  $('#tNow').textContent = playhead.toFixed(2);
  $('#tDur').textContent = spec ? spec.shot.end.toFixed(2) : '0.00';
  const sl = $('#timeSlider');
  if(spec){ sl.min = spec.shot.start; sl.max = spec.shot.end; }
  sl.value = playhead;
}
function setPlaying(v){
  setStatePlaying(v);
  $('#playBtn').setAttribute('aria-pressed', String(v));
  $('#playBtn').setAttribute('aria-label', v ? '暂停' : '播放');
  $('#icoPlay').style.display = v ? 'none' : '';
  $('#icoPause').style.display = v ? '' : 'none';
}
$('#playBtn').onclick = () => {
  setPlaying(!playing);
  if(playing){ const v=$('#vid'); if(v.src && v.readyState>=1) v.play().catch(()=>{}); }
  markDirty();
};
$('#timeSlider').oninput = e => { playhead = parseFloat(e.target.value); syncTimeUI(); updateScene(); updateCamera(); syncVideo(); markDirty(); };
$('#speedSel').onchange = e => { speed = parseFloat(e.target.value); };
$('#pitchAdj').oninput = e => { pitchAdj = parseFloat(e.target.value); $('#pitchVal').textContent = (pitchAdj>0?'+':'')+pitchAdj.toFixed(1)+'°'; updateCamera(); updateFramingInfo(); markDirty(); };
$('#alignBtn').onclick = () => { pitchAdj = VIDEO_PITCH_ADJ; $('#pitchAdj').value = pitchAdj; $('#pitchVal').textContent = '+'+pitchAdj+'°'; updateCamera(); updateFramingInfo(); markDirty(); };
$('#aspectSel').onchange = e => setAspect(e.target.value);
$('#railBtn').onclick = () => $('#hud').classList.toggle('open');
/* 键盘：空格播放/暂停 · ←→ 单帧步进 · Shift+←→ 1 秒 · Home/End 首尾
   1-6 切换镜头 · M 模仿开关 · R 录制 */
const VIEW_KEYS = ['schema','match','overlay','audience','close','top'];
document.addEventListener('keydown', e => {
  if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable) return;
  const step = 1/(motion?.fps || 24);
  const seekTo = t => { playhead = clamp(t, spec.shot.start, spec.shot.end); syncTimeUI(); updateScene(); updateCamera(); syncVideo(); markDirty(); };
  switch(e.code){
    case 'Space': e.preventDefault(); setPlaying(!playing); return;
    case 'ArrowLeft': e.preventDefault(); seekTo(playhead - (e.shiftKey ? 1 : step)); return;
    case 'ArrowRight': e.preventDefault(); seekTo(playhead + (e.shiftKey ? 1 : step)); return;
    case 'Home': e.preventDefault(); seekTo(spec.shot.start); return;
    case 'End': e.preventDefault(); seekTo(spec.shot.end); return;
    case 'KeyM': e.preventDefault(); setMimic(!mimic); return;
    case 'KeyR': e.preventDefault(); toggleRecording(); return;
  }
  if(/^Digit[1-6]$/.test(e.code)){
    const v = VIEW_KEYS[+e.code.slice(5) - 1];
    if(v){ e.preventDefault(); window.__app.setView(v); }
  }
});
document.querySelectorAll('#viewSeg button').forEach(b => {
  b.onclick = () => {
    setStateView(b.dataset.view);
    freeCam = false;
    document.querySelectorAll('#viewSeg button').forEach(x => x.setAttribute('aria-pressed', String(x===b)));
    setOverlay(view === 'overlay');
    updateCamera();
    markDirty();
  };
});
['tgTrail','tgGhost','tgSet'].forEach(id => {
  $('#'+id).onchange = e => {
    const on = e.target.checked;
    if(id === 'tgTrail') trails.forEach(t => { t.line.visible = on; });
    if(id === 'tgGhost' && ghosts[0]) ghosts[0].inst.visible = on;
    if(id === 'tgSet' && setGroup) setGroup.visible = on;
    markDirty();
  };
});
$('#tgMap').onchange = e => {
  $('#mapwrap').hidden = !e.target.checked;
  $('#inspector').hidden = !e.target.checked && $('#vidwrap').hidden;
  markDirty();
};
$('#tgVideo').onchange = e => {
  $('#vidwrap').hidden = !e.target.checked;
  $('#inspector').hidden = !e.target.checked && $('#mapwrap').hidden;
  markDirty();
};

/* ---------- 模仿（源视频姿态驱动） ---------- */
function updateProvenance(){
  const model = humanAsset
    ? '人体模型：Mixamo <b>Michelle</b>（models/human.glb）'
    : '人体模型：<b>程序化占位人偶</b>（GLB 未加载）';
  const backend = motion && motion.model ? motion.model.replace(/^mediapipe-/, '').replace(/-/g, ' ') : '源视频检测';
  const pose = mimic ? `肢体动作逐帧模仿自源视频（${backend}）` : '肢体姿态取自源视频帧';
  const el = $('#provText');
  if(el) el.innerHTML = `环境与轨迹来自 JSON（schema v1）；${model}；<b id="provPose">${pose}</b>。`;
}
function setMimic(on){
  if(on && !motion) on = false;
  setStateMimic(!!on);
  const btn = $('#mimicBtn');
  btn.setAttribute('aria-pressed', String(mimic));
  updateProvenance();
  if(!mimic) resetFigurePose();
  updateScene();
  markDirty();
}
$('#mimicBtn').onclick = () => setMimic(!mimic);
async function tryLoadMotion(){
  if(location.protocol === 'file:') return null;   /* file:// 下 fetch 会被 CORS 拦截 */
  const cands = [];
  const specBase = specName.replace(/\.json$/i,'');
  if(specBase) cands.push(specBase+'.motion.json');
  const vidBase = (videoName || '').replace(/\.[^.]+$/,'');
  if(vidBase) cands.push(vidBase+'.motion.json');
  for(const c of cands){
    try{
      const res = await fetch(encodeURI(c), {cache:'no-store'});
      if(!res.ok) continue;
      const d = await res.json();
      if(d && Array.isArray(d.poses) && d.poses.length){
        /* v2：提取器已完成修复+去噪并给出语义通道（转身/侧倾/重心/接触），查看器只做重定向；
           v1：运行时补齐（兼容旧文件）。 */
        if(d.version >= 2 && d.chan){ d.posesD = d.poses; d.chanV2 = true; return d; }
        return deriveMotionChannels(denoiseMotion(d));
      }
    }catch(e){}
  }
  return null;
}
function refreshMimicUI(){
  const btn = $('#mimicBtn'), info = $('#mimicInfo');
  if(motion){
    btn.disabled = false;
    const dur = Math.abs((motion.duration||0) - (spec ? spec.shot.end : 0));
    const warn = dur > 1.5 ? ' · 时长不匹配' : '';
    info.textContent = `${motion.poses.length} 帧 · conf ${(motion.quality||0).toFixed(2)}${warn}`;
    info.title = warn ? `动作文件时长 ${motion.duration?.toFixed(2)}s ≠ 规格 ${spec?.shot.end?.toFixed(2)}s` : '';
  }else{
    btn.disabled = true;
    const base = specName.replace(/\.json$/i,'');
    info.textContent = `缺 ${base}.motion.json`;
    info.title = '生成：python3 tools/extract_motion.py <视频> --spec <规格.json>';
  }
}

/* ---------- 规格加载 ---------- */
const fileCache = new Map();
function addOption(sel, value, label){
  if([...sel.options].some(o => o.value === value)) return;
  const o = document.createElement('option');
  o.value = value; o.textContent = label || value;
  sel.appendChild(o);
}
async function fetchSpec(name){
  if(fileCache.has(name)) return fileCache.get(name);
  const res = await fetch(encodeURI(name), {cache:'no-store'});
  if(!res.ok) throw new Error(`读取 ${name} 失败：HTTP ${res.status}`);
  const raw = await res.json();
  fileCache.set(name, raw);
  return raw;
}
async function loadSpec(name, raw, deep){
  const data = raw || await fetchSpec(name);
  setStateSpec(normalize(data));
  setStateSpecName(name);
  setStateMotion(await tryLoadMotion());
  /* 先取动作文件，buildFigure 按其比例重建 */
  buildScene();
  setMimic(!!motion);
  refreshMimicUI();
  layoutFrame();
  markDirty();
  $('#specChip').textContent = name;
  const meta = [
    `时长 <b>${spec.shot.end.toFixed(2)}s</b>`,
    `主体 <b>${spec.subjects.length}</b>`,
    `布景 <b>${spec.set.length}</b>`,
    `move <b>${spec.camera.move}</b>`,
    spec.events.length ? `事件 <b>${spec.events.length}</b>` : ''
  ].filter(Boolean).join(' · ');
  $('#meta').innerHTML = `${meta} · 主视角 <b>${spec.primary?.id ?? '—'}</b>`;
  const evBox = $('#events');
  evBox.textContent = '';
  spec.events.forEach(ev => {
    const t = Number(ev.t ?? ev.time ?? ev.start);
    const label = ev.what || ev.label || ev.desc || ev.type || '事件';
    const row = document.createElement('div');
    const stamp = document.createElement('b');
    stamp.textContent = Number.isFinite(t) ? t.toFixed(2)+'s' : '—';
    row.appendChild(stamp);
    row.appendChild(document.createTextNode(' '+String(label)));
    evBox.appendChild(row);
  });
  window.__app.spec = spec;
  window.__app.currentSpecName = name;
  setPlaying(false);
  /* 深链：时间/镜头/画幅随规格一起还原（放在 ready 之前，调用方拿到即为最终态） */
  if(deep){
    if(deep.t != null){ playhead = clamp(deep.t, spec.shot.start, spec.shot.end); syncTimeUI(); }
    if(deep.aspect && ['viewport','9:16','16:9','video'].includes(deep.aspect)) setAspect(deep.aspect);
    if(deep.view && ['schema','match','overlay','audience','close','top'].includes(deep.view)) setView(deep.view);
  }
  syncURL();
  window.__app.ready = true;
}
$('#specSelect').onchange = async e => {
  const name = e.target.value;
  if(!name) return;
  try{ await loadSpec(name); }
  catch(err){ showError('载入规格失败', String(err.message||err), err.stack||''); }
};
$('#pickBtn').onclick = () => $('#fileInput').click();
$('#fileInput').onchange = async e => {
  for(const f of e.target.files){
    try{
      const raw = JSON.parse(await f.text());
      fileCache.set(f.name, raw);
      addOption($('#specSelect'), f.name, `${f.name}（本地）`);
    }catch(err){ showError('解析 JSON 失败', `${f.name}：${err.message||err}`, ''); }
  }
  e.target.value = '';
  const sel = $('#specSelect');
  if(sel.options.length > 1){
    sel.value = sel.options[sel.options.length-1].value;
    try{ await loadSpec(sel.value); }catch(err){ showError('载入规格失败', String(err.message||err), ''); }
  }
};
$('#urlBtn').onclick = () => {
  const v = prompt('输入 JSON 文件名或完整 URL');
  if(!v) return;
  const name = v.trim();
  addOption($('#specSelect'), name, name);
  $('#specSelect').value = name;
  loadSpec(name).catch(err => showError('载入规格失败', String(err.message||err), err.stack||''));
};

/* ---------- 深链：URL 携带 规格/时间/镜头/画幅，刷新与分享都回到同一画面 ---------- */
function syncURL(){
  if(!spec) return;
  const q = new URLSearchParams();
  q.set('spec', specName);
  if(Math.abs(playhead - spec.shot.start) > 1e-3) q.set('t', playhead.toFixed(2));
  if(view !== 'schema') q.set('view', view);
  if(aspectMode !== 'viewport') q.set('aspect', aspectMode);
  const url = location.pathname + '?' + q.toString();
  if(url !== location.pathname + location.search) history.replaceState(null, '', url);
}
function readURL(){
  const q = new URLSearchParams(location.search);
  const t = parseFloat(q.get('t'));
  return {spec:q.get('spec') || '', t:Number.isFinite(t) ? t : null,
          view:q.get('view') || '', aspect:q.get('aspect') || ''};
}

/* ---------- 视频对照 ---------- */
async function listMedia(){
  if(location.protocol === 'file:') return [];
  try{
    const res = await fetch('/api/media', {cache:'no-store'});
    if(!res.ok) return [];
    const j = await res.json();
    return Array.isArray(j) ? j : (j.media || []);
  }catch(e){ return []; }
}
function setVideoSrc(name){
  const v = $('#vid');
  setStateVideoName(name);
  v.src = encodeURI('/'+name);
  v.load();
  $('#vidHint').textContent = '时间轴与 3D 联动';
}
function syncVideo(){
  const v = $('#vid');
  if(!v.src) return;
  if(v.playbackRate !== speed) v.playbackRate = speed;
  if(playing){
    if(v.paused && v.readyState >= 2) v.play().catch(()=>{});
  }else if(!v.paused){ v.pause(); }
  if(v.readyState >= 1 && Math.abs(v.currentTime - playhead) > .08){
    try{ v.currentTime = Math.min(playhead, v.duration || playhead); }catch(e){}
  }
}
$('#vidSel').onchange = e => { if(e.target.value) setVideoSrc(e.target.value); };
$('#vid').addEventListener('loadeddata', () => { if(vidTex){ vidTex.dispose(); vidTex = null; } if(overlayOn && scene) scene.background = videoTexture(); });
$('#vid').addEventListener('loadedmetadata', () => {
  const v = $('#vid');
  setStateVideoAspect(v.videoWidth && v.videoHeight ? v.videoWidth/v.videoHeight : null);
  if(aspectMode === 'video') layoutFrame();
  markDirty();
});
$('#vid').addEventListener('play', markDirty);
$('#vid').addEventListener('pause', markDirty);
$('#vid').addEventListener('error', () => {
  $('#vidHint').textContent = '视频加载失败（文件名需 URL 编码 / 需 http 服务）';
});

async function listSpecs(){
  if(location.protocol === 'file:') return {items:[], src:null};
  try{
    const res = await fetch('/api/specs', {cache:'no-store'});
    if(res.ok){
      const j = await res.json();
      const items = Array.isArray(j) ? j : (j.specs||[]);
      if(items.length) return {items: items.map(x => typeof x==='string'?x:x.name), src:'目录自动枚举'};
    }
  }catch(e){}
  try{
    const res = await fetch('specs.json', {cache:'no-store'});
    if(res.ok){
      const j = await res.json();
      const items = Array.isArray(j) ? j : (j.specs||[]);
      if(items.length) return {items: items.map(x => typeof x==='string'?x:x.name), src:'specs.json 清单'};
    }
  }catch(e){}
  return {items:[], src:null};
}

/* ---------- 启动 ---------- */
async function boot(){
  const {THREE:T, OrbitControls:OC} = await loadThree();
  setStateTHREE(T);
  OrbitControls = OC;

  setStateScene(new THREE.Scene());
  setStateRenderer(new THREE.WebGLRenderer({antialias:true, alpha:true, powerPreference:'high-performance'}));
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  $('#viewport').appendChild(renderer.domElement);
  setStateCamera(new THREE.PerspectiveCamera(50, 1, .05, 1200));
  setStateControls(new OrbitControls(camera, renderer.domElement));
  controls.enableDamping = true;
  controls.dampingFactor = .08;
  controls.addEventListener('change', () => markDirty());
  controls.addEventListener('start', () => {
    setStateFreeCam(true);
    document.querySelectorAll('#viewSeg button').forEach(x => x.setAttribute('aria-pressed','false'));
    markDirty();
  });

  addEventListener('resize', () => {
    layoutFrame();
    sizeMinimap();
  });
  sizeMinimap();

  setStateMediaList(await listMedia());
  window.__app.media = mediaList;
  const vsel = $('#vidSel');
  vsel.innerHTML = '';
  const none = document.createElement('option');
  none.value = ''; none.textContent = '— 不对照 —';
  vsel.appendChild(none);
  mediaList.forEach(m => addOption(vsel, m.name, m.name));
  if(mediaList.length){
    vsel.value = mediaList[0].name;
    setVideoSrc(mediaList[0].name);
    setStateAspectMode('video');
    $('#aspectSel').value = 'video';
  }else{
    $('#vidwrap').hidden = true;
    $('#inspector').hidden = $('#mapwrap').hidden;
    $('#tgVideo').checked = false;
  }

  /* 人体模型（Mixamo 骨骼 GLB）：失败则回退程序化人偶 */
  const gltf = await loadHumanModel();
  if(gltf){
    const wrapper = new THREE.Group();
    wrapper.add(gltf.scene);
    /* 绑定姿势快照：wrapper 跨规格复用，重建时骨架带着上一规格动画的姿势，
       蒙皮包围盒（geometryBox 用蒙皮后顶点）会随姿势漂移 → 定标 k 与居中偏移漂
       20cm 级、rig.torso 漂 7%。测量前必须还原成出厂姿势。 */
    const bindPose = [];
    wrapper.updateMatrixWorld(true);
    wrapper.traverse(o => { if(o.isBone) bindPose.push({bone:o, q:o.quaternion.clone(), p:o.position.clone(), s:o.scale.clone()}); });
    humanAsset = {gltf, wrapper, bindPose};
  }
  updateProvenance();

  const found = await listSpecs();
  const sel = $('#specSelect');
  sel.innerHTML = '';
  const ph = document.createElement('option');
  ph.value = ''; ph.textContent = '— 选择规格 —';
  sel.appendChild(ph);
  found.items.forEach(n => addOption(sel, n));
  const deep = readURL();
  if(found.items.length){
    $('#banner').textContent = `已发现 ${found.items.length} 个规格文件（${found.src}）`;
    $('#banner').classList.remove('hidden');
    const want = found.items.includes(deep.spec) ? deep.spec : found.items[0];
    await loadSpec(want, null, deep);
    sel.value = want;
  }else{
    $('#banner').textContent = location.protocol === 'file:'
      ? 'file 模式：无法自动枚举目录，请用「本地选择…」载入 JSON；或运行 python3 serve.py 后刷新'
      : '未发现规格 JSON：请用「本地选择…」或「按路径载入…」';
    $('#banner').classList.remove('hidden');
  }
  layoutFrame();
  window.__app.ready = true;
  loop();
  setInterval(updatePerfInfo, 500);
}
function sizeMinimap(){
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  mm.width = MM_SIZE*dpr; mm.height = MM_SIZE*dpr;
  mm.style.width = MM_SIZE+'px'; mm.style.height = MM_SIZE+'px';
  mmCtx.setTransform(dpr,0,0,dpr,0,0);
}
function loop(){
  requestAnimationFrame(loop);
  if(!loopRunning) return;
  const now = performance.now();
  const dt = lastTickAt ? Math.min((now-lastTickAt)/1000, 1) : 0;
  setStateLastTickAt(now);
  if(playing && spec){
    const span = Math.max(.001, spec.shot.end - spec.shot.start);
    setStatePlayhead(playhead + dt*speed);
    if(playhead > spec.shot.end) playhead = spec.shot.start + (playhead - spec.shot.end)%span;
    syncTimeUI();
    setStateDirty(true);
    if(now - lastURLSync > 500){ lastURLSync = now; syncURL(); }   /* 深链跟随播放头（节流） */
  }
  if(!dirty) return;
  if(now - lastRenderAt < BUDGET.minFrameIntervalMs) return;
  const prevRenderAt = lastRenderAt;
  setStateLastRenderAt(now);
  const t0 = performance.now();
  updateScene();
  updateCamera();
  updateFramingInfo();
  controls.update();
  syncVideo();
  drawMinimap();
  renderer.render(scene, camera);
  const submitMs = performance.now() - t0;
  // WebGL 提交是异步的：真实负载体现在"两次绘制之间的间隔"，而非提交耗时
  setStateFrameMs((prevRenderAt && now - prevRenderAt < 2000) ? (now - prevRenderAt) : submitMs);
  setStateSubmitMsLast(submitMs);
  renderCount++;
  setStateDirty(false);
  adaptQuality(frameMs);
}
function updatePerfInfo(){
  const el = $('#perfInfo');
  if(!el || !renderer) return;
  const r = renderer.info.render;
  const fps = frameMs > 0 ? (1000/Math.max(frameMs,.1)).toFixed(1) : '—';
  const html = `fps <b>${fps}</b> · 帧时 <b>${frameMs.toFixed(1)}ms</b> · 画质 <b>${QUALITY_LEVELS[qualityLevel].id}</b> · 人群 <b>${crowdMesh?crowdMesh.count:0}</b> · Draw <b>${r.calls}</b> · 三角 <b>${(r.triangles/1000).toFixed(1)}k</b>`;
  if(el.innerHTML !== html) el.innerHTML = html;
}

/* ---------- 调试钩子 ---------- */
window.__app = {
  ready:false, specs:[], spec:null, currentSpecName:'', media:[],
  get playhead(){ return playhead; },
  get playing(){ return playing; },
  views:['schema','audience','close','top'],
  seek(t){ playhead = t; syncTimeUI(); updateScene(); updateCamera(); syncVideo(); markDirty(); syncURL(); return playhead; },
  setPlaying, setView, setAspect,
  toggleRecording, recording: () => !!(recorder && recorder.state !== 'inactive'),
  get renderCount(){ return renderCount; },
  get quality(){ return QUALITY_LEVELS[qualityLevel].id; },
  get frameMs(){ return frameMs; },
  get budget(){ return {...BUDGET}; },
  markDirty(){ dirty = true; },
  setQuality(l){ setQuality(l); },
  get motion(){ return motion ? {frames:motion.frames, fps:motion.fps, duration:motion.duration, quality:motion.quality, video:motion.video, model:motion.model} : null; },
  get model(){
    let boneNames=[], skinned=0, tris=0;
    if(humanAsset){
      humanAsset.wrapper.traverse(o=>{
        if(o.isBone) boneNames.push(o.name);
        if(o.isSkinnedMesh){ skinned++; const g=o.geometry; tris += g.index? g.index.count/3 : g.attributes.position.count/3; }
      });
    }
    return {loaded:!!humanAsset, url:MODEL_URL, kind:rig?rig.kind:'proc', torso:rig?+rig.torso.toFixed(4):null,
            boneNames:boneNames.slice(0,6), boneCount:boneNames.length, skinned, tris};
  },
  ikError(){
    if(!rig || !rig.lastTarget) return null;
    const out = {};
    for(const k of ['handL','handR','ankleL','ankleR']){
      const t = rig.lastTarget[k], o = rig.nodes.get(k);
      if(t && o){ const p = o.getWorldPosition(new THREE.Vector3()); out[k] = +p.distanceTo(t).toFixed(4); }
    }
    return out;
  },
  /* 单点穿模深度（诊断/自检用）：正 = 在躯干网格内，量级 = 到较近表面的距离 */
  penetrationAt(x,y,z){
    const sm = figGroup && figGroup.getObjectByProperty('isSkinnedMesh', true);
    if(!sm || !sm.geometry.index) return null;
    if(figGroup) figGroup.updateMatrixWorld(true);
    const skin = skinVertexData(sm);
    const body = bodyTriangles(sm, skin);
    if(!body) return null;
    const vc = videoCamState();
    const camPos = (vc && vc.pos) ? vc.pos : camera.position;
    const diag = {};
    const d = bodyIntervalDepth(new THREE.Vector3(x,y,z), body, camPos, diag);
    return {d:+d.toFixed(4), tp:+diag.tp.toFixed(4), hits:diag.hits};
  },
  /* 穿模明细（诊断）：逐采样点的代理穿透 vs 网格真值 —— 定位是手腕/肘/中段/指尖在体内 */
  collisionDetail(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); }
    const sm = figGroup && figGroup.getObjectByProperty('isSkinnedMesh', true);
    const ax = rigAxes();
    const out = {};
    if(sm && sm.geometry.index){
      if(figGroup) figGroup.updateMatrixWorld(true);
      const skin = skinVertexData(sm);
      const body = bodyTriangles(sm, skin);
      const vc = videoCamState();
      const camPos = (vc && vc.pos) ? vc.pos : camera.position;
      const wp = k => { const o = rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
      const R = rig.proxyR || {};
      for(const [tag,chain] of [['L',['shoulderL','elbowL','handL']],['R',['shoulderR','elbowR','handR']]]){
        const skip = ['upperArm'+tag,'foreArm'+tag,'hand'+tag];
        const caps = bodyProxies(skip, ax);
        const p0 = wp(chain[0]), p1 = wp(chain[1]), p2 = wp(chain[2]);
        if(!p0||!p1||!p2) continue;
        const r0 = R['upperArm'+tag]||.06, r1 = R['foreArm'+tag]||.05, r2 = R['hand'+tag]||.04;
        const samples = [['上臂中', p0.clone().lerp(p1,.5), r0], ['肘', p1, r1], ['前臂中', p1.clone().lerp(p2,.5), r1], ['腕', p2, r2]];
        const tipB = rig.fingerTip && rig.fingerTip[tag];
        if(tipB){ const tip = tipB.getWorldPosition(new THREE.Vector3());
          if(tip.distanceTo(p2)>1e-4){ samples.push(['指尖中', p2.clone().lerp(tip,.5), r2*.5]); samples.push(['指尖', tip, .004]); } }
        out[tag] = samples.map(([nm,p,r])=>{
          let proxy = -1e9;
          for(const c of caps){ const v = c.pen(p); if(v > proxy) proxy = v; }
          const truePen = bodyIntervalDepth(p, body, camPos);
          return {nm, proxy:+(proxy-r).toFixed(4), truth:+truePen.toFixed(4)};
        });
      }
    }
    if(Number.isFinite(t)) playhead = save;
    return out;
  },
  /* 碰撞状态（渲染后实测）：手臂/腿相对身体代理的穿透量 + 链采样点的网格真值 */
  collisionState(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); }
    const ax = rigAxes();
    const out = {};
    for(const [tag,chain] of [['L',['shoulderL','elbowL','handL']],['R',['shoulderR','elbowR','handR']]])
      out['arm'+tag] = +chainPenetration(chain, tag, ax).toFixed(4);
    for(const [tag,chain] of [['L',['hipL','kneeL','ankleL']],['R',['hipR','kneeR','ankleR']]])
      out['leg'+tag] = +chainPenetration(chain, tag, ax).toFixed(4);
    /* 真值：链采样点相对躯干网格的穿透（运行时的收敛判据） */
    const sm = figGroup && figGroup.getObjectByProperty('isSkinnedMesh', true);
    if(sm && sm.geometry.index){
      if(figGroup) figGroup.updateMatrixWorld(true);
      const topo = bodyTopology(sm);
      if(topo){
        const body = bodyWorldNow(sm, topo);
        const vc = videoCamState();
        const camPos = (vc && vc.pos) ? vc.pos : camera.position;
        for(const [tag,chain] of [['L',['shoulderL','elbowL','handL']],['R',['shoulderR','elbowR','handR']]])
          out['true'+tag] = +chainTruthPen(chain, tag, {body, camPos}).pen.toFixed(4);
      }
    }
    if(Number.isFinite(t)) playhead = save;
    return out;
  },
  /* 穿模收敛的逐阶段诊断：弯折侧向偏移 off0（推出前）/ offPush / offTilt + 推出量 */
  pushDiag(){
    return rig && rig.pushDiag ? JSON.parse(JSON.stringify(rig.pushDiag)) : null;
  },
  /* 网格级穿透实测（穿模判据）：手/前臂/袖筒的顶点 vs 躯干网格真实三角面 */
  penetration(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); }
    const r = meshPenetration();
    if(Number.isFinite(t)) playhead = save;
    return r;
  },
  /* 碰撞代理半径（实测值）+ 推出量统计 */
  proxyInfo(){    const R = rig && rig.proxyR ? rig.proxyR : null;
    const out = {};
    if(R) for(const k in R) out[k] = +R[k].toFixed(4);
    const pr = rig && rig.torsoProfile;
    const caps = bodyProxies(null, rigAxes()) || [];
    return {radii:out, push: rig && rig.pushOut ? {...rig.pushOut} : null,
            pushTruth: rig && rig.pushTruth ? {...rig.pushTruth} : null,
            profile: pr ? {t0:+pr.t0.toFixed(3), t1:+pr.t1.toFixed(3),
                           bins:pr.bins.map(v=>({lat:+v.lat.toFixed(3), dep:+v.dep.toFixed(3)}))} : null,
            caps: caps.map(c=>({n:c.name, kind:c.kind, r:c.kind==='cap'?+c.r.toFixed(3):null,
                                 h:c.kind==='ell'?[c.hLat,c.hDep,c.hUp].map(v=>+v.toFixed(3)):null})),
    chainPen:(()=>{ const ax = rigAxes(); const o = {};
              for(const [tg,ch] of [['L',['shoulderL','elbowL','handL']],['R',['shoulderR','elbowR','handR']]]) o['arm'+tg] = +chainPenetration(ch, tg, ax).toFixed(4);
              for(const [tg,ch] of [['L',['hipL','kneeL','ankleL']],['R',['hipR','kneeR','ankleR']]]) o['leg'+tg] = +chainPenetration(ch, tg, ax).toFixed(4);
              return o; })(),
            tip:(()=>{ const o = {};
              for(const tg of ['L','R']){
                const b = rig.fingerTip && rig.fingerTip[tg];
                const hb = rig.nodes.get(tg==='L'?'handL':'handR');
                if(b && hb){ const tp = b.getWorldPosition(new THREE.Vector3()), wr = hb.getWorldPosition(new THREE.Vector3());
                  o[tg] = {tip:tp.toArray().map(v=>+v.toFixed(3)), wrist:wr.toArray().map(v=>+v.toFixed(3)),
                           len:+tp.distanceTo(wr).toFixed(3)}; }
              } return o; })()};
  },
  get mimic(){ return mimic; },
  setMimic(b){ setMimic(b); },
  /* 弯折侧向：肘/膝相对「肩-手 / 髋-踝」弦线、沿角色前向的偏移。
     解剖约束：肘偏后（小臂朝前）、膝偏前（小腿朝后）——符号反了就是鸡翅膀/反关节。 */
  bendSide(t){
    if(!rig || !rig.nodes) return null;
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); playhead = save; }
    const fwd = new THREE.Vector3(0,0,1).applyQuaternion(rig.root.getWorldQuaternion(new THREE.Quaternion()));
    const side = (a,b,c)=>{
      const A=rig.nodes.get(a), B=rig.nodes.get(b), C=rig.nodes.get(c);
      if(!A||!B||!C) return null;
      const pA=A.getWorldPosition(new THREE.Vector3()), pB=B.getWorldPosition(new THREE.Vector3()), pC=C.getWorldPosition(new THREE.Vector3());
      const d=pC.clone().sub(pA);
      const len=d.length();
      if(len<1e-6) return null;
      d.divideScalar(len);
      const e=pB.clone().sub(pA);
      const perp=e.clone().addScaledVector(d, -e.dot(d));
      return +perp.dot(fwd).toFixed(4);
    };
    return {armL:side('shoulderL','elbowL','handL'), armR:side('shoulderR','elbowR','handR'),
            legL:side('hipL','kneeL','ankleL'), legR:side('hipR','kneeR','ankleR')};
  },
  /* 自由机位聚焦：相机置于目标点前方 dist 处，yawDeg 相对人物正面。
     人物特写截图与面部/发型检查用（freecam 后 updateCamera 不再接管相机）。 */
  focus(target, dist, yawDeg=0, pitchDeg=0){
    const t = new THREE.Vector3(target.x, target.y, target.z);
    let fx = 0, fz = 1;
    if(rig && rig.basis && rig.yawNode){
      const f = new THREE.Vector3(rig.basis.fwd.x, rig.basis.fwd.y, rig.basis.fwd.z)
        .applyQuaternion(rig.yawNode.getWorldQuaternion(new THREE.Quaternion()));
      fx = f.x; fz = f.z;
    }
    const fl = Math.hypot(fx, fz) || 1; fx /= fl; fz /= fl;
    const cy = Math.cos((yawDeg||0)*DEG), sy = Math.sin((yawDeg||0)*DEG);
    const pit = (pitchDeg||0)*DEG;
    const dir = new THREE.Vector3((fx*cy + fz*sy)*Math.cos(pit), Math.sin(pit),
      (-fx*sy + fz*cy)*Math.cos(pit)).normalize();
    camera.position.copy(t).addScaledVector(dir, dist);
    controls.target.copy(t);
    controls.update();
    setStateView('free');
    /* 绕开各机位分支（它们会自行摆相机） */
    setStateFreeCam(true);
    markDirty();
    return {pos:{x:camera.position.x, y:camera.position.y, z:camera.position.z}};
  },
  /* 手/脚在画面上的位置（与源视频逐点比对：视频一致性的直接度量） */
  handScreen(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); playhead = save; }
    if(!rig || !rig.nodes) return null;
    const proj = name => {
      const o = rig.nodes.get(name);
      if(!o) return null;
      const p = o.getWorldPosition(new THREE.Vector3()).project(camera);
      return {x:(p.x+1)/2, y:(p.y+1)/2};
    };
    return {handL:proj('handL'), handR:proj('handR'), footL:proj('footL'), footR:proj('footR'),
            ankleL:proj('ankleL'), ankleR:proj('ankleR')};
  },
  /* 掌向实测：当前掌面法线（世界系）与目标法线的夹角 */
  palmState(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); }
    const ax = rigAxes();
    const gp = k => { const o = rig.nodes && rig.nodes.get(k); return o ? o.getWorldPosition(new THREE.Vector3()) : null; };
    const out = {};
    for(const tag of ['L','R']){
      const pa = rig.palmAxis && rig.palmAxis[tag];
      const handB = rig.nodes && rig.nodes.get(tag==='L'?'handL':'handR');
      const elbowB = rig.nodes && rig.nodes.get(tag==='L'?'elbowL':'elbowR');
      if(!pa || !handB || !elbowB){ out[tag] = null; continue; }
      const wrist = handB.getWorldPosition(new THREE.Vector3());
      const elbow = elbowB.getWorldPosition(new THREE.Vector3());
      const F = elbow.clone().sub(wrist).normalize();
      const N = pa.axis.clone().applyQuaternion(handB.getWorldQuaternion(new THREE.Quaternion())).normalize();
      const Np = N.clone().addScaledVector(F, -N.dot(F)).normalize();
      const other = gp(tag==='L'?'handR':'handL');
      const D = palmDesiredNormal(tag, wrist, elbow, ax, other);
      const Dp = D.clone().addScaledVector(F, -D.dot(F)).normalize();
      out[tag] = {normal:Np.toArray().map(v=>+v.toFixed(3)),
                  desired:Dp.toArray().map(v=>+v.toFixed(3)),
                  angleDeg:+(Math.acos(clamp(Np.dot(Dp), -1, 1))*180/Math.PI).toFixed(1),
                  label:(palmLabelAt(playhead, tag)||{}).face || null,
                  source: rig.palmAxis[tag].source,
                  wrist:wrist.toArray().map(v=>+v.toFixed(3)),
                  elbow:elbow.toArray().map(v=>+v.toFixed(3)),
                  forearm:F.toArray().map(v=>+v.toFixed(3)),
                  other: other ? other.toArray().map(v=>+v.toFixed(3)) : null,
                  mid:(rigHipJointPoint() || wrist).toArray().map(v=>+v.toFixed(3)),
                  cam:(videoCamState()||{pos:new THREE.Vector3()}).pos.toArray().map(v=>+v.toFixed(3))};
    }
    if(Number.isFinite(t)) playhead = save;
    return out;
  },
  /* 人偶在画面上的实际投影范围（对照机位校验用）：
     相机反解是解析式的，真正落到画面上的人偶高度才是可比量。 */
  figureExtent(){
    if(!rig || !rig.nodes) return null;
    const proj = name => {
      const o = rig.nodes.get(name);
      if(!o) return null;
      const p = o.getWorldPosition(new THREE.Vector3()).project(camera);
      return {x:(p.x+1)/2, y:(p.y+1)/2};
    };
    /* 下沿量**鞋底**而不是踝骨：踝骨离地 12cm，用它当脚底会让剪影高度
       系统性偏矮 7~9%（E13 的「实测渲染占比」就是这么偏的）。 */
    const sole = name => {
      const o = rig.nodes.get(name);
      if(!o) return null;
      const p = o.getWorldPosition(new THREE.Vector3());
      p.y -= (rig.ankleGroundY || 0);
      const q = p.project(camera);
      return {x:(q.x+1)/2, y:(q.y+1)/2};
    };
    const head = proj('head'), fl = sole('footL'), fr = sole('footR');
    if(!head||!fl||!fr) return null;
    /* 上沿取**可见剪影的发顶**（外挂的头发/兜帽几何也算），不是头骨 ——
       头骨在颈，比发顶低 13%~18%，拿头骨比会系统性低估人偶高度。 */
    let crown = null;
    if(figGroup){
      const box = geometryBox(figGroup);
      const hp = rig.nodes.get('head').getWorldPosition(new THREE.Vector3());
      const p = new THREE.Vector3(hp.x, box.max.y, hp.z).project(camera);
      crown = {x:(p.x+1)/2, y:(p.y+1)/2};
    }
    const top = crown ? Math.max(crown.y, head.y) : head.y;
    const bot = Math.min(fl.y, fr.y);
    const box = figGroup ? geometryBox(figGroup) : null;
    const soleY = o => { const p = o.getWorldPosition(new THREE.Vector3()); p.y -= (rig.ankleGroundY||0); return p.y; };
    const aL = rig.nodes.get('ankleL')||rig.nodes.get('footL'), aR = rig.nodes.get('ankleR')||rig.nodes.get('footR');
    return {head, crown, footL:fl, footR:fr, top:+top.toFixed(4), bot:+bot.toFixed(4), frac:+(top-bot).toFixed(4),
            worldTop: box ? +box.max.y.toFixed(4) : null,
            worldSole: (aL&&aR) ? +Math.min(soleY(aL), soleY(aR)).toFixed(4) : null};
  },
  /* 长发/兜帽的几何信息：包围盒与肩线的关系（发型保真校验用） */
  hairInfo(){
    const head = rig && rig.nodes.get('head');
    const g = head && head.getObjectByName('hair');
    if(!head || !g) return null;
    const box = new THREE.Box3().setFromObject(g);
    const sh = rig.nodes.get('shoulderL'), sh2 = rig.nodes.get('shoulderR');
    const p = o => { const v = new THREE.Vector3(); o.getWorldPosition(v); return v; };
    const shoulderY = (sh && sh2) ? (p(sh).y + p(sh2).y)/2 : null;
    const headY = p(head).y;
    return {minY:+box.min.y.toFixed(4), maxY:+box.max.y.toFixed(4),
            width:+(box.max.x-box.min.x).toFixed(4), depth:+(box.max.z-box.min.z).toFixed(4),
            headY:+headY.toFixed(4), shoulderY: shoulderY==null?null:+shoulderY.toFixed(4),
            drop: shoulderY==null?null:+(shoulderY-box.min.y).toFixed(4)};
  },
  hipScreen(t){           /* 髋中点投影到屏幕（0..1，左上原点）—— 叠加对齐校验用 */
    const save=playhead;
    if(Number.isFinite(t)){ playhead=t; updateScene(); updateCamera(); playhead=save; }
    const hips = rigHipJointPoint();
    if(!hips) return null;
    const p = hips.project(camera);
    return {x:(p.x+1)/2, y:(p.y+1)/2};   /* 屏幕坐标：左下原点，y 向上 */
  },
  motionProbe(t){
    const save=playhead;
    if(Number.isFinite(t)){ playhead=t; updateScene(); updateCamera(); playhead=save; }
    const pose = poseAt(t);
    const g=n=>{const o=rig&&rig.nodes.get(n); if(!o) return null; const p=new THREE.Vector3(); o.getWorldPosition(p); return [+p.x.toFixed(3),+p.y.toFixed(3),+p.z.toFixed(3)];};
    return {
      version: motion&&motion.version, scale: rig&&rig.scale, restHipY: rig&&rig.restHipY,
      lastTarget: rig&&rig.lastTarget,
      hipWant: rig&&rig.hipWant, hipDelta: rig&&rig.hipDelta, hipsLocal: rig?[+rig.hipsRest.x.toFixed(4),+rig.hipsRest.y.toFixed(4),+rig.hipsRest.z.toFixed(4)]:null,
      hipsPos: rig?[+rig.nodes.get('hips').position.x.toFixed(4),+rig.nodes.get('hips').position.y.toFixed(4),+rig.nodes.get('hips').position.z.toFixed(4)]:null,
      hips:g('hips'), head:g('head'), ankleL:g('ankleL'), ankleR:g('ankleR'),
      footL:g('footL'), footR:g('footR'), shoulderL:g('shoulderL'), shoulderR:g('shoulderR'),
      handL:g('handL'), handR:g('handR'),
      ankY: pose?[pose[MKEY.lan],pose[MKEY.ran]]:null,
      shMid: pose?joint2(pose,MKEY.lsh,MKEY.rsh):null,
      wr: pose?[pose[MKEY.lwr],pose[MKEY.rwr]]:null,
      knee: pose?[pose[MKEY.lkn],pose[MKEY.rkn]]:null,
    };
  },
  rigDump(t){
    const save=playhead;
    if(Number.isFinite(t)){ playhead=t; updateScene(); updateCamera(); playhead=save; }
    rig.root.updateMatrixWorld(true);
    const out={};
    for(const nm of ['hips','spine1','spine2','chest','neck','head','shoulderL','shoulderR']){
      const j = rig.nodes.get(nm);
      if(!j) continue;
      const q=j.getWorldQuaternion(new THREE.Quaternion());
      const p=j.getWorldPosition(new THREE.Vector3());
      out[nm]={w:[q.x,q.y,q.z,q.w].map(v=>+v.toFixed(4)), p:[p.x,p.y,p.z].map(v=>+v.toFixed(4)),
                roll:+(Math.asin(clamp(2*(q.w*q.x+q.y*q.z),-1,1))*180/Math.PI).toFixed(1)};
    }
    const L=rig.nodes.get('shoulderL'), R=rig.nodes.get('shoulderR');
    if(L&&R){
      const a=L.getWorldPosition(new THREE.Vector3()), b=R.getWorldPosition(new THREE.Vector3());
      const d=a.clone().sub(b); const len=d.length();
      out.line={elev:+(Math.asin(clamp(d.y/len,-1,1))*180/Math.PI).toFixed(1), len:+len.toFixed(3)};
    }
    return out;
  },
  chestRoll(t){
    const save=playhead;
    if(Number.isFinite(t)){ playhead=t; updateScene(); updateCamera(); playhead=save; }
    return {roll:lastChestRoll*180/Math.PI, hip:lastHipRoll*180/Math.PI};
  },
  planted(t){
    const k = footPlanted('L', t) ? 1 : 0;
    return [k, footPlanted('R', t) ? 1 : 0];
  },
  async loadSpec(name, raw){        /* 调试/健壮性检查用：直接载入指定规格对象 */
    try{ await loadSpec(name, raw); return {ok:true}; }
    catch(e){ return {ok:false, error:String(e && e.message || e)}; }
  },
  bodyYawApplied(){
    if(!rig || !rig.yawNode) return 0;
    return rig.yawNode.rotation.y;
  },
  bodyYaw(t){
    if(!Number.isFinite(t)) return chanYaw(playhead);
    const save = playhead;
    setStatePlayhead(t);
    updateScene(); updateCamera(); playhead = save;
    return chanYaw(t);
  },
  bendJumps(){
    if(!rig || !rig.bendState) return 0;
    let n = 0;
    for(const [,st] of rig.bendState) n += st.jump;
    return n;
  },
  figurePose(t){
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); playhead = save; }
    const out = {};
    for(const n of ['hips','head','handL','handR','footL','footR']){
      const j = figNodes.get(n);
      if(j){ const p = new THREE.Vector3(); j.getWorldPosition(p); out[n] = [p.x,p.y,p.z]; }
    }
    return out;
  },
  limbAngles(t){
    if(!rig || !rig.nodes) return null;
    const save = playhead;
    if(Number.isFinite(t)){ playhead = t; updateScene(); updateCamera(); playhead = save; }
    const ang = (a,b,c)=>{
      const A=rig.nodes.get(a), B=rig.nodes.get(b), C=rig.nodes.get(c);
      if(!A||!B||!C) return null;
      const v1=A.getWorldPosition(new THREE.Vector3()).sub(B.getWorldPosition(new THREE.Vector3()));
      const v2=C.getWorldPosition(new THREE.Vector3()).sub(B.getWorldPosition(new THREE.Vector3()));
      return +(v1.angleTo(v2)*180/Math.PI).toFixed(1);
    };
    const line = (a,b)=>{                 /* 仰角：与偏航无关，可直接与视频的图像系比对 */
      const A=rig.nodes.get(a), B=rig.nodes.get(b);
      if(!A||!B) return null;
      const d=A.getWorldPosition(new THREE.Vector3()).sub(B.getWorldPosition(new THREE.Vector3()));
      const len=d.length();
      if(len<1e-5) return null;
      return +(Math.asin(clamp(d.y/len,-1,1))*180/Math.PI).toFixed(1);
    };
    return {elbowL:ang('shoulderL','elbowL','handL'), elbowR:ang('shoulderR','elbowR','handR'),
            kneeL:ang('hipL','kneeL','ankleL'), kneeR:ang('hipR','kneeR','ankleR'),
            shoulderLine:line('shoulderL','shoulderR'), hipLine:line('hipL','hipR')};
  },
  figureFacing(){
    if(!figGroup) return null;
    const v = new THREE.Vector3(0,0,1).applyQuaternion(figGroup.getWorldQuaternion(new THREE.Quaternion()));
    return [v.x, v.y, v.z];
  },
  sample(t){
    if(!spec) return [];
    return spec.subjects.map(s => {
      const p = samplePath(s.path, t);
      const mount = spec.partHeight[s.id];
      return {id:s.id, pos:[-p.x,p.d], world:[p.x, mount===null?0:mount, p.d], facing:p.facing};
    });
  },
  cameraState(){
    const d = new THREE.Vector3(0,0,-1).applyQuaternion(camera.quaternion);
    return {pos:{x:camera.position.x, y:camera.position.y, z:camera.position.z},
            dir:{x:d.x, y:d.y, z:d.z}, fov:camera.fov, aspect:camera.aspect, hfov: spec?spec.camera.fov_deg:50};
  },
  framing(){ return framingRatio(); },
  sceneStats(){
    let meshes=0, lines=0, lights=0, points=0;
    scene.traverse(o => { if(o.isMesh) meshes++; else if(o.isLine) lines++; if(o.isLight) lights++; if(o.isPoints) points++; });
    return {meshes, lines, lights, points, figNodes:[...figNodes.keys()], memory:{...renderer.info.memory}, render:{...renderer.info.render}};
  },
  videoState(){
    const v = $('#vid');
    return {src:v.currentSrc||v.src, readyState:v.readyState, duration:v.duration, currentTime:v.currentTime, paused:v.paused, w:v.videoWidth, h:v.videoHeight};
  },
  get scene(){ return scene; },
  get renderer(){ return renderer; },
  get camera(){ return camera; },
  get setGroup(){ return setGroup; },
  get figGroup(){ return figGroup; },
  setLoop(v){ loopRunning = !!v; },
  renderNow(){ dirty = false; updateScene(); updateCamera(); controls.update(); renderer.render(scene, camera); renderCount++; },
  snapshot(){ dirty = false; updateScene(); updateCamera(); controls.update(); renderer.render(scene, camera); renderCount++; return renderer.domElement.toDataURL('image/png'); }
};

boot().catch(err => showError('启动失败', String(err?.message||err), err?.stack||''));
