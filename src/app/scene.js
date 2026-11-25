import {$, DEG, clamp, pick, rand} from '../main.js';
import {BUDGET, QUALITY_LEVELS, THREE, crowdMesh, dirty, lightBeams, sceneRoot, setStateBaseFog, setStateCrowdMesh, setStateLightPool, setStatePhonePoints, setStatePoolOrigin, setStateQualityLevel, setStateSeatMesh, setStateSpotBeam, setStateSpotLight, setStateSpotTarget, spec, spotLight} from './state.js';

/* ---------- 竞技场生成器（全部由 set 推导） ---------- */
function radialTexture(){
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64,64,0,64,64,64);
  grd.addColorStop(0,'rgba(255,255,255,.72)');
  grd.addColorStop(.45,'rgba(210,225,255,.42)');
  grd.addColorStop(1,'rgba(160,190,255,0)');
  g.fillStyle = grd; g.fillRect(0,0,128,128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
/* 舞台 LED 地板：暗色面板 + 彩色发光格线（源视频的地面就是这种灯板阵） */
function ledFloorTexture(){
  const c = document.createElement('canvas');
  c.width = 512; c.height = 384;
  const g = c.getContext('2d');
  g.fillStyle = '#05070d'; g.fillRect(0,0,512,384);
  const hues = [0xff2fa0, 0x18d8e0, 0xf2eaff, 0x8a3cff, 0x2f6bff, 0xffd23f];
  const cols = 7, rows = 5, cw = 512/cols, ch = 384/rows;
  for(let i=0;i<cols;i++) for(let j=0;j<rows;j++){
    const col = hues[(Math.random()*hues.length)|0];
    const r=(col>>16)&255, gg=(col>>8)&255, b=col&255;
    const x=i*cw, y=j*ch;
    g.fillStyle=`rgba(${r},${gg},${b},.16)`;
    g.fillRect(x+3,y+3,cw-6,ch-6);
    g.strokeStyle=`rgba(${r},${gg},${b},.9)`;
    g.lineWidth=2.6;
    g.strokeRect(x+3.5,y+3.5,cw-7,ch-7);
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}
function buildStage(item){
  const [w,d,h] = item.size;
  const g = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.PlaneGeometry(w,d),
    new THREE.MeshBasicMaterial({color:0x05070d})
  );
  base.rotation.x = -Math.PI/2;
  base.position.y = h + .004;
  g.add(base);
  const led = new THREE.Mesh(
    new THREE.PlaneGeometry(w,d),
    new THREE.MeshBasicMaterial({map:ledFloorTexture()})
  );
  led.rotation.x = -Math.PI/2;
  led.position.y = h + .012;
  g.add(led);
  const pool = new THREE.Mesh(
    new THREE.PlaneGeometry(4,4),
    new THREE.MeshBasicMaterial({map:radialTexture(), transparent:true, opacity:.30, blending:THREE.AdditiveBlending, depthWrite:false})
  );
  pool.rotation.x = -Math.PI/2;
  pool.position.y = h + .03;
  pool.scale.setScalar(1.15);
  g.add(pool);
  setStateLightPool(pool);
  setStatePoolOrigin({x:item.x, z:item.d, halfW:w/2, halfD:d/2});
  return g;
}
function coprimeStride(cells){
  for(let s = 104729; s > 1; s--){
    let a = cells, b = s;
    while(b){ const t = a % b; a = b; b = t; }
    if(a === 1) return s;
  }
  return 1;
}
/* 人群背景墙：程序化生成密集观众剪影 + 手机灯点 —— 一张贴图就能把
   稀疏实例后面的空旷变成实心人海（源视频的背景就是这样一片暖色人潮） */
function crowdTexture(){
  const c = document.createElement('canvas');
  c.width = 1024; c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#12141f'; g.fillRect(0,0,1024,256);
  const rows = [[205,1.0],[160,.82],[118,.66]];
  for(const [baseY, sc] of rows){
    for(let i=0;i<1400;i++){
      const x = rand(0,1024), y = baseY + rand(-12,12)*sc;
      const r = (2.4*sc + rand(0,1.8)) * 1.6;
      const v = rand(.55,1.15);
      g.fillStyle = `rgb(${Math.round(255*v)},${Math.round(196*v*.88)},${Math.round(168*v*.78)})`;
      g.beginPath(); g.arc(x,y,r,0,6.283); g.fill();
      g.fillStyle = `rgba(20,16,26,.55)`;
      g.beginPath(); g.arc(x,y+r*1.5,r*1.15,0,6.283); g.fill();
    }
  }
  for(let i=0;i<260;i++){
    const x=rand(0,1024), y=rand(70,215), r=rand(.9,2.4);
    const grd=g.createRadialGradient(x,y,0,x,y,r*5);
    grd.addColorStop(0,'rgba(255,246,214,.95)'); grd.addColorStop(.35,'rgba(255,226,160,.35)');
    grd.addColorStop(1,'rgba(255,226,160,0)');
    g.fillStyle=grd; g.beginPath(); g.arc(x,y,r*5,0,6.283); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function buildAudience(item){  const [w,d] = item.size;
  const g = new THREE.Group();
  const spacing = .45;
  const cols = Math.max(1, Math.floor(w/spacing));
  const rows = Math.max(1, Math.floor(d/spacing));
  const cells = cols*rows;
  const n = Math.min(BUDGET.maxCrowd, cells);
  const stride = coprimeStride(cells);
  const geo = new THREE.CapsuleGeometry(.2, 1.3, 1, 5);
  const mat = new THREE.MeshLambertMaterial({color:0xffffff});
  const inst = new THREE.InstancedMesh(geo, mat, n);
  const dummy = new THREE.Object3D();
  const colors = [0x2b3342,0x39404f,0x43354a,0x27384a,0x473d3c];
  for(let i=0;i<n;i++){
    const cell = (i*stride) % cells;
    const col = cell % cols, row = Math.floor(cell/cols);
    dummy.position.set(
      -w/2 + spacing*(col+.5) + rand(-.18,.18),
      .85*rand(.92,1.08),
      -d/2 + spacing*(row+.5) + rand(-.18,.18)
    );
    dummy.rotation.set(0, rand(0,Math.PI), 0);
    dummy.scale.set(rand(.9,1.1), rand(.9,1.1), rand(.9,1.1));
    dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
    inst.setColorAt(i, new THREE.Color(pick(colors)));
  }
  inst.instanceMatrix.needsUpdate = true;
  if(inst.instanceColor) inst.instanceColor.needsUpdate = true;
  inst.userData.total = n;
  setStateCrowdMesh(inst);
  g.add(inst);
  const lights = Math.min(BUDGET.maxPhoneLights, Math.floor(n*.35));
  const pts = new Float32Array(lights*3);
  for(let i=0;i<lights;i++){
    const col = Math.floor(Math.random()*cols), row = Math.floor(Math.random()*rows);
    pts[i*3]   = -w/2 + spacing*(col+.5) + rand(-.3,.3);
    pts[i*3+1] = rand(1.4,2.1);
    pts[i*3+2] = -d/2 + spacing*(row+.5) + rand(-.3,.3);
  }
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute('position', new THREE.BufferAttribute(pts,3));
  const phone = new THREE.Points(pgeo, new THREE.PointsMaterial({color:0xfff2c8, size:.30, sizeAttenuation:true, transparent:true, opacity:1, blending:THREE.AdditiveBlending, depthWrite:false}));
  phone.userData.total = lights;
  setStatePhonePoints(phone);
  g.add(phone);
  const wallH = d*1.15;
  const wall = new THREE.Mesh(
    new THREE.PlaneGeometry(w, wallH),
    new THREE.MeshBasicMaterial({map:crowdTexture(), transparent:false, depthWrite:true})
  );
  wall.position.set(0, wallH/2, d/2 - .2);
  wall.renderOrder = -1;
  g.add(wall);
  return g;
}
function pushBox(positions, indices, cx, cy, cz, w, h, d){
  const x0=cx-w/2, x1=cx+w/2, y0=cy-h/2, y1=cy+h/2, z0=cz-d/2, z1=cz+d/2;
  const v = positions.length/3;
  positions.push(
    x0,y0,z0, x1,y0,z0, x1,y1,z0, x0,y1,z0,
    x0,y0,z1, x1,y0,z1, x1,y1,z1, x0,y1,z1);
  const f = [[0,1,2],[0,2,3],[5,4,7],[5,7,6],[4,0,3],[4,3,7],[1,5,6],[1,6,2],[3,2,6],[3,6,7],[4,5,1],[4,1,0]];
  for(const t of f) indices.push(v+t[2], v+t[1], v+t[0]);
}
function buildGrandstand(item){
  const [w,d,h] = item.size;
  const g = new THREE.Group();
  const tiers = Math.max(7, Math.round(h/2.6));
  const dz = d/tiers, dy = h/tiers;
  const seats = [];
  const positions = [], indices = [];
  for(let k=0;k<tiers;k++){
    const tw = w - k*4;
    pushBox(positions, indices, 0, dy*(k+.5), -d/2 + dz*(k+.5), tw, dy, dz*.96);
    const people = Math.floor(tw/2.2);
    for(let i=0;i<people;i++){
      seats.push([-tw/2 + 1.1 + i*2.2 + rand(-.5,.5), dy*k + .95, -d/2 + dz*(k+.5) + rand(-.3,.3)]);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions,3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  g.add(new THREE.Mesh(geo, new THREE.MeshLambertMaterial({color:0x2f3849})));
  const inst = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(.16, 1.0, 1, 4),
    new THREE.MeshBasicMaterial({color:0x0d1017}), seats.length);
  const dummy = new THREE.Object3D();
  seats.forEach((s,i) => {
    dummy.position.set(s[0], s[1], s[2]);
    dummy.updateMatrix();
    inst.setMatrixAt(i, dummy.matrix);
  });
  inst.instanceMatrix.needsUpdate = true;
  inst.userData.total = seats.length;
  setStateSeatMesh(inst);
  g.add(inst);
  return g;
}
function buildLamp(item){
  const [w,d,h] = item.size;
  const g = new THREE.Group();
  /* 桁架（深色金属），而不是一整块发光白板 */
  const truss = new THREE.Mesh(
    new THREE.BoxGeometry(w, h*.5, d*.5),
    new THREE.MeshStandardMaterial({color:0x2a3040, roughness:.7, metalness:.5})
  );
  truss.position.set(0, h/2, 0);
  g.add(truss);
  const rail = new THREE.Mesh(
    new THREE.BoxGeometry(w*1.02, h*.12, d*.12),
    new THREE.MeshStandardMaterial({color:0x3a4256, roughness:.6, metalness:.6})
  );
  rail.position.set(0, h*.28, 0);
  g.add(rail);
  /* 一排灯具：彩色灯罩 + 向下光束（实例化 → 2 个 draw call） */
  const colors = [0xff5ad0, 0x7fd8ff, 0xffe27a, 0x9b7bff, 0x6affc8];
  const n = 9;
  const lampGeo = new THREE.CylinderGeometry(h*.16, h*.22, h*.42, 10);
  const lamps = new THREE.InstancedMesh(
    lampGeo,
    new THREE.MeshStandardMaterial({color:0xffffff, roughness:.5, emissive:0xffffff, emissiveIntensity:.9}),
    n);
  const beamGeo = new THREE.ConeGeometry(2.6, item.base, 14, 1, true);
  const beams = new THREE.InstancedMesh(
    beamGeo,
    new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:.05, blending:THREE.AdditiveBlending, depthWrite:false, side:THREE.DoubleSide}),
    n);
  const dummy = new THREE.Object3D();
  for(let i=0;i<n;i++){
    const x = (i - (n-1)/2) * (w/n) * .92;
    const col = new THREE.Color(colors[i % colors.length]);
    dummy.position.set(x, h*.26, 0); dummy.rotation.set(0,0,0); dummy.scale.setScalar(1);
    dummy.updateMatrix(); lamps.setMatrixAt(i, dummy.matrix); lamps.setColorAt(i, col);
    dummy.position.set(x, -item.base/2 + h*.2, 0);
    dummy.updateMatrix(); beams.setMatrixAt(i, dummy.matrix); beams.setColorAt(i, col);
  }
  lamps.instanceMatrix.needsUpdate = true; beams.instanceMatrix.needsUpdate = true;
  if(lamps.instanceColor) lamps.instanceColor.needsUpdate = true;
  if(beams.instanceColor) beams.instanceColor.needsUpdate = true;
  g.add(lamps); g.add(beams);
  lightBeams.push(beams);
  return g;
}
export function greyBox(cls, size){
  const [w,d,h] = size;
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({color:0x6d7482, roughness:.85});
  const dark = new THREE.MeshStandardMaterial({color:0x4a505c, roughness:.9});
  const add = (gw,gh,gd,x,y,z,m=mat) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(gw,gh,gd), m);
    b.position.set(x,y,z); g.add(b); return b;
  };
  if(cls.includes('chair')){
    add(w*.92,h*.12,d*.9, 0, h*.5, 0);
    add(w*.9, h*.88, d*.1, 0, h*.62, -d*.42);
    add(w*.08, h*.5, d*.08, -w*.38, h*.25, -d*.34, dark);
    add(w*.08, h*.5, d*.08,  w*.38, h*.25, -d*.34, dark);
  }else if(cls.includes('table')){
    add(w, h*.1, d, 0, h*.95, 0);
    add(w*.08, h*.9, d*.08, -w*.4, h*.45, -d*.4, dark);
    add(w*.08, h*.9, d*.08,  w*.4, h*.45, -d*.4, dark);
  }else if(cls.includes('sofa')){
    add(w, h*.42, d, 0, h*.21, 0);
    add(w, h*.58, d*.18, 0, h*.71, -d*.4);
    add(w*.18, h*.36, d*.9, -w*.41, h*.6, 0);
    add(w*.18, h*.36, d*.9,  w*.41, h*.6, 0);
  }else if(cls.includes('car')){
    add(w, h*.42, d, 0, h*.28, 0);
    add(w*.55, h*.4, d*.5, 0, h*.66, 0);
  }else if(cls.includes('crate')){
    add(w, h, d, 0, h/2, 0);
  }else if(cls.includes('bench')){
    add(w, h*.12, d, 0, h*.94, 0);
    add(w*.1, h*.9, d*.14, -w*.42, h*.45, 0, dark);
    add(w*.1, h*.9, d*.14,  w*.42, h*.45, 0, dark);
  }else if(cls.includes('pillar')){
    add(w*.8, h, d*.8, 0, h/2, 0);
  }else{
    add(w, h, d, 0, h/2, 0);
  }
  return g;
}
export function buildSet(){
  const g = new THREE.Group();
  for(const item of spec.set){
    const c = item.cls.toLowerCase();
    let node;
    if(c.includes('stage')) node = buildStage(item);
    else if(c.includes('audience')) node = buildAudience(item);
    else if(c.includes('grandstand')) node = buildGrandstand(item);
    else if(c.includes('lamp')) node = buildLamp(item);
    else node = greyBox(c, item.size);
    node.position.set(item.x, item.base, item.d);
    node.rotation.y = -item.yaw*DEG;   /* 镜像系下 yaw 反号 */
    g.add(node);
  }
  return g;
}
export function buildEnvironment(){
  const g = new THREE.Group();
  const env = spec.environment;
  if(env.ground && env.ground !== 'none'){
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(600,600), new THREE.MeshLambertMaterial({color:0x0b0e14}));
    ground.rotation.x = -Math.PI/2;
    ground.position.y = -0.02;
    g.add(ground);
  }
  scene.background = new THREE.Color(env.sky ? 0x8fb6d8 : 0x070810);
  setStateBaseFog(env.sky ? null : new THREE.FogExp2(0x0d1020, .014));
  scene.fog = baseFog;
  return g;
}
export let ambLight = null, spotMain = null;
export function buildLights(){
  const g = new THREE.Group();
  ambLight = new THREE.AmbientLight(0x93a2c8, .5);
  g.add(ambLight);
  g.add(new THREE.HemisphereLight(0xa8bcf0, 0x1b2130, .85));
  const violet = new THREE.DirectionalLight(0xb98cff, 2.2);
  violet.position.set(-16, 16, 26);
  g.add(violet);
  /* 追舞者的暖聚光（第 4 盏，贴满 maxLights 预算） */
  setStateSpotTarget(new THREE.Object3D());
  sceneRoot.add(spotTarget);
  setStateSpotLight(new THREE.SpotLight(0xffd9a0, 220, 0, .55, .6, 1.6));
  spotMain = spotLight;
  spotLight.position.set(0, 9, -4);
  spotLight.target = spotTarget;
  g.add(spotLight);
  setStateSpotBeam(new THREE.Mesh(
    new THREE.ConeGeometry(1.5, 9.6, 24, 1, true),
    new THREE.MeshBasicMaterial({color:0xffe3b8, transparent:true, opacity:.05, blending:THREE.AdditiveBlending, depthWrite:false, side:THREE.DoubleSide})
  ));
  spotBeam.position.set(0, 4.8, -2);
  g.add(spotBeam);
  return g;
}

/* ---------- 画质档位：帧时驱动，带迟滞与预热豁免 ---------- */
export function setQuality(level){
  setStateQualityLevel(clamp(level|0, 0, QUALITY_LEVELS.length-1));
  const q = QUALITY_LEVELS[qualityLevel];
  if(crowdMesh){
    const total = crowdMesh.userData.total || 0;
    crowdMesh.count = Math.min(BUDGET.maxCrowd, Math.max(1, Math.floor(total * q.crowd / BUDGET.maxCrowd)));
  }
  if(phonePoints){
    const total = phonePoints.userData.total || 0;
    phonePoints.geometry.setDrawRange(0, Math.min(total, q.phone));
    phonePoints.visible = q.phone > 0;
  }
  if(seatMesh){
    const total = seatMesh.userData.total || 0;
    seatMesh.count = qualityLevel >= 2 ? Math.floor(total*.5) : total;
  }
  if(scene) scene.fog = q.fog ? baseFog : null;
  if(renderer) renderer.setPixelRatio(Math.min(devicePixelRatio, q.pixelRatio));
  markDirty();
}
export function adaptQuality(ms){
  const now = performance.now();
  if(firstRenderAt === 0){ firstRenderAt = now; }
  const elapsed = now - firstRenderAt;
  if(frameSamples < BUDGET.warmupFrames && elapsed < 1500){ frameSamples++; return; }
  frameSamples++;
  const over = BUDGET.maxFrameMs*1.5, under = BUDGET.maxFrameMs*.6;
  if(ms > over){ overBudgetMs += ms; underBudgetMs = 0; }
  else if(ms < under){ underBudgetMs += ms; overBudgetMs = 0; }
  else { overBudgetMs = 0; underBudgetMs = 0; }
  if(overBudgetMs > 2000 && qualityLevel < QUALITY_LEVELS.length-1){ setQuality(qualityLevel+1); overBudgetMs = 0; }
  else if(underBudgetMs > 3000 && qualityLevel > 0){ setQuality(qualityLevel-1); underBudgetMs = 0; }
}

/* ---------- 按需渲染：dirty 是唯一的重绘入口 ---------- */
export function markDirty(){ dirty = true; }
