import {$, CDN_BASES, FIG, samplePath} from '../main.js';
import {SLEEVE_R, findBoneByName, measurePalmAxes, measureProxyRadii, measureTorsoProfile, sm} from './solver.js';
import {THREE, figP, motion, sceneRoot, setStateFigGroup, spec} from './state.js';

/* ---------- 骨骼抽象：程序化人偶与 GLB 模型共用同一套 IK ---------- */
export const MODEL_URL = 'models/human.glb';
const BONE_MAP = {
  hips:'mixamorig:Hips', spine1:'mixamorig:Spine', spine2:'mixamorig:Spine1',
  chest:'mixamorig:Spine2', head:'mixamorig:Head',
  shoulderL:'mixamorig:LeftArm', elbowL:'mixamorig:LeftForeArm', handL:'mixamorig:LeftHand',
  hipL:'mixamorig:LeftUpLeg', kneeL:'mixamorig:LeftLeg', ankleL:'mixamorig:LeftFoot', footL:'mixamorig:LeftToeBase',
  shoulderR:'mixamorig:RightArm', elbowR:'mixamorig:RightForeArm', handR:'mixamorig:RightHand',
  hipR:'mixamorig:RightUpLeg', kneeR:'mixamorig:RightLeg', ankleR:'mixamorig:RightFoot', footR:'mixamorig:RightToeBase'
};
const CHAINS = [
  ['shoulderL','elbowL','handL'], ['shoulderR','elbowR','handR'],
  ['hipL','kneeL','ankleL'], ['hipR','kneeR','ankleR']
];
export let rig = null;                 /* {kind, nodes, meta, torso, scale, basis, initQuat} */
export let humanAsset = null;          /* {gltf, wrapper} 加载一次，跨规格复用 */
let recolorTexture = null;      /* 换装贴图同样只加载一次（重建人偶时不重复上传） */
let hairGroup = null;           /* 长发几何只建一次：模型 wrapper 跨规格复用，
                                   每次重建都新挂一份会在头骨上越堆越多 */

/* ---------- 人偶比例（程序化回退；GLB 模型走实测骨长） ---------- */
function figureParams(){
  const p = spec.primary;
  const H = p ? p.size[2] : 1.6;
  if(!motion || !motion.body){
    return {
      H, k:H, thigh:H*FIG.thigh, shin:H*FIG.shin, upperArm:H*FIG.upperArm, foreArm:H*FIG.foreArm,
      shoulderX:null, hipX:null,
      hipY:H*FIG.hipY, chestLocal:H*(FIG.chestY-FIG.hipY), headLocal:H*(FIG.headY-FIG.chestY),
      headR:H*FIG.headR, footH:H*FIG.footH, torsoTop:H*(FIG.sweaterTop-FIG.hipY), torsoBottom:H*(FIG.sweaterBottom-FIG.hipY)
    };
  }
  const b = motion.body;
  const thigh = (b.legThighL+b.legThighR)/2, shin = (b.legShinL+b.legShinR)/2;
  const shoulderY = (b.shoulderL[1]+b.shoulderR[1])/2;
  const headU = 0.62;
  const totalU = thigh + shin + shoulderY + headU;
  const k = H/Math.max(.1, totalU);
  const footH = k*0.10;
  const hipY = k*(thigh+shin) + footH;
  return {
    H, k, thigh:k*thigh, shin:k*shin,
    upperArm:k*(b.armUpperL+b.armUpperR)/2, foreArm:k*(b.armForeL+b.armForeR)/2,
    shoulderX:k*Math.abs(b.shoulderL[0]), hipX:k*Math.abs(b.hipL[0]),
    hipY, chestLocal:k*shoulderY, headLocal:k*headU*.45,
    headR:k*.24, footH,
    torsoTop:k*(shoulderY+0.24), torsoBottom:k*-0.06
  };
}

function partOffset(kind, side){
  const parts = spec.parts.filter(s => spec.partKind[s.id] === kind);
  if(!parts.length) return 0;
  const prim = samplePath(spec.primary.path, spec.shot.start);
  const label = parts.find(s => {
    const t = `${s.cls} ${s.desc}`;
    return side > 0 ? /left|左/.test(t) : /right|右/.test(t);
  });
  const target = label || parts.find(s => {
    const off = samplePath(s.path, spec.shot.start).x - prim.x;
    return side > 0 ? off > 0 : off < 0;
  }) || parts[0];
  return Math.abs(samplePath(target.path, spec.shot.start).x - prim.x);
}

/* 合并几何体（省 draw call）：把各自的顶点按矩阵变换后拼成一个 BufferGeometry。
   three 的 BufferGeometryUtils 需要额外 import，这里手写最小版本：
   只处理 position/normal/uv + 索引，够用于本项目的发片。 */
function mergeGeometries(list){
  const pos=[], nor=[], uv=[], idx=[];
  const v=new THREE.Vector3(), w=new THREE.Vector3();
  for(const g of list){
    const p=g.attributes.position, n=g.attributes.normal, u=g.attributes.uv;
    for(let i=0;i<p.count;i++){
      v.fromBufferAttribute(p,i); pos.push(v.x,v.y,v.z);
      w.fromBufferAttribute(n,i); nor.push(w.x,w.y,w.z);
      uv.push(u.getX(i), u.getY(i));
    }
    const base=pos.length/3 - p.count;
    const ix=g.index ? g.index.array : null;
    if(ix){ for(let i=0;i<ix.length;i++) idx.push(ix[i]+base); }
    else { for(let i=0;i<p.count;i++) idx.push(i+base); }
  }
  const out=new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  out.setAttribute('normal', new THREE.Float32BufferAttribute(nor,3));
  out.setAttribute('uv', new THREE.Float32BufferAttribute(uv,2));
  out.setIndex(idx);
  return out;
}
/* 摆放一个几何体：先烘进矩阵再合并（合并后不能再单独变换） */
function placed(geo, px,py,pz, sx,sy,sz, rz=0){
  const g=geo.clone();
  const m=new THREE.Matrix4().compose(
    new THREE.Vector3(px,py,pz),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1), rz),
    new THREE.Vector3(sx,sy,sz));
  g.applyMatrix4(m);
  return g;
}
/* 长发：Mixamo 骨骼没有发骨，用挂在头骨上的发片补出剪影
   （源视频表演者是及腰长发 + 刘海，模型是双发髻，剪影差别很大）。
   注意：骨骼局部单位是 Mixamo 导出单位（100 单位 = 1 米），
   头骨局部系为 +X=自身左 / +Y=上 / +Z=面朝。
   发髻会被头发主体吞掉；两侧与后方的发帘挂在 hairSwing 组上，跟随头部摆动。
   发片合并成 2 个 mesh（静态部分 + 摆动部分），避免拉高 draw call。 */
function buildLongHair(){
  if(hairGroup) return hairGroup;      /* 模型 wrapper 跨规格复用，头发只建一次 */
  const mat = new THREE.MeshStandardMaterial({color:0x1a1410, roughness:.9, side:THREE.DoubleSide});
  const capMat = new THREE.MeshStandardMaterial({color:0x2b2b33, roughness:.95});
  const hoodMat = new THREE.MeshStandardMaterial({color:0x1b2540, roughness:.95});
  const g = new THREE.Group(); g.name = 'hair';
  /* 头骨局部系：+X=自身左 / +Y=上 / +Z=面朝；单位=Mixamo 导出单位（1 单位 = 1cm）。
     源视频人物：毛线帽 + 及腰黑直发（两缕垂在肩前）+ 颈后一小团帽兜。
     模型头（含发壳）实测：颅顶在头骨上方 22.5cm、头宽 ±8.2cm、脸在 +Z、脑后 −6cm。
     发缕用**旋转体（Lathe）**：闭合、末端收尖、贴头骨 ——
     开放圆筒（openEnded）背面被剔除，从正面看是一张能看穿的平板（"面纱"观感）。 */
  const strand = (profile, seg=20) => new THREE.LatheGeometry(
    profile.map(p => new THREE.Vector2(p[0], p[1])), seg);
  const back = placed(strand([[8.2,3],[8.9,-4],[8.5,-14],[7.5,-26],[5.9,-35],[3.6,-41],[1.0,-43]]),
                      0, -19, -4.5, 1.28, 1, .52);
  const side = strand([[4.4,1],[4.9,-7],[4.5,-19],[3.5,-29],[2.1,-37],[0.8,-40]]);
  const sideL = placed(side, 7.2, -18, .6, .58, 1, .82);
  const sideR = placed(side, -7.2, -18, .6, .58, 1, .82);
  const g2 = new THREE.Group(); g2.name = 'hairSwing';
  g2.add(new THREE.Mesh(mergeGeometries([back, sideL, sideR]), mat));
  g.add(g2);
  g.add(new THREE.Mesh(mergeGeometries([placed(new THREE.SphereGeometry(1, 24, 14, 0, Math.PI*2, 0, 1.95),
                        0, 12.5, -1, 9.8, 10.5, 10.2)]), capMat));   /* 毛线帽刚性跟随头骨 */
  /* 帽衫兜帽：源视频的连帽衫在颈后堆着一团帽子（藏青色，与发色区分开） */
  const hood = new THREE.Mesh(new THREE.SphereGeometry(7, 16, 12), hoodMat);
  hood.scale.set(1.1, .8, .85);
  hood.position.set(0, -5.5, -7);
  g.add(hood);
  g.traverse(o => { o.frustumCulled = false; });
  hairGroup = g;
  return g;
}
/* 可见剪影的世界包围盒：按几何体的**真实顶点**算，不用 Box3.setFromObject。
   SkinnedMesh 的包围盒是按蒙皮后的静止姿势算的，Mixamo 的骨链静止姿势与几何体
   不一致，包围盒比实际身高高 3% —— 按它归一化，人偶比规格声明的身高矮一截，
   「对照机位」里与视频表演者对不上（E13 实测渲染占比偏差 14%）。 */
export function geometryBox(obj){
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  obj.updateMatrixWorld(true);
  obj.traverse(o=>{
    if(!o.isMesh || !o.geometry || !o.geometry.attributes || !o.geometry.attributes.position) return;
    const g = o.geometry;
    let b;
    if(o.isSkinnedMesh){
      /* 蒙皮网格必须按**蒙皮后**的顶点算包围盒。
         geometry.boundingBox 是未蒙皮的原始位置，而 Mixamo 导出把 mesh 挂在
         scale=0.01 的 Character 节点下 —— 直接乘节点矩阵会把真实范围算小 100 倍，
         「可见身高」会被头发（挂在头骨上的局部几何）主导，脚底也定位不准。 */
      o.computeBoundingBox();
      b = o.boundingBox;
    }else{
      if(!g.boundingBox) g.computeBoundingBox();   /* 按位置属性重算，不用 accessor 的陈旧 min/max */
      b = g.boundingBox;
    }
    for(let i=0;i<8;i++){
      v.set(i&1?b.max.x:b.min.x, i&2?b.max.y:b.min.y, i&4?b.max.z:b.min.z).applyMatrix4(o.matrixWorld);
      box.expandByPoint(v);
    }
  });
  return box;
}
/* 记录每根骨相对其子的静止偏移与长度 —— IK 与具体模型解耦的依据 */
function captureRig(nodes, root){
  root.updateMatrixWorld(true);
  const meta = new Map(), initQuat = new Map();
  for(const chain of CHAINS){
    const [u,l,e] = chain;
    const up = nodes.get(u), lo = nodes.get(l), en = nodes.get(e);
    if(!up || !lo || !en) continue;
    const upW = up.getWorldPosition(new THREE.Vector3());
    const loW = lo.getWorldPosition(new THREE.Vector3());
    const enW = en.getWorldPosition(new THREE.Vector3());
    meta.set(u, {worldLen:upW.distanceTo(loW)});
    meta.set(l, {worldLen:loW.distanceTo(enW)});
  }
  for(const [k,o] of nodes) initQuat.set(k, o.quaternion.clone());
  /* 身体基：左轴=左手−右手，上轴=头−髋，前轴=左×上 */
  const lh = nodes.get('handL'), rh = nodes.get('handR'), hd = nodes.get('head'), hp = nodes.get('hips');
  let left = new THREE.Vector3(1,0,0), up = new THREE.Vector3(0,1,0);
  if(lh && rh) left = lh.getWorldPosition(new THREE.Vector3()).sub(rh.getWorldPosition(new THREE.Vector3()));
  if(hd && hp) up = hd.getWorldPosition(new THREE.Vector3()).sub(hp.getWorldPosition(new THREE.Vector3()));
  if(left.lengthSq() < 1e-8) left.set(1,0,0);
  if(up.lengthSq() < 1e-8) up.set(0,1,0);
  left.normalize(); up.normalize();
  const fwd = left.clone().cross(up).normalize();
  const inv = root.getWorldQuaternion(new THREE.Quaternion()).invert();
  const basis = {left:left.clone().applyQuaternion(inv), up:up.clone().applyQuaternion(inv), fwd:fwd.clone().applyQuaternion(inv)};
  let torso = .4, shMid = null;
  if(nodes.get('shoulderL') && nodes.get('shoulderR') && hp){
    shMid = nodes.get('shoulderL').getWorldPosition(new THREE.Vector3())
      .add(nodes.get('shoulderR').getWorldPosition(new THREE.Vector3())).multiplyScalar(.5);
    torso = shMid.distanceTo(hp.getWorldPosition(new THREE.Vector3()));
  }
  /* 髋关节中点（Mixamo 的 LeftUpLeg 起点 = 股骨头；程序化人偶的 hipL 组）：
     源视频的髋关键点也在这个位置 —— 「对照机位」的相机基准与手臂的肩高补偿都要用它。
     Mixamo 的骨盆根节点（mixamorig:Hips）比股骨头高 8.6cm，两者不能混用：
     拿骨盆根当髋关节会让整条腿短 8.6cm（髋高解算基准错位 → 人偶永久下蹲）。 */
  const hipJL = nodes.get('hipL'), hipJR = nodes.get('hipR');
  let hipMid = null;
  if(hipJL && hipJR) hipMid = hipJL.getWorldPosition(new THREE.Vector3())
    .add(hipJR.getWorldPosition(new THREE.Vector3())).multiplyScalar(.5);
  const torsoHip = (shMid && hipMid) ? shMid.distanceTo(hipMid) : null;
  /* 踝骨静止高度（脚踩地时踝关节离地多少）—— 髋高几何解算要用 */
  let ankleY = 0, na = 0;
  for(const nm of ['ankleL','ankleR']){
    const o = nodes.get(nm);
    if(o){ ankleY += o.getWorldPosition(new THREE.Vector3()).y; na++; }
  }
  return {meta, initQuat, basis, torso:Math.max(.05, torso), ankleY: na ? ankleY/na : 0,
          hipMid, torsoHip};
}

/* ---------- 程序化人偶（模型缺失时的回退） ---------- */
function buildProceduralFigure(){
  const p = spec.primary;
  if(!p) return null;
  const H = p.size[2];
  figP = figureParams();
  const P = figP;
  const root = new THREE.Group();
  root.name = 'root';
  const yawNode = new THREE.Group();
  yawNode.name = 'bodyYaw';
  const hips = new THREE.Group();
  hips.name = 'hips';
  hips.position.y = P.hipY;
  yawNode.add(hips);
  root.add(yawNode);
  const torso = new THREE.Group();
  torso.name = 'torso';
  hips.add(torso);
  const swH = Math.max(.02, P.torsoTop - P.torsoBottom);
  const sw = new THREE.Mesh(new THREE.BoxGeometry(p.size[0], swH, p.size[1]), new THREE.MeshStandardMaterial({color:FIG.sweater, roughness:.95}));
  sw.position.y = (P.torsoTop + P.torsoBottom)/2;
  torso.add(sw);
  const chest = new THREE.Group();
  chest.name = 'chest';
  chest.position.y = P.chestLocal;
  torso.add(chest);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(H*.022, H*.03, H*.05, 12), new THREE.MeshStandardMaterial({color:FIG.skin, roughness:.8}));
  neck.position.y = H*.02;
  chest.add(neck);
  const head = new THREE.Mesh(new THREE.SphereGeometry(P.headR, 24, 18), new THREE.MeshStandardMaterial({color:FIG.skin, roughness:.75}));
  head.name = 'head';
  head.position.y = P.headLocal;
  chest.add(head);
  const hairH = Math.max(.02, P.headLocal + P.headR);
  const hair = new THREE.Mesh(new THREE.BoxGeometry(P.headR*1.9, hairH, P.headR*1.6), new THREE.MeshStandardMaterial({color:FIG.hair, roughness:.9}));
  hair.position.set(0, P.headLocal + P.headR*.35, -P.headR*.55);
  chest.add(hair);

  const shoulderX = (motion && motion.body) ? P.shoulderX : (partOffset('arm', 1) || p.size[0]*.36);
  const hipX = (motion && motion.body) ? P.hipX : (partOffset('leg', 1) || p.size[0]*.22);
  const skinM = new THREE.MeshStandardMaterial({color:FIG.skin, roughness:.75});
  const jeansM = new THREE.MeshStandardMaterial({color:FIG.jeans, roughness:.85});
  const shoeM = new THREE.MeshStandardMaterial({color:0xe8e8ea, roughness:.7});

  for(const side of [1,-1]){
    const tag = side > 0 ? 'L' : 'R';
    const shoulder = new THREE.Group();
    shoulder.name = 'shoulder'+tag;
    shoulder.position.set(side*shoulderX, 0, 0);
    chest.add(shoulder);
    const sleeveR = Math.max(.012, H*FIG.sleeveR);
    const upper = new THREE.Mesh(new THREE.CylinderGeometry(sleeveR, sleeveR*.82, P.upperArm, 14), new THREE.MeshStandardMaterial({color:FIG.sweater, roughness:.95}));
    upper.position.y = -P.upperArm/2;
    shoulder.add(upper);

    const elbow = new THREE.Group();
    elbow.name = 'elbow'+tag;
    elbow.position.y = -P.upperArm;
    shoulder.add(elbow);
    const fore = new THREE.Mesh(new THREE.CylinderGeometry(sleeveR*.78, sleeveR*.5, P.foreArm, 12), new THREE.MeshStandardMaterial({color:FIG.sweater, roughness:.95}));
    fore.position.y = -P.foreArm/2;
    elbow.add(fore);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(H*.028, 12, 10), skinM);
    hand.name = 'hand'+tag;
    hand.position.y = -P.foreArm;
    elbow.add(hand);

    const hip = new THREE.Group();
    hip.name = 'hip'+tag;
    hip.position.set(side*hipX, 0, 0);
    hips.add(hip);
    const legR = Math.max(.01, H*FIG.legR);
    const thigh = new THREE.Mesh(new THREE.CylinderGeometry(legR, legR*.85, P.thigh, 12), jeansM);
    thigh.position.y = -P.thigh/2;
    hip.add(thigh);
    const knee = new THREE.Group();
    knee.name = 'knee'+tag;
    knee.position.y = -P.thigh;
    hip.add(knee);
    const shin = new THREE.Mesh(new THREE.CylinderGeometry(legR*.85, legR*.7, P.shin, 12), jeansM);
    shin.position.y = -P.shin/2;
    knee.add(shin);
    const ankle = new THREE.Group();
    ankle.name = 'ankle'+tag;
    ankle.position.y = -P.shin;
    knee.add(ankle);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(H*.05, P.footH, H*.12), shoeM);
    foot.name = 'foot'+tag;
    foot.position.set(0, -P.footH/2, H*.022);
    ankle.add(foot);
  }
  sceneRoot.add(root);
  setStateFigGroup(root);
  const nodes = new Map();
  root.traverse(o => { if(o.name){ nodes.set(o.name, o); figNodes.set(o.name, o); } });
  const cap = captureRig(nodes, root);
  const pThigh = cap.meta.get('hipL')?.worldLen || .3, pShin = cap.meta.get('kneeL')?.worldLen || .3;
  const pHipJointY = cap.hipMid ? cap.hipMid.y : P.hipY;
  rig = {kind:'proc', nodes, meta:cap.meta, initQuat:cap.initQuat, basis:cap.basis,
         torso:cap.torso, scale:P.k, root, yawNode, baseQuat:root.quaternion.clone(),
         hipsRest:hips.position.clone(), restHipY:P.hipY,
         hipJointY:pHipJointY, torsoHip:cap.torsoHip,
         vertReach:Math.max(.05, pHipJointY - P.footH), ankleGroundY:P.footH,
         proxyR:{pelvis:p.size[0]*.30, torso:p.size[0]*.30, head:P.headR,
                 thighL:H*FIG.legR, thighR:H*FIG.legR, shinL:H*FIG.legR*.8, shinR:H*FIG.legR*.8,
                 handL:H*.028, handR:H*.028,
                 upperArmL:H*FIG.sleeveR, upperArmR:H*FIG.sleeveR,
                 foreArmL:H*FIG.sleeveR*.78, foreArmR:H*FIG.sleeveR*.78}};
  return root;
}

/* ---------- GLB 人体模型（Mixamo 骨骼） ---------- */
export async function loadHumanModel(){
  if(location.protocol === 'file:') return null;      /* file:// 下 fetch 被 CORS 拦截 */
  for(const base of CDN_BASES){
    try{
      let mod;
      try{ mod = await import(base.addons+'loaders/GLTFLoader.js'); }
      catch(e){
        const res = await fetch(base.addons+'loaders/GLTFLoader.js');
        if(!res.ok) throw new Error('GLTFLoader HTTP '+res.status);
        const src = await res.text();
        const url = URL.createObjectURL(new Blob([src.replace(/(['"])three\1/g, `'${base.three}'`)],{type:'text/javascript'}));
        try{ mod = await import(url); } finally{ URL.revokeObjectURL(url); }
      }
      const loader = new mod.GLTFLoader();
      return await loader.loadAsync(MODEL_URL);
    }catch(e){ console.warn('人体模型加载失败:', e?.message); }
  }
  return null;
}
export function buildHumanFigure(){
  const p = spec.primary;
  if(!p || !humanAsset) return null;
  const H = p.size[2];
  const wrapper = humanAsset.wrapper;
  const outer = new THREE.Group();
  outer.name = 'root';
  const yawNode = new THREE.Group();          /* 身体绕竖轴的转身（与 schema 朝向分层） */
  yawNode.name = 'bodyYaw';
  outer.add(yawNode);
  yawNode.add(wrapper);
  sceneRoot.add(outer);
  wrapper.scale.setScalar(1);
  wrapper.position.set(0,0,0);
  wrapper.quaternion.identity();
  /* 还原绑定姿势：蒙皮包围盒的定标/居中必须量出厂姿势，量上一规格残留的姿势
     会让整具人偶平移 20cm 级、比例漂 2%（蒙皮顶点随姿势动） */
  if(humanAsset.bindPose){
    for(const bp of humanAsset.bindPose){ bp.bone.quaternion.copy(bp.q); bp.bone.position.copy(bp.p); bp.bone.scale.copy(bp.s); }
  }
  wrapper.updateMatrixWorld(true);

  /* 骨名归一化索引（three.js 会剥掉分隔符） */
  const norm = t => t.toLowerCase().replace(/[^a-z0-9]/g, '');
  const rawBones = new Map();
  wrapper.traverse(o => { if(o.isBone) rawBones.set(norm(o.name), o); });

  /* 用骨架基把模型摆正：世界 +X=自身左, +Y=上, +Z=前（不依赖模型内部坐标约定） */
  const hipsB = rawBones.get('mixamorighips'), headB = rawBones.get('mixamorighead');
  const lhB = rawBones.get('mixamoriglefthand'), rhB = rawBones.get('mixamorigrighthand');
  if(hipsB && headB && lhB && rhB){
    const up = headB.getWorldPosition(new THREE.Vector3()).sub(hipsB.getWorldPosition(new THREE.Vector3())).normalize();
    const left = lhB.getWorldPosition(new THREE.Vector3()).sub(rhB.getWorldPosition(new THREE.Vector3())).normalize();
    const fwd = left.clone().cross(up).normalize();
    /* R 需满足 R·up=+Y, R·left=+X, R·fwd=+Z ⇒ R = 基矩阵的转置（逆） */
    wrapper.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(left, up, fwd).transpose());
    wrapper.updateMatrixWorld(true);
  }

  const bones = new Map();
  wrapper.traverse(o=>{ if(o.isBone) bones.set(o.name, o); });
  const boneIndex = new Map();
  for(const [n,b] of bones) boneIndex.set(norm(n), b);
  const findBone = name => boneIndex.get(norm(name));
  const nodes = new Map();
  for(const [logical, boneName] of Object.entries(BONE_MAP)){
    const b = findBone(boneName);
    if(b) nodes.set(logical, b);
  }
  nodes.set('root', outer);
  if(!nodes.get('hips') || !nodes.get('head') || !nodes.get('handL')){
    console.warn('骨骼映射失败，可用骨名:', [...bones.keys()].slice(0, 24).join(','));
    sceneRoot.remove(outer);
    return null;
  }
  for(const [k,o] of nodes) figNodes.set(k, o);
  /* 长发 + 帽衫兜帽挂在头骨上（Mixamo 骨骼没有发骨，剪影只能外挂几何）。
     必须在校身高之前挂上：身高按**可见剪影**（发顶→鞋底）配平，
     漏掉头发就会把人偶量矮一截（与视频里的表演者对不上）。 */
  const headBone = nodes.get('head');
  if(headBone) headBone.add(buildLongHair());
  wrapper.updateMatrixWorld(true);

  const box = geometryBox(wrapper);
  const k = H / Math.max(.01, box.max.y - box.min.y);
  wrapper.scale.setScalar(k);
  wrapper.updateMatrixWorld(true);
  const box2 = geometryBox(wrapper);
  wrapper.position.set(-(box2.min.x+box2.max.x)/2, -box2.min.y, -(box2.min.z+box2.max.z)/2);
  outer.updateMatrixWorld(true);

  /* 换装：若存在重着色贴图则覆盖（与源视频的服装配色一致：黄毛衣→深藏青上衣）。
     贴图只加载一次：换规格会重建人偶，每次重建都新建贴图会漏显存。 */
  if(!recolorTexture){
    recolorTexture = new THREE.TextureLoader().load('models/human_recolor.png', t => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.flipY = false;                 /* GLTF 贴图约定：UV 原点在左上 */
      t.needsUpdate = true;
    }, undefined, () => console.warn('重着色贴图缺失，使用原始贴图'));
    recolorTexture.colorSpace = THREE.SRGBColorSpace;
    recolorTexture.flipY = false;
  }
  const recolorTex = recolorTexture;
  wrapper.traverse(o => {
    if(!o.isMesh || !o.material) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for(const mt of mats){
      if(mt.map){ mt.map = recolorTex; mt.needsUpdate = true; }
    }
  });

  /* 收窄裤腿：模型本体是气球裤（大腿最宽处接近腰宽），源视频是阔腿直筒牛仔裤。
     腿骨的局部 Y 是骨长方向，只缩放局部 X/Z 即只改截面粗细、不改骨长；
     腿骨的祖先缩放都是均匀的 → 世界矩阵无剪切，IK 精度不受影响。 */
  for(const [bone, sc] of [['mixamorig:LeftUpLeg', [0.86,1,0.92]], ['mixamorig:RightUpLeg', [0.86,1,0.92]],
                           ['mixamorig:LeftLeg', [0.90,1,0.94]], ['mixamorig:RightLeg', [0.90,1,0.94]]]){
    const b = findBone(bone);
    if(b) b.scale.set(sc[0], sc[1], sc[2]);
  }
  /* 宽松袖筒：源视频的连帽衫袖子肥大，模型本体是细胳膊 —— 用一圈同色几何
     加粗上臂与前臂（Mixamo 骨链的局部 +Y 指向子骨，骨长 = 子骨平移量）。
     袖筒半径大于模型臂径 → 完全包住，不会与手臂网格打架。 */
  const SLEEVE = {hud:0x171a36};
  for(const [bone, rad, len] of [['mixamorig:LeftArm', 6.2, 24.2], ['mixamorig:RightArm', 6.2, 24.2],
                                 ['mixamorig:LeftForeArm', 5.2, 21.6], ['mixamorig:RightForeArm', 5.2, 21.6]]){
    const b = findBone(bone);
    if(!b || b.getObjectByName('sleeve')) continue;
    const sl = new THREE.Mesh(new THREE.CapsuleGeometry(rad, Math.max(1, len - 2*rad), 4, 12),
      new THREE.MeshStandardMaterial({color:SLEEVE.hud, roughness:.95}));
    sl.name = 'sleeve';
    sl.position.set(0, len/2, 0);
    sl.frustumCulled = false;
    b.add(sl);
  }
  /* 白 T 恤下摆：源视频的连帽衫下缘露出一圈白边。
     贴图里腰腹是一整块（藏青），这条白边用一圈几何补出。
     髋骨局部 +Y 朝上（Mixamo 髋骨自带 −90° 绕 X 的旋转），所以往下要取负号；
     位置按源视频实测：白边在身高 56% 处（髋关节上方一点、连帽衫下缘）。 */
  const hipsBone = nodes.get('hips');
  if(hipsBone && !hipsBone.getObjectByName('hemBand')){
    const band = new THREE.Mesh(new THREE.CylinderGeometry(10.2, 10.7, 3.5, 20),
      new THREE.MeshStandardMaterial({color:0xe6e4e0, roughness:.9}));
    band.name = 'hemBand';
    band.position.set(0, -7.9, 0);
    band.frustumCulled = false;
    hipsBone.add(band);
  }

  setStateFigGroup(outer);
  const cap = captureRig(nodes, outer);
  /* 碰撞代理半径：蒙皮网格实测（一次遍历顶点，按主骨骼分组取垂距 95 分位）。
     袖筒是外挂几何（不在蒙皮网格里），半径取建袖筒时的已知值 × 模型缩放。 */
  const sm = wrapper.getObjectByProperty('isSkinnedMesh', true);
  let proxyR = null;
  if(sm && sm.skeleton && sm.geometry.attributes.skinIndex){
    const wpOf = name => {
      const b = findBoneByName(wrapper, name);
      return b ? b.getWorldPosition(new THREE.Vector3()) : null;
    };
    const axisAt = (posName, fromName, toName) => {
      const A = wpOf(posName);
      if(!A) return null;
      const F = wpOf(fromName), T = wpOf(toName);
      let d = (F && T) ? T.clone().sub(F) : new THREE.Vector3(0,1,0);
      if(d.lengthSq() < 1e-9) d = new THREE.Vector3(0,1,0);
      return {a:A, d:d.normalize()};
    };
    const axes = {
      pelvis: axisAt('mixamorig:Hips','mixamorig:Hips','mixamorig:Spine'),
      torso:  axisAt('mixamorig:Spine2','mixamorig:Spine','mixamorig:Spine2'),
      head:   axisAt('mixamorig:Head', null, null),
      thighL: axisAt('mixamorig:LeftUpLeg','mixamorig:LeftUpLeg','mixamorig:LeftLeg'),
      thighR: axisAt('mixamorig:RightUpLeg','mixamorig:RightUpLeg','mixamorig:RightLeg'),
      shinL:  axisAt('mixamorig:LeftLeg','mixamorig:LeftLeg','mixamorig:LeftFoot'),
      shinR:  axisAt('mixamorig:RightLeg','mixamorig:RightLeg','mixamorig:RightFoot'),
      handL:  axisAt('mixamorig:LeftHand','mixamorig:LeftForeArm','mixamorig:LeftHand'),
      handR:  axisAt('mixamorig:RightHand','mixamorig:RightForeArm','mixamorig:RightHand')
    };
    const measured = measureProxyRadii(sm, sm.skeleton, axes);
    proxyR = {};
    for(const k in measured) proxyR[k] = measured[k].p95;
    for(const k of ['handL','handR']) if(measured[k]) proxyR[k] = measured[k].max;
    /* 袖筒是外挂几何，不在蒙皮网格里 → 半径用建筒时的已知值换算到米。
       换算因子 = **骨的世界缩放**（模型的 Character 节点带 0.01 缩放：骨长单位 1cm=0.01m，
       再乘 wrapper 的缩放）。直接用 wrapper 的缩放会差 100 倍（半径变成 6.7 米）。 */
    const boneScale = (name) => {
      const b = findBoneByName(wrapper, name);
      return b ? b.getWorldScale(new THREE.Vector3()).x : 0;
    };
    const sUp = boneScale('mixamorig:LeftArm'), sFo = boneScale('mixamorig:LeftForeArm');
    proxyR['upperArmL'] = proxyR['upperArmR'] = SLEEVE_R.upper * sUp;
    proxyR['foreArmL']  = proxyR['foreArmR']  = SLEEVE_R.fore * sFo;
    /* 代理半径的合理性闸门：人体任何部位的碰撞半径都 < 0.5m，
       超过说明单位/空间搞错了 —— 宁可放弃代理也不能让坏半径把手推出几米。 */
    for(const k in proxyR){
      if(!(proxyR[k] > 0.005 && proxyR[k] < .5)){ console.warn('代理半径越界，已丢弃:', k, proxyR[k]); delete proxyR[k]; }
    }
  }
  /* 中指末节骨（手的实际指向）：手的代理胶囊要沿**手指**方向，不是前臂方向 */
  const fingerTip = {};
  for(const [tag, bone] of [['L','mixamorig:LeftHandMiddle3'],['R','mixamorig:RightHandMiddle3']]){
    const b = findBoneByName(wrapper, bone);
    if(b) fingerTip[tag] = b;
  }
  /* 指根（掌骨起点）：手部可见的第二个突出点 —— 指根比指尖更容易插进躯干/大腿，
     穿模判据只测腕点与指尖会漏掉「指根在里面、指尖在外面」。 */
  const fingerRoot = {L:[], R:[]}, fingerMid = {L:[], R:[]}, fingerTips = {L:[], R:[]};
  for(const [tag, side] of [['L','Left'],['R','Right']]){
    for(const f of ['Index','Middle','Ring','Pinky','Thumb']){
      const b1 = findBoneByName(wrapper, 'mixamorig:'+side+'Hand'+f+'1');
      if(b1) fingerRoot[tag].push(b1);
      const b2 = findBoneByName(wrapper, 'mixamorig:'+side+'Hand'+f+'2');
      if(b2) fingerMid[tag].push(b2);
      const b3 = findBoneByName(wrapper, 'mixamorig:'+side+'Hand'+f+'3');
      if(b3) fingerTips[tag].push(b3);
    }
  }
  /* 掌面法线（手骨局部系）：从模型动画里的手指屈曲实测 */
  const palmAxis = measurePalmAxes(humanAsset.gltf, wrapper);
  wrapper.updateMatrixWorld(true);
  /* 躯干半径剖面：沿「髋关节 → 胸腔」轴分箱实测 */
  let torsoProfile = null;
  if(sm && sm.skeleton && cap.hipMid){
    const chestW = nodes.get('chest') ? nodes.get('chest').getWorldPosition(new THREE.Vector3()) : null;
    if(chestW){
      const axis = chestW.clone().sub(cap.hipMid);
      const h = axis.length();
      if(h > 1e-3){
        axis.divideScalar(h);
        const lat = new THREE.Vector3(1,0,0).cross(axis);
        if(lat.lengthSq() < 1e-8) lat.set(1,0,0); else lat.normalize();
        const fwd = axis.clone().cross(lat).normalize();
        torsoProfile = measureTorsoProfile(sm, sm.skeleton, cap.hipMid, axis, lat, fwd);
      }
    }
  }
  const hips = nodes.get('hips');
  const thigh = cap.meta.get('hipL')?.worldLen || .3, shin = cap.meta.get('kneeL')?.worldLen || .3;
  const hipJointY = cap.hipMid ? cap.hipMid.y : (cap.ankleY + thigh + shin);
  rig = {kind:'model', nodes, meta:cap.meta, initQuat:cap.initQuat, basis:cap.basis,
         torso:cap.torso, scale:cap.torso, root:outer, yawNode, baseQuat:outer.quaternion.clone(),
         hipsRest:hips.position.clone(),
         restHipY:hips.getWorldPosition(new THREE.Vector3()).y,
         hipJointY, torsoHip:cap.torsoHip, vertReach:Math.max(.05, hipJointY - cap.ankleY),
         ankleGroundY:cap.ankleY, modelScale:k, proxyR, palmAxis, torsoProfile, fingerTip, fingerRoot, fingerMid, fingerTips};
  return outer;
}
export function buildFigure(){
  return humanAsset ? (buildHumanFigure() || buildProceduralFigure()) : buildProceduralFigure();
}
