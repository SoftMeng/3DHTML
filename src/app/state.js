import {humanAsset} from './figure.js';
import {$} from '../main.js';

/* ---------- 状态 ---------- */
export let THREE, OrbitControls, renderer, scene, camera, controls;
export let sceneRoot = null, setGroup = null, envGroup = null, figGroup = null, lightPool = null;
/* 灯排光束（实例化圆锥）：对照机位下相机常在光锥内，锥面会盖住人偶 → 对照视角隐藏 */
export let lightBeams = [];
export let spec = null, specName = '';
export let groups = new Map(), trails = new Map(), ghosts = [], shells = [], eventMarks = [], figNodes = new Map();
export let headRing = null;
export let view = 'schema', freeCam = false, aspectMode = 'viewport', pitchAdj = 0;
export let lastURLSync = 0;
/* 对照类视角：叠加轨迹/残影/光束，这些是数据叠加层，会干扰与视频的逐帧比对 */
export const isCompareView = () => view === 'match' || view === 'overlay';
export let playhead = 0, playing = false, speed = 1, loopRunning = true;
export let dirty = true, renderCount = 0, frameMs = 0, submitMsLast = 0, qualityLevel = 0;
export let lastRenderAt = 0, lastTickAt = 0, overBudgetMs = 0, underBudgetMs = 0, frameSamples = 0, firstRenderAt = 0;
export let crowdMesh = null, phonePoints = null, seatMesh = null, baseFog = null, poolOrigin = null;
export let spotLight = null, spotBeam = null, spotTarget = null;
export let camState = {pos:{x:0,y:0,z:0}, dir:{x:0,y:0,z:0}};
export let mediaList = [], videoAspect = null, videoName = '';
export let motion = null, mimic = false, motionInfo = '';
export let figP = null;                       /* 人偶比例参数（米） */
export let motionTargets = null;              /* IK 平滑目标缓存 */
export let motionTargets3 = null;             /* 三维偏移的低通缓存（pose3d 路径） */
export const mm = $('#minimap'), mmCtx = mm.getContext('2d');
export const MM_SIZE = 228;

/* ---------- 资源预算契约（产品层硬约束，与 JSON 体量无关） ---------- */
export const BUDGET = {
  maxFrameMs: 33,
  minFrameIntervalMs: 33,
  maxCrowd: 900,
  maxPhoneLights: 180,
  maxLights: 4,
  maxTriangles: 120000,
  maxDrawCalls: 40,
  idleRedraws: 1,
  warmupFrames: 20
};
export const GHOST_COUNT = 10;
export const QUALITY_LEVELS = [
  {id:'L0', crowd:BUDGET.maxCrowd, phone:BUDGET.maxPhoneLights, fog:true,  pixelRatio:2},
  {id:'L1', crowd:450, phone:90, fog:true,  pixelRatio:1},
  {id:'L2', crowd:200, phone:0,  fog:false, pixelRatio:1}
];

/* ---------- 释放 ---------- */
function disposeObject(obj){
  obj.traverse(o => {
    if(o.geometry) o.geometry.dispose();
    const m = o.material;
    if(m) (Array.isArray(m)?m:[m]).forEach(x => x.dispose());
  });
}
export function clearScene(){
  if(humanAsset && humanAsset.wrapper.parent) humanAsset.wrapper.parent.remove(humanAsset.wrapper);
  if(sceneRoot){ scene.remove(sceneRoot); disposeObject(sceneRoot); sceneRoot = null; }
  groups.clear(); trails.clear(); ghosts = []; shells = []; eventMarks = []; figNodes.clear();
  headRing = null; setGroup = null; envGroup = null; figGroup = null; lightPool = null;
  lightBeams = [];
  spotLight = null; spotBeam = null; spotTarget = null;
  motionTargets = null;
}

/* ---------- 跨模块写入的 setter（ESM 里导入的绑定不可写） ---------- */
export function setStateTHREE(v){ THREE = v; }
export function setStateAspectMode(v){ aspectMode = v; }
export function setStateBaseFog(v){ baseFog = v; }
export function setStateCamState(v){ camState = v; }
export function setStateCamera(v){ camera = v; }
export function setStateControls(v){ controls = v; }
export function setStateCrowdMesh(v){ crowdMesh = v; }
export function setStateDirty(v){ dirty = v; }
export function setStateEnvGroup(v){ envGroup = v; }
export function setStateFigGroup(v){ figGroup = v; }
export function setStateFrameMs(v){ frameMs = v; }
export function setStateFreeCam(v){ freeCam = v; }
export function setStateHeadRing(v){ headRing = v; }
export function setStateLastRenderAt(v){ lastRenderAt = v; }
export function setStateLastTickAt(v){ lastTickAt = v; }
export function setStateLightPool(v){ lightPool = v; }
export function setStateMediaList(v){ mediaList = v; }
export function setStateMimic(v){ mimic = v; }
export function setStateMotion(v){ motion = v; }
export function setStatePhonePoints(v){ phonePoints = v; }
export function setStatePlayhead(v){ playhead = v; }
export function setStatePlaying(v){ playing = v; }
export function setStatePoolOrigin(v){ poolOrigin = v; }
export function setStateQualityLevel(v){ qualityLevel = v; }
export function setStateRenderer(v){ renderer = v; }
export function setStateScene(v){ scene = v; }
export function setStateSceneRoot(v){ sceneRoot = v; }
export function setStateSeatMesh(v){ seatMesh = v; }
export function setStateSetGroup(v){ setGroup = v; }
export function setStateSpec(v){ spec = v; }
export function setStateSpecName(v){ specName = v; }
export function setStateSpotBeam(v){ spotBeam = v; }
export function setStateSpotLight(v){ spotLight = v; }
export function setStateSpotTarget(v){ spotTarget = v; }
export function setStateSubmitMsLast(v){ submitMsLast = v; }
export function setStateVideoAspect(v){ videoAspect = v; }
export function setStateVideoName(v){ videoName = v; }
export function setStateView(v){ view = v; }
