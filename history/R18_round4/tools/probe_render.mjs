#!/usr/bin/env node
/**
 * 渲染探针（单浏览器实例、一次性）：在指定机位/时刻取画面快照 + 关键量导出。
 *
 * 用法:
 *   node tools/probe_render.mjs --view=match --times=0,4,7.21,10,12.5 --tag=base
 *   node tools/probe_render.mjs --view=overlay --times=6.5,10 --tag=ov
 *
 * 纪律：同一时刻只开 1 个 Chrome；每步硬超时；finally kill 并 ps 自查。
 */
import {spawn, execSync} from 'node:child_process';
import {readFileSync, writeFileSync, existsSync, rmSync} from 'node:fs';
import {setTimeout as sleep} from 'node:timers/promises';
import {join, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import http from 'node:http';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9355;
const DEBUG_PORT = 9356;
const BASE = `http://127.0.0.1:${PORT}/`;
const SHOTS = join(ROOT, 'screenshots');

const arg = (k, d) => {
  const m = process.argv.find(a => a.startsWith(`--${k}=`));
  return m ? m.slice(k.length + 3) : d;
};
const VIEW = arg('view', 'match');
const TIMES = arg('times', '0,4,7.21,10').split(',').map(Number);
const TAG = arg('tag', 'probe');
const SPEC = arg('spec', '示例.json');
const QUERY = arg('query', '');        /* 深链还原验证：直接用 URL 参数启动 */

function httpGet(url, timeoutMs = 6000) {
  return new Promise((res, rej) => {
    const req = http.get(url, r => {
      let b = ''; r.on('data', c => b += c); r.on('end', () => res({status: r.statusCode, body: b}));
    });
    req.on('error', rej);
    req.setTimeout(timeoutMs, () => { req.destroy(); rej(new Error('http 超时 ' + url)); });
  });
}
class CDP {
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map(); this.handlers = [];
    ws.addEventListener('message', ev => {
      const m = JSON.parse(ev.data);
      if (m.id && this.pending.has(m.id)) {
        const p = this.pending.get(m.id); this.pending.delete(m.id);
        m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
      } else if (m.method) this.handlers.forEach(h => h(m));
    });
  }
  on(fn) { this.handlers.push(fn); }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    const payload = {id, method, params};
    if (sessionId) payload.sessionId = sessionId;
    this.ws.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => this.pending.set(id, {resolve, reject}));
  }
}
async function launchChrome() {
  const profile = `/tmp/probe-chrome-profile-${PORT}`;
  rmSync(profile, {recursive: true, force: true});
  const proc = spawn(CHROME, [
    '--headless=new', '--enable-unsafe-swiftshader', '--use-angle=swiftshader', '--no-first-run',
    '--disable-extensions', '--disable-background-networking', '--disable-sync', '--mute-audio',
    '--autoplay-policy=no-user-gesture-required', '--force-device-scale-factor=1', '--window-size=400,720',
    `--remote-debugging-port=${DEBUG_PORT}`, `--user-data-dir=${profile}`, 'about:blank'
  ], {stdio: 'ignore', detached: true});
  for (let i = 0; i < 40; i++) {
    try { const r = await httpGet(`http://127.0.0.1:${DEBUG_PORT}/json/version`, 2500); if (r.status === 200) return proc; } catch (e) {}
    await sleep(250);
  }
  throw new Error('Chrome 未能在 10s 内就绪');
}
async function connect() {
  const ver = JSON.parse((await httpGet(`http://127.0.0.1:${DEBUG_PORT}/json/version`)).body);
  const ws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error('WS 连接失败')); });
  const cdp = new CDP(ws);
  const {targetId} = await cdp.send('Target.createTarget', {url: 'about:blank'});
  const {sessionId} = await cdp.send('Target.attachToTarget', {targetId, flatten: true});
  return {cdp, sid: sessionId};
}

let chrome = null, server = null;
async function main() {
  let up = false;
  try { up = (await httpGet(BASE, 2500)).status === 200; } catch (e) {}
  if (!up) {
    server = spawn('python3', [join(ROOT, 'serve.py'), '--port', String(PORT)], {cwd: ROOT, stdio: 'ignore'});
    for (let i = 0; i < 20; i++) {
      await sleep(400);
      try { if ((await httpGet(BASE, 2000)).status === 200) { up = true; break; } } catch (e) {}
    }
    if (!up) throw new Error('serve.py 未就绪');
  }
  chrome = await launchChrome();
  const {cdp, sid} = await connect();
  await cdp.send('Runtime.enable', {}, sid);
  await cdp.send('Page.enable', {}, sid);
  const evl = async (expr, ms = 20000) => {
    const r = await Promise.race([
      cdp.send('Runtime.evaluate', {expression: expr, returnByValue: true, awaitPromise: true}, sid),
      new Promise((_, rej) => setTimeout(() => rej(new Error('超时:' + expr.slice(0, 60))), ms))
    ]);
    if (r.exceptionDetails) throw new Error('JS 异常: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
    return r.result.value;
  };
  const loaded = new Promise(res => {
    const h = m => { if (m.method === 'Page.loadEventFired') { cdp.handlers = cdp.handlers.filter(x => x !== h); res(); } };
    cdp.on(h);
  });
  await cdp.send('Page.navigate', {url: BASE + QUERY}, sid);
  await Promise.race([loaded, sleep(15000)]);
  for (let i = 0; i < 80; i++) {
    if (await evl('!!(window.__app&&window.__app.ready)')) break;
    await sleep(200);
  }
  await evl('window.__app.setLoop(false);window.__app.setQuality(2);');
  if (!QUERY) {
    await evl(`(()=>{const s=document.querySelector('#specSelect');s.value=${JSON.stringify(SPEC)};s.dispatchEvent(new Event('change'));return 1;})()`);
    await sleep(1200);
    await evl(`window.__app.setView(${JSON.stringify(VIEW)});`);
    await sleep(600);
  } else {
    /* 深链还原状态：错误面板必须为空，spec/view/aspect/t 须与 URL 一致 */
    const err = await evl('document.querySelector("#err").hidden ? "" : (document.querySelector("#errMsg").textContent+" | "+(document.querySelector("#errDetail").textContent||"").slice(0,300))');
    const st = await evl('JSON.stringify({ready:!!window.__app.ready, spec:window.__app.currentSpecName, t:+window.__app.playhead.toFixed(2), aspect:document.querySelector("#aspectSel").value, view:[...document.querySelectorAll("#viewSeg button")].filter(b=>b.getAttribute("aria-pressed")==="true").map(b=>b.dataset.view)[0]})');
    console.log('深链还原:', st, err ? ('错误面板=' + err) : '错误面板=空');
    if (err) throw new Error('深链启动报错: ' + err);
  }

  const out = [];
  for (const t of TIMES) {
    await evl(`window.__app.seek(${t})`);
    await sleep(250);
    const data = await evl('window.__app.snapshot()');
    const name = join(SHOTS, `probe_${TAG}_${String(t).replace('.', '_')}.png`);
    writeFileSync(name, Buffer.from(data.split(',')[1], 'base64'));
    const probe = await evl(`JSON.stringify({t:${t}, probe:window.__app.motionProbe(${t}), limbs:window.__app.limbAngles(${t}), hs:window.__app.handScreen(${t}), bend:window.__app.bendSide(${t}),
      model:window.__app.model, cam:window.__app.cameraState(), ext:window.__app.figureExtent(), ik:window.__app.ikError(),
      pen:window.__app.penetration(${t}), palm:window.__app.palmState(${t}), proxy:window.__app.proxyInfo()})`);
    out.push(JSON.parse(probe));
    console.log('→', name);
  }
  writeFileSync(join(SHOTS, `probe_${TAG}.json`), JSON.stringify(out, null, 1));
  console.log('探针数据 →', join(SHOTS, `probe_${TAG}.json`));
}
main().catch(e => { console.error('FAIL:', e.message); process.exitCode = 1; })
  .finally(() => {
    try { if (chrome) { try { process.kill(-chrome.pid, 'SIGKILL'); } catch (e) {} } } catch (e) {}
    try { if (server) server.kill('SIGTERM'); } catch (e) {}
    setTimeout(() => {
      let n = 0;
      try { n = parseInt(execSync('ps aux | grep "[C]hrome.*' + DEBUG_PORT + '" | wc -l').toString().trim(), 10); } catch (e) { n = 0; }
      console.log('残留 Chrome 进程数 =', n);
      process.exit(process.exitCode || 0);
    }, 800);
  });
