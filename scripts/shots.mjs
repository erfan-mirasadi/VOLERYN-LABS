// Headless screenshots of the studio for checking a build by eye: node scripts/shots.mjs <origin> <outDir> [prefix]
// Needs Google Chrome on macOS. Shoots every preset on the dark backdrop, the explosion at four moments after its
// burst, and the whole studio, and prints the HUD and any console errors.
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';

const [, , origin, outDir, prefix = 'shot'] = process.argv;
const port = Number(process.env.CDP_PORT ?? 9463);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
mkdirSync(outDir, { recursive: true });
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${outDir}/chrome-profile-${port}`,
  '--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist', '--hide-scrollbars', '--mute-audio', 'about:blank',
], { stdio: 'ignore' });

try {
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch {}
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener('open', resolve, { once: true }));
  let nextId = 0;
  const pending = new Map();
  const problems = [];
  ws.addEventListener('message', (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
    else if (msg.method === 'Runtime.exceptionThrown') problems.push('exception: ' + (msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text).slice(0, 300));
    else if (msg.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(msg.params.type)) problems.push(msg.params.type + ': ' + msg.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 300));
  });
  const send = (method, params = {}) => new Promise((resolve) => { const id = ++nextId; pending.set(id, resolve); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.result?.value;
  const click = (name) => evaluate(`[...document.querySelectorAll('button')].find((b) => b.textContent === '${name}').click()`);
  const hud = () => evaluate(`[...document.querySelectorAll('div')].find((d) => d.textContent.startsWith('fps'))?.textContent`);
  const shot = async (name, clip = { x: 0, y: 0, width: 1120, height: 900 }) => {
    const result = await send('Page.captureScreenshot', { format: 'png', clip: { ...clip, scale: 1 } });
    writeFileSync(`${outDir}/${prefix}-${name}.png`, Buffer.from(result.result.data, 'base64'));
  };
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: `${origin}/studio` });
  await sleep(10000);
  await shot('studio', { x: 0, y: 0, width: 1440, height: 900 });
  await click('Dark');
  for (const preset of ['Fire', 'Campfire', 'Smoke']) {
    await click(preset);
    await sleep(6000);
    await shot(preset.toLowerCase());
    console.log(`${preset}: ${await hud()}`);
  }
  await click('Explosion');
  const t0 = performance.now();
  for (const [at, name] of [[120, 'burst-a'], [400, 'burst-b'], [1200, 'burst-c'], [3000, 'burst-d']]) {
    await sleep(Math.max(0, at - (performance.now() - t0)));
    await shot(name);
  }
  console.log(`Explosion: ${await hud()}`);
  console.log(problems.length ? problems.join('\n') : 'no console errors or warnings');
} finally {
  chrome.kill();
}
