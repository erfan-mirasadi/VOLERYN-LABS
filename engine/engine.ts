import { createFireSound, type FireSound } from './fire-sound';
import { requestGpu } from './gpu';
import { GpuTimer } from './gpu-timer';
import { type OfflineJob, type RenderedFrame, renderOffline } from './offline-render';
import { createRenderer, type Vec3 } from './render';
import { BLAST_DECAY, type Box, BURST_EVERY, DT, GRID, MAX_BOXES, type Preset, type Tweaks } from './scene';
import { createSimulation } from './simulation';

const FOV = (35 * Math.PI) / 180;
const ORBIT_DISTANCE = 3;
const MAX_PIXEL_RATIO = 2;
const TAA_BLEND = 0.2;
const TAA_BLEND_MOVING = 0.5;
// The raymarch resolution relative to the canvas: it steps down while frames drop and back up once they have held.
const SCALE_MIN = 0.375;
const SCALE_MAX = 1;
const SCALE_STEP = 0.125;
const SCALE_HOLD_MS = 4000;

export type EngineStats = {
  fps: number;
  displayHz: number;
  renderScale: number;
  gpuMs: number;
  simulationMs: number;
  lightingMs: number;
  raymarchMs: number;
  taaMs: number;
  compositeMs: number;
  stepsPerSecond: number;
  residual: number;
  grid: readonly number[];
  adapterName: string;
  hasTimestamps: boolean;
  sound: number | AudioContextState | 'off';
};

export type Engine = Awaited<ReturnType<typeof createEngine>>;

export async function createEngine(canvas: HTMLCanvasElement, initialPreset: Preset) {
  const { device, format, adapterName, hasTimestamps } = await requestGpu();
  const context = canvas.getContext('webgpu')!;
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const simulation = createSimulation(device);
  const renderer = createRenderer(device, format, simulation.volumes);
  const timer = new GpuTimer(device, hasTimestamps);

  let preset = initialPreset;
  let tweaks: Tweaks = { heat: 1, vorticity: initialPreset.vorticity, wind: 0, seed: 0, blue: initialPreset.blue, sparks: initialPreset.sparks  };
  let boxes: Box[] = initialPreset.objects?.map((box) => ({ ...box })) ?? [];
  let simTime = 0;
  let burstStart = 0;
  let accumulator = 0;
  let lastFrame = 0;
  let frame = 0;
  let running = true;
  let visible = true;
  let looping = false;
  let exporting = false;
  let audio: AudioContext | null = null;
  let sound: FireSound | null = null;
  let soundOn = false;

  const target: Vec3 = [0, 0.72, 0];
  let azimuth = 0.5;
  let polar = 1.32;
  let spinAzimuth = 0;
  let spinPolar = 0;
  let drag: { x: number; y: number; id: number } | null = null;
  const clampPolar = (value: number) => Math.min(1.75, Math.max(0.6, value));
  const eye = (): Vec3 => [
    target[0] + ORBIT_DISTANCE * Math.sin(polar) * Math.sin(azimuth),
    target[1] + ORBIT_DISTANCE * Math.cos(polar),
    target[2] + ORBIT_DISTANCE * Math.sin(polar) * Math.cos(azimuth),
  ];

  let fpsFrames = 0;
  let fpsSince = 0;
  let fps = 0;
  let stepsSince = 0;
  let stepCount = 0;
  let stepsPerSecond = 0;
  let residual = 0;
  let frameIndex = 0;
  let renderScale = 0.5;
  let scaleHoldUntil = 0;
  let displayHz = 60;
  const frameGaps: number[] = [];

  // Frames are dropping when rAF runs clearly below the display rate, which the shortest gaps reveal.
  function adjustQuality(now: number) {
    frameGaps.sort((a, b) => a - b);
    displayHz = Math.min(360, 1000 / frameGaps[Math.floor(frameGaps.length * 0.1)]);
    frameGaps.length = 0;
    const dropping = fps < displayHz * 0.92;
    if (dropping && renderScale > SCALE_MIN) {
      renderScale -= SCALE_STEP;
      scaleHoldUntil = now + SCALE_HOLD_MS;
    } else if (!dropping && now > scaleHoldUntil && renderScale < SCALE_MAX) {
      renderScale += SCALE_STEP;
    }
  }

  // Sound plays only while the live loop runs; it fades out before the context stops so pausing never clicks.
  const audible = () => soundOn && visible && running && !exporting;
  function syncSound() {
    if (!audio || !sound) return;
    sound.fade(audible());
    if (audible()) {
      void audio.resume();
      return;
    }
    setTimeout(() => {
      if (audio && !audible()) void audio.suspend();
    }, 80);
  }

  function startBurst() {
    burstStart = simTime;
    if (soundOn && preset.mode === 'explosion') sound?.burst();
  }

  // One simulation step of dt seconds; a live explosion re-arms its charge every BURST_EVERY seconds, an exported one goes off once.
  function advance(pass: GPUComputePassEncoder, dt: number, slot = 0) {
    if (!exporting && preset.mode === 'explosion' && simTime - burstStart > BURST_EVERY) startBurst();
    const age = simTime - burstStart;
    const emitting = preset.mode !== 'explosion' || age < (preset.burst ?? 0);
    const blast = preset.mode === 'explosion' ? (preset.blast ?? 0) * Math.exp(-age / BLAST_DECAY) : 0;
    simulation.writeParams(slot, { preset, tweaks, boxes, time: simTime, dt, emitting, blast });
    simulation.step(pass, slot);
    simTime += dt;
    stepCount++;
  }

  function tick(now: number) {
    if (!running || !visible || exporting) {
      looping = false;
      return;
    }
    const gap = now - lastFrame;
    accumulator = Math.min(accumulator + gap / 1000, 4 * DT);
    lastFrame = now;
    if (gap > 0) frameGaps.push(gap);
    const spinning = Math.abs(spinAzimuth) + Math.abs(spinPolar) > 1e-4;
    if (!drag && spinning) {
      azimuth += spinAzimuth;
      polar = clampPolar(polar + spinPolar);
      spinAzimuth *= 0.9;
      spinPolar *= 0.9;
    }

    const encoder = device.createCommandEncoder();
    const simulationPass = encoder.beginComputePass({ timestampWrites: timer.timestamps('simulation') });
    for (let n = 0; accumulator >= DT && n < 2; n++) {
      advance(simulationPass, DT, n);
      accumulator -= DT;
    }
    simulationPass.end();
    const lightingPass = encoder.beginComputePass({ timestampWrites: timer.timestamps('lighting') });
    simulation.updateLighting(lightingPass);
    lightingPass.end();

    renderer.draw(encoder, context.getCurrentTexture().createView(), {
      camera: { eye: eye(), target, tanHalfFov: Math.tan(FOV / 2) },
      preset,
      blue: tweaks.blue,
      time: simTime,
      index: frameIndex++,
      scalarIndex: simulation.scalarIndex,
      width: Math.max(1, Math.round(canvas.width * renderScale)),
      height: Math.max(1, Math.round(canvas.height * renderScale)),
      blend: drag || spinning ? TAA_BLEND_MOVING : TAA_BLEND,
      timestamps: { render: timer.timestamps('render'), taa: timer.timestamps('taa'), composite: timer.timestamps('composite') },
    });
    const collect = timer.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    collect();
    if (soundOn && sound) {
      if (drag || spinning) sound.listen(eye(), target);
      sound.update();
    }

    fpsFrames++;
    if (now - fpsSince > 1000) {
      fps = (fpsFrames * 1000) / (now - fpsSince);
      stepsPerSecond = ((stepCount - stepsSince) * 1000) / (now - fpsSince);
      fpsSince = now;
      fpsFrames = 0;
      stepsSince = stepCount;
      adjustQuality(now);
      simulation.readResidual()?.then((value) => (residual = value), () => {});
    }
    frame = requestAnimationFrame(tick);
  }

  function startLoop() {
    if (looping || !running || exporting) return;
    looping = true;
    lastFrame = fpsSince = performance.now();
    fpsFrames = 0;
    frame = requestAnimationFrame(tick);
  }

  const onPointerDown = (event: PointerEvent) => {
    drag = { x: event.clientX, y: event.clientY, id: event.pointerId };
    canvas.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: PointerEvent) => {
    if (drag?.id !== event.pointerId) return;
    spinAzimuth = -(event.clientX - drag.x) * 0.008;
    spinPolar = -(event.clientY - drag.y) * 0.006;
    drag.x = event.clientX;
    drag.y = event.clientY;
    azimuth += spinAzimuth;
    polar = clampPolar(polar + spinPolar);
  };
  const onPointerUp = (event: PointerEvent) => {
    if (drag?.id === event.pointerId) drag = null;
  };
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);

  const resize = () => {
    const scale = Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO);
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale));
    canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale));
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);
  resize();

  // Only simulate while the canvas is on screen, so the rest of the page costs no GPU time.
  const visibilityObserver = new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting) && !document.hidden;
    syncSound();
    if (visible) startLoop();
  });
  visibilityObserver.observe(canvas);
  const onVisibilityChange = () => {
    visible = !document.hidden;
    syncSound();
    if (visible) startLoop();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);

  const onDeviceLost = device.lost.then((info) => {
    running = false;
    syncSound();
    return info;
  });

  simulation.clear();
  startLoop();

  return {
    lost: onDeviceLost,
    setPreset(next: Preset) {
      preset = next;
      tweaks = { heat: 1, vorticity: next.vorticity, wind: 0, seed: tweaks.seed, blue: next.blue, sparks: next.sparks  };
      boxes = next.objects?.map((box) => ({ ...box })) ?? [];
      simulation.clear();
      renderer.resetHistory();
      sound?.scene(preset, tweaks, boxes);
      startBurst();
    },
    setTweaks(next: Partial<Tweaks>) {
      tweaks = { ...tweaks, ...next };
      sound?.scene(preset, tweaks, boxes);
    },
    addBox(box: Box) {
      boxes = [...boxes, box].slice(-MAX_BOXES);
      sound?.scene(preset, tweaks, boxes);
    },
    // Called inside the click: browsers only start an AudioContext during a user gesture.
    setSound(on: boolean) {
      soundOn = on;
      if (on && !audio) {
        audio = new AudioContext();
        sound = createFireSound(audio, ORBIT_DISTANCE);
        sound.scene(preset, tweaks, boxes);
        sound.listen(eye(), target);
      }
      syncSound();
    },
    // Everything an export needs to be re-rendered later.
    snapshot() {
      return {
        preset,
        tweaks: { ...tweaks },
        boxes: boxes.map((box) => ({ ...box })),
        camera: { eye: eye(), target: [...target] as Vec3, tanHalfFov: Math.tan(FOV / 2) },
        orbit: { azimuth, polar, distance: ORBIT_DISTANCE, fovDegrees: (FOV * 180) / Math.PI },
      };
    },
    // Pauses the live loop, restarts the simulation from nothing and renders the job frame by frame.
    async render(job: OfflineJob, onFrame: (frame: RenderedFrame) => Promise<void>, signal: AbortSignal) {
      if (exporting) throw new Error('An export is already running.');
      exporting = true;
      cancelAnimationFrame(frame);
      looping = false;
      syncSound();
      fps = stepsPerSecond = 0;
      simulation.clear();
      simTime = burstStart = 0;
      const step = (dt: number) => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        advance(pass, dt);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return simTime;
      };
      try {
        await renderOffline({ device, context, canvas, simulation, renderer, preset, blue: tweaks.blue, step }, job, onFrame, signal);
      } finally {
        exporting = false;
        syncSound();
        stepsSince = stepCount;
        renderer.resetHistory();
        startLoop();
      }
    },
    stats(): EngineStats {
      const times = timer.averages();
      return {
        fps,
        displayHz,
        renderScale,
        gpuMs: times.frame ?? 0,
        simulationMs: times.simulation ?? 0,
        lightingMs: times.lighting ?? 0,
        raymarchMs: times.render ?? 0,
        taaMs: times.taa ?? 0,
        compositeMs: times.composite ?? 0,
        stepsPerSecond,
        residual,
        grid: GRID,
        adapterName,
        hasTimestamps,
        sound: !soundOn || !audio || !sound ? 'off' : audio.state === 'running' ? sound.level() : audio.state,
      };
    },
    dispose() {
      void audio?.close();
      audio = null;
      running = false;
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      visibilityObserver.disconnect();
      document.removeEventListener('visibilitychange', onVisibilityChange);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      timer.dispose();
      renderer.dispose();
      simulation.dispose();
      // Destroying the device retires this configuration without touching one a newer engine put on the same canvas.
      device.destroy();
    },
  };
}
