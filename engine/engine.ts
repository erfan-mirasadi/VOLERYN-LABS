import { requestGpu } from './gpu';
import { GpuTimer } from './gpu-timer';
import { createRenderer, type Vec3 } from './render';
import { type Box, DT, GRID, MAX_BOXES, type Preset, type Tweaks } from './scene';
import { createSimulation } from './simulation';

const FOV = (35 * Math.PI) / 180;
const ORBIT_DISTANCE = 3;
const MAX_PIXEL_RATIO = 1.25;
const BURST_LENGTH = 0.3;
const BURST_EVERY = 5;

export type EngineStats = {
  fps: number;
  simulationMs: number;
  lightingMs: number;
  renderMs: number;
  stepsPerSecond: number;
  grid: readonly number[];
  adapterName: string;
  hasTimestamps: boolean;
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
  let tweaks: Tweaks = { heat: 1, vorticity: initialPreset.vorticity, wind: 0 };
  let boxes: Box[] = initialPreset.objects?.map((box) => ({ ...box })) ?? [];
  let simTime = 0;
  let burstStart = 0;
  let accumulator = 0;
  let lastFrame = 0;
  let frame = 0;
  let running = true;
  let visible = true;
  let looping = false;

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

  function tick(now: number) {
    if (!running || !visible) {
      looping = false;
      return;
    }
    accumulator = Math.min(accumulator + (now - lastFrame) / 1000, 4 * DT);
    lastFrame = now;
    if (!drag && Math.abs(spinAzimuth) + Math.abs(spinPolar) > 1e-4) {
      azimuth += spinAzimuth;
      polar = clampPolar(polar + spinPolar);
      spinAzimuth *= 0.9;
      spinPolar *= 0.9;
    }

    const encoder = device.createCommandEncoder();
    const simulationPass = encoder.beginComputePass({ timestampWrites: timer.timestamps('simulation') });
    for (let n = 0; accumulator >= DT && n < 2; n++) {
      if (preset.mode === 'explosion' && simTime - burstStart > BURST_EVERY) burstStart = simTime;
      const emitting = preset.mode !== 'explosion' || simTime - burstStart < BURST_LENGTH;
      simulation.writeParams(preset, tweaks, boxes, simTime, emitting);
      simulation.step(simulationPass);
      accumulator -= DT;
      simTime += DT;
      stepCount++;
    }
    simulationPass.end();
    const lightingPass = encoder.beginComputePass({ timestampWrites: timer.timestamps('lighting') });
    simulation.updateLighting(lightingPass);
    lightingPass.end();

    renderer.writeView(
      { eye: eye(), target, tanHalfFov: Math.tan(FOV / 2), aspect: canvas.width / canvas.height, rect: [0, 0, canvas.width, canvas.height] },
      preset,
      simTime,
    );
    renderer.draw(encoder, context.getCurrentTexture().createView(), simulation.scalarIndex, timer.timestamps('render'));
    const collect = timer.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    collect();

    fpsFrames++;
    if (now - fpsSince > 500) {
      fps = (fpsFrames * 1000) / (now - fpsSince);
      stepsPerSecond = ((stepCount - stepsSince) * 1000) / (now - fpsSince);
      fpsSince = now;
      fpsFrames = 0;
      stepsSince = stepCount;
    }
    frame = requestAnimationFrame(tick);
  }

  function startLoop() {
    if (looping || !running) return;
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
    if (visible) startLoop();
  });
  visibilityObserver.observe(canvas);
  const onVisibilityChange = () => {
    visible = !document.hidden;
    if (visible) startLoop();
  };
  document.addEventListener('visibilitychange', onVisibilityChange);

  const onDeviceLost = device.lost.then((info) => {
    running = false;
    return info;
  });

  simulation.clear();
  startLoop();

  return {
    lost: onDeviceLost,
    setPreset(next: Preset) {
      preset = next;
      tweaks = { ...tweaks, vorticity: next.vorticity };
      boxes = next.objects?.map((box) => ({ ...box })) ?? [];
      simulation.clear();
      burstStart = simTime;
    },
    setTweaks(next: Partial<Tweaks>) {
      tweaks = { ...tweaks, ...next };
    },
    addBox(box: Box) {
      boxes = [...boxes, box].slice(-MAX_BOXES);
    },
    stats(): EngineStats {
      const times = timer.averages();
      return {
        fps,
        simulationMs: times.simulation ?? 0,
        lightingMs: times.lighting ?? 0,
        renderMs: times.render ?? 0,
        stepsPerSecond,
        grid: GRID,
        adapterName,
        hasTimestamps,
      };
    },
    dispose() {
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
