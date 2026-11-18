import { createDepthScale, DETAIL, type DepthEstimator, type DepthMap } from './depth';
import { createFireSound, type FireSound } from './fire-sound';
import { requestGpu } from './gpu';
import { GpuTimer } from './gpu-timer';
import { type OfflineJob, type RenderedFrame, renderOffline } from './offline-render';
import { type Plate, seekPlate } from './plate';
import { type Camera, cameraBasis, createRenderer, type Vec3 } from './render';
import { BLAST_DECAY, type Box, BURST_EVERY, DT, MAX_BOXES, type Preset, QUALITIES, type Quality, type Tweaks } from './scene';
import { createSimulation } from './simulation';

const FOV = (35 * Math.PI) / 180;
const ORBIT_DISTANCE = 3;
const MAX_PIXEL_RATIO = 2;
const TAA_BLEND = 0.2;
const TAA_BLEND_MOVING = 0.5;
// The raymarch resolution relative to the canvas, within the quality's bounds: it steps down while frames fall
// short of the target rate and back up once they have held; a step up that fails waits twice as long next time.
const SCALE_STEP = 0.125;
const SCALE_HOLD_MS = 3000;
const SCALE_HOLD_MAX_MS = 48000;
// While the camera moves the march runs this much lower in scale; the frame is softer and smooth, then sharpens.
const ORBIT_SCALE_DROP = 0.25;

export type EngineStats = {
  fps: number;
  displayHz: number;
  renderScale: number;
  renderWidth: number;
  renderHeight: number;
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

// Where the picture sits in the frame: x and y slide it, in half-frames to the right and up, and scale zooms it
// about that point. All three leave the view of the fire itself as the orbit set it.
export type Framing = { x: number; y: number; scale: number };

// Where the fire stands in the footage's depth, 0 in front of everything and 1 behind it all, and whether the live
// view marks what covers it.
export type Occlusion = { depth: number; show: boolean };

export type Engine = Awaited<ReturnType<typeof createEngine>>;

export async function createEngine(canvas: HTMLCanvasElement, initialPreset: Preset, initialQuality: Quality = QUALITIES.balanced) {
  const { device, format, adapterName, hasTimestamps } = await requestGpu();
  const context = canvas.getContext('webgpu')!;
  context.configure({ device, format, alphaMode: 'premultiplied' });

  let quality = initialQuality;
  let simulation = createSimulation(device, quality);
  let renderer = createRenderer(device, format, simulation.volumes, quality);
  const timer = new GpuTimer(device, hasTimestamps);

  let preset = initialPreset;
  let tweaks: Tweaks = { heat: 1, vorticity: initialPreset.vorticity, wind: 0, windYaw: 0, seed: 0, blue: initialPreset.blue, sparks: initialPreset.sparks, smoke: 1, width: 1 };
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
  let plate: Plate | null = null;
  let framing: Framing = { x: 0, y: 0, scale: 1 };
  // The footage's depth while the user has it on: the estimator, the scale that ties every frame to the shot's
  // first, and the map of the frame the plate is showing.
  let depth: { estimator: DepthEstimator; scale: ReturnType<typeof createDepthScale>; map: DepthMap | null } | null = null;
  let occlusion: Occlusion = { depth: 0.5, show: false };
  // Counts depth runs; one that finds the count moved on has been replaced and stops.
  let depthRun = 0;
  // The depth setting as nearness, a little past both ends of the shot's range so 0 covers nothing and 1 everything.
  const nearness = () => 1.05 - 1.1 * occlusion.depth;

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
  const camera = (): Camera => ({ eye: eye(), target, tanHalfFov: Math.tan(FOV / 2) / framing.scale, shift: [framing.x, framing.y] });

  let fpsFrames = 0;
  let fpsSince = 0;
  let fps = 0;
  let stepsSince = 0;
  let stepCount = 0;
  let stepsPerSecond = 0;
  let residual = 0;
  let frameIndex = 0;
  let renderScale = quality.scaleMax;
  let liveWidth = 1;
  let liveHeight = 1;
  let scaleHoldUntil = 0;
  let scaleHoldMs = SCALE_HOLD_MS;
  let lastRiseAt = -Infinity;
  let displayHz = 60;
  const frameGaps: number[] = [];
  let pendingQuality: Quality | null = null;

  // The display rate comes from the shortest frame gaps. The target is the lower of it and the quality's rate: a
  // sharp 60 on a 120 Hz screen is worth more than a blurred 110.
  function adjustQuality(now: number) {
    frameGaps.sort((a, b) => a - b);
    displayHz = Math.min(360, 1000 / frameGaps[Math.floor(frameGaps.length * 0.1)]);
    frameGaps.length = 0;
    const target = Math.min(displayHz, quality.targetFps);
    if (fps < target * 0.9 && renderScale > quality.scaleMin) {
      renderScale = Math.max(quality.scaleMin, renderScale - SCALE_STEP);
      scaleHoldMs = now - lastRiseAt < 3000 ? Math.min(scaleHoldMs * 2, SCALE_HOLD_MAX_MS) : SCALE_HOLD_MS;
      scaleHoldUntil = now + scaleHoldMs;
    } else if (fps >= target * 0.97 && now > scaleHoldUntil && renderScale < quality.scaleMax) {
      renderScale = Math.min(quality.scaleMax, renderScale + SCALE_STEP);
      lastRiseAt = now;
    }
  }

  // A new grid or light volume means a new simulation, which starts the fire over; anything else is a render setting.
  function applyQuality(next: Quality) {
    const rebuild = next.grid.join() !== quality.grid.join() || next.lightScale !== quality.lightScale;
    quality = next;
    if (rebuild) {
      renderer.dispose();
      simulation.dispose();
      simulation = createSimulation(device, quality);
      renderer = createRenderer(device, format, simulation.volumes, quality);
      renderer.setPlate(plate);
      renderer.setDepth(depth?.map ?? null, nearness());
      renderer.markCover(occlusion.show);
      simulation.clear();
      startBurst();
    } else {
      renderer.setQuality(quality);
    }
    renderScale = Math.min(quality.scaleMax, Math.max(quality.scaleMin, renderScale));
    scaleHoldUntil = 0;
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
    if (audible() && preset.mode === 'explosion') sound?.burst();
  }

  // One simulation step of dt seconds; a live explosion re-arms its charge every BURST_EVERY seconds, an exported one goes off once.
  function advance(pass: GPUComputePassEncoder, dt: number, slot = 0, next?: () => GPUComputePassEncoder) {
    if (!exporting && preset.mode === 'explosion' && simTime - burstStart > BURST_EVERY) startBurst();
    const age = simTime - burstStart;
    const emitting = preset.mode !== 'explosion' || age < (preset.burst ?? 0);
    const blast = preset.mode === 'explosion' ? (preset.blast ?? 0) * Math.exp(-age / BLAST_DECAY) : 0;
    simulation.writeParams(slot, { preset, tweaks, boxes, time: simTime, dt, emitting, blast });
    simulation.step(pass, slot, next);
    simTime += dt;
    stepCount++;
  }

  function tick(now: number) {
    if (!running || !visible || exporting) {
      looping = false;
      return;
    }
    if (pendingQuality) {
      applyQuality(pendingQuality);
      pendingQuality = null;
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

    // The lighter march lasts only while the orbit still visibly moves; the blend keeps easing until it fully stops.
    const moving = Boolean(drag) || Math.abs(spinAzimuth) + Math.abs(spinPolar) > 2e-3;
    const scale = moving ? Math.max(quality.scaleMin, renderScale - ORBIT_SCALE_DROP) : renderScale;
    liveWidth = Math.max(1, Math.round(canvas.width * scale));
    liveHeight = Math.max(1, Math.round(canvas.height * scale));
    if (plate?.source instanceof HTMLVideoElement && plate.source.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      renderer.writePlate(plate.source);
    }
    renderer.draw(encoder, context.getCurrentTexture().createView(), {
      camera: camera(),
      preset,
      blue: tweaks.blue,
      time: simTime,
      index: frameIndex++,
      scalarIndex: simulation.scalarIndex,
      width: liveWidth,
      height: liveHeight,
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

  // Estimates the depth of one frame of the plate, at `longSide` pixels of detail, and hands it to the renderer.
  async function readDepth(source: Plate['source'], run: number, longSide: number) {
    if (!plate || !depth) return;
    const map = depth.scale(await depth.estimator.estimate(source, plate.width, plate.height, longSide));
    if (run !== depthRun) return;
    depth.map = map;
    renderer.setDepth(map, nearness());
  }

  // A playing video's depth follows it as fast as the model runs at live detail, a few frames behind the picture,
  // until another run takes over. An export does not wait on it: it reads each frame it renders at full detail.
  async function followDepth(video: HTMLVideoElement, run: number) {
    while (run === depthRun) {
      await new Promise(requestAnimationFrame);
      if (run !== depthRun || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) continue;
      // A video that is being swapped out or is between frames has nothing to read; the next turn tries again.
      const frame = await createImageBitmap(video).catch(() => null);
      if (!frame) continue;
      await readDepth(frame, run, DETAIL.live);
      frame.close();
    }
  }

  // Starts the plate's depth over and resolves once its first frame's depth is showing: that frame sets the scale
  // every later one is read on. With depth off or no plate it clears the matte instead.
  async function startDepth() {
    const run = ++depthRun;
    const video = plate?.source instanceof HTMLVideoElement ? plate.source : null;
    if (!plate || !depth) {
      // A run this one replaced may have left the video paused on its first frame.
      if (!exporting) void video?.play();
      return renderer.setDepth(null, 0);
    }
    depth.scale = createDepthScale();
    depth.map = null;
    if (video) {
      video.pause();
      await seekPlate(plate, 0);
    }
    await readDepth(plate.source, run, video ? DETAIL.live : DETAIL.full);
    if (!video || run !== depthRun || exporting) return;
    void video.play();
    void followDepth(video, run);
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
      tweaks = { ...tweaks, heat: 1, vorticity: next.vorticity, wind: 0, blue: next.blue, sparks: next.sparks, smoke: 1, width: 1 };
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
    // Takes effect at the next frame, or at once while the loop is idle; an export keeps the quality it started with.
    setQuality(next: Quality) {
      if (looping || exporting) pendingQuality = next;
      else applyQuality(next);
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
    // The user's footage behind the fire, or none; the frame it is drawn in is the canvas.
    setPlate(next: Plate | null) {
      plate = next;
      renderer.setPlate(plate);
      return startDepth();
    },
    // Turns the footage's depth on with an estimator, or off with none; resolves once the first frame's depth shows.
    setDepth(estimator: DepthEstimator | null) {
      depth = estimator && { estimator, scale: createDepthScale(), map: null };
      return startDepth();
    },
    setOcclusion(next: Occlusion) {
      occlusion = next;
      renderer.markCover(next.show);
      renderer.setHoldout(nearness());
    },
    // The nearness of the footage at a point of the frame, both from 0 to 1, or null with no depth to read.
    nearnessAt(u: number, v: number) {
      const map = depth?.map;
      if (!map) return null;
      const x = Math.min(map.width - 1, Math.floor(u * map.width));
      const y = Math.min(map.height - 1, Math.floor(v * map.height));
      return map.data[y * map.width + x];
    },
    setFraming(next: Framing) {
      framing = next;
      renderer.resetHistory();
    },
    // Canvas pixel of a world point, or null behind the camera; and the floor point under a canvas pixel.
    project(point: Vec3): [number, number] | null {
      const view = camera();
      const { forward, right, up } = cameraBasis(view);
      const r = point.map((v, i) => v - view.eye[i]) as Vec3;
      const z = dot(r, forward);
      if (z <= 0.01) return null;
      const aspect = canvas.clientWidth / canvas.clientHeight;
      return [
        (0.5 + 0.5 * (dot(r, right) / (z * view.tanHalfFov * aspect) + framing.x)) * canvas.clientWidth,
        (0.5 - 0.5 * (dot(r, up) / (z * view.tanHalfFov) + framing.y)) * canvas.clientHeight,
      ];
    },
    floorPoint(x: number, y: number): Vec3 | null {
      const view = camera();
      const { forward, right, up } = cameraBasis(view);
      const sx = (2 * (x / canvas.clientWidth) - 1 - framing.x) * view.tanHalfFov * (canvas.clientWidth / canvas.clientHeight);
      const sy = (1 - 2 * (y / canvas.clientHeight) - framing.y) * view.tanHalfFov;
      const dir = forward.map((f, i) => f + right[i] * sx + up[i] * sy) as Vec3;
      if (dir[1] >= -1e-4) return null;
      const t = -view.eye[1] / dir[1];
      return view.eye.map((e, i) => e + dir[i] * t) as Vec3;
    },
    // Everything an export needs to be re-rendered later.
    snapshot() {
      return {
        preset,
        tweaks: { ...tweaks },
        boxes: boxes.map((box) => ({ ...box })),
        camera: { ...camera(), target: [...target] as Vec3 },
        orbit: { azimuth, polar, distance: ORBIT_DISTANCE, fovDegrees: (FOV * 180) / Math.PI },
        framing: { ...framing },
        plate: plate && { name: plate.name, width: plate.width, height: plate.height },
        occlusion: plate && depth ? { depth: occlusion.depth } : null,
        grid: simulation.grid,
      };
    },
    // Pauses the live loop, restarts the simulation from nothing, on the job's grid if it names one, and renders the
    // job frame by frame; the preview's grid comes back afterwards. A job that composites the footage or holds the
    // fire out behind it steps the plate, and its depth, through the clip a frame at a time; the plate plays on
    // when it is done.
    async render(job: OfflineJob, onFrame: (frame: RenderedFrame) => Promise<void>, signal: AbortSignal) {
      if (exporting) throw new Error('An export is already running.');
      exporting = true;
      cancelAnimationFrame(frame);
      looping = false;
      syncSound();
      fps = stepsPerSecond = 0;
      const previewQuality = quality;
      const footage = job.plate || job.holdout ? plate : null;
      const video = footage?.source instanceof HTMLVideoElement ? footage.source : null;
      // Whatever depth run was following the plate stops here; the export takes the frames it needs itself.
      const run = ++depthRun;
      video?.pause();
      const showPlate = footage
        ? async (seconds: number) => {
            await seekPlate(footage, seconds);
            renderer.writePlate(footage.source);
            if (video) await readDepth(video, run, DETAIL.full);
          }
        : undefined;
      const step = (dt: number) => {
        let encoder = device.createCommandEncoder();
        let pass = encoder.beginComputePass();
        const next = () => {
          pass.end();
          device.queue.submit([encoder.finish()]);
          encoder = device.createCommandEncoder();
          pass = encoder.beginComputePass();
          return pass;
        };
        advance(pass, dt, 0, next);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return simTime;
      };
      try {
        if (job.simulation) {
          device.pushErrorScope('out-of-memory');
          applyQuality({ ...quality, ...job.simulation });
          if (await device.popErrorScope()) throw new Error('Not enough GPU memory for this simulation grid; try a smaller one.');
        }
        simulation.clear();
        simTime = burstStart = 0;
        await renderOffline({ device, context, canvas, simulation, renderer, preset, blue: tweaks.blue, step, showPlate }, job, onFrame, signal);
      } finally {
        exporting = false;
        if (depth) void startDepth();
        else void video?.play();
        if (job.simulation) applyQuality(previewQuality);
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
        renderWidth: liveWidth,
        renderHeight: liveHeight,
        gpuMs: times.frame ?? 0,
        simulationMs: times.simulation ?? 0,
        lightingMs: times.lighting ?? 0,
        raymarchMs: times.render ?? 0,
        taaMs: times.taa ?? 0,
        compositeMs: times.composite ?? 0,
        stepsPerSecond,
        residual,
        grid: simulation.grid,
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

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
