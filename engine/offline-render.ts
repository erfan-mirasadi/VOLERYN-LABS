import { type Camera, cameraBasis, type Renderer, unit, type Vec3 } from './render';
import { BOX_MAX, BOX_MIN, type Preset, type SimulationGrid, type Tweaks } from './scene';
import { LOOP_SOURCE, type Simulation } from './simulation';

const MAX_OVERLAP = 32;
// A kept loop frame is the scalars and two half-size flow fields, ten bytes a cell; the kept set stays under this.
const KEPT_BUDGET_BYTES = 3 * 2 ** 30;
const RIM_SAMPLES = 64;
// The content probe's frame size and, as raw half bits, the alpha a pixel needs to count as content.
const PROBE_SIZE = 64;
const PROBE_ALPHA = new Uint16Array(new Float16Array([0.01]).buffer)[0];
// One staging slot above this size is enough: two would hold a quarter gigabyte for a 4K EXR.
const SINGLE_SLOT_BYTES = 128 * 2 ** 20;

export type Plane = 'display' | 'clean' | 'fire' | 'firelight' | 'smoke' | 'data';
export type Fps = 24 | 25 | 30 | 60;

export type OfflineJob = {
  width: number;
  height: number;
  fps: Fps;
  frames: number;
  // Seconds simulated before the first frame.
  warmup: number;
  // Simulated frames per rendered one; a flipbook keeps every stride-th frame so its cells span the clip.
  stride: number;
  samples: number;
  camera: Camera;
  loop: boolean;
  read: Plane[];
  // How long the shutter stays open, as a share of the frame interval; 0 renders the instant.
  shutter: number;
  // The grid to simulate on; without one the engine renders on the grid it is showing.
  simulation?: SimulationGrid;
  // Lays the display frame over the loaded footage; the light groups stay the fire's own.
  plate?: boolean;
  // Holds the display frame out behind whatever in the footage is nearer than the fire, by the footage's depth.
  holdout?: boolean;
};

// Raw rgba16float texels, four halves per pixel, row y starting at y * stride halves.
export type RenderedFrame = {
  index: number;
  width: number;
  height: number;
  stride: number;
  planes: Partial<Record<Plane, Uint16Array>>;
};

// Progress counts completed GPU work within the current phase. Setup has no known total; an ETA is emitted
// only after completed work has supplied a timing sample. The export layer adds encoding and saving phases.
export type OfflineProgress = {
  phase: 'setup' | 'warmup' | 'loop' | 'render';
  completed: number;
  total: number;
  elapsedMs: number;
  etaMs?: number;
  simulatedSeconds?: number;
};

export type OfflineWork = { simulationSteps: number; volumeSamples: number; displayImages: number };

export type OfflineContext = {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  simulation: Simulation;
  renderer: Renderer;
  preset: Preset;
  blue: number;
  tweaks: Tweaks;
  // Runs one simulation step and returns the simulation time after it.
  step: (dt: number) => number;
  // Puts the footage plate, and its depth, on the frame for a clip time, when the job uses either.
  showPlate?: (seconds: number) => Promise<void>;
  onProgress?: (progress: OfflineProgress) => void;
  onWork?: (work: OfflineWork) => void;
  // A look-ahead requests a conservative envelope for the whole loop before any kept frames are captured.
  // Context simulation/renderer may be getters: an adaptive resize replaces both between steps.
  adaptDomain?: (lookAheadSeconds?: number) => Promise<void>;
};

export function halton(index: number, base: number) {
  let f = 1;
  let r = 0;
  for (let i = index; i > 0; i = Math.floor(i / base)) {
    f /= base;
    r += f * (i % base);
  }
  return r;
}

export const substepsFor = (fps: Fps) => Math.ceil(60 / fps);

// The pixel filter: a Gaussian of half a pixel cut at a pixel and a half, as pbrt's is, drawn by its own
// distribution so every sample keeps the same weight. The sub-pixel offset of sample s.
const FILTER_SIGMA = 0.5;
const FILTER_RADIUS = 1.5;
function filterOffset(s: number): [number, number] {
  const r = Math.min(FILTER_RADIUS, FILTER_SIGMA * Math.sqrt(-2 * Math.log(1 - halton(s + 1, 2))));
  const angle = 2 * Math.PI * halton(s + 1, 3);
  return [r * Math.cos(angle), r * Math.sin(angle)];
}
// How many frames the loop's tail is crossfaded toward its head: a quarter of the clip, as far as the kept frames fit.
export const loopOverlap = (frames: number, grid: readonly number[]) =>
  Math.min(MAX_OVERLAP, Math.floor(KEPT_BUDGET_BYTES / (10 * grid[0] * grid[1] * grid[2])), Math.max(4, Math.round(frames / 4)));

// A narrow lens on the user's orbit direction, pulled back until the domain's inscribed cylinder fits a square frame
// with a margin: the box corners would hold an oblique view further out for nothing.
export function fitCamera(eye: Vec3, fovDegrees = 20, margin = 0.92, travel: Vec3 = [0, 0, 0]): Camera {
  const originalCentre = BOX_MIN.map((lo, k) => (lo + BOX_MAX[k]) / 2) as Vec3;
  const centre = originalCentre.map((c, k) => c + travel[k] / 2) as Vec3;
  const { forward, right, up } = cameraBasis({ eye, target: originalCentre, tanHalfFov: 0 });
  const t = Math.tan((fovDegrees * Math.PI) / 360);
  const radius = (BOX_MAX[0] - BOX_MIN[0]) / 2;
  let distance = 0;
  for (let k = 0; k < RIM_SAMPLES; k++) {
    const angle = (2 * Math.PI * k) / RIM_SAMPLES;
    for (const y of [BOX_MIN[1], BOX_MAX[1]]) {
      // Bound both ends of the translating window, before the content probe tightens the lens.
      for (const along of [-0.5, 0.5]) {
        const rel: Vec3 = [radius * Math.cos(angle) + along * travel[0], y - originalCentre[1] + along * travel[1], radius * Math.sin(angle) + along * travel[2]];
        distance = Math.max(distance, Math.max(Math.abs(dot(rel, right)), Math.abs(dot(rel, up))) / (margin * t) - dot(rel, forward));
      }
    }
  }
  return { eye: centre.map((c, i) => c - forward[i] * distance) as Vec3, target: centre, tanHalfFov: t };
}

// Fit an observed world envelope using the user's existing view direction. Adaptive fields may extend beyond
// the initial box, so their world-space bounds are also needed when the low-resolution image probe misses them.
export function fitBoundsCamera(camera: Camera, bounds: { min: Vec3; max: Vec3 }, fovDegrees = 20, margin = 0.92): Camera {
  const centre = bounds.min.map((low, axis) => (low + bounds.max[axis]) / 2) as Vec3;
  const { forward, right, up } = cameraBasis(camera);
  const tanHalfFov = Math.tan(fovDegrees * Math.PI / 360);
  let distance = 0.01;
  for (const x of [bounds.min[0], bounds.max[0]]) for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]]) {
    const rel: Vec3 = [x - centre[0], y - centre[1], z - centre[2]];
    distance = Math.max(distance, Math.max(Math.abs(dot(rel, right)), Math.abs(dot(rel, up))) / (margin * tanHalfFov) - dot(rel, forward));
  }
  return { eye: centre.map((value, axis) => value - forward[axis] * distance) as Vec3, target: centre, tanHalfFov };
}

// Narrows a fitted lens onto what the clip actually shows: a small draft of the same job finds the alpha extents over
// every frame, then the eye stays put while the view turns to their centre and zooms to them with the margin.
export async function fitToContent(render: (job: OfflineJob, onFrame: (frame: RenderedFrame) => Promise<void>) => Promise<void>, job: OfflineJob, margin = 0.92): Promise<Camera> {
  const n = PROBE_SIZE;
  let x0 = n;
  let x1 = -1;
  let y0 = n;
  let y1 = -1;
  await render({ ...job, width: n, height: n, samples: 1, read: ['display'] }, async ({ planes, stride }) => {
    const display = planes.display!;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const at = y * stride + 4 * x;
        // Additive embers have zero extinction alpha, but still belong inside the fitted frame.
        if (Math.max(display[at], display[at + 1], display[at + 2], display[at + 3]) < PROBE_ALPHA) continue;
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
      }
    }
  });
  if (x1 < 0) return job.camera;
  const { eye, target, tanHalfFov } = job.camera;
  const { forward, right, up } = cameraBasis(job.camera);
  const aspect = job.width / job.height;
  // Pixel bounds, padded by one probe pixel, as offsets from the view axis in tangent units.
  const edge = (pixel: number) => Math.min(1, Math.max(-1, (pixel / n) * 2 - 1));
  const xMin = edge(x0 - 1) * tanHalfFov * aspect;
  const xMax = edge(x1 + 2) * tanHalfFov * aspect;
  const yMin = -edge(y1 + 2) * tanHalfFov;
  const yMax = -edge(y0 - 1) * tanHalfFov;
  const cx = (xMin + xMax) / 2;
  const cy = (yMin + yMax) / 2;
  const dir = unit([forward[0] + cx * right[0] + cy * up[0], forward[1] + cx * right[1] + cy * up[1], forward[2] + cx * right[2] + cy * up[2]]);
  const distance = Math.hypot(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
  return {
    eye,
    target: [eye[0] + dir[0] * distance, eye[1] + dir[1] * distance, eye[2] + dir[2] * distance],
    tanHalfFov: Math.max((xMax - xMin) / 2 / aspect, (yMax - yMin) / 2) / margin,
  };
}

// Restarts nothing itself: the caller clears the simulation first. Every frame then takes the same substeps and
// the same samples, so the same job gives the same bytes. Frames are read back through a ring of staging buffers
// while the GPU renders the next one; onFrame sees each frame in order and may hold it until it returns.
export async function renderOffline(ctx: OfflineContext, job: OfflineJob, onFrame: (frame: RenderedFrame) => Promise<void>, signal: AbortSignal) {
  const { device } = ctx;
  const substeps = substepsFor(job.fps);
  const dt = 1 / (job.fps * substeps);
  let time = 0;
  const work: OfflineWork = { simulationSteps: 0, volumeSamples: 0, displayImages: 0 };
  let stepBatch = 1;
  let stepsSinceProbe = 0;
  const pause = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
  const progress = (phase: OfflineProgress['phase'], completed: number, total: number, started: number, simulatedSeconds?: number) => {
    const elapsedMs = performance.now() - started;
    ctx.onProgress?.({ phase, completed, total, elapsedMs,
      ...(completed >= 2 && completed < total && { etaMs: elapsedMs / completed * (total - completed) }),
      ...(simulatedSeconds !== undefined && { simulatedSeconds }),
    });
  };
  // Keep the queue short enough to accept a cancel and repaint. These are the same fixed solver steps as before;
  // only their submission batches change. A measured slow step reduces the next batch to one.
  const advance = async (count: number, adapt: boolean, completed?: (done: number) => void) => {
    let queued = 0;
    let batchStarted = performance.now();
    for (let n = 1; n <= count; n++) {
      signal.throwIfAborted();
      time = ctx.step(dt);
      work.simulationSteps++;
      queued++;
      stepsSinceProbe++;
      // Probe timing is part of the physical replay. Never let the performance-tuned queue batch move a
      // resize from step four to step five on a faster/slower run.
      if (queued >= stepBatch || n === count || (adapt && ctx.adaptDomain && stepsSinceProbe >= 4)) {
        await device.queue.onSubmittedWorkDone();
        const msPerStep = (performance.now() - batchStarted) / queued;
        stepBatch = Math.max(1, Math.min(4, Math.floor(250 / Math.max(1, msPerStep))));
        signal.throwIfAborted();
        if (adapt && ctx.adaptDomain && stepsSinceProbe >= 4) {
          await ctx.adaptDomain();
          stepsSinceProbe = 0;
          signal.throwIfAborted();
        }
        completed?.(n);
        await pause();
        signal.throwIfAborted();
        queued = 0;
        batchStarted = performance.now();
      }
    }
  };
  const warmupSteps = Math.round(job.warmup / dt);
  if (warmupSteps > 0) {
    const started = performance.now();
    progress('warmup', 0, warmupSteps, started, 0);
    await pause();
    await advance(warmupSteps, true, (done) => progress('warmup', done, warmupSteps, started, done * dt));
  }
  signal.throwIfAborted();
  if (ctx.adaptDomain) {
    // Kept loop fields must share one geometry. Reserve their full envelope now, then keep it fixed until the
    // loop is finished; ordinary clips can continue expanding between frames.
    await ctx.adaptDomain(job.loop ? (job.frames + MAX_OVERLAP) * job.stride / job.fps : undefined);
    stepsSinceProbe = 0;
    signal.throwIfAborted();
  }
  const overlap = job.loop ? loopOverlap(job.frames, ctx.simulation.grid) : 0;
  const bytesPerRow = Math.ceil((8 * job.width) / 256) * 256;
  const planeBytes = bytesPerRow * job.height;
  const slotBytes = job.read.length * planeBytes;
  const slots = slotBytes > SINGLE_SLOT_BYTES ? 1 : 2;
  let targetRenderer = ctx.renderer;
  let targets: ReturnType<Renderer['createExportTargets']> | undefined;
  const createTargets = async () => {
    device.pushErrorScope('out-of-memory');
    const next = ctx.renderer.createExportTargets(job.width, job.height, { overPlate: Boolean(job.plate), holdout: Boolean(job.holdout), clean: job.read.includes('clean') });
    if (await device.popErrorScope()) {
      next.destroy();
      throw new Error('Not enough GPU memory for this size; try 1920 × 1080.');
    }
    targets?.destroy();
    targets = next;
    targetRenderer = ctx.renderer;
  };
  const setupStarted = performance.now();
  progress('setup', 0, 0, setupStarted);
  await pause();
  await createTargets();
  device.pushErrorScope('out-of-memory');
  const staging = Array.from({ length: slots }, () => device.createBuffer({ size: slotBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  if (await device.popErrorScope()) {
    targets?.destroy();
    for (const buffer of staging) buffer.destroy();
    throw new Error('Not enough GPU memory for this size; try 1920 × 1080.');
  }
  const busy = staging.map(() => Promise.resolve());
  let previous = Promise.resolve();
  let renderingStarted = performance.now();
  const loopStarted = performance.now();
  if (overlap) progress('loop', 0, overlap, loopStarted);
  else progress('render', 0, job.frames, renderingStarted);
  let sampleBatch = 1;
  try {
    for (let i = 0; i < job.frames + overlap; i++) {
      signal.throwIfAborted();
      await advance(substeps * job.stride, !job.loop);
      const simulation = ctx.simulation;
      if (i < overlap) {
        simulation.keepFrame(i);
        await device.queue.onSubmittedWorkDone();
        progress('loop', i + 1, overlap, loopStarted);
        await pause();
        continue;
      }
      const j = i - overlap;
      if (j === 0 && overlap) {
        renderingStarted = performance.now();
        progress('render', 0, job.frames, renderingStarted);
      }
      if (ctx.renderer !== targetRenderer) await createTargets();
      const renderer = ctx.renderer;
      const frameTargets = targets!;
      let source = simulation.scalarIndex;
      // There is no kept set of sparks to blend toward, so they fade out over the loop's tail and in over its head.
      let sparks = j < overlap ? smooth((j + 1) / overlap) : 1;
      if (i >= job.frames) {
        const x = smooth((i - job.frames + 1) / overlap);
        simulation.blendLoop(i - job.frames, x);
        source = LOOP_SOURCE;
        sparks = 1 - x;
      }
      const lighting = device.createCommandEncoder();
      const pass = lighting.beginComputePass();
      simulation.updateLighting(pass, source);
      pass.end();
      device.queue.submit([lighting.finish()]);
      // A few temporal samples at a time bound cancellation latency even for a 4K frame. Progress follows GPU
      // completion, rather than counting commands merely queued on the CPU.
      const frameTime = job.stride / job.fps;
      let queued = 0;
      let sampleStarted = performance.now();
      for (let s = 0; s < job.samples; s++) {
        signal.throwIfAborted();
        renderer.drawExportSample(frameTargets, {
          camera: job.camera, preset: ctx.preset, blue: ctx.blue, tweaks: ctx.tweaks, offset: simulation.offset,
          time, source, jitter: filterOffset(s), frame: j * job.samples + s, weight: 1 / job.samples,
          frameTime, timeOffset: job.shutter * frameTime * (halton(s + 1, 5) - 0.5), first: s === 0, sparks,
        });
        work.volumeSamples++;
        queued++;
        if (queued >= sampleBatch || s + 1 === job.samples) {
          await device.queue.onSubmittedWorkDone();
          const msPerSample = (performance.now() - sampleStarted) / queued;
          sampleBatch = Math.max(1, Math.min(4, Math.floor(250 / Math.max(1, msPerSample))));
          signal.throwIfAborted();
          progress('render', j + (s + 1) / job.samples, job.frames, renderingStarted);
          await pause();
          signal.throwIfAborted();
          queued = 0;
          sampleStarted = performance.now();
        }
      }
      const slot = j % slots;
      await busy[slot];
      signal.throwIfAborted();
      await ctx.showPlate?.((j * job.stride) / job.fps);
      signal.throwIfAborted();
      const buffer = staging[slot];
      const encoder = device.createCommandEncoder();
      renderer.drawDisplay(encoder, frameTargets);
      work.displayImages += frameTargets.cleanDisplay ? 2 : 1;
      renderer.drawPreview(encoder, ctx.context.getCurrentTexture().createView(), [ctx.canvas.width, ctx.canvas.height], frameTargets);
      job.read.forEach((plane, p) => {
        const texture = plane === 'display' ? frameTargets.display : plane === 'clean' ? frameTargets.cleanDisplay! : frameTargets.layers[plane];
        encoder.copyTextureToBuffer({ texture }, { buffer, offset: p * planeBytes, bytesPerRow, rowsPerImage: job.height }, [job.width, job.height]);
      });
      device.queue.submit([encoder.finish()]);
      // Promise.all keeps the map handled when an earlier frame's sink failed and the finally destroys the buffer.
      previous = busy[slot] = Promise.all([previous, buffer.mapAsync(GPUMapMode.READ)]).then(async () => {
        signal.throwIfAborted();
        const planes: RenderedFrame['planes'] = {};
        job.read.forEach((plane, p) => (planes[plane] = new Uint16Array(buffer.getMappedRange(p * planeBytes, planeBytes))));
        try { await onFrame({ index: j, width: job.width, height: job.height, stride: bytesPerRow / 2, planes }); }
        finally { buffer.unmap(); }
      });
    }
    await previous;
  } finally {
    await previous.catch(() => {});
    for (const buffer of staging) buffer.destroy();
    targets?.destroy();
    ctx.simulation.releaseKept();
    ctx.onWork?.({ ...work });
  }
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const smooth = (x: number) => x * x * (3 - 2 * x);
