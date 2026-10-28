import { type Camera, cameraBasis, type Renderer, unit, type Vec3 } from './render';
import { BOX_MAX, BOX_MIN, type Preset } from './scene';
import { LOOP_SOURCE, type Simulation } from './simulation';

const MAX_OVERLAP = 32;
const RIM_SAMPLES = 64;
// The content probe's frame size and, as raw half bits, the alpha a pixel needs to count as content.
const PROBE_SIZE = 64;
const PROBE_ALPHA = new Uint16Array(new Float16Array([0.01]).buffer)[0];
// One staging slot above this size is enough: two would hold a quarter gigabyte for a 4K EXR.
const SINGLE_SLOT_BYTES = 128 * 2 ** 20;

export type Plane = 'display' | 'fire' | 'firelight' | 'smoke' | 'data';
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
};

// Raw rgba16float texels, four halves per pixel, row y starting at y * stride halves.
export type RenderedFrame = {
  index: number;
  width: number;
  height: number;
  stride: number;
  planes: Partial<Record<Plane, Uint16Array>>;
};

export type OfflineContext = {
  device: GPUDevice;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  simulation: Simulation;
  renderer: Renderer;
  preset: Preset;
  blue: number;
  // Runs one simulation step and returns the simulation time after it.
  step: (dt: number) => number;
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
// How many frames the loop's tail is crossfaded toward its head.
export const loopOverlap = (frames: number) => Math.min(MAX_OVERLAP, Math.max(4, Math.round(frames / 4)));

// A narrow lens on the user's orbit direction, pulled back until the domain's inscribed cylinder fits a square frame
// with a margin: the box corners would hold an oblique view further out for nothing.
export function fitCamera(eye: Vec3, fovDegrees = 20, margin = 0.92): Camera {
  const centre = BOX_MIN.map((lo, k) => (lo + BOX_MAX[k]) / 2) as Vec3;
  const { forward, right, up } = cameraBasis({ eye, target: centre, tanHalfFov: 0 });
  const t = Math.tan((fovDegrees * Math.PI) / 360);
  const radius = (BOX_MAX[0] - BOX_MIN[0]) / 2;
  let distance = 0;
  for (let k = 0; k < RIM_SAMPLES; k++) {
    const angle = (2 * Math.PI * k) / RIM_SAMPLES;
    for (const y of [BOX_MIN[1], BOX_MAX[1]]) {
      const rel: Vec3 = [radius * Math.cos(angle), y - centre[1], radius * Math.sin(angle)];
      distance = Math.max(distance, Math.max(Math.abs(dot(rel, right)), Math.abs(dot(rel, up))) / (margin * t) - dot(rel, forward));
    }
  }
  return { eye: centre.map((c, i) => c - forward[i] * distance) as Vec3, target: centre, tanHalfFov: t };
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
        if (display[y * stride + 4 * x + 3] < PROBE_ALPHA) continue;
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
  const { device, simulation, renderer } = ctx;
  const substeps = substepsFor(job.fps);
  const dt = 1 / (job.fps * substeps);
  let time = 0;
  const warmupSteps = Math.round(job.warmup / dt);
  for (let n = 1; n <= warmupSteps; n++) {
    time = ctx.step(dt);
    if (n % 60 === 0) {
      await device.queue.onSubmittedWorkDone();
      signal.throwIfAborted();
    }
  }
  const overlap = job.loop ? loopOverlap(job.frames) : 0;
  const bytesPerRow = Math.ceil((8 * job.width) / 256) * 256;
  const planeBytes = bytesPerRow * job.height;
  const slotBytes = job.read.length * planeBytes;
  const slots = slotBytes > SINGLE_SLOT_BYTES ? 1 : 2;
  device.pushErrorScope('out-of-memory');
  const targets = renderer.createExportTargets(job.width, job.height);
  const staging = Array.from({ length: slots }, () => device.createBuffer({ size: slotBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  if (await device.popErrorScope()) {
    targets.destroy();
    for (const buffer of staging) buffer.destroy();
    throw new Error('Not enough GPU memory for this size; try 1920 × 1080.');
  }
  const busy = staging.map(() => Promise.resolve());
  let previous = Promise.resolve();
  try {
    for (let i = 0; i < job.frames + overlap; i++) {
      signal.throwIfAborted();
      for (let s = 0; s < substeps * job.stride; s++) time = ctx.step(dt);
      if (i < overlap) {
        simulation.keepFrame(i);
        continue;
      }
      const j = i - overlap;
      let source = simulation.scalarIndex;
      if (i >= job.frames) {
        const x = (i - job.frames + 1) / overlap;
        simulation.blendLoop(i - job.frames, x * x * (3 - 2 * x));
        source = LOOP_SOURCE;
      }
      const lighting = device.createCommandEncoder();
      const pass = lighting.beginComputePass();
      simulation.updateLighting(pass, source);
      pass.end();
      device.queue.submit([lighting.finish()]);
      for (let s = 0; s < job.samples; s++) {
        renderer.drawExportSample(targets, {
          camera: job.camera,
          preset: ctx.preset,
          blue: ctx.blue,
          time,
          source,
          jitter: [halton(s + 1, 2) - 0.5, halton(s + 1, 3) - 0.5],
          frame: j * job.samples + s,
          weight: 1 / job.samples,
          frameTime: job.stride / job.fps,
          first: s === 0,
        });
      }
      const slot = j % slots;
      await busy[slot];
      const buffer = staging[slot];
      const encoder = device.createCommandEncoder();
      renderer.drawDisplay(encoder, targets);
      renderer.drawPreview(encoder, ctx.context.getCurrentTexture().createView(), [ctx.canvas.width, ctx.canvas.height], targets);
      job.read.forEach((plane, p) => {
        const texture = plane === 'display' ? targets.display : targets.layers[plane];
        encoder.copyTextureToBuffer({ texture }, { buffer, offset: p * planeBytes, bytesPerRow, rowsPerImage: job.height }, [job.width, job.height]);
      });
      device.queue.submit([encoder.finish()]);
      // Promise.all keeps the map handled when an earlier frame's sink failed and the finally destroys the buffer.
      previous = busy[slot] = Promise.all([previous, buffer.mapAsync(GPUMapMode.READ)]).then(async () => {
        const planes: RenderedFrame['planes'] = {};
        job.read.forEach((plane, p) => (planes[plane] = new Uint16Array(buffer.getMappedRange(p * planeBytes, planeBytes))));
        await onFrame({ index: j, width: job.width, height: job.height, stride: bytesPerRow / 2, planes });
        buffer.unmap();
      });
    }
    await previous;
  } finally {
    await previous.catch(() => {});
    for (const buffer of staging) buffer.destroy();
    targets.destroy();
    simulation.releaseKept();
  }
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
