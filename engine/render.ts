import { COMPOSITE, RENDER, RESOLVE } from './render-shaders';
import { DENSITY, EXPOSURE, FIRE_LIGHT, FLOW_PERIOD, GRID, type Preset } from './scene';
import type { Volumes } from './simulation';

const VIEW_FLOATS = 40;
const TARGET_FORMAT: GPUTextureFormat = 'rgba16float';
const LIVE_STEP_CELLS = 1.5;
const LIVE_MAX_STEPS = 160;
export const BOX_MIN: Vec3 = [-0.5, 0, (-0.5 * GRID[2]) / GRID[0]];
export const BOX_MAX: Vec3 = [0.5, GRID[1] / GRID[0], (0.5 * GRID[2]) / GRID[0]];

export type Vec3 = [number, number, number];

export type Camera = { eye: Vec3; target: Vec3; tanHalfFov: number };

export type Frame = {
  camera: Camera;
  preset: Preset;
  time: number;
  // Counts rendered frames; it drives the per-frame jitter.
  index: number;
  scalarIndex: number;
  // Raymarch target size; the result is upsampled onto the canvas.
  width: number;
  height: number;
  // Weight of the new frame in the temporal blend.
  blend: number;
  timestamps: Record<'render' | 'taa' | 'composite', GPURenderPassTimestampWrites | undefined>;
};

type ViewParams = {
  camera: Camera;
  preset: Preset;
  time: number;
  width: number;
  height: number;
  // Sub-pixel offset of the sample in pixels, and the frame counter that moves the ray-start dither.
  jitter: [number, number];
  frame: number;
  stepCells: number;
  maxSteps: number;
  sampleWeight: number;
  // Seconds between exported frames, which turns velocity into motion; 0 live.
  frameTime: number;
};

export type Renderer = ReturnType<typeof createRenderer>;

export function createRenderer(device: GPUDevice, format: GPUTextureFormat, volumes: Volumes) {
  const pipeline = (module: GPUShaderModule, entryPoint: string, targets: GPUColorTargetState[]) =>
    device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint, targets },
      primitive: { topology: 'triangle-list' },
    });
  const fullscreen = (code: string, target: GPUTextureFormat) => pipeline(device.createShaderModule({ code }), 'fs', [{ format: target }]);
  const renderModule = device.createShaderModule({ code: RENDER });
  const raymarch = pipeline(renderModule, 'fs', [{ format: TARGET_FORMAT }]);
  const resolve = fullscreen(RESOLVE, TARGET_FORMAT);
  const composite = fullscreen(COMPOSITE, format);

  const viewBuffer = device.createBuffer({ size: VIEW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const blendBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const raymarchGroups = volumes.scalars.map((scalars, j) =>
    device.createBindGroup({
      layout: raymarch.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: viewBuffer } },
        { binding: 1, resource: scalars.createView() },
        { binding: 2, resource: volumes.shadow.createView() },
        { binding: 3, resource: volumes.sampler },
        { binding: 4, resource: { buffer: volumes.params } },
        { binding: 5, resource: volumes.light.createView() },
        { binding: 6, resource: volumes.noise.values.createView() },
        { binding: 7, resource: volumes.noise.repeat },
        { binding: 8, resource: volumes.flows[j][0].createView() },
        { binding: 9, resource: volumes.flows[j][1].createView() },
      ],
    }),
  );

  let targets: { width: number; height: number; current: GPUTexture; history: GPUTexture[]; resolveGroups: GPUBindGroup[]; compositeGroups: GPUBindGroup[] } | null = null;
  let historyIndex = 0;
  let historyValid = false;

  function ensureTargets(width: number, height: number) {
    if (targets && targets.width === width && targets.height === height) return targets;
    disposeTargets();
    const make = () =>
      device.createTexture({ size: [width, height], format: TARGET_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const current = make();
    const history = [make(), make()];
    const views = history.map((texture) => texture.createView());
    targets = {
      width,
      height,
      current,
      history,
      resolveGroups: views.map((view) =>
        device.createBindGroup({
          layout: resolve.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: blendBuffer } },
            { binding: 1, resource: current.createView() },
            { binding: 2, resource: view },
          ],
        }),
      ),
      compositeGroups: views.map((view) =>
        device.createBindGroup({
          layout: composite.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: view },
            { binding: 1, resource: volumes.sampler },
          ],
        }),
      ),
    };
    historyValid = false;
    return targets;
  }

  function disposeTargets() {
    if (!targets) return;
    targets.current.destroy();
    targets.history.forEach((texture) => texture.destroy());
    targets = null;
  }

  const viewData = new Float32Array(VIEW_FLOATS);
  function writeView({ camera, preset, time, width, height, jitter, frame, stepCells, maxSteps, sampleWeight, frameTime }: ViewParams) {
    const { forward, right, up } = cameraBasis(camera);
    // Flow A's weight is 0 the moment it resets and 1 when B does; B gets the rest.
    const flowWeight = 1 - Math.abs(1 - 2 * ((time / FLOW_PERIOD) % 1));
    viewData.set([
      ...camera.eye, camera.tanHalfFov,
      ...forward, width / height,
      ...right, frameTime,
      ...up, stepCells / GRID[0],
      ...BOX_MIN, EXPOSURE,
      ...BOX_MAX, DENSITY,
      ...preset.tint, preset.glow,
      -jitter[0], -jitter[1], width, height,
      maxSteps, FIRE_LIGHT, frame, preset.fireDensity,
      preset.detail, flowWeight, sampleWeight, 0,
    ]);
    device.queue.writeBuffer(viewBuffer, 0, viewData);
  }

  const fullscreenPass = (encoder: GPUCommandEncoder, view: GPUTextureView, pipeline: GPURenderPipeline, group: GPUBindGroup, timestampWrites?: GPURenderPassTimestampWrites) => {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      timestampWrites,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  };

  function draw(encoder: GPUCommandEncoder, canvasView: GPUTextureView, frame: Frame) {
    const { current, history, resolveGroups, compositeGroups } = ensureTargets(frame.width, frame.height);
    writeView({ ...frame, jitter: [0, 0], frame: frame.index, stepCells: LIVE_STEP_CELLS, maxSteps: LIVE_MAX_STEPS, sampleWeight: 1, frameTime: 0 });
    device.queue.writeBuffer(blendBuffer, 0, new Float32Array([historyValid ? frame.blend : 1]));
    const next = 1 - historyIndex;
    fullscreenPass(encoder, current.createView(), raymarch, raymarchGroups[frame.scalarIndex], frame.timestamps.render);
    fullscreenPass(encoder, history[next].createView(), resolve, resolveGroups[historyIndex], frame.timestamps.taa);
    fullscreenPass(encoder, canvasView, composite, compositeGroups[next], frame.timestamps.composite);
    historyIndex = next;
    historyValid = true;
  }

  return {
    draw,
    resetHistory: () => {
      historyValid = false;
    },
    dispose() {
      disposeTargets();
      viewBuffer.destroy();
      blendBuffer.destroy();
    },
  };
}

export function cameraBasis({ eye, target }: Camera) {
  const forward = unit(sub(target, eye));
  const right = unit(cross(forward, [0, 1, 0]));
  return { forward, right, up: cross(right, forward) };
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const unit = (a: Vec3): Vec3 => {
  const length = Math.hypot(...a);
  return [a[0] / length, a[1] / length, a[2] / length];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
