import { COMPOSITE, DISPLAY, RENDER, RESOLVE } from './render-shaders';
import { BOX_MAX, BOX_MIN, DENSITY, EXPOSURE, FIRE_LIGHT, FLOW_PERIOD, GRID, type Preset } from './scene';
import type { Volumes } from './simulation';
import { createSparkRenderer } from './sparks';

const VIEW_FLOATS = 40;
const TARGET_FORMAT: GPUTextureFormat = 'rgba16float';
const LIVE_STEP_CELLS = 1.5;
const LIVE_MAX_STEPS = 160;
// The export marches twice as densely and sums its samples by blending, so N samples make an exact mean.
export const EXPORT_STEP_CELLS = 0.75;
const EXPORT_MAX_STEPS = 320;
const ADD: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };

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

export type ExportSample = {
  camera: Camera;
  preset: Preset;
  time: number;
  // Which scalar volume to read: the current state or the loop blend.
  source: number;
  jitter: [number, number];
  frame: number;
  weight: number;
  frameTime: number;
  first: boolean;
};

export type Renderer = ReturnType<typeof createRenderer>;
export type ExportTargets = ReturnType<Renderer['createExportTargets']>;

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
  const volumeEntries = (j: number): GPUBindGroupEntry[] => [
    { binding: 0, resource: { buffer: viewBuffer } },
    { binding: 1, resource: volumes.scalars[j].createView() },
    { binding: 2, resource: volumes.shadow.createView() },
    { binding: 3, resource: volumes.sampler },
    { binding: 4, resource: { buffer: volumes.params } },
    { binding: 5, resource: volumes.light.createView() },
    { binding: 6, resource: volumes.noise.values.createView() },
    { binding: 7, resource: volumes.noise.repeat },
    { binding: 8, resource: volumes.flows[j][0].createView() },
    { binding: 9, resource: volumes.flows[j][1].createView() },
    { binding: 10, resource: volumes.velocity.createView() },
  ];
  const raymarchGroups = volumes.scalars.map((_, j) => device.createBindGroup({ layout: raymarch.getBindGroupLayout(0), entries: volumeEntries(j) }));
  const sparks = createSparkRenderer(device, viewBuffer, volumes, TARGET_FORMAT);

  // Export pipelines and bind groups are made on the first export, so the landing page's hero pays nothing for them.
  let exportPipelines: { march: GPURenderPipeline; display: GPURenderPipeline; groups: GPUBindGroup[] } | null = null;
  function exportResources() {
    if (exportPipelines) return exportPipelines;
    const march = pipeline(renderModule, 'fsExport', Array.from({ length: 4 }, () => ({ format: TARGET_FORMAT, blend: { color: ADD, alpha: ADD } })));
    const groups = volumes.scalars.map((_, j) => device.createBindGroup({ layout: march.getBindGroupLayout(0), entries: volumeEntries(j) }));
    exportPipelines = { march, display: fullscreen(DISPLAY, TARGET_FORMAT), groups };
    return exportPipelines;
  }

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

  const fullscreenPass = (encoder: GPUCommandEncoder, view: GPUTextureView, pipeline: GPURenderPipeline, group: GPUBindGroup, timestampWrites?: GPURenderPassTimestampWrites, then?: (pass: GPURenderPassEncoder) => void) => {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      timestampWrites,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    then?.(pass);
    pass.end();
  };

  function draw(encoder: GPUCommandEncoder, canvasView: GPUTextureView, frame: Frame) {
    const { current, history, resolveGroups, compositeGroups } = ensureTargets(frame.width, frame.height);
    writeView({ ...frame, jitter: [0, 0], frame: frame.index, stepCells: LIVE_STEP_CELLS, maxSteps: LIVE_MAX_STEPS, sampleWeight: 1, frameTime: 0 });
    device.queue.writeBuffer(blendBuffer, 0, new Float32Array([historyValid ? frame.blend : 1]));
    const next = 1 - historyIndex;
    fullscreenPass(encoder, current.createView(), raymarch, raymarchGroups[frame.scalarIndex], frame.timestamps.render);
    // Sparks go on after the temporal blend: it would dim a streak that moves every frame, and the next resolve
    // clamps them out of the history, so they never ghost.
    fullscreenPass(encoder, history[next].createView(), resolve, resolveGroups[historyIndex], frame.timestamps.taa, (pass) => sparks.draw(pass, frame.scalarIndex));
    fullscreenPass(encoder, canvasView, composite, compositeGroups[next], frame.timestamps.composite);
    historyIndex = next;
    historyValid = true;
  }

  // The light groups and the data target, the tonemapped display frame, and the groups that read them.
  function createExportTargets(width: number, height: number) {
    const { display: displayPipeline } = exportResources();
    const make = () =>
      device.createTexture({ size: [width, height], format: TARGET_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const layers = { fire: make(), firelight: make(), smoke: make(), data: make() };
    const display = make();
    return {
      width,
      height,
      layers,
      display,
      displayGroup: device.createBindGroup({
        layout: displayPipeline.getBindGroupLayout(0),
        entries: [layers.fire, layers.firelight, layers.smoke].map((layer, binding) => ({ binding, resource: layer.createView() })),
      }),
      previewGroup: device.createBindGroup({
        layout: composite.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: display.createView() },
          { binding: 1, resource: volumes.sampler },
        ],
      }),
      destroy() {
        for (const texture of [...Object.values(layers), display]) texture.destroy();
      },
    };
  }

  // One sample of an export frame, in its own submit: the view uniform is written before the queue runs, and a
  // short command buffer stays clear of the GPU watchdog even at 4K.
  function drawExportSample(targets: ExportTargets, sample: ExportSample) {
    const { march, groups } = exportResources();
    const { width, height } = targets;
    writeView({ ...sample, width, height, stepCells: EXPORT_STEP_CELLS, maxSteps: EXPORT_MAX_STEPS, sampleWeight: sample.weight });
    const loadOp: GPULoadOp = sample.first ? 'clear' : 'load';
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: Object.values(targets.layers).map((layer) => ({ view: layer.createView(), loadOp, storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } })),
    });
    pass.setPipeline(march);
    pass.setBindGroup(0, groups[sample.source]);
    pass.draw(3);
    sparks.drawExport(pass, sample.source);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function drawDisplay(encoder: GPUCommandEncoder, targets: ExportTargets) {
    fullscreenPass(encoder, targets.display.createView(), exportResources().display, targets.displayGroup);
  }

  // The finished frame letterboxed on the canvas, so the viewport shows the export as it renders.
  function drawPreview(encoder: GPUCommandEncoder, canvasView: GPUTextureView, [canvasWidth, canvasHeight]: [number, number], targets: ExportTargets) {
    const scale = Math.min(canvasWidth / targets.width, canvasHeight / targets.height);
    const width = Math.floor(targets.width * scale);
    const height = Math.floor(targets.height * scale);
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: canvasView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
    pass.setViewport((canvasWidth - width) / 2, (canvasHeight - height) / 2, width, height, 0, 1);
    pass.setPipeline(composite);
    pass.setBindGroup(0, targets.previewGroup);
    pass.draw(3);
    pass.end();
  }

  return {
    draw,
    createExportTargets,
    drawExportSample,
    drawDisplay,
    drawPreview,
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
export const unit = (a: Vec3): Vec3 => {
  const length = Math.hypot(...a);
  return [a[0] / length, a[1] / length, a[2] / length];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
