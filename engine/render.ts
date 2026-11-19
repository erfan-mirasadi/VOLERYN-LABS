import { type ColourModels, packColourModels } from './colour-models';
import type { DepthMap } from './depth';
import type { Plate } from './plate';
import { COMPOSITE, DISPLAY, PRESENT, PREVIEW, RENDER, RESOLVE } from './render-shaders';
import { BOX_MAX, BOX_MIN, DENSITY, EXPOSURE, FIRE_LIGHT, FLOW_PERIOD, GRID, type Preset, type Quality, type Tweaks } from './scene';
import type { Volumes } from './simulation';
import { createSparkRenderer } from './sparks';

const VIEW_FLOATS = 48;
const TARGET_FORMAT: GPUTextureFormat = 'rgba16float';
// The export marches densely, three quarters of a cell of whatever grid it renders, and sums its samples by
// blending, so N samples make an exact mean. The step is kept in base cells, like the view's.
const EXPORT_STEP_CELLS = 0.75;
const EXPORT_MAX_STEPS = 320;
export function exportMarch(grid: readonly number[]) {
  const k = grid[0] / GRID[0];
  return { stepCells: EXPORT_STEP_CELLS / k, maxSteps: EXPORT_MAX_STEPS * k };
}
const ADD: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };
const BAND_PIXELS = 1.5 * 2 ** 20;

export type Vec3 = [number, number, number];

// shift slides the picture across the frame, in half-frames to the right and up, without turning the view.
export type Camera = { eye: Vec3; target: Vec3; tanHalfFov: number; shift?: [number, number] };

export type Frame = {
  camera: Camera;
  preset: Preset;
  tweaks: Tweaks;
  // World-space origin shift of the simulation window; velocities remain in world space.
  offset: Vec3;
  blue: number;
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
  tweaks: Tweaks;
  // World-space origin shift of the simulation window; velocities remain in world space.
  offset: Vec3;
  blue: number;
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
  // Seconds this sample looks back from the frame's state, or ahead when negative: its place in the shutter.
  timeOffset: number;
};

export type ExportSample = {
  camera: Camera;
  preset: Preset;
  tweaks: Tweaks;
  // World-space origin shift of the simulation window; velocities remain in world space.
  offset: Vec3;
  blue: number;
  time: number;
  // Which scalar volume to read: the current state or the loop blend.
  source: number;
  jitter: [number, number];
  frame: number;
  weight: number;
  frameTime: number;
  timeOffset: number;
  first: boolean;
  // How much of the sparks to show, 0 to 1: a loop fades them across its seam.
  sparks: number;
};

export type Renderer = ReturnType<typeof createRenderer>;
export type ExportTargets = ReturnType<Renderer['createExportTargets']>;

export function createRenderer(device: GPUDevice, format: GPUTextureFormat, volumes: Volumes, initialQuality: Quality) {
  // The live raymarch's step and step count; a quality change without a new grid only swaps these.
  let live = { stepCells: initialQuality.stepCells, maxSteps: initialQuality.maxSteps };
  const offline = exportMarch(initialQuality.grid);
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
  const present = fullscreen(PRESENT, format);
  const composite = fullscreen(PREVIEW, format);
  // The same bilinear pass into a float target, to carry the history over to a new target size.
  const resample = fullscreen(COMPOSITE, TARGET_FORMAT);
  // The footage behind the fire, read as linear light; one clear texel until a plate is loaded.
  const noPlate = device.createTexture({ size: [1, 1], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING });
  let plate = noPlate;
  // The plate's depth, each texel its nearness and how steeply that changes, and where the fire stands in it: what is
  // nearer covers it. With no depth one texel far behind everything covers nothing.
  const noDepth = device.createTexture({ size: [1, 1], format: 'rg32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: noDepth }, new Float32Array([-1e4, 0]), { bytesPerRow: 8 }, [1, 1]);
  let depth = noDepth;
  // The frame the depth was read from, texel for texel, and the colour models of either side of the fire.
  const noColours = device.createTexture({ size: [1, 1], format: 'rgba8unorm-srgb', usage: GPUTextureUsage.TEXTURE_BINDING });
  let colours = noColours;
  const holdBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const modelsBuffer = device.createBuffer({ size: packColourModels(null).byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(modelsBuffer, 0, packColourModels(null));
  const layersBuffer = (plate: number, tint: number) => {
    const buffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, new Float32Array([plate, tint]));
    return buffer;
  };
  const presentLayers = layersBuffer(1, 0);

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

  type Targets = { width: number; height: number; current: GPUTexture; history: GPUTexture[]; resolveGroups: GPUBindGroup[]; presentGroups: GPUBindGroup[]; resampleGroups: GPUBindGroup[] };
  let targets: Targets | null = null;
  // Targets of the old size, destroyed once the frame that resampled them has been submitted.
  let retired: Targets | null = null;
  let historyIndex = 0;
  let historyValid = false;

  // Targets for this size. When the size changes, the resolved history is resampled into the new one so a quality
  // change shows the same converged frame, not the raw jittered march.
  function ensureTargets(encoder: GPUCommandEncoder, width: number, height: number) {
    if (targets && targets.width === width && targets.height === height) return targets;
    const old = targets;
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
      presentGroups: views.map((view) =>
        device.createBindGroup({
          layout: present.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: view },
            { binding: 1, resource: volumes.sampler },
            { binding: 2, resource: plate.createView() },
            { binding: 3, resource: depth.createView() },
            { binding: 4, resource: { buffer: holdBuffer } },
            { binding: 5, resource: { buffer: presentLayers } },
            { binding: 6, resource: colours.createView() },
            { binding: 7, resource: { buffer: modelsBuffer } },
          ],
        }),
      ),
      resampleGroups: views.map((view) =>
        device.createBindGroup({
          layout: resample.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: view },
            { binding: 1, resource: volumes.sampler },
          ],
        }),
      ),
    };
    if (old && historyValid) {
      fullscreenPass(encoder, views[historyIndex], resample, old.resampleGroups[historyIndex]);
      retireTargets(old);
    } else {
      if (old) retireTargets(old);
      historyValid = false;
    }
    return targets;
  }

  function retireTargets(old: Targets) {
    destroyTargets(retired);
    retired = old;
  }

  function destroyTargets(which: Targets | null) {
    if (!which) return;
    which.current.destroy();
    which.history.forEach((texture) => texture.destroy());
  }

  function disposeTargets() {
    destroyTargets(retired);
    destroyTargets(targets);
    retired = targets = null;
  }

  const viewData = new Float32Array(VIEW_FLOATS);
  function writeView({ camera, preset, tweaks, offset, blue, time, width, height, jitter, frame, stepCells, maxSteps, sampleWeight, frameTime, timeOffset }: ViewParams) {
    const { forward, right, up } = cameraBasis(camera);
    // Flow A's weight is 0 the moment it resets and 1 when B does; B gets the rest.
    const flowWeight = 1 - Math.abs(1 - 2 * ((time / FLOW_PERIOD) % 1));
    viewData.set([
      ...camera.eye, camera.tanHalfFov,
      ...forward, width / height,
      ...right, frameTime,
      ...up, stepCells / GRID[0],
      ...BOX_MIN.map((value, axis) => value + offset[axis]), EXPOSURE * 2 ** tweaks.exposure,
      ...BOX_MAX.map((value, axis) => value + offset[axis]), DENSITY * tweaks.smokeDensity,
      ...preset.tint, preset.glow,
      -jitter[0], -jitter[1], width, height,
      maxSteps, FIRE_LIGHT, frame, preset.fireDensity * tweaks.fireOpacity,
      preset.detail * tweaks.detail, flowWeight, sampleWeight, blue,
      timeOffset, 0, ...(camera.shift ?? [0, 0]),
      tweaks.temperature, tweaks.ambient, tweaks.sparkSize, tweaks.keyLight,
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
    // The previous frame's command buffer, which may have read the retired targets, has been submitted by now.
    destroyTargets(retired);
    retired = null;
    const { current, history, resolveGroups, presentGroups } = ensureTargets(encoder, frame.width, frame.height);
    writeView({ ...frame, jitter: [0, 0], frame: frame.index, stepCells: live.stepCells, maxSteps: live.maxSteps, sampleWeight: 1, frameTime: 0, timeOffset: 0 });
    device.queue.writeBuffer(blendBuffer, 0, new Float32Array([historyValid ? frame.blend : 1]));
    const next = 1 - historyIndex;
    fullscreenPass(encoder, current.createView(), raymarch, raymarchGroups[frame.scalarIndex], frame.timestamps.render);
    // Sparks go on after the temporal blend: it would dim a streak that moves every frame, and the next resolve
    // clamps them out of the history, so they never ghost.
    fullscreenPass(encoder, history[next].createView(), resolve, resolveGroups[historyIndex], frame.timestamps.taa, (pass) => sparks.draw(pass, frame.scalarIndex));
    fullscreenPass(encoder, canvasView, present, presentGroups[next], frame.timestamps.composite);
    historyIndex = next;
    historyValid = true;
  }

  // A new plate, or none: the live targets go with the old one, since their bind groups hold it.
  function setPlate(next: Plate | null) {
    if (plate !== noPlate) plate.destroy();
    plate = noPlate;
    if (next) {
      plate = device.createTexture({
        size: [next.width, next.height],
        format: 'rgba8unorm-srgb',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      writePlate(next.source);
    }
    disposeTargets();
    historyValid = false;
  }

  // The plate's current frame; a video is copied again whenever it has moved on.
  function writePlate(source: Plate['source']) {
    device.queue.copyExternalImageToTexture({ source }, { texture: plate }, [plate.width, plate.height]);
  }

  // The depth of the plate's footage, or none, and where the fire stands in it: whatever is nearer than `nearness`
  // covers the fire. A map of another size takes a new texture, and the live targets go with the old one.
  function setDepth(map: DepthMap | null, nearness: number, reach: number, models: ColourModels | null) {
    const fits = map ? depth !== noDepth && depth.width === map.width && depth.height === map.height : depth === noDepth;
    if (!fits) {
      if (depth !== noDepth) depth.destroy();
      if (colours !== noColours) colours.destroy();
      const make = (format: GPUTextureFormat) => device.createTexture({ size: [map!.width, map!.height], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      depth = map ? make('rg32float') : noDepth;
      colours = map ? make('rgba8unorm-srgb') : noColours;
      disposeTargets();
      historyValid = false;
    }
    if (map) {
      device.queue.writeTexture({ texture: depth }, withSlope(map), { bytesPerRow: 8 * map.width }, [map.width, map.height]);
      device.queue.writeTexture({ texture: colours }, map.colour, { bytesPerRow: 4 * map.width }, [map.width, map.height]);
    }
    setHoldout(nearness, reach, models);
  }

  // `reach` is how far across the frame, as a share of its long side, the cover's soft edge may spread.
  function setHoldout(nearness: number, reach: number, models: ColourModels | null) {
    device.queue.writeBuffer(holdBuffer, 0, new Float32Array([nearness, reach]));
    device.queue.writeBuffer(modelsBuffer, 0, packColourModels(models));
  }

  function markCover(on: boolean) {
    device.queue.writeBuffer(presentLayers, 0, new Float32Array([1, on ? 1 : 0]));
  }

  // The light groups and the data target, the tonemapped display frame, and the preview's group. The display frame
  // goes over the footage plate when the export composites it, and behind the plate's foreground when it holds the
  // fire out.
  function createExportTargets(width: number, height: number, { overPlate, holdout }: { overPlate: boolean; holdout: boolean }) {
    const make = () =>
      device.createTexture({ size: [width, height], format: TARGET_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    const layers = { fire: make(), firelight: make(), smoke: make(), data: make() };
    const display = make();
    const displayLayers = layersBuffer(overPlate ? 1 : 0, 0);
    return {
      width,
      height,
      layers,
      display,
      displayLayers,
      overPlate,
      holdout,
      previewGroup: device.createBindGroup({
        layout: composite.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: display.createView() },
          { binding: 1, resource: volumes.sampler },
        ],
      }),
      destroy() {
        for (const texture of [...Object.values(layers), display]) texture.destroy();
        displayLayers.destroy();
      },
    };
  }

  // One sample of an export frame, marched in bands of about BAND_PIXELS, each its own submit: the view uniform is
  // written before the queue runs, and no command buffer runs long enough to trip a GPU watchdog, even at 4K on
  // the render grid. The first band of a frame's first sample clears the layers; the sparks go on with the last.
  function drawExportSample(targets: ExportTargets, sample: ExportSample) {
    const { march, groups } = exportResources();
    const { width, height } = targets;
    writeView({ ...sample, width, height, ...offline, sampleWeight: sample.weight });
    const bands = Math.ceil((width * height) / BAND_PIXELS);
    const rows = Math.ceil(height / bands);
    for (let b = 0; b < bands; b++) {
      const y = b * rows;
      const h = Math.min(rows, height - y);
      if (h <= 0) break;
      const loadOp: GPULoadOp = sample.first && b === 0 ? 'clear' : 'load';
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: Object.values(targets.layers).map((layer) => ({ view: layer.createView(), loadOp, storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } })),
      });
      pass.setPipeline(march);
      pass.setBindGroup(0, groups[sample.source]);
      pass.setScissorRect(0, y, width, h);
      pass.draw(3);
      if (y + h >= height) {
        pass.setScissorRect(0, 0, width, height);
        sparks.drawExport(pass, sample.source, sample.sparks);
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
  }

  // The group is made for each frame: a video's frames bring depth maps of their own, which may take a new texture.
  function drawDisplay(encoder: GPUCommandEncoder, { layers, display, displayLayers, overPlate, holdout }: ExportTargets) {
    const { display: displayPipeline } = exportResources();
    const group = device.createBindGroup({
      layout: displayPipeline.getBindGroupLayout(0),
      entries: [
        ...[layers.fire, layers.firelight, layers.smoke, overPlate ? plate : noPlate].map((texture, binding) => ({ binding, resource: texture.createView() })),
        { binding: 4, resource: volumes.sampler },
        { binding: 5, resource: (holdout ? depth : noDepth).createView() },
        { binding: 6, resource: { buffer: holdBuffer } },
        { binding: 7, resource: { buffer: displayLayers } },
        { binding: 8, resource: (holdout ? colours : noColours).createView() },
        { binding: 9, resource: { buffer: modelsBuffer } },
      ],
    });
    fullscreenPass(encoder, display.createView(), displayPipeline, group);
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
    setPlate,
    writePlate,
    setDepth,
    setHoldout,
    markCover,
    createExportTargets,
    drawExportSample,
    drawDisplay,
    drawPreview,
    resetHistory: () => {
      historyValid = false;
    },
    setQuality(quality: Quality) {
      live = { stepCells: quality.stepCells, maxSteps: quality.maxSteps };
    },
    dispose() {
      disposeTargets();
      sparks.dispose();
      if (plate !== noPlate) plate.destroy();
      noPlate.destroy();
      if (depth !== noDepth) depth.destroy();
      if (colours !== noColours) colours.destroy();
      noDepth.destroy();
      noColours.destroy();
      holdBuffer.destroy();
      modelsBuffer.destroy();
      presentLayers.destroy();
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

// Each texel's nearness beside how much it changes per texel, from its neighbours on either side: the cover's edge
// is soft where the depth slopes and sharp where it jumps.
function withSlope({ width, height, data }: DepthMap) {
  const texels = new Float32Array(2 * width * height);
  const at = (x: number, y: number) => data[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      texels[2 * i] = data[i];
      texels[2 * i + 1] = 0.5 * Math.hypot(at(x + 1, y) - at(x - 1, y), at(x, y + 1) - at(x, y - 1));
    }
  }
  return texels;
}
