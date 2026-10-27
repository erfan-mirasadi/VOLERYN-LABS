import { RENDER } from './render-shaders';
import { DENSITY, EXPOSURE, FIRE_LIGHT, GRID, type Preset } from './scene';
import type { Volumes } from './simulation';

const VIEW_FLOATS = 36;
const BOX_MIN = [-0.5, 0, (-0.5 * GRID[2]) / GRID[0]];
const BOX_MAX = [0.5, GRID[1] / GRID[0], (0.5 * GRID[2]) / GRID[0]];

export type Vec3 = [number, number, number];

export type View = {
  eye: Vec3;
  target: Vec3;
  tanHalfFov: number;
  aspect: number;
  rect: [number, number, number, number];
  // Which AOV the fragment shader writes: 0 beauty, 1 fire, 2 smoke and solids, 3 alpha, 4 depth, 5 temperature.
  layer?: number;
};

export type Renderer = ReturnType<typeof createRenderer>;

export function createRenderer(device: GPUDevice, format: GPUTextureFormat, volumes: Volumes) {
  const shader = device.createShaderModule({ code: RENDER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shader, entryPoint: 'vs' },
    fragment: { module: shader, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const viewBuffer = device.createBuffer({ size: VIEW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const groups = volumes.scalars.map((scalars) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: viewBuffer } },
        { binding: 1, resource: scalars.createView() },
        { binding: 2, resource: volumes.shadow.createView() },
        { binding: 3, resource: volumes.sampler },
        { binding: 4, resource: { buffer: volumes.params } },
        { binding: 5, resource: volumes.light.createView() },
      ],
    }),
  );
  const viewData = new Float32Array(VIEW_FLOATS);

  function writeView(view: View, preset: Preset, time: number) {
    const forward = unit(sub(view.target, view.eye));
    const right = unit(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    viewData.set([
      ...view.eye, view.tanHalfFov,
      ...forward, view.aspect,
      ...right, time,
      ...up, 1.5 / GRID[0],
      ...BOX_MIN, EXPOSURE,
      ...BOX_MAX, DENSITY,
      ...preset.tint, preset.glow,
      ...view.rect,
      view.layer ?? 0, FIRE_LIGHT, 0, 0,
    ]);
    device.queue.writeBuffer(viewBuffer, 0, viewData);
  }

  function draw(encoder: GPUCommandEncoder, target: GPUTextureView, scalarIndex: number, timestampWrites?: GPURenderPassTimestampWrites) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      timestampWrites,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, groups[scalarIndex]);
    pass.draw(3);
    pass.end();
  }

  return { writeView, draw, dispose: () => viewBuffer.destroy() };
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const unit = (a: Vec3): Vec3 => {
  const length = Math.hypot(...a);
  return [a[0] / length, a[1] / length, a[2] / length];
};
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
