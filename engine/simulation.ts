import { bindResources } from './gpu';
import { createNoiseTextures, type NoiseTextures } from './noise-textures';
import { createPressureSolver } from './pressure';
import {
  type Box,
  GRID,
  LIGHT,
  LIGHT_BLUR_PASSES,
  LIGHT_GRID,
  LIGHT_SCALE,
  MAX_BOXES,
  MODE_INDEX,
  type Preset,
  SHADOW_GRID,
  type Tweaks,
} from './scene';
import { CLEAR, EMISSION, HEADER, LIGHT_BLUR, SHADERS } from './simulation-shaders';

const PARAM_FLOATS = 64;
const FLOW_GRID = GRID.map((n) => n / 2) as [number, number, number];
// The residual readback skips the sponge under the lid, where the open top is only approximate.
const LIVE_HEIGHT = Math.floor(GRID[1] * 0.8);
const ROW_BYTES = Math.ceil((GRID[0] * 4) / 256) * 256;

export type Volumes = {
  scalars: [GPUTexture, GPUTexture];
  // Per scalar index, the two displacement fields the flame detail rides on.
  flows: [GPUTexture, GPUTexture][];
  shadow: GPUTexture;
  light: GPUTexture;
  params: GPUBuffer;
  sampler: GPUSampler;
  noise: NoiseTextures;
};

export type StepParams = { preset: Preset; tweaks: Tweaks; boxes: Box[]; time: number; dt: number; emitting: boolean; blast: number };

export type Simulation = ReturnType<typeof createSimulation>;

export function createSimulation(device: GPUDevice) {
  const volume = (format: GPUTextureFormat, size: readonly number[] = GRID, usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING) =>
    device.createTexture({ size: [...size], dimension: '3d', format, usage });
  // Velocity on the cell faces: the state, then within a step the advected field and the field with forces added.
  const velocity = volume('rgba16float');
  const advected = volume('rgba16float');
  const forced = volume('rgba16float');
  // A spare volume: the MacCormack prediction, then the cell-centred forces.
  const scratch = volume('rgba16float');
  const scalars: [GPUTexture, GPUTexture] = [volume('rgba16float'), volume('rgba16float')];
  // The displacement fields are smooth, so half the grid carries them at an eighth of the fetch footprint.
  const flows: [GPUTexture, GPUTexture][] = [0, 1].map(() => [volume('rgba16float', FLOW_GRID), volume('rgba16float', FLOW_GRID)]);
  const curl = volume('rgba16float');
  const divergence = volume('r32float', GRID, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);
  const shadow = volume('rgba16float', SHADOW_GRID);
  const light = [volume('rgba16float', LIGHT_GRID), volume('rgba16float', LIGHT_GRID)];
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });
  const both = <T>(make: (i: number) => T) => [make(0), make(1)];
  // A frame may run two steps before it submits, and writing one buffer twice would hand both steps the second
  // step's params, so each step slot has its own.
  const paramSlots = both(() => device.createBuffer({ size: PARAM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const staging = device.createBuffer({ size: ROW_BYTES * GRID[1] * GRID[2], usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const pressure = createPressureSolver(device, paramSlots[0], divergence);
  const noise = createNoiseTextures(device);

  const compute = (code: string, constants?: Record<string, number>) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants } });
  const pipelines = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, compute(HEADER + body)])) as Record<
    keyof typeof SHADERS,
    GPUComputePipeline
  >;
  const clearHalf = compute(CLEAR('rgba16float'));
  const clearFloat = compute(CLEAR('r32float'));
  const emission = compute(EMISSION(LIGHT_SCALE));
  const lightBlur = [compute(LIGHT_BLUR, { dilate: 0 }), compute(LIGHT_BLUR, { dilate: 1 })];

  const bind = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture | GPUSampler)[]) => bindResources(device, pipeline, resources);
  // The step's bind groups, one set per params slot.
  const groups = paramSlots.map((params) => ({
    predictVelocity: bind(pipelines.predictVelocity, [params, velocity, velocity, scratch, sampler]),
    correctVelocity: bind(pipelines.correctVelocity, [params, velocity, velocity, scratch, advected, sampler]),
    curl: bind(pipelines.curl, [params, advected, curl]),
    forces: both((j) => bind(pipelines.forces, [params, scalars[j], curl, scratch, noise.curl, noise.repeat])),
    applyForces: bind(pipelines.applyForces, [params, advected, scratch, forced, pressure.solid]),
    divergence: both((j) => bind(pipelines.divergence, [params, forced, divergence, scalars[j], pressure.solid])),
    project: bind(pipelines.project, [params, forced, pressure.texture, velocity, pressure.solid]),
    predict: both((j) => bind(pipelines.predict, [params, velocity, scalars[j], scratch, sampler])),
    correct: both((j) => bind(pipelines.correct, [params, velocity, scalars[j], scratch, scalars[1 - j], sampler])),
    advectFlow: both((j) => bind(pipelines.advectFlow, [params, velocity, ...flows[j], ...flows[1 - j], sampler])),
  }));
  const residual = both((j) => bind(pipelines.divergence, [paramSlots[0], velocity, divergence, scalars[j], pressure.solid]));
  const lighting = {
    shadow: both((j) => bind(pipelines.shadow, [paramSlots[0], scalars[j], sampler, shadow])),
    emission: both((j) => bind(emission, [scalars[j], light[0]])),
    blur: lightBlur.map((pipeline) => both((k) => bind(pipeline, [light[k], light[1 - k]]))),
  };
  const clearGroups = [
    ...[velocity, ...scalars, ...flows.flat()].map((t) => [clearHalf, bind(clearHalf, [t])] as const),
    [clearFloat, bind(clearFloat, [pressure.texture])] as const,
  ];

  const workgroups = GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const flowWorkgroups = FLOW_GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const shadowWorkgroups = SHADOW_GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const lightWorkgroups = LIGHT_GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const run = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, size = workgroups) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...size);
  };

  // Which scalar texture holds the latest state; the pair ping-pongs every step.
  let sc = 0;
  const paramData = new Float32Array(PARAM_FLOATS);

  function writeParams(slot: number, { preset, tweaks, boxes, time, dt, emitting, blast }: StepParams) {
    const boxData = Array.from({ length: MAX_BOXES }, (_, i) => {
      const box = boxes[i];
      return box ? [box.x, box.y, box.z, box.yaw, box.hx, box.hy, box.hz, 1] : [0, 0, 0, 0, 0, 0, 0, 0];
    }).flat();
    paramData.set([
      ...GRID, dt,
      time, preset.buoyancy * tweaks.heat, preset.cooling, tweaks.vorticity,
      tweaks.wind, 0, 0, preset.burn,
      ...preset.emitter, preset.radius,
      emitting ? preset.emit : 0, preset.smokeYield, preset.heat * tweaks.heat, preset.dissipation,
      ...LIGHT, MODE_INDEX[preset.mode],
      preset.turbulence, preset.damping, preset.expansion, blast,
      preset.ignite, tweaks.seed, 0, 0,
      ...boxData,
    ]);
    device.queue.writeBuffer(paramSlots[slot], 0, paramData);
  }

  // Velocity first: advect, add forces, project; then the scalars ride the divergence-free field.
  function step(pass: GPUComputePassEncoder, slot: number) {
    const g = groups[slot];
    pressure.updateSolids(pass);
    run(pass, pipelines.predictVelocity, g.predictVelocity);
    run(pass, pipelines.correctVelocity, g.correctVelocity);
    run(pass, pipelines.curl, g.curl);
    run(pass, pipelines.forces, g.forces[sc]);
    run(pass, pipelines.applyForces, g.applyForces);
    run(pass, pipelines.divergence, g.divergence[sc]);
    pressure.solve(pass);
    run(pass, pipelines.project, g.project);
    run(pass, pipelines.predict, g.predict[sc]);
    run(pass, pipelines.correct, g.correct[sc]);
    run(pass, pipelines.advectFlow, g.advectFlow[sc], flowWorkgroups);
    sc = 1 - sc;
  }

  // The shadow volume and the flame's light volume, both read by the render; the blur ping-pongs and ends in light[0].
  // The first two blurs also dilate the occupancy by two coarse cells, the margin the raymarch's empty-space skip needs.
  function updateLighting(pass: GPUComputePassEncoder) {
    run(pass, pipelines.shadow, lighting.shadow[sc], shadowWorkgroups);
    run(pass, emission, lighting.emission[sc], lightWorkgroups);
    for (let i = 0; i < LIGHT_BLUR_PASSES * 2; i++) {
      const dilate = i < 2 ? 1 : 0;
      run(pass, lightBlur[dilate], lighting.blur[dilate][i % 2], lightWorkgroups);
    }
  }

  let reading = false;
  // The largest divergence the projection left in the live region, in cells per second: a health check on the solver.
  function readResidual(): Promise<number> | null {
    if (reading) return null;
    reading = true;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    run(pass, pipelines.divergence, residual[1 - sc]);
    pass.end();
    encoder.copyTextureToBuffer({ texture: divergence }, { buffer: staging, bytesPerRow: ROW_BYTES, rowsPerImage: GRID[1] }, [...GRID]);
    device.queue.submit([encoder.finish()]);
    return staging.mapAsync(GPUMapMode.READ).then(() => {
      const values = new Float32Array(staging.getMappedRange());
      const stride = ROW_BYTES / 4;
      let worst = 0;
      for (let z = 0; z < GRID[2]; z++) {
        for (let y = 0; y < LIVE_HEIGHT; y++) {
          const row = (z * GRID[1] + y) * stride;
          for (let x = 0; x < GRID[0]; x++) worst = Math.max(worst, Math.abs(values[row + x]));
        }
      }
      staging.unmap();
      reading = false;
      return worst;
    });
  }

  function clear() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const [pipeline, group] of clearGroups) run(pass, pipeline, group);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function dispose() {
    for (const texture of [velocity, advected, forced, scratch, ...scalars, ...flows.flat(), curl, divergence, shadow, ...light]) texture.destroy();
    noise.values.destroy();
    noise.curl.destroy();
    pressure.dispose();
    for (const buffer of paramSlots) buffer.destroy();
    staging.destroy();
  }

  const volumes: Volumes = { scalars, flows, shadow, light: light[0], params: paramSlots[0], sampler, noise };
  return { volumes, writeParams, step, updateLighting, readResidual, clear, dispose, get scalarIndex() { return sc; } };
}
