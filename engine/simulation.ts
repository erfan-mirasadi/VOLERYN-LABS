import { bindResources } from './gpu';
import { createPressureSolver } from './pressure';
import {
  type Box,
  DT,
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

const PARAM_FLOATS = 60;
// The residual readback skips the sponge under the lid, where the open top is only approximate.
const LIVE_HEIGHT = Math.floor(GRID[1] * 0.8);
const ROW_BYTES = Math.ceil((GRID[0] * 4) / 256) * 256;

export type Volumes = {
  scalars: [GPUTexture, GPUTexture];
  shadow: GPUTexture;
  light: GPUTexture;
  params: GPUBuffer;
  sampler: GPUSampler;
};

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
  const params = device.createBuffer({ size: PARAM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const staging = device.createBuffer({ size: ROW_BYTES * GRID[1] * GRID[2], usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const pressure = createPressureSolver(device, params, divergence);

  const compute = (code: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pipelines = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, compute(HEADER + body)])) as Record<
    keyof typeof SHADERS,
    GPUComputePipeline
  >;
  const clearHalf = compute(CLEAR('rgba16float'));
  const clearFloat = compute(CLEAR('r32float'));
  const emission = compute(EMISSION(LIGHT_SCALE));
  const lightBlur = compute(LIGHT_BLUR);

  const bind = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture | GPUSampler)[]) => bindResources(device, pipeline, resources);
  const both = <T>(make: (i: number) => T) => [make(0), make(1)];
  const groups = {
    predictVelocity: bind(pipelines.predictVelocity, [params, velocity, velocity, scratch, sampler]),
    correctVelocity: bind(pipelines.correctVelocity, [params, velocity, velocity, scratch, advected, sampler]),
    curl: bind(pipelines.curl, [params, advected, curl]),
    forces: both((j) => bind(pipelines.forces, [params, scalars[j], curl, scratch])),
    applyForces: bind(pipelines.applyForces, [params, advected, scratch, forced, pressure.solid]),
    divergence: both((j) => bind(pipelines.divergence, [params, forced, divergence, scalars[j], pressure.solid])),
    residual: both((j) => bind(pipelines.divergence, [params, velocity, divergence, scalars[j], pressure.solid])),
    project: bind(pipelines.project, [params, forced, pressure.texture, velocity, pressure.solid]),
    predict: both((j) => bind(pipelines.predict, [params, velocity, scalars[j], scratch, sampler])),
    correct: both((j) => bind(pipelines.correct, [params, velocity, scalars[j], scratch, scalars[1 - j], sampler])),
    shadow: both((j) => bind(pipelines.shadow, [params, scalars[j], sampler, shadow])),
    emission: both((j) => bind(emission, [scalars[j], light[0]])),
    lightBlur: both((k) => bind(lightBlur, [light[k], light[1 - k]])),
    clear: [
      ...[velocity, ...scalars].map((t) => [clearHalf, bind(clearHalf, [t])] as const),
      [clearFloat, bind(clearFloat, [pressure.texture])] as const,
    ],
  };

  const workgroups = GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
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

  function writeParams(preset: Preset, tweaks: Tweaks, boxes: Box[], time: number, emitting: boolean) {
    const boxData = Array.from({ length: MAX_BOXES }, (_, i) => {
      const box = boxes[i];
      return box ? [box.x, box.y, box.z, box.yaw, box.hx, box.hy, box.hz, 1] : [0, 0, 0, 0, 0, 0, 0, 0];
    }).flat();
    paramData.set([
      ...GRID, DT,
      time, preset.buoyancy * tweaks.heat, preset.cooling, tweaks.vorticity,
      tweaks.wind, 0, 0, preset.burn,
      ...preset.emitter, preset.radius,
      emitting ? preset.emit : 0, preset.smokeYield, preset.heat * tweaks.heat, preset.dissipation,
      ...LIGHT, MODE_INDEX[preset.mode],
      preset.turbulence, preset.damping, preset.expansion, 0,
      ...boxData,
    ]);
    device.queue.writeBuffer(params, 0, paramData);
  }

  // Velocity first: advect, add forces, project; then the scalars ride the divergence-free field.
  function step(pass: GPUComputePassEncoder) {
    pressure.updateSolids(pass);
    run(pass, pipelines.predictVelocity, groups.predictVelocity);
    run(pass, pipelines.correctVelocity, groups.correctVelocity);
    run(pass, pipelines.curl, groups.curl);
    run(pass, pipelines.forces, groups.forces[sc]);
    run(pass, pipelines.applyForces, groups.applyForces);
    run(pass, pipelines.divergence, groups.divergence[sc]);
    pressure.solve(pass);
    run(pass, pipelines.project, groups.project);
    run(pass, pipelines.predict, groups.predict[sc]);
    run(pass, pipelines.correct, groups.correct[sc]);
    sc = 1 - sc;
  }

  // The shadow volume and the flame's light volume, both read by the render; the blur ping-pongs and ends in light[0].
  function updateLighting(pass: GPUComputePassEncoder) {
    run(pass, pipelines.shadow, groups.shadow[sc], shadowWorkgroups);
    run(pass, emission, groups.emission[sc], lightWorkgroups);
    for (let i = 0; i < LIGHT_BLUR_PASSES * 2; i++) run(pass, lightBlur, groups.lightBlur[i % 2], lightWorkgroups);
  }

  let reading = false;
  // The largest divergence the projection left in the live region, in cells per second: a health check on the solver.
  function readResidual(): Promise<number> | null {
    if (reading) return null;
    reading = true;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    run(pass, pipelines.divergence, groups.residual[1 - sc]);
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
    for (const [pipeline, group] of groups.clear) run(pass, pipeline, group);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function dispose() {
    for (const texture of [velocity, advected, forced, scratch, ...scalars, curl, divergence, shadow, ...light]) texture.destroy();
    pressure.dispose();
    params.destroy();
    staging.destroy();
  }

  const volumes: Volumes = { scalars, shadow, light: light[0], params, sampler };
  return { volumes, writeParams, step, updateLighting, readResidual, clear, dispose, get scalarIndex() { return sc; } };
}
