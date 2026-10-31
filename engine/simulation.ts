import { bindResources } from './gpu';
import { createNoiseTextures, type NoiseTextures } from './noise-textures';
import { createPressureSolver } from './pressure';
import { type Box, DENSITY, GRID, LIGHT, lightBlurDispatches, MAX_BOXES, MODE_INDEX, type Preset, type Quality, type Tweaks } from './scene';
import { CLEAR, EMISSION, HEADER, LIGHT_BLUR, MIX, SHADERS } from './simulation-shaders';
import { createSparkSimulation } from './sparks';

const PARAM_FLOATS = 64;
// Index of the loop blend among the scalar and flow volumes; the solver never reads it, only lighting and rendering.
export const LOOP_SOURCE = 2;

export type Volumes = {
  // The two states the step ping-pongs between, then the loop blend of the current state toward a kept one.
  scalars: GPUTexture[];
  // Per scalar index, the two displacement fields the flame detail rides on.
  flows: [GPUTexture, GPUTexture][];
  velocity: GPUTexture;
  shadow: GPUTexture;
  light: GPUTexture;
  params: GPUBuffer;
  sampler: GPUSampler;
  noise: NoiseTextures;
  // The spark particles the render draws over the raymarch.
  sparks: GPUBuffer;
};

export type StepParams = { preset: Preset; tweaks: Tweaks; boxes: Box[]; time: number; dt: number; emitting: boolean; blast: number };

export type Simulation = ReturnType<typeof createSimulation>;

export function createSimulation(device: GPUDevice, quality: Quality) {
  const grid = quality.grid;
  // Cells per base-grid cell: preset values in cells are scaled by it, so a finer grid runs the same fire.
  const k = grid[0] / GRID[0];
  const half = (n: number) => Math.ceil(n / 2);
  const flowGrid = grid.map(half) as [number, number, number];
  const shadowGrid = flowGrid;
  const lightGrid = grid.map((n) => Math.ceil(n / quality.lightScale)) as [number, number, number];
  const blurDispatches = lightBlurDispatches(quality);
  // The residual readback skips the sponge under the lid, where the open top is only approximate.
  const liveHeight = Math.floor(grid[1] * 0.8);
  const rowBytes = Math.ceil((grid[0] * 4) / 256) * 256;
  const volume = (format: GPUTextureFormat, size: readonly number[] = grid, usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING) =>
    device.createTexture({ size: [...size], dimension: '3d', format, usage });
  const copyable = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC;
  // Velocity on the cell faces: the state, then within a step the advected field and the field with forces added.
  const velocity = volume('rgba16float');
  const advected = volume('rgba16float');
  const forced = volume('rgba16float');
  // A spare volume: the MacCormack prediction, then the cell-centred forces.
  const scratch = volume('rgba16float');
  const states = [0, 1, LOOP_SOURCE];
  const scalars = states.map(() => volume('rgba16float', grid, copyable));
  // The displacement fields are smooth, so half the grid carries them at an eighth of the fetch footprint.
  const flows: [GPUTexture, GPUTexture][] = states.map(() => [volume('rgba16float', flowGrid, copyable), volume('rgba16float', flowGrid, copyable)]);
  const curl = volume('rgba16float');
  const divergence = volume('r32float', grid, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);
  const shadow = volume('rgba16float', shadowGrid);
  const light = [volume('rgba16float', lightGrid), volume('rgba16float', lightGrid)];
  // The smoke in each light cell, which the blur lets the flame's light through or not.
  const thickness = volume('r32float', lightGrid);
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
  const blend = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const staging = device.createBuffer({ size: rowBytes * grid[1] * grid[2], usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const pressure = createPressureSolver(device, paramSlots[0], divergence, grid);
  const noise = createNoiseTextures(device);
  const sparks = createSparkSimulation(device, { paramSlots, velocity, scalars, sampler });

  const compute = (code: string, constants?: Record<string, number>) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants } });
  const pipelines = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, compute(HEADER + body)])) as Record<
    keyof typeof SHADERS,
    GPUComputePipeline
  >;
  const clearHalf = compute(CLEAR('rgba16float'));
  const clearFloat = compute(CLEAR('r32float'));
  const emission = compute(EMISSION(quality.lightScale));
  // Light crossing a light cell of smoke is absorbed by this much per unit of smoke, half of a straight march's
  // worth, since the blur stands in for scattering around as much as through.
  const absorb = (0.5 * DENSITY * quality.lightScale) / grid[0];
  const lightBlur = [compute(LIGHT_BLUR, { dilate: 0, absorb }), compute(LIGHT_BLUR, { dilate: 1, absorb })];
  const mix = compute(MIX);

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
    shadow: states.map((j) => bind(pipelines.shadow, [paramSlots[0], scalars[j], sampler, shadow])),
    emission: states.map((j) => bind(emission, [scalars[j], light[0], thickness])),
    blur: lightBlur.map((pipeline) => both((i) => bind(pipeline, [light[i], light[1 - i], thickness]))),
  };
  const clearGroups = [
    ...[velocity, ...scalars, ...flows.flat()].map((t) => [clearHalf, bind(clearHalf, [t])] as const),
    [clearFloat, bind(clearFloat, [pressure.texture])] as const,
  ];

  const workgroups = grid.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const flowWorkgroups = flowGrid.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const shadowWorkgroups = shadowGrid.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const lightWorkgroups = lightGrid.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const run = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, size = workgroups) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...size);
  };

  // Which scalar texture holds the latest state; the pair ping-pongs every step.
  let sc = 0;
  const paramData = new Float32Array(PARAM_FLOATS);

  // Positions, sizes, speeds and accelerations given in base cells become this grid's cells; rates per second,
  // per-cell concentrations and the blast's divergence are the same on any grid.
  function writeParams(slot: number, { preset, tweaks, boxes, time, dt, emitting, blast }: StepParams) {
    const boxData = Array.from({ length: MAX_BOXES }, (_, i) => {
      const box = boxes[i];
      return box ? [box.x * k, box.y * k, box.z * k, box.yaw, box.hx * k, box.hy * k, box.hz * k, 1] : [0, 0, 0, 0, 0, 0, 0, 0];
    }).flat();
    const wind = tweaks.wind * k;
    paramData.set([
      ...grid, dt,
      time, preset.buoyancy * tweaks.heat * k, preset.cooling, tweaks.vorticity * k,
      wind * Math.cos((tweaks.windYaw * Math.PI) / 180), 0, wind * Math.sin((tweaks.windYaw * Math.PI) / 180), preset.burn,
      ...preset.emitter.map((c) => c * k), preset.radius * k,
      emitting ? preset.emit : 0, preset.smokeYield * tweaks.smoke, preset.heat * tweaks.heat, preset.dissipation,
      ...LIGHT, MODE_INDEX[preset.mode],
      preset.turbulence * k, preset.damping, preset.expansion, blast,
      preset.ignite, tweaks.seed, tweaks.width, 0,
      ...boxData,
    ]);
    device.queue.writeBuffer(paramSlots[slot], 0, paramData);
    sparks.writeParams(slot, tweaks.sparks);
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
    sparks.step(pass, slot, sc);
  }

  // The shadow volume and the flame's light volume, both read by the render; the blur ping-pongs an even number
  // of times and ends in light[0]. The first two blurs also dilate the occupancy by two coarse cells, the margin
  // the raymarch's empty-space skip needs.
  function updateLighting(pass: GPUComputePassEncoder, source = sc) {
    run(pass, pipelines.shadow, lighting.shadow[source], shadowWorkgroups);
    run(pass, emission, lighting.emission[source], lightWorkgroups);
    for (let i = 0; i < blurDispatches; i++) {
      const dilate = i < 2 ? 1 : 0;
      run(pass, lightBlur[dilate], lighting.blur[dilate][i % 2], lightWorkgroups);
    }
  }

  // Frames kept for the loop: the state and its flow fields, copied whole.
  const kept: { scalars: GPUTexture; flows: GPUTexture[] }[] = [];
  const keepable = (size: readonly number[]) =>
    device.createTexture({ size: [...size], dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });

  function keepFrame(i: number) {
    kept[i] ??= { scalars: keepable(grid), flows: [keepable(flowGrid), keepable(flowGrid)] };
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture({ texture: scalars[sc] }, { texture: kept[i].scalars }, [...grid]);
    flows[sc].forEach((flow, f) => encoder.copyTextureToTexture({ texture: flow }, { texture: kept[i].flows[f] }, [...flowGrid]));
    device.queue.submit([encoder.finish()]);
  }

  // Blends the current state toward kept frame i into the loop volumes. Its own submit: the weight is written
  // before the queue runs, so two blends in one command buffer would both see the second weight.
  function blendLoop(i: number, weight: number) {
    device.queue.writeBuffer(blend, 0, new Float32Array([weight]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    run(pass, mix, bind(mix, [blend, scalars[sc], kept[i].scalars, scalars[LOOP_SOURCE]]));
    flows[sc].forEach((flow, f) => run(pass, mix, bind(mix, [blend, flow, kept[i].flows[f], flows[LOOP_SOURCE][f]]), flowWorkgroups));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function releaseKept() {
    for (const frame of kept) [frame.scalars, ...frame.flows].forEach((texture) => texture.destroy());
    kept.length = 0;
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
    encoder.copyTextureToBuffer({ texture: divergence }, { buffer: staging, bytesPerRow: rowBytes, rowsPerImage: grid[1] }, [...grid]);
    device.queue.submit([encoder.finish()]);
    return staging.mapAsync(GPUMapMode.READ).then(() => {
      const values = new Float32Array(staging.getMappedRange());
      const stride = rowBytes / 4;
      let worst = 0;
      for (let z = 0; z < grid[2]; z++) {
        for (let y = 0; y < liveHeight; y++) {
          const row = (z * grid[1] + y) * stride;
          for (let x = 0; x < grid[0]; x++) worst = Math.max(worst, Math.abs(values[row + x]));
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
    sparks.clear();
  }

  function dispose() {
    releaseKept();
    for (const texture of [velocity, advected, forced, scratch, ...scalars, ...flows.flat(), curl, divergence, shadow, ...light, thickness]) texture.destroy();
    noise.values.destroy();
    noise.curl.destroy();
    pressure.dispose();
    sparks.dispose();
    for (const buffer of [...paramSlots, blend, staging]) buffer.destroy();
  }

  const volumes: Volumes = { scalars, flows, velocity, shadow, light: light[0], params: paramSlots[0], sampler, noise, sparks: sparks.buffer };
  return { grid, volumes, writeParams, step, updateLighting, keepFrame, blendLoop, releaseKept, readResidual, clear, dispose, get scalarIndex() { return sc; } };
}
