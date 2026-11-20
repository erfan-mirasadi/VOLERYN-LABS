import { DOMAIN_PROBE, REMAP_VOLUME, type DomainLayout, type DomainProbe } from './domain';
import { bindResources, computePipeline } from './gpu';
import { createNoiseTextures, type NoiseTextures } from './noise-textures';
import { createPressureSolver } from './pressure';
import { type Box, DENSITY, GRID, LIGHT_PATH, lightBlurDispatches, MAX_BOXES, MODE_INDEX, type Preset, type Quality, type Tweaks } from './scene';
import { CLEAR, EMISSION, HEADER, LIGHT_BLUR, MIX, SHADERS } from './simulation-shaders';
import { createSparkSimulation } from './sparks';

const PARAM_FLOATS = 80;
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
  const cellsPerUnit = quality.cellsPerUnit ?? grid[0];
  const k = cellsPerUnit / GRID[0];
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
    computePipeline(device, code, constants);
  const pipelines = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, compute(HEADER + body)])) as Record<
    keyof typeof SHADERS,
    GPUComputePipeline
  >;
  const clearHalf = compute(CLEAR('rgba16float'));
  const clearFloat = compute(CLEAR('r32float'));
  const emission = compute(EMISSION(quality.lightScale));
  // Each pass absorbs its share of the whole path's smoke, at half a straight march's rate, since the blur stands
  // in for scattering around as much as through.
  const absorb = (0.5 * DENSITY * LIGHT_PATH) / blurDispatches;
  const lightBlur = [compute(LIGHT_BLUR, { dilate: 0, absorb }), compute(LIGHT_BLUR, { dilate: 1, absorb })];
  const mix = compute(MIX);

  const bind = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture | GPUSampler)[]) => bindResources(device, pipeline, resources);
  // The step's bind groups, one set per params slot.
  const groups = paramSlots.map((params) => ({
    initializeVelocity: bind(pipelines.initializeVelocity, [params, velocity]),
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
    emission: states.map((j) => bind(emission, [paramSlots[0], scalars[j], light[0], thickness])),
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
  // World displacement of the computational box. Explosions follow ambient horizontal transport,
  // retaining their local detail instead of being sliced away at a stationary side boundary.
  const offset: [number, number, number] = [...(quality.domainOffset ?? [0, 0, 0])];
  let lastParams: StepParams | null = null;
  let needsInitialVelocity = true;
  let movingStep = false;
  const paramData = new Float32Array(PARAM_FLOATS);

  // Positions, sizes, speeds and accelerations given in base cells become this grid's cells; rates per second,
  // per-cell concentrations and the blast's divergence are the same on any grid.
  function writeParams(slot: number, stepParams: StepParams, advanceOrigin = true) {
    lastParams = stepParams;
    const { preset, tweaks, boxes, time, dt, emitting, blast } = stepParams;
    const boxData = Array.from({ length: MAX_BOXES }, (_, i) => {
      const box = boxes[i];
      return box ? [box.x * k - offset[0] * cellsPerUnit, box.y * k - offset[1] * cellsPerUnit, box.z * k - offset[2] * cellsPerUnit, box.yaw, box.hx * k, box.hy * k, box.hz * k, 1] : [0, 0, 0, 0, 0, 0, 0, 0];
    }).flat();
    // A spatially uniform, slowly varying far-field velocity stays divergence-free; the pressure solve
    // deflects it around solids. Gust is amplitude, not a new random impulse on every frame.
    const gust = 0.52 * Math.sin(time * 1.17 + tweaks.seed * 0.71) + 0.3 * Math.sin(time * 2.63 + tweaks.seed * 1.37) + 0.18 * Math.sin(time * 0.41 + 1.4);
    const wind = tweaks.wind * k * (1 + tweaks.windGust * gust);
    const windVector: [number, number, number] = [wind * Math.cos((tweaks.windYaw * Math.PI) / 180), 0, wind * Math.sin((tweaks.windYaw * Math.PI) / 180)];
    movingStep = preset.mode === 'explosion';
    const transport: [number, number, number] = movingStep ? windVector : [0, 0, 0];
    if (advanceOrigin) for (let axis = 0; axis < 3; axis++) offset[axis] += transport[axis] * dt / cellsPerUnit;
    const lightYaw = tweaks.lightYaw * Math.PI / 180;
    const lightElevation = tweaks.lightElevation * Math.PI / 180;
    const lightDirection = [Math.cos(lightYaw) * Math.cos(lightElevation), Math.sin(lightElevation), Math.sin(lightYaw) * Math.cos(lightElevation)];
    paramData.set([
      ...grid, dt,
      time, preset.buoyancy * tweaks.buoyancy * k, preset.cooling * tweaks.cooling, tweaks.vorticity * k,
      ...windVector, preset.burn * tweaks.burnRate,
      // The charge and its brief blast stay anchored in world space as the box follows the plume.
      ...preset.emitter.map((c, axis) => c * k - offset[axis] * cellsPerUnit), preset.radius * k,
      emitting ? preset.emit * tweaks.fuel : 0, preset.smokeYield * tweaks.smoke, preset.heat * tweaks.heat, preset.dissipation,
      ...lightDirection, MODE_INDEX[preset.mode],
      preset.turbulence * tweaks.turbulence * k, preset.damping, preset.expansion * tweaks.expansion, blast * tweaks.expansion,
      preset.ignite, tweaks.seed, tweaks.width, tweaks.disturbance,
      ...boxData,
      tweaks.temperature, tweaks.smokeDensity, tweaks.fireOpacity, 1,
      ...transport, cellsPerUnit,
      ...offset.map((n) => n * cellsPerUnit), 0,
    ]);
    device.queue.writeBuffer(paramSlots[slot], 0, paramData);
    sparks.writeParams(slot, tweaks.sparks, tweaks.sparkSize, tweaks.sparkLife);
  }

  // Velocity first: advect, add forces, project; then the scalars ride the divergence-free field. An export passes
  // `next`, which submits what is encoded so far and opens a fresh pass, so a render-grid step spans three short
  // command buffers instead of one that could trip a GPU watchdog.
  function step(pass: GPUComputePassEncoder, slot: number, next?: () => GPUComputePassEncoder) {
    const g = groups[slot];
    if (needsInitialVelocity) {
      if (movingStep) run(pass, pipelines.initializeVelocity, g.initializeVelocity);
      needsInitialVelocity = false;
    }
    pressure.updateSolids(pass);
    run(pass, pipelines.predictVelocity, g.predictVelocity);
    run(pass, pipelines.correctVelocity, g.correctVelocity);
    run(pass, pipelines.curl, g.curl);
    run(pass, pipelines.forces, g.forces[sc]);
    run(pass, pipelines.applyForces, g.applyForces);
    run(pass, pipelines.divergence, g.divergence[sc]);
    pass = next?.() ?? pass;
    pressure.solve(pass);
    pass = next?.() ?? pass;
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
    offset.splice(0, 3, ...(quality.domainOffset ?? [0, 0, 0]));
    needsInitialVelocity = true;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const [pipeline, group] of clearGroups) run(pass, pipeline, group);
    pass.end();
    device.queue.submit([encoder.finish()]);
    sparks.clear();
  }


  let probing = false;
  const probePipeline = compute(DOMAIN_PROBE);
  const probeBuffer = device.createBuffer({ size: 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const probeReadback = device.createBuffer({ size: 40, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const probeGroups = scalars.map((scalar) => bind(probePipeline, [scalar, velocity, probeBuffer]));
  async function probeDomain(): Promise<DomainProbe | null> {
    if (probing || !lastParams) return null;
    probing = true;
    try {
      device.queue.writeBuffer(probeBuffer, 0, new Uint32Array([...grid, 0, 0, 0, 0, 0, 0, 0]));
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      run(pass, probePipeline, probeGroups[sc]); pass.end();
      encoder.copyBufferToBuffer(probeBuffer, 0, probeReadback, 0, 40); device.queue.submit([encoder.finish()]);
      await probeReadback.mapAsync(GPUMapMode.READ);
      const ints = new Uint32Array(probeReadback.getMappedRange()).slice(); probeReadback.unmap();
      const speeds = new Float32Array(ints.buffer);
      return { min: [ints[0], ints[1], ints[2]], max: [ints[3], ints[4], ints[5]], speed: [speeds[6] / cellsPerUnit, speeds[7] / cellsPerUnit, speeds[8] / cellsPerUnit], occupied: ints[9] };
    } finally { probing = false; }
  }

  async function copyStateFrom(previous: Simulation) {
    const params = previous.lastParams;
    if (!params) { clear(); return; }
    clear();
    writeParams(0, { ...params, dt: 0 }, false);
    const remap = compute(REMAP_VOLUME);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    run(pass, pipelines.initializeVelocity, groups[0].initializeVelocity);
    const uniforms: GPUBuffer[] = [];
    const transfer = (source: GPUTexture, target: GPUTexture, kind: number, size: [number, number, number]) => {
      const uniform = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); uniforms.push(uniform);
      const wind = [paramData[8], paramData[9], paramData[10]];
      device.queue.writeBuffer(uniform, 0, new Float32Array([...previous.offset, previous.cellsPerUnit, ...offset, cellsPerUnit, ...previous.grid, kind, ...grid, 0, ...wind, 0]));
      run(pass, remap, bind(remap, [uniform, source, target, sampler]), size.map((n) => Math.ceil(n / 4)) as [number, number, number]);
    };
    transfer(previous.volumes.velocity, velocity, 1, [...grid]);
    transfer(previous.volumes.scalars[previous.scalarIndex], scalars[0], 0, [...grid]);
    for (let i = 0; i < 2; i++) transfer(previous.volumes.flows[previous.scalarIndex][i], flows[0][i], 2, flowGrid);
    pass.end(); device.queue.submit([encoder.finish()]);
    sparks.remapFrom(previous.volumes.sparks, previous.cellsPerUnit, cellsPerUnit, previous.offset, offset);
    await device.queue.onSubmittedWorkDone();
    for (const uniform of uniforms) uniform.destroy();
    sc = 0; needsInitialVelocity = false;
  }

  function dispose() {
    releaseKept();
    for (const texture of [velocity, advected, forced, scratch, ...scalars, ...flows.flat(), curl, divergence, shadow, ...light, thickness]) texture.destroy();
    noise.values.destroy();
    noise.curl.destroy();
    pressure.dispose();
    sparks.dispose();
    for (const buffer of [...paramSlots, blend, staging, probeBuffer, probeReadback]) buffer.destroy();
  }

  const volumes: Volumes = { scalars, flows, velocity, shadow, light: light[0], params: paramSlots[0], sampler, noise, sparks: sparks.buffer };
  return { grid, cellsPerUnit, volumes, copyStateFrom, probeDomain, get lastParams() { return lastParams; }, get layout(): DomainLayout { return { grid: [...grid], cellsPerUnit, offset: [...offset] }; }, writeParams, step, updateLighting, keepFrame, blendLoop, releaseKept, readResidual, clear, dispose, get scalarIndex() { return sc; }, get offset(): [number, number, number] { return [...offset]; } };
}
