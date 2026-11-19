// Invoked only by the development lab. It tests the real GPU solver without a renderer or image metrics.
import { requestGpu } from './gpu';
import { defaultTweaks, DT, QUALITIES, type Box, type Preset, type Tweaks } from './scene';
import { createSimulation } from './simulation';
import explosion from '../presets/explosion.json';

const READ_VOLUME = /* wgsl */ `
@group(0) @binding(0) var source: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> values: array<vec4<f32>>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) c: vec3<u32>) {
  let n = textureDimensions(source);
  if (any(c >= n)) { return; }
  values[(c.z * n.y + c.y) * n.x + c.x] = textureLoad(source, vec3<i32>(c), 0);
}
`;

const SEED_PASSIVE_VOLUME = /* wgsl */ `
struct Seed { wind: vec3<f32>, padding: f32 }
@group(0) @binding(0) var<uniform> S: Seed;
@group(0) @binding(1) var velocity: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var scalars: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) c: vec3<u32>) {
  let n = textureDimensions(velocity);
  if (any(c >= n)) { return; }
  let centre = vec3<f32>(f32(n.x) * 0.5, f32(n.y) * 0.42, f32(n.z) * 0.5);
  let d = (vec3<f32>(c) - centre) / 6.0;
  // Start with resting storage, as a real clear does: the solver must initialise the ambient field.
  textureStore(velocity, vec3<i32>(c), vec4<f32>(vec3<f32>(0.0), S.padding));
  // An inert fuel tracer: it neither weighs down the flow nor ignites when the finite source is off.
  textureStore(scalars, vec3<i32>(c), vec4<f32>(0.0, 0.0, exp(-dot(d, d)), 0.0));
}
`;

export async function runFlowDiagnostics(signal?: AbortSignal) {
  signal?.throwIfAborted();
  const { device, adapterName } = await requestGpu();
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  device.pushErrorScope('validation');
  const simulation = createSimulation(device, QUALITIES.fast);
  const [nx, ny, nz] = simulation.grid;
  const bytes = nx * ny * nz * 16;
  const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: READ_VOLUME }), entryPoint: 'main' } });
  async function read(texture: GPUTexture) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: output } },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    return result;
  }
  const seedParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const seedPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: SEED_PASSIVE_VOLUME }), entryPoint: 'main' } });
  function seedMovingInvariant(wind: number) {
    device.queue.writeBuffer(seedParams, 0, new Float32Array([wind, 0, 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(seedPipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: seedPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: seedParams } }, { binding: 1, resource: simulation.volumes.velocity.createView() },
      { binding: 2, resource: simulation.volumes.scalars[simulation.scalarIndex].createView() },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
    pass.end(); device.queue.submit([encoder.finish()]);
  }
  const quiet: Preset = { ...explosion, mode: 'fire', emit: 0, burn: 0, buoyancy: 0, turbulence: 0, expansion: 0, damping: 1 } as Preset;
  const cases: { name: string; wind: number; boxes: Box[]; stress?: boolean; moving?: boolean }[] = [
    { name: 'Positive through-flow', wind: 60, boxes: [] },
    { name: 'Negative through-flow', wind: -60, boxes: [] },
    { name: 'Flow around solid', wind: 60, boxes: [{ x: 32, y: 20, z: 32, hx: 7, hy: 7, hz: 7, yaw: 0 }] },
    { name: 'Moving explosion window positive wind', wind: 60, boxes: [], moving: true },
    { name: 'Moving explosion window negative wind', wind: -60, boxes: [], moving: true },
    { name: 'Maximum motion and source controls', wind: 60, boxes: [], stress: true },
  ];
  const reports = [];
  try {
    for (const test of cases) {
      signal?.throwIfAborted();
      const preset: Preset = test.stress ? explosion as Preset : { ...quiet, mode: test.moving ? 'explosion' : test.boxes.length ? 'objects' : 'fire', dissipation: test.moving ? 0 : quiet.dissipation };
      const tweaks: Tweaks = { ...defaultTweaks(preset), wind: test.wind, windYaw: 0, windGust: 0, vorticity: 0, turbulence: 0, disturbance: 0, sparks: 0 };
      if (test.stress) Object.assign(tweaks, { heat: 2, fuel: 2.5, width: 2.5, seed: 59, burnRate: 3, cooling: 0.2, buoyancy: 2.5, smoke: 2.5, vorticity: 14, turbulence: 3, disturbance: 3, expansion: 2, windGust: 1 });
      simulation.clear();
      const resetOffset = simulation.offset;
      if (test.moving) seedMovingInvariant(test.wind);
      const steps = test.stress ? 120 : 60;
      for (let step = 0; step < steps; step++) {
        const time = step * DT;
        simulation.writeParams(0, { preset, tweaks, boxes: test.boxes, time, dt: DT, emitting: !test.stress || time < (preset.burst ?? 0), blast: test.stress ? (preset.blast ?? 0) * Math.exp(-time / 0.08) : 0 });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        simulation.step(pass, 0);
        pass.end();
        device.queue.submit([encoder.finish()]);
        // Keep the queue bounded and yield to the lab UI while running the actual fluid steps.
        if (step % 12 === 11) {
          await device.queue.onSubmittedWorkDone();
          signal?.throwIfAborted();
        }
      }
      signal?.throwIfAborted();
      const velocity = await read(simulation.volumes.velocity);
      const scalars = await read(simulation.volumes.scalars[simulation.scalarIndex]);
      let finite = true, meanX = 0, meanY = 0, meanZ = 0, count = 0, maxSpeed = 0, maxInsideSpeed = 0, crossRms = 0;
      let peakHeat = 0, sootMass = 0, tracerMass = 0, tracerX = 0, tracerY = 0, tracerZ = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const at = ((z * ny + y) * nx + x) * 4;
        const vx = velocity[at], vy = velocity[at + 1], vz = velocity[at + 2];
        for (let c = 0; c < 4; c++) finite &&= Number.isFinite(velocity[at + c]) && Number.isFinite(scalars[at + c]);
        const speed = Math.hypot(vx, vy, vz);
        maxSpeed = Math.max(maxSpeed, speed);
        peakHeat = Math.max(peakHeat, scalars[at + 1]);
        sootMass += scalars[at];
        const tracer = scalars[at + 2];
        tracerMass += tracer; tracerX += tracer * x; tracerY += tracer * y; tracerZ += tracer * z;
        const inSolid = test.boxes.some((box) => Math.abs(x - box.x) < box.hx - 1 && Math.abs(y - box.y) < box.hy - 1 && Math.abs(z - box.z) < box.hz - 1);
        if (inSolid) maxInsideSpeed = Math.max(maxInsideSpeed, speed);
        else if (x > 5 && x < nx - 6 && y > 3 && y < ny * 0.75 && z > 5 && z < nz - 6) {
          meanX += vx; meanY += vy; meanZ += vz; crossRms += vy * vy + vz * vz; count++;
        }
      }
      const residual = await simulation.readResidual();
      const round = (n: number) => Number(n.toFixed(4));
      const flowMatches = test.stress || Math.sign(meanX) === Math.sign(test.wind) && Math.abs(meanX / count) > Math.abs(test.wind) * 0.7;
      const offset = simulation.offset;
      const expectedOffset = test.moving ? [test.wind * steps * DT / nx, 0, 0] : test.stress ? null : [0, 0, 0];
      const tracerCentroid = tracerMass > 0 ? [tracerX / tracerMass, tracerY / tracerMass, tracerZ / tracerMass] : null;
      const movingInvariant = !test.moving || Boolean(tracerCentroid && Math.abs(tracerCentroid[0] - nx * 0.5) < 0.01 && Math.abs(tracerCentroid[1] - ny * 0.42) < 0.02 && Math.abs(tracerCentroid[2] - nz * 0.5) < 0.01 && peakHeat === 0);
      const originMatches = resetOffset.every((value) => value === 0) && (!expectedOffset || offset.every((value, axis) => Math.abs(value - expectedOffset[axis]) < 1e-8));
      reports.push({ name: test.name, steps, seconds: round(steps * DT), finite, meanVelocity: [meanX, meanY, meanZ].map((n) => round(n / count)), transverseRms: round(Math.sqrt(crossRms / count)), maxSpeed: round(maxSpeed), maxInsideSpeed: round(maxInsideSpeed), peakHeat: round(peakHeat), sootMass: round(sootMass), offset: offset.map(round), expectedOffset: expectedOffset?.map(round) ?? null, tracerMass: round(tracerMass), localTracerCentroid: tracerCentroid?.map(round) ?? null, movingInvariant, originMatches, residual: residual === null ? null : round(residual), pass: finite && flowMatches && maxInsideSpeed < 0.001 && movingInvariant && originMatches });
    }
    const validation = await device.popErrorScope();
    if (validation) errors.push(validation.message);
    return { adapter: adapterName, grid: [...simulation.grid], cases: reports, validationErrors: errors, pass: reports.every((report) => report.pass) && !errors.length };
  } finally {
    simulation.dispose(); output.destroy(); readback.destroy(); seedParams.destroy(); device.destroy();
  }
}
