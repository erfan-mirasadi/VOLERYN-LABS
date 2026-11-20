// Development-only tests of the actual GPU fields and adaptive remapping, with no image-based substitutes.
import { DOMAIN_PROBE, planDomain } from './domain';
import { computePipeline, requestGpu } from './gpu';
import { BOX_MIN, defaultTweaks, DT, QUALITIES, type Preset, type Quality } from './scene';
import { createSimulation, type Simulation } from './simulation';
import fire from '../presets/fire.json';

const READ = /* wgsl */ `
@group(0) @binding(0) var src:texture_3d<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<vec4<f32>>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id)c:vec3<u32>){let n=textureDimensions(src);if(any(c>=n)){return;}out[(c.z*n.y+c.y)*n.x+c.x]=textureLoad(src,vec3<i32>(c),0);}`;
const SPARK = 'struct Spark {pos:vec3<f32>,age:f32,vel:vec3<f32>,life:f32}';
const SEED = SPARK + /* wgsl */ `
@group(0) @binding(0) var velocity:texture_storage_3d<rgba16float,write>;
@group(0) @binding(1) var scalar:texture_storage_3d<rgba16float,write>;
@group(0) @binding(2) var flowA:texture_storage_3d<rgba16float,write>;
@group(0) @binding(3) var flowB:texture_storage_3d<rgba16float,write>;
@group(0) @binding(4) var<storage,read_write> sparks:array<Spark>;
@compute @workgroup_size(4,4,4) fn main(@builtin(global_invocation_id)c:vec3<u32>){
 let n=textureDimensions(scalar);if(any(c>=n)){return;}
 let d=(vec3<f32>(c)-vec3<f32>(48.0,40.0,32.0))/6.0;
 textureStore(scalar,vec3<i32>(c),vec4<f32>(0.0,0.0,exp(-dot(d,d)),0.0));
 textureStore(velocity,vec3<i32>(c),vec4<f32>(24.0,2.0,-1.0,0.0));
 if(all(c<textureDimensions(flowA))){textureStore(flowA,vec3<i32>(c),vec4<f32>(3.0,-2.0,1.0,0.0));textureStore(flowB,vec3<i32>(c),vec4<f32>(-1.0,4.0,2.0,0.0));}
 let i=(c.z*n.y+c.y)*n.x+c.x;if(i<8u){sparks[i]=Spark(vec3<f32>(20.0+f32(i),40.0,32.0),0.2+f32(i)*0.01,vec3<f32>(24.0,2.0,-1.0),1.2);}
}`;
const READ_SPARKS = SPARK + /* wgsl */ `
@group(0) @binding(0) var<storage,read> src:array<Spark>;
@group(0) @binding(1) var<storage,read_write> out:array<Spark>;
@compute @workgroup_size(8) fn main(@builtin(global_invocation_id)i:vec3<u32>){if(i.x<8u){out[i.x]=src[i.x];}}`;

export async function runDomainDiagnostics(signal?: AbortSignal) {
  signal?.throwIfAborted();
  const { device, adapterName } = await requestGpu();
  const problems: string[] = [];
  device.addEventListener('uncapturederror', (event) => problems.push(event.error.message));
  device.pushErrorScope('validation');
  const simulations = new Set<Simulation>();
  const make = (quality: Quality) => { const simulation = createSimulation(device, quality); simulations.add(simulation); return simulation; };
  const dispose = (simulation: Simulation) => { simulation.dispose(); simulations.delete(simulation); };
  const readPipeline = computePipeline(device, READ);
  async function read(texture: GPUTexture) {
    signal?.throwIfAborted();
    const n = [texture.width, texture.height, texture.depthOrArrayLayers];
    const bytes = n.reduce((a, b) => a * b, 16);
    const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(readPipeline); pass.setBindGroup(0, device.createBindGroup({ layout: readPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: { buffer: output } }] }));
    pass.dispatchWorkgroups(...n.map((v) => Math.ceil(v / 4)) as [number, number, number]); pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ); const values = new Float32Array(staging.getMappedRange()).slice(); staging.unmap(); output.destroy(); staging.destroy();
    return values;
  }
  async function readSparks(simulation: Simulation) {
    const pipeline = computePipeline(device, READ_SPARKS), bytes = 8 * 32;
    const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: simulation.volumes.sparks } }, { binding: 1, resource: { buffer: output } }] })); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, staging, 0, bytes); device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(staging.getMappedRange()).slice(); staging.unmap(); output.destroy(); staging.destroy(); return values;
  }
  const quiet = { ...fire, emit: 0, burn: 0, buoyancy: 0, turbulence: 0, expansion: 0, cooling: 0, dissipation: 0 } as Preset;
  async function measure(simulation: Simulation) {
    const scalar = await read(simulation.volumes.scalars[simulation.scalarIndex]), velocity = await read(simulation.volumes.velocity), flow = await read(simulation.volumes.flows[simulation.scalarIndex][0]), sparks = await readSparks(simulation);
    const [nx, ny, nz] = simulation.grid, d = simulation.cellsPerUnit, origin = simulation.offset;
    let mass = 0; const centre = [0, 0, 0], speed = [0, 0, 0]; let finite = true;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = ((z * ny + y) * nx + x) * 4, weight = scalar[at + 2]; mass += weight;
      const cell = [x, y, z];
      for (let axis = 0; axis < 3; axis++) { centre[axis] += weight * (BOX_MIN[axis] + origin[axis] + (cell[axis] + 0.5) / d); speed[axis] += weight * velocity[at + axis] / d; }
      for (let c = 0; c < 4; c++) finite &&= Number.isFinite(scalar[at + c]) && Number.isFinite(velocity[at + c]);
    }
    for (let axis = 0; axis < 3; axis++) { centre[axis] /= mass; speed[axis] /= mass; }
    const flowDims = simulation.grid.map((v) => Math.ceil(v / 2));
    const cell = centre.map((world, axis) => Math.max(0, Math.min(flowDims[axis] - 1, Math.floor((world - BOX_MIN[axis] - origin[axis]) * d / 2))));
    const at = ((cell[2] * flowDims[1] + cell[1]) * flowDims[0] + cell[0]) * 4;
    const worldFlow = [flow[at] / d, flow[at + 1] / d, flow[at + 2] / d];
    const particles = Array.from({ length: 8 }, (_, i) => ({ position: [0, 1, 2].map((axis) => BOX_MIN[axis] + origin[axis] + (sparks[8 * i + axis] + 0.5) / d), velocity: [4, 5, 6].map((axis) => sparks[8 * i + axis] / d), age: sparks[8 * i + 3], life: sparks[8 * i + 7] }));
    return { mass: mass / d ** 3, centre, speed, worldFlow, particles, finite };
  }
  const distance = (a: number[], b: number[]) => Math.max(...a.map((n, i) => Math.abs(n - b[i])));
  const results: Record<string, unknown>[] = [];
  try {
    const original = make(QUALITIES.fast); original.clear();
    const tweaks = { ...defaultTweaks(quiet), wind: 24, windGust: 0, vorticity: 0, turbulence: 0, disturbance: 0, sparks: 0 };
    original.writeParams(0, { preset: quiet, tweaks, boxes: [], time: 0, dt: DT, emitting: false, blast: 0 });
    const seed = computePipeline(device, SEED), encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(seed); pass.setBindGroup(0, device.createBindGroup({ layout: seed.getBindGroupLayout(0), entries: [original.volumes.velocity, original.volumes.scalars[original.scalarIndex], ...original.volumes.flows[original.scalarIndex]].map((texture, binding) => ({ binding, resource: texture.createView() })).concat([{ binding: 4, resource: { buffer: original.volumes.sparks } } as never]) })); pass.dispatchWorkgroups(16, 24, 16); pass.end(); device.queue.submit([encoder.finish()]);
    const before = await measure(original);
    let previous = original;
    for (const [name, quality, tolerance] of [
      ['Negative-origin same-density growth', { ...QUALITIES.fast, grid: [128, 128, 96], cellsPerUnit: 64, domainOffset: [-0.5, 0, -0.25] }, 0.002],
      ['Capacity coarsening preserves world extent', { ...QUALITIES.fast, grid: [64, 64, 48], cellsPerUnit: 32, domainOffset: [-0.5, 0, -0.25] }, 0.025],
    ] as [string, Quality, number][]) {
      signal?.throwIfAborted(); const next = make(quality); await next.copyStateFrom(previous); const after = await measure(next);
      const massError = Math.abs(after.mass / before.mass - 1), centreError = distance(after.centre, before.centre), velocityError = distance(after.speed, before.speed), flowError = distance(after.worldFlow, before.worldFlow);
      const particleError = Math.max(...after.particles.map((particle, i) => Math.max(distance(particle.position, before.particles[i].position), distance(particle.velocity, before.particles[i].velocity), Math.abs(particle.age - before.particles[i].age), Math.abs(particle.life - before.particles[i].life))));
      results.push({ name, grid: next.grid, cellsPerUnit: next.cellsPerUnit, offset: next.offset, massError, centreError, velocityError, flowError, particleError, finite: after.finite, pass: after.finite && massError < tolerance && centreError < tolerance && velocityError < 0.002 && flowError < 0.002 && particleError < 0.00001 });
      previous = next;
    }
    const layout = { grid: [64, 96, 64] as [number, number, number], cellsPerUnit: 64, offset: [0, 0, 0] as [number, number, number] };
    const noChurn = planDomain(layout, { min: [22, 0, 22], max: [42, 25, 42], speed: [0, 0.2, 0], occupied: 100 }, fire as Preset, defaultTweaks(fire as Preset), [], 1_572_864, device.limits.maxTextureDimension3D) === null;
    const shrunk = planDomain({ grid: [128, 128, 96], cellsPerUnit: 32, offset: [-1.5, 0, -1] }, { min: [52, 0, 32], max: [72, 30, 48], speed: [0, 0, 0], occupied: 100 }, fire as Preset, defaultTweaks(fire as Preset), [], 1_572_864, device.limits.maxTextureDimension3D, 0.12, 64);
    results.push({ name: 'Planner avoids idle churn and releases empty tails', noChurn, shrunk, pass: noChurn && Boolean(shrunk && shrunk.cellsPerUnit === 64 && shrunk.grid.reduce((a, b) => a * b, 1) <= 1_572_864) });
    for (const simulation of [...simulations]) dispose(simulation);
    // A real continuous source crosses the old right bound while the grid grows around it.
    const source = { ...quiet, emit: 8, radius: 5, emitter: [24, 8, 32] } as Preset;
    const windTweaks = { ...defaultTweaks(source), wind: 60, windGust: 0, vorticity: 0, turbulence: 0, disturbance: 0, sparks: 0 };
    let continuous = make(QUALITIES.fast); continuous.clear(); let resizeCount = 0;
    for (let step = 0; step < 96; step++) {
      signal?.throwIfAborted();
      continuous.writeParams(0, { preset: source, tweaks: windTweaks, boxes: [], time: step * DT, dt: DT, emitting: true, blast: 0 });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); continuous.step(pass, 0); pass.end(); device.queue.submit([encoder.finish()]);
      if (step % 4 === 3) {
        const probe = await continuous.probeDomain();
        if (probe) {
          const layout = planDomain(continuous.layout, probe, source, windTweaks, [], 1_572_864, device.limits.maxTextureDimension3D, 0.12, 64);
          if (layout) { const next = make({ ...QUALITIES.fast, ...layout, domainOffset: layout.offset }); await next.copyStateFrom(continuous); dispose(continuous); continuous = next; resizeCount++; }
        }
      }
    }
    const occupied = await continuous.probeDomain();
    const rightmostWorld = occupied ? BOX_MIN[0] + continuous.offset[0] + (occupied.max[0] + 1) / continuous.cellsPerUnit : -Infinity;
    const rightPadding = occupied ? (continuous.grid[0] - occupied.max[0] - 1) / continuous.cellsPerUnit : 0;
    results.push({ name: 'Continuous source under wind crosses old boundary without a cut', steps: 96, resizeCount, grid: continuous.grid, cellsPerUnit: continuous.cellsPerUnit, offset: continuous.offset, rightmostWorld, oldRightBoundary: 0.5, rightPadding, pass: resizeCount > 0 && rightmostWorld > 0.65 && rightPadding > 0.04 });
    dispose(continuous);
    const seedFields: Float32Array[] = [], hashes: number[] = [];
    for (const seed of [0, 0, 59]) {
      const simulation = make(QUALITIES.fast); simulation.clear();
      const seedTweaks = { ...defaultTweaks(fire as Preset), seed, wind: 0, windGust: 0, vorticity: 0, turbulence: 0, disturbance: 0, sparks: 0 };
      for (let step = 0; step < 20; step++) {
        signal?.throwIfAborted(); simulation.writeParams(0, { preset: fire as Preset, tweaks: seedTweaks, boxes: [], time: step * DT, dt: DT, emitting: true, blast: 0 });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass(); simulation.step(pass, 0); pass.end(); device.queue.submit([encoder.finish()]);
        if (step % 4 === 3) await device.queue.onSubmittedWorkDone();
      }
      const field = await read(simulation.volumes.scalars[simulation.scalarIndex]);
      let hash = 2166136261; for (const value of new Uint8Array(field.buffer)) hash = Math.imul(hash ^ value, 16777619) >>> 0;
      hashes.push(hash); seedFields.push(field); dispose(simulation);
    }
    let difference = 0, reference = 0;
    for (let i = 0; i < seedFields[0].length; i++) { difference += Math.abs(seedFields[0][i] - seedFields[2][i]); reference += Math.abs(seedFields[0][i]); }
    const relativeDifference = difference / Math.max(reference, 1e-9);
    results.push({ name: 'Seed repeatability and coherent new realization', seeds: [0, 0, 59], hashes, relativeDifference, pass: hashes[0] === hashes[1] && hashes[0] !== hashes[2] && relativeDifference > 0.1 });
    // Force compilation of the production bounds kernel as part of this diagnostic, too.
    computePipeline(device, DOMAIN_PROBE);
    const validation = await device.popErrorScope(); if (validation) problems.push(validation.message);
    return { adapter: adapterName, cases: results, validationErrors: problems, pass: !problems.length && results.every((result) => result.pass) };
  } finally { await device.queue.onSubmittedWorkDone(); for (const simulation of simulations) dispose(simulation); device.destroy(); }
}
