import { bindResources, computePipeline } from './gpu';
import type { Volumes } from './simulation';
import { SPARK_COUNT, SPARK_DRAW, SPARK_STEP } from './spark-shaders';

// Sparks per second the whole pool trickles at full strength, besides the pops, and how many an explosion throws
// per unit of blast.
const SPAWN_RATE = 280;
const BURST_PER_BLAST = 55;
const FLOATS_PER_SPARK = 8;
const WORKGROUP = 64;
const ADD: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one', operation: 'add' };

type SparkVolumes = { paramSlots: GPUBuffer[]; velocity: GPUTexture; scalars: GPUTexture[]; sampler: GPUSampler };

// The particles and the compute step that moves and respawns them; one spawn uniform per params slot, like the step's.
export function createSparkSimulation(device: GPUDevice, { paramSlots, velocity, scalars, sampler }: SparkVolumes) {
  const buffer = device.createBuffer({ size: SPARK_COUNT * FLOATS_PER_SPARK * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const spawn = paramSlots.map(() => device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  const pipeline = computePipeline(device, SPARK_STEP);
  const groups = paramSlots.map((params, slot) => scalars.map((sca) => bindResources(device, pipeline, [params, buffer, velocity, sca, sampler, spawn[slot]])));
  // Every spark starts dead: its age past its life.
  const dead = new Float32Array(SPARK_COUNT * FLOATS_PER_SPARK);
  for (let i = 0; i < SPARK_COUNT; i++) dead[i * FLOATS_PER_SPARK + 3] = 1;
  const spawnData = new Float32Array(8);

  return {
    buffer,
    writeParams(slot: number, strength: number, size = 1, lifetime = 1) {
      spawnData.set([(SPAWN_RATE * strength) / SPARK_COUNT, (BURST_PER_BLAST * strength) / SPARK_COUNT, strength, size, lifetime]);
      device.queue.writeBuffer(spawn[slot], 0, spawnData);
    },
    step(pass: GPUComputePassEncoder, slot: number, scalarIndex: number) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, groups[slot][scalarIndex]);
      pass.dispatchWorkgroups(SPARK_COUNT / WORKGROUP);
    },
    clear() {
      device.queue.writeBuffer(buffer, 0, dead);
    },
    remapFrom(source: GPUBuffer, oldDensity: number, newDensity: number, oldOffset: number[], newOffset: number[]) {
      const code = `struct R { oldOffset: vec3<f32>, oldDensity: f32, newOffset: vec3<f32>, newDensity: f32 }
struct Spark { pos: vec3<f32>, age: f32, vel: vec3<f32>, life: f32 }
@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var<storage,read> old: array<Spark>;
@group(0) @binding(2) var<storage,read_write> next: array<Spark>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) { let i=id.x; if(i>=arrayLength(&old)){return;} var s=old[i]; s.pos=((s.pos+vec3<f32>(0.5))/r.oldDensity+r.oldOffset-r.newOffset)*r.newDensity-vec3<f32>(0.5); s.vel*=r.newDensity/r.oldDensity; next[i]=s; }`;
      const remap = computePipeline(device, code);
      const uniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uniform, 0, new Float32Array([...oldOffset, oldDensity, ...newOffset, newDensity]));
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(remap); pass.setBindGroup(0, bindResources(device, remap, [uniform, source, buffer])); pass.dispatchWorkgroups(SPARK_COUNT / WORKGROUP); pass.end(); device.queue.submit([encoder.finish()]);
      void device.queue.onSubmittedWorkDone().then(() => uniform.destroy());
    },
    dispose() {
      buffer.destroy();
      for (const uniform of spawn) uniform.destroy();
    },
  };
}

// The streak pass: additive onto the live frame after its temporal resolve, or into the export's fire layer alone.
export function createSparkRenderer(device: GPUDevice, view: GPUBuffer, volumes: Volumes, format: GPUTextureFormat) {
  const shader = device.createShaderModule({ code: SPARK_DRAW });
  // How much of the sparks a draw shows; a looping export fades them across its seam.
  const fade = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let shown = 0;
  const show = (weight: number) => {
    if (weight === shown) return;
    shown = weight;
    device.queue.writeBuffer(fade, 0, new Float32Array([weight]));
  };
  // A pipeline and its bind groups, one per scalar volume.
  const make = (entryPoint: string, targets: GPUColorTargetState[]) => {
    const pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: shader, entryPoint: 'vs' },
      fragment: { module: shader, entryPoint, targets },
      primitive: { topology: 'triangle-strip' },
    });
    const groups = volumes.scalars.map((scalars) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: view } },
          { binding: 1, resource: { buffer: volumes.params } },
          { binding: 2, resource: scalars.createView() },
          { binding: 3, resource: volumes.sampler },
          { binding: 4, resource: { buffer: volumes.sparks } },
          { binding: 5, resource: { buffer: fade } },
        ],
      }),
    );
    return { pipeline, groups };
  };
  const fire: GPUColorTargetState = { format, blend: { color: ADD, alpha: ADD } };
  const live = make('fs', [fire]);
  let exported: ReturnType<typeof make> | null = null;

  const draw = (pass: GPURenderPassEncoder, { pipeline, groups }: typeof live, scalarIndex: number) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, groups[scalarIndex]);
    pass.draw(4, SPARK_COUNT);
  };

  return {
    draw(pass: GPURenderPassEncoder, scalarIndex: number) {
      show(1);
      draw(pass, live, scalarIndex);
    },
    // The export's other layers are attached with their writes masked off, so one draw inside its pass adds to fire only.
    drawExport(pass: GPURenderPassEncoder, scalarIndex: number, weight: number) {
      exported ??= make('fsExport', [fire, ...Array.from({ length: 3 }, () => ({ format, writeMask: 0 }))]);
      show(weight);
      draw(pass, exported, scalarIndex);
    },
    dispose() {
      fade.destroy();
    },
  };
}
