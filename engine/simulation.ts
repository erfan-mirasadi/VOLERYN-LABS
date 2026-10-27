import { CLEAR, HEADER, SHADERS } from './simulation-shaders';
import { type Box, DT, GRID, JACOBI, LIGHT, MAX_BOXES, MODE_INDEX, type Preset, SHADOW_GRID, type Tweaks } from './scene';

const PARAM_FLOATS = 60;

export type Volumes = {
  scalars: [GPUTexture, GPUTexture];
  shadow: GPUTexture;
  params: GPUBuffer;
  sampler: GPUSampler;
};

export type Simulation = ReturnType<typeof createSimulation>;

export function createSimulation(device: GPUDevice) {
  const volume = (format: GPUTextureFormat, size: readonly number[] = GRID) =>
    device.createTexture({
      size: [...size],
      dimension: '3d',
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
  const velocity = [volume('rgba16float'), volume('rgba16float')];
  const scalars: [GPUTexture, GPUTexture] = [volume('rgba16float'), volume('rgba16float')];
  const predicted = volume('rgba16float');
  const curl = volume('rgba16float');
  const pressure = [volume('r32float'), volume('r32float')];
  const divergence = volume('r32float');
  const shadow = volume('rgba16float', SHADOW_GRID);
  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
    addressModeW: 'clamp-to-edge',
  });
  const params = device.createBuffer({ size: PARAM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const compute = (code: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pipelines = Object.fromEntries(Object.entries(SHADERS).map(([name, body]) => [name, compute(HEADER + body)])) as Record<
    keyof typeof SHADERS,
    GPUComputePipeline
  >;
  const clearHalf = compute(CLEAR('rgba16float'));
  const clearFloat = compute(CLEAR('r32float'));

  const bind = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture | GPUSampler)[]) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map((resource, binding) => ({
        binding,
        resource: resource instanceof GPUBuffer ? { buffer: resource } : resource instanceof GPUTexture ? resource.createView() : resource,
      })),
    });
  const both = <T>(make: (i: number) => T) => [make(0), make(1)];
  const groups = {
    advectVelocity: both((i) => bind(pipelines.advectVelocity, [params, velocity[i], velocity[1 - i], sampler])),
    predict: both((i) => both((j) => bind(pipelines.predict, [params, velocity[i], scalars[j], predicted, sampler]))),
    correct: both((i) => both((j) => bind(pipelines.correct, [params, velocity[i], scalars[j], predicted, scalars[1 - j], sampler]))),
    curl: both((i) => bind(pipelines.curl, [params, velocity[i], curl])),
    forces: both((i) => both((j) => bind(pipelines.forces, [params, velocity[i], scalars[j], curl, velocity[1 - i]]))),
    divergence: both((i) => bind(pipelines.divergence, [params, velocity[i], divergence])),
    jacobi: both((k) => bind(pipelines.jacobi, [params, pressure[k], divergence, pressure[1 - k]])),
    project: both((i) => both((k) => bind(pipelines.project, [params, velocity[i], pressure[k], velocity[1 - i]]))),
    shadow: both((j) => bind(pipelines.shadow, [params, scalars[j], sampler, shadow])),
    clear: [
      ...[...velocity, ...scalars, predicted, curl].map((t) => [clearHalf, bind(clearHalf, [t])] as const),
      ...[...pressure, divergence].map((t) => [clearFloat, bind(clearFloat, [t])] as const),
    ],
  };

  const workgroups = GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const shadowWorkgroups = SHADOW_GRID.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const run = (pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, group: GPUBindGroup, size = workgroups) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...size);
  };

  // Ping-pong indices: which velocity, scalar and pressure texture currently holds the latest state.
  let vc = 0;
  let sc = 0;
  let pc = 0;
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
      preset.turbulence, preset.damping, 0, 0,
      ...boxData,
    ]);
    device.queue.writeBuffer(params, 0, paramData);
  }

  function step(pass: GPUComputePassEncoder) {
    const va = 1 - vc;
    run(pass, pipelines.advectVelocity, groups.advectVelocity[vc]);
    run(pass, pipelines.predict, groups.predict[vc][sc]);
    run(pass, pipelines.correct, groups.correct[vc][sc]);
    sc = 1 - sc;
    run(pass, pipelines.curl, groups.curl[va]);
    run(pass, pipelines.forces, groups.forces[va][sc]);
    run(pass, pipelines.divergence, groups.divergence[vc]);
    for (let i = 0; i < JACOBI; i++) {
      run(pass, pipelines.jacobi, groups.jacobi[pc]);
      pc = 1 - pc;
    }
    run(pass, pipelines.project, groups.project[vc][pc]);
    vc = va;
  }

  function updateShadow(pass: GPUComputePassEncoder) {
    run(pass, pipelines.shadow, groups.shadow[sc], shadowWorkgroups);
  }

  function clear() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const [pipeline, group] of groups.clear) run(pass, pipeline, group);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function dispose() {
    for (const texture of [...velocity, ...scalars, predicted, curl, ...pressure, divergence, shadow]) texture.destroy();
    params.destroy();
  }

  const volumes: Volumes = { scalars, shadow, params, sampler };
  return { volumes, writeParams, step, updateShadow, clear, dispose, get scalarIndex() { return sc; } };
}
