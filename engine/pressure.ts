import { bindResources } from './gpu';
import { PROLONGATE, RESTRICT, SMOOTH } from './pressure-shaders';
import { GRID } from './scene';

const CYCLES = 2;
const SWEEPS = 2;
const COARSE_SWEEPS = 8;
const COARSEST = 4;

type Dispatch = { pipeline: GPUComputePipeline; group: GPUBindGroup; workgroups: [number, number, number] };

export type PressureSolver = ReturnType<typeof createPressureSolver>;

// Geometric multigrid for the pressure: V-cycles over halved grids with red-black Gauss-Seidel on each level.
// The level-0 pressure is kept between steps as the next solve's starting guess.
export function createPressureSolver(device: GPUDevice, params: GPUBuffer, divergence: GPUTexture) {
  const sizes: number[][] = [[...GRID]];
  while (Math.min(...sizes[sizes.length - 1]) > COARSEST) sizes.push(sizes[sizes.length - 1].map((n) => n / 2));
  const volume = (size: number[]) =>
    device.createTexture({ size, dimension: '3d', format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  const levels = sizes.map((size, l) => ({ size, pressure: volume(size), rhs: l === 0 ? divergence : volume(size) }));

  const shader = (code: string) => device.createShaderModule({ code });
  const compute = (module: GPUShaderModule, constants?: Record<string, number>) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main', constants } });
  const smoothShader = shader(SMOOTH);
  const restrictShader = shader(RESTRICT);
  const prolongate = compute(shader(PROLONGATE));
  const workgroups = (size: number[]) => size.map((n) => Math.ceil(n / 4)) as [number, number, number];
  const dispatch = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture)[], size: number[]): Dispatch => ({
    pipeline,
    group: bindResources(device, pipeline, resources),
    workgroups: workgroups(size),
  });

  const steps = levels.map((level, l) => {
    const scale = 2 ** l;
    const coarse = levels[l + 1];
    const [width, height, depth] = level.size;
    return {
      smooth: [0, 1].map((parity) => dispatch(compute(smoothShader, { parity, scale }), [params, level.pressure, level.rhs], [width / 2, height, depth])),
      restrict: coarse && dispatch(compute(restrictShader, { scale }), [params, level.pressure, level.rhs, coarse.pressure, coarse.rhs], coarse.size),
      lift: coarse && dispatch(prolongate, [coarse.pressure, level.pressure], level.size),
    };
  });

  const run = (pass: GPUComputePassEncoder, { pipeline, group, workgroups }: Dispatch) => {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...workgroups);
  };

  function cycle(pass: GPUComputePassEncoder, l: number) {
    const { smooth, restrict, lift } = steps[l];
    const sweeps = restrict ? SWEEPS : COARSE_SWEEPS;
    for (let i = 0; i < sweeps; i++) smooth.forEach((half) => run(pass, half));
    if (!restrict || !lift) return;
    run(pass, restrict);
    cycle(pass, l + 1);
    run(pass, lift);
    for (let i = 0; i < SWEEPS; i++) smooth.forEach((half) => run(pass, half));
  }

  return {
    texture: levels[0].pressure,
    solve(pass: GPUComputePassEncoder) {
      for (let i = 0; i < CYCLES; i++) cycle(pass, 0);
    },
    dispose() {
      for (const level of levels.slice(1)) level.rhs.destroy();
      for (const level of levels) level.pressure.destroy();
    },
  };
}
