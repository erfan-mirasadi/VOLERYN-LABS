import { bindResources } from './gpu';
import { COARSEN, MASK, PROLONGATE, RESTRICT, SMOOTH } from './pressure-shaders';
import { GRID } from './scene';

const CYCLES = 2;
const SWEEPS = 2;
const COARSE_SWEEPS = 8;
const COARSEST = 4;

type Dispatch = { pipeline: GPUComputePipeline; group: GPUBindGroup; workgroups: [number, number, number] };

export type PressureSolver = ReturnType<typeof createPressureSolver>;

// Geometric multigrid for the pressure: V-cycles over halved grids with red-black Gauss-Seidel on each level.
// The solver also owns the solid masks, one per level; the level-0 pressure persists as the next solve's starting guess.
export function createPressureSolver(device: GPUDevice, params: GPUBuffer, divergence: GPUTexture) {
  const sizes: number[][] = [[...GRID]];
  while (Math.min(...sizes[sizes.length - 1]) > COARSEST) sizes.push(sizes[sizes.length - 1].map((n) => n / 2));
  const volume = (size: number[]) =>
    device.createTexture({ size, dimension: '3d', format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  const levels = sizes.map((size, l) => ({ size, pressure: volume(size), rhs: l === 0 ? divergence : volume(size), solid: volume(size) }));

  const compute = (code: string, constants?: Record<string, number>) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants } });
  const smooth = [0, 1].map((parity) => compute(SMOOTH, { parity }));
  const restrict = compute(RESTRICT);
  const prolongate = compute(PROLONGATE);
  const mask = compute(MASK);
  const coarsen = compute(COARSEN);
  const dispatch = (pipeline: GPUComputePipeline, resources: (GPUBuffer | GPUTexture)[], size: number[]): Dispatch => ({
    pipeline,
    group: bindResources(device, pipeline, resources),
    workgroups: size.map((n) => Math.ceil(n / 4)) as [number, number, number],
  });

  const steps = levels.map((level, l) => {
    const coarse = levels[l + 1];
    const [width, height, depth] = level.size;
    return {
      smooth: smooth.map((pipeline) => dispatch(pipeline, [level.pressure, level.rhs, level.solid], [width / 2, height, depth])),
      restrict: coarse && dispatch(restrict, [level.pressure, level.rhs, level.solid, coarse.pressure, coarse.rhs], coarse.size),
      lift: coarse && dispatch(prolongate, [coarse.pressure, level.pressure], level.size),
      coarsen: coarse && dispatch(coarsen, [level.solid, coarse.solid], coarse.size),
    };
  });
  const finestMask = dispatch(mask, [params, levels[0].solid], levels[0].size);

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
    solid: levels[0].solid,
    updateSolids(pass: GPUComputePassEncoder) {
      run(pass, finestMask);
      for (const { coarsen } of steps) if (coarsen) run(pass, coarsen);
    },
    solve(pass: GPUComputePassEncoder) {
      for (let i = 0; i < CYCLES; i++) cycle(pass, 0);
    },
    dispose() {
      for (const level of levels.slice(1)) level.rhs.destroy();
      for (const level of levels) {
        level.pressure.destroy();
        level.solid.destroy();
      }
    },
  };
}
