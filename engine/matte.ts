import type { DepthMap } from './depth';
import { box, COEFFICIENTS, FINISH, SOURCE } from './matte-shaders';

// The window sums behind a covariance cancel to a few parts in ten thousand, more than half floats hold, so the
// working textures are full floats; the finished matte is halves, which a sampler can interpolate.
const WORK: GPUTextureFormat = 'rgba32float';
const DONE: GPUTextureFormat = 'rgba16float';
// The window means a colour guide needs: the colour and the cover, their products, the squares and the mixed products.
const MOMENTS = 4;
const GROUP = 8;

// The matte that holds the fire out behind whatever in the footage is nearer than it. The depth map is coarse, a
// few plate pixels to the texel, so its edges alone would be soft and a little off. A fast guided filter (He and
// Sun, 2015) fixes both: the fit runs at the depth map's size, and what it leaves is applied to the plate at full
// size by whoever draws the frame, which puts the matte's edges on the plate's own. The passes are compute shaders
// because four full-float targets are more than a render pass may write at once.
export function createMatte(device: GPUDevice, sampler: GPUSampler) {
  const pipeline = (code: string, constants?: Record<string, number>) =>
    device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main', constants } });
  const source = pipeline(SOURCE);
  const across = pipeline(box(MOMENTS), { dx: 1, dy: 0 });
  const down = pipeline(box(MOMENTS), { dx: 0, dy: 1 });
  const fit = pipeline(COEFFICIENTS);
  const acrossFit = pipeline(box(1), { dx: 1, dy: 0 });
  const finish = pipeline(FINISH);
  const hold = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // What the drawing shaders read with no depth: zero coefficients between zero bounds, which cover nothing.
  const none = device.createTexture({ size: [1, 1], format: DONE, usage: GPUTextureUsage.TEXTURE_BINDING });

  type Work = { depth: GPUTexture; a: GPUTexture[]; b: GPUTexture[]; c: GPUTexture[]; coefficients: GPUTexture; bounds: GPUTexture };
  let work: Work | null = null;

  function release() {
    if (work) Object.values(work).flat().forEach((texture) => texture.destroy());
    work = null;
  }

  // Takes a new depth map, or none; true when the finished textures are different ones from before.
  function setDepth(map: DepthMap | null) {
    const kept = Boolean(work && map && work.depth.width === map.width && work.depth.height === map.height);
    if (!kept) {
      const had = Boolean(work);
      release();
      if (!map) return had;
      const make = (format: GPUTextureFormat, usage: number) => device.createTexture({ size: [map.width, map.height], format, usage: usage | GPUTextureUsage.TEXTURE_BINDING });
      const set = () => Array.from({ length: MOMENTS }, () => make(WORK, GPUTextureUsage.STORAGE_BINDING));
      work = { depth: make('r32float', GPUTextureUsage.COPY_DST), a: set(), b: set(), c: set(), coefficients: make(DONE, GPUTextureUsage.STORAGE_BINDING), bounds: make(DONE, GPUTextureUsage.STORAGE_BINDING) };
    }
    if (map) device.queue.writeTexture({ texture: work!.depth }, map.data, { bytesPerRow: 4 * map.width }, [map.width, map.height]);
    return !kept;
  }

  // Refits the matte for the plate as it now is and for where the fire stands: nearer than `nearness` covers it.
  function update(plate: GPUTexture, nearness: number) {
    if (!work) return;
    const { depth, a, b, c, coefficients, bounds } = work;
    device.queue.writeBuffer(hold, 0, new Float32Array([nearness]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const run = (using: GPUComputePipeline, resources: (GPUTexture | GPUSampler | GPUBuffer)[]) => {
      const entries = resources.map((resource, binding) => ({ binding, resource: resource instanceof GPUTexture ? resource.createView() : resource instanceof GPUBuffer ? { buffer: resource } : resource }));
      pass.setPipeline(using);
      pass.setBindGroup(0, device.createBindGroup({ layout: using.getBindGroupLayout(0), entries }));
      pass.dispatchWorkgroups(Math.ceil(depth.width / GROUP), Math.ceil(depth.height / GROUP));
    };
    run(source, [hold, plate, depth, sampler, ...a]);
    run(across, [...a, ...b]);
    run(down, [...b, ...c]);
    run(fit, [...c, a[0]]);
    run(acrossFit, [a[0], b[0]]);
    run(finish, [hold, b[0], depth, coefficients, bounds]);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  return {
    none,
    get coefficients() {
      return work?.coefficients ?? none;
    },
    get bounds() {
      return work?.bounds ?? none;
    },
    setDepth,
    update,
    dispose() {
      release();
      none.destroy();
      hold.destroy();
    },
  };
}
