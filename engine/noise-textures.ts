// Tileable 32^3 noise baked once on the CPU: four channels of value-noise fBm, and the curl of three of them, which is
// divergence-free so the pressure solve leaves it alone when it stirs the gas.
const SIZE = 32;
const LATTICES = [4, 8, 16];
const WEIGHTS = [1, 0.5, 0.25];

export type NoiseTextures = ReturnType<typeof createNoiseTextures>;

export function createNoiseTextures(device: GPUDevice) {
  const channels = [0, 1, 2, 3].map(bakeChannel);
  const noise = new Uint8Array(SIZE ** 3 * 4);
  const curl = new Int8Array(SIZE ** 3 * 4);
  const curlFloat = new Float32Array(SIZE ** 3 * 3);
  const index = (x: number, y: number, z: number) => (((z + SIZE) % SIZE) * SIZE + ((y + SIZE) % SIZE)) * SIZE + ((x + SIZE) % SIZE);
  const slope = (f: Float32Array, x: number, y: number, z: number, axis: number) => {
    const d = [axis === 0 ? 1 : 0, axis === 1 ? 1 : 0, axis === 2 ? 1 : 0];
    return 0.5 * (f[index(x + d[0], y + d[1], z + d[2])] - f[index(x - d[0], y - d[1], z - d[2])]);
  };
  let peak = 0;
  for (let z = 0; z < SIZE; z++) {
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const i = index(x, y, z);
        channels.forEach((channel, c) => (noise[4 * i + c] = Math.round(255 * channel[i])));
        const [fx, fy, fz] = channels;
        curlFloat[3 * i] = slope(fz, x, y, z, 1) - slope(fy, x, y, z, 2);
        curlFloat[3 * i + 1] = slope(fx, x, y, z, 2) - slope(fz, x, y, z, 0);
        curlFloat[3 * i + 2] = slope(fy, x, y, z, 0) - slope(fx, x, y, z, 1);
        for (let k = 0; k < 3; k++) peak = Math.max(peak, Math.abs(curlFloat[3 * i + k]));
      }
    }
  }
  for (let i = 0; i < SIZE ** 3; i++) for (let k = 0; k < 3; k++) curl[4 * i + k] = Math.round((127 * curlFloat[3 * i + k]) / peak);

  const upload = (format: GPUTextureFormat, data: Uint8Array<ArrayBuffer> | Int8Array<ArrayBuffer>) => {
    const texture = device.createTexture({ size: [SIZE, SIZE, SIZE], dimension: '3d', format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, data, { bytesPerRow: SIZE * 4, rowsPerImage: SIZE }, [SIZE, SIZE, SIZE]);
    return texture;
  };
  return {
    noise: upload('rgba8unorm', noise),
    curl: upload('rgba8snorm', curl),
    repeat: device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat' }),
  };
}

// Three octaves of value noise on lattices that divide the texture, so every octave tiles; normalized to [0, 1].
function bakeChannel(seed: number) {
  const out = new Float32Array(SIZE ** 3);
  const fade = (t: number) => t * t * (3 - 2 * t);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  LATTICES.forEach((n, octave) => {
    const next = random(seed * 8 + octave + 1);
    const lattice = Float32Array.from({ length: n ** 3 }, next);
    const at = (x: number, y: number, z: number) => lattice[((z % n) * n + (y % n)) * n + (x % n)];
    const spacing = SIZE / n;
    for (let z = 0; z < SIZE; z++) {
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          const u = (x + 0.5) / spacing, v = (y + 0.5) / spacing, w = (z + 0.5) / spacing;
          const i = Math.floor(u), j = Math.floor(v), k = Math.floor(w);
          const fu = fade(u - i), fv = fade(v - j), fw = fade(w - k);
          const near = lerp(lerp(at(i, j, k), at(i + 1, j, k), fu), lerp(at(i, j + 1, k), at(i + 1, j + 1, k), fu), fv);
          const far = lerp(lerp(at(i, j, k + 1), at(i + 1, j, k + 1), fu), lerp(at(i, j + 1, k + 1), at(i + 1, j + 1, k + 1), fu), fv);
          out[(z * SIZE + y) * SIZE + x] += WEIGHTS[octave] * lerp(near, far, fw);
        }
      }
    }
  });
  let lo = Infinity, hi = -Infinity;
  for (const value of out) { lo = Math.min(lo, value); hi = Math.max(hi, value); }
  return out.map((value) => (value - lo) / (hi - lo));
}

// A small seeded generator (mulberry32), so the bake is the same on every machine.
function random(seed: number) {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
