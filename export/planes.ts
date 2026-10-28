import type { RenderedFrame } from '@/engine/offline-render';
import { type Camera, cameraBasis, type Vec3 } from '@/engine/render';
import type { ExrChannel } from './exr';
import type { Bytes } from './png';

const FAR = 1e10;

// Every half-float bit pattern as a float, so a frame's texels are read by lookup.
let halfTable: Float32Array | undefined;
export const halves = () => (halfTable ??= new Float32Array(new Float16Array(Uint16Array.from({ length: 65536 }, (_, i) => i).buffer)));

// The display plane as packed 8- or 16-bit RGBA. Straight alpha is raised to the brightest channel, so a bare flame,
// which covers nothing, still survives the division.
export function toRgba({ width, height, stride, planes }: RenderedFrame, bits: 8 | 16, straight: boolean): Bytes {
  const display = planes.display!;
  const half = halves();
  const peak = bits === 8 ? 255 : 65535;
  const out = new Uint8Array(width * height * (bits / 2));
  let o = 0;
  const put = (value: number) => {
    const q = Math.round(Math.min(Math.max(value, 0), 1) * peak);
    if (bits === 16) out[o++] = q >> 8;
    out[o++] = q & 255;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * stride + 4 * x;
      let r = half[display[i]];
      let g = half[display[i + 1]];
      let b = half[display[i + 2]];
      let a = half[display[i + 3]];
      if (straight) {
        a = Math.max(a, r, g, b);
        if (a > 0) {
          r /= a;
          g /= a;
          b /= a;
        }
      }
      put(r);
      put(g);
      put(b);
      put(a);
    }
  }
  return out;
}

// The light groups as they came back, the beauty as their sum, and depth and motion divided out of the coverage.
export function toExrChannels({ width, height, stride, planes }: RenderedFrame): ExrChannel[] {
  const { fire, firelight, smoke, data } = planes as Record<string, Uint16Array>;
  const half = halves();
  const n = width * height;
  const groups = ['fire', 'firelight', 'smoke'].map((name) => ({ name, source: planes[name as 'fire']!, channels: [0, 1, 2].map(() => new Uint16Array(n)) }));
  const alpha = new Uint16Array(n);
  const temperature = new Uint16Array(n);
  const beauty = [0, 1, 2].map(() => new Float16Array(n));
  const u = new Float16Array(n);
  const v = new Float16Array(n);
  const z = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0, i = y * width, o = y * stride; x < width; x++, i++, o += 4) {
      for (let k = 0; k < 3; k++) {
        for (const group of groups) group.channels[k][i] = group.source[o + k];
        beauty[k][i] = half[fire[o + k]] + half[firelight[o + k]] + half[smoke[o + k]];
      }
      alpha[i] = fire[o + 3];
      temperature[i] = firelight[o + 3];
      const coverage = half[data[o + 3]];
      z[i] = coverage > 0 ? half[data[o + 2]] / coverage : FAR;
      u[i] = coverage > 0 ? half[data[o]] / coverage : 0;
      v[i] = coverage > 0 ? -half[data[o + 1]] / coverage : 0;
    }
  }
  const bits = (floats: Float16Array) => new Uint16Array(floats.buffer);
  return [
    ...groups.flatMap(({ name, channels }) => channels.map((data, k) => ({ name: `${name}.${'RGB'[k]}`, data }))),
    ...beauty.map((data, k) => ({ name: 'RGB'[k], data: bits(data) })),
    { name: 'A', data: alpha },
    { name: 'temperature.Y', data: temperature },
    { name: 'forward.u', data: bits(u) },
    { name: 'forward.v', data: bits(v) },
    { name: 'Z', data: z },
  ];
}

// Row-vector matrices as the EXR spec defines them: camera space is left-handed with +z the view direction, and
// NDC runs from (0, 0) at the top left to (1, 1).
export function cameraMatrices(camera: Camera, aspect: number) {
  const { forward: f, right: r, up: u } = cameraBasis(camera);
  const e = camera.eye;
  const t = camera.tanHalfFov;
  const worldToCamera = [r[0], u[0], f[0], 0, r[1], u[1], f[1], 0, r[2], u[2], f[2], 0, -dot(r, e), -dot(u, e), -dot(f, e), 1];
  const projection = [0.5 / (t * aspect), 0, 0, 0, 0, -0.5 / t, 0, 0, 0.5, 0.5, 0, 1, 0, 0, 1, 0];
  const worldToNDC = Array.from({ length: 16 }, (_, i) => [0, 1, 2, 3].reduce((sum, k) => sum + worldToCamera[(i >> 2) * 4 + k] * projection[k * 4 + (i & 3)], 0));
  return { worldToCamera, worldToNDC };
}

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
