import type { RenderedFrame } from '@/engine/offline-render';
import { halves, toRgba } from './planes';

// Motion sheets map this much UV per frame onto the full byte range; faster motion clamps.
export const MOTION_RANGE = 0.1;

// A square-celled sheet filled row by row from the top left: 8-bit colour, and optionally a motion sheet whose R
// and G hold the UV each pixel moves to the next frame, 128 meaning still. V runs down for Unreal, up for Unity.
export function createSheet(cols: number, rows: number, cell: number, straight: boolean, motion: 'off' | 'unreal' | 'unity') {
  const width = cols * cell;
  const height = rows * cell;
  const colour = new Uint8Array(width * height * 4);
  const vectors = motion === 'off' ? null : new Uint8Array(width * height * 4);
  if (vectors) new Uint32Array(vectors.buffer).fill(0xff008080);
  const encode = (uv: number) => Math.round(Math.min(Math.max(0.5 + uv / (2 * MOTION_RANGE), 0), 1) * 255);

  function place(frame: RenderedFrame) {
    const left = (frame.index % cols) * cell;
    const top = Math.floor(frame.index / cols) * cell;
    const rgba = toRgba(frame, 8, straight);
    for (let y = 0; y < cell; y++) colour.set(rgba.subarray(y * cell * 4, (y + 1) * cell * 4), ((top + y) * width + left) * 4);
    if (!vectors) return;
    const data = frame.planes.data!;
    const half = halves();
    const sign = motion === 'unity' ? -1 : 1;
    for (let y = 0; y < cell; y++) {
      for (let x = 0; x < cell; x++) {
        const o = y * frame.stride + 4 * x;
        const coverage = half[data[o + 3]];
        if (coverage <= 0) continue;
        const at = ((top + y) * width + left + x) * 4;
        vectors[at] = encode(half[data[o]] / coverage / cell);
        vectors[at + 1] = encode((sign * half[data[o + 1]]) / coverage / cell);
      }
    }
  }

  return { width, height, colour, vectors, place };
}
