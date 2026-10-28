import type { Engine } from '@/engine/engine';
import { type Fps, loopOverlap, type OfflineJob, type RenderedFrame, substepsFor } from '@/engine/offline-render';
import { EXPORT_STEP_CELLS } from '@/engine/render';
import type { Destination } from './files';
import { type Bytes, encodeJson, encodePng } from './png';

// Seconds simulated before the first frame, so a fire has risen and settled; an explosion starts at its charge.
const WARMUP = 3;

export const SIZES = ['1280x720', '1920x1080', '2560x1440', '3840x2160', '1080x1080', '1080x1920'] as const;

export type ExportSettings = {
  mode: 'sequence' | 'flipbook' | 'video';
  format: 'png' | 'exr';
  alpha: 'premultiplied' | 'straight';
  bits: 8 | 16;
  size: (typeof SIZES)[number];
  fps: Fps;
  seconds: 2 | 5 | 10 | 20;
  grid: '4x4' | '8x4' | '8x8';
  cell: 128 | 256 | 512;
  motion: 'off' | 'unreal' | 'unity';
  samples: 4 | 16;
  loop: boolean;
};

export const DEFAULT_SETTINGS: ExportSettings = {
  mode: 'sequence',
  format: 'png',
  alpha: 'premultiplied',
  bits: 8,
  size: '1920x1080',
  fps: 30,
  seconds: 5,
  grid: '8x8',
  cell: 256,
  motion: 'unreal',
  samples: 4,
  loop: false,
};

export type Progress = { done: number; total: number; msPerFrame: number };

type Sink = { frame(frame: RenderedFrame): Promise<void>; finish(): Promise<void>; abort?(): Promise<void> };

export function folderName(presetId: string, seed: number) {
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
  return `voleryn-${presetId}-seed${seed}-${stamp}`;
}

// Every half-float bit pattern as a float, so a frame's texels are read by lookup.
let halfTable: Float32Array | undefined;
export const halves = () => (halfTable ??= new Float32Array(new Float16Array(Uint16Array.from({ length: 65536 }, (_, i) => i).buffer)));

export async function runExport(engine: Engine, settings: ExportSettings, destination: Destination, onProgress: (progress: Progress) => void, signal: AbortSignal) {
  const snapshot = engine.snapshot();
  const { preset } = snapshot;
  const explosion = preset.mode === 'explosion';
  const [width, height] = settings.size.split('x').map(Number);
  const job: OfflineJob = {
    width,
    height,
    fps: settings.fps,
    frames: settings.seconds * settings.fps,
    warmup: explosion ? 0 : WARMUP,
    samples: settings.samples,
    camera: snapshot.camera,
    loop: settings.loop && !explosion,
    read: ['display'],
  };
  const sink = pngSink(destination, preset.id, settings);
  const sidecar = {
    app: 'Voleryn Labs',
    version: 1,
    preset,
    tweaks: snapshot.tweaks,
    boxes: snapshot.boxes,
    camera: { ...snapshot.camera, orbit: snapshot.orbit },
    render: {
      width: job.width,
      height: job.height,
      fps: job.fps,
      frames: job.frames,
      first: 1,
      warmup: job.warmup,
      substeps: substepsFor(job.fps),
      dt: 1 / (job.fps * substepsFor(job.fps)),
      samples: job.samples,
      stepCells: EXPORT_STEP_CELLS,
      loop: { overlap: job.loop ? loopOverlap(job.frames) : 0 },
    },
    output: {
      mode: settings.mode,
      format: settings.format,
      alpha: settings.alpha,
      bits: settings.bits,
      colour: `sRGB-encoded display image (ACES fit), alpha ${settings.alpha}`,
      ...(settings.alpha === 'straight' && { note: 'alpha raised to max(rgb) so emissive flame survives' }),
    },
  };
  const started = performance.now();
  try {
    await engine.render(
      job,
      async (frame) => {
        await sink.frame(frame);
        onProgress({ done: frame.index + 1, total: job.frames, msPerFrame: (performance.now() - started) / (frame.index + 1) });
      },
      signal,
    );
    await sink.finish();
    await destination.write(`${preset.id}.json`, encodeJson(sidecar));
    await destination.finish();
  } catch (error) {
    await sink.abort?.();
    destination.discard();
    throw error;
  }
}

function pngSink(destination: Destination, id: string, settings: ExportSettings): Sink {
  return {
    frame: async (frame) => destination.write(`${id}.${String(frame.index + 1).padStart(4, '0')}.png`, await encodePng(frame.width, frame.height, settings.bits, toRgba(frame, settings.bits, settings.alpha === 'straight'))),
    finish: async () => {},
  };
}

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
