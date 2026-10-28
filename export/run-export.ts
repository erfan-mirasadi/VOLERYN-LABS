import type { Engine } from '@/engine/engine';
import { fitCamera, type Fps, loopOverlap, type OfflineJob, type RenderedFrame, substepsFor } from '@/engine/offline-render';
import { EXPORT_STEP_CELLS } from '@/engine/render';
import { encodeExr } from './exr';
import type { Destination } from './files';
import { createSheet, MOTION_RANGE } from './flipbook';
import { createMp4Sink } from './mp4';
import { cameraMatrices, toExrChannels, toRgba } from './planes';
import { encodeJson, encodePng } from './png';

// Seconds simulated before the first frame, so a fire has risen and settled; an explosion starts at its charge.
const WARMUP = 3;
const FLIPBOOK_FOV = 20;
// A sheet simulates at 60 fps and keeps every stride-th frame, so its cells span the clip at whatever playback rate that gives.
const FLIPBOOK_FPS = 60;

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

export async function runExport(engine: Engine, settings: ExportSettings, destination: Destination, onProgress: (progress: Progress) => void, signal: AbortSignal) {
  const snapshot = engine.snapshot();
  const { preset } = snapshot;
  const explosion = preset.mode === 'explosion';
  const flipbook = settings.mode === 'flipbook';
  const exr = settings.mode === 'sequence' && settings.format === 'exr';
  const [cols, rows] = settings.grid.split('x').map(Number);
  const [width, height] = flipbook ? [settings.cell, settings.cell] : settings.size.split('x').map(Number);
  const cells = cols * rows;
  const stride = flipbook ? Math.ceil((settings.seconds * FLIPBOOK_FPS) / cells) : 1;
  const job: OfflineJob = {
    width,
    height,
    fps: flipbook ? FLIPBOOK_FPS : settings.fps,
    frames: flipbook ? cells : settings.seconds * settings.fps,
    stride,
    warmup: explosion ? 0 : WARMUP,
    samples: settings.samples,
    camera: flipbook ? fitCamera(snapshot.camera.eye, FLIPBOOK_FOV) : snapshot.camera,
    loop: settings.loop && !explosion,
    read: exr ? ['fire', 'firelight', 'smoke', 'data'] : flipbook && settings.motion !== 'off' ? ['display', 'data'] : ['display'],
  };
  const overlap = job.loop ? loopOverlap(job.frames) : 0;
  const sidecar = {
    app: 'Voleryn Labs',
    version: 1,
    preset,
    tweaks: snapshot.tweaks,
    boxes: snapshot.boxes,
    camera: { ...job.camera, orbit: snapshot.orbit },
    render: {
      width: job.width,
      height: job.height,
      fps: job.fps,
      frames: job.frames,
      stride,
      first: 1,
      warmup: job.warmup,
      substeps: substepsFor(job.fps),
      dt: 1 / (job.fps * substepsFor(job.fps)),
      samples: job.samples,
      stepCells: EXPORT_STEP_CELLS,
      loop: { overlap, method: 'field-space crossfade' },
    },
    output: outputNotes(settings, exr),
    ...(flipbook && {
      flipbook: {
        cols,
        rows,
        frames: job.frames,
        fps: job.fps / stride,
        seconds: (job.frames * stride) / job.fps,
        stride,
        cell: settings.cell,
        order: 'row-major from top-left',
        alpha: settings.alpha,
        lens: FLIPBOOK_FOV,
        framing: 'the domain\'s inscribed cylinder with an 8% margin, from the live orbit direction',
        ...(settings.motion !== 'off' && {
          motion: {
            file: `${preset.id}-motion.png`,
            engine: settings.motion,
            range: MOTION_RANGE,
            decode: `uv = (rg - 0.5) * 2 * ${MOTION_RANGE} per frame`,
            vAxis: settings.motion === 'unity' ? 'up' : 'down',
            neutral: 128,
          },
        }),
      },
    }),
  };
  const sink = flipbook
    ? flipbookSink(destination, preset.id, settings, cols, rows)
    : settings.mode === 'video'
      ? await videoSink(destination, preset.id, job)
      : exr
        ? exrSink(destination, preset.id, job, JSON.stringify(sidecar))
        : pngSink(destination, preset.id, settings);
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

function outputNotes(settings: ExportSettings, exr: boolean) {
  if (exr) {
    return {
      mode: settings.mode,
      format: 'exr',
      colour: 'scene-linear Rec.709/D65, premultiplied; RGB = fire + firelight + smoke; Z planar camera depth, far 1e10; forward.u/v pixels per frame, +v up; temperature in simulation units',
    };
  }
  if (settings.mode === 'video') return { mode: 'video', format: 'mp4', colour: 'H.264 4:2:0 BT.709 limited range of the sRGB display image over black' };
  return {
    mode: settings.mode,
    format: 'png',
    alpha: settings.alpha,
    bits: settings.mode === 'sequence' ? settings.bits : 8,
    colour: `sRGB-encoded display image (ACES fit), alpha ${settings.alpha}`,
    ...(settings.alpha === 'straight' && { note: 'alpha raised to max(rgb) so emissive flame survives' }),
  };
}

const frameName = (id: string, index: number, extension: string) => `${id}.${String(index + 1).padStart(4, '0')}.${extension}`;

function pngSink(destination: Destination, id: string, settings: ExportSettings): Sink {
  return {
    frame: async (frame) => destination.write(frameName(id, frame.index, 'png'), await encodePng(frame.width, frame.height, settings.bits, toRgba(frame, settings.bits, settings.alpha === 'straight'))),
    finish: async () => {},
  };
}

function exrSink(destination: Destination, id: string, job: OfflineJob, comments: string): Sink {
  const meta = { fps: job.fps, ...cameraMatrices(job.camera, job.width / job.height), comments };
  return {
    frame: async (frame) => destination.write(frameName(id, frame.index, 'exr'), await encodeExr(frame.width, frame.height, toExrChannels(frame), meta)),
    finish: async () => {},
  };
}

// The premultiplied display colour is the picture over black, so the alpha byte is simply ignored.
async function videoSink(destination: Destination, id: string, job: OfflineJob): Promise<Sink> {
  const mp4 = await createMp4Sink(destination, `${id}.mp4`, job.width, job.height, job.fps);
  return { frame: (frame) => mp4.frame(toRgba(frame, 8, false), frame.index), finish: mp4.finish, abort: mp4.abort };
}

function flipbookSink(destination: Destination, id: string, settings: ExportSettings, cols: number, rows: number): Sink {
  const sheet = createSheet(cols, rows, settings.cell, settings.alpha === 'straight', settings.motion);
  return {
    frame: async (frame) => sheet.place(frame),
    async finish() {
      await destination.write(`${id}-sheet.png`, await encodePng(sheet.width, sheet.height, 8, sheet.colour));
      if (sheet.vectors) await destination.write(`${id}-motion.png`, await encodePng(sheet.width, sheet.height, 8, sheet.vectors));
    },
  };
}
