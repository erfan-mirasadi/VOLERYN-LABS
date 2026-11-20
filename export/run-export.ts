import type { Engine } from '@/engine/engine';
import { fitCamera, fitBoundsCamera, fitToContent, type Fps, loopOverlap, type OfflineJob, type OfflineProgress, type RenderedFrame, substepsFor } from '@/engine/offline-render';
import { DEPTH_MODEL, DETAIL } from '@/engine/depth';
import { cameraBasis, exportMarch, type Camera } from '@/engine/render';
import { GRID, RENDER_GRID } from '@/engine/scene';
import { FIRE_RADIANCE } from '@/engine/blackbody';
import type { Vec3 } from '@/engine/render';
import { encodeExr } from './exr';
import type { Destination } from './files';
import { createSheet, MOTION_RANGE } from './flipbook';
import { createMp4Sink, videoBitrateFor } from './mp4';
import { createWebmSink } from './webm';
import { cameraMatrices, toExrChannels, toRgba } from './planes';
import { encodePng } from './png';

// Seconds simulated before the first frame, so a fire has risen and its smoke column has reached the top of the
// box; an explosion starts at its charge.
const DEFAULT_PREROLL = 6;
const FLIPBOOK_FOV = 20;
// A sheet simulates at 60 fps and keeps every stride-th frame, so its cells span the clip at whatever playback rate that gives.
const FLIPBOOK_FPS = 60;

export const SIZES = ['1280x720', '1920x1080', '2560x1440', '3840x2160', '4096x2160', '1080x1080', '1080x1920'] as const;
const MAX_FOOTAGE_SIDE = 4096;

// With footage loaded the frame is the footage's own, shrunk to fit 4096 pixels and made even for the video encoder.
export function footageSize({ width, height }: { width: number; height: number }) {
  const scale = Math.min(1, MAX_FOOTAGE_SIDE / Math.max(width, height));
  return [width, height].map((side) => 2 * Math.round((side * scale) / 2));
}

export type ExportSettings = {
  mode: 'sequence' | 'flipbook' | 'video';
  format: 'png' | 'exr';
  videoFormat: 'webm' | 'mp4';
  alpha: 'premultiplied' | 'straight';
  bits: 8 | 16;
  size: (typeof SIZES)[number];
  fps: Fps;
  seconds: 2 | 5 | 10 | 20;
  grid: '4x4' | '8x4' | '8x8';
  cell: 128 | 256 | 512;
  motion: 'off' | 'unreal' | 'unity';
  // How long the shutter stays open, as a share of the frame interval: 180 degrees or none.
  shutter: 0 | 0.5;
  samples: 4 | 16 | 32 | 64;
  fidelity: 'draft' | 'final';
  preroll: 0 | 2 | 6;
  loop: boolean;
  // With footage loaded: the fire laid over it, or the fire alone on the footage's frame.
  footage: 'composite' | 'off';
};

export const DEFAULT_SETTINGS: ExportSettings = {
  mode: 'sequence',
  format: 'png',
  videoFormat: 'webm',
  alpha: 'straight',
  bits: 16,
  size: '1920x1080',
  fps: 30,
  seconds: 5,
  grid: '8x8',
  cell: 256,
  motion: 'unreal',
  shutter: 0.5,
  samples: 16,
  fidelity: 'final',
  preroll: DEFAULT_PREROLL,
  loop: false,
  footage: 'composite',
};

export type Progress = Omit<OfflineProgress, 'phase'> & {
  phase: OfflineProgress['phase'] | 'probe' | 'encoding' | 'saving';
  framesDone: number;
  framesTotal: number;
  detail?: string;
};

// What a sheet holds: its cells span the clip at the playback rate the stride leaves.
export function flipbookTiming({ grid, seconds }: ExportSettings) {
  const [cols, rows] = grid.split('x').map(Number);
  const cells = cols * rows;
  const stride = Math.ceil((seconds * FLIPBOOK_FPS) / cells);
  return { cols, rows, cells, stride, seconds: (cells * stride) / FLIPBOOK_FPS, fps: FLIPBOOK_FPS / stride };
}

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
  const { cols, rows, cells, stride } = flipbookTiming(settings);
  const footage = flipbook ? null : snapshot.plate;
  const composited = Boolean(footage) && !exr && settings.footage === 'composite';
  const [width, height] = flipbook ? [settings.cell, settings.cell] : footage ? footageSize(footage) : settings.size.split('x').map(Number);
  // A moving explosion can travel outside the initial domain. The coarse content probe must see the whole path.
  const clipSeconds = flipbook ? cells * stride / FLIPBOOK_FPS : settings.seconds;
  const travelDistance = explosion ? snapshot.tweaks.wind * (1 + snapshot.tweaks.windGust) * clipSeconds / GRID[0] : 0;
  const heading = snapshot.tweaks.windYaw * Math.PI / 180;
  const travel: Vec3 = [travelDistance * Math.cos(heading), 0, travelDistance * Math.sin(heading)];
  const job: OfflineJob = {
    width,
    height,
    fps: flipbook ? FLIPBOOK_FPS : settings.fps,
    frames: flipbook ? cells : settings.seconds * settings.fps,
    stride: flipbook ? stride : 1,
    warmup: explosion ? 0 : (settings.preroll ?? DEFAULT_PREROLL),
    samples: settings.samples,
    camera: flipbook ? fitCamera(snapshot.camera.eye, FLIPBOOK_FOV, 0.92, travel) : snapshot.camera,
    loop: settings.loop && !explosion,
    read: exr ? ['fire', 'firelight', 'smoke', 'data'] : flipbook && settings.motion !== 'off' ? ['display', 'data'] : composited ? ['display', 'clean'] : ['display'],
    shutter: settings.shutter,
    simulation: settings.fidelity === 'draft' ? { grid: [128, 192, 128], lightScale: 4 } : RENDER_GRID,
    plate: composited,
    holdout: composited && Boolean(snapshot.occlusion),
  };
  let framesDone = 0;
  let actualLoopOverlap = 0;
  let active: Sink | undefined;
  const pause = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
  let lastPhase: Progress['phase'] = 'setup';
  let stageStarted = performance.now();
  const stage = (phase: Progress['phase'], detail?: string) => {
    if (phase !== lastPhase) { lastPhase = phase; stageStarted = performance.now(); }
    onProgress({ phase, completed: phase === 'encoding' || phase === 'saving' ? framesDone : 0,
      total: phase === 'encoding' || phase === 'saving' ? job.frames : 0,
      elapsedMs: performance.now() - stageStarted, framesDone, framesTotal: job.frames, ...(detail && { detail }),
    });
  };
  const report = (state: OfflineProgress, probe = false) => {
    if (!probe && state.phase === 'loop') actualLoopOverlap = state.total;
    const detail = probe ? ({ setup: 'Preparing framing pass', warmup: 'Simulating pre-roll for framing',
      loop: 'Preparing loop for framing', render: 'Scanning the animation' } as const)[state.phase] : undefined;
    onProgress({ ...state, phase: probe ? 'probe' : state.phase, framesDone, framesTotal: job.frames, ...(detail && { detail }) });
  };
  // File writes are a separate stage from encoding. A folder write may be slow even after every GPU frame has
  // completed; a zip download also has a final assembly step. Neither receives an invented ETA.
  const output: Destination = {
    name: destination.name,
    ...(destination.openStream && { openStream: (file: string) => destination.openStream!(file) }),
    async write(file, bytes) {
      signal.throwIfAborted();
      stage('saving', `Writing ${file}`);
      await pause();
      signal.throwIfAborted();
      await destination.write(file, bytes);
      signal.throwIfAborted();
    },
    finish: () => destination.finish(),
    discard: () => destination.discard(),
  };
  stage('setup', 'Preparing GPU');
  await pause();
  try {
    signal.throwIfAborted();
    if (flipbook) {
      const probeCamera = job.camera;
      job.camera = await fitToContent((probe, onFrame) => engine.render(probe, onFrame, signal, (state) => report(state, true)), job);
      // The image probe cannot report material outside its own lens. Reuse the occupied envelope measured by
      // that same simulation, with no second physical pass, when adaptive growth left the original view.
      const envelope = engine.renderInfo()?.bounds;
      if (envelope && !boundsInsideView(probeCamera, envelope)) job.camera = fitBoundsCamera(probeCamera, envelope, FLIPBOOK_FOV);
    }
    signal.throwIfAborted();
    const sidecar = sidecarFor(snapshot, settings, job);
    stage('setup', settings.mode === 'video' ? 'Preparing video encoder' : 'Preparing output');
    await pause();
    const primary = flipbook
      ? flipbookSink(output, preset.id, settings, cols, rows)
      : settings.mode === 'video'
        ? await videoOutput(output, preset.id, job, settings, composited,
          (frame) => stage('encoding', `Encoding matched outputs ${frame + 1} / ${job.frames}`))
        : exr
          ? exrSink(output, preset.id, job, JSON.stringify(sidecar))
          : pngSink(output, preset.id, settings);
    const sink = composited && settings.mode === 'sequence' && !exr
      ? withCompanion(primary, pngSink(output, `${preset.id}-clean`, settings), 'clean',
        (frame) => stage('encoding', `Encoding transparent frame ${frame + 1} / ${job.frames}`))
      : primary;
    active = sink;
    signal.throwIfAborted();
    await engine.render(
      job,
      async (frame) => {
        signal.throwIfAborted();
        stage('encoding', `Encoding frame ${frame.index + 1} / ${job.frames}`);
        await pause();
        signal.throwIfAborted();
        await sink.frame(frame);
        signal.throwIfAborted();
        framesDone = frame.index + 1;
      },
      signal,
      (state) => report(state),
    );
    signal.throwIfAborted();
    stage('encoding', flipbook ? 'Building flipbook sheets' : settings.mode === 'video' ? 'Finalizing video' : 'Finishing sequence');
    await pause();
    signal.throwIfAborted();
    await sink.finish();
    signal.throwIfAborted();
    const actual = engine.renderInfo();
    const completedSidecar = actual ? { ...sidecar, render: { ...sidecar.render,
      loop: { ...sidecar.render.loop, overlap: actualLoopOverlap, overlapRole: 'actual kept loop preparation frames' },
      adaptiveDomain: { ...actual, note: 'Bounds are occupied world-space material measured over pre-roll and the clip. Layouts record the actual density and dimensions; the requested initial grid is not held constant.',
        layouts: actual.layouts.map((layout) => {
          const march = exportMarch(layout.grid, layout.cellsPerUnit);
          const span = layout.grid.map((n) => n / layout.cellsPerUnit);
          return { ...layout, stepCells: march.stepCells,
            maxSteps: Math.max(march.maxSteps, Math.ceil(Math.hypot(...span) * GRID[0] / march.stepCells)) };
        }),
      },
    } } : sidecar;
    await output.write(`${preset.id}.json`, new TextEncoder().encode(JSON.stringify(completedSidecar, null, 2)));
    stage('saving', 'Finishing export');
    await pause();
    signal.throwIfAborted();
    await destination.finish();
    signal.throwIfAborted();
  } catch (error) {
    await active?.abort?.();
    await destination.discard();
    throw error;
  }
}

// Flipbook cells are square. A finite camera is an intentional crop for a shot, but automatic sprite fitting
// must include every occupied corner of the animation's adaptive envelope, even when the image probe missed it.
function boundsInsideView(camera: Camera, bounds: { min: Vec3; max: Vec3 }) {
  const { forward, right, up } = cameraBasis(camera);
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (const x of [bounds.min[0], bounds.max[0]]) for (const y of [bounds.min[1], bounds.max[1]]) for (const z of [bounds.min[2], bounds.max[2]]) {
    const rel: Vec3 = [x - camera.eye[0], y - camera.eye[1], z - camera.eye[2]];
    const depth = dot(rel, forward);
    if (depth <= 0) return false;
    const ndc = [dot(rel, right) / (depth * camera.tanHalfFov) + (camera.shift?.[0] ?? 0), dot(rel, up) / (depth * camera.tanHalfFov) + (camera.shift?.[1] ?? 0)];
    if (ndc.some((value) => Math.abs(value) > 1)) return false;
  }
  return true;
}

// Every input the job had, so the same frames can be rendered again.
function sidecarFor({ preset, tweaks, boxes, orbit, framing, plate, occlusion, grid: previewGrid }: ReturnType<Engine['snapshot']>, settings: ExportSettings, job: OfflineJob) {
  const exr = settings.mode === 'sequence' && settings.format === 'exr';
  const flipbook = settings.mode === 'flipbook';
  const [cols, rows] = settings.grid.split('x').map(Number);
  const grid = job.simulation?.grid ?? previewGrid;
  const overlap = job.loop ? loopOverlap(job.frames, grid) : 0;
  return {
    app: 'Voleryn Labs',
    version: 2,
    renderer: 'volumetric-soot-v2',
    preset,
    tweaks,
    boxes,
    camera: { ...job.camera, orbit, ...(!flipbook && { framing }) },
    ...(plate && !flipbook && {
      footage: {
        ...plate,
        composited: Boolean(job.plate),
        ...(occlusion && {
          occlusion: job.holdout
            ? { ...occlusion, model: DEPTH_MODEL.name, detail: DETAIL.full, matte: 'the footage’s depth cut at the fire, with colour models of either side placing what is finer than the depth map; applied to the display image' }
            : { ...occlusion, note: 'not applied: the EXR layers are the fire alone' },
        }),
      },
    }),
    render: {
      width: job.width,
      height: job.height,
      fps: job.fps,
      frames: job.frames,
      stride: job.stride,
      first: 1,
      warmup: job.warmup,
      substeps: substepsFor(job.fps),
      dt: 1 / (job.fps * substepsFor(job.fps)),
      samples: job.samples,
      filter: 'gaussian, sigma 0.5 px, cut at 1.5 px',
      shutter: job.shutter,
      grid: [...grid],
      gridRole: 'requested initial grid; actual adaptive layouts and density are recorded in the JSON sidecar',
      stepCells: exportMarch(grid, job.simulation?.cellsPerUnit ?? grid[0]).stepCells,
      stepCellsRole: 'initial ray step in base cells; actual layout-dependent steps are recorded in the JSON sidecar',
      domain: 'adaptive occupied-material envelope; capacity limits change sampling density rather than clipping the material bounds',
      emissionCalibration: FIRE_RADIANCE,
      colourTemperature: 'reference Kelvin changes chromaticity at matched luminance; thermal radiance follows the fixed 1900 K calibration and Brightness EV scales emission',
      loop: {
        overlap,
        overlapRole: 'initial estimate; actual kept preparation frames are recorded in the JSON sidecar',
        method: 'field-space crossfade',
        ...(overlap > 0 && { note: 'during the crossfade the flame detail phase and the motion vectors follow the live state, not the blend' }),
      },
    },
    output: outputNotes(settings, exr, Boolean(job.plate), preset.id, job),
    ...(flipbook && {
      flipbook: {
        cols,
        rows,
        frames: job.frames,
        fps: job.fps / job.stride,
        seconds: (job.frames * job.stride) / job.fps,
        stride: job.stride,
        cell: settings.cell,
        order: 'row-major from top-left',
        alpha: settings.alpha,
        lens: (2 * Math.atan(job.camera.tanHalfFov) * 180) / Math.PI,
        framing: 'fitted to the clip\'s content with an 8% margin, from the live orbit direction',
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
}

function outputNotes(settings: ExportSettings, exr: boolean, composited: boolean, id: string, job: OfflineJob) {
  if (exr) {
    return {
      mode: settings.mode,
      format: 'exr',
      colour: 'scene-linear Rec.709/D65, premultiplied; RGB = fire + firelight + smoke; Z planar camera depth, far 1e10; forward.u/v pixels per frame, +v up; temperature in normalized simulation units',
      temperatureCalibration: 'Chromaticity K = 1000 + (referenceK - 1000) * (1 - exp(-max(heat, 0))) / (1 - exp(-1)); referenceK is tweaks.temperature. RGB is normalized to the luminance of the fixed 1900 K thermal curve before Brightness EV and exposure.',
    };
  }
  const cleanDescription = {
    holdout: false, content: 'complete VFX on transparency, without footage or foreground masks',
    sampling: 'same accumulated volume and spark samples, camera, shutter, geometry and timing; no second simulation or raymarch',
    width: job.width, height: job.height, fps: job.fps, frames: job.frames,
  };
  const over = composited ? 'the footage, composited in linear light' : 'black';
  if (settings.mode === 'video') {
    const format = settings.videoFormat ?? 'webm';
    const bitrate = videoBitrateFor(job.width, job.height, job.fps);
    if (format === 'webm') return {
      mode: 'video', format, codec: 'vp9', bits: 8, bitrate, file: `${id}.webm`, alpha: composited ? 'opaque' : 'straight',
      colour: `VP9 4:2:0 BT.709 display image ${composited ? `over ${over}` : 'with alpha'}`,
      ...(composited && { cleanCompanion: { ...cleanDescription, format, codec: 'vp9', bits: 8, bitrate,
        file: `${id}-clean.webm`, alpha: 'straight', encoding: 'same colour codec settings as the composite; WebM alpha side data' } }),
    };
    return { mode: 'video', format, codec: 'avc', bits: 8, bitrate, file: `${id}.mp4`,
      colour: `H.264 4:2:0 BT.709 limited range of the display image over ${over}`,
      cleanCompanion: { ...cleanDescription, format: 'png', filePattern: `${id}-clean.%04d.png`, bits: 16, alpha: settings.alpha,
        reason: 'H.264 MP4 has no alpha channel; the lossless 16-bit PNG master preserves transparency at the same geometry and timing' },
    };
  }
  const companion = composited ? { cleanCompanion: { ...cleanDescription,
    format: 'png', filePattern: `${id}-clean.%04d.png`, bits: settings.bits, alpha: settings.alpha,
    colour: 'sRGB-encoded display image (ACES fit), matching the selected PNG precision and alpha convention',
  } } : {};
  return {
    mode: settings.mode,
    format: 'png',
    alpha: settings.alpha,
    bits: settings.mode === 'sequence' ? settings.bits : 8,
    colour: composited ? `sRGB-encoded display image (ACES fit) over ${over}, opaque` : `sRGB-encoded display image (ACES fit), alpha ${settings.alpha}`,
    ...(settings.alpha === 'straight' && { note: 'alpha raised to max(rgb) so emissive flame survives' }),
    ...companion,
  };
}

const frameName = (id: string, index: number, extension: string) => `${id}.${String(index + 1).padStart(4, '0')}.${extension}`;

function pngSink(destination: Destination, id: string, settings: Pick<ExportSettings, 'bits' | 'alpha'>): Sink {
  return {
    frame: async (frame) => destination.write(frameName(id, frame.index, 'png'), await encodePng(frame.width, frame.height, settings.bits, toRgba(frame, settings.bits, settings.alpha === 'straight' && 'display'))),
    finish: async () => {},
  };
}

// Matched sinks consume one frame together, so codec/file differences never change the camera, frame count,
// temporal samples or simulation. Complete both operations before cleanup, including after a failure.
function withCompanion(primary: Sink, companion: Sink, plane: 'display' | 'clean', onEncoding: (frame: number) => void): Sink {
  const settle = async (operations: Promise<void>[]) => {
    for (const result of await Promise.allSettled(operations)) if (result.status === 'rejected') throw result.reason;
  };
  return {
    async frame(frame) {
      const clean = frame.planes[plane];
      if (!clean) throw new Error('The transparent companion plane is missing.');
      onEncoding(frame.index);
      await settle([primary.frame(frame), companion.frame({ ...frame, planes: { display: clean } })]);
    },
    finish: () => settle([primary.finish(), companion.finish()]),
    abort: () => settle([primary.abort?.() ?? Promise.resolve(), companion.abort?.() ?? Promise.resolve()]),
  };
}

function exrSink(destination: Destination, id: string, job: OfflineJob, comments: string): Sink {
  const meta = { fps: job.fps, ...cameraMatrices(job.camera, job.width / job.height), comments };
  return {
    frame: async (frame) => destination.write(frameName(id, frame.index, 'exr'), await encodeExr(frame.width, frame.height, toExrChannels(frame), meta)),
    finish: async () => {},
  };
}

// The premultiplied display colour is the picture over black, or over the footage, so the alpha byte is simply ignored.
async function videoSink(destination: Destination, id: string, job: OfflineJob, format: ExportSettings['videoFormat'], alpha: boolean): Promise<Sink> {
  const video = format === 'webm'
    ? await createWebmSink(destination, `${id}.webm`, job.width, job.height, job.fps, alpha)
    : await createMp4Sink(destination, `${id}.mp4`, job.width, job.height, job.fps);
  return { frame: (frame) => video.frame(toRgba(frame, 8, alpha && 'display'), frame.index), finish: video.finish, abort: video.abort };
}

async function videoOutput(destination: Destination, id: string, job: OfflineJob, settings: ExportSettings, composited: boolean, onEncoding: (frame: number) => void): Promise<Sink> {
  const format = settings.videoFormat ?? 'webm';
  const primary = await videoSink(destination, id, job, format, format === 'webm' && !composited);
  try {
    if (format === 'webm') {
      if (!composited) return primary;
      const clean = await videoSink(destination, `${id}-clean`, job, format, true);
      return withCompanion(primary, clean, 'clean', onEncoding);
    }
    const clean = pngSink(destination, `${id}-clean`, { bits: 16, alpha: settings.alpha });
    return withCompanion(primary, clean, composited ? 'clean' : 'display', onEncoding);
  } catch (error) { await primary.abort?.(); throw error; }
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
