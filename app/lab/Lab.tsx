'use client';

/* eslint-disable @next/next/no-img-element */
import { useEffect, useRef, useState } from 'react';
import type { Engine } from '@/engine/engine';
import { loadPlate, releasePlate, type Plate } from '@/engine/plate';
import type { OfflineJob, OfflineProgress, RenderedFrame } from '@/engine/offline-render';
import { defaultTweaks, QUALITIES, RENDER_GRID, type SimulationGrid, type Tweaks } from '@/engine/scene';
import { presetById } from '@/presets/presets';
import type { Destination } from '@/export/files';
import { createMp4Sink } from '@/export/mp4';
import { halves, toRgba } from '@/export/planes';
import { encodePng } from '@/export/png';

const GRIDS: Record<string, SimulationGrid> = {
  quick: { grid: [64, 96, 64], lightScale: 4 },
  standard: { grid: [128, 192, 128], lightScale: 4 },
  production: RENDER_GRID,
};
const ORIGINAL: Partial<Tweaks> = { heat: 1.44, vorticity: 2.4, wind: 32, windYaw: 40, seed: 59, blue: 0.25, sparks: 0.03, smoke: 1.67, width: 1.4 };
const SCENES = ['original-test', 'explosion', 'fire', 'smoke', 'campfire'];
type ImageResult = { name: string; url: string };
type Metrics = {
  hash: string; finite: boolean; alphaMin: number; alphaMax: number; coverage: number; centroidX: number;
  peakLinear: number; peakFirelight: number; peakSmoke: number;
  normalizedTemperature: { p10: number; p50: number; p90: number; max: number; pixels: number; note: string };
};
const button = 'rounded border border-line px-3 py-2 text-sm hover:border-ember disabled:opacity-40';
const select = 'rounded border border-line bg-background px-2 py-1';

// Inspect the actual half-float planes rather than a screenshot or the tone-mapped video.
function inspect(frame: RenderedFrame): Metrics {
  const table = halves();
  const fire = frame.planes.fire!;
  let hash = 2166136261;
  let finite = true;
  let alphaMin = Infinity;
  let alphaMax = -Infinity;
  let mass = 0;
  let weightedX = 0;
  let peakLinear = 0;
  let peakFirelight = 0;
  let peakSmoke = 0;
  let heatMax = 0;
  let heatCount = 0;
  const heatBins = new Uint32Array(1024);
  const light = frame.planes.firelight!;
  for (const [name, plane] of Object.entries(frame.planes)) {
    for (let y = 0; y < frame.height; y++) for (let x = 0; x < frame.width * 4; x++) {
      const bits = plane![y * frame.stride + x];
      hash = Math.imul(hash ^ bits, 16777619);
      const value = table[bits];
      if (!Number.isFinite(value)) finite = false;
      if ((x & 3) !== 3) {
        if (name === 'fire') peakLinear = Math.max(peakLinear, value);
        if (name === 'firelight') peakFirelight = Math.max(peakFirelight, value);
        if (name === 'smoke') peakSmoke = Math.max(peakSmoke, value);
      }
    }
  }
  for (let y = 0; y < frame.height; y++) for (let x = 0; x < frame.width; x++) {
    const alpha = table[fire[y * frame.stride + x * 4 + 3]];
    alphaMin = Math.min(alphaMin, alpha);
    alphaMax = Math.max(alphaMax, alpha);
    mass += alpha;
    weightedX += x * alpha;
    const heat = table[light[y * frame.stride + x * 4 + 3]];
    if (alpha > 0.01 && Number.isFinite(heat)) {
      heatMax = Math.max(heatMax, heat);
      heatBins[Math.min(1023, Math.floor(Math.max(0, heat) * 64))]++;
      heatCount++;
    }
  }
  const quantile = (fraction: number) => {
    if (!heatCount) return 0;
    let count = 0;
    for (let i = 0; i < heatBins.length; i++) { count += heatBins[i]; if (count >= fraction * heatCount) return (i + 0.5) / 64; }
    return heatMax;
  };
  return { hash: (hash >>> 0).toString(16).padStart(8, '0'), finite, alphaMin, alphaMax, coverage: mass / (frame.width * frame.height), centroidX: mass > 0 ? weightedX / mass : 0, peakLinear, peakFirelight, peakSmoke,
    normalizedTemperature: { p10: quantile(0.1), p50: quantile(0.5), p90: quantile(0.9), max: heatMax, pixels: heatCount,
      note: 'Maximum normalized heat along each ray, averaged over samples, in pixels with extinction alpha > 0.01. Percentiles use 1/64-unit bins; this is not Kelvin.' } };
}

export default function Lab() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const controller = useRef<AbortController | null>(null);
  const urls = useRef<string[]>([]);
  const plateRef = useRef<Plate | null>(null);
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [scene, setScene] = useState('original-test');
  const [grid, setGrid] = useState('quick');
  const [resolution, setResolution] = useState('640x360');
  const [samples, setSamples] = useState(4);
  const [shutter, setShutter] = useState(0.5);
  const [warmup, setWarmup] = useState(0.35);
  const [wind, setWind] = useState(32);
  const [exposure, setExposure] = useState(0);
  const [brightness, setBrightness] = useState(0);
  const [temperature, setTemperature] = useState(1900);
  const [heat, setHeat] = useState(1.44);
  const [sootYield, setSootYield] = useState(1.67);
  const [fuel, setFuel] = useState(1);
  const [positionX, setPositionX] = useState(0);
  const [positionY, setPositionY] = useState(0);
  const [scale, setScale] = useState(1);
  const [plateName, setPlateName] = useState('');
  const [progress, setProgress] = useState('Starting GPU');
  const [report, setReport] = useState<unknown>(null);
  const [images, setImages] = useState<ImageResult[]>([]);
  const [video, setVideo] = useState('');
  const [videoExtension, setVideoExtension] = useState('mp4');
  const [matchedCleanVideo, setMatchedCleanVideo] = useState('');
  const [sidecar, setSidecar] = useState('');
  const [error, setError] = useState('');
  const url = (blob: Blob) => { const value = URL.createObjectURL(blob); urls.current.push(value); return value; };

  useEffect(() => {
    const generatedUrls = urls.current;
    const targetCanvas = canvas.current!;
    let disposed = false;
    import('@/engine/engine').then(({ createEngine }) => disposed ? null : createEngine(targetCanvas, presetById('explosion'), QUALITIES.fast)).then((engine) => {
      if (!engine) return;
      if (disposed) return engine.dispose();
      engineRef.current = engine;
      engine.setTweaks(ORIGINAL);
      setReady(true);
      setProgress(`Ready · ${engine.stats().adapterName}`);
      engine.lost.then((info) => { if (!disposed) { setReady(false); setError(`GPU device lost: ${info.message}`); } });
    }).catch((failure) => { if (!disposed) setError(String(failure)); });
    return () => {
      disposed = true;
      controller.current?.abort();
      engineRef.current?.dispose();
      if (plateRef.current) releasePlate(plateRef.current);
      generatedUrls.forEach(URL.revokeObjectURL);
    };
  }, []);

  function configure(id = scene, override: Partial<Tweaks> = {}) {
    const engine = engineRef.current!;
    const preset = presetById(id === 'original-test' ? 'explosion' : id);
    const tweaks = { ...defaultTweaks(preset), seed: 59, ...(id === 'original-test' ? ORIGINAL : {}), wind, exposure, brightness, temperature, heat, smoke: sootYield, fuel, ...override };
    engine.setPreset(preset);
    engine.setTweaks(tweaks);
    engine.setFraming({ x: positionX, y: positionY, scale });
    return engine;
  }

  function job(engine: Engine, frames: number, secondsBefore = 0): OfflineJob {
    const [width, height] = resolution.split('x').map(Number);
    return { width, height, fps: 30, frames, warmup: secondsBefore, stride: 1, samples, camera: engine.snapshot().camera, loop: false, shutter,
      simulation: GRIDS[grid], read: ['display', 'fire', 'firelight', 'smoke'], plate: Boolean(plateRef.current), holdout: false };
  }

  async function work(task: (signal: AbortSignal) => Promise<void>) {
    if (!engineRef.current || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true); setError(''); setReport(null); setMatchedCleanVideo('');
    try { await task(abort.signal); }
    catch (failure) { setError(abort.signal.aborted ? 'Cancelled' : failure instanceof Error ? failure.message : String(failure)); }
    finally { setBusy(false); controller.current = null; }
  }

  async function capture(frame: RenderedFrame, name: string) {
    // A contact frame is the actual display image over black, matching the MP4. Opaque alpha avoids
    // image viewers showing the straight RGB values of almost transparent sparks at full brightness.
    const rgba = toRgba(frame, 8, false);
    for (let i = 3; i < rgba.length; i += 4) rgba[i] = 255;
    const png = await encodePng(frame.width, frame.height, 8, rgba);
    const result = { name, url: url(new Blob([png], { type: 'image/png' })) };
    setImages((current) => [...current, result]);
  }

  async function still() {
    await work(async (signal) => {
      setImages([]); setVideo('');
      const engine = configure();
      const snapshot = engine.snapshot();
      const renderJob = job(engine, 1, warmup);
      setProgress(`Rendering still at ${warmup.toFixed(2)} s`);
      let metrics: Metrics | undefined;
      await engine.render(renderJob, async (frame) => { metrics = inspect(frame); await capture(frame, `${scene}-${warmup.toFixed(2)}s`); }, signal);
      setReport({ metrics, snapshot, job: renderJob, domain: engine.renderInfo() });
      setProgress('Still complete');
    });
  }

  async function clip(seconds: number) {
    await work(async (signal) => {
      setVideoExtension('mp4');
      setImages([]); setVideo(''); setSidecar('');
      const engine = configure();
      const snapshot = engine.snapshot();
      const renderJob = job(engine, seconds * 30, scene === 'fire' || scene === 'campfire' ? 1 : 0);
      let videoUrl = '';
      const destination: Destination = {
        name: 'Lab memory',
        async write(_file, bytes) { videoUrl = url(new Blob([bytes], { type: 'video/mp4' })); },
        async finish() {}, async discard() {},
      };
      const sink = await createMp4Sink(destination, 'lab.mp4', renderJob.width, renderJob.height, 30);
      const checkpoints = new Set([0, 3, 9, 20, 35, 59, 89, 119, 149].filter((i) => i < renderJob.frames));
      let finite = true;
      let alphaBounded = true;
      let maxLinear = 0;
      let maxFirelight = 0;
      let maxSmoke = 0;
      const contactMetrics: { frame: number; metrics: Metrics }[] = [];
      const started = performance.now();
      try {
        await engine.render(renderJob, async (frame) => {
          signal.throwIfAborted();
          const metrics = inspect(frame);
          finite &&= metrics.finite;
          alphaBounded &&= metrics.alphaMin >= -0.001 && metrics.alphaMax <= 1.001;
          maxLinear = Math.max(maxLinear, metrics.peakLinear);
          maxFirelight = Math.max(maxFirelight, metrics.peakFirelight);
          maxSmoke = Math.max(maxSmoke, metrics.peakSmoke);
          await sink.frame(toRgba(frame, 8, false), frame.index);
          if (checkpoints.has(frame.index)) { contactMetrics.push({ frame: frame.index, metrics }); await capture(frame, `${scene}-${((frame.index + 1) / 30).toFixed(2)}s`); }
          setProgress(`Frame ${frame.index + 1} / ${renderJob.frames} · ${((performance.now() - started) / 1000).toFixed(1)} s elapsed`);
        }, signal);
        await sink.finish();
        setVideo(videoUrl);
        const summary = { scene, seed: 59, job: renderJob, snapshot, domain: engine.renderInfo(), finite, alphaBounded, maxLinear, maxFirelight, maxSmoke, contactMetrics, elapsedSeconds: (performance.now() - started) / 1000 };
        setReport(summary);
        setSidecar(url(new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' })));
        setProgress(`Completed ${seconds} s MP4 · ${renderJob.width} × ${renderJob.height}`);
      } catch (failure) { await sink.abort(); throw failure; }
    });
  }

  async function checks() {
    await work(async (signal) => {
      setImages([]); setVideo('');
      const results: Record<string, unknown> = {};
      for (const preset of ['smoke', 'fire', 'explosion']) {
        const hashes: string[] = [];
        for (let repeat = 0; repeat < 2; repeat++) {
          setProgress(`Checking ${preset} · repeat ${repeat + 1} / 2`);
          const engine = configure(preset, { ...defaultTweaks(presetById(preset)), seed: 59, wind: 0, windGust: 0 });
          await engine.render(job(engine, 1, preset === 'explosion' ? 0.3 : 1), async (frame) => {
            const metrics = inspect(frame); hashes.push(metrics.hash);
            if (repeat === 0) { results[preset] = metrics; await capture(frame, `${preset}-check`); }
          }, signal);
        }
        results[`${preset}Deterministic`] = hashes[0] === hashes[1];
        setReport({ ...results });
      }
      const centroids: Record<string, number> = {};
      for (const speed of [-40, 0, 40]) {
        setProgress(`Checking wind ${speed} cell/s`);
        const engine = configure('fire', { ...defaultTweaks(presetById('fire')), seed: 59, wind: speed, windYaw: 0, windGust: 0 });
        await engine.render(job(engine, 1, 1.5), async (frame) => { centroids[String(speed)] = inspect(frame).centroidX; await capture(frame, `wind-${speed}`); }, signal);
      }
      results.windCentroidPixels = centroids;
      results.windSeparationPixels = Math.abs(centroids['40'] - centroids['-40']);
      results.windMovesOppositeDirections = (centroids['40'] - centroids['0']) * (centroids['-40'] - centroids['0']) < 0;
      for (const temperature of [1000, 3000]) {
        setProgress(`Checking appearance limits · ${temperature} K`);
        const engine = configure('explosion', { ...defaultTweaks(presetById('explosion')), seed: 59, wind: 0,
          heat: 2, fuel: 2.5, sparks: 1, sparkSize: 3, sparkLife: 3, temperature, exposure: 5 });
        await engine.render(job(engine, 1, 0.3), async (frame) => {
          results[`appearance${temperature}K`] = inspect(frame);
          await capture(frame, `appearance-${temperature}K`);
        }, signal);
      }
      const stable = ['smoke', 'fire', 'explosion'].every((id) => results[`${id}Deterministic`] === true);
      const metrics = Object.values(results).filter((value): value is Metrics => typeof value === 'object' && value !== null && 'finite' in value);
      results.pass = stable && results.windMovesOppositeDirections === true && metrics.every((m) => m.finite && m.alphaMin >= 0 && m.alphaMax <= 1.001);
      setReport(results); setProgress(results.pass ? 'Render regression checks passed' : 'Render regression checks found a failure');
    });
  }

  async function checkExportProgress() {
    await work(async (signal) => {
      setImages([]); setVideo('');
      const results: Record<string, unknown> = {};
      const run = async (seconds: number, cancel: boolean) => {
        signal.throwIfAborted();
        const engine = configure('fire', { ...defaultTweaks(presetById('fire')), seed: 59 });
        const renderJob: OfflineJob = { ...job(engine, 1, seconds), width: 16, height: 16, samples: 1,
          simulation: GRIDS.quick, plate: false, holdout: false, loop: false };
        const stop = new AbortController();
        const forwardAbort = () => stop.abort(signal.reason);
        signal.addEventListener('abort', forwardAbort, { once: true });
        const phases: string[] = [];
        const warmup: OfflineProgress[] = [];
        let delivered = false;
        let finite = false;
        let aborted = false;
        let requestedAt = 0;
        const started = performance.now();
        try {
          await engine.render(renderJob, async (frame) => { delivered = true; finite = inspect(frame).finite; }, stop.signal, (state) => {
            if (phases[phases.length - 1] !== state.phase) phases.push(state.phase);
            if (state.phase === 'warmup') warmup.push(state);
            setProgress(`${cancel ? 'Cancel check' : `${seconds} s pre-roll benchmark`} · ${state.phase} · ${state.completed.toFixed(1)} / ${state.total}`);
            if (cancel && state.phase === 'warmup' && state.completed >= 4 && !requestedAt) {
              requestedAt = performance.now();
              stop.abort();
            }
          });
        } catch (failure) {
          if (!stop.signal.aborted || signal.aborted) throw failure;
          aborted = true;
        } finally { signal.removeEventListener('abort', forwardAbort); }
        signal.throwIfAborted();
        const elapsedMs = performance.now() - started;
        const last = warmup[warmup.length - 1];
        const monotone = warmup.every((state, i) => !i || state.completed >= warmup[i - 1].completed);
        return { seconds, phases, elapsedMs, warmupElapsedMs: last?.elapsedMs ?? null,
          warmupCompletedSteps: last?.completed ?? 0, warmupTotalSteps: last?.total ?? 0,
          warmupMonotone: monotone, delivered, finite, aborted,
          cancelLatencyMs: requestedAt ? performance.now() - requestedAt : null,
          pass: monotone && (cancel ? aborted && !delivered && requestedAt > 0 : delivered && finite && warmup.length > 1 && last?.completed === last?.total),
        };
      };
      const full = await run(6, false);
      results.preroll6 = full; setReport({ ...results });
      const short = await run(2, false);
      results.preroll2 = short; setReport({ ...results });
      const cancelled = await run(6, true);
      results.cancelWarmup = cancelled;
      results.shorterPrerollMeasuredRatio = short.elapsedMs > 0 ? full.elapsedMs / short.elapsedMs : null;
      results.note = 'Both renders start from cleared fields and use the real solver at 64 cells per base unit with adaptive bounds, 16 × 16 output, one frame and one sample. Shorter pre-roll is a different starting state, not a quality-preserving cache.';
      results.pass = full.pass && short.pass && cancelled.pass;
      setReport(results);
      setProgress(results.pass ? 'Export progress and cancellation checks passed' : 'Export progress checks found a failure');
    });
  }

  async function checkCleanCompanion() {
    await work(async (signal) => {
      setImages([]); setVideo('');
      const engine = configure('explosion', { ...defaultTweaks(presetById('explosion')), seed: 59 });
      const originalPlate = plateRef.current;
      const originalFraming = engine.snapshot().framing;
      const surface = document.createElement('canvas'); surface.width = 128; surface.height = 72;
      const paint = surface.getContext('2d')!; paint.fillStyle = '#b2b8bf'; paint.fillRect(0, 0, 128, 72);
      const fixture: Plate = { name: 'Dual-output fixture', width: 128, height: 72, source: await createImageBitmap(surface) };
      const renderJob: OfflineJob = { ...job(engine, 1, 0.35), width: 128, height: 72, samples: 4,
        simulation: GRIDS.quick, read: ['display', 'clean', 'fire', 'firelight', 'smoke'], plate: true, holdout: false };
      let cleanCopy: Uint16Array | undefined;
      let compositeAlphaMin = 1;
      let cleanAlphaMin = 1;
      let cleanAlphaMax = 0;
      let finite = true;
      let standaloneMatches = false;
      const hash = (plane: Uint16Array) => {
        let value = 2166136261;
        for (const bits of plane) value = Math.imul(value ^ bits, 16777619);
        return (value >>> 0).toString(16).padStart(8, '0');
      };
      let cleanHash = '';
      let standaloneHash = '';
      try {
        await engine.setPlate(fixture); engine.setFraming({ x: 0, y: 0, scale: 1 });
        renderJob.camera = engine.snapshot().camera;
        setProgress('Rendering composite and clean planes in one GPU render');
        await engine.render(renderJob, async (frame) => {
          cleanCopy = frame.planes.clean!.slice();
          cleanHash = hash(cleanCopy);
          const values = halves();
          for (let y = 0; y < frame.height; y++) for (let x = 0; x < frame.width; x++) {
            const at = y * frame.stride + 4 * x;
            compositeAlphaMin = Math.min(compositeAlphaMin, values[frame.planes.display![at + 3]]);
            const alpha = values[cleanCopy[at + 3]];
            cleanAlphaMin = Math.min(cleanAlphaMin, alpha); cleanAlphaMax = Math.max(cleanAlphaMax, alpha);
          }
          finite = inspect(frame).finite;
          await capture(frame, 'dual-composite');
          const transparent: RenderedFrame = { ...frame, planes: { display: cleanCopy } };
          const png = await encodePng(frame.width, frame.height, 8, toRgba(transparent, 8, 'display'));
          setImages((current) => [...current, { name: 'dual-clean-transparent', url: url(new Blob([png], { type: 'image/png' })) }]);
        }, signal);
        setProgress('Checking clean plane against a standalone reference render');
        // The second pass is a diagnostic reference only. The production export writes both deliverables from
        // the single dual-plane pass above; it never repeats this simulation or raymarch for the companion.
        await engine.render({ ...renderJob, plate: false, read: ['display'] }, async (frame) => {
          standaloneHash = hash(frame.planes.display!);
          standaloneMatches = Boolean(cleanCopy) && standaloneHash === cleanHash;
        }, signal);
        const benchmark = async (dual: boolean) => {
          const started = performance.now();
          let encodedBytes = 0;
          const memory: Destination = { name: 'Companion benchmark memory', async write(_name, bytes) { encodedBytes += bytes.length; }, async finish() {}, async discard() {} };
          const encoder = await createMp4Sink(memory, 'benchmark.mp4', 640, 360, 30);
          const task: OfflineJob = { ...renderJob, width: 640, height: 360, frames: 6, read: dual ? ['display', 'clean'] : ['display'] };
          try {
            await engine.render(task, async (frame) => {
              signal.throwIfAborted();
              const clean = async () => {
                const transparent: RenderedFrame = { ...frame, planes: { display: frame.planes.clean! } };
                const png = await encodePng(frame.width, frame.height, 8, toRgba(transparent, 8, 'display'));
                encodedBytes += png.length;
              };
              const pending = [encoder.frame(toRgba(frame, 8, false), frame.index)];
              if (dual) pending.push(clean());
              for (const result of await Promise.allSettled(pending)) if (result.status === 'rejected') throw result.reason;
            }, signal, (state) => setProgress(`${dual ? 'Composite + transparent PNG' : 'Composite only'} benchmark · ${state.phase} · ${state.completed.toFixed(1)} / ${state.total}`));
            await encoder.finish();
          } catch (failure) { await encoder.abort(); throw failure; }
          return { elapsedMs: performance.now() - started, encodedBytes, work: engine.renderInfo()?.work ?? null };
        };
        const primaryBenchmark = await benchmark(false);
        const pairedBenchmark = await benchmark(true);
        const samePhysicalWork = Boolean(primaryBenchmark.work && pairedBenchmark.work)
          && primaryBenchmark.work?.simulationSteps === pairedBenchmark.work?.simulationSteps
          && primaryBenchmark.work?.volumeSamples === pairedBenchmark.work?.volumeSamples;
        const overheadFactor = pairedBenchmark.elapsedMs / primaryBenchmark.elapsedMs;
        const pass = finite && compositeAlphaMin > 0.999 && cleanAlphaMin === 0 && cleanAlphaMax > 0.01 && standaloneMatches && samePhysicalWork;
        setReport({ pass, finite, compositeAlphaMin, cleanAlphaMin, cleanAlphaMax, cleanHash, standaloneHash, standaloneMatches,
          dualOutputPhysicalPasses: 1, standaloneReferencePasses: 1,
          performance: { primaryBenchmark, pairedBenchmark, samePhysicalWork, overheadFactor,
            under2xInThisBenchmark: overheadFactor < 2,
            note: 'Six 640 × 360 frames, four samples, quick 64-cell density. Timings include actual H.264 finalization and transparent PNG encoding into memory. Primary then paired run; browser/cache timing varies. PNG compression and storage remain additional work.' },
          note: 'Composite and transparent clean planes share one set of volume/spark samples. A separate reference pass is used only by this diagnostic to verify the clean image. Clean output contains the full effect without footage holdouts.' });
        setProgress(pass ? 'Clean companion checks passed' : 'Clean companion checks found a failure');
      } finally { await engine.setPlate(originalPlate); engine.setFraming(originalFraming); releasePlate(fixture); }
    });
  }

  async function domainDiagnostics() {
    await work(async (signal) => {
      setProgress('Running adaptive-domain field preservation diagnostics');
      signal.throwIfAborted();
      const { runDomainDiagnostics } = await import('@/engine/domain-diagnostics');
      const result = await runDomainDiagnostics(signal);
      signal.throwIfAborted();
      setReport(result);
      setProgress(result.pass ? 'GPU domain diagnostics passed' : 'GPU domain diagnostics found a failure');
    });
  }

  async function checkMatchedVideo() {
    await work(async (signal) => {
      setImages([]); setVideo(''); setSidecar('');
      const { runMatchedVideoCheck } = await import('./matched-video-check');
      const result = await runMatchedVideoCheck(engineRef.current!, plateRef.current, signal, setProgress);
      setReport(result.report);
      const composite = result.files.get('matched-composite.webm');
      const transparent = result.files.get('matched-clean.webm');
      const decoded = result.files.get('matched-clean-decoded.png');
      if (composite) { setVideoExtension('webm'); setVideo(url(new Blob([composite], { type: 'video/webm' }))); }
      if (transparent) setMatchedCleanVideo(url(new Blob([transparent], { type: 'video/webm' })));
      if (decoded) setImages([{ name: 'Decoded WebM alpha', url: url(new Blob([decoded], { type: 'image/png' })) }]);
      setProgress(result.report.pass ? 'Matched WebM encoding and alpha checks passed' : 'Matched WebM check unavailable or failed');
    });
  }

  async function flowDiagnostics() {
    await work(async (signal) => {
      setProgress('Running velocity, solid-boundary and maximum-control GPU diagnostics');
      signal.throwIfAborted();
      const { runFlowDiagnostics } = await import('@/engine/lab-diagnostics');
      const result = await runFlowDiagnostics(signal);
      signal.throwIfAborted();
      setReport(result);
      setProgress(result.pass ? 'GPU flow diagnostics passed' : 'GPU flow diagnostics found a failure');
    });
  }

  async function loadTestPlateFixture() {
    // A deterministic fixture exercises the real plate upload/composite path without a native file picker.
    const surface = document.createElement('canvas');
    surface.width = 640; surface.height = 360;
    const context = surface.getContext('2d')!;
    context.fillStyle = '#b2b8bf'; context.fillRect(0, 0, surface.width, surface.height);
    const blob = await new Promise<Blob>((resolve, reject) => surface.toBlob((value) => value ? resolve(value) : reject(new Error('Could not create the test plate.')), 'image/png'));
    await openPlate(new File([blob], 'Neutral test plate.png', { type: 'image/png' }));
  }

  function removeTestPlate() {
    void engineRef.current?.setPlate(null);
    if (plateRef.current) releasePlate(plateRef.current);
    plateRef.current = null; setPlateName('');
  }

  function downloadReport() {
    const link = document.createElement('a');
    link.href = url(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    link.download = 'lab-report.json';
    link.click();
  }

  async function openPlate(file?: File) {
    if (!file) return;
    try {
      const plate = await loadPlate(file);
      await engineRef.current?.setPlate(plate);
      if (plateRef.current) releasePlate(plateRef.current);
      plateRef.current = plate; setPlateName(plate.name);
    } catch (failure) { setError(String(failure)); }
  }

  return <main className="mx-auto max-w-[1440px] p-5 text-foreground">
    <h1 className="text-xl font-medium">Voleryn GPU lab · development only</h1>
    <p className="mt-1 text-sm text-muted">Real engine, fixed 30 fps, seed 59. Outputs use the selected simulation grid and offline renderer. Production grid can take several minutes.</p>
    <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div>
        <div className="relative aspect-video overflow-hidden border border-line bg-black"><canvas ref={canvas} aria-label="Lab engine preview" className="h-full w-full" /></div>
        {video && <div className="mt-3"><video controls loop src={video} className="w-full bg-black" /><a href={video} download={`voleryn-${scene}-seed59.${videoExtension}`} className="mt-2 inline-block underline">Download {videoExtension.toUpperCase()}</a>{sidecar && <a href={sidecar} download="lab-render.json" className="ml-4 underline">Download render metadata</a>}</div>}
        {matchedCleanVideo && <div className="mt-3"><video controls loop src={matchedCleanVideo} className="w-full bg-[repeating-conic-gradient(#303030_0_25%,#151515_0_50%)] bg-[length:20px_20px]" /><a href={matchedCleanVideo} download="matched-clean.webm" className="mt-2 inline-block underline">Download transparent WebM</a></div>}
        <p role="status" className="mt-3 font-mono text-xs">{progress}</p>
        {error && <p role="alert" className="mt-2 text-red-400">{error}</p>}
      </div>
      <fieldset disabled={!ready || busy} className="flex flex-col gap-3 text-sm">
        <label className="flex justify-between gap-2">Scene<select aria-label="Scene" className={select} value={scene} onChange={(e) => { setScene(e.target.value); setWind(e.target.value === 'original-test' ? 32 : 0); setHeat(e.target.value === 'original-test' ? 1.44 : 1); setSootYield(e.target.value === 'original-test' ? 1.67 : 1); setFuel(1); setTemperature(defaultTweaks(presetById(e.target.value === 'original-test' ? 'explosion' : e.target.value)).temperature); setBrightness(0); }}>{SCENES.map((id) => <option key={id}>{id}</option>)}</select></label>
        <label className="flex justify-between gap-2">Simulation grid<select aria-label="Simulation grid" className={select} value={grid} onChange={(e) => setGrid(e.target.value)}>{Object.entries(GRIDS).map(([id, value]) => <option key={id} value={id}>{id} · {value.grid.join(' × ')}</option>)}</select></label>
        <label className="flex justify-between gap-2">Resolution<select aria-label="Resolution" className={select} value={resolution} onChange={(e) => setResolution(e.target.value)}>{['640x360', '1280x720', '1920x1080', '3840x2160'].map((id) => <option key={id}>{id}</option>)}</select></label>
        <label className="flex justify-between gap-2">Samples<select aria-label="Samples" className={select} value={samples} onChange={(e) => setSamples(Number(e.target.value))}>{[1, 4, 16, 32].map((n) => <option key={n}>{n}</option>)}</select></label>
        <label className="flex justify-between gap-2">Shutter<select aria-label="Shutter" className={select} value={shutter} onChange={(e) => setShutter(Number(e.target.value))}><option value={0}>Off</option><option value={0.5}>180°</option></select></label>
        <label className="flex justify-between gap-2">Still time (s)<input aria-label="Still time" type="number" min={0} max={5} step={0.05} value={warmup} onChange={(e) => setWarmup(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Wind (cell/s)<input aria-label="Wind speed" type="number" min={-60} max={60} value={wind} onChange={(e) => setWind(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Exposure (EV)<input aria-label="Exposure" type="number" min={-5} max={5} step={0.25} value={exposure} onChange={(e) => setExposure(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Brightness (EV)<input aria-label="Brightness" type="number" min={-5} max={5} step={0.25} value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Temperature (K)<input aria-label="Temperature" type="number" min={1000} max={3000} step={10} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Heat release (×)<input aria-label="Heat release" type="number" min={0.4} max={2} step={0.05} value={heat} onChange={(e) => setHeat(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Soot yield (×)<input aria-label="Soot yield" type="number" min={0} max={2.5} step={0.05} value={sootYield} onChange={(e) => setSootYield(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex justify-between gap-2">Fuel amount (×)<input aria-label="Fuel amount" type="number" min={0.1} max={2.5} step={0.05} value={fuel} onChange={(e) => setFuel(Number(e.target.value))} className={`${select} w-24`} /></label>
        <label className="flex flex-col gap-1">Optional footage<input aria-label="Footage" type="file" accept="image/*,video/*" onChange={(e) => void openPlate(e.target.files?.[0])} /></label>
        <div className="flex gap-2"><button className={button} onClick={() => void loadTestPlateFixture()}>Use test plate</button>{plateName && <button className={button} onClick={removeTestPlate}>Remove plate</button>}</div>
        {plateName && <p className="text-xs text-muted">{plateName} · composited, no depth holdout in lab</p>}
        {(['x', 'y', 'scale'] as const).map((axis) => <label key={axis} className="flex justify-between gap-2">Placement {axis}<input aria-label={`Placement ${axis}`} type="number" min={axis === 'scale' ? 0.2 : -1} max={axis === 'scale' ? 3 : 1} step={0.05} value={axis === 'x' ? positionX : axis === 'y' ? positionY : scale} onChange={(e) => (axis === 'x' ? setPositionX : axis === 'y' ? setPositionY : setScale)(Number(e.target.value))} className={`${select} w-24`} /></label>)}
        <div className="grid grid-cols-2 gap-2"><button className={button} onClick={() => void still()}>Render still</button><button className={button} onClick={() => void checks()}>Run checks</button><button className={button} onClick={() => void flowDiagnostics()}>Flow diagnostics</button><button className={button} onClick={() => void checkExportProgress()}>Check export progress</button><button className={button} onClick={() => void domainDiagnostics()}>Domain diagnostics</button><button className={button} onClick={() => void checkCleanCompanion()}>Check clean companion</button><button className={button} onClick={() => void checkMatchedVideo()}>Check matched video</button><button className={button} onClick={() => void clip(2)}>Render 2 s</button><button className={button} onClick={() => void clip(5)}>Render 5 s</button></div>
      </fieldset>
    </div>
    {busy && <button onClick={() => controller.current?.abort()} className={`${button} mt-3`}>Cancel render</button>}
    {images.length > 0 && <section className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-3">{images.map((image) => <figure key={image.url}><img src={image.url} alt={image.name} className="w-full border border-line bg-black" /><figcaption className="mt-1 text-xs"><a href={image.url} download={`${image.name}.png`} className="underline">{image.name} · PNG</a></figcaption></figure>)}</section>}
    {report !== null && <section className="mt-5">
      <button className={button} onClick={downloadReport}>Download validation report</button>
      <pre className="mt-2 max-h-[480px] overflow-auto border border-line p-3 text-xs" aria-label="Validation report">{JSON.stringify(report, null, 2)}</pre>
    </section>}
  </main>;
}
