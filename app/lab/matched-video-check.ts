import type { Engine } from '@/engine/engine';
import type { Plate } from '@/engine/plate';
import { releasePlate } from '@/engine/plate';
import { defaultTweaks } from '@/engine/scene';
import { presetById } from '@/presets/presets';
import type { Destination } from '@/export/files';
import { createWebmSink, WebmEncodingUnavailable } from '@/export/webm';
import { videoBitrateFor } from '@/export/mp4';
import { toRgba } from '@/export/planes';
import { encodePng } from '@/export/png';

// Development-only, genuine codec/GPU exercise. Four shared GPU frames create the two native VP9 files;
// the resulting container, alpha packets and decoded pixels are inspected with the actual media library.
export async function runMatchedVideoCheck(engine: Engine, originalPlate: Plate | null, signal: AbortSignal, progress: (message: string) => void) {
  const width = 160, height = 90, fps = 30, frames = 4;
  const files = new Map<string, Uint8Array<ArrayBuffer>>();
  const memory: Destination = { name: 'Matched WebM diagnostic', async write(name, bytes) { files.set(name, bytes); }, async finish() {}, async discard() { files.clear(); } };
  const surface = document.createElement('canvas'); surface.width = width; surface.height = height;
  const context = surface.getContext('2d')!; context.fillStyle = '#406080'; context.fillRect(0, 0, width, height);
  const fixture: Plate = { name: 'Matched video fixture', width, height, source: await createImageBitmap(surface) };
  const framing = engine.snapshot().framing;
  let primary: Awaited<ReturnType<typeof createWebmSink>> | undefined;
  let clean: Awaited<ReturnType<typeof createWebmSink>> | undefined;
  try {
    signal.throwIfAborted();
    progress('Checking actual VP9 and alpha encoding support');
    primary = await createWebmSink(memory, 'matched-composite.webm', width, height, fps, false);
    clean = await createWebmSink(memory, 'matched-clean.webm', width, height, fps, true);
    const preset = presetById('explosion'); engine.setPreset(preset); engine.setTweaks({ ...defaultTweaks(preset), seed: 59 });
    engine.setFraming({ x: 0, y: 0, scale: 1 }); await engine.setPlate(fixture);
    const started = performance.now();
    await engine.render({ width, height, fps, frames, warmup: 0.35, stride: 1, samples: 4, loop: false, shutter: 0.5,
      camera: engine.snapshot().camera, simulation: { grid: [64, 96, 64], lightScale: 4 }, read: ['display', 'clean'], plate: true, holdout: false },
    async (frame) => {
      signal.throwIfAborted();
      const transparent = { ...frame, planes: { display: frame.planes.clean! } };
      for (const result of await Promise.allSettled([
        primary!.frame(toRgba(frame, 8, false), frame.index), clean!.frame(toRgba(transparent, 8, 'display'), frame.index),
      ])) if (result.status === 'rejected') throw result.reason;
    }, signal, (state) => progress(`Matched WebM · ${state.phase} · ${state.completed.toFixed(1)} / ${state.total}`));
    for (const result of await Promise.allSettled([primary.finish(), clean.finish()])) if (result.status === 'rejected') throw result.reason;
    const work = engine.renderInfo()?.work;
    const { Input, BlobSource, ALL_FORMATS, VideoSampleSink, EncodedPacketSink } = await import('mediabunny');
    const decoded = [];
    for (const name of ['matched-composite.webm', 'matched-clean.webm']) {
      signal.throwIfAborted(); progress(`Inspecting and decoding ${name}`);
      const bytes = files.get(name)!;
      const input = new Input({ source: new BlobSource(new Blob([bytes], { type: 'video/webm' })), formats: ALL_FORMATS });
      try {
        const track = await input.getPrimaryVideoTrack();
        if (!track) throw new Error(`${name} has no video track.`);
        const packets = new EncodedPacketSink(track);
        const packet = await packets.getFirstPacket();
        const timestamps: number[] = [];
        for await (const encoded of packets.packets()) timestamps.push(encoded.timestamp);
        timestamps.sort((a, b) => a - b);
        const sample = await new VideoSampleSink(track).getSample(2 / fps);
        if (!sample) throw new Error(`${name} has no decoded sample.`);
        let minAlpha = 255, maxAlpha = 0, softPixels = 0;
        try {
          const pixels = new Uint8Array(sample.allocationSize({ format: 'RGBA' }));
          await sample.copyTo(pixels, { format: 'RGBA' });
          for (let at = 3; at < pixels.length; at += 4) {
            minAlpha = Math.min(minAlpha, pixels[at]); maxAlpha = Math.max(maxAlpha, pixels[at]);
            if (pixels[at] > 0 && pixels[at] < 255) softPixels++;
          }
          if (name.includes('clean')) files.set('matched-clean-decoded.png', await encodePng(width, height, 8, pixels));
        } finally { sample.close(); }
        decoded.push({ name, bytes: bytes.length, width: await track.getDisplayWidth(), height: await track.getDisplayHeight(),
          packets: timestamps.length, timestamps, codec: await track.getCodecParameterString(),
          markedTransparent: await track.canBeTransparent(), firstAlphaPacketBytes: packet?.sideData.alpha?.length ?? 0,
          minAlpha, maxAlpha, softPixels });
      } finally { input.dispose(); }
    }
    const [composite, transparent] = decoded;
    const sameTiming = composite.timestamps.length === frames
      && transparent.timestamps.length === frames
      && composite.timestamps.every((time, index) => time === transparent.timestamps[index] && Math.abs(time - index / fps) < 0.001);
    const pass = sameTiming && decoded.every((item) => item.width === width && item.height === height && item.codec === composite.codec)
      && composite.minAlpha === 255 && transparent.markedTransparent && transparent.firstAlphaPacketBytes > 0
      && transparent.minAlpha === 0 && transparent.maxAlpha > 4 && transparent.softPixels > 0
      && work?.simulationSteps === 29 && work.volumeSamples === 16 && work.displayImages === 8;
    return { report: { pass, unsupported: false, selected: { width, height, fps, frames, codec: 'vp9', bitrate: videoBitrateFor(width, height, fps), samples: 4, shutter: 0.5 },
      work, decoded, sameTiming, elapsedMs: performance.now() - started, note: 'One real GPU render; two native VP9 files with matching colour encoding settings. Alpha was verified by decoding the exported file.' }, files };
  } catch (error) {
    await Promise.allSettled([primary?.abort(), clean?.abort()]);
    if (error instanceof WebmEncodingUnavailable) return { report: { pass: false, unsupported: true, message: error.message }, files };
    throw error;
  } finally { await engine.setPlate(originalPlate); engine.setFraming(framing); releasePlate(fixture); }
}
