import type { Destination } from './files';

// H.264 in an MP4 through WebCodecs and mediabunny, which is loaded here on the first video export only, so the
// studio never pays for it otherwise. Frames stream into the picked folder; the zip path collects them in memory.
export async function createMp4Sink(destination: Destination, file: string, width: number, height: number, fps: number) {
  const { Output, Mp4OutputFormat, StreamTarget, BufferTarget, VideoSampleSource, VideoSample, canEncodeVideo, QUALITY_HIGH } = await import('mediabunny');
  if (!(await canEncodeVideo('avc', { width, height, quality: QUALITY_HIGH }))) throw new Error('This browser cannot encode H.264 at this size.');
  const stream = await destination.openStream?.(file);
  const target = stream ? new StreamTarget(stream, { chunked: true }) : new BufferTarget();
  const output = new Output({ format: new Mp4OutputFormat({ fastStart: stream ? false : 'in-memory' }), target });
  const source = new VideoSampleSource({ codec: 'avc', bitrate: QUALITY_HIGH, keyFrameInterval: 2 });
  output.addVideoTrack(source, { frameRate: fps });
  await output.start();
  const colorSpace = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false } as const;
  return {
    async frame(rgba: Uint8Array, index: number) {
      const sample = new VideoSample(toI420(rgba, width, height), { format: 'I420', codedWidth: width, codedHeight: height, timestamp: index / fps, duration: 1 / fps, colorSpace });
      await source.add(sample);
      sample.close();
    },
    async finish() {
      await output.finalize();
      if (target instanceof BufferTarget && target.buffer) await destination.write(file, new Uint8Array(target.buffer));
    },
    abort: () => output.cancel(),
  };
}

// BT.709 limited-range 4:2:0 done here rather than by the encoder, so every player reads the file the same way.
// Chroma is the mean of each 2x2 block, so flame edges do not alias; every size offered is even.
function toI420(rgba: Uint8Array, width: number, height: number) {
  const luma = width * height;
  const out = new Uint8Array(luma + luma / 2);
  const chroma = luma / 4;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      out[y * width + x] = 16 + ((47 * rgba[i] + 157 * rgba[i + 1] + 16 * rgba[i + 2]) >> 8);
      if ((x | y) & 1) continue;
      const j = i + width * 4;
      const r = rgba[i] + rgba[i + 4] + rgba[j] + rgba[j + 4];
      const g = rgba[i + 1] + rgba[i + 5] + rgba[j + 1] + rgba[j + 5];
      const b = rgba[i + 2] + rgba[i + 6] + rgba[j + 2] + rgba[j + 6];
      const c = (y >> 1) * (width >> 1) + (x >> 1);
      out[luma + c] = 128 + ((-26 * r - 87 * g + 112 * b) >> 10);
      out[luma + chroma + c] = 128 + ((112 * r - 102 * g - 10 * b) >> 10);
    }
  }
  return out;
}
