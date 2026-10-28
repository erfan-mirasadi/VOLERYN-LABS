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
  return {
    async frame(rgbx: Uint8Array<ArrayBuffer>, index: number) {
      const sample = new VideoSample(rgbx, { format: 'RGBX', codedWidth: width, codedHeight: height, timestamp: index / fps, duration: 1 / fps });
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
