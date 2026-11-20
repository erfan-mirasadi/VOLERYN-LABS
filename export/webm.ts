import type { Destination } from './files';
import { toI420, videoBitrateFor } from './mp4';

export class WebmEncodingUnavailable extends Error {
  constructor() {
    super('This browser cannot encode VP9 at the selected size and frame rate. Choose MP4 + PNG master or a PNG sequence.');
    this.name = 'WebmEncodingUnavailable';
  }
}

// Mediabunny encodes the alpha plane separately and writes it as WebM alpha side data. Native WebCodecs
// alpha support is not assumed. Both members of a pair receive identical colour encoding and timing settings.
// https://mediabunny.dev/guide/quick-start#creating-transparent-video
export async function createWebmSink(destination: Destination, file: string, width: number, height: number, fps: number, alpha: boolean) {
  const { Output, WebMOutputFormat, StreamTarget, BufferTarget, VideoSampleSource, VideoSample, canEncodeVideo } = await import('mediabunny');
  const bitrate = videoBitrateFor(width, height, fps);
  if (!(await canEncodeVideo('vp9', { width, height, bitrate, frameRate: fps, alpha: alpha ? 'keep' : 'discard', latencyMode: 'quality' }))) {
    throw new WebmEncodingUnavailable();
  }
  const stream = await destination.openStream?.(file);
  const target = stream ? new StreamTarget(stream, { chunked: true }) : new BufferTarget();
  const output = new Output({ format: new WebMOutputFormat({ appendOnly: Boolean(stream) }), target });
  const source = new VideoSampleSource({ codec: 'vp9', bitrate, keyFrameInterval: 2, alpha: alpha ? 'keep' : 'discard', latencyMode: 'quality' });
  output.addVideoTrack(source, { frameRate: fps, canBeTransparent: alpha });
  await output.start();
  const colorSpace = { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false } as const;
  return {
    async frame(rgba: Uint8Array, index: number) {
      const colour = toI420(rgba, width, height);
      const data = alpha ? new Uint8Array(colour.length + width * height) : colour;
      if (alpha) {
        data.set(colour);
        for (let pixel = 0; pixel < width * height; pixel++) data[colour.length + pixel] = rgba[4 * pixel + 3];
      }
      const sample = new VideoSample(data, { format: alpha ? 'I420A' : 'I420', codedWidth: width, codedHeight: height,
        timestamp: index / fps, duration: 1 / fps, colorSpace });
      try { await source.add(sample); }
      finally { sample.close(); }
    },
    async finish() {
      await output.finalize();
      if (target instanceof BufferTarget && target.buffer) await destination.write(file, new Uint8Array(target.buffer));
    },
    abort: () => output.cancel(),
  };
}
