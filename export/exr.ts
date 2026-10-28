import { type Bytes, concat, zlib } from './png';

const LINES_PER_CHUNK = 16;
const MAGIC = [0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0];
const REC709 = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329];

// Half bits or floats, one sample per pixel, row-major.
export type ExrChannel = { name: string; data: Uint16Array | Float32Array };
export type ExrMeta = { fps: number; worldToCamera: number[]; worldToNDC: number[]; comments: string };

// A scanline EXR, ZIP-compressed sixteen lines at a time, with the camera, frame rate, Rec.709 chromaticities and a
// comment in the header. Channels are sorted by code unit as the format requires.
export async function encodeExr(width: number, height: number, channels: ExrChannel[], meta: ExrMeta): Promise<Bytes> {
  const sorted = [...channels].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  const window = ints([0, 0, width - 1, height - 1]);
  const header = concat([
    new Uint8Array(MAGIC),
    attribute('channels', 'chlist', channelList(sorted)),
    attribute('compression', 'compression', new Uint8Array([3])),
    attribute('dataWindow', 'box2i', window),
    attribute('displayWindow', 'box2i', window),
    attribute('lineOrder', 'lineOrder', new Uint8Array([0])),
    attribute('pixelAspectRatio', 'float', floats([1])),
    attribute('screenWindowCenter', 'v2f', floats([0, 0])),
    attribute('screenWindowWidth', 'float', floats([1])),
    attribute('chromaticities', 'chromaticities', floats(REC709)),
    attribute('framesPerSecond', 'rational', ints([meta.fps, 1])),
    attribute('worldToCamera', 'm44f', floats(meta.worldToCamera)),
    attribute('worldToNDC', 'm44f', floats(meta.worldToNDC)),
    attribute('comments', 'string', new TextEncoder().encode(meta.comments)),
    new Uint8Array(1),
  ]);
  const bytesPerLine = sorted.reduce((n, channel) => n + width * (channel.data instanceof Float32Array ? 4 : 2), 0);
  const chunks: Bytes[] = [];
  for (let first = 0; first < height; first += LINES_PER_CHUNK) {
    const lines = Math.min(LINES_PER_CHUNK, height - first);
    const raw = new Uint8Array(lines * bytesPerLine);
    let at = 0;
    for (let y = first; y < first + lines; y++) {
      for (const { data } of sorted) {
        const row = data.subarray(y * width, (y + 1) * width);
        raw.set(new Uint8Array(row.buffer, row.byteOffset, row.byteLength), at);
        at += row.byteLength;
      }
    }
    const packed = await zlib(predict(raw));
    const body = packed.length < raw.length ? packed : raw;
    chunks.push(concat([ints([first, body.length]), body]));
  }
  const table = new Uint8Array(chunks.length * 8);
  const view = new DataView(table.buffer);
  let offset = header.length + table.length;
  chunks.forEach((chunk, i) => {
    view.setBigUint64(i * 8, BigInt(offset), true);
    offset += chunk.length;
  });
  return concat([header, table, ...chunks]);
}

// OpenEXR's ZIP filter: even bytes first, then odd bytes, then each byte as its difference from the one before.
function predict(raw: Uint8Array): Bytes {
  const n = raw.length;
  const half = (n + 1) >> 1;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i & 1 ? half + (i >> 1) : i >> 1] = raw[i];
  let previous = out[0];
  for (let i = 1; i < n; i++) {
    const current = out[i];
    out[i] = current - previous + 128;
    previous = current;
  }
  return out;
}

function attribute(name: string, type: string, value: Uint8Array) {
  return concat([text(name), text(type), ints([value.length]), value]);
}

function channelList(channels: ExrChannel[]) {
  const parts = channels.map(({ name, data }) => concat([text(name), ints([data instanceof Float32Array ? 2 : 1]), new Uint8Array(4), ints([1, 1])]));
  return concat([...parts, new Uint8Array(1)]);
}

function text(value: string) {
  return new TextEncoder().encode(`${value}\0`);
}

function ints(values: number[]) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, i) => view.setInt32(i * 4, value, true));
  return out;
}

function floats(values: number[]) {
  const out = new Uint8Array(values.length * 4);
  const view = new DataView(out.buffer);
  values.forEach((value, i) => view.setFloat32(i * 4, value, true));
  return out;
}
