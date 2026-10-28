export type Bytes = Uint8Array<ArrayBuffer>;

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

export function crc32(bytes: Uint8Array) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// 'deflate' is the zlib-wrapped stream that both PNG and EXR expect.
export async function zlib(bytes: Bytes): Promise<Bytes> {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function concat(parts: Uint8Array[]): Bytes {
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

// An sRGB-tagged RGBA PNG from packed rows (16-bit samples big-endian), every row Sub-filtered.
export async function encodePng(width: number, height: number, depth: 8 | 16, rgba: Uint8Array): Promise<Bytes> {
  const bpp = depth / 2;
  const stride = width * bpp;
  const filtered = new Uint8Array(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    const src = y * stride;
    const dst = y * (stride + 1) + 1;
    filtered[dst - 1] = 1;
    for (let i = 0; i < stride; i++) filtered[dst + i] = rgba[src + i] - (i >= bpp ? rgba[src + i - bpp] : 0);
  }
  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header[8] = depth;
  header[9] = 6;
  const gamma = new Uint8Array(4);
  new DataView(gamma.buffer).setUint32(0, 45455);
  return concat([SIGNATURE, chunk('IHDR', header), chunk('sRGB', new Uint8Array(1)), chunk('gAMA', gamma), chunk('IDAT', await zlib(filtered)), chunk('IEND', new Uint8Array(0))]);
}

function chunk(type: string, data: Uint8Array) {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
