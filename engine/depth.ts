// Monocular depth for the footage plate, estimated on this device: Depth Anything V2 Small (Apache-2.0) run by ONNX
// Runtime on WebGPU. Nothing here loads until the user asks for depth; then the model comes once from Hugging Face,
// pinned to a commit so the file can never change under the app, and stays in the browser's cache.
const REPOSITORY = 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/4472b7362082ad9968fee890ca0f1e5aca36b93d/onnx';
export const DEPTH_MODEL = { name: 'Depth Anything V2 Small', megabytes: 50 };
const CACHE = 'voleryn-models';
// The model reads the frame at this many pixels on its long side, in whole 14-pixel patches.
const LONG_SIDE = 518;
const PATCH = 14;
const MEAN = [0.485, 0.456, 0.406];
const DEVIATION = [0.229, 0.224, 0.225];

// Row-major values, width * height of them. From the model they are relative inverse depth, larger when nearer, on
// a scale of its own; scaled to a shot they are nearness, 0 at the far end of the shot's first frame and 1 at its near end.
export type DepthMap = { width: number; height: number; data: Float32Array<ArrayBuffer> };
export type DepthEstimator = { estimate(source: CanvasImageSource, width: number, height: number): Promise<DepthMap> };

let loading: Promise<DepthEstimator> | undefined;

// Resolves to the one estimator the page keeps; onProgress hears the share of the model downloaded so far.
export function loadDepthEstimator(onProgress: (fraction: number) => void) {
  return (loading ??= createEstimator(onProgress).catch((error: unknown) => {
    loading = undefined;
    throw error;
  }));
}

async function createEstimator(onProgress: (fraction: number) => void): Promise<DepthEstimator> {
  // Half floats halve the download; a GPU without them in its shaders takes the full-float model.
  const adapter = await navigator.gpu.requestAdapter();
  const file = adapter?.features.has('shader-f16') ? 'model_fp16.onnx' : 'model.onnx';
  const [ort, model] = await Promise.all([import('onnxruntime-web/webgpu'), fetchModel(`${REPOSITORY}/${file}`, onProgress)]);
  // One thread needs no cross-origin isolation, and the work is on the GPU anyway. The runtime's warnings are about
  // which of the model's nodes it keeps on the CPU, which is its own business; only errors reach the console.
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(model, { executionProviders: ['webgpu'], logSeverityLevel: 3 });
  const canvas = new OffscreenCanvas(PATCH, PATCH);
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  // A session runs one frame at a time, so calls queue behind each other.
  let last: Promise<unknown> = Promise.resolve();

  async function run(source: CanvasImageSource, sourceWidth: number, sourceHeight: number): Promise<DepthMap> {
    const scale = LONG_SIDE / Math.max(sourceWidth, sourceHeight);
    const width = PATCH * Math.max(1, Math.round((sourceWidth * scale) / PATCH));
    const height = PATCH * Math.max(1, Math.round((sourceHeight * scale) / PATCH));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.imageSmoothingQuality = 'high';
    context.drawImage(source, 0, 0, width, height);
    const rgba = context.getImageData(0, 0, width, height).data;
    const area = width * height;
    const pixels = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
      for (let c = 0; c < 3; c++) pixels[c * area + i] = (rgba[4 * i + c] / 255 - MEAN[c]) / DEVIATION[c];
    }
    const output = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', pixels, [1, 3, height, width]) });
    const depth = output[session.outputNames[0]];
    return { width: depth.dims[depth.dims.length - 1], height: depth.dims[depth.dims.length - 2], data: Float32Array.from(depth.data as Float32Array) };
  }

  return {
    estimate(source, width, height) {
      const next = last.then(() => run(source, width, height));
      last = next.catch(() => {});
      return next;
    },
  };
}

async function fetchModel(url: string, onProgress: (fraction: number) => void) {
  const cache = await caches.open(CACHE).catch(() => null);
  const cached = await cache?.match(url);
  if (cached) return new Uint8Array(await cached.arrayBuffer());
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error('The depth model could not be downloaded.');
  const total = Number(response.headers.get('content-length')) || DEPTH_MODEL.megabytes * 1e6;
  const reader = response.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(Math.min(1, received / total));
  }
  const model = new Uint8Array(await new Blob(chunks).arrayBuffer());
  // A full disk only costs the next visit the download again.
  await cache?.put(url, new Response(model)).catch(() => {});
  return model;
}

const STRIDE = 16;

// The line s * x + t that best carries one frame's raw depth onto the reference frame's, fitted twice: the second
// time without the pixels the first fit missed by most, which is whatever moved in the shot. Frames that no longer
// agree on what is near, after a cut or a long pan, keep the model's own scale.
function fitScale(x: Float32Array, y: Float32Array): [scale: number, shift: number] {
  let s = 1;
  let t = 0;
  let within = Infinity;
  for (let pass = 0; pass < 2; pass++) {
    let n = 0;
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    let syy = 0;
    for (let i = 0; i < x.length; i += STRIDE) {
      if (Math.abs(s * x[i] + t - y[i]) > within) continue;
      n++;
      sx += x[i];
      sy += y[i];
      sxx += x[i] * x[i];
      sxy += x[i] * y[i];
      syy += y[i] * y[i];
    }
    const varX = sxx - (sx * sx) / n;
    const varY = syy - (sy * sy) / n;
    const cov = sxy - (sx * sy) / n;
    if (n < 64 || cov <= 0 || cov * cov < 0.5 * varX * varY) return [1, 0];
    s = cov / varX;
    t = (sy - s * sx) / n;
    within = 1.5 * Math.sqrt(Math.max(varY - s * cov, 0) / n);
  }
  return [s, t];
}

// Turns the model's raw maps into nearness that means the same on every frame of a shot. The first map it is given
// is the reference: its 1st and 99th percentiles become 0 and 1. The model's scale wanders a little from frame to
// frame, so each later map is first fitted back onto the reference; then the same depth setting cuts the same
// place in the shot whichever frame is showing, live or in an export.
export function createDepthScale() {
  let reference: { raw: Float32Array; low: number; span: number } | null = null;
  return ({ width, height, data }: DepthMap): DepthMap => {
    let s = 1;
    let t = 0;
    if (reference?.raw.length === data.length) {
      [s, t] = fitScale(data, reference.raw);
    } else {
      const sorted = data.filter((_, i) => i % 4 === 0).sort();
      const low = sorted[Math.floor(0.01 * sorted.length)];
      reference = { raw: data, low, span: Math.max(sorted[Math.floor(0.99 * sorted.length)] - low, 1e-6) };
    }
    const { low, span } = reference;
    return { width, height, data: data.map((value) => (s * value + t - low) / span) };
  };
}
