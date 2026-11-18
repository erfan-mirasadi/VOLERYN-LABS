import type { DepthMap } from './depth';

// What the footage looks like in front of the fire and behind it, as two mixtures of Gaussians over colour, fitted
// to the frame the depth was read from. The depth map is a few plate pixels to the texel and blurs whatever is thinner
// than that: branches against the sky come out as one near blur, and the sky between them is read as near. Colour
// knows better there. The holdout weighs the depth against how much likelier a pixel's colour is in front than
// behind, the way GrabCut weighs its colour models (Rother, Kolmogorov and Blake, 2004).
const PARTS = 5;
// Colours are compared as the square root of linear light, close to how different they look.
const FLOOR = 0.025;
const SAMPLES = 12000;
const PASSES = 8;
// How far from the fire's depth, in nearness, a texel has to be to teach a side what it looks like.
const MARGIN = 0.01;
const RAMP = 0.05;

type Mixture = { mean: Float32Array; variance: Float32Array; weight: Float32Array };
export type ColourModels = { front: Mixture; behind: Mixture };

const toLinear = (byte: number) => {
  const c = byte / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};
const GUIDE = Float32Array.from({ length: 256 }, (_, byte) => Math.sqrt(toLinear(byte)));
const smoothstep = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// The models for a frame's depth with the fire at `nearness`, started from the last frame's so a shot's models move
// smoothly from frame to frame instead of jumping between equally good fits.
export function fitColourModels({ width, height, data, colour }: DepthMap, nearness: number, previous: ColourModels | null): ColourModels {
  const step = Math.max(1, Math.round(Math.sqrt((width * height) / SAMPLES)));
  const samples: number[] = [];
  const front: number[] = [];
  const behind: number[] = [];
  for (let y = step >> 1; y < height; y += step) {
    for (let x = step >> 1; x < width; x += step) {
      const i = y * width + x;
      samples.push(GUIDE[colour[4 * i]], GUIDE[colour[4 * i + 1]], GUIDE[colour[4 * i + 2]]);
      front.push(smoothstep(nearness + MARGIN, nearness + MARGIN + RAMP, data[i]));
      behind.push(1 - smoothstep(nearness - MARGIN - RAMP, nearness - MARGIN, data[i]));
    }
  }
  const points = Float32Array.from(samples);
  const behindModel = fit(points, Float32Array.from(behind), previous?.behind);
  const frontWeights = Float32Array.from(front);
  let frontModel = fit(points, frontWeights, previous?.front);
  // Sky seen between branches is near by the depth but looks like what is behind: the front model is fitted again
  // without the texels the behind model explains better, so it keeps the branches and lets the sky go.
  for (let n = 0; n < frontWeights.length; n++) {
    const p = points.subarray(3 * n, 3 * n + 3);
    frontWeights[n] *= 1 / (1 + Math.exp(Math.max(-30, Math.min(30, logLikelihood(p, behindModel) - logLikelihood(p, frontModel)))));
  }
  frontModel = fit(points, frontWeights, frontModel);
  return { front: frontModel, behind: behindModel };
}

// The models for the shaders: each part's mean and log weight, then its variance and log normaliser, the front
// mixture's parts first.
export function packColourModels(models: ColourModels | null) {
  const packed = new Float32Array(2 * PARTS * 8);
  if (!models) return neutral(packed);
  [models.front, models.behind].forEach((mixture, side) => {
    for (let k = 0; k < PARTS; k++) {
      const at = (side * PARTS + k) * 8;
      let normaliser = 0;
      for (let c = 0; c < 3; c++) {
        packed[at + c] = mixture.mean[3 * k + c];
        packed[at + 4 + c] = mixture.variance[3 * k + c];
        normaliser -= 0.5 * Math.log(2 * Math.PI * mixture.variance[3 * k + c]);
      }
      packed[at + 3] = Math.log(Math.max(mixture.weight[k], 1e-12));
      packed[at + 7] = normaliser;
    }
  });
  return packed;
}

// Both sides alike: colour then tells the holdout nothing and the depth decides alone.
function neutral(packed: Float32Array<ArrayBuffer>) {
  for (let k = 0; k < 2 * PARTS; k++) {
    packed.set([0.5, 0.5, 0.5, Math.log(1 / PARTS), 1, 1, 1, -1.5 * Math.log(2 * Math.PI)], 8 * k);
  }
  return packed;
}

function logLikelihood(p: Float32Array, { mean, variance, weight }: Mixture) {
  let best = -Infinity;
  const terms = new Float64Array(PARTS);
  for (let k = 0; k < PARTS; k++) {
    let sum = Math.log(Math.max(weight[k], 1e-12));
    for (let c = 0; c < 3; c++) {
      const v = variance[3 * k + c];
      const d = p[c] - mean[3 * k + c];
      sum -= 0.5 * ((d * d) / v + Math.log(2 * Math.PI * v));
    }
    terms[k] = sum;
    best = Math.max(best, sum);
  }
  let total = 0;
  for (let k = 0; k < PARTS; k++) total += Math.exp(terms[k] - best);
  return best + Math.log(total);
}

// Weighted expectation-maximisation with diagonal covariances, seeded by k-means++ on a fixed sequence when there
// is nothing to start from, so the same frame always fits the same way.
function fit(points: Float32Array, weights: Float32Array, start?: Mixture): Mixture {
  const count = weights.length;
  let total = 0;
  for (let n = 0; n < count; n++) total += weights[n];
  const mixture = start ? { mean: start.mean.slice(), variance: start.variance.slice(), weight: start.weight.slice() } : seed(points, weights, total);
  if (total < 32) return mixture;
  const responsibility = new Float64Array(PARTS);
  for (let pass = 0; pass < PASSES; pass++) {
    const mass = new Float64Array(PARTS);
    const sum = new Float64Array(3 * PARTS);
    const squares = new Float64Array(3 * PARTS);
    for (let n = 0; n < count; n++) {
      if (weights[n] < 1e-3) continue;
      const p = points.subarray(3 * n, 3 * n + 3);
      let best = -Infinity;
      for (let k = 0; k < PARTS; k++) {
        let l = Math.log(Math.max(mixture.weight[k], 1e-12));
        for (let c = 0; c < 3; c++) {
          const v = mixture.variance[3 * k + c];
          const d = p[c] - mixture.mean[3 * k + c];
          l -= 0.5 * ((d * d) / v + Math.log(v));
        }
        responsibility[k] = l;
        best = Math.max(best, l);
      }
      let norm = 0;
      for (let k = 0; k < PARTS; k++) norm += responsibility[k] = Math.exp(responsibility[k] - best);
      for (let k = 0; k < PARTS; k++) {
        const r = (weights[n] * responsibility[k]) / norm;
        mass[k] += r;
        for (let c = 0; c < 3; c++) {
          sum[3 * k + c] += r * p[c];
          squares[3 * k + c] += r * p[c] * p[c];
        }
      }
    }
    for (let k = 0; k < PARTS; k++) {
      mixture.weight[k] = mass[k] / total;
      if (mass[k] < 1e-6) continue;
      for (let c = 0; c < 3; c++) {
        const mean = sum[3 * k + c] / mass[k];
        mixture.mean[3 * k + c] = mean;
        mixture.variance[3 * k + c] = Math.max(squares[3 * k + c] / mass[k] - mean * mean, FLOOR * FLOOR);
      }
    }
  }
  return mixture;
}

function seed(points: Float32Array, weights: Float32Array, total: number): Mixture {
  const mean = new Float32Array(3 * PARTS).fill(0.5);
  const variance = new Float32Array(3 * PARTS).fill(0.04);
  const weight = new Float32Array(PARTS).fill(1 / PARTS);
  if (total < 32) return { mean, variance, weight };
  let state = 0x9e3779b9;
  const random = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const count = weights.length;
  const nearest = new Float64Array(count).fill(Infinity);
  for (let k = 0; k < PARTS; k++) {
    let mass = 0;
    for (let n = 0; n < count; n++) mass += weights[n] * (k === 0 ? 1 : nearest[n]);
    let target = random() * mass;
    let chosen = 0;
    for (let n = 0; n < count && target > 0; n++) {
      target -= weights[n] * (k === 0 ? 1 : nearest[n]);
      chosen = n;
    }
    for (let c = 0; c < 3; c++) mean[3 * k + c] = points[3 * chosen + c];
    for (let n = 0; n < count; n++) {
      let d = 0;
      for (let c = 0; c < 3; c++) d += (points[3 * n + c] - mean[3 * k + c]) ** 2;
      nearest[n] = Math.min(nearest[n], d);
    }
  }
  return { mean, variance, weight };
}
