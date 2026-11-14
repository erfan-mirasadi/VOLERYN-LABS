import { GUIDE } from './render-shaders';

// How far past the fire's own depth, in nearness, the footage has to come before it covers the fire fully: a hard
// line through a depth map would stair-step along the ground the fire stands on.
const SOFT = 0.04;
// Half the width of the guided filter's window, in depth texels: far enough to reach the plate's real edge from where
// the depth map put it.
const RADIUS = 4;
// How much contrast an edge in the plate needs before the matte follows it instead of the blurred depth.
const EPSILON = 2e-4;
// How far from the depth map's own edge, in depth texels, the plate may move the matte's. Further in, the foreground
// is certain and stays opaque whatever its colours do; further out the background is certain and stays clear.
const BAND = 1;

const WINDOW = (2 * RADIUS + 1).toFixed(1);
const each = (count: number, line: (i: number) => string) => Array.from({ length: count }, (_, i) => line(i)).join('\n');

// The rough cover: what the depth map alone says is nearer than the fire.
const ROUGH = /* wgsl */ `
struct Hold { nearness: f32 }
@group(0) @binding(0) var<uniform> H: Hold;
fn rough(near: f32) -> f32 { return smoothstep(H.nearness, H.nearness + ${SOFT}, near); }
`;

// The guided filter's inputs at the depth map's size: the plate's colour as the filter reads it, averaged over each
// depth texel from sixteen bilinear taps, the rough cover, and the products whose window means make the covariances.
export const SOURCE = GUIDE + ROUGH + /* wgsl */ `
@group(0) @binding(1) var plate: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var values: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var covered: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var squares: texture_storage_2d<rgba32float, write>;
@group(0) @binding(7) var mixed: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(depth);
  if (any(id.xy >= size)) { return; }
  let at = vec2<i32>(id.xy);
  var colour = vec3<f32>(0.0);
  for (var y = 0; y < 4; y++) {
    for (var x = 0; x < 4; x++) {
      let uv = (vec2<f32>(id.xy) + (vec2<f32>(f32(x), f32(y)) + 0.5) * 0.25) / vec2<f32>(size);
      colour += guide(textureSampleLevel(plate, lin, uv, 0.0).rgb);
    }
  }
  colour /= 16.0;
  let cover = rough(textureLoad(depth, at, 0).x);
  textureStore(values, at, vec4<f32>(colour, cover));
  textureStore(covered, at, vec4<f32>(colour * cover, 0.0));
  textureStore(squares, at, vec4<f32>(colour * colour, 0.0));
  textureStore(mixed, at, vec4<f32>(colour.r * colour.g, colour.r * colour.b, colour.g * colour.b, 0.0));
}`;

// One direction of a box mean over `count` textures at once; the two directions in turn average a square window.
export const box = (count: number) => /* wgsl */ `
override dx: i32;
override dy: i32;
${each(count, (i) => `@group(0) @binding(${i}) var src${i}: texture_2d<f32>;`)}
${each(count, (i) => `@group(0) @binding(${count + i}) var dst${i}: texture_storage_2d<rgba32float, write>;`)}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(src0);
  if (any(id.xy >= size)) { return; }
  let last = vec2<i32>(size) - vec2<i32>(1);
${each(count, (i) => `  var sum${i} = vec4<f32>(0.0);`)}
  for (var k = -${RADIUS}; k <= ${RADIUS}; k++) {
    let at = clamp(vec2<i32>(id.xy) + k * vec2<i32>(dx, dy), vec2<i32>(0), last);
${each(count, (i) => `    sum${i} += textureLoad(src${i}, at, 0);`)}
  }
${each(count, (i) => `  textureStore(dst${i}, vec2<i32>(id.xy), sum${i} / ${WINDOW});`)}
}`;

// He, Sun and Tang's guided filter with a colour guide: in each window the cover is fitted as a plane over the
// plate's colour, dot(a, colour) + b, from the window's means. Where the plate has an edge the plane is steep and the
// cover takes the edge's shape; where it is flat the plane is level at the mean cover. Three channels tell a dark
// green tree from a dark blue hill, which brightness alone cannot. The fit solves (covariance + EPSILON) a = cov by a
// Cholesky factorisation: Cramer's rule loses most of a float's digits when the channels move together.
export const COEFFICIENTS = /* wgsl */ `
@group(0) @binding(0) var values: texture_2d<f32>;
@group(0) @binding(1) var covered: texture_2d<f32>;
@group(0) @binding(2) var squares: texture_2d<f32>;
@group(0) @binding(3) var mixed: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id.xy >= textureDimensions(values))) { return; }
  let at = vec2<i32>(id.xy);
  let m = textureLoad(values, at, 0);
  let mean = m.rgb;
  let cov = textureLoad(covered, at, 0).rgb - mean * m.a;
  let diagonal = textureLoad(squares, at, 0).rgb - mean * mean + vec3<f32>(${EPSILON});
  let off = textureLoad(mixed, at, 0).rgb - vec3<f32>(mean.r * mean.g, mean.r * mean.b, mean.g * mean.b);
  let l00 = sqrt(diagonal.r);
  let l10 = off.x / l00;
  let l20 = off.y / l00;
  let l11 = sqrt(diagonal.g - l10 * l10);
  let l21 = (off.z - l20 * l10) / l11;
  let l22 = sqrt(diagonal.b - l20 * l20 - l21 * l21);
  let y0 = cov.r / l00;
  let y1 = (cov.g - l10 * y0) / l11;
  let y2 = (cov.b - l20 * y0 - l21 * y1) / l22;
  let a2 = y2 / l22;
  let a1 = (y1 - l21 * a2) / l11;
  let a0 = (y0 - l10 * a1 - l20 * a2) / l00;
  let a = vec3<f32>(a0, a1, a2);
  textureStore(dst, at, vec4<f32>(a, m.a - dot(a, mean)));
}`;

// The second half of the coefficients' own box mean, and beside them the bounds the finished matte keeps to: the
// least and the most rough cover within BAND texels. Inside solid foreground both are 1, in open background both 0,
// and only in the strip along a depth edge is there room between them for the plate to place the matte.
export const FINISH = ROUGH + /* wgsl */ `
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var coefficients: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var bounds: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = textureDimensions(src);
  if (any(id.xy >= size)) { return; }
  let at = vec2<i32>(id.xy);
  let last = vec2<i32>(size) - vec2<i32>(1);
  var plane = vec4<f32>(0.0);
  for (var k = -${RADIUS}; k <= ${RADIUS}; k++) {
    plane += textureLoad(src, clamp(at + vec2<i32>(0, k), vec2<i32>(0), last), 0);
  }
  var nearest = -1e9;
  var farthest = 1e9;
  for (var y = -${BAND}; y <= ${BAND}; y++) {
    for (var x = -${BAND}; x <= ${BAND}; x++) {
      let near = textureLoad(depth, clamp(at + vec2<i32>(x, y), vec2<i32>(0), last), 0).x;
      nearest = max(nearest, near);
      farthest = min(farthest, near);
    }
  }
  textureStore(coefficients, at, plane / ${WINDOW});
  textureStore(bounds, at, vec4<f32>(rough(farthest), rough(nearest), 0.0, 0.0));
}`;
