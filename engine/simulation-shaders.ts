import { DENSITY } from './scene';

export const PARAMS_STRUCT = /* wgsl */ `
struct Params {
  size: vec3<f32>,
  dt: f32,
  time: f32,
  buoyancy: f32,
  cooling: f32,
  vorticity: f32,
  wind: vec3<f32>,
  burn: f32,
  emitter: vec3<f32>,
  radius: f32,
  emit: f32,
  smokeYield: f32,
  heat: f32,
  dissipation: f32,
  lightDir: vec3<f32>,
  mode: f32,
  turbulence: f32,
  damping: f32,
  pad0: f32,
  pad1: f32,
  objects: array<vec4<f32>, 8>,
}
`;

// Up to four boxes, each stored as (centre, yaw) and (half size, enabled), in grid cells.
export const SOLIDS = /* wgsl */ `
fn boxDistance(p: vec3<f32>, i: i32) -> f32 {
  let placement = P.objects[2 * i];
  let extent = P.objects[2 * i + 1];
  if (extent.w < 0.5) { return 1e5; }
  let o = p - placement.xyz;
  let c = cos(placement.w);
  let s = sin(placement.w);
  let local = vec3<f32>(c * o.x + s * o.z, o.y, c * o.z - s * o.x);
  let d = abs(local) - extent.xyz;
  return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}
fn solidDistance(p: vec3<f32>) -> f32 {
  var d = 1e5;
  for (var i = 0; i < 4; i++) { d = min(d, boxDistance(p, i)); }
  return d;
}
`;

export const HEADER = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
const X = vec3<i32>(1, 0, 0);
const Y = vec3<i32>(0, 1, 0);
const Z = vec3<i32>(0, 0, 1);
fn dims() -> vec3<i32> { return vec3<i32>(P.size); }
fn inside(c: vec3<i32>) -> bool { return all(c >= vec3<i32>(0)) && all(c < dims()); }
fn clampc(c: vec3<i32>) -> vec3<i32> { return clamp(c, vec3<i32>(0), dims() - vec3<i32>(1)); }
fn coord(p: vec3<f32>) -> vec3<f32> { return (p + vec3<f32>(0.5)) / P.size; }
fn hash3(p: vec3<f32>) -> f32 { return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453); }
fn noise3(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(hash3(i), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x);
  let b = mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x);
  let c = mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x);
  let d = mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
` + SOLIDS;

export const SHADERS = {
  advectVelocity: /* wgsl */ `
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var lin: sampler;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let back = vec3<f32>(c) - P.dt * textureLoad(src, c, 0).xyz;
  textureStore(dst, c, vec4<f32>(textureSampleLevel(src, lin, coord(back), 0.0).xyz, 0.0));
}`,
  predict: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var lin: sampler;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let back = vec3<f32>(c) - P.dt * textureLoad(vel, c, 0).xyz;
  textureStore(dst, c, textureSampleLevel(src, lin, coord(back), 0.0));
}`,
  correct: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var old: texture_3d<f32>;
@group(0) @binding(3) var hat: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var lin: sampler;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  let v = textureLoad(vel, c, 0).xyz;
  let back = p - P.dt * v;
  let ahead = textureSampleLevel(hat, lin, coord(p + P.dt * v), 0.0);
  var q = textureLoad(hat, c, 0) + 0.5 * (textureLoad(old, c, 0) - ahead);
  // MacCormack can overshoot, so clamp to the cells the back-traced sample was blended from.
  let base = vec3<i32>(floor(back));
  var lo = vec4<f32>(1e9);
  var hi = vec4<f32>(-1e9);
  for (var k = 0; k < 8; k++) {
    let s = textureLoad(old, clampc(base + vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1)), 0);
    lo = min(lo, s);
    hi = max(hi, s);
  }
  q = clamp(q, lo, hi);
  // Fuel burns into heat and soot; heat and soot then fade.
  let burned = q.z * (1.0 - exp(-P.burn * P.dt));
  q.z -= burned;
  q.y = (q.y + burned * P.heat) * exp(-P.cooling * P.dt);
  q.x = (q.x + burned * P.smokeYield) * exp(-P.dissipation * P.dt);
  let flicker = noise3(p * 0.3 + vec3<f32>(0.0, -P.time * 4.0, P.time * 1.3));
  if (P.mode == 3.0) {
    // Solids give off fuel from a thin skin around them, so the flames take their shape.
    let sd = solidDistance(p);
    if (sd < 0.0) {
      q = vec4<f32>(0.0);
    } else if (sd < 2.5) {
      let amount = (1.0 - sd / 2.5) * (0.35 + 1.3 * flicker) * P.emit * P.dt;
      q.z += amount;
      q.y += amount * 0.25;
    }
  } else {
    let d = length(p - P.emitter) / P.radius;
    if (d < 1.0) {
      var amount = (1.0 - d * d) * (0.35 + 1.3 * flicker) * P.emit * P.dt;
      // An explosion needs a lumpy charge; an even ball of fuel burns into a perfect sphere.
      if (P.mode == 2.0) { amount *= 2.0 * flicker * flicker; }
      if (P.mode == 1.0) {
        q.x += amount * 0.8;
        q.y += amount * 0.5;
      } else {
        q.z += amount;
        q.y += amount * 0.25;
      }
    }
  }
  // A soft sponge under the lid hides the open top boundary.
  q *= 1.0 - 0.15 * smoothstep(0.8, 1.0, p.y / P.size.y);
  textureStore(dst, c, max(q, vec4<f32>(0.0)));
}`,
  curl: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
fn at(c: vec3<i32>) -> vec3<f32> { return textureLoad(vel, clampc(c), 0).xyz; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let l = at(c - X);
  let r = at(c + X);
  let d = at(c - Y);
  let u = at(c + Y);
  let b = at(c - Z);
  let f = at(c + Z);
  let w = 0.5 * vec3<f32>((u.z - d.z) - (f.y - b.y), (f.x - b.x) - (r.z - l.z), (r.y - l.y) - (u.x - d.x));
  textureStore(dst, c, vec4<f32>(w, length(w)));
}`,
  forces: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var sca: texture_3d<f32>;
@group(0) @binding(3) var curl: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
fn spin(c: vec3<i32>) -> f32 { return textureLoad(curl, clampc(c), 0).w; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  let s = textureLoad(sca, c, 0);
  var v = textureLoad(vel, c, 0).xyz;
  v.y += P.dt * (P.buoyancy * s.y - 4.0 * s.x);
  // Vorticity confinement: push along N x omega to give back the swirls the coarse grid smooths away.
  let grad = 0.5 * vec3<f32>(spin(c + X) - spin(c - X), spin(c + Y) - spin(c - Y), spin(c + Z) - spin(c - Z));
  let len = length(grad);
  if (len > 1e-4) {
    v += P.dt * P.vorticity * cross(grad / len, textureLoad(curl, c, 0).xyz);
  }
  v += P.dt * P.wind * (p.y / P.size.y);
  // Rolling noise stirs the hot gas; the pressure solve keeps only its swirling part.
  let drift = vec3<f32>(0.0, -P.time * 1.5, 0.0);
  let swirl = vec3<f32>(noise3(p * 0.12 + drift), noise3(p * 0.12 + drift + vec3<f32>(31.7, 0.0, 7.3)), noise3(p * 0.12 + drift + vec3<f32>(11.1, 0.0, 23.9))) - vec3<f32>(0.5);
  v += P.dt * P.turbulence * swirl * clamp(s.y + 0.3 * s.x, 0.0, 1.5);
  let fromEmitter = p - P.emitter;
  let d = length(fromEmitter) / P.radius;
  if (d < 1.0) {
    v.y += P.dt * P.emit * 0.8 * (1.0 - d);
  }
  if (P.mode == 2.0 && P.emit > 0.0 && d < 1.8) {
    v += P.dt * P.emit * 1.5 * fromEmitter / max(length(fromEmitter), 0.5);
  }
  v *= exp(-P.damping * P.dt) * (1.0 - 0.06 * smoothstep(0.8, 1.0, p.y / P.size.y));
  textureStore(dst, c, vec4<f32>(v, 0.0));
}`,
  divergence: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
fn at(c: vec3<i32>) -> vec3<f32> { return textureLoad(vel, clampc(c), 0).xyz; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let div = 0.5 * ((at(c + X).x - at(c - X).x) + (at(c + Y).y - at(c - Y).y) + (at(c + Z).z - at(c - Z).z));
  textureStore(dst, c, vec4<f32>(div, 0.0, 0.0, 0.0));
}`,
  jacobi: /* wgsl */ `
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var divergence: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<r32float, write>;
fn pressure(c: vec3<i32>) -> f32 {
  if (c.y >= dims().y) { return 0.0; }
  return textureLoad(src, clampc(c), 0).x;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let sum = pressure(c - X) + pressure(c + X) + pressure(c - Y) + pressure(c + Y) + pressure(c - Z) + pressure(c + Z);
  textureStore(dst, c, vec4<f32>((sum - textureLoad(divergence, c, 0).x) / 6.0, 0.0, 0.0, 0.0));
}`,
  project: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn pressure(c: vec3<i32>) -> f32 {
  if (c.y >= dims().y) { return 0.0; }
  return textureLoad(src, clampc(c), 0).x;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  var v = textureLoad(vel, c, 0).xyz - 0.5 * vec3<f32>(pressure(c + X) - pressure(c - X), pressure(c + Y) - pressure(c - Y), pressure(c + Z) - pressure(c - Z));
  let n = dims();
  if (c.x == 0 || c.x == n.x - 1) { v.x = 0.0; }
  if (c.z == 0 || c.z == n.z - 1) { v.z = 0.0; }
  if (c.y == 0) { v.y = max(v.y, 0.0); }
  if (P.mode == 3.0 && solidDistance(vec3<f32>(c)) < 0.0) { v = vec3<f32>(0.0); }
  textureStore(dst, c, vec4<f32>(v, 0.0));
}`,
  shadow: /* wgsl */ `
@group(0) @binding(1) var sca: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = textureDimensions(dst);
  if (any(id >= n)) { return; }
  let box = vec3<f32>(1.0, P.size.y / P.size.x, P.size.z / P.size.x);
  let march = P.lightDir * 0.03 / box;
  var q = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(n);
  var depth = 0.0;
  for (var i = 0; i < 64; i++) {
    q += march;
    if (any(q < vec3<f32>(0.0)) || any(q > vec3<f32>(1.0))) { break; }
    depth += textureSampleLevel(sca, lin, q, 0.0).x;
  }
  textureStore(dst, vec3<i32>(id), vec4<f32>(exp(-depth * 0.03 * ${DENSITY.toFixed(1)}), 0.0, 0.0, 1.0));
}`,
};

export const CLEAR = (format: 'rgba16float' | 'r32float') => /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<${format}, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  textureStore(dst, vec3<i32>(id), vec4<f32>(0.0));
}`;
