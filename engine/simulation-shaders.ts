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
  expansion: f32,
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
fn isSolid(p: vec3<f32>) -> bool { return P.mode == 3.0 && solidDistance(p) < 0.0; }
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
` + SOLIDS + /* wgsl */ `
// Velocity lives on the cell faces: texel c holds the flow through its lower x, y and z faces.
// The side walls, the floor and solids block their faces; the top stays open.
fn blocked(c: vec3<i32>) -> vec3<bool> {
  let p = vec3<f32>(c);
  let here = isSolid(p);
  return vec3<bool>(
    c.x == 0 || here || isSolid(p - vec3<f32>(1.0, 0.0, 0.0)),
    c.y == 0 || here || isSolid(p - vec3<f32>(0.0, 1.0, 0.0)),
    c.z == 0 || here || isSolid(p - vec3<f32>(0.0, 0.0, 1.0)));
}
`;

// Reads the face-stored velocity at any point and traces where the gas there came from; needs vel and lin bound.
const TRACE = /* wgsl */ `
fn faceOffset(axis: i32) -> vec3<f32> { return 0.5 * vec3<f32>(f32(axis == 0), f32(axis == 1), f32(axis == 2)); }
fn velocityAt(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    textureSampleLevel(vel, lin, coord(p + faceOffset(0)), 0.0).x,
    textureSampleLevel(vel, lin, coord(p + faceOffset(1)), 0.0).y,
    textureSampleLevel(vel, lin, coord(p + faceOffset(2)), 0.0).z);
}
// Where the gas now at p was one step ago, by the midpoint rule; a negative dt asks where it will be.
fn trace(p: vec3<f32>, dt: f32) -> vec3<f32> {
  let mid = p - 0.5 * dt * velocityAt(p);
  return p - dt * velocityAt(mid);
}
`;

export const SHADERS = {
  predictVelocity: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var lin: sampler;
${TRACE}
fn carried(axis: i32, p: vec3<f32>) -> f32 {
  let offset = faceOffset(axis);
  let value = textureSampleLevel(src, lin, coord(trace(p - offset, P.dt) + offset), 0.0);
  return value[axis];
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  textureStore(dst, c, vec4<f32>(carried(0, p), carried(1, p), carried(2, p), 0.0));
}`,
  correctVelocity: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var old: texture_3d<f32>;
@group(0) @binding(3) var hat: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var lin: sampler;
${TRACE}
fn corrected(axis: i32, c: vec3<i32>) -> f32 {
  let offset = faceOffset(axis);
  let face = vec3<f32>(c) - offset;
  let ahead = textureSampleLevel(hat, lin, coord(trace(face, -P.dt) + offset), 0.0);
  let predicted = textureLoad(hat, c, 0);
  let was = textureLoad(old, c, 0);
  let q = predicted[axis] + 0.5 * (was[axis] - ahead[axis]);
  // MacCormack can overshoot, so clamp to the stored values the back-traced sample was blended from.
  let base = vec3<i32>(floor(trace(face, P.dt) + offset));
  var lo = 1e9;
  var hi = -1e9;
  for (var k = 0; k < 8; k++) {
    let s = textureLoad(old, clampc(base + vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1)), 0);
    lo = min(lo, s[axis]);
    hi = max(hi, s[axis]);
  }
  return clamp(q, lo, hi);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  textureStore(dst, c, vec4<f32>(corrected(0, c), corrected(1, c), corrected(2, c), 0.0));
}`,
  predict: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var src: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var lin: sampler;
${TRACE}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  textureStore(dst, c, textureSampleLevel(src, lin, coord(trace(vec3<f32>(c), P.dt)), 0.0));
}`,
  correct: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var old: texture_3d<f32>;
@group(0) @binding(3) var hat: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var lin: sampler;
${TRACE}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  let back = trace(p, P.dt);
  let ahead = textureSampleLevel(hat, lin, coord(trace(p, -P.dt)), 0.0);
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
  // The burn rate rides in w so the pressure solve can let burning gas expand.
  q.w = burned / P.dt;
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
// The velocity at a cell's centre: the mean of each pair of opposite faces.
fn at(c: vec3<i32>) -> vec3<f32> {
  let lower = textureLoad(vel, clampc(c), 0).xyz;
  let upper = vec3<f32>(textureLoad(vel, clampc(c + X), 0).x, textureLoad(vel, clampc(c + Y), 0).y, textureLoad(vel, clampc(c + Z), 0).z);
  return 0.5 * (lower + upper);
}
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
@group(0) @binding(1) var sca: texture_3d<f32>;
@group(0) @binding(2) var curl: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn spin(c: vec3<i32>) -> f32 { return textureLoad(curl, clampc(c), 0).w; }
// The velocity each cell gains this step, taken at its centre; applyForces spreads it onto the faces.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  let s = textureLoad(sca, c, 0);
  var v = vec3<f32>(0.0, P.dt * (P.buoyancy * s.y - 4.0 * s.x), 0.0);
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
  textureStore(dst, c, vec4<f32>(v, 0.0));
}`,
  applyForces: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var force: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  // Each face takes the mean of the two cells it separates.
  let here = textureLoad(force, c, 0).xyz;
  let below = vec3<f32>(textureLoad(force, clampc(c - X), 0).x, textureLoad(force, clampc(c - Y), 0).y, textureLoad(force, clampc(c - Z), 0).z);
  var v = textureLoad(vel, c, 0).xyz + 0.5 * (here + below);
  v *= exp(-P.damping * P.dt) * (1.0 - 0.06 * smoothstep(0.8, 1.0, p.y / P.size.y));
  textureStore(dst, c, vec4<f32>(select(v, vec3<f32>(0.0), blocked(c)), 0.0));
}`,
  divergence: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var sca: texture_3d<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  if (isSolid(vec3<f32>(c))) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let n = dims();
  let v = textureLoad(vel, c, 0).xyz;
  // The far faces: the side walls block, the open top repeats the face below.
  let right = select(textureLoad(vel, clampc(c + X), 0).x, 0.0, c.x == n.x - 1);
  let above = select(textureLoad(vel, clampc(c + Y), 0).y, v.y, c.y == n.y - 1);
  let front = select(textureLoad(vel, clampc(c + Z), 0).z, 0.0, c.z == n.z - 1);
  let div = (right - v.x) + (above - v.y) + (front - v.z);
  // Burning gas expands: the solve targets this much divergence instead of none.
  let expansion = P.expansion * textureLoad(sca, c, 0).w;
  textureStore(dst, c, vec4<f32>(div - expansion, 0.0, 0.0, 0.0));
}`,
  project: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var pressure: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
fn at(c: vec3<i32>) -> f32 { return textureLoad(pressure, clampc(c), 0).x; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let q = at(c);
  let v = textureLoad(vel, c, 0).xyz - vec3<f32>(q - at(c - X), q - at(c - Y), q - at(c - Z));
  textureStore(dst, c, vec4<f32>(select(v, vec3<f32>(0.0), blocked(c)), 0.0));
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

// Flame colour from temperature: an approximate blackbody curve, brighter with the fourth power of temperature.
export const BLACKBODY = /* wgsl */ `
fn blackbody(t: f32) -> vec3<f32> {
  let k = clamp(8.0 + 7.0 * t, 8.0, 40.0);
  let g = clamp((99.4708 * log(k) - 161.1196) / 255.0, 0.0, 1.0);
  let b = select(clamp((138.5177 * log(max(k - 10.0, 1.0)) - 305.0448) / 255.0, 0.0, 1.0), 0.0, k <= 19.0);
  return pow(vec3<f32>(1.0, g, b), vec3<f32>(2.2)) * pow(k / 15.0, 4.0);
}
fn emission(t: f32) -> vec3<f32> { return blackbody(t) * smoothstep(0.2, 1.0, t); }
`;

// The flame's light, averaged over each block of fine cells into the coarse light volume.
export const EMISSION = (scale: number) => BLACKBODY + /* wgsl */ `
@group(0) @binding(0) var sca: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  let base = vec3<i32>(id) * ${scale};
  var sum = vec3<f32>(0.0);
  for (var z = 0; z < ${scale}; z++) {
    for (var y = 0; y < ${scale}; y++) {
      for (var x = 0; x < ${scale}; x++) {
        sum += emission(textureLoad(sca, base + vec3<i32>(x, y, z), 0).y);
      }
    }
  }
  textureStore(dst, vec3<i32>(id), vec4<f32>(sum / ${(scale * scale * scale).toFixed(1)}, 1.0));
}`;

// Spreads the light volume one cell outwards; run a few times it stands in for light scattering through the smoke.
export const LIGHT_BLUR = /* wgsl */ `
@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(textureDimensions(dst));
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var sum = vec3<f32>(0.0);
  var weight = 0.0;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let w = 1.0 / (1.0 + f32(x * x + y * y + z * z));
        let p = c + vec3<i32>(x, y, z);
        if (all(p >= vec3<i32>(0)) && all(p < n)) { sum += textureLoad(src, p, 0).rgb * w; }
        weight += w;
      }
    }
  }
  textureStore(dst, c, vec4<f32>(sum / weight, 1.0));
}`;

export const CLEAR = (format: 'rgba16float' | 'r32float') => /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<${format}, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  textureStore(dst, vec3<i32>(id), vec4<f32>(0.0));
}`;
