import { TONEMAP } from './render-shaders';
import { DT } from './scene';
import { BLACKBODY, HEADER, PARAMS_STRUCT, SOLIDS, TRACE } from './simulation-shaders';

export const SPARK_COUNT = 4096;
// A fresh spark's temperature and the one it goes out at, on the flame's blackbody scale.
const SPARK_HEAT = 1.4;
const SPARK_OUT = 0.45;

// One spark: position and velocity in grid cells, age and lifetime in seconds. Dead once its age reaches its life.
const SPARK_STRUCT = /* wgsl */ `
struct Spark { pos: vec3<f32>, age: f32, vel: vec3<f32>, life: f32 }
`;

// Moves every spark one step and respawns dead ones from the burning gas. Spawns are rolled from a hash of the
// spark's index, the simulation time and the seed, so an export plays back the same sparks every time.
export const SPARK_STEP = HEADER + SPARK_STRUCT + /* wgsl */ `
// Per-spark spawn chance per second, and the burst an explosion throws per unit of blast.
struct Spawn { rate: f32, burst: f32 }
@group(0) @binding(1) var<storage, read_write> sparks: array<Spark, ${SPARK_COUNT}>;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var sca: texture_3d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var<uniform> S: Spawn;
${TRACE}
// Seconds a spark takes to pick up the gas velocity, how fast it sinks through the gas in cells/s², the random
// tumble it adds each step, and the pop that throws it out of the fuel, in cells/s.
const DRAG = 0.25;
const SETTLE = 30.0;
const TUMBLE = 30.0;
const POP = 80.0;
const POP_UP = 25.0;
const BURST_SPEED = 220.0;
const BURST_UP = 60.0;
const SKIN = 2.5;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand(base: u32, k: u32) -> f32 { return f32(pcg(base + k)) / 4294967296.0; }
fn unitVector(u: f32, v: f32) -> vec3<f32> {
  let z = 2.0 * u - 1.0;
  let r = sqrt(max(1.0 - z * z, 0.0));
  let a = 6.2831853 * v;
  return vec3<f32>(r * cos(a), z, r * sin(a));
}
fn insideBox(p: vec3<f32>) -> bool { return all(p > vec3<f32>(-0.5)) && all(p < P.size - vec3<f32>(0.5)); }
fn burning(p: vec3<f32>) -> bool {
  let s = textureSampleLevel(sca, lin, coord(p), 0.0);
  return s.w > 1.0 || s.y > 1.0;
}
// The n-th enabled box, or -1.
fn pickBox(u: f32) -> i32 {
  var count = 0;
  for (var i = 0; i < 4; i++) { count += i32(P.objects[2 * i + 1].w > 0.5); }
  var pick = min(i32(u * f32(count)), count - 1);
  for (var i = 0; i < 4; i++) {
    if (P.objects[2 * i + 1].w > 0.5) {
      if (pick == 0) { return i; }
      pick--;
    }
  }
  return -1;
}
// A point in the fuel: the burning gas around the emitter, or in objects mode the skin of one of the boxes.
fn candidate(base: u32, k: u32) -> vec3<f32> {
  let u = vec3<f32>(rand(base, k), rand(base, k + 1u), rand(base, k + 2u));
  if (P.mode != 3.0) { return P.emitter + unitVector(u.x, u.y) * 2.0 * P.radius * pow(u.z, 1.0 / 3.0); }
  let i = pickBox(rand(base, k + 3u));
  if (i < 0) { return vec3<f32>(-10.0); }
  let placement = P.objects[2 * i];
  let local = (2.0 * u - vec3<f32>(1.0)) * (P.objects[2 * i + 1].xyz + vec3<f32>(SKIN));
  let c = cos(placement.w);
  let s = sin(placement.w);
  let p = placement.xyz + vec3<f32>(c * local.x - s * local.z, local.y, s * local.x + c * local.z);
  let sd = solidDistance(p);
  return select(vec3<f32>(-10.0), p, sd >= 0.0 && sd < SKIN);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  var s = sparks[i];
  let base = pcg(i ^ pcg(bitcast<u32>(P.time) ^ pcg(bitcast<u32>(P.seed) + 0x9e3779b9u)));
  if (s.age < s.life) {
    var v = mix(s.vel, velocityAt(s.pos), 1.0 - exp(-P.dt / DRAG));
    v.y -= SETTLE * P.dt;
    v += (vec3<f32>(rand(base, 0u), rand(base, 1u), rand(base, 2u)) - vec3<f32>(0.5)) * TUMBLE * sqrt(P.dt);
    s.pos += v * P.dt;
    s.vel = v;
    s.age += P.dt;
    if (!insideBox(s.pos) || isSolid(s.pos)) { s.age = s.life; }
    sparks[i] = s;
    return;
  }
  if (rand(base, 0u) < S.rate * P.dt) {
    for (var k = 0u; k < 4u; k++) {
      let p = candidate(base, 8u + 4u * k);
      if (!insideBox(p) || !burning(p)) { continue; }
      // Most pops are gentle, a few fling the spark well clear of the flame.
      let r = rand(base, 24u);
      let pop = rand(base, 27u);
      s.pos = p;
      s.vel = velocityAt(p) + unitVector(rand(base, 25u), rand(base, 26u)) * POP * pop * pop + vec3<f32>(0.0, POP_UP * (0.5 + rand(base, 28u)), 0.0);
      s.life = 1.2 + 2.8 * r * r;
      s.age = 0.0;
      sparks[i] = s;
      return;
    }
  } else if (P.mode == 2.0 && rand(base, 1u) < S.burst * P.blast * P.dt) {
    // The blast throws burning debris out of the charge, outward and up.
    let dir = unitVector(rand(base, 2u), rand(base, 3u));
    let speed = rand(base, 5u);
    s.pos = P.emitter + dir * P.radius * 0.8 * pow(rand(base, 4u), 1.0 / 3.0);
    s.vel = dir * BURST_SPEED * (0.15 + 0.85 * speed * speed) + vec3<f32>(0.0, BURST_UP, 0.0);
    s.life = 0.8 + 2.4 * rand(base, 6u);
    s.age = 0.0;
    sparks[i] = s;
  }
}`;

// Draws each live spark as a short streak along its motion, a soft line of blackbody colour at HDR intensity,
// dimmed by the smoke and flame between it and the camera. One quad per spark, strip order.
export const SPARK_DRAW = BLACKBODY + TONEMAP + SPARK_STRUCT + /* wgsl */ `
struct View {
  camPos: vec3<f32>,
  tanHalf: f32,
  forward: vec3<f32>,
  aspect: f32,
  right: vec3<f32>,
  frameTime: f32,
  up: vec3<f32>,
  stepSize: f32,
  boxMin: vec3<f32>,
  exposure: f32,
  boxMax: vec3<f32>,
  density: f32,
  smokeTint: vec3<f32>,
  fire: f32,
  viewport: vec4<f32>,
  maxSteps: f32,
  fireLight: f32,
  frame: f32,
  fireDensity: f32,
  detail: f32,
  flowWeight: f32,
  sampleWeight: f32,
  pad: f32,
}
@group(0) @binding(0) var<uniform> V: View;
` + PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var sca: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var<storage, read> sparks: array<Spark, ${SPARK_COUNT}>;
` + SOLIDS + /* wgsl */ `
// Seconds of motion a streak shows, the brightness of a fresh spark, and the steps of the occlusion march.
const STREAK = ${(3 * DT).toFixed(5)};
const GAIN = 2.5;
const OCCLUSION_STEPS = 20;
const HEAT = ${SPARK_HEAT.toFixed(2)};
const COOLING = ${Math.log(SPARK_HEAT / SPARK_OUT).toFixed(4)};

struct Out {
  @builtin(position) position: vec4<f32>,
  // The streak's head and tail in pixels; its colour and half width; how much of it the gas lets through.
  @location(0) @interpolate(flat) segment: vec4<f32>,
  @location(1) @interpolate(flat) colour: vec4<f32>,
  @location(2) @interpolate(flat) seen: f32,
}

fn toWorld(cells: vec3<f32>) -> vec3<f32> { return V.boxMin + (cells + vec3<f32>(0.5)) / P.size * (V.boxMax - V.boxMin); }
// Pixel coordinates of a world point, shifted by the sample's jitter the way the raymarch's rays are.
fn toPixel(p: vec3<f32>) -> vec2<f32> {
  let r = p - V.camPos;
  let z = dot(r, V.forward);
  return vec2<f32>(0.5 + 0.5 * dot(r, V.right) / (z * V.tanHalf * V.aspect), 0.5 - 0.5 * dot(r, V.up) / (z * V.tanHalf)) * V.viewport.zw + V.viewport.xy;
}
// The flame's opaque core; its warm fringe and the plume above glow but let a spark show through.
fn core(t: f32) -> f32 { return smoothstep(0.6, 1.2, t); }
// Transmittance from the spark to the camera through the smoke and the flame's core, 0 behind a solid.
fn transmittance(p: vec3<f32>) -> f32 {
  let dir = normalize(V.camPos - p);
  let inv = 1.0 / dir;
  let t0 = (V.boxMin - p) * inv;
  let t1 = (V.boxMax - p) * inv;
  let tExit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  let span = tExit / f32(OCCLUSION_STEPS);
  let size = V.boxMax - V.boxMin;
  var depth = 0.0;
  for (var k = 0; k < OCCLUSION_STEPS; k++) {
    let q = (p + dir * (f32(k) + 0.5) * span - V.boxMin) / size;
    if (P.mode == 3.0 && solidDistance(q * P.size - vec3<f32>(0.5)) < 0.0) { return 0.0; }
    let s = textureSampleLevel(sca, lin, q, 0.0);
    depth += (s.x * V.density + V.fireDensity * core(s.y)) * span;
  }
  return exp(-depth);
}

@vertex
fn vs(@builtin(vertex_index) corner: u32, @builtin(instance_index) i: u32) -> Out {
  var out: Out;
  out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  let s = sparks[i];
  if (s.age >= s.life) { return out; }
  let p = toWorld(s.pos);
  if (dot(p - V.camPos, V.forward) < 0.05) { return out; }
  let head = toPixel(p);
  let tail = toPixel(p - s.vel / P.size.x * STREAK);
  let len = distance(head, tail);
  let along = select((tail - head) / len, vec2<f32>(1.0, 0.0), len < 1e-3);
  let across = vec2<f32>(-along.y, along.x);
  // About two pixels wide at 1080p, scaled with the frame, never thinner than a pixel and a half.
  let w = max(0.75, min(V.viewport.z, V.viewport.w) / 1080.0);
  let end = select(head - along * w, tail + along * w, corner >= 2u);
  let px = end + across * select(w, -w, (corner & 1u) == 1u);
  out.position = vec4<f32>(px / V.viewport.zw * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.5, 1.0);
  // Cools from yellow through orange to dull red over its life and winks out at the end.
  let lived = s.age / s.life;
  let heat = HEAT * exp(-COOLING * lived);
  let seen = transmittance(p);
  // A thinner or longer streak spreads the same light over fewer pixels.
  let glow = blackbody(heat) * smoothstep(1.0, 0.8, lived) * GAIN * V.exposure * seen / (w * (1.0 + len / 32.0));
  out.segment = vec4<f32>(head, tail);
  out.colour = vec4<f32>(glow, w);
  out.seen = seen;
  return out;
}

fn profile(in: Out) -> f32 {
  let a = in.segment.xy;
  let ab = in.segment.zw - a;
  let t = clamp(dot(in.position.xy - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  let d = distance(in.position.xy, a + ab * t) / in.colour.w;
  let x = max(1.0 - d * d, 0.0);
  return x * x;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  let k = profile(in);
  return vec4<f32>(tonemap(in.colour.rgb * k), in.seen * k);
}

// Into the export's fire layer, linear and premultiplied like the raymarch's, scaled by the sample weight.
@fragment
fn fsExport(in: Out) -> @location(0) vec4<f32> {
  let k = profile(in);
  return vec4<f32>(in.colour.rgb * k, in.seen * k) * V.sampleWeight;
}`;
