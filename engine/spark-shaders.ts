import { BLACKBODY, HEADER, PARAMS_STRUCT, SOLIDS, TRACE } from './simulation-shaders';

export const SPARK_COUNT = 16384;
// A fresh spark's temperature and the one it goes out at, on the flame's blackbody scale.
const SPARK_HEAT = 1.65;
const SPARK_OUT = 0.30;

// One spark: position and velocity in grid cells, age and lifetime in seconds. Dead once its age reaches its life.
const SPARK_STRUCT = /* wgsl */ `
struct Spark { pos: vec3<f32>, age: f32, vel: vec3<f32>, life: f32 }
`;

// Moves every spark one step and respawns dead ones from the burning gas: a steady trickle from anywhere the fire
// burns, low in the fuel mostly but up to the flame's tips, and now and then a pop that throws a cluster from one
// spot. Spawns are rolled from a hash of the spark's index, the simulation time and the seed, so an export plays
// back the same sparks every time.
export const SPARK_STEP = HEADER + SPARK_STRUCT + /* wgsl */ `
// Per-spark spawn chance per second, the burst an explosion throws per unit of blast, and the sparks slider.
struct Spawn { rate: f32, burst: f32, strength: f32, size: f32, life: f32 }
@group(0) @binding(1) var<storage, read_write> sparks: array<Spark, ${SPARK_COUNT}>;
@group(0) @binding(2) var vel: texture_3d<f32>;
@group(0) @binding(3) var sca: texture_3d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var<uniform> S: Spawn;
${TRACE}
// Gravity and smooth aerodynamic flutter in base cells/s², and the pop that throws a fragment out of the fuel
// in base cells/s. Its response to the gas depends on its individual size.
const GRAVITY = 85.0;
const TUMBLE = 8.0;
const POP = 80.0;
const POP_UP = 25.0;
const BURST_SPEED = 220.0;
const BURST_UP = 60.0;
const SKIN = 2.5;
// Pops come about every POP_EVERY seconds, most of the time, last POP_WINDOW and throw POP_MIN to POP_MAX sparks
// at full strength, with a shared kick of POP_THROW base cells/s that leans upward.
const POP_EVERY = 0.5;
const POP_WINDOW = 0.12;
const POP_CHANCE = 0.7;
const POP_MIN = 18.0;
const POP_MAX = 75.0;
const POP_THROW = 140.0;

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
  return s.w > 1.0 || s.y > 0.6;
}
// A point in the column above the emitter: within two and a half radii of its axis, from the fuel up to near the
// lid, most of them low down where the fuel burns.
fn columnPoint(u: vec3<f32>) -> vec3<f32> {
  let angle = 6.2831853 * u.x;
  let r = 2.5 * P.radius * sqrt(u.y);
  let bottom = P.emitter.y - P.radius;
  return vec3<f32>(P.emitter.x + r * cos(angle) * P.width, bottom + (0.85 * P.size.y - bottom) * pow(u.z, 1.5), P.emitter.z + r * sin(angle) * P.width);
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
// Where a spark may be born, before the burning test. An explosion samples the local domain because its moving
// window may have left the original charge behind. Other flames use the source column or the objects' skin.
fn candidate(base: u32, k: u32) -> vec3<f32> {
  let u = vec3<f32>(rand(base, k), rand(base, k + 1u), rand(base, k + 2u));
  if (P.mode == 2.0) {
    // Rejection by burning() finds occupied hot gas anywhere in the moving window; the old world-space source
    // can be outside it after the blast. No emitter-centred sampling volume follows that source out of the grid.
    return u * (P.size - vec3<f32>(1.0));
  }
  if (P.mode != 3.0) { return columnPoint(u); }
  if (rand(base, k + 4u) < 0.5) { return vec3<f32>(u.x * P.size.x, pow(u.y, 1.5) * 0.85 * P.size.y, u.z * P.size.z); }
  let i = pickBox(rand(base, k + 3u));
  if (i < 0) { return vec3<f32>(-10.0); }
  let placement = P.objects[2 * i];
  let local = (2.0 * u - vec3<f32>(1.0)) * (P.objects[2 * i + 1].xyz + vec3<f32>(SKIN * cellScale()));
  let c = cos(placement.w);
  let s = sin(placement.w);
  let p = placement.xyz + vec3<f32>(c * local.x - s * local.z, local.y, s * local.x + c * local.z);
  let sd = solidDistance(p);
  return select(vec3<f32>(-10.0), p, sd >= 0.0 && sd < SKIN * cellScale());
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  var s = sparks[i];
  let base = pcg(i ^ pcg(bitcast<u32>(P.time) ^ pcg(bitcast<u32>(P.seed) + 0x9e3779b9u)));
  let cells = cellScale();
  // Small embers entrain in the gas quickly; heavier fragments keep their ballistic launch and settle.
  // The same stable size class also controls projected area and cooling in the draw pass.
  let material = pcg(i ^ pcg(bitcast<u32>(P.seed) + 0x68bc21ebu));
  let grain = rand(material, 0u);
  let size = S.size * (0.45 + 1.8 * grain * grain * grain);
  let dragTime = clamp(0.06 + 0.3 * size * size, 0.04, 1.6);
  if (s.age < s.life) {
    // Fragments can leave the fluid grid without disappearing. Outside it the unresolved air is ambient wind;
    // sampling a clamped edge texel would instead pin them to the last hot or fast cell on that boundary.
    var gas = P.wind;
    if (insideBox(s.pos)) { gas = velocityAt(s.pos); }
    var v = mix(s.vel, gas, 1.0 - exp(-P.dt / dragTime));
    v.y -= GRAVITY * cells * P.dt;
    let phase = vec3<f32>(rand(material, 1u), rand(material, 2u), rand(material, 3u)) * 6.2831853;
    v += sin(vec3<f32>(7.1, 9.3, 5.7) * P.time + phase) * TUMBLE * cells * P.dt / sqrt(max(size, 0.1));
    // Position is local to the translating computational window. Velocity remains in the world frame for
    // gas entrainment, ballistic inertia and the camera shutter; subtract only the window's transport here.
    s.pos += (v - P.transport) * P.dt;
    s.vel = v;
    s.age += P.dt;
    if (s.pos.y < -0.5 || isSolid(s.pos)) { s.age = s.life; }
    sparks[i] = s;
    return;
  }
  // This step's pop, the same for every spark: whether the window pops at all, when, from where and how hard.
  let window = floor(P.time / POP_EVERY);
  let pb = pcg(u32(window) ^ pcg(bitcast<u32>(P.seed) + 0x3c6ef372u));
  let popAt = (window + rand(pb, 0u) * (1.0 - POP_WINDOW / POP_EVERY)) * POP_EVERY;
  if (rand(pb, 1u) < POP_CHANCE && P.time >= popAt && P.time < popAt + POP_WINDOW) {
    let site = candidate(pb, 2u);
    let count = mix(POP_MIN, POP_MAX, rand(pb, 7u));
    if (rand(base, 60u) < S.strength * count / (${SPARK_COUNT}.0 * POP_WINDOW) * P.dt && insideBox(site) && burning(site)) {
      let p = site + unitVector(rand(base, 61u), rand(base, 62u)) * 0.5 * P.radius * pow(rand(base, 63u), 1.0 / 3.0);
      let kick = normalize(unitVector(rand(pb, 8u), rand(pb, 9u)) + vec3<f32>(0.0, 0.8, 0.0));
      let r = rand(base, 64u);
      s.pos = p;
      s.vel = velocityAt(p) + (kick * (0.3 + 0.7 * rand(base, 65u)) + unitVector(rand(base, 66u), rand(base, 67u)) * 0.4) * POP_THROW * cells;
      s.life = S.life * sqrt(size) * (0.5 + 1.9 * r * r);
      s.age = 0.0;
      sparks[i] = s;
      return;
    }
  }
  if (rand(base, 0u) < S.rate * P.dt) {
    let attempts = select(4u, 12u, P.mode == 2.0);
    for (var k = 0u; k < attempts; k++) {
      let p = candidate(base, 40u + 5u * k);
      if (!insideBox(p) || !burning(p)) { continue; }
      // Most pops are gentle, a few fling the spark well clear of the flame.
      let r = rand(base, 24u);
      let pop = rand(base, 27u);
      s.pos = p;
      s.vel = velocityAt(p) + unitVector(rand(base, 25u), rand(base, 26u)) * POP * cells * pop * pop + vec3<f32>(0.0, POP_UP * cells * (0.5 + rand(base, 28u)), 0.0);
      s.life = S.life * sqrt(size) * (0.45 + 2.2 * r * r);
      s.age = 0.0;
      sparks[i] = s;
      return;
    }
  } else if (P.mode == 2.0 && rand(base, 1u) < S.burst * P.blast * P.dt) {
    // The blast throws burning debris out of the charge, outward and up.
    let dir = unitVector(rand(base, 2u), rand(base, 3u));
    let speed = rand(base, 5u);
    s.pos = P.emitter + dir * P.radius * vec3<f32>(P.width, 1.0, P.width) * 0.8 * pow(rand(base, 4u), 1.0 / 3.0);
    s.vel = dir * BURST_SPEED * cells * (0.15 + 0.85 * speed * speed) + vec3<f32>(0.0, BURST_UP * cells, 0.0);
    s.life = S.life * sqrt(size) * (0.35 + 1.8 * rand(base, 6u));
    s.age = 0.0;
    sparks[i] = s;
  }
}`;

// Draws each live spark as a tiny point of blackbody colour with a faint bloom, shrinking as it burns down; the few
// big fragments and the rare defocused ember are larger and dimmer, a fast one smears into a thin streak, and all are
// dimmed by the smoke and flame between them and the camera. One quad per spark, strip order.
export const SPARK_DRAW = BLACKBODY + SPARK_STRUCT + /* wgsl */ `
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
  blue: f32,
  timeOffset: f32,
  pad: f32,
  shift: vec2<f32>,
  appearance: vec4<f32>,
}
@group(0) @binding(0) var<uniform> V: View;
` + PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var sca: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
@group(0) @binding(4) var<storage, read> sparks: array<Spark, ${SPARK_COUNT}>;
// How much of the sparks this draw shows: a looping export fades them out over its tail and in over its head.
struct Fade { weight: f32 }
@group(0) @binding(5) var<uniform> F: Fade;
` + SOLIDS + /* wgsl */ `
// Preview integrates a short exposure analytically. Export samples the selected shutter in time, so it must
// draw a point at each sample instead of blurring the particle a second time (or blurring when the shutter is off).
const PREVIEW_SHUTTER = 0.0083333;
const GAIN = 2.2;
const OCCLUSION_STEPS = 20;
const HEAT = ${SPARK_HEAT.toFixed(2)};
const COOLING = ${Math.log(SPARK_HEAT / SPARK_OUT).toFixed(4)};

struct Out {
  @builtin(position) position: vec4<f32>,
  // The dot's centre and streak end in pixels; its colour and radius; how much of it the gas lets through.
  @location(0) @interpolate(flat) segment: vec4<f32>,
  @location(1) @interpolate(flat) colour: vec4<f32>,
  @location(2) @interpolate(flat) seen: f32,
}

fn toWorld(cells: vec3<f32>) -> vec3<f32> { return V.boxMin + (cells + vec3<f32>(0.5)) / P.size * (V.boxMax - V.boxMin); }
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn rand(base: u32, k: u32) -> f32 { return f32(pcg(base + k)) / 4294967296.0; }
// Pixel coordinates of a world point, shifted by the sample's jitter the way the raymarch's rays are.
fn toPixel(p: vec3<f32>) -> vec2<f32> {
  let r = p - V.camPos;
  let z = dot(r, V.forward);
  let ndc = vec2<f32>(dot(r, V.right) / (z * V.tanHalf * V.aspect), dot(r, V.up) / (z * V.tanHalf)) + V.shift;
  return vec2<f32>(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * V.viewport.zw + V.viewport.xy;
}
// Transmittance from the spark to the camera through the smoke and the flame's core, 0 behind a solid.
fn transmittance(p: vec3<f32>) -> f32 {
  let toCamera = V.camPos - p;
  let distanceToCamera = length(toCamera);
  let dir = toCamera / max(distanceToCamera, 1e-6);
  // Integrate only the ray segment inside the actual fluid volume. This also works for a ballistic fragment
  // outside the moving box: a ray that misses it is clear, while one crossing the plume still loses light.
  var enter = 0.0;
  var leave = distanceToCamera;
  for (var axis = 0; axis < 3; axis++) {
    if (abs(dir[axis]) < 1e-6) {
      if (p[axis] < V.boxMin[axis] || p[axis] > V.boxMax[axis]) { return 1.0; }
    } else {
      let near = (V.boxMin[axis] - p[axis]) / dir[axis];
      let far = (V.boxMax[axis] - p[axis]) / dir[axis];
      enter = max(enter, min(near, far));
      leave = min(leave, max(near, far));
    }
  }
  if (leave <= enter) { return 1.0; }
  let span = (leave - enter) / f32(OCCLUSION_STEPS);
  let size = V.boxMax - V.boxMin;
  var depth = 0.0;
  for (var k = 0; k < OCCLUSION_STEPS; k++) {
    let q = (p + dir * (enter + (f32(k) + 0.5) * span) - V.boxMin) / size;
    if (P.mode == 3.0 && solidDistance(q * P.size - vec3<f32>(0.5)) < 0.0) { return 0.0; }
    let s = textureSampleLevel(sca, lin, q, 0.0);
    let incandescentSoot = 0.12 * V.fireDensity * min(1.5, 0.35 * s.x + 0.015 * s.w) * flame(s.y);
    depth += (s.x * V.density + incandescentSoot) * span;
  }
  return exp(-depth);
}

@vertex
fn vs(@builtin(vertex_index) corner: u32, @builtin(instance_index) i: u32) -> Out {
  var out: Out;
  out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  let s = sparks[i];
  let age = s.age - V.timeOffset;
  if (age < 0.0 || age >= s.life) { return out; }
  let p = toWorld(s.pos - s.vel * V.timeOffset);
  let distanceToCamera = dot(p - V.camPos, V.forward);
  if (distanceToCamera < 0.05) { return out; }
  let material = pcg(i ^ pcg(bitcast<u32>(P.seed) + 0x68bc21ebu));
  let grain = rand(material, 0u);
  let lived = age / s.life;
  let head = toPixel(p);
  let worldVelocity = s.vel / P.size * (V.boxMax - V.boxMin);
  let shutter = select(PREVIEW_SHUTTER, 0.0, V.frameTime > 0.0);
  let tail = toPixel(p - worldVelocity * shutter);
  // A world-space fragment becomes smaller with distance and grows when the lens zooms in. Subpixel fragments
  // keep a smooth reconstruction footprint, with their energy reduced by their actual projected area.
  let physicalRadius = 0.00042 * V.appearance.z * (0.45 + 1.8 * grain * grain * grain) * (1.0 - 0.45 * lived);
  let projected = physicalRadius * 0.5 * V.viewport.w / (distanceToCamera * V.tanHalf);
  let radius = max(projected, 0.65);
  let travel = distance(head, tail);
  let along = select((tail - head) / max(travel, 1e-3), vec2<f32>(1.0, 0.0), travel < 1e-3);
  let across = vec2<f32>(-along.y, along.x);
  let reach = 3.0 * radius;
  let end = select(head - along * reach, tail + along * reach, corner >= 2u);
  let px = end + across * select(reach, -reach, (corner & 1u) == 1u);
  out.position = vec4<f32>(px / V.viewport.zw * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.5, 1.0);
  let heat = HEAT * (0.78 + 0.48 * rand(material, 4u)) * exp(-COOLING * lived);
  // Aerodynamic tumble changes apparent emitting area smoothly; it does not choose a new random colour per frame.
  let phase = 6.2831853 * rand(material, 5u);
  let flicker = 0.72 + 0.18 * sin(age * (9.0 + 15.0 * rand(material, 6u)) + phase)
    + 0.10 * sin(age * (21.0 + 17.0 * rand(material, 7u)) + phase);
  let seen = transmittance(p) * F.weight;
  let area = min(1.0, projected * projected / (radius * radius));
  let smear = 1.0 + travel / (2.0 * radius);
  let fade = smoothstep(0.0, 0.035, age) * (1.0 - smoothstep(0.72, 1.0, lived));
  let emission = blackbody(heat) * GAIN * flicker * fade * area * V.exposure * seen / smear;
  out.segment = vec4<f32>(head, tail);
  out.colour = vec4<f32>(emission, radius);
  out.seen = seen;
  return out;
}

fn profile(in: Out) -> f32 {
  let a = in.segment.xy;
  let ab = in.segment.zw - a;
  let t = clamp(dot(in.position.xy - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
  let d = distance(in.position.xy, a + ab * t) / in.colour.w;
  // A tight bright core with the faint bloom of a hot point.
  return exp(-2.5 * d * d) + 0.025 * exp(-0.5 * d * d);
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  let k = profile(in);
  // Emitting particles add radiance; they do not punch a dark alpha-shaped hole in the footage.
  return vec4<f32>(in.colour.rgb * k, 0.0);
}

// Into the export's fire layer, linear and premultiplied like the raymarch's, scaled by the sample weight.
@fragment
fn fsExport(in: Out) -> @location(0) vec4<f32> {
  let k = profile(in);
  return vec4<f32>(in.colour.rgb * k, 0.0) * V.sampleWeight;
}`;
