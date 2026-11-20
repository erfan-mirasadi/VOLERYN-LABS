import { DENSITY, FLOW_PERIOD } from './scene';
export { BLACKBODY } from './blackbody';
import { BLACKBODY, FIRE_RADIANCE } from './blackbody';

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
  blast: f32,
  ignite: f32,
  seed: f32,
  width: f32,
  disturbance: f32,
  objects: array<vec4<f32>, 8>,
  // Reference Kelvin at unit heat, smoke shading density, flame opacity, padding.
  appearance: vec4<f32>,
  // Moving-window velocity in current-grid cells/s. Velocity textures retain absolute world velocity.
  transport: vec3<f32>,
  cellsPerUnit: f32,
  originCells: vec3<f32>,
  originPad: f32,
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
// Cells per base-grid cell: lengths, speeds and noise periods written in base cells are scaled by it.
fn cellScale() -> f32 { return P.cellsPerUnit / 64.0; }
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
// Seed 0 shifts nothing, so the default look stays; any other seed moves the stirring and flicker noise elsewhere.
fn seedShift() -> vec3<f32> { return vec3<f32>(fract(P.seed * 0.618034) * 97.0, 0.0, fract(P.seed * 0.754878) * 89.0); }
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
// The charge is a porous, seeded volume, not a hot sphere with noise painted on its surface.
// Shape is shared by sourcing and blast expansion so their lobes move together.
fn sourceShape(p: vec3<f32>) -> vec4<f32> {
  let local = (p - P.emitter) / max(P.radius * vec3<f32>(P.width, 1.0, P.width), vec3<f32>(0.01));
  let baseDistance = length(local);
  if (baseDistance > 1.6 || P.mode == 3.0) { return vec4<f32>(baseDistance, 1.0, 1.0, 0.0); }
  if (P.mode != 2.0) {
    // A seed chooses a coherent arrangement of source hot spots, not just the phase of flicker.
    let angle = 6.2831853 * fract(P.seed * 0.618034);
    let rotated = vec3<f32>(cos(angle) * local.x + sin(angle) * local.z, local.y, cos(angle) * local.z - sin(angle) * local.x);
    let spread = 0.3 + 0.28 * fract(P.seed * 0.374761);
    let a = rotated - vec3<f32>(spread, 0.0, 0.18);
    let b = rotated - vec3<f32>(-0.36, 0.15, -spread);
    let c = rotated - vec3<f32>(0.05, -0.15, -0.35);
    let spots = exp(-7.0 * dot(a, a)) + 0.85 * exp(-9.0 * dot(b, b)) + 0.65 * exp(-11.0 * dot(c, c));
    let contour = 0.8 + 0.35 * noise3(local * 2.2 + seedShift() * 0.11);
    return vec4<f32>(baseDistance / contour, 0.12 + 1.8 * min(spots, 1.4), 1.0, 0.0);
  }
  let offset = seedShift() * 0.13;
  let broad = noise3(local * 2.1 + offset);
  let fine = noise3(local * 5.3 + offset + vec3<f32>(7.1));
  let warp = vec3<f32>(noise3(local * 2.7 + offset), noise3(local * 2.7 + offset + vec3<f32>(19.3)), noise3(local * 2.7 + offset + vec3<f32>(41.7))) - vec3<f32>(0.5);
  let radius = 0.7 + 0.55 * broad;
  let distance = length(local + 0.3 * warp) / radius;
  // The rich pockets are cool and sooty; mixed pockets can ignite. Their coherent source masks
  // become advected scalar fields, rather than fresh noise painted into each later frame.
  let mixture = 0.65 * broad + 0.35 * fine;
  let porosity = 0.04 + 2.2 * smoothstep(0.38, 0.67, mixture);
  let richness = smoothstep(0.7, 1.5, porosity);
  let ignition = smoothstep(0.38, 0.68, noise3(local * 3.7 + offset + vec3<f32>(4.3, 9.1, 2.8)));
  return vec4<f32>(distance, porosity, ignition, richness);
}
` + SOLIDS;

// Velocity lives on the cell faces: texel c holds the flow through its lower x, y and z faces.
// The floor and solids block their faces. Side boundaries prescribe the ambient wind's normal
// flow at BOTH sides; their pressure is Neumann, so projection changes the interior flow around obstacles
// without cancelling that through-flow. The top stays open. Needs the solid mask bound as solid.
const BLOCKED = /* wgsl */ `
fn solidCell(c: vec3<i32>) -> bool { return textureLoad(solid, clampc(c), 0).x > 0.5; }
fn blocked(c: vec3<i32>) -> vec3<bool> {
  let here = solidCell(c);
  return vec3<bool>(here || solidCell(c - X), c.y == 0 || here || solidCell(c - Y), here || solidCell(c - Z));
}
`;

// Reads the face-stored velocity at any point and traces where the gas there came from; needs vel and lin bound.
export const TRACE = /* wgsl */ `
fn faceOffset(axis: i32) -> vec3<f32> { return 0.5 * vec3<f32>(f32(axis == 0), f32(axis == 1), f32(axis == 2)); }
fn velocityAt(p: vec3<f32>) -> vec3<f32> {
  if (p.x < -0.5 || p.z < -0.5 || p.x > P.size.x - 0.5 || p.z > P.size.z - 0.5) { return P.wind; }
  return vec3<f32>(
    textureSampleLevel(vel, lin, coord(p + faceOffset(0)), 0.0).x,
    textureSampleLevel(vel, lin, coord(p + faceOffset(1)), 0.0).y,
    textureSampleLevel(vel, lin, coord(p + faceOffset(2)), 0.0).z);
}
// Where the gas now at p was one step ago, by the midpoint rule; a negative dt asks where it will be.
fn trace(p: vec3<f32>, dt: f32) -> vec3<f32> {
  let mid = p - 0.5 * dt * (velocityAt(p) - P.transport);
  return p - dt * (velocityAt(mid) - P.transport);
}
`;

export const SHADERS = {
  initializeVelocity: /* wgsl */ `
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  // A newly created explosion is immersed in already-moving ambient air. The following pressure
  // projection still adds thermal motion and enforces the floor and any normal flow constraints.
  textureStore(dst, c, vec4<f32>(P.wind, 0.0));
}`,
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
  let back = trace(vec3<f32>(c), P.dt);
  let outside = any(back < vec3<f32>(-0.5)) || any(back > P.size - vec3<f32>(0.5));
  // Incoming ambient air carries no soot, heat or fuel. Clamping an inflow to an edge cell
  // otherwise keeps re-injecting smoke and stretches it into a sheet along the domain wall.
  textureStore(dst, c, select(textureSampleLevel(src, lin, coord(back), 0.0), vec4<f32>(0.0), outside));
}`,
  correct: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var old: texture_3d<f32>;
@group(0) @binding(3) var hat: texture_3d<f32>;
@group(0) @binding(4) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var lin: sampler;
${TRACE}
// How fast soot burns out in gas hot enough to oxidise it, per second.
const SOOT_BURNOUT = 8.0;
// The share of the cooling rate left to gas too cool to glow.
const AFTERHEAT_COOLING = 0.04;
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
  if (any(back < vec3<f32>(-0.5)) || any(back > P.size - vec3<f32>(0.5))) { q = vec4<f32>(0.0); }
  // Stable heat diffusion mixes the advected thermal field with its six neighbours. The coefficient
  // scales with cells per world length squared; an exponential convex blend remains bounded on every grid.
  let neighbourHeat = (textureLoad(hat, clampc(c + X), 0).y + textureLoad(hat, clampc(c - X), 0).y + textureLoad(hat, clampc(c + Y), 0).y + textureLoad(hat, clampc(c - Y), 0).y + textureLoad(hat, clampc(c + Z), 0).y + textureLoad(hat, clampc(c - Z), 0).y) / 6.0;
  q.y = mix(q.y, neighbourHeat, 1.0 - exp(-3.0 * cellScale() * cellScale() * P.dt));
  // Fuel burns where the gas is hot enough to ignite, into heat and soot; the heat then fades.
  // Reduced mixing-limited combustion: rich pockets cannot consume their whole charge at once.
  // The neighbouring fuel contrast is an entrainment proxy, not a separately solved oxygen field.
  let fuelGradient = 0.5 * vec3<f32>(textureLoad(old, clampc(c + X), 0).z - textureLoad(old, clampc(c - X), 0).z, textureLoad(old, clampc(c + Y), 0).z - textureLoad(old, clampc(c - Y), 0).z, textureLoad(old, clampc(c + Z), 0).z - textureLoad(old, clampc(c - Z), 0).z) / cellScale();
  let mixing = clamp(0.18 + 0.82 / (1.0 + 0.7 * q.z) + 0.25 * length(fuelGradient), 0.18, 1.0);
  let rate = P.burn * mixing * smoothstep(P.ignite, P.ignite + 0.25, q.y);
  let kineticBurn = q.z * (1.0 - exp(-rate * P.dt));
  // An explosion's concentrated charge is mixing-limited: entrained air, not the amount of fuel
  // in a cell, bounds instantaneous heat release. This prevents all dense pockets flashing white together.
  // Concentrations and rates are grid independent; the exposed burn-rate multiplier also scales this limit.
  let supply = 0.15 * P.burn * mixing * P.dt;
  let burned = select(kineticBurn, min(kineticBurn, supply), P.mode == 2.0);
  q.z -= burned;
  // The burn rate rides in w so the pressure solve can let burning gas expand.
  q.w = burned / P.dt;
  // A reduced cooling curve loses heat quickly while incandescent and slowly after it goes dark.
  // Retaining this afterheat carries the smoke upward; this is not a resolved radiative energy solve.
  let hot = q.y + burned * P.heat;
  q.y = hot * exp(-P.cooling * mix(AFTERHEAT_COOLING, 1.0, smoothstep(0.05, 0.2, hot)) * P.dt);
  // Soot forms wherever fuel burns and oxidises while the gas around it stays hot. Oxidation slows
  // as the gas cools: soot left at the tips and in air-quenched eddies becomes smoke.
  // These thresholds are reduced-model values, independent of the artistic Kelvin calibration.
  // Smoke that has no flame under it is not soot on its way through one, so it keeps what it has.
  let burnout = select(0.0, SOOT_BURNOUT * smoothstep(0.5, 0.8, q.y), P.burn > 0.0);
  q.x = (q.x + burned * P.smokeYield) * exp(-(burnout + P.dissipation) * P.dt);
  let k = cellScale();
  let flicker = noise3((p + P.originCells) * 0.3 / k + vec3<f32>(0.0, -P.time * 4.0, P.time * 1.3) + seedShift());
  if (P.mode == 3.0) {
    // Solids give off fuel from a thin skin around them, so the flames take their shape.
    let sd = solidDistance(p);
    if (sd < 0.0) {
      q = vec4<f32>(0.0);
    } else if (sd < 2.5 * k) {
      // Upper faces burn, undersides smoulder, and hot spots wander slowly along the log.
      let up = solidDistance(p + vec3<f32>(0.0, 0.75 * k, 0.0)) - solidDistance(p - vec3<f32>(0.0, 0.75 * k, 0.0));
      let spot = noise3((p + P.originCells) * 0.5 / k + vec3<f32>(0.0, P.time * 0.3, 0.0) + seedShift());
      let skin = (0.25 + 0.75 * smoothstep(-0.5, 1.0, up)) * (0.4 + 0.6 * spot);
      let amount = (1.0 - sd / (2.5 * k)) * (0.35 + 1.3 * flicker) * P.emit * P.dt * skin;
      q.z += amount;
      q.y += amount * 0.25;
    }
  } else {
    let shape = sourceShape(p);
    let d = shape.x;
    if (d < 1.0 && P.emit > 0.0) {
      var amount = (1.0 - d * d) * (0.35 + 1.3 * flicker) * P.emit * P.dt;
      amount *= shape.y;
      if (P.mode == 1.0) {
        q.x += amount * 0.8;
        q.y += amount * 0.5;
      } else {
        q.z += amount;
        if (P.mode == 2.0) {
          // Ignite pockets of the charge; combustion supplies the rest of the heat over time.
          // Heating every fuel voxel at full strength made an opaque uniform yellow ball immediately.
          let ignition = (0.12 + 0.7 * shape.z * (1.0 - 0.7 * shape.w)) * smoothstep(0.0, 0.3, 1.0 - d);
          q.y = max(q.y, ignition);
          // A practical fuel charge includes cool particulate products. Keeping these concentrated in
          // rich pockets gives the advancing luminous sheets real dark volume to roll around and light.
          q.x += amount * P.smokeYield * 0.1 * shape.w;
        } else {
          q.y += amount * 0.25;
        }
      }
    }
  }
  // Adaptive bounds keep occupied gas away from the approximate open boundary; no material sponge
  // erases the upper plume before its domain has a chance to grow.
  textureStore(dst, c, max(q, vec4<f32>(0.0)));
}`,
  advectFlow: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var srcA: texture_3d<f32>;
@group(0) @binding(3) var srcB: texture_3d<f32>;
@group(0) @binding(4) var dstA: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var dstB: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var lin: sampler;
${TRACE}
const FLOW_PERIOD = ${FLOW_PERIOD.toFixed(2)};
// Two displacement fields ride the flow and take turns resetting to zero, so the detail noise the render looks up
// through them travels with the gas and never stretches for more than a period. They live on half the grid.
fn resets(offset: f32) -> bool { return floor(P.time / FLOW_PERIOD + offset) != floor((P.time - P.dt) / FLOW_PERIOD + offset); }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (any(c >= vec3<i32>(textureDimensions(dstA)))) { return; }
  let p = 2.0 * vec3<f32>(c) + vec3<f32>(0.5);
  let back = trace(p, P.dt);
  let shift = back - p - P.transport * P.dt;
  let a = textureSampleLevel(srcA, lin, coord(back), 0.0).xyz + shift;
  let b = textureSampleLevel(srcB, lin, coord(back), 0.0).xyz + shift;
  textureStore(dstA, c, vec4<f32>(select(a, vec3<f32>(0.0), resets(0.0)), 0.0));
  textureStore(dstB, c, vec4<f32>(select(b, vec3<f32>(0.0), resets(0.5)), 0.0));
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
@group(0) @binding(4) var stir: texture_3d<f32>;
@group(0) @binding(5) var rep: sampler;
fn spin(c: vec3<i32>) -> f32 { return textureLoad(curl, clampc(c), 0).w; }
// A flame's body is lifted in step with its temperature. Gas that has stopped glowing is still hundreds of degrees
// hot, and by the gas law most of the weight a gas can lose is gone by twice the air's temperature, so below the
// glow the lift follows that curve instead of the straight line, which would leave the smoke column standing.
fn lift(t: f32) -> f32 { return max(t, 0.5 * t / (t + 0.1)); }
// The velocity each cell gains this step, taken at its centre; applyForces spreads it onto the faces.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  let k = cellScale();
  let s = textureLoad(sca, c, 0);
  // Soot weighs next to nothing in the gas that carries it; the mist of a smoke source is heavier and sinks a little.
  let weight = select(4.0 * k * s.x, 0.0, P.burn > 0.0);
  var v = vec3<f32>(0.0, P.dt * (P.buoyancy * lift(s.y) - weight), 0.0);
  // Vorticity confinement: push along N x omega to give back the swirls the coarse grid smooths away.
  let grad = 0.5 * vec3<f32>(spin(c + X) - spin(c - X), spin(c + Y) - spin(c - Y), spin(c + Z) - spin(c - Z));
  let len = length(grad);
  if (len > 1e-4) {
    v += P.dt * P.vorticity * cross(grad / len, textureLoad(curl, c, 0).xyz);
  }
  // Rising curl noise stirs the hot gas and the reaction sheet; being divergence-free it survives the projection.
  // A second octave, a third the size and rising faster, lives where the gas is hot and fuel is burning: the small
  // movement that keeps a flame's tongues alive without shaking the whole plume.
  let rise = vec3<f32>(0.0, -12.0 * k * P.time, 0.0) + k * seedShift();
  let stirring = textureSampleLevel(stir, rep, (p + P.originCells + rise) / ((18.0 + 15.0 * fract(P.seed * 0.317)) * k), 0.0).xyz;
  let rise2 = vec3<f32>(0.0, -27.0 * k * P.time, 0.0) + 2.0 * k * seedShift();
  let fine = textureSampleLevel(stir, rep, (p + P.originCells + rise2) / (9.0 * k), 0.0).xyz;
  // Coarse churning and fine breakup have independent controls. Disturbance lives near the thermal
  // front and weak smoke cap, so adding edge detail need not shake the entire fireball.
  let heatGradient = 0.5 * vec3<f32>(textureLoad(sca, clampc(c + X), 0).y - textureLoad(sca, clampc(c - X), 0).y, textureLoad(sca, clampc(c + Y), 0).y - textureLoad(sca, clampc(c - Y), 0).y, textureLoad(sca, clampc(c + Z), 0).y - textureLoad(sca, clampc(c - Z), 0).y) / k;
  let front = clamp(2.0 * length(heatGradient) + 0.1 * s.x, 0.0, 1.0);
  v += P.dt * P.turbulence * stirring * clamp(s.y + 0.3 * s.x + 0.05 * s.w, 0.0, 1.5);
  v += P.dt * (18.0 * k * P.disturbance) * fine * front;
  let d = sourceShape(p).x;
  if (d < 1.0) {
    v.y += P.dt * P.emit * 0.8 * k * (1.0 - d);
  }
  textureStore(dst, c, vec4<f32>(v, 0.0));
}`,
  applyForces: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var force: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var solid: texture_3d<f32>;
${BLOCKED}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let p = vec3<f32>(c);
  // Each face takes the mean of the two cells it separates.
  let here = textureLoad(force, c, 0).xyz;
  let below = vec3<f32>(textureLoad(force, clampc(c - X), 0).x, textureLoad(force, clampc(c - Y), 0).y, textureLoad(force, clampc(c - Z), 0).z);
  var v = textureLoad(vel, c, 0).xyz + 0.5 * (here + below);
  // Drag is relative to the ambient wind, rather than to a stationary box. The prescribed side
  // flux and Neumann pressure boundaries keep a constant wind from being projected away.
  v = P.wind + (v - P.wind) * exp(-P.damping * P.dt);
  if (c.x == 0) { v.x = P.wind.x; }
  if (c.z == 0) { v.z = P.wind.z; }
  textureStore(dst, c, vec4<f32>(select(v, vec3<f32>(0.0), blocked(c)), 0.0));
}`,
  divergence: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<r32float, write>;
@group(0) @binding(3) var sca: texture_3d<f32>;
@group(0) @binding(4) var solid: texture_3d<f32>;
${BLOCKED}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  if (solidCell(c)) {
    textureStore(dst, c, vec4<f32>(0.0));
    return;
  }
  let n = dims();
  let v = textureLoad(vel, c, 0).xyz;
  // The far side faces are not stored: they carry the same ambient normal flux as the near
  // faces. The open top repeats the face below and is the outlet for thermal expansion.
  let right = select(textureLoad(vel, clampc(c + X), 0).x, P.wind.x, c.x == n.x - 1);
  let above = select(textureLoad(vel, clampc(c + Y), 0).y, v.y, c.y == n.y - 1);
  let front = select(textureLoad(vel, clampc(c + Z), 0).z, P.wind.z, c.z == n.z - 1);
  let div = (right - v.x) + (above - v.y) + (front - v.z);
  // Burning gas expands, and a blast pushes out from the charge: the solve targets this much divergence instead of
  // none, so the push goes through the projection instead of being stripped by it. The blast is uneven, so the
  // fireball breaks into lobes instead of staying a ball.
  let shape = sourceShape(vec3<f32>(c));
  let expansion = P.expansion * textureLoad(sca, c, 0).w + P.blast * max(1.0 - shape.x * shape.x, 0.0) * shape.y;
  textureStore(dst, c, vec4<f32>(div - expansion, 0.0, 0.0, 0.0));
}`,
  project: /* wgsl */ `
@group(0) @binding(1) var vel: texture_3d<f32>;
@group(0) @binding(2) var pressure: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var solid: texture_3d<f32>;
${BLOCKED}
fn at(c: vec3<i32>) -> f32 { return textureLoad(pressure, clampc(c), 0).x; }
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = vec3<i32>(id);
  if (!inside(c)) { return; }
  let q = at(c);
  var v = textureLoad(vel, c, 0).xyz - vec3<f32>(q - at(c - X), q - at(c - Y), q - at(c - Z));
  if (c.x == 0) { v.x = P.wind.x; }
  if (c.z == 0) { v.z = P.wind.z; }
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
  let box = P.size / P.cellsPerUnit;
  let march = P.lightDir * 0.03 / box;
  let start = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(n);
  var q = start;
  var depth = 0.0;
  for (var i = 0; i < 256; i++) {
    q += march;
    if (any(q < vec3<f32>(0.0)) || any(q > vec3<f32>(1.0))) { break; }
    depth += textureSampleLevel(sca, lin, q, 0.0).x;
  }
  // A second march straight up: how much of the sky each cell still sees.
  var up = start;
  var depthUp = 0.0;
  for (var i = 0; i < 256; i++) {
    up.y += 0.03 / box.y;
    if (up.y > 1.0) { break; }
    depthUp += textureSampleLevel(sca, lin, up, 0.0).x;
  }
  let scale = 0.03 * ${DENSITY.toFixed(1)} * P.appearance.y;
  textureStore(dst, vec3<i32>(id), vec4<f32>(exp(-depth * scale), exp(-depthUp * scale), 0.0, 1.0));
}`,
};

// The flame's light, averaged over each block of fine cells into the coarse light volume; its alpha holds the block's
// occupancy (the most smoke or heat in it) so the raymarch can skip empty space. The block's mean smoke goes to a
// second volume, which the blur reads to let the light through thin smoke and hold it back in thick.
export const EMISSION = (scale: number) => PARAMS_STRUCT + BLACKBODY + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var sca: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var thick: texture_storage_3d<r32float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  let base = vec3<i32>(id) * ${scale};
  var sum = vec3<f32>(0.0);
  var smoke = 0.0;
  var occupancy = 0.0;
  for (var z = 0; z < ${scale}; z++) {
    for (var y = 0; y < ${scale}; y++) {
      for (var x = 0; x < ${scale}; x++) {
        let s = textureLoad(sca, base + vec3<i32>(x, y, z), 0);
        sum += blackbodyAt(s.y, P.appearance.x) * flame(s.y) * min(s.x * 0.35 + s.w * 0.015, 2.0) * P.appearance.z * P.appearance.w * ${FIRE_RADIANCE.toFixed(3)};
        smoke += s.x;
        occupancy = max(occupancy, max(s.x, s.y));
      }
    }
  }
  let cells = ${(scale * scale * scale).toFixed(1)};
  textureStore(dst, vec3<i32>(id), vec4<f32>(sum / cells, occupancy));
  textureStore(thick, vec3<i32>(id), vec4<f32>(smoke * P.appearance.y / cells, 0.0, 0.0, 0.0));
}`;

// Spreads the light volume one cell outwards; run a few times it stands in for light scattering through the smoke.
// Light arriving through a smoky cell is absorbed on the way, by absorb per unit of smoke, so thick smoke shades
// itself from the flame while thin smoke glows through. With dilate set the occupancy grows by a cell too.
export const LIGHT_BLUR = /* wgsl */ `
override dilate: u32;
override absorb: f32;
@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var thick: texture_3d<f32>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(textureDimensions(dst));
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  var sum = vec3<f32>(0.0);
  var weight = 0.0;
  var occupancy = textureLoad(src, c, 0).a;
  let here = textureLoad(thick, c, 0).x;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let w = 1.0 / (1.0 + f32(x * x + y * y + z * z));
        let p = c + vec3<i32>(x, y, z);
        if (all(p >= vec3<i32>(0)) && all(p < n)) {
          let tap = textureLoad(src, p, 0);
          let through = exp(-absorb * 0.5 * (here + textureLoad(thick, p, 0).x));
          sum += tap.rgb * w * through;
          if (dilate == 1u) { occupancy = max(occupancy, tap.a); }
        }
        weight += w;
      }
    }
  }
  textureStore(dst, c, vec4<f32>(sum / weight, occupancy));
}`;

// Blends the current field toward a kept one for the seamless loop; it runs on the scalars and the flow fields alike.
export const MIX = /* wgsl */ `
struct Blend { weight: f32 }
@group(0) @binding(0) var<uniform> B: Blend;
@group(0) @binding(1) var now: texture_3d<f32>;
@group(0) @binding(2) var kept: texture_3d<f32>;
@group(0) @binding(3) var dst: texture_storage_3d<rgba16float, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  let c = vec3<i32>(id);
  textureStore(dst, c, mix(textureLoad(now, c, 0), textureLoad(kept, c, 0), B.weight));
}`;

export const CLEAR = (format: 'rgba16float' | 'r32float') => /* wgsl */ `
@group(0) @binding(0) var dst: texture_storage_3d<${format}, write>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (any(id >= textureDimensions(dst))) { return; }
  textureStore(dst, vec3<i32>(id), vec4<f32>(0.0));
}`;
