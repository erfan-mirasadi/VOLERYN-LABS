import { GRID, type Box, type Preset, type Tweaks } from './scene';

export type DomainLayout = { grid: [number, number, number]; cellsPerUnit: number; offset: [number, number, number] };
export type DomainProbe = { min: [number, number, number]; max: [number, number, number]; speed: [number, number, number]; occupied: number };
const TILE = 16;

// Keep the complete material envelope. At the dense memory budget the sampling density changes, not its bounds.
export function planDomain(layout: DomainLayout, probe: DomainProbe, preset: Preset, tweaks: Tweaks, boxes: Box[], maxCells: number, maxDimension: number, lookAhead = 0.12, preferredDensity = layout.cellsPerUnit): DomainLayout | null {
  const density = layout.cellsPerUnit;
  const low = layout.offset.slice();
  const high = layout.grid.map((n, axis) => layout.offset[axis] + n / density);
  const desiredLow = [Infinity, 0, Infinity], desiredHigh = [-Infinity, 0, -Infinity];
  const include = (a: number[], b: number[]) => {
    for (let axis = 0; axis < 3; axis++) { desiredLow[axis] = Math.min(desiredLow[axis], a[axis]); desiredHigh[axis] = Math.max(desiredHigh[axis], b[axis]); }
  };
  if (probe.occupied) {
    const padding = probe.speed.map((speed) => Math.max(8 / GRID[0], speed * lookAhead + 4 / GRID[0]));
    include(probe.min.map((cell, axis) => layout.offset[axis] + cell / density - padding[axis]), probe.max.map((cell, axis) => layout.offset[axis] + (cell + 1) / density + padding[axis]));
  }
  if (preset.mode !== 'explosion') {
    if (preset.mode === 'objects') {
      for (const box of boxes) {
        const radius = Math.hypot(box.hx, box.hz) + 6;
        include([(box.x - radius) / 64, Math.max(0, (box.y - box.hy - 6) / 64), (box.z - radius) / 64], [(box.x + radius) / 64, (box.y + box.hy + 6) / 64, (box.z + radius) / 64]);
      }
    } else {
      const radius = [preset.radius * tweaks.width + 8, preset.radius + 8, preset.radius * tweaks.width + 8];
      include(preset.emitter.map((c, axis) => (c - radius[axis]) / 64), preset.emitter.map((c, axis) => (c + radius[axis]) / 64));
    }
  }
  if (!Number.isFinite(desiredLow[0])) return null;
  desiredLow[1] = 0;
  for (const axis of [0, 2]) {
    if (desiredHigh[axis] - desiredLow[axis] < 1) {
      // Anchor a minimum-width envelope to the existing tile grid whenever it already fits.
      // Recentering it on every small hotspot movement causes needless reallocations.
      const anchor = Math.min(desiredLow[axis], Math.max(desiredHigh[axis] - 1, low[axis]));
      desiredLow[axis] = anchor; desiredHigh[axis] = anchor + 1;
    }
  }
  desiredHigh[1] = Math.max(1.5, desiredHigh[1]);
  const mustGrow = desiredLow.some((n, axis) => n < low[axis] - 1e-5) || desiredHigh.some((n, axis) => n > high[axis] + 1e-5);
  const currentVolume = high.reduce((v, n, axis) => v * (n - low[axis]), 1);
  const desiredVolume = desiredHigh.reduce((v, n, axis) => v * (n - desiredLow[axis]), 1);
  const nearCapacity = layout.grid.reduce((a, b) => a * b, 1) > maxCells * 0.7 || density < preferredDensity * 0.95;
  const canShrink = nearCapacity && currentVolume > desiredVolume * 1.8;
  if (!mustGrow && !canShrink) return null;
  // At capacity release empty tails instead of retaining every historical extent. A large volume-ratio
  // hysteresis avoids repeated crop/reallocation as the padded active region moves by a few cells.
  const requiredLow = desiredLow.map((n, axis) => nearCapacity || canShrink ? n : Math.min(n, low[axis]));
  const requiredHigh = desiredHigh.map((n, axis) => nearCapacity || canShrink ? n : Math.max(n, high[axis]));
  for (let axis = 0; axis < 3; axis++) {
    if (requiredLow[axis] < low[axis] - 1e-5 && axis !== 1) requiredLow[axis] -= TILE / density;
    if (requiredHigh[axis] > high[axis] + 1e-5) requiredHigh[axis] += TILE / density;
  }
  let nextDensity = canShrink ? preferredDensity : density;
  const make = () => {
    const origin = requiredLow.map((n, axis) => axis === 1 ? 0 : Math.floor(n * nextDensity / TILE + 1e-7) * TILE / nextDensity) as [number, number, number];
    const grid = requiredHigh.map((n, axis) => Math.max(TILE, Math.ceil((n - origin[axis]) * nextDensity / TILE - 1e-7) * TILE)) as [number, number, number];
    return { grid, cellsPerUnit: nextDensity, offset: origin };
  };
  let next = make(), attempts = 0;
  while ((next.grid.reduce((a, b) => a * b, 1) > maxCells || Math.max(...next.grid) > maxDimension) && attempts++ < 128) { nextDensity *= 0.94; next = make(); }
  if (attempts >= 128 || nextDensity < 2) throw new Error('The adaptive domain needs more memory than this device can provide at a usable sampling density. Reduce wind, source width or clip duration.');
  const changed = next.grid.some((n, axis) => n !== layout.grid[axis]) || Math.abs(next.cellsPerUnit - density) > 1e-6 || next.offset.some((n, axis) => Math.abs(n - layout.offset[axis]) > 1e-6);
  return changed ? next : null;
}

export const DOMAIN_PROBE = /* wgsl */ `
@group(0) @binding(0) var scalars: texture_3d<f32>;
@group(0) @binding(1) var velocity: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> bounds: array<atomic<u32>, 10>;
var<workgroup> reduced: array<atomic<u32>, 10>;
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) c: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
  if (lane < 10u) { atomicStore(&reduced[lane], select(0u, 0xffffffffu, lane < 3u)); }
  workgroupBarrier();
  let n = textureDimensions(scalars);
  if (all(c < n)) {
    let s = textureLoad(scalars, vec3<i32>(c), 0);
    if (max(s.x, max(0.25 * s.y, 0.1 * s.z)) >= 0.003) {
      let v = abs(textureLoad(velocity, vec3<i32>(c), 0).xyz);
      for (var axis = 0u; axis < 3u; axis++) {
        atomicMin(&reduced[axis], c[axis]); atomicMax(&reduced[3u + axis], c[axis]);
        atomicMax(&reduced[6u + axis], bitcast<u32>(v[axis]));
      }
      atomicAdd(&reduced[9], 1u);
    }
  }
  workgroupBarrier();
  if (lane < 10u && atomicLoad(&reduced[9]) > 0u) {
    let value = atomicLoad(&reduced[lane]);
    if (lane < 3u) { atomicMin(&bounds[lane], value); }
    else if (lane < 9u) { atomicMax(&bounds[lane], value); }
    else { atomicAdd(&bounds[lane], value); }
  }
}
`;

// Remap by world position. A same-density, tile-aligned growth samples precisely at old texel centres;
// capacity-driven regrids use trilinear filtering and convert velocity/displacements into the new cell units.
export const REMAP_VOLUME = /* wgsl */ `
struct Remap { oldOffset: vec3<f32>, oldDensity: f32, newOffset: vec3<f32>, newDensity: f32, oldGrid: vec3<f32>, kind: f32, newGrid: vec3<f32>, padding: f32, wind: vec3<f32>, pad2: f32 }
@group(0) @binding(0) var<uniform> R: Remap;
@group(0) @binding(1) var source: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var lin: sampler;
fn tap(q: vec3<f32>) -> vec4<f32> {
  if (any(q < vec3<f32>(0.0)) || any(q > vec3<f32>(1.0))) { return select(vec4<f32>(0.0), vec4<f32>(R.wind * (R.oldDensity / R.newDensity), 0.0), R.kind == 1.0); }
  return textureSampleLevel(source, lin, q, 0.0);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) c: vec3<u32>) {
  let n = textureDimensions(dst);
  if (any(c >= n)) { return; }
  let world = R.newOffset + (vec3<f32>(c) + vec3<f32>(0.5)) / vec3<f32>(n) * R.newGrid / R.newDensity;
  let q = (world - R.oldOffset) * R.oldDensity / R.oldGrid;
  var value = tap(q);
  let ratio = R.newDensity / R.oldDensity;
  if (R.kind == 1.0) {
    let delta = 0.5 * (vec3<f32>(1.0 / R.oldDensity - 1.0 / R.newDensity)) * R.oldDensity / R.oldGrid;
    value = vec4<f32>(tap(q + vec3<f32>(delta.x, 0.0, 0.0)).x, tap(q + vec3<f32>(0.0, delta.y, 0.0)).y, tap(q + vec3<f32>(0.0, 0.0, delta.z)).z, 0.0);
    value = vec4<f32>(value.xyz * ratio, 0.0);
  } else if (R.kind == 2.0) { value = vec4<f32>(value.xyz * ratio, 0.0); }
  textureStore(dst, vec3<i32>(c), value);
}
`;
