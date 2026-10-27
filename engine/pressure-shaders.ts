import { HEADER } from './simulation-shaders';

// The seven-point Laplacian with the top open: the ghost above the lid holds zero pressure, any other missing neighbour drops out.
// Every shader binds the level's pressure as "pressure" and has "scale", its cell size in simulation cells.
const STENCIL = /* wgsl */ `
fn solidAt(c: vec3<i32>) -> bool { return isSolid((vec3<f32>(c) + vec3<f32>(0.5)) * scale - vec3<f32>(0.5)); }
fn neighbour(c: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  if (c.y >= n.y) { return vec2<f32>(0.0, 1.0); }
  if (any(c < vec3<i32>(0)) || any(c >= n) || solidAt(c)) { return vec2<f32>(0.0); }
  return vec2<f32>(textureLoad(pressure, c).x, 1.0);
}
// The neighbours' pressure sum and how many of them count.
fn stencil(c: vec3<i32>, n: vec3<i32>) -> vec2<f32> {
  return neighbour(c - X, n) + neighbour(c + X, n) + neighbour(c - Y, n) + neighbour(c + Y, n) + neighbour(c - Z, n) + neighbour(c + Z, n);
}
`;

// Red-black Gauss-Seidel: a half-pass updates one colour from the other, so it runs in place.
export const SMOOTH = HEADER + /* wgsl */ `
override parity: u32;
override scale: f32;
@group(0) @binding(1) var pressure: texture_storage_3d<r32float, read_write>;
@group(0) @binding(2) var rhs: texture_3d<f32>;
${STENCIL}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(textureDimensions(pressure));
  var c = vec3<i32>(id);
  c.x = 2 * c.x + ((c.y + c.z + i32(parity)) & 1);
  if (any(c >= n)) { return; }
  if (solidAt(c)) {
    textureStore(pressure, c, vec4<f32>(0.0));
    return;
  }
  let t = stencil(c, n);
  textureStore(pressure, c, vec4<f32>((t.x - textureLoad(rhs, c, 0).x) / max(t.y, 1.0), 0.0, 0.0, 0.0));
}`;

// The mean residual of eight fine cells becomes the coarse right-hand side; a cell twice the size needs four times the source.
export const RESTRICT = HEADER + /* wgsl */ `
override scale: f32;
@group(0) @binding(1) var pressure: texture_storage_3d<r32float, read>;
@group(0) @binding(2) var rhs: texture_3d<f32>;
@group(0) @binding(3) var coarsePressure: texture_storage_3d<r32float, write>;
@group(0) @binding(4) var coarseRhs: texture_storage_3d<r32float, write>;
${STENCIL}
fn residual(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (solidAt(c)) { return 0.0; }
  let t = stencil(c, n);
  return textureLoad(rhs, c, 0).x - (t.x - t.y * textureLoad(pressure, c).x);
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(textureDimensions(pressure));
  let coarse = vec3<i32>(id);
  if (any(coarse >= n / 2)) { return; }
  var sum = 0.0;
  for (var k = 0; k < 8; k++) {
    sum += residual(2 * coarse + vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1), n);
  }
  textureStore(coarseRhs, coarse, vec4<f32>(0.5 * sum, 0.0, 0.0, 0.0));
  textureStore(coarsePressure, coarse, vec4<f32>(0.0));
}`;

// Adds the coarse correction to every fine cell, interpolated between the eight coarse cells around it.
export const PROLONGATE = /* wgsl */ `
const X = vec3<i32>(1, 0, 0);
const Y = vec3<i32>(0, 1, 0);
const Z = vec3<i32>(0, 0, 1);
@group(0) @binding(0) var coarse: texture_3d<f32>;
@group(0) @binding(1) var pressure: texture_storage_3d<r32float, read_write>;
// Above the lid the correction is zero, like the pressure; at the walls it continues flat.
fn tap(c: vec3<i32>, n: vec3<i32>) -> f32 {
  if (c.y >= n.y) { return 0.0; }
  return textureLoad(coarse, clamp(c, vec3<i32>(0), n - vec3<i32>(1)), 0).x;
}
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n = vec3<i32>(textureDimensions(pressure));
  let c = vec3<i32>(id);
  if (any(c >= n)) { return; }
  let nc = vec3<i32>(textureDimensions(coarse));
  let q = (vec3<f32>(c) + vec3<f32>(0.5)) * 0.5 - vec3<f32>(0.5);
  let b = vec3<i32>(floor(q));
  let f = q - floor(q);
  let near = mix(mix(tap(b, nc), tap(b + X, nc), f.x), mix(tap(b + Y, nc), tap(b + X + Y, nc), f.x), f.y);
  let far = mix(mix(tap(b + Z, nc), tap(b + X + Z, nc), f.x), mix(tap(b + Y + Z, nc), tap(b + X + Y + Z, nc), f.x), f.y);
  let value = mix(near, far, f.z);
  textureStore(pressure, c, textureLoad(pressure, c) + vec4<f32>(value, 0.0, 0.0, 0.0));
}`;
