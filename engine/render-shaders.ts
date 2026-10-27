import { PARAMS_STRUCT, SOLIDS } from './simulation-shaders';

export const RENDER = /* wgsl */ `
struct View {
  camPos: vec3<f32>,
  tanHalf: f32,
  forward: vec3<f32>,
  aspect: f32,
  right: vec3<f32>,
  time: f32,
  up: vec3<f32>,
  stepSize: f32,
  boxMin: vec3<f32>,
  exposure: f32,
  boxMax: vec3<f32>,
  density: f32,
  smokeTint: vec3<f32>,
  fire: f32,
  viewport: vec4<f32>,
  layer: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
}
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var sca: texture_3d<f32>;
@group(0) @binding(2) var shade: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
` + PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(4) var<uniform> P: Params;
` + SOLIDS + /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(corners[i], 0.0, 1.0);
}

// Flame colour from temperature: an approximate blackbody curve, brighter with the fourth power of temperature.
fn blackbody(t: f32) -> vec3<f32> {
  let k = clamp(8.0 + 7.0 * t, 8.0, 40.0);
  let g = clamp((99.4708 * log(k) - 161.1196) / 255.0, 0.0, 1.0);
  let b = select(clamp((138.5177 * log(max(k - 10.0, 1.0)) - 305.0448) / 255.0, 0.0, 1.0), 0.0, k <= 19.0);
  return pow(vec3<f32>(1.0, g, b), vec3<f32>(2.2)) * pow(k / 15.0, 4.0);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
  let m = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
  return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn heatRamp(t: f32) -> vec3<f32> {
  let x = clamp(t / 3.0, 0.0, 1.0) * 3.0;
  return clamp(vec3<f32>(x, x - 1.0, x - 2.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash2(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453); }

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let local = (frag.xy - V.viewport.xy) / V.viewport.zw;
  let ndc = vec2<f32>(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0);
  let dir = normalize(V.forward + ndc.x * V.aspect * V.tanHalf * V.right + ndc.y * V.tanHalf * V.up);
  let inv = 1.0 / dir;
  let t0 = (V.boxMin - V.camPos) * inv;
  let t1 = (V.boxMax - V.camPos) * inv;
  let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tFar <= max(tNear, 0.0)) { return select(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0), V.layer > 0.5); }
  let size = V.boxMax - V.boxMin;
  var t = max(tNear, 0.0) + V.stepSize * hash2(frag.xy);
  var transmittance = 1.0;
  var fire = vec3<f32>(0.0);
  var smoke = vec3<f32>(0.0);
  var solid = vec3<f32>(0.0);
  var depthSum = 0.0;
  var weightSum = 0.0;
  var hottest = 0.0;
  for (var i = 0; i < 160; i++) {
    if (t > tFar || transmittance < 0.01) { break; }
    let q = (V.camPos + dir * t - V.boxMin) / size;
    if (P.mode == 3.0) {
      let g = q * P.size - vec3<f32>(0.5);
      if (solidDistance(g) < 0.0) {
        // Bisect back to the real surface; the coarse, jittered march would speckle it.
        var outside = t - V.stepSize;
        var hit = t;
        for (var b = 0; b < 5; b++) {
          let mid = 0.5 * (outside + hit);
          if (solidDistance((V.camPos + dir * mid - V.boxMin) / size * P.size - vec3<f32>(0.5)) < 0.0) { hit = mid; } else { outside = mid; }
        }
        let qh = (V.camPos + dir * hit - V.boxMin) / size;
        let gh = qh * P.size - vec3<f32>(0.5);
        let e = 0.75;
        let n = normalize(vec3<f32>(
          solidDistance(gh + vec3<f32>(e, 0.0, 0.0)) - solidDistance(gh - vec3<f32>(e, 0.0, 0.0)),
          solidDistance(gh + vec3<f32>(0.0, e, 0.0)) - solidDistance(gh - vec3<f32>(0.0, e, 0.0)),
          solidDistance(gh + vec3<f32>(0.0, 0.0, e)) - solidDistance(gh - vec3<f32>(0.0, 0.0, e))));
        let lit = max(dot(n, P.lightDir), 0.0) * textureSampleLevel(shade, lin, qh, 0.0).x + 0.18;
        let heat = textureSampleLevel(sca, lin, qh + n * 3.0 / P.size, 0.0).y;
        solid += transmittance * (vec3<f32>(0.075, 0.06, 0.05) * lit + blackbody(heat * 0.9) * V.fire * 0.3);
        depthSum += transmittance * hit;
        weightSum += transmittance;
        hottest = max(hottest, heat);
        transmittance = 0.0;
        break;
      }
    }
    let s = textureSampleLevel(sca, lin, q, 0.0);
    let absorb = exp(-s.x * V.density * V.stepSize);
    let light = textureSampleLevel(shade, lin, q, 0.0).x;
    fire += transmittance * blackbody(s.y) * V.fire * smoothstep(0.2, 1.0, s.y) * V.stepSize;
    smoke += transmittance * (V.smokeTint * (0.85 * light + 0.12) + blackbody(s.y * 0.7) * 0.16 * V.fire) * (1.0 - absorb);
    depthSum += transmittance * (1.0 - absorb) * t;
    weightSum += transmittance * (1.0 - absorb);
    hottest = max(hottest, s.y);
    transmittance *= absorb;
    t += V.stepSize;
  }
  let alpha = 1.0 - transmittance;
  // Layers for compositing: 1 fire light, 2 smoke and solids, 3 alpha, 4 depth (near is white), 5 temperature.
  if (V.layer == 1.0) { return vec4<f32>(tonemap(fire * V.exposure), 1.0); }
  if (V.layer == 2.0) { return vec4<f32>(tonemap((smoke + solid) * V.exposure), 1.0); }
  if (V.layer == 3.0) { return vec4<f32>(vec3<f32>(alpha), 1.0); }
  if (V.layer == 4.0) {
    let near = select(0.0, 1.0 - clamp(depthSum / max(weightSum, 1e-4) - 5.7, 0.0, 1.0), weightSum > 0.02);
    return vec4<f32>(vec3<f32>(near), 1.0);
  }
  if (V.layer == 5.0) { return vec4<f32>(heatRamp(hottest), 1.0); }
  return vec4<f32>(tonemap((fire + smoke + solid) * V.exposure), alpha);
}`;
