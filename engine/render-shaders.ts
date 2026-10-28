import { GROUND_COLOR, KEY_COLOR, KEY_STRENGTH, NOISE_PERIOD, SKY_COLOR } from './scene';
import { BLACKBODY, PARAMS_STRUCT, SOLIDS } from './simulation-shaders';

const vec3 = (v: number[]) => `vec3<f32>(${v.map((x) => x.toFixed(3)).join(', ')})`;

export const RENDER = BLACKBODY + /* wgsl */ `
const KEY_COLOR = ${vec3(KEY_COLOR)};
const KEY_STRENGTH = ${KEY_STRENGTH.toFixed(3)};
const SKY_COLOR = ${vec3(SKY_COLOR)};
const GROUND_COLOR = ${vec3(GROUND_COLOR)};
const NOISE_PERIOD = ${NOISE_PERIOD.toFixed(1)};
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
  fireLight: f32,
  frame: f32,
  fireDensity: f32,
  detail: f32,
  flowWeight: f32,
  pad3: f32,
  pad4: f32,
}
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var sca: texture_3d<f32>;
@group(0) @binding(2) var shade: texture_3d<f32>;
@group(0) @binding(3) var lin: sampler;
` + PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(4) var<uniform> P: Params;
@group(0) @binding(5) var glow: texture_3d<f32>;
@group(0) @binding(6) var noise: texture_3d<f32>;
@group(0) @binding(7) var rep: sampler;
@group(0) @binding(8) var flowA: texture_3d<f32>;
@group(0) @binding(9) var flowB: texture_3d<f32>;
` + SOLIDS + /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(corners[i], 0.0, 1.0);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
  let m = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
  return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn heatRamp(t: f32) -> vec3<f32> {
  let x = clamp(t / 3.0, 0.0, 1.0) * 3.0;
  return clamp(vec3<f32>(x, x - 1.0, x - 2.0), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Interleaved gradient noise, shifted by the golden ratio each frame so the TAA averages the step offsets away.
fn jitter(p: vec2<f32>, frame: f32) -> f32 {
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y) + frame * 0.618034);
}

// Henyey-Greenstein with the 4 pi folded in, so g = 0 gives 1; smoke scatters mostly forward with a little back.
fn hg(mu: f32, g: f32) -> f32 { return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5); }
fn dualLobe(mu: f32, k: f32) -> f32 { return 0.8 * hg(mu, 0.55 * k) + 0.2 * hg(mu, -0.25 * k); }

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
  var t = max(tNear, 0.0) + V.stepSize * jitter(frag.xy, V.frame);
  var transmittance = 1.0;
  var fire = vec3<f32>(0.0);
  var smoke = vec3<f32>(0.0);
  var solid = vec3<f32>(0.0);
  var depthSum = 0.0;
  var weightSum = 0.0;
  var hottest = 0.0;
  // lightDir points at the light, so mu = 1 looks straight into it: backlit smoke gets the bright forward lobe.
  let mu = dot(dir, P.lightDir);
  let phase = vec3<f32>(dualLobe(mu, 1.0), dualLobe(mu, 0.5), dualLobe(mu, 0.25));
  for (var i = 0; i < 160; i++) {
    if (t > tFar || transmittance < 0.01) { break; }
    let q = (V.camPos + dir * t - V.boxMin) / size;
    let g = q * P.size - vec3<f32>(0.5);
    var sd = 1e5;
    if (P.mode == 3.0) {
      sd = solidDistance(g);
      if (sd < 0.0) {
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
        let shadeHere = textureSampleLevel(shade, lin, qh, 0.0);
        let lit = KEY_STRENGTH * KEY_COLOR * max(dot(n, P.lightDir), 0.0) * shadeHere.x + mix(GROUND_COLOR, SKY_COLOR, shadeHere.y);
        let surface = textureSampleLevel(sca, lin, qh + n * 2.0 / P.size, 0.0);
        let flameLight = textureSampleLevel(glow, lin, qh, 0.0).rgb * V.fireLight * V.fire;
        let ember = blackbody(surface.y) * smoothstep(1.0, 6.0, surface.w) * 0.25 * V.fire;
        solid += transmittance * (vec3<f32>(0.03, 0.025, 0.02) * (lit + flameLight) + ember);
        depthSum += transmittance * hit;
        weightSum += transmittance;
        hottest = max(hottest, surface.y);
        transmittance = 0.0;
        break;
      }
    }
    let lightHere = textureSampleLevel(glow, lin, q, 0.0);
    // Three steps are 4.5 cells, inside the occupancy's 8-cell margin and short of any solid the SDF reports.
    if (lightHere.a < 0.02 && sd > 4.5) {
      t += 3.0 * V.stepSize;
      continue;
    }
    var s = textureSampleLevel(sca, lin, q, 0.0);
    var breakup = 1.0;
    if (V.detail > 0.0 && s.y > 0.2) {
      // Sub-cell tongues: noise looked up where the gas came from, so it travels with the flow. The two fields are
      // blended by value, never by coordinate: a coordinate blend alternates between a frozen pattern and one moving
      // at twice the gas speed.
      let nA = textureSampleLevel(noise, rep, (g + textureSampleLevel(flowA, lin, q, 0.0).xyz) / NOISE_PERIOD, 0.0);
      let nB = textureSampleLevel(noise, rep, (g + textureSampleLevel(flowB, lin, q, 0.0).xyz) / NOISE_PERIOD, 0.0);
      let n = mix(nB, nA, V.flowWeight) - vec4<f32>(0.5);
      let gate = clamp(s.y, 0.0, 1.0);
      s = textureSampleLevel(sca, lin, q + n.xyz * 2.0 * V.detail * gate / P.size, 0.0);
      breakup = 1.0 + 1.2 * n.w * gate;
    }
    let light = textureSampleLevel(shade, lin, q, 0.0);
    let flameLight = lightHere.rgb * V.fireLight * V.fire;
    let sigmaSmoke = s.x * V.density;
    let sigmaFire = V.fireDensity * flame(s.y) * breakup;
    let sigma = sigmaSmoke + sigmaFire;
    let opacity = 1.0 - exp(-sigma * V.stepSize);
    let reach = transmittance * opacity / max(sigma, 1e-4);
    fire += reach * sigmaFire * blackbody(s.y) * V.fire;
    // Wrenninge octaves: each softer and weaker, standing in for light scattered more than once.
    let key = phase.x * light.x + 0.5 * phase.y * sqrt(light.x) + 0.25 * phase.z * sqrt(sqrt(light.x));
    let ambient = mix(GROUND_COLOR, SKY_COLOR, light.y);
    smoke += reach * sigmaSmoke * V.smokeTint * (KEY_STRENGTH * KEY_COLOR * key + ambient + flameLight);
    depthSum += transmittance * opacity * t;
    weightSum += transmittance * opacity;
    hottest = max(hottest, s.y);
    transmittance *= 1.0 - opacity;
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

// Temporal blend of the jittered raymarch with the last frame, the history clamped to the current neighbourhood so it cannot ghost.
export const RESOLVE = /* wgsl */ `
struct Taa { blend: f32 }
@group(0) @binding(0) var<uniform> T: Taa;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(corners[i], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(frag.xy);
  let last = vec2<i32>(textureDimensions(current)) - vec2<i32>(1);
  let now = textureLoad(current, p, 0);
  var lo = now;
  var hi = now;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let s = textureLoad(current, clamp(p + vec2<i32>(x, y), vec2<i32>(0), last), 0);
      lo = min(lo, s);
      hi = max(hi, s);
    }
  }
  let old = clamp(textureLoad(history, p, 0), lo, hi);
  return mix(old, now, T.blend);
}`;

// Bilinear upsample of the resolved frame onto the canvas.
export const COMPOSITE = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  let c = corners[i];
  return Out(vec4<f32>(c, 0.0, 1.0), vec2<f32>(0.5 * (c.x + 1.0), 0.5 * (1.0 - c.y)));
}
@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  return textureSampleLevel(src, lin, in.uv, 0.0);
}`;
