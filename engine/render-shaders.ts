import { GROUND_COLOR, KEY_COLOR, KEY_STRENGTH, NOISE_PERIOD, SKY_COLOR } from './scene';
import { BLACKBODY, PARAMS_STRUCT, SOLIDS } from './simulation-shaders';

const vec3 = (v: number[]) => `vec3<f32>(${v.map((x) => x.toFixed(3)).join(', ')})`;

const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(corners[i], 0.0, 1.0);
}`;

// The ACES fit, then the sRGB transfer curve, so a frame saved as sRGB decodes to what the viewport shows.
export const TONEMAP = /* wgsl */ `
fn tonemap(c: vec3<f32>) -> vec3<f32> {
  let m = clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
  return select(1.055 * pow(m, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * m, m <= vec3<f32>(0.0031308));
}`;

export const RENDER = BLACKBODY + TONEMAP + /* wgsl */ `
const KEY_COLOR = ${vec3(KEY_COLOR)};
const KEY_STRENGTH = ${KEY_STRENGTH.toFixed(3)};
const SKY_COLOR = ${vec3(SKY_COLOR)};
const GROUND_COLOR = ${vec3(GROUND_COLOR)};
const NOISE_PERIOD = ${NOISE_PERIOD.toFixed(1)};
const BLUE_FLAME = vec3<f32>(1.2, 2.8, 10.0);
const ALBEDO = vec3<f32>(0.03, 0.025, 0.02);
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
@group(0) @binding(10) var vel: texture_3d<f32>;
` + SOLIDS + FULLSCREEN_VS + /* wgsl */ `

// Interleaved gradient noise, slid across the screen each frame so the TAA averages the step offsets into grain
// rather than a standing dither screen.
fn jitter(p: vec2<f32>, frame: f32) -> f32 {
  let q = p + 5.588238 * (frame % 64.0);
  return fract(52.9829189 * fract(0.06711056 * q.x + 0.00583715 * q.y));
}

// Henyey-Greenstein with the 4 pi folded in, so g = 0 gives 1; smoke scatters mostly forward with a little back.
fn hg(mu: f32, g: f32) -> f32 { return (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * mu, 1.5); }
fn dualLobe(mu: f32, k: f32) -> f32 { return 0.8 * hg(mu, 0.45 * k) + 0.2 * hg(mu, -0.25 * k); }

// The solver keeps velocity on the lower cell faces in cells per second; one cell is 1 / P.size.x world units.
fn velocityAt(q: vec3<f32>) -> vec3<f32> {
  let h = 0.5 / P.size;
  return vec3<f32>(
    textureSampleLevel(vel, lin, q + vec3<f32>(h.x, 0.0, 0.0), 0.0).x,
    textureSampleLevel(vel, lin, q + vec3<f32>(0.0, h.y, 0.0), 0.0).y,
    textureSampleLevel(vel, lin, q + vec3<f32>(0.0, 0.0, h.z), 0.0).z) / P.size.x;
}

// The light groups add up to the beauty: fire is everything that glows, firelight the flame's light on smoke and
// solids, smoke the key light and sky on them. depth and velocity are visibility-weighted means along the ray.
struct March {
  fire: vec3<f32>,
  firelight: vec3<f32>,
  smoke: vec3<f32>,
  alpha: f32,
  depth: f32,
  seen: f32,
  heat: f32,
  dir: vec3<f32>,
  velocity: vec3<f32>,
}

fn march(frag: vec2<f32>) -> March {
  var m: March;
  let local = (frag - V.viewport.xy) / V.viewport.zw;
  let ndc = vec2<f32>(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0);
  let dir = normalize(V.forward + ndc.x * V.aspect * V.tanHalf * V.right + ndc.y * V.tanHalf * V.up);
  m.dir = dir;
  let inv = 1.0 / dir;
  let t0 = (V.boxMin - V.camPos) * inv;
  let t1 = (V.boxMax - V.camPos) * inv;
  let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  if (tFar <= max(tNear, 0.0)) { return m; }
  let size = V.boxMax - V.boxMin;
  var t = max(tNear, 0.0) + V.stepSize * jitter(frag, V.frame);
  var transmittance = 1.0;
  var depthSum = 0.0;
  var weightSum = 0.0;
  var hottest = 0.0;
  var velocitySum = vec3<f32>(0.0);
  // lightDir points at the light, so mu = 1 looks straight into it: backlit smoke gets the bright forward lobe.
  let mu = dot(dir, P.lightDir);
  let phase = vec3<f32>(dualLobe(mu, 1.0), dualLobe(mu, 0.5), dualLobe(mu, 0.25));
  for (var i = 0; i < i32(V.maxSteps); i++) {
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
        m.smoke += transmittance * ALBEDO * lit;
        m.firelight += transmittance * ALBEDO * flameLight;
        m.fire += transmittance * ember;
        depthSum += transmittance * hit;
        weightSum += transmittance;
        hottest = max(hottest, surface.y);
        transmittance = 0.0;
        break;
      }
    }
    // Blocks holding gas thinner than 0.02 count as empty and are crossed 4.5 cells at a time, inside the
    // occupancy's dilated 8-cell margin and short of any solid the SDF reports.
    let blocks = vec3<i32>(textureDimensions(glow));
    if (textureLoad(glow, min(vec3<i32>(q * vec3<f32>(blocks)), blocks - 1), 0).a < 0.02 && sd > 4.5) {
      t += 4.5 / P.size.x;
      continue;
    }
    let lightHere = textureSampleLevel(glow, lin, q, 0.0);
    var s = textureSampleLevel(sca, lin, q, 0.0);
    var breakup = 1.0;
    if (V.detail > 0.0 && s.y > 0.2) {
      // Sub-cell tongues: noise looked up where the gas came from, so it travels with the flow. The two fields are
      // blended by value, never by coordinate: a coordinate blend alternates between a frozen pattern and one moving
      // at twice the gas speed. The lookup is stretched upwards so the tongues streak with the rise instead of
      // mottling, and the hot core keeps its own shape.
      let stretch = vec3<f32>(1.0, 0.5, 1.0) / NOISE_PERIOD;
      let nA = textureSampleLevel(noise, rep, (g + textureSampleLevel(flowA, lin, q, 0.0).xyz) * stretch, 0.0);
      let nB = textureSampleLevel(noise, rep, (g + textureSampleLevel(flowB, lin, q, 0.0).xyz) * stretch, 0.0);
      let n = mix(nB, nA, V.flowWeight) - vec4<f32>(0.5);
      let gate = smoothstep(0.15, 0.6, s.y) * (1.0 - 0.5 * smoothstep(1.2, 2.5, s.y));
      s = textureSampleLevel(sca, lin, q + n.xyz * 2.0 * V.detail * gate / P.size, 0.0);
      breakup = 1.0 + 1.2 * n.w * gate;
    }
    // Where fuel is burning right now the flame is hottest and whitest; the burnt gas around it is dimmer and more
    // orange. Fuel burns lean and blue where it enters, along the floor of the emitter or the skin of a solid, and
    // that layer glows blue through almost no soot; above it the soot takes over and the flame goes yellow.
    let reaction = smoothstep(1.0, 10.0, s.w);
    let fresh = select(1.0 - smoothstep(0.0, 0.6 * P.radius, g.y - max(P.emitter.y - P.radius, 0.0)), 1.0 - smoothstep(1.0, 3.0, sd), P.mode == 3.0);
    let lean = min(2.0 * V.blue, 1.0) * reaction * fresh;
    let light = textureSampleLevel(shade, lin, q, 0.0);
    let flameLight = lightHere.rgb * V.fireLight * V.fire;
    let sigmaSmoke = s.x * V.density;
    let sigmaFire = V.fireDensity * flame(s.y) * breakup * (1.0 - 0.9 * lean);
    let sigma = sigmaSmoke + sigmaFire;
    let opacity = 1.0 - exp(-sigma * V.stepSize);
    let reach = transmittance * opacity / max(sigma, 1e-4);
    m.fire += reach * sigmaFire * blackbody(s.y + 0.5 * reaction) * (0.7 + 0.5 * reaction) * V.fire;
    m.fire += transmittance * V.stepSize * lean * BLUE_FLAME * V.fire;
    // Wrenninge octaves: each softer and weaker, standing in for light scattered more than once; the weights sum to
    // one so an unshadowed cell gets exactly the key.
    let key = (4.0 * phase.x * light.x + 2.0 * phase.y * sqrt(light.x) + phase.z * sqrt(sqrt(light.x))) / 7.0;
    let ambient = mix(GROUND_COLOR, SKY_COLOR, light.y);
    let scattered = reach * sigmaSmoke * V.smokeTint;
    m.smoke += scattered * (KEY_STRENGTH * KEY_COLOR * key + ambient);
    m.firelight += scattered * flameLight;
    depthSum += transmittance * opacity * t;
    weightSum += transmittance * opacity;
    // Only the export asks for motion; the live view skips the velocity taps.
    if (V.frameTime > 0.0) { velocitySum += transmittance * opacity * velocityAt(q); }
    hottest = max(hottest, s.y);
    transmittance *= 1.0 - opacity;
    t += V.stepSize;
  }
  m.fire *= V.exposure;
  m.firelight *= V.exposure;
  m.smoke *= V.exposure;
  m.alpha = 1.0 - transmittance;
  m.depth = depthSum / max(weightSum, 1e-6);
  m.velocity = velocitySum / max(weightSum, 1e-6);
  m.seen = weightSum;
  m.heat = hottest;
  return m;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let m = march(frag.xy);
  return vec4<f32>(tonemap(m.fire + m.firelight + m.smoke), m.alpha);
}

fn toPixel(p: vec3<f32>) -> vec2<f32> {
  let r = p - V.camPos;
  let z = dot(r, V.forward);
  return vec2<f32>(0.5 + 0.5 * dot(r, V.right) / (z * V.tanHalf * V.aspect), 0.5 - 0.5 * dot(r, V.up) / (z * V.tanHalf)) * V.viewport.zw;
}

// The light groups as linear premultiplied colour, plus temperature in firelight.a and a data target holding the
// forward motion in pixels per frame (+y down), the planar camera depth and the coverage, the first three weighted
// by the coverage so samples that miss leave them untouched. Every target is scaled by the sample weight and
// summed by additive blending, so N samples give an exact mean.
struct ExportOut {
  @location(0) fire: vec4<f32>,
  @location(1) firelight: vec4<f32>,
  @location(2) smoke: vec4<f32>,
  @location(3) data: vec4<f32>,
}

@fragment
fn fsExport(@builtin(position) frag: vec4<f32>) -> ExportOut {
  let m = march(frag.xy);
  var data = vec4<f32>(0.0);
  if (m.seen > 1e-3) {
    let p = V.camPos + m.dir * m.depth;
    data = vec4<f32>(toPixel(p + m.velocity * V.frameTime) - toPixel(p), m.depth * dot(m.dir, V.forward), 1.0);
  }
  let w = V.sampleWeight;
  return ExportOut(vec4<f32>(m.fire, m.alpha) * w, vec4<f32>(m.firelight, m.heat) * w, vec4<f32>(m.smoke, 0.0) * w, data * w);
}`;

// The accumulated light groups through the same tonemap as the live view: what a PNG or MP4 frame holds.
export const DISPLAY = TONEMAP + /* wgsl */ `
@group(0) @binding(0) var fire: texture_2d<f32>;
@group(0) @binding(1) var firelight: texture_2d<f32>;
@group(0) @binding(2) var smoke: texture_2d<f32>;
` + FULLSCREEN_VS + /* wgsl */ `
@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(frag.xy);
  let f = textureLoad(fire, p, 0);
  return vec4<f32>(tonemap(f.rgb + textureLoad(firelight, p, 0).rgb + textureLoad(smoke, p, 0).rgb), f.a);
}`;

// Temporal blend of the jittered raymarch with the last frame, the history clamped to the current neighbourhood so it cannot ghost.
export const RESOLVE = /* wgsl */ `
struct Taa { blend: f32 }
@group(0) @binding(0) var<uniform> T: Taa;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
` + FULLSCREEN_VS + /* wgsl */ `
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
