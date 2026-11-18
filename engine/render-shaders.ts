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
// Fitting each channel on its own bleaches a hot flame to white; fitting the brightest channel and scaling the
// colour down to it keeps the hue. Mostly the first with some of the second: the core still brightens towards
// white where it is hottest, but keeps its yellow and the detail in it on the way there.
export const TONEMAP = /* wgsl */ `
fn encode(m: vec3<f32>) -> vec3<f32> { return select(1.055 * pow(m, vec3<f32>(1.0 / 2.4)) - 0.055, 12.92 * m, m <= vec3<f32>(0.0031308)); }
fn decode(c: vec3<f32>) -> vec3<f32> { return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045)); }
fn aces(c: vec3<f32>) -> vec3<f32> { return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0)); }
fn tonemap(c: vec3<f32>) -> vec3<f32> {
  let peak = max(max(c.r, c.g), max(c.b, 1e-4));
  return encode(mix(aces(c), c / peak * aces(vec3<f32>(peak)).x, 0.3));
}
// A premultiplied display frame over the footage plate, less the share of it the plate's foreground covers. The two
// meet in linear light, where a flame adds to what is behind it and smoke takes away its share: the frame is
// decoded, laid over the plate by its alpha and encoded again. With no plate loaded the plate is one clear texel
// and the frame comes back as it went in.
fn over(frame: vec4<f32>, plate: vec4<f32>, cover: f32) -> vec4<f32> {
  let fire = vec4<f32>(decode(frame.rgb), frame.a) * (1.0 - cover);
  let behind = plate.a * (1.0 - fire.a);
  return vec4<f32>(encode(fire.rgb + plate.rgb * behind), fire.a + behind);
}`;

// plate: whether the frame is laid over the footage or left alone on its own alpha. tint: how strongly the live view
// marks what covers the fire.
const LAYERS = /* wgsl */ `
struct Layers { plate: f32, tint: f32 }
`;

// How much of the fire the footage's foreground covers: the plate's depth, read between its texels, against the
// nearness the fire stands at. Cover comes in over a band of nearness above the fire's, and the band follows the
// depth's own slope across `H.reach` of the frame: where the depth jumps, at the edge of a pole or a wall, the band
// is crossed within a texel or two; where it only slopes, through the blur the depth map makes of branches against
// the sky or along the ground under the fire, the edge fades over that distance instead of drawing the map's own
// blotchy outline. The band stays between SOFT and WIDEST, so a solid object a little nearer than the fire is never
// partly see-through. With no depth the map is one texel far behind everything and covers nothing. The bindings are
// the including shader's: `depth`, a nearness and its slope per texel, and `H`.
const SOFT = 0.04;
const WIDEST = 0.3;
const HOLDOUT = /* wgsl */ `
struct Hold { nearness: f32, reach: f32 }
fn depthAt(texel: vec2<i32>) -> vec2<f32> {
  return textureLoad(depth, clamp(texel, vec2<i32>(0), vec2<i32>(textureDimensions(depth)) - vec2<i32>(1)), 0).xy;
}
fn coverAt(uv: vec2<f32>) -> f32 {
  let size = vec2<f32>(textureDimensions(depth));
  let at = uv * size - 0.5;
  let corner = vec2<i32>(floor(at));
  let f = fract(at);
  let here = mix(
    mix(depthAt(corner), depthAt(corner + vec2<i32>(1, 0)), f.x),
    mix(depthAt(corner + vec2<i32>(0, 1)), depthAt(corner + vec2<i32>(1, 1)), f.x),
    f.y
  );
  let ahead = here.x - H.nearness;
  let band = clamp(here.y * H.reach * max(size.x, size.y), ${SOFT}, ${WIDEST});
  return smoothstep(0.0, max(band, 1.5 * fwidth(ahead)), ahead);
}
`;

export const RENDER = BLACKBODY + TONEMAP + /* wgsl */ `
const KEY_COLOR = ${vec3(KEY_COLOR)};
const KEY_STRENGTH = ${KEY_STRENGTH.toFixed(3)};
const SKY_COLOR = ${vec3(SKY_COLOR)};
const GROUND_COLOR = ${vec3(GROUND_COLOR)};
const NOISE_PERIOD = ${NOISE_PERIOD.toFixed(1)};
const BLUE_FLAME = vec3<f32>(0.5, 1.4, 4.5);
// Emission per unit of extinction: the flame is thin enough to see through, so its light is turned up.
const FLAME_GAIN = 2.2;
const ALBEDO = vec3<f32>(0.03, 0.025, 0.02);
// Soot absorbs about as 1 / wavelength: red, green and blue against green. Light that crosses it reddens.
const SOOT_ABSORPTION = vec3<f32>(0.9, 1.0, 1.18);
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
fn mean(v: vec3<f32>) -> f32 { return (v.r + v.g + v.b) / 3.0; }

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
  // The lens shift slides the picture across the frame without turning the view.
  let ndc = vec2<f32>(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0) - V.shift;
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
  var transmittance = vec3<f32>(1.0);
  var depthSum = 0.0;
  var weightSum = 0.0;
  var hottest = 0.0;
  var velocitySum = vec3<f32>(0.0);
  // lightDir points at the light, so mu = 1 looks straight into it: backlit smoke gets the bright forward lobe.
  let mu = dot(dir, P.lightDir);
  let phase = vec3<f32>(dualLobe(mu, 1.0), dualLobe(mu, 0.5), dualLobe(mu, 0.25));
  for (var i = 0; i < i32(V.maxSteps); i++) {
    if (t > tFar || max(transmittance.r, max(transmittance.g, transmittance.b)) < 0.01) { break; }
    var q = (V.camPos + dir * t - V.boxMin) / size;
    var sd = 1e5;
    if (P.mode == 3.0) {
      sd = solidDistance(q * P.size - vec3<f32>(0.5));
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
        depthSum += mean(transmittance) * hit;
        weightSum += mean(transmittance);
        hottest = max(hottest, surface.y);
        transmittance = vec3<f32>(0.0);
        break;
      }
    }
    // Motion blur: the gas here at the sample's moment in the shutter is the gas that is now a little along the
    // flow, so the lookup moves with the velocity. Only the export asks; live the velocity taps are skipped.
    var velocity = vec3<f32>(0.0);
    if (V.frameTime > 0.0) {
      velocity = velocityAt(q);
      q = clamp(q + velocity * V.timeOffset / size, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let g = q * P.size - vec3<f32>(0.5);
    // Blocks holding gas thinner than 0.02 count as empty and are crossed almost two blocks at a time, inside the
    // occupancy's two-block dilated margin and short of any solid the SDF reports.
    let blocks = vec3<i32>(textureDimensions(glow));
    let skip = 2.0 * P.size.x / f32(blocks.x) - 0.5;
    if (textureLoad(glow, min(vec3<i32>(q * vec3<f32>(blocks)), blocks - 1), 0).a < 0.02 && sd > skip) {
      t += skip / P.size.x;
      continue;
    }
    let lightHere = textureSampleLevel(glow, lin, q, 0.0);
    var s = textureSampleLevel(sca, lin, q, 0.0);
    var breakup = 1.0;
    var coreTexture = 1.0;
    if (V.detail > 0.0 && s.y > 0.2) {
      // Sub-cell tongues: noise looked up where the gas came from, so it travels with the flow. The two fields are
      // blended by value, never by coordinate: a coordinate blend alternates between a frozen pattern and one moving
      // at twice the gas speed. The lookup is stretched upwards so the tongues streak with the rise instead of
      // mottling, and everything is measured in base cells so the noise keeps its size on a finer grid.
      let k = cellScale();
      let gb = g / k;
      let dA = textureSampleLevel(flowA, lin, q, 0.0).xyz / k;
      let dB = textureSampleLevel(flowB, lin, q, 0.0).xyz / k;
      let stretch = vec3<f32>(1.0, 0.5, 1.0) / NOISE_PERIOD;
      let n = mix(textureSampleLevel(noise, rep, (gb + dB) * stretch, 0.0), textureSampleLevel(noise, rep, (gb + dA) * stretch, 0.0), V.flowWeight) - vec4<f32>(0.5);
      // A finer octave on the same fields tears the tongues into wisps, and a third, finer still, frays the edge.
      let f = mix(textureSampleLevel(noise, rep, (gb + dB) * stretch * 2.7 + vec3<f32>(0.37), 0.0), textureSampleLevel(noise, rep, (gb + dA) * stretch * 2.7 + vec3<f32>(0.37), 0.0), V.flowWeight) - vec4<f32>(0.5);
      let w = mix(textureSampleLevel(noise, rep, (gb + dB) * stretch * 4.2 + vec3<f32>(0.71), 0.0), textureSampleLevel(noise, rep, (gb + dA) * stretch * 4.2 + vec3<f32>(0.71), 0.0), V.flowWeight) - vec4<f32>(0.5);
      let edge = 1.0 - smoothstep(0.5, 1.2, s.y);
      let gate = smoothstep(0.15, 0.6, s.y) * (1.0 - 0.5 * smoothstep(1.2, 2.5, s.y));
      s = textureSampleLevel(sca, lin, q + (n.xyz * 1.6 + f.xyz * 0.9 + w.xyz * 0.3 * edge) * V.detail * gate / (P.size / k), 0.0);
      // Strands: the flame thins out where the noise dips and bunches where it peaks, so tongues separate with
      // gaps between them; the core keeps a floor so it never hollows out, and carries the fine noise as brightness.
      let ridge = 1.2 * n.w + 0.9 * f.w + 0.5 * w.w * edge;
      breakup = mix(1.0, 0.1 + 1.9 * smoothstep(-0.45, 0.5, ridge), gate);
      coreTexture = 1.0 + 0.4 * f.w * smoothstep(0.8, 1.8, s.y);
    }
    // Where fuel is burning right now the flame is hottest and whitest; the burnt gas around it is dimmer and more
    // orange. Fuel burns lean and blue where it enters, along the floor of the emitter or the skin of a solid, and
    // that layer glows blue through almost no soot; above it the soot takes over and the flame goes yellow.
    let reaction = smoothstep(1.0, 10.0, s.w);
    let fresh = select(1.0 - smoothstep(0.0, 0.4 * P.radius, g.y - max(P.emitter.y - P.radius, 0.0)), 1.0 - smoothstep(1.0, 2.5, sd / cellScale()), P.mode == 3.0);
    let lean = min(2.0 * V.blue, 1.0) * reaction * fresh;
    let light = textureSampleLevel(shade, lin, q, 0.0);
    let flameLight = lightHere.rgb * V.fireLight * V.fire;
    // The smoke's tint is its single-scattering albedo: that share of what it stops is scattered, evenly over the
    // spectrum. The rest is absorbed, more of it in the blue, and what absorbs also emits: soot that is still hot
    // glows with the flame's own colour for its temperature, and goes black only once it has cooled.
    let smoke = s.x * V.density;
    let absorbing = (vec3<f32>(1.0) - V.smokeTint) * SOOT_ABSORPTION;
    let extinction = V.smokeTint + absorbing;
    let sigmaFire = V.fireDensity * flame(s.y) * breakup * (1.0 - 0.95 * lean);
    let sigma = smoke * extinction + vec3<f32>(sigmaFire);
    let opacity = vec3<f32>(1.0) - exp(-sigma * V.stepSize);
    let reach = transmittance * opacity / max(sigma, vec3<f32>(1e-4));
    let glowing = vec3<f32>(sigmaFire) + smoke * absorbing * flame(s.y);
    m.fire += reach * glowing * blackbody(s.y + 0.5 * reaction) * (0.7 + 0.5 * reaction) * coreTexture * FLAME_GAIN * V.fire;
    m.fire += transmittance * V.stepSize * lean * BLUE_FLAME * V.fire;
    // Wrenninge octaves: each softer and weaker, standing in for light scattered more than once. Every further
    // bounce costs the albedo again, so white smoke fills its own shadows and soot, which keeps a fifth of the light
    // per bounce, stays black in them. The key reddens on its way in as the view does on its way out.
    let sun = pow(vec3<f32>(light.x), extinction);
    let key = (4.0 * phase.x * sun + V.smokeTint * (2.0 * phase.y * sqrt(sun) + V.smokeTint * phase.z * sqrt(sqrt(sun)))) / 7.0;
    let ambient = mix(GROUND_COLOR, SKY_COLOR, light.y);
    let scattered = reach * smoke * V.smokeTint;
    m.smoke += scattered * (KEY_STRENGTH * KEY_COLOR * key + ambient);
    m.firelight += scattered * flameLight;
    let seen = mean(transmittance * opacity);
    depthSum += seen * t;
    weightSum += seen;
    velocitySum += seen * velocity;
    hottest = max(hottest, s.y);
    transmittance *= vec3<f32>(1.0) - opacity;
    t += V.stepSize;
  }
  m.fire *= V.exposure;
  m.firelight *= V.exposure;
  m.smoke *= V.exposure;
  m.alpha = 1.0 - mean(transmittance);
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
  let ndc = vec2<f32>(dot(r, V.right) / (z * V.tanHalf * V.aspect), dot(r, V.up) / (z * V.tanHalf)) + V.shift;
  return vec2<f32>(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * V.viewport.zw;
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

// The accumulated light groups through the same tonemap as the live view: what a PNG or MP4 frame holds. The
// footage's foreground covers its share of the frame, and the frame goes over the plate when the export composites it.
export const DISPLAY = TONEMAP + LAYERS + HOLDOUT + /* wgsl */ `
@group(0) @binding(0) var fire: texture_2d<f32>;
@group(0) @binding(1) var firelight: texture_2d<f32>;
@group(0) @binding(2) var smoke: texture_2d<f32>;
@group(0) @binding(3) var plate: texture_2d<f32>;
@group(0) @binding(4) var lin: sampler;
@group(0) @binding(5) var depth: texture_2d<f32>;
@group(0) @binding(6) var<uniform> H: Hold;
@group(0) @binding(7) var<uniform> L: Layers;
` + FULLSCREEN_VS + /* wgsl */ `
@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(frag.xy);
  let f = textureLoad(fire, p, 0);
  let frame = vec4<f32>(tonemap(f.rgb + textureLoad(firelight, p, 0).rgb + textureLoad(smoke, p, 0).rgb), f.a);
  let uv = frag.xy / vec2<f32>(textureDimensions(fire));
  let behind = textureSampleLevel(plate, lin, uv, 0.0);
  return over(frame, vec4<f32>(behind.rgb, behind.a * L.plate), coverAt(uv));
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

const UV_VS = /* wgsl */ `
struct Out { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
  var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  let c = corners[i];
  return Out(vec4<f32>(c, 0.0, 1.0), vec2<f32>(0.5 * (c.x + 1.0), 0.5 * (1.0 - c.y)));
}`;

// Bilinear resample of a frame into another size.
export const COMPOSITE = UV_VS + /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  return textureSampleLevel(src, lin, in.uv, 0.0);
}`;

// The resolved live frame upsampled onto the canvas, over the footage plate and behind its foreground, which the
// view can mark in a colour no fire has.
export const PRESENT = TONEMAP + LAYERS + HOLDOUT + UV_VS + /* wgsl */ `
const COVER_TINT = vec3<f32>(0.0, 0.62, 1.0);
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var plate: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_2d<f32>;
@group(0) @binding(4) var<uniform> H: Hold;
@group(0) @binding(5) var<uniform> L: Layers;
@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
  let behind = textureSampleLevel(plate, lin, in.uv, 0.0);
  let cover = coverAt(in.uv);
  let shown = over(textureSampleLevel(src, lin, in.uv, 0.0), behind, cover);
  return vec4<f32>(mix(shown.rgb, COVER_TINT, 0.5 * cover * L.tint), shown.a);
}`;
