export const VOLUME_NOISE_SIZE = 48;

export function makeVolumeNoise() {
  const data = new Uint8Array(VOLUME_NOISE_SIZE ** 3);
  let state = 0x431db7c5;
  for (let i = 0; i < data.length; i++) {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    data[i] = state >>> 24;
  }
  return data;
}

export const volumeShader = /* glsl */ `
const vec3 volumeKey = vec3(-0.622, 0.703, -0.350);

float volumeNoise(vec3 p) {
  // Smooth the interpolation within each lattice cell in one filtered lookup.
  vec3 cell = floor(p);
  vec3 f = p - cell;
  return textureLod(uNoise, (cell + f * f * (3.0 - 2.0 * f) + 0.5) / ${VOLUME_NOISE_SIZE}.0, 0.0).r;
}

float smokePuff(vec3 p, vec3 center, vec3 radii) {
  vec3 q = (p - center) / radii;
  return 1.0 - dot(q, q);
}

struct PyroState {
  float time;
  float burning;
  float blast;
  float expansion;
  float cooling;
};

vec2 pyroFields(vec3 p, PyroState state, bool fineDetail) {
  float h = p.y;
  float rise = state.time;
  float floorFade = smoothstep(0.007, 0.095, h);
  vec2 fields = vec2(0.0);

  if (state.blast < 0.999) {
    // Advection and counter-rotating distortion keep the small-scale structure moving
    // through the large volume instead of moving the whole silhouette as a solid.
    vec3 flow = p * vec3(2.48, 2.80, 2.64) + vec3(8.1, 4.5 - rise * 1.59, 20.3);
    float roll = volumeNoise(flow);
    vec3 twist = vec3(
      0.62 * sin(4.3 * p.z + h * 2.6 - rise * 0.97),
      1.61 * (roll - 0.5),
      0.69 * cos(3.9 * p.x - h * 2.8 + rise * 1.03)
    );
    float eddy = volumeNoise(2.38 * (flow + twist) + vec3(17.3, 4.7, 2.1));
    float grain = fineDetail ? volumeNoise(6.8 * (flow + 0.46 * twist) + vec3(2.6, 22.7, 11.2)) : 0.50;
    float erosion = (roll - 0.50) * 0.72 + (eddy - 0.49) * 0.48 + (grain - 0.50) * 0.36;

    if (state.burning < 0.999) {
      // Overlapping, asymmetrically rising clouds. Their individual radius falls
      // all the way to zero before the ray bounds, including the faint top wisps.
      vec3 q = p;
      q.x += (0.16 + 0.042 * h) * sin(h * 2.9 - rise * 0.83) + 0.11 * twist.x;
      q.z += (0.09 + 0.039 * h) * cos(h * 2.5 - rise * 0.68) + 0.11 * twist.z;
      q.y += 0.13 * twist.y;
      float tube = smokePuff(q, vec3(0.00, 0.22, 0.01), vec3(0.23, 0.37, 0.20));
      float low = smokePuff(q, vec3(0.10, 0.69, 0.07), vec3(0.41, 0.51, 0.36));
      float fold = smokePuff(q, vec3(-0.14, 1.28, -0.08), vec3(0.53, 0.64, 0.48));
      float high = smokePuff(q, vec3(0.17, 1.95, 0.11), vec3(0.63, 0.67, 0.55));
      float cap = smokePuff(q, vec3(-0.31, 2.48, -0.09), vec3(0.74, 0.73, 0.63));
      float shed = smokePuff(q, vec3(-0.83, 2.91, -0.21), vec3(0.69, 0.64, 0.50));
      float envelope = max(max(max(tube, low), max(fold, high)), max(cap, shed - 0.26));
      float sideWisp = smokePuff(q, vec3(0.93, 2.05, 0.04), vec3(0.38, 0.37, 0.31));
      float rakingWisp = smokePuff(q, vec3(-1.12, 2.58, -0.29), vec3(0.39, 0.24, 0.33));
      float trailing = smoothstep(0.13, 0.68, max(sideWisp, rakingWisp) + erosion) * 0.21;
      float voids = (0.40 + 0.81 * smoothstep(0.18, 0.72, 0.43 * roll + 0.48 * eddy + 0.09 * grain));
      float density = (smoothstep(-0.035, 0.60, envelope + erosion) * voids + trailing) * floorFade;
      fields.x = density * (1.0 - state.burning);
    }
    if (state.burning > 0.001) {
      // A central updraft splits into longer outer tongues. Gaps on the reaction
      // front and less emissive folds prevent a single saturated orange column.
      float taper = smoothstep(0.29, 2.48, h);
      vec2 core = vec2(0.19 * sin(h * 2.93 - rise * 1.20) * taper, 0.17 * cos(h * 2.26 - rise * 1.39) * taper);
      float plume = 0.58 - taper * 0.31 - length((p.xz - core) * vec2(1.0, 1.24));
      vec2 branch = core + vec2(0.17 + 0.33 * taper, -0.07 - 0.13 * taper);
      float tongue = 0.28 - taper * 0.09 - length((p.xz - branch) * vec2(1.0, 1.31));
      float height = 2.72 + 0.26 * (eddy - 0.42) + 0.26 * roll;
      float flameCap = 1.0 - smoothstep(1.30, height, h);
      float curlingTongue = 0.24
        - length((p.xz - core - vec2(-0.31 - 0.25 * taper, 0.14 + 0.09 * taper)) * vec2(1.09, 1.29));
      float front = max(max(plume, tongue * 1.12), curlingTongue - 0.13 * taper)
        + 0.64 * erosion - 0.025;
      float reaction = smoothstep(-0.08, 0.34, front);
      float temperature = reaction * floorFade * flameCap
        * mix(0.11, 1.16, smoothstep(0.21, 0.78, 0.34 * roll + 0.53 * eddy + 0.13 * grain));
      vec3 sootP = p;
      sootP.x += 0.16 * twist.x - 0.10;
      sootP.z += 0.14 * twist.z;
      float hotSoot = smokePuff(sootP, vec3(0.02, 1.18, -0.11), vec3(0.69, 1.06, 0.60));
      float rolledSoot = smokePuff(sootP, vec3(0.32, 2.15, -0.23), vec3(0.66, 0.87, 0.58));
      float shedSoot = smokePuff(sootP, vec3(0.72, 2.57, -0.38), vec3(0.54, 0.76, 0.48));
      float sootShape = max(max(hotSoot - 0.16, rolledSoot), shedSoot - 0.18);
      float soot = smoothstep(0.03, 0.67, sootShape + 0.93 * erosion) * floorFade * (0.45 - 0.26 * temperature);
      fields += vec2(soot, temperature) * state.burning;
    }
  }

  if (state.blast > 0.001) {
    // Different scaled lobes gather and curl as the initial fireball rises.
    vec3 center = vec3(0.15 * state.expansion, 0.45 + 0.85 * state.expansion, 0.0);
    float extent = 0.31 + 1.17 * state.expansion;
    vec3 q = (p - center) / vec3(extent, extent * 0.91, extent * 0.96);
    q += 0.128 * sin(q.yzx * 4.35 + q.zxy * 1.29 + 3.44 * state.expansion);
    float roll = volumeNoise(q * 2.79 + vec3(11.9, 4.6, 26.4));
    float eddy = volumeNoise(q * 7.27 + vec3(5.7, 18.9 - rise * 0.34, 17.9) + (roll - 0.5) * 1.42);
    float grain = fineDetail ? volumeNoise(q * 17.8 + vec3(1.4, 21.0, 9.6)) : 0.5;
    float fracture = (roll - 0.52) * 0.67 + (eddy - 0.50) * 0.43 + (grain - 0.50) * 0.29;
    float crown = smokePuff(q, vec3(-0.18, 0.26, -0.21), vec3(0.88, 0.82, 0.78));
    float side = smokePuff(q, vec3(0.43, -0.11, 0.19), vec3(0.75, 0.71, 0.83));
    float base = smokePuff(q, vec3(-0.29, -0.33, 0.17), vec3(0.84, 0.64, 0.68));
    float shape = max(max(crown, side), base) + fracture;
    float body = smoothstep(-0.07, 0.54, shape) * floorFade;
    float flash = (0.095 + 0.89 * state.cooling);
    float rollTemperature = smoothstep(0.32, 0.76, 0.60 * eddy + 0.25 * roll + 0.15 * grain);
    float shell = 1.0 - 0.68 * smoothstep(0.49, 1.14, shape);
    float heat = body * flash * shell * (0.115 + 1.06 * rollTemperature);
    float soot = body * (0.46 + 0.73 * (1.0 - flash)) * (1.02 - 0.69 * rollTemperature * flash);
    vec3 delayed = p + vec3(0.24, -0.22, 0.19);
    delayed.y += 0.21 * (roll - 0.5);
    float cap = smokePuff(delayed, center + vec3(-0.24, 0.61, -0.26), vec3(extent * 1.17, extent * 0.84, extent * 0.93));
    soot += smoothstep(0.11, 0.58, cap + fracture) * (0.10 + 0.58 * (1.0 - flash)) * floorFade;
    fields = mix(fields, vec2(soot, heat), state.blast);
  }
  return fields;
}

vec3 pyroRadiance(float heat) {
  vec3 red = vec3(1.00, 0.043, 0.004);
  vec3 orange = vec3(1.00, 0.155, 0.011);
  vec3 amber = vec3(1.00, 0.527, 0.091);
  vec3 color = mix(red, orange, smoothstep(0.09, 0.68, heat));
  color = mix(color, amber, smoothstep(0.52, 0.93, heat));
  return color * heat * heat * (2.5 + heat * 2.3);
}

vec4 marchVolume(vec3 ro, vec3 rd, float maxT, float time, float phase, float detail, float jitter) {
  PyroState state;
  state.time = time + 0.91 * phase;
  state.burning = smoothstep(0.82, 1.21, phase);
  state.blast = smoothstep(1.94, 2.25, phase);
  state.expansion = 1.0 - exp(-4.35 * max(phase - 2.025, 0.0));
  state.cooling = 1.0 - smoothstep(2.34, 3.03, phase);

  vec3 extent = vec3(2.07, 4.20, 1.62) + state.blast * vec3(0.26, 0.07, 0.40);
  vec3 safeDir = vec3(rd.x >= 0.0 ? 1.0 : -1.0, rd.y >= 0.0 ? 1.0 : -1.0, rd.z >= 0.0 ? 1.0 : -1.0)
    * max(abs(rd), vec3(0.00001));
  vec3 nearSide = (vec3(-extent.x, 0.007, -extent.z) - ro) / safeDir;
  vec3 farSide = (extent - ro) / safeDir;
  vec3 start = min(nearSide, farSide);
  vec3 stop = max(nearSide, farSide);
  float enter = max(max(max(start.x, start.y), start.z), 0.0);
  float exit = min(min(min(stop.x, stop.y), stop.z), maxT);
  if (exit <= enter) return vec4(0.0);

  float samples = mix(34.0, 88.0, detail);
  float stepSize = max((exit - enter) / samples, mix(0.085, 0.030, detail));
  float t = enter + stepSize * mix(0.13, 0.87, clamp(jitter, 0.0, 1.0));
  float angle = dot(rd, volumeKey);
  float forwardScatter = 0.78 + 0.39 * angle + 0.31 * angle * angle;
  float sootExtinction = mix(4.8, 4.1, state.burning);
  float transmission = 1.0;
  vec3 rgb = vec3(0.0);
  for (int i = 0; i < 89; ++i) {
    if (t >= exit || transmission < 0.009) break;
    vec3 p = ro + rd * t;
    vec2 fields = pyroFields(p, state, detail > 0.15);
    float soot = fields.x;
    float heat = fields.y;
    if (soot + heat > 0.004) {
      float sigma = soot * sootExtinction + heat * 0.94;
      float alpha = 1.0 - exp(-sigma * min(stepSize, exit - t));
      vec3 radiance = pyroRadiance(heat);
      if (soot > 0.008) {
        float nearShadow = pyroFields(p + volumeKey * 0.13, state, detail > 0.72).x;
        float midShadow = pyroFields(p + volumeKey * 0.39, state, false).x;
        float farShadow = detail > 0.62 ? pyroFields(p + volumeKey * 0.89, state, false).x : midShadow * 0.60;
        float shadow = exp(-nearShadow * 1.32 - midShadow * 1.76 - farShadow * 1.66);
        vec3 ambient = mix(vec3(0.013, 0.016, 0.020), vec3(0.008, 0.009, 0.012), state.burning);
        vec3 key = mix(vec3(0.79, 0.83, 0.91), vec3(0.39, 0.40, 0.42), state.burning) * shadow * forwardScatter;
        vec3 heatOffset = p - vec3(0.0, mix(0.52, 0.96, state.blast), 0.0);
        float glow = state.burning * mix(0.11, 0.24 * state.cooling, state.blast)
          * exp(-dot(heatOffset, heatOffset) * mix(2.1, 1.2, state.blast));
        radiance += soot * sootExtinction * (ambient + key + glow * vec3(1.0, 0.21, 0.027));
      }
      rgb += transmission * radiance * (alpha / sigma);
      transmission *= 1.0 - alpha;
    }
    t += stepSize;
  }
  return vec4(rgb, 1.0 - transmission);
}
`;
