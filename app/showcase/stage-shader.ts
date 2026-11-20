export const stageShader = /* glsl */ `
const float stageRear = -5.2;
const float stageSide = 6.4;
const float stageCeiling = 5.3;
const vec2 stageWindowCenter = vec2(-3.78, 3.94);
const vec2 stageWindowExtent = vec2(1.63, .44);

float stagePlane(float offset, float direction, float previous) {
  if (abs(direction) < .0001) return previous;
  float t = offset / direction;
  return t > 0.001 ? min(t, previous) : previous;
}

float stageDistance(vec3 ro, vec3 rd) {
  float t = 100.0;
  if (rd.y < 0.0) t = stagePlane(-ro.y, rd.y, t);
  if (rd.y > 0.0) t = stagePlane(stageCeiling - ro.y, rd.y, t);
  if (rd.z < 0.0) t = stagePlane(stageRear - ro.z, rd.z, t);
  if (rd.x < 0.0) t = stagePlane(-stageSide - ro.x, rd.x, t);
  if (rd.x > 0.0) t = stagePlane(stageSide - ro.x, rd.x, t);
  return t;
}

float stageHash(vec2 p) {
  vec3 h = fract(vec3(p.xyx) * vec3(.1031, .11369, .13787));
  h += dot(h, h.yzx + 19.19);
  return fract((h.x + h.y) * h.z);
}

float stageNoise(vec2 p) {
  vec2 cell = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(stageHash(cell), stageHash(cell + vec2(1, 0)), f.x),
    mix(stageHash(cell + vec2(0, 1)), stageHash(cell + vec2(1, 1)), f.x),
    f.y
  );
}

float stageJoint(float x, float pitch, float width, float footprint) {
  float distanceToJoint = abs(fract(x / pitch + .5) - .5) * pitch;
  return 1.0 - smoothstep(width, width + footprint, distanceToJoint);
}

float stageWindow(vec2 p, float edge) {
  vec2 d = abs(p - stageWindowCenter) - stageWindowExtent;
  return 1.0 - smoothstep(-edge, edge, max(d.x, d.y));
}

vec3 stageColor(vec3 ro, vec3 rd, float t, vec3 effectPosition, vec3 effectLight, float rimLight) {
  if (t > 99.0) return vec3(.007, .010, .014);
  vec3 p = ro + rd * t;
  vec3 n = vec3(0, 1, 0);
  vec2 surface = p.xz;
  vec3 base = vec3(.078, .087, .095);
  float floorSurface = 1.0;
  bool rearWall = false;
  if (p.z <= stageRear + .002 && p.y > .002) {
    n = vec3(0, 0, 1);
    surface = p.xy;
    base = vec3(.086, .093, .096);
    floorSurface = 0.0;
    rearWall = true;
  } else if (abs(p.x) >= stageSide - .002 && p.y > .002) {
    n = vec3(-sign(p.x), 0, 0);
    surface = p.zy;
    base = vec3(.066, .075, .083);
    floorSurface = 0.0;
  } else if (p.y >= stageCeiling - .002) {
    n = vec3(0, -1, 0);
    base = vec3(.056, .063, .070);
    floorSurface = 0.0;
  }

  // Subtle mottled concrete. Fade the smallest aggregate when the viewing ray gets shallow.
  float facing = max(.12, abs(dot(n, rd)));
  float footprint = max(.005, t * .0017 / facing);
  float cloudy = stageNoise(surface * 1.25 + vec2(3.0, 7.0));
  float aggregate = stageNoise(surface * 14.0) - .5;
  float finish = .87 + .27 * cloudy + .13 * aggregate * (1.0 - smoothstep(.015, .07, footprint));
  float brushed = 1.0;
  float steel = 0.0;
  float jointU = stageJoint(surface.x + 1.58, 3.55, .005, footprint);
  float jointV = stageJoint(surface.y - .64, 3.55, .005, footprint);
  float panels = max(jointU, floorSurface * jointV);
  panels *= 1.0 - smoothstep(.024, .1, footprint);
  base *= finish * mix(1.0, .76, panels);

  if (floorSurface > .5 && dot(surface, surface) < 2.20) {
    // A flush, blackened steel insert. Its outer collar catches the key while the slots stay dark.
    float r = length(surface);
    float aa = max(.003, footprint * .62);
    float outer = 1.0 - smoothstep(1.28 - aa, 1.28 + aa, r);
    float seat = smoothstep(1.22 - aa, 1.22 + aa, r) * outer;
    float collar = smoothstep(1.035 - aa, 1.035 + aa, r) * (1.0 - smoothstep(1.22 - aa, 1.22 + aa, r));
    float inner = 1.0 - smoothstep(1.035 - aa, 1.035 + aa, r);
    brushed = .5 + .5 * stageNoise(surface * vec2(46.0, 4.0));
    float shine = smoothstep(.982, 1.02, r) * (1.0 - smoothstep(1.03, 1.08, r));
    float slot = stageJoint(surface.x + .075, .175, .017, aa);
    float open = slot * (1.0 - smoothstep(.89 - aa, .89 + aa, r))
      * smoothstep(.30 - aa, .30 + aa, r) * (1.0 - smoothstep(.016, .07, aa));
    vec3 metal = vec3(.058, .063, .065) * (.84 + .24 * brushed);
    metal *= 1.0 - .74 * open;
    metal += vec3(.055, .061, .066) * shine;
    vec3 casing = metal * inner + vec3(.081, .086, .085) * collar + vec3(.018, .022, .026) * seat;
    float insetShadow = .23 * (1.0 - smoothstep(1.27, 1.42, r));
    base = base * (1.0 - insetShadow) * (1.0 - outer) + casing;
    steel = clamp(collar + inner * .9, 0.0, 1.0);
  }

  // A large, cool off-camera key grazes the back wall and the polished floor.
  vec3 keyOffset = vec3(-3.4, 4.1, -2.4) - p;
  float keyDistance = dot(keyOffset, keyOffset);
  vec3 keyDirection = keyOffset * inversesqrt(max(keyDistance, .001));
  float keyDiffuse = max(0.0, dot(n, keyDirection)) * 9.2 / (8.0 + keyDistance);
  vec3 keyColor = mix(vec3(.60, .78, 1.0), vec3(.76, .79, .80), rimLight);
  float lowerWall = mix(smoothstep(0.0, 1.1, p.y), 1.0, floorSurface);
  float corner = mix(.57, 1.0, lowerWall);
  float rearEdge = smoothstep(0.0, 1.1, max(0.0, p.z - stageRear));
  corner *= mix(.67, 1.0, max(rearEdge, 1.0 - floorSurface));
  float sideEdge = smoothstep(0.0, 1.48, stageSide - abs(p.x));
  corner *= mix(.46, 1.0, mix(sideEdge, 1.0, 1.0 - floorSurface));
  vec3 diffuse = base * corner * (vec3(.098, .115, .138) + keyColor * keyDiffuse);

  if (rearWall) {
    // Set luminous frosted panes back behind a dark reveal. The view ray shifts the glass
    // against the reveal as the camera moves.
    float recess = stageWindow(surface, footprint);
    vec2 pane = surface + rd.xy * (.17 / max(.16, -rd.z));
    float glass = stageWindow(pane, footprint) * recess;
    float mullion = 1.0 - smoothstep(.015, .015 + footprint, abs(pane.x - stageWindowCenter.x + .28));
    float sky = .16 + .18 * smoothstep(3.48, 4.26, pane.y);
    vec3 paneLight = keyColor * sky * (1.0 - .88 * mullion);
    diffuse = mix(diffuse, vec3(.006, .009, .012), recess) + glass * paneLight;
  }

  // The volume contributes soft direct spill and a wide specular’s tail to the floor.
  vec3 fireOffset = effectPosition - p;
  float fireDistance = dot(fireOffset, fireOffset);
  vec3 fireDirection = fireOffset * inversesqrt(max(fireDistance, .001));
  float fireNormal = max(0.0, dot(n, fireDirection));
  float incident = fireNormal / (1.2 + fireDistance);
  diffuse += base * effectLight * incident;
  if (floorSurface < .5) return diffuse;

  vec3 keyHalf = normalize(keyDirection - rd + vec3(0.0, .0001, 0.0));
  vec3 fireHalf = normalize(fireDirection - rd + vec3(0.0, .0001, 0.0));
  float keyHighlight = pow(max(0.0, dot(n, keyHalf)), mix(14.0, 32.0, steel))
    * keyDiffuse * mix(.032, .110, steel);
  float fireHighlight = pow(max(0.0, dot(n, fireHalf)), mix(11.0, 26.0, steel))
    * incident * mix(.072, .186, steel) * (.90 + .10 * brushed);
  vec3 reflection = reflect(rd, n);
  float travel = (stageRear - p.z) / min(-.001, reflection.z);
  vec2 reflectedWall = p.xy + travel * reflection.xy;
  float frostedReflection = stageWindow(reflectedWall, mix(.85, .37, steel));
  float fresnel = .021 + .23 * pow(1.0 - facing, 4.0);
  return diffuse + keyColor * (keyHighlight + frostedReflection * fresnel)
    + effectLight * fireHighlight;
}
`;
