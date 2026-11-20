import { stageShader } from './stage-shader';
import { makeVolumeNoise, volumeShader, VOLUME_NOISE_SIZE } from './volume-shader';

const vertexSource = `#version 300 es
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}
`;

const fragmentSource = `#version 300 es
precision highp float;
uniform highp sampler3D uNoise;
uniform vec2 uResolution;
uniform vec3 uCamera;
uniform vec3 uTarget;
uniform vec3 uProjection;
uniform vec3 uScene;
out vec4 pixel;

${stageShader}
${volumeShader}

void main() {
  vec2 p = (2.0 * gl_FragCoord.xy - uResolution) / uResolution.y;
  vec3 forward = normalize(uTarget - uCamera);
  vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, forward);
  vec3 rd = normalize(forward + right * (p.x * uProjection.x + uProjection.y)
    + up * (p.y * uProjection.x + uProjection.z));
  float time = uScene.x;
  float phase = uScene.y;
  float fire = smoothstep(0.86, 1.16, phase) * (1.0 - smoothstep(1.76, 2.0, phase));
  float blast = smoothstep(1.96, 2.23, phase) * (1.0 - smoothstep(2.33, 2.83, phase));
  vec3 light = vec3(3.4, 0.78, 0.12) * fire * (0.94 + 0.06 * sin(time * 4.3))
    + vec3(6.5, 1.45, 0.24) * blast;
  float surface = stageDistance(uCamera, rd);
  vec3 color = stageColor(uCamera, rd, surface, vec3(0.0, 0.43 + blast * 0.82, 0.0), light, max(fire, blast));
  float grain = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  vec4 volume = marchVolume(uCamera, rd, surface, time, phase, uScene.z, grain);
  color = max(color * (1.0 - volume.a) + volume.rgb, 0.0);
  float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
  color /= 1.0 + 0.55 * luminance;
  color *= 1.0 - 0.14 * smoothstep(0.45, 1.7, length(p * vec2(0.7, 0.85)));
  vec3 displayColor = mix(color * 12.92, 1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, color));
  pixel = vec4(displayColor + (grain - 0.5) * 0.0035, 1.0);
}
`;

interface DrawFrame {
  phase: number;
  time: number;
  orbit: number;
  elevation: number;
  width: number;
  height: number;
  animating: boolean;
  interactive: boolean;
}

function createShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Shader allocation failed');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  return shader;
}

export async function createShowcaseRenderer(canvas: HTMLCanvasElement, signal: AbortSignal) {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    depth: false,
    stencil: false,
    failIfMajorPerformanceCaveat: true,
  });
  if (!gl) throw new Error('WebGL2 unavailable');

  const vertex = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (!program) throw new Error('Program allocation failed');
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);

  const compiler = gl.getExtension('KHR_parallel_shader_compile');
  if (compiler) {
    while (!signal.aborted && !gl.isContextLost() && !gl.getProgramParameter(program, compiler.COMPLETION_STATUS_KHR)) {
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }
  if (signal.aborted) {
    gl.deleteProgram(program);
    signal.throwIfAborted();
  }
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(message || 'Scene shader could not link');
  }
  const noise = gl.createTexture();
  if (!noise) {
    gl.deleteProgram(program);
    throw new Error('Noise texture allocation failed');
  }
  gl.useProgram(program);
  gl.bindTexture(gl.TEXTURE_3D, noise);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.REPEAT);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const size = VOLUME_NOISE_SIZE;
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, size, size, size, 0, gl.RED, gl.UNSIGNED_BYTE, makeVolumeNoise());

  const resolution = gl.getUniformLocation(program, 'uResolution');
  gl.uniform2f(resolution, canvas.width, canvas.height);
  const camera = gl.getUniformLocation(program, 'uCamera');
  const target = gl.getUniformLocation(program, 'uTarget');
  const projection = gl.getUniformLocation(program, 'uProjection');
  const scene = gl.getUniformLocation(program, 'uScene');
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  let query: WebGLQuery | null = null;
  let quality = 1.0;
  let gpuTime = 0;
  let lastAdaptation = 0;

  function adjustQuality(now: number, ms: number) {
    gpuTime = gpuTime ? 0.8 * gpuTime + 0.2 * ms : ms;
    if (now - lastAdaptation < 360) return;
    if (gpuTime > 25) quality = Math.max(0.50, quality * 0.84);
    if (gpuTime < 14) quality = Math.min(1, quality + 0.055);
    lastAdaptation = now;
  }

  return {
    draw(frame: DrawFrame, now: number, relativeCadence: number) {
      if (timer && query && gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) {
        if (!gl.getParameter(timer.GPU_DISJOINT_EXT)) {
          adjustQuality(now, Number(gl.getQueryParameter(query, gl.QUERY_RESULT)) / 1e6);
        }
        gl.deleteQuery(query);
        query = null;
      } else if (!timer && relativeCadence > 0) {
        // This is frame cadence, so only use it to detect sustained stalls.
        adjustQuality(now, 17 + (relativeCadence - 1) * 44);
      }
      const aspect = frame.width / frame.height;
      const DPR = Math.min(window.devicePixelRatio || 1, 1.55);
      const budget = aspect < 0.8 ? 570_000 : 1_240_000;
      const detail = frame.animating || frame.interactive ? quality : Math.max(0.85, quality);
      const sizeScale = Math.min(DPR, Math.sqrt(budget * detail * detail / (frame.width * frame.height)));
      const width = Math.round(frame.width * sizeScale);
      const height = Math.round(frame.height * sizeScale);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
        gl.uniform2f(resolution, width, height);
      }
      // World-space dolly and orbit. The camera passes opposite sides of the emitter.
      const phase = frame.phase;
      const azimuth = -0.43 + 0.69 * Math.sin(phase * 0.92) + frame.orbit;
      const radius = aspect < 0.82 ? 7.15 - 0.23 * Math.sin(phase * 1.55) : 6.05 - 0.52 * Math.sin(phase * 1.28);
      const eyeHeight = 1.89 + 0.44 * Math.cos(phase * 1.41) + frame.elevation;
      gl.uniform3f(camera, Math.sin(azimuth) * radius, eyeHeight, Math.cos(azimuth) * radius);
      gl.uniform3f(target, 0, 1.51, 0);
      const fov = Math.max(0.375, Math.min(0.66, 0.29 / aspect));
      const shift = Math.max(0, Math.min(0.29, (aspect - 0.70) * 0.205));
      gl.uniform3f(projection, fov, -shift, -0.015);
      gl.uniform3f(scene, frame.time, phase, detail);
      const measure = timer && !query && (frame.animating || frame.interactive);
      if (measure) {
        query = gl.createQuery();
        if (query) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (measure && query) gl.endQuery(timer.TIME_ELAPSED_EXT);
      return gpuTime;
    },

    dispose() {
      if (query) gl.deleteQuery(query);
      gl.deleteTexture(noise);
      gl.deleteProgram(program);
    },
  };
}

export type ShowcaseRenderer = Awaited<ReturnType<typeof createShowcaseRenderer>>;
