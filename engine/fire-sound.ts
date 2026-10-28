import type { Vec3 } from './render';
import { type Box, BOX_MAX, BOX_MIN, GRID, type Preset, type Tweaks } from './scene';

const NOISE_SECONDS = 2;
// The loops replay one noise buffer at unrelated rates, so no repeat is ever audible.
const BED_RATE = 0.31;
const AIR_RATE = 0.73;
const FLUTTER_RATE = 0.017;
const LOOKAHEAD = 0.1;
const MASTER = 0.8;
const DEBRIS_RATE = 80;
const DEBRIS_DECAY = 0.5;
// How long the fireball roars at full strength before it dies away with the simulation's cooling.
const ROAR_HOLD = 0.3;

export type FireSound = ReturnType<typeof createFireSound>;

export function createFireSound(ctx: BaseAudioContext, orbitDistance: number) {
  const rate = ctx.sampleRate;
  const noise = ctx.createBuffer(1, NOISE_SECONDS * rate, rate);
  const noiseData = noise.getChannelData(0);
  for (let i = 0; i < noiseData.length; i++) noiseData[i] = Math.random() * 2 - 1;
  const click = ctx.createBuffer(1, Math.round(0.06 * rate), rate);
  const clickData = click.getChannelData(0);
  for (let i = 0; i < clickData.length; i++) clickData[i] = (Math.random() * 2 - 1) * Math.exp(-i / (0.0025 * rate));

  const glide = (param: AudioParam, value: number, tau = 0.1) => {
    param.cancelScheduledValues(ctx.currentTime);
    param.setTargetAtTime(value, ctx.currentTime, tau);
  };
  const gain = (value = 0) => new GainNode(ctx, { gain: value });
  const filter = (type: BiquadFilterType, frequency: number, Q = 0.7) => new BiquadFilterNode(ctx, { type, frequency, Q });
  const loop = (playbackRate: number) => {
    const source = new AudioBufferSourceNode(ctx, { buffer: noise, loop: true, playbackRate });
    source.start(0, Math.random() * NOISE_SECONDS);
    return source;
  };
  const panner = () => new PannerNode(ctx, { panningModel: 'equalpower', distanceModel: 'inverse', refDistance: orbitDistance });
  // RMS of the flutter noise after a lowpass at cutoff, so a depth means the same at any cutoff and sample rate.
  const flutterRms = (cutoff: number) => 0.577 * Math.sqrt((1.11 * cutoff) / ((FLUTTER_RATE * rate) / 2));

  const compressor = new DynamicsCompressorNode(ctx, { threshold: -12, knee: 6, ratio: 4, attack: 0.003, release: 0.25 });
  const master = gain();
  const analyser = new AnalyserNode(ctx, { fftSize: 2048 });
  compressor.connect(master).connect(analyser).connect(ctx.destination);
  const base = panner();
  const plume = panner();
  base.connect(compressor);
  plume.connect(compressor);

  const body = gain(1);
  body.connect(base);
  const bed = loop(BED_RATE);
  const rumble = gain();
  const roarFilter = filter('lowpass', 500);
  const roar = gain();
  bed.connect(filter('lowpass', 110)).connect(rumble).connect(body);
  bed.connect(roarFilter).connect(roar).connect(body);

  const flutter = loop(FLUTTER_RATE);
  const puffFilter = filter('lowpass', 4);
  const puff = gain();
  const gust = gain();
  flutter.connect(puffFilter).connect(puff).connect(body.gain);
  flutter.connect(filter('lowpass', 0.4)).connect(gust);

  const air = loop(AIR_RATE);
  const hiss = gain();
  const wind = gain();
  air.connect(filter('bandpass', 2200)).connect(hiss).connect(plume);
  air.connect(filter('lowpass', 400, 0.5)).connect(wind).connect(plume);
  gust.connect(wind.gain);

  const world = (cell: number, axis: 0 | 1 | 2) => BOX_MIN[axis] + ((cell + 0.5) / GRID[axis]) * (BOX_MAX[axis] - BOX_MIN[axis]);
  const place = (node: PannerNode, cell: readonly number[], lean: number, rise: number) => {
    glide(node.positionX, world(cell[0], 0) + lean);
    glide(node.positionY, world(cell[1], 1) + rise);
    glide(node.positionZ, world(cell[2], 2));
  };

  let rest = { rumble: 0, roar: 0, cutoff: 500 };
  let crackleRate = 0;
  let crackleGain = 0;
  let nextCrackle = 0;
  let strength = 0;
  let cooling = 1;
  let burstAt = -Infinity;
  let burstStrength = 0;

  function scene(preset: Preset, tweaks: Tweaks, boxes: Box[]) {
    const h = tweaks.heat;
    const w = Math.abs(tweaks.wind) / 60;
    const objects = preset.mode === 'objects';
    // Emitting cells: the 2.5-cell skin at mean weight 0.5 around each box, or the (1 - d²) ball of the fire kernel.
    const cells = objects
      ? boxes.reduce((sum, b) => sum + 10 * (b.hx * b.hy + b.hy * b.hz + b.hz * b.hx), 0)
      : 0.4 * (4 / 3) * Math.PI * preset.radius ** 3;
    const burning = preset.mode !== 'explosion';
    const power = burning ? (preset.emit * preset.heat * h * cells) / 47600 : 0;
    const flow = burning ? (preset.emit * cells) / 36600 : 0;
    const centre = objects ? (['x', 'y', 'z'] as const).map((key) => boxes.reduce((sum, b) => sum + b[key], 0) / boxes.length) : preset.emitter;
    place(base, centre, 0, 0);
    place(plume, centre, (0.35 * tweaks.wind) / 60, 0.7);

    rest = { rumble: 0.35 * Math.sqrt(power), roar: 0.3 * Math.sqrt(power), cutoff: 250 + 350 * Math.min(1.5, (preset.buoyancy * h) / 40) };
    glide(rumble.gain, rest.rumble);
    glide(roar.gain, rest.roar);
    glide(roarFilter.frequency, rest.cutoff);
    const puffRate = 2.5 * Math.sqrt((preset.buoyancy * h) / 32) + 0.5 * tweaks.vorticity + 2 * w;
    glide(puffFilter.frequency, puffRate);
    glide(puff.gain, (0.25 + (0.5 * tweaks.vorticity) / 14) / flutterRms(puffRate));
    glide(hiss.gain, Math.sqrt(flow) * (0.03 + (0.025 * preset.turbulence) / 60) + 0.08 * w);
    const windLevel = 0.8 * w ** 1.5;
    glide(wind.gain, windLevel);
    glide(gust.gain, (0.6 * windLevel) / flutterRms(0.4));

    crackleRate = power * (objects ? 35 : 4);
    crackleGain = objects ? 0.9 : 0.5;
    strength = Math.min(1.3, (preset.expansion * preset.emit * h) / 5.6);
    cooling = preset.cooling;
  }

  // A wood crackle: a few milliseconds of noise ringing through a random band; one in eight is a deeper pop.
  function crackle(time: number) {
    const pop = Math.random() < 1 / 8;
    const u = Math.random();
    const source = new AudioBufferSourceNode(ctx, { buffer: click, playbackRate: pop ? 0.25 + 0.15 * u : 0.6 + 1.2 * u });
    const ring = pop ? filter('bandpass', 300 * 3 ** Math.random(), 3) : filter('bandpass', 1500 * (10 / 3) ** Math.random(), 2 + 6 * Math.random());
    source.connect(ring).connect(gain(crackleGain * (pop ? 2 : 1) * (0.3 + 0.7 * Math.random() ** 2))).connect(base);
    source.start(time);
  }

  // Crackles are a Poisson process scheduled a little ahead of the clock, once per frame and with no timers.
  function update() {
    const now = ctx.currentTime;
    nextCrackle = Math.max(nextCrackle, now);
    while (nextCrackle < now + LOOKAHEAD) {
      const age = nextCrackle - burstAt;
      const rate = crackleRate + (age >= 0 && age < 1.8 ? DEBRIS_RATE * burstStrength * Math.exp(-age / DEBRIS_DECAY) : 0);
      if (rate < 0.1) {
        nextCrackle = now + LOOKAHEAD;
        break;
      }
      crackle(nextCrackle);
      nextCrackle -= Math.log(1 - Math.random()) / rate;
    }
  }

  function burst() {
    const t = ctx.currentTime;
    burstAt = t;
    burstStrength = strength;
    const crack = new AudioBufferSourceNode(ctx, { buffer: click });
    crack.connect(gain(0.8 * strength)).connect(base);
    crack.start(t);

    // The thump skips the panners: low end carries no direction.
    const thump = new OscillatorNode(ctx, { frequency: 110 });
    const thumpGain = gain();
    thump.frequency.setValueAtTime(110, t);
    thump.frequency.exponentialRampToValueAtTime(32, t + 0.35);
    thumpGain.gain.setValueAtTime(0, t);
    thumpGain.gain.linearRampToValueAtTime(0.7 * strength, t + 0.01);
    thumpGain.gain.exponentialRampToValueAtTime(0.001, t + 0.7);
    thump.connect(thumpGain).connect(compressor);
    thump.start(t);
    thump.stop(t + 0.75);

    const rush = new AudioBufferSourceNode(ctx, { buffer: noise, loop: true, playbackRate: 0.7 });
    const sweep = filter('lowpass', 5000);
    const tail = gain();
    sweep.frequency.setValueAtTime(5000, t);
    sweep.frequency.exponentialRampToValueAtTime(120, t + 2.5);
    tail.gain.setValueAtTime(0, t);
    tail.gain.linearRampToValueAtTime(0.75 * strength, t + 0.03);
    tail.gain.exponentialRampToValueAtTime(0.001, t + 2.8);
    rush.connect(sweep).connect(tail).connect(base);
    rush.start(t, Math.random() * NOISE_SECONDS);
    rush.stop(t + 2.9);

    // The fireball roars while it burns, then dies away as fast as the simulation cools it.
    const swell: [AudioParam, number, number][] = [
      [roar.gain, 0.8 * strength, rest.roar],
      [rumble.gain, 0.8 * strength, rest.rumble],
      [roarFilter.frequency, 1200, rest.cutoff],
    ];
    for (const [param, peak, settle] of swell) {
      glide(param, peak, 0.02);
      param.setTargetAtTime(settle, t + ROAR_HOLD, 1 / cooling);
    }
  }

  function listen(eye: Vec3, target: Vec3) {
    const listener = ctx.listener;
    const forward = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]] as Vec3;
    // Firefox has no AudioParams on the listener.
    if (!listener.positionX) {
      listener.setPosition(...eye);
      listener.setOrientation(...forward, 0, 1, 0);
      return;
    }
    [listener.positionX, listener.positionY, listener.positionZ].forEach((param, i) => glide(param, eye[i], 0.03));
    [listener.forwardX, listener.forwardY, listener.forwardZ].forEach((param, i) => glide(param, forward[i], 0.03));
  }

  const samples = new Float32Array(analyser.fftSize);
  function level() {
    analyser.getFloatTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) sum += sample * sample;
    return Math.max(-99, 10 * Math.log10(sum / samples.length));
  }

  return {
    scene,
    update,
    burst,
    listen,
    level,
    fade: (on: boolean) => glide(master.gain, on ? MASTER : 0, on ? 0.03 : 0.01),
  };
}
