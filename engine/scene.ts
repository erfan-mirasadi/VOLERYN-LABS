// The base grid: every cell value in a preset refers to it, and the engine scales those to the grid it runs.
export const GRID = [64, 96, 64] as const;

// What a preview or an export spends: the simulation grid, how coarse the flame's light volume is, how densely
// the raymarch steps (in base-grid cells) and how the live resolution may scale to hold the target frame rate.
export type Quality = {
  grid: readonly [number, number, number];
  lightScale: number;
  stepCells: number;
  maxSteps: number;
  scaleMin: number;
  scaleMax: number;
  targetFps: number;
};

export const QUALITIES = {
  fast: { grid: GRID, lightScale: 4, stepCells: 2, maxSteps: 128, scaleMin: 0.5, scaleMax: 0.875, targetFps: 60 },
  balanced: { grid: GRID, lightScale: 4, stepCells: 1.5, maxSteps: 160, scaleMin: 0.625, scaleMax: 1, targetFps: 60 },
  fine: { grid: [96, 144, 96], lightScale: 4, stepCells: 1, maxSteps: 240, scaleMin: 0.75, scaleMax: 1, targetFps: 30 },
  ultra: { grid: [128, 192, 128], lightScale: 4, stepCells: 0.75, maxSteps: 320, scaleMin: 0.75, scaleMax: 1, targetFps: 30 },
} as const satisfies Record<string, Quality>;
export type QualityId = keyof typeof QUALITIES;

// What every render simulates on: sixty-four times the base cells and about two gigabytes of GPU memory, which
// only an offline render can afford.
export type SimulationGrid = Pick<Quality, 'grid' | 'lightScale'>;
export const RENDER_GRID: SimulationGrid = { grid: [256, 384, 256], lightScale: 8 };

// The flame's light volume is blurred to a Gaussian of LIGHT_SIGMA base cells whatever its cell size, so the light
// reaches the same distance on every grid; each 27-tap pass adds 0.567 cells² of variance. On the way the smoke
// absorbs the light as if it crossed LIGHT_PATH box widths of it, shared out over the passes.
export const LIGHT_SIGMA = 7.4;
export const LIGHT_PATH = 0.375;
export const lightBlurDispatches = ({ grid, lightScale }: SimulationGrid) => {
  const cell = (lightScale * GRID[0]) / grid[0];
  return Math.max(2, Math.round((LIGHT_SIGMA * LIGHT_SIGMA) / (0.567 * cell * cell)));
};
export const FIRE_LIGHT = 12;
export const DT = 1 / 60;
export const DENSITY = 14;
export const EXPOSURE = 1.0;
export const LIGHT = normalize([-0.45, 0.8, 0.4]);
// The key light and the sky and ground that light the smoke from above and below.
export const KEY_COLOR = [1.0, 0.95, 0.88];
export const KEY_STRENGTH = 0.7;
export const SKY_COLOR = [0.2, 0.23, 0.32];
export const GROUND_COLOR = [0.08, 0.07, 0.06];
export const MAX_BOXES = 4;
// The simulation box in world units: a cell is 1/GRID[0] wide and the floor sits at y = 0.
export const BOX_MIN = [-0.5, 0, (-0.5 * GRID[2]) / GRID[0]] as const;
export const BOX_MAX = [0.5, GRID[1] / GRID[0], (0.5 * GRID[2]) / GRID[0]] as const;
// Explosions go off this often, and their blast fades this fast after the charge goes off, in seconds.
export const BURST_EVERY = 5;
export const BLAST_DECAY = 0.08;
// The flame's sub-cell detail: cells per noise tile, and how often each of the two flow fields that carry it resets.
export const NOISE_PERIOD = 8;
export const FLOW_PERIOD = 1.0;

// An axis-aligned box in grid cells, turned about y; its skin gives off fuel and its body blocks the flow.
export type Box = { x: number; y: number; z: number; yaw: number; hx: number; hy: number; hz: number };

export type EmitterMode = 'fire' | 'smoke' | 'explosion' | 'objects';

export type Preset = {
  id: string;
  name: string;
  mode: EmitterMode;
  buoyancy: number;
  cooling: number;
  burn: number;
  // Fuel only burns where the gas is at least this hot.
  ignite: number;
  heat: number;
  smokeYield: number;
  dissipation: number;
  emit: number;
  radius: number;
  emitter: [number, number, number];
  vorticity: number;
  turbulence: number;
  damping: number;
  expansion: number;
  tint: [number, number, number];
  glow: number;
  // Extinction of the glowing soot per box width; with glow it gives the flame a body that hides what is behind it.
  fireDensity: number;
  // How far, in cells, the flame's sub-cell noise displaces the lookup; 0 turns it off.
  detail: number;
  // How much the fresh fuel at the base burns blue before soot forms; the Blue flame slider starts here.
  blue: number;
  // How many sparks the fire throws off, 0 to 1; the studio's default for its Sparks tweak.
  sparks: number;
  // Explosions only: how long the charge feeds, in seconds, and the peak divergence per cell at its centre, in cells/s.
  burst?: number;
  blast?: number;
  objects?: Box[];
};

// Live tweaks the studio layers over a preset without changing the preset itself.
// Wind blows along windYaw degrees from +x towards +z; smoke scales the preset's yield; width stretches the emitter sideways.
export type Tweaks = { heat: number; vorticity: number; wind: number; windYaw: number; seed: number; blue: number; sparks: number; smoke: number; width: number };

export const MODE_INDEX: Record<EmitterMode, number> = { fire: 0, smoke: 1, explosion: 2, objects: 3 };

function normalize(v: number[]) {
  const length = Math.hypot(...v);
  return v.map((x) => x / length) as [number, number, number];
}
