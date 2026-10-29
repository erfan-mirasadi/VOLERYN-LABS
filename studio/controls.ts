import type { Tweaks } from '@/engine/scene';

// The studio's live controls over a preset, with the range each slider offers.
export const SLIDERS = [
  { key: 'heat', label: 'Heat', min: 0.5, max: 1.6, step: 0.01 },
  { key: 'vorticity', label: 'Vorticity', min: 0, max: 14, step: 0.1 },
  { key: 'wind', label: 'Wind', min: -60, max: 60, step: 1 },
  { key: 'seed', label: 'Seed', min: 0, max: 99, step: 1 },
  { key: 'blue', label: 'Blue flame', min: 0, max: 1, step: 0.01 },
  { key: 'sparks', label: 'Sparks', min: 0, max: 1, step: 0.01 },
  { key: 'smoke', label: 'Smoke', min: 0, max: 2, step: 0.01 },
  { key: 'width', label: 'Width', min: 0.5, max: 3, step: 0.05 },
] as const satisfies readonly { key: keyof Tweaks; label: string; min: number; max: number; step: number }[];
