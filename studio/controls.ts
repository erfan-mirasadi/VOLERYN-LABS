import type { Tweaks } from '@/engine/scene';

export type ControlGroup = 'source' | 'motion' | 'shading' | 'sparks';
export type Slider = {
  key: keyof Tweaks; label: string; min: number; max: number; step: number;
  group: ControlGroup; help: string; unit?: string; advanced?: boolean; explosion?: boolean;
};

// Multipliers change one physical or optical property, without secretly changing the others.
export const SLIDERS: readonly Slider[] = [
  { key: 'heat', label: 'Heat release', min: 0.4, max: 2, step: 0.01, group: 'source', unit: '×', help: 'Energy released by burning fuel. Changes combustion and the resulting motion.' },
  { key: 'fuel', label: 'Fuel amount', min: 0.1, max: 2.5, step: 0.01, group: 'source', unit: '×', help: 'Fuel fed into the source. More fuel gives a richer, smokier burn.' },
  { key: 'width', label: 'Source width', min: 0.5, max: 2.5, step: 0.05, group: 'source', unit: '×', help: 'Horizontal extent of the source, before the gas expands.' },
  { key: 'smoke', label: 'Soot yield', min: 0, max: 2.5, step: 0.01, group: 'source', unit: '×', help: 'Smoke produced by combustion. Use Smoke density below to adjust its rendered opacity.' },
  { key: 'seed', label: 'Seed', min: 0, max: 999, step: 1, group: 'source', help: 'Deterministic source variation. Replay to compare the same seed from ignition.' },
  { key: 'burnRate', label: 'Burn rate', min: 0.2, max: 3, step: 0.01, group: 'source', unit: '×', advanced: true, help: 'How quickly fuel reacts after ignition. Lower values prolong burning.' },
  { key: 'cooling', label: 'Cooling', min: 0.2, max: 3, step: 0.01, group: 'source', unit: '×', advanced: true, help: 'Rate of heat loss. Higher values shorten the incandescent phase.' },
  { key: 'blue', label: 'Blue base', min: 0, max: 1, step: 0.01, group: 'source', advanced: true, help: 'Blue contribution around fresh fuel. Usually leave at zero for a sooty explosion.' },
  { key: 'wind', label: 'Wind speed', min: -60, max: 60, step: 1, group: 'motion', unit: 'cell/s', help: 'Ambient flow in base grid cells per second; 64 cells span the box. Negative values reverse the direction.' },
  { key: 'windYaw', label: 'Direction', min: 0, max: 359, step: 1, group: 'motion', unit: '°', help: 'Wind heading around the floor, also adjustable by dragging the arrow in the view.' },
  { key: 'windGust', label: 'Gusts', min: 0, max: 1, step: 0.01, group: 'motion', help: 'Smooth variation of wind speed. Zero gives steady wind.' },
  { key: 'buoyancy', label: 'Lift', min: 0, max: 2.5, step: 0.01, group: 'motion', unit: '×', advanced: true, help: 'Upward acceleration from hot gas, independent of how brightly it renders.' },
  { key: 'turbulence', label: 'Large swirls', min: 0, max: 3, step: 0.01, group: 'motion', unit: '×', advanced: true, help: 'Broad curling motion that breaks the overall silhouette.' },
  { key: 'vorticity', label: 'Vortex strength', min: 0, max: 14, step: 0.1, group: 'motion', advanced: true, help: 'Restores small vortices lost to grid advection. Extreme values can look restless.' },
  { key: 'disturbance', label: 'Fine breakup', min: 0, max: 3, step: 0.01, group: 'motion', unit: '×', advanced: true, help: 'Small-scale motion concentrated around hot, occupied gas.' },
  { key: 'expansion', label: 'Expansion', min: 0.2, max: 2, step: 0.01, group: 'motion', unit: '×', advanced: true, help: 'Expansion from burning gas and the initial blast.', explosion: true },
  { key: 'exposure', label: 'Exposure', min: -5, max: 5, step: 0.05, group: 'shading', unit: 'EV', help: 'Fire and smoke exposure, in stops. The footage itself is unchanged.' },
  { key: 'temperature', label: 'Temperature', min: 1000, max: 3000, step: 10, group: 'shading', unit: 'K', help: 'Reference blackbody temperature for the incandescent soot. Hotter values are whiter and brighter; use Exposure to balance the shot.' },
  { key: 'smokeDensity', label: 'Smoke density', min: 0, max: 3, step: 0.01, group: 'shading', unit: '×', help: 'Rendered smoke extinction. Independent of the amount of smoke simulated.' },
  { key: 'fireOpacity', label: 'Flame body', min: 0, max: 3, step: 0.01, group: 'shading', unit: '×', advanced: true, help: 'Extinction of incandescent soot. Lower values reveal more of the background.' },
  { key: 'detail', label: 'Flow detail', min: 0, max: 3, step: 0.01, group: 'shading', unit: '×', advanced: true, help: 'Flow-carried volume detail. Zero shows only the simulated grid.' },
  { key: 'ambient', label: 'Smoke light', min: 0, max: 3, step: 0.01, group: 'shading', unit: '×', advanced: true, help: 'Environment illumination of smoke. Lower it for night shots; raise it for bright daylight.' },
  { key: 'keyLight', label: 'Key light', min: 0, max: 3, step: 0.01, group: 'shading', unit: '×', advanced: true, help: 'Directional light on smoke. Match the strength of sunlight or the main light in the shot.' },
  { key: 'lightYaw', label: 'Light heading', min: -180, max: 180, step: 1, group: 'shading', unit: '°', advanced: true, help: 'Direction of the main light around the scene. Updates smoke self-shadowing.' },
  { key: 'lightElevation', label: 'Light height', min: 5, max: 85, step: 1, group: 'shading', unit: '°', advanced: true, help: 'Main light elevation above the horizon. Low angles reveal the folds in the smoke.' },
  { key: 'sparks', label: 'Amount', min: 0, max: 1, step: 0.01, group: 'sparks', help: 'Rate of incandescent particles. Low values suit distant explosions.' },
  { key: 'sparkSize', label: 'Size', min: 0.25, max: 3, step: 0.05, group: 'sparks', unit: '×', advanced: true, help: 'Physical particle size. Smaller embers cool sooner and follow the gas more closely.' },
  { key: 'sparkLife', label: 'Lifetime', min: 0.25, max: 3, step: 0.05, group: 'sparks', unit: '×', advanced: true, help: 'Cooling lifetime of particles. Export trails follow the selected camera shutter.' },
];

export function clampTweak(key: keyof Tweaks, value: number) {
  const slider = SLIDERS.find((item) => item.key === key);
  if (!slider || !Number.isFinite(value)) return null;
  const bounded = Math.min(slider.max, Math.max(slider.min, value));
  return Number((Math.round(bounded / slider.step) * slider.step).toFixed(4));
}
