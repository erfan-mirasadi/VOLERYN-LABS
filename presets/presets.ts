import type { Preset } from '@/engine/scene';
import campfire from './campfire.json';
import explosion from './explosion.json';
import fire from './fire.json';
import smoke from './smoke.json';

export const PRESETS = [fire, campfire, smoke, explosion] as Preset[];

export const presetById = (id: string) => PRESETS.find((preset) => preset.id === id) ?? PRESETS[0];
