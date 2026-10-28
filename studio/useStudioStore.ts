import { create } from 'zustand';
import type { Engine } from '@/engine/engine';
import type { Tweaks } from '@/engine/scene';
import { presetById, PRESETS } from '@/presets/presets';

export type GpuStatus = 'loading' | 'ready' | 'unsupported' | 'error';
export type Backdrop = 'checker' | 'dark' | 'light';

type StudioState = {
  engine: Engine | null;
  status: GpuStatus;
  presetId: string;
  tweaks: Tweaks;
  backdrop: Backdrop;
  exporting: boolean;
  sound: boolean;
  setEngine: (engine: Engine | null, status: GpuStatus) => void;
  setPreset: (presetId: string) => void;
  setTweak: (key: keyof Tweaks, value: number) => void;
  addCube: () => void;
  setBackdrop: (backdrop: Backdrop) => void;
  setExporting: (exporting: boolean) => void;
  toggleSound: () => void;
};

export const useStudioStore = create<StudioState>()((set, get) => ({
  engine: null,
  status: 'loading',
  presetId: PRESETS[0].id,
  tweaks: { heat: 1, vorticity: PRESETS[0].vorticity, wind: 0, seed: 0, blue: PRESETS[0].blue, sparks: PRESETS[0].sparks  },
  backdrop: 'checker',
  exporting: false,
  sound: false,
  setEngine: (engine, status) => set({ engine, status, sound: false }),
  setPreset: (presetId) => {
    const preset = presetById(presetId);
    get().engine?.setPreset(preset);
    set({ presetId, tweaks: { heat: 1, vorticity: preset.vorticity, wind: 0, seed: get().tweaks.seed, blue: preset.blue, sparks: preset.sparks  } });
  },
  setTweak: (key, value) => {
    get().engine?.setTweaks({ [key]: value });
    set((state) => ({ tweaks: { ...state.tweaks, [key]: value } }));
  },
  addCube: () => {
    const half = 3 + Math.random() * 3;
    get().engine?.addBox({
      x: 18 + Math.random() * 28,
      y: 8 + Math.random() * 26,
      z: 22 + Math.random() * 20,
      yaw: Math.random() * Math.PI,
      hx: half,
      hy: half,
      hz: half,
    });
  },
  setBackdrop: (backdrop) => set({ backdrop }),
  setExporting: (exporting) => set({ exporting }),
  toggleSound: () => {
    const sound = !get().sound;
    get().engine?.setSound(sound);
    set({ sound });
  },
}));
