import { create } from 'zustand';
import type { DepthEstimator } from '@/engine/depth';
import type { Engine, Framing, Occlusion } from '@/engine/engine';
import { type Plate, releasePlate } from '@/engine/plate';
import { defaultTweaks, QUALITIES, type QualityId, type Tweaks } from '@/engine/scene';
import { presetById, PRESETS } from '@/presets/presets';

export type GpuStatus = 'loading' | 'ready' | 'unsupported' | 'error';
export type Backdrop = 'checker' | 'dark' | 'light';

type StudioState = {
  engine: Engine | null;
  status: GpuStatus;
  presetId: string;
  tweaks: Tweaks;
  backdrop: Backdrop;
  plate: Plate | null;
  framing: Framing;
  // The footage's depth is on while this holds the estimator.
  depthEstimator: DepthEstimator | null;
  occlusion: Occlusion;
  // The next click on the footage sets the fire's depth to the spot clicked.
  picking: boolean;
  quality: QualityId;
  exporting: boolean;
  sound: boolean;
  setEngine: (engine: Engine | null, status: GpuStatus) => void;
  setQuality: (quality: QualityId) => void;
  setPreset: (presetId: string) => void;
  setTweak: (key: keyof Tweaks, value: number) => void;
  resetTweaks: () => void;
  addCube: () => void;
  setBackdrop: (backdrop: Backdrop) => void;
  setPlate: (plate: Plate | null) => void;
  setFraming: (patch: Partial<Framing>) => void;
  setDepthEstimator: (estimator: DepthEstimator | null) => Promise<void>;
  setOcclusion: (patch: Partial<Occlusion>) => void;
  setPicking: (picking: boolean) => void;
  setExporting: (exporting: boolean) => void;
  toggleSound: () => void;
};

const CENTRED: Framing = { x: 0, y: 0, scale: 1 };

export const useStudioStore = create<StudioState>()((set, get) => ({
  engine: null,
  status: 'loading',
  presetId: PRESETS[0].id,
  tweaks: defaultTweaks(PRESETS[0]),
  backdrop: 'checker',
  plate: null,
  framing: CENTRED,
  depthEstimator: null,
  occlusion: { depth: 0.5, softness: 0.5, show: false },
  picking: false,
  quality: 'balanced',
  exporting: false,
  sound: false,
  setEngine: (engine, status) => {
    engine?.setTweaks(get().tweaks);
    engine?.setQuality(QUALITIES[get().quality]);
    engine?.setFraming(get().framing);
    engine?.setOcclusion(get().occlusion);
    void engine?.setPlate(get().plate);
    void engine?.setDepth(get().depthEstimator);
    set({ engine, status, sound: false });
  },
  setQuality: (quality) => {
    get().engine?.setQuality(QUALITIES[quality]);
    set({ quality });
  },
  setPreset: (presetId) => {
    const preset = presetById(presetId);
    get().engine?.setPreset(preset);
    set({ presetId, tweaks: { ...defaultTweaks(preset), seed: get().tweaks.seed, windYaw: get().tweaks.windYaw } });
  },
  setTweak: (key, value) => {
    get().engine?.setTweaks({ [key]: value });
    set((state) => ({ tweaks: { ...state.tweaks, [key]: value } }));
  },
  resetTweaks: () => {
    const { engine, presetId, tweaks: current } = get();
    const tweaks = { ...defaultTweaks(presetById(presetId)), seed: current.seed, windYaw: current.windYaw };
    engine?.setTweaks(tweaks);
    engine?.restart();
    set({ tweaks });
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
  // Taking the footage away puts the fire back in the middle of the frame.
  setPlate: (plate) => {
    const { engine, plate: old, framing } = get();
    void engine?.setPlate(plate);
    if (old) releasePlate(old);
    set({ plate, picking: false });
    if (!plate && framing !== CENTRED) get().setFraming(CENTRED);
  },
  setFraming: (patch) => {
    const framing = { ...get().framing, ...patch };
    get().engine?.setFraming(framing);
    set({ framing });
  },
  setDepthEstimator: async (depthEstimator) => {
    set({ depthEstimator, picking: false });
    await get().engine?.setDepth(depthEstimator);
  },
  setOcclusion: (patch) => {
    const occlusion = { ...get().occlusion, ...patch };
    get().engine?.setOcclusion(occlusion);
    set({ occlusion });
  },
  setPicking: (picking) => set({ picking }),
  setExporting: (exporting) => set({ exporting }),
  toggleSound: () => {
    const sound = !get().sound;
    get().engine?.setSound(sound);
    set({ sound });
  },
}));
