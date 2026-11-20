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
  canUndo: boolean;
  recordUndo: (restore: () => void) => void;
  beginAdjustment: (pointerId: number) => void;
  undo: () => void;
  setEngine: (engine: Engine | null, status: GpuStatus) => void;
  setQuality: (quality: QualityId) => void;
  setPreset: (presetId: string, trackUndo?: boolean) => void;
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

export const CENTRED: Framing = { x: 0, y: 0, scale: 1 };

export const DEFAULT_OCCLUSION: Occlusion = { depth: 0.5, softness: 0.5, show: false };
const UNDO_LIMIT = 100;

export const useStudioStore = create<StudioState>()((set, get) => {
  const history: (() => void)[] = [];
  let gesture: { finish: () => void; recorded: boolean } | null = null;
  let restoring = false;

  function recordUndo(restore: () => void) {
    if (restoring || gesture?.recorded) return;
    if (gesture) gesture.recorded = true;
    history.push(restore);
    if (history.length > UNDO_LIMIT) history.shift();
  }

  function applyPreset(presetId: string, tweaks: Tweaks) {
    const { engine } = get();
    engine?.setPreset(presetById(presetId));
    engine?.setTweaks(tweaks);
    set({ presetId, tweaks });
  }

  function remember(restore: () => void) {
    recordUndo(restore);
    if (!get().canUndo && history.length) set({ canUndo: true });
  }

  return {
    engine: null,
    status: 'loading',
    presetId: PRESETS[0].id,
    tweaks: defaultTweaks(PRESETS[0]),
    backdrop: 'checker',
    plate: null,
    framing: CENTRED,
    depthEstimator: null,
    occlusion: DEFAULT_OCCLUSION,
    picking: false,
    quality: 'balanced',
    exporting: false,
    sound: false,
    canUndo: false,
    recordUndo: remember,
    beginAdjustment: (pointerId) => {
      gesture?.finish();
      const finish = () => {
        window.removeEventListener('pointerup', finishPointer, true);
        window.removeEventListener('pointercancel', finishPointer, true);
        window.removeEventListener('blur', finish);
        gesture = null;
      };
      const finishPointer = (event: PointerEvent) => {
        if (event.pointerId === pointerId) finish();
      };
      gesture = { finish, recorded: false };
      window.addEventListener('pointerup', finishPointer, true);
      window.addEventListener('pointercancel', finishPointer, true);
      window.addEventListener('blur', finish);
    },
    undo: () => {
      if (get().exporting) return;
      gesture?.finish();
      const restore = history.pop();
      if (!restore) return;
      restoring = true;
      try {
        restore();
      } finally {
        restoring = false;
        set({ canUndo: history.length > 0 });
      }
    },
    setEngine: (engine, status) => {
      // The route/preset can change while the lazy GPU engine is still starting.
      engine?.setPreset(presetById(get().presetId));
      engine?.setTweaks(get().tweaks);
      engine?.setQuality(QUALITIES[get().quality]);
      engine?.onOrbitEdit((restore) => remember(() => {
        if (get().engine === engine) restore();
      }));
      engine?.setFraming(get().framing);
      engine?.setOcclusion(get().occlusion);
      void engine?.setPlate(get().plate);
      void engine?.setDepth(get().depthEstimator);
      set({ engine, status, sound: false });
    },
    setQuality: (quality) => {
      const previous = get().quality;
      if (previous === quality) return;
      remember(() => get().setQuality(previous));
      get().engine?.setQuality(QUALITIES[quality]);
      set({ quality });
    },
    setPreset: (presetId, trackUndo = true) => {
      const previous = get();
      if (presetId === previous.presetId) return;
      if (trackUndo) {
        const boxes = previous.engine?.snapshot().boxes;
        remember(() => {
          applyPreset(previous.presetId, previous.tweaks);
          if (boxes) get().engine?.setBoxes(boxes);
        });
      }
      applyPreset(presetId, { ...defaultTweaks(presetById(presetId)), seed: previous.tweaks.seed, windYaw: previous.tweaks.windYaw });
    },
    setTweak: (key, value) => {
      const { engine, tweaks } = get();
      if (tweaks[key] === value) return;
      remember(() => get().setTweak(key, tweaks[key]));
      engine?.setTweaks({ [key]: value });
      set({ tweaks: { ...tweaks, [key]: value } });
    },
    resetTweaks: () => {
      const { engine, presetId, tweaks: previous } = get();
      const tweaks = defaultTweaks(presetById(presetId));
      if ((Object.keys(tweaks) as (keyof Tweaks)[]).every((key) => tweaks[key] === previous[key])) return;
      remember(() => {
        get().engine?.setTweaks(previous);
        set({ tweaks: previous });
      });
      engine?.setTweaks(tweaks);
      engine?.restart();
      set({ tweaks });
    },
    addCube: () => {
      const { engine } = get();
      if (!engine) return;
      const previous = engine.snapshot().boxes;
      remember(() => get().engine?.setBoxes(previous));
      const half = 3 + Math.random() * 3;
      engine.addBox({
        x: 18 + Math.random() * 28,
        y: 8 + Math.random() * 26,
        z: 22 + Math.random() * 20,
        yaw: Math.random() * Math.PI,
        hx: half,
        hy: half,
        hz: half,
      });
    },
    setBackdrop: (backdrop) => {
      const previous = get().backdrop;
      if (previous === backdrop) return;
      remember(() => get().setBackdrop(previous));
      set({ backdrop });
    },
    setPlate: (plate) => {
      const { engine, plate: old } = get();
      void engine?.setPlate(plate);
      if (old) releasePlate(old);
      set({ plate, picking: false });
    },
    setFraming: (patch) => {
      const previous = get().framing;
      const framing = { ...previous, ...patch };
      if (framing.x === previous.x && framing.y === previous.y && framing.scale === previous.scale) return;
      remember(() => get().setFraming(previous));
      get().engine?.setFraming(framing);
      set({ framing });
    },
    setDepthEstimator: async (depthEstimator) => {
      set({ depthEstimator, picking: false });
      await get().engine?.setDepth(depthEstimator);
    },
    setOcclusion: (patch) => {
      const previous = get().occlusion;
      const occlusion = { ...previous, ...patch };
      if (occlusion.depth === previous.depth && occlusion.softness === previous.softness && occlusion.show === previous.show) return;
      remember(() => get().setOcclusion(previous));
      get().engine?.setOcclusion(occlusion);
      set({ occlusion });
    },
    setPicking: (picking) => set({ picking }),
    setExporting: (exporting) => set({ exporting }),
    toggleSound: () => {
      const sound = !get().sound;
      remember(() => get().toggleSound());
      get().engine?.setSound(sound);
      set({ sound });
    },
  };
});
