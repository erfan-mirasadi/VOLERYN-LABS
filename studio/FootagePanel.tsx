'use client';

import { useState } from 'react';
import type { Framing } from '@/engine/engine';
import { loadPlate } from '@/engine/plate';
import { useStudioStore } from './useStudioStore';

const FRAMING = [
  { key: 'x', label: 'Position X', min: -1, max: 1, step: 0.005 },
  { key: 'y', label: 'Position Y', min: -1, max: 1, step: 0.005 },
  { key: 'scale', label: 'Scale', min: 0.2, max: 3, step: 0.01 },
] as const satisfies readonly { key: keyof Framing; label: string; min: number; max: number; step: number }[];

const SLIDER_ROW = 'grid grid-cols-[5rem_1fr_3rem] items-center gap-3';
const VALUE = 'text-right font-mono text-xs tabular-nums text-muted';
const PILL = 'rounded-full border border-line px-3 py-1 text-muted hover:text-foreground aria-pressed:border-ember aria-pressed:text-ember';

// The user's own video or still behind the fire, where the fire sits on it, and how deep into the shot it stands.
export default function FootagePanel() {
  const { plate, framing, depthEstimator, occlusion, picking, status, exporting, setPlate, setFraming, setDepthEstimator, setOcclusion, setPicking } = useStudioStore();
  const [error, setError] = useState<string | null>(null);
  // The share of the depth model fetched so far while it is on its way; null otherwise.
  const [fetched, setFetched] = useState<number | null>(null);

  const report = (failure: unknown) => setError(failure instanceof Error ? failure.message : String(failure));

  async function open(file: File | undefined) {
    if (!file) return;
    setError(null);
    try {
      setPlate(await loadPlate(file));
    } catch (failure) {
      report(failure);
    }
  }

  // The estimator, its runtime and its model are fetched here, on the user's word, and nowhere else.
  async function turnDepth(on: boolean) {
    setError(null);
    if (!on) return setDepthEstimator(null);
    setFetched(0);
    try {
      const { loadDepthEstimator } = await import('@/engine/depth');
      await setDepthEstimator(await loadDepthEstimator(setFetched));
    } catch (failure) {
      await setDepthEstimator(null);
      report(failure);
    } finally {
      setFetched(null);
    }
  }

  return (
    <fieldset disabled={status !== 'ready' || exporting} className="flex flex-col gap-4 text-sm">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Footage</legend>
      {plate ? (
        <>
          <div className="flex items-baseline justify-between gap-3">
            <p className="min-w-0 truncate">{plate.name}</p>
            <button type="button" onClick={() => setPlate(null)} className="text-muted hover:text-foreground">
              Remove
            </button>
          </div>
          {FRAMING.map(({ key, label, min, max, step }) => (
            <label key={key} className={SLIDER_ROW}>
              <span>{label}</span>
              <input type="range" min={min} max={max} step={step} value={framing[key]} onChange={(event) => setFraming({ [key]: Number(event.target.value) })} className="accent-ember" />
              <span className={VALUE}>{framing[key]}</span>
            </label>
          ))}
          <p className="text-xs text-muted">
            {plate.width} × {plate.height}. Drag the fire to place it; near the middle it snaps back. Drag around it to turn it to the shot’s angle.
          </p>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={Boolean(depthEstimator) || fetched !== null} disabled={fetched !== null} onChange={(event) => void turnDepth(event.target.checked)} className="accent-ember" />
            Behind the shot’s foreground
          </label>
          {fetched !== null && (
            <div className="flex flex-col gap-1">
              <progress value={fetched} max={1} className="h-1.5 w-full appearance-none overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-line [&::-webkit-progress-value]:bg-ember" />
              <p className="font-mono text-xs tabular-nums text-muted">{fetched < 1 ? `Getting the depth model · ${Math.round(fetched * 100)}%` : 'Reading the shot’s depth'}</p>
            </div>
          )}
          {depthEstimator && fetched === null && (
            <>
              <label className={SLIDER_ROW}>
                <span>Depth</span>
                <input type="range" min={0} max={1} step={0.005} value={occlusion.depth} onChange={(event) => setOcclusion({ depth: Number(event.target.value) })} className="accent-ember" />
                <span className={VALUE}>{occlusion.depth}</span>
              </label>
              <div className="flex flex-wrap gap-2">
                <button type="button" aria-pressed={picking} onClick={() => setPicking(!picking)} className={PILL}>
                  {picking ? 'Click a spot in the shot' : 'Pick depth in the shot'}
                </button>
                <button type="button" aria-pressed={occlusion.show} onClick={() => setOcclusion({ show: !occlusion.show })} className={PILL}>
                  Show the matte
                </button>
              </div>
            </>
          )}
          <p className="text-xs text-muted">
            {depthEstimator
              ? 'The fire goes behind whatever is nearer than its depth. Pick a spot where it should stand, or slide it back until the foreground closes over it. A playing video’s depth is read live at half detail, a few frames behind; a still, and every frame of a render, at full detail.'
              : 'Reads the shot’s depth on this device, so the fire can stand behind a wall or a person. The first use fetches a 50 MB model; the footage itself never leaves the device.'}
          </p>
        </>
      ) : (
        <>
          <label className="cursor-pointer self-start rounded-full border border-line px-3 py-1 text-muted focus-within:text-foreground hover:text-foreground">
            Add footage
            <input
              type="file"
              accept="video/*,image/*"
              className="sr-only"
              onChange={(event) => {
                void open(event.target.files?.[0]);
                event.target.value = '';
              }}
            />
          </label>
          <p className="text-xs text-muted">A video or a still of your own to put the fire on. It stays on this device.</p>
        </>
      )}
      {error && (
        <p role="alert" className="text-muted">
          {error}
        </p>
      )}
    </fieldset>
  );
}
