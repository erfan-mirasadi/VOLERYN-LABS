'use client';

import { useState } from 'react';
import type { Framing } from '@/engine/engine';
import { loadPlate } from '@/engine/plate';
import ParameterControl from './ParameterControl';
import { CENTRED, DEFAULT_OCCLUSION, useStudioStore } from './useStudioStore';

const FRAMING = [
  { key: 'x', label: 'Position X', min: -1, max: 1, step: 0.005, help: 'Move the effect horizontally in the frame. Positive values move it right.' },
  { key: 'y', label: 'Position Y', min: -1, max: 1, step: 0.005, help: 'Move the effect vertically in the frame. Positive values move it up.' },
  { key: 'scale', label: 'Scale', min: 0.2, max: 3, step: 0.01, help: 'Scale the effect in the camera frame. The simulation itself keeps its size.' },
] as const satisfies readonly { key: keyof Framing; label: string; min: number; max: number; step: number; help: string }[];

export function PlacementPanel() {
  const { framing, status, exporting, setFraming } = useStudioStore();
  return (
    <details className="border-b border-line pb-3">
      <summary className="cursor-pointer text-sm" title="Move or scale the effect within the rendered frame">
        Placement <span className="ml-1 font-mono text-xs tabular-nums text-muted">{framing.scale.toFixed(2)}×</span>
      </summary>
      <fieldset disabled={status !== 'ready' || exporting} className="mt-3 flex flex-col gap-3 text-sm" aria-label="Placement controls">
        {FRAMING.map(({ key, ...control }) => (
          <ParameterControl key={key} {...control} value={framing[key]} defaultValue={CENTRED[key]}
            unit={key === 'scale' ? '×' : undefined} onChange={(value) => setFraming({ [key]: value })} />
        ))}
      </fieldset>
    </details>
  );
}

const PILL = 'rounded-full border border-line px-3 py-1 text-muted hover:text-foreground aria-pressed:border-ember aria-pressed:text-ember';

// The user's own video or still behind the fire, where the fire sits on it, and how deep into the shot it stands.
export default function FootagePanel() {
  const { plate, depthEstimator, occlusion, picking, status, exporting, setPlate, setDepthEstimator, setOcclusion, setPicking } = useStudioStore();
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
              <ParameterControl label="Depth" help="Place the effect behind parts of the footage that are closer to the camera."
                min={0} max={1} step={0.005} value={occlusion.depth} defaultValue={DEFAULT_OCCLUSION.depth}
                onChange={(value) => setOcclusion({ depth: value })} />
              <ParameterControl label="Softness" help="Blend the edges of the foreground mask around the effect."
                min={0} max={1} step={0.01} value={occlusion.softness} defaultValue={DEFAULT_OCCLUSION.softness}
                onChange={(value) => setOcclusion({ softness: value })} />
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
              ? 'The fire goes behind whatever is nearer than its depth. Pick a spot where it should stand, or slide it back until the foreground closes over it. A playing video’s depth is read at half detail, a few frames behind; a held frame, a still and every frame of a render at full detail.'
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
