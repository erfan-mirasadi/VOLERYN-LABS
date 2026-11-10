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

// The user's own video or still behind the fire, and where the fire sits on it. Orbiting still turns the fire.
export default function FootagePanel() {
  const { plate, framing, status, exporting, setPlate, setFraming } = useStudioStore();
  const [error, setError] = useState<string | null>(null);

  async function open(file: File | undefined) {
    if (!file) return;
    setError(null);
    try {
      setPlate(await loadPlate(file));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
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
            <label key={key} className="grid grid-cols-[5rem_1fr_3rem] items-center gap-3">
              <span>{label}</span>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={framing[key]}
                onChange={(event) => setFraming({ [key]: Number(event.target.value) })}
                className="accent-ember"
              />
              <span className="text-right font-mono text-xs tabular-nums text-muted">{framing[key]}</span>
            </label>
          ))}
          <p className="text-xs text-muted">
            {plate.width} × {plate.height}. Drag the viewport to turn the fire to the shot’s angle, then place and size it here.
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
