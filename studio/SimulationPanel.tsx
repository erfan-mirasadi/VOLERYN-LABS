'use client';

import { PRESETS, presetById } from '@/presets/presets';
import { SLIDERS } from './controls';
import { type Backdrop, useStudioStore } from './useStudioStore';

const BACKDROPS: { id: Backdrop; label: string }[] = [
  { id: 'checker', label: 'Checker' },
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' },
];

export const chip = (active: boolean) =>
  `rounded-full px-3 py-1 text-sm transition-colors ${active ? 'bg-foreground text-background' : 'text-muted hover:text-foreground'}`;

export default function SimulationPanel() {
  const { presetId, tweaks, backdrop, status, exporting, setPreset, setTweak, addCube, setBackdrop } = useStudioStore();
  const disabled = status !== 'ready' || exporting;
  const preset = presetById(presetId);

  return (
    <div className="flex flex-col gap-6 text-sm">
      <fieldset disabled={disabled} className="flex flex-col gap-2">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Preset</legend>
        <div className="flex flex-wrap gap-1 rounded-full border border-line p-1">
          {PRESETS.map((item) => (
            <button key={item.id} type="button" onClick={() => setPreset(item.id)} aria-pressed={item.id === presetId} className={chip(item.id === presetId)}>
              {item.name}
            </button>
          ))}
        </div>
        {preset.mode === 'objects' && (
          <button type="button" onClick={addCube} className="self-start rounded-full border border-line px-3 py-1 text-muted hover:text-foreground">
            Add cube
          </button>
        )}
      </fieldset>

      <fieldset disabled={disabled} className="flex flex-col gap-4">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Simulation</legend>
        {SLIDERS.map(({ key, label, min, max, step }) => (
          <label key={key} className="grid grid-cols-[5rem_1fr_3rem] items-center gap-3">
            <span>{label}</span>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={tweaks[key]}
              onChange={(event) => setTweak(key, Number(event.target.value))}
              className="accent-ember"
            />
            <span className="text-right font-mono text-xs tabular-nums text-muted">{tweaks[key]}</span>
          </label>
        ))}
      </fieldset>

      <fieldset className="flex flex-col gap-2">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Backdrop</legend>
        <div className="flex gap-1 rounded-full border border-line p-1 self-start">
          {BACKDROPS.map((item) => (
            <button key={item.id} type="button" onClick={() => setBackdrop(item.id)} aria-pressed={item.id === backdrop} className={chip(item.id === backdrop)}>
              {item.label}
            </button>
          ))}
        </div>
      </fieldset>
    </div>
  );
}
