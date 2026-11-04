'use client';

import { QUALITIES, type QualityId } from '@/engine/scene';
import { PRESETS, presetById } from '@/presets/presets';
import { SLIDERS } from './controls';
import { type Backdrop, useStudioStore } from './useStudioStore';

// What each preview quality spends, in the words of the person choosing it.
const QUALITY_NOTES: Record<QualityId, string> = {
  fast: 'Coarse march, scales down to half size for 60 fps.',
  balanced: 'The base grid, full size at 60 fps where the GPU allows.',
  fine: 'A 96 × 144 × 96 grid and a finer march, aimed at 30 fps. Restarts the fire.',
  ultra: 'A 128 × 192 × 128 grid: eight times the cells, for strong GPUs. Restarts the fire.',
};

const BACKDROPS: { id: Backdrop; label: string }[] = [
  { id: 'checker', label: 'Checker' },
  { id: 'dark', label: 'Dark' },
  { id: 'light', label: 'Light' },
];

export const chip = (active: boolean) =>
  `rounded-full px-3 py-1 text-sm transition-colors ${active ? 'bg-foreground text-background' : 'text-muted hover:text-foreground'}`;

export default function SimulationPanel() {
  const { presetId, tweaks, backdrop, quality, status, exporting, setPreset, setTweak, addCube, setBackdrop, setQuality } = useStudioStore();
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

      <fieldset disabled={exporting} className="flex flex-col gap-2">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Preview</legend>
        <div className="flex flex-wrap gap-1 self-start rounded-full border border-line p-1">
          {(Object.keys(QUALITIES) as QualityId[]).map((id) => (
            <button key={id} type="button" onClick={() => setQuality(id)} aria-pressed={id === quality} className={chip(id === quality)}>
              {id[0].toUpperCase() + id.slice(1)}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted">{QUALITY_NOTES[quality]}</p>
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
