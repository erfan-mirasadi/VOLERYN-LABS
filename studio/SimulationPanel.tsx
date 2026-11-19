'use client';

import { useState } from 'react';
import { QUALITIES, type QualityId } from '@/engine/scene';
import { PRESETS, presetById } from '@/presets/presets';
import { clampTweak, SLIDERS, type ControlGroup, type Slider } from './controls';
import { type Backdrop, useStudioStore } from './useStudioStore';

const QUALITY_NOTES: Record<QualityId, string> = {
  fast: 'Fastest preview, with a coarser image.',
  balanced: 'Standard preview. Final renders use a finer simulation.',
  fine: 'Finer simulation, aimed at 30 fps. Restarts the fire.',
  ultra: 'Highest preview detail, for strong GPUs. Restarts the fire.',
};
const GROUPS: { id: ControlGroup; label: string; note: string }[] = [
  { id: 'source', label: 'Source & combustion', note: 'Changes affect newly emitted gas. Replay to compare from ignition.' },
  { id: 'motion', label: 'Motion & wind', note: 'Wind moves gas through the scene. Fine controls shape the flow.' },
  { id: 'shading', label: 'Match the shot', note: 'Adjust the rendered look without changing the simulation.' },
  { id: 'sparks', label: 'Sparks', note: 'Incandescent particles with varied size, drag and cooling.' },
];
const BACKDROPS: { id: Backdrop; label: string }[] = [{ id: 'checker', label: 'Checker' }, { id: 'dark', label: 'Dark' }, { id: 'light', label: 'Light' }];

export const chip = (active: boolean) =>
  `rounded-full px-3 py-1 text-sm transition-colors ${active ? 'bg-foreground text-background' : 'text-muted hover:text-foreground'}`;

function NumericValue({ slider, value, set }: { slider: Slider; value: number; set: (value: number) => void }) {
  // Keep a local edit so entering 0.25 does not clamp its first character mid-keystroke.
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    const next = draft === null || draft.trim() === '' ? null : clampTweak(slider.key, Number(draft));
    if (next !== null) set(next);
    setDraft(null);
  };
  return <input
    type="number" aria-label={`${slider.label} value`} min={slider.min} max={slider.max} step={slider.step}
    value={draft ?? value} onChange={(event) => setDraft(event.target.value)} onBlur={commit}
    onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); if (event.key === 'Escape') { event.preventDefault(); setDraft(null); } }}
    className="w-[4.5rem] rounded border border-line bg-background px-1.5 py-1 text-right font-mono text-xs tabular-nums"
  />;
}

export default function SimulationPanel() {
  const { engine, presetId, tweaks, backdrop, quality, status, exporting, setPreset, setTweak, resetTweaks, addCube, setBackdrop, setQuality } = useStudioStore();
  const disabled = status !== 'ready' || exporting;
  const preset = presetById(presetId);
  const row = (slider: Slider) => {
    const { key, label, min, max, step, help, unit } = slider;
    return <div key={key} className="flex flex-col gap-1.5" title={help}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={`control-${key}`}>{label}</label>
        <div className="flex items-center gap-1"><NumericValue slider={slider} value={tweaks[key]} set={(value) => setTweak(key, value)} /><span className="w-7 text-right text-[10px] text-muted">{unit}</span></div>
      </div>
      <input id={`control-${key}`} type="range" min={min} max={max} step={step} value={tweaks[key]} aria-describedby={`help-${key}`} onChange={(event) => setTweak(key, Number(event.target.value))} className="w-full accent-ember" />
      <span id={`help-${key}`} className="sr-only">{help}</span>
    </div>;
  };

  return <div className="flex flex-col gap-6 text-sm">
    <fieldset disabled={disabled} className="flex flex-col gap-3">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Preset</legend>
      <div className="flex flex-wrap gap-1 rounded-full border border-line p-1">
        {PRESETS.map((item) => <button key={item.id} type="button" onClick={() => setPreset(item.id)} aria-pressed={item.id === presetId} className={chip(item.id === presetId)}>{item.name}</button>)}
      </div>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={() => engine?.restart()} className="rounded-full border border-line px-3 py-1 hover:text-ember">Replay</button>
        <button type="button" onClick={resetTweaks} className="rounded-full border border-line px-3 py-1 text-muted hover:text-foreground">Reset controls</button>
        {preset.mode === 'objects' && <button type="button" onClick={addCube} className="rounded-full border border-line px-3 py-1 text-muted hover:text-foreground">Add cube</button>}
      </div>
    </fieldset>
    {GROUPS.map(({ id, label, note }) => {
      const sliders = SLIDERS.filter((slider) => slider.group === id && (!slider.explosion || preset.mode === 'explosion'));
      const advanced = sliders.filter((slider) => slider.advanced);
      return <fieldset key={id} disabled={disabled} className="flex flex-col gap-3">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">{label}</legend>
        {sliders.filter((slider) => !slider.advanced).map(row)}
        {advanced.length > 0 && <details className="border-t border-line pt-2">
          <summary className="cursor-pointer text-xs text-muted hover:text-foreground">Fine controls</summary>
          <div className="mt-3 flex flex-col gap-3">{advanced.map(row)}</div>
        </details>}
        <p className="text-xs leading-relaxed text-muted">{note}</p>
      </fieldset>;
    })}
    <fieldset disabled={disabled} className="flex flex-col gap-2">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Preview</legend>
      <div className="flex flex-wrap gap-1 self-start rounded-full border border-line p-1">
        {(Object.keys(QUALITIES) as QualityId[]).map((id) => <button key={id} type="button" onClick={() => setQuality(id)} aria-pressed={id === quality} className={chip(id === quality)}>{id[0].toUpperCase() + id.slice(1)}</button>)}
      </div>
      <p className="text-xs text-muted">{QUALITY_NOTES[quality]}</p>
    </fieldset>
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Backdrop</legend>
      <div className="flex gap-1 self-start rounded-full border border-line p-1">
        {BACKDROPS.map((item) => <button key={item.id} type="button" onClick={() => setBackdrop(item.id)} aria-pressed={item.id === backdrop} className={chip(item.id === backdrop)}>{item.label}</button>)}
      </div>
    </fieldset>
  </div>;
}
