'use client';

import { useId, useState, type KeyboardEvent } from 'react';
import { useStudioStore } from './useStudioStore';

export function RestoreButton({ label, defaultUnchanged, onRestore }: {
  label: string;
  defaultUnchanged: boolean;
  onRestore: () => void;
}) {
  const name = `Restore default for ${label}`;
  return (
    <button type="button" aria-label={name} title={name} disabled={defaultUnchanged} onClick={onRestore}
      className="grid size-7 shrink-0 place-items-center rounded border border-line text-muted hover:text-ember focus-visible:outline-ember disabled:opacity-35">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M3.5 10a8.5 8.5 0 1 1 1.8 8.1M3.5 5.5V10H8" />
      </svg>
    </button>
  );
}

type ParameterProps = {
  id?: string;
  label: string;
  help: string;
  value: number;
  defaultValue: number;
  min: number;
  max: number;
  step: number;
  precision?: number;
  showSlider?: boolean;
  unit?: string;
  onChange: (value: number) => void;
};

export default function ParameterControl({ id, label, help, value, defaultValue, min, max, step, precision = 3, showSlider = true, unit, onChange }: ParameterProps) {
  const autoId = useId();
  const controlId = id ?? autoId;
  const helpId = `${controlId}-help`;
  const [draft, setDraft] = useState<string | null>(null);
  const beginAdjustment = useStudioStore((state) => state.beginAdjustment);
  const format = (number: number) => Number(number.toFixed(precision));
  const apply = (number: number) => {
    if (Number.isFinite(number)) onChange(format(Math.min(max, Math.max(min, number))));
  };
  const commit = () => {
    if (draft?.trim()) apply(Number(draft.replace(',', '.')));
    setDraft(null);
  };
  const adjust = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.currentTarget.blur();
      return;
    }
    if (event.key === 'Escape' && draft !== null) {
      event.preventDefault();
      setDraft(null);
      return;
    }
    const direction = event.key === 'ArrowUp' ? 1 : event.key === 'ArrowDown' ? -1 : 0;
    if (!direction) return;
    event.preventDefault();
    const exact = draft?.trim() ? Number(draft.replace(',', '.')) : value;
    const increment = event.altKey ? 10 ** -precision : event.shiftKey ? step * 10 : precision > 0 ? Math.min(step, 0.01) : 1;
    apply((Number.isFinite(exact) ? exact : value) + direction * increment);
    setDraft(null);
  };
  const shortcutHelp = precision > 0 ? ' Use the number field for exact values. Arrow keys adjust in fine steps; hold Option for the smallest step.' : '';
  return (
    <div className="flex flex-col gap-1.5" title={help + shortcutHelp}>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={showSlider ? controlId : `${controlId}-value`} className="min-w-0">{label}</label>
        <div className="flex shrink-0 items-center gap-1">
          <input id={`${controlId}-value`} type="text" inputMode={precision > 0 ? 'decimal' : 'numeric'} aria-label={`${label} value`} aria-describedby={helpId}
            value={draft ?? format(value)} data-undo-text={draft !== null}
            onFocus={(event) => event.currentTarget.select()} onChange={(event) => setDraft(event.target.value)}
            onBlur={commit} onKeyDown={adjust}
            className="w-[4.25rem] rounded border border-line bg-background px-1.5 py-1 text-right font-mono text-xs tabular-nums focus-visible:outline-ember" />
          <span className="w-6 text-right text-[10px] text-muted">{unit}</span>
          <RestoreButton label={label} defaultUnchanged={value === defaultValue} onRestore={() => onChange(defaultValue)} />
        </div>
      </div>
      {showSlider && <input id={controlId} type="range" min={min} max={max} step={step} value={value} aria-describedby={helpId}
        onPointerDown={(event) => { if (event.button === 0) beginAdjustment(event.pointerId); }}
        onChange={(event) => apply(Number(event.target.value))} className="w-full accent-ember" />}
      <span id={helpId} className="sr-only">{help}{shortcutHelp}</span>
    </div>
  );
}
