'use client';

import { useRef, useState } from 'react';
import { openDestination } from '@/export/files';
import { DEFAULT_SETTINGS, type ExportSettings, folderName, type Progress, runExport, SIZES } from '@/export/run-export';
import { presetById } from '@/presets/presets';
import { chip } from './SimulationPanel';
import { useStudioStore } from './useStudioStore';

type Mode = ExportSettings['mode'];
type Row = {
  key: keyof ExportSettings;
  label: string;
  options: [value: string | number, label: string][];
  modes: Mode[];
  hidden?: (settings: ExportSettings) => boolean;
};

const MODES: [Mode, string][] = [['sequence', 'Sequence'], ['flipbook', 'Flipbook'], ['video', 'Video']];
const ALL: Mode[] = ['sequence', 'flipbook', 'video'];
const ROWS: Row[] = [
  { key: 'format', label: 'Format', options: [['png', 'PNG'], ['exr', 'EXR']], modes: ['sequence'] },
  { key: 'alpha', label: 'Alpha', options: [['premultiplied', 'Premultiplied'], ['straight', 'Straight']], modes: ['sequence', 'flipbook'], hidden: (s) => s.format === 'exr' },
  { key: 'bits', label: 'Bits', options: [[8, '8-bit'], [16, '16-bit']], modes: ['sequence'], hidden: (s) => s.format === 'exr' },
  { key: 'size', label: 'Size', options: SIZES.map((size) => [size, size.replace('x', ' × ')]), modes: ['sequence', 'video'] },
  { key: 'fps', label: 'Frame rate', options: [24, 25, 30, 60].map((fps) => [fps, `${fps} fps`]), modes: ['sequence', 'video'] },
  { key: 'seconds', label: 'Length', options: [2, 5, 10, 20].map((seconds) => [seconds, `${seconds} s`]), modes: ALL },
  { key: 'grid', label: 'Grid', options: [['4x4', '4 × 4'], ['8x4', '8 × 4'], ['8x8', '8 × 8']], modes: ['flipbook'] },
  { key: 'cell', label: 'Cell', options: [128, 256, 512].map((px) => [px, `${px} px`]), modes: ['flipbook'] },
  { key: 'motion', label: 'Motion vectors', options: [['off', 'Off'], ['unreal', 'Unreal'], ['unity', 'Unity']], modes: ['flipbook'] },
  { key: 'samples', label: 'Quality', options: [[4, 'Draft · 4 samples'], [16, 'Final · 16 samples']], modes: ALL },
];

const clock = (ms: number) => {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};

export default function ExportPanel() {
  const { engine, status, exporting, presetId, tweaks, setExporting } = useStudioStore();
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const explosion = presetById(presetId).mode === 'explosion';
  const set = (patch: Partial<ExportSettings>) => setSettings((current) => ({ ...current, ...patch }));
  const rows = ROWS.filter((row) => row.modes.includes(settings.mode) && !row.hidden?.(settings));

  async function render() {
    if (!engine) return;
    const controller = new AbortController();
    abort.current = controller;
    setMessage(null);
    try {
      // The picker needs the click's user activation, so nothing may await before it.
      const destination = await openDestination(folderName(presetId, tweaks.seed));
      if (!destination) return;
      setExporting(true);
      await runExport(engine, settings, destination, setProgress, controller.signal);
      setMessage(`Saved ${destination.name}`);
    } catch (error) {
      setMessage(controller.signal.aborted ? 'Cancelled' : error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
      setProgress(null);
    }
  }

  return (
    <fieldset disabled={status !== 'ready'} className="flex flex-col gap-4 text-sm">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Export</legend>
      <div className="flex flex-wrap gap-1 self-start rounded-full border border-line p-1">
        {MODES.map(([mode, label]) => (
          <button
            key={mode}
            type="button"
            disabled={exporting}
            onClick={() => set({ mode, loop: mode === 'flipbook' })}
            aria-pressed={mode === settings.mode}
            className={chip(mode === settings.mode)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-[6rem_1fr] items-center gap-x-3 gap-y-2">
        {rows.map(({ key, label, options }) => (
          <label key={key} className="contents">
            <span>{label}</span>
            <select
              value={String(settings[key])}
              disabled={exporting}
              onChange={(event) => set({ [key]: typeof settings[key] === 'number' ? Number(event.target.value) : event.target.value } as Partial<ExportSettings>)}
              className="min-w-0 rounded-md border border-line bg-background px-2 py-1 text-foreground"
            >
              {options.map(([value, text]) => (
                <option key={value} value={value}>
                  {text}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={settings.loop && !explosion} disabled={exporting || explosion} onChange={(event) => set({ loop: event.target.checked })} className="accent-ember" />
        Seamless loop
      </label>
      <button
        type="button"
        onClick={exporting ? () => abort.current?.abort() : render}
        className="self-start rounded-full bg-foreground px-4 py-1.5 font-medium text-background transition-colors hover:bg-ember disabled:opacity-50"
      >
        {exporting ? 'Cancel' : 'Render'}
      </button>
      {progress && (
        <div className="flex flex-col gap-1">
          <progress
            value={progress.done}
            max={progress.total}
            className="h-1.5 w-full appearance-none overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-line [&::-webkit-progress-value]:bg-ember"
          />
          <p className="font-mono text-xs tabular-nums text-muted">
            {progress.done === 0 ? 'Warming up' : `Frame ${progress.done} / ${progress.total}`}
            {progress.msPerFrame > 0 && ` · ${(progress.msPerFrame / 1000).toFixed(2)} s per frame · about ${clock((progress.total - progress.done) * progress.msPerFrame)} left`}
          </p>
        </div>
      )}
      {message && <p className="text-muted">{message}</p>}
      <p className="text-xs text-muted">Chrome and Edge write into a folder you pick; other browsers download a zip.</p>
    </fieldset>
  );
}
