'use client';

import { useRef, useState } from 'react';
import { openDestination } from '@/export/files';
import { DEFAULT_SETTINGS, type ExportSettings, flipbookTiming, folderName, footageSize, type Progress, runExport, SIZES } from '@/export/run-export';
import { presetById } from '@/presets/presets';
import { chip } from './SimulationPanel';
import { RestoreButton } from './ParameterControl';
import { WebmEncodingUnavailable } from '@/export/webm';
import { useStudioStore } from './useStudioStore';

type Mode = ExportSettings['mode'];
type Row = {
  key: keyof ExportSettings;
  label: string;
  options: [value: string | number, label: string][];
  modes: Mode[];
  advanced?: boolean;
  help: string;
  hidden?: (settings: ExportSettings, footage: boolean, explosion: boolean) => boolean;
};

const MODES: [Mode, string, string][] = [
  ['video', 'Video', 'A playable video. WebM can keep transparency; MP4 includes a separate PNG master.'],
  ['sequence', 'Images', 'Numbered PNG or multilayer EXR frames for compositing and editing.'],
  ['flipbook', 'Flipbook', 'A packed texture sheet for real-time engines, with optional motion vectors.'],
];
const ALL: Mode[] = ['sequence', 'flipbook', 'video'];
const ROWS: Row[] = [
  { key: 'format', label: 'Format', options: [['png', 'PNG'], ['exr', 'EXR · HDR layers']], modes: ['sequence'], help: 'PNG is a lossless display image with alpha. EXR keeps scene-linear HDR fire, smoke, lighting, depth and motion layers.' },
  { key: 'videoFormat', label: 'Format', options: [['webm', 'WebM · VP9 + alpha'], ['mp4', 'MP4 + PNG master']], modes: ['video'], help: 'WebM keeps real video transparency. With footage it creates a matching VP9 composite and clean video. MP4 creates an H.264 review video plus a lossless 16-bit PNG master because H.264 cannot store alpha.' },
  { key: 'footage', label: 'Output', options: [['composite', 'Composite + clean'], ['off', 'Effect only']], modes: ['sequence', 'video'], hidden: (s, footage) => !footage || (s.mode === 'sequence' && s.format === 'exr'), help: 'Composite + clean produces both versions from the same samples. The clean version includes the complete effect on transparency, with no footage masks.' },
  { key: 'size', label: 'Size', options: SIZES.map((size) => [size, size.replace('x', ' × ')]), modes: ['sequence', 'video'], hidden: (_, footage) => footage, help: 'Output pixel dimensions. With footage loaded, both versions use the footage frame, scaled down only when it exceeds 4096 pixels.' },
  { key: 'fps', label: 'Frame rate', options: [24, 25, 30, 60].map((fps) => [fps, `${fps} fps`]), modes: ['sequence', 'video'], help: 'Frames per second. Composite and clean outputs always use the same timestamps and frame count.' },
  { key: 'seconds', label: 'Duration', options: [2, 5, 10, 20].map((seconds) => [seconds, `${seconds} s`]), modes: ALL, help: 'Recorded animation duration. Pre-roll happens before this and is not included in the delivered clip.' },
  { key: 'fidelity', label: 'Detail', options: [['draft', 'Draft'], ['final', 'Final']], modes: ALL, help: 'Draft starts at 128 cells per unit; Final at 256. Adaptive bounds preserve the material envelope within GPU memory. Actual density is recorded in the export.' },
  { key: 'bits', label: 'Precision', options: [[8, '8-bit'], [16, '16-bit']], modes: ['sequence'], hidden: (s) => s.format === 'exr', help: '16-bit PNG preserves smoother gradients for grading. Both PNG versions use this precision; neither is silently reduced to 8-bit.' },
  { key: 'grid', label: 'Sheet grid', options: [['4x4', '4 × 4'], ['8x4', '8 × 4'], ['8x8', '8 × 8']], modes: ['flipbook'], help: 'Columns and rows in the texture sheet. More cells give more animation frames across the selected duration.' },
  { key: 'cell', label: 'Cell size', options: [128, 256, 512].map((px) => [px, `${px} px`]), modes: ['flipbook'], help: 'Pixel width and height of one animation cell. Sheet dimensions equal cell size multiplied by its columns and rows.' },
  { key: 'alpha', label: 'PNG alpha', options: [['straight', 'Straight'], ['premultiplied', 'Premultiplied']], modes: ALL, advanced: true, hidden: (s) => (s.mode === 'sequence' && s.format === 'exr') || (s.mode === 'video' && s.videoFormat !== 'mp4'), help: 'Straight stores unassociated colour for image compositors. Premultiplied stores colour already weighted by alpha. PNG pairs use the same convention; WebM uses straight alpha internally and EXR keeps scene-linear premultiplied layers.' },
  { key: 'samples', label: 'Samples', options: [4, 16, 32, 64].map((n) => [n, `${n} per pixel`]), modes: ALL, advanced: true, help: 'More samples reduce raymarch grain and smooth shutter blur. Both outputs share these exact samples; increasing the value increases GPU render time.' },
  { key: 'shutter', label: 'Motion blur', options: [[0, 'Off'], [0.5, '180° shutter']], modes: ALL, advanced: true, help: 'A 180° shutter samples half a frame interval for natural motion blur. Off freezes each frame at one instant.' },
  { key: 'preroll', label: 'Pre-roll', options: [[0, '0 s · ignition'], [2, '2 s'], [6, '6 s · developed']], modes: ALL, advanced: true, hidden: (_settings, _footage, explosion) => explosion, help: 'Simulated time before the first recorded frame. Shorter pre-roll reduces the wait and starts with a less developed plume.' },
  { key: 'motion', label: 'Motion vectors', options: [['off', 'Off'], ['unreal', 'Unreal'], ['unity', 'Unity']], modes: ['flipbook'], advanced: true, help: 'Writes a companion motion texture for interpolating between cells. Choose the vertical-vector convention expected by Unreal or Unity.' },
];

const clock = (ms: number) => {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
};

function phaseLabel(progress: Progress) {
  if (progress.detail) return progress.detail;
  switch (progress.phase) {
    case 'setup': return 'Preparing GPU';
    case 'warmup': return 'Simulating pre-roll';
    case 'probe': return 'Finding flipbook framing';
    case 'loop': return 'Preparing seamless loop';
    case 'render': return `Rendering frame ${Math.max(1, Math.min(progress.total, Math.ceil(progress.completed)))} / ${progress.total}`;
    case 'encoding': return 'Encoding output';
    case 'saving': return 'Saving output';
  }
}

export default function ExportPanel() {
  const { engine, status, exporting, presetId, tweaks, plate, depthEstimator, setExporting } = useStudioStore();
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [unsupportedWebm, setUnsupportedWebm] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const explosion = presetById(presetId).mode === 'explosion';
  const set = (patch: Partial<ExportSettings>) => {
    if (exporting) return;
    const previous = settings;
    const next = { ...previous, ...patch };
    if (Object.keys(patch).every((key) => previous[key as keyof ExportSettings] === next[key as keyof ExportSettings])) return;
    useStudioStore.getState().recordUndo(() => setSettings(previous));
    setSettings(next);
    setUnsupportedWebm(false);
  };
  const rows = ROWS.filter((row) => row.modes.includes(settings.mode) && !row.hidden?.(settings, Boolean(plate), explosion));
  const sheet = settings.mode === 'flipbook' ? flipbookTiming(settings) : null;
  const composited = Boolean(plate) && !sheet && settings.footage === 'composite' && (settings.mode === 'video' || settings.format !== 'exr');
  const matchedVideo = settings.mode === 'video' && (settings.videoFormat ?? DEFAULT_SETTINGS.videoFormat) === 'webm';
  const outputSummary = settings.mode === 'video'
    ? matchedVideo
      ? composited ? 'Matching VP9 WebM videos: composite + transparent clean.' : 'Transparent VP9 WebM video.'
      : 'H.264 MP4 review video + lossless 16-bit PNG master, with the same frames and timing.'
    : settings.mode === 'sequence'
      ? settings.format === 'exr' ? 'Scene-linear EXR frames with HDR light, depth and motion layers.'
        : `${composited ? 'Matching composite + clean' : 'Transparent'} ${settings.bits}-bit PNG frames · ${settings.alpha} alpha.`
      : 'PNG texture sheet, fitted to the animation from your current orbit.';
  const row = ({ key, label, options, help }: Row) => {
    const id = `export-${key}`;
    return <div key={key} className="grid grid-cols-[5.5rem_minmax(0,1fr)] items-center gap-2" title={help}>
      <label htmlFor={id} className="flex items-center gap-1">{label}<span aria-hidden="true" className="text-[10px] text-muted">?</span></label>
      <div className="flex min-w-0 items-center gap-1">
        <select id={id} value={String(settings[key] ?? DEFAULT_SETTINGS[key])} aria-describedby={`${id}-help`} title={help}
          onChange={(event) => set({ [key]: typeof DEFAULT_SETTINGS[key] === 'number' ? Number(event.target.value) : event.target.value } as Partial<ExportSettings>)}
          className="min-w-0 flex-1 rounded-md border border-line bg-background px-2 py-1 text-foreground">
          {options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
        </select>
        <RestoreButton label={`export ${label}`} defaultUnchanged={(settings[key] ?? DEFAULT_SETTINGS[key]) === DEFAULT_SETTINGS[key]} onRestore={() => set({ [key]: DEFAULT_SETTINGS[key] })} />
      </div>
      <span id={`${id}-help`} className="sr-only">{help}</span>
    </div>;
  };

  async function render() {
    if (!engine || abort.current) return;
    const controller = new AbortController();
    abort.current = controller;
    setMessage(null);
    setCancelling(false);
    setUnsupportedWebm(false);
    try {
      // The picker needs the click's user activation, so nothing may await before it.
      const destination = await openDestination(folderName(presetId, tweaks.seed));
      if (!destination) return;
      setExporting(true);
      await runExport(engine, settings, destination, setProgress, controller.signal);
      setMessage(`Saved ${destination.name}`);
    } catch (error) {
      setUnsupportedWebm(error instanceof WebmEncodingUnavailable);
      setMessage(controller.signal.aborted ? 'Cancelled' : error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
      setProgress(null);
      setCancelling(false);
      if (abort.current === controller) abort.current = null;
    }
  }

  return (
    <fieldset disabled={status !== 'ready'} className="flex flex-col gap-4 text-sm">
      <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Export</legend>
      <fieldset disabled={exporting} className="flex flex-wrap items-center gap-1">
        <div className="flex flex-wrap gap-1 rounded-full border border-line p-1">
        {MODES.map(([mode, label, help]) => (
          <button
            key={mode}
            type="button"
            disabled={exporting}
            onClick={() => set({ mode, loop: mode === 'flipbook' })}
            aria-pressed={mode === settings.mode}
            title={help}
            className={chip(mode === settings.mode)}
          >
            {label}
          </button>
        ))}
        </div>
        <RestoreButton label="export mode" defaultUnchanged={settings.mode === DEFAULT_SETTINGS.mode} onRestore={() => set({ mode: DEFAULT_SETTINGS.mode })} />
      </fieldset>
      <fieldset disabled={exporting} className="flex flex-col gap-3">
        <legend className="mb-2 text-xs uppercase tracking-wider text-muted">Basic</legend>
        {rows.filter((item) => !item.advanced).map(row)}
        {plate && !sheet && <p className="font-mono text-xs tabular-nums text-muted" title="Both deliverables match the footage dimensions. Frames larger than 4096 pixels are scaled down evenly.">{footageSize(plate).join(' × ')} · footage size</p>}
        {plate && depthEstimator && settings.mode === 'sequence' && settings.format === 'exr' && (
          <p className="text-xs text-muted" title="EXR contains the complete effect with HDR emission and alpha, for compositing in your editor.">EXR contains the complete effect on transparency.</p>
        )}
        {sheet && (
          <p className="font-mono text-xs tabular-nums text-muted" title="Cells are the delivered animation frames. Playback rate follows how the selected duration is distributed across the sheet.">
            {sheet.cells} cells · {sheet.seconds.toFixed(1)} s · {+sheet.fps.toFixed(2)} fps playback
          </p>
        )}
      </fieldset>
      <details className="border-t border-line pt-3">
        <summary className="cursor-pointer text-xs text-muted hover:text-foreground" title="Optional alpha, sampling, motion and pre-roll controls. The current defaults are ready to render.">Advanced</summary>
        <fieldset disabled={exporting} className="mt-3 flex flex-col gap-3">
          {rows.filter((item) => item.advanced).map(row)}
          <div className="flex items-center justify-between gap-2" title={explosion ? 'Explosions are single events and cannot use the seamless continuous loop.' : 'Blends the final gas fields toward the start. This adds loop preparation time.'}>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={settings.loop && !explosion} disabled={explosion} onChange={(event) => set({ loop: event.target.checked })} className="accent-ember" />
              Seamless loop
            </label>
            <RestoreButton label="export loop" defaultUnchanged={settings.loop === DEFAULT_SETTINGS.loop} onRestore={() => set({ loop: DEFAULT_SETTINGS.loop })} />
          </div>
        </fieldset>
      </details>
      <p className="text-xs leading-relaxed text-muted" title="Composite and clean versions reuse the same simulation, camera, shutter and volume samples. File encoding is additional work.">{outputSummary}</p>
      <button
        type="button"
        onClick={exporting ? () => { setCancelling(true); abort.current?.abort(); } : render}
        disabled={cancelling}
        title={exporting ? 'Stop after the current GPU or encoder operation and remove incomplete output.' : 'Render with these settings. Progress separates GPU setup, pre-roll, rendering, encoding and saving.'}
        className="self-start rounded-full bg-foreground px-4 py-1.5 font-medium text-background transition-colors hover:bg-ember disabled:opacity-50"
      >
        {cancelling ? 'Stopping…' : exporting ? 'Cancel' : 'Render'}
      </button>
      {progress && (
        <div className="flex flex-col gap-1" role="status" aria-live="polite" title="Progress counts completed work in the current stage. Time estimates appear only after measured progress; they do not include later stages.">
          <progress
            aria-label={phaseLabel(progress)}
            value={progress.total > 0 ? progress.completed : undefined}
            max={progress.total || 1}
            className="h-1.5 w-full appearance-none overflow-hidden rounded-full [&::-webkit-progress-bar]:bg-line [&::-webkit-progress-value]:bg-ember"
          />
          <p className="font-mono text-xs tabular-nums text-muted">
            {cancelling ? 'Stopping after the current operation…' : phaseLabel(progress)}
            {!cancelling && progress.simulatedSeconds !== undefined && ` · ${progress.simulatedSeconds.toFixed(1)} s simulated`}
            {!cancelling && progress.etaMs !== undefined && progress.etaMs > 0 && ` · about ${clock(progress.etaMs)} for this stage`}
          </p>
          <p className="font-mono text-xs tabular-nums text-muted">
            {progress.framesDone} / {progress.framesTotal} frames completed
            {progress.elapsedMs >= 1000 && ` · ${clock(progress.elapsedMs)} in this stage`}
          </p>
        </div>
      )}
      {message && <p className="text-muted">{message}</p>}
      {unsupportedWebm && <button type="button" onClick={() => set({ videoFormat: 'mp4' })} className="self-start rounded border border-line px-3 py-1 text-xs hover:text-ember" title="Choose the compatible H.264 video plus 16-bit PNG master. Your geometry, timing and render detail remain selected.">Use MP4 + PNG master</button>}
      <p className="text-xs text-muted" title="Export time depends on adaptive simulation size, output pixels, samples and encoding. A second alpha video or PNG master adds encoder and storage work.">Use Draft to match the shot, then Final for delivery. Chrome and Edge save to a chosen folder; other browsers download a zip.</p>
    </fieldset>
  );
}
