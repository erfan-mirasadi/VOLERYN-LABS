import { PRESETS } from '@/presets/presets';
import { SLIDERS } from '@/studio/controls';

const fire = PRESETS[0];
// Where each slider starts for the Fire preset, as the studio sets it.
const START = { heat: 1, vorticity: fire.vorticity, wind: 0, seed: 0, blue: fire.blue, sparks: fire.sparks, smoke: 1, width: 1 };
const ALSO: [term: string, detail: string][] = [
  ['Wind direction', 'An arrow on the floor of the box; drag its head to turn the wind.'],
  ['Sound', 'Synthesised from the same numbers: a rumble and roar from the heat, flutter from the vorticity, hiss and crackle from the flow, placed in 3D around the camera.'],
  ['Backdrop', 'Checker, dark or light, to judge the alpha before exporting.'],
];

const signed = (value: number) => String(value).replace('-', '−');

export default function Controls() {
  return (
    <div className="flex flex-col gap-12">
      <ul className="border-t border-line">
        {SLIDERS.map(({ key, label, min, max }) => (
          <li key={key} className="grid grid-cols-[6rem_2.25rem_1fr_2.25rem] items-center gap-3 border-b border-line py-4 text-sm sm:grid-cols-[7rem_2.5rem_1fr_2.5rem] sm:gap-4">
            <span>{label}</span>
            <span className="font-mono text-xs tabular-nums text-muted">{signed(min)}</span>
            <span className="relative h-px bg-line">
              <span className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ember" style={{ left: `${((START[key] - min) / (max - min)) * 100}%` }} />
            </span>
            <span className="text-right font-mono text-xs tabular-nums text-muted">{signed(max)}</span>
          </li>
        ))}
      </ul>
      <dl className="flex flex-col gap-5">
        {ALSO.map(([term, detail]) => (
          <div key={term} className="grid gap-1 text-sm sm:grid-cols-[8.5rem_1fr] sm:gap-4">
            <dt className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted sm:pt-1">{term}</dt>
            <dd className="leading-relaxed text-muted">{detail}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
