import { type Box, GRID, type Preset } from '@/engine/scene';
import { PRESETS } from '@/presets/presets';

const [WIDTH, HEIGHT] = GRID;
// Every sixteenth cell line of the box, as faint graph paper behind the setup.
const RULINGS = [...[16, 32, 48].map((x) => `M${x} 0V${HEIGHT}`), ...[16, 32, 48, 64, 80].map((y) => `M0 ${y}H${WIDTH}`)].join('');
const LABEL = 'font-mono text-[11px] uppercase tracking-[0.14em] text-muted';

// A box's front face as it turns about y: both horizontal half-sizes project onto the view.
const faceWidth = (box: Box) => 2 * (box.hx * Math.abs(Math.cos(box.yaw)) + box.hz * Math.abs(Math.sin(box.yaw)));

const setupOf = (preset: Preset) => {
  switch (preset.mode) {
    case 'objects':
      return `${preset.objects?.length ?? 0} burning objects`;
    case 'smoke':
      return `Smoke source, radius ${preset.radius}`;
    case 'explosion':
      return `Charge, blast ${preset.blast}`;
    default:
      return `Fuel kernel, radius ${preset.radius}`;
  }
};

// The preset's setup drawn to scale in the simulation box, seen from the front.
function Setup({ preset }: { preset: Preset }) {
  const [x, y] = preset.emitter;
  const clip = `setup-${preset.id}`;
  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} aria-hidden="true" className={`h-auto w-full ${preset.mode === 'smoke' ? 'text-foreground' : 'text-ember'}`}>
      <defs>
        <clipPath id={clip}>
          <rect width={WIDTH} height={HEIGHT} />
        </clipPath>
      </defs>
      <g stroke="currentColor" className="text-muted" vectorEffect="non-scaling-stroke">
        <path d={RULINGS} strokeOpacity="0.12" vectorEffect="non-scaling-stroke" />
        <rect x="0.5" y="0.5" width={WIDTH - 1} height={HEIGHT - 1} fill="none" strokeOpacity="0.45" vectorEffect="non-scaling-stroke" />
      </g>
      <g clipPath={`url(#${clip})`} fill="currentColor" fillOpacity="0.3" stroke="currentColor" className="transition-[fill-opacity] duration-300 group-hover:[fill-opacity:0.55]">
        {preset.mode === 'objects' ? (
          preset.objects?.map((box, i) => (
            <rect key={i} x={box.x - faceWidth(box) / 2} y={HEIGHT - box.y - box.hy} width={faceWidth(box)} height={2 * box.hy} vectorEffect="non-scaling-stroke" />
          ))
        ) : (
          <circle cx={x} cy={HEIGHT - y} r={preset.radius} vectorEffect="non-scaling-stroke" />
        )}
        {preset.mode === 'explosion' && (
          <circle cx={x} cy={HEIGHT - y} r={preset.radius * 3} fill="none" strokeOpacity="0.5" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
        )}
      </g>
    </svg>
  );
}

export default function Presets() {
  return (
    <ul className="grid grid-cols-2 gap-x-6 gap-y-12 sm:grid-cols-4 sm:gap-x-8">
      {PRESETS.map((preset) => (
        <li key={preset.id} className="group flex flex-col gap-4">
          <Setup preset={preset} />
          <div className="flex flex-col gap-1">
            <h3 className="font-medium">{preset.name}</h3>
            <p className={LABEL}>{setupOf(preset)}</p>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 border-t border-line pt-3 font-mono text-xs tabular-nums text-muted">
            {(
              [
                ['buoyancy', preset.buoyancy],
                ['burn', preset.burn],
                ['cooling', preset.cooling],
                ['vorticity', preset.vorticity],
              ] as const
            ).map(([name, value]) => (
              <div key={name} className="contents">
                <dt>{name}</dt>
                <dd className="text-right text-foreground">{value}</dd>
              </div>
            ))}
          </dl>
        </li>
      ))}
    </ul>
  );
}
