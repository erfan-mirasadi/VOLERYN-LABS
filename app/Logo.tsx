import Link from 'next/link';

// The simulation box seen head on: six cells lit from the floor, fading as the heat rises and leaning a little with the draught.
const CELLS: [column: number, row: number, opacity: number][] = [
  [0, 0, 0.12],
  [1, 0, 0.22],
  [0, 1, 0.42],
  [1, 1, 0.6],
  [0, 2, 1],
  [1, 2, 0.86],
];

export function LogoMark({ size = 24 }: { size?: number }) {
  return (
    <svg width={(size * 2) / 3} height={size} viewBox="0 0 20 30" aria-hidden="true" className="shrink-0 text-ember">
      {CELLS.map(([column, row, opacity]) => (
        <rect key={`${column}${row}`} x={column * 10 + 0.8} y={row * 10 + 0.8} width={8.4} height={8.4} rx={1} fill="currentColor" opacity={opacity} />
      ))}
    </svg>
  );
}

export default function Logo({ href = '/' }: { href?: string }) {
  return (
    <Link href={href} className="flex items-center gap-2.5 text-foreground">
      <LogoMark size={22} />
      <span className="text-base font-medium tracking-[-0.01em]">Voleryn</span>
      <span className="rounded-[3px] border border-line px-1.5 py-[3px] font-mono text-[9px] uppercase leading-none tracking-[0.18em] text-muted">Labs</span>
    </Link>
  );
}
