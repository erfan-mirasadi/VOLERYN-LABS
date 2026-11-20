import Link from 'next/link';

// Six cells, an upward shift: a small simulation grid rather than an element-specific flame mark.
const CELLS: [column: number, row: number, opacity: number][] = [[0, 0, 0.18], [1, 0, 0.38], [0, 1, 0.54], [1, 1, 0.78], [0, 2, 1], [1, 2, 1]];

export function LogoMark({ size = 24 }: { size?: number }) {
  return <svg width={(size * 2) / 3} height={size} viewBox="0 0 20 30" aria-hidden="true" className="shrink-0 text-ember">{CELLS.map(([column, row, opacity]) => <rect key={`${column}${row}`} x={column * 10 + 0.8} y={row * 10 + 0.8} width={8.4} height={8.4} rx={0.5} fill="currentColor" opacity={opacity} />)}</svg>;
}

export default function Logo({ href = '/' }: { href?: string }) {
  return <Link href={href} prefetch={false} aria-label="Voleryn Labs home" className="voleryn-wordmark"><LogoMark size={26} /><span className="voleryn-wordmark-name">Voleryn</span><span className="voleryn-wordmark-labs">LABS</span></Link>;
}
