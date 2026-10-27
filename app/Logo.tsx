import Link from 'next/link';

// The mark is a volume seen from its corner: the two lit faces form the V, the top face stays a wire outline.
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="voleryn-ember" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0" stopColor="#ff7a1a" />
          <stop offset="1" stopColor="#ffb347" />
        </linearGradient>
      </defs>
      <path d="M16 16 L7 11 L7 21 L16 26 Z" fill="url(#voleryn-ember)" />
      <path d="M16 16 L25 11 L25 21 L16 26 Z" fill="url(#voleryn-ember)" opacity="0.55" />
      <path d="M16 6 L25 11 L16 16 L7 11 Z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  );
}

export default function Logo({ href = '/' }: { href?: string }) {
  return (
    <Link href={href} className="flex items-center gap-2.5 text-foreground">
      <LogoMark />
      <span className="text-sm font-semibold uppercase tracking-[0.28em]">
        Voleryn <span className="font-normal text-muted">Labs</span>
      </span>
    </Link>
  );
}
