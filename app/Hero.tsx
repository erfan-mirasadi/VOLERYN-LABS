import Link from 'next/link';
import Chamber from './Chamber';

export default function Hero() {
  return (
    <section className="grid gap-12 py-12 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end lg:gap-20 lg:py-20">
      <div className="flex max-w-2xl flex-col gap-8">
        <h1 className="text-balance text-[clamp(3rem,7.5vw,6.5rem)] font-medium leading-[0.95] tracking-[-0.035em]">Fire, simulated.</h1>
        <p className="max-w-lg text-lg leading-relaxed text-muted">
          Voleryn Labs runs a gas simulation on your GPU and renders fire, smoke and explosions in real time, in the browser. Export sequences, EXR
          layers, flipbooks and video with alpha, for compositing and game engines.
        </p>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-4">
          <Link
            href="/studio"
            className="inline-flex h-11 items-center rounded-md bg-ember px-5 text-sm font-medium text-black transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ember"
          >
            Open the studio
          </Link>
          <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">WebGPU · Chrome, Edge or Safari 26</span>
        </div>
      </div>
      {/* On desktop the frame shrinks with the viewport's height so the base of the fire stays above the fold. */}
      <figure className="flex w-full max-w-[32rem] flex-col gap-3 lg:w-[min(32rem,max(20rem,calc((100vh_-_12rem)*5/6)))]">
        <Chamber />
        <figcaption className="flex justify-between font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          <span>Live simulation</span>
          <span>Drag to orbit</span>
        </figcaption>
      </figure>
    </section>
  );
}
