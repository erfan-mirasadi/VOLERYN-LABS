import Link from 'next/link';
import { GRID } from '@/engine/scene';
import { PRESETS } from '@/presets/presets';
import HeroDemo from './HeroDemo';

export default function Hero() {
  return (
    <section className="grid gap-12 py-10 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center lg:gap-16 lg:py-16">
      <div className="flex max-w-xl flex-col gap-6">
        <h1 className="text-balance text-4xl font-semibold leading-[1.02] tracking-tight sm:text-5xl xl:text-6xl">
          Real-time fire and smoke in your browser.
        </h1>
        <p className="text-lg leading-relaxed text-muted">
          Voleryn Labs simulates fire, smoke and explosions on your GPU and exports them with alpha, as PNG or EXR sequences, flipbooks and
          MP4, for compositing and game engines.
        </p>
        <Link
          href="/studio"
          className="mt-2 inline-flex h-11 items-center self-start rounded-full bg-ember px-6 text-sm font-medium text-black transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ember"
        >
          Open the studio
        </Link>
      </div>
      {/* On desktop the frame shrinks with the viewport's height so the base of the fire stays above the fold. */}
      <figure className="flex w-full max-w-[34rem] flex-col gap-3 lg:w-[min(30rem,max(20rem,calc((100vh_-_11rem)*5/6)))] xl:w-[min(34rem,max(20rem,calc((100vh_-_11rem)*5/6)))]">
        <div className="overflow-hidden rounded-xl border border-line bg-black">
          <HeroDemo />
        </div>
        <figcaption className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs tabular-nums text-muted">
          <span>{PRESETS[0].name} preset</span>
          <span>{GRID.join(' × ')} cells</span>
          <span>drag to orbit</span>
        </figcaption>
      </figure>
    </section>
  );
}
