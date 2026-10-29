import Link from 'next/link';
import type { ReactNode } from 'react';
import { GRID } from '@/engine/scene';
import Controls from './Controls';
import Export from './Export';
import Hero from './Hero';
import Logo, { LogoMark } from './Logo';
import Presets from './Presets';

const LABEL = 'font-mono text-[11px] uppercase tracking-[0.14em] text-muted';

type ChapterProps = { index: string; title: string; lead: string; children: ReactNode };

// A numbered chapter: title and lead on the left stay put while the content on the right scrolls.
function Chapter({ index, title, lead, children }: ChapterProps) {
  return (
    <section className="grid gap-10 border-t border-line py-20 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] lg:gap-20 lg:py-28">
      <div className="flex flex-col gap-5 lg:sticky lg:top-24 lg:self-start">
        <span className={LABEL}>{index}</span>
        <h2 className="max-w-md text-balance text-3xl font-medium tracking-[-0.02em] sm:text-4xl">{title}</h2>
        <p className="max-w-md text-lg leading-relaxed text-muted">{lead}</p>
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-6 sm:px-10">
      <header className="flex items-center justify-between py-6">
        <Logo />
        <Link href="/studio" className={`${LABEL} transition-colors hover:text-foreground`}>
          Open studio
        </Link>
      </header>

      <main className="flex-1">
        <Hero />

        <Chapter
          index="01"
          title="Four presets, tuned against footage."
          lead={`A preset is a set of physical parameters: buoyancy, burn rate, cooling, soot yield, turbulence. Each was matched by eye against reference footage. The diagrams show each setup to scale inside the ${GRID.join(' × ')} box.`}
        >
          <Presets />
        </Chapter>

        <Chapter
          index="02"
          title="Shape it while it burns."
          lead="Eight controls act on the running simulation; nothing is re-cooked. The wind has a direction you drag on the floor of the box, and the sound is synthesised from the same numbers."
        >
          <Controls />
        </Chapter>

        <Chapter
          index="03"
          title="Layers a compositor expects."
          lead="An export re-runs the simulation offline at the size and quality you choose. Alpha comes from the volume itself, so fire adds and smoke layers without keying. EXR frames split the light into groups and carry the camera in the header."
        >
          <Export />
        </Chapter>

        <section className="flex flex-col gap-8 border-t border-line py-20 lg:flex-row lg:items-end lg:justify-between lg:py-28">
          <h2 className="max-w-lg text-balance text-3xl font-medium tracking-[-0.02em] sm:text-4xl">Runs on your GPU, in your browser. Nothing is uploaded.</h2>
          <Link
            href="/studio"
            className="inline-flex h-11 items-center self-start rounded-md bg-ember px-5 text-sm font-medium text-black transition hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ember"
          >
            Open the studio
          </Link>
        </section>
      </main>

      <footer className={`flex flex-wrap items-center justify-between gap-4 border-t border-line py-8 ${LABEL}`}>
        <span className="flex items-center gap-2">
          <LogoMark size={16} /> Voleryn Labs
        </span>
        <span>Built on WebGPU</span>
      </footer>
    </div>
  );
}
