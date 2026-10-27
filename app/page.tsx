import Link from 'next/link';
import HeroDemo from './HeroDemo';

const POINTS = [
  ['Simulated, not stock', 'A grid-based gas simulation burns fuel into heat and soot, sixty times a second, on your own GPU.'],
  ['Alpha that composites', 'Premultiplied alpha straight from the volume render, so fire adds and smoke layers.'],
  ['Made for the pipeline', 'Flipbooks for game engines, image sequences and layered EXR for compositors.'],
];

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-16 px-5 py-8 sm:px-8">
      <header className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-[0.3em]">Voleryn Labs</span>
        <Link href="/studio" className="rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:text-foreground">
          Open studio
        </Link>
      </header>

      <section className="grid items-center gap-10 lg:grid-cols-[1.1fr_1fr]">
        <div className="flex flex-col gap-6">
          <h1 className="text-balance text-4xl font-semibold leading-tight sm:text-6xl">Real fire, in the browser, rendered for production.</h1>
          <p className="max-w-lg text-lg text-muted">
            Simulate fire, smoke and explosions on your GPU and export elements with alpha for games, motion design and compositing.
          </p>
          <Link href="/studio" className="self-start rounded-full bg-ember px-6 py-3 font-medium text-black hover:brightness-110">
            Start a fire
          </Link>
        </div>
        <HeroDemo />
      </section>

      <section className="grid gap-8 border-t border-line pt-10 sm:grid-cols-3">
        {POINTS.map(([title, body]) => (
          <div key={title} className="flex flex-col gap-2">
            <h2 className="font-medium">{title}</h2>
            <p className="text-sm text-muted">{body}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
