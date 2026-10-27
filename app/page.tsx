import Link from 'next/link';
import HeroDemo from './HeroDemo';
import Logo, { LogoMark } from './Logo';

const POINTS = [
  {
    title: 'Simulated, not stock',
    body: 'A grid-based gas simulation burns fuel into heat and soot sixty times a second, so every frame is new and every angle is yours.',
  },
  {
    title: 'Alpha that composites',
    body: 'Premultiplied alpha comes straight out of the volume render: fire adds, smoke layers, nothing needs keying.',
  },
  {
    title: 'Made for the pipeline',
    body: 'Flipbooks for game engines, image sequences for motion design and layered EXR for compositors.',
  },
];

const STEPS = [
  ['Pick a preset', 'Fire, campfire, smoke or an explosion, each tuned against real footage.'],
  ['Tune it live', 'Heat, swirl and wind respond while the simulation keeps running.'],
  ['Watch the real thing', 'The preview is the same simulation the final render re-runs, only sharper.'],
  ['Export', 'PNG and EXR sequences, flipbooks and MP4, written straight to disk.'],
];

export default function Home() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
        <Logo />
        <Link href="/studio" className="rounded-full border border-line px-4 py-1.5 text-sm text-muted transition-colors hover:border-foreground hover:text-foreground">
          Open studio
        </Link>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-24 px-5 pb-24 pt-8 sm:px-8">
        <section className="grid items-center gap-12 overflow-x-clip lg:grid-cols-[1.05fr_1fr]">
          <div className="flex flex-col gap-7">
            <p className="text-xs font-medium uppercase tracking-[0.25em] text-ember">WebGPU · runs on your GPU</p>
            <h1 className="text-balance text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">Real fire, in the browser, rendered for production.</h1>
            <p className="max-w-lg text-lg leading-relaxed text-muted">
              Simulate fire, smoke and explosions on your own graphics card and export elements with alpha for games, motion design and compositing.
            </p>
            <div className="flex flex-wrap items-center gap-4">
              <Link href="/studio" className="rounded-full bg-ember px-6 py-3 font-medium text-black transition hover:brightness-110">
                Start a fire
              </Link>
              <a href="#how" className="text-sm text-muted transition-colors hover:text-foreground">
                How it works
              </a>
            </div>
          </div>
          <div className="relative">
            <div aria-hidden="true" className="absolute -inset-16 -z-10 rounded-full bg-[radial-gradient(closest-side,rgba(255,122,26,0.18),transparent)]" />
            <div className="overflow-hidden rounded-2xl border border-line bg-black">
              <HeroDemo />
              <p className="flex flex-wrap items-center justify-between gap-x-6 gap-y-1 border-t border-line px-4 py-2.5 font-mono text-[11px] uppercase tracking-wider text-muted">
                <span>Live simulation · 64 × 96 × 64</span>
                <span>Drag to orbit</span>
              </p>
            </div>
          </div>
        </section>

        <section className="grid gap-10 border-t border-line pt-12 sm:grid-cols-3">
          {POINTS.map(({ title, body }) => (
            <div key={title} className="flex flex-col gap-3">
              <h2 className="text-lg font-medium">{title}</h2>
              <p className="text-sm leading-relaxed text-muted">{body}</p>
            </div>
          ))}
        </section>

        <section id="how" className="flex flex-col gap-10 border-t border-line pt-12">
          <h2 className="text-2xl font-semibold tracking-tight">How it works</h2>
          <ol className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map(([title, body], index) => (
              <li key={title} className="flex flex-col gap-3 border-l border-line pl-5">
                <span className="font-mono text-xs text-ember">{String(index + 1).padStart(2, '0')}</span>
                <h3 className="font-medium">{title}</h3>
                <p className="text-sm leading-relaxed text-muted">{body}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-8 text-sm text-muted sm:px-8">
          <span className="flex items-center gap-2">
            <LogoMark size={18} /> Voleryn Labs
          </span>
          <span>Everything runs in your browser. Nothing is uploaded.</span>
        </div>
      </footer>
    </div>
  );
}
