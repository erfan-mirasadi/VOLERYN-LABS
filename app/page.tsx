import Link from 'next/link';

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-16 px-5 py-8 sm:px-8">
      <header className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-[0.3em]">Voleryn Labs</span>
        <Link href="/studio" className="rounded-full border border-line px-4 py-1.5 text-sm text-muted hover:text-foreground">
          Open studio
        </Link>
      </header>
      <section className="flex max-w-2xl flex-col gap-6">
        <h1 className="text-balance text-4xl font-semibold leading-tight sm:text-6xl">Real fire, in the browser, rendered for production.</h1>
        <p className="max-w-lg text-lg text-muted">
          Simulate fire, smoke and explosions on your GPU and export elements with alpha for games, motion design and compositing.
        </p>
      </section>
    </main>
  );
}
