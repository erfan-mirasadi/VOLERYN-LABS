import Link from 'next/link';
import Hero from './Hero';
import Logo, { LogoMark } from './Logo';
import Specs from './Specs';

export default function Home() {
  return (
    <div className="flex flex-1 flex-col">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6 sm:px-8">
        <Logo />
        <Link href="/studio" className="text-sm text-muted transition-colors hover:text-foreground">
          Open studio
        </Link>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 sm:px-8">
        <Hero />
        <Specs />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-6 py-8 text-sm text-muted sm:px-8">
          <span className="flex items-center gap-2">
            <LogoMark size={18} /> Voleryn Labs
          </span>
          <Link href="/studio" className="transition-colors hover:text-foreground">
            Open the studio
          </Link>
        </div>
      </footer>
    </div>
  );
}
