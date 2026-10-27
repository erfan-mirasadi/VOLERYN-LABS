import Link from 'next/link';
import SimulationPanel from './SimulationPanel';
import Viewport from './Viewport';

export default function Studio() {
  return (
    <div className="flex min-h-dvh flex-col lg:h-dvh lg:flex-row">
      <Viewport />
      <aside className="flex w-full flex-col gap-8 border-t border-line bg-panel p-5 lg:w-80 lg:border-l lg:border-t-0 lg:overflow-y-auto">
        <Link href="/" className="text-xs font-semibold uppercase tracking-[0.3em] text-muted hover:text-foreground">
          Voleryn Labs
        </Link>
        <SimulationPanel />
      </aside>
    </div>
  );
}
