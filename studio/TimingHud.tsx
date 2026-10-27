'use client';

import { useEffect, useState } from 'react';
import type { EngineStats } from '@/engine/engine';
import { useStudioStore } from './useStudioStore';

const REFRESH_MS = 250;

export default function TimingHud() {
  const engine = useStudioStore((state) => state.engine);
  const [stats, setStats] = useState<EngineStats | null>(null);

  useEffect(() => {
    if (!engine) return;
    const timer = setInterval(() => setStats(engine.stats()), REFRESH_MS);
    return () => clearInterval(timer);
  }, [engine]);

  if (!stats) return null;
  const rows: [string, string][] = [
    ['fps', `${stats.fps.toFixed(0)} / ${stats.displayHz.toFixed(0)}`],
    ['scale', `${Math.round(stats.renderScale * 100)}%`],
    ['gpu frame', stats.hasTimestamps ? `${stats.gpuMs.toFixed(2)} ms` : 'no timestamps'],
    ['simulation', `${stats.simulationMs.toFixed(2)} ms`],
    ['lighting', `${stats.lightingMs.toFixed(2)} ms`],
    ['raymarch', `${stats.raymarchMs.toFixed(2)} ms`],
    ['taa', `${stats.taaMs.toFixed(2)} ms`],
    ['composite', `${stats.compositeMs.toFixed(2)} ms`],
    ['steps', `${stats.stepsPerSecond.toFixed(0)} /s`],
    ['divergence', stats.residual.toExponential(1)],
    ['grid', stats.grid.join(' × ')],
  ];

  return (
    <dl className="pointer-events-none grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 rounded-md bg-background/80 px-3 py-2 font-mono text-xs text-muted backdrop-blur">
      {rows.map(([name, value]) => (
        <div key={name} className="contents">
          <dt>{name}</dt>
          <dd className="text-right tabular-nums text-foreground">{value}</dd>
        </div>
      ))}
      <dt className="col-span-2 mt-1 truncate text-[10px]">{stats.adapterName}</dt>
    </dl>
  );
}
