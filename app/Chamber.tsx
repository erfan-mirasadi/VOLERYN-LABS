'use client';

import { useEffect, useRef, useState } from 'react';
import type { Engine, EngineStats } from '@/engine/engine';
import { GRID } from '@/engine/scene';
import { PRESETS } from '@/presets/presets';
import FireCanvas from '@/studio/FireCanvas';
import type { GpuStatus } from '@/studio/useStudioStore';

const STATE: Record<GpuStatus, string> = { loading: 'Starting', ready: 'Live', unsupported: 'No WebGPU', error: 'GPU stopped' };
const REFRESH_MS = 500;
const LABEL = 'font-mono text-[10px] uppercase tracking-[0.14em] text-muted';

// The live simulation in a plain instrument frame: preset and GPU state along the top, grid and frame timing along the bottom.
export default function Chamber() {
  const engine = useRef<Engine | null>(null);
  const [status, setStatus] = useState<GpuStatus>('loading');
  const [stats, setStats] = useState<EngineStats | null>(null);

  useEffect(() => {
    if (status !== 'ready') return;
    const timer = setInterval(() => setStats(engine.current?.stats() ?? null), REFRESH_MS);
    return () => clearInterval(timer);
  }, [status]);

  const onEngine = (next: Engine | null, state: GpuStatus) => {
    engine.current = next;
    setStatus(state);
    if (state !== 'ready') setStats(null);
  };

  return (
    <div className="relative border border-line bg-black">
      <div className="relative aspect-[4/5] sm:aspect-[5/6]">
        <FireCanvas preset={PRESETS[0]} onEngine={onEngine} label="Live fire simulation; drag to orbit" />
      </div>
      {status === 'unsupported' && (
        <p className="pointer-events-none absolute inset-0 flex items-center justify-center p-10 text-center text-sm leading-relaxed text-muted">
          The live simulation needs WebGPU. Open this page in Chrome, Edge or Safari 26 on a computer.
        </p>
      )}
      <div className={`pointer-events-none absolute inset-0 ${LABEL}`}>
        <span className="absolute left-3 top-3">{PRESETS[0].name}</span>
        <span className="absolute right-3 top-3 flex items-center gap-1.5">
          <span className={`size-1.5 rounded-full ${status === 'ready' ? 'animate-pulse bg-ember' : 'bg-muted/40'}`} />
          {STATE[status]}
        </span>
        <span className="absolute bottom-3 left-3">{GRID.join(' × ')} cells</span>
        {stats && (
          <span className="absolute bottom-3 right-3 tabular-nums">
            {stats.fps.toFixed(0)} fps{stats.hasTimestamps && ` · ${stats.gpuMs.toFixed(1)} ms`}
          </span>
        )}
      </div>
    </div>
  );
}
