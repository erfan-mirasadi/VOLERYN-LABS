'use client';

import { useEffect, useRef } from 'react';
import type { Engine } from '@/engine/engine';
import { WebGpuUnavailable } from '@/engine/gpu';
import type { Preset } from '@/engine/scene';
import type { GpuStatus } from './useStudioStore';

type Props = {
  preset: Preset;
  onEngine: (engine: Engine | null, status: GpuStatus) => void;
  label: string;
};

// Fills its positioned parent with one canvas and owns the engine on it; the engine module loads only when the canvas mounts.
export default function FireCanvas({ preset, onEngine, label }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current!;
    let engine: Engine | null = null;
    let cancelled = false;
    import('@/engine/engine')
      .then(({ createEngine }) => createEngine(canvas, preset))
      .then((created) => {
        if (cancelled) return created.dispose();
        engine = created;
        onEngine(engine, 'ready');
        engine.lost.then(() => onEngine(null, 'error'));
      })
      .catch((error: unknown) => {
        console.error(error);
        onEngine(null, error instanceof WebGpuUnavailable ? 'unsupported' : 'error');
      });
    return () => {
      cancelled = true;
      engine?.dispose();
      onEngine(null, 'loading');
    };
    // The engine is created once per canvas; later preset changes go through engine.setPreset.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <canvas ref={canvasRef} aria-label={label} className="absolute inset-0 h-full w-full cursor-grab touch-pan-y active:cursor-grabbing" />;
}
