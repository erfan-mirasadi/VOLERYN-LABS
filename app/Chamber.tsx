'use client';

import { useState } from 'react';
import { PRESETS } from '@/presets/presets';
import FireCanvas from '@/studio/FireCanvas';
import type { GpuStatus } from '@/studio/useStudioStore';

export default function HeroDemo() {
  const [status, setStatus] = useState<GpuStatus>('loading');

  return (
    <div className="relative aspect-[4/5] w-full overflow-hidden rounded-xl bg-black sm:aspect-[5/6]">
      <FireCanvas preset={PRESETS[0]} onEngine={(_, next) => setStatus(next)} label="Live fire simulation; drag to orbit" />
      {status === 'unsupported' && (
        <p className="absolute inset-0 flex items-center justify-center p-8 text-center text-sm text-muted">
          The live demo needs WebGPU. Open this page in Chrome, Edge or Safari 26 on a computer.
        </p>
      )}
    </div>
  );
}
