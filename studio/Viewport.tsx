'use client';

import { presetById } from '@/presets/presets';
import FireCanvas from './FireCanvas';
import TimingHud from './TimingHud';
import { useStudioStore } from './useStudioStore';

const BACKDROP_CLASS = {
  checker: 'bg-[repeating-conic-gradient(#2a2d35_0_25%,#1b1d24_0_50%)] bg-[length:24px_24px]',
  dark: 'bg-black',
  light: 'bg-[#e9e7e1]',
};

const STATUS_MESSAGE = {
  loading: 'Starting the GPU…',
  ready: null,
  unsupported: 'This browser has no WebGPU. Use Chrome, Edge or Safari 26 on a desktop GPU.',
  error: 'The GPU stopped. Reload the page to start again.',
};

export default function Viewport() {
  const { backdrop, status, presetId, setEngine } = useStudioStore();
  const message = STATUS_MESSAGE[status];

  return (
    <div className={`relative min-h-[60vh] flex-1 overflow-hidden ${BACKDROP_CLASS[backdrop]}`}>
      <FireCanvas preset={presetById(presetId)} onEngine={setEngine} label="Fire simulation; drag to orbit" />
      <div className="absolute left-3 top-3">
        <TimingHud />
      </div>
      {message && (
        <p className="absolute inset-x-0 bottom-0 bg-background/80 px-4 py-3 text-center text-sm text-muted backdrop-blur">{message}</p>
      )}
    </div>
  );
}
