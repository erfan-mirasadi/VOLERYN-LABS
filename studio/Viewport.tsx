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
  const { backdrop, status, presetId, sound, setEngine, toggleSound } = useStudioStore();
  const message = STATUS_MESSAGE[status];

  return (
    <div className={`relative min-h-[60vh] flex-1 overflow-hidden ${BACKDROP_CLASS[backdrop]}`}>
      <FireCanvas preset={presetById(presetId)} onEngine={setEngine} label="Fire simulation; drag to orbit" />
      <div className="absolute left-3 top-3">
        <TimingHud />
      </div>
      {status === 'ready' && (
        <button
          type="button"
          onClick={toggleSound}
          aria-pressed={sound}
          aria-label="Sound"
          className="absolute right-3 top-3 grid size-11 place-items-center rounded-full bg-background/80 text-muted backdrop-blur transition-colors hover:text-foreground aria-pressed:text-ember"
        >
          <svg viewBox="0 0 24 24" aria-hidden className="size-5" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 9.5h3.5L12 6v12l-4.5-3.5H4z" />
            <path d={sound ? 'M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11' : 'm16 10 4 4m0-4-4 4'} />
          </svg>
        </button>
      )}
      {message && (
        <p className="absolute inset-x-0 bottom-0 bg-background/80 px-4 py-3 text-center text-sm text-muted backdrop-blur">{message}</p>
      )}
    </div>
  );
}
