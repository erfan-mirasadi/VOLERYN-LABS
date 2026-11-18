'use client';

import { useEffect, useState } from 'react';
import { useStudioStore } from './useStudioStore';

const BUTTON = 'grid size-9 shrink-0 place-items-center rounded-full hover:text-foreground disabled:opacity-40';
const ICON = { viewBox: '0 0 24 24', 'aria-hidden': true, className: 'size-4', fill: 'currentColor' } as const;

// The video plate's own transport: play it or hold it on a frame, step a frame either way, or scrub through it.
// A held frame's depth is read at full detail, as a render reads it.
export default function PlateTransport({ video }: { video: HTMLVideoElement }) {
  const engine = useStudioStore((state) => state.engine);
  const exporting = useStudioStore((state) => state.exporting);
  const [time, setTime] = useState(video.currentTime);
  const [paused, setPaused] = useState(video.paused);
  // Browsers do not say a video's frame rate, so it is measured while it plays: the shortest step between two frames
  // shown one after the other. Seeks, an export's among them, jump by other amounts and are left out.
  const [measured, setMeasured] = useState<number | null>(null);

  useEffect(() => {
    let request = 0;
    let last: VideoFrameCallbackMetadata | null = null;
    const onFrame = (_: number, frame: VideoFrameCallbackMetadata) => {
      const length = last && frame.presentedFrames === last.presentedFrames + 1 && !video.paused ? frame.mediaTime - last.mediaTime : 0;
      if (length > 0) setMeasured((known) => Math.min(known ?? length, length));
      last = frame;
      setTime(frame.mediaTime);
      request = video.requestVideoFrameCallback(onFrame);
    };
    request = video.requestVideoFrameCallback(onFrame);
    const onState = () => setPaused(video.paused);
    const onSeeked = () => setTime(video.currentTime);
    video.addEventListener('play', onState);
    video.addEventListener('pause', onState);
    video.addEventListener('seeked', onSeeked);
    return () => {
      video.cancelVideoFrameCallback(request);
      video.removeEventListener('play', onState);
      video.removeEventListener('pause', onState);
      video.removeEventListener('seeked', onSeeked);
    };
  }, [video]);

  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  const frameLength = measured ?? 1 / 30;
  const frames = Math.max(1, Math.round(duration / frameLength));
  const shown = Math.min(frames - 1, Math.floor(time / frameLength + 0.01));
  // Aimed at the middle of a frame: a time on a frame's very start can land on the one before it.
  const show = (index: number) => engine?.setPlateTime((Math.min(Math.max(index, 0), frames - 1) + 0.5) * frameLength);
  const step = (by: number) => {
    engine?.playPlate(false);
    show(shown + by);
  };

  return (
    <div className="absolute inset-x-3 bottom-3 mx-auto flex max-w-md items-center gap-1 rounded-full bg-background/80 px-1.5 text-muted backdrop-blur">
      <button type="button" aria-label="Previous frame" disabled={exporting} onClick={() => step(-1)} className={BUTTON}>
        <svg {...ICON}>
          <path d="M6 5h2v14H6zM18 5v14L9 12z" />
        </svg>
      </button>
      <button type="button" aria-label={paused ? 'Play' : 'Pause'} disabled={exporting} onClick={() => engine?.playPlate(paused)} className={BUTTON}>
        <svg {...ICON}>
          <path d={paused ? 'M7 5v14l12-7z' : 'M7 5h3.5v14H7zm6.5 0H17v14h-3.5z'} />
        </svg>
      </button>
      <button type="button" aria-label="Next frame" disabled={exporting} onClick={() => step(1)} className={BUTTON}>
        <svg {...ICON}>
          <path d="M16 5h2v14h-2zM6 5v14l9-7z" />
        </svg>
      </button>
      <input
        type="range"
        aria-label="Footage time"
        min={0}
        max={frames - 1}
        value={shown}
        disabled={exporting}
        onChange={(event) => show(Number(event.target.value))}
        className="mx-1 min-w-0 flex-1 accent-ember"
      />
      <span className="shrink-0 pr-2 font-mono text-xs tabular-nums">
        {shown + 1} / {frames}
      </span>
    </div>
  );
}
