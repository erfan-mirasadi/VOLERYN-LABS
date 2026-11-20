'use client';

import { useEffect, useRef, useState } from 'react';
import { useStudioStore } from './useStudioStore';

// How close to the middle of the frame, in half-frames, the fire has to come for the magnet to take it.
const SNAP = 0.04;
// The fire stands on the origin and its flames reach about this high, in world units.
const FLAME_HEIGHT = 1;

const settle = (value: number) => (Math.abs(value) < SNAP ? 0 : Math.round(Math.min(1, Math.max(-1, value)) * 1000) / 1000);

// A spot picked for the fire to stand on does not cover it: the fire's depth is set this much nearer than the spot.
const PICK_MARGIN = 0.02;

// The fire as something to take hold of once footage is loaded: drag it across the frame, and near the middle each
// axis snaps back to where it started, with a guide line while it holds. Dragging anywhere else still orbits. While
// a depth is being picked the whole frame takes the click instead.
export default function PlacementOverlay() {
  const { engine, framing, picking, exporting, beginAdjustment, setFraming, setOcclusion, setPicking } = useStudioStore();
  const handle = useRef<HTMLDivElement>(null);
  const drag = useRef<{ id: number; x: number; y: number; from: { x: number; y: number } } | null>(null);
  const [held, setHeld] = useState<{ x: boolean; y: boolean } | null>(null);

  useEffect(() => {
    if (!engine) return;
    let frame = 0;
    const follow = () => {
      const base = engine.project([0, 0, 0]);
      const top = engine.project([0, FLAME_HEIGHT, 0]);
      if (base && top && handle.current) {
        const height = base[1] - top[1];
        const { style } = handle.current;
        style.left = `${base[0] - height / 4}px`;
        style.top = `${top[1]}px`;
        style.width = `${height / 2}px`;
        style.height = `${height}px`;
      }
      frame = requestAnimationFrame(follow);
    };
    frame = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(frame);
  }, [engine]);

  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    const start = drag.current;
    if (start?.id !== event.pointerId) return;
    const frame = event.currentTarget.parentElement!;
    const x = settle(start.from.x + (2 * (event.clientX - start.x)) / frame.clientWidth);
    const y = settle(start.from.y - (2 * (event.clientY - start.y)) / frame.clientHeight);
    setFraming({ x, y });
    setHeld({ x: x === 0, y: y === 0 });
  };
  const release = () => {
    drag.current = null;
    setHeld(null);
  };

  return (
    <>
      <div
        ref={handle}
        aria-hidden
        className={`absolute touch-none ${exporting ? 'pointer-events-none' : 'cursor-move'}`}
        onPointerDown={(event) => {
          if (event.button !== 0 || exporting) return;
          beginAdjustment(event.pointerId);
          drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, from: { x: framing.x, y: framing.y } };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={move}
        onPointerUp={release}
        onPointerCancel={release}
        onLostPointerCapture={release}
      />
      {held?.x && <div aria-hidden className="pointer-events-none absolute inset-y-0 left-1/2 w-px bg-ember/80" />}
      {held?.y && <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 h-px bg-ember/80" />}
      {picking && !exporting && (
        <div
          aria-hidden
          className="absolute inset-0 cursor-crosshair"
          onClick={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            const near = engine?.nearnessAt((event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height);
            // The engine's depth setting runs from 1.05 nearness at 0 down to -0.05 at 1.
            if (near != null) setOcclusion({ depth: Math.round(Math.min(1, Math.max(0, (1.05 - near - PICK_MARGIN) / 1.1)) * 200) / 200 });
            setPicking(false);
          }}
        />
      )}
    </>
  );
}
