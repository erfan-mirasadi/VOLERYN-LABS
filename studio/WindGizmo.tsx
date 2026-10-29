'use client';

import { useEffect, useRef } from 'react';
import { useStudioStore } from './useStudioStore';

const CENTRE: [number, number, number] = [0, 0.005, 0];
const LENGTH = 0.32;

// An arrow on the floor of the simulation box showing where the wind blows; drag its head to turn it.
export default function WindGizmo() {
  const { engine, setTweak } = useStudioStore();
  const line = useRef<SVGLineElement>(null);
  const head = useRef<SVGGElement>(null);
  const dragging = useRef<number | null>(null);

  useEffect(() => {
    if (!engine) return;
    let frame = 0;
    const update = () => {
      const { wind, windYaw } = useStudioStore.getState().tweaks;
      const yaw = (windYaw * Math.PI) / 180;
      const from = engine.project(CENTRE);
      const to = engine.project([CENTRE[0] + LENGTH * Math.cos(yaw), CENTRE[1], CENTRE[2] + LENGTH * Math.sin(yaw)]);
      if (from && to && line.current && head.current) {
        line.current.setAttribute('x1', String(from[0]));
        line.current.setAttribute('y1', String(from[1]));
        line.current.setAttribute('x2', String(to[0]));
        line.current.setAttribute('y2', String(to[1]));
        const angle = (Math.atan2(to[1] - from[1], to[0] - from[0]) * 180) / Math.PI;
        head.current.setAttribute('transform', `translate(${to[0]} ${to[1]}) rotate(${angle})`);
        head.current.style.opacity = wind === 0 ? '0.55' : '1';
      }
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [engine]);

  const turn = (event: React.PointerEvent<SVGGElement>) => {
    if (!engine) return;
    const rect = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const floor = engine.floorPoint(event.clientX - rect.left, event.clientY - rect.top);
    if (!floor) return;
    const yaw = (Math.atan2(floor[2] - CENTRE[2], floor[0] - CENTRE[0]) * 180) / Math.PI;
    setTweak('windYaw', Math.round((yaw + 360) % 360));
  };

  return (
    <svg aria-hidden className="pointer-events-none absolute inset-0 size-full overflow-visible text-ember">
      <line ref={line} stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeDasharray="4 6" opacity={0.7} />
      <g
        ref={head}
        className="pointer-events-auto cursor-grab active:cursor-grabbing"
        onPointerDown={(event) => {
          dragging.current = event.pointerId;
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => dragging.current === event.pointerId && turn(event)}
        onPointerUp={() => (dragging.current = null)}
      >
        <circle r={14} fill="var(--color-background)" fillOpacity={0.8} stroke="currentColor" strokeWidth={1.5} />
        <path d="M-5 -6 L7 0 L-5 6 Z" fill="currentColor" />
      </g>
    </svg>
  );
}
