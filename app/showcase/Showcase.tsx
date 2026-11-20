'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import type { ShowcaseRenderer } from './renderer';

const chapters = [
  { id: 'smoke', label: 'Smoke', title: 'Matter\nin motion.', preset: 'smoke', anchor: 0 },
  { id: 'fire', label: 'Fire', title: 'Fire.', preset: 'fire', anchor: 1.40 },
  { id: 'explosion', label: 'Explosion', title: 'Explosion.', preset: 'explosion', anchor: 2.30 },
];
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

export default function Showcase() {
  const journey = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const interaction = useRef<HTMLDivElement>(null);
  const updateMotion = useRef<(motion: boolean) => void>(null);
  const [chapterIndex, setChapterIndex] = useState(0);
  const [playing, setPlaying] = useState(true);
  const chapter = chapters[chapterIndex];

  useEffect(() => {
    const region = journey.current!;
    const frame = stage.current!;
    const surface = canvas.current!;
    const input = interaction.current!;
    const abort = new AbortController();
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let visible = false;
    let motion = !reduceMotion.matches;
    let renderer: ShowcaseRenderer | undefined;
    let initializing = false;
    let failed = false;
    let raf = 0;
    let phase = 0;
    let staticPhase = 0;
    let currentChapter = 0;
    let width = frame.clientWidth;
    let height = frame.clientHeight;
    let orbit = 0;
    let elevation = 0;
    let time = 2;
    let lastDraw = 0;
    let scrollChanged = true;
    let needsFrame = true;
    let sampleTime = 0;
    let samples = 0;
    let intervalSum = 0;
    let drag: { id: number; x: number; y: number } | undefined;

    function schedule() {
      if (!raf && visible && !document.hidden) raf = requestAnimationFrame(draw);
    }

    async function initialize() {
      if (renderer || initializing || failed || abort.signal.aborted) return;
      initializing = true;
      try {
        const { createShowcaseRenderer } = await import('./renderer');
        abort.signal.throwIfAborted();
        renderer = await createShowcaseRenderer(surface, abort.signal);
        needsFrame = true;
        schedule();
      } catch (error) {
        if (!abort.signal.aborted) {
          failed = true;
          frame.dataset.render = 'still';
          // Shader failures should be visible during development.
          if (process.env.NODE_ENV !== 'production') console.warn('Showcase could not start', error);
        }
      } finally {
        initializing = false;
      }
    }

    function readScroll() {
      const distance = Math.max(1, region.offsetHeight - height);
      const nextPhase = clamp(-region.getBoundingClientRect().top / distance * 3, 0, 3);
      const nextChapter = Math.min(2, Math.floor(nextPhase + 0.0001));
      if (currentChapter !== nextChapter) {
        currentChapter = nextChapter;
        setChapterIndex(currentChapter);
        staticPhase = chapters[currentChapter].anchor;
        needsFrame = true;
      }
      if (motion && Math.abs(phase - nextPhase) > 0.0001) needsFrame = true;
      phase = nextPhase;
      frame.style.setProperty('--travel', String(phase / 3));
    }

    function draw(now: number) {
      raf = 0;
      if (!visible || document.hidden) return;
      if (scrollChanged) {
        readScroll();
        scrollChanged = false;
      }
      if (!renderer || !width || !height) return;
      const ambient = motion && (phase < 2 || phase > 2.32);
      const cadence = lastDraw ? now - lastDraw : 0;
      const interval = width < 680 ? 1000 / 24 : 1000 / 30;
      if (needsFrame || (ambient && cadence >= interval - 1)) {
        if (ambient) time += Math.min(cadence, 100) / 1000;
        const timing = renderer.draw({
          phase: motion ? phase : staticPhase,
          time, orbit, elevation, width, height,
          animating: ambient,
          interactive: motion && needsFrame,
        }, now, ambient ? cadence / interval : 0);
        if (process.env.NODE_ENV !== 'production' && ambient && cadence > 0 && cadence < 170) {
          intervalSum += cadence;
          samples++;
          if (now - sampleTime > 1300 && samples >= 12) {
            frame.dataset.cadence = (1000 * samples / intervalSum).toFixed(1);
            frame.dataset.gpuMs = timing.toFixed(1);
            intervalSum = 0;
            samples = 0;
            sampleTime = now;
          }
        }
        if (frame.dataset.render !== 'live') frame.dataset.render = 'live';
        lastDraw = now;
        needsFrame = false;
      }
      if (ambient) schedule();
    }

    function onScroll() {
      scrollChanged = true;
      schedule();
    }

    function onResize() {
      width = frame.clientWidth;
      height = frame.clientHeight;
      needsFrame = true;
      onScroll();
    }

    function changeMotion(next: boolean) {
      if (motion && !next) staticPhase = phase;
      motion = next;
      lastDraw = 0;
      needsFrame = true;
      setPlaying(next);
      onScroll();
    }

    function onPreference() {
      changeMotion(!reduceMotion.matches);
    }

    function onVisibility() {
      lastDraw = 0;
      if (document.hidden) cancelAnimationFrame(raf);
      else {
        needsFrame = true;
        onScroll();
      }
      if (document.hidden) raf = 0;
    }

    function onLost(event: Event) {
      event.preventDefault();
      renderer = undefined;
      frame.dataset.render = 'still';
    }

    function onRestored() {
      failed = false;
      if (visible) void initialize();
    }

    function onPress(event: PointerEvent) {
      if (!renderer || drag || event.button !== 0) return;
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
      input.setPointerCapture(event.pointerId);
      input.dataset.dragging = 'true';
    }

    function onMove(event: PointerEvent) {
      if (!drag || drag.id !== event.pointerId) return;
      orbit = clamp(orbit - (event.clientX - drag.x) / 370, -0.68, 0.68);
      elevation = clamp(elevation + (event.clientY - drag.y) / 330, -0.35, 0.62);
      drag.x = event.clientX;
      drag.y = event.clientY;
      needsFrame = true;
      schedule();
    }

    function onRelease() {
      drag = undefined;
      delete input.dataset.dragging;
    }

    function onKey(event: KeyboardEvent) {
      if (!renderer || (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight' && event.key !== 'Home')) return;
      event.preventDefault();
      orbit = event.key === 'Home' ? 0 : clamp(orbit + (event.key === 'ArrowLeft' ? -0.12 : 0.12), -0.68, 0.68);
      if (event.key === 'Home') elevation = 0;
      needsFrame = true;
      schedule();
    }

    const resize = new ResizeObserver(onResize);
    const intersection = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting && entry.intersectionRatio >= 0.04;
      lastDraw = 0;
      if (visible) {
        needsFrame = true;
        void initialize();
        onScroll();
      } else {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    }, { threshold: 0.04 });
    resize.observe(frame);
    intersection.observe(frame);
    updateMotion.current = changeMotion;
    const initial = requestAnimationFrame(onPreference);
    window.addEventListener('scroll', onScroll, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    reduceMotion.addEventListener('change', onPreference);
    surface.addEventListener('webglcontextlost', onLost);
    surface.addEventListener('webglcontextrestored', onRestored);
    input.addEventListener('pointerdown', onPress);
    input.addEventListener('pointermove', onMove);
    input.addEventListener('lostpointercapture', onRelease);
    input.addEventListener('keydown', onKey);
    return () => {
      abort.abort();
      updateMotion.current = null;
      cancelAnimationFrame(initial);
      cancelAnimationFrame(raf);
      resize.disconnect();
      intersection.disconnect();
      window.removeEventListener('scroll', onScroll);
      document.removeEventListener('visibilitychange', onVisibility);
      reduceMotion.removeEventListener('change', onPreference);
      surface.removeEventListener('webglcontextlost', onLost);
      surface.removeEventListener('webglcontextrestored', onRestored);
      input.removeEventListener('pointerdown', onPress);
      input.removeEventListener('pointermove', onMove);
      input.removeEventListener('lostpointercapture', onRelease);
      input.removeEventListener('keydown', onKey);
      renderer?.dispose();
    };
  }, []);

  return (
    <section className="showcase-journey" ref={journey} aria-label="Interactive 3D effects showcase">
      <div className="showcase-anchors" aria-hidden="true">
        {chapters.map(({ id, anchor }) => <span key={id} id={id} style={{ top: `${anchor / 3 * 100}%` }} />)}
      </div>
      <div className="showcase-stage" ref={stage} data-scene={chapter.id}>
        <div className="showcase-poster" />
        <canvas className="showcase-canvas" ref={canvas} aria-hidden="true" />
        <div className="showcase-orbit" ref={interaction} tabIndex={0} role="group" aria-label={`Orbit the ${chapter.label.toLowerCase()} scene. Use left and right arrow keys to rotate, or Home to reset.`} />
        <header className="showcase-header">
          <a className="showcase-name" href="#smoke">Voleryn</a>
          <Link className="showcase-open" prefetch={false} href="/studio">Open Studio <span aria-hidden="true">↗</span></Link>
        </header>
        <div className="showcase-title">
          {chapterIndex === 0 ? <h1>{chapter.title}</h1> : <h2>{chapter.title}</h2>}
          {chapterIndex === 0 ? <p>Scroll to explore <span aria-hidden="true">↓</span></p> :
            <Link prefetch={false} href={`/studio?preset=${chapter.preset}`}>Open this effect <span aria-hidden="true">↗</span></Link>}
        </div>
        <div className="showcase-controls">
          <nav aria-label="Effects" className="showcase-chapters">
            {chapters.map(({ id, label }, index) => <a href={`#${id}`} key={id} aria-current={index === chapterIndex ? 'step' : undefined}>{label}</a>)}
          </nav>
          <span className="showcase-progress" aria-hidden="true" />
          <span className="showcase-status">Still preview</span>
          <div className="showcase-actions">
            <span>Drag to orbit</span>
            <button type="button" onClick={() => updateMotion.current?.(!playing)} aria-label={playing ? 'Pause scene motion' : 'Play scene motion'}>
              {playing ? <><span aria-hidden="true">Ⅱ</span> Pause</> : <><span aria-hidden="true">▷</span> Play</>}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
