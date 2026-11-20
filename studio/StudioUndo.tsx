'use client';

import { useEffect } from 'react';
import { useStudioStore } from './useStudioStore';

export default function StudioUndo() {
  const canUndo = useStudioStore((state) => state.canUndo);
  const exporting = useStudioStore((state) => state.exporting);
  const undo = useStudioStore((state) => state.undo);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.isComposing || event.shiftKey || event.altKey || !(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== 'z') return;
      const target = event.target;
      if (target instanceof HTMLElement) {
        if (target.isContentEditable || target instanceof HTMLTextAreaElement) return;
        if (target instanceof HTMLInputElement && ['text', 'search', 'number', 'email', 'url', 'tel', 'password'].includes(target.type) && target.dataset.undoText !== 'false') return;
      }
      const studio = useStudioStore.getState();
      if (!studio.canUndo || studio.exporting) return;
      event.preventDefault();
      studio.undo();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  return (
    <button type="button" aria-keyshortcuts="Meta+Z Control+Z" title="Undo the last studio edit (⌘Z / Ctrl+Z)"
      disabled={!canUndo || exporting} onClick={undo}
      className="shrink-0 rounded border border-line px-2.5 py-1.5 text-xs text-muted hover:text-foreground disabled:opacity-40">
      Undo
    </button>
  );
}
