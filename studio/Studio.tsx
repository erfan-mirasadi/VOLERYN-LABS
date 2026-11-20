import Logo from '@/app/Logo';
import ExportPanel from './ExportPanel';
import FootagePanel, { PlacementPanel } from './FootagePanel';
import SimulationPanel from './SimulationPanel';
import Viewport from './Viewport';
import StudioUndo from './StudioUndo';

export default function Studio() {
  return (
    <div className="fixed inset-0 flex h-dvh min-h-0 flex-col overflow-clip lg:flex-row">
      <Viewport />
      <aside aria-label="Studio controls" tabIndex={0} className="relative flex min-h-0 w-full flex-[1_1_46%] flex-col gap-8 overflow-y-auto overscroll-contain border-t border-line bg-panel p-5 lg:w-80 lg:flex-[0_0_20rem] lg:border-l lg:border-t-0">
        <div className="flex items-center justify-between gap-2">
          <Logo />
          <StudioUndo />
        </div>
        <PlacementPanel />
        <SimulationPanel />
        <FootagePanel />
        <ExportPanel />
      </aside>
    </div>
  );
}
