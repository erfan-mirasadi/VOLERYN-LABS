import { GRID } from '@/engine/scene';

type Group = { title: string; lead: string; rows: [term: string, detail: string][] };

const GROUPS: Group[] = [
  {
    title: 'Simulate',
    lead: `A ${GRID.join(' × ')} gas simulation burns fuel into heat and soot at sixty steps a second, and a volume raymarcher draws it lit by its own flame. It runs on your GPU through WebGPU, in Chrome, Edge or Safari 26; nothing leaves your machine.`,
    rows: [
      ['Presets', 'Fire, Campfire, Smoke and Explosion.'],
      ['Controls', 'Heat, vorticity, wind and its direction, seed, blue flame, sparks, smoke and width, all live while it runs.'],
      ['Sound', 'Procedural fire sound placed in 3D around the camera and driven by the same controls.'],
    ],
  },
  {
    title: 'Export',
    lead: 'An export re-runs the simulation offline at the size and quality you pick. Alpha comes out of the volume itself, so fire adds and smoke layers without keying.',
    rows: [
      ['Sequences', 'PNG at 8 or 16 bits, premultiplied or straight; or EXR with fire, firelight and smoke as light groups, plus temperature, motion and depth.'],
      ['Flipbooks', '4 × 4, 8 × 4 or 8 × 8 sheets at 128 to 512 px a cell, with motion vectors for Unreal or Unity.'],
      ['Video', 'H.264 MP4.'],
      ['Sizes', '1280 × 720 to 3840 × 2160, or square and vertical, at 24 to 60 fps for 2 to 20 seconds. Fire and smoke can loop seamlessly.'],
      ['Output', 'Chrome and Edge write into a folder you pick; other browsers download a zip.'],
    ],
  },
];

export default function Specs() {
  return (
    <section className="flex flex-col gap-20 border-t border-line py-20 lg:gap-28 lg:py-28">
      {GROUPS.map(({ title, lead, rows }) => (
        <div key={title} className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,34rem)] lg:gap-16">
          <div className="flex max-w-xl flex-col gap-4">
            <h2 className="text-3xl font-semibold tracking-tight">{title}</h2>
            <p className="text-lg leading-relaxed text-muted">{lead}</p>
          </div>
          <dl className="divide-y divide-line border-y border-line">
            {rows.map(([term, detail]) => (
              <div key={term} className="grid gap-1 py-4 sm:grid-cols-[8rem_1fr] sm:gap-6">
                <dt className="text-sm text-muted">{term}</dt>
                <dd className="leading-relaxed">{detail}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </section>
  );
}
