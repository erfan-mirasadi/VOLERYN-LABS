const LAYERS: [layer: string, channels: string, holds: string][] = [
  ['fire', '.R .G .B', 'Everything that glows: the flame and its sparks'],
  ['firelight', '.R .G .B', 'The flame’s light on the smoke and on solids'],
  ['smoke', '.R .G .B', 'The key light and the sky on the smoke'],
  ['beauty', 'R G B A', 'The three groups summed, with coverage'],
  ['temperature', '.Y', 'Heat along the ray, for grading the flame'],
  ['forward', '.u .v', 'Motion in screen space'],
  ['depth', 'Z', 'Distance from the camera, in full float'],
];

const FORMATS: [term: string, detail: string][] = [
  ['Sequences', 'PNG at 8 or 16 bits, premultiplied or straight. EXR in half float, with the camera matrices, frame rate and Rec. 709 chromaticities in the header.'],
  ['Flipbooks', '4 × 4, 8 × 4 or 8 × 8 sheets at 128 to 512 px a cell, looped seamlessly, with motion vectors for Unreal or Unity.'],
  ['Video', 'H.264 MP4.'],
  ['Size and length', '1280 × 720 to 3840 × 2160, square or vertical, at 24 to 60 fps for 2 to 20 seconds. Fire and smoke can loop.'],
  ['Where it goes', 'Chrome and Edge write into a folder you pick; other browsers download a zip.'],
];

const LABEL = 'font-mono text-[11px] uppercase tracking-[0.14em] text-muted';

export default function Export() {
  return (
    <div className="flex flex-col gap-12">
      <table className="w-full border-collapse text-sm">
        <caption className={`pb-4 text-left ${LABEL}`}>EXR layers in every frame</caption>
        <thead>
          <tr className={LABEL}>
            <th className="pb-3 text-left font-normal">Layer</th>
            <th className="pb-3 text-left font-normal">Channels</th>
            <th className="pb-3 text-left font-normal">Holds</th>
          </tr>
        </thead>
        <tbody>
          {LAYERS.map(([layer, channels, holds]) => (
            <tr key={layer} className="border-t border-line">
              <td className="py-3 pr-4 font-mono text-xs">{layer}</td>
              <td className="whitespace-nowrap py-3 pr-4 font-mono text-xs text-muted">{channels}</td>
              <td className="py-3 text-muted">{holds}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="flex flex-col gap-5">
        {FORMATS.map(([term, detail]) => (
          <div key={term} className="grid gap-1 text-sm sm:grid-cols-[7rem_1fr] sm:gap-4">
            <dt className={`${LABEL} sm:pt-1`}>{term}</dt>
            <dd className="leading-relaxed text-muted">{detail}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
