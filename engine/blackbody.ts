// Planck spectra integrated against the CIE 1931 standard observer, then transformed to linear sRGB.
// The observer uses the asymmetric Gaussian fit in Wyman, Sloan & Shirley (2013), equation 4:
// https://jcgt.org/published/0002/02/01/ . PBRT describes spectral blackbody emission here:
// https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions .
// This small table is built once when the module loads. A ray only interpolates two entries; it never integrates
// a spectrum. Values keep relative visible radiance (red at 1900 K = 1), rather than normalizing each colour,
// so cooling soot loses light before it loses opacity. Five-nanometre bins and 100 K entries are approximations.
// Relative spectral radiance still needs a camera/emitter calibration. Keep direct and indirect fire light in
// the same units; exposure then grades that signal without an arbitrary several-stop indirect-light boost.
export const FIRE_RADIANCE = 1.5;
const MIN_KELVIN = 800;
const MAX_KELVIN = 4000;
const KELVIN_STEP = 100;
const gaussian = (wavelength: number, centre: number, left: number, right: number) => {
  const t = (wavelength - centre) * (wavelength < centre ? left : right);
  return Math.exp(-0.5 * t * t);
};

export function spectralBlackbody(kelvin: number): [number, number, number] {
  const temperature = Math.max(MIN_KELVIN, Math.min(MAX_KELVIN, kelvin));
  let x = 0;
  let y = 0;
  let z = 0;
  for (let nm = 360; nm <= 830; nm += 5) {
    const wavelength = nm * 1e-9;
    // Common 2hc² and wavelength-bin factors cancel in the shared radiance normalization.
    const radiance = 1 / (wavelength ** 5 * Math.expm1(0.01438776877 / (wavelength * temperature)));
    const cx = 0.362 * gaussian(nm, 442, 0.0624, 0.0374)
      + 1.056 * gaussian(nm, 599.8, 0.0264, 0.0323) - 0.065 * gaussian(nm, 501.1, 0.049, 0.0382);
    const cy = 0.821 * gaussian(nm, 568.8, 0.0213, 0.0247) + 0.286 * gaussian(nm, 530.9, 0.0613, 0.0322);
    const cz = 1.217 * gaussian(nm, 437, 0.0845, 0.0278) + 0.681 * gaussian(nm, 459, 0.0385, 0.0725);
    x += radiance * cx;
    y += radiance * cy;
    z += radiance * cz;
  }
  // Very warm spectra extend outside sRGB. Clip only the negative out-of-gamut components, before exposure.
  return [Math.max(0, 3.2406 * x - 1.5372 * y - 0.4986 * z),
    Math.max(0, -0.9689 * x + 1.8758 * y + 0.0415 * z),
    Math.max(0, 0.0557 * x - 0.204 * y + 1.057 * z)];
}

const reference = spectralBlackbody(1900)[0];
const count = (MAX_KELVIN - MIN_KELVIN) / KELVIN_STEP + 1;
const table = Array.from({ length: count }, (_, i) => {
  const rgb = spectralBlackbody(MIN_KELVIN + i * KELVIN_STEP).map((v) => (v / reference).toExponential(7));
  return `vec3<f32>(${rgb.join(', ')})`;
}).join(',\n');

export const BLACKBODY = /* wgsl */ `
const BLACKBODY_TABLE = array<vec3<f32>, ${count}>(
${table}
);
fn blackbodyKelvin(kelvin: f32) -> vec3<f32> {
  let at = (clamp(kelvin, ${MIN_KELVIN}.0, ${MAX_KELVIN}.0) - ${MIN_KELVIN}.0) / ${KELVIN_STEP}.0;
  let low = min(u32(at), ${count - 2}u);
  return mix(BLACKBODY_TABLE[low], BLACKBODY_TABLE[low + 1u], at - f32(low));
}
// The solver stores reduced thermal energy, not Kelvin. A finite-fuel, air-mixed flame has a bounded thermal
// range: energy cannot be extrapolated linearly to arbitrarily hot blackbody radiation. This monotone calibration
// preserves referenceKelvin at unit energy and approaches a finite combustion-temperature ceiling smoothly.
// At the default 1900 K reference: energy 1 -> 1900 K, 2 -> 2231 K, 3 -> 2353 K; ceiling 2424 K.
// This is a reduced-model calibration, not a chemically resolved adiabatic-temperature calculation.
// Art-directed colour temperature is decoupled from radiance. Physical thermal energy still controls
// cooling brightness on the calibrated 1900 K curve. The colour control changes chromaticity at matching
// luminance; the separate Brightness EV control scales emission. At reference 1900 K this is unchanged.
fn blackbodyAt(t: f32, referenceKelvin: f32) -> vec3<f32> {
  let thermal = (1.0 - exp(-max(0.0, t))) / 0.6321205588;
  let calibrated = blackbodyKelvin(1000.0 + thermal * 900.0);
  let colour = blackbodyKelvin(1000.0 + thermal * max(0.0, referenceKelvin - 1000.0));
  let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
  return colour * dot(calibrated, luma) / max(dot(colour, luma), 1e-12);
}
fn blackbody(t: f32) -> vec3<f32> { return blackbodyAt(t, 1900.0); }
fn flame(t: f32) -> f32 { return smoothstep(0.15, 0.65, t); }
`;
