# Pyro model and validation

Voleryn uses an art-directable volumetric fluid model for fire, smoke, and explosions. Its numerical fields and optical model have physical motivation; the resulting temperatures, reaction rates, and light levels are not calibrated measurements of a particular fuel or real explosion.

## Fluid and combustion

`engine/simulation.ts`, `engine/simulation-shaders.ts`, and `engine/pressure*.ts` implement a staggered velocity grid, midpoint backtracing, bounded MacCormack advection, and multigrid pressure projection. Projection targets combustion/blast expansion rather than zero divergence everywhere. The floor and object colliders block normal flow; the upper boundary is approximate and has a time-based damping region. The general advection/force/projection approach and GPU fluid representation are discussed in [GPU Gems 3, chapter 30](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids).

The scalar texture stores soot density, normalized thermal energy, unburned fuel, and the current burn rate. Fuel reacts above an ignition threshold. Fuel concentration and its gradient provide a reduced entrainment proxy; explosions additionally cap fuel consumption per second by that proxy. No oxygen or chemical species field is solved, and pressure projection is not a compressible shock-wave simulation. Seeded, coherent source masks place cool, sooty rich pockets beside mixed ignition pockets. Source ignition stops when the finite charge stops emitting.

Heat diffuses toward its six advected neighbours with a bounded convex blend, `1 - exp(-3 k² dt)`, where `k` is grid width divided by 64. Cooling is exponential with slower residual cooling after the gas stops glowing; it is not a resolved radiative energy equation. Soot production, burnout, buoyancy, vorticity confinement, broad stirring, and fine disturbance remain reduced, adjustable models. The separation of thermal/density fields and shape operators follows the concepts documented by [SideFX: simulation fields and projection](https://www.sidefx.com/docs/houdini/pyro/background.html) and [pyro look development](https://www.sidefx.com/docs/houdini/pyro/lookdev), not an implementation of Houdini's solver.

## Wind and the moving explosion window

Wind is ambient velocity in **base cells per second**: 64 cells span one box width. It is not metres per second without a chosen shot scale. Both horizontal boundaries prescribe matching ambient normal flux; side pressure uses a Neumann condition. Drag acts relative to ambient flow, allowing pressure to deflect it around object colliders. Incoming air carries no fuel or soot.

An explosion's finite computational window follows horizontal ambient wind. Stored velocities remain world velocities; backtracing and particle position updates use `worldVelocity - windowVelocity`. World origin advances by `windowVelocity * dt / gridWidth`; rendering translates the box by that origin. The source and brief blast remain anchored in world space. Initial ambient velocity is set explicitly on the first explosion step. Clear/replay resets the origin, and preview rearming clears before starting another charge. Fire, smoke, and object modes retain stationary domains. Explosion seamless looping is disabled.

This prevents sustained wind from carrying the entire explosion through a stationary side wall. The window has finite extent and does not grow vertically or follow arbitrary separated lobes.

## Light, colour, and controls

`engine/blackbody.ts` integrates relative blackbody spectra at 5 nm intervals, converts through the CIE XYZ observer, then produces linear sRGB/Rec.709 values. It uses the asymmetric Gaussian observer fit from [Wyman, Sloan, and Shirley (2013), equation 4](https://jcgt.org/published/0002/02/01/paper.pdf). Blackbody spectral emission is described in [PBRT's spectral-distribution chapter](https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions). The runtime interpolates a table from 800–4000 K in 100 K steps.

Thermal energy `h` is mapped to an artistic reference temperature:

`K = 1000 + (referenceK - 1000) * (1 - exp(-max(h, 0))) / (1 - exp(-1))`

The lookup clamps to its table range. At the default reference of 1900 K, energy 1 maps to 1900 K, energy 2 to about 2231 K, and the mapping approaches 2424 K. This bounded calibration avoids interpreting normalized energy as arbitrarily hot measured Kelvin. Relative radiance is normalized to red at 1900 K; a shared gain calibrates direct and indirect fire. It is not absolute photometry.

Hot air alone does not supply flame opacity. Soot and active reaction determine incandescent extinction; ray integration also accounts for smoke absorption/scattering, self-shadowing, and approximate fire illumination. Flow-carried detail perturbs the volume coherently. Sparks use world velocity, gravity, drag, finite cooling lifetimes, and shutter-dependent trails.

Use controls in this order:

1. **Source:** Fuel amount, source width, heat release, soot yield, burn rate, cooling, and seed determine the simulated burn.
2. **Motion:** Wind/direction/gusts, lift, broad swirls, vortex strength, fine breakup, and expansion shape transport.
3. **Shading:** Reference temperature, exposure in stops, smoke density, flame body, and flow detail change appearance. Key strength/direction and ambient smoke light help match the plate. Soot yield changes simulated material; smoke density changes optical extinction.
4. **Sparks:** Amount, size, and lifetime control a separate particle population. Size also changes aerodynamic response.

For shot finishing, prefer scene-linear, premultiplied Rec.709/D65 EXR sequences. Beauty is the sum of fire, firelight, and smoke groups; depth, forward motion, and normalized temperature are supplied separately. EXR layers contain the effect alone and are not cut by the plate's depth holdout. Display PNG/MP4 outputs already contain the display transform. Add plate light spill, reflections, lens response, grain, and any detailed roto in the compositor; automatic depth holdout is not a substitute for those passes.

## Development checks

Run `npm run dev`, then open `/lab` in a WebGPU-capable browser. The route returns not found outside development.

- **Flow diagnostics** reads actual GPU fields: positive/negative wind, flow around a solid, positive/negative moving-window tracer cases, and maximum motion/source controls.
- **Run checks** exercises deterministic rendering, finite output, alpha bounds, and wind response through the real offline renderer.
- Use **Render still** at several times, then **Render 5 s** with the intended grid, samples, shutter, and footage. Inspect the downloaded MP4, checkpoint PNGs, and render metadata for colour, temporal behaviour, and boundary loss. The lab's optional plate has no depth holdout; test that separately in the studio.
- Match with Draft (`128 × 192 × 128`) before delivery with Final (`256 × 384 × 256`). A passing preview or typecheck does not validate a final-grid export.

### Recorded verification, 2026-11-19

All six actual GPU flow cases passed without validation errors. At ±60 cells/s, the moving-window inert tracer remained at local centroid `[32, 40.32, 32]`, while world origin reached ±0.9375 box widths after one second. These cases produced no heat and zero measured projection residual. The maximum-control case remained finite, with maximum speed about 170.8 base cells/s and maximum residual about 0.1484 cells/s. Residual readback excludes the upper 20% damping/boundary region. These are regression checks, not a proof of general stability or physical accuracy.

The new moving-window implementation completed a five-second render on the `256 × 384 × 256` grid at 1920 × 1080, 30 fps (150 frames), 32 samples, and 0.5 shutter. Settings included seed 59, wind 32 at 40°, heat release 1.44, soot yield 1.67, width 1.4, blue contribution 0.25, and sparks 0.03. Every raw output plane and the display plane remained finite, alpha stayed bounded, maximum fire value was 3.078125, and no GPU warnings/errors were reported. The run took approximately 314 seconds.

Smoke and fire remained visible at three and five seconds without the earlier fixed-box clipping. Framing was widened to show physical wind drift, and no footage was used, so this is not a matched-camera or matched-plate comparison with the earlier test. Near the final frame, the moving plume partly leaves the lower-right camera frustum as it approaches the camera; this is distinct from losing the field at a simulation boundary. An earlier finite five-second render had exposed the stationary-domain clipping and was not sufficient validation of this fix.

Render regression checks also passed deterministic repeats, signed wind response (88.54-pixel centroid separation at 640-pixel width), and finite appearance extremes at reference temperatures of 1000 K and 3000 K with +5 EV exposure. The EXR beauty uses Float32 channels; a round-trip test preserved the HDR value 120001. These checks establish the tested numerical/export behavior, not physical accuracy, complete format parity, or automatic integration into arbitrary footage.

A separate 3840 × 2160 still at 32 samples on the Final grid also completed with finite planes and bounded alpha. It used the default explosion energy, no wind, and a 0.7-second warmup; this validates a 4K still, not a complete 4K clip.

A neutral opaque plate fixture exercised the actual file-to-plate and GPU composite path. The four background corners were preserved within one 8-bit code value and the composited PNG was fully opaque. This is a compositing regression check, not a match to the original live-action shot. Studio numeric entry committed 0.75 correctly, Escape cancelled a later edit, and Reset restored the default.

CPU checks of the actual flipbook fitting functions included positive, negative, and diagonal window travel, with the requested frame margin, and RGB-only content with zero alpha. The latter correctly refitted the camera rather than discarding additive particles.
