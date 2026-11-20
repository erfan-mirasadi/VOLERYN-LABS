# Pyro model and validation

Voleryn uses an art-directable volumetric fluid model for fire, smoke, and explosions. Its numerical fields and optical model have physical motivation; the resulting temperatures, reaction rates, and light levels are not calibrated measurements of a particular fuel or real explosion.

## Fluid and combustion

`engine/simulation.ts`, `engine/simulation-shaders.ts`, and `engine/pressure*.ts` implement a staggered velocity grid, midpoint backtracing, bounded MacCormack advection, and multigrid pressure projection. Projection targets combustion/blast expansion rather than zero divergence everywhere. The floor and object colliders block normal flow; the upper boundary is approximate and is kept outside occupied gas by adaptive padding. There is no upper material sponge deleting the plume. The general advection/force/projection approach and GPU fluid representation are discussed in [GPU Gems 3, chapter 30](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids).

The scalar texture stores soot density, normalized thermal energy, unburned fuel, and the current burn rate. Fuel reacts above an ignition threshold. Fuel concentration and its gradient provide a reduced entrainment proxy; explosions additionally cap fuel consumption per second by that proxy. No oxygen or chemical species field is solved, and pressure projection is not a compressible shock-wave simulation. Seeded, coherent source masks place cool, sooty rich pockets beside mixed ignition pockets. Source ignition stops when the finite charge stops emitting.

Heat diffuses toward its six advected neighbours with a bounded convex blend, `1 - exp(-3 k² dt)`, where `k = cellsPerUnit / 64`. Sampling density is independent of the adaptive grid dimensions. Cooling is exponential with slower residual cooling after the gas stops glowing; it is not a resolved radiative energy equation. Soot production, burnout, buoyancy, vorticity confinement, broad stirring, and fine disturbance remain reduced, adjustable models. The separation of thermal/density fields and shape operators follows the concepts documented by [SideFX: simulation fields and projection](https://www.sidefx.com/docs/houdini/pyro/background.html) and [pyro look development](https://www.sidefx.com/docs/houdini/pyro/lookdev), not an implementation of Houdini's solver.

## Adaptive bounds, wind, and sampling density

Wind is ambient velocity in **base cells per second**: 64 base cells span one world unit. It is not metres per second without a chosen shot scale. Both horizontal boundaries prescribe matching ambient normal flux; side pressure uses a Neumann condition. Drag acts relative to ambient flow, allowing pressure to deflect it around object colliders. Incoming air carries no fuel or soot.

All modes now inspect active material on the GPU and enlarge the domain horizontally and upward before it reaches the padded boundary. The probe tracks `max(soot, 0.25 * heat, 0.1 * fuel) >= 0.003`, plus velocity component bounds; padding includes a prediction interval and a minimum spatial margin. Sources and object colliders remain anchored in world space. This follows the reference-field bounds plus velocity padding approach described in [SideFX's resizing documentation](https://www.sidefx.com/docs/houdini/pyro/sparsity.html); this implementation remains dense rather than sparse.

`cellsPerUnit` determines world cell size independently of grid dimensions. Growth is aligned to 16-cell tiles. At capacity, empty tails can be released with volume-ratio hysteresis; the minimum source envelope stays anchored to avoid resizing for every small hotspot change. GPU probes first reduce 64 cells in workgroup memory before updating global bounds, rather than contending on global atomics for every voxel.

Reallocation preserves the current scalar field, both advected detail fields, world velocity, and particle order/age/lifetime. Same-density, tile-aligned growth maps to existing texel centres, preserving samples up to texture/half-float precision. Capacity-driven coarsening uses trilinear resampling; it can smooth detail and is not exactly mass-conservative. Velocity and detail displacement values are converted to the new cell units. New cells contain ambient velocity and zero material. Fine pressure is recomputed by the subsequent projection rather than copied as an exact checkpoint, so a resized continuation is not promised to be bitwise identical to a preallocated solve.

The dense allocation has a finite cell/device budget. When growth exceeds it, the full active envelope is retained at a lower sampling density; the HUD and export domain history report that effective density. If even a usable density cannot be allocated, the engine reports a capacity error instead of continuing with a hidden cut. Numerical traces below the activity threshold may be released. This is not an infinite, sparse, or uniformly high-resolution domain. The physical floor stays fixed; arbitrary user walls and a compressible blast solver are not implemented.

Explosions additionally use a window following horizontal ambient wind. Stored velocities remain world velocities; backtracing and particle position updates use `worldVelocity - windowVelocity`. World origin advances by `windowVelocity * dt / cellsPerUnit`. The finite source and brief blast stay anchored in world space; initial ambient velocity is set explicitly. Preview rearming returns to a fresh charge, and explosion seamless looping is disabled. Continuous sources stay in the adaptive domain rather than moving away with their plume. Loop exports reserve a conservative remaining-clip envelope before keeping their fixed-layout crossfade states.

Identical compute pipelines are cached per GPU device, and procedural noise is baked once on the CPU. These caches remove repeated setup work during resizing; they do not skip physical warmup steps or restore a cached fluid checkpoint. Export progress reports setup, warmup, loop preparation, and rendered frames. Flipbook fitting and sidecar metadata use the sampled world envelope and actual layout history, not just the requested starting grid.

## Light, colour, and controls

`engine/blackbody.ts` integrates relative blackbody spectra at 5 nm intervals, converts through the CIE XYZ observer, then produces linear sRGB/Rec.709 values. It uses the asymmetric Gaussian observer fit from [Wyman, Sloan, and Shirley (2013), equation 4](https://jcgt.org/published/0002/02/01/paper.pdf). Blackbody spectral emission is described in [PBRT's spectral-distribution chapter](https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions). The runtime interpolates a table from 800–4000 K in 100 K steps.

Thermal energy `h` controls emission on a bounded artistic calibration:

`calibratedK = 1000 + 900 * (1 - exp(-max(h, 0))) / (1 - exp(-1))`

At energy 1 this is 1900 K, at energy 2 about 2231 K, and it approaches 2424 K. The **Flame color** control maps the same energy through its chosen reference Kelvin value for chromaticity, then matches that colour to the calibrated curve's luminance. It therefore changes colour without forcing a large exposure change. **Brightness**, in EV, scales fire emission once for both direct and indirect contributions; **Exposure** grades the whole effect. The lookup clamps to 800–4000 K. These are art-direction controls, not measured temperatures or absolute photometry.

Hot air alone does not supply flame opacity. Soot and active reaction determine incandescent extinction; ray integration also accounts for smoke absorption/scattering, self-shadowing, and approximate fire illumination. Flow-carried detail perturbs the volume coherently. Sparks use world velocity, gravity, drag, finite cooling lifetimes, and shutter-dependent trails.

Use controls in this order:

1. **Source:** Fuel amount, source width, heat release, soot yield, burn rate, cooling, and seed determine the simulated burn. Seed changes coherent source hotspot arrangement, contour, and broad eddy scale, rather than only shifting flicker; changing it restarts the same deterministic time sequence.
2. **Motion:** Wind/direction/gusts, lift, broad swirls, vortex strength, fine breakup, and expansion shape transport.
3. **Shading:** Flame colour, fire brightness and overall exposure in stops, smoke density, flame body, and flow detail change appearance. Key strength/direction and ambient smoke light help match the plate. Soot yield changes simulated material; smoke density changes optical extinction.
4. **Sparks:** Amount, size, and lifetime control a separate particle population. Size also changes aerodynamic response.

For shot finishing, prefer scene-linear, premultiplied Rec.709/D65 EXR sequences. Beauty is the sum of fire, firelight, and smoke groups; depth, forward motion, and normalized temperature are supplied separately. EXR layers contain the effect alone and are not cut by the plate's depth holdout. Display PNG/MP4 outputs already contain the display transform. Add plate light spill, reflections, lens response, grain, and any detailed roto in the compositor; automatic depth holdout is not a substitute for those passes.

## Development checks

Run `npm run dev`, then open `/lab` in a WebGPU-capable browser. The route returns not found outside development.

- **Flow diagnostics** reads actual GPU fields: positive/negative wind, flow around a solid, positive/negative moving-window tracer cases, and maximum motion/source controls.
- **Domain diagnostics** tests actual GPU mass/centroid, world velocity, detail flow, and particle preservation across negative-origin growth/coarsening; continuous wind crossing the old bound; planner hysteresis; and repeat/new seeds.
- **Run checks** exercises deterministic rendering, finite output, alpha bounds, and wind response through the real offline renderer.
- Use **Render still** at several times, then **Render 5 s** with the intended grid, samples, shutter, and footage. Inspect the downloaded MP4, checkpoint PNGs, and render metadata for colour, temporal behaviour, and boundary loss. The lab's optional plate has no depth holdout; test that separately in the studio.
- Match with Draft (initially `128 × 192 × 128`) before delivery with Final (initially `256 × 384 × 256`). Adaptive dimensions and effective density can change during either job. A passing preview or typecheck does not validate a final-grid export.

### Adaptive-domain verification, 2026-11-19

The actual GPU Domain diagnostics passed all cases with no validation errors:

- Same-density, negative-origin growth preserved scalar mass, world centroid, world velocity, detail flow and particle position/velocity/age/lifetime with zero measured error in the fixture.
- Capacity coarsening preserved the world extent with relative scalar-mass error `0.0000502174` (about 0.0050%) and maximum centroid error `2.79e-7` world units. World velocity, sampled detail flow and particle state had zero measured error.
- A continuous source under wind 60 ran for 96 steps, resized three times to `160 × 96 × 64`, and carried active material to world x `1.40625`, beyond the former right boundary at `0.5`. The new right padding was `0.59375` world units.
- Planner fixtures avoided idle-envelope churn and released empty historical tails while recovering the preferred density.
- Two seed-0 runs produced the same field hash `2508684356`; seed 59 produced `1509133828`. Their normalized field difference was `0.62343897`, demonstrating both repeatability and a materially different source realization in this fixture.

A separate four-second continuous-fire still under wind 40 completed with finite output, peak linear fire value about 7.293 and maximum alpha about 0.9956. Its sampled occupied world bounds were x `[-0.125, 2.78125]`, y `[0, 0.5]`, z `[-0.171875, 0.21875]`. The recorded grid X sizes grew `64 → 96 → 128 → 160 → 192 → 224`, with Y/Z at `96/64` and sampling density unchanged at 64 cells per world unit. This demonstrates retained fire beyond the former side bound in that actual still; it is not a full Final-grid clip or a benchmark of every wind setting.

These tests read actual fields and particle buffers. They validate the specified transfers and trajectories, not arbitrary future wind histories, perfect mass conservation under every coarsening, or a full Final-grid delivery render. The physical floor remains enabled by the current product scope; adaptive growth applies to the sides and top.

### Earlier moving-window verification, 2026-11-19

The results below predate the all-mode adaptive-domain changes. They remain historical evidence for their exact tested revision; they do not validate adaptive growth, capacity resampling, the new seed source, or the updated colour/brightness controls. Current adaptive GPU diagnostics are tracked separately until their actual results are recorded.

All six actual GPU flow cases passed without validation errors. At ±60 cells/s, the moving-window inert tracer remained at local centroid `[32, 40.32, 32]`, while world origin reached ±0.9375 box widths after one second. These cases produced no heat and zero measured projection residual. The maximum-control case remained finite, with maximum speed about 170.8 base cells/s and maximum residual about 0.1484 cells/s. Residual readback excludes the upper 20% damping/boundary region. These are regression checks, not a proof of general stability or physical accuracy.

The new moving-window implementation completed a five-second render on the `256 × 384 × 256` grid at 1920 × 1080, 30 fps (150 frames), 32 samples, and 0.5 shutter. Settings included seed 59, wind 32 at 40°, heat release 1.44, soot yield 1.67, width 1.4, blue contribution 0.25, and sparks 0.03. Every raw output plane and the display plane remained finite, alpha stayed bounded, maximum fire value was 3.078125, and no GPU warnings/errors were reported. The run took approximately 314 seconds.

Smoke and fire remained visible at three and five seconds without the earlier fixed-box clipping. Framing was widened to show physical wind drift, and no footage was used, so this is not a matched-camera or matched-plate comparison with the earlier test. Near the final frame, the moving plume partly leaves the lower-right camera frustum as it approaches the camera; this is distinct from losing the field at a simulation boundary. An earlier finite five-second render had exposed the stationary-domain clipping and was not sufficient validation of this fix.

Render regression checks also passed deterministic repeats, signed wind response (88.54-pixel centroid separation at 640-pixel width), and finite appearance extremes at reference temperatures of 1000 K and 3000 K with +5 EV exposure. The EXR beauty uses Float32 channels; a round-trip test preserved the HDR value 120001. These checks establish the tested numerical/export behavior, not physical accuracy, complete format parity, or automatic integration into arbitrary footage.

A separate 3840 × 2160 still at 32 samples on the Final grid also completed with finite planes and bounded alpha. It used the default explosion energy, no wind, and a 0.7-second warmup; this validates a 4K still, not a complete 4K clip.

A neutral opaque plate fixture exercised the actual file-to-plate and GPU composite path. The four background corners were preserved within one 8-bit code value and the composited PNG was fully opaque. This is a compositing regression check, not a match to the original live-action shot. Studio numeric entry committed 0.75 correctly, Escape cancelled a later edit, and Reset restored the default.

CPU checks of the actual flipbook fitting functions included positive, negative, and diagonal window travel, with the requested frame margin, and RGB-only content with zero alpha. The latter correctly refitted the camera rather than discarding additive particles.
