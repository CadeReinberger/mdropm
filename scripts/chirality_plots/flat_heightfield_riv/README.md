# Flat-heightfield first-principles RIV bundle

The included `data.pkl` is the new kinematic chirality map generated from the
new flat-heightfield ALE trajectory in `results.pkl`. It is a NumPy array,
ready for the included copy of `riv_code.py`. The original files outside this
folder have not been overwritten.

## Replot the included data

Only NumPy, SciPy, and Matplotlib are needed; FEniCS is not required to replot.
From this folder:

```bash
python riv_code.py
python riv_og.py
```

These commands save PNGs and line-scan CSVs to `outputs/`, without opening
windows. They also work when invoked by absolute path from another directory.
Use `--show` for interactive windows or `--output-dir PATH` for another folder.
`requirements.txt` lists the plotting dependencies.

## Data and coordinate conventions

- `data.pkl`: 1000 x 1000 array, **minus alpha in degrees**, matching the
  existing `chirality_map.py` export convention. Columns increase in simulation
  x from -1.5 to 1.5 mm; rows increase in simulation y from 0 to 3.25 mm.
  NaN means no deposited prediction at that location.
- `map_fields.npz`: the same array, physical grid coordinates, positive alpha
  in degrees, and explicit coverage masks.
- `predictor_samples.npz`: positions, signed speeds, and alpha at every saved
  time and material node before spatial interpolation.
- `results.pkl`: raw trajectory with `out_t` and `out_x`. Each state contains
  all x coordinates, then all y coordinates, then all contact angles in radians.
- `setup.pkl`, `manifest.json`, `checkpoint.pkl`: serialized inputs, full
  numerical/physical parameters, source hashes, and resumable solver state.
- `validation.json`, `map_metadata.json`, and `environment.json`: numerical
  checks, map provenance/coverage, and the environment used.
- `Tom_Science_symmetric_linear_transposed.csv`: unchanged experimental input
  used by the included `riv_og.py`.

The RIV copies retain the original flips, transpose, heatmap sign, scan paths,
black guide curve, and 0-to-4 display registration. That registration stretches
this 3 x 3.25 mm simulation crop into a 4 x 4 image; use `map_fields.npz` and
`outputs/chirality_map.png` for actual simulation coordinates. The inherited
heatmap and line-scan display transforms are retained separately; do not use
the overlaid RIV curves as a new physical-coordinate calibration.

`outputs/riv_code_colormap.png` retains the original +/-0.15 color scale,
which clips much of the predicted angle amplitude. The additional labeled
`outputs/riv_code_degrees.png` uses the full predicted range. The numerical
pickle and CSVs are never clipped or rescaled to fit the color scale.

The predictor is `canting_estimator.normal_motion_curvature`, using the same
centered material-node difference as the ODE. It retains the existing
K_alpha = 0.2 mm; there is **no fitted amplitude calibration**, and these values
are angles, not calibrated experimental g-factors. Linear interpolation uses
only actual swept cells, with reflected triangulation diagonals to preserve
symmetry. No spatial smoothing, time-spline acceleration, or extrapolation
into unswept liquid is used.

## Run setup and reproduction

The physical height field is exactly h(x,y) = 0.1 mm throughout the 10 x 10 mm
cell, corresponding to the study's 200 micrometre full plate-gap convention.
The initial drop radius is 3 mm, with contact angle theta_r = 50 degrees.
Only the top outer gas boundary is open; the other three are no-flux walls.
Current material constants and the 0.001 rad smoothed contact-angle law are
retained. The run uses 80 material nodes, cubic FEM fields, quadratic mesh
geometry, 0.30 mm mesh size, and BDF tolerances 1e-6/1e-8, saving every 0.01
model-time unit. Exact left/right symmetry is enforced as in `canting_refine.py`.

Long runs rebuild meshes between three-unit integration segments. Within a
segment, topology remains fixed and gradients use L2 projection. Rebuilt
meshes map polar reference positions back to material labels on the receded
boundary. Every new reference mesh is checked against constant, x-linear,
and y-linear manufactured pressure solutions.

The source snapshot under `reproduction/` includes the solver dependencies in
the same directory layout as the original repository. In a compatible
FEniCSx 0.9 environment, run from that directory:

```bash
python -B scripts/chirality_plots/flat_heightfield_run.py --duration 24 --output-dir new_run
python -B scripts/chirality_plots/flat_heightfield_map.py --results new_run/results.pkl --output-dir new_map
```

The driver resumes matching checkpoints and rejects incompatible settings.
The shipped trajectory was computed as a fresh three-unit run using
`canting_refine.py`, followed by segmented continuation to 21 and then 24.
Those seed manifests/hashes are nested in `manifest.json`; the full trajectory
is retained in the single `results.pkl`.

## Completed run and checks

The final trajectory reaches time **24** with **2,401 saved states**. Its top
contact-line point is at y = -0.03867 mm, below the map crop. Every grid point
inside the initial drop in the requested crop has been swept and has a valid
prediction. The remaining 11.79% of the rectangular raster lies outside the
initial drop and is NaN. The drop still has 14.72% of its initial area outside
the crop; this is a complete run for the requested map, not complete evaporation.

The map range is **-2.33532 to +2.33532 degrees** in the stored sign convention.
The fresh first three time units match the previous same-settings refined
reference exactly. All 10 predictor/map tests passed. Manufactured pressure
checks passed at every rebuilt reference mesh; the minimum liquid mesh corner
area ratio over the long run was 0.443 (the solver rejects ratios <= 0.15).

Repeating the last three time units with mesh rebuilding every 1.5 instead of
3 units changes the map by at most **0.01022 degrees**, with RMS change
**0.000410 degrees** on a 301 x 326 comparison grid. The largest final position
change is **0.0000575 mm**. This is a focused rebuilding-sensitivity check,
not a full 160-node/finer-mesh convergence study or an absolute physical error
bound. See `validation.json` for the complete numbers.

The simulation chain took about 10.4 minutes of solver wall time, excluding
the separate checks and plotting. `SHA256SUMS` identifies every bundled file.
