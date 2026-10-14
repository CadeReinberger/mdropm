# mdropm — clean working copy

Self-contained simulation code, selected saved runs, and the scripts needed
for canting plots, colormaps, line scans, and animations. This directory can be
moved out of the original repository. Commands below are run from this folder;
the scripts also accept absolute paths and work from other working directories.

## Environment

For simulations and pressure animations, use the existing environment:

```bash
conda activate fenicsx-env
```

Or create an environment from the versions used to validate this copy:

```bash
conda env create -f environment.yml
conda activate mdropm-clean
```

For plots and contact-line animations without FEniCS:

```bash
python -m pip install -r requirements.txt
```

MP4 output also needs `ffmpeg` on PATH (included in `environment.yml`).
GIF output needs only the Python dependencies; choose `--format gif`.
All plotting and animation commands save files without opening windows.

## Run simulations

```bash
# Full flat-height run, suitable for the included colormap crop (t = 24).
python -B main.py

# A canted run, with displayed front/back gaps of 400/200 microns.
python -B main.py --ratio 1 --duration 3 --output-dir runs/canted

# The refined 24-height canting sweep, saving to runs/canting/height-*.
python -B canting_refine.py --workers 4

# Small smoke run at the production spatial resolution.
python -B main.py --duration 0.02 --output-dir runs/smoke
```

`main.py` defaults to `runs/drop/`. It uses the newer persistent moving-mesh
gas and liquid FEM solver at the top level in `canting_ale_fem.py`, with L2
gradient recovery and mesh rebuilding between three-minute segments.
`--ratio` means `h_front/h_back - 1`; the back half-height is 0.1 mm.
The original geometry is a radius-3-mm circular drop in a 10 × 10 mm cell,
with only the top gas boundary open. These drivers enforce left/right symmetry.

Both runners default to 80 boundary nodes, cubic FEM fields, a 0.30-mm mesh,
and BDF tolerances of `1e-6/1e-8`. Options include `--nodes`, `--lc`, `--order`,
`--rtol`, `--atol`, and `--duration`. The sweep accepts comma-separated
`--ratios`, for example `--ratios 0,0.1,1,2`. Run each process in serial;
use the sweep's `--workers` option for independent processes.

Repeat the same command to resume a checkpoint or skip a completed run.
Changed settings are rejected for an existing output directory: choose a new
directory when changing parameters or duration. Saved results contain `out_t`
and `out_x`; each state stores all x coordinates, then y coordinates, then
contact angles in radians. Times are model minutes and positions are mm.

For custom setups, the geometry/parameter modules and `run.solve_full` are
retained. `time_integrator` now defaults to the moving FEM solver too; its
unsegmented API assumes an initially centered circular drop. Use `main.py`
for long runs with mesh rebuilding. `gas_fem.py` and `liquid_fem.py` supply
the mesh builders and retain their original reference field-solve utilities.

## Canting plot

```bash
# Reproduce the published-style figure from the copied 24-run archive.
python scripts/canting_plotter.py

# Plot a newly completed sweep instead.
python scripts/canting_plotter.py --data-dir runs/canting
```

Outputs are `outputs/canting/canting_clean.png`, `canting_clean.csv`, and
`canting_clean_fit.json`. The default uses the copied 80-node production runs
and five 160-node checks. The two finer-mesh and two tighter-tolerance runs
are also retained with their audit records. The fitted curve is calibrated
to 1 degree/mm at a displayed front gap of 400 microns. Error bars represent
post-processing sensitivity. Details are in [the canting notes](docs/canting_reference.md).

The completed older 20-node runs are kept for comparison. To plot them:

```bash
python scripts/canting_plotter.py \
  --data-dir archived/canting/legacy/canting_production_20 \
  --fallback-dir archived/canting/legacy/canting_production_lsoda_20 \
  --output outputs/canting/legacy.png \
  --csv outputs/canting/legacy.csv \
  --fit-json outputs/canting/legacy_fit.json
```

## Colormaps and line scans

```bash
# Rebuild the physical-coordinate chirality map from the saved full trajectory.
python scripts/flat_heightfield_map.py

# Prediction and experimental RIV-style colormaps plus line scans.
python scripts/riv_code.py
python scripts/riv_og.py

# Process a new run and use its exported map for the RIV-style figures.
python scripts/flat_heightfield_map.py --results runs/drop/results.pkl --output-dir outputs/new_map
python scripts/riv_code.py --data outputs/new_map/data.pkl --output-dir outputs/new_map
```

Defaults read the included completed flat-height run and experimental CSV,
and write to `outputs/colormaps/`. The physical map also exports `data.pkl`,
coordinate/mask arrays, predictor samples, and a trajectory/area figure.
`--grid-size` controls its raster size (default 1000). The crop is
x = [-1.5, 1.5], y = [0, 3.25] mm. The predictor requires a monotonically
receding contact line with the smoothed contact-angle law used by these runners.

`riv_code.py` preserves the existing display registration and ±0.15-degree
color scale; it also saves a labeled figure covering the full predicted range.
Model angles and experimental g-factors are distinct observables. See
[the flat-height notes](docs/flat_heightfield_reference.md) for sign, coordinate,
coverage, and normalization conventions.

## Animations

```bash
# Defaults use the included full flat-height trajectory.
python scripts/full_animation.py
python -B scripts/pressure_field_animation.py

# Animate a new run; adjust frame count, playback rate, and output format.
python scripts/full_animation.py --results runs/canted/results.pkl --frames 120 --fps 24
python -B scripts/pressure_field_animation.py --results runs/canted/results.pkl --frames 60 --format mp4
```

Both scripts save GIF and MP4 files under `outputs/animations/` by default.
Use `--output-dir` to keep animations of different runs separate. The first
colors the contact line by contact angle and does not need FEniCS. The second
uses the refined pressure solve and projected gradient for each frame; it
reads the sibling `setup.pkl` and, for compact sweep setups, `manifest.json`.
`--dpi` controls image resolution; the pressure script also accepts
`--grid-size`, `--setup`, and `--no-streamlines`.

## Included data and provenance

| Location | Purpose |
| --- | --- |
| `archived/canting/production80/` | 24 completed refined canting runs |
| `archived/canting/pilot160/`, `mesh80/`, `time80/` | Saved resolution/tolerance checks |
| `archived/canting/legacy/` | Nine completed old production points and a mesh check |
| `archived/canting/audit/` | Original convergence tables, summary, and comparison |
| `archived/flat_heightfield/` | Completed t=24 trajectory, precomputed RIV map, setup, checkpoint, and validation |
| `data/` | Experimental CSV used by `riv_og.py` |

Archives retain their original data, checkpoints, and manifests. New runs go
under `runs/`; generated plots go under `outputs/`. The old flat-run source
hashes describe the original source layout, so use a new directory for a fresh
run instead of trying to resume that archived run with the relocated driver.
The original source paths and checksums are recorded in `COPY_MANIFEST.json`;
adapted code entries also record the checksum of this copy. No file outside
this directory is needed at runtime.

## Checks

```bash
python -B -m pytest -q -p no:cacheprovider tests
python -B canting_refine.py --diagnostic --ratios 1 --output-dir runs/fem_check
```

The first runs the retained 13 numerical identity and predictor tests without
FEniCS. The second checks manufactured pressure solutions and continuity of
the FEM derivative, and requires the FEM environment.
