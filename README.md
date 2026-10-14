# mdropm

Simulation Code for A Contact-Line and Evaporation Driven Model of a Drop Drying Between Canted Plates

## Environment

```bash
conda env create -f environment.yml
conda activate mdropm-clean
```

For plots and contact-line animations (without FEniCS)

```bash
python -m pip install -r requirements.txt
```

MP4 output also needs `ffmpeg` on PATH (included in `environment.yml`).
GIF output needs only the Python dependencies; choose `--format gif`.
All plotting and animation commands save files without opening windows.

## Run simulations

```bash
# Full flat-height run.
python -B main.py

# A canted run, with displayed front/back gaps of 400/200 microns.
python -B main.py --ratio 1 --duration 3 --output-dir runs/canted

# The refined 24-height canting sweep, saving to runs/canting/height-*.
python -B canting_refine.py --workers 4
```

`main.py` defaults to `runs/drop/`. It uses the moving-mesh
gas and liquid FEM solver at the top level in `canting_ale_fem.py`, with L2
gradient recovery and mesh rebuilding between three-minute segments.
`--ratio` means `h_front/h_back - 1`; the back half-height is 0.1 mm.
The original geometry is a radius-3-mm circular drop in a 10 × 10 mm cell,
with only the top gas boundary open. 

Both runners default to 80 boundary nodes, cubic FEM fields, a 0.30-mm mesh,
and BDF tolerances of `1e-6/1e-8`. Options include `--nodes`, `--lc`, `--order`,
`--rtol`, `--atol`, and `--duration`. The sweep accepts comma-separated
`--ratios`, for example `--ratios 0,0.1,1,2`. Run each process in serial;
use the sweep's `--workers` option for independent processes.

For custom setups, use the geometry/parameter modules and `run.solve_full` directly.

## Canting plot

```bash
# Reproduce the canting plot we make
python scripts/canting_plotter.py
```

Outputs are `outputs/canting/canting_clean.png`, `canting_clean.csv`, and
`canting_clean_fit.json`. 

## Colormaps and line scans

```bash
# Rebuild the physical-coordinate chirality map from the saved full trajectory.
python scripts/flat_heightfield_map.py

# Prediction and experimental colormaps plus line scans.
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
Use `--output-dir` to keep animations of different runs separate. 
