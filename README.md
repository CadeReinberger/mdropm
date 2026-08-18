# mdropm
Simulation Code for A Contact-Line and Evaporation Driven Model of a Drop Drying Between Canted Plates

## Canting sweep

Regenerate the chirality-versus-height plot from the corrected production runs:

```bash
python scripts/chirality_plots/canting_plotter.py
```

This writes `scripts/chirality_plots/canting_clean.png`, the underlying values
to `canting_clean.csv`, and fit parameters/covariance to
`canting_clean_fit.json`. Error bars combine trajectory-analysis sensitivity
with the relative discrepancy measured by the finer-mesh check. The dark curve
is a weighted fit of
`baseline + amplitude*(1-exp(-(height_ratio/scale)^shape))`; positive
amplitude, scale, and shape guarantee a smooth monotone saturation response,
but no simulated points are deleted or moved. The orange marker is a
finer-mesh rerun.

The production workflow also fixes two numerical/modeling defects in the old
sweep: the plate half-length and drop radius were swapped during contact-angle
initialization, and the piecewise contact-angle speed law had derivative jumps
that made stiff integration branch-dependent. The latter is regularized over
0.001 rad (0.057 degrees).

New canting sweeps are checkpointed after each outer integration interval:

```bash
python canting_runner.py --output-dir canting_runs
```

Run that same command after an interruption. Completed height values are
skipped and incomplete values resume from their atomically written
`checkpoint.pkl`. Use `--ratios 0,0.5,1,1.5,2` to select raw front/back height
changes and `--nodes 40` for a higher-resolution sweep.

Use `--profile production_lsoda` as the robust fallback if BDF spends too long
at a contact-angle transition. Both profiles use the same tolerances and are
safe to stop and resume. `--profile mesh_reference` uses a finer FEM mesh for
selected convergence checks.
