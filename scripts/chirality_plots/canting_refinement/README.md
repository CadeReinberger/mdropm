# Refined canting calculation

`../cant_clean.py` re-analyzes the saved refined trajectories and writes
`../canting_clean.png`, `../canting_clean.csv`, and `../canting_clean_fit.json`.
It does not need FEniCS or launch simulations. The previous figure/data/fit are
preserved in `baseline/`; the original production directories are untouched.

## Results of this saved sweep

- 24 production points, versus 9 previously; every point is an independent
  80-node simulation, not a value sampled from the fitted line.
- Maximum raw-slope change at the checked heights: **1.3183%** for 80 → 160
  nodes, **0.01081%** for mesh size 0.30 → 0.20 mm, and **0.00858%** for
  tenfold tighter ODE tolerances. These are observed differences, not rigorous
  bounds or the displayed error bars.
- The presentation fit has RMS residual **0.00307 degrees/mm**. The broad,
  shallow maximum near 320 µm and gentle subsequent decline are retained.
- Compared with the preserved, separately calibrated old curve, the 200-µm
  value changed from 0.24570 to **0.30021 degrees/mm**; the 600-µm value changed
  from 1.00681 to **0.96424 degrees/mm**. Common points from 220 µm upward
  differ by less than 4.3%. The fitted 400-µm value is exactly 1 degree/mm;
  its independent simulated point is **1.00172 degrees/mm**.
- Thirteen focused and existing regression tests pass. The old-data analysis
  reproduces the preserved nine values and error bars unchanged. Re-running
  the default simulation command successfully skips all 24 completed cases.

`../cant_send.py` and `../comp_send.py` are refreshed, small, **frozen exports**
of these 24 values, and their PNGs have been regenerated. The experimental
data and seven-tape height convention in `comp_send.py` are unchanged. These
standalone scripts intentionally do not read the simulation files; future
reruns of the solver/plotter do not automatically rewrite their hardcoded data.
Their previous versions are also preserved in `baseline/`.

## Reproduce

From `scripts/chirality_plots`, using the existing FEniCS environment:

```bash
/home/wcr/.conda/envs/fenicsx-env/bin/python canting_refine.py --workers 6
python cant_clean.py
python canting_refinement_report.py
```

The default sweep has 24 **independent simulations** at displayed front heights
200, 202, 204, 208, 212, 216, 220, 225, 230, 235, 240, 250, 260, 275, 290,
310, 330, 360, 400, 420, 450, 500, 550, and 600 µm. The runner's `--ratios`
argument means `h_front/h_back - 1`, so displayed height is `200*(1+ratio)` µm.
This differs from the fit coordinate `r=(h_front-h_back)/(h_front+h_back)`.

Completed cases are skipped, interrupted cases resume their atomic checkpoints,
and incompatible saved manifests are rejected. `--workers` runs separate serial
FEM processes; do not run this driver with `mpirun`. All cases retain their raw
trajectories, parameters, FEM source hash, checkpoint, and timing/mesh checks.

To reproduce the isolated checks (also resumable):

```bash
/home/wcr/.conda/envs/fenicsx-env/bin/python canting_refine.py --nodes 160 --ratios 0,0.1,0.3,1,2 --workers 3 --output-dir canting_refinement/pilot160
/home/wcr/.conda/envs/fenicsx-env/bin/python canting_refine.py --lc 0.2 --ratios 0.1,1 --workers 2 --output-dir canting_refinement/mesh80
/home/wcr/.conda/envs/fenicsx-env/bin/python canting_refine.py --rtol 1e-7 --atol 1e-9 --ratios 0.1,1 --workers 2 --output-dir canting_refinement/time80
python -m pytest -q test_canting_refinement.py
```

## What changed and why

1. **Remove remeshing discontinuities.** The old solver rebuilt its mesh at
   every ODE evaluation and interpolated a discontinuous element gradient into
   a continuous space. In a 20-node diagnostic, a displacement of just
   `1e-8 mm` changed the RHS by about `0.265`. The new solver builds each mesh
   once, moves its coordinates continuously, and uses an L2 gradient projection.
   The gas equation `div(h grad(w))=0`, liquid equation
   `div(h^3 grad(p))=0`, material parameters, and external boundary conditions
   are unchanged. Liquid boundary values use periodic cubic interpolation.
   Exact left/right symmetry is enforced in the RHS for this symmetric setup.
2. **Increase resolution.** Production uses 80 contact-line nodes (previously
   20), cubic FEM fields, mesh size 0.30 mm, BDF tolerances `1e-6/1e-8`, and
   301 saved samples over the same three model-time units. Separate 160-node,
   0.20-mm-mesh, and tighter-ODE-tolerance checks do not change the displayed
   error bars. The audit compares raw slopes **before calibration**, without
   fitting a separate scale for each resolution.
3. **Avoid second time derivatives near pinning.** If the contact-line velocity
   is `v=sigma*n`, with discrete material tangent `T=D_s r`, then its trajectory
   curvature is `cross(T,D_s v)/(|T|^2*|sigma|)`. The new estimator uses exactly
   the same centered difference `D_s` as the ODE, not a continuum product-rule
   approximation. Curvature is still multiplied by the original arbitrary
   `K_alpha=0.2 mm`; opposite points provide the same mirror-paired secant
   `(alpha_right-alpha_left)/(x_right-x_left)` in degrees/mm.
4. **Average spatially, not by stored sample count.** The Gaussian average is
   centered at `(1.35, 2.7*cos(pi/6)) mm`, as before, with width 0.30 mm. Weights
   now include the swept-area Jacobian `|sigma|*|T| ds dt`. Slowly moving and
   nearly pinned material points therefore do not dominate simply because they
   produce many saved samples in one place. This changes the estimator from a
   time-sample-weighted robust location to a local spatial mean; it is not a
   claim that those two observables are identical. No interpolation or
   smoothing across plate heights is applied to the simulation points.

The averaging window is fixed at `[1/18, 17/6]` model-time units, matching the
old physical window rather than discarding a sampling-dependent number of
frames. Endpoint interpolation and trapezoidal time weights make saved-sample
thinning a meaningful check. Speeds below `1e-6` are excluded because curvature
at a stationary trajectory is undefined.

## Error bars, fit, and calibration

Each error bar is **post-processing sensitivity only**: half the range from
Gaussian widths 0.25/0.30/0.35 mm, stationary-speed cutoffs `1e-7/1e-6/1e-5`,
and retaining every 1st/2nd/4th saved sample. It is not a confidence interval or
a bound on mesh error, ODE error, physical-model error, or amplitude calibration.
The old spline estimator remains available automatically for legacy runs.

The black line is a descriptive exponential-rise fit with an optional gradual
decline. It is separate from the blue computed points; no points are moved to
that curve and no fit-uncertainty band is drawn. Mesh-check discrepancies are
used only as a floor in the fit weights, not in the displayed error bars.

The x-axis retains the requested doubled-height convention. One common scale
factor calibrates the **fitted curve** at displayed 400 µm to 1 degree/mm, i.e.
4 degrees across 4 mm. The measured simulation point there can differ slightly
from the fitted value. The normalization is imposed, not an independently
predicted absolute angle. Its exact factor is in `../canting_clean_fit.json`.

## Evidence and limits

`audit/summary.json` records all completed convergence checks, old/new values,
source hashes, symmetry residual, and minimum mesh area ratios.
`audit/convergence.csv` contains raw per-case comparisons;
`audit/comparison.png` shows the old/new curves and discretization checks.
Manufactured-solution and small-perturbation checks are in `diagnostic*/`.
These test constant pressure, linear `p=x`, and (in the final-geometry checks)
the nontrivial exact vertical solution with `h(y)^3*p'(y)=constant`.

Finite resolution remains a limitation, particularly near depinning. The
moving-mesh guard monitors signed corner-triangle areas and rejects strong
distortion; it is not a proof of positivity of every curved-element Jacobian.
The recorded convergence checks matter more than the visual smoothness of a
fitted line. Exploratory estimator comparisons remain in `analyze_pilots.py`.
