# Canting method and archived convergence evidence

Adapted from the original `scripts/chirality_plots/canting_refinement/README.md`.
Use the commands in [the clean README](../README.md) with this directory layout.
The original trajectories and audit records are in `../archived/canting/`.

The retained sweep contains 24 independent 80-node simulations. The saved
checks found maximum raw-slope changes of 1.3183% for 80 to 160 boundary nodes,
0.01081% for mesh size 0.30 to 0.20 mm, and 0.00858% for tenfold tighter ODE
tolerances. These are observed differences, not rigorous error bounds.
The original audit summary and convergence CSV are retained unchanged.
Their source paths refer to the original repository; the raw data now live
under `archived/canting/{production80,pilot160,mesh80,time80}`.

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
predicted absolute angle. Its exact factor is in the regenerated
`outputs/canting/canting_clean_fit.json`.

## Limits

The moving-mesh guard monitors signed corner-triangle areas and rejects strong
distortion; it is not a proof of positivity of every curved-element Jacobian.
The 160-node checks remain particularly relevant near depinning. The copied
legacy runs can be re-analyzed independently; old and refined estimators must
not be mixed into one primary curve.
