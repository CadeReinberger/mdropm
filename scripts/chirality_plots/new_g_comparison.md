# Code Spec: Three-Way Comparison of the Canting \(g\)-Factor

## Goal

Create a new script

```text
scripts/chirality_plots/canting_threeway.py
```

that produces

```text
scripts/chirality_plots/canting_threeway.png
```

and compares **three estimates of the same canting-response quantity \(g\)** as a function of plate canting / normalized height difference:

1. **Current trajectory-based calculation**: exactly the cleaned calculation currently used for `canting_clean.png` / `canting_plotter.py`.
2. **Large-CASR formula**: the full asymptotic formula derived below, using one initial gas-phase solve for each canting value.
3. **Gas-log-flux approximation**: the simplified formula
   \[
   g \approx \partial_s^2 \log J
   \]
   evaluated at the high-gap pole of the initial circular droplet.

The purpose is diagnostic. Do not tune the new methods to agree with the trajectory calculation. Put all three on the same axes using the same canting values whenever possible.

---

## Existing code to reuse

Do not rewrite the current cleaned trajectory estimator.

Import and reuse functionality from:

```text
scripts/chirality_plots/canting_plotter.py
```

In particular, reuse its run discovery / merge logic and its current `Estimate` values so that the "trajectory" curve in the new plot is numerically the same calculation that underlies the current cleaned canting plot.

Relevant existing concepts/functions include:

- `normalized_height(setup)`
- `analyze_directory(...)`
- `merge_estimates(...)`
- `Estimate`
- the existing primary/fallback run directories

The current normalized canting coordinate is

\[
\xi_h = \frac{h_{\rm front}-h_{\rm back}}
             {h_{\rm front}+h_{\rm back}}.
\]

Use this same x-coordinate for all three curves.

The default simulation directories should remain consistent with `canting_plotter.py`, currently:

```text
canting_production_20
canting_production_lsoda_20
```

The current trajectory-based calculation should retain its existing post-processing and error bars. The two gas-based curves do not need uncertainty bars in the first implementation.

---

# Mathematical specification

## 1. Geometry and notation

The initial droplet is a circle of radius

\[
R=L_d.
\]

The gap half-height is planar:

\[
h(\mathbf x)=a+\mathbf b\cdot\mathbf x,
\]

where \(\mathbf b=\nabla h\) is constant.

Do **not** hard-code whether canting is along `x` or `y`. Existing production runs may use the rotated convention relative to the derivation.

Define

\[
b = |\mathbf b|.
\]

For positive canting, define the **high-gap pole**

\[
\mathbf x_0 = R\frac{\mathbf b}{|\mathbf b|}.
\]

If \(b=0\), choose either pole consistently; by symmetry both new estimates should be zero (up to numerical error).

Let \(s\) denote signed arclength along the initial circle, with \(s=0\) at \(\mathbf x_0\), and let increasing \(s\) follow one fixed orientation around the circle.

At the pole,

\[
h_0=h(\mathbf x_0).
\]

The outward unit normal at the pole is parallel to \(\mathbf b\), so

\[
h_n(0)=b,\qquad h_s(0)=0.
\]

Define

\[
\psi_0 = \arctan b,
\qquad
\chi_0 = \theta_r+\psi_0.
\]

Use the existing `Lambda` and `Lambda_pr` functions from `geo_util.py`; do not make a second independent implementation unless necessary.

---

## 2. Large-CASR active-set equation

On a receding region in the \(k_r\to\infty\) limit,

\[
\theta=\theta_r,
\]

and the contact-line speed \(V\) is determined by mass balance rather than by a finite-slope CASR.

For planar \(h\), the limiting 1-D contact-line equation is

\[
A V_s + M D V = M E,
\]

with

\[
A=\frac{h_s}{1+h_n^2},
\]

\[
M=\frac{2}{h\,\Lambda'(\theta_r+\psi)},
\]

\[
D=1-\Lambda(\theta_r+\psi)h_n,
\]

and

\[
E=k_w w_n-k_p h^2p_n.
\]

For the **initial circular droplet**, the contact-angle initialization is chosen so that Young-Laplace pressure is constant around the interface. Therefore

\[
p_n=0
\]

identically, and hence

\[
E=k_w w_n.
\]

All multiplicative constants in \(E\), including \(k_w\), cancel from the \(g\)-formula below. Thus the new calculations only need the **shape of the gas normal-flux profile**.

Define the positive evaporative-flux-shaped quantity

\[
J(s)=-w_n(s).
\]

The sign of \(J\) is irrelevant inside the ratios as long as it is handled consistently; prefer \(J>0\) at the interface.

---

# Method A: Full large-CASR formula

At the high-gap pole,

\[
A(0)=0.
\]

Define

\[
D_0=1-b\Lambda(\chi_0),
\]

\[
B_0=
\frac{2D_0}
     {h_0\Lambda'(\chi_0)},
\]

and

\[
a_1=A_s(0)
=
-\frac{b}{R(1+b^2)}.
\]

Then define

\[
c_1=\frac{B_0}{B_0+a_1},
\qquad
c_2=\frac{B_0}{B_0+2a_1}.
\]

The purely geometric second derivative is

\[
D_{ss}(0)
=
\frac{1}{R^2}
\left[
b\Lambda(\chi_0)
+
\frac{b^2}{1+b^2}\Lambda'(\chi_0)
\right].
\]

The full large-CASR prediction is

\[
\boxed{
g_{\rm full}
=
c_2
\left[
\frac{J_{ss}}{J}
-
\frac{D_{ss}}{D_0}
\right]
-
c_1^2
\left(
\frac{J_s}{J}
\right)^2
}
\]

with all \(J\)-quantities evaluated at \(s=0\).

Implement this formula directly.

### Flat-gap special case

When \(b=0\),

\[
a_1=0,\qquad D_{ss}=0,
\]

and symmetry should also give \(J_s(0)=0\) and \(J_{ss}(0)\) consistent with zero canting response.

The plotted result at zero canting should be numerically very near zero. Avoid a separate analytic hard-coded zero unless needed to prevent a division/normal-direction ambiguity at exactly \(b=0\).

---

# Method B: Gas-log-flux approximation

The simplified prediction is

\[
\boxed{
g_{\rm flux}
=
\frac{J_{ss}}{J}
-
\left(\frac{J_s}{J}\right)^2
=
\left.\partial_s^2\log J\right|_{s=0}.
}
\]

This uses the exact same gas solution and the exact same fitted values of

\[
J,\quad J_s,\quad J_{ss}
\]

as Method A.

Do **not** independently refit the gas data for Method B. One gas solve and one local flux fit should produce both new estimates.

---

# Gas solve

## Physical setup

For every canting value, solve the gas problem on the **initial circular droplet**, not on a time-evolved droplet:

\[
\nabla\cdot(h\nabla w)=0.
\]

Use the existing gas-phase boundary conditions and physical parameters already implemented in the repository:

- \(w=w_{\rm eq}\) on the droplet,
- no-flux on taped exterior boundaries,
- existing homogeneous Dirichlet/opening condition on the open side.

Use `physical_params.physical_params()` without changing physical constants.

The new asymptotic formulas are independent of the absolute prefactor \(D c_g/c_l\), so there is no need to manipulate `k_w`.

## Reconstruct each heightscape

For each existing simulation/setup used in the trajectory curve, read its `setup.pkl` and reconstruct the same planar heightscape.

Prefer to reuse the exact existing constructor/configuration used to produce the canting production runs rather than infer a new orientation by hand.

The important rule is:

> The gas calculation for a data point must use the same \(h_{\rm front}\), \(h_{\rm back}\), plate dimensions, tape geometry, and canting orientation as the simulation point it is compared against.

Do not assume that "front" corresponds to the x-direction; inspect/reuse the production setup logic.

---

# Computing \(J(s)\), \(J_s(0)\), and \(J_{ss}(0)\)

## First implementation: reuse the existing gas solver

For minimum implementation risk, it is acceptable to use the existing

```python
compute_concentration_gradients(...)
```

machinery on a sufficiently well-resolved initial circular interface.

This should perform **one gas FEM solve per canting value**, returning \(\nabla w\) at all interface control points.

For each circular interface node,

\[
J_j = -\nabla w_j\cdot n_j,
\]

where \(n_j\) is the exact outward normal of the initial circle.

Because the initial interface is known analytically, use exact circle normals rather than finite-differencing node positions.

If the existing recovered-gradient values are too noisy, improve only the **boundary-flux extraction**. Do not introduce time integration.

A more principled optional refinement is to evaluate/project

\[
-h\nabla w\cdot n
\]

or \(-\nabla w\cdot n\) directly on the inner boundary facets and fit that 1-D trace. Since \(h\) is smooth and known, either convention is fine only if the formulas are adjusted consistently. The formulas above are written for

\[
J=-w_n.
\]

---

# Spatial fit for flux derivatives

The quantity requires a second tangential derivative, so do not use a raw 3-point finite difference unless it is only a diagnostic.

Use a smooth local fit of \(J(s)\) around the high-gap pole.

Preferred implementation:

1. Parameterize the circle by signed arclength \(s\in(-\pi R,\pi R]\) centered at the high-gap pole.
2. Select a symmetric neighborhood around \(s=0\).
3. Fit a local polynomial of degree 4 (degree 5 is also acceptable) by least squares.
4. Read
   \[
   J(0),\quad J_s(0),\quad J_{ss}(0)
   \]
   analytically from the polynomial coefficients.

Make the neighborhood configurable.

Suggested defaults:

```python
FLUX_FIT_DEGREE = 4
FLUX_FIT_HALF_WIDTH_NODES = 5
```

meaning roughly 11 points total if the center node is included.

If the initial circular mesh does not place a node exactly at the high-gap pole, rotate/reparameterize the sampled \(s\)-coordinates so the pole is at \(s=0\); do not approximate the pole using the nearest point.

### Stability check

For each canting value, also evaluate the estimate for at least a small set of nearby fit widths, e.g.

```python
half_width_nodes = (4, 5, 6)
```

and print/report the spread as a diagnostic.

Do not use this spread as an error bar in the first plot unless doing so is trivial.

The important thing is to detect cases where \(J_{ss}\) is dominated by FEM/recovery noise.

---

# Units and scaling

Be careful here.

The current trajectory code includes its historical factor

```python
K_ALPHA_MM = 0.2
```

inside the trajectory-curvature definition and then reports a slope in degrees/mm.

The new formulas produce a geometrical

\[
g=\partial_s\kappa
\]

with dimensions of inverse-length-squared before any legacy plotting/scaling convention is applied.

Before plotting, inspect the exact final scaling used by the cleaned canting curve and convert the two new estimates into the **same plotted units/convention**.

Do not silently choose a scale factor to make the curves agree.

Instead:

- document the conversion in a helper such as `convert_g_to_plot_units(...)`;
- apply the same explicit physical/legacy factor to both new methods;
- include a short code comment explaining why the factor is necessary.

If the cleaned curve is not literally the same dimensional \(g\), preserve the existing curve unchanged and label the distinction clearly in code/comments. The intent of this script is partly to test whether the current estimator is in fact approximating this limiting \(g\).

---

# Plot

Produce a **single panel** resembling the current `canting_clean.png` plot, but containing the three calculations.

Suggested labels:

```text
Trajectory (current)
Large-CASR
Gas log-flux
```

Requirements:

- x-axis: same normalized height/canting coordinate currently used by `canting_plotter.py`;
- y-axis: same units and label as the current cleaned plot after explicit conversion of the two new formulas;
- current trajectory points: preserve their current error bars;
- full large-CASR curve: connected markers/line;
- gas-log-flux curve: connected markers/line;
- legend present;
- grid/style should remain close to the existing clean canting plot;
- save to `canting_threeway.png`;
- do not overwrite `canting_clean.png`.

Do not add a fitted saturating-response line unless it is already necessary to reproduce the visual role of the current cleaned curve. The primary goal is to compare the **raw three estimates**.

---

# Command-line behavior

`canting_threeway.py` should run standalone from the repository environment.

Support at least:

```bash
python scripts/chirality_plots/canting_threeway.py
```

Useful optional arguments:

```text
--data-dir
--fallback-dir
--output
--flux-fit-degree
--flux-fit-half-width
--gas-nodes
```

Defaults should be sensible and should match the existing canting analysis where possible.

---

# Suggested code structure

Something approximately like:

```python
def load_comparison_cases(...):
    """Return the unique canting/setup cases and current trajectory estimates."""

def make_initial_problem(setup, ...):
    """Construct the initial circular drop and matching height/tape/problem universe."""

def gas_flux_on_initial_circle(case, ...):
    """One gas solve; return signed arclength samples and J=-dw/dn."""

def fit_flux_derivatives(s, J, ...):
    """Return J0, Js0, Jss0 plus fit diagnostics."""

def large_casr_g(case, J0, Js0, Jss0):
    """Implement the full formula."""

def gas_log_flux_g(J0, Js0, Jss0):
    return Jss0 / J0 - (Js0 / J0)**2

def convert_g_to_plot_units(g):
    """Explicitly reconcile geometric units with the legacy cleaned plot."""

def main():
    ...
```

Reuse existing repository code instead of creating duplicate versions of geometry, physical parameters, `Lambda`, run discovery, or trajectory post-processing.

---

# Performance expectations

This script must **not run the temporal droplet solver** for the two new methods.

For each unique canting value, the new calculation should require approximately:

```text
1 gas FEM solve
+ cheap 1-D flux fitting
```

The current trajectory curve should be read from existing saved simulations.

For the first implementation, remeshing the gas domain once per canting value is acceptable. If runtime is still inconvenient, optimize by reusing the initial gas mesh because the square-minus-circle domain is identical across canting values and only the coefficient \(h\) changes.

Do not optimize this before correctness is established.

---

# Diagnostics to print/save

For each canting value, print one concise line containing at least:

```text
height_ratio
h_front
h_back
b
J0
Js0
Jss0
g_full
g_flux
g_trajectory
```

Also warn if:

- \(J_0\le 0\);
- either denominator \(B_0+a_1\) or \(B_0+2a_1\) is close to zero;
- the local-fit-width sensitivity is large;
- the gas FEM solve or boundary gradient contains non-finite values.

Optionally write these values to

```text
canting_threeway.csv
```

if convenient.

---

# Validation / acceptance criteria

The implementation is complete when all of the following hold.

1. Running `canting_threeway.py` produces `canting_threeway.png`.
2. The trajectory series agrees with the existing cleaned canting calculation to numerical precision because it reuses the same implementation.
3. The two new series require no temporal solve.
4. Method A and Method B use the **same single gas solve and same flux derivative fit** at each canting value.
5. At zero canting, both new predictions are approximately zero within numerical differentiation/FEM error.
6. Changing the local flux-fit half-width modestly does not qualitatively change either new curve.
7. Increasing initial-circle gas resolution changes the new curves substantially less than the difference among the three methods.
8. No arbitrary multiplicative calibration is applied to make the methods match.
9. The script makes it obvious in code which formula corresponds to:
   - the full large-CASR result;
   - the log-flux approximation;
   - the existing trajectory estimate.

---

# Main formulas in one place

For reference, at the high-gap pole of an initial radius-\(R\) circle:

\[
b=|\nabla h|,
\qquad
\psi_0=\arctan b,
\qquad
\chi_0=\theta_r+\psi_0,
\]

\[
D_0=1-b\Lambda(\chi_0),
\]

\[
B_0=\frac{2D_0}{h_0\Lambda'(\chi_0)},
\]

\[
a_1=-\frac{b}{R(1+b^2)},
\]

\[
c_1=\frac{B_0}{B_0+a_1},
\qquad
c_2=\frac{B_0}{B_0+2a_1},
\]

\[
D_{ss}
=
\frac{1}{R^2}
\left[
b\Lambda(\chi_0)
+
\frac{b^2}{1+b^2}\Lambda'(\chi_0)
\right].
\]

With

\[
J=-\partial_n w,
\]

the full result is

\[
\boxed{
g_{\rm full}
=
c_2
\left(
\frac{J_{ss}}{J}
-
\frac{D_{ss}}{D_0}
\right)
-
c_1^2
\left(\frac{J_s}{J}\right)^2
}
\]

and the simplified result is

\[
\boxed{
g_{\rm flux}
=
\frac{J_{ss}}{J}
-
\left(\frac{J_s}{J}\right)^2.
}
\]

All flux quantities are evaluated at the high-gap pole.

---

# Scope restraint

Keep this PR/script focused.

Do **not**:

- replace the full time integrator;
- implement a general complementarity/contact-line solver;
- modify the finite-CASR production simulations;
- refactor unrelated FEM code;
- calibrate the asymptotic formulas to the trajectory data.

This is a three-way numerical comparison intended to answer one question:

> Does the cleaned finite-CASR trajectory estimate of \(g\) agree with the large-CASR one-gas-solve prediction, and how much of the answer is already captured by \(g=\partial_s^2\log J\)?
