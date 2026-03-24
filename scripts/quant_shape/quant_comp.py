"""
Quantitative shape comparison: theory vs experiment.

shape_err(poly1, poly2) optimizes over all complex affine maps z -> az + b
(translations, rotations, dilations) applied to poly2, then returns
  1 - (intersection area) / (mean area of the two polygons)
as an error measure (0 = perfect match, 1 = no overlap).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.optimize import minimize

DIR = "/home/wcr/git/cornell/mdropm/scripts/quant_shape"
NUMS = ["one", "two", "three", "four"]


# ---------------------------------------------------------------------------
# Core shape-error function
# ---------------------------------------------------------------------------

def _apply_complex_affine(poly: Polygon, re_a: float, im_a: float,
                          re_b: float, im_b: float) -> Polygon:
    """Apply z -> a*z + b to every vertex of poly."""
    coords = np.array(poly.exterior.coords)
    z = coords[:, 0] + 1j * coords[:, 1]
    w = (re_a + 1j * im_a) * z + (re_b + 1j * im_b)
    return Polygon(zip(w.real, w.imag))


def _overlap_ratio(poly1: Polygon, poly2_t: Polygon) -> float:
    """Intersection area / mean area.  Returns 0 on degeneracy."""
    if not poly2_t.is_valid:
        poly2_t = poly2_t.buffer(0)
    mean_area = (poly1.area + poly2_t.area) / 2.0
    if mean_area < 1e-12:
        return 0.0
    inter = poly1.intersection(poly2_t)
    return inter.area / mean_area


def shape_err(poly1: Polygon, poly2: Polygon) -> float:
    """
    Fix poly1, optimise z -> a*z + b applied to poly2 to maximise
    (intersection area) / (mean area).  Returns 1 minus that maximum.
    """
    # ------------------------------------------------------------------
    # Warm-start: align centroids and match areas
    # ------------------------------------------------------------------
    c1 = np.array(poly1.centroid.coords[0])
    c2 = np.array(poly2.centroid.coords[0])
    scale0 = np.sqrt(poly1.area / poly2.area) if poly2.area > 1e-12 else 1.0
    # a*c2 + b = c1  →  b = c1 - a*c2  (with a = scale0, rotation = 0)
    b0 = (c1[0] - scale0 * c2[0]) + 1j * (c1[1] - scale0 * c2[1])
    x0 = np.array([scale0, 0.0, b0.real, b0.imag])

    def neg_overlap(params):
        poly2_t = _apply_complex_affine(poly2, *params)
        return -_overlap_ratio(poly1, poly2_t)

    # ------------------------------------------------------------------
    # Local refinement from warm start (fast)
    # ------------------------------------------------------------------
    res = minimize(neg_overlap, x0, method="Nelder-Mead",
                   options={"maxiter": 20_000, "xatol": 1e-7, "fatol": 1e-7,
                            "adaptive": True})

    best = res.fun

    # ------------------------------------------------------------------
    # Also try a few rotated starts in case Nelder-Mead gets stuck
    # ------------------------------------------------------------------
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False)[1:]:
        a_try = scale0 * np.exp(1j * angle)
        b_try = (c1[0] + 1j * c1[1]) - a_try * (c2[0] + 1j * c2[1])
        x_try = np.array([a_try.real, a_try.imag, b_try.real, b_try.imag])
        r = minimize(neg_overlap, x_try, method="Nelder-Mead",
                     options={"maxiter": 10_000, "xatol": 1e-7, "fatol": 1e-7,
                              "adaptive": True})
        if r.fun < best:
            best = r.fun

    return float(1.0 + best)   # 1 - max_overlap


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_polygon(series: str, kind: str, num: str) -> Polygon:
    """Load X,Y columns from {series}_{kind}_{num}.csv and return a Polygon."""
    path = f"{DIR}/{series}_{kind}_{num}.csv"
    df = pd.read_csv(path)
    coords = list(zip(df["X"], df["Y"]))
    return Polygon(coords)


def make_ngon(n: int = 200, radius: float = 1.0) -> Polygon:
    """Return a regular n-gon centred at the origin with the given radius."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(zip(radius * np.cos(angles), radius * np.sin(angles)))


# ---------------------------------------------------------------------------
# Compute errors
# ---------------------------------------------------------------------------

series_list = ["tri_cant", "flat_cant"]
errors = {s: [] for s in series_list}

for series in series_list:
    for num in NUMS:
        poly_exp    = load_polygon(series, "exp",    num)
        poly_theory = load_polygon(series, "theory", num)
        err = shape_err(poly_theory, poly_exp)
        errors[series].append(err)
        print(f"{series}  t={num:5s}  error = {err:.4f}")

# flat_empty: theory is a regular 200-gon (unit radius; shape_err handles scale)
errors["flat_empty"] = []
theory_circle = make_ngon(200, radius=1.0)
for num in NUMS:
    poly_exp = load_polygon("flat_empty", "exp", num)
    err = shape_err(theory_circle, poly_exp)
    errors["flat_empty"].append(err)
    print(f"flat_empty  t={num:5s}  error = {err:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

time_labels = [1, 2, 3, 4]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(time_labels, [e * 100 for e in errors["tri_cant"]],
        marker="o", linewidth=2, label="U-Tape, Canted")
ax.plot(time_labels, [e * 100 for e in errors["flat_cant"]],
        marker="s", linewidth=2, label="U-Tape, Flat")
ax.plot(time_labels, [e * 100 for e in errors["flat_empty"]],
        marker="^", linewidth=2, linestyle="--", label="Tripod Tape, Flat")

ax.set_xticks(time_labels)
ax.set_xticklabels(["$t_1$", "$t_2$", "$t_3$", "$t_4$"])
ax.set_xlabel("Time point", fontsize=12)
ax.set_ylabel("Shape error (%)", fontsize=12)
ax.set_title("Theory vs. Experiment: Shape Comparison Error", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5)
ax.set_ylim(bottom=0)

fig.tight_layout()
out_path = f"{DIR}/quant_comp.png"
fig.savefig(out_path, dpi=200)
print(f"\nSaved → {out_path}")
