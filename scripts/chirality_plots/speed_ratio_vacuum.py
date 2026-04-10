#!/usr/bin/env python3
"""
Build and plot the ratio
    ratio = |dw/dn| / sqrt((dx/dt)^2 + (dy/dt)^2)
from simulation outputs in each `vacuum_tests/*/results.pkl` and `setup.pkl`.

This script groups runs by close `w_eq` values and writes one plot per group.

Notes:
- Time-spline fitting reuses `to_spline_set` from chirality_map_smoothed.py.
- `compute_concentration_gradients` is defined in gas_fem.py (not liquid_fem.py),
  and returns gradient vectors at each droplet point. We use their magnitudes.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Ensure repo root is importable when running
# `python scripts/chirality_plots/speed_ratio_vacuum.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import heightscape
import tapescape
from physical_params import physical_params
from problem_universe import problem_universe
from solver_params import solver_params


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _rebuild_heightscape(hs_dict: dict):
    ufl_str = hs_dict["ufl_str"]
    poly_str = ufl_str.replace("x[0]", "x").replace("x[1]", "y")
    return heightscape.constructors.from_poly(poly_str)


def _rebuild_tapescape(tp_dict: dict):
    n = int(tp_dict["n"])
    xs = np.asarray(tp_dict["xs"], dtype=float)
    ys = np.asarray(tp_dict["ys"], dtype=float)
    types = list(tp_dict["types"])
    segs = []
    for i in range(n):
        start_pt = np.array([xs[i], ys[i]], dtype=float)
        end_pt = np.array([xs[(i + 1) % n], ys[(i + 1) % n]], dtype=float)
        segs.append(tapescape.EXTERNAL_BC_SEGMENT(start_pt, end_pt, types[i]))
    return tapescape.EXTERNAL_BCS(segs)


def _rebuild_solver_params(sp_dict: dict):
    sps = solver_params()
    for k, v in sp_dict.items():
        setattr(sps, k, v)
    return sps


def _rebuild_physical_params(pp_dict: dict):
    return physical_params(**pp_dict)


def _load_problem_universe(setup_path: Path):
    setup = _load_pickle(setup_path)
    hs = _rebuild_heightscape(setup["heightscape"])
    ts = _rebuild_tapescape(setup["tapescape"])
    sps = _rebuild_solver_params(setup["solver_params"])
    pps = _rebuild_physical_params(setup["physical_params"])
    return problem_universe(hs, ts, sps, pps)


def _load_results(results_path: Path):
    res = _load_pickle(results_path)
    all_ts = np.asarray(res["out_t"], dtype=float)
    all_vecs = [np.asarray(v, dtype=float) for v in res["out_x"]]
    n = all_vecs[0].size // 3
    all_xs = [v[:n] for v in all_vecs]
    all_ys = [v[n : 2 * n] for v in all_vecs]
    return all_ts, all_xs, all_ys


def _extract_w_eq(setup_path: Path) -> float:
    setup = _load_pickle(setup_path)
    if isinstance(setup, dict):
        pp = setup.get("physical_params")
        if isinstance(pp, dict) and "w_eq" in pp:
            return float(pp["w_eq"])
        if "w_eq" in setup:
            return float(setup["w_eq"])
    raise KeyError(f"Could not find w_eq in {setup_path}")


def _group_runs_by_weq(vacuum_root: Path, atol: float, rtol: float):
    groups = []
    for run_dir in sorted(vacuum_root.iterdir()):
        if not run_dir.is_dir():
            continue
        setup_path = run_dir / "setup.pkl"
        results_path = run_dir / "results.pkl"
        if not (setup_path.exists() and results_path.exists()):
            continue

        w_eq = _extract_w_eq(setup_path)
        matched = False
        for group in groups:
            if np.isclose(w_eq, group["w_eq"], atol=atol, rtol=rtol):
                group["run_dirs"].append(run_dir)
                matched = True
                break
        if not matched:
            groups.append({"w_eq": w_eq, "run_dirs": [run_dir]})
    groups.sort(key=lambda g: g["w_eq"])
    return groups


def _safe_grad_mag(grad_vec):
    g = np.asarray(grad_vec, dtype=float).reshape(-1)
    if g.size < 2:
        return np.nan
    return float(np.hypot(g[0], g[1]))


def _to_spline_set_fallback(ts, xs, ys, n_intervals=2, smooth_fallback=1e-6):
    # Local fallback copied from scripts/chirality_map_smoothed.py.
    from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline

    t = np.asarray(ts, dtype=float)
    if t.ndim != 1:
        raise ValueError("ts must be 1D")
    T = t.size
    if len(xs) != T or len(ys) != T:
        raise ValueError("xs and ys must have outer length equal to len(ts)")
    if T < 4:
        raise ValueError("Need at least 4 time samples for a cubic spline regression")

    if not np.all(np.diff(t) > 0):
        uniq_t, uniq_idx = np.unique(t, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        t = t[uniq_idx]
        xs = [xs[i] for i in uniq_idx]
        ys = [ys[i] for i in uniq_idx]
        T = t.size
        if T < 4 or not np.all(np.diff(t) > 0):
            raise ValueError("ts must be strictly increasing after removing duplicates")

    N = len(xs[0])
    for l in range(T):
        if len(xs[l]) != N or len(ys[l]) != N:
            raise ValueError("Each xs[l], ys[l] must have the same length")

    if n_intervals < 2:
        raise ValueError("n_intervals must be >= 2")

    n_int = n_intervals - 1
    use_lsq = (T > n_int + 3) and (n_int > 0)
    if use_lsq:
        knots = np.linspace(t[0], t[-1], n_intervals + 1)[1:-1]
        knots = np.unique(knots)
        if knots.size == 0:
            use_lsq = False
    else:
        knots = np.array([], dtype=float)

    x_splines = []
    y_splines = []
    for s_ind in range(N):
        x_vals = np.asarray([xs[l][s_ind] for l in range(T)], dtype=float)
        y_vals = np.asarray([ys[l][s_ind] for l in range(T)], dtype=float)

        if use_lsq:
            try:
                xspl = LSQUnivariateSpline(t, x_vals, t=knots, k=3)
            except Exception:
                s = float(smooth_fallback) * T * (np.var(x_vals) + 1e-12)
                xspl = UnivariateSpline(t, x_vals, k=3, s=s)
            try:
                yspl = LSQUnivariateSpline(t, y_vals, t=knots, k=3)
            except Exception:
                s = float(smooth_fallback) * T * (np.var(y_vals) + 1e-12)
                yspl = UnivariateSpline(t, y_vals, k=3, s=s)
        else:
            s = float(smooth_fallback) * T * (np.var(x_vals) + 1e-12)
            xspl = UnivariateSpline(t, x_vals, k=3, s=s)
            s = float(smooth_fallback) * T * (np.var(y_vals) + 1e-12)
            yspl = UnivariateSpline(t, y_vals, k=3, s=s)

        x_splines.append(xspl)
        y_splines.append(yspl)

    return x_splines, y_splines


def _get_to_spline_set():
    # Prefer importing project logic, fallback if module side-effects fail.
    script_dir = Path(__file__).resolve().parent
    cwd0 = Path.cwd()
    try:
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        os.chdir(script_dir)
        from chirality_map_smoothed import to_spline_set as imported_to_spline_set

        return imported_to_spline_set
    except Exception:
        return _to_spline_set_fallback
    finally:
        os.chdir(cwd0)


def build_ratio_samples(
    pu,
    ts: np.ndarray,
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    n_time_samples: int,
    speed_eps: float = 1e-12,
):
    try:
        from gas_fem import compute_concentration_gradients
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing FEM dependency while importing gas_fem "
            "(commonly gmsh/dolfinx stack). Install runtime deps to compute gradients."
        ) from exc

    to_spline_set = _get_to_spline_set()
    x_splines, y_splines = to_spline_set(ts, xs, ys)
    n_pts = len(x_splines)
    t_eval = np.linspace(float(ts[0]), float(ts[-1]), num=n_time_samples)

    all_xy = []
    all_ratio = []

    for t in t_eval:
        cur_x = np.array([x_splines[i](t) for i in range(n_pts)], dtype=float)
        cur_y = np.array([y_splines[i](t) for i in range(n_pts)], dtype=float)
        cur_speed = np.array(
            [
                np.hypot(x_splines[i](t, 1), y_splines[i](t, 1))
                for i in range(n_pts)
            ],
            dtype=float,
        )

        grads = compute_concentration_gradients(cur_x, cur_y, pu)
        grad_mag = np.array([_safe_grad_mag(g) for g in grads], dtype=float)

        mask = np.isfinite(grad_mag) & np.isfinite(cur_speed) & (cur_speed > speed_eps)
        if np.any(mask):
            xy = np.column_stack((cur_x[mask], cur_y[mask]))
            ratio = grad_mag[mask] / cur_speed[mask]
            all_xy.append(xy)
            all_ratio.append(ratio)

    if not all_xy:
        raise RuntimeError("No valid ratio samples were produced.")

    sample_xy = np.vstack(all_xy)
    sample_ratio = np.concatenate(all_ratio)
    return sample_xy, sample_ratio


def plot_ratio_field(
    sample_xy,
    sample_ratio,
    out_path: Path,
    title: str = "Speed Ratio Field",
    dpi: int = 400,
    grid_n: int = 450,
    x_rect: tuple[float, float] = (-1.0, 1.0),
    y_rect: tuple[float, float] = (1.0, 3.0),
):
    interp = LinearNDInterpolator(sample_xy, sample_ratio, fill_value=np.nan)
    x_lo, x_hi = map(float, x_rect)
    y_lo, y_hi = map(float, y_rect)
    if not (x_lo < x_hi and y_lo < y_hi):
        raise ValueError(
            f"Invalid rectangle bounds: x_rect={x_rect}, y_rect={y_rect}. "
            "Expected lower < upper in each dimension."
        )

    in_rect = (
        (sample_xy[:, 0] >= x_lo)
        & (sample_xy[:, 0] <= x_hi)
        & (sample_xy[:, 1] >= y_lo)
        & (sample_xy[:, 1] <= y_hi)
    )
    finite_ratio = sample_ratio[np.isfinite(sample_ratio) & in_rect]
    if finite_ratio.size == 0:
        raise ValueError("No finite ratio values inside rectangle for colorscale.")
    ratio_vmax = float(np.percentile(finite_ratio, 95))

    xg = np.linspace(x_lo, x_hi, grid_n)
    yg = np.linspace(y_lo, y_hi, grid_n)
    X, Y = np.meshgrid(xg, yg, indexing="xy")
    Z = interp(X, Y)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=dpi)
    m = ax.pcolormesh(X, Y, Z, shading="auto", cmap="hsv", vmax=ratio_vmax)
    ax.scatter(
        sample_xy[:, 0],
        sample_xy[:, 1],
        c=sample_ratio,
        s=7,
        cmap="hsv",
        vmax=ratio_vmax,
        edgecolors="none",
        alpha=0.7,
    )
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label(r"$|dw/dn| \,/\, \sqrt{(dx/dt)^2 + (dy/dt)^2}$", fontsize=12)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(alpha=0.15, linewidth=0.5)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return interp


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vacuum-root",
        type=Path,
        default=root / "vacuum_tests",
        help="Directory that contains vacuum run subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "scripts" / "chirality_plots" / "vacuum_stuff",
        help="Output directory for per-w_eq plots",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=50,
        help="Number of equally spaced time samples",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Output DPI",
    )
    parser.add_argument(
        "--weq-atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for grouping close w_eq values",
    )
    parser.add_argument(
        "--weq-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for grouping close w_eq values",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vacuum_root = args.vacuum_root.resolve()
    groups = _group_runs_by_weq(vacuum_root, atol=args.weq_atol, rtol=args.weq_rtol)
    if not groups:
        raise RuntimeError(f"No runnable vacuum test directories found in {vacuum_root}")

    for group in groups:
        w_eq = float(group["w_eq"])
        sample_xy_parts = []
        sample_ratio_parts = []
        for run_dir in group["run_dirs"]:
            setup_path = run_dir / "setup.pkl"
            results_path = run_dir / "results.pkl"
            pu = _load_problem_universe(setup_path)
            ts, xs, ys = _load_results(results_path)
            sample_xy, sample_ratio = build_ratio_samples(
                pu=pu, ts=ts, xs=xs, ys=ys, n_time_samples=args.n_time
            )
            sample_xy_parts.append(sample_xy)
            sample_ratio_parts.append(sample_ratio)

        merged_xy = np.vstack(sample_xy_parts)
        merged_ratio = np.concatenate(sample_ratio_parts)
        out_path = args.out_dir / f"{round(w_eq * 1000)}_speed_rat_plot.png"
        title = f"Speed Ratio Field (w_eq={w_eq:.3f})"
        plot_ratio_field(
            merged_xy,
            merged_ratio,
            out_path,
            title=title,
            dpi=args.dpi,
        )
        print(
            f"Saved {out_path} from {len(group['run_dirs'])} run(s) "
            f"with w_eq~{w_eq:.6g}"
        )


if __name__ == "__main__":
    main()
