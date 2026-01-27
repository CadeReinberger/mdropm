#!/usr/bin/env python3
"""
Overlay animation: evap vs casr

Reads:
  - evap.pkl : expects keys 'ts_list' and 'vecs_list'
      vecs_list[i][0::2] -> x points at time ts_list[i]
      vecs_list[i][1::2] -> y points at time ts_list[i]
  - casr.pkl : expects keys 'out_t' and 'out_x'
      out_x[i] length = 3n; first n are x, next n are y (last n ignored)

Pipeline:
  1) For each dataset, at each recorded time, fit a *spatial* periodic cubic spline
     through the points and compute its area A(t).
  2) Find cutoff time T such that A(T) = 0.5 * A(0) (first crossing, linearly interpolated).
  3) For each point index j, fit a *time* cubic spline x_j(t), y_j(t).
  4) Sample M evenly spaced times from 0..T_e and 0..T_c (same M).
  5) Scale evap coordinates by factor s = sqrt(A_c(0)/A_e(0)) so the *spatial spline areas*
     match at t=0.
  6) For each frame, build spatial periodic splines from the sampled points and plot both,
     overlayed, with fixed x/y limits, saved as an animation.

Outputs:
  - overlay_animation.mp4 (preferred if ffmpeg is available)
  - overlay_animation.gif (fallback)

Requirements:
  pip install numpy scipy matplotlib
"""

import os
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.interpolate import CubicSpline


# -----------------------------
# I/O helpers
# -----------------------------
def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pk.load(f)


def load_evap(path="evap.pkl"):
    d = load_pkl(path)
    ts = np.asarray(d["ts_list"], dtype=float)
    vecs_list = d["vecs_list"]

    # Build arrays: X[t_i, j], Y[t_i, j]
    xs = []
    ys = []
    for v in vecs_list:
        v = np.asarray(v, dtype=float).ravel()
        x = v[0::2]
        y = v[1::2]
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 2 or ys.ndim != 2:
        raise ValueError("evap.pkl: could not form 2D arrays of points across time.")
    if xs.shape != ys.shape:
        raise ValueError("evap.pkl: xs and ys shapes mismatch.")

    # sanity: constant N through time
    if len({row.size for row in xs}) != 1:
        raise ValueError("evap.pkl: number of points changes across time; this script assumes fixed ordering/size.")
    return ts, xs, ys


def load_casr(path="casr.pkl"):
    d = load_pkl(path)
    ts = np.asarray(d["out_t"], dtype=float)
    out_x = d["out_x"]

    xs = []
    ys = []
    for row in out_x:
        row = np.asarray(row, dtype=float).ravel()
        if row.size % 3 != 0:
            raise ValueError(f"casr.pkl: out_x row length {row.size} not divisible by 3.")
        n = row.size // 3
        x = row[:n]
        y = row[n:2 * n]
        xs.append(x)
        ys.append(y)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError("casr.pkl: xs and ys shapes mismatch.")
    if len({row.size for row in xs}) != 1:
        raise ValueError("casr.pkl: number of points changes across time; this script assumes fixed ordering/size.")
    return ts, xs, ys


# -----------------------------
# Spatial periodic spline + area
# -----------------------------
def periodic_spatial_splines(x: np.ndarray, y: np.ndarray):
    """
    Given N points (x[j], y[j]) in order around the closed curve,
    fit periodic cubic splines x(s), y(s) with s in [0, 1].
    SciPy's CubicSpline(periodic) requires first=last; we append.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for a closed spline")

    x2 = np.concatenate([x, [x[0]]])
    y2 = np.concatenate([y, [y[0]]])
    s = np.linspace(0.0, 1.0, n + 1)

    sx = CubicSpline(s, x2, bc_type="periodic")
    sy = CubicSpline(s, y2, bc_type="periodic")
    return sx, sy


def sample_spatial_spline(sx, sy, n_sample=800):
    s = np.linspace(0.0, 1.0, n_sample, endpoint=False)
    xs = sx(s)
    ys = sy(s)
    return xs, ys


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    # Shoelace; assumes points are ordered around boundary
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def spline_area_from_points(x: np.ndarray, y: np.ndarray, n_sample=1200) -> float:
    sx, sy = periodic_spatial_splines(x, y)
    xs, ys = sample_spatial_spline(sx, sy, n_sample=n_sample)
    return polygon_area(xs, ys)


def areas_over_time(ts: np.ndarray, X: np.ndarray, Y: np.ndarray, n_sample=1200) -> np.ndarray:
    A = np.zeros_like(ts, dtype=float)
    for i in range(ts.size):
        A[i] = spline_area_from_points(X[i], Y[i], n_sample=n_sample)
    return A


def cutoff_time_half_area(ts: np.ndarray, A: np.ndarray) -> float:
    """
    Find first time T where A(T) = 0.5*A0, using linear interpolation between samples.
    """
    A0 = A[0]
    target = 0.5 * A0

    # If never crosses, raise with info.
    # We define crossing as first i where A[i] <= target (assuming decreasing-ish).
    idx = np.where(A <= target)[0]
    if idx.size == 0:
        raise RuntimeError(f"Area never drops to 50%: min(A)/A0 = {A.min()/A0:.3f}")
    i = int(idx[0])
    if i == 0:
        return float(ts[0])

    t0, t1 = ts[i - 1], ts[i]
    a0, a1 = A[i - 1], A[i]
    if a0 == a1:
        return float(t1)
    # linear interpolate for a(t)=target
    alpha = (target - a0) / (a1 - a0)
    return float(t0 + alpha * (t1 - t0))


# -----------------------------
# Time splines for each point
# -----------------------------
def time_splines_per_point(ts: np.ndarray, X: np.ndarray, Y: np.ndarray):
    """
    Returns lists of CubicSpline objects sx_list[j], sy_list[j] giving x_j(t), y_j(t).
    Assumes point ordering is consistent in time.
    """
    ts = np.asarray(ts, dtype=float)
    if np.any(np.diff(ts) <= 0):
        # enforce strictly increasing
        order = np.argsort(ts)
        ts = ts[order]
        X = X[order]
        Y = Y[order]

    n_points = X.shape[1]
    sx_list = []
    sy_list = []
    for j in range(n_points):
        sx_list.append(CubicSpline(ts, X[:, j], bc_type="not-a-knot", extrapolate=True))
        sy_list.append(CubicSpline(ts, Y[:, j], bc_type="not-a-knot", extrapolate=True))
    return ts, sx_list, sy_list


def eval_points_at_times(sx_list, sy_list, t: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([sx(t) for sx in sx_list], dtype=float)
    y = np.array([sy(t) for sy in sy_list], dtype=float)
    return x, y


# -----------------------------
# Animation
# -----------------------------
def main(
    evap_path="evap.pkl",
    casr_path="casr.pkl",
    n_frames=240,
    spatial_sample=900,
    area_sample=1400,
    fps=30,
):
    # ---- Load
    te_raw, Xe_raw, Ye_raw = load_evap(evap_path)
    tc_raw, Xc_raw, Yc_raw = load_casr(casr_path)

    Ne = Xe_raw.shape[1]
    Nc = Xc_raw.shape[1]
    print(f"[info] evap: {te_raw.size} times, N_e={Ne} points")
    print(f"[info] casr: {tc_raw.size} times, N_c={Nc} points")

    # ---- Areas + cutoffs
    Ae = areas_over_time(te_raw, Xe_raw, Ye_raw, n_sample=area_sample)
    Ac = areas_over_time(tc_raw, Xc_raw, Yc_raw, n_sample=area_sample)

    Te = cutoff_time_half_area(te_raw, Ae)
    Tc = cutoff_time_half_area(tc_raw, Ac)
    print(f"[info] T_e (50% area) = {Te:.6g}")
    print(f"[info] T_c (50% area) = {Tc:.6g}")

    # ---- Scale evap so areas match at t=0 (spatial spline area)
    # Use the *spline-based* areas at first recorded times.
    scale = np.sqrt(Ac[0] / Ae[0])
    print(f"[info] area match scale factor (apply to evap x,y) = {scale:.6g}")

    # ---- Time splines per point
    te_sorted, ex_spl, ey_spl = time_splines_per_point(te_raw, Xe_raw, Ye_raw)
    tc_sorted, cx_spl, cy_spl = time_splines_per_point(tc_raw, Xc_raw, Yc_raw)

    # ---- Frame times (same number of frames)
    tE = np.linspace(0.0, Te, n_frames)
    tC = np.linspace(0.0, Tc, n_frames)

    # ---- Precompute axis limits (based on sampled control points across all frames)
    # This is conservative enough for plotting spline samples too, with padding.
    allx = []
    ally = []
    for k in range(n_frames):
        xe, ye = eval_points_at_times(ex_spl, ey_spl, tE[k])
        xc, yc = eval_points_at_times(cx_spl, cy_spl, tC[k])
        xe *= scale
        ye *= scale
        allx.append(xe); allx.append(xc)
        ally.append(ye); ally.append(yc)

    allx = np.concatenate(allx)
    ally = np.concatenate(ally)
    xmin, xmax = np.min(allx), np.max(allx)
    ymin, ymax = np.min(ally), np.max(ally)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = 0.06
    xlim = (xmin - pad * dx, xmax + pad * dx)
    ylim = (ymin - pad * dy, ymax + pad * dy)

    # ---- Setup plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)
    ax.set_title("evap vs casr (area-matched at t=0)")

    # lines for spline samples
    (line_e,) = ax.plot([], [], linewidth=2.0, linestyle="-", label="evap (scaled)")
    (line_c,) = ax.plot([], [], linewidth=2.0, linestyle="--", label="casr")
    # optional control points (light)
    (pts_e,) = ax.plot([], [], marker="o", linestyle="None", markersize=2.5, alpha=0.5)
    (pts_c,) = ax.plot([], [], marker="o", linestyle="None", markersize=2.5, alpha=0.5)

    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        va="top", ha="left"
    )

    ax.legend(loc="upper right")

    def update(frame):
        # Evaluate control points at their respective times
        xe, ye = eval_points_at_times(ex_spl, ey_spl, float(tE[frame]))
        xc, yc = eval_points_at_times(cx_spl, cy_spl, float(tC[frame]))

        # Scale evap
        xe *= scale
        ye *= scale

        # Fit spatial periodic splines and sample them
        sx_e, sy_e = periodic_spatial_splines(xe, ye)
        sx_c, sy_c = periodic_spatial_splines(xc, yc)

        xs_e, ys_e = sample_spatial_spline(sx_e, sy_e, n_sample=spatial_sample)
        xs_c, ys_c = sample_spatial_spline(sx_c, sy_c, n_sample=spatial_sample)

        line_e.set_data(xs_e, ys_e)
        line_c.set_data(xs_c, ys_c)

        pts_e.set_data(xe, ye)
        pts_c.set_data(xc, yc)

        time_text.set_text(
            f"frame {frame+1}/{n_frames}\n"
            f"t_e = {tE[frame]:.4g} / {Te:.4g}\n"
            f"t_c = {tC[frame]:.4g} / {Tc:.4g}"
        )
        return line_e, line_c, pts_e, pts_c, time_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=True)

    # ---- Save
    out_mp4 = "overlay_animation.mp4"
    out_gif = "overlay_animation.gif"

    saved = False
    try:
        # If ffmpeg is installed, this usually works.
        anim.save(out_mp4, dpi=160, fps=fps)
        print(f"[done] wrote {out_mp4}")
        saved = True
    except Exception as e:
        print(f"[warn] mp4 save failed ({e}). Trying GIF...")

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=140)
            print(f"[done] wrote {out_gif}")
            saved = True
        except Exception as e:
            print(f"[error] gif save failed too: {e}")
            print("Install ffmpeg or pillow, or lower n_frames/spatial_sample.")
            raise

    plt.close(fig)


if __name__ == "__main__":
    # You can tweak these defaults without touching the rest.
    main(
        evap_path="evap8.pkl",
        casr_path="casr.pkl",
        n_frames=240,        # number of time samples / animation frames
        spatial_sample=900,  # samples along each spatial spline per frame
        area_sample=1400,    # samples used for area computation
        fps=30,
    )

