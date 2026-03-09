#!/usr/bin/env python3
"""Animate interpolated droplet pressure fields into an MP4.

Reads simulation output from ../out/results.pkl and setup metadata from
../out/settings.pkl (fallback: ../out/setup.pkl), interpolates states to a
fixed number of evenly spaced times, solves the liquid FEM pressure field at
those times, and writes an MP4 with fixed x/y limits and droplet outline.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.plot import vtk_mesh
from ffmpy import FFmpeg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import heightscape
import tapescape
from liquid_fem import solve_pressure_field
from physical_params import physical_params
from problem_universe import problem_universe
from solver_params import solver_params

# ------------------------------
# User-configurable constants
# ------------------------------
NUM_FRAMES = 120
FPS = 24
RESULTS_PATH = REPO_ROOT / "out" / "results.pkl"
SETTINGS_PATH = REPO_ROOT / "out" / "settings.pkl"
SETUP_FALLBACK_PATH = REPO_ROOT / "out" / "setup.pkl"
OUTPUT_MP4 = Path(__file__).resolve().parent / "pressure_field_animation.mp4"
OUTPUT_GIF = Path(__file__).resolve().parent / "pressure_field_animation.gif"
AXIS_PADDING_FRAC = 0.05
CMAP = "viridis"


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def rebuild_heightscape(hs_dict: dict):
    ufl_str = hs_dict["ufl_str"]
    poly_str = ufl_str.replace("x[0]", "x").replace("x[1]", "y")
    return heightscape.constructors.from_poly(poly_str)


def rebuild_tapescape(tp_dict: dict):
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


def rebuild_solver_params(sp_dict: dict):
    sps = solver_params()
    for key, val in sp_dict.items():
        setattr(sps, key, val)

    # Make mesh output path robust regardless of current working directory.
    mesh_file = Path(sps.LIQUID_PHASE_MESH_FILE)
    if not mesh_file.is_absolute():
        sps.LIQUID_PHASE_MESH_FILE = str(REPO_ROOT / mesh_file)

    return sps


def load_problem_universe(settings_or_setup_path: Path):
    setup = load_pickle(settings_or_setup_path)
    hs = rebuild_heightscape(setup["heightscape"])
    ts = rebuild_tapescape(setup["tapescape"])
    sps = rebuild_solver_params(setup["solver_params"])
    pps = physical_params(**setup["physical_params"])
    start_drop = setup["start_drop"]
    pu = problem_universe(hs, ts, sps, pps)
    return pu, start_drop


def load_results(path: Path):
    res = load_pickle(path)
    ts = np.asarray(res["out_t"], dtype=float)
    x_vecs = np.asarray([np.asarray(v, dtype=float) for v in res["out_x"]], dtype=float)
    if ts.ndim != 1 or x_vecs.ndim != 2:
        raise ValueError("Unexpected shape for results.pkl")
    if x_vecs.shape[0] != ts.shape[0]:
        raise ValueError("Length mismatch between out_t and out_x")
    return ts, x_vecs


def interpolate_states(raw_t: np.ndarray, raw_x: np.ndarray, frame_t: np.ndarray):
    m, dim = raw_x.shape
    out = np.zeros((frame_t.size, dim), dtype=float)

    j = 0
    for i, t in enumerate(frame_t):
        while j + 1 < m and raw_t[j + 1] < t:
            j += 1

        if t <= raw_t[0]:
            out[i] = raw_x[0]
            continue
        if t >= raw_t[-1]:
            out[i] = raw_x[-1]
            continue

        t0, t1 = raw_t[j], raw_t[j + 1]
        tau = (t - t0) / (t1 - t0)
        out[i] = (1.0 - tau) * raw_x[j] + tau * raw_x[j + 1]

    return out


def compute_boundary_pressure(dr_x, dr_y, dr_theta, pu, ds):
    n = dr_x.size
    gamma = pu.phys_ps.gamma
    hs = pu.htscp

    p_arr = np.zeros(n, dtype=float)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = i - 1

        x_s = (dr_x[ip1] - dr_x[im1]) / (2.0 * ds)
        y_s = (dr_y[ip1] - dr_y[im1]) / (2.0 * ds)
        norm = np.hypot(x_s, y_s)
        if norm <= 1e-14:
            raise ValueError("Degenerate boundary tangent encountered")

        n_hat = np.array([y_s, -x_s], dtype=float) / norm

        h = float(hs.h(dr_x[i], dr_y[i]))
        hx = float(hs.hx(dr_x[i], dr_y[i]))
        hy = float(hs.hy(dr_x[i], dr_y[i]))
        grad_h = np.array([hx, hy], dtype=float)

        psi = np.arctan(np.dot(n_hat, grad_h))
        p_arr[i] = -gamma * np.cos(dr_theta[i] + psi) / h

    return p_arr


def mesh_triangles_and_values(uh, mesh):
    # Interpolate to a P1 space for plotting instead of repeatedly calling
    # `uh.eval` on guessed cells, which is a fragile C++ boundary in DOLFINx.
    v_plot = functionspace(mesh, ("CG", 1))
    u_plot = Function(v_plot)
    u_plot.interpolate(uh)

    cells, _, coords = vtk_mesh(v_plot)
    cells = np.asarray(cells, dtype=np.int32).reshape(-1, 4)
    triangles = cells[:, 1:4].copy()
    verts = np.asarray(coords[:, :2], dtype=float)
    values = np.asarray(u_plot.x.array, dtype=float).copy()

    return verts, triangles, values


def frame_bounds(xs: np.ndarray, ys: np.ndarray, pad_frac: float):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    dx = xmax - xmin
    dy = ymax - ymin
    padx = pad_frac * (dx if dx > 0 else 1.0)
    pady = pad_frac * (dy if dy > 0 else 1.0)

    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)


def render_animation(frames, xlim, ylim, vmin, vmax, output_gif: Path, output_mp4: Path, fps: int):
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    images = []
    for ind, fr in enumerate(frames):
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
        tri = ax.tripcolor(
            fr["verts"][:, 0],
            fr["verts"][:, 1],
            fr["triangles"],
            fr["values"],
            shading="gouraud",
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
        )
        ax.plot(fr["x_outline"], fr["y_outline"], color="black", linewidth=1.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Pressure Field, t={fr['t']:.3f}")
        fig.colorbar(tri, ax=ax, label="pressure")
        fig.tight_layout()

        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        plt.close(fig)

        print(f"Rendered frame {ind + 1}/{len(frames)}")

    duration = 1.0 / fps
    imageio.mimsave(output_gif, images, duration=duration)

    ff = FFmpeg(inputs={str(output_gif): None}, outputs={str(output_mp4): "-y"})
    ff.run()


def main():
    settings_path = SETTINGS_PATH if SETTINGS_PATH.exists() else SETUP_FALLBACK_PATH
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_PATH}")
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Missing setup/settings file. Checked: {SETTINGS_PATH} and {SETUP_FALLBACK_PATH}"
        )

    pu, start_drop = load_problem_universe(settings_path)
    raw_t, raw_state = load_results(RESULTS_PATH)

    n = raw_state.shape[1] // 3
    if n < 3:
        raise ValueError("Need at least 3 boundary points")

    ds = float(start_drop.get("L", 2.0 * np.pi) / n)

    frame_t = np.linspace(float(raw_t[0]), float(raw_t[-1]), NUM_FRAMES)
    frame_state = interpolate_states(raw_t, raw_state, frame_t)

    all_x = frame_state[:, :n]
    all_y = frame_state[:, n : 2 * n]
    xlim, ylim = frame_bounds(all_x, all_y, AXIS_PADDING_FRAC)

    frames = []
    all_vals = []

    for i in range(NUM_FRAMES):
        cur = frame_state[i]
        dr_x = cur[:n]
        dr_y = cur[n : 2 * n]
        dr_theta = cur[2 * n : 3 * n]

        p_boundary = compute_boundary_pressure(dr_x, dr_y, dr_theta, pu, ds)
        uh, _, mesh = solve_pressure_field(
            dr_x,
            dr_y,
            p_boundary,
            pu.htscp,
            pu.phys_ps,
            pu.sol_ps,
            viz=False,
        )

        verts, triangles, values = mesh_triangles_and_values(uh, mesh)
        x_outline = np.r_[dr_x, dr_x[0]]
        y_outline = np.r_[dr_y, dr_y[0]]

        frames.append(
            {
                "t": float(frame_t[i]),
                "verts": verts,
                "triangles": triangles,
                "values": values,
                "x_outline": x_outline,
                "y_outline": y_outline,
            }
        )
        all_vals.append(values)

        print(f"Solved pressure field {i + 1}/{NUM_FRAMES}")

    values_concat = np.concatenate(all_vals)
    vmin = float(np.min(values_concat))
    vmax = float(np.max(values_concat))

    render_animation(frames, xlim, ylim, vmin, vmax, OUTPUT_GIF, OUTPUT_MP4, FPS)
    print(f"Saved animation to: {OUTPUT_MP4}")


if __name__ == "__main__":
    main()
