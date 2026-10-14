#!/usr/bin/env python3
"""Animate the liquid pressure-gradient magnitude and flow directions.

Reads a saved trajectory and its sibling setup/manifest. Each frame uses the
refined FEM pressure solve and L2-projected gradient on its current geometry.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.plot import vtk_mesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import heightscape
import tapescape
from main import RebasedPhase
from physical_params import physical_params
from problem_universe import problem_universe
from solver_params import solver_params
from animation_utils import (
    add_animation_arguments, animation_writers, interpolate_states,
    load_results, validate_animation_arguments,
)

AXIS_PADDING_FRAC = 0.05
CMAP = "viridis"
STREAM_DENSITY = 0.55
VECTOR_GRID_SIZE = 64


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
    if "heightscape" in setup:
        hs = rebuild_heightscape(setup["heightscape"])
        ts = rebuild_tapescape(setup["tapescape"])
        sps = rebuild_solver_params(setup["solver_params"])
        pps = physical_params(**setup["physical_params"])
        start_drop = setup["start_drop"]
    else:
        # Canting sweeps store compact geometry in setup.pkl and full
        # physical/numerical parameters in their manifest.
        manifest = json.loads(settings_or_setup_path.with_name("manifest.json").read_text())
        front, back = setup["ht_front"], setup["ht_back"]
        hs = heightscape.constructors.rect_interp_htscape((10, 10), (back, front, front, back))
        ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)
        sps = rebuild_solver_params(manifest["solver_parameters"])
        pps = physical_params(**manifest["physical_parameters"])
        start_drop = {"L": 2*np.pi*setup["radius"]}
    pu = problem_universe(hs, ts, sps, pps)
    return pu, start_drop


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


def sample_gradient_field(verts, triangles, grad_x, grad_y, xlim, ylim, grid_size):
    """Interpolate the recovered FEM gradient, leaving exterior points masked."""
    triangulation = mtri.Triangulation(verts[:, 0], verts[:, 1], triangles)
    x_grid = np.linspace(*xlim, grid_size)
    y_grid = np.linspace(*ylim, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    gx = mtri.LinearTriInterpolator(triangulation, grad_x)(xx, yy).filled(np.nan)
    gy = mtri.LinearTriInterpolator(triangulation, grad_y)(xx, yy).filled(np.nan)
    return dict(x_grid=x_grid, y_grid=y_grid, grad_x=gx, grad_y=gy, grad_mag=np.hypot(gx, gy))


def frame_bounds(xs: np.ndarray, ys: np.ndarray, pad_frac: float):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    dx = xmax - xmin
    dy = ymax - ymin
    padx = pad_frac * (dx if dx > 0 else 1.0)
    pady = pad_frac * (dy if dy > 0 else 1.0)

    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)


def render_animation(frames, xlim, ylim, vmin, vmax, args):
    fig = plt.figure(figsize=(7.2, 6.0), dpi=args.dpi)
    with animation_writers(fig, args) as writers:
        for index, frame in enumerate(frames):
            fig.clear()
            ax = fig.add_subplot(111)
            color = ax.pcolormesh(
                frame["x_grid"], frame["y_grid"], np.ma.masked_invalid(frame["grad_mag"]),
                shading="auto", cmap=CMAP, vmin=vmin, vmax=vmax,
            )
            if not args.no_streamlines:
                ax.streamplot(
                    frame["x_grid"], frame["y_grid"],
                    np.ma.masked_invalid(frame["grad_x"]), np.ma.masked_invalid(frame["grad_y"]),
                    color=(0.08, 0.08, 0.08, 0.55), density=STREAM_DENSITY,
                    linewidth=0.7, arrowsize=0.8, minlength=0.08,
                )
            ax.plot(frame["x_outline"], frame["y_outline"], color="black", linewidth=1.8)
            ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="x (mm)", ylabel="y (mm)",
                   title=f"Pressure gradient magnitude, t = {frame['t']:.3f} min")
            fig.colorbar(color, ax=ax, label=r"$|\nabla p|$")
            fig.tight_layout()
            for writer in writers:
                writer.grab_frame()
            if index == 0 or (index + 1) % 20 == 0 or index == len(frames) - 1:
                print(f"Rendered frame {index + 1}/{len(frames)}", flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_animation_arguments(parser, "pressure_field_animation")
    parser.add_argument("--setup", type=Path, help="defaults to setup.pkl beside --results")
    parser.add_argument("--grid-size", type=int, default=VECTOR_GRID_SIZE)
    parser.add_argument("--no-streamlines", action="store_true")
    args = parser.parse_args()
    validate_animation_arguments(parser, args)
    if args.grid_size < 4:
        parser.error("grid size must be at least four")
    setup_path = args.setup or args.results.with_name("setup.pkl")
    pu, start_drop = load_problem_universe(setup_path)
    raw_t, raw_state = load_results(args.results)

    n = raw_state.shape[1] // 3
    if n < 3:
        raise ValueError("Need at least 3 boundary points")

    ds = float(start_drop.get("L", 2.0 * np.pi) / n)

    frame_t = np.linspace(float(raw_t[0]), float(raw_t[-1]), args.frames)
    frame_state = interpolate_states(raw_t, raw_state, frame_t)

    all_x = frame_state[:, :n]
    all_y = frame_state[:, n : 2 * n]
    xlim, ylim = frame_bounds(all_x, all_y, AXIS_PADDING_FRAC)

    frames = []
    all_vals = []

    for i in range(args.frames):
        cur = frame_state[i]
        dr_x = cur[:n]
        dr_y = cur[n : 2 * n]
        dr_theta = cur[2 * n : 3 * n]

        p_boundary = compute_boundary_pressure(dr_x, dr_y, dr_theta, pu, ds)
        phase = RebasedPhase(dr_x, dr_y, pu, gas=False)
        phase.gradients(dr_x, dr_y, p_boundary)
        verts, triangles, grad_x = mesh_triangles_and_values(phase.projection.u.sub(0), phase.mesh)
        _, _, grad_y = mesh_triangles_and_values(phase.projection.u.sub(1), phase.mesh)
        x_outline = np.r_[dr_x, dr_x[0]]
        y_outline = np.r_[dr_y, dr_y[0]]

        grad_field = sample_gradient_field(verts, triangles, grad_x, grad_y, xlim, ylim, args.grid_size)

        frames.append(
            {
                "t": float(frame_t[i]),
                "x_grid": grad_field["x_grid"],
                "y_grid": grad_field["y_grid"],
                "grad_x": grad_field["grad_x"],
                "grad_y": grad_field["grad_y"],
                "grad_mag": grad_field["grad_mag"],
                "x_outline": x_outline,
                "y_outline": y_outline,
            }
        )
        all_vals.append(grad_field["grad_mag"][np.isfinite(grad_field["grad_mag"])])

        print(f"Solved pressure field {i + 1}/{args.frames}", flush=True)

    values_concat = np.concatenate(all_vals)
    vmin = float(np.min(values_concat))
    vmax = float(np.max(values_concat))

    render_animation(frames, xlim, ylim, vmin, vmax, args)


if __name__ == "__main__":
    main()
