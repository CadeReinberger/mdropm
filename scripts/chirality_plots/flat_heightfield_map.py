#!/usr/bin/env python3
"""Export the current kinematic chirality predictor on the RIV crop.

Each interpolation triangle belongs to an actual swept contact-line cell.
The unswept interior and points outside the initial drop remain NaN.
data.pkl is the same ndarray interface and sign convention as chirality_map.py:
rows increase in physical y, columns increase in x, values are -alpha in degrees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flat-heightfield-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import matplotlib.tri as mtri
import numpy as np

from canting_estimator import K_ALPHA_MM, DEFAULT_SPEED_FLOOR, normal_motion_curvature


def atomic_pickle(value, destination):
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".map-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(value, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def trajectory_field(results, manifest, speed_floor=DEFAULT_SPEED_FLOOR):
    times = np.asarray(results["out_t"], dtype=float)
    states = np.asarray(results["out_x"], dtype=float)
    if (states.ndim != 2 or states.shape[1] % 3 or len(times) != len(states)
            or len(times) < 2 or not np.all(np.diff(times) > 0)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(times))):
        raise ValueError("invalid trajectory arrays")
    x, y, theta = np.split(states, 3, axis=1)
    if x.shape[1] != manifest["nodes"]:
        raise ValueError("manifest and trajectory have different node counts")
    p = manifest["physical_parameters"]
    width = p["casr_smoothing_rad"]
    if width <= 0:
        raise ValueError("the current predictor expects the smoothed contact-angle law")
    speed = (p["k_a"]*width*np.logaddexp(0, (theta-p["theta_a"])/width)
             - p["k_r"]*width*np.logaddexp(0, (p["theta_r"]-theta)/width))
    if np.any(speed >= 0):
        raise ValueError("swept-strip map requires a monotonically receding contact line")
    alpha, area = normal_motion_curvature(x, y, speed)
    valid = np.isfinite(alpha) & (np.abs(speed) >= speed_floor)
    # Connect neighboring material points at successive saved times, including
    # the periodic seam. Do not triangulate across the unswept interior.
    ids = np.arange(x.size).reshape(x.shape)
    a, b = ids[:-1].ravel(), np.roll(ids[:-1], -1, axis=1).ravel()
    c, d = ids[1:].ravel(), np.roll(ids[1:], -1, axis=1).ravel()
    # Reflect the diagonal with x so interpolation preserves the left/right
    # antisymmetry of the symmetric run, including its rapid initial transient.
    right = (x.ravel()[a] + x.ravel()[b] + x.ravel()[c] + x.ravel()[d]) >= 0
    triangles = np.vstack((np.c_[a[right], b[right], c[right]], np.c_[b[right], d[right], c[right]],
                           np.c_[a[~right], b[~right], d[~right]], np.c_[a[~right], d[~right], c[~right]]))
    vertices = np.c_[x.ravel(), y.ravel()]
    edges1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    edges2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    signed_area2 = edges1[:, 0]*edges2[:, 1] - edges1[:, 1]*edges2[:, 0]
    if np.any(signed_area2 < -1e-12):
        raise ValueError("folded swept cells; cannot make a single-valued deposition map")
    mask = ~np.all(valid.ravel()[triangles], axis=1) | (signed_area2 <= 1e-14)
    triangulation = mtri.Triangulation(x.ravel(), y.ravel(), triangles, mask=mask)
    values = -np.rad2deg(np.where(valid, alpha, 0))
    interpolator = mtri.LinearTriInterpolator(triangulation, values.ravel())
    return times, x, y, speed, alpha, valid, interpolator


def export_map(results_path, output, *, grid_size=1000, speed_floor=DEFAULT_SPEED_FLOOR):
    results_path, output = Path(results_path), Path(output)
    with results_path.open("rb") as stream:
        results = pickle.load(stream)
    manifest = json.loads(results_path.with_name("manifest.json").read_text())
    times, x, y, speed, alpha, valid, interpolator = trajectory_field(results, manifest, speed_floor)
    x_eval = np.linspace(-1.5, 1.5, grid_size)
    y_eval = np.linspace(0.0, 3.25, grid_size)
    xx, yy = np.meshgrid(x_eval, y_eval)
    field = interpolator(xx, yy).filled(np.nan)
    covered = np.isfinite(field)
    if not covered.any():
        raise ValueError("no swept trajectory intersects the requested crop")
    crop_points = np.c_[xx.ravel(), yy.ravel()]
    initially_inside = PolygonPath(np.c_[x[0], y[0]]).contains_points(crop_points).reshape(xx.shape)
    finally_inside = PolygonPath(np.c_[x[-1], y[-1]]).contains_points(crop_points).reshape(xx.shape)
    expected_swept = initially_inside & ~finally_inside
    mirror = (x.shape[1]//2 - np.arange(x.shape[1])) % x.shape[1]
    symmetry = max(float(np.max(np.abs(x+x[:, mirror]))), float(np.max(np.abs(y-y[:, mirror]))))
    output.mkdir(parents=True, exist_ok=True)
    atomic_pickle(field, output / "data.pkl")
    np.savez_compressed(output / "map_fields.npz", x_mm=x_eval, y_mm=y_eval,
                        data=field, alpha_degrees=-field, valid_mask=covered,
                        initially_inside=initially_inside, finally_inside=finally_inside)
    np.savez_compressed(output / "predictor_samples.npz", time=times, x_mm=x, y_mm=y,
                        signed_speed=speed, alpha_radians=alpha, valid=valid)
    finite = np.abs(field[covered])
    limit = float(np.percentile(finite, 99)) or 1.0
    cm = plt.get_cmap("seismic").copy()
    cm.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    im = ax.pcolormesh(x_eval, y_eval, -field, cmap=cm, vmin=-limit, vmax=limit, shading="auto", rasterized=True)
    for index, style, label in [(0, "-", "Initial contact line"), (-1, "--", "Final contact line")]:
        ax.plot(np.r_[x[index], x[index, 0]], np.r_[y[index], y[index, 0]], style, color="black", lw=1.4, label=label)
    ax.set(xlim=(x_eval[0], x_eval[-1]), ylim=(y_eval[0], y_eval[-1]), aspect="equal",
           xlabel="Simulation x (mm)", ylabel="Simulation y (mm)",
           title="Flat-heightfield chirality prediction")
    fig.colorbar(im, ax=ax, label=r"$\alpha$ (degrees), $K_\alpha = 0.2$ mm", extend="both")
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(output / "chirality_map.png", dpi=250)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for index in np.unique(np.linspace(0, len(times)-1, 18, dtype=int)):
        axes[0].plot(np.r_[x[index], x[index, 0]], np.r_[y[index], y[index, 0]], lw=0.8,
                     color=plt.cm.viridis(index/(len(times)-1)))
    axes[0].plot([-1.5, 1.5, 1.5, -1.5, -1.5], [0, 0, 3.25, 3.25, 0], "r--", lw=1.1)
    axes[0].set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)", title="Contact-line evolution and map crop")
    areas = np.abs(np.sum(x*np.roll(y,-1,axis=1)-y*np.roll(x,-1,axis=1),axis=1))/2
    axes[1].plot(times, areas/areas[0])
    axes[1].set(xlabel="Model time (min)", ylabel="Area / initial area", title="Drying trajectory")
    axes[1].grid(alpha=0.3)
    fig.savefig(output / "trajectory.png", dpi=200)
    plt.close(fig)
    stats = dict(format_version=1, source_results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(),
                 source_manifest_sha256=hashlib.sha256(results_path.with_name("manifest.json").read_bytes()).hexdigest(),
                 predictor_source_sha256=hashlib.sha256(Path(sys.modules[normal_motion_curvature.__module__].__file__).read_bytes()).hexdigest(),
                 exporter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 predictor="canting_estimator.normal_motion_curvature", K_alpha_mm=K_ALPHA_MM,
                 amplitude_calibration="none; the existing arbitrary K_alpha is retained",
                 interpolation="linear within swept time/material cells; no smoothing or extrapolation",
                 speed_floor=speed_floor, nodes=x.shape[1], saved_samples=len(times),
                 time_range=[float(times[0]),float(times[-1])], final_area_ratio=float(areas[-1]/areas[0]),
                 final_top_y_mm=float(y[-1].max()), x_mm=[-1.5,1.5], y_mm=[0.0,3.25],
                 grid_shape=list(field.shape), data_pickle_sign="minus alpha in degrees",
                 data_pickle_layout="rows increasing y, columns increasing x", missing_value="NaN",
                 crop_coverage_fraction=float(covered.mean()),
                 initial_drop_crop_coverage_fraction=float(covered[initially_inside].mean()),
                 expected_swept_coverage_fraction=float(covered[expected_swept].mean()),
                 remaining_liquid_crop_fraction=float(finally_inside[initially_inside].mean()),
                 data_range_degrees=[float(np.nanmin(field)),float(np.nanmax(field))],
                 mirror_geometry_error_mm=symmetry,
                 mirror_map_error_degrees=float(np.nanmax(np.abs(field+field[:, ::-1]))),
                 physical_map_color_limit_degrees=limit)
    (output / "map_metadata.json").write_text(json.dumps(stats, indent=2)+"\n")
    # Check the crucial serialized artifact by loading it from disk again.
    with (output / "data.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(restored, field)
    print(json.dumps(stats, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--speed-floor", type=float, default=DEFAULT_SPEED_FLOOR)
    args = parser.parse_args()
    if args.grid_size < 2 or not np.isfinite(args.speed_floor) or args.speed_floor <= 0:
        parser.error("grid size must be at least two and speed floor finite and positive")
    export_map(args.results, args.output_dir, grid_size=args.grid_size, speed_floor=args.speed_floor)


if __name__ == "__main__":
    main()
