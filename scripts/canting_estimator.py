"""Curvature of normal contact-line motion, without second time derivatives.

For the simulation's r_t = sigma*n and T = D_s r, the turning rate is
omega = cross(T, D_s(sigma*n)) / |T|**2 and curvature is omega / |sigma|.
Use the *same discrete D_s* as the ODE; a continuum product rule is not an
identity for finite differences. No elliptic solve or time-spline fit is needed.

The local average uses swept-area quadrature |sigma|*|T| ds dt, so stationary
points and densely saved time samples do not get disproportionate weight.
This is a local spatial average of the same mirror-paired alpha/x quantity,
not the point value of a spatial interpolant. K_alpha remains an arbitrary
common amplitude, calibrated by the plotter only after analysis.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

K_ALPHA_MM = 0.2
TARGET_MM = (1.35, 2.7 * np.cos(np.pi / 6))
TIME_WINDOW = (1 / 18, 17 / 6)
KERNEL_WIDTHS_MM = (0.25, 0.30, 0.35)
SPEED_FLOORS = (1e-7, 1e-6, 1e-5)
DEFAULT_SPEED_FLOOR = 1e-6


def normal_motion_curvature(x, y, speed):
    """Return alpha in radians and swept-area Jacobian for (time, node) arrays.

    The omitted constant material-node spacing cancels in normalized averages.
    At exactly stationary points curvature is undefined (NaN), with zero area.
    """
    if x.shape != y.shape or x.shape != speed.shape or x.ndim != 2:
        raise ValueError("x, y, and signed speed must have matching 2-D shapes")
    def derivative(a):
        return (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / 2
    tx, ty = derivative(x), derivative(y)
    tangent = np.hypot(tx, ty)
    if np.any(tangent <= 0):
        raise ValueError("degenerate contact-line tangent")
    vx, vy = speed * ty / tangent, -speed * tx / tangent
    omega = (tx * derivative(vy) - ty * derivative(vx)) / tangent**2
    alpha = np.full_like(x, np.nan, dtype=float)
    np.divide(K_ALPHA_MM * omega, np.abs(speed), out=alpha, where=speed != 0)
    return alpha, np.abs(speed) * tangent


def slope_samples(results_path: Path, *, stride: int = 1):
    """Mirror-paired secants, locations, area weights, and speeds from a run."""
    with Path(results_path).open('rb') as stream:
        results = pickle.load(stream)
    manifest = json.loads((Path(results_path).parent / 'manifest.json').read_text())
    p = manifest['physical_parameters']
    times = np.asarray(results['out_t'], dtype=float)
    states = np.asarray(results['out_x'], dtype=float)
    if (states.ndim != 2 or states.shape[1] % 12 or len(times) != len(states)
            or len(times) < 10 or not np.all(np.diff(times) > 0)
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(times))):
        raise ValueError('invalid trajectory arrays')
    if times[0] > TIME_WINDOW[0] or times[-1] < TIME_WINDOW[1]:
        raise ValueError('trajectory does not cover the fixed analysis time window')
    # Interpolate the two window endpoints so thinning saved samples does not
    # silently change the physical time interval being averaged.
    inside = np.flatnonzero((times > TIME_WINDOW[0]) & (times < TIME_WINDOW[1]))[::stride]
    selected_t = np.r_[TIME_WINDOW[0], times[inside], TIME_WINDOW[1]]
    selected = np.column_stack([np.interp(selected_t, times, col) for col in states.T])
    x, y, theta = np.split(selected, 3, axis=1)
    width = p['casr_smoothing_rad']
    if width <= 0:
        raise ValueError('this estimator expects the smoothed contact-angle law')
    speed = (p['k_a'] * width * np.logaddexp(0, (theta - p['theta_a']) / width)
             - p['k_r'] * width * np.logaddexp(0, (p['theta_r'] - theta) / width))
    alpha, jacobian = normal_motion_curvature(x, y, speed)
    nodes = x.shape[1]
    mirror = (nodes // 2 - np.arange(nodes)) % nodes
    xm, ym = (x - x[:, mirror]) / 2, (y + y[:, mirror]) / 2
    valid = (xm > 0.25) & np.isfinite(alpha) & np.isfinite(alpha[:, mirror])
    values = np.rad2deg((alpha - alpha[:, mirror])[valid] / (2 * xm[valid]))
    distances_sq = ((xm - TARGET_MM[0])**2 + (ym - TARGET_MM[1])**2)[valid]
    dt = np.diff(selected_t)
    time_weights = np.r_[dt[0], dt[:-1] + dt[1:], dt[-1]] / 2
    areas = (0.5 * (jacobian + jacobian[:, mirror]) * time_weights[:, None])[valid]
    speeds = np.minimum(np.abs(speed), np.abs(speed[:, mirror]))[valid]
    return values, distances_sq, areas, speeds, nodes


def average_samples(samples, *, width: float = 0.30, speed_floor: float = DEFAULT_SPEED_FLOOR):
    values, distances_sq, areas, speeds, _ = samples
    valid = (speeds >= speed_floor) & np.isfinite(values)
    weights = areas[valid] * np.exp(-distances_sq[valid] / (2 * width**2))
    if len(weights) < 4 or weights.sum() <= 0:
        raise ValueError('no moving trajectory samples near the evaluation point')
    return float(np.average(values[valid], weights=weights))


def estimate_kinematic(results_path: Path):
    """Central slope, post-processing half-range, and number of nodes.

    Vary spatial averaging width, the near-stationary speed cutoff, and saved
    sample thinning. This is sensitivity, not a confidence interval or a bound
    on mesh/ODE error. Central values always come from one independent run.
    """
    samples = slope_samples(results_path)
    central = average_samples(samples)
    variants = [average_samples(samples, width=w, speed_floor=v)
                for w in KERNEL_WIDTHS_MM for v in SPEED_FLOORS]
    variants.extend(average_samples(slope_samples(results_path, stride=s)) for s in (2, 4))
    return central, 0.5 * (max(variants) - min(variants)), samples[-1]
