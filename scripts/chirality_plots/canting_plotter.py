#!/usr/bin/env python3
"""Plot canting against the simulated chirality estimate.

This analysis deliberately works from the saved contact-line trajectories.  The
old implementation first built a scattered 2-D alpha field and then took a
two-point difference of that interpolant.  Second time derivatives plus two
layers of interpolation made that estimate visibly oscillatory.  Here, alpha
is computed on each smoothed material-point trajectory and opposite points are
paired before doing a local, robust average near the requested location.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

K_ALPHA_MM = 0.2
R_EFF_MM = 0.9 * 3.0
ANG_EVAL_RAD = np.deg2rad(30.0)
KNOT_INTERVALS = (3, 4, 5, 6, 7)
KERNEL_WIDTHS_MM = (0.25, 0.30, 0.35)
DEFAULT_KNOT_INTERVALS = 5
DEFAULT_KERNEL_WIDTH_MM = 0.30


@dataclass(frozen=True)
class Estimate:
    height_ratio: float
    chirality_slope: float
    numerical_error: float
    nodes: int
    run_id: str
    source: Path


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def normalized_height(setup: dict) -> float:
    """Return (h_front - h_back) / (h_front + h_back)."""
    front = float(setup["ht_front"])
    back = float(setup.get("ht_back", 0.1))
    if front + back == 0:
        raise ValueError("ht_front + ht_back is zero")
    return (front - back) / (front + back)


def _trajectory_arrays(results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = load_pickle(results_path)
    times = np.asarray(result["out_t"], dtype=float)
    states = np.asarray(result["out_x"], dtype=float)
    if states.ndim != 2 or states.shape[1] % 3:
        raise ValueError(f"invalid state array shape {states.shape}")
    if len(times) != len(states) or len(times) < 10:
        raise ValueError("result has too few or mismatched time samples")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(states)):
        raise ValueError("result contains non-finite values")

    # Match the established chirality-map analysis: omit startup and the final
    # 5%, where spline boundary derivatives are least reliable.
    stop = int(0.95 * len(times))
    times = times[2:stop]
    states = states[2:stop]
    nodes = states.shape[1] // 3
    return times, states[:, :nodes], states[:, nodes : 2 * nodes]


def _fit_trajectories(
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    knot_intervals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fitted x, y, and signed trajectory curvature at sample times."""
    knots = np.linspace(times[0], times[-1], knot_intervals + 1)[1:-1]
    fit_x = np.empty_like(xs)
    fit_y = np.empty_like(ys)
    alpha = np.full_like(xs, np.nan)

    for node in range(xs.shape[1]):
        sx = LSQUnivariateSpline(times, xs[:, node], knots, k=3)
        sy = LSQUnivariateSpline(times, ys[:, node], knots, k=3)
        xt, yt = sx(times, 1), sy(times, 1)
        xtt, ytt = sx(times, 2), sy(times, 2)
        speed_sq = xt * xt + yt * yt
        valid = speed_sq > np.finfo(float).eps

        fit_x[:, node] = sx(times)
        fit_y[:, node] = sy(times)
        alpha[valid, node] = K_ALPHA_MM * (
            xt[valid] * ytt[valid] - yt[valid] * xtt[valid]
        ) / np.power(speed_sq[valid], 1.5)

    return fit_x, fit_y, alpha


def _huber_location(values: np.ndarray, weights: np.ndarray) -> float:
    """Robust weighted location; protects the derivative estimate from spikes."""
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if len(values) < 4:
        raise ValueError("too few samples near the evaluation point")
    location = float(np.average(values, weights=weights))
    for _ in range(30):
        residual = values - location
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        scale = max(float(scale), np.finfo(float).eps)
        robust = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), scale * 1e-12))
        updated = float(np.average(values, weights=weights * robust))
        if abs(updated - location) < 1e-10:
            return updated
        location = updated
    return location


def estimate_slope(
    results_path: Path,
    *,
    knot_intervals: int = DEFAULT_KNOT_INTERVALS,
    kernel_width_mm: float = DEFAULT_KERNEL_WIDTH_MM,
) -> tuple[float, int]:
    """Estimate d(alpha)/dx in degrees/mm from one saved simulation."""
    times, xs, ys = _trajectory_arrays(results_path)
    fit_x, fit_y, alpha = _fit_trajectories(times, xs, ys, knot_intervals)
    nodes = xs.shape[1]

    # Node i+n/2 is its mirror across x=0 for the circular initial mesh.  Pairing
    # them cancels common-mode interpolation/differentiation error and enforces
    # the expected odd chirality response without altering the raw trajectories.
    opposite = (nodes // 2 - np.arange(nodes)) % nodes
    target_x = R_EFF_MM * np.sin(ANG_EVAL_RAD)
    target_y = R_EFF_MM * np.cos(ANG_EVAL_RAD)
    slopes: list[np.ndarray] = []
    distances_sq: list[np.ndarray] = []

    for node, mirror in enumerate(opposite):
        x_mid = 0.5 * (fit_x[:, node] - fit_x[:, mirror])
        y_mid = 0.5 * (fit_y[:, node] + fit_y[:, mirror])
        valid = (
            (x_mid > 0.25)
            & np.isfinite(alpha[:, node])
            & np.isfinite(alpha[:, mirror])
        )
        slopes.append(np.rad2deg((alpha[valid, node] - alpha[valid, mirror]) / (2.0 * x_mid[valid])))
        distances_sq.append((x_mid[valid] - target_x) ** 2 + (y_mid[valid] - target_y) ** 2)

    values = np.concatenate(slopes)
    distance_sq = np.concatenate(distances_sq)
    weights = np.exp(-distance_sq / (2.0 * kernel_width_mm**2))
    return _huber_location(values, weights), nodes


def _matching_flat_result(base_dir: Path) -> Path | None:
    """Use the matching 20-node flat run, not its legacy 8-node replacement."""
    for run_dir in base_dir.iterdir():
        setup_path = run_dir / "setup.pkl"
        if not setup_path.exists() or abs(normalized_height(load_pickle(setup_path))) > 1e-12:
            continue
        candidate = run_dir / "og.pkl"
        if candidate.exists():
            return candidate
    return None


def discover_runs(base_dir: Path) -> list[tuple[float, Path, str]]:
    discovered: list[tuple[float, Path, str]] = []
    flat_override = _matching_flat_result(base_dir)
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        setup_path, result_path = run_dir / "setup.pkl", run_dir / "results.pkl"
        if not setup_path.exists() or not result_path.exists():
            continue
        ratio = normalized_height(load_pickle(setup_path))
        if abs(ratio) < 1e-12 and flat_override is not None:
            result_path = flat_override
        discovered.append((ratio, result_path, run_dir.name))
    return discovered


def analyze_run(ratio: float, result_path: Path, run_id: str) -> Estimate:
    sensitivity = []
    nodes = 0
    for intervals in KNOT_INTERVALS:
        for width in KERNEL_WIDTHS_MM:
            value, nodes = estimate_slope(
                result_path, knot_intervals=intervals, kernel_width_mm=width
            )
            sensitivity.append(value)
    central, nodes = estimate_slope(result_path)
    # Half-range is intentionally conservative and exposes post-processing
    # sensitivity; it is not a claim about stochastic simulation uncertainty.
    error = 0.5 * (max(sensitivity) - min(sensitivity))
    return Estimate(ratio, central, error, nodes, run_id, result_path)


def analyze_directory(base_dir: Path) -> list[Estimate]:
    estimates = []
    for ratio, result_path, run_id in discover_runs(base_dir):
        try:
            estimates.append(analyze_run(ratio, result_path, run_id))
        except Exception as exc:
            print(f"[skip] {run_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return sorted(estimates, key=lambda estimate: estimate.height_ratio)


def write_csv(
    path: Path,
    primary: list[Estimate],
    validation: list[Estimate],
    discretization_fraction: float,
) -> None:
    def portable_source(source: Path) -> str:
        try:
            return str(source.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(source)

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "series",
                "height_ratio",
                "chirality_slope_deg_per_mm",
                "postprocessing_error",
                "estimated_total_numerical_error",
                "nodes",
                "run_id",
                "source",
            ]
        )
        for series, estimates in (("primary", primary), ("resolution_validation", validation)):
            for item in estimates:
                writer.writerow(
                    [
                        series,
                        item.height_ratio,
                        item.chirality_slope,
                        item.numerical_error,
                        np.hypot(item.numerical_error, discretization_fraction * item.chirality_slope),
                        item.nodes,
                        item.run_id,
                        portable_source(item.source),
                    ]
                )


def saturating_response(
    x: np.ndarray,
    baseline: float,
    amplitude: float,
    scale: float,
    shape: float,
) -> np.ndarray:
    """Monotone stretched-exponential saturation response."""
    return baseline + amplitude * (1.0 - np.exp(-np.power(x / scale, shape)))


def fit_saturating_response(estimates: list[Estimate], discretization_fraction: float = 0.0):
    x = np.array([item.height_ratio for item in estimates])
    y = np.array([item.chirality_slope for item in estimates])
    sigma = np.maximum(
        np.hypot(
            np.array([item.numerical_error for item in estimates]),
            discretization_fraction * y,
        ),
        1e-6,
    )
    params, covariance = curve_fit(
        saturating_response,
        x,
        y,
        p0=(float(y[0]), float(max(y) - y[0]), 0.07, 1.2),
        sigma=sigma,
        absolute_sigma=True,
        bounds=((-np.inf, 0.0, 1e-4, 0.1), (np.inf, np.inf, 10.0, 5.0)),
        maxfev=20000,
    )
    return params, covariance


def merge_estimates(*groups: list[Estimate]) -> list[Estimate]:
    """Combine completed run directories, preferring the first for duplicates."""
    merged: dict[float, Estimate] = {}
    for group in groups:
        for estimate in group:
            merged.setdefault(round(estimate.height_ratio, 12), estimate)
    return sorted(merged.values(), key=lambda item: item.height_ratio)


def plot_canting(
    ax,
    base_dir: Path | None = None,
    *,
    fallback_dir: Path | None = None,
    validation_dir: Path | None = None,
) -> tuple[list[Estimate], list[Estimate], np.ndarray, np.ndarray, float]:
    base_dir = Path(base_dir or (REPO_ROOT / "canting_production_20")).resolve()
    fallback_dir = Path(fallback_dir or (REPO_ROOT / "canting_production_lsoda_20")).resolve()
    validation_dir = Path(validation_dir or (REPO_ROOT / "canting_mesh_reference_20")).resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(f"primary data directory not found: {base_dir}")

    primary = merge_estimates(
        analyze_directory(base_dir),
        analyze_directory(fallback_dir) if fallback_dir.is_dir() else [],
    )
    validation = analyze_directory(validation_dir) if validation_dir.is_dir() else []
    if not primary:
        raise RuntimeError(f"no valid canting results found in {base_dir}")

    x = np.array([item.height_ratio for item in primary])
    y = np.array([item.chirality_slope for item in primary])
    relative_mesh_differences = []
    for check in validation:
        matches = [item for item in primary if np.isclose(item.height_ratio, check.height_ratio)]
        if matches and matches[0].chirality_slope != 0:
            relative_mesh_differences.append(
                abs(check.chirality_slope - matches[0].chirality_slope) / abs(matches[0].chirality_slope)
            )
    discretization_fraction = max(relative_mesh_differences, default=0.0)
    error = np.hypot(
        np.array([item.numerical_error for item in primary]),
        discretization_fraction * y,
    )
    ax.errorbar(
        x,
        y,
        yerr=error,
        fmt="o",
        color="tab:blue",
        capsize=3,
        ms=6,
        label=f"Simulation ({primary[0].nodes} nodes)",
    )

    if validation:
        vx = np.array([item.height_ratio for item in validation])
        vy = np.array([item.chirality_slope for item in validation])
        ve = np.array([item.numerical_error for item in validation])
        ax.errorbar(
            vx,
            vy,
            yerr=ve,
            fmt="s",
            mfc="none",
            color="tab:orange",
            capsize=3,
            ms=7,
            label="Finer-mesh check",
        )

    params, covariance = fit_saturating_response(primary, discretization_fraction)
    sample_x = np.linspace(0.0, max(x), 400)
    sample_y = saturating_response(sample_x, *params)
    baseline, amplitude, scale, shape = params
    scaled_power = np.power(sample_x / scale, shape)
    exponential = np.exp(-scaled_power)
    shape_column = np.zeros_like(sample_x)
    positive = sample_x > 0
    shape_column[positive] = (
        amplitude
        * exponential[positive]
        * scaled_power[positive]
        * np.log(sample_x[positive] / scale)
    )
    jacobian = np.column_stack(
        (
            np.ones_like(sample_x),
            1.0 - exponential,
            -amplitude * exponential * shape * scaled_power / scale,
            shape_column,
        )
    )
    fit_std = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian), 0.0))
    ax.plot(
        sample_x,
        sample_y,
        color="0.2",
        lw=2.0,
        label="Saturating fit",
    )
    ax.fill_between(sample_x, sample_y - fit_std, sample_y + fit_std, color="0.2", alpha=0.14, linewidth=0)

    ax.set_xlabel(r"$(h_{\mathrm{front}}-h_{\mathrm{back}})/(h_{\mathrm{front}}+h_{\mathrm{back}})$")
    ax.set_ylabel(r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    return primary, validation, params, covariance, discretization_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "canting_production_20")
    parser.add_argument("--fallback-dir", type=Path, default=REPO_ROOT / "canting_production_lsoda_20")
    parser.add_argument("--validation-dir", type=Path, default=REPO_ROOT / "canting_mesh_reference_20")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "canting_clean.png")
    parser.add_argument("--csv", type=Path, default=SCRIPT_DIR / "canting_clean.csv")
    parser.add_argument("--fit-json", type=Path, default=SCRIPT_DIR / "canting_clean_fit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    primary, validation, params, covariance, discretization_fraction = plot_canting(
        ax,
        args.data_dir,
        fallback_dir=args.fallback_dir,
        validation_dir=args.validation_dir,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.fit_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    write_csv(args.csv, primary, validation, discretization_fraction)
    fit_payload = {
        "model": "baseline + amplitude * (1 - exp(-(height_ratio / scale)**shape))",
        "parameters": dict(zip(("baseline", "amplitude", "scale", "shape"), map(float, params))),
        "covariance": np.asarray(covariance).tolist(),
        "discretization_fraction_from_mesh_check": float(discretization_fraction),
        "unweighted_residual_rms_deg_per_mm": float(
            np.sqrt(
                np.mean(
                    (
                        saturating_response(
                            np.array([item.height_ratio for item in primary]),
                            *params,
                        )
                        - np.array([item.chirality_slope for item in primary])
                    )
                    ** 2
                )
            )
        ),
        "plateau_deg_per_mm": float(params[0] + params[1]),
    }
    args.fit_json.write_text(json.dumps(fit_payload, indent=2) + "\n")
    print(f"Saved {args.output}, {args.csv}, and {args.fit_json} ({len(primary)} primary, {len(validation)} validation points)")
    for item in primary:
        total_error = np.hypot(item.numerical_error, discretization_fraction * item.chirality_slope)
        print(f"  {item.height_ratio:8.5f}  {item.chirality_slope:8.5f} +/- {total_error:.5f}")


if __name__ == "__main__":
    main()
