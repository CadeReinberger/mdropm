#!/usr/bin/env python3
"""Plot canting against the simulated chirality estimate.

Refined runs use the discrete normal-motion identity for curvature and a local
swept-area average of mirror-paired slopes (see canting_estimator.py). There is
no second time derivative or fit across heights in the individual data points.
The legacy trajectory-spline estimator remains available for old saved runs.

The figure uses twice the saved plate height and calibrates the arbitrary angle
amplitude to a slope of 1 degree/mm at a displayed front height of 400 microns.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import curve_fit

try:
    from .canting_estimator import K_ALPHA_MM, estimate_kinematic
except ImportError:
    from canting_estimator import K_ALPHA_MM, estimate_kinematic

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REFINED_DATA_DIR = SCRIPT_DIR / "canting_refinement" / "production80"
REFINED_VALIDATION_DIR = SCRIPT_DIR / "canting_refinement" / "pilot160"

HEIGHT_CONVENTION_MULTIPLIER = 2.0
CALIBRATION_FRONT_HEIGHT_UM = 400.0
CALIBRATION_SLOPE_DEG_PER_MM = 1.0  # (+2 - (-2)) degrees / (+2 - (-2)) mm
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
    method: str = "trajectory_spline"


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

    # Reflection across x=0 maps material node i to n/2-i. Pairing these nodes
    # cancels common-mode interpolation/differentiation error and enforces
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
    manifest_path = result_path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("numerical_method", "").startswith("persistent_radial_mesh_"):
            central, error, nodes = estimate_kinematic(result_path)
            return Estimate(ratio, central, error, nodes, run_id, result_path, "normal_motion_swept_area")
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
                "h_front_um",
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
                setup = load_pickle(item.source.parent / "setup.pkl")
                writer.writerow(
                    [
                        series,
                        item.height_ratio,
                        HEIGHT_CONVENTION_MULTIPLIER * 1000.0 * float(setup["ht_front"]),
                        item.chirality_slope,
                        item.numerical_error,
                        item.numerical_error,
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
    decay: float = 0.0,
) -> np.ndarray:
    """Empirical exponential rise, optionally allowing a gentle high-cant decline.

    decay=0 is exactly the legacy monotone saturation model. This curve is a
    presentation fit, not a replacement for any simulation or a physics law.
    """
    return baseline + amplitude * (1.0 - np.exp(-np.power(x / scale, shape))) * np.exp(-decay * x)


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
    refined = all(item.method == "normal_motion_swept_area" for item in estimates)
    initial = [float(y[0]), float(max(y) - y[0]), 0.07, 1.2]
    lower, upper = [-np.inf, 0.0, 1e-4, 0.1], [np.inf, np.inf, 10.0, 5.0]
    if refined:
        initial.append(0.1)
        lower.append(0.0)
        upper.append(10.0)
    params, covariance = curve_fit(
        saturating_response,
        x,
        y,
        p0=initial,
        sigma=sigma,
        absolute_sigma=True,
        bounds=(lower, upper),
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
) -> tuple[list[Estimate], list[Estimate], np.ndarray, np.ndarray, float, float]:
    base_dir = Path(base_dir or REFINED_DATA_DIR).resolve()
    fallback_dir = Path(fallback_dir).resolve() if fallback_dir is not None else None
    if not base_dir.is_dir():
        raise FileNotFoundError(f"primary data directory not found: {base_dir}")

    primary = merge_estimates(
        analyze_directory(base_dir),
        analyze_directory(fallback_dir) if fallback_dir is not None and fallback_dir.is_dir() else [],
    )
    if not primary:
        raise RuntimeError(f"no valid canting results found in {base_dir}")
    methods = {item.method for item in primary}
    if len(methods) != 1:
        raise ValueError("do not combine the legacy and refined estimators into one primary curve")
    default_validation = REFINED_VALIDATION_DIR if methods == {"normal_motion_swept_area"} else REPO_ROOT / "canting_mesh_reference_20"
    validation_dir = Path(validation_dir or default_validation).resolve()
    validation = analyze_directory(validation_dir) if validation_dir.is_dir() else []
    if any(item.method not in methods for item in validation):
        raise ValueError("resolution checks must use the same estimator as the primary curve")

    x = np.array([item.height_ratio for item in primary])
    # The production sweep holds the back height fixed.  Convert the existing
    # fit coordinate r=(h_front-h_back)/(h_front+h_back) to the doubled-height
    # convention in um.  Trajectory coordinates remain in mm.
    setup = load_pickle(primary[0].source.parent / "setup.pkl")
    back_height_um = HEIGHT_CONVENTION_MULTIPLIER * 1000.0 * float(setup.get("ht_back", 0.1))
    front_height_um = back_height_um * (1.0 + x) / (1.0 - x)
    relative_mesh_differences = []
    for check in validation:
        matches = [item for item in primary if np.isclose(item.height_ratio, check.height_ratio)]
        if matches and matches[0].chirality_slope != 0:
            relative_mesh_differences.append(
                abs(check.chirality_slope - matches[0].chirality_slope) / abs(matches[0].chirality_slope)
            )
    discretization_fraction = max(relative_mesh_differences, default=0.0)

    # Calibrate only this figure and its exports; the shared trajectory
    # estimator retains its original K_ALPHA_MM for the other analyses.
    params, covariance = fit_saturating_response(primary, discretization_fraction)
    reference_ratio = (CALIBRATION_FRONT_HEIGHT_UM - back_height_um) / (
        CALIBRATION_FRONT_HEIGHT_UM + back_height_um
    )
    reference_slope = float(saturating_response(reference_ratio, *params))
    if not np.isfinite(reference_slope) or reference_slope <= 0:
        raise ValueError("canting calibration requires a finite, positive reference slope")
    chirality_scale = CALIBRATION_SLOPE_DEG_PER_MM / reference_slope
    primary, validation = (
        [
            replace(
                item,
                chirality_slope=chirality_scale * item.chirality_slope,
                numerical_error=chirality_scale * item.numerical_error,
            )
            for item in estimates
        ]
        for estimates in (primary, validation)
    )
    parameter_scale = np.ones(len(params))
    parameter_scale[:2] = chirality_scale
    params = params * parameter_scale
    covariance = covariance * np.outer(parameter_scale, parameter_scale)
    y = np.array([item.chirality_slope for item in primary])
    error = np.array([item.numerical_error for item in primary])
    ax.errorbar(
        front_height_um,
        y,
        yerr=error,
        fmt="o",
        color="tab:blue",
        capsize=3,
        ms=6,
        label=f"Simulation ({primary[0].nodes} nodes)",
    )

    sample_x = np.linspace(0.0, max(x), 400)
    if 0.0 <= reference_ratio <= max(x):
        sample_x = np.sort(
            np.append(
                sample_x[~np.isclose(sample_x, reference_ratio, rtol=0.0, atol=1e-12)],
                reference_ratio,
            )
        )
    sample_y = saturating_response(sample_x, *params)
    sample_front_height_um = back_height_um * (1.0 + sample_x) / (1.0 - sample_x)
    ax.plot(
        sample_front_height_um,
        sample_y,
        color="black",
        lw=2.0,
        label="Smooth response fit" if len(params) == 5 else "Saturating fit",
    )

    ax.set_xlabel(r"Front plate height $h_{\mathrm{front}}$ ($\mu$m)")
    ax.set_ylabel(r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    return primary, validation, params, covariance, discretization_fraction, chirality_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REFINED_DATA_DIR)
    parser.add_argument("--fallback-dir", type=Path, help="optional explicit fallback; never mix old/new runs implicitly")
    parser.add_argument("--validation-dir", type=Path, help="defaults to the matching refined or legacy resolution checks")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "canting_clean.png")
    parser.add_argument("--csv", type=Path, default=SCRIPT_DIR / "canting_clean.csv")
    parser.add_argument("--fit-json", type=Path, default=SCRIPT_DIR / "canting_clean_fit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    (
        primary,
        validation,
        params,
        covariance,
        discretization_fraction,
        chirality_scale,
    ) = plot_canting(
        ax, args.data_dir, fallback_dir=args.fallback_dir, validation_dir=args.validation_dir
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.fit_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    write_csv(args.csv, primary, validation)
    fit_payload = {
        "model": "baseline + amplitude * (1 - exp(-(height_ratio / scale)**shape))"
                 + (" * exp(-decay * height_ratio)" if len(params) == 5 else ""),
        "fit_role": "descriptive presentation curve; data points are independent simulation estimates",
        "error_bar_source": "postprocessing_half_range",
        "analysis_methods": sorted({item.method for item in primary}),
        "postprocessing_sensitivity": {
            "normal_motion_swept_area": {
                "kernel_widths_mm": [0.25, 0.30, 0.35],
                "speed_floors": [1e-7, 1e-6, 1e-5],
                "saved_sample_strides": [1, 2, 4],
                "time_window": [1 / 18, 17 / 6],
            },
            "trajectory_spline": {"knot_intervals": KNOT_INTERVALS, "kernel_widths_mm": KERNEL_WIDTHS_MM},
        },
        "primary_data_directory": str(args.data_dir.resolve()),
        "calibration": {
            "height_convention_multiplier": HEIGHT_CONVENTION_MULTIPLIER,
            "reference_front_height_um": CALIBRATION_FRONT_HEIGHT_UM,
            "reference_slope_deg_per_mm": CALIBRATION_SLOPE_DEG_PER_MM,
            "slope_multiplier": float(chirality_scale),
            "uncalibrated_k_alpha_mm": K_ALPHA_MM,
            "effective_k_alpha_mm": float(chirality_scale * K_ALPHA_MM),
        },
        "parameters": dict(zip(("baseline", "amplitude", "scale", "shape", "decay"), map(float, params))),
        "covariance": np.asarray(covariance).tolist(),
        "discretization_fraction_from_mesh_check": float(discretization_fraction),
        "discretization_fraction_used_only_for_fit_weights": float(discretization_fraction),
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
        "plateau_deg_per_mm": float(params[0] + params[1]) if len(params) == 4 or params[4] == 0 else None,
        "slope_at_max_height_deg_per_mm": float(saturating_response(primary[-1].height_ratio, *params)),
    }
    args.fit_json.write_text(json.dumps(fit_payload, indent=2) + "\n")
    print(f"Saved {args.output}, {args.csv}, and {args.fit_json} ({len(primary)} primary, {len(validation)} validation points)")
    print(
        f"Calibrated by {chirality_scale:.8g}: "
        f"{CALIBRATION_SLOPE_DEG_PER_MM:g} degrees/mm at "
        f"{CALIBRATION_FRONT_HEIGHT_UM:g} microns front height"
    )
    print("  h_front_um  chirality_slope_deg_per_mm +/- postprocessing_error")
    for item in primary:
        setup = load_pickle(item.source.parent / "setup.pkl")
        front_height_um = HEIGHT_CONVENTION_MULTIPLIER * 1000.0 * float(setup["ht_front"])
        print(f"  {front_height_um:10.3f}  {item.chirality_slope:8.5f} +/- {item.numerical_error:.5f}")


if __name__ == "__main__":
    main()
