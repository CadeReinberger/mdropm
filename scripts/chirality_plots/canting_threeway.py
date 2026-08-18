#!/usr/bin/env python3
"""Compare trajectory, large-CASR, and gas-log-flux canting estimates.

The trajectory estimates are imported from :mod:`canting_plotter` unchanged.
Each other estimate uses one gas FEM solve on the corresponding initial circle;
no temporal droplet solve is performed here.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Package import under pytest, and direct import when run as a script.
    from .canting_plotter import (
        K_ALPHA_MM,
        Estimate,
        analyze_directory,
        load_pickle,
        merge_estimates,
        normalized_height,
    )
except ImportError:  # pragma: no cover - exercised by the standalone command
    from canting_plotter import (  # type: ignore[no-redef]
        K_ALPHA_MM,
        Estimate,
        analyze_directory,
        load_pickle,
        merge_estimates,
        normalized_height,
    )
from geo_util import Lambda, Lambda_pr  # noqa: E402


FLUX_FIT_DEGREE = 4
FLUX_FIT_HALF_WIDTH_NODES = 5
DEFAULT_GAS_NODES = 80
DENOMINATOR_RTOL = 1e-8
FIT_SPREAD_WARNING_FRACTION = 0.25


@dataclass(frozen=True)
class ComparisonCase:
    estimate: Estimate
    setup: dict
    setup_path: Path


@dataclass(frozen=True)
class InitialProblem:
    x: np.ndarray
    y: np.ndarray
    normals: np.ndarray
    radius: float
    heightscape: object
    universe: object


@dataclass(frozen=True)
class FluxDerivatives:
    J0: float
    Js0: float
    Jss0: float
    sample_count: int
    half_width_nodes: int


@dataclass(frozen=True)
class FormulaGeometry:
    b: float
    h0: float
    D0: float
    B0: float
    a1: float
    c1: float
    c2: float
    Dss: float


@dataclass(frozen=True)
class ComparisonResult:
    case: ComparisonCase
    flux: FluxDerivatives
    geometry: FormulaGeometry
    g_full: float
    g_flux: float
    full_fit_spread: float
    flux_fit_spread: float


def _setup_path_for_estimate(
    estimate: Estimate, data_dir: Path, fallback_dir: Path
) -> Path:
    """Locate the setup belonging to a merged trajectory estimate."""
    beside_result = estimate.source.parent / "setup.pkl"
    candidates = [
        beside_result,
        data_dir / estimate.run_id / "setup.pkl",
        fallback_dir / estimate.run_id / "setup.pkl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        setup = load_pickle(candidate)
        if np.isclose(normalized_height(setup), estimate.height_ratio, atol=1e-12):
            return candidate
    raise FileNotFoundError(
        f"could not locate setup.pkl for {estimate.run_id} "
        f"(height ratio {estimate.height_ratio:g})"
    )


def load_comparison_cases(
    data_dir: Path, fallback_dir: Path
) -> list[ComparisonCase]:
    """Return merged setups and the unmodified current trajectory estimates."""
    data_dir = data_dir.resolve()
    fallback_dir = fallback_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"primary data directory not found: {data_dir}")

    estimates = merge_estimates(
        analyze_directory(data_dir),
        analyze_directory(fallback_dir) if fallback_dir.is_dir() else [],
    )
    if not estimates:
        raise RuntimeError(f"no valid canting results found in {data_dir}")

    cases = []
    for estimate in estimates:
        setup_path = _setup_path_for_estimate(estimate, data_dir, fallback_dir)
        cases.append(ComparisonCase(estimate, load_pickle(setup_path), setup_path))
    return cases


def make_initial_problem(setup: dict, gas_nodes: int) -> InitialProblem:
    """Rebuild the production plate/tape configuration and an exact circle."""
    if gas_nodes < 12 or gas_nodes % 4:
        raise ValueError("--gas-nodes must be a multiple of 4 and at least 12")

    # build_problem is the authoritative constructor for these production runs:
    # 10 x 10 mm plate, top opening, y-oriented canting, and production gas FEM
    # settings.  Replace its landscape with the exact saved front/back heights
    # so this remains correct if a sweep changes the baseline height.
    from canting_runner import build_problem
    import heightscape
    from physical_params import physical_params

    front = float(setup["ht_front"])
    back = float(setup.get("ht_back", 0.1))
    if back <= 0:
        raise ValueError(f"ht_back must be positive, got {back}")
    raw_ratio = front / back - 1.0
    _, _, _, universe = build_problem(raw_ratio, gas_nodes, "production")
    hs = heightscape.constructors.rect_interp_htscape(
        (10, 10), (back, front, front, back)
    )
    universe.htscp = hs
    # The formulas use the repository defaults, not any fitted or rescaled
    # physical constants.  CASR smoothing from the time solver is irrelevant.
    universe.phys_ps = physical_params()

    radius = float(setup.get("radius", 3.0))
    angles = np.linspace(0.0, 2.0 * np.pi, gas_nodes, endpoint=False)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    return InitialProblem(
        x=radius * normals[:, 0],
        y=radius * normals[:, 1],
        normals=normals,
        radius=radius,
        heightscape=hs,
        universe=universe,
    )


def _pole_direction(heightscape: object) -> tuple[np.ndarray, float]:
    gradient = np.array(
        [heightscape.hx(0.0, 0.0), heightscape.hy(0.0, 0.0)], dtype=float
    )
    b = float(np.linalg.norm(gradient))
    if b <= np.finfo(float).eps:
        # +y is the continuous b -> 0 pole for the production canting sweep.
        return np.array([0.0, 1.0]), 0.0
    return gradient / b, b


def gas_flux_on_initial_circle(
    problem: InitialProblem,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform one gas solve and return pole-centered arclength and J=-dw/dn."""
    from gas_fem import compute_concentration_gradients

    gradients = np.asarray(
        compute_concentration_gradients(problem.x, problem.y, problem.universe),
        dtype=float,
    ).reshape(len(problem.x), -1)
    if gradients.shape[1] < 2 or not np.all(np.isfinite(gradients[:, :2])):
        raise FloatingPointError("gas boundary gradient contains non-finite values")

    J = -np.einsum("ij,ij->i", gradients[:, :2], problem.normals)
    pole, _ = _pole_direction(problem.heightscape)
    # atan2(cross(pole, n), dot(pole, n)) gives a signed, wrap-safe CCW
    # angle even when no interface node lies exactly at the pole.
    cross = pole[0] * problem.normals[:, 1] - pole[1] * problem.normals[:, 0]
    dot = problem.normals @ pole
    s = problem.radius * np.arctan2(cross, dot)
    order = np.argsort(s)
    return s[order], J[order]


def fit_flux_derivatives(
    s: np.ndarray,
    J: np.ndarray,
    *,
    degree: int = FLUX_FIT_DEGREE,
    half_width_nodes: int = FLUX_FIT_HALF_WIDTH_NODES,
) -> FluxDerivatives:
    """Fit a local polynomial and analytically evaluate J, J_s, and J_ss."""
    s = np.asarray(s, dtype=float)
    J = np.asarray(J, dtype=float)
    if s.ndim != 1 or J.shape != s.shape or len(s) < 2 * half_width_nodes + 1:
        raise ValueError("flux samples must be matching one-dimensional arrays")
    if degree < 2:
        raise ValueError("flux fit degree must be at least 2")
    if half_width_nodes < 2:
        raise ValueError("flux fit half-width must be at least 2 nodes")
    if not np.all(np.isfinite(s)) or not np.all(np.isfinite(J)):
        raise FloatingPointError("non-finite gas flux samples")

    # The samples cover one uniform circle.  A median is insensitive to which
    # side of the atan2 branch cut contains the omitted periodic interval.
    spacing = float(np.median(np.diff(np.sort(s))))
    if spacing <= 0:
        raise ValueError("arclength samples must contain distinct circle nodes")
    selected = np.abs(s) <= (half_width_nodes + 0.5) * spacing
    local_s, local_J = s[selected], J[selected]
    if len(local_s) <= degree:
        raise ValueError(
            f"only {len(local_s)} local samples for degree-{degree} flux fit"
        )

    scale = float(np.max(np.abs(local_s)))
    if scale <= 0:
        raise ValueError("degenerate arclength neighborhood")
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        coefficients = np.polynomial.polynomial.polyfit(
            local_s / scale, local_J, degree
        )
    J0 = float(coefficients[0])
    Js0 = float(coefficients[1] / scale)
    Jss0 = float(2.0 * coefficients[2] / scale**2)
    return FluxDerivatives(J0, Js0, Jss0, len(local_s), half_width_nodes)


def _formula_geometry(problem: InitialProblem) -> FormulaGeometry:
    pole, b = _pole_direction(problem.heightscape)
    x0 = problem.radius * pole
    h0 = float(problem.heightscape.h(x0[0], x0[1]))
    theta_r = float(problem.universe.phys_ps.theta_r)
    chi0 = theta_r + np.arctan(b)
    lambda0 = float(Lambda(chi0))
    lambda_pr0 = float(Lambda_pr(chi0))
    D0 = 1.0 - b * lambda0
    if np.isclose(h0 * lambda_pr0, 0.0):
        raise ZeroDivisionError("h0 * Lambda'(chi0) is too close to zero")
    B0 = 2.0 * D0 / (h0 * lambda_pr0)
    a1 = -b / (problem.radius * (1.0 + b**2))
    denom1, denom2 = B0 + a1, B0 + 2.0 * a1
    scale = max(1.0, abs(B0), abs(a1))
    if abs(denom1) <= DENOMINATOR_RTOL * scale:
        warnings.warn(f"B0 + a1 is close to zero ({denom1:.6g})", stacklevel=2)
    if abs(denom2) <= DENOMINATOR_RTOL * scale:
        warnings.warn(f"B0 + 2*a1 is close to zero ({denom2:.6g})", stacklevel=2)
    if abs(D0) <= DENOMINATOR_RTOL:
        warnings.warn(f"D0 is close to zero ({D0:.6g})", stacklevel=2)
    c1 = B0 / denom1
    c2 = B0 / denom2
    Dss = (
        b * lambda0 + b**2 / (1.0 + b**2) * lambda_pr0
    ) / problem.radius**2
    return FormulaGeometry(b, h0, D0, B0, a1, c1, c2, Dss)


def large_casr_g(
    geometry: FormulaGeometry, J0: float, Js0: float, Jss0: float
) -> float:
    """Evaluate the full large-CASR pole formula from the design document."""
    if np.isclose(J0, 0.0):
        raise ZeroDivisionError("fitted J0 is too close to zero")
    return float(
        geometry.c2 * (Jss0 / J0 - geometry.Dss / geometry.D0)
        - geometry.c1**2 * (Js0 / J0) ** 2
    )


def gas_log_flux_g(J0: float, Js0: float, Jss0: float) -> float:
    """Evaluate d^2(log J)/ds^2 at the high-gap pole."""
    if np.isclose(J0, 0.0):
        raise ZeroDivisionError("fitted J0 is too close to zero")
    return float(Jss0 / J0 - (Js0 / J0) ** 2)


def convert_g_to_plot_units(g: float | np.ndarray) -> float | np.ndarray:
    """Convert geometric 1/mm^2 to the cleaned plot's degrees/mm convention."""
    # canting_plotter defines alpha = K_ALPHA_MM * trajectory curvature (radians)
    # and reports d(alpha)/dx in degrees/mm.  Applying the same legacy physical
    # factor to d(curvature)/ds gives the comparable plotted convention.
    return np.rad2deg(K_ALPHA_MM * np.asarray(g))


def _fit_widths(center: int, sample_count: int, degree: int) -> tuple[int, ...]:
    candidates = range(max(2, center - 1), center + 2)
    return tuple(
        width
        for width in candidates
        if 2 * width + 1 <= sample_count and 2 * width + 1 > degree
    )


def analyze_gas_case(
    case: ComparisonCase,
    *,
    gas_nodes: int,
    fit_degree: int,
    fit_half_width: int,
) -> ComparisonResult:
    """Use one gas solve and one family of local fits for both gas formulas."""
    problem = make_initial_problem(case.setup, gas_nodes)
    s, J = gas_flux_on_initial_circle(problem)  # the only gas solve
    geometry = _formula_geometry(problem)
    widths = _fit_widths(fit_half_width, len(s), fit_degree)
    if fit_half_width not in widths:
        raise ValueError("gas-node count is too small for the requested local fit")

    fits = {
        width: fit_flux_derivatives(
            s, J, degree=fit_degree, half_width_nodes=width
        )
        for width in widths
    }
    full_values = np.array(
        [large_casr_g(geometry, f.J0, f.Js0, f.Jss0) for f in fits.values()]
    )
    flux_values = np.array(
        [gas_log_flux_g(f.J0, f.Js0, f.Jss0) for f in fits.values()]
    )
    central = fits[fit_half_width]
    if central.J0 <= 0:
        warnings.warn(
            f"{case.estimate.run_id}: fitted J0 is non-positive ({central.J0:.6g})",
            stacklevel=2,
        )
    return ComparisonResult(
        case=case,
        flux=central,
        geometry=geometry,
        g_full=large_casr_g(geometry, central.J0, central.Js0, central.Jss0),
        g_flux=gas_log_flux_g(central.J0, central.Js0, central.Jss0),
        full_fit_spread=float(np.ptp(full_values)),
        flux_fit_spread=float(np.ptp(flux_values)),
    )


def trajectory_errors(
    cases: list[ComparisonCase], validation_dir: Path
) -> tuple[np.ndarray, float]:
    """Reproduce canting_plotter's post-processing plus mesh-check error bars."""
    validation = analyze_directory(validation_dir) if validation_dir.is_dir() else []
    relative_mesh_differences = []
    for check in validation:
        matches = [
            case.estimate
            for case in cases
            if np.isclose(case.estimate.height_ratio, check.height_ratio)
        ]
        if matches and matches[0].chirality_slope != 0:
            relative_mesh_differences.append(
                abs(check.chirality_slope - matches[0].chirality_slope)
                / abs(matches[0].chirality_slope)
            )
    discretization_fraction = max(relative_mesh_differences, default=0.0)
    errors = np.array(
        [
            np.hypot(
                case.estimate.numerical_error,
                discretization_fraction * case.estimate.chirality_slope,
            )
            for case in cases
        ]
    )
    return errors, discretization_fraction


def write_csv(path: Path, results: list[ComparisonResult], errors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "height_ratio",
                "h_front_mm",
                "h_back_mm",
                "b",
                "J0",
                "Js0",
                "Jss0",
                "g_full_per_mm2",
                "g_flux_per_mm2",
                "g_full_plot_deg_per_mm",
                "g_flux_plot_deg_per_mm",
                "g_trajectory_deg_per_mm",
                "g_trajectory_error_deg_per_mm",
                "full_fit_width_spread_per_mm2",
                "flux_fit_width_spread_per_mm2",
                "run_id",
            ]
        )
        for result, error in zip(results, errors):
            case = result.case
            writer.writerow(
                [
                    case.estimate.height_ratio,
                    case.setup["ht_front"],
                    case.setup.get("ht_back", 0.1),
                    result.geometry.b,
                    result.flux.J0,
                    result.flux.Js0,
                    result.flux.Jss0,
                    result.g_full,
                    result.g_flux,
                    float(convert_g_to_plot_units(result.g_full)),
                    float(convert_g_to_plot_units(result.g_flux)),
                    case.estimate.chirality_slope,
                    error,
                    result.full_fit_spread,
                    result.flux_fit_spread,
                    case.estimate.run_id,
                ]
            )


def plot_comparison(
    results: list[ComparisonResult], errors: np.ndarray, output: Path
) -> None:
    x = np.array([result.case.estimate.height_ratio for result in results])
    trajectory = np.array(
        [result.case.estimate.chirality_slope for result in results]
    )
    full = convert_g_to_plot_units(np.array([result.g_full for result in results]))
    flux = convert_g_to_plot_units(np.array([result.g_flux for result in results]))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.errorbar(
        x,
        trajectory,
        yerr=errors,
        fmt="o",
        color="tab:blue",
        capsize=3,
        ms=6,
        label="Trajectory (current)",
    )
    ax.plot(x, full, "s-", color="tab:orange", ms=5, lw=1.7, label="Large-CASR")
    ax.plot(x, flux, "^-", color="tab:green", ms=5, lw=1.7, label="Gas log-flux")
    ax.set_xlabel(
        r"$(h_{\mathrm{front}}-h_{\mathrm{back}})/(h_{\mathrm{front}}+h_{\mathrm{back}})$"
    )
    ax.set_ylabel(r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _relative_spread(spread: float, central: float) -> float:
    return spread / max(abs(central), 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "canting_production_20")
    parser.add_argument(
        "--fallback-dir", type=Path, default=REPO_ROOT / "canting_production_lsoda_20"
    )
    parser.add_argument(
        "--validation-dir", type=Path, default=REPO_ROOT / "canting_mesh_reference_20"
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "canting_threeway.png")
    parser.add_argument("--csv", type=Path, default=SCRIPT_DIR / "canting_threeway.csv")
    parser.add_argument("--flux-fit-degree", type=int, default=FLUX_FIT_DEGREE)
    parser.add_argument(
        "--flux-fit-half-width", type=int, default=FLUX_FIT_HALF_WIDTH_NODES
    )
    parser.add_argument("--gas-nodes", type=int, default=DEFAULT_GAS_NODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_comparison_cases(args.data_dir, args.fallback_dir)
    errors, discretization_fraction = trajectory_errors(cases, args.validation_dir)

    print(
        "height_ratio h_front h_back b J0 Js0 Jss0 "
        "g_full g_flux g_trajectory"
    )
    results = []
    for case in cases:
        result = analyze_gas_case(
            case,
            gas_nodes=args.gas_nodes,
            fit_degree=args.flux_fit_degree,
            fit_half_width=args.flux_fit_half_width,
        )
        results.append(result)
        print(
            f"{case.estimate.height_ratio:.8g} "
            f"{float(case.setup['ht_front']):.8g} "
            f"{float(case.setup.get('ht_back', 0.1)):.8g} "
            f"{result.geometry.b:.8g} "
            f"{result.flux.J0:.8g} {result.flux.Js0:.8g} "
            f"{result.flux.Jss0:.8g} {result.g_full:.8g} "
            f"{result.g_flux:.8g} {case.estimate.chirality_slope:.8g}"
        )
        full_relative = _relative_spread(result.full_fit_spread, result.g_full)
        flux_relative = _relative_spread(result.flux_fit_spread, result.g_flux)
        print(
            f"  fit-width spread: full={result.full_fit_spread:.3g} "
            f"({full_relative:.1%}), flux={result.flux_fit_spread:.3g} "
            f"({flux_relative:.1%})"
        )
        if max(full_relative, flux_relative) > FIT_SPREAD_WARNING_FRACTION:
            warnings.warn(
                f"{case.estimate.run_id}: local-fit-width sensitivity exceeds "
                f"{FIT_SPREAD_WARNING_FRACTION:.0%}",
                stacklevel=1,
            )

    plot_comparison(results, errors, args.output)
    write_csv(args.csv, results, errors)
    print(
        f"Saved {args.output} and {args.csv} ({len(results)} cases; "
        f"trajectory mesh-error fraction {discretization_fraction:.3%})"
    )


if __name__ == "__main__":
    main()
