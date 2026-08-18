#!/usr/bin/env python3
"""Compare the cleaned canting numerics with the experimental measurements.

This is intentionally a presentation-only plot: it reuses the trajectory
analysis from ``canting_plotter.py`` and does not launch any simulations.  The
fourth nonzero-cant numerical point is omitted, as requested.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canting_plotter import REPO_ROOT, SCRIPT_DIR, analyze_directory, merge_estimates
from scipy.interpolate import CubicSpline, Akima1DInterpolator

# Experimental points digitized for the original comparison figure.
EXPERIMENTAL_HEIGHT_RATIO = np.array(
    [
        0.00046292408994360674,
        0.00046292408994360674,
        0.04907405786553596,
        0.04861105960092645,
        0.06342596562310825,
        0.06388886498816321,
        0.07268513971886222,
        0.12361114119305902,
        0.18518521265728374,
        0.12407404055811398,
        0.09166663452431145,
        0.18796300444583172,
        0.24537033877745715,
        0.26018514590008446,
        0.31666663205182266,
        0.31666663205182266,
        0.3837963365429207,
        0.4296296538050764,
        0.501388846444497,
        0.5569443855167933,
    ]
)

EXPERIMENTAL_G_SLOPE = np.array(
    [
        0.3571428912026467,
        0.3571428912026467,
        0.5194806793644906,
        0.5373377935804468,
        0.4935064980102673,
        0.4496752891377342,
        0.4253246946064571,
        0.41396106070582195,
        0.49675341156322045,
        0.6493506324614973,
        0.8977272515454614,
        0.7532468376925127,
        0.8392856686146327,
        0.8652596765735633,
        1.0194805709924328,
        1.0649351065949728,
        0.9334414799773735,
        0.8474026490552535,
        0.8279220347139976,
        0.7759741054937828,
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "canting_production_20")
    parser.add_argument(
        "--fallback-dir",
        type=Path,
        default=REPO_ROOT / "canting_production_lsoda_20",
    )
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "canting_comper.png")
    return parser.parse_args()


def numerical_points(data_dir: Path, fallback_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load the clean numerical series and omit its fourth nonzero point."""
    primary = merge_estimates(
        analyze_directory(data_dir),
        analyze_directory(fallback_dir) if fallback_dir.is_dir() else [],
    )
    if not primary:
        raise RuntimeError(f"no valid canting results found in {data_dir}")

    x = np.array([item.height_ratio for item in primary])
    y = np.array([item.chirality_slope for item in primary])
    nonzero = np.flatnonzero(~np.isclose(x, 0.0))
    if len(nonzero) < 4:
        raise RuntimeError("fewer than four nonzero-cant numerical points were found")

    omitted_index = [] #np.array([int(nonzero[3]), int(nonzero[5])])
    keep = np.ones(len(primary), dtype=bool)
    keep[omitted_index] = False
    return x[keep], y[keep], omitted_index


def origin_preserving_scale(numerical_x: np.ndarray, numerical_y: np.ndarray) -> float:
    """Fit numerical_y = scale * experimental_y with a fixed zero intercept."""
    overlap = (
        (EXPERIMENTAL_HEIGHT_RATIO >= numerical_x.min())
        & (EXPERIMENTAL_HEIGHT_RATIO <= numerical_x.max())
    )
    experimental_y = EXPERIMENTAL_G_SLOPE[overlap]
    numerical_at_experiment = np.interp(
        EXPERIMENTAL_HEIGHT_RATIO[overlap], numerical_x, numerical_y
    )
    return float(
        np.dot(experimental_y, numerical_at_experiment)
        / np.dot(experimental_y, experimental_y)
    )


def main() -> None:
    args = parse_args()
    numerical_x, numerical_y, omitted_index = numerical_points(
        args.data_dir.resolve(), args.fallback_dir.resolve()
    )
    scale = origin_preserving_scale(numerical_x, numerical_y)

    fig, numerical_ax = plt.subplots(figsize=(8, 6), dpi=300)
    experimental_ax = numerical_ax.twinx()

    # Smooth cubic spline through the simulation points.
    order = np.argsort(numerical_x)
    x_spline = numerical_x[order]
    y_spline = numerical_y[order]

    # spline = CubicSpline(x_spline, y_spline)
    spline = Akima1DInterpolator(x_spline, y_spline)
    x_fine = np.linspace(x_spline.min(), x_spline.max(), 500)

    numerical_ax.plot(
        x_fine,
        spline(x_fine),
        color="tab:blue",
        linewidth=2.0,
        label="Simulation",
    )

    '''
    numerical_ax.plot(
        numerical_x,
        numerical_y,
        color="tab:blue",
        linewidth=2.0,
        label="Simulation",
    ) '''
    numerical_ax.scatter(
        numerical_x,
        numerical_y,
        color="tab:blue",
        s=42,
        zorder=3,
    )
    experimental_ax.scatter(
        EXPERIMENTAL_HEIGHT_RATIO,
        EXPERIMENTAL_G_SLOPE,
        color="m",
        s=52,
        alpha=0.8,
        label="Experiment",
    )

    numerical_ax.set_xlabel(
        r"$(h_{\mathrm{front}}-h_{\mathrm{back}})/(h_{\mathrm{front}}+h_{\mathrm{back}})$"
    )
    numerical_ax.set_ylabel(r"Chirality slope $d\alpha/dx$ ($^\circ$/mm)")
    experimental_ax.set_ylabel("Experimental g-factor slope")
    numerical_ax.grid(True, alpha=0.25)

    # The two limits differ by one constant scale.  Consequently y=0 lands at
    # exactly the same (off-canvas) position on both axes even though the plot
    # does not need to waste vertical space displaying it.
    values_in_numerical_units = np.concatenate(
        (numerical_y, scale * EXPERIMENTAL_G_SLOPE)
    )
    padding = 0.08 * np.ptp(values_in_numerical_units)
    numerical_limits = (
        float(values_in_numerical_units.min() - padding),
        float(values_in_numerical_units.max() + padding),
    )
    numerical_ax.set_ylim(*numerical_limits)
    experimental_ax.set_ylim(*(limit / scale for limit in numerical_limits))

    x_max = max(float(numerical_x.max()), float(EXPERIMENTAL_HEIGHT_RATIO.max()))
    numerical_ax.set_xlim(-0.025, x_max + 0.025)
    numerical_handles, numerical_labels = numerical_ax.get_legend_handles_labels()
    experimental_handles, experimental_labels = experimental_ax.get_legend_handles_labels()
    numerical_ax.legend(
        numerical_handles + experimental_handles,
        numerical_labels + experimental_labels,
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    omitted_nonzero_number = 4
    print(
        f"Omitted numerical point {omitted_nonzero_number} (array index {omitted_index}); "
        f"origin-preserving numerical/experimental scale = {scale:.6g}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
