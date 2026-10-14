# =============================================================================
# A note on the MMP map orientation:  By Default matplotlib sets the (0,0) 
# coordinate to the bottom left. However the beamline maps from the top left.
# This has the effect to horizontally mirror the mapping.  Rather than resetting
# the coordinate in the plot and then rotating 90 degrees, the numpy array has
# just been transposed prior to plotting
# =============================================================================

import argparse
from pathlib import Path
import sys
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
import pickle as pkl

# border_width = 3
# labeflsize = 24
# ticklabelsize = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.pkl"

# Load data from CSV file
# Option 1: CSV with columns [x, y, value]
# Uncomment and modify the filename:
# import pandas as pd
# df = pd.read_csv('your_data.csv')
# x_data = df['x'].values  # or use column index df.iloc[:, 0]
# y_data = df['y'].values
# z_data = df['value'].values
# 
# # Create regular grid (if data is not already gridded)
# from scipy.interpolate import griddata
# x = np.linspace(x_data.min(), x_data.max(), 100)
# y = np.linspace(y_data.min(), y_data.max(), 100)
# X, Y = np.meshgrid(x, y)
# Z = griddata((x_data, y_data), z_data, (X, Y), method='cubic')

def _build_riv_code_data():
    Z = None
    with DATA_PATH.open("rb") as file:
        Z = np.asarray(pkl.load(file), dtype=float)
        if Z.ndim != 2 or min(Z.shape) < 2 or not np.isfinite(Z).any():
            raise ValueError("data.pkl must contain a 2-D chirality array")
        Z = np.flipud(Z)

    x = np.linspace(0, 4, Z.shape[1])
    y = np.linspace(0, 4, Z.shape[0])

    control_points = (
        np.array(
            [
                [0.0, 2.9],
                [2.0, 3.5],
                [4.0, 2.9],
            ]
        )
        - np.array([0, 0.4])
    )

    t = np.linspace(0, 1, len(control_points))
    t_fine = np.linspace(0, 1, 500)

    fx = interp1d(t, control_points[:, 0], kind="quadratic")
    fy = interp1d(t, control_points[:, 1], kind="quadratic")

    curve_x = fx(t_fine)
    curve_y = fy(t_fine)

    def create_scans_from_path(ycurve):
        x_idx = np.interp(curve_x, x, np.arange(len(x)))
        y_idx = np.interp(ycurve, y, np.arange(len(y)))
        coords = np.array([y_idx, x_idx])
        line_scan = map_coordinates(Z, coords, order=1)
        return line_scan

    curves = {}
    scans = {}
    for i in range(3):
        curves[f"y{i}"] = curve_y - i * 1
        scans[f"scan{i}"] = create_scans_from_path(curves[f"y{i}"])

    return Z, x, y, curve_x, curves, scans


def plot_riv_code(ax_heatmap, ax_line, add_colorbar=True):
    Z, x, y, curve_x, curves, scans = _build_riv_code_data()

    border_width = 3
    labeflsize = 28
    colors = ["c", "m", "y", "g"]
    g_range = 2.5
    lwidth = 4

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    im = ax_heatmap.imshow(
        Z.T,
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="seismic",
        aspect="equal",
        vmin=-g_range,
        vmax=g_range,
    )
    for location in ["top", "bottom", "left", "right"]:
        ax_heatmap.spines[location].set_linewidth(border_width)

    for i, curve in enumerate(curves):
        ax_heatmap.plot(4 - curves[curve], curve_x, color=colors[i], linewidth=lwidth)

    ax_heatmap.plot(
        4 - (np.sqrt(14 - 0.9 * (curve_x - 2) ** 2) - 0.09),
        curve_x,
        color="black",
        linewidth=6,
    )
    ax_heatmap.set_xlabel("X (mm)")
    ax_heatmap.set_ylabel("Y (mm)")
    ax_heatmap.set_title("Predicted Skewness Angle Alpha")

    if add_colorbar:
        cbar = ax_heatmap.figure.colorbar(im, ax=ax_heatmap, pad=0.02)
        cbar.outline.set_linewidth(border_width)
        cbar.ax.tick_params(width=border_width, length=6)
        cbar.set_label("Alpha (Degrees)", fontsize=labeflsize)

    for i, scan in enumerate(scans):
        ax_line.plot(curve_x, scans[scan], color=colors[i], linewidth=lwidth)

    ax_line.set(xlim=[0.5, 3.5], ylim=[-g_range, g_range])
    ax_line.set_xticks([1, 2, 3])
    ax_line.set_xlabel("Distance along curve (mm)")
    ax_line.set_ylabel("Skewness Angle (Degrees)")
    ax_line.set_title("Line Scan Profile")
    ax_line.grid(True, alpha=0.3)
    for location in ["top", "bottom", "left", "right"]:
        ax_line.spines[location].set_linewidth(border_width)
    ax_line.tick_params(width=border_width, length=6)

    return im


def plot_riv_code_heatmap(ax_heatmap, add_colorbar=True, strip_labels=False, color_limit=0.15):
    Z, x, y, curve_x, curves, _scans = _build_riv_code_data()

    border_width = 3
    labeflsize = 28
    colors = ["c", "m", "y", "g"]
    g_range = color_limit
    lwidth = 4

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )

    im = ax_heatmap.imshow(
        -Z.T,
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="seismic",
        aspect="equal",
        vmin=-g_range,
        vmax=g_range,
    )
    for location in ["top", "bottom", "left", "right"]:
        ax_heatmap.spines[location].set_linewidth(border_width)

    for i, curve in enumerate(curves):
        ax_heatmap.plot(4 - curves[curve], curve_x, color=colors[i], linewidth=lwidth)

    ax_heatmap.plot(
        4 - (np.sqrt(14 - 0.9 * (curve_x - 2) ** 2) - 0.09),
        curve_x,
        color="black",
        linewidth=6,
    )

    if strip_labels:
        ax_heatmap.set_xlabel("")
        ax_heatmap.set_ylabel("")
        ax_heatmap.set_title("")
        ax_heatmap.tick_params(
            labelbottom=False,
            labelleft=False,
            labelright=False,
            labeltop=False,
        )
    else:
        ax_heatmap.set_xlabel("X (mm)")
        ax_heatmap.set_ylabel("Y (mm)")
        ax_heatmap.set_title("Predicted Skewness Angle Alpha")

    if add_colorbar:
        cax = inset_axes(
            ax_heatmap,
            width="5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1, 1),
            bbox_transform=ax_heatmap.transAxes,
            borderpad=0,
        )
        cbar = ax_heatmap.figure.colorbar(im, cax=cax)
        cbar.outline.set_linewidth(border_width)
        cbar.ax.tick_params(width=border_width, length=6)
        if strip_labels:
            cbar.set_label("")
            cbar.ax.tick_params(
                labelleft=False,
                labelright=False,
                labelbottom=False,
                labeltop=False,
            )
        else:
            cbar.set_label("Alpha (Degrees)", fontsize=labeflsize)

    return im


def plot_riv_code_line(ax_line):
    _Z, _x, _y, curve_x, _curves, scans = _build_riv_code_data()

    border_width = 3
    colors = ["c", "m", "y", "g"]
    g_range = 2.5
    lwidth = 4

    for i, scan in enumerate(scans):
        ax_line.plot(curve_x, scans[scan], color=colors[i], linewidth=lwidth)

    ax_line.set(xlim=[0.5, 3.5], ylim=[-g_range, g_range])
    ax_line.set_xticks([1, 2, 3])
    ax_line.set_xlabel("Distance along curve (mm)")
    ax_line.set_ylabel("Skewness Angle (Degrees)")
    ax_line.set_title("Line Scan Profile")
    ax_line.grid(True, alpha=0.3)
    for location in ["top", "bottom", "left", "right"]:
        ax_line.spines[location].set_linewidth(border_width)
    ax_line.tick_params(width=border_width, length=6)


def main():
    global DATA_PATH
    parser = argparse.ArgumentParser(description="Render the saved first-principles map and line scans.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--color-limit", type=float, default=0.15, help="legacy heatmap color limit in degrees")
    parser.add_argument("--show", action="store_true", help="also open interactive figures")
    args = parser.parse_args()
    if not np.isfinite(args.color_limit) or args.color_limit <= 0:
        parser.error("color limit must be finite and positive")
    DATA_PATH = args.data
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 8))
    plot_riv_code_heatmap(ax_heatmap, add_colorbar=True, strip_labels=True, color_limit=args.color_limit)
    fig_heatmap.savefig(args.output_dir / "riv_code_colormap.png", dpi=300, bbox_inches="tight")

    fig_line, ax_line = plt.subplots(figsize=(6.75, 6))
    plot_riv_code_line(ax_line)
    fig_line.tight_layout()
    fig_line.savefig(args.output_dir / "riv_code_linescans.png", dpi=300)
    Z, _x, _y, curve_x, curves, scans = _build_riv_code_data()
    np.savetxt(args.output_dir / "riv_code_linescans.csv",
               np.column_stack([curve_x, *curves.values(), *scans.values()]), delimiter=",",
               header="distance_legacy_mm,y0_legacy_mm,y1_legacy_mm,y2_legacy_mm,scan0_degrees,scan1_degrees,scan2_degrees", comments="")
    # A labeled version exposes the actual model amplitude without clipping
    # it to the original experimental-style +/-0.15 color scale.
    fig_degrees, ax_degrees = plt.subplots(figsize=(8, 8))
    plot_riv_code_heatmap(ax_degrees, color_limit=float(np.nanmax(np.abs(Z))) or 1.0)
    fig_degrees.savefig(args.output_dir / "riv_code_degrees.png", dpi=300, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close("all")
    print(f"Saved RIV prediction figures and scans to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

#%%
# # Optional: Use parametric equations for more complex curves
# # Example: Sinusoidal curve
# def create_sinusoidal_path(amplitude=2, frequency=2, x_range=(1, 9), n_points=500):
#     x_curve = np.linspace(x_range[0], x_range[1], n_points)
#     y_curve = 5 + amplitude * np.sin(frequency * np.pi * (x_curve - x_range[0]) / (x_range[1] - x_range[0]))
#     return x_curve, y_curve

# # Uncomment to use sinusoidal path instead:
# # curve_x, curve_y = create_sinusoidal_path()
