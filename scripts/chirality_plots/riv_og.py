#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 20:05:40 2025

@author: rivercarson
written with claude ai and slight modifications by hand
"""

# =============================================================================
# A note on the MMP map orientation:  By Default matplotlib sets the (0,0) 
# coordinate to the bottom left. However the beamline maps from the top left.
# This has the effect to horizontally mirror the mapping.  Rather than resetting
# the coordinate in the plot and then rotating 90 degrees, the numpy array has
# just been transposed prior to plotting
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

# border_width = 3
# labeflsize = 24
# ticklabelsize = 20
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
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

def _build_riv_og_data():
    # Option 2: CSV is already a 2D grid (rows=y, columns=x)
    Z = np.loadtxt("Tom_Science_symmetric_linear_transposed.csv", delimiter=",")
    x = np.linspace(0, 4, Z.shape[1])
    y = np.linspace(0, 4, Z.shape[0])

    control_points = np.array(
        [
            [0.0, 3.4],
            [2.0, 3.90],
            [4.0, 2.9],
        ]
    ) - np.array([0, .1])

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
    for i in range(4):
        curves[f"y{i}"] = curve_y - i * 0.5
        scans[f"scan{i}"] = create_scans_from_path(curves[f"y{i}"])

    return Z, x, y, curve_x, curves, scans


def plot_riv_og(ax_heatmap, ax_line, add_colorbar=True):
    Z, x, y, curve_x, curves, scans = _build_riv_og_data()

    border_width = 3
    labeflsize = 28
    colors = ["c", "m", "y", "g"]
    g_range = 0.15
    lwidth = 4

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
        ax_heatmap.plot(4 - curves[curve], curve_x, "--", color=colors[i], linewidth=lwidth)

    ax_heatmap.set_xlabel("X (mm)")
    ax_heatmap.set_ylabel("Y (mm)")
    ax_heatmap.set_title("Measured G-Factors Across Film")

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
        cbar.set_label("G-Factor", fontsize=labeflsize)

    for i, scan in enumerate(scans):
        ax_line.plot(curve_x, scans[scan], color=colors[i], linewidth=lwidth)

    ax_line.set(xlim=[0, 4], ylim=[-g_range, g_range])
    ax_line.set_xticks([0, 1, 2, 3, 4])
    ax_line.grid(alpha=0.25)
    ax_line.set_xlabel("Distance along curve (mm)")
    ax_line.set_ylabel("G-Factor")
    ax_line.set_title("Line Scan Profile")
    for location in ["top", "bottom", "left", "right"]:
        ax_line.spines[location].set_linewidth(border_width)
    ax_line.tick_params(width=border_width, length=6)

    return im


def plot_riv_og_heatmap(ax_heatmap, add_colorbar=True, strip_labels=False):
    Z, x, y, curve_x, curves, _scans = _build_riv_og_data()

    border_width = 3
    labeflsize = 28
    colors = ["c", "m", "y", "g"]
    g_range = 0.15
    lwidth = 4

    im = ax_heatmap.imshow(
        np.flipud(Z.T),
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="seismic",
        aspect="equal",
        vmin=-g_range,
        vmax=g_range,
    )

    for location in ["top", "bottom", "left", "right"]:
        ax_heatmap.spines[location].set_linewidth(border_width)

    for i, curve in enumerate(curves):
        ax_heatmap.plot(4 - curves[curve], curve_x, "--", color=colors[i], linewidth=lwidth)

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
        ax_heatmap.set_title("Measured G-Factors Across Film")

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
            cbar.ax.tick_params(labelleft=False, labelright=False)
        else:
            cbar.set_label("G-Factor", fontsize=labeflsize)

    return im


def plot_riv_og_line(ax_line):
    _Z, _x, _y, curve_x, _curves, scans = _build_riv_og_data()

    border_width = 3
    colors = ["c", "m", "y", "g"]
    g_range = 0.15
    lwidth = 4

    for i, scan in enumerate(scans):
        ax_line.plot(curve_x, scans[scan], color=colors[i], linewidth=lwidth)

    ax_line.set(xlim=[0, 4], ylim=[-g_range, g_range])
    ax_line.set_xticks([0, 1, 2, 3, 4])
    ax_line.grid(alpha=0.25)
    ax_line.set_xlabel("Distance along curve (mm)")
    ax_line.set_ylabel("G-Factor")
    ax_line.set_title("Line Scan Profile")
    for location in ["top", "bottom", "left", "right"]:
        ax_line.spines[location].set_linewidth(border_width)
    ax_line.tick_params(width=border_width, length=6)


def main():
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 8))
    plot_riv_og_heatmap(ax_heatmap, add_colorbar=True, strip_labels=True)
    # fig_heatmap.tight_layout()
    fig_heatmap.savefig("riv_og_colormap.png", dpi=300)

    fig_line, ax_line = plt.subplots(figsize=(6.75, 6))
    plot_riv_og_line(ax_line)
    fig_line.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

#%%
# # Optional: Use parametric equations for more complex curvesi
# # Example: Sinusoidal curve
# def create_sinusoidal_path(amplitude=2, frequency=2, x_range=(1, 9), n_points=500):
#     x_curve = np.linspace(x_range[0], x_range[1], n_points)
#     y_curve = 5 + amplitude * np.sin(frequency * np.pi * (x_curve - x_range[0]) / (x_range[1] - x_range[0]))
#     return x_curve, y_curve

# # Uncomment to use sinusoidal path instead:
# # curve_x, curve_y = create_sinusoidal_path()
