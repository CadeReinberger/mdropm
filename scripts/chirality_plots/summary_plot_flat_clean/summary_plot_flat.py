"""Plot five evenly spaced snapshots from the adjacent results.pkl.

Uses linear time interpolation and periodic cubic spatial interpolation,
with the same five-panel appearance as the flat-run summary.
Run: python summary_plot_flat.py
"""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def make_summary():
    with Path(__file__).resolve().with_name("results.pkl").open("rb") as stream:
        results = pickle.load(stream)
    times = np.asarray(results["out_t"])
    states = np.asarray(results["out_x"])
    nodes = states.shape[1] // 3
    sample_times = np.linspace(times[0], times[-1], 5)
    # Linear interpolation in time, as in sum_plt.py.
    outlines = np.column_stack([
        np.interp(sample_times, times, column)
        for column in states[:, :2*nodes].T
    ]).reshape(5, 2, nodes)

    # The original summary uses periodic cubic interpolation in space.
    parameter = np.linspace(0, 1, outlines.shape[2] + 1)
    evaluate = np.linspace(0, 1, 101)
    curves = []
    for outline in outlines:
        closed = np.column_stack((outline, outline[:, 0]))
        curves.append(CubicSpline(parameter, closed, axis=1, bc_type="periodic")(evaluate))
    curves = np.asarray(curves)

    # Bounds are in the displayed coordinates (-y, x), shared by all panels.
    display_x, display_y = -curves[:, 1], curves[:, 0]
    xlim = (1.1 * display_x.min(), 1.1 * display_x.max())
    ylim = (1.1 * display_y.min(), 1.1 * display_y.max())
    radius = np.mean(np.hypot(outlines[0, 0], outlines[0, 1]))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for i, (ax, x, y) in enumerate(zip(axes, display_x, display_y)):
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(x, y, "k", linewidth=2)
        if i > 0:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, linestyle="--",
                                    edgecolor="gray", linewidth=1))
    fig.tight_layout(w_pad=0.3)
    return fig


if __name__ == "__main__":
    figure = make_summary()
    output = Path(__file__).resolve().with_name("summary_plot_flat.png")
    figure.savefig(output, dpi=300)
    plt.close(figure)
    print(f"Saved {output}")
