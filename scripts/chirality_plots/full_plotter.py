#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt

from riv_og import plot_riv_og
from riv_code import plot_riv_code
from canting_plotter import plot_canting
from drying_rate import plot_drying_rate


def main():
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: a, b, e
    plot_riv_og(axes[0, 0], axes[0, 1], add_colorbar=True)
    plot_canting(axes[0, 2])

    # Bottom row: c, d, f
    plot_riv_code(axes[1, 0], axes[1, 1], add_colorbar=True)
    plot_drying_rate(axes[1, 2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
