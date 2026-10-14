#!/usr/bin/env python3
"""Animate a saved contact line, colored by contact angle, as GIF and/or MP4."""
import argparse

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import CubicSpline

from animation_utils import (
    add_animation_arguments, animation_writers, interpolate_states,
    load_results, validate_animation_arguments,
)


def space_interp_to_plottable(xs, ys, thetas, num=200):
    """Retain the original periodic cubic interpolation of the contact line."""
    parameter = np.linspace(0, 1, len(xs) + 1)
    evaluation = np.linspace(0, 1, num + 1)
    return tuple(CubicSpline(parameter, np.r_[values, values[0]], bc_type="periodic")(evaluation)
                 for values in (xs, ys, thetas))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_animation_arguments(parser, "full_animation")
    parser.add_argument("--points", type=int, default=200, help="spatial interpolation points per frame")
    args = parser.parse_args()
    validate_animation_arguments(parser, args)
    if args.points < 3:
        parser.error("points must be at least three")
    times, states = load_results(args.results)
    frame_times = np.linspace(times[0], times[-1], args.frames)
    frame_states = interpolate_states(times, states, frame_times)
    n = states.shape[1] // 3
    angles = np.rad2deg(states[:, 2*n:])
    lower, upper = float(angles.min()), float(angles.max())
    if upper - lower < 1e-8:
        lower, upper = lower - 0.5, upper + 0.5

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=args.dpi)
    line = LineCollection([], cmap="gist_rainbow", norm=Normalize(lower, upper), linewidth=2)
    ax.add_collection(line)
    for values, setter in ((states[:, :n], ax.set_xlim), (states[:, n:2*n], ax.set_ylim)):
        padding = 0.08 * max(float(np.ptp(values)), 1.0)
        setter(float(values.min()) - padding, float(values.max()) + padding)
    ax.set(aspect="equal", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(line, ax=ax, label="Contact angle (degrees)")
    title = ax.set_title("")
    fig.tight_layout()
    with animation_writers(fig, args) as writers:
        for index, (time, state) in enumerate(zip(frame_times, frame_states)):
            x, y, theta = space_interp_to_plottable(*np.split(state, 3), num=args.points)
            points = np.c_[x, y]
            line.set_segments(np.stack((points[:-1], points[1:]), axis=1))
            line.set_array(np.rad2deg((theta[:-1] + theta[1:]) / 2))
            title.set_text(f"Contact line, t = {time:.3f} min")
            for writer in writers:
                writer.grab_frame()
            if index == 0 or (index + 1) % 20 == 0 or index == args.frames - 1:
                print(f"Rendered frame {index + 1}/{args.frames}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
