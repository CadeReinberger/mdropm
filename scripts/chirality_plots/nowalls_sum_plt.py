import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import sys
from bisect import bisect
import imageio
from ffmpy import FFmpeg
from tqdm import tqdm
from time import sleep as waitasecond

with open('nowalls.pkl', 'rb') as in_file:
    res_dict = pickle.load(in_file)

UE = 1
end_prop = .7
end_ind = int(end_prop * len(res_dict['t']))

all_ts = res_dict['t'][2:end_ind:UE]
all_xs = res_dict['x'][2:end_ind:UE]
all_ys = res_dict['y'][2:end_ind:UE]
all_thetas = res_dict['theta'][2:end_ind:UE]

x0, y0 = np.array(res_dict['x'][0]), np.array(res_dict['y'][0])
initial_radius = np.mean(np.sqrt(x0**2 + y0**2))

def get_bounds(xs, ys, thetas, extra_factor = .1):
    # Run the main guy to do main guy things
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    min_theta, max_theta = np.inf, -np.inf
    
    for ind in range(len(xs)):
        for (cur_x, cur_y, cur_theta) in zip(xs[ind], ys[ind], thetas[ind]):
            min_x, max_x = min(min_x, cur_x), max(max_x, cur_x)
            min_y, max_y = min(min_y, cur_y), max(max_y, cur_y)
            min_theta, max_theta = min(min_theta, cur_theta), max(max_theta, cur_theta)
    min_x, max_x = (1 + extra_factor) * min_x, (1 + extra_factor) * max_x
    min_y, max_y = (1 + extra_factor) * min_y, (1 + extra_factor) * max_y
    # min_theta, max_theta = (1 + extra_factor) * min_theta, (1 + extra_factor) * max_theta
    
    return ((min_x, max_x), (min_y, max_y), (min_theta, max_theta))

def get_time_interped_points(ts, xs, ys, thetas, t):
    # first, find the index of the t
    if t < ts[0] or t > ts[-1]:
        raise Exception('You absolute buffoon!')
    if t in ts:
        t_ind = ts.index(t)
        return xs[t_ind], ys[t_ind], thetas[t_ind]
    # binary search for the index
    t_ind = bisect(ts, t) - 1
    # Now compute the points we need
    xs_i, ys_i, thetas_i = xs[t_ind], ys[t_ind], thetas[t_ind]
    xs_ip1, ys_ip1, thetas_ip1 = xs[t_ind+1], ys[t_ind+1], thetas[t_ind+1]
    tau = (t - ts[t_ind]) / (ts[t_ind+1] - ts[t_ind])
    res_xs = [tau*xs_ip1[s_ind] + (1-tau)*xs_i[s_ind] for s_ind in range(len(xs_i))]
    res_ys = [tau*ys_ip1[s_ind] + (1-tau)*ys_i[s_ind] for s_ind in range(len(ys_i))]
    res_thetas = [tau*thetas_ip1[s_ind] + (1-tau)*thetas_i[s_ind] for s_ind in range(len(thetas_i))]
    return res_xs, res_ys, res_thetas

def space_interp_to_plottable(xs, ys, thetas, num=None):
    # Intialize the number of plotting points
    num = num if num is not None else 5*len(xs)
    
    # First, fit the Cubic Slines
    spl_vals = np.linspace(0, 1, len(xs)+1)
    x_vals, y_vals, theta_vals = list(xs)[::], list(ys)[::], list(thetas)[::]
    x_vals.append(x_vals[0]), y_vals.append(y_vals[0]), theta_vals.append(theta_vals[0])
    x_spline = CubicSpline(spl_vals, x_vals, bc_type='periodic')
    y_spline = CubicSpline(spl_vals, y_vals, bc_type='periodic')
    theta_spline = CubicSpline(spl_vals, theta_vals, bc_type='periodic')

    # Next make the points to plot
    spl_eval = np.linspace(0, 1, num+1)
    x_pts = x_spline(spl_eval)
    y_pts = y_spline(spl_eval)
    theta_pts = theta_spline(spl_eval)

    # Finally, return it
    return x_pts, y_pts, theta_pts

def space_and_time_interp(ts, xs, ys, thetas, n_t, n_s):
    # First make the times to use
    n_t = n_t if n_t is not None else 3*len(ts)
    max_t = ts[-1]
    min_t = ts[0]
    res_t = np.linspace(min_t, max_t, n_t)
    
    # Initialize the results
    res_x, res_y, res_theta = [], [], []

    # Now get the results for each x, y, t
    for t in res_t:
        xt, yt, thetat = get_time_interped_points(ts, xs, ys, thetas, t)
        x, y, theta = space_interp_to_plottable(xt, yt, thetat, num=n_s)
        res_x.append(x)
        res_y.append(y)
        res_theta.append(theta)

    return res_t, res_x, res_y, res_theta

def find_topmost_at_x0(x, y):
    """
    Given a closed curve (x, y), return (x_label, y_label) = (0, max_y)
    among the points where x ≈ 0 (within tolerance).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    tol = 1e-2
    x0_inds = np.where(np.abs(x) < tol)[0]

    if len(x0_inds) == 0:
        raise ValueError("No points found with x ≈ 0")

    # Return the y-max among x≈0
    top_ind = x0_inds[np.argmax(y[x0_inds])]
    return x[top_ind], y[top_ind]

import numpy as np
import matplotlib.pyplot as plt

def make_multi_plot(ts, xs, ys, thetas, n_t=10, n_s=200):
    # Interpolate space & time
    pt, px, py, ptheta = space_and_time_interp(ts, xs, ys, thetas, n_t=n_t, n_s=n_s)
    # Keep interpolation unchanged, but only plot/save up to the penultimate time.
    # pt_plot = pt[::-1]
    # px_plot = px[::-1]
    # py_plot = py[::-1]
    pt_plot, px_plot, py_plot = pt, px, py

    # Global bounds (so every small plot has identical axes)
    bds = get_bounds(px, py, ptheta)

    # Create a horizontal row of small axes
    n = len(pt_plot)
    fig_width = max(3.0 * n, 4.0)  # scale width with number of panels
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 3.2), squeeze=False)
    axes = axes[0]  # flatten

    for i, (ax, x, y) in enumerate(zip(axes, px_plot, py_plot)):
        ax.set_aspect('equal')
        ax.set_xlim(bds[0])
        ax.set_ylim(bds[1])
        ax.set_xticks([])
        ax.set_yticks([])

        # Simple black polyline (no color mapping)
        ax.plot(-y, x, 'k', linewidth=2)

        if i > 0:
            circle = plt.Circle((0, 0), initial_radius, fill=False, linestyle='--', edgecolor='gray', linewidth=1)
            ax.add_patch(circle)

    fig.tight_layout(w_pad=0.3)
    plt.savefig('nowalls_sum.png')


def main():
    make_multi_plot(all_ts, all_xs, all_ys, all_thetas, n_t=4, n_s=100)

if __name__ == "__main__":
    main()
        
