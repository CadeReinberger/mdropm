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

current_area_loc = '../out/results.pkl'
with open(current_area_loc, 'rb') as in_file:
    res_dict = pickle.load(in_file)

UE = 1
all_ts = res_dict['out_t'][::UE]
all_vecs = res_dict['out_x'][::UE]
n = len(all_vecs[0]) // 3
all_xs = [vec[:n] for vec in all_vecs]
all_ys = [vec[n:2*n] for vec in all_vecs]
all_thetas = [vec[2*n:3*n] for vec in all_vecs]

print(f'size: {(len(all_ts), len(all_xs[0]))}')
print(f'ts: {all_ts}')

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
    min_theta, max_theta = (1 + extra_factor) * min_theta, (1 + extra_factor) * max_theta

    print(f'BDS: {(min_x, min_y, max_x, max_y)}')

    return ((min_x, max_x), (min_y, max_y), (min_theta, max_theta))

def get_time_interped_points(ts, xs, ys, thetas, t):
    # first, find the index of the t
    if t < ts[0] - 1e-6 or t > ts[-1] + 1e-6:
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
    t0 = ts[0]
    max_t = ts[-1]
    res_t = np.linspace(t0, max_t, n_t)

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


def make_plot(x, y, theta, t, max_t, bds, out_path):
    # Create line segments between consecutive points
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Compute average angle for each segment
    theta_segment = np.rad2deg(0.5 * (theta[:-1] + theta[1:]))

    # Normalize theta for coloring
    colmin, colmax = [np.rad2deg(i) for i in bds[2]]
    norm = Normalize(vmin=colmin, vmax=colmax)

    # Create colored line collection
    lc = LineCollection(segments, cmap='gist_rainbow', norm=norm)
    lc.set_array(theta_segment)
    lc.set_linewidth(2)

    # Now we make the plot
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    plt.xlim(bds[0]), plt.ylim(bds[1])
    plt.xticks([]), plt.yticks([])
    plt.gcf().set_dpi(600) # just to make it clear
    plt.colorbar(lc, ax=ax)

    # --- Time Progress Bar ---
    # Add small inset axes below the main plot
    bar_ax = ax.inset_axes([0.1, -0.1, 0.6, 0.02])  # [x0, y0, width, height] in axis fraction coords
    bar_ax.set_xlim(0, max_t)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')
    # Background bar
    bar_ax.fill_between([0, max_t], 0, 1, color='lightgray', alpha=0.5)
    # Progress
    bar_ax.fill_between([0, t], 0, 1, color='black')

    # --- Time Label ---
    minutes = int(t)
    seconds = int(round((t - minutes) * 60))
    time_str = f"t = {minutes}:{seconds:02d}"

    ax.text(0.75, -0.13, time_str,
            transform=ax.transAxes,
            ha='left', va='center',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.savefig(out_path)
    plt.close(fig)

def make_animation(ts, xs, ys, thetas, n_t=None, n_s=None):
    # First get the interpolated stuff
    pt, px, py, ptheta = space_and_time_interp(ts, xs, ys, thetas, n_t, n_s)

    # Get the global bounds for x, y, theta
    bds = get_bounds(px, py, ptheta)
    max_t = ts[-1]
    print(f'bds: {bds}')
    all_images = []
    for (ind, t) in enumerate(tqdm(pt)):
        out_path = f'tmp/frame_{ind}_t_{round(1000*t)}.png'
        make_plot(px[ind], py[ind], ptheta[ind], t, max_t, bds, out_path)
        all_images.append(imageio.imread(out_path))
    imageio.mimsave('../out/full_anim.gif', all_images, duration=5)

    # Next, convert the gif to an mp4
    ff = FFmpeg(inputs={'../out/full_anim.gif': None},
                outputs={'../out/full_anim.mp4': '-y'})
    ff.run()
    waitasecond(1)

    # And we're done
    print("All Done! Animations complete! :)")


def main():
    make_animation(all_ts, all_xs, all_ys, all_thetas, n_t=200, n_s=200)

if __name__ == "__main__":
    main()
