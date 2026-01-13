import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator
import pickle
from matplotlib import pyplot as plt

current_area_loc = '../out/results.pkl'
with open(current_area_loc, 'rb') as in_file:
    res_dict = pickle.load(in_file)

K_ALPHA = 1 # Minutes

UE = 4
all_ts = res_dict['out_t'][2::UE]
all_vecs = res_dict['out_x'][2::UE]
n = len(all_vecs[0]) // 3
all_xs = [vec[:n] for vec in all_vecs]
all_ys = [vec[n:2*n] for vec in all_vecs]
all_thetas = [vec[2*n:3*n] for vec in all_vecs]


def _periodic_resample_one(x, y, theta, num=50):
    """
    x, y, theta: 1D arrays/list, sampled at equally spaced s on a periodic loop.
    Returns arrays of length `num`, also equally spaced in s, periodic-smoothed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    N = len(x)
    if len(y) != N or len(theta) != N:
        raise ValueError("x, y, theta must have the same length")

    # Parameter s is equally spaced; use [0,1) and append endpoint for periodic spline
    s = np.linspace(0.0, 1.0, N, endpoint=False)
    s_ext = np.concatenate([s, [1.0]])

    # Enforce periodic endpoint equality for CubicSpline(bc_type='periodic')
    x_ext = np.concatenate([x, [x[0]]])
    y_ext = np.concatenate([y, [y[0]]])

    # Theta: spline cos/sin instead of theta directly (avoids 2π wrap headaches)
    c = np.cos(theta)
    sn = np.sin(theta)
    c_ext = np.concatenate([c, [c[0]]])
    sn_ext = np.concatenate([sn, [sn[0]]])

    x_spl = CubicSpline(s_ext, x_ext, bc_type="periodic")
    y_spl = CubicSpline(s_ext, y_ext, bc_type="periodic")
    c_spl = CubicSpline(s_ext, c_ext, bc_type="periodic")
    s_spl = CubicSpline(s_ext, sn_ext, bc_type="periodic")

    s_new = np.linspace(0.0, 1.0, num, endpoint=False)
    x_new = x_spl(s_new)
    y_new = y_spl(s_new)

    c_new = c_spl(s_new)
    sn_new = s_spl(s_new)
    theta_new = np.arctan2(sn_new, c_new)

    return x_new, y_new, theta_new


def resample_all_curves(all_xs, all_ys, all_thetas, num=50):
    """
    all_xs/all_ys/all_thetas are lists over time_ind.
    Each entry is a same-length list/array over s samples.
    Returns new (all_xs, all_ys, all_thetas) with each time slice resampled to `num`.
    """
    new_xs, new_ys, new_thetas = [], [], []
    for x, y, th in zip(all_xs, all_ys, all_thetas):
        xr, yr, thr = _periodic_resample_one(x, y, th, num=num)
        new_xs.append(xr)
        new_ys.append(yr)
        new_thetas.append(thr)
    return new_xs, new_ys, new_thetas

all_xs, all_ys, all_thetas = resample_all_curves(all_xs, all_ys, all_thetas, num=50)

def get_bounds(xs, ys, extra_factor = .1):
    # Run the main guy to do main guy things
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf

    for ind in range(len(xs)):
        for (cur_x, cur_y) in zip(xs[ind], ys[ind]):
            min_x, max_x = min(min_x, cur_x), max(max_x, cur_x)
            min_y, max_y = min(min_y, cur_y), max(max_y, cur_y)
    min_x, max_x = (1 + extra_factor) * min_x, (1 + extra_factor) * max_x
    min_y, max_y = (1 + extra_factor) * min_y, (1 + extra_factor) * max_y

    return ((min_x, max_x), (min_y, max_y))

def to_spline_set(ts, xs, ys):
    # Let's get the number we want
    N = len(xs[0])

    # Now make all the splines
    x_splines = []
    y_splines = []
    for ind in range(N):
        x_vals = [xs[l][ind] for l in range(len(ts))]
        y_vals = [ys[l][ind] for l in range(len(ts))]
        x_spl = CubicSpline(ts, x_vals, bc_type='natural')
        y_spl = CubicSpline(ts, y_vals, bc_type='natural')
        x_splines.append(x_spl), y_splines.append(y_spl)

    return x_splines, y_splines

def compute_alpha(x_spl, y_spl, t, k_alpha=K_ALPHA):
    xt, yt = x_spl(t, 1), y_spl(t, 1)
    xtt, ytt = x_spl(t, 2), y_spl(t, 2)
    if (xt**2 + yt**2) == 0:
        return None
    return k_alpha * (xt*ytt - yt*xtt) / (xt**2 + yt**2)

def make_alpha_interpolator(ts, xs, ys, k_alpha=K_ALPHA):
    # First compute all the splines since we'll need them
    x_splines, y_splines = to_spline_set(ts, xs, ys)
    ns = len(x_splines)

    # Now let's make the guy to interpolate
    all_in, all_out = [], []
    t_eval = np.linspace(ts[0], ts[-1], num=len(ts))
    for s_ind in range(ns):
        for t in t_eval: # Could do anything here
            x_val, y_val = x_splines[s_ind](t), y_splines[s_ind](t)
            alpha_val = compute_alpha(x_splines[s_ind], y_splines[s_ind], t, k_alpha=k_alpha)
            if alpha_val is not None:
                all_in.append(np.array([x_val, y_val]))
                all_out.append(compute_alpha(x_splines[s_ind], y_splines[s_ind], t, k_alpha=k_alpha))

    # Now make the interpolator
    alpha_interp = LinearNDInterpolator(all_in, all_out, fill_value=0)
    return alpha_interp

def make_alpha_plot(ts, xs, ys, k_alpha=K_ALPHA):
        alpha_interp = make_alpha_interpolator(ts, xs, ys, k_alpha=k_alpha)

        # Make a mesh grid so we can do this
        (x_min,x_max), (y_min, y_max) = get_bounds(xs, ys)
        x = np.linspace(x_min, x_max, 1000)
        y = np.linspace(y_min, y_max, 1000)
        X, Y = np.meshgrid(x, y)

        # Interpoalte this mamajama
        Z = np.rad2deg(alpha_interp(X, Y))

        # Get the colorscale we want
        z_flat = Z.flatten()
        z_flat_nonzero = z_flat[z_flat != 0]
        vlim = np.percentile(np.abs(z_flat_nonzero), 95)

        # Make the plot
        plt.figure() # make a new plot from the old one
        plt.pcolormesh(X, Y, Z, cmap='bwr', vmin=-vlim, vmax=vlim)
        plt.colorbar()
        plt.title(r'Predicted Skewness Angle ($^\circ$)')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_dpi(1000)
        plt.savefig(f'../out/chirality_prediction.png', dpi=1000, bbox_inches='tight')


def main():
    print('Starting to produce chirality plot...')
    make_alpha_plot(all_ts, all_xs, all_ys)
    print('Chirality Plot Complete!')

if __name__ == '__main__':
    main()
