import numpy as np
from chirality_map import to_spline_set
from scipy.interpolate import CubicSpline, LinearNDInterpolator
import pickle
from matplotlib import pyplot as plt

current_area_loc = '../../archived/run_25_half/out/results.pkl'
with open(current_area_loc, 'rb') as in_file:
    res_dict = pickle.load(in_file)

K_ALPHA = 2 / 10 # mm

start_ind = 2
UE = 1
end_prop = .95
end_ind = int(end_prop * len(res_dict['out_t']))

all_ts = res_dict['out_t'][2:end_ind:UE]
all_vecs = res_dict['out_x'][2:end_ind:UE]
n = len(all_vecs[0]) // 3
all_xs = [vec[:n] for vec in all_vecs]
all_ys = [vec[n:2*n] for vec in all_vecs]
all_thetas = [vec[2*n:3*n] for vec in all_vecs]

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

def compute_alpha(x_spl, y_spl, t, k_alpha=K_ALPHA):
    xt, yt = x_spl(t, 1), y_spl(t, 1)
    xtt, ytt = x_spl(t, 2), y_spl(t, 2)
    if (xt**2 + yt**2) == 0:
        return None
    return k_alpha * (xt*ytt - yt*xtt) / (xt**2 + yt**2)**1 # 1 power for length delay

def make_alpha_interpolator(ts, xs, ys, k_alpha=K_ALPHA):
    # First compute all the splines since we'll need them
    x_splines, y_splines = to_spline_set(ts, xs, ys, n_intervals=5)
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

def make_alpha_top_plot(ts, xs, ys, k_alpha=K_ALPHA, ALPHA_DIFF=45):
        alpha_interp = make_alpha_interpolator(ts, xs, ys, k_alpha=k_alpha)

        # Make a mesh grid so we can do this
        (x_min,x_max), (y_min, y_max) = get_bounds(xs, ys)
        R = .25*sum(sum(abs(x) for x in y) for y in get_bounds(xs, ys, extra_factor=0))

        plt.figure()

        for shft in np.linspace(0, 2, num=5):

            theta_eval = np.linspace(.5*np.pi-np.deg2rad(ALPHA_DIFF), .5*np.pi+np.deg2rad(ALPHA_DIFF), 10)
            x_eval = .98*R*np.cos(theta_eval)
            y_eval = .98*R*np.sin(theta_eval)*np.ones(len(x_eval)) - shft

            #y_eval = .99*R*np.sin(theta_eval)

            # Interpoalte this mamajama
            Z = np.rad2deg(alpha_interp(x_eval, y_eval))

            # Now we cubic spline
            cs = CubicSpline(x_eval[::-1], -Z[::-1])

            # Make the plotting points
            x_plt = np.linspace(min(x_eval), max(x_eval), num=200)
            z_plt = cs(x_plt)
        
            # Make the plot
            plt.plot(100 * x_plt / R, z_plt, label=f'shift: {shft}')
        
        plt.title(r'Predicted Skewness Angle ($^\circ$)')
        plt.xlabel('x length (% radius)')
        plt.ylabel('alpha (degrees)')
        plt.legend()
        plt.gcf().set_dpi(1000)
        plt.savefig(f'line_scans_theory.png', dpi=1000, bbox_inches='tight')


def main():
    # print('Starting to produce chirality plots...')
    make_alpha_top_plot(all_ts, all_xs, all_ys)
    # print('Chirality Plots Complete!')

if __name__ == '__main__':
    main()

