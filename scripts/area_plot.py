import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator
import pickle
from matplotlib import pyplot as plt

current_area_loc = '../archived/run_25_half/out/results.pkl'
with open(current_area_loc, 'rb') as in_file:
    res_dict = pickle.load(in_file)

K_ALPHA = 1 # Minutes

start_ind = 2
UE = 1
end_prop = .95
end_ind = int(end_prop * len(res_dict['out_t']))

all_ts = np.array(res_dict['out_t'][2:end_ind:UE])
all_vecs = res_dict['out_x'][2:end_ind:UE]
n = len(all_vecs[0]) // 3
all_xs = [vec[:n] for vec in all_vecs]
all_ys = [vec[n:2*n] for vec in all_vecs]
all_thetas = [vec[2*n:3*n] for vec in all_vecs]

def area_shoelace(x, y):
    return .5 * np.abs(sum(x*np.roll(y, 1) - y*np.roll(x, 1)))

all_areas = np.array([area_shoelace(x, y) for (x, y) in zip(all_xs, all_ys)])

'''
LOGIC FOR PLOTTING ALL THE AREAS. SHOWS WE HAVE GOOD SMOOTHNESS

plt.plot(all_ts, all_areas, 'k--')
plt.show()
'''

def fit_poly_find_next_root_and_plot(x, y, order=3, eps=1e-12, ax=None):
    """
    Fit a polynomial of given order to (x, y), find the first real root > x[-1],
    and plot the data plus fitted polynomial on [x[0], root].

    Returns
    -------
    root : float
        The first real root strictly greater than x[-1].
    coeffs : np.ndarray
        Polynomial coefficients in descending powers (np.polyfit convention).
    poly : np.poly1d
        The fitted polynomial as a callable object.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D arrays/lists of the same length.")
    if len(x) < order + 1:
        raise ValueError(f"Need at least {order+1} points to fit a degree-{order} polynomial.")

    # Fit polynomial
    coeffs = np.polyfit(x, y, order)
    poly = np.poly1d(coeffs)

    # Find roots of fitted polynomial
    roots = np.roots(coeffs)

    # Filter to real roots
    real_roots = roots[np.abs(roots.imag) < 1e-9].real

    # Find the first root strictly after x[-1]
    x_last = x[-1]
    candidates = np.sort(real_roots[real_roots > x_last + eps])

    if candidates.size == 0:
        raise RuntimeError(
            "No real root found to the right of x[-1] for the fitted polynomial."
        )

    root = float(candidates[0])

    # Plot
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(x, y, label="data")

    xs = np.linspace(x[0], root, 400)
    ax.plot(xs, poly(xs), label=f"degree-{order} fit")

    ax.axvline(x_last, linestyle="--", linewidth=1, label="x[-1]")
    ax.axvline(root, linestyle="--", linewidth=1, label="next root")
    ax.set_xlim(min(x[0], x_last), root)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()

    return root, coeffs, poly

root, _, _ = fit_poly_find_next_root_and_plot(all_ts, all_areas)

print(f'Drying Guess: {root}')
