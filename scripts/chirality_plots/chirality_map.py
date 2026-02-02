import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator, Akima1DInterpolator
import pickle
from matplotlib import pyplot as plt

current_area_loc = 'archived/run_25_half/out/results.pkl'
with open(current_area_loc, 'rb') as in_file:
    res_dict = pickle.load(in_file)

K_ALPHA = .2 # Minutes

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

# all_xs, all_ys, all_thetas = resample_all_curves(all_xs, all_ys, all_thetas, num=10)

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

def to_spline_set_old(ts, xs, ys):
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
        
        # Let's make these splines a bit more robust
        # x_spl = Akima1DInterpolator(ts, x_vals)
        # y_spl = Akima1DInterpolator(ts, y_vals)

        x_splines.append(x_spl), y_splines.append(y_spl)

    return x_splines, y_splines

def to_spline_set(ts, xs, ys, n_intervals=8, smooth_fallback=1e-6):
    """
    Drop-in replacement for to_spline_set() that does *spline regression* instead of
    exact interpolation, to stabilize x', y', x'', y''.

    Fits cubic least-squares splines with a small fixed knot set:
        scipy.interpolate.LSQUnivariateSpline

    Parameters
    ----------
    ts : (T,) array-like
        Times (must be strictly increasing for the spline routines).
    xs, ys : list length T, each entry array-like length N
        Your data layout: xs[t_ind][s_ind], ys[t_ind][s_ind].
    n_intervals : int
        Number of knot intervals (default 5). This creates (n_intervals-1) interior knots.
        Example: n_intervals=5 -> 4 interior knots -> pretty strong smoothing.
    smooth_fallback : float
        If LSQUnivariateSpline fails (e.g. not enough points), fallback to
        UnivariateSpline with smoothing `s = smooth_fallback * T * var(data)`.

    Returns
    -------
    x_splines, y_splines : lists of spline objects length N
        Each spline supports calls like:
            x_splines[s_ind](t)      # value
            x_splines[s_ind](t, 1)   # first derivative
            x_splines[s_ind](t, 2)   # second derivative
    """
    from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline

    t = np.asarray(ts, dtype=float)
    if t.ndim != 1:
        raise ValueError("ts must be 1D")
    T = t.size
    if len(xs) != T or len(ys) != T:
        raise ValueError("xs and ys must have outer length equal to len(ts)")
    if T < 4:
        raise ValueError("Need at least 4 time samples for a cubic spline regression")

    # Ensure strictly increasing t
    if not np.all(np.diff(t) > 0):
        # If you have duplicates, this will *not* be well-defined; remove duplicates deterministically.
        # Keep first occurrence of each unique time.
        uniq_t, uniq_idx = np.unique(t, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        t = t[uniq_idx]
        xs = [xs[i] for i in uniq_idx]
        ys = [ys[i] for i in uniq_idx]
        T = t.size
        if T < 4 or not np.all(np.diff(t) > 0):
            raise ValueError("ts must be strictly increasing after removing duplicates")

    # number of curves
    N = len(xs[0])
    for l in range(T):
        if len(xs[l]) != N or len(ys[l]) != N:
            raise ValueError("Each xs[l], ys[l] must have the same length (number of s_ind)")

    # Build interior knots: n_intervals => (n_intervals-1) interior knots
    if n_intervals < 2:
        raise ValueError("n_intervals must be >= 2")

    n_int = n_intervals - 1
    # Need at least k+1 + n_int points for LSQUnivariateSpline with k=3 (rule of thumb)
    # Practically, require T > n_int + 3
    use_lsq = (T > n_int + 3) and (n_int > 0)

    if use_lsq:
        # place interior knots evenly in time, excluding endpoints
        # (avoid exact endpoints — LSQUnivariateSpline requires strictly inside)
        knots = np.linspace(t[0], t[-1], n_intervals + 1)[1:-1]
        # If time range is tiny, this can produce duplicates; guard:
        knots = np.unique(knots)
        if knots.size == 0:
            use_lsq = False
    else:
        knots = np.array([], dtype=float)

    x_splines = []
    y_splines = []

    for s_ind in range(N):
        x_vals = np.asarray([xs[l][s_ind] for l in range(T)], dtype=float)
        y_vals = np.asarray([ys[l][s_ind] for l in range(T)], dtype=float)

        if use_lsq:
            try:
                xspl = LSQUnivariateSpline(t, x_vals, t=knots, k=3)
            except Exception:
                # fallback: smoothing spline with tiny smoothing scaled to variance
                s = float(smooth_fallback) * T * (np.var(x_vals) + 1e-12)
                xspl = UnivariateSpline(t, x_vals, k=3, s=s)

            try:
                yspl = LSQUnivariateSpline(t, y_vals, t=knots, k=3)
            except Exception:
                s = float(smooth_fallback) * T * (np.var(y_vals) + 1e-12)
                yspl = UnivariateSpline(t, y_vals, k=3, s=s)
        else:
            # If you don't have enough points for LSQ spline w/ knots, use smoothing spline.
            s = float(smooth_fallback) * T * (np.var(x_vals) + 1e-12)
            xspl = UnivariateSpline(t, x_vals, k=3, s=s)

            s = float(smooth_fallback) * T * (np.var(y_vals) + 1e-12)
            yspl = UnivariateSpline(t, y_vals, k=3, s=s)

        x_splines.append(xspl)
        y_splines.append(yspl)

    return x_splines, y_splines


def compute_alpha(x_spl, y_spl, t, k_alpha=K_ALPHA):
    xt, yt = x_spl(t, 1), y_spl(t, 1)
    xtt, ytt = x_spl(t, 2), y_spl(t, 2)
    if (xt**2 + yt**2) == 0:
        return None
    return k_alpha * (xt*ytt - yt*xtt) / (xt**2 + yt**2)**1.5

def make_alpha_interpolator(ts, xs, ys, k_alpha=K_ALPHA):
    # First compute all the splines since we'll need them
    x_splines, y_splines = to_spline_set(ts, xs, ys)
    ns = len(x_splines)

    # Now let's make the guy to interpolate
    all_in, all_out = [], []
    t_eval = np.linspace(ts[0], ts[-1], num=len(ts)) # TODO: FIX
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


def make_alpha_interpolator_loc(loc, k_alpha=K_ALPHA):
    with open(loc, 'rb') as in_file:
        res_dict = pickle.load(in_file)


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

    return make_alpha_interpolator(all_ts, all_xs, all_ys, k_alpha=k_alpha)

def smooth_grid_field(Z, x_eval, y_eval, method="helmholtz",
                      sigma=None, length_scale=None, iters=200, tol=1e-6):
    """
    Smooth a 2D scalar field Z(y,x) living on a tensor-product grid defined by
    x_eval (Nx,) and y_eval (Ny,).

    Parameters
    ----------
    Z : (Ny, Nx) array
        Field values on the grid (e.g. output of your interpolator evaluated on meshgrid).
        Assumes Z[i,j] corresponds to y_eval[i], x_eval[j].
    x_eval, y_eval : 1D arrays
        Grid coordinates (uniformly spaced is assumed for the PDE method).
    method : {"gaussian", "helmholtz"}
        - "gaussian": scipy.ndimage.gaussian_filter
        - "helmholtz": solve (I - lambda * Laplacian) u = f with Neumann BCs via Jacobi iterations
    sigma : float or tuple, optional
        Gaussian blur std in *grid cells* if method="gaussian".
        If None, chosen from length_scale if provided.
    length_scale : float, optional
        Smoothing length scale in *physical units* (same units as x,y).
        For "gaussian", sigma is derived from length_scale.
        For "helmholtz", lambda is derived from length_scale.
    iters : int
        Max iterations for helmholtz solver.
    tol : float
        Relative stopping tolerance for helmholtz solver.

    Returns
    -------
    Zs : (Ny, Nx) array
        Smoothed field.
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)

    if Z.ndim != 2:
        raise ValueError("Z must be 2D (Ny, Nx).")
    Ny, Nx = Z.shape
    if x_eval.ndim != 1 or y_eval.ndim != 1:
        raise ValueError("x_eval and y_eval must be 1D arrays.")
    if Nx != x_eval.size or Ny != y_eval.size:
        raise ValueError(f"Shape mismatch: Z is {(Ny,Nx)} but x_eval,y_eval are {(x_eval.size,y_eval.size)}.")

    # grid spacing (assume uniform)
    if Nx < 2 or Ny < 2:
        return Z.copy()
    dx = float(x_eval[1] - x_eval[0])
    dy = float(y_eval[1] - y_eval[0])
    if dx == 0 or dy == 0:
        return Z.copy()

    # Handle NaNs by filling with 0 before smoothing (common if your interpolator returned nan).
    # If you want smarter inpainting, say so, but this is usually fine for "outside region" zeros.
    Z0 = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    method = method.lower().strip()
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter

        if sigma is None:
            if length_scale is None:
                sigma = 1.0
            else:
                # convert physical length_scale to grid-cells sigma
                sigma_x = max(0.5, float(length_scale / abs(dx)))
                sigma_y = max(0.5, float(length_scale / abs(dy)))
                sigma = (sigma_y, sigma_x)  # (y,x) order

        # mode="nearest" behaves like a mild Neumann-ish condition at the boundary
        return gaussian_filter(Z0, sigma=sigma, mode="nearest")

    if method != "helmholtz":
        raise ValueError("method must be 'gaussian' or 'helmholtz'.")

    # -----------------------------
    # Helmholtz smoothing
    # Solve: (I - lambda * Laplacian) U = Z0   with Neumann BC
    #
    # Choose lambda from length_scale: lambda ~ L^2 is a decent physical scaling.
    # Bigger lambda => stronger smoothing.
    # -----------------------------
    if length_scale is None:
        # default: smooth over ~2 grid cells
        L = 2.0 * max(abs(dx), abs(dy))
    else:
        L = float(length_scale)

    lam = L * L

    # Precompute stencil coeffs
    ax = lam / (dx * dx)
    ay = lam / (dy * dy)
    denom = 1.0 + 2.0 * ax + 2.0 * ay

    U = Z0.copy()
    # Weighted Jacobi parameter (stable for this SPD system)
    omega = 2.0 / 3.0

    # helper: apply Neumann BC via "mirror" indexing
    def _shift_x_plus(A):
        B = np.empty_like(A)
        B[:, :-1] = A[:, 1:]
        B[:, -1] = A[:, -2]   # mirror at boundary
        return B

    def _shift_x_minus(A):
        B = np.empty_like(A)
        B[:, 1:] = A[:, :-1]
        B[:, 0] = A[:, 1]     # mirror
        return B

    def _shift_y_plus(A):
        B = np.empty_like(A)
        B[:-1, :] = A[1:, :]
        B[-1, :] = A[-2, :]   # mirror
        return B

    def _shift_y_minus(A):
        B = np.empty_like(A)
        B[1:, :] = A[:-1, :]
        B[0, :] = A[1, :]     # mirror
        return B

    normZ = np.linalg.norm(Z0.ravel()) + 1e-30

    for _ in range(int(iters)):
        Uxp = _shift_x_plus(U)
        Uxm = _shift_x_minus(U)
        Uyp = _shift_y_plus(U)
        Uym = _shift_y_minus(U)

        # Jacobi update for (I - lam*Lap)U = Z0  =>  (1+2ax+2ay)U - ax(Uxp+Uxm) - ay(Uyp+Uym) = Z0
        U_new = (Z0 + ax * (Uxp + Uxm) + ay * (Uyp + Uym)) / denom

        # relaxation
        U_next = (1.0 - omega) * U + omega * U_new

        # convergence check (relative change)
        rel = np.linalg.norm((U_next - U).ravel()) / normZ
        U = U_next
        if rel < tol:
            break

    return U


def make_alpha_plot(ts, xs, ys, k_alpha=K_ALPHA):
        alpha_interp = make_alpha_interpolator(ts, xs, ys, k_alpha=k_alpha)

        # Make a mesh grid so we can do this
        (x_min,x_max), (y_min, y_max) = get_bounds(xs, ys)
        # TODO: FIX
        (x_min, x_max), (y_min, y_max) = (-1.625, 1.625), (0, 3.25)
        x = np.linspace(x_min, x_max, 1000)
        y = np.linspace(y_min, y_max, 1000)
        X, Y = np.meshgrid(x, y)

        # Interpoalte this mamajama
        Z = np.rad2deg(alpha_interp(X, Y))
        
        with open('data.pkl', 'wb') as file:
            pickle.dump(Z, file)
        # Z = smooth_grid_field(Z, x, y, method="helmholtz", length_scale=0.025)  # pick a physical length

        # Get the colorscale we want
        z_flat = Z.flatten()
        z_flat_nonzero = z_flat[z_flat != 0]
        vlim = np.percentile(np.abs(z_flat_nonzero), 90)

        # Make outline to add to plot
        t_min = np.arccos(1.625/3)
        t = np.linspace(t_min, np.pi-t_min, num=100)
        # Shift down a bit because bdry is 1-sided
        outline_x, outline_y = 2.99*np.cos(t), 2.97*np.sin(t)

        # Make the plot
        plt.figure() # make a new plot from the old one
        plt.pcolormesh(X, Y, Z, cmap='bwr', vmin=-vlim, vmax=vlim)
        plt.plot(outline_x, outline_y, 'k-', linewidth=3)
        plt.colorbar()
        plt.title(r'Predicted Skewness Angle ($^\circ$)')
        plt.xticks([])
        plt.yticks([])
        plt.gcf().set_dpi(1000)
        plt.savefig(f'chiral_map.png', dpi=1000, bbox_inches='tight')

def plot_xy_spline_fits(
    ts,
    xs,
    ys,
    s_inds=None,
    dense=400,
    show=True,
    savepath=None,
    max_curves=50,
    marker_size=10,
    line_width=1.5,
):
    """Plot x(t) and y(t) sample points and their fitted cubic splines.

    This is meant as a quick sanity-check that `to_spline_set()` is fitting what you
    think it is fitting.

    Notes about your data layout
    ----------------------------
    In this file, `xs` and `ys` are lists over time index `l` (same length as `ts`).
    Each entry `xs[l]` / `ys[l]` is a list/array over spatial index `s_ind`.
    For a fixed `s_ind`, the sample points are:
        t -> x_vals[l] = xs[l][s_ind]
        t -> y_vals[l] = ys[l][s_ind]
    """
    import numpy as np
    from matplotlib import pyplot as plt

    ts = np.asarray(ts, dtype=float)
    if len(xs) != len(ts) or len(ys) != len(ts):
        raise ValueError("xs and ys must have the same outer length as ts")
    if len(ts) < 2:
        raise ValueError("Need at least 2 time points to plot spline fits")

    # Fit splines
    x_splines, y_splines = to_spline_set(ts, xs, ys)
    N = len(x_splines)

    # Choose which curves to show
    if s_inds is None:
        if N <= max_curves:
            s_inds = list(range(N))
        else:
            s_inds = np.linspace(0, N - 1, num=max_curves, dtype=int).tolist()
    else:
        s_inds = list(s_inds)

    t_dense = np.linspace(ts[0], ts[-1], int(dense))

    fig, (ax_x, ax_y) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    for s_ind in s_inds:
        if s_ind < 0 or s_ind >= N:
            continue

        x_vals = np.array([xs[l][s_ind] for l in range(len(ts))], dtype=float)
        y_vals = np.array([ys[l][s_ind] for l in range(len(ts))], dtype=float)

        # sample points
        ax_x.scatter(ts, x_vals, s=marker_size)
        ax_y.scatter(ts, y_vals, s=marker_size)

        # spline curves
        ax_x.plot(t_dense, x_splines[s_ind](t_dense), linewidth=line_width)
        ax_y.plot(t_dense, y_splines[s_ind](t_dense), linewidth=line_width)

    ax_x.set_ylabel("x(t)")
    ax_y.set_ylabel("y(t)")
    ax_y.set_xlabel("t")
    ax_x.set_title("Sample points (markers) vs. CubicSpline fits (lines)")

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()

    return fig, (ax_x, ax_y)



def main():
    print('Starting to produce chirality plot...')
    s_inds = list(range(1, 25, 5))
    # plot_xy_spline_fits(all_ts, all_xs, all_ys, s_inds, savepath=None, show=True)
    make_alpha_plot(all_ts, all_xs, all_ys)
    print('Chirality Plot Complete!')

if __name__ == '__main__':
    main()
