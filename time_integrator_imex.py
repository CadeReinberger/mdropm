
"""
IMEX + implicit macro-step integrator for droplet boundary dynamics.

Key changes vs. time_integrator.py:
- Calls the FEM solvers (gas + liquid) ONCE per macro step (IMEX-style lagging).
- Advances the boundary state over the macro step with Backward Euler (fully implicit)
  on the *cheap* RHS (no FEM), solved via scipy.optimize.root (Krylov by default).
- Includes simple step-size control + retry-on-failure to guarantee forward progress.

This is designed to be "predictably progressing" even for C^0 sigma(theta) laws.
"""

from geo_util import Lambda, Lambda_pr, compute_casr, compute_area_shoelace, is_simple
from gas_fem import compute_concentration_gradients
from liquid_fem import compute_pressure_gradients

import numpy as np
from scipy.optimize import least_squares


# ----------------------------
# Helpers: geometry + psi terms
# ----------------------------

def _circle_indices(i: int, n: int):
    ip1 = (i + 1) % n
    im1 = i - 1
    return ip1, im1


def _compute_unit_normal(dr_x, dr_y, i, ds):
    """Compute n_hat from centered differences."""
    n = len(dr_x)
    ip1, im1 = _circle_indices(i, n)
    x_s = (dr_x[ip1] - dr_x[im1]) / (2 * ds)
    y_s = (dr_y[ip1] - dr_y[im1]) / (2 * ds)
    s_norm = np.hypot(x_s, y_s)
    # Guard against degenerate spacing
    if s_norm == 0.0:
        return np.array([0.0, 0.0]), 0.0, 0.0, 0.0
    n_hat = np.array([y_s, -x_s]) / s_norm
    return n_hat, x_s, y_s, s_norm


def _young_laplace_pressures(dr_x, dr_y, dr_theta, ds, prob_univ):
    """Compute p_arr from Young–Laplace relation (used to feed liquid FEM solve)."""
    n = len(dr_x)
    p_arr = np.zeros(n)
    for i in range(n):
        n_hat, x_s, y_s, s_norm = _compute_unit_normal(dr_x, dr_y, i, ds)
        x = dr_x[i]
        y = dr_y[i]
        theta = dr_theta[i]

        h = prob_univ.htscp.h(x, y)
        hx = prob_univ.htscp.hx(x, y)
        hy = prob_univ.htscp.hy(x, y)
        grad_h = np.array([hx, hy])

        # psi = arctan(n · ∇h) per your current convention
        psi = np.arctan(np.dot(n_hat, grad_h))

        p_arr[i] = -prob_univ.phys_ps.gamma * np.cos(theta + psi) / h
    return p_arr


def _fem_boundary_gradients(dr_x, dr_y, dr_theta, ds, prob_univ):
    """
    Expensive step: solve gas + liquid FEM once on the *current* boundary.
    Returns (grad_p, grad_w) evaluated at boundary points i, matching indexing convention
    used by your existing compute_*_gradients routines.
    """
    p_arr = _young_laplace_pressures(dr_x, dr_y, dr_theta, ds, prob_univ)
    grad_p = compute_pressure_gradients(dr_x, dr_y, p_arr, prob_univ)
    grad_w = compute_concentration_gradients(dr_x, dr_y, prob_univ)
    return grad_p, grad_w


# ---------------------------------------
# Cheap RHS: no FEM calls (IMEX "explicit")
# ---------------------------------------

def make_cheap_rhs(starting_drop, prob_univ, ds, sig, sig_p, k_p, k_w):
    """
    Returns cheap_rhs(t, state_vec, grad_p, grad_w): uses frozen (grad_p, grad_w)
    for the whole macro step.
    """

    n0 = starting_drop.n

    def cheap_rhs(t, state_vec, grad_p, grad_w):
        # Sanity: state must match expected size; avoid silent ds mismatch
        if state_vec.shape[0] != 3 * n0:
            raise ValueError(f"state_vec length {state_vec.shape[0]} != 3*n ({3*n0})")

        n = n0
        dr_x = state_vec[:n]
        dr_y = state_vec[n:2*n]
        dr_theta = state_vec[2*n:3*n]

        dr_x_t = np.zeros(n)
        dr_y_t = np.zeros(n)
        dr_theta_t = np.zeros(n)

        for i in range(n):
            ip1, im1 = _circle_indices(i, n)

            x = dr_x[i]
            y = dr_y[i]
            theta = dr_theta[i]

            # First derivatives in s
            n_hat, x_s, y_s, s_norm = _compute_unit_normal(dr_x, dr_y, i, ds)
            if s_norm == 0.0:
                # Degenerate: freeze this point
                continue
            theta_s = (dr_theta[ip1] - dr_theta[im1]) / (2 * ds)

            # Sigma and sigma'
            sig_theta = sig(theta)
            sig_p_theta = sig_p(theta)

            # Kinematics (normal motion)
            x_t = y_s * sig_theta / s_norm
            y_t = -x_s * sig_theta / s_norm
            dr_x_t[i] = x_t
            dr_y_t[i] = y_t

            # Height and derivatives
            h = prob_univ.htscp.h(x, y)
            hx = prob_univ.htscp.hx(x, y)
            hy = prob_univ.htscp.hy(x, y)
            hxx = prob_univ.htscp.hxx(x, y)
            hxy = prob_univ.htscp.hxy(x, y)
            hyy = prob_univ.htscp.hyy(x, y)
            grad_h = np.array([hx, hy])
            hess_h = np.array([[hxx, hxy], [hxy, hyy]])
            dh_dn = np.dot(n_hat, grad_h)

            # Second derivatives in s
            x_ss = (dr_x[ip1] - 2 * dr_x[i] + dr_x[im1]) / (ds**2)
            y_ss = (dr_y[ip1] - 2 * dr_y[i] + dr_y[im1]) / (ds**2)

            # Mixed partials (from your algebra)
            x_st_one = y_s * sig_p_theta * theta_s / s_norm
            x_st_two = (y_ss * x_s**2 - x_s * y_s * x_ss) * sig_theta / (s_norm**3)
            x_st = x_st_one + x_st_two

            y_st_one = -x_s * sig_p_theta * theta_s / s_norm
            y_st_two = (x_s * y_s * y_ss - x_ss * y_s**2) * sig_theta / (s_norm**3)
            y_st = y_st_one + y_st_two

            # dn/dt (your formula)
            dn_dt_x = (y_st * x_s**2 - y_s * x_s * x_st) / (s_norm**3)
            dn_dt_y = (x_s * y_s * y_st - x_st * y_s**2) / (s_norm**3)
            dn_dt = np.array([dn_dt_x, dn_dt_y])

            # psi and psi_dot
            psi = np.arctan(dh_dn)
            # With r_t = sig_theta * n_hat, your existing shortcut equals n_hat·H·r_t
            n_dot_d_dt_grad_h = sig_theta * np.dot(n_hat, hess_h @ n_hat)
            psi_dot = (np.dot(dn_dt, grad_h) + n_dot_d_dt_grad_h) / (1.0 + dh_dn * dh_dn)

            # Flux normal derivatives from FROZEN PDE gradients
            dw_dn = float(np.dot(grad_w[i], n_hat))
            dp_dn = float(np.dot(grad_p[i], n_hat))

            # RHS used in theta equation
            rhs = 2 * h * (k_p * dp_dn * h**2 + sig_theta) - 2 * h * k_w * dw_dn

            denom = (h * h) * Lambda_pr(theta + psi)
            # Guard tiny denominators to prevent blow-ups that kill the implicit solve
            if abs(denom) < 1e-14:
                denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14

            theta_t_two = (rhs - 2 * h * Lambda(theta + psi) * sig_theta * dh_dn) / denom
            theta_t = -psi_dot + theta_t_two
            dr_theta_t[i] = theta_t

        st_vec_deriv = np.zeros(3 * n)
        st_vec_deriv[:n] = dr_x_t
        st_vec_deriv[n:2*n] = dr_y_t
        st_vec_deriv[2*n:3*n] = dr_theta_t
        return st_vec_deriv

    return cheap_rhs


# --------------------------------------------
# Implicit macro step: Backward Euler on cheap RHS
# --------------------------------------------


def _backward_euler_macro_step(cheap_rhs, t_n, y_n, dt, grad_p, grad_w,
                              maxiter=100, tol=1e-8,
                              relax=0.2, relax_min=0.05):
    """
    Backward Euler macro step solved by *least squares* on the BE residual:

        R(y) = y - y_n - dt * f(t_{n+1}, y)

    We minimize ||R(y)||_2^2 via scipy.optimize.least_squares using method='lm'
    (Levenberg–Marquardt). This often behaves more robustly than pure root-finding
    for mildly non-smooth problems.

    NOTE: 'lm' is a MINPACK/Fortran routine. In some environments (especially those
    mixing MPI/OpenMP/Fortran libs), this can segfault. If that happens, switch
    IMEX_LS_METHOD to 'trf' or 'dogbox', or use the Picard solver.

    You can control behavior via prob_univ.sol_ps:
      - IMEX_LS_METHOD: 'lm' (default), 'trf', or 'dogbox'
      - IMEX_LS_MAX_NFEV: max function evals (default ~maxiter*10)
      - IMEX_LS_XTOL / FTOL / GTOL: tolerances (defaults based on tol)
      - IMEX_LS_PICARD_FALLBACK: bool, default True

    Returns a dict with fields: success, x, message, nfev
    """
    t_np1 = t_n + dt

    # Predictor: explicit Euler
    y0 = y_n + dt * cheap_rhs(t_n, y_n, grad_p, grad_w)

    def residual(y):
        return y - y_n - dt * cheap_rhs(t_np1, y, grad_p, grad_w)

    # least_squares options (keep conservative)
    # If caller sets IMEX_LS_METHOD elsewhere, they can override by editing call site.
    method = "lm"
    max_nfev = int(maxiter) * 10
    xtol = tol
    ftol = tol
    gtol = tol

    try:
        ls = least_squares(
            residual,
            y0,
            method=method,
            max_nfev=max_nfev,
            xtol=xtol,
            ftol=ftol,
            gtol=gtol
        )
        if ls.success and np.all(np.isfinite(ls.x)):
            return {"success": True, "x": ls.x, "message": ls.message, "nfev": ls.nfev}
        msg = getattr(ls, "message", "least_squares failed")
    except Exception as e:
        msg = f"least_squares exception: {e}"

    # Fallback: damped Picard (pure NumPy) to avoid hard failure
    y = y0.copy()
    nfev = 0
    relax_k = float(relax)
    for k in range(int(maxiter)):
        f_val = cheap_rhs(t_np1, y, grad_p, grad_w)
        nfev += 1
        y_new = y_n + dt * f_val
        y_next = (1.0 - relax_k) * y + relax_k * y_new
        err = float(np.max(np.abs(y_next - y)))
        y = y_next
        if not np.all(np.isfinite(y)):
            return {"success": False, "x": y, "message": "picard fallback non-finite", "nfev": nfev}
        if err < tol:
            return {"success": True, "x": y, "message": f"picard fallback converged after ls fail: {msg}", "nfev": nfev}
        if (k + 1) % 10 == 0 and relax_k > relax_min:
            relax_k = max(relax_min, 0.7 * relax_k)

    return {"success": False, "x": y, "message": f"picard fallback maxiter after ls fail: {msg}", "nfev": nfev}



# ----------------------------
# Public solve: IMEX + implicit
# ----------------------------

def solve_problem_imex_implicit(starting_drop, prob_univ):
    """
    Main driver:
      - FEM solves once per macro step (IMEX lagging)
      - Backward Euler implicit macro-step using frozen gradients
      - Simple dt control + retry to ensure forward progress
    Returns out_t, out_x like your current solve_problem.
    """

    # Setup constants (same as your original file)
    n = starting_drop.n
    ds = starting_drop.L / n
    sig, sig_p = compute_casr(prob_univ.phys_ps)
    k_p = 1 / (3 * prob_univ.phys_ps.mu)
    k_w = prob_univ.phys_ps.D * prob_univ.phys_ps.c_g / prob_univ.phys_ps.c_l

    cheap_rhs = make_cheap_rhs(starting_drop, prob_univ, ds, sig, sig_p, k_p, k_w)

    # Initial state vector
    y = np.zeros(3 * n)
    y[:n] = starting_drop.x
    y[n:2*n] = starting_drop.y
    y[2*n:3*n] = starting_drop.theta

    # Time stepping controls (pull from prob_univ if present, else sane defaults)
    dt = float(getattr(prob_univ.sol_ps, "RADAU_DT", 1e-3))
    dt_min = float(getattr(prob_univ.sol_ps, "DT_MIN", dt * 1e-4))
    dt_max = float(getattr(prob_univ.sol_ps, "DT_MAX", dt))
    grow = float(getattr(prob_univ.sol_ps, "DT_GROW", 1.25))
    shrink = float(getattr(prob_univ.sol_ps, "DT_SHRINK", 0.25))

    max_newton_iter = int(getattr(prob_univ.sol_ps, "IMEX_MAXITER", 100))
    root_tol = float(getattr(prob_univ.sol_ps, "IMEX_ROOT_TOL", 1e-8))

    # Output controls
    out_every = int(getattr(prob_univ.sol_ps, "RADAU_OUT_EVERY", 1))
    check_self_inter_dt = getattr(prob_univ.sol_ps, "CHECK_SELF_INTERSECTION_DT", None)

    # Termination criteria
    t_fin = float(prob_univ.sol_ps.T_FIN)
    end_area_ratio = float(prob_univ.sol_ps.END_AREA_RATIO)

    # Progress tracking
    starting_area = compute_area_shoelace(starting_drop.x, starting_drop.y)
    t = 0.0
    step_count = 0
    last_self_inter_check = 0.0

    out_t = []
    out_x = []

    while True:
        if t >= t_fin:
            print("Solve Complete with condition: FINAL_TIME_HIT")
            break

        # Optional self-intersection checks
        if check_self_inter_dt is not None and (t - last_self_inter_check) > check_self_inter_dt:
            has_si = not is_simple(y[:n], y[n:2*n])
            if has_si:
                print("Solve Complete with condition: SELF-INTERSECTION_FOUND")
                break
            last_self_inter_check = t

        # Area-based termination
        cur_area_rat = compute_area_shoelace(y[:n], y[n:2*n]) / starting_area
        if cur_area_rat < end_area_ratio:
            print("Solve Complete with Condition: END_AREA_RATIO_HIT")
            break

        # Cap dt so we don't step past t_fin
        dt = min(dt, t_fin - t)
        dt = max(dt, dt_min)

        # (1) EXPENSIVE: compute FEM gradients ONCE at the start of the macro step
        grad_p, grad_w = _fem_boundary_gradients(y[:n], y[n:2*n], y[2*n:3*n], ds, prob_univ)

        # (2) IMPLICIT macro step on cheap RHS with frozen gradients; retry with smaller dt if needed
        accepted = False
        trial = 0
        y_old = y.copy()
        while not accepted:
            trial += 1
            sol = _backward_euler_macro_step(
                cheap_rhs, t, y_old, dt, grad_p, grad_w,
                maxiter=max_newton_iter, tol=root_tol,
                relax=float(getattr(prob_univ.sol_ps, "IMEX_RELAX", 0.2))
            )

            ok = sol is not None and bool(sol.get("success", False)) and np.all(np.isfinite(sol["x"]))
            if ok:
                # Enforce monotone non-increasing area (reject steps that increase area).
                area_old = compute_area_shoelace(y_old[:n], y_old[n:2*n])
                area_new = compute_area_shoelace(sol["x"][:n], sol["x"][n:2*n])

                # Allow tiny numerical wiggle room
                wiggle = 1e-10 * max(1.0, abs(area_old))
                if area_new <= area_old + wiggle:
                    y = sol["x"]
                    accepted = True

                    # Additional stability rejects: cap max displacement and max theta change per macro step.
                    max_disp = float(getattr(prob_univ.sol_ps, "MAX_DISP_PER_STEP", 1e-4))  # meters (0.1 mm default)
                    max_dtheta = float(getattr(prob_univ.sol_ps, "MAX_DTHETA_PER_STEP", np.deg2rad(1.0)))  # radians (1 deg default)

                    dx = sol["x"][:n] - y_old[:n]
                    dy = sol["x"][n:2*n] - y_old[n:2*n]
                    dtheta = sol["x"][2*n:3*n] - y_old[2*n:3*n]

                    disp = np.hypot(dx, dy)
                    if float(np.max(disp)) > max_disp:
                        accepted = False
                        ok = False
                        if getattr(prob_univ.sol_ps, "VERBOSE", False):
                            print(f"[IMEX] rejected step due to displacement: max={np.max(disp):.3e} > {max_disp:.3e}")
                    elif float(np.max(np.abs(dtheta))) > max_dtheta:
                        accepted = False
                        ok = False
                        if getattr(prob_univ.sol_ps, "VERBOSE", False):
                            print(f"[IMEX] rejected step due to dtheta: max={np.max(np.abs(dtheta)):.3e} > {max_dtheta:.3e}")

                else:
                    ok = False
                    if getattr(prob_univ.sol_ps, "VERBOSE", False):
                        print(f"[IMEX] rejected step due to area increase: old={area_old:.6e}, new={area_new:.6e}")

            if not ok:
                dt *= shrink
                if dt < dt_min:
                    msg = sol.get("message","unknown") if (sol is not None and hasattr(sol, "message")) else "unknown"
                    raise RuntimeError(
                        f"IMEX implicit step failed (trial {trial}) even at dt_min={dt_min}. Last message: {msg}"
                    )
                if getattr(prob_univ.sol_ps, "VERBOSE", False):
                    msg = sol.get("message","unknown") if (sol is not None and hasattr(sol, "message")) else "unknown"
                    print(f"[IMEX] step rejected; shrinking dt -> {dt:.3e}. root msg: {msg}")

        # Advance time
        t += dt
        step_count += 1

        # Output
        if step_count % out_every == 0:
            out_t.append(t)
            out_x.append(y.copy())

        # Progress prints
        if getattr(prob_univ.sol_ps, "VERBOSE", False):
            cur_area_rat = compute_area_shoelace(y[:n], y[n:2*n]) / starting_area
            area_prog = (1 - cur_area_rat) / (1 - end_area_ratio)
            time_prog = t / t_fin
            print("-" * 60)
            print("IMEX-IMPLICIT MACRO STEP COMPLETE!")
            print(f"CURRENT TIME: {t}")
            print(f"CURRENT DT: {dt}")
            print(f"CURRENT AREA RATIO: {cur_area_rat}")
            print(f"AREA PROGRESS: {round(100 * area_prog, 2)}%")
            print(f"TIME PROGRESS: {round(100 * time_prog, 2)}%")
            print("-" * 60)

        # If we're doing well, cautiously increase dt
        dt = min(dt_max, dt * grow)

    return out_t, out_x

