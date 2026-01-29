import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from scipy.optimize import minimize_scalar, brentq


@dataclass
class Params:
    A: float
    B: float
    R: float
    theta_r: float  # radians
    # numerical controls
    s_grid: int = 4000
    refine_window: int = 10          # number of grid steps on each side to bracket local refine
    p_scan: int = 400                # number of p samples for bracketing g(p)=0
    tol_theta: float = 1e-10
    tol_p: float = 1e-12


def h_on_circle(s: np.ndarray, A: float, B: float, R: float) -> np.ndarray:
    return A + B * R * np.sin(s)


def psi(s: np.ndarray, B: float) -> np.ndarray:
    return np.arctan(B * np.sin(s))


def theta_of_s_p(s: np.ndarray, p: float, A: float, B: float, R: float) -> np.ndarray:
    """
    theta(s) = -psi(s) + arccos(-p * h(gamma(s))).
    Assumes feasibility |p*h|<=1 holds; otherwise arccos will get NaNs if not clipped.
    """
    arg = -p * h_on_circle(s, A, B, R)
    # Do NOT clip here: we want infeasible p to show up as invalid.
    return -psi(s, B) + np.arccos(arg)


def feasible_p_interval(A: float, B: float, R: float) -> Tuple[float, float]:
    """
    Enforce |p * h(s)| <= 1 for all s.
    h(s) ranges over [A - B R, A + B R] since sin(s) ∈ [-1,1].
    So max_{s} |h(s)| = max(|A - BR|, |A + BR|).
    Then |p| <= 1 / maxabs, unless maxabs=0 (then h(s)=0 for all s and any p is feasible).
    """
    hmin = A - B * R
    hmax = A + B * R
    maxabs = max(abs(hmin), abs(hmax))
    if maxabs == 0.0:
        return (-np.inf, np.inf)
    pmax = 1.0 / maxabs
    return (-pmax, pmax)


def theta_min_for_p(p: float, par: Params) -> float:
    """
    Compute min_{s ∈ [-pi, pi]} theta(s,p) robustly:
      1) coarse grid search
      2) local refine around each detected candidate minimum (including endpoints)
    """
    A, B, R = par.A, par.B, par.R

    # Coarse grid (include endpoints)
    s = np.linspace(-np.pi, np.pi, par.s_grid, endpoint=True)
    th = theta_of_s_p(s, p, A, B, R)

    if np.any(~np.isfinite(th)):
        # infeasible p (violates arccos domain)
        return np.nan

    # Find candidate minima indices: local minima on grid + global min index
    # Local minima: th[i] <= th[i-1] and th[i] <= th[i+1]
    cand = []
    for i in range(1, len(s) - 1):
        if th[i] <= th[i - 1] and th[i] <= th[i + 1]:
            cand.append(i)
    cand.append(int(np.argmin(th)))
    cand = sorted(set(cand))

    # Always include endpoints as candidates (because min might occur at boundary)
    cand.extend([0, len(s) - 1])
    cand = sorted(set(cand))

    ds = s[1] - s[0]
    best = float("inf")

    def obj(x):
        return float(theta_of_s_p(np.array([x]), p, A, B, R)[0])

    for i in cand:
        left_i = max(0, i - par.refine_window)
        right_i = min(len(s) - 1, i + par.refine_window)
        a = s[left_i]
        b = s[right_i]

        # Ensure proper ordering
        if b <= a + 1e-15:
            best = min(best, th[i])
            continue

        res = minimize_scalar(obj, bounds=(a, b), method="bounded", options={"xatol": 1e-12})
        if res.success and np.isfinite(res.fun):
            best = min(best, float(res.fun))
        else:
            best = min(best, th[i])

    return best


def g_of_p(p: float, par: Params) -> float:
    """g(p) = min_s theta(s,p) - theta_r"""
    tmin = theta_min_for_p(p, par)
    if not np.isfinite(tmin):
        return np.nan
    return tmin - par.theta_r


def solve_p_star(par: Params) -> Tuple[Optional[float], dict]:
    """
    Returns (p_star, info).
    If multiple solutions exist, returns the one with smallest |p| among found roots.
    """
    p_lo, p_hi = feasible_p_interval(par.A, par.B, par.R)

    # Handle degenerate "any p is feasible" case (only happens if maxabs==0, i.e. A=0 and B*R=0)
    if not np.isfinite(p_lo) and not np.isfinite(p_hi):
        # Here h(gamma(s)) ≡ 0, so arccos(-p*h)=arccos(0)=pi/2 independent of p.
        # theta(s)= -atan(B sin s)+pi/2. If B=0 too, theta ≡ pi/2.
        # We can just check whether min matches theta_r.
        s = np.linspace(-np.pi, np.pi, par.s_grid, endpoint=True)
        th = -psi(s, par.B) + (np.pi / 2.0)
        tmin = float(np.min(th))
        if abs(tmin - par.theta_r) <= par.tol_theta:
            return 0.0, {"status": "degenerate_h_zero_any_p", "tmin": tmin, "p_interval": (p_lo, p_hi)}
        return None, {"status": "no_solution_deg_h_zero", "tmin": tmin, "p_interval": (p_lo, p_hi)}

    # Slightly shrink interval to avoid numerical issues right at |p*h|=1
    # (arccos derivative blows up there).
    shrink = 1e-12
    p_lo2 = p_lo * (1.0 - shrink)
    p_hi2 = p_hi * (1.0 - shrink)

    # Scan for sign changes of g(p)
    ps = np.linspace(p_lo2, p_hi2, par.p_scan, endpoint=True)
    gs = []
    for p in ps:
        val = g_of_p(float(p), par)
        gs.append(val)
    gs = np.array(gs, dtype=float)

    # Filter NaNs (should be none if feasibility interval is correct, but keep safe)
    finite = np.isfinite(gs)
    ps_f = ps[finite]
    gs_f = gs[finite]

    info = {
        "status": None,
        "p_interval": (p_lo, p_hi),
        "p_interval_used": (float(p_lo2), float(p_hi2)),
        "scan_points": len(ps),
        "roots_found": [],
    }

    if len(ps_f) < 2:
        info["status"] = "scan_failed_all_nan"
        return None, info

    # Collect brackets where g changes sign or hits near-zero
    brackets: List[Tuple[float, float]] = []
    for i in range(len(ps_f) - 1):
        a, b = float(ps_f[i]), float(ps_f[i + 1])
        ga, gb = float(gs_f[i]), float(gs_f[i + 1])

        if abs(ga) <= par.tol_theta:
            brackets.append((a, a))
            continue
        if ga == 0.0:
            brackets.append((a, a))
            continue
        if np.sign(ga) != np.sign(gb):
            brackets.append((a, b))

    # Deduplicate near-identical brackets
    uniq = []
    for br in brackets:
        if not uniq:
            uniq.append(br)
            continue
        if abs(br[0] - uniq[-1][0]) < 1e-10 and abs(br[1] - uniq[-1][1]) < 1e-10:
            continue
        uniq.append(br)
    brackets = uniq

    if not brackets:
        info["status"] = "no_bracket_found"
        # Provide diagnostics: min/max of g on scan
        info["g_min_scan"] = float(np.nanmin(gs_f))
        info["g_max_scan"] = float(np.nanmax(gs_f))
        return None, info

    roots = []
    for a, b in brackets:
        if a == b:
            p0 = a
            # polish p0 with a tiny local search, but it might already be fine
            roots.append(p0)
            continue

        def gg(x):
            return g_of_p(float(x), par)

        try:
            r = brentq(gg, a, b, xtol=par.tol_p, rtol=par.tol_p, maxiter=200)
            roots.append(float(r))
        except ValueError:
            # bracket might be bad due to numerical noise; skip
            continue

    # Validate roots: must satisfy feasibility and min condition
    valid = []
    for r in roots:
        # feasibility check (strict)
        s = np.linspace(-np.pi, np.pi, 5000, endpoint=True)
        hh = h_on_circle(s, par.A, par.B, par.R)
        if np.max(np.abs(r * hh)) > 1.0 + 1e-10:
            continue

        tmin = theta_min_for_p(r, par)
        if not np.isfinite(tmin):
            continue

        if abs(tmin - par.theta_r) <= 5e-8:
            valid.append((r, tmin))

    info["roots_found"] = valid

    if not valid:
        info["status"] = "roots_found_but_failed_validation"
        return None, info

    # Choose the solution with smallest |p| (you can change this policy)
    valid.sort(key=lambda rt: abs(rt[0]))
    p_star = float(valid[0][0])
    info["status"] = "success"
    info["chosen_tmin"] = float(valid[0][1])

    return p_star, info

def compute_p_star_over_gamma(hs, HL, R, ps):
    A = hs.h(0, 0)
    B = (hs.h(0, HL) - A) / HL
    par = Params(A=A, B=B, R=R, theta_r=ps.theta_r)
    p_star, info = solve_p_star(par)
    return p_star


if __name__ == "__main__":
    # -------------------------
    # USER INPUTS (edit these)
    # -------------------------
    A = .15
    B = .5/5         # >= 0
    R = 3.0         # > 0
    theta_r = np.deg2rad(50)   # radians

    par = Params(A=A, B=B, R=R, theta_r=theta_r)

    p_star, info = solve_p_star(par)

    print("=== RESULT ===")
    print("p_star:", p_star)
    print("status:", info.get("status"))
    print("feasible p interval:", info.get("p_interval"))
    if info.get("status") == "success":
        print("min theta at p_star:", info.get("chosen_tmin"))
        # quick sanity check:
        s = np.linspace(-np.pi, np.pi, 8000, endpoint=True)
        theta_vals = theta_of_s_p(s, p_star, A, B, R)
        print("min theta (grid check):", float(np.min(theta_vals)))
        print("max |p*h| (grid check):", float(np.max(np.abs(p_star * h_on_circle(s, A, B, R)))))
    else:
        # diagnostics
        for k, v in info.items():
            if k not in {"roots_found"}:
                print(f"{k}: {v}")
        print("roots_found (validated):", info.get("roots_found"))

