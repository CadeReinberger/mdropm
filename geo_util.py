import numpy as np
from shapely.geometry import LineString, Polygon
from itertools import combinations

def compute_area_shoelace(x, y):
    return .5 * np.abs(sum(x*np.roll(y, 1) - y*np.roll(x, 1)))

def min_distance_within(x, y):
    return min(np.hypot(x[i]-x[j], y[i]-y[j]) for i in range(len(x)) for j in range(len(y)) if i != j)

def min_distance_between(x1, y1, x2, y2):
    return min(np.hypot(x1[i]-x2[j], y1[i]-y2[j]) for i in range(len(x1)) for j in range(len(x2)))

def min_distance_neighbors(x, y):
    return min(np.hypot(x[(i+1)%len(x)]-x[i], y[(i+1)%len(x)]-y[i]) for i in range(len(x)))

def gas_lc(drop_x, drop_y, ts, sps):
    if not sps.GAS_PHASE_DYNAMIC_LC:
        return sps.GAS_PHASE_DEFAULT_LC

    # Get the tapescape to a points list
    ext_x = np.array([seg.start_pt[0] for seg in ts.segments])
    ext_y = np.array([seg.start_pt[1] for seg in ts.segments])

    # First compute the distance along the slide
    d_sigma = min_distance_within(ext_x, ext_y)
    
    # Next, compute the distance between the slide and the drop
    d_mu = min_distance_between(drop_x, drop_y, ext_x, ext_y)

    # Next, compute the distance along the droplet itself
    d_delta = min_distance_neighbors(drop_x, drop_y)

    # Now, compute and return the minimum LC
    (k_sigma, k_mu, k_delta) = sps.GAS_PHASE_LC_KS

    # Now we compute the result that we want
    lc = np.inf

    if sps.GAS_PHASE_DEFAULT_LC is not None:
        lc = sps.GAS_PHASE_DEFAULT_LC

    if k_sigma is not None:
        lc = min(lc, k_sigma * d_sigma)
    
    if k_mu is not None:
        lc = min(lc, k_mu * d_mu)

    if k_delta is not None:
        lc = min(lc, k_delta * d_delta)

    # And we've got it all, return it
    return lc

def liquid_lc(drop_x, drop_y, sps):
    lc = np.inf

    if sps.LIQUID_PHASE_DEFAULT_LC is not None:
        lc = sps.LIQUID_PHASE_DEFAULT_LC

    if sps.LIQUID_PHASE_DYNAMIC_LC_K is not None:
        lc = min(lc, min_distance_neighbors(drop_x, drop_y) * sps.LIQUID_PHASE_DYNAMIC_LC_K)

    return lc 

def make_polygon_projector(x, y, u):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    n = len(x)

    # Segment start/end points (A[i] -> B[i] = A[(i+1)%n])
    ax, ay = x, y
    bx = np.roll(x, -1)
    by = np.roll(y, -1)

    # Segment direction vectors, squared lengths, and cumulative arc-length
    sdx = bx - ax                            # shape (n,)
    sdy = by - ay
    seg_len_sq = sdx*sdx + sdy*sdy          # shape (n,)
    seg_len    = np.sqrt(seg_len_sq)        # shape (n,)
    cum_len    = np.r_[0.0, np.cumsum(seg_len)]  # shape (n+1,)

    # u value at the end vertex of each segment, pre-indexed
    u_next = u[(np.arange(n) + 1) % n]

    def query(px, py):
        """Accept scalar or 1-D array of query points; return u values in same shape."""
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)
        scalar = px.ndim == 0
        px = np.atleast_1d(px)
        py = np.atleast_1d(py)

        # (N, n): vector from each segment start to each query point
        ex = px[:, None] - ax[None, :]
        ey = py[:, None] - ay[None, :]

        # Scalar projection onto segment direction, clamped to [0, 1]
        t = (ex * sdx[None, :] + ey * sdy[None, :]) / seg_len_sq[None, :]
        t = np.clip(t, 0.0, 1.0)                                # (N, n)

        # Squared distance to closest point on each segment
        dist_sq = (px[:, None] - (ax[None, :] + t * sdx[None, :]))**2 \
                + (py[:, None] - (ay[None, :] + t * sdy[None, :]))**2  # (N, n)

        # Best segment and t for each query point
        seg_idx = np.argmin(dist_sq, axis=1)            # (N,)
        t_best  = t[np.arange(len(px)), seg_idx]        # (N,)

        # Linear interpolation of u along the winning segment
        u_vals = u[seg_idx] + t_best * (u_next[seg_idx] - u[seg_idx])  # (N,)

        return u_vals[0] if scalar else u_vals

    return query


def Lambda(theta):
    if np.isclose(theta, np.pi/2):
        return 0
    return (.5*np.pi - theta) / np.cos(theta)**2 - np.tan(theta)

def Lambda_pr(theta):
    if np.isclose(theta, np.pi/2):
        return -2/3
    return ((np.pi - 2*theta)*np.tan(theta) - 2) / np.cos(theta)**2 

def compute_casr(pps):
    # First, get the constants from the pps dataclass
    theta_a, theta_r = pps.theta_a, pps.theta_r
    k_a, k_r = pps.k_a, pps.k_r

    # Make the CASR (linear assumption here)
    def sigma(theta):
        if theta > theta_a:
            return k_a * (theta - theta_a)
        if theta < theta_r:
            return k_r * (theta - theta_r)
        else:
            return 0

    def sigma_prime(theta):
        if theta > theta_a: 
            return k_a
        if theta < theta_r: 
            return k_r
        else:
            return 0

    return sigma, sigma_prime

def is_simple(x, y):
    # First, make all the line segments we want
    all_segments = []
    n = len(x)
    for i in range(n):
        xi, yi = x[i], y[i]
        ip1 = (i+1)%n
        xip1, yip1 = x[ip1], y[ip1]
        seg = LineString([(xi, yi), (xip1, yip1)])
        all_segments.append(seg)

    # Next, we iterate over them all and see if we have any nontrivial intersections
    for (i1, i2) in combinations(range(n), 2):
        if i2 == ((i1+1)%n) or i1 == ((i2+1)%n):
            # They're neighbors, we don't care about their self-intersection
            continue
        seg1, seg2 = all_segments[i1], all_segments[i2]
        if seg1.intersects(seg2):
            return False # We have a self-intersection

    return True
