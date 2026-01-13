import numpy as np
from shapely.geometry import Point, LinearRing

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
        lc = min(lc, min_dist_neighbors(drop_x, drop_y) * sps.LIQUID_PHASE_DYNAMIC_LC_K)

    return lc 

def make_polygon_projector(x, y, u):
    # Make the Linear Ring so Shapely knows wassup
    coords = list(zip(x,y))
    ring = LinearRing(coords)

    # Make the segment lengths and cum arclength
    x_circ = np.r_[x, x[0]]
    y_circ = np.r_[y, y[0]]
    seg_len = np.hypot(np.diff(x_circ), np.diff(y_circ))
    cum_len = np.r_[0, np.cumsum(seg_len)]

    def query(px, py):
        P = Point(px, py)

        # Distance to closest point
        s = ring.project(p)

        # Find the segment that we want
        i = int(np.searchsorted(cum, s, side='right')-1)
        if i == n:
            raise Exception('What the fuck?')

        # Compute the linear interpolation
        u_val = np.interp(s, (cum[i], cum[i+1]), (u[i], u[(i+1)%len(u)]))
        
        # Return the result
        return u_val

    # Return the querying function
    return query
