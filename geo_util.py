import numpy as np

def compute_area_shoelace(x, y):
    return .5 * np.abs(sum(x*np.roll(y, 1) - y*np.roll(x, 1)))

def min_dist_within(x, y):
    return min(np.hypot(x[i]-x[j], y[i]-y[j]) for i in range(len(x)) for j in range(len(y)) if i != j)

def min_distance_between(x1, y1, x2, y2):
    return min(np.hypot(x1[i]-x2[j], y1[i]-y2[j]) for i in range(len(x1)) for j in range(len(x2)))

def min_distance_neighbors(x, y):
    return min(np.hypot(x[(i+1)%len(x)]-x[i], y[(i+1)%len(x)]-y[i]) for i in range(len(x)))

def gas_lc(drop_x, drop_y, ext_x, ext_y, lc_ks, def_lc, sps):
    if not sps.GAS_PHASE_DYNAMIC_LC:
        return sps.GAS_PHASE_DEFAULT_LC

    # First compute the distance along the slide
    d_sigma = min_dist_within(ext_x, ext_y)
    
    # Next, compute the distance between the slide and the drop
    d_mu = min_distance_between(drop_x, drop_y, ext_x, ext_y)

    # Next, compute the distance along the droplet itself
    d_delta = min_dist_neighbors(drop_x, drop_y)

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
        lc = sps.LIQUID_PHASE_DEFAULT_LC)

    if sps.LIQUID_PHASE_DYNAMIC_LC_K is not None:
        lc = min(lc, min_dist_neighbors(drop_x, drop_y) * sps.LIQUID_PHASE_DYNAMIC_LC_K)

    return lc 
