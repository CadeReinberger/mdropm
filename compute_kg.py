import numpy as np
import droplet

def compute_kg_derivative(prob_univ, n = 10):
    # First, make the droplet with 4n points where it's just about to go
    drop = droplet.constructors.make_droplet_about_to_drop(n, 3, prob_univ.htscp, prob_univ.phys_ps)
    N = 4 * n

    # Next, we compute the point we care about
    n_star = n

    # Now, let's get some stuff from the droplet we'll need
    dr_x = drop.x
    dr_y = drop.y 
    dr_theta = drop.theta
    ds = drop.L / N
    sig, sig_p = compute_casr(prob_univ.phys_ps)
    k_w = prob_univ.phys_ps.D * prob_univ.phys_ps.c_g / prob_univ.phys_ps.c_l

    eval_pts = (n_star - 1, n_star, n_star + 1)
    
    # Okay, let's compute some derivatives

    # Compute the s-derivatives in the obvious way
    x_s = (dr_x[i+1] - dr_x[i-1]) / (2 * ds)
    y_s = (dr_y[i+1] - dr_y[i-1]) / (2 * ds)
    n_hat = np.array([y_s, -x_s]) / np.hypot(x_s, y_s)

    # Compute the current height
