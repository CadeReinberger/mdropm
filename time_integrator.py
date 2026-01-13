from geo_util import Lambda, Lambda_pr
from gas_fem import compute_concentration_gradients
from liquid_fem import compute_pressure_gradients

def compute_deriv_function(starting_drop, prob_univ):
    
    # Compute a few global utils and such
    n = starting_drop.n
    h = starting_drop.L / n
    sig = prob_univ.phys_ps.sigma
    sig_p = prob_univ.phys_ps.sigma_prime
    k_p = 1 / (3 * prob_univ.phys_ps.mu)
    k_w = prob_univ.phys_ps.D * prob_univ.phys_ps.c_g / prob_univ.phys_ps.c_l

    # Let's make the function that will define our ODE
    def compute_deriv(t, state_vec):

        # First, let's get the x, y, p we're gonna use
        n = len(state_vec)//3
        dr_x = state_vec[:n]
        dr_y = state_vec[n:2*n]
        dr_theta = state_vec[2*n:3*n]

        # Initialize some arrays to do our Young-Laplace thing
        p_arr = np.zeros(n)
        
        # Iterate through and compute our resulting pressures
        for i in range(n):
            # First, get i+1 and i-1 circularly
            ip1 = (i+1)%n
            im1 = i-1

            # Get the point we care about here
            x = dr_x[i]
            y = dr_y[i]
            theta = dr_theta[i]

            # Compute the s-derivatives in the obvious way
            x_s = (x[ip1] - x[im1]) / (2 * h)
            y_s = (y[ip1] - y[im1]) / (2 * h)
            n_hat = np.array([y_s, -x_s]) / np.hypot(x_s, y_s)

            # Next, compute our height and height derivatives
            h = prob_univ.htscp.h(x, y)
            hx = prob_univ.htscp.hx(x, y)
            hy = prob_univ.htscp.hy(x, y)
            grad_h = np.array([hx, hy])

            # Now, compute our psi
            psi = np.arctan(np.linalg.dot(n_hat, grad_h))

            # Now compute our pressure field
            p = -prob_univ.phys_ps.gamma * np.cos(theta + psi) / h

            # Add it to our pressure array
            p_arr[i] = p

        # Now, we use the liquid and gase phase fem solvers to get gradients
        grad_p = compute_pressure_gradients(dr_x, dr_y, p_arr, prob_univ)
        grad_w = compute_pressure_gradients(dr_x, dr_y, prob_univ)

        # Now, we initialize some arrays to store our derivatives
        dr_x_t = np.zeros(n)
        dr_y_t = np.zeros(n)
        dr_theta_t = np.zeros(n)

        # Iterate one-by-one (Maybe Vectorize one day) to compute our derivatives
        for i in range(n):
            # First, get i+1 and i-1 circularly
            ip1 = (i+1)%n
            im1 = i-1

            # Get the point we care about here
            x = dr_x[i]
            y = dr_y[i]
            theta = dr_theta[i]

            # Compute the s-derivatives in the obvious way
            x_s = (x[ip1] - x[im1]) / (2 * h)
            y_s = (y[ip1] - y[im1]) / (2 * h)
            s_norm = np.hypot(x_s, y_s)

            # Compute sigma and sigma prime, which we'll need
            sig_theta = sig(theta)
            sig_p_theta = sig_p(theta)

            # Compute x_t and y_t here directly
            x_t = y_s * sig_theta / s_norm
            y_t = -x_s * sig_theta / s_norm

            # Add x_t and y_t to our stored vectors
            dr_x_t[i] = x_t
            dr_y_t[i] = y_t

            # Make the normal vector we're gonna want
            n_hat = np.array([y_s, -x_s]) / s_norm

            # Now, compute all the h-derivative we're gonna need in time
            h = prob_univ.htscp.h(x, y)
            hx = prob_univ.htscp.hx(x, y)
            hy = prob_univ.htscp.hy(x, y)
            hxx = prob_univ.htscp.hxx(x, y)
            hxy = prob_univ.htscp.hxy(x, y)
            hyy = prob_univ.htscp.hyy(x, y)
            grad_h = np.array([hx, hy])
            hess_h = np.array([[hxx, hxy], [hxy, hyy]])
            dh_dn = np.linalg.dot(n_hat, grad(h))

            # Now compute our second derivatives in s
            x_ss = (x[ip1] - 2*x[i] + x[im1]) / (h*h)
            y_ss = (y[ip1] - 2*y[i] + y[im1]) / (h*h)

            # Now compute our mixed partials from the PDE
            x_st_one = y_s * sig_p_theta * theta_s / s_norm
            x_st_two = (y_ss*x_s**2 - x_s*y_s*x_ss) * sig_theta / (s_norm**3)
            x_st = x_st_one + x_st_two
            y_st_one = -x_s * sig_p_theta * theta_s / s_norm
            y_st_two = (x_s*y_s*y_ss - x_ss*y_s**2) * sig_theta / (s_norm**3)
            y_st = y_st_one + y_st_two 

            # Now we compute dn_dt 
            dn_dt_x = (y_st*x_s**2 - y_s*x_s*x_st) / (s_norm**3)
            dn_dt_y = (x_s*y_s_y_st - x_st*y_s**2) / (s_norm**3)
            dn_dt = np.array([dn_dt_x, dn_dt_y])

            # Compute another term
            n_dot_d_dt_grad_h = sig_theta * np.linalg.dot(n_hat, hess_h @ n_hat)

            # Compute psi and its derivative
            psi_dot = (np.linalg.dot(dn_dt, grad_h) + n_dot_d_dt_grad_h) / np.hypot(1, dh_dn)**2
            psi = np.arctan(dh/dn)

            # Compute the normal derivatives we'll need for our rhs
            dw_dn = np.linalg.dot(grad_w[i], n_hat)
            dp_dn = np.linalg.dot(grad_p[i], n_hat)

            # Compute the RHS we'll need in a bit
            rhs = 2*h*(k_p*dp_dn*h**2 + sig_theta) - 2*h*k_w*dw_dn
            
            # Now compute the theta_t we'll need
            theta_t_two_num = rhs - 2 * h * Lambda(theta+psi) * sig_theta * dh_dn
            theta_t_two_denom = h * h * Lambda_pr(theta + psi)
            theta_t_two = theta_t_two_num / theta_t_two_denom
            theta_t = -psi_dot + theta_t_two

            # Finally, add the theta_dot that we want here
            dr_theta_t[i] = theta_t

        # Now we compute the derivative of our state vector
        st_vec_deriv = np.zeros(3*n)
        st_vec_deriv[:n] = dr_x_t
        st_vec_deriv[n:2*n] = dr_y_t
        st_vec_deriv[2*n:3*n] = dr_theta_t

        return st_vec_deriv

    return compute_deriv


