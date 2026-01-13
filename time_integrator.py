from geo_util import Lambda, Lambda_pr, compute_casr, compute_area_shoelace, is_simple
from gas_fem import compute_concentration_gradients
from liquid_fem import compute_pressure_gradients
from scipy.integrate import solve_ivp
import numpy as np

def compute_deriv_function(starting_drop, prob_univ):
    
    # Compute a few global utils and such
    n = starting_drop.n
    ds = starting_drop.L / n
    sig, sig_p = compute_casr(prob_univ.phys_ps)
    k_p = 1 / (3 * prob_univ.phys_ps.mu)
    k_w = prob_univ.phys_ps.D * prob_univ.phys_ps.c_g / prob_univ.phys_ps.c_l

    # Let's make the function that will define our ODE
    def compute_deriv(t, state_vec):

        # We may need this but I don't think we do
        # nonlocal n, h, sig, sig_p, k_p, k_w

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
            x_s = (dr_x[ip1] - dr_x[im1]) / (2 * ds)
            y_s = (dr_y[ip1] - dr_y[im1]) / (2 * ds)
            n_hat = np.array([y_s, -x_s]) / np.hypot(x_s, y_s)

            # Next, compute our height and height derivatives
            h = prob_univ.htscp.h(x, y)
            hx = prob_univ.htscp.hx(x, y)
            hy = prob_univ.htscp.hy(x, y)
            grad_h = np.array([hx, hy])

            # Now, compute our psi
            psi = np.arctan(np.dot(n_hat, grad_h))

            # Now compute our pressure field
            p = -prob_univ.phys_ps.gamma * np.cos(theta + psi) / h

            # Add it to our pressure array
            p_arr[i] = p

        # Now, we use the liquid and gase phase fem solvers to get gradients
        grad_p = compute_pressure_gradients(dr_x, dr_y, p_arr, prob_univ)
        grad_w = compute_concentration_gradients(dr_x, dr_y, prob_univ)

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
            x_s = (dr_x[ip1] - dr_x[im1]) / (2 * ds)
            y_s = (dr_y[ip1] - dr_y[im1]) / (2 * ds)
            s_norm = np.hypot(x_s, y_s)
            theta_s = (dr_theta[ip1] - dr_theta[im1]) / (2 * ds)

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
            dh_dn = np.dot(n_hat, grad_h)

            # Now compute our second derivatives in s
            x_ss = (dr_x[ip1] - 2*dr_x[i] + dr_x[im1]) / (ds**2)
            y_ss = (dr_y[ip1] - 2*dr_y[i] + dr_y[im1]) / (ds**2)

            # Now compute our mixed partials from the PDE
            x_st_one = y_s * sig_p_theta * theta_s / s_norm
            x_st_two = (y_ss*x_s**2 - x_s*y_s*x_ss) * sig_theta / (s_norm**3)
            x_st = x_st_one + x_st_two
            y_st_one = -x_s * sig_p_theta * theta_s / s_norm
            y_st_two = (x_s*y_s*y_ss - x_ss*y_s**2) * sig_theta / (s_norm**3)
            y_st = y_st_one + y_st_two 

            # Now we compute dn_dt 
            dn_dt_x = (y_st*x_s**2 - y_s*x_s*x_st) / (s_norm**3)
            dn_dt_y = (x_s*y_s*y_st - x_st*y_s**2) / (s_norm**3)
            dn_dt = np.array([dn_dt_x, dn_dt_y])

            # Compute another term
            n_dot_d_dt_grad_h = sig_theta * np.dot(n_hat, hess_h @ n_hat)

            # Compute psi and its derivative
            psi_dot = (np.dot(dn_dt, grad_h) + n_dot_d_dt_grad_h) / np.hypot(1, dh_dn)**2
            psi = np.arctan(dh_dn)

            # Compute the normal derivatives we'll need for our rhs
            dw_dn = np.dot(grad_w[i], n_hat)
            dp_dn = np.dot(grad_p[i], n_hat)

            # Compute the RHS we'll need in a bit
            rhs = 2*h*(k_p*dp_dn*h**2 + sig_theta) - 2*h*k_w*dw_dn
            
            # Now compute the theta_t we'll need
            theta_t_two_num = rhs - 2 * h * Lambda(theta+psi) * sig_theta * dh_dn
            theta_t_two_denom = h * h * Lambda_pr(theta + psi)
            theta_t_two = theta_t_two_num / theta_t_two_denom
            theta_t = -psi_dot + theta_t_two

            # Finally, add the theta_dot that we want here
            dr_theta_t[i] = theta_t

        print(f'dr_theta_t: {dr_theta_t}')
        # Now we compute the derivative of our state vector
        st_vec_deriv = np.zeros(3*n)
        st_vec_deriv[:n] = dr_x_t
        st_vec_deriv[n:2*n] = dr_y_t
        st_vec_deriv[2*n:3*n] = dr_theta_t

        # Add in a time print so we have some progress idea here
        if prob_univ.sol_ps.VERBOSE:
            print('-' * 60 + f'\nSOLVING @ TIME: {t}\n' + '-' * 60 + '\n')

        return st_vec_deriv

    return compute_deriv

def solve_problem(starting_drop, prob_univ):
    
    # First, make the derivative function that we're gonna need
    f = compute_deriv_function(starting_drop, prob_univ)

    # Now make the x0 that we're gonna use
    n = starting_drop.n
    x0 = np.zeros(3*n)
    x0[:n] = starting_drop.x
    x0[n:2*n] = starting_drop.y
    x0[2*n:3*n] = starting_drop.theta

    # Initialize our loop variables
    cur_t = 0
    cur_x = x0

    # Get our radau stuff plz
    radau_dt = prob_univ.sol_ps.RADAU_DT
    subdiv_radau = prob_univ.sol_ps.SUBDIV_RADAU
    radau_eval_linsp_n = prob_univ.sol_ps.RADAU_EVAL_LINSPACE_N
    radau_out_every = prob_univ.sol_ps.RADAU_OUT_EVERY
    check_self_inter_dt = prob_univ.sol_ps.CHECK_SELF_INTERSECTION_DT

    # Intialize our checking variables
    last_self_inter_check = 0
    radau_iter_count = 0
    starting_area = compute_area_shoelace(starting_drop.x, starting_drop.y)

    # Initialize our ouput things
    out_t = []
    out_x = []

    while True:

        # Check if we're over our timestep
        if cur_t > prob_univ.sol_ps.T_FIN:
            print('Solve Complete with condition: FINAL_TIME_HIT')
            break 

        # If it's time, check for a self intersection
        if check_self_inter_dt is not None:           
            if cur_t - last_self_inter_check > check_self_inter_dt:
                has_si = not is_simple(cur_x[:n], cur_x[n:2*n])
                if has_si:
                    print('Solve Complete with condition: SELF-INTERSECTION_FOUND')
                    break
                last_self_inter_check = cur_t

        # We're not done, let's prep the currennt radau step
        new_t = cur_t + radau_dt
        cur_t_eval = [new_t]
        if subdiv_radau:
            cur_t_eval = np.linspace(cur_t, new_t, num=radau_eval_linsp_n)[1:]

        sol = solve_ivp(f, (cur_t, cur_t + radau_dt), cur_x, t_eval=cur_t_eval, method='LSODA')
        cur_t, cur_x = sol.t[-1], sol.y[:, -1]
        radau_iter_count += 1

        # Now we add this to our output
        if subdiv_radau:
            print(f'sol_y: {sol.y}')
            out_t.extend(sol.t)
            out_x.extend([sol.y[:, i] for i in range(radau_eval_linsp_n-1)])
        else:
            if radau_iter_count % radau_out_every == 0:
                out_t.append(cur_t)
                out_x.append(cur_x)

        # Let's compute our progress to make some useful output here
        cur_area_rat = compute_area_shoelace(cur_x[:n], cur_x[n:2*n]) / starting_area
        area_prog = (1 - cur_area_rat) / (1 - prob_univ.sol_ps.END_AREA_RATIO)
        time_prog = cur_t / prob_univ.sol_ps.T_FIN

        # If we hit our area target, call it!
        if cur_area_rat < prob_univ.sol_ps.END_AREA_RATIO:
            print('Solve Complete with Condition: END_AREA_RATIO_HIT')
            break

        # If we're not done and verbose, let's progress print
        if prob_univ.sol_ps.VERBOSE:
            print('-' * 60 + '\n' + '-' * 60 + '\n\n')
            print('RADAU STEP COMPLETE!')
            print(f'CURRENT TIME: {cur_t}')
            print(f'CURRENT AREA RATIO: {cur_area_rat}')
            print(f'AREA PROGRESS: {round(100*area_prog, 2)}%')
            print(f'TIME PROGRESS: {round(100*time_prog, 2)}%\n')
            print('-' * 60 + '\n' + '-' * 60 + '\n\n')

    # We've broken out of the solver loop, so we're done
    return out_t, out_x


def solve_problem_rk4(starting_drop, prob_univ):
    # First, make the derivative function that we're gonna need
    f = compute_deriv_function(starting_drop, prob_univ)

    # Now make the x0 that we're gonna use
    n = starting_drop.n
    x0 = np.zeros(3*n)
    x0[:n] = starting_drop.x
    x0[n:2*n] = starting_drop.y
    x0[2*n:3*n] = starting_drop.theta

    # Initialize our loop variables
    cur_t = 0
    cur_x = x0

    # Get our radau stuff plz
    radau_dt = prob_univ.sol_ps.RADAU_DT
    radau_out_every = prob_univ.sol_ps.RADAU_OUT_EVERY
    check_self_inter_dt = prob_univ.sol_ps.CHECK_SELF_INTERSECTION_DT

    # Intialize our checking variables
    last_self_inter_check = 0
    radau_iter_count = 0
    starting_area = compute_area_shoelace(starting_drop.x, starting_drop.y)

    # Initialize our ouput things
    out_t = []
    out_x = []

    while True:

        # Check if we're over our timestep
        if cur_t > prob_univ.sol_ps.T_FIN:
            print('Solve Complete with condition: FINAL_TIME_HIT')
            break

        # If it's time, check for a self intersection
        if check_self_inter_dt is not None:
            if cur_t - last_self_inter_check > check_self_inter_dt:
                has_si = not is_simple(cur_x[:n], cur_x[n:2*n])
                if has_si:
                    print('Solve Complete with condition: SELF-INTERSECTION_FOUND')
                    break
                last_self_inter_check = cur_t

        # We're not done, let's do the RK4 step
        # Note that the t doesn't matter, that's why we don't update it
        k1 = radau_dt*f(cur_t, cur_x)
        k2 = radau_dt*f(cur_t, cur_x+.5*k1)
        k3 = radau_dt*f(cur_t, cur_x+.5*k2)
        k4 = radau_dt*f(cur_t, cur_x+k3)

        # Update our output here
        cur_t, cur_x = cur_t + radau_dt, cur_x + (k1+2*k2+2*k3+k4)/6
        radau_iter_count += 1

        if radau_iter_count % radau_out_every == 0:
            out_t.append(cur_t)
            out_x.append(cur_x)

        # Let's compute our progress to make some useful output here
        cur_area_rat = compute_area_shoelace(cur_x[:n], cur_x[n:2*n]) / starting_area
        area_prog = (1 - cur_area_rat) / (1 - prob_univ.sol_ps.END_AREA_RATIO)
        time_prog = cur_t / prob_univ.sol_ps.T_FIN

        # If we hit our area target, call it!
        if cur_area_rat < prob_univ.sol_ps.END_AREA_RATIO:
            print('Solve Complete with Condition: END_AREA_RATIO_HIT')
            break

        # If we're not done and verbose, let's progress print
        if prob_univ.sol_ps.VERBOSE:
            print('-' * 60 + '\n' + '-' * 60 + '\n\n')
            print('RADAU STEP COMPLETE!')
            print(f'CURRENT TIME: {cur_t}')
            print(f'CURRENT AREA RATIO: {cur_area_rat}')
            print(f'AREA PROGRESS: {round(100*area_prog, 2)}%')
            print(f'TIME PROGRESS: {round(100*time_prog, 2)}%\n')
            print('-' * 60 + '\n' + '-' * 60 + '\n\n')

    # We've broken out of the solver loop, so we're done
    return out_t, out_x
