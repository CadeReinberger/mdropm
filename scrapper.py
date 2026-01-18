import heightscape
import tapescape
import physical_params
import solver_params
import droplet
import numpy as np
from time_integrator import solve_problem, solve_problem_rk4
from problem_universe import problem_universe
from matplotlib import pyplot as plt
from pickler import pickle_output

def diagnostic_plot(out_t, out_x):
    LABEL_DIGS=5
    n = len(out_x[0])//3
    plt.figure()
    for ind in range(len(out_t)):
        plt.plot(out_x[ind][:n], out_x[ind][n:2*n], label=f't={round(out_t[ind], LABEL_DIGS)}')
    plt.legend()
    plt.figure()
    s = np.linspace(0, 2*np.pi, num=n)
    for ind in range(len(out_t)):
        plt.plot(s, out_x[ind][2*n:], label=f't={round(out_t[ind], LABEL_DIGS)}')
    plt.legend()
    plt.show()


# Make the heightscape
# hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))
hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))

# Make the tapescape
ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

# Make the Physical Params
pps = physical_params.physical_params()

# Make the solver params
sps = solver_params.solver_params()

# Make the problem universe
pu = problem_universe(hs, ts, sps, pps) 

# Make the Starting Droplet
drop = droplet.constructors.make_circular_flat_drop(20, 3, hs)

# Solve the time integration problem
out_t, out_x = solve_problem(drop, pu)
print('Well We Did Something!')

# Plot the result to get a feel
diagnostic_plot(out_t, out_x)

# Pickle the result
pickle_output(out_t, out_x, sps.OUTPUT_DIR + 'results.pkl')

    
