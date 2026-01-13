import heightscape
import tapescape
import physical_params
import solver_params
import droplet
import numpy as np
from time_integrator import solve_problem
from problem_universe import problem_universe

# Make the heightscape
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
drop = droplet.constructors.make_circular_flat_drop(50, 3, hs)

# Solve the time integration problem
out_t, out_x = solve_problem(drop, pu)

print('Well We Did Something!')
