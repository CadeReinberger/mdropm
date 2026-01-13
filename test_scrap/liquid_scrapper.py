from liquid_fem import solve_pressure_field, compute_pressure_gradients
import heightscape
import tapescape
import physical_params
import solver_params
import numpy as np
from problem_universe import problem_universe
import subprocess

# Make the heightscape
hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))

# Make the tapescape
ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

# Make the Physical Params
pps = physical_params.physical_params()

# Make the solver params
sps = solver_params.solver_params()

# Make the Starting Droplet
theta = np.linspace(0, 2*np.pi, num=51)[:-1]
dr_x, dr_y = 3*np.cos(theta), 3*np.sin(theta)
ps = np.sin(theta)**2

# Make the problem universe
pu = problem_universe(hs, ts, sps, pps)

# Check the compute gradients
grads = compute_pressure_gradients(dr_x, dr_y, ps, pu)
print(f'grads: {grads}')

# Do the visualization check
#solve_pressure_field(dr_x, dr_y, ps, hs, pps, sps, viz=True)

# FOR VIZ CHECKING
#subprocess.run(['gmsh', sps.LIQUID_PHASE_MESH_FILE])
