from gas_fem import solve_concentration_field
import heightscape
import tapescape
import physical_params
import solver_params
import numpy as np

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

# Do the visualization check
solve_concentration_field(dr_x, dr_y, hs,ts,pps,sps, viz=True)
