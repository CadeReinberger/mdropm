import heightscape
import tapescape
import droplet

from physical_params import physical_params
from solver_params import solver_params
from problem_universe import problem_universe

from run import run_full

def main():
    
    # Make the heightscape
    hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))
    
    # Make the tapescape
    ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

    # Make the physical parameters
    pps = physical_params()

    # Make the solver parameters
    sps = solver_params()

    # Make the problem universe
    pu = problem_universe(hs, ts, sps, pps)

    # Make the starting droplet
    sd = droplet.constructors.make_circular_flat_drop(25, 3, hs)

    # Now we solve the full problem
    run_full(pu, sd)

if __name__ == '__main__':
    main()
