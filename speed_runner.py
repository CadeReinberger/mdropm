import heightscape
import tapescape
import physical_params
import solver_params
import droplet
import numpy as np
from time_integrator import solve_problem, solve_problem_rk4
from problem_universe import problem_universe
from matplotlib import pyplot as plt
from pickler import pickle_output, pickle_speed
import uuid
import os

def big_print(a_str):
    print('\n' * 10)
    print('-' * 50)
    print(f'\t\t{a_str}\t\t')
    print('-' * 50)
    print('\n' * 10)

def run_default_speed(speed_ratio):
    hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))

    # Make the tapescape like normal
    ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

    # Make the physical parameters
    pps = physical_params.physical_params()

    # Change the D Value
    pps.D = pps.D * (1 + speed_ratio)

    # Make the solver params
    sps = solver_params.solver_params()
    sps.RADAU_DT = sps.RADAU_DT / (1 + speed_ratio)
    sps.T_FIN = sps.T_FIN / (1 + speed_ratio)

    # Make the problem universe
    pu = problem_universe(hs, ts, sps, pps)

    # Make the starting droplet
    drop = droplet.constructors.make_drop_about_to_drop(5, 3, 5, hs, pps)

    # Solve the droplet, only as far as we need
    out_t, out_x = solve_problem(drop, pu)

    # Make the output filename 
    run_id = uuid.uuid4().hex
    os.mkdir(f'speed_tests/{run_id}')

    # Pickle the result
    pickle_output(out_t, out_x, f'speed_tests/{run_id}/results.pkl')
    pickle_speed(pps.D, f'speed_tests/{run_id}/setup.pkl')

# speed_ratios = [-.75, -.5, -.25, 0, .25, .5, .75]
speed_ratios = [1, 2, 3]
for speed_ratio in speed_ratios:
    big_print(f'RUNNING FOR {speed_ratio}')
    run_default_speed(speed_ratio)
    big_print(f'DONE RUNNING {speed_ratio}')
    
