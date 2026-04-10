import heightscape
import tapescape
import physical_params
import solver_params
import droplet
import numpy as np
from time_integrator import solve_problem, solve_problem_rk4
from problem_universe import problem_universe
from matplotlib import pyplot as plt
from pickler import pickle_output, pickle_canting
import uuid
import os
from run import solve_full

def big_print(a_str):
    print('\n' * 10)
    print('-' * 50)
    print(f'\t\t{a_str}\t\t')
    print('-' * 50)
    print('\n' * 10)

def run_default_slight(w_star_ratio):

    # Make our heightscape
    hs = heightscape.constructors.rect_interp_htscape((10, 10), (.1, .2, .2, .1))

    # Make the tapescape like normal
    ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

    # Make the physical parameters
    pps = physical_params.physical_params()
    pps.w_eq = w_star_ratio * pps.w_eq

    # Make the solver params
    sps = solver_params.solver_params()

    # Make the problem universe
    pu = problem_universe(hs, ts, sps, pps)

    # Make the starting droplet
    drop = droplet.constructors.make_drop_about_to_drop(5, 3, 5, hs, pps)

    # Make the output infrastrcture so we have it
    run_id = uuid.uuid4().hex
    dir_name = f'vacuum_tests/{run_id}/'
    os.mkdir(dir_name)

    # Update the settings output directory so our run will save right
    sps.OUTPUT_DIR = dir_name

    # Make the problem universe we'll need now
    pu = problem_universe(hs, ts ,sps, pps)

    # Finally, return the result
    solve_full(drop, pu)

# Nice and easy I guess
w_star_ratios = (.5, .75, 1, 1.5, 2)
for w_star_ratio in w_star_ratios:
    big_print(f'RUNNING FOR {w_star_ratio}')
    run_default_slight(w_star_ratio)
    big_print(f'DONE RUNNING {w_star_ratio}')
    
