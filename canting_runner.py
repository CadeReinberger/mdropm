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

def big_print(a_str):
    print('\n' * 10)
    print('-' * 50)
    print(f'\t\t{a_str}\t\t')
    print('-' * 50)
    print('\n' * 10)

def run_default_slight(ht_ratio):
    # Height at the front and the back to use
    ht_back = .1
    ht_front = ht_back * (1 + ht_ratio)

    # Make our heightscape
    hs = heightscape.constructors.rect_interp_htscape((10, 10), (ht_back, ht_front, ht_front, ht_back))

    # Make the tapescape like normal
    ts = tapescape.constructors.make_rectangle_top_open(-5, -5, 5, 5)

    # Make the physical parameters
    pps = physical_params.physical_params()

    # Make the solver params
    sps = solver_params.solver_params()

    # Make the problem universe
    pu = problem_universe(hs, ts, sps, pps)

    # Make the starting droplet
    drop = droplet.constructors.make_drop_about_to_drop(10, 3, 5, hs, pps)

    print(f"Well we're trying to solve I think...")
    # Solve the droplet, only as far as we need
    out_t, out_x = solve_problem(drop, pu)

    # Make the output filename 
    run_id = uuid.uuid4().hex
    os.mkdir(f'canting_tests_2/{run_id}')

    # Pickle the result
    pickle_output(out_t, out_x, f'canting_tests_2/{run_id}/results.pkl')
    pickle_canting(ht_front, ht_back, 3, f'canting_tests_2/{run_id}/setup.pkl')

# Nice and easy I guess
ht_ratios = np.linspace(0, 2, num=8)[4:]
for ht_ratio in ht_ratios:
    big_print(f'RUNNING FOR {ht_ratio}')
    run_default_slight(ht_ratio)
    big_print(f'DONE RUNNING {ht_ratio}')
    
