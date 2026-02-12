from time_integrator import solve_problem
from pickler import pickle_ouput, pickle_setup 

def solve_full(start_drop, prob_univ):
    
    # First, Let's solve the problem
    out_t, out_x = solve_problem(out_t, out_x)

    # Now, we pickle the setup
    pickle_setup(start_drop, pu, out_file = sps.OUTPUT_DIR + 'setup.pkl')

    # Next, we pickle the problem
    pickle_output(out_t, out_x, sps.OUTPUT_DIR + 'results.pkl')

    # Let's print something big here
    print('\n' * 5)
    print('SOLVE COMPLETE!!!')
    print('\n' * 5)

