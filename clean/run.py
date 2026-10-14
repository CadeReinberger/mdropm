from time_integrator import solve_problem
from pickler import pickle_output, pickle_setup 
from pathlib import Path

def solve_full(start_drop, prob_univ):
    
    # First, Let's solve the problem
    out_t, out_x = solve_problem(start_drop, prob_univ)

    # Now, we pickle the setup
    output = Path(prob_univ.sol_ps.OUTPUT_DIR)
    pickle_setup(start_drop, prob_univ, out_file=output / 'setup.pkl')

    # Next, we pickle the problem
    pickle_output(out_t, out_x, output / 'results.pkl')

    # Let's print something big here
    print('\n' * 5)
    print('SOLVE COMPLETE!!!')
    print('\n' * 5)

