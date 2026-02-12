import pickle as pk
from dataclasses import asdict

def pickle_output(out_t, out_x, out_path):
    pk_dict = {'out_t': out_t, 'out_x': out_x}
    with open(out_path, 'wb') as file:
        pk.dump(pk_dict, file)

def pickle_canting(ht_f, ht_b, rad, out_path):
    pk_dict = {'ht_front' : ht_f, 'ht_back' : ht_b, 'radius' : rad}
    with open(out_path, 'wb') as file:
        pk.dump(pk_dict, file)

def pickle_speed(D, out_path):
    pk_dict = {'D' : D}
    with open(out_path, 'wb') as file:
        pk.dump(pk_dict, file)

'''
Alright, let's pickle everything that we have. 
Try and take all of the starting data that we're gonna need. 
'''
def pickle_full(drop, pu, out_file = 'out/setup.pkl'):
    # First, let's pickle the starting heightscape
    hs_dict = pu.htscp.to_pickle_dict()

    # Next, let's pickle the physical params
    pp_dict = asdict(pu.phys_ps)

    # Next, let's pickle the solver parameters
    sp_dict = asdict(pu.sol_ps)

    # Now, we pickle the tapescape
    tp_dict = asdict(pu.picklable())

    # Now, we gotta pickle the droplet
    drop_dict = drop.to_pickle_dict()

    # Now, we make the big dict, and pickle it!
    setup_dict = {'heightscape' : hs_dict,
                  'tapescape' : tp_dict, 
                  'solver_params' : sp_dict, 
                  'physical_params' : pp_dict,
                  'start_drop' : drop_dict}

    # Now we pickle the full thing
    with open(out_file, 'wb') as file:
        pk.dump(setup_dict, file)
