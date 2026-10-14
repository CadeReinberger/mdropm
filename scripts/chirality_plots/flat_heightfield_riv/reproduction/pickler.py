import pickle as pk
import os
import tempfile
from dataclasses import asdict


def atomic_pickle(obj, out_path):
    """Durably replace ``out_path`` without ever exposing a partial pickle."""
    out_path = os.fspath(out_path)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".pickle-", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as file:
            pk.dump(obj, file, protocol=pk.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, out_path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

def pickle_output(out_t, out_x, out_path):
    pk_dict = {'out_t': out_t, 'out_x': out_x}
    atomic_pickle(pk_dict, out_path)

def pickle_canting(ht_f, ht_b, rad, out_path):
    pk_dict = {'ht_front' : ht_f, 'ht_back' : ht_b, 'radius' : rad}
    atomic_pickle(pk_dict, out_path)

def pickle_speed(D, out_path):
    pk_dict = {'D' : D}
    atomic_pickle(pk_dict, out_path)

'''
Alright, let's pickle everything that we have. 
Try and take all of the starting data that we're gonna need. 
'''
def pickle_setup(drop, pu, out_file = 'out/setup.pkl'):
    # First, let's pickle the starting heightscape
    hs_dict = pu.htscp.to_pickle_dict()

    # Next, let's pickle the physical params
    pp_dict = asdict(pu.phys_ps)

    # Next, let's pickle the solver parameters
    sp_dict = asdict(pu.sol_ps)

    # Now, we pickle the tapescape
    tp_dict = pu.tpscp.picklable()

    # Now, we gotta pickle the droplet
    drop_dict = drop.to_pickle_dict()

    # Now, we make the big dict, and pickle it!
    setup_dict = {'heightscape' : hs_dict,
                  'tapescape' : tp_dict, 
                  'solver_params' : sp_dict, 
                  'physical_params' : pp_dict,
                  'start_drop' : drop_dict}

    # Now we pickle the full thing
    atomic_pickle(setup_dict, out_file)
