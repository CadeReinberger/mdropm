import pickle as pk

def pickle_output(out_t, out_x, out_path):
    pk_dict = {'out_t': out_t, 'out_x': out_x}
    with open(out_path, 'wb') as file:
        pk.dump(pk_dict, file)

