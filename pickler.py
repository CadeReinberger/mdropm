import pickle as pk

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
