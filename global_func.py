import pickle
import os
import copy
import numpy as np

def write_file(z, path, name):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(z, f)


def read_file(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        z = pickle.load(f)
        return z


def var_parser(x, x_type):
    # Make sure the type of variable x is valid
    assert (x is None) or \
           isinstance(x, (int, np.int16, np.int32, np.int64)) or \
           isinstance(x, (float, np.float16, np.float32, np.float64)) or \
           isinstance(x, str)

    # Convert variable x to a variable of type x_type (this type will never be list / tuple)
    if x in {None, 'None', 'none', 'NONE', 'Null', 'null', 'NULL', 'NaN', 'nan', 'NAN', 'NA'}:
        return None

    if x_type == 'int':
        return int(x)
    elif x_type == 'float':
        return float(x)
    elif x_type == 'str':
        return str(x)
    elif x_type == 'bool':
        return False if x in {
            False, 'False', 'false', 'FALSE', 'f', 'F', 'No', 'no', 'NO', 'n', 'N', '0', '0.0', 0} else bool(x)
    else:
        assert (x_type == 'eval') and (not isinstance(eval(x), (list, tuple)))
        return eval(x)


def hyper_parser_single(x, x_type, n=None):
    if n is None:
        # x is not a list / tuple; the type of variable this parser will return is also not a list / tuple
        return var_parser(x, x_type)
    else:
        # x could be a list / tuple or not; the type of variable this parser will return is a list of length n
        if isinstance(x, (list, tuple)):
            assert (len(x) == n) and (len(set(x)) != 1)
            return [var_parser(i, x_type) for i in x]
        else:
            return [var_parser(x, x_type) for _ in range(n)]


def hyper_parser(hyper):
    h = dict()
    h['n_layer'] = hyper_parser_single(hyper['n_layer'], 'int')
    h['n_neuron'] = hyper_parser_single(hyper['n_neuron'], 'int', int(hyper['n_layer']))
    h['act_func'] = hyper_parser_single(hyper['act_func'], 'str', int(hyper['n_layer']))
    h['batch_norm'] = hyper_parser_single(hyper['batch_norm'], 'str', int(hyper['n_layer']))
    h['dropout_prob'] = hyper_parser_single(hyper['dropout_prob'], 'float', int(hyper['n_layer']))
    h['residual'] = hyper_parser_single(hyper['residual'], 'bool')
    h['bandwidth'] = hyper_parser_single(hyper['bandwidth'], 'float')
    h['gpu'] = hyper_parser_single(hyper['gpu'], 'str')
    h['dir_train'] = hyper_parser_single(hyper['dir_train'], 'str')
    h['dir_method_train'] = hyper_parser_single(hyper['dir_method_train'], 'str')
    h['n_epoch'] = hyper_parser_single(hyper['n_epoch'], 'int')
    h['optimizer'] = hyper_parser_single(hyper['optimizer'], 'eval')
    h['lr_scheduler'] = hyper_parser_single(hyper['lr_scheduler'], 'eval')
    h['l1_reg_lambda'] = hyper_parser_single(hyper['l1_reg_lambda'], 'float')
    h['l2_reg_lambda'] = hyper_parser_single(hyper['l2_reg_lambda'], 'float')
    h['batch_size_train'] = hyper_parser_single(hyper['batch_size_train'], 'int')
    h['shuffle_train'] = hyper_parser_single(hyper['shuffle_train'], 'bool')
    h['num_workers_train'] = hyper_parser_single(hyper['num_workers_train'], 'int')
    h['pin_memory_train'] = hyper_parser_single(hyper['pin_memory_train'], 'bool')
    h['type_predict'] = hyper_parser_single(hyper['type_predict'], 'str')
    h['thres_predict'] = hyper_parser_single(hyper['thres_predict'], 'float')
    h['batch_size_predict'] = hyper_parser_single(hyper['batch_size_predict'], 'int')
    h['num_workers_predict'] = hyper_parser_single(hyper['num_workers_predict'], 'int')
    h['pin_memory_predict'] = hyper_parser_single(hyper['pin_memory_predict'], 'bool')
    h['print_freq'] = hyper_parser_single(hyper['print_freq'], 'str')
    h['eval_save_freq'] = hyper_parser_single(hyper['eval_save_freq'], 'str')
    h['save_checkpoints'] = hyper_parser_single(hyper['save_checkpoints'], 'bool')
    assert set(h.keys()) == set(hyper.keys())
    return h


def get_hyper_model(hyper, partition_flatten_cleaned, final_output_dim):
    # Assume the feedforward networks (MLPs) of all output nodes use the same hyperparameter setting given by hyper
    hyper_model = {output_node: copy.deepcopy(hyper) for output_node in partition_flatten_cleaned}

    root_name = list(partition_flatten_cleaned.keys())[-1]
    hyper_model[root_name]['n_neuron'][-1] = final_output_dim
    hyper_model[root_name]['act_func'][-1] = 'Identity'

    return hyper_model
