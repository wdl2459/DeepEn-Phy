import os
import pyreadr
import torch.nn as nn
import torch
import numpy as np
from microbiome_tree import MicrobiomeTree
from microbiome_tree_model import MicrobiomeTreeModel
from microbiome_tree_data import MicrobiomeTreeData
from train_predict import train, predict, set_seed
from global_func import hyper_parser, get_hyper_model
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error

os.chdir('/path/to/working/directory')

########################################################################################################################
# Prepare data

dir_data = '/path/to/data/directory'

# Change name of the tree file if needed; 3 columns: start (of branch), end (of branch), length
tree_tab = pyreadr.read_r(os.path.join(dir_data, 'tree_tab_genus.Rdata'))['tree_tab']
tree_tab['start'] = tree_tab['start'].astype(str)
tree_tab['end'] = tree_tab['end'].astype(str)
all_node_name = set(tree_tab['start']) | set(tree_tab['end'])
assert all_node_name == {str(i + 1) for i in range(len(all_node_name))}

# Change size of the train set if needed
n_train = 6000

# Change name of the train and validation file if needed; samples by taxa and host phenotypes
train_val_df = pyreadr.read_r(os.path.join(dir_data, 'train_set_genus.Rdata'))['train_set'].sample(
    frac=1, random_state=0).reset_index(drop=True)

# Change name of the test file if needed; samples by taxa and host phenotypes
test_df = pyreadr.read_r(os.path.join(dir_data, 'test_set_genus.Rdata'))['test_set']

# Name of microbes, name of host phenotype, dictionary when the phenotype is categorical
x_name, y_name, y_mapping = 'taxon', 'smoking_binary', {'0': 0, '1': 1}

# # Name of microbes, name of host phenotype, None when the phenotype is continuous
# x_name, y_name, y_mapping = 'taxon', 'bmi', None

dataset_train = MicrobiomeTreeData(train_val_df.iloc[:n_train, :].reset_index(drop=True),
                                   x_name=x_name, y_name=y_name, y_mapping=y_mapping)
dataset_val = MicrobiomeTreeData(train_val_df.iloc[n_train:, :].reset_index(drop=True),
                                 x_name=x_name, y_name=y_name, y_mapping=y_mapping)
dataset_test = MicrobiomeTreeData(test_df,
                                  x_name=x_name, y_name=y_name, y_mapping=y_mapping)
assert dataset_train.leaf_name_to_index == dataset_val.leaf_name_to_index
assert dataset_train.leaf_name_to_index == dataset_test.leaf_name_to_index

if y_mapping is None:
    type_predict = 'regression'
elif len(y_mapping) >= 3:
    type_predict = 'multiple_classification'
else:
    assert len(y_mapping) == 2
    type_predict = 'binary_classification'

########################################################################################################################
# Ensemble learning:
# For each bandwidth, find the best model according to the performance on the validation set
# Apply all these models to the test set, and average the scores

band_candidate = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]

metrics_test = []
score_matrix_test = []

for band_chosen in band_candidate:

    # The directory to save the trained model
    dir_train = '/path/to/training/directory{:}'.format(band_chosen)

    # Initialize the microbiome tree
    tree = MicrobiomeTree(dir_train, 'microbiome_tree')

    # Add nodes to the microbiome tree
    for name in all_node_name:
        tree.add_node(name)

    # Add edges to the microbiome tree
    for i in tree_tab.index:
        tree.add_edge(tree_tab.loc[i, 'start'], tree_tab.loc[i, 'end'], tree_tab.loc[i, 'length'])

    # Preprocess the tree to get distance_to_root, level, descendent, antecedent for each node
    tree.preprocess()

    # Hyperparameters, which can be changed if needed
    hyper = {
         'n_layer': 1,
         'n_neuron': 5,
         'act_func': 'ELU',
         'batch_norm': 'before_act',
         'dropout_prob': 0.0,
         'residual': 'True',
         'bandwidth': band_chosen,
         'gpu': 'None',
         'dir_train': dir_train,
         'dir_method_train': 'raise_error',
         'n_epoch': 50,
         'optimizer': "{'name': 'Adam', 'lr': 5e-3, 'weight_decay': 0}",
         'lr_scheduler': "{'name': 'linear_with_warmup', 'update_freq': 'batch', 'num_warmup_steps': 100}",
         'l1_reg_lambda': 0.001,
         'l2_reg_lambda': 0.0,
         'batch_size_train': 512,
         'shuffle_train': 'True',
         'num_workers_train': 0,
         'pin_memory_train': 'True',
         'type_predict': type_predict,
         'thres_predict': 0.5,
         'batch_size_predict': 'None',
         'num_workers_predict': 0,
         'pin_memory_predict': 'False',
         'print_freq': '10 update',
         'eval_save_freq': '10 update',
         'save_checkpoints': 'True'
    }
    hyper = hyper_parser(hyper)

    partition, partition_flatten_cleaned = tree.get_partition(bandwidth=hyper['bandwidth'], verbose=False)

    if hyper['type_predict'] == 'regression':
        final_output_dim = 1
    elif hyper['type_predict'] == 'multiple_classification':
        final_output_dim = len(y_mapping)
    else:
        assert hyper['type_predict'] == 'binary_classification'
        final_output_dim = 1
    hyper_model = get_hyper_model(hyper, partition_flatten_cleaned, final_output_dim)

    torch.manual_seed(0)
    model = MicrobiomeTreeModel(hyper_model, partition_flatten_cleaned, dataset_train.leaf_name_to_index)
    if hyper['type_predict'] == 'regression':
        loss_func = nn.MSELoss(reduction='mean')
    elif hyper['type_predict'] == 'multiple_classification':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        assert hyper['type_predict'] == 'binary_classification'
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')

    set_seed(seed=round(band_chosen * 1000))
    info_train, info_val = train(dataset_train, dataset_val, model, loss_func, hyper)

    # Find the best model according to the performance on the validation set, and apply to the test set

    # Example 1: Classification
    record_ROC_AUC = 0
    record_update = 0

    for key in info_val:
        if info_val[key]['metrics_val']['ROC-AUC of 1'] > record_ROC_AUC:
            record_ROC_AUC = info_val[key]['metrics_val']['ROC-AUC of 1']
            record_update = key

    model.load_state_dict(torch.load(os.path.join(dir_train, 'model_update{:}.pth'.format(record_update))))

    # Get prediction, probability, feature, evaluation metrics
    pred, score, logit, metrics = predict(dataset_test, model, loss_func, device=torch.device('cpu'),
                                          type_predict='binary_classification', thres_predict=0.5,
                                          batch_size_predict=None, num_workers_predict=0, pin_memory_predict=False)
    metrics_test.append(metrics)

    score_list = score.tolist()
    score_List = sum(score_list, [])
    score_matrix_test.append(score_List)

    # # Example 2: Regression
    #
    # record_mse = float('Inf')
    # record_update = 0
    #
    # for key in info_val:
    #     if info_val[key]['metrics_val']['Loss'] < record_mse:
    #         record_mse = info_val[key]['metrics_val']['Loss']
    #         record_update = key
    #
    # model.load_state_dict(torch.load(os.path.join(dir_train, 'model_update{:}.pth'.format(record_update))))
    #
    # # Get prediction, prediction, prediction, evaluation metrics
    # pred, score, logit, metrics = predict(dataset_test, model, loss_func, device=torch.device('cpu'),
    #                                       type_predict='regression', thres_predict=0.5,
    #                                       batch_size_predict=None, num_workers_predict=0, pin_memory_predict=False)
    # metrics_test.append(metrics)
    #
    # pred_list = pred.tolist()
    # pred_List = sum(pred_list, [])
    # score_matrix_test.append(pred_List)

# Performance of each PhyNN
print(metrics_test)

# Example 1: Classification
# Ensemble performance: average score (probability) to get the final results
average_score_test = list(map(lambda x: sum(x) / len(x), zip(*score_matrix_test)))
average_pred_test = []
for kk in range(len(average_score_test)):
    average_pred_test.append(1 * (average_score_test[kk] > 0.5))

print('ROC-AUC:', roc_auc_score(dataset_test.y_data, average_score_test, average='weighted'))
print('F1-Score:', f1_score(dataset_test.y_data, average_pred_test, average='weighted'))

# # Example 2: Regression
# # Ensemble performance: average prediction (value) to get the final results
# average_pred_test = list(map(lambda x: sum(x) / len(x), zip(*score_matrix_test)))
#
# mse = mean_squared_error(dataset_test.y_data, average_pred_test)
# rmse = np.sqrt(mse)
# print('MSE:', mse)
# print('RMSE:', rmse)
