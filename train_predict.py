import time
import re
import os
import random
import shutil
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import customized_lr_scheduler
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, l1_loss, mse_loss
from microbiome_tree_data import MicrobiomeTreeData
from collections import defaultdict
from global_func import write_file

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_freq(freq_string, steps_per_epoch):
    match = re.match(r'^(?P<freq>[0-9]+)\s*(?P<unit>[a-zA-Z]+)$', freq_string)
    if match.group('unit') in {'epoch', 'epochs'}:
        return int(match.group('freq')) * steps_per_epoch
    else:
        return int(match.group('freq'))

def get_reg_sum(reg_params, reg_type, reg_lambda, batch):
    if reg_lambda == 0.0:
        return torch.tensor(0., device=batch.device, requires_grad=True)

    reg_sum = torch.tensor(0., device=batch.device, requires_grad=True)
    for param in reg_params:
        if reg_type == 'l1':
            reg_sum = reg_sum + l1_loss(input=param, target=torch.zeros_like(param), reduction='sum')
        else:
            assert reg_type == 'l2'
            reg_sum = reg_sum + mse_loss(input=param, target=torch.zeros_like(param), reduction='sum')

    return reg_sum * reg_lambda / (2 * batch.shape[0])

def create_dir(path_to_dir, dir_method):
    if os.path.exists(path_to_dir):
        if dir_method == 'delete_existing':
            shutil.rmtree(path_to_dir)
        elif dir_method == 'rename_existing':
            os.rename(path_to_dir.rstrip('/'), path_to_dir.rstrip('/') + datetime.now().strftime("_%m%d%y_%H%M%S"))
    os.makedirs(path_to_dir, exist_ok=(dir_method == 'use_existing'))

def train(dataset_train, dataset_val, model, loss_func, hyper_train):
    assert loss_func.reduction == 'mean'

    create_dir(hyper_train['dir_train'], hyper_train['dir_method_train'])
    write_file(hyper_train, hyper_train['dir_train'], 'hyperparameter')

    dataloader_train = DataLoader(dataset_train, batch_size=hyper_train['batch_size_train'],
                                  shuffle=hyper_train['shuffle_train'], num_workers=hyper_train['num_workers_train'],
                                  pin_memory=hyper_train['pin_memory_train'], drop_last=False,
                                  worker_init_fn=seed_worker)

    print_freq = get_freq(hyper_train['print_freq'], len(dataloader_train))
    eval_save_freq = get_freq(hyper_train['eval_save_freq'], len(dataloader_train))

    device = torch.device('cuda:{:}'.format(hyper_train['gpu'])
                          if torch.cuda.is_available() and hyper_train['gpu'] is not None else 'cpu')
    model.to(device)

    optimizer_kwargs = {k: v for k, v in hyper_train['optimizer'].items() if k != 'name'}
    optimizer = getattr(optim, hyper_train['optimizer']['name'])(model.parameters(), **optimizer_kwargs)

    lr_scheduler_kwargs = {**{'num_training_steps': len(dataloader_train) * hyper_train['n_epoch']},
                           **hyper_train['lr_scheduler']}
    lr_scheduler = getattr(customized_lr_scheduler, hyper_train['lr_scheduler']['name'])(
        optimizer, **lr_scheduler_kwargs)

    reg_params = [param for name, param in model.named_parameters() if 'weight' in name]

    """
    info_val is a dictionary of dictionary of dictionary. For example, it could look like
    {
      n_updates1: {'metrics_val': {'Loss': 0.02, 'Accuracy': 0.7}, 
                   'path_saved': {'model': 'path/to/model1', 'optim': 'path/to/optim1'}},
      n_updates2: {'metrics_val': {'Loss': 0.01, 'Accuracy': 0.8}, 
                   'path_saved': {'model': 'path/to/model2', 'optim': 'path/to/optim2'}},
      ...
    }
    """
    info_val = defaultdict(dict)
    all_batch_loss = list()
    n_updates = 0
    tic = time.time()
    for epoch_id in range(hyper_train['n_epoch']):
        for batch_id, (x_batch, y_batch) in enumerate(dataloader_train):
            # If x or y has only 1 dimension, dataloader will make it of size (batch_size, 1), not (batch_size, )
            x_batch = x_batch.to(device, non_blocking=hyper_train['pin_memory_train'])
            y_batch = y_batch.to(device, non_blocking=hyper_train['pin_memory_train'])

            l1_reg_sum = get_reg_sum(reg_params, 'l1', hyper_train['l1_reg_lambda'], x_batch)
            l2_reg_sum = get_reg_sum(reg_params, 'l2', hyper_train['l2_reg_lambda'], x_batch)

            output_batch = model(x_batch)
            if hyper_train['type_predict'] == 'multiple_classification':
                # Multiple classification uses CrossEntropyLoss, which requires the target to have shape (batch_size, )
                # but y_batch yielded by the dataloader always has shape (batch_size, 1), so we squeeze y_batch
                loss = loss_func(output_batch, torch.squeeze(y_batch, -1).long())
            else:
                loss = loss_func(output_batch, y_batch)
            all_batch_loss.append(loss)
            loss = loss + l1_reg_sum + l2_reg_sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_updates += 1

            toc = time.time()

            is_end_of_training = (epoch_id == hyper_train['n_epoch'] - 1 and batch_id == len(dataloader_train) - 1)
            if n_updates % eval_save_freq == 0 or is_end_of_training:
                if dataset_val is not None and len(dataset_val) > 0:
                    _, _, _, metrics_val = predict(dataset_val, model, loss_func, device,
                                                   type_predict=hyper_train['type_predict'],
                                                   thres_predict=hyper_train['thres_predict'],
                                                   batch_size_predict=hyper_train['batch_size_predict'],
                                                   num_workers_predict=hyper_train['num_workers_predict'],
                                                   pin_memory_predict=hyper_train['pin_memory_predict'])

                    print_val = ' - '.join(['Validation {0:}: {1:.4f}'.format(k, v) for k, v in metrics_val.items()
                                            if not isinstance(v, (pd.DataFrame, pd.Series))])
                    print('==== Epoch ID: {0:d} - Batch ID: {1:d} - Total Updates: {2:d} - Total Time: {3:.2f} second(s)'
                          ' - LR: {4:} - Training Batch Loss: {5:.4f} - {6:} ===='.format(
                           epoch_id, batch_id, n_updates, toc - tic, '{:.2e}'.format(optimizer.param_groups[0]['lr']),
                           all_batch_loss[-1].item(), print_val))

                    info_val[n_updates]['metrics_val'] = metrics_val

                if hyper_train['save_checkpoints'] or is_end_of_training:
                    path_saved_model = os.path.join(hyper_train['dir_train'], 'model_update{:d}.pth'.format(n_updates))
                    torch.save(model.state_dict(), path_saved_model)

                    path_saved_optim = os.path.join(hyper_train['dir_train'], 'optim_update{:d}.pth'.format(n_updates))
                    torch.save(optimizer.state_dict(), path_saved_optim)

                    info_val[n_updates]['path_saved'] = {'model': path_saved_model, 'optim': path_saved_optim}
            elif n_updates % print_freq == 0:
                print('==== Epoch ID: {0:d} - Batch ID: {1:d} - Total Updates: {2:d} - Total Time: {3:.2f} second(s)'
                      ' - LR: {4:} - Training Batch Loss: {5:.4f} ===='.format(
                       epoch_id, batch_id, n_updates, toc - tic, '{:.2e}'.format(optimizer.param_groups[0]['lr']),
                       all_batch_loss[-1].item()))

            if (lr_scheduler is not None) and (hyper_train['lr_scheduler']['update_freq'] == 'batch'):
                lr_scheduler.step()

        if (lr_scheduler is not None) and (hyper_train['lr_scheduler']['update_freq'] == 'epoch'):
            lr_scheduler.step()

    info_train = {(i + 1): {'metrics_train': {'Loss': batch_loss.item()}} for i, batch_loss in enumerate(all_batch_loss)}
    info_val = dict(info_val)
    write_file(info_train, hyper_train['dir_train'], 'info_train')
    write_file(info_val, hyper_train['dir_train'], 'info_val')

    torch.cuda.empty_cache()

    return info_train, info_val

def predict(data, model, loss_func, device, type_predict, thres_predict=0.5,
            batch_size_predict=None, num_workers_predict=1, pin_memory_predict=False):
    """
    :param data: either a MicrobiomeTreeData instance (bulk prediction) or a p-dim numpy array (single prediction)
    :param model: the model used to predict
    :param loss_func: the loss function used to calculate prediction loss
    :param device: device the model is on
    :param type_predict: "regression" or "multiple_classification" or "binary_classification"
    :param thres_predict: threshold for binary classification (predict 1 if score >= threshold; otherwise predict 0)
                          (will be ignored if type_predict is "regression" or "multiple_classification")
    :param batch_size_predict: batch size to use when data is a MicrobiomeTreeData instance
                               (if None, use all samples in a batch)
    :param num_workers_predict: num_workers in DataLoader
    :param pin_memory_predict: pin_memory in DataLoader
    :return: predictions, scores, logits and metrics
    """
    original_model_state = 'train' if model.training else 'eval'
    if original_model_state == 'train':
        model.eval()

    metrics = dict()
    if isinstance(data, MicrobiomeTreeData):
        dataloader = DataLoader(data, batch_size=batch_size_predict if batch_size_predict is not None else len(data),
                                shuffle=False, num_workers=num_workers_predict, pin_memory=pin_memory_predict,
                                drop_last=False, worker_init_fn=seed_worker)
        with torch.no_grad():
            if data.y_data is not None:
                assert loss_func.reduction == 'mean'

                all_y_batch = list()
                all_output_batch = list()
                all_loss = list()
                for batch_id, (x_batch, y_batch) in enumerate(dataloader):
                    x_batch = x_batch.to(device, non_blocking=pin_memory_predict)
                    y_batch = y_batch.to(device, non_blocking=pin_memory_predict)
                    all_y_batch.append(y_batch)

                    output_batch = model(x_batch)
                    all_output_batch.append(output_batch)
                    if type_predict == 'multiple_classification':
                        loss = loss_func(output_batch, torch.squeeze(y_batch, -1).long())
                    else:
                        loss = loss_func(output_batch, y_batch)
                    all_loss.append(loss * x_batch.shape[0])  # loss_func.reduction is 'mean'

                true = torch.cat(all_y_batch, dim=0).numpy()
                logit = torch.cat(all_output_batch, dim=0)
                pred, score = get_pred_score(logit, type_predict, thres_predict)

                metrics['Loss'] = torch.stack(all_loss).sum().item() / len(data)
                if type_predict in {'multiple_classification', 'binary_classification'}:
                    metrics.update(get_classification_metrics(pred, score, true, data.y_mapping))
            else:
                all_output_batch = list()
                for batch_id, x_batch in enumerate(dataloader):
                    x_batch = x_batch.to(device, non_blocking=pin_memory_predict)

                    output_batch = model(x_batch)
                    all_output_batch.append(output_batch)
                logit = torch.cat(all_output_batch, dim=0)
                pred, score = get_pred_score(logit, type_predict, thres_predict)
    else:
        assert isinstance(data, np.ndarray) and len(data.shape) == 1
        with torch.no_grad():
            x_batch = torch.from_numpy(data.reshape(1, -1)).to(device, non_blocking=pin_memory_predict)
            output_batch = model(x_batch)

            logit = output_batch
            pred, score = get_pred_score(logit, type_predict, thres_predict)

            pred = pred[0]
            if score is not None:
                score = score[0]

    if original_model_state == 'train':
        model.train()

    return pred, score, logit, metrics

def get_pred_score(output, type_predict, thres_predict):
    """
    This function should be called in the environment "with torch.no_grad()"

    It assumes scores for multiple and binary classifications are calculated by softmax and sigmoid, respectively,
    so it needs to be modified if this is not the case
    """
    if type_predict == 'regression':
        score = None
        pred = output.numpy()
    elif type_predict == 'multiple_classification':
        # output has shape (batch_size, c),
        # and pred has shape (batch_size, 1) whose values are integers between 0 and c - 1
        score = softmax(output, dim=-1).numpy()
        _, max_index = torch.max(output, dim=-1, keepdim=True)
        pred = max_index.numpy().astype(int)
    else:
        # output has shape (batch_size, 1),
        # and pred has shape (batch_size, 1) whose values are either 0 or 1
        assert type_predict == 'binary_classification' and output.shape[-1] == 1
        score = torch.sigmoid(output).numpy()
        pred = (score >= thres_predict).astype(int)
    return pred, score

def get_classification_metrics(pred, score, true, y_mapping):
    metrics = dict()

    labels, numbers = zip(*sorted(list(y_mapping.items()), key=lambda x: x[1]))
    n_labels = len(labels)

    assert numbers == tuple(range(n_labels))
    assert pred.shape == true.shape and pred.shape[-1] == 1  # pred and true are numpy arrays of shape (n, 1)

    confusion_matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]
    for i in range(pred.shape[0]):
        confusion_matrix[int(pred[i, 0])][int(true[i, 0])] += 1
    metrics['Confusion Table'] = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    n_total = metrics['Confusion Table'].values.sum()
    n_pred = metrics['Confusion Table'].sum(axis=1)
    n_true = metrics['Confusion Table'].sum(axis=0)
    n_correct = pd.Series([metrics['Confusion Table'].loc[label, label] for label in labels], index=labels)

    metrics['Accuracy'] = n_correct.sum() / n_total
    metrics['Recall'] = n_correct.divide(n_true)
    metrics['Precision'] = n_correct.divide(n_pred)
    metrics['F1-Score'] = (2 * metrics['Recall'].multiply(metrics['Precision'])).divide(
                            metrics['Recall'].add(metrics['Precision']))
    metrics['Recall of 1'] = metrics['Recall'].loc[labels[1]]
    metrics['Precision of 1'] = metrics['Precision'].loc[labels[1]]
    
    # Compute ROC/PR AUC
    true = pd.DataFrame(data=true)
    if len(y_mapping) > 2:
        true = pd.get_dummies(true.astype('category'))
    
    score = pd.DataFrame(data=score)
    
    ROC_AUC = []
    PR_AUC = []
    for i in range(score.shape[1]):
        ROC_AUC.append(roc_auc_score(true.iloc[:, i], score.iloc[:, i])) 
        PR_AUC.append(average_precision_score(true.iloc[:, i], score.iloc[:, i]))
        
    ROC_AUC = pd.Series(data=ROC_AUC)
    PR_AUC = pd.Series(data=PR_AUC)
    metrics['ROC-AUC'] = ROC_AUC
    metrics['PR-AUC'] = PR_AUC  
    
    if len(metrics['ROC-AUC']) > 1:
        metrics['ROC-AUC of 1'] = metrics['ROC-AUC'].iloc[1]
        metrics['PR-AUC of 1'] = metrics['PR-AUC'].iloc[1]
    else:
        metrics['ROC-AUC of 1'] = metrics['ROC-AUC'].iloc[0]
        metrics['PR-AUC of 1'] = metrics['PR-AUC'].iloc[0]
        
    return metrics
