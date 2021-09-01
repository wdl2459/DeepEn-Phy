import torch
import torch.nn as nn

class MicrobiomeTreeModel(nn.Module):
    def __init__(self, hyper_model, partition_flatten_cleaned, leaf_name_to_index):
        super().__init__()

        self.partition_flatten_cleaned = partition_flatten_cleaned
        self.leaf_name_to_index = leaf_name_to_index
        self.root_name = list(partition_flatten_cleaned.keys())[-1]

        self.all_layer = nn.ModuleDict()
        for output_node, input_nodes in self.partition_flatten_cleaned.items():
            # Calculate the input dimension
            input_dim = 0
            for input_node in input_nodes:
                if input_node in self.leaf_name_to_index:
                    # This input node is a leaf node that has data in it
                    input_dim += 1
                else:
                    # This input node is the output node of another operation,
                    # so we add its output dimension (the last element of 'n_neuron')
                    input_dim += hyper_model[input_node]['n_neuron'][-1]

            # Build each small feed forward network (MLP)
            self.all_layer[output_node] = FeedForwardNet(hyper_model[output_node], input_dim)

        self.info = {'n_net': len(self.all_layer),
                     'n_trainable_param': sum(p.numel() for p in self.parameters() if p.requires_grad)}
        print('A model consisting of {0:d} feedforward networks and {1:d} trainable parameters built'.format(
            self.info['n_net'], self.info['n_trainable_param']))

    def forward(self, x):
        all_output = dict()
        for output_node, input_nodes in self.partition_flatten_cleaned.items():
            inputs = torch.zeros(x.shape[0], 0, device=x.device)
            for input_node in input_nodes:
                if input_node in self.leaf_name_to_index:
                    inputs = torch.cat((inputs, x[:, [self.leaf_name_to_index[input_node]]]), dim=1)
                else:
                    inputs = torch.cat((inputs, all_output[input_node]), dim=1)
            all_output[output_node] = self.all_layer[output_node](inputs)
        return all_output[self.root_name]


class FeedForwardNet(nn.Module):
    def __init__(self, hyper_model_single, input_dim):
        super().__init__()

        if hyper_model_single['residual']:
            if input_dim != hyper_model_single['n_neuron'][-1]:
                self.res_proj = nn.Linear(input_dim, hyper_model_single['n_neuron'][-1], bias=False)
            else:
                self.res_proj = nn.Identity()
        else:
            self.res_proj = None

        all_layer = list()
        for i in range(hyper_model_single['n_layer']):
            output_dim = hyper_model_single['n_neuron'][i]

            all_layer.append(nn.Linear(input_dim, output_dim))
            if hyper_model_single['batch_norm'][i] == 'before_act':
                all_layer.append(nn.BatchNorm1d(output_dim))
                all_layer.append(getattr(nn, hyper_model_single['act_func'][i])())
            elif hyper_model_single['batch_norm'][i] == 'after_act':
                all_layer.append(getattr(nn, hyper_model_single['act_func'][i])())
                all_layer.append(nn.BatchNorm1d(output_dim))
            else:
                all_layer.append(getattr(nn, hyper_model_single['act_func'][i])())
            all_layer.append(nn.Dropout(p=hyper_model_single['dropout_prob'][i]))

            input_dim = output_dim
        self.all_layer = nn.Sequential(*all_layer)

    def forward(self, x):
        return self.all_layer(x) if (self.res_proj is None) else (self.all_layer(x) + self.res_proj(x))
