from torch.optim.lr_scheduler import LambdaLR

"""
In this module, you can define any function that returns a customized lr_scheduler
The inputs of these functions must be: optimizer, **kwargs
The output of these functions must be: lr_scheduler or None
"""

def null(optimizer, **kwargs):
    return None


def linear_with_warmup(optimizer, **kwargs):
    """
    This function is borrowed from the get_linear_schedule_with_warmup function in transformers.optimization

    It increases the learning rate linearly from 0 to the initial lr set in the optimizer during a warmup period, and
    then decreases the learning rate linearly from the initial lr set in the optimizer to 0

    optimizer: the optimizer for which to schedule the learning rate
    num_warmup_steps: the number of steps for the warmup phase
    num_training_steps: the total number of training steps
    last_epoch (defaults to -1): the index of the last epoch when resuming training
    """
    kwargs = {**{'last_epoch': -1}, **kwargs}

    def lr_lambda(current_step: int):
        if current_step < kwargs['num_warmup_steps']:
            return float(current_step) / float(max(1, kwargs['num_warmup_steps']))
        return max(0.0, float(kwargs['num_training_steps'] - current_step) / float(
            max(1, kwargs['num_training_steps'] - kwargs['num_warmup_steps'])))

    return LambdaLR(optimizer, lr_lambda, kwargs['last_epoch'])
