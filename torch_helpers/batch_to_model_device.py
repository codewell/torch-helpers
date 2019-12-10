from .dict_to_device import dict_to_device
from .list_to_device import list_to_device


def batch_to_model_device(batch, model):
    device = next(model.parameters()).device
    if isinstance(batch, dict):
        return dict_to_device(batch, device)
    elif isinstance(batch, list):
        return list_to_device(batch, device)
    else:
        raise ValueError('Expected batch to be type list or dict')
