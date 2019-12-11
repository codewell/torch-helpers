from .list_to_device import list_to_device


def dict_to_device(container, device):
    reconstruction = dict()
    for key, value in container.items():
        if isinstance(value, dict):
            reconstruction[key] = dict_to_device(value, device)
        elif isinstance(value, list):
            reconstruction[key] = list_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            reconstruction[key] = value.to(device)
        else:
            reconstruction[key] = value
    return reconstruction
