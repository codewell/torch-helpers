def list_to_device(list, device):
    reconstruction = []
    for value in list:
        if isinstance(value, dict):
            reconstruction.append(dict_to_device(value, device))
        elif isinstance(value, list):
            reconstruction.append(list_to_device(value, device))
        elif isinstance(value, torch.Tensor):
            reconstruction.append(value.to(device))
        else:
            reconstruction.append(value)
    return reconstruction
