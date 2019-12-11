import torch


def json_to_device(container, device):
    reconstruction, container_iter = (
        ([], enumerate(container)) if isinstance(container, list)
        else ({}, container.items())
    )
    add_to_container = (
        (lambda container, key, value: container.append(value))
        if isinstance(reconstruction, list)
        else (lambda container, key, value: container.update([(key, value)]))
    )
    for key, value in container_iter:
        if type(value) in (list, dict):
            add_to_container(reconstruction, key, json_to_device(value, device))
        elif isinstance(value, torch.Tensor):
            add_to_container(reconstruction, key, value.to(device))
        else:
            add_to_container(reconstruction, key, value)
    return reconstruction
