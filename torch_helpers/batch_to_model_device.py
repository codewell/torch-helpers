def batch_to_model_device(batch, model):
    return {
        name: tensor.to(next(model.parameters()).device)
        for name, tensor in batch.items()
    }
