import torch

if torch.cuda.is_available():
    device = "cuda"

    def to_device(tensor):
        return tensor.to(device)

    def to_device_model(model):
        model.to("cuda")

else:
    device = "cpu"

    # on cpu we need to use double as otherwise ill-conditioning in sums
    # causes numerical instability
    def to_device(tensor):
        return tensor.double()

    def to_device_model(model):
        model.double()
