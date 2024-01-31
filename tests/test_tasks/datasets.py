from torchvision import datasets, transforms

default_datapath = "tmp"


def get_mnist():
    return datasets.MNIST(
        root=default_datapath,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
