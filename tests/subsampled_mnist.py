import os
import torch

from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

default_datapath = 'tmp'
if 'SLURM_TMPDIR' in os.environ:
    default_datapath = os.path.join(os.environ['SLURM_TMPDIR'], 'data')

def get_dataset(data_path=default_datapath, which_set='train', generate=True):
    if not os.path.isfile(os.path.join(data_path, 'subs_mnist.pt')) and generate:
        generate_dataset(data_path)
    t = torch.load(os.path.join(data_path, 'subs_mnist.pt'))[which_set]
    # if corruption > 0.:
    #     corrupted_mask = torch.empty(*t[1].size()).bernoulli_(corruption).long()
    #     corrupted_labels = torch.randint_like(t[1], 0, t[1].max() + 1)
    #     t = (t[0], (1 - corrupted_mask) * t[1] + corrupted_mask * corrupted_labels)
    return TensorDataset(*t)


def generate_dataset(data_path=default_datapath):
    size = 10

    train_set = Subset(MNIST(root=data_path, train=True, download=True,
                       transform=ToTensor()), range(40000))
    valid_set = Subset(MNIST(root=data_path, train=True, download=True,
                       transform=ToTensor()), range(40000, 50000))
    test_set = MNIST(root=data_path, train=False, download=True,
                     transform=ToTensor())

    train_loader = DataLoader(train_set, batch_size=40000)
    valid_loader = DataLoader(valid_set, batch_size=10000)
    test_loader = DataLoader(test_set, batch_size=10000)

    # Extract normalization

    def get_stats(train_loader):
        train_x, train_t = next(iter(train_loader))
        train_x = train_x.view(-1, 784)
        # mu = train_x.mean(dim=0)
        # train_x -= mu
        cov = torch.mm(train_x.t(), train_x) / train_x.size(0)
        _, u = torch.symeig(cov, eigenvectors=True)
        # e = e[-size:]
        u = u[:, -size:]
        # proj = torch.mm(u, torch.diag(1./(e)**0.5))
        # proj = torch.mm(u, u.t()[:, -u.size(1):])
        # reproject to random ON basis
        basis = torch.empty(size, size)
        torch.nn.init.orthogonal_(basis)
        proj = torch.mm(u, basis)

        return proj

    def transform(loader, proj):
        x, t = next(iter(loader))
        x = x.view(x.size(0), -1)
        # x -= mu
        x = torch.mm(x, proj)
        return (x, t)

    proj = get_stats(train_loader)
    train = transform(train_loader, proj)
    valid = transform(valid_loader, proj)
    test = transform(test_loader, proj)

    data = {'train': train, 'valid': valid, 'test': test}

    torch.save(data, os.path.join(data_path, 'subs_mnist.pt'))
    print('saved to ' + os.path.join(data_path, 'subs_mnist.pt'))


if __name__ == '__main__':
    generate_dataset()