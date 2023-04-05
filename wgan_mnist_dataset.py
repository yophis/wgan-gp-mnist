import torch
from torch.utils.data import Dataset


class GANInput(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        x = self.dataset[i]
        z = torch.randn(128)
        # e = random.gauss(0., 1.)
        e = torch.randn(1).item()
        return x, z, e

    def __len__(self):
        return self.dataset.shape[0]

