from torch.utils.data import DataLoader, Dataset
import torch


def get_labels(dataset):
    loader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=10, drop_last=False)
    labels = []
    for _, target in loader:
        labels.append(target)
    labels = torch.cat(labels, dim=0)
    return labels.numpy()


class MemoryDataset(Dataset):
    def __init__(self, X, y, n_class, transform=None, target_transform=None):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y)
        self.n_class = n_class
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = self.X[item]
        y = self.y[item]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
