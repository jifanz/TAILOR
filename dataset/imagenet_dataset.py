import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision.datasets import ImageNet
from urllib.request import urlretrieve
from dataset.utils import MemoryDataset
import psutil


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.
    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.
    Returns
    -------
    filename : str
        The location of the downloaded file.
    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    filename, _ = urlretrieve(url, filename=destination)


urls = {'train': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar',
        'val': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar',
        'test': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar',
        'dev_kit': 'https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz'}


def download_imagenet2012(root, phase):
    url = urls[phase]
    filename = os.path.basename(url)

    if os.path.exists(os.path.join(root, filename)):
        print(f'{filename} dataset already exists!')
    else:
        download_url(url, destination=os.path.join(root, filename), progress_bar=True)
        print(f'{filename} dataset downloaded!')


def get_imagenet_dataset():
    download_imagenet2012("./data", "train")
    download_imagenet2012("./data", "val")
    download_imagenet2012("./data", "dev_kit")

    n_class = 1000
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose(
        [lambda x: torch.LongTensor([x]),
         lambda x: torch.flatten(F.one_hot(x, n_class))])

    train_dataset = ImageNet("./data", split="train", target_transform=target_transform, transform=train_transform)
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=(300000, len(train_dataset) - 300000),
    #                                                  generator=torch.Generator().manual_seed(1234))
    test_dataset = ImageNet("./data", split="val", target_transform=target_transform, transform=test_transform)
    # if psutil.virtual_memory().total > 200000000000:
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False,
    #                                                num_workers=40)
    #     train_imgs, train_labels = next(iter(train_loader))
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=40)
    #     test_imgs, test_labels = next(iter(test_loader))
    #     train_dataset = MemoryDataset(train_imgs, train_labels, n_class)
    #     test_dataset = MemoryDataset(test_imgs, test_labels, n_class)
    return train_dataset, test_dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train, test = get_imagenet_dataset()
    print(len(train), len(test))
    loader = DataLoader(train, batch_size=2)
    x, y = next(iter(loader))
    print(x.size(), y.size())
