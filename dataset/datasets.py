from dataset.car_dataset import get_car_dataset
from dataset.celeba_dataset import get_celeb_dataset
from dataset.voc_dataset import get_voc_dataset
from dataset.coco_dataset import get_coco_dataset
from dataset.caltech256_dataset import get_caltech256_dataset
from dataset.kuzushiji_dataset import get_kuzushiji49_dataset
from dataset.cifar10_imb_dataset import get_cifar10_imb_dataset
from dataset.cifar100_imb_dataset import get_cifar100_imb_dataset
from dataset.svhn_imb_dataset import get_svhn_imb_dataset
from dataset.imagenet_dataset import get_imagenet_dataset
import torch


def get_dataset(name, batch_size):
    if name == "car":
        train, test = get_car_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, True, 10
    elif name == "celeb":
        train, test = get_celeb_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, True, 40
    elif name == "voc":
        train, test = get_voc_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, True, 20
    elif name == "coco":
        train, test = get_coco_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, True, train.num_classes
    elif name == "caltech":
        train, test = get_caltech256_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, 256
    elif name == "kuzushiji":
        train, test = get_kuzushiji49_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, 49
    elif "cifar10_imb" in name:
        n_class = int(name.split("_")[-1])
        train, test = get_cifar10_imb_dataset(n_class)
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, n_class
    elif "cifar100_imb" in name:
        n_class = int(name.split("_")[-1])
        train, test = get_cifar100_imb_dataset(n_class)
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, n_class
    elif "svhn_imb" in name:
        n_class = int(name.split("_")[-1])
        train, test = get_svhn_imb_dataset(n_class)
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, n_class
    elif "imagenet" in name:
        train, test = get_imagenet_dataset()
        val, test = torch.utils.data.random_split(test, [batch_size // 2, len(test) - batch_size // 2],
                                                  generator=torch.Generator().manual_seed(42))
        return train, val, test, False, 1000
