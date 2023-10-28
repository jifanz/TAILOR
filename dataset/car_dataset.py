import urllib.request as urllib2
import tarfile
from scipy.io import loadmat
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def getting_data(url, path):
    data = urllib2.urlopen(url)
    tar_package = tarfile.open(fileobj=data, mode="r:gz")
    tar_package.extractall(path)
    tar_package.close()
    return print("Data extracted and saved.")


def getting_metadata(url, filename):
    """
    Downloading a metadata file from a specific url and save it to the disc.
    """
    labels = urllib2.urlopen(url)
    file = open(filename, "wb")
    file.write(labels.read())
    file.close()
    return print("Metadata downloaded and saved.")


class MetaParsing:
    """
    Class for parsing image and meta-data for the Stanford car dataset to create a custom dataset.
    path: The filepah to the metadata in .mat format.
    *args: Accepts dictionaries with self-created labels which will be extracted from the metadata (e.g. {0: "Audi", 1: "BMW", 3: "Other").
    year: Can be defined to create two classes (<=year and later).
    """

    def __init__(self):
        if not os.path.exists("./data/carimages"):
            getting_data("http://ai.stanford.edu/~jkrause/car196/car_ims.tgz", "./data/carimages")
        if not os.path.exists("./data/car_metadata.mat"):
            getting_metadata("http://ai.stanford.edu/~jkrause/car196/cars_annos.mat", "./data/car_metadata.mat")
        self.mat = loadmat("./data/car_metadata.mat")
        self.year = 2009
        self.annotations = np.transpose(self.mat["annotations"])
        # Extracting the file name for each sample
        self.file_names = [annotation[0][0][0].split("/")[-1] for annotation in self.annotations]
        # Extracting the index of the label for each sample
        self.label_indices = [annotation[0][5][0][0] for annotation in self.annotations]
        # Extracting the car names as strings
        self.car_names = [x[0] for x in self.mat["class_names"][0]]
        # Create a list with car names instead of label indices for each sample
        self.translated_car_names = [self.car_names[x - 1] for x in self.label_indices]
        self.name2class = {"Audi": 0, "BMW": 1, "Chevrolet": 2, "Dodge": 3, "Ford": 4, "Convertible": 5, "Coupe": 6,
                           "SUV": 7, "Van": 8}

    def parsing(self):
        labels = np.zeros((len(self.translated_car_names), len(self.name2class) + 1))
        for i, x in enumerate(self.translated_car_names):
            for name in self.name2class:
                if name in x:
                    labels[i, self.name2class[name]] = 1
            if int(x.split(" ")[-1]) <= self.year:
                labels[i, len(self.name2class)] = 1
        return labels, self.file_names


class CarDataset(Dataset):
    def __init__(self, transform):
        self.labels, self.file_names = MetaParsing().parsing()
        assert len(self.labels) == len(self.file_names)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_loc = os.path.join("./data/carimages/car_ims", self.file_names[idx])
        image = Image.open(img_loc).convert('RGB')
        single_img = self.transform(image)

        return single_img, self.labels[idx]


def get_car_dataset():
    dataset = CarDataset(transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [len(dataset) - len(dataset) // 5, len(dataset) // 5],
                                                                generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset


if __name__ == "__main__":
    train, test = get_car_dataset()
    print(len(train), len(test))
