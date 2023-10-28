from torchvision.datasets import CelebA
from torchvision import transforms


def get_celeb_dataset():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    train_dataset = CelebA(root="./data", split="train", target_type="attr", transform=transform, download=True)
    test_dataset = CelebA(root="./data", split="test", target_type="attr", transform=transform, download=True)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train, test = get_celeb_dataset()
    print(len(train), len(test))
