import torch
from torchvision import datasets, transforms
import argparse

from config import TRAINING, data_dir 

def load_cifar10(target_directory, download, batch_size: int = 64) -> torch.utils.data.DataLoader:
    # tranform data to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = datasets.CIFAR10(root=target_directory, train=True, download=download, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_set = datasets.CIFAR10(root=target_directory, train=False, download=download, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    
    print("CIFAR-10 dataset has been downloaded and saved in", target_directory)

    return train_loader, test_loader

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", type=bool, default=False)
    args = parser.parse_args()

    # Load CIFAR-10 dataset
    train_loader, test_loader = load_cifar10(data_dir, batch_size=TRAINING["batch_size"], download=args.download)

