import torch
from torch import nn
from torchvision import datasets, transforms
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.unfold(x) # B, (C * patch_size * patch_size), num_patches
        x = x.permute(0, 2, 1) # B, num_patches, (C * patch_size * patch_size)
        return x

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", action='store_true'
    )
    args = parser.parse_args()

    return args

def load_cifar10(target_directory, download, batch_size: int) -> torch.utils.data.DataLoader:
    logging.info("Downloading CIFAR-10 dataset")
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
    
    logging.info(
        f"CIFAR-10 dataset has been downloaded and saved in {target_directory}\n"
        f"Dataset has {len(train_loader)} train and {len(test_loader)} test samples"
    )

    return train_loader, test_loader

def load_cifar10(target_directory, download, batch_size: int) -> torch.utils.data.DataLoader:
    logging.info("Downloading CIFAR-10 dataset")
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
    
    logging.info(
        f"CIFAR-10 dataset has been downloaded and saved in {target_directory}\n"
        f"Dataset has {len(train_loader)} train and {len(test_loader)} test samples"
    )

    return train_loader, test_loader