import torch
import torchvision
from torchvision import datasets, transforms
import argparse
import logging

from src.config import TRAINING, DATA_DIR 
from src.utils import display_image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", type=bool, default=False)
    args = parser.parse_args()

    return args

def load_cifar10(target_directory, download, batch_size: int = 64) -> torch.utils.data.DataLoader:
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
    
    logging.info(f"CIFAR-10 dataset has been downloaded and saved in {target_directory}")

    return train_loader, test_loader

if __name__ == "__main__":
    args = get_args()

    # load CIFAR-10 dataset
    try:
        train_loader, test_loader = load_cifar10(DATA_DIR, download=args.download, batch_size=TRAINING["batch_size"])
    except Exception as e:
        logging.error(f"Error loading CIFAR-10 dataset: {e}")
        logging.error("Please run the script with the flag '--download True' to download the dataset")

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    logging.info("Displaying some random images from the CIFAR-10 dataset")
    logging.info(f"Each image has dimensions: {images.shape[1:4]}")
    display_image(torchvision.utils.make_grid(images))

