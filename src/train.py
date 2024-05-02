import logging
import argparse

from src.model import VisionTransformer
from src.data import load_cifar10 
from src.config import TRAINING, DATA, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", type=bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # initialize arguments
    args = get_args()

    # load CIFAR-10 dataset
    try:
        train_loader, test_loader = load_cifar10(DATA_DIR, download=args.download, batch_size=TRAINING["batch_size"])
    except Exception as e:
        logging.error(f"Error loading CIFAR-10 dataset: {e}")
        logging.error("Please run the script with the flag '--download True' to download the dataset")

    # initialize model
    model = VisionTransformer(
        image_width=DATA["image_width"], image_height=DATA["image_height"], channel_size=DATA["channels"],
        patch_size=DATA["patch_size"], latent_space_dim=DATA["latent_space_dim"]
    )