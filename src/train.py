import logging
import argparse
import torch

from src.model import VisionTransformer
from src.data import load_cifar10 
from src.config import TRAINING, DATA, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", type=bool, default=False)
    parser.add_argument(
        "--num_epochs", type=int, default=100 # check default value
    )
    parser.add_argument(
        "--lr", help="Set initial learning rate", type=float, default=0.001 # check default value
    )

    args = parser.parse_args()

    return args
def train_model(
        model: torch.nn.Module,
        train_data: torch.utils.data.Dataloader,
        val_data: torch.utils.data.Dataloader | None = None,
        optimizer: torch.optim.Optimizer | None = torch.optim.Adam(),
        num_epochs: int = 100
):
    phases = ["train", "val"] if val_data is not None else ["train"]
    for epoch in num_epochs:
        running_loss = 0. # set running loss for current epoch to zero
        for phase in phases: 
            if(phase == "train"):
                model.train(True) # set into training mode
            else:
                model.eval() # set into eval mode
            for i, data in enumerate(train_data): 
                x, y = data
                optimizer.zero_grad()

                _, loss = model(x, y)

                loss.backwar()

                # perform optimizer step
                optimizer.step()

                running_loss += loss.item()

            running_loss /= i # report mean loss for epoch
            logging.info(f"Epoch {epoch} {phase}-loss: {running_loss}")

    return model


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