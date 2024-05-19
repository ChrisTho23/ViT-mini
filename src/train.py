import logging
import argparse
import torch

from src.model import VisionTransformer
from src.data import load_cifar10 
from src.config import TRAINING, DATA, DATA_DIR, MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", type=bool, default=False)
    parser.add_argument(
        "--num_epochs", type=int, default=TRAINING["num_epochs"]
    )
    parser.add_argument(
        "--lr", help="Set initial learning rate", type=float, default=TRAINING["learning_rate"]
    )

    args = parser.parse_args()

    return args

def train_model(
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        val_data: torch.utils.data.DataLoader | None = None,
):
    phases = ["train", "val"] if val_data is not None else ["train"]
    for epoch in range(num_epochs):
        running_loss = 0. # set running loss for current epoch to zero
        for phase in phases:
            logging.info(f"Epoch {epoch}, phase: {phase}")
            if(phase == "train"):
                model.train(True) # set into training mode
            else:
                model.eval() # set into eval mode
            for i, data in enumerate(train_data): 
                x, y = data
                optimizer.zero_grad()

                _, loss = model(x, y)

                loss.backward()

                # perform optimizer step
                optimizer.step()

                running_loss += loss.item()

                # report loss every 100th iteration
                if(i % 100 == 0):
                    running_loss /= 100
                    logging.info(f"Epoch {epoch}, Iteration {i}, loss: {running_loss}")
                    running_loss = 0.

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
        image_width=DATA["image_width"], image_height=DATA["image_height"], channel_size=DATA["channel_size"],
        patch_size=DATA["patch_size"], latent_space_dim=MODEL["latent_space_dim"], dim_ff = MODEL["dim_ff"], 
        num_heads=MODEL["num_heads"], depth=MODEL["depth"], num_classes=DATA["num_classes"]
    )

    # initialize optimizer
    optim = torch.optim.Adam(params=model.parameters(True), lr=args.lr)
    train_model(model=model, 
                train_data=train_loader,
                optimizer=optim,
                num_epochs=args.num_epochs)
    