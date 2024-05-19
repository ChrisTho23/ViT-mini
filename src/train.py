import logging
import argparse
import torch
import wandb

from src.model import VisionTransformer
from src.data import load_cifar10 
from src.config import TRAINING, DATA, DATA_DIR, MODEL_DIR, MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download", help="Flag to download the CIFAR-10 dataset", action='store_true'
    )
    parser.add_argument(
        "--num_epochs", type=int, default=TRAINING["num_epochs"]
    )
    parser.add_argument(
        "--batch_size", type=int, default=TRAINING["batch_size"]
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
    logging.info("Training model...")

    model.train()

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}")

        running_loss = 0. # set running loss for current epoch to zero

        for i, data in enumerate(train_data): 
            x, y = data
            optimizer.zero_grad()

            _, loss = model(x, y)

            loss.backward()

            # perform optimizer step
            optimizer.step()

            wandb.log({f"train_loss": loss.item()})
            running_loss += loss.item()

            # report loss every 100th iteration
            if((i + 1) % 100 == 0):
                running_loss /= 100
                logging.info(f"Epoch {epoch}, batch {i + 1} - Train loss: {running_loss}")
                running_loss = 0.

    return model

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, test_data: torch.utils.data.DataLoader):
    logging.info("Evaluating model...")

    model.eval()

    acc = 0.
    test_loss = 0.

    for i, (x_test, y_test) in enumerate(test_data):
        logits, loss = model(x_test, y_test)

        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        accuracy = (preds == y_test).float().mean()

        wandb.log({"batch_test_loss": loss.item(), "batch_test_accuracy": accuracy.item()})
        logging.info(f"Batch {i + 1} test metrics - Loss: {loss.item()}, Accuracy: {accuracy.item()}")

        acc = (acc * i + accuracy.item()) / (i + 1)
        test_loss = (test_loss * i + loss.item()) / (i + 1)

        wandb.log({"test_loss": test_loss, "test_accuracy": acc})

    logging.info(f"Final test metrics - Loss: {test_loss}, Accuracy: {acc}")


if __name__ == "__main__":
    # initialize arguments
    args = get_args()

    # initialize wandb
    wandb.init(
        project="ViT-mini",

        config={
            "dataset": "CIFAR-10",
            "patch_size": DATA["patch_size"],
            **args.__dict__,
            **MODEL
        }
    )

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

    # train model
    model = train_model(
        model=model, 
        train_data=train_loader,
        optimizer=optim,
        num_epochs=args.num_epochs,
        val_data=test_loader
    )

    # evaluate model
    evaluate_model(model, test_loader)

    # save model
    torch.save(model.state_dict(), MODEL_DIR / f"model_{wandb.run.id}.pt")