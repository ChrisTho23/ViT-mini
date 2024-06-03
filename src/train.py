import logging
import argparse
import torch
import wandb
from datetime import datetime

from src.model import VisionTransformer
from src.data import load_cifar10 
from src.config import TRAINING, DATA, DATA_DIR, MODEL_DIR, MODEL, SEED

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

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, 
    data: torch.utils.data.DataLoader,
    type: str,
    device: torch.device
):
    logging.info(f"Evaluating model on {type} set ...")

    model.eval()

    acc = 0.
    loss = 0.
    num_batches = len(data)

    for i, (x_test, y_test) in enumerate(data):
        x_test, y_test = x_test.to(device), y_test.to(device)
        logits, loss = model(x_test, y_test)

        preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        accuracy = (preds == y_test).float().mean()

        acc += accuracy.item()
        loss += loss.item()

    acc /= num_batches
    loss /= num_batches

    wandb.log({f"{type}_loss": loss, f"{type}_accuracy": acc})
    logging.info(f"{type} metrics - loss: {loss}, accuracy: {acc}")

    return loss

def train_model(
        model: torch.nn.Module,
        train_data: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        patience: int,
        val_data: torch.utils.data.DataLoader | None = None,
):
    logging.info("Training model...")

    n = len(train_data)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}")
        wandb.log({"epoch": epoch})

        model.train()

        running_loss = 0. # set running loss for current epoch to zero

        for i, data in enumerate(train_data): 
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            _, loss = model(x, y)

            loss.backward()

            # perform optimizer step
            optimizer.step()

            running_loss += loss.item()

            # report loss every 100th iteration
            if((i + 1) % 100 == 0):
                running_loss /= 100
                wandb.log({"train_loss": running_loss, "batch": (epoch * n + i+1)})
                logging.info(f"Epoch {epoch}, batch {i + 1} - Train loss: {running_loss}")
                running_loss = 0.

        if val_data:
            # evaluate model on val set
            epoch_val_loss = evaluate_model(
                model=model, 
                data=val_data,
                type="val",
                device=device
            )
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                timestamp = datetime.now().strftime("%m_%d_%H")
                torch.save(model.state_dict(), MODEL_DIR / f"model_{timestamp}.pt")
                patience_counter = 0
            else: 
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    return model

if __name__ == "__main__":
    # initialize arguments
    args = get_args()

    # initialize wandb
    wandb.init(
        project="ViT-mini",
        config={
            "dataset": "CIFAR-10",
            "patch_size": DATA["patch_size"],
            **vars(args),
            **MODEL
        }
    )

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # load CIFAR-10 dataset
    try:
        train_loader, val_loader, test_loader = load_cifar10(
            target_directory=DATA_DIR, 
            download=args.download, 
            batch_size=TRAINING["batch_size"],
            seed=SEED,
            val_size=TRAINING["val_size"],    
        )
    except Exception as e:
        logging.error(f"Error loading CIFAR-10 dataset: {e}")
        logging.error("Please run the script with the flag '--download True' to download the dataset")

    # initialize model
    model = VisionTransformer(
        image_width=DATA["image_width"], image_height=DATA["image_height"], channel_size=DATA["channel_size"],
        patch_size=DATA["patch_size"], latent_space_dim=MODEL["latent_space_dim"], dim_ff = MODEL["dim_ff"], 
        num_heads=MODEL["num_heads"], depth=MODEL["depth"], num_classes=DATA["num_classes"]
    ).to(device)

    # initialize optimizer
    optim = torch.optim.Adam(params=model.parameters(True), lr=args.lr, weight_decay=TRAINING["weight_decay"])

    model.train()

    # train model
    model = train_model(
        model=model, 
        train_data=train_loader,
        optimizer=optim,
        device=device,
        num_epochs=args.num_epochs,
        patience= TRAINING["patience"],
        val_data=val_loader,
    )

    # evaluate model
    _ = evaluate_model(
        model=model, 
        data=test_loader,
        type="test",
        device=device
    )