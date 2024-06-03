from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'

SEED = 42

DATA = {
    "image_width": 32,
    "image_height": 32,
    "channel_size": 3,
    "patch_size": 2,
    "num_classes": 10,
}

TRAINING = {
    "batch_size": 64,
    "num_epochs": 7, # 7 or 14 
    "learning_rate": 8e-6, # 2e-4 - 8e-4
    "weight_decay": 0.1,
    "val_size": 0.1,
    "patience": 2,
}

MODEL = {
    "latent_space_dim": 512,
    "dim_ff": 2048,
    "num_heads": 8,
    "depth": 6,
}