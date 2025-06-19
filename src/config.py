from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'model'

SEED = 42

DATA = {
    "image_width": 32,
    "image_height": 32,
    "channel_size": 3,
    "patch_size": 8, # larger patches for less computation?
    "num_classes": 10,
}

TRAINING = {
    "batch_size": 64,
    "num_epochs": 7, # 7 or 14 
    "learning_rate": 3e-4, # maybe slightly higher?
    "weight_decay": 1e-5,
    "val_size": 0.1,
    "patience": 2,
}

MODEL = {
    "latent_space_dim": 256, # probably increase for good performance
    "dim_ff": 1024, # probably increase for good performance
    "num_heads": 8,
    "depth": 2, # increase for good performance
}