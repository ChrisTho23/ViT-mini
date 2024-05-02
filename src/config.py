from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

DATA = {
    "image_width": 32,
    "image_height": 32,
    "channels": 3,
    "patch_size": 8,
    "latent_space_dim": 256,
}

TRAINING = {
    "batch_size": 64,
}