import pytest
import logging

from src.model import VisionTransformer
from src.data import load_cifar10
from src.config import TRAINING, DATA, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@pytest.fixture(scope="class")
def load_data(request):
    """Fixture to load CIFAR-10 data."""
    try:
        train_loader, _ = load_cifar10(DATA_DIR, download=True, batch_size=TRAINING["batch_size"])
        request.cls.train_loader = train_loader
    except Exception as e:
        pytest.fail(f"Failed to load CIFAR-10 dataset: {e}")

@pytest.fixture(scope="class")
def model(request):
    """Fixture to initialize the Vision Transformer model."""
    vit_model = VisionTransformer(
        image_width=DATA["image_width"], image_height=DATA["image_height"],
        channel_size=DATA["channels"], patch_size=DATA["patch_size"],
        latent_space_dim=DATA["latent_space_dim"]
    )
    request.cls.model = vit_model

@pytest.mark.usefixtures("load_data", "model")
class TestVisionTransformer:
    """Test class for Vision Transformer functionality."""

    def test_embedding_shape(self):
        """Test that embedding shape is as expected after a forward pass."""
        data_iter = iter(self.train_loader)
        images, _ = next(data_iter)
        embedding = self.model(images)
        logging.info(f"Output embedding shape: {embedding.shape}")

        num_patches = (DATA["image_height"] * DATA["image_width"]) // (DATA["patch_size"]**2)
        expected_shape = (TRAINING["batch_size"], num_patches, DATA["latent_space_dim"])
        assert embedding.shape == expected_shape, f"Expected shape {expected_shape}, but got {embedding.shape}"