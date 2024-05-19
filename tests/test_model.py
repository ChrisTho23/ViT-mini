import pytest
import logging

from src.model import VisionTransformer
from src.data import load_cifar10
from src.config import TRAINING, DATA, MODEL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@pytest.fixture(scope="class")
def load_data(request, tmpdir_factory):
    """Fixture to load CIFAR-10 data."""
    try:
        temp_dir = tmpdir_factory.mktemp("data")
        train_loader, _ = load_cifar10(str(temp_dir), download=True, batch_size=TRAINING["batch_size"])
        return train_loader
    except Exception as e:
        pytest.fail(f"Failed to load CIFAR-10 dataset: {e}")

@pytest.fixture(scope="class")
def model(request):
    """Fixture to initialize the Vision Transformer model."""
    vit_model = VisionTransformer(
        image_width=DATA["image_width"], image_height=DATA["image_height"],
        channel_size=DATA["channel_size"], patch_size=DATA["patch_size"],
        latent_space_dim=MODEL["latent_space_dim"], dim_ff=MODEL["dim_ff"],
        num_heads=MODEL["num_heads"], depth=MODEL["depth"], 
        num_classes=DATA["num_classes"]
    )
    request.cls.model = vit_model

@pytest.mark.usefixtures("load_data", "model")
class TestVisionTransformer:
    """Test class for Vision Transformer functionality."""

    def test_embedding_shape(self, load_data):
        """Test that embedding shape is as expected after a forward pass."""
        data_iter = iter(load_data)
        images, _ = next(data_iter)
        logits, loss = self.model(images)
        logging.info(f"Output embedding shape: {logits.shape}, loss is {loss}")

        expected_shape = (TRAINING["batch_size"], DATA["num_classes"])
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, but got {logits.shape}"