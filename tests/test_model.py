import pytest
import torch
import logging

from src.model import VisionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@pytest.fixture(scope="class")
def load_data():
    """Fixture to create synthetic data for testing."""
    batch_size = 64
    num_channels = 3
    image_size = 32
    num_samples = 100
    
    images = torch.rand(batch_size, num_channels, image_size, image_size)
    labels = torch.randint(0, 10, (batch_size,))
    
    data_loader = [(images, labels)] * (num_samples // batch_size)
    return data_loader

@pytest.fixture(scope="class")
def model():
    """Fixture to initialize the Vision Transformer model with custom parameters."""
    custom_params = {
        "image_width": 32,
        "image_height": 32,
        "channel_size": 3,
        "patch_size": 4,
        "latent_space_dim": 128,
        "dim_ff": 512,
        "num_heads": 8,
        "depth": 6,
        "num_classes": 10
    }
    
    vit_model = VisionTransformer(
        image_width=custom_params["image_width"], 
        image_height=custom_params["image_height"],
        channel_size=custom_params["channel_size"], 
        patch_size=custom_params["patch_size"],
        latent_space_dim=custom_params["latent_space_dim"], 
        dim_ff=custom_params["dim_ff"],
        num_heads=custom_params["num_heads"], 
        depth=custom_params["depth"], 
        num_classes=custom_params["num_classes"]
    )
    
    return vit_model

@pytest.mark.usefixtures("load_data", "model")
class TestVisionTransformer:
    """Test class for Vision Transformer functionality."""

    def test_embedding_shape(self, load_data, model):
        """Test that embedding shape is as expected after a forward pass."""
        data_iter = iter(load_data)
        images, _ = next(data_iter)
        logits, loss = model(images)
        logging.info(f"Output embedding shape: {logits.shape}, loss is {loss}")

        expected_shape = (64, 10)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, but got {logits.shape}"