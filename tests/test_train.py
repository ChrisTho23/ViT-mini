import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock

from src.model import VisionTransformer
from src.train import train_model, evaluate_model

# Test train_model function
@pytest.fixture
def synthetic_data():
    """Fixture to create a DataLoader with synthetic data for testing."""
    batch_size = 4
    num_channels = 3
    image_size = 16
    num_classes = 10
    data = torch.rand(batch_size * 5, num_channels, image_size, image_size, dtype=torch.float)
    labels = torch.randint(0, num_classes, (batch_size * 5,), dtype=torch.long)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, num_channels, image_size, num_classes, batch_size

@patch('wandb.log')
@patch('wandb.init', MagicMock())
def test_train_model(mock_wandb_log, synthetic_data):
    """Test the train_model function with synthetic data."""
    dataloader, num_channels, image_size, num_classes, _ = synthetic_data
    emb_dim = 16
    dim_ff = 32
    num_heads = 2
    depth = 1
    lr = 1e-3

    model = VisionTransformer(
        image_width=image_size, image_height=image_size, channel_size=num_channels,
        patch_size=4, latent_space_dim=emb_dim, dim_ff=dim_ff,
        num_heads=num_heads, depth=depth, num_classes=num_classes
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    trained_model = train_model(
        model=model,
        train_data=dataloader,
        optimizer=optimizer,
        num_epochs=2
    )

    assert isinstance(trained_model, torch.nn.Module)

@patch('wandb.log')
@patch('wandb.init', MagicMock())
def test_evaluate_model(mock_wandb_log, synthetic_data):
    """Test the evaluate_model function with synthetic data."""
    dataloader, num_channels, image_size, num_classes, batch_size = synthetic_data
    emb_dim = 16
    dim_ff = 32
    num_heads = 2
    depth = 1

    model = VisionTransformer(
        image_width=image_size, image_height=image_size, channel_size=num_channels,
        patch_size=4, latent_space_dim=emb_dim, dim_ff=dim_ff,
        num_heads=num_heads, depth=depth, num_classes=num_classes
    )
    
    # Manually setting model to eval mode and mocking state
    model.eval = MagicMock()
    model.forward = MagicMock(return_value=(torch.rand(batch_size, num_classes), torch.tensor(1.0)))

    evaluate_model(model, dataloader)

    assert model.eval.called