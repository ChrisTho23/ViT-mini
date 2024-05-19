import pytest
import torch

from src.model_components import FeedForward, ClassificationHead, TransformerBlock

@pytest.fixture
def synthetic_data():
    """Fixture to create synthetic data for testing."""
    batch_size = 16
    emb_dim = 64
    context_length = 10
    data = torch.rand(batch_size, context_length, emb_dim, dtype=torch.float)
    return data

@pytest.fixture
def feedforward():
    """Fixture to initialize the FeedForward layer."""
    dim_in = 64
    dim_ff = 128
    dim_out = 64
    return FeedForward(dim_in=dim_in, dim_ff=dim_ff, dim_out=dim_out, dtype=torch.float)

@pytest.fixture
def classification_head():
    """Fixture to initialize the ClassificationHead."""
    dim_in = 64
    dim_ff = 128
    dim_out = 10
    return ClassificationHead(dim_in=dim_in, dim_ff=dim_ff, dim_out=dim_out, dtype=torch.float)

@pytest.fixture
def transformer_block():
    """Fixture to initialize the TransformerBlock."""
    emb_dim = 64
    dim_ff = 128
    num_heads = 8
    context_length = 10
    return TransformerBlock(
        emb_dim=emb_dim, 
        dim_ff=dim_ff, 
        num_heads=num_heads, 
        context_length=context_length, 
        dtype=torch.float
    )

def test_feedforward(feedforward, synthetic_data):
    """Test the FeedForward layer."""
    output = feedforward(synthetic_data)
    expected_shape = synthetic_data.shape
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_classification_head(classification_head, synthetic_data):
    """Test the ClassificationHead."""
    output = classification_head(synthetic_data)
    expected_shape = (synthetic_data.shape[0], synthetic_data.shape[1], 10)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_transformer_block(transformer_block, synthetic_data):
    """Test the TransformerBlock."""
    output = transformer_block(synthetic_data)
    expected_shape = synthetic_data.shape
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
