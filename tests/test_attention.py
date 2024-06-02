import pytest
import torch
import numpy as np
import logging

from src.attention import SelfAttentionLayer, MultiHeadSelfAttentionLayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@pytest.fixture(scope="class")
def dummy_data(request):
    """Fixture to create dummy data for testing."""
    batch_size = 8
    context_length = 16
    latent_dim = 32
    dummy_input = torch.tensor(np.random.rand(batch_size, context_length, latent_dim))
    request.cls.dummy_input = dummy_input

@pytest.fixture(scope="class")
def self_attention_layer(request):
    """Fixture to initialize the SelfAttentionLayer."""
    layer = SelfAttentionLayer(dim_in=32, dim_out=32)
    request.cls.self_attention_layer = layer

@pytest.fixture(scope="class")
def multihead_self_attention_layer(request):
    """Fixture to initialize the MultiHeadSelfAttentionLayer."""
    layer = MultiHeadSelfAttentionLayer(dim_in=32, dim_out=32, num_heads=4, context_length=16)
    request.cls.multihead_self_attention_layer = layer

@pytest.mark.usefixtures("dummy_data", "self_attention_layer", "multihead_self_attention_layer")
class TestSelfAttentionLayers:
    """Test class for Self-Attention Layers functionality."""

    def test_self_attention_forward_shape(self):
        """Test the shape of the output from the SelfAttentionLayer."""
        output = self.self_attention_layer(self.dummy_input)
        expected_shape = self.dummy_input.shape
        logging.info(f"Self-Attention Layer output shape: {output.shape}")
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}."

    def test_multihead_self_attention_forward_shape(self):
        """Test the shape of the output from the MultiHeadSelfAttentionLayer."""
        output = self.multihead_self_attention_layer(self.dummy_input)
        expected_shape = self.dummy_input.shape
        logging.info(f"Multi-Head Self-Attention Layer output shape: {output.shape}")
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
