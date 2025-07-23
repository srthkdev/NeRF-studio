import torch
import numpy as np
import pytest
from app.ml.nerf.model import PositionalEncoding


def test_positional_encoding_output_shape():
    """Test that the positional encoding produces the correct output shape."""
    # Test with 3D input (position)
    pos_encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
    x = torch.randn(5, 3)  # Batch of 5 3D positions
    
    encoded = pos_encoder(x)
    
    # Expected output dimension: 3 * (1 + 2*10) = 3 * 21 = 63
    assert encoded.shape == (5, 63)
    
    # Test with 3D input (view direction)
    view_encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=True)
    d = torch.randn(5, 3)  # Batch of 5 3D directions
    
    encoded = view_encoder(d)
    
    # Expected output dimension: 3 * (1 + 2*4) = 3 * 9 = 27
    assert encoded.shape == (5, 27)


def test_positional_encoding_output_dim_property():
    """Test that the output_dim property returns the correct value."""
    # Position encoding (3D -> 63D)
    pos_encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
    assert pos_encoder.output_dim == 63
    
    # View direction encoding (3D -> 27D)
    view_encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=True)
    assert view_encoder.output_dim == 27
    
    # Test without including input
    encoder = PositionalEncoding(num_frequencies=5, input_dim=2, include_input=False)
    assert encoder.output_dim == 20  # 2 * (2*5) = 20


def test_positional_encoding_mathematical_correctness():
    """Test the mathematical correctness of the positional encoding."""
    # Create a simple 1D input for easier verification
    encoder = PositionalEncoding(num_frequencies=3, input_dim=1, include_input=True)
    x = torch.tensor([[1.0]])
    
    encoded = encoder(x)
    
    # Expected output:
    # [x, sin(2^0*π*x), cos(2^0*π*x), sin(2^1*π*x), cos(2^1*π*x), sin(2^2*π*x), cos(2^2*π*x)]
    # = [1.0, sin(1.0), cos(1.0), sin(2.0), cos(2.0), sin(4.0), cos(4.0)]
    expected = torch.cat([
        x,
        torch.sin(x * 2**0),
        torch.cos(x * 2**0),
        torch.sin(x * 2**1),
        torch.cos(x * 2**1),
        torch.sin(x * 2**2),
        torch.cos(x * 2**2)
    ], dim=-1)
    
    assert torch.allclose(encoded, expected)


def test_positional_encoding_input_validation():
    """Test that the positional encoding validates input dimensions."""
    encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
    
    # Correct input dimension
    x = torch.randn(5, 3)
    encoder(x)  # Should not raise an error
    
    # Incorrect input dimension
    x_wrong = torch.randn(5, 4)
    with pytest.raises(ValueError):
        encoder(x_wrong)


def test_positional_encoding_frequency_bands():
    """Test that frequency bands are correctly generated."""
    encoder = PositionalEncoding(num_frequencies=5, input_dim=3, include_input=True)
    
    # Check frequency bands: [2^0, 2^1, 2^2, 2^3, 2^4]
    expected_bands = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    assert torch.allclose(encoder.freq_bands, expected_bands)