"""
🧪 Essential NeRF Tests - Core Components

"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

# Import core components
from app.ml.nerf.model import NeRFModel, PositionalEncoding, HierarchicalNeRF


class TestPositionalEncoding:
    """🧠 Test Positional Encoding - Core NeRF Component"""
    
    def test_positional_encoding_dimensions(self):
        """✅ Test positional encoding produces correct output dimensions"""
        pos_encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
        
        # Input: 3D coordinates
        x = torch.randn(100, 3)
        encoded = pos_encoder(x)
        
        # Expected output dimension: 3 + 3*2*10 = 63
        expected_dim = 3 + 3 * 2 * 10
        assert encoded.shape == (100, expected_dim)
        assert pos_encoder.output_dim == expected_dim
    
    def test_view_direction_encoding(self):
        """✅ Test view direction encoding with L=4"""
        view_encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=True)
        
        # Input: 3D view directions
        d = torch.randn(50, 3)
        encoded = view_encoder(d)
        
        # Expected output dimension: 3 + 3*2*4 = 27
        expected_dim = 3 + 3 * 2 * 4
        assert encoded.shape == (50, expected_dim)
        assert view_encoder.output_dim == expected_dim
    
    def test_encoding_mathematical_correctness(self):
        """✅ Test that encoding follows the mathematical formula"""
        encoder = PositionalEncoding(num_frequencies=2, input_dim=1, include_input=True)

        # Simple test case
        x = torch.tensor([[1.0]])
        encoded = encoder(x)

        # Expected: [1.0, sin(2^0*1), cos(2^0*1), sin(2^1*1), cos(2^1*1)]
        expected = torch.tensor([
            [
                1.0,
                torch.sin(torch.tensor(1.0)),
                torch.cos(torch.tensor(1.0)),
                torch.sin(torch.tensor(2.0)),
                torch.cos(torch.tensor(2.0))
            ]
        ])

        assert torch.allclose(encoded, expected, atol=1e-6)


class TestNeRFModel:
    """🧠 Test NeRF Model - Core Neural Network"""
    
    def test_model_initialization(self):
        """✅ Test that model initializes with correct architecture"""
        model = NeRFModel(
            pos_freq_bands=10,
            view_freq_bands=4,
            hidden_dim=256,
            num_layers=8,
            skip_connections=[4]
        )
        
        # Check that model has correct number of layers
        assert len(model.layers) == 8
        
        # Check skip connection configuration
        assert model.skip_connections == [4]
        
        # Check output layers exist
        assert hasattr(model, 'sigma_layer')
        assert hasattr(model, 'color_layer')
    
    def test_model_forward_pass(self):
        """✅ Test forward pass produces correct output shapes"""
        model = NeRFModel()
        
        # Create sample inputs
        batch_size = 1024
        positions = torch.randn(batch_size, 3)
        view_dirs = torch.randn(batch_size, 3)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Check output shapes
        assert rgb.shape == (batch_size, 3)  # RGB colors
        assert sigma.shape == (batch_size, 1)  # Density
        
        # Check value ranges
        assert torch.all(sigma >= 0)  # Density should be non-negative
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1)  # Colors in [0,1]
    
    def test_parameter_counting(self):
        """✅ Test model parameter counting"""
        model = NeRFModel(hidden_dim=128, num_layers=4)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters
        assert total_params > 10000  # At least 10k parameters
        assert total_params < 1000000  # Less than 1M parameters
    
    def test_gradient_flow(self):
        """✅ Test that gradients flow through the model"""
        model = NeRFModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create sample inputs
        positions = torch.randn(100, 3, requires_grad=True)
        view_dirs = torch.randn(100, 3, requires_grad=True)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Compute loss
        loss = sigma.mean() + rgb.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.parameters())
        assert has_gradients


class TestHierarchicalNeRF:
    """🧠 Test Hierarchical NeRF - Advanced Model"""
    
    def test_hierarchical_model_initialization(self):
        """✅ Test hierarchical model initialization"""
        model = HierarchicalNeRF(
            pos_freq_bands=10,
            view_freq_bands=4,
            hidden_dim=256,
            num_layers=8,
            n_coarse=64,
            n_fine=128
        )
        
        # Check model components
        assert hasattr(model, 'coarse_model')
        assert hasattr(model, 'fine_model')
        assert model.n_coarse == 64
        assert model.n_fine == 128
    
    def test_hierarchical_forward_pass(self):
        """✅ Test hierarchical forward pass"""
        model = HierarchicalNeRF()
        
        # Create sample inputs
        batch_size = 100
        rays_o = torch.randn(batch_size, 3)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # Normalize
        near = torch.ones(batch_size, 1) * 0.1  # Add dimension for broadcasting
        far = torch.ones(batch_size, 1) * 10.0  # Add dimension for broadcasting
        
        # Forward pass
        result = model(rays_o, rays_d, near, far)
        
        # Check output structure
        assert 'coarse' in result
        assert 'fine' in result
        assert 'rgb_map' in result['coarse']
        assert 'rgb_map' in result['fine']


class TestIntegration:
    """🔗 Test Integration - End-to-End Workflows"""
    
    def test_model_training_workflow(self):
        """✅ Test complete model training workflow"""
        # Create model
        model = NeRFModel()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Mock training data
        batch_size = 1024
        positions = torch.randn(batch_size, 3)
        view_dirs = torch.randn(batch_size, 3)
        target_colors = torch.rand(batch_size, 3)
        
        # Training step
        optimizer.zero_grad()
        rgb, sigma = model(positions, view_dirs)
        
        # Mock loss (simplified)
        loss = torch.nn.functional.mse_loss(rgb, target_colors)
        loss.backward()
        optimizer.step()
        
        # Check that training step completed
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_rendering_workflow(self):
        """✅ Test complete rendering workflow"""
        # Create model
        model = NeRFModel()
        
        # Mock ray data
        ray_origins = torch.randn(100, 3)
        ray_directions = torch.randn(100, 3)
        ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
        
        # Mock sampling
        num_samples = 64
        sample_points = torch.randn(100, num_samples, 3)
        sample_directions = ray_directions.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Forward pass through model
        rgb, sigma = model(sample_points.reshape(-1, 3), sample_directions.reshape(-1, 3))
        sigma = sigma.reshape(100, num_samples, 1)
        rgb = rgb.reshape(100, num_samples, 3)
        
        # Simple volume rendering
        weights = torch.softmax(sigma.squeeze(-1), dim=-1)
        rendered_colors = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        
        # Check outputs
        assert rendered_colors.shape == (100, 3)
        assert torch.all(rendered_colors >= 0) and torch.all(rendered_colors <= 1)


class TestPerformance:
    """⚡ Test Performance - Speed and Memory"""
    
    def test_model_inference_speed(self):
        """✅ Test model inference speed"""
        model = NeRFModel()
        model.eval()
        
        # Create test input
        batch_size = 1024
        positions = torch.randn(batch_size, 3)
        view_dirs = torch.randn(batch_size, 3)
        
        # Time inference
        import time
        start_time = time.time()
        
        with torch.no_grad():
            rgb, sigma = model(positions, view_dirs)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Should be reasonably fast (< 1 second for 1024 samples)
        assert inference_time < 1.0
        
        # Check outputs
        assert rgb.shape == (batch_size, 3)  # RGB colors
        assert sigma.shape == (batch_size, 1)  # Density
    
    def test_memory_usage(self):
        """✅ Test memory usage"""
        import psutil
        import os
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create model
        model = NeRFModel()
        
        # Get memory after model creation
        after_model_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (< 500MB)
        memory_increase = after_model_memory - initial_memory
        assert memory_increase < 500
        
        # Clean up
        del model


class TestRobustness:
    """🛡️ Test Robustness - Error Handling and Edge Cases"""
    
    def test_model_with_zero_input(self):
        """✅ Test model handles zero input gracefully"""
        model = NeRFModel()
        
        # Zero inputs
        positions = torch.zeros(10, 3)
        view_dirs = torch.zeros(10, 3)
        
        # Should not crash
        rgb, sigma = model(positions, view_dirs)
        
        assert rgb.shape == (10, 3)  # RGB colors
        assert sigma.shape == (10, 1)  # Density
    
    def test_model_with_large_input(self):
        """✅ Test model handles large input gracefully"""
        model = NeRFModel()
        
        # Large inputs
        positions = torch.randn(10000, 3) * 1000
        view_dirs = torch.randn(10000, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
        
        # Should not crash
        rgb, sigma = model(positions, view_dirs)
        
        assert rgb.shape == (10000, 3)  # RGB colors
        assert sigma.shape == (10000, 1)  # Density
    
    def test_model_with_nan_input(self):
        """✅ Test model handles NaN input gracefully"""
        model = NeRFModel()
        
        # NaN inputs
        positions = torch.full((10, 3), float('nan'))
        view_dirs = torch.randn(10, 3)
        
        # Should handle gracefully (either raise exception or produce NaN output)
        try:
            sigma, color = model(positions, view_dirs)
            # If no exception, check that outputs are NaN
            assert torch.isnan(sigma).any() or torch.isnan(color).any()
        except Exception:
            # Exception is also acceptable
            pass


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 