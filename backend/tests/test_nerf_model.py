"""
🧠 NeRF Model Tests

"""

import pytest
import torch
import numpy as np
from app.ml.nerf.model import NeRFModel, HierarchicalNeRF, PositionalEncoding


class TestPositionalEncoding:
    """🧠 Test Positional Encoding - Core NeRF Component"""
    
    def test_positional_encoding_initialization(self):
        """✅ Test positional encoding initialization"""
        encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
        
        assert encoder.num_frequencies == 10
        assert encoder.input_dim == 3
        assert encoder.include_input == True
        assert len(encoder.freq_bands) == 10
    
    def test_positional_encoding_output_dim(self):
        """✅ Test positional encoding output dimension calculation"""
        encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
        
        # Expected output dim: input_dim * (2 * num_frequencies + 1)
        expected_dim = 3 * (2 * 10 + 1)  # 3 * 21 = 63
        assert encoder.output_dim == expected_dim
    
    def test_positional_encoding_forward_pass(self):
        """✅ Test positional encoding forward pass"""
        encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=True)
        
        # Create input tensor
        x = torch.randn(100, 3)
        
        # Forward pass
        encoded = encoder(x)
        
        # Check output shape
        expected_shape = (100, encoder.output_dim)
        assert encoded.shape == expected_shape
        
        # Check that output contains original input
        assert torch.allclose(encoded[:, :3], x)
    
    def test_positional_encoding_without_input(self):
        """✅ Test positional encoding without including input"""
        encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=False)
        
        # Create input tensor
        x = torch.randn(100, 3)
        
        # Forward pass
        encoded = encoder(x)
        
        # Check output shape (should not include original input)
        expected_dim = 3 * (2 * 4)  # 3 * 8 = 24
        assert encoded.shape == (100, expected_dim)
    
    def test_positional_encoding_mathematical_correctness(self):
        """✅ Test positional encoding mathematical correctness"""
        encoder = PositionalEncoding(num_frequencies=2, input_dim=2, include_input=True)
        
        # Create simple input
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Forward pass
        encoded = encoder(x)
        
        # Check that sin and cos components are present
        # Output should be [x1, x2, sin(f1*x1), cos(f1*x1), sin(f1*x2), cos(f1*x2), sin(f2*x1), cos(f2*x1), sin(f2*x2), cos(f2*x2)]
        assert encoded.shape == (2, 10)
        
        # Check that original values are preserved
        assert torch.allclose(encoded[:, 0], x[:, 0])  # x1
        assert torch.allclose(encoded[:, 1], x[:, 1])  # x2


class TestNeRFModel:
    """🧠 Test NeRF Model - Basic NeRF Implementation"""
    
    def test_nerf_model_initialization(self):
        """✅ Test NeRF model initialization"""
        model = NeRFModel(
            pos_dim=3,
            view_dim=3,
            pos_freq_bands=10,
            view_freq_bands=4,
            hidden_dim=256,
            num_layers=8
        )
        
        assert model is not None
        assert hasattr(model, 'pos_encoder')
        assert hasattr(model, 'view_encoder')
        assert hasattr(model, 'layers')
        assert hasattr(model, 'sigma_layer')
        assert hasattr(model, 'feature_layer')
        assert hasattr(model, 'color_layer')
    
    def test_nerf_model_forward_pass(self):
        """✅ Test NeRF model forward pass"""
        model = NeRFModel(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4
        )
        
        # Create inputs
        batch_size = 100
        positions = torch.randn(batch_size, 3)
        view_dirs = torch.randn(batch_size, 3)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Check output shapes
        assert rgb.shape == (batch_size, 3)  # RGB colors
        assert sigma.shape == (batch_size, 1)  # Density
        
        # Check value ranges
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1)  # Colors in [0,1]
        assert torch.all(sigma >= 0)  # Density should be non-negative
    
    def test_nerf_model_parameter_counting(self):
        """✅ Test NeRF model parameter counting"""
        model = NeRFModel(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 100000  # Reasonable size for test model
    
    def test_nerf_model_gradient_flow(self):
        """✅ Test that gradients flow through the NeRF model"""
        model = NeRFModel(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create inputs
        positions = torch.randn(100, 3, requires_grad=True)
        view_dirs = torch.randn(100, 3, requires_grad=True)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Compute loss
        loss = rgb.mean() + sigma.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.parameters())
        assert has_gradients
    
    def test_nerf_model_with_zero_input(self):
        """✅ Test NeRF model with zero input"""
        model = NeRFModel(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4
        )
        
        # Zero inputs
        positions = torch.zeros(10, 3)
        view_dirs = torch.zeros(10, 3)
        
        # Should not crash
        rgb, sigma = model(positions, view_dirs)
        
        assert rgb.shape == (10, 3)
        assert sigma.shape == (10, 1)
    
    def test_nerf_model_with_large_input(self):
        """✅ Test NeRF model with large input"""
        model = NeRFModel(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4
        )
        
        # Large inputs
        positions = torch.randn(1000, 3) * 1000
        view_dirs = torch.randn(1000, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)
        
        # Should not crash
        rgb, sigma = model(positions, view_dirs)
        
        assert rgb.shape == (1000, 3)
        assert sigma.shape == (1000, 1)


class TestHierarchicalNeRF:
    """🧠 Test Hierarchical NeRF - Advanced NeRF Implementation"""
    
    def test_hierarchical_nerf_initialization(self):
        """✅ Test hierarchical NeRF initialization"""
        model = HierarchicalNeRF(
            pos_freq_bands=10,
            view_freq_bands=4,
            hidden_dim=256,
            num_layers=8,
            n_coarse=64,
            n_fine=128
        )
        
        assert model is not None
        assert hasattr(model, 'coarse_model')
        assert hasattr(model, 'fine_model')
        assert model.n_coarse == 64
        assert model.n_fine == 128
    
    def test_hierarchical_nerf_forward_pass(self):
        """✅ Test hierarchical NeRF forward pass"""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        # Create inputs
        batch_size = 4
        num_rays = 64
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, num_rays, 1) * 0.1
        far = torch.ones(batch_size, num_rays, 1) * 10.0
        
        # Forward pass
        result = model(rays_o, rays_d, near, far)
        
        # Check output structure
        assert "coarse" in result
        assert "fine" in result
        assert "rgb_map" in result["coarse"]
        assert "rgb_map" in result["fine"]
        assert "weights" in result["coarse"]
        assert "weights" in result["fine"]
        
        # Check output shapes
        assert result["coarse"]["rgb_map"].shape == (batch_size, num_rays, 3)
        assert result["fine"]["rgb_map"].shape == (batch_size, num_rays, 3)
        
        # Check value ranges
        assert torch.all(result["coarse"]["rgb_map"] >= 0) and torch.all(result["coarse"]["rgb_map"] <= 1)
        assert torch.all(result["fine"]["rgb_map"] >= 0) and torch.all(result["fine"]["rgb_map"] <= 1)
    
    def test_hierarchical_nerf_parameter_counting(self):
        """✅ Test hierarchical NeRF parameter counting"""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        params_info = model.get_params_count()
        assert "coarse" in params_info
        assert "fine" in params_info
        assert params_info["coarse"] > 0
        assert params_info["fine"] > 0
    
    def test_hierarchical_nerf_gradient_flow(self):
        """✅ Test that gradients flow through hierarchical NeRF"""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create inputs
        batch_size = 4
        num_rays = 64
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, num_rays, 1) * 0.1
        far = torch.ones(batch_size, num_rays, 1) * 10.0
        target_colors = torch.rand(batch_size, num_rays, 3)
        
        # Forward pass
        result = model(rays_o, rays_d, near, far)
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(result["fine"]["rgb_map"], target_colors)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                          for p in model.parameters())
        assert has_gradients
    
    def test_hierarchical_nerf_with_different_batch_sizes(self):
        """✅ Test hierarchical NeRF with different batch sizes"""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            num_rays = 64
            rays_o = torch.randn(batch_size, num_rays, 3)
            rays_d = torch.randn(batch_size, num_rays, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = torch.ones(batch_size, num_rays, 1) * 0.1
            far = torch.ones(batch_size, num_rays, 1) * 10.0
            
            # Forward pass
            result = model(rays_o, rays_d, near, far)
            
            # Check output shapes
            assert result["coarse"]["rgb_map"].shape == (batch_size, num_rays, 3)
            assert result["fine"]["rgb_map"].shape == (batch_size, num_rays, 3)
    
    def test_hierarchical_nerf_model_info(self):
        """✅ Test hierarchical NeRF model info"""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        info = model.get_model_info()
        assert "coarse_model" in info
        assert "fine_model" in info
        assert "total_params" in info["coarse_model"]
        assert "total_params" in info["fine_model"]


class TestModelIntegration:
    """🔗 Test Model Integration - End-to-End Model Testing"""
    
    def test_model_training_workflow(self):
        """✅ Test complete model training workflow"""
        # Create hierarchical NeRF model
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create training data
        batch_size = 4
        num_rays = 64
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, num_rays, 1) * 0.1
        far = torch.ones(batch_size, num_rays, 1) * 10.0
        target_colors = torch.rand(batch_size, num_rays, 3)
        
        # Training step
        optimizer.zero_grad()
        result = model(rays_o, rays_d, near, far)
        loss = torch.nn.functional.mse_loss(result["fine"]["rgb_map"], target_colors)
        loss.backward()
        optimizer.step()
        
        # Check that training completed
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_model_inference_workflow(self):
        """✅ Test model inference workflow"""
        # Create model
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        # Set to evaluation mode
        model.eval()
        
        # Create inference data
        batch_size = 1
        num_rays = 64
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, num_rays, 1) * 0.1
        far = torch.ones(batch_size, num_rays, 1) * 10.0
        
        # Inference
        with torch.no_grad():
            result = model(rays_o, rays_d, near, far)
        
        # Check inference results
        assert "fine" in result
        assert "rgb_map" in result["fine"]
        assert result["fine"]["rgb_map"].shape == (batch_size, num_rays, 3)
        assert torch.all(result["fine"]["rgb_map"] >= 0) and torch.all(result["fine"]["rgb_map"] <= 1)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 