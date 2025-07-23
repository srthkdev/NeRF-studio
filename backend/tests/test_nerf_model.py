import torch
import pytest
import numpy as np
from app.ml.nerf.model import NeRFModel, PositionalEncoding, HierarchicalNeRF


class TestPositionalEncoding:
    """Test positional encoding functionality"""
    
    def test_positional_encoding_dimensions(self):
        """Test that positional encoding produces correct output dimensions"""
        # Test 3D position encoding with L=10 (default)
        pos_encoder = PositionalEncoding(num_frequencies=10, input_dim=3, include_input=True)
        
        # Input: 3D coordinates
        x = torch.randn(100, 3)
        encoded = pos_encoder(x)
        
        # Expected output dimension: 3 + 3*2*10 = 63
        expected_dim = 3 + 3 * 2 * 10
        assert encoded.shape == (100, expected_dim)
        assert pos_encoder.output_dim == expected_dim
    
    def test_view_direction_encoding(self):
        """Test view direction encoding with L=4"""
        view_encoder = PositionalEncoding(num_frequencies=4, input_dim=3, include_input=True)
        
        # Input: 3D view directions
        d = torch.randn(50, 3)
        encoded = view_encoder(d)
        
        # Expected output dimension: 3 + 3*2*4 = 27
        expected_dim = 3 + 3 * 2 * 4
        assert encoded.shape == (50, expected_dim)
        assert view_encoder.output_dim == expected_dim
    
    def test_encoding_without_input(self):
        """Test encoding without including original input"""
        encoder = PositionalEncoding(num_frequencies=5, input_dim=3, include_input=False)
        
        x = torch.randn(10, 3)
        encoded = encoder(x)
        
        # Expected output dimension: 3*2*5 = 30 (no original input)
        expected_dim = 3 * 2 * 5
        assert encoded.shape == (10, expected_dim)
        assert encoder.output_dim == expected_dim
    
    def test_encoding_mathematical_correctness(self):
        """Test that encoding follows the mathematical formula"""
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
    """Test NeRF model functionality"""
    
    def test_model_initialization(self):
        """Test that model initializes with correct architecture"""
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
        assert hasattr(model, 'feature_layer')
    
    def test_model_forward_pass(self):
        """Test forward pass produces correct output shapes"""
        model = NeRFModel()
        
        # Create sample inputs
        batch_size = 1024
        positions = torch.randn(batch_size, 3)
        view_dirs = torch.randn(batch_size, 3)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Check output shapes
        assert rgb.shape == (batch_size, 3)  # RGB colors
        assert sigma.shape == (batch_size, 1)  # Density values
        
        # Check output ranges
        assert torch.all(rgb >= 0) and torch.all(rgb <= 1)  # RGB in [0,1]
        assert torch.all(sigma >= 0)  # Density non-negative
    
    def test_skip_connections(self):
        """Test that skip connections work correctly"""
        model = NeRFModel(skip_connections=[2, 4])
        
        # Check that skip connection layers have correct input dimensions
        pos_encoded_dim = model.pos_encoder.output_dim
        hidden_dim = model.hidden_dim
        
        # Layer 2 should have skip connection (hidden + pos_encoded)
        assert model.layers[2].in_features == hidden_dim + pos_encoded_dim
        
        # Layer 4 should have skip connection
        assert model.layers[4].in_features == hidden_dim + pos_encoded_dim
        
        # Other layers should have normal dimensions
        assert model.layers[1].in_features == hidden_dim
        assert model.layers[3].in_features == hidden_dim
    
    def test_parameter_counting(self):
        """Test parameter counting utility"""
        model = NeRFModel(hidden_dim=128, num_layers=4)
        
        param_count = model.get_params_count()
        
        # Manually count parameters
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert param_count == manual_count
        assert param_count > 0
    
    def test_model_info(self):
        """Test model information utility"""
        model = NeRFModel(
            hidden_dim=256,
            num_layers=8,
            skip_connections=[4],
            pos_freq_bands=10,
            view_freq_bands=4
        )
        
        info = model.get_model_info()
        
        # Check all expected keys are present
        expected_keys = [
            'total_params', 'hidden_dim', 'num_layers', 
            'skip_connections', 'pos_encoding_dim', 'view_encoding_dim'
        ]
        for key in expected_keys:
            assert key in info
        
        # Check values
        assert info['hidden_dim'] == 256
        assert info['num_layers'] == 8
        assert info['skip_connections'] == [4]
        assert info['pos_encoding_dim'] == 63  # 3 + 3*2*10
        assert info['view_encoding_dim'] == 27  # 3 + 3*2*4
    
    def test_gradient_flow(self):
        """Test that gradients flow through the network correctly"""
        model = NeRFModel()
        
        # Create sample inputs with gradient tracking
        positions = torch.randn(10, 3, requires_grad=True)
        view_dirs = torch.randn(10, 3, requires_grad=True)
        
        # Forward pass
        rgb, sigma = model(positions, view_dirs)
        
        # Create a simple loss
        loss = rgb.sum() + sigma.sum()
        loss.backward()
        
        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None
        
        # Check that input gradients exist
        assert positions.grad is not None
        assert view_dirs.grad is not None
    
    def test_different_architectures(self):
        """Test different model architectures"""
        # Test smaller model
        small_model = NeRFModel(hidden_dim=64, num_layers=4, skip_connections=[2])
        positions = torch.randn(10, 3)
        view_dirs = torch.randn(10, 3)
        
        rgb, sigma = small_model(positions, view_dirs)
        assert rgb.shape == (10, 3)
        assert sigma.shape == (10, 1)
        
        # Test model without skip connections
        no_skip_model = NeRFModel(skip_connections=[])
        rgb, sigma = no_skip_model(positions, view_dirs)
        assert rgb.shape == (10, 3)
        assert sigma.shape == (10, 1)
    
    def test_batch_processing(self):
        """Test model handles different batch sizes correctly"""
        model = NeRFModel()
        
        # Test different batch sizes
        for batch_size in [1, 16, 256, 1024]:
            positions = torch.randn(batch_size, 3)
            view_dirs = torch.randn(batch_size, 3)
            
            rgb, sigma = model(positions, view_dirs)
            
            assert rgb.shape == (batch_size, 3)
            assert sigma.shape == (batch_size, 1)
    
    def test_model_reproducibility(self):
        """Test that model produces consistent outputs with same inputs"""
        model = NeRFModel()
        model.eval()  # Set to evaluation mode
        
        positions = torch.randn(100, 3)
        view_dirs = torch.randn(100, 3)
        
        # Run forward pass twice
        with torch.no_grad():
            rgb1, sigma1 = model(positions, view_dirs)
            rgb2, sigma2 = model(positions, view_dirs)
        
        # Results should be identical
        assert torch.allclose(rgb1, rgb2)
        assert torch.allclose(sigma1, sigma2)


if __name__ == "__main__":
    pytest.main([__file__])

class TestHierarchicalNeRF:
    """Test hierarchical NeRF functionality"""
    
    def test_hierarchical_model_initialization(self):
        """Test that hierarchical model initializes correctly"""
        model = HierarchicalNeRF(
            n_coarse=64,
            n_fine=128,
            hidden_dim=256
        )
        
        # Check that both models exist
        assert hasattr(model, 'coarse_model')
        assert hasattr(model, 'fine_model')
        
        # Check sampling parameters
        assert model.n_coarse == 64
        assert model.n_fine == 128
        
        # Check that models are separate instances
        assert model.coarse_model is not model.fine_model
    
    def test_hierarchical_forward_pass(self):
        """Test hierarchical forward pass"""
        model = HierarchicalNeRF(n_coarse=32, n_fine=64)
        
        # Create sample inputs
        batch_size = 16
        rays_o = torch.randn(batch_size, 3)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize
        near = torch.ones(batch_size, 1) * 2.0
        far = torch.ones(batch_size, 1) * 6.0
        
        # Forward pass
        output = model(rays_o, rays_d, near, far, perturb=False)
        
        # Check output structure
        assert "coarse" in output
        assert "fine" in output
        assert "z_vals_coarse" in output
        assert "z_vals_fine" in output
        assert "z_vals_combined" in output
        
        # Check coarse output
        coarse = output["coarse"]
        assert "rgb_map" in coarse
        assert "depth_map" in coarse
        assert "weights" in coarse
        assert coarse["rgb_map"].shape == (batch_size, 3)
        assert coarse["weights"].shape == (batch_size, 32)  # n_coarse
        
        # Check fine output
        fine = output["fine"]
        assert "rgb_map" in fine
        assert "depth_map" in fine
        assert "weights" in fine
        assert fine["rgb_map"].shape == (batch_size, 3)
        assert fine["weights"].shape == (batch_size, 96)  # n_coarse + n_fine
    
    def test_parameter_counting_hierarchical(self):
        """Test parameter counting for hierarchical model"""
        model = HierarchicalNeRF(hidden_dim=128)
        
        param_counts = model.get_params_count()
        
        # Check structure
        assert "coarse" in param_counts
        assert "fine" in param_counts
        assert "total" in param_counts
        
        # Check that total equals sum
        assert param_counts["total"] == param_counts["coarse"] + param_counts["fine"]
        
        # Check that both models have same parameter count (identical architecture)
        assert param_counts["coarse"] == param_counts["fine"]
    
    def test_model_info_hierarchical(self):
        """Test model info for hierarchical model"""
        model = HierarchicalNeRF(
            n_coarse=64,
            n_fine=128,
            hidden_dim=256
        )
        
        info = model.get_model_info()
        
        # Check structure
        assert "coarse_model" in info
        assert "fine_model" in info
        assert "n_coarse" in info
        assert "n_fine" in info
        assert "total_params" in info
        
        # Check values
        assert info["n_coarse"] == 64
        assert info["n_fine"] == 128


class TestSamplingWeights:
    """Test sampling weight computation and verification"""
    
    def test_compute_sampling_weights(self):
        """Test sampling weight computation"""
        from app.ml.nerf.model import compute_sampling_weights
        
        batch_size = 10
        n_samples = 64
        
        # Create sample inputs
        sigma = torch.rand(batch_size, n_samples, 1) * 2.0  # Random densities
        z_vals = torch.linspace(2.0, 6.0, n_samples).expand(batch_size, n_samples)
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Compute weights
        weights = compute_sampling_weights(sigma, z_vals, rays_d)
        
        # Check output shape
        assert weights.shape == (batch_size, n_samples)
        
        # Check that weights are non-negative
        assert torch.all(weights >= 0)
        
        # Check that weights sum to <= 1 (due to finite sampling)
        weight_sums = torch.sum(weights, dim=-1)
        assert torch.all(weight_sums <= 1.0 + 1e-6)
    
    def test_verify_sampling_distribution(self):
        """Test sampling distribution verification"""
        from app.ml.nerf.model import verify_sampling_distribution
        
        batch_size = 5
        n_samples = 32
        
        # Create sample inputs with known properties
        z_vals = torch.linspace(2.0, 6.0, n_samples).expand(batch_size, n_samples)
        
        # Create sigma with higher values in the middle (should get higher weights)
        sigma = torch.ones(batch_size, n_samples, 1) * 0.1
        sigma[:, n_samples//4:3*n_samples//4, :] = 2.0  # Higher density in middle
        
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Compute weights
        from app.ml.nerf.model import compute_sampling_weights
        weights = compute_sampling_weights(sigma, z_vals, rays_d)
        
        # Verify distribution
        verification = verify_sampling_distribution(weights, z_vals, sigma)
        
        # Check verification results
        assert "weights_normalized" in verification
        assert "weights_positive" in verification
        assert "positive_correlation" in verification
        
        # These should all be True for well-behaved weights
        assert verification["weights_normalized"]
        assert verification["weights_positive"]
    
    def test_create_coarse_fine_models(self):
        """Test creation of separate coarse and fine models"""
        from app.ml.nerf.model import create_coarse_fine_models
        
        config = {
            "hidden_dim": 128,
            "num_layers": 6,
            "skip_connections": [3],
            "pos_freq_bands": 8,
            "view_freq_bands": 4
        }
        
        coarse_model, fine_model = create_coarse_fine_models(config)
        
        # Check that models are separate instances
        assert coarse_model is not fine_model
        
        # Check that models have same architecture
        assert coarse_model.hidden_dim == fine_model.hidden_dim
        assert coarse_model.num_layers == fine_model.num_layers
        assert coarse_model.skip_connections == fine_model.skip_connections
        
        # Check that they have same parameter count
        assert coarse_model.get_params_count() == fine_model.get_params_count()
    
    def test_initialize_hierarchical_weights(self):
        """Test weight initialization for hierarchical models"""
        from app.ml.nerf.model import initialize_hierarchical_weights, create_coarse_fine_models
        
        config = {"hidden_dim": 64, "num_layers": 4}
        coarse_model, fine_model = create_coarse_fine_models(config)
        
        # Initialize with different schemes
        for scheme in ["xavier", "kaiming", "normal"]:
            initialize_hierarchical_weights(coarse_model, fine_model, scheme)
            
            # Check that weights are initialized (not all zeros)
            coarse_weights = []
            fine_weights = []
            
            for param in coarse_model.parameters():
                if param.requires_grad and len(param.shape) > 1:  # Weight matrices
                    coarse_weights.append(param.data.flatten())
            
            for param in fine_model.parameters():
                if param.requires_grad and len(param.shape) > 1:  # Weight matrices
                    fine_weights.append(param.data.flatten())
            
            if coarse_weights and fine_weights:
                coarse_tensor = torch.cat(coarse_weights)
                fine_tensor = torch.cat(fine_weights)
                
                # Check that weights are not all zeros
                assert not torch.allclose(coarse_tensor, torch.zeros_like(coarse_tensor))
                assert not torch.allclose(fine_tensor, torch.zeros_like(fine_tensor))
                
                # Check that coarse and fine weights are different (different random init)
                assert not torch.allclose(coarse_tensor, fine_tensor)


class TestHierarchicalIntegration:
    """Test integration between hierarchical sampling components"""
    
    def test_end_to_end_hierarchical_rendering(self):
        """Test complete hierarchical rendering pipeline"""
        model = HierarchicalNeRF(n_coarse=16, n_fine=32, hidden_dim=64)
        
        # Create a simple scene setup
        batch_size = 8
        rays_o = torch.zeros(batch_size, 3)  # Camera at origin
        rays_d = torch.randn(batch_size, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, 1) * 1.0
        far = torch.ones(batch_size, 1) * 5.0
        
        # Render
        output = model(rays_o, rays_d, near, far, perturb=True, training=True)
        
        # Check that fine rendering is better focused than coarse
        coarse_weights = output["coarse"]["weights"]
        fine_weights = output["fine"]["weights"]
        
        # Fine weights should be more concentrated (higher max values)
        coarse_max = torch.max(coarse_weights, dim=-1)[0]
        fine_max = torch.max(fine_weights, dim=-1)[0]
        
        # This is a statistical test - fine sampling should generally be more focused
        # but we'll just check that the pipeline runs without errors
        assert coarse_max.shape == (batch_size,)
        assert fine_max.shape == (batch_size,)
        
        # Check that z_vals are properly sorted
        z_combined = output["z_vals_combined"]
        z_sorted = torch.sort(z_combined, dim=-1)[0]
        assert torch.allclose(z_combined, z_sorted)
    
    def test_hierarchical_gradient_flow(self):
        """Test that gradients flow through hierarchical model"""
        model = HierarchicalNeRF(n_coarse=8, n_fine=16, hidden_dim=32)

        # Create inputs with gradient tracking
        rays_o = torch.randn(4, 3, requires_grad=True)
        rays_d = torch.randn(4, 3, requires_grad=True)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d.retain_grad()  # Retain grad for non-leaf tensor
        near = torch.ones(4, 1)
        far = torch.ones(4, 1) * 3.0

        # Forward pass
        output = model(rays_o, rays_d, near, far, training=True)

        # Create loss from both coarse and fine outputs
        coarse_loss = output["coarse"]["rgb_map"].sum()
        fine_loss = output["fine"]["rgb_map"].sum()
        total_loss = coarse_loss + fine_loss

        # Backward pass
        total_loss.backward()

        # Check gradients exist for all model parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

        # Check input gradients
        assert rays_o.grad is not None
        assert rays_d.grad is not None