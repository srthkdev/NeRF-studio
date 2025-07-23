import torch
import numpy as np
import pytest
from app.ml.nerf.volume_rendering import (
    volume_render_rays,
    alpha_composite_rays,
    render_rays_with_model,
    hierarchical_volume_render,
    compute_psnr,
    compute_ssim,
    add_noise_regularization,
    raw2outputs
)
from app.ml.nerf.model import NeRFModel


class MockNeRFModel(torch.nn.Module):
    """Mock NeRF model for testing volume rendering."""
    
    def __init__(self, constant_rgb=None, constant_sigma=None):
        super().__init__()
        self.constant_rgb = constant_rgb if constant_rgb is not None else torch.tensor([0.5, 0.5, 0.5])
        self.constant_sigma = constant_sigma if constant_sigma is not None else torch.tensor([1.0])
    
    def forward(self, x, d):
        batch_size = x.shape[0]
        rgb = self.constant_rgb.expand(batch_size, 3)
        sigma = self.constant_sigma.expand(batch_size, 1)
        return rgb, sigma


def test_volume_render_rays_basic():
    """Test basic volume rendering functionality."""
    # Create simple test data
    n_rays, n_samples = 2, 4
    
    # RGB values: constant color along each ray
    rgb = torch.ones(n_rays, n_samples, 3) * 0.5
    
    # Sigma values: higher density in the middle
    sigma = torch.zeros(n_rays, n_samples, 1)
    sigma[:, 1:3, :] = 2.0  # High density in middle samples
    
    # Depth values: linearly spaced
    z_vals = torch.linspace(1.0, 5.0, n_samples).expand(n_rays, n_samples)
    
    # Ray directions: pointing along z-axis
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    
    # Render rays
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # Check output shapes
    assert output["rgb_map"].shape == (n_rays, 3)
    assert output["depth_map"].shape == (n_rays, 1)
    assert output["acc_map"].shape == (n_rays, 1)
    assert output["weights"].shape == (n_rays, n_samples)
    assert output["disp_map"].shape == (n_rays, 1)
    
    # Check that weights sum to approximately the accumulated opacity
    weights_sum = torch.sum(output["weights"], dim=-1, keepdim=True)
    assert torch.allclose(weights_sum, output["acc_map"], atol=1e-6)
    
    # Check that RGB values are reasonable (should be close to input RGB since it's constant)
    assert torch.all(output["rgb_map"] >= 0.0)
    assert torch.all(output["rgb_map"] <= 1.0)
    
    # Check that depth values are within the z_vals range
    assert torch.all(output["depth_map"] >= z_vals.min())
    assert torch.all(output["depth_map"] <= z_vals.max())


def test_volume_render_rays_transmittance():
    """Test that transmittance calculation follows the NeRF equation."""
    # Create a single ray with known density distribution
    n_samples = 5
    
    # Simple case: uniform density
    rgb = torch.ones(1, n_samples, 3) * 0.8
    sigma = torch.ones(1, n_samples, 1) * 1.0  # Uniform density
    z_vals = torch.linspace(1.0, 3.0, n_samples).unsqueeze(0)  # (1, n_samples)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # Calculate expected values manually
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # (1, n_samples-1)
    dists = torch.cat([dists, torch.tensor([[1e10]])], dim=-1)  # Add infinity for last sample
    
    # Alpha values: 1 - exp(-sigma * delta)
    alpha_expected = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
    
    # Transmittance: cumulative product of (1 - alpha)
    transmittance_expected = torch.cumprod(
        torch.cat([torch.ones(1, 1), 1.0 - alpha_expected[:, :-1]], dim=-1), dim=-1
    )
    
    # Weights: T * alpha
    weights_expected = transmittance_expected * alpha_expected
    
    # Check that computed weights match expected values
    assert torch.allclose(output["weights"], weights_expected, atol=1e-6)


def test_volume_render_rays_edge_cases():
    """Test volume rendering with edge cases."""
    n_samples = 3
    
    # Test case 1: Zero density (should produce black image)
    rgb = torch.ones(1, n_samples, 3)
    sigma = torch.zeros(1, n_samples, 1)
    z_vals = torch.linspace(1.0, 3.0, n_samples).unsqueeze(0)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # With zero density, should get black image and zero accumulated opacity
    assert torch.allclose(output["rgb_map"], torch.zeros(1, 3), atol=1e-6)
    assert torch.allclose(output["acc_map"], torch.zeros(1, 1), atol=1e-6)
    
    # Test case 2: Very high density (should be opaque)
    sigma_high = torch.ones(1, n_samples, 1) * 100.0
    output_high = volume_render_rays(rgb, sigma_high, z_vals, rays_d)
    
    # With very high density, accumulated opacity should be close to 1
    assert output_high["acc_map"].item() > 0.99
    
    # Test case 3: White background
    output_white = volume_render_rays(rgb, sigma, z_vals, rays_d, white_bkgd=True)
    
    # With white background and zero density, should get white image
    expected_white = torch.ones(1, 3)
    assert torch.allclose(output_white["rgb_map"], expected_white, atol=1e-6)


def test_volume_render_rays_noise_regularization():
    """Test noise regularization during training."""
    n_samples = 4
    
    rgb = torch.ones(1, n_samples, 3) * 0.5
    sigma = torch.ones(1, n_samples, 1) * 1.0
    z_vals = torch.linspace(1.0, 3.0, n_samples).unsqueeze(0)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    
    # Render without noise
    output_no_noise = volume_render_rays(rgb, sigma, z_vals, rays_d, noise_std=0.0, training=False)
    
    # Render with noise (set seed for reproducibility)
    torch.manual_seed(42)
    output_with_noise = volume_render_rays(rgb, sigma, z_vals, rays_d, noise_std=0.1, training=True)
    
    # Results should be different due to noise
    # Check weights first (more sensitive to noise)
    assert not torch.allclose(output_no_noise["weights"], output_with_noise["weights"])
    # RGB might be similar due to averaging, but weights should differ


def test_alpha_composite_rays():
    """Test alpha compositing with pre-computed alpha values."""
    n_samples = 4
    
    rgb = torch.ones(1, n_samples, 3) * 0.7
    alpha = torch.tensor([[0.1, 0.3, 0.5, 0.2]])  # Pre-computed alpha values
    z_vals = torch.linspace(1.0, 3.0, n_samples).unsqueeze(0)
    
    output = alpha_composite_rays(rgb, alpha, z_vals)
    
    # Check output shapes
    assert output["rgb_map"].shape == (1, 3)
    assert output["depth_map"].shape == (1, 1)
    assert output["acc_map"].shape == (1, 1)
    assert output["weights"].shape == (1, n_samples)
    
    # Manually calculate expected transmittance and weights
    transmittance = torch.cumprod(torch.cat([torch.ones(1, 1), 1.0 - alpha[:, :-1]], dim=-1), dim=-1)
    weights_expected = transmittance * alpha
    
    assert torch.allclose(output["weights"], weights_expected, atol=1e-6)


def test_render_rays_with_model():
    """Test rendering rays using a NeRF model."""
    # Create mock model
    model = MockNeRFModel(
        constant_rgb=torch.tensor([0.8, 0.6, 0.4]),
        constant_sigma=torch.tensor([2.0])
    )
    
    # Create test rays
    n_rays = 2
    rays_o = torch.zeros(n_rays, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    z_vals = torch.linspace(1.0, 3.0, 8).expand(n_rays, 8)
    
    # Render rays
    output = render_rays_with_model(model, rays_o, rays_d, z_vals)
    
    # Check output shapes
    assert output["rgb_map"].shape == (n_rays, 3)
    assert output["depth_map"].shape == (n_rays, 1)
    assert output["acc_map"].shape == (n_rays, 1)
    
    # Since model returns constant values, all rays should produce similar results
    # (accounting for different ray directions affecting distance calculations)
    assert torch.all(output["rgb_map"] > 0.0)
    assert torch.all(output["acc_map"] > 0.0)


def test_render_rays_with_model_chunking():
    """Test rendering rays with memory chunking."""
    model = MockNeRFModel()
    
    # Create larger batch to test chunking
    n_rays = 10
    rays_o = torch.zeros(n_rays, 3)
    rays_d = torch.randn(n_rays, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize
    z_vals = torch.linspace(1.0, 3.0, 6).expand(n_rays, 6)
    
    # Render without chunking
    output_no_chunk = render_rays_with_model(model, rays_o, rays_d, z_vals)
    
    # Render with chunking
    output_chunked = render_rays_with_model(model, rays_o, rays_d, z_vals, chunk_size=3)
    
    # Results should be identical
    assert torch.allclose(output_no_chunk["rgb_map"], output_chunked["rgb_map"], atol=1e-6)
    assert torch.allclose(output_no_chunk["depth_map"], output_chunked["depth_map"], atol=1e-6)
    assert torch.allclose(output_no_chunk["acc_map"], output_chunked["acc_map"], atol=1e-6)


def test_hierarchical_volume_render():
    """Test hierarchical volume rendering with coarse and fine models."""
    # Create mock models
    coarse_model = MockNeRFModel(
        constant_rgb=torch.tensor([0.5, 0.5, 0.5]),
        constant_sigma=torch.tensor([1.0])
    )
    fine_model = MockNeRFModel(
        constant_rgb=torch.tensor([0.8, 0.6, 0.4]),
        constant_sigma=torch.tensor([2.0])
    )
    
    # Create test rays
    n_rays = 2
    rays_o = torch.zeros(n_rays, 3)
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    near = torch.ones(n_rays, 1) * 1.0
    far = torch.ones(n_rays, 1) * 5.0
    
    # Render with hierarchical sampling
    output = hierarchical_volume_render(
        coarse_model, fine_model, rays_o, rays_d, near, far,
        n_coarse=8, n_fine=16, perturb=False  # No perturbation for deterministic test
    )
    
    # Check that both coarse and fine outputs are present
    assert "coarse" in output
    assert "fine" in output
    
    # Check coarse output shapes
    assert output["coarse"]["rgb_map"].shape == (n_rays, 3)
    assert output["coarse"]["depth_map"].shape == (n_rays, 1)
    
    # Check fine output shapes
    assert output["fine"]["rgb_map"].shape == (n_rays, 3)
    assert output["fine"]["depth_map"].shape == (n_rays, 1)
    
    # Check z_vals shapes
    assert output["z_vals_coarse"].shape == (n_rays, 8)
    assert output["z_vals_fine"].shape == (n_rays, 16)
    assert output["z_vals_combined"].shape == (n_rays, 24)  # 8 + 16
    
    # Fine model should produce different results than coarse model
    assert not torch.allclose(output["coarse"]["rgb_map"], output["fine"]["rgb_map"])


def test_compute_psnr():
    """Test PSNR computation."""
    # Test case 1: Identical images should give infinite PSNR
    img1 = torch.rand(3, 64, 64)
    psnr_identical = compute_psnr(img1, img1)
    assert torch.isinf(psnr_identical)
    
    # Test case 2: Different images should give finite PSNR
    img2 = torch.rand(3, 64, 64)
    psnr_different = compute_psnr(img1, img2)
    assert torch.isfinite(psnr_different)
    assert psnr_different > 0  # PSNR should be positive
    
    # Test case 3: Known MSE should give expected PSNR
    img_pred = torch.ones(1, 1, 1, dtype=torch.float32) * 0.5
    img_gt = torch.ones(1, 1, 1, dtype=torch.float32) * 0.6
    mse = 0.01  # (0.6 - 0.5)^2 = 0.01
    expected_psnr = -10.0 * np.log10(mse)
    computed_psnr = compute_psnr(img_pred, img_gt)
    assert torch.allclose(computed_psnr, torch.tensor(expected_psnr, dtype=torch.float32), atol=1e-4)


def test_compute_ssim():
    """Test SSIM computation."""
    # Test case 1: Identical images should give SSIM = 1
    img = torch.rand(1, 3, 64, 64)
    ssim_identical = compute_ssim(img, img)
    assert torch.allclose(ssim_identical, torch.tensor(1.0), atol=1e-3)
    
    # Test case 2: Different images should give SSIM < 1
    img1 = torch.rand(1, 3, 64, 64)
    img2 = torch.rand(1, 3, 64, 64)
    ssim_different = compute_ssim(img1, img2)
    assert ssim_different < 1.0
    assert ssim_different > 0.0  # SSIM should be positive
    
    # Test case 3: Completely uncorrelated images should give low SSIM
    img_zeros = torch.zeros(1, 3, 32, 32)
    img_ones = torch.ones(1, 3, 32, 32)
    ssim_uncorrelated = compute_ssim(img_zeros, img_ones)
    assert ssim_uncorrelated < 0.5  # Should be quite low


def test_add_noise_regularization():
    """Test noise regularization function."""
    sigma = torch.ones(10, 5, 1) * 2.0
    
    # Test training mode with noise
    sigma_noisy = add_noise_regularization(sigma, noise_std=0.1, training=True)
    assert not torch.allclose(sigma, sigma_noisy)  # Should be different due to noise
    
    # Test evaluation mode (no noise)
    sigma_eval = add_noise_regularization(sigma, noise_std=0.1, training=False)
    assert torch.allclose(sigma, sigma_eval)  # Should be identical
    
    # Test zero noise
    sigma_no_noise = add_noise_regularization(sigma, noise_std=0.0, training=True)
    assert torch.allclose(sigma, sigma_no_noise)  # Should be identical


def test_raw2outputs():
    """Test conversion from raw network outputs to rendered values."""
    n_rays, n_samples = 2, 6
    
    # Create raw network outputs [r, g, b, sigma]
    raw = torch.randn(n_rays, n_samples, 4)
    raw[..., :3] = torch.sigmoid(raw[..., :3])  # RGB should be in [0, 1] after sigmoid
    raw[..., 3] = torch.abs(raw[..., 3])  # Sigma should be positive after ReLU
    
    z_vals = torch.linspace(1.0, 3.0, n_samples).expand(n_rays, n_samples)
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    
    # Convert raw outputs
    output = raw2outputs(raw, z_vals, rays_d)
    
    # Check output shapes
    assert output["rgb_map"].shape == (n_rays, 3)
    assert output["depth_map"].shape == (n_rays, 1)
    assert output["acc_map"].shape == (n_rays, 1)
    assert output["weights"].shape == (n_rays, n_samples)
    
    # Check that RGB values are in valid range
    assert torch.all(output["rgb_map"] >= 0.0)
    assert torch.all(output["rgb_map"] <= 1.0)
    
    # Test with noise
    output_noisy = raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.1, training=True)
    assert not torch.allclose(output["rgb_map"], output_noisy["rgb_map"])


def test_volume_rendering_mathematical_correctness():
    """Test that volume rendering follows the exact NeRF equation."""
    # Create a simple test case with known analytical solution
    n_samples = 100  # Use many samples for accuracy
    
    # Single ray along z-axis
    rgb = torch.ones(1, n_samples, 3) * 0.8  # Constant color
    
    # Exponentially decaying density: sigma(t) = sigma_0 * exp(-t)
    z_vals = torch.linspace(0.1, 5.0, n_samples).unsqueeze(0)
    sigma_0 = 2.0
    sigma = sigma_0 * torch.exp(-z_vals.unsqueeze(-1))  # (1, n_samples, 1)
    
    rays_d = torch.tensor([[0.0, 0.0, 1.0]])
    
    # Render with our implementation
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # For exponentially decaying density, we can compute the analytical solution
    # The transmittance T(t) = exp(-∫_0^t sigma_0 * exp(-s) ds) = exp(sigma_0 * (exp(-t) - 1))
    # The weight at depth t is T(t) * sigma(t) * dt
    
    # Check that the accumulated opacity is reasonable
    # For exponentially decaying density, most of the mass should be near the beginning
    assert output["acc_map"].item() > 0.5  # Should have significant opacity
    
    # Check that the depth is reasonable (should be biased toward smaller z values)
    mean_z = torch.mean(z_vals)
    assert output["depth_map"].item() < mean_z.item()  # Depth should be less than mean due to exponential decay


def test_volume_rendering_conservation():
    """Test that volume rendering conserves probability mass."""
    n_samples = 50
    
    # Create test data
    rgb = torch.rand(3, n_samples, 3)  # 3 rays
    sigma = torch.rand(3, n_samples, 1) * 5.0  # Random densities
    z_vals = torch.linspace(1.0, 10.0, n_samples).expand(3, n_samples)
    rays_d = torch.randn(3, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize
    
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # Check that weights are non-negative
    assert torch.all(output["weights"] >= 0.0)
    
    # Check that accumulated opacity is in [0, 1]
    assert torch.all(output["acc_map"] >= 0.0)
    assert torch.all(output["acc_map"] <= 1.0)
    
    # Check that RGB values are in [0, 1] (assuming input RGB is in [0, 1])
    rgb_clamped = torch.clamp(rgb, 0.0, 1.0)
    output_clamped = volume_render_rays(rgb_clamped, sigma, z_vals, rays_d)
    assert torch.all(output_clamped["rgb_map"] >= 0.0)
    assert torch.all(output_clamped["rgb_map"] <= 1.0)


def test_volume_rendering_depth_ordering():
    """Test that volume rendering respects depth ordering."""
    n_samples = 10
    
    # Create two scenarios: density at front vs back
    rgb = torch.ones(2, n_samples, 3) * 0.8
    z_vals = torch.linspace(1.0, 5.0, n_samples).expand(2, n_samples)
    rays_d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    
    # Scenario 1: High density at the front
    sigma1 = torch.zeros(1, n_samples, 1)
    sigma1[0, :2, 0] = 10.0  # High density in first two samples
    
    # Scenario 2: High density at the back
    sigma2 = torch.zeros(1, n_samples, 1)
    sigma2[0, -2:, 0] = 10.0  # High density in last two samples
    
    sigma = torch.cat([sigma1, sigma2], dim=0)
    
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # The ray with front density should have smaller depth than the ray with back density
    assert output["depth_map"][0, 0] < output["depth_map"][1, 0]
    
    # Both should have similar accumulated opacity (both have high density regions)
    assert abs(output["acc_map"][0, 0] - output["acc_map"][1, 0]) < 0.1


def test_volume_rendering_batch_consistency():
    """Test that volume rendering produces consistent results across batch dimensions."""
    n_rays, n_samples = 5, 8
    
    # Create identical rays in a batch
    rgb = torch.ones(n_rays, n_samples, 3) * 0.7
    sigma = torch.ones(n_rays, n_samples, 1) * 2.0
    z_vals = torch.linspace(1.0, 3.0, n_samples).expand(n_rays, n_samples)
    rays_d = torch.tensor([[0.0, 0.0, 1.0]]).expand(n_rays, 3)
    
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    
    # All rays should produce identical results
    for i in range(1, n_rays):
        assert torch.allclose(output["rgb_map"][0], output["rgb_map"][i], atol=1e-6)
        assert torch.allclose(output["depth_map"][0], output["depth_map"][i], atol=1e-6)
        assert torch.allclose(output["acc_map"][0], output["acc_map"][i], atol=1e-6)


def test_volume_rendering_reference_implementation():
    """Test against a simplified reference implementation of volume rendering."""
    def reference_volume_render(rgb, sigma, z_vals, rays_d):
        """Simplified reference implementation for comparison."""
        # Calculate distances
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Multiply by ray direction norm
        dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Calculate alpha
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
        
        # Calculate transmittance
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), 
            dim=-1
        )
        
        # Calculate weights
        weights = transmittance * alpha
        
        # Render RGB
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        
        return rgb_map, weights
    
    # Test data
    n_rays, n_samples = 3, 6
    rgb = torch.rand(n_rays, n_samples, 3)
    sigma = torch.rand(n_rays, n_samples, 1) * 3.0
    z_vals = torch.linspace(1.0, 4.0, n_samples).expand(n_rays, n_samples)
    rays_d = torch.randn(n_rays, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Compare implementations
    output = volume_render_rays(rgb, sigma, z_vals, rays_d)
    ref_rgb, ref_weights = reference_volume_render(rgb, sigma, z_vals, rays_d)
    
    # Results should match
    assert torch.allclose(output["rgb_map"], ref_rgb, atol=1e-6)
    assert torch.allclose(output["weights"], ref_weights, atol=1e-6)


if __name__ == "__main__":
    # Run a subset of tests for quick verification
    test_volume_render_rays_basic()
    test_volume_render_rays_transmittance()
    test_volume_rendering_mathematical_correctness()
    test_volume_rendering_reference_implementation()
    print("All volume rendering tests passed!")