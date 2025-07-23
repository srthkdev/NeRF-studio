import torch
import numpy as np
import pytest
from backend.app.ml.nerf.rays import (
    generate_rays,
    generate_rays_from_camera,
    sample_stratified,
    sample_hierarchical,
    create_ray_batch,
    create_efficient_ray_batch,
    combine_hierarchical_samples,
    sample_pdf,
    generate_camera_rays
)


def test_generate_rays_shape():
    """Test that the ray generation produces the correct output shapes."""
    height, width = 10, 15
    
    # Create test camera parameters
    intrinsics = torch.tensor([
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Identity rotation, camera at origin
    extrinsics = torch.eye(4)
    
    rays = generate_rays(height, width, intrinsics, extrinsics)
    
    # Check output shapes
    assert rays["origins"].shape == (height, width, 3)
    assert rays["directions"].shape == (height, width, 3)
    assert rays["near"].shape == (height, width, 1)
    assert rays["far"].shape == (height, width, 1)


def test_ray_direction_normalization():
    """Test that ray directions are properly normalized."""
    height, width = 10, 15
    
    # Create test camera parameters
    intrinsics = torch.tensor([
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Identity rotation, camera at origin
    extrinsics = torch.eye(4)
    
    rays = generate_rays(height, width, intrinsics, extrinsics)
    
    # Check that directions are normalized
    directions_norm = torch.norm(rays["directions"], dim=-1)
    assert torch.allclose(directions_norm, torch.ones_like(directions_norm), atol=1e-6)


def test_ray_origins():
    """Test that ray origins match the camera position."""
    height, width = 10, 15
    
    # Create test camera parameters
    intrinsics = torch.tensor([
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Camera at position (1, 2, 3)
    translation = torch.tensor([1.0, 2.0, 3.0])
    rotation = torch.eye(3)
    extrinsics = torch.eye(4)
    extrinsics[:3, 3] = translation
    
    rays = generate_rays(height, width, intrinsics, extrinsics)
    
    # Check that all ray origins match the camera position
    expected_origins = translation.view(1, 1, 3).expand(height, width, 3)
    assert torch.allclose(rays["origins"], expected_origins)


def test_stratified_sampling():
    """Test stratified sampling along rays."""
    # Create simple rays
    origins = torch.zeros(2, 3)  # 2 rays starting at origin
    directions = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # pointing in z and y directions
    near = torch.ones(2, 1) * 1.0
    far = torch.ones(2, 1) * 5.0
    n_samples = 10
    
    # Test without perturbation for deterministic results
    points, z_vals = sample_stratified(origins, directions, near, far, n_samples, perturb=False)
    
    # Check shapes
    assert points.shape == (2, n_samples, 3)
    assert z_vals.shape == (2, n_samples)
    
    # Check z values are linearly spaced between near and far
    expected_z_vals = torch.linspace(1.0, 5.0, n_samples)
    assert torch.allclose(z_vals[0], expected_z_vals)
    assert torch.allclose(z_vals[1], expected_z_vals)
    
    # Check points lie on the rays
    # For ray 1 (along z-axis), x and y should be 0, z should match z_vals
    assert torch.allclose(points[0, :, 0], torch.zeros(n_samples))
    assert torch.allclose(points[0, :, 1], torch.zeros(n_samples))
    assert torch.allclose(points[0, :, 2], z_vals[0])
    
    # For ray 2 (along y-axis), x and z should be 0, y should match z_vals
    assert torch.allclose(points[1, :, 0], torch.zeros(n_samples))
    assert torch.allclose(points[1, :, 1], z_vals[1])
    assert torch.allclose(points[1, :, 2], torch.zeros(n_samples))


def test_hierarchical_sampling():
    """Test hierarchical sampling based on weights."""
    # Create simple rays
    origins = torch.zeros(1, 3)  # 1 ray starting at origin
    directions = torch.tensor([[0.0, 0.0, 1.0]])  # pointing in z direction
    
    # Create coarse samples
    n_coarse = 8
    z_vals = torch.linspace(2.0, 8.0, n_coarse).view(1, n_coarse)
    
    # Create weights that focus on the middle of the ray
    weights = torch.zeros(1, n_coarse)
    weights[0, 3:5] = 1.0  # Higher weights in the middle
    
    # Test hierarchical sampling
    n_fine = 16
    points, z_vals_fine = sample_hierarchical(origins, directions, z_vals, weights, n_fine, perturb=False)
    
    # Check shapes
    assert points.shape == (1, n_fine, 3)
    assert z_vals_fine.shape == (1, n_fine)
    
    # Check that fine samples are concentrated where weights are high
    # Count samples in the middle region
    middle_region = (z_vals_fine >= z_vals[0, 3]) & (z_vals_fine <= z_vals[0, 4])
    middle_count = middle_region.sum().item()
    
    # We expect more samples in the middle region than in other regions
    assert middle_count > n_fine / 4  # At least 25% of samples should be in the middle region


def test_ray_batching():
    """Test ray batching for efficient processing."""
    # Create a small ray dictionary
    height, width = 4, 5
    rays_dict = {
        "origins": torch.randn(height, width, 3),
        "directions": torch.randn(height, width, 3),
        "near": torch.ones(height, width, 1),
        "far": torch.ones(height, width, 1) * 5.0
    }
    
    # Test batching
    batch_size = 5
    batches = create_ray_batch(rays_dict, batch_size)
    
    # Check number of batches
    total_rays = height * width
    expected_batches = (total_rays + batch_size - 1) // batch_size  # Ceiling division
    assert len(batches) == expected_batches
    
    # Check batch sizes
    for i, batch in enumerate(batches):
        if i < len(batches) - 1:
            assert batch["origins"].shape[0] == batch_size
        else:
            # Last batch might be smaller
            assert batch["origins"].shape[0] <= batch_size
        
        # Check that all components have the same batch size
        assert batch["directions"].shape[0] == batch["origins"].shape[0]
        assert batch["near"].shape[0] == batch["origins"].shape[0]
        assert batch["far"].shape[0] == batch["origins"].shape[0]


def test_generate_rays_from_camera():
    """Test ray generation from simplified camera parameters."""
    height, width = 10, 15
    focal_length = 1000.0
    
    # Camera at position (1, 2, 3) looking at origin
    camera_position = torch.tensor([1.0, 2.0, 3.0])
    camera_rotation = torch.eye(3)  # Identity rotation
    
    rays = generate_rays_from_camera(height, width, focal_length, camera_position, camera_rotation)
    
    # Check output shapes
    assert rays["origins"].shape == (height, width, 3)
    assert rays["directions"].shape == (height, width, 3)
    
    # Check that all ray origins match the camera position
    expected_origins = camera_position.view(1, 1, 3).expand(height, width, 3)
    assert torch.allclose(rays["origins"], expected_origins)
    
    # Check that directions are normalized
    directions_norm = torch.norm(rays["directions"], dim=-1)
    assert torch.allclose(directions_norm, torch.ones_like(directions_norm), atol=1e-6)


def test_combine_hierarchical_samples():
    """Test combining coarse and fine samples."""
    # Create simple test data
    n_coarse, n_fine = 8, 16
    z_vals_coarse = torch.linspace(2.0, 8.0, n_coarse).view(1, n_coarse)
    z_vals_fine = torch.linspace(3.0, 7.0, n_fine).view(1, n_fine)
    
    # Create points along z-axis
    points_coarse = torch.zeros(1, n_coarse, 3)
    points_coarse[..., 2] = z_vals_coarse
    
    points_fine = torch.zeros(1, n_fine, 3)
    points_fine[..., 2] = z_vals_fine
    
    # Combine samples
    z_vals_combined, points_combined = combine_hierarchical_samples(
        z_vals_coarse, z_vals_fine, points_coarse, points_fine
    )
    
    # Check shapes
    assert z_vals_combined.shape == (1, n_coarse + n_fine)
    assert points_combined.shape == (1, n_coarse + n_fine, 3)
    
    # Check that z values are sorted
    assert torch.all(z_vals_combined[:, 1:] >= z_vals_combined[:, :-1])
    
    # Check that points correspond to z values
    assert torch.allclose(points_combined[..., 2], z_vals_combined)
    
    # Check that x and y coordinates are still zero
    assert torch.allclose(points_combined[..., :2], torch.zeros_like(points_combined[..., :2]))


def test_sample_pdf():
    """Test sampling from a PDF."""
    # Create bins and weights
    bins = torch.linspace(2.0, 8.0, 10).view(1, 10)
    weights = torch.zeros(1, 9)
    weights[0, 4:6] = 1.0  # Higher weights in the middle
    
    # Sample from PDF
    n_samples = 100
    samples = sample_pdf(bins, weights, n_samples, perturb=False)
    
    # Check shape
    assert samples.shape == (1, n_samples)
    
    # Check that samples are within the bin range
    assert torch.all(samples >= bins.min())
    assert torch.all(samples <= bins.max())
    
    # Check that samples are concentrated where weights are high
    middle_region = (samples >= bins[0, 4]) & (samples <= bins[0, 6])
    middle_count = middle_region.sum().item()
    
    # We expect most samples to be in the middle region
    assert middle_count > n_samples * 0.7  # At least 70% of samples should be in the middle region


def test_efficient_ray_batch():
    """Test efficient ray batching with DataLoader."""
    # Create a small ray dictionary
    height, width = 4, 5
    rays_dict = {
        "origins": torch.randn(height, width, 3),
        "directions": torch.randn(height, width, 3),
        "near": torch.ones(height, width, 1),
        "far": torch.ones(height, width, 1) * 5.0
    }
    
    # Add RGB values for training
    rays_dict["rgb"] = torch.rand(height, width, 3)
    
    # Create efficient ray batch
    batch_size = 5
    ray_loader = create_efficient_ray_batch(rays_dict, batch_size, include_rgb=True)
    
    # Check that we can iterate through the loader
    batch_count = 0
    total_rays = 0
    
    for batch in ray_loader:
        batch_count += 1
        total_rays += batch["origins"].shape[0]
        
        # Check that all components have the same batch size
        assert batch["directions"].shape[0] == batch["origins"].shape[0]
        assert batch["near"].shape[0] == batch["origins"].shape[0]
        assert batch["far"].shape[0] == batch["origins"].shape[0]
        assert batch["rgb"].shape[0] == batch["origins"].shape[0]
    
    # Check that we processed all rays
    assert total_rays == height * width


def test_generate_camera_rays():
    """Test generating rays for multiple camera poses."""
    # Create camera poses
    n_cameras = 3
    camera_poses = torch.eye(4).unsqueeze(0).repeat(n_cameras, 1, 1)
    
    # Set different camera positions
    for i in range(n_cameras):
        camera_poses[i, :3, 3] = torch.tensor([i, i, i])
    
    # Generate rays
    height, width = 10, 15
    focal_length = 1000.0
    rays = generate_camera_rays(camera_poses, height, width, focal_length)
    
    # Check shapes
    assert rays["origins"].shape[0] == n_cameras * height * width
    assert rays["directions"].shape[0] == n_cameras * height * width
    
    # Check camera indices
    camera_indices = rays["camera_idx"].reshape(-1)
    for i in range(n_cameras):
        # Each camera should have height*width rays
        assert torch.sum(camera_indices == i) == height * width


def test_ray_direction_distribution():
    """Test that ray directions cover the expected field of view."""
    height, width = 100, 100
    focal_length = 100.0  # Shorter focal length for wider FOV
    
    # Camera at origin looking along z-axis
    camera_position = torch.zeros(3)
    camera_rotation = torch.eye(3)
    
    rays = generate_rays_from_camera(height, width, focal_length, camera_position, camera_rotation)
    
    # Calculate field of view
    fov_x = 2 * np.arctan(width / (2 * focal_length))
    fov_y = 2 * np.arctan(height / (2 * focal_length))
    
    # Check that ray directions span the expected field of view
    directions = rays["directions"]
    
    # Calculate angles from z-axis
    cos_angles_x = directions[:, width//2, 2]  # Central vertical column
    angles_x = torch.acos(cos_angles_x)
    max_angle_x = angles_x.max().item()
    
    cos_angles_y = directions[height//2, :, 2]  # Central horizontal row
    angles_y = torch.acos(cos_angles_y)
    max_angle_y = angles_y.max().item()
    
    # Check that maximum angles are close to half the FOV
    assert abs(max_angle_x - fov_y/2) < 0.1
    assert abs(max_angle_y - fov_x/2) < 0.1


def test_stratified_sampling_distribution():
    """Test that stratified sampling produces the expected distribution."""
    # Create simple rays
    origins = torch.zeros(1, 3)
    directions = torch.tensor([[0.0, 0.0, 1.0]])
    near = torch.ones(1, 1) * 1.0
    far = torch.ones(1, 1) * 5.0
    n_samples = 1000
    
    # Sample with perturbation for statistical testing
    points, z_vals = sample_stratified(origins, directions, near, far, n_samples, perturb=True)
    
    # Check that z values are within the near-far range
    assert torch.all(z_vals >= near)
    assert torch.all(z_vals <= far)
    
    # Check that the distribution is roughly uniform
    # Divide the range into bins and count samples in each bin
    n_bins = 10
    bin_edges = torch.linspace(1.0, 5.0, n_bins + 1)
    bin_counts = torch.zeros(n_bins)
    
    for i in range(n_bins):
        bin_counts[i] = ((z_vals >= bin_edges[i]) & (z_vals < bin_edges[i+1])).sum()
    
    # Expected count per bin for uniform distribution
    expected_count = n_samples / n_bins
    
    # Check that counts are roughly uniform (within 30% of expected)
    assert torch.all(bin_counts > expected_count * 0.7)
    assert torch.all(bin_counts < expected_count * 1.3)


def test_hierarchical_sampling_distribution():
    """Test that hierarchical sampling concentrates samples according to weights."""
    # Create simple rays
    origins = torch.zeros(1, 3)
    directions = torch.tensor([[0.0, 0.0, 1.0]])
    
    # Create coarse samples
    n_coarse = 16
    z_vals = torch.linspace(1.0, 5.0, n_coarse).view(1, n_coarse)
    
    # Create weights that focus on a specific region
    weights = torch.zeros(1, n_coarse)
    target_region = (n_coarse // 4, 3 * n_coarse // 4)  # Middle half
    weights[0, target_region[0]:target_region[1]] = 1.0
    
    # Sample with hierarchical sampling
    n_fine = 1000
    _, z_vals_fine = sample_hierarchical(origins, directions, z_vals, weights, n_fine, perturb=False)
    
    # Define the target region in z-space
    target_z_min = z_vals[0, target_region[0]]
    target_z_max = z_vals[0, target_region[1] - 1]
    
    # Count samples in the target region
    in_target = ((z_vals_fine >= target_z_min) & (z_vals_fine <= target_z_max)).sum().item()
    
    # We expect most samples to be in the target region
    target_ratio = in_target / n_fine
    assert target_ratio > 0.8  # At least 80% of samples should be in the target region