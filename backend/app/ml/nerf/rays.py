import torch
import numpy as np
from typing import Tuple, Dict, Optional, Union, List


def generate_rays(
    height: int,
    width: int,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Generate rays for each pixel in an image.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        intrinsics: Camera intrinsic matrix of shape (3, 3)
        extrinsics: Camera extrinsic matrix of shape (4, 4)
        near: Near plane distance
        far: Far plane distance
        device: Device to place tensors on
        
    Returns:
        Dictionary containing:
            - origins: Ray origins of shape (height, width, 3)
            - directions: Ray directions of shape (height, width, 3)
            - near: Near plane distance for each ray (height, width, 1)
            - far: Far plane distance for each ray (height, width, 1)
    """
    # Move tensors to the specified device
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)
    
    # Create pixel coordinates grid
    i, j = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Convert pixel coordinates to normalized device coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Convert pixel coordinates to camera coordinates
    x = (j - cx) / fx
    y = (i - cy) / fy
    z = torch.ones_like(x)
    
    # Stack to create direction vectors in camera space
    directions_cam = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    
    # Extract rotation matrix and translation vector from extrinsics
    rotation = extrinsics[:3, :3]  # (3, 3)
    translation = extrinsics[:3, 3]  # (3,)
    
    # Convert directions from camera space to world space
    directions = torch.matmul(directions_cam.view(-1, 3), rotation.T).view(height, width, 3)
    
    # Normalize ray directions
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    
    # Ray origins are the camera position in world space
    origins = translation.view(1, 1, 3).expand(height, width, 3)
    
    # Create near and far tensors
    near_tensor = torch.ones_like(origins[..., :1]) * near
    far_tensor = torch.ones_like(origins[..., :1]) * far
    
    return {
        "origins": origins,
        "directions": directions,
        "near": near_tensor,
        "far": far_tensor
    }


def generate_rays_from_camera(
    height: int,
    width: int,
    focal_length: float,
    camera_position: torch.Tensor,
    camera_rotation: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Generate rays from simplified camera parameters.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Focal length in pixels
        camera_position: Camera position in world space (3,)
        camera_rotation: Camera rotation matrix (3, 3)
        near: Near plane distance
        far: Far plane distance
        device: Device to place tensors on
        
    Returns:
        Dictionary containing ray information
    """
    # Create intrinsic matrix
    intrinsics = torch.eye(3, device=device)
    intrinsics[0, 0] = focal_length
    intrinsics[1, 1] = focal_length
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    
    # Create extrinsic matrix
    extrinsics = torch.eye(4, device=device)
    extrinsics[:3, :3] = camera_rotation
    extrinsics[:3, 3] = camera_position
    
    return generate_rays(height, width, intrinsics, extrinsics, near, far, device)


def sample_stratified(
    origins: torch.Tensor,
    directions: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    n_samples: int,
    perturb: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stratified sampling along rays.
    
    Args:
        origins: Ray origins of shape (..., 3)
        directions: Ray directions of shape (..., 3)
        near: Near plane distance of shape (..., 1)
        far: Far plane distance of shape (..., 1)
        n_samples: Number of samples per ray
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Tuple of:
            - points: Sampled points of shape (..., n_samples, 3)
            - z_vals: Depth values of shape (..., n_samples)
    """
    # Create stratified depth values
    t_vals = torch.linspace(0., 1., n_samples, device=origins.device)
    
    # Sample linearly between near and far
    z_vals = near * (1. - t_vals) + far * t_vals
    
    # Expand z_vals to match the shape of origins
    z_vals = z_vals.expand(list(origins.shape[:-1]) + [n_samples])
    
    # Add random perturbation if requested
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        
        # Stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=origins.device)
        z_vals = lower + (upper - lower) * t_rand
    
    # Expand dimensions for broadcasting
    origins_expanded = origins.unsqueeze(-2)  # (..., 1, 3)
    directions_expanded = directions.unsqueeze(-2)  # (..., 1, 3)
    z_vals_expanded = z_vals.unsqueeze(-1)  # (..., n_samples, 1)
    
    # Compute 3D sample points: r(t) = o + t*d
    points = origins_expanded + directions_expanded * z_vals_expanded
    
    return points, z_vals


def sample_hierarchical(
    origins: torch.Tensor,
    directions: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hierarchical sampling along rays based on weights from coarse model.
    
    Args:
        origins: Ray origins of shape (..., 3)
        directions: Ray directions of shape (..., 3)
        z_vals: Depth values from coarse sampling of shape (..., n_coarse)
        weights: Weights from coarse model of shape (..., n_coarse)
        n_samples: Number of additional samples per ray
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Tuple of:
            - points: Sampled points of shape (..., n_samples, 3)
            - z_vals_fine: Depth values of shape (..., n_samples)
    """
    # Create PDF from weights
    weights = weights + 1e-5  # Prevent division by zero
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Create CDF from PDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (..., n_coarse+1)
    
    # Take uniform samples
    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=origins.device)
    else:
        u = torch.linspace(0., 1., n_samples, device=origins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    
    # Invert CDF to find sample locations
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    
    # Clamp indices to valid range for z_vals (which has n_coarse elements)
    below = torch.clamp(inds - 1, min=0, max=z_vals.shape[-1] - 1)
    above = torch.clamp(inds, min=0, max=z_vals.shape[-1] - 1)
    
    # Get surrounding CDF values
    inds_g = torch.stack([below, above], dim=-1)  # (..., n_samples, 2)
    
    # Gather corresponding CDF and z_vals
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, 
                         index=inds_g)  # (..., n_samples, 2)
    
    matched_shape = list(inds_g.shape[:-1]) + [z_vals.shape[-1]]
    z_vals_g = torch.gather(z_vals.unsqueeze(-2).expand(matched_shape), dim=-1, 
                           index=inds_g)  # (..., n_samples, 2)
    
    # Linear interpolation
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    z_vals_fine = z_vals_g[..., 0] + t * (z_vals_g[..., 1] - z_vals_g[..., 0])
    
    # Expand dimensions for broadcasting
    origins_expanded = origins.unsqueeze(-2)  # (..., 1, 3)
    directions_expanded = directions.unsqueeze(-2)  # (..., 1, 3)
    z_vals_expanded = z_vals_fine.unsqueeze(-1)  # (..., n_samples, 1)
    
    # Compute 3D sample points: r(t) = o + t*d
    points = origins_expanded + directions_expanded * z_vals_expanded
    
    return points, z_vals_fine


def combine_hierarchical_samples(
    z_vals_coarse: torch.Tensor,
    z_vals_fine: torch.Tensor,
    points_coarse: torch.Tensor,
    points_fine: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combine coarse and fine samples for hierarchical sampling.
    
    Args:
        z_vals_coarse: Depth values from coarse sampling of shape (..., n_coarse)
        z_vals_fine: Depth values from fine sampling of shape (..., n_fine)
        points_coarse: Points from coarse sampling of shape (..., n_coarse, 3)
        points_fine: Points from fine sampling of shape (..., n_fine, 3)
        
    Returns:
        Tuple of:
            - z_vals_combined: Combined depth values of shape (..., n_coarse + n_fine)
            - points_combined: Combined points of shape (..., n_coarse + n_fine, 3)
    """
    # Combine z values and sort
    z_vals_combined = torch.cat([z_vals_coarse, z_vals_fine], dim=-1)  # (..., n_coarse + n_fine)
    _, indices = torch.sort(z_vals_combined, dim=-1)
    
    # Sort z values
    z_vals_combined = torch.gather(z_vals_combined, dim=-1, index=indices)
    
    # Combine points
    points_combined = torch.cat([points_coarse, points_fine], dim=-2)  # (..., n_coarse + n_fine, 3)
    
    # Expand indices for gathering points
    indices_expanded = indices.unsqueeze(-1).expand(*indices.shape, 3)
    
    # Sort points
    points_combined = torch.gather(points_combined, dim=-2, index=indices_expanded)
    
    return z_vals_combined, points_combined


def create_ray_batch(
    rays_dict: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """
    Create batches of rays for efficient GPU processing.
    
    Args:
        rays_dict: Dictionary containing ray information (origins, directions, etc.)
        batch_size: Number of rays in each batch
        shuffle: Whether to shuffle rays before batching
        
    Returns:
        List of dictionaries containing batched ray information
    """
    # Flatten ray tensors
    origins = rays_dict["origins"].reshape(-1, 3)
    directions = rays_dict["directions"].reshape(-1, 3)
    near = rays_dict["near"].reshape(-1, 1)
    far = rays_dict["far"].reshape(-1, 1)
    
    # Get total number of rays
    n_rays = origins.shape[0]
    
    # Create indices for batching
    indices = torch.arange(n_rays, device=origins.device)
    if shuffle:
        indices = indices[torch.randperm(n_rays, device=origins.device)]
    
    # Create batches
    batches = []
    for i in range(0, n_rays, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = {
            "origins": origins[batch_indices],
            "directions": directions[batch_indices],
            "near": near[batch_indices],
            "far": far[batch_indices]
        }
        batches.append(batch)
    
    return batches


def create_efficient_ray_batch(
    rays_dict: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = True,
    include_rgb: bool = False
) -> torch.utils.data.DataLoader:
    """
    Create an efficient DataLoader for ray batches.
    
    Args:
        rays_dict: Dictionary containing ray information (origins, directions, etc.)
        batch_size: Number of rays in each batch
        shuffle: Whether to shuffle rays before batching
        include_rgb: Whether to include RGB values in the batch (for training)
        
    Returns:
        DataLoader for efficient ray batch processing
    """
    # Flatten ray tensors
    origins = rays_dict["origins"].reshape(-1, 3)
    directions = rays_dict["directions"].reshape(-1, 3)
    near = rays_dict["near"].reshape(-1, 1)
    far = rays_dict["far"].reshape(-1, 1)
    
    # Create dataset
    ray_data = {
        "origins": origins,
        "directions": directions,
        "near": near,
        "far": far
    }
    
    if include_rgb and "rgb" in rays_dict:
        ray_data["rgb"] = rays_dict["rgb"].reshape(-1, 3)
    
    # Create TensorDataset
    tensor_list = list(ray_data.values())
    dataset = torch.utils.data.TensorDataset(*tensor_list)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # No multiprocessing for GPU tensors
        pin_memory=True  # Pin memory for faster GPU transfer
    )
    
    # Create a wrapper to convert back to dictionary format
    keys = list(ray_data.keys())
    
    class RayBatchLoader:
        def __init__(self, dataloader, keys):
            self.dataloader = dataloader
            self.keys = keys
            
        def __iter__(self):
            for batch in self.dataloader:
                yield {k: v for k, v in zip(self.keys, batch)}
                
        def __len__(self):
            return len(self.dataloader)
    
    return RayBatchLoader(dataloader, keys)


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb: bool = True
) -> torch.Tensor:
    """
    Sample from a 1D probability density function (PDF) defined by weights.
    
    Args:
        bins: Bin locations of shape (..., n_bins)
        weights: Weights for each bin of shape (..., n_bins-1)
        n_samples: Number of samples to draw
        perturb: Whether to add random perturbation to samples
        
    Returns:
        Sampled values of shape (..., n_samples)
    """
    # Add a small epsilon to weights to prevent NaNs
    weights = weights + 1e-5
    
    # Normalize weights to get PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Get CDF from PDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # (..., n_bins)
    
    # Take uniform samples
    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=bins.device)
    else:
        u = torch.linspace(0., 1., n_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    
    # Invert CDF to find sample locations
    u = u.contiguous()
    
    # Find indices of bins that u falls into
    inds = torch.searchsorted(cdf, u, right=True)
    
    # Clamp indices to valid range
    below = torch.clamp(inds - 1, min=0, max=cdf.shape[-1] - 1)
    above = torch.clamp(inds, min=0, max=cdf.shape[-1] - 1)
    
    # Get surrounding CDF and bin values
    inds_g = torch.stack([below, above], dim=-1)  # (..., n_samples, 2)
    
    # Gather corresponding CDF values
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, 
                         index=inds_g)  # (..., n_samples, 2)
    
    # Gather corresponding bin values
    matched_shape = list(inds_g.shape[:-1]) + [bins.shape[-1]]
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1, 
                         index=inds_g)  # (..., n_samples, 2)
    
    # Linear interpolation
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples


def generate_camera_rays(
    camera_poses: torch.Tensor,
    height: int,
    width: int,
    focal_length: float,
    near: float = 2.0,
    far: float = 6.0,
    device: torch.device = torch.device("cpu")
) -> Dict[str, torch.Tensor]:
    """
    Generate rays for multiple camera poses.
    
    Args:
        camera_poses: Camera poses of shape (n_cameras, 4, 4)
        height: Image height in pixels
        width: Image width in pixels
        focal_length: Focal length in pixels
        near: Near plane distance
        far: Far plane distance
        device: Device to place tensors on
        
    Returns:
        Dictionary containing ray information for all cameras
    """
    n_cameras = camera_poses.shape[0]
    all_rays = []
    
    for i in range(n_cameras):
        # Extract camera position and rotation
        extrinsics = camera_poses[i]
        
        # Create intrinsic matrix
        intrinsics = torch.eye(3, device=device)
        intrinsics[0, 0] = focal_length
        intrinsics[1, 1] = focal_length
        intrinsics[0, 2] = width / 2.0
        intrinsics[1, 2] = height / 2.0
        
        # Generate rays for this camera
        rays = generate_rays(height, width, intrinsics, extrinsics, near, far, device)
        
        # Add camera index
        rays["camera_idx"] = torch.ones_like(rays["near"]) * i
        
        all_rays.append(rays)
    
    # Combine rays from all cameras
    combined_rays = {}
    for key in all_rays[0].keys():
        # Reshape each camera's rays to (height*width, ...) and then concatenate
        reshaped_rays = [r[key].reshape(-1, *r[key].shape[2:]) for r in all_rays]
        combined_rays[key] = torch.cat(reshaped_rays, dim=0)
    
    return combined_rays