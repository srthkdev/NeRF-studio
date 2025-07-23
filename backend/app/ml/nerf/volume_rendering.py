import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
from app.ml.nerf.rays import sample_stratified, sample_hierarchical, combine_hierarchical_samples


def volume_render_rays(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_bkgd: bool = False,
    noise_std: float = 0.0,
    training: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Volume rendering along rays using the NeRF equation.
    
    Args:
        rgb: RGB values of shape (..., n_samples, 3)
        sigma: Density values of shape (..., n_samples, 1)
        z_vals: Depth values of shape (..., n_samples)
        rays_d: Ray directions of shape (..., 3)
        white_bkgd: Whether to use white background
        noise_std: Standard deviation of noise to add to sigma during training
        training: Whether in training mode
        
    Returns:
        Dictionary containing:
            - rgb_map: Rendered RGB values of shape (..., 3)
            - depth_map: Rendered depth values of shape (..., 1)
            - acc_map: Accumulated opacity of shape (..., 1)
            - weights: Sample weights of shape (..., n_samples)
            - disp_map: Disparity map of shape (..., 1)
    """
    # Add noise to sigma during training for regularization
    if training and noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * noise_std
    
    # Calculate distances between consecutive samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)  # (..., n_samples)
    
    # Multiply distances by ray direction norm to get actual distances
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Calculate alpha values: alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (..., n_samples)
    
    # Calculate transmittance: T(t) = prod_{i=1}^{t-1} (1 - alpha_i)
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), 
        dim=-1
    )  # (..., n_samples)
    
    # Calculate weights: w(t) = T(t) * alpha(t)
    weights = transmittance * alpha  # (..., n_samples)
    
    # Render RGB: C = sum(w_i * c_i)
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (..., 3)
    
    # Render depth: d = sum(w_i * z_i)
    depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)  # (..., 1)
    
    # Accumulated opacity: sum(w_i)
    acc_map = torch.sum(weights, dim=-1, keepdim=True)  # (..., 1)
    
    # Disparity map: 1 / depth
    disp_map = 1.0 / torch.clamp(depth_map, min=1e-6)  # (..., 1)
    
    # Add white background if requested
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map)
    
    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
        "disp_map": disp_map
    }


def alpha_composite_rays(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    z_vals: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Alpha compositing with pre-computed alpha values.
    
    Args:
        rgb: RGB values of shape (..., n_samples, 3)
        alpha: Alpha values of shape (..., n_samples)
        z_vals: Depth values of shape (..., n_samples)
        
    Returns:
        Dictionary containing rendered values
    """
    # Calculate transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), 
        dim=-1
    )
    
    # Calculate weights
    weights = transmittance * alpha
    
    # Render RGB
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    
    # Render depth
    depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)
    
    # Accumulated opacity
    acc_map = torch.sum(weights, dim=-1, keepdim=True)
    
    # Disparity map
    disp_map = 1.0 / torch.clamp(depth_map, min=1e-6)
    
    return {
        "rgb_map": rgb_map,
        "depth_map": depth_map,
        "acc_map": acc_map,
        "weights": weights,
        "disp_map": disp_map
    }


def render_rays_with_model(
    model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    chunk_size: Optional[int] = None,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Render rays using a NeRF model.
    
    Args:
        model: NeRF model that takes (x, d) and returns (rgb, sigma)
        rays_o: Ray origins of shape (..., 3)
        rays_d: Ray directions of shape (..., 3)
        z_vals: Depth values of shape (..., n_samples)
        chunk_size: Chunk size for memory efficiency
        **kwargs: Additional arguments for volume_render_rays
        
    Returns:
        Dictionary containing rendered values
    """
    # Calculate 3D points along rays
    rays_o_expanded = rays_o.unsqueeze(-2)  # (..., 1, 3)
    rays_d_expanded = rays_d.unsqueeze(-2)  # (..., 1, 3)
    z_vals_expanded = z_vals.unsqueeze(-1)  # (..., n_samples, 1)
    
    points = rays_o_expanded + rays_d_expanded * z_vals_expanded  # (..., n_samples, 3)
    
    # Flatten for processing
    points_flat = points.reshape(-1, 3)  # (n_rays * n_samples, 3)
    rays_d_flat = rays_d.unsqueeze(-2).expand_as(points).reshape(-1, 3)  # (n_rays * n_samples, 3)
    
    # Process in chunks if specified
    if chunk_size is not None:
        rgb_list = []
        sigma_list = []
        
        for i in range(0, points_flat.shape[0], chunk_size):
            chunk_points = points_flat[i:i+chunk_size]
            chunk_dirs = rays_d_flat[i:i+chunk_size]
            
            # Forward pass through model
            chunk_rgb, chunk_sigma = model(chunk_points, chunk_dirs)
            
            rgb_list.append(chunk_rgb)
            sigma_list.append(chunk_sigma)
        
        rgb_flat = torch.cat(rgb_list, dim=0)
        sigma_flat = torch.cat(sigma_list, dim=0)
    else:
        # Forward pass through model
        rgb_flat, sigma_flat = model(points_flat, rays_d_flat)
    
    # Reshape back to original dimensions
    rgb = rgb_flat.reshape(*points.shape)  # (..., n_samples, 3)
    sigma = sigma_flat.reshape(*points.shape[:-1], 1)  # (..., n_samples, 1)
    
    # Volume render
    return volume_render_rays(rgb, sigma, z_vals, rays_d, **kwargs)


def hierarchical_volume_render(
    coarse_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Hierarchical volume rendering with coarse and fine models.
    
    Args:
        coarse_model: Coarse NeRF model
        fine_model: Fine NeRF model
        rays_o: Ray origins of shape (..., 3)
        rays_d: Ray directions of shape (..., 3)
        near: Near plane distances of shape (..., 1)
        far: Far plane distances of shape (..., 1)
        n_coarse: Number of coarse samples
        n_fine: Number of fine samples
        perturb: Whether to add perturbation to samples
        **kwargs: Additional arguments for volume rendering
        
    Returns:
        Dictionary containing coarse and fine rendered values
    """
    # Coarse sampling
    points_coarse, z_vals_coarse = sample_stratified(
        rays_o, rays_d, near, far, n_coarse, perturb=perturb
    )
    
    # Render with coarse model
    coarse_output = render_rays_with_model(
        coarse_model, rays_o, rays_d, z_vals_coarse, **kwargs
    )
    
    # Fine sampling based on coarse weights
    points_fine, z_vals_fine = sample_hierarchical(
        rays_o, rays_d, z_vals_coarse, coarse_output["weights"], n_fine, perturb=perturb
    )
    
    # Combine coarse and fine samples
    z_vals_combined, points_combined = combine_hierarchical_samples(
        z_vals_coarse, z_vals_fine, points_coarse, points_fine
    )
    
    # Render with fine model using combined samples
    fine_output = render_rays_with_model(
        fine_model, rays_o, rays_d, z_vals_combined, **kwargs
    )
    
    return {
        "coarse": coarse_output,
        "fine": fine_output,
        "z_vals_coarse": z_vals_coarse,
        "z_vals_fine": z_vals_fine,
        "z_vals_combined": z_vals_combined
    }


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image of shape (..., C, H, W) or (..., H, W, C)
        target: Target image of same shape
        
    Returns:
        PSNR value
    """
    # Ensure images are in the same format
    if pred.shape[-1] == 3:  # (..., H, W, C)
        pred = pred.permute(..., -1, -3, -2)  # (..., C, H, W)
        target = target.permute(..., -1, -3, -2)  # (..., C, H, W)
    
    # Compute MSE
    mse = F.mse_loss(pred, target)
    
    # PSNR = 20 * log10(1 / sqrt(MSE)) = -10 * log10(MSE)
    psnr = -10.0 * torch.log10(mse)
    
    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image of shape (..., C, H, W)
        target: Target image of same shape
        window_size: Size of the SSIM window
        
    Returns:
        SSIM value
    """
    # Ensure images are in the correct format
    if pred.shape[-1] == 3:  # (..., H, W, C)
        pred = pred.permute(..., -1, -3, -2)  # (..., C, H, W)
        target = target.permute(..., -1, -3, -2)  # (..., C, H, W)
    
    # Use PyTorch's SSIM implementation if available, otherwise compute manually
    try:
        from torchmetrics.functional import structural_similarity_index_measure
        return structural_similarity_index_measure(pred, target, window_size=window_size)
    except ImportError:
        # Manual SSIM computation
        return _compute_ssim_manual(pred, target, window_size)


def _compute_ssim_manual(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Manual SSIM computation."""
    # This is a simplified SSIM implementation
    # For production use, consider using torchmetrics or similar libraries
    
    # Convert to grayscale if needed
    if pred.shape[-3] == 3:  # RGB
        pred_gray = 0.299 * pred[..., 0, :, :] + 0.587 * pred[..., 1, :, :] + 0.114 * pred[..., 2, :, :]
        target_gray = 0.299 * target[..., 0, :, :] + 0.587 * target[..., 1, :, :] + 0.114 * target[..., 2, :, :]
    else:
        pred_gray = pred.squeeze(-3)
        target_gray = target.squeeze(-3)
    
    # Simple SSIM approximation
    mu_pred = F.avg_pool2d(pred_gray, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target_gray, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool2d(pred_gray ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target_gray ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred_gray * target_gray, window_size, stride=1, padding=window_size//2) - mu_pred_target
    
    # SSIM constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # SSIM formula
    ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
    
    return torch.mean(ssim)


def add_noise_regularization(
    sigma: torch.Tensor,
    noise_std: float = 0.0,
    training: bool = False
) -> torch.Tensor:
    """
    Add noise regularization to sigma values during training.
    
    Args:
        sigma: Density values of shape (..., 1)
        noise_std: Standard deviation of noise
        training: Whether in training mode
        
    Returns:
        Sigma values with optional noise
    """
    if training and noise_std > 0.0:
        return sigma + torch.randn_like(sigma) * noise_std
    else:
        return sigma


def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
    training: bool = False,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Convert raw network outputs to rendered values.
    
    Args:
        raw: Raw network outputs of shape (..., n_samples, 4) [r, g, b, sigma]
        z_vals: Depth values of shape (..., n_samples)
        rays_d: Ray directions of shape (..., 3)
        raw_noise_std: Standard deviation of noise to add to sigma
        white_bkgd: Whether to use white background
        training: Whether in training mode
        **kwargs: Additional arguments for volume rendering
        
    Returns:
        Dictionary containing rendered values
    """
    # Extract RGB and sigma from raw outputs
    rgb = torch.sigmoid(raw[..., :3])  # RGB in [0, 1]
    sigma = F.relu(raw[..., 3:])  # Sigma should be positive
    
    # Add noise regularization if specified
    if raw_noise_std > 0.0:
        sigma = add_noise_regularization(sigma, raw_noise_std, training)
    
    # Volume render
    return volume_render_rays(rgb, sigma, z_vals, rays_d, white_bkgd=white_bkgd, **kwargs)