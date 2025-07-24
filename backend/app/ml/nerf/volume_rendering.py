
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

def volume_render_rays(
    radiance_field: torch.Tensor,
    z_vals: torch.Tensor,
    ray_directions: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Perform volume rendering on rays.
    
    Args:
        radiance_field: Radiance field output of shape (..., n_samples, 4)
        z_vals: Depth values of shape (..., n_samples)
        ray_directions: Ray directions of shape (..., 3)
        
    Returns:
        Dictionary containing:
            - rgb_map: Rendered RGB map of shape (..., 3)
            - depth_map: Rendered depth map of shape (...,)
            - acc_map: Accumulated opacity map of shape (...,)
            - weights: Weights for each sample of shape (..., n_samples)
    """
    # Extract RGB and sigma from radiance field
    rgb = torch.sigmoid(radiance_field[..., :3])  # (..., n_samples, 3)
    sigma = torch.relu(radiance_field[..., 3])  # (..., n_samples)
    
    # Calculate distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10, device=dists.device)
    ], dim=-1)  # (..., n_samples)
    
    # Multiply by ray direction norm to get real distance
    dists = dists * torch.norm(ray_directions.unsqueeze(-2), dim=-1)
    
    # Calculate alpha (opacity) for each sample
    alpha = 1. - torch.exp(-sigma * dists)  # (..., n_samples)
    
    # Calculate transmittance
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1. - alpha + 1e-10
        ], dim=-1), dim=-1
    )[..., :-1]  # (..., n_samples)
    
    # Calculate weights for each sample
    weights = alpha * transmittance  # (..., n_samples)
    
    # Calculate RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (..., 3)
    
    # Calculate depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)
    
    # Calculate accumulated opacity map
    acc_map = torch.sum(weights, dim=-1)
    
    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map,
        'acc_map': acc_map,
        'weights': weights
    }

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image tensor of shape (..., 3)
        img2: Second image tensor of shape (..., 3)
        
    Returns:
        PSNR value
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20. * torch.log10(1.0 / torch.sqrt(mse))

def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        img1: First image tensor of shape (..., H, W, C)
        img2: Second image tensor of shape (..., H, W, C)
        window_size: Size of the Gaussian window
        size_average: Whether to average SSIM over all pixels
        
    Returns:
        SSIM value
    """
    # Ensure images are in the correct format (B, C, H, W)
    if len(img1.shape) == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    
    # Create Gaussian window
    from torch.nn.functional import conv2d
    
    def create_window(window_size, channel):
        _1D_window = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2) for x in range(window_size)]))
        _2D_window = _1D_window.unsqueeze(1) @ _1D_window.unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # Compute means
    mu1 = conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def render_rays_with_model(
    model,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    white_bkgd: bool = False,
    noise_std: float = 0.0,
    training: bool = False,
    chunk_size: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Render rays using a NeRF model.
    
    Args:
        model: NeRF model
        rays_o: Ray origins of shape (..., 3)
        rays_d: Ray directions of shape (..., 3)
        z_vals: Depth values of shape (..., n_samples)
        white_bkgd: Whether to use white background
        noise_std: Standard deviation of noise to add to sigma
        training: Whether in training mode
        chunk_size: Chunk size for memory management
        
    Returns:
        Dictionary containing rendered outputs
    """
    # Expand dimensions for broadcasting
    rays_o_expanded = rays_o.unsqueeze(-2)  # (..., 1, 3)
    rays_d_expanded = rays_d.unsqueeze(-2)  # (..., 1, 3)
    z_vals_expanded = z_vals.unsqueeze(-1)  # (..., n_samples, 1)
    
    # Compute 3D sample points: r(t) = o + t*d
    points = rays_o_expanded + rays_d_expanded * z_vals_expanded  # (..., n_samples, 3)
    
    # Create view directions for each sample point
    viewdirs = rays_d_expanded.expand_as(points)  # (..., n_samples, 3)
    
    # Flatten for model input
    original_shape = points.shape[:-1]  # (..., n_samples)
    points_flat = points.reshape(-1, 3)
    viewdirs_flat = viewdirs.reshape(-1, 3)
    
    # Process in chunks if specified
    if chunk_size is not None and points_flat.shape[0] > chunk_size:
        rgb_chunks = []
        sigma_chunks = []
        
        for i in range(0, points_flat.shape[0], chunk_size):
            chunk_points = points_flat[i:i+chunk_size]
            chunk_viewdirs = viewdirs_flat[i:i+chunk_size]
            
            # Forward pass through model
            chunk_rgb, chunk_sigma = model(chunk_points, chunk_viewdirs)
            rgb_chunks.append(chunk_rgb)
            sigma_chunks.append(chunk_sigma)
        
        rgb_flat = torch.cat(rgb_chunks, dim=0)
        sigma_flat = torch.cat(sigma_chunks, dim=0)
    else:
        # Forward pass through model
        rgb_flat, sigma_flat = model(points_flat, viewdirs_flat)
    
    # Reshape back to original dimensions
    rgb = rgb_flat.reshape(*original_shape, 3)  # (..., n_samples, 3)
    sigma = sigma_flat.reshape(*original_shape, 1)  # (..., n_samples, 1)
    
    # Create radiance field tensor
    radiance_field = torch.cat([rgb, sigma], dim=-1)  # (..., n_samples, 4)
    
    # Volume render
    output = volume_render_rays(radiance_field, z_vals, rays_d)
    
    # Add noise regularization during training
    if training and noise_std > 0.0:
        sigma_noisy = add_noise_regularization(sigma, noise_std, training)
        radiance_field_noisy = torch.cat([rgb, sigma_noisy], dim=-1)
        output = volume_render_rays(radiance_field_noisy, z_vals, rays_d)
    
    # Apply white background if requested
    if white_bkgd:
        output['rgb_map'] = output['rgb_map'] + (1.0 - output['acc_map'].unsqueeze(-1))
    
    return output


def alpha_composite_rays(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    z_vals: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Alpha composite rays with pre-computed alpha values.
    
    Args:
        rgb: RGB values of shape (..., n_samples, 3)
        alpha: Alpha values of shape (..., n_samples)
        z_vals: Depth values of shape (..., n_samples)
        
    Returns:
        Dictionary containing composited outputs
    """
    # Calculate transmittance
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha[..., :-1] + 1e-10
        ], dim=-1), dim=-1
    )[..., :-1]  # (..., n_samples)
    
    # Calculate weights
    weights = transmittance * alpha  # (..., n_samples)
    
    # Calculate RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (..., 3)
    
    # Calculate depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)  # (...,)
    
    # Calculate accumulated opacity map
    acc_map = torch.sum(weights, dim=-1)  # (...,)
    
    # Calculate disparity map
    disp_map = 1.0 / torch.clamp(depth_map / (acc_map + 1e-10), min=1e-10)
    
    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map.unsqueeze(-1),
        'acc_map': acc_map.unsqueeze(-1),
        'disp_map': disp_map.unsqueeze(-1),
        'weights': weights
    }


def hierarchical_volume_render(
    coarse_model,
    fine_model,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    n_coarse: int = 64,
    n_fine: int = 128,
    perturb: bool = True,
    white_bkgd: bool = False,
    noise_std: float = 0.0,
    training: bool = False
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
        white_bkgd: Whether to use white background
        noise_std: Standard deviation of noise to add to sigma
        training: Whether in training mode
        
    Returns:
        Dictionary containing coarse and fine rendered outputs
    """
    from app.ml.nerf.rays import sample_stratified, sample_hierarchical, combine_hierarchical_samples
    
    # Coarse sampling
    points_coarse, z_vals_coarse = sample_stratified(
        rays_o, rays_d, near, far, n_coarse, perturb=perturb
    )
    
    # Render with coarse model
    coarse_output = render_rays_with_model(
        coarse_model, rays_o, rays_d, z_vals_coarse,
        white_bkgd=white_bkgd, noise_std=noise_std, training=training
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
        fine_model, rays_o, rays_d, z_vals_combined,
        white_bkgd=white_bkgd, noise_std=noise_std, training=training
    )
    
    return {
        "coarse": coarse_output,
        "fine": fine_output,
        "z_vals_coarse": z_vals_coarse,
        "z_vals_fine": z_vals_fine,
        "z_vals_combined": z_vals_combined
    }


def add_noise_regularization(
    sigma: torch.Tensor,
    noise_std: float,
    training: bool
) -> torch.Tensor:
    """
    Add noise regularization to density predictions during training.
    
    Args:
        sigma: Density values of shape (..., n_samples, 1)
        noise_std: Standard deviation of noise
        training: Whether in training mode
        
    Returns:
        Density values with noise added if training
    """
    if training and noise_std > 0.0:
        noise = torch.randn_like(sigma) * noise_std
        return sigma + noise
    return sigma


def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
    training: bool = False
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
        
    Returns:
        Dictionary containing rendered outputs
    """
    # Extract RGB and sigma from raw outputs
    rgb = torch.sigmoid(raw[..., :3])  # (..., n_samples, 3)
    sigma = torch.relu(raw[..., 3])  # (..., n_samples)
    
    # Add noise to sigma during training
    if training and raw_noise_std > 0.0:
        noise = torch.randn_like(sigma) * raw_noise_std
        sigma = sigma + noise
    
    # Calculate distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([
        dists,
        torch.full_like(dists[..., :1], 1e10, device=dists.device)
    ], dim=-1)  # (..., n_samples)
    
    # Multiply by ray direction norm to get real distance
    dists = dists * torch.norm(rays_d.unsqueeze(-2), dim=-1)
    
    # Calculate alpha (opacity) for each sample
    alpha = 1.0 - torch.exp(-sigma * dists)  # (..., n_samples)
    
    # Calculate transmittance
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha[..., :-1] + 1e-10
        ], dim=-1), dim=-1
    )[..., :-1]  # (..., n_samples)
    
    # Calculate weights for each sample
    weights = transmittance * alpha  # (..., n_samples)
    
    # Calculate RGB map
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (..., 3)
    
    # Calculate depth map
    depth_map = torch.sum(weights * z_vals, dim=-1)  # (...,)
    
    # Calculate accumulated opacity map
    acc_map = torch.sum(weights, dim=-1)  # (...,)
    
    # Calculate disparity map
    disp_map = 1.0 / torch.clamp(depth_map / (acc_map + 1e-10), min=1e-10)
    
    # Apply white background if requested
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1))
    
    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map.unsqueeze(-1),
        'acc_map': acc_map.unsqueeze(-1),
        'disp_map': disp_map.unsqueeze(-1),
        'weights': weights
    }
