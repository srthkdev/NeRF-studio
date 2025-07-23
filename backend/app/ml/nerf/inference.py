import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import time
from pathlib import Path

from app.ml.nerf.model import HierarchicalNeRF
from app.ml.nerf.volume_rendering import volume_render_rays as volume_render
from app.ml.nerf.rays import generate_rays, sample_stratified

logger = logging.getLogger(__name__)

@dataclass
class RenderConfig:
    """Configuration for rendering."""
    image_width: int = 800
    image_height: int = 600
    fov: float = 60.0
    near: float = 0.1
    far: float = 10.0
    n_coarse: int = 64
    n_fine: int = 128
    chunk_size: int = 4096
    use_view_frustum_culling: bool = True
    use_adaptive_sampling: bool = True
    quality_level: str = "medium"  # low, medium, high

class FastNeRFInference:
    """
    Fast inference pipeline for novel view synthesis.
    Optimized for real-time rendering with view frustum culling and adaptive sampling.
    """
    
    def __init__(self, model: HierarchicalNeRF, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Cache for rendered views
        self.render_cache = {}
        self.cache_size = 100
        
        # Performance tracking
        self.render_times = []
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def render_novel_view(self, 
                         camera_to_world: np.ndarray,
                         config: RenderConfig = None) -> np.ndarray:
        """
        Render a novel view from the given camera pose.
        
        Args:
            camera_to_world: 4x4 camera to world transformation matrix
            config: Rendering configuration
            
        Returns:
            Rendered image as numpy array (H, W, 3)
        """
        if config is None:
            config = RenderConfig()
        
        start_time = time.time()
        
        # Generate camera rays
        rays_o, rays_d = generate_rays(
            camera_to_world=camera_to_world,
            image_width=config.image_width,
            image_height=config.image_height,
            fov=config.fov,
            near=config.near,
            far=config.far
        )
        
        # Apply view frustum culling if enabled
        if config.use_view_frustum_culling:
            rays_o, rays_d = self._apply_view_frustum_culling(rays_o, rays_d)
        
        # Render in chunks
        image = self._render_chunks(rays_o, rays_d, config)
        
        # Track performance
        render_time = time.time() - start_time
        self.render_times.append(render_time)
        
        logger.info(f"Rendered view in {render_time:.3f}s")
        
        return image
    
    def _apply_view_frustum_culling(self, 
                                   rays_o: torch.Tensor, 
                                   rays_d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply view frustum culling to reduce ray count.
        
        Args:
            rays_o: Ray origins (N, 3)
            rays_d: Ray directions (N, 3)
            
        Returns:
            Filtered rays_o and rays_d
        """
        # Simple culling: remove rays that point away from scene center
        scene_center = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        # Calculate ray-scene center distances
        ray_to_center = scene_center - rays_o
        ray_to_center_norm = torch.norm(ray_to_center, dim=-1, keepdim=True)
        ray_to_center_normalized = ray_to_center / ray_to_center_norm
        
        # Keep rays that point towards scene center (dot product > 0)
        dot_products = torch.sum(rays_d * ray_to_center_normalized, dim=-1)
        valid_mask = dot_products > 0.1  # Small threshold to include some side rays
        
        return rays_o[valid_mask], rays_d[valid_mask]
    
    def _render_chunks(self, 
                      rays_o: torch.Tensor, 
                      rays_d: torch.Tensor, 
                      config: RenderConfig) -> np.ndarray:
        """
        Render rays in chunks to manage memory.
        
        Args:
            rays_o: Ray origins (N, 3)
            rays_d: Ray directions (N, 3)
            config: Rendering configuration
            
        Returns:
            Rendered image as numpy array
        """
        num_rays = rays_o.shape[0]
        chunk_size = config.chunk_size
        
        # Initialize output image
        image = torch.zeros((config.image_height, config.image_width, 3), device=self.device)
        ray_indices = torch.arange(num_rays, device=self.device)
        
        # Render in chunks
        for i in range(0, num_rays, chunk_size):
            end_idx = min(i + chunk_size, num_rays)
            chunk_rays_o = rays_o[i:end_idx]
            chunk_rays_d = rays_d[i:end_idx]
            
            # Render chunk
            chunk_rgb = self._render_ray_chunk(chunk_rays_o, chunk_rays_d, config)
            
            # Map back to image coordinates
            chunk_indices = ray_indices[i:end_idx]
            image_flat = image.view(-1, 3)
            image_flat[chunk_indices] = chunk_rgb
        
        return image.cpu().numpy()
    
    def _render_ray_chunk(self, 
                         rays_o: torch.Tensor, 
                         rays_d: torch.Tensor, 
                         config: RenderConfig) -> torch.Tensor:
        """
        Render a chunk of rays.
        
        Args:
            rays_o: Ray origins (N, 3)
            rays_d: Ray directions (N, 3)
            config: Rendering configuration
            
        Returns:
            Rendered RGB values (N, 3)
        """
        with torch.no_grad():
            # Sample points along rays
            if config.use_adaptive_sampling:
                t_vals, sample_points, sample_dirs = self._adaptive_sampling(
                    rays_o, rays_d, config
                )
            else:
                t_vals, sample_points, sample_dirs = sample_rays(
                    rays_o, rays_d, config.n_coarse, config.near, config.far
                )
            
            # Forward pass through model
            rgb, sigma = self.model(sample_points, sample_dirs)
            
            # Volume rendering
            rendered_rgb = volume_render(rgb, sigma, t_vals)
            
            return rendered_rgb
    
    def _adaptive_sampling(self, 
                          rays_o: torch.Tensor, 
                          rays_d: torch.Tensor, 
                          config: RenderConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptive sampling based on coarse network predictions.
        
        Args:
            rays_o: Ray origins (N, 3)
            rays_d: Ray directions (N, 3)
            config: Rendering configuration
            
        Returns:
            Tuple of (t_vals, sample_points, sample_dirs)
        """
        # Coarse sampling
        t_vals_coarse, sample_points_coarse, sample_dirs_coarse = sample_rays(
            rays_o, rays_d, config.n_coarse, config.near, config.far
        )
        
        # Get coarse predictions
        with torch.no_grad():
            rgb_coarse, sigma_coarse = self.model(sample_points_coarse, sample_dirs_coarse)
        
        # Fine sampling based on coarse density
        t_vals_fine, sample_points_fine, sample_dirs_fine = self._importance_sampling(
            rays_o, rays_d, sigma_coarse, t_vals_coarse, config.n_fine
        )
        
        # Combine coarse and fine samples
        t_vals = torch.cat([t_vals_coarse, t_vals_fine], dim=-1)
        sample_points = torch.cat([sample_points_coarse, sample_points_fine], dim=-2)
        sample_dirs = torch.cat([sample_dirs_coarse, sample_dirs_fine], dim=-2)
        
        # Sort by t_vals
        sorted_indices = torch.argsort(t_vals, dim=-1)
        t_vals = torch.gather(t_vals, -1, sorted_indices)
        sample_points = torch.gather(sample_points, -2, sorted_indices.unsqueeze(-1).expand_as(sample_points))
        sample_dirs = torch.gather(sample_dirs, -2, sorted_indices.unsqueeze(-1).expand_as(sample_dirs))
        
        return t_vals, sample_points, sample_dirs
    
    def _importance_sampling(self, 
                           rays_o: torch.Tensor, 
                           rays_d: torch.Tensor, 
                           sigma_coarse: torch.Tensor, 
                           t_vals_coarse: torch.Tensor, 
                           n_fine: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Importance sampling based on coarse density predictions.
        
        Args:
            rays_o: Ray origins (N, 3)
            rays_d: Ray directions (N, 3)
            sigma_coarse: Coarse density predictions (N, n_coarse, 1)
            t_vals_coarse: Coarse t values (N, n_coarse)
            n_fine: Number of fine samples
            
        Returns:
            Tuple of (t_vals_fine, sample_points_fine, sample_dirs_fine)
        """
        # Calculate weights from coarse density
        weights = sigma_coarse.squeeze(-1) * torch.diff(t_vals_coarse, dim=-1)
        weights = torch.cat([weights, torch.zeros_like(weights[:, :1])], dim=-1)
        
        # Normalize weights
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)
        
        # Sample fine points
        t_vals_fine = self._sample_pdf(t_vals_coarse, weights, n_fine)
        sample_points_fine = rays_o.unsqueeze(1) + t_vals_fine.unsqueeze(-1) * rays_d.unsqueeze(1)
        sample_dirs_fine = rays_d.unsqueeze(1).expand_as(sample_points_fine)
        
        return t_vals_fine, sample_points_fine, sample_dirs_fine
    
    def _sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Sample from probability density function.
        
        Args:
            bins: Bin edges (N, n_bins)
            weights: Weights for each bin (N, n_bins)
            n_samples: Number of samples to draw
            
        Returns:
            Sampled t values (N, n_samples)
        """
        # Convert to PDF
        pdf = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        
        # Sample from CDF
        u = torch.rand(weights.shape[0], n_samples, device=self.device)
        u = u.contiguous()
        
        # Find indices
        indices = torch.searchsorted(cdf, u, right=True)
        indices = torch.clamp(indices, 0, bins.shape[-1] - 1)
        
        # Interpolate
        below = torch.gather(bins, -1, indices - 1)
        above = torch.gather(bins, -1, indices)
        cdf_below = torch.gather(cdf, -1, indices - 1)
        cdf_above = torch.gather(cdf, -1, indices)
        
        # Linear interpolation
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_below) / denom
        samples = below + t * (above - below)
        
        return samples
    
    def render_camera_path(self, 
                          camera_poses: List[np.ndarray], 
                          config: RenderConfig = None) -> List[np.ndarray]:
        """
        Render a sequence of views along a camera path.
        
        Args:
            camera_poses: List of camera to world transformation matrices
            config: Rendering configuration
            
        Returns:
            List of rendered images
        """
        images = []
        
        for i, pose in enumerate(camera_poses):
            logger.info(f"Rendering frame {i+1}/{len(camera_poses)}")
            image = self.render_novel_view(pose, config)
            images.append(image)
        
        return images
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get rendering performance statistics."""
        if not self.render_times:
            return {}
        
        return {
            "mean_render_time": np.mean(self.render_times),
            "std_render_time": np.std(self.render_times),
            "min_render_time": np.min(self.render_times),
            "max_render_time": np.max(self.render_times),
            "total_renders": len(self.render_times)
        }
    
    def clear_cache(self):
        """Clear render cache."""
        self.render_cache.clear()
        logger.info("Render cache cleared")

def create_inference_pipeline(checkpoint_path: str, device: str = "cuda") -> FastNeRFInference:
    """
    Create inference pipeline from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        
    Returns:
        Configured inference pipeline
    """
    # Load checkpoint to get model config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create model
    model = HierarchicalNeRF(
        pos_freq_bands=config.get('pos_freq_bands', 10),
        view_freq_bands=config.get('view_freq_bands', 4),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 8),
        n_coarse=config.get('n_coarse', 64),
        n_fine=config.get('n_fine', 128)
    )
    
    # Create inference pipeline
    pipeline = FastNeRFInference(model, device)
    pipeline.load_checkpoint(checkpoint_path)
    
    return pipeline

if __name__ == "__main__":
    # Test inference pipeline
    print("Fast NeRF inference module loaded successfully") 