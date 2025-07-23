import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class PositionalEncoding:
    """Positional encoding for NeRF as described in the original paper"""
    
    def __init__(self, num_frequencies: int, input_dim: int = 3, include_input: bool = True):
        """
        Initialize positional encoding
        
        Args:
            num_frequencies: Number of frequency bands (L in the paper)
            input_dim: Dimension of input features (default: 3 for xyz coordinates)
            include_input: Whether to include the original input in the encoding
        """
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.include_input = include_input
        self.funcs = [torch.sin, torch.cos]
        
        # Frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Encoded tensor of shape [..., input_dim * (2 * num_frequencies) + (input_dim if include_input else 0)]
        """
        # Validate input dimensions
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.shape[-1]}")
            
        out = []
        if self.include_input:
            out.append(x)
            
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(x * freq))
                
        return torch.cat(out, dim=-1)
    
    @property
    def output_dim(self) -> int:
        """Get the output dimension of the encoding"""
        return self.input_dim * ((2 * self.num_frequencies) + (1 if self.include_input else 0))


class NeRFModel(nn.Module):
    """Neural Radiance Field model implementation"""
    
    def __init__(
        self,
        pos_dim: int = 3,
        view_dim: int = 3,
        pos_freq_bands: int = 10,
        view_freq_bands: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list = [4],
    ):
        """
        Initialize NeRF model
        
        Args:
            pos_dim: Dimension of position input (typically 3)
            view_dim: Dimension of view direction input (typically 3)
            pos_freq_bands: Number of frequency bands for position encoding
            view_freq_bands: Number of frequency bands for view direction encoding
            hidden_dim: Width of MLP layers
            num_layers: Number of MLP layers
            skip_connections: Layers with skip connections
        """
        super().__init__()
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(pos_freq_bands, input_dim=pos_dim, include_input=True)
        self.view_encoder = PositionalEncoding(view_freq_bands, input_dim=view_dim, include_input=True)
        
        # Calculate input dimensions after encoding
        pos_encoded_dim = self.pos_encoder.output_dim
        view_encoded_dim = self.view_encoder.output_dim
        
        # MLP layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(pos_encoded_dim, hidden_dim))
        
        # Hidden layers
        for i in range(1, num_layers):
            if i in skip_connections:
                self.layers.append(nn.Linear(hidden_dim + pos_encoded_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layers
        self.sigma_layer = nn.Linear(hidden_dim, 1)  # Density output
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)  # Feature vector for color
        self.color_layer = nn.Linear(hidden_dim + view_encoded_dim, 3)  # RGB output
        
        # Store model parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Position tensor of shape [..., 3]
            d: View direction tensor of shape [..., 3]
            
        Returns:
            Tuple of (rgb, sigma) tensors
        """
        # Encode inputs
        x_encoded = self.pos_encoder(x)
        d_encoded = self.view_encoder(d)
        
        # Initial layer
        h = F.relu(self.layers[0](x_encoded))
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers):
            if i in self.skip_connections:
                h = torch.cat([h, x_encoded], dim=-1)
            h = F.relu(self.layers[i](h))
        
        # Density prediction
        sigma = F.relu(self.sigma_layer(h))
        
        # Color prediction
        feature = self.feature_layer(h)
        h = torch.cat([feature, d_encoded], dim=-1)
        rgb = torch.sigmoid(self.color_layer(h))  # Sigmoid to ensure [0,1] range
        
        return rgb, sigma
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_params_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Get detailed model information"""
        return {
            'total_params': self.get_params_count(),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'skip_connections': self.skip_connections,
            'pos_encoding_dim': self.pos_encoder.output_dim,
            'view_encoding_dim': self.view_encoder.output_dim,
        }


class HierarchicalNeRF(nn.Module):
    """
    Hierarchical NeRF with coarse and fine networks for importance sampling.

    Usage Example:
        >>> model = HierarchicalNeRF(n_coarse=64, n_fine=128)
        >>> rays_o = torch.randn(batch_size, 3)
        >>> rays_d = torch.randn(batch_size, 3)
        >>> rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        >>> near = torch.ones(batch_size, 1) * 2.0
        >>> far = torch.ones(batch_size, 1) * 6.0
        >>> output = model(rays_o, rays_d, near, far, perturb=True, training=True)
        >>> print(output['coarse']['rgb_map'].shape, output['fine']['rgb_map'].shape)

    Returns:
        Dictionary with keys:
            - 'coarse': Output from coarse model (dict)
            - 'fine': Output from fine model (dict)
            - 'z_vals_coarse': Coarse sample depths
            - 'z_vals_fine': Fine sample depths
            - 'z_vals_combined': Combined sample depths
    """
    
    def __init__(
        self,
        pos_dim: int = 3,
        view_dim: int = 3,
        pos_freq_bands: int = 10,
        view_freq_bands: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list = [4],
        n_coarse: int = 64,
        n_fine: int = 128,
    ):
        """
        Initialize hierarchical NeRF model with coarse and fine networks
        
        Args:
            pos_dim: Dimension of position input (typically 3)
            view_dim: Dimension of view direction input (typically 3)
            pos_freq_bands: Number of frequency bands for position encoding
            view_freq_bands: Number of frequency bands for view direction encoding
            hidden_dim: Width of MLP layers
            num_layers: Number of MLP layers
            skip_connections: Layers with skip connections
            n_coarse: Number of coarse samples per ray
            n_fine: Number of fine samples per ray
        """
        super().__init__()
        
        # Create coarse and fine networks
        self.coarse_model = NeRFModel(
            pos_dim=pos_dim,
            view_dim=view_dim,
            pos_freq_bands=pos_freq_bands,
            view_freq_bands=view_freq_bands,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_connections=skip_connections,
        )
        
        self.fine_model = NeRFModel(
            pos_dim=pos_dim,
            view_dim=view_dim,
            pos_freq_bands=pos_freq_bands,
            view_freq_bands=view_freq_bands,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_connections=skip_connections,
        )
        
        # Store sampling parameters
        self.n_coarse = n_coarse
        self.n_fine = n_fine
    
    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        perturb: bool = True,
        white_bkgd: bool = False,
        noise_std: float = 0.0,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical NeRF
        
        Args:
            rays_o: Ray origins of shape (..., 3)
            rays_d: Ray directions of shape (..., 3)
            near: Near plane distances of shape (..., 1)
            far: Far plane distances of shape (..., 1)
            perturb: Whether to add perturbation to samples
            white_bkgd: Whether to use white background
            noise_std: Standard deviation of noise to add to sigma
            training: Whether in training mode
            
        Returns:
            Dictionary containing coarse and fine rendered outputs
        """
        from app.ml.nerf.rays import sample_stratified, sample_hierarchical, combine_hierarchical_samples
        from app.ml.nerf.volume_rendering import render_rays_with_model
        
        # Coarse sampling
        points_coarse, z_vals_coarse = sample_stratified(
            rays_o, rays_d, near, far, self.n_coarse, perturb=perturb
        )
        
        # Render with coarse model
        coarse_output = render_rays_with_model(
            self.coarse_model, rays_o, rays_d, z_vals_coarse,
            white_bkgd=white_bkgd, noise_std=noise_std, training=training
        )
        
        # Fine sampling based on coarse weights
        points_fine, z_vals_fine = sample_hierarchical(
            rays_o, rays_d, z_vals_coarse, coarse_output["weights"], self.n_fine, perturb=perturb
        )
        
        # Combine coarse and fine samples
        z_vals_combined, points_combined = combine_hierarchical_samples(
            z_vals_coarse, z_vals_fine, points_coarse, points_fine
        )
        
        # Render with fine model using combined samples
        fine_output = render_rays_with_model(
            self.fine_model, rays_o, rays_d, z_vals_combined,
            white_bkgd=white_bkgd, noise_std=noise_std, training=training
        )
        
        return {
            "coarse": coarse_output,
            "fine": fine_output,
            "z_vals_coarse": z_vals_coarse,
            "z_vals_fine": z_vals_fine,
            "z_vals_combined": z_vals_combined
        }
    
    def get_params_count(self) -> Dict[str, int]:
        """Get parameter counts for both networks"""
        return {
            "coarse": self.coarse_model.get_params_count(),
            "fine": self.fine_model.get_params_count(),
            "total": self.coarse_model.get_params_count() + self.fine_model.get_params_count()
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information"""
        coarse_info = self.coarse_model.get_model_info()
        fine_info = self.fine_model.get_model_info()
        
        return {
            "coarse_model": coarse_info,
            "fine_model": fine_info,
            "n_coarse": self.n_coarse,
            "n_fine": self.n_fine,
            "total_params": coarse_info["total_params"] + fine_info["total_params"]
        }


def compute_sampling_weights(
    sigma: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    noise_std: float = 0.0,
    training: bool = False
) -> torch.Tensor:
    """
    Compute sampling weights from density predictions for importance sampling
    
    Args:
        sigma: Density values of shape (..., n_samples, 1)
        z_vals: Depth values of shape (..., n_samples)
        rays_d: Ray directions of shape (..., 3)
        noise_std: Standard deviation of noise to add to sigma
        training: Whether in training mode
        
    Returns:
        Sampling weights of shape (..., n_samples)
    """
    # Add noise to sigma during training
    if training and noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * noise_std
    
    # Calculate distances between consecutive samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
    
    # Multiply distances by ray direction norm
    dists = dists * torch.norm(rays_d, dim=-1, keepdim=True)
    
    # Calculate alpha values
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)
    
    # Calculate transmittance
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha[..., :-1]], dim=-1), 
        dim=-1
    )
    
    # Calculate weights
    weights = transmittance * alpha
    
    return weights


def verify_sampling_distribution(
    weights: torch.Tensor,
    z_vals: torch.Tensor,
    sigma: torch.Tensor,
    tolerance: float = 1e-3
) -> Dict[str, bool]:
    """
    Verify that sampling weights follow the density predictions
    
    Args:
        weights: Sampling weights of shape (..., n_samples)
        z_vals: Depth values of shape (..., n_samples)
        sigma: Density values of shape (..., n_samples, 1)
        tolerance: Tolerance for verification
        
    Returns:
        Dictionary with verification results
    """
    # Check that weights sum to approximately 1 (or less due to finite sampling)
    weight_sums = torch.sum(weights, dim=-1)
    weights_normalized = torch.all(weight_sums <= 1.0 + tolerance)
    
    # Check that weights are non-negative
    weights_positive = torch.all(weights >= -tolerance)
    
    # Check that higher density regions get higher weights (correlation test)
    sigma_flat = sigma.squeeze(-1)
    
    # Compute correlation between weights and sigma
    # This is a simplified check - in practice, the relationship is more complex
    weight_sigma_corr = []
    for i in range(weights.shape[0]):
        if weights.shape[0] > 1:  # Batch dimension exists
            w = weights[i].flatten()
            s = sigma_flat[i].flatten()
        else:
            w = weights.flatten()
            s = sigma_flat.flatten()
        
        # Remove zero weights for correlation calculation
        mask = w > tolerance
        if torch.sum(mask) > 1:
            corr = torch.corrcoef(torch.stack([w[mask], s[mask]]))[0, 1]
            weight_sigma_corr.append(corr)
    
    # Check if correlation is positive (weights should increase with density)
    positive_correlation = True
    if weight_sigma_corr:
        avg_corr = torch.mean(torch.stack(weight_sigma_corr))
        positive_correlation = avg_corr > 0
    
    return {
        "weights_normalized": weights_normalized.item(),
        "weights_positive": weights_positive.item(),
        "positive_correlation": positive_correlation
    }


def create_coarse_fine_models(
    model_config: Dict[str, any]
) -> Tuple[NeRFModel, NeRFModel]:
    """
    Create separate coarse and fine NeRF models with identical architectures
    
    Args:
        model_config: Configuration dictionary for model parameters
        
    Returns:
        Tuple of (coarse_model, fine_model)
    """
    coarse_model = NeRFModel(**model_config)
    fine_model = NeRFModel(**model_config)
    
    return coarse_model, fine_model


def initialize_hierarchical_weights(
    coarse_model: NeRFModel,
    fine_model: NeRFModel,
    initialization_scheme: str = "xavier"
) -> None:
    """
    Initialize weights for hierarchical NeRF models
    
    Args:
        coarse_model: Coarse NeRF model
        fine_model: Fine NeRF model
        initialization_scheme: Weight initialization scheme
    """
    def init_weights(model, scheme):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif scheme == "normal":
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    init_weights(coarse_model, initialization_scheme)
    init_weights(fine_model, initialization_scheme)