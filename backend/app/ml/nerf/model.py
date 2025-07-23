import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class PositionalEncoding:
    """Positional encoding for NeRF as described in the original paper"""
    
    def __init__(self, num_frequencies: int, include_input: bool = True):
        """
        Initialize positional encoding
        
        Args:
            num_frequencies: Number of frequency bands (L in the paper)
            include_input: Whether to include the original input in the encoding
        """
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.funcs = [torch.sin, torch.cos]
        
        # Frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Encoded tensor of shape [..., dim * (2 * num_frequencies) + (dim if include_input else 0)]
        """
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
        return x_dim * ((2 * self.num_frequencies) + (1 if self.include_input else 0))


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
        self.pos_encoder = PositionalEncoding(pos_freq_bands, include_input=True)
        self.view_encoder = PositionalEncoding(view_freq_bands, include_input=True)
        
        # Calculate input dimensions after encoding
        pos_encoded_dim = pos_dim * (1 + 2 * pos_freq_bands)
        view_encoded_dim = view_dim * (1 + 2 * view_freq_bands)
        
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
    
    def get_params_count(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)