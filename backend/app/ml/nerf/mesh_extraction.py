import torch
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MeshExtractor:
    """
    Extract meshes from trained NeRF models using marching cubes.
    Supports multiple export formats: GLTF, OBJ, PLY.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def sample_density_field(self, 
                           bounds: Tuple[float, float, float, float, float, float],
                           resolution: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample the density field on a regular grid.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
            resolution: Grid resolution
            
        Returns:
            Tuple of (vertices, densities)
        """
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Create regular grid
        x = torch.linspace(x_min, x_max, resolution, device=self.device)
        y = torch.linspace(y_min, y_max, resolution, device=self.device)
        z = torch.linspace(z_min, z_max, resolution, device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Sample density field
        densities = []
        batch_size = 8192  # Process in batches to avoid memory issues
        
        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                
                # Create dummy view directions (not used for density)
                dummy_dirs = torch.zeros_like(batch_points)
                
                # Forward pass to get density
                rgb, sigma = self.model(batch_points, dummy_dirs)
                densities.append(sigma.squeeze(-1).cpu())
        
        densities = torch.cat(densities, dim=0)
        densities = densities.reshape(resolution, resolution, resolution)
        
        return points.cpu().numpy(), densities.cpu().numpy()
    
    def extract_mesh_marching_cubes(self, 
                                   densities: np.ndarray,
                                   bounds: Tuple[float, float, float, float, float, float],
                                   iso_level: float = 0.5) -> Optional[trimesh.Trimesh]:
        """
        Extract mesh using marching cubes algorithm.
        
        Args:
            densities: 3D density field
            bounds: Scene bounds
            iso_level: Iso-surface level for marching cubes
            
        Returns:
            Trimesh object or None if extraction fails
        """
        try:
            # Apply marching cubes
            vertices, faces, normals, values = trimesh.creation.marching_cubes(
                densities, level=iso_level
            )
            
            # Scale vertices to world coordinates
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            scale = np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / (densities.shape[0] - 1)
            offset = np.array([x_min, y_min, z_min])
            
            vertices = vertices * scale + offset
            
            # Create mesh
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                normals=normals
            )
            
            # Clean up mesh
            mesh = self._clean_mesh(mesh)
            
            return mesh
            
        except Exception as e:
            logger.error(f"Mesh extraction failed: {e}")
            return None
    
    def _clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Clean and optimize the extracted mesh.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Cleaned mesh
        """
        # Remove duplicate vertices
        mesh.remove_duplicate_vertices()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Fill holes
        mesh.fill_holes()
        
        # Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        return mesh
    
    def export_mesh(self, 
                   mesh: trimesh.Trimesh, 
                   output_path: str, 
                   format: str = 'gltf') -> bool:
        """
        Export mesh to various formats.
        
        Args:
            mesh: Mesh to export
            output_path: Output file path
            format: Export format ('gltf', 'obj', 'ply')
            
        Returns:
            True if export successful
        """
        try:
            if format.lower() == 'gltf':
                # Export as GLTF with PBR materials
                scene = trimesh.Scene()
                scene.add_geometry(mesh)
                
                # Add basic material
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.5
                )
                mesh.visual.material = material
                
                scene.export(output_path)
                
            elif format.lower() == 'obj':
                mesh.export(output_path)
                
            elif format.lower() == 'ply':
                mesh.export(output_path)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Mesh exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")
            return False
    
    def extract_and_export(self, 
                          bounds: Tuple[float, float, float, float, float, float],
                          output_dir: str,
                          resolution: int = 128,
                          iso_level: float = 0.5,
                          formats: List[str] = ['gltf', 'obj', 'ply']) -> Dict[str, str]:
        """
        Extract mesh and export to multiple formats.
        
        Args:
            bounds: Scene bounds
            output_dir: Output directory
            resolution: Grid resolution
            iso_level: Iso-surface level
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample density field
        logger.info("Sampling density field...")
        points, densities = self.sample_density_field(bounds, resolution)
        
        # Extract mesh
        logger.info("Extracting mesh...")
        mesh = self.extract_mesh_marching_cubes(densities, bounds, iso_level)
        
        if mesh is None:
            return {}
        
        # Export to different formats
        exported_files = {}
        base_name = "nerf_mesh"
        
        for format in formats:
            output_path = os.path.join(output_dir, f"{base_name}.{format}")
            if self.export_mesh(mesh, output_path, format):
                exported_files[format] = output_path
        
        return exported_files

def extract_mesh_from_checkpoint(checkpoint_path: str,
                                output_dir: str,
                                bounds: Tuple[float, float, float, float, float, float] = (-2, 2, -2, 2, -2, 2),
                                resolution: int = 128,
                                formats: List[str] = ['gltf', 'obj', 'ply']) -> Dict[str, str]:
    """
    Extract mesh from a trained NeRF checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for mesh files
        bounds: Scene bounds for mesh extraction
        resolution: Grid resolution
        formats: Export formats
        
    Returns:
        Dictionary of exported file paths
    """
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Create model
        from backend.app.ml.nerf.model import HierarchicalNeRF
        
        model = HierarchicalNeRF(
            pos_freq_bands=config.get('pos_freq_bands', 10),
            view_freq_bands=config.get('view_freq_bands', 4),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 8),
            n_coarse=config.get('n_coarse', 64),
            n_fine=config.get('n_fine', 128)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract mesh
        extractor = MeshExtractor(model)
        return extractor.extract_and_export(bounds, output_dir, resolution, formats=formats)
        
    except Exception as e:
        logger.error(f"Mesh extraction from checkpoint failed: {e}")
        return {}

if __name__ == "__main__":
    # Test mesh extraction
    print("Mesh extraction module loaded successfully") 