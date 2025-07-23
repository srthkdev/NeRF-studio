import torch
import numpy as np
import trimesh
import json
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    GLTF = "gltf"
    OBJ = "obj"
    PLY = "ply"
    USD = "usd"
    FBX = "fbx"
    STL = "stl"

@dataclass
class ExportConfig:
    """Configuration for mesh export"""
    format: ExportFormat
    resolution: int = 128
    texture_resolution: int = 1024
    include_textures: bool = True
    bake_textures: bool = True
    optimize_mesh: bool = True
    compression: bool = True
    quality: str = "high"  # low, medium, high
    bounds: Optional[List[float]] = None

class TextureBaker:
    """Texture baking utility for NeRF models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def bake_texture_from_nerf(self, mesh: trimesh.Trimesh, 
                              texture_resolution: int = 1024) -> trimesh.Trimesh:
        """Bake texture from NeRF model to mesh"""
        # Generate UV coordinates if not present
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
            mesh = self._generate_uv_coordinates(mesh)
            
        # Sample colors from NeRF at UV positions
        uv_coords = mesh.visual.uv
        colors = self._sample_colors_at_uv(mesh, uv_coords, texture_resolution)
        
        # Create texture image
        texture_image = self._create_texture_image(colors, texture_resolution)
        
        # Apply texture to mesh
        mesh.visual.texture = texture_image
        return mesh
        
    def _generate_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Generate UV coordinates for mesh"""
        # Simple spherical UV mapping
        vertices = mesh.vertices
        u = 0.5 + np.arctan2(vertices[:, 2], vertices[:, 0]) / (2 * np.pi)
        v = 0.5 + np.arcsin(np.clip(vertices[:, 1], -1, 1)) / np.pi
        
        uv_coords = np.column_stack([u, v])
        mesh.visual.uv = uv_coords
        return mesh
        
    def _sample_colors_at_uv(self, mesh: trimesh.Trimesh, 
                           uv_coords: np.ndarray, 
                           texture_resolution: int) -> np.ndarray:
        """Sample colors from NeRF at UV coordinates"""
        # Convert UV to 3D positions
        positions = self._uv_to_3d_positions(mesh, uv_coords)
        
        # Sample from NeRF model
        with torch.no_grad():
            positions_tensor = torch.tensor(positions, dtype=torch.float32, device=self.device)
            colors = self.model(positions_tensor)[:, :3].cpu().numpy()
            
        return colors
        
    def _uv_to_3d_positions(self, mesh: trimesh.Trimesh, 
                           uv_coords: np.ndarray) -> np.ndarray:
        """Convert UV coordinates to 3D positions on mesh surface"""
        # Interpolate positions based on UV coordinates
        # This is a simplified version - in practice you'd use proper barycentric interpolation
        vertices = mesh.vertices
        faces = mesh.faces
        
        # For simplicity, use vertex positions directly
        # In a real implementation, you'd interpolate based on UV coordinates
        return vertices
        
    def _create_texture_image(self, colors: np.ndarray, 
                             texture_resolution: int) -> np.ndarray:
        """Create texture image from sampled colors"""
        # Reshape colors to texture image
        texture_size = int(np.sqrt(len(colors)))
        if texture_size * texture_size != len(colors):
            # Pad or crop to make square
            texture_size = int(np.sqrt(len(colors)))
            colors = colors[:texture_size * texture_size]
            
        texture = colors.reshape(texture_size, texture_size, 3)
        
        # Resize to target resolution
        from PIL import Image
        texture_image = Image.fromarray((texture * 255).astype(np.uint8))
        texture_image = texture_image.resize((texture_resolution, texture_resolution))
        
        return np.array(texture_image)

class MeshOptimizer:
    """Mesh optimization utilities"""
    
    def __init__(self):
        pass
        
    def optimize_mesh(self, mesh: trimesh.Trimesh, 
                     target_faces: Optional[int] = None,
                     quality: str = "high") -> trimesh.Trimesh:
        """Optimize mesh by reducing face count and improving quality"""
        if target_faces is None:
            # Set target based on quality
            current_faces = len(mesh.faces)
            if quality == "low":
                target_faces = max(100, current_faces // 10)
            elif quality == "medium":
                target_faces = max(500, current_faces // 4)
            else:  # high
                target_faces = max(1000, current_faces // 2)
                
        # Decimate mesh
        optimized_mesh = self.decimate_mesh(mesh, target_faces)
        
        # Clean up mesh
        optimized_mesh = self.clean_mesh(optimized_mesh)
        
        return optimized_mesh
        
    def decimate_mesh(self, mesh: trimesh.Trimesh, 
                     target_faces: int) -> trimesh.Trimesh:
        """Decimate mesh to target face count"""
        if len(mesh.faces) <= target_faces:
            return mesh
            
        # Use trimesh's simplify method
        try:
            simplified = mesh.simplify_quadratic_decimation(target_faces)
            return simplified
        except Exception as e:
            logger.warning(f"Failed to decimate mesh: {e}")
            return mesh
            
    def clean_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Clean mesh by removing duplicate vertices, fixing normals, etc."""
        # Remove duplicate vertices
        mesh = mesh.remove_duplicate_vertices()
        
        # Fix normals
        mesh = mesh.fix_normals()
        
        # Remove degenerate faces
        mesh = mesh.remove_degenerate_faces()
        
        return mesh

class AdvancedMeshExporter:
    """
    Advanced mesh exporter with USD support, texture baking, and optimization.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback
        
    def _update_progress(self, stage: str, progress: float, message: str = ""):
        """Update progress through callback"""
        if self.progress_callback:
            self.progress_callback(stage, progress, message)
            
    def extract_mesh_with_textures(self, 
                                 config: ExportConfig,
                                 output_dir: str) -> Dict[str, str]:
        """
        Extract mesh with texture baking and optimization.
        
        Args:
            config: Export configuration
            output_dir: Output directory for files
            
        Returns:
            Dictionary mapping format to file path
        """
        self._update_progress("initialization", 0.0, "Initializing export...")
        
        # Set bounds
        if config.bounds is None:
            config.bounds = [-2, 2, -2, 2, -2, 2]
            
        # Extract base mesh
        self._update_progress("mesh_extraction", 0.1, "Extracting mesh...")
        mesh = self._extract_base_mesh(config)
        
        # Optimize mesh if requested
        if config.optimize_mesh:
            self._update_progress("optimization", 0.3, "Optimizing mesh...")
            mesh = self._optimize_mesh(mesh, config)
            
        # Bake textures if requested
        if config.bake_textures and config.include_textures:
            self._update_progress("texture_baking", 0.5, "Baking textures...")
            mesh = self._bake_textures(mesh, config)
            
        # Export in multiple formats
        self._update_progress("export", 0.7, "Exporting files...")
        exported_files = self._export_multiple_formats(mesh, config, output_dir)
        
        # Compress if requested
        if config.compression:
            self._update_progress("compression", 0.9, "Compressing files...")
            exported_files = self._compress_files(exported_files, output_dir)
            
        self._update_progress("complete", 1.0, "Export complete!")
        return exported_files
        
    def _extract_base_mesh(self, config: ExportConfig) -> trimesh.Trimesh:
        """Extract base mesh using marching cubes"""
        x_min, x_max, y_min, y_max, z_min, z_max = config.bounds
        
        # Create regular grid
        x = torch.linspace(x_min, x_max, config.resolution, device=self.device)
        y = torch.linspace(y_min, y_max, config.resolution, device=self.device)
        z = torch.linspace(z_min, z_max, config.resolution, device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Sample density field
        densities = []
        batch_size = 8192
        
        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i+batch_size]
                dummy_dirs = torch.zeros_like(batch_points)
                rgb, sigma = self.model(batch_points, dummy_dirs)
                densities.append(sigma.squeeze(-1).cpu())
                
        densities = torch.cat(densities).reshape(config.resolution, config.resolution, config.resolution)
        densities = densities.numpy()
        
        # Apply marching cubes
        vertices, faces, normals, values = trimesh.creation.marching_cubes(
            densities, level=0.5, spacing=[(x_max - x_min) / config.resolution] * 3
        )
        
        # Adjust vertex positions
        vertices[:, 0] = vertices[:, 0] * (x_max - x_min) / config.resolution + x_min
        vertices[:, 1] = vertices[:, 1] * (y_max - y_min) / config.resolution + y_min
        vertices[:, 2] = vertices[:, 2] * (z_max - z_min) / config.resolution + z_min
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
        
    def _optimize_mesh(self, mesh: trimesh.Trimesh, config: ExportConfig) -> trimesh.Trimesh:
        """Optimize mesh quality and reduce complexity"""
        # Remove duplicate vertices
        mesh.remove_duplicate_vertices()
        
        # Remove degenerate faces
        mesh.remove_degenerate_faces()
        
        # Fill holes
        mesh.fill_holes()
        
        # Simplify mesh based on quality setting
        if config.quality == "low":
            target_faces = len(mesh.faces) // 4
        elif config.quality == "medium":
            target_faces = len(mesh.faces) // 2
        else:  # high
            target_faces = len(mesh.faces)
            
        if target_faces < len(mesh.faces):
            mesh = mesh.simplify_quadratic_decimation(target_faces)
            
        # Smooth mesh slightly
        mesh = mesh.smoothed()
        
        return mesh
        
    def _bake_textures(self, mesh: trimesh.Trimesh, config: ExportConfig) -> trimesh.Trimesh:
        """Bake textures from neural radiance field"""
        # Generate UV coordinates if not present
        if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
            mesh = self._generate_uv_coordinates(mesh)
            
        # Create texture atlas
        texture_size = config.texture_resolution
        texture_atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        # Sample colors from NeRF for each UV coordinate
        uv_coords = mesh.visual.uv
        colors = []
        
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(uv_coords), batch_size):
                batch_uv = uv_coords[i:i+batch_size]
                
                # Convert UV to 3D positions on mesh surface
                batch_positions = self._uv_to_3d_positions(mesh, batch_uv)
                batch_positions = torch.tensor(batch_positions, device=self.device, dtype=torch.float32)
                
                # Create view directions (simple approach - could be improved)
                batch_dirs = torch.zeros_like(batch_positions)
                batch_dirs[:, 2] = 1.0  # Looking along Z-axis
                
                # Sample colors from NeRF
                rgb, _ = self.model(batch_positions, batch_dirs)
                colors.append(rgb.cpu().numpy())
                
        colors = np.concatenate(colors, axis=0)
        colors = (colors * 255).astype(np.uint8)
        
        # Map colors to texture atlas
        for i, (uv, color) in enumerate(zip(uv_coords, colors)):
            u, v = int(uv[0] * texture_size), int(uv[1] * texture_size)
            u = max(0, min(u, texture_size - 1))
            v = max(0, min(v, texture_size - 1))
            texture_atlas[v, u] = color
            
        # Apply texture to mesh
        mesh.visual.texture = texture_atlas
        mesh.visual.face_materials = np.zeros(len(mesh.faces), dtype=int)
        
        return mesh
        
    def _generate_uv_coordinates(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Generate UV coordinates for mesh"""
        # Simple spherical UV mapping
        vertices = mesh.vertices
        
        # Convert to spherical coordinates
        r = np.linalg.norm(vertices, axis=1)
        theta = np.arccos(vertices[:, 2] / (r + 1e-8))
        phi = np.arctan2(vertices[:, 1], vertices[:, 0])
        
        # Convert to UV coordinates
        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi
        
        # Create UV coordinates for each face
        uv_coords = []
        for face in mesh.faces:
            face_uv = []
            for vertex_idx in face:
                face_uv.append([u[vertex_idx], v[vertex_idx]])
            uv_coords.extend(face_uv)
            
        mesh.visual.uv = np.array(uv_coords)
        return mesh
        
    def _uv_to_3d_positions(self, mesh: trimesh.Trimesh, uv_coords: np.ndarray) -> np.ndarray:
        """Convert UV coordinates to 3D positions on mesh surface"""
        # Simple barycentric interpolation
        positions = []
        
        for uv in uv_coords:
            # Find closest face
            face_idx = 0  # Simplified - could use more sophisticated mapping
            face = mesh.faces[face_idx]
            
            # Get face vertices
            v1, v2, v3 = mesh.vertices[face]
            
            # Simple interpolation (could be improved with proper barycentric coordinates)
            position = (v1 + v2 + v3) / 3
            positions.append(position)
            
        return np.array(positions)
        
    def _export_multiple_formats(self, 
                               mesh: trimesh.Trimesh, 
                               config: ExportConfig,
                               output_dir: str) -> Dict[str, str]:
        """Export mesh in multiple formats"""
        exported_files = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export each format
        for format_enum in ExportFormat:
            if format_enum == config.format:
                try:
                    file_path = self._export_single_format(mesh, format_enum, config, output_dir)
                    exported_files[format_enum.value] = file_path
                except Exception as e:
                    logger.error(f"Failed to export {format_enum.value}: {e}")
                    
        return exported_files
        
    def _export_single_format(self, 
                            mesh: trimesh.Trimesh,
                            format_enum: ExportFormat,
                            config: ExportConfig,
                            output_dir: str) -> str:
        """Export mesh in a single format"""
        timestamp = int(time.time())
        filename = f"nerf_mesh_{timestamp}.{format_enum.value}"
        filepath = os.path.join(output_dir, filename)
        
        if format_enum == ExportFormat.GLTF:
            # Export as GLTF with textures
            mesh.export(filepath, include_normals=True)
            
        elif format_enum == ExportFormat.OBJ:
            # Export as OBJ with material file
            mesh.export(filepath, include_normals=True)
            
        elif format_enum == ExportFormat.PLY:
            # Export as PLY
            mesh.export(filepath, include_normals=True)
            
        elif format_enum == ExportFormat.USD:
            # Export as USD (requires additional dependencies)
            self._export_usd(mesh, filepath, config)
            
        elif format_enum == ExportFormat.FBX:
            # Export as FBX (requires additional dependencies)
            self._export_fbx(mesh, filepath, config)
            
        elif format_enum == ExportFormat.STL:
            # Export as STL
            mesh.export(filepath)
            
        return filepath
        
    def _export_usd(self, mesh: trimesh.Trimesh, filepath: str, config: ExportConfig):
        """Export mesh as USD format"""
        try:
            # Try to import USD libraries
            import pxr
            from pxr import Usd, UsdGeom, Sdf, Gf
            
            # Create USD stage
            stage = Usd.Stage.CreateNew(filepath)
            
            # Create mesh
            mesh_path = "/World/NeRFMesh"
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            
            # Set vertices
            usd_mesh.CreatePointsAttr().Set(mesh.vertices.tolist())
            
            # Set face counts and indices
            face_counts = [len(face) for face in mesh.faces]
            face_indices = [idx for face in mesh.faces for idx in face]
            usd_mesh.CreateFaceVertexCountsAttr().Set(face_counts)
            usd_mesh.CreateFaceVertexIndicesAttr().Set(face_indices)
            
            # Set normals if available
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                usd_mesh.CreateNormalsAttr().Set(mesh.vertex_normals.tolist())
                
            # Add texture if available
            if hasattr(mesh.visual, 'texture') and mesh.visual.texture is not None:
                # Create texture file
                texture_path = filepath.replace('.usd', '_texture.png')
                import imageio
                imageio.imwrite(texture_path, mesh.visual.texture)
                
                # Create material
                material_path = "/World/Material"
                material = pxr.UsdShade.Material.Define(stage, material_path)
                
                # Create shader
                shader_path = "/World/Material/Shader"
                shader = pxr.UsdShade.Shader.Define(stage, shader_path)
                shader.CreateIdAttr("UsdPreviewSurface")
                
                # Connect texture
                texture_input = shader.CreateInput("diffuseColor", pxr.Sdf.ValueTypeNames.Color3f)
                texture_file = pxr.UsdShade.Shader.Define(stage, "/World/Material/Texture")
                texture_file.CreateIdAttr("UsdUVTexture")
                texture_file.CreateInput("file", pxr.Sdf.ValueTypeNames.Asset).Set(texture_path)
                texture_file.CreateOutput("rgb", pxr.Sdf.ValueTypeNames.Color3f).ConnectToSource(texture_input)
                
                # Bind material to mesh
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                usd_mesh.GetPrim().ApplyAPI(pxr.UsdShade.MaterialBindingAPI)
                pxr.UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)
                
            stage.Save()
            
        except ImportError:
            # Fallback to simple USD export without textures
            logger.warning("USD libraries not available, creating simple USD file")
            self._export_simple_usd(mesh, filepath)
            
    def _export_simple_usd(self, mesh: trimesh.Trimesh, filepath: str):
        """Export simple USD file without external dependencies"""
        usd_content = f"""#usda 1.0
(
    "Generated by NeRF Studio"
)

def Mesh "NeRFMesh"
{{
    float3[] extent = [{mesh.bounds[0][0]}, {mesh.bounds[0][1]}, {mesh.bounds[1][0]}, {mesh.bounds[1][1]}, {mesh.bounds[2][0]}, {mesh.bounds[2][1]}]
    int[] faceVertexCounts = [{', '.join(str(len(face)) for face in mesh.faces)}]
    int[] faceVertexIndices = [{', '.join(str(idx) for face in mesh.faces for idx in face)}]
    point3f[] points = [{', '.join(f'({v[0]}, {v[1]}, {v[2]})' for v in mesh.vertices)}]
}}
"""
        with open(filepath, 'w') as f:
            f.write(usd_content)
            
    def _export_fbx(self, mesh: trimesh.Trimesh, filepath: str, config: ExportConfig):
        """Export mesh as FBX format"""
        try:
            # Try to use trimesh's FBX export
            mesh.export(filepath)
        except Exception as e:
            logger.warning(f"FBX export failed: {e}")
            # Fallback to OBJ format
            fallback_path = filepath.replace('.fbx', '.obj')
            mesh.export(fallback_path)
            logger.info(f"Exported as OBJ instead: {fallback_path}")
            
    def _compress_files(self, exported_files: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """Compress exported files into a zip archive"""
        timestamp = int(time.time())
        zip_path = os.path.join(output_dir, f"nerf_export_{timestamp}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for format_name, file_path in exported_files.items():
                if os.path.exists(file_path):
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    
        return {"compressed": zip_path}


class ExportProgressTracker:
    """Track export progress and provide status updates"""
    
    def __init__(self):
        self.stages = {
            "initialization": 0.0,
            "mesh_extraction": 0.1,
            "optimization": 0.3,
            "texture_baking": 0.5,
            "export": 0.7,
            "compression": 0.9,
            "complete": 1.0
        }
        self.current_stage = "initialization"
        self.stage_progress = 0.0
        self.messages = []
        
    def update(self, stage: str, progress: float, message: str = ""):
        """Update progress"""
        self.current_stage = stage
        self.stage_progress = progress
        if message:
            self.messages.append(f"{stage}: {message}")
            
    def get_overall_progress(self) -> float:
        """Get overall progress as percentage"""
        stage_start = self.stages.get(self.current_stage, 0.0)
        stage_end = 1.0
        for stage, progress in self.stages.items():
            if stage == self.current_stage:
                break
            stage_start = progress
            
        for stage, progress in self.stages.items():
            if stage == self.current_stage:
                stage_end = progress
                break
                
        stage_progress = stage_end - stage_start
        return stage_start + (stage_progress * self.stage_progress)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "stage": self.current_stage,
            "progress": self.get_overall_progress(),
            "message": self.messages[-1] if self.messages else "",
            "messages": self.messages
        }


def create_advanced_exporter(model, device='cpu') -> AdvancedMeshExporter:
    """Factory function to create advanced mesh exporter"""
    return AdvancedMeshExporter(model, device)


def export_with_progress_tracking(model, 
                                config: ExportConfig,
                                output_dir: str,
                                progress_callback=None) -> Dict[str, str]:
    """Export mesh with progress tracking"""
    exporter = AdvancedMeshExporter(model)
    
    if progress_callback:
        exporter.set_progress_callback(progress_callback)
        
    return exporter.extract_mesh_with_textures(config, output_dir) 