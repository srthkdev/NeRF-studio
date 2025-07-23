import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from app.ml.nerf.advanced_export import (
    AdvancedMeshExporter, 
    ExportConfig, 
    ExportFormat, 
    ExportProgressTracker,
    TextureBaker,
    MeshOptimizer
)
from app.ml.nerf.model import HierarchicalNeRF


class TestExportProgressTracker:
    def test_initialization(self):
        tracker = ExportProgressTracker()
        assert tracker.stage == "idle"
        assert tracker.progress == 0.0
        assert tracker.message == ""
        assert tracker.messages == []

    def test_update(self):
        tracker = ExportProgressTracker()
        tracker.update("mesh_extraction", 0.5, "Extracting mesh...")
        
        assert tracker.stage == "mesh_extraction"
        assert tracker.progress == 0.5
        assert tracker.message == "Extracting mesh..."
        assert len(tracker.messages) == 1

    def test_get_status(self):
        tracker = ExportProgressTracker()
        tracker.update("texture_baking", 0.75, "Baking textures...")
        
        status = tracker.get_status()
        assert status["stage"] == "texture_baking"
        assert status["progress"] == 0.75
        assert status["message"] == "Baking textures..."
        assert "messages" in status

    def test_complete(self):
        tracker = ExportProgressTracker()
        tracker.complete("Export completed successfully")
        
        assert tracker.stage == "complete"
        assert tracker.progress == 1.0
        assert tracker.message == "Export completed successfully"

    def test_error(self):
        tracker = ExportProgressTracker()
        tracker.error("Export failed")
        
        assert tracker.stage == "error"
        assert tracker.progress == 0.0
        assert tracker.message == "Export failed"


class TestExportConfig:
    def test_default_config(self):
        config = ExportConfig(format=ExportFormat.GLTF)
        assert config.format == ExportFormat.GLTF
        assert config.resolution == 128
        assert config.texture_resolution == 1024
        assert config.include_textures is True
        assert config.bake_textures is True
        assert config.optimize_mesh is True
        assert config.compression is True
        assert config.quality == "high"
        assert config.bounds == [-2, 2, -2, 2, -2, 2]

    def test_custom_config(self):
        config = ExportConfig(
            format=ExportFormat.USD,
            resolution=256,
            texture_resolution=2048,
            include_textures=False,
            bake_textures=False,
            optimize_mesh=False,
            compression=False,
            quality="low",
            bounds=[-1, 1, -1, 1, -1, 1]
        )
        assert config.format == ExportFormat.USD
        assert config.resolution == 256
        assert config.texture_resolution == 2048
        assert config.include_textures is False
        assert config.bake_textures is False
        assert config.optimize_mesh is False
        assert config.compression is False
        assert config.quality == "low"
        assert config.bounds == [-1, 1, -1, 1, -1, 1]


class TestTextureBaker:
    def test_initialization(self):
        baker = TextureBaker()
        assert baker is not None

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_bake_texture_from_nerf(self, mock_trimesh):
        # Mock model
        model = Mock()
        model.forward.return_value = (
            torch.randn(100, 3),  # RGB values
            torch.randn(100, 1)   # Density values
        )
        
        # Mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_mesh.faces = np.random.randint(0, 100, (50, 3))
        
        baker = TextureBaker()
        texture = baker.bake_texture_from_nerf(model, mock_mesh, 512)
        
        assert texture.shape == (512, 512, 3)
        assert texture.dtype == np.uint8

    def test_generate_uv_coordinates(self):
        baker = TextureBaker()
        vertices = np.random.rand(100, 3)
        faces = np.random.randint(0, 100, (50, 3))
        
        uvs = baker.generate_uv_coordinates(vertices, faces)
        assert uvs.shape == (100, 2)
        assert np.all(uvs >= 0) and np.all(uvs <= 1)


class TestMeshOptimizer:
    def test_initialization(self):
        optimizer = MeshOptimizer()
        assert optimizer is not None

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_optimize_mesh(self, mock_trimesh):
        # Mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = np.random.rand(100, 3)
        mock_mesh.faces = np.random.randint(0, 100, (50, 3))
        mock_mesh.process.return_value = mock_mesh
        
        optimizer = MeshOptimizer()
        optimized_mesh = optimizer.optimize_mesh(mock_mesh, "high")
        
        assert optimized_mesh is not None
        mock_mesh.process.assert_called()

    def test_decimate_mesh(self):
        optimizer = MeshOptimizer()
        vertices = np.random.rand(100, 3)
        faces = np.random.randint(0, 100, (50, 3))
        
        decimated_vertices, decimated_faces = optimizer.decimate_mesh(
            vertices, faces, target_faces=25
        )
        
        assert len(decimated_faces) <= 25
        assert len(decimated_vertices) <= 100


class TestAdvancedMeshExporter:
    def setup_method(self):
        # Create a simple NeRF model for testing
        self.model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=32,
            n_fine=64
        )
        self.device = "cpu"
        self.exporter = AdvancedMeshExporter(self.model, self.device)

    def test_initialization(self):
        assert self.exporter.model is not None
        assert self.exporter.device == "cpu"
        assert self.exporter.progress_callback is None

    def test_set_progress_callback(self):
        callback = Mock()
        self.exporter.set_progress_callback(callback)
        assert self.exporter.progress_callback == callback

    def test_extract_density_grid(self):
        config = ExportConfig(format=ExportFormat.GLTF, resolution=64)
        
        density_grid = self.exporter.extract_density_grid(config)
        
        assert density_grid.shape == (64, 64, 64)
        assert density_grid.dtype == torch.float32

    def test_marching_cubes(self):
        density_grid = torch.randn(32, 32, 32)
        
        vertices, faces = self.exporter.marching_cubes(density_grid, 0.5)
        
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(vertices) > 0
        assert len(faces) > 0

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_create_mesh(self, mock_trimesh):
        vertices = np.random.rand(100, 3)
        faces = np.random.randint(0, 100, (50, 3))
        
        mock_mesh = Mock()
        mock_trimesh.Trimesh.return_value = mock_mesh
        
        mesh = self.exporter.create_mesh(vertices, faces)
        
        assert mesh is not None
        mock_trimesh.Trimesh.assert_called_with(vertices=vertices, faces=faces)

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_export_gltf(self, mock_trimesh):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock mesh
            mock_mesh = Mock()
            mock_mesh.export.return_value = b"gltf_data"
            
            config = ExportConfig(format=ExportFormat.GLTF)
            output_path = os.path.join(temp_dir, "test.gltf")
            
            result = self.exporter.export_gltf(mock_mesh, config, output_path)
            
            assert result == output_path
            mock_mesh.export.assert_called()

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_export_obj(self, mock_trimesh):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock mesh
            mock_mesh = Mock()
            mock_mesh.export.return_value = b"obj_data"
            
            config = ExportConfig(format=ExportFormat.OBJ)
            output_path = os.path.join(temp_dir, "test.obj")
            
            result = self.exporter.export_obj(mock_mesh, config, output_path)
            
            assert result == output_path
            mock_mesh.export.assert_called()

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_export_ply(self, mock_trimesh):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock mesh
            mock_mesh = Mock()
            mock_mesh.export.return_value = b"ply_data"
            
            config = ExportConfig(format=ExportFormat.PLY)
            output_path = os.path.join(temp_dir, "test.ply")
            
            result = self.exporter.export_ply(mock_mesh, config, output_path)
            
            assert result == output_path
            mock_mesh.export.assert_called()

    @patch('backend.app.ml.nerf.advanced_export.trimesh')
    def test_export_usd(self, mock_trimesh):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock mesh
            mock_mesh = Mock()
            mock_mesh.export.return_value = b"usd_data"
            
            config = ExportConfig(format=ExportFormat.USD)
            output_path = os.path.join(temp_dir, "test.usd")
            
            result = self.exporter.export_usd(mock_mesh, config, output_path)
            
            assert result == output_path
            mock_mesh.export.assert_called()

    def test_extract_mesh_with_textures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                format=ExportFormat.GLTF,
                resolution=32,
                include_textures=True,
                bake_textures=True
            )
            
            # Mock progress callback
            callback = Mock()
            self.exporter.set_progress_callback(callback)
            
            # Mock texture baking
            with patch.object(self.exporter, 'bake_texture') as mock_bake:
                mock_bake.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                
                result = self.exporter.extract_mesh_with_textures(config, temp_dir)
                
                assert isinstance(result, dict)
                assert "gltf" in result
                assert os.path.exists(result["gltf"])
                
                # Verify progress callback was called
                assert callback.call_count > 0

    def test_extract_mesh_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                format=ExportFormat.GLTF,
                resolution=32,
                include_textures=False,
                bake_textures=False
            )
            
            result = self.exporter.extract_mesh_with_textures(config, temp_dir)
            
            assert isinstance(result, dict)
            assert "gltf" in result
            assert os.path.exists(result["gltf"])

    def test_multiple_formats(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                format=ExportFormat.GLTF,
                resolution=32,
                include_textures=False
            )
            
            # Test multiple formats
            formats = [ExportFormat.GLTF, ExportFormat.OBJ, ExportFormat.PLY]
            results = {}
            
            for fmt in formats:
                config.format = fmt
                result = self.exporter.extract_mesh_with_textures(config, temp_dir)
                results[fmt.value] = result[fmt.value]
            
            # Verify all files were created
            for fmt, path in results.items():
                assert os.path.exists(path)

    def test_error_handling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(format=ExportFormat.GLTF, resolution=32)
            
            # Test with invalid model
            invalid_model = Mock()
            invalid_model.forward.side_effect = Exception("Model error")
            
            exporter = AdvancedMeshExporter(invalid_model, self.device)
            
            with pytest.raises(Exception):
                exporter.extract_mesh_with_textures(config, temp_dir)

    def test_compression(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                format=ExportFormat.GLTF,
                resolution=32,
                compression=True
            )
            
            result = self.exporter.extract_mesh_with_textures(config, temp_dir)
            
            # Verify compressed file exists
            compressed_path = result["gltf"] + ".gz"
            assert os.path.exists(compressed_path)

    def test_quality_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test different quality settings
            for quality in ["low", "medium", "high"]:
                config = ExportConfig(
                    format=ExportFormat.GLTF,
                    resolution=32,
                    quality=quality
                )
                
                result = self.exporter.extract_mesh_with_textures(config, temp_dir)
                assert os.path.exists(result["gltf"])


if __name__ == "__main__":
    pytest.main([__file__]) 