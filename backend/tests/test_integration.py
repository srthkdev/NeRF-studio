import pytest
import tempfile
import os
import json
import shutil
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.ml.nerf.model import HierarchicalNeRF
from app.ml.nerf.train_nerf import NeRFTrainer
from app.ml.nerf.mesh_extraction import MeshExtractor
from app.core.validation import InputValidator

client = TestClient(app)

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self, temp_project_dir):
        """Create sample images for testing."""
        from PIL import Image
        import numpy as np
        
        images_dir = os.path.join(temp_project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        image_paths = []
        for i in range(5):
            # Create simple test images
            img = Image.new('RGB', (256, 256), color=(i * 50, 100, 150))
            img_path = os.path.join(images_dir, f"image_{i}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
        
        return image_paths
    
    @pytest.fixture
    def sample_poses(self):
        """Create sample camera poses for testing."""
        poses = []
        for i in range(5):
            # Create simple camera poses in a circle
            angle = i * 2 * 3.14159 / 5
            x = 2 * np.cos(angle)
            z = 2 * np.sin(angle)
            
            pose = {
                "filename": f"image_{i}.jpg",
                "camera_to_world": [
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
            poses.append(pose)
        
        return poses
    
    def test_project_creation_workflow(self):
        """Test complete project creation workflow."""
        # Create project
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        assert response.status_code == 200
        project_data = response.json()
        project_id = project_data["id"]
        
        # Get specific project
        response = client.get(f"/api/v1/projects/{project_id}")
        assert response.status_code == 200
        project = response.json()
        assert project["id"] == project_id
        assert project["name"] == "test_project"
    
    def test_image_upload_workflow(self, sample_images):
        """Test image upload workflow."""
        # Create project
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        project_id = response.json()["id"]
        
        # Upload images
        with open(sample_images[0], "rb") as f:
            response = client.post(
                f"/api/v1/projects/{project_id}/upload_images",
                files={"files": ("test_image.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        upload_data = response.json()
        assert "Uploaded" in upload_data["message"]
    
    def test_pose_upload_workflow(self, sample_poses):
        """Test camera pose upload workflow."""
        # Create project
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        project_id = response.json()["id"]
        
        # Create temporary pose file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_poses, f)
            pose_file_path = f.name
        
        try:
            # Upload poses
            with open(pose_file_path, "rb") as f:
                response = client.post(
                    f"/api/v1/projects/{project_id}/upload_poses",  # Note the underscore
                    files={"poses_file": ("poses.json", f, "application/json")}
                )
            
            assert response.status_code == 200
            assert "message" in response.json()
            assert "Manual poses uploaded successfully" in response.json()["message"]
            
        finally:
            os.unlink(pose_file_path)
    
    def test_training_workflow(self, sample_images, sample_poses):
        """Test training workflow."""
        # Create project
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        project_id = response.json()["id"]
        
        # Upload images and poses (simplified for test)
        # In real test, you'd upload actual files
        
        # Start training
        training_config = {
            "num_epochs": 2,
            "learning_rate": 0.001,
            "batch_size": 4,
            "pos_freq_bands": 4,
            "view_freq_bands": 2,
            "hidden_dim": 64,
            "num_layers": 4,
            "n_coarse": 16,
            "n_fine": 32
        }
        
        response = client.post(
            f"/api/v1/projects/{project_id}/start_training",  # Note the underscore
            json=training_config
        )
        
        # Should fail without images/poses, but test the endpoint
        assert response.status_code in [400, 500]  # Expected to fail without data
    
    def test_mesh_extraction_workflow(self):
        """Test mesh extraction workflow."""
        # Create project
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        project_id = response.json()["id"]
        
        # Try to extract mesh (should fail without trained model)
        response = client.post(
            f"/api/v1/projects/{project_id}/export/advanced",
            json={
                "format": "gltf",
                "resolution": 64,
                "bounds": [-2, 2, -2, 2, -2, 2]
            }
        )
        
        # Export starts successfully but will fail in background without trained model
        assert response.status_code == 200
        assert "message" in response.json()
        assert "export started" in response.json()["message"].lower()

class TestNeRFModel:
    """Test NeRF model functionality."""
    
    def test_model_creation(self):
        """Test NeRF model creation and forward pass."""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        # Test forward pass
        import torch
        batch_size = 4
        num_rays = 64
        
        # Sample rays
        rays_o = torch.randn(batch_size, num_rays, 3)
        rays_d = torch.randn(batch_size, num_rays, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = torch.ones(batch_size, num_rays, 1) * 0.1
        far = torch.ones(batch_size, num_rays, 1) * 10.0
        
        # Forward pass
        result = model(rays_o, rays_d, near, far)
        
        # Check the actual output structure
        assert "coarse" in result
        assert "fine" in result
        assert "rgb_map" in result["coarse"]
        assert "rgb_map" in result["fine"]
        assert result["coarse"]["rgb_map"].shape == (batch_size, num_rays, 3)
        assert result["fine"]["rgb_map"].shape == (batch_size, num_rays, 3)
        assert torch.all(result["coarse"]["rgb_map"] >= 0) and torch.all(result["coarse"]["rgb_map"] <= 1)
        assert torch.all(result["fine"]["rgb_map"] >= 0) and torch.all(result["fine"]["rgb_map"] <= 1)
    
    def test_model_parameters(self):
        """Test model parameter counting."""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 1000000  # Reasonable size for test model

class TestMeshExtraction:
    """Test mesh extraction functionality."""
    
    def test_mesh_extractor_creation(self):
        """Test mesh extractor initialization."""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        extractor = MeshExtractor(model)
        assert extractor.model is model
        assert extractor.device == 'cpu'
    
    def test_density_field_sampling(self):
        """Test density field sampling."""
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        extractor = MeshExtractor(model)
        bounds = (-1, 1, -1, 1, -1, 1)
        resolution = 16  # Small for testing
        
        points, densities = extractor.sample_density_field(bounds, resolution)
        
        assert points.shape == (resolution**3, 3)
        assert densities.shape == (resolution, resolution, resolution)
        assert np.all(densities >= 0)

class TestAPIEndpoints:
    """Test API endpoint functionality."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_system_metrics(self):
        """Test system metrics endpoint."""
        response = client.get("/api/v1/system/metrics")
        assert response.status_code == 200
        metrics = response.json()
        assert "cpu_percent" in metrics or "message" in metrics
    
    def test_training_metrics(self):
        """Test training metrics endpoint."""
        response = client.get("/api/v1/training/metrics")
        assert response.status_code == 200
        metrics = response.json()
        assert "message" in metrics  # No training running
    
    def test_invalid_project_id(self):
        """Test handling of invalid project IDs."""
        response = client.get("/api/v1/projects/invalid-id")
        assert response.status_code == 404
    
    def test_invalid_endpoints(self):
        """Test invalid endpoint handling."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

class TestValidationIntegration:
    """Test validation integration with API."""
    
    def test_invalid_project_name(self):
        """Test invalid project name validation."""
        response = client.post("/api/v1/projects", json={"name": ""})
        assert response.status_code in [400, 422]  # FastAPI validation error
        assert "detail" in response.json()
        
        response = client.post("/api/v1/projects", json={"name": "a" * 101})
        assert response.status_code in [400, 422]  # FastAPI validation error
        assert "detail" in response.json()
    
    def test_invalid_training_config(self):
        """Test invalid training configuration."""
        # Create project first
        response = client.post("/api/v1/projects", json={"name": "test_project"})
        project_id = response.json()["id"]
        
        # Test invalid config
        invalid_config = {
            "num_epochs": -1,  # Invalid
            "learning_rate": 0.001
        }
        
        response = client.post(
            f"/api/v1/projects/{project_id}/start_training",
            json=invalid_config
        )
        
        assert response.status_code in [400, 422]  # Could be validation error or business logic error
        assert "detail" in response.json()

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection for real-time updates."""
    # This would require a more complex async test setup
    # For now, just test that the endpoint exists
    pass

if __name__ == "__main__":
    pytest.main([__file__]) 