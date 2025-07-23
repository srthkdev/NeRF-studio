import pytest
import tempfile
import os
import json
import numpy as np
from PIL import Image
from app.core.validation import InputValidator, ValidationError, validate_api_input

class TestInputValidator:
    """Test input validation functionality."""
    
    def test_validate_project_name(self):
        """Test project name validation."""
        # Valid names
        assert InputValidator.validate_project_name("test") == "test"
        assert InputValidator.validate_project_name("  test  ") == "test"
        assert InputValidator.validate_project_name("test-project") == "test-project"
        
        # Invalid names
        with pytest.raises(ValidationError):
            InputValidator.validate_project_name("")
        
        with pytest.raises(ValidationError):
            InputValidator.validate_project_name("   ")
        
        with pytest.raises(ValidationError):
            InputValidator.validate_project_name("a" * 101)  # Too long
        
        with pytest.raises(ValidationError):
            InputValidator.validate_project_name("test<file")
    
    def test_validate_image_file(self):
        """Test image file validation."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Create a simple test image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp_file.name)
            
            # Test valid image
            result = InputValidator.validate_image_file(tmp_file.name)
            assert result['width'] == 100
            assert result['height'] == 100
            assert result['mode'] == 'RGB'
            assert result['format'] == 'PNG'
            
            # Clean up
            os.unlink(tmp_file.name)
        
        # Test non-existent file
        with pytest.raises(ValidationError):
            InputValidator.validate_image_file("nonexistent.png")
    
    def test_validate_camera_poses(self):
        """Test camera pose validation."""
        # Valid poses
        valid_poses = [
            {
                "filename": "image1.jpg",
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        ]
        
        result = InputValidator.validate_camera_poses(valid_poses)
        assert len(result) == 1
        assert result[0]["filename"] == "image1.jpg"
        
        # Invalid poses
        with pytest.raises(ValidationError):
            InputValidator.validate_camera_poses([])
        
        with pytest.raises(ValidationError):
            InputValidator.validate_camera_poses([
                {"filename": "test.jpg"}  # Missing camera_to_world
            ])
        
        with pytest.raises(ValidationError):
            InputValidator.validate_camera_poses([
                {
                    "filename": "test.jpg",
                    "camera_to_world": [[1, 2, 3]]  # Wrong matrix size
                }
            ])
    
    def test_validate_training_config(self):
        """Test training configuration validation."""
        # Valid config
        valid_config = {
            "num_epochs": 100,
            "learning_rate": 0.001,
            "batch_size": 32,
            "pos_freq_bands": 10,
            "view_freq_bands": 4,
            "hidden_dim": 256,
            "num_layers": 8,
            "n_coarse": 64,
            "n_fine": 128
        }
        
        result = InputValidator.validate_training_config(valid_config)
        assert result == valid_config
        
        # Invalid config
        with pytest.raises(ValidationError):
            InputValidator.validate_training_config({
                "num_epochs": -1  # Out of bounds
            })
        
        with pytest.raises(ValidationError):
            InputValidator.validate_training_config({
                "learning_rate": "invalid"  # Wrong type
            })
    
    def test_validate_mesh_extraction_params(self):
        """Test mesh extraction parameters validation."""
        # Valid parameters
        bounds = [-2, 2, -2, 2, -2, 2]
        resolution = 128
        formats = ["gltf", "obj", "ply"]
        
        result = InputValidator.validate_mesh_extraction_params(bounds, resolution, formats)
        assert result[0] == bounds
        assert result[1] == resolution
        assert result[2] == ["gltf", "obj", "ply"]
        
        # Invalid bounds
        with pytest.raises(ValidationError):
            InputValidator.validate_mesh_extraction_params([1, 2], 128, ["gltf"])
        
        with pytest.raises(ValidationError):
            InputValidator.validate_mesh_extraction_params([2, 1, -2, 2, -2, 2], 128, ["gltf"])
        
        # Invalid resolution
        with pytest.raises(ValidationError):
            InputValidator.validate_mesh_extraction_params(bounds, 8, ["gltf"])
        
        # Invalid formats
        with pytest.raises(ValidationError):
            InputValidator.validate_mesh_extraction_params(bounds, 128, ["invalid"])
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert InputValidator.sanitize_filename("test.jpg") == "test.jpg"
        assert InputValidator.sanitize_filename("test<file.jpg") == "test_file.jpg"
        assert InputValidator.sanitize_filename("  test.jpg  ") == "test.jpg"
        assert InputValidator.sanitize_filename("") == "unnamed"
        assert InputValidator.sanitize_filename(".") == "unnamed"
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Create temporary file in current directory
        with tempfile.NamedTemporaryFile(delete=False, dir='.') as tmp_file:
            tmp_path = tmp_file.name
        
        # Valid path
        result = InputValidator.validate_file_path(tmp_path)
        assert result == tmp_path
        
        # Non-existent file
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path("nonexistent.txt")
        
        # Non-existent file (not required)
        result = InputValidator.validate_file_path("nonexistent.txt", must_exist=False)
        assert result == "nonexistent.txt"
        
        # Invalid path
        with pytest.raises(ValidationError):
            InputValidator.validate_file_path("../test.txt")
        
        # Clean up
        os.unlink(tmp_path)

class TestValidationHelpers:
    """Test validation helper functions."""
    
    def test_validate_api_input(self):
        """Test API input validation."""
        # Valid input
        data = {"key": "value"}
        result = validate_api_input(data)
        assert result == data
        
        # Valid input with required fields
        result = validate_api_input(data, required_fields=["key"])
        assert result == data
        
        # Invalid input
        with pytest.raises(ValidationError):
            validate_api_input("not a dict")
        
        # Missing required field
        with pytest.raises(ValidationError):
            validate_api_input({"key": "value"}, required_fields=["missing"]) 