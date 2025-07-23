import os
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error."""
    pass

class InputValidator:
    """Input validation and sanitization utilities."""
    
    # Supported image formats
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Maximum file sizes (in bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_POSE_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Training parameter bounds
    TRAINING_BOUNDS = {
        'num_epochs': (1, 10000),
        'learning_rate': (1e-6, 1.0),
        'batch_size': (1, 1024),
        'pos_freq_bands': (1, 20),
        'view_freq_bands': (1, 10),
        'hidden_dim': (32, 1024),
        'num_layers': (2, 16),
        'n_coarse': (16, 256),
        'n_fine': (32, 512)
    }
    
    @staticmethod
    def validate_image_file(file_path: str) -> Dict[str, Any]:
        """
        Validate image file format and properties.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with image metadata
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                raise ValidationError(f"Image file not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > InputValidator.MAX_IMAGE_SIZE:
                raise ValidationError(f"Image file too large: {file_size} bytes (max: {InputValidator.MAX_IMAGE_SIZE})")
            
            # Check file extension
            _, ext = os.path.splitext(file_path.lower())
            if ext not in InputValidator.SUPPORTED_IMAGE_FORMATS:
                raise ValidationError(f"Unsupported image format: {ext}")
            
            # Validate image with PIL
            with Image.open(file_path) as img:
                # Check image dimensions
                width, height = img.size
                if width < 64 or height < 64:
                    raise ValidationError(f"Image too small: {width}x{height} (min: 64x64)")
                if width > 8192 or height > 8192:
                    raise ValidationError(f"Image too large: {width}x{height} (max: 8192x8192)")
                
                # Check image mode
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    raise ValidationError(f"Unsupported image mode: {img.mode}")
                
                return {
                    'width': width,
                    'height': height,
                    'mode': img.mode,
                    'format': img.format,
                    'size': file_size
                }
                
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Image validation failed: {str(e)}")
    
    @staticmethod
    def validate_camera_poses(poses_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate camera pose data.
        
        Args:
            poses_data: List of camera pose dictionaries
            
        Returns:
            Validated and sanitized pose data
            
        Raises:
            ValidationError: If validation fails
        """
        if not poses_data:
            raise ValidationError("No camera poses provided")
        
        validated_poses = []
        
        for i, pose in enumerate(poses_data):
            try:
                # Check required fields
                required_fields = ['filename', 'camera_to_world']
                for field in required_fields:
                    if field not in pose:
                        raise ValidationError(f"Missing required field '{field}' in pose {i}")
                
                # Validate filename
                filename = pose['filename']
                if not isinstance(filename, str) or not filename.strip():
                    raise ValidationError(f"Invalid filename in pose {i}")
                
                # Validate camera_to_world matrix
                camera_to_world = pose['camera_to_world']
                if not isinstance(camera_to_world, list) or len(camera_to_world) != 4:
                    raise ValidationError(f"Invalid camera_to_world matrix in pose {i}")
                
                for row in camera_to_world:
                    if not isinstance(row, list) or len(row) != 4:
                        raise ValidationError(f"Invalid camera_to_world matrix row in pose {i}")
                    
                    for val in row:
                        if not isinstance(val, (int, float)):
                            raise ValidationError(f"Invalid matrix value in pose {i}")
                
                # Convert to numpy for additional validation
                matrix = np.array(camera_to_world)
                
                # Check if matrix is valid (not all zeros, reasonable scale)
                if np.allclose(matrix, 0):
                    raise ValidationError(f"Camera matrix is all zeros in pose {i}")
                
                # Check scale (camera should be within reasonable bounds)
                translation = matrix[:3, 3]
                if np.any(np.abs(translation) > 1000):
                    raise ValidationError(f"Camera translation too large in pose {i}")
                
                validated_poses.append(pose)
                
            except Exception as e:
                if isinstance(e, ValidationError):
                    raise
                raise ValidationError(f"Pose validation failed for pose {i}: {str(e)}")
        
        return validated_poses
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training configuration parameters.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Validated and sanitized configuration
            
        Raises:
            ValidationError: If validation fails
        """
        validated_config = {}
        
        for param, value in config.items():
            if param in InputValidator.TRAINING_BOUNDS:
                min_val, max_val = InputValidator.TRAINING_BOUNDS[param]
                
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Invalid type for {param}: expected number")
                
                if value < min_val or value > max_val:
                    raise ValidationError(f"{param} out of bounds: {value} (range: {min_val}-{max_val})")
                
                validated_config[param] = value
            else:
                # Unknown parameter - log warning but allow
                logger.warning(f"Unknown training parameter: {param}")
                validated_config[param] = value
        
        return validated_config
    
    @staticmethod
    def validate_project_name(name: str) -> str:
        """
        Validate and sanitize project name.
        
        Args:
            name: Project name
            
        Returns:
            Sanitized project name
            
        Raises:
            ValidationError: If validation fails
        """
        if not name or not name.strip():
            raise ValidationError("Project name cannot be empty")
        
        # Remove leading/trailing whitespace
        name = name.strip()
        
        # Check length
        if len(name) > 100:
            raise ValidationError("Project name too long (max: 100 characters)")
        
        # Check for invalid characters
        if re.search(r'[<>:"/\\|?*]', name):
            raise ValidationError("Project name contains invalid characters")
        
        return name
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = True) -> str:
        """
        Validate file path for security.
        
        Args:
            path: File path
            must_exist: Whether file must exist
            
        Returns:
            Normalized file path
            
        Raises:
            ValidationError: If validation fails
        """
        # Normalize path
        path = os.path.normpath(path)
        
        # Check for path traversal attempts
        if '..' in path or path.startswith('/'):
            raise ValidationError("Invalid file path")
        
        # Check if file exists (if required)
        if must_exist and not os.path.exists(path):
            raise ValidationError(f"File not found: {path}")
        
        return path
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove leading/trailing whitespace and dots
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename or 'unnamed'
    
    @staticmethod
    def validate_mesh_extraction_params(bounds: List[float], 
                                      resolution: int,
                                      formats: List[str]) -> Tuple[List[float], int, List[str]]:
        """
        Validate mesh extraction parameters.
        
        Args:
            bounds: Scene bounds [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution: Grid resolution
            formats: Export formats
            
        Returns:
            Validated parameters
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate bounds
        if len(bounds) != 6:
            raise ValidationError("Bounds must have exactly 6 values")
        
        for i, bound in enumerate(bounds):
            if not isinstance(bound, (int, float)):
                raise ValidationError(f"Invalid bound value at index {i}")
        
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            raise ValidationError("Invalid bounds: min values must be less than max values")
        
        # Validate resolution
        if not isinstance(resolution, int) or resolution < 16 or resolution > 512:
            raise ValidationError("Resolution must be an integer between 16 and 512")
        
        # Validate formats
        valid_formats = {'gltf', 'obj', 'ply'}
        if not formats:
            raise ValidationError("At least one export format must be specified")
        
        for fmt in formats:
            if fmt.lower() not in valid_formats:
                raise ValidationError(f"Unsupported format: {fmt}")
        
        return bounds, resolution, [fmt.lower() for fmt in formats]

def validate_api_input(data: Dict[str, Any], required_fields: List[str] = None) -> Dict[str, Any]:
    """
    Generic API input validation.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError("Input must be a dictionary")
    
    if required_fields:
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}")
    
    return data 