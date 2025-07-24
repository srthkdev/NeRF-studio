import subprocess
import os
import json
import numpy as np
import logging
import shutil
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class COLMAPProcessor:
    """
    COLMAP processor for automatic camera pose estimation.
    Handles feature extraction, matching, and sparse reconstruction.
    """
    
    def __init__(self, colmap_path: str = "colmap"):
        self.colmap_path = colmap_path
        self.workspace_dir = None
        
    def create_workspace(self, project_dir: Path, image_dir: Path) -> Path:
        """
        Create COLMAP workspace directory structure.
        
        Args:
            project_dir: Project directory (Path object)
            image_dir: Directory containing images (Path object)
            
        Returns:
            Path to COLMAP workspace
        """
        workspace_dir = project_dir / "colmap_workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        # Create COLMAP directory structure
        sparse_dir = workspace_dir / "sparse"
        dense_dir = workspace_dir / "dense"
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        # Copy images to workspace
        images_workspace = workspace_dir / "images"
        images_workspace.mkdir(exist_ok=True)
        
        # Copy images (or create symlinks)
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                dst = images_workspace / img_file.name
                if not dst.exists():
                    # Use symlink for efficiency if not on Windows, otherwise copy
                    if os.name != 'nt':
                        os.symlink(img_file, dst)
                    else:
                        shutil.copy2(img_file, dst)
        
        self.workspace_dir = workspace_dir
        return workspace_dir
    
    def run_feature_extraction(self, quality: str = "medium") -> bool:
        """
        Run COLMAP feature extraction.
        
        Args:
            quality: Feature extraction quality (low, medium, high)
            
        Returns:
            True if successful
        """
        try:
            quality_params = {
                "low": {"max_image_size": 1024, "max_num_features": 2048},
                "medium": {"max_image_size": 2048, "max_num_features": 4096},
                "high": {"max_image_size": 4096, "max_num_features": 8192}
            }
            
            params = quality_params.get(quality, quality_params["medium"])
            
            cmd = [
                self.colmap_path, "feature_extractor",
                "--database_path", os.path.join(self.workspace_dir, "database.db"),
                "--image_path", os.path.join(self.workspace_dir, "images"),
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                "--SiftExtraction.max_image_size", str(params["max_image_size"]),
                "--SiftExtraction.max_num_features", str(params["max_num_features"])
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                logger.error(f"Feature extraction failed: {result.stderr}")
                return False
            
            logger.info("Feature extraction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return False
    
    def run_feature_matching(self, method: str = "exhaustive") -> bool:
        """
        Run COLMAP feature matching.
        
        Args:
            method: Matching method (exhaustive, sequential, vocabulary_tree)
            
        Returns:
            True if successful
        """
        try:
            cmd = [
                self.colmap_path, "exhaustive_matcher" if method == "exhaustive" else "sequential_matcher",
                "--database_path", os.path.join(self.workspace_dir, "database.db")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                logger.error(f"Feature matching failed: {result.stderr}")
                return False
            
            logger.info("Feature matching completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Feature matching error: {e}")
            return False
    
    def run_sparse_reconstruction(self) -> bool:
        """
        Run COLMAP sparse reconstruction.
        
        Returns:
            True if successful
        """
        try:
            sparse_dir = os.path.join(self.workspace_dir, "sparse")
            
            cmd = [
                self.colmap_path, "mapper",
                "--database_path", os.path.join(self.workspace_dir, "database.db"),
                "--image_path", os.path.join(self.workspace_dir, "images"),
                "--output_path", sparse_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                logger.error(f"Sparse reconstruction failed: {result.stderr}")
                return False
            
            logger.info("Sparse reconstruction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Sparse reconstruction error: {e}")
            return False
    
    def extract_camera_poses(self) -> List[Dict[str, Any]]:
        """
        Extract camera poses from COLMAP reconstruction.
        
        Returns:
            List of camera pose dictionaries
        """
        try:
            # Find the reconstruction file
            sparse_dir = os.path.join(self.workspace_dir, "sparse")
            reconstruction_files = list(Path(sparse_dir).glob("*.bin"))
            
            if not reconstruction_files:
                logger.error("No reconstruction files found")
                return []
            
            # Use the first reconstruction file
            reconstruction_file = reconstruction_files[0]
            
            # Convert binary to text format
            text_file = reconstruction_file.with_suffix(".txt")
            cmd = [
                self.colmap_path, "model_converter",
                "--input_path", str(reconstruction_file),
                "--output_path", str(text_file),
                "--output_type", "TXT"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Model conversion failed: {result.stderr}")
                return []
            
            # Parse camera poses
            poses = self._parse_camera_poses(text_file)
            
            return poses
            
        except Exception as e:
            logger.error(f"Camera pose extraction error: {e}")
            return []
    
    def _parse_camera_poses(self, text_file: Path) -> List[Dict[str, Any]]:
        """
        Parse camera poses from COLMAP text format.
        
        Args:
            text_file: Path to COLMAP text reconstruction file
            
        Returns:
            List of camera pose dictionaries
        """
        poses = []
        
        try:
            with open(text_file, 'r') as f:
                lines = f.readlines()
            
            # Find image section
            image_section = False
            for line in lines:
                line = line.strip()
                
                if line == "# Image list with two lines of data per image:":
                    image_section = True
                    continue
                
                if image_section and line.startswith("#"):
                    continue
                
                if image_section and line:
                    # Parse image line
                    parts = line.split()
                    if len(parts) >= 9:
                        image_id = int(parts[0])
                        qw, qx, qy, qz = map(float, parts[1:5])
                        tx, ty, tz = map(float, parts[5:8])
                        filename = parts[9]
                        
                        # Convert quaternion to rotation matrix
                        rotation_matrix = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                        
                        # Create camera-to-world transformation matrix
                        camera_to_world = np.eye(4)
                        camera_to_world[:3, :3] = rotation_matrix
                        camera_to_world[:3, 3] = [tx, ty, tz]
                        
                        poses.append({
                            "filename": filename,
                            "camera_to_world": camera_to_world.tolist(),
                            "image_id": image_id
                        })
            
            logger.info(f"Extracted {len(poses)} camera poses")
            return poses
            
        except Exception as e:
            logger.error(f"Camera pose parsing error: {e}")
            return []
    
    def _quaternion_to_rotation_matrix(self, qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            qw, qx, qy, qz: Quaternion components
            
        Returns:
            3x3 rotation matrix
        """
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*qx**2 - 2*qz**2, 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return R
    
    def run_full_pipeline(self, project_dir: str, image_dir: str, quality: str = "medium") -> List[Dict[str, Any]]:
        """
        Run full COLMAP pipeline: feature extraction, matching, and reconstruction.
        
        Args:
            project_dir: Project directory
            image_dir: Directory containing images
            quality: Processing quality
            
        Returns:
            List of camera pose dictionaries
        """
        try:
            # Create workspace
            self.create_workspace(project_dir, image_dir)
            
            # Run pipeline steps
            if not self.run_feature_extraction(quality):
                return []
            
            if not self.run_feature_matching():
                return []
            
            if not self.run_sparse_reconstruction():
                return []
            
            # Extract camera poses
            poses = self.extract_camera_poses()
            
            # Save poses to poses.npy
            if poses:
                poses_array = np.array([p["camera_to_world"] for p in poses])
                np.save(Path(project_dir) / "poses.npy", poses_array)
                logger.info(f"Saved {len(poses)} poses to {Path(project_dir) / "poses.npy"}")

            return poses
            
        except Exception as e:
            logger.error(f"COLMAP pipeline error: {e}")
            return []

class ManualPoseProcessor:
    """
    Manual pose specification and validation utilities.
    """
    
    @staticmethod
    def validate_camera_pose(pose: Dict[str, Any]) -> bool:
        """
        Validate camera pose data.
        
        Args:
            pose: Camera pose dictionary
            
        Returns:
            True if valid
        """
        try:
            # Check required fields
            if "filename" not in pose or "camera_to_world" not in pose:
                return False
            
            # Validate filename
            if not isinstance(pose["filename"], str) or not pose["filename"].strip():
                return False
            
            # Validate camera matrix
            matrix = np.array(pose["camera_to_world"])
            if matrix.shape != (4, 4):
                return False
            
            # Check if matrix is valid (not all zeros, reasonable scale)
            if np.allclose(matrix, 0):
                return False
            
            # Check scale (camera should be within reasonable bounds)
            translation = matrix[:3, 3]
            if np.any(np.abs(translation) > 1000):
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def create_camera_matrix(position: List[float], 
                           look_at: List[float], 
                           up: List[float] = [0, 1, 0]) -> np.ndarray:
        """
        Create camera-to-world transformation matrix from position and look-at point.
        
        Args:
            position: Camera position [x, y, z]
            look_at: Point to look at [x, y, z]
            up: Up vector [x, y, z]
            
        Returns:
            4x4 camera-to-world transformation matrix
        """
        position = np.array(position)
        look_at = np.array(look_at)
        up = np.array(up)
        
        # Calculate forward direction
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        # Calculate right direction
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up direction
        up = np.cross(right, forward)
        
        # Create transformation matrix
        camera_to_world = np.eye(4)
        camera_to_world[:3, 0] = right
        camera_to_world[:3, 1] = up
        camera_to_world[:3, 2] = -forward
        camera_to_world[:3, 3] = position
        
        return camera_to_world
    
    @staticmethod
    def create_circular_camera_path(center: List[float], 
                                  radius: float, 
                                  height: float, 
                                  num_poses: int) -> List[np.ndarray]:
        """
        Create a circular camera path around a center point.
        
        Args:
            center: Center point [x, y, z]
            radius: Circle radius
            height: Camera height above center
            num_poses: Number of camera poses
            
        Returns:
            List of camera-to-world transformation matrices
        """
        center = np.array(center)
        poses = []
        
        for i in range(num_poses):
            angle = 2 * np.pi * i / num_poses
            
            # Calculate camera position
            x = center[0] + radius * np.cos(angle)
            y = center[1] + height
            z = center[2] + radius * np.sin(angle)
            
            position = [x, y, z]
            look_at = center.tolist()
            
            # Create camera matrix
            camera_matrix = ManualPoseProcessor.create_camera_matrix(position, look_at)
            poses.append(camera_matrix)
        
        return poses

def estimate_camera_poses_from_images(project_dir: Path, 
                                    image_dir: Path, 
                                    quality: str = "medium") -> List[Dict[str, Any]]:
    """
    Estimate camera poses from image collection using COLMAP.
    
    Args:
        project_dir: Project directory
        image_dir: Directory containing images
        quality: Processing quality (low, medium, high)
        
    Returns:
        List of camera pose dictionaries
    """
    processor = COLMAPProcessor()
    return processor.run_full_pipeline(project_dir, image_dir, quality)

def validate_pose_file(pose_file_path: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate pose file and extract poses.
    
    Args:
        pose_file_path: Path to pose file (JSON)
        
    Returns:
        Tuple of (is_valid, poses_list)
    """
    try:
        with open(pose_file_path, 'r') as f:
            poses_data = json.load(f)
        
        if not isinstance(poses_data, list):
            return False, []
        
        # Validate each pose
        valid_poses = []
        for pose in poses_data:
            if ManualPoseProcessor.validate_camera_pose(pose):
                valid_poses.append(pose)
        
        return len(valid_poses) > 0, valid_poses
        
    except Exception as e:
        logger.error(f"Pose file validation error: {e}")
        return False, []

def extract_camera_poses(colmap_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract camera poses from COLMAP output directory.
    
    Args:
        colmap_dir: Directory containing COLMAP output files
        
    Returns:
        Tuple of (camera_matrix, poses_array)
    """
    try:
        processor = COLMAPProcessor()
        processor.workspace_dir = colmap_dir
        poses = processor.extract_camera_poses()
        
        if not poses:
            raise ValueError("No camera poses found")
        
        # Extract camera matrix from first pose
        first_pose = poses[0]
        if "camera_matrix" in first_pose:
            K = np.array(first_pose["camera_matrix"])
        else:
            # Default camera matrix
            K = np.array([[1000, 0, 400], [0, 1000, 300], [0, 0, 1]])
        
        # Convert poses to numpy array
        poses_array = []
        for pose in poses:
            if "camera_to_world" in pose:
                poses_array.append(np.array(pose["camera_to_world"]))
        
        if not poses_array:
            raise ValueError("No valid poses found")
        
        return K, np.array(poses_array)
        
    except Exception as e:
        logger.error(f"Error extracting camera poses: {e}")
        raise


if __name__ == "__main__":
    print("COLMAP utilities module loaded successfully")