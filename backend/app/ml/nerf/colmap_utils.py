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
        logger.info(f"Creating workspace in project_dir: {project_dir}")
        logger.info(f"Image directory: {image_dir}")
        
        workspace_dir = project_dir / "colmap_workspace"
        logger.info(f"Workspace directory: {workspace_dir}")
        
        try:
            workspace_dir.mkdir(exist_ok=True)
            logger.info(f"Created workspace directory: {workspace_dir}")
        except Exception as e:
            logger.error(f"Failed to create workspace directory: {e}")
            raise
        
        # Create COLMAP directory structure
        sparse_dir = workspace_dir / "sparse"
        dense_dir = workspace_dir / "dense"
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        # Copy images to workspace
        images_workspace = workspace_dir / "images"
        images_workspace.mkdir(exist_ok=True)
        
        # Copy images (or create symlinks)
        logger.info(f"Copying images from {image_dir} to {images_workspace}")
        image_count = 0
        for img_file in image_dir.iterdir():
            if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                dst = images_workspace / img_file.name
                if not dst.exists():
                    # Always copy files to avoid symlink issues with COLMAP
                    shutil.copy2(img_file, dst)
                    logger.info(f"Copied file: {img_file.name}")
                    image_count += 1
        logger.info(f"Total images processed: {image_count}")
        
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
            
            # Use absolute paths for COLMAP
            database_path = os.path.abspath(os.path.join(self.workspace_dir, "database.db"))
            image_path = os.path.abspath(os.path.join(self.workspace_dir, "images"))
            
            # Ensure database directory exists
            database_dir = os.path.dirname(database_path)
            os.makedirs(database_dir, exist_ok=True)
            
            logger.info(f"Database path: {database_path}")
            logger.info(f"Image path: {image_path}")
            
            cmd = [
                self.colmap_path, "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_path,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                "--SiftExtraction.max_image_size", str(params["max_image_size"]),
                "--SiftExtraction.max_num_features", str(params["max_num_features"])
            ]
            
            logger.info(f"Running COLMAP feature extraction: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                logger.error(f"Feature extraction failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
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
            database_path = os.path.abspath(os.path.join(self.workspace_dir, "database.db"))
            cmd = [
                self.colmap_path, "exhaustive_matcher" if method == "exhaustive" else "sequential_matcher",
                "--database_path", database_path
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
            database_path = os.path.abspath(os.path.join(self.workspace_dir, "database.db"))
            image_path = os.path.abspath(os.path.join(self.workspace_dir, "images"))
            sparse_dir = os.path.abspath(os.path.join(self.workspace_dir, "sparse"))
            
            cmd = [
                self.colmap_path, "mapper",
                "--database_path", database_path,
                "--image_path", image_path,
                "--output_path", sparse_dir
            ]
            
            logger.info(f"Running COLMAP sparse reconstruction: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            if result.returncode != 0:
                logger.error(f"Sparse reconstruction failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
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
            # Find the reconstruction file - COLMAP creates sparse/0/ directory
            sparse_dir = os.path.join(self.workspace_dir, "sparse")
            logger.info(f"Looking for reconstruction files in: {sparse_dir}")
            
            if not os.path.exists(sparse_dir):
                logger.error(f"Sparse directory does not exist: {sparse_dir}")
                return []
            
            # Look for subdirectories (COLMAP creates sparse/0/, sparse/1/, etc.)
            sparse_subdirs = [d for d in Path(sparse_dir).iterdir() if d.is_dir()]
            logger.info(f"Found sparse subdirectories: {sparse_subdirs}")
            
            if not sparse_subdirs:
                logger.error("No sparse subdirectories found")
                return []
            
            # Use the first subdirectory (usually sparse/0/)
            reconstruction_dir = sparse_subdirs[0]
            logger.info(f"Using reconstruction directory: {reconstruction_dir}")
            
            # Look for reconstruction files in the subdirectory
            reconstruction_files = list(reconstruction_dir.glob("*.bin"))
            logger.info(f"Found reconstruction files: {reconstruction_files}")
            
            if not reconstruction_files:
                logger.error("No reconstruction files found")
                return []
            
            # Use the images.bin file for camera poses
            images_bin = reconstruction_dir / "images.bin"
            if not images_bin.exists():
                logger.error("images.bin not found")
                return []
            
            # Convert binary to text format
            text_file = images_bin.with_suffix(".txt")
            cmd = [
                self.colmap_path, "model_converter",
                "--input_path", str(reconstruction_dir),
                "--output_path", str(text_file.parent),
                "--output_type", "TXT"
            ]
            
            logger.info(f"Running model converter: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Model conversion failed: {result.stderr}")
                return []
            
            # Parse camera poses from the images.txt file
            images_txt = text_file.parent / "images.txt"
            if not images_txt.exists():
                logger.error(f"images.txt not found at {images_txt}")
                return []
            
            poses = self._parse_camera_poses(images_txt)
            
            return poses
            
        except Exception as e:
            logger.error(f"Camera pose extraction error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            
            logger.info(f"Parsing {len(lines)} lines from {text_file}")
            
            # Find image section
            image_section = False
            i = 0
            
            while i < len(lines):
                line = lines[i].strip()
                
                if line == "# Image list with two lines of data per image:":
                    image_section = True
                    i += 1
                    continue
                
                if image_section and line.startswith("#"):
                    i += 1
                    continue
                
                if image_section and line:
                    try:
                        # Parse image line - format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                        parts = line.split()
                        if len(parts) >= 9:
                            image_id = int(parts[0])
                            qw, qx, qy, qz = map(float, parts[1:5])
                            tx, ty, tz = map(float, parts[5:8])
                            camera_id = int(parts[8])
                            filename = parts[9] if len(parts) > 9 else f"image_{image_id}.jpg"
                            
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
                            
                            logger.info(f"Parsed pose for image {filename}")
                        else:
                            logger.warning(f"Skipping line with insufficient parts: {line}")
                            
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse line: {line.strip()}, error: {e}")
                
                i += 1
            
            logger.info(f"Successfully parsed {len(poses)} camera poses")
            return poses
            
        except Exception as e:
            logger.error(f"Camera pose parsing error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
            # Convert strings to Path objects
            project_dir_path = Path(project_dir)
            image_dir_path = Path(image_dir)
            
            logger.info(f"Starting COLMAP pipeline for project: {project_dir}")
            logger.info(f"Image directory: {image_dir}")
            logger.info(f"Quality setting: {quality}")
            
            # Create workspace
            logger.info("Creating COLMAP workspace...")
            self.create_workspace(project_dir_path, image_dir_path)
            
            # Run pipeline steps
            logger.info("Running feature extraction...")
            if not self.run_feature_extraction(quality):
                logger.error("Feature extraction failed")
                return []
            
            logger.info("Running feature matching...")
            if not self.run_feature_matching():
                logger.error("Feature matching failed")
                return []
            
            logger.info("Running sparse reconstruction...")
            if not self.run_sparse_reconstruction():
                logger.error("Sparse reconstruction failed")
                return []
            
            # Extract camera poses
            logger.info("Extracting camera poses...")
            poses = self.extract_camera_poses()
            
            if not poses:
                logger.error("No camera poses extracted")
                return []
            
            # Save poses to poses.npy
            logger.info(f"Saving {len(poses)} poses to poses.npy...")
            poses_array = np.array([p["camera_to_world"] for p in poses])
            poses_path = Path(project_dir) / "poses.npy"
            np.save(poses_path, poses_array)
            logger.info(f"Saved poses to {poses_path}")

            return poses
            
        except Exception as e:
            logger.error(f"COLMAP pipeline error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    Estimate camera poses from images using COLMAP.
    
    Args:
        project_dir: Project directory
        image_dir: Directory containing images
        quality: COLMAP processing quality (low, medium, high)
        
    Returns:
        List of camera pose dictionaries
    """
    try:
        logger.info(f"Starting pose estimation for project: {project_dir}")
        logger.info(f"Image directory: {image_dir}")
        logger.info(f"Quality setting: {quality}")
        
        # Create COLMAP processor
        processor = COLMAPProcessor()
        
        # Create workspace and copy images
        workspace_dir = processor.create_workspace(project_dir, image_dir)
        logger.info(f"Created workspace: {workspace_dir}")
        
        # Run full COLMAP pipeline
        poses = processor.run_full_pipeline(str(project_dir), str(image_dir), quality)
        
        if not poses:
            logger.error("COLMAP pipeline failed to produce poses")
            return []
        
        logger.info(f"Successfully estimated poses for {len(poses)} images")
        return poses
        
    except Exception as e:
        logger.error(f"Pose estimation failed: {e}")
        raise

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