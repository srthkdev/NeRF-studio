from fastapi import APIRouter, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Body, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import shutil
# UUID imports removed - using string UUIDs
import os
from PIL import Image
import threading
import time
import asyncio
import json
import logging
from pathlib import Path

from app.ml.nerf.train_nerf import NeRFTrainer
from app.ml.nerf.model import HierarchicalNeRF
from app.ml.nerf.rays import generate_rays
from app.ml.nerf.volume_rendering import volume_render_rays
import base64
import io
from app.ml.nerf.mesh_extraction import extract_mesh_from_checkpoint
import zipfile
import tempfile
from app.core.validation import InputValidator, ValidationError
from datetime import datetime
from starlette.responses import FileResponse, StreamingResponse
from app.core.monitoring import get_system_metrics, get_training_metrics, get_training_summary
from app.ml.nerf.inference import FastNeRFInference, RenderConfig, create_inference_pipeline
import cv2
from app.ml.nerf.colmap_utils import estimate_camera_poses_from_images, validate_pose_file, ManualPoseProcessor
from app.ml.nerf.advanced_export import AdvancedMeshExporter, ExportConfig, ExportFormat, ExportProgressTracker
from app.core.performance_monitor import get_performance_monitor, PerformanceMonitor, TrainingMetrics, SystemMetrics

def create_simple_circular_poses(num_images, radius=3.0, height=1.0):
    """Create simple circular camera poses for training."""
    import numpy as np
    
    poses = []
    
    for i in range(num_images):
        # Create circular path around origin
        angle = 2 * np.pi * i / num_images
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height
        
        # Create camera-to-world transformation matrix
        # Camera looks at origin
        look_at = np.array([0, 0, 0])
        position = np.array([x, y, z])
        
        # Calculate camera coordinate system
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0, 1, 0]))
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create rotation matrix
        rotation = np.eye(3)
        rotation[:, 0] = right
        rotation[:, 1] = up
        rotation[:, 2] = -forward  # Negative because camera looks down -z
        
        # Create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = position
        
        poses.append({
            "filename": f"image_{i}.jpg",
            "camera_to_world": transform.tolist(),
            "image_id": i
        })
    
    return poses

from app.database import get_db
from app.schemas import ProjectCreate, ProjectUpdate, TrainingJobCreate, ProjectInDB, TrainingJobInDB
from app.services.project_service import ProjectService, JobService
from app.core.websocket_manager import manager as websocket_manager
from app.core.config import settings

router = APIRouter()

logger = logging.getLogger(__name__)

# In-memory storage for active trainers (should be managed by a proper job queue in production)
ACTIVE_TRAINERS: Dict[str, NeRFTrainer] = {}

async def get_project_service(db: AsyncSession = Depends(get_db)) -> ProjectService:
    return ProjectService(db)

async def get_job_service(db: AsyncSession = Depends(get_db)) -> JobService:
    return JobService(db)

@router.post("/projects", response_model=ProjectInDB)
async def create_project(
    project_in: ProjectCreate,
    project_service: ProjectService = Depends(get_project_service)
):
    try:
        logger.info(f"Creating project with name: {project_in.name}")
        project = await project_service.create_project(project_in)
        logger.info(f"Project created successfully: {project.id}")
        return project
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/projects", response_model=List[ProjectInDB])
async def list_projects(
    project_service: ProjectService = Depends(get_project_service)
):
    return await project_service.get_all_projects()

@router.get("/projects/{project_id}", response_model=ProjectInDB)
async def get_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    success = await project_service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}

@router.post("/projects/{project_id}/upload_images")
async def upload_images(
    project_id: str,
    files: List[UploadFile] = File(...),
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project_dir = Path(project.data.get("project_dir"))
    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    uploaded_filenames = []
    for file in files:
        file_path = images_dir / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)
        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}")

    # Update project metadata with image list
    await project_service.update_project_data(project_id, {"images": uploaded_filenames})

    return {"message": f"Uploaded {len(uploaded_filenames)} images", "files": uploaded_filenames}

@router.post("/projects/{project_id}/estimate_poses")
async def estimate_poses(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = Path(project.data.get("project_dir"))
    images_dir = project_dir / "images"
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise HTTPException(status_code=400, detail="No images found in project directory.")

    try:
        # This function needs to be adapted to return poses in a format suitable for storage
        # For now, it will save to poses.npy and project_meta.json as before
        # In a real system, this would be a background task
        logger.info(f"Starting pose estimation for project {project_id}")
        poses = estimate_camera_poses_from_images(project_dir, images_dir)
        
        # Load the generated poses and update project data
        poses_path = project_dir / "poses.npy"
        if poses_path.exists() and poses:
            # For simplicity, we'll just note that poses are available.
            # In a real app, you might load and store a simplified representation or metadata.
            await project_service.update_project_data(project_id, {"poses_estimated": True})
            logger.info(f"Pose estimation completed successfully for project {project_id}")
            return {"message": f"Camera pose estimation completed successfully. {len(poses)} poses saved to project directory."}
        else:
            logger.error(f"Pose estimation failed to produce poses.npy for project {project_id}")
            raise HTTPException(status_code=500, detail="Pose estimation failed to produce poses.npy.")
    except Exception as e:
        logger.error(f"Pose estimation failed for project {project_id}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Pose estimation failed: {str(e)}")

@router.post("/projects/{project_id}/upload_poses")
async def upload_poses(
    project_id: str,
    poses_file: UploadFile = File(...),
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = Path(project.data.get("project_dir"))
    poses_path = project_dir / "poses.json" # Assuming JSON for manual upload
    
    try:
        content = await poses_file.read()
        poses_data = json.loads(content)
        
        # Basic validation (can be expanded)
        if not isinstance(poses_data, list) or not all(isinstance(p, dict) for p in poses_data):
            raise ValueError("Invalid poses file format. Expected a list of pose dictionaries.")
            
        with open(poses_path, "wb") as buffer:
            buffer.write(content)
            
        await project_service.update_project_data(project_id, {"manual_poses_uploaded": True})
        return {"message": "Manual poses uploaded successfully."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for poses file.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload manual poses for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload manual poses: {e}")


async def _run_training_in_background(
    project_id: str,
    job_id: str,
    config: Dict[str, Any],
    project_service: ProjectService,
    job_service: JobService
):
    try:
        trainer = NeRFTrainer(
            project_id=project_id,
            job_id=job_id,
            config=config,
            project_service=project_service, # Pass service to trainer for updates
            job_service=job_service,
            websocket_manager=websocket_manager # Pass manager to trainer for real-time updates
        )
        ACTIVE_TRAINERS[job_id] = trainer
        await job_service.update_job_status(job_id, "running")
        await trainer.train()
        await job_service.update_job_status(job_id, "completed")
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        await job_service.update_job_status(job_id, "failed")
    finally:
        if job_id in ACTIVE_TRAINERS:
            del ACTIVE_TRAINERS[job_id]

@router.post("/projects/{project_id}/start_training", response_model=TrainingJobInDB)
async def start_training(
    project_id: str,
    config: Dict[str, Any] = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    project_service: ProjectService = Depends(get_project_service),
    job_service: JobService = Depends(get_job_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate project has images
    project_dir = Path(f"data/projects/{project_id}")
    image_dir = project_dir / "images"
    
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail="No images found. Please upload images first.")
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.JPG"))
    if not image_files:
        raise HTTPException(status_code=400, detail="No valid images found. Please upload images first.")
    
    logger.info(f"Found {len(image_files)} images for project {project_id}")

    # Check if poses exist, if not estimate them automatically
    pose_file = project_dir / "poses.npy"
    if not pose_file.exists():
        logger.info(f"No poses found for project {project_id}, estimating poses automatically...")
        try:
            # Estimate poses using COLMAP
            poses = estimate_camera_poses_from_images(project_dir, image_dir, quality="medium")
            if not poses:
                logger.warning("COLMAP pose estimation failed, using simple circular poses as fallback")
                # Create simple circular poses as fallback
                import numpy as np
                poses = create_simple_circular_poses(len(image_files))
                if poses:
                    poses_array = np.array([p["camera_to_world"] for p in poses])
                    np.save(pose_file, poses_array)
                    logger.info(f"Created {len(poses)} simple circular poses as fallback")
                else:
                    raise HTTPException(status_code=500, detail="Failed to create camera poses. Please check your images.")
            else:
                logger.info(f"Successfully estimated poses for {len(poses)} images")
        except Exception as e:
            logger.error(f"Pose estimation failed for project {project_id}: {e}")
            # Try fallback simple poses
            try:
                logger.info("Attempting fallback simple pose creation...")
                import numpy as np
                poses = create_simple_circular_poses(len(image_files))
                if poses:
                    poses_array = np.array([p["camera_to_world"] for p in poses])
                    np.save(pose_file, poses_array)
                    logger.info(f"Created {len(poses)} simple circular poses as fallback")
                else:
                    raise HTTPException(status_code=500, detail=f"Pose estimation failed: {str(e)}")
            except Exception as fallback_error:
                logger.error(f"Fallback pose creation also failed: {fallback_error}")
                raise HTTPException(status_code=500, detail=f"Pose estimation failed: {str(e)}")

    # Create a new training job entry in the database
    job_create = TrainingJobCreate(project_id=project_id, config=config)
    job = await job_service.create_job(job_create)

    # Run training in a background task
    background_tasks.add_task(
        _run_training_in_background,
        project_id,
        job.id,
        config,
        project_service,
        job_service
    )

    return job

@router.get("/jobs/{job_id}", response_model=TrainingJobInDB)
async def get_job_status(
    job_id: str,
    job_service: JobService = Depends(get_job_service)
):
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/projects/{project_id}/jobs", response_model=List[TrainingJobInDB])
async def get_project_jobs(
    project_id: str,
    job_service: JobService = Depends(get_job_service)
):
    return await job_service.get_jobs_for_project(project_id)

@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket_manager.connect(websocket, str(job_id))
    try:
        while True:
            # Keep connection alive, or handle client messages if needed
            await websocket.receive_text() 
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, str(job_id))
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        websocket_manager.disconnect(websocket, str(job_id))

from app.core.performance_monitor import get_performance_monitor, SystemMetrics, TrainingMetrics, collect_system_metrics
from dataclasses import asdict

@router.get("/system/metrics")
async def get_current_system_metrics():
    metrics = collect_system_metrics()
    return asdict(metrics)

@router.get("/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str, job_service: JobService = Depends(get_job_service)):
    trainer = ACTIVE_TRAINERS.get(job_id)
    if trainer:
        return trainer.get_metrics()
    else:
        job = await job_service.get_job(job_id) # Fetch from DB if not active
        if job and job.metrics:
            return job.metrics
        raise HTTPException(status_code=404, detail="Job not found or no active metrics.")

async def _load_latest_model_checkpoint(project_dir: Path) -> Tuple[HierarchicalNeRF, Dict]:
    checkpoint_dir = project_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = sorted(checkpoint_dir.glob("nerf_step_*.pth"), key=os.path.getmtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_checkpoint_path = checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu')) # Load to CPU first

    # Reconstruct model from config saved in checkpoint
    model_config = checkpoint['config']
    model = HierarchicalNeRF(
        pos_freq_bands=model_config.get('pos_freq_bands', 10),
        view_freq_bands=model_config.get('view_freq_bands', 4),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 8),
        n_coarse=model_config.get('n_coarse', 64),
        n_fine=model_config.get('n_fine', 128)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode

    return model, model_config

@router.post("/projects/{project_id}/render")
async def render_novel_view(
    project_id: str,
    request_data: Dict[str, Any] = Body(...),
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = Path(project.data.get("project_dir"))
    
    # Extract parameters from request
    pose = request_data.get("pose", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2.5, 0, 0, 0, 1])
    resolution = request_data.get("resolution", 400)
    
    try:
        model, model_config = await _load_latest_model_checkpoint(project_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Convert pose list to 4x4 tensor
        c2w = torch.tensor(pose, dtype=torch.float32).reshape(4, 4).to(device)

        # Simplified intrinsics (should ideally be loaded from project metadata or model config)
        # Assuming square images and focal length derived from training config
        H, W = resolution, resolution
        focal = max(model_config.get('img_wh', [400, 400])) * 0.5 # Use focal from training config if available
        intrinsics = torch.tensor([
            [focal, 0.0, W/2.0],
            [0.0, focal, H/2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=device)

        # Generate rays for the novel view
        rays = generate_rays(H, W, intrinsics, c2w, 
                             near=model_config.get('near', 2.0), 
                             far=model_config.get('far', 6.0), 
                             device=device)
        
        # Flatten rays for model input
        rays_o = rays['origins'].reshape(-1, 3)
        rays_d = rays['directions'].reshape(-1, 3)
        near = rays['near'].reshape(-1, 1)
        far = rays['far'].reshape(-1, 1)

        # Render the image in batches to avoid OOM for high resolutions
        chunk_size = 1024 * 8 # Adjust based on GPU memory
        rendered_pixels = []
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i:i+chunk_size]
                chunk_rays_d = rays_d[i:i+chunk_size]
                chunk_near = near[i:i+chunk_size]
                chunk_far = far[i:i+chunk_size]

                output = model(
                    chunk_rays_o, chunk_rays_d, chunk_near, chunk_far,
                    perturb=False, training=False
                )
                rendered_pixels.append(output['fine']['rgb_map'].cpu())
        
        rgb_map = torch.cat(rendered_pixels, dim=0).reshape(H, W, 3).numpy()
        rgb_map = (rgb_map * 255).astype(np.uint8) # Scale to 0-255

        # Convert to PNG and base64 encode
        _, buffer = cv2.imencode('.png', cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return {"image": f"data:image/png;base64,{encoded_image}"}

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error rendering novel view for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render novel view: {e}")

from app.ml.nerf.advanced_export import create_advanced_exporter, ExportConfig, ExportFormat, ExportProgressTracker

# In-memory storage for export progress trackers
EXPORT_PROGRESS_TRACKERS: Dict[str, ExportProgressTracker] = {}

async def _run_export_in_background(
    project_id: str,
    export_job_id: str,
    export_config_dict: Dict[str, Any],
    project_service: ProjectService,
    job_service: JobService # Not directly used here, but good for consistency
):
    try:
        project = await project_service.get_project(project_id)
        if not project:
            raise ValueError("Project not found for export.")

        project_dir = Path(project.data.get("project_dir"))
        output_dir = project_dir / "exports" / str(export_job_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the latest model checkpoint
        model, model_config = await _load_latest_model_checkpoint(project_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Create ExportConfig from dictionary
        export_format = ExportFormat(export_config_dict.get("format", "gltf"))
        export_config = ExportConfig(
            format=export_format,
            resolution=export_config_dict.get("resolution", 128),
            texture_resolution=export_config_dict.get("texture_resolution", 1024),
            include_textures=export_config_dict.get("include_textures", True),
            bake_textures=export_config_dict.get("bake_textures", True),
            optimize_mesh=export_config_dict.get("optimize_mesh", True),
            compression=export_config_dict.get("compression", True),
            quality=export_config_dict.get("quality", "high"),
            bounds=export_config_dict.get("bounds")
        )

        exporter = create_advanced_exporter(model, device=device)
        
        # Set up progress tracking
        progress_tracker = ExportProgressTracker()
        EXPORT_PROGRESS_TRACKERS[export_job_id] = progress_tracker
        exporter.set_progress_callback(progress_tracker.update)

        logger.info(f"Starting export for project {project_id}, job {export_job_id}")
        exported_files = exporter.extract_mesh_with_textures(export_config, str(output_dir))
        logger.info(f"Export complete for project {project_id}, job {export_job_id}. Files: {exported_files}")

        # Update project data with export info
        await project_service.update_project_data(
            project_id, 
            {"last_export": {"job_id": str(export_job_id), "files": exported_files}}
        )

    except Exception as e:
        logger.error(f"Export job {export_job_id} failed: {e}")
        if export_job_id in EXPORT_PROGRESS_TRACKERS:
            EXPORT_PROGRESS_TRACKERS[export_job_id].update("failed", 0.0, str(e))
    finally:
        # Clean up tracker after a delay or when client disconnects
        pass

@router.post("/projects/{project_id}/export/advanced")
async def export_model(
    project_id: str,
    export_config: Dict[str, Any] = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    project_service: ProjectService = Depends(get_project_service),
    job_service: JobService = Depends(get_job_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    import uuid
    export_job_id = str(uuid.uuid4()) # Generate a unique ID for this export job

    background_tasks.add_task(
        _run_export_in_background,
        project_id,
        export_job_id,
        export_config,
        project_service,
        job_service
    )
    
    return {"message": "Model export started in background.", "export_job_id": str(export_job_id)}

@router.get("/exports/{export_job_id}/status")
async def get_export_status(
    export_job_id: str
):
    tracker = EXPORT_PROGRESS_TRACKERS.get(export_job_id)
    if not tracker:
        raise HTTPException(status_code=404, detail="Export job not found or expired.")
    return tracker.get_status()

@router.get("/projects/{project_id}/download_export")
async def download_export(
    project_id: str,
    file_name: str,
    project_service: ProjectService = Depends(get_project_service)
):
    project = await project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = Path(project.data.get("project_dir"))
    file_path = project_dir / "exports" / file_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Exported file not found.")

    return FileResponse(path=file_path, filename=file_name, media_type="application/octet-stream")
