from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import numpy as np
import torch
import shutil
from uuid import uuid4
import os
from PIL import Image
import threading
import time
import asyncio
from fastapi import HTTPException
import json
import logging
from backend.app.ml.nerf.train_nerf import NeRFTrainer
from backend.app.ml.nerf.model import HierarchicalNeRF
from backend.app.ml.nerf.rays import generate_rays
from backend.app.ml.nerf.volume_rendering import volume_render_rays
import base64
import io
from backend.app.ml.nerf.mesh_extraction import extract_mesh_from_checkpoint
import zipfile
import tempfile
import shutil
from backend.app.core.validation import InputValidator, ValidationError, validate_api_input
from datetime import datetime
from fastapi import Body
from starlette.responses import FileResponse, BackgroundTask
from backend.app.core.monitoring import get_system_metrics, get_training_metrics, get_training_summary, record_training_step

router = APIRouter()

# In-memory pose storage for demonstration
POSES = []

PROJECTS = {}
PROJECT_ROOT = "data/projects"
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Global job storage with trainer instances
JOBS = {}
TRAINERS = {}

@router.post("/manual_pose_upload")
def manual_pose_upload(pose: List[float] = Form(...)):
    """Upload a single camera pose (16 floats for 4x4 matrix)"""
    if len(pose) != 16:
        return {"error": "Pose must have 16 values (4x4 matrix)"}
    pose_matrix = np.array(pose, dtype=np.float32).reshape(4, 4)
    POSES.append(pose_matrix)
    return {"status": "Pose uploaded", "pose": pose_matrix.tolist()}

@router.get("/poses")
def get_poses():
    """Get all uploaded poses"""
    return {"poses": [p.tolist() for p in POSES]}

@router.post("/validate_pose")
def validate_pose(pose: List[float] = Form(...)):
    """Validate a camera pose (check if it's a valid SE(3) matrix)"""
    if len(pose) != 16:
        return {"error": "Pose must have 16 values (4x4 matrix)"}
    pose_matrix = np.array(pose, dtype=np.float32).reshape(4, 4)
    # Check last row is [0,0,0,1]
    if not np.allclose(pose_matrix[3], [0,0,0,1]):
        return {"error": "Last row must be [0,0,0,1]"}
    # Check rotation part is orthogonal
    R = pose_matrix[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        return {"error": "Rotation part is not orthogonal"}
    return {"status": "Pose is valid"}

@router.get("/visualize_poses")
def visualize_poses():
    """Return a simple 3D scatter of camera centers for visualization (as JSON)"""
    centers = [p[:3, 3].tolist() for p in POSES]
    return {"camera_centers": centers}

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "nerf-studio-backend"}

@router.post("/projects")
async def create_project(name: str = Form(...)):
    """Create a new project."""
    try:
        # Validate project name
        validated_name = InputValidator.validate_project_name(name)
        
        project_id = str(uuid4())
        project_dir = os.path.join(PROJECT_ROOT, project_id)
        os.makedirs(project_dir, exist_ok=True)

        PROJECTS[project_id] = {"id": project_id, "name": validated_name, "dir": project_dir, "images": [], "config": {}, "poses": {}, "mesh_files": {}}
        return {"project_id": project_id, "name": validated_name}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Project creation failed: {e}")
        raise HTTPException(status_code=500, detail="Project creation failed")

@router.get("/projects")
def list_projects():
    return {"projects": list(PROJECTS.values())}

@router.get("/projects/{project_id}")
def get_project(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    return PROJECTS[project_id]

@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(PROJECTS[project_id]["dir"], ignore_errors=True)
    del PROJECTS[project_id]
    return {"status": "deleted"}

@router.post("/projects/{project_id}/upload_images")
async def upload_images(
    project_id: str,
    files: List[UploadFile] = File(...)
):
    """Upload images for a project."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        uploaded_files = []
        
        for file in files:
            try:
                # Validate file
                if not file.filename:
                    continue
                
                # Sanitize filename
                safe_filename = InputValidator.sanitize_filename(file.filename)
                
                # Save file
                file_path = os.path.join(project["dir"], "images", safe_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Validate image after saving
                try:
                    image_meta = InputValidator.validate_image_file(file_path)
                    uploaded_files.append({
                        "filename": safe_filename,
                        "path": file_path,
                        "metadata": image_meta
                    })
                except ValidationError as e:
                    # Remove invalid file
                    os.remove(file_path)
                    raise e
                    
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=f"Invalid image {file.filename}: {str(e)}")
            except Exception as e:
                logging.error(f"Image upload failed for {file.filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Upload failed for {file.filename}")
        
        # Update project
        project["images"].extend(uploaded_files)
        save_project_meta(project_id, {"config": project["config"], "poses": project["poses"], "images": project["images"]})
        
        return {"message": f"Uploaded {len(uploaded_files)} images", "files": uploaded_files}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Image upload failed: {e}")
        raise HTTPException(status_code=500, detail="Image upload failed")

@router.post("/projects/{project_id}/upload-poses")
async def upload_poses(
    project_id: str,
    poses_file: UploadFile = File(...)
):
    """Upload camera poses for a project."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate file size
        content = await poses_file.read()
        if len(content) > InputValidator.MAX_POSE_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Pose file too large")
        
        # Parse JSON
        try:
            poses_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Validate poses
        try:
            validated_poses = InputValidator.validate_camera_poses(poses_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Save poses
        poses_path = os.path.join(project["dir"], "poses.json")
        with open(poses_path, "w") as f:
            json.dump(validated_poses, f, indent=2)
        
        # Update project
        project["poses"] = validated_poses
        save_project_meta(project_id, {"config": project["config"], "poses": project["poses"], "images": project["images"]})
        
        return {"message": f"Uploaded {len(validated_poses)} camera poses"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Pose upload failed: {e}")
        raise HTTPException(status_code=500, detail="Pose upload failed")

@router.post("/projects/{project_id}/start-training")
async def start_training(
    project_id: str,
    config: Dict[str, Any] = Body(...)
):
    """Start training for a project."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate training config
        try:
            validated_config = InputValidator.validate_training_config(config)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if project has required data
        if not project.get("images"):
            raise HTTPException(status_code=400, detail="No images uploaded")
        if not project.get("poses"):
            raise HTTPException(status_code=400, detail="No camera poses uploaded")
        
        # Start training job
        job_id = str(uuid4())
        job = {
            "id": job_id,
            "project_id": project_id,
            "status": "running",
            "config": validated_config,
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "metrics": {}
        }
        
        JOBS[job_id] = job # Changed from TRAINING_JOBS to JOBS
        project["current_job"] = job_id
        
        # Start training in background
        asyncio.create_task(run_training_job(job_id, project, validated_config))
        
        return {"job_id": job_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Training start failed: {e}")
        raise HTTPException(status_code=500, detail="Training start failed")

# Utility to load/save project metadata (config, poses)
def get_project_meta_path(project_id):
    return os.path.join(PROJECTS[project_id]["dir"], "project_meta.json")

def load_project_meta(project_id):
    path = get_project_meta_path(project_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"config": {}, "poses": {}}

def save_project_meta(project_id, meta):
    path = get_project_meta_path(project_id)
    with open(path, "w") as f:
        json.dump(meta, f)

# --- Project Config Endpoints ---
@router.get("/projects/{project_id}/config")
def get_project_config(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    return meta.get("config", {})

@router.post("/projects/{project_id}/config")
def set_project_config(project_id: str, config: dict):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    meta["config"] = config
    save_project_meta(project_id, meta)
    return {"status": "ok"}

# --- Camera Pose Endpoints ---
@router.post("/projects/{project_id}/poses")
def upload_project_poses(project_id: str, poses: dict):
    """Upload camera poses for images. poses: {filename: pose_matrix (16 floats)}"""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    meta.setdefault("poses", {})
    for fname, pose in poses.items():
        if not isinstance(pose, list) or len(pose) != 16:
            continue
        meta["poses"][fname] = pose
    save_project_meta(project_id, meta)
    return {"status": "ok", "poses": meta["poses"]}

@router.get("/projects/{project_id}/poses")
def get_project_poses(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    return meta.get("poses", {})

# --- Update image listing to include pose metadata ---
@router.get("/projects/{project_id}/images")
def list_project_images_with_poses(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    poses = meta.get("poses", {})
    images = PROJECTS[project_id]["images"]
    return {"images": [{"filename": img, "pose": poses.get(img)} for img in images]}

# --- Pose visualization endpoint ---
@router.get("/projects/{project_id}/visualize_poses")
def visualize_project_poses(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    meta = load_project_meta(project_id)
    poses = meta.get("poses", {})
    centers = []
    for pose in poses.values():
        if isinstance(pose, list) and len(pose) == 16:
            # Camera center is translation part (last column)
            centers.append([pose[12], pose[13], pose[14]])
    return {"camera_centers": centers}

@router.post("/convert_pose")
def convert_pose(pose: List[float] = Form(...), convention: str = Form("colmap_to_nerf")):
    """Convert pose between coordinate conventions (stub)"""
    # This is a stub; real implementation would handle axis flips, etc.
    return {"converted_pose": pose, "convention": convention}

@router.post("/adjust_pose")
def adjust_pose(pose: List[float] = Form(...), dx: float = Form(0.0), dy: float = Form(0.0), dz: float = Form(0.0)):
    """Interactively adjust pose translation (stub)"""
    pose_matrix = np.array(pose, dtype=np.float32).reshape(4, 4)
    pose_matrix[:3, 3] += np.array([dx, dy, dz], dtype=np.float32)
    return {"adjusted_pose": pose_matrix.flatten().tolist()}

@router.post("/jobs/submit")
def submit_training_job(project_id: str = Form(...), config: str = Form("{}")):
    """Submit a NeRF training job with configuration"""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    job_id = str(uuid4())
    
    # Parse config
    try:
        training_config = json.loads(config)
    except json.JSONDecodeError:
        training_config = {}
    
    # Create job entry
    JOBS[job_id] = {
        "id": job_id, 
        "project_id": project_id, 
        "status": "queued",
        "config": training_config,
        "created_at": time.time()
    }
    
    # Start training in background thread
    def run_training():
        try:
            trainer = NeRFTrainer(project_id, job_id, training_config)
            TRAINERS[job_id] = trainer
            
            # Start training
            trainer.train()
            
        except Exception as e:
            logging.error(f"Training job {job_id} failed: {e}")
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
    
    # Start training thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    return {"job_id": job_id, "status": "queued", "config": training_config}

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Get detailed job status and metrics"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = JOBS[job_id]
    
    # Get real-time metrics if trainer exists
    if job_id in TRAINERS:
        trainer = TRAINERS[job_id]
        metrics = trainer.get_metrics()
        
        # Update job status from trainer
        job["status"] = metrics["status"]
        job["progress"] = metrics["progress"]
        
        return {
            **job,
            "metrics": metrics,
            "eta_seconds": metrics["eta_seconds"]
        }
    
    return job

@router.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str):
    """Get detailed training metrics for a job"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id not in TRAINERS:
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    trainer = TRAINERS[job_id]
    return trainer.get_metrics()

@router.get("/jobs/{job_id}/metrics_history")
def get_job_metrics_history(job_id: str):
    """Get full metrics history for plotting"""
    if job_id not in TRAINERS:
        raise HTTPException(status_code=404, detail="Trainer not found")
    
    trainer = TRAINERS[job_id]
    return trainer.metrics_history

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running training job"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id in TRAINERS:
        trainer = TRAINERS[job_id]
        trainer.update_job_status("cancelled", trainer.progress)
    
    JOBS[job_id]["status"] = "cancelled"
    return {"status": "cancelled"}

@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress and metrics"""
    await websocket.accept()
    try:
        while True:
            if job_id not in JOBS:
                await websocket.send_json({"error": "Job not found"})
                break
            
            job = JOBS[job_id]
            
            # Get real-time metrics if trainer exists
            if job_id in TRAINERS:
                trainer = TRAINERS[job_id]
                metrics = trainer.get_metrics()
                
                # Update job status from trainer
                job["status"] = metrics["status"]
                job["progress"] = metrics["progress"]
                
                # Send comprehensive update
                await websocket.send_json({
                    "status": metrics["status"],
                    "progress": metrics["progress"],
                    "step": metrics["step"],
                    "loss": metrics["loss"],
                    "psnr": metrics["psnr"],
                    "lr": metrics["lr"],
                    "best_psnr": metrics["best_psnr"],
                    "eta_seconds": metrics["eta_seconds"]
                })
            else:
                # Send basic job status
                await websocket.send_json({
                    "status": job["status"],
                    "progress": job.get("progress", 0)
                })
            
            # Check if job is complete
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            
            # Wait before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        pass
    finally:
        # Clean up if needed
        pass

# --- NeRF Inference and Rendering Endpoints ---

@router.post("/projects/{project_id}/render")
def render_novel_view(project_id: str, pose: List[float] = Form(...), width: int = Form(400), height: int = Form(400)):
    """Render a novel view from a given camera pose using the trained NeRF model"""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Find the latest checkpoint
    checkpoint_dir = os.path.join(PROJECTS[project_id]["dir"], "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise HTTPException(status_code=404, detail="No trained model found")
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No trained model found")
    
    # Load the latest checkpoint
    latest_checkpoint = sorted(checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    try:
        # Load model and checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        # Create model
        model = HierarchicalNeRF(
            pos_freq_bands=config.get('pos_freq_bands', 10),
            view_freq_bands=config.get('view_freq_bands', 4),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 8),
            n_coarse=config.get('n_coarse', 64),
            n_fine=config.get('n_fine', 128)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Convert pose to tensor
        pose_matrix = torch.tensor(pose, dtype=torch.float32).reshape(4, 4)
        
        # Create intrinsics
        focal = max(width, height) * 0.5
        intrinsics = torch.tensor([
            [focal, 0.0, width/2.0],
            [0.0, focal, height/2.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # Generate rays
        rays = generate_rays(height, width, intrinsics, pose_matrix, near=2.0, far=6.0)
        
        # Render
        with torch.no_grad():
            output = model(
                rays['origins'].reshape(-1, 3),
                rays['directions'].reshape(-1, 3),
                rays['near'].reshape(-1, 1),
                rays['far'].reshape(-1, 1),
                perturb=False, training=False
            )
            
            rgb = output['fine']['rgb_map'].reshape(height, width, 3)
            depth = output['fine']['depth_map'].reshape(height, width, 1)
        
        # Convert to image
        rgb_img = (rgb * 255).clamp(0, 255).to(torch.uint8)
        rgb_img = rgb_img.cpu().numpy()
        
        # Convert to base64 for frontend
        img = Image.fromarray(rgb_img)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "image": f"data:image/png;base64,{img_base64}",
            "depth": depth.cpu().numpy().tolist(),
            "width": width,
            "height": height
        }
        
    except Exception as e:
        logging.error(f"Rendering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rendering failed: {str(e)}")

@router.get("/projects/{project_id}/checkpoints")
def list_checkpoints(project_id: str):
    """List available model checkpoints for a project"""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    checkpoint_dir = os.path.join(PROJECTS[project_id]["dir"], "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return {"checkpoints": []}
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            path = os.path.join(checkpoint_dir, f)
            stat = os.stat(path)
            checkpoints.append({
                "filename": f,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "step": int(f.split('_')[-1].split('.')[0]) if '_' in f else 0
            })
    
    # Sort by step
    checkpoints.sort(key=lambda x: x["step"])
    return {"checkpoints": checkpoints}

@router.post("/projects/{project_id}/extract_mesh")
async def extract_mesh(
    project_id: str,
    bounds: List[float] = Body(default=[-2, 2, -2, 2, -2, 2]),
    resolution: int = Body(default=128),
    formats: List[str] = Body(default=["gltf", "obj", "ply"])
):
    """Extract mesh from trained NeRF model."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Validate mesh extraction parameters
        try:
            validated_bounds, validated_resolution, validated_formats = InputValidator.validate_mesh_extraction_params(
                bounds, resolution, formats
            )
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Check if model is trained
        checkpoint_path = os.path.join(project["dir"], "checkpoints", "latest.pt")
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=400, detail="No trained model found. Please train the model first.")
        
        # Create output directory
        mesh_dir = os.path.join(project["dir"], "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        
        # Extract mesh
        exported_files = extract_mesh_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=mesh_dir,
            bounds=tuple(validated_bounds),
            resolution=validated_resolution,
            formats=validated_formats
        )
        
        if not exported_files:
            raise HTTPException(status_code=500, detail="Mesh extraction failed")
        
        # Update project with mesh info
        project["mesh_files"] = exported_files
        save_project_meta(project_id, {"config": project["config"], "poses": project["poses"], "images": project["images"], "mesh_files": project["mesh_files"]})
        
        return {
            "message": "Mesh extracted successfully",
            "files": exported_files,
            "project_id": project_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Mesh extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/render_path")
def render_camera_path(project_id: str, num_frames: int = 50):
    """Render a camera path around the scene for video generation"""
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Generate a simple circular camera path
    # In a full implementation, you would:
    # 1. Load the trained model
    # 2. Generate camera poses along a path
    # 3. Render each frame
    # 4. Return the frames or save as video
    
    # For now, return a placeholder
    return {
        "status": "camera_path_rendering_not_implemented",
        "message": "Camera path rendering will be implemented in the next iteration",
        "num_frames": num_frames
    }

@router.get("/projects/{project_id}/download-mesh")
async def download_mesh(project_id: str, format: str = "gltf"):
    """Download extracted mesh file."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        mesh_files = project.get("mesh_files", {})
        if format not in mesh_files:
            raise HTTPException(status_code=404, detail=f"Mesh in {format} format not found")
        
        file_path = mesh_files[format]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Mesh file not found")
        
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=f"nerf_mesh_{project_id}.{format}"
        )
        
    except Exception as e:
        logging.error(f"Mesh download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/download-all-meshes")
async def download_all_meshes(project_id: str):
    """Download all extracted mesh formats as a ZIP file."""
    try:
        project = PROJECTS[project_id]
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        mesh_files = project.get("mesh_files", {})
        if not mesh_files:
            raise HTTPException(status_code=404, detail="No mesh files found")
        
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zipf:
                for format, file_path in mesh_files.items():
                    if os.path.exists(file_path):
                        zipf.write(file_path, f"nerf_mesh_{project_id}.{format}")
            
            # Return ZIP file
            return FileResponse(
                tmp_file.name,
                media_type="application/zip",
                filename=f"nerf_meshes_{project_id}.zip",
                background=BackgroundTask(lambda: os.unlink(tmp_file.name))
            )
        
    except Exception as e:
        logging.error(f"Mesh download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/metrics")
async def get_current_system_metrics():
    """Get current system resource metrics."""
    try:
        metrics = get_system_metrics()
        if metrics:
            return {
                "timestamp": metrics.timestamp,
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "memory_used_gb": round(metrics.memory_used_gb, 2),
                "memory_total_gb": round(metrics.memory_total_gb, 2),
                "gpu_utilization": metrics.gpu_utilization,
                "gpu_memory_used_gb": round(metrics.gpu_memory_used_gb, 2) if metrics.gpu_memory_used_gb else None,
                "gpu_memory_total_gb": round(metrics.gpu_memory_total_gb, 2) if metrics.gpu_memory_total_gb else None,
                "gpu_temperature": metrics.gpu_temperature
            }
        else:
            return {"message": "No metrics available"}
    except Exception as e:
        logging.error(f"System metrics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system metrics")

@router.get("/training/metrics")
async def get_current_training_metrics():
    """Get current training performance metrics."""
    try:
        metrics = get_training_metrics()
        if metrics:
            return {
                "timestamp": metrics.timestamp,
                "step": metrics.step,
                "loss": round(metrics.loss, 6),
                "psnr": round(metrics.psnr, 2),
                "learning_rate": round(metrics.learning_rate, 6),
                "training_speed": round(metrics.training_speed, 2),
                "memory_usage_gb": round(metrics.memory_usage_gb, 2),
                "gpu_utilization": metrics.gpu_utilization
            }
        else:
            return {"message": "No training metrics available"}
    except Exception as e:
        logging.error(f"Training metrics failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training metrics")

@router.get("/training/summary")
async def get_training_performance_summary():
    """Get training performance summary."""
    try:
        summary = get_training_summary()
        if summary:
            return {
                "total_steps": summary.get("total_steps", 0),
                "total_time_seconds": round(summary.get("total_time_seconds", 0), 2),
                "current_loss": round(summary.get("current_loss", 0), 6),
                "current_psnr": round(summary.get("current_psnr", 0), 2),
                "average_loss": round(summary.get("average_loss", 0), 6),
                "average_psnr": round(summary.get("average_psnr", 0), 2),
                "average_speed": round(summary.get("average_speed", 0), 2),
                "current_speed": round(summary.get("current_speed", 0), 2),
                "memory_usage_gb": round(summary.get("memory_usage_gb", 0), 2),
                "gpu_utilization": summary.get("gpu_utilization")
            }
        else:
            return {"message": "No training summary available"}
    except Exception as e:
        logging.error(f"Training summary failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training summary")