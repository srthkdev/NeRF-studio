from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from typing import List
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

router = APIRouter()

# In-memory pose storage for demonstration
POSES = []

PROJECTS = {}
PROJECT_ROOT = "data/projects"
os.makedirs(PROJECT_ROOT, exist_ok=True)

JOBS = {}

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
def create_project(name: str = Form(...)):
    """Create a new NeRF project"""
    project_id = str(uuid4())
    project_dir = os.path.join(PROJECT_ROOT, project_id)
    os.makedirs(project_dir, exist_ok=True)
    PROJECTS[project_id] = {"id": project_id, "name": name, "dir": project_dir, "images": []}
    return {"project_id": project_id, "name": name}

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
def upload_images(project_id: str, files: list[UploadFile] = File(...)):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    img_dir = os.path.join(PROJECTS[project_id]["dir"], "images")
    os.makedirs(img_dir, exist_ok=True)
    saved = []
    metadata = []
    for file in files:
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        dest = os.path.join(img_dir, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
        # Extract metadata
        try:
            with Image.open(dest) as img:
                info = {
                    "filename": file.filename,
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode
                }
                metadata.append(info)
        except Exception as e:
            metadata.append({"filename": file.filename, "error": str(e)})
    PROJECTS[project_id]["images"].extend(saved)
    return {"uploaded": saved, "metadata": metadata}

@router.get("/projects/{project_id}/images")
def list_project_images(project_id: str):
    if project_id not in PROJECTS:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"images": PROJECTS[project_id]["images"]}

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
def submit_training_job(project_id: str = Form(...)):
    job_id = str(uuid4())
    JOBS[job_id] = {"id": job_id, "project_id": project_id, "status": "queued"}
    # Start a background thread to simulate training
    def train_job():
        JOBS[job_id]["status"] = "running"
        for i in range(5):
            JOBS[job_id]["progress"] = int((i+1)*20)
            time.sleep(1)
        JOBS[job_id]["status"] = "completed"
    t = threading.Thread(target=train_job, daemon=True)
    t.start()
    return {"job_id": job_id, "status": "queued"}

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    JOBS[job_id]["status"] = "cancelled"
    return {"status": "cancelled"}

@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            if job_id not in JOBS:
                await websocket.send_json({"error": "Job not found"})
                break
            job = JOBS[job_id]
            await websocket.send_json({"status": job["status"], "progress": job.get("progress", 0)})
            if job["status"] in ("completed", "cancelled"):
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass