import tempfile
import os
from fastapi.testclient import TestClient
from backend.app.api.v1.api import router
import numpy as np
from PIL import Image
import threading
import time

client = TestClient(router)

def test_manual_pose_upload_and_get():
    pose = list(np.eye(4).flatten())
    resp = client.post("/manual_pose_upload", data={"pose": pose})
    assert resp.status_code == 200
    assert resp.json()["status"] == "Pose uploaded"
    resp2 = client.get("/poses")
    assert resp2.status_code == 200
    assert len(resp2.json()["poses"]) >= 1

def test_validate_pose():
    pose = list(np.eye(4).flatten())
    resp = client.post("/validate_pose", data={"pose": pose})
    assert resp.status_code == 200
    assert resp.json()["status"] == "Pose is valid"
    # Invalid pose
    bad_pose = [1.0]*16
    resp2 = client.post("/validate_pose", data={"pose": bad_pose})
    assert resp2.status_code == 200
    assert "error" in resp2.json()

def test_visualize_poses():
    resp = client.get("/visualize_poses")
    assert resp.status_code == 200
    assert "camera_centers" in resp.json()

def test_project_crud_and_upload():
    # Create project
    resp = client.post("/projects", data={"name": "TestProj"})
    assert resp.status_code == 200
    project_id = resp.json()["project_id"]
    # Upload images
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        Image.new('RGB', (32, 32)).save(img_path)
        with open(img_path, "rb") as f:
            files = {"files": ("test.png", f, "image/png")}
            resp2 = client.post(f"/projects/{project_id}/upload_images", files=files)
            assert resp2.status_code == 200
            assert "uploaded" in resp2.json()
    # List images
    resp3 = client.get(f"/projects/{project_id}/images")
    assert resp3.status_code == 200
    # Delete project
    resp4 = client.delete(f"/projects/{project_id}")
    assert resp4.status_code == 200

def test_job_lifecycle():
    # Create project
    resp = client.post("/projects", data={"name": "JobProj"})
    project_id = resp.json()["project_id"]
    # Submit job
    resp2 = client.post("/jobs/submit", data={"project_id": project_id})
    assert resp2.status_code == 200
    job_id = resp2.json()["job_id"]
    # Poll status
    for _ in range(7):
        resp3 = client.get(f"/jobs/{job_id}")
        assert resp3.status_code == 200
        if resp3.json()["status"] in ("completed", "cancelled"):
            break
        time.sleep(0.5)
    # Cancel job
    resp4 = client.post(f"/jobs/{job_id}/cancel")
    assert resp4.status_code == 200

def test_websocket_job_progress():
    # Create project
    resp = client.post("/projects", data={"name": "WSProj"})
    project_id = resp.json()["project_id"]
    # Submit job
    resp2 = client.post("/jobs/submit", data={"project_id": project_id})
    job_id = resp2.json()["job_id"]
    # WebSocket client
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        for _ in range(7):
            data = ws.receive_json()
            assert "status" in data
            if data["status"] in ("completed", "cancelled"):
                break 