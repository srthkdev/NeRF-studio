import os
import subprocess
import numpy as np
import re

def run_colmap_feature_extraction(image_dir, database_path, colmap_bin="colmap"):
    cmd = [colmap_bin, "feature_extractor",
           "--database_path", database_path,
           "--image_path", image_dir]
    subprocess.run(cmd, check=True)

def run_colmap_exhaustive_matcher(database_path, colmap_bin="colmap"):
    cmd = [colmap_bin, "exhaustive_matcher",
           "--database_path", database_path]
    subprocess.run(cmd, check=True)

def run_colmap_mapper(image_dir, database_path, output_dir, colmap_bin="colmap"):
    cmd = [colmap_bin, "mapper",
           "--database_path", database_path,
           "--image_path", image_dir,
           "--output_path", output_dir]
    subprocess.run(cmd, check=True)

def run_colmap_bundle_adjuster(model_dir, colmap_bin="colmap"):
    cmd = [colmap_bin, "bundle_adjuster",
           "--input_path", model_dir,
           "--output_path", model_dir]
    subprocess.run(cmd, check=True)

def parse_colmap_cameras_txt(cameras_txt):
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            # Format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Assume PINHOLE or SIMPLE_PINHOLE
            fx = float(parts[4])
            fy = float(parts[5]) if len(parts) > 5 else fx
            cx = float(parts[6])
            cy = float(parts[7])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            return K  # Only one camera assumed
    return np.eye(3)

def parse_colmap_images_txt(images_txt):
    poses = []
    with open(images_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            # Convert quaternion to rotation matrix
            q = np.array([qw, qx, qy, qz], dtype=np.float32)
            R = quat2mat(q)
            t = np.array([tx, ty, tz], dtype=np.float32)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses.append(pose)
    return np.stack(poses) if poses else np.zeros((0, 4, 4))

def quat2mat(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ], dtype=np.float32)

def extract_camera_poses(model_dir):
    cameras_txt = os.path.join(model_dir, "cameras.txt")
    images_txt = os.path.join(model_dir, "images.txt")
    K = parse_colmap_cameras_txt(cameras_txt)
    poses = parse_colmap_images_txt(images_txt)
    return K, poses 