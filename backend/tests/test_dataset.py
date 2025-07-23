import os
import numpy as np
import torch
import tempfile
from backend.app.ml.nerf.dataset import NeRFDataset, get_ray_batches
from PIL import Image

def create_dummy_data(tmpdir, num_images=3, img_wh=(32, 32)):
    image_dir = os.path.join(tmpdir, "images")
    os.makedirs(image_dir, exist_ok=True)
    pose_file = os.path.join(tmpdir, "poses.npy")
    poses = []
    for i in range(num_images):
        img = Image.fromarray(np.random.randint(0, 255, (img_wh[1], img_wh[0], 3), dtype=np.uint8))
        img.save(os.path.join(image_dir, f"img_{i}.png"))
        pose = np.eye(4)
        pose[:3, 3] = np.random.rand(3)
        poses.append(pose)
    np.save(pose_file, np.stack(poses))
    return image_dir, pose_file

def test_nerf_dataset_and_ray_batches():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir, pose_file = create_dummy_data(tmpdir)
        dataset = NeRFDataset(image_dir, pose_file, img_wh=(32, 32))
        assert len(dataset) == 3
        sample = dataset[0]
        assert sample['image'].shape[1:] == (32, 32)
        assert sample['pose'].shape == (4, 4)
        intrinsics = torch.eye(3)
        batches = list(get_ray_batches(dataset, intrinsics, batch_size=64, device='cpu'))
        assert all('rgb' in b for b in batches)
        assert all(b['rgb'].shape[1] == 3 for b in batches) 