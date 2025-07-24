import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class NeRFDataset(Dataset):
    """
    Dataset for NeRF: loads images and camera poses, provides ray batches for training.
    """
    def __init__(self, image_dir, pose_file, img_wh=(400, 400), transform=None):
        self.image_dir = image_dir
        self.pose_file = pose_file
        self.img_wh = img_wh
        self.transform = transform or transforms.Compose([
            transforms.Resize(img_wh),
            transforms.ToTensor(),
        ])
        self.images = self._load_images()
        self.poses = self._load_poses()
        assert len(self.images) == len(self.poses), "Mismatch between images and poses"

    def _load_images(self):
        files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        images = []
        for fname in files:
            img = Image.open(os.path.join(self.image_dir, fname)).convert('RGB')
            img = self.transform(img)
            images.append(img)
        return images

    def _load_poses(self):
        # Assume pose_file is a numpy file with shape (N, 4, 4)
        poses = np.load(self.pose_file)
        return [torch.from_numpy(p).float() for p in poses]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'pose': self.poses[idx],
        }

# Utility for ray batching from dataset

def get_ray_batches(dataset, intrinsics, near=2.0, far=6.0, batch_size=1024, shuffle=True, device='cpu'):
    from app.ml.nerf.rays import generate_rays
    all_rays = []
    all_rgbs = []
    for i in range(len(dataset)):
        img = dataset[i]['image'].to(device)
        pose = dataset[i]['pose'].to(device)
        H, W = img.shape[1:]
        rays = generate_rays(H, W, intrinsics, pose, near, far, device)
        # Flatten rays and RGBs
        rays_flat = {k: v.reshape(-1, v.shape[-1]) for k, v in rays.items()}
        rgbs_flat = img.permute(1, 2, 0).reshape(-1, 3)
        all_rays.append(rays_flat)
        all_rgbs.append(rgbs_flat)
    # Concatenate all images
    rays_cat = {k: torch.cat([r[k] for r in all_rays], dim=0) for k in all_rays[0]}
    rgbs_cat = torch.cat(all_rgbs, dim=0)
    # Shuffle
    if shuffle:
        idx = torch.randperm(rgbs_cat.shape[0])
        rays_cat = {k: v[idx] for k, v in rays_cat.items()}
        rgbs_cat = rgbs_cat[idx]
    # Batch
    for i in range(0, rgbs_cat.shape[0], batch_size):
        batch = {k: v[i:i+batch_size] for k, v in rays_cat.items()}
        batch['rgb'] = rgbs_cat[i:i+batch_size]
        yield batch