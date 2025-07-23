import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from backend.app.ml.nerf.dataset import NeRFDataset, get_ray_batches
from backend.app.ml.nerf.model import HierarchicalNeRF
from backend.app.ml.nerf.volume_rendering import compute_psnr, compute_ssim
import math

# Config
image_dir = "data/images"
pose_file = "data/poses.npy"
intrinsics = torch.tensor([
    [1000.0, 0.0, 200.0],
    [0.0, 1000.0, 200.0],
    [0.0, 0.0, 1.0]
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_wh = (400, 400)
batch_size = 2048
lr = 5e-4
num_epochs = 10
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# Data
train_dataset = NeRFDataset(image_dir, pose_file, img_wh=img_wh)

# Model
model = HierarchicalNeRF(n_coarse=64, n_fine=128, hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
writer = SummaryWriter()

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_batches = 0
    for batch in get_ray_batches(train_dataset, intrinsics, batch_size=batch_size, device=device):
        rays_o = batch['origins']
        rays_d = batch['directions']
        near = batch['near']
        far = batch['far']
        target_rgb = batch['rgb']
        optimizer.zero_grad()
        output = model(rays_o, rays_d, near, far, perturb=True, training=True)
        rgb_pred = output['fine']['rgb_map']
        loss = criterion(rgb_pred, target_rgb)
        loss.backward()
        optimizer.step()
        psnr = compute_psnr(rgb_pred, target_rgb)
        total_loss += loss.item()
        total_psnr += psnr.item()
        total_batches += 1
    avg_loss = total_loss / total_batches
    avg_psnr = total_psnr / total_batches
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('PSNR/train', avg_psnr, epoch)
    writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"nerf_epoch{epoch+1}.pth"))
    scheduler.step()

print("Training complete.") 