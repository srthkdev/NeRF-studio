import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
# UUID imports removed - using string UUIDs

from app.ml.nerf.dataset import NeRFDataset, get_ray_batches
from app.ml.nerf.model import HierarchicalNeRF
from app.ml.nerf.volume_rendering import compute_psnr, compute_ssim
from app.ml.nerf.rays import generate_rays
from app.services.project_service import ProjectService, JobService
from app.core.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeRFTrainer:
    """
    NeRF Trainer that integrates with the backend job system.
    Provides real-time metrics streaming and supports multiple dataset formats.
    """
    
    def __init__(
        self,
        project_id: str,
        job_id: str,
        config: Dict,
        project_service: ProjectService,
        job_service: JobService,
        websocket_manager: ConnectionManager
    ):
        self.project_id = project_id
        self.job_id = job_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.project_service = project_service
        self.job_service = job_service
        self.websocket_manager = websocket_manager
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.global_step = 0
        self.best_psnr = 0.0
        
        # Metrics tracking
        self.metrics_history = {
            'loss': [],
            'psnr': [],
            'lr': [],
            'time_per_step': []
        }
        
        # Job status
        self.job_status = "initializing"
        self.progress = 0
        self.eta_seconds = 0
        
        # Setup paths
        self.project_dir = Path(f"data/projects/{project_id}")
        self.checkpoint_dir = self.project_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Load configuration with defaults
        self._load_config()
        
    def _load_config(self):
        """Load training configuration with sensible defaults."""
        self.batch_size = self.config.get('batch_size', 4096)
        self.lr = self.config.get('lr', 5e-4)
        self.lr_decay = self.config.get('lr_decay', 0.1)
        self.lr_decay_steps = self.config.get('lr_decay_steps', 1000)
        self.num_epochs = self.config.get('num_epochs', 200000)
        self.save_every = self.config.get('save_every', 5000)
        self.log_every = self.config.get('log_every', 100)
        self.val_every = self.config.get('val_every', 5000)
        
        # Model config
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.num_layers = self.config.get('num_layers', 8)
        self.n_coarse = self.config.get('n_coarse', 64)
        self.n_fine = self.config.get('n_fine', 128)
        self.pos_freq_bands = self.config.get('pos_freq_bands', 10)
        self.view_freq_bands = self.config.get('view_freq_bands', 4)
        
        # Dataset config
        self.img_wh = tuple(self.config.get('img_wh', [400, 400]))
        self.near = self.config.get('near', 2.0)
        self.far = self.config.get('far', 6.0)
        
    def setup_dataset(self):
        """Setup dataset from project images and poses."""
        try:
            image_dir = self.project_dir / "images"
            pose_file = self.project_dir / "poses.npy"
            
            # Check if poses exist, if not create from project metadata
            if not pose_file.exists():
                # This part needs to be handled by the API before training starts
                # For now, we assume poses.npy is already created by colmap_utils
                pass
            
            if not pose_file.exists():
                raise ValueError("No camera poses found. Please upload or estimate poses first.")
            
            # Create dataset
            self.dataset = NeRFDataset(
                image_dir=str(image_dir),
                pose_file=str(pose_file),
                img_wh=self.img_wh
            )
            
            # Create intrinsics matrix (simplified - could be loaded from metadata)
            H, W = self.img_wh
            focal = max(H, W) * 0.5  # Simplified focal length
            self.intrinsics = torch.tensor([
                [focal, 0.0, W/2.0],
                [0.0, focal, H/2.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32, device=self.device)
            
            logger.info(f"Dataset loaded: {len(self.dataset)} images")
            
        except Exception as e:
            logger.error(f"Failed to setup dataset: {e}")
            raise
    
    def _create_poses_from_metadata(self):
        """This method is deprecated. Pose creation should happen before training starts."""
        pass
    
    def setup_model(self):
        """Setup NeRF model and optimizer."""
        try:
            # Create model
            self.model = HierarchicalNeRF(
                pos_freq_bands=self.pos_freq_bands,
                view_freq_bands=self.view_freq_bands,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                n_coarse=self.n_coarse,
                n_fine=self.n_fine
            ).to(self.device)
            
            # Create optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            
            # Create scheduler
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.lr_decay
            )
            
            logger.info(f"Model created with {self.model.get_params_count():,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Extract batch data
        rays_o = batch['origins'].to(self.device)
        rays_d = batch['directions'].to(self.device)
        near = batch['near'].to(self.device)
        far = batch['far'].to(self.device)
        target_rgb = batch['rgb'].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        output = self.model(
            rays_o, rays_d, near, far, 
            perturb=True, training=True
        )
        
        # Compute loss
        rgb_pred = output['fine']['rgb_map']
        loss = nn.functional.mse_loss(rgb_pred, target_rgb)
        
        # Add coarse loss if available
        if 'coarse' in output:
            rgb_coarse = output['coarse']['rgb_map']
            loss_coarse = nn.functional.mse_loss(rgb_coarse, target_rgb)
            loss = loss + 0.1 * loss_coarse
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute metrics
        psnr = compute_psnr(rgb_pred, target_rgb)
        
        return {
            'loss': loss.item(),
            'psnr': psnr.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    async def validate(self) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        
        # Use a subset of validation data
        val_indices = list(range(0, len(self.dataset), max(1, len(self.dataset) // 5)))
        
        total_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for idx in val_indices:
                data = self.dataset[idx]
                img = data['image'].unsqueeze(0).to(self.device)
                pose = data['pose'].unsqueeze(0).to(self.device)
                
                H, W = img.shape[1:]
                rays = generate_rays(H, W, self.intrinsics, pose, self.near, self.far, self.device)
                
                # Render
                output = self.model(
                    rays['origins'].reshape(-1, 3),
                    rays['directions'].reshape(-1, 3),
                    rays['near'].reshape(-1, 1),
                    rays['far'].reshape(-1, 1),
                    perturb=False, training=False
                )
                
                rgb_pred = output['fine']['rgb_map'].reshape(H, W, 3)
                target = img.squeeze(0).permute(1, 2, 0)
                
                loss = nn.functional.mse_loss(rgb_pred, target)
                psnr = compute_psnr(rgb_pred, target)
                
                total_loss += loss.item()
                total_psnr += psnr.item()
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_psnr': total_psnr / num_batches
        }
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        checkpoint_path = self.checkpoint_dir / f"nerf_step_{step:06d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only the latest checkpoint to save space
        old_checkpoints = list(self.checkpoint_dir.glob("nerf_step_*.pth"))
        if len(old_checkpoints) > 1:
            old_checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_ckpt in old_checkpoints[:-1]:
                old_ckpt.unlink()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    async def update_job_status(self, status: str, progress: int, eta_seconds: int = 0):
        """Update job status for the backend and broadcast via WebSocket."""
        self.job_status = status
        self.progress = progress
        self.eta_seconds = eta_seconds
        
        metrics_data = self.get_metrics()
        metrics_data["status"] = status
        metrics_data["progress"] = progress
        metrics_data["eta_seconds"] = eta_seconds

        # Update database
        await self.job_service.update_job_progress(self.job_id, metrics_data)

        # Broadcast to WebSocket clients
        await self.websocket_manager.broadcast(str(self.job_id), metrics_data)
    
    def get_metrics(self) -> Dict:
        """Get current training metrics."""
        return {
            'step': self.global_step,
            'loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else 0.0,
            'psnr': self.metrics_history['psnr'][-1] if self.metrics_history['psnr'] else 0.0,
            'lr': self.metrics_history['lr'][-1] if self.metrics_history['lr'] else self.lr,
            'best_psnr': self.best_psnr,
            'eta_seconds': self.eta_seconds,
            'status': self.job_status,
            'progress': self.progress
        }
    
    async def train(self):
        """Main training loop."""
        try:
            await self.update_job_status("setting_up", 0)
            
            # Setup
            self.setup_dataset()
            self.setup_model()
            
            await self.update_job_status("training", 0)
            
            # Training loop
            start_time = time.time()
            
            for step in range(self.num_epochs):
                step_start_time = time.time()
                
                # Get batch
                try:
                    batch = next(self._get_batch_iterator())
                except StopIteration:
                    # Restart iterator
                    batch = next(self._get_batch_iterator())
                
                # Training step
                metrics = self.train_step(batch)
                
                # Update metrics history
                for key, value in metrics.items():
                    self.metrics_history[key].append(value)
                
                # Update best PSNR
                if metrics['psnr'] > self.best_psnr:
                    self.best_psnr = metrics['psnr']
                
                # Update global step
                self.global_step = step
                
                # Update job status
                progress = int((step / self.num_epochs) * 100)
                elapsed_time = time.time() - start_time
                if step > 0:
                    eta_seconds = int((elapsed_time / step) * (self.num_epochs - step))
                else:
                    eta_seconds = 0
                
                await self.update_job_status("training", progress, eta_seconds)
                
                # Logging
                if step % self.log_every == 0:
                    step_time = time.time() - step_start_time
                    self.metrics_history['time_per_step'].append(step_time)
                    
                    logger.info(
                        f"Step {step}/{self.num_epochs} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"PSNR: {metrics['psnr']:.2f} | "
                        f"LR: {metrics['lr']:.6f} | "
                        f"Time: {step_time:.3f}s"
                    )
                
                # Validation
                if step % self.val_every == 0 and step > 0:
                    val_metrics = await self.validate()
                    logger.info(
                        f"Validation | Loss: {val_metrics['val_loss']:.4f} | "
                        f"PSNR: {val_metrics['val_psnr']:.2f}"
                    )
                
                # Save checkpoint
                if step % self.save_every == 0 and step > 0:
                    self.save_checkpoint(step)
                
                # Learning rate scheduling
                if step % self.lr_decay_steps == 0 and step > 0:
                    self.scheduler.step()
                
                # Check for early stopping or job cancellation
                # In a real async setup, you'd check a cancellation flag
                # if self.job_status == "cancelled":
                #     logger.info("Training cancelled by user")
                #     break
            
            # Final save
            self.save_checkpoint(self.num_epochs)
            await self.update_job_status("completed", 100)
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            await self.update_job_status("failed", 0)
            raise
    
    def _get_batch_iterator(self):
        """Get batch iterator for training."""
        try:
            return get_ray_batches(
                self.dataset, 
                self.intrinsics, 
                batch_size=self.batch_size, 
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to create batch iterator: {e}")
            raise

if __name__ == "__main__":
    # This block is for local testing and requires a running DB and services
    # It's typically not run directly in a production setup.
    print("NeRFTrainer is designed to be run as a background task via FastAPI.")
    print("Please start the FastAPI application and use the API endpoints to initiate training.")
