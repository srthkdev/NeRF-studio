import psutil
import torch
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None

@dataclass
class TrainingMetrics:
    """Training performance metrics."""
    timestamp: float
    step: int
    loss: float
    psnr: float
    learning_rate: float
    training_speed: float  # steps per second
    memory_usage_gb: float
    gpu_utilization: Optional[float] = None

class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.gpu_available = torch.cuda.is_available()
        
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # GPU metrics
        gpu_utilization = None
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None
        gpu_temperature = None
        
        if self.gpu_available:
            try:
                # GPU utilization (requires nvidia-ml-py or similar)
                gpu_utilization = self._get_gpu_utilization()
                
                # GPU memory
                gpu_memory = torch.cuda.memory_stats()
                gpu_memory_used_gb = gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # GPU temperature
                gpu_temperature = self._get_gpu_temperature()
                
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_temperature=gpu_temperature
        )
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage."""
        try:
            # This would require nvidia-ml-py or similar library
            # For now, return None
            return None
        except Exception:
            return None
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature in Celsius."""
        try:
            # This would require nvidia-ml-py or similar library
            # For now, return None
            return None
        except Exception:
            return None
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_metrics_history(self, max_points: int = 100) -> list[SystemMetrics]:
        """Get recent metrics history."""
        metrics = []
        while len(metrics) < max_points:
            try:
                metric = self.metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break
        return metrics

class TrainingMonitor:
    """Training performance monitoring."""
    
    def __init__(self):
        self.metrics_history: list[TrainingMetrics] = []
        self.max_history_size = 1000
        self.start_time = None
        self.last_step_time = None
    
    def start_training(self):
        """Mark training start."""
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.metrics_history.clear()
    
    def record_step(self, step: int, loss: float, psnr: float, learning_rate: float):
        """Record training step metrics."""
        current_time = time.time()
        
        # Calculate training speed
        training_speed = 0.0
        if self.last_step_time and step > 0:
            time_diff = current_time - self.last_step_time
            if time_diff > 0:
                training_speed = 1.0 / time_diff
        
        # Get current memory usage
        memory_usage_gb = psutil.virtual_memory().used / (1024**3)
        
        # Get GPU utilization if available
        gpu_utilization = None
        if torch.cuda.is_available():
            try:
                # This would require nvidia-ml-py or similar
                pass
            except Exception:
                pass
        
        metrics = TrainingMetrics(
            timestamp=current_time,
            step=step,
            loss=loss,
            psnr=psnr,
            learning_rate=learning_rate,
            training_speed=training_speed,
            memory_usage_gb=memory_usage_gb,
            gpu_utilization=gpu_utilization
        )
        
        self.metrics_history.append(metrics)
        self.last_step_time = current_time
        
        # Keep history size manageable
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the latest training metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, max_points: int = 100) -> list[TrainingMetrics]:
        """Get recent training metrics history."""
        if max_points >= len(self.metrics_history):
            return self.metrics_history.copy()
        return self.metrics_history[-max_points:]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        total_time = latest.timestamp - self.start_time if self.start_time else 0
        
        # Calculate averages
        losses = [m.loss for m in self.metrics_history]
        psnrs = [m.psnr for m in self.metrics_history]
        speeds = [m.training_speed for m in self.metrics_history if m.training_speed > 0]
        
        return {
            "total_steps": latest.step,
            "total_time_seconds": total_time,
            "current_loss": latest.loss,
            "current_psnr": latest.psnr,
            "average_loss": sum(losses) / len(losses) if losses else 0,
            "average_psnr": sum(psnrs) / len(psnrs) if psnrs else 0,
            "average_speed": sum(speeds) / len(speeds) if speeds else 0,
            "current_speed": latest.training_speed,
            "memory_usage_gb": latest.memory_usage_gb,
            "gpu_utilization": latest.gpu_utilization
        }

# Global monitoring instances
system_monitor = SystemMonitor()
training_monitor = TrainingMonitor()

def start_system_monitoring():
    """Start global system monitoring."""
    system_monitor.start_monitoring()

def stop_system_monitoring():
    """Stop global system monitoring."""
    system_monitor.stop_monitoring()

def get_system_metrics() -> Optional[SystemMetrics]:
    """Get current system metrics."""
    return system_monitor.get_latest_metrics()

def get_training_metrics() -> Optional[TrainingMetrics]:
    """Get current training metrics."""
    return training_monitor.get_latest_metrics()

def record_training_step(step: int, loss: float, psnr: float, learning_rate: float):
    """Record a training step."""
    training_monitor.record_step(step, loss, psnr, learning_rate)

def get_training_summary() -> Dict[str, Any]:
    """Get training performance summary."""
    return training_monitor.get_training_summary() 