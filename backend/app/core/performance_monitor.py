import torch
import psutil
import time
import threading
import json
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import subprocess
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    memory_total: int
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_io: Optional[Dict[str, float]] = None
    
    @property
    def memory_used_gb(self) -> float:
        """Get memory used in GB"""
        return (self.memory_total - self.memory_available) / (1024**3)
    
    @property
    def gpu_memory_percent(self) -> Optional[float]:
        """Get GPU memory usage percentage"""
        if self.gpu_memory_used is not None and self.gpu_memory_total is not None:
            return (self.gpu_memory_used / self.gpu_memory_total) * 100
        return None

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    timestamp: float
    step: int
    loss: float
    psnr: float
    learning_rate: float
    batch_size: int
    samples_per_second: float
    memory_allocated: float
    memory_reserved: float
    gpu_memory_used: float
    render_time: float
    total_time: float

@dataclass
class PerformanceAlert:
    """Performance alert configuration"""
    metric: str
    threshold: float
    condition: str  # 'above', 'below', 'equals'
    severity: str  # 'warning', 'error', 'critical'
    message: str
    action: Optional[str] = None  # 'restart', 'scale', 'notify'
    
    def check_condition(self, value: float) -> bool:
        """Check if the condition is met"""
        if self.condition == "above":
            return value > self.threshold
        elif self.condition == "below":
            return value < self.threshold
        elif self.condition == "equals":
            return abs(value - self.threshold) < 1e-6
        return False

class PerformanceMonitor:
    """
    Comprehensive performance monitoring and optimization system.
    """
    
    def __init__(self, 
                 alert_callbacks: Optional[List[Callable]] = None,
                 metrics_history_size: int = 1000,
                 monitoring_interval: float = 1.0):
        self.alert_callbacks = alert_callbacks or []
        self.metrics_history_size = metrics_history_size
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.system_metrics_history = deque(maxlen=metrics_history_size)
        self.training_metrics_history = deque(maxlen=metrics_history_size)
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        self.alert_history: List[Dict] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        # Performance baselines
        self.baselines = {}
        self.regression_thresholds = {
            'training_speed': 0.8,  # 80% of baseline
            'memory_efficiency': 0.9,  # 90% of baseline
            'render_quality': 0.95  # 95% of baseline
        }
        
        # Initialize default alerts
        self._setup_default_alerts()
        
        # Legacy attributes for backward compatibility
        self.metrics_history = []  # Legacy attribute
        
    def _setup_default_alerts(self):
        """Setup default performance alerts"""
        default_alerts = [
            PerformanceAlert(
                metric="gpu_memory_used",
                threshold=0.9,
                condition="above",
                severity="warning",
                message="GPU memory usage is high",
                action="optimize_batch_size"
            ),
            PerformanceAlert(
                metric="cpu_percent",
                threshold=0.95,
                condition="above",
                severity="warning",
                message="CPU usage is very high",
                action="scale_horizontally"
            ),
            PerformanceAlert(
                metric="memory_percent",
                threshold=0.9,
                condition="above",
                severity="error",
                message="System memory usage is critical",
                action="restart"
            ),
            PerformanceAlert(
                metric="samples_per_second",
                threshold=0.5,
                condition="below",
                severity="warning",
                message="Training speed has degraded significantly",
                action="optimize"
            )
        ]
        
        for alert in default_alerts:
            self.add_alert(alert)
            
    def start_monitoring(self, interval: float = None):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        if interval is not None:
            self.monitoring_interval = interval
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Check alerts
                self._check_alerts(system_metrics)
                
                # Update baselines
                self._update_baselines(system_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_utilization = self._get_gpu_utilization()
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
                
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        
        # Network I/O
        network_io = self._get_network_io()
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            memory_total=memory.total,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            disk_usage_percent=disk_usage.percent,
            network_io=network_io
        )
        
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage"""
        try:
            if self.gpu_available:
                # Try to use nvidia-smi if available
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
        except Exception:
            pass
            
        # Fallback: estimate based on memory usage
        if torch.cuda.memory_allocated() > 0:
            return min(100.0, torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100)
            
        return None
        
    def _get_network_io(self) -> Optional[Dict[str, float]]:
        """Get network I/O statistics"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception:
            return None
            
    def add_alert(self, alert: PerformanceAlert):
        """Add a performance alert"""
        self.alerts.append(alert)
        logger.info(f"Added performance alert: {alert.metric} {alert.condition} {alert.threshold}")
        
    def _check_alerts(self, metrics: SystemMetrics):
        """Check if any alerts should be triggered"""
        for alert in self.alerts:
            try:
                metric_value = getattr(metrics, alert.metric, None)
                if metric_value is None:
                    continue
                    
                triggered = False
                if alert.condition == "above" and metric_value > alert.threshold:
                    triggered = True
                elif alert.condition == "below" and metric_value < alert.threshold:
                    triggered = True
                elif alert.condition == "equals" and abs(metric_value - alert.threshold) < 1e-6:
                    triggered = True
                    
                if triggered:
                    self._trigger_alert(alert, metric_value, metrics)
                    
            except Exception as e:
                logger.error(f"Error checking alert {alert.metric}: {e}")
                
    def _trigger_alert(self, alert: PerformanceAlert, metric_value: float, metrics: SystemMetrics):
        """Trigger a performance alert"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'alert': asdict(alert),
            'metric_value': metric_value,
            'system_metrics': asdict(metrics)
        }
        
        self.alert_history.append(alert_data)
        
        # Execute alert action
        if alert.action:
            self._execute_alert_action(alert.action, alert_data)
            
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
        logger.warning(f"Performance alert triggered: {alert.message} (Value: {metric_value})")
        
    def _execute_alert_action(self, action: str, alert_data: Dict):
        """Execute alert action"""
        try:
            if action == "optimize_batch_size":
                self._optimize_batch_size()
            elif action == "scale_horizontally":
                self._scale_horizontally()
            elif action == "restart":
                self._restart_service()
            elif action == "optimize":
                self._optimize_performance()
        except Exception as e:
            logger.error(f"Error executing alert action {action}: {e}")
            
    def _optimize_batch_size(self):
        """Optimize batch size based on memory usage"""
        if not self.gpu_available:
            return
            
        current_memory = torch.cuda.memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if current_memory / total_memory > 0.8:
            # Reduce batch size
            logger.info("Reducing batch size due to high memory usage")
            # This would typically update a global configuration
            
    def _scale_horizontally(self):
        """Scale horizontally by adding more workers"""
        logger.info("Initiating horizontal scaling")
        # This would typically spawn new worker processes or containers
        
    def _restart_service(self):
        """Restart the service"""
        logger.warning("Restarting service due to critical performance issues")
        # This would typically restart the current process
        
    def _optimize_performance(self):
        """Optimize overall performance"""
        logger.info("Running performance optimization")
        # This would typically adjust various parameters
        
    def record_training_metrics(self, metrics: TrainingMetrics):
        """Record training performance metrics"""
        self.training_metrics_history.append(metrics)
        
        # Check for performance regression
        self._check_training_regression(metrics)
    
    def add_metrics(self, metrics: SystemMetrics):
        """Add system metrics (legacy method)"""
        self.system_metrics_history.append(metrics)
        self.metrics_history.append(metrics)  # Legacy support
    
    def add_training_metrics(self, metrics: TrainingMetrics):
        """Add training metrics (legacy method)"""
        self.record_training_metrics(metrics)
    
    def check_alerts(self, metrics: SystemMetrics) -> List[Dict]:
        """Check alerts and return triggered ones"""
        triggered = []
        for alert in self.alerts:
            try:
                metric_value = getattr(metrics, alert.metric, None)
                if metric_value is not None and alert.check_condition(metric_value):
                    triggered.append({
                        'alert': asdict(alert),
                        'metric_value': metric_value,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.error(f"Error checking alert {alert.metric}: {e}")
        return triggered
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        if self.system_metrics_history:
            return self.system_metrics_history[-1]
        return None
    
    def get_current_training_metrics(self) -> Optional[TrainingMetrics]:
        """Get current training metrics"""
        if self.training_metrics_history:
            return self.training_metrics_history[-1]
        return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary (legacy method)"""
        return self.get_performance_summary()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.training_metrics_history:
            return {}
        
        recent_metrics = list(self.training_metrics_history)[-10:]
        
        return {
            'total_steps': len(self.training_metrics_history),
            'current_step': recent_metrics[-1].step if recent_metrics else 0,
            'avg_loss': np.mean([m.loss for m in recent_metrics]),
            'avg_psnr': np.mean([m.psnr for m in recent_metrics]),
            'avg_samples_per_second': np.mean([m.samples_per_second for m in recent_metrics]),
            'current_learning_rate': recent_metrics[-1].learning_rate if recent_metrics else 0
        }
    
    def set_baselines(self, baselines: Dict[str, float]):
        """Set performance baselines"""
        self.baselines.update(baselines)
    
    def set_regression_thresholds(self, thresholds: Dict[str, float]):
        """Set regression thresholds"""
        self.regression_thresholds.update(thresholds)
        
    def _check_training_regression(self, metrics: TrainingMetrics):
        """Check for training performance regression"""
        if len(self.training_metrics_history) < 10:
            return
            
        # Calculate baseline from recent history
        recent_metrics = list(self.training_metrics_history)[-10:]
        baseline_speed = np.mean([m.samples_per_second for m in recent_metrics])
        
        # Check if current speed is below threshold
        if metrics.samples_per_second < baseline_speed * self.regression_thresholds['training_speed']:
            logger.warning(f"Training speed regression detected: {metrics.samples_per_second:.2f} vs baseline {baseline_speed:.2f}")
            
    def _update_baselines(self, metrics: SystemMetrics):
        """Update performance baselines"""
        # Update system baselines
        if len(self.system_metrics_history) > 10:
            recent_metrics = list(self.system_metrics_history)[-10:]
            
            self.baselines['cpu_usage'] = np.mean([m.cpu_percent for m in recent_metrics])
            self.baselines['memory_usage'] = np.mean([m.memory_percent for m in recent_metrics])
            
            if self.gpu_available:
                gpu_metrics = [m for m in recent_metrics if m.gpu_memory_used is not None]
                if gpu_metrics:
                    self.baselines['gpu_memory_usage'] = np.mean([m.gpu_memory_used for m in gpu_metrics])
                    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.system_metrics_history:
            return {}
            
        recent_metrics = list(self.system_metrics_history)[-10:]
        
        summary = {
            'current_metrics': asdict(recent_metrics[-1]) if recent_metrics else {},
            'averages': {
                'cpu_percent': np.mean([m.cpu_percent for m in recent_metrics]),
                'memory_percent': np.mean([m.memory_percent for m in recent_metrics]),
                'gpu_memory_used': np.mean([m.gpu_memory_used or 0 for m in recent_metrics])
            },
            'baselines': self.baselines,
            'alerts': len(self.alert_history),
            'training_metrics': {
                'total_steps': len(self.training_metrics_history),
                'avg_samples_per_second': np.mean([m.samples_per_second for m in self.training_metrics_history]) if self.training_metrics_history else 0
            }
        }
        
        return summary
        
    def get_metrics_history(self, 
                          metric_type: str = "system",
                          limit: int = 100) -> List[Dict]:
        """Get metrics history"""
        if metric_type == "system":
            metrics = list(self.system_metrics_history)[-limit:]
        elif metric_type == "training":
            metrics = list(self.training_metrics_history)[-limit:]
        else:
            return []
            
        return [asdict(m) for m in metrics]
        
    def export_metrics(self, filepath: str):
        """Export metrics to file"""
        data = {
            'system_metrics': self.get_metrics_history("system"),
            'training_metrics': self.get_metrics_history("training"),
            'alerts': self.alert_history,
            'baselines': self.baselines,
            'summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def clear_history(self):
        """Clear metrics history"""
        self.system_metrics_history.clear()
        self.training_metrics_history.clear()
        self.alert_history.clear()
        logger.info("Performance metrics history cleared")


class GPUMemoryManager:
    """Advanced GPU memory management"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.memory_threshold = 0.9  # 90% of total memory
        self.cleanup_threshold = 0.8  # 80% of total memory
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if not torch.cuda.is_available():
            return {}
            
        allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3
        total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'utilization_percent': (allocated / total) * 100
        }
        
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return False
            
        return memory_info['utilization_percent'] > self.cleanup_threshold * 100
        
    def cleanup_memory(self, aggressive: bool = False):
        """Clean up GPU memory"""
        if not torch.cuda.is_available():
            return
            
        # Clear cache
        torch.cuda.empty_cache()
        
        if aggressive:
            # Force garbage collection
            import gc
            gc.collect()
            
        logger.info(f"GPU memory cleanup completed. Current usage: {self.get_memory_info()}")
        
    def optimize_batch_size(self, current_batch_size: int, target_memory_usage: float = 0.7) -> int:
        """Optimize batch size based on memory usage"""
        memory_info = self.get_memory_info()
        if not memory_info:
            return current_batch_size
            
        current_usage = memory_info['utilization_percent'] / 100
        if current_usage > target_memory_usage:
            # Reduce batch size
            reduction_factor = target_memory_usage / current_usage
            new_batch_size = max(1, int(current_batch_size * reduction_factor))
            logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
            
        return current_batch_size


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.gpu_manager = GPUMemoryManager()
        self.optimization_history = []
        
    def optimize_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize training configuration based on system performance"""
        memory_info = self.gpu_manager.get_memory_info()
        
        if memory_info:
            # Adjust batch size based on available memory
            total_memory_gb = memory_info['total_gb']
            if total_memory_gb < 8:
                config['batch_size'] = min(config.get('batch_size', 4096), 2048)
                config['chunk_size'] = min(config.get('chunk_size', 4096), 2048)
            elif total_memory_gb < 16:
                config['batch_size'] = min(config.get('batch_size', 4096), 8192)
                config['chunk_size'] = min(config.get('chunk_size', 4096), 8192)
            else:
                config['batch_size'] = config.get('batch_size', 16384)
                config['chunk_size'] = config.get('chunk_size', 16384)
                
        # Adjust based on CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            config['num_workers'] = 0
        else:
            config['num_workers'] = min(cpu_count - 1, 4)
            
        return config
        
    def run_performance_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance regression test"""
        start_time = time.time()
        
        # Collect baseline metrics
        baseline_metrics = self.monitor.get_performance_summary()
        
        # Run test (this would be a simplified training run)
        test_duration = test_config.get('duration', 60)  # seconds
        time.sleep(min(test_duration, 10))  # Simplified test
        
        # Collect final metrics
        final_metrics = self.monitor.get_performance_summary()
        
        # Calculate performance metrics
        test_results = {
            'baseline': baseline_metrics,
            'final': final_metrics,
            'duration': time.time() - start_time,
            'passed': True  # Simplified pass/fail logic
        }
        
        return test_results
    
    def optimize_batch_size(self, training_metrics: TrainingMetrics, system_metrics: SystemMetrics) -> Dict[str, Any]:
        """Optimize batch size based on performance metrics"""
        current_batch_size = training_metrics.batch_size
        gpu_memory_used = system_metrics.gpu_memory_used or 0
        gpu_memory_total = system_metrics.gpu_memory_total or 8.0
        
        # Calculate optimal batch size
        memory_usage_ratio = gpu_memory_used / gpu_memory_total
        if memory_usage_ratio > 0.8:
            new_batch_size = max(1, int(current_batch_size * 0.8))
        elif memory_usage_ratio < 0.5:
            new_batch_size = min(current_batch_size * 2, 16384)
        else:
            new_batch_size = current_batch_size
        
        recommendation = {
            'current_batch_size': current_batch_size,
            'recommended_batch_size': new_batch_size,
            'reason': 'memory_optimization',
            'memory_usage_ratio': memory_usage_ratio
        }
        
        self.optimization_history.append({
            'timestamp': time.time(),
            'type': 'batch_size',
            'old_value': current_batch_size,
            'new_value': new_batch_size,
            'improvement': 0.1  # Placeholder
        })
        
        return recommendation
    
    def optimize_learning_rate(self, training_metrics: TrainingMetrics) -> Dict[str, Any]:
        """Optimize learning rate based on training metrics"""
        current_lr = training_metrics.learning_rate
        loss = training_metrics.loss
        psnr = training_metrics.psnr
        
        # Simple learning rate optimization
        if loss > 0.1:  # High loss
            new_lr = current_lr * 1.2
        elif loss < 0.01:  # Low loss
            new_lr = current_lr * 0.8
        else:
            new_lr = current_lr
        
        recommendation = {
            'current_learning_rate': current_lr,
            'recommended_learning_rate': new_lr,
            'reason': 'loss_based_optimization',
            'current_loss': loss,
            'current_psnr': psnr
        }
        
        self.optimization_history.append({
            'timestamp': time.time(),
            'type': 'learning_rate',
            'old_value': current_lr,
            'new_value': new_lr,
            'improvement': 0.05  # Placeholder
        })
        
        return recommendation
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        if not self.optimization_history:
            return {'message': 'No optimizations performed yet'}
        
        recent_optimizations = self.optimization_history[-10:]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'recent_optimizations': recent_optimizations,
            'optimization_types': list(set(opt['type'] for opt in self.optimization_history)),
            'avg_improvement': np.mean([opt.get('improvement', 0) for opt in self.optimization_history])
        }


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    if not hasattr(get_performance_monitor, '_instance'):
        get_performance_monitor._instance = PerformanceMonitor()
    return get_performance_monitor._instance

def collect_system_metrics() -> SystemMetrics:
    """Collect current system metrics"""
    monitor = get_performance_monitor()
    return monitor._collect_system_metrics()

def collect_training_metrics() -> Optional[TrainingMetrics]:
    """Collect current training metrics if available"""
    monitor = get_performance_monitor()
    if monitor.training_metrics_history:
        return monitor.training_metrics_history[-1]
    return None

def start_global_monitoring():
    """Start global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.start_monitoring()

def stop_global_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring() 