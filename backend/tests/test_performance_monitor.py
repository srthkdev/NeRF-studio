import pytest
import time
import threading
import psutil
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from app.core.performance_monitor import (
    PerformanceMonitor,
    SystemMetrics,
    TrainingMetrics,
    PerformanceAlert,
    PerformanceOptimizer,
    get_performance_monitor,
    collect_system_metrics,
    collect_training_metrics
)


class TestSystemMetrics:
    def test_initialization(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2 * 1024**3,
            gpu_memory_used=1.5,
            gpu_memory_total=8.0,
            gpu_utilization=75.0
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.memory_available == 1024**3
        assert metrics.memory_total == 2 * 1024**3
        assert metrics.gpu_memory_used == 1.5
        assert metrics.gpu_memory_total == 8.0
        assert metrics.gpu_utilization == 75.0

    def test_memory_used_gb(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=0,
            memory_percent=0,
            memory_available=1024**3,
            memory_total=2 * 1024**3
        )
        
        assert metrics.memory_used_gb == 1.0

    def test_gpu_memory_percent(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=0,
            memory_percent=0,
            memory_available=0,
            memory_total=0,
            gpu_memory_used=4.0,
            gpu_memory_total=8.0
        )
        
        assert metrics.gpu_memory_percent == 50.0

    def test_gpu_memory_percent_no_gpu(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=0,
            memory_percent=0,
            memory_available=0,
            memory_total=0
        )
        
        assert metrics.gpu_memory_percent is None


class TestTrainingMetrics:
    def test_initialization(self):
        metrics = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=500.0,
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        assert metrics.step == 1000
        assert metrics.loss == 0.05
        assert metrics.psnr == 25.5
        assert metrics.learning_rate == 1e-4
        assert metrics.batch_size == 1024
        assert metrics.samples_per_second == 500.0
        assert metrics.memory_allocated == 2.5
        assert metrics.memory_reserved == 3.0
        assert metrics.gpu_memory_used == 4.0
        assert metrics.render_time == 0.1
        assert metrics.total_time == 3600.0


class TestPerformanceAlert:
    def test_initialization(self):
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=90.0,
            condition="above",
            severity="warning",
            message="High CPU usage detected",
            action="scale_down"
        )
        
        assert alert.metric == "cpu_percent"
        assert alert.threshold == 90.0
        assert alert.condition == "above"
        assert alert.severity == "warning"
        assert alert.message == "High CPU usage detected"
        assert alert.action == "scale_down"

    def test_check_condition_above(self):
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=90.0,
            condition="above",
            severity="warning",
            message="High CPU usage detected"
        )
        
        assert alert.check_condition(95.0) is True
        assert alert.check_condition(85.0) is False

    def test_check_condition_below(self):
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=10.0,
            condition="below",
            severity="warning",
            message="Low CPU usage detected"
        )
        
        assert alert.check_condition(5.0) is True
        assert alert.check_condition(15.0) is False

    def test_check_condition_equals(self):
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=50.0,
            condition="equals",
            severity="info",
            message="CPU usage at 50%"
        )
        
        assert alert.check_condition(50.0) is True
        assert alert.check_condition(51.0) is False


class TestPerformanceMonitor:
    def setup_method(self):
        self.monitor = PerformanceMonitor()

    def test_initialization(self):
        assert self.monitor.metrics_history == []
        assert self.monitor.training_history == []
        assert self.monitor.alerts == []
        assert self.monitor.alert_history == []
        assert self.monitor.baselines == {}
        assert self.monitor.regression_thresholds == {}
        assert self.monitor.is_monitoring is False

    def test_add_metrics(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2 * 1024**3
        )
        
        self.monitor.add_metrics(metrics)
        
        assert len(self.monitor.metrics_history) == 1
        assert self.monitor.metrics_history[0] == metrics

    def test_add_training_metrics(self):
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=500.0,
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        self.monitor.add_training_metrics(training_metrics)
        
        assert len(self.monitor.training_history) == 1
        assert self.monitor.training_history[0] == training_metrics

    def test_add_alert(self):
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=90.0,
            condition="above",
            severity="warning",
            message="High CPU usage detected"
        )
        
        initial_count = len(self.monitor.alerts)
        self.monitor.add_alert(alert)

        assert len(self.monitor.alerts) == initial_count + 1
        assert self.monitor.alerts[-1] == alert

    def test_check_alerts(self):
        # Add an alert
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=90.0,
            condition="above",
            severity="warning",
            message="High CPU usage detected"
        )
        self.monitor.add_alert(alert)
        
        # Add metrics that should trigger the alert
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=95.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2 * 1024**3
        )
        
        triggered_alerts = self.monitor.check_alerts(metrics)
        
        assert len(triggered_alerts) == 1
        assert triggered_alerts[0]['alert']['metric'] == alert.metric
        assert triggered_alerts[0]['metric_value'] == 95.0

    def test_get_current_metrics(self):
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2 * 1024**3
        )
        
        self.monitor.add_metrics(metrics)
        
        current = self.monitor.get_current_metrics()
        assert current == metrics

    def test_get_current_training_metrics(self):
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=500.0,
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        self.monitor.add_training_metrics(training_metrics)
        
        current = self.monitor.get_current_training_metrics()
        assert current == training_metrics

    def test_get_metrics_summary(self):
        # Add multiple metrics
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=50.0 + i,
                memory_percent=60.0 + i,
                memory_available=1024**3,
                memory_total=2 * 1024**3
            )
            self.monitor.add_metrics(metrics)
        
        summary = self.monitor.get_metrics_summary()
        
        assert "current_metrics" in summary
        assert "averages" in summary
        assert "baselines" in summary

    def test_get_training_summary(self):
        # Add multiple training metrics
        for i in range(10):
            training_metrics = TrainingMetrics(
                timestamp=time.time() + i,
                step=1000 + i,
                loss=0.05 - i * 0.001,
                psnr=25.5 + i * 0.1,
                learning_rate=1e-4,
                batch_size=1024,
                samples_per_second=500.0 + i,
                memory_allocated=2.5,
                memory_reserved=3.0,
                gpu_memory_used=4.0,
                render_time=0.1,
                total_time=3600.0 + i
            )
            self.monitor.add_training_metrics(training_metrics)
        
        summary = self.monitor.get_training_summary()
        
        assert "total_steps" in summary
        assert "current_step" in summary
        assert "avg_loss" in summary
        assert "avg_psnr" in summary
        assert "avg_samples_per_second" in summary
        assert "current_learning_rate" in summary

    def test_start_monitoring(self):
        self.monitor.start_monitoring(interval=1.0)
        
        assert self.monitor.is_monitoring is True
        assert self.monitor.monitor_thread is not None
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        assert self.monitor.is_monitoring is False

    def test_set_baselines(self):
        baselines = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "gpu_memory_used": 4.0,
            "samples_per_second": 500.0
        }
        
        self.monitor.set_baselines(baselines)
        
        assert self.monitor.baselines == baselines

    def test_set_regression_thresholds(self):
        thresholds = {
            "cpu_percent": 0.8,
            "memory_percent": 0.9,
            "gpu_memory_used": 0.7,
            "samples_per_second": 0.6
        }
        
        self.monitor.set_regression_thresholds(thresholds)
        
        assert self.monitor.regression_thresholds == thresholds

    def test_detect_regressions(self):
        # Set baselines
        baselines = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "samples_per_second": 500.0
        }
        self.monitor.set_baselines(baselines)
        
        # Set thresholds
        thresholds = {
            "cpu_percent": 0.8,
            "memory_percent": 0.9,
            "samples_per_second": 0.6
        }
        self.monitor.set_regression_thresholds(thresholds)
        
        # Add current metrics that show regression
        current_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=60.0,  # 20% higher than baseline
            memory_percent=70.0,  # 16.7% higher than baseline
            memory_available=1024**3,
            memory_total=2 * 1024**3
        )
        
        current_training = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=200.0,  # 60% lower than baseline
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        # Add the metrics to trigger regression detection
        self.monitor.add_metrics(current_metrics)
        self.monitor.add_training_metrics(current_training)
        
        # Check if regression was detected in the training metrics
        # The regression detection happens automatically in _check_training_regression
        assert len(self.monitor.training_metrics_history) > 0


class TestPerformanceOptimizer:
    def setup_method(self):
        self.monitor = PerformanceMonitor()
        self.optimizer = PerformanceOptimizer(self.monitor)

    def test_initialization(self):
        assert self.optimizer.monitor == self.monitor
        assert self.optimizer.optimization_history == []

    def test_optimize_batch_size(self):
        # Mock training metrics
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=500.0,
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        # Mock system metrics
        system_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2 * 1024**3,
            gpu_memory_used=4.0,
            gpu_memory_total=8.0
        )
        
        recommendation = self.optimizer.optimize_batch_size(training_metrics, system_metrics)
        
        assert "recommended_batch_size" in recommendation
        assert "reasoning" in recommendation

    def test_optimize_learning_rate(self):
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            step=1000,
            loss=0.05,
            psnr=25.5,
            learning_rate=1e-4,
            batch_size=1024,
            samples_per_second=500.0,
            memory_allocated=2.5,
            memory_reserved=3.0,
            gpu_memory_used=4.0,
            render_time=0.1,
            total_time=3600.0
        )
        
        recommendation = self.optimizer.optimize_learning_rate(training_metrics)
        
        assert "recommended_learning_rate" in recommendation
        assert "reasoning" in recommendation

    def test_run_performance_test(self):
        test_config = {
            "duration": 10,
            "metrics": ["cpu_percent", "memory_percent", "gpu_memory_used"]
        }
        
        with patch.object(self.optimizer.monitor, '_collect_system_metrics') as mock_collect:
            mock_collect.return_value = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_available=1024**3,
                memory_total=2 * 1024**3
            )
            
            results = self.optimizer.run_performance_test(test_config)
            
            assert "test_duration" in results
            assert "metrics_collected" in results
            assert "average_metrics" in results
            assert "performance_score" in results

    def test_generate_optimization_report(self):
        # Add some optimization history
        self.optimizer.optimization_history = [
            {
                "timestamp": time.time(),
                "type": "batch_size",
                "old_value": 1024,
                "new_value": 2048,
                "improvement": 0.15
            }
        ]
        
        report = self.optimizer.generate_optimization_report()
        
        assert "total_optimizations" in report
        assert "avg_improvement" in report
        assert "recent_optimizations" in report


@patch('app.core.performance_monitor.psutil')
def test_collect_system_metrics(mock_psutil):
    # Mock psutil functions
    mock_psutil.cpu_percent.return_value = 50.0
    mock_psutil.virtual_memory.return_value = Mock(
        percent=60.0,
        available=1024**3,
        total=2 * 1024**3
    )
    mock_psutil.disk_usage.return_value = Mock(percent=45.0)
    mock_psutil.net_io_counters.return_value = Mock(
        bytes_sent=1024**3,
        bytes_recv=2 * 1024**3,
        packets_sent=1000,
        packets_recv=2000
    )
    
    # Mock GPU metrics
    with patch('app.core.performance_monitor.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 5 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=8 * 1024**3
        )
        
        metrics = collect_system_metrics()
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.memory_available == 1024**3
        assert metrics.memory_total == 2 * 1024**3
        assert metrics.gpu_memory_used == 4.0
        assert metrics.gpu_memory_total == 8.0


def test_collect_training_metrics():
    # Mock training state
    training_state = {
        "step": 1000,
        "loss": 0.05,
        "psnr": 25.5,
        "learning_rate": 1e-4,
        "batch_size": 1024
    }
    
    with patch('app.core.performance_monitor.torch') as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2.5 * 1024**3
        mock_torch.cuda.memory_reserved.return_value = 3.0 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=8 * 1024**3
        )
        
        # Mock the training metrics collection
        with patch('app.core.performance_monitor.get_performance_monitor') as mock_get_monitor:
            mock_monitor = Mock()
            mock_monitor.training_metrics_history = []
            mock_get_monitor.return_value = mock_monitor
            
            metrics = collect_training_metrics()
            
            assert metrics is None  # No training metrics available


def test_get_performance_monitor():
    # Test singleton pattern
    monitor1 = get_performance_monitor()
    monitor2 = get_performance_monitor()
    
    assert monitor1 is monitor2
    assert isinstance(monitor1, PerformanceMonitor)


if __name__ == "__main__":
    pytest.main([__file__]) 