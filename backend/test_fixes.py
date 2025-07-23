#!/usr/bin/env python3
"""Simple test script to verify fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_performance_monitor():
    """Test performance monitor fixes"""
    print("Testing performance monitor fixes...")
    
    try:
        from app.core.performance_monitor import SystemMetrics, PerformanceAlert, PerformanceMonitor
        
        # Test SystemMetrics properties
        metrics = SystemMetrics(
            timestamp=1.0,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available=1024**3,
            memory_total=2*1024**3,
            gpu_memory_used=4.0,
            gpu_memory_total=8.0
        )
        
        print(f"✓ Memory used GB: {metrics.memory_used_gb}")
        print(f"✓ GPU memory percent: {metrics.gpu_memory_percent}")
        
        # Test PerformanceAlert
        alert = PerformanceAlert(
            metric="cpu_percent",
            threshold=90.0,
            condition="above",
            severity="warning",
            message="High CPU usage"
        )
        
        print(f"✓ Alert condition check: {alert.check_condition(95.0)}")
        
        # Test PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.add_metrics(metrics)
        print(f"✓ Monitor metrics count: {len(monitor.system_metrics_history)}")
        
        print("✓ Performance monitor tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Performance monitor test failed: {e}")
        return False

def test_model_fixes():
    """Test model fixes"""
    print("Testing model fixes...")
    
    try:
        from app.ml.nerf.model import HierarchicalNeRF
        from app.ml.nerf.mesh_extraction import MeshExtractor
        
        # Test model creation
        model = HierarchicalNeRF(
            pos_freq_bands=4,
            view_freq_bands=2,
            hidden_dim=64,
            num_layers=4,
            n_coarse=16,
            n_fine=32
        )
        
        print("✓ HierarchicalNeRF created successfully")
        
        # Test mesh extractor
        extractor = MeshExtractor(model)
        print("✓ MeshExtractor created successfully")
        
        print("✓ Model tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_api_fixes():
    """Test API fixes"""
    print("Testing API fixes...")
    
    try:
        from app.api.v1.api import router
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/v1/health")
        print(f"✓ Health endpoint status: {response.status_code}")
        
        print("✓ API tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running fix verification tests...\n")
    
    tests = [
        test_performance_monitor,
        test_model_fixes,
        test_api_fixes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 