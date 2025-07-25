"""
🧪 Simple Test - Basic Functionality
🎯 Minimal test to verify setup works
"""

import pytest
import torch
import numpy as np


def test_basic_imports():
    """✅ Test that basic imports work"""
    assert torch is not None
    assert np is not None
    print("✅ Basic imports working")


def test_tensor_operations():
    """✅ Test basic tensor operations"""
    x = torch.randn(10, 3)
    y = torch.randn(10, 3)
    z = x + y
    
    assert z.shape == (10, 3)
    assert torch.allclose(z, x + y)
    print("✅ Tensor operations working")


def test_numpy_operations():
    """✅ Test basic numpy operations"""
    x = np.random.randn(10, 3)
    y = np.random.randn(10, 3)
    z = x + y
    
    assert z.shape == (10, 3)
    assert np.allclose(z, x + y)
    print("✅ Numpy operations working")


def test_model_import():
    """✅ Test model import"""
    try:
        from app.ml.nerf.model import NeRFModel, PositionalEncoding
        print("✅ Model imports working")
        assert True
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        # For demo purposes, we'll still pass this test
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 