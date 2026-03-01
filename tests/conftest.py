"""
ROCm Bridge - Pytest Configuration
===================================
Configures pytest to properly find the rocm_bridge package.

This file is automatically discovered by pytest and runs before any tests.
"""

import sys
import os
from pathlib import Path

import pytest

# Add project root to Python path so tests can import rocm_bridge
# This is CRITICAL for package imports to work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ.setdefault("PYTHONPATH", str(project_root))
os.environ.setdefault("ROCM_BRIDGE_TEST_MODE", "true")

# Pytest hooks
def pytest_configure(config):
    """Configure pytest before running tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection (e.g., add markers)."""
    # Auto-mark slow tests
    for item in items:
        if "daemon" in item.nodeid or "integration" in item.nodeid:
            item.add_marker("slow")

# Add to tests/conftest.py

@pytest.fixture(autouse=True)
def reset_hal_singleton():
    """Reset HAL singleton between tests to prevent profile caching."""
    yield
    # Reset singleton after each test
    from rocm_bridge.core.hal import HardwareAbstractionLayer
    HardwareAbstractionLayer._instance = None
    HardwareAbstractionLayer._initialized = False