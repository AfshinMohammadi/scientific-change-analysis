"""
Pytest Configuration
====================

Configuration and fixtures for pytest.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is given."""
    if config.getoption("--runslow"):
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
