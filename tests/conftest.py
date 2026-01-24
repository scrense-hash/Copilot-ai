"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures available to all test modules
- Pytest configuration hooks
- Test environment setup
"""

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to Python path so tests can import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure test environment variables are set early enough (during test collection),
# because the app loads config at import time.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("LOG_LEVEL", "DEBUG")
os.environ.setdefault("LOG_PATH", "/tmp/copilot_ai_test.log")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    yield

    # Cleanup if needed
    pass


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path."""
    return project_root


@pytest.fixture(autouse=True)
def reset_last_selected():
    """Reset last selected model between tests to avoid order coupling."""
    import copilot_ai_service

    copilot_ai_service._last_selected = None
    yield
    copilot_ai_service._last_selected = None
