"""
pytest configuration: register asyncio mode.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure asyncio mode for all async tests."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
