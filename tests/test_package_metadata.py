"""Smoke tests for the installable top-level package."""

from importlib.metadata import version

import chicken_behavior_lab


def test_package_version_matches_distribution_metadata():
    assert chicken_behavior_lab.__version__ == "0.1.0"
    assert version("chicken-behavior-lab") == chicken_behavior_lab.__version__
