"""Ensure lightweight model imports do not require PyTorch."""

import os
import subprocess
import sys


def test_data_model_imports_do_not_eagerly_import_torch():
    code = """
import sys
from chicken_behavior_lab.models.detection import Detection
from chicken_behavior_lab.models.keypoint import Keypoint
from chicken_behavior_lab.models.skeleton import Skeleton
assert Detection is not None
assert Keypoint is not None
assert Skeleton is not None
assert "torch" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr



def test_media_manifest_import_does_not_require_torch():
    code = """
import sys
from chicken_behavior_lab.io.media_manifest_io import MediaManifestIO
from chicken_behavior_lab.media.manifest import MediaAsset, MediaManifest
assert MediaManifestIO is not None
assert MediaAsset is not None
assert MediaManifest is not None
assert "torch" not in sys.modules
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
