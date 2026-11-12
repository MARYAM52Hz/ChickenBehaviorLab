from __future__ import annotations

import tomllib
from pathlib import Path


def test_learning_extra_installs_temporal_evaluator_dependency():
    project = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )

    learning_dependencies = project["project"]["optional-dependencies"]["learning"]

    assert any(item.lower().startswith("scikit-learn") for item in learning_dependencies)
