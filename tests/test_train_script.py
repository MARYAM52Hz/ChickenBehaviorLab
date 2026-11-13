from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_train_script_runs_with_group_split_and_current_trainer(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    torch.set_num_threads(1)

    feature_root = tmp_path / "features"
    feature_root.mkdir()
    edge_index = np.asarray(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 3, 2, 5, 4],
        ],
        dtype=np.int64,
    )
    for label_index in range(2):
        node_features = (
            np.arange(42, dtype=np.float32).reshape(6, 7)
            + label_index
        )
        edge_attr = np.full((6, 8), label_index + 1, dtype=np.float32)
        np.savez(
            feature_root / f"class_{label_index}.npz",
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    records = []
    for video_index in range(10):
        for label_index, label in enumerate(("feeding", "walking")):
            records.append(
                {
                    "sample_id": f"video_{video_index:02d}_{label}",
                    "track_id": f"bird_{label_index}",
                    "video_id": f"video_{video_index:02d}",
                    "feature_file": f"class_{label_index}.npz",
                    "label": label,
                }
            )

    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps({"samples": records}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoints" / "baseline.pt"

    script_path = Path(__file__).parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location(
        "chicken_behavior_lab_train_script",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "choose_device",
        lambda: torch.device("cpu"),
    )

    module.main(
        [
            "--annotations",
            str(manifest),
            "--features",
            str(feature_root),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--validation-fraction",
            "0.2",
            "--split-group",
            "video",
            "--checkpoint",
            str(checkpoint),
        ]
    )

    assert checkpoint.is_file()
