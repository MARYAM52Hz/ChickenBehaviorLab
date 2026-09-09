from pathlib import Path

import numpy as np
import torch

from chicken_behavior_lab.dataset import (
    DatasetFactory,
    PyGGraphDataset,
)


def create_feature_file(
    path: Path,
) -> None:

    node_features = np.random.randn(
        6,
        7,
    ).astype(
        np.float32
    )

    edge_index = np.array(
        [
            [
                0, 1, 2,
                3, 4, 5,
            ],
            [
                1, 0, 3,
                2, 5, 4,
            ],
        ],
        dtype=np.int64,
    )

    edge_attr = np.random.randn(
        6,
        8,
    ).astype(
        np.float32
    )

    np.savez(
        path,
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def test_dataset_factory(
    tmp_path,
):

    feature_root = (
        tmp_path
        / "features"
    )

    feature_root.mkdir()

    create_feature_file(
        feature_root
        / "sample_001.npz"
    )

    annotation_file = (
        tmp_path
        / "annotations.json"
    )

    annotation_file.write_text(
        """
        {
          "dataset_version": "0.1.0",
          "samples": [
            {
              "sample_id": "sample_001",
              "track_id": "track_001",
              "video_id": "video_001",
              "feature_file": "sample_001.npz",
              "label": "walking"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    factory = DatasetFactory(
        annotation_file=annotation_file,
        feature_root=feature_root,
    )

    samples = factory.build()

    assert len(
        samples
    ) == 1

    sample = samples[0]

    assert (
        sample.sample_id
        == "sample_001"
    )

    assert (
        sample.label
        == "walking"
    )

    assert (
        sample.node_features.shape
        == (6, 7)
    )

    assert (
        sample.edge_index.shape
        == (2, 6)
    )

    assert (
        sample.edge_features.shape
        == (6, 8)
    )


def test_pyg_dataset():

    from chicken_behavior_lab.dataset import (
        GraphSample,
    )

    sample_1 = GraphSample(
        sample_id="sample_001",
        node_features=torch.randn(
            6,
            7,
        ),
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        edge_features=torch.randn(
            2,
            8,
        ),
        label="walking",
        metadata={},
    )

    sample_2 = GraphSample(
        sample_id="sample_002",
        node_features=torch.randn(
            6,
            7,
        ),
        edge_index=torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        edge_features=torch.randn(
            2,
            8,
        ),
        label="feeding",
        metadata={},
    )

    dataset = PyGGraphDataset(
        [
            sample_1,
            sample_2,
        ]
    )

    assert len(
        dataset
    ) == 2

    assert (
        dataset.labels
        == [
            "feeding",
            "walking",
        ]
    )

    data = dataset[0]

    assert (
        data.x.shape
        == (6, 7)
    )

    assert (
        data.edge_attr.shape
        == (2, 8)
    )

    assert data.y.shape == (
        1,
    )
