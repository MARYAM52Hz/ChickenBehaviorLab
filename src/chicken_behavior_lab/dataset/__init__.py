from chicken_behavior_lab.dataset.sample import (
    GraphSample,
)

from chicken_behavior_lab.dataset.graph_dataset import (
    GraphDataset,
)

from chicken_behavior_lab.dataset.pyg_dataset import (
    PyGGraphDataset,
)

from chicken_behavior_lab.dataset.temporal_sample import (
    TemporalGraphSample,
)

from chicken_behavior_lab.dataset.temporal_dataset import (
    TemporalGraphDataset,
)

from chicken_behavior_lab.dataset.temporal_builder import (
    TemporalSequenceBuilder,
    TemporalWindow,
)

from chicken_behavior_lab.dataset.temporal_pyg_dataset import (
    TemporalPyGDataset,
)

from chicken_behavior_lab.dataset.temporal_batch import (
    TemporalBatch,
)

from chicken_behavior_lab.dataset.temporal_collate import (
    TemporalCollator,
    make_temporal_dataloader,
)

from chicken_behavior_lab.dataset.group_split import (
    SplitGroups,
    normalize_group_id,
    collect_groups,
    build_group_id,
    collect_group_ids,
    validate_group_split,
    split_group_ids,
)

from chicken_behavior_lab.dataset.group_subset import (
    GroupSubset,
)

from chicken_behavior_lab.dataset.group_splitter import (
    DatasetSplits,
    split_dataset_by_group,
)

from chicken_behavior_lab.dataset.temporal_split_builder import (
    TemporalDatasetSplits,
    build_temporal_splits,
)


__all__ = [
    # ---------------------------------------------------------
    # Basic graph dataset
    # ---------------------------------------------------------

    "GraphSample",
    "GraphDataset",
    "PyGGraphDataset",

    # ---------------------------------------------------------
    # Temporal graph representation
    # ---------------------------------------------------------

    "TemporalGraphSample",
    "TemporalGraphDataset",
    "TemporalSequenceBuilder",
    "TemporalWindow",
    "TemporalPyGDataset",

    # ---------------------------------------------------------
    # Temporal batching
    # ---------------------------------------------------------

    "TemporalBatch",
    "TemporalCollator",
    "make_temporal_dataloader",

    # ---------------------------------------------------------
    # Group-aware splitting
    # ---------------------------------------------------------

    "SplitGroups",
    "normalize_group_id",
    "collect_groups",
    "build_group_id",
    "collect_group_ids",
    "validate_group_split",
    "split_group_ids",

    "GroupSubset",

    "DatasetSplits",
    "split_dataset_by_group",

    # ---------------------------------------------------------
    # Temporal split construction
    # ---------------------------------------------------------

    "TemporalDatasetSplits",
    "build_temporal_splits",
]
