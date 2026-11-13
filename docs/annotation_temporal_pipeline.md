# Annotation-Level Temporal Graph Pipeline

## Sample contract

The main behavior-learning pipeline uses **one CVAT behavior annotation as one
independent supervised sample**. Adjacent annotations remain separate even
when they have the same behavior label.

Each sample is a `GraphSample` whose graph is a `TemporalSkeletonGraph`:

- node features: `(T, V, F)`
- shared spatial topology: `(2, E)`
- edge features: `(T, E, D)`, when available
- frame validity: `(T,)`
- node validity: `(T, V)`
- edge validity: `(T, E)`, derived from the validity of both endpoints

Annotation frame intervals are inclusive, so the graph must contain exactly
`end_frame - start_frame + 1` temporal steps. The default
`TemporalFeatureBuilder()` preserves all supplied frames. Explicit
`sequence_length` remains available for legacy fixed-length feature workflows,
but a truncated or padded sequence must not be passed off as a complete
annotation interval.

## Batching and model behavior

`TemporalPyGDataset` converts each annotation-level `GraphSample` directly
to one PyG temporal data item. It does not build another temporal window.

`TemporalCollator` pads variable-length samples to the longest sample in the
batch. Padding is represented by a false frame mask and false node/edge masks.
Edges are removed from each frame's message-passing graph whenever either
endpoint is invalid.

### Fixed topology contract

All samples collated into one temporal batch must use the same number and
ordering of nodes, the same number of edges, and an identical edge_index.
The collator also requires edge features to be present consistently and to
have the same feature dimension. A sample with a different keypoint set or
edge topology cannot be mixed into that batch.

A missing or low-confidence keypoint should therefore remain in its canonical
node position and be represented through the node-validity mask. Removing
nodes or changing their order is not supported by the current batching path.
The intended experiment uses one canonical skeleton definition throughout.

The temporal GNN performs masked node pooling and compacts valid frames before
the GRU. This supports both right-padding and internal invalid-frame gaps.

## Splitting and leakage

Group splitting occurs before training. The default is video-level splitting.
Track-level splitting must use the pair `(video_id, track_id)`; track IDs alone
are not globally unique. Label mappings are derived from training samples and
reused unchanged for validation and test.

The dataset-level two-way and three-way splitters use the same stratified
group-assignment routine. It balances behavior annotation counts, group
counts, and total sample counts while keeping every group intact. Rare
behaviors receive priority during assignment; multiple seeded restarts and
local group moves reduce sensitivity to the initial ordering. The random seed
makes a given input split reproducible.

Stratification cannot create independent examples that do not exist. A
behavior present in one group can only occur in one split and is preferentially
kept in training. Coverage across two or three splits is only possible when
the behavior occurs in at least two or three distinct groups, respectively;
even then, co-occurring labels and group constraints can make full coverage
infeasible. The split diagnostics report per-class sample counts, the number
of distinct groups containing each behavior, and which splits contain it.

## Legacy components and API compatibility

`TemporalSequenceBuilder` and `TemporalGraphSample` remain available for
older frame-graph workflows, but they are not part of the main annotation-level
experiment path.

`TemporalPyGDataset` accepts either one of the following homogeneous input
collections:

- `GraphSample` objects whose graph is already a `TemporalSkeletonGraph`
  (the current annotation-level path)
- `TemporalGraphSample` objects containing one `SkeletonGraph` per frame
  (the legacy frame-graph path)

The legacy adapter stacks frame graphs into the same `[T, V, F]` data
contract and derives frame, node, and edge masks. A dataset must not mix the
two sample types. Both paths can use the current `TemporalCollator`.

The experiment configuration's `sequence_length` and `sequence_stride`
fields are retained for configuration compatibility and are not used to
re-window annotation-level samples.
