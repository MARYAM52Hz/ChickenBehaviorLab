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

The temporal GNN performs masked node pooling and compacts valid frames before
the GRU. This supports both right-padding and internal invalid-frame gaps.

## Splitting and leakage

Group splitting occurs before training. The default is video-level splitting.
Track-level splitting must use the pair `(video_id, track_id)`; track IDs alone
are not globally unique. Label mappings are derived from training samples and
reused unchanged for validation and test.

## Legacy components

`TemporalSequenceBuilder` and `TemporalGraphSample` remain available for
older frame-graph workflows, but they are not part of the main annotation-level
experiment path. The experiment configuration's `sequence_length` and
`sequence_stride` fields are retained for compatibility and are not used to
re-window annotation-level samples.
