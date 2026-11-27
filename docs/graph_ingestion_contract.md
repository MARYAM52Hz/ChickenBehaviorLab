# Graph ingestion contract

The training-facing graph representation is a temporal skeleton graph with:

- node features shaped `(T, V, F)`;
- shared spatial connectivity shaped `(2, E)`;
- optional frame and node masks;
- temporal edge attributes shaped `(T, E, D)`.

## Declaring the combined edge-feature contract

New NPZ feature archives and graph JSON records may declare:

- `edge_feature_names`: the exact ordered nine-channel contract;
- `node_feature_names`: optional names for the node-feature channels when they do not use the legacy canonical XY or seven-channel layout.

The edge-feature order is:

1. `relative_dx`
2. `relative_dy`
3. `distance`
4. `delta_vx`
5. `delta_vy`
6. `delta_ax`
7. `delta_ay`
8. `source_confidence`
9. `target_confidence`

When `edge_feature_names` is present, ingestion validates all of the following before constructing a training sample:

- exact channel names and ordering;
- expected `(T, E, 9)` shape;
- finite edge values;
- values match the canonical edge features recomputed from node features;
- invalid edges are zeroed consistently with the optional frame and node masks;
- edge topology is already canonical, deduplicated, and bidirectional.

Optional masks use `frame_mask[T]` and `node_mask[T,V]`. If omitted, all
frames and nodes are treated as valid. Masks must be boolean or real numeric
arrays containing only 0 and 1; complex-valued masks are rejected. Graph JSON
loading preserves the supplied mask values until validation.

All ingestion paths, including legacy inputs without edge metadata, reject
non-finite node and edge features. Edge arrays must have consistent temporal
and edge dimensions; an empty edge set is valid when represented as
`edge_index[2,0]`. For JSON with a declared edge contract, the missing
trailing edge-feature width of an empty nested list is recovered from
`edge_feature_names`.

The loader rejects a declared contract that fails any check. It does not silently reorder edges or attempt to guess how edge attributes should be remapped.

## Legacy compatibility

Archives and JSON graph records without `edge_feature_names` continue through
the legacy loading path. Their edge attributes are shape-checked and checked
for finite values, but are not interpreted as the new nine-channel contract.
This allows older experiments to remain readable while making new-format data
explicit and auditable.

New data exports should always include `edge_feature_names`. The CVAT adapter and real-video export are still separate future work; this contract does not claim that a CVAT export has been integrated or validated.


## In-memory graph and PyG boundary

`SkeletonGraph` and `TemporalSkeletonGraph` also validate edge-index integer
dtypes and boolean/binary masks before downstream conversion. Temporal node
feature dimensions must be non-zero. This prevents direct in-memory
`GraphSample` construction from bypassing the dtype and mask checks that file
ingestion performs.

The temporal PyG path retains the shared spatial topology and temporal edge
attributes. Empty spatial edge sets are supported as `edge_index[2,0]` with
`edge_attr[T,0,D]`; the temporal collator and edge-aware GNN should preserve
these shapes without inventing self-edges.
