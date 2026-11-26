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

For declared-contract inputs, optional masks use `frame_mask[T]` and
`node_mask[T,V]`. If omitted, all frames and nodes are treated as valid.
Graph JSON loading preserves these masks when they are supplied.

The loader rejects a declared contract that fails any check. It does not silently reorder edges or attempt to guess how edge attributes should be remapped.

## Legacy compatibility

Archives and JSON graph records without `edge_feature_names` continue through the legacy loading path. Their edge attributes are shape-checked by the graph and dataset classes, but are not interpreted as the new nine-channel contract. This allows older experiments to remain readable while making new-format data explicit and auditable.

New data exports should always include `edge_feature_names`. The CVAT adapter and real-video export are still separate future work; this contract does not claim that a CVAT export has been integrated or validated.
