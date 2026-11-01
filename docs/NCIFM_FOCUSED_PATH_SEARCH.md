# NCIFM v0.5 — Focused Coordinate Path Search

**Status:** v0.5.3 is locally validated by the owner (32 regression tests passed) and hosted CI passed. v0.5.4 adds source-guided scheduling; validation is pending.

## Objective

Find a computational dependency path between one exact source tensor coordinate and one exact target tensor coordinate in a captured execution graph. The search is backward from the target, but the returned path is ordered from source to target.

This mode complements rather than replaces the v0.4 full dependency-closure tracer.

## API

    trace_focused_coordinate_path(graph, source_tensor_id, source_coordinate,
        target_tensor_id, target_coordinate, tensor_values=None,
        max_depth=128, max_nodes=50000, max_candidates=100000,
        max_frontier=10000, parameter_policy="compact",
        search_strategy="source_guided")

The existing trace_coordinate_graph API remains the full dependency-closure mode.

## Search behaviour

1. Validate the source and target tensor IDs and coordinates.
2. Start at the target coordinate and walk backward through supported ATen dependency rules using trace_tensor_coordinate. By default, prioritize queued coordinates using shortest tensor-level distance from the requested source; use `search_strategy="breadth_first"` for the previous FIFO ordering.
3. Explore activation and input-index dependencies.
4. Retain parameter dependencies as compact records by default. If the requested source is a parameter coordinate, test whether it belongs to a compact parameter group and allow it to terminate the search.
5. Deduplicate identical activation dependency groups before enumerating their scalar members. This avoids repeatedly expanding the same LayerNorm/softmax input slice for every output coordinate while preserving a valid representative route through the first encountered target coordinate. Focused search does not enumerate every alternate parent edge and is not a complete scalar dependency graph.
6. Reconstruct a representative path in forward computational order.
7. Report unsupported operations, value-required dependencies, depth limits and resource limits explicitly.

The search does not expand parameter groups by default. Set parameter_policy to expand only when a parameter-source query or an explicitly broader search requires it. Identical activation groups are expanded once and counted in the result summary as `activation_dependency_groups_deduplicated`; the first encountered group target is retained as the representative route. Candidate and node budgets remain in force.

## v0.5.2 source-aware search

The focused tracer now builds a conservative tensor-level reachability index by following captured operand-to-operator-to-output edges from the requested source tensor. Activation dependency groups and direct activation dependencies whose tensor IDs are not reachable from that source are skipped before scalar-coordinate scheduling. This pruning is only a coarse impossibility filter; supported coordinate rules remain the authority for exact scalar dependencies.

Before enumerating a compact activation dependency group, the tracer checks whether the exact requested source coordinate belongs to that group. A match creates the exact source-to-current-coordinate link immediately, avoiding expansion of the rest of the group. Parameter groups remain visible as compact provenance records, but unrelated parameter tensors are not expanded while searching for an activation source.

New summary fields:
- `coordinate_nodes_processed`: coordinates dequeued and expanded (also retained under the legacy `coordinate_nodes_examined` field).
- `coordinate_nodes_scheduled`: processed plus queued coordinates.
- `queue_remaining_at_stop`: coordinates still queued when search stopped.
- `source_reachable_tensor_count`: number of runtime tensor nodes reachable from the requested source tensor.

These metrics distinguish a full scheduling budget from the smaller number of coordinates actually processed. Tensor-level reachability does not prove a coordinate-level path; it only removes branches that cannot contain the requested source tensor at all.

## Result interpretation

Statuses:
- path_found: a validated source-to-target dependency path was reconstructed.
- path_not_found: supported search space was exhausted without finding the requested source.
- inconclusive_limit: a node, candidate or frontier limit prevented completion.
- inconclusive_unsupported: an unsupported operator/rule prevented completion.

A found path may coexist with unresolved alternative branches. Inspect result.search_complete and frontier. A path_not_found result is conclusive only when search_complete is true.

Embedding lookup indices are routing dependencies: the token ID selects an embedding row and is not an additive numerical contribution. A path is a mathematical operand dependency for one captured execution; it is not contribution magnitude, semantic interpretation or causal necessity.

## Synthetic tests

    python -m py_compile src/metaobserver/focused_path_tracer.py src/experiments/trace_focused_coordinate_path.py
    python -m pytest tests/test_focused_path_tracer.py tests/test_coordinate_tracer.py tests/test_path_tracer.py tests/test_neural_connectivity.py tests/test_metaobserver.py -q

The focused tests cover a known embedding/mean/linear path, an embedding/LayerNorm/linear path with shared-group deduplication, exact activation-source matching without group enumeration, pruning of an unrelated second input branch, an explicitly requested parameter coordinate, unsupported operators, node-budget exhaustion, invalid coordinates and identical source/target coordinates. Existing full-closure tests remain in the regression suite.

## GPT-2 apple experiment

    python -m src.experiments.trace_focused_coordinate_path --trace-report reports/neural_connectivity/apple_demo_v04/trace_report.json --runtime-graphs reports/neural_connectivity/apple_demo_v04/runtime_computational_graphs.json --pass prompt_prefill --source-input input_ids --input-position 3 --output-name logits --output-position 4 --output-token-id 198 --max-depth 128 --max-nodes 50000 --max-candidates 100000 --parameter-policy compact --output reports/neural_connectivity/apple_demo_v05/apple_focused_path.json

The CLI reads saved prompt token IDs from the trace report so embedding lookup indices can be resolved. It expects a rank-2 token input and rank-3 logits output.

## Validation boundary

The GPT-2 v0.4.2 query previously reached its 20,000-node limit without reaching the selected input coordinate. That result was inconclusive, not evidence of disconnection. No v0.5 GPT-2 result is claimed until the owner runs the new command and shares the actual output.

## v0.5.4 — Source-guided scheduling

The source-reachability pass now records shortest tensor-level distances from the requested source instead of only a reachable-tensor set. The default `source_guided` strategy uses these distances to prioritize the backward coordinate queue, exploring source-nearer tensor branches first. `breadth_first` retains FIFO breadth-first ordering for regression comparisons and reproducibility.

This is strictly a scheduling heuristic: it does not discard reachable tensors or scalar coordinates, and it does not replace the operation-local coordinate rules. With sufficient budgets, both strategies search the same supported dependency space. Node, candidate, depth and frontier limits still apply. The result summary records `search_strategy` and `target_tensor_distance`.

A regression test exercises a model in which two LayerNorm output coordinates share an identical activation dependency group. It verifies that deduplicating the shared group still yields a valid representative source-to-target path under breadth-first ordering. A second test verifies that both search strategies find the same known linear dependency.
## v0.5.3 — Budget draining and search profiling

The v0.5.3 correction changes resource-limit handling. Reaching `max_nodes` or `max_candidates` stops the current expansion from scheduling or examining additional dependencies, but it no longer aborts the entire search immediately. Coordinates already in the queue continue to be processed. This allows the tracer to discover a valid path among work that was already scheduled while still recording that unexplored dependencies remain.

A result may therefore report `status: path_found` together with `resource_limit_reached: true` and `search_complete: false`. The path is a validated dependency route; the resource-limit flag means the search did not exhaust all alternative branches. A path-not-found result remains conclusive only when `search_complete` is true.

The tracer now records:
- `reachability_seconds`: time spent building the source-reachable tensor set.
- `search_seconds`: total focused-search loop time.
- `operator_profile`: coordinate-expansion count and cumulative `trace_tensor_coordinate` time grouped by ATen operator.
- `queue_peak_size`, `queue_remaining_at_stop`, and `max_depth_processed`.
- `coordinate_nodes_processed` and `coordinate_nodes_recorded`. The legacy `coordinate_nodes_examined` field now aliases the actual processed count rather than the number of records retained.

Operator timings measure the coordinate-rule expansion call, not total time spent enumerating groups, scheduling coordinates, serializing JSON, or executing the model. Use them to identify candidates for deeper profiling, not as a complete end-to-end performance breakdown.

A captured PyTorch regression test uses two clone branches from the same input and a three-node scheduling budget. The target model-output alias and its producer add coordinate consume the first two slots; the add then queues one clone branch in the third slot and reaches the limit on the other. Expanding the already-queued clone must still discover the exact source coordinate. This keeps the test on the real runtime-graph capture path while avoiding the deeper fan-in of linear layers.

