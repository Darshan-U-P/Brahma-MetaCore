# NCIFM v0.5 — Focused Coordinate Path Search

**Status:** Implementation committed to the feature branch. Local owner execution and GPT-2 validation are still pending.

## Objective

Find a computational dependency path between one exact source tensor coordinate and one exact target tensor coordinate in a captured execution graph. The search is backward from the target, but the returned path is ordered from source to target.

This mode complements rather than replaces the v0.4 full dependency-closure tracer.

## API

    trace_focused_coordinate_path(graph, source_tensor_id, source_coordinate,
        target_tensor_id, target_coordinate, tensor_values=None,
        max_depth=128, max_nodes=50000, max_candidates=100000,
        max_frontier=10000, parameter_policy="compact")

The existing trace_coordinate_graph API remains the full dependency-closure mode.

## Search behaviour

1. Validate the source and target tensor IDs and coordinates.
2. Start at the target coordinate and walk backward through supported ATen dependency rules using trace_tensor_coordinate.
3. Explore activation and input-index dependencies.
4. Retain parameter dependencies as compact records by default. If the requested source is a parameter coordinate, test whether it belongs to a compact parameter group and allow it to terminate the search.
5. Deduplicate identical activation dependency groups before enumerating their scalar members. This avoids repeatedly expanding the same LayerNorm/softmax input slice for every output coordinate while preserving a valid route through the first encountered target coordinate.
6. Reconstruct a representative path in forward computational order.
7. Report unsupported operations, value-required dependencies, depth limits and resource limits explicitly.

The search does not expand parameter groups by default. Set parameter_policy to expand only when a parameter-source query or an explicitly broader search requires it. Identical activation groups are expanded once and counted in the result summary as `activation_dependency_groups_deduplicated`; the first encountered group target is retained as the representative route. Candidate and node budgets remain in force.

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

The focused tests cover a known embedding/mean/linear path, an embedding/LayerNorm/linear path with shared-group deduplication, an explicitly requested parameter coordinate, unsupported operators, node-budget exhaustion, invalid coordinates and identical source/target coordinates. Existing full-closure tests remain in the regression suite.

## GPT-2 apple experiment

    python -m src.experiments.trace_focused_coordinate_path --trace-report reports/neural_connectivity/apple_demo_v04/trace_report.json --runtime-graphs reports/neural_connectivity/apple_demo_v04/runtime_computational_graphs.json --pass prompt_prefill --source-input input_ids --input-position 3 --output-name logits --output-position 4 --output-token-id 198 --max-depth 128 --max-nodes 50000 --max-candidates 100000 --parameter-policy compact --output reports/neural_connectivity/apple_demo_v05/apple_focused_path.json

The CLI reads saved prompt token IDs from the trace report so embedding lookup indices can be resolved. It expects a rank-2 token input and rank-3 logits output.

## Validation boundary

The GPT-2 v0.4.2 query previously reached its 20,000-node limit without reaching the selected input coordinate. That result was inconclusive, not evidence of disconnection. No v0.5 GPT-2 result is claimed until the owner runs the new command and shares the actual output.
