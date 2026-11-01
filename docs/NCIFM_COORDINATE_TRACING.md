# NCIFM v0.4 — Coordinate-Resolved Dependency Tracing

**Status:** v0.4.1 has passed the owner's 21-test local suite; its GPT-2 coordinate run remains incomplete. v0.4.2 traversal and compact linear-fan-in changes are implemented on the feature branch; validation is pending.

## Objective

v0.4 extends the v0.3 tensor/operator dependency graph by resolving selected scalar coordinates through supported ATen operations. It records which scalar coordinates of input tensors and parameters are mathematical operands of a selected output coordinate.

This is a bounded and operation-specific tracer, not a claim of exhaustive neuron-level explanation.

## Important compatibility requirement

The v0.2 GPT-2 runtime graphs do not contain operator argument metadata. Coordinate tracing therefore requires a fresh runtime capture produced by the updated `RuntimeTensorGraph` (schema `0.3.0`). Do not run this tool against an older `apple_demo_v02/runtime_computational_graphs.json` and interpret missing rules as absent connections.

## Captured operator metadata

Each operator node now stores JSON-safe schema argument metadata:
- Scalar arguments such as axes, dimensions, normalized shapes, sizes and flags.
- References from named operator arguments to their tensor operand indexes.
- No tensor payloads are copied into the operator metadata.

The runtime graph retains its existing operand/produces edges. The schema version is now `0.3.0`.

## Coordinate rules in v0.4

Implemented local dependency rules include:
- Embedding lookups, including selected embedding-table row resolution when input token IDs are available.
- Elementwise unary operations and broadcast elementwise binary operations.
- Linear layers and matrix multiplication / batched matrix multiplication.
- `addmm` matrix products and addends.
- Layer normalization and native layer normalization.
- Softmax normalization.
- Transpose, permute, matrix transpose, view/reshape, slice, select, squeeze, unsqueeze and expand.
- Sum and mean reductions.

Embedding index dependencies are labelled separately from numeric value dependencies: a token ID selects a parameter row; it is not an additive numerical contribution.

Unsupported operators, missing argument metadata, fan-in limits, depth limits and graph-size limits are reported as explicit frontiers. The tracer does not infer rules for unknown operations.

## Limits and interpretation

- The operation-local API resolves one output scalar coordinate into its immediate input operand coordinates.
- The recursive trace walks backward from a selected output coordinate with configurable depth, node, edge and fan-in limits.
- A truncated or frontier-limited trace is incomplete. Missing edges under these conditions must not be interpreted as evidence that a dependency does not exist.
- A coordinate dependency describes mathematical operand structure for one execution. It does not measure contribution magnitude or causal necessity.
- Attention kernels, fused operators and data-dependent indexing operations may remain unsupported until their operator-specific rules are validated.
- This does not yet provide a complete neuron/attention-head map for GPT-2.

## Tests

```powershell
python -m py_compile src\metaobserver\computational_graph.py src\metaobserver\coordinate_tracer.py src\experiments\trace_coordinate_dependencies.py
python -m pytest tests\test_coordinate_tracer.py tests\test_path_tracer.py tests\test_neural_connectivity.py tests\test_metaobserver.py -q
```

The unit tests use small PyTorch models to test embedding row selection, broadcast addition, linear/addmm fan-in, end-to-end token-index reachability through embedding/mean/linear, and invalid coordinate rejection.

## Run against GPT-2

First make a new capture with operator argument metadata. Preserve the old report:

```powershell
python -m src.experiments.run_neural_connectivity_gpt2 `
  --model openai-community/gpt2 `
  --text "What is an apple?" `
  --output reports/neural_connectivity/apple_demo_v04 `
  --max-new-tokens 2
```

Then trace one selected logit coordinate backward:

```powershell
python -m src.experiments.trace_coordinate_dependencies `
  --trace-report reports/neural_connectivity/apple_demo_v04/trace_report.json `
  --runtime-graphs reports/neural_connectivity/apple_demo_v04/runtime_computational_graphs.json `
  --pass prompt_prefill `
  --input-position 3 `
  --output-token-rank 0 `
  --max-depth 12 `
  --max-nodes 20000 `
  --max-edges 50000 `
  --output reports/neural_connectivity/apple_demo_v04/apple_coordinate_dependencies.json
```

The report identifies whether the selected input token coordinate was reached in the expanded graph, lists scalar input and parameter coordinates, and records unresolved frontiers. Start with a smaller depth (for example, `--max-depth 4`) when exploring resource use.

## Acceptance gates

1. Runtime operator metadata is captured without retaining tensor payloads.
2. Coordinate rules are tested against manually understood small models.
3. Embedding row selection is resolved from the saved input IDs.
4. Linear, addmm and broadcast dependencies identify correct scalar coordinates.
5. Recursive traces report explicit resource limits and unsupported frontiers.
6. The existing v0.1-v0.3 tests continue to pass.
7. No claim of a complete GPT-2 scalar graph is made until unsupported and fused operators have validated rules.


## v0.4.1 — Compact high-fan-in groups

The first GPT-2 run exposed a scale problem in the explicit scalar-edge representation. A selected logit passes through the tied output embedding projection and final LayerNorm. Expanding 768 LayerNorm output coordinates independently into all 768 coordinates in each normalized slice attempts to materialize 589,824 input edges for that slice family before continuing upstream.

The v0.4.1 implementation represents LayerNorm, softmax and reduction fan-in as compact `dependency_groups`. Each group stores the source tensor, fixed coordinate prefix, varying axes, axis sizes, operand index, relation and dependency count. The recursive walk still visits each member coordinate (subject to the configured limits), but the JSON graph no longer needs one edge record per member-to-output pair.

The explicit edge cap now stops edge expansion immediately and emits a bounded frontier record rather than appending one record for every remaining dependency. This prevents the frontier list from growing far beyond the configured edge limit.

The first GPT-2 coordinate result was inconclusive: the 50,000-edge cap was reached before input token position 3 was reached. A null reachability result is not evidence of no dependency. The compact-group change is intended to make the next trace more tractable; it does not guarantee a complete GPT-2 scalar graph.

A new small-model test validates compact LayerNorm group metadata and confirms that all normalized input coordinates are reached recursively. Run the full tests and repeat the GPT-2 coordinate experiment before treating v0.4.1 as validated.


## v0.4.2 — Unique coordinate scheduling and compact linear fan-in

The owner's v0.4.1 rerun passed compilation and **21 tests**, but the GPT-2 coordinate query remained incomplete: 13,911 coordinate nodes, 50,000 explicit edges, 768 dependency groups, 2,304 parameter coordinates, 577,545 frontier records, zero reached input coordinates, and null focus reachability. The edge limit was reached and the graph was truncated. This result does not establish that the selected token is disconnected from the logit.

Code inspection identified a traversal inefficiency in addition to the graph's genuine fan-in:
- Every output coordinate in a shared normalization slice can refer to the same source-coordinate group.
- v0.4.1 preserved the compact group record but scheduled its source coordinates repeatedly for each output coordinate.
- Duplicate pending coordinates consumed the node budget before the walker could finish processing unique coordinates. When the cap was reached, the implementation emitted a frontier record for each remaining duplicate candidate.

The v0.4.2 follow-up changes:
- Track scheduled coordinate IDs, not just already-processed nodes, so a source tensor coordinate is queued once.
- Deduplicate traversal of identical dependency-group source slices while retaining separate logical group records for their distinct target coordinates.
- Represent `aten::linear`, `aten::matmul`, `aten::bmm`, and `aten::addmm` contraction fan-in as compact dependency groups. Bias and addend dependencies remain explicit scalar dependencies.
- Record a bounded group-level `max_nodes` frontier summary instead of writing one frontier item for every omitted group member.
- Add a recursive LayerNorm-plus-Linear regression test designed to expose repeated group scheduling under a small node cap, and update the local linear/addmm coordinate assertions for compact groups.

**Owner-local validation:** compilation passed and **22 tests passed in 3.77 seconds**. The GPT-2 query against the existing v0.4 capture reached the 20,000-node limit, with 19,213 explicit edges (below the 50,000 edge limit), 2,306 dependency groups, 4,626 parameter coordinates and 10,767 frontier records. Zero input coordinates were reached and focus reachability remained null. The trace was truncated; 6,141 frontier records were classified as unsupported or unresolved in the summary. This moves the observed bottleneck from explicit-edge expansion to the configured coordinate-node budget, but it does not establish that the selected token is disconnected from the logit.

The next diagnostic is to summarize frontier reasons, operator names and module scopes in `apple_coordinate_dependencies_v042.json`. That evidence will distinguish missing coordinate rules from the breadth of a supported dependency expansion. No GPT-2 input-to-logit scalar path or complete scalar graph is claimed. The coordinate graph remains a bounded mathematical dependency trace for one execution, not semantic attribution or causal proof.
