# NCIFM v0.4 — Coordinate-Resolved Dependency Tracing

**Status:** Implementation added; local execution and validation are pending.

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
