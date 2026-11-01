# NCIFM — Computational Path Query Engine

**Milestone:** NCIFM v0.3 — upstream/downstream tensor path queries.

## What this adds

The path-query engine consumes the v0.2 runtime graph and resolves downstream tensors/operators reachable from an input tensor; upstream operators, activations, external operands and parameters feeding the logits tensor; the intersection containing nodes on at least one source-to-target route; one shortest representative route; parameter dependencies; and operator counts grouped by module.

It does not enumerate every possible route (which can be combinatorially large).

## Run against the saved apple experiment

    python -m py_compile src\metaobserver\path_tracer.py src\experiments\trace_computational_path.py
    python -m pytest tests/test_path_tracer.py tests/test_neural_connectivity.py tests/test_metaobserver.py -q

    python -m src.experiments.trace_computational_path --trace-report reports/neural_connectivity/apple_demo_v02/trace_report.json --runtime-graphs reports/neural_connectivity/apple_demo_v02/runtime_computational_graphs.json --pass prompt_prefill --input-position 3 --output-token-rank 0 --output reports/neural_connectivity/apple_demo_v02/apple_to_top_logit_path.json

To query generation_step_3, replace the pass name and choose a separate output filename.

## Coordinate semantics

The input position selects a token coordinate in the original [batch, sequence] input tensor. The output position/token ID selects a scalar coordinate in the [batch, sequence, vocabulary] logits tensor. The graph still represents whole tensors and ATen operators. The coordinate is metadata; the tool does not claim that it has isolated all scalar-level connections from that token ID to that vocabulary logit.

To query a non-final output position, pass output-position N and output-token-id ID. Saved top-token ranks are available only for the final sequence position.

## Output

The report contains query metadata and token/logit coordinates, the full upstream dependency subgraph including parameters, the source-to-target route subgraph, a shortest representative route, parameter names/shapes feeding the target tensor, operator counts grouped by module, and interpretation limitations.

## Evidence limits

A recorded computational dependency means a tensor is an operand to an operation that contributes to a later tensor in this execution. It does not quantify how much a parameter or input token contributes, prove a human-readable semantic function, or establish causal necessity. Those require gradient/attribution diagnostics and controlled ablation or activation-patching experiments.
