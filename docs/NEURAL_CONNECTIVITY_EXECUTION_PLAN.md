# Brahma MetaCore — Neural Connectivity Execution Plan

**Milestone:** NCIFM v0.1 — GPT-2 passive trace prototype.

## Objective

For an input such as `What is an apple?`, record tokenization, module and parameter inventory, input-conditioned hidden states, per-layer attention tensors with token-position axes, MLP/module outputs, logits, top next-token probabilities and optional greedy-generation steps.

## Components

1. Hugging Face GPT-2 model/tokenizer loader.
2. Architecture/module/parameter inventory including tensor dimensions and statistics.
3. Token and position recorder.
4. Runtime module hooks with guaranteed cleanup.
5. Attention capture using eager attention outputs.
6. Hidden-state and MLP-output capture.
7. Structural module/parameter ownership graph.
8. Execution-order graph, explicitly distinguished from a true fine-grained tensor dependency graph.
9. JSON report writer using a new experiment directory.

## Graph semantics

Structural edges describe module containment and parameter ownership. Execution-order edges describe observed stage ordering only. Attention weights describe token-position routing; neither attention nor co-activation alone establishes semantic meaning or causality. Future tensor-level dependencies require operator-level tracing (e.g. FX/export or architecture-specific instrumentation), including tensor operands and parameter operands. Causal attribution requires interventions and controls.

## Execution phases

1. Run unit tests for graph creation, inventory, JSON serialization and hook cleanup.
2. Run one GPT-2 prompt and inspect trace artifacts.
3. Compare captured logits with a direct forward pass.
4. Repeat identical prompt/model revision to test numerical repeatability.
5. Sweep controlled prompt variants (apple/banana/car; fruit/company; context changes).
6. Add interventions only after passive trace fidelity is validated.
7. Generalize adapter APIs to other Transformers, then ANN/CNN models.
8. Build interactive 3D visualization on a stable graph schema.

## Windows PowerShell

```powershell
# Activate the project's compatible environment first.
python -m pip install transformers
python -m pytest tests/test_neural_connectivity.py tests/test_metaobserver.py -q

python -m src.experiments.run_neural_connectivity_gpt2 `
  --model openai-community/gpt2 `
  --text "What is an apple?" `
  --output reports/neural_connectivity/apple_demo `
  --max-new-tokens 8

Get-Item reports/neural_connectivity/apple_demo/trace_report.json
Get-Item reports/neural_connectivity/apple_demo/model_graph.json
```

The first execution downloads GPT-2. Auto device selection uses CUDA when available and CPU otherwise. With a 4 GB GPU, begin with short prompts. Full attention values are retained only below a size threshold; summary statistics remain available.

## Storage and reproducibility

Use unique run directories; preserve prior outputs. Store model revision, environment, device, input text, token IDs, tokens and tensor metadata. Avoid committing checkpoints or large traces. Later batch runs should use resumable manifests, compressed shards and disk quotas.

## Acceptance gates

- Inventory and graph serialize to JSON.
- Token IDs and token strings are preserved.
- Hidden states, attention and MLP outputs are captured for supported GPT-2 implementations.
- Logits and top next-token probabilities are reported.
- Hooks are removed even when execution raises.
- Existing MetaObserver tests remain unchanged.
- Direct-forward parity and repeatability are checked before interpreting traces.

## Not included in v0.1

Exhaustive input coverage, automatic neuron semantics, complete neuron-to-neuron causal path decomposition, interactive 3D visualization, automatic weight editing, capability transfer or model construction.
