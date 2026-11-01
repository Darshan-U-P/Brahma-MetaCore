# Brahma MetaCore — NCIFM Execution Plan

**Milestone:** Neural Connectivity and Information-Flow Mapper v0.2  
**Prototype:** GPT-2 Small / Hugging Face Transformers

## Evidence from the first local run

The v0.1 run loaded GPT-2 on CUDA, inventoried 124,439,808 parameters and 163 modules, captured 12 prompt attention layers with shape [1, 12, 5, 5], and recorded 13 hidden-state stages. The supplied JSON showed that module-hook data had been overwritten by generation: attention described the five-token prompt while the final module activations described a 12-token context.

## Objectives

1. Keep prompt prefill and every autoregressive generation pass in separate records.
2. Store each pass's token IDs, tokens, context length, module outputs, hidden states and optional attention.
3. Reuse prompt-prefill logits for generation step zero.
4. Compare instrumented prompt logits with an uninstrumented direct forward pass.
5. Preserve static module/parameter ownership graph.
6. Capture actual ATen tensor operands and outputs, with module-scope annotations.
7. Store the runtime graph separately from the structural graph.

## Graph semantics

**Structural graph:** module containment and parameter ownership.

**Runtime tensor dependency graph:** PyTorch dispatch mode observes ATen operations and the tensors consumed/produced. Each operation becomes an operator node, input tensors connect to it with operand edges, and its result tensors connect with produces edges. Every operator is tagged with its innermost active module where available.

This is an execution-specific dependency graph. It does not decompose dense matrix multiplication into every scalar neuron-to-neuron contribution and does not establish semantic meaning or causal importance. Attention weights are token-position routing values, not explanations.

GPT-2's Q/K/V projection is fused in c_attn. The report describes the logical Q/K/V contiguous slices and preserves the fused tensor.

## Files written per run

- trace_report.json — tokenization, model inventory, separate pass records, module activations, hidden states, attention, logits, generation sequence and parity check.
- model_graph.json — static module/parameter ownership graph.
- runtime_computational_graphs.json — ATen operator/tensor dependency graph for captured passes.

## Windows commands

```powershell
git pull origin feature/neural-connectivity-mapper-v01

python -m py_compile src\metaobserver\computational_graph.py src\metaobserver\neural_connectivity.py
python -m pytest tests/test_neural_connectivity.py tests/test_metaobserver.py -q

# Use a new directory; do not overwrite apple_demo v0.1.
python -m src.experiments.run_neural_connectivity_gpt2 `
  --model openai-community/gpt2 `
  --text "What is an apple?" `
  --output reports/neural_connectivity/apple_demo_v02 `
  --max-new-tokens 8
```

Optional full attention at each generation step:

```powershell
python -m src.experiments.run_neural_connectivity_gpt2 `
  --model openai-community/gpt2 `
  --text "What is an apple?" `
  --output reports/neural_connectivity/apple_demo_v02_full_attention `
  --max-new-tokens 8 `
  --capture-generation-attention
```

Optional operator graphs for generation passes (higher overhead; start with fewer tokens):

```powershell
python -m src.experiments.run_neural_connectivity_gpt2 `
  --model openai-community/gpt2 `
  --text "What is an apple?" `
  --output reports/neural_connectivity/apple_demo_v02_full_ops `
  --max-new-tokens 3 `
  --trace-generation-ops
```

Default behavior captures an operator graph for prompt prefill and pass-specific module/hidden-state records during generation. Generation attention and generation operator graphs are opt-in.

## Acceptance gates

- Prompt module activations match the prompt sequence length.
- Every generated-token forward pass has a separate context length and activation record.
- No pass overwrites another.
- Runtime graph includes model-input, parameter, operator and computed-tensor nodes plus operand/produces edges.
- Operator graph reports truncation if max_ops is reached.
- Instrumented logits match direct-forward logits within tolerance.
- Hooks are removed after success or exceptions.
- Existing MetaObserver tests remain unchanged.

## Later stages

Add graph path queries, upstream logit tracing, downstream input tracing, controlled interventions, interactive 3D visualization and architecture adapters for other Transformers, ANN and CNN families.

## Resource controls

- Operator trace cap: 50,000 ATen operations per captured pass.
- Activation values retained up to 50,000 per tensor; larger tensors keep a prefix and complete summary statistics.
- Prompt attention is captured by default.
- Generation attention and generation-pass operator graphs are opt-in.
- Keep reports/checkpoints out of Git and preserve each experiment in a separate output directory.
