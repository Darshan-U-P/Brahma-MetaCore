"""GPT-2 input-to-output tracing for Brahma MetaCore NCIFM."""
from __future__ import annotations
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils._pytree import tree_flatten
from src.metaobserver.computational_graph import RuntimeTensorGraph

def _tensor_record(tensor: torch.Tensor, max_values: int = 50_000) -> dict[str, Any]:
    """Summarize one tensor and retain bounded values for inspection."""
    detached = tensor.detach()
    stats = detached.float()
    record: dict[str, Any] = {
        "shape": list(detached.shape), "dtype": str(detached.dtype),
        "device": str(detached.device), "numel": detached.numel(),
        "mean": stats.mean().item() if stats.numel() else 0.0,
        "std": stats.std(unbiased=False).item() if stats.numel() else 0.0,
        "min": stats.min().item() if stats.numel() else 0.0,
        "max": stats.max().item() if stats.numel() else 0.0,
    }
    if detached.numel() <= max_values:
        record["values"] = detached.cpu().tolist()
        record["values_complete"] = True
    elif max_values > 0:
        record["values_prefix"] = detached.reshape(-1)[:max_values].cpu().tolist()
        record["values_complete"] = False
    return record

def inspect_gpt2(model: nn.Module, include_parameter_prefix: int = 0) -> dict[str, Any]:
    """Inventory module hierarchy and every parameter tensor."""
    modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        modules.append({
            "id": f"module:{name}", "name": name,
            "module_type": type(module).__name__,
            "children": [f"module:{name}.{child}" for child, _ in module.named_children()],
            "direct_parameters": [
                f"parameter:{name}.{parameter_name}"
                for parameter_name, _ in module.named_parameters(recurse=False)
            ],
        })
    parameters = []
    for name, parameter in model.named_parameters():
        item = {
            "id": f"parameter:{name}", "name": name,
            "shape": list(parameter.shape), "dtype": str(parameter.dtype),
            "numel": parameter.numel(), "requires_grad": parameter.requires_grad,
            "statistics": _tensor_record(parameter, max_values=0),
        }
        if include_parameter_prefix > 0:
            item["values_prefix"] = parameter.detach().flatten()[:include_parameter_prefix].cpu().tolist()
        parameters.append(item)
    config = getattr(model, "config", None)
    return {
        "model_type": getattr(config, "model_type", type(model).__name__),
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "modules": modules, "parameters": parameters,
        "config": config.to_dict() if config is not None else {},
    }

def build_dependency_graph(model_map: dict[str, Any]) -> dict[str, Any]:
    """Build structural containment and parameter-ownership edges."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for module in model_map["modules"]:
        nodes.append({"id": module["id"], "kind": "module",
                      "label": module["name"], "module_type": module["module_type"]})
        edges.extend({"source": module["id"], "target": child, "relation": "contains"}
                     for child in module["children"])
        edges.extend({"source": module["id"], "target": parameter, "relation": "owns_parameter"}
                     for parameter in module["direct_parameters"])
    nodes.extend({"id": parameter["id"], "kind": "parameter",
                  "label": parameter["name"], "shape": parameter["shape"]}
                 for parameter in model_map["parameters"])
    return {
        "graph_type": "structural_module_parameter_graph",
        "edge_semantics": (
            "Module containment and parameter ownership only; runtime tensor "
            "dependencies are stored in runtime_computational_graphs.json."
        ),
        "nodes": nodes, "edges": edges,
    }

def _capture_relevant_module(name: str) -> bool:
    if name in {"transformer.wte", "transformer.wpe", "transformer.drop",
                "transformer.ln_f", "lm_head"}:
        return True
    suffixes = (
        ".ln_1", ".attn.c_attn", ".attn.c_proj", ".attn",
        ".ln_2", ".mlp.c_fc", ".mlp.act", ".mlp.c_proj", ".mlp",
    )
    return name.startswith("transformer.h.") and name.endswith(suffixes)

class GPT2TraceRecorder:
    """Capture one prompt prefill and separate records for each generation pass."""
    def __init__(
        self, model: nn.Module, tokenizer: Any, top_k: int = 20,
        max_ops: int = 50_000, max_activation_values: int = 50_000,
        capture_generation_attention: bool = False,
        trace_generation_ops: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.max_ops = max_ops
        self.max_activation_values = max_activation_values
        self.capture_generation_attention = capture_generation_attention
        self.trace_generation_ops = trace_generation_ops
        self._active_pass: dict[str, Any] | None = None
        self._handles: list[Any] = []

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _install_activation_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not _capture_relevant_module(name):
                continue
            def capture(current_module: nn.Module, module_inputs: tuple[Any, ...],
                        output: Any, module_name: str = name) -> None:
                if self._active_pass is None:
                    return
                input_tensors = [
                    item for item in tree_flatten(module_inputs)[0]
                    if isinstance(item, torch.Tensor)
                ]
                output_tensors = [
                    item for item in tree_flatten(output)[0]
                    if isinstance(item, torch.Tensor)
                ]
                self._active_pass["module_events"].append({
                    "module": module_name,
                    "module_type": type(current_module).__name__,
                    "input_shapes": [list(item.shape) for item in input_tensors],
                    "outputs_ref": output_tensors,
                })
            self._handles.append(module.register_forward_hook(capture))

    def _finalize_module_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        finalized = []
        for event in events:
            outputs = [
                _tensor_record(value, self.max_activation_values)
                for value in event.pop("outputs_ref")
            ]
            finalized.append({
                "module": event["module"], "module_type": event["module_type"],
                "input_shapes": event["input_shapes"], "outputs": outputs,
            })
        return finalized

    def _top_predictions(self, logits: torch.Tensor) -> list[dict[str, Any]]:
        probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
        values, indices = torch.topk(probabilities, k=min(self.top_k, probabilities.numel()))
        return [
            {"token_id": int(token_id),
             "token": self.tokenizer.decode([int(token_id)]),
             "probability": float(probability)}
            for probability, token_id in zip(values.cpu().tolist(), indices.cpu().tolist())
        ]

    def _run_pass(self, pass_id: str, input_ids: torch.Tensor, *,
                  capture_attention: bool, capture_ops: bool
                  ) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any] | None]:
        current: dict[str, Any] = {
            "pass_id": pass_id, "module_events": [],
            "input_token_ids": input_ids[0].detach().cpu().tolist(),
            "sequence_length": int(input_ids.shape[1]),
        }
        self._active_pass = current
        graph = RuntimeTensorGraph(self.model, max_ops=self.max_ops) if capture_ops else None
        try:
            with torch.inference_mode():
                if graph is not None:
                    graph.register_input("input_ids", input_ids)
                    graph.install_module_scopes()
                    with graph:
                        outputs = self.model(
                            input_ids=input_ids, output_attentions=capture_attention,
                            output_hidden_states=True, use_cache=False, return_dict=True,
                        )
                        graph.add_output("logits", outputs.logits)
                else:
                    outputs = self.model(
                        input_ids=input_ids, output_attentions=capture_attention,
                        output_hidden_states=True, use_cache=False, return_dict=True,
                    )
        finally:
            if graph is not None:
                graph.remove_module_scopes()
            self._active_pass = None

        logits = outputs.logits.detach()
        pass_record: dict[str, Any] = {
            "pass_id": pass_id,
            "input_token_ids": current["input_token_ids"],
            "input_tokens": self.tokenizer.convert_ids_to_tokens(current["input_token_ids"]),
            "sequence_length": current["sequence_length"],
            "module_outputs": self._finalize_module_events(current["module_events"]),
            "hidden_states": [], "attention": [],
            "logits": _tensor_record(logits, max_values=0),
            "last_position_top_tokens": self._top_predictions(logits),
        }
        for index, hidden in enumerate(outputs.hidden_states or ()):
            item = _tensor_record(hidden, self.max_activation_values)
            item["stage"] = "embedding_output" if index == 0 else f"block_output:{index - 1}"
            pass_record["hidden_states"].append(item)
        for layer_index, attention in enumerate(outputs.attentions or ()):
            pass_record["attention"].append({
                "layer": layer_index, "shape": list(attention.shape),
                "head_count": int(attention.shape[1]),
                "query_length": int(attention.shape[2]),
                "key_length": int(attention.shape[3]),
                **_tensor_record(attention, max_values=1_000_000),
            })

        # GPT-2's Q/K/V are slices of a single fused c_attn projection.
        hidden_size = int(getattr(getattr(self.model, "config", None), "n_embd", 0) or 0)
        if hidden_size:
            for event in pass_record["module_outputs"]:
                if event["module"].endswith(".attn.c_attn") and event["outputs"]:
                    shape = event["outputs"][0]["shape"]
                    if shape and shape[-1] == 3 * hidden_size:
                        event["logical_qkv_layout"] = {
                            "source": "fused c_attn projection output",
                            "split_dimension": -1,
                            "split_sizes": [hidden_size] * 3,
                            "projections": ["query", "key", "value"],
                            "per_projection_shape": shape[:-1] + [hidden_size],
                            "note": "Logical contiguous slices; the source projection is fused.",
                        }
        graph_dict = graph.to_dict() if graph is not None else None
        return pass_record, logits, graph_dict

    def trace(self, text: str, output_dir: str | Path,
              max_new_tokens: int = 0) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._remove_hooks()
        self.model.eval()
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        input_token_ids = input_ids[0].detach().cpu().tolist()
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_token_ids)
        passes: dict[str, Any] = {}
        computational_graphs: dict[str, Any] = {}
        generation_steps: list[dict[str, Any]] = []
        try:
            self._install_activation_hooks()
            prompt_record, prompt_logits, prompt_graph = self._run_pass(
                "prompt_prefill", input_ids, capture_attention=True, capture_ops=True,
            )
            passes["prompt_prefill"] = prompt_record
            if prompt_graph is not None:
                computational_graphs["prompt_prefill"] = prompt_graph

            # Compare instrumented prefill logits with a direct uninstrumented pass.
            with torch.inference_mode():
                direct = self.model(
                    input_ids=input_ids, output_attentions=False,
                    output_hidden_states=False, use_cache=False, return_dict=True,
                ).logits
            max_abs_diff = float((prompt_logits.float() - direct.float()).abs().max().item())
            parity = {
                "max_absolute_logit_difference": max_abs_diff,
                "allclose": bool(torch.allclose(
                    prompt_logits.float(), direct.float(), rtol=1e-4, atol=1e-5
                )),
                "rtol": 1e-4, "atol": 1e-5,
            }

            context = input_ids.clone()
            for step in range(max_new_tokens):
                if step == 0:
                    step_logits = prompt_logits[0, -1].float()
                    source_pass = "prompt_prefill"
                else:
                    pass_id = f"generation_step_{step}"
                    generation_record, step_output, step_graph = self._run_pass(
                        pass_id, context,
                        capture_attention=self.capture_generation_attention,
                        capture_ops=self.trace_generation_ops,
                    )
                    passes[pass_id] = generation_record
                    if step_graph is not None:
                        computational_graphs[pass_id] = step_graph
                    step_logits = step_output[0, -1].float()
                    source_pass = pass_id
                probabilities = torch.softmax(step_logits, dim=-1)
                selected_id = int(torch.argmax(probabilities).item())
                selected_token = self.tokenizer.decode([selected_id])
                generation_steps.append({
                    "step": step, "source_pass": source_pass,
                    "context_length": int(context.shape[1]),
                    "context_token_ids": context[0].detach().cpu().tolist(),
                    "selected_token_id": selected_id,
                    "selected_token": selected_token,
                    "selected_probability": float(probabilities[selected_id].item()),
                })
                context = torch.cat([
                    context, torch.tensor([[selected_id]], dtype=input_ids.dtype, device=device)
                ], dim=1)
            try:
                generated_text = self.tokenizer.decode(context[0], skip_special_tokens=True)
            except TypeError:
                generated_text = self.tokenizer.decode(context[0])
        finally:
            self._active_pass = None
            self._remove_hooks()

        model_map = inspect_gpt2(self.model)
        structural_graph = build_dependency_graph(model_map)
        experiment_id = datetime.now(timezone.utc).strftime("gpt2_%Y%m%dT%H%M%S_%fZ")
        structural_path = output_path / "model_graph.json"
        runtime_path = output_path / "runtime_computational_graphs.json"
        structural_path.write_text(json.dumps(structural_graph, indent=2), encoding="utf-8")
        runtime_path.write_text(json.dumps({
            "schema_version": "0.2.0", "passes": computational_graphs,
        }, indent=2), encoding="utf-8")

        prompt = passes["prompt_prefill"]
        attention_modules = {
            event["module"]: event for event in prompt["module_outputs"]
            if event["module"].endswith(".attn")
        }
        mlp_outputs = {
            event["module"]: event for event in prompt["module_outputs"]
            if event["module"].endswith(".mlp")
        }
        report = {
            "schema_version": "0.2.0", "experiment_id": experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {"text": text, "input_ids": input_token_ids,
                      "tokens": input_tokens, "sequence_length": len(input_token_ids)},
            "model": model_map, "structural_graph": structural_graph,
            "runtime_trace": {
                "passes": passes,
                "module_outputs": prompt["module_outputs"],
                "attention_modules": attention_modules, "mlp_outputs": mlp_outputs,
                "hidden_states": prompt["hidden_states"], "attention": prompt["attention"],
                "logits": prompt["logits"],
                "last_position_top_tokens": prompt["last_position_top_tokens"],
                "generation_steps": generation_steps, "generated_text": generated_text,
                "execution_passes": [
                    {"pass_id": pass_id, "sequence_length": record["sequence_length"],
                     "module_event_count": len(record["module_outputs"]),
                     "attention_layer_count": len(record["attention"]),
                     "operator_graph_captured": pass_id in computational_graphs}
                    for pass_id, record in passes.items()
                ],
                "operator_graph_summary": {
                    pass_id: graph["capture"] for pass_id, graph in computational_graphs.items()
                },
                "parity_check": parity,
            },
            "artifacts": {"structural_graph": str(structural_path),
                          "runtime_computational_graphs": str(runtime_path)},
            "metadata": {
                "device": str(device), "torch_version": torch.__version__,
                "python_version": platform.python_version(),
                "attention_note": "Attention is token-position routing information, not semantic or causal explanation.",
                "dependency_note": "The runtime graph records actual ATen tensor operands and outputs for each captured pass; it is not semantic attribution.",
                "generation_capture_policy": {
                    "generation_attention": self.capture_generation_attention,
                    "generation_operator_graphs": self.trace_generation_ops,
                    "prompt_attention": True, "prompt_operator_graph": True,
                },
            },
        }
        report_path = output_path / "trace_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            "report_path": str(report_path),
            "structural_graph_path": str(structural_path),
            "runtime_graph_path": str(runtime_path),
            "experiment_id": experiment_id, "parameter_count": model_map["parameter_count"],
            "module_count": len(model_map["modules"]),
            "attention_layers_captured": len(prompt["attention"]),
            "hidden_state_stages": len(prompt["hidden_states"]),
            "operator_graph_summary": report["runtime_trace"]["operator_graph_summary"],
            "parity_check": parity, "generated_text": generated_text,
        }

def main() -> None:
    parser = argparse.ArgumentParser(description="Brahma MetaCore GPT-2 Neural Connectivity tracer")
    parser.add_argument("--model", default="openai-community/gpt2")
    parser.add_argument("--text", default="What is an apple?")
    parser.add_argument("--output", default="reports/neural_connectivity/gpt2_demo")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-ops", type=int, default=50_000)
    parser.add_argument("--max-activation-values", type=int, default=50_000)
    parser.add_argument("--capture-generation-attention", action="store_true")
    parser.add_argument("--trace-generation-ops", action="store_true")
    args = parser.parse_args()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install Transformers: python -m pip install transformers") from exc
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager"
    ).to(device)
    recorder = GPT2TraceRecorder(
        model, tokenizer, max_ops=args.max_ops,
        max_activation_values=args.max_activation_values,
        capture_generation_attention=args.capture_generation_attention,
        trace_generation_ops=args.trace_generation_ops,
    )
    print(json.dumps(recorder.trace(args.text, args.output, args.max_new_tokens), indent=2))

if __name__ == "__main__":
    main()
