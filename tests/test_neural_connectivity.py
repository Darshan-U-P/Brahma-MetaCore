import json
import torch
import torch.nn as nn
from src.metaobserver.computational_graph import RuntimeTensorGraph
from src.metaobserver.neural_connectivity import GPT2TraceRecorder, build_dependency_graph, inspect_gpt2

class TinyConfig:
    model_type = "tiny-gpt2"
    n_embd = 8
    def to_dict(self):
        return {"model_type": self.model_type, "n_embd": self.n_embd}

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(32, 8)
        self.transformer.wpe = nn.Embedding(16, 8)
        self.transformer.h = nn.ModuleList([nn.ModuleDict({
            "ln_1": nn.LayerNorm(8), "attn": nn.Linear(8, 8),
            "ln_2": nn.LayerNorm(8),
            "mlp": nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)),
        })])
        self.transformer.ln_f = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, input_ids, output_attentions=True, output_hidden_states=True,
                use_cache=False, return_dict=True):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self.transformer.wte(input_ids) + self.transformer.wpe(positions)
        hidden_states = [hidden]
        for block in self.transformer.h:
            residual = hidden
            hidden = block["ln_1"](hidden)
            hidden = residual + block["attn"](hidden)
            residual = hidden
            hidden = block["ln_2"](hidden)
            hidden = residual + block["mlp"](hidden)
            hidden_states.append(hidden)
        hidden = self.transformer.ln_f(hidden)
        class Result: pass
        result = Result()
        result.logits = self.lm_head(hidden)
        result.hidden_states = tuple(hidden_states) if output_hidden_states else None
        result.attentions = None
        return result

class TinyTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    def convert_ids_to_tokens(self, ids):
        return [f"tok_{token_id}" for token_id in ids]
    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"tok{int(token_id)}" for token_id in ids)

def test_inspector_inventories_modules_and_parameters():
    model = TinyModel()
    report = inspect_gpt2(model)
    assert report["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert any(item["name"] == "transformer.wte" for item in report["modules"])
    assert any(item["name"] == "transformer.wte.weight" for item in report["parameters"])

def test_structural_graph_has_module_and_parameter_edges():
    graph = build_dependency_graph(inspect_gpt2(TinyModel()))
    assert graph["nodes"]
    assert any(edge["relation"] == "owns_parameter" for edge in graph["edges"])
    assert any(edge["relation"] == "contains" for edge in graph["edges"])

def test_runtime_graph_tracks_real_tensor_operands_and_outputs():
    model = TinyModel().eval()
    ids = torch.tensor([[1, 2, 3]])
    recorder = RuntimeTensorGraph(model, max_ops=5000)
    recorder.register_input("input_ids", ids)
    recorder.install_module_scopes()
    try:
        with torch.inference_mode(), recorder:
            output = model(ids).logits
            recorder.add_output("logits", output)
    finally:
        recorder.remove_module_scopes()
    graph = recorder.to_dict()
    assert graph["capture"]["operator_count"] > 0
    assert any(node["kind"] == "model_input" for node in graph["nodes"])
    assert any(node["kind"] == "parameter" for node in graph["nodes"])
    assert any(node["kind"] == "operator" for node in graph["nodes"])
    assert any(edge["relation"] == "operand" for edge in graph["edges"])
    assert any(edge["relation"] == "produces" for edge in graph["edges"])
    assert any(node.get("module_scope") == "transformer.wte"
               for node in graph["nodes"] if node["kind"] == "operator")

def test_runtime_graph_reports_operator_limit():
    model = TinyModel().eval()
    ids = torch.tensor([[1, 2, 3]])
    recorder = RuntimeTensorGraph(model, max_ops=1)
    recorder.register_input("input_ids", ids)
    with torch.inference_mode(), recorder:
        model(ids)
    assert recorder.to_dict()["capture"]["truncated"] is True
    assert recorder.to_dict()["capture"]["operator_count"] == 1

def test_trace_separates_prompt_and_generation_passes(tmp_path):
    model = TinyModel().eval()
    hook_counts = {
        name: len(module._forward_hooks) + len(module._forward_pre_hooks)
        for name, module in model.named_modules()
    }
    recorder = GPT2TraceRecorder(model, TinyTokenizer(), max_ops=5000)
    result = recorder.trace("What is an apple?", tmp_path, max_new_tokens=3)
    report = json.loads((tmp_path / "trace_report.json").read_text(encoding="utf-8"))
    passes = report["runtime_trace"]["passes"]
    assert set(passes) == {"prompt_prefill", "generation_step_1", "generation_step_2"}
    assert passes["prompt_prefill"]["sequence_length"] == 3
    assert passes["generation_step_1"]["sequence_length"] == 4
    assert passes["generation_step_2"]["sequence_length"] == 5

    def embedding_shape(pass_record):
        event = next(e for e in pass_record["module_outputs"]
                     if e["module"] == "transformer.wte")
        return event["outputs"][0]["shape"]
    assert embedding_shape(passes["prompt_prefill"]) == [1, 3, 8]
    assert embedding_shape(passes["generation_step_1"]) == [1, 4, 8]
    assert len(report["runtime_trace"]["generation_steps"]) == 3
    assert report["runtime_trace"]["generation_steps"][0]["source_pass"] == "prompt_prefill"
    assert report["runtime_trace"]["parity_check"]["allclose"] is True
    assert result["parity_check"]["allclose"] is True
    assert (tmp_path / "runtime_computational_graphs.json").exists()
    assert result["operator_graph_summary"]["prompt_prefill"]["operator_count"] > 0
    for name, module in model.named_modules():
        assert len(module._forward_hooks) + len(module._forward_pre_hooks) == hook_counts[name]
