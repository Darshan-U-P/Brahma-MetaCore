import pytest
import torch
import torch.nn as nn
from src.metaobserver.computational_graph import RuntimeTensorGraph
from src.metaobserver.path_tracer import PathTraceError, trace_tensor_path

class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 3)
    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        pooled = hidden.mean(dim=1)
        return self.projection(pooled)

def capture_graph():
    model = MiniModel().eval()
    ids = torch.tensor([[1, 2, 3]])
    recorder = RuntimeTensorGraph(model, max_ops=500)
    recorder.register_input("input_ids", ids)
    with torch.inference_mode(), recorder:
        logits = model(ids)
        recorder.add_output("logits", logits)
    return recorder.to_dict()

def test_path_query_finds_input_to_logits_and_parameter_dependencies():
    report = trace_tensor_path(
        capture_graph(), source="input_ids", target="model_output:logits",
        input_coordinate={"batch": 0, "sequence_position": 1},
        output_coordinate={"batch": 0, "sequence_position": 0, "vocabulary_token_id": 2},
    )
    assert report["summary"]["reachable"] is True
    assert report["summary"]["graph_is_acyclic"] is True
    assert report["summary"]["source_to_target_route_node_count"] > 2
    assert report["summary"]["upstream_parameter_count"] == 3
    assert report["shortest_route"][0]["label"] == "input_ids"
    assert report["shortest_route"][-1]["label"] == "logits"
    assert report["query"]["input_coordinate"]["sequence_position"] == 1
    assert len(report["source_to_target_subgraph"]["edges"]) > 0
    assert len(report["complete_target_upstream_subgraph"]["nodes"]) >= len(
        report["source_to_target_subgraph"]["nodes"]
    )

def test_path_query_rejects_unknown_or_disconnected_source():
    graph = capture_graph()
    with pytest.raises(PathTraceError):
        trace_tensor_path(graph, source="missing", target="model_output:logits")
    with pytest.raises(PathTraceError):
        trace_tensor_path(graph, source="parameter:embedding.weight", target="model_input")

def test_path_query_rejects_wrong_graph_type():
    with pytest.raises(PathTraceError):
        trace_tensor_path({"graph_type": "structural_module_parameter_graph",
                           "nodes": [], "edges": []})
