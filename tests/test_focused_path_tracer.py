import torch
import torch.nn as nn
import pytest

from src.metaobserver.computational_graph import RuntimeTensorGraph
from src.metaobserver.coordinate_tracer import CoordinateTraceError
from src.metaobserver.focused_path_tracer import trace_focused_coordinate_path


def capture(model, input_tensor):
    model.eval()
    recorder = RuntimeTensorGraph(model, max_ops=2000)
    input_id = recorder.register_input("input_ids", input_tensor)
    with torch.inference_mode(), recorder:
        output = model(input_tensor)
        recorder.add_output("logits", output)
    return recorder.to_dict(), input_id


class EmbeddingMeanProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 3)

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids).mean(dim=1))


def test_focused_path_finds_known_token_to_logit_dependency():
    ids = torch.tensor([[3, 9]])
    graph, input_id = capture(EmbeddingMeanProjection(), ids)
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 1],
        "model_output:logits", [0, 1],
        tensor_values={input_id: [[3, 9]]},
        max_depth=32, max_nodes=2000, max_candidates=5000,
    )

    assert result["result"]["status"] == "path_found"
    assert result["result"]["path_found"] is True
    assert result["path"]
    assert result["path"][0]["source_coordinate_node_id"].startswith("coord::" + input_id + "::[")
    assert result["path"][-1]["target_coordinate_node_id"].startswith("coord::model_output:logits::[")
    assert any(edge["relation"] == "lookup_index" for edge in result["path"])
    assert result["summary"]["parameter_groups_retained"] > 0


class EmbeddingLayerNormProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.normalization = nn.LayerNorm(4)
        self.projection = nn.Linear(4, 3)

    def forward(self, input_ids):
        return self.projection(self.normalization(self.embedding(input_ids)))


def test_shared_layer_norm_activation_group_is_expanded_once():
    ids = torch.tensor([[3, 9]])
    graph, input_id = capture(EmbeddingLayerNormProjection(), ids)
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 0],
        "model_output:logits", [0, 0],
        tensor_values={input_id: [[3, 9]]},
        max_depth=32, max_nodes=2000, max_candidates=100,
    )

    assert result["result"]["status"] == "path_found"
    assert result["result"]["path_found"] is True
    assert any(edge["operator"].startswith(
        ("aten::layer_norm.", "aten::native_layer_norm.")
    ) for edge in result["path"])
    assert result["summary"]["activation_dependency_groups_deduplicated"] >= 1
    assert result["summary"]["activation_dependency_groups_expanded"] >= 1
    assert result["summary"]["candidate_dependencies_examined"] < 100


class LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, values):
        return self.projection(values)


def test_parameter_coordinate_can_be_requested_as_source():
    graph, _ = capture(LinearOnly(), torch.randn(1, 4))
    weight = next(
        node for node in graph["nodes"]
        if node.get("kind") == "parameter" and node.get("label") == "projection.weight"
    )
    result = trace_focused_coordinate_path(
        graph,
        weight["id"], [1, 2],
        "model_output:logits", [0, 1],
        max_depth=8, max_nodes=100, max_candidates=100,
    )
    assert result["result"]["status"] == "path_found"
    assert result["path"][0]["source_coordinate_node_id"].startswith("coord::" + weight["id"] + "::[")


class UnsupportedSin(nn.Module):
    def forward(self, values):
        return torch.sin(values)


def test_unsupported_operator_is_reported_as_inconclusive():
    graph, input_id = capture(UnsupportedSin(), torch.randn(1, 2))
    result = trace_focused_coordinate_path(
        graph, input_id, [0, 0],
        "model_output:logits", [0, 0],
        max_depth=8, max_nodes=100, max_candidates=100,
    )
    assert result["result"]["status"] == "inconclusive_unsupported"
    assert not result["result"]["path_found"]
    assert any(item["reason"] == "unsupported_operator" for item in result["frontier"])


def test_node_budget_reports_inconclusive_limit():
    graph, input_id = capture(LinearOnly(), torch.randn(1, 4))
    result = trace_focused_coordinate_path(
        graph, input_id, [0, 0],
        "model_output:logits", [0, 1],
        max_depth=8, max_nodes=1, max_candidates=100,
    )
    assert result["result"]["status"] == "inconclusive_limit"
    assert result["summary"]["resource_limit_reached"]


def test_invalid_source_coordinate_is_rejected():
    graph, input_id = capture(LinearOnly(), torch.randn(1, 4))
    with pytest.raises(CoordinateTraceError):
        trace_focused_coordinate_path(
            graph, input_id, [0, 4],
            "model_output:logits", [0, 1],
        )


def test_same_coordinate_source_and_target_returns_empty_path():
    graph, input_id = capture(LinearOnly(), torch.randn(1, 4))
    result = trace_focused_coordinate_path(
        graph, input_id, [0, 1], input_id, [0, 1],
    )
    assert result["result"]["status"] == "path_found"
    assert result["path"] == []
