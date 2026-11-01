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


class SharedLayerNormBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        self.normalization = nn.LayerNorm(4)

    def forward(self, values):
        normalized = self.normalization(self.projection(values))
        return normalized[..., 0] + normalized[..., 1]


def test_deduplicated_shared_group_preserves_a_valid_representative_path():
    graph, input_id = capture(SharedLayerNormBranches(), torch.randn(1, 4))
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 2],
        "model_output:logits", [0],
        max_depth=32, max_nodes=2000, max_candidates=500,
        search_strategy="breadth_first",
    )

    assert result["result"]["status"] == "path_found"
    assert result["result"]["path_found"] is True
    assert result["path"]
    assert result["path"][0]["source_coordinate_node_id"].startswith(
        "coord::" + input_id + "::["
    )
    assert result["path"][-1]["target_coordinate_node_id"].startswith(
        "coord::model_output:logits::["
    )
    assert result["summary"]["activation_dependency_groups_deduplicated"] >= 1


def test_source_guided_search_prefers_shorter_source_route_under_node_budget():
    class LongAndShortBranches(nn.Module):
        def forward(self, values):
            long_branch = values.clone().clone().clone()
            short_branch = values.clone()
            return long_branch + short_branch

    graph, input_id = capture(LongAndShortBranches(), torch.randn(1, 4))
    guided = trace_focused_coordinate_path(
        graph,
        input_id, [0, 2],
        "model_output:logits", [0, 2],
        max_depth=16, max_nodes=3, max_candidates=10,
        search_strategy="source_guided",
    )
    breadth_first = trace_focused_coordinate_path(
        graph,
        input_id, [0, 2],
        "model_output:logits", [0, 2],
        max_depth=16, max_nodes=3, max_candidates=10,
        search_strategy="breadth_first",
    )

    assert guided["result"]["path_found"] is True
    assert guided["result"]["status"] == "path_found"
    assert guided["summary"]["resource_limit_reached"] is True
    assert breadth_first["result"]["path_found"] is False
    assert breadth_first["result"]["status"] == "inconclusive_limit"


def test_source_guided_and_breadth_first_strategies_both_find_linear_path():
    graph, input_id = capture(LinearOnly(), torch.randn(1, 4))
    for strategy in ("source_guided", "breadth_first"):
        result = trace_focused_coordinate_path(
            graph,
            input_id, [0, 2],
            "model_output:logits", [0, 1],
            max_depth=8, max_nodes=100, max_candidates=100,
            search_strategy=strategy,
        )
        assert result["result"]["path_found"] is True
        assert result["summary"]["search_strategy"] == strategy


def test_shared_layer_norm_activation_group_is_expanded_once():
    ids = torch.tensor([[3, 9]])
    graph, input_id = capture(EmbeddingLayerNormProjection(), ids)
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 0],
        "model_output:logits", [0, 0, 0],
        tensor_values={input_id: [[3, 9]]},
        max_depth=32, max_nodes=2000, max_candidates=100,
        search_strategy="breadth_first",
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


def test_activation_source_coordinate_matches_compact_linear_group_without_enumeration():
    graph, input_id = capture(LinearOnly(), torch.randn(1, 4))
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 2],
        "model_output:logits", [0, 1],
        max_depth=8, max_nodes=100, max_candidates=100,
    )

    assert result["result"]["status"] == "path_found"
    assert result["path"][0]["source_coordinate_node_id"].startswith(
        "coord::" + input_id + "::["
    )
    assert result["summary"]["candidate_dependencies_examined"] == 1
    assert result["summary"]["activation_dependency_groups_expanded"] == 0
    assert result["summary"]["coordinate_nodes_scheduled"] >= 1
    assert result["summary"]["coordinate_nodes_processed"] >= 1


def test_source_reachability_prunes_unrelated_model_input_branch():
    class TwoInputResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 3)

        def forward(self, values, side):
            return self.projection(values) + side

    model = TwoInputResidual().eval()
    values = torch.randn(1, 4)
    side = torch.randn(1, 3)
    recorder = RuntimeTensorGraph(model, max_ops=2000)
    source_id = recorder.register_input("values", values)
    side_id = recorder.register_input("side", side)
    with torch.inference_mode(), recorder:
        output = model(values, side)
        recorder.add_output("logits", output)
    graph = recorder.to_dict()

    result = trace_focused_coordinate_path(
        graph,
        source_id, [0, 2],
        "model_output:logits", [0, 1],
        max_depth=16, max_nodes=100, max_candidates=100,
    )

    assert result["result"]["status"] == "path_found"
    assert side_id not in {
        node["tensor_id"] for node in result["nodes"]
    }
    assert result["summary"]["source_reachable_tensor_count"] >= 2


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


def test_node_budget_drains_already_scheduled_coordinates_before_stopping():
    class TwoBranchClone(nn.Module):
        def forward(self, values):
            return values.clone() + values.clone()

    graph, input_id = capture(TwoBranchClone(), torch.randn(1, 4))
    result = trace_focused_coordinate_path(
        graph,
        input_id, [0, 2],
        "model_output:logits", [0, 2],
        max_depth=8, max_nodes=3, max_candidates=10,
    )

    # The add output schedules the first clone branch, then reaches the node
    # limit on the second. The already-queued clone must still be expanded.
    assert result["result"]["path_found"] is True, result
    assert result["result"]["status"] == "path_found"
    assert result["summary"]["resource_limit_reached"] is True
    assert result["summary"]["coordinate_nodes_processed"] >= 2
    assert result["summary"]["queue_remaining_at_stop"] == 0
    assert result["result"]["search_complete"] is False
