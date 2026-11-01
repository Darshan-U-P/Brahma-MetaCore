import torch
import torch.nn as nn
import pytest

from src.metaobserver.computational_graph import RuntimeTensorGraph
from src.metaobserver.coordinate_tracer import (
    CoordinateTraceError,
    trace_coordinate_graph,
    trace_tensor_coordinate,
)


def capture_graph(model, input_ids, output_name="output"):
    model.eval()
    recorder = RuntimeTensorGraph(model, max_ops=2000)
    input_node_id = recorder.register_input("input_ids", input_ids)
    with torch.inference_mode(), recorder:
        output = model(input_ids)
        recorder.add_output(output_name, output)
    return recorder.to_dict(), input_node_id


def output_source(graph, output_name="output"):
    edge = next(
        edge for edge in graph["edges"]
        if edge["target"] == f"model_output:{output_name}"
        and edge["relation"] == "exposed_as_model_output"
    )
    return edge["source"]


class EmbeddingOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids):
        return self.embedding(input_ids)


def test_embedding_coordinate_resolves_selected_table_row_and_index():
    ids = torch.tensor([[4, 7]])
    graph, input_node_id = capture_graph(EmbeddingOnly(), ids, "embeddings")
    tensor_values = {input_node_id: [[4, 7]]}
    embedding_output_id = output_source(graph, "embeddings")

    result = trace_tensor_coordinate(
        graph, embedding_output_id, [0, 1, 2], tensor_values=tensor_values
    )

    assert result["status"] == "expanded"
    index_dependency = next(
        item for item in result["dependencies"] if item["relation"] == "lookup_index"
    )
    row_dependency = next(
        item for item in result["dependencies"]
        if item["relation"] == "selected_embedding_row"
    )
    assert index_dependency["tensor_id"] == input_node_id
    assert index_dependency["coordinate"] == [0, 1]
    weight_node = next(
        node for node in graph["nodes"]
        if node["id"] == row_dependency["tensor_id"]
    )
    assert weight_node["kind"] == "parameter"
    assert row_dependency["coordinate"] == [7, 2]


class AddBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.arange(4, dtype=torch.float32))

    def forward(self, values):
        return values + self.bias


def test_broadcast_add_maps_output_coordinate_to_both_operands():
    values = torch.randn(1, 2, 4)
    graph, input_node_id = capture_graph(AddBias(), values)
    output_id = output_source(graph)

    result = trace_tensor_coordinate(graph, output_id, [0, 1, 2])
    dependencies = result["dependencies"]

    activation = next(item for item in dependencies
                      if item["tensor_id"] == input_node_id)
    bias = next(
        item for item in dependencies
        if next(node for node in graph["nodes"] if node["id"] == item["tensor_id"])["label"] == "bias"
    )
    assert activation["coordinate"] == [0, 1, 2]
    assert bias["coordinate"] == [2]


class LinearOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def forward(self, values):
        return self.projection(values)


def test_linear_or_addmm_coordinate_expands_input_and_weight_terms():
    values = torch.randn(1, 4)
    graph, input_node_id = capture_graph(LinearOnly(), values)
    output_id = output_source(graph)

    alias = trace_tensor_coordinate(graph, "model_output:output", [0, 1])
    assert alias["dependencies"][0]["tensor_id"] == output_id
    result = trace_tensor_coordinate(graph, output_id, [0, 1])
    assert result["status"] == "expanded"

    groups = result.get("dependency_groups", [])
    dependencies = result["dependencies"]
    input_groups = [
        item for item in groups
        if item["relation"] in {"linear_input", "matmul_left"}
        and item["tensor_id"] == input_node_id
    ]
    weight_groups = [
        item for item in groups
        if item["relation"] in {"linear_weight", "matmul_right"}
    ]
    if input_groups:
        assert len(input_groups) == 1
        assert input_groups[0]["dependency_count"] == 4
        assert input_groups[0]["coordinate_prefix"] == [0, 0]
        assert input_groups[0]["varying_axes"] == [1]
        assert len(weight_groups) == 1
    else:
        input_dependencies = [
            item for item in dependencies if item["tensor_id"] == input_node_id
        ]
        assert len(input_dependencies) == 4
        assert {tuple(item["coordinate"]) for item in input_dependencies} == {
            (0, 0), (0, 1), (0, 2), (0, 3)
        }
        assert any(
            item["relation"] in {"linear_weight", "matmul_right"}
            for item in dependencies
        )


class EmbeddingMeanProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 3)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        return self.projection(pooled)


def test_recursive_coordinate_graph_connects_token_index_to_selected_logit():
    ids = torch.tensor([[3, 9]])
    graph, input_node_id = capture_graph(EmbeddingMeanProjection(), ids, "logits")
    result = trace_coordinate_graph(
        graph,
        "model_output:logits",
        [0, 1],
        tensor_values={input_node_id: [[3, 9]]},
        focus_source_tensor_id=input_node_id,
        focus_source_coordinate=[0, 1],
        max_depth=20,
        max_nodes=2000,
        max_edges=5000,
    )

    assert result["query"]["focus_source_reaches_target"] is True
    assert any(
        item["tensor_id"] == input_node_id and item["coordinate"] == [0, 1]
        for item in result["input_coordinates"]
    )
    assert any(
        node["tensor_kind"] == "parameter"
        and node["tensor_label"] == "embedding.weight"
        and node["coordinate"] == [9, 0]
        for node in result["parameter_coordinates"]
    )
    assert result["summary"]["truncated"] is False


def test_coordinate_shape_validation_rejects_invalid_rank_and_index():
    ids = torch.tensor([[4, 7]])
    graph, _ = capture_graph(EmbeddingOnly(), ids, "embeddings")
    output_id = output_source(graph, "embeddings")

    with pytest.raises(CoordinateTraceError):
        trace_tensor_coordinate(graph, output_id, [0, 1])
    with pytest.raises(CoordinateTraceError):
        trace_tensor_coordinate(graph, output_id, [0, 2, 0])


class NormalizeOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)

    def forward(self, values):
        return self.norm(values)


def test_layer_norm_uses_compact_dependency_group_and_expands_all_input_coordinates():
    values = torch.randn(1, 4)
    graph, input_node_id = capture_graph(NormalizeOnly(), values)
    output_id = output_source(graph)

    immediate = trace_tensor_coordinate(graph, output_id, [0, 2])
    assert immediate["status"] == "expanded"
    assert len(immediate["dependency_groups"]) == 1
    group = immediate["dependency_groups"][0]
    assert group["tensor_id"] == input_node_id
    assert group["coordinate_prefix"] == [0, 0]
    assert group["varying_axes"] == [1]
    assert group["axis_sizes"] == [4]
    assert group["dependency_count"] == 4

    recursive = trace_coordinate_graph(
        graph,
        "model_output:output",
        [0, 2],
        focus_source_tensor_id=input_node_id,
        focus_source_coordinate=[0, 3],
        max_depth=8,
        max_nodes=100,
        max_edges=100,
    )
    assert recursive["query"]["focus_source_reaches_target"] is True
    assert len(recursive["input_coordinates"]) == 4
    assert recursive["summary"]["dependency_group_count"] >= 1
    assert recursive["summary"]["truncated"] is False



class NormalizeThenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.projection = nn.Linear(4, 3)

    def forward(self, values):
        return self.projection(self.norm(values))


def test_recursive_trace_deduplicates_reused_layer_norm_groups_under_node_cap():
    values = torch.randn(1, 4)
    graph, input_node_id = capture_graph(NormalizeThenLinear(), values)

    result = trace_coordinate_graph(
        graph,
        "model_output:output",
        [0, 1],
        focus_source_tensor_id=input_node_id,
        focus_source_coordinate=[0, 3],
        max_depth=16,
        max_nodes=32,
        max_edges=100,
    )

    assert result["query"]["focus_source_reaches_target"] is True
    assert any(
        item["tensor_id"] == input_node_id and item["coordinate"] == [0, 3]
        for item in result["input_coordinates"]
    )
    assert result["summary"]["truncated"] is False
    assert not any(item["reason"] == "max_nodes" for item in result["frontier"])
