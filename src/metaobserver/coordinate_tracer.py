"""Coordinate-aware dependency tracing for captured ATen execution graphs.

The tracer expands one selected tensor scalar at a time. Supported operator rules
are explicit and conservative: unsupported or value-dependent operations become
reported frontiers rather than guessed connections.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from pathlib import Path
from typing import Any


class CoordinateTraceError(ValueError):
    """Raised when a tensor coordinate or trace query is invalid."""


def _node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph.get("nodes", [])
    indexed = {node["id"]: node for node in nodes}
    if len(indexed) != len(nodes):
        raise CoordinateTraceError("Graph contains duplicate node IDs.")
    return indexed


def _validate_coordinate(node: dict[str, Any], coordinate: tuple[int, ...]) -> None:
    shape = node.get("shape")
    if shape is None:
        raise CoordinateTraceError(
            f"Node {node.get('id')} has no tensor shape and cannot be coordinate-traced."
        )
    if len(coordinate) != len(shape):
        raise CoordinateTraceError(
            f"Coordinate {coordinate} has rank {len(coordinate)}; "
            f"tensor {node.get('id')} has shape {shape}."
        )
    for axis, (index, size) in enumerate(zip(coordinate, shape)):
        if not isinstance(index, int) or index < 0 or index >= size:
            raise CoordinateTraceError(
                f"Coordinate index {index!r} is outside axis {axis} of shape {shape}."
            )


def _argument(op: dict[str, Any], name: str, default: Any = None) -> Any:
    return op.get("argument_metadata", {}).get(name, default)


def _integer(value: Any, default: int | None = None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _operand_index(op: dict[str, Any], name: str, default: int | None = None) -> int | None:
    value = _argument(op, name)
    if isinstance(value, dict):
        return _integer(value.get("tensor_operand_index"), default)
    return default


def _broadcast_coordinate(output_coordinate: tuple[int, ...],
                          input_shape: list[int]) -> tuple[int, ...]:
    if len(input_shape) > len(output_coordinate):
        raise CoordinateTraceError("Broadcast input rank exceeds output rank.")
    padded = [1] * (len(output_coordinate) - len(input_shape)) + list(input_shape)
    mapped = []
    for out_index, size in zip(output_coordinate, padded):
        if size == 1:
            mapped.append(0)
        elif size > 0 and out_index < size:
            mapped.append(out_index)
        else:
            raise CoordinateTraceError(
                f"Cannot broadcast coordinate {output_coordinate} to shape {input_shape}."
            )
    return tuple(mapped[len(padded) - len(input_shape):])


def _value_at(values: Any, coordinate: tuple[int, ...]) -> Any:
    current = values
    for index in coordinate:
        current = current[index]
    return current


def _coordinate_from_offset(shape: list[int], strides: list[int],
                            offset: int) -> tuple[int, ...] | None:
    if len(shape) != len(strides):
        return None
    coordinate = [0] * len(shape)
    remainder = offset
    # A non-overlapping strided view has a unique coordinate for an offset.
    for axis in sorted(range(len(shape)), key=lambda item: strides[item], reverse=True):
        stride = strides[axis]
        if stride == 0:
            if shape[axis] != 1:
                return None
            coordinate[axis] = 0
        else:
            index, remainder = divmod(remainder, stride)
            if index >= shape[axis]:
                return None
            coordinate[axis] = index
    if remainder:
        return None
    return tuple(coordinate)


def _cartesian_coordinates(prefix: tuple[int, ...],
                           axis_sizes: list[int]) -> list[tuple[int, ...]]:
    if not axis_sizes:
        return [prefix]
    return [
        prefix + tuple(suffix)
        for suffix in itertools.product(*(range(size) for size in axis_sizes))
    ]


def trace_tensor_coordinate(
    graph: dict[str, Any],
    tensor_id: str,
    coordinate: list[int] | tuple[int, ...],
    tensor_values: dict[str, Any] | None = None,
    max_fan_in: int = 4096,
) -> dict[str, Any]:
    """Resolve the immediate scalar dependencies of one tensor coordinate.

    Returns dependencies as (operand tensor, coordinate) pairs. This operation-
    local API is useful for inspecting one selected neuron/logit without
    recursively materializing a potentially enormous scalar graph.
    """
    if graph.get("graph_type") != "runtime_tensor_dependency_graph":
        raise CoordinateTraceError("Input graph is not a runtime tensor dependency graph.")
    if max_fan_in < 1:
        raise CoordinateTraceError("max_fan_in must be at least 1.")
    nodes = _node_index(graph)
    if tensor_id not in nodes:
        raise CoordinateTraceError(f"Unknown tensor node {tensor_id!r}.")
    current = nodes[tensor_id]
    coord = tuple(coordinate)
    _validate_coordinate(current, coord)
    tensor_values = tensor_values or {}

    def terminal(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "tensor_id": tensor_id,
            "coordinate": list(coord),
            "dependencies": [],
            "operator": None,
            "module_scope": None,
        }

    if current.get("kind") in {"model_input", "parameter", "external_tensor"}:
        return terminal("terminal")

    edges = graph.get("edges", [])
    if current.get("kind") == "model_output":
        alias = next(
            (edge for edge in edges
             if edge.get("target") == tensor_id
             and edge.get("relation") == "exposed_as_model_output"),
            None,
        )
        if alias is None:
            return terminal("missing_output_alias")
        source = nodes[alias["source"]]
        _validate_coordinate(source, coord)
        return {
            "status": "expanded",
            "tensor_id": tensor_id,
            "coordinate": list(coord),
            "dependencies": [{
                "tensor_id": source["id"],
                "coordinate": list(coord),
                "relation": "output_alias",
                "operand_index": None,
            }],
            "operator": "model_output_alias",
            "module_scope": None,
        }

    producer_edge = next(
        (edge for edge in edges
         if edge.get("target") == tensor_id and edge.get("relation") == "produces"),
        None,
    )
    if producer_edge is None:
        return terminal("missing_producer")
    op = nodes[producer_edge["source"]]
    operand_edges = sorted(
        (edge for edge in edges
         if edge.get("target") == op["id"] and edge.get("relation") == "operand"),
        key=lambda edge: edge.get("operand_index", -1),
    )
    operands = {
        edge.get("operand_index"): nodes[edge["source"]]
        for edge in operand_edges
    }
    op_name = op.get("operator", "")
    output_index = producer_edge.get("output_index", 0)

    def dependency(index: int | None, mapped: tuple[int, ...] | list[int],
                   relation: str = "value_dependency") -> dict[str, Any]:
        if index is None or index not in operands:
            raise CoordinateTraceError(
                f"{op_name} requires tensor operand {index!r}, but it was not captured."
            )
        operand = operands[index]
        mapped_tuple = tuple(mapped)
        _validate_coordinate(operand, mapped_tuple)
        return {
            "tensor_id": operand["id"],
            "coordinate": list(mapped_tuple),
            "relation": relation,
            "operand_index": index,
        }

    def finish(dependencies: list[dict[str, Any]]) -> dict[str, Any]:
        if len(dependencies) > max_fan_in:
            return {
                "status": "fan_in_limit",
                "tensor_id": tensor_id,
                "coordinate": list(coord),
                "dependencies": [],
                "operator": op_name,
                "module_scope": op.get("module_scope"),
                "required_fan_in": len(dependencies),
                "max_fan_in": max_fan_in,
            }
        return {
            "status": "expanded",
            "tensor_id": tensor_id,
            "coordinate": list(coord),
            "dependencies": dependencies,
            "operator": op_name,
            "module_scope": op.get("module_scope"),
        }

    if output_index != 0:
        return {
            "status": "unsupported_output_index",
            "tensor_id": tensor_id,
            "coordinate": list(coord),
            "dependencies": [],
            "operator": op_name,
            "module_scope": op.get("module_scope"),
        }

    # Elementwise unary operations preserve coordinates.
    unary_prefixes = (
        "aten::relu.", "aten::gelu.", "aten::silu.", "aten::sigmoid.",
        "aten::tanh.", "aten::dropout.", "aten::contiguous.", "aten::clone.",
        "aten::detach.", "aten::neg.", "aten::to.", "aten::alias.",
        "aten::relu_", "aten::gelu_",
    )
    if op_name.startswith(unary_prefixes):
        return finish([dependency(_operand_index(op, "self", 0), coord)])

    # Broadcasting elementwise operations.
    if op_name.startswith((
        "aten::add.", "aten::sub.", "aten::mul.", "aten::div.",
        "aten::maximum.", "aten::minimum.", "aten::where.",
    )):
        indices = []
        for name, fallback in (("self", 0), ("other", 1)):
            index = _operand_index(op, name, fallback)
            if index is not None and index in operands:
                operand = operands[index]
                mapped = _broadcast_coordinate(coord, operand["shape"])
                indices.append(dependency(index, mapped))
        if op_name.startswith("aten::where."):
            cond_index = _operand_index(op, "condition", 0)
            if cond_index in operands and all(
                item["operand_index"] != cond_index for item in indices
            ):
                indices.insert(0, dependency(
                    cond_index,
                    _broadcast_coordinate(coord, operands[cond_index]["shape"]),
                    "selection_condition",
                ))
        return finish(indices)

    # Embedding lookup: the index chooses a parameter row; it is not a numeric
    # additive contribution. The selected row is resolved from captured input IDs.
    if op_name.startswith("aten::embedding."):
        weight_index = _operand_index(op, "weight", 0)
        indices_index = _operand_index(op, "indices", 1)
        embedding_coordinate = coord[-1:]
        indices_coordinate = coord[:-1]
        if weight_index not in operands or indices_index not in operands:
            return {
                "status": "missing_embedding_operands",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        index_tensor = operands[indices_index]
        index_values = tensor_values.get(index_tensor["id"])
        dependencies = [dependency(indices_index, indices_coordinate, "lookup_index")]
        if index_values is None:
            return {
                "status": "value_required",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": dependencies, "operator": op_name,
                "module_scope": op.get("module_scope"),
                "required_value": {
                    "tensor_id": index_tensor["id"],
                    "coordinate": list(indices_coordinate),
                    "purpose": "resolve selected embedding table row",
                },
            }
        selected_id = int(_value_at(index_values, indices_coordinate))
        dependencies.append(dependency(
            weight_index, (selected_id, embedding_coordinate[0]), "selected_embedding_row"
        ))
        return finish(dependencies)

    # Linear layer: y[..., out] = x[..., k] * W[out, k] + b[out].
    if op_name.startswith("aten::linear."):
        input_index = _operand_index(op, "input", 0)
        weight_index = _operand_index(op, "weight", 1)
        bias_index = _operand_index(op, "bias", 2)
        if input_index not in operands or weight_index not in operands:
            raise CoordinateTraceError("Linear operator is missing input or weight.")
        input_node, weight_node = operands[input_index], operands[weight_index]
        if len(coord) != len(input_node["shape"]) or len(weight_node["shape"]) != 2:
            raise CoordinateTraceError("Unsupported linear tensor ranks.")
        output_feature = coord[-1]
        input_width = input_node["shape"][-1]
        if weight_node["shape"] != [current["shape"][-1], input_width]:
            raise CoordinateTraceError("Linear weight shape is not [out_features, in_features].")
        prefix = coord[:-1]
        dependencies = []
        for feature in range(input_width):
            dependencies.append(dependency(
                input_index, prefix + (feature,), "linear_input"
            ))
            dependencies.append(dependency(
                weight_index, (output_feature, feature), "linear_weight"
            ))
        if bias_index in operands:
            dependencies.append(dependency(
                bias_index, (output_feature,), "linear_bias"
            ))
        return finish(dependencies)

    # Matrix multiply / batched matrix multiply.
    if op_name.startswith(("aten::matmul.", "aten::bmm.")):
        left_index = _operand_index(op, "self", 0)
        right_index = _operand_index(op, "other", 1)
        if left_index not in operands or right_index not in operands:
            raise CoordinateTraceError("Matmul is missing an operand.")
        left, right = operands[left_index], operands[right_index]
        left_shape, right_shape = left["shape"], right["shape"]
        if len(left_shape) < 2 or len(right_shape) < 2:
            return {
                "status": "unsupported_rank",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        batch_rank = len(coord) - 2
        left_batch_rank, right_batch_rank = len(left_shape) - 2, len(right_shape) - 2
        batch_coordinate = coord[:batch_rank]
        row, column = coord[-2:]
        contracted = left_shape[-1]
        if contracted != right_shape[-2]:
            raise CoordinateTraceError("Matmul contracted dimensions do not match.")
        def batch_map(source_shape: list[int], source_rank: int) -> tuple[int, ...]:
            padded_shape = [1] * (batch_rank - source_rank) + source_shape
            padded_coord = [0] * (batch_rank - source_rank) + list(batch_coordinate)
            mapped = [0 if size == 1 else index
                      for size, index in zip(padded_shape, padded_coord)]
            return tuple(mapped[batch_rank - source_rank:])
        left_prefix = batch_map(left_shape[:-2], left_batch_rank)
        right_prefix = batch_map(right_shape[:-2], right_batch_rank)
        dependencies = []
        for inner in range(contracted):
            dependencies.append(dependency(
                left_index, left_prefix + (row, inner), "matmul_left"
            ))
            dependencies.append(dependency(
                right_index, right_prefix + (inner, column), "matmul_right"
            ))
        return finish(dependencies)

    # addmm: self + mat1 @ mat2.
    if op_name.startswith("aten::addmm."):
        self_index = _operand_index(op, "self", 0)
        mat1_index = _operand_index(op, "mat1", 1)
        mat2_index = _operand_index(op, "mat2", 2)
        if any(index not in operands for index in (self_index, mat1_index, mat2_index)):
            raise CoordinateTraceError("addmm is missing one or more tensor operands.")
        matrix_coordinate = coord
        mat1, mat2 = operands[mat1_index], operands[mat2_index]
        dependencies = [
            dependency(self_index,
                       _broadcast_coordinate(matrix_coordinate, operands[self_index]["shape"]),
                       "addmm_addend")
        ]
        inner_size = mat1["shape"][-1]
        for inner in range(inner_size):
            dependencies.append(dependency(
                mat1_index, (matrix_coordinate[0], inner), "matmul_left"
            ))
            dependencies.append(dependency(
                mat2_index, (inner, matrix_coordinate[1]), "matmul_right"
            ))
        return finish(dependencies)

    # Layer normalization: an output feature depends on the whole normalized
    # slice, plus the matching affine scale and shift when present.
    if op_name.startswith(("aten::layer_norm.", "aten::native_layer_norm.")):
        input_index = _operand_index(op, "input", 0)
        weight_index = _operand_index(op, "weight", 2)
        bias_index = _operand_index(op, "bias", 3)
        normalized_shape = _argument(op, "normalized_shape")
        if not isinstance(normalized_shape, list) or not normalized_shape:
            return {
                "status": "missing_operator_arguments",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        normalized_shape = [int(size) for size in normalized_shape]
        count = 1
        for size in normalized_shape:
            count *= size
        if count + 2 > max_fan_in:
            return {
                "status": "fan_in_limit",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
                "required_fan_in": count + 2, "max_fan_in": max_fan_in,
            }
        prefix = coord[:-len(normalized_shape)]
        dependencies = [
            dependency(input_index, item, "normalization_statistic_or_value")
            for item in _cartesian_coordinates(prefix, normalized_shape)
        ]
        affine_coordinate = coord[-len(normalized_shape):]
        if weight_index in operands:
            dependencies.append(dependency(weight_index, affine_coordinate, "layer_norm_scale"))
        if bias_index in operands:
            dependencies.append(dependency(bias_index, affine_coordinate, "layer_norm_shift"))
        return finish(dependencies)

    # Softmax output at one position depends on the entire normalization axis.
    if op_name.startswith(("aten::softmax.", "aten::_softmax.")):
        input_index = _operand_index(op, "self", 0)
        input_shape = operands[input_index]["shape"]
        axis = _integer(_argument(op, "dim"))
        if axis is None:
            return {
                "status": "missing_operator_arguments",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        axis = axis % len(input_shape)
        dependencies = []
        for value in range(input_shape[axis]):
            mapped = list(coord)
            mapped[axis] = value
            dependencies.append(dependency(input_index, mapped, "softmax_normalizer"))
        return finish(dependencies)

    # Layout-only coordinate transformations.
    if op_name.startswith("aten::transpose."):
        input_index = _operand_index(op, "self", 0)
        axis_a = _integer(_argument(op, "dim0"))
        axis_b = _integer(_argument(op, "dim1"))
        if axis_a is None or axis_b is None:
            return terminal("missing_operator_arguments")
        input_coordinate = list(coord)
        axis_a %= len(input_coordinate)
        axis_b %= len(input_coordinate)
        input_coordinate[axis_a], input_coordinate[axis_b] = (
            input_coordinate[axis_b], input_coordinate[axis_a]
        )
        return finish([dependency(input_index, input_coordinate, "transpose_reindex")])

    if op_name.startswith("aten::permute."):
        input_index = _operand_index(op, "self", 0)
        dims = _argument(op, "dims")
        if not isinstance(dims, list):
            return terminal("missing_operator_arguments")
        dims = [int(dim) % len(dims) for dim in dims]
        input_coordinate = [0] * len(dims)
        for output_axis, input_axis in enumerate(dims):
            input_coordinate[input_axis] = coord[output_axis]
        return finish([dependency(input_index, input_coordinate, "permute_reindex")])

    if op_name.startswith("aten::t."):
        input_index = _operand_index(op, "self", 0)
        shape = operands[input_index]["shape"]
        if len(shape) != 2:
            return {
                "status": "unsupported_rank",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        return finish([dependency(input_index, (coord[1], coord[0]), "transpose_reindex")])

    if op_name.startswith(("aten::view.", "aten::reshape.")):
        input_index = _operand_index(op, "self", 0)
        input_node = operands[input_index]
        output_strides, input_strides = current.get("stride"), input_node.get("stride")
        if output_strides is None or input_strides is None:
            return terminal("missing_stride_metadata")
        offset = sum(index * stride for index, stride in zip(coord, output_strides))
        mapped = _coordinate_from_offset(input_node["shape"], input_strides, offset)
        if mapped is None:
            return {
                "status": "unsupported_stride_mapping",
                "tensor_id": tensor_id, "coordinate": list(coord),
                "dependencies": [], "operator": op_name,
                "module_scope": op.get("module_scope"),
            }
        return finish([dependency(input_index, mapped, "view_reindex")])

    if op_name.startswith("aten::slice."):
        input_index = _operand_index(op, "self", 0)
        axis = _integer(_argument(op, "dim"), 0)
        start = _integer(_argument(op, "start"), 0)
        step = _integer(_argument(op, "step"), 1)
        input_shape = operands[input_index]["shape"]
        axis %= len(input_shape)
        if start is None or step in (None, 0):
            return terminal("missing_operator_arguments")
        if start < 0:
            start += input_shape[axis]
        mapped = list(coord)
        mapped[axis] = start + coord[axis] * step
        return finish([dependency(input_index, mapped, "slice_reindex")])

    if op_name.startswith("aten::select."):
        input_index = _operand_index(op, "self", 0)
        axis = _integer(_argument(op, "dim"))
        selected = _integer(_argument(op, "index"))
        if axis is None or selected is None:
            return terminal("missing_operator_arguments")
        input_shape = operands[input_index]["shape"]
        axis %= len(input_shape)
        if selected < 0:
            selected += input_shape[axis]
        mapped = list(coord)
        mapped.insert(axis, selected)
        return finish([dependency(input_index, mapped, "select_reindex")])

    if op_name.startswith("aten::unsqueeze."):
        input_index = _operand_index(op, "self", 0)
        axis = _integer(_argument(op, "dim"))
        if axis is None:
            return terminal("missing_operator_arguments")
        rank = len(operands[input_index]["shape"])
        axis = axis if axis >= 0 else axis + rank + 1
        mapped = list(coord)
        mapped.pop(axis)
        return finish([dependency(input_index, mapped, "unsqueeze_reindex")])

    if op_name.startswith("aten::squeeze."):
        input_index = _operand_index(op, "self", 0)
        input_shape = operands[input_index]["shape"]
        axis = _integer(_argument(op, "dim"))
        if axis is None:
            mapped = [value for value, size in zip(coord, input_shape) if size != 1]
        else:
            axis %= len(input_shape)
            mapped = list(coord)
            mapped.insert(axis, 0)
        return finish([dependency(input_index, mapped, "squeeze_reindex")])

    if op_name.startswith("aten::expand."):
        input_index = _operand_index(op, "self", 0)
        input_shape = operands[input_index]["shape"]
        padded = [1] * (len(coord) - len(input_shape)) + input_shape
        mapped = [0 if size == 1 else value for size, value in zip(padded, coord)]
        return finish([dependency(input_index, mapped[-len(input_shape):] if input_shape else (), "expand_reindex")])

    # Reductions such as mean/sum depend on all elements in reduced dimensions.
    if op_name.startswith(("aten::mean.", "aten::sum.")):
        input_index = _operand_index(op, "self", 0)
        input_shape = operands[input_index]["shape"]
        dims = _argument(op, "dim")
        keepdim = bool(_argument(op, "keepdim", False))
        if dims is None:
            axes = list(range(len(input_shape)))
        elif isinstance(dims, list):
            axes = sorted({int(axis) % len(input_shape) for axis in dims})
        else:
            axes = [int(dims) % len(input_shape)]
        if keepdim:
            base = list(coord)
        else:
            base = []
            cursor = 0
            for axis in range(len(input_shape)):
                if axis not in axes:
                    base.append(coord[cursor])
                    cursor += 1
                else:
                    base.append(0)
        axis_sizes = [input_shape[axis] for axis in axes]
        dependencies = []
        for reduced_values in itertools.product(*(range(size) for size in axis_sizes)):
            mapped = list(base)
            for axis, value in zip(axes, reduced_values):
                mapped[axis] = value
            dependencies.append(dependency(input_index, mapped, "reduction_input"))
        return finish(dependencies)

    return {
        "status": "unsupported_operator",
        "tensor_id": tensor_id,
        "coordinate": list(coord),
        "dependencies": [],
        "operator": op_name,
        "module_scope": op.get("module_scope"),
    }


def _coordinate_node_id(tensor_id: str, coordinate: tuple[int, ...]) -> str:
    encoded = ",".join(str(value) for value in coordinate)
    return f"coord::{tensor_id}::[{encoded}]"


def trace_coordinate_graph(
    graph: dict[str, Any],
    target_tensor_id: str,
    target_coordinate: list[int] | tuple[int, ...],
    tensor_values: dict[str, Any] | None = None,
    focus_source_tensor_id: str | None = None,
    focus_source_coordinate: list[int] | tuple[int, ...] | None = None,
    max_depth: int = 12,
    max_nodes: int = 20_000,
    max_edges: int = 50_000,
    max_fan_in: int = 4096,
) -> dict[str, Any]:
    """Build a bounded backward scalar-coordinate dependency graph."""
    if max_depth < 0 or max_nodes < 1 or max_edges < 1:
        raise CoordinateTraceError("Trace limits must be non-negative and node/edge limits positive.")
    nodes = _node_index(graph)
    if target_tensor_id not in nodes:
        raise CoordinateTraceError(f"Unknown target tensor {target_tensor_id!r}.")
    target_coord = tuple(target_coordinate)
    _validate_coordinate(nodes[target_tensor_id], target_coord)
    tensor_values = tensor_values or {}

    coordinate_nodes: dict[str, dict[str, Any]] = {}
    coordinate_edges: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    queue = deque([(target_tensor_id, target_coord, 0)])
    truncated = False

    while queue:
        current_tensor_id, current_coord, depth = queue.popleft()
        coordinate_id = _coordinate_node_id(current_tensor_id, current_coord)
        if coordinate_id in coordinate_nodes:
            continue
        tensor_node = nodes[current_tensor_id]
        if len(coordinate_nodes) >= max_nodes:
            truncated = True
            frontier.append({
                "coordinate_node_id": coordinate_id,
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coord),
                "reason": "max_nodes",
            })
            continue
        coordinate_nodes[coordinate_id] = {
            "id": coordinate_id,
            "tensor_id": current_tensor_id,
            "tensor_label": tensor_node.get("label", current_tensor_id),
            "tensor_kind": tensor_node.get("kind"),
            "coordinate": list(current_coord),
            "tensor_shape": tensor_node.get("shape"),
            "depth": depth,
        }

        if tensor_node.get("kind") in {"model_input", "parameter", "external_tensor"}:
            frontier.append({
                "coordinate_node_id": coordinate_id,
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coord),
                "reason": tensor_node.get("kind"),
            })
            continue
        if depth >= max_depth:
            frontier.append({
                "coordinate_node_id": coordinate_id,
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coord),
                "reason": "max_depth",
            })
            continue

        expansion = trace_tensor_coordinate(
            graph, current_tensor_id, current_coord,
            tensor_values=tensor_values, max_fan_in=max_fan_in,
        )
        if expansion["status"] != "expanded":
            frontier.append({
                "coordinate_node_id": coordinate_id,
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coord),
                "reason": expansion["status"],
                "operator": expansion.get("operator"),
                "module_scope": expansion.get("module_scope"),
                "required_fan_in": expansion.get("required_fan_in"),
            })
        for item in expansion.get("dependencies", []):
            source_coordinate = tuple(item["coordinate"])
            source_id = _coordinate_node_id(item["tensor_id"], source_coordinate)
            if len(coordinate_edges) >= max_edges:
                truncated = True
                frontier.append({
                    "coordinate_node_id": source_id,
                    "tensor_id": item["tensor_id"],
                    "coordinate": list(source_coordinate),
                    "reason": "max_edges",
                })
                continue
            coordinate_edges.append({
                "source": source_id,
                "target": coordinate_id,
                "relation": item["relation"],
                "operand_index": item.get("operand_index"),
                "operator": expansion.get("operator"),
                "module_scope": expansion.get("module_scope"),
            })
            if source_id not in coordinate_nodes:
                queue.append((item["tensor_id"], source_coordinate, depth + 1))

    target_coordinate_id = _coordinate_node_id(target_tensor_id, target_coord)
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in coordinate_nodes}
    for edge in coordinate_edges:
        if edge["source"] in adjacency and edge["target"] in adjacency:
            adjacency[edge["source"]].append(edge["target"])

    focused_source_id = None
    focus_reachable = None
    if focus_source_tensor_id is not None and focus_source_coordinate is not None:
        focused = tuple(focus_source_coordinate)
        if focus_source_tensor_id not in nodes:
            raise CoordinateTraceError(f"Unknown focus source tensor {focus_source_tensor_id!r}.")
        _validate_coordinate(nodes[focus_source_tensor_id], focused)
        focused_source_id = _coordinate_node_id(focus_source_tensor_id, focused)
        if focused_source_id in coordinate_nodes:
            visited = set()
            pending = deque([focused_source_id])
            while pending:
                current_id = pending.popleft()
                if current_id in visited:
                    continue
                visited.add(current_id)
                pending.extend(adjacency.get(current_id, []))
            focus_reachable = target_coordinate_id in visited

    input_coordinates = [
        item for item in coordinate_nodes.values()
        if item["tensor_kind"] == "model_input"
    ]
    parameter_coordinates = [
        item for item in coordinate_nodes.values()
        if item["tensor_kind"] == "parameter"
    ]
    unsupported_frontiers = sum(
        item["reason"] not in {"model_input", "parameter", "external_tensor", "max_depth"}
        for item in frontier
    )
    terminal_reasons = {"model_input", "parameter", "external_tensor"}
    unresolved_reasons = [
        item["reason"] for item in frontier
        if item["reason"] not in terminal_reasons
    ]
    return {
        "graph_type": "coordinate_dependency_graph",
        "schema_version": "0.1.0",
        "query": {
            "target_tensor_id": target_tensor_id,
            "target_coordinate": list(target_coord),
            "focus_source_tensor_id": focus_source_tensor_id,
            "focus_source_coordinate": (
                list(focus_source_coordinate) if focus_source_coordinate is not None else None
            ),
            "focus_source_reaches_target": focus_reachable,
            "coordinate_direction": "backward_from_selected_output",
        },
        "summary": {
            "coordinate_node_count": len(coordinate_nodes),
            "coordinate_edge_count": len(coordinate_edges),
            "input_coordinate_count": len(input_coordinates),
            "parameter_coordinate_count": len(parameter_coordinates),
            "frontier_count": len(frontier),
            "unsupported_or_unresolved_frontier_count": unsupported_frontiers,
            "truncated": truncated,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "max_edges": max_edges,
            "max_fan_in": max_fan_in,
            "complete": not truncated and not unresolved_reasons,
        },
        "input_coordinates": input_coordinates,
        "parameter_coordinates": parameter_coordinates,
        "frontier": frontier,
        "nodes": list(coordinate_nodes.values()),
        "edges": coordinate_edges,
        "interpretation_limits": [
            "This is a coordinate-level dependency expansion for one captured execution.",
            "Each supported rule describes mathematical operand dependency, not contribution magnitude.",
            "Embedding index coordinates select a table row; they are routing/index dependencies, not additive numeric contributions.",
            "Unsupported operators and resource limits are reported as frontiers; omitted paths must not be interpreted as absent dependencies.",
            "A dependency edge does not establish semantic meaning or causal necessity.",
        ],
    }


def build_coordinate_trace_from_reports(
    trace_report: dict[str, Any],
    runtime_graphs: dict[str, Any],
    pass_name: str = "prompt_prefill",
    output_position: int | None = None,
    output_token_id: int | None = None,
    output_token_rank: int = 0,
    input_position: int | None = None,
    max_depth: int = 12,
    max_nodes: int = 20_000,
    max_edges: int = 50_000,
    max_fan_in: int = 4096,
) -> dict[str, Any]:
    """Resolve a GPT-2 logit coordinate and trace its scalar dependencies."""
    if pass_name not in runtime_graphs.get("passes", {}):
        raise CoordinateTraceError(f"Execution pass {pass_name!r} was not found.")
    graph = runtime_graphs["passes"][pass_name]
    pass_record = trace_report.get("runtime_trace", {}).get("passes", {}).get(pass_name)
    if pass_record is None:
        raise CoordinateTraceError(f"Trace report does not contain pass {pass_name!r}.")
    nodes = _node_index(graph)
    logits_id = "model_output:logits"
    if logits_id not in nodes:
        raise CoordinateTraceError("Runtime graph has no model_output:logits tensor.")
    logits_shape = nodes[logits_id].get("shape")
    if not isinstance(logits_shape, list) or len(logits_shape) != 3:
        raise CoordinateTraceError("Expected logits with shape [batch, sequence, vocabulary].")
    position = logits_shape[1] - 1 if output_position is None else output_position
    if position < 0 or position >= logits_shape[1]:
        raise CoordinateTraceError("Output position is outside the logits sequence dimension.")
    probability = None
    token = None
    if output_token_id is None:
        if position != logits_shape[1] - 1:
            raise CoordinateTraceError(
                "Top-token rank is saved only for the final position; provide output_token_id."
            )
        predictions = pass_record.get("last_position_top_tokens", [])
        if output_token_rank < 0 or output_token_rank >= len(predictions):
            raise CoordinateTraceError("Output token rank is outside the saved top-token list.")
        chosen = predictions[output_token_rank]
        output_token_id = int(chosen["token_id"])
        probability = chosen.get("probability")
        token = chosen.get("token")
    elif output_token_id < 0 or output_token_id >= logits_shape[2]:
        raise CoordinateTraceError("Output token ID is outside the vocabulary.")

    input_tensor = next(
        (node for node in nodes.values()
         if node.get("kind") == "model_input" and node.get("label") == "input_ids"),
        None,
    )
    tensor_values = {}
    focus_tensor_id = None
    focus_coordinate = None
    if input_tensor is not None:
        input_ids = pass_record.get("input_token_ids")
        if input_ids is not None:
            tensor_values[input_tensor["id"]] = [input_ids]
        if input_position is not None:
            if input_position < 0 or input_position >= logits_shape[1]:
                raise CoordinateTraceError("Input position is outside the prompt sequence.")
            focus_tensor_id = input_tensor["id"]
            focus_coordinate = [0, input_position]

    result = trace_coordinate_graph(
        graph,
        logits_id,
        [0, position, int(output_token_id)],
        tensor_values=tensor_values,
        focus_source_tensor_id=focus_tensor_id,
        focus_source_coordinate=focus_coordinate,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_fan_in=max_fan_in,
    )
    result["query"].update({
        "pass_name": pass_name,
        "input_text": trace_report.get("input", {}).get("text"),
        "output_token_id": int(output_token_id),
        "output_token": token,
        "output_probability": probability,
        "input_position": input_position,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace scalar tensor-coordinate dependencies in an NCIFM runtime graph."
    )
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--runtime-graphs", required=True)
    parser.add_argument("--pass", dest="pass_name", default="prompt_prefill")
    parser.add_argument("--output-position", type=int, default=None)
    parser.add_argument("--output-token-id", type=int, default=None)
    parser.add_argument("--output-token-rank", type=int, default=0)
    parser.add_argument("--input-position", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--max-edges", type=int, default=50000)
    parser.add_argument("--max-fan-in", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_report = json.loads(Path(args.trace_report).read_text(encoding="utf-8"))
    runtime_graphs = json.loads(Path(args.runtime_graphs).read_text(encoding="utf-8"))
    result = build_coordinate_trace_from_reports(
        trace_report,
        runtime_graphs,
        pass_name=args.pass_name,
        output_position=args.output_position,
        output_token_id=args.output_token_id,
        output_token_rank=args.output_token_rank,
        input_position=args.input_position,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_fan_in=args.max_fan_in,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "query": result["query"],
        "summary": result["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
