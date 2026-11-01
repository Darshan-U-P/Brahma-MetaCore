"""Operator-level tensor dependency capture for one PyTorch execution pass."""
from __future__ import annotations

import weakref
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "requires_grad": bool(tensor.requires_grad),
        "stride": list(tensor.stride()),
    }


def _encode_argument(value: Any, operand_positions: dict[int, list[int]],
                     cursors: dict[int, int]) -> Any:
    """Convert operator arguments to JSON-safe values, preserving tensor refs."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        position = cursors.get(key, 0)
        positions = operand_positions.get(key, [])
        cursors[key] = position + 1
        return {
            "tensor_operand_index": positions[position]
            if position < len(positions) else None
        }
    if isinstance(value, tuple):
        return [_encode_argument(item, operand_positions, cursors) for item in value]
    if isinstance(value, list):
        return [_encode_argument(item, operand_positions, cursors) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_argument(item, operand_positions, cursors)
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return {
            "slice": [
                _encode_argument(value.start, operand_positions, cursors),
                _encode_argument(value.stop, operand_positions, cursors),
                _encode_argument(value.step, operand_positions, cursors),
            ]
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    # Avoid retaining arbitrary Python objects or tensor contents in the report.
    return str(value)


def _operator_argument_metadata(func: Any, args: tuple[Any, ...],
                                kwargs: dict[str, Any],
                                input_tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Capture schema argument names and scalar arguments without tensor payloads."""
    schema = getattr(func, "_schema", None)
    if schema is None:
        return {}
    operand_positions: dict[int, list[int]] = {}
    for index, tensor in enumerate(input_tensors):
        operand_positions.setdefault(id(tensor), []).append(index)
    cursors: dict[int, int] = {}

    # Encode in the same traversal order used by tree_flatten((args, kwargs)).
    encoded_args = [
        _encode_argument(value, operand_positions, cursors) for value in args
    ]
    encoded_kwargs = {
        key: _encode_argument(value, operand_positions, cursors)
        for key, value in kwargs.items()
    }
    result: dict[str, Any] = {}
    for index, argument in enumerate(schema.arguments):
        if index < len(encoded_args):
            result[argument.name] = encoded_args[index]
        elif argument.name in encoded_kwargs:
            result[argument.name] = encoded_kwargs[argument.name]
    return result


class RuntimeTensorGraph(TorchDispatchMode):
    """Record actual tensor operands and results at the ATen operator boundary.

    This is an execution-specific computational dependency graph. It does not
    assign semantic meaning to neurons or prove causal contribution.
    """

    def __init__(self, model: torch.nn.Module, max_ops: int = 50_000) -> None:
        super().__init__()
        if max_ops < 1:
            raise ValueError("max_ops must be at least 1")
        self.model = model
        self.max_ops = max_ops
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self._tensor_lookup: dict[int, tuple[weakref.ReferenceType, str]] = {}
        self._next_tensor = 0
        self._next_op = 0
        self._module_stack: list[str] = []
        self._module_handles: list[Any] = []
        self.truncated = False
        for name, parameter in model.named_parameters():
            self._register_tensor(parameter, "parameter", name, parameter_name=name)

    def _register_tensor(self, tensor: torch.Tensor, kind: str, label: str,
                         parameter_name: str | None = None,
                         force_new: bool = False) -> str:
        key = id(tensor)
        current = self._tensor_lookup.get(key)
        if not force_new and current is not None and current[0]() is tensor:
            return current[1]
        node_id = f"tensor:{self._next_tensor}"
        self._next_tensor += 1
        node: dict[str, Any] = {
            "id": node_id, "kind": kind, "label": label,
            **tensor_metadata(tensor),
        }
        if parameter_name is not None:
            node["parameter_name"] = parameter_name
        self.nodes.append(node)

        def cleanup(ref: weakref.ReferenceType, tensor_key: int = key,
                    registered_id: str = node_id) -> None:
            entry = self._tensor_lookup.get(tensor_key)
            if entry is not None and entry[0] is ref and entry[1] == registered_id:
                self._tensor_lookup.pop(tensor_key, None)

        self._tensor_lookup[key] = (weakref.ref(tensor, cleanup), node_id)
        return node_id

    def register_input(self, name: str, tensor: torch.Tensor) -> str:
        return self._register_tensor(tensor, "model_input", name)

    def _lookup_or_external(self, tensor: torch.Tensor) -> str:
        entry = self._tensor_lookup.get(id(tensor))
        if entry is not None and entry[0]() is tensor:
            return entry[1]
        return self._register_tensor(tensor, "external_tensor", "untracked_tensor_input")

    def install_module_scopes(self) -> None:
        """Annotate operators with their innermost active nn.Module."""
        if self._module_handles:
            return
        for name, module in self.model.named_modules():
            scope = name or "<root>"

            def before(_module: torch.nn.Module, _inputs: tuple[Any, ...],
                       current_scope: str = scope) -> None:
                self._module_stack.append(current_scope)

            def after(_module: torch.nn.Module, _inputs: tuple[Any, ...],
                      output: Any, current_scope: str = scope) -> Any:
                if self._module_stack and self._module_stack[-1] == current_scope:
                    self._module_stack.pop()
                elif current_scope in self._module_stack:
                    self._module_stack.remove(current_scope)
                return output

            self._module_handles.append(module.register_forward_pre_hook(before))
            self._module_handles.append(module.register_forward_hook(after))

    def remove_module_scopes(self) -> None:
        for handle in self._module_handles:
            handle.remove()
        self._module_handles.clear()
        self._module_stack.clear()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self._next_op >= self.max_ops:
            self.truncated = True
            return func(*args, **kwargs)

        flat_inputs, _ = tree_flatten((args, kwargs))
        input_tensors = [item for item in flat_inputs if isinstance(item, torch.Tensor)]
        input_ids = [self._lookup_or_external(item) for item in input_tensors]
        argument_metadata = _operator_argument_metadata(
            func, args, kwargs, input_tensors
        )

        result = func(*args, **kwargs)
        flat_outputs, _ = tree_flatten(result)
        output_tensors = [item for item in flat_outputs if isinstance(item, torch.Tensor)]
        op_id = f"op:{self._next_op}"
        self._next_op += 1

        schema = getattr(func, "_schema", None)
        if schema is not None:
            overload = getattr(func, "_overloadname", "")
            op_name = f"{schema.name}.{overload}" if overload else schema.name
        else:
            op_name = str(func)

        self.nodes.append({
            "id": op_id,
            "kind": "operator",
            "operator": op_name,
            "module_scope": self._module_stack[-1] if self._module_stack else None,
            "input_count": len(input_ids),
            "output_count": len(output_tensors),
            "argument_metadata": argument_metadata,
        })
        for index, tensor_id in enumerate(input_ids):
            self.edges.append({
                "source": tensor_id, "target": op_id,
                "relation": "operand", "operand_index": index,
            })
        for index, tensor in enumerate(output_tensors):
            tensor_id = self._register_tensor(
                tensor, "computed_tensor", f"{op_name}:output_{index}", force_new=True
            )
            self.edges.append({
                "source": op_id, "target": tensor_id,
                "relation": "produces", "output_index": index,
            })
        return result

    def add_output(self, name: str, tensor: torch.Tensor) -> None:
        source = self._lookup_or_external(tensor)
        output_id = f"model_output:{name}"
        self.nodes.append({
            "id": output_id, "kind": "model_output", "label": name,
            **tensor_metadata(tensor),
        })
        self.edges.append({
            "source": source, "target": output_id,
            "relation": "exposed_as_model_output",
        })

    def to_dict(self) -> dict[str, Any]:
        operator_count = sum(node["kind"] == "operator" for node in self.nodes)
        tensor_count = sum(node["kind"] != "operator" for node in self.nodes)
        return {
            "graph_type": "runtime_tensor_dependency_graph",
            "schema_version": "0.3.0",
            "semantics": (
                "Edges connect tensor operands to actual ATen operators and "
                "operators to output tensors for this execution pass. Operator "
                "argument metadata records JSON-safe scalar arguments and "
                "operand references for coordinate-aware analysis. This is "
                "computational dependency evidence, not semantic interpretation "
                "or causal attribution."
            ),
            "capture": {
                "operator_count": operator_count,
                "tensor_node_count": tensor_count,
                "edge_count": len(self.edges),
                "max_ops": self.max_ops,
                "truncated": self.truncated,
            },
            "nodes": self.nodes,
            "edges": self.edges,
        }
