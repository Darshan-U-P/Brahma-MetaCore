"""Focused source-to-target coordinate dependency search for NCIFM."""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from typing import Any

from src.metaobserver.coordinate_tracer import (
    CoordinateTraceError,
    _coordinate_node_id,
    _node_index,
    _validate_coordinate,
    trace_tensor_coordinate,
)


def _group_contains(group: dict[str, Any], coordinate: tuple[int, ...]) -> bool:
    """Return whether a scalar coordinate belongs to a compact dependency group."""
    prefix = list(group["coordinate_prefix"])
    axes = list(group["varying_axes"])
    sizes = list(group["axis_sizes"])
    if len(prefix) != len(coordinate):
        return False
    for axis, size in zip(axes, sizes):
        if axis < 0 or axis >= len(coordinate) or not (0 <= coordinate[axis] < size):
            return False
    varying = set(axes)
    return all(coordinate[i] == prefix[i] for i in range(len(prefix)) if i not in varying)


def _source_tensor_distances(
    graph: dict[str, Any], source_tensor_id: str
) -> dict[str, int]:
    """Return shortest tensor-level distances downstream from the source.

    Distances are a search-ordering heuristic only. They never prune a
    coordinate dependency: all source-reachable branches remain eligible, and
    the coordinate-level operator rules remain authoritative.
    """
    nodes = _node_index(graph)
    if source_tensor_id not in nodes:
        raise CoordinateTraceError(f"Unknown source tensor {source_tensor_id!r}.")

    operator_inputs: dict[str, list[str]] = {}
    operator_outputs: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        if relation == "operand":
            operator_inputs.setdefault(edge["target"], []).append(edge["source"])
        elif relation == "produces":
            operator_outputs.setdefault(edge["source"], []).append(edge["target"])

    downstream: dict[str, set[str]] = {}
    for operator_id, inputs in operator_inputs.items():
        outputs = operator_outputs.get(operator_id, [])
        for input_tensor_id in inputs:
            downstream.setdefault(input_tensor_id, set()).update(outputs)

    distances = {source_tensor_id: 0}
    pending = deque([source_tensor_id])
    while pending:
        current = pending.popleft()
        next_distance = distances[current] + 1
        for output_tensor_id in downstream.get(current, ()):
            if output_tensor_id not in distances:
                distances[output_tensor_id] = next_distance
                pending.append(output_tensor_id)
    return distances


def trace_focused_coordinate_path(
    graph: dict[str, Any],
    source_tensor_id: str,
    source_coordinate: list[int] | tuple[int, ...],
    target_tensor_id: str,
    target_coordinate: list[int] | tuple[int, ...],
    tensor_values: dict[str, Any] | None = None,
    max_depth: int = 128,
    max_nodes: int = 50_000,
    max_candidates: int = 100_000,
    max_frontier: int = 10_000,
    parameter_policy: str = "compact",
    search_strategy: str = "source_guided",
) -> dict[str, Any]:
    """Find a dependency path from one exact source coordinate to a target.

    Search is backward from target. The returned path is ordered from source
    toward target. Parameter groups are compact terminal records by default.
    """
    if max_depth < 0 or min(max_nodes, max_candidates, max_frontier) < 1:
        raise CoordinateTraceError("Depth must be non-negative and all other limits positive.")
    if parameter_policy not in {"compact", "expand"}:
        raise CoordinateTraceError("parameter_policy must be 'compact' or 'expand'.")
    if search_strategy not in {"source_guided", "breadth_first"}:
        raise CoordinateTraceError(
            "search_strategy must be source_guided or breadth_first."
        )

    nodes = _node_index(graph)
    if source_tensor_id not in nodes:
        raise CoordinateTraceError(f"Unknown source tensor {source_tensor_id!r}.")
    if target_tensor_id not in nodes:
        raise CoordinateTraceError(f"Unknown target tensor {target_tensor_id!r}.")
    source = tuple(source_coordinate)
    target = tuple(target_coordinate)
    _validate_coordinate(nodes[source_tensor_id], source)
    _validate_coordinate(nodes[target_tensor_id], target)
    values = tensor_values or {}
    reachability_started = time.perf_counter()
    source_tensor_distances = _source_tensor_distances(graph, source_tensor_id)
    source_reachable_tensors = set(source_tensor_distances)
    reachability_seconds = time.perf_counter() - reachability_started

    source_id = _coordinate_node_id(source_tensor_id, source)
    target_id = _coordinate_node_id(target_tensor_id, target)
    coordinate_nodes: dict[str, dict[str, Any]] = {}
    toward_target: dict[str, dict[str, Any]] = {}
    parameter_groups: list[dict[str, Any]] = []
    # Many output coordinates (notably LayerNorm and softmax) share the exact
    # same activation dependency slice. Expand each identical slice once.
    # Its first target coordinate is already connected to the requested target,
    # so a source found in that slice still has a valid route to the target.
    seen_activation_groups: set[tuple[Any, ...]] = set()
    activation_groups_deduplicated = 0
    frontier: list[dict[str, Any]] = []
    queue: list[tuple[int, int, int, int, str, tuple[int, ...]]] = []
    queue_sequence = itertools.count()

    def enqueue_coordinate(
        tensor_id: str, coordinate: tuple[int, ...], depth: int
    ) -> None:
        distance = source_tensor_distances.get(tensor_id, 1_000_000_000)
        if search_strategy == "source_guided":
            primary, secondary = distance, depth
        else:
            primary, secondary = depth, distance
        heapq.heappush(
            queue,
            (primary, secondary, next(queue_sequence), depth, tensor_id, coordinate),
        )

    enqueue_coordinate(target_tensor_id, target, 0)
    queue_peak_size = 1
    scheduled = {target_id}
    candidates_examined = 0
    coordinate_nodes_processed = 0
    max_depth_processed = 0
    resource_limited = False
    operator_profile: dict[str, dict[str, float | int]] = {}
    search_started = time.perf_counter()
    unsupported_seen = False
    found = source_id == target_id

    def add_frontier(item: dict[str, Any]) -> None:
        nonlocal resource_limited
        if len(frontier) < max_frontier:
            frontier.append(item)
        else:
            resource_limited = True

    def retain_parameter(
        tensor_id: str,
        relation: str,
        coordinate: tuple[int, ...] | None = None,
        group: dict[str, Any] | None = None,
        operator: str | None = None,
        module_scope: str | None = None,
        target_coordinate_id: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "tensor_id": tensor_id,
            "relation": relation,
            "operator": operator,
            "module_scope": module_scope,
            "target_coordinate_node_id": target_coordinate_id,
        }
        if coordinate is not None:
            record["coordinate"] = list(coordinate)
        if group is not None:
            record["coordinate_prefix"] = list(group["coordinate_prefix"])
            record["varying_axes"] = list(group["varying_axes"])
            record["axis_sizes"] = list(group["axis_sizes"])
            record["dependency_count"] = group["dependency_count"]
        parameter_groups.append(record)

    while queue and not found:
        _, _, _, depth, current_tensor_id, current_coordinate = heapq.heappop(queue)
        coordinate_nodes_processed += 1
        max_depth_processed = max(max_depth_processed, depth)
        limit_this_expansion = False
        current_id = _coordinate_node_id(current_tensor_id, current_coordinate)
        tensor_node = nodes[current_tensor_id]
        coordinate_nodes[current_id] = {
            "id": current_id,
            "tensor_id": current_tensor_id,
            "tensor_label": tensor_node.get("label", current_tensor_id),
            "tensor_kind": tensor_node.get("kind"),
            "coordinate": list(current_coordinate),
            "tensor_shape": tensor_node.get("shape"),
            "depth": depth,
        }

        if current_id == source_id:
            found = True
            break
        if tensor_node.get("kind") in {"model_input", "parameter", "external_tensor"}:
            add_frontier({
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coordinate),
                "reason": tensor_node.get("kind"),
            })
            continue
        if depth >= max_depth:
            resource_limited = True
            add_frontier({
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coordinate),
                "reason": "max_depth",
            })
            continue

        expansion_started = time.perf_counter()
        expansion = trace_tensor_coordinate(
            graph, current_tensor_id, current_coordinate,
            tensor_values=values, max_fan_in=max_candidates,
        )
        expansion_seconds = time.perf_counter() - expansion_started
        operator_name = expansion.get("operator") or "<unknown>"
        operator_stats = operator_profile.setdefault(
            operator_name, {"coordinates_expanded": 0, "elapsed_seconds": 0.0}
        )
        operator_stats["coordinates_expanded"] += 1
        operator_stats["elapsed_seconds"] += expansion_seconds
        status = expansion.get("status", "unknown")
        if status != "expanded":
            add_frontier({
                "tensor_id": current_tensor_id,
                "coordinate": list(current_coordinate),
                "reason": status,
                "operator": expansion.get("operator"),
                "module_scope": expansion.get("module_scope"),
                "required_fan_in": expansion.get("required_fan_in"),
            })
            unsupported_seen = True

        dependency_groups = expansion.get("dependency_groups", [])
        if search_strategy == "source_guided":
            dependency_groups = sorted(
                dependency_groups,
                key=lambda group: source_tensor_distances.get(
                    group["tensor_id"], 1_000_000_000
                ),
            )
        for group in dependency_groups:
            group_tensor_id = group["tensor_id"]
            group_node = nodes[group_tensor_id]
            relation = group["relation"]
            if group_node.get("kind") == "parameter":
                retain_parameter(
                    group_tensor_id, relation, group=group,
                    operator=expansion.get("operator"),
                    module_scope=expansion.get("module_scope"),
                    target_coordinate_id=current_id,
                )
                if group_tensor_id == source_tensor_id and _group_contains(group, source):
                    toward_target[source_id] = {
                        "next": current_id, "relation": relation,
                        "operator": expansion.get("operator"),
                        "module_scope": expansion.get("module_scope"),
                    }
                    coordinate_nodes.setdefault(source_id, {
                        "id": source_id, "tensor_id": source_tensor_id,
                        "tensor_label": group_node.get("label", source_tensor_id),
                        "tensor_kind": group_node.get("kind"),
                        "coordinate": list(source),
                        "tensor_shape": group_node.get("shape"),
                        "depth": depth + 1,
                    })
                    found = True
                    break
                # Parameter tensors cannot lie on a path from an activation
                # source. Keep their compact provenance, but do not enumerate
                # them when searching for this source.
                if group_tensor_id not in source_reachable_tensors:
                    continue
                if parameter_policy == "compact":
                    continue

            # Match the exact requested activation coordinate against the
            # compact group before enumerating its scalar members.
            if group_tensor_id == source_tensor_id and _group_contains(group, source):
                toward_target[source_id] = {
                    "next": current_id, "relation": relation,
                    "operator": expansion.get("operator"),
                    "module_scope": expansion.get("module_scope"),
                }
                coordinate_nodes.setdefault(source_id, {
                    "id": source_id, "tensor_id": source_tensor_id,
                    "tensor_label": group_node.get("label", source_tensor_id),
                    "tensor_kind": group_node.get("kind"),
                    "coordinate": list(source),
                    "tensor_shape": group_node.get("shape"),
                    "depth": depth + 1,
                })
                found = True
                break

            # A tensor-level disconnected branch cannot contain the requested
            # source coordinate. Prune it before generating scalar coordinates.
            if group_tensor_id not in source_reachable_tensors:
                continue

            # Deduplicate identical activation fan-in groups. This is especially
            # important for normalization: every output feature at one sequence
            # position depends on the same normalized input slice.
            group_key = (
                group_tensor_id,
                tuple(group["coordinate_prefix"]),
                tuple(group["varying_axes"]),
                tuple(group["axis_sizes"]),
                relation,
                expansion.get("operator"),
                expansion.get("module_scope"),
            )
            if group_node.get("kind") != "parameter":
                if group_key in seen_activation_groups:
                    activation_groups_deduplicated += 1
                    continue
                seen_activation_groups.add(group_key)

            axes = group["varying_axes"]
            sizes = group["axis_sizes"]
            prefix = list(group["coordinate_prefix"])
            for varying in itertools.product(*(range(size) for size in sizes)):
                candidates_examined += 1
                if candidates_examined > max_candidates:
                    resource_limited = True
                    limit_this_expansion = True
                    add_frontier({
                        "reason": "max_candidates",
                        "tensor_id": group_tensor_id,
                        "dependency_group": relation,
                    })
                    break
                mapped = list(prefix)
                for axis, value in zip(axes, varying):
                    mapped[axis] = value
                dependency_coordinate = tuple(mapped)
                dependency_id = _coordinate_node_id(group_tensor_id, dependency_coordinate)
                toward_target.setdefault(dependency_id, {
                    "next": current_id, "relation": relation,
                    "operator": expansion.get("operator"),
                    "module_scope": expansion.get("module_scope"),
                })
                if dependency_id == source_id:
                    coordinate_nodes.setdefault(source_id, {
                        "id": source_id, "tensor_id": source_tensor_id,
                        "tensor_label": nodes[source_tensor_id].get("label", source_tensor_id),
                        "tensor_kind": nodes[source_tensor_id].get("kind"),
                        "coordinate": list(source),
                        "tensor_shape": nodes[source_tensor_id].get("shape"),
                        "depth": depth + 1,
                    })
                    found = True
                    break
                if dependency_id not in scheduled:
                    if len(scheduled) >= max_nodes:
                        resource_limited = True
                        limit_this_expansion = True
                        add_frontier({
                            "reason": "max_nodes", "tensor_id": group_tensor_id,
                            "coordinate": list(dependency_coordinate),
                        })
                        break
                    scheduled.add(dependency_id)
                    enqueue_coordinate(group_tensor_id, dependency_coordinate, depth + 1)
                    queue_peak_size = max(queue_peak_size, len(queue))
            if found or limit_this_expansion:
                break
        if found:
            break
        if limit_this_expansion:
            continue

        dependencies = expansion.get("dependencies", [])
        if search_strategy == "source_guided":
            dependencies = sorted(
                dependencies,
                key=lambda dependency: source_tensor_distances.get(
                    dependency["tensor_id"], 1_000_000_000
                ),
            )
        for dependency in dependencies:
            dep_tensor_id = dependency["tensor_id"]
            dep_coordinate = tuple(dependency["coordinate"])
            dep_id = _coordinate_node_id(dep_tensor_id, dep_coordinate)
            dep_node = nodes[dep_tensor_id]
            # Exact source matches are terminal checks, not branch expansion.
            # Check them before enforcing the candidate-enumeration budget.
            if dep_id == source_id:
                if dep_node.get("kind") == "parameter":
                    retain_parameter(
                        dep_tensor_id, dependency["relation"], coordinate=dep_coordinate,
                        operator=expansion.get("operator"),
                        module_scope=expansion.get("module_scope"),
                        target_coordinate_id=current_id,
                    )
                toward_target[source_id] = {
                    "next": current_id, "relation": dependency["relation"],
                    "operator": expansion.get("operator"),
                    "module_scope": expansion.get("module_scope"),
                }
                coordinate_nodes.setdefault(source_id, {
                    "id": source_id, "tensor_id": source_tensor_id,
                    "tensor_label": dep_node.get("label", source_tensor_id),
                    "tensor_kind": dep_node.get("kind"),
                    "coordinate": list(source),
                    "tensor_shape": dep_node.get("shape"),
                    "depth": depth + 1,
                })
                found = True
                break
            candidates_examined += 1
            if candidates_examined > max_candidates:
                resource_limited = True
                limit_this_expansion = True
                add_frontier({"reason": "max_candidates", "tensor_id": dep_tensor_id})
                break
            if dep_node.get("kind") == "parameter":
                retain_parameter(
                    dep_tensor_id, dependency["relation"], coordinate=dep_coordinate,
                    operator=expansion.get("operator"),
                    module_scope=expansion.get("module_scope"),
                    target_coordinate_id=current_id,
                )
                if dep_tensor_id != source_tensor_id or dep_coordinate != source:
                    continue
            elif dep_tensor_id not in source_reachable_tensors:
                continue
            toward_target.setdefault(dep_id, {
                "next": current_id, "relation": dependency["relation"],
                "operator": expansion.get("operator"),
                "module_scope": expansion.get("module_scope"),
            })
            if dep_id == source_id:
                coordinate_nodes.setdefault(source_id, {
                    "id": source_id, "tensor_id": source_tensor_id,
                    "tensor_label": dep_node.get("label", source_tensor_id),
                    "tensor_kind": dep_node.get("kind"),
                    "coordinate": list(source),
                    "tensor_shape": dep_node.get("shape"),
                    "depth": depth + 1,
                })
                found = True
                break
            if dep_id not in scheduled:
                if len(scheduled) >= max_nodes:
                    resource_limited = True
                    limit_this_expansion = True
                    add_frontier({
                        "reason": "max_nodes", "tensor_id": dep_tensor_id,
                        "coordinate": list(dep_coordinate),
                    })
                    break
                scheduled.add(dep_id)
                enqueue_coordinate(dep_tensor_id, dep_coordinate, depth + 1)
                queue_peak_size = max(queue_peak_size, len(queue))
        if found:
            break
        if limit_this_expansion:
            continue

    search_seconds = time.perf_counter() - search_started
    path: list[dict[str, Any]] = []
    if found:
        cursor = source_id
        seen: set[str] = set()
        while cursor != target_id:
            if cursor in seen or cursor not in toward_target:
                found = False
                path = []
                break
            seen.add(cursor)
            link = toward_target[cursor]
            path.append({
                "source_coordinate_node_id": cursor,
                "target_coordinate_node_id": link["next"],
                "relation": link["relation"],
                "operator": link.get("operator"),
                "module_scope": link.get("module_scope"),
            })
            cursor = link["next"]

    terminal_frontier_reasons = {"model_input", "parameter", "external_tensor"}
    unresolved_frontier = any(
        item.get("reason") not in terminal_frontier_reasons for item in frontier
    )
    search_complete = (
        not queue and not resource_limited and not unsupported_seen
        and not unresolved_frontier
    )
    if found:
        status = "path_found"
    elif resource_limited:
        status = "inconclusive_limit"
    elif unsupported_seen:
        status = "inconclusive_unsupported"
    else:
        status = "path_not_found"

    return {
        "graph_type": "focused_coordinate_path",
        "schema_version": "0.1.0",
        "query": {
            "source": {"tensor_id": source_tensor_id, "coordinate": list(source)},
            "target": {"tensor_id": target_tensor_id, "coordinate": list(target)},
            "coordinate_direction": "backward_search_forward_path",
            "parameter_policy": parameter_policy,
        },
        "result": {
            "status": status,
            "path_found": found,
            "search_complete": search_complete,
        },
        "path": path,
        "nodes": list(coordinate_nodes.values()),
        "parameter_dependency_groups": parameter_groups,
        "frontier": frontier,
        "summary": {
            "coordinate_nodes_examined": coordinate_nodes_processed,
            "coordinate_nodes_recorded": len(coordinate_nodes),
            "coordinate_nodes_processed": coordinate_nodes_processed,
            "coordinate_nodes_scheduled": len(scheduled),
            "queue_remaining_at_stop": len(queue),
            "queue_peak_size": queue_peak_size,
            "max_depth_processed": max_depth_processed,
            "source_reachable_tensor_count": len(source_reachable_tensors),
            "target_tensor_distance": source_tensor_distances.get(target_tensor_id),
            "search_strategy": search_strategy,
            "reachability_seconds": round(reachability_seconds, 6),
            "search_seconds": round(search_seconds, 6),
            "operator_profile": {
                name: {
                    "coordinates_expanded": stats["coordinates_expanded"],
                    "elapsed_seconds": round(stats["elapsed_seconds"], 6),
                }
                for name, stats in sorted(
                    operator_profile.items(),
                    key=lambda item: item[1]["elapsed_seconds"],
                    reverse=True,
                )
            },
            "candidate_dependencies_examined": candidates_examined,
            "path_length": len(path),
            "parameter_groups_retained": len(parameter_groups),
            "activation_dependency_groups_deduplicated": activation_groups_deduplicated,
            "activation_dependency_groups_expanded": len(seen_activation_groups),
            "unresolved_frontier_count": len(frontier),
            "resource_limit_reached": resource_limited,
            "unsupported_operator_or_rule_seen": unsupported_seen,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
            "max_candidates": max_candidates,
            "max_frontier": max_frontier,
        },
        "interpretation_limits": [
            "A path establishes a mathematical operand dependency for this captured execution, not contribution magnitude or semantic meaning.",
            "Embedding lookup indices are routing dependencies, not additive numeric contributions.",
            "A found path may coexist with unresolved alternative branches; inspect search_complete and frontier.",
            "A path_not_found result is conclusive only when search_complete is true.",
            "Compact parameter groups identify participating parameter coordinates without enumerating every scalar by default.",
            "Identical activation dependency groups are expanded once; their shared scalar dependencies remain connected to the first encountered target coordinate.",
            "Tensor-level source reachability prunes disconnected activation branches; exact coordinate dependencies are still determined only by supported operator rules.",
            "coordinate_nodes_processed counts dequeued coordinates; coordinate_nodes_scheduled includes both processed and queued coordinates.",
        ],
    }
