"""Upstream/downstream path queries over captured runtime tensor graphs."""
from __future__ import annotations
import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

class PathTraceError(ValueError):
    """Raised when a requested path or tensor coordinate is invalid."""

def _walk(start: str, adjacency: dict[str, list[str]]) -> set[str]:
    visited: set[str] = set()
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        queue.extend(adjacency.get(node_id, ()))
    return visited

def _resolve_node(nodes_by_id: dict[str, dict[str, Any]], query: str) -> str:
    if query in nodes_by_id:
        return query
    matches = [node_id for node_id, node in nodes_by_id.items()
               if node.get("label") == query or node.get("name") == query]
    if len(matches) != 1:
        raise PathTraceError(
            f"Node query {query!r} resolved to {len(matches)} nodes; use an exact node ID."
        )
    return matches[0]

def _is_dag(node_ids: set[str], adjacency: dict[str, list[str]]) -> bool:
    indegree = {node_id: 0 for node_id in node_ids}
    for source in node_ids:
        for target in adjacency.get(source, ()):
            if target in indegree:
                indegree[target] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in adjacency.get(source, ()):
            if target in indegree:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
    return visited == len(node_ids)

def _shortest_path(source: str, target: str,
                   adjacency: dict[str, list[str]]) -> list[str]:
    queue = deque([source])
    previous: dict[str, str | None] = {source: None}
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for neighbour in adjacency.get(current, ()):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    if target not in previous:
        raise PathTraceError(f"No directed path exists from {source!r} to {target!r}.")
    path = []
    current: str | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))

def trace_tensor_path(graph: dict[str, Any], source: str = "input_ids",
                      target: str = "model_output:logits",
                      input_coordinate: dict[str, int] | None = None,
                      output_coordinate: dict[str, int] | None = None) -> dict[str, Any]:
    """Trace forward reachability, output ancestors and a source-target subgraph."""
    if graph.get("graph_type") != "runtime_tensor_dependency_graph":
        raise PathTraceError("Input graph is not a runtime tensor dependency graph.")
    node_list = graph.get("nodes", [])
    nodes_by_id = {node["id"]: node for node in node_list}
    if len(nodes_by_id) != len(node_list):
        raise PathTraceError("Graph contains duplicate node IDs.")
    if not nodes_by_id:
        raise PathTraceError("Graph contains no nodes.")
    source_id, target_id = _resolve_node(nodes_by_id, source), _resolve_node(nodes_by_id, target)
    edges = graph.get("edges", [])
    adjacency = {node_id: [] for node_id in nodes_by_id}
    reverse = {node_id: [] for node_id in nodes_by_id}
    for edge in edges:
        start, end = edge["source"], edge["target"]
        if start not in nodes_by_id or end not in nodes_by_id:
            raise PathTraceError(f"Edge references unknown nodes: {start!r} -> {end!r}")
        adjacency[start].append(end)
        reverse[end].append(start)
    downstream = _walk(source_id, adjacency)
    if target_id not in downstream:
        raise PathTraceError(f"No path from source {source_id!r} to target {target_id!r}.")
    upstream = _walk(target_id, reverse)
    route_nodes = downstream.intersection(upstream)
    shortest = _shortest_path(source_id, target_id, adjacency)

    def subgraph(node_ids: set[str]) -> dict[str, Any]:
        return {
            "nodes": [nodes_by_id[node_id] for node_id in nodes_by_id if node_id in node_ids],
            "edges": [edge for edge in edges
                      if edge["source"] in node_ids and edge["target"] in node_ids],
        }

    upstream_nodes = [nodes_by_id[node_id] for node_id in upstream]
    parameters = sorted(
        (node for node in upstream_nodes if node.get("kind") == "parameter"),
        key=lambda item: item.get("parameter_name", item.get("label", "")),
    )
    operators = [node for node in upstream_nodes if node.get("kind") == "operator"]
    module_counts = Counter(node.get("module_scope") or "<unscoped>" for node in operators)
    route_edge_count = sum(
        1 for edge in edges if edge["source"] in route_nodes and edge["target"] in route_nodes
    )
    source_node, target_node = nodes_by_id[source_id], nodes_by_id[target_id]
    return {
        "graph_type": "computational_path_query", "schema_version": "0.1.0",
        "query": {
            "source_id": source_id, "source_label": source_node.get("label", source_id),
            "target_id": target_id, "target_label": target_node.get("label", target_id),
            "input_coordinate": input_coordinate, "output_coordinate": output_coordinate,
            "coordinate_semantics": (
                "Coordinates select positions/indexes within endpoint tensors. The graph "
                "remains tensor/operator-level; scalar neuron dependencies are not isolated."
            ),
        },
        "summary": {
            "reachable": True,
            "graph_is_acyclic": _is_dag(set(nodes_by_id), adjacency),
            "source_downstream_node_count": len(downstream),
            "target_upstream_node_count": len(upstream),
            "source_to_target_route_node_count": len(route_nodes),
            "source_to_target_route_edge_count": route_edge_count,
            "shortest_route_node_count": len(shortest),
            "upstream_operator_count": len(operators),
            "upstream_parameter_count": len(parameters),
            "upstream_module_operator_counts": dict(sorted(module_counts.items())),
        },
        "shortest_route": [
            {
                "node_id": node_id, "kind": nodes_by_id[node_id].get("kind"),
                "label": nodes_by_id[node_id].get(
                    "operator", nodes_by_id[node_id].get("label", node_id)
                ),
                "module_scope": nodes_by_id[node_id].get("module_scope"),
                "shape": nodes_by_id[node_id].get("shape"),
            } for node_id in shortest
        ],
        "parameter_dependencies": [
            {"node_id": node["id"],
             "name": node.get("parameter_name", node.get("label")),
             "shape": node.get("shape"), "dtype": node.get("dtype")}
            for node in parameters
        ],
        "source_to_target_subgraph": subgraph(route_nodes),
        "complete_target_upstream_subgraph": subgraph(upstream),
        "interpretation_limits": [
            "This is a tensor/operator dependency trace for one recorded execution.",
            "The shortest route is one representative route, not the only route.",
            "The complete target-upstream subgraph contains all recorded operands feeding the target, including parameters.",
            "Input/output coordinates annotate tensor positions; they do not decompose matrix operations into scalar neuron edges.",
            "Computational dependency alone does not establish semantic meaning or causal contribution.",
        ],
    }

def build_query_from_reports(trace_report: dict[str, Any], runtime_graphs: dict[str, Any],
                             pass_name: str = "prompt_prefill",
                             input_position: int | None = None,
                             output_position: int | None = None,
                             output_token_id: int | None = None,
                             output_token_rank: int = 0) -> dict[str, Any]:
    """Resolve token/logit coordinates and query one recorded forward pass."""
    if pass_name not in runtime_graphs.get("passes", {}):
        raise PathTraceError(f"Execution pass {pass_name!r} was not found.")
    graph = runtime_graphs["passes"][pass_name]
    pass_record = trace_report.get("runtime_trace", {}).get("passes", {}).get(pass_name)
    if pass_record is None:
        raise PathTraceError(f"Trace report does not contain pass {pass_name!r}.")
    input_tokens = pass_record.get("input_tokens", [])
    input_ids = pass_record.get("input_token_ids", [])
    if input_position is not None:
        if input_position < 0 or input_position >= len(input_ids):
            raise PathTraceError(f"input_position {input_position} is outside 0..{len(input_ids)-1}.")
        input_coordinate = {
            "batch": 0, "sequence_position": input_position,
            "token_id": int(input_ids[input_position]),
            "token": input_tokens[input_position] if input_position < len(input_tokens) else None,
        }
    else:
        input_coordinate = None
    logits_shape = next(
        (node.get("shape") for node in graph.get("nodes", [])
         if node.get("kind") == "model_output" and node.get("label") == "logits"), None
    )
    if not logits_shape or len(logits_shape) != 3:
        raise PathTraceError("Expected a [batch, sequence, vocabulary] logits tensor.")
    position = logits_shape[1] - 1 if output_position is None else output_position
    if position < 0 or position >= logits_shape[1]:
        raise PathTraceError(f"output_position {position} is outside 0..{logits_shape[1]-1}.")
    if output_token_id is None:
        if position != logits_shape[1] - 1:
            raise PathTraceError("Top-token rank is saved only for the final position; supply output_token_id.")
        predictions = pass_record.get("last_position_top_tokens", [])
        if output_token_rank < 0 or output_token_rank >= len(predictions):
            raise PathTraceError("output_token_rank is outside the saved top-token list.")
        chosen = predictions[output_token_rank]
        output_token_id = int(chosen["token_id"])
        output_token, output_probability = chosen.get("token"), chosen.get("probability")
    else:
        if output_token_id < 0 or output_token_id >= logits_shape[2]:
            raise PathTraceError(f"output_token_id {output_token_id} is outside vocabulary size {logits_shape[2]}.")
        output_token, output_probability = None, None
    output_coordinate = {
        "batch": 0, "sequence_position": position,
        "vocabulary_token_id": int(output_token_id),
        "token": output_token, "probability": output_probability,
    }
    result = trace_tensor_path(
        graph, source="input_ids", target="model_output:logits",
        input_coordinate=input_coordinate, output_coordinate=output_coordinate,
    )
    result["query"].update({
        "pass_name": pass_name, "input_text": trace_report.get("input", {}).get("text"),
        "input_tokens": input_tokens, "input_token_ids": input_ids,
    })
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Query paths in a captured NCIFM runtime graph.")
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--runtime-graphs", required=True)
    parser.add_argument("--pass", dest="pass_name", default="prompt_prefill")
    parser.add_argument("--input-position", type=int, default=None)
    parser.add_argument("--output-position", type=int, default=None)
    parser.add_argument("--output-token-id", type=int, default=None)
    parser.add_argument("--output-token-rank", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trace_report = json.loads(Path(args.trace_report).read_text(encoding="utf-8"))
    runtime_graphs = json.loads(Path(args.runtime_graphs).read_text(encoding="utf-8"))
    result = build_query_from_reports(
        trace_report, runtime_graphs, pass_name=args.pass_name,
        input_position=args.input_position, output_position=args.output_position,
        output_token_id=args.output_token_id, output_token_rank=args.output_token_rank,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "path_report": str(output_path), "query": result["query"],
        "summary": result["summary"], "shortest_route": result["shortest_route"],
    }, indent=2))

if __name__ == "__main__":
    main()
