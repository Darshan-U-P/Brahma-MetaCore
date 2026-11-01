"""CLI for focused NCIFM source-to-target coordinate path searches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.metaobserver.coordinate_tracer import CoordinateTraceError
from src.metaobserver.focused_path_tracer import trace_focused_coordinate_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find a focused scalar-coordinate dependency path in an NCIFM runtime graph."
    )
    parser.add_argument("--trace-report", required=True)
    parser.add_argument("--runtime-graphs", required=True)
    parser.add_argument("--pass", dest="pass_name", default="prompt_prefill")
    parser.add_argument("--source-input", default="input_ids")
    parser.add_argument("--input-position", type=int, required=True)
    parser.add_argument("--output-name", default="logits")
    parser.add_argument("--output-position", type=int, required=True)
    parser.add_argument("--output-token-id", type=int, required=True)
    parser.add_argument("--max-depth", type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=50_000)
    parser.add_argument("--max-candidates", type=int, default=100_000)
    parser.add_argument("--max-frontier", type=int, default=10_000)
    parser.add_argument("--parameter-policy", choices=("compact", "expand"), default="compact")
    parser.add_argument(
        "--search-strategy",
        choices=("source_guided", "breadth_first"),
        default="source_guided",
        help="Prioritize coordinates near the requested source, or use FIFO breadth-first order.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_report = json.loads(Path(args.trace_report).read_text(encoding="utf-8"))
    runtime_graphs = json.loads(Path(args.runtime_graphs).read_text(encoding="utf-8"))
    if args.pass_name not in runtime_graphs.get("passes", {}):
        raise CoordinateTraceError(f"Execution pass {args.pass_name!r} was not found.")
    graph = runtime_graphs["passes"][args.pass_name]
    pass_record = trace_report.get("runtime_trace", {}).get("passes", {}).get(args.pass_name)
    if pass_record is None:
        raise CoordinateTraceError(f"Trace report does not contain pass {args.pass_name!r}.")

    source_node = next(
        (node for node in graph.get("nodes", [])
         if node.get("kind") == "model_input" and node.get("label") == args.source_input),
        None,
    )
    target_id = f"model_output:{args.output_name}"
    target_node = next(
        (node for node in graph.get("nodes", []) if node.get("id") == target_id),
        None,
    )
    if source_node is None:
        raise CoordinateTraceError(f"Model input {args.source_input!r} was not captured.")
    if target_node is None:
        raise CoordinateTraceError(f"Model output {target_id!r} was not captured.")
    if len(source_node.get("shape", [])) != 2 or source_node["shape"][0] != 1:
        raise CoordinateTraceError("CLI currently expects a batch-size-1 rank-2 token input tensor.")
    if len(target_node.get("shape", [])) != 3 or target_node["shape"][0] != 1:
        raise CoordinateTraceError("CLI currently expects batch-size-1 logits [1, sequence, vocabulary].")
    input_ids = pass_record.get("input_token_ids")
    if input_ids is None:
        raise CoordinateTraceError("Trace report has no saved input_token_ids for lookup resolution.")
    if args.input_position < 0 or args.input_position >= source_node["shape"][1]:
        raise CoordinateTraceError("Input position is outside the captured input sequence.")
    if args.output_position < 0 or args.output_position >= target_node["shape"][1]:
        raise CoordinateTraceError("Output position is outside the logits sequence.")
    if args.output_token_id < 0 or args.output_token_id >= target_node["shape"][2]:
        raise CoordinateTraceError("Output token ID is outside the logits vocabulary.")

    result = trace_focused_coordinate_path(
        graph,
        source_node["id"],
        [0, args.input_position],
        target_id,
        [0, args.output_position, args.output_token_id],
        tensor_values={source_node["id"]: [input_ids]},
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        max_candidates=args.max_candidates,
        max_frontier=args.max_frontier,
        parameter_policy=args.parameter_policy,
        search_strategy=args.search_strategy,
    )
    result["query"].update({
        "pass_name": args.pass_name,
        "input_text": trace_report.get("input", {}).get("text"),
        "input_token_id": input_ids[args.input_position],
        "output_token_id": args.output_token_id,
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "query": result["query"],
        "result": result["result"],
        "summary": result["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
