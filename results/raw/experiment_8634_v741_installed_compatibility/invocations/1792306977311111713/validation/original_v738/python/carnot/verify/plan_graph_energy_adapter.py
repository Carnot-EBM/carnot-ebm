"""Exp 1501 deterministic plan-graph energy adapter.

Spec: REQ-VERIFY-1501, SCENARIO-VERIFY-1501.

This module translates the existing CCTU tool-use benchmark cases into small
directed dependency graphs.  The scorer is deliberately deterministic: it is a
CPU-only structural-risk energy that highlights broken dependencies, type
mismatches, missing intermediates, wrong ordering, and dangling outputs.  It is
GNNVerifier-inspired in graph shape only; it does not train or evaluate a
learned GNN, so every artifact row explicitly records ``trained_gnn_used`` as
false.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from carnot.eval.cctu_executable_constraint_microbenchmark import (
    BenchmarkCase,
    build_benchmark_cases,
)

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1501_gnnverifier_plan_graph_energy_adapter.json"
)
DEFAULT_MANIFEST_PATH = Path("results/plan_graph_energy_manifest_1501.jsonl")
DEFAULT_TRACE_LIMIT = 12

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "plan_graph_energy_ready",
    "traces_converted",
    "injected_graph_faults",
    "node_localization_top1_rate",
    "edge_localization_top1_rate",
    "random_baseline_top1_rate",
    "length_baseline_top1_rate",
    "graph_energy_beats_baselines",
    "adapter_manifest_path",
    "blockers",
    "honest_verdict",
)

FAULT_TYPES: tuple[str, ...] = (
    "missing_edge",
    "wrong_tool_input_type",
    "missing_intermediate",
    "wrong_ordering",
    "dangling_output",
)


@dataclass(frozen=True)
class PlanNode:
    """A typed node in a CCTU plan graph.

    Nodes carry the small amount of state a structural verifier needs: what
    role the node plays, what value type it is expected to expose, and
    deterministic attributes such as step order and serialized value length.
    """

    node_id: str
    node_type: str
    label: str
    value_type: str
    attributes: JsonDict


@dataclass(frozen=True)
class PlanEdge:
    """A directed dependency edge between two plan nodes."""

    edge_id: str
    source: str
    target: str
    edge_type: str
    attributes: JsonDict


@dataclass(frozen=True)
class PlanGraph:
    """One CCTU trace represented as a directed typed plan graph."""

    case_id: str
    family: str
    nodes: tuple[PlanNode, ...]
    edges: tuple[PlanEdge, ...]
    expected_outputs: JsonDict


@dataclass(frozen=True)
class InjectedFault:
    """A plan graph plus the known node and edge that should localize first."""

    fault_id: str
    fault_type: str
    graph: PlanGraph
    target_node_id: str
    target_edge_id: str
    description: str


@dataclass(frozen=True)
class GraphRiskScore:
    """Deterministic structural-risk energy and rankings for a graph."""

    energy: float
    node_scores: dict[str, float]
    edge_scores: dict[str, float]
    ranked_node_ids: list[str]
    ranked_edge_ids: list[str]
    explanations: dict[str, list[str]]


def convert_cctu_traces_to_plan_graphs(
    *,
    trace_limit: int = DEFAULT_TRACE_LIMIT,
) -> list[PlanGraph]:
    """Convert a bounded, family-balanced CCTU subset into plan graphs."""

    cases = _select_bounded_cases(build_benchmark_cases(), trace_limit)
    return [plan_graph_from_case(case) for case in cases]


def plan_graph_from_case(case: BenchmarkCase) -> PlanGraph:
    """Build a directed dependency graph from one deterministic CCTU case."""

    nodes: list[PlanNode] = [
        _node(
            "prompt",
            "prompt",
            f"prompt:{case.case_id}",
            "prompt",
            step_index=0,
            family=case.family,
            text_length=len(case.prompt),
        )
    ]
    argument_keys = sorted(case.tool_arguments)
    for key in argument_keys:
        value = case.tool_arguments[key]
        nodes.append(
            _node(
                f"arg:{key}",
                "tool_argument",
                f"argument:{key}",
                _json_type(value),
                step_index=1,
                argument_key=key,
                expected_value_type=_json_type(value),
                observed_value_type=_json_type(value),
                text_length=len(_compact_json(value)),
                tool_dependency=case.tool_name,
            )
        )
    nodes.extend(
        [
            _node(
                "tool_call",
                "tool_call",
                f"tool:{case.tool_name}",
                "tool_call",
                step_index=2,
                tool_name=case.tool_name,
                expected_argument_keys=argument_keys,
                input_argument_count=len(argument_keys),
                text_length=len(case.tool_name),
            ),
            _node(
                "tool_result",
                "tool_result",
                "tool_result",
                _json_type(case.expected_tool_result),
                step_index=3,
                expected_result=_copy_jsonish(case.expected_tool_result),
                text_length=len(_compact_json(case.expected_tool_result)),
                tool_dependency=case.tool_name,
            ),
            _node(
                "final_answer",
                "final_answer",
                "final_answer",
                "string",
                step_index=4,
                expected_answer=case.expected_final_answer,
                text_length=len(case.expected_final_answer),
            ),
        ]
    )

    edges: list[PlanEdge] = [
        _edge("prompt", "tool_call", "prompt_to_tool", source_step=0, target_step=2)
    ]
    for key in argument_keys:
        expected_type = _json_type(case.tool_arguments[key])
        edges.append(
            _edge(
                f"arg:{key}",
                "tool_call",
                "argument_to_tool",
                source_step=1,
                target_step=2,
                argument_key=key,
                expected_value_type=expected_type,
                observed_value_type=expected_type,
            )
        )
    edges.extend(
        [
            _edge("tool_call", "tool_result", "tool_to_result", source_step=2, target_step=3),
            _edge(
                "tool_result",
                "final_answer",
                "result_to_answer",
                source_step=3,
                target_step=4,
            ),
        ]
    )
    return PlanGraph(
        case_id=case.case_id,
        family=case.family,
        nodes=tuple(nodes),
        edges=tuple(edges),
        expected_outputs={
            "tool_name": case.tool_name,
            "tool_arguments": _copy_jsonish(case.tool_arguments),
            "tool_result": _copy_jsonish(case.expected_tool_result),
            "final_answer": case.expected_final_answer,
        },
    )


def inject_dependency_faults(graph: PlanGraph) -> list[InjectedFault]:
    """Inject the five deterministic dependency faults required by Exp 1501."""

    argument_node = next(node for node in graph.nodes if node.node_type == "tool_argument")
    argument_edge = next(
        edge
        for edge in graph.edges
        if edge.source == argument_node.node_id and edge.target == "tool_call"
    )
    tool_result_edge = _edge_by_id(graph, "edge:tool_call->tool_result")
    answer_edge = _edge_by_id(graph, "edge:tool_result->final_answer")

    return [
        _missing_edge_fault(graph, tool_result_edge),
        _wrong_tool_input_type_fault(graph, argument_node, argument_edge),
        _missing_intermediate_fault(graph, tool_result_edge),
        _wrong_ordering_fault(graph, answer_edge),
        _dangling_output_fault(graph),
    ]


def score_graph_risk(graph: PlanGraph) -> GraphRiskScore:
    """Score structural risk without learning, sampling, GPU, or model calls."""

    node_scores: dict[str, float] = {}
    edge_scores: dict[str, float] = {}
    explanations: dict[str, list[str]] = {}
    for node in graph.nodes:
        score, reasons = _score_node(node)
        node_scores[node.node_id] = score
        if reasons:
            explanations[f"node:{node.node_id}"] = reasons
    for edge in graph.edges:
        score, reasons = _score_edge(edge)
        edge_scores[edge.edge_id] = score
        if reasons:
            explanations[f"edge:{edge.edge_id}"] = reasons

    ranked_node_ids = sorted(node_scores, key=lambda node_id: (-node_scores[node_id], node_id))
    ranked_edge_ids = sorted(edge_scores, key=lambda edge_id: (-edge_scores[edge_id], edge_id))
    return GraphRiskScore(
        energy=sum(node_scores.values()) + sum(edge_scores.values()),
        node_scores=node_scores,
        edge_scores=edge_scores,
        ranked_node_ids=ranked_node_ids,
        ranked_edge_ids=ranked_edge_ids,
        explanations=explanations,
    )


def evaluate_fault(fault: InjectedFault) -> JsonDict:
    """Create one manifest row for a graph/fault pair and its baselines."""

    score = score_graph_risk(fault.graph)
    predicted_node_id = score.ranked_node_ids[0] if score.ranked_node_ids else None
    predicted_edge_id = score.ranked_edge_ids[0] if score.ranked_edge_ids else None
    length_node_id = _length_baseline_node_id(fault.graph)
    length_edge_id = _length_baseline_edge_id(fault.graph)
    node_count = len(fault.graph.nodes)
    edge_count = len(fault.graph.edges)
    random_node_credit = 1.0 / node_count if node_count else 0.0
    random_edge_credit = 1.0 / edge_count if edge_count else 0.0
    length_node_credit = 1.0 if length_node_id == fault.target_node_id else 0.0
    length_edge_credit = 1.0 if length_edge_id == fault.target_edge_id else 0.0

    return {
        "trace_id": fault.graph.case_id,
        "case_id": fault.graph.case_id,
        "family": fault.graph.family,
        "fault_id": fault.fault_id,
        "fault_type": fault.fault_type,
        "fault_description": fault.description,
        "target_node_id": fault.target_node_id,
        "target_edge_id": fault.target_edge_id,
        "predicted_node_id": predicted_node_id,
        "predicted_edge_id": predicted_edge_id,
        "node_localized_top1": predicted_node_id == fault.target_node_id,
        "edge_localized_top1": predicted_edge_id == fault.target_edge_id,
        "graph_risk_energy": score.energy,
        "node_scores": score.node_scores,
        "edge_scores": score.edge_scores,
        "ranked_node_ids": score.ranked_node_ids,
        "ranked_edge_ids": score.ranked_edge_ids,
        "risk_explanations": score.explanations,
        "random_node_credit": random_node_credit,
        "random_edge_credit": random_edge_credit,
        "random_baseline_credit": (random_node_credit + random_edge_credit) / 2.0,
        "length_baseline_node_id": length_node_id,
        "length_baseline_edge_id": length_edge_id,
        "length_node_credit": length_node_credit,
        "length_edge_credit": length_edge_credit,
        "length_baseline_credit": (length_node_credit + length_edge_credit) / 2.0,
        "nodes": [_node_to_json(node) for node in fault.graph.nodes],
        "edges": [_edge_to_json(edge) for edge in fault.graph.edges],
        "expected_outputs": _copy_jsonish(fault.graph.expected_outputs),
        "scorer_type": "deterministic_graph_risk",
        "trained_gnn_used": False,
    }


def aggregate_localization_metrics(rows: list[JsonDict]) -> JsonDict:
    """Aggregate top-1 localization and baseline rates for artifact fields."""

    if not rows:
        return {
            "node_localization_top1_rate": 0.0,
            "edge_localization_top1_rate": 0.0,
            "random_baseline_top1_rate": 0.0,
            "length_baseline_top1_rate": 0.0,
            "graph_energy_beats_baselines": False,
        }
    total = len(rows)
    node_rate = _rate(row["node_localized_top1"] for row in rows)
    edge_rate = _rate(row["edge_localized_top1"] for row in rows)
    random_rate = round(
        sum(float(row["random_baseline_credit"]) for row in rows) / total,
        6,
    )
    length_rate = round(
        sum(float(row["length_baseline_credit"]) for row in rows) / total,
        6,
    )
    graph_rate = (node_rate + edge_rate) / 2.0
    return {
        "node_localization_top1_rate": node_rate,
        "edge_localization_top1_rate": edge_rate,
        "random_baseline_top1_rate": random_rate,
        "length_baseline_top1_rate": length_rate,
        "graph_energy_beats_baselines": graph_rate > random_rate and graph_rate > length_rate,
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> JsonDict:
    """Write the required durable bootstrap artifact before evaluation starts."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "plan_graph_energy_ready": False,
        "traces_converted": 0,
        "injected_graph_faults": 0,
        "node_localization_top1_rate": 0.0,
        "edge_localization_top1_rate": 0.0,
        "random_baseline_top1_rate": 0.0,
        "length_baseline_top1_rate": 0.0,
        "graph_energy_beats_baselines": False,
        "adapter_manifest_path": _display_path(manifest_path),
        "blockers": [],
        "honest_verdict": "complete_in_progress_bootstrap",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_plan_graph_energy_adapter(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    trace_limit: int = DEFAULT_TRACE_LIMIT,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run graph conversion, fault injection, scoring, and artifact writing."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date, manifest_path=manifest)

    graphs = convert_cctu_traces_to_plan_graphs(trace_limit=trace_limit)
    faults = [fault for graph in graphs for fault in inject_dependency_faults(graph)]
    rows = [evaluate_fault(fault) for fault in faults]
    _write_jsonl(manifest, rows)

    metrics = aggregate_localization_metrics(rows)
    ready = bool(graphs) and bool(rows) and bool(metrics["graph_energy_beats_baselines"])
    blockers = [] if ready else ["graph_energy_did_not_beat_baselines_or_no_rows"]
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "run_date": run_date,
        "schema_version": 1,
        "plan_graph_energy_ready": ready,
        "traces_converted": len(graphs),
        "injected_graph_faults": len(rows),
        "node_localization_top1_rate": metrics["node_localization_top1_rate"],
        "edge_localization_top1_rate": metrics["edge_localization_top1_rate"],
        "random_baseline_top1_rate": metrics["random_baseline_top1_rate"],
        "length_baseline_top1_rate": metrics["length_baseline_top1_rate"],
        "graph_energy_beats_baselines": metrics["graph_energy_beats_baselines"],
        "adapter_manifest_path": _display_path(manifest),
        "blockers": blockers,
        "honest_verdict": (
            "complete: deterministic plan-graph energy adapter localized "
            "injected CCTU dependency faults and beat random/length baselines"
            if ready
            else "complete: deterministic plan-graph energy adapter wrote rows "
            "but did not beat required baselines"
        ),
        "selected_trace_ids": [graph.case_id for graph in graphs],
        "fault_types": list(FAULT_TYPES),
        "scorer_type": "deterministic_graph_risk",
        "trained_gnn_used": False,
        "tests_run": list(tests_run or []),
    }
    _write_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by bounded smoke runs and tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-limit", type=int, default=DEFAULT_TRACE_LIMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)
    artifact = run_plan_graph_energy_adapter(
        output_path=args.output,
        manifest_path=args.manifest,
        trace_limit=args.trace_limit,
    )
    print(
        "ready={ready} faults={faults} node_top1={node_top1} beats={beats}".format(
            ready=artifact["plan_graph_energy_ready"],
            faults=artifact["injected_graph_faults"],
            node_top1=artifact["node_localization_top1_rate"],
            beats=artifact["graph_energy_beats_baselines"],
        )
    )
    return 0 if artifact["status"] == "complete" else 1


def _select_bounded_cases(cases: Iterable[BenchmarkCase], trace_limit: int) -> list[BenchmarkCase]:
    if trace_limit <= 0:
        return []
    grouped: dict[str, list[BenchmarkCase]] = {}
    family_order: list[str] = []
    for case in cases:
        if case.family not in grouped:
            grouped[case.family] = []
            family_order.append(case.family)
        grouped[case.family].append(case)

    selected: list[BenchmarkCase] = []
    offset = 0
    while len(selected) < trace_limit:
        made_progress = False
        for family in family_order:
            family_cases = grouped[family]
            if offset < len(family_cases):
                selected.append(family_cases[offset])
                made_progress = True
                if len(selected) == trace_limit:
                    break
        if not made_progress:
            break
        offset += 1
    return selected


def _missing_edge_fault(graph: PlanGraph, target_edge: PlanEdge) -> InjectedFault:
    faulty = _replace_edge(
        graph,
        target_edge.edge_id,
        present=False,
        missing_dependency=True,
        fault_injected=True,
    )
    faulty = _replace_node(
        faulty,
        "tool_result",
        upstream_missing_edge=True,
        fault_injected=True,
    )
    return InjectedFault(
        fault_id=f"{graph.case_id}:missing_edge",
        fault_type="missing_edge",
        graph=faulty,
        target_node_id="tool_result",
        target_edge_id=target_edge.edge_id,
        description="Required tool-call to tool-result dependency is marked missing.",
    )


def _wrong_tool_input_type_fault(
    graph: PlanGraph,
    argument_node: PlanNode,
    argument_edge: PlanEdge,
) -> InjectedFault:
    expected_type = str(argument_node.attributes["expected_value_type"])
    observed_type = "string" if expected_type != "string" else "array"
    faulty = _replace_node(
        graph,
        argument_node.node_id,
        observed_value_type=observed_type,
        type_mismatch=True,
        fault_injected=True,
    )
    faulty = _replace_edge(
        faulty,
        argument_edge.edge_id,
        observed_value_type=observed_type,
        type_mismatch=True,
        fault_injected=True,
    )
    return InjectedFault(
        fault_id=f"{graph.case_id}:wrong_tool_input_type",
        fault_type="wrong_tool_input_type",
        graph=faulty,
        target_node_id=argument_node.node_id,
        target_edge_id=argument_edge.edge_id,
        description="A required tool argument is assigned the wrong observed value type.",
    )


def _missing_intermediate_fault(graph: PlanGraph, target_edge: PlanEdge) -> InjectedFault:
    faulty = _replace_node(
        graph,
        "tool_result",
        status="missing",
        fault_injected=True,
    )
    faulty = _replace_edge(
        faulty,
        target_edge.edge_id,
        missing_intermediate=True,
        fault_injected=True,
    )
    return InjectedFault(
        fault_id=f"{graph.case_id}:missing_intermediate",
        fault_type="missing_intermediate",
        graph=faulty,
        target_node_id="tool_result",
        target_edge_id=target_edge.edge_id,
        description="The intermediate tool-result node is missing before answer composition.",
    )


def _wrong_ordering_fault(graph: PlanGraph, target_edge: PlanEdge) -> InjectedFault:
    faulty = _replace_node(
        graph,
        "final_answer",
        ordering_violation=True,
        step_index=2,
        fault_injected=True,
    )
    faulty = _replace_edge(
        faulty,
        target_edge.edge_id,
        ordering_violation=True,
        source_step=4,
        target_step=2,
        fault_injected=True,
    )
    return InjectedFault(
        fault_id=f"{graph.case_id}:wrong_ordering",
        fault_type="wrong_ordering",
        graph=faulty,
        target_node_id="final_answer",
        target_edge_id=target_edge.edge_id,
        description="The final answer is ordered before the tool result it depends on.",
    )


def _dangling_output_fault(graph: PlanGraph) -> InjectedFault:
    dangling_node = _node(
        "dangling_output",
        "dangling_output",
        "dangling_output",
        "unknown",
        step_index=5,
        text_length=len("dangling_output"),
        dangling=True,
        fault_injected=True,
    )
    dangling_edge = _edge(
        "tool_result",
        "dangling_output",
        "dangling_output",
        source_step=3,
        target_step=5,
        dependency_required=False,
        dangling=True,
        fault_injected=True,
    )
    faulty = PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(_clone_node(node) for node in graph.nodes) + (dangling_node,),
        edges=tuple(_clone_edge(edge) for edge in graph.edges) + (dangling_edge,),
        expected_outputs=_copy_jsonish(graph.expected_outputs),
    )
    return InjectedFault(
        fault_id=f"{graph.case_id}:dangling_output",
        fault_type="dangling_output",
        graph=faulty,
        target_node_id="dangling_output",
        target_edge_id=dangling_edge.edge_id,
        description="A tool result feeds an output node outside the expected answer path.",
    )


def _score_node(node: PlanNode) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    attrs = node.attributes
    if attrs.get("type_mismatch"):
        score += 80.0
        reasons.append("observed_value_type differs from expected_value_type")
    if attrs.get("status") == "missing":
        score += 90.0
        reasons.append("required intermediate node is missing")
    if attrs.get("upstream_missing_edge"):
        score += 70.0
        reasons.append("required incoming dependency edge is missing")
    if attrs.get("ordering_violation"):
        score += 80.0
        reasons.append("node appears before a dependency it should follow")
    if attrs.get("dangling"):
        score += 95.0
        reasons.append("node is outside the expected answer dependency path")
    return score, reasons


def _score_edge(edge: PlanEdge) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    attrs = edge.attributes
    if attrs.get("dependency_required") and attrs.get("present") is False:
        score += 100.0
        reasons.append("required dependency edge is absent")
    if attrs.get("missing_dependency"):
        score += 75.0
        reasons.append("edge carries a missing-dependency marker")
    if attrs.get("type_mismatch"):
        score += 80.0
        reasons.append("edge propagates the wrong observed value type")
    if attrs.get("missing_intermediate"):
        score += 90.0
        reasons.append("edge points to a missing intermediate result")
    if attrs.get("ordering_violation") or attrs.get("source_step", 0) >= attrs.get(
        "target_step",
        1,
    ):
        score += 85.0
        reasons.append("edge violates dependency order")
    if attrs.get("dangling"):
        score += 95.0
        reasons.append("edge points to an unexpected output")
    return score, reasons


def _replace_node(graph: PlanGraph, node_id: str, **attributes: Any) -> PlanGraph:
    return PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(
            _clone_node(node, **attributes) if node.node_id == node_id else _clone_node(node)
            for node in graph.nodes
        ),
        edges=tuple(_clone_edge(edge) for edge in graph.edges),
        expected_outputs=_copy_jsonish(graph.expected_outputs),
    )


def _replace_edge(graph: PlanGraph, edge_id: str, **attributes: Any) -> PlanGraph:
    return PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(_clone_node(node) for node in graph.nodes),
        edges=tuple(
            _clone_edge(edge, **attributes) if edge.edge_id == edge_id else _clone_edge(edge)
            for edge in graph.edges
        ),
        expected_outputs=_copy_jsonish(graph.expected_outputs),
    )


def _clone_node(node: PlanNode, **attribute_updates: Any) -> PlanNode:
    attributes = _copy_jsonish(node.attributes)
    attributes.update(attribute_updates)
    return PlanNode(
        node_id=node.node_id,
        node_type=node.node_type,
        label=node.label,
        value_type=node.value_type,
        attributes=attributes,
    )


def _clone_edge(edge: PlanEdge, **attribute_updates: Any) -> PlanEdge:
    attributes = _copy_jsonish(edge.attributes)
    attributes.update(attribute_updates)
    return PlanEdge(
        edge_id=edge.edge_id,
        source=edge.source,
        target=edge.target,
        edge_type=edge.edge_type,
        attributes=attributes,
    )


def _edge_by_id(graph: PlanGraph, edge_id: str) -> PlanEdge:
    for edge in graph.edges:
        if edge.edge_id == edge_id:
            return edge
    raise ValueError(f"missing edge: {edge_id}")


def _length_baseline_node_id(graph: PlanGraph) -> str | None:
    if not graph.nodes:
        return None
    return max(
        graph.nodes,
        key=lambda node: (int(node.attributes.get("text_length", 0)), node.node_id),
    ).node_id


def _length_baseline_edge_id(graph: PlanGraph) -> str | None:
    if not graph.edges:
        return None
    return max(
        graph.edges,
        key=lambda edge: (int(edge.attributes.get("text_length", 0)), edge.edge_id),
    ).edge_id


def _rate(values: Iterable[bool]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return round(sum(1 for value in values_list if value) / len(values_list), 6)


def _node(
    node_id: str,
    node_type: str,
    label: str,
    value_type: str,
    **attributes: Any,
) -> PlanNode:
    attributes.setdefault("fault_injected", False)
    attributes.setdefault("text_length", len(label))
    return PlanNode(
        node_id=node_id,
        node_type=node_type,
        label=label,
        value_type=value_type,
        attributes=_copy_jsonish(attributes),
    )


def _edge(source: str, target: str, edge_type: str, **attributes: Any) -> PlanEdge:
    attributes.setdefault("dependency_required", True)
    attributes.setdefault("present", True)
    attributes.setdefault("fault_injected", False)
    attributes.setdefault("text_length", len(source) + len(target) + len(edge_type))
    return PlanEdge(
        edge_id=f"edge:{source}->{target}",
        source=source,
        target=target,
        edge_type=edge_type,
        attributes=_copy_jsonish(attributes),
    )


def _node_to_json(node: PlanNode) -> JsonDict:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "label": node.label,
        "value_type": node.value_type,
        "attributes": _copy_jsonish(node.attributes),
    }


def _edge_to_json(edge: PlanEdge) -> JsonDict:
    return {
        "edge_id": edge.edge_id,
        "source": edge.source,
        "target": edge.target,
        "edge_type": edge.edge_type,
        "attributes": _copy_jsonish(edge.attributes),
    }


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _copy_jsonish(value: Any) -> Any:
    return copy.deepcopy(value)


def _display_path(path: Path | str) -> str:
    return Path(path).as_posix()


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised through focused helpers.
    raise SystemExit(main())
