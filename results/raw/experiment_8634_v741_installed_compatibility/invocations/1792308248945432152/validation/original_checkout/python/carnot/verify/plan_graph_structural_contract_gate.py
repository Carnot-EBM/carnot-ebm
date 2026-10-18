"""Exp 1510 deterministic plan-graph structural contract gate.

Spec: REQ-VERIFY-1510, SCENARIO-VERIFY-1510.

This module turns the plan-graph signal from Exp 1501 into a pre-execution
gate.  The checks are intentionally small and exact: they verify that a tool
plan has the prerequisite nodes, the acquisition path from tool call to final
answer, monotonic dependency ordering, required argument acquisition, and no
known incompatible API sequence before any tool would be invoked.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from carnot.verify.plan_graph_energy_adapter import (
    PlanEdge,
    PlanGraph,
    PlanNode,
    convert_cctu_traces_to_plan_graphs,
)

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
CONTRACT_SCHEMA_VERSION = "plan-graph-structural-contract/v1"
DEFAULT_TRACE_LIMIT = 12
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1510_plan_graph_structural_contract_gate.json")
DEFAULT_MANIFEST_PATH = Path("results/plan_graph_structural_contracts_1510.jsonl")
DEFAULT_EXP1501_ARTIFACT_PATH = Path(
    "results/experiment_1501_gnnverifier_plan_graph_energy_adapter.json"
)
DEFAULT_EXP1509_EVENT_MANIFEST_PATH = Path("results/executable_monitor_events_1509.jsonl")

CONTRACT_FAMILIES: tuple[str, ...] = (
    "graph_prerequisites",
    "acquisition_path",
    "tool_ordering",
    "required_object_acquisition",
    "incompatible_api_use",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "structural_contract_gate_ready",
    "plan_graphs_checked",
    "contracts_defined",
    "violations_injected",
    "violations_detected",
    "false_accept_rate",
    "false_reject_rate",
    "random_baseline_detection_rate",
    "length_baseline_detection_rate",
    "contract_manifest_path",
    "blockers",
    "honest_verdict",
)

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "contract_schema_version",
    "graph_id",
    "case_id",
    "family",
    "graph_label",
    "contract_id",
    "contract_family",
    "expected_violation",
    "detected_violation",
    "classifier_outcome",
    "random_baseline_detected",
    "length_baseline_detected",
    "contract_evidence",
)


@dataclass(frozen=True)
class StructuralContract:
    """A compact, serializable contract definition for one graph family.

    Each contract stores only the facts needed by deterministic checks.  The
    executable logic stays in this module so the artifact can be inspected
    without accepting arbitrary code as part of the contract.
    """

    contract_id: str
    contract_family: str
    description: str
    required_node_types: tuple[str, ...] = ()
    required_edge_types: tuple[str, ...] = ()
    required_path_edges: tuple[tuple[str, str], ...] = ()
    incompatible_operations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class GateInputs:
    """Loaded Exp 1501 graphs plus optional Exp 1509 runtime-event counts."""

    graphs: list[PlanGraph]
    runtime_events_loaded: int
    runtime_event_counts_by_case: dict[str, int]
    blockers: list[str]


@dataclass(frozen=True)
class InjectedContractViolation:
    """A graph mutated to violate exactly one structural contract family."""

    violation_id: str
    violation_type: str
    contract_family: str
    target_contract_id: str
    graph: PlanGraph
    description: str


def define_structural_contracts() -> tuple[StructuralContract, ...]:
    """Return the deterministic Exp 1510 contract schema."""

    return (
        StructuralContract(
            contract_id="contract:graph_prerequisites",
            contract_family="graph_prerequisites",
            description="Core prompt, tool, result, answer nodes and prompt-to-tool edge exist.",
            required_node_types=("prompt", "tool_call", "tool_result", "final_answer"),
            required_edge_types=("prompt_to_tool",),
        ),
        StructuralContract(
            contract_id="contract:acquisition_path",
            contract_family="acquisition_path",
            description="The tool result is acquired and then consumed by the final answer.",
            required_path_edges=(("tool_call", "tool_result"), ("tool_result", "final_answer")),
        ),
        StructuralContract(
            contract_id="contract:tool_ordering",
            contract_family="tool_ordering",
            description="Every present dependency edge points from an earlier step to a later step.",
        ),
        StructuralContract(
            contract_id="contract:required_object_acquisition",
            contract_family="required_object_acquisition",
            description="Every declared tool argument is acquired by the tool call before use.",
            required_edge_types=("argument_to_tool",),
        ),
        StructuralContract(
            contract_id="contract:incompatible_api_use",
            contract_family="incompatible_api_use",
            description="Known incompatible operations cannot target the same acquired object.",
            incompatible_operations=(
                ("delete", "read_after_delete"),
                ("finalize", "write_after_finalize"),
            ),
        ),
    )


def contract_to_json(contract: StructuralContract) -> JsonDict:
    """Serialize one contract definition for artifacts and tests."""

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_family": contract.contract_family,
        "description": contract.description,
        "required_node_types": list(contract.required_node_types),
        "required_edge_types": list(contract.required_edge_types),
        "required_path_edges": [list(edge) for edge in contract.required_path_edges],
        "incompatible_operations": [list(pair) for pair in contract.incompatible_operations],
    }


def load_gate_inputs(
    *,
    exp1501_artifact_path: Path | str = DEFAULT_EXP1501_ARTIFACT_PATH,
    exp1509_event_manifest_path: Path | str = DEFAULT_EXP1509_EVENT_MANIFEST_PATH,
    trace_limit: int = DEFAULT_TRACE_LIMIT,
) -> GateInputs:
    """Load Exp 1501 plan graph cases and optional normalized Exp 1509 events."""

    blockers: list[str] = []
    exp1501_path = Path(exp1501_artifact_path)
    exp1509_path = Path(exp1509_event_manifest_path)
    exp1501_artifact = _load_json_if_exists(exp1501_path)

    if exp1501_artifact is None:
        blockers.append(f"missing_exp1501_artifact:{exp1501_path}")
        selected_trace_ids: list[str] = []
    elif (
        exp1501_artifact.get("status") != "complete"
        or exp1501_artifact.get("plan_graph_energy_ready") is not True
    ):
        blockers.append(f"exp1501_not_ready:{exp1501_path}")
        selected_trace_ids = []
    else:
        selected_trace_ids = [
            str(trace_id) for trace_id in exp1501_artifact.get("selected_trace_ids", [])
        ]

    graphs = _select_graphs_from_exp1501(selected_trace_ids, trace_limit)
    runtime_events = _read_jsonl(exp1509_path) if exp1509_path.exists() else []
    if not exp1509_path.exists():
        blockers.append(f"missing_exp1509_event_manifest:{exp1509_path}")
    event_counts = _runtime_event_counts(runtime_events)
    if not graphs:
        blockers.append("no_plan_graphs_loaded")

    return GateInputs(
        graphs=graphs,
        runtime_events_loaded=len(runtime_events),
        runtime_event_counts_by_case=event_counts,
        blockers=blockers,
    )


def inject_structural_contract_violations(graph: PlanGraph) -> list[InjectedContractViolation]:
    """Inject one deterministic violation for each structural contract family."""

    argument_edge = next(edge for edge in graph.edges if edge.edge_type == "argument_to_tool")
    return [
        InjectedContractViolation(
            violation_id=f"{graph.case_id}:missing_prerequisite_edge",
            violation_type="missing_prerequisite_edge",
            contract_family="graph_prerequisites",
            target_contract_id="contract:graph_prerequisites",
            graph=_replace_edge(graph, "edge:prompt->tool_call", present=False),
            description="The prompt-to-tool prerequisite edge is absent before execution.",
        ),
        InjectedContractViolation(
            violation_id=f"{graph.case_id}:broken_acquisition_path",
            violation_type="broken_acquisition_path",
            contract_family="acquisition_path",
            target_contract_id="contract:acquisition_path",
            graph=_replace_edge(graph, "edge:tool_call->tool_result", present=False),
            description="The tool call no longer acquires the tool result.",
        ),
        InjectedContractViolation(
            violation_id=f"{graph.case_id}:tool_ordering_violation",
            violation_type="tool_ordering_violation",
            contract_family="tool_ordering",
            target_contract_id="contract:tool_ordering",
            graph=_replace_node(
                _replace_edge(
                    graph,
                    "edge:tool_result->final_answer",
                    source_step=3,
                    target_step=2,
                ),
                "final_answer",
                step_index=2,
            ),
            description="The final answer is ordered before the result it depends on.",
        ),
        InjectedContractViolation(
            violation_id=f"{graph.case_id}:missing_required_object_acquisition",
            violation_type="missing_required_object_acquisition",
            contract_family="required_object_acquisition",
            target_contract_id="contract:required_object_acquisition",
            graph=_replace_edge(graph, argument_edge.edge_id, present=False),
            description="A required tool argument is not acquired by the tool call.",
        ),
        InjectedContractViolation(
            violation_id=f"{graph.case_id}:incompatible_api_operations",
            violation_type="incompatible_api_operations",
            contract_family="incompatible_api_use",
            target_contract_id="contract:incompatible_api_use",
            graph=_append_api_conflict(graph),
            description="The plan reads an object after scheduling a delete on the same object.",
        ),
    ]


def evaluate_graph_contracts(
    graph: PlanGraph,
    contracts: Iterable[StructuralContract] | None = None,
    *,
    graph_label: str = "known_good",
    graph_index: int = 0,
    injected_violation: InjectedContractViolation | None = None,
    runtime_event_count: int = 0,
) -> list[JsonDict]:
    """Evaluate one graph against all structural contracts."""

    contract_list = tuple(contracts or define_structural_contracts())
    random_family = _random_baseline_family(graph.case_id, graph_index, contract_list)
    length_family = _length_baseline_family(graph, contract_list)
    rows: list[JsonDict] = []
    for contract in contract_list:
        result = _evaluate_contract(graph, contract)
        expected_violation = bool(
            injected_violation and contract.contract_family == injected_violation.contract_family
        )
        detected = bool(result["detected_violation"])
        rows.append(
            {
                "contract_schema_version": CONTRACT_SCHEMA_VERSION,
                "graph_id": f"{graph.case_id}:{graph_label}:{graph_index}",
                "case_id": graph.case_id,
                "family": graph.family,
                "graph_label": graph_label,
                "graph_index": graph_index,
                "contract_id": contract.contract_id,
                "contract_family": contract.contract_family,
                "expected_violation": expected_violation,
                "detected_violation": detected,
                "violation_code": result["violation_code"],
                "violation_codes": result["violation_codes"],
                "classifier_outcome": _classifier_outcome(expected_violation, detected),
                "random_baseline_detected": expected_violation
                and random_family == contract.contract_family,
                "length_baseline_detected": expected_violation
                and length_family == contract.contract_family,
                "injected_violation_id": (
                    injected_violation.violation_id if injected_violation else None
                ),
                "injected_violation_type": (
                    injected_violation.violation_type if injected_violation else None
                ),
                "runtime_event_count": runtime_event_count,
                "contract_evidence": result["contract_evidence"],
            }
        )
    return rows


def validate_contract_manifest_row(row: JsonDict) -> list[str]:
    """Return schema validation errors for one manifest row."""

    errors = [f"missing:{field}" for field in REQUIRED_MANIFEST_FIELDS if field not in row]
    if row.get("contract_schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("invalid:contract_schema_version")
    return errors


def aggregate_gate_metrics(rows: list[JsonDict]) -> JsonDict:
    """Aggregate false accepts, false rejects, and baseline detection rates."""

    expected_violation_rows = [row for row in rows if row.get("expected_violation")]
    expected_clean_rows = [row for row in rows if not row.get("expected_violation")]
    false_accepts = [
        row for row in expected_violation_rows if row.get("detected_violation") is not True
    ]
    false_rejects = [row for row in expected_clean_rows if row.get("detected_violation") is True]
    graph_ids = {str(row.get("graph_id")) for row in rows}
    contract_ids = {str(row.get("contract_id")) for row in rows}
    return {
        "plan_graphs_checked": len(graph_ids),
        "contracts_defined": len(contract_ids),
        "violations_injected": len(expected_violation_rows),
        "violations_detected": sum(
            1 for row in expected_violation_rows if row.get("detected_violation") is True
        ),
        "false_accept_rate": _rate(false_accepts, expected_violation_rows),
        "false_reject_rate": _rate(false_rejects, expected_clean_rows),
        "random_baseline_detection_rate": _rate(
            [row for row in expected_violation_rows if row.get("random_baseline_detected")],
            expected_violation_rows,
        ),
        "length_baseline_detection_rate": _rate(
            [row for row in expected_violation_rows if row.get("length_baseline_detected")],
            expected_violation_rows,
        ),
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable in-progress artifact before input loading starts."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "structural_contract_gate_ready": False,
        "plan_graphs_checked": 0,
        "contracts_defined": 0,
        "violations_injected": 0,
        "violations_detected": 0,
        "false_accept_rate": None,
        "false_reject_rate": None,
        "random_baseline_detection_rate": None,
        "length_baseline_detection_rate": None,
        "contract_manifest_path": _display_path(manifest_path),
        "blockers": [],
        "honest_verdict": "complete: in-progress Exp 1510 bootstrap artifact",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_structural_contract_gate(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    exp1501_artifact_path: Path | str = DEFAULT_EXP1501_ARTIFACT_PATH,
    exp1509_event_manifest_path: Path | str = DEFAULT_EXP1509_EVENT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    trace_limit: int = DEFAULT_TRACE_LIMIT,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run graph loading, deterministic contract checks, baselines, and artifacts."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, manifest_path=manifest, run_date=run_date)

    gate_inputs = load_gate_inputs(
        exp1501_artifact_path=exp1501_artifact_path,
        exp1509_event_manifest_path=exp1509_event_manifest_path,
        trace_limit=trace_limit,
    )
    contracts = define_structural_contracts()
    rows: list[JsonDict] = []
    graph_index = 0
    for graph in gate_inputs.graphs:
        runtime_count = gate_inputs.runtime_event_counts_by_case.get(graph.case_id, 0)
        rows.extend(
            evaluate_graph_contracts(
                graph,
                contracts,
                graph_label="known_good",
                graph_index=graph_index,
                runtime_event_count=runtime_count,
            )
        )
        for violation in inject_structural_contract_violations(graph):
            graph_index += 1
            rows.extend(
                evaluate_graph_contracts(
                    violation.graph,
                    contracts,
                    graph_label="injected_violation",
                    graph_index=graph_index,
                    injected_violation=violation,
                    runtime_event_count=runtime_count,
                )
            )
        graph_index += 1

    _write_jsonl(manifest, rows)
    schema_errors = [
        f"{row.get('graph_id', 'unknown')}:{','.join(errors)}"
        for row in rows
        if (errors := validate_contract_manifest_row(row))
    ]
    metrics = aggregate_gate_metrics(rows)
    blockers = [*gate_inputs.blockers, *schema_errors]
    if metrics["false_accept_rate"] is None:
        blockers.append("false_accept_rate_not_reported")
    ready = bool(
        manifest.exists()
        and rows
        and metrics["contracts_defined"] > 0
        and metrics["false_accept_rate"] is not None
        and not blockers
    )
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "run_date": run_date,
        "schema_version": 1,
        "structural_contract_gate_ready": ready,
        "plan_graphs_checked": metrics["plan_graphs_checked"],
        "contracts_defined": metrics["contracts_defined"],
        "violations_injected": metrics["violations_injected"],
        "violations_detected": metrics["violations_detected"],
        "false_accept_rate": metrics["false_accept_rate"],
        "false_reject_rate": metrics["false_reject_rate"],
        "random_baseline_detection_rate": metrics["random_baseline_detection_rate"],
        "length_baseline_detection_rate": metrics["length_baseline_detection_rate"],
        "contract_manifest_path": _display_path(manifest),
        "blockers": blockers,
        "honest_verdict": (
            "complete: deterministic plan-graph structural contract gate rejects "
            "injected pre-execution violations"
            if ready
            else "complete: blocked before structural contract gate readiness"
        ),
        "runtime_events_loaded": gate_inputs.runtime_events_loaded,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_families": list(CONTRACT_FAMILIES),
        "tests_run": list(tests_run or []),
    }
    _write_json(output, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for deterministic Exp 1510 runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-limit", type=int, default=DEFAULT_TRACE_LIMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args(argv)
    artifact = run_structural_contract_gate(
        output_path=args.output,
        manifest_path=args.manifest,
        trace_limit=args.trace_limit,
    )
    print(
        "ready={ready} violations={violations} false_accept={false_accept}".format(
            ready=artifact["structural_contract_gate_ready"],
            violations=artifact["violations_detected"],
            false_accept=artifact["false_accept_rate"],
        )
    )
    return 0 if artifact["status"] == "complete" else 1


def _select_graphs_from_exp1501(selected_trace_ids: list[str], trace_limit: int) -> list[PlanGraph]:
    if trace_limit <= 0:
        return []
    candidate_limit = max(trace_limit, len(selected_trace_ids), DEFAULT_TRACE_LIMIT)
    candidates = convert_cctu_traces_to_plan_graphs(trace_limit=candidate_limit)
    if not selected_trace_ids:
        return candidates[:trace_limit]
    selected = set(selected_trace_ids)
    return [graph for graph in candidates if graph.case_id in selected][:trace_limit]


def _evaluate_contract(graph: PlanGraph, contract: StructuralContract) -> JsonDict:
    if contract.contract_family == "graph_prerequisites":
        return _check_graph_prerequisites(graph, contract)
    if contract.contract_family == "acquisition_path":
        return _check_acquisition_path(graph, contract)
    if contract.contract_family == "tool_ordering":
        return _check_tool_ordering(graph)
    if contract.contract_family == "required_object_acquisition":
        return _check_required_object_acquisition(graph)
    return _check_incompatible_api_use(graph, contract)


def _check_graph_prerequisites(graph: PlanGraph, contract: StructuralContract) -> JsonDict:
    node_types = {node.node_type for node in graph.nodes}
    edge_types = {edge.edge_type for edge in graph.edges if edge.attributes.get("present", True)}
    missing_nodes = [
        node_type for node_type in contract.required_node_types if node_type not in node_types
    ]
    missing_edges = [
        edge_type for edge_type in contract.required_edge_types if edge_type not in edge_types
    ]
    codes = []
    if missing_nodes:
        codes.append("missing_prerequisite_node")
    if missing_edges:
        codes.append("missing_prerequisite_edge")
    return _contract_result(
        codes,
        {
            "node_types": sorted(node_types),
            "present_edge_types": sorted(edge_types),
            "missing_node_types": missing_nodes,
            "missing_edge_types": missing_edges,
        },
    )


def _check_acquisition_path(graph: PlanGraph, contract: StructuralContract) -> JsonDict:
    missing = [
        {"source": source, "target": target}
        for source, target in contract.required_path_edges
        if not _has_present_edge(graph, source, target)
    ]
    return _contract_result(
        ["broken_acquisition_path"] if missing else [],
        {
            "required_path_edges": [list(edge) for edge in contract.required_path_edges],
            "missing": missing,
        },
    )


def _check_tool_ordering(graph: PlanGraph) -> JsonDict:
    nodes = {node.node_id: node for node in graph.nodes}
    violations: list[JsonDict] = []
    for edge in graph.edges:
        if edge.attributes.get("present") is False:
            continue
        source_step = _edge_step(edge, "source_step", nodes.get(edge.source))
        target_step = _edge_step(edge, "target_step", nodes.get(edge.target))
        if source_step >= target_step:
            violations.append(
                {
                    "edge_id": edge.edge_id,
                    "source_step": source_step,
                    "target_step": target_step,
                }
            )
    return _contract_result(
        ["tool_ordering_violation"] if violations else [],
        {"ordering_violations": violations},
    )


def _check_required_object_acquisition(graph: PlanGraph) -> JsonDict:
    missing_arguments: list[str] = []
    for node in graph.nodes:
        if node.node_type != "tool_argument":
            continue
        if not _has_present_edge(graph, node.node_id, "tool_call"):
            missing_arguments.append(node.node_id)
    return _contract_result(
        ["missing_required_object_acquisition"] if missing_arguments else [],
        {"missing_argument_edges": missing_arguments},
    )


def _check_incompatible_api_use(graph: PlanGraph, contract: StructuralContract) -> JsonDict:
    operations_by_object: dict[str, set[str]] = {}
    for node in graph.nodes:
        if node.node_type != "api_operation":
            continue
        object_id = str(node.attributes.get("object_id") or "")
        operation = str(node.attributes.get("api_operation") or "")
        operations_by_object.setdefault(object_id, set()).add(operation)
    conflicts = [
        {"object_id": object_id, "operation_pair": [left, right]}
        for object_id, operations in sorted(operations_by_object.items())
        for left, right in contract.incompatible_operations
        if left in operations and right in operations
    ]
    return _contract_result(
        ["incompatible_api_operations"] if conflicts else [],
        {"operation_conflicts": conflicts},
    )


def _contract_result(codes: list[str], evidence: JsonDict) -> JsonDict:
    return {
        "detected_violation": bool(codes),
        "violation_code": codes[0] if codes else None,
        "violation_codes": codes,
        "contract_evidence": evidence,
    }


def _classifier_outcome(expected_violation: bool, detected_violation: bool) -> str:
    if expected_violation and detected_violation:
        return "true_positive"
    if expected_violation and not detected_violation:
        return "false_accept"
    if not expected_violation and detected_violation:
        return "false_reject"
    return "true_negative"


def _random_baseline_family(
    case_id: str,
    graph_index: int,
    contracts: tuple[StructuralContract, ...],
) -> str:
    index = (sum(ord(char) for char in case_id) + graph_index) % len(contracts)
    return contracts[index].contract_family


def _length_baseline_family(graph: PlanGraph, contracts: tuple[StructuralContract, ...]) -> str:
    total_length = sum(int(node.attributes.get("text_length", 0)) for node in graph.nodes)
    total_length += sum(int(edge.attributes.get("text_length", 0)) for edge in graph.edges)
    return contracts[total_length % len(contracts)].contract_family


def _replace_node(graph: PlanGraph, node_id: str, **attribute_updates: Any) -> PlanGraph:
    return PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(
            _clone_node(node, **attribute_updates) if node.node_id == node_id else _clone_node(node)
            for node in graph.nodes
        ),
        edges=tuple(_clone_edge(edge) for edge in graph.edges),
        expected_outputs=_copy_jsonish(graph.expected_outputs),
    )


def _replace_edge(graph: PlanGraph, edge_id: str, **attribute_updates: Any) -> PlanGraph:
    return PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(_clone_node(node) for node in graph.nodes),
        edges=tuple(
            _clone_edge(edge, **attribute_updates) if edge.edge_id == edge_id else _clone_edge(edge)
            for edge in graph.edges
        ),
        expected_outputs=_copy_jsonish(graph.expected_outputs),
    )


def _append_api_conflict(graph: PlanGraph) -> PlanGraph:
    conflict_nodes = (
        PlanNode(
            node_id="api:delete:tool_result",
            node_type="api_operation",
            label="delete tool_result",
            value_type="operation",
            attributes={
                "step_index": 5,
                "text_length": 18,
                "api_operation": "delete",
                "object_id": "tool_result",
            },
        ),
        PlanNode(
            node_id="api:read_after_delete:tool_result",
            node_type="api_operation",
            label="read_after_delete tool_result",
            value_type="operation",
            attributes={
                "step_index": 6,
                "text_length": 29,
                "api_operation": "read_after_delete",
                "object_id": "tool_result",
            },
        ),
    )
    return PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(_clone_node(node) for node in graph.nodes) + conflict_nodes,
        edges=tuple(_clone_edge(edge) for edge in graph.edges),
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


def _has_present_edge(graph: PlanGraph, source: str, target: str) -> bool:
    return any(
        edge.source == source
        and edge.target == target
        and edge.attributes.get("present", True) is True
        for edge in graph.edges
    )


def _edge_step(edge: PlanEdge, attr_name: str, node: PlanNode | None) -> int:
    if attr_name in edge.attributes:
        return int(edge.attributes[attr_name])
    if node is None:
        return 0
    return int(node.attributes.get("step_index", 0))


def _runtime_event_counts(runtime_events: list[JsonDict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in runtime_events:
        case_id = str(event.get("case_id") or "")
        if case_id:
            counts[case_id] = counts.get(case_id, 0) + 1
    return counts


def _rate(numerator_rows: list[JsonDict], denominator_rows: list[JsonDict]) -> float | None:
    if not denominator_rows:
        return None
    return round(len(numerator_rows) / len(denominator_rows), 6)


def _copy_jsonish(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _load_json_if_exists(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _display_path(path: Path | str) -> str:
    return Path(path).as_posix()


if __name__ == "__main__":  # pragma: no cover - exercised through focused helpers.
    raise SystemExit(main())
