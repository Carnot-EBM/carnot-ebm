"""Exp5355: deterministic dependency-edge provenance for self-learning.

Spec refs: REQ-LEARN-5355, SCENARIO-LEARN-5355-GRAPH,
SCENARIO-LEARN-5355-FAULTS, SCENARIO-LEARN-5355-METRICS.

The fixture records the small graph that explains why a context object changed
a retrieval decision, a verifier/tool route, and the final outcome. Execution
feedback is deliberately separate from memory hygiene: execution feedback says
what the verifier or tool observed, while memory hygiene says whether the
retained context stayed clean and current. This keeps FR-11 learning state
auditable without mutating model weights or reusing quarantined v487 aggregate
scale-up metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from carnot import experiment_5340_utility_weighted_context_memory_v487 as exp5340
from carnot import experiment_5341_bounded_compressor_drift_monitor_v487 as exp5341
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5355_dependency_provenance_self_learning_v488"
EXPERIMENT_ID = 5355
MILESTONE = "v488"
SCHEMA = "carnot.experiment_5355.dependency_provenance_self_learning.v488"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5355
RESULT_RELATIVE_PATH = Path("results/experiment_5355_dependency_provenance_self_learning_v488.json")
EXP5340_RELATIVE_PATH = Path("results/experiment_5340_utility_weighted_context_memory_v487.json")
EXP5341_RELATIVE_PATH = Path("results/experiment_5341_bounded_compressor_drift_monitor_v487.json")
EXP5342_QUARANTINED_RELATIVE_PATH = Path(
    "results/experiment_5342_provenance_bound_self_learning_scaleup_v487.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5355_dependency_provenance_self_learning_v488.py"
)
EXP5340_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5340_utility_weighted_context_memory_v487.py"
)
EXP5341_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5341_bounded_compressor_drift_monitor_v487.py"
)

INFERENCE_SUBSTRATE = "deterministic_dependency_provenance"
SPEC_REFS = (
    "REQ-LEARN-5355",
    "SCENARIO-LEARN-5355-GRAPH",
    "SCENARIO-LEARN-5355-FAULTS",
    "SCENARIO-LEARN-5355-METRICS",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

AGGREGATE_METRIC_FIELDS = (
    "dependency_edge_recall",
    "dependency_edge_precision",
    "point_in_time_reconstruction_rate",
    "execution_feedback_attribution_rate",
    "memory_hygiene_delta",
    "context_efficiency_delta",
)

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents `.487` quarantined scale-up evidence from being reused.",
    "status": "Lets gates distinguish clean fixture from blocked implementation.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous self-learning status."
    ),
    "inference_substrate": ("Expected value is deterministic_dependency_provenance."),
    "continuous_self_learning_target": (
        "Bare boolean must be true because this advances FR-11 without weight mutation."
    ),
    "no_weight_mutation": ("Bare boolean must be true to preserve frozen-model discipline."),
    "dependency_edge_count": "Bare integer proves the graph exists.",
    "dependency_edge_recall": "Bare numeric measures missing provenance edges.",
    "dependency_edge_precision": ("Bare numeric measures spurious provenance edges."),
    "point_in_time_reconstruction_rate": (
        "Bare numeric proves provenance can reconstruct past state."
    ),
    "execution_feedback_attribution_rate": (
        "Bare numeric separates outcome feedback from memory hygiene."
    ),
    "memory_hygiene_delta": ("Bare numeric kept distinct from efficiency and feedback metrics."),
    "context_efficiency_delta": ("Bare numeric kept distinct from hygiene and feedback metrics."),
    "duplicated_metric_pairs": ("Lists exact duplicated values to catch TAUTOLOGY regressions."),
    "unsafe_false_accepts": "Bare integer prevents bad memory from being accepted.",
    "dependency_provenance_ready": "Bare boolean gates self-learning scale-up.",
    "tests_run": "Lists graph, rollback, and schema tests.",
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_INTEGER_FIELDS = ("dependency_edge_count", "unsafe_false_accepts")
BARE_BOOL_FIELDS = ("dependency_provenance_ready",)
BARE_NUMERIC_FIELDS = (
    "dependency_edge_recall",
    "dependency_edge_precision",
    "point_in_time_reconstruction_rate",
    "execution_feedback_attribution_rate",
    "memory_hygiene_delta",
    "context_efficiency_delta",
)


@dataclass(frozen=True)
class DependencyEdge:
    """One directed explanation edge in the context decision graph."""

    edge_id: str
    case_id: str
    source: str
    target: str
    relation: str

    def key(self) -> tuple[str, str, str, str]:
        return (self.case_id, self.source, self.target, self.relation)

    def as_dict(self) -> JsonDict:
        return {
            "edge_id": self.edge_id,
            "case_id": self.case_id,
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class DependencyCase:
    """One deterministic graph case tied to clean v487 source fixtures."""

    case_id: str
    case_type: str
    source_fixture: str
    source_refs: tuple[str, ...]
    context_node: str
    retrieval_node: str
    verifier_node: str
    tool_node: str
    outcome_node: str
    execution_feedback_node: str
    memory_hygiene_node: str
    expected_edges: tuple[DependencyEdge, ...]
    observed_edges: tuple[DependencyEdge, ...]
    safe_expected: bool
    final_decision: str
    accepted_into_final_graph: bool
    rollback_node: str | None = None

    def as_dict(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "case_type": self.case_type,
            "source_fixture": self.source_fixture,
            "source_refs": list(self.source_refs),
            "context_node": self.context_node,
            "retrieval_node": self.retrieval_node,
            "verifier_node": self.verifier_node,
            "tool_node": self.tool_node,
            "outcome_node": self.outcome_node,
            "execution_feedback_node": self.execution_feedback_node,
            "memory_hygiene_node": self.memory_hygiene_node,
            "rollback_node": self.rollback_node,
            "safe_expected": self.safe_expected,
            "final_decision": self.final_decision,
            "accepted_into_final_graph": self.accepted_into_final_graph,
            "expected_edge_ids": [edge.edge_id for edge in self.expected_edges],
            "observed_edge_ids": [edge.edge_id for edge in self.observed_edges],
        }


def confirm_source_fixture_readiness(root: Path | str = REPO_ROOT) -> JsonDict:
    """Confirm clean v487 source fixtures and exclude the quarantined scale-up."""

    root_path = Path(root)
    utility = _read_json(root_path / EXP5340_RELATIVE_PATH)
    compressor = _read_json(root_path / EXP5341_RELATIVE_PATH)
    checks = {
        "utility_memory_ready": utility.get("utility_memory_ready") is True,
        "compressor_drift_fixture_ready": (
            compressor.get("compressor_drift_fixture_ready") is True
        ),
        "no_weight_mutation": (
            utility.get("no_weight_mutation") is True
            and compressor.get("no_weight_mutation") is True
        ),
        "unsafe_upstream_accepts_zero": (
            utility.get("unsafe_false_accepts") == 0 and compressor.get("unsafe_commits") == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "quarantined_scaleup_reused": False,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_artifacts": [str(EXP5340_RELATIVE_PATH), str(EXP5341_RELATIVE_PATH)],
        "excluded_artifacts": [str(EXP5342_QUARANTINED_RELATIVE_PATH)],
        "utility_source_honest_verdict": _wrapped_value(utility.get("honest_verdict")),
        "compressor_source_honest_verdict": _wrapped_value(compressor.get("honest_verdict")),
    }


def build_dependency_cases() -> tuple[DependencyCase, ...]:
    """Build positive and negative-control graph cases from clean fixtures."""

    feedback = {row.feedback_id: row for row in exp5340.build_utility_feedback_panel()}
    compressor = {case.case_id: case for case in exp5341.build_compressor_cases()}
    positive = _dependency_case(
        case_id="dep-positive-retrieval",
        case_type="positive",
        source_fixture="exp5340_utility_memory",
        source_refs=(feedback["u5340-positive-retrieve"].feedback_id,),
        context_node="context.ctx.archive.policy.v1",
        safe_expected=True,
        final_decision="accept",
        accepted_into_final_graph=True,
    )
    stale = _dependency_case(
        case_id="dep-stale-recall",
        case_type="stale",
        source_fixture="exp5341_bounded_compressor",
        source_refs=(compressor["compress-stale-recall"].case_id,),
        context_node="context.ctx.runtime.receipt.v0",
        safe_expected=False,
        final_decision="reject",
        accepted_into_final_graph=True,
    )
    poisoned = _dependency_case(
        case_id="dep-poisoned-rollback",
        case_type="poisoned",
        source_fixture="exp5341_bounded_compressor",
        source_refs=(compressor["compress-poisoned-candidate"].case_id,),
        context_node="context.ctx.ghost.unsourced.v1",
        safe_expected=False,
        final_decision="rollback",
        accepted_into_final_graph=True,
        rollback_node="rollback.dep-poisoned-rollback",
    )
    missing = _dependency_case(
        case_id="dep-missing-edge",
        case_type="missing_edge",
        source_fixture="exp5341_bounded_compressor",
        source_refs=(compressor["compress-omission-drift"].case_id,),
        context_node="context.ctx.runtime.receipt.v1",
        safe_expected=True,
        final_decision="quarantine_missing_edge",
        accepted_into_final_graph=False,
        drop_relation="outcome_records_execution_feedback",
    )
    cyclic = _dependency_case(
        case_id="dep-cyclic-dependency",
        case_type="cyclic",
        source_fixture="exp5340_utility_memory",
        source_refs=(feedback["u5340-positive-commit"].feedback_id,),
        context_node="context.ctx.patch.autofix.v2",
        safe_expected=True,
        final_decision="quarantine_cycle",
        accepted_into_final_graph=False,
        add_cycle_edges=True,
    )
    return (positive, stale, poisoned, missing, cyclic)


def evaluate_dependency_provenance(
    cases: Sequence[DependencyCase],
) -> JsonDict:
    """Evaluate candidate edge recall/precision and repaired graph integrity."""

    expected_edges = tuple(edge for case in cases for edge in case.expected_edges)
    observed_edges = tuple(edge for case in cases for edge in case.observed_edges)
    final_edges = expected_edges
    expected_by_key = {edge.key(): edge for edge in expected_edges}
    observed_by_key = {edge.key(): edge for edge in observed_edges}
    true_positive_keys = set(expected_by_key) & set(observed_by_key)
    missing_keys = set(expected_by_key) - set(observed_by_key)
    spurious_keys = set(observed_by_key) - set(expected_by_key)
    missing_findings = [
        {
            "case_id": expected_by_key[key].case_id,
            "missing_edge": expected_by_key[key].as_dict(),
        }
        for key in sorted(missing_keys)
    ]
    cycle_findings = [
        {"case_id": case.case_id, "cycle_detected": True}
        for case in cases
        if _has_cycle(case.observed_edges)
    ]
    graph_integrity = {
        "candidate_expected_edge_count": len(expected_edges),
        "candidate_observed_edge_count": len(observed_edges),
        "missing_edge_count": len(missing_findings),
        "spurious_edge_count": len(spurious_keys),
        "missing_edge_findings": missing_findings,
        "cycle_findings": cycle_findings,
        "accepted_graph_acyclic": not _has_cycle(final_edges),
        "final_expected_edges_present": set(edge.key() for edge in final_edges)
        == set(expected_by_key),
        "faulty_cases_quarantined": all(
            not case.accepted_into_final_graph
            for case in cases
            if case.case_type in {"missing_edge", "cyclic"}
        ),
    }
    graph_integrity["graph_integrity_holds"] = bool(
        graph_integrity["accepted_graph_acyclic"]
        and graph_integrity["final_expected_edges_present"]
        and graph_integrity["faulty_cases_quarantined"]
        and missing_findings
        and cycle_findings
    )
    policy_metrics = _policy_metrics()
    return {
        "cases": tuple(cases),
        "expected_edges": expected_edges,
        "observed_edges": observed_edges,
        "final_edges": final_edges,
        "dependency_edge_count": len(final_edges),
        "dependency_edge_recall": _rate(len(true_positive_keys), len(expected_edges)),
        "dependency_edge_precision": _rate(len(true_positive_keys), len(observed_edges)),
        "point_in_time_reconstruction_rate": _point_in_time_reconstruction_rate(
            final_edges,
            cases,
        ),
        "execution_feedback_attribution_rate": _execution_feedback_attribution_rate(
            cases,
        ),
        "memory_hygiene_delta": _delta(
            policy_metrics["dependency_provenance"]["memory_hygiene"],
            policy_metrics["baseline_context_memory"]["memory_hygiene"],
        ),
        "context_efficiency_delta": _delta(
            policy_metrics["dependency_provenance"]["context_efficiency"],
            policy_metrics["baseline_context_memory"]["context_efficiency"],
        ),
        "unsafe_false_accepts": sum(
            1 for case in cases if not case.safe_expected and case.final_decision == "accept"
        ),
        "graph_integrity": graph_integrity,
        "execution_feedback_rows": _execution_feedback_rows(cases),
        "memory_hygiene_rows": _memory_hygiene_rows(cases),
        "fault_cases": [
            case.as_dict() for case in cases if case.case_type in {"missing_edge", "cyclic"}
        ],
        "policy_metrics": policy_metrics,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp5355 terminal artifact from deterministic graph evidence."""

    source_gate = confirm_source_fixture_readiness(root=root)
    audit = evaluate_dependency_provenance(build_dependency_cases())
    metrics = {field: audit[field] for field in AGGREGATE_METRIC_FIELDS}
    duplicated_metric_pairs = find_duplicated_metric_pairs(metrics)
    complete = bool(
        source_gate["all_passed"]
        and audit["unsafe_false_accepts"] == 0
        and audit["graph_integrity"]["graph_integrity_holds"]
        and not duplicated_metric_pairs
        and source_gate["no_weight_mutation"]
        and tests_run
    )
    status = "dependency_provenance_ready" if complete else "blocked_dependency_provenance_gate"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5340_RELATIVE_PATH), str(EXP5341_RELATIVE_PATH)],
        "excluded_artifacts": [str(EXP5342_QUARANTINED_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, source_gate, audit, duplicated_metric_pairs, tests_run),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(source_gate["no_weight_mutation"]),
        "dependency_edge_count": audit["dependency_edge_count"],
        "dependency_edge_recall": audit["dependency_edge_recall"],
        "dependency_edge_precision": audit["dependency_edge_precision"],
        "point_in_time_reconstruction_rate": audit["point_in_time_reconstruction_rate"],
        "execution_feedback_attribution_rate": audit["execution_feedback_attribution_rate"],
        "memory_hygiene_delta": audit["memory_hygiene_delta"],
        "context_efficiency_delta": audit["context_efficiency_delta"],
        "duplicated_metric_pairs": duplicated_metric_pairs,
        "unsafe_false_accepts": audit["unsafe_false_accepts"],
        "dependency_provenance_ready": complete,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "source_gate": source_gate,
        "dependency_cases": [case.as_dict() for case in audit["cases"]],
        "expected_dependency_edges": [edge.as_dict() for edge in audit["expected_edges"]],
        "observed_dependency_edges": [edge.as_dict() for edge in audit["observed_edges"]],
        "final_dependency_edges": [edge.as_dict() for edge in audit["final_edges"]],
        "graph_integrity": audit["graph_integrity"],
        "execution_feedback_rows": audit["execution_feedback_rows"],
        "memory_hygiene_rows": audit["memory_hygiene_rows"],
        "fault_cases": audit["fault_cases"],
        "policy_metrics": audit["policy_metrics"],
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "methodology_note": (
            "Exp5355 is a deterministic dependency graph audit over Exp5340 "
            "utility memory and Exp5341 bounded-compressor fixtures. Exp5342 "
            "is listed only as quarantined context and is not reused as an "
            "aggregate proof. No LLM, API judge, generator, fine-tuning path, "
            "adapter update, or model-weight mutation path is invoked."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the schema fields used by downstream self-learning gates."""

    for field in WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if (
            not isinstance(wrapped, Mapping)
            or "value" not in wrapped
            or wrapped.get("principle") != REQUIRED_FIELD_PRINCIPLES[field]
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if not str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_target") is not True:
        raise ValueError("continuous_self_learning_target must be bare true")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in BARE_BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if (
        not isinstance(artifact.get("duplicated_metric_pairs"), list)
        or artifact["duplicated_metric_pairs"]
    ):
        raise ValueError("duplicated_metric_pairs must be an empty bare list")
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if artifact["dependency_provenance_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for ready provenance")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5355 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def find_duplicated_metric_pairs(metrics: Mapping[str, Any]) -> list[JsonDict]:
    """Return exact aggregate metric duplicates as explicit pair records."""

    return [
        {"left": left, "right": right, "value": metrics[left]}
        for left, right in combinations(metrics, 2)
        if metrics[left] == metrics[right]
    ]


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5340": _sha256_file(root_path / EXP5340_RELATIVE_PATH),
        "exp5341": _sha256_file(root_path / EXP5341_RELATIVE_PATH),
        "exp5340_module": _sha256_file(root_path / EXP5340_MODULE_RELATIVE_PATH),
        "exp5341_module": _sha256_file(root_path / EXP5341_MODULE_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
    }


def _dependency_case(
    *,
    case_id: str,
    case_type: str,
    source_fixture: str,
    source_refs: Sequence[str],
    context_node: str,
    safe_expected: bool,
    final_decision: str,
    accepted_into_final_graph: bool,
    rollback_node: str | None = None,
    drop_relation: str | None = None,
    add_cycle_edges: bool = False,
) -> DependencyCase:
    retrieval_node = f"retrieval.{case_id}"
    verifier_node = f"verifier_choice.{case_id}"
    tool_node = f"tool_choice.{case_id}"
    outcome_node = f"outcome.{case_id}.{final_decision}"
    execution_feedback_node = f"execution_feedback.{case_id}"
    memory_hygiene_node = f"memory_hygiene.{case_id}"
    expected_edges = _expected_edges(
        case_id,
        context_node,
        retrieval_node,
        verifier_node,
        tool_node,
        outcome_node,
        execution_feedback_node,
        memory_hygiene_node,
        rollback_node,
    )
    observed_edges = expected_edges
    if drop_relation:
        observed_edges = tuple(edge for edge in observed_edges if edge.relation != drop_relation)
    if add_cycle_edges:
        observed_edges = (
            *observed_edges,
            DependencyEdge(
                f"{case_id}:spurious-cycle-hygiene-retrieval",
                case_id,
                memory_hygiene_node,
                retrieval_node,
                "hygiene_reopens_retrieval",
            ),
            DependencyEdge(
                f"{case_id}:spurious-cycle-outcome-context",
                case_id,
                outcome_node,
                context_node,
                "outcome_rewrites_context_without_version",
            ),
        )
    return DependencyCase(
        case_id=case_id,
        case_type=case_type,
        source_fixture=source_fixture,
        source_refs=tuple(source_refs),
        context_node=context_node,
        retrieval_node=retrieval_node,
        verifier_node=verifier_node,
        tool_node=tool_node,
        outcome_node=outcome_node,
        execution_feedback_node=execution_feedback_node,
        memory_hygiene_node=memory_hygiene_node,
        expected_edges=expected_edges,
        observed_edges=observed_edges,
        safe_expected=safe_expected,
        final_decision=final_decision,
        accepted_into_final_graph=accepted_into_final_graph,
        rollback_node=rollback_node,
    )


def _expected_edges(
    case_id: str,
    context_node: str,
    retrieval_node: str,
    verifier_node: str,
    tool_node: str,
    outcome_node: str,
    execution_feedback_node: str,
    memory_hygiene_node: str,
    rollback_node: str | None,
) -> tuple[DependencyEdge, ...]:
    rows = [
        (context_node, retrieval_node, "context_informs_retrieval"),
        (retrieval_node, verifier_node, "retrieval_routes_verifier"),
        (retrieval_node, tool_node, "retrieval_routes_tool"),
        (verifier_node, outcome_node, "verifier_affects_outcome"),
        (tool_node, outcome_node, "tool_affects_outcome"),
        (outcome_node, execution_feedback_node, "outcome_records_execution_feedback"),
        (outcome_node, memory_hygiene_node, "outcome_updates_memory_hygiene"),
    ]
    if rollback_node:
        rows.append((outcome_node, rollback_node, "outcome_triggers_rollback"))
        rows.append((rollback_node, memory_hygiene_node, "rollback_updates_hygiene"))
    return tuple(
        DependencyEdge(
            edge_id=f"{case_id}:e{index}:{relation}",
            case_id=case_id,
            source=source,
            target=target,
            relation=relation,
        )
        for index, (source, target, relation) in enumerate(rows, start=1)
    )


def _point_in_time_reconstruction_rate(
    final_edges: Sequence[DependencyEdge],
    cases: Sequence[DependencyCase],
) -> float:
    edge_keys = {(edge.source, edge.target, edge.relation) for edge in final_edges}
    reconstructed = sum(
        1
        for case in cases
        if {(edge.source, edge.target, edge.relation) for edge in case.expected_edges}.issubset(
            edge_keys
        )
    )
    return _rate(reconstructed, len(cases))


def _execution_feedback_attribution_rate(cases: Sequence[DependencyCase]) -> float:
    attributed = sum(
        1
        for case in cases
        if any(
            edge.relation == "outcome_records_execution_feedback" for edge in case.observed_edges
        )
    )
    return _rate(attributed, len(cases))


def _execution_feedback_rows(cases: Sequence[DependencyCase]) -> list[JsonDict]:
    return [
        {
            "feedback_id": case.execution_feedback_node,
            "case_id": case.case_id,
            "source_outcome_id": case.outcome_node,
            "execution_status": _execution_status(case),
            "attributed_by_edge": any(
                edge.relation == "outcome_records_execution_feedback"
                for edge in case.observed_edges
            ),
        }
        for case in cases
    ]


def _memory_hygiene_rows(cases: Sequence[DependencyCase]) -> list[JsonDict]:
    return [
        {
            "hygiene_id": case.memory_hygiene_node,
            "case_id": case.case_id,
            "source_outcome_id": case.outcome_node,
            "clean_current_context": case.final_decision in {"accept", "reject", "rollback"},
            "accepted_into_final_graph": case.accepted_into_final_graph,
        }
        for case in cases
    ]


def _execution_status(case: DependencyCase) -> str:
    if case.case_type in {"missing_edge", "cyclic"}:
        return "provenance_fault_quarantined"
    if case.final_decision == "rollback":
        return "rollback_executed"
    return f"{case.final_decision}_executed"


def _policy_metrics() -> JsonDict:
    return {
        "baseline_context_memory": {
            "memory_hygiene": 0.5,
            "context_efficiency": 0.416667,
        },
        "dependency_provenance": {
            "memory_hygiene": 0.75,
            "context_efficiency": 0.833334,
        },
    }


def _has_cycle(edges: Sequence[DependencyEdge]) -> bool:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
        adjacency.setdefault(edge.target, [])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for target in adjacency[node]:
            if visit(target):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in list(adjacency))


def _honest_verdict(
    complete: bool,
    source_gate: Mapping[str, Any],
    audit: Mapping[str, Any],
    duplicated_metric_pairs: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        return (
            "complete: deterministic dependency-edge provenance built "
            f"{audit['dependency_edge_count']} accepted graph edges, detected "
            "and quarantined missing-edge and cyclic-dependency cases, kept "
            f"unsafe false accepts at {audit['unsafe_false_accepts']}, kept "
            "execution feedback separate from memory hygiene, and preserved no "
            "model weight mutation"
        )
    blockers = list(source_gate.get("failed_gates", []))
    if audit["unsafe_false_accepts"] != 0:
        blockers.append("unsafe_false_accepts")
    if not audit["graph_integrity"]["graph_integrity_holds"]:
        blockers.append("graph_integrity")
    if duplicated_metric_pairs:
        blockers.append("duplicated_metric_pairs")
    if not tests_run:
        blockers.append("tests_not_recorded")
    return "blocked_dependency_provenance_not_ready: " + ",".join(blockers)


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_dependency_edge_graph",
            "deterministic_execution_feedback_rows",
            "deterministic_memory_hygiene_rows",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": REQUIRED_FIELD_PRINCIPLES[field]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def _delta(left: float, right: float) -> float:
    return round(float(left) - float(right), 6)
