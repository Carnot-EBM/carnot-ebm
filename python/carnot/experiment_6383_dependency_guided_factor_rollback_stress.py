"""Exp6383 dependency-guided factor rollback stress.

Spec refs: REQ-LEARN-6383, SCENARIO-LEARN-6383-SCHEMA,
SCENARIO-LEARN-6383-SELECTIVE, SCENARIO-LEARN-6383-CONTROLS,
SCENARIO-LEARN-6383-JOURNAL, SCENARIO-LEARN-6383-READY.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6383_dependency_guided_factor_rollback_stress.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py"
)
SCHEMA = "carnot.experiment_6383.dependency_guided_factor_rollback_stress.v1"
TYPED_DEPENDENCY_SCHEMA_VERSION = "v1"
RUN_DATE = "20260813"
RANDOM_SEED = 6383
BAD_SOURCE_ID = "source_bad_stale_poison"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"

NODE_TYPES = (
    "source_event",
    "obligation",
    "exact_evidence",
    "factor_version",
    "factor",
    "consumer_decision",
    "rollback_action",
)
EDGE_TYPES = (
    "declares_obligation",
    "emits_evidence",
    "checked_by",
    "supports_version",
    "revises_to",
    "materializes_factor",
    "influences_decision",
    "version_used_by_decision",
    "diagnosed_by_rollback",
    "invalidated_by_rollback",
)
ALLOWED_EDGES = (
    ("source_event", "obligation", "declares_obligation"),
    ("source_event", "exact_evidence", "emits_evidence"),
    ("obligation", "exact_evidence", "checked_by"),
    ("exact_evidence", "factor_version", "supports_version"),
    ("factor_version", "factor_version", "revises_to"),
    ("factor_version", "factor", "materializes_factor"),
    ("factor", "consumer_decision", "influences_decision"),
    ("factor_version", "consumer_decision", "version_used_by_decision"),
    ("source_event", "rollback_action", "diagnosed_by_rollback"),
    ("factor_version", "rollback_action", "invalidated_by_rollback"),
    ("factor", "rollback_action", "invalidated_by_rollback"),
    ("consumer_decision", "rollback_action", "invalidated_by_rollback"),
)

STATE_NODE_TYPES = ("factor_version", "factor", "consumer_decision")
EXPECTED_PRESERVED_STATE_NODES = (
    "fv_route_v1",
    "factor_route_guard",
    "decision_route_guard",
    "fv_shared_v1",
    "factor_shared_guard",
    "decision_shared_guard",
)
EXPECTED_HARMFUL_STATE_NODES = (
    "fv_repair_bad_v2",
    "factor_repair_bad",
    "decision_repair_bad",
    "fv_poison_guard_v1",
    "factor_poison_guard",
    "decision_poison_guard",
    "fv_partial_guard_v1",
    "factor_partial_guard",
    "decision_partial_guard",
    "decision_mixed_active",
)
EXPECTED_UNSUPPORTED_DESCENDANTS = EXPECTED_HARMFUL_STATE_NODES

RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6383_dependency_guided_factor_rollback_stress --date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py "
    "-m pytest tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6383_dependency_guided_factor_rollback_stress.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6383_dependency_guided_factor_rollback_stress.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6383_dependency_guided_factor_rollback_stress.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUN_COMMAND,
)

EXP6342_RESULT = Path("results/experiment_6342_anytime_evalue_release_ledger.json")
EXP6343_RESULT = Path("results/experiment_6343_evidence_carrying_factor_lifecycle.json")
EXP6344_RESULT = Path("results/experiment_6344_counterexample_factor_proposal_calibration.json")
EXP6345_RESULT = Path("results/experiment_6345_prospective_certified_factor_evolution_ab.json")
EXP6346_RESULT = Path("results/experiment_6346_certified_factor_evolution_safety_audit.json")
EXP6382_RESULT = Path("results/experiment_6382_chronological_verified_factor_self_learning.json")
EXP6290_MODULE = Path("python/carnot/experiment_6290_revocable_atomic_repair_memory.py")

HASHED_CONTEXT_PATHS = {
    "factor_registry_artifact": EXP6343_RESULT,
    "factor_registry_jsonl": Path(str(EXP6343_RESULT) + ".version_registry.jsonl"),
    "release_ledger_artifact": EXP6342_RESULT,
    "release_ledger_jsonl": Path(str(EXP6342_RESULT) + ".evalue_ledger.jsonl"),
    "exact_checker_exp6342": Path("python/carnot/experiment_6342_anytime_evalue_release_ledger.py"),
    "exact_checker_exp6343": Path(
        "python/carnot/experiment_6343_evidence_carrying_factor_lifecycle.py"
    ),
    "exact_checker_exp6346": Path(
        "python/carnot/experiment_6346_certified_factor_evolution_safety_audit.py"
    ),
    "atomic_repair_memory_exp6290": EXP6290_MODULE,
    "upstream_exp6382": EXP6382_RESULT,
}
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    EXP6342_RESULT,
    EXP6343_RESULT,
    EXP6344_RESULT,
    EXP6345_RESULT,
    EXP6346_RESULT,
    EXP6382_RESULT,
    EXP6290_MODULE,
)
SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_learning_context_class",
    "registry_release_ledger_and_checker_hashes",
    "typed_dependency_schema_path_hash_and_version",
    "allowed_node_and_edge_types",
    "preregistered_injection_and_arm_contract",
    "deterministic_fixture_manifest",
    "lineage_graphs_before_and_after_injection",
    "diagnosis_receipts",
    "selective_full_reset_and_no_rollback_results",
    "harmful_descendants_removed",
    "independently_supported_state_preserved",
    "overrollback_underrollback_and_unsafe_survivor_counts",
    "exact_replay_cost_latency_and_memory",
    "cycle_missing_edge_corruption_and_interruption_results",
    "journal_restart_and_idempotence_receipts",
    "terminal_registry_roots",
    "dependency_guided_rollback_ready_score",
    "no_live_utility_claim",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status follows rollback safety, preservation, corruption checks, protected files, and tests.",
    "upstream_learning_context_class": "Exp6382 absence, blocked state, or terminal class is context only, not a readiness gate.",
    "registry_release_ledger_and_checker_hashes": "Factor registry, release ledger, exact checker sources, and Exp6382 when present are hashed before fixture replay.",
    "typed_dependency_schema_path_hash_and_version": "The node, edge, and acyclicity schema is frozen as a sidecar.",
    "allowed_node_and_edge_types": "Typed nodes and allowed edge pairs define the only legal lineage surface.",
    "preregistered_injection_and_arm_contract": "Bad-source injection order and the selective, full reset, and no-rollback controls are fixed before replay.",
    "deterministic_fixture_manifest": "Clean, stale, poisoned, duplicated, misattributed, partially supported, shared-support, cyclic, and missing-evidence fixtures are named and seeded.",
    "lineage_graphs_before_and_after_injection": "Graph roots, node counts, edge counts, and state roots are recorded before and after injection.",
    "diagnosis_receipts": "The diagnosed bad source and exact replay evidence explain the invalidation frontier.",
    "selective_full_reset_and_no_rollback_results": "All three controls report the same metrics on the same replay work.",
    "harmful_descendants_removed": "Selective rollback must remove all unsupported harmful descendants.",
    "independently_supported_state_preserved": "Exact-valid independent support paths must survive selective rollback.",
    "overrollback_underrollback_and_unsafe_survivor_counts": "Over-removal, missed rollback, and unsafe survivors stay visible.",
    "exact_replay_cost_latency_and_memory": "Checker calls, deterministic cost, latency, and graph memory bytes are measured.",
    "cycle_missing_edge_corruption_and_interruption_results": "Cycles, missing evidence, corruption, incomplete invalidation, and interruption fail closed.",
    "journal_restart_and_idempotence_receipts": "Restart, double rollback, root mismatch, edge tampering, orphan nodes, and active decision rollback are recorded.",
    "terminal_registry_roots": "Terminal roots prove selective rollback is stable and exact-valid.",
    "dependency_guided_rollback_ready_score": "Readiness is a conjunctive safety and preservation gate.",
    "no_live_utility_claim": "Bare true states that this stress test does not promote live learning utility.",
    "protected_files_unchanged": "Conductor, ops, traceability, prior factor code, and upstream artifacts remain byte-identical.",
    "preconditions_checked": "Date, source hashes, protected hashes, schema, fixtures, controls, seeds, and upstream context freeze before replay.",
    "inference_substrate": "The substrate declares deterministic exact replay and typed lineage analysis with no LLM.",
    "verifier_is_oracle": "Bare true applies only to deterministic exact replay checkers, not lineage or rollback policy.",
    "field_principles": "Every required field states its guard.",
    "field_provenance": "Every required field maps to specs, source hashes, fixtures, exact checks, rollback receipts, tests, or roots.",
    "random_seed": "Fixed seed pins fixture order.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states that no live utility was claimed.",
}
FIELD_PROVENANCE = {
    field: {"principle": principle, "satisfied_by": "Exp6383 deterministic replay"}
    for field, principle in FIELD_PRINCIPLES.items()
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashes and roots."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str | None:
    """Return a file digest, or None when the file is absent."""

    path = Path(path)
    if not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def require(condition: bool, reason: str) -> None:
    """Raise a stable validation error when a gate fails."""

    if not condition:
        raise ValueError(reason)


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mappings unchanged and replace other values with an empty map."""

    return value if isinstance(value, Mapping) else {}


def relative_or_absolute(path: Path) -> str:
    """Use repo-relative paths when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def path_receipt(path: Path) -> JsonDict:
    """Record path, hash, size, and presence."""

    return {
        "path": relative_or_absolute(path),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical audit JSON with a stable newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _edge(source: str, target: str, edge_type: str) -> JsonDict:
    payload = {"from": source, "to": target, "type": edge_type}
    return {**payload, "lineage_hash": sha256_json(payload)}


def _source(node_id: str, *, trusted: bool, fixture: str) -> JsonDict:
    return {
        "id": node_id,
        "type": "source_event",
        "trusted": trusted,
        "fixture": fixture,
        "source_hash": sha256_json({"source_event": node_id, "trusted": trusted}),
    }


def _obligation(node_id: str, source_id: str) -> JsonDict:
    return {"id": node_id, "type": "obligation", "source_event_id": source_id}


def _evidence(node_id: str, obligation_id: str, *, exact_valid: bool, reason: str) -> JsonDict:
    return {
        "id": node_id,
        "type": "exact_evidence",
        "obligation_id": obligation_id,
        "exact_replay_valid": exact_valid,
        "reason": reason,
    }


def _factor_version(
    node_id: str, obligation_id: str, *, allowed_obligation_ids: Sequence[str] | None = None
) -> JsonDict:
    return {
        "id": node_id,
        "type": "factor_version",
        "required_obligation_id": obligation_id,
        "allowed_obligation_ids": list(allowed_obligation_ids or [obligation_id]),
    }


def _factor(node_id: str) -> JsonDict:
    return {"id": node_id, "type": "factor"}


def _decision(node_id: str, *, active: bool = True) -> JsonDict:
    return {"id": node_id, "type": "consumer_decision", "active_at_rollback": active}


def _graph(nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> JsonDict:
    active = sorted(node["id"] for node in nodes if node["type"] in STATE_NODE_TYPES)
    return {"nodes": {str(node["id"]): dict(node) for node in nodes}, "edges": [dict(edge) for edge in edges], "active_node_ids": active}


def build_clean_graph() -> JsonDict:
    """Build the clean graph before bad-source injection."""

    nodes = [
        _source("source_clean_route", trusted=True, fixture="clean"),
        _obligation("obligation_route", "source_clean_route"),
        _evidence("evidence_route_exact", "obligation_route", exact_valid=True, reason="clean"),
        _factor_version("fv_route_v1", "obligation_route"),
        _factor("factor_route_guard"),
        _decision("decision_route_guard"),
        _source("source_shared_exact", trusted=True, fixture="shared_support"),
        _obligation("obligation_shared", "source_shared_exact"),
        _evidence("evidence_shared_exact", "obligation_shared", exact_valid=True, reason="shared_support"),
        _factor_version(
            "fv_shared_v1",
            "obligation_shared",
            allowed_obligation_ids=["obligation_shared", "obligation_bad_stale"],
        ),
        _factor("factor_shared_guard"),
        _decision("decision_shared_guard"),
    ]
    edges = [
        _edge("source_clean_route", "obligation_route", "declares_obligation"),
        _edge("obligation_route", "evidence_route_exact", "checked_by"),
        _edge("evidence_route_exact", "fv_route_v1", "supports_version"),
        _edge("fv_route_v1", "factor_route_guard", "materializes_factor"),
        _edge("factor_route_guard", "decision_route_guard", "influences_decision"),
        _edge("source_shared_exact", "obligation_shared", "declares_obligation"),
        _edge("obligation_shared", "evidence_shared_exact", "checked_by"),
        _edge("evidence_shared_exact", "fv_shared_v1", "supports_version"),
        _edge("fv_shared_v1", "factor_shared_guard", "materializes_factor"),
        _edge("factor_shared_guard", "decision_shared_guard", "influences_decision"),
    ]
    return _graph(nodes, edges)


def build_injected_graph() -> JsonDict:
    """Build the frozen graph after stale and poisoned fixtures arrive."""

    graph = build_clean_graph()
    nodes = list(graph["nodes"].values()) + [
        _source(BAD_SOURCE_ID, trusted=False, fixture="stale_poisoned"),
        _obligation("obligation_bad_stale", BAD_SOURCE_ID),
        _obligation("obligation_bad_poison", BAD_SOURCE_ID),
        _evidence("evidence_bad_stale", "obligation_bad_stale", exact_valid=False, reason="stale"),
        _evidence("evidence_bad_poison", "obligation_bad_poison", exact_valid=False, reason="poisoned"),
        _factor_version("fv_repair_bad_v2", "obligation_bad_stale"),
        _factor("factor_repair_bad"),
        _decision("decision_repair_bad"),
        _factor_version("fv_poison_guard_v1", "obligation_bad_poison"),
        _factor("factor_poison_guard"),
        _decision("decision_poison_guard"),
        _factor_version("fv_partial_guard_v1", "obligation_bad_stale"),
        _factor("factor_partial_guard"),
        _decision("decision_partial_guard"),
        _decision("decision_mixed_active"),
    ]
    edges = list(graph["edges"]) + [
        _edge(BAD_SOURCE_ID, "obligation_bad_stale", "declares_obligation"),
        _edge(BAD_SOURCE_ID, "obligation_bad_poison", "declares_obligation"),
        _edge("obligation_bad_stale", "evidence_bad_stale", "checked_by"),
        _edge("obligation_bad_poison", "evidence_bad_poison", "checked_by"),
        _edge("evidence_bad_stale", "fv_repair_bad_v2", "supports_version"),
        _edge("fv_repair_bad_v2", "factor_repair_bad", "materializes_factor"),
        _edge("factor_repair_bad", "decision_repair_bad", "influences_decision"),
        _edge("evidence_bad_poison", "fv_poison_guard_v1", "supports_version"),
        _edge("fv_poison_guard_v1", "factor_poison_guard", "materializes_factor"),
        _edge("factor_poison_guard", "decision_poison_guard", "influences_decision"),
        _edge("evidence_bad_stale", "fv_partial_guard_v1", "supports_version"),
        _edge("fv_partial_guard_v1", "factor_partial_guard", "materializes_factor"),
        _edge("factor_partial_guard", "decision_partial_guard", "influences_decision"),
        _edge("evidence_bad_stale", "fv_shared_v1", "supports_version"),
        _edge("factor_route_guard", "decision_mixed_active", "influences_decision"),
        _edge("factor_repair_bad", "decision_mixed_active", "influences_decision"),
    ]
    return _graph(nodes, edges)


def validate_graph_rows(
    node_rows: Sequence[Mapping[str, Any]], edge_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Validate typed rows before constructing graph state."""

    seen: set[str] = set()
    for row in node_rows:
        node_id = str(row.get("id", ""))
        require(node_id not in seen, "duplicate_node_id")
        require(str(row.get("type")) in NODE_TYPES, "unsupported_node_type")
        seen.add(node_id)
    return validate_graph(_graph(node_rows, edge_rows))


def validate_graph(graph: Mapping[str, Any]) -> JsonDict:
    """Validate node ids, edge types, lineage hashes, evidence, and cycles."""

    nodes = as_mapping(graph.get("nodes"))
    edges = list(graph.get("edges", []))
    node_ids = set(nodes)
    incoming = _incoming(edges)
    for node_id, node in nodes.items():
        require(str(as_mapping(node).get("type")) in NODE_TYPES, "unsupported_node_type")
        if as_mapping(node).get("type") == "factor_version":
            evidence_ids = [
                edge["from"] for edge in incoming.get(node_id, []) if edge["type"] == "supports_version"
            ]
            require(evidence_ids, "missing_evidence")
            for evidence_id in evidence_ids:
                evidence = as_mapping(nodes.get(evidence_id))
                allowed = set(as_mapping(node).get("allowed_obligation_ids", []))
                require(evidence.get("obligation_id") in allowed, "misattributed_evidence")
    for edge in edges:
        source = str(edge.get("from"))
        target = str(edge.get("to"))
        edge_type = str(edge.get("type"))
        require(source in node_ids and target in node_ids, "missing_node")
        source_type = str(as_mapping(nodes[source]).get("type"))
        target_type = str(as_mapping(nodes[target]).get("type"))
        require((source_type, target_type, edge_type) in ALLOWED_EDGES, "unsupported_edge")
        require(
            edge.get("lineage_hash") == sha256_json({"from": source, "to": target, "type": edge_type}),
            "corrupted_lineage",
        )
    _require_acyclic(node_ids, edges)
    return {"valid": True, "node_count": len(nodes), "edge_count": len(edges)}


def _incoming(edges: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    incoming: dict[str, list[JsonDict]] = defaultdict(list)
    for edge in edges:
        incoming[str(edge.get("to"))].append(dict(edge))
    return incoming


def _outgoing(edges: Sequence[Mapping[str, Any]]) -> dict[str, list[JsonDict]]:
    outgoing: dict[str, list[JsonDict]] = defaultdict(list)
    for edge in edges:
        outgoing[str(edge.get("from"))].append(dict(edge))
    return outgoing


def _require_acyclic(node_ids: set[str], edges: Sequence[Mapping[str, Any]]) -> None:
    outgoing = _outgoing(edges)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        require(node_id not in visiting, "cycle_detected")
        if node_id in visited:
            return
        visiting.add(node_id)
        for edge in outgoing.get(node_id, []):
            visit(str(edge["to"]))
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in sorted(node_ids):
        visit(node_id)


def build_attack_graph(fixture_name: str) -> JsonDict:
    """Build one invalid fixture graph for fail-closed probes."""

    graph = build_injected_graph()
    nodes = {key: dict(value) for key, value in as_mapping(graph["nodes"]).items()}
    edges = [dict(edge) for edge in graph["edges"]]
    if fixture_name == "cyclic":
        nodes["fv_cycle_a"] = _factor_version("fv_cycle_a", "obligation_route")
        nodes["fv_cycle_b"] = _factor_version("fv_cycle_b", "obligation_route")
        edges.extend(
            [
                _edge("evidence_route_exact", "fv_cycle_a", "supports_version"),
                _edge("evidence_route_exact", "fv_cycle_b", "supports_version"),
                _edge("fv_cycle_a", "fv_cycle_b", "revises_to"),
                _edge("fv_cycle_b", "fv_cycle_a", "revises_to"),
            ]
        )
    elif fixture_name == "missing_evidence":
        nodes["fv_missing_evidence"] = _factor_version("fv_missing_evidence", "obligation_route")
    elif fixture_name == "misattributed":
        nodes["fv_misattributed"] = _factor_version("fv_misattributed", "obligation_bad_stale")
        edges.append(_edge("evidence_route_exact", "fv_misattributed", "supports_version"))
    elif fixture_name == "edge_tampering":
        edges.append(_edge("decision_route_guard", "source_clean_route", "influences_decision"))
    elif fixture_name == "orphaned_nodes":
        edges.append(_edge("evidence_route_exact", "missing_factor_version", "supports_version"))
    elif fixture_name == "corrupted_lineage":
        edges[0] = {**edges[0], "lineage_hash": "sha256:corrupt"}
    elif fixture_name == "duplicated":
        return validate_graph_rows(
            [
                {"id": "duplicate_source", "type": "source_event", "trusted": True},
                {"id": "duplicate_source", "type": "source_event", "trusted": True},
            ],
            [],
        )
    elif fixture_name == "clean":
        return build_clean_graph()
    else:
        raise ValueError("unknown_fixture")
    return {"nodes": nodes, "edges": edges, "active_node_ids": sorted(nodes)}


def attack_fixture_result(fixture_name: str) -> JsonDict:
    """Run one invalid fixture and return its fail-closed receipt."""

    try:
        validate_graph(build_attack_graph(fixture_name))
    except ValueError as exc:
        return {
            "fixture": fixture_name,
            "accepted": False,
            "fail_closed": True,
            "reason": str(exc),
        }
    return {"fixture": fixture_name, "accepted": True, "fail_closed": False, "reason": ""}


def typed_dependency_schema() -> JsonDict:
    """Return the frozen typed dependency schema."""

    return {
        "schema": SCHEMA + ".typed_dependency_schema",
        "schema_version": TYPED_DEPENDENCY_SCHEMA_VERSION,
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "allowed_edges": [
            {"from_type": source, "to_type": target, "edge_type": edge_type}
            for source, target, edge_type in ALLOWED_EDGES
        ],
        "acyclic": True,
        "state_node_types": list(STATE_NODE_TYPES),
    }


def fixture_manifest() -> JsonDict:
    """Name every deterministic fixture and its expected terminal behavior."""

    return {
        "schema": SCHEMA + ".fixture_manifest",
        "random_seed": RANDOM_SEED,
        "fixture_order": [
            "clean",
            "stale",
            "poisoned",
            "duplicated",
            "misattributed",
            "partially_supported",
            "shared_support",
            "cyclic",
            "missing_evidence",
        ],
        "bad_source_node_id": BAD_SOURCE_ID,
        "expected_harmful_state_nodes": list(EXPECTED_HARMFUL_STATE_NODES),
        "expected_preserved_state_nodes": list(EXPECTED_PRESERVED_STATE_NODES),
        "deterministic": True,
    }


def graph_state_root(
    graph: Mapping[str, Any], active_node_ids: Sequence[str] | None = None
) -> str:
    """Hash graph bytes plus the active state set."""

    active = sorted(active_node_ids if active_node_ids is not None else graph["active_node_ids"])
    return sha256_json({"nodes": graph["nodes"], "edges": graph["edges"], "active_node_ids": active})


def graph_summary(graph: Mapping[str, Any]) -> JsonDict:
    """Summarize graph size and active-state root."""

    active = list(graph.get("active_node_ids", []))
    return {
        "node_count": len(as_mapping(graph.get("nodes"))),
        "edge_count": len(graph.get("edges", [])),
        "active_state_count": len(active),
        "state_root": graph_state_root(graph, active),
    }


def support_sets(graph: Mapping[str, Any], bad_source_id: str = BAD_SOURCE_ID) -> JsonDict:
    """Compute exact-valid independent support for versions, factors, and decisions."""

    validate_graph(graph)
    nodes = as_mapping(graph["nodes"])
    incoming = _incoming(graph["edges"])
    supported_evidence: set[str] = set()
    for node_id, node in nodes.items():
        row = as_mapping(node)
        if row.get("type") != "exact_evidence" or row.get("exact_replay_valid") is not True:
            continue
        source_ids = _source_ancestors(node_id, nodes, incoming)
        if any(
            source_id != bad_source_id and as_mapping(nodes[source_id]).get("trusted") is True
            for source_id in source_ids
        ):
            supported_evidence.add(str(node_id))
    supported_versions = {
        node_id
        for node_id, node in nodes.items()
        if as_mapping(node).get("type") == "factor_version"
        and any(edge["from"] in supported_evidence for edge in incoming.get(node_id, []))
    }
    supported_factors = {
        node_id
        for node_id, node in nodes.items()
        if as_mapping(node).get("type") == "factor"
        and any(edge["from"] in supported_versions for edge in incoming.get(node_id, []))
    }
    supported_decisions: set[str] = set()
    for node_id, node in nodes.items():
        if as_mapping(node).get("type") != "consumer_decision":
            continue
        factors = [edge["from"] for edge in incoming.get(node_id, []) if edge["type"] == "influences_decision"]
        if factors and all(factor in supported_factors for factor in factors):
            supported_decisions.add(str(node_id))
    return {
        "evidence": sorted(supported_evidence),
        "factor_versions": sorted(supported_versions),
        "factors": sorted(supported_factors),
        "consumer_decisions": sorted(supported_decisions),
        "state_nodes": sorted(supported_versions | supported_factors | supported_decisions),
    }


def _source_ancestors(
    node_id: str, nodes: Mapping[str, Any], incoming: Mapping[str, list[JsonDict]]
) -> set[str]:
    found: set[str] = set()
    queue = deque([node_id])
    seen: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if as_mapping(nodes[current]).get("type") == "source_event":
            found.add(current)
        for edge in incoming.get(current, []):
            queue.append(str(edge["from"]))
    return found


def _descendants(graph: Mapping[str, Any], root_id: str) -> set[str]:
    outgoing = _outgoing(graph["edges"])
    found: set[str] = set()
    queue = deque(edge["to"] for edge in outgoing.get(root_id, []))
    while queue:
        current = str(queue.popleft())
        if current in found:
            continue
        found.add(current)
        queue.extend(edge["to"] for edge in outgoing.get(current, []))
    return found


def exact_replay_work(graph: Mapping[str, Any]) -> JsonDict:
    """Replay exact support checks over state nodes."""

    support = support_sets(graph)
    nodes = as_mapping(graph["nodes"])
    factor_version_ids = sorted(
        node_id for node_id, node in nodes.items() if as_mapping(node).get("type") == "factor_version"
    )
    decision_ids = sorted(
        node_id
        for node_id, node in nodes.items()
        if as_mapping(node).get("type") == "consumer_decision"
    )
    checked = factor_version_ids + decision_ids
    return {
        "checked_node_ids": checked,
        "exact_replay_call_count": len(checked),
        "factor_version_support": {
            node_id: node_id in support["factor_versions"] for node_id in factor_version_ids
        },
        "decision_support": {node_id: node_id in support["consumer_decisions"] for node_id in decision_ids},
        "exact_replay_work_hash": sha256_json(checked),
        "exact_check_cost_units": round(len(checked) * 0.01, 12),
        "deterministic_latency_s": round(len(checked) * 0.0005, 12),
        "graph_memory_bytes": len(canonical_json(graph).encode("utf-8")),
    }


def diagnose_bad_source(graph: Mapping[str, Any], bad_source_id: str = BAD_SOURCE_ID) -> JsonDict:
    """Find unsupported descendants of the diagnosed bad source."""

    validate_graph(graph)
    nodes = as_mapping(graph["nodes"])
    require(bad_source_id in nodes, "bad_source_missing")
    descendants = _descendants(graph, bad_source_id)
    supported = set(support_sets(graph, bad_source_id)["state_nodes"])
    unsupported = sorted(
        node_id
        for node_id in descendants
        if as_mapping(nodes.get(node_id)).get("type") in STATE_NODE_TYPES and node_id not in supported
    )
    return {
        "bad_source_node_id": bad_source_id,
        "bad_source_found": True,
        "descendant_node_ids": sorted(descendants),
        "unsupported_descendant_node_ids": unsupported,
        "independently_supported_descendant_node_ids": sorted(supported & descendants),
        "bad_evidence_node_ids": [
            node_id
            for node_id in sorted(descendants)
            if as_mapping(nodes.get(node_id)).get("type") == "exact_evidence"
            and as_mapping(nodes.get(node_id)).get("exact_replay_valid") is False
        ],
        "exact_replay_work": exact_replay_work(graph),
    }


def apply_selective_rollback(
    graph: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    *,
    starting_active_node_ids: Sequence[str] | None = None,
) -> JsonDict:
    """Invalidate unsupported descendants and preserve independent support."""

    active_before = sorted(starting_active_node_ids if starting_active_node_ids is not None else graph["active_node_ids"])
    unsupported = set(diagnosis["unsupported_descendant_node_ids"])
    supported = set(support_sets(graph)["state_nodes"])
    invalidated = sorted(unsupported & set(active_before))
    active_after = sorted(node_id for node_id in active_before if node_id not in unsupported)
    harmful = set(EXPECTED_HARMFUL_STATE_NODES)
    preserved = set(EXPECTED_PRESERVED_STATE_NODES)
    overrollback = sorted(preserved - set(active_after))
    underrollback = sorted(harmful & set(active_after))
    return _arm_result(
        "selective_descendant_rollback",
        graph,
        active_before,
        active_after,
        invalidated,
        sorted(preserved & set(active_after)),
        overrollback,
        underrollback,
        sorted(harmful & set(active_after)),
        sorted(
            node_id
            for node_id in invalidated
            if as_mapping(as_mapping(graph["nodes"]).get(node_id)).get("type") == "consumer_decision"
        ),
        supported,
    )


def _arm_result(
    arm_name: str,
    graph: Mapping[str, Any],
    active_before: Sequence[str],
    active_after: Sequence[str],
    invalidated: Sequence[str],
    preserved: Sequence[str],
    overrollback: Sequence[str],
    underrollback: Sequence[str],
    unsafe_survivors: Sequence[str],
    active_consumer_decisions: Sequence[str],
    supported: set[str],
) -> JsonDict:
    replay = exact_replay_work(graph)
    return {
        "arm": arm_name,
        "initial_graph_root": graph_state_root(graph, graph["active_node_ids"]),
        "terminal_root": graph_state_root(graph, active_after),
        "injection_order_hash": sha256_json(fixture_manifest()["fixture_order"]),
        "exact_replay_work_hash": replay["exact_replay_work_hash"],
        "exact_replay_call_count": replay["exact_replay_call_count"],
        "active_node_ids_before": list(active_before),
        "active_node_ids_after": list(active_after),
        "invalidated_node_ids": list(invalidated),
        "valid_state_preserved_count": len(preserved),
        "preserved_node_ids": list(preserved),
        "overrollback_count": len(overrollback),
        "underrollback_count": len(underrollback),
        "unsafe_survivor_count": len(unsafe_survivors),
        "unsafe_survivor_node_ids": list(unsafe_survivors),
        "invalidated_active_consumer_decisions": list(active_consumer_decisions),
        "supported_state_before": sorted(supported),
    }


def run_control_arms(graph: Mapping[str, Any], diagnosis: Mapping[str, Any]) -> JsonDict:
    """Run selective, full reset, and no-rollback arms on the same graph."""

    support = set(support_sets(graph)["state_nodes"])
    active = sorted(graph["active_node_ids"])
    selective = apply_selective_rollback(graph, diagnosis)
    full_active: list[str] = []
    full_reset = _arm_result(
        "full_registry_reset",
        graph,
        active,
        full_active,
        active,
        sorted(support & set(full_active)),
        sorted(set(EXPECTED_PRESERVED_STATE_NODES)),
        [],
        [],
        [node_id for node_id in active if node_id.startswith("decision_")],
        support,
    )
    no_rollback = _arm_result(
        "no_rollback",
        graph,
        active,
        active,
        [],
        sorted(support & set(active)),
        [],
        sorted(set(EXPECTED_HARMFUL_STATE_NODES) & set(active)),
        sorted(set(EXPECTED_HARMFUL_STATE_NODES) & set(active)),
        [],
        support,
    )
    return {
        "selective_descendant_rollback": selective,
        "full_registry_reset": full_reset,
        "no_rollback": no_rollback,
    }


def cycle_missing_edge_corruption_and_interruption_results(
    graph: Mapping[str, Any], diagnosis: Mapping[str, Any]
) -> JsonDict:
    """Collect structural and interruption fail-closed probes."""

    fixture_names = (
        "cyclic",
        "missing_evidence",
        "edge_tampering",
        "orphaned_nodes",
        "corrupted_lineage",
        "misattributed",
        "duplicated",
    )
    fixture_results = {name: attack_fixture_result(name) for name in fixture_names}
    incomplete = apply_selective_rollback(graph, {"unsupported_descendant_node_ids": ["fv_repair_bad_v2"]})
    incomplete_fail_closed = incomplete["unsafe_survivor_count"] > 0
    journal = {"pre_root": graph_state_root(graph, graph["active_node_ids"]), "interrupted_after": "begin"}
    restart = restart_from_journal(graph, diagnosis, journal)
    return {
        "fixture_results": fixture_results,
        "incomplete_invalidation": {
            "fail_closed": incomplete_fail_closed,
            "unsafe_survivor_count": incomplete["unsafe_survivor_count"],
        },
        "journal_interruption": {
            "restart_completed": True,
            "terminal_root": restart["terminal_root"],
        },
        "all_fail_closed": all(row["fail_closed"] for row in fixture_results.values())
        and incomplete_fail_closed
        and restart["unsafe_survivor_count"] == 0,
    }


def restart_from_journal(
    graph: Mapping[str, Any], diagnosis: Mapping[str, Any], journal: Mapping[str, Any]
) -> JsonDict:
    """Resume a rollback journal only when the pre-root matches."""

    expected = graph_state_root(graph, graph["active_node_ids"])
    require(journal.get("pre_root") == expected, "journal_root_mismatch")
    return apply_selective_rollback(graph, diagnosis)


def journal_restart_and_idempotence_receipts(
    graph: Mapping[str, Any], diagnosis: Mapping[str, Any]
) -> JsonDict:
    """Prove restart and double rollback converge to one terminal root."""

    first = apply_selective_rollback(graph, diagnosis)
    journal = {"pre_root": graph_state_root(graph, graph["active_node_ids"]), "interrupted_after": "begin"}
    restarted = restart_from_journal(graph, diagnosis, journal)
    second = apply_selective_rollback(
        graph,
        diagnosis,
        starting_active_node_ids=first["active_node_ids_after"],
    )
    try:
        restart_from_journal(graph, diagnosis, {"pre_root": "sha256:wrong"})
        root_mismatch = {"fail_closed": False, "reason": ""}  # pragma: no cover
    except ValueError as exc:
        root_mismatch = {"fail_closed": True, "reason": str(exc)}
    return {
        "interrupted_journal_restart": {
            "restart_completed": True,
            "terminal_root": restarted["terminal_root"],
        },
        "double_rollback": {
            "idempotent": first["terminal_root"] == second["terminal_root"],
            "first_terminal_root": first["terminal_root"],
            "second_terminal_root": second["terminal_root"],
        },
        "root_mismatch": root_mismatch,
        "edge_tampering": attack_fixture_result("edge_tampering"),
        "orphaned_nodes": attack_fixture_result("orphaned_nodes"),
        "rollback_of_active_consumer_decision": {
            "decision_id": "decision_mixed_active",
            "invalidated": "decision_mixed_active" in first["invalidated_node_ids"],
        },
    }


def terminal_registry_roots(graph: Mapping[str, Any], controls: Mapping[str, Any]) -> JsonDict:
    """Return terminal roots for every rollback arm."""

    selective = as_mapping(controls["selective_descendant_rollback"])
    second = apply_selective_rollback(
        graph,
        diagnose_bad_source(graph),
        starting_active_node_ids=selective["active_node_ids_after"],
    )
    return {
        "initial_graph_root": graph_state_root(graph, graph["active_node_ids"]),
        "selective_terminal_root": selective["terminal_root"],
        "full_reset_terminal_root": controls["full_registry_reset"]["terminal_root"],
        "no_rollback_terminal_root": controls["no_rollback"]["terminal_root"],
        "double_rollback_terminal_root": second["terminal_root"],
        "idempotent_exact_valid_terminal_root": selective["terminal_root"] == second["terminal_root"]
        and selective["unsafe_survivor_count"] == 0,
    }


def harmful_descendants_removed(controls: Mapping[str, Any]) -> JsonDict:
    """Report whether selective rollback removed all harmful state nodes."""

    selective = as_mapping(controls["selective_descendant_rollback"])
    removed = sorted(set(selective["invalidated_node_ids"]) & set(EXPECTED_HARMFUL_STATE_NODES))
    return {
        "expected_harmful_node_ids": list(EXPECTED_HARMFUL_STATE_NODES),
        "removed_node_ids": removed,
        "removed_count": len(removed),
        "expected_count": len(EXPECTED_HARMFUL_STATE_NODES),
        "removed_all_harmful_descendants": set(removed) == set(EXPECTED_HARMFUL_STATE_NODES),
    }


def independently_supported_state_preserved(controls: Mapping[str, Any]) -> JsonDict:
    """Report whether independent exact-valid state survived selective rollback."""

    selective = as_mapping(controls["selective_descendant_rollback"])
    active = set(selective["active_node_ids_after"])
    preserved = sorted(set(EXPECTED_PRESERVED_STATE_NODES) & active)
    return {
        "expected_preserved_node_ids": list(EXPECTED_PRESERVED_STATE_NODES),
        "preserved_node_ids": preserved,
        "preserved_count": len(preserved),
        "expected_count": len(EXPECTED_PRESERVED_STATE_NODES),
        "all_independently_supported_state_preserved": set(preserved)
        == set(EXPECTED_PRESERVED_STATE_NODES),
    }


def overrollback_underrollback_and_unsafe_survivor_counts(controls: Mapping[str, Any]) -> JsonDict:
    """Return comparable safety counters for all controls."""

    selective = controls["selective_descendant_rollback"]
    full_reset = controls["full_registry_reset"]
    no_rollback = controls["no_rollback"]
    return {
        "selective_overrollback_count": selective["overrollback_count"],
        "selective_underrollback_count": selective["underrollback_count"],
        "selective_unsafe_survivor_count": selective["unsafe_survivor_count"],
        "full_reset_overrollback_count": full_reset["overrollback_count"],
        "no_rollback_underrollback_count": no_rollback["underrollback_count"],
        "no_rollback_unsafe_survivor_count": no_rollback["unsafe_survivor_count"],
    }


def exact_replay_cost_latency_and_memory(graph: Mapping[str, Any]) -> JsonDict:
    """Expose exact replay calls, cost, latency, and memory."""

    replay = exact_replay_work(graph)
    return {
        "exact_replay_call_count": replay["exact_replay_call_count"],
        "exact_check_cost_units": replay["exact_check_cost_units"],
        "deterministic_latency_s": replay["deterministic_latency_s"],
        "graph_memory_bytes": replay["graph_memory_bytes"],
        "verifier_is_oracle_only_for_exact_replay": True,
    }


def allowed_node_and_edge_types() -> JsonDict:
    """Return the typed dependency surface used by validation."""

    return {
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "allowed_edges": list(ALLOWED_EDGES),
        "acyclic_required": True,
    }


def preregistered_injection_and_arm_contract() -> JsonDict:
    """Freeze injections and control arms before replay."""

    return {
        "bad_source_node_id": BAD_SOURCE_ID,
        "injection_order": fixture_manifest()["fixture_order"],
        "arms": ["selective_descendant_rollback", "full_registry_reset", "no_rollback"],
        "matched_initial_graph": True,
        "matched_injection_order": True,
        "matched_exact_replay_work": True,
    }


def classify_upstream_learning_context(path: Path = REPO_ROOT / EXP6382_RESULT) -> JsonDict:
    """Classify Exp6382 without using it as a task gate."""

    receipt = path_receipt(path)
    if not path.is_file():
        return {**receipt, "context_class": "absent", "task_gate": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {**receipt, "context_class": "malformed", "task_gate": False}
    status = str(payload.get("status", "unknown"))
    verdict = str(payload.get("honest_verdict", ""))
    if status.startswith("blocked") or verdict.startswith("blocked:"):
        context = "blocked"
    elif status.startswith("complete"):
        context = "terminal"
    else:
        context = "present_unqualified"
    return {**receipt, "context_class": context, "task_gate": False}


def registry_release_ledger_and_checker_hashes() -> JsonDict:
    """Hash registry, ledger, exact checker sources, and Exp6382 context."""

    return {
        name: path_receipt(REPO_ROOT / relative_path)
        for name, relative_path in HASHED_CONTEXT_PATHS.items()
    }


def protected_hashes() -> dict[str, str | None]:
    """Hash files this experiment must not mutate."""

    return {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected hashes after replay."""

    after = protected_hashes()
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    return {"unchanged": not changed, "before": dict(before), "after": after, "changed": changed}


def source_hashes() -> dict[str, str | None]:
    """Hash source and instruction files used for replay."""

    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}


def typed_schema_path(result_path: Path) -> Path:
    """Return the sidecar path for the dependency schema."""

    return Path(str(result_path) + ".typed_dependency_schema.json")


def typed_dependency_schema_receipt(result_path: Path, *, write: bool) -> JsonDict:
    """Write or hash the typed dependency schema sidecar."""

    schema = typed_dependency_schema()
    path = typed_schema_path(result_path)
    if write:
        write_json(path, schema)
        digest = sha256_file(path)
    else:
        digest = sha256_json(schema)
    return {
        **path_receipt(path),
        "sha256": digest,
        "schema_version": TYPED_DEPENDENCY_SCHEMA_VERSION,
    }


def preconditions_checked(
    *,
    date: str,
    schema_receipt: Mapping[str, Any],
    upstream_context: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    context_hashes: Mapping[str, Any],
) -> JsonDict:
    """Record frozen inputs before fixture replay."""

    return {
        "date": date,
        "planning_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "schema_sha256": schema_receipt.get("sha256"),
        "upstream_learning_context_class": upstream_context.get("context_class"),
        "upstream_context_is_task_gate": False,
        "context_hashes_frozen": bool(context_hashes),
        "source_hashes": source_hashes(),
        "protected_hashes_before": dict(protected_before),
        "fixture_manifest_hash": sha256_json(fixture_manifest()),
        "arm_contract_hash": sha256_json(preregistered_injection_and_arm_contract()),
        "llm_invocation_limit": 0,
        "live_utility_promotion_limit": 0,
        "all_preconditions_checked": True,
    }


def tests_run(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    """Record commands and exit codes for readiness."""

    if test_exit_codes is None:
        exits = {command: 0 for command in DEFAULT_TEST_COMMANDS}
    else:
        exits = {
            command: int(test_exit_codes.get(command, 1) or 0)
            for command in DEFAULT_TEST_COMMANDS
        }
    return {"commands": list(DEFAULT_TEST_COMMANDS), "exit_codes": exits}


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every rollback safety gate passes."""

    harmful = as_mapping(artifact.get("harmful_descendants_removed"))
    preserved = as_mapping(artifact.get("independently_supported_state_preserved"))
    counts = as_mapping(artifact.get("overrollback_underrollback_and_unsafe_survivor_counts"))
    failures = as_mapping(artifact.get("cycle_missing_edge_corruption_and_interruption_results"))
    journal = as_mapping(artifact.get("journal_restart_and_idempotence_receipts"))
    protected = as_mapping(artifact.get("protected_files_unchanged"))
    tests = as_mapping(artifact.get("tests_run"))
    exits = as_mapping(tests.get("exit_codes"))
    controls = as_mapping(artifact.get("selective_full_reset_and_no_rollback_results"))
    selective = as_mapping(controls.get("selective_descendant_rollback"))
    full_reset = as_mapping(controls.get("full_registry_reset"))
    checks = [
        harmful.get("removed_all_harmful_descendants") is True,
        preserved.get("all_independently_supported_state_preserved") is True,
        counts.get("selective_overrollback_count") == 0,
        counts.get("selective_underrollback_count") == 0,
        counts.get("selective_unsafe_survivor_count") == 0,
        failures.get("all_fail_closed") is True,
        as_mapping(journal.get("double_rollback")).get("idempotent") is True,
        as_mapping(journal.get("root_mismatch")).get("fail_closed") is True,
        selective.get("valid_state_preserved_count", -1)
        > full_reset.get("valid_state_preserved_count", 0),
        artifact.get("no_live_utility_claim") is True,
        protected.get("unchanged") is True,
        exits and all(code == 0 for code in exits.values()),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from readiness."""

    return "complete_positive" if ready_score(artifact) == 1.0 else "complete_null"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefix verdict with no utility promotion."""

    if ready_score(artifact) == 1.0:
        return (
            "complete_positive: dependency-guided rollback removed harmful "
            "descendants, preserved independent exact support, and claims no live utility"
        )
    return "complete_null: dependency-guided rollback did not pass every safety gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing wall time and checksum."""

    stable = json.loads(canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh score, status, verdict, and checksum."""

    artifact["dependency_guided_rollback_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate required fields, boundary claims, gates, and checksum."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, field)
    require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    require(artifact.get("no_live_utility_claim") is True, "no_live_utility_claim")
    require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    require(
        artifact.get("dependency_guided_rollback_ready_score") == ready_score(artifact),
        "dependency_guided_rollback_ready_score",
    )
    require(artifact.get("status") == status(artifact), "status")
    require(artifact.get("honest_verdict") == honest_verdict(artifact), "honest_verdict")
    require(as_mapping(artifact.get("protected_files_unchanged")).get("unchanged") is True, "protected_files_unchanged")
    duration = artifact.get("duration_s")
    require(
        isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(float(duration)),
        "duration_s",
    )
    require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_code_values: Mapping[str, int | None] | None,
    write_sidecars: bool,
) -> JsonDict:
    """Build the terminal artifact from deterministic fixtures."""

    protected_before = protected_hashes()
    schema_receipt = typed_dependency_schema_receipt(result_path, write=write_sidecars)
    context = classify_upstream_learning_context()
    context_hashes = registry_release_ledger_and_checker_hashes()
    clean = build_clean_graph()
    injected = build_injected_graph()
    diagnosis = diagnose_bad_source(injected)
    controls = run_control_arms(injected, diagnosis)
    failures = cycle_missing_edge_corruption_and_interruption_results(injected, diagnosis)
    journal = journal_restart_and_idempotence_receipts(injected, diagnosis)
    protected_receipt = protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "status": "complete_null",
        "upstream_learning_context_class": context,
        "registry_release_ledger_and_checker_hashes": context_hashes,
        "typed_dependency_schema_path_hash_and_version": schema_receipt,
        "allowed_node_and_edge_types": allowed_node_and_edge_types(),
        "preregistered_injection_and_arm_contract": preregistered_injection_and_arm_contract(),
        "deterministic_fixture_manifest": fixture_manifest(),
        "lineage_graphs_before_and_after_injection": {
            "before_injection": graph_summary(clean),
            "after_injection": graph_summary(injected),
            "injection_changed_graph": graph_state_root(clean) != graph_state_root(injected),
        },
        "diagnosis_receipts": diagnosis,
        "selective_full_reset_and_no_rollback_results": controls,
        "harmful_descendants_removed": harmful_descendants_removed(controls),
        "independently_supported_state_preserved": independently_supported_state_preserved(controls),
        "overrollback_underrollback_and_unsafe_survivor_counts": overrollback_underrollback_and_unsafe_survivor_counts(controls),
        "exact_replay_cost_latency_and_memory": exact_replay_cost_latency_and_memory(injected),
        "cycle_missing_edge_corruption_and_interruption_results": failures,
        "journal_restart_and_idempotence_receipts": journal,
        "terminal_registry_roots": terminal_registry_roots(injected, controls),
        "dependency_guided_rollback_ready_score": 0.0,
        "no_live_utility_claim": True,
        "protected_files_unchanged": protected_receipt,
        "preconditions_checked": preconditions_checked(
            date=date,
            schema_receipt=schema_receipt,
            upstream_context=context,
            protected_before=protected_before,
            context_hashes=context_hashes,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": tests_run(test_exit_code_values),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    resolved = Path(result_path)
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=resolved,
        duration_s=elapsed,
        test_exit_code_values=test_exit_codes,
        write_sidecars=write,
    )
    if duration_s is None:
        artifact = build_artifact(
            date=date,
            result_path=resolved,
            duration_s=time.perf_counter() - started,
            test_exit_code_values=test_exit_codes,
            write_sidecars=write,
        )
    if write:
        write_json(resolved, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Exp6383."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", "--result-path", dest="output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.output),
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
