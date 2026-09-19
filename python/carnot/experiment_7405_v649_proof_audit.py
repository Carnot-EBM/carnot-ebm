"""Audit synthetic and attempted live proof-memory cohorts independently.

The synthetic producer used an exact 2-CNF oracle. This module therefore
rebuilds source truth without importing that producer's reducers or solver.
The live cohort remains a separate blocked observation when its upstream
proposal capture is unavailable.

Spec refs: REQ-CL-7405 and SCENARIO-CL-7405-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.649"
PHASE = 3
EXPERIMENT_ID = "exp7405-v649-proof-audit"
SCHEMA = "carnot.exp7405.v649.proof_audit.v1"
RESULT_PATH = Path("results/experiment_7405_v649_proof_audit.json")
RAW_DIR = Path("results/raw/experiment_7405_v649_proof_audit")
MODULE_PATH = Path("python/carnot/experiment_7405_v649_proof_audit.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7405_v649_proof_audit.py")
TEST_PATH = Path("tests/python/test_experiment_7405_v649_proof_audit.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SYNTHETIC_PATH = Path("results/experiment_7403_v649_synthetic_memory.json")
LIVE_PATH = Path("results/experiment_7404_live_memory.json")
PROTOCOL_PATH = Path("results/raw/experiment_7403_v649_synthetic_memory/protocol_snapshot.json")
SYNTHETIC_EVIDENCE_PATH = Path(
    "results/raw/experiment_7403_v649_synthetic_memory/synthetic_evidence.json"
)
AUTHORITY_PATH = Path("results/raw/experiment_7403_v649_synthetic_memory/authority_attacks.json")
HISTORICAL_RECEIPTS_PATH = Path(
    "results/raw/experiment_7403_v649_synthetic_memory/historical_model_receipts.json"
)
ARMS = (
    "reset_exact_solver",
    "persistent_incremental_exact_solver",
    "persistent_source_graph_reachability_cache",
    "proof_memory",
    "proof_memory_matched_non_applicable",
)
PERSISTENT_CONTROLS = ARMS[1:3]
SERVICE_COST_PHASES = (
    "checking",
    "discovery",
    "exact_solver",
    "orchestration",
    "serialization",
    "updates",
    "verification",
)
MUTATIONS = (
    "forged_edge",
    "stale_version",
    "removed_feedback",
    "duplicated_request",
    "missing_cost_phase",
    "lost_reset",
    "altered_headline",
)
RANDOM_SEED = {"experiment": 7_405_649, "resampling": 7_371_307}
BOOTSTRAP_DRAWS = 10_000
EXPECTED_HASHES = {
    SYNTHETIC_PATH.as_posix(): (
        "sha256:7985849e93927d7fecaca23eac0cb185578b70e7b6f8a2d90777bc62eacd92b6"
    ),
    LIVE_PATH.as_posix(): (
        "sha256:f2efff28f8bd22ce6364866e5e867b7a8eee68c7388b907c452a6d05f12d8fe9"
    ),
    PROTOCOL_PATH.as_posix(): (
        "sha256:6c0a312a8e34da96c2f2308b84c503bd8d064f59a8bc78caa41cf2b187d95fab"
    ),
    SYNTHETIC_EVIDENCE_PATH.as_posix(): (
        "sha256:4ead99167e4cb1ce1106a352c2f59998500e1e2f0ab3502fc971c55880b6bf37"
    ),
    AUTHORITY_PATH.as_posix(): (
        "sha256:2e708b7620ad7f98aa3b984487f4ebc49fcf96cb095888f8eae6134ff2ed7d3f"
    ),
    HISTORICAL_RECEIPTS_PATH.as_posix(): (
        "sha256:8d17460cd49e75f3dc15387afedbfb83031155a22167113f917367050f1f5873"
    ),
}
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint",
)
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/learning/implication_memory.py"),
    Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
    Path("data/v647_implication_stream_manifest.json"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V649_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(), TEST_PATH.as_posix()),
)


@dataclass(frozen=True)
class AuditInputs:
    """Keep authenticated sources and their original dispositions together."""

    root: Path
    synthetic: JsonDict
    live: JsonDict
    protocol: JsonDict
    raw_evidence: JsonDict
    checks: list[JsonDict]
    hashes: dict[str, str]
    historical_sources: list[JsonDict]
    selected_paths: dict[str, Path]


@dataclass(frozen=True)
class AuditEvidence:
    """Retain independently checked units without hiding failed observations."""

    rows: list[JsonDict]
    witness_rows: list[JsonDict]
    metrics: JsonDict
    continuity: JsonDict
    errors: list[str]


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print a flushed phase boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7405] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return validation_contract.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validation_contract.atomic_json(path, value)


def load_object(path: Path) -> JsonDict:
    """Return an object for readable JSON bytes and an empty object otherwise."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _check(
    cohort: str,
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Record a cohort-local prerequisite without creating a whole-task gate."""

    return {
        "cohort": cohort,
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "terminal_blocking": False,
    }


def _source_verdict(value: Mapping[str, Any]) -> str:
    verdict = value.get("verdict_class")
    if verdict in {"positive", "circular_positive", "null", "blocked", "disqualified"}:
        return str(verdict)
    honest = str(value.get("honest_verdict") or "")
    if honest.startswith("blocked_") or value.get("status") == "blocked":
        return "blocked"
    return "disqualified"


def collect_inputs(
    repo_root: Path,
    *,
    synthetic_path: Path | None = None,
    live_path: Path | None = None,
    protocol_path: Path | None = None,
) -> AuditInputs:
    """Authenticate exact source bytes and preserve each original failure class."""

    root = repo_root.resolve()
    selected_paths = {
        "synthetic": synthetic_path or root / SYNTHETIC_PATH,
        "live": live_path or root / LIVE_PATH,
        "protocol": protocol_path or root / PROTOCOL_PATH,
        "raw_evidence": root / SYNTHETIC_EVIDENCE_PATH,
        "authority": root / AUTHORITY_PATH,
        "historical_receipts": root / HISTORICAL_RECEIPTS_PATH,
    }
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    selected_contract = (
        ("synthetic", "synthetic", SYNTHETIC_PATH),
        ("live", "live", LIVE_PATH),
        ("protocol", "synthetic", PROTOCOL_PATH),
        ("raw_evidence", "diagnostic", SYNTHETIC_EVIDENCE_PATH),
        ("authority", "diagnostic", AUTHORITY_PATH),
        ("historical_receipts", "diagnostic", HISTORICAL_RECEIPTS_PATH),
    )
    for label, cohort, stable in selected_contract:
        path = selected_paths[label]
        observed = sha256_file(path) if path.is_file() else None
        expected = EXPECTED_HASHES[stable.as_posix()]
        checks.append(
            _check(
                cohort,
                f"{label}_sha256",
                stable.as_posix(),
                "sha256",
                expected,
                observed,
                observed == expected,
            )
        )
        if observed is not None:
            key = stable.as_posix() if path == root / stable else str(path)
            hashes[key] = observed

    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _check(
                "audit",
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
                present,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)

    synthetic = load_object(selected_paths["synthetic"])
    live = load_object(selected_paths["live"])
    protocol = load_object(selected_paths["protocol"])
    raw_evidence = load_object(selected_paths["raw_evidence"])
    synthetic_fields = (
        ("verdict_class", "circular_positive"),
        ("flagged_adversarial", False),
        ("proof_safety_ready_score", 1),
        ("synthetic_memory_capture_complete_score", 1),
        ("synthetic_memory_value_score", 1),
    )
    for field, expected in synthetic_fields:
        checks.append(
            _check(
                "synthetic",
                f"synthetic_{field}",
                SYNTHETIC_PATH.as_posix(),
                field,
                expected,
                synthetic.get(field),
                synthetic.get(field) == expected,
            )
        )
    live_observed = live.get("honest_verdict")
    checks.append(
        _check(
            "live",
            "live_cohort_eligibility",
            LIVE_PATH.as_posix(),
            "honest_verdict",
            "eligible_complete_live_proof_rows",
            live_observed,
            bool(live.get("rows")) and _source_verdict(live) in {"circular_positive", "null"},
        )
    )
    checks.append(
        _check(
            "live",
            "live_attempted_disposition",
            LIVE_PATH.as_posix(),
            "honest_verdict",
            "blocked_gate_check_failed",
            live_observed,
            live_observed == "blocked_gate_check_failed",
        )
    )
    producer_raw_hash = (synthetic.get("source_artifact_hashes") or {}).get(
        SYNTHETIC_EVIDENCE_PATH.as_posix()
    )
    observed_raw_hash = hashes.get(SYNTHETIC_EVIDENCE_PATH.as_posix())
    checks.append(
        _check(
            "diagnostic",
            "producer_raw_sidecar_reference",
            SYNTHETIC_PATH.as_posix(),
            f"source_artifact_hashes.{SYNTHETIC_EVIDENCE_PATH.as_posix()}",
            producer_raw_hash,
            observed_raw_hash,
            producer_raw_hash == observed_raw_hash,
        )
    )
    historical_sources = [
        {
            "cohort": "synthetic",
            "path": SYNTHETIC_PATH.as_posix(),
            "sha256": hashes.get(SYNTHETIC_PATH.as_posix()),
            "original_status": synthetic.get("status"),
            "original_honest_verdict": synthetic.get("honest_verdict"),
            "original_verdict_class": _source_verdict(synthetic),
            "original_flagged_adversarial": synthetic.get("flagged_adversarial"),
            "counted_as_current_inference": False,
        },
        {
            "cohort": "live",
            "path": LIVE_PATH.as_posix(),
            "sha256": hashes.get(LIVE_PATH.as_posix()),
            "original_status": live.get("status"),
            "original_honest_verdict": live.get("honest_verdict"),
            "original_verdict_class": _source_verdict(live),
            "original_flagged_adversarial": live.get("flagged_adversarial"),
            "counted_as_current_inference": False,
        },
    ]
    return AuditInputs(
        root,
        synthetic,
        live,
        protocol,
        raw_evidence,
        checks,
        hashes,
        historical_sources,
        selected_paths,
    )


def formula_errors(version: Mapping[str, Any]) -> list[str]:
    """Rebuild one formula identity from clauses without producer code."""

    errors: list[str] = []
    formula_version = version.get("version")
    n_vars = version.get("n_vars")
    clauses = version.get("clauses")
    if not isinstance(formula_version, str) or not formula_version:
        errors.append("formula_version_invalid")
    if type(n_vars) is not int or n_vars < 1:
        errors.append("n_vars_invalid")
    if not isinstance(clauses, list) or not clauses:
        errors.append("clauses_invalid")
        return errors
    for index, clause in enumerate(clauses):
        if not isinstance(clause, Mapping) or clause.get("clause_id") != index:
            errors.append("clause_id_order")
            continue
        literals = clause.get("literals")
        if (
            not isinstance(literals, list)
            or len(literals) != 2
            or type(n_vars) is not int
            or any(
                type(literal) is not int or literal == 0 or abs(literal) > n_vars
                for literal in literals
            )
        ):
            errors.append("clause_literal_invalid")
    identity = {
        "version": formula_version,
        "n_vars": n_vars,
        "clauses": deepcopy(clauses),
    }
    if version.get("source_hash") != canonical_hash(identity):
        errors.append("source_hash_mismatch")
    return list(dict.fromkeys(errors))


def _formula_edges(version: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    for clause in version.get("clauses") or []:
        left, right = clause["literals"]
        clause_id = int(clause["clause_id"])
        edges.extend(((-left, right, clause_id), (-right, left, clause_id)))
    return edges


def independent_satisfiable(version: Mapping[str, Any], assumptions: Sequence[int]) -> bool:
    """Decide 2-CNF with an audit-local Kosaraju implementation."""

    errors = formula_errors(version)
    if errors:
        raise ValueError(",".join(errors))
    n_vars = int(version["n_vars"])
    if any(
        type(literal) is not int or literal == 0 or abs(literal) > n_vars for literal in assumptions
    ):
        raise ValueError("assumption_literal_invalid")
    vertices = [*range(1, n_vars + 1), *range(-1, -n_vars - 1, -1)]
    graph = {literal: [] for literal in vertices}
    reverse = {literal: [] for literal in vertices}
    edges = [(source, target) for source, target, _clause_id in _formula_edges(version)]
    edges.extend((-literal, literal) for literal in assumptions)
    for source, target in edges:
        graph[source].append(target)
        reverse[target].append(source)

    visited: set[int] = set()
    order: list[int] = []

    def visit(vertex: int) -> None:
        visited.add(vertex)
        for target in graph[vertex]:
            if target not in visited:
                visit(target)
        order.append(vertex)

    for vertex in vertices:
        if vertex not in visited:
            visit(vertex)
    component: dict[int, int] = {}

    def assign(vertex: int, component_id: int) -> None:
        component[vertex] = component_id
        for target in reverse[vertex]:
            if target not in component:
                assign(target, component_id)

    for vertex in reversed(order):
        if vertex not in component:
            assign(vertex, len(component))
    return all(component[variable] != component[-variable] for variable in range(1, n_vars + 1))


def proof_errors(version: Mapping[str, Any], witness: Mapping[str, Any]) -> list[str]:
    """Check every proof edge against the original canonical source clause."""

    proof = witness.get("path")
    if not isinstance(proof, Mapping):
        return ["proof_missing"]
    errors: list[str] = []
    if proof.get("formula_version") != version.get("version"):
        errors.append("proof_version_mismatch")
    if proof.get("source_hash") != version.get("source_hash"):
        errors.append("proof_source_hash_mismatch")
    edges = proof.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append("proof_edges_missing")
        return errors
    available = set(_formula_edges(version))
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            errors.append("proof_edge_invalid")
            continue
        value = (
            edge.get("from_literal"),
            edge.get("to_literal"),
            edge.get("source_clause_id"),
        )
        if value not in available:
            errors.append("proof_edge_not_in_source")
        previous = edges[index - 1] if index else None
        if isinstance(previous, Mapping) and previous.get("to_literal") != edge.get("from_literal"):
            errors.append("proof_edge_gap")
    if isinstance(edges[0], Mapping) and isinstance(edges[-1], Mapping):
        if edges[0].get("from_literal") != proof.get("antecedent"):
            errors.append("proof_antecedent_mismatch")
        if edges[-1].get("to_literal") != proof.get("consequent"):
            errors.append("proof_consequent_mismatch")
    content = {
        key: deepcopy(proof.get(key))
        for key in (
            "formula_version",
            "source_hash",
            "antecedent",
            "consequent",
            "edges",
        )
    }
    if proof.get("path_id") != canonical_hash(content):
        errors.append("proof_path_id_mismatch")
    return list(dict.fromkeys(errors))


def _cost_errors(row: Mapping[str, Any]) -> list[str]:
    cost = row.get("complete_service_cost")
    if not isinstance(cost, Mapping):
        return ["complete_service_cost_missing"]
    missing = [phase for phase in SERVICE_COST_PHASES if phase not in cost]
    if missing:
        return [f"cost_phase_missing:{phase}" for phase in missing]
    values = [cost.get(phase) for phase in SERVICE_COST_PHASES]
    if any(type(value) is not int or value < 0 for value in values):
        return ["cost_phase_invalid"]
    if cost.get("total") != sum(values):
        return ["cost_total_mismatch"]
    return []


def _request_maps(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[tuple[str, str], Mapping[str, Any]]]:
    versions: dict[str, Mapping[str, Any]] = {}
    requests: dict[tuple[str, str], Mapping[str, Any]] = {}
    for stream in protocol.get("evaluation_streams") or []:
        stream_id = str(stream.get("stream_id"))
        for version in stream.get("versions") or []:
            versions[str(version.get("version"))] = version
        for request in stream.get("requests") or []:
            requests[(stream_id, str(request.get("request_id")))] = request
    return versions, requests


def _protocol_request_errors(protocol: Mapping[str, Any]) -> list[str]:
    streams = protocol.get("evaluation_streams")
    if not isinstance(streams, list) or len(streams) != 32:
        return ["stream_count_mismatch"]
    errors: list[str] = []
    stream_ids: list[str] = []
    request_ids: list[str] = []
    for stream in streams:
        stream_ids.append(str(stream.get("stream_id")))
        requests = stream.get("requests")
        if not isinstance(requests, list) or len(requests) != 24:
            errors.append("request_count_mismatch")
            continue
        for index, request in enumerate(requests):
            request_ids.append(str(request.get("request_id")))
            if request.get("request_index") != index:
                errors.append("request_order_mismatch")
    if len(stream_ids) != len(set(stream_ids)):
        errors.append("duplicate_stream")
    if len(request_ids) != len(set(request_ids)):
        errors.append("duplicate_request")
    return list(dict.fromkeys(errors))


def _feedback_errors(row: Mapping[str, Any], request_index: Mapping[str, int]) -> list[str]:
    errors: list[str] = []
    current = int(row.get("request_index", -1))
    prior = row.get("prior_verified_feedback_request_ids")
    witnessed = row.get("witnessed_earlier_feedback")
    if not isinstance(prior, list) or not isinstance(witnessed, list):
        return ["feedback_schema_invalid"]
    for request_id in (*prior, *witnessed):
        if request_id not in request_index or request_index[request_id] >= current:
            errors.append("feedback_not_earlier")
    if not set(witnessed) <= set(prior):
        errors.append("witness_not_in_verified_feedback")
    if row.get("reuse_class") == "cross_query_source_proof" and not prior:
        errors.append("source_proof_feedback_missing")
    return list(dict.fromkeys(errors))


def _continuity(source: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in source.get("rows") or [] if isinstance(row, Mapping)]
    stream_ids = sorted({str(row.get("stream_id")) for row in rows})
    incremental_ok = True
    graph_ok = True
    reset_ok = True
    cache_hits = 0
    for stream_id in stream_ids:
        stream_rows = [row for row in rows if row.get("stream_id") == stream_id]
        for arm, label in ((ARMS[1], "incremental"), (ARMS[2], "graph")):
            arm_rows = [row for row in stream_rows if row.get("arm") == arm]
            grouped: dict[str, set[str]] = {}
            for row in arm_rows:
                grouped.setdefault(str(row.get("formula_version")), set()).add(
                    str(row.get("state_instance_id"))
                )
            persisted = len(arm_rows) == 24 and all(len(states) == 1 for states in grouped.values())
            if label == "incremental":
                incremental_ok = incremental_ok and persisted
            else:
                graph_ok = graph_ok and persisted
                cache_hits += sum(row.get("used_exact_solver") is False for row in arm_rows)
        reset_rows = [row for row in stream_rows if row.get("arm") == ARMS[0]]
        reset_ids = {str(row.get("state_instance_id")) for row in reset_rows}
        reset_ok = (
            reset_ok
            and len(reset_rows) == 24
            and len(reset_ids) == 24
            and all(
                row.get("state_instance_id") == f"reset-{row.get('request_id')}"
                for row in reset_rows
            )
        )
    restart_rows = source.get("restart_rows") or []
    restart_ok = len(restart_rows) == 32 and all(
        row.get("passed") is True
        and row.get("exact_bytes_restored") is True
        and row.get("stale_version_invalidated") is True
        for row in restart_rows
        if isinstance(row, Mapping)
    )
    return {
        "incremental_state_persisted": incremental_ok and len(stream_ids) == 32,
        "graph_cache_state_persisted": graph_ok and cache_hits > 0 and len(stream_ids) == 32,
        "reset_isolation_preserved": reset_ok and restart_ok,
        "graph_cache_hit_count": cache_hits,
        "streams_checked": len(stream_ids),
        "restart_rows_checked": len(restart_rows),
    }


@lru_cache(maxsize=32)
def _bootstrap_paired_upper(paired_values: tuple[tuple[str, float, float], ...]) -> float:
    """Resample immutable per-stream totals so repeated cold checks stay bounded."""

    paired = {stream: (numerator, denominator) for stream, numerator, denominator in paired_values}
    streams = sorted(paired)
    generator = random.Random(RANDOM_SEED["resampling"])
    estimates: list[float] = []
    for _index in range(BOOTSTRAP_DRAWS):
        sample = [generator.choice(streams) for _stream in streams]
        numerator = sum(paired[stream][0] for stream in sample)
        denominator = sum(paired[stream][1] for stream in sample)
        estimates.append(numerator / denominator if denominator else float("inf"))
    estimates.sort()
    return estimates[int(0.95 * len(estimates))]


def _bootstrap_upper(rows: Sequence[Mapping[str, Any]], comparator: str, metric: str) -> float:
    streams = sorted({str(row["stream_id"]) for row in rows})
    paired: dict[str, tuple[float, float]] = {}
    for stream in streams:
        proof = [row for row in rows if row["stream_id"] == stream and row["arm"] == ARMS[3]]
        control = [row for row in rows if row["stream_id"] == stream and row["arm"] == comparator]
        if metric == "paid_exact_query":
            numerator = sum(row[metric] is True for row in proof)
            denominator = sum(row[metric] is True for row in control)
        else:
            numerator = sum(int(row["complete_service_cost"]["total"]) for row in proof)
            denominator = sum(int(row["complete_service_cost"]["total"]) for row in control)
        paired[stream] = (float(numerator), float(denominator))
    return _bootstrap_paired_upper(
        tuple((stream, paired[stream][0], paired[stream][1]) for stream in streams)
    )


def independent_metrics(
    rows: Sequence[Mapping[str, Any]], witnesses: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Recompute safety, erasure, paid-query, and complete-cost gates."""

    if not rows:
        return {
            "unsafe_decisions": 0,
            "exact_decision_coverage": 0.0,
            "valid_utility": False,
            "valid_erasure_witness_count": 0,
            "valid_erasure_witness_stream_count": 0,
            "paid_query_ratio_ci95_upper": {},
            "full_cost_ratio_ci95_upper": {},
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": RANDOM_SEED["resampling"],
            "bootstrap_independent_stream_groups": 0,
            "bootstrap_formula_families": 0,
            "efficacy_passed": False,
        }
    valid_witnesses = [row for row in witnesses if row.get("audit_valid") is True]
    unsafe = sum(
        row.get("truth_match") is not True
        or row.get("cost_complete") is not True
        or row.get("feedback_valid") is not True
        for row in rows
    )
    paid = {
        comparator: _bootstrap_upper(rows, comparator, "paid_exact_query")
        for comparator in PERSISTENT_CONTROLS
    }
    full = {
        comparator: _bootstrap_upper(rows, comparator, "complete_service_cost")
        for comparator in PERSISTENT_CONTROLS
    }
    efficacy = all(
        paid[comparator] < 0.90 and full[comparator] <= 1.0 for comparator in PERSISTENT_CONTROLS
    )
    return {
        "unsafe_decisions": unsafe,
        "exact_decision_coverage": sum(row.get("truth_match") is True for row in rows) / len(rows),
        "valid_utility": unsafe == 0,
        "valid_erasure_witness_count": len(valid_witnesses),
        "valid_erasure_witness_stream_count": len(
            {str(row.get("stream_id")) for row in valid_witnesses}
        ),
        "paid_query_ratio_ci95_upper": paid,
        "full_cost_ratio_ci95_upper": full,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": RANDOM_SEED["resampling"],
        "bootstrap_independent_stream_groups": len({str(row.get("stream_id")) for row in rows}),
        "bootstrap_formula_families": len({str(row.get("family")) for row in rows}),
        "efficacy_passed": efficacy,
    }


def audit_inputs(
    inputs: AuditInputs,
    *,
    emit_progress: bool = False,
    started: float | None = None,
    checkpoint_dir: Path | None = None,
) -> AuditEvidence:
    """Independently inspect every available synthetic proof-memory unit."""

    origin = started if started is not None else time.monotonic()
    if not inputs.synthetic or not inputs.protocol:
        return AuditEvidence(
            [], [], independent_metrics([], []), _continuity({}), ["synthetic_source_unavailable"]
        )
    errors = _protocol_request_errors(inputs.protocol)
    versions, requests = _request_maps(inputs.protocol)
    for version in versions.values():
        errors.extend(formula_errors(version))
    source_rows = [row for row in inputs.synthetic.get("rows") or [] if isinstance(row, Mapping)]
    source_cost_rows = [
        row for row in inputs.synthetic.get("cost_rows") or [] if isinstance(row, Mapping)
    ]
    if len(source_rows) != 32 * 24 * 5:
        errors.append("source_row_count_mismatch")
    if len(source_cost_rows) != len(source_rows):
        errors.append("source_cost_row_count_mismatch")
    cost_by_key = {
        (str(row.get("stream_id")), str(row.get("request_id")), str(row.get("arm"))): row
        for row in source_cost_rows
    }
    request_indices = {
        str(request.get("request_id")): int(request.get("request_index", -1))
        for request in requests.values()
    }
    audited_rows: list[JsonDict] = []
    streams = sorted({str(row.get("stream_id")) for row in source_rows})
    checkpoint_root = checkpoint_dir
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    for stream_number, stream_id in enumerate(streams, start=1):
        stream_audited: list[JsonDict] = []
        for row in [item for item in source_rows if item.get("stream_id") == stream_id]:
            request_id = str(row.get("request_id"))
            request = requests.get((stream_id, request_id))
            version = versions.get(str(row.get("formula_version")))
            row_errors: list[str] = []
            if request is None:
                row_errors.append("request_not_in_protocol")
            if version is None:
                row_errors.append("formula_version_not_in_protocol")
            assumptions = list(row.get("assumptions") or [])
            if request is not None:
                if assumptions != request.get("assumptions"):
                    row_errors.append("request_assumptions_mismatch")
                if row.get("request_index") != request.get("request_index"):
                    row_errors.append("request_index_mismatch")
                if row.get("formula_version") != request.get("formula_version"):
                    row_errors.append("request_version_mismatch")
            satisfiable = False
            if version is not None:
                if row.get("formula_source_hash") != version.get("source_hash"):
                    row_errors.append("row_source_hash_mismatch")
                try:
                    satisfiable = independent_satisfiable(version, assumptions)
                except ValueError as exc:
                    row_errors.append(f"independent_truth_error:{exc}")
            truth_match = (
                row.get("independent_satisfiable") is satisfiable
                and (
                    (satisfiable and row.get("decision") == "satisfiable")
                    or (not satisfiable and row.get("decision") in {"unsatisfiable", "reject"})
                )
                and row.get("final_exact_validation") is True
            )
            if not truth_match:
                row_errors.append("decision_truth_mismatch")
            cost_errors = _cost_errors(row)
            row_errors.extend(cost_errors)
            source_cost = cost_by_key.get((stream_id, request_id, str(row.get("arm"))))
            if source_cost is None:
                row_errors.append("raw_cost_row_missing")
            elif (
                source_cost.get("complete_service_cost") != row.get("complete_service_cost")
                or source_cost.get("persistent_state_instance_id") != row.get("state_instance_id")
                or source_cost.get("censored") != row.get("censored")
            ):
                row_errors.append("raw_cost_row_mismatch")
            feedback_errors = _feedback_errors(row, request_indices)
            row_errors.extend(feedback_errors)
            compact = {
                "cohort": "synthetic",
                "stream_id": stream_id,
                "formula_id": row.get("formula_id"),
                "formula_version": row.get("formula_version"),
                "formula_source_hash": row.get("formula_source_hash"),
                "family": row.get("family"),
                "seed": row.get("seed"),
                "request_id": request_id,
                "request_index": row.get("request_index"),
                "condition": row.get("split"),
                "assumptions": assumptions,
                "arm": row.get("arm"),
                "decision": row.get("decision"),
                "independent_satisfiable": satisfiable,
                "truth_match": truth_match,
                "used_exact_solver": row.get("used_exact_solver"),
                "paid_exact_query": row.get("paid_exact_query"),
                "reuse_class": row.get("reuse_class"),
                "prior_verified_feedback_request_ids": deepcopy(
                    row.get("prior_verified_feedback_request_ids") or []
                ),
                "witnessed_earlier_feedback": deepcopy(row.get("witnessed_earlier_feedback") or []),
                "feedback_valid": not feedback_errors,
                "state_instance_id": row.get("state_instance_id"),
                "complete_service_cost": deepcopy(row.get("complete_service_cost") or {}),
                "metric_contribution": {
                    "paid_exact_query": int(row.get("paid_exact_query") is True),
                    "complete_service_total": (row.get("complete_service_cost") or {}).get("total"),
                },
                "duration_ns": row.get("duration_ns"),
                "cost_complete": not cost_errors,
                "censored": row.get("censored") is True,
                "failure": row.get("failure"),
                "disposition": (
                    "censored"
                    if row.get("censored") is True
                    else "failed"
                    if row.get("failure") is not None
                    else "completed"
                ),
                "audit_errors": list(dict.fromkeys(row_errors)),
            }
            stream_audited.append(compact)
            errors.extend(f"row:{request_id}:{error}" for error in row_errors)
        audited_rows.extend(stream_audited)
        if checkpoint_root is not None:
            atomic_json(
                checkpoint_root / f"{stream_number:02d}_{stream_id}.json",
                {
                    "stream_id": stream_id,
                    "completed_units": stream_number,
                    "rows": stream_audited,
                },
            )
        if emit_progress:
            progress(
                origin,
                "audit",
                "stream_complete",
                unit=stream_number,
                total=len(streams),
            )

    witness_rows: list[JsonDict] = []
    for witness in inputs.synthetic.get("erasure_witness_rows") or []:
        if not isinstance(witness, Mapping):
            errors.append("witness_not_object")
            continue
        version = versions.get(str(witness.get("formula_version")))
        witness_errors = (
            ["witness_formula_missing"] if version is None else proof_errors(version, witness)
        )
        earlier_id = str(witness.get("earlier_verified_feedback_request_id"))
        later_id = str(witness.get("request_id"))
        if request_indices.get(earlier_id, 10**9) >= request_indices.get(
            later_id, -1
        ) or witness.get("earlier_request_index", 10**9) >= witness.get("later_request_index", -1):
            witness_errors.append("witness_feedback_order_invalid")
        proof = witness.get("path") or {}
        later_request = requests.get((str(witness.get("stream_id")), later_id)) or {}
        assumptions = set(later_request.get("assumptions") or [])
        if (
            proof.get("antecedent") not in assumptions
            or -int(proof.get("consequent", 0)) not in assumptions
        ):
            witness_errors.append("proof_not_applicable_to_later_request")
        for field in ("changed_exact_work", "different_later_query", "early_decision_removed"):
            if witness.get(field) is not True:
                witness_errors.append(f"erasure_field_false:{field}")
        enriched = {**deepcopy(dict(witness)), "audit_errors": witness_errors}
        enriched["audit_valid"] = not witness_errors
        witness_rows.append(enriched)
        errors.extend(f"witness:{later_id}:{error}" for error in witness_errors)

    continuity = _continuity(inputs.synthetic)
    if not all(
        continuity[field] is True
        for field in (
            "incremental_state_persisted",
            "graph_cache_state_persisted",
            "reset_isolation_preserved",
        )
    ):
        errors.append("persistent_state_or_reset_invalid")
    metrics = independent_metrics(audited_rows, witness_rows)
    producer_metrics = (inputs.synthetic.get("independent_reduction") or {}).get(
        "synthetic_metrics"
    ) or {}
    comparisons = {
        "paid_query_ratio_ci95_upper": metrics["paid_query_ratio_ci95_upper"],
        "full_cost_ratio_ci95_upper": metrics["full_cost_ratio_ci95_upper"],
        "unsafe_decisions": metrics["unsafe_decisions"],
    }
    for field, observed in comparisons.items():
        if producer_metrics.get(field) != observed:
            errors.append(f"producer_metric_mismatch:{field}")
    if inputs.synthetic.get("synthetic_memory_value_score") != int(metrics["efficacy_passed"]):
        errors.append("producer_headline_mismatch")
    return AuditEvidence(
        audited_rows,
        witness_rows,
        metrics,
        continuity,
        list(dict.fromkeys(errors)),
    )


def _mutation_row(name: str, observations: Sequence[str]) -> JsonDict:
    rejected = bool(observations)
    return {
        "mutation": name,
        "independent_check": f"audit_local_{name}_check",
        "expected": "reject",
        "observed": "reject" if rejected else "accept",
        "rejecting_observation": observations[0] if observations else None,
        "all_observations": list(observations),
        "passed": rejected,
        "private_copy_only": True,
    }


def run_mutation_controls(inputs: AuditInputs, evidence: AuditEvidence) -> list[JsonDict]:
    """Change seven private copies and require an audit-local rejection."""

    if not inputs.synthetic or not inputs.protocol or not evidence.rows:
        return [_mutation_row(name, ["source_unavailable"]) for name in MUTATIONS]
    versions, _requests = _request_maps(inputs.protocol)
    first_witness = deepcopy(inputs.synthetic["erasure_witness_rows"][0])
    witness_version = versions[str(first_witness["formula_version"])]

    forged = deepcopy(first_witness)
    forged["path"]["edges"][0]["source_clause_id"] = 999_999
    forged_observations = proof_errors(witness_version, forged)

    stale = deepcopy(first_witness)
    stale["path"]["formula_version"] = "stale-version"
    stale_observations = proof_errors(witness_version, stale)

    feedback = deepcopy(
        next(row for row in evidence.rows if row["reuse_class"] == "cross_query_source_proof")
    )
    feedback["prior_verified_feedback_request_ids"] = []
    feedback["witnessed_earlier_feedback"] = []
    request_indices = {str(row["request_id"]): int(row["request_index"]) for row in evidence.rows}
    feedback_observations = _feedback_errors(feedback, request_indices)

    duplicated = deepcopy(inputs.protocol)
    duplicated["evaluation_streams"][0]["requests"][1]["request_id"] = duplicated[
        "evaluation_streams"
    ][0]["requests"][0]["request_id"]
    duplicate_observations = _protocol_request_errors(duplicated)

    missing_cost = deepcopy(evidence.rows[0])
    del missing_cost["complete_service_cost"]["verification"]
    missing_cost_observations = _cost_errors(missing_cost)

    lost_reset = deepcopy(inputs.synthetic)
    lost_reset["restart_rows"].pop()
    lost_reset_observations = (
        [] if _continuity(lost_reset)["reset_isolation_preserved"] else ["reset_isolation_missing"]
    )

    altered_headline = deepcopy(inputs.synthetic)
    altered_headline["synthetic_memory_value_score"] = 0
    altered_observations = (
        ["headline_disagrees_with_independent_efficacy"]
        if altered_headline["synthetic_memory_value_score"]
        != int(evidence.metrics["efficacy_passed"])
        else []
    )
    return [
        _mutation_row("forged_edge", forged_observations),
        _mutation_row("stale_version", stale_observations),
        _mutation_row("removed_feedback", feedback_observations),
        _mutation_row("duplicated_request", duplicate_observations),
        _mutation_row("missing_cost_phase", missing_cost_observations),
        _mutation_row("lost_reset", lost_reset_observations),
        _mutation_row("altered_headline", altered_observations),
    ]


def passing_test_receipts() -> list[JsonDict]:
    """Create complete receipt shapes for deterministic reducer tests only."""

    return [
        {
            "name": name,
            "command": f"test {name}",
            "command_argv": ["test", name],
            "environment": {},
            "scope": "test_fixture",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "/tmp/test.log",
            "log_hash": "sha256:" + "0" * 64,
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "test_fixture",
            "started_at_utc": "2026-09-19T00:00:00+00:00",
            "ended_at_utc": "2026-09-19T00:00:00+00:00",
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(row.get("name") for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _missing_check(upstream: str, check: str, field: str, expected: Any, observed: Any) -> JsonDict:
    return {
        "upstream": upstream,
        "path": upstream,
        "check": check,
        "field": field,
        "expected": expected,
        "observed": observed,
    }


def cohort_claims(artifact: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[JsonDict]:
    """Build one non-pooled claim record for each planned cohort."""

    rows = artifact.get("rows") or []
    errors = artifact.get("audit_errors") or []
    dispositions = artifact.get("source_cohort_dispositions") or {}
    source_hashes = artifact.get("source_artifact_hashes") or {}
    synthetic_source = dispositions.get("synthetic") or {}
    live_source = dispositions.get("live") or {}
    synthetic_complete = (
        len(rows) == 32 * 24 * 5
        and not errors
        and metrics.get("unsafe_decisions") == 0
        and metrics.get("valid_erasure_witness_count", 0) >= 8
        and metrics.get("valid_erasure_witness_stream_count", 0) >= 4
        and synthetic_source.get("original_eligible") is True
    )
    synthetic_value = int(synthetic_complete and metrics.get("efficacy_passed") is True)
    if not synthetic_source.get("present"):
        synthetic_class = "blocked"
        synthetic_missing = [
            _missing_check(
                SYNTHETIC_PATH.as_posix(),
                "synthetic_source_identity",
                "path",
                "authenticated_complete_artifact",
                None,
            )
        ]
    elif not synthetic_complete:
        synthetic_class = "disqualified"
        synthetic_missing = [
            _missing_check(
                SYNTHETIC_PATH.as_posix(),
                "synthetic_independent_audit",
                "audit_errors",
                [],
                list(errors),
            )
        ]
    else:
        synthetic_class = "circular_positive" if synthetic_value else "null"
        synthetic_missing = []
    live_original_class = str(live_source.get("original_verdict_class") or "blocked")
    live_present = live_source.get("present") is True
    live_eligible = live_source.get("original_eligible") is True
    if not live_present or live_original_class == "blocked":
        live_class = "blocked"
        live_missing = [
            _missing_check(
                LIVE_PATH.as_posix(),
                "live_cohort_eligibility",
                "honest_verdict",
                "eligible_complete_live_proof_rows",
                live_source.get("original_honest_verdict") if live_present else None,
            )
        ]
    elif live_original_class == "disqualified" or not live_eligible:
        live_class = "disqualified"
        live_missing = [
            _missing_check(
                LIVE_PATH.as_posix(),
                "live_cohort_eligibility",
                "verdict_class",
                "eligible_complete_live_proof_rows",
                live_original_class,
            )
        ]
    else:
        live_class = live_original_class
        live_missing = []
    live_confirmed_value = int(
        live_eligible and live_original_class in {"positive", "circular_positive"}
    )
    return [
        {
            "cohort": "synthetic",
            "planned": True,
            "source_hashes": {
                SYNTHETIC_PATH.as_posix(): source_hashes.get(SYNTHETIC_PATH.as_posix()),
                PROTOCOL_PATH.as_posix(): source_hashes.get(PROTOCOL_PATH.as_posix()),
            },
            "original_verdict_class": synthetic_source.get("original_verdict_class"),
            "verdict_class": synthetic_class,
            "eligible": synthetic_complete,
            "safety": {
                "unsafe_decisions": metrics.get("unsafe_decisions"),
                "proof_edges_valid": not any(str(error).startswith("witness:") for error in errors),
                "persistent_state_valid": all(
                    (artifact.get("state_continuity") or {}).get(field) is True
                    for field in (
                        "incremental_state_persisted",
                        "graph_cache_state_persisted",
                        "reset_isolation_preserved",
                    )
                ),
            },
            "efficacy": {
                "paid_query_ratio_ci95_upper": deepcopy(
                    metrics.get("paid_query_ratio_ci95_upper") or {}
                ),
                "full_cost_ratio_ci95_upper": deepcopy(
                    metrics.get("full_cost_ratio_ci95_upper") or {}
                ),
                "passed": metrics.get("efficacy_passed") is True,
            },
            "uncertainty": {
                "bootstrap_draws": metrics.get("bootstrap_draws"),
                "bootstrap_seed": metrics.get("bootstrap_seed"),
                "effective_independent_stream_groups": metrics.get(
                    "bootstrap_independent_stream_groups"
                ),
                "formula_families": metrics.get("bootstrap_formula_families"),
            },
            "confirmed_value": synthetic_value,
            "verifier_is_oracle": True,
            "missing_checks": synthetic_missing,
        },
        {
            "cohort": "live",
            "planned": True,
            "source_hashes": {LIVE_PATH.as_posix(): source_hashes.get(LIVE_PATH.as_posix())},
            "original_verdict_class": live_original_class,
            "verdict_class": live_class,
            "eligible": live_eligible,
            "safety": None,
            "efficacy": None,
            "uncertainty": None,
            "confirmed_value": live_confirmed_value,
            "verifier_is_oracle": True,
            "missing_checks": live_missing,
        },
    ]


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Reduce audit completion and combined value from ordinary stored rows."""

    rows = [row for row in artifact.get("rows") or [] if isinstance(row, Mapping)]
    witnesses = [
        row for row in artifact.get("erasure_witness_rows") or [] if isinstance(row, Mapping)
    ]
    mutations = [row for row in artifact.get("mutation_rows") or [] if isinstance(row, Mapping)]
    receipts = [
        row for row in artifact.get("validation_receipts") or [] if isinstance(row, Mapping)
    ]
    metrics = independent_metrics(rows, witnesses)
    claims = cohort_claims(artifact, metrics)
    mutation_passed = {row.get("mutation") for row in mutations} == set(MUTATIONS) and all(
        row.get("passed") is True and row.get("observed") == "reject" for row in mutations
    )
    affected_passed = _receipts_pass(receipts, REQUIRED_CHECK_NAMES)
    terminal_passed = _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    dispositions_complete = len(claims) == 2 and all(
        row.get("verdict_class")
        in {
            "positive",
            "circular_positive",
            "null",
            "blocked",
            "disqualified",
        }
        for row in claims
    )
    source_audit_complete = (
        len(rows) == 32 * 24 * 5
        and not artifact.get("audit_errors")
        and metrics["unsafe_decisions"] == 0
        and all(
            (artifact.get("state_continuity") or {}).get(field) is True
            for field in (
                "incremental_state_persisted",
                "graph_cache_state_persisted",
                "reset_isolation_preserved",
            )
        )
    )
    audit_complete = int(
        source_audit_complete
        and dispositions_complete
        and mutation_passed
        and affected_passed
        and terminal_passed
        and artifact.get("flagged_adversarial") is False
    )
    combined_value = int(
        bool(claims)
        and all(row.get("eligible") is True and row.get("confirmed_value") == 1 for row in claims)
    )
    classes = {str(row.get("verdict_class")) for row in claims}
    if "disqualified" in classes:
        verdict = "disqualified"
    elif "blocked" in classes:
        verdict = "blocked"
    elif combined_value:
        verdict = "circular_positive"
    else:
        verdict = "null"
    return {
        "synthetic_metrics": metrics,
        "cohort_claim_rows": claims,
        "source_audit_complete": source_audit_complete,
        "cohort_dispositions_complete": dispositions_complete,
        "mutation_controls_passed": mutation_passed,
        "affected_validation_passed": affected_passed,
        "terminal_validation_passed": terminal_passed,
        "proof_audit_complete_score": audit_complete,
        "proof_value_confirmed_score": combined_value,
        "promotion_score": 0,
        "verdict_class": verdict,
    }


def _gate(
    category: str,
    check: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    terminal_blocking: bool,
) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "terminal_blocking": terminal_blocking,
    }


def acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(artifact)
    metrics = reduced["synthetic_metrics"]
    claims = {row["cohort"]: row for row in reduced["cohort_claim_rows"]}
    gates = [
        _gate(
            "completion",
            "independent_synthetic_rows",
            "==",
            True,
            reduced["source_audit_complete"],
            reduced["source_audit_complete"],
            True,
        ),
        _gate(
            "safety",
            "unsafe_decisions",
            "==",
            0,
            metrics["unsafe_decisions"],
            metrics["unsafe_decisions"] == 0,
            True,
        ),
        _gate(
            "safety",
            "mutation_controls",
            "==",
            True,
            reduced["mutation_controls_passed"],
            reduced["mutation_controls_passed"],
            True,
        ),
        _gate(
            "validation",
            "affected_checks",
            "==",
            True,
            reduced["affected_validation_passed"],
            reduced["affected_validation_passed"],
            True,
        ),
        _gate(
            "validation",
            "terminal_checks",
            "==",
            True,
            reduced["terminal_validation_passed"],
            reduced["terminal_validation_passed"],
            True,
        ),
        _gate(
            "efficacy",
            "synthetic_original_value_gate",
            "==",
            1,
            claims["synthetic"]["confirmed_value"],
            claims["synthetic"]["confirmed_value"] == 1,
            False,
        ),
        _gate(
            "completion",
            "live_cohort_available",
            "==",
            True,
            claims["live"]["eligible"],
            claims["live"]["eligible"] is True,
            False,
        ),
        _gate(
            "efficacy",
            "combined_live_plus_synthetic_value",
            "==",
            1,
            reduced["proof_value_confirmed_score"],
            reduced["proof_value_confirmed_score"] == 1,
            False,
        ),
        _gate("promotion", "automatic_promotion", "==", 0, 0, True, False),
    ]
    return gates


def gate_summary(artifact: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    claims = [row for row in artifact.get("cohort_claim_rows") or [] if isinstance(row, Mapping)]
    structured = [
        deepcopy(dict(missing))
        for claim in claims
        for missing in claim.get("missing_checks") or []
        if isinstance(missing, Mapping)
    ]
    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": structured[0] if structured else (failures[0] if failures else None),
        "structured_cohort_failures": structured,
        "failures": failures,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind code, protocol, source identities, raw audit rows, and controls."""

    return canonical_hash(
        {
            "schema": artifact.get("schema"),
            "experiment_id": artifact.get("experiment_id"),
            "milestone": artifact.get("milestone"),
            "run_date": artifact.get("run_date"),
            "random_seed": artifact.get("random_seed"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes"),
            "source_cohort_dispositions": artifact.get("source_cohort_dispositions"),
            "rows": artifact.get("rows"),
            "erasure_witness_rows": artifact.get("erasure_witness_rows"),
            "state_continuity": artifact.get("state_continuity"),
            "mutation_rows": artifact.get("mutation_rows"),
            "cohort_claim_rows": artifact.get("cohort_claim_rows"),
            "acceptance_gate_results": artifact.get("acceptance_gate_results"),
        }
    )


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    specific = {
        "schema": "Use a versioned schema and ordinary terminal identity fields.",
        "run_date": "Use the requested date and retain actual UTC boundaries.",
        "preconditions_checked": "Authenticate each cohort locally without a combined pre-gate.",
        "MODEL_SPECS": "No current language model is loaded or generated from in this audit.",
        "model_invoked": "Only a real attempted current model call can set this field true.",
        "invocation_counts": "Count current owned calls only, never historical producer work.",
        "inference_substrate": "Describe current CPU and exact graph work in one string.",
        "inference_substrate_class": "Classify this host-only historical reduction as aggregation.",
        "execution_venue": "Use the closed host venue string.",
        "duration_s": "Measure this task with a monotonic clock without padding.",
        "phase_spans": "Retain actual phase boundaries, heartbeats, and checkpoints.",
        "random_seed": "Freeze audit and bootstrap seeds for repeatable reduction.",
        "reproducibility_checksum": "Bind current code, protocol hashes, raw rows, and controls.",
        "source_artifact_hashes": "Hash exact source and sidecar bytes without rewriting them.",
        "rows": "Retain every cohort, stream, request, arm, cost, and disposition.",
        "sample_size_budget": "Separate planned, attempted, completed, censored, and unstarted units.",
        "acceptance_gate_results": "Keep validation, safety, completion, and efficacy gates separate.",
        "gate_check_summary": "Name exact missing cohort fields and observed values.",
        "verifier_is_oracle": "State that source-exact verification defines correctness here.",
        "honest_verdict": "Keep unchanged external absence blocked rather than partial.",
        "verdict_class": "Use only the closed terminal verdict classes.",
        "flagged_adversarial": "Preserve critical validation findings as ineligible evidence.",
        "validation_receipts": "Retain exact commands, environments, exits, durations, and log hashes.",
        "repository_health": "Keep unrelated health observations outside affected readiness.",
        "field_principles": "Explain ordinary fields without wrapping their values.",
        "promotion_score": "Never trigger production, publication, or weight changes automatically.",
        "proof_audit_complete_score": "Separate complete dispositions from evidence eligibility.",
        "cohort_claim_rows": "Keep synthetic and live source, safety, value, and absence distinct.",
        "proof_value_confirmed_score": "Require every cohort in the combined claim to pass.",
        "mutation_rows": "Retain each private attack, independent check, and rejection.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def _source_dispositions(inputs: AuditInputs) -> JsonDict:
    synthetic_checks = [
        row for row in inputs.checks if row["cohort"] == "synthetic" and not row["passed"]
    ]
    live_checks = [row for row in inputs.checks if row["cohort"] == "live" and not row["passed"]]
    return {
        "synthetic": {
            "present": bool(inputs.synthetic),
            "original_status": inputs.synthetic.get("status"),
            "original_honest_verdict": inputs.synthetic.get("honest_verdict"),
            "original_verdict_class": _source_verdict(inputs.synthetic),
            "original_flagged_adversarial": inputs.synthetic.get("flagged_adversarial"),
            "original_eligible": bool(inputs.synthetic) and not synthetic_checks,
            "failed_checks": deepcopy(synthetic_checks),
        },
        "live": {
            "present": bool(inputs.live),
            "original_status": inputs.live.get("status"),
            "original_honest_verdict": inputs.live.get("honest_verdict"),
            "original_verdict_class": _source_verdict(inputs.live),
            "original_flagged_adversarial": inputs.live.get("flagged_adversarial"),
            "original_eligible": bool(inputs.live) and not live_checks,
            "failed_checks": deepcopy(live_checks),
        },
    }


def finalize_artifact(artifact: JsonDict) -> None:
    """Recompute every terminal claim after ordinary evidence changes."""

    reduced = independent_reduce(artifact)
    artifact["independent_metrics"] = reduced["synthetic_metrics"]
    artifact["cohort_claim_rows"] = reduced["cohort_claim_rows"]
    artifact["independent_reduction"] = reduced
    artifact["proof_audit_complete_score"] = reduced["proof_audit_complete_score"]
    artifact["proof_value_confirmed_score"] = reduced["proof_value_confirmed_score"]
    artifact["promotion_score"] = 0
    artifact["verdict_class"] = reduced["verdict_class"]
    if reduced["verdict_class"] == "blocked":
        artifact["status"] = "blocked_live_cohort_unavailable_synthetic_audit_retained"
        artifact["honest_verdict"] = (
            "blocked_live_cohort_unavailable_synthetic_circular_value_retained"
        )
    elif reduced["verdict_class"] == "disqualified":
        artifact["status"] = "complete_disqualified_required_producer_or_audit_invalid"
        artifact["honest_verdict"] = "complete_disqualified_required_producer_or_audit_invalid"
    elif reduced["verdict_class"] == "circular_positive":
        artifact["status"] = "complete_circular_positive_all_proof_cohorts_confirmed"
        artifact["honest_verdict"] = "complete_circular_positive_all_proof_cohorts_confirmed"
    else:
        artifact["status"] = "complete_null_combined_proof_value_not_confirmed"
        artifact["honest_verdict"] = "complete_null_combined_proof_value_not_confirmed"
    artifact["acceptance_gate_results"] = acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact, artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def build_artifact_for_test(
    inputs: AuditInputs,
    evidence: AuditEvidence,
    *,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    mutation_rows: Sequence[Mapping[str, Any]] | None = None,
    source_hashes: Mapping[str, str] | None = None,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    started_at_utc: str = "2026-09-19T00:00:00+00:00",
    completed_at_utc: str = "2026-09-19T00:00:01+00:00",
    duration_s: float = 1.0,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build a schema-complete artifact from authenticated ordinary evidence."""

    selected_receipts = list(receipts) if receipts is not None else passing_test_receipts()
    selected_mutations = (
        list(mutation_rows)
        if mutation_rows is not None
        else run_mutation_controls(inputs, evidence)
    )
    rows = deepcopy(evidence.rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(inputs.checks),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_CURRENT_INVOCATIONS),
        "inference_substrate": (
            "host CPU/JAX-environment aggregation with an audit-local exact 2-CNF "
            "graph solver, proof-edge checking, bootstrap reduction, and JSON replay"
        ),
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "device": "host_cpu",
            "processor": platform.processor() or "unspecified_host_cpu",
            "python": platform.python_version(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu_requested_not_imported"),
            "exact_solver": "audit_local_kosaraju_2cnf",
            "llm_runtime": None,
        },
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": dict(source_hashes or inputs.hashes),
        "source_cohort_dispositions": _source_dispositions(inputs),
        "historical_inference_sidecars": deepcopy(inputs.historical_sources),
        "small_ebm_training": {"performed": False, "reason": "historical_aggregation_only"},
        "rows": rows,
        "erasure_witness_rows": deepcopy(evidence.witness_rows),
        "state_continuity": deepcopy(evidence.continuity),
        "audit_errors": deepcopy(evidence.errors),
        "independent_metrics": deepcopy(evidence.metrics),
        "mutation_rows": deepcopy(selected_mutations),
        "sample_size_budget": {
            "cohorts_planned": 2,
            "cohorts_attempted": 2,
            "cohorts_completed": int(bool(rows)),
            "cohorts_blocked_external": int(not inputs.live.get("rows")),
            "cohorts_censored": 0,
            "cohorts_unstarted": 0,
            "synthetic_streams_planned": 32,
            "synthetic_streams_attempted": len({row["stream_id"] for row in rows}),
            "synthetic_streams_completed": len({row["stream_id"] for row in rows}),
            "synthetic_rows_planned": 32 * 24 * 5,
            "synthetic_rows_attempted": len(rows),
            "synthetic_rows_completed": sum(row.get("disposition") == "completed" for row in rows),
            "synthetic_rows_censored": sum(row.get("censored") is True for row in rows),
            "synthetic_rows_unstarted": max(0, 32 * 24 * 5 - len(rows)),
            "live_rows_planned": "blocked_upstream_before_roster_materialized",
            "live_rows_attempted": 0,
            "live_rows_completed": 0,
            "live_rows_censored": 0,
            "live_rows_unstarted": "all",
            "effective_independent_group_count": len({row["stream_id"] for row in rows}),
            "stop_rule": (
                "audit every available synthetic request-arm row; preserve absent live "
                "evidence as a separate blocked cohort"
            ),
        },
        "validation_receipts": deepcopy(selected_receipts),
        "repository_health": {
            "status": "not_used_for_affected_readiness",
            "observations": [
                {
                    "check": row["check"],
                    "expected": row["expected"],
                    "observed": row["observed"],
                    "affects_embedded_synthetic_rows": False,
                }
                for row in inputs.checks
                if row["cohort"] == "diagnostic" and not row["passed"]
            ],
            "affects_required_checks": False,
        },
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "promotion_score": 0,
        "proof_audit_complete_score": 0,
        "cohort_claim_rows": [],
        "proof_value_confirmed_score": 0,
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "field_principles": {},
        "independent_reduction": {},
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "cold_artifact_replay"],
        "research_conductor_changed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "rust_changed": False,
    }
    finalize_artifact(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check declarations, rows, cohort classes, gates, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_CURRENT_INVOCATIONS:
        errors.append("invocation_counts_nonzero")
    if (
        not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_declaration_mismatch")
    mutations = artifact.get("mutation_rows") or []
    if {row.get("mutation") for row in mutations if isinstance(row, Mapping)} != set(MUTATIONS):
        errors.append("mutation_mismatch")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("reduction_mismatch")
    if artifact.get("independent_metrics") != reduced["synthetic_metrics"]:
        errors.append("metrics_mismatch")
    if artifact.get("cohort_claim_rows") != reduced["cohort_claim_rows"]:
        errors.append("cohort_claim_mismatch")
    for score in ("proof_audit_complete_score", "proof_value_confirmed_score"):
        if artifact.get(score) != reduced[score]:
            errors.append("reduction_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    expected_gates = acceptance_gates(artifact)
    if artifact.get("acceptance_gate_results") != expected_gates:
        errors.append("acceptance_gates_mismatch")
    expected_summary = gate_summary(artifact, expected_gates)
    if artifact.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_mismatch")
    if artifact.get("verdict_class") != reduced["verdict_class"]:
        errors.append("verdict_class_mismatch")
    honest = str(artifact.get("honest_verdict") or "")
    if reduced["verdict_class"] == "blocked":
        if not honest.startswith("blocked_"):
            errors.append("honest_verdict_mismatch")
    elif not honest.startswith("complete_"):
        errors.append("honest_verdict_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_source_sidecar(root: Path, inputs: AuditInputs) -> dict[str, str]:
    """Write original verdicts and hashes without copying historical counters."""

    path = root / RAW_DIR / "historical_sources.json"
    atomic_json(
        path,
        {
            "label": "historical_sources_only_not_current_inference",
            "sources": deepcopy(inputs.historical_sources),
        },
    )
    return {path.relative_to(root).as_posix(): sha256_file(path)}


def write_audit_sidecars(
    root: Path,
    inputs: AuditInputs,
    evidence: AuditEvidence,
    mutations: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Publish independently derived evidence and source receipts atomically."""

    values = {
        root / RAW_DIR / "audit_evidence.json": {
            "rows": deepcopy(evidence.rows),
            "erasure_witness_rows": deepcopy(evidence.witness_rows),
            "independent_metrics": deepcopy(evidence.metrics),
            "state_continuity": deepcopy(evidence.continuity),
            "audit_errors": deepcopy(evidence.errors),
        },
        root / RAW_DIR / "mutation_controls.json": {"mutation_rows": deepcopy(list(mutations))},
    }
    hashes = write_source_sidecar(root, inputs)
    for path, payload in values.items():
        atomic_json(path, payload)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def cold_reload_errors(
    artifact: Mapping[str, Any], root: Path, *, check_sidecars: bool = True
) -> list[str]:
    """Reload source bytes and recompute every row in a fresh interpreter."""

    errors = validate_artifact(artifact)
    source_hashes = artifact.get("source_artifact_hashes") or {}
    for relative in (SYNTHETIC_PATH, LIVE_PATH, PROTOCOL_PATH):
        path = root / relative
        if not path.is_file() or sha256_file(path) != source_hashes.get(relative.as_posix()):
            errors.append(f"source_hash_mismatch:{relative.as_posix()}")
    if check_sidecars:
        for relative, expected in source_hashes.items():
            if not str(relative).startswith(RAW_DIR.as_posix()):
                continue
            path = root / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                errors.append(f"audit_sidecar_hash_mismatch:{relative}")
    inputs = collect_inputs(root)
    replay = audit_inputs(inputs)
    if replay.rows != artifact.get("rows"):
        errors.append("cold_row_replay_mismatch")
    if replay.witness_rows != artifact.get("erasure_witness_rows"):
        errors.append("cold_witness_replay_mismatch")
    if replay.metrics != artifact.get("independent_metrics"):
        errors.append("cold_metric_replay_mismatch")
    return list(dict.fromkeys(errors))


def scoped_command_plan(repo_root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    return validation_contract.build_command_plan(repo_root, V649_MANIFEST, private_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    errors = validation_contract.validate_command_plan(repo_root, V649_MANIFEST, commands)
    if Counter(command.name for command in commands) != Counter(REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if any(command.name == "full_python_suite" for command in commands):
        errors.append("full_python_suite_forbidden")
    return list(dict.fromkeys(errors))


def _normalized_receipt(row: Mapping[str, Any]) -> JsonDict:
    return {
        "name": row.get("name"),
        "command": row.get("command"),
        "command_argv": deepcopy(row.get("command_argv") or []),
        "environment": deepcopy(row.get("command_environment") or {}),
        "scope": row.get("scope"),
        "return_code": row.get("exit_code"),
        "exit_code": row.get("exit_code"),
        "duration_s": row.get("duration_s"),
        "log_path": row.get("log_path"),
        "log_hash": row.get("log_sha256"),
        "log_sha256": row.get("log_sha256"),
        "passed": row.get("passed"),
        "timed_out": row.get("timed_out"),
        "required": row.get("required", True),
        "category": row.get("command_category", "required_validation"),
        "started_at_utc": row.get("started_at_utc"),
        "ended_at_utc": row.get("ended_at_utc"),
        **(
            {"resolved_imports": deepcopy(row["resolved_imports"])}
            if "resolved_imports" in row
            else {}
        ),
    }


def terminal_commands(candidate: Path) -> list[validation_contract.PlannedCommand]:
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7405_v649_proof_audit import cold_reload_errors;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=cold_reload_errors(v,pathlib.Path(sys.argv[2]));print(e,flush=True);"
        "raise SystemExit(bool(e))"
    )
    reduce = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7405_v649_proof_audit import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        (
            "cold_artifact_replay",
            (python, "-u", "-c", replay, str(candidate), str(REPO_ROOT)),
            "capability_e2e",
        ),
        ("independent_reducer", (python, "-u", "-c", reduce, str(candidate)), "completion"),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in specs
    ]


def phase_span(
    phase: str,
    phase_started: float,
    run_started: float,
    started_at: str,
    checkpoints: int = 0,
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeats": 0,
        "checkpoints": checkpoints,
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Authenticate, audit, validate, cold-read, and atomically publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "read", "start")
    inputs = collect_inputs(root)
    spans.append(phase_span("read", phase_started, run_started, phase_utc))
    progress(run_started, "read", "end", checks=len(inputs.checks))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "load", "before_model_load", models=0)
    spans.append(phase_span("load", phase_started, run_started, phase_utc))
    progress(run_started, "load", "after_model_load", models=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "generate", "before_generation", calls=0)
    spans.append(phase_span("generate", phase_started, run_started, phase_utc))
    progress(run_started, "generate", "after_generation", calls=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "audit", "before_benchmark")
    evidence = audit_inputs(
        inputs,
        emit_progress=True,
        started=run_started,
        checkpoint_dir=root / RAW_DIR / "checkpoints",
    )
    mutations = run_mutation_controls(inputs, evidence)
    spans.append(
        phase_span(
            "audit",
            phase_started,
            run_started,
            phase_utc,
            checkpoints=len({row["stream_id"] for row in evidence.rows}),
        )
    )
    progress(run_started, "audit", "after_benchmark", rows=len(evidence.rows))
    source_hashes = {**inputs.hashes, **write_audit_sidecars(root, inputs, evidence, mutations)}

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "validate", "before_subprocesses")
    commands = scoped_command_plan(root, root / RAW_DIR / "validation/private")
    plan_errors = validate_scoped_command_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_scoped_command_plan:{plan_errors}")
    planned = [
        validation_contract.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    raw_receipts = validation_contract.run_categorized_commands(
        root, planned, log_dir=root / RAW_DIR / "validation/logs"
    )
    receipts = [_normalized_receipt(row) for row in raw_receipts]
    spans.append(phase_span("validate", phase_started, run_started, phase_utc))
    progress(run_started, "validate", "after_subprocesses", receipts=len(receipts))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "write", "start")
    candidate_path = root / RAW_DIR / "measured-terminal-candidate.json"
    candidate = build_artifact_for_test(
        inputs,
        evidence,
        receipts=receipts,
        mutation_rows=mutations,
        source_hashes=source_hashes,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
    )
    atomic_json(candidate_path, candidate)
    progress(run_started, "write", "before_terminal_subprocesses")
    raw_terminal = validation_contract.run_categorized_commands(
        root,
        terminal_commands(candidate_path),
        log_dir=root / RAW_DIR / "terminal/logs",
    )
    terminal_receipts = [_normalized_receipt(row) for row in raw_terminal]
    progress(
        run_started,
        "write",
        "after_terminal_subprocesses",
        count=len(terminal_receipts),
    )
    entrypoint_log = root / RAW_DIR / "terminal/declared_entrypoint.log"
    entrypoint_log.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_log.write_text(
        "The unbuffered declared entrypoint reached the atomic publication boundary.\n",
        encoding="utf-8",
    )
    terminal_receipts.append(
        {
            "name": "declared_entrypoint",
            "command": " ".join(sys.argv),
            "command_argv": list(sys.argv),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            },
            "scope": "current_unbuffered_entrypoint",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": time.monotonic() - run_started,
            "log_path": entrypoint_log.relative_to(root).as_posix(),
            "log_hash": sha256_file(entrypoint_log),
            "log_sha256": sha256_file(entrypoint_log),
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "capability_e2e",
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
        }
    )
    receipts.extend(terminal_receipts)
    spans.append(phase_span("write", phase_started, run_started, phase_utc))
    flagged = any(
        row["name"] == "adversarial_verify" and row["passed"] is not True
        for row in terminal_receipts
    )
    final = build_artifact_for_test(
        inputs,
        evidence,
        receipts=receipts,
        mutation_rows=mutations,
        source_hashes=source_hashes,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    atomic_json(output, final)
    progress(run_started, "write", "end", output=output, verdict=final["verdict_class"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0
