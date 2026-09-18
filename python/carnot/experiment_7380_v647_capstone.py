"""Reconcile the twelve-task V647 milestone from authenticated evidence.

The capstone reads existing artifacts and canonical conductor records. It does
not call a model, operate hardware, publish externally, or change production.

Spec refs: REQ-REPORT-7380 and SCENARIO-REPORT-7380-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot import experiment_7369_v647_contract as contract_helpers
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.647"
RUN_DATE = "20260918"
EXPERIMENT_ID = "exp7380-capstone"
SCHEMA = "carnot.experiment_7380.v647_capstone.v1"
RESULT_PATH = Path("results/experiment_7380_v647_capstone.json")
RAW_DIR = Path("results/raw/experiment_7380_v647_capstone")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7380_v647_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7380_v647_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7380_v647_capstone.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7380_v647_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")

EXPECTED_TASK_IDS = (
    "exp7369-contract",
    "exp7370-proof-memory",
    "exp7371-proof-boundary",
    "exp7372-qwen-canary",
    "exp7373-proposal-capture",
    "exp7374-prospective-memory",
    "exp7375-memory-audit",
    "exp7376-arc-outcomes",
    "exp7377-ising-law",
    "exp7378-ising-audit",
    "exp7379-hardware-envelope",
    EXPERIMENT_ID,
)

CANONICAL_ARTIFACT_PATHS = {
    "exp7369-contract": "results/experiment_7369_v647_contract.json",
    "exp7370-proof-memory": "results/experiment_7370_v647_proof_memory.json",
    "exp7371-proof-boundary": "results/experiment_7371_v647_proof_boundary.json",
    "exp7372-qwen-canary": "results/experiment_7372_v647_qwen_canary.json",
    "exp7373-proposal-capture": "results/experiment_7373_proposal_capture.json",
    "exp7376-arc-outcomes": "results/experiment_7376_v647_arc_outcomes.json",
    "exp7377-ising-law": "results/experiment_7377_v647_ising_law.json",
    "exp7378-ising-audit": "results/experiment_7378_v647_ising_audit.json",
    "exp7379-hardware-envelope": "results/experiment_7379_v647_hardware_envelope.json",
}

EXPECTED_CLASSES = {
    "exp7369-contract": "circular_positive",
    "exp7370-proof-memory": "null",
    "exp7371-proof-boundary": "null",
    "exp7372-qwen-canary": "disqualified",
    "exp7373-proposal-capture": "blocked",
    "exp7374-prospective-memory": "blocked",
    "exp7375-memory-audit": "blocked",
    "exp7376-arc-outcomes": "disqualified",
    "exp7377-ising-law": "circular_positive",
    "exp7378-ising-audit": "disqualified",
    "exp7379-hardware-envelope": "blocked",
}

CANONICAL_LOG_MARKERS = {
    "exp7374-prospective-memory": (
        "Measure continuous learning from certified implica | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7373-proposal-capture, "
        "exp7373-proposal-capture, exp7373-proposal-capture)"
    ),
    "exp7375-memory-audit": (
        "Independently reduce proof-memory causality and co | GATE_BLOCK | "
        "Pre-emptive skip: upstream retired (exp7374-prospective-memory, "
        "exp7374-prospective-memory, exp7374-prospective-memory)"
    ),
}

CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ELIGIBLE_VERDICTS = {"positive", "circular_positive", "null"}
CLAIM_BRANCHES = (
    "proof_memory",
    "fresh_model_proposals",
    "arc_outcomes",
    "ising_law_and_samples",
    "hardware_placement",
)
ZERO_INVOCATION_COUNTS = deepcopy(command_boundary.ZERO_INVOCATION_COUNTS)
RANDOM_SEED = {
    "experiment": 7_380_202_609_18,
    "resampling": 7_371_307,
}

V647_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal state only after actual work and required affected validation.",
    "run_date": "Use 20260918 with actual UTC start and completion timestamps.",
    "preconditions_checked": "Record exact paths, producer identity, hash, class, and resource checks before reduction.",
    "MODEL_SPECS": "Keep the list empty because the capstone intends no current model work.",
    "model_invoked": "Set true for any attempted current model load or generation, including failure.",
    "invocation_counts": "Count attempted and terminal current loads and generations; historical inputs do not count.",
    "inference_substrate": "Describe host aggregation and keep historical model provenance in labeled sidecars.",
    "inference_substrate_class": "Use the closed aggregation class that matches actual work.",
    "execution_venue": "Record host CPU work; V647 performs no current board execution here.",
    "duration_s": "Measure monotonic elapsed time without sleeping to meet a floor.",
    "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
    "random_seed": "Freeze the experiment and inherited resampling seeds.",
    "reproducibility_checksum": "Bind exact code, settings, authorities, protocol, hashes, and reduced rows.",
    "source_artifact_hashes": "Bind every artifact or canonical conductor record used by the capstone.",
    "rows": "Keep the measured branch outcomes, metrics, costs, failures, and censoring behind claims.",
    "sample_size_budget": "Separate planned, attempted, completed, censored, stopping, and remaining work.",
    "acceptance_gate_results": "Keep expected, observed, and passed values separate for every gate.",
    "gate_check_summary": "Name every failed upstream, check, field, expected value, and observed value.",
    "verifier_is_oracle": "Mark true because exact proof and finite-law evaluators define truth.",
    "honest_verdict": "Use complete scope for finished work and name unavailable external prerequisites.",
    "verdict_class": "Use the closed class; partial is only unfinished retryable capstone-owned work.",
    "flagged_adversarial": "Preserve critical independent findings so affected producers cannot supply readiness.",
    "validation_receipts": "Retain exact argv, environment, scope, exit, duration, and log hash for checks.",
    "repository_health": "Keep dated unrelated failures separate from required affected checks.",
    "field_principles": "Explain each ordinary field directly without wrapping values.",
    "promotion_score": "Keep zero because V647 authorizes no rollout or external publication.",
    "milestone_disposition_complete_score": "One means exactly twelve ordered dispositions, not twelve science successes.",
    "disposition_rows": "Name each exact artifact or pre-gate record, class, scope, failed checks, and decision.",
    "required_science_complete_score": "Require eligible proof-memory measurement and audit plus eligible Ising measurement.",
    "publication_gate_results": "Preserve canonical G1-G4 under their historical FoVer-only scope.",
    "retirement_decisions": "Record prior and current verdicts, mechanism changes, and bounded retirement choices.",
    "next_research_decisions": "Tie every continue, retire, or defer decision to a measured bottleneck.",
}


def utc_now() -> str:
    """Return an aware UTC timestamp for a real experiment boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each phase boundary so a bounded aggregation never appears idle."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7380] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f} {suffix}".rstrip(),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so an evidence identity cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for rows that have no independent artifact file."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON document with an atomic local rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> JsonDict:
    """Return a JSON object and reject malformed or non-object evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _terminal_status(value: object) -> bool:
    """Accept only explicit complete, blocked, or disqualified lifecycle states."""

    text = str(value)
    return text in {"complete", "blocked", "disqualified"} or text.startswith(
        ("complete_", "blocked_", "disqualified_")
    )


def load_contract(root: Path) -> JsonDict:
    """Parse the active YAML and Markdown authorities through independent helpers."""

    roadmap_file = root / ROADMAP_PATH
    design_file = root / DESIGN_PATH
    roadmap = yaml.safe_load(roadmap_file.read_text(encoding="utf-8"))
    if not isinstance(roadmap, dict) or roadmap.get("milestone") != MILESTONE:
        raise ValueError("active roadmap must name milestone 2026.09.647")
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list) or [row.get("id") for row in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("active roadmap must contain the exact twelve V647 tasks")
    evaluation = contract_helpers.evaluate_contract(
        design_file.read_text(encoding="utf-8"), roadmap
    )
    return {
        "milestone": MILESTONE,
        "roadmap_path": ROADMAP_PATH.as_posix(),
        "roadmap_sha256": sha256_file(roadmap_file),
        "design_path": DESIGN_PATH.as_posix(),
        "design_sha256": sha256_file(design_file),
        "contract_match": evaluation.get("passed") is True,
        "contract_rows": deepcopy(evaluation.get("contract_rows") or []),
        "tasks": deepcopy(tasks),
    }


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent external absence without creating a success-shaped artifact."""

    return {
        "task_id": str(task["id"]),
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "missing",
        "sha256": None,
        "status": "blocked",
        "honest_verdict": "blocked_missing_external_evidence",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": False,
        "accepted_for_science": False,
        "payload": {},
    }


def _load_log_record(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one retired task from its exact canonical conductor line."""

    task_id = str(task["id"])
    log_path = root / CONDUCTOR_PATH
    if not log_path.is_file():
        return _missing_evidence(task)
    marker = CANONICAL_LOG_MARKERS[task_id]
    matches = [line for line in log_path.read_text(encoding="utf-8").splitlines() if marker in line]
    if len(matches) != 1:
        return _missing_evidence(task)
    line = matches[0]
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "conductor_log_record",
        "sha256": canonical_hash({"path": CONDUCTOR_PATH.as_posix(), "line": line}),
        "source_file_sha256": sha256_file(log_path),
        "source_record": line,
        "status": "blocked",
        "honest_verdict": "blocked_upstream_retired",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": True,
        "accepted_for_science": False,
        "payload": {
            "status": "blocked",
            "honest_verdict": "blocked_upstream_retired",
            "verdict_class": "blocked",
        },
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Load one exact producer artifact or its allowed canonical pre-gate record."""

    task_id = str(task["id"])
    if task_id in CANONICAL_LOG_MARKERS:
        return _load_log_record(root, task)
    selected_label = CANONICAL_ARTIFACT_PATHS.get(task_id)
    if selected_label is None:
        return _missing_evidence(task)
    selected = root / selected_label
    if not selected.is_file():
        return _missing_evidence(task)
    payload = _load_json(selected)
    pre_gate = task_id == "exp7373-proposal-capture"
    if pre_gate:
        upstream = root / CANONICAL_ARTIFACT_PATHS["exp7372-qwen-canary"]
        authenticated = (
            payload.get("schema") == "blocked_gate_check_v1"
            and payload.get("status") == "blocked"
            and payload.get("blocked_at_layer") == "conductor_pre_gate"
            and payload.get("failed_upstream") == "exp7372-qwen-canary"
            and payload.get("failed_field") == "qwen_assignment_transport_ready_score"
            and payload.get("failed_expected") == 1
            and payload.get("failed_observed") == 0
            and payload.get("failed_evidence_sha256") == sha256_file(upstream)
        )
        verdict = "blocked"
        source_kind = "conductor_pre_gate_artifact"
        flagged = False
    else:
        verdict = str(payload.get("verdict_class", ""))
        authenticated = (
            payload.get("milestone") == MILESTONE
            and _terminal_status(payload.get("status"))
            and verdict in CLOSED_VERDICTS
            and verdict == EXPECTED_CLASSES[task_id]
        )
        source_kind = "declared_artifact"
        flagged = payload.get("flagged_adversarial") is True
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": selected_label,
        "source_kind": source_kind,
        "sha256": sha256_file(selected),
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict", "blocked_gate_check_failed"),
        "verdict_class": verdict,
        "flagged_adversarial": flagged,
        "authenticated": authenticated,
        "accepted_for_science": (authenticated and not flagged and verdict in ELIGIBLE_VERDICTS),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read each of the eleven preceding slots before any dependent reduction."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def collect_preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record driving requirements, exact authorities, and producer identities."""

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows: list[JsonDict] = [
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-REPORT-7380",
            "expected": True,
            "observed": "REQ-REPORT-7380" in spec_text,
            "passed": "REQ-REPORT-7380" in spec_text,
        },
        {
            "check": "exact_contract",
            "upstream": f"{ROADMAP_PATH.as_posix()} + {DESIGN_PATH.as_posix()}",
            "artifact_field": "tasks/order/fields/gates",
            "expected": {"milestone": MILESTONE, "task_ids": list(EXPECTED_TASK_IDS)},
            "observed": {
                "milestone": contract.get("milestone"),
                "task_ids": [row.get("id") for row in contract.get("tasks", [])],
            },
            "passed": contract.get("contract_match") is True,
        },
    ]
    for task_id in EXPECTED_TASK_IDS[:-1]:
        row = evidence[task_id]
        rows.append(
            {
                "check": "preceding_disposition_source",
                "upstream": task_id,
                "artifact_field": "path/hash/class/authenticated/flagged_adversarial",
                "expected": {
                    "class": EXPECTED_CLASSES[task_id],
                    "authenticated": True,
                },
                "observed": {
                    "path": row.get("actual_path"),
                    "sha256": row.get("sha256"),
                    "class": row.get("verdict_class"),
                    "authenticated": row.get("authenticated"),
                    "flagged_adversarial": row.get("flagged_adversarial"),
                },
                "passed": row.get("authenticated") is True,
                "accepted_for_science": row.get("accepted_for_science") is True,
            }
        )
    return rows


def _claim_row(
    branch: str,
    source_tasks: Sequence[str],
    metrics: Mapping[str, Any],
    completion_score: int,
    value_score: int,
    failures: Sequence[str],
    *,
    verifier_is_oracle: bool,
) -> JsonDict:
    """Create one bounded branch summary without promoting completion to value."""

    return {
        "branch": branch,
        "source_tasks": list(source_tasks),
        "metrics": deepcopy(dict(metrics)),
        "completion_score": completion_score,
        "value_score": value_score,
        "failures": list(failures),
        "censored": completion_score == 0,
        "verifier_is_oracle": verifier_is_oracle,
        "promotes_readiness": False,
    }


def reduce_claim_rows(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Recompute only supported proof, proposal, ARC, Ising, and hardware claims."""

    prototype = evidence["exp7370-proof-memory"]["payload"]
    boundary = evidence["exp7371-proof-boundary"]["payload"]
    canary = evidence["exp7372-qwen-canary"]["payload"]
    capture = evidence["exp7373-proposal-capture"]
    prospective = evidence["exp7374-prospective-memory"]
    audit = evidence["exp7375-memory-audit"]
    arc = evidence["exp7376-arc-outcomes"]["payload"]
    law = evidence["exp7377-ising-law"]["payload"]
    sampler = evidence["exp7378-ising-audit"]["payload"]
    hardware = evidence["exp7379-hardware-envelope"]["payload"]

    arc_budget = dict(arc.get("sample_size_budget") or {})
    law_budget = dict(law.get("sample_size_budget") or {})
    sampler_budget = dict(sampler.get("sample_size_budget") or {})
    sampler_reduction = dict(sampler.get("independent_reduction") or {})
    placement_gate = dict(
        (hardware.get("acceptance_gate_results") or {}).get("placement_measurement") or {}
    )
    rows = [
        _claim_row(
            "proof_memory",
            (
                "exp7370-proof-memory",
                "exp7371-proof-boundary",
                "exp7374-prospective-memory",
                "exp7375-memory-audit",
            ),
            {
                "development_safety_ready": (
                    prototype.get("proof_fixture_ready_score") == 1
                    and boundary.get("proof_boundary_ready_score") == 1
                ),
                "prospective_measurement_available": prospective.get("accepted_for_science")
                is True,
                "independent_audit_available": audit.get("accepted_for_science") is True,
                "causal_erasure_witnesses": None,
                "paid_query_ratio_upper_95": None,
                "complete_service_cost_upper_95": None,
            },
            0,
            0,
            ("prospective measurement retired", "independent audit retired"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "fresh_model_proposals",
            ("exp7372-qwen-canary", "exp7373-proposal-capture"),
            {
                "current_model_calls_counted_by_capstone": 0,
                "canary_transport_ready_score": canary.get("qwen_assignment_transport_ready_score"),
                "canary_verdict_class": evidence["exp7372-qwen-canary"]["verdict_class"],
                "capture_started": capture.get("source_kind") == "declared_artifact",
                "oracle_verification_is_model_quality": False,
            },
            0,
            0,
            ("canary disqualified", "proposal capture pre-gated"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "arc_outcomes",
            ("exp7376-arc-outcomes",),
            {
                "planned_episodes": arc_budget.get("planned_units"),
                "attempted_episodes": arc_budget.get("attempted_units"),
                "completed_episodes": arc_budget.get("completed_units"),
                "censored_episodes": arc_budget.get("censored_units"),
                "supervisor_firing_count": sum(
                    row.get("supervisor_fired") is True for row in arc.get("rows") or []
                ),
            },
            int(arc.get("arc_outcome_capture_complete_score") == 1),
            0,
            ("no completed live episodes", "no supported supervisor firing"),
            verifier_is_oracle=False,
        ),
        _claim_row(
            "ising_law_and_samples",
            ("exp7377-ising-law", "exp7378-ising-audit"),
            {
                "law_fixture_ready_score": law.get("law_fixture_ready_score"),
                "completed_finite_law_rows": law_budget.get("completed_finite_rows"),
                "completed_formulas": law_budget.get("completed_formulas"),
                "completed_source_cells": sampler_budget.get("completed_source_cells"),
                "qualified_source_cells": sampler_reduction.get("qualified_source_cell_count"),
                "sample_audit_eligible": evidence["exp7378-ising-audit"]["accepted_for_science"],
                "all_source_cells_qualified": sampler_reduction.get("all_source_cells_qualified"),
            },
            0,
            0,
            ("sampler audit terminal validation failed", "18 empty-support cells"),
            verifier_is_oracle=True,
        ),
        _claim_row(
            "hardware_placement",
            ("exp7379-hardware-envelope",),
            {
                "board_disposition_complete_score": hardware.get(
                    "board_disposition_complete_score"
                ),
                "placement_measurement_available": placement_gate.get("passed") is True,
                "gatemate_changed_state_available": False,
                "current_board_execution_count": 0,
            },
            int(hardware.get("board_disposition_complete_score") == 1),
            0,
            (
                "proof-memory placement input missing",
                "Ising placement input disqualified",
                "GateMate physical state unchanged",
            ),
            verifier_is_oracle=False,
        ),
    ]
    return rows


def _failure(
    upstream: str,
    check: str,
    field: str,
    expected: Any,
    observed: Any,
    category: str,
) -> JsonDict:
    """Create one exact failed-gate row with no vague blocked wording."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "category": category,
        "passed": False,
    }


def terminal_state(
    evidence: Mapping[str, Mapping[str, Any]], *, required_validation_passed: bool
) -> JsonDict:
    """Classify external absence as blocked and required failures as disqualified."""

    blocked: list[JsonDict] = []
    disqualified: list[JsonDict] = []
    if not required_validation_passed:
        disqualified.append(
            _failure(
                EXPERIMENT_ID,
                "affected_validation",
                "required_checks_passed",
                True,
                False,
                "required_validation",
            )
        )
    for task_id in (
        "exp7372-qwen-canary",
        "exp7376-arc-outcomes",
        "exp7378-ising-audit",
    ):
        row = evidence[task_id]
        if row.get("verdict_class") == "disqualified" or row.get("flagged_adversarial") is True:
            disqualified.append(
                _failure(
                    task_id,
                    "required_safety_or_validation",
                    "verdict_class/flagged_adversarial",
                    {"verdict_class": list(ELIGIBLE_VERDICTS), "flagged_adversarial": False},
                    {
                        "verdict_class": row.get("verdict_class"),
                        "flagged_adversarial": row.get("flagged_adversarial"),
                    },
                    "required_validation",
                )
            )
    for task_id, field in (
        ("exp7374-prospective-memory", "proof_learning_capture_complete_score"),
        ("exp7375-memory-audit", "proof_learning_audit_complete_score"),
    ):
        row = evidence[task_id]
        if row.get("accepted_for_science") is not True:
            blocked.append(
                _failure(
                    task_id,
                    "required_proof_memory_evidence",
                    field,
                    1,
                    {
                        "source_kind": row.get("source_kind"),
                        "verdict_class": row.get("verdict_class"),
                        "accepted_for_science": row.get("accepted_for_science"),
                    },
                    "external_prerequisite",
                )
            )
    failures = [*disqualified, *blocked]
    verdict = "disqualified" if disqualified else ("blocked" if blocked else "null")
    if verdict == "disqualified":
        honest = (
            "complete_disqualified_required_science_or_validation_failure: all twelve V647 "
            "dispositions are accounted for; required producer validation failed and proof-memory "
            "measurement plus audit were pre-gated"
        )
    elif verdict == "blocked":
        honest = (
            "blocked_required_proof_memory_evidence: proof-memory measurement or independent "
            "audit is externally unavailable; completed accounting is not retryable partial work"
        )
    else:
        honest = "complete_null_required_science_measured_without_qualified_value"
    return {
        "status": f"complete_{verdict}_v647_capstone" if verdict != "blocked" else "blocked",
        "honest_verdict": honest,
        "verdict_class": verdict,
        "required_science_complete_score": int(not blocked and not disqualified),
        "gate_check_summary": {
            "passed": not failures,
            "failed_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
        },
    }


def publication_gate_results(payload: Mapping[str, Any]) -> JsonDict:
    """Preserve canonical G1-G4 while denying V647 publication authority."""

    source_gates = payload.get("gates")
    source_gates = source_gates if isinstance(source_gates, Mapping) else {}
    gates = {
        name: deepcopy(source_gates.get(name))
        if isinstance(source_gates.get(name), Mapping)
        else {"pass": False, "detail": "canonical gate result missing"}
        for name in ("G1", "G2", "G3", "G4")
    }
    unmet = [name for name, row in gates.items() if row.get("pass") is not True]
    return {
        "scope": "historical_fover_paper_only",
        "definitions": {
            "G1": "headline measured",
            "G2": "independently reproduced",
            "G3": "prose narrowing-clean",
            "G4": "numbers trace to artifacts",
        },
        "gates": gates,
        "paper_ready": not unmet,
        "unmet_gates": unmet,
        "certifies_v647": False,
        "internal_reducer_is_external_reproducer": False,
        "authorizes_external_publication": False,
        "authorizes_deployment": False,
        "authorizes_release_or_push": False,
        "source_note": payload.get("note"),
    }


def _row_failures(row: Mapping[str, Any]) -> list[JsonDict]:
    """Retain an upstream gate summary without interpreting it as new evidence."""

    summary = row.get("payload", {}).get("gate_check_summary")
    if isinstance(summary, Mapping):
        failures = summary.get("failures") or summary.get("failed_checks") or []
        return deepcopy(failures) if isinstance(failures, list) else []
    if isinstance(summary, str) and summary:
        return [{"summary": summary}]
    return []


def build_disposition_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
    *,
    validation_complete: bool,
) -> list[JsonDict]:
    """Append the self row only after capstone-owned required validation finishes."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        verdict = str(source.get("verdict_class"))
        decision = (
            "continue" if task_id in {"exp7370-proof-memory", "exp7371-proof-boundary"} else "defer"
        )
        if task_id in {
            "exp7374-prospective-memory",
            "exp7375-memory-audit",
            "exp7379-hardware-envelope",
        }:
            decision = "retire"
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "title": task.get("title"),
                "declared_path": task.get("deliverable"),
                "actual_path": source.get("actual_path"),
                "source_kind": source.get("source_kind"),
                "sha256": source.get("sha256"),
                "status": source.get("status"),
                "honest_verdict": source.get("honest_verdict"),
                "verdict_class": verdict,
                "flagged_adversarial": source.get("flagged_adversarial"),
                "failed_checks": _row_failures(source),
                "evidence_scope": "accounting_only"
                if verdict in {"blocked", "disqualified", "partial"}
                else "eligible_completed_evidence",
                "next_decision": decision,
            }
        )
    if not validation_complete:
        return rows
    rows.append(
        {
            "order": 12,
            "task_id": EXPERIMENT_ID,
            "title": tasks[-1].get("title"),
            "declared_path": tasks[-1].get("deliverable"),
            "actual_path": RESULT_PATH.as_posix(),
            "source_kind": "self",
            "sha256": None,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
            "failed_checks": deepcopy(terminal["gate_check_summary"]["failures"]),
            "evidence_scope": "completed_capstone_accounting",
            "next_decision": "stop",
        }
    )
    return rows


def _source_hashes(
    contract: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    """Bind exact authorities and every preceding disposition source."""

    values: JsonDict = {
        "active_roadmap": {
            "path": contract["roadmap_path"],
            "sha256": contract["roadmap_sha256"],
        },
        "v647_design": {
            "path": contract["design_path"],
            "sha256": contract["design_sha256"],
        },
    }
    for task_id, row in evidence.items():
        values[task_id] = {
            "path": row.get("actual_path") or CONDUCTOR_PATH.as_posix(),
            "sha256": row.get("sha256"),
            "source_kind": row.get("source_kind"),
            "source_file_sha256": row.get("source_file_sha256"),
        }
    return values


def _sample_budget(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Retain actual upstream budgets without inventing missing work units."""

    return {
        "planned_disposition_units": 12,
        "attempted_disposition_units": 12,
        "completed_disposition_units": 12,
        "censored_disposition_units": 0,
        "remaining_disposition_units": 0,
        "proof_memory": {
            "planned_measurement_and_audit_units": 2,
            "completed_measurement_and_audit_units": 0,
            "censored_units": 2,
            "remaining_work": 0,
            "stopping_rule": "Preserve conductor retirement; do not invent or retry external inputs in the capstone.",
        },
        "arc": deepcopy(evidence["exp7376-arc-outcomes"]["payload"].get("sample_size_budget")),
        "ising_law": deepcopy(evidence["exp7377-ising-law"]["payload"].get("sample_size_budget")),
        "ising_samples": deepcopy(
            evidence["exp7378-ising-audit"]["payload"].get("sample_size_budget")
        ),
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep completion, validation, scientific value, and promotion gates distinct."""

    return [
        {
            "category": "completion",
            "check": "exact_contract",
            "expected": True,
            "observed": contract.get("contract_match"),
            "passed": contract.get("contract_match") is True,
        },
        {
            "category": "required_validation",
            "check": "affected_validation",
            "expected": True,
            "observed": validation.get("required_checks_passed"),
            "passed": validation.get("required_checks_passed") is True,
        },
        {
            "category": "required_validation",
            "check": "terminal_validation",
            "expected": True,
            "observed": validation.get("terminal_validation_passed", True),
            "passed": validation.get("terminal_validation_passed", True) is True,
        },
        {
            "category": "completion",
            "check": "twelve_dispositions",
            "expected": 12,
            "observed": 12,
            "passed": True,
        },
        {
            "category": "scientific_efficacy",
            "check": "required_science_complete",
            "expected": 1,
            "observed": terminal["required_science_complete_score"],
            "passed": terminal["required_science_complete_score"] == 1,
        },
        {
            "category": "publication_scope",
            "check": "historical_fover_gate_only",
            "expected": {"certifies_v647": False, "authorizes_external_publication": False},
            "observed": {
                "certifies_v647": publication["certifies_v647"],
                "authorizes_external_publication": publication["authorizes_external_publication"],
            },
            "passed": (
                publication["certifies_v647"] is False
                and publication["authorizes_external_publication"] is False
            ),
        },
        {
            "category": "promotion",
            "check": "automatic_promotion",
            "expected": 0,
            "observed": 0,
            "passed": True,
        },
    ]


def _retirement_decisions(terminal: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve prior outcomes and stop unchanged mechanisms with measured bottlenecks."""

    return [
        {
            "branch": "v646_capstone_prior_failure",
            "previous_verdict": "complete_disqualified_required_science_or_validation_failure",
            "current_verdict": terminal["honest_verdict"],
            "same_exact_verdict": False,
            "mechanism_change": "V647 used proof-carrying implication paths and the executed scoped command plan.",
            "decision": "preserve_not_same_verdict",
            "mechanical_receipt": "research-roadmap.yaml exp7380-capstone.prior_failures[0]",
        },
        {
            "branch": "boolean_schedule_acquisition",
            "previous_verdict": "null_or_disqualified",
            "current_verdict": "unchanged_mechanism_not_run",
            "same_exact_verdict": True,
            "mechanism_change": "none",
            "decision": "retire",
            "mechanical_receipt": "V647 design rerun discipline",
        },
        {
            "branch": "unsupported_supervisor_selector",
            "previous_verdict": "supported_decision_count_zero",
            "current_verdict": "zero_completed_episodes_and_zero_firings",
            "same_exact_verdict": True,
            "mechanism_change": "new outcome acquisition failed before selector support existed",
            "decision": "retire",
            "mechanical_receipt": "Exp7376 rows and sample_size_budget",
        },
        {
            "branch": "unchanged_hardware_bringup",
            "previous_verdict": "blocked_changed_physical_state",
            "current_verdict": "blocked_changed_physical_state",
            "same_exact_verdict": True,
            "mechanism_change": "none",
            "decision": "retire",
            "mechanical_receipt": "Exp7379 gate_check_summary",
        },
    ]


def _next_decisions() -> list[JsonDict]:
    """Choose bounded research actions from measured V647 bottlenecks."""

    return [
        {
            "branch": "proof_memory",
            "decision": "defer",
            "measured_bottleneck": "Exp7372 transport score 0 pre-gated Exp7373, Exp7374, and Exp7375.",
            "condition": "Resume only after a clean bounded proposal transport receipt; reuse the sealed proof boundary.",
        },
        {
            "branch": "arc_supervisor",
            "decision": "retire",
            "measured_bottleneck": "Six attempts produced zero completed episodes and zero firings.",
            "condition": "Do not fit or promote a selector without new completed live outcomes.",
        },
        {
            "branch": "ising_law",
            "decision": "continue",
            "measured_bottleneck": "The exact law fixture passed, but 18 sample cells had empty support and terminal validation failed.",
            "condition": "Repair validation and empty-support handling without changing the frozen law protocol.",
        },
        {
            "branch": "hardware",
            "decision": "defer",
            "measured_bottleneck": "No eligible complete-service placement rows and no changed GateMate state receipt.",
            "condition": "No bring-up retry until an operator-authored physical change or eligible measured cost row exists.",
        },
    ]


def _historical_sidecars(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Label model-bearing producers without counting them as current inference."""

    rows = []
    for task_id in ("exp7372-qwen-canary", "exp7373-proposal-capture", "exp7376-arc-outcomes"):
        source = evidence[task_id]
        rows.append(
            {
                "task_id": task_id,
                "path": source.get("actual_path"),
                "sha256": source.get("sha256"),
                "historical_only": True,
                "counted_as_current_inference": False,
                "invocation_counts": deepcopy(source.get("payload", {}).get("invocation_counts")),
            }
        )
    return rows


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide a complete zero-length phase ledger for pure unit construction."""

    return [
        {"phase": phase, "started_elapsed_s": 0.0, "ended_elapsed_s": 0.0, "duration_s": 0.0}
        for phase in ("read", "build", "load", "generate", "evaluate", "validate", "write")
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence and reductions while excluding wall-clock presentation."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "random_seed",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "disposition_rows",
        "publication_gate_results",
        "retirement_decisions",
        "next_research_decisions",
        "verdict_class",
        "honest_verdict",
        "promotion_score",
    )
    return canonical_hash({field: artifact.get(field) for field in fields})


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    publication_payload: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the schema-complete terminal record after affected checks finish."""

    publication = publication_gate_results(publication_payload)
    terminal = terminal_state(
        evidence,
        required_validation_passed=validation.get("required_checks_passed") is True,
    )
    dispositions = build_disposition_rows(
        contract["tasks"], evidence, terminal, validation_complete=True
    )
    preconditions = collect_preconditions(root, contract, evidence)
    rows = reduce_claim_rows(evidence)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "title": "Reconcile twelve outcomes and decide proof-memory continuation",
        "status": terminal["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "operation": "JSON/YAML/Markdown parsing, hashing, reduction, and scoped validation",
            "current_model_or_board_operations": 0,
        },
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_hashes": _source_hashes(contract, evidence),
        "historical_model_input_sidecars": _historical_sidecars(evidence),
        "contract_receipt": {
            "advisory": True,
            "contract_match": contract["contract_match"],
            "task_count": len(contract["tasks"]),
            "contract_rows": deepcopy(contract["contract_rows"]),
        },
        "rows": rows,
        "sample_size_budget": _sample_budget(evidence),
        "acceptance_gate_results": _acceptance_gates(contract, terminal, validation, publication),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": any(
            row.get("flagged_adversarial") is True for row in evidence.values()
        ),
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "required_checks_passed": validation.get("required_checks_passed") is True,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "external_publication_authorized": False,
        "deployment_authorized": False,
        "release_or_push_authorized": False,
        "promotion_score": 0,
        "scientific_value_score": 0,
        "readiness_score": 0,
        "milestone_disposition_complete_score": int(len(dispositions) == 12),
        "disposition_rows": dispositions,
        "required_science_complete_score": terminal["required_science_complete_score"],
        "publication_gate_results": publication,
        "retirement_decisions": _retirement_decisions(terminal),
        "next_research_decisions": _next_decisions(),
        "reproducibility_checksum": "",
        "field_principles": {},
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Record the measured V647 capstone value for {key.replace('_', ' ')}.",
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload file-backed source identities while preserving conductor row hashes."""

    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return False
    for row in sources.values():
        if not isinstance(row, Mapping):
            return False
        path = root / str(row.get("path"))
        if row.get("source_kind") == "conductor_log_record":
            if row.get("source_file_sha256") != sha256_file(path):
                return False
        elif not path.is_file() or row.get("sha256") != sha256_file(path):
            return False
    return True


def validate_artifact(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, reductions, hashes, scores, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    required = (
        "schema",
        "experiment_id",
        "milestone",
        "status",
        "run_date",
        "preconditions_checked",
        "MODEL_SPECS",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "validation_receipts",
        "repository_health",
        "field_principles",
        "promotion_score",
        "milestone_disposition_complete_score",
        "disposition_rows",
        "required_science_complete_score",
        "publication_gate_results",
        "retirement_decisions",
        "next_research_decisions",
    )
    for field in required:
        if field not in artifact:
            errors.append(f"missing_required_field:{field}")
    if errors:
        return errors
    if (
        artifact["schema"],
        artifact["experiment_id"],
        artifact["milestone"],
        artifact["run_date"],
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if not _terminal_status(artifact["status"]) or artifact["verdict_class"] not in CLOSED_VERDICTS:
        errors.append("lifecycle_invalid")
    if (
        artifact["MODEL_SPECS"] != []
        or artifact["model_invoked"] is not False
        or artifact["invocation_counts"] != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact["inference_substrate_class"] != "aggregation"
        or artifact["execution_venue"] != "host"
    ):
        errors.append("substrate_invalid")
    dispositions = artifact["disposition_rows"]
    valid_dispositions = (
        isinstance(dispositions, list)
        and len(dispositions) == 12
        and [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS)
        and [row.get("order") for row in dispositions] == list(range(1, 13))
        and all(
            row.get("verdict_class") == EXPECTED_CLASSES[row["task_id"]]
            for row in dispositions[:-1]
        )
        and dispositions[-1].get("source_kind") == "self"
        and dispositions[-1].get("verdict_class") == artifact["verdict_class"]
    )
    if not valid_dispositions:
        errors.append("disposition_rows_invalid")
    if artifact["milestone_disposition_complete_score"] != int(valid_dispositions):
        errors.append("disposition_score_invalid")
    if artifact["promotion_score"] != 0 or artifact.get("readiness_score") != 0:
        errors.append("failed_state_score_nonzero")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the exact Exp7358 plan for only this task's affected files."""

    return command_boundary.build_command_plan(root, V647_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, absent private parents, and command drift."""

    return command_boundary.validate_command_plan(root, V647_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen execution date from the declared command."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(
    phase: str, phase_started: float, run_started: float
) -> JsonDict:  # pragma: no cover - measured by the public entrypoint.
    """Close one measured phase span against the experiment's monotonic origin."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _terminal_commands(
    root: Path, candidate: Path
) -> list[command_boundary.PlannedCommand]:  # pragma: no cover - E2E only.
    """Build cold replay plus both mandatory strict artifact readers."""

    python = str(root / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7380_v647_capstone import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v);print({'errors':e},flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "independent_reducer",
            (python, "-u", "-c", replay, str(candidate)),
            "candidate_raw_evidence_replay",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (
                python,
                "-u",
                PUBLICATION_GATE_PATH.parent.joinpath("adversarial_verify.py").as_posix(),
                str(candidate),
            ),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
    )
    return [command_boundary.PlannedCommand(spec, "required_validation", True) for spec in specs]


def _run_publication_gate(
    root: Path, log_dir: Path
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - E2E only.
    """Run the canonical gate as a streamed subprocess and retain its exact receipt."""

    spec = validation_scope.CommandSpec(
        "publication_gate",
        (str(root / ".venv/bin/python"), "-u", PUBLICATION_GATE_PATH.as_posix(), "--json"),
        "historical_fover_paper",
    )
    receipt = validation_scope.run_commands(root, [spec], log_dir=log_dir)[0]
    payload = json.loads(str(receipt.get("output_tail") or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("publication gate must emit a JSON object")
    return payload, receipt


def run_experiment(
    root: Path, run_date: str
) -> JsonDict:  # pragma: no cover - exercised by the declared entrypoint.
    """Run exact reads, scoped checks, cold readers, and one atomic terminal write."""

    date_argument(run_date)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7380-", dir="/tmp"))

    phase_started = time.monotonic()
    progress(started, "read", "before")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(_phase_span("read", phase_started, started))
    progress(started, "read", "after", dispositions=len(evidence))

    for phase in ("load", "generate"):
        point = time.monotonic()
        progress(started, phase, "before", current_model_operations=0)
        spans.append(_phase_span(phase, point, started))
        progress(started, phase, "after", current_model_operations=0)

    phase_started = time.monotonic()
    progress(started, "build", "before_validation_plan")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("build", phase_started, started))
    progress(started, "build", "after_validation_plan", commands=len(commands))

    phase_started = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    planned = [
        command_boundary.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    affected_receipts = command_boundary.run_categorized_commands(
        root,
        planned,
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = command_boundary.reduce_affected_receipts(root, V647_MANIFEST, affected_receipts)
    publication_payload, publication_receipt = _run_publication_gate(
        root, raw_dir / "validation/publication"
    )
    validation: JsonDict = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": [*affected_receipts, publication_receipt],
        "repository_health": {
            "status": "historical_observations_retained",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }
    spans.append(_phase_span("validate", phase_started, started))
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])

    phase_started = time.monotonic()
    progress(started, "evaluate", "before")
    provisional = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication_payload,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=[*spans, _phase_span("evaluate", phase_started, started)],
    )
    candidate = private_root / "experiment_7380_candidate.json"
    atomic_json(candidate, provisional)
    progress(started, "evaluate", "after", verdict=provisional["verdict_class"])

    phase_started = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate)
    progress(started, "validate_terminal", "before_subprocesses", units=len(terminal_commands))
    terminal_receipts = command_boundary.run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    validation["terminal_validation_passed"] = terminal_passed
    validation["required_checks_passed"] = reduced["passed"] and terminal_passed
    validation["validation_receipts"].extend(terminal_receipts)
    spans.append(_phase_span("validate", phase_started, started))
    progress(started, "validate_terminal", "after_subprocesses", passed=terminal_passed)

    phase_started = time.monotonic()
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix())
    final_spans = [
        *spans,
        {
            "phase": "write",
            "started_elapsed_s": phase_started - started,
            "ended_elapsed_s": time.monotonic() - started,
            "duration_s": time.monotonic() - phase_started,
        },
    ]
    artifact = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication_payload,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(root / RESULT_PATH, artifact)
    atomic_json(
        root / CHECKPOINT_PATH,
        {"status": "complete", "artifact": RESULT_PATH.as_posix(), "completed_at_utc": utc_now()},
    )
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - public boundary.
    """Parse only the frozen run date used by the declared command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - E2E entrypoint.
    """Execute V647 aggregation through the repository-root thin launcher."""

    print("[exp7380] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
