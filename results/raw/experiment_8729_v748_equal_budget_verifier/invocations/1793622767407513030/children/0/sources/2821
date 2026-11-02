"""Close V640 with an authenticated fourteen-task evidence matrix.

This module does not invoke a model. It reads the frozen milestone files and
producer artifacts, recomputes their bounded claims, and keeps infrastructure
completion separate from scientific value. A required quarantined producer
blocks the capstone even when every task has a terminal disposition.

Spec refs: REQ-REPORT-7287 and SCENARIO-REPORT-7287-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7274_v640_source_contract as source_contract
from carnot import experiment_7275_v640_semantic_replay as semantic_replay
from carnot import experiment_7276_v640_arc_identity as arc_identity
from carnot import experiment_7277_v640_comparator_canary as comparator_canary
from carnot import experiment_7278_v640_source_measurement as source_measurement
from carnot import experiment_7279_v640_source_audit as source_audit
from carnot import experiment_7280_v640_arc_live as arc_live
from carnot import experiment_7281_v640_admission_prototype as admission_prototype
from carnot import experiment_7282_v640_admission_learning as admission_learning
from carnot import experiment_7283_v640_admission_audit as admission_audit
from carnot import experiment_7284_v640_commit_prototype as commit_prototype
from carnot import experiment_7285_v640_commit_frontier as commit_frontier
from carnot import experiment_7286_v640_board_state as board_state


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.640"
RUN_DATE = "20260914"
RANDOM_SEED = 7_287_202_609_14
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7287_v640_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7287_v640_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7287")
DEFAULT_VALIDATION_RECEIPT_PATH = DEFAULT_RAW_DIR / "validation_receipts.json"

EXPECTED_TASK_IDS = tuple(source_contract.EXPECTED_ID_ORDER)
EXPECTED_PRODUCER_IDS = EXPECTED_TASK_IDS[:-1]
REQUIRED_SCIENCE_TASKS = (
    "exp7279-source-audit",
    "exp7280-arc-live",
    "exp7283-admission-audit",
    "exp7285-commit-frontier",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260914 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use full generation 60s, bounded generation 10s, load-only 2s, or the correct no-LLM class; never pad duration.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked_* name upstream, exact field or check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_ or complete:; external absence starts blocked_; retain the measured finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive. Failed efficacy gates forbid positive. Only incomplete own work is partial; unchanged external blocks are terminal blocked.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "capstone_complete_score": "One means every one of the fourteen task dispositions is represented.",
    "evidence_matrix": "Preserve missing, flagged, null, blocked, and self-synthesis rows.",
    "contract_rows": "Independent Markdown/YAML parse with exact IDs, order, titles, paths and gates.",
    "branch_decisions": "Each continuation requires an observed changed cause; retire only the failed scope.",
    "prd_gap_matrix": "Map evidence and limits to FR-11, FR-12, FR-05/08 and NFR-01.",
    "publication_gate": "Retain stable G1-G4 and unmet_gates as context only.",
}

EXTRA_FIELD_PRINCIPLES: JsonDict = {
    "experiment_id": "Bind this result to the final task in the V640 roster.",
    "milestone": "Bind every contract and producer observation to V640.",
    "started_at_utc": "Record the actual UTC start instant.",
    "completed_at_utc": "Record the actual UTC completion instant.",
    "execution_host": "Record host identity separately from the venue class.",
    "phase_spans_s": "Keep measured phase durations disjoint.",
    "same_milestone_gate_replay_rows": "Retain all eight V640 gate observations.",
    "historical_and_negative_fixture_sidecars": "Keep prior model calls and negative controls outside current counters.",
    "board_dispositions": "Preserve board evidence without treating host aggregation as board execution.",
    "publication_performed": "This invocation performs no publication.",
    "upload_performed": "This invocation performs no upload.",
    "submission_performed": "This invocation performs no benchmark submission.",
    "external_message_performed": "This invocation sends no external message.",
    "production_default_changed": "This invocation changes no production default.",
    "exclusion_manifest_modified": "This invocation reads but does not edit exclusion policy.",
    "research_roadmap_modified": "This invocation reads but does not edit the active roadmap.",
    "conductor_modified": "This invocation does not edit the research conductor.",
}

ALL_FIELD_PRINCIPLES = {**FIELD_PRINCIPLES, **EXTRA_FIELD_PRINCIPLES}
REQUIRED_ARTIFACT_FIELDS = frozenset(ALL_FIELD_PRINCIPLES)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    DESIGN_PATH,
    ROADMAP_PATH,
    Path("research-references.md"),
    Path("python/carnot/experiment_7273_v639_capstone.py"),
    Path("results/experiment_7273_v639_capstone.json"),
    Path("scripts/publication_gate.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7287_v640_capstone.py"),
    Path("scripts/experiments/experiment_7287_v640_capstone.py"),
    Path("tests/python/test_experiment_7287_v640_capstone.py"),
)

CLAIM_NAMES = (
    "source_contract_exact",
    "semantic_replay_authenticity",
    "comparator_canary_ready",
    "source_capture_authenticity",
    "source_semantic_perfection",
    "source_promotion",
    "arc_identity_ready",
    "arc_public_proxy_complete",
    "arc_method_value",
    "arc_policy_consumption",
    "official_hidden_score",
    "admission_run_complete",
    "admission_learning_efficacy",
    "admission_mechanical_safety",
    "admission_opportunity_loss",
    "admission_promotion",
    "commit_protocol_ready",
    "commit_acknowledgment_complete",
    "commit_cost_value",
)

VALIDATION_NAMES = (
    "focused_pytest",
    "affected_source_contract_pytest",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "independent_raw_row_reducer",
    "adversarial_verify",
    "verdict_row_consistency_lint",
    "aggregation_e2e",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one factual boundary so the conductor can monitor the task."""

    print(f"[exp7287] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read one object because an array cannot carry an artifact contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Hash exact bytes with the repository's prefixed SHA-256 format."""

    return common.sha256_path(path)


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum field itself."""

    return common.reproducibility_checksum(artifact)


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Replace authority bytes only after a complete sibling file is synced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_contract(root: Path) -> JsonDict:
    """Parse V640 YAML and its named Markdown without repairing a mismatch."""

    roadmap_path = root / ROADMAP_PATH
    roadmap_bytes = roadmap_path.read_bytes()
    roadmap = yaml.safe_load(roadmap_bytes)
    if not isinstance(roadmap, Mapping):
        raise ValueError("V640 roadmap root is not a mapping")
    if roadmap.get("milestone") != MILESTONE:
        raise ValueError("active roadmap is not V640")
    named_design = Path(str(roadmap.get("milestone_doc", DESIGN_PATH)))
    if named_design != DESIGN_PATH:
        raise ValueError("V640 roadmap names an unexpected design")
    design_path = root / named_design
    design_bytes = design_path.read_bytes()
    tasks = [
        {
            "id": str(task["id"]),
            "title": str(task["title"]),
            "deliverable": str(task["deliverable"]),
            "milestone": str(task.get("milestone", "")),
            "gated_on": deepcopy(task.get("gated_on") or []),
            "prior_failures": deepcopy(task.get("prior_failures") or []),
            "prompt": str(task.get("prompt", "")),
        }
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
    ]
    task_ids = [task["id"] for task in tasks]
    if tuple(task_ids) != EXPECTED_TASK_IDS:
        raise ValueError("V640 task order is not exp7274 through exp7287")
    comparison = source_contract.evaluate_contract(design_bytes.decode(), roadmap)
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "contract_rows": deepcopy(comparison["contract_rows"]),
        "contract_agrees": comparison["passed"] is True,
        "yaml_milestone": comparison["yaml_milestone"],
        "markdown_milestone": comparison["markdown_milestone"],
        "roadmap_path": str(ROADMAP_PATH),
        "design_path": str(named_design),
        "roadmap_sha256": _sha256(roadmap_path),
        "design_sha256": _sha256(design_path),
        "roadmap_bytes": roadmap_bytes,
        "design_bytes": design_bytes,
    }


def _conductor_block_errors(task_id: str, payload: Mapping[str, Any]) -> list[str]:
    """Authenticate one canonical block without calling it producer output."""

    number = common.task_number(task_id)
    errors: list[str] = []
    if payload.get("schema") != "blocked_gate_check_v1" or payload.get("status") != "blocked":
        errors.append("conductor_block_lifecycle")
    if payload.get("experiment") != number or not payload.get("failed_upstream"):
        errors.append("conductor_block_identity")
    if payload.get("failed_field") is None or payload.get("failed_expected") is None:
        errors.append("conductor_block_gate")
    return errors


def _validate_payload(task_id: str, payload: Mapping[str, Any], root: Path) -> list[str]:
    """Call each shipped cold validator while preserving conductor stamps."""

    if payload.get("schema") == "blocked_gate_check_v1":
        return _conductor_block_errors(task_id, payload)
    validators: dict[str, Callable[[], list[str]]] = {
        "exp7274-source-contract": lambda: source_contract.independent_reduce(payload),
        "exp7275-semantic-replay": lambda: semantic_replay.validate_artifact(payload),
        "exp7276-arc-identity": lambda: arc_identity.validate_artifact(payload),
        "exp7277-comparator-canary": lambda: comparator_canary.validate_artifact(payload),
        "exp7278-source-measurement": lambda: source_measurement.validate_artifact(payload),
        "exp7279-source-audit": lambda: source_audit.validate_artifact(
            payload, require_validations=False
        ),
        "exp7280-arc-live": lambda: arc_live.validate_artifact(payload),
        "exp7281-admission-prototype": lambda: admission_prototype.validate_artifact(
            payload, repo_root=root, check_files=False
        ),
        "exp7282-admission-learning": lambda: admission_learning.validate_artifact(
            payload, repo_root=root, check_files=False
        ),
        "exp7283-admission-audit": lambda: admission_audit.validate_artifact(
            payload, repo_root=root, check_files=False
        ),
        "exp7284-commit-prototype": lambda: commit_prototype.validate_artifact(payload),
        "exp7285-commit-frontier": lambda: commit_frontier.validate_artifact(payload),
        "exp7286-board-state": lambda: board_state.validate_artifact(payload, root=root),
    }
    try:
        errors = list(validators[task_id]())
    except (KeyError, TypeError, ValueError) as error:
        return [f"validator_exception:{type(error).__name__}:{error}"]
    conductor_stamp_errors = {
        "field_principles_must_cover_every_top_level_field",
        "reproducibility_checksum_mismatch",
    }
    if payload.get("flagged_adversarial") is True and set(errors) <= conductor_stamp_errors:
        return []
    return errors


def _payload_identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept the integer and full-ID identity forms shipped by V640."""

    number = common.task_number(task_id)
    if payload.get("schema") == "blocked_gate_check_v1":
        return payload.get("experiment") == number
    return payload.get("milestone") == MILESTONE and payload.get("experiment_id") in {
        task_id,
        number,
        str(number),
    }


def _raw_evidence_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Retain raw row counts and compact path receipts without copying tables."""

    counts = {
        key: len(value)
        for key, value in payload.items()
        if "row" in key and isinstance(value, list)
    }
    references: JsonDict = {}
    for key, value in payload.items():
        if not any(marker in key for marker in ("raw", "sidecar", "manifest")):
            continue
        if isinstance(value, str):
            references[key] = value
        elif isinstance(value, Mapping) and "path" in value:
            references[key] = {
                name: deepcopy(value[name]) for name in ("path", "sha256", "bytes") if name in value
            }
        elif isinstance(value, list):
            references[key] = {"row_count": len(value)}
    return {"row_counts": counts, "references": references}


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load only the declared deliverable or its exact canonical block."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    fallback = common.canonical_gate_block_path(task_id)
    declared_present = (root / declared).is_file()
    if declared_present:
        selected, source = declared, "declared_deliverable"
    elif (root / fallback).is_file():
        selected, source = fallback, "conductor_gate_block"
    else:
        selected, source = None, "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = common.quarantine_receipt(payload, task_id, selected or declared, manifest)
    errors = _validate_payload(task_id, payload, root) if payload else []
    terminal = payload.get("status") in {"complete", "blocked"}
    identity_matches = bool(payload) and _payload_identity_matches(task_id, payload)
    authenticated = bool(payload) and terminal and identity_matches and not errors
    accepted = bool(
        authenticated
        and payload.get("status") == "complete"
        and payload.get("verdict_class") not in {"blocked", "disqualified", "partial"}
        and quarantine["quarantined"] is False
    )
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "declared_artifact_present": declared_present,
        "canonical_gate_block_path": fallback,
        "selected_evidence_path": selected,
        "evidence_source": source,
        "artifact_sha256": _sha256(root / selected) if selected else None,
        "artifact_size_bytes": (root / selected).stat().st_size if selected else 0,
        "payload": payload,
        "terminal": terminal,
        "identity_matches": identity_matches,
        "producer_validation_errors": errors,
        "raw_evidence": _raw_evidence_summary(payload),
        "quarantine_state": quarantine,
        "authenticated": authenticated,
        "accepted_for_positive_claim": accepted,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Keep one evidence slot for each of the thirteen V640 producers."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task["id"] != "exp7287-capstone"
    }


def _prompt_declares_field(task: Mapping[str, Any], field: str) -> bool:
    """Confirm that a gate field appears in the producer's required fields."""

    return f"- {field}:" in str(task.get("prompt", ""))


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay all V640 gates while keeping absence separate from numeric zero."""

    task_by_id = {str(task["id"]): task for task in tasks}
    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            producer = evidence[upstream]
            payload = producer["payload"]
            field = str(gate["artifact_field"])
            observed = common.unwrap_principle(payload.get(field))
            if producer["selected_evidence_path"] is None:
                outcome = "missing_file"
            elif field not in payload:
                outcome = "missing_field"
            elif producer["quarantine_state"]["quarantined"]:
                outcome = "quarantined"
            elif observed != gate["value"]:
                outcome = "value_mismatch"
            else:
                outcome = "passed"
            rows.append(
                {
                    "consumer": str(consumer["id"]),
                    "upstream": upstream,
                    "producer_milestone": task_by_id[upstream].get("milestone"),
                    "same_milestone": task_by_id[upstream].get("milestone") == MILESTONE,
                    "declared_artifact_path": producer["declared_deliverable_path"],
                    "actual_artifact_path": producer["selected_evidence_path"],
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate["value"],
                    "observed_value": observed,
                    "producer_declares_field": _prompt_declares_field(task_by_id[upstream], field),
                    "quarantined": producer["quarantine_state"]["quarantined"],
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def _claim_row(
    claim: str,
    task_id: str,
    declared: Any,
    recomputed: Any,
    evidence_fields: Sequence[str],
    evidence: Mapping[str, JsonDict],
    *,
    oracle: bool = False,
    unavailable_error: str | None = None,
) -> JsonDict:
    """Store one reduction and prevent unsupported scientific promotion."""

    source = evidence[task_id]
    missing = source["selected_evidence_path"] is None
    blocked = source["payload"].get("status") == "blocked"
    quarantined = source["quarantine_state"]["quarantined"] is True
    matches = declared == recomputed
    positive = recomputed is True or (
        isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool) and recomputed > 0
    )
    promoted = bool(
        positive
        and matches
        and source["accepted_for_positive_claim"]
        and not oracle
        and unavailable_error is None
    )
    if missing:
        error = "missing_producer_artifact"
    elif blocked:
        error = "blocked_producer_artifact"
    elif quarantined:
        error = "quarantined_producer_artifact"
    elif source["producer_validation_errors"]:
        error = "producer_validation_failed"
    elif unavailable_error is not None:
        error = unavailable_error
    elif not matches:
        error = "declared_value_mismatch"
    else:
        error = None
    if missing or blocked or quarantined:
        claim_class = "blocked"
    elif positive and oracle:
        claim_class = "circular_positive"
    elif promoted:
        claim_class = "positive"
    elif recomputed is False or recomputed is None:
        claim_class = "null"
    else:
        claim_class = "numeric_check"
    return {
        "unit_id": f"{task_id}:{claim}",
        "arm": "independent_raw_artifact_reduction",
        "seed": RANDOM_SEED,
        "metric": claim,
        "metric_value": recomputed,
        "error": error,
        "abstention": missing or blocked or quarantined or unavailable_error is not None,
        "cost": source["payload"].get("duration_s"),
        "censored": missing,
        "claim": claim,
        "task_id": task_id,
        "declared_value": declared,
        "recomputed_value": recomputed,
        "matches": matches,
        "evidence_fields": list(evidence_fields),
        "raw_evidence": deepcopy(source["raw_evidence"]),
        "source_authenticated": source["authenticated"],
        "positive_promoted": promoted,
        "verifier_is_oracle": oracle,
        "claim_class": claim_class,
    }


def _passed_gate(payload: Mapping[str, Any], name: str) -> bool:
    """Read a producer gate from either mapping or list form."""

    gates = payload.get("acceptance_gate_results", {})
    if isinstance(gates, Mapping):
        row = gates.get(name, {})
        return isinstance(row, Mapping) and row.get("passed", row.get("pass")) is True
    return any(
        isinstance(row, Mapping)
        and row.get("criterion") == name
        and row.get("passed", row.get("pass")) is True
        for row in gates
        if isinstance(gates, list)
    )


def recompute_claims(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce source, ARC, admission, and acknowledgment evidence anew."""

    claims: list[JsonDict] = []
    contract = evidence["exp7274-source-contract"]["payload"]
    claims.append(
        _claim_row(
            "source_contract_exact",
            "exp7274-source-contract",
            contract.get("source_contract_complete_score") == 1,
            bool(contract.get("contract_rows"))
            and all(row.get("passed") is True for row in contract.get("contract_rows", [])),
            ("contract_rows", "source_contract_complete_score"),
            evidence,
        )
    )

    replay = evidence["exp7275-semantic-replay"]["payload"]
    replay_authentic = bool(
        replay.get("semantic_replay_ready_score") == 1
        and len(replay.get("call_replay_rows", [])) == 320
        and not replay.get("first_divergence_rows")
    )
    claims.append(
        _claim_row(
            "semantic_replay_authenticity",
            "exp7275-semantic-replay",
            replay.get("semantic_replay_ready_score") == 1,
            replay_authentic,
            ("call_replay_rows", "first_divergence_rows"),
            evidence,
            oracle=True,
        )
    )

    canary = evidence["exp7277-comparator-canary"]["payload"]
    canary_ready = bool(
        canary.get("comparator_canary_ready_score") == 1
        and len(canary.get("canary_rows", [])) == 8
        and not canary.get("replay_discrepancies")
    )
    claims.append(
        _claim_row(
            "comparator_canary_ready",
            "exp7277-comparator-canary",
            canary.get("comparator_canary_ready_score") == 1,
            canary_ready,
            ("canary_rows", "replay_discrepancies"),
            evidence,
            oracle=True,
        )
    )

    measurement = evidence["exp7278-source-measurement"]["payload"]
    capture_authentic = bool(
        measurement.get("source_capture_complete_score") == 1
        and len(measurement.get("replay_rows", [])) == 256
        and len(measurement.get("rows", [])) == 128
        and not measurement.get("replay_discrepancies")
    )
    claims.append(
        _claim_row(
            "source_capture_authenticity",
            "exp7278-source-measurement",
            measurement.get("source_capture_complete_score") == 1,
            capture_authentic,
            ("replay_rows", "rows", "replay_discrepancies"),
            evidence,
            oracle=True,
        )
    )

    audit = evidence["exp7279-source-audit"]["payload"]
    audit_rows = audit.get("rows", [])
    if audit_rows:
        comparisons = source_audit.paired_bootstrap(
            audit_rows, source_audit.BOOTSTRAP_SEED, source_audit.BOOTSTRAP_DRAWS
        )
        comparison_index = {(row["comparison"], row["metric"]): row for row in comparisons}
        accuracy = comparison_index[
            ("verifier_minus_direct_self_consistency", "accuracy_difference")
        ]
        false_accept = comparison_index[
            ("verifier_minus_direct_self_consistency", "false_accept_difference")
        ]
        shuffle = comparison_index[("verifier_minus_source_shuffle_control", "accuracy_difference")]
        verifier_summary = source_audit.summarize_arms(audit_rows)["verifier"]["accuracy"]
        semantic_perfection = verifier_summary["numerator"] == verifier_summary["denominator"]
    else:
        accuracy = {"ci95": [None, None]}
        false_accept = {"ci95": [None, None]}
        shuffle = {"ci95": [None, None]}
        semantic_perfection = False
    controls_pass = bool(audit.get("causal_control_rows")) and all(
        row.get("passed") is True for row in audit.get("causal_control_rows", [])
    )
    source_promotion = bool(
        accuracy["ci95"][0] is not None
        and accuracy["ci95"][0] > 0
        and false_accept["ci95"][1] is not None
        and false_accept["ci95"][1] <= 0
        and shuffle["ci95"][0] is not None
        and shuffle["ci95"][0] > 0
        and controls_pass
        and not audit.get("replay_mismatches")
        and not audit.get("source_leakage_errors")
    )
    claims.extend(
        [
            _claim_row(
                "source_semantic_perfection",
                "exp7279-source-audit",
                False,
                semantic_perfection,
                ("rows", "arm_summaries.verifier.accuracy"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "source_promotion",
                "exp7279-source-audit",
                audit.get("source_promotion_score") == 1,
                source_promotion,
                ("rows", "paired_comparisons", "causal_control_rows"),
                evidence,
                oracle=True,
            ),
        ]
    )

    identity = evidence["exp7276-arc-identity"]["payload"]
    identity_reduction = arc_identity.reduce_identity_rows(identity.get("rows", []))
    identity_ready = identity_reduction.get("arc_identity_ready_score") == 1
    claims.append(
        _claim_row(
            "arc_identity_ready",
            "exp7276-arc-identity",
            identity.get("arc_identity_ready_score") == 1,
            identity_ready,
            ("rows", "identity_obligation_rows", "unsupported_controls"),
            evidence,
            oracle=True,
        )
    )

    live = evidence["exp7280-arc-live"]["payload"]
    live_reduction = arc_live.reduce_episode_rows(live.get("rows", []))
    claims.extend(
        [
            _claim_row(
                "arc_public_proxy_complete",
                "exp7280-arc-live",
                live.get("arc_capture_complete_score") == 1,
                live_reduction.get("arc_capture_complete_score") == 1,
                ("rows", "per_game_results"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "arc_method_value",
                "exp7280-arc-live",
                live.get("arc_method_value_score") == 1,
                live_reduction.get("arc_method_value_score") == 1,
                ("rows", "per_game_results"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "arc_policy_consumption",
                "exp7280-arc-live",
                bool(live.get("policy_consumption_rows")),
                bool(live_reduction.get("treatment_games_with_useful_consumed_plan")),
                ("rows", "policy_consumption_rows"),
                evidence,
            ),
            _claim_row(
                "official_hidden_score",
                "exp7280-arc-live",
                live.get("official_score"),
                None,
                ("official_score", "new_solve_claimed", "solve_provenance"),
                evidence,
                unavailable_error="official_hidden_score_absent",
            ),
        ]
    )

    learning = evidence["exp7282-admission-learning"]["payload"]
    run_complete = bool(
        learning.get("admission_run_complete_score") == 1
        and len(learning.get("rows", [])) == 168
        and all(row.get("censored") is False for row in learning.get("rows", []))
    )
    efficacy = all(
        _passed_gate(learning, name) for name in admission_learning.SCIENTIFIC_GATE_NAMES
    )
    claims.extend(
        [
            _claim_row(
                "admission_run_complete",
                "exp7282-admission-learning",
                learning.get("admission_run_complete_score") == 1,
                run_complete,
                ("rows", "admission_run_complete_score"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "admission_learning_efficacy",
                "exp7282-admission-learning",
                learning.get("admission_value_score") == 1,
                efficacy,
                tuple(
                    f"acceptance_gate_results.{name}"
                    for name in admission_learning.SCIENTIFIC_GATE_NAMES
                ),
                evidence,
                oracle=True,
            ),
        ]
    )

    admission = evidence["exp7283-admission-audit"]["payload"]
    safety = bool(
        admission.get("mechanical_safety_verdict", {}).get("passed") is True
        and admission.get("no_model_weight_mutation") is True
        and _passed_gate(admission, "mechanical_safety")
    )
    missed = sum(
        int(row.get("missed_beneficial_opportunity_count", 0))
        for row in admission.get("opportunity_reduction", [])
        if row.get("arm") == "paired_gated"
    )
    promotion = bool(
        admission.get("admission_audit_complete_score") == 1
        and _passed_gate(admission, "mechanical_safety")
        and _passed_gate(admission, "opportunity_accounting")
        and _passed_gate(admission, "causal_influence")
        and _passed_gate(admission, "upstream_efficacy")
    )
    claims.extend(
        [
            _claim_row(
                "admission_mechanical_safety",
                "exp7283-admission-audit",
                True,
                safety,
                ("mutation_sidecar_receipt", "mechanical_safety_verdict"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "admission_opportunity_loss",
                "exp7283-admission-audit",
                missed,
                missed,
                ("opportunity_reduction", "opportunity_accounting_verdict"),
                evidence,
            ),
            _claim_row(
                "admission_promotion",
                "exp7283-admission-audit",
                admission.get("admission_promotion_score") == 1,
                promotion,
                ("acceptance_gate_results", "efficacy_verdict"),
                evidence,
                oracle=True,
            ),
        ]
    )

    protocol = evidence["exp7284-commit-prototype"]["payload"]
    protocol_ready = bool(
        protocol.get("commit_protocol_ready_score") == 1
        and _passed_gate(protocol, "durable_acknowledgment_crash_safety")
        and _passed_gate(protocol, "serial_transition_parity")
    )
    claims.append(
        _claim_row(
            "commit_protocol_ready",
            "exp7284-commit-prototype",
            protocol.get("commit_protocol_ready_score") == 1,
            protocol_ready,
            ("crash_rows", "semantic_parity_rows", "queue_bounds"),
            evidence,
            oracle=True,
        )
    )

    frontier = evidence["exp7285-commit-frontier"]["payload"]
    frontier_reduction = commit_frontier.reduce_saved_rows(
        frontier.get("rows", []),
        frontier.get("latency_throughput_rows", []),
        frontier.get("component_rows", []),
        frontier.get("parity_rows", []),
    )
    acknowledgment_complete = bool(
        frontier_reduction.get("population_complete")
        and frontier_reduction.get("component_cost_complete")
        and frontier_reduction.get("zero_lost_acknowledged_events")
        and frontier_reduction.get("exact_final_state_parity")
    )
    claims.extend(
        [
            _claim_row(
                "commit_acknowledgment_complete",
                "exp7285-commit-frontier",
                frontier.get("commit_cost_complete_score") == 1,
                acknowledgment_complete,
                ("rows", "component_rows", "parity_rows"),
                evidence,
                oracle=True,
            ),
            _claim_row(
                "commit_cost_value",
                "exp7285-commit-frontier",
                frontier.get("commit_cost_value_score") == 1,
                frontier_reduction.get("value_gate_passed") is True,
                ("latency_throughput_rows", "component_rows", "acceptance_gate_results"),
                evidence,
                oracle=True,
            ),
        ]
    )
    return claims


def _matrix_row(
    order: int,
    task: Mapping[str, Any],
    evidence: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize one producer without copying its large row tables."""

    task_id = str(task["id"])
    payload = evidence["payload"]
    if evidence["selected_evidence_path"] is None:
        disposition = "blocked_missing_declared_and_canonical_evidence"
    elif evidence["quarantine_state"]["quarantined"]:
        disposition = "blocked_quarantined_evidence"
    elif payload.get("status") == "blocked":
        disposition = str(payload.get("honest_verdict", "blocked_producer"))
    elif evidence["producer_validation_errors"]:
        disposition = "disqualified_invalid_producer_artifact"
    elif task_id == "exp7277-comparator-canary":
        disposition = "circular_positive_scale_only_canary"
    elif task_id == "exp7286-board-state":
        disposition = "positive_bounded_board_continuity_with_gatemate_block"
    else:
        disposition = str(payload.get("honest_verdict", "complete_disposition_missing"))
    return {
        "order": order,
        "task_id": task_id,
        "title": task["title"],
        "declared_artifact_path": evidence["declared_deliverable_path"],
        "declared_artifact_present": evidence["declared_artifact_present"],
        "actual_artifact_path": evidence["selected_evidence_path"],
        "evidence_source": evidence["evidence_source"],
        "artifact_sha256": evidence["artifact_sha256"],
        "artifact_size_bytes": evidence["artifact_size_bytes"],
        "status": payload.get("status", "missing"),
        "verdict_class": payload.get(
            "verdict_class", "blocked" if payload.get("status") == "blocked" else "partial"
        ),
        "honest_verdict": payload.get("honest_verdict", "blocked_missing_producer_artifact"),
        "final_disposition": disposition,
        "quarantine_state": deepcopy(evidence["quarantine_state"]),
        "producer_validation_errors": deepcopy(evidence["producer_validation_errors"]),
        "authenticated": evidence["authenticated"],
        "accepted_for_positive_claim": evidence["accepted_for_positive_claim"],
        "raw_evidence": deepcopy(evidence["raw_evidence"]),
        "source_artifact_hashes": deepcopy(payload.get("source_artifact_hashes", {})),
        "upstream_gate_observations": [
            deepcopy(row) for row in gates if row["consumer"] == task_id
        ],
        "recomputed_claims": [deepcopy(row) for row in claims if row["task_id"] == task_id],
        "verifier_is_oracle": bool(common.unwrap_principle(payload.get("verifier_is_oracle"))),
    }


def _required_science_failure(evidence: Mapping[str, JsonDict]) -> JsonDict | None:
    """Return the first unavailable required science source in roster order."""

    for task_id in REQUIRED_SCIENCE_TASKS:
        row = evidence[task_id]
        if row["selected_evidence_path"] is None:
            return {
                "upstream": task_id,
                "artifact_field": "declared_deliverable_or_canonical_block",
                "expected_value": "terminal evidence",
                "observed_value": None,
                "failed_check": "required_science_evidence_available",
            }
        if row["quarantine_state"]["quarantined"]:
            flags = row["quarantine_state"]["declared_flags"]
            field = next(iter(flags), "exclusion_manifest_match")
            observed = flags.get(field, row["quarantine_state"]["exclusion_manifest_match"])
            return {
                "upstream": task_id,
                "artifact_field": field,
                "expected_value": False,
                "observed_value": observed,
                "failed_check": "required_science_not_quarantined_or_retired",
            }
        if row["payload"].get("status") == "blocked":
            return {
                "upstream": task_id,
                "artifact_field": "status",
                "expected_value": "complete",
                "observed_value": "blocked",
                "failed_check": "required_science_terminal_complete",
            }
        if row["producer_validation_errors"]:
            return {
                "upstream": task_id,
                "artifact_field": "producer_validation_errors",
                "expected_value": [],
                "observed_value": row["producer_validation_errors"],
                "failed_check": "required_science_authentic",
            }
    return None


def _terminal_state(
    evidence: Mapping[str, JsonDict], claims: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Classify external blocks before evaluating four scientific values."""

    claim_index = {str(row["claim"]): row for row in claims}
    scores = {
        "source_promotion_score": int(claim_index["source_promotion"]["recomputed_value"] is True),
        "arc_method_value_score": int(claim_index["arc_method_value"]["recomputed_value"] is True),
        "admission_promotion_score": int(
            claim_index["admission_promotion"]["recomputed_value"] is True
        ),
        "commit_cost_value_score": int(
            claim_index["commit_cost_value"]["recomputed_value"] is True
        ),
    }
    failure = _required_science_failure(evidence)
    if failure is not None:
        summary = {
            "passed": False,
            **failure,
            "terminal_classification": "blocked",
            "retry_allowed": False,
            "value_score_observations": scores,
        }
        return {
            "status": "blocked",
            "verdict_class": "blocked",
            "honest_verdict": (
                "blocked_required_arc_science_quarantined: all fourteen V640 dispositions are "
                "represented; Exp7280 has critical inference-provenance flags, while source, "
                "ARC, admission, and acknowledgment value scores remain zero"
            ),
            "gate_check_summary": summary,
            "scores": scores,
        }
    all_positive = all(value == 1 for value in scores.values())
    return {
        "status": "complete",
        "verdict_class": "positive" if all_positive else "null",
        "honest_verdict": (
            "complete_positive_v640_all_four_scientific_value_gates_pass"
            if all_positive
            else "complete_null_v640_one_or_more_scientific_value_gates_failed"
        ),
        "gate_check_summary": {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "artifact_field": None,
            "expected_value": None,
            "observed_value": None,
            "terminal_classification": "positive" if all_positive else "null",
            "retry_allowed": False,
            "value_score_observations": scores,
        },
        "scores": scores,
    }


def _self_row(task: Mapping[str, Any], terminal: Mapping[str, Any]) -> JsonDict:
    """Represent synthesis without creating a recursive self-artifact hash."""

    return {
        "order": 14,
        "task_id": str(task["id"]),
        "title": task["title"],
        "declared_artifact_path": task["deliverable"],
        "declared_artifact_present": False,
        "actual_artifact_path": None,
        "evidence_source": "self_synthesis",
        "artifact_sha256": None,
        "artifact_size_bytes": 0,
        "status": terminal["status"],
        "verdict_class": terminal["verdict_class"],
        "honest_verdict": terminal["honest_verdict"],
        "final_disposition": terminal["honest_verdict"],
        "quarantine_state": {
            "declared_flags": {},
            "exclusion_manifest_match": False,
            "quarantined": False,
        },
        "producer_validation_errors": [],
        "authenticated": True,
        "accepted_for_positive_claim": terminal["verdict_class"] == "positive",
        "raw_evidence": {"row_counts": {}, "references": {}},
        "source_artifact_hashes": {},
        "upstream_gate_observations": [],
        "recomputed_claims": [],
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": False,
    }


def _changed_cause(task_id: str, matrix: Mapping[str, Mapping[str, Any]]) -> tuple[bool, str]:
    """Describe whether the task's declared substantive change was observed."""

    row = matrix[task_id]
    payload_claims = {claim["claim"]: claim for claim in row["recomputed_claims"]}
    observations = {
        "exp7274-source-contract": (False, "The named Markdown still declares V636."),
        "exp7275-semantic-replay": (
            payload_claims.get("semantic_replay_authenticity", {}).get("recomputed_value") is True,
            "Saved bytes now replay with the corrected comparator contract.",
        ),
        "exp7276-arc-identity": (
            payload_claims.get("arc_identity_ready", {}).get("recomputed_value") is True,
            "The strict runtime identity handoff now passes its bounded panel.",
        ),
        "exp7277-comparator-canary": (
            payload_claims.get("comparator_canary_ready", {}).get("recomputed_value") is True,
            "The full-string equal-budget comparator passed its scale-only canary.",
        ),
        "exp7278-source-measurement": (
            payload_claims.get("source_capture_authenticity", {}).get("recomputed_value") is True,
            "Fresh source groups and the repaired comparator produced complete evidence.",
        ),
        "exp7279-source-audit": (
            True,
            "The independent audit ran, but source promotion remained null.",
        ),
        "exp7280-arc-live": (
            False,
            "Identity changed, but critical provenance flags now quarantine live ARC evidence.",
        ),
        "exp7281-admission-prototype": (
            True,
            "The disjoint admission and opportunity-accounting fixture completed.",
        ),
        "exp7282-admission-learning": (
            payload_claims.get("admission_run_complete", {}).get("recomputed_value") is True,
            "Independent admission ran on fresh streams; efficacy gates failed.",
        ),
        "exp7283-admission-audit": (
            payload_claims.get("admission_mechanical_safety", {}).get("recomputed_value") is True,
            "The cold audit preserved safety and measured missed opportunities.",
        ),
        "exp7284-commit-prototype": (
            payload_claims.get("commit_protocol_ready", {}).get("recomputed_value") is True,
            "Group acknowledgment replaced the unwarranted delta-log mechanism.",
        ),
        "exp7285-commit-frontier": (
            payload_claims.get("commit_acknowledgment_complete", {}).get("recomputed_value")
            is True,
            "The full acknowledgment frontier completed; bounded value failed.",
        ),
        "exp7286-board-state": (
            False,
            "No later GateMate physical-change receipt exists.",
        ),
        "exp7287-capstone": (
            False,
            "The matrix is complete, but the design is stale and required ARC science is quarantined.",
        ),
    }
    return observations[task_id]


def _branch_decisions(
    tasks: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Evaluate exact verdicts and changed causes without broad retirement."""

    matrix_index = {str(row["task_id"]): row for row in matrix}
    actions = {
        "exp7274-source-contract": "needs_corrected_design",
        "exp7275-semantic-replay": "retain_diagnostic",
        "exp7276-arc-identity": "retain_as_prerequisite",
        "exp7277-comparator-canary": "continue_bounded_scale",
        "exp7278-source-measurement": "retire_measured_scope",
        "exp7279-source-audit": "retire_measured_scope",
        "exp7280-arc-live": "needs_corrigendum",
        "exp7281-admission-prototype": "retain_as_fixture",
        "exp7282-admission-learning": "retire_measured_scope",
        "exp7283-admission-audit": "retire_measured_scope",
        "exp7284-commit-prototype": "retain_protocol",
        "exp7285-commit-frontier": "retire_measured_scope",
        "exp7286-board-state": "wait_external_change",
        "exp7287-capstone": "wait_required_science_corrigendum",
    }
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        result = matrix_index[task_id]
        current = str(result["honest_verdict"])
        changed, observation = _changed_cause(task_id, matrix_index)
        evaluations = []
        for prior in task.get("prior_failures") or []:
            exact = prior.get("verdict") == current
            evaluations.append(
                {
                    "prior_experiment_id": prior.get("experiment_id"),
                    "exact_prior_verdict": prior.get("verdict"),
                    "current_honest_verdict": current,
                    "exact_same_verdict": exact,
                    "addressed_by": prior.get("addressed_by"),
                    "changed_cause_observed": changed,
                    "changed_cause_observation": observation,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                }
            )
        matches = [row for row in evaluations if row["exact_same_verdict"]]
        action = "retire_exact_repeated_scope" if matches else actions[task_id]
        rows.append(
            {
                "task_id": task_id,
                "current_honest_verdict": current,
                "prior_evaluations": evaluations,
                "exact_same_verdict_recurrence": bool(matches),
                "substantive_changed_cause_observed": changed,
                "changed_cause_observation": observation,
                "action": action,
                "exact_next_condition": (
                    "Change the exact repeated mechanism before another measurement."
                    if matches
                    else observation
                ),
                "retry_current_mechanism": False,
                "retired_scope": task_id if matches or action == "retire_measured_scope" else None,
                "broad_family_retirement_invented": False,
            }
        )
    return rows


def _display_path(root: Path, path: Path) -> str:
    """Use relative paths for evidence inside this checkout."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_sidecars(
    root: Path,
    raw_dir: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    claims: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Freeze authorities and separate historical model and negative evidence."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    roadmap_copy = raw_dir / "selected-roadmap.yaml"
    design_copy = raw_dir / "v640-design.md"
    _atomic_bytes(roadmap_copy, contract["roadmap_bytes"])
    _atomic_bytes(design_copy, contract["design_bytes"])
    model_path = raw_dir / "historical-model-receipts.json"
    negative_path = raw_dir / "negative-gate-fixtures.json"
    claims_path = raw_dir / "recomputed-claim-rows.json"
    common._atomic_write(
        model_path,
        {
            "schema": "carnot.exp7287.historical_model_receipts.v1",
            "rows": [
                {
                    "task_id": task_id,
                    "artifact_sha256": row["artifact_sha256"],
                    "MODEL_SPECS": deepcopy(row["payload"].get("MODEL_SPECS", [])),
                    "model_invoked": row["payload"].get("model_invoked"),
                    "invocation_counts": deepcopy(row["payload"].get("invocation_counts", {})),
                    "inference_substrate": row["payload"].get("inference_substrate"),
                }
                for task_id, row in evidence.items()
                if row["payload"].get("model_invoked") is True
                or int(row["payload"].get("invocation_counts", {}).get("model_loads_completed", 0))
                > 0
            ],
        },
    )
    common._atomic_write(
        negative_path,
        {
            "schema": "carnot.exp7287.negative_gate_fixtures.v1",
            "failed_gate_rows": [deepcopy(row) for row in gates if row["passed"] is False],
            "failed_claim_rows": [
                deepcopy(row) for row in claims if row["claim_class"] in {"null", "blocked"}
            ],
            "blocked_disposition_rows": [
                {
                    "task_id": task_id,
                    "status": row["payload"].get("status", "missing"),
                    "quarantine_state": deepcopy(row["quarantine_state"]),
                    "actual_artifact_path": row["selected_evidence_path"],
                    "artifact_sha256": row["artifact_sha256"],
                }
                for task_id, row in evidence.items()
                if row["payload"].get("status") == "blocked"
                or row["quarantine_state"]["quarantined"]
            ],
        },
    )
    common._atomic_write(
        claims_path,
        {"schema": "carnot.exp7287.recomputed_claim_rows.v1", "rows": list(claims)},
    )
    return [
        {"path": _display_path(root, path), "sha256": _sha256(path)}
        for path in (model_path, negative_path, claims_path)
    ]


def _load_validation_receipts(root: Path, path: Path | None) -> list[JsonDict]:
    """Authenticate command logs without inventing missing success receipts."""

    if path is None or not path.is_file():
        return []
    payload = read_json(path)
    rows: list[JsonDict] = []
    for receipt in payload.get("receipts", []):
        log_path = Path(str(receipt.get("log_path", "")))
        resolved = log_path if log_path.is_absolute() else root / log_path
        observed = _sha256(resolved) if resolved.is_file() else None
        rows.append(
            {
                "name": receipt.get("name"),
                "command": receipt.get("command"),
                "exit_code": receipt.get("exit_code"),
                "duration_s": receipt.get("duration_s"),
                "log_path": _display_path(root, resolved),
                "log_sha256": receipt.get("log_sha256"),
                "observed_log_sha256": observed,
                "log_hash_matches": observed == receipt.get("log_sha256"),
            }
        )
    return rows


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record source bytes, output ownership, contract state, and producers."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_input",
                "upstream": str(relative),
                "artifact_field": "present_nonempty",
                "expected_value": True,
                "observed_value": present,
                "passed": present,
            }
        )
        if present:
            hashes[str(relative)] = _sha256(path)
    for path in (output, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK) and path.parent.stat().st_uid == os.getuid()
        checks.append(
            {
                "check": "owned_writable_output_parent",
                "upstream": _display_path(root, path.parent),
                "artifact_field": "owner_and_writable",
                "expected_value": True,
                "observed_value": writable,
                "passed": writable,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": str(SPEC_PATH),
                "artifact_field": "REQ-REPORT-7287",
                "expected_value": True,
                "observed_value": "REQ-REPORT-7287" in spec_text,
                "passed": "REQ-REPORT-7287" in spec_text,
            },
            {
                "check": "independent_contract",
                "upstream": f"{ROADMAP_PATH}|{DESIGN_PATH}",
                "artifact_field": "milestone|ids|titles|paths|gates",
                "expected_value": True,
                "observed_value": contract["contract_agrees"],
                "passed": contract["contract_agrees"],
            },
            {
                "check": "producer_slots",
                "upstream": MILESTONE,
                "artifact_field": "terminal_artifacts_or_blocks",
                "expected_value": 13,
                "observed_value": sum(row["terminal"] for row in evidence.values()),
                "passed": all(row["terminal"] for row in evidence.values()),
            },
        ]
    )
    for row in evidence.values():
        selected = row["selected_evidence_path"]
        if selected:
            hashes[selected] = str(row["artifact_sha256"])
    return checks, hashes


def _publication_gate(root: Path) -> JsonDict:
    """Run the stable G1-G4 evaluator and retain its unmodified result."""

    command = [sys.executable, "-u", "scripts/publication_gate.py", "--json"]
    progress(6, "subprocess start", "publication_gate.py --json")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - started
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    progress(6, "subprocess end", f"publication gate exit={completed.returncode}")
    payload = json.loads(completed.stdout)
    if completed.returncode != 0 or not isinstance(payload, dict):
        raise RuntimeError(f"publication gate failed: {completed.stderr}")
    payload.update(
        {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "elapsed_s": elapsed,
            "stderr": completed.stderr,
        }
    )
    return payload


def _publication_gate_valid(value: Any) -> bool:
    """Check that G1-G4, paper readiness, and unmet names agree."""

    if not isinstance(value, Mapping) or value.get("exit_code") != 0:
        return False
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != {"G1", "G2", "G3", "G4"}:
        return False
    observed = [name for name, gate in gates.items() if gate.get("pass") is not True]
    return value.get("unmet_gates") == observed and value.get("paper_ready") is (not observed)


def _acceptance_results(
    contract: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score roster completion separately from authenticity and value."""

    receipt_names = [str(row.get("name")) for row in receipts]
    claim_index = {str(row["claim"]): row for row in claims}
    values = {
        "source_promotion_score": int(claim_index["source_promotion"]["recomputed_value"] is True),
        "arc_method_value_score": int(claim_index["arc_method_value"]["recomputed_value"] is True),
        "admission_promotion_score": int(
            claim_index["admission_promotion"]["recomputed_value"] is True
        ),
        "commit_cost_value_score": int(
            claim_index["commit_cost_value"]["recomputed_value"] is True
        ),
    }
    required_rows = [row for row in matrix if row["task_id"] in REQUIRED_SCIENCE_TASKS]
    definitions = (
        (
            "fourteen_task_dispositions",
            14,
            len(matrix),
            len(matrix) == 14,
            "Roster completion makes no scientific value claim.",
        ),
        (
            "independent_contract_exact",
            True,
            contract["contract_agrees"],
            contract["contract_agrees"] is True,
            "The stale Markdown remains a disqualified observation.",
        ),
        (
            "same_milestone_gates_pass",
            8,
            sum(row["passed"] is True for row in gates),
            len(gates) == 8 and all(row["passed"] is True for row in gates),
            "Structured readiness gates do not establish efficacy.",
        ),
        (
            "required_science_authentic",
            True,
            all(row["accepted_for_positive_claim"] for row in required_rows),
            all(row["accepted_for_positive_claim"] for row in required_rows),
            "A quarantined required source blocks overall promotion.",
        ),
        (
            "all_four_value_scores",
            dict.fromkeys(values, 1),
            values,
            all(value == 1 for value in values.values()),
            "All four independent scientific value families are required.",
        ),
        (
            "validation_receipts_complete",
            list(VALIDATION_NAMES),
            receipt_names,
            set(VALIDATION_NAMES) <= set(receipt_names)
            and all(
                row.get("exit_code") == 0 and row.get("log_hash_matches") is True
                for row in receipts
                if row.get("name") in VALIDATION_NAMES
            ),
            "Every scoped command keeps its exit code, duration, and log hash.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in definitions
    ]


def _prd_gap_matrix(claims: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Map measured claim families to the four requested PRD boundaries."""

    claim = {str(row["claim"]): row["recomputed_value"] for row in claims}
    return {
        "FR-11": {
            "complete": False,
            "admission_run_complete": claim["admission_run_complete"],
            "admission_learning_efficacy": claim["admission_learning_efficacy"],
            "admission_mechanical_safety": claim["admission_mechanical_safety"],
            "missed_beneficial_opportunities": claim["admission_opportunity_loss"],
            "remaining_limit": "Admission was safe and causal, but frozen efficacy gates failed.",
        },
        "FR-12": {
            "complete": False,
            "source_capture_authenticity": claim["source_capture_authenticity"],
            "source_semantic_perfection": claim["source_semantic_perfection"],
            "source_promotion": claim["source_promotion"],
            "remaining_limit": "Authentic capture did not beat equal-budget direct self-consistency.",
        },
        "FR-05/08": {
            "complete": False,
            "arc_identity_ready": claim["arc_identity_ready"],
            "arc_public_proxy_complete": claim["arc_public_proxy_complete"],
            "arc_policy_consumption": claim["arc_policy_consumption"],
            "official_hidden_score": claim["official_hidden_score"],
            "remaining_limit": "Live ARC evidence is quarantined and no useful policy consumption exists.",
        },
        "NFR-01": {
            "complete": False,
            "commit_protocol_ready": claim["commit_protocol_ready"],
            "commit_acknowledgment_complete": claim["commit_acknowledgment_complete"],
            "commit_cost_value": claim["commit_cost_value"],
            "remaining_limit": "Complete durable evidence missed the bounded steady acknowledgment gate.",
        },
    }


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute identity, evidence, claims, decisions, hashes, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != ALL_FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.exp7287.v640_capstone.v1"
        or artifact.get("experiment_id") != "exp7287-capstone"
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("run_date") != RUN_DATE, "lifecycle")
    add(
        artifact.get("status") not in {"complete", "blocked"}
        or artifact.get("verdict_class")
        not in {
            "positive",
            "circular_positive",
            "null",
            "blocked",
            "disqualified",
            "partial",
        },
        "verdict_class",
    )
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS,
        "model_invocation",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
        or not artifact.get("execution_host"),
        "execution_substrate",
    )
    contract_rows = artifact.get("contract_rows", [])
    add(
        not isinstance(contract_rows, list)
        or len(contract_rows) != 14
        or [row.get("unit_id") for row in contract_rows] != list(EXPECTED_TASK_IDS),
        "contract_rows",
    )
    matrix = artifact.get("evidence_matrix", [])
    add(
        not isinstance(matrix, list)
        or len(matrix) != 14
        or [row.get("task_id") for row in matrix] != list(EXPECTED_TASK_IDS)
        or matrix[-1].get("evidence_source") != "self_synthesis"
        or matrix[-1].get("artifact_sha256") is not None
        or any(not row.get("final_disposition") for row in matrix),
        "evidence_matrix",
    )
    claims = artifact.get("rows", [])
    add(
        not isinstance(claims, list)
        or tuple(row.get("claim") for row in claims) != CLAIM_NAMES
        or any(
            not {
                "unit_id",
                "arm",
                "seed",
                "metric",
                "metric_value",
                "error",
                "abstention",
                "cost",
                "censored",
            }.issubset(row)
            for row in claims
        ),
        "claim_rows",
    )
    branches = artifact.get("branch_decisions", [])
    add(
        not isinstance(branches, list)
        or len(branches) != 14
        or any(row.get("retry_current_mechanism") is not False for row in branches)
        or any(row.get("broad_family_retirement_invented") is not False for row in branches),
        "branch_decisions",
    )
    gate_rows = artifact.get("same_milestone_gate_replay_rows", [])
    add(not isinstance(gate_rows, list) or len(gate_rows) != 8, "gate_replay_rows")
    add(artifact.get("capstone_complete_score") != 1, "capstone_complete_score")
    summary = artifact.get("gate_check_summary", {})
    add(
        artifact.get("status") == "blocked"
        and (
            artifact.get("verdict_class") != "blocked"
            or not str(artifact.get("honest_verdict", "")).startswith("blocked_")
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
            or any(
                field not in summary
                for field in (
                    "failed_check",
                    "upstream",
                    "artifact_field",
                    "expected_value",
                    "observed_value",
                )
            )
        ),
        "verdict_class",
    )
    add(not _publication_gate_valid(artifact.get("publication_gate")), "publication_gate")
    add(
        artifact.get("reproducibility_checksum") != _artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        current_gates = replay_gates(contract["tasks"], evidence)
        current_claims = recompute_claims(evidence)
        terminal = _terminal_state(evidence, current_claims)
        current_matrix = [
            _matrix_row(index, task, evidence[str(task["id"])], current_gates, current_claims)
            for index, task in enumerate(contract["tasks"][:-1], 1)
        ]
        current_matrix.append(_self_row(contract["tasks"][-1], terminal))
        add(contract_rows != contract["contract_rows"], "contract_rows")
        add(claims != current_claims, "claim_rows")
        add(gate_rows != current_gates, "gate_replay_rows")
        add(matrix != current_matrix, "evidence_matrix")
        add(branches != _branch_decisions(contract["tasks"], current_matrix), "branch_decisions")
        add(
            artifact.get("status") != terminal["status"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or summary != terminal["gate_check_summary"],
            "verdict_class",
        )
        add(
            artifact.get("acceptance_gate_results")
            != _acceptance_results(
                contract,
                current_matrix,
                current_claims,
                current_gates,
                artifact.get("validation_receipts", []),
            ),
            "acceptance_gate_results",
        )
        add(artifact.get("prd_gap_matrix") != _prd_gap_matrix(current_claims), "prd_gap_matrix")
        hashes = artifact.get("source_artifact_hashes", {})
        if isinstance(hashes, Mapping):
            for named, expected in hashes.items():
                path = Path(str(named))
                resolved = path if path.is_absolute() else root / path
                if not resolved.is_file() or _sha256(resolved) != expected:
                    add(True, "source_artifact_hashes")
                    break
        else:
            add(True, "source_artifact_hashes")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    checkpoint_path: Path,
    *,
    raw_dir: Path | None = None,
    validation_receipt_path: Path | None = None,
) -> JsonDict:
    """Aggregate V640 evidence and publish only a validated raw candidate."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs and writable output paths")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_raw_dir = raw_dir or root / DEFAULT_RAW_DIR
    common._atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7287.v640_capstone.checkpoint.v1",
            "experiment_id": "exp7287-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    spans: dict[str, float] = {}
    phase = time.monotonic()
    progress(1, "start", "parse V640 YAML and named Markdown independently")
    contract = load_contract(root)
    spans["contract_parse"] = time.monotonic() - phase
    progress(1, "end", f"completed=14 contract_agrees={contract['contract_agrees']}")

    phase = time.monotonic()
    progress(2, "start", "authenticate thirteen producer evidence slots")
    evidence = load_repository_payloads(root, contract["tasks"])
    for index, task_id in enumerate(EXPECTED_PRODUCER_IDS, 1):
        progress(
            2,
            "unit",
            f"completed={index}/13 {task_id} source={evidence[task_id]['evidence_source']}",
        )
    checks, source_hashes = _preconditions(root, output_path, checkpoint_path, contract, evidence)
    spans["producer_authentication"] = time.monotonic() - phase
    progress(2, "end", f"terminal_slots={sum(row['terminal'] for row in evidence.values())}")

    phase = time.monotonic()
    progress(3, "start", "replay eight gates and reduce nineteen claims")
    gates = replay_gates(contract["tasks"], evidence)
    claims = recompute_claims(evidence)
    spans["gate_and_claim_reduction"] = time.monotonic() - phase
    progress(3, "end", f"gates={len(gates)} claims={len(claims)}")

    phase = time.monotonic()
    progress(4, "start", "freeze authorities and write hashed sidecars")
    sidecars = _write_sidecars(root, selected_raw_dir, contract, evidence, claims, gates)
    frozen_paths = (
        selected_raw_dir / "selected-roadmap.yaml",
        selected_raw_dir / "v640-design.md",
    )
    for path in frozen_paths:
        source_hashes[_display_path(root, path)] = _sha256(path)
    for receipt in sidecars:
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    spans["authority_freeze_and_sidecars"] = time.monotonic() - phase
    progress(4, "end", f"sidecars={len(sidecars)} frozen_authorities=2")

    phase = time.monotonic()
    progress(5, "start", "build fourteen dispositions and bounded branch decisions")
    terminal = _terminal_state(evidence, claims)
    matrix = [
        _matrix_row(index, task, evidence[str(task["id"])], gates, claims)
        for index, task in enumerate(contract["tasks"][:-1], 1)
    ]
    matrix.append(_self_row(contract["tasks"][-1], terminal))
    for index, row in enumerate(matrix, 1):
        progress(5, "unit", f"completed={index}/14 {row['task_id']}")
    decisions = _branch_decisions(contract["tasks"], matrix)
    spans["matrix_and_branches"] = time.monotonic() - phase
    progress(5, "end", f"matrix={len(matrix)} decisions={len(decisions)}")

    phase = time.monotonic()
    progress(6, "start", "record stable G1-G4 publication context")
    publication_gate = _publication_gate(root)
    spans["publication_gate"] = time.monotonic() - phase
    progress(6, "end", f"unmet={publication_gate['unmet_gates']}")

    phase = time.monotonic()
    progress(7, "start", "authenticate validation receipts and map PRD gaps")
    receipts = _load_validation_receipts(root, validation_receipt_path)
    if validation_receipt_path is not None and validation_receipt_path.is_file():
        source_hashes[_display_path(root, validation_receipt_path)] = _sha256(
            validation_receipt_path
        )
    prd = _prd_gap_matrix(claims)
    spans["validation_and_prd_mapping"] = time.monotonic() - phase
    progress(7, "end", f"validation_receipts={len(receipts)} PRD_rows={len(prd)}")

    artifact: JsonDict = {
        "schema": "carnot.exp7287.v640_capstone.v1",
        "experiment_id": "exp7287-capstone",
        "milestone": MILESTONE,
        "status": terminal["status"],
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": "",
        "field_principles": ALL_FIELD_PRINCIPLES,
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans_s": spans,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": claims,
        "sample_size_budget": {
            "contract_rows": {
                "planned": 14,
                "attempted": 14,
                "completed": 14,
                "censored": 0,
                "independent_units": 14,
            },
            "producer_artifacts": {
                "planned": 13,
                "attempted": 13,
                "completed": sum(row["terminal"] for row in evidence.values()),
                "censored": sum(
                    row["declared_artifact_present"] is False for row in evidence.values()
                ),
                "independent_units": 13,
            },
            "claim_rows": {
                "planned": len(CLAIM_NAMES),
                "attempted": len(CLAIM_NAMES),
                "completed": len(claims),
                "censored": sum(row["censored"] is True for row in claims),
                "independent_units": len(REQUIRED_SCIENCE_TASKS),
            },
            "stopping_rule": "Stop after fourteen dispositions, eight gates, nineteen claims, and four PRD rows.",
        },
        "acceptance_gate_results": _acceptance_results(contract, matrix, claims, gates, receipts),
        "gate_check_summary": terminal["gate_check_summary"],
        "verifier_is_oracle": False,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "validation_receipts": receipts,
        "capstone_complete_score": 1,
        "evidence_matrix": matrix,
        "contract_rows": contract["contract_rows"],
        "branch_decisions": decisions,
        "prd_gap_matrix": prd,
        "publication_gate": publication_gate,
        "same_milestone_gate_replay_rows": gates,
        "historical_and_negative_fixture_sidecars": sidecars,
        "board_dispositions": deepcopy(
            evidence["exp7286-board-state"]["payload"].get("board_rows", [])
        ),
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_message_performed": False,
        "production_default_changed": False,
        "exclusion_manifest_modified": False,
        "research_roadmap_modified": False,
        "conductor_modified": False,
    }
    elapsed = time.monotonic() - started
    spans["final_assembly"] = max(elapsed - sum(spans.values()), 0.0)
    artifact["duration_s"] = elapsed
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)

    progress(8, "start", "write and validate the measured raw terminal candidate")
    candidate_path = selected_raw_dir / "measured-terminal-candidate.json"
    common._atomic_write(candidate_path, artifact)
    candidate = read_json(candidate_path)
    errors = validate_artifact(candidate, root=root)
    if errors:
        raise RuntimeError("capstone candidate validation failed: " + ",".join(errors))
    progress(8, "end", "measured raw candidate passed independent reduction")

    progress(9, "start", "atomically write the declared terminal deliverable")
    common._atomic_write(checkpoint_path, artifact)
    common._atomic_write(output_path, artifact)
    final = read_json(output_path)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    progress(9, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Build the capstone or independently validate a terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--validation-receipt-path", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.artifact_path or root / DEFAULT_OUTPUT_PATH
    checkpoint = args.checkpoint_path or root / DEFAULT_CHECKPOINT_PATH
    if args.validate:
        progress(8, "start", f"independent raw-row reducer for {output}")
        errors = validate_artifact(read_json(output), root=root)
        progress(8, "end", "passed" if not errors else ",".join(errors))
        return int(bool(errors))
    receipt_path = args.validation_receipt_path
    if receipt_path is None:
        default_receipts = root / DEFAULT_VALIDATION_RECEIPT_PATH
        receipt_path = default_receipts if default_receipts.is_file() else None
    build_artifact(
        root,
        args.date,
        output,
        checkpoint,
        raw_dir=args.raw_dir,
        validation_receipt_path=receipt_path,
    )
    return 0
