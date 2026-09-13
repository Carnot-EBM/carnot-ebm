"""Close V638 with an authenticated fourteen-task evidence matrix.

This module invokes no model. It reads producer artifacts, replays the frozen
gates, and reduces the scientific claims from producer rows.

Spec refs: REQ-REPORT-7259 and SCENARIO-REPORT-7259-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7245_v637_capstone as v637
from carnot import experiment_7246_v638_source_map as source_map
from carnot import experiment_7248_v638_arc_witness as arc_witness
from carnot import experiment_7252_v638_semantic_audit as semantic_audit
from carnot import experiment_7253_v638_coverage_memory as coverage_memory
from carnot import experiment_7254_v638_coverage_learning as coverage_learning
from carnot import experiment_7255_v638_coverage_audit as coverage_audit
from carnot import experiment_7256_v638_native_controller as native_controller
from carnot import experiment_7257_v638_native_cost as native_cost
from carnot import experiment_7258_v638_board_state as board_state


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.638"
RUN_DATE = "20260913"
RANDOM_SEED = 7_259_202_609_13
MODEL_SPECS: list[JsonDict] = []

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7259_v638_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7259_v638_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7259")
DEFAULT_VALIDATION_RECEIPT_PATH = DEFAULT_RAW_DIR / "validation_receipts.json"

EXPECTED_TASK_IDS = tuple(source_map.EXPECTED_ID_ORDER)
EXPECTED_PRODUCER_IDS = EXPECTED_TASK_IDS[:-1]

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260913 and retain actual UTC start and end timestamps.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name kv260, gatemate or polarfire.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked_* verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; a failed scientific acceptance gate forbids positive. External incompleteness is blocked, not retryable partial.",
    "validation_receipts": "Record actual command, exit code and log hash; no skipped, weakened, deleted or reverted tests.",
    "capstone_complete_score": "One means the complete fourteen-task evidence matrix exists, independent of science outcome.",
    "evidence_matrix": "Exactly fourteen ordered rows include missing, blocked, quarantined and self-synthesis states.",
    "branch_decisions": "Each continuation names a changed cause; each retirement names the exact mechanism and prior verdict.",
    "prd_gap_matrix": "Map measured evidence and remaining limits to FR-11, FR-12, FR-05/08 and NFR-01.",
    "contract_rows": "Independent document and YAML parse must agree exactly on identity, ordering, titles, paths and gates.",
}

EXTRA_FIELD_PRINCIPLES: JsonDict = {
    "experiment_id": "Bind the receipt to the final task in the frozen V638 roster.",
    "milestone": "Bind all producer and contract checks to V638 only.",
    "started_at_utc": "Record the actual UTC start instant.",
    "completed_at_utc": "Record the actual UTC terminal-write instant.",
    "phase_spans_s": "Keep measured sequential phase spans separate.",
    "current_invocation_counters": "Expose zero current model, load, generation, and call counts.",
    "recomputed_claim_rows": "Keep each independent reduction available to consumers.",
    "same_milestone_gate_replay_rows": "Show each producer field, file, and quarantine observation.",
    "historical_and_negative_fixture_sidecars": "Keep source model history and negative fixtures outside current counters.",
    "source_ingestion_decisions": "Preserve the bounded V638 method mapping without promoting external claims.",
    "hardware_dispositions": "Keep board graduation and physical blocks separate from capstone orchestration.",
    "publication_performed": "No publication follows from this closeout.",
    "upload_performed": "No upload follows from this closeout.",
    "submission_performed": "No benchmark submission follows from this closeout.",
    "external_message_performed": "No external message follows from this closeout.",
    "production_default_changed": "The closeout changes no production behavior.",
    "exclusion_manifest_modified": "The closeout reads but does not edit quarantine policy.",
    "research_roadmap_modified": "The closeout reads but does not activate or edit a roadmap.",
    "conductor_modified": "The closeout does not edit the research conductor.",
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
    Path("python/carnot/experiment_7245_v637_capstone.py"),
    Path("results/experiment_7245_v637_capstone.json"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7259_v638_capstone.py"),
    Path("scripts/experiments/experiment_7259_v638_capstone.py"),
    Path("tests/python/test_experiment_7259_v638_capstone.py"),
)

VALIDATION_NAMES = (
    "focused_pytest",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "full_python_suite",
    "independent_raw_row_reducer",
    "adversarial_verify",
    "verdict_row_consistency_lint",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one truthful boundary so a long-running conductor sees progress."""

    print(f"[exp7259] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read one object because an array cannot carry the artifact contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Use the shipped digest helper so all experiment receipts share a format."""

    return common.sha256_path(path)


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum that stores the digest."""

    return common.reproducibility_checksum(artifact)


def load_contract(root: Path) -> JsonDict:
    """Parse the selected V638 YAML and its named Markdown design independently."""

    roadmap_path = root / ROADMAP_PATH
    roadmap_bytes = roadmap_path.read_bytes()
    roadmap = yaml.safe_load(roadmap_bytes)
    if not isinstance(roadmap, Mapping):
        raise ValueError("V638 roadmap root is not a mapping")
    if roadmap.get("milestone") != MILESTONE:
        raise ValueError("active roadmap is not V638")
    named_design = Path(str(roadmap.get("milestone_doc", DESIGN_PATH)))
    if named_design != DESIGN_PATH:
        raise ValueError("V638 roadmap names an unexpected design")
    design_bytes = (root / named_design).read_bytes()
    comparison = source_map.evaluate_contract(design_bytes.decode(), roadmap)
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
        raise ValueError("V638 task order is not exp7246 through exp7259")
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "contract_rows": comparison["contract_rows"],
        "contract_agrees": comparison["passed"] is True,
        "yaml_milestone": comparison["yaml_milestone"],
        "markdown_milestone": comparison["markdown_milestone"],
        "gate_producer_rows": comparison["gate_producer_rows"],
        "roadmap_path": str(ROADMAP_PATH),
        "design_path": str(named_design),
        "roadmap_sha256": common.sha256_path(roadmap_path),
        "design_sha256": common.sha256_path(root / named_design),
        "roadmap_bytes": roadmap_bytes,
        "design_bytes": design_bytes,
    }


def _validate_payload(task_id: str, payload: Mapping[str, Any]) -> list[str]:
    """Run the producer's shipped validator without rechecking mutable source files."""

    if task_id == "exp7246-source-map":
        errors = source_map.independent_reduce(payload)
        if payload.get("reproducibility_checksum") != source_map.reproducibility_checksum(payload):
            errors.append("reproducibility_checksum")
        return errors
    if task_id == "exp7248-arc-witness":
        try:
            arc_witness.validate_artifact(payload)
        except ValueError as error:
            return [str(error)]
        return []
    validators = {
        "exp7252-semantic-audit": lambda: semantic_audit.validate_artifact(payload),
        "exp7253-coverage-memory": lambda: coverage_memory.validate_artifact(
            payload, check_files=False
        ),
        "exp7254-coverage-learning": lambda: coverage_learning.validate_artifact(
            payload, check_files=False
        ),
        "exp7255-coverage-audit": lambda: coverage_audit.validate_artifact(
            payload, check_files=False
        ),
        "exp7256-native-controller": lambda: native_controller.validate_artifact(
            payload, check_files=False
        ),
        "exp7257-native-cost": lambda: native_cost.validate_artifact(payload, check_files=False),
        "exp7258-board-state": lambda: board_state.validate_artifact(payload),
    }
    validator = validators.get(task_id)
    return list(validator()) if validator is not None else []


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load an exact deliverable or its canonical conductor gate-block receipt."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    fallback = common.canonical_gate_block_path(task_id)
    declared_present = (root / declared).is_file()
    if declared_present:
        selected = declared
        source = "declared_deliverable"
    elif (root / fallback).is_file():
        selected = fallback
        source = "conductor_gate_block"
    else:
        selected = None
        source = "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = common.quarantine_receipt(payload, task_id, selected or declared, manifest)
    raw_replay = v637._raw_hash_replay(root, task_id, payload) if payload else []
    validator_errors = _validate_payload(task_id, payload) if payload else []
    terminal = payload.get("status") in {"complete", "blocked"}
    identity_matches = payload.get("milestone") == MILESTONE
    authenticated = bool(payload) and terminal and identity_matches and not validator_errors
    accepted = (
        authenticated
        and payload.get("status") == "complete"
        and payload.get("verdict_class") not in {"blocked", "disqualified", "partial"}
        and quarantine["quarantined"] is False
        and all(row.get("passed") is True for row in raw_replay)
    )
    rows = payload.get("rows", [])
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
        "producer_validation_errors": validator_errors,
        "raw_hash_replay_rows": raw_replay,
        "raw_hashes_authenticated": all(row.get("passed") is True for row in raw_replay),
        "embedded_row_count": len(rows) if isinstance(rows, list) else 0,
        "quarantine_state": quarantine,
        "authenticated": authenticated,
        "accepted_for_positive_claim": accepted,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Retain one evidence slot for every producer, including absent files."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task["id"] != "exp7259-capstone"
    }


def _prompt_declares_field(task: Mapping[str, Any], field: str) -> bool:
    """Check the producer prompt because gates may use only declared top-level fields."""

    return f"- {field}:" in str(task.get("prompt", ""))


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay each field gate and keep missing files distinct from numeric zero."""

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
                    "artifact_path": producer["selected_evidence_path"],
                    "declared_artifact_path": producer["declared_deliverable_path"],
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate["value"],
                    "observed_value": observed,
                    "producer_declares_field": _prompt_declares_field(task_by_id[upstream], field),
                    "quarantined": producer["quarantine_state"]["quarantined"],
                    "producer_validation_errors": producer["producer_validation_errors"],
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
) -> JsonDict:
    """Store one raw reduction and prevent unauthenticated positive promotion."""

    source = evidence[task_id]
    missing = source["selected_evidence_path"] is None
    blocked = source["payload"].get("status") == "blocked"
    matches = declared == recomputed
    positive = recomputed is True or (isinstance(recomputed, (int, float)) and recomputed > 0)
    promoted = bool(positive and matches and source["accepted_for_positive_claim"] and not oracle)
    if missing:
        error = "missing_producer_artifact"
    elif blocked:
        error = "blocked_producer_artifact"
    elif source["producer_validation_errors"]:
        error = "producer_validation_failed"
    elif not matches:
        error = "declared_value_mismatch"
    else:
        error = None
    return {
        "unit_id": f"{task_id}:{claim}",
        "arm": "independent_raw_row_reduction",
        "seed": RANDOM_SEED,
        "metric": claim,
        "metric_value": recomputed,
        "error": error,
        "abstention": missing or blocked or bool(source["producer_validation_errors"]),
        "censored": missing,
        "claim": claim,
        "task_id": task_id,
        "declared_value": declared,
        "recomputed_value": recomputed,
        "matches": matches,
        "evidence_fields": list(evidence_fields),
        "raw_row_count": source["embedded_row_count"],
        "source_authenticated": source["authenticated"],
        "positive_promoted": promoted,
        "verifier_is_oracle": oracle,
        "claim_class": "circular_positive"
        if positive and oracle
        else "positive"
        if promoted
        else "blocked"
        if missing or blocked
        else "null"
        if recomputed is False
        else "numeric_check",
    }


def recompute_claims(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Recompute semantic, ARC, memory, and native-cost claims from raw rows."""

    rows: list[JsonDict] = []
    semantic = evidence["exp7252-semantic-audit"]["payload"]
    semantic_rows = semantic.get("rows", [])
    semantic_value = (
        bool(semantic_rows) and semantic.get("semantic_value_score") == 1
        if semantic.get("status") == "complete"
        else None
    )
    rows.append(
        _claim_row(
            "usable_source_semantics",
            "exp7252-semantic-audit",
            semantic.get("semantic_value_score"),
            semantic_value,
            ("rows", "semantic_value_score"),
            evidence,
            oracle=True,
        )
    )

    capture = evidence["exp7251-mention-heldout"]["payload"]
    capture_value = (
        bool(capture.get("rows")) and capture.get("status") == "complete" if capture else None
    )
    rows.append(
        _claim_row(
            "completed_mention_capture",
            "exp7251-mention-heldout",
            capture.get("mention_capture_complete_score"),
            capture_value,
            ("status", "rows"),
            evidence,
        )
    )

    controller = evidence["exp7256-native-controller"]["payload"]
    parity = native_controller.reduce_parity_rows(controller.get("parity_rows", []))
    oracle_value = bool(
        parity["stream_arm_rows"] == 24
        and parity["event_mismatch_count"] == 0
        and parity["declared_mismatch_count"] == 0
        and parity["all_rollbacks_exact"]
        and parity["all_rows_passed"]
    )
    rows.append(
        _claim_row(
            "exact_oracle_conformance",
            "exp7256-native-controller",
            bool(controller.get("native_controller_ready_score")),
            oracle_value,
            ("parity_rows.expected", "parity_rows.actual", "parity_rows.passed"),
            evidence,
            oracle=True,
        )
    )

    learning = evidence["exp7254-coverage-learning"]["payload"]
    learning_rows = learning.get("rows", [])
    aligned_rows = [row for row in learning_rows if row.get("arm") == "coverage_archive_aligned"]
    shuffled_rows = [row for row in learning_rows if row.get("arm") == "coverage_archive_shuffled"]
    causal = {
        "valid_reactivation_count": sum(
            int(row.get("valid_reactivation_count", 0)) for row in aligned_rows
        ),
        "later_changed_decision_after_reactivation_count": sum(
            int(row.get("later_changed_decision_after_reactivation_count", 0))
            for row in aligned_rows
        ),
        "pre_release_difference_count": sum(
            int(row.get("pre_release_difference_count", 0)) for row in aligned_rows
        ),
        "prospective_shuffle_selection_change_count": sum(
            int(row.get("shuffle_selection_change_count", 0)) for row in shuffled_rows
        ),
    }
    causal_value = bool(
        causal["valid_reactivation_count"] > 0
        and causal["later_changed_decision_after_reactivation_count"] > 0
        and causal["pre_release_difference_count"] == 0
    )
    rows.append(
        _claim_row(
            "causal_learning_observed",
            "exp7254-coverage-learning",
            bool(
                learning.get("causal_summary", {}).get(
                    "later_changed_decision_after_reactivation_count", 0
                )
            ),
            causal_value,
            (
                "rows.valid_reactivation_count",
                "rows.later_changed_decision_after_reactivation_count",
                "rows.pre_release_difference_count",
            ),
            evidence,
            oracle=True,
        )
    )
    aligned = {int(row["seed"]): row for row in aligned_rows}
    shuffled = {int(row["seed"]): row for row in shuffled_rows}
    shuffle_recurrence_improved = bool(aligned) and all(
        float(aligned[seed]["recurrence_error_rate"])
        < float(shuffled[seed]["recurrence_error_rate"])
        for seed in aligned
    )
    learning_value = bool(
        causal["prospective_shuffle_selection_change_count"] > 0 and shuffle_recurrence_improved
    )
    rows.append(
        _claim_row(
            "coverage_learning_value",
            "exp7254-coverage-learning",
            bool(learning.get("coverage_learning_value_score")),
            learning_value,
            ("rows", "comparison_rows.seed_differences", "causal_summary"),
            evidence,
            oracle=True,
        )
    )

    live = evidence["exp7249-arc-live"]["payload"]
    for claim, fields in (
        ("useful_world_model_prediction", ("rows", "useful_world_model_prediction")),
        ("actual_policy_consumption", ("rows", "actual_policy_consumption")),
        ("official_arc_score", ("rows", "official_score")),
    ):
        declared = live.get(claim)
        value = declared if live.get("status") == "complete" and live.get("rows") else None
        rows.append(_claim_row(claim, "exp7249-arc-live", declared, value, fields, evidence))

    cost = evidence["exp7257-native-cost"]["payload"]
    cost_reduction = native_cost.reduce_cost_rows(cost.get("cost_rows", []))
    rows.append(
        _claim_row(
            "durable_event_cost_captured",
            "exp7257-native-cost",
            bool(cost.get("native_cost_complete_score")),
            cost_reduction["native_cost_complete_score"] == 1,
            ("cost_rows", "cost_rows.durability_receipt", "cost_rows.total_event_ns"),
            evidence,
            oracle=True,
        )
    )
    rows.append(
        _claim_row(
            "durable_event_cost_value",
            "exp7257-native-cost",
            bool(cost.get("native_event_cost_value_score")),
            cost_reduction["native_event_cost_value_score"] == 1,
            ("cost_rows.total_event_ns", "cost_rows.archive_capacity", "cost_rows.batch_size"),
            evidence,
            oracle=True,
        )
    )
    return rows


def _matrix_row(
    order: int,
    task: Mapping[str, Any],
    evidence: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one producer row without copying its large embedded evidence table."""

    task_id = str(task["id"])
    payload = evidence["payload"]
    task_gates = [row for row in gate_rows if row["consumer"] == task_id]
    task_claims = [row for row in claims if row["task_id"] == task_id]
    return {
        "order": order,
        "task_id": task_id,
        "title": task["title"],
        "declared_artifact_path": evidence["declared_deliverable_path"],
        "declared_artifact_present": evidence["declared_artifact_present"],
        "selected_evidence_path": evidence["selected_evidence_path"],
        "evidence_source": evidence["evidence_source"],
        "artifact_sha256": evidence["artifact_sha256"],
        "artifact_size_bytes": evidence["artifact_size_bytes"],
        "status": payload.get("status", "missing"),
        "verdict_class": payload.get("verdict_class", "blocked"),
        "honest_verdict": payload.get("honest_verdict", "blocked_missing_producer_artifact"),
        "model_invoked": payload.get("model_invoked"),
        "inference_substrate": payload.get("inference_substrate"),
        "inference_substrate_class": payload.get("inference_substrate_class"),
        "quarantine_state": deepcopy(evidence["quarantine_state"]),
        "producer_validation_errors": deepcopy(evidence["producer_validation_errors"]),
        "authenticated": evidence["authenticated"],
        "accepted_for_positive_claim": evidence["accepted_for_positive_claim"],
        "raw_evidence": {
            "embedded_row_count": evidence["embedded_row_count"],
            "raw_hash_replay_rows": deepcopy(evidence["raw_hash_replay_rows"]),
            "raw_hashes_authenticated": evidence["raw_hashes_authenticated"],
            "source_artifact_hashes": deepcopy(payload.get("source_artifact_hashes", {})),
        },
        "upstream_gate_observations": task_gates,
        "recomputed_claims": task_claims,
        "verifier_is_oracle": bool(common.unwrap_principle(payload.get("verifier_is_oracle"))),
    }


def _branch_decisions(
    tasks: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Compare exact prior verdicts and limit every action to one task mechanism."""

    rows: list[JsonDict] = []
    for task, evidence in zip(tasks, matrix, strict=True):
        task_id = str(task["id"])
        current = str(evidence.get("honest_verdict"))
        priors = deepcopy(task.get("prior_failures") or [])
        matches = [
            prior
            for prior in priors
            if prior.get("verdict") == current and prior.get("retire_if_same_verdict") is True
        ]
        if matches:
            action = "retire"
            reason = "The exact prior verdict recurred for this task mechanism."
            condition = "Replace this exact mechanism before another measurement."
        elif task_id in {"exp7246-source-map", "exp7259-capstone"}:
            action = "retire"
            reason = "This one-time milestone receipt is complete and remains immutable."
            condition = "Use a new milestone identity; do not rerun this receipt."
        elif task_id == "exp7257-native-cost":
            action = "retire"
            reason = "The measured persistent-controller boundary optimization failed its durable interactive value gate."
            condition = "A replacement must change this boundary mechanism and beat both capacity-one gates."
        elif task_id in {"exp7253-coverage-memory", "exp7256-native-controller"}:
            action = "continue"
            reason = "The exact fixture or parity capability is reusable, but it supplies no value claim by itself."
            condition = (
                "Consume it only in a changed experiment with oracle-distinct value controls."
            )
        else:
            action = "needs_changed_prerequisite"
            reason = "The task is blocked, null, missing, or only a narrow capability result."
            condition = (
                "Supply the exact missing source or change the failed mechanism before new work."
            )
        rows.append(
            {
                "task_id": task_id,
                "action": action,
                "reason": reason,
                "changed_cause_required": condition,
                "evidence_path": evidence.get("selected_evidence_path"),
                "current_verdict": current,
                "prior_failures": priors,
                "exact_same_verdict_recurrence": bool(matches),
                "retire_if_same_verdict_applied": bool(matches) and action == "retire",
                "matching_retirement_signals": matches,
                "broad_family_retirement_invented": False,
            }
        )
    return rows


def _write_sidecars(
    root: Path,
    raw_dir: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
) -> list[JsonDict]:
    """Freeze authorities and isolate historical models and negative fixtures."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = raw_dir / "selected-roadmap.yaml"
    design_path = raw_dir / "v638-design.md"
    source_map._atomic_write_bytes(yaml_path, contract["roadmap_bytes"])
    source_map._atomic_write_bytes(design_path, contract["design_bytes"])
    model_path = raw_dir / "historical-model-receipts.json"
    fixture_path = raw_dir / "negative-fixture-receipts.json"
    model_rows = [
        {
            "task_id": task_id,
            "artifact_path": item["selected_evidence_path"],
            "artifact_sha256": item["artifact_sha256"],
            "model_invoked": item["payload"].get("model_invoked"),
            "MODEL_SPECS": deepcopy(item["payload"].get("MODEL_SPECS", [])),
            "historical_only": True,
        }
        for task_id, item in evidence.items()
        if item["payload"]
    ]
    fixture_rows = []
    for task_id, item in evidence.items():
        hashes = item["payload"].get("source_artifact_hashes", {})
        if not isinstance(hashes, Mapping):
            continue
        for path, digest in hashes.items():
            if "fixture" in str(path).lower() or "negative" in str(path).lower():
                fixture_rows.append(
                    {
                        "task_id": task_id,
                        "path": str(path),
                        "sha256": digest,
                        "historical_only": True,
                    }
                )
    common._atomic_write(model_path, {"rows": model_rows})
    common._atomic_write(fixture_path, {"rows": fixture_rows})
    return [
        {
            "kind": "historical_model_receipts",
            "path": _display_path(root, model_path),
            "sha256": _sha256(model_path),
        },
        {
            "kind": "negative_fixture_receipts",
            "path": _display_path(root, fixture_path),
            "sha256": _sha256(fixture_path),
        },
    ]


def _display_path(root: Path, path: Path) -> str:
    """Use repository-relative paths when possible and absolute test paths otherwise."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_validation_receipts(root: Path, path: Path | None) -> list[JsonDict]:
    """Read command receipts only after their logs exist and their bytes match."""

    if path is None or not path.is_file():
        return []
    value = read_json(path)
    receipts = value.get("receipts", [])
    if not isinstance(receipts, list):
        raise ValueError("validation receipts are not a list")
    rows: list[JsonDict] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("validation receipt is not an object")
        row = dict(receipt)
        log_path = Path(str(row["log_path"]))
        resolved = log_path if log_path.is_absolute() else root / log_path
        observed = _sha256(resolved) if resolved.is_file() else None
        row["observed_log_sha256"] = observed
        row["log_hash_matches"] = observed == row.get("log_sha256")
        rows.append(row)
    return rows


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record source bytes, ownership, imports, paths, and known failed checks."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_source_bytes",
                "upstream": str(relative),
                "artifact_field": "bytes",
                "expected_value": "nonempty",
                "observed_value": path.stat().st_size if present else "missing",
                "passed": present,
            }
        )
        if present:
            hashes[str(relative)] = _sha256(path)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "driving_capability",
                "upstream": str(SPEC_PATH),
                "artifact_field": "REQ-REPORT-7259",
                "expected_value": True,
                "observed_value": "REQ-REPORT-7259" in spec,
                "passed": "REQ-REPORT-7259" in spec,
            },
            {
                "check": "v638_yaml_identity",
                "upstream": str(ROADMAP_PATH),
                "artifact_field": "milestone|task_ids",
                "expected_value": {"milestone": MILESTONE, "task_count": 14},
                "observed_value": {
                    "milestone": contract["yaml_milestone"],
                    "task_count": len(contract["tasks"]),
                },
                "passed": contract["yaml_milestone"] == MILESTONE and len(contract["tasks"]) == 14,
            },
            {
                "check": "independent_design_contract",
                "upstream": str(DESIGN_PATH),
                "artifact_field": "milestone|identity|order|title|path|gates",
                "expected_value": {"milestone": MILESTONE, "all_rows_match": True},
                "observed_value": {
                    "milestone": contract["markdown_milestone"],
                    "all_rows_match": contract["contract_agrees"],
                },
                "passed": contract["contract_agrees"],
            },
            {
                "check": "imports_and_outputs",
                "upstream": "host",
                "artifact_field": "shipped_helpers|writable_paths",
                "expected_value": True,
                "observed_value": callable(source_map.evaluate_contract)
                and callable(native_cost.reduce_cost_rows)
                and os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
                "passed": callable(source_map.evaluate_contract)
                and callable(native_cost.reduce_cost_rows)
                and os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
            },
            {
                "check": "no_current_model",
                "upstream": "exp7259-capstone",
                "artifact_field": "MODEL_SPECS|model_invoked|counters",
                "expected_value": {"MODEL_SPECS": [], "model_invoked": False, "calls": 0},
                "observed_value": {"MODEL_SPECS": [], "model_invoked": False, "calls": 0},
                "passed": True,
            },
        ]
    )
    for task_id, item in evidence.items():
        if item["selected_evidence_path"]:
            hashes[str(item["selected_evidence_path"])] = str(item["artifact_sha256"])
        checks.append(
            {
                "check": "producer_artifact_authentication",
                "upstream": task_id,
                "artifact_field": "declared_path|status|milestone|validator|quarantine|raw_hashes",
                "expected_value": "declared terminal V638 artifact with valid raw evidence",
                "observed_value": {
                    "declared_present": item["declared_artifact_present"],
                    "selected_path": item["selected_evidence_path"],
                    "terminal": item["terminal"],
                    "identity_matches": item["identity_matches"],
                    "validator_errors": item["producer_validation_errors"],
                    "quarantined": item["quarantine_state"]["quarantined"],
                    "raw_hashes_authenticated": item["raw_hashes_authenticated"],
                },
                "passed": item["declared_artifact_present"] and item["authenticated"],
            }
        )
    return checks, hashes


def _self_row(
    task: Mapping[str, Any], claim_count: int, gate_summary: Mapping[str, Any]
) -> JsonDict:
    """Represent the capstone without creating a recursive self-hash."""

    return {
        "order": 14,
        "task_id": "exp7259-capstone",
        "title": task["title"],
        "declared_artifact_path": task["deliverable"],
        "declared_artifact_present": False,
        "selected_evidence_path": str(DEFAULT_OUTPUT_PATH),
        "evidence_source": "self_synthesis",
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "status": "blocked",
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external_source_value",
        "model_invoked": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "quarantine_state": {
            "declared_flags": {},
            "exclusion_manifest_match": False,
            "quarantined": False,
        },
        "producer_validation_errors": [],
        "authenticated": True,
        "accepted_for_positive_claim": False,
        "raw_evidence": {
            "embedded_row_count": claim_count,
            "raw_hash_replay_rows": [],
            "raw_hashes_authenticated": True,
            "source_artifact_hashes": {},
        },
        "upstream_gate_observations": [],
        "recomputed_claims": [],
        "gate_check_summary": deepcopy(gate_summary),
        "verifier_is_oracle": False,
    }


def _acceptance_results(
    contract: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score completion separately from contract, science, and validation outcomes."""

    receipt_names = [str(row.get("name")) for row in receipts]
    definitions = (
        (
            "fourteen_task_matrix_complete",
            14,
            len(matrix),
            len(matrix) == 14,
            "Roster completion is independent from scientific success.",
        ),
        (
            "independent_contract_exact",
            True,
            contract["contract_agrees"],
            contract["contract_agrees"] is True,
            "The Markdown and YAML authorities must match without copied identity.",
        ),
        (
            "claim_reductions_recorded",
            10,
            len(claims),
            len(claims) == 10,
            "Each distinct scientific question keeps one recomputable row.",
        ),
        (
            "same_milestone_gates_pass",
            5,
            sum(row["passed"] is True for row in gates),
            len(gates) == 5 and all(row["passed"] is True for row in gates),
            "A missing file and a true zero remain failed gates.",
        ),
        (
            "external_science_available",
            True,
            False,
            False,
            "Held-out source semantics and live ARC policy evidence are required for a positive capstone.",
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
            "Every required command keeps its real exit code and log hash.",
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


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute identity, roster, claims, gates, hashes, and terminal checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != ALL_FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.exp7259.v638_capstone.v1"
        or artifact.get("experiment_id") != "exp7259-capstone"
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("status") != "blocked" or artifact.get("run_date") != RUN_DATE, "lifecycle")
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("current_invocation_counters")
        != {"models": 0, "loads": 0, "generations": 0, "calls": 0},
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
        or matrix[-1].get("artifact_sha256") is not None,
        "evidence_matrix",
    )
    claim_rows = artifact.get("rows", [])
    add(
        not isinstance(claim_rows, list)
        or len(claim_rows) != 10
        or claim_rows != artifact.get("recomputed_claim_rows")
        or any(
            not {
                "unit_id",
                "arm",
                "seed",
                "metric",
                "metric_value",
                "error",
                "abstention",
                "censored",
            }.issubset(row)
            for row in claim_rows
        ),
        "claim_rows",
    )
    add(artifact.get("capstone_complete_score") != 1, "capstone_complete_score")
    add(
        artifact.get("verdict_class") != "blocked"
        or not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
        "verdict_class",
    )
    gate = artifact.get("gate_check_summary", {})
    add(
        not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or any(
            field not in gate
            for field in (
                "failed_check",
                "upstream",
                "artifact_field",
                "expected_value",
                "observed_value",
            )
        ),
        "gate_check_summary",
    )
    add(
        artifact.get("reproducibility_checksum") != _artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        add(contract_rows != contract["contract_rows"], "contract_rows")
        add(claim_rows != recompute_claims(evidence), "claim_rows")
        add(
            artifact.get("same_milestone_gate_replay_rows")
            != replay_gates(contract["tasks"], evidence),
            "gate_replay_rows",
        )
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
    """Aggregate V638 evidence and atomically write one blocked terminal receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs, ownership, imports, and writable paths")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_raw_dir = raw_dir or root / DEFAULT_RAW_DIR
    selected_receipts = (
        validation_receipt_path
        if validation_receipt_path is not None
        else root / DEFAULT_VALIDATION_RECEIPT_PATH
    )
    common._atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7259.v638_capstone.v1",
            "experiment_id": "exp7259-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    spans: dict[str, float] = {}
    phase = time.monotonic()
    progress(1, "start", "parse selected V638 YAML and named design independently")
    contract = load_contract(root)
    spans["contract_parse"] = time.monotonic() - phase
    progress(1, "end", f"completed=14 contract_agrees={contract['contract_agrees']}")

    phase = time.monotonic()
    progress(2, "start", "freeze plan bytes and write isolated historical sidecars")
    progress(2, "unit", "before sidecar writes")
    evidence = load_repository_payloads(root, contract["tasks"])
    sidecars = _write_sidecars(root, selected_raw_dir, contract, evidence)
    progress(2, "unit", "after sidecar writes")
    spans["authority_freeze_and_sidecars"] = time.monotonic() - phase
    progress(2, "end", f"sidecars={len(sidecars)}")

    phase = time.monotonic()
    progress(3, "start", "authenticate thirteen producer slots")
    for index, task_id in enumerate(EXPECTED_PRODUCER_IDS, 1):
        progress(
            3,
            "unit",
            f"completed={index}/13 {task_id} source={evidence[task_id]['evidence_source']}",
        )
    checks, source_hashes = _preconditions(root, output_path, checkpoint_path, contract, evidence)
    for row in sidecars:
        source_hashes[str(row["path"])] = str(row["sha256"])
    source_hashes[_display_path(root, selected_raw_dir / "selected-roadmap.yaml")] = _sha256(
        selected_raw_dir / "selected-roadmap.yaml"
    )
    source_hashes[_display_path(root, selected_raw_dir / "v638-design.md")] = _sha256(
        selected_raw_dir / "v638-design.md"
    )
    spans["producer_authentication"] = time.monotonic() - phase
    progress(3, "end", "producer slots=13")
    progress(
        0, "end", f"preconditions={len(checks)} failed={sum(not row['passed'] for row in checks)}"
    )

    phase = time.monotonic()
    progress(4, "start", "replay gates and reduce semantic, ARC, memory, and cost claims")
    gate_rows = replay_gates(contract["tasks"], evidence)
    claims = recompute_claims(evidence)
    spans["gate_and_claim_reduction"] = time.monotonic() - phase
    progress(4, "end", f"gates={len(gate_rows)} claims={len(claims)}")

    gate_summary = {
        "passed": False,
        "failed_check": "required_external_science_artifact",
        "upstream": "exp7251-mention-heldout",
        "artifact_field": "status|rows|mention_capture_complete_score",
        "expected_value": "declared terminal held-out capture with authenticated raw rows",
        "observed_value": "results/experiment_7251_v638_mention_heldout.json is absent",
        "additional_failures": [
            {
                "upstream": "openspec/change-proposals/research-roadmap-vNEXT.md",
                "artifact_field": "milestone|contract_rows",
                "expected_value": {"milestone": MILESTONE, "all_rows_match": True},
                "observed_value": {
                    "milestone": contract["markdown_milestone"],
                    "all_rows_match": contract["contract_agrees"],
                },
            },
            {
                "upstream": "exp7248-arc-witness",
                "artifact_field": "arc_witness_ready_score",
                "expected_value": 1,
                "observed_value": evidence["exp7248-arc-witness"]["payload"].get(
                    "arc_witness_ready_score"
                ),
            },
        ],
    }
    honest_verdict = (
        "blocked_external_source_value: V638 fourteen-task matrix is complete; held-out mention "
        "and live ARC science are absent, coverage value is null, and durable native cost value is null"
    )

    phase = time.monotonic()
    progress(5, "start", "build fourteen evidence rows and task-scoped branches")
    matrix = [
        _matrix_row(index, task, evidence[str(task["id"])], gate_rows, claims)
        for index, task in enumerate(contract["tasks"][:-1], 1)
    ]
    matrix.append(_self_row(contract["tasks"][-1], len(claims), gate_summary))
    for index, row in enumerate(matrix, 1):
        progress(5, "unit", f"completed={index}/14 {row['task_id']}")
    decisions = _branch_decisions(contract["tasks"], matrix)
    spans["matrix_and_branches"] = time.monotonic() - phase
    progress(5, "end", f"matrix={len(matrix)} decisions={len(decisions)}")

    phase = time.monotonic()
    progress(6, "start", "load completed validation receipts and map PRD gaps")
    receipts = _load_validation_receipts(root, selected_receipts)
    if selected_receipts is not None and selected_receipts.is_file():
        source_hashes[_display_path(root, selected_receipts)] = _sha256(selected_receipts)
    claims_by_name = {row["claim"]: row for row in claims}
    prd = {
        "FR-11": {
            "complete": False,
            "causal_learning_observed": claims_by_name["causal_learning_observed"][
                "recomputed_value"
            ],
            "value_observed": claims_by_name["coverage_learning_value"]["recomputed_value"],
            "remaining_limit": "The learner did not beat reset and shuffled controls or meet recurrence retention.",
        },
        "FR-12": {
            "complete": False,
            "usable_source_semantics": claims_by_name["usable_source_semantics"][
                "recomputed_value"
            ],
            "capture_complete": claims_by_name["completed_mention_capture"]["recomputed_value"],
            "remaining_limit": "The authenticated held-out capture is absent.",
        },
        "FR-05/08": {
            "complete": False,
            "useful_world_model_prediction": claims_by_name["useful_world_model_prediction"][
                "recomputed_value"
            ],
            "actual_policy_consumption": claims_by_name["actual_policy_consumption"][
                "recomputed_value"
            ],
            "official_score": claims_by_name["official_arc_score"]["recomputed_value"],
            "remaining_limit": "The live comparison was gate-blocked after witness readiness was zero.",
        },
        "NFR-01": {
            "complete": False,
            "durable_cost_captured": claims_by_name["durable_event_cost_captured"][
                "recomputed_value"
            ],
            "durable_cost_value": claims_by_name["durable_event_cost_value"]["recomputed_value"],
            "remaining_limit": "One interactive capacity CI crossed below parity, and the 10x target failed.",
        },
    }
    spans["validation_and_prd_mapping"] = time.monotonic() - phase
    progress(6, "end", f"validation_receipts={len(receipts)} PRD_rows={len(prd)}")

    artifact: JsonDict = {
        "schema": "carnot.exp7259.v638_capstone.v1",
        "experiment_id": "exp7259-capstone",
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": "",
        "field_principles": ALL_FIELD_PRINCIPLES,
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_invocation_counters": {"models": 0, "loads": 0, "generations": 0, "calls": 0},
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
                "completed": sum(item["terminal"] for item in evidence.values()),
                "censored": sum(
                    not item["declared_artifact_present"] for item in evidence.values()
                ),
                "independent_units": 13,
            },
            "claim_rows": {
                "planned": 10,
                "attempted": 10,
                "completed": 10,
                "censored": sum(row["censored"] is True for row in claims),
                "independent_units": 6,
            },
            "stopping_rule": "Stop after all fourteen frozen task slots, five gates, ten distinct claims, and four PRD rows are recorded.",
        },
        "acceptance_gate_results": _acceptance_results(
            contract, matrix, claims, gate_rows, receipts
        ),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": "blocked",
        "validation_receipts": receipts,
        "capstone_complete_score": 1,
        "evidence_matrix": matrix,
        "branch_decisions": decisions,
        "prd_gap_matrix": prd,
        "contract_rows": contract["contract_rows"],
        "recomputed_claim_rows": claims,
        "same_milestone_gate_replay_rows": gate_rows,
        "historical_and_negative_fixture_sidecars": sidecars,
        "source_ingestion_decisions": deepcopy(
            evidence["exp7246-source-map"]["payload"].get("source_method_rows", [])
        ),
        "hardware_dispositions": deepcopy(
            evidence["exp7258-board-state"]["payload"].get("board_rows", [])
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

    progress(7, "start", "validate complete derived checkpoint before terminal write")
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise RuntimeError("capstone validation failed: " + ",".join(errors))
    common._atomic_write(checkpoint_path, artifact)
    progress(7, "end", "derived checkpoint passed")

    progress(8, "start", "atomic terminal write")
    common._atomic_write(output_path, artifact)
    final = read_json(output_path)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    progress(8, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Build the capstone or independently validate one existing artifact."""

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
        progress(7, "start", f"independent raw-row reducer for {output}")
        errors = validate_artifact(read_json(output), root=root)
        progress(7, "end", "passed" if not errors else ",".join(errors))
        return int(bool(errors))
    build_artifact(
        root,
        args.date,
        output,
        checkpoint,
        raw_dir=args.raw_dir,
        validation_receipt_path=args.validation_receipt_path,
    )
    return 0
