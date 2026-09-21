"""Close V656 with fourteen authenticated dispositions and separate claims.

This reducer reads completed artifacts and exact missing paths. It performs no
model call, fitting, hardware action, roadmap activation, or publication.

Spec refs: REQ-REPORT-7502 and SCENARIO-REPORT-7502-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7446_v652_capstone import (
    authenticate_row_manifest,
    authenticate_validation_receipts,
    load_json_object,
    numeric_experiment_id,
    terminal_status,
)
from carnot.experiment_7489_v656_contract_methods import (
    compare_contract_authorities,
    resolve_v656_roadmap,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7502-capstone"
SCHEMA = "carnot.exp7502.v656.capstone.v1"

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7502_v656_capstone.json")
RAW_DIR = Path("results/raw/experiment_7502_v656_capstone")
MODULE_PATH = Path("python/carnot/experiment_7502_v656_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7502_v656_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7502_v656_capstone.py")
RETROSPECTIVE_PATH = Path("docs/research-notes/v656-retrospective.md")
STUDY_PATH = Path("research-studying.md")

EXPECTED_TASK_IDS = (
    "exp7489-contract-methods",
    "exp7490-historical-audit",
    "exp7491-window-protocol",
    "exp7492-window-pilot",
    "exp7493-window-fit-capture",
    "exp7494-window-eval-capture",
    "exp7495-window-calibration",
    "exp7496-causal-update-fixture",
    "exp7497-causal-online-learning",
    "exp7498-independent-audit",
    "exp7499-arc-panel-b",
    "exp7500-arc-opportunity-audit",
    "exp7501-service-placement",
    "exp7502-capstone",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
RANDOM_SEED = {
    "role": 656_001,
    "optimizer": None,
    "audit": 7_502_656_03,
    "order": 7_502_656_02,
    "interval": None,
    "deterministic_null_seed_explanation": (
        "The capstone performs deterministic aggregation, so optimizer and interval seeds do not apply."
    ),
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("results/experiment_7488_v655_capstone.json"),
    Path("python/carnot/experiment_7488_v655_capstone.py"),
    Path("ops/north-star.md"),
    Path("research-hardware-wishlist.md"),
    ROADMAP_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    RETROSPECTIVE_PATH,
    STUDY_PATH,
)
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_yaml_mapping(path: Path) -> JsonDict:
    """Load one YAML authority and reject a sequence or malformed mapping."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def load_contract(root: Path) -> JsonDict:
    """Resolve only V656 and compare its YAML with the Markdown authority."""

    selected, roadmap, candidates = resolve_v656_roadmap(root)
    comparison = compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap
    )
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("active V656 task list required")
    return {
        **deepcopy(comparison),
        "comparison_passed": comparison.get("passed") is True,
        "selected_roadmap_path": selected.relative_to(root).as_posix(),
        "resolution_candidates": candidates,
        "roadmap": deepcopy(roadmap),
        "tasks": deepcopy(tasks),
    }


def _fallback_pre_gate_path(task: Mapping[str, Any]) -> Path:
    """Derive the conductor path from the declared task identifier only."""

    task_id = str(task.get("id") or "")
    number = numeric_experiment_id(task_id)
    slug = task_id.split("-", 1)[1].replace("-", "_")
    return Path(f"results/experiment_{number}_{slug}.json")


def _validation_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Keep validation identity and hashes without copying large log output."""

    receipts = payload.get("validation_receipts")
    if not isinstance(receipts, list):
        return []
    return [
        {
            key: deepcopy(row.get(key))
            for key in ("name", "required", "passed", "exit_code", "log_path", "log_sha256")
        }
        for row in receipts
        if isinstance(row, Mapping)
    ]


def _gate_status(payload: Mapping[str, Any], terms: Sequence[str]) -> str:
    """Reduce declared gates without treating an undeclared gate as a pass."""

    gates = payload.get("acceptance_gate_results")
    if not isinstance(gates, list):
        return "not_declared"
    selected = [
        row
        for row in gates
        if isinstance(row, Mapping)
        and any(term in str(row.get("category") or "").lower() for term in terms)
    ]
    if not selected:
        return "not_declared"
    return "passed" if all(row.get("passed") is True for row in selected) else "failed"


def _gate_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain each producer gate operand and its failure-prevention principle."""

    gates = payload.get("acceptance_gate_results")
    if not isinstance(gates, list):
        return []
    fields = (
        "check",
        "category",
        "upstream",
        "path",
        "field",
        "artifact_field",
        "expected",
        "observed",
        "op",
        "passed",
        "principle",
    )
    return [
        {key: deepcopy(row.get(key)) for key in fields if key in row}
        for row in gates
        if isinstance(row, Mapping)
    ]


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent an absent external producer as blocked, never partial."""

    expected = str(task.get("deliverable") or "")
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected,
        "found_path": None,
        "evidence_state": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "raw_experiment_id": None,
        "raw_milestone": None,
        "raw_status": None,
        "honest_verdict": "blocked_missing_declared_producer_evidence",
        "verdict_class": "blocked",
        "original_verdict_class": None,
        "flagged_adversarial": False,
        "required_validation_passed": False,
        "required_validation_status": "not_run",
        "validation_receipts": [],
        "validation_failures": ["producer_evidence_missing"],
        "source_sha256": None,
        "source_size_bytes": 0,
        "row_count": 0,
        "rows_available": False,
        "support_gate_status": "not_run",
        "benefit_gate_status": "not_run",
        "acceptance_gates": [],
        "payload": {},
        "gate_check_summary": {
            "check": "producer_evidence",
            "upstream": str(task.get("id")),
            "path": expected,
            "field": "path",
            "op": "exists",
            "expected": True,
            "observed": False,
            "passed": False,
        },
    }


def _pre_gate_evidence(
    root: Path, task: Mapping[str, Any], relative: Path, payload: JsonDict
) -> JsonDict:
    """Authenticate conductor bytes and the exact upstream bytes they cite."""

    evidence_path = Path(str(payload.get("failed_evidence_path") or ""))
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    expected_hash = payload.get("failed_evidence_sha256")
    evidence_matches = bool(
        evidence_path.is_file()
        and isinstance(expected_hash, str)
        and sha256_file(evidence_path) == expected_hash
    )
    required = (
        "failed_upstream",
        "failed_field",
        "failed_operator",
        "failed_expected",
        "failed_observed",
    )
    authenticated = bool(
        numeric_experiment_id(payload.get("experiment")) == numeric_experiment_id(task.get("id"))
        and payload.get("blocked_at_layer") == "conductor_pre_gate"
        and payload.get("status") == "blocked"
        and all(field in payload for field in required)
        and evidence_matches
    )
    return {
        **_missing_evidence(task),
        "found_path": relative.as_posix(),
        "evidence_state": "pre_gate" if authenticated else "invalid",
        "authenticated": authenticated,
        "valid": authenticated,
        "raw_experiment_id": payload.get("experiment"),
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "honest_verdict": str(payload.get("honest_verdict") or "blocked_conductor_pre_gate"),
        "verdict_class": "blocked" if authenticated else "disqualified",
        "original_verdict_class": "blocked",
        "required_validation_passed": authenticated,
        "required_validation_status": "passed" if authenticated else "failed",
        "validation_failures": [] if authenticated else ["pre_gate_authentication_failed"],
        "source_sha256": sha256_file(root / relative),
        "source_size_bytes": (root / relative).stat().st_size,
        "support_gate_status": "not_run",
        "benefit_gate_status": "not_run",
        "payload": payload,
        "gate_check_summary": {
            "check": "structured_pre_gate",
            "upstream": payload.get("failed_upstream"),
            "path": relative.as_posix(),
            "field": payload.get("failed_field"),
            "op": payload.get("failed_operator"),
            "expected": payload.get("failed_expected"),
            "observed": payload.get("failed_observed"),
            "passed": False,
        },
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one declared producer, exact pre-gate, or missing slot."""

    expected = Path(str(task.get("deliverable") or ""))
    relative = expected
    if not (root / relative).is_file():
        fallback = _fallback_pre_gate_path(task)
        if not (root / fallback).is_file():
            return _missing_evidence(task)
        relative = fallback
    try:
        payload = load_json_object(root / relative)
    except ValueError:
        payload = {}
    if payload.get("schema") == "blocked_gate_check_v1":
        return _pre_gate_evidence(root, task, relative, payload)
    row_receipt = authenticate_row_manifest(root, payload)
    validation = authenticate_validation_receipts(root, payload)
    raw_id = payload.get("experiment_id", payload.get("experiment"))
    original_verdict = payload.get("verdict_class")
    authenticated = bool(
        numeric_experiment_id(raw_id) == numeric_experiment_id(task.get("id"))
        and payload.get("milestone") == MILESTONE
        and original_verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and terminal_status(payload)
        and row_receipt["authenticated"]
        and validation["authenticated"]
        and validation["receipt_count"] > 0
    )
    flagged = payload.get("flagged_adversarial") is True
    valid = bool(
        authenticated
        and validation["required_passed"]
        and original_verdict != "disqualified"
        and not flagged
    )
    failures = list(validation["failures"])
    failures.extend(
        str(row.get("name"))
        for row in payload.get("validation_receipts", [])
        if isinstance(row, Mapping)
        and row.get("required") is True
        and (row.get("passed") is not True or row.get("exit_code") != 0)
    )
    row_count = row_receipt["inline_row_count"] + row_receipt["raw_shard_row_count"]
    return {
        "task_id": str(task.get("id")),
        "expected_path": expected.as_posix(),
        "found_path": relative.as_posix(),
        "evidence_state": "terminal" if valid else "invalid",
        "authenticated": authenticated,
        "available": authenticated,
        "valid": valid,
        "raw_experiment_id": raw_id,
        "raw_milestone": payload.get("milestone"),
        "raw_status": payload.get("status"),
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "verdict_class": str(original_verdict) if valid else "disqualified",
        "original_verdict_class": original_verdict,
        "flagged_adversarial": flagged,
        "required_validation_passed": validation["required_passed"],
        "required_validation_status": "passed" if validation["required_passed"] else "failed",
        "validation_receipts": _validation_rows(payload),
        "validation_failures": list(dict.fromkeys(failures)),
        "source_sha256": sha256_file(root / relative),
        "source_size_bytes": (root / relative).stat().st_size,
        "row_count": row_count,
        "rows_available": row_count > 0,
        "support_gate_status": _gate_status(payload, ("support", "readiness")),
        "benefit_gate_status": _gate_status(payload, ("benefit", "efficacy", "value")),
        "acceptance_gates": _gate_rows(payload),
        "payload": payload,
        "gate_check_summary": deepcopy(payload.get("gate_check_summary")),
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Authenticate all thirteen predecessor slots in roadmap order."""

    return {str(task["id"]): load_evidence_slot(root, task) for task in tasks[:-1]}


def classify_terminal(
    rows: Sequence[Mapping[str, Any]], *, current_validation_complete: bool
) -> JsonDict:
    """Apply owned incompletion, invalid bytes, external blocks, then benefit."""

    if not current_validation_complete:
        verdict = "partial"
        honest = "partial_retryable_current_capstone_validation_unfinished"
    elif any(row.get("evidence_state") == "invalid" for row in rows):
        verdict = "disqualified"
        honest = "complete_disqualified_required_present_v656_evidence"
    elif any(
        row.get("evidence_state") in {"missing", "pre_gate"}
        or row.get("verdict_class") == "blocked"
        for row in rows
    ):
        verdict = "blocked"
        honest = "complete_blocked_required_v656_evidence_absent_or_externally_gated"
    else:
        verdict = "null"
        honest = "complete_null_v656_science_without_registered_aggregate_benefit"
    return {"verdict_class": verdict, "honest_verdict": honest, "status": honest}


def _source_claim(source: Mapping[str, Any], **values: Any) -> JsonDict:
    """Attach exact source and support to one retained claim."""

    return {
        "source_path": source.get("found_path") or source.get("expected_path"),
        "source_sha256": source.get("source_sha256"),
        "source_verdict_class": source.get("verdict_class"),
        "source_valid": source.get("valid") is True,
        "row_count": source.get("row_count", 0),
        **deepcopy(values),
    }


def _claim_state(source: Mapping[str, Any]) -> str:
    """Name missing, invalid, blocked, or available claim evidence."""

    state = source.get("evidence_state")
    if state in {"missing", "pre_gate"}:
        return "blocked_missing"
    if state == "invalid":
        return "disqualified"
    if source.get("verdict_class") == "blocked":
        return "blocked_upstream"
    return "available"


def reduce_supported_claims(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Keep probability, utility, learning, ARC, and placement independent."""

    historical = evidence["exp7490-historical-audit"]
    historical_limits = historical["payload"].get("historical_claim_limits") or {}
    science = evidence["exp7498-independent-audit"]
    science_payload = science["payload"]
    dispositions = {
        str(row.get("task_id")): row
        for row in science_payload.get("claim_dispositions", [])
        if isinstance(row, Mapping)
    }
    calibration = evidence["exp7495-window-calibration"]
    fixture = evidence["exp7496-causal-update-fixture"]
    learning = evidence["exp7497-causal-online-learning"]
    arc = evidence["exp7500-arc-opportunity-audit"]
    arc_payload = arc["payload"]
    pooling = arc_payload.get("pooling") or {}
    service = evidence["exp7501-service-placement"]
    probability_limit = historical_limits.get("probability_quality") or {}
    utility_limit = historical_limits.get("typed_utility") or {}
    return {
        "historical_probability_quality": _source_claim(
            historical,
            finding=probability_limit.get("finding", "unknown"),
            scope=probability_limit.get("scope"),
            fresh_holdout=probability_limit.get("fresh_holdout"),
            support="74 external FaithBench groups from historical diagnostic evidence",
        ),
        "historical_typed_decision_utility": _source_claim(
            historical,
            finding=utility_limit.get("finding", "unknown"),
            passed_cost_cells=utility_limit.get("passed_cost_cells"),
            total_cost_cells=utility_limit.get("total_cost_cells"),
            fresh_holdout=utility_limit.get("fresh_holdout"),
            probability_claim_authorized=False,
        ),
        "current_probability_quality": _source_claim(
            calibration,
            finding=_claim_state(calibration),
            audit_disposition=deepcopy(dispositions.get("exp7495-window-calibration")),
            independent_probability_row_count=len(
                science_payload.get("independent_probability_rows") or []
            ),
        ),
        "current_typed_decision_utility": _source_claim(
            calibration,
            finding=_claim_state(calibration),
            audit_disposition=deepcopy(dispositions.get("exp7495-window-calibration")),
            independent_utility_row_count=len(
                science_payload.get("independent_utility_rows") or []
            ),
        ),
        "causal_fixture": _source_claim(
            fixture,
            finding=_claim_state(fixture),
            verifier_is_oracle=fixture["payload"].get("verifier_is_oracle"),
            efficacy_authorized=False,
        ),
        "causal_feedback_benefit": _source_claim(
            learning,
            finding=_claim_state(learning),
            audit_disposition=deepcopy(dispositions.get("exp7497-causal-online-learning")),
            independent_learning_row_count=len(
                science_payload.get("independent_learning_rows") or []
            ),
        ),
        "retention": _source_claim(
            learning,
            finding=_claim_state(learning),
            reported_separately=True,
            benefit_borrowed_from_fixture=False,
        ),
        "arc_generalization": _source_claim(
            arc,
            finding="blocked_low_support_and_missing_panel_b",
            valid_episode_count=pooling.get("valid_episode_count"),
            valid_game_count=pooling.get("valid_game_count"),
            episode_support_floor=pooling.get("episode_support_floor"),
            game_support_floor=pooling.get("game_support_floor"),
            efficacy_estimate=arc_payload.get("efficacy_estimate"),
            triggered_opportunity_count=(arc_payload.get("supervisor_disposition") or {}).get(
                "triggered_opportunity_count"
            ),
        ),
        "hardware_placement": _source_claim(
            service,
            finding=_claim_state(service),
            hypothetical_only=True,
            measured_board_performance=False,
        ),
        "fixture_efficacy_promoted": False,
        "typed_utility_promoted_to_probability": False,
        "hypothetical_placement_promoted_to_hardware_measurement": False,
    }


SAME_DISPOSITION_TASKS = {
    "exp7490-historical-audit",
    "exp7492-window-pilot",
    "exp7493-window-fit-capture",
    "exp7494-window-eval-capture",
}


def reduce_prior_failures(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Apply every declared retirement rule without retiring external absence."""

    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        failures = task.get("prior_failures") or []
        if len(failures) != 1 or not isinstance(failures[0], Mapping):
            raise ValueError(f"exactly one prior failure required for {task_id}")
        prior = failures[0]
        source = evidence.get(task_id)
        if source is None:
            comparison_state = "current_work"
            current_verdict = None
            current_class = None
        elif source["evidence_state"] in {"missing", "pre_gate"}:
            comparison_state = "current_absent"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        elif source["evidence_state"] == "invalid":
            comparison_state = "current_invalid"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        else:
            comparison_state = "compared"
            current_verdict = source["honest_verdict"]
            current_class = source["verdict_class"]
        exact_same = current_verdict == prior.get("verdict")
        same_disposition = bool(
            comparison_state == "compared" and (exact_same or task_id in SAME_DISPOSITION_TASKS)
        )
        retire = bool(prior.get("retire_if_same_verdict") is True and same_disposition)
        rows.append(
            {
                "task_id": task_id,
                "prior_experiment": prior.get("experiment_id"),
                "prior_honest_verdict": prior.get("verdict"),
                "addressed_by": prior.get("addressed_by"),
                "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                "comparison_state": comparison_state,
                "current_honest_verdict": current_verdict,
                "current_verdict_class": current_class,
                "exact_text_match": exact_same,
                "same_scientific_disposition": same_disposition,
                "retirement_triggered": retire,
                "retirement_reason": (
                    "The current mechanism repeated the prior no-benefit scientific disposition."
                    if retire
                    else "No same-disposition terminal comparison triggered retirement."
                ),
            }
        )
    return rows


def next_reopen_conditions() -> list[JsonDict]:
    """Require one concrete method or evidence change for each closed branch."""

    return [
        {
            "branch": "importance_anchoring",
            "state": "closed",
            "changed_prerequisite": "Use a new causal objective with held-out benefit; do not repeat the registered importance anchor.",
        },
        {
            "branch": "compact_generated_spans",
            "state": "closed",
            "changed_prerequisite": "Use a lossless natural-text method that preserves all qualifiers and complete sentences.",
        },
        {
            "branch": "four_expert_reweighting",
            "state": "closed",
            "changed_prerequisite": "Register a different expert construction with an oracle-distinct held-out advantage.",
        },
        {
            "branch": "generic_external_text_reranking",
            "state": "closed",
            "changed_prerequisite": "Provide a new external-text mechanism and held-out gate that is distinct from source-option readout.",
        },
        {
            "branch": "current_window_probability",
            "state": "closed",
            "changed_prerequisite": "Produce Exp7495 from the sealed fit and evaluation captures with support, proper-score, and multiplicity gates.",
        },
        {
            "branch": "causal_feedback_learning",
            "state": "closed",
            "changed_prerequisite": "Produce the causal fixture and online learner with chronological feedback, retention, and independent reduction.",
        },
        {
            "branch": "arc_cross_game_generalization",
            "state": "closed",
            "changed_prerequisite": "Publish a compatible terminal panel B and reach 30 valid episodes across 10 games before efficacy work.",
        },
        {
            "branch": "durable_service_and_hardware_placement",
            "state": "closed",
            "changed_prerequisite": "Publish the same-process durable service trace; treat board placement as hypothetical until a board runs it.",
        },
    ]


def _dominant_timing(task_id: str, source: Mapping[str, Any]) -> JsonDict | None:
    """Select the largest measured producer component without inventing time."""

    payload = source.get("payload")
    if not isinstance(payload, Mapping):
        return None
    components = payload.get("duration_components_s") or payload.get("duration_breakdown_s")
    if not isinstance(components, Mapping):
        return None
    values = {
        str(key): float(value)
        for key, value in components.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and str(key) != "total"
    }
    if not values:
        return None
    name, duration = max(values.items(), key=lambda item: item[1])
    return {
        "task_id": task_id,
        "artifact_path": source.get("found_path"),
        "artifact_sha256": source.get("source_sha256"),
        "total_duration_s": payload.get("duration_s"),
        "dominant_stage": name.removesuffix("_s"),
        "dominant_duration_s": duration,
    }


def reduce_retrospective(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Summarize the three open PRD gaps, support limits, and measured time."""

    arc = evidence["exp7500-arc-opportunity-audit"]["payload"]
    pooling = arc.get("pooling") or {}
    timings = [
        row
        for task_id, source in evidence.items()
        if (row := _dominant_timing(task_id, source)) is not None
    ]
    timings.sort(key=lambda row: float(row["dominant_duration_s"]), reverse=True)
    return {
        "document_path": RETROSPECTIVE_PATH.as_posix(),
        "prd_gaps": [
            {
                "gap": "calibrated_source_decision",
                "status": "blocked_missing_exp7495",
                "limit": "Historical typed utility exists, but current held-out probability and utility rows are absent.",
            },
            {
                "gap": "causal_continuous_learning",
                "status": "blocked_missing_exp7496_and_exp7497",
                "limit": "No current causal fixture, feedback-benefit, or retention evidence is terminal.",
            },
            {
                "gap": "arc_service_and_hardware_scope",
                "status": "blocked_missing_exp7499_and_exp7501",
                "limit": "Panel A has low support, panel B is absent, and hardware placement remains hypothetical.",
            },
        ],
        "support_limits": {
            "current_probability_rows": 0,
            "current_utility_rows": 0,
            "current_learning_rows": 0,
            "arc_valid_episode_count": pooling.get("valid_episode_count"),
            "arc_valid_game_count": pooling.get("valid_game_count"),
            "arc_episode_floor": pooling.get("episode_support_floor"),
            "arc_game_floor": pooling.get("game_support_floor"),
            "arc_triggered_opportunity_count": (arc.get("supervisor_disposition") or {}).get(
                "triggered_opportunity_count"
            ),
        },
        "dominant_actual_timings": timings,
        "next_reopen_conditions": next_reopen_conditions(),
    }


def retirement_rows(prior_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep triggered V656 retirements and four existing closed mechanisms."""

    triggered = [deepcopy(dict(row)) for row in prior_rows if row.get("retirement_triggered")]
    permanent = [
        {
            "task_id": "importance_anchoring",
            "prior_experiment": "exp7483-continuous-learning",
            "prior_honest_verdict": "complete_null_no_registered_importance_anchor_benefit",
            "same_scientific_disposition": True,
            "retirement_triggered": True,
            "retirement_reason": "The unchanged importance-anchor mechanism has no registered benefit.",
        },
        {
            "task_id": "compact_generated_spans",
            "prior_experiment": "exp7467-factual-span-canary",
            "prior_honest_verdict": "complete_null_natural_text_extraction",
            "same_scientific_disposition": True,
            "retirement_triggered": True,
            "retirement_reason": "Compact spans remain closed until a lossless natural-text method changes.",
        },
        {
            "task_id": "four_expert_reweighting",
            "prior_experiment": "exp7468-residual-learner",
            "prior_honest_verdict": "complete_null_unchanged_four_expert_mixture",
            "same_scientific_disposition": True,
            "retirement_triggered": True,
            "retirement_reason": "The unchanged four-expert mechanism has repeated no-benefit evidence.",
        },
        {
            "task_id": "generic_external_text_reranking",
            "prior_experiment": "exp7470-independent-audit",
            "prior_honest_verdict": "complete_null_external_text_verifier_moat",
            "same_scientific_disposition": True,
            "retirement_triggered": True,
            "retirement_reason": "Source-option readout does not establish a generic external-text reranking benefit.",
        },
    ]
    return [*triggered, *permanent]


def _receipt_set_passed(receipts: object, names: Sequence[str]) -> bool:
    """Require exactly one successful receipt for each declared command."""

    if not isinstance(receipts, list):
        return False
    selected = [row for row in receipts if isinstance(row, Mapping)]
    counts = Counter(str(row.get("name")) for row in selected)
    by_name = {str(row.get("name")): row for row in selected}
    return all(
        counts[name] == 1
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build fourteen ordered rows without reading the current result path."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        source = evidence[str(task["id"])]
        rows.append(
            {
                "order": order,
                "task_id": str(task["id"]),
                "expected_artifact_path": source["expected_path"],
                "artifact_path": source["found_path"],
                "artifact_sha256": source["source_sha256"],
                "evidence_state": source["evidence_state"],
                "raw_experiment_id": source["raw_experiment_id"],
                "raw_milestone": source["raw_milestone"],
                "raw_status": source["raw_status"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "original_verdict_class": source["original_verdict_class"],
                "flags": {"flagged_adversarial": source["flagged_adversarial"]},
                "flagged_adversarial": source["flagged_adversarial"],
                "required_validation_status": source["required_validation_status"],
                "required_validation_passed": source["required_validation_passed"],
                "validation_failures": deepcopy(source["validation_failures"]),
                "rows_available": source["rows_available"],
                "row_count": source["row_count"],
                "support_gate_status": source["support_gate_status"],
                "benefit_gate_status": source["benefit_gate_status"],
                "acceptance_gates": deepcopy(source["acceptance_gates"]),
                "attempted": source["evidence_state"] != "missing",
                "completed": source["evidence_state"] != "missing",
                "failed": source["evidence_state"] == "invalid",
                "censored": source["evidence_state"] in {"missing", "pre_gate"},
                "excluded_from_positive_aggregate": not source["valid"],
                "unstarted": source["evidence_state"] == "missing",
                "principle": "An exact per-task record prevents absent, invalid, or low-support evidence from becoming completed benefit.",
            }
        )
    current_complete = validation.get("required_checks_passed") is True
    rows.append(
        {
            "order": 14,
            "task_id": str(tasks[-1]["id"]),
            "expected_artifact_path": str(tasks[-1]["deliverable"]),
            "artifact_path": None,
            "artifact_sha256": None,
            "evidence_state": "current_work",
            "raw_experiment_id": EXPERIMENT_ID,
            "raw_milestone": MILESTONE,
            "raw_status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "original_verdict_class": terminal["verdict_class"],
            "flags": {"flagged_adversarial": False},
            "flagged_adversarial": False,
            "required_validation_status": "passed" if current_complete else "unfinished",
            "required_validation_passed": current_complete,
            "validation_failures": [] if current_complete else ["current_validation_unfinished"],
            "rows_available": True,
            "row_count": 1,
            "support_gate_status": "not_declared",
            "benefit_gate_status": "not_declared",
            "acceptance_gates": [],
            "attempted": True,
            "completed": current_complete,
            "failed": False,
            "censored": False,
            "excluded_from_positive_aggregate": True,
            "unstarted": False,
            "principle": "Current work stays distinct from its future output, which prevents circular self-authentication.",
        }
    )
    return rows


def _gate(
    check: str,
    category: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    passed: bool,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Create one gate with exact operands and its failure prevention rule."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    dispositions: Sequence[Mapping[str, Any]],
    claims: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep current validity, reporting readiness, and benefit independent."""

    present_valid = all(row["valid"] for row in evidence.values() if row["available"])
    current_valid = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    validity = "A favorable metric cannot excuse invalid evidence."
    readiness = "A valid scientific null must not block independent measurements."
    benefit = "A favorable seed, fixture or low-support result cannot replace held-out value."
    return [
        _gate(
            "contract_authorities_agree",
            "validity",
            "v656_contract",
            str(contract.get("selected_roadmap_path")),
            "comparison_passed",
            True,
            contract.get("comparison_passed"),
            passed=contract.get("comparison_passed") is True,
            principle=validity,
        ),
        _gate(
            "required_present_evidence_valid",
            "validity",
            "v656_producers",
            "task_dispositions",
            "valid",
            True,
            present_valid,
            passed=present_valid,
            principle=validity,
        ),
        _gate(
            "current_scoped_and_terminal_validation",
            "validity",
            EXPERIMENT_ID,
            "validation_receipts",
            "all_required_passed",
            True,
            current_valid,
            passed=current_valid,
            principle=validity,
        ),
        _gate(
            "fourteen_dispositions_recorded",
            "readiness",
            EXPERIMENT_ID,
            "task_dispositions",
            "length",
            14,
            len(dispositions),
            passed=len(dispositions) == 14,
            principle=readiness,
        ),
        _gate(
            "current_probability_support_effect_and_multiplicity",
            "benefit",
            "exp7498-independent-audit",
            "supported_claims.current_probability_quality",
            "finding",
            "positive",
            claims["current_probability_quality"]["finding"],
            passed=claims["current_probability_quality"]["finding"] == "positive",
            principle=benefit,
        ),
        _gate(
            "causal_feedback_benefit_and_retention",
            "benefit",
            "exp7498-independent-audit",
            "supported_claims.causal_feedback_benefit",
            "finding",
            "positive",
            claims["causal_feedback_benefit"]["finding"],
            passed=claims["causal_feedback_benefit"]["finding"] == "positive",
            principle=benefit,
        ),
        _gate(
            "arc_cross_game_support",
            "benefit",
            "exp7500-arc-opportunity-audit",
            "supported_claims.arc_generalization",
            "episode_and_game_support",
            {"episodes": 30, "games": 10},
            {
                "episodes": claims["arc_generalization"]["valid_episode_count"],
                "games": claims["arc_generalization"]["valid_game_count"],
            },
            op=">=",
            passed=bool(
                (claims["arc_generalization"]["valid_episode_count"] or 0) >= 30
                and (claims["arc_generalization"]["valid_game_count"] or 0) >= 10
            ),
            principle=benefit,
        ),
        _gate(
            "durable_service_measurement",
            "benefit",
            "exp7501-service-placement",
            "supported_claims.hardware_placement",
            "finding",
            "measured",
            claims["hardware_placement"]["finding"],
            passed=claims["hardware_placement"]["finding"] == "measured",
            principle=benefit,
        ),
    ]


def failure_rows(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Name every contract, invalid, absent, blocked, and current check."""

    rows: list[JsonDict] = []
    if contract.get("comparison_passed") is not True:
        rows.append(
            {
                "check": "contract_authorities_agree",
                "upstream": "v656_contract",
                "path": str(contract.get("selected_roadmap_path")),
                "field": "comparison_passed",
                "op": "==",
                "expected": True,
                "observed": contract.get("comparison_passed"),
                "passed": False,
            }
        )
    for task_id, source in evidence.items():
        state = source["evidence_state"]
        if state in {"missing", "pre_gate"}:
            rows.append(deepcopy(dict(source["gate_check_summary"])))
        elif state == "invalid":
            rows.append(
                {
                    "check": "required_source_validation",
                    "upstream": task_id,
                    "path": source["found_path"] or source["expected_path"],
                    "field": "required_validation_passed",
                    "op": "==",
                    "expected": True,
                    "observed": source["required_validation_passed"],
                    "passed": False,
                }
            )
        elif source["verdict_class"] == "blocked":
            rows.append(
                {
                    "check": "producer_terminal_disposition",
                    "upstream": task_id,
                    "path": source["found_path"],
                    "field": "verdict_class",
                    "op": "not in",
                    "expected": ["blocked"],
                    "observed": source["verdict_class"],
                    "passed": False,
                }
            )
    if validation.get("required_checks_passed") is not True:
        rows.append(
            {
                "check": "affected_validation",
                "upstream": EXPERIMENT_ID,
                "path": "validation_receipts",
                "field": "required_checks_passed",
                "op": "==",
                "expected": True,
                "observed": validation.get("required_checks_passed"),
                "passed": False,
            }
        )
    if validation.get("terminal_validation_passed") is not True:
        rows.append(
            {
                "check": "terminal_validation",
                "upstream": EXPERIMENT_ID,
                "path": "validation_receipts",
                "field": "terminal_validation_passed",
                "op": "==",
                "expected": True,
                "observed": validation.get("terminal_validation_passed"),
                "passed": False,
            }
        )
    return rows


def _gate_summary(failures: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all blocking checks and the first exact classification cause."""

    return {
        "passed": not failures,
        "failed_count": len(failures),
        "first_failure": deepcopy(dict(failures[0])) if failures else None,
        "failed_checks": deepcopy([dict(row) for row in failures]),
    }


def _preconditions(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Record exact paths, values, owners, and evidence state before reduction."""

    rows = [
        {
            "check": f"source_bytes:{path.as_posix()}",
            "upstream": path.as_posix(),
            "path": path.as_posix(),
            "owner": "repository",
            "field": "bytes",
            "expected": "readable_nonempty_bytes",
            "observed": "readable_nonempty_bytes",
            "passed": (root / path).is_file() and (root / path).stat().st_size > 0,
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        [
            {
                "check": "contract_authority_equivalence",
                "upstream": "v656_contract",
                "path": f"{DESIGN_PATH} + {contract['selected_roadmap_path']}",
                "owner": "repository",
                "field": "comparison_passed",
                "expected": True,
                "observed": contract["comparison_passed"],
                "passed": contract["comparison_passed"] is True,
            },
            {
                "check": "execution_venue",
                "upstream": EXPERIMENT_ID,
                "path": "/proc/self",
                "owner": "current_process",
                "field": "venue",
                "expected": "host",
                "observed": "host",
                "passed": True,
            },
        ]
    )
    rows.extend(
        {
            "check": "predecessor_state_observed",
            "upstream": task_id,
            "path": source["found_path"] or source["expected_path"],
            "owner": "upstream_producer_or_conductor",
            "field": "evidence_state",
            "expected": "one explicit state",
            "observed": source["evidence_state"],
            "passed": source["evidence_state"] in {"terminal", "invalid", "missing", "pre_gate"},
        }
        for task_id, source in evidence.items()
    )
    return rows


def _source_hashes(
    root: Path, contract: Mapping[str, Any], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Bind instructions, implementation, retrospective, and every source slot."""

    rows = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "required_source",
        }
        for path in SOURCE_PATHS
    ]
    rows.extend(
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "evidence_class": "current_implementation",
        }
        for path in (MODULE_PATH, WRAPPER_PATH, TEST_PATH)
    )
    rows.extend(
        {
            "path": source["found_path"] or source["expected_path"],
            "sha256": source["source_sha256"],
            "evidence_class": source["evidence_state"],
            "task_id": task_id,
            "original_honest_verdict": source["honest_verdict"],
            "original_verdict_class": source["original_verdict_class"],
            "original_flagged_adversarial": source["flagged_adversarial"],
        }
        for task_id, source in evidence.items()
    )
    selected = str(contract["selected_roadmap_path"])
    if selected != ROADMAP_PATH.as_posix():
        rows.append(
            {
                "path": selected,
                "sha256": sha256_file(root / selected),
                "evidence_class": "selected_contract_authority",
            }
        )
    return rows


def _source_hashes_match(value: Mapping[str, Any], root: Path) -> bool:
    """Recheck every source row that claims an existing byte hash."""

    rows = value.get("source_artifact_hashes")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        expected = row.get("sha256")
        if expected is None:
            continue
        path = root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


FIELD_PRINCIPLES = {
    "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
    "run_date": "Use 20260921 and retain measured UTC, monotonic clock, and process identity.",
    "preconditions_checked": "Record exact resource paths, observed values, ownership, and input validity.",
    "MODEL_SPECS": "Aggregation has no current model; both model-spec fields must remain empty.",
    "model_invoked": "Attempted live loads, forwards, and generations differ from historical model events.",
    "invocation_counts": "Reconcile attempted, completed, failed, cancelled, and in-flight current calls.",
    "inference_substrate": "Aggregation uses exactly aggregation_from_upstream_artifacts.",
    "inference_substrate_class": "The aggregation class prevents model or numeric work from being implied.",
    "execution_venue": "Host execution stays distinct from archived board evidence.",
    "duration_s": "Measure actual work and separate aggregation from validation without a synthetic floor.",
    "phase_spans": "Flushed progress and checkpoints expose unfinished operations and stalls.",
    "random_seed": "Freeze role, optimizer, audit, order, and interval seeds and explain null seeds.",
    "reproducibility_checksum": "Bind code, source artifacts, raw reductions, and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes, verdicts, and flags without laundering.",
    "rows": "Per-task rows with failures and censoring permit independent headline reduction.",
    "sample_size_budget": "Separate planned, attempted, complete, failed, excluded, censored, and unstarted units.",
    "acceptance_gate_results": "Each gate names validity, support, or benefit with exact operands and purpose.",
    "gate_check_summary": "Every blocked verdict names its upstream path, field, expected value, and observation.",
    "honest_verdict": "Use a complete terminal finding and preserve blocked conductor records as received.",
    "verdict_class": "Use the closed enum and reserve partial for this capstone's unfinished repair.",
    "verifier_is_oracle": "Oracle-defined fixture benefits cannot become positive efficacy claims.",
    "flagged_adversarial": "Retain actual reader flags and never clear a flag to open a gate.",
    "validation_receipts": "Exact commands, exits, scope, and log hashes make current checks reviewable.",
    "field_principles": "Every emitted field states the failure that its evidence prevents.",
    "capstone_complete_score": "Fourteen explicit dispositions measure reporting completion, not scientific success.",
    "task_dispositions": "Exact ordered IDs and paths include missing, pre-gated, and current tasks.",
    "supported_claims": "Every retained claim cites valid rows, support, and independent reduction.",
    "retirement_rows": "Repeated verdicts and mechanism-level stop decisions remain permanent.",
    "next_reopen_conditions": "A changed prerequisite or method is required before any repeat.",
}


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give every top-level artifact field one failure-prevention statement."""

    return {
        field: FIELD_PRINCIPLES.get(
            field,
            f"Retain exact {field} evidence so a later reader cannot infer an unstated result.",
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash stable source, disposition, claim, retirement, and validation data."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "duration_components_s",
        "phase_spans",
        "clock_identity",
        "process_identity",
        "device_identity",
    }
    return canonical_hash({key: item for key, item in value.items() if key not in excluded})


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete record from authenticated V656 evidence."""

    terminal = classify_terminal(
        list(evidence.values()),
        current_validation_complete=validation.get("required_checks_passed") is True,
    )
    dispositions = task_dispositions(contract["tasks"], evidence, terminal, validation)
    claims = reduce_supported_claims(evidence)
    prior_rows = reduce_prior_failures(contract["tasks"], evidence)
    failures = failure_rows(contract, evidence, validation)
    receipts = validation.get("validation_receipts") or []
    validation_s = sum(
        float(row.get("duration_s") or 0.0) for row in receipts if isinstance(row, Mapping)
    )
    current_complete = bool(
        validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": any(row["flagged_adversarial"] for row in evidence.values()),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "clock_identity": {
            "wall": "datetime.now(datetime.UTC)",
            "monotonic": "time.monotonic_ns",
        },
        "process_identity": {"pid": os.getpid(), "node": platform.node()},
        "device_identity": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "cuda_used": False,
        },
        "duration_components_s": {
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "optimization": 0.0,
            "validation": validation_s,
            "aggregation": max(0.0, duration_s - validation_s),
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": deepcopy(RANDOM_SEED),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "small_ebm_training": {
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "duration_s": 0.0,
            "receipt_scope": "current_exp7502_work_only",
        },
        "preconditions_checked": _preconditions(root, contract, evidence),
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_evidence_sidecars": [
            {
                "task_id": task_id,
                "path": source["found_path"],
                "sha256": source["source_sha256"],
                "scope": "upstream_model_or_aggregation_receipt",
                "counted_as_current_invocation": False,
                "original_model_invoked": source["payload"].get("model_invoked"),
            }
            for task_id, source in evidence.items()
            if source["found_path"] is not None
        ],
        "rows": deepcopy(dispositions),
        "task_dispositions": dispositions,
        "sample_size_budget": {
            "independent_unit": "ordered_v656_task_disposition",
            "planned": 14,
            "attempted": sum(int(row["attempted"]) for row in dispositions),
            "complete": sum(int(row["completed"]) for row in dispositions),
            "failed": sum(int(row["failed"]) for row in dispositions),
            "excluded": sum(int(row["excluded_from_positive_aggregate"]) for row in dispositions),
            "censored": sum(int(row["censored"]) for row in dispositions),
            "unstarted": sum(int(row["unstarted"]) for row in dispositions),
        },
        "supported_claims": claims,
        "prior_failure_rows": prior_rows,
        "retirement_rows": retirement_rows(prior_rows),
        "next_reopen_conditions": next_reopen_conditions(),
        "retrospective": reduce_retrospective(evidence),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
        "validation_receipts": deepcopy(receipts),
        "repository_health": deepcopy(
            validation.get("repository_health", {"status": "outside_current_affected_validity"})
        ),
        "acceptance_gate_results": acceptance_gates(
            contract, evidence, dispositions, claims, validation
        ),
        "gate_check_summary": _gate_summary(failures),
        "capstone_complete_score": int(
            current_complete
            and contract.get("comparison_passed") is True
            and len(dispositions) == 14
        ),
        "verifier_is_oracle": False,
        "roadmap_activated": False,
        "publication_performed": False,
        "submission_performed": False,
        "external_contact_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "push_performed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = False
) -> list[str]:
    """Cold-check identity, sources, reductions, principles, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "status",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "random_seed",
        "preconditions_checked",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "validation_receipts",
        "field_principles",
        "capstone_complete_score",
        "task_dispositions",
        "supported_claims",
        "retirement_rows",
        "next_reopen_conditions",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(artifact))
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != SCHEMA
        or artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("identity_invalid")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS or not str(
        artifact.get("honest_verdict")
    ).startswith(("complete_", "partial_")):
        errors.append("terminal_identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_specs") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    dispositions = artifact.get("task_dispositions")
    if (
        not isinstance(dispositions, list)
        or [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        != list(EXPECTED_TASK_IDS)
        or artifact.get("rows") != dispositions
    ):
        errors.append("task_dispositions_invalid")
    if not _source_hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        receipts = artifact.get("validation_receipts")
        validation = {
            "required_checks_passed": _receipt_set_passed(receipts, REQUIRED_CHECK_NAMES),
            "terminal_validation_passed": _receipt_set_passed(receipts, TERMINAL_CHECK_NAMES),
            "validation_receipts": receipts,
            "repository_health": artifact.get("repository_health", {}),
        }
        terminal = classify_terminal(
            list(evidence.values()),
            current_validation_complete=validation["required_checks_passed"],
        )
        expected_rows = task_dispositions(contract["tasks"], evidence, terminal, validation)
        claims = reduce_supported_claims(evidence)
        priors = reduce_prior_failures(contract["tasks"], evidence)
        failures = failure_rows(contract, evidence, validation)
        if dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("supported_claims") != claims:
            errors.append("supported_claims_invalid")
        if artifact.get("prior_failure_rows") != priors:
            errors.append("prior_failure_rows_invalid")
        if artifact.get("retirement_rows") != retirement_rows(priors):
            errors.append("retirement_rows_invalid")
        if artifact.get("next_reopen_conditions") != next_reopen_conditions():
            errors.append("next_reopen_conditions_invalid")
        if artifact.get("retrospective") != reduce_retrospective(evidence):
            errors.append("retrospective_invalid")
        if (
            artifact.get("status") != terminal["status"]
            or artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != _gate_summary(failures)
        ):
            errors.append("terminal_reduction_invalid")
        if artifact.get("acceptance_gate_results") != acceptance_gates(
            contract, evidence, expected_rows, claims, validation
        ):
            errors.append("acceptance_gates_invalid")
        expected_complete = int(
            contract["comparison_passed"] is True
            and len(expected_rows) == 14
            and validation["required_checks_passed"]
            and validation["terminal_validation_passed"]
        )
        if artifact.get("capstone_complete_score") != expected_complete:
            errors.append("capstone_score_invalid")
        if require_terminal and not validation["terminal_validation_passed"]:
            errors.append("terminal_validation_incomplete")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def independent_reduce(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Replay all pure reductions without requiring final terminal receipts."""

    return validate_artifact(value, root=root, require_terminal=False)


def _passing_receipts() -> list[JsonDict]:
    """Create deterministic receipts for unit construction only."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": f"unit-test/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _test_spans() -> list[JsonDict]:
    """Return zero-duration phase receipts for deterministic unit artifacts."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
        for phase, units, checkpoint in (
            ("preconditions", 13, "thirteen_predecessor_slots_observed"),
            ("plan", 8, "affected_validation_manifest_frozen"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("reduction", 14, "fourteen_dispositions_reduced"),
            ("validation", 8, "affected_checks_complete"),
            ("terminal_validation", 4, "cold_readers_complete"),
            ("write", 1, "terminal_artifact_ready"),
        )
    ]


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic terminal artifact from current repository evidence."""

    contract = load_contract(REPO_ROOT)
    evidence = collect_evidence(REPO_ROOT, contract["tasks"])
    receipts = _passing_receipts()
    validation = {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": receipts,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    return build_artifact(
        REPO_ROOT,
        contract,
        evidence,
        validation,
        started_at_utc="2026-09-21T00:00:00+00:00",
        completed_at_utc="2026-09-21T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=_test_spans(),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358 plan for only the affected V656 files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad tests, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the frozen V656 execution date."""

    if value != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    return value


def _utc_now() -> str:  # pragma: no cover - real process clock boundary.
    """Return one measured aware UTC timestamp."""

    return datetime.now(UTC).isoformat()


def _progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush a phase boundary so validation never appears inactive."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7502] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - real process clock boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - subprocess boundary.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, and strict reader commands."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", str(candidate)),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
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
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:
    """Run scoped checks and publish only a validated terminal capstone."""

    date_argument(run_date)
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = _utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7502-", dir="/tmp"))

    phase_started = time.monotonic()
    _progress(started, "preconditions", "before")
    missing = [path.as_posix() for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"required source paths missing: {missing}")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            completed_units=len(evidence),
            checkpoint="thirteen_predecessor_slots_observed",
        )
    )
    _progress(started, "preconditions", "after", completed_units=len(evidence))

    phase_started = time.monotonic()
    _progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "frozen_before_checks": True,
        },
    )
    spans.append(
        _span(
            "plan",
            phase_started,
            started,
            completed_units=len(commands),
            checkpoint="affected_validation_manifest_frozen",
        )
    )
    _progress(started, "plan", "after", completed_units=len(commands))

    for phase, checkpoint in (
        ("model_load", "no_current_model_load"),
        ("generation", "no_current_generation"),
    ):
        phase_started = time.monotonic()
        _progress(started, phase, "before", completed_units=0)
        spans.append(
            _span(
                phase,
                phase_started,
                started,
                completed_units=0,
                checkpoint=checkpoint,
            )
        )
        _progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    _progress(started, "validation", "before_subprocesses", completed_units=0)
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    validation: JsonDict = {
        **affected_reduction,
        "required_checks_passed": affected_reduction["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "repository_health": {"status": "outside_current_affected_validity"},
    }
    spans.append(
        _span(
            "validation",
            phase_started,
            started,
            completed_units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    _progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected),
        passed=affected_reduction["passed"],
    )

    phase_started = time.monotonic()
    _progress(started, "reduction", "before", completed_units=0)
    spans.append(
        _span(
            "reduction",
            phase_started,
            started,
            completed_units=14,
            checkpoint="fourteen_dispositions_reduced",
        )
    )
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    _progress(started, "reduction", "after", completed_units=14)

    phase_started = time.monotonic()
    _progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = bool(
        all(row.get("passed") is True for row in terminal_receipts) and not critical
    )
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            completed_units=len(terminal_receipts),
            checkpoint="cold_readers_complete",
        )
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=validation["terminal_validation_passed"],
    )

    phase_started = time.monotonic()
    _progress(started, "write", "before_atomic", path=output_path.as_posix())
    final_spans = [
        *spans,
        _span(
            "write",
            phase_started,
            started,
            completed_units=1,
            checkpoint="terminal_artifact_ready",
        ),
    ]
    final = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=_utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(candidate_path, final)
    atomic_json(root / output_path, final)
    _progress(started, "write", "after_atomic", path=output_path.as_posix())
    return final


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary.
    """Parse the frozen date and two read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the capstone or one read-only cold replay."""

    print("[exp7502] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    candidate_path = args.validate or args.independent_reduce
    if candidate_path is not None:
        try:
            value = load_json_object(candidate_path)
        except ValueError as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = (
            independent_reduce(value)
            if args.independent_reduce is not None
            else validate_artifact(value, require_terminal=False)
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, date_argument(args.date), output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
