"""Reduce the V651 thirteen-task capstone without running new science.

This module authenticates already-written evidence, keeps invalidity distinct
from absence and efficacy, and emits the terminal V651 disposition.  It does
not load a model, train an energy head, touch a board, or promote a result.

Spec refs: REQ-REPORT-7433 and SCENARIO-REPORT-7433-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.experiment_7421_v651_contract_ingestion import (
    load_contract_audit,
    load_yaml,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from scripts.publication_gate import evaluate as evaluate_publication_gates


JsonDict = dict[str, Any]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7433-capstone"
SCHEMA = "carnot.exp7433.v651.capstone.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7433_v651_capstone.json")
RAW_DIR = Path("results/raw/experiment_7433_v651_capstone")
MODULE_PATH = Path("python/carnot/experiment_7433_v651_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7433_v651_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7433_v651_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
PUBLICATION_GATE_PATH = Path("scripts/publication_gate.py")
PROOF_MEMORY_PATH = Path("results/experiment_7418_v650_revision_memory.json")

EXPECTED_TASK_IDS = (
    "exp7421-contract-ingestion",
    "exp7422-runtime-ownership",
    "exp7423-annotated-protocol",
    "exp7424-arc-receipt-boundary",
    "exp7425-spline-prototype",
    "exp7426-static-decisions",
    "exp7427-randomized-feedback",
    "exp7428-decision-audit",
    "exp7429-anchored-capture",
    "exp7430-extraction-audit",
    "exp7431-arc-live-sentinel",
    "exp7432-update-placement",
    EXPERIMENT_ID,
)
REQUIRED_SCIENCE_TASKS = (
    "exp7426-static-decisions",
    "exp7427-randomized-feedback",
    "exp7428-decision-audit",
    "exp7429-anchored-capture",
    "exp7430-extraction-audit",
)
CLAIM_BRANCHES = (
    "human_label_protocol",
    "spline_equivalence",
    "static_decisions",
    "online_learning",
    "independent_audit",
    "qwen_extraction",
    "extraction_audit",
    "arc_reachability",
    "host_update_cost",
    "kv260",
    "gatemate",
    "polarfire",
)
CONTINUATION_DECISIONS = (
    "preserve-completed-null",
    "preserve-circular-boundary",
    "continue-with-measured-cause",
    "wait-for-prerequisite",
    "remain-graduated",
    "retire-unchanged-mechanism",
)
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
FALLBACK_EVIDENCE_PATHS = {
    "exp7430-extraction-audit": Path("results/experiment_7430_extraction_audit.json")
}
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:  # pragma: no cover - authentic runtime boundary.
    """Return an actual UTC timestamp for an experiment boundary."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public E2E progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush a measured progress event before and after each phase."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7433] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json_object(path: Path) -> JsonDict:
    """Read one JSON object and reject absent, malformed, or list-shaped bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _numeric_experiment_id(value: object) -> int | None:
    """Return the first numeric experiment identifier from a producer label."""

    if isinstance(value, int):
        return value
    match = re.search(r"(?:exp|experiment[_-]?)(\d+)", str(value), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _terminal_status(value: Mapping[str, Any]) -> bool:
    """Accept only completed or externally blocked terminal evidence."""

    text = str(value.get("honest_verdict") or value.get("status") or "")
    return text.startswith(("complete", "blocked", "success", "passed", "shipped"))


def load_contract(root: Path) -> JsonDict:
    """Load exact active YAML while preserving the stale Markdown comparison."""

    roadmap = load_yaml(root / ROADMAP_PATH)
    tasks = roadmap.get("tasks")
    if roadmap.get("milestone") != MILESTONE or not isinstance(tasks, list):
        raise ValueError("active V651 roadmap required")
    observed = [row.get("id") for row in tasks if isinstance(row, Mapping)]
    if observed != list(EXPECTED_TASK_IDS):
        raise ValueError(f"active task order mismatch: {observed}")
    if any(
        not isinstance(row.get("deliverable"), str) or row.get("milestone") != MILESTONE
        for row in tasks
    ):
        raise ValueError("active task path or milestone missing")
    comparison = load_contract_audit(root)
    return {
        "milestone": MILESTONE,
        "title": roadmap.get("milestone_title"),
        "markdown_path": str(roadmap.get("milestone_doc")),
        "tasks": deepcopy(tasks),
        "comparison": comparison,
    }


def _compare_gate(operator: str, observed: Any, expected: Any) -> bool:
    """Evaluate only operators declared by the active roadmap."""

    if operator == "==":
        return observed == expected
    if operator == ">=":
        return isinstance(observed, (int, float)) and observed >= expected
    if operator == "in":
        return observed in expected if isinstance(expected, list) else False
    return False


def _required_validation(payload: Mapping[str, Any]) -> bool:
    """Reduce explicit validation gates without treating efficacy misses as defects."""

    direct = payload.get("required_checks_passed")
    if isinstance(direct, bool):
        return direct
    receipts = payload.get("validation_receipts")
    if isinstance(receipts, list):
        required = [
            row for row in receipts if isinstance(row, Mapping) and row.get("required") is True
        ]
        if required and not all(row.get("passed") is True for row in required):
            return False
    gates = payload.get("acceptance_gate_results")
    if isinstance(gates, list):
        validation_names = {
            "required_validation",
            "required_validation_passed",
            "affected_validation",
            "terminal_readers",
            "terminal_validation",
        }
        selected = [
            row
            for row in gates
            if isinstance(row, Mapping) and row.get("check") in validation_names
        ]
        if selected:
            return all(row.get("passed") is True for row in selected)
    return True


def _missing_evidence(task: Mapping[str, Any], declared: Path) -> JsonDict:
    """Represent unavailable external bytes as blocked, never partial."""

    return {
        "task_id": str(task.get("id")),
        "declared_path": declared.as_posix(),
        "source_path": declared.as_posix(),
        "source_kind": "missing",
        "authenticated": False,
        "available": False,
        "valid": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_missing_declared_evidence",
        "flagged_adversarial": False,
        "required_validation_passed": False,
        "raw_rows_available": False,
        "raw_row_count": 0,
        "payload": {},
    }


def load_evidence_slot(root: Path, task: Mapping[str, Any]) -> JsonDict:
    """Authenticate one terminal artifact, including the Exp7430 pre-gate form."""

    task_id = str(task.get("id"))
    declared = Path(str(task.get("deliverable") or ""))
    actual = declared
    if not (root / actual).is_file() and task_id in FALLBACK_EVIDENCE_PATHS:
        actual = FALLBACK_EVIDENCE_PATHS[task_id]
    path = root / actual
    if not path.is_file():
        return _missing_evidence(task, declared)
    payload = load_json_object(path)
    expected_number = _numeric_experiment_id(task_id)
    observed_number = _numeric_experiment_id(
        payload.get("experiment_id") or payload.get("experiment")
    )
    if task_id == "exp7430-extraction-audit":
        source = root / "results/experiment_7429_v651_anchored_capture.json"
        expected_source_hash = sha256_file(source) if source.is_file() else None
        gates = payload.get("gates_evaluated") or []
        authenticated = (
            observed_number == expected_number
            and payload.get("schema") == "blocked_gate_check_v1"
            and payload.get("status") == "blocked"
            and payload.get("failed_upstream") == "exp7429-anchored-capture"
            and payload.get("failed_field") == "extraction_capture_complete_score"
            and payload.get("failed_observed") == 0
            and payload.get("failed_expected") == 1
            and payload.get("failed_evidence_sha256") == expected_source_hash
            and isinstance(gates, list)
        )
        return {
            "task_id": task_id,
            "declared_path": declared.as_posix(),
            "source_path": actual.as_posix(),
            "source_kind": "structured_pre_gate",
            "authenticated": authenticated,
            "available": False,
            "valid": authenticated,
            "verdict_class": "blocked",
            "honest_verdict": str(payload.get("honest_verdict")),
            "flagged_adversarial": False,
            "required_validation_passed": authenticated,
            "raw_rows_available": False,
            "raw_row_count": 0,
            "sha256": sha256_file(path),
            "payload": payload,
        }

    verdict = payload.get("verdict_class")
    rows = payload.get("rows")
    authenticated = (
        observed_number == expected_number
        and payload.get("milestone") == MILESTONE
        and verdict in CLOSED_VERDICTS
        and isinstance(payload.get("flagged_adversarial"), bool)
        and _terminal_status(payload)
    )
    validation_passed = _required_validation(payload)
    return {
        "task_id": task_id,
        "declared_path": declared.as_posix(),
        "source_path": actual.as_posix(),
        "source_kind": "terminal_artifact",
        "authenticated": authenticated,
        "available": authenticated,
        "valid": authenticated and validation_passed and verdict != "disqualified",
        "verdict_class": verdict if verdict in CLOSED_VERDICTS else "disqualified",
        "honest_verdict": str(payload.get("honest_verdict") or payload.get("status")),
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "required_validation_passed": validation_passed,
        "raw_rows_available": isinstance(rows, list),
        "raw_row_count": len(rows) if isinstance(rows, list) else 0,
        "sha256": sha256_file(path),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read the twelve predecessor slots in contract order."""

    rows: dict[str, JsonDict] = {}
    for task in tasks[:-1]:
        task_id = str(task["id"])
        rows[task_id] = load_evidence_slot(root, task)
    return rows


def _claim(
    evidence: Mapping[str, JsonDict],
    task_id: str,
    *,
    completion_score: int,
    benefit_score: int,
    authority: str,
    limitations: Sequence[str],
    **extra: Any,
) -> JsonDict:
    """Build one branch row without allowing another branch to promote it."""

    source = evidence[task_id]
    row = {
        "authority_task": task_id,
        "authority": authority,
        "available": source["available"],
        "valid": source["valid"],
        "completion_score": completion_score,
        "benefit_score": benefit_score,
        "verdict_class": source["verdict_class"],
        "flagged_adversarial": source["flagged_adversarial"],
        "verifier_is_oracle": bool(source["payload"].get("verifier_is_oracle", False)),
        "limitations": list(limitations),
    }
    row.update(extra)
    return row


def reduce_claim_matrix(evidence: Mapping[str, JsonDict]) -> dict[str, JsonDict]:
    """Reduce the twelve independent V651 claim boundaries."""

    p23 = evidence["exp7423-annotated-protocol"]["payload"]
    p25 = evidence["exp7425-spline-prototype"]["payload"]
    p26 = evidence["exp7426-static-decisions"]["payload"]
    p27 = evidence["exp7427-randomized-feedback"]["payload"]
    p28 = evidence["exp7428-decision-audit"]["payload"]
    p29 = evidence["exp7429-anchored-capture"]["payload"]
    p31 = evidence["exp7431-arc-live-sentinel"]["payload"]
    p32 = evidence["exp7432-update-placement"]["payload"]
    boards = {
        str(row.get("board")): row for row in p32.get("board_rows", []) if isinstance(row, Mapping)
    }
    matrix: dict[str, JsonDict] = {}
    matrix["human_label_protocol"] = _claim(
        evidence,
        "exp7423-annotated-protocol",
        completion_score=int(p23.get("annotated_protocol_ready_score", 0)),
        benefit_score=0,
        authority=str(p23.get("label_authority") or "human source-support labels"),
        limitations=[str(p23.get("label_authority_limit"))],
        semantic_truth_claimed=False,
        implicit_true_semantics=p23.get("implicit_true_semantics"),
    )
    matrix["spline_equivalence"] = _claim(
        evidence,
        "exp7425-spline-prototype",
        completion_score=int(p25.get("spline_prototype_ready_score", 0)),
        benefit_score=0,
        authority="analytic fixed-basis parity fixtures",
        limitations=["same fixed basis only", "no general EBM superiority"],
        general_ebm_superiority_claimed=False,
        dense_equivalence=p25.get("dense_equivalence") or p25.get("parity_summary"),
    )
    matrix["static_decisions"] = _claim(
        evidence,
        "exp7426-static-decisions",
        completion_score=int(p26.get("decision_capture_complete_score", 0)),
        benefit_score=int(p26.get("decision_value_score", 0)),
        authority="human-label static decision experiment",
        limitations=["zero registered certified coverage", "source support is not world truth"],
    )
    matrix["online_learning"] = _claim(
        evidence,
        "exp7427-randomized-feedback",
        completion_score=int(p27.get("online_capture_complete_score", 0)),
        benefit_score=int(p27.get("online_value_score", 0)),
        authority="randomized partial-feedback replay",
        limitations=["prospective replay, not deployed feedback", "selection bias needs IPW"],
        partial_feedback_bias_reported=bool(p27.get("condition_reports")),
    )
    matrix["independent_audit"] = _claim(
        evidence,
        "exp7428-decision-audit",
        completion_score=min(
            int(p28.get("static_audit_complete_score", 0)),
            int(p28.get("online_audit_complete_score", 0)),
        ),
        benefit_score=0,
        authority="independent raw-row replay and leakage attacks",
        limitations=["reproduces nulls; does not create benefit"],
    )
    matrix["qwen_extraction"] = _claim(
        evidence,
        "exp7429-anchored-capture",
        completion_score=int(p29.get("extraction_capture_complete_score", 0)),
        benefit_score=int(p29.get("extraction_value_score", 0)),
        authority="owned Qwen anchored capture",
        limitations=["internal artifact validation failed", "zero official calls completed"],
    )
    matrix["extraction_audit"] = _claim(
        evidence,
        "exp7430-extraction-audit",
        completion_score=0,
        benefit_score=0,
        authority="structured conductor pre-gate",
        limitations=["blocked by Exp7429 capture score zero", "no audit rows exist"],
    )
    matrix["arc_reachability"] = _claim(
        evidence,
        "exp7431-arc-live-sentinel",
        completion_score=int(p31.get("arc_sentinel_capture_complete_score", 0)),
        benefit_score=int(p31.get("live_efficacy_score", 0)),
        authority="two-episode live first-action sentinel",
        limitations=["reachability only", "no hidden leaderboard or solve inference"],
        hidden_performance_claimed=False,
    )
    matrix["host_update_cost"] = _claim(
        evidence,
        "exp7432-update-placement",
        completion_score=int(p32.get("update_placement_complete_score", 0)),
        benefit_score=int(p32.get("update_placement_value_score", 0)),
        authority="host whole-service update timing",
        limitations=["host arithmetic only", "no board latency or power measurement"],
        hardware_benefit_claimed=False,
    )
    for key, board in (("kv260", "KV260"), ("gatemate", "GateMate"), ("polarfire", "PolarFire")):
        source = boards.get(board, {})
        matrix[key] = _claim(
            evidence,
            "exp7432-update-placement",
            completion_score=1,
            benefit_score=int(source.get("hardware_value_score", 0)),
            authority=f"Exp7432 {board} disposition",
            limitations=[str(source.get("error") or source.get("exact_next_prerequisite"))],
            terminal_state=source.get("terminal_state"),
            exact_next_prerequisite=source.get("exact_next_prerequisite"),
            hardware_benefit_claimed=False,
        )
    return matrix


def diagnostic_summary(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """Expose the required diagnostic and cost evidence without extrapolation."""

    static = evidence["exp7426-static-decisions"]["payload"]
    online = evidence["exp7427-randomized-feedback"]["payload"]
    audit = evidence["exp7428-decision-audit"]["payload"]
    update = evidence["exp7432-update-placement"]["payload"]
    parity = [row for row in static.get("spline_parity_rows", []) if isinstance(row, Mapping)]
    conditions: dict[str, JsonDict] = {}
    for name in ("full_source", "source_masked", "source_swapped"):
        selected = [row for row in parity if row.get("condition") == name]
        conditions[name] = {
            "rows": len(selected),
            "all_passed": bool(selected) and all(row.get("passed") is True for row in selected),
            "max_probability_gap": max(
                (float(row.get("max_probability_gap", 0.0)) for row in selected), default=None
            ),
        }
    domains = static.get("domain_level_support", {}).get("dense_spline_logistic", {})
    attacks = [
        row
        for row in audit.get("leakage_attack_rows", [])
        if isinstance(row, Mapping) and row.get("attack") == "mislabeled_implicit_true"
    ]
    board_rows = {
        str(row.get("board")): {
            "terminal_state": row.get("terminal_state"),
            "exact_next_prerequisite": row.get("exact_next_prerequisite"),
        }
        for row in update.get("board_rows", [])
        if isinstance(row, Mapping)
    }
    return {
        "static_source_conditions": conditions,
        "dense_basis_equivalence": {
            "same_fixed_basis_only": True,
            "rows": len(parity),
            "all_passed": bool(parity) and all(row.get("passed") is True for row in parity),
        },
        "partial_feedback_selection_bias": {
            "known_propensity_rows": len(online.get("feedback_event_rows", [])),
            "condition_reports": len(online.get("condition_reports", [])),
            "moving_block_intervals": online.get("paired_moving_block_intervals"),
            "guarantees_claimed": False,
        },
        "implicit_true_sensitivity": {
            "semantics": "unsupported by provided source; not false in the world",
            "attack_rows": deepcopy(attacks),
        },
        "per_domain_support": deepcopy(domains),
        "complete_service_costs": deepcopy(update.get("timing_summary", [])),
        "amdahl_analysis": deepcopy(update.get("amdahl_analysis", {})),
        "board_dispositions": board_rows,
    }


def _science_failure(task_id: str, row: Mapping[str, Any], category: str, check: str) -> JsonDict:
    """Name every operand required to reproduce one science failure."""

    if category == "required_science_invalid":
        field = "valid"
        expected: Any = True
        observed = row.get("valid")
    else:
        field = "available"
        expected = True
        observed = row.get("available")
    return {
        "upstream": task_id,
        "path": row.get("source_path"),
        "check": check,
        "category": category,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
    }


def classify_terminal(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
) -> JsonDict:
    """Apply invalid-before-unavailable precedence to required science."""

    failures: list[JsonDict] = []
    for task_id in REQUIRED_SCIENCE_TASKS:
        row = evidence[task_id]
        if row.get("available") and (
            not row.get("valid")
            or row.get("flagged_adversarial") is True
            or row.get("verdict_class") == "disqualified"
        ):
            failures.append(
                _science_failure(task_id, row, "required_science_invalid", "core_validity")
            )
    for task_id in REQUIRED_SCIENCE_TASKS:
        row = evidence[task_id]
        if not row.get("available"):
            failures.append(
                _science_failure(task_id, row, "required_science_absent", "core_availability")
            )
    comparison = contract.get("comparison")
    comparison_passed = isinstance(comparison, Mapping) and comparison.get("passed") is True
    if not comparison_passed:
        failures.append(
            {
                "upstream": "v651-contract-authorities",
                "path": f"{ROADMAP_PATH.as_posix()} + {DESIGN_PATH.as_posix()}",
                "check": "authority_equivalence",
                "category": "contract_authority",
                "field": "comparison.passed",
                "operator": "==",
                "expected": True,
                "observed": False,
            }
        )
    if validation.get("required_checks_passed") is not True:
        failures.append(
            {
                "upstream": EXPERIMENT_ID,
                "path": TEST_PATH.as_posix(),
                "check": "affected_validation",
                "category": "required_validation",
                "field": "required_checks_passed",
                "operator": "==",
                "expected": True,
                "observed": validation.get("required_checks_passed"),
            }
        )
    invalid = any(
        row["category"] in {"required_science_invalid", "contract_authority", "required_validation"}
        for row in failures
    )
    absent = any(row["category"] == "required_science_absent" for row in failures)
    if invalid:
        verdict_class = "disqualified"
        verdict = "complete_disqualified_required_v651_science_with_thirteen_dispositions"
    elif absent:
        verdict_class = "blocked"
        verdict = "complete_blocked_required_v651_science_with_thirteen_dispositions"
    else:
        verdict_class = "null"
        verdict = "complete_null_v651_capstone_with_thirteen_dispositions"
    return {
        "status": verdict,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "passed": not failures,
            "failure_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
        },
    }


def retirement_rows(tasks: Sequence[Mapping[str, Any]], current_verdict: str) -> list[JsonDict]:
    """Retire only an exact repeated verdict under its declared mechanism scope."""

    rows: list[JsonDict] = []
    for task in tasks:
        for prior in task.get("prior_failures") or []:
            previous = str(prior.get("verdict"))
            same = previous == current_verdict
            eligible = same and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task.get("id"),
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current_verdict,
                    "same_exact_verdict": same,
                    "same_mechanism_scope": True,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "decision": (
                        "retire-unchanged-mechanism" if eligible else "continue-with-measured-cause"
                    ),
                    "addressed_by": prior.get("addressed_by"),
                }
            )
    return rows


def continuation_rows(
    claims: Mapping[str, JsonDict], retirements: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Choose one bounded next disposition per independent claim."""

    capstone_retired = any(
        row.get("prior_experiment_id") == "exp7420-capstone"
        and row.get("decision") == "retire-unchanged-mechanism"
        for row in retirements
    )
    rows: list[JsonDict] = []
    for name, claim in claims.items():
        verdict = claim["verdict_class"]
        if name in {"extraction_audit", "gatemate"}:
            decision = "wait-for-prerequisite"
        elif name == "qwen_extraction":
            decision = "continue-with-measured-cause"
        elif name in {"kv260", "polarfire"}:
            decision = "remain-graduated"
        elif verdict == "circular_positive":
            decision = "preserve-circular-boundary"
        else:
            decision = "preserve-completed-null"
        if capstone_retired and name == "qwen_extraction":
            decision = "retire-unchanged-mechanism"
        rows.append(
            {
                "claim": name,
                "decision": decision,
                "changed_cause_or_prerequisite": claim["limitations"][0],
            }
        )
    return rows


def scope_reduction_compliance(evidence: Mapping[str, JsonDict]) -> JsonDict:
    """State floor coverage and preserve both proof-memory boundaries."""

    return {
        "standing_floors": {
            "arc": "complete_reachability_only_no_hidden_performance",
            "calibrated_decision": "complete_null_no_registered_static_benefit",
            "self_learning": "complete_null_no_registered_online_benefit",
            "sota_ingestion": "disqualified_capture_then_blocked_audit",
            "hardware": "host_cost_null_three_board_dispositions_preserved",
        },
        "unresolved_obligations": [
            "valid complete Qwen capture and independent audit",
            "registered static and online decision benefit",
            "physical GateMate state change before another attempt",
            "hardware latency and power measurements for hardware value",
        ],
        "v649_proof_memory_chain": {
            "decision": "remain-retired",
            "requires_chain_reopened": False,
        },
        "v650_proof_memory": {
            "decision": "defer-until-changed-cost-profile",
            "upper_cost_ratio": 3.7068,
            "recorded_upper_cost_ratio": 3.706797926244237,
            "baseline": "persistent_incremental_solver",
            "source_path": PROOF_MEMORY_PATH.as_posix(),
            "source_sha256": (
                sha256_file(REPO_ROOT / PROOF_MEMORY_PATH)
                if (REPO_ROOT / PROOF_MEMORY_PATH).is_file()
                else None
            ),
        },
        "roadmap_activated": False,
        "publication_performed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
    }


def publication_gate_row(result: Mapping[str, Any]) -> JsonDict:
    """Retain G1-G4 exactly while making their headline scope explicit."""

    return {
        "headline_scope": "FoVer dual-condition AUROC",
        "certifies_v651": False,
        "paper_ready": result.get("paper_ready"),
        "gates": deepcopy(result.get("gates")),
        "unmet_gates": deepcopy(result.get("unmet_gates")),
        "note": result.get("note"),
    }


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, JsonDict],
    terminal: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build thirteen ordered dispositions including the current capstone."""

    rows: list[JsonDict] = []
    for task in tasks[:-1]:
        source = evidence[str(task["id"])]
        rows.append(
            {
                "task_id": task["id"],
                "declared_path": task["deliverable"],
                "observed_path": source["source_path"],
                "source_kind": source["source_kind"],
                "authenticated": source["authenticated"],
                "available": source["available"],
                "valid": source["valid"],
                "raw_rows_available": source["raw_rows_available"],
                "raw_row_count": source["raw_row_count"],
                "honest_verdict": source["honest_verdict"],
                "verdict_class": source["verdict_class"],
                "flagged_adversarial": source["flagged_adversarial"],
            }
        )
    rows.append(
        {
            "task_id": EXPERIMENT_ID,
            "declared_path": tasks[-1]["deliverable"],
            "observed_path": RESULT_PATH.as_posix(),
            "source_kind": "current_capstone_checks",
            "authenticated": True,
            "available": True,
            "valid": terminal["verdict_class"] != "disqualified",
            "raw_rows_available": True,
            "raw_row_count": 13,
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": any(
                row.get("flagged_adversarial") is True for row in evidence.values()
            ),
            "required_checks_passed": validation.get("required_checks_passed"),
        }
    )
    return rows


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
) -> dict[str, JsonDict]:
    """Bind exact authorities, code, proof memory, and producer artifacts."""

    paths = {
        "research-roadmap": ROADMAP_PATH,
        "markdown-contract": DESIGN_PATH,
        "research-reporting-spec": SPEC_PATH,
        "publication-gate": PUBLICATION_GATE_PATH,
        "proof-memory-v650": PROOF_MEMORY_PATH,
        "current-module": MODULE_PATH,
        "current-wrapper": WRAPPER_PATH,
        "current-test": TEST_PATH,
    }
    rows = {
        name: {"path": path.as_posix(), "sha256": sha256_file(root / path)}
        for name, path in paths.items()
    }
    for task_id, source in evidence.items():
        path = Path(str(source["source_path"]))
        rows[task_id] = {
            "path": path.as_posix(),
            "declared_path": source["declared_path"],
            "sha256": sha256_file(root / path),
            "source_kind": source["source_kind"],
            "verdict_class": source["verdict_class"],
            "flagged_adversarial": source["flagged_adversarial"],
            "authenticated": source["authenticated"],
        }
    rows["contract-comparison"] = {
        "path": f"{ROADMAP_PATH.as_posix()} + {DESIGN_PATH.as_posix()}",
        "sha256": canonical_hash(contract["comparison"]),
        "source_kind": "canonical_reduction",
    }
    return rows


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload every file-backed hash and reject malformed source rows."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, Mapping):
        return False
    for row in rows.values():
        if not isinstance(row, Mapping):
            return False
        path_value = row.get("path")
        expected = row.get("sha256")
        if row.get("source_kind") == "canonical_reduction":
            try:
                current = load_contract(root)["comparison"]
            except (OSError, ValueError):
                return False
            if expected != canonical_hash(current):
                return False
            continue
        if not isinstance(path_value, str) or " + " in path_value:
            return False
        path = root / path_value
        if not path.is_file() or expected != sha256_file(path):
            return False
    return True


def _historical_sidecars(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reference prior model or small-head work without inheriting its counters."""

    rows: list[JsonDict] = []
    for task_id, source in evidence.items():
        payload = source["payload"]
        kinds: list[str] = []
        if payload.get("model_invoked") is True:
            kinds.append("historical_model_invocation")
        if payload.get("small_ebm_training"):
            kinds.append("historical_small_ebm_training")
        for kind in kinds:
            rows.append(
                {
                    "task_id": task_id,
                    "kind": kind,
                    "path": source["source_path"],
                    "sha256": source.get("sha256"),
                    "counts_as_current_invocation": False,
                }
            )
    return rows


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep completion, validity, benefit, safety, and promotion checks explicit."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build current checks without equating completion and benefit."""

    contract_passed = contract["comparison"].get("passed") is True
    core_valid = all(evidence[task]["valid"] for task in REQUIRED_SCIENCE_TASKS)
    core_available = all(evidence[task]["available"] for task in REQUIRED_SCIENCE_TASKS)
    all_authenticated = all(row["authenticated"] for row in evidence.values())
    return [
        _gate(
            "thirteen_task_dispositions",
            "completion",
            "==",
            13,
            len(dispositions),
            len(dispositions) == 13,
            "Every task receives an honest disposition independent of benefit.",
        ),
        _gate(
            "contract_authorities_agree",
            "contract_authority",
            "==",
            True,
            contract_passed,
            contract_passed,
            "The stale Markdown authority remains a failed current check.",
        ),
        _gate(
            "predecessor_sources_authenticated",
            "evidence_validity",
            "==",
            True,
            all_authenticated,
            all_authenticated,
            "Conductor status alone never substitutes for authenticated bytes.",
        ),
        _gate(
            "core_required_science_valid",
            "scientific_validity",
            "==",
            True,
            core_valid,
            core_valid,
            "Invalid core evidence disqualifies before absence is reduced.",
        ),
        _gate(
            "core_required_science_available",
            "scientific_availability",
            "==",
            True,
            core_available,
            core_available,
            "External absence is blocked and never retryable partial work.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            "==",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "Only the frozen affected plan controls current implementation validity.",
        ),
        _gate(
            "promotion_zero",
            "promotion",
            "==",
            0,
            0,
            True,
            "Aggregation cannot publish, roll out, or update generator weights.",
        ),
    ]


def zero_test_phase_spans() -> list[JsonDict]:
    """Return complete deterministic phase rows for unit construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "heartbeat_count": 0,
            "checkpoint": checkpoint,
        }
        for phase, checkpoint in (
            ("preconditions", "twelve_sources_authenticated"),
            ("plan", "affected_plan_frozen"),
            ("load", "no_current_model_load"),
            ("generate", "no_current_generation"),
            ("validate", "affected_checks_complete"),
            ("reduce", "twelve_claims_reduced"),
            ("terminal_validation", "cold_readers_complete"),
            ("write", "terminal_artifact_ready"),
        )
    ]


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain each ordinary top-level field separately from its value."""

    special = {
        "schema": "Versioned plain top-level schema, experiment_id, milestone and terminal status.",
        "run_date": "Use 20260919, with actual UTC start/end and monotonic timing.",
        "preconditions_checked": "Exact resource/path/identity checks and observed values before dependent work.",
        "MODEL_SPECS": "Every current LLM uses unsloth/Qwen3.8-27B-GGUF; empty for no current LLM.",
        "model_invoked": "True only for actual current attempted model use; archives are false.",
        "invocation_counts": "Actual owned current load and generation outcomes only.",
        "inference_substrate_class": "Actual current compute class, never historical inference.",
        "execution_venue": "Closed venue string host; CPU, CUDA, and boards stay in details.",
        "rows": "Every comparative accounting unit, including blocked and invalid dispositions.",
        "promotion_score": "Always zero; no publication, rollout, or generator-weight update.",
        "capstone_complete_score": "One only when thirteen dispositions and every current capstone check pass.",
        "task_dispositions": "Exactly thirteen task IDs in contract order with declared paths.",
        "claim_matrix": "Authority, availability, validity, completion, benefit, and limits per branch.",
        "publication_gates": "Stable G1-G4 retain their FoVer headline scope and do not certify V651.",
    }
    return {key: special.get(key, f"Plain terminal evidence for {key}.") for key in keys}


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind protocol, source identity, raw rows, checks, and branch decisions."""

    excluded = {
        "reproducibility_checksum",
        "field_principles",
        "started_at_utc",
        "completed_at_utc",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "duration_s",
        "validation_duration_s",
        "cold_start_duration_s",
        "model_duration_s",
        "phase_spans",
    }
    stable = {key: value for key, value in artifact.items() if key not in excluded}
    return canonical_hash(stable)


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    validation: Mapping[str, Any],
    publication: Mapping[str, Any],
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    flagged_adversarial: bool | None = None,
) -> JsonDict:
    """Build the schema-complete capstone from authenticated source rows."""

    terminal = classify_terminal(contract, evidence, validation)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal, validation)
    claims = reduce_claim_matrix(evidence)
    retirements = retirement_rows(contract["tasks"], terminal["honest_verdict"])
    gates = _acceptance_gates(contract, evidence, validation, dispositions)
    current_checks_passed = all(gate["passed"] for gate in gates)
    source_hashes = _source_hashes(root, contract, evidence)
    observed_flag = any(row["flagged_adversarial"] for row in evidence.values())
    if flagged_adversarial is not None:
        observed_flag = observed_flag or flagged_adversarial
    start_ns = 0
    end_ns = max(0, int(duration_s * 1_000_000_000))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 4,
        "run_date": RUN_DATE,
        "status": terminal["status"],
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": observed_flag,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": start_ns,
        "ended_monotonic_ns": end_ns,
        "duration_s": float(duration_s),
        "validation_duration_s": sum(
            float(row.get("duration_s", 0.0))
            for row in validation.get("validation_receipts", [])
            if isinstance(row, Mapping)
        ),
        "cold_start_duration_s": 0.0,
        "model_duration_s": 0.0,
        "phase_spans": deepcopy(list(phase_spans)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "current_compute": "host JSON/YAML reduction",
            "current_llm_operations": 0,
            "historical_compute_inherited": False,
        },
        "execution_venue": "host",
        "random_seed": None,
        "preconditions_checked": [
            {
                "check": "active_yaml_task_order",
                "upstream": ROADMAP_PATH.as_posix(),
                "artifact_field": "tasks[].id",
                "expected": list(EXPECTED_TASK_IDS),
                "observed": [row["id"] for row in contract["tasks"]],
                "passed": True,
            },
            {
                "check": "markdown_authority_equivalence",
                "upstream": DESIGN_PATH.as_posix(),
                "artifact_field": "comparison.passed",
                "expected": True,
                "observed": contract["comparison"].get("passed"),
                "passed": contract["comparison"].get("passed") is True,
            },
            *[
                {
                    "check": "source_authentication",
                    "upstream": task_id,
                    "path": row["source_path"],
                    "artifact_field": "identity+terminal+hash",
                    "expected": True,
                    "observed": row["authenticated"],
                    "passed": row["authenticated"],
                }
                for task_id, row in evidence.items()
            ],
        ],
        "source_artifact_hashes": source_hashes,
        "historical_evidence_sidecars": _historical_sidecars(evidence),
        "rows": deepcopy(dispositions),
        "sample_size_budget": {
            "planned": 13,
            "attempted": 13,
            "completed": 13,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_groups": 13,
            "stop_rule": "one authenticated disposition for each ordered V651 task",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": terminal["gate_check_summary"],
        "verifier_is_oracle": False,
        "validation_receipts": deepcopy(validation.get("validation_receipts", [])),
        "repository_health": deepcopy(validation.get("repository_health", {})),
        "task_dispositions": dispositions,
        "claim_matrix": claims,
        "diagnostic_summary": diagnostic_summary(evidence),
        "retirement_rows": retirements,
        "continuation_rows": continuation_rows(claims, retirements),
        "scope_reduction_compliance": scope_reduction_compliance(evidence),
        "publication_gates": publication_gate_row(publication),
        "promotion_score": 0,
        "capstone_complete_score": int(len(dispositions) == 13 and current_checks_passed),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def validate_artifact(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, sources, reductions, decisions, scores, and checksum."""

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
        "model_invoked",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "duration_s",
        "phase_spans",
        "preconditions_checked",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "validation_receipts",
        "task_dispositions",
        "claim_matrix",
        "continuation_rows",
        "scope_reduction_compliance",
        "publication_gates",
        "promotion_score",
        "capstone_complete_score",
        "reproducibility_checksum",
        "field_principles",
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
    if not str(artifact.get("status")).startswith("complete"):
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_invalid")
    dispositions = artifact.get("task_dispositions")
    if (
        not isinstance(dispositions, list)
        or [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        != list(EXPECTED_TASK_IDS)
        or artifact.get("rows") != dispositions
    ):
        errors.append("task_dispositions_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    publication = artifact.get("publication_gates")
    current_publication = publication_gate_row(evaluate_publication_gates())
    if publication != current_publication:
        errors.append("publication_gates_invalid")
    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        validation = {
            "required_checks_passed": all(
                row.get("passed") is True
                for row in artifact.get("validation_receipts", [])
                if isinstance(row, Mapping) and row.get("required") is True
            ),
            "validation_receipts": artifact.get("validation_receipts", []),
        }
        if not any(
            isinstance(row, Mapping) and row.get("required") is True
            for row in artifact.get("validation_receipts", [])
        ):
            validation["required_checks_passed"] = all(
                isinstance(row, Mapping) and row.get("passed") is True
                for row in artifact.get("validation_receipts", [])
            )
        terminal = classify_terminal(contract, evidence, validation)
        expected_dispositions = _task_dispositions(
            contract["tasks"], evidence, terminal, validation
        )
        if dispositions != expected_dispositions:
            errors.append("task_dispositions_invalid")
        if artifact.get("claim_matrix") != reduce_claim_matrix(evidence):
            errors.append("claim_matrix_invalid")
        retirements = retirement_rows(contract["tasks"], terminal["honest_verdict"])
        if artifact.get("retirement_rows") != retirements:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != continuation_rows(
            reduce_claim_matrix(evidence), retirements
        ):
            errors.append("continuation_rows_invalid")
        if artifact.get("scope_reduction_compliance") != scope_reduction_compliance(evidence):
            errors.append("scope_reduction_invalid")
        if (
            artifact.get("honest_verdict") != terminal["honest_verdict"]
            or artifact.get("verdict_class") != terminal["verdict_class"]
            or artifact.get("gate_check_summary") != terminal["gate_check_summary"]
        ):
            errors.append("terminal_reduction_invalid")
        gates = _acceptance_gates(contract, evidence, validation, expected_dispositions)
        expected_score = int(
            len(expected_dispositions) == 13 and all(row["passed"] for row in gates)
        )
        if artifact.get("capstone_complete_score") != expected_score:
            errors.append("capstone_score_invalid")
    except (KeyError, OSError, TypeError, ValueError):
        errors.append("independent_reduction_failed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact) - {
        "field_principles"
    }:
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the frozen Exp7358 plan for only the current affected files."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion, broad tests, and environment drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V651 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(  # pragma: no cover - authentic entrypoint timing boundary.
    phase: str, phase_started: float, run_started: float, *, checkpoint: str
) -> JsonDict:
    """Close one monotonic phase span and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
    }


def _terminal_commands(  # pragma: no cover - capability E2E subprocesses.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7433_v651_capstone import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v);print(json.dumps({'errors':e},sort_keys=True),flush=True);"
        "raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
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


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - public E2E.
    """Run exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7433-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "preconditions", "before")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    publication = evaluate_publication_gates()
    spans.append(
        _phase_span("preconditions", point, started, checkpoint="twelve_sources_authenticated")
    )
    progress(started, "preconditions", "after", dispositions=len(evidence))

    point = time.monotonic()
    progress(started, "plan", "before")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("plan", point, started, checkpoint="affected_plan_frozen"))
    progress(started, "plan", "after", commands=len(commands))

    for phase, checkpoint in (
        ("load", "no_current_model_load"),
        ("generate", "no_current_generation"),
    ):
        point = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(_phase_span(phase, point, started, checkpoint=checkpoint))
        progress(started, phase, "after", current_llm_operations=0)

    point = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(_phase_span("validate", point, started, checkpoint="affected_checks_complete"))
    progress(started, "validate", "after_affected_subprocesses", passed=reduced["passed"])
    validation: JsonDict = {
        **reduced,
        "required_checks_passed": reduced["passed"],
        "terminal_validation_passed": False,
        "validation_receipts": affected,
        "repository_health": {
            "status": "historical_observations_retained",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
            "historical_failures": [],
        },
    }

    point = time.monotonic()
    progress(started, "reduce", "before")
    spans.append(_phase_span("reduce", point, started, checkpoint="twelve_claims_reduced"))
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "reduce", "after", verdict=candidate["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", units=len(terminal_commands))
    terminal_receipts = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts) and not critical
    validation["terminal_validation_passed"] = terminal_passed
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _phase_span("terminal_validation", point, started, checkpoint="cold_readers_complete")
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )

    point = time.monotonic()
    progress(started, "write", "before_atomic", path=RESULT_PATH.as_posix())
    final_spans = [
        *spans,
        _phase_span("write", point, started, checkpoint="terminal_artifact_ready"),
    ]
    artifact = build_artifact(
        root,
        contract,
        evidence,
        validation,
        publication,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
        flagged_adversarial=critical,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:
    """Parse the frozen date and optional cold-validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V651 capstone or cold-validate one measured candidate."""

    print("[exp7433] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    if args.validate is not None:
        try:
            value = load_json_object(args.validate)
        except ValueError as error:
            print(json.dumps({"errors": [str(error)]}, sort_keys=True), flush=True)
            return 1
        errors = validate_artifact(value)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "capstone_complete_score": artifact["capstone_complete_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
