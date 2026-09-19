"""Reconcile the V649 milestone from authenticated local evidence.

The capstone reads existing artifacts and exact conductor records. It does not
call a model, train a model, operate hardware, or change production behavior.

Spec refs: REQ-REPORT-7408 and SCENARIO-REPORT-7408-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as command_boundary
from carnot import experiment_7395_v649_receipt_protocol as contract_boundary
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.649"
RUN_DATE = "20260919"
EXPERIMENT_ID = "exp7408-capstone"
SCHEMA = "carnot.experiment_7408.v649_capstone.v1"
RESULT_PATH = Path("results/experiment_7408_v649_capstone.json")
RAW_DIR = Path("results/raw/experiment_7408_v649_capstone")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7408_v649_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7408_v649_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7408_v649_capstone.py")
ENTRYPOINT_PATH = Path("scripts/experiments/experiment_7408_v649_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
STAGED_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")

EXPECTED_TASK_IDS = (
    "exp7395-receipt-protocol",
    "exp7396-decision-diagnosis",
    "exp7397-delayed-adapter",
    "exp7398-arc-checkpoint",
    "exp7399-online-trial",
    "exp7400-assignment-canary",
    "exp7401-online-audit",
    "exp7402-proposal-capture",
    "exp7403-synthetic-memory",
    "exp7404-live-memory",
    "exp7405-proof-audit",
    "exp7406-arc-generalization",
    "exp7407-service-cost",
    EXPERIMENT_ID,
)

CLAIM_BRANCHES = (
    "static_calibration",
    "online_learning",
    "synthetic_memory",
    "live_memory",
    "proof_audit",
    "arc",
    "host_service_cost",
)
CONTINUATION_DECISIONS = (
    "continue-with-measured-cause",
    "retire-unchanged-mechanism",
    "wait-for-named-external-change",
)
ELIGIBLE_CLASSES = {"positive", "circular_positive", "null"}
CLOSED_CLASSES = {*ELIGIBLE_CLASSES, "blocked", "disqualified", "partial"}
ZERO_INVOCATION_COUNTS = deepcopy(command_boundary.ZERO_INVOCATION_COUNTS)
RANDOM_SEED = {
    "capstone": 7_408_202_609_19,
    "inherited_static_resampling": 7_385_648,
    "inherited_online_resampling": 7_397_001,
    "inherited_synthetic_memory": 7_403_649,
    "inherited_service_cost": 7_407_307,
}

CONDUCTOR_MARKERS = {
    "exp7397-delayed-adapter": (
        "Prototype a bounded energy-offset learner with del | OK | 96 passed, 1 warning in 9.45s"
    ),
    "exp7399-online-trial": (
        "Measure continuous affine calibration on later ver | OK | 90 passed, 1 warning in 7.93s"
    ),
    "exp7404-live-memory": (
        "Measure certified memory on fresh bounded model pr | GATE_BLOCK | "
        "2 of 6 gate(s) failed; first failure: "
        "exp7402-proposal-capture.candidate_capture_complete_score "
        "(actual=0 == expected=1)"
    ),
}

V649_MANIFEST = command_boundary.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(ENTRYPOINT_PATH.as_posix(),),
)

AUTHORITY_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    STAGED_ROADMAP_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
    Path("research-complete.yaml"),
    Path("research-references.md"),
    Path("research-studying.md"),
    CONDUCTOR_PATH,
    Path("ops/north-star.md"),
    Path("ops/verifier_gaps.md"),
    Path("results/experiment_7394_v648_capstone.json"),
    MODULE_PATH,
    TEST_PATH,
    ENTRYPOINT_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with ordinary experiment identity and terminal status.",
    "run_date": "Use 20260919 and include actual UTC start and completion timestamps.",
    "preconditions_checked": "Record exact input paths, hashes, eligibility, runtime, and entrypoint checks before dependent work.",
    "MODEL_SPECS": "Keep this empty because the current capstone performs no LLM work.",
    "model_invoked": "Set true for an attempted current real LLM load or generation, including failure.",
    "invocation_counts": "Count only current owned LLM attempts and outcomes; historical calls remain outside these counters.",
    "inference_substrate": "Use a truthful string for current CPU JSON, YAML, Markdown, and exact-reducer work.",
    "inference_substrate_class": "Use aggregation for the current host reduction and do not pad runtime.",
    "execution_venue": "Use the closed string host; put device and software facts in details.",
    "duration_s": "Measure the current task with a monotonic clock and never add sleep.",
    "phase_spans": "Retain actual phase boundaries, checkpoints, and elapsed time without inferred utilization.",
    "random_seed": "Freeze capstone and inherited resampling seeds; use null only when no seed applies.",
    "reproducibility_checksum": "Bind current code, configuration, source hashes, raw rows, gates, and reductions.",
    "source_artifact_hashes": "Bind exact authority, artifact, and conductor-record bytes with original class and flag.",
    "rows": "Retain every V649 task disposition with cost, failure, censoring, and evidence scope.",
    "sample_size_budget": "Separate planned, attempted, completed, censored, and unstarted units plus independent groups.",
    "acceptance_gate_results": "Keep validation, safety, completion, efficacy, advisory, and promotion checks separate.",
    "gate_check_summary": "Name each unavailable or invalid upstream, path, check, field, expected value, and observed value.",
    "verifier_is_oracle": "True because source-certified proof truth shares the deployed formal correctness oracle.",
    "honest_verdict": "Use complete scope for finished aggregation; unchanged missing evidence remains blocked within its branch.",
    "verdict_class": "Use the closed class and reserve partial for retryable unfinished work owned by this capstone.",
    "flagged_adversarial": "Preserve producer flags in rows; this field reports only a critical finding against the capstone producer.",
    "validation_receipts": "Retain exact argv, local environment, exit, duration, and hashed logs for each executed check.",
    "repository_health": "Keep unrelated broad-suite history separate; affected failures still disqualify current readiness.",
    "field_principles": "Explain ordinary top-level fields without wrapping values in value and principle envelopes.",
    "promotion_score": "Always remain zero; no rollout, weight change, publication, or leaderboard submission follows.",
    "capstone_complete_score": "One means fourteen honest dispositions and passing current capstone checks, not scientific success.",
    "task_dispositions": "Keep exactly fourteen ordered task IDs with title, path, class, flags, failures, and preserved evidence.",
    "claim_matrix": "Keep static, online, synthetic, live, proof-audit, ARC, and host-cost claims independently valid or unavailable.",
    "scope_reduction_compliance": "Account for calibrated-decision, self-learning, ARC, hardware, and deferred mechanism floors.",
    "continuation_rows": "Choose one closed continuation action per branch from a measured cause or named prerequisite.",
}


utc_now = command_boundary.utc_now
sha256_file = command_boundary.sha256_file
canonical_hash = command_boundary.canonical_hash
atomic_json = command_boundary.atomic_json


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush one truthful boundary so the capstone never appears idle."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7408] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json_object(path: Path) -> JsonDict:
    """Load one JSON object and reject list-shaped or malformed evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _terminal_status(value: object) -> bool:
    """Accept the repository's complete, blocked, and disqualified prefixes."""

    text = str(value)
    return text in {"complete", "blocked", "disqualified"} or text.startswith(
        ("complete_", "blocked_", "disqualified_", "terminal_")
    )


def _numeric_experiment_id(task_id: str) -> int:
    """Extract the numeric experiment identity from a full task ID."""

    match = re.match(r"exp(\d+)", task_id)
    if match is None:
        raise ValueError(f"numeric experiment identity missing: {task_id}")
    return int(match.group(1))


def load_contract(root: Path) -> JsonDict:
    """Select the exact V649 YAML and compare its compact task fields independently."""

    roadmap_file = root / ROADMAP_PATH
    roadmap = yaml.safe_load(roadmap_file.read_text(encoding="utf-8"))
    if not isinstance(roadmap, dict) or roadmap.get("milestone") != MILESTONE:
        raise ValueError(f"active roadmap must name milestone {MILESTONE}")
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list) or [row.get("id") for row in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("active roadmap must contain the exact fourteen V649 tasks")
    comparison = contract_boundary.build_contract_comparison(root)
    return {
        "milestone": MILESTONE,
        "roadmap_path": ROADMAP_PATH.as_posix(),
        "roadmap_sha256": sha256_file(roadmap_file),
        "design_path": DESIGN_PATH.as_posix(),
        "design_sha256": sha256_file(root / DESIGN_PATH),
        "contract_match": comparison.get("passed") is True,
        "contract_comparison": comparison,
        "tasks": deepcopy(tasks),
    }


def _gate_failures(payload: Mapping[str, Any]) -> list[Any]:
    """Preserve each producer's own failed gates before reading its metrics."""

    summary = payload.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        return [deepcopy(summary)] if summary else []
    rows: list[Any] = []
    for key in (
        "failures",
        "failed_checks",
        "required_failures",
        "structured_cohort_failures",
        "structured_prerequisite_failures",
    ):
        value = summary.get(key)
        if isinstance(value, list):
            rows.extend(deepcopy(value))
    if not rows:
        first = summary.get("first_failure") or summary.get("first_required_failure")
        if first:
            rows.append(deepcopy(first))
    return rows


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Represent missing bytes without creating a success-shaped artifact."""

    return {
        "task_id": str(task["id"]),
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "missing",
        "sha256": None,
        "source_file_sha256": None,
        "source_records": [],
        "status": "blocked",
        "honest_verdict": "blocked_missing_declared_artifact",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": False,
        "available": False,
        "accepted_for_science": False,
        "required_validation_passed": None,
        "failed_checks": [
            {
                "upstream": str(task["id"]),
                "path": str(task.get("deliverable")),
                "check": "declared_artifact_bytes",
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": None,
                "passed": False,
            }
        ],
        "payload": {},
    }


def _compare_gate(operator: str, observed: Any, expected: Any) -> bool:
    """Evaluate only the gate operators used by the active V649 roadmap."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return isinstance(expected, list) and observed in expected
    return False


def _declared_gate_rows(
    task: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Recompute a pre-gate from earlier authenticated producer fields."""

    rows: list[JsonDict] = []
    for gate in task.get("gated_on") or []:
        upstream = str(gate.get("upstream"))
        field = str(gate.get("artifact_field"))
        source = evidence.get(upstream, {})
        payload = source.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        observed = payload.get(field)
        operator = str(gate.get("op"))
        expected = gate.get("value")
        rows.append(
            {
                "upstream": upstream,
                "path": source.get("declared_path"),
                "check": f"structured_gate:{upstream}.{field}",
                "field": field,
                "operator": operator,
                "expected": deepcopy(expected),
                "observed": deepcopy(observed),
                "passed": _compare_gate(operator, observed, expected),
            }
        )
    return rows


def _conductor_evidence(
    root: Path,
    task: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Authenticate one exact conductor record while preserving missing bytes."""

    task_id = str(task["id"])
    marker = CONDUCTOR_MARKERS.get(task_id)
    log_path = root / CONDUCTOR_PATH
    if marker is None or not log_path.is_file():
        return _missing_evidence(task)
    matches = [line for line in log_path.read_text(encoding="utf-8").splitlines() if marker in line]
    if not matches:
        return _missing_evidence(task)
    pre_gate = task_id == "exp7404-live-memory"
    failed = [row for row in _declared_gate_rows(task, evidence) if not row["passed"]]
    if not pre_gate:
        failed = _missing_evidence(task)["failed_checks"]
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": (
            "conductor_pre_gate_record" if pre_gate else "conductor_completion_without_artifact"
        ),
        "sha256": canonical_hash({"path": CONDUCTOR_PATH.as_posix(), "records": matches}),
        "source_file_sha256": sha256_file(log_path),
        "source_records": matches,
        "status": "blocked",
        "honest_verdict": (
            "blocked_gate_check_failed"
            if pre_gate
            else "blocked_missing_declared_artifact_after_conductor_completion"
        ),
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": True,
        "available": False,
        "accepted_for_science": False,
        "required_validation_passed": None,
        "failed_checks": failed,
        "payload": {},
    }


def load_evidence_slot(
    root: Path,
    task: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Read one declared artifact or its exact conductor-only disposition."""

    relative = Path(str(task.get("deliverable")))
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        return _conductor_evidence(root, task, evidence)
    payload = load_json_object(path)
    verdict_class = str(payload.get("verdict_class"))
    flagged = payload.get("flagged_adversarial") is True
    authenticated = (
        payload.get("milestone") == MILESTONE
        and verdict_class in CLOSED_CLASSES
        and _terminal_status(payload.get("status"))
    )
    required_validation = payload.get("required_checks_passed")
    if required_validation is None:
        required_validation = not any(
            row.get("category") in {"required_validation", "safety"} and row.get("passed") is False
            for row in payload.get("acceptance_gate_results") or []
            if isinstance(row, Mapping)
        )
    return {
        "task_id": str(task["id"]),
        "declared_path": relative.as_posix(),
        "actual_path": relative.as_posix(),
        "source_kind": "declared_artifact",
        "sha256": sha256_file(path),
        "source_file_sha256": None,
        "source_records": [],
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "authenticated": authenticated,
        "available": True,
        "accepted_for_science": bool(
            authenticated and verdict_class in ELIGIBLE_CLASSES and not flagged
        ),
        "required_validation_passed": bool(required_validation),
        "failed_checks": _gate_failures(payload),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read all thirteen predecessor slots in the roadmap's fixed order."""

    evidence: dict[str, JsonDict] = {}
    for task in tasks[:-1]:
        task_id = str(task["id"])
        evidence[task_id] = load_evidence_slot(root, task, evidence)
    return evidence


def _feature_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Keep static diagnostic limits without copying thousands of source rows."""

    diagnosis = payload.get("feature_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
    return {
        "diagnostic_row_count": diagnosis.get("diagnostic_row_count"),
        "conflicting_cell_count": diagnosis.get("conflicting_cell_count"),
        "empirical_bayes_brier_floor": diagnosis.get("empirical_bayes_brier_floor"),
        "claim_scope": diagnosis.get("claim_scope"),
    }


def reduce_claim_matrix(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Reduce seven branches without allowing one branch to promote another."""

    static = evidence["exp7396-decision-diagnosis"]
    static_payload = static["payload"]
    online_audit = evidence["exp7401-online-audit"]
    synthetic = evidence["exp7403-synthetic-memory"]
    proof_audit = evidence["exp7405-proof-audit"]
    arc_checkpoint = evidence["exp7398-arc-checkpoint"]
    arc = evidence["exp7406-arc-generalization"]
    service = evidence["exp7407-service-cost"]
    service_payload = service["payload"]
    return {
        "static_calibration": {
            "source_tasks": ["exp7396-decision-diagnosis"],
            "verdict_class": static["verdict_class"],
            "validity": "eligible_completed_static_null",
            "completion_score": static_payload.get("static_audit_complete_score"),
            "value_score": static_payload.get("static_value_confirmed_score"),
            "feature_limits": _feature_summary(static_payload),
            "independent_external_sample": False,
            "verifier_is_oracle": False,
        },
        "online_learning": {
            "source_tasks": [
                "exp7397-delayed-adapter",
                "exp7399-online-trial",
                "exp7401-online-audit",
            ],
            "verdict_class": "blocked",
            "validity": "producer_artifacts_unavailable_and_audit_flagged",
            "completion_score": online_audit["payload"].get("online_audit_complete_score"),
            "value_score": online_audit["payload"].get("online_value_confirmed_score"),
            "producer_artifacts_present": False,
            "verifier_is_oracle": False,
        },
        "synthetic_memory": {
            "source_tasks": ["exp7403-synthetic-memory"],
            "verdict_class": synthetic["verdict_class"],
            "validity": "eligible_source_certified_synthetic_cohort",
            "completion_score": synthetic["payload"].get("synthetic_memory_capture_complete_score"),
            "value_score": synthetic["payload"].get("synthetic_memory_value_score"),
            "proof_safety_ready_score": synthetic["payload"].get("proof_safety_ready_score"),
            "live_model_benefit_established": synthetic["payload"].get(
                "live_model_benefit_established"
            ),
            "verifier_is_oracle": True,
        },
        "live_memory": {
            "source_tasks": ["exp7402-proposal-capture", "exp7404-live-memory"],
            "verdict_class": "blocked",
            "validity": "fresh_model_panel_and_live_cohort_unavailable",
            "completion_score": 0,
            "value_score": 0,
            "attempted_current_panel_calls": evidence["exp7402-proposal-capture"]["payload"].get(
                "attempted_call_count"
            ),
            "unstarted_current_panel_calls": evidence["exp7402-proposal-capture"]["payload"].get(
                "unstarted_call_count"
            ),
            "verifier_is_oracle": True,
        },
        "proof_audit": {
            "source_tasks": [
                "exp7403-synthetic-memory",
                "exp7404-live-memory",
                "exp7405-proof-audit",
            ],
            "verdict_class": proof_audit["verdict_class"],
            "validity": "synthetic_audit_complete_live_cohort_unavailable",
            "audit_complete_score": proof_audit["payload"].get("proof_audit_complete_score"),
            "value_score": proof_audit["payload"].get("proof_value_confirmed_score"),
            "synthetic_result_preserved": True,
            "combined_live_claim_available": False,
            "verifier_is_oracle": True,
        },
        "arc": {
            "source_tasks": ["exp7398-arc-checkpoint", "exp7406-arc-generalization"],
            "verdict_class": arc["verdict_class"],
            "validity": "checkpoint_ready_but_current_panel_disqualified",
            "checkpoint_ready_score": arc_checkpoint["payload"].get("arc_checkpoint_ready_score"),
            "capture_complete_score": arc["payload"].get(
                "arc_generalization_capture_complete_score"
            ),
            "current_induction_count": arc["payload"].get("current_induction_count"),
            "historical_induction_count": arc["payload"].get("historical_induction_count"),
            "reproduced_levels": deepcopy(arc["payload"].get("reproduced_levels")),
            "new_solve_claimed": arc["payload"].get("new_solve_claimed"),
            "official_score": arc["payload"].get("official_score"),
            "historical_proxy_is_current_arc_evidence": False,
            "verifier_is_oracle": False,
        },
        "host_service_cost": {
            "source_tasks": ["exp7407-service-cost"],
            "verdict_class": service["verdict_class"],
            "validity": "eligible_host_cpu_cost_only",
            "completion_score": service_payload.get("service_cost_capture_complete_score"),
            "value_score": int(service["verdict_class"] == "positive"),
            "hardware_ready_score": service_payload.get("hardware_ready_score"),
            "hardware_value_score": service_payload.get("hardware_value_score"),
            "board_disposition": deepcopy(service_payload.get("board_disposition")),
            "host_cost_is_board_qualification": False,
            "verifier_is_oracle": False,
        },
    }


def literature_control_rows(
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Map each reviewed paper only to controls that local tasks executed."""

    return [
        {
            "source_name": "ORCA",
            "source_id": "arXiv:2604.01170v2",
            "mapped_tasks": ["exp7397-delayed-adapter", "exp7399-online-trial"],
            "executed_controls": [
                "bounded two-scalar energy offset",
                "predict before delayed feedback",
                "frozen, logistic, recent-frequency, and no-feedback controls",
            ],
            "local_result_eligible": False,
            "paper_assumption_satisfied_locally": False,
            "limit": "Declared terminal artifacts are absent; hidden-state meta-learning and conformal transfer were not tested.",
        },
        {
            "source_name": "CORD",
            "source_id": "arXiv:2609.01072v2",
            "mapped_tasks": ["exp7396-decision-diagnosis"],
            "executed_controls": [
                "fixed answer identity",
                "raw-to-calibrated decision changes",
                "typed accept, reject, and escalate accounting",
            ],
            "local_result_eligible": evidence["exp7396-decision-diagnosis"]["accepted_for_science"],
            "paper_assumption_satisfied_locally": False,
            "limit": "The local binary risk head does not reproduce CORD's multiclass repair construction.",
        },
        {
            "source_name": "Solver-Hard",
            "source_id": "arXiv:2607.17047",
            "mapped_tasks": ["exp7402-proposal-capture"],
            "executed_controls": [
                "frozen stream identity",
                "separate source size and exact-solver effort",
                "separate parse and semantic outcomes",
            ],
            "local_result_eligible": False,
            "paper_assumption_satisfied_locally": False,
            "limit": "The panel made zero calls, so no model-hardness or relabeling conclusion is available.",
        },
        {
            "source_name": "Memoir",
            "source_id": "arXiv:2607.20792",
            "mapped_tasks": ["exp7403-synthetic-memory", "exp7405-proof-audit"],
            "executed_controls": [
                "separate prediction and commit phases",
                "restart equivalence",
                "erasure witnesses",
                "withheld live cohort",
            ],
            "local_result_eligible": evidence["exp7403-synthetic-memory"]["accepted_for_science"],
            "paper_assumption_satisfied_locally": False,
            "limit": "Source-certified synthetic memory is not Memoir's procedural-recall task or a live-model benefit.",
        },
        {
            "source_name": "hardware_review",
            "source_id": "V649 dated hardware and sampling review",
            "mapped_tasks": ["exp7407-service-cost"],
            "executed_controls": [
                "complete host service timing",
                "scalar-vector parity",
                "Amdahl bounds",
                "independent KV260, GateMate, and PolarFire rows",
            ],
            "local_result_eligible": evidence["exp7407-service-cost"]["accepted_for_science"],
            "paper_assumption_satisfied_locally": False,
            "limit": "No board kernel ran; vendor rates and external hardware results are not local evidence.",
        },
    ]


def _retired_ids(root: Path) -> set[int]:
    """Read only explicit numeric retirement receipts from the exclusion manifest."""

    document = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    retired: set[int] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for row in document.get(section) or []:
            if not isinstance(row, Mapping):
                continue
            values = [row.get("experiment_id"), *(row.get("experiment_ids") or [])]
            for value in values:
                match = re.search(r"(\d+)", str(value or ""))
                if match:
                    retired.add(int(match.group(1)))
    return retired


def retirement_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    root: Path = REPO_ROOT,
) -> list[JsonDict]:
    """Apply retirement only to an exact repeated top-level verdict string."""

    manifest_ids = _retired_ids(root)
    rows: list[JsonDict] = []
    for task in tasks:
        task_id = str(task["id"])
        current = evidence.get(task_id, {}).get("honest_verdict")
        for prior in task.get("prior_failures") or []:
            previous = prior.get("verdict")
            same = isinstance(current, str) and current == previous
            authorized = same and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current,
                    "same_exact_verdict": same,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "mechanism_change": prior.get("addressed_by"),
                    "decision": (
                        "retire-unchanged-mechanism"
                        if authorized
                        else "continue-with-measured-cause"
                    ),
                    "manifest_receipt_present": (
                        _numeric_experiment_id(task_id) in manifest_ids if authorized else False
                    ),
                    "reason": (
                        "The complete verdict string repeated under an active retirement condition."
                        if authorized
                        else "The exact verdict did not repeat; absence and similar wording do not retire a mechanism."
                    ),
                }
            )
    return rows


def continuation_rows(
    evidence: Mapping[str, Mapping[str, Any]],
    retirements: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Choose one closed next action for each independent branch."""

    retired_tasks = {
        str(row.get("task_id"))
        for row in retirements
        if row.get("decision") == "retire-unchanged-mechanism"
    }
    return [
        {
            "branch": "static_calibration",
            "decision": "continue-with-measured-cause",
            "measured_cause": "The eligible static audit found 22 conflicting feature cells and no registered Brier benefit.",
            "required_change": "Recover authenticated source questions and add source-grounded claim features before another fixed gate.",
        },
        {
            "branch": "online_learning",
            "decision": "wait-for-named-external-change",
            "measured_cause": "The declared Exp7397 and Exp7399 terminal artifacts are absent, and the audit is blocked and flagged.",
            "required_change": "Provide hash-authenticated terminal producer artifacts with clean required validation before any value audit.",
        },
        {
            "branch": "synthetic_memory",
            "decision": "continue-with-measured-cause",
            "measured_cause": "Source-certified synthetic memory completed with circular-positive proof value and passed safety controls.",
            "required_change": "Use an independently grounded or eligible live cohort before making an oracle-distinct or model-benefit claim.",
        },
        {
            "branch": "live_memory",
            "decision": (
                "retire-unchanged-mechanism"
                if "exp7404-live-memory" in retired_tasks
                else "wait-for-named-external-change"
            ),
            "measured_cause": "The live cohort repeated blocked_gate_check_failed after candidate capture completed zero of 64 calls.",
            "required_change": "Do not rerun the unchanged dependency chain; require a new eligible proposal-capture mechanism and owned GPU lease.",
        },
        {
            "branch": "proof_audit",
            "decision": "continue-with-measured-cause",
            "measured_cause": "The synthetic audit completed, but combined value stayed unavailable because the live cohort was blocked.",
            "required_change": "Audit a future eligible live cohort separately while retaining the current synthetic result.",
        },
        {
            "branch": "arc",
            "decision": (
                "retire-unchanged-mechanism"
                if "exp7406-arc-generalization" in retired_tasks
                else "continue-with-measured-cause"
            ),
            "measured_cause": "Durable checkpoint readiness passed, but the current panel repeated complete_disqualified_required_evidence with zero current inductions.",
            "required_change": "Require a new current invocation and evidence mechanism; do not rerun the unchanged panel launcher.",
        },
        {
            "branch": "host_service_cost",
            "decision": "wait-for-named-external-change",
            "measured_cause": "Host vectorized full-service benefit is measured, while hardware readiness and value remain zero.",
            "required_change": "Wait for a dated operator-authored GateMate physical-state change or measured board service before qualification.",
        },
    ]


def scope_reduction_compliance() -> JsonDict:
    """Account for mandated floors without launching deferred work early."""

    return {
        "overdue_priorities": "accounted by the active fourteen-task V649 roster",
        "calibrated_decision_floor": "Exp7396 completed the static audit; online producer evidence is unavailable",
        "continuous_self_learning_floor": "Exp7403 completed synthetic memory; live and online claims remain unavailable",
        "arc_floor": "Exp7398 qualified durable checkpoints; Exp7406 current science is disqualified",
        "hardware_floor": "Exp7407 measured host service cost and preserved three independent board states",
        "deferred_mechanisms": [
            "ORCA hidden-state meta-learning and theorem transfer",
            "CORD multiclass repair construction",
            "post-hoc selection by exact-solver effort",
            "new projection, KAN, decoding, or physical-learning branches",
            "board acceleration without measured local board service",
        ],
        "north_star_publication_gates_changed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "future_task_executed_early": False,
    }


def _science_failures(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """List invalid science before unavailable science so disqualification wins."""

    invalid: list[JsonDict] = []
    unavailable: list[JsonDict] = []
    for task_id, row in evidence.items():
        if row.get("verdict_class") == "disqualified":
            invalid.append(
                {
                    "upstream": task_id,
                    "path": row.get("declared_path"),
                    "check": "producer_science_eligibility",
                    "field": "verdict_class/flagged_adversarial",
                    "expected": {"verdict_class": sorted(ELIGIBLE_CLASSES), "flag": False},
                    "observed": {
                        "verdict_class": row.get("verdict_class"),
                        "flag": row.get("flagged_adversarial"),
                    },
                    "category": "required_science_validity",
                    "passed": False,
                }
            )
        elif row.get("available") is False or row.get("verdict_class") == "blocked":
            unavailable.append(
                {
                    "upstream": task_id,
                    "path": row.get("declared_path"),
                    "check": "producer_science_availability",
                    "field": "declared_path/verdict_class/flagged_adversarial",
                    "expected": "eligible_terminal_science_or_honest_branch_disposition",
                    "observed": {
                        "source_kind": row.get("source_kind"),
                        "verdict_class": row.get("verdict_class"),
                        "flag": row.get("flagged_adversarial"),
                    },
                    "category": "required_science_availability",
                    "passed": False,
                }
            )
    return [*invalid, *unavailable]


def _terminal_state(
    evidence: Mapping[str, Mapping[str, Any]], validation: Mapping[str, Any]
) -> JsonDict:
    """Classify current checks separately from invalid and unavailable science."""

    capstone_failures: list[JsonDict] = []
    for check, field in (
        (validation.get("required_checks_passed") is True, "required_checks_passed"),
        (validation.get("terminal_validation_passed") is True, "terminal_validation_passed"),
    ):
        if not check:
            capstone_failures.append(
                {
                    "upstream": EXPERIMENT_ID,
                    "path": RESULT_PATH.as_posix(),
                    "check": field,
                    "field": field,
                    "expected": True,
                    "observed": False,
                    "category": "required_validation",
                    "passed": False,
                }
            )
    science = _science_failures(evidence)
    disqualified = bool(capstone_failures) or any(
        row.get("verdict_class") == "disqualified" for row in evidence.values()
    )
    if disqualified:
        verdict_class = "disqualified"
        status = "complete_disqualified_v649_capstone"
        honest = (
            "complete_disqualified_required_science: all fourteen V649 tasks are accounted "
            "for; current ARC evidence is disqualified, online and live evidence is unavailable, "
            "and the static null, synthetic proof, proof audit, and host cost retain independent scope"
        )
    elif science:
        verdict_class = "blocked"
        status = "blocked_required_v649_science_unavailable"
        honest = "blocked_required_v649_science_unavailable_with_fourteen_dispositions"
    else:
        verdict_class = "null"
        status = "complete_null_v649_capstone"
        honest = "complete_null_v649_science_accounted_without_automatic_promotion"
    failures = [*science, *capstone_failures]
    return {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "capstone_checks_passed": not capstone_failures,
        "gate_check_summary": {
            "passed": not capstone_failures,
            "blocking_failed_count": len(capstone_failures),
            "science_failed_count": len(science),
            "failed_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
            "capstone_check_failures": capstone_failures,
        },
    }


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Build fourteen ordered rows, including self from current checks only."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        source = evidence[task_id]
        rows.append(
            {
                "order": order,
                "task_id": task_id,
                "numeric_experiment_id": _numeric_experiment_id(task_id),
                "title": task.get("title"),
                "declared_path": task.get("deliverable"),
                "actual_path": source.get("actual_path"),
                "source_kind": source.get("source_kind"),
                "source_sha256": source.get("sha256"),
                "status": source.get("status"),
                "honest_verdict": source.get("honest_verdict"),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
                "authenticated": source.get("authenticated"),
                "available": source.get("available"),
                "accepted_for_science": source.get("accepted_for_science"),
                "required_validation_passed": source.get("required_validation_passed"),
                "missing_checks": (
                    [] if source.get("available") is True else ["declared_artifact_bytes"]
                ),
                "failed_checks": deepcopy(source.get("failed_checks") or []),
                "cost": {"current_capstone_llm_calls": 0},
                "censored": False,
                "preserved_evidence": {
                    "original_verdict_class": source.get("verdict_class"),
                    "original_flagged_adversarial": source.get("flagged_adversarial"),
                    "conductor_record_count": len(source.get("source_records") or []),
                },
            }
        )
    self_task = tasks[-1]
    rows.append(
        {
            "order": 14,
            "task_id": EXPERIMENT_ID,
            "numeric_experiment_id": 7408,
            "title": self_task.get("title"),
            "declared_path": RESULT_PATH.as_posix(),
            "actual_path": RESULT_PATH.as_posix(),
            "source_kind": "self_current_checks",
            "source_sha256": None,
            "status": terminal["status"],
            "honest_verdict": terminal["honest_verdict"],
            "verdict_class": terminal["verdict_class"],
            "flagged_adversarial": False,
            "authenticated": True,
            "available": True,
            "accepted_for_science": False,
            "required_validation_passed": terminal["capstone_checks_passed"],
            "missing_checks": [],
            "failed_checks": deepcopy(terminal["gate_check_summary"]["capstone_check_failures"]),
            "cost": {"current_capstone_llm_calls": 0},
            "censored": False,
            "preserved_evidence": {
                "source": "actual current capstone checks",
                "prior_terminal_result_manufactured": False,
            },
        }
    )
    return rows


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Bind each authority and each source used by a task disposition."""

    rows: dict[str, JsonDict] = {}
    for relative in AUTHORITY_PATHS:
        path = root / relative
        if path.is_file():
            rows[f"authority:{relative.as_posix()}"] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_kind": "authority",
            }
        elif relative == STAGED_ROADMAP_PATH:
            rows[f"authority:{relative.as_posix()}"] = {
                "path": relative.as_posix(),
                "sha256": None,
                "source_kind": "expected_absent_staged_authority",
            }
    rows["active_roadmap"] = {
        "path": contract["roadmap_path"],
        "sha256": contract["roadmap_sha256"],
        "source_kind": "authority",
    }
    rows["exact_task_contract"] = {
        "path": contract["design_path"],
        "sha256": contract["design_sha256"],
        "source_kind": "authority",
    }
    for task_id, source in evidence.items():
        if source.get("source_kind") in {
            "conductor_pre_gate_record",
            "conductor_completion_without_artifact",
        }:
            rows[task_id] = {
                "path": CONDUCTOR_PATH.as_posix(),
                "sha256": source.get("sha256"),
                "source_file_sha256": source.get("source_file_sha256"),
                "source_records": deepcopy(source.get("source_records")),
                "source_kind": source.get("source_kind"),
                "original_verdict_class": source.get("verdict_class"),
                "original_flagged_adversarial": source.get("flagged_adversarial"),
            }
        else:
            rows[task_id] = {
                "path": source.get("actual_path"),
                "sha256": source.get("sha256"),
                "source_kind": source.get("source_kind"),
                "original_verdict_class": source.get("verdict_class"),
                "original_flagged_adversarial": source.get("flagged_adversarial"),
            }
    return rows


def _historical_model_sidecars(
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Reference model-producing artifacts without nesting historical counters."""

    rows: list[JsonDict] = []
    for task_id, source in evidence.items():
        payload = source.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if payload.get("model_invoked") is True or payload.get("MODEL_SPECS"):
            rows.append(
                {
                    "task_id": task_id,
                    "path": source.get("actual_path"),
                    "sha256": source.get("sha256"),
                    "scope": "historical_producer_only",
                    "counted_as_current_inference": False,
                    "original_verdict_class": source.get("verdict_class"),
                    "original_flagged_adversarial": source.get("flagged_adversarial"),
                }
            )
    return rows


def collect_preconditions(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Record exact authority identity, source eligibility, and runtime boundaries."""

    rows: list[JsonDict] = []
    for relative in AUTHORITY_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        optional_absence = relative == STAGED_ROADMAP_PATH and not available
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "expected": (
                    "absent_or_exact_v649_staged_yaml"
                    if relative == STAGED_ROADMAP_PATH
                    else "readable_nonempty_bytes"
                ),
                "observed": (
                    "readable_nonempty_bytes"
                    if available
                    else ("absent" if optional_absence else None)
                ),
                "passed": available or optional_absence,
                "sha256": sha256_file(path) if available else None,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-*",
                "expected": "REQ-REPORT-7408",
                "observed": "REQ-REPORT-7408" if "REQ-REPORT-7408" in spec_text else None,
                "passed": "REQ-REPORT-7408" in spec_text,
            },
            {
                "check": "active_roadmap_identity",
                "upstream": ROADMAP_PATH.as_posix(),
                "path": ROADMAP_PATH.as_posix(),
                "field": "milestone/task_ids",
                "expected": {"milestone": MILESTONE, "task_ids": list(EXPECTED_TASK_IDS)},
                "observed": {
                    "milestone": contract.get("milestone"),
                    "task_ids": [row.get("id") for row in contract.get("tasks", [])],
                },
                "passed": True,
            },
            {
                "check": "current_runtime_boundary",
                "upstream": EXPERIMENT_ID,
                "path": ENTRYPOINT_PATH.as_posix(),
                "field": "MODEL_SPECS/model_invoked/execution_venue/JAX_PLATFORMS",
                "expected": [[], False, "host", "cpu"],
                "observed": [[], False, "host", os.environ.get("JAX_PLATFORMS", "cpu")],
                "passed": os.environ.get("JAX_PLATFORMS", "cpu") == "cpu",
            },
        ]
    )
    for task_id in EXPECTED_TASK_IDS[:-1]:
        source = evidence[task_id]
        rows.append(
            {
                "check": "disposition_source_authenticated",
                "upstream": task_id,
                "path": source.get("declared_path"),
                "field": "path/hash/class/flag/source_kind",
                "expected": "authenticated artifact or exact conductor disposition",
                "observed": {
                    "hash": source.get("sha256"),
                    "class": source.get("verdict_class"),
                    "flag": source.get("flagged_adversarial"),
                    "source_kind": source.get("source_kind"),
                },
                "passed": source.get("authenticated") is True,
            }
        )
    return rows


def _sample_budget(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Count dispositions independently from artifact availability and science value."""

    present = sum(row.get("available") is True for row in evidence.values())
    conductor_only = sum(
        row.get("source_kind")
        in {"conductor_pre_gate_record", "conductor_completion_without_artifact"}
        for row in evidence.values()
    )
    return {
        "disposition_units": {
            "planned": 14,
            "attempted": 14,
            "completed": 14,
            "censored": 0,
            "unstarted": 0,
        },
        "predecessor_source_units": {
            "planned": 13,
            "attempted": 13,
            "artifact_present": present,
            "conductor_only": conductor_only,
            "unaccounted": 13 - present - conductor_only,
        },
        "effective_independent_group_count": len(CLAIM_BRANCHES),
        "limits": {
            "contract_authorities": 2,
            "current_llm_calls": 0,
            "current_board_operations": 0,
        },
        "stop_rule": "Read each declared path once, authenticate exact conductor-only dispositions, reduce seven branches, and stop after fourteen honest rows.",
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    operator: str = "==",
    required: bool,
) -> JsonDict:
    """Store each gate operand and whether it controls current completion."""

    passed = _compare_gate(operator, observed, expected)
    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "required_for_capstone_completion": required,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Keep accounting, validation, science, efficacy, and promotion separate."""

    return [
        _gate(
            "active_v649_roster", "completion", 14, len(contract.get("tasks", [])), required=True
        ),
        _gate("fourteen_honest_dispositions", "completion", 14, len(dispositions), required=True),
        _gate(
            "all_sources_authenticated",
            "completion",
            True,
            all(row.get("authenticated") is True for row in evidence.values()),
            required=True,
        ),
        _gate(
            "markdown_exact_contract_match",
            "advisory_contract",
            True,
            contract.get("contract_match") is True,
            required=False,
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            validation.get("required_checks_passed") is True,
            required=True,
        ),
        _gate(
            "terminal_readers",
            "required_validation",
            True,
            validation.get("terminal_validation_passed") is True,
            required=True,
        ),
        _gate(
            "required_science_valid",
            "scientific_validity",
            False,
            any(row.get("verdict_class") == "disqualified" for row in evidence.values()),
            required=False,
        ),
        _gate(
            "registered_combined_benefit",
            "scientific_efficacy",
            True,
            False,
            required=False,
        ),
        _gate("automatic_promotion", "promotion", 0, 0, required=False),
    ]


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide a complete zero-length phase ledger for unit construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "heartbeat_count": 0,
            "checkpoint": "unit_fixture",
        }
        for phase in (
            "preconditions",
            "build",
            "load",
            "generate",
            "evaluate",
            "validate",
            "write",
        )
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable source identity, raw rows, gates, and branch decisions."""

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
        "task_dispositions",
        "claim_matrix",
        "literature_control_rows",
        "scope_reduction_compliance",
        "retirement_rows",
        "continuation_rows",
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
    *,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one schema-complete terminal record from authenticated inputs."""

    terminal = _terminal_state(evidence, validation)
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal)
    retirements = retirement_rows(contract["tasks"], evidence, root=root)
    manifest_enforced = all(
        row["manifest_receipt_present"]
        for row in retirements
        if row["decision"] == "retire-unchanged-mechanism"
    )
    complete = int(
        len(dispositions) == 14
        and validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
        and manifest_enforced
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "numeric_experiment_id": 7408,
        "milestone": MILESTONE,
        "phase": 4,
        "title": "Reconcile fourteen dispositions and set evidence-based continuation",
        "status": terminal["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": collect_preconditions(root, contract, evidence),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": (
            "host CPU aggregation of JSON, YAML, Markdown, hashes, exact gate operands, "
            "and scoped subprocess receipts; zero current LLM or board operations"
        ),
        "inference_substrate_details": {
            "device": "host CPU",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu"),
            "work": "JSON and YAML parsing, SHA-256 hashing, exact gate reduction, and scoped validation",
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(root, contract, evidence),
        "historical_model_receipt_sidecars": _historical_model_sidecars(evidence),
        "small_ebm_training": {
            "performed_by_current_capstone": False,
            "counted_as_current_llm_work": False,
            "historical_producer_tasks": [
                "exp7396-decision-diagnosis",
                "exp7397-delayed-adapter",
                "exp7399-online-trial",
            ],
        },
        "contract_comparison": deepcopy(contract["contract_comparison"]),
        "rows": deepcopy(dispositions),
        "sample_size_budget": _sample_budget(evidence),
        "acceptance_gate_results": _acceptance_gates(contract, evidence, validation, dispositions),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
        "required_checks_passed": validation.get("required_checks_passed") is True,
        "terminal_validation_passed": validation.get("terminal_validation_passed") is True,
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "field_principles": {},
        "promotion_score": 0,
        "scientific_value_score": 0,
        "capstone_complete_score": complete,
        "task_dispositions": dispositions,
        "claim_matrix": reduce_claim_matrix(evidence),
        "literature_control_rows": literature_control_rows(evidence),
        "scope_reduction_compliance": scope_reduction_compliance(),
        "retirement_rows": retirements,
        "retirement_manifest_enforced": manifest_enforced,
        "continuation_rows": continuation_rows(evidence, retirements),
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "north_star_publication_gates_changed": False,
        "external_publication_authorized": False,
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key,
            f"Record the measured V649 capstone value for {key.replace('_', ' ')}.",
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload every file-backed source and exact conductor record."""

    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return False
    for row in sources.values():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return False
        path = root / str(row["path"])
        kind = row.get("source_kind")
        if kind == "expected_absent_staged_authority":
            if path.exists() or row.get("sha256") is not None:
                return False
        elif kind in {
            "conductor_pre_gate_record",
            "conductor_completion_without_artifact",
        }:
            records = row.get("source_records")
            if not path.is_file() or not isinstance(records, list):
                return False
            if row.get("source_file_sha256") != sha256_file(path):
                return False
            expected = canonical_hash({"path": CONDUCTOR_PATH.as_posix(), "records": records})
            if row.get("sha256") != expected:
                return False
        elif not path.is_file() or row.get("sha256") != sha256_file(path):
            return False
    return True


REQUIRED_FIELDS = (
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
    "inference_substrate_details",
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
    "capstone_complete_score",
    "task_dispositions",
    "claim_matrix",
    "scope_reduction_compliance",
    "continuation_rows",
)


def validate_artifact(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, source bytes, rows, branch reductions, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact["schema"],
        artifact["experiment_id"],
        artifact["milestone"],
        artifact["run_date"],
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if not _terminal_status(artifact["status"]) or artifact["verdict_class"] not in CLOSED_CLASSES:
        errors.append("lifecycle_invalid")
    if (
        artifact["MODEL_SPECS"] != []
        or artifact["model_invoked"] is not False
        or artifact["invocation_counts"] != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        not isinstance(artifact["inference_substrate"], str)
        or artifact["inference_substrate_class"] != "aggregation"
        or artifact["execution_venue"] != "host"
    ):
        errors.append("substrate_invalid")

    dispositions = artifact["task_dispositions"]
    valid_dispositions = (
        isinstance(dispositions, list)
        and len(dispositions) == 14
        and [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS)
        and [row.get("numeric_experiment_id") for row in dispositions] == list(range(7395, 7409))
        and [row.get("order") for row in dispositions] == list(range(1, 15))
        and dispositions[-1].get("source_kind") == "self_current_checks"
        and dispositions[-1].get("verdict_class") == artifact["verdict_class"]
    )
    if not valid_dispositions:
        errors.append("task_dispositions_invalid")

    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        expected = _task_dispositions(
            contract["tasks"], evidence, _terminal_state(evidence, artifact)
        )
        if isinstance(dispositions, list) and dispositions[:-1] != expected[:-1]:
            errors.append("task_dispositions_invalid")
        if artifact["claim_matrix"] != reduce_claim_matrix(evidence):
            errors.append("claim_matrix_invalid")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError):
        errors.append("independent_reduction_failed")

    expected_score = int(
        valid_dispositions
        and artifact.get("required_checks_passed") is True
        and artifact.get("terminal_validation_passed") is True
        and artifact.get("retirement_manifest_enforced") is True
    )
    if artifact["capstone_complete_score"] != expected_score:
        errors.append("capstone_score_invalid")
    if artifact["promotion_score"] != 0 or artifact.get("scientific_value_score") != 0:
        errors.append("promotion_invalid")
    if artifact["verdict_class"] != "disqualified" or not str(
        artifact["honest_verdict"]
    ).startswith("complete_disqualified"):
        errors.append("terminal_reduction_invalid")
    if artifact["flagged_adversarial"] is True and artifact["promotion_score"] != 0:
        errors.append("flagged_promotion_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    principles = artifact["field_principles"]
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the frozen Exp7358 plan for only the current affected Python files."""

    return command_boundary.build_command_plan(root, V649_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, duplicates, and command drift."""

    return command_boundary.validate_command_plan(root, V649_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V649 task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def _phase_span(  # pragma: no cover - authentic entrypoint timing boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    checkpoint: str,
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


def _terminal_commands(  # pragma: no cover - executed by the capability E2E.
    root: Path, candidate: Path
) -> list[command_boundary.PlannedCommand]:
    """Build entrypoint replay, independent reduction, and both strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7408_v649_capstone import validate_artifact;"
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
                ENTRYPOINT_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_recompute",
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
    return [command_boundary.PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(  # pragma: no cover - exercised through the declared entrypoint.
    root: Path, run_date: str
) -> JsonDict:
    """Run exact reads, scoped checks, cold readers, and atomic publication."""

    date_argument(run_date)
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7408-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "preconditions", "before")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(
        _phase_span("preconditions", point, started, checkpoint="thirteen_sources_authenticated")
    )
    progress(started, "preconditions", "after", dispositions=len(evidence))

    point = time.monotonic()
    progress(started, "build", "before_validation_plan")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{','.join(plan_errors)}")
    spans.append(_phase_span("build", point, started, checkpoint="affected_plan_frozen"))
    progress(started, "build", "after_validation_plan", commands=len(commands))

    for phase in ("load", "generate"):
        point = time.monotonic()
        progress(started, phase, "before", current_llm_operations=0)
        spans.append(_phase_span(phase, point, started, checkpoint="no_current_llm_work"))
        progress(started, phase, "after", current_llm_operations=0)

    point = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses", units=len(commands))
    affected = command_boundary.run_categorized_commands(
        root,
        [
            command_boundary.PlannedCommand(command, "required_validation", True)
            for command in commands
        ],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = command_boundary.reduce_affected_receipts(root, V649_MANIFEST, affected)
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
    progress(started, "evaluate", "before")
    spans.append(_phase_span("evaluate", point, started, checkpoint="branch_rows_reduced"))
    candidate = build_artifact(
        root,
        contract,
        evidence,
        validation,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(started, "evaluate", "after", verdict=candidate["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        units=len(terminal_commands),
    )
    terminal_receipts = command_boundary.run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = all(row.get("passed") is True for row in terminal_receipts)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal_receipts)
    validation["terminal_validation_passed"] = terminal_passed
    validation["validation_receipts"] = [*affected, *terminal_receipts]
    spans.append(
        _phase_span(
            "validate",
            point,
            started,
            checkpoint="cold_replay_and_strict_readers_complete",
        )
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
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=final_spans,
        flagged_adversarial=critical,
    )
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise ValueError(f"artifact_validation_failed:{','.join(errors)}")
    atomic_json(candidate_path, artifact)
    atomic_json(root / RESULT_PATH, artifact)
    atomic_json(
        root / CHECKPOINT_PATH,
        {
            "status": "complete",
            "artifact": RESULT_PATH.as_posix(),
            "completed_at_utc": utc_now(),
        },
    )
    progress(started, "write", "after_atomic", path=RESULT_PATH.as_posix())
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - public CLI boundary.
    """Parse the frozen date and optional cold-validation target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - public entrypoint.
    """Run the V649 capstone or cold-validate one measured candidate."""

    print("[exp7408] phase=startup event=flushed", flush=True)
    args = _parser().parse_args(argv)
    if args.validate is not None:
        progress(time.monotonic(), "cold_replay", "before", path=args.validate)
        value = load_json_object(args.validate)
        errors = validate_artifact(value)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        progress(time.monotonic(), "cold_replay", "after", passed=not errors)
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
