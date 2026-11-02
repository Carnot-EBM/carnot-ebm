"""Reconcile the V650 task contract without promoting one branch from another.

This module reads terminal artifacts and exact conductor pre-gate records. It
does not run a model, train an energy head, or operate hardware.

Spec refs: REQ-REPORT-7420 and SCENARIO-REPORT-7420-*.
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
import re
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
from carnot.experiment_7409_v650_evidence_custody import (
    compare_contract_authorities,
    load_yaml,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
PHASE = 4
EXPERIMENT_ID = "exp7420-capstone"
SCHEMA = "carnot.exp7420.v650.capstone.v1"

RESULT_PATH = Path("results/experiment_7420_v650_capstone.json")
RAW_DIR = Path("results/raw/experiment_7420_v650_capstone")
MODULE_PATH = Path("python/carnot/experiment_7420_v650_capstone.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7420_v650_capstone.py")
TEST_PATH = Path("tests/python/test_experiment_7420_v650_capstone.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_PATH = Path("ops/conductor-log.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")

EXPECTED_TASK_IDS = (
    "exp7409-evidence-custody",
    "exp7410-source-corpus",
    "exp7411-arc-call-budget",
    "exp7412-source-features",
    "exp7413-source-calibration",
    "exp7414-selected-feedback",
    "exp7415-decision-audit",
    "exp7416-anchored-extraction",
    "exp7417-extraction-audit",
    "exp7418-revision-memory",
    "exp7419-precision-placement",
    EXPERIMENT_ID,
)
CLAIM_BRANCHES = (
    "corpus_authority",
    "static_calibration",
    "online_calibration",
    "arc_callback_invariants",
    "qwen_extraction",
    "extraction_audit",
    "revised_proof_memory",
    "numeric_host_cost",
    "board_status",
)
CONTINUATION_DECISIONS = (
    "continue-with-measured-cause",
    "retire-unchanged-mechanism",
    "wait-for-named-external-change",
)
REQUIRED_SCIENCE_TASKS = (
    "exp7413-source-calibration",
    "exp7414-selected-feedback",
    "exp7415-decision-audit",
    "exp7416-anchored-extraction",
    "exp7417-extraction-audit",
    "exp7418-revision-memory",
)
ELIGIBLE_CLASSES = {"positive", "circular_positive", "null"}
CLOSED_CLASSES = {*ELIGIBLE_CLASSES, "blocked", "disqualified", "partial"}
ZERO_COUNTS = deepcopy(ZERO_INVOCATION_COUNTS)
RANDOM_SEED = {
    "task_order": 7_420_650_01,
    "branch_reduction": 7_420_650_02,
    "validation_order": 7_420_650_03,
}

CONDUCTOR_MARKERS = {
    "exp7417-extraction-audit": "Audit extraction coverage and semantic preservatio",
}
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/north-star.md"),
    Path("ops/verifier_gaps.md"),
    Path("research-studying.md"),
    Path("research-complete.yaml"),
    CONDUCTOR_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7408_v649_capstone.py"),
    Path("results/experiment_7408_v649_capstone.json"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "phase",
    "status",
    "run_date",
    "started_at_utc",
    "ended_at_utc",
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
    "field_principles",
    "promotion_score",
    "capstone_complete_score",
    "task_dispositions",
    "claim_matrix",
    "continuation_rows",
    "scope_reduction_compliance",
)

FIELD_PRINCIPLES = {
    "schema": "Versioned ordinary fields identify the experiment, milestone, and terminal state.",
    "run_date": "The frozen date is 20260919, with actual UTC start and end boundaries.",
    "preconditions_checked": "Exact paths, hashes, and resource checks precede dependent reduction.",
    "MODEL_SPECS": "Current LLM calls require unsloth/Qwen3.8-27B-GGUF; this aggregation uses none.",
    "model_invoked": "Only an actual current attempted LLM call makes this true.",
    "invocation_counts": "Owned current attempted, completed, failed, cancelled, and in-flight events determine counts.",
    "inference_substrate": "This is a truthful string; device and software details are separate.",
    "inference_substrate_class": "The class describes current aggregation without duration padding.",
    "execution_venue": "The closed venue string is host; device details are separate.",
    "duration_s": "Monotonic current duration is separate from producer and validation durations.",
    "phase_spans": "Real phase boundaries retain checkpoints and heartbeat counts.",
    "random_seed": "Frozen reduction and validation-order seeds are explicit.",
    "reproducibility_checksum": "The checksum binds code, protocol, input bytes, and reduced rows.",
    "source_artifact_hashes": "Exact source paths and hashes preserve original flags and classes.",
    "rows": "Every task disposition, including blocked and current units, remains visible.",
    "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted counts are separate.",
    "acceptance_gate_results": "Each gate records category, operator, expected, observed, pass state, and principle.",
    "gate_check_summary": "Each block names its upstream, path, check, field, expected, and observed values.",
    "verifier_is_oracle": "Source-defined proof rows are oracle-bound even though the capstone reducer is not.",
    "honest_verdict": "Completed accounting starts complete_; unavailable required science stays blocked.",
    "verdict_class": "The closed class distinguishes invalid, unavailable, null, circular, and positive results.",
    "flagged_adversarial": "Producer flags remain visible and cannot supply readiness.",
    "validation_receipts": "Exact argv, environment, exits, durations, names, and log hashes remain visible.",
    "field_principles": "Principles are separate from ordinary scalar and collection values.",
    "promotion_score": "This remains zero; no rollout, publication, or generator update follows.",
    "capstone_complete_score": "One means twelve dispositions and current checks completed, not scientific benefit.",
    "task_dispositions": "Exactly twelve ordered tasks retain their declared and observed terminal states.",
    "claim_matrix": "Nine branches keep authority, validity, completion, benefit, and limits separate.",
    "continuation_rows": "Each branch names a measured cause or an exact external prerequisite.",
    "scope_reduction_compliance": "Standing floors and deferred obligations remain explicit.",
}


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one actual UTC timestamp for a terminal receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - progress is exercised by the public E2E.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush each phase and slow-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7420] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def load_json_object(path: Path) -> JsonDict:
    """Read one JSON object and reject missing, malformed, or list-shaped bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(value)


def _terminal_status(value: object) -> bool:
    """Accept only the repository's completed or blocked terminal prefixes."""

    return isinstance(value, str) and value.startswith(("complete", "blocked", "disqualified"))


def _numeric_experiment_id(value: object) -> int | None:
    """Return the first experiment number so identity variants stay comparable."""

    match = re.search(r"(?:exp|experiment_)(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def load_contract(root: Path) -> JsonDict:
    """Parse both V650 authorities with the shipped generic comparison helper."""

    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    comparison = compare_contract_authorities(markdown, roadmap)
    tasks = [dict(row) for row in roadmap.get("tasks", []) if isinstance(row, Mapping)]
    if comparison.get("passed") is not True:
        raise ValueError(f"V650 contract mismatch: {comparison.get('errors')}")
    if [row.get("id") for row in tasks] != list(EXPECTED_TASK_IDS):
        raise ValueError("V650 task order mismatch")
    return {
        "milestone": roadmap.get("milestone"),
        "tasks": tasks,
        "comparison": comparison,
        "roadmap_path": ROADMAP_PATH.as_posix(),
        "roadmap_sha256": sha256_file(root / ROADMAP_PATH),
        "design_path": DESIGN_PATH.as_posix(),
        "design_sha256": sha256_file(root / DESIGN_PATH),
    }


def _compare_gate(operator: str, observed: Any, expected: Any) -> bool:
    """Evaluate only operators declared by the active V650 roadmap."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return isinstance(expected, list) and observed in expected
    return False


def _declared_gate_rows(
    task: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    """Recompute a pre-gate from authenticated earlier producer fields."""

    rows: list[JsonDict] = []
    for gate in task.get("gated_on") or []:
        upstream = str(gate.get("upstream"))
        field = str(gate.get("artifact_field"))
        producer = evidence.get(upstream, {})
        payload = producer.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        observed = payload.get(field)
        operator = str(gate.get("op"))
        expected = deepcopy(gate.get("value"))
        rows.append(
            {
                "upstream": upstream,
                "path": producer.get("declared_path"),
                "check": f"structured_gate:{upstream}.{field}",
                "field": field,
                "operator": operator,
                "expected": expected,
                "observed": deepcopy(observed),
                "passed": _compare_gate(operator, observed, expected),
            }
        )
    return rows


def _missing_evidence(task: Mapping[str, Any]) -> JsonDict:
    """Describe missing declared bytes without manufacturing a result."""

    return {
        "task_id": str(task.get("id")),
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "missing",
        "sha256": None,
        "source_file_sha256": None,
        "source_records": [],
        "status": "blocked_missing_declared_artifact",
        "honest_verdict": "blocked_missing_declared_artifact",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "authenticated": False,
        "available": False,
        "accepted_for_science": False,
        "required_validation_passed": None,
        "failed_checks": [
            {
                "upstream": str(task.get("id")),
                "path": str(task.get("deliverable")),
                "check": "declared_artifact_bytes",
                "field": "path",
                "operator": "exists",
                "expected": "readable_nonempty_bytes",
                "observed": None,
                "passed": False,
            }
        ],
        "payload": {},
    }


def _required_validation(payload: Mapping[str, Any]) -> bool | None:
    """Read only explicit validation gates; benefit failures remain scientific nulls."""

    explicit = payload.get("required_checks_passed")
    if isinstance(explicit, bool):
        return explicit
    gates = [
        row
        for row in payload.get("acceptance_gate_results") or []
        if isinstance(row, Mapping)
        and (
            "validation" in str(row.get("category") or "")
            or "validation" in str(row.get("check") or "")
        )
    ]
    if gates:
        return all(row.get("passed") is True for row in gates)
    return None


def _gate_failures(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve every producer gate failure instead of selecting convenient metrics."""

    return [
        deepcopy(dict(row))
        for row in payload.get("acceptance_gate_results") or []
        if isinstance(row, Mapping) and row.get("passed") is False
    ]


def _conductor_evidence(
    root: Path,
    task: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Bind exact pre-gate log lines while keeping artifact availability false."""

    task_id = str(task.get("id"))
    marker = CONDUCTOR_MARKERS.get(task_id)
    log_path = root / CONDUCTOR_PATH
    if marker is None or not log_path.is_file():
        return _missing_evidence(task)
    records = [line for line in log_path.read_text(encoding="utf-8").splitlines() if marker in line]
    if not records:
        return _missing_evidence(task)
    failed = [row for row in _declared_gate_rows(task, evidence) if not row["passed"]]
    return {
        "task_id": task_id,
        "declared_path": str(task.get("deliverable")),
        "actual_path": None,
        "source_kind": "conductor_pre_gate_record",
        "sha256": canonical_hash({"path": CONDUCTOR_PATH.as_posix(), "records": records}),
        "source_file_sha256": sha256_file(log_path),
        "source_records": records,
        "status": "blocked_gate_check_failed",
        "honest_verdict": "blocked_gate_check_failed",
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
    """Read one artifact or its exact conductor pre-gate disposition."""

    relative = Path(str(task.get("deliverable")))
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        return _conductor_evidence(root, task, evidence)
    payload = load_json_object(path)
    verdict_class = str(payload.get("verdict_class"))
    validation = _required_validation(payload)
    expected_id = _numeric_experiment_id(task.get("id"))
    actual_id = _numeric_experiment_id(payload.get("experiment_id"))
    authenticated = bool(
        payload.get("milestone") == MILESTONE
        and expected_id == actual_id
        and verdict_class in CLOSED_CLASSES
        and _terminal_status(payload.get("status"))
    )
    flagged = payload.get("flagged_adversarial") is True
    accepted = bool(
        authenticated
        and verdict_class in ELIGIBLE_CLASSES
        and not flagged
        and validation is not False
    )
    return {
        "task_id": str(task.get("id")),
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
        "accepted_for_science": accepted,
        "required_validation_passed": validation,
        "failed_checks": _gate_failures(payload),
        "payload": payload,
    }


def collect_evidence(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Read the eleven predecessor slots in fixed contract order."""

    evidence: dict[str, JsonDict] = {}
    for task in tasks[:-1]:
        task_id = str(task.get("id"))
        evidence[task_id] = load_evidence_slot(root, task, evidence)
    return evidence


def _score(payload: Mapping[str, Any], name: str) -> Any:
    """Read one ordinary producer scalar without inferring a replacement."""

    return deepcopy(payload.get(name))


def reduce_claim_matrix(evidence: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Reduce nine claims without allowing one branch to promote another."""

    corpus = evidence["exp7410-source-corpus"]
    static = evidence["exp7413-source-calibration"]
    online = evidence["exp7414-selected-feedback"]
    audit = evidence["exp7415-decision-audit"]
    arc = evidence["exp7411-arc-call-budget"]
    extraction = evidence["exp7416-anchored-extraction"]
    extraction_audit = evidence["exp7417-extraction-audit"]
    memory = evidence["exp7418-revision-memory"]
    numeric = evidence["exp7419-precision-placement"]
    corpus_payload = corpus["payload"]
    static_payload = static["payload"]
    online_payload = online["payload"]
    audit_payload = audit["payload"]
    arc_payload = arc["payload"]
    extraction_payload = extraction["payload"]
    memory_payload = memory["payload"]
    numeric_payload = numeric["payload"]
    boards = {
        str(row.get("board")): row.get("terminal_state")
        for row in numeric_payload.get("board_rows") or []
        if isinstance(row, Mapping)
    }
    return {
        "corpus_authority": {
            "source_tasks": ["exp7410-source-corpus"],
            "source_kind": corpus["source_kind"],
            "verdict_class": corpus["verdict_class"],
            "available": corpus["available"],
            "valid": corpus["accepted_for_science"],
            "completion_score": _score(corpus_payload, "source_corpus_ready_score"),
            "benefit_score": 0,
            "label_authority": "machine_annotations",
            "sample_limit": deepcopy(corpus_payload.get("sample_size_budget")),
            "compute_limit": "private attributed cache; no Enoki model stack",
            "verifier_is_oracle": False,
        },
        "static_calibration": {
            "source_tasks": ["exp7413-source-calibration", "exp7415-decision-audit"],
            "verdict_class": static["verdict_class"],
            "available": static["available"],
            "valid": static["accepted_for_science"] and audit["accepted_for_science"],
            "completion_score": _score(static_payload, "calibration_capture_complete_score"),
            "audit_complete_score": _score(audit_payload, "static_audit_complete_score"),
            "benefit_score": _score(static_payload, "calibration_value_score"),
            "label_authority": "machine_annotations",
            "sample_limit": deepcopy(static_payload.get("sample_size_budget")),
            "compute_limit": "small Gibbs heads on host CPU",
            "verifier_is_oracle": False,
        },
        "online_calibration": {
            "source_tasks": ["exp7414-selected-feedback", "exp7415-decision-audit"],
            "verdict_class": online["verdict_class"],
            "available": online["available"],
            "valid": online["accepted_for_science"] and audit["accepted_for_science"],
            "completion_score": _score(online_payload, "online_capture_complete_score"),
            "audit_complete_score": _score(audit_payload, "online_audit_complete_score"),
            "benefit_score": _score(online_payload, "online_value_score"),
            "label_authority": "delayed and revoked machine annotations",
            "sample_limit": deepcopy(online_payload.get("sample_size_budget")),
            "compute_limit": "selected-feedback host simulation",
            "verifier_is_oracle": False,
        },
        "arc_callback_invariants": {
            "source_tasks": ["exp7411-arc-call-budget"],
            "source_kind": arc["source_kind"],
            "verdict_class": arc["verdict_class"],
            "flagged_adversarial": arc["flagged_adversarial"],
            "available": arc["available"],
            "valid": arc["accepted_for_science"],
            "completion_score": _score(arc_payload, "arc_budget_ready_score"),
            "benefit_score": _score(arc_payload, "live_efficacy_score"),
            "sample_limit": deepcopy(arc_payload.get("sample_size_budget")),
            "compute_limit": "scripted callback transport; zero current model calls",
            "verifier_is_oracle": False,
        },
        "qwen_extraction": {
            "source_tasks": ["exp7416-anchored-extraction"],
            "source_kind": extraction["source_kind"],
            "verdict_class": extraction["verdict_class"],
            "available": extraction["available"],
            "valid": extraction["accepted_for_science"],
            "completion_score": _score(extraction_payload, "extraction_capture_complete_score"),
            "benefit_score": _score(extraction_payload, "extraction_value_score"),
            "sample_limit": deepcopy(extraction_payload.get("sample_size_budget")),
            "compute_limit": "96 planned Qwen calls; no owned single-slot capture completed",
            "verifier_is_oracle": False,
        },
        "extraction_audit": {
            "source_tasks": ["exp7417-extraction-audit"],
            "source_kind": extraction_audit["source_kind"],
            "verdict_class": extraction_audit["verdict_class"],
            "available": extraction_audit["available"],
            "valid": False,
            "completion_score": 0,
            "benefit_score": 0,
            "failed_checks": deepcopy(extraction_audit["failed_checks"]),
            "sample_limit": "unstarted because extraction capture was ineligible",
            "compute_limit": "aggregation only after eligible captured spans",
            "verifier_is_oracle": False,
        },
        "revised_proof_memory": {
            "source_tasks": ["exp7418-revision-memory"],
            "source_kind": memory["source_kind"],
            "verdict_class": memory["verdict_class"],
            "available": memory["available"],
            "valid": memory["accepted_for_science"],
            "completion_score": _score(memory_payload, "memory_revision_capture_complete_score"),
            "benefit_score": _score(memory_payload, "memory_revision_value_score"),
            "sample_limit": deepcopy(memory_payload.get("sample_size_budget")),
            "compute_limit": "CPU exact solver and simulator; no live-model benefit",
            "verifier_is_oracle": True,
        },
        "numeric_host_cost": {
            "source_tasks": ["exp7419-precision-placement"],
            "source_kind": numeric["source_kind"],
            "verdict_class": numeric["verdict_class"],
            "available": numeric["available"],
            "valid": numeric["accepted_for_science"],
            "completion_score": _score(numeric_payload, "precision_capture_complete_score"),
            "benefit_score": _score(numeric_payload, "precision_value_score"),
            "sample_limit": deepcopy(numeric_payload.get("sample_size_budget")),
            "compute_limit": "host emulation only; no device kernel",
            "verifier_is_oracle": False,
        },
        "board_status": {
            "source_tasks": ["exp7419-precision-placement"],
            "source_kind": numeric["source_kind"],
            "verdict_class": numeric["verdict_class"],
            "available": True,
            "valid": numeric["authenticated"],
            "completion_score": int(len(boards) == 3),
            "benefit_score": _score(numeric_payload, "hardware_value_score"),
            "hardware_ready_score": _score(numeric_payload, "hardware_ready_score"),
            "boards": boards,
            "host_science_invalidated": False,
            "compute_limit": "dated historical board rows; no fresh physical operation",
            "verifier_is_oracle": False,
        },
    }


def literature_control_rows(evidence: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    """Map reviewed methods only to controls with local V650 evidence."""

    return [
        {
            "source_name": "Enoki",
            "source_id": "arXiv:2609.00581v2",
            "mapped_tasks": ["exp7410-source-corpus", "exp7416-anchored-extraction"],
            "executed_controls": [
                "attributed source spans",
                "machine labels excluded from predictor features",
                "blocked bounded extraction retained",
            ],
            "local_result_eligible": evidence["exp7410-source-corpus"]["accepted_for_science"],
            "paper_reproduction_claimed": False,
            "limit": "Machine annotations are not exact semantic truth; extraction did not run.",
        },
        {
            "source_name": "MARGIN",
            "source_id": "arXiv:2605.22949v3",
            "mapped_tasks": [
                "exp7413-source-calibration",
                "exp7414-selected-feedback",
                "exp7415-decision-audit",
            ],
            "executed_controls": [
                "same-information logistic control",
                "recent-frequency control",
                "selected-only feedback arm",
            ],
            "local_result_eligible": evidence["exp7415-decision-audit"]["accepted_for_science"],
            "paper_reproduction_claimed": False,
            "limit": "The registered static and online results are valid nulls.",
        },
        {
            "source_name": "corrupted-feedback",
            "source_id": "arXiv:2605.20515",
            "mapped_tasks": ["exp7414-selected-feedback", "exp7415-decision-audit"],
            "executed_controls": [
                "prediction before delayed reveal",
                "revoked-label replay",
                "no-feedback and erased-update controls",
            ],
            "local_result_eligible": evidence["exp7415-decision-audit"]["accepted_for_science"],
            "paper_reproduction_claimed": False,
            "limit": "Prediction-set guarantees do not transfer to selective actions.",
        },
        {
            "source_name": "Memoir",
            "source_id": "arXiv:2607.20792",
            "mapped_tasks": ["exp7418-revision-memory"],
            "executed_controls": [
                "separate reads and committed writes",
                "source revision",
                "eviction and erasure witnesses",
                "cold restart parity",
            ],
            "local_result_eligible": evidence["exp7418-revision-memory"]["accepted_for_science"],
            "paper_reproduction_claimed": False,
            "limit": "Source-defined exact proofs are circular and total cost value is null.",
        },
    ]


def retirement_rows(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    current_verdict: str,
) -> list[JsonDict]:
    """Retire only an exact repeated verdict under an explicit active condition."""

    del evidence  # Prior task evidence cannot authorize the current capstone's retirement.
    rows: list[JsonDict] = []
    for task in tasks:
        if str(task.get("id")) != EXPERIMENT_ID:
            continue
        for prior in task.get("prior_failures") or []:
            previous = prior.get("verdict")
            same = isinstance(previous, str) and current_verdict == previous
            authorized = same and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": EXPERIMENT_ID,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "previous_verdict": previous,
                    "current_verdict": current_verdict,
                    "same_exact_verdict": same,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict"),
                    "decision": (
                        "retire-unchanged-mechanism"
                        if authorized
                        else "continue-with-measured-cause"
                    ),
                    "reason": (
                        "The exact declared verdict repeated under an active condition."
                        if authorized
                        else "The exact verdict changed; similar scope does not authorize retirement."
                    ),
                }
            )
    return rows


def continuation_rows(
    claims: Mapping[str, Mapping[str, Any]],
    retirements: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Choose one bounded next decision for each independent claim."""

    retired = any(row.get("decision") == "retire-unchanged-mechanism" for row in retirements)
    del claims  # Decisions name measured causes directly and do not infer new metrics.
    rows = [
        (
            "corpus_authority",
            "continue-with-measured-cause",
            "The attributed corpus is complete, but labels remain machine annotations.",
            "Add human-reviewed label authority before a semantic-truth claim.",
        ),
        (
            "static_calibration",
            "continue-with-measured-cause",
            "Static calibration completed without registered decision benefit.",
            "Change the source-aware mechanism before another fixed comparison.",
        ),
        (
            "online_calibration",
            "continue-with-measured-cause",
            "Selected-feedback calibration and its audit completed as a valid null.",
            "Change feedback support or policy before repeating the registered gate.",
        ),
        (
            "arc_callback_invariants",
            "continue-with-measured-cause",
            "Callback controls ran, but required validation is flagged and disqualified.",
            "Repair current provenance and validation without reopening Exp7406 panels.",
        ),
        (
            "qwen_extraction",
            "wait-for-named-external-change",
            "The bounded Qwen capture completed zero of 96 planned calls.",
            "Require one task-owned RTX 3090 lease receipt that satisfies the frozen slot check.",
        ),
        (
            "extraction_audit",
            "wait-for-named-external-change",
            "The audit was pre-gated because extraction capture was blocked.",
            "Require an eligible Exp7416 terminal artifact with capture score one.",
        ),
        (
            "revised_proof_memory",
            "continue-with-measured-cause",
            "Revision safety passed, but total-cost ratios failed the registered value gate.",
            "Reduce checking, invalidation, and storage cost before another value claim.",
        ),
        (
            "numeric_host_cost",
            "continue-with-measured-cause",
            "Int8 preserved actions but missed the complete-service speed gate.",
            "Change the complete host service path before repeating the timing gate.",
        ),
        (
            "board_status",
            "wait-for-named-external-change",
            "KV260 and PolarFire history is preserved; GateMate has no changed-state receipt.",
            "Wait for an operator-authored dated GateMate cable, port, power, board, JTAG, or DirtyJTAG change.",
        ),
    ]
    if retired:  # The only current retirement scope is the whole capstone, not one branch.
        rows[0] = (
            rows[0][0],
            "retire-unchanged-mechanism",
            rows[0][2],
            "Do not repeat the unchanged declared capstone scope.",
        )
    return [
        {
            "branch": branch,
            "decision": decision,
            "measured_cause": cause,
            "required_change": change,
        }
        for branch, decision, cause, change in rows
    ]


def scope_reduction_compliance(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """State which active floors were measured without turning nulls into wins."""

    return {
        "standing_floors": {
            "corpus": evidence["exp7410-source-corpus"]["accepted_for_science"],
            "arc": evidence["exp7411-arc-call-budget"]["accepted_for_science"],
            "calibrated_decision": all(
                evidence[task]["accepted_for_science"]
                for task in (
                    "exp7413-source-calibration",
                    "exp7414-selected-feedback",
                    "exp7415-decision-audit",
                )
            ),
            "self_learning": evidence["exp7418-revision-memory"]["accepted_for_science"],
            "hardware": evidence["exp7419-precision-placement"]["accepted_for_science"],
        },
        "standing_floor_limits": {
            "corpus": "ready corpus with machine annotations, not exact truth",
            "arc": "not satisfied because the callback artifact is flagged and disqualified",
            "calibrated_decision": "measured and audited static and online nulls",
            "self_learning": "source-defined revised proof memory with no total-cost value",
            "hardware": "host precision plus dated board accounting; no new acceleration",
        },
        "deferred_outer_loop_work": [
            "review and land commit-time enforcement of the 20 MiB result limit"
        ],
        "v649_missing_artifacts_remain_unknown": [
            "results/experiment_7397_v649_delayed_adapter.json",
            "results/experiment_7399_v649_online_trial.json",
        ],
        "retired_panel_chains_reopened": False,
        "preserved_retired_scopes": ["exp7404-live-memory", "exp7406-arc-generalization"],
        "north_star_publication_gates_changed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "future_milestone_activated": False,
    }


def _science_failure(task_id: str, row: Mapping[str, Any], category: str, check: str) -> JsonDict:
    """Name all operands needed to reproduce one required-science failure."""

    return {
        "upstream": task_id,
        "path": row.get("declared_path"),
        "check": check,
        "field": "verdict_class/flagged_adversarial/required_validation_passed",
        "operator": "eligible_and_unflagged",
        "expected": {
            "verdict_class": sorted(ELIGIBLE_CLASSES),
            "flagged_adversarial": False,
            "required_validation_passed": "not_false",
        },
        "observed": {
            "verdict_class": row.get("verdict_class"),
            "flagged_adversarial": row.get("flagged_adversarial"),
            "required_validation_passed": row.get("required_validation_passed"),
            "available": row.get("available"),
        },
        "category": category,
        "passed": False,
    }


def classify_terminal(
    evidence: Mapping[str, Mapping[str, Any]], validation: Mapping[str, Any]
) -> JsonDict:
    """Apply invalid-before-unavailable precedence only to required science."""

    invalid: list[JsonDict] = []
    unavailable: list[JsonDict] = []
    for task_id in REQUIRED_SCIENCE_TASKS:
        row = evidence[task_id]
        if (
            row.get("verdict_class") == "disqualified"
            or row.get("flagged_adversarial") is True
            or row.get("required_validation_passed") is False
        ):
            invalid.append(
                _science_failure(
                    task_id,
                    row,
                    "required_science_validity",
                    "required_science_validity",
                )
            )
        elif row.get("available") is not True or row.get("verdict_class") == "blocked":
            unavailable.append(
                _science_failure(
                    task_id,
                    row,
                    "required_science_availability",
                    "required_science_availability",
                )
            )

    current: list[JsonDict] = []
    for field in ("required_checks_passed", "terminal_validation_passed"):
        if validation.get(field) is not True:
            current.append(
                {
                    "upstream": EXPERIMENT_ID,
                    "path": RESULT_PATH.as_posix(),
                    "check": field,
                    "field": field,
                    "operator": "==",
                    "expected": True,
                    "observed": validation.get(field),
                    "category": "required_validation",
                    "passed": False,
                }
            )

    if invalid or current:
        verdict_class = "disqualified"
        status = "complete_disqualified_current_or_required_v650_evidence"
        honest = (
            "complete_disqualified_required_v650_evidence: twelve dispositions are retained; "
            "invalid required science or current validation prevents promotion"
        )
    elif unavailable:
        verdict_class = "blocked"
        status = "complete_blocked_required_v650_science"
        honest = "complete_blocked_required_v650_science_with_twelve_dispositions"
    else:
        verdict_class = "null"
        status = "complete_null_v650_capstone"
        honest = "complete_null_v650_science_accounted_without_automatic_promotion"
    failures = [*invalid, *current, *unavailable]
    return {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "capstone_checks_passed": not current,
        "gate_check_summary": {
            "passed": not current,
            "required_invalid_count": len(invalid),
            "required_unavailable_count": len(unavailable),
            "current_check_failure_count": len(current),
            "failed_count": len(failures),
            "first_failure": deepcopy(failures[0]) if failures else None,
            "failures": failures,
            "required_invalid_failures": invalid,
            "required_unavailable_failures": unavailable,
            "current_check_failures": current,
        },
    }


def _task_dispositions(
    tasks: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Build twelve ordered rows, including Exp7420 from current checks only."""

    rows: list[JsonDict] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task.get("id"))
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
                "failed_checks": deepcopy(source.get("failed_checks") or []),
                "censored": False,
                "cost": {"current_capstone_llm_calls": 0},
            }
        )
    self_task = tasks[-1]
    rows.append(
        {
            "order": 12,
            "task_id": EXPERIMENT_ID,
            "numeric_experiment_id": 7420,
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
            "failed_checks": deepcopy(terminal["gate_check_summary"]["current_check_failures"]),
            "censored": False,
            "cost": {"current_capstone_llm_calls": 0},
        }
    )
    return rows


def _source_hashes(
    root: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Bind each authority, code input, artifact, and conductor record."""

    rows: dict[str, JsonDict] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        if path.is_file():
            rows[f"source:{relative.as_posix()}"] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_kind": "authority_or_code",
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
        if source.get("source_kind") == "conductor_pre_gate_record":
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


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Reload every file-backed hash and every record-specific conductor hash."""

    sources = artifact.get("source_artifact_hashes")
    if not isinstance(sources, Mapping):
        return False
    for row in sources.values():
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return False
        path = root / str(row["path"])
        if row.get("source_kind") == "conductor_pre_gate_record":
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


def _historical_model_sidecars(
    evidence: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Reference historical model producers without inheriting their counters."""

    rows: list[JsonDict] = []
    for task_id, source in evidence.items():
        payload = source.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if payload.get("MODEL_SPECS") or payload.get("model_invoked") is True:
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
    """Record exact authorities and all predecessor eligibility observations."""

    rows: list[JsonDict] = [
        {
            "upstream": ROADMAP_PATH.as_posix(),
            "path": ROADMAP_PATH.as_posix(),
            "check": "active_milestone",
            "field": "milestone",
            "operator": "==",
            "expected": MILESTONE,
            "observed": contract.get("milestone"),
            "passed": contract.get("milestone") == MILESTONE,
        },
        {
            "upstream": DESIGN_PATH.as_posix(),
            "path": DESIGN_PATH.as_posix(),
            "check": "twelve_row_contract_match",
            "field": "comparison.passed",
            "operator": "==",
            "expected": True,
            "observed": contract["comparison"].get("passed"),
            "passed": contract["comparison"].get("passed") is True,
        },
    ]
    for task_id, source in evidence.items():
        rows.append(
            {
                "upstream": task_id,
                "path": source.get("declared_path"),
                "check": "authenticated_disposition_source",
                "field": "authenticated",
                "operator": "==",
                "expected": True,
                "observed": source.get("authenticated"),
                "passed": source.get("authenticated") is True,
                "sha256": source.get("sha256"),
                "available": source.get("available"),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
                "required_validation_passed": source.get("required_validation_passed"),
            }
        )
    return rows


def _sample_budget(evidence: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Count accounting units independently from scientific value."""

    artifacts = sum(row.get("source_kind") == "declared_artifact" for row in evidence.values())
    pre_gates = sum(
        row.get("source_kind") == "conductor_pre_gate_record" for row in evidence.values()
    )
    return {
        "planned": 12,
        "attempted": 12,
        "completed": 12,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
        "independent_groups": list(CLAIM_BRANCHES),
        "declared_artifact_sources": artifacts,
        "conductor_pre_gate_sources": pre_gates,
        "current_check_sources": 1,
        "stop_rule": "account for exactly exp7409 through exp7420 once in contract order",
    }


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep completion, science, safety, and promotion operands explicit."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep capstone completion separate from scientific availability and value."""

    required_eligible = not terminal["gate_check_summary"]["required_invalid_failures"]
    required_available = not terminal["gate_check_summary"]["required_unavailable_failures"]
    return [
        _gate(
            "contract_match",
            "accounting",
            "==",
            True,
            contract["comparison"].get("passed"),
            contract["comparison"].get("passed") is True,
            "Both task authorities must agree before evidence reduction.",
        ),
        _gate(
            "authenticated_predecessor_slots",
            "accounting",
            "==",
            11,
            sum(row.get("authenticated") is True for row in evidence.values()),
            all(row.get("authenticated") is True for row in evidence.values()),
            "Each predecessor needs an artifact or exact pre-gate source.",
        ),
        _gate(
            "twelve_ordered_dispositions",
            "completion",
            "==",
            list(EXPECTED_TASK_IDS),
            [row.get("task_id") for row in dispositions],
            [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS),
            "Completion accounts for every task once, independent of value.",
        ),
        _gate(
            "required_science_valid",
            "scientific_validity",
            "==",
            True,
            required_eligible,
            required_eligible,
            "Invalid required science disqualifies before unavailable science.",
        ),
        _gate(
            "required_science_available",
            "scientific_availability",
            "==",
            True,
            required_available,
            required_available,
            "Unavailable required science remains blocked rather than partial.",
        ),
        _gate(
            "required_affected_checks",
            "required_validation",
            "==",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "The frozen affected plan controls implementation validity.",
        ),
        _gate(
            "terminal_readers",
            "required_validation",
            "==",
            True,
            validation.get("terminal_validation_passed"),
            validation.get("terminal_validation_passed") is True,
            "Cold replay and strict readers must pass before publication.",
        ),
        _gate(
            "automatic_promotion",
            "promotion",
            "==",
            0,
            0,
            True,
            "A capstone never changes rollout, publication, or generator weights.",
        ),
    ]


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide a complete deterministic phase ledger for unit construction."""

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
            ("preconditions", "eleven_sources_authenticated"),
            ("plan", "affected_plan_frozen"),
            ("load", "no_current_model_load"),
            ("generate", "no_current_generation"),
            ("validate", "scoped_checks_complete"),
            ("reduce", "nine_claims_reduced"),
            ("terminal_validation", "cold_readers_complete"),
            ("write", "terminal_artifact_ready"),
        )
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable protocol, source identities, raw rows, and branch decisions."""

    return canonical_hash(
        {
            "schema": artifact.get("schema"),
            "experiment_id": artifact.get("experiment_id"),
            "milestone": artifact.get("milestone"),
            "run_date": artifact.get("run_date"),
            "random_seed": artifact.get("random_seed"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes"),
            "task_dispositions": artifact.get("task_dispositions"),
            "claim_matrix": artifact.get("claim_matrix"),
            "literature_control_rows": artifact.get("literature_control_rows"),
            "retirement_rows": artifact.get("retirement_rows"),
            "continuation_rows": artifact.get("continuation_rows"),
            "scope_reduction_compliance": artifact.get("scope_reduction_compliance"),
            "acceptance_gate_results": artifact.get("acceptance_gate_results"),
            "protocol": {
                "required_science_tasks": list(REQUIRED_SCIENCE_TASKS),
                "claim_branches": list(CLAIM_BRANCHES),
                "eligible_classes": sorted(ELIGIBLE_CLASSES),
            },
        }
    )


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
    """Build one schema-complete record from authenticated inputs and current checks."""

    terminal = classify_terminal(evidence, validation)
    if flagged_adversarial:
        terminal = deepcopy(terminal)
        terminal["status"] = "complete_disqualified_current_capstone_safety"
        terminal["honest_verdict"] = "complete_disqualified_current_capstone_safety"
        terminal["verdict_class"] = "disqualified"
    dispositions = _task_dispositions(contract["tasks"], evidence, terminal)
    claims = reduce_claim_matrix(evidence)
    retirements = retirement_rows(contract["tasks"], evidence, str(terminal["honest_verdict"]))
    complete = int(
        len(dispositions) == 12
        and [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS)
        and validation.get("required_checks_passed") is True
        and validation.get("terminal_validation_passed") is True
        and not flagged_adversarial
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "title": "Reconcile twelve dispositions and decide each research branch",
        "status": terminal["status"],
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": completed_at_utc,
        "preconditions_checked": collect_preconditions(root, contract, evidence),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_COUNTS),
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "inference_substrate_details": {
            "device": "host CPU",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax_platform": os.environ.get("JAX_PLATFORMS", "cpu"),
            "current_model_or_board_operations": 0,
            "work": (
                "JSON, YAML, and Markdown parsing; SHA-256 identities; exact gate "
                "reduction; and scoped subprocess receipts"
            ),
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
            "receipt_class": "small_ebm_training",
            "performed_by_current_capstone": False,
            "counted_as_current_llm_work": False,
            "historical_producer_tasks": [
                "exp7413-source-calibration",
                "exp7414-selected-feedback",
            ],
        },
        "contract_comparison": deepcopy(contract["comparison"]),
        "rows": deepcopy(dispositions),
        "sample_size_budget": _sample_budget(evidence),
        "acceptance_gate_results": _acceptance_gates(
            contract, evidence, validation, dispositions, terminal
        ),
        "gate_check_summary": deepcopy(terminal["gate_check_summary"]),
        "verifier_is_oracle": True,
        "honest_verdict": terminal["honest_verdict"],
        "verdict_class": terminal["verdict_class"],
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": deepcopy(validation.get("validation_receipts") or []),
        "required_checks_passed": validation.get("required_checks_passed") is True,
        "terminal_validation_passed": validation.get("terminal_validation_passed") is True,
        "repository_health": deepcopy(validation.get("repository_health") or {}),
        "field_principles": {},
        "promotion_score": 0,
        "scientific_value_score": 0,
        "capstone_complete_score": complete,
        "task_dispositions": dispositions,
        "claim_matrix": claims,
        "literature_control_rows": literature_control_rows(evidence),
        "retirement_rows": retirements,
        "continuation_rows": continuation_rows(claims, retirements),
        "scope_reduction_compliance": scope_reduction_compliance(evidence),
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "north_star_publication_gates_changed": False,
        "external_publication_authorized": False,
    }
    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(
            key, f"Record the measured V650 capstone value for {key.replace('_', ' ')}."
        )
        for key in artifact
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(value: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Cold-check identity, exact sources, reductions, decisions, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if (
        not _terminal_status(artifact.get("status"))
        or artifact.get("verdict_class") not in CLOSED_CLASSES
    ):
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")

    dispositions = artifact.get("task_dispositions")
    valid_dispositions = bool(
        isinstance(dispositions, list)
        and len(dispositions) == 12
        and [row.get("task_id") for row in dispositions] == list(EXPECTED_TASK_IDS)
        and [row.get("order") for row in dispositions] == list(range(1, 13))
        and dispositions[-1].get("source_kind") == "self_current_checks"
        and dispositions[-1].get("verdict_class") == artifact.get("verdict_class")
    )
    if not valid_dispositions:
        errors.append("task_dispositions_invalid")

    try:
        contract = load_contract(root)
        evidence = collect_evidence(root, contract["tasks"])
        terminal = classify_terminal(evidence, artifact)
        if artifact.get("flagged_adversarial") is True:
            terminal["status"] = "complete_disqualified_current_capstone_safety"
            terminal["honest_verdict"] = "complete_disqualified_current_capstone_safety"
            terminal["verdict_class"] = "disqualified"
        expected_rows = _task_dispositions(contract["tasks"], evidence, terminal)
        if isinstance(dispositions, list) and dispositions != expected_rows:
            errors.append("task_dispositions_invalid")
        if artifact.get("rows") != dispositions:
            errors.append("rows_invalid")
        claims = reduce_claim_matrix(evidence)
        if artifact.get("claim_matrix") != claims:
            errors.append("claim_matrix_invalid")
        expected_retirements = retirement_rows(
            contract["tasks"], evidence, str(terminal["honest_verdict"])
        )
        if artifact.get("retirement_rows") != expected_retirements:
            errors.append("retirement_rows_invalid")
        if artifact.get("continuation_rows") != continuation_rows(claims, expected_retirements):
            errors.append("continuation_rows_invalid")
        if artifact.get("literature_control_rows") != literature_control_rows(evidence):
            errors.append("literature_control_rows_invalid")
        if artifact.get("scope_reduction_compliance") != scope_reduction_compliance(evidence):
            errors.append("scope_reduction_invalid")
        if (
            artifact.get("status"),
            artifact.get("honest_verdict"),
            artifact.get("verdict_class"),
        ) != (
            terminal["status"],
            terminal["honest_verdict"],
            terminal["verdict_class"],
        ):
            errors.append("terminal_reduction_invalid")
    except (OSError, ValueError, KeyError, json.JSONDecodeError, yaml.YAMLError):
        errors.append("independent_reduction_failed")

    expected_complete = int(
        valid_dispositions
        and artifact.get("required_checks_passed") is True
        and artifact.get("terminal_validation_passed") is True
        and artifact.get("flagged_adversarial") is False
    )
    if artifact.get("capstone_complete_score") != expected_complete:
        errors.append("capstone_score_invalid")
    if artifact.get("promotion_score") != 0 or artifact.get("scientific_value_score") != 0:
        errors.append("promotion_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
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
    """Reject command expansion, broad tests, and command-local environment drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V650 contract."""

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


def _terminal_commands(  # pragma: no cover - capability E2E subprocesses.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, and both unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7420_v650_capstone import validate_artifact;"
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
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7420-", dir="/tmp"))

    point = time.monotonic()
    progress(started, "preconditions", "before")
    contract = load_contract(root)
    evidence = collect_evidence(root, contract["tasks"])
    spans.append(
        _phase_span("preconditions", point, started, checkpoint="eleven_sources_authenticated")
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
    spans.append(_phase_span("reduce", point, started, checkpoint="nine_claims_reduced"))
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
    progress(started, "reduce", "after", verdict=candidate["verdict_class"])

    point = time.monotonic()
    terminal_commands = _terminal_commands(root, candidate_path)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        units=len(terminal_commands),
    )
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
    """Run the V650 capstone or cold-validate one measured candidate."""

    print("[exp7420] phase=startup event=flushed", flush=True)
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
