"""Cold-reduce archived capture evidence without running a model.

The reducer treats the authenticated schedule and terminal call states as the
accounting authority. Proposal quality remains evidence, but it cannot change
whether every scheduled call has a terminal disposition.

Spec refs: REQ-REPORT-7359 and SCENARIO-REPORT-7359-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.experiment_7330_v644_public_learner import canonical_bytes, sha256_json
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7359-capture-reducer"
SCHEMA = "carnot.exp7359.v646.capture_reducer.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7359_v646_capture_reducer.json")
RAW_DIR = Path("results/raw/experiment_7359_v646_capture_reducer")
MODULE_PATH = Path("python/carnot/experiment_7359_v646_capture_reducer.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7359_v646_capture_reducer.py")
TEST_PATH = Path("tests/python/test_experiment_7359_v646_capture_reducer.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
VALIDATION_CONTRACT_PATH = Path("results/experiment_7358_v646_validation_contract.json")
HISTORICAL_RESULT_PATH = Path("results/experiment_7348_v645_plan_capture.json")
HISTORICAL_RAW_DIR = Path("results/raw/experiment_7348_v645_plan_capture")
SCHEDULE_MANIFEST_PATH = HISTORICAL_RAW_DIR / "schedule_manifest.json"
CANDIDATE_MANIFEST_PATH = HISTORICAL_RAW_DIR / "candidate_manifest.json"
EVALUATION_PATH = HISTORICAL_RAW_DIR / "private_evaluation.json"
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS: JsonDict = {
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
TERMINAL_STATES = frozenset({"response", "request_error", "cancelled"})
CLOSED_VERDICTS = frozenset(
    {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
)

V646_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    VALIDATION_CONTRACT_PATH,
    HISTORICAL_RESULT_PATH,
    SCHEDULE_MANIFEST_PATH,
    CANDIDATE_MANIFEST_PATH,
    EVALUATION_PATH,
)


def sha256_text(value: str) -> str:
    """Hash exact text so response identity does not depend on parsing."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash exact candidate-file bytes so whitespace changes remain visible."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a durable input without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    """Hash one JSON value with the project's stable JSON encoding."""

    return sha256_bytes(canonical_bytes(value))


def _deduplicate(values: Sequence[str]) -> list[str]:
    """Keep the first occurrence so diagnostics remain stable and readable."""

    return list(dict.fromkeys(values))


def _state_counts(calls: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive budget counts from terminal state instead of mutable score fields."""

    states = Counter(str(row.get("terminal_state")) for row in calls)
    attempted = states["response"] + states["request_error"]
    return {
        "attempted_units": attempted,
        "completed_units": states["response"],
        "failed_units": states["request_error"],
        "cancelled_units": states["cancelled"],
        "censored_units": states["request_error"] + states["cancelled"],
    }


def _identity_errors(schedule_row: Mapping[str, Any], call: Mapping[str, Any]) -> list[str]:
    """Compare the immutable identity fields that join a call to its schedule."""

    call_id = str(schedule_row.get("call_id"))
    errors: list[str] = []
    for field in (
        "call_index",
        "request_id",
        "prompt_sha256",
        "public_request_sha256",
    ):
        if field in schedule_row and call.get(field) != schedule_row.get(field):
            errors.append(f"{field}_mismatch:{call_id}")
    state = str(call.get("terminal_state"))
    if state not in TERMINAL_STATES:
        errors.append(f"terminal_state_invalid:{call_id}")
    expected_attempted = state in {"response", "request_error"}
    if call.get("attempted") is not expected_attempted:
        errors.append(f"attempted_state_mismatch:{call_id}")
    if "raw_reply_sha256" in call and call.get("raw_reply_sha256") != sha256_text(
        str(call.get("raw_reply") or "")
    ):
        errors.append(f"raw_reply_sha256_mismatch:{call_id}")
    if not isinstance(call.get("runtime_identity_receipt"), Mapping):
        errors.append(f"runtime_receipt_missing:{call_id}")
    return errors


def _normalized_assignments(row: Mapping[str, Any]) -> JsonDict | None:
    """Map a renamed twin plan back to the original public identifiers."""

    plan = row.get("decoded_plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("assignments"), Mapping):
        return None
    assignments = dict(plan["assignments"])
    if row.get("pair_side") != "twin":
        return assignments
    inverse = {str(value): str(key) for key, value in dict(row.get("renaming_map") or {}).items()}
    return {inverse.get(str(name), str(name)): value for name, value in assignments.items()}


def _pair_rows(
    schedule: Sequence[Mapping[str, Any]],
    calls_by_id: Mapping[str, Mapping[str, Any]],
    evaluator_by_id: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Retain every schedule-defined original and identifier-twin comparison."""

    groups: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for row in schedule:
        key = (str(row.get("panel_id")), int(row.get("candidate_index", -1)))
        groups[key][str(row.get("pair_side"))] = str(row.get("call_id"))
    pairs: list[JsonDict] = []
    for (panel_id, candidate_index), sides in sorted(groups.items()):
        original_id = sides.get("original")
        twin_id = sides.get("twin")
        original = calls_by_id.get(original_id or "")
        twin = calls_by_id.get(twin_id or "")
        both_present = original is not None and twin is not None
        original_plan = _normalized_assignments(original or {})
        twin_plan = _normalized_assignments(twin or {})
        pairs.append(
            {
                "panel_id": panel_id,
                "candidate_index": candidate_index,
                "original_call_id": original_id,
                "twin_call_id": twin_id,
                "both_present": both_present,
                "both_terminal": bool(
                    both_present
                    and original.get("terminal_state") in TERMINAL_STATES
                    and twin.get("terminal_state") in TERMINAL_STATES
                ),
                "both_source_valid": bool(
                    both_present
                    and original.get("parse_status") == "valid"
                    and twin.get("parse_status") == "valid"
                ),
                "renamed_assignments_match": (
                    original_plan == twin_plan
                    if original_plan is not None and twin_plan is not None
                    else None
                ),
                "original_hidden_rule_accepted": evaluator_by_id.get(original_id or "", {}).get(
                    "hidden_rule_accepted"
                ),
                "twin_hidden_rule_accepted": evaluator_by_id.get(twin_id or "", {}).get(
                    "hidden_rule_accepted"
                ),
                "disposition": "complete" if both_present else "incomplete",
                "censored": bool(
                    (original and original.get("terminal_state") != "response")
                    or (twin and twin.get("terminal_state") != "response")
                ),
            }
        )
    return pairs


def reduce_capture(
    schedule: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    candidate_bytes: Mapping[str, bytes],
    *,
    expected_schedule_sha256: str,
    evaluator_rows: Sequence[Mapping[str, Any]] = (),
    cost_rows: Sequence[Mapping[str, Any]] = (),
    final_readiness_score: int | None = None,
) -> JsonDict:
    """Reduce one explicit schedule without consulting proposal quality or readiness."""

    del final_readiness_score
    errors: list[str] = []
    if sha256_json(list(schedule)) != expected_schedule_sha256:
        errors.append("schedule_sha256_mismatch")

    schedule_ids = [str(row.get("call_id")) for row in schedule]
    call_ids = [str(row.get("call_id")) for row in calls]
    for call_id, count in Counter(schedule_ids).items():
        if not call_id or call_id == "None":
            errors.append("schedule_call_id_missing")
        elif count > 1:
            errors.append(f"duplicate_schedule_call_id:{call_id}")
    for call_id, count in Counter(call_ids).items():
        if count > 1:
            errors.append(f"duplicate_call_id:{call_id}")

    calls_by_id = {str(row.get("call_id")): row for row in calls}
    evaluator_by_id = {str(row.get("call_id")): row for row in evaluator_rows}
    costs_by_id = {str(row.get("call_id")): row for row in cost_rows}
    schedule_set = set(schedule_ids)
    for call_id in schedule_ids:
        if call_id not in calls_by_id:
            errors.append(f"missing_call:{call_id}")
    for call_id in call_ids:
        if call_id not in schedule_set:
            errors.append(f"unexpected_call:{call_id}")

    call_rows: list[JsonDict] = []
    matched_calls: list[Mapping[str, Any]] = []
    for schedule_row in schedule:
        call_id = str(schedule_row.get("call_id"))
        call = calls_by_id.get(call_id)
        if call is None:
            continue
        matched_calls.append(call)
        errors.extend(_identity_errors(schedule_row, call))
        exact_bytes = candidate_bytes.get(call_id)
        if exact_bytes is None:
            errors.append(f"candidate_bytes_missing:{call_id}")
            candidate_sha = None
        else:
            candidate_sha = sha256_bytes(exact_bytes)
            if exact_bytes != canonical_bytes(call) + b"\n":
                errors.append(f"candidate_bytes_mismatch:{call_id}")
        evaluation = evaluator_by_id.get(call_id, {})
        state = str(call.get("terminal_state"))
        failures = []
        if call.get("parse_status") != "valid":
            failures.append("source_parse_failure")
        if state == "request_error":
            failures.append("request_error")
        elif state == "cancelled":
            failures.append("cancelled")
        if evaluation.get("hidden_rule_accepted") is False:
            failures.append("hidden_rule_rejected")
        call_rows.append(
            {
                "call_id": call_id,
                "call_index": call.get("call_index"),
                "request_id": call.get("request_id"),
                "panel_id": call.get("panel_id"),
                "candidate_index": call.get("candidate_index"),
                "pair_side": call.get("pair_side"),
                "terminal_state": state,
                "attempted": state in {"response", "request_error"},
                "censored": state in {"request_error", "cancelled"},
                "parse_status": call.get("parse_status"),
                "hidden_rule_accepted": evaluation.get("hidden_rule_accepted"),
                "candidate_bytes_sha256": candidate_sha,
                "raw_reply_sha256": call.get("raw_reply_sha256"),
                "runtime_receipt_sha256": _canonical_hash(call.get("runtime_identity_receipt")),
                "historical_costs": deepcopy(dict(costs_by_id.get(call_id) or {})),
                "failures": failures,
                "disposition": "terminal" if state in TERMINAL_STATES else "invalid",
            }
        )

    state_counts = _state_counts(matched_calls)
    reduced_budget = {"planned_units": len(schedule), **state_counts}
    source_failures = sum(row.get("parse_status") != "valid" for row in matched_calls)
    hidden_failures = sum(row.get("hidden_rule_accepted") is False for row in evaluator_rows)
    stable_errors = _deduplicate(errors)
    return {
        "capture_complete_score": int(not stable_errors and len(matched_calls) == len(schedule)),
        "reduced_budget": reduced_budget,
        "errors": stable_errors,
        "call_rows": call_rows,
        "identifier_twin_rows": _pair_rows(schedule, calls_by_id, evaluator_by_id),
        "semantic_failure_counts": {
            "source_parse_failures": source_failures,
            "hidden_rule_failures": hidden_failures,
        },
    }


def independent_reduce_capture(
    schedule: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    candidate_bytes: Mapping[str, bytes],
    *,
    expected_schedule_sha256: str,
) -> JsonDict:
    """Recompute coverage with a separate index-and-counter implementation."""

    errors: list[str] = []
    if sha256_json(list(schedule)) != expected_schedule_sha256:
        errors.append("schedule_sha256_mismatch")
    schedule_counts = Counter(str(row.get("call_id")) for row in schedule)
    call_counts = Counter(str(row.get("call_id")) for row in calls)
    for call_id, count in schedule_counts.items():
        if not call_id or call_id == "None":
            errors.append("schedule_call_id_missing")
        elif count != 1:
            errors.append(f"duplicate_schedule_call_id:{call_id}")
    for call_id, count in call_counts.items():
        if count != 1:
            errors.append(f"duplicate_call_id:{call_id}")
        if call_id not in schedule_counts:
            errors.append(f"unexpected_call:{call_id}")
    indexed_schedule = {str(row.get("call_id")): row for row in schedule}
    indexed_calls = {str(row.get("call_id")): row for row in calls}
    for call_id, schedule_row in indexed_schedule.items():
        call = indexed_calls.get(call_id)
        if call is None:
            errors.append(f"missing_call:{call_id}")
            continue
        errors.extend(_identity_errors(schedule_row, call))
        blob = candidate_bytes.get(call_id)
        if blob is None:
            errors.append(f"candidate_bytes_missing:{call_id}")
        elif blob != canonical_bytes(call) + b"\n":
            errors.append(f"candidate_bytes_mismatch:{call_id}")
    budget = {"planned_units": len(schedule), **_state_counts(calls)}
    stable_errors = _deduplicate(errors)
    return {
        "capture_complete_score": int(not stable_errors and len(calls) == len(schedule)),
        "errors": stable_errors,
        "reduced_budget": budget,
    }


def diagnose_historical_mismatches(
    original: Mapping[str, Any], candidate: Mapping[str, Any]
) -> JsonDict:
    """Measure both Exp7348 failures without changing its archived verdict."""

    observed_budget = deepcopy(dict(original.get("sample_size_budget") or {}))
    accounting_fields = (
        "planned_units",
        "attempted_units",
        "completed_units",
        "failed_units",
        "cancelled_units",
        "censored_units",
    )
    expected_budget = {field: observed_budget.get(field) for field in accounting_fields}
    observed_only = sorted(set(observed_budget) - set(accounting_fields))
    return {
        "sample_size_budget_mismatch": {
            "failed_field": "sample_size_budget",
            "expected_value": expected_budget,
            "observed_value": observed_budget,
            "observed_only_fields": observed_only,
            "writer": "experiment_7348_v645_plan_capture.run_experiment",
            "reducer": "experiment_7348_v645_plan_capture.independent_reduce",
            "cause": "the writer added policy metadata but the historical reducer required full-dictionary equality",
            "control": "compare named immutable call-state fields and retain policy metadata separately",
        },
        "plan_capture_complete_score_mismatch": {
            "failed_field": "plan_capture_complete_score",
            "expected_value": candidate.get("plan_capture_complete_score"),
            "observed_value": original.get("plan_capture_complete_score"),
            "candidate_observed_value": candidate.get("plan_capture_complete_score"),
            "writer": "experiment_7348_v645_plan_capture.run_experiment.internal_validation_branch",
            "reducer": "experiment_7348_v645_plan_capture.independent_reduce",
            "cause": "final readiness classification overwrote accounting completion",
            "control": "derive completion only from schedule coverage and terminal call states",
        },
    }


def _gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    *,
    principle: str,
) -> JsonDict:
    """Record both sides of a prerequisite or acceptance decision."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
        "principle": principle,
    }


def validation_contract_gate_rows(producer: Mapping[str, Any]) -> list[JsonDict]:
    """Reject an absent, unfinished, unsafe, or unready same-milestone contract."""

    expected = {
        "experiment_id": "exp7358-validation-contract",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "complete_validation_contract_null_science",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_contract_ready_score": 1,
    }
    rows = [
        _gate_row(
            f"validation_contract_{field}",
            "exp7358-validation-contract",
            field,
            value,
            producer.get(field),
            principle="Only the terminal unflagged V646 validation boundary may authorize dependent checks.",
        )
        for field, value in expected.items()
    ]
    quarantined = (
        "quarantin"
        in " ".join(str(producer.get(field, "")) for field in ("status", "honest_verdict")).lower()
    )
    rows.append(
        _gate_row(
            "validation_contract_not_quarantined",
            "exp7358-validation-contract",
            "quarantined",
            False,
            quarantined,
            principle="Quarantined evidence cannot authorize dependent accounting work.",
        )
    )
    return rows


def _load_object(path: Path) -> JsonDict:
    """Load one required JSON object and return an empty value on bad bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate_paths(schedule: Sequence[Mapping[str, Any]]) -> list[Path]:
    """Resolve only schedule-owned candidate files and ignore stale run debris."""

    return [
        HISTORICAL_RAW_DIR / "calls" / f"call_{int(row.get('call_index', -1)):03d}.json"
        for row in schedule
    ]


def collect_preconditions(
    repo_root: Path,
) -> tuple[list[JsonDict], dict[str, str], JsonDict]:  # pragma: no cover - entrypoint I/O.
    """Check every exact producer and raw byte path before replay begins."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                principle="Dependent work needs the exact declared input bytes.",
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    producer = _load_object(root / VALIDATION_CONTRACT_PATH)
    checks.extend(validation_contract_gate_rows(producer))
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        _gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7359",
            "REQ-REPORT-7359" if "REQ-REPORT-7359" in spec else None,
            principle="The implementation must follow a checked-in requirement.",
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    checks.append(
        _gate_row(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "experiment_id: 7359" in exclusion,
            principle="An excluded task must stop before it consumes historical evidence.",
        )
    )

    schedule_manifest = _load_object(root / SCHEDULE_MANIFEST_PATH)
    schedule = schedule_manifest.get("schedule")
    schedule_rows = schedule if isinstance(schedule, list) else []
    candidate_paths = _candidate_paths(schedule_rows)
    missing_paths = [path.as_posix() for path in candidate_paths if not (root / path).is_file()]
    checks.append(
        _gate_row(
            "scheduled_candidate_files",
            SCHEDULE_MANIFEST_PATH.as_posix(),
            "candidate_file_paths",
            [],
            missing_paths,
            principle="Every scheduled call needs its exact archived candidate byte file.",
        )
    )
    for relative in candidate_paths:
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = sha256_file(path)
    context = {
        "producer": producer,
        "schedule_manifest": schedule_manifest,
        "candidate_paths": [path.as_posix() for path in candidate_paths],
    }
    return checks, hashes, context


def _gate_summary(gates: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate and retain exact expected and observed values."""

    failed = [dict(row) for row in gates.values() if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the record without recursively hashing its checksum field."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return _canonical_hash(value)


def _field_principles(fields: Sequence[str]) -> JsonDict:
    """Explain each field without wrapping its ordinary value."""

    principles = {
        "schema": "Version this record and keep ordinary experiment and milestone fields.",
        "experiment_id": "Keep the correction receipt addressable by its ordinary task identity.",
        "milestone": "Bind the diagnostic to the V646 contract that authorized it.",
        "phase": "Record the requested milestone phase without changing the active roadmap.",
        "status": "Use a terminal status only after actual work and affected validation.",
        "run_date": "Use 20260917 and pair it with real UTC timestamps.",
        "started_at_utc": "Record the real UTC start while duration uses monotonic time.",
        "completed_at_utc": "Record the real UTC completion of the published receipt.",
        "preconditions_checked": "Show exact upstream, field, expected value, and observed value before replay.",
        "MODEL_SPECS": "List current intended models; this cold task intends none.",
        "model_invoked": "Any attempted current load or generation would make this true.",
        "invocation_counts": "Keep all current model counts at zero and separate historical receipts.",
        "inference_substrate": "Name the actual host hash and JSON reduction work.",
        "inference_substrate_class": "Use aggregation because this task only combines archived evidence.",
        "execution_venue": "Name host computation and make no new board claim.",
        "host_computation": "Record the actual host machine, processor, and Python runtime.",
        "duration_s": "Use measured monotonic elapsed time without padding.",
        "phase_spans": "Keep disjoint measured load, generation, evaluation, validation, and write spans.",
        "random_seed": "Use null because the byte replay has no random operation.",
        "reproducibility_checksum": "Bind code, settings, inputs, evaluator evidence, and raw evidence.",
        "source_artifact_hashes": "Authenticate every producer, manifest, candidate file, and changed source.",
        "rows": "Keep every raw call disposition, cost, failure, and censoring outcome.",
        "sample_size_budget": "Freeze replay units from the explicit authenticated schedule.",
        "acceptance_gate_results": "Keep validation, safety, and scientific value gates separate.",
        "gate_check_summary": "Name failed checks with exact expected and observed values.",
        "verifier_is_oracle": "Keep true because the historical private evaluator defines semantic truth.",
        "honest_verdict": "Distinguish accounting readiness from unavailable current science.",
        "verdict_class": "Use the closed terminal enum and reserve partial for retryable owned work.",
        "flagged_adversarial": "A current critical finding would prevent accounting readiness.",
        "validation_receipts": "Keep each exact command, scope, exit, elapsed time, and log hash.",
        "repository_health": "Keep unrelated dated failures outside affected required checks.",
        "field_principles": "Explain fields separately without wrapping scalar gates or dictionaries.",
        "capture_reducer_ready_score": "One means only that the accounting implementation passed independent checks.",
        "mismatch_diagnosis": "Record each failed field, values, writer, reducer, cause, and control.",
        "historical_correction": "Preserve the original verdict and flag in a separate hash-bound receipt.",
        "reduced_budget": "Derive every call count from immutable terminal states.",
        "identifier_twin_rows": "Retain every original and renamed identifier comparison.",
        "semantic_failure_counts": "Keep parse and oracle failures without making them accounting failures.",
        "historical_inference_sidecars": "Label and hash old model work without current invocation credit.",
        "source_evidence": "Give cold validators exact reloadable paths for all raw evidence.",
        "independent_reduction": "Show agreement from code that does not call the primary reducer.",
        "accounting_completion_score": "Report schedule coverage without using usefulness or final readiness.",
        "scientific_value_score": "Stay zero because V646 ran no new scientific inference.",
        "promotion_score": "Stay zero because accounting readiness cannot authorize science consumers.",
        "affected_manifest": "Declare the exact tests, changed modules, and static entrypoint.",
        "production_defaults_changed": "Confirm this diagnostic did not change production behavior.",
        "historical_artifact_modified": "Confirm Exp7348 remains byte-identical and quarantined.",
        "artifact_stage": "Distinguish an internal preterminal candidate from the published terminal receipt.",
    }
    return {field: principles[field] for field in fields}


def _historical_sidecar(
    original: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Label old Qwen work so it cannot be mistaken for current inference."""

    runtime_hashes = Counter(_canonical_hash(row.get("runtime_identity_receipt")) for row in calls)
    return {
        "source": "exp7348",
        "source_experiment_id": original.get("experiment_id"),
        "artifact_path": HISTORICAL_RESULT_PATH.as_posix(),
        "artifact_sha256": sha256_file(REPO_ROOT / HISTORICAL_RESULT_PATH),
        "label": "historical_diagnostic_only_no_current_generation",
        "historical_MODEL_SPECS": deepcopy(original.get("MODEL_SPECS") or []),
        "historical_model_invoked": original.get("model_invoked"),
        "historical_invocation_counts": deepcopy(original.get("invocation_counts") or {}),
        "load_receipt_sha256": _canonical_hash(original.get("load_receipt") or {}),
        "gpu_receipts_sha256": _canonical_hash(original.get("gpu_receipts") or {}),
        "runtime_receipt_groups": [
            {"runtime_receipt_sha256": receipt_hash, "call_count": count}
            for receipt_hash, count in sorted(runtime_hashes.items())
        ],
        "authorizes_current_inference": False,
    }


def _utc_now() -> str:  # pragma: no cover - wall-clock evidence.
    """Return a real UTC timestamp for an artifact or command boundary."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7359] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _close_span(
    spans: list[JsonDict], phase: str, phase_started: float, run_started: float, units: int
) -> None:  # pragma: no cover - entrypoint measurement.
    """Close one measured phase and keep spans disjoint in monotonic time."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "start_s": phase_started - run_started,
            "end_s": ended - run_started,
            "duration_s": ended - phase_started,
            "completed_units": units,
        }
    )


def _repository_health() -> JsonDict:
    """Keep the dated V645 broad-suite failures separate from current checks."""

    return {
        "status": "degraded_historical_failures_retained",
        "affects_required_checks": False,
        "as_of": "2026-09-17",
        "historical_failures": [
            {
                "date": "2026-09-16",
                "source": "exp7348-plan-capture",
                "classification": "original_internal_validation_failure_preserved",
                "resolved_by_current_science": False,
            }
        ],
    }


def _build_artifact(
    *,
    started_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    original: Mapping[str, Any],
    candidate: Mapping[str, Any],
    schedule_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    reduction: Mapping[str, Any],
    independent: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    affected_passed: bool,
    terminal_passed: bool,
    stage: str,
) -> JsonDict:  # pragma: no cover - exercised by the end-to-end entrypoint.
    """Assemble the correction receipt while keeping science scores at zero."""

    agreement = {
        key: reduction.get(key) == independent.get(key)
        for key in ("capture_complete_score", "reduced_budget", "errors")
    }
    accounting_passed = bool(
        all(row.get("passed") is True for row in preconditions)
        and not reduction.get("errors")
        and all(agreement.values())
        and affected_passed
        and terminal_passed
    )
    gates = {
        "preconditions": {
            "check": "preconditions",
            "upstream": "declared_inputs",
            "artifact_field": "preconditions_checked",
            "expected": True,
            "observed": all(row.get("passed") is True for row in preconditions),
            "passed": all(row.get("passed") is True for row in preconditions),
            "principle": "All exact producers and raw paths must pass before replay.",
        },
        "schedule_and_candidate_integrity": {
            "check": "schedule_and_candidate_integrity",
            "upstream": "exp7348-plan-capture",
            "artifact_field": "reduced_budget",
            "expected": [],
            "observed": deepcopy(reduction.get("errors") or []),
            "passed": not reduction.get("errors"),
            "principle": "Every scheduled call needs one exact terminal candidate file.",
        },
        "independent_reducer_agreement": {
            "check": "independent_reducer_agreement",
            "upstream": EXPERIMENT_ID,
            "artifact_field": "independent_reduction",
            "expected": {key: True for key in agreement},
            "observed": agreement,
            "passed": all(agreement.values()),
            "principle": "A second implementation must reproduce the accounting result.",
        },
        "affected_validation": {
            "check": "affected_validation",
            "upstream": "exp7358-validation-contract",
            "artifact_field": "validation_receipts",
            "expected": True,
            "observed": affected_passed,
            "passed": affected_passed,
            "principle": "Only explicit affected tests and changed modules gate this task.",
        },
        "terminal_validation": {
            "check": "terminal_validation",
            "upstream": EXPERIMENT_ID,
            "artifact_field": "validation_receipts",
            "expected": True,
            "observed": terminal_passed,
            "passed": terminal_passed,
            "principle": "Cold reduction and both strict readers must accept the candidate.",
        },
        "scientific_value": {
            "check": "scientific_value",
            "upstream": EXPERIMENT_ID,
            "artifact_field": "scientific_value_score",
            "expected": 1,
            "observed": 0,
            "passed": False,
            "principle": "Historical replay is not new V646 scientific evidence.",
        },
    }
    calls = list(candidate_manifest.get("calls") or [])
    source_evidence = {
        "schedule_manifest_path": SCHEDULE_MANIFEST_PATH.as_posix(),
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
        "candidate_file_paths": _candidate_paths(list(schedule_manifest.get("schedule") or [])),
        "evaluation_path": EVALUATION_PATH.as_posix(),
        "historical_result_path": HISTORICAL_RESULT_PATH.as_posix(),
        "historical_terminal_candidate_path": (
            HISTORICAL_RAW_DIR / "terminal_candidate.json"
        ).as_posix(),
    }
    source_evidence["candidate_file_paths"] = [
        path.as_posix() for path in source_evidence["candidate_file_paths"]
    ]
    original_hash = hashes.get(HISTORICAL_RESULT_PATH.as_posix())
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": (
            "complete_capture_reducer_null_science"
            if stage == "terminal"
            else "complete_preterminal_capture_reducer_candidate"
        ),
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "operation": "hash_authenticated_json_reduction",
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": deepcopy(list(reduction.get("call_rows") or [])),
        "sample_size_budget": {
            **deepcopy(dict(reduction.get("reduced_budget") or {})),
            "stopping_rule": "Replay each call in the explicit authenticated schedule once; do not replace failed or cancelled calls.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_accounting_reducer_ready_no_current_science",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": _repository_health(),
        "field_principles": {},
        "capture_reducer_ready_score": int(accounting_passed),
        "mismatch_diagnosis": diagnose_historical_mismatches(original, candidate),
        "historical_correction": {
            "source_artifact_path": HISTORICAL_RESULT_PATH.as_posix(),
            "source_artifact_sha256": original_hash,
            "original_status": original.get("status"),
            "original_honest_verdict": original.get("honest_verdict"),
            "original_verdict_class": original.get("verdict_class"),
            "original_flagged_adversarial": original.get("flagged_adversarial"),
            "original_plan_capture_complete_score": original.get("plan_capture_complete_score"),
            "original_internal_validation_errors": deepcopy(
                original.get("internal_validation_errors") or []
            ),
            "diagnostic_capture_completion": reduction.get("capture_complete_score"),
            "disposition": "original_disqualification_and_quarantine_preserved",
            "counts_as_v646_science": False,
        },
        "reduced_budget": deepcopy(dict(reduction.get("reduced_budget") or {})),
        "identifier_twin_rows": deepcopy(list(reduction.get("identifier_twin_rows") or [])),
        "semantic_failure_counts": deepcopy(dict(reduction.get("semantic_failure_counts") or {})),
        "historical_inference_sidecars": [_historical_sidecar(original, calls)],
        "source_evidence": source_evidence,
        "independent_reduction": deepcopy(dict(independent)),
        "accounting_completion_score": reduction.get("capture_complete_score", 0),
        "scientific_value_score": 0,
        "promotion_score": 0,
        "affected_manifest": {
            "experiment_id": V646_MANIFEST.experiment_id,
            "test_paths": list(V646_MANIFEST.test_paths),
            "changed_modules": list(V646_MANIFEST.changed_modules),
            "static_paths": list(V646_MANIFEST.static_paths),
        },
        "production_defaults_changed": False,
        "historical_artifact_modified": False,
        "artifact_stage": stage,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _blocked_artifact(
    *,
    started_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
) -> JsonDict:  # pragma: no cover - external absence branch.
    """Publish exact external absence with no dependent replay or validation work."""

    failed = [dict(row) for row in preconditions if row.get("passed") is not True]
    planned = 0
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "blocked_capture_reducer_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host_cpu_precondition_checks_only",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "host_computation": {
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
            "operation": "precondition_checks_only",
        },
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": None,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "rows": [],
        "sample_size_budget": {
            "planned_units": planned,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "cancelled_units": planned,
            "censored_units": planned,
            "stopping_rule": "Stop before dependent replay when an external prerequisite fails.",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {
            "all_passed": False,
            "failed_count": len(failed),
            "first_failure": failed[0] if failed else None,
        },
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_capture_reducer_prerequisite",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {"status": "not_evaluated", "affects_required_checks": False},
        "field_principles": {},
        "capture_reducer_ready_score": 0,
        "mismatch_diagnosis": {},
        "historical_correction": {},
        "reduced_budget": {},
        "identifier_twin_rows": [],
        "semantic_failure_counts": {},
        "historical_inference_sidecars": [],
        "source_evidence": {},
        "independent_reduction": {},
        "accounting_completion_score": 0,
        "scientific_value_score": 0,
        "promotion_score": 0,
        "affected_manifest": {
            "experiment_id": V646_MANIFEST.experiment_id,
            "test_paths": list(V646_MANIFEST.test_paths),
            "changed_modules": list(V646_MANIFEST.changed_modules),
            "static_paths": list(V646_MANIFEST.static_paths),
        },
        "production_defaults_changed": False,
        "historical_artifact_modified": False,
        "artifact_stage": "terminal",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def _reload_reduction(
    artifact: Mapping[str, Any], repo_root: Path
) -> tuple[JsonDict, JsonDict]:  # pragma: no cover - covered by terminal artifact test.
    """Reload archived manifests and candidate files from the artifact's exact paths."""

    evidence = dict(artifact.get("source_evidence") or {})
    schedule_manifest = _load_object(repo_root / str(evidence.get("schedule_manifest_path", "")))
    candidate_manifest = _load_object(repo_root / str(evidence.get("candidate_manifest_path", "")))
    evaluation = _load_object(repo_root / str(evidence.get("evaluation_path", "")))
    schedule = list(schedule_manifest.get("schedule") or [])
    calls = list(candidate_manifest.get("calls") or [])
    paths = [repo_root / str(path) for path in evidence.get("candidate_file_paths") or []]
    blobs: dict[str, bytes] = {}
    for row, path in zip(schedule, paths, strict=False):
        if path.is_file():
            blobs[str(row.get("call_id"))] = path.read_bytes()
    original = _load_object(repo_root / HISTORICAL_RESULT_PATH)
    reduced = reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=str(schedule_manifest.get("schedule_sha256")),
        evaluator_rows=list(evaluation.get("rows") or []),
        cost_rows=list(original.get("generation_cost_rows") or []),
    )
    independent = independent_reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=str(schedule_manifest.get("schedule_sha256")),
    )
    return reduced, independent


def validate_artifact(value: object, *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Cold-check schema, raw replay, score independence, and exact source hashes."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")

    if artifact.get("verdict_class") == "blocked":
        if artifact.get("rows") or artifact.get("validation_receipts"):
            errors.append("blocked_artifact_has_dependent_work")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_summary_missing")
        if artifact.get("capture_reducer_ready_score") != 0:
            errors.append("blocked_ready_score_nonzero")
    else:
        reduced, independent = _reload_reduction(artifact, repo_root)
        if reduced.get("errors"):
            errors.append("archived_reduction_failed")
        expected = {
            "reduced_budget": reduced.get("reduced_budget"),
            "rows": reduced.get("call_rows"),
            "identifier_twin_rows": reduced.get("identifier_twin_rows"),
            "semantic_failure_counts": reduced.get("semantic_failure_counts"),
            "accounting_completion_score": reduced.get("capture_complete_score"),
            "independent_reduction": independent,
        }
        if any(artifact.get(field) != expected_value for field, expected_value in expected.items()):
            errors.append("stored_reduction_mismatch")
        terminal_names = {
            row.get("name")
            for row in artifact.get("validation_receipts") or []
            if row.get("passed") is True and row.get("exit_code") == 0
        }
        terminal_passed = {
            "independent_reducer",
            "adversarial_verify",
            "verdict_row_consistency_strict",
        } <= terminal_names
        affected = validation_contract.reduce_affected_receipts(
            repo_root,
            V646_MANIFEST,
            [
                row
                for row in artifact.get("validation_receipts") or []
                if row.get("name") in validation_scope.REQUIRED_CHECK_NAMES
            ],
        )["passed"]
        expected_ready = int(
            not reduced.get("errors")
            and reduced.get("capture_complete_score") == 1
            and reduced.get("reduced_budget") == independent.get("reduced_budget")
            and affected
            and terminal_passed
        )
        if artifact.get("capture_reducer_ready_score") != expected_ready:
            errors.append("capture_reducer_ready_score_mismatch")
        if artifact.get("scientific_value_score") != 0 or artifact.get("promotion_score") != 0:
            errors.append("accounting_promoted_as_science")
        correction = dict(artifact.get("historical_correction") or {})
        if (
            correction.get("original_verdict_class") != "disqualified"
            or correction.get("original_flagged_adversarial") is not True
        ):
            errors.append("historical_quarantine_not_preserved")

    for relative, expected_hash in dict(artifact.get("source_artifact_hashes") or {}).items():
        path = repo_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            errors.append(f"source_hash_mismatch:{relative}")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return _deduplicate(errors)


def _terminal_commands(
    candidate: Path,
) -> list[validation_contract.PlannedCommand]:  # pragma: no cover
    """Build the cold reducer and two strict terminal readers."""

    python = str(REPO_ROOT / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7359_v646_capture_reducer import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "candidate"
            ),
            "completion",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "candidate",
            ),
            "safety",
            True,
        ),
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(
                "verdict_row_consistency_strict",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "candidate",
            ),
            "completion",
            True,
        ),
    ]


def _terminal_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for each terminal reader."""

    names = ("independent_reducer", "adversarial_verify", "verdict_row_consistency_strict")
    return all(
        sum(
            row.get("name") == name and row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
        )
        == 1
        for name in names
    )


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - executed through the declared entrypoint.
    """Authenticate, replay, validate, and atomically publish one correction receipt."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    started = time.monotonic()
    started_at = _utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, context = collect_preconditions(root)
    _close_span(spans, "load", phase_started, started, len(preconditions))
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    progress(started, "preconditions", "end", passed=preconditions_passed)
    if not preconditions_passed:
        blocked = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            preconditions=preconditions,
            hashes=hashes,
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        validation_contract.atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", path=output_path)
        return blocked

    phase_started = time.monotonic()
    progress(started, "generation", "start", model_calls=0)
    _close_span(spans, "generation", phase_started, started, 0)
    progress(started, "generation", "end", model_calls=0)

    phase_started = time.monotonic()
    progress(started, "evaluation", "start")
    original = _load_object(root / HISTORICAL_RESULT_PATH)
    candidate = _load_object(root / (HISTORICAL_RAW_DIR / "terminal_candidate.json"))
    schedule_manifest = _load_object(root / SCHEDULE_MANIFEST_PATH)
    candidate_manifest = _load_object(root / CANDIDATE_MANIFEST_PATH)
    evaluation = _load_object(root / EVALUATION_PATH)
    schedule = list(schedule_manifest.get("schedule") or [])
    calls = list(candidate_manifest.get("calls") or [])
    candidate_paths = _candidate_paths(schedule)
    blobs = {
        str(row.get("call_id")): (root / path).read_bytes()
        for row, path in zip(schedule, candidate_paths, strict=True)
    }
    reduction = reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=str(schedule_manifest.get("schedule_sha256")),
        evaluator_rows=list(evaluation.get("rows") or []),
        cost_rows=list(original.get("generation_cost_rows") or []),
    )
    independent = independent_reduce_capture(
        schedule,
        calls,
        blobs,
        expected_schedule_sha256=str(schedule_manifest.get("schedule_sha256")),
    )
    hashes.update(
        {
            MODULE_PATH.as_posix(): sha256_file(root / MODULE_PATH),
            WRAPPER_PATH.as_posix(): sha256_file(root / WRAPPER_PATH),
            TEST_PATH.as_posix(): sha256_file(root / TEST_PATH),
            (HISTORICAL_RAW_DIR / "terminal_candidate.json").as_posix(): sha256_file(
                root / (HISTORICAL_RAW_DIR / "terminal_candidate.json")
            ),
        }
    )
    _close_span(spans, "evaluation", phase_started, started, len(calls))
    progress(
        started,
        "evaluation",
        "end",
        calls=len(calls),
        errors=len(reduction.get("errors") or []),
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7359-validation-", dir="/tmp"))
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    phase_started = time.monotonic()
    commands = validation_contract.build_command_plan(root, V646_MANIFEST, private_root)
    plan_errors = validation_contract.validate_command_plan(root, V646_MANIFEST, commands)
    progress(started, "validation", "before_affected_subprocesses", plan_errors=len(plan_errors))
    affected_receipts: list[JsonDict] = []
    if not plan_errors:
        affected_receipts = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=raw_dir / "validation/affected",
        )
    affected_passed = bool(
        not plan_errors
        and validation_contract.reduce_affected_receipts(root, V646_MANIFEST, affected_receipts)[
            "passed"
        ]
    )
    progress(started, "validation", "after_affected_subprocesses", passed=affected_passed)

    preterminal = _build_artifact(
        started_at=started_at,
        duration_s=time.monotonic() - started,
        spans=spans,
        preconditions=preconditions,
        hashes=hashes,
        original=original,
        candidate=candidate,
        schedule_manifest=schedule_manifest,
        candidate_manifest=candidate_manifest,
        reduction=reduction,
        independent=independent,
        validation_receipts=affected_receipts,
        affected_passed=affected_passed,
        terminal_passed=False,
        stage="preterminal",
    )
    candidate_path = raw_dir / "terminal_candidate.json"
    validation_contract.atomic_json(candidate_path, preterminal)
    progress(started, "validation", "before_terminal_subprocesses", candidate=candidate_path)
    terminal_receipts = validation_contract.run_categorized_commands(
        root,
        _terminal_commands(candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    terminal_passed = _terminal_receipts_pass(terminal_receipts)
    _close_span(
        spans,
        "validation",
        phase_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    progress(started, "validation", "after_terminal_subprocesses", passed=terminal_passed)

    phase_started = time.monotonic()
    artifact = _build_artifact(
        started_at=started_at,
        duration_s=time.monotonic() - started,
        spans=spans,
        preconditions=preconditions,
        hashes=hashes,
        original=original,
        candidate=candidate,
        schedule_manifest=schedule_manifest,
        candidate_manifest=candidate_manifest,
        reduction=reduction,
        independent=independent,
        validation_receipts=[*affected_receipts, *terminal_receipts],
        affected_passed=affected_passed,
        terminal_passed=terminal_passed,
        stage="terminal",
    )
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = time.monotonic() - started
    _close_span(spans, "write", phase_started, started, 1)
    artifact["phase_spans"] = deepcopy(spans)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validation_errors = validate_artifact(artifact, repo_root=root)
    if validation_errors:
        artifact["status"] = "complete_capture_reducer_disqualified"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_capture_reducer_validation_failed"
        artifact["flagged_adversarial"] = True
        artifact["capture_reducer_ready_score"] = 0
        artifact["scientific_value_score"] = 0
        artifact["promotion_score"] = 0
        artifact["internal_validation_errors"] = validation_errors
        artifact["field_principles"]["internal_validation_errors"] = (
            "Retain each exact current validation failure and prevent promotion."
        )
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    progress(started, "write", "before_atomic", path=output_path)
    validation_contract.atomic_json(root / output_path, artifact)
    progress(
        started,
        "write",
        "after_atomic",
        path=output_path,
        ready=artifact["capture_reducer_ready_score"],
    )
    return artifact


def _date_argument(value: str) -> str:
    """Reject a run date outside the fixed V646 execution contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the cold correction or validate one existing correction record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(_load_object(args.validate))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = run_experiment(run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "capture_reducer_ready_score": artifact["capture_reducer_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
