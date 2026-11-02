"""Report a blocked portable-calibration branch without inventing measurements.

The roadmap permits Rust parity and service timing only after Exp7561 is valid
and ready. This module still preserves board history and exact validation when
that external gate is closed.

Spec refs: REQ-REPORT-7571 and SCENARIO-REPORT-7571-*.
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
import shutil
import sys
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
from carnot.experiment_7367_v646_board_disposition import search_changed_state_receipt
from carnot.experiment_7473_v654_board_continuity import normalize_changed_state
from carnot.experiment_7513_v657_placement_continuity import reduce_board_rows
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7571-v661-portable-calibration"
SCHEMA = "carnot.exp7571.v661.portable_calibration.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7571_v661_portable_calibration.json")
RAW_DIR = Path("results/raw/experiment_7571_v661_portable_calibration")
MODULE_PATH = Path("python/carnot/experiment_7571_v661_portable_calibration.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7571_v661_portable_calibration.py")
TEST_PATH = Path("tests/python/test_experiment_7571_v661_portable_calibration.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROTOTYPE_PATH = Path("results/experiment_7561_v661_recalibration_prototype.json")
SERVICE_PATH = Path("results/experiment_7558_v660_service_boundary.json")
PROTOCOL_PATH = Path(
    "results/raw/experiment_7561_v661_recalibration_prototype/frozen_learning_protocol.json"
)
STATE_SCHEMA_PATH = Path(
    "results/raw/experiment_7561_v661_recalibration_prototype/numerical_state_schema.json"
)
INPUT_PATHS = (
    PROTOTYPE_PATH,
    SERVICE_PATH,
    PROTOCOL_PATH,
    STATE_SCHEMA_PATH,
    SPEC_PATH,
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("research-hardware-wishlist.md"),
)
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_blocked_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or no data when external bytes are unusable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and preserve the source's literal terminal state."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    value = load_object(resolved)
    return {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "original_honest_verdict": value.get("honest_verdict"),
        "original_verdict_class": value.get("verdict_class"),
        "original_flagged_adversarial": value.get("flagged_adversarial"),
    }


def _check(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:
    """Record both operands so a failed gate cannot become a zero result."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "eq",
        "passed": observed == expected,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate producer state, specifications, and local resources first."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in INPUT_PATHS:
        path = repo / relative
        present = path.is_file() and path.stat().st_size > 0
        rows.append(
            _check(
                f"source_readable:{relative.as_posix()}",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
            )
        )
        if present:
            sources[relative.as_posix()] = source_row(path, repo)

    prototype = load_object(repo / PROTOTYPE_PATH)
    service = load_object(repo / SERVICE_PATH)
    readiness = prototype.get("recalibration_ready_score")
    rows.append(
        _check(
            "exp7561_recalibration_ready_score",
            "Exp7561",
            PROTOTYPE_PATH.as_posix(),
            "recalibration_ready_score",
            1,
            readiness,
        )
    )
    spec_text = (
        (repo / SPEC_PATH).read_text(encoding="utf-8") if (repo / SPEC_PATH).is_file() else ""
    )
    rows.append(
        _check(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7571",
            "REQ-REPORT-7571" if "REQ-REPORT-7571" in spec_text else None,
        )
    )
    tmp = os.statvfs("/tmp")
    free_bytes = tmp.f_bavail * tmp.f_frsize
    rows.append(
        _check(
            "scratch_capacity",
            "/tmp",
            "/tmp",
            "free_bytes_at_least_16MiB",
            True,
            free_bytes >= 16 * 1024 * 1024,
        )
    )
    failed = next((row for row in rows if row["passed"] is not True), None)
    blocker = (
        {
            key: deepcopy(failed[key])
            for key in ("check", "upstream", "path", "field", "expected", "observed", "op")
        }
        if failed
        else None
    )
    allowed_class = prototype.get("verdict_class") in {
        "positive",
        "null",
        "circular_positive",
    }
    fixture = prototype.get("fixture_summary") or {}
    numerical = prototype.get("numerical_qualification") or {}
    solver_state = {
        "verdict_class": prototype.get("verdict_class"),
        "allowed_verdict_class": allowed_class,
        "flagged_adversarial": prototype.get("flagged_adversarial"),
        "numerical_qualification_passed": numerical.get("passed"),
        "fixture_lifecycle_passed": fixture.get("fixture_gates_passed"),
        "primary_solver": numerical.get("primary_solver"),
        "independent_solver": numerical.get("independent_solver"),
        "state_schema_path": STATE_SCHEMA_PATH.as_posix(),
        "protocol_path": PROTOCOL_PATH.as_posix(),
    }
    return {
        "rows": rows,
        "blocker": blocker,
        "kernel_branch_ready": bool(
            failed is None
            and allowed_class
            and prototype.get("flagged_adversarial") is False
            and numerical.get("passed") is True
            and fixture.get("fixture_gates_passed") is True
        ),
        "prototype": prototype,
        "service_artifact": service,
        "prototype_solver_and_fixture_state": solver_state,
        "source_artifact_hashes": sources,
        "resource_observations": {
            "logical_cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cargo_path": shutil.which("cargo"),
            "tmp_free_bytes": free_bytes,
            "rust_measurement_started": False,
        },
    }


def scan_gatemate_receipts(root: Path, raw_path: Path) -> JsonDict:
    """Search approved local receipt files without contacting any board."""

    result = search_changed_state_receipt(root, raw_path)
    raw = load_object(raw_path)
    candidates = [row for row in raw.get("candidate_rows") or [] if isinstance(row, Mapping)]
    dates = [str(row["receipt_date"]) for row in candidates if row.get("receipt_date")]
    result["latest_receipt_date"] = max(dates) if dates else "20260823"
    return normalize_changed_state(result)


def build_board_rows(
    service_artifact: Mapping[str, Any], changed_state: Mapping[str, Any]
) -> list[JsonDict]:
    """Retain fabric, CPU-dispatch, and physical-blocker scopes separately."""

    rows, _summary = reduce_board_rows(
        {"board_rows": deepcopy(list(service_artifact.get("board_rows") or []))}, changed_state
    )
    for row in rows:
        row["historical_source"] = SERVICE_PATH.as_posix()
        row["present_reachability_asserted"] = False
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
        if row.get("board") == "GateMate":
            accepted = int((changed_state.get("accepted_receipt_count", 0) or 0))
            row["gate_check_summary"] = {
                "check": "dated_operator_physical_change_receipt",
                "upstream": "operator-authored GateMate receipt search",
                "path": changed_state.get("search_receipt_path"),
                "field": "accepted_receipt_count",
                "expected": ">=1",
                "observed": accepted,
                "op": ">=",
                "passed": accepted >= 1,
            }
    return rows


def _receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one successful receipt for every scoped and terminal command."""

    expected = {*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        grouped.setdefault(str(row.get("name") or ""), []).append(row)
    return all(
        len(grouped.get(name, [])) == 1
        and grouped[name][0].get("passed") is True
        and grouped[name][0].get("exit_code") == 0
        and grouped[name][0].get("timed_out") is not True
        for name in expected
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute blocked readiness, service accounting, and board continuity."""

    rows = list(value.get("board_rows") or [])
    boards_complete = bool(
        [row.get("board") for row in rows] == ["KV260", "PolarFire", "GateMate"]
        and rows[0].get("exact_claim_scope") == "historical_kv260_fpga_fabric_sampling_only"
        and rows[0].get("future_access") == "ssh kria only"
        and rows[0].get("architecture_limit") == "k_max<=5"
        and rows[1].get("exact_claim_scope")
        == "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
        and rows[1].get("fpga_sampling_claimed") is False
        and rows[2].get("current_disposition")
        in {
            "blocked_unchanged_physical_prerequisite",
            "changed_physical_prerequisite_recorded_future_probe_only",
        }
        and all(row.get("hardware_operations_issued") == [] for row in rows)
    )
    budget = value.get("sample_size_budget") or {}
    parity = budget.get("parity_cases") or {}
    service = budget.get("paired_service_trials") or {}
    blocked = value.get("external_blocker") or {}
    no_kernel = bool(
        blocked
        and parity.get("attempted") == 0
        and parity.get("completed") == 0
        and parity.get("unstarted") == 10_000
    )
    no_service = bool(
        blocked
        and service.get("attempted") == 0
        and service.get("completed") == 0
        and service.get("unstarted") == 30
    )
    declarations = bool(
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate_class") == "no_model_load"
        and value.get("inference_substrate") == "aggregation_from_upstream_artifacts"
        and value.get("execution_venue") == "host"
    )
    return {
        "portable_kernel_ready_score": int(not blocked and not no_kernel),
        "service_measurement_complete_score": int(not blocked and not no_service),
        "board_continuity_complete_score": int(boards_complete),
        "current_inference_declarations_valid": declarations,
        "required_validation_passed": _receipts_pass(value.get("validation_receipts") or []),
        "parity_cases_started": not no_kernel,
        "paired_service_trials_started": not no_service,
        "board_row_count": len(rows),
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str,
    path: str,
    field: str,
) -> JsonDict:
    """Attach one exact operand and the scientific boundary it protects."""

    return {
        "check": check,
        "category": category,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "eq",
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Keep external readiness, validity, continuity, and benefit separate."""

    blocker = value.get("external_blocker") or {}
    return [
        _gate(
            str(blocker.get("check") or "exp7561_recalibration_ready_score"),
            "readiness",
            blocker.get("expected", 1),
            blocker.get("observed", 1),
            not blocker,
            "A dependent kernel cannot repair an invalid upstream prototype.",
            upstream=str(blocker.get("upstream") or "Exp7561"),
            path=str(blocker.get("path") or PROTOTYPE_PATH),
            field=str(blocker.get("field") or "recalibration_ready_score"),
        ),
        _gate(
            "board_continuity_complete",
            "readiness",
            1,
            reduction["board_continuity_complete_score"],
            reduction["board_continuity_complete_score"] == 1,
            "Each board retains its own historical or blocked scope.",
            upstream="Exp7558",
            path="board_rows",
            field="board_continuity_complete_score",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            reduction["required_validation_passed"],
            reduction["required_validation_passed"] is True,
            "Invalid evidence cannot support science.",
            upstream="current_work",
            path="validation_receipts",
            field="required_validation_passed",
        ),
        _gate(
            "measured_portable_or_hardware_benefit",
            "benefit",
            True,
            False,
            False,
            "Completion cannot substitute for empirical value.",
            upstream="current_work",
            path="kernel_and_service_costs",
            field="measured_benefit",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]], blocker: Mapping[str, Any]) -> JsonDict:
    """Make the exact upstream failure the first terminal diagnostic."""

    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [row.get("check") for row in failed],
        "first_failure": deepcopy(dict(blocker)) if blocker else deepcopy(failed[0]),
    }


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Name the omission or false claim that every emitted field prevents."""

    specific = {
        "experiment_id": "Exact identity prevents another task from substituting evidence.",
        "preconditions_checked": "Observed inputs prevent fabricated fallback data.",
        "MODEL_SPECS": "An empty plan prevents historical model work becoming current work.",
        "model_specs": "No resolved model identity is valid for no-load work.",
        "model_invoked": "False prevents aggregation from being called inference.",
        "invocation_counts": "Typed zero counters expose hidden current calls.",
        "inference_substrate_class": "The no-load class applies the correct runtime floor.",
        "inference_substrate": "The substrate separates byte reduction from generation.",
        "execution_venue": "A legal venue keeps CPU identity separate.",
        "duration_s": "Monotonic timing prevents invented runtime claims.",
        "random_seed": "Frozen seeds prevent outcome-dependent ordering.",
        "reproducibility_checksum": "The checksum binds settings, evidence, code, and rows.",
        "rows": "Every board remains visible and missing is not zero.",
        "sample_size_budget": "Unstarted units prevent external absence becoming failure data.",
        "acceptance_gate_results": "Typed gates separate validity, readiness, and benefit.",
        "gate_check_summary": "Exact failed operands make the block reproducible.",
        "honest_verdict": "The complete prefix closes an externally blocked task.",
        "verdict_class": "The closed class prevents blocked evidence becoming null.",
        "verifier_is_oracle": "Probabilistic energy cannot certify source truth.",
        "flagged_adversarial": "The safety determination cannot be erased to open a gate.",
        "validation_receipts": "Exact command outcomes prevent unrun checks appearing green.",
        "field_principles": "Every field states the failure that it prevents.",
        "portable_kernel_ready_score": "Compiled parity is required before portability is ready.",
        "service_measurement_complete_score": "Thirty equal-durability pairs are required.",
        "board_continuity_complete_score": "All three distinct board scopes are required.",
        "board_rows": "Historical fabric, CPU dispatch, and physical blocks stay distinct.",
        "kernel_and_service_costs": "Different denominators cannot be merged into a speedup.",
        "hardware_acceleration_bound": "Host execution cannot imply device gain.",
    }
    return {
        key: specific.get(key, f"Retaining {key} prevents silent omission or scope drift.")
        for key in keys
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable evidence while excluding clocks and this self-reference."""

    excluded = {
        "reproducibility_checksum",
        "duration_s",
        "phase_spans",
        "process_identity",
        "started_at_utc",
        "completed_at_utc",
    }
    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key not in excluded}
    )


def _sample_budget() -> JsonDict:
    """Retain preregistered units as unstarted after the external gate closes."""

    return {
        "parity_cases": {
            "planned": 10_000,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 10_000,
        },
        "paired_service_trials": {
            "planned": 30,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 30,
            "events_per_trial": 256,
        },
    }


def _blocked_verdict(blocker: Mapping[str, Any]) -> str:
    """Convert one exact failed check into a stable terminal suffix."""

    reason = "".join(
        character if character.isalnum() else "_" for character in str(blocker.get("check"))
    ).strip("_")
    return f"complete_blocked_{reason or 'external_precondition'}"


def build_artifact(
    *,
    root: Path,
    preconditions: Mapping[str, Any],
    board_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] = (),
    evidence_mode: str = "sealed_real",
    started_at_utc: str | None = None,
    completed_at_utc: str | None = None,
) -> JsonDict:
    """Assemble the complete blocked result without dependent measurements."""

    blocker = deepcopy(dict(preconditions.get("blocker") or {}))
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "Portable constrained-calibration kernel and board continuity",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "worktree_root": str(root.resolve()),
        "evidence_mode": evidence_mode,
        "preconditions_checked": deepcopy(list(preconditions.get("rows") or [])),
        "source_artifact_hashes": deepcopy(dict(preconditions.get("source_artifact_hashes") or {})),
        "input_roles": {
            PROTOTYPE_PATH.as_posix(): "external_kernel_readiness_gate",
            SERVICE_PATH.as_posix(): "historical_service_and_board_custody",
        },
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_calls": {"counted_as_current": False},
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "device_identity": {
            "device_type": "CPU",
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "duration_s": float(duration_s),
        "phase_spans": deepcopy(list(phase_spans)),
        "process_identity": {
            "pid": os.getpid(),
            "python": sys.executable,
            "worktree": str(root.resolve()),
        },
        "random_seed": {
            "model": None,
            "fitting": 7_561_001,
            "ordering": 7_571_001,
            "bootstrap": 7_571_002,
        },
        "prototype_solver_and_fixture_state": deepcopy(
            dict(preconditions.get("prototype_solver_and_fixture_state") or {})
        ),
        "resource_observations": deepcopy(dict(preconditions.get("resource_observations") or {})),
        "rows": deepcopy(list(board_rows)),
        "board_rows": deepcopy(list(board_rows)),
        "sample_size_budget": _sample_budget(),
        "external_blocker": blocker,
        "honest_verdict": _blocked_verdict(blocker),
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "positive_claim": False,
        "positive_portability": False,
        "predictive_benefit_claimed": False,
        "hardware_speed_claimed": False,
        "portable_kernel_ready_score": 0,
        "service_measurement_complete_score": 0,
        "board_continuity_complete_score": 0,
        "kernel_and_service_costs": {
            "status": "not_measured_external_prototype_blocked",
            "python_warm_kernel_ns": None,
            "rust_warm_kernel_ns": None,
            "kernel_speedup_ratio": None,
            "process_startup_ns": None,
            "state_transfer_ns": None,
            "complete_service_python_ns": None,
            "complete_service_rust_ns": None,
            "complete_service_speedup_ratio": None,
            "durability_semantics_equal": None,
        },
        "hardware_acceleration_bound": {
            "denominator": "complete_service_time_not_measured",
            "measured_replaceable_fraction": None,
            "ideal_kernel_only_speedup": None,
            "device_speedup_assumed": None,
            "board_or_device_gain_claimed": False,
            "placement_decision": "deferred_no_measured_cost",
        },
        "hardware_operations_issued": [],
        "generator_weights_changed": False,
        "external_publication_performed": False,
        "research_conductor_modified": False,
        "e2e_004": {
            "applicable": False,
            "status": "not_started_external_prototype_blocked",
            "reason": "The blocked branch adds no shared Rust binding or serialized kernel.",
        },
        "capability_e2e": {
            "declared_entrypoint": True,
            "cold_replay": True,
            "predict_release_update_persist_reload": "not_started_external_prototype_blocked",
        },
        "validation_receipts": deepcopy(list(validation_receipts)),
        "validation_summary": {
            "required_checks_passed": _receipts_pass(validation_receipts),
            "scoped_only": True,
            "unscoped_suite_run": False,
        },
        "repository_health": {
            "status": "not_evaluated_by_scoped_task",
            "pre_existing_debt_affects_required_checks": False,
        },
    }
    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["portable_kernel_ready_score"] = reduction["portable_kernel_ready_score"]
    value["service_measurement_complete_score"] = reduction["service_measurement_complete_score"]
    value["board_continuity_complete_score"] = reduction["board_continuity_complete_score"]
    gates = _acceptance_gates(value, reduction)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates, blocker)
    value["field_principles"] = _field_principles(
        [*value, "field_principles", "reproducibility_checksum"]
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def _private_context() -> JsonDict:
    """Build a deterministic blocked context for schema and mutation tests."""

    real = collect_preconditions(REPO_ROOT)
    blocker = {
        "check": "exp7561_recalibration_ready_score",
        "upstream": "Exp7561",
        "path": PROTOTYPE_PATH.as_posix(),
        "field": "recalibration_ready_score",
        "expected": 1,
        "observed": 0,
        "op": "eq",
    }
    return {
        "rows": [
            {
                **deepcopy(blocker),
                "passed": False,
            }
        ],
        "blocker": blocker,
        "source_artifact_hashes": {},
        "prototype_solver_and_fixture_state": deepcopy(real["prototype_solver_and_fixture_state"]),
        "resource_observations": {"fixture": True, "rust_measurement_started": False},
        "service_artifact": real["service_artifact"],
    }


def build_test_artifact(
    root: Path, *, validation_receipts: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build one compact blocked artifact for pure reader tests."""

    context = _private_context()
    changed_state = {
        "accepted_receipt_count": 0,
        "exists": False,
        "search_receipt_path": "private/gatemate_receipt_search.json",
    }
    boards = build_board_rows(context["service_artifact"], changed_state)
    return build_artifact(
        root=root,
        preconditions=context,
        board_rows=boards,
        validation_receipts=validation_receipts,
        duration_s=0.1,
        evidence_mode="private_fixture",
    )


def _verify_sources(value: Mapping[str, Any], root: Path) -> list[str]:
    """Re-hash every declared source without repairing missing or changed bytes."""

    errors: list[str] = []
    for label, row in dict(value.get("source_artifact_hashes") or {}).items():
        path = Path(str(row.get("path") or label))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
            errors.append(f"source_hash_invalid:{label}")
        elif resolved.stat().st_size != row.get("bytes"):
            errors.append(f"source_size_invalid:{label}")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Reject identity, blocked accounting, custody, gate, or checksum drift."""

    errors: list[str] = []
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("artifact_identity_mismatch")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("task_binding_mismatch")
    if value.get("MODEL_SPECS") != [] or value.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if (
        value.get("model_invoked") is not False
        or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_invocation_claim_invalid")
    if (
        value.get("inference_substrate_class") != "no_model_load"
        or value.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or value.get("execution_venue") != "host"
    ):
        errors.append("inference_declaration_invalid")
    blocker = value.get("external_blocker")
    required_blocker = {"check", "upstream", "path", "field", "expected", "observed", "op"}
    if not isinstance(blocker, Mapping) or not required_blocker <= set(blocker):
        errors.append("external_blocker_incomplete")
        blocker = {}
    if value.get("verdict_class") != "blocked" or value.get("honest_verdict") != _blocked_verdict(
        blocker
    ):
        errors.append("terminal_verdict_mismatch")
    for field in (
        "portable_kernel_ready_score",
        "service_measurement_complete_score",
        "board_continuity_complete_score",
    ):
        if type(value.get(field)) is not int or value.get(field) not in (0, 1):
            errors.append(f"score_not_bare_numeric:{field}")
    try:
        reduction = independent_reduce(value)
    except (AttributeError, IndexError, TypeError, ValueError):
        reduction = {}
        errors.append("independent_reduction_failed")
    if reduction:
        if value.get("independent_reduction") != reduction:
            errors.append("independent_reduction_mismatch")
        for field in (
            "portable_kernel_ready_score",
            "service_measurement_complete_score",
            "board_continuity_complete_score",
        ):
            if value.get(field) != reduction[field]:
                errors.append(f"{field}_mismatch")
        gates = _acceptance_gates(value, reduction)
        if value.get("acceptance_gate_results") != gates:
            errors.append("acceptance_gate_results_mismatch")
        if value.get("gate_check_summary") != _gate_summary(gates, blocker):
            errors.append("gate_check_summary_mismatch")
    costs = value.get("kernel_and_service_costs") or {}
    forbidden_costs = (
        "python_warm_kernel_ns",
        "rust_warm_kernel_ns",
        "kernel_speedup_ratio",
        "process_startup_ns",
        "state_transfer_ns",
        "complete_service_python_ns",
        "complete_service_rust_ns",
        "complete_service_speedup_ratio",
        "durability_semantics_equal",
    )
    if any(costs.get(field) is not None for field in forbidden_costs):
        errors.append("blocked_cost_must_be_unmeasured")
    bound = value.get("hardware_acceleration_bound") or {}
    if any(
        bound.get(field) is not None
        for field in (
            "measured_replaceable_fraction",
            "ideal_kernel_only_speedup",
            "device_speedup_assumed",
        )
    ):
        errors.append("blocked_hardware_bound_must_be_unmeasured")
    if (
        value.get("positive_portability") is not False
        or value.get("predictive_benefit_claimed") is not False
        or value.get("hardware_speed_claimed") is not False
        or value.get("hardware_operations_issued") != []
    ):
        errors.append("blocked_claim_scope_invalid")
    if not _receipts_pass(value.get("validation_receipts") or []):
        errors.append("required_validation_failed")
    principles = value.get("field_principles") or {}
    if set(value) - set(principles):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results") or []
    if any(not row.get("principle") for row in gates):
        errors.append("gate_principle_missing")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    if verify_sources:
        errors.extend(_verify_sources(value, root))
    return list(dict.fromkeys(errors))


def cold_replay(path: Path, *, root: Path = REPO_ROOT, verify_sources: bool = True) -> list[str]:
    """Read serialized bytes through the public defensive validator."""

    value = load_object(path)
    if not value:
        return ["artifact_not_object"]
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def independent_replay(path: Path, *, verify_sources: bool = True) -> list[str]:
    """Recompute blocked and board scores in a fresh reader process."""

    value = load_object(path)
    if not value:
        return ["artifact_not_object"]
    root = Path(str(value.get("worktree_root") or REPO_ROOT)).resolve()
    return validate_artifact(value, root=root, verify_sources=verify_sources)


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze serial tests, changed-module coverage, lint, type, and spec checks."""

    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if errors:  # pragma: no cover - the shared builder returns a valid frozen plan.
        raise ValueError("validation_plan_invalid:" + ",".join(errors))
    return commands


def terminal_commands(candidate: Path, root: Path = REPO_ROOT) -> list[PlannedCommand]:
    """Build four bounded commands over the exact terminal candidate bytes."""

    common = ("--date", RUN_DATE, "--root", str(root.resolve()))
    specifications = (
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[0],
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[1],
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[2],
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_terminal_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            TERMINAL_CHECK_NAMES[3],
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact_terminal_candidate",
            300.0,
        ),
    )
    return [PlannedCommand(specification, "validity", True) for specification in specifications]


def progress(  # pragma: no cover - visible through the declared entrypoint.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase and slow-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7571] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(  # pragma: no cover - monotonic timing belongs to the entrypoint.
    phase: str, phase_started: float, run_started: float, units: int
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_offset_s": phase_started - run_started,
        "end_offset_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
    }


def _provisional_terminal_receipts() -> list[JsonDict]:  # pragma: no cover
    """Give a private candidate its final reader shape before commands run."""

    return [
        {
            "name": name,
            "command": "pending exact terminal candidate check",
            "command_argv": ["pending", name],
            "scope": "private_provisional_candidate",
            "exit_code": 0,
            "duration_s": 0.0,
            "log_path": "pending",
            "log_sha256": "sha256:" + "0" * 64,
            "passed": True,
            "timed_out": False,
            "output_tail": "private provisional receipt; never published",
            "command_category": "validity",
            "required": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]


def _append_owned_sources(
    root: Path, context: JsonDict, manifest: Path
) -> None:  # pragma: no cover
    """Bind implementation, tests, entrypoint, specification, and manifest bytes."""

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        context["source_artifact_hashes"][relative.as_posix()] = source_row(root / relative, root)
    label = manifest.relative_to(root).as_posix()
    context["source_artifact_hashes"][label] = source_row(manifest, root)


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str, *, output_path: Path | None = None
) -> JsonDict:
    """Authenticate, preserve boards, validate exact bytes, and publish atomically."""

    if run_date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    repo = root.resolve()
    destination = output_path or repo / RESULT_PATH
    raw_root = repo / RAW_DIR
    candidate_path = raw_root / "measured_terminal_candidate.json"
    exact_path = raw_root / "exact_terminal_candidate.json"
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    raw_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        raw_root / "preconditions_checkpoint.json",
        {
            "rows": context["rows"],
            "blocker": context["blocker"],
            "resource_observations": context["resource_observations"],
            "prototype_solver_and_fixture_state": context["prototype_solver_and_fixture_state"],
        },
    )
    spans.append(_span("preconditions", phase_started, started, len(context["rows"])))
    progress(
        started,
        "preconditions",
        "complete",
        completed_units=len(context["rows"]),
        kernel_branch_ready=context["kernel_branch_ready"],
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0))
        progress(started, phase, "after", completed=0)

    progress(started, "board_continuity", "before_receipt_search", planned=3)
    phase_started = time.monotonic()
    changed_state = scan_gatemate_receipts(repo, raw_root / "gatemate_changed_state_evidence.json")
    boards = build_board_rows(context["service_artifact"], changed_state)
    spans.append(_span("board_continuity", phase_started, started, len(boards)))
    progress(
        started,
        "board_continuity",
        "after_receipt_search",
        completed=len(boards),
        hardware_operations=0,
    )
    if context["kernel_branch_ready"]:
        raise RuntimeError("ready_kernel_branch_requires_rust_implementation")

    progress(started, "kernel_parity", "before", planned=10_000)
    phase_started = time.monotonic()
    spans.append(_span("kernel_parity", phase_started, started, 0))
    progress(started, "kernel_parity", "after", completed=0, unstarted=10_000)
    progress(started, "service_benchmark", "before", planned=30)
    phase_started = time.monotonic()
    spans.append(_span("service_benchmark", phase_started, started, 0))
    progress(started, "service_benchmark", "after", completed=0, unstarted=30)

    progress(started, "manifest", "start")
    manifest_path = raw_root / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )
    _append_owned_sources(repo, context, manifest_path)
    progress(started, "manifest", "complete", completed=1)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7571-", dir="/tmp"))
    commands = build_validation_commands(repo, private_root)
    planned = [PlannedCommand(command, "validity", True) for command in commands]
    progress(started, "scoped_validation", "before_subprocesses", planned=len(planned))
    phase_started = time.monotonic()
    affected = run_categorized_commands(
        repo,
        planned,
        log_dir=raw_root / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_summary = reduce_affected_receipts(repo, AFFECTED_MANIFEST, affected)
    spans.append(_span("scoped_validation", phase_started, started, len(affected)))
    progress(
        started,
        "scoped_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_summary["passed"],
    )
    if affected_summary["passed"] is not True:
        raise RuntimeError("required_scoped_validation_failed")

    provisional = build_artifact(
        root=repo,
        preconditions=context,
        board_rows=boards,
        validation_receipts=[*affected, *_provisional_terminal_receipts()],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, provisional)
    progress(started, "candidate", "after_serialization")

    first_plan = terminal_commands(candidate_path, repo)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(first_plan))
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        repo,
        first_plan,
        log_dir=raw_root / "validation/terminal_provisional",
        heartbeat_s=60.0,
    )
    spans.append(_span("terminal_validation", phase_started, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=all(row.get("passed") is True for row in terminal),
    )
    if not all(row.get("passed") is True for row in terminal):
        raise RuntimeError("terminal_candidate_validation_failed")

    final = build_artifact(
        root=repo,
        preconditions=context,
        board_rows=boards,
        validation_receipts=[*affected, *terminal],
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=datetime.now(UTC).isoformat(),
    )
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError("terminal_artifact_invalid:" + ",".join(errors))
    atomic_json(exact_path, final)

    exact_plan = terminal_commands(exact_path, repo)
    progress(
        started,
        "exact_candidate_validation",
        "before_subprocesses",
        planned=len(exact_plan),
    )
    exact = run_categorized_commands(
        repo,
        exact_plan,
        log_dir=raw_root / "validation/terminal_exact",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "exact_candidate_validation",
        "after_subprocesses",
        completed=len(exact),
        passed=all(row.get("passed") is True for row in exact),
    )
    if not all(row.get("passed") is True for row in exact):
        raise RuntimeError("exact_terminal_candidate_validation_failed")

    progress(started, "publish", "before_atomic_terminal", path=destination)
    atomic_json(destination, final)
    if sha256_file(destination) != sha256_file(exact_path):
        raise RuntimeError("published_bytes_differ_from_exact_candidate")
    if destination.stat().st_size >= 20 * 1024 * 1024:
        raise RuntimeError("terminal_artifact_exceeds_20MiB")
    progress(
        started,
        "publish",
        "after_atomic_terminal",
        bytes=destination.stat().st_size,
        verdict=final["honest_verdict"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date, root, output, and two read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    args = parser.parse_args(argv)
    if args.date != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one strict read-only candidate check."""

    args = parse_args(argv)
    if args.cold_replay:
        errors = cold_replay(
            args.cold_replay,
            root=args.root,
            verify_sources=not args.no_source_check,
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce:
        errors = independent_replay(
            args.independent_reduce,
            verify_sources=not args.no_source_check,
        )
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    output = (  # pragma: no cover - producer path runs through the declared capability E2E.
        args.output if args.output.is_absolute() else args.root / args.output
    )
    run_experiment(args.root, args.date, output_path=output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
