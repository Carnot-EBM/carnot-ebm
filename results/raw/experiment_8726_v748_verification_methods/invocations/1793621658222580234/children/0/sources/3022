"""Measure the count-memory service boundary without changing board state.

The service branch consumes Exp7523 only after that producer has run. The board
branch remains useful when the producer is absent. Historical timing stays
context and never becomes an equal-semantics speedup claim.

Spec refs: REQ-HW-7528 and SCENARIO-HW-7528-*.
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
from carnot.experiment_7513_v657_placement_continuity import (
    _scan_gatemate as scan_gatemate,
    reduce_board_rows,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7528-v658-service-boundary"
SCHEMA = "carnot.exp7528.v658.service_boundary.v1"
RESULT_PATH = Path("results/experiment_7528_v658_service_boundary.json")
RAW_DIR = Path("results/raw/experiment_7528_v658_service_boundary")
MODULE_PATH = Path("python/carnot/experiment_7528_v658_service_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7528_v658_service_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7528_v658_service_boundary.py")
SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
COUNT_MODULE_PATH = Path("python/carnot/experiment_7523_v658_count_memory.py")
COUNT_ARTIFACT_PATH = Path("results/experiment_7523_v658_count_memory.json")
PLACEMENT_PATH = Path("results/experiment_7513_v657_placement_continuity.json")
SERVICE_TRACE_PATH = Path("results/experiment_7514_v657_service_trace.json")
GRADUATION_PATH = Path("results/experiment_7314_v642_board_continuity.json")

MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
NAMED_INPUTS = (
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
    Path("python/carnot/experiment_7513_v657_placement_continuity.py"),
    Path("python/carnot/experiment_7514_v657_service_trace.py"),
    PLACEMENT_PATH,
    SERVICE_TRACE_PATH,
    GRADUATION_PATH,
    Path("research-hardware-wishlist.md"),
    SPEC_PATH,
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
AFFECTED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for absent or invalid bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and retain a source artifact's terminal flags."""

    resolved = path.resolve()
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


def _precondition(
    check: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    branch: str,
) -> JsonDict:
    """Store exact prerequisite operands so missing data cannot become zero."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "branch": branch,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate named inputs before reading the same-milestone producer."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    for relative in NAMED_INPUTS:
        path = repo / relative
        present = path.is_file() and path.stat().st_size > 0
        branch = "board" if relative in {PLACEMENT_PATH, GRADUATION_PATH} else "required"
        rows.append(
            _precondition(
                "resource_readable",
                relative.as_posix(),
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
                branch=branch,
            )
        )
        if present:
            sources[relative.as_posix()] = source_row(path, repo)

    spec_text = (repo / SPEC_PATH).read_text(encoding="utf-8")
    rows.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-HW-7528",
            "REQ-HW-7528" if "REQ-HW-7528" in spec_text else None,
            branch="required",
        )
    )
    exclusion_text = (repo / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    rows.append(
        _precondition(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            EXPERIMENT_ID in exclusion_text,
            branch="required",
        )
    )

    count_module_present = (repo / COUNT_MODULE_PATH).is_file()
    count_artifact_present = (repo / COUNT_ARTIFACT_PATH).is_file()
    rows.extend(
        [
            _precondition(
                "count_memory_module_available",
                "Exp7523",
                COUNT_MODULE_PATH.as_posix(),
                "presence",
                True,
                count_module_present,
                branch="service",
            ),
            _precondition(
                "count_memory_artifact_available",
                "Exp7523",
                COUNT_ARTIFACT_PATH.as_posix(),
                "presence",
                True,
                count_artifact_present,
                branch="service",
            ),
        ]
    )
    count_artifact: JsonDict = {}
    if count_module_present and count_artifact_present:
        count_artifact = load_object(repo / COUNT_ARTIFACT_PATH)
        sources[COUNT_ARTIFACT_PATH.as_posix()] = source_row(repo / COUNT_ARTIFACT_PATH, repo)
        rows.append(
            _precondition(
                "count_memory_ready",
                "Exp7523",
                COUNT_ARTIFACT_PATH.as_posix(),
                "count_memory_ready_score",
                1,
                count_artifact.get("count_memory_ready_score"),
                branch="service",
            )
        )

    placement = load_object(repo / PLACEMENT_PATH)
    trace = load_object(repo / SERVICE_TRACE_PATH)
    graduation = load_object(repo / GRADUATION_PATH)
    rows.extend(
        [
            _precondition(
                "latest_board_ledger_complete",
                "Exp7513",
                PLACEMENT_PATH.as_posix(),
                "board_continuity_complete_score",
                1,
                placement.get("board_continuity_complete_score"),
                branch="board",
            ),
            _precondition(
                "graduation_ledger_complete",
                "Exp7314",
                GRADUATION_PATH.as_posix(),
                "board_continuity_complete_score",
                1,
                graduation.get("board_continuity_complete_score"),
                branch="board",
            ),
        ]
    )
    required_ready = all(row["passed"] for row in rows if row["branch"] == "required")
    service_failed = next(
        (row for row in rows if row["branch"] == "service" and not row["passed"]), None
    )
    board_ready = required_ready and all(row["passed"] for row in rows if row["branch"] == "board")
    blocker = None
    if service_failed is not None:
        blocker = {
            key: service_failed[key]
            for key in ("check", "upstream", "path", "field", "expected", "observed")
        }
    return {
        "rows": rows,
        "source_artifact_hashes": sources,
        "required_ready": required_ready,
        "service_ready": required_ready and service_failed is None,
        "board_ready": board_ready,
        "service_blocker": blocker,
        "count_artifact": count_artifact,
        "placement": placement,
        "service_trace": trace,
        "graduation": graduation,
    }


def audit_board_rows(
    placement: Mapping[str, Any], changed_state: Mapping[str, Any] | None
) -> tuple[list[JsonDict], JsonDict]:
    """Reuse the approved reducer and forbid current reachability inference."""

    rows, summary = reduce_board_rows(placement, changed_state)
    for row in rows:
        row["present_reachability_asserted"] = False
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
    return rows, summary


def historical_service_context(placement: Mapping[str, Any], trace: Mapping[str, Any]) -> JsonDict:
    """Retain historical cost while making all semantic mismatches explicit."""

    reduction = dict(trace.get("service_reduction") or {})
    bounds = dict(reduction.get("amdahl_bounds") or trace.get("amdahl_bounds") or {})
    return {
        "exp7514_update_only_ceiling": bounds.get("ideal_infinite_speed_update_kernel_ceiling"),
        "exp7514_update_kernel_share": bounds.get("measured_update_kernel_share"),
        "exp7514_shared_native_forward_calls": reduction.get("shared_native_forward_calls"),
        "exp7514_operation_parameter_count": 33,
        "exp7513_timed_parameter_count": 81,
        "exp7513_numeric_ready_score": placement.get("numeric_placement_ready_score"),
        "composition_class": "hypothetical_bound_only",
        "mismatch_checks": {
            "native_call_count_matched": False,
            "operation_context_matched": False,
            "durability_semantics_matched": False,
            "synchronization_semantics_matched": False,
        },
        "target_100x_met": False,
        "whole_service_speedup": None,
    }


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one passing receipt for every affected and terminal command."""

    required = set(AFFECTED_CHECK_NAMES) | set(TERMINAL_CHECK_NAMES)
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in receipts:
        by_name.setdefault(str(row.get("name")), []).append(row)
    return all(
        len(by_name.get(name, [])) == 1
        and by_name[name][0].get("passed") is True
        and by_name[name][0].get("exit_code") == 0
        and by_name[name][0].get("timed_out") is not True
        for name in required
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute service, board, declaration, and validation readiness."""

    board_rows = value.get("board_rows") or []
    boards = [row.get("board") for row in board_rows if isinstance(row, Mapping)]
    board_complete = bool(
        boards == ["KV260", "PolarFire", "GateMate"]
        and board_rows[0].get("exact_claim_scope") == "historical_kv260_fpga_fabric_sampling_only"
        and board_rows[0].get("future_access") == "ssh kria only"
        and board_rows[1].get("exact_claim_scope")
        == "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
        and board_rows[2].get("current_disposition")
        in {
            "blocked_unchanged_physical_prerequisite",
            "changed_physical_prerequisite_recorded_future_probe_only",
        }
        and all(row.get("hardware_operations_issued") == [] for row in board_rows)
        and all(row.get("present_reachability_asserted") is False for row in board_rows)
    )
    service_rows = value.get("service_rows") or []
    component_names = {
        "lookup_ns",
        "increment_ns",
        "allocation_ns",
        "serialization_ns",
        "fsync_ns",
        "acknowledgement_ns",
    }
    batch_256 = [row for row in service_rows if row.get("arm") == "batch_256"]
    batch_one = [row for row in service_rows if row.get("arm") == "batch_one_ack"]
    service_complete = bool(
        value.get("service_branch_status") == "complete_measured"
        and value.get("operation_identity", {}).get("bin_count") == 8
        and len(batch_256) == 30
        and len(batch_one) == 30
        and all(component_names <= set(row) for row in service_rows)
        and value.get("restart_parity", {}).get("exact_prediction_parity") is True
    )
    declarations = bool(
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate") == INFERENCE_SUBSTRATE
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("execution_venue") == EXECUTION_VENUE
    )
    return {
        "service_cost_complete_score": int(service_complete),
        "board_continuity_complete_score": int(board_complete),
        "current_inference_declarations_valid": declarations,
        "required_validation_passed": _validation_passed(value.get("validation_receipts") or []),
        "hardware_operation_count": sum(
            int(row.get("hardware_operation_count", 0) or 0) for row in board_rows
        ),
        "service_row_count": len(service_rows),
        "board_row_count": len(board_rows),
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
    op: str = "eq",
) -> JsonDict:
    """Keep every gate operand next to the failure it can prevent."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": op,
        "passed": passed,
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field,
    }


def _acceptance_gates(value: Mapping[str, Any], reduction: Mapping[str, Any]) -> list[JsonDict]:
    """Separate external availability, evidence validity, and benefit."""

    blocker = value.get("service_blocker") or {}
    return [
        _gate(
            str(blocker.get("check") or "count_memory_ready"),
            "readiness",
            blocker.get("expected", True),
            blocker.get("observed", value.get("service_branch_status") == "complete_measured"),
            not blocker,
            "The producer must exist before this task reads or times its operation.",
            upstream=str(blocker.get("upstream") or "Exp7523"),
            path=str(blocker.get("path") or COUNT_ARTIFACT_PATH.as_posix()),
            field=str(blocker.get("field") or "count_memory_ready_score"),
        ),
        _gate(
            "current_inference_declarations",
            "validity",
            True,
            reduction["current_inference_declarations_valid"],
            reduction["current_inference_declarations_valid"] is True,
            "Zero balanced counters prevent historical Qwen work from becoming current work.",
            upstream="current_work",
            path="invocation_counts",
            field="inference_declarations",
        ),
        _gate(
            "board_continuity_complete",
            "readiness",
            1,
            reduction["board_continuity_complete_score"],
            reduction["board_continuity_complete_score"] == 1,
            "The board audit stays complete when the service branch is blocked.",
            upstream="Exp7513+Exp7314",
            path="board_rows",
            field="board_continuity_complete_score",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            reduction["required_validation_passed"],
            reduction["required_validation_passed"] is True,
            "A favorable or blocked result cannot excuse a failed reader.",
            upstream="validation_receipts",
            path="validation_receipts",
            field="required_checks_passed",
        ),
        _gate(
            "exact_operation_service_cost",
            "readiness",
            1,
            reduction["service_cost_complete_score"],
            reduction["service_cost_complete_score"] == 1,
            "Only the exact eight-bin durable operation can complete service cost.",
            upstream="Exp7523",
            path="service_rows",
            field="service_cost_complete_score",
        ),
        _gate(
            "equal_semantics_whole_service_100x",
            "benefit",
            100.0,
            value.get("whole_service_speedup"),
            bool(
                isinstance(value.get("whole_service_speedup"), (int, float))
                and float(value["whole_service_speedup"]) >= 100.0
            ),
            "A historical component ceiling is not an equal-semantics whole service.",
            upstream="current_equal_semantics_comparison",
            path="whole_service_speedup",
            field="whole_service_speedup",
            op=">=",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """List all failed operands with exact source routing."""

    failed = [
        {
            key: row.get(key)
            for key in (
                "check",
                "category",
                "upstream",
                "path",
                "field",
                "expected",
                "observed",
            )
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_check": failed[0] if failed else None,
        "failed_checks": failed,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the evidence failure prevented by each emitted field."""

    specific = {
        "schema": "Version, experiment identity, and milestone bind the reader contract.",
        "run_date": "The fixed run date stays separate from measured UTC and monotonic time.",
        "preconditions_checked": "Exact paths and values prevent invented readiness.",
        "MODEL_SPECS": "An empty list proves that this run loaded no model.",
        "model_specs": "The lowercase mirror prevents a model alias from hiding a load.",
        "model_invoked": "False separates current work from historical Qwen provenance.",
        "invocation_counts": "Balanced zero counters expose a concealed model operation.",
        "inference_substrate_class": "No-model-load selects the correct evidence class.",
        "inference_substrate": "Aggregation names cached evidence instead of live inference.",
        "execution_venue": "Host CPU work stays distinct from historical board evidence.",
        "duration_s": "Measured elapsed time cannot be replaced by a padded estimate.",
        "phase_spans": "Phase boundaries expose waiting, skipped work, and validation time.",
        "random_seed": "Frozen protocol seeds prevent outcome-driven selection.",
        "reproducibility_checksum": "One digest binds sources, roles, rows, and declarations.",
        "source_artifact_hashes": "Exact bytes preserve source flags and claim scope.",
        "rows": "Per-unit rows retain service and board dispositions without imputation.",
        "sample_size_budget": "Planned, attempted, completed, and unstarted units stay separate.",
        "acceptance_gate_results": "Gate operands keep validity separate from benefit.",
        "gate_check_summary": "Every block names its exact path, field, and observed value.",
        "honest_verdict": "A complete prefix closes external absence without claiming success.",
        "verdict_class": "The closed class separates blocked evidence from null benefit.",
        "verifier_is_oracle": "False prevents fixture timing from becoming formal proof.",
        "flagged_adversarial": "A real verifier finding cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits, and logs bind the checked scope.",
        "field_principles": "Reasons keep each emitted field auditable.",
        "service_cost_complete_score": "A bare 0 or 1 requires exact-operation cost and restart evidence.",
        "board_continuity_complete_score": "A bare 0 or 1 requires three dated dispositions without a current probe.",
        "operation_identity": "Bin count, arithmetic, persistence, and acknowledgement bind the operation.",
        "board_rows": "KV260 fabric, PolarFire CPU, and GateMate physical scope remain separate.",
        "whole_service_speedup": "Null prevents mismatched historical kernels from becoming service speedup.",
    }
    return {
        field: specific.get(
            field, "This field preserves one auditable part of the terminal evidence record."
        )
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable code, settings, roles, raw rows, and historical identities."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "run_date": value.get("run_date"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "preconditions_checked": value.get("preconditions_checked"),
            "operation_identity": value.get("operation_identity"),
            "service_rows": value.get("service_rows"),
            "restart_parity": value.get("restart_parity"),
            "board_rows": value.get("board_rows"),
            "historical_service_context": value.get("historical_service_context"),
            "MODEL_SPECS": value.get("MODEL_SPECS"),
            "model_specs": value.get("model_specs"),
            "model_invoked": value.get("model_invoked"),
            "invocation_counts": value.get("invocation_counts"),
            "validation_manifest": value.get("validation_manifest"),
            "validation_receipts": value.get("validation_receipts"),
        }
    )


def finalize_artifact(value: JsonDict) -> JsonDict:
    """Derive terminal scores, gates, verdict, principles, and checksum."""

    for field in (
        "independent_reduction",
        "service_cost_complete_score",
        "board_continuity_complete_score",
        "acceptance_gate_results",
        "gate_check_summary",
        "honest_verdict",
        "verdict_class",
        "status",
        "field_principles",
        "reproducibility_checksum",
    ):
        value.pop(field, None)
    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["service_cost_complete_score"] = reduction["service_cost_complete_score"]
    value["board_continuity_complete_score"] = reduction["board_continuity_complete_score"]
    gates = _acceptance_gates(value, reduction)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates)
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    if not validity:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_required_validation_failed"
    elif value.get("service_blocker"):
        verdict_class = "blocked"
        honest_verdict = "complete_blocked_missing_count_memory_prototype"
    elif reduction["service_cost_complete_score"] != 1:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_service_measurement_incomplete"
    else:
        verdict_class = "null"
        honest_verdict = "complete_null_count_memory_service_measured_100x_target_unmet"
    value["verdict_class"] = verdict_class
    value["honest_verdict"] = honest_verdict
    value["status"] = honest_verdict
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def _source_revision(root: Path) -> str:  # pragma: no cover - real repository identity.
    """Read the current Git revision without changing repository state."""

    head = (root / ".git/HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    reference = root / ".git" / head.removeprefix("ref: ")
    if reference.is_file():
        return reference.read_text(encoding="utf-8").strip()
    for line in (root / ".git/packed-refs").read_text(encoding="utf-8").splitlines():
        if line.endswith(" " + head.removeprefix("ref: ")):
            return line.split(" ", 1)[0]
    return "unresolved"


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    service_blocker: Mapping[str, Any] | None,
    service_rows: Sequence[Mapping[str, Any]],
    restart_parity: Mapping[str, Any],
    board_rows: Sequence[Mapping[str, Any]],
    changed_state: Mapping[str, Any],
    historical_context: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    elapsed_s: float,
    process_id: int,
    source_revision: str,
) -> JsonDict:
    """Assemble one production-shaped terminal candidate from raw evidence."""

    service_complete = service_blocker is None and len(service_rows) == 60
    attempted_256 = sum(row.get("arm") == "batch_256" for row in service_rows)
    attempted_one = sum(row.get("arm") == "batch_one_ack" for row in service_rows)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "started_monotonic_ns": 0,
        "ended_monotonic_ns": int(elapsed_s * 1_000_000_000),
        "process_identity": {
            "pid": process_id,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "source_revision": source_revision,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": elapsed_s,
        "duration_breakdown_s": {
            "authoring": None,
            "authoring_scope": "outside_current_entrypoint_not_measured",
            "validation": sum(
                float(row.get("duration_s", 0.0) or 0.0) for row in validation_receipts
            ),
            "historical_capture": sum(
                float(row.get("duration_s", 0.0) or 0.0)
                for row in phase_spans
                if row.get("phase") in {"preconditions", "board_audit"}
            ),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "selection": 658028,
            "fitting": 658023,
            "arrival": 658023,
            "bootstrap": 658029,
        },
        "source_artifact_hashes": deepcopy(dict(sources)),
        "service_branch_status": (
            "complete_measured" if service_complete else "blocked_external_prerequisite"
        ),
        "service_blocker": deepcopy(dict(service_blocker)) if service_blocker else None,
        "operation_identity": {
            "operation": "exp7523_count_memory_update",
            "bin_count": 8,
            "state": "eight_label_sums_and_eight_counts_with_frozen_priors",
            "arithmetic": "one_fixed_bin_lookup_then_integer_sum_and_count_increment",
            "predictor": "matched_frozen_exp7523_posterior_mean_composition",
            "queue_semantics": "256_fixture_events_per_independent_batch",
            "persistence": "serialize_flush_fsync_atomic_replace",
            "acknowledgement": "returned_only_after_durable_replace",
            "empirical_learning_claimed": False,
            "device_performance_claimed": False,
        },
        "service_rows": [deepcopy(dict(row)) for row in service_rows],
        "restart_parity": deepcopy(dict(restart_parity)),
        "historical_service_context": deepcopy(dict(historical_context)),
        "whole_service_speedup": None,
        "target_100x_met": False,
        "board_rows": [deepcopy(dict(row)) for row in board_rows],
        "gatemate_changed_state": deepcopy(dict(changed_state)),
        "hardware_operations_issued": [],
        "hardware_disposition": {
            "counter_state_size": "constant_eight_bins",
            "cpu_update_feasible": True,
            "eventual_fpga_mapping": "small_ram_or_lut_candidate_not_measured",
            "larger_board_acquisition_justified": False,
            "larger_board_reason": (
                "No exact count-memory service fraction is available. The only negligible "
                "historical fraction is an operation-mismatched 33-parameter update."
            ),
            "extropic_disposition": "future_authenticated_hardware_codesign_question",
            "vendor_contacted": False,
            "hardware_ordered": False,
        },
        "sample_size_budget": {
            "batch_256": {
                "planned": 30,
                "attempted": attempted_256,
                "completed": attempted_256,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 30 - attempted_256,
            },
            "batch_one_ack": {
                "planned": 30,
                "attempted": attempted_one,
                "completed": attempted_one,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 30 - attempted_one,
            },
            "boards": {
                "planned": 3,
                "attempted": len(board_rows),
                "completed": len(board_rows),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": max(0, 3 - len(board_rows)),
            },
        },
        "rows": [
            *[deepcopy(dict(row)) for row in service_rows],
            *[deepcopy(dict(row)) for row in board_rows],
        ],
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay_required": True,
            "numbered_runtime_e2e": "not_applicable_reporting_only",
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
    }
    return finalize_artifact(value)


def _passing_receipts() -> list[JsonDict]:
    """Create named passing receipts for deterministic schema fixtures."""

    return [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "log_sha256": "sha256:" + "0" * 64,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_fixture_artifact() -> JsonDict:
    """Build a deterministic blocked artifact for mutation and replay tests."""

    placement = load_object(REPO_ROOT / PLACEMENT_PATH)
    trace = load_object(REPO_ROOT / SERVICE_TRACE_PATH)
    changed_state = {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260823",
        "hardware_operations_issued": [],
    }
    board_rows, _summary = audit_board_rows(placement, changed_state)
    blocker = {
        "check": "count_memory_module_available",
        "upstream": "Exp7523",
        "path": COUNT_MODULE_PATH.as_posix(),
        "field": "presence",
        "expected": True,
        "observed": False,
    }
    preconditions = [
        _precondition(
            blocker["check"],
            blocker["upstream"],
            blocker["path"],
            blocker["field"],
            blocker["expected"],
            blocker["observed"],
            branch="service",
        ),
        _precondition(
            "board_fixture",
            "Exp7513+Exp7314",
            "board_rows",
            "board_count",
            3,
            len(board_rows),
            branch="board",
        ),
    ]
    return build_artifact(
        preconditions=preconditions,
        sources={},
        service_blocker=blocker,
        service_rows=[],
        restart_parity={
            "attempted": False,
            "exact_prediction_parity": None,
            "reason": "blocked_missing_count_memory_prototype",
        },
        board_rows=board_rows,
        changed_state=changed_state,
        historical_context=historical_service_context(placement, trace),
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        started_at_utc="2026-09-22T00:00:00+00:00",
        completed_at_utc="2026-09-22T00:00:01+00:00",
        elapsed_s=1.0,
        process_id=0,
        source_revision="fixture",
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, raw reduction, sources, gates, and checksums."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": ZERO_INVOCATION_COUNTS,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": EXECUTION_VENUE,
        "verifier_is_oracle": False,
        "hardware_operations_issued": [],
    }
    inference_fields = {
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "inference_substrate",
        "execution_venue",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(
                "current_inference_declaration_invalid"
                if field in inference_fields
                else f"field_invalid:{field}"
            )
    if value.get("whole_service_speedup") is not None:
        errors.append("whole_service_speedup_must_be_null")
    if value.get("target_100x_met") is not False:
        errors.append("target_100x_must_remain_unmet")
    try:
        reduction = independent_reduce(value)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        reduction = {}
    if reduction != value.get("independent_reduction"):
        errors.append("independent_reduction_mismatch")
    if reduction:
        gates = _acceptance_gates(value, reduction)
        if gates != value.get("acceptance_gate_results"):
            errors.append("acceptance_gate_results_mismatch")
        if _gate_summary(gates) != value.get("gate_check_summary"):
            errors.append("gate_check_summary_mismatch")
        if value.get("service_cost_complete_score") != reduction["service_cost_complete_score"]:
            errors.append("service_cost_complete_score_mismatch")
        if (
            value.get("board_continuity_complete_score")
            != reduction["board_continuity_complete_score"]
        ):
            errors.append("board_continuity_complete_score_mismatch")
    blocker = value.get("service_blocker")
    if blocker:
        required = {"check", "upstream", "path", "field", "expected", "observed"}
        if not required <= set(blocker):
            errors.append("service_blocker_incomplete")
        if value.get("honest_verdict") != "complete_blocked_missing_count_memory_prototype":
            errors.append("blocked_verdict_invalid")
        if value.get("verdict_class") != "blocked":
            errors.append("blocked_class_invalid")
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        errors.append("field_principles_incomplete")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or any(not row.get("principle") for row in gates):
        errors.append("gate_principle_missing")
    if verify_sources:
        for label, row in dict(value.get("source_artifact_hashes") or {}).items():
            path = Path(str(row.get("path") or label))
            resolved = path if path.is_absolute() else root / path
            if not resolved.is_file() or sha256_file(resolved) != row.get("sha256"):
                errors.append(f"source_hash_invalid:{label}")
    if value.get("reproducibility_checksum") != reproducibility_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_commands(private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze exact affected files with private pytest and coverage state."""

    private_root.mkdir(parents=True, exist_ok=True)
    return build_command_plan(REPO_ROOT, VALIDATION_MANIFEST, private_root)


def terminal_commands(candidate: Path) -> list[PlannedCommand]:
    """Build exact-candidate replay, reduction, and strict safety readers."""

    common = ("--date", RUN_DATE, "--root", ".")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--cold-replay",
                str(candidate),
            ),
            "capability_end_to_end",
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                *common,
                "--independent-reduce",
                str(candidate),
            ),
            "service_and_board_rows",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "terminal_candidate",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def utc_now() -> str:  # pragma: no cover - real execution boundary.
    """Return one aware UTC boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase and slow-operation boundaries with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7528] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed: int,
    checkpoint: str,
) -> JsonDict:  # pragma: no cover - real monotonic boundary.
    """Close one phase with its completed-unit checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed,
        "checkpoint": checkpoint,
    }


def _add_current_sources(repo: Path, sources: JsonDict) -> None:  # pragma: no cover
    """Bind implementation files only after their complete bytes exist."""

    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = source_row(path, repo)


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised by the capability entrypoint.
    """Authenticate, audit, validate, cold replay, and publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    repo = root.resolve()
    started = time.monotonic()
    started_utc = utc_now()
    spans: list[JsonDict] = []
    raw_dir = repo / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    progress(started, "preconditions", "start")
    phase_started = time.monotonic()
    context = collect_preconditions(repo)
    atomic_json(
        raw_dir / "preconditions_checkpoint.json",
        {
            "rows": context["rows"],
            "service_blocker": context["service_blocker"],
            "required_ready": context["required_ready"],
            "service_ready": context["service_ready"],
            "board_ready": context["board_ready"],
        },
    )
    spans.append(
        _span(
            "preconditions",
            phase_started,
            started,
            len(context["rows"]),
            "preconditions_checkpoint.json",
        )
    )
    progress(
        started,
        "preconditions",
        "complete",
        required=context["required_ready"],
        service=context["service_ready"],
        board=context["board_ready"],
    )

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0, f"no_{phase}"))
        progress(started, phase, "after", completed=0)

    progress(started, "board_audit", "before_receipt_scan", planned=3)
    phase_started = time.monotonic()
    changed_state: JsonDict = {}
    if context["board_ready"]:
        changed_state = scan_gatemate(repo, raw_dir / "gatemate_changed_state_evidence.json")
    board_rows, board_summary = audit_board_rows(context["placement"], changed_state or None)
    spans.append(_span("board_audit", phase_started, started, len(board_rows), "three_board_rows"))
    progress(
        started,
        "board_audit",
        "after_receipt_scan",
        completed=len(board_rows),
        score=board_summary["board_continuity_complete_score"],
        hardware_operations=0,
    )

    progress(started, "service_benchmark", "before", planned=60)
    phase_started = time.monotonic()
    service_rows: list[JsonDict] = []
    restart_parity: JsonDict = {
        "attempted": False,
        "exact_prediction_parity": None,
        "reason": "blocked_missing_count_memory_prototype",
    }
    if context["service_ready"]:
        raise RuntimeError("valid Exp7523 requires its exact service adapter")
    spans.append(
        _span(
            "service_benchmark",
            phase_started,
            started,
            0,
            "blocked_external_prerequisite",
        )
    )
    progress(started, "service_benchmark", "after", completed=0, blocked=True)

    sources = deepcopy(dict(context["source_artifact_hashes"]))
    _add_current_sources(repo, sources)
    historical = historical_service_context(context["placement"], context["service_trace"])

    private = Path(tempfile.mkdtemp(prefix="exp7528-validation-", dir="/tmp"))
    commands = build_validation_commands(private)
    plan_errors = validate_command_plan(repo, VALIDATION_MANIFEST, commands)
    progress(
        started,
        "affected_validation",
        "before_subprocesses",
        planned=len(commands),
        plan_errors=len(plan_errors),
    )
    phase_started = time.monotonic()
    affected = (
        []
        if plan_errors
        else run_categorized_commands(
            repo,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    )
    affected_reduction = reduce_affected_receipts(repo, VALIDATION_MANIFEST, affected)
    spans.append(
        _span(
            "affected_validation",
            phase_started,
            started,
            len(affected),
            "scoped_commands",
        )
    )
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed=len(affected),
        passed=affected_reduction["passed"],
    )
    if plan_errors or not affected_reduction["passed"]:
        raise RuntimeError(f"affected_validation_failed:{plan_errors}:{affected_reduction}")

    provisional = [
        {
            "name": name,
            "required": True,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
            "provisional_for_candidate_reader": True,
        }
        for name in TERMINAL_CHECK_NAMES
    ]
    candidate = build_artifact(
        preconditions=context["rows"],
        sources=sources,
        service_blocker=context["service_blocker"],
        service_rows=service_rows,
        restart_parity=restart_parity,
        board_rows=board_rows,
        changed_state=changed_state,
        historical_context=historical,
        validation_receipts=[*affected, *provisional],
        phase_spans=spans,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        elapsed_s=time.monotonic() - started,
        process_id=os.getpid(),
        source_revision=_source_revision(repo),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_serialization")

    planned_terminal = terminal_commands(candidate_path)
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(planned_terminal),
    )
    phase_started = time.monotonic()
    terminal = run_categorized_commands(
        repo, planned_terminal, log_dir=raw_dir / "validation/terminal"
    )
    spans.append(
        _span(
            "terminal_validation",
            phase_started,
            started,
            len(terminal),
            "cold_replay_reduction_and_strict_readers",
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed=len(terminal),
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        preconditions=context["rows"],
        sources=sources,
        service_blocker=context["service_blocker"],
        service_rows=service_rows,
        restart_parity=restart_parity,
        board_rows=board_rows,
        changed_state=changed_state,
        historical_context=historical,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
        elapsed_s=time.monotonic() - started,
        process_id=os.getpid(),
        source_revision=_source_revision(repo),
    )
    errors = validate_artifact(final, root=repo)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "write", "before_atomic_terminal", path=output_path)
    atomic_json(candidate_path, final)
    atomic_json(repo / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["honest_verdict"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and fresh-process reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment or one strict fresh-process reader."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_replay is not None:
        value = load_object(args.cold_replay)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = load_object(args.independent_reduce)
        errors = (
            validate_artifact(value, root=root, verify_sources=not args.no_source_check)
            if value
            else ["artifact_unreadable_or_not_object"]
        )
        reduction = independent_reduce(value) if value and not errors else {}
        print(json.dumps({"errors": errors, "reduction": reduction}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(root, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
