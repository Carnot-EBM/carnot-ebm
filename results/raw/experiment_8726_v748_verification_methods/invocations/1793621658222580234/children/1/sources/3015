"""Measure a controlled native-readout request through durable acknowledgement.

The experiment replays training labels supplied by an existing qualified
fixture. It does not measure human feedback acquisition or predictive benefit.

Spec refs: REQ-CL-7514 and SCENARIO-CL-7514-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import tempfile
import time
from typing import Any, Callable

import numpy as np

from carnot import experiment_7463_v654_semif_e0_logprob_parity as parity
from carnot import experiment_7492_v656_window_pilot as native_helpers
from carnot import experiment_7493_v656_window_fit_capture as capture
from carnot import experiment_7504_v657_evidence_interface as evidence
from carnot import experiment_7505_v657_energy_fit as energy_fit
from carnot import experiment_7506_v657_causal_prototype as causal
from carnot import experiment_7513_v657_placement_continuity as placement
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.sota_models import cached_current_model, current_model
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7514-v657-service-trace"
SCHEMA = "carnot.exp7514.v657.service_trace.v1"
MODEL_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_HF_ID]
model_specs = [MODEL_HF_ID]
INFERENCE_SUBSTRATE = "live_llm_embedding_extraction"
INFERENCE_SUBSTRATE_CLASS = "model_load_no_generation"
EXECUTION_VENUE = "host"
READOUT_KIND = "option_logits"
GROUP_COUNT = 24
TRAINING_DENOMINATOR = 176
FORWARD_BUDGET = 192
CAPTURE_CEILING_S = 900.0
MODEL_LOAD_TIMEOUT_S = 900.0
OPTION_ORDERS = evidence.OPTION_ORDERS

RESULT_PATH = Path("results/experiment_7514_v657_service_trace.json")
RAW_DIR = Path("results/raw/experiment_7514_v657_service_trace")
MODULE_PATH = Path("python/carnot/experiment_7514_v657_service_trace.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7514_v657_service_trace.py")
TEST_PATH = Path("tests/python/test_experiment_7514_v657_service_trace.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PROTOCOL_RAW = Path("results/raw/experiment_7491_v656_window_protocol")
FEATURE_PATH = Path("results/raw/experiment_7504_v657_evidence_interface/features.jsonl")
EVALUATOR_PATH = PROTOCOL_RAW / "evaluators.jsonl"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))

UPSTREAM_PATHS = {
    "Exp7504": Path("results/experiment_7504_v657_evidence_interface.json"),
    "Exp7505": Path("results/experiment_7505_v657_energy_fit.json"),
    "Exp7506": Path("results/experiment_7506_v657_causal_prototype.json"),
    "Exp7513": Path("results/experiment_7513_v657_placement_continuity.json"),
}
REQUIRED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7492_v656_window_pilot.py"),
    Path("python/carnot/experiment_7493_v656_window_fit_capture.py"),
    Path("python/carnot/experiment_7473_v654_board_continuity.py"),
    Path("openspec/capabilities/hardware/spec.md"),
    SPEC_PATH,
    FEATURE_PATH,
    EVALUATOR_PATH,
    *(UPSTREAM_PATHS.values()),
)
NUMERIC_STAGES = (
    "feature_conversion",
    "prediction",
    "supplied_label_receipt",
    "numeric_update",
    "guard",
    "serialization",
    "fsync",
    "durable_acknowledgement",
)
SHARED_STAGES = (
    "queued",
    "text_preparation",
    "tokenization",
    "native_prefill",
    "option_extraction",
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


class ServiceTraceError(ValueError):
    """Reject a trace whose source, timing, or durability meaning changed."""


def utc_now() -> str:
    """Return an aware UTC boundary for a measured operation."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush progress so model work and child checks remain observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7514] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def build_frozen_schedule(
    feature_rows: Sequence[Mapping[str, Any]], plan_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 24 complete training groups without changing window qualifiers."""

    training = [row for row in feature_rows if row.get("role") == "training"]
    identities = [str(row.get("group_id") or "") for row in training]
    if len(training) != TRAINING_DENOMINATOR or len(set(identities)) != len(training):
        raise ServiceTraceError("training_group_denominator_invalid")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in plan_rows:
        if row.get("eligible") is True and row.get("role") == "training":
            by_group[str(row.get("group_id") or "")].append(row)
    selected: list[JsonDict] = []
    selected_groups = 0
    for group_id in identities:
        group = by_group.get(group_id, [])
        if not group:
            continue
        if any(
            (row.get("arm") == "whole_response" and row.get("window_index") is not None)
            or (row.get("arm") == "focused_window" and not isinstance(row.get("window_index"), int))
            for row in group
        ):
            raise ServiceTraceError("qualified_group_window_count_invalid")
        windows = sorted(
            {
                int(row["window_index"])
                for row in group
                if row.get("arm") == "focused_window" and isinstance(row.get("window_index"), int)
            }
        )
        if windows != list(range(len(windows))):
            raise ServiceTraceError("qualified_group_window_count_invalid")
        if len(windows) > 3:
            continue
        expected_cells = {("whole_response", None), *{("focused_window", item) for item in windows}}
        observed_cells = {(str(row.get("arm")), row.get("window_index")) for row in group}
        if observed_cells != expected_cells:
            raise ServiceTraceError("qualified_group_cell_invalid")
        for cell in expected_cells:
            orders = {
                tuple(str(item) for item in row.get("option_order") or [])
                for row in group
                if (str(row.get("arm")), row.get("window_index")) == cell
            }
            if orders != set(OPTION_ORDERS):
                raise ServiceTraceError("semantic_order_pair_invalid")
        if len(group) != 2 * len(expected_cells):
            raise ServiceTraceError("semantic_order_pair_invalid")
        selected.extend(deepcopy([dict(row) for row in group]))
        selected_groups += 1
        if selected_groups == GROUP_COUNT:
            break
    if selected_groups != GROUP_COUNT:
        raise ServiceTraceError("qualified_training_support_invalid")
    if len(selected) > FORWARD_BUDGET:  # pragma: no cover - cell gates cap this first.
        raise ServiceTraceError("native_forward_budget_exceeded")
    return selected


def make_update_head(
    training_features: Any, *, learning_rate: float, residual_bound: float
) -> causal.BoundedResidualHead:
    """Create the qualified Exp7506 bounded Brier head on training inputs."""

    return causal.BoundedResidualHead.from_training(
        training_features,
        learning_rate=learning_rate,
        residual_bound=residual_bound,
        loss="brier",
        basis_kind="local",
    )


def _stage(
    intervals: list[JsonDict], name: str, cursor_ns: int, operation: Callable[[], Any]
) -> tuple[Any, int]:
    """Measure one exclusive stage and attribute dispatch time to that stage."""

    value = operation()
    ended_ns = time.monotonic_ns()
    intervals.append(
        {
            "stage": name,
            "start_ns": cursor_ns,
            "end_ns": ended_ns,
            "duration_ns": ended_ns - cursor_ns,
        }
    )
    return value, ended_ns


def _trace_numeric_arm(
    *,
    request_id: str,
    group_id: str,
    arm: str,
    features: Sequence[float],
    base_probability: float,
    label: int,
    head: causal.BoundedResidualHead,
    durable_root: Path,
    order_index: int,
) -> JsonDict:
    """Trace one numeric arm until its exact bytes are durably acknowledged."""

    intervals: list[JsonDict] = []
    cursor = time.monotonic_ns()
    vector, cursor = _stage(
        intervals, "feature_conversion", cursor, lambda: np.asarray(features, dtype=np.float64)
    )
    prediction, cursor = _stage(
        intervals,
        "prediction",
        cursor,
        lambda: head.predict(vector, base_probability=float(base_probability)),
    )
    supplied, cursor = _stage(
        intervals,
        "supplied_label_receipt",
        cursor,
        lambda: {
            "label": int(label),
            "origin": "replayed_training_label",
            "human_feedback_acquired": False,
        },
    )
    if arm == "update":
        update, cursor = _stage(
            intervals,
            "numeric_update",
            cursor,
            lambda: head.update(vector, int(label), base_probability=float(base_probability)),
        )
    else:
        update, cursor = _stage(
            intervals,
            "numeric_update",
            cursor,
            lambda: {"status": "no_update", "state_hash_after": head.state_hash},
        )
    guard, cursor = _stage(
        intervals,
        "guard",
        cursor,
        lambda: {
            "finite_probability": math.isfinite(float(prediction["probability"])),
            "finite_state": bool(update.get("state_hash_after")),
            "passed": math.isfinite(float(prediction["probability"]))
            and bool(update.get("state_hash_after")),
        },
    )
    payload = {
        "schema": "carnot.exp7514.v657.durable_request.v1",
        "request_id": request_id,
        "group_id": group_id,
        "arm": arm,
        "prediction": prediction,
        "supplied_label": supplied,
        "update": update,
        "guard": guard,
    }
    encoded, cursor = _stage(
        intervals,
        "serialization",
        cursor,
        lambda: (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    path = durable_root / request_id / f"{arm}.json"

    def durable_write() -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            written = stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return written

    bytes_written, cursor = _stage(intervals, "fsync", cursor, durable_write)
    acknowledged, cursor = _stage(
        intervals,
        "durable_acknowledgement",
        cursor,
        lambda: path.is_file() and path.read_bytes() == encoded,
    )
    total_ns = sum(int(row["duration_ns"]) for row in intervals)
    return {
        "request_id": request_id,
        "group_id": group_id,
        "arm": arm,
        "arm_order_index": order_index,
        "status": "complete" if guard["passed"] and acknowledged else "failed",
        "label_origin": "replayed_training_label",
        "human_feedback_acquired": False,
        "prediction_probability": float(prediction["probability"]),
        "state_hash_after": update["state_hash_after"],
        "stage_order": [row["stage"] for row in intervals],
        "stage_intervals": intervals,
        "total_ns": total_ns,
        "accounted_ns": total_ns,
        "unaccounted_ns": 0,
        "bytes_written": bytes_written,
        "serialized_sha256": canonical_hash(payload),
        "durable_path": str(path),
        "fsync_completed": True,
        "acknowledged": bool(acknowledged),
        "cancelled": False,
    }


def trace_numeric_pair(
    *,
    request_id: str,
    group_id: str,
    features: Sequence[float],
    base_probability: float,
    label: int,
    update_head: causal.BoundedResidualHead,
    durable_root: Path,
    update_first: bool,
) -> list[JsonDict]:
    """Run matched update and no-update arms with alternating order."""

    frozen = causal.BoundedResidualHead.from_payload(update_head.to_payload())
    arms = ("update", "no_update") if update_first else ("no_update", "update")
    rows = []
    for order_index, arm in enumerate(arms):
        rows.append(
            _trace_numeric_arm(
                request_id=request_id,
                group_id=group_id,
                arm=arm,
                features=features,
                base_probability=base_probability,
                label=label,
                head=update_head if arm == "update" else frozen,
                durable_root=durable_root,
                order_index=order_index,
            )
        )
    return rows


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for finite integer timings."""

    if not values:
        return None
    ordered = sorted(int(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def reduce_service_trace(
    request_rows: Sequence[Mapping[str, Any]], shared_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reduce raw per-arm rows without making completion depend on benefit."""

    by_request: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    interval_coverage = True
    durable_parity = True
    for row in request_rows:
        request_id = str(row.get("request_id") or "")
        arm = str(row.get("arm") or "")
        if arm in by_request[request_id]:
            interval_coverage = False
        by_request[request_id][arm] = row
        intervals = row.get("stage_intervals")
        interval_coverage &= (
            row.get("status") == "complete"
            and row.get("stage_order") == list(NUMERIC_STAGES)
            and isinstance(intervals, list)
            and sum(int(item.get("duration_ns", -1)) for item in intervals)
            == row.get("accounted_ns")
            and row.get("accounted_ns") == row.get("total_ns")
            and row.get("unaccounted_ns") == 0
            and row.get("cancelled") is False
        )
        durable_parity &= row.get("fsync_completed") is True and row.get("acknowledged") is True
    shared_by_request = {str(row.get("request_id") or ""): row for row in shared_rows}
    completed = {
        arm: sum(
            1
            for arms in by_request.values()
            if arm in arms and arms[arm].get("status") == "complete"
        )
        for arm in ("no_update", "update")
    }
    paired_ids = sorted(
        request_id
        for request_id, arms in by_request.items()
        if set(arms) == {"no_update", "update"}
        and all(arms[arm].get("status") == "complete" for arm in arms)
        and shared_by_request.get(request_id, {}).get("complete") is True
    )
    overheads = [
        int(by_request[item]["update"]["total_ns"]) - int(by_request[item]["no_update"]["total_ns"])
        for item in paired_ids
    ]
    update_kernel = [
        next(
            int(stage["duration_ns"])
            for stage in by_request[item]["update"]["stage_intervals"]
            if stage["stage"] == "numeric_update"
        )
        for item in paired_ids
    ]
    update_service = [
        int(shared_by_request[item]["shared_readout_ns"])
        + int(by_request[item]["update"]["total_ns"])
        for item in paired_ids
    ]
    kernel_share = (
        statistics.median(update_kernel) / statistics.median(update_service)
        if update_service and statistics.median(update_service) > 0
        else 0.0
    )
    ideal = math.inf if kernel_share >= 1.0 else 1.0 / (1.0 - kernel_share)
    generated = sum(int(row.get("generated_token_count", -1)) for row in shared_rows)
    forward_calls = sum(int(row.get("native_forward_count", 0)) for row in shared_rows)
    support = len(paired_ids) >= 20 and min(completed.values(), default=0) >= 20
    return {
        "completed_requests_per_arm": completed,
        "paired_request_count": len(paired_ids),
        "paired_support_passed": support,
        "interval_coverage_passed": interval_coverage and len(request_rows) == 2 * len(by_request),
        "durable_parity_passed": durable_parity and bool(request_rows),
        "shared_readout_complete": len(shared_rows) == len(by_request)
        and set(shared_by_request) == set(by_request),
        "shared_native_forward_calls": forward_calls,
        "generated_token_count": generated,
        "paired_overhead_ns": {
            "median": _percentile(overheads, 0.5) if support else None,
            "p95": _percentile(overheads, 0.95) if support else None,
            "descriptive_values": overheads,
        },
        "amdahl_bounds": {
            "measured_update_kernel_share": kernel_share,
            "ideal_infinite_speed_update_kernel_ceiling": ideal,
            "classification": "ideal_bound_not_measured_device_speedup",
        },
    }


def _invocation_counts(forwards: int, *, loaded: bool = True) -> JsonDict:
    """Build balanced current counts for one load and zero generations."""

    def row(attempted: int, completed: int) -> JsonDict:
        return {
            "attempted": attempted,
            "completed": completed,
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
        }

    return {
        "model_loads": row(int(loaded), int(loaded)),
        "forward_calls": row(forwards, forwards),
        "generation_calls": row(0, 0),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Seal one raw shard with fsync before returning its byte identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": len(payload),
        "rows": len(rows),
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum itself."""

    return canonical_hash(
        {key: deepcopy(item) for key, item in value.items() if key != "reproducibility_checksum"}
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep each gate operand and its failure mode next to the result."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": passed,
        "principle": principle,
    }


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all failed operands and the first exact routing failure."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {"passed": not failed, "failed_checks": failed, "failure": failed[0] if failed else None}


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one successful receipt for every affected and terminal check."""

    passed = {str(row.get("name")) for row in receipts if row.get("exit_code") == 0}
    return set((*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)).issubset(passed)


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain the evidence failure prevented by every emitted field."""

    specific = {
        "MODEL_SPECS": "The mandated identifier prevents a legacy smoke model from replacing current Qwen.",
        "model_specs": "The duplicate machine field prevents a reader from losing model identity.",
        "invocation_counts": "Balanced counters prevent cached arithmetic from posing as a native call.",
        "inference_substrate": "The live readout label prevents historical aggregation from posing as current inference.",
        "inference_substrate_class": "The no-generation class applies the two-second floor without padding work.",
        "request_rows": "Per-arm rows prevent a summary from hiding failed fsync or unmatched schedules.",
        "native_model_receipt": "Hash, tokenizer, build, PID, CUDA UUID, and offload prevent substituted inference.",
        "service_boundaries": "The boundary prevents replayed labels from becoming a human-latency claim.",
        "amdahl_bounds": "The measured denominator prevents a kernel-only speedup from becoming a service claim.",
        "service_trace_complete_score": "A bare validity bit stays independent of benefit.",
        "durable_service_claim_ready_score": "A separate readiness bit requires support and complete accounting.",
    }
    return {
        field: specific.get(
            field, "This field preserves an auditable operand and prevents silent evidence drift."
        )
        for field in fields
    }


def _finalize(value: JsonDict) -> JsonDict:
    """Add complete field principles and a checksum after all fields exist."""

    value["field_principles"] = {}
    value["reproducibility_checksum"] = ""
    value["field_principles"] = _field_principles(tuple(value))
    value["reproducibility_checksum"] = artifact_checksum(value)
    return value


def _service_gates(
    reduction: Mapping[str, Any], *, identity_valid: bool, validation_passed: bool
) -> list[JsonDict]:
    """Separate evidence validity, support readiness, and unmeasured benefit."""

    return [
        _gate(
            "native_identity_and_offload",
            "validity",
            True,
            identity_valid,
            identity_valid,
            "Current model identity and CUDA ownership prevent a CPU fallback claim.",
        ),
        _gate(
            "interval_coverage",
            "validity",
            True,
            reduction.get("interval_coverage_passed"),
            reduction.get("interval_coverage_passed") is True,
            "Exclusive coverage prevents hidden or double-counted machine time.",
        ),
        _gate(
            "durable_parity",
            "validity",
            True,
            reduction.get("durable_parity_passed"),
            reduction.get("durable_parity_passed") is True,
            "Both arms must cross serialization, fsync, and acknowledgement.",
        ),
        _gate(
            "paired_support",
            "readiness",
            True,
            reduction.get("paired_support_passed"),
            reduction.get("paired_support_passed") is True,
            "At least 20 pairs prevent unsupported median and p95 summaries.",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            validation_passed,
            validation_passed,
            "Required readers can disqualify favorable timing evidence.",
        ),
        _gate(
            "efficacy_not_inferred_from_timing",
            "benefit",
            False,
            False,
            True,
            "A timing trace does not establish predictive benefit.",
        ),
    ]


def build_artifact(
    *,
    root: Path,
    request_rows: Sequence[Mapping[str, Any]],
    shared_rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    native_model_receipt: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    ended_at: str,
    duration_s: float,
    raw_receipts: Mapping[str, Any],
    exp7513_bound: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble a terminal candidate from raw current-work receipts."""

    reduction = reduce_service_trace(request_rows, shared_rows)
    identity_valid = native_model_receipt.get("identity_authenticated") is True
    validation_ok = _validation_passed(validation_receipts)
    complete = int(
        identity_valid
        and reduction["interval_coverage_passed"]
        and reduction["durable_parity_passed"]
        and reduction["shared_readout_complete"]
        and reduction["generated_token_count"] == 0
        and 0 < reduction["shared_native_forward_calls"] <= FORWARD_BUDGET
        and validation_ok
    )
    ready = int(complete == 1 and reduction["paired_support_passed"])
    gates = _service_gates(
        reduction, identity_valid=identity_valid, validation_passed=validation_ok
    )
    budget = {
        "groups": {
            "planned": GROUP_COUNT,
            "attempted": len(shared_rows),
            "completed": sum(row.get("complete") is True for row in shared_rows),
            "excluded": GROUP_COUNT - len(shared_rows),
            "failed": sum(row.get("complete") is not True for row in shared_rows),
            "censored": 0,
            "unstarted": max(0, GROUP_COUNT - len(shared_rows)),
        },
        "arms": {
            "planned": GROUP_COUNT * 2,
            "attempted": len(request_rows),
            "completed": sum(row.get("status") == "complete" for row in request_rows),
            "excluded": 0,
            "failed": sum(row.get("status") != "complete" for row in request_rows),
            "censored": 0,
            "unstarted": max(0, GROUP_COUNT * 2 - len(request_rows)),
        },
        "native_forwards": {
            "planned_maximum": FORWARD_BUDGET,
            "attempted": reduction["shared_native_forward_calls"],
            "completed": reduction["shared_native_forward_calls"],
            "failed": 0,
            "cancelled": 0,
            "in_flight": 0,
            "unstarted": FORWARD_BUDGET - reduction["shared_native_forward_calls"],
        },
    }
    amdahl = deepcopy(reduction["amdahl_bounds"])
    amdahl["exp7513_numeric_substitution_bound"] = deepcopy(exp7513_bound)
    value: JsonDict = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "terminal_status": "complete",
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "process_identity": {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": list(model_specs),
        "model_invoked": True,
        "invocation_counts": _invocation_counts(reduction["shared_native_forward_calls"]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "kind": "live_llm_embedding_extraction",
            "readout_kind": READOUT_KIND,
            "shared_across_numeric_arms": True,
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "duration_breakdown_s": {
            "cold_model_load": float(native_model_receipt.get("cold_load_s", 0.0)),
            "shared_native_readout": sum(
                int(row.get("shared_readout_ns", 0)) for row in shared_rows
            )
            / 1e9,
            "numeric_arms": sum(int(row.get("total_ns", 0)) for row in request_rows) / 1e9,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {"fit": 750601, "arrival": 751401, "audit": 751402, "bootstrap": 751403},
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "rows": [
            {
                "unit_id": row.get("request_id"),
                "group_id": row.get("group_id"),
                "status": "complete" if row.get("complete") else "failed",
                "failed": row.get("complete") is not True,
                "censored": False,
            }
            for row in shared_rows
        ],
        "request_rows": [deepcopy(dict(row)) for row in request_rows],
        "shared_readout_rows": [deepcopy(dict(row)) for row in shared_rows],
        "native_call_rows": [deepcopy(dict(row)) for row in native_rows],
        "raw_receipts": deepcopy(dict(raw_receipts)),
        "sample_size_budget": budget,
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "honest_verdict": "complete_null_controlled_service_trace_ready_no_efficacy_or_sla_claim"
        if ready
        else "complete_null_service_trace_valid_paired_support_not_ready",
        "verdict_class": "null",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "service_trace_complete_score": complete,
        "durable_service_claim_ready_score": ready,
        "native_model_receipt": deepcopy(dict(native_model_receipt)),
        "readout_kind": READOUT_KIND,
        "generated_token_count": 0,
        "service_boundaries": {
            "starts": "owned_request_text_preparation",
            "ends": "durable_acknowledgement_after_fsync",
            "label_input": "explicit_supplied_replayed_training_label",
            "human_feedback_acquisition_time": "unknown_excluded",
            "fpga_tsu_transfer_cost": "unknown_excluded",
            "production_sla_claimed": False,
        },
        "service_reduction": reduction,
        "amdahl_bounds": amdahl,
        "ebm_efficacy_claimed": False,
        "measured_100x_claimed": False,
        "external_publication_performed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "push_performed": False,
    }
    return _finalize(value)


def _fixture_receipts() -> list[JsonDict]:
    """Return named passing receipts for deterministic schema tests only."""

    return [
        {"name": name, "exit_code": 0, "required": True, "scope": "fixture"}
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def build_artifact_for_test(root: Path, *, request_count: int = 20) -> JsonDict:
    """Build production-shaped circular timing evidence without loading Qwen."""

    feature_rows = [
        [float(index + offset) / 100.0 for offset in range(4)]
        for index in range(TRAINING_DENOMINATOR)
    ]
    head = make_update_head(np.asarray(feature_rows), learning_rate=0.03, residual_bound=1.0)
    request_rows: list[JsonDict] = []
    shared_rows: list[JsonDict] = []
    native_rows: list[JsonDict] = []
    for index in range(request_count):
        request_id = f"request-{index:02d}"
        group_id = f"group-{index:03d}"
        request_rows.extend(
            trace_numeric_pair(
                request_id=request_id,
                group_id=group_id,
                features=feature_rows[index],
                base_probability=0.4,
                label=index % 2,
                update_head=head,
                durable_root=root / "durable",
                update_first=index % 2 == 0,
            )
        )
        shared_rows.append(
            {
                "request_id": request_id,
                "group_id": group_id,
                "complete": True,
                "native_forward_count": 8,
                "generated_token_count": 0,
                "stage_order": list(SHARED_STAGES),
                "shared_readout_ns": 1_000_000 + index,
                "queued_ns": 100 + index,
                "unaccounted_ns": 0,
            }
        )
        native_rows.extend(
            {
                "call_id": f"{request_id}-call-{call}",
                "request_id": request_id,
                "group_id": group_id,
                "disposition": "complete",
                "generated_tokens": 0,
            }
            for call in range(8)
        )
    raw_root = root / "raw"
    raw = {
        "request_rows": _write_jsonl(raw_root / "request_rows.jsonl", request_rows),
        "shared_rows": _write_jsonl(raw_root / "shared_rows.jsonl", shared_rows),
        "native_rows": _write_jsonl(raw_root / "native_rows.jsonl", native_rows),
    }
    preconditions = [
        {
            "check": "fixture_inputs",
            "upstream": "test",
            "field": "available",
            "path": str(root),
            "expected": True,
            "observed": True,
            "op": "eq",
            "passed": True,
            "principle": "A complete fixture prevents a vacuous reducer test.",
        }
    ]
    model = {
        "hf_id": MODEL_HF_ID,
        "model_path": "/fixture/Qwen3.8-27B-Q4_K_M.gguf",
        "model_sha256": "sha256:fixture-model",
        "quantization": "Q4_K_M",
        "tokenizer_identity": {"source": "embedded_gguf"},
        "llama_cpp_build": {"module_sha256": "sha256:fixture-runtime"},
        "cuda_uuid": "GPU-fixture",
        "owned_pid": os.getpid(),
        "observed_offload": {"cuda_offload": True},
        "identity_authenticated": True,
        "cold_load_s": 2.0,
    }
    return build_artifact(
        root=root,
        request_rows=request_rows,
        shared_rows=shared_rows,
        native_rows=native_rows,
        native_model_receipt=model,
        preconditions=preconditions,
        source_hashes=[],
        validation_receipts=_fixture_receipts(),
        phase_spans=[
            {"phase": "fixture", "start_s": 0.0, "end_s": 2.1, "completed_units": request_count}
        ],
        started_at="2026-09-22T00:00:00Z",
        ended_at="2026-09-22T00:00:02.100000Z",
        duration_s=2.1,
        raw_receipts=raw,
        exp7513_bound={"qualified": True, "classification": "fixture_bound", "value": 1.0},
    )


def build_blocked_artifact(failed: Mapping[str, Any], *, root: Path = REPO_ROOT) -> JsonDict:
    """Publish an exact external prerequisite failure without dependent rows."""

    failure = {
        key: deepcopy(failed.get(key))
        for key in ("check", "upstream", "field", "path", "expected", "observed")
    }
    name = str(failed.get("check") or "precondition").replace(":", "_").replace("/", "_")
    gate = {
        **failure,
        "category": "validity",
        "op": failed.get("op", "eq"),
        "passed": False,
        "principle": str(
            failed.get("principle")
            or "Exact prerequisite failure prevents invented dependent measurement."
        ),
    }
    value: JsonDict = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "terminal_status": "complete",
        "preconditions_checked": [deepcopy(dict(failed))],
        "MODEL_SPECS": list(MODEL_SPECS),
        "model_specs": list(model_specs),
        "model_invoked": False,
        "invocation_counts": _invocation_counts(0, loaded=False),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {"fit": 750601, "arrival": 751401, "audit": 751402, "bootstrap": 751403},
        "source_artifact_hashes": [],
        "rows": [],
        "request_rows": [],
        "shared_readout_rows": [],
        "native_call_rows": [],
        "raw_receipts": {},
        "sample_size_budget": {
            "groups": {
                "planned": GROUP_COUNT,
                "attempted": 0,
                "completed": 0,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": GROUP_COUNT,
            }
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {"passed": False, "failed_checks": [gate], "failure": failure},
        "honest_verdict": f"complete_blocked_{name}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "service_trace_complete_score": 0,
        "durable_service_claim_ready_score": 0,
        "native_model_receipt": {},
        "readout_kind": READOUT_KIND,
        "generated_token_count": 0,
        "service_boundaries": {"human_feedback_acquisition_time": "unknown_excluded"},
        "service_reduction": {},
        "amdahl_bounds": {},
        "ebm_efficacy_claimed": False,
        "measured_100x_claimed": False,
    }
    del root
    return _finalize(value)


def _raw_receipts_valid(value: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared raw shard without trusting terminal summaries."""

    receipts = value.get("raw_receipts")
    if not isinstance(receipts, Mapping) or set(receipts) != {
        "request_rows",
        "shared_rows",
        "native_rows",
    }:
        return False
    for receipt in receipts.values():
        if not isinstance(receipt, Mapping):
            return False
        path = Path(str(receipt.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or sha256_file(resolved) != receipt.get("sha256"):
            return False
        if resolved.stat().st_size != receipt.get("bytes"):
            return False
    return True


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_validation: bool = True
) -> list[str]:
    """Cold-check identity, reduction, raw bytes, scores, gates, and checksum."""

    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "version": 1,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "terminal_status": "complete",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": model_specs,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "readout_kind": READOUT_KIND,
        "generated_token_count": 0,
        "verifier_is_oracle": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"field_invalid:{field}")
    if value.get("verdict_class") == "blocked":
        if value.get("request_rows") != [] or value.get("service_trace_complete_score") != 0:
            errors.append("blocked_rows_or_score_invalid")
        summary = value.get("gate_check_summary")
        if not isinstance(summary, Mapping) or not isinstance(summary.get("failure"), Mapping):
            errors.append("blocked_gate_summary_invalid")
    else:
        request_rows = value.get("request_rows")
        shared_rows = value.get("shared_readout_rows")
        native_rows = value.get("native_call_rows")
        if not isinstance(request_rows, list) or not isinstance(shared_rows, list):
            errors.append("request_rows_invalid")
            reduction: JsonDict = {}
        else:
            reduction = reduce_service_trace(request_rows, shared_rows)
            if reduction != value.get("service_reduction"):
                errors.append("service_reduction_mismatch")
        model = value.get("native_model_receipt")
        identity = isinstance(model, Mapping) and model.get("identity_authenticated") is True
        validation_ok = _validation_passed(value.get("validation_receipts") or [])
        gates = _service_gates(reduction, identity_valid=identity, validation_passed=validation_ok)
        if gates != value.get("acceptance_gate_results"):
            errors.append("acceptance_gate_results_mismatch")
        if gate_check_summary(gates) != value.get("gate_check_summary"):
            errors.append("gate_check_summary_mismatch")
        complete = int(
            identity
            and reduction.get("interval_coverage_passed") is True
            and reduction.get("durable_parity_passed") is True
            and reduction.get("shared_readout_complete") is True
            and reduction.get("generated_token_count") == 0
            and 0 < int(reduction.get("shared_native_forward_calls", 0)) <= FORWARD_BUDGET
            and validation_ok
        )
        ready = int(complete == 1 and reduction.get("paired_support_passed") is True)
        if value.get("service_trace_complete_score") != complete:
            errors.append("service_trace_complete_score_mismatch")
        if value.get("durable_service_claim_ready_score") != ready:
            errors.append("durable_service_claim_ready_score_mismatch")
        if value.get("invocation_counts") != _invocation_counts(
            int(reduction.get("shared_native_forward_calls", 0))
        ):
            errors.append("invocation_counts_invalid")
        if not isinstance(native_rows, list) or len(native_rows) != int(
            reduction.get("shared_native_forward_calls", -1)
        ):
            errors.append("native_call_rows_invalid")
        if not _raw_receipts_valid(value, root):
            errors.append("raw_receipts_invalid")
        if require_validation and not validation_ok:
            errors.append("required_validation_failed")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(value):
        errors.append("field_principles_incomplete")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> JsonDict:
    """Recompute terminal scores from raw rows and exact current declarations."""

    errors = validate_artifact(value, root=root, require_validation=require_terminal)
    return {
        "passed": not errors,
        "errors": errors,
        "service_trace_complete_score": value.get("service_trace_complete_score"),
        "durable_service_claim_ready_score": value.get("durable_service_claim_ready_score"),
        "service_reduction": value.get("service_reduction"),
    }


def _precondition(
    check: str,
    upstream: str,
    field: str,
    path: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Record the exact prerequisite operand before dependent work starts."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "path": path,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
        "principle": "Exact prerequisite identity prevents guessed dependent evidence.",
    }


def _source(path: Path, root: Path) -> JsonDict:
    """Hash original bytes while preserving prior disposition fields."""

    resolved = path.resolve()
    label = (
        resolved.relative_to(root.resolve()).as_posix()
        if resolved.is_relative_to(root.resolve())
        else str(resolved)
    )
    row: JsonDict = {
        "path": label,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if path.suffix == ".json":
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            row.update(
                {
                    "original_honest_verdict": value.get("honest_verdict"),
                    "original_verdict_class": value.get("verdict_class"),
                    "original_flagged_adversarial": value.get("flagged_adversarial"),
                }
            )
    return row


def _load_jsonl(path: Path) -> list[JsonDict]:
    """Load one qualified JSONL producer without altering its order."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def collect_preconditions(
    root: Path, *, started: float
) -> tuple[list[JsonDict], JsonDict, list[JsonDict]]:  # pragma: no cover - live gate.
    """Authenticate required sources, upstreams, cached Qwen, and an idle GPU."""

    checks: list[JsonDict] = []
    sources: list[JsonDict] = []
    for relative in (*REQUIRED_INPUT_PATHS, MODULE_PATH, WRAPPER_PATH, TEST_PATH):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"resource_exists:{relative.as_posix()}",
                "filesystem",
                "is_file",
                str(path),
                True,
                available,
            )
        )
        if available:
            sources.append(_source(path, root))
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            "OpenSpec",
            "REQ-*",
            SPEC_PATH.as_posix(),
            "REQ-CL-7514",
            "REQ-CL-7514" if "REQ-CL-7514" in spec_text else None,
        )
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "ops/exclusion_manifest.yaml",
            False,
            "7514" in exclusion,
        )
    )
    upstreams = {
        name: json.loads((root / relative).read_text(encoding="utf-8"))
        for name, relative in UPSTREAM_PATHS.items()
        if (root / relative).is_file()
    }
    upstream_expectations = {
        "Exp7504": ("evidence_ready_score", 1),
        "Exp7505": ("energy_fit_ready_score", 1),
        "Exp7506": ("causal_update_ready_score", 1),
        "Exp7513": ("numeric_placement_ready_score", 1),
    }
    validators = {
        "Exp7504": evidence.validate_artifact(
            upstreams.get("Exp7504", {}), root=root, require_validation=True
        ),
        "Exp7505": energy_fit.cold_replay(
            root / UPSTREAM_PATHS["Exp7505"], root=root, verify_sources=False
        ),
        "Exp7506": causal.cold_replay(
            root / UPSTREAM_PATHS["Exp7506"], root=root, verify_sources=False
        ),
        "Exp7513": placement.validate_artifact(upstreams.get("Exp7513", {}), root=root),
    }
    for name, (field, expected) in upstream_expectations.items():
        value = upstreams.get(name, {})
        observed = (
            value.get(field)
            if not validators[name] and value.get("flagged_adversarial") is False
            else None
        )
        checks.append(
            _precondition(
                f"{name.lower()}_valid_ready",
                name,
                field,
                UPSTREAM_PATHS[name].as_posix(),
                expected,
                observed,
            )
        )

    cached = cached_current_model(gpu_index=0)
    model_path = Path(str(cached.get("model_path"))) if cached else Path("/missing-qwen-gguf")
    mandate = current_model()
    inventory = parity._gpu_inventory()
    idle = [
        row
        for row in inventory
        if row["memory_free_mb"] >= 20_000
        and row["memory_used_mb"] <= 1_024
        and row["utilization_pct"] <= 5
    ]
    selected_gpu = deepcopy(idle[0]) if idle else {}
    checks.extend(
        (
            _precondition(
                "force_live",
                "environment",
                "CARNOT_FORCE_LIVE",
                "environment",
                "1",
                os.environ.get("CARNOT_FORCE_LIVE"),
            ),
            _precondition(
                "cached_current_model",
                "cached_current_model",
                "hf_id",
                str(model_path),
                MODEL_HF_ID,
                cached.get("hf_id") if cached else None,
            ),
            _precondition(
                "quantization",
                "current_model",
                "quantization",
                "carnot.inference.sota_models",
                "Q4_K_M",
                mandate.get("quantization"),
            ),
            _precondition(
                "cached_gguf",
                str(model_path),
                "is_file",
                str(model_path),
                True,
                model_path.is_file(),
            ),
            _precondition(
                "idle_cuda_device", "nvidia-smi", "idle_candidate", "nvidia-smi", True, bool(idle)
            ),
        )
    )
    model_sha256 = None
    if model_path.is_file():
        progress(started, "preconditions", "before_model_hash", path=model_path)
        model_sha256 = native_helpers._call_with_heartbeats(
            lambda: sha256_file(model_path),
            started=started,
            phase="preconditions",
            operation_name="model_sha256",
        )
        progress(started, "preconditions", "after_model_hash", sha256=model_sha256)
        expected_hash = (
            json.loads((root / "results/experiment_7493_v656_window_fit_capture.json").read_text())[
                "model_identity"
            ]
        )["model_sha256"]
        checks.append(
            _precondition(
                "cached_weight_hash",
                "Exp7493",
                "model_sha256",
                str(model_path),
                expected_hash,
                model_sha256,
            )
        )
        sources.append(
            {
                "path": str(model_path),
                "sha256": model_sha256,
                "bytes": model_path.stat().st_size,
                "quantization": mandate.get("quantization"),
            }
        )
    return (
        checks,
        {
            "upstreams": upstreams,
            "model_path": model_path,
            "model_sha256": model_sha256,
            "selected_gpu": selected_gpu,
            "gpu_inventory": inventory,
            "quantization": mandate.get("quantization"),
        },
        sources,
    )


def _phase_span(
    phase: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
    """Close one monotonic phase with its completed unit count."""

    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": time.monotonic() - run_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _prepare_live_schedule(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Join qualified training labels to the existing sealed prompt plan."""

    fit = energy_fit.load_fit_rows(root)
    training = [dict(row) for row in fit["training"]]
    raw = root / PROTOCOL_RAW
    predictors = _load_jsonl(raw / "predictors.jsonl")
    groups = _load_jsonl(raw / "groups.jsonl")
    windows = _load_jsonl(raw / "windows.jsonl")
    requests = _load_jsonl(raw / "requests.jsonl")
    plan = capture.build_capture_plan(predictors, groups, windows, requests)
    schedule = build_frozen_schedule(training, plan)
    return schedule, training, predictors, windows


def _capture_service_rows(
    *,
    root: Path,
    runner: capture.WindowCaptureRunner,
    schedule: Sequence[Mapping[str, Any]],
    training: Sequence[Mapping[str, Any]],
    predictors: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    started: float,
    live_started: float,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Capture current option logits once, then compare both numeric arms."""

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in schedule:
        by_group[str(cell["group_id"])].append(cell)
    training_by_group = {str(row["group_id"]): row for row in training}
    native_rows: list[JsonDict] = []
    shared_rows: list[JsonDict] = []
    request_rows: list[JsonDict] = []
    update_artifact = json.loads((root / UPSTREAM_PATHS["Exp7506"]).read_text(encoding="utf-8"))
    selection = update_artifact["fixture_rows"]["hyperparameter_selection"]
    update_head = make_update_head(
        np.asarray([row["features"][:4] for row in training], dtype=np.float64),
        learning_rate=float(selection["selected_learning_rate"]),
        residual_bound=float(selection["selected_residual_bound"]),
    )
    temperature = float(
        json.loads((root / UPSTREAM_PATHS["Exp7505"]).read_text(encoding="utf-8"))[
            "temperature_baseline"
        ]["selected_temperature"]
    )
    checkpoint = root / RAW_DIR / "checkpoint.json"
    durable_root = root / RAW_DIR / "durable_requests"
    previous_complete_ns = time.monotonic_ns()
    for group_index, (group_id, cells) in enumerate(by_group.items(), start=1):
        if time.monotonic() - live_started >= CAPTURE_CEILING_S:
            break
        group_native: list[JsonDict] = []
        queued_ns = max(0, time.monotonic_ns() - previous_complete_ns)
        text_preparation_ns = 0
        tokenize_ns = 0
        prefill_ns = 0
        extraction_ns = 0
        failed = False
        for call_index, cell in enumerate(cells, start=1):
            if time.monotonic() - live_started >= CAPTURE_CEILING_S:
                failed = True
                break
            progress(
                started,
                "native_readout",
                "before_forward",
                group=group_index,
                call=call_index,
                planned=len(cells),
            )
            prepare_started = time.monotonic_ns()
            prepared = deepcopy(dict(cell))
            str(prepared["prompt"]).encode("utf-8")
            text_preparation_ns += time.monotonic_ns() - prepare_started
            try:
                row = native_helpers._call_with_heartbeats(
                    lambda prepared=prepared: native_helpers._score_native_prompt(runner, prepared),
                    started=started,
                    phase="native_readout",
                    operation_name=f"option_forward:{group_id}:{call_index}",
                )
            except BaseException as exc:  # noqa: BLE001 - raw failure remains visible.
                row = {
                    **{key: deepcopy(value) for key, value in prepared.items() if key != "prompt"},
                    "disposition": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                    "generated_tokens": 0,
                }
                failed = True
            else:
                tokenize_ns += int(float(row["tokenize_s"]) * 1e9)
                prefill_ns += int(float(row["prefill_s"]) * 1e9)
                extraction_ns += int(float(row["readout_s"]) * 1e9)
            row["service_request_id"] = f"service-{group_index:02d}"
            group_native.append(row)
            native_rows.append(row)
            progress(
                started,
                "native_readout",
                "after_forward",
                group=group_index,
                call=call_index,
                disposition=row["disposition"],
            )
            if failed:
                break
        if failed or len(group_native) != len(cells):
            atomic_json(
                checkpoint,
                {
                    "completed_groups": len(shared_rows),
                    "native_rows": native_rows,
                    "failed_group": group_id,
                },
            )
            continue
        feature = evidence.build_feature_rows(group_native, predictors, windows)[0]
        source = training_by_group[group_id]
        base_probability = float(
            energy_fit._apply_temperature(
                np.asarray([feature["raw_whole_expectation"]], dtype=np.float64), temperature
            )[0]
        )
        request_id = f"service-{group_index:02d}"
        shared_ns = queued_ns + text_preparation_ns + tokenize_ns + prefill_ns + extraction_ns
        shared_rows.append(
            {
                "request_id": request_id,
                "group_id": group_id,
                "complete": True,
                "native_forward_count": len(group_native),
                "generated_token_count": 0,
                "stage_order": list(SHARED_STAGES),
                "queued_ns": queued_ns,
                "text_preparation_ns": text_preparation_ns,
                "tokenization_ns": tokenize_ns,
                "native_prefill_ns": prefill_ns,
                "option_extraction_ns": extraction_ns,
                "shared_readout_ns": shared_ns,
                "unaccounted_ns": 0,
            }
        )
        request_rows.extend(
            trace_numeric_pair(
                request_id=request_id,
                group_id=group_id,
                features=feature["features"][:4],
                base_probability=base_probability,
                label=int(source["label"]),
                update_head=update_head,
                durable_root=durable_root,
                update_first=group_index % 2 == 1,
            )
        )
        atomic_json(
            checkpoint,
            {
                "completed_groups": len(shared_rows),
                "native_rows": native_rows,
                "shared_rows": shared_rows,
                "request_rows": request_rows,
            },
        )
        previous_complete_ns = time.monotonic_ns()
        progress(
            started,
            "native_readout",
            "request_checkpointed",
            completed=len(shared_rows),
            planned=len(by_group),
        )
    return native_rows, shared_rows, request_rows


def _exp7513_substitution_bound(value: Mapping[str, Any]) -> JsonDict:
    """Explain why the 81-parameter placement cannot replace this 33-parameter update."""

    return {
        "qualified": False,
        "value": None,
        "upstream_numeric_placement_ready_score": value.get("numeric_placement_ready_score"),
        "reason": "operation_shape_mismatch_exp7513_81_parameter_head_vs_exp7506_33_parameter_update",
        "unchanged_measured_stages_included": True,
        "classification": "not_substituted_no_device_speedup_claim",
    }


def _terminal_commands(candidate: Path) -> list[PlannedCommand]:  # pragma: no cover
    """Build exact-candidate fresh readers and safety checks."""

    relative = candidate.relative_to(REPO_ROOT).as_posix()
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--cold-replay",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "independent_raw_reduction",
            (
                ".venv/bin/python",
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--independent-reduce",
                relative,
            ),
            "completion",
            180,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (".venv/bin/python", "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                ".venv/bin/python",
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                relative,
            ),
            "completion",
            180,
        ),
    )
    return [
        PlannedCommand(spec, "safety" if spec.name == "adversarial_verify" else "completion", True)
        for spec in specs
    ]


def run_experiment(
    root: Path, run_date: str, *, output: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - declared capability E2E.
    """Run one owned Qwen load, controlled readout, durable pair, and readers."""

    if run_date != RUN_DATE:
        raise ServiceTraceError(f"run_date_mismatch:{run_date}")
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(run_started, "preconditions", "start")
    phase_started = time.monotonic()
    checks, context, sources = collect_preconditions(root, started=run_started)
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(failed, root=root)
        atomic_json(root / output, blocked)
        progress(run_started, "preconditions", "complete_blocked", check=failed["check"])
        return blocked
    spans.append(
        _phase_span("preconditions", phase_started, run_started, len(checks), "authenticated")
    )
    progress(run_started, "preconditions", "complete", checked=len(checks))

    progress(run_started, "schedule", "start")
    phase_started = time.monotonic()
    schedule, training, predictors, windows = _prepare_live_schedule(root)
    spans.append(_phase_span("schedule", phase_started, run_started, len(schedule), "frozen"))
    progress(run_started, "schedule", "complete", calls=len(schedule), groups=GROUP_COUNT)
    raw_dir = root / RAW_DIR
    manifest_path = raw_dir / "affected_validation_manifest.json"
    atomic_json(
        manifest_path,
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
    )
    sources.append(_source(manifest_path, root))

    gpu = dict(context["selected_gpu"])
    model_path = Path(context["model_path"])
    lease = lease_api.GpuLease.acquire(
        runtime_dir=LEASE_RUNTIME_DIR,
        task_id=EXPERIMENT_ID,
        device_uuid=str(gpu["uuid"]),
        expected_model=str(model_path),
        vram_before_mb=int(gpu["memory_used_mb"]),
        ttl_s=CAPTURE_CEILING_S + MODEL_LOAD_TIMEOUT_S + 300.0,
    )
    owner = lease.owner_receipt()
    runner = capture.WindowCaptureRunner(model_path, int(gpu["index"]))
    native_model_receipt: JsonDict = {}
    native_rows: list[JsonDict] = []
    shared_rows: list[JsonDict] = []
    request_rows: list[JsonDict] = []
    inference_ok = False
    live_started = time.monotonic()
    try:
        lease.transition("admitted")
        lease.transition("loading")
        phase_started = time.monotonic()
        progress(run_started, "model_load", "before_model_load", gpu_uuid=gpu["uuid"])
        load_started = time.monotonic()
        identity = native_helpers._call_with_heartbeats(
            lambda: runner.load(weight_sha256=str(context["model_sha256"])),
            started=run_started,
            phase="model_load",
            operation_name="native_qwen_load",
        )
        cold_load_s = time.monotonic() - load_started
        if cold_load_s > MODEL_LOAD_TIMEOUT_S:
            raise ServiceTraceError("model_load_ceiling_exceeded")
        process_rows = native_helpers._gpu_process_rows(os.getpid())
        selected_rows = [row for row in process_rows if row["gpu_uuid"] == gpu["uuid"]]
        peak_vram = max((int(row["used_memory_mb"]) for row in selected_rows), default=0)
        resident_mb = parity._gpu_memory(int(gpu["index"]))
        historical_identity = context["upstreams"].get("Exp7504", {})
        del historical_identity
        pilot_identity = json.loads(
            (root / "results/experiment_7493_v656_window_fit_capture.json").read_text(
                encoding="utf-8"
            )
        )["model_identity"]
        runtime_match = (
            identity.get("model_sha256") == pilot_identity.get("model_sha256")
            and identity.get("tokenizer_identity") == pilot_identity.get("tokenizer_identity")
            and identity.get("llama_cpp_build") == pilot_identity.get("llama_cpp_build")
        )
        placement_authenticated = bool(selected_rows) and peak_vram > 1_000
        native_model_receipt = {
            **identity,
            "hf_id": MODEL_HF_ID,
            "quantization": context["quantization"],
            "owned_pid": os.getpid(),
            "cuda_uuid": gpu["uuid"],
            "owned_gpu_process_rows": process_rows,
            "peak_owned_vram_mb": peak_vram,
            "observed_offload": {
                "baseline_vram_mb": gpu["memory_used_mb"],
                "resident_vram_mb": resident_mb,
                "cuda_offload": placement_authenticated,
            },
            "qualified_pilot_identity_match": runtime_match,
            "identity_authenticated": runtime_match and placement_authenticated,
            "cold_load_s": cold_load_s,
            "lease_id": owner["lease_id"],
        }
        if native_model_receipt["identity_authenticated"] is not True:
            raise ServiceTraceError("native_model_identity_or_offload_invalid")
        lease.transition("resident", vram_mb=resident_mb)
        lease.transition("inferencing")
        spans.append(_phase_span("model_load", phase_started, run_started, 1, "resident"))
        progress(run_started, "model_load", "after_model_load", owned_vram_mb=peak_vram)

        phase_started = time.monotonic()
        progress(run_started, "native_readout", "before_forward_loop", planned=len(schedule))
        native_rows, shared_rows, request_rows = _capture_service_rows(
            root=root,
            runner=runner,
            schedule=schedule,
            training=training,
            predictors=predictors,
            windows=windows,
            started=run_started,
            live_started=live_started,
        )
        inference_ok = len(shared_rows) >= 20 and all(
            row.get("disposition") == "complete" for row in native_rows
        )
        spans.append(
            _phase_span(
                "native_readout",
                phase_started,
                run_started,
                len(native_rows),
                "requests_checkpointed",
            )
        )
        progress(
            run_started,
            "native_readout",
            "after_forward_loop",
            calls=len(native_rows),
            requests=len(shared_rows),
        )
    finally:
        progress(run_started, "model_unload", "before_model_unload")
        runner.close()
        gc.collect()
        progress(run_started, "model_unload", "after_model_unload")
        phase = lease.document.get("phase")
        if phase in {"resident", "inferencing"}:
            lease.transition("unloading")
            after_mb = parity._gpu_memory(int(gpu["index"]))
            unload_observed = after_mb <= int(gpu["memory_used_mb"]) + 1_024
            lease.transition(
                "validating",
                vram_mb=after_mb,
                exit_code=0 if inference_ok else 1,
                unload_observed=unload_observed,
            )
            lease.transition(
                "terminal_complete" if inference_ok and unload_observed else "terminal_blocked"
            )
        elif phase in {"preflight", "admitted", "loading"}:
            lease.transition("terminal_blocked")
        release = lease.release()
    native_model_receipt["gpu_lease"] = {**owner, "release": release}

    progress(run_started, "raw_receipts", "before_serialization", requests=len(shared_rows))
    phase_started = time.monotonic()
    raw_receipts = {
        "request_rows": _write_jsonl(raw_dir / "request_rows.jsonl", request_rows),
        "shared_rows": _write_jsonl(raw_dir / "shared_rows.jsonl", shared_rows),
        "native_rows": _write_jsonl(raw_dir / "native_rows.jsonl", native_rows),
    }
    if any(int(row["bytes"]) >= 20 * 1024 * 1024 for row in raw_receipts.values()):
        raise ServiceTraceError("raw_sidecar_size_limit")
    spans.append(
        _phase_span("raw_receipts", phase_started, run_started, len(raw_receipts), "sealed")
    )
    progress(run_started, "raw_receipts", "after_serialization", sidecars=len(raw_receipts))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7514-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        raise ServiceTraceError("validation_plan_invalid:" + ",".join(plan_errors))
    phase_started = time.monotonic()
    progress(run_started, "affected_validation", "before_subprocesses", planned=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "validity", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "affected_validation", phase_started, run_started, len(affected), "checks_terminal"
        )
    )
    progress(
        run_started,
        "affected_validation",
        "after_subprocesses",
        passed=affected_reduction["passed"],
    )
    if affected_reduction["passed"] is not True:
        raise ServiceTraceError("affected_validation_failed")

    bound = _exp7513_substitution_bound(context["upstreams"]["Exp7513"])
    candidate = build_artifact(
        root=root,
        request_rows=request_rows,
        shared_rows=shared_rows,
        native_rows=native_rows,
        native_model_receipt=native_model_receipt,
        preconditions=checks,
        source_hashes=sources,
        validation_receipts=affected,
        phase_spans=spans,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        raw_receipts=raw_receipts,
        exp7513_bound=bound,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    if candidate_path.stat().st_size >= 20 * 1024 * 1024:
        raise ServiceTraceError("candidate_size_limit")

    terminal_plan = _terminal_commands(candidate_path)
    phase_started = time.monotonic()
    progress(run_started, "terminal_validation", "before_subprocesses", planned=len(terminal_plan))
    terminal = run_categorized_commands(
        root,
        terminal_plan,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    spans.append(
        _phase_span(
            "terminal_validation", phase_started, run_started, len(terminal), "readers_terminal"
        )
    )
    terminal_passed = all(row.get("passed") is True for row in terminal)
    critical = any("CRITICAL" in str(row.get("output_tail") or "") for row in terminal)
    progress(
        run_started,
        "terminal_validation",
        "after_subprocesses",
        passed=terminal_passed,
        critical=critical,
    )
    if not terminal_passed or critical:
        raise ServiceTraceError("terminal_validation_failed")

    final = build_artifact(
        root=root,
        request_rows=request_rows,
        shared_rows=shared_rows,
        native_rows=native_rows,
        native_model_receipt=native_model_receipt,
        preconditions=checks,
        source_hashes=sources,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        started_at=started_at,
        ended_at=utc_now(),
        duration_s=time.monotonic() - run_started,
        raw_receipts=raw_receipts,
        exp7513_bound=bound,
    )
    errors = validate_artifact(final, root=root, require_validation=True)
    if errors:
        raise ServiceTraceError("final_validation_failed:" + ",".join(errors))
    destination = root / output
    atomic_json(destination, final)
    if destination.stat().st_size >= 20 * 1024 * 1024:
        raise ServiceTraceError("terminal_artifact_size_limit")
    progress(
        run_started,
        "publish",
        "complete",
        service_trace_complete_score=final["service_trace_complete_score"],
        durable_service_claim_ready_score=final["durable_service_claim_ready_score"],
    )
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the thin entrypoint and fresh-reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run live measurement or one read-only exact-candidate reader."""

    arguments = parse_args(argv)
    if arguments.date != RUN_DATE:
        raise ServiceTraceError(f"run_date_mismatch:{arguments.date}")
    if arguments.cold_replay is not None:
        value = json.loads(arguments.cold_replay.read_text(encoding="utf-8"))
        errors = validate_artifact(value, root=REPO_ROOT, require_validation=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if arguments.independent_reduce is not None:
        value = json.loads(arguments.independent_reduce.read_text(encoding="utf-8"))
        reduced = independent_reduce(value, root=REPO_ROOT, require_terminal=False)
        print(json.dumps(reduced, sort_keys=True), flush=True)
        return int(not reduced["passed"])
    artifact = run_experiment(REPO_ROOT, arguments.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "service_trace_complete_score": artifact["service_trace_complete_score"],
                "durable_service_claim_ready_score": artifact["durable_service_claim_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(artifact["verdict_class"] in {"disqualified", "partial"})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
