"""Measure durable count-memory service cost without touching any board.

The experiment uses the shipped count event machine. It measures the host
persistence path and preserves historical board claims as read-only evidence.

Spec refs: REQ-REPORT-7558 and SCENARIO-REPORT-7558-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import statistics
import tempfile
import time
from typing import Any

from carnot import experiment_7534_v659_count_memory as count_memory
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
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7558-v660-service-boundary"
SCHEMA = "carnot.exp7558.v660.service_boundary.v1"
RESULT_PATH = Path("results/experiment_7558_v660_service_boundary.json")
RAW_DIR = Path("results/raw/experiment_7558_v660_service_boundary")
MODULE_PATH = Path("python/carnot/experiment_7558_v660_service_boundary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7558_v660_service_boundary.py")
TEST_PATH = Path("tests/python/test_experiment_7558_v660_service_boundary.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
COUNT_MODULE_PATH = Path("python/carnot/experiment_7534_v659_count_memory.py")
COUNT_ARTIFACT_PATH = Path("results/experiment_7534_v659_count_memory.json")
BOARD_PATHS = {
    "results/experiment_7314_v642_board_continuity.json",
    "results/experiment_7459_v653_board_continuity.json",
    "results/experiment_7513_v657_placement_continuity.json",
    "results/experiment_7528_v658_service_boundary.json",
}
MODEL_SPECS: list[JsonDict] = []
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
EXECUTION_VENUE = "host"
ZERO_INVOCATION_COUNTS = {
    operation: {
        state: 0 for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
    }
    for operation in ("model_loads", "forward_calls", "generation_calls")
}
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_raw_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def load_object(path: Path) -> JsonDict:
    """Return a JSON object, or an empty object when bytes are unavailable."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def source_row(path: Path, root: Path) -> JsonDict:
    """Bind exact bytes and retain the source's literal terminal disposition."""

    resolved = path.resolve()
    try:
        label = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:  # pragma: no cover - only external source paths use this label.
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
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:
    """Keep an exact failed operand so absence cannot become a zero measurement."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": expected,
        "observed": observed,
        "op": "eq",
        "passed": observed == expected,
    }


def collect_preconditions(root: Path) -> JsonDict:
    """Authenticate the prototype, four board ledgers, spec, and local resources."""

    repo = root.resolve()
    rows: list[JsonDict] = []
    sources: dict[str, JsonDict] = {}
    required = [
        COUNT_MODULE_PATH.as_posix(),
        COUNT_ARTIFACT_PATH.as_posix(),
        *sorted(BOARD_PATHS),
        SPEC_PATH.as_posix(),
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
    ]
    for relative in required:
        path = repo / relative
        present = path.is_file() and path.stat().st_size > 0
        rows.append(
            _precondition(
                "source_readable",
                relative,
                relative,
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else None,
            )
        )
        if present:
            sources[relative] = source_row(path, repo)

    prototype = load_object(repo / COUNT_ARTIFACT_PATH)
    rows.extend(
        [
            _precondition(
                "prototype_ready",
                "Exp7534",
                COUNT_ARTIFACT_PATH.as_posix(),
                "count_memory_ready_score",
                1,
                prototype.get("count_memory_ready_score"),
            ),
            _precondition(
                "prototype_not_flagged",
                "Exp7534",
                COUNT_ARTIFACT_PATH.as_posix(),
                "flagged_adversarial",
                False,
                prototype.get("flagged_adversarial"),
            ),
        ]
    )
    spec_text = (
        (repo / SPEC_PATH).read_text(encoding="utf-8") if (repo / SPEC_PATH).is_file() else ""
    )
    rows.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7558",
            "REQ-REPORT-7558" if "REQ-REPORT-7558" in spec_text else None,
        )
    )
    tmp_free = os.statvfs("/tmp").f_bavail * os.statvfs("/tmp").f_frsize
    rows.append(
        _precondition(
            "scratch_capacity",
            "/tmp",
            "/tmp",
            "free_bytes_at_least_16MiB",
            True,
            tmp_free >= 16 * 1024 * 1024,
        )
    )
    failed = next((row for row in rows if not row["passed"]), None)
    board_artifacts = {relative: load_object(repo / relative) for relative in BOARD_PATHS}
    blocker = (
        {key: failed[key] for key in ("check", "upstream", "path", "field", "expected", "observed")}
        if failed
        else None
    )
    return {
        "rows": rows,
        "ready": failed is None,
        "blocker": blocker,
        "prototype": prototype,
        "board_artifacts": board_artifacts,
        "source_artifact_hashes": sources,
        "resource_observations": {
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "tmp_free_bytes": tmp_free,
        },
    }


def frozen_events(count: int = 256) -> list[tuple[str, float, int]]:
    """Create the fixed event order used by both durability policies."""

    return [
        (f"event-{index:03d}", (index % 100 + 0.5) / 101.0, (index * 17 + 3) % 2)
        for index in range(count)
    ]


def _hardware_identity(path: Path) -> JsonDict:
    """Record host and mounted-storage identity without starting a subprocess."""

    stat = os.stat(path)
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:  # pragma: no cover - Linux production hosts expose /proc/cpuinfo.
        pass
    return {
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "machine": platform.machine(),
        "storage_device_identity": f"st_dev:{stat.st_dev}",
        "scratch_root": "/tmp",
    }


def _durable_payload_write(payload: Mapping[str, Any], path: Path) -> JsonDict:
    """Measure serialization, file fsync, rename, and directory durability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    segments: JsonDict = {}
    started = time.perf_counter_ns()
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    segments["serialize_ns"] = time.perf_counter_ns() - started
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    started = time.perf_counter_ns()
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        segments["write_ns"] = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        os.fsync(stream.fileno())
        segments["file_fsync_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    os.replace(temporary, path)
    segments["rename_ns"] = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    segments["directory_fsync_ns"] = time.perf_counter_ns() - started
    return {"segments": segments, "bytes": len(encoded)}


def measure_policy_trial(
    policy: str,
    trial: int,
    events: Sequence[tuple[str, float, int]],
    scratch: Path,
) -> JsonDict:
    """Run one complete policy trial with the same event and durability meaning."""

    if policy not in {"batch_one", "batch_256"}:
        raise ValueError(f"unknown_policy:{policy}")
    scratch.mkdir(parents=True, exist_ok=True)
    state_path = scratch / "state.json"
    machine = count_memory.CountEventMachine.create(count_memory.default_config())
    totals = {
        name: 0
        for name in (
            "predict_ns",
            "release_update_ns",
            "serialize_ns",
            "write_ns",
            "file_fsync_ns",
            "rename_ns",
            "directory_fsync_ns",
            "reload_ns",
        )
    }
    prediction_started: dict[str, int] = {}
    acknowledgment_ns: dict[str, int] = {}
    bytes_written = 0
    trial_started = time.perf_counter_ns()

    if policy == "batch_one":
        for release_index, (event_id, probability, label) in enumerate(events):
            prediction_started[event_id] = time.perf_counter_ns()
            started = time.perf_counter_ns()
            machine.predict(event_id, probability, release_index=release_index)
            totals["predict_ns"] += time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            machine.release(release_index, [(event_id, label)])
            machine.acknowledge(release_index)
            totals["release_update_ns"] += time.perf_counter_ns() - started
            persisted = _durable_payload_write(machine.to_payload(), state_path)
            bytes_written += int(persisted["bytes"])
            for name, value in persisted["segments"].items():
                totals[name] += int(value)
            started = time.perf_counter_ns()
            machine = count_memory.CountEventMachine.load(state_path)
            totals["reload_ns"] += time.perf_counter_ns() - started
            acknowledgment_ns[event_id] = time.perf_counter_ns()
    else:
        feedback: list[tuple[str, int]] = []
        for event_id, probability, label in events:
            prediction_started[event_id] = time.perf_counter_ns()
            started = time.perf_counter_ns()
            machine.predict(event_id, probability, release_index=0)
            totals["predict_ns"] += time.perf_counter_ns() - started
            feedback.append((event_id, label))
        started = time.perf_counter_ns()
        machine.release(0, feedback)
        machine.acknowledge(0)
        totals["release_update_ns"] += time.perf_counter_ns() - started
        persisted = _durable_payload_write(machine.to_payload(), state_path)
        bytes_written += int(persisted["bytes"])
        for name, value in persisted["segments"].items():
            totals[name] += int(value)
        started = time.perf_counter_ns()
        machine = count_memory.CountEventMachine.load(state_path)
        totals["reload_ns"] += time.perf_counter_ns() - started
        acknowledged = time.perf_counter_ns()
        acknowledgment_ns = {event_id: acknowledged for event_id, _p, _label in events}

    full_service_ns = time.perf_counter_ns() - trial_started
    totals["full_service_ns"] = full_service_ns
    expected_ids = {event_id for event_id, _probability, _label in events}
    observed_ids = machine.arms["local"].processed_event_ids
    ack_latencies = [
        acknowledgment_ns[event_id] - prediction_started[event_id] for event_id in expected_ids
    ]
    return {
        "row_type": "service_trial",
        "disposition": "complete",
        "policy": policy,
        "trial": trial,
        "state_temperature": "cold" if trial == 0 else "warm",
        "scratch_storage": "/tmp",
        "event_count": len(events),
        "acknowledged_event_count": len(acknowledgment_ns),
        "bytes_written": bytes_written,
        "latency_segments_ns": totals,
        "per_event_ack_latency_ns": {
            "median": statistics.median(ack_latencies),
            "maximum": max(ack_latencies),
        },
        "ack_latency_includes_batch_delay": True,
        "durability_semantics": "file_fsync_atomic_rename_directory_fsync_then_ack",
        "event_sequence_sha256": canonical_hash(events),
        "state_hash": machine.state_hash(),
        "reconstruction_parity": observed_ids == expected_ids,
        "observed_hardware": _hardware_identity(scratch),
    }


def _crash_child(boundary: str, path_text: str, connection: Any) -> None:  # pragma: no cover
    """Stop at a registered owned-child boundary until the parent terminates us."""

    path = Path(path_text)
    machine = count_memory.CountEventMachine.load(path)
    machine.predict("crash-event", 0.4, release_index=1)
    machine.release(1, [("crash-event", 1)])
    machine.acknowledge(1)
    payload = machine.to_payload()
    temporary = path.with_name(f".{path.name}.crash-{os.getpid()}")
    if boundary == "pre_commit":
        connection.send(boundary)
    else:
        with temporary.open("wb") as stream:
            stream.write((json.dumps(payload, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        if boundary == "post_file_fsync":
            connection.send(boundary)
        else:
            os.replace(temporary, path)
            connection.send(boundary)
    while True:
        time.sleep(1)


def run_crash_panel(root: Path) -> list[JsonDict]:
    """Terminate only three owned children and reconstruct their durable states."""

    output: list[JsonDict] = []
    for boundary in ("pre_commit", "post_file_fsync", "post_rename"):
        directory = root / boundary
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state.json"
        baseline = count_memory.CountEventMachine.create(count_memory.default_config())
        baseline.predict("acknowledged-baseline", 0.3, release_index=0)
        baseline.release(0, [("acknowledged-baseline", 0)])
        baseline.acknowledge(0)
        _durable_payload_write(baseline.to_payload(), state_path)
        process_context = multiprocessing.get_context("spawn")
        parent, child = process_context.Pipe(duplex=False)
        process = process_context.Process(
            target=_crash_child,
            args=(boundary, str(state_path), child),
            name=f"exp7558-{boundary}",
        )
        process.start()
        child.close()
        observed = parent.poll(10.0) and parent.recv() == boundary
        owned_pid = process.pid
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():  # pragma: no cover - terminate closes the bounded child.
            process.kill()
            process.join(timeout=10.0)
        recovered = count_memory.CountEventMachine.load(state_path)
        processed = recovered.arms["local"].processed_event_ids
        baseline_present = "acknowledged-baseline" in processed
        crash_count = int("crash-event" in processed)
        lost = int(not baseline_present)
        duplicates = max(0, len(processed) - len(set(processed)))
        passed = observed and lost == 0 and duplicates == 0 and crash_count in {0, 1}
        output.append(
            {
                "row_type": "crash_recovery",
                "disposition": "complete" if passed else "failed",
                "boundary": boundary,
                "owned_child_pid": owned_pid,
                "owned_child_only": owned_pid is not None,
                "boundary_observed": observed,
                "acknowledged_before_child": 1,
                "acknowledged_after_child": 0,
                "durable_event_count": len(processed),
                "lost_acknowledged_event_count": lost,
                "duplicate_update_count": duplicates,
                "unacknowledged_crash_event_count": crash_count,
                "passed": passed,
            }
        )
    return output


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_service_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> JsonDict:
    """Reduce paired trials and bound only the measured update fraction."""

    by_policy = {
        policy: sorted(
            (row for row in rows if row.get("policy") == policy),
            key=lambda row: int(row["trial"]),
        )
        for policy in ("batch_one", "batch_256")
    }
    one = [float(row["latency_segments_ns"]["full_service_ns"]) for row in by_policy["batch_one"]]
    batch = [float(row["latency_segments_ns"]["full_service_ns"]) for row in by_policy["batch_256"]]
    differences = [left - right for left, right in zip(one, batch, strict=True)]
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choices(differences, k=len(differences))) for _ in range(1000)
    ]
    batch_update = statistics.median(
        float(row["latency_segments_ns"]["release_update_ns"]) for row in by_policy["batch_256"]
    )
    batch_service = statistics.median(batch)
    update_share = min(1.0, batch_update / batch_service) if batch_service else 0.0
    ideal = None if update_share >= 1.0 else 1.0 / (1.0 - update_share)
    return {
        "paired_trial_count": len(differences),
        "batch_one_median_full_service_ns": statistics.median(one),
        "batch_256_median_full_service_ns": batch_service,
        "paired_median_difference_ns": statistics.median(differences),
        "paired_difference_interval_ns": {
            "method": "paired_bootstrap_1000",
            "lower": _percentile(bootstrap, 0.025),
            "upper": _percentile(bootstrap, 0.975),
        },
        "hardware_acceleration_bound": {
            "denominator": "measured_full_service_ns",
            "measured_update_fraction": update_share,
            "assumed_component_speedup": None,
            "ideal_update_only_speedup": ideal,
            "vendor_ratio_imported": False,
        },
    }


def build_board_rows(
    artifacts: Mapping[str, Mapping[str, Any]], changed_state: Mapping[str, Any] | None
) -> tuple[list[JsonDict], JsonDict]:
    """Preserve fabric, CPU-dispatch, and physical-gate claims separately."""

    placement_path = "results/experiment_7513_v657_placement_continuity.json"
    rows, summary = reduce_board_rows(artifacts.get(placement_path, {}), changed_state)
    for row in rows:
        row["historical_sources"] = sorted(BOARD_PATHS)
        row["present_reachability_asserted"] = False
        row["hardware_operations_issued"] = []
        row["hardware_operation_count"] = 0
        if row.get("board") == "GateMate":
            accepted = int(
                ((row.get("changed_state_evidence") or {}).get("accepted_receipt_count", 0)) or 0
            )
            row["gate_check_summary"] = {
                "check": "dated_operator_physical_change_receipt",
                "upstream": "operator-authored GateMate receipt search",
                "path": "results/raw/experiment_7558_v660_service_boundary/gatemate_changed_state_evidence.json",
                "field": "accepted_receipt_count",
                "expected": ">=1",
                "observed": accepted,
                "op": ">=",
                "passed": accepted >= 1,
            }
    return rows, summary


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required = set(validation_scope.REQUIRED_CHECK_NAMES) | set(TERMINAL_CHECK_NAMES)
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
    """Recompute service, crash, board, inference, and validation readiness."""

    service_rows = list(value.get("service_rows") or [])
    policies = {
        policy: [row for row in service_rows if row.get("policy") == policy]
        for policy in ("batch_one", "batch_256")
    }
    required_segments = {
        "predict_ns",
        "release_update_ns",
        "serialize_ns",
        "write_ns",
        "file_fsync_ns",
        "rename_ns",
        "directory_fsync_ns",
        "reload_ns",
        "full_service_ns",
    }
    service_complete = bool(
        all(len(rows) == 30 for rows in policies.values())
        and all(row.get("event_count") == 256 for row in service_rows)
        and all(row.get("acknowledged_event_count") == 256 for row in service_rows)
        and all(row.get("reconstruction_parity") is True for row in service_rows)
        and all(
            required_segments <= set(row.get("latency_segments_ns") or {}) for row in service_rows
        )
        and all(row.get("bytes_written", 0) > 0 for row in service_rows)
        and all(row.get("ack_latency_includes_batch_delay") is True for row in service_rows)
    )
    crash_rows = list(value.get("crash_rows") or [])
    crash_complete = bool(
        [row.get("boundary") for row in crash_rows]
        == ["pre_commit", "post_file_fsync", "post_rename"]
        and all(row.get("owned_child_only") is True for row in crash_rows)
        and all(row.get("lost_acknowledged_event_count") == 0 for row in crash_rows)
        and all(row.get("duplicate_update_count") == 0 for row in crash_rows)
        and all(row.get("passed") is True for row in crash_rows)
    )
    board_rows = list(value.get("board_rows") or [])
    board_complete = bool(
        [row.get("board") for row in board_rows] == ["KV260", "PolarFire", "GateMate"]
        and board_rows[0].get("exact_claim_scope") == "historical_kv260_fpga_fabric_sampling_only"
        and board_rows[1].get("exact_claim_scope")
        == "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
        and board_rows[2].get("current_disposition")
        in {
            "blocked_unchanged_physical_prerequisite",
            "changed_physical_prerequisite_recorded_future_probe_only",
        }
        and all(row.get("hardware_operations_issued") == [] for row in board_rows)
    )
    declarations = bool(
        value.get("MODEL_SPECS") == []
        and value.get("model_specs") == []
        and value.get("model_invoked") is False
        and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and value.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and value.get("inference_substrate") == INFERENCE_SUBSTRATE
        and value.get("execution_venue") == EXECUTION_VENUE
    )
    return {
        "service_cost_complete_score": int(service_complete and crash_complete),
        "board_continuity_complete_score": int(board_complete),
        "crash_recovery_complete_score": int(crash_complete),
        "current_inference_declarations_valid": declarations,
        "required_validation_passed": _validation_passed(value.get("validation_receipts") or []),
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
    blocker = value.get("external_blocker") or {}
    return [
        _gate(
            str(blocker.get("check") or "prototype_ready"),
            "readiness",
            blocker.get("expected", 1),
            blocker.get("observed", 1),
            not blocker,
            "The shipped learner must exist before service timing begins.",
            upstream=str(blocker.get("upstream") or "Exp7534"),
            path=str(blocker.get("path") or COUNT_ARTIFACT_PATH),
            field=str(blocker.get("field") or "count_memory_ready_score"),
        ),
        _gate(
            "durable_service_complete",
            "readiness",
            1,
            reduction["service_cost_complete_score"],
            reduction["service_cost_complete_score"] == 1,
            "Update-only timing cannot stand in for durable acknowledgement and reload.",
            upstream="current_work",
            path="service_rows+crash_rows",
            field="service_cost_complete_score",
        ),
        _gate(
            "board_continuity_complete",
            "readiness",
            1,
            reduction["board_continuity_complete_score"],
            reduction["board_continuity_complete_score"] == 1,
            "Each board keeps its own historical or blocked disposition.",
            upstream="Exp7314+Exp7459+Exp7513+Exp7528",
            path="board_rows",
            field="board_continuity_complete_score",
        ),
        _gate(
            "current_inference_declarations",
            "validity",
            True,
            reduction["current_inference_declarations_valid"],
            reduction["current_inference_declarations_valid"] is True,
            "Zero typed counters prevent historical model work from becoming current work.",
            upstream="current_work",
            path="invocation_counts",
            field="current_inference_declarations_valid",
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            reduction["required_validation_passed"],
            reduction["required_validation_passed"] is True,
            "Invalid evidence cannot support a service or hardware conclusion.",
            upstream="validation_receipts",
            path="validation_receipts",
            field="required_validation_passed",
        ),
        _gate(
            "measured_dominant_hardware_cost",
            "benefit",
            True,
            False,
            False,
            "A new device needs a measured dominant cost, not a vendor ratio.",
            upstream="current_service_measurement",
            path="hardware_acceleration_bound",
            field="measured_dominant_hardware_cost",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [
        {
            key: row.get(key)
            for key in ("check", "upstream", "path", "field", "expected", "observed")
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
    specific = {
        "experiment_id": "The exact task ID prevents a nearby service run from being substituted.",
        "preconditions_checked": "Exact paths and operands prevent fabricated fallback evidence.",
        "MODEL_SPECS": "An empty list proves that the planned task loads no model.",
        "model_specs": "The lowercase mirror prevents a model alias from hiding a load.",
        "model_invoked": "False separates current CPU work from historical Qwen work.",
        "invocation_counts": "Typed zero counters expose any attempted current model call.",
        "inference_substrate_class": "The class selects the no-model evidence boundary.",
        "inference_substrate": "Aggregation names the use of authenticated upstream records.",
        "execution_venue": "The legal host enum stays separate from device identity.",
        "duration_s": "Monotonic current work cannot be replaced by padded historical time.",
        "phase_spans": "Boundaries expose skipped work and long operations.",
        "random_seed": "Frozen bootstrap and event seeds prevent outcome-driven selection.",
        "reproducibility_checksum": "One digest binds code, sources, settings, and raw rows.",
        "rows": "Per-unit evidence prevents missing work from becoming an average.",
        "sample_size_budget": "Planned, failed, censored, and unstarted units remain distinct.",
        "acceptance_gate_results": "Validity, readiness, and benefit cannot replace each other.",
        "gate_check_summary": "Every failed check retains exact routing and operands.",
        "honest_verdict": "A complete prefix closes the work without overstating benefit.",
        "verdict_class": "The closed class separates a valid null from a block or failure.",
        "verifier_is_oracle": "False prevents a deterministic fixture from becoming proof.",
        "flagged_adversarial": "A verifier finding cannot be cleared to open promotion.",
        "validation_receipts": "Commands, exits, hashes, and cold replay bind checked scope.",
        "field_principles": "Each field states the evidence failure it prevents.",
        "service_cost_complete_score": "One requires durable acknowledgement and cold reload.",
        "board_continuity_complete_score": "One requires all three independent board rows.",
        "service_rows": "Segments expose the full denominator and update-only numerator.",
        "board_rows": "Fabric, board CPU, and physical blockers stay separate.",
        "hardware_acceleration_bound": "Amdahl uses measured service fractions only.",
    }
    return {
        field: specific.get(field, "This field preserves an auditable part of the evidence record.")
        for field in fields
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Bind stable identity, source bytes, settings, rows, and validation receipts."""

    return canonical_hash(
        {
            "schema": value.get("schema"),
            "experiment_id": value.get("experiment_id"),
            "milestone": value.get("milestone"),
            "run_date": value.get("run_date"),
            "source_artifact_hashes": value.get("source_artifact_hashes"),
            "preconditions_checked": value.get("preconditions_checked"),
            "random_seed": value.get("random_seed"),
            "service_rows": value.get("service_rows"),
            "crash_rows": value.get("crash_rows"),
            "board_rows": value.get("board_rows"),
            "service_summary": value.get("service_summary"),
            "invocation_counts": value.get("invocation_counts"),
            "validation_manifest": value.get("validation_manifest"),
            "validation_receipts": value.get("validation_receipts"),
        }
    )


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    resource_observations: Mapping[str, Any],
    blocker: Mapping[str, Any] | None,
    service_rows: Sequence[Mapping[str, Any]],
    crash_rows: Sequence[Mapping[str, Any]],
    board_rows: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
    process_id: int,
    started_at_utc: str,
    completed_at_utc: str,
) -> JsonDict:
    """Assemble one terminal candidate and derive every headline from raw rows."""

    summary = (
        summarize_service_rows(service_rows, 7558001)
        if len(service_rows) == 60
        else {
            "paired_trial_count": 0,
            "hardware_acceleration_bound": {
                "denominator": "measured_full_service_ns",
                "measured_update_fraction": None,
                "assumed_component_speedup": None,
                "ideal_update_only_speedup": None,
                "vendor_ratio_imported": False,
            },
        }
    )
    attempted = len(service_rows)
    value: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "process_identity": {
            "pid": process_id,
            "hostname": platform.node(),
            "python": platform.python_version(),
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "resource_observations": deepcopy(dict(resource_observations)),
        "source_artifact_hashes": deepcopy(dict(sources)),
        "external_blocker": deepcopy(dict(blocker)) if blocker else None,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_invocation_counts": {
            "scope": "source_artifacts_only_not_current_work",
            "counted_as_current": False,
        },
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "planned_inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "execution_venue": EXECUTION_VENUE,
        "device_identity": deepcopy(dict(resource_observations)),
        "duration_s": duration_s,
        "duration_breakdown_s": {
            "authoring": None,
            "authoring_scope": "outside_current_entrypoint_not_measured",
            "validation": sum(
                float(row.get("duration_s", 0.0) or 0.0) for row in validation_receipts
            ),
            "historical": 0.0,
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "event_order": 7558000,
            "bootstrap": 7558001,
            "fitting": 7558002,
            "sampling": 7558003,
        },
        "operation_identity": {
            "prototype": "Exp7534 CountEventMachine",
            "event_count_per_trial": 256,
            "trial_count_per_policy": 30,
            "policies": ["batch_one", "batch_256"],
            "durability": "file_fsync_atomic_rename_directory_fsync_then_ack",
            "lifecycle": "predict_release_update_persist_ack_reload",
            "whole_inference_acceleration_claimed": False,
        },
        "service_rows": [deepcopy(dict(row)) for row in service_rows],
        "crash_rows": [deepcopy(dict(row)) for row in crash_rows],
        "board_rows": [deepcopy(dict(row)) for row in board_rows],
        "service_summary": summary,
        "hardware_acceleration_bound": deepcopy(summary["hardware_acceleration_bound"]),
        "current_acceleration_path": "cpu_or_rust_batching_and_persistence",
        "new_hardware_purchase_justified": False,
        "new_hardware_purchase_condition": "measured_dominant_service_cost_required",
        "positive_claim": False,
        "no_headroom": {
            "hardware_speedup_claim": True,
            "reason": "No authenticated device measurement or component speedup exists.",
        },
        "hardware_operations_issued": [],
        "sample_size_budget": {
            "service_trials": {
                "planned": 60,
                "attempted": attempted,
                "completed": attempted,
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 60 - attempted,
            },
            "crash_boundaries": {
                "planned": 3,
                "attempted": len(crash_rows),
                "completed": sum(row.get("passed") is True for row in crash_rows),
                "excluded": 0,
                "failed": sum(row.get("passed") is not True for row in crash_rows),
                "censored": 0,
                "unstarted": 3 - len(crash_rows),
            },
            "boards": {
                "planned": 3,
                "attempted": len(board_rows),
                "completed": len(board_rows),
                "excluded": 0,
                "failed": 0,
                "censored": 0,
                "unstarted": 3 - len(board_rows),
            },
        },
        "rows": [
            *[deepcopy(dict(row)) for row in service_rows],
            *[deepcopy(dict(row)) for row in crash_rows],
            *[deepcopy(dict(row)) for row in board_rows],
        ],
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
        },
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "raw_custody": {
            "service_rows_embedded": True,
            "crash_rows_embedded": True,
            "source_hashes_bound": True,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay_required": True,
            "lifecycle_exercised": "predict_release_update_persist_ack_reload",
            "numbered_runtime_e2e": "not_applicable_no_shared_training_binding_arc_or_telemetry_change",
            "private_llm_off_smoke": "not_applicable_no_arc_runtime_change",
        },
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "external_publication_performed": False,
        "push_performed": False,
    }
    reduction = independent_reduce(value)
    value["independent_reduction"] = reduction
    value["service_cost_complete_score"] = reduction["service_cost_complete_score"]
    value["board_continuity_complete_score"] = reduction["board_continuity_complete_score"]
    gates = _acceptance_gates(value, reduction)
    value["acceptance_gate_results"] = gates
    value["gate_check_summary"] = _gate_summary(gates)
    validity = all(row["passed"] for row in gates if row["category"] == "validity")
    if not validity:  # pragma: no cover - real entrypoint stops before invalid publication.
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_required_validation_failed"
    elif blocker:
        value["verdict_class"] = "blocked"
        value["honest_verdict"] = f"complete_blocked_{blocker['check']}"
        value["gate_check_summary"]["failed_check"] = deepcopy(dict(blocker))
    elif reduction["service_cost_complete_score"] != 1:  # pragma: no cover
        value["verdict_class"] = "disqualified"
        value["honest_verdict"] = "complete_disqualified_durable_service_incomplete"
    else:
        value["verdict_class"] = "null"
        value["honest_verdict"] = (
            "complete_null_durable_service_measured_hardware_benefit_unmeasured"
        )
    value["status"] = value["honest_verdict"]
    value["field_principles"] = _field_principles(
        (*value.keys(), "field_principles", "reproducibility_checksum")
    )
    value["reproducibility_checksum"] = reproducibility_checksum(value)
    return value


def _passing_receipts() -> list[JsonDict]:
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
        for name in (*validation_scope.REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]


def _fixture_service_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    sequence_hash = canonical_hash(frozen_events())
    for trial in range(30):
        for policy, full, update in (
            ("batch_one", 2_000_000 + trial, 200_000 + trial),
            ("batch_256", 1_000_000 + trial, 150_000 + trial),
        ):
            segments = {
                "predict_ns": 100_000,
                "release_update_ns": update,
                "serialize_ns": 100_000,
                "write_ns": 100_000,
                "file_fsync_ns": 100_000,
                "rename_ns": 100_000,
                "directory_fsync_ns": 100_000,
                "reload_ns": 100_000,
                "full_service_ns": full,
            }
            rows.append(
                {
                    "row_type": "service_trial",
                    "disposition": "complete",
                    "policy": policy,
                    "trial": trial,
                    "state_temperature": "cold" if trial == 0 else "warm",
                    "scratch_storage": "/tmp",
                    "event_count": 256,
                    "acknowledged_event_count": 256,
                    "bytes_written": 4096 if policy == "batch_256" else 1_048_576,
                    "latency_segments_ns": segments,
                    "per_event_ack_latency_ns": {"median": full / 256, "maximum": full},
                    "ack_latency_includes_batch_delay": True,
                    "durability_semantics": "file_fsync_atomic_rename_directory_fsync_then_ack",
                    "event_sequence_sha256": sequence_hash,
                    "state_hash": f"fixture-{policy}-{trial}",
                    "reconstruction_parity": True,
                    "observed_hardware": {"hostname": "fixture", "scratch_root": "/tmp"},
                }
            )
    return rows


def _fixture_crash_rows() -> list[JsonDict]:
    return [
        {
            "row_type": "crash_recovery",
            "disposition": "complete",
            "boundary": boundary,
            "owned_child_pid": index + 1,
            "owned_child_only": True,
            "boundary_observed": True,
            "acknowledged_before_child": 1,
            "acknowledged_after_child": 0,
            "durable_event_count": 1 + int(boundary == "post_rename"),
            "lost_acknowledged_event_count": 0,
            "duplicate_update_count": 0,
            "unacknowledged_crash_event_count": int(boundary == "post_rename"),
            "passed": True,
        }
        for index, boundary in enumerate(("pre_commit", "post_file_fsync", "post_rename"))
    ]


def build_fixture_artifact(tmp_path: Path) -> JsonDict:
    """Build complete deterministic evidence for schema and mutation tests."""

    context = collect_preconditions(REPO_ROOT)
    boards, _summary = build_board_rows(context["board_artifacts"], None)
    return build_artifact(
        preconditions=context["rows"],
        sources={},
        resource_observations={"fixture": True, "scratch": str(tmp_path)},
        blocker=None,
        service_rows=_fixture_service_rows(),
        crash_rows=_fixture_crash_rows(),
        board_rows=boards,
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        duration_s=1.0,
        process_id=0,
        started_at_utc="2026-09-23T00:00:00+00:00",
        completed_at_utc="2026-09-23T00:00:01+00:00",
    )


def build_blocked_artifact(blocker: Mapping[str, Any], tmp_path: Path) -> JsonDict:
    """Publish complete external absence without inventing service rows."""

    context = collect_preconditions(REPO_ROOT)
    boards, _summary = build_board_rows(context["board_artifacts"], None)
    precondition = _precondition(
        str(blocker["check"]),
        str(blocker["upstream"]),
        str(blocker["path"]),
        str(blocker["field"]),
        blocker["expected"],
        blocker["observed"],
    )
    return build_artifact(
        preconditions=[precondition],
        sources={},
        resource_observations={"fixture": True, "scratch": str(tmp_path)},
        blocker=blocker,
        service_rows=[],
        crash_rows=[],
        board_rows=boards,
        validation_receipts=_passing_receipts(),
        phase_spans=[],
        duration_s=0.1,
        process_id=0,
        started_at_utc="2026-09-23T00:00:00+00:00",
        completed_at_utc="2026-09-23T00:00:01+00:00",
    )


def validate_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, verify_sources: bool = True
) -> list[str]:
    """Cold-check identity, raw reduction, source hashes, gates, and checksums."""

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
        "positive_claim": False,
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
    bound = value.get("hardware_acceleration_bound") or {}
    if bound.get("assumed_component_speedup") is not None:
        errors.append("imported_speedup_forbidden")
    if bound.get("vendor_ratio_imported") is not False:
        errors.append("vendor_ratio_imported")
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
        expected_summary = _gate_summary(gates)
        if value.get("external_blocker"):
            expected_summary["failed_check"] = deepcopy(dict(value["external_blocker"]))
        if expected_summary != value.get("gate_check_summary"):
            errors.append("gate_check_summary_mismatch")
        if value.get("service_cost_complete_score") != reduction["service_cost_complete_score"]:
            errors.append("service_cost_complete_score_mismatch")
        if (
            value.get("board_continuity_complete_score")
            != reduction["board_continuity_complete_score"]
        ):
            errors.append("board_continuity_complete_score_mismatch")
    blocker = value.get("external_blocker")
    if blocker:
        required = {"check", "upstream", "path", "field", "expected", "observed"}
        if not required <= set(blocker):
            errors.append("external_blocker_incomplete")
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
    """Freeze explicit tests, module, wrapper, private temp, and coverage paths."""

    private_root.mkdir(parents=True, exist_ok=True)
    return build_command_plan(REPO_ROOT, VALIDATION_MANIFEST, private_root)


def terminal_commands(candidate: Path) -> list[PlannedCommand]:
    """Build cold replay, independent reduction, and both strict readers."""

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
            "service_crash_and_board_rows",
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


def utc_now() -> str:  # pragma: no cover - real entrypoint clock.
    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and long-operation boundary with monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7558] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _span(
    phase: str, phase_started: float, run_started: float, completed: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - real entrypoint clock.
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
    for relative in (MODULE_PATH, WRAPPER_PATH, TEST_PATH, SPEC_PATH):
        path = repo / relative
        if path.is_file():
            sources[relative.as_posix()] = source_row(path, repo)


def run_experiment(
    root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through the capability entrypoint.
    """Measure, validate in fresh processes, and publish only terminal evidence."""

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
            "blocker": context["blocker"],
            "resource_observations": context["resource_observations"],
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
    progress(started, "preconditions", "complete", ready=context["ready"])

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned=0)
        phase_started = time.monotonic()
        spans.append(_span(phase, phase_started, started, 0, f"no_{phase}"))
        progress(started, phase, "after", completed=0)

    progress(started, "board_continuity", "before_receipt_search", planned=3)
    phase_started = time.monotonic()
    changed_state = scan_gatemate(repo, raw_dir / "gatemate_changed_state_evidence.json")
    board_rows, board_summary = build_board_rows(context["board_artifacts"], changed_state)
    spans.append(
        _span("board_continuity", phase_started, started, len(board_rows), "read_only_board_rows")
    )
    progress(
        started,
        "board_continuity",
        "after_receipt_search",
        completed=len(board_rows),
        score=board_summary["board_continuity_complete_score"],
        hardware_operations=0,
    )

    service_rows: list[JsonDict] = []
    crash_rows: list[JsonDict] = []
    progress(started, "service_benchmark", "before", planned=60)
    phase_started = time.monotonic()
    if context["ready"]:
        events = frozen_events()
        scratch_root = Path(tempfile.mkdtemp(prefix="exp7558-service-", dir="/tmp"))
        for trial in range(30):
            for policy in ("batch_one", "batch_256"):
                progress(
                    started,
                    "service_benchmark",
                    "before_trial",
                    policy=policy,
                    trial=trial + 1,
                    completed=len(service_rows),
                )
                row = measure_policy_trial(
                    policy, trial, events, scratch_root / policy / f"trial-{trial:02d}"
                )
                service_rows.append(row)
                progress(
                    started,
                    "service_benchmark",
                    "after_trial",
                    policy=policy,
                    trial=trial + 1,
                    completed=len(service_rows),
                )
        crash_rows = run_crash_panel(scratch_root / "crash")
    spans.append(
        _span("service_benchmark", phase_started, started, len(service_rows), "sixty_trials")
    )
    progress(
        started,
        "service_benchmark",
        "after",
        completed=len(service_rows),
        crash_boundaries=len(crash_rows),
    )

    sources = deepcopy(dict(context["source_artifact_hashes"]))
    _add_current_sources(repo, sources)
    private = Path(tempfile.mkdtemp(prefix="exp7558-validation-", dir="/tmp"))
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
        _span("affected_validation", phase_started, started, len(affected), "scoped_commands")
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
        resource_observations=context["resource_observations"],
        blocker=context["blocker"],
        service_rows=service_rows,
        crash_rows=crash_rows,
        board_rows=board_rows,
        validation_receipts=[*affected, *provisional],
        phase_spans=spans,
        duration_s=time.monotonic() - started,
        process_id=os.getpid(),
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    progress(started, "candidate", "before_serialization", path=candidate_path)
    atomic_json(candidate_path, candidate)
    progress(started, "candidate", "after_serialization")

    planned_terminal = terminal_commands(candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", planned=4)
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
        resource_observations=context["resource_observations"],
        blocker=context["blocker"],
        service_rows=service_rows,
        crash_rows=crash_rows,
        board_rows=board_rows,
        validation_receipts=[*affected, *terminal],
        phase_spans=spans,
        duration_s=time.monotonic() - started,
        process_id=os.getpid(),
        started_at_utc=started_utc,
        completed_at_utc=utc_now(),
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
    """Parse the fixed date, producer path, and two read-only reader modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one strict fresh-process reader."""

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
    run_experiment(root, args.date, output_path=args.output)  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
