"""Qualify callback compatibility and owned-process cleanup before another capture.

This module uses the historical response only as immutable input. It runs fake
transport around real short-lived processes and the shipped GPU lease protocol,
so the failure paths are exercised without loading or calling a model.

Spec refs: REQ-REPORT-7448 and SCENARIO-REPORT-7448-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, TypedDict, cast

from carnot import experiment_7442_v652_span_capture as span_capture
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    sidecar_reference,
    validate_current_work_receipt,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7448-v653-capture-lifecycle"
TASK_ID = "experiment_7448_v653_capture_lifecycle"
SCHEMA = "carnot.exp7448.v653.capture_lifecycle.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")
RAW_DIR = Path("results/raw/experiment_7448_v653_capture_lifecycle")
MODULE_PATH = Path("python/carnot/experiment_7448_v653_capture_lifecycle.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7448_v653_capture_lifecycle.py")
TEST_PATH = Path("tests/python/test_experiment_7448_v653_capture_lifecycle.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HISTORICAL_ARTIFACT_PATH = Path("results/experiment_7442_v652_span_capture.json")
HISTORICAL_RAW_PATH = Path(
    "results/raw/experiment_7442_v652_span_capture/owned_runtime/native/call_00.json"
)

MODEL_SPECS: list[str] = []
INFERENCE_SUBSTRATE = "host_fake_transport_owned_process_lifecycle_audit"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
INTERRUPTED_EXIT_CODE = 75

AFFECTED_CHECK_NAMES = REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(
        TEST_PATH.as_posix(),
        "tests/python/test_experiment_7442_v652_span_capture.py",
        "tests/python/test_gpu_lease_phase_journal.py",
    ),
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
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7442_v652_span_capture.py"),
    Path("python/carnot/experiment_7209_v635_span_canary.py"),
    Path("python/carnot/experiment_7422_v651_runtime_ownership.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    SPEC_PATH,
    HISTORICAL_ARTIFACT_PATH,
    HISTORICAL_RAW_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned top-level schema and exact experiment identity, milestone, and terminal status.",
    "run_date": "Use 20260920 with actual UTC and monotonic boundaries plus boot and segment identity.",
    "preconditions_checked": "Name actual paths, resources, identities, and historical flags before dependent work.",
    "MODEL_SPECS": "Use an empty list because this qualification performs no current LLM task.",
    "model_invoked": "Keep attempted current work separate from archived model-shaped events.",
    "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight current calls.",
    "inference_substrate": "Describe fake transport and real owned host processes without importing historical CUDA.",
    "inference_substrate_class": "Declare no_model_load because no current model is loaded or called.",
    "execution_venue": "Use host while retaining historical device facts only in typed sidecars.",
    "duration_s": "Measure real callback and owned-process lifecycle work without padding.",
    "computation_duration_s": "Use zero because this audit performs no numeric fit or inference computation.",
    "phase_spans": "Bind progress, checkpoints, completed units, boot identity, and monotonic segments.",
    "random_seed": "Explain that deterministic lifecycle mutations use no random sampling.",
    "reproducibility_checksum": "Bind code, protocol, historical bytes, lifecycle rows, and validation scope.",
    "source_artifact_hashes": "Preserve exact upstream bytes, original classes, and flags without rehabilitation.",
    "rows": "Retain every callback, failure, ownership mutation, and restart disposition.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted cases.",
    "acceptance_gate_results": "Name each validity gate with its operator and exact operands.",
    "gate_check_summary": "Expose the first exact failed upstream field while retaining all gates.",
    "verifier_is_oracle": "False because this infrastructure audit does not score semantic truth.",
    "honest_verdict": "Use complete_null for qualified infrastructure without claiming extraction benefit.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings because flagged evidence cannot supply readiness.",
    "validation_receipts": "Record exact scoped commands, environments, exits, durations, and log hashes.",
    "field_principles": "Explain field intent separately while gate values stay bare scalars.",
    "promotion_score": "Remain zero because this work authorizes no rollout, weight change, or publication.",
    "capture_lifecycle_ready_score": "One requires callback compatibility and every owned failure-path check.",
    "lifecycle_rows": "Keep one row for every exception, cancellation, ownership mutation, and restart case.",
    "historical_failure_hashes": "Preserve the original parse exception and quarantined source identity.",
}

REQUIRED_CASES = frozenset(
    {
        "callback_shape",
        "callback_exception",
        "development_gate_closure",
        "timeout",
        "partial_response",
        "matching_owner_evidence",
        "missing_owner_identity",
        "owner_pid_reuse",
        "server_pid_reuse",
        "changed_boot_identity",
        "foreign_live_owner",
        "interruption_restart",
    }
)

SCHEDULE_FIELDS = (
    "call_id",
    "request_id",
    "pair_id",
    "unit_id",
    "case_index",
    "group_id",
    "response_id",
    "condition",
    "capture_phase",
    "arm",
    "arm_order",
    "seed",
    "paragraph",
    "paragraph_sha256",
    "clipped",
    "complete_response_coverage_eligible",
    "max_new_tokens",
    "temperature",
    "generation_count",
    "grammar_mask",
    "parser_retry_count",
    "prompt",
    "prompt_sha256",
)


class CallbackRow(TypedDict, total=False):
    """Fields guaranteed at the one boundary shared with the native loop."""

    parse_status: str
    parse_errors: list[str]
    callback_disposition: str
    semantic_success: bool
    development_usable: bool
    raw_request_sha256: str
    raw_response_sha256: str
    raw_reply_sha256: str
    terminal_state: str


class DevelopmentGateClosed(RuntimeError):
    """The fixed canary supplied too few usable extractions to continue."""


def utc_now() -> str:  # pragma: no cover - actual wall-clock evidence.
    """Return one aware UTC boundary for an artifact or subprocess event."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Emit every phase and potentially long operation boundary immediately."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7448] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash a complete artifact without its self-referential checksum field."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty mapping."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def boot_identity() -> str:
    """Read the Linux boot ID that gives monotonic timestamps their epoch."""

    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()


def identity_evidence(
    *,
    task_id: str,
    lease_id: str,
    owner_pid: int,
    owner_pid_start_ticks: int,
    server_pid: int,
    server_pid_start_ticks: int,
    boot_id: str,
    clock_segments: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the complete durable identity required before any callback."""

    return {
        "task_id": task_id,
        "lease_id": lease_id,
        "owner_pid": int(owner_pid),
        "owner_pid_start_ticks": int(owner_pid_start_ticks),
        "server_pid": int(server_pid),
        "server_pid_start_ticks": int(server_pid_start_ticks),
        "boot_id": boot_id,
        "clock_segments": [deepcopy(dict(row)) for row in clock_segments],
    }


def _wait_for_start_ticks(process: subprocess.Popen[str]) -> int:
    """Wait briefly for procfs to expose one just-created child identity."""

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ticks = lease_api.proc_start_ticks(process.pid)
        if ticks is not None:
            return ticks
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise RuntimeError("owned_child_identity_unavailable")


def assert_native_consumer_shape(row: Mapping[str, Any]) -> None:
    """Exercise every generic field the shipped native consumer indexes."""

    if row["terminal_state"] == "response":
        str(row["parse_status"])
        list(row["parse_errors"])


def normalize_callback_row(
    schedule: Mapping[str, Any], transport: Mapping[str, Any]
) -> CallbackRow:
    """Add one typed native-loop shape without changing raw transport bytes.

    Exp7442 owns the extraction parser. This boundary only supplies its generic
    callback contract and keeps correct-empty separate from semantic success.
    """

    row = span_capture.build_capture_row(schedule, transport)
    terminal = str(row.get("terminal_state") or "")
    if terminal == "response" and row.get("finish_reason") in {"length", "max_tokens"}:
        disposition = "partial_response"
    elif terminal != "response" and "timeout" in str(row.get("error") or "").lower():
        disposition = "timeout"
    elif row.get("parse_valid") is True and not list(row.get("claims") or []):
        disposition = "correct_empty"
    elif row.get("parse_valid") is True:
        disposition = "parsed_nonempty"
    else:
        disposition = "malformed"
    row.update(
        {
            "callback_disposition": disposition,
            "semantic_success": bool(
                disposition == "parsed_nonempty" and row.get("completed_valid_output") is True
            ),
            "development_usable": bool(
                disposition == "parsed_nonempty" and span_capture._usable(row)
            ),
        }
    )
    assert_native_consumer_shape(row)
    return cast(CallbackRow, row)


def prepare_historical_callback_fixture(root: Path, private_root: Path) -> JsonDict:
    """Copy the exact persisted response privately and reproduce the old KeyError."""

    source = root / HISTORICAL_RAW_PATH
    artifact = load_object(root / HISTORICAL_ARTIFACT_PATH)
    native = load_object(source)
    development = artifact.get("development_rows") or []
    if not native or not development or not isinstance(development[0], Mapping):
        raise ValueError("historical_callback_fixture_unavailable")
    private_root.mkdir(parents=True, exist_ok=True)
    private_path = private_root / "historical_call_00.json"
    lease_api.write_bytes_atomic(private_path, source.read_bytes())
    schedule = {key: deepcopy(development[0].get(key)) for key in SCHEDULE_FIELDS}
    transport = deepcopy(native)
    normalized = span_capture.build_capture_row(schedule, transport)
    historical_shape = deepcopy(normalized)
    historical_shape.pop("parse_status", None)
    historical_shape.pop("parse_errors", None)
    legacy_error: str | None = None
    try:
        assert_native_consumer_shape(historical_shape)
    except KeyError as exc:
        legacy_error = f"KeyError:{exc}"
    return {
        "private_raw_path": private_path,
        "raw_sha256": sha256_file(private_path),
        "schedule": schedule,
        "transport": transport,
        "legacy_error": legacy_error,
        "raw_request_sha256": span_capture._transport_hash(native.get("raw_request") or {}),
        "raw_response_sha256": span_capture._transport_hash(native.get("raw_response") or {}),
        "raw_reply_sha256": canonical_hash(str(native.get("raw_reply") or "")),
    }


def _stop_owned_child(process: subprocess.Popen[str]) -> list[JsonDict]:
    """Stop only the child object created by this harness and record each signal."""

    signals: list[JsonDict] = []
    if process.poll() is None:
        process.terminate()
        signals.append({"pid": process.pid, "signal": "SIGTERM", "owned": True})
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        signals.append({"pid": process.pid, "signal": "SIGKILL", "owned": True})
        process.wait(timeout=3)
    return signals


def run_owned_lifecycle_case(case: str, private_root: Path) -> JsonDict:
    """Exercise one callback exit around a real child and a shipped lease."""

    allowed = {"callback_exception", "development_gate_closure", "timeout", "partial_response"}
    if case not in allowed:
        raise ValueError(f"unknown_lifecycle_case:{case}")
    private_root.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(  # noqa: S603 - fixed interpreter and constant child body.
        [sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server_ticks = _wait_for_start_ticks(process)
    lease = lease_api.GpuLease.acquire(
        runtime_dir=private_root / "lease",
        task_id=TASK_ID,
        device_uuid=f"FAKE-{case}",
        expected_model="fake-transport/no-model",
        vram_before_mb=0,
        ttl_s=20,
    )
    owner_ticks = lease_api.proc_start_ticks(os.getpid())
    if owner_ticks is None:
        lease.close()
        _stop_owned_child(process)
        raise RuntimeError("owner_identity_unavailable")
    case_started = time.monotonic_ns()
    binding = identity_evidence(
        task_id=TASK_ID,
        lease_id=lease.lease_id,
        owner_pid=os.getpid(),
        owner_pid_start_ticks=owner_ticks,
        server_pid=process.pid,
        server_pid_start_ticks=server_ticks,
        boot_id=boot_identity(),
        clock_segments=[
            {"segment": "owned_server_lifetime", "start_ns": case_started, "end_ns": case_started},
            {"segment": "callback", "start_ns": case_started, "end_ns": case_started},
        ],
    )
    fixture = prepare_historical_callback_fixture(REPO_ROOT, private_root / "fixture")
    transport = deepcopy(fixture["transport"])
    if case == "timeout":
        transport.update(
            {
                "raw_response": {"error": {"type": "timeout"}},
                "raw_reply": "",
                "terminal_state": "request_error",
                "error": "request_timeout",
                "finish_reason": None,
            }
        )
    elif case == "partial_response":
        transport.update({"raw_reply": "{", "finish_reason": "length"})
    raw_path = private_root / "raw_before_callback.json"
    atomic_json(raw_path, transport)
    cleanup_entered = False
    terminal_state = "unknown"
    callback_row: Mapping[str, Any] = {}
    signals: list[JsonDict] = []
    release: JsonDict = {"released": False}
    try:
        callback_row = normalize_callback_row(fixture["schedule"], transport)
        if case == "callback_exception":
            raise RuntimeError("private_callback_exception")
        if case == "development_gate_closure" and not callback_row["development_usable"]:
            raise DevelopmentGateClosed("development_gate_closed")
        terminal_state = "timeout" if case == "timeout" else "partial_response"
    except DevelopmentGateClosed:
        terminal_state = "development_gate_closed"
    except RuntimeError as exc:
        if str(exc) != "private_callback_exception":
            raise
        terminal_state = "callback_exception"
    finally:
        cleanup_entered = True
        binding["clock_segments"][1]["end_ns"] = time.monotonic_ns()
        signals = _stop_owned_child(process)
        binding["clock_segments"][0]["end_ns"] = time.monotonic_ns()
        try:
            lease.transition("terminal_blocked")
            release = lease.release()
        except lease_api.LeaseError:
            lease.close()
            raise
    return {
        "case": case,
        "condition": case,
        "terminal_state": terminal_state,
        "passed": bool(
            terminal_state
            in {"callback_exception", "development_gate_closed", "timeout", "partial_response"}
            and cleanup_entered
            and process.poll() is not None
            and release.get("released") is True
            and all(row.get("pid") == process.pid for row in signals)
        ),
        "disposition": "complete",
        "censored": False,
        "raw_persisted_before_callback": raw_path.is_file(),
        "callback_disposition": callback_row.get("callback_disposition"),
        "cleanup_entered": cleanup_entered,
        "child_reaped": process.poll() is not None,
        "lease_released": release.get("released") is True,
        "owned_signals": signals,
        "foreign_signal_count": 0,
        "identity_binding": binding,
        "journal_path": str(lease.journal_path),
    }


def build_private_identity_binding(private_root: Path) -> JsonDict:
    """Create deterministic identity evidence for recovery mutation tests."""

    private_root.mkdir(parents=True, exist_ok=True)
    del private_root
    pid = os.getpid()
    ticks = lease_api.proc_start_ticks(pid)
    if ticks is None:
        raise RuntimeError("owner_identity_unavailable")
    return identity_evidence(
        task_id=TASK_ID,
        lease_id="lease:matching-private-evidence",
        owner_pid=pid,
        owner_pid_start_ticks=ticks,
        server_pid=pid,
        server_pid_start_ticks=ticks,
        boot_id=boot_identity(),
        clock_segments=[{"segment": "fixture", "start_ns": 1, "end_ns": 2}],
    )


def authenticate_recovery(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> JsonDict:
    """Authenticate durable continuity without guessing from a path or PID alone."""

    required = {
        "task_id",
        "lease_id",
        "owner_pid",
        "owner_pid_start_ticks",
        "server_pid",
        "server_pid_start_ticks",
        "boot_id",
        "clock_segments",
    }
    if not required.issubset(observed) or not isinstance(observed.get("owner_pid"), int):
        return {"authenticated": False, "reason": "missing_owner_identity", "signals_sent": []}
    if observed.get("boot_id") != expected.get("boot_id"):
        return {"authenticated": False, "reason": "boot_identity_changed", "signals_sent": []}
    segments = observed.get("clock_segments")
    if (
        not isinstance(segments, list)
        or not segments
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("start_ns"), int)
            or not isinstance(row.get("end_ns"), int)
            or row["end_ns"] < row["start_ns"]
            for row in segments
        )
    ):
        return {"authenticated": False, "reason": "clock_segment_invalid", "signals_sent": []}
    if observed.get("owner_pid") == expected.get("owner_pid") and observed.get(
        "owner_pid_start_ticks"
    ) != expected.get("owner_pid_start_ticks"):
        return {"authenticated": False, "reason": "owner_pid_reused", "signals_sent": []}
    if observed.get("server_pid") == expected.get("server_pid") and observed.get(
        "server_pid_start_ticks"
    ) != expected.get("server_pid_start_ticks"):
        return {"authenticated": False, "reason": "server_pid_reused", "signals_sent": []}
    identity_fields = required - {"clock_segments"}
    if all(observed.get(field) == expected.get(field) for field in identity_fields):
        return {"authenticated": True, "reason": "durable_identity_match", "signals_sent": []}
    owner_pid = int(observed["owner_pid"])
    owner_ticks = int(observed.get("owner_pid_start_ticks", -1))
    if lease_api.process_start_matches(owner_pid, owner_ticks):
        return {"authenticated": False, "reason": "foreign_live_owner", "signals_sent": []}
    return {"authenticated": False, "reason": "durable_identity_mismatch", "signals_sent": []}


def _fixture_worker(private_root: Path, stage: str) -> int:
    """Persist or resume one callback in a separate process for restart proof."""

    private_root.mkdir(parents=True, exist_ok=True)
    ledger_path = private_root / "restart_ledger.json"
    raw_path = private_root / "raw_response.json"
    if stage == "persist":
        fixture = prepare_historical_callback_fixture(REPO_ROOT, private_root / "fixture")
        lease_api.write_bytes_atomic(raw_path, fixture["private_raw_path"].read_bytes())
        state = {
            "request_attempts": 1,
            "raw_sha256": sha256_file(raw_path),
            "raw_persisted": True,
            "acknowledgement_persisted": False,
            "schedule": fixture["schedule"],
            "transport": fixture["transport"],
            "child_identities": [
                {
                    "stage": stage,
                    "pid": os.getpid(),
                    "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                }
            ],
        }
        atomic_json(ledger_path, state)
        print(json.dumps({"stage": stage, "raw_persisted": True}), flush=True)
        return INTERRUPTED_EXIT_CODE
    state = load_object(ledger_path)
    if stage != "resume" or state.get("raw_persisted") is not True:
        return 2
    if state.get("raw_sha256") != sha256_file(raw_path):
        return 3
    row = normalize_callback_row(state["schedule"], state["transport"])
    atomic_json(private_root / "acknowledged_callback_row.json", row)
    state["acknowledgement_persisted"] = True
    state["acknowledgement_sha256"] = sha256_file(private_root / "acknowledged_callback_row.json")
    state.setdefault("child_identities", []).append(
        {
            "stage": stage,
            "pid": os.getpid(),
            "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
        }
    )
    atomic_json(ledger_path, state)
    print(json.dumps({"stage": stage, "acknowledged": True}), flush=True)
    return 0


def _run_fixture_process(root: Path, private_root: Path, stage: str) -> tuple[int, list[str]]:
    """Stream one short fresh-process fixture and return its real exit."""

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{root / 'python'}:{root}",
        }
    )
    process = subprocess.Popen(  # noqa: S603 - fixed module and explicit arguments.
        [
            str(root / ".venv/bin/python"),
            "-u",
            "-m",
            "carnot.experiment_7448_v653_capture_lifecycle",
            "--fixture-worker",
            stage,
            "--private-root",
            str(private_root),
        ],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(f"[exp7448-restart:{stage}] {line.rstrip()}", flush=True)
    return process.wait(timeout=10), lines


def run_interruption_restart(root: Path, private_root: Path) -> JsonDict:
    """Prove raw-first restart through two fresh producer callback processes."""

    started_ns = time.monotonic_ns()
    private_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="carnot-exp7448-restart-") as directory:
        child_root = Path(directory)
        first_exit, first_lines = _run_fixture_process(root, child_root, "persist")
        resume_exit, resume_lines = _run_fixture_process(root, child_root, "resume")
        state = load_object(child_root / "restart_ledger.json")
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}:restart:{os.getpid()}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={"fake_transport": True, "fresh_processes": 2},
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    receipt["status"] = "terminal_complete"
    attempts = int(state.get("request_attempts", 0) or 0)
    acknowledged = state.get("acknowledgement_persisted") is True
    return {
        "case": "interruption_restart",
        "condition": "fresh_process_raw_first_restart",
        "terminal_state": "restart_complete" if resume_exit == 0 else "restart_failed",
        "passed": bool(
            first_exit == INTERRUPTED_EXIT_CODE
            and resume_exit == 0
            and attempts == 1
            and acknowledged
        ),
        "disposition": "complete",
        "censored": False,
        "first_exit_code": first_exit,
        "resume_exit_code": resume_exit,
        "raw_persisted_before_callback": state.get("raw_persisted") is True,
        "request_attempts": attempts,
        "duplicate_completed_requests": max(0, attempts - 1),
        "acknowledgement_persisted": acknowledged,
        "lost_acknowledgements": int(not acknowledged),
        "foreign_signal_count": 0,
        "signals_sent": [],
        "child_identities": deepcopy(state.get("child_identities") or []),
        "subprocess_output": {"persist": first_lines, "resume": resume_lines},
        "current_work_receipt": receipt,
    }


def _ownership_rows(private_root: Path) -> list[JsonDict]:
    """Run every durable recovery mutation, including one live foreign child."""

    expected = build_private_identity_binding(private_root / "identity")
    cases: list[tuple[str, JsonDict, str]] = []
    cases.append(("matching_owner_evidence", deepcopy(expected), "durable_identity_match"))
    missing = deepcopy(expected)
    missing.pop("owner_pid")
    cases.append(("missing_owner_identity", missing, "missing_owner_identity"))
    owner_reuse = deepcopy(expected)
    owner_reuse["owner_pid_start_ticks"] += 1
    cases.append(("owner_pid_reuse", owner_reuse, "owner_pid_reused"))
    server_reuse = deepcopy(expected)
    server_reuse["server_pid_start_ticks"] += 1
    cases.append(("server_pid_reuse", server_reuse, "server_pid_reused"))
    rebooted = deepcopy(expected)
    rebooted["boot_id"] = "changed-boot-identity"
    cases.append(("changed_boot_identity", rebooted, "boot_identity_changed"))
    rows: list[JsonDict] = []
    for case, observed, wanted in cases:
        decision = authenticate_recovery(observed, expected)
        rows.append(
            {
                "case": case,
                "condition": case,
                "terminal_state": "authenticated" if decision["authenticated"] else "rejected",
                "passed": decision["reason"] == wanted,
                "disposition": "complete",
                "censored": False,
                "observed_reason": decision["reason"],
                "expected_reason": wanted,
                "identity_binding": observed,
                "signals_sent": decision["signals_sent"],
                "foreign_signal_count": 0,
            }
        )

    process = subprocess.Popen([sys.executable, "-u", "-c", "import time; time.sleep(30)"])
    try:
        ticks = _wait_for_start_ticks(process)
        foreign = deepcopy(expected)
        foreign.update(
            {
                "lease_id": "lease:foreign-live-owner",
                "owner_pid": process.pid,
                "owner_pid_start_ticks": ticks,
            }
        )
        decision = authenticate_recovery(foreign, expected)
        rows.append(
            {
                "case": "foreign_live_owner",
                "condition": "foreign_live_owner",
                "terminal_state": "rejected",
                "passed": decision["reason"] == "foreign_live_owner" and process.poll() is None,
                "disposition": "complete",
                "censored": False,
                "observed_reason": decision["reason"],
                "expected_reason": "foreign_live_owner",
                "identity_binding": foreign,
                "signals_sent": decision["signals_sent"],
                "foreign_signal_count": 0,
            }
        )
    finally:
        _stop_owned_child(process)
    return rows


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str = EXPERIMENT_ID,
    path: str = RESULT_PATH.as_posix(),
    field: str | None = None,
) -> JsonDict:
    """Keep a gate's type, exact operands, authority, and rule visible."""

    return {
        "check": check,
        "category": category,
        "op": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field or check,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first exact failure without hiding later failed checks."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("path") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "field": first.get("field") if first else "gate_check_summary",
        "op": first.get("op") if first else "==",
        "expected": deepcopy(first.get("expected")) if first else True,
        "observed": deepcopy(first.get("observed")) if first else True,
        "passed": not failed,
    }


def historical_failure_hashes(root: Path) -> JsonDict:
    """Preserve the measured exception and source disposition from exact bytes."""

    artifact_path = root / HISTORICAL_ARTIFACT_PATH
    raw_path = root / HISTORICAL_RAW_PATH
    artifact = load_object(artifact_path)
    return {
        "experiment_artifact_path": HISTORICAL_ARTIFACT_PATH.as_posix(),
        "experiment_artifact_sha256": sha256_file(artifact_path),
        "raw_response_path": HISTORICAL_RAW_PATH.as_posix(),
        "raw_response_sha256": sha256_file(raw_path),
        "producer_runtime_error": artifact.get("producer_runtime_error"),
        "honest_verdict": artifact.get("honest_verdict"),
        "verdict_class": artifact.get("verdict_class"),
        "flagged_adversarial": artifact.get("flagged_adversarial"),
        "quarantined_from_current_science": True,
    }


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate every local input and the original historical flags."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        exists = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "==",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if exists else "missing",
                exists,
                "Each declared input must exist before its dependent audit branch runs.",
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
        if exists:
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
            }
    historical = load_object(root / HISTORICAL_ARTIFACT_PATH)
    expected = {
        "producer_runtime_error": "KeyError:'parse_status'",
        "honest_verdict": "complete_disqualified_span_capture_producer_runtime_error",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
    }
    for field, wanted in expected.items():
        checks.append(
            _gate(
                f"historical_{field}",
                "precondition",
                "==",
                wanted,
                historical.get(field),
                historical.get(field) == wanted,
                "The repair must preserve the original failure instead of rehabilitating it.",
                upstream=HISTORICAL_ARTIFACT_PATH.as_posix(),
                path=HISTORICAL_ARTIFACT_PATH.as_posix(),
                field=field,
            )
        )
    return checks, hashes


def _span(
    phase: str, phase_started_ns: int, run_started_ns: int, completed_units: int, checkpoint: str
) -> JsonDict:
    """Close one measured phase and bind it to this host boot."""

    ended_ns = time.monotonic_ns()
    return {
        "phase": phase,
        "start_s": (phase_started_ns - run_started_ns) / 1_000_000_000,
        "end_s": (ended_ns - run_started_ns) / 1_000_000_000,
        "duration_s": (ended_ns - phase_started_ns) / 1_000_000_000,
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "checkpoint_at_utc": utc_now(),
        "boot_id": boot_identity(),
        "clock_segment": {"start_ns": phase_started_ns, "end_ns": ended_ns},
    }


def _callback_case(root: Path, private_root: Path) -> JsonDict:
    """Reduce the historical callback mismatch through the typed boundary."""

    fixture = prepare_historical_callback_fixture(root, private_root)
    row = normalize_callback_row(fixture["schedule"], fixture["transport"])
    passed = bool(
        fixture["legacy_error"] == "KeyError:'parse_status'"
        and row["parse_status"] == "valid"
        and row["callback_disposition"] == "correct_empty"
        and row["semantic_success"] is False
        and row["development_usable"] is False
        and row["raw_request_sha256"] == fixture["raw_request_sha256"]
        and row["raw_response_sha256"] == fixture["raw_response_sha256"]
        and row["raw_reply_sha256"] == fixture["raw_reply_sha256"]
    )
    return {
        "case": "callback_shape",
        "condition": "historical_empty_claims_response",
        "terminal_state": "normalized",
        "passed": passed,
        "disposition": "complete",
        "censored": False,
        "legacy_error": fixture["legacy_error"],
        "callback_disposition": row["callback_disposition"],
        "parse_status": row["parse_status"],
        "semantic_success": row["semantic_success"],
        "development_usable": row["development_usable"],
        "raw_request_sha256": row["raw_request_sha256"],
        "raw_response_sha256": row["raw_response_sha256"],
        "raw_reply_sha256": row["raw_reply_sha256"],
        "foreign_signal_count": 0,
    }


def _lifecycle_gates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce callback, cleanup, ownership, and restart branches separately."""

    by_case = {str(row.get("case")): row for row in rows}
    groups = (
        ("typed_callback_compatibility", {"callback_shape"}, "validity"),
        (
            "owned_failure_path_cleanup",
            {"callback_exception", "development_gate_closure", "timeout", "partial_response"},
            "safety",
        ),
        (
            "durable_recovery_authentication",
            {
                "matching_owner_evidence",
                "missing_owner_identity",
                "owner_pid_reuse",
                "server_pid_reuse",
                "changed_boot_identity",
                "foreign_live_owner",
            },
            "safety",
        ),
        ("fresh_process_restart_conservation", {"interruption_restart"}, "completion"),
    )
    gates: list[JsonDict] = []
    for check, cases, category in groups:
        observed = sum(by_case.get(case, {}).get("passed") is True for case in cases)
        gates.append(
            _gate(
                check,
                category,
                "==",
                len(cases),
                observed,
                observed == len(cases),
                "Every independently named lifecycle mutation must pass before readiness.",
                field="lifecycle_rows.passed",
            )
        )
    return gates


def build_fixture_artifact(root: Path, private_root: Path) -> JsonDict:
    """Run all no-model lifecycle cases and build a cold-replayable artifact."""

    run_started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    phase_started = time.monotonic_ns()
    preconditions, source_hashes = collect_preconditions(root)
    spans.append(
        _span(
            "preconditions",
            phase_started,
            run_started_ns,
            len(preconditions),
            "inputs_authenticated",
        )
    )
    if any(row.get("passed") is not True for row in preconditions):
        raise ValueError("fixture_precondition_failed")

    phase_started = time.monotonic_ns()
    rows = [_callback_case(root, private_root / "callback")]
    for case in ("callback_exception", "development_gate_closure", "timeout", "partial_response"):
        rows.append(run_owned_lifecycle_case(case, private_root / "lifecycle" / case))
    rows.extend(_ownership_rows(private_root / "ownership"))
    rows.append(run_interruption_restart(root, private_root / "restart"))
    spans.append(
        _span("lifecycle_mutations", phase_started, run_started_ns, len(rows), "mutations_complete")
    )

    ended_ns = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=f"{EXPERIMENT_ID}:{os.getpid()}:{run_started_ns}",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=INFERENCE_SUBSTRATE,
        inference_substrate_details={
            "fake_transport": True,
            "real_short_lived_children": True,
            "model_device": None,
            "historical_cuda_is_current": False,
            "boot_id": boot_identity(),
        },
        inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
        execution_venue=EXECUTION_VENUE,
        started_monotonic_ns=run_started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=(
            sidecar_reference(
                root / HISTORICAL_ARTIFACT_PATH,
                root=root,
                scope="historical_model_receipts",
            ),
            sidecar_reference(
                root / HISTORICAL_RAW_PATH,
                root=root,
                scope="historical_model_receipts",
            ),
        ),
        phase_spans=spans,
        small_ebm_training={"performed": False, "reason": "lifecycle_audit_only"},
    )
    gates = _lifecycle_gates(rows)
    ready = int(
        {str(row.get("case")) for row in rows} == REQUIRED_CASES
        and all(row.get("passed") is True for row in rows)
        and all(row.get("passed") is True for row in gates)
    )
    history = historical_failure_hashes(root)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "status": "complete_null",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "preconditions_checked": preconditions,
        **receipt,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "model_duration_s": 0.0,
        "computation_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "random_seed": {
            "fit": None,
            "projection": None,
            "stream": None,
            "resampling": None,
            "reason": "deterministic lifecycle mutations use no randomness",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "historical_failure_hashes": history,
        "rows": deepcopy(rows),
        "lifecycle_rows": rows,
        "sample_size_budget": {
            "planned": len(REQUIRED_CASES),
            "attempted": len(rows),
            "completed": sum(row.get("passed") is True for row in rows),
            "failed": sum(row.get("passed") is not True for row in rows),
            "censored": sum(row.get("censored") is True for row in rows),
            "unstarted": len(REQUIRED_CASES - {str(row.get("case")) for row in rows}),
            "independent_units": len(rows),
            "stop_rule": "Run each frozen lifecycle mutation once; do not retry a failed case.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": "complete_null_capture_lifecycle_qualified"
        if ready
        else "complete_disqualified_capture_lifecycle_failed",
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "capture_lifecycle_ready_score": ready,
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": "pending",
            "numbered_e2e": "not_applicable_isolated_reporting_study",
        },
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def required_receipt_errors(
    receipts: Sequence[Mapping[str, Any]], *, require_terminal: bool
) -> list[str]:
    """Require one successful receipt for every declared scoped reader."""

    wanted = set(AFFECTED_CHECK_NAMES)
    if require_terminal:
        wanted.update(TERMINAL_CHECK_NAMES)
    errors: list[str] = []
    for name in sorted(wanted):
        matches = [row for row in receipts if row.get("name") == name]
        if len(matches) != 1:
            errors.append(f"receipt_count:{name}:{len(matches)}")
        elif (
            matches[0].get("passed") is not True
            or matches[0].get("exit_code") != 0
            or matches[0].get("timed_out") is True
        ):
            errors.append(f"receipt_failed:{name}")
    return errors


def _validation_names(receipts: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("name")) for row in receipts}


def independent_reduce_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool | None = None
) -> list[str]:
    """Recompute identities, lifecycle readiness, current work, and verdict."""

    errors: list[str] = []
    expected_identity = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "MODEL_SPECS": [],
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "promotion_score": 0,
        "verifier_is_oracle": False,
    }
    for field, expected in expected_identity.items():
        if value.get(field) != expected:
            errors.append(f"identity_mismatch:{field}")
    current_errors = validate_current_work_receipt(value, root=root)
    errors.extend(current_errors)
    if value.get("model_invoked") is not False:
        errors.append("model_invoked_mismatch")
    if value.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_mismatch")

    history = historical_failure_hashes(root)
    if value.get("historical_failure_hashes") != history:
        errors.append("historical_failure_hashes_mismatch")
    rows = value.get("lifecycle_rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        errors.append("lifecycle_rows_invalid")
        row_values: list[Mapping[str, Any]] = []
    else:
        row_values = rows
    if value.get("rows") != rows:
        errors.append("rows_mismatch")
    cases = {str(row.get("case")) for row in row_values}
    if cases != REQUIRED_CASES:
        errors.append("lifecycle_case_set_mismatch")
    expected_ready = int(
        cases == REQUIRED_CASES and all(row.get("passed") is True for row in row_values)
    )
    receipts = value.get("validation_receipts")
    receipt_values = (
        [row for row in receipts if isinstance(row, Mapping)] if isinstance(receipts, list) else []
    )
    names = _validation_names(receipt_values)
    validation_declared = bool(names)
    terminal_required = (
        set(TERMINAL_CHECK_NAMES).issubset(names) if require_terminal is None else require_terminal
    )
    if validation_declared:
        receipt_errors = required_receipt_errors(receipt_values, require_terminal=terminal_required)
        if receipt_errors:
            expected_ready = 0
            errors.extend(receipt_errors)
    if value.get("capture_lifecycle_ready_score") != expected_ready:
        errors.append("ready_score_mismatch")
    if value.get("capture_lifecycle_ready_score") != int(
        all(row.get("passed") is True for row in row_values)
    ):
        errors.append("lifecycle_readiness_mismatch")
    expected_verdict = "null" if expected_ready else "disqualified"
    expected_honest = (
        "complete_null_capture_lifecycle_qualified"
        if expected_ready
        else "complete_disqualified_capture_lifecycle_failed"
    )
    if value.get("verdict_class") != expected_verdict:
        errors.append("verdict_mismatch")
    if value.get("honest_verdict") != expected_honest:
        errors.append("honest_verdict_mismatch")
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list):
        errors.append("acceptance_gates_invalid")
    elif value.get("gate_check_summary") != _gate_summary(gates):
        errors.append("gate_check_summary_mismatch")
    if set(value.get("field_principles") or {}) != set(FIELD_PRINCIPLES):
        errors.append("field_principles_mismatch")
    source_hashes = value.get("source_artifact_hashes")
    if not isinstance(source_hashes, Mapping):
        errors.append("source_hashes_invalid")
    else:
        for label, receipt in source_hashes.items():
            if not isinstance(receipt, Mapping):
                errors.append(f"source_receipt_invalid:{label}")
                continue
            path = root / str(receipt.get("path") or label)
            observed = sha256_file(path) if path.is_file() else None
            if receipt.get("sha256") != observed:
                errors.append(f"source_hash_mismatch:{label}")
    return list(dict.fromkeys(errors))


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool | None = None
) -> list[str]:
    """Cold-check independent reductions, byte receipts, and final checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors = independent_reduce_artifact(value, root=root, require_terminal=require_terminal)
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    for row in value.get("validation_receipts") or []:
        if not isinstance(row, Mapping):
            errors.append("validation_receipt_invalid")
            continue
        log_path = root / str(row.get("log_path") or "")
        if log_path.is_file() and row.get("log_sha256") != sha256_file(log_path):
            errors.append(f"validation_log_hash_mismatch:{row.get('name')}")
    return list(dict.fromkeys(errors))


def build_validation_commands(root: Path, private_root: Path) -> list[CommandSpec]:
    """Build and freeze the Exp7358/Exp7303 affected-file command plan."""

    commands = build_command_plan(root, VALIDATION_MANIFEST, private_root)
    errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if errors:
        raise ValueError("invalid_validation_plan:" + ",".join(errors))
    return list(commands)


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build the two cold readers and two unchanged terminal guards."""

    python = str(root / ".venv/bin/python")
    relative = candidate.relative_to(root).as_posix()
    reducer = (
        "import json,sys; from pathlib import Path; "
        "from carnot import experiment_7448_v653_capture_lifecycle as m; "
        "p=Path(sys.argv[1]); v=json.loads(p.read_text()); "
        "e=m.independent_reduce_artifact(v, root=Path.cwd()); "
        "print(json.dumps({'errors':e},sort_keys=True),flush=True); raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--validate", relative),
            "capability_e2e",
            180,
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, relative),
            "completion",
            180,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", relative),
            "safety",
            180,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", relative),
            "completion",
            180,
        ),
    ]


def _finalize_with_validation(
    artifact: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], *, require_terminal: bool
) -> JsonDict:
    """Attach measured validation and derive readiness without changing rows."""

    result = deepcopy(dict(artifact))
    receipt_rows = [deepcopy(dict(row)) for row in receipts]
    errors = required_receipt_errors(receipt_rows, require_terminal=require_terminal)
    lifecycle_ok = all(row.get("passed") is True for row in result["lifecycle_rows"])
    passed = lifecycle_ok and not errors
    gates = [
        deepcopy(dict(row))
        for row in result["acceptance_gate_results"]
        if row.get("check") != "required_validation"
    ]
    gates.append(
        _gate(
            "required_validation",
            "validity",
            "==",
            [],
            errors,
            not errors,
            "Every frozen affected and terminal reader must exit successfully.",
            field="validation_receipts",
        )
    )
    result.update(
        {
            "status": "complete_null" if passed else "complete_required_validation_failed",
            "completed_at_utc": utc_now(),
            "capture_lifecycle_ready_score": int(passed),
            "honest_verdict": "complete_null_capture_lifecycle_qualified"
            if passed
            else "complete_disqualified_capture_lifecycle_failed",
            "verdict_class": "null" if passed else "disqualified",
            "flagged_adversarial": any(
                row.get("name") == "adversarial_verify" and row.get("passed") is not True
                for row in receipt_rows
            ),
            "validation_receipts": receipt_rows,
            "validation_duration_s": sum(
                float(row.get("duration_s", 0.0) or 0.0) for row in receipt_rows
            ),
            "acceptance_gate_results": gates,
            "gate_check_summary": _gate_summary(gates),
        }
    )
    result["capability_e2e"]["fresh_process_cold_replay"] = (
        "passed" if require_terminal and not errors else "pending"
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - exercised by the declared capability E2E.
    """Run mutations, scoped checks, cold readers, and one atomic publication."""

    started = time.monotonic()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(started, "preconditions", "start", completed_units=0)
    if run_date != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{run_date}")
    progress(started, "lifecycle", "start", completed_units=0)
    artifact = build_fixture_artifact(root, raw_dir / "private")
    progress(
        started,
        "lifecycle",
        "complete",
        completed_units=len(artifact["lifecycle_rows"]),
        ready=artifact["capture_lifecycle_ready_score"],
    )

    progress(started, "affected_validation", "before_subprocesses", completed_units=0)
    commands = build_validation_commands(root, raw_dir / "validation_private")
    affected_receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60,
    )
    affected = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected_receipts)
    if affected.get("passed") is not True:
        progress(started, "affected_validation", "failed", completed_units=len(affected_receipts))
    candidate = _finalize_with_validation(artifact, affected_receipts, require_terminal=False)
    candidate_path = raw_dir / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(affected_receipts),
        passed=affected.get("passed"),
    )

    terminal_specs = _terminal_commands(root, candidate_path)
    progress(started, "terminal_validation", "before_subprocesses", planned=len(terminal_specs))
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                command,
                "safety" if command.name == "adversarial_verify" else "completion",
                True,
            )
            for command in terminal_specs
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60,
    )
    final = _finalize_with_validation(
        artifact, [*affected_receipts, *terminal_receipts], require_terminal=True
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        final.update(
            {
                "status": "complete_internal_validation_failed",
                "capture_lifecycle_ready_score": 0,
                "honest_verdict": "complete_disqualified_capture_lifecycle_failed",
                "verdict_class": "disqualified",
                "flagged_adversarial": True,
                "internal_validation_errors": errors,
            }
        )
        final["reproducibility_checksum"] = artifact_checksum(final)
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=not errors,
    )
    progress(started, "publish", "before_atomic_publish", path=output)
    atomic_json(output, final)
    progress(
        started,
        "publish",
        "after_atomic_publish",
        completed_units=1,
        verdict=final["honest_verdict"],
    )
    return final


def date_argument(value: str) -> str:
    """Accept only the fixed execution date used by the V653 contract."""

    if value != RUN_DATE:
        raise ValueError(f"run_date_mismatch:{value}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    """Run the qualification, validate a candidate, or serve a private fixture."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--fixture-worker", choices=("persist", "resume"))
    parser.add_argument("--private-root", type=Path)
    args = parser.parse_args(argv)
    if args.fixture_worker:
        if args.private_root is None:
            parser.error("--private-root is required with --fixture-worker")
        return _fixture_worker(args.private_root, args.fixture_worker)
    if args.validate is not None:
        value = load_object(args.validate)
        names = _validation_names(value.get("validation_receipts") or [])
        errors = validate_artifact(
            value,
            root=REPO_ROOT,
            require_terminal=set(TERMINAL_CHECK_NAMES).issubset(names),
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "capture_lifecycle_ready_score": result["capture_lifecycle_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
