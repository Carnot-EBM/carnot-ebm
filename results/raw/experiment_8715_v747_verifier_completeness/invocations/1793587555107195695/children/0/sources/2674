"""Audit migrated GPU leases in fresh processes without loading a model.

The controller uses task-owned child processes and isolated runtime paths.
An owner receipt is the barrier for contention, so elapsed time never grants
lease authority. Spec refs: REQ-INFRA-7079 and SCENARIO-INFRA-7079-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_7079_v620_gpu_lease_audit.json"
UPSTREAM_PATH = REPO_ROOT / "results/experiment_7078_v620_gpu_lease_migration.json"
EXPECTED_UPSTREAM_SHA256 = "sha256:8cd5e208f247cc8def8bcef260ec3e8681aef3f166c75a838597dd6decdb9c15"
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
AUDIT_RUNTIME_DIR = Path(
    os.environ.get("CARNOT_GPU_LEASE_AUDIT_RUNTIME_DIR", "/tmp/carnot-gpu-lease-audit-7079")
)
INFERENCE_SUBSTRATE = "fresh_process_os_lease_audit_no_llm"
RANDOM_SEED = 7_079_202_609_06
PROCESS_TIMEOUT_S = 10.0
CRASH_EXIT_CODE = 79
EXPECTED_MODEL = "no-model-loaded/lease-audit.gguf"

PRECONDITION_CHECK_IDS = (
    "upstream_artifact_hash",
    "upstream_compatibility_ready",
    "exact_idle_rtx_3090_topology",
    "unattributed_gpu_processes",
    "lease_preflight_available",
    "clean_stop_authority",
    "isolated_audit_paths_writable",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "upstream_gate_rows",
    "rows",
    "gpu_topology_rows",
    "same_device_race_rows",
    "independent_device_rows",
    "crash_recovery_rows",
    "pid_identity_rows",
    "phase_history_rows",
    "checksum_rows",
    "release_rows",
    "fresh_reread_rows",
    "post_audit_preflight_rows",
    "signals_sent",
    "model_load_count",
    "gpu_lease_cold_audit_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for every field makes omitted evidence visible.",
    "preconditions_checked": "Read-only gates stop work when ownership is uncertain.",
    "inference_substrate": "The substrate states that no language model produced evidence.",
    "duration_s": "Measured wall time exposes an interrupted process audit.",
    "source_artifact_hashes": "Source hashes bind evidence to reviewed code and inputs.",
    "cited_upstream_artifacts": "An exact citation identifies the migration contract used.",
    "upstream_gate_rows": "Bare upstream gates prevent stale readiness from being inherited.",
    "rows": "Recomputed gates stop prose from overriding failed evidence.",
    "gpu_topology_rows": "UUID, index, model, and idle state identify both target devices.",
    "same_device_race_rows": "Kernel exclusion must admit one owner and reject one contender.",
    "independent_device_rows": "Different UUIDs must progress without a shared global lock.",
    "crash_recovery_rows": "Crash evidence proves kernel release and safe durable recovery.",
    "pid_identity_rows": "PID start ticks prevent a reused number from gaining authority.",
    "phase_history_rows": "Ordered events prove that owners did not skip required phases.",
    "checksum_rows": "Content hashes expose changed journals and changed event history.",
    "release_rows": "Terminal release proves each device can pass to a later task.",
    "fresh_reread_rows": "A separate process proves evidence survives its writer.",
    "post_audit_preflight_rows": "The entrance consumer must see both devices as available.",
    "signals_sent": "An empty ledger proves the audit did not signal another owner.",
    "model_load_count": "Zero separates lease evidence from model execution.",
    "gpu_lease_cold_audit_ready_score": "One requires every cold-audit gate to pass.",
    "random_seed": "A fixed protocol identifier supports exact reruns.",
    "reproducibility_checksum": "A terminal hash detects later artifact mutation.",
    "gate_check_summary": "Expected and observed values make a block actionable.",
    "verifier_is_oracle": "False prevents infrastructure evidence from becoming a science claim.",
    "verdict_class": "A closed class gives automation one terminal interpretation.",
    "honest_verdict": "A class-matched prefix states readiness without model claims.",
}


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes or return an explicit missing sentinel."""

    candidate = Path(path)
    try:
        return "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the self-referential checksum."""

    return lease_api.sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Use one expected-observed structure for every decision."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and lift the first failure into a short diagnosis."""

    checks = [deepcopy(dict(row)) for row in rows]
    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "checks": checks,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _writable(path: Path) -> bool:
    """Test atomic-write-compatible access without retaining probe files."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".exp7079-write-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def _default_gpu_probe() -> JsonDict:
    from carnot.experiment_6966_gguf_load_envelope_canary import gpu_inventory

    return gpu_inventory()


def _default_stop_authority_probe() -> JsonDict:
    from carnot import experiment_7065_v619_three_family_entrance_bank as entrance_api

    return entrance_api._stop_authority_probe()


def _default_lease_probe(devices: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    from carnot import experiment_7065_v619_three_family_entrance_bank as entrance_api

    return entrance_api._lease_probe(devices)


def collect_preconditions(
    *,
    upstream_path: Path,
    expected_upstream_hash: str,
    result_path: Path,
    audit_runtime_dir: Path,
    gpu_probe: Callable[[], JsonDict] = _default_gpu_probe,
    stop_authority_probe: Callable[[], JsonDict] = _default_stop_authority_probe,
    writable_probe: Callable[[Path], bool] = _writable,
    lease_probe: Callable[[Sequence[Mapping[str, Any]]], list[JsonDict]] = _default_lease_probe,
) -> JsonDict:
    """Check upstream, hardware, owners, authority, and isolated paths first."""

    upstream: JsonDict
    try:
        loaded = json.loads(upstream_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("upstream_not_object")
        upstream = dict(loaded)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        upstream = {"read_error": f"{type(exc).__name__}:{exc}"}
    observed_hash = sha256_file(upstream_path)
    gpu = gpu_probe()
    devices = [deepcopy(dict(row)) for row in gpu.get("devices", [])]
    processes = [deepcopy(dict(row)) for row in gpu.get("processes", [])]
    leases = lease_probe(devices)
    stop = stop_authority_probe()
    topology_rows = [
        {
            **deepcopy(row),
            "expected_index": index,
            "uuid_present": bool(row.get("uuid")),
            "rtx_3090": "RTX 3090" in str(row.get("name", "")),
            "idle": int(row.get("utilization_gpu_pct", -1)) == 0,
            "passed": row.get("index") == index
            and bool(row.get("uuid"))
            and "RTX 3090" in str(row.get("name", ""))
            and int(row.get("utilization_gpu_pct", -1)) == 0,
        }
        for index, row in enumerate(devices)
    ]
    topology_ok = bool(
        gpu.get("query_ok") is True
        and len(devices) == 2
        and len({str(row.get("uuid", "")) for row in devices}) == 2
        and all(row["passed"] for row in topology_rows)
    )
    lease_ok = len(leases) == 2 and all(row.get("classification") == "available" for row in leases)
    path_state = {
        "result_path": writable_probe(result_path),
        "audit_runtime_dir": writable_probe(audit_runtime_dir / "probe.json"),
    }
    upstream_rows = [
        gate_row(
            "upstream_artifact_hash",
            expected_upstream_hash,
            observed_hash,
            observed_hash == expected_upstream_hash,
        ),
        gate_row(
            "upstream_compatibility_ready",
            1,
            upstream.get("gpu_lease_compatibility_ready_score"),
            upstream.get("gpu_lease_compatibility_ready_score") == 1,
        ),
    ]
    checks = [
        upstream_rows[0],
        upstream_rows[1],
        gate_row("exact_idle_rtx_3090_topology", 2, topology_rows, topology_ok),
        gate_row("unattributed_gpu_processes", [], processes, not processes),
        gate_row("lease_preflight_available", ["available", "available"], leases, lease_ok),
        gate_row("clean_stop_authority", True, stop.get("observed"), stop.get("passed") is True),
        gate_row(
            "isolated_audit_paths_writable",
            {"result_path": True, "audit_runtime_dir": True},
            path_state,
            all(path_state.values()),
        ),
    ]
    return {
        "all_passed": all(row["passed"] for row in checks),
        "checks": checks,
        "upstream": upstream,
        "upstream_hash": observed_hash,
        "upstream_gate_rows": upstream_rows,
        "gpu_inventory": deepcopy(gpu),
        "gpu_topology_rows": topology_rows,
        "lease_preflight_rows": deepcopy(leases),
        "stop_authority": deepcopy(stop),
        "path_state": path_state,
        "signals_sent": [],
    }


def _worker_command(runtime_dir: Path, device_uuid: str, task_id: str, behavior: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "carnot.experiment_7079_v620_gpu_lease_audit",
        "--worker",
        "--runtime-dir",
        str(runtime_dir),
        "--device-uuid",
        device_uuid,
        "--task-id",
        task_id,
        "--behavior",
        behavior,
    ]


def _start_worker(command: Sequence[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _readline_bounded(process: subprocess.Popen[str]) -> JsonDict:
    if process.stdout is None:
        raise RuntimeError("worker_stdout_missing")
    selector = selectors.DefaultSelector()
    try:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(PROCESS_TIMEOUT_S):
            process.kill()
            process.wait(timeout=PROCESS_TIMEOUT_S)
            raise TimeoutError("worker_receipt_timeout")
        line = process.stdout.readline()
    finally:
        selector.close()
    if not line:
        stderr = "" if process.stderr is None else process.stderr.read()
        raise RuntimeError(f"worker_receipt_missing:{stderr}")
    return dict(json.loads(line))


def _finish_worker(
    process: subprocess.Popen[str], *, input_text: str | None = None
) -> tuple[int, list[JsonDict], str]:
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=PROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate(timeout=PROCESS_TIMEOUT_S)
        raise TimeoutError("worker_completion_timeout") from exc
    rows = [dict(json.loads(line)) for line in stdout.splitlines() if line.strip()]
    return int(process.returncode or 0), rows, stderr


def _run_worker(command: Sequence[str]) -> tuple[int, list[JsonDict], str]:
    return _finish_worker(_start_worker(command))


def _complete(lease: lease_api.GpuLease) -> None:
    lease.transition("admitted")
    lease.transition("loading")
    lease.transition("resident", vram_mb=0)
    lease.transition("inferencing")
    lease.transition("unloading")
    lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
    lease.transition("terminal_complete")


def worker_main(argv: Sequence[str] | None = None) -> int:
    """Run one fresh-process lease action for the controller."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--behavior",
        choices=(
            "hold_complete",
            "hold_short",
            "full",
            "crash",
            "recover",
            "abandon_live",
            "reread",
        ),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.behavior == "reread":
        path = lease_api.journal_path_for(args.runtime_dir, args.device_uuid)
        try:
            document = lease_api.read_journal(path)
            errors = lease_api.validate_journal_document(document, check_freshness=False)
        except lease_api.LeaseError as exc:
            print(json.dumps({"outcome": type(exc).__name__, "reason": str(exc)}), flush=True)
            return 4
        history = [row["phase"] for row in document["phase_history"]]
        event_checksums_valid = all(
            lease_api.event_checksum(row) == row.get("event_checksum")
            for row in document["phase_history"]
        )
        print(
            json.dumps(
                {
                    "outcome": "reread",
                    "reader_pid": os.getpid(),
                    "errors": errors,
                    "history": history,
                    "journal_checksum_valid": lease_api.journal_checksum(document)
                    == document.get("checksum"),
                    "event_checksums_valid": event_checksums_valid,
                    "released": document.get("released"),
                    "phase": document.get("phase"),
                }
            ),
            flush=True,
        )
        return int(bool(errors))
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=args.runtime_dir,
            task_id=args.task_id,
            device_uuid=args.device_uuid,
            expected_model=EXPECTED_MODEL,
            vram_before_mb=0,
            ttl_s=30.0,
        )
    except lease_api.LeaseBusy as exc:
        print(
            json.dumps({"outcome": "LeaseBusy", "reason": str(exc), "signals_sent": []}),
            flush=True,
        )
        return 3
    except (lease_api.JournalError, lease_api.RecoveryError) as exc:
        print(
            json.dumps({"outcome": type(exc).__name__, "reason": str(exc), "signals_sent": []}),
            flush=True,
        )
        return 4
    acquired_row = {
        "outcome": "acquired",
        "owner": lease.owner_receipt(),
        "journal_path": str(lease.journal_path),
    }
    if args.behavior == "abandon_live":
        lease.close()
        acquired_row["lock_released_owner_live"] = True
    print(json.dumps(acquired_row), flush=True)
    if args.behavior == "crash":
        os._exit(CRASH_EXIT_CODE)
    if args.behavior == "abandon_live":
        if sys.stdin.readline().strip() != "exit":
            return 5
        return 0
    if args.behavior in {"hold_complete", "hold_short"}:
        if sys.stdin.readline().strip() != "continue":
            lease.close()
            return 5
    if args.behavior == "hold_short":
        lease.transition("admitted")
        lease.transition("terminal_blocked")
    else:
        _complete(lease)
    release = lease.release()
    print(json.dumps({"outcome": "released", "release": release}), flush=True)
    return 0


def _kernel_lock_available(runtime_dir: Path, device_uuid: str) -> bool:
    path = lease_api.lock_path_for(runtime_dir, device_uuid)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


def _outcome(rows: Sequence[Mapping[str, Any]], name: str) -> JsonDict:
    return deepcopy(dict(next((row for row in rows if row.get("outcome") == name), {})))


def run_process_audit(runtime_dir: Path, devices: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run exclusion, independence, crash, full history, and reread workers."""

    if len(devices) != 2:
        raise ValueError("two_devices_required")
    uuids = [str(row["uuid"]) for row in devices]

    race_dir = runtime_dir / "same-device-race"
    holder_command = _worker_command(race_dir, uuids[0], "exp7079-race-owner", "hold_complete")
    contender_command = _worker_command(race_dir, uuids[0], "exp7079-race-contender", "full")
    holder = _start_worker(holder_command)
    owner_line = _readline_bounded(holder)
    contender_code, contender_rows, _ = _run_worker(contender_command)
    holder_code, holder_rows, _ = _finish_worker(holder, input_text="continue\n")
    race_row = {
        "device_uuid": uuids[0],
        "holder_command": holder_command,
        "contender_command": contender_command,
        "owner_receipt_observed_before_contender": owner_line.get("outcome") == "acquired",
        "timing_is_authority": False,
        "acquired_count": int(owner_line.get("outcome") == "acquired")
        + sum(row.get("outcome") == "acquired" for row in contender_rows),
        "lease_busy_count": sum(row.get("outcome") == "LeaseBusy" for row in contender_rows),
        "holder_exit_code": holder_code,
        "contender_exit_code": contender_code,
        "holder_terminal_rows": holder_rows,
        "signals_sent": [],
    }
    race_row["passed"] = bool(
        race_row["owner_receipt_observed_before_contender"]
        and race_row["acquired_count"] == 1
        and race_row["lease_busy_count"] == 1
        and holder_code == 0
        and contender_code == 3
        and _outcome(holder_rows, "released")
    )

    independent_dir = runtime_dir / "independent-devices"
    independent_commands = [
        _worker_command(independent_dir, uuid, f"exp7079-independent-{index}", "hold_short")
        for index, uuid in enumerate(uuids)
    ]
    independent_processes = [_start_worker(command) for command in independent_commands]
    independent_owners = [_readline_bounded(process) for process in independent_processes]
    independent_finished = [
        _finish_worker(process, input_text="continue\n") for process in independent_processes
    ]
    independent_rows = []
    for index, uuid in enumerate(uuids):
        code, terminal_rows, stderr = independent_finished[index]
        document = lease_api.read_journal(lease_api.journal_path_for(independent_dir, uuid))
        independent_rows.append(
            {
                "device_uuid": uuid,
                "command": independent_commands[index],
                "owner": independent_owners[index].get("owner"),
                "exit_code": code,
                "stderr": stderr,
                "history": [row["phase"] for row in document["phase_history"]],
                "released": document["released"],
                "terminal_rows": terminal_rows,
                "signals_sent": [],
                "passed": independent_owners[index].get("outcome") == "acquired"
                and code == 0
                and document["phase"] == "terminal_blocked"
                and document["released"] is True,
            }
        )

    crash_dir = runtime_dir / "crash-recovery"
    crash_command = _worker_command(crash_dir, uuids[0], "exp7079-crashed-owner", "crash")
    crash_process = _start_worker(crash_command)
    crash_acquire = _readline_bounded(crash_process)
    crash_code, _, _ = _finish_worker(crash_process)
    crash_document = lease_api.read_journal(lease_api.journal_path_for(crash_dir, uuids[0]))
    crash_owner = dict(crash_acquire.get("owner", {}))
    old_absent = not lease_api.process_start_matches(
        int(crash_owner["pid"]), int(crash_owner["pid_start_ticks"])
    )
    lock_released = _kernel_lock_available(crash_dir, uuids[0])
    recovery_command = _worker_command(crash_dir, uuids[0], "exp7079-recovery-owner", "recover")
    recovery_code, recovery_rows, _ = _run_worker(recovery_command)
    recovery_acquire = _outcome(recovery_rows, "acquired")
    recovery_owner = dict(recovery_acquire.get("owner", {}))
    recovery = dict(recovery_owner.get("recovery", {}))
    recovered_document = lease_api.read_journal(lease_api.journal_path_for(crash_dir, uuids[0]))
    preserved_identity = all(
        (
            recovery.get("previous_checksum") == crash_document.get("checksum"),
            recovery.get("previous_task_id") == crash_document.get("task_id"),
            recovery.get("previous_pid") == crash_owner.get("pid"),
            recovery.get("previous_pid_start_ticks") == crash_owner.get("pid_start_ticks"),
        )
    )
    crash_row = {
        "device_uuid": uuids[0],
        "crash_command": crash_command,
        "recovery_command": recovery_command,
        "crash_exit_code": crash_code,
        "recovery_exit_code": recovery_code,
        "kernel_lock_released": lock_released,
        "recovery_performed": recovery.get("performed") is True,
        "preserved_previous_identity": preserved_identity,
        "old_lease_id": crash_document.get("lease_id"),
        "new_lease_id": recovered_document.get("lease_id"),
        "old_token_digest": crash_document.get("owner", {}).get("token_digest"),
        "new_token_digest": recovered_document.get("owner", {}).get("token_digest"),
        "recovery": recovery,
        "signals_sent": [],
    }
    crash_row["passed"] = bool(
        crash_code == CRASH_EXIT_CODE
        and recovery_code == 0
        and old_absent
        and lock_released
        and recovery.get("performed") is True
        and preserved_identity
        and crash_row["old_lease_id"] != crash_row["new_lease_id"]
        and crash_row["old_token_digest"] != crash_row["new_token_digest"]
        and recovered_document.get("released") is True
    )

    live_dir = runtime_dir / "live-owner-protection"
    live_command = _worker_command(live_dir, uuids[0], "exp7079-live-owner", "abandon_live")
    live_process = _start_worker(live_command)
    live_acquire = _readline_bounded(live_process)
    live_recovery_command = _worker_command(live_dir, uuids[0], "exp7079-live-recovery", "recover")
    live_recovery_code, live_recovery_rows, _ = _run_worker(live_recovery_command)
    live_exit_code, _, _ = _finish_worker(live_process, input_text="exit\n")
    live_failure = live_recovery_rows[0] if live_recovery_rows else {}
    reused = deepcopy(crash_document)
    reused["owner"]["pid_start_ticks"] = int(reused["owner"]["pid_start_ticks"]) + 1
    reused["checksum"] = lease_api.journal_checksum(reused)
    reused_errors = lease_api.validate_journal_document(
        reused,
        expected_pid=int(crash_document["owner"]["pid"]),
        expected_pid_start_ticks=int(crash_document["owner"]["pid_start_ticks"]),
        check_freshness=False,
    )
    pid_rows = [
        {
            "case": "crashed_owner_absent",
            "pid": crash_owner["pid"],
            "pid_start_ticks": crash_owner["pid_start_ticks"],
            "process_identity_live": not old_absent,
            "passed": old_absent,
        },
        {
            "case": "matching_live_owner",
            "owner_receipt": live_acquire.get("owner"),
            "lock_released_owner_live": live_acquire.get("lock_released_owner_live"),
            "outcome": live_failure.get("outcome"),
            "reason": live_failure.get("reason"),
            "recovery_exit_code": live_recovery_code,
            "owner_exit_code": live_exit_code,
            "fail_closed": live_failure.get("outcome") == "RecoveryError",
            "passed": live_failure.get("outcome") == "RecoveryError"
            and live_recovery_code == 4
            and live_exit_code == 0,
        },
        {
            "case": "pid_start_mismatch",
            "recorded_start_ticks": crash_document["owner"]["pid_start_ticks"],
            "replayed_start_ticks": reused["owner"]["pid_start_ticks"],
            "errors": reused_errors,
            "fail_closed": "pid_start_mismatch" in reused_errors,
            "passed": "pid_start_mismatch" in reused_errors,
        },
    ]

    full_dir = runtime_dir / "full-real-device-sequences"
    phase_rows: list[JsonDict] = []
    checksum_rows: list[JsonDict] = []
    release_rows: list[JsonDict] = []
    reread_rows: list[JsonDict] = []
    for index, uuid in enumerate(uuids):
        full_command = _worker_command(full_dir, uuid, f"exp7079-full-{index}", "full")
        code, output_rows, stderr = _run_worker(full_command)
        acquired = _outcome(output_rows, "acquired")
        released = _outcome(output_rows, "released")
        document = lease_api.read_journal(lease_api.journal_path_for(full_dir, uuid))
        history = [row["phase"] for row in document["phase_history"]]
        validation_errors = lease_api.validate_journal_document(document, check_freshness=False)
        event_valid = all(
            lease_api.event_checksum(row) == row.get("event_checksum")
            for row in document["phase_history"]
        )
        phase_rows.append(
            {
                "case": "full_sequence",
                "device_uuid": uuid,
                "writer_pid": acquired.get("owner", {}).get("pid"),
                "history": history,
                "expected_history": list(lease_api.COMPLETE_PHASE_SEQUENCE),
                "passed": code == 0
                and history == list(lease_api.COMPLETE_PHASE_SEQUENCE)
                and not validation_errors,
            }
        )
        checksum_rows.append(
            {
                "case": "full_sequence",
                "device_uuid": uuid,
                "journal_checksum_valid": lease_api.journal_checksum(document)
                == document.get("checksum"),
                "event_checksums_valid": event_valid,
                "validation_errors": validation_errors,
                "passed": event_valid and not validation_errors,
            }
        )
        release_rows.append(
            {
                "case": "full_sequence",
                "device_uuid": uuid,
                "exit_code": code,
                "stderr": stderr,
                "release": released.get("release"),
                "terminal_phase": document.get("phase"),
                "released": document.get("released"),
                "signals_sent": [],
                "passed": code == 0
                and released.get("release", {}).get("released") is True
                and document.get("released") is True
                and document.get("phase") == "terminal_complete",
            }
        )
        reread_command = _worker_command(full_dir, uuid, f"exp7079-reread-{index}", "reread")
        reread_code, reread_output, reread_stderr = _run_worker(reread_command)
        reread = _outcome(reread_output, "reread")
        reread_rows.append(
            {
                "device_uuid": uuid,
                "command": reread_command,
                "reader_pid": reread.get("reader_pid"),
                "writer_pid": acquired.get("owner", {}).get("pid"),
                "exit_code": reread_code,
                "stderr": reread_stderr,
                **reread,
                "passed": reread_code == 0
                and reread.get("reader_pid") != acquired.get("owner", {}).get("pid")
                and reread.get("history") == list(lease_api.COMPLETE_PHASE_SEQUENCE)
                and reread.get("journal_checksum_valid") is True
                and reread.get("event_checksums_valid") is True
                and reread.get("released") is True,
            }
        )
    return {
        "same_device_race_rows": [race_row],
        "independent_device_rows": independent_rows,
        "crash_recovery_rows": [crash_row],
        "pid_identity_rows": pid_rows,
        "phase_history_rows": phase_rows,
        "checksum_rows": checksum_rows,
        "release_rows": release_rows,
        "fresh_reread_rows": reread_rows,
    }


def _failure_row(
    case: str, expected_type: type[BaseException], action: Callable[[], Any]
) -> JsonDict:
    try:
        action()
    except expected_type as exc:
        return {
            "case": case,
            "outcome": type(exc).__name__,
            "reason": str(exc),
            "fail_closed": True,
            "passed": True,
            "signals_sent": [],
        }
    except Exception as exc:  # noqa: BLE001 - wrong failure type must remain evidence.
        return {
            "case": case,
            "outcome": type(exc).__name__,
            "reason": str(exc),
            "fail_closed": False,
            "passed": False,
            "signals_sent": [],
        }
    return {
        "case": case,
        "outcome": "accepted",
        "reason": "unsafe action accepted",
        "fail_closed": False,
        "passed": False,
        "signals_sent": [],
    }


def build_adversarial_rows(runtime_dir: Path, device_uuid: str) -> JsonDict:
    """Exercise owner, expiry, phase, checksum, and release rejection paths."""

    owner = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "owner-mismatch",
        task_id="exp7079-owner-mismatch",
        device_uuid=device_uuid,
        expected_model=EXPECTED_MODEL,
        vram_before_mb=0,
    )
    owner_row = _failure_row(
        "owner_mismatch", lease_api.OwnershipError, lambda: owner.heartbeat(token="wrong")
    )
    owner.close()

    expired = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "expiry",
        task_id="exp7079-expiry",
        device_uuid=device_uuid,
        expected_model=EXPECTED_MODEL,
        vram_before_mb=0,
    )
    expiry_row = _failure_row(
        "expiry",
        lease_api.LeaseExpired,
        lambda: expired.heartbeat(now_ns=int(expired.document["expires_monotonic_ns"]) + 1),
    )
    expired.close()

    skip = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "phase-skip",
        task_id="exp7079-phase-skip",
        device_uuid=device_uuid,
        expected_model=EXPECTED_MODEL,
        vram_before_mb=0,
    )
    skip_row = _failure_row(
        "phase_skip", lease_api.TransitionError, lambda: skip.transition("loading")
    )
    skip.close()

    checksum = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "checksum-mutation",
        task_id="exp7079-checksum-mutation",
        device_uuid=device_uuid,
        expected_model=EXPECTED_MODEL,
        vram_before_mb=0,
    )
    changed = deepcopy(checksum.document)
    changed["task_id"] = "mutated-without-checksum"
    checksum.journal_path.write_text(json.dumps(changed), encoding="utf-8")
    checksum_row = _failure_row("checksum_mutation", lease_api.JournalError, checksum.heartbeat)
    checksum.close()

    incomplete = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir / "incomplete-release",
        task_id="exp7079-incomplete-release",
        device_uuid=device_uuid,
        expected_model=EXPECTED_MODEL,
        vram_before_mb=0,
    )
    incomplete_row = _failure_row(
        "incomplete_release", lease_api.TransitionError, incomplete.release
    )
    incomplete.close()
    return {
        "pid_identity_rows": [owner_row],
        "phase_history_rows": [skip_row, expiry_row],
        "checksum_rows": [checksum_row],
        "release_rows": [incomplete_row],
    }


def _post_audit_probe(devices: Sequence[Mapping[str, Any]], runtime_dir: Path) -> list[JsonDict]:
    """Call the shipped Exp7065 classifier against the released audit journals."""

    from carnot import experiment_7065_v619_three_family_entrance_bank as entrance_api

    original = entrance_api.LEASE_RUNTIME_DIR
    entrance_api.LEASE_RUNTIME_DIR = runtime_dir / "full-real-device-sequences"
    try:
        return entrance_api._lease_probe(devices)
    finally:
        entrance_api.LEASE_RUNTIME_DIR = original


def _row_passes(row: Mapping[str, Any]) -> bool:
    return row.get("passed") is True or row.get("fail_closed") is True


def build_gate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Cold-reduce each evidence family into the readiness gates."""

    preconditions = artifact.get("preconditions_checked", {})
    precondition_ok = isinstance(preconditions, Mapping) and preconditions.get("all_passed") is True
    race = artifact.get("same_device_race_rows", [])
    independent = artifact.get("independent_device_rows", [])
    crash = artifact.get("crash_recovery_rows", [])
    identities = artifact.get("pid_identity_rows", [])
    phases = artifact.get("phase_history_rows", [])
    checksums = artifact.get("checksum_rows", [])
    releases = artifact.get("release_rows", [])
    rereads = artifact.get("fresh_reread_rows", [])
    post = artifact.get("post_audit_preflight_rows", [])
    identity_cases = {row.get("case") for row in identities}
    phase_cases = {row.get("case") for row in phases}
    checksum_cases = {row.get("case") for row in checksums}
    release_cases = {row.get("case") for row in releases}
    return [
        gate_row("preconditions", True, precondition_ok, precondition_ok),
        gate_row(
            "same_device_exclusion",
            {"rows": 1, "acquired": 1, "busy": 1},
            deepcopy(race),
            len(race) == 1 and all(_row_passes(row) for row in race),
        ),
        gate_row(
            "independent_device_progress",
            2,
            sum(_row_passes(row) for row in independent),
            len(independent) == 2 and all(_row_passes(row) for row in independent),
        ),
        gate_row(
            "crash_recovery",
            1,
            sum(_row_passes(row) for row in crash),
            len(crash) == 1 and all(_row_passes(row) for row in crash),
        ),
        gate_row(
            "pid_identity",
            ["crashed_owner_absent", "matching_live_owner", "owner_mismatch", "pid_start_mismatch"],
            sorted(identity_cases),
            identity_cases
            == {
                "crashed_owner_absent",
                "matching_live_owner",
                "pid_start_mismatch",
                "owner_mismatch",
            }
            and all(_row_passes(row) for row in identities),
        ),
        gate_row(
            "ordered_phases",
            ["expiry", "full_sequence", "phase_skip"],
            sorted(phase_cases),
            phase_cases == {"full_sequence", "phase_skip", "expiry"}
            and sum(row.get("case") == "full_sequence" for row in phases) == 2
            and all(_row_passes(row) for row in phases),
        ),
        gate_row(
            "checksums",
            ["checksum_mutation", "full_sequence"],
            sorted(checksum_cases),
            checksum_cases == {"full_sequence", "checksum_mutation"}
            and sum(row.get("case") == "full_sequence" for row in checksums) == 2
            and all(_row_passes(row) for row in checksums),
        ),
        gate_row(
            "terminal_release",
            ["full_sequence", "incomplete_release"],
            sorted(release_cases),
            release_cases == {"full_sequence", "incomplete_release"}
            and sum(row.get("case") == "full_sequence" for row in releases) == 2
            and all(_row_passes(row) for row in releases),
        ),
        gate_row(
            "fresh_reread",
            2,
            sum(_row_passes(row) for row in rereads),
            len(rereads) == 2 and all(_row_passes(row) for row in rereads),
        ),
        gate_row(
            "post_audit_preflight",
            ["available", "available"],
            [row.get("classification") for row in post],
            len(post) == 2 and all(row.get("classification") == "available" for row in post),
        ),
        gate_row(
            "signals_sent", [], artifact.get("signals_sent"), artifact.get("signals_sent") == []
        ),
        gate_row(
            "model_load_count",
            0,
            artifact.get("model_load_count"),
            artifact.get("model_load_count") == 0,
        ),
    ]


def _source_hashes(upstream_path: Path) -> JsonDict:
    paths = (
        upstream_path,
        REPO_ROOT / "python/carnot/gpu_lease_phase_journal.py",
        REPO_ROOT / "python/carnot/experiment_7065_v619_three_family_entrance_bank.py",
        REPO_ROOT / "python/carnot/experiment_7079_v620_gpu_lease_audit.py",
        REPO_ROOT / "tests/python/test_experiment_7079_v620_gpu_lease_audit.py",
        REPO_ROOT / "openspec/capabilities/research-harnesses/spec.md",
    )
    rows = [
        {
            "path": str(path.relative_to(REPO_ROOT))
            if path.is_relative_to(REPO_ROOT)
            else str(path),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {"files": rows, "manifest_hash": lease_api.sha256_json(rows)}


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-check schema, evidence projections, verdict, and final checksum."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields_mismatch")
    if set(artifact.get("field_principles", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("model_load_count") != 0:
        errors.append("model_load_count_mismatch")
    if artifact.get("signals_sent") != []:
        errors.append("signals_sent_not_empty")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    expected_rows = build_gate_rows(artifact)
    if artifact.get("rows") != expected_rows:
        errors.append("gate_rows_mismatch")
    expected_score = int(bool(expected_rows) and all(row["passed"] for row in expected_rows))
    if artifact.get("gpu_lease_cold_audit_ready_score") != expected_score:
        errors.append("readiness_score_mismatch")
    post = artifact.get("post_audit_preflight_rows", [])
    if artifact.get("gpu_lease_cold_audit_ready_score") == 1 and not (
        isinstance(post, list)
        and len(post) == 2
        and all(row.get("classification") == "available" for row in post)
    ):
        errors.append("post_audit_preflight_mismatch")
    expected_class = "null" if expected_score == 1 else "blocked"
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if not str(artifact.get("honest_verdict", "")).startswith(expected_class + "_"):
        errors.append("honest_verdict_prefix_mismatch")
    if artifact.get("gate_check_summary") != gate_summary(expected_rows):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    upstream_path: Path = UPSTREAM_PATH,
    expected_upstream_hash: str = EXPECTED_UPSTREAM_SHA256,
    result_path: Path = RESULT_PATH,
    runtime_dir: Path = AUDIT_RUNTIME_DIR,
    gpu_probe: Callable[[], JsonDict] = _default_gpu_probe,
    stop_authority_probe: Callable[[], JsonDict] = _default_stop_authority_probe,
    writable_probe: Callable[[Path], bool] = _writable,
    lease_probe: Callable[[Sequence[Mapping[str, Any]]], list[JsonDict]] = _default_lease_probe,
    process_audit: Callable[[Path, Sequence[Mapping[str, Any]]], JsonDict] = run_process_audit,
    adversarial_probe: Callable[[Path, str], JsonDict] = build_adversarial_rows,
    post_audit_probe: Callable[
        [Sequence[Mapping[str, Any]], Path], list[JsonDict]
    ] = _post_audit_probe,
) -> JsonDict:
    """Run gated process fixtures and publish one terminal cold-audit artifact."""

    started = time.monotonic()
    preconditions = collect_preconditions(
        upstream_path=upstream_path,
        expected_upstream_hash=expected_upstream_hash,
        result_path=result_path,
        audit_runtime_dir=runtime_dir,
        gpu_probe=gpu_probe,
        stop_authority_probe=stop_authority_probe,
        writable_probe=writable_probe,
        lease_probe=lease_probe,
    )
    devices = list(preconditions["gpu_inventory"].get("devices", []))
    evidence: JsonDict = {
        "same_device_race_rows": [],
        "independent_device_rows": [],
        "crash_recovery_rows": [],
        "pid_identity_rows": [],
        "phase_history_rows": [],
        "checksum_rows": [],
        "release_rows": [],
        "fresh_reread_rows": [],
    }
    post_rows: list[JsonDict] = []
    if preconditions["all_passed"]:
        runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_runtime_dir = Path(tempfile.mkdtemp(prefix=".exp7079-run-", dir=runtime_dir))
        process = process_audit(run_runtime_dir, devices)
        adversarial = adversarial_probe(run_runtime_dir / "adversarial", str(devices[0]["uuid"]))
        for key in evidence:
            evidence[key] = [
                *deepcopy(list(process.get(key, []))),
                *deepcopy(list(adversarial.get(key, []))),
            ]
        post_rows = post_audit_probe(devices, run_runtime_dir)
    source_hashes = _source_hashes(upstream_path)
    upstream_hash = preconditions["upstream_hash"]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.monotonic() - started, 6),
        "source_artifact_hashes": source_hashes,
        "cited_upstream_artifacts": [
            {
                "experiment_id": "experiment_7078_v620_gpu_lease_migration",
                "path": str(upstream_path.relative_to(REPO_ROOT))
                if upstream_path.is_relative_to(REPO_ROOT)
                else str(upstream_path),
                "sha256": upstream_hash,
                "fields_imported": ["gpu_lease_compatibility_ready_score"],
            }
        ],
        "upstream_gate_rows": deepcopy(preconditions["upstream_gate_rows"]),
        "rows": [],
        "gpu_topology_rows": deepcopy(preconditions["gpu_topology_rows"]),
        **evidence,
        "post_audit_preflight_rows": deepcopy(post_rows),
        "signals_sent": [],
        "model_load_count": 0,
        "gpu_lease_cold_audit_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_gpu_lease_cold_audit_gate_failed",
    }
    rows = build_gate_rows(artifact)
    score = int(all(row["passed"] for row in rows))
    artifact["rows"] = rows
    artifact["gpu_lease_cold_audit_ready_score"] = score
    artifact["gate_check_summary"] = gate_summary(rows)
    if score == 1:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "null_gpu_lease_cold_audit_ready_no_model_quality_claim"
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_invalid:" + ",".join(errors))
    lease_api.write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in arguments:
        arguments.remove("--worker")
        return worker_main(arguments)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default="20260906")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--runtime-dir", type=Path, default=AUDIT_RUNTIME_DIR)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(arguments)
    if args.validate:
        try:
            artifact = json.loads(args.output.read_text(encoding="utf-8"))
            errors = validate_artifact(artifact)
        except (OSError, json.JSONDecodeError) as exc:
            errors = [f"artifact_unreadable:{type(exc).__name__}"]
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return int(bool(errors))
    artifact = run(date=args.date, result_path=args.output, runtime_dir=args.runtime_dir)
    print(
        json.dumps(
            {
                "gpu_lease_cold_audit_ready_score": artifact["gpu_lease_cold_audit_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
