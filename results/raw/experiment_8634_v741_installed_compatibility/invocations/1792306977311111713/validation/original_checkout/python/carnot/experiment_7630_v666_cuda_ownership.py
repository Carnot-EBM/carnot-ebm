"""Qualify CUDA process ownership without loading a model.

The launcher treats every unregistered compute context as foreign. Small memory
use, a familiar command, or ancestry by itself does not grant permission to
share a GPU or signal a process.

Spec refs: REQ-REPORT-7630 and SCENARIO-REPORT-7630-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json, sha256_file


Json = dict[str, Any]
RUN_DATE = "20260924"
MILESTONE = "2026.09.666"
EXPERIMENT_ID = "experiment_7630_v666_cuda_ownership"
SCHEMA = "carnot.exp7630.v666.cuda_ownership.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_7630_v666_cuda_ownership.json")
RAW_DIR = Path("results/raw/experiment_7630_v666_cuda_ownership")
LAUNCH_CONFIG_PATH = RAW_DIR / "launch_config.json"
MODULE_PATH = Path("python/carnot/experiment_7630_v666_cuda_ownership.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7630_v666_cuda_ownership.py")
TEST_PATH = Path("tests/python/test_experiment_7630_v666_cuda_ownership.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
HISTORICAL_RESULT = Path("results/experiment_7617_v665_schema_pilot.json")
SCHEMA_RESULT = Path("results/experiment_7616_v665_evidence_schema.json")
GPU_REQUIRED_FREE_MB = 20_000
LEASE_WAIT_S = 180.0
LEASE_PROGRESS_S = 15.0
MODEL_SPECS: list[str] = []
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
PLANNED_INFERENCE_SUBSTRATE_CLASS = "no_model_load"


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print a flushed boundary so a bounded local wait stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7630] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for independent reduction and custody checks."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Json:
    """Read one JSON object and reject every other top-level shape."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def historical_inventory(artifact: Mapping[str, Any]) -> list[Json]:
    """Recover the exact inventory that drove the old Exp7617 selector."""

    checks = artifact.get("preconditions_checked") or []
    for check in checks:
        if isinstance(check, Mapping) and check.get("check") == "exclusive_cuda_capacity":
            observed = check.get("observed")
            if isinstance(observed, list):
                return [deepcopy(dict(row)) for row in observed if isinstance(row, Mapping)]
    return []


def old_selector(inventory: Sequence[Mapping[str, Any]], *, current_pid: int) -> Json | None:
    """Reproduce Exp7617's selector exactly for the regression fixture."""

    candidates = [
        deepcopy(dict(row))
        for row in inventory
        if int(row.get("memory_free_mb") or 0) >= GPU_REQUIRED_FREE_MB
        and all(
            int(process.get("pid") or -1) == current_pid
            for process in row.get("processes") or []
            if isinstance(process, Mapping)
        )
    ]
    return min(candidates, key=lambda row: int(row["index"])) if candidates else None


def _proc_ancestry(pid: int) -> list[int]:  # pragma: no cover - live Linux process boundary.
    ancestry: list[int] = []
    current = pid
    for _ in range(128):
        if current <= 0 or current in ancestry:
            break
        ancestry.append(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
            parent = next(line for line in status.splitlines() if line.startswith("PPid:"))
            current = int(parent.split()[1])
        except (OSError, StopIteration, ValueError, IndexError):
            break
    return ancestry


def process_identity(pid: int) -> Json | None:  # pragma: no cover - live Linux process boundary.
    """Read enough Linux identity to distinguish a live PID from PID reuse."""

    start_ticks = lease_api.proc_start_ticks(pid)
    if start_ticks is None:
        return None
    try:
        command_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        command = command_bytes.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        boot_line = next(
            line
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
            if line.startswith("btime ")
        )
        boot_seconds = int(boot_line.split()[1])
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
        started = datetime.fromtimestamp(boot_seconds + start_ticks / clock_ticks, tz=UTC)
    except (OSError, StopIteration, ValueError, IndexError):
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "start_time": started.isoformat().replace("+00:00", "Z"),
        "command": command,
        "ancestry": _proc_ancestry(pid),
    }


class ProcessRegistry:
    """Register only child identities created by the current task."""

    def __init__(self, *, task_id: str, owner_pid: int, owner_start_ticks: int) -> None:
        self.task_id = task_id
        self.owner_pid = owner_pid
        self.owner_start_ticks = owner_start_ticks
        self._children: dict[int, int] = {}

    @classmethod
    def current(cls) -> ProcessRegistry:
        """Create a registry bound to this exact Linux process identity."""

        pid = os.getpid()
        ticks = lease_api.proc_start_ticks(pid)
        if ticks is None:
            raise RuntimeError("current_process_start_time_unavailable")
        return cls(task_id=EXPERIMENT_ID, owner_pid=pid, owner_start_ticks=ticks)

    def register(self, *, pid: int, start_ticks: int, ancestry: Sequence[int]) -> None:
        """Register a matching descendant; ancestry without registration is insufficient."""

        if pid == self.owner_pid or self.owner_pid not in ancestry[1:]:
            raise ValueError("not_current_task_descendant")
        self._children[int(pid)] = int(start_ticks)

    def classify(self, *, pid: int, start_ticks: int, ancestry: Sequence[int]) -> Json:
        """Explain whether one identity is an exact registered task child."""

        if pid == self.owner_pid:
            reason = "owner_is_not_registered_child"
        elif pid in self._children and self._children[pid] != start_ticks:
            reason = "pid_reuse_start_time_mismatch"
        elif pid not in self._children:
            reason = (
                "descendant_not_registered"
                if self.owner_pid in ancestry[1:]
                else "unrelated_process"
            )
        elif self.owner_pid not in ancestry[1:]:
            reason = "registered_identity_not_descendant"
        else:
            reason = "registered_current_task_descendant"
        accepted = reason == "registered_current_task_descendant"
        return {
            "task_id": self.task_id,
            "pid": pid,
            "start_ticks": start_ticks,
            "ancestry": list(ancestry),
            "registered_ownership": accepted,
            "rejection_reason": None if accepted else reason,
        }


def enrich_inventory_process_identities(inventory: Sequence[Mapping[str, Any]]) -> list[Json]:
    """Attach current process evidence without changing GPU state."""

    enriched: list[Json] = []
    for device in inventory:
        copied = deepcopy(dict(device))
        processes: list[Json] = []
        for process in device.get("processes") or []:
            if not isinstance(process, Mapping):
                continue
            row = deepcopy(dict(process))
            identity = process_identity(int(row.get("pid") or -1))
            if identity is not None:
                row.update(identity)
            processes.append(row)
        copied["processes"] = processes
        enriched.append(copied)
    return enriched


def select_owned_capacity(
    inventory: Sequence[Mapping[str, Any]], registry: ProcessRegistry
) -> tuple[Json | None, list[Json]]:
    """Select capacity only when every compute context has registered ownership."""

    candidates: list[Json] = []
    ownership_rows: list[Json] = []
    for device in inventory:
        device_rows: list[Json] = []
        for process in device.get("processes") or []:
            if not isinstance(process, Mapping):
                continue
            identity_ready = all(
                key in process
                for key in ("pid", "start_ticks", "ancestry", "command", "start_time")
            )
            if identity_ready:
                decision = registry.classify(
                    pid=int(process["pid"]),
                    start_ticks=int(process["start_ticks"]),
                    ancestry=[int(value) for value in process["ancestry"]],
                )
            else:
                decision = {
                    "task_id": registry.task_id,
                    "pid": int(process.get("pid") or -1),
                    "start_ticks": process.get("start_ticks"),
                    "ancestry": list(process.get("ancestry") or []),
                    "registered_ownership": False,
                    "rejection_reason": "process_identity_unavailable",
                }
            row = {
                "device_uuid": device.get("uuid"),
                "pid": int(process.get("pid") or -1),
                "start_time": process.get("start_time"),
                "command": process.get("command") or process.get("name"),
                "used_memory_mb": int(process.get("used_memory_mb") or 0),
                **decision,
            }
            device_rows.append(row)
            ownership_rows.append(row)
        enough_memory = int(device.get("memory_free_mb") or 0) >= GPU_REQUIRED_FREE_MB
        if enough_memory and all(row["registered_ownership"] is True for row in device_rows):
            candidates.append(deepcopy(dict(device)))
    selected = min(candidates, key=lambda row: int(row["index"])) if candidates else None
    return selected, ownership_rows


def cpu_preflight_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the CPU preflight environment before numerical packages import."""

    environment = dict(base or os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    return environment


def tokenizer_qualification_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Describe the later CPU-only embedded-tokenizer child contract."""

    environment = cpu_preflight_environment(base)
    environment["CARNOT_TOKENIZER_N_GPU_LAYERS"] = "0"
    return environment


def model_worker_environment(
    physical_gpu_uuid: str, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Expose only one selected physical UUID to a future model child."""

    if not physical_gpu_uuid.startswith("GPU-"):
        raise ValueError("physical_gpu_uuid_required")
    environment = cpu_preflight_environment(base)
    environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_uuid
    return environment


def _device_by_uuid(inventory: Sequence[Mapping[str, Any]], uuid: str) -> Json | None:
    return next(
        (deepcopy(dict(row)) for row in inventory if str(row.get("uuid")) == uuid),
        None,
    )


def recheck_before_launch(
    selected_uuid: str,
    snapshots: Sequence[Sequence[Mapping[str, Any]]],
    registry: ProcessRegistry,
) -> Json:
    """Require safe inventory after lease acquisition and immediately before launch."""

    if len(snapshots) != 2:
        raise ValueError("two_inventory_rechecks_required")
    labels = ("after_lease_acquisition", "immediately_before_model_launch")
    observations: list[Json] = []
    for label, inventory in zip(labels, snapshots, strict=True):
        device = _device_by_uuid(inventory, selected_uuid)
        if device is None:
            return {
                "passed": False,
                "reason": "selected_device_missing",
                "observations": observations,
            }
        if device.get("oom_observed") is True:
            return {"passed": False, "reason": "oom_observed", "observations": observations}
        if int(device.get("memory_free_mb") or 0) < GPU_REQUIRED_FREE_MB:
            return {
                "passed": False,
                "reason": "free_memory_below_floor",
                "observations": observations,
            }
        selected, process_rows = select_owned_capacity([device], registry)
        observation = {
            "checkpoint": label,
            "device": device,
            "process_ownership_rows": process_rows,
            "selected": selected is not None,
        }
        observations.append(observation)
        if selected is None:
            return {
                "passed": False,
                "reason": "foreign_allocation_after_lease",
                "observations": observations,
            }
    return {"passed": True, "reason": "both_rechecks_passed", "observations": observations}


def acquire_cooperative_lease(
    *,
    runtime_dir: Path,
    device_uuid: str,
    vram_before_mb: int,
    max_wait_s: float = LEASE_WAIT_S,
    poll_s: float = LEASE_PROGRESS_S,
    progress_fn: Callable[[Json], None] | None = None,
) -> lease_api.GpuLease:
    """Acquire one shipped kernel-backed lease with a finite visible wait."""

    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            return lease_api.GpuLease.acquire(
                runtime_dir=runtime_dir,
                task_id=EXPERIMENT_ID,
                device_uuid=device_uuid,
                expected_model="no-model/protocol-qualification",
                vram_before_mb=vram_before_mb,
                ttl_s=max(30.0, max_wait_s + 15.0),
            )
        except (lease_api.LeaseBusy, lease_api.RecoveryError):
            elapsed = time.monotonic() - started
            if progress_fn is not None:
                progress_fn(
                    {
                        "event": "lease_wait",
                        "attempt": attempts,
                        "elapsed_s": elapsed,
                        "device_uuid": device_uuid,
                    }
                )
            if elapsed >= max_wait_s:
                raise lease_api.LeaseBusy(f"lease_wait_timeout:{device_uuid}") from None
            time.sleep(max(0.0, min(poll_s, max_wait_s - elapsed)))


def _terminal_release(lease: lease_api.GpuLease) -> Json:
    """End a no-model qualification lease without inventing unload evidence."""

    lease.transition("terminal_blocked")
    return lease.release()


def _stop_owned_child(process: subprocess.Popen[str]) -> list[Json]:
    """Signal only the exact child object created by this task."""

    signals: list[Json] = []
    if process.poll() is None:
        process.terminate()
        signals.append({"pid": process.pid, "signal": "SIGTERM", "owned": True})
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        signals.append({"pid": process.pid, "signal": "SIGKILL", "owned": True})
        process.wait(timeout=3.0)
    return signals


def exercise_lease_race(runtime_dir: Path) -> Json:
    """Start two synchronized lease attempts and require exactly one winner."""

    start = threading.Barrier(2)
    finish = threading.Barrier(2)
    outcomes: list[Json] = []
    lock = threading.Lock()

    def contender(index: int) -> None:
        lease: lease_api.GpuLease | None = None
        start.wait(timeout=5.0)
        try:
            lease = acquire_cooperative_lease(
                runtime_dir=runtime_dir,
                device_uuid="GPU-fake-race",
                vram_before_mb=0,
                max_wait_s=0.0,
                poll_s=0.0,
            )
            row = {"launcher": index, "outcome": "acquired", "lease_id": lease.lease_id}
        except lease_api.LeaseBusy:
            row = {"launcher": index, "outcome": "busy", "lease_id": None}
        with lock:
            outcomes.append(row)
        finish.wait(timeout=5.0)
        if lease is not None:
            _terminal_release(lease)

    threads = [threading.Thread(target=contender, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    winner_count = sum(row["outcome"] == "acquired" for row in outcomes)
    busy_count = sum(row["outcome"] == "busy" for row in outcomes)
    return {
        "case": "two_launchers_racing",
        "winner_count": winner_count,
        "busy_count": busy_count,
        "outcomes": sorted(outcomes, key=lambda row: row["launcher"]),
        "passed": len(outcomes) == 2 and winner_count == 1 and busy_count == 1,
        "foreign_signal_count": 0,
    }


def exercise_stale_lease_recovery(runtime_dir: Path) -> Json:
    """Recover an abandoned journal only after its exact owner has exited."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603 - fixed interpreter and this module's bounded worker.
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--abandon-lease-dir",
            str(runtime_dir),
        ],
        env=cpu_preflight_environment(),
        capture_output=True,
        text=True,
        timeout=10.0,
        check=True,
    )
    lease = acquire_cooperative_lease(
        runtime_dir=runtime_dir,
        device_uuid="GPU-fake-stale",
        vram_before_mb=0,
        max_wait_s=1.0,
        poll_s=0.01,
    )
    owner = lease.owner_receipt()
    release = _terminal_release(lease)
    return {
        "case": "stale_lease",
        "passed": owner["recovery"]["performed"] is True and release["released"] is True,
        "recovery_performed": owner["recovery"]["performed"],
        "previous_pid": owner["recovery"].get("previous_pid"),
        "signals_sent": owner["recovery"]["signals_sent"],
        "foreign_signal_count": 0,
    }


def _sleeping_child(environment: Mapping[str, str] | None = None) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603 - fixed interpreter and constant child body.
        [
            sys.executable,
            "-u",
            "-c",
            "import json, os, time; print(json.dumps(dict(os.environ)), flush=True); time.sleep(30)",
        ],
        env=dict(environment or os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def exercise_interrupted_cleanup(runtime_dir: Path) -> Json:
    """Simulate interruption and release only the task child and its lease."""

    child = _sleeping_child(cpu_preflight_environment())
    lease = acquire_cooperative_lease(
        runtime_dir=runtime_dir,
        device_uuid="GPU-fake-interrupt",
        vram_before_mb=0,
        max_wait_s=1.0,
        poll_s=0.01,
    )
    signals: list[Json] = []
    release: Json = {"released": False}
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        signals = _stop_owned_child(child)
        release = _terminal_release(lease)
    return {
        "case": "interrupted_cleanup",
        "passed": child.poll() is not None and release["released"] is True,
        "owned_child_reaped": child.poll() is not None,
        "owned_signals": signals,
        "lease_released": release["released"],
        "foreign_signal_count": 0,
    }


def _wait_identity(pid: int) -> Json:
    """Wait briefly for a task-created child to expose its Linux identity."""

    for _ in range(100):
        identity = process_identity(pid)
        if identity is not None:
            return identity
        time.sleep(0.01)
    raise RuntimeError("owned_child_identity_unavailable")


def exercise_fake_launch_e2e(runtime_dir: Path) -> Json:
    """Drive fake inventory through lease, isolated child, rechecks, and cleanup."""

    registry = ProcessRegistry.current()
    clean_device = {
        "index": 0,
        "uuid": "GPU-fake-e2e",
        "name": "fake device",
        "memory_total_mb": 24_576,
        "memory_used_mb": 0,
        "memory_free_mb": 24_576,
        "utilization_pct": 0,
        "processes": [],
    }
    selected, _ = select_owned_capacity([clean_device], registry)
    if selected is None:
        raise RuntimeError("fake_device_selection_failed")
    environment = model_worker_environment(str(selected["uuid"]))
    child = _sleeping_child(environment)
    lease: lease_api.GpuLease | None = None
    release: Json = {"released": False}
    signals: list[Json] = []
    recheck: Json = {"passed": False, "reason": "not_started"}
    child_environment: Json = {}
    try:
        assert child.stdout is not None
        child_environment = json.loads(child.stdout.readline())
        identity = _wait_identity(child.pid)
        registry.register(
            pid=child.pid,
            start_ticks=int(identity["start_ticks"]),
            ancestry=identity["ancestry"],
        )
        process_row = {
            "pid": child.pid,
            "name": "owned fake model worker",
            "used_memory_mb": 256,
            **identity,
        }
        owned_device = deepcopy(clean_device)
        owned_device.update(memory_used_mb=256, memory_free_mb=24_320, processes=[process_row])
        lease = acquire_cooperative_lease(
            runtime_dir=runtime_dir / "lease",
            device_uuid=str(selected["uuid"]),
            vram_before_mb=0,
            max_wait_s=1.0,
            poll_s=0.01,
        )
        recheck = recheck_before_launch(
            str(selected["uuid"]), [[owned_device], [owned_device]], registry
        )
    finally:
        signals = _stop_owned_child(child)
        if lease is not None:
            release = _terminal_release(lease)
    return {
        "case": "fake_inventory_lease_owned_child_cleanup",
        "passed": bool(
            recheck.get("passed") is True
            and child_environment.get("CUDA_VISIBLE_DEVICES") == "GPU-fake-e2e"
            and child.poll() is not None
            and release.get("released") is True
        ),
        "selected_uuid": selected["uuid"],
        "child_pid": child.pid,
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": child_environment.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": child_environment.get("JAX_PLATFORMS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": child_environment.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "recheck": recheck,
        "owned_child_reaped": child.poll() is not None,
        "owned_signals": signals,
        "lease_released": release.get("released") is True,
        "foreign_signal_count": 0,
    }


def _unit_row(case: str, passed: bool, provenance: Mapping[str, Any]) -> Json:
    """Keep one independent fixture unit with absolute accounting operands."""

    return {
        "unit_id": case,
        "arm": "ownership_protocol",
        "absolute_pass_count": int(passed),
        "numerator": int(passed),
        "denominator": 1,
        "seed": None,
        "direction": "pass_requires_fail_closed_or_exact_owned_success",
        "censored": False,
        "disposition": "complete",
        "passed": bool(passed),
        "raw_provenance": deepcopy(dict(provenance)),
    }


def fixture_rows(private_root: Path) -> list[Json]:
    """Run every independent protocol fixture without a model or physical GPU."""

    historical = historical_inventory(load_json(REPO_ROOT / HISTORICAL_RESULT))
    historical_selected = old_selector(historical, current_pid=os.getpid())
    historical_owned, historical_processes = select_owned_capacity(
        historical, ProcessRegistry.current()
    )
    historical_case = {
        "old_selected": historical_selected,
        "ownership_selected": historical_owned,
        "inventory": historical,
        "process_ownership_rows": historical_processes,
    }
    foreign = {
        "pid": 700,
        "name": "python",
        "used_memory_mb": 256,
        "start_ticks": 70,
        "start_time": "fixture",
        "command": "unrelated process",
        "ancestry": [700, 1],
    }
    foreign_selected, foreign_rows = select_owned_capacity(
        [
            {
                "index": 0,
                "uuid": "GPU-fake-foreign",
                "memory_free_mb": 23_912,
                "processes": [foreign],
            }
        ],
        ProcessRegistry.current(),
    )
    foreign_case = {"selected": foreign_selected, "process_ownership_rows": foreign_rows}
    registry = ProcessRegistry(task_id="pid-reuse-fixture", owner_pid=100, owner_start_ticks=10)
    registry.register(pid=200, start_ticks=20, ancestry=[200, 100])
    reuse = registry.classify(pid=200, start_ticks=21, ancestry=[200, 100])
    low_device = {
        "index": 0,
        "uuid": "GPU-fake-low-memory",
        "memory_free_mb": GPU_REQUIRED_FREE_MB - 1,
        "processes": [],
    }
    low_selected, _ = select_owned_capacity([low_device], ProcessRegistry.current())
    race = exercise_lease_race(private_root / "race")
    stale = exercise_stale_lease_recovery(private_root / "stale")
    cleanup = exercise_interrupted_cleanup(private_root / "cleanup")
    e2e = exercise_fake_launch_e2e(private_root / "e2e")
    return [
        _unit_row(
            "historical_256_mib_unknown",
            historical_selected is None and historical_owned is None,
            historical_case,
        ),
        _unit_row("foreign_256_mib_context", foreign_selected is None, foreign_case),
        _unit_row(
            "pid_reuse",
            reuse["rejection_reason"] == "pid_reuse_start_time_mismatch",
            reuse,
        ),
        _unit_row(
            "free_memory_floor",
            low_selected is None,
            {"selected": low_selected, "device": low_device, "floor_mb": GPU_REQUIRED_FREE_MB},
        ),
        _unit_row("two_launchers_racing", bool(race["passed"]), race),
        _unit_row("stale_lock_recovery", bool(stale["passed"]), stale),
        _unit_row("interrupted_cleanup", bool(cleanup["passed"]), cleanup),
        _unit_row("owned_child_launch_e2e", bool(e2e["passed"]), e2e),
    ]


def _source_hash(artifact: Mapping[str, Any], suffix: str) -> str | None:
    for row in artifact.get("source_artifact_hashes") or []:
        if isinstance(row, Mapping) and str(row.get("path") or "").endswith(suffix):
            value = row.get("sha256")
            return str(value) if value is not None else None
    return None


def _count_jsonl(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def replay_exp7616_authority(root: Path, *, schema_path: Path | None = None) -> Json:
    """Recompute role, schema, and lifecycle custody from exact source bytes."""

    from carnot import experiment_7616_v665_evidence_schema as schema_protocol

    artifact_path = root / SCHEMA_RESULT
    artifact = load_json(artifact_path)
    checksum = schema_protocol.reproducibility_checksum(artifact) if artifact else None
    artifact_checksum_ok = checksum == artifact.get("reproducibility_checksum")

    actual_schema = (
        schema_path
        or root / RAW_DIR.parent / "experiment_7616_v665_evidence_schema/schema_authority.json"
    )
    expected_schema_hash = _source_hash(artifact, "schema_authority.json")
    schema_hash = sha256_file(actual_schema) if actual_schema.is_file() else None
    schema_ready = bool(
        artifact_checksum_ok
        and artifact.get("evidence_schema_ready_score") == 1
        and schema_hash == expected_schema_hash
    )

    role = artifact.get("role_contract_receipt")
    role = role if isinstance(role, Mapping) else {}
    role_path = root / str(role.get("role_manifest_path") or "")
    role_hash = sha256_file(role_path) if role_path.is_file() else None
    role_rows: list[Json] = []
    for sidecar in role.get("sidecars") or []:
        if not isinstance(sidecar, Mapping):
            continue
        path = root / str(sidecar.get("path") or "")
        observed = sidecar.get("observed")
        observed = observed if isinstance(observed, Mapping) else {}
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_rows = _count_jsonl(path) if path.is_file() else None
        role_rows.append(
            {
                "path": str(sidecar.get("path")),
                "expected_sha256": observed.get("sha256"),
                "observed_sha256": actual_hash,
                "expected_rows": observed.get("rows"),
                "observed_rows": actual_rows,
                "passed": actual_hash == observed.get("sha256")
                and actual_rows == observed.get("rows"),
            }
        )
    role_ready = bool(
        artifact_checksum_ok
        and artifact.get("role_contract_ready_score") == 1
        and role.get("ready") is True
        and role_hash == role.get("role_manifest_sha256")
        and role_rows
        and all(row["passed"] for row in role_rows)
    )

    lifecycle = artifact.get("guarded_update_lifecycle_receipt")
    lifecycle = lifecycle if isinstance(lifecycle, Mapping) else {}
    lifecycle_path = root / str(lifecycle.get("upstream_path") or "")
    lifecycle_hash = sha256_file(lifecycle_path) if lifecycle_path.is_file() else None
    lifecycle_ready = bool(
        artifact_checksum_ok
        and artifact.get("guarded_update_ready_score") == 1
        and lifecycle.get("authenticated") is True
        and lifecycle.get("checksum_authenticated") is True
        and lifecycle.get("configuration_matches") is True
        and lifecycle.get("restart_parity") is True
        and lifecycle_hash == lifecycle.get("upstream_sha256")
    )
    return {
        "artifact_path": SCHEMA_RESULT.as_posix(),
        "artifact_sha256": sha256_file(artifact_path) if artifact_path.is_file() else None,
        "artifact_checksum_authenticated": artifact_checksum_ok,
        "schema_path": str(actual_schema),
        "schema_expected_sha256": expected_schema_hash,
        "schema_observed_sha256": schema_hash,
        "schema_authority_ready_score": int(schema_ready),
        "role_manifest_path": str(role.get("role_manifest_path") or ""),
        "role_manifest_expected_sha256": role.get("role_manifest_sha256"),
        "role_manifest_observed_sha256": role_hash,
        "role_sidecar_rows": role_rows,
        "role_contract_ready_score": int(role_ready),
        "guarded_update_path": str(lifecycle.get("upstream_path") or ""),
        "guarded_update_expected_sha256": lifecycle.get("upstream_sha256"),
        "guarded_update_observed_sha256": lifecycle_hash,
        "guarded_update_ready_score": int(lifecycle_ready),
    }


def write_launch_config(path: Path) -> Json:
    """Freeze the future launcher contract without making a resource promise."""

    value: Json = {
        "schema": "carnot.exp7630.launch_config.v1",
        "gpu_required_free_mb": GPU_REQUIRED_FREE_MB,
        "foreign_compute_veto": True,
        "lease": {
            "acquisition_timeout_s": LEASE_WAIT_S,
            "progress_interval_s": LEASE_PROGRESS_S,
            "owner_identity": ["pid", "pid_start_ticks"],
            "recheck_points": ["after_lease_acquisition", "immediately_before_model_launch"],
        },
        "cpu_preflight": {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "before_numerical_imports": True,
        },
        "tokenizer_qualification": {
            "spawned_child": True,
            "CUDA_VISIBLE_DEVICES": "",
            "n_gpu_layers": 0,
            "execute_in_this_task": False,
        },
        "model_worker": {
            "CUDA_VISIBLE_DEVICES": "selected_physical_gpu_uuid",
            "physical_uuid_only": True,
            "before_numerical_imports": True,
        },
        "persistent_resource_ready_promise": False,
    }
    atomic_json(path, value)
    return value


def gate_row(
    check: str,
    *,
    category: str,
    upstream: str,
    path: str,
    field: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str = "Unavailable or unauthenticated evidence fails closed.",
) -> Json:
    """Retain complete operands so an operator can reproduce one decision."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "condition": f"{field} {operator} expected value",
        "governing_principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> Json:
    """Expose every failure and the first row with all diagnostic operands."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        "first_failure": failures[0] if failures else None,
    }


def independent_reduce_rows(rows: Sequence[Mapping[str, Any]]) -> Json:
    """Reduce raw unit operands without trusting a proposer readiness score."""

    unit_ids = [str(row.get("unit_id") or "") for row in rows]
    complete = [
        row for row in rows if row.get("disposition") == "complete" and row.get("censored") is False
    ]
    passed = [row for row in complete if row.get("passed") is True]
    errors: list[str] = []
    if not rows:
        errors.append("qualification_rows_missing")
    if len(unit_ids) != len(set(unit_ids)):
        errors.append("duplicate_unit_id")
    for row in rows:
        if row.get("denominator") != 1 or row.get("numerator") not in {0, 1}:
            errors.append(f"invalid_absolute_operands:{row.get('unit_id')}")
        if row.get("numerator") != int(row.get("passed") is True):
            errors.append(f"pass_operand_mismatch:{row.get('unit_id')}")
    return {
        "intended_units": len(rows),
        "observed_units": len(complete),
        "passed_units": len(passed),
        "failed_units": len(complete) - len(passed),
        "censored_units": len(rows) - len(complete),
        "unit_ids": unit_ids,
        "protocol_passed": bool(rows) and len(passed) == len(rows) and not errors,
        "errors": errors,
    }


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind the terminal record while excluding its self-referential digest."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _acceptance_gates(*, validity: bool, readiness: bool, capacity: bool) -> list[Json]:
    principle = "Infrastructure validity and readiness do not establish learned benefit."
    return [
        gate_row(
            "authenticated_inputs_and_authority",
            category="validity",
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH.as_posix(),
            field="authority_and_fixture_validity",
            operator="eq",
            expected=True,
            observed=validity,
            passed=validity,
            principle=principle,
        ),
        gate_row(
            "launch_protocol_qualification",
            category="readiness",
            upstream=EXPERIMENT_ID,
            path=LAUNCH_CONFIG_PATH.as_posix(),
            field="launch_protocol_ready_score",
            operator="eq",
            expected=1,
            observed=int(readiness),
            passed=readiness,
            principle="Readiness covers ownership, races, isolation, rechecks, and cleanup only.",
        ),
        gate_row(
            "probability_benefit",
            category="probability_benefit",
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH.as_posix(),
            field="empirical_probability_benefit_measured",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            principle=principle,
        ),
        gate_row(
            "utility_benefit",
            category="utility",
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH.as_posix(),
            field="utility_benefit_measured",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            principle=principle,
        ),
        gate_row(
            "retention_benefit",
            category="retention",
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH.as_posix(),
            field="retention_benefit_measured",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            principle=principle,
        ),
        gate_row(
            "fresh_confirmatory_evidence",
            category="freshness",
            upstream=EXPERIMENT_ID,
            path=RESULT_PATH.as_posix(),
            field="fresh_confirmatory_claim_allowed",
            operator="eq",
            expected=True,
            observed=False,
            passed=False,
            principle=principle,
        ),
        gate_row(
            "current_capacity_snapshot",
            category="readiness",
            upstream="nvidia-smi_read_only",
            path=RESULT_PATH.as_posix(),
            field="current_capacity_available",
            operator="eq",
            expected=True,
            observed=capacity,
            passed=capacity,
            principle="Capacity is a snapshot and every future model task must reacquire and recheck.",
        ),
    ]


def _field_principles(keys: Sequence[str]) -> Json:
    """Give every terminal field an interpretation rule in the same artifact."""

    specific = {
        "honest_verdict": "Completion names finished work; it does not imply benefit.",
        "verdict_class": "Use only the closed verdict enum; protocol readiness is null.",
        "flagged_adversarial": "A critical terminal reader finding opens no downstream gate.",
        "gate_check_summary": "Blocked work names the first exact failed operand and every failed check.",
        "acceptance_gate_results": "Validity, readiness, probability, utility, retention, and freshness stay separate.",
        "rows": "Each independent unit retains absolute operands and raw provenance.",
        "sample_size_budget": "Repeated views, orders, and seeds never multiply independent units.",
        "preconditions_checked": "Only actual input and resource observations count as checked.",
        "inference_substrate": "Current fake-process work cannot inherit historical model activity.",
        "inference_substrate_class": "No model load or generation occurred in this task.",
        "MODEL_SPECS": "No-call work uses an empty actual model list.",
        "model_invoked": "Historical rows do not count as a current invocation.",
        "execution_venue": "The host venue keeps fake UUIDs separate from read-only physical inventory.",
        "phase_spans": "Measured disjoint stages retain completed and pending unit positions.",
        "invocation_counts": "Current loads, forwards, generations, and tokens are independently zero.",
        "duration_s": "Use measured monotonic elapsed time and never pad it.",
        "random_seed": "Deterministic process fixtures require no stochastic seed.",
        "reproducibility_checksum": "Bind immutable inputs, configuration, rows, and reduction code.",
        "source_artifact_hashes": "Inputs, current sidecars, and planned outputs have distinct classes.",
        "validation_receipts": "Only real command exits and exact log hashes establish validation.",
        "verifier_is_oracle": "Exact control truth cannot establish oracle-distinct learned advantage.",
        "launch_protocol_ready_score": "One qualifies ownership, races, isolation, rechecks, and cleanup only.",
        "role_contract_ready_score": "One requires the Exp7616 role manifest and every sidecar byte.",
        "guarded_update_ready_score": "One requires the existing lifecycle bytes and checksum custody.",
        "current_capacity_available": "This read-only snapshot is not a reservation or future promise.",
        "process_ownership_rows": "PID, start time, ancestry, and registration all precede authorization.",
        "launch_config_path": "The config freezes a future protocol and reserves no device.",
    }
    all_keys = tuple(dict.fromkeys((*keys, "field_principles")))
    return {
        key: specific.get(key, "Retain this field as literal current-task evidence.")
        for key in all_keys
    }


def _process_rows(rows: Sequence[Mapping[str, Any]]) -> list[Json]:
    result: list[Json] = []
    for row in rows:
        provenance = row.get("raw_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        for process in provenance.get("process_ownership_rows") or []:
            if isinstance(process, Mapping):
                result.append(deepcopy(dict(process)))
        recheck = provenance.get("recheck")
        recheck = recheck if isinstance(recheck, Mapping) else {}
        for observation in recheck.get("observations") or []:
            if not isinstance(observation, Mapping):
                continue
            for process in observation.get("process_ownership_rows") or []:
                if isinstance(process, Mapping):
                    result.append(deepcopy(dict(process)))
    return result


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    current_inventory: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    duration_s: float,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> Json:
    """Build a complete no-model protocol record from raw independent units."""

    copied_rows = [deepcopy(dict(row)) for row in rows]
    reduction = independent_reduce_rows(copied_rows)
    authority_ready = all(
        authority.get(field) == 1
        for field in (
            "schema_authority_ready_score",
            "role_contract_ready_score",
            "guarded_update_ready_score",
        )
    )
    preconditions_ready = all(row.get("passed") is True for row in preconditions)
    validity = bool(authority_ready and preconditions_ready and reduction["protocol_passed"])
    launch_ready = bool(validity)
    registry = ProcessRegistry.current()
    current_selected, current_process_rows = select_owned_capacity(current_inventory, registry)
    capacity = current_selected is not None
    acceptance = _acceptance_gates(
        validity=validity,
        readiness=launch_ready,
        capacity=capacity,
    )
    claim_gates = [row for row in acceptance if row["category"] != "readiness"]
    invocation_counts = {
        **deepcopy(ZERO_INVOCATION_COUNTS),
        "forward_passes_attempted": 0,
        "forward_passes_completed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    artifact: Json = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7630,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "title": "CUDA process ownership and isolated preflight qualification",
        "honest_verdict": (
            "complete_null_cuda_ownership_protocol_ready"
            if launch_ready
            else "complete_null_cuda_ownership_protocol_not_ready"
        ),
        "verdict_class": "null",
        "flagged_adversarial": False,
        "gate_check_summary": gate_summary(
            [row for row in acceptance if row["category"] in {"validity", "readiness"}]
        ),
        "acceptance_gate_results": acceptance,
        "rows": copied_rows,
        "independent_reduction": reduction,
        "sample_size_budget": {
            "intended_independent_units": len(copied_rows),
            "observed_independent_units": reduction["observed_units"],
            "excluded_independent_units": 0,
            "censored_independent_units": reduction["censored_units"],
            "repeated_orders_views_or_seeds_added_to_sample_size": 0,
        },
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": "host_cpu_process_identity_fake_gpu_inventory_and_kernel_lease",
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "planned_inference_substrate_class": PLANNED_INFERENCE_SUBSTRATE_CLASS,
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": MODEL_SPECS,
        "target_model": "not_applicable_no_model_load",
        "planned_MODEL_SPECS": MODEL_SPECS,
        "historical_models": ["unsloth/Qwen3.8-27B-GGUF"],
        "model_invoked": False,
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": os.uname().nodename,
            "owner_pid": os.getpid(),
            "physical_gpu_inventory_read_only": [deepcopy(dict(row)) for row in current_inventory],
            "fixture_device_uuids": sorted(
                {
                    str(value)
                    for row in copied_rows
                    for value in [
                        (row.get("raw_provenance") or {}).get("selected_uuid")
                        if isinstance(row.get("raw_provenance"), Mapping)
                        else None
                    ]
                    if value
                }
            ),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "invocation_counts": invocation_counts,
        "duration_s": float(duration_s),
        "random_seed": {"value": None, "purpose": "deterministic process and lock fixtures"},
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": {},
        "verifier_is_oracle": True,
        "empirical_probability_benefit_measured": False,
        "utility_benefit_measured": False,
        "retention_benefit_measured": False,
        "fresh_confirmatory_claim_allowed": False,
        "positive_claim": False,
        "launch_protocol_ready_score": int(launch_ready),
        "evidence_schema_ready_score": int(authority.get("schema_authority_ready_score") == 1),
        "role_contract_ready_score": int(authority.get("role_contract_ready_score") == 1),
        "guarded_update_ready_score": int(authority.get("guarded_update_ready_score") == 1),
        "current_capacity_available": capacity,
        "current_capacity_selected_uuid": current_selected.get("uuid")
        if current_selected
        else None,
        "persistent_resource_ready_promise": False,
        "process_ownership_rows": [*_process_rows(copied_rows), *current_process_rows],
        "launch_config_path": LAUNCH_CONFIG_PATH.as_posix(),
        "exp7616_authority_replay": deepcopy(dict(authority)),
        "claimed_benefit_gate_passed": all(row["passed"] for row in claim_gates),
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_training_authorized": False,
        "default_promotion_authorized": False,
        "research_roadmap_modified": False,
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = None
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    failed_checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> Json:
    """Build a complete external-input block without fabricated owned work."""

    artifact = build_artifact(
        rows=[],
        authority={},
        current_inventory=[],
        preconditions=failed_checks,
        source_hashes=[],
        phase_spans=[],
        duration_s=duration_s,
    )
    first = next((row for row in failed_checks if row.get("passed") is not True), None)
    reason = str(first.get("check") if isinstance(first, Mapping) else "precondition")
    artifact.update(
        {
            "honest_verdict": f"complete_blocked_{reason}",
            "verdict_class": "blocked",
            "gate_check_summary": gate_summary(failed_checks),
            "launch_protocol_ready_score": 0,
            "current_capacity_available": False,
        }
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> Json:
    """Cold-check identity, no-model truth, raw rows, readiness, and checksum."""

    required = {
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "sample_size_budget",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "planned_inference_substrate_class",
        "MODEL_SPECS",
        "model_invoked",
        "execution_venue",
        "phase_spans",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "source_artifact_hashes",
        "validation_receipts",
        "verifier_is_oracle",
        "field_principles",
        "launch_protocol_ready_score",
        "role_contract_ready_score",
        "guarded_update_ready_score",
        "current_capacity_available",
        "process_ownership_rows",
        "launch_config_path",
    }
    errors = [f"missing_field:{field}" for field in sorted(required - set(value))]
    rows = value.get("rows")
    rows = rows if isinstance(rows, list) else []
    reduction = independent_reduce_rows(rows)
    blocked = value.get("verdict_class") == "blocked"
    if not str(value.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_prefix_missing")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if value.get("inference_substrate_class") != "no_model_load":
        errors.append("wrong_substrate_class")
    if (
        value.get("MODEL_SPECS") != []
        or value.get("model_specs") != []
        or value.get("model_invoked") is not False
    ):
        errors.append("model_activity_claimed")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(int(count or 0) != 0 for count in counts.values()):
        errors.append("nonzero_current_invocation")
    expected_ready = int(
        value.get("verdict_class") == "null"
        and reduction["protocol_passed"]
        and value.get("role_contract_ready_score") == 1
        and value.get("guarded_update_ready_score") == 1
        and value.get("evidence_schema_ready_score") == 1
        and all(row.get("passed") is True for row in value.get("preconditions_checked") or [])
    )
    if value.get("launch_protocol_ready_score") != expected_ready:
        errors.append("launch_protocol_score_mismatch")
    if blocked:
        summary = value.get("gate_check_summary")
        first = summary.get("first_failure") if isinstance(summary, Mapping) else None
        diagnostic = {"check", "upstream", "path", "field", "operator", "expected", "observed"}
        if not isinstance(first, Mapping) or not diagnostic.issubset(first):
            errors.append("blocked_gate_diagnostics_incomplete")
    if (
        value.get("current_capacity_available") is True
        and value.get("persistent_resource_ready_promise") is not False
    ):
        errors.append("capacity_promised_persistently")
    principles = value.get("field_principles")
    if not isinstance(principles, Mapping) or any(key not in principles for key in value):
        errors.append("field_principle_missing")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return {"valid": not errors, "errors": errors, "independent_reduction": reduction}


def cold_replay(path: Path) -> Json:
    """Reload exact candidate bytes and validate them without model work."""

    value = load_json(path)
    if not value:
        return {"valid": False, "errors": ["candidate_unavailable"], "independent_reduction": {}}
    return validate_artifact(value)


def independent_reduce_artifact(path: Path) -> Json:
    """Recompute protocol counts from persisted rows only."""

    value = load_json(path)
    rows = value.get("rows") if isinstance(value, Mapping) else None
    return independent_reduce_rows(rows if isinstance(rows, list) else [])


def _source_row(path: Path, *, producer: str, source_class: str) -> Json:
    return {
        "path": str(path.resolve()),
        "producer": producer,
        "source_class": source_class,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def collect_preconditions(root: Path) -> tuple[list[Json], list[Json]]:
    """Authenticate named inputs without treating any planned output as input."""

    named = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7617_v665_schema_pilot.py"),
        Path("python/carnot/experiment_7581_v662_arc_bounded_canary.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("scripts/gpu_executor.py"),
        HISTORICAL_RESULT,
        SCHEMA_RESULT,
        SPEC_PATH,
    )
    checks: list[Json] = []
    hashes: list[Json] = []
    for relative in named:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"named_input:{relative.name}",
                category="validity",
                upstream="declared_task_input",
                path=str(path),
                field="readable_nonempty_file",
                operator="eq",
                expected=True,
                observed=present,
                passed=present,
            )
        )
        if present:
            hashes.append(
                _source_row(path, producer="declared_task_input", source_class="pre_gate_input")
            )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_row(
            "driving_requirement",
            category="validity",
            upstream="research_reporting_spec",
            path=str(root / SPEC_PATH),
            field="REQ-*",
            operator="contains",
            expected="REQ-REPORT-7630",
            observed="REQ-REPORT-7630" if "REQ-REPORT-7630" in spec_text else None,
            passed="REQ-REPORT-7630" in spec_text,
        )
    )
    return checks, hashes


def affected_validation_manifest() -> Json:
    """Freeze the only files admitted to current changed-behavior validation."""

    return {
        "requirement": "REQ-REPORT-7630",
        "tests": [TEST_PATH.as_posix()],
        "changed_modules": [MODULE_PATH.as_posix()],
        "static_paths": [WRAPPER_PATH.as_posix()],
    }


def build_validation_commands(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build serial scoped tests, coverage, Ruff, mypy, and specification checks."""

    manifest = affected_validation_manifest()
    return validation_scope.build_scoped_commands(
        root,
        manifest["tests"],
        manifest["changed_modules"],
        static_paths=manifest["static_paths"],
        basetemp=private_root / "pytest",
        coverage_file=private_root / ".coverage.exp7630",
    )


def terminal_commands(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Declare fresh reducers and both strict readers for one candidate path."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--cold-replay", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--independent-reduce", str(candidate)),
            "persisted_rows_only",
            180.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_terminal_candidate",
            180.0,
        ),
    ]


def _commands_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is False
        for row in receipts
    )


def _phase_span(
    phase: str,
    *,
    task_started: float,
    phase_started: float,
    planned: int,
    completed: int,
    pending: str | None = None,
) -> Json:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_offset_s": phase_started - task_started,
        "ended_offset_s": ended - task_started,
        "duration_s": ended - phase_started,
        "planned_units": planned,
        "completed_units": completed,
        "pending_operation": pending,
        "checkpoint_position": completed,
    }


def _current_inventory() -> list[Json]:  # pragma: no cover - read-only host boundary.
    from carnot.experiment_7581_v662_arc_bounded_canary import gpu_inventory

    return enrich_inventory_process_identities(gpu_inventory())


def _authority_checks(root: Path, authority: Mapping[str, Any]) -> list[Json]:
    fields = (
        "schema_authority_ready_score",
        "role_contract_ready_score",
        "guarded_update_ready_score",
    )
    return [
        gate_row(
            f"exp7616_{field}",
            category="validity",
            upstream="exp7616_terminal_independent_replay",
            path=str(root / SCHEMA_RESULT),
            field=field,
            operator="eq",
            expected=1,
            observed=authority.get(field),
            passed=authority.get(field) == 1,
        )
        for field in fields
    ]


def _reader_outcomes(receipts: Sequence[Mapping[str, Any]]) -> Json:
    return {
        str(row.get("name")): {
            "passed": row.get("passed") is True,
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
    }


def _refresh_artifact(artifact: Json) -> None:
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)


def run_experiment(root: Path, run_date: str, output: Path) -> int:  # pragma: no cover
    """Run the no-model qualification and publish only after exact readers pass."""

    task_started = time.monotonic()
    phase_spans: list[Json] = []
    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7630-"))
    progress(task_started, "preconditions", "before")
    phase_started = time.monotonic()
    checks, source_hashes = collect_preconditions(root)
    authority = replay_exp7616_authority(root)
    checks.extend(_authority_checks(root, authority))
    phase_spans.append(
        _phase_span(
            "preconditions",
            task_started=task_started,
            phase_started=phase_started,
            planned=len(checks),
            completed=sum(row["passed"] is True for row in checks),
        )
    )
    progress(task_started, "preconditions", "after", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(checks, duration_s=time.monotonic() - task_started)
        artifact["run_date"] = run_date
        artifact["source_artifact_hashes"] = source_hashes
        artifact["phase_spans"] = phase_spans
        _refresh_artifact(artifact)
        atomic_json(output, artifact)
        return 0

    progress(task_started, "launch_config", "before")
    config = write_launch_config(root / LAUNCH_CONFIG_PATH)
    source_hashes.append(
        _source_row(
            root / LAUNCH_CONFIG_PATH,
            producer=EXPERIMENT_ID,
            source_class="current_frozen_output",
        )
    )
    progress(task_started, "launch_config", "after", config_sha256=canonical_hash(config))

    progress(task_started, "protocol_fixtures", "before")
    phase_started = time.monotonic()
    rows = fixture_rows(private_root / "fixtures")
    phase_spans.append(
        _phase_span(
            "protocol_fixtures",
            task_started=task_started,
            phase_started=phase_started,
            planned=len(rows),
            completed=sum(row["passed"] is True for row in rows),
        )
    )
    progress(task_started, "protocol_fixtures", "after", completed=len(rows))

    progress(task_started, "current_capacity_snapshot", "before")
    phase_started = time.monotonic()
    current_inventory = _current_inventory()
    phase_spans.append(
        _phase_span(
            "current_capacity_snapshot",
            task_started=task_started,
            phase_started=phase_started,
            planned=1,
            completed=1,
        )
    )
    progress(task_started, "current_capacity_snapshot", "after", devices=len(current_inventory))

    artifact = build_artifact(
        rows=rows,
        authority=authority,
        current_inventory=current_inventory,
        preconditions=checks,
        source_hashes=source_hashes,
        phase_spans=phase_spans,
        duration_s=time.monotonic() - task_started,
    )
    artifact["run_date"] = run_date
    _refresh_artifact(artifact)

    progress(task_started, "affected_validation", "before")
    phase_started = time.monotonic()
    (private_root / "pytest").mkdir(parents=True, exist_ok=True)
    affected = validation_scope.run_commands(
        root,
        build_validation_commands(root, private_root),
        log_dir=private_root / "logs/affected",
        extra_env={"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        heartbeat_s=60.0,
    )
    validation_passed = _commands_passed(affected)
    phase_spans.append(
        _phase_span(
            "affected_validation",
            task_started=task_started,
            phase_started=phase_started,
            planned=len(affected),
            completed=sum(row["passed"] is True for row in affected),
        )
    )
    artifact["phase_spans"] = phase_spans
    artifact["validation_receipts"] = [deepcopy(dict(row)) for row in affected]
    artifact["acceptance_gate_results"].append(
        gate_row(
            "affected_validation",
            category="validity",
            upstream=EXPERIMENT_ID,
            path=affected_validation_manifest()["tests"][0],
            field="all_scoped_commands_passed",
            operator="eq",
            expected=True,
            observed=validation_passed,
            passed=validation_passed,
        )
    )
    if not validation_passed:
        artifact["honest_verdict"] = "complete_disqualified_affected_validation"
        artifact["verdict_class"] = "disqualified"
        artifact["launch_protocol_ready_score"] = 0
    artifact["duration_s"] = time.monotonic() - task_started
    _refresh_artifact(artifact)
    progress(task_started, "affected_validation", "after", passed=validation_passed)

    candidate = private_root / "terminal_candidate.json"
    atomic_json(candidate, artifact)
    progress(task_started, "terminal_readers", "before")
    first_readers = validation_scope.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=private_root / "logs/terminal",
        extra_env={"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        heartbeat_s=60.0,
    )
    artifact["validation_receipts"].extend(deepcopy(dict(row)) for row in first_readers)
    artifact["terminal_reader_outcomes"] = _reader_outcomes(first_readers)
    adversarial = next(
        (row for row in first_readers if row.get("name") == "adversarial_verify"), {}
    )
    artifact["flagged_adversarial"] = adversarial.get("passed") is not True
    artifact["duration_s"] = time.monotonic() - task_started
    _refresh_artifact(artifact)
    atomic_json(candidate, artifact)

    progress(task_started, "exact_terminal_reread", "before")
    exact_readers = validation_scope.run_commands(
        root,
        terminal_commands(root, candidate),
        log_dir=private_root / "logs/terminal_exact",
        extra_env={"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        heartbeat_s=60.0,
    )
    readers_passed = _commands_passed(first_readers) and _commands_passed(exact_readers)
    progress(task_started, "exact_terminal_reread", "after", passed=readers_passed)
    if not validation_passed or not readers_passed:
        return 1
    atomic_json(output, load_json(candidate))
    progress(task_started, "publication", "after", output=output)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse production and read-only modes without importing numerical packages."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--fixture-worker", type=Path)
    parser.add_argument("--abandon-lease-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or one fresh read-only candidate operation."""

    started = time.monotonic()
    progress(started, "startup", "before")
    args = parse_args(argv)
    if args.abandon_lease_dir is not None:  # pragma: no cover - spawned stale-owner worker.
        lease_api.GpuLease.acquire(
            runtime_dir=args.abandon_lease_dir,
            task_id=EXPERIMENT_ID,
            device_uuid="GPU-fake-stale",
            expected_model="no-model/protocol-qualification",
            vram_before_mb=0,
            ttl_s=30.0,
        )
        os._exit(0)
    if args.fixture_worker is not None:
        atomic_json(
            args.fixture_worker,
            {
                "pid": os.getpid(),
                "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        progress(started, "fixture_worker", "after")
        return 0
    if args.cold_replay is not None:
        result = cold_replay(args.cold_replay)
        print(json.dumps(result, sort_keys=True), flush=True)
        progress(started, "cold_replay", "after", valid=result["valid"])
        return 0 if result["valid"] else 1
    if args.independent_reduce is not None:
        result = independent_reduce_artifact(args.independent_reduce)
        print(json.dumps(result, sort_keys=True), flush=True)
        progress(started, "independent_reduce", "after", passed=result["protocol_passed"])
        return 0 if result["protocol_passed"] else 1
    root = args.repo_root.resolve()
    if root != REPO_ROOT.resolve():
        raise ValueError(f"repository_root_mismatch:{root}")
    return run_experiment(root, str(args.date), (root / args.output).resolve())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
