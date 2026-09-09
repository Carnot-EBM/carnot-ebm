"""Diagnose Qwen3.8 cache, runner, GPU processes, and leases without acting.

The preflight reads live ownership evidence. It does not load model weights,
start a server, acquire a lease, or signal any process. A process name is only
descriptive. It never grants ownership.

Spec refs: REQ-HARNESS-7160 and SCENARIO-HARNESS-7160-*.
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
import re
import shlex
import shutil
import socket
import subprocess
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import canonical_json, command_hash, parse_proc_stat
from carnot.inference.sota_models import resolve_cached_gguf
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
TASK_ID = "experiment_7160_v631_qwen38_lease_diagnosis"
RANDOM_SEED = 7_160_202_609_09
RESULT_PATH = Path("results/experiment_7160_v631_qwen38_lease_diagnosis.json")
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
PREFERRED_QUANT = "Q4_K_M"
LEASE_SCHEMA = lease_api.SCHEMA
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
STOP_AUTHORITY_MARKER = Path.home() / ".carnot" / "stop-authority-armed"
LEGACY_GPU_STATE_PATH = Path("ops/gpu_memory_state.json")
INFERENCE_SUBSTRATE = "read_only_gpu_process_and_cache_diagnosis"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
MIN_IDLE_FREE_MB = 20_000
MAX_IDLE_USED_MB = 1_024
MAX_IDLE_UTILIZATION_PCT = 5

REQUIRED_SOURCE_PATHS = (
    Path("results/experiment_7157_v630_qwen38_runtime.json"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/gpu_lease_phase_journal.py"),
    Path("scripts/gpu_monitor.py"),
    Path("openspec/capabilities/research-harnesses/spec.md"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "gpu_process_rows",
    "lease_ownership_rows",
    "cache_identity_rows",
    "runner_capability_rows",
    "stop_authority_receipt",
    "qwen38_runtime_preflight_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Field reasons distinguish a real ownership receipt from a process-name guess.",
    "status": "A terminal value prevents repeated waiting on unchanged external state.",
    "preconditions_checked": "Named local tools and files make a diagnostic block actionable.",
    "run_date": "The date fixes the process and lease snapshot in time.",
    "inference_substrate": "Use read_only_gpu_process_and_cache_diagnosis because no model weights are opened.",
    "inference_substrate_class": "Use no_model_load, or blocked_no_run when even diagnosis cannot run, so no model-duration floor is implied.",
    "execution_venue": "Host plus GPU inventory identifies where the read-only checks ran.",
    "duration_s": "Measured duration proves the diagnostic ran rather than copying exp7157 prose.",
    "source_artifact_hashes": "Hashes bind the diagnosis to exp7157, registry, supervisor, and specs.",
    "rows": "One row per GPU, process, lease, cache item, and runner check makes readiness reconstructable.",
    "gpu_process_rows": "Per-process PID, age, command, GPU UUID, and memory expose the exact conflict.",
    "lease_ownership_rows": "Lease-to-PID evidence prevents unsafe adoption or teardown.",
    "cache_identity_rows": "Repository, filename, revision, bytes, and hash prevent model substitution.",
    "runner_capability_rows": "Runner checks show whether bounded structured generation is available without loading weights.",
    "stop_authority_receipt": "The marker state explains why the task did not clear an orphan.",
    "qwen38_runtime_preflight_ready_score": "One gates model load; zero is an honest external block, not a quality result.",
    "random_seed": "A fixed row-order seed makes tests reproducible even though diagnosis is deterministic.",
    "reproducibility_checksum": "The checksum detects altered process, lease, cache, or runner evidence.",
    "gate_check_summary": "A blocked result names the first failed check with expected and observed values.",
    "verifier_is_oracle": "False records that this task checks resources, not model correctness.",
    "verdict_class": "Use the closed enum positive | circular_positive | null | blocked | disqualified | partial for structural aggregation.",
    "honest_verdict": "The terminal text reports ready or the exact stable resource block.",
}

_SECRET_OPTION_MARKERS = ("api-key", "apikey", "authorization", "password", "secret", "token")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | Path) -> str:
    """Hash a source file that is not a model weight file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_artifact_hashes(root: Path) -> JsonDict:
    """Bind the diagnosis to its reviewed sources and prior result."""

    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else "missing"
        for path in REQUIRED_SOURCE_PATHS
    }


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep the expected and observed values for one falsifiable gate."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all evidence except the checksum that contains the hash."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def typed_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build a stable flat view for row-oriented consumers."""

    rows: list[JsonDict] = []
    for row in artifact.get("gpu_process_rows", []):
        row_type = "gpu" if row.get("pid") is None else "gpu_process"
        rows.append({"row_type": row_type, **deepcopy(dict(row))})
    for field, row_type in (
        ("lease_ownership_rows", "lease_ownership"),
        ("cache_identity_rows", "cache_identity"),
        ("runner_capability_rows", "runner_capability"),
    ):
        rows.extend(
            {"row_type": row_type, **deepcopy(dict(row))} for row in artifact.get(field, [])
        )
    stop = artifact.get("stop_authority_receipt")
    if isinstance(stop, Mapping) and stop:
        rows.append({"row_type": "stop_authority", **deepcopy(dict(stop))})
    return rows


def base_artifact(run_date: str, *, root: Path | None = None) -> JsonDict:
    """Create every required field before any host resource check."""

    repository = root or Path(__file__).resolve().parents[2]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": {"host": socket.gethostname(), "gpu_uuids": []},
        "duration_s": 0.0,
        "source_artifact_hashes": source_artifact_hashes(repository),
        "rows": [],
        "gpu_process_rows": [],
        "lease_ownership_rows": [],
        "cache_identity_rows": [],
        "runner_capability_rows": [],
        "stop_authority_receipt": {},
        "qwen38_runtime_preflight_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_row("diagnosis_terminal", True, False, False),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "running_qwen38_runtime_preflight",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    """Publish one complete checkpoint with the repository atomic writer."""

    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def redact_command(command: Sequence[str]) -> list[str]:
    """Remove secret values while preserving the remaining argument evidence."""

    redacted: list[str] = []
    hide_next = False
    for raw in command:
        item = str(raw)
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        lowered = item.lower()
        if item.startswith("-") and any(marker in lowered for marker in _SECRET_OPTION_MARKERS):
            if "=" in item:
                redacted.append(item.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(item)
                hide_next = True
            continue
        redacted.append(re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1<redacted>@", item))
    return redacted


def _command_value(command: Sequence[str], flags: set[str]) -> str | None:
    for index, item in enumerate(command):
        if item in flags and index + 1 < len(command):
            return str(command[index + 1])
        for flag in flags:
            if item.startswith(flag + "="):
                return item.split("=", 1)[1]
    return None


def _content_addressed_model_hash(path: Path) -> str | None:
    """Use the cache target name as its hash without reading weight bytes."""

    try:
        real_name = path.resolve(strict=True).name
    except OSError:
        return None
    return "sha256:" + real_name.lower() if _HASH_RE.fullmatch(real_name) else None


def resolve_cache_identity(
    *, resolver: Callable[[str, str], str | None] = resolve_cached_gguf
) -> list[JsonDict]:
    """Resolve only the exact Qwen3.8 file and never open its weight bytes."""

    resolved = resolver(QWEN_MODEL_ID, PREFERRED_QUANT)
    path = Path(resolved) if resolved else None
    exists = bool(path and path.is_file())
    model_hash = _content_addressed_model_hash(path) if path and exists else None
    filename = path.name if path else None
    valid = bool(
        exists
        and filename == QWEN_FILENAME
        and model_hash
        and path is not None
        and "models--unsloth--Qwen3.8-27B-GGUF" in str(path)
    )
    return [
        {
            "repository": QWEN_MODEL_ID,
            "filename": filename,
            "path": str(path) if path else None,
            "real_path": str(path.resolve()) if path and exists else None,
            "revision": snapshot_revision(path) if path and exists else None,
            "bytes": path.stat().st_size if path and exists else None,
            "sha256": model_hash,
            "hash_source": "content_addressed_cache_target" if model_hash else None,
            "weights_opened": False,
            "valid": valid,
        }
    ]


def _progress(phase: int, event: str, **fields: Any) -> None:
    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _run_subprocess(command: Sequence[str], *, phase: int, timeout_s: float = 20.0) -> JsonDict:
    safe_command = redact_command(command)
    _progress(phase, "subprocess_start", command=safe_command)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
        receipt = {
            "command": safe_command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        receipt = {
            "command": safe_command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
        }
    _progress(phase, "subprocess_end", command=safe_command, returncode=receipt["returncode"])
    return receipt


def _listening_ports(pid: int, proc_root: Path) -> list[int]:
    inodes: set[str] = set()
    try:
        for descriptor in (proc_root / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[(\d+)\]", target)
            if match:
                inodes.add(match.group(1))
    except OSError:
        return []
    ports: set[int] = set()
    for table in (proc_root / "net/tcp", proc_root / "net/tcp6"):
        try:
            lines = table.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and fields[9] in inodes:
                ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def _read_proc_identity(pid: int, *, proc_root: Path = Path("/proc")) -> JsonDict:
    _progress(2, "procfs_read_start", pid=pid)
    directory = proc_root / str(pid)
    try:
        stat = parse_proc_stat((directory / "stat").read_text(encoding="utf-8"))
        raw_command = (directory / "cmdline").read_bytes().split(b"\x00")
        command = redact_command([item.decode("utf-8", "replace") for item in raw_command if item])
        uptime_s = float((proc_root / "uptime").read_text(encoding="utf-8").split()[0])
        ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        age_s = max(0.0, uptime_s - int(stat["start_time_ticks"]) / ticks_per_second)
        started_epoch = time.time() - age_s
        ports = _listening_ports(pid, proc_root)
        model_path = _command_value(command, {"--model", "-m"})
        identity = {
            "proc_exists": True,
            "ppid": stat["ppid"],
            "process_group_id": stat["process_group_id"],
            "session_id": stat["session_id"],
            "start_time_ticks": stat["start_time_ticks"],
            "process_start_utc": datetime.fromtimestamp(started_epoch, UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "age_s": round(age_s, 3),
            "command": command,
            "command_text": shlex.join(command),
            "command_sha256": command_hash(command),
            "open_port": ports[0] if len(ports) == 1 else None,
            "open_ports": ports,
            "model_path": model_path,
            "model_sha256": _content_addressed_model_hash(Path(model_path)) if model_path else None,
        }
    except (OSError, ValueError, IndexError) as exc:
        identity = {
            "proc_exists": False,
            "ppid": None,
            "process_group_id": None,
            "session_id": None,
            "start_time_ticks": None,
            "process_start_utc": None,
            "age_s": None,
            "command": [],
            "command_text": "",
            "command_sha256": None,
            "open_port": None,
            "open_ports": [],
            "model_path": None,
            "model_sha256": None,
            "proc_error": f"{type(exc).__name__}: {exc}",
        }
    _progress(2, "procfs_read_end", pid=pid, exists=identity["proc_exists"])
    return identity


def collect_gpu_process_rows() -> tuple[list[JsonDict], list[JsonDict]]:
    """Read both NVIDIA tables and enrich every compute PID from procfs."""

    gpu_receipt = _run_subprocess(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        phase=2,
    )
    process_receipt = _run_subprocess(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        phase=2,
    )
    devices: list[JsonDict] = []
    for line in str(gpu_receipt.get("stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 7:
            devices.append(
                {
                    "gpu_index": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "gpu_name": fields[2],
                    "gpu_utilization_pct": int(float(fields[3])),
                    "gpu_memory_total_mb": int(float(fields[4])),
                    "gpu_memory_used_mb": int(float(fields[5])),
                    "gpu_memory_free_mb": int(float(fields[6])),
                }
            )
    by_uuid = {row["gpu_uuid"]: row for row in devices}
    applications: list[JsonDict] = []
    identities: dict[int, JsonDict] = {}
    for line in str(process_receipt.get("stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4 or not fields[1].isdigit() or fields[0] not in by_uuid:
            continue
        pid = int(fields[1])
        if pid not in identities:
            identities[pid] = _read_proc_identity(pid)
        applications.append(
            {
                **deepcopy(by_uuid[fields[0]]),
                "pid": pid,
                "nvidia_process_name": fields[2],
                "gpu_process_memory_mb": int(float(fields[3])),
                **deepcopy(identities[pid]),
                "ownership_classification": "pending",
                "matching_lease_id": None,
                "ownership_evidence_errors": [],
            }
        )
    rows: list[JsonDict] = []
    for device in devices:
        attached = [row for row in applications if row["gpu_uuid"] == device["gpu_uuid"]]
        if attached:
            rows.extend(attached)
        else:
            rows.append(
                {
                    **deepcopy(device),
                    "pid": None,
                    "gpu_process_memory_mb": 0,
                    "proc_exists": None,
                    "ppid": None,
                    "process_group_id": None,
                    "session_id": None,
                    "start_time_ticks": None,
                    "process_start_utc": None,
                    "age_s": None,
                    "command": [],
                    "command_text": "",
                    "command_sha256": None,
                    "open_port": None,
                    "model_path": None,
                    "model_sha256": None,
                    "ownership_classification": "idle",
                    "matching_lease_id": None,
                    "ownership_evidence_errors": [],
                }
            )
    return rows, [gpu_receipt, process_receipt]


def scan_lease_rows(runtime_dir: Path, process_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Read all canonical journals without locking or changing them."""

    rows: list[JsonDict] = []
    for path in sorted(runtime_dir.glob("device-*.journal.json")):
        _progress(3, "procfs_lease_read_start", path=str(path))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, Mapping):
                raise ValueError("lease_not_object")
            owner = document.get("owner") if isinstance(document.get("owner"), Mapping) else {}
            owner_pid = owner.get("pid")
            owner_ticks = owner.get("pid_start_ticks")
            matching_owner = next(
                (
                    row
                    for row in process_rows
                    if row.get("pid") == owner_pid
                    and row.get("start_time_ticks") == owner_ticks
                    and row.get("proc_exists") is True
                ),
                None,
            )
            server = document.get("server") if isinstance(document.get("server"), Mapping) else {}
            expires = document.get("expires_monotonic_ns")
            validation_errors = lease_api.validate_journal_document(document, check_freshness=False)
            row = {
                "lease_path": str(path),
                "readable": True,
                "canonical": document.get("schema") == LEASE_SCHEMA and not validation_errors,
                "schema": document.get("schema"),
                "checksum_valid": document.get("checksum") == lease_api.journal_checksum(document),
                "lease_id": document.get("lease_id"),
                "task_id": document.get("task_id"),
                "device_uuid": document.get("device_uuid"),
                "owner_pid": owner_pid,
                "owner_start_ticks": owner_ticks,
                "owner_executable": owner.get("executable"),
                "owner_argv_digest": owner.get("argv_digest"),
                "expected_model": document.get("expected_model"),
                "port": document.get("port", server.get("port")),
                "model_sha256": document.get("model_sha256", server.get("model_sha256")),
                "phase": document.get("phase"),
                "released": document.get("released"),
                "expires_monotonic_ns": expires,
                "fresh": bool(
                    isinstance(expires, int)
                    and time.monotonic_ns() <= expires
                    and document.get("released") is not True
                ),
                "owner_live": matching_owner is not None,
                "signals_sent": deepcopy(
                    dict(document.get("recovery") or {}).get("signals_sent", [])
                ),
                "error": ",".join(validation_errors) or None,
            }
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            row = {
                "lease_path": str(path),
                "readable": False,
                "canonical": False,
                "schema": None,
                "checksum_valid": False,
                "lease_id": None,
                "task_id": None,
                "device_uuid": None,
                "owner_pid": None,
                "owner_start_ticks": None,
                "owner_executable": None,
                "owner_argv_digest": None,
                "expected_model": None,
                "port": None,
                "model_sha256": None,
                "phase": None,
                "released": None,
                "expires_monotonic_ns": None,
                "fresh": False,
                "owner_live": False,
                "signals_sent": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        _progress(3, "procfs_lease_read_end", path=str(path), readable=row["readable"])
    return rows


def _lease_match_errors(process: Mapping[str, Any], lease: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    checks = (
        (lease.get("readable") is True, "lease_unreadable"),
        (lease.get("canonical") is True, "lease_not_canonical"),
        (lease.get("checksum_valid") is True, "lease_checksum_invalid"),
        (str(lease.get("lease_id", "")).startswith("lease:"), "lease_id_missing"),
        (lease.get("released") is False, "lease_released"),
        (lease.get("fresh") is True, "lease_not_fresh"),
        (lease.get("owner_live") is True, "lease_owner_not_live"),
        (lease.get("device_uuid") == process.get("gpu_uuid"), "lease_device_mismatch"),
        (lease.get("owner_pid") == process.get("pid"), "lease_owner_pid_mismatch"),
        (
            lease.get("owner_start_ticks") == process.get("start_time_ticks"),
            "lease_owner_start_mismatch",
        ),
        (
            lease.get("expected_model") == process.get("model_path"),
            "lease_expected_model_mismatch",
        ),
        (lease.get("port") is not None, "lease_port_missing"),
        (lease.get("port") == process.get("open_port"), "lease_port_mismatch"),
        (str(lease.get("model_sha256", "")).startswith("sha256:"), "lease_model_hash_missing"),
        (
            lease.get("model_sha256") == process.get("model_sha256"),
            "lease_model_hash_mismatch",
        ),
        (bool(lease.get("task_id")), "lease_owner_task_missing"),
    )
    errors.extend(error for passed, error in checks if not passed)
    return list(dict.fromkeys(errors))


def classify_process_rows(
    process_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    *,
    current_task_id: str,
) -> list[JsonDict]:
    """Classify each PID only from a complete current ownership receipt."""

    classified: list[JsonDict] = []
    for source in process_rows:
        row = deepcopy(dict(source))
        if row.get("pid") is None:
            row.update(
                {
                    "ownership_classification": "idle",
                    "matching_lease_id": None,
                    "ownership_evidence_errors": [],
                }
            )
            classified.append(row)
            continue
        if row.get("proc_exists") is not True:
            row.update(
                {
                    "ownership_classification": "conflicting",
                    "matching_lease_id": None,
                    "ownership_evidence_errors": ["process_identity_missing"],
                }
            )
            classified.append(row)
            continue
        candidates = [
            lease for lease in lease_rows if lease.get("device_uuid") == row.get("gpu_uuid")
        ]
        evaluated = [(lease, _lease_match_errors(row, lease)) for lease in candidates]
        exact = next(((lease, errors) for lease, errors in evaluated if not errors), None)
        if exact is None:
            errors = (
                min((errors for _lease, errors in evaluated), key=len)
                if evaluated
                else ["current_canonical_lease_missing"]
            )
            row.update(
                {
                    "ownership_classification": "conflicting",
                    "matching_lease_id": None,
                    "ownership_evidence_errors": errors,
                }
            )
        else:
            lease, _errors = exact
            row.update(
                {
                    "ownership_classification": (
                        "owned" if lease.get("task_id") == current_task_id else "adoptable"
                    ),
                    "matching_lease_id": lease.get("lease_id"),
                    "ownership_evidence_errors": [],
                }
            )
        classified.append(row)
    return classified


def cache_identity_errors(cache_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rebuild the exact cache decision from recorded metadata."""

    if len(cache_rows) != 1:
        return ["cache_row_count_mismatch"]
    row = cache_rows[0]
    path = Path(str(row.get("path") or ""))
    real_path = Path(str(row.get("real_path") or ""))
    real_name = real_path.name.lower()
    derived_hash = "sha256:" + real_name if _HASH_RE.fullmatch(real_name) else None
    checks = (
        (row.get("repository") == QWEN_MODEL_ID, "cache_repository_mismatch"),
        (row.get("filename") == QWEN_FILENAME, "cache_filename_mismatch"),
        (path.name == QWEN_FILENAME, "cache_path_filename_mismatch"),
        (
            "models--unsloth--Qwen3.8-27B-GGUF" in str(path),
            "cache_repository_path_mismatch",
        ),
        (bool(row.get("revision")), "cache_revision_missing"),
        (isinstance(row.get("bytes"), int) and row["bytes"] > 0, "cache_bytes_invalid"),
        (derived_hash is not None, "cache_content_address_missing"),
        (row.get("sha256") == derived_hash, "cache_hash_mismatch"),
        (
            row.get("hash_source") == "content_addressed_cache_target",
            "cache_hash_source_mismatch",
        ),
        (row.get("weights_opened") is False, "cache_weights_opened"),
    )
    errors = [error for passed, error in checks if not passed]
    if row.get("valid") is not (not errors):
        errors.append("cache_valid_flag_mismatch")
    return errors


def runner_capability_errors(runner_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rebuild runner support from the three bounded command receipts."""

    if len(runner_rows) != 1:
        return ["runner_row_count_mismatch"]
    row = runner_rows[0]
    runner_path = str(row.get("runner_path") or "")
    receipts = list(row.get("command_receipts") or [])
    expected_commands = [
        [runner_path, "--version"],
        [runner_path, "--help"],
        ["ldd", runner_path],
    ]
    commands = [receipt.get("command") for receipt in receipts]
    returncodes = [receipt.get("returncode") for receipt in receipts]
    version_text = ""
    help_text = ""
    linkage_text = ""
    if len(receipts) == 3:
        version_text = str(receipts[0].get("stdout") or receipts[0].get("stderr") or "").strip()
        help_text = f"{receipts[1].get('stdout', '')}\n{receipts[1].get('stderr', '')}".lower()
        linkage_text = f"{receipts[2].get('stdout', '')}\n{receipts[2].get('stderr', '')}".lower()
    bounded = "--n-predict" in help_text
    structured = "--grammar" in help_text or "json-schema" in help_text
    cuda = "libggml-cuda" in linkage_text and "libcuda.so" in linkage_text
    receipt_commands_safe = not any(
        QWEN_FILENAME in str(argument)
        for receipt in receipts
        for argument in receipt.get("command", [])
    )
    checks = (
        (bool(runner_path), "runner_path_missing"),
        (row.get("exists") is True, "runner_missing"),
        (row.get("executable") is True, "runner_not_executable"),
        (len(receipts) == 3, "runner_receipt_count_mismatch"),
        (commands == expected_commands, "runner_receipt_commands_mismatch"),
        (returncodes == [0, 0, 0], "runner_receipt_failed"),
        (bool(version_text), "runner_version_missing"),
        (row.get("version") == version_text, "runner_version_mismatch"),
        (row.get("version_check_ok") is True, "runner_version_check_failed"),
        (row.get("help_check_ok") is True, "runner_help_check_failed"),
        (row.get("cuda_linkage_confirmed") is cuda and cuda, "runner_cuda_mismatch"),
        (
            row.get("task_owned_process_groups") is True,
            "runner_process_group_support_missing",
        ),
        (
            row.get("bounded_token_generation") is bounded and bounded,
            "runner_bounded_generation_missing",
        ),
        (
            row.get("grammar_or_json_output") is structured and structured,
            "runner_structured_output_missing",
        ),
        (row.get("owned_teardown") is True, "runner_owned_teardown_missing"),
        (row.get("model_argument_present") is False, "runner_model_argument_present"),
        (receipt_commands_safe, "runner_model_argument_recorded"),
    )
    errors = [error for passed, error in checks if not passed]
    if row.get("valid") is not (not errors):
        errors.append("runner_valid_flag_mismatch")
    return errors


def readiness_decision(
    process_rows: Sequence[Mapping[str, Any]],
    lease_rows: Sequence[Mapping[str, Any]],
    cache_rows: Sequence[Mapping[str, Any]],
    runner_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce exact cache, runner, process, and lease evidence to readiness."""

    cache_ok = not cache_identity_errors(cache_rows)
    runner_ok = not runner_capability_errors(runner_rows)
    expected_hash = cache_rows[0].get("sha256") if cache_ok else None
    available: list[str] = []
    conflicts: list[JsonDict] = []
    conflicting_leases: list[str] = []
    gpu_uuids = sorted({str(row.get("gpu_uuid")) for row in process_rows if row.get("gpu_uuid")})
    for gpu_uuid in gpu_uuids:
        rows = [row for row in process_rows if row.get("gpu_uuid") == gpu_uuid]
        sample = rows[0]
        processes = [row for row in rows if row.get("pid") is not None]
        for row in processes:
            if row.get("ownership_classification") == "conflicting":
                conflicts.append(
                    {
                        "pid": int(row["pid"]),
                        "gpu_uuid": gpu_uuid,
                        "memory_mb": int(row.get("gpu_process_memory_mb", 0) or 0),
                    }
                )
        matched_ids = {
            str(row.get("matching_lease_id")) for row in processes if row.get("matching_lease_id")
        }
        live_unmatched = [
            lease
            for lease in lease_rows
            if lease.get("device_uuid") == gpu_uuid
            and lease.get("canonical") is True
            and lease.get("released") is False
            and lease.get("fresh") is True
            and lease.get("owner_live") is True
            and str(lease.get("lease_id")) not in matched_ids
        ]
        conflicting_leases.extend(str(lease.get("lease_id")) for lease in live_unmatched)
        is_rtx = "RTX 3090" in str(sample.get("gpu_name", ""))
        low_utilization = (
            int(sample.get("gpu_utilization_pct", 100) or 0) <= MAX_IDLE_UTILIZATION_PCT
        )
        if not processes:
            allocation_clear = (
                int(sample.get("gpu_memory_free_mb", 0) or 0) >= MIN_IDLE_FREE_MB
                and int(sample.get("gpu_memory_used_mb", 0) or 0) <= MAX_IDLE_USED_MB
            )
        else:
            allocation_clear = all(
                row.get("ownership_classification") in {"owned", "adoptable"}
                and row.get("model_sha256") == expected_hash
                for row in processes
            )
        if is_rtx and low_utilization and allocation_clear and not live_unmatched:
            available.append(gpu_uuid)
    failed_check = (
        "cached_qwen38_q4"
        if not cache_ok
        else "cuda_llama_runtime"
        if not runner_ok
        else "idle_rtx_3090"
        if not available
        else None
    )
    return {
        "score": int(cache_ok and runner_ok and bool(available)),
        "cache_ok": cache_ok,
        "runner_ok": runner_ok,
        "available_gpu_uuids": available,
        "conflicting_processes": conflicts,
        "conflicting_lease_ids": sorted(set(conflicting_leases)),
        "failed_check": failed_check,
    }


def collect_runner_capabilities(server_path: Path) -> list[JsonDict]:
    """Probe version, help, and linkage without passing a model argument."""

    version = _run_subprocess([str(server_path), "--version"], phase=5)
    help_receipt = _run_subprocess([str(server_path), "--help"], phase=5)
    linkage = _run_subprocess(["ldd", str(server_path)], phase=5)
    help_text = f"{help_receipt.get('stdout', '')}\n{help_receipt.get('stderr', '')}".lower()
    link_text = f"{linkage.get('stdout', '')}\n{linkage.get('stderr', '')}".lower()
    supervisor_source = (
        Path(__file__)
        .with_name("inference")
        .joinpath("llama_server_supervisor.py")
        .read_text(encoding="utf-8")
    )
    capabilities = {
        "task_owned_process_groups": "start_new_session=True" in supervisor_source,
        "bounded_token_generation": "--n-predict" in help_text,
        "grammar_or_json_output": "--grammar" in help_text or "json-schema" in help_text,
        "owned_teardown": "cleanup_recorded_identity" in supervisor_source,
    }
    cuda = "libggml-cuda" in link_text and "libcuda.so" in link_text
    exists = server_path.is_file()
    executable = exists and os.access(server_path, os.X_OK)
    valid = bool(
        exists
        and executable
        and version.get("returncode") == 0
        and help_receipt.get("returncode") == 0
        and linkage.get("returncode") == 0
        and cuda
        and all(capabilities.values())
    )
    version_text = str(version.get("stdout") or version.get("stderr") or "").strip()
    return [
        {
            "runner_path": str(server_path),
            "exists": exists,
            "executable": executable,
            "version": version_text,
            "version_check_ok": version.get("returncode") == 0,
            "help_check_ok": help_receipt.get("returncode") == 0,
            "cuda_linkage_confirmed": cuda,
            **capabilities,
            "model_argument_present": any(
                QWEN_FILENAME in argument
                for receipt in (version, help_receipt, linkage)
                for argument in receipt.get("command", [])
            ),
            "valid": valid,
            "command_receipts": [version, help_receipt, linkage],
        }
    ]


def stop_authority_receipt(marker: Path = STOP_AUTHORITY_MARKER) -> JsonDict:
    """Report the operator switch without changing it or any process."""

    present = marker.is_file()
    return {
        "marker_path": str(marker),
        "marker_present": present,
        "state": "armed" if present else "disarmed",
        "signals_sent": [],
        "actions_taken": [],
    }


def _resource_gate_rows(decision: Mapping[str, Any]) -> list[JsonDict]:
    return [
        gate_row(
            "cached_qwen38_q4", True, decision.get("cache_ok"), decision.get("cache_ok") is True
        ),
        gate_row(
            "cuda_llama_runtime",
            True,
            decision.get("runner_ok"),
            decision.get("runner_ok") is True,
        ),
        gate_row(
            "idle_rtx_3090",
            {"minimum_count": 1, "no_conflicting_process": True},
            {
                "available_gpu_uuids": deepcopy(decision.get("available_gpu_uuids", [])),
                "conflicting_processes": deepcopy(decision.get("conflicting_processes", [])),
                "conflicting_lease_ids": deepcopy(decision.get("conflicting_lease_ids", [])),
            },
            bool(decision.get("available_gpu_uuids")),
        ),
    ]


def finalize_artifact(
    artifact: Mapping[str, Any],
    *,
    checks: Sequence[Mapping[str, Any]],
    gpu_process_rows: Sequence[Mapping[str, Any]],
    lease_ownership_rows: Sequence[Mapping[str, Any]],
    cache_identity_rows: Sequence[Mapping[str, Any]],
    runner_capability_rows: Sequence[Mapping[str, Any]],
    stop_authority_receipt: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Finish with a ready result or one exact terminal resource block."""

    result = deepcopy(dict(artifact))
    leases = [deepcopy(dict(row)) for row in lease_ownership_rows]
    processes = classify_process_rows(gpu_process_rows, leases, current_task_id=TASK_ID)
    cache = [deepcopy(dict(row)) for row in cache_identity_rows]
    runner = [deepcopy(dict(row)) for row in runner_capability_rows]
    diagnostic_checks = [deepcopy(dict(row)) for row in checks]
    diagnostic_ok = all(row.get("passed") is True for row in diagnostic_checks)
    decision = readiness_decision(processes, leases, cache, runner)
    all_checks = diagnostic_checks + (_resource_gate_rows(decision) if diagnostic_ok else [])
    failed = next((row for row in all_checks if row.get("passed") is not True), None)
    summary = (
        deepcopy(failed) if failed is not None else gate_row("all_readiness_gates", 1, 1, True)
    )
    score = int(diagnostic_ok and decision["score"] == 1)
    result.update(
        {
            "status": "completed" if score else "blocked",
            "preconditions_checked": all_checks,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": (
                INFERENCE_SUBSTRATE_CLASS if diagnostic_ok else "blocked_no_run"
            ),
            "execution_venue": {
                "host": dict(result.get("execution_venue") or {}).get("host", socket.gethostname()),
                "gpu_uuids": sorted(
                    {str(row.get("gpu_uuid")) for row in processes if row.get("gpu_uuid")}
                ),
            },
            "duration_s": round(max(0.0, float(duration_s)), 6),
            "gpu_process_rows": processes,
            "lease_ownership_rows": leases,
            "cache_identity_rows": cache,
            "runner_capability_rows": runner,
            "stop_authority_receipt": deepcopy(dict(stop_authority_receipt)),
            "qwen38_runtime_preflight_ready_score": score,
            "gate_check_summary": summary,
            "verifier_is_oracle": False,
            "verdict_class": "positive" if score else "blocked",
            "honest_verdict": (
                "complete_positive_qwen38_runtime_preflight_ready"
                if score
                else f"blocked_{summary.get('check') or 'diagnostic_dependency'}"
            ),
        }
    )
    result["rows"] = typed_rows(result)
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _load_artifact(value: Mapping[str, Any] | str | Path | object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        try:
            loaded = json.loads(Path(value).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, Mapping) else None
    return None


def validate_artifact(value: Mapping[str, Any] | str | Path | object) -> list[str]:
    """Cold-check schema, ownership reduction, terminal state, rows, and hash."""

    artifact = _load_artifact(value)
    if artifact is None:
        return ["artifact_unreadable_or_not_object"]
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["artifact_fields_mismatch"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("execution_venue"), Mapping):
        errors.append("execution_venue_invalid")
    if float(artifact.get("duration_s", -1) or 0) < 0:
        errors.append("duration_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    stop = artifact.get("stop_authority_receipt")
    if isinstance(stop, Mapping) and (
        stop.get("signals_sent") not in (None, []) or stop.get("actions_taken") not in (None, [])
    ):
        errors.append("read_only_contract_violated")
    leases = list(artifact.get("lease_ownership_rows") or [])
    process_rows = list(artifact.get("gpu_process_rows") or [])
    recomputed_processes = classify_process_rows(process_rows, leases, current_task_id=TASK_ID)
    for observed, recomputed in zip(process_rows, recomputed_processes, strict=True):
        for field in (
            "ownership_classification",
            "matching_lease_id",
            "ownership_evidence_errors",
        ):
            if observed.get(field) != recomputed.get(field):
                errors.append("process_classification_mismatch")
                break
    cache_rows = list(artifact.get("cache_identity_rows") or [])
    runner_rows = list(artifact.get("runner_capability_rows") or [])
    cache_errors = cache_identity_errors(cache_rows)
    runner_errors = runner_capability_errors(runner_rows)
    decision = readiness_decision(
        recomputed_processes,
        leases,
        cache_rows,
        runner_rows,
    )
    checks = list(artifact.get("preconditions_checked") or [])
    resource_names = {"cached_qwen38_q4", "cuda_llama_runtime", "idle_rtx_3090"}
    diagnostic_checks = [row for row in checks if row.get("check") not in resource_names]
    diagnostic_ok = bool(diagnostic_checks) and all(
        row.get("passed") is True for row in diagnostic_checks
    )
    if diagnostic_ok and cache_errors:
        errors.append("cache_identity_mismatch")
    if diagnostic_ok and runner_errors:
        errors.append("runner_capability_mismatch")
    expected_resource_checks = _resource_gate_rows(decision) if diagnostic_ok else []
    observed_resource_checks = [row for row in checks if row.get("check") in resource_names]
    if observed_resource_checks != expected_resource_checks:
        errors.append("resource_gate_rows_mismatch")
    expected_checks = diagnostic_checks + expected_resource_checks
    expected_score = int(diagnostic_ok and decision["score"] == 1)
    if artifact.get("qwen38_runtime_preflight_ready_score") != expected_score:
        errors.append("readiness_score_mismatch")
    failed = next((row for row in expected_checks if row.get("passed") is not True), None)
    expected_summary = deepcopy(failed) if failed else gate_row("all_readiness_gates", 1, 1, True)
    if artifact.get("gate_check_summary") != expected_summary:
        errors.append("gate_check_summary_mismatch")
    expected_status = "completed" if expected_score else "blocked"
    expected_class = "positive" if expected_score else "blocked"
    expected_substrate_class = INFERENCE_SUBSTRATE_CLASS if diagnostic_ok else "blocked_no_run"
    expected_verdict = (
        "complete_positive_qwen38_runtime_preflight_ready"
        if expected_score
        else f"blocked_{expected_summary.get('check') or 'diagnostic_dependency'}"
    )
    if artifact.get("status") != expected_status:
        errors.append("status_mismatch")
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class_mismatch")
    if artifact.get("inference_substrate_class") != expected_substrate_class:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("honest_verdict") != expected_verdict:
        errors.append("honest_verdict_mismatch")
    if artifact.get("rows") != typed_rows(artifact):
        errors.append("typed_rows_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def collect_diagnostic_checks(
    *, root: Path, run_date: str, result_path: Path, lease_dir: Path
) -> list[JsonDict]:
    """Check the tools and paths needed to make a live read-only diagnosis."""

    source_state = {str(path): (root / path).is_file() for path in REQUIRED_SOURCE_PATHS}
    test_path = root / "tests/python/test_experiment_7160_v631_qwen38_lease_diagnosis.py"
    wrapper_path = root / "scripts/experiments/experiment_7160_v631_qwen38_lease_diagnosis.py"
    legacy_state = root / LEGACY_GPU_STATE_PATH
    output_state = {
        "path": str(result_path),
        "exists_after_first_write": result_path.is_file(),
        "parent_writable": os.access(result_path.parent, os.W_OK),
    }
    lease_state = {
        "canonical_lease_dir": str(lease_dir),
        "canonical_lease_dir_exists": lease_dir.is_dir(),
        "legacy_gpu_memory_state_path": str(legacy_state),
        "legacy_gpu_memory_state_exists": legacy_state.is_file(),
        "legacy_gpu_memory_state_required": False,
    }
    return [
        gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE),
        gate_row(
            "source_paths",
            {str(path): True for path in REQUIRED_SOURCE_PATHS},
            source_state,
            all(source_state.values()),
        ),
        gate_row(
            "focused_tests_spec_and_wrapper",
            {"test": True, "spec": True, "wrapper": True},
            {
                "test": test_path.is_file(),
                "spec": (root / "openspec/capabilities/research-harnesses/spec.md").is_file(),
                "wrapper": wrapper_path.is_file(),
            },
            test_path.is_file()
            and wrapper_path.is_file()
            and (root / "openspec/capabilities/research-harnesses/spec.md").is_file(),
        ),
        gate_row(
            "nvidia_smi", "executable", shutil.which("nvidia-smi"), bool(shutil.which("nvidia-smi"))
        ),
        gate_row(
            "procfs",
            {"readable": True},
            {"readable": os.access("/proc", os.R_OK)},
            os.access("/proc", os.R_OK),
        ),
        gate_row(
            "lease_evidence_paths",
            {"canonical_lease_dir_exists": True, "legacy_gpu_memory_state_required": False},
            lease_state,
            lease_dir.is_dir(),
        ),
        gate_row(
            "cache_resolver",
            "callable",
            callable(resolve_cached_gguf),
            callable(resolve_cached_gguf),
        ),
        gate_row(
            "output_path",
            {"path": str(result_path), "exists_after_first_write": True, "parent_writable": True},
            output_state,
            output_state["exists_after_first_write"] and output_state["parent_writable"],
        ),
    ]


def run_experiment(
    *, root: Path, run_date: str, result_path: Path, lease_dir: Path = LEASE_RUNTIME_DIR
) -> JsonDict:
    """Run one read-only host diagnosis and persist one terminal artifact."""

    started = time.perf_counter()
    _progress(0, "phase_start", name="schema_complete_artifact")
    artifact = base_artifact(run_date, root=root)
    _progress(0, "artifact_write_start", path=str(result_path), state="running")
    write_artifact(result_path, artifact)
    _progress(0, "artifact_write_end", path=str(result_path), state="running")
    _progress(0, "phase_end", name="schema_complete_artifact")

    _progress(1, "phase_start", name="diagnostic_dependencies")
    checks = collect_diagnostic_checks(
        root=root, run_date=run_date, result_path=result_path, lease_dir=lease_dir
    )
    _progress(
        1, "phase_end", name="diagnostic_dependencies", passed=all(row["passed"] for row in checks)
    )
    if any(row["passed"] is not True for row in checks):
        result = finalize_artifact(
            artifact,
            checks=checks,
            gpu_process_rows=[],
            lease_ownership_rows=[],
            cache_identity_rows=[],
            runner_capability_rows=[],
            stop_authority_receipt=stop_authority_receipt(),
            duration_s=time.perf_counter() - started,
        )
    else:
        _progress(2, "phase_start", name="gpu_and_procfs_snapshot")
        process_rows, query_receipts = collect_gpu_process_rows()
        gpu_query_ok = all(receipt.get("returncode") == 0 for receipt in query_receipts)
        checks.append(gate_row("gpu_queries", True, gpu_query_ok, gpu_query_ok))
        _progress(2, "phase_end", name="gpu_and_procfs_snapshot", rows=len(process_rows))

        _progress(3, "phase_start", name="canonical_lease_snapshot")
        leases = scan_lease_rows(lease_dir, process_rows)
        _progress(3, "phase_end", name="canonical_lease_snapshot", rows=len(leases))

        _progress(4, "phase_start", name="exact_cache_identity")
        cache = resolve_cache_identity()
        _progress(4, "phase_end", name="exact_cache_identity", valid=cache[0]["valid"])

        _progress(5, "phase_start", name="cuda_runner_capabilities")
        runner = collect_runner_capabilities(resolve_native_llama_server())
        _progress(5, "phase_end", name="cuda_runner_capabilities", valid=runner[0]["valid"])

        _progress(6, "phase_start", name="ownership_and_allocation_reduction")
        result = finalize_artifact(
            artifact,
            checks=checks,
            gpu_process_rows=process_rows,
            lease_ownership_rows=leases,
            cache_identity_rows=cache,
            runner_capability_rows=runner,
            stop_authority_receipt=stop_authority_receipt(),
            duration_s=time.perf_counter() - started,
        )
        _progress(
            6,
            "phase_end",
            name="ownership_and_allocation_reduction",
            readiness=result["qwen38_runtime_preflight_ready_score"],
        )

    _progress(7, "phase_start", name="final_artifact")
    _progress(7, "artifact_write_start", path=str(result_path), state=result["status"])
    write_artifact(result_path, result)
    _progress(7, "artifact_write_end", path=str(result_path), state=result["status"])
    _progress(7, "artifact_validation_start", path=str(result_path))
    errors = validate_artifact(result_path)
    _progress(7, "artifact_validation_end", valid=not errors, errors=errors)
    _progress(7, "phase_end", name="final_artifact", status=result["status"])
    if errors:
        raise ValueError(f"terminal_artifact_invalid:{','.join(errors)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root()
    if args.validate is not None:
        _progress(8, "artifact_validation_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(8, "artifact_validation_end", valid=not errors, errors=errors)
        return int(bool(errors))
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    result = run_experiment(root=root, run_date=args.date, result_path=result_path)
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "qwen38_runtime_preflight_ready_score": result[
                    "qwen38_runtime_preflight_ready_score"
                ],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
