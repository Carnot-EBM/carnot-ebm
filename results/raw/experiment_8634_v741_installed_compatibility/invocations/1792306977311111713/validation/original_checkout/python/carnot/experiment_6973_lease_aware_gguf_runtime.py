"""Run a lease-aware sequential CUDA handoff across three local GGUF families.

The controller owns both device leases. Each model runs in a fresh child and
holds a private readiness socket. The next family starts only after the child,
socket, VRAM, and lease journals all reach a verified terminal state.

Spec refs: REQ-INFRA-6973, SCENARIO-INFRA-6973-LEASE,
SCENARIO-INFRA-6973-STALE, SCENARIO-INFRA-6973-SERVER,
SCENARIO-INFRA-6973-TEARDOWN, and SCENARIO-INFRA-6973-BARE-READINESS.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6966_gguf_load_envelope_canary import (
    build_vram_release_row,
    canonical_json,
    embedded_tokenizer_probe,
    gpu_inventory,
    llama_cpp_probe,
    parse_offloaded_layers,
    sha256_text,
)
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import (
    capture_process_lineage,
    read_process_identity,
    sha256_file,
    write_json_atomic,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "carnot.experiment_6973_lease_aware_gguf_runtime"
EXPERIMENT_ID = "experiment_6973_lease_aware_gguf_runtime"
SCHEMA = "carnot.experiment_6973.lease_aware_gguf_runtime.v1"
RESULT_PATH = REPO_ROOT / "results/experiment_6973_lease_aware_gguf_runtime.json"
CHECKPOINT_PATH = (
    REPO_ROOT / "results/checkpoints/experiment_6973_lease_aware_gguf_runtime/rows.json"
)
LEASE_RUNTIME_DIR = Path(os.environ.get("CARNOT_GPU_LEASE_RUNTIME_DIR", "/tmp/carnot-gpu-leases"))
RUN_DATE = "20260904"
RANDOM_SEED = 6_973_202_609_04
PREFERRED_QUANT = "Q4_K_M"
INFERENCE_SUBSTRATE = "live_local_llama_cpp_three_family_lease_owned_cuda"
FIXED_PROMPT = (
    "State one reason why an owned model process must release GPU memory before "
    "the next model starts."
)
MAX_OUTPUT_TOKENS = 32
MODEL_TIMEOUT_S = 1_800.0
LEASE_TTL_S = MODEL_TIMEOUT_S + 300.0
VRAM_RELEASE_TIMEOUT_S = 180.0
VRAM_RELEASE_TOLERANCE_MB = 512
POLL_INTERVAL_S = 0.25
READY_TIMEOUT_S = 20.0

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

LOAD_CONFIG: JsonDict = {
    "config_id": "lease_owned_dual_cuda_ctx16384",
    "n_ctx": 16_384,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ubatch": 512,
    "main_gpu": 0,
    "split_mode": "layer",
    "tensor_split": [0.5, 0.5],
    "use_mmap": True,
    "use_mlock": False,
    "visible_devices": [0, 1],
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "tokenizer_source": "embedded_gguf",
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "live_duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "gpu_topology",
    "lease_rows",
    "process_ownership_rows",
    "baseline_gpu_memory_rows",
    "load_config_rows",
    "live_generation_rows",
    "gpu_runtime_rows",
    "teardown_rows",
    "vram_release_rows",
    "checkpoint_rows",
    "runtime_handoff_complete_score",
    "lease_aware_runtime_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema lets a verifier reject incompatible evidence.",
    "experiment_id": "A stable identifier binds the artifact to this task.",
    "run_date": "The execution date prevents silent protocol drift.",
    "field_principles": "A reason for every field makes the evidence contract auditable.",
    "preconditions_checked": "Measured gates prevent unsafe inference after resource drift.",
    "inference_substrate": "The exact substrate excludes CPU, remote, and unowned work.",
    "duration_s": "Total wall time exposes truncated or synthetic execution.",
    "live_duration_s": "Generation time separates live inference from setup work.",
    "source_artifact_hashes": "Source hashes bind the result to prior evidence and code.",
    "MODEL_SPECS": "Exact declarations prevent silent model-family substitution.",
    "models_used": "Ordered family IDs make complete model coverage falsifiable.",
    "model_file_hashes": "File hashes bind each result to exact cached model bytes.",
    "gpu_topology": "Device UUIDs and capacity identify the tested hardware.",
    "lease_rows": "Owner-bound journals prove exclusive authority without sending signals.",
    "process_ownership_rows": "PID lineage prevents foreign output from becoming task evidence.",
    "baseline_gpu_memory_rows": "Pre-load memory makes later recovery measurable.",
    "load_config_rows": "Frozen context and offload settings prevent a CPU fallback.",
    "live_generation_rows": "One terminal row per family preserves failures without pooling.",
    "gpu_runtime_rows": "PID-linked samples and layer counts prove CUDA execution.",
    "teardown_rows": "Exit and port receipts prove only the owned worker ended.",
    "vram_release_rows": "Both devices must recover before a sequential handoff.",
    "checkpoint_rows": "Atomic checkpoints retain completed family evidence after interruption.",
    "runtime_handoff_complete_score": "Completion requires one terminal row for each exact family.",
    "lease_aware_runtime_ready_score": "Readiness requires CUDA output and a clean owned release for all families.",
    "random_seed": "A fixed seed makes the bounded completion repeatable.",
    "reproducibility_checksum": "A content hash detects later artifact mutation.",
    "gate_check_summary": "Expected and observed values make a blocked gate actionable.",
    "verifier_is_oracle": "False states that runtime evidence does not judge answer correctness.",
    "verdict_class": "A closed class keeps terminal automation states unambiguous.",
    "honest_verdict": "A class-consistent prefix reports the measured terminal state.",
}


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the mandated pair first, then add the missing exact family."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        path = paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": path,
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": (
                    "cached_sota_pair(gpu_indices=(0, 1))"
                    if model_id in paths
                    else "resolve_cached_gguf exact family extension"
                ),
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject substitutions, non-primary GGUFs, and single-device placement."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        path = str(row.get("model_path", ""))
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif Path(path).suffix.lower() != ".gguf" or "mmproj" in Path(path).name.lower():
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"dual_gpu_indices_missing:{model_id}")
        if row.get("headline_eligible") is not True:
            errors.append(f"headline_eligibility_missing:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:
    """Run one bounded host probe and preserve its exact terminal state."""

    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def checkpoint_is_writable(path: Path) -> bool:
    """Test the actual checkpoint directory with an atomic-write-compatible file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".write-probe-", dir=path.parent)
        os.close(descriptor)
        Path(name).unlink()
        return True
    except OSError:
        return False


def classify_lease_document(
    document: Mapping[str, Any],
    process_match: Callable[[int, int], bool] = lease_api.process_start_matches,
) -> JsonDict:
    """Classify a journal without treating a dead owner as a live lease."""

    if document.get("released") is True:
        return {"classification": "released", "owner_live": False, "foreign_live": False}
    owner = document.get("owner")
    if not isinstance(owner, Mapping):
        return {"classification": "invalid", "owner_live": False, "foreign_live": False}
    pid = owner.get("pid")
    start_ticks = owner.get("pid_start_ticks")
    if not isinstance(pid, int) or not isinstance(start_ticks, int):
        return {"classification": "invalid", "owner_live": False, "foreign_live": False}
    live = bool(process_match(pid, start_ticks))
    return {
        "classification": "live_foreign" if live else "stale_recoverable",
        "owner_live": live,
        "foreign_live": live,
    }


def lease_ledger_snapshot(
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path = LEASE_RUNTIME_DIR,
    *,
    reader: Callable[[Path], JsonDict] = lease_api.read_journal,
    process_match: Callable[[int, int], bool] = lease_api.process_start_matches,
) -> list[JsonDict]:
    """Read one durable journal per device and retain unreadable failures."""

    rows: list[JsonDict] = []
    for device in devices:
        device_uuid = str(device.get("uuid", ""))
        path = lease_api.journal_path_for(runtime_dir, device_uuid)
        try:
            document = reader(path)
            classification = classify_lease_document(document, process_match)
            rows.append(
                {
                    "device_uuid": device_uuid,
                    "journal_path": str(path),
                    "readable": True,
                    **classification,
                    "signals_sent": [],
                    "document": deepcopy(document),
                }
            )
        except Exception as exc:  # noqa: BLE001 - an unreadable journal is evidence.
            rows.append(
                {
                    "device_uuid": device_uuid,
                    "journal_path": str(path),
                    "readable": False,
                    "classification": "unreadable",
                    "owner_live": False,
                    "foreign_live": False,
                    "signals_sent": [],
                    "document": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def _check(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def collect_preconditions(
    *,
    model_specs: Sequence[Mapping[str, Any]],
    checkpoint_path: Path,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
    llama_probe: Callable[[], JsonDict] = llama_cpp_probe,
    writable_probe: Callable[[Path], bool] = checkpoint_is_writable,
    ledger_probe: Callable[[Sequence[Mapping[str, Any]]], list[JsonDict]] | None = None,
    metadata_probe: Callable[[Mapping[str, Any]], JsonDict] = embedded_tokenizer_probe,
) -> JsonDict:
    """Check hardware, files, bindings, journals, tokenizers, and checkpoint."""

    gpu = gpu_probe()
    devices = list(gpu.get("devices", []))
    processes = list(gpu.get("processes", []))
    binding = llama_probe()
    ledgers = ledger_probe(devices) if ledger_probe is not None else lease_ledger_snapshot(devices)
    metadata_rows = [metadata_probe(model) for model in model_specs]
    spec_errors = model_spec_errors(model_specs)
    checkpoint_writable = writable_probe(checkpoint_path)
    files = {
        str(row.get("hf_id")): Path(str(row.get("model_path", ""))).is_file() for row in model_specs
    }
    ledger_ok = len(ledgers) == 2 and all(
        row.get("readable") is True and row.get("foreign_live") is not True for row in ledgers
    )
    checks = [
        _check(
            "nvidia_device_count",
            2,
            len(devices),
            gpu.get("query_ok") is True
            and len(devices) == 2
            and all("NVIDIA" in str(row.get("name", "")) for row in devices),
        ),
        _check("foreign_gpu_compute_processes", [], processes, not processes),
        _check("exact_model_specs", [], spec_errors, not spec_errors),
        _check(
            "all_three_cached_gguf_files",
            {model_id: True for model_id in REQUIRED_MODEL_IDS},
            files,
            len(files) == 3 and all(files.values()),
        ),
        _check(
            "llama_cpp_cuda_bindings",
            {"importable": True, "gpu_offload": True},
            binding,
            binding.get("importable") is True and binding.get("gpu_offload") is True,
        ),
        _check(
            "lease_ledger_available_without_live_foreign_owner",
            {"readable_count": 2, "foreign_live_count": 0},
            {
                "readable_count": sum(row.get("readable") is True for row in ledgers),
                "foreign_live_count": sum(row.get("foreign_live") is True for row in ledgers),
                "rows": ledgers,
            },
            ledger_ok,
        ),
        _check(
            "checkpoint_writable",
            True,
            checkpoint_writable,
            checkpoint_writable,
        ),
        _check(
            "embedded_gguf_tokenizers",
            {model_id: True for model_id in REQUIRED_MODEL_IDS},
            {str(row.get("model_id")): row.get("passed") for row in metadata_rows},
            len(metadata_rows) == 3 and all(row.get("passed") is True for row in metadata_rows),
        ),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "gpu_topology": gpu,
        "baseline_gpu_memory_rows": _memory_rows(gpu),
        "llama_cpp": binding,
        "lease_ledger_rows": ledgers,
        "gguf_metadata_rows": metadata_rows,
        "checkpoint_path": str(checkpoint_path),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every check and the first failed expected-observed pair."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def worker_execute(
    payload: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Load, generate with the embedded tokenizer, close, and collect."""

    llm: Any = None
    started_ns = clock()
    result: JsonDict
    close_called = False
    try:
        if llama_factory is None:
            from llama_cpp import Llama

            llama_factory = Llama
        llm = llama_factory(
            model_path=str(payload["model_path"]),
            n_ctx=int(LOAD_CONFIG["n_ctx"]),
            n_gpu_layers=int(LOAD_CONFIG["n_gpu_layers"]),
            n_batch=int(LOAD_CONFIG["n_batch"]),
            n_ubatch=int(LOAD_CONFIG["n_ubatch"]),
            main_gpu=int(LOAD_CONFIG["main_gpu"]),
            split_mode=1,
            tensor_split=list(LOAD_CONFIG["tensor_split"]),
            use_mmap=bool(LOAD_CONFIG["use_mmap"]),
            use_mlock=bool(LOAD_CONFIG["use_mlock"]),
            seed=RANDOM_SEED,
            verbose=True,
        )
        loaded_ns = clock()
        response = llm.create_completion(
            FIXED_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            top_p=1.0,
            seed=RANDOM_SEED,
        )
        ended_ns = clock()
        choice = (response.get("choices") or [{}])[0]
        usage = dict(response.get("usage", {}))
        output = str(choice.get("text") or "")
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if completion_tokens <= 0 and output:
            completion_tokens = len(llm.tokenize(output.encode("utf-8"), add_bos=False))
        live_duration = max(0.0, (ended_ns - loaded_ns) / 1_000_000_000)
        result = {
            "model_id": payload["model_id"],
            "terminal_state": "complete" if output else "failed",
            "load_duration_s": max(0.0, (loaded_ns - started_ns) / 1_000_000_000),
            "live_duration_s": live_duration,
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": completion_tokens,
            "tokens_per_second": completion_tokens / live_duration if live_duration > 0 else 0.0,
            "output": output,
            "output_hash": sha256_text(output),
            "finish_reason": str(choice.get("finish_reason") or "unknown"),
            "exception_type": None,
            "exception_message": None,
            "exception_traceback": None,
        }
    except Exception as exc:  # noqa: BLE001 - exact backend failures belong in the receipt.
        result = {
            "model_id": payload.get("model_id"),
            "terminal_state": "failed",
            "load_duration_s": 0.0,
            "live_duration_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens_per_second": 0.0,
            "output": "",
            "output_hash": sha256_text(""),
            "finish_reason": None,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "exception_traceback": traceback.format_exc(),
        }
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
                close_called = True
        llm = None
        gc.collect()
    result["model_close_called"] = close_called
    result["garbage_collection_ran"] = True
    return result


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """Read parent PID and start ticks without confusing spaces in process names."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = stat[stat.rfind(")") + 2 :].split()
        return int(suffix[1]), int(suffix[19])
    except (OSError, ValueError, IndexError):
        return None


def port_owner_pids(port: int) -> list[int]:  # pragma: no cover - live ss boundary.
    """Return listener PIDs reported by the kernel socket tool."""

    receipt = _run_command(("ss", "-Hlnpt", f"sport = :{int(port)}"))
    return sorted({int(value) for value in re.findall(r"pid=(\d+)", receipt["stdout"])})


def capture_server_ownership(
    pid: int,
    parent_pid: int,
    command: Sequence[str],
    port: int,
    *,
    port_owner_probe: Callable[[int], list[int]] = port_owner_pids,
    lineage_probe: Callable[[int, Mapping[str, Any]], JsonDict] = capture_process_lineage,
    task_identity: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Bind the worker, readiness socket, and process tree to this controller."""

    identity = _proc_stat(pid)
    observed_parent, start_ticks = identity if identity is not None else (None, None)
    task = dict(task_identity or read_process_identity(parent_pid) or {})
    lineage = lineage_probe(pid, task) if task else {"owned": False, "chain": []}
    owners = port_owner_probe(port)
    lineage_owned = lineage.get("owned") is True or lineage.get("task_owned") is True
    return {
        "pid": int(pid),
        "expected_parent_pid": int(parent_pid),
        "parent_pid": observed_parent,
        "pid_start_ticks": start_ticks,
        "command": list(command),
        "command_hash": sha256_text(canonical_json(list(command))),
        "port": int(port),
        "port_owner_pids": owners,
        "port_owned_by_child": owners == [int(pid)],
        "process_tree": deepcopy(lineage),
        "owned": bool(
            identity is not None
            and observed_parent == int(parent_pid)
            and owners == [int(pid)]
            and lineage_owned
        ),
        "signals_sent": [],
    }


def owned_process_absent(pid: int, start_ticks: int | None) -> bool:
    """Treat only an absent or reused PID as proof that the worker ended."""

    identity = _proc_stat(pid)
    return identity is None or identity[1] != start_ticks


def terminate_owned_process(pid: int, parent_pid: int, start_ticks: int | None) -> bool:
    """Signal only the exact child identity created by this controller."""

    if _proc_stat(pid) != (int(parent_pid), start_ticks):
        return False
    os.killpg(pid, signal.SIGTERM)
    return True


def _memory_rows(gpu: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "index": row.get("index"),
            "uuid": row.get("uuid"),
            "memory_used_mb": row.get("memory_used_mb"),
            "memory_free_mb": row.get("memory_free_mb"),
        }
        for row in gpu.get("devices", [])
    ]


def foreign_process_rows(rows: Sequence[Mapping[str, Any]], *, owner_pid: int) -> list[JsonDict]:
    """Return GPU processes that do not match the exact task controller PID."""

    return [deepcopy(dict(row)) for row in rows if row.get("pid") != int(owner_pid)]


def retain_verified_ownership(current: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Keep the first verified socket receipt after the worker closes its port."""

    return deepcopy(dict(current if current.get("owned") is True else candidate))


def _choose_free_port() -> int:  # pragma: no cover - live socket boundary.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _port_is_free(port: int) -> bool:  # pragma: no cover - live socket boundary.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def acquire_owned_leases(
    *,
    model: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
    journal_before: Mapping[str, Any],
    lease_factory: Callable[..., Any] = lease_api.GpuLease.acquire,
) -> tuple[list[Any], list[JsonDict]]:
    """Acquire both device locks through the shipped owner-bound API."""

    leases: list[Any] = []
    rows: list[JsonDict] = []
    try:
        for device in sorted(devices, key=lambda row: int(row.get("index", 0))):
            device_uuid = str(device["uuid"])
            lease = lease_factory(
                runtime_dir=runtime_dir,
                task_id=EXPERIMENT_ID,
                device_uuid=device_uuid,
                expected_model=str(model["model_path"]),
                vram_before_mb=int(device.get("memory_used_mb", 0) or 0),
                ttl_s=LEASE_TTL_S,
            )
            leases.append(lease)
            owner = dict(lease.owner_receipt())
            lease.transition("admitted")
            lease.transition("loading")
            rows.append(
                {
                    "model_id": model["hf_id"],
                    "device_uuid": device_uuid,
                    "task_id": EXPERIMENT_ID,
                    "owner_pid": owner.get("pid"),
                    "owner_pid_start_ticks": owner.get("pid_start_ticks"),
                    "owner_verified": (
                        owner.get("task_id") == EXPERIMENT_ID
                        and owner.get("device_uuid") == device_uuid
                        and owner.get("expected_model") == str(model["model_path"])
                    ),
                    "expected_model": str(model["model_path"]),
                    "journal_before_acquisition": deepcopy(journal_before.get(device_uuid)),
                    "journal_after_acquisition": deepcopy(lease.document),
                    "journal_after_release": None,
                    "journal_validation_errors": [],
                    "recovery": deepcopy(owner.get("recovery", {})),
                    "release_receipt": {},
                    "signals_sent": [],
                    "consistent": False,
                }
            )
    except Exception:
        for lease in leases:
            try:
                if lease.document.get("phase") in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                    lease.release()
                else:
                    lease.close()
            except lease_api.LeaseError:
                lease.close()
        raise
    return leases, rows


def lease_row_is_consistent(row: Mapping[str, Any]) -> bool:
    """Recompute a released lease receipt without checking a historical live PID."""

    after = row.get("journal_after_release")
    recovery = row.get("recovery")
    release = row.get("release_receipt")
    return bool(
        row.get("model_id") in REQUIRED_MODEL_IDS
        and row.get("task_id") == EXPERIMENT_ID
        and str(row.get("device_uuid", "")).startswith("GPU-")
        and isinstance(row.get("owner_pid"), int)
        and int(row.get("owner_pid", 0)) > 1
        and isinstance(row.get("owner_pid_start_ticks"), int)
        and int(row.get("owner_pid_start_ticks", -1)) >= 0
        and row.get("owner_verified") is True
        and isinstance(after, Mapping)
        and after.get("released") is True
        and after.get("phase") in lease_api.TERMINAL_PHASES
        and isinstance(recovery, Mapping)
        and recovery.get("signals_sent") == []
        and isinstance(release, Mapping)
        and release.get("released") is True
        and release.get("signals_sent") == []
        and row.get("journal_validation_errors") == []
        and row.get("signals_sent") == []
    )


def _finalize_owned_leases(  # pragma: no cover - exercised by the live E2E run.
    leases: Sequence[Any],
    rows: list[JsonDict],
    *,
    complete: bool,
    vram_release: Mapping[str, Any],
    exit_code: int,
) -> None:
    """Terminalize and release only leases owned by this controller."""

    after_by_uuid = {
        str(row.get("uuid")): int(row.get("memory_used_mb", 0) or 0)
        for row in vram_release.get("after_rows", [])
    }
    for lease, row in zip(leases, rows, strict=True):
        try:
            phase = str(lease.document.get("phase"))
            if phase in {"resident", "inferencing"}:
                lease.transition("unloading")
                phase = "unloading"
            if phase == "unloading" and vram_release.get("passed") is True:
                lease.transition(
                    "validating",
                    vram_mb=after_by_uuid.get(str(lease.device_uuid), 0),
                    exit_code=exit_code,
                    unload_observed=True,
                )
                lease.transition("terminal_complete" if complete else "terminal_blocked")
            elif phase in {"preflight", "admitted", "loading"}:
                lease.transition("terminal_blocked")
            else:
                lease.close()
                row["journal_after_release"] = deepcopy(lease.document)
                row["journal_validation_errors"] = ["vram_release_not_proved"]
                continue
            row["release_receipt"] = lease.release()
            row["journal_after_release"] = deepcopy(lease.document)
            row["journal_validation_errors"] = lease_api.validate_journal_document(
                lease.document, check_freshness=False
            )
        except lease_api.LeaseError as exc:
            lease.close()
            row["journal_after_release"] = deepcopy(lease.document)
            row["journal_validation_errors"] = [f"{type(exc).__name__}: {exc}"]
        row["consistent"] = lease_row_is_consistent(row)


def _wait_for_vram_release(  # pragma: no cover - exercised by the live E2E run.
    baseline_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    *,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
) -> JsonDict:
    """Poll until both devices return to the per-attempt baseline."""

    deadline = time.monotonic() + VRAM_RELEASE_TIMEOUT_S
    latest = gpu_probe()
    while True:
        after_rows = _memory_rows(latest)
        row = build_vram_release_row(
            model_id=model_id, baseline_rows=baseline_rows, after_rows=after_rows
        )
        row["after_rows"] = after_rows
        if row["passed"] is True or time.monotonic() >= deadline:
            return row
        time.sleep(1.0)
        latest = gpu_probe()


def _empty_attempt(model: Mapping[str, Any], reason: str) -> JsonDict:
    """Build a terminal blocked row when safe acquisition fails before launch."""

    model_id = str(model["hf_id"])
    return {
        "model_id": model_id,
        "terminal_state": "blocked",
        "config_id": LOAD_CONFIG["config_id"],
        "load_config": deepcopy(LOAD_CONFIG),
        "live_cuda": False,
        "gpu_uuids_used": [],
        "output": "",
        "output_hash": sha256_text(""),
        "process_exit_code": None,
        "owned_process_absent": True,
        "process_owned": False,
        "port_release_confirmed": True,
        "teardown_complete": True,
        "vram_release_passed": True,
        "lease_consistent": False,
        "offloaded_layers": 0,
        "total_layers": None,
        "live_duration_s": 0.0,
        "tokens_per_second": 0.0,
        "exception_type": "LeaseAcquisitionError",
        "exception_message": reason,
        "exception_traceback": None,
        "lease_rows": [],
        "process_ownership": {"model_id": model_id, "owned": False, "signals_sent": []},
        "gpu_runtime": {"model_id": model_id, "live_cuda": False, "gpu_samples": []},
        "teardown": {"model_id": model_id, "passed": True, "signals_sent": []},
        "vram_release": {"model_id": model_id, "passed": True},
        "handoff_safe": False,
    }


def run_model_attempt(  # pragma: no cover - exercised by the required live E2E run.
    *,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    devices: Sequence[Mapping[str, Any]],
    runtime_dir: Path,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
) -> JsonDict:
    """Acquire both leases, run one owned child, and prove full release."""

    runtime_dir.mkdir(parents=True, exist_ok=True)
    baseline = gpu_probe()
    baseline_rows = _memory_rows(baseline)
    if foreign_process_rows(baseline.get("processes", []), owner_pid=os.getpid()):
        return _empty_attempt(model, "foreign_gpu_compute_process_appeared")
    before_rows = lease_ledger_snapshot(devices, LEASE_RUNTIME_DIR)
    before_by_uuid = {str(row["device_uuid"]): row.get("document") for row in before_rows}
    leases: list[Any] = []
    lease_rows: list[JsonDict] = []
    try:
        leases, lease_rows = acquire_owned_leases(
            model=model,
            devices=devices,
            runtime_dir=LEASE_RUNTIME_DIR,
            journal_before=before_by_uuid,
        )
    except lease_api.LeaseError as exc:
        for lease in leases:
            lease.close()
        return _empty_attempt(model, f"{type(exc).__name__}: {exc}")

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(model["hf_id"])).strip("-").lower()
    attempt_dir = runtime_dir / slug
    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload_path = attempt_dir / "payload.json"
    output_path = attempt_dir / "worker.json"
    ready_path = attempt_dir / "ready.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    port = _choose_free_port()
    write_json_atomic(
        payload_path,
        {"model_id": model["hf_id"], "model_path": model["model_path"]},
    )
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--worker-payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
        "--worker-ready",
        str(ready_path),
        "--worker-port",
        str(port),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    task_identity = read_process_identity(os.getpid()) or {}
    samples: list[JsonDict] = []
    ownership: JsonDict = {"owned": False, "signals_sent": []}
    timed_out = False
    resident_recorded = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        start_ticks = lease_api.proc_start_ticks(process.pid)
        deadline = time.monotonic() + MODEL_TIMEOUT_S
        while process.poll() is None and time.monotonic() < deadline:
            sample = gpu_probe()
            sample["monotonic_ns"] = time.monotonic_ns()
            samples.append(sample)
            if ready_path.is_file():
                ownership = retain_verified_ownership(
                    ownership,
                    capture_server_ownership(
                        process.pid,
                        os.getpid(),
                        command,
                        port,
                        task_identity=task_identity,
                    ),
                )
            owned_gpu_uuids = {
                str(row.get("gpu_uuid"))
                for row in sample.get("processes", [])
                if row.get("pid") == process.pid
            }
            if not resident_recorded and owned_gpu_uuids == {
                str(device["uuid"]) for device in devices
            }:
                used_by_uuid = {
                    str(device["uuid"]): int(device.get("memory_used_mb", 0) or 0)
                    for device in sample.get("devices", [])
                }
                for lease in leases:
                    lease.transition("resident", vram_mb=used_by_uuid.get(lease.device_uuid, 0))
                    lease.transition("inferencing")
                resident_recorded = True
            time.sleep(POLL_INTERVAL_S)
        if process.poll() is None:
            timed_out = True
            if terminate_owned_process(process.pid, os.getpid(), start_ticks):
                ownership.setdefault("signals_sent", []).append("SIGTERM")
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if _proc_stat(process.pid) == (os.getpid(), start_ticks):
                        os.killpg(process.pid, signal.SIGKILL)
                        ownership.setdefault("signals_sent", []).append("SIGKILL")
        process.wait(timeout=30)

    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if output_path.is_file():
        worker = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        worker = _empty_attempt(model, "worker_timeout" if timed_out else "worker_output_missing")
    if ownership.get("owned") is not True and ready_path.is_file():
        ownership = {
            **ownership,
            "ready_receipt": json.loads(ready_path.read_text(encoding="utf-8")),
        }
    layer_row = parse_offloaded_layers(stderr_text)
    release = _wait_for_vram_release(baseline_rows, str(model["hf_id"]), gpu_probe=gpu_probe)
    absent = owned_process_absent(process.pid, start_ticks)
    port_released = _port_is_free(port)
    owned_samples = [
        {"monotonic_ns": sample["monotonic_ns"], **row}
        for sample in samples
        for row in sample.get("processes", [])
        if row.get("pid") == process.pid
    ]
    gpu_uuids = sorted({str(row.get("gpu_uuid")) for row in owned_samples})
    baseline_by_index = {int(row["index"]): int(row["memory_used_mb"]) for row in baseline_rows}
    peak_by_index = {
        index: max(
            [
                int(device.get("memory_used_mb", 0) or 0)
                for sample in samples
                for device in sample.get("devices", [])
                if int(device.get("index", -1)) == index
            ]
            or [baseline]
        )
        for index, baseline in baseline_by_index.items()
    }
    memory_delta = {
        str(index): peak_by_index[index] - baseline for index, baseline in baseline_by_index.items()
    }
    provisional_complete = bool(
        worker.get("terminal_state") == "complete"
        and worker.get("output")
        and ownership.get("owned") is True
        and process.returncode == 0
        and absent
        and port_released
        and release.get("passed") is True
        and layer_row["offloaded"] > 0
        and len(gpu_uuids) == 2
    )
    _finalize_owned_leases(
        leases,
        lease_rows,
        complete=provisional_complete,
        vram_release=release,
        exit_code=int(process.returncode or 0),
    )
    lease_consistent = len(lease_rows) == 2 and all(
        lease_row_is_consistent(row) for row in lease_rows
    )
    lease_ready = lease_consistent and all(
        dict(row.get("journal_after_release", {})).get("phase") == "terminal_complete"
        for row in lease_rows
    )
    teardown_complete = bool(absent and port_released and worker.get("model_close_called") is True)
    row: JsonDict = {
        **dict(worker),
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "model_file_hash": sha256_file(model["model_path"]),
        "config_id": config["config_id"],
        "load_config": deepcopy(dict(config)),
        "offloaded_layers": layer_row["offloaded"],
        "total_layers": layer_row["total"],
        "live_cuda": bool(layer_row["offloaded"] > 0 and len(gpu_uuids) == 2),
        "gpu_uuids_used": gpu_uuids,
        "gpu_memory_delta_mb": memory_delta,
        "process_exit_code": process.returncode,
        "owned_process_absent": absent,
        "process_owned": ownership.get("owned") is True,
        "port_release_confirmed": port_released,
        "teardown_complete": teardown_complete,
        "vram_release_passed": release.get("passed") is True,
        "lease_consistent": lease_consistent,
        "lease_ready": lease_ready,
        "lease_rows": lease_rows,
        "process_ownership": {"model_id": model["hf_id"], **ownership},
        "gpu_runtime": {
            "model_id": model["hf_id"],
            "live_cuda": bool(layer_row["offloaded"] > 0 and len(gpu_uuids) == 2),
            "gpu_uuids_used": gpu_uuids,
            "gpu_memory_delta_mb": memory_delta,
            "gpu_samples": owned_samples,
            "offloaded_layers": layer_row["offloaded"],
            "total_layers": layer_row["total"],
        },
        "teardown": {
            "model_id": model["hf_id"],
            "process_exit_code": process.returncode,
            "owned_process_absent": absent,
            "model_close_called": worker.get("model_close_called"),
            "port": port,
            "port_release_confirmed": port_released,
            "signals_sent": ownership.get("signals_sent", []),
            "passed": teardown_complete,
        },
        "vram_release": release,
        "backend_stdout_hash": sha256_text(stdout_text),
        "backend_stderr_hash": sha256_text(stderr_text),
        "backend_stderr": stderr_text,
    }
    row["terminal_state"] = "complete" if provisional_complete and lease_ready else "failed"
    row["handoff_safe"] = bool(teardown_complete and release.get("passed") and lease_consistent)
    return row


def _row_handoff_safe(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("handoff_safe", True)
        and row.get("owned_process_absent") is True
        and row.get("port_release_confirmed") is True
        and row.get("teardown_complete") is True
        and row.get("vram_release_passed") is True
        and row.get("lease_consistent") is True
        and row.get("lease_ready", True) is True
    )


def reduce_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Derive bare completion and readiness from exact per-family rows."""

    exact = [row.get("model_id") for row in rows] == list(REQUIRED_MODEL_IDS)
    terminal = exact and all(
        row.get("terminal_state") in {"complete", "failed", "blocked"} for row in rows
    )
    ready = terminal and all(
        row.get("terminal_state") == "complete"
        and row.get("config_id") == LOAD_CONFIG["config_id"]
        and int(dict(row.get("load_config", {})).get("n_ctx", 0) or 0) >= 16_384
        and row.get("live_cuda") is True
        and len(row.get("gpu_uuids_used", [])) == 2
        and bool(str(row.get("output", "")))
        and row.get("output_hash") == sha256_text(str(row.get("output", "")))
        and row.get("process_exit_code") == 0
        and row.get("process_owned") is True
        and int(row.get("offloaded_layers", 0) or 0) > 0
        and float(row.get("live_duration_s", 0.0) or 0.0) > 0
        and float(row.get("tokens_per_second", 0.0) or 0.0) > 0
        and _row_handoff_safe(row)
        and len(row.get("lease_rows", [])) == 2
        and all(lease_row_is_consistent(lease) for lease in row.get("lease_rows", []))
        and all(
            dict(lease.get("journal_after_release", {})).get("phase") == "terminal_complete"
            for lease in row.get("lease_rows", [])
        )
        for row in rows
    )
    return int(terminal), int(ready)


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the digest that contains the hash."""

    return sha256_text(
        canonical_json(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    live_generation_rows: Sequence[Mapping[str, Any]] = (),
    source_artifact_hashes: Mapping[str, Any] | None = None,
    model_file_hashes: Mapping[str, Any] | None = None,
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one complete artifact from measured detail rows."""

    generations = [deepcopy(dict(row)) for row in live_generation_rows]
    complete_score, ready_score = reduce_scores(generations)
    preflight_passed = preconditions.get("all_passed") is True
    if not preflight_passed:
        verdict_class = "blocked"
        verdict = "blocked_lease_aware_gguf_runtime"
    elif ready_score == 1:
        verdict_class = "positive"
        verdict = "complete: all three lease-owned GGUF families generated on CUDA and released"
    elif complete_score == 1:
        verdict_class = "null"
        verdict = "complete_null_lease_aware_gguf_runtime"
    else:
        verdict_class = "partial"
        verdict = "partial_lease_aware_gguf_runtime"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": sum(
            float(row.get("live_duration_s", 0.0) or 0.0) for row in generations
        ),
        "source_artifact_hashes": dict(source_artifact_hashes or {}),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [str(row.get("hf_id")) for row in model_specs],
        "model_file_hashes": dict(model_file_hashes or {}),
        "gpu_topology": deepcopy(dict(preconditions.get("gpu_topology", {}))),
        "lease_rows": [
            deepcopy(dict(lease)) for row in generations for lease in row.get("lease_rows", [])
        ],
        "process_ownership_rows": [
            deepcopy(dict(row.get("process_ownership", {}))) for row in generations
        ],
        "baseline_gpu_memory_rows": deepcopy(
            list(preconditions.get("baseline_gpu_memory_rows", []))
        ),
        "load_config_rows": [deepcopy(LOAD_CONFIG)],
        "live_generation_rows": generations,
        "gpu_runtime_rows": [deepcopy(dict(row.get("gpu_runtime", {}))) for row in generations],
        "teardown_rows": [deepcopy(dict(row.get("teardown", {}))) for row in generations],
        "vram_release_rows": [deepcopy(dict(row.get("vram_release", {}))) for row in generations],
        "checkpoint_rows": [deepcopy(dict(row)) for row in checkpoint_rows],
        "runtime_handoff_complete_score": complete_score,
        "lease_aware_runtime_ready_score": ready_score,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(preconditions.get("checks", [])),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Cold-recompute projections, scores, verdict, and checksum."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing_field:{field}" for field in missing)
    if missing:
        return errors
    if set(dict(artifact.get("field_principles", {}))) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("schema_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("models_used") != list(REQUIRED_MODEL_IDS):
        errors.append("models_used_mismatch")
    specs = artifact.get("MODEL_SPECS", [])
    if [row.get("hf_id") for row in specs] != list(REQUIRED_MODEL_IDS):
        errors.append("model_specs_mismatch")
    generations = artifact.get("live_generation_rows", [])
    lease_projection = [
        deepcopy(dict(lease)) for row in generations for lease in row.get("lease_rows", [])
    ]
    if artifact.get("lease_rows") != lease_projection:
        errors.append("lease_rows_projection_mismatch")
    projections = (
        ("process_ownership_rows", "process_ownership"),
        ("gpu_runtime_rows", "gpu_runtime"),
        ("teardown_rows", "teardown"),
        ("vram_release_rows", "vram_release"),
    )
    for field, child in projections:
        expected = [deepcopy(dict(row.get(child, {}))) for row in generations]
        if artifact.get(field) != expected:
            errors.append(f"{field}_projection_mismatch")
    derived_complete, derived_ready = reduce_scores(generations)
    for field, expected in (
        ("runtime_handoff_complete_score", derived_complete),
        ("lease_aware_runtime_ready_score", derived_ready),
    ):
        value = artifact.get(field)
        if type(value) is not int:
            errors.append(f"gate_score_not_bare_int:{field}")
        elif value != expected:
            errors.append(f"gate_score_mismatch:{field}")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    try:
        durations_valid = (
            float(artifact.get("duration_s", -1)) >= 0
            and float(artifact.get("live_duration_s", -1)) >= 0
        )
    except (TypeError, ValueError):
        durations_valid = False
    if not durations_valid:
        errors.append("duration_invalid")
    preflight_passed = artifact.get("preconditions_checked", {}).get("all_passed") is True
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if not preflight_passed:
        summary = artifact.get("gate_check_summary")
        if verdict_class != "blocked" or verdict != "blocked_lease_aware_gguf_runtime":
            errors.append("blocked_verdict_mismatch")
        if not isinstance(summary, Mapping) or not all(
            key in summary for key in ("failed_check", "expected_value", "observed_value")
        ):
            errors.append("blocked_gate_summary_incomplete")
    elif derived_ready == 1:
        if verdict_class != "positive" or not verdict.startswith("complete:"):
            errors.append("positive_verdict_mismatch")
    elif derived_complete == 1:
        if verdict_class != "null" or not verdict.startswith("complete_null"):
            errors.append("null_verdict_mismatch")
    elif verdict_class != "partial" or not verdict.startswith("partial_"):
        errors.append("partial_verdict_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def checkpoint_model_row(path: Path, manifest_hash: str, row: Mapping[str, Any]) -> JsonDict:
    """Append one terminal family row while refusing manifest or row drift."""

    document: JsonDict = {"manifest_hash": manifest_hash, "rows": []}
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("manifest_hash") != manifest_hash:
            raise ValueError("checkpoint_manifest_mismatch")
    rows = list(document.get("rows", []))
    existing = next((item for item in rows if item.get("model_id") == row.get("model_id")), None)
    if existing is not None:
        if existing != dict(row):
            raise ValueError("checkpoint_model_row_mismatch")
        return {"model_id": row.get("model_id"), "written": False, "path": str(path)}
    rows.append(deepcopy(dict(row)))
    write_json_atomic(path, {"manifest_hash": manifest_hash, "rows": rows})
    return {"model_id": row.get("model_id"), "written": True, "path": str(path)}


def _source_hashes() -> dict[str, str | None]:  # pragma: no cover - live file boundary.
    paths = {
        "experiment_6966": REPO_ROOT / "results/experiment_6966_gguf_load_envelope_canary.json",
        "experiment_5284": REPO_ROOT
        / "results/experiment_5284_sota_runtime_offload_receipt_repair_v483.json",
        "module": Path(__file__),
        "wrapper": REPO_ROOT / "scripts/experiments/experiment_6973_lease_aware_gguf_runtime.py",
        "tests": REPO_ROOT / "tests/python/test_experiment_6973_lease_aware_gguf_runtime.py",
        "spec": REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md",
        "lease_api": REPO_ROOT / "python/carnot/gpu_lease_phase_journal.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _model_file_hashes(  # pragma: no cover - hashes the required large model files.
    model_specs: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    return {str(row["hf_id"]): sha256_file(row["model_path"]) for row in model_specs}


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preflight_fn: Callable[[Sequence[Mapping[str, Any]], Path], JsonDict] | None = None,
    attempt_runner: Callable[..., JsonDict] = run_model_attempt,
    source_hash_fn: Callable[[], Mapping[str, Any]] = _source_hashes,
    model_hash_fn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] = _model_file_hashes,
) -> JsonDict:
    """Preflight, run three safe sequential attempts, and checkpoint each row."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    preconditions = (
        collect_preconditions(model_specs=specs, checkpoint_path=checkpoint_path)
        if preflight_fn is None
        else preflight_fn(specs, checkpoint_path)
    )
    source_hashes = dict(source_hash_fn())
    file_hashes = dict(model_hash_fn(specs))
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    manifest_hash = sha256_text(canonical_json({"models": specs, "config": LOAD_CONFIG}))
    runtime_dir = checkpoint_path.parent / "attempts"
    if preconditions.get("all_passed") is True:
        devices = list(preconditions.get("gpu_topology", {}).get("devices", []))
        for model in specs:
            row = attempt_runner(
                model=model,
                config=LOAD_CONFIG,
                devices=devices,
                runtime_dir=runtime_dir,
            )
            rows.append(deepcopy(dict(row)))
            checkpoints.append(checkpoint_model_row(checkpoint_path, manifest_hash, row))
            artifact = build_artifact(
                run_date=run_date,
                duration_s=time.perf_counter() - started,
                model_specs=specs,
                preconditions=preconditions,
                live_generation_rows=rows,
                source_artifact_hashes=source_hashes,
                model_file_hashes=file_hashes,
                checkpoint_rows=checkpoints,
            )
            write_json_atomic(result_path, artifact)
            if not _row_handoff_safe(row):
                break
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        model_specs=specs,
        preconditions=preconditions,
        live_generation_rows=rows,
        source_artifact_hashes=source_hashes,
        model_file_hashes=file_hashes,
        checkpoint_rows=checkpoints,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def _worker_main(  # pragma: no cover - private live subprocess boundary.
    payload_path: Path,
    output_path: Path,
    ready_path: Path,
    port: int,
) -> int:
    """Hold an owned readiness socket while the direct llama.cpp worker runs."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind(("127.0.0.1", int(port)))
        listener.listen(1)
        write_json_atomic(
            ready_path,
            {
                "pid": os.getpid(),
                "pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
                "port": port,
            },
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        row = worker_execute(payload)
        write_json_atomic(output_path, row)
        return 0 if row["terminal_state"] == "complete" else 1
    finally:
        listener.close()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
    """Run the controller, private worker, or cold artifact validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-ready", type=Path)
    parser.add_argument("--worker-port", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_payload is not None:
        if args.worker_output is None or args.worker_ready is None or args.worker_port is None:
            parser.error("worker mode requires output, ready, and port")
        return _worker_main(
            args.worker_payload, args.worker_output, args.worker_ready, args.worker_port
        )
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(artifact)
        print(canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        checkpoint_path=args.checkpoint_path,
    )
    errors = validate_artifact(artifact)
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "runtime_handoff_complete_score": artifact["runtime_handoff_complete_score"],
                "lease_aware_runtime_ready_score": artifact["lease_aware_runtime_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
