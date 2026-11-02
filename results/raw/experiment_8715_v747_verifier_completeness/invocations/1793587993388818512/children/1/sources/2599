"""Qualify task-owned runtime receipts on three local SOTA GGUF models.

The experiment uses the receipt helper from REQ-REPORT-6924. A small child
worker exposes load, generation, and close boundaries to the parent task.
This lets the receipt bind Linux process identity and live CUDA samples to
each phase without creating another telemetry format.

Spec refs: REQ-REPORT-6928 and SCENARIO-REPORT-6928-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from carnot.inference.sota_models import resolve_cached_gguf
from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]
Resolver = Callable[[str, str], str | None]

TASK_ID = "exp6928-sota-runtime-receipt-qualification"
RESULT_RELATIVE_PATH = Path("results/experiment_6928_sota_runtime_receipt_qualification.json")
INFERENCE_SUBSTRATE = "task_owned_local_gguf_cuda_inference"
RANDOM_SEED = 6928
CONTEXT_SIZE = 512
OFFLOAD_LAYERS = -1
MAX_TOKENS = 8
MODEL_TIMEOUT_S = 300.0
TELEMETRY_INTERVAL_S = 0.25
MIN_DISK_FREE_BYTES = 1024**3
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/experiments/experiment_6928_sota_runtime_receipt_qualification.py"
)

MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "family": "qwen3.6_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-31B-it",
        "family": "gemma4_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": "Q4_K_M",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "family": "gemma4_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": "Q4_K_M",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "model_specs",
    "model_rows",
    "model_file_rows",
    "runner_selection_rows",
    "process_lineage_rows",
    "task_phase_timing_rows",
    "task_gpu_telemetry_rows",
    "model_concurrency_rows",
    "server_lifecycle_rows",
    "cache_state_rows",
    "teardown_rows",
    "fresh_process_recheck_rows",
    "forged_receipt_rejection_rows",
    "task_runtime_receipt",
    "random_seed",
    "reproducibility_checksum",
    "sota_runtime_receipt_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states the evidence principle that justifies it.",
    "preconditions_checked": "Hardware claims start only after falsifiable resource checks pass.",
    "inference_substrate": "The substrate separates live local CUDA inference from fixtures and hosted services.",
    "duration_s": "Measured elapsed time bounds the cost and makes implausible runtime claims visible.",
    "source_artifact_hashes": "Source hashes bind the result to the code and prior receipt it qualifies.",
    "rows": "Canonical receipt rows keep raw phase evidence available for independent replay.",
    "model_specs": "A fixed model set prevents a smaller substitute from satisfying the qualification.",
    "model_rows": "One terminal row per model preserves successes and failures without survivorship bias.",
    "model_file_rows": "File paths and hashes bind model names to the exact local weights used.",
    "runner_selection_rows": "Runner identity and commands distinguish executed code from a label.",
    "process_lineage_rows": "Kernel process identities prove that each inference child belongs to this task.",
    "task_phase_timing_rows": "Monotonic phase intervals support ordering and duration reconstruction.",
    "task_gpu_telemetry_rows": "PID-linked GPU samples prove residency, utilization, and requested offload.",
    "model_concurrency_rows": "Recomputed lifecycles prove the models loaded one at a time.",
    "server_lifecycle_rows": "Start and stop events expose the full lifetime of each model worker.",
    "cache_state_rows": "Explicit hits and misses prevent an absent model from becoming silent evidence.",
    "teardown_rows": "Exit, reap, and post-exit residency checks reveal leaked worker processes.",
    "fresh_process_recheck_rows": "A separate interpreter tests claims without trusting producer state.",
    "forged_receipt_rejection_rows": "Critical mutations demonstrate that attribution checks fail closed.",
    "task_runtime_receipt": "The complete serialized receipt lets later tasks repeat the independent check.",
    "random_seed": "A fixed seed makes the bounded generation request reproducible.",
    "reproducibility_checksum": "The checksum binds the date, models, devices, sources, and seed.",
    "sota_runtime_receipt_ready_score": "One requires all three task-owned dual-CUDA model receipts and fresh replay.",
    "gate_check_summary": "Every blocked result reports the expected and observed gate values.",
    "verifier_is_oracle": "False keeps infrastructure validation outside scientific authority.",
    "verdict_class": "The verdict class keeps an advisory receipt separate from a science result.",
    "honest_verdict": "A terminal prefix makes completion or blockage machine-readable.",
}

SOURCE_PATHS = (
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/experiment_6928_sota_runtime_receipt_qualification.py"),
    Path("scripts/experiments/experiment_6928_sota_runtime_receipt_qualification.py"),
    Path("results/experiment_6924_task_runtime_receipt_adoption.json"),
)


def _check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one preflight row with comparable expected and observed values."""

    return {
        "check": name,
        "expected_value": expected,
        "observed_value": observed,
        "passed": bool(passed),
    }


def _run_command(command: Sequence[str], *, timeout_s: float = 10.0) -> JsonDict:
    """Run one bounded read-only host probe and retain its terminal output."""

    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def query_gpus() -> list[JsonDict]:
    """Read stable GPU names and UUIDs from the NVIDIA management interface."""

    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["ok"]:
        return []
    rows: list[JsonDict] = []
    for line in str(result["stdout"]).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_total_mb": int(float(parts[3])),
                    "memory_used_mb": int(float(parts[4])),
                    "utilization_pct": int(float(parts[5])),
                }
            )
        except ValueError:
            continue
    return rows


def llama_cpp_status() -> JsonDict:
    """Confirm that the installed llama.cpp Python binding reports CUDA offload."""

    try:
        import llama_cpp
        from llama_cpp import llama_cpp as backend

        supported = bool(backend.llama_supports_gpu_offload())
        return {
            "importable": True,
            "version": getattr(llama_cpp, "__version__", "unknown"),
            "supports_gpu_offload": supported,
            "module_path": str(Path(llama_cpp.__file__).resolve()),
        }
    except Exception as exc:  # pragma: no cover - depends on the host package
        return {
            "importable": False,
            "version": None,
            "supports_gpu_offload": False,
            "module_path": None,
            "error": repr(exc),
        }


def _exact_quantization(path: Path, requested: str) -> str:
    """Return the quantization token present in the selected GGUF filename."""

    tokens = (
        "UD-Q8_XL",
        "UD-Q5_K_M",
        "UD-Q4_K_M",
        "Q8_0",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
        "Q2_K",
    )
    upper_name = path.name.upper()
    for token in tokens:
        if token in upper_name:
            return token
    return requested


def resolve_model_files(
    *, resolver: Resolver = resolve_cached_gguf
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Resolve every mandated model and preserve both cache hits and misses."""

    model_files: list[JsonDict] = []
    checks: list[JsonDict] = []
    for spec in MODEL_SPECS:
        error = ""
        try:
            resolved = resolver(str(spec["hf_id"]), str(spec["quantization"]))
        except Exception as exc:
            resolved = None
            error = repr(exc)
        path = Path(resolved).absolute() if resolved else None
        hit = bool(path and path.is_file())
        row = {
            **dict(spec),
            "requested_quantization": spec["quantization"],
            "quantization": _exact_quantization(path, str(spec["quantization"]))
            if hit and path is not None
            else spec["quantization"],
            "model_path": str(path) if hit and path is not None else None,
            "model_file": path.name if hit and path is not None else None,
            "model_sha256": receipts.sha256_file(path) if hit and path is not None else None,
            "size_bytes": path.stat().st_size if hit and path is not None else 0,
            "cache_state": "hit" if hit else "miss",
            "resolver_error": error or None,
        }
        model_files.append(row)
        checks.append(
            _check(
                f"model_cache:{spec['hf_id']}",
                "cached_or_resolvable_gguf",
                str(path) if hit and path is not None else error or "missing",
                hit,
            )
        )
    return model_files, checks


def check_preconditions(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> JsonDict:
    """Check hardware, CUDA, model files, helper availability, and disk space."""

    target = output_path or repo_root / RESULT_RELATIVE_PATH
    model_files, model_checks = resolve_model_files()
    gpu_rows = query_gpus()
    rtx_rows = [
        row
        for row in gpu_rows
        if "RTX 3090" in str(row.get("name"))
        and str(row.get("uuid", "")).startswith("GPU-")
        and int(row.get("memory_total_mb", 0)) >= 23_000
    ]
    selected_gpus = sorted(rtx_rows, key=lambda row: int(row["index"]))[:2]
    unique_uuids = {str(row["uuid"]) for row in selected_gpus}
    llama = llama_cpp_status()
    disk = shutil.disk_usage(repo_root)
    target_parent = target.parent
    target_writable = target_parent.is_dir() and os.access(target_parent, os.W_OK)
    checks = [
        _check(
            "dual_authenticated_rtx3090_cuda_devices",
            2,
            len(selected_gpus),
            len(selected_gpus) == 2 and len(unique_uuids) == 2,
        ),
        _check(
            "llama_cpp_cuda_offload",
            True,
            llama,
            llama.get("importable") is True and llama.get("supports_gpu_offload") is True,
        ),
        _check(
            "existing_task_runtime_receipt_helper",
            "existing_file",
            "existing_file"
            if (repo_root / "python/carnot/task_runtime_receipts.py").is_file()
            else "missing",
            (repo_root / "python/carnot/task_runtime_receipts.py").is_file(),
        ),
        _check(
            "result_path_writable",
            "writable",
            "writable" if target_writable else "not_writable_or_missing",
            target_writable,
        ),
        _check(
            "disk_free_bytes",
            f">={MIN_DISK_FREE_BYTES}",
            disk.free,
            disk.free >= MIN_DISK_FREE_BYTES,
        ),
        *model_checks,
    ]
    return {
        "checks": checks,
        "model_files": model_files,
        "gpus": selected_gpus,
        "llama_cpp": llama,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
    }


def _append_once(values: list[str], value: str) -> None:
    """Add one stable validation reason without duplicate noise."""

    if value not in values:
        values.append(value)


def qualify_receipt_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_pid: int,
    expected_gpu_uuids: Sequence[str],
) -> JsonDict:
    """Apply the three-model CUDA qualification on existing receipt rows."""

    base = receipts.validate_adoption_rows(
        rows,
        expected_task_id=TASK_ID,
        expected_task_pid=expected_task_pid,
    )
    reasons = list(base["reasons"])
    expected_devices = tuple(str(value) for value in expected_gpu_uuids)
    if len(expected_devices) != 2 or any(not value for value in expected_devices):
        _append_once(reasons, "unresolved_gpu_identity")
    if len(set(expected_devices)) != len(expected_devices):
        _append_once(reasons, "unresolved_gpu_identity")

    rows_by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        identity = row.get("model_identity")
        model_id = str(identity.get("hf_id", "")) if isinstance(identity, Mapping) else ""
        if model_id:
            rows_by_model.setdefault(model_id, []).append(row)

    lifecycle_intervals: list[tuple[int, int, str]] = []
    qualified_models: list[str] = []
    for spec in MODEL_SPECS:
        model_id = str(spec["hf_id"])
        model_rows = rows_by_model.get(model_id, [])
        phases = {str(row.get("phase")) for row in model_rows}
        if not {"model_load", "generation", "teardown"} <= phases:
            _append_once(reasons, f"missing_model_phase:{model_id}")
        cache_valid = bool(model_rows) and all(
            isinstance(row.get("model_identity"), Mapping)
            and row["model_identity"].get("cache_state") == "hit"
            and bool(row["model_identity"].get("model_path"))
            and bool(row["model_identity"].get("model_sha256"))
            for row in model_rows
        )
        if not cache_valid:
            _append_once(reasons, f"model_cache_not_bound:{model_id}")

        started = {
            str(row.get("server_lifecycle", {}).get("server_id")): row.get("server_lifecycle", {})
            for row in model_rows
            if isinstance(row.get("server_lifecycle"), Mapping)
            and row["server_lifecycle"].get("event") == "started"
        }
        stopped = {
            str(row.get("server_lifecycle", {}).get("server_id")): row.get("server_lifecycle", {})
            for row in model_rows
            if isinstance(row.get("server_lifecycle"), Mapping)
            and row["server_lifecycle"].get("event") == "teardown"
            and row["server_lifecycle"].get("process_exit_confirmed") is True
            and row["server_lifecycle"].get("process_reaped") is True
        }
        lifecycle_valid = bool(started) and all(
            server_id in stopped and stopped[server_id].get("pid") == started_row.get("pid")
            for server_id, started_row in started.items()
        )
        if not lifecycle_valid:
            _append_once(reasons, f"model_teardown_incomplete:{model_id}")

        generation_rows = [row for row in model_rows if row.get("phase") == "generation"]
        device_valid = len(generation_rows) == 1
        for generation in generation_rows:
            child_pids = {int(pid) for pid in generation.get("child_pids", [])}
            samples = generation.get("gpu_samples", [])
            owned_devices = {
                str(sample.get("device_uuid"))
                for sample in samples
                if isinstance(sample, Mapping)
                and sample.get("pid") in child_pids
                and int(sample.get("pid_memory_mb", 0) or 0) > 0
                and sample.get("offload_layers") == OFFLOAD_LAYERS
                and "device_memory_used_mb" in sample
                and "utilization_pct" in sample
            }
            if set(generation.get("device_ids", [])) != set(expected_devices):
                device_valid = False
            if owned_devices != set(expected_devices):
                device_valid = False
        if not device_valid:
            _append_once(reasons, "gpu_device_ownership_incomplete")

        clocks = [
            (int(row["monotonic_start_ns"]), int(row["monotonic_end_ns"]))
            for row in model_rows
            if isinstance(row.get("monotonic_start_ns"), int)
            and isinstance(row.get("monotonic_end_ns"), int)
            and int(row["monotonic_end_ns"]) >= int(row["monotonic_start_ns"])
        ]
        if clocks:
            lifecycle_intervals.append(
                (min(start for start, _end in clocks), max(end for _start, end in clocks), model_id)
            )
        if (
            {"model_load", "generation", "teardown"} <= phases
            and cache_valid
            and lifecycle_valid
            and device_valid
        ):
            qualified_models.append(model_id)

    ordered = sorted(lifecycle_intervals)
    expected_order = [str(spec["hf_id"]) for spec in MODEL_SPECS]
    if [model_id for _start, _end, model_id in ordered] != expected_order:
        _append_once(reasons, "model_load_order_mismatch")
    for left, right in zip(ordered, ordered[1:], strict=False):
        if right[0] < left[1]:
            _append_once(reasons, "sequential_model_overlap")

    device_reasons = {
        "unresolved_gpu_identity",
        "gpu_uuid_mismatch",
        "gpu_sample_pid_mismatch",
        "gpu_sample_missing",
        "gpu_sample_field_missing",
        "gpu_sample_outside_phase",
        "gpu_telemetry_gap",
        "gpu_device_ownership_incomplete",
    }
    sequential_reasons = {
        "model_load_order_mismatch",
        "sequential_model_overlap",
        "overlap_unexplained",
        "invalid_monotonic_interval",
    }
    teardown_reasons = {reason for reason in reasons if "teardown" in reason}
    return {
        **base,
        "accepted": not reasons and len(qualified_models) == len(MODEL_SPECS),
        "reasons": reasons,
        "device_ownership_valid": not bool(device_reasons.intersection(reasons)),
        "sequential_model_lifecycle_valid": not bool(sequential_reasons.intersection(reasons)),
        "teardown_complete": base["teardown_complete"] and not teardown_reasons,
        "qualified_model_count": len(qualified_models),
        "qualified_model_ids": qualified_models,
    }


def replay_task_runtime_receipt(
    path: Path,
    *,
    task_pid: int,
    gpu_uuids: Sequence[str],
) -> JsonDict:
    """Read serialized rows and recompute all qualification properties."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    report = qualify_receipt_rows(
        rows,
        expected_task_pid=task_pid,
        expected_gpu_uuids=gpu_uuids,
    )
    if payload.get("receipt_sha256") != receipts.sha256_json(rows):
        _append_once(report["reasons"], "receipt_payload_hash_mismatch")
        report["accepted"] = False
    return report


def fresh_process_recheck(
    path: Path,
    *,
    task_pid: int,
    gpu_uuids: Sequence[str],
) -> JsonDict:
    """Use a new interpreter to replay the receipt from serialized rows."""

    program = (
        "import json,sys; "
        "from pathlib import Path; "
        "from carnot.experiment_6928_sota_runtime_receipt_qualification "
        "import replay_task_runtime_receipt as replay; "
        "print(json.dumps(replay(Path(sys.argv[1]), task_pid=int(sys.argv[2]), "
        "gpu_uuids=json.loads(sys.argv[3])), sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(path), str(task_pid), json.dumps(list(gpu_uuids))],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("COV_CORE_") and key != "COVERAGE_PROCESS_START"
        },
    )
    if completed.returncode != 0:
        return {
            "accepted": False,
            "returncode": completed.returncode,
            "reasons": ["fresh_process_replay_failed"],
            "stderr": completed.stderr.strip(),
        }
    report = json.loads(completed.stdout)
    report["returncode"] = completed.returncode
    return report


def forged_receipt_rejection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_pid: int,
    gpu_uuids: Sequence[str],
) -> list[JsonDict]:
    """Mutate four critical attribution fields and require rejection."""

    attacks: list[tuple[str, list[JsonDict]]] = []

    copied: list[JsonDict] = deepcopy(list(rows))
    copied[0]["task_id"] = "foreign-task-receipt"
    copied[0] = receipts.seal_adoption_row(copied[0])
    attacks.append(("copied_foreign_task_receipt", copied))

    cross_process: list[JsonDict] = deepcopy(list(rows))
    identity = dict(cross_process[0].get("task_process_identity", {}))
    identity["pid"] = task_pid + 1
    cross_process[0]["task_process_identity"] = identity
    cross_process[0] = receipts.seal_adoption_row(cross_process[0])
    attacks.append(("cross_process_receipt", cross_process))

    wrong_uuid: list[JsonDict] = deepcopy(list(rows))
    generation = next(row for row in wrong_uuid if row.get("phase") == "generation")
    generation["gpu_samples"][0]["device_uuid"] = "GPU-forged"
    index = wrong_uuid.index(generation)
    wrong_uuid[index] = receipts.seal_adoption_row(generation)
    attacks.append(("gpu_uuid_mismatch", wrong_uuid))

    no_teardown = [deepcopy(row) for row in rows if row.get("phase") != "teardown"]
    attacks.append(("missing_teardown", no_teardown))

    output: list[JsonDict] = []
    for attack_id, mutated in attacks:
        report = qualify_receipt_rows(
            mutated,
            expected_task_pid=task_pid,
            expected_gpu_uuids=gpu_uuids,
        )
        output.append(
            {
                "attack_id": attack_id,
                "rejected": not report["accepted"],
                "reasons": report["reasons"],
            }
        )
    return output


def _runner_selection(llama_status: Mapping[str, Any], command: Sequence[str]) -> JsonDict:
    """Bind the child interpreter, llama.cpp package, CUDA support, and command."""

    module_path = Path(str(llama_status.get("module_path") or ""))
    return {
        "runner_id": "llama-cpp-python-worker",
        "binary_path": sys.executable,
        "binary_sha256": receipts.sha256_file(sys.executable),
        "module_path": str(module_path) if str(module_path) != "." else None,
        "module_sha256": receipts.sha256_file(module_path),
        "llama_cpp_version": llama_status.get("version"),
        "supports_gpu_offload": llama_status.get("supports_gpu_offload"),
        "cuda_visible_devices": "0,1",
        "command": list(command),
        "command_sha256": receipts.sha256_json(list(command)),
        "substrate": "cuda_gguf",
        "selected": True,
    }


def _worker_command(model: Mapping[str, Any]) -> list[str]:
    """Build the exact child command for one bounded model lifecycle."""

    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--worker",
        "--model-path",
        str(model["model_path"]),
        "--context",
        str(CONTEXT_SIZE),
        "--offload-layers",
        str(OFFLOAD_LAYERS),
        "--seed",
        str(RANDOM_SEED),
    ]


def _emit_worker_event(event: str, **fields: Any) -> None:
    """Write one line so the parent can mark an exact worker phase boundary."""

    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def worker_main(*, model_path: str, context: int, offload_layers: int, seed: int) -> int:
    """Load, generate, and close only when the task parent requests each phase."""

    model: Any = None
    stage = "startup"
    _emit_worker_event("started", pid=os.getpid())
    try:
        if sys.stdin.readline().strip() != "load":
            raise RuntimeError("worker did not receive load command")
        stage = "model_load"
        from llama_cpp import Llama

        model = Llama(
            model_path=model_path,
            n_ctx=context,
            n_batch=128,
            n_ubatch=128,
            n_gpu_layers=offload_layers,
            tensor_split=[0.5, 0.5],
            seed=seed,
            verbose=False,
        )
        _emit_worker_event("loaded", pid=os.getpid())
        if sys.stdin.readline().strip() != "generate":
            raise RuntimeError("worker did not receive generate command")
        stage = "generation"
        result = model(
            "Reply with only the number: 2 + 2 =",
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            seed=seed,
            echo=False,
        )
        choice = result.get("choices", [{}])[0]
        _emit_worker_event(
            "generated",
            pid=os.getpid(),
            text=str(choice.get("text", "")),
            usage=result.get("usage", {}),
        )
        if sys.stdin.readline().strip() != "close":
            raise RuntimeError("worker did not receive close command")
        stage = "teardown"
        close = getattr(model, "close", None)
        if callable(close):
            close()
        model = None
        _emit_worker_event("closed", pid=os.getpid())
        return 0
    except Exception as exc:
        _emit_worker_event("error", stage=stage, error=repr(exc), pid=os.getpid())
        close = getattr(model, "close", None)
        if callable(close):
            close()
        return 1


def _send_worker_command(process: subprocess.Popen[str], command: str) -> None:
    """Send one phase command and fail if the child input pipe is unavailable."""

    if process.stdin is None:
        raise RuntimeError("worker stdin is unavailable")
    process.stdin.write(command + "\n")
    process.stdin.flush()


def _read_worker_event(process: subprocess.Popen[str], *, timeout_s: float) -> JsonDict | None:
    """Read one JSON event within a short bound, or return no event yet."""

    if process.stdout is None:
        raise RuntimeError("worker stdout is unavailable")
    ready, _write, _error = select.select([process.stdout], [], [], timeout_s)
    if not ready:
        return None
    line = process.stdout.readline()
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"worker returned non-JSON output: {line[:200]!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("worker event is not an object")
    return value


def _wait_worker_event(
    process: subprocess.Popen[str], expected_event: str, *, timeout_s: float
) -> JsonDict:
    """Wait for one named child event and surface child errors immediately."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        event = _read_worker_event(
            process, timeout_s=min(0.25, max(0.0, deadline - time.monotonic()))
        )
        if event is None:
            if process.poll() is not None:
                raise RuntimeError(
                    f"worker exited before {expected_event}: returncode={process.returncode}"
                )
            continue
        if event.get("event") == "error":
            raise RuntimeError(f"worker error during {event.get('stage')}: {event.get('error')}")
        if event.get("event") == expected_event:
            return event
    raise TimeoutError(f"worker timed out waiting for {expected_event}")


def sample_gpu_telemetry(child_pid: int, gpus: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Join device metrics to per-process residency for the task child PID."""

    device_result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: dict[str, tuple[int, int]] = {}
    if device_result["ok"]:
        for line in str(device_result["stdout"]).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                devices[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
            except ValueError:
                continue
    residency: dict[str, int] = {}
    if process_result["ok"]:
        for line in str(process_result["stdout"]).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3 or not parts[0].isdigit() or int(parts[0]) != child_pid:
                continue
            try:
                residency[parts[1]] = residency.get(parts[1], 0) + int(float(parts[2]))
            except ValueError:
                continue
    clock = time.monotonic_ns()
    return [
        {
            "pid": child_pid,
            "device_uuid": str(gpu["uuid"]),
            "pid_memory_mb": residency.get(str(gpu["uuid"]), 0),
            "device_memory_used_mb": devices.get(str(gpu["uuid"]), (0, 0))[0],
            "utilization_pct": devices.get(str(gpu["uuid"]), (0, 0))[1],
            "offload_layers": OFFLOAD_LAYERS,
            "monotonic_ns": clock,
            "sample_age_s": 0.0,
        }
        for gpu in gpus
    ]


def _wait_generation(
    process: subprocess.Popen[str],
    gpus: Sequence[Mapping[str, Any]],
    *,
    timeout_s: float,
) -> tuple[JsonDict, list[JsonDict]]:
    """Sample both GPUs until the worker returns its bounded generation."""

    deadline = time.monotonic() + timeout_s
    samples: list[JsonDict] = []
    while time.monotonic() < deadline:
        samples.extend(sample_gpu_telemetry(process.pid, gpus))
        event = _read_worker_event(process, timeout_s=TELEMETRY_INTERVAL_S)
        if event is None:
            if process.poll() is not None:
                raise RuntimeError(
                    f"worker exited during generation: returncode={process.returncode}"
                )
            continue
        if event.get("event") == "error":
            raise RuntimeError(f"worker error during {event.get('stage')}: {event.get('error')}")
        if event.get("event") == "generated":
            return event, samples
    raise TimeoutError("worker generation timed out")


def _post_teardown_gpu_state(child_pid: int, gpus: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Confirm that the reaped child has no remaining GPU process residency."""

    samples = sample_gpu_telemetry(child_pid, gpus)
    return {
        "child_pid_absent_from_compute_apps": all(
            int(row["pid_memory_mb"]) == 0 for row in samples
        ),
        "per_device_memory_used_mb": {
            str(row["device_uuid"]): int(row["device_memory_used_mb"]) for row in samples
        },
        "vram_after_teardown_mb": sum(int(row["device_memory_used_mb"]) for row in samples),
    }


def _new_template(work_dir: Path) -> Any:  # pragma: no cover - live integration boundary
    """Construct the shared template only for a live model subprocess."""

    from scripts.experiment_template import ExperimentTemplate

    return ExperimentTemplate(
        6928,
        "SOTA runtime receipt qualification",
        "results/unused_exp6928_worker.json",
        repo_root=work_dir,
        seed=RANDOM_SEED,
    )


def _execute_one_model(
    model: Mapping[str, Any],
    gpus: Sequence[Mapping[str, Any]],
    llama_status: Mapping[str, Any],
    work_dir: Path,
    model_index: int,
) -> JsonDict:
    """Run one controlled child lifecycle and return its existing receipt rows."""

    command = _worker_command(model)
    runner = _runner_selection(llama_status, command)
    receipt_path = work_dir / f"model-{model_index}-receipt.json"
    stderr_path = work_dir / f"model-{model_index}-stderr.log"
    template = _new_template(work_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    started_at = time.perf_counter()
    process: subprocess.Popen[str] | None = None
    error = ""
    output_event: JsonDict = {}
    post_teardown: JsonDict = {}
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            started_event = _wait_worker_event(process, "started", timeout_s=10.0)
            if int(started_event.get("pid", -1)) != process.pid:
                raise RuntimeError("worker start PID does not match the spawned child")
            server_id = f"exp6928-model-{model_index}-worker"
            with template.task_runtime_receipts(
                receipt_path,
                task_id=TASK_ID,
                control_id=f"model-{model_index}",
                runner_selection=runner,
                model_identity={
                    **dict(model),
                    "model_identity_bound": True,
                },
                device_ids=[str(gpu["uuid"]) for gpu in gpus],
                model_count=len(MODEL_SPECS),
                concurrency_group="exp6928-sequential-models",
                config={
                    "context_size": CONTEXT_SIZE,
                    "offload_layers": OFFLOAD_LAYERS,
                    "max_tokens": MAX_TOKENS,
                    "random_seed": RANDOM_SEED,
                    "tensor_split": [0.5, 0.5],
                    "child_command": command,
                },
            ) as runtime:
                with runtime.phase(
                    "model_load",
                    model_id=str(model["hf_id"]),
                    child_pids=[process.pid],
                    server_lifecycle={
                        "event": "started",
                        "server_id": server_id,
                        "pid": process.pid,
                    },
                    metadata={"child_command": command, "model_index": model_index},
                ) as state:
                    _send_worker_command(process, "load")
                    loaded = _wait_worker_event(process, "loaded", timeout_s=MODEL_TIMEOUT_S)
                    state["raw_output_bytes"] = json.dumps(loaded, sort_keys=True)
                with runtime.phase(
                    "generation",
                    model_id=str(model["hf_id"]),
                    child_pids=[process.pid],
                    telemetry_sample_gap_limit_s=5.0,
                    metadata={"child_command": command, "model_index": model_index},
                ) as state:
                    _send_worker_command(process, "generate")
                    output_event, samples = _wait_generation(
                        process, gpus, timeout_s=MODEL_TIMEOUT_S
                    )
                    state["raw_output_bytes"] = json.dumps(output_event, sort_keys=True)
                    state["gpu_samples"] = samples
                teardown_lifecycle = {
                    "event": "teardown",
                    "server_id": server_id,
                    "pid": process.pid,
                    "process_exit_confirmed": False,
                    "process_reaped": False,
                    "vram_after_teardown_mb": None,
                }
                with runtime.phase(
                    "teardown",
                    model_id=str(model["hf_id"]),
                    child_pids=[process.pid],
                    server_lifecycle=teardown_lifecycle,
                    metadata={"child_command": command, "model_index": model_index},
                ) as state:
                    _send_worker_command(process, "close")
                    closed = _wait_worker_event(process, "closed", timeout_s=30.0)
                    process.wait(timeout=30.0)
                    post_teardown = _post_teardown_gpu_state(process.pid, gpus)
                    teardown_lifecycle.update(
                        {
                            "process_exit_confirmed": process.returncode == 0,
                            "process_reaped": process.poll() is not None,
                            **post_teardown,
                        }
                    )
                    state["raw_output_bytes"] = json.dumps(closed, sort_keys=True)
                    state["exit_status"] = {
                        "returncode": process.returncode,
                        "timed_out": False,
                        "signal": None,
                    }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)

    receipt_payload: JsonDict = {}
    if receipt_path.is_file():
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows = receipt_payload.get("rows", [])
    stderr_hash = receipts.sha256_file(stderr_path)
    return {
        "hf_id": model["hf_id"],
        "family": model["family"],
        "model_path": model["model_path"],
        "model_file": model["model_file"],
        "model_sha256": model["model_sha256"],
        "quantization": model["quantization"],
        "cache_state": model["cache_state"],
        "runner_id": runner["runner_id"],
        "runner_binary": runner["binary_path"],
        "command": command,
        "command_sha256": runner["command_sha256"],
        "child_pid": process.pid if process is not None else None,
        "gpu_uuids": [str(gpu["uuid"]) for gpu in gpus],
        "offload_layers": OFFLOAD_LAYERS,
        "context_size": CONTEXT_SIZE,
        "generated_text": output_event.get("text"),
        "usage": output_event.get("usage", {}),
        "stderr_sha256": stderr_hash,
        "duration_s": round(time.perf_counter() - started_at, 9),
        "post_teardown": post_teardown,
        "returncode": process.returncode if process is not None else None,
        "status": "complete" if not error and process and process.returncode == 0 else "failed",
        "error": error or None,
        "rows": rows,
        "phase_timings": template._phase_timings,
        "runner_selection": runner,
    }


def execute_live_models(
    model_files: Sequence[Mapping[str, Any]],
    gpus: Sequence[Mapping[str, Any]],
    llama_status: Mapping[str, Any],
    work_dir: Path,
) -> JsonDict:
    """Load all three models sequentially and combine their receipt rows."""

    model_runs: list[JsonDict] = []
    all_rows: list[JsonDict] = []
    for index, model in enumerate(model_files):
        run_row = _execute_one_model(model, gpus, llama_status, work_dir, index)
        model_runs.append(run_row)
        all_rows.extend(run_row.pop("rows"))

    task_pid = os.getpid()
    task_identity = receipts.read_process_identity(task_pid)
    if task_identity is None:
        raise RuntimeError("task process identity became unavailable")
    validation = qualify_receipt_rows(
        all_rows,
        expected_task_pid=task_pid,
        expected_gpu_uuids=[str(gpu["uuid"]) for gpu in gpus],
    )
    payload = receipts.build_adoption_receipt(
        task_id=TASK_ID,
        task_process_identity=task_identity,
        rows=all_rows,
        validation=validation,
    )
    receipt_path = work_dir / "task-runtime-receipt.json"
    receipts.write_adoption_receipt(receipt_path, payload)
    return {
        "task_pid": task_pid,
        "model_rows": model_runs,
        "rows": all_rows,
        "validation": validation,
        "task_runtime_receipt": payload,
        "receipt_path": receipt_path,
    }


def _gate_rows(preconditions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert every failed preflight row to the required gate summary."""

    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in preconditions
        if row.get("passed") is not True
    ]


def _empty_artifact(date: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Create the complete blocked schema before any model work begins."""

    return {
        "schema": "carnot.sota_runtime_receipt_qualification.v1",
        "experiment_id": 6928,
        "run_date": date,
        "status": "blocked",
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "model_rows": [],
        "model_file_rows": [],
        "runner_selection_rows": [],
        "process_lineage_rows": [],
        "task_phase_timing_rows": [],
        "task_gpu_telemetry_rows": [],
        "model_concurrency_rows": [],
        "server_lifecycle_rows": [],
        "cache_state_rows": [],
        "teardown_rows": [],
        "fresh_process_recheck_rows": [],
        "forged_receipt_rejection_rows": [],
        "task_runtime_receipt": {},
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": receipts.sha256_json(
            {"date": date, "random_seed": RANDOM_SEED, "model_specs": MODEL_SPECS}
        )[7:23],
        "sota_runtime_receipt_ready_score": 0,
        "gate_check_summary": _gate_rows(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_sota_runtime_receipt_qualification",
    }


def blocked_artifact(*, date: str, preconditions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the exact terminal artifact for a failed precondition."""

    return _empty_artifact(date, preconditions)


def _source_hashes(repo_root: Path) -> JsonDict:
    """Hash the helper, template, experiment, wrapper, and prior receipt."""

    return {path.as_posix(): receipts.sha256_file(repo_root / path) for path in SOURCE_PATHS}


def _runtime_gate_rows(
    model_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    fresh: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Explain each runtime condition that prevents readiness."""

    gates: list[JsonDict] = []
    failed_models = [str(row.get("hf_id")) for row in model_rows if row.get("status") != "complete"]
    if failed_models:
        gates.append(
            {
                "failed_check": "all_three_model_runs_complete",
                "expected_value": [spec["hf_id"] for spec in MODEL_SPECS],
                "observed_value": failed_models,
            }
        )
    if validation.get("accepted") is not True:
        gates.append(
            {
                "failed_check": "task_owned_cuda_receipt_validation",
                "expected_value": [],
                "observed_value": validation.get("reasons", []),
            }
        )
    if fresh.get("accepted") is not True:
        gates.append(
            {
                "failed_check": "fresh_process_receipt_replay",
                "expected_value": True,
                "observed_value": fresh.get("reasons", fresh),
            }
        )
    false_accepts = [row.get("attack_id") for row in attacks if row.get("rejected") is not True]
    if false_accepts:
        gates.append(
            {
                "failed_check": "forged_receipts_fail_closed",
                "expected_value": [],
                "observed_value": false_accepts,
            }
        )
    return gates


def run(
    *,
    date: str,
    output_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    repo_root: Path = REPO_ROOT,
) -> JsonDict:
    """Run preflight, three sequential models, replay, attacks, and artifact write."""

    started = time.perf_counter()
    preflight = check_preconditions(repo_root=repo_root, output_path=output_path)
    checks = preflight["checks"]
    if any(row.get("passed") is not True for row in checks):
        artifact = blocked_artifact(date=date, preconditions=checks)
        artifact["duration_s"] = round(time.perf_counter() - started, 9)
        artifact["model_file_rows"] = preflight.get("model_files", [])
        artifact["cache_state_rows"] = [
            {
                "hf_id": row.get("hf_id"),
                "cache_state": row.get("cache_state"),
                "model_path": row.get("model_path"),
            }
            for row in preflight.get("model_files", [])
        ]
        artifact["source_artifact_hashes"] = _source_hashes(repo_root)
        receipts.write_json_atomic(output_path, artifact)
        return artifact

    with tempfile.TemporaryDirectory(prefix="carnot-exp6928-") as directory:
        execution = execute_live_models(
            preflight["model_files"],
            preflight["gpus"],
            preflight["llama_cpp"],
            Path(directory),
        )
        gpu_uuids = [str(gpu["uuid"]) for gpu in preflight["gpus"]]
        fresh = fresh_process_recheck(
            execution["receipt_path"],
            task_pid=execution["task_pid"],
            gpu_uuids=gpu_uuids,
        )
        attacks = forged_receipt_rejection_rows(
            execution["rows"],
            task_pid=execution["task_pid"],
            gpu_uuids=gpu_uuids,
        )

    model_rows = execution["model_rows"]
    rows = execution["rows"]
    validation = execution["validation"]
    ready = bool(
        len(model_rows) == len(MODEL_SPECS)
        and all(row.get("status") == "complete" for row in model_rows)
        and validation.get("accepted") is True
        and fresh.get("accepted") is True
        and all(row.get("rejected") is True for row in attacks)
    )
    source_hashes = _source_hashes(repo_root)
    artifact = _empty_artifact(date, checks)
    artifact.update(
        {
            "status": "complete" if ready else "blocked",
            "duration_s": round(time.perf_counter() - started, 9),
            "source_artifact_hashes": source_hashes,
            "rows": rows,
            "model_specs": [dict(row) for row in preflight["model_files"]],
            "model_rows": model_rows,
            "model_file_rows": [dict(row) for row in preflight["model_files"]],
            "runner_selection_rows": [dict(row["runner_selection"]) for row in model_rows],
            "process_lineage_rows": [
                {
                    "hf_id": row.get("model_identity", {}).get("hf_id"),
                    "phase": row.get("phase"),
                    "process_lineage": deepcopy(row.get("process_lineage", [])),
                }
                for row in rows
                if row.get("process_lineage")
            ],
            "task_phase_timing_rows": [
                {
                    "hf_id": model_row["hf_id"],
                    **dict(timing),
                }
                for model_row in model_rows
                for timing in model_row["phase_timings"]
            ],
            "task_gpu_telemetry_rows": [
                {
                    "hf_id": row.get("model_identity", {}).get("hf_id"),
                    "phase": row.get("phase"),
                    **dict(sample),
                }
                for row in rows
                for sample in row.get("gpu_samples", [])
            ],
            "model_concurrency_rows": [
                {
                    "declared_mode": "sequential",
                    "declared_model_count": len(MODEL_SPECS),
                    "load_order": [row["hf_id"] for row in model_rows],
                    "recomputed_peak_model_concurrency": fresh.get("peak_model_concurrency"),
                    "sequential_model_lifecycle_valid": fresh.get(
                        "sequential_model_lifecycle_valid"
                    ),
                }
            ],
            "server_lifecycle_rows": [
                {
                    "hf_id": row.get("model_identity", {}).get("hf_id"),
                    **dict(row["server_lifecycle"]),
                }
                for row in rows
                if row.get("server_lifecycle")
            ],
            "cache_state_rows": [
                {
                    "hf_id": row["hf_id"],
                    "cache_state": row["cache_state"],
                    "model_path": row["model_path"],
                    "model_sha256": row["model_sha256"],
                }
                for row in preflight["model_files"]
            ],
            "teardown_rows": [
                {
                    "hf_id": row.get("model_identity", {}).get("hf_id"),
                    **dict(row["server_lifecycle"]),
                }
                for row in rows
                if row.get("server_lifecycle", {}).get("event") == "teardown"
            ],
            "fresh_process_recheck_rows": [fresh],
            "forged_receipt_rejection_rows": attacks,
            "task_runtime_receipt": execution["task_runtime_receipt"],
            "reproducibility_checksum": receipts.sha256_json(
                {
                    "date": date,
                    "random_seed": RANDOM_SEED,
                    "model_files": preflight["model_files"],
                    "gpu_uuids": gpu_uuids,
                    "source_artifact_hashes": source_hashes,
                }
            )[7:23],
            "sota_runtime_receipt_ready_score": int(ready),
            "gate_check_summary": []
            if ready
            else _runtime_gate_rows(model_rows, validation, fresh, attacks),
            "verdict_class": "null" if ready else "blocked",
            "honest_verdict": "complete_sota_runtime_receipt_qualified"
            if ready
            else "blocked_sota_runtime_receipt_qualification",
        }
    )
    receipts.write_json_atomic(output_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Run either the internal worker protocol or the dated experiment."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--context", type=int, default=CONTEXT_SIZE)
    parser.add_argument("--offload-layers", type=int, default=OFFLOAD_LAYERS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.model_path:
            parser.error("--worker requires --model-path")
        return worker_main(
            model_path=args.model_path,
            context=args.context,
            offload_layers=args.offload_layers,
            seed=args.seed,
        )
    if not args.date:
        parser.error("--date is required")
    artifact = run(date=args.date, output_path=args.output)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "sota_runtime_receipt_ready_score": artifact["sota_runtime_receipt_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin command entry point
    raise SystemExit(main())
