"""Measure a task-owned load envelope for the three required GGUF families.

The controller stops before model work when another process owns GPU memory.
When the GPUs are free, each load runs in a fresh child. The child closes its
model and exits before the controller admits the next load.

Spec refs: REQ-INFRA-6966, SCENARIO-INFRA-6966-CONFIG,
SCENARIO-INFRA-6966-OWNERSHIP, SCENARIO-INFRA-6966-TEARDOWN, and
SCENARIO-INFRA-6966-BARE-GATES.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.task_runtime_receipts import sha256_file


JsonDict = dict[str, Any]
MODULE_NAME = "carnot.experiment_6966_gguf_load_envelope_canary"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6966_gguf_load_envelope_canary.json"
CHECKPOINT_PATH = (
    REPO_ROOT / "results/checkpoints/experiment_6966_gguf_load_envelope_canary/rows.json"
)
EXP6962_RESULT_PATH = REPO_ROOT / "results/experiment_6962_queue_regulated_self_learning.json"
RUN_DATE = "20260904"
RANDOM_SEED = 6_966_202_609_04
PREFERRED_QUANT = "Q4_K_M"
FIXED_PROMPT = (
    "In one concise paragraph, explain why a model process must exit before a second "
    "large model loads on the same GPUs."
)
MAX_OUTPUT_TOKENS = 128
MODEL_TIMEOUT_S = 1_800.0
VRAM_RELEASE_TIMEOUT_S = 180.0
VRAM_RELEASE_TOLERANCE_MB = 512
POLL_INTERVAL_S = 0.25
INFERENCE_SUBSTRATE = "live_local_llama_cpp_three_family_dual_cuda"

REQUIRED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

EXP6962_LOAD_CONFIG: JsonDict = {
    "config_id": "exp6962_exact_single_cuda",
    "changed_factor": None,
    "n_ctx": 8192,
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ubatch": 512,
    "main_gpu": 0,
    "split_mode": "layer",
    "tensor_split": None,
    "use_mmap": True,
    "use_mlock": False,
    "visible_devices": [0],
    "verbose": False,
}
PROMOTED_CONFIG_ID = "dual_cuda_even_split_ctx16384"
_CONFIG_COMPARE_FIELDS = (
    "n_ctx",
    "n_gpu_layers",
    "n_batch",
    "n_ubatch",
    "main_gpu",
    "split_mode",
    "tensor_split",
    "use_mmap",
    "use_mlock",
    "visible_devices",
)

REQUIRED_ARTIFACT_FIELDS = (
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
    "baseline_gpu_memory_rows",
    "reproduction_rows",
    "load_config_rows",
    "vocab_probe_rows",
    "live_generation_rows",
    "gpu_runtime_rows",
    "process_ownership_rows",
    "teardown_rows",
    "vram_release_rows",
    "checkpoint_rows",
    "gguf_load_canary_complete_score",
    "gguf_runtime_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks prevent a model load from stealing another process's GPUs.",
    "inference_substrate": "The exact substrate prevents CPU or remote work from counting as CUDA evidence.",
    "duration_s": "Measured total time exposes synthetic or truncated runs.",
    "live_duration_s": "Measured generation time separates live model work from preflight.",
    "source_artifact_hashes": "Source hashes bind the canary to the failed run and implementation bytes.",
    "MODEL_SPECS": "Exact resolved model declarations prevent silent family substitution.",
    "models_used": "The ordered hub IDs make three-family coverage explicit.",
    "model_file_hashes": "Full hashes bind each load to exact cached GGUF bytes.",
    "gpu_topology": "Device identity and capacity explain the tested memory envelope.",
    "baseline_gpu_memory_rows": "Baseline readings make later VRAM recovery falsifiable.",
    "reproduction_rows": "The original failing configuration must be observed before a fix is promoted.",
    "load_config_rows": "A frozen one-factor ladder identifies which memory change caused recovery.",
    "vocab_probe_rows": "Embedded-tokenizer evidence separates tokenizer health from weight-load health.",
    "live_generation_rows": "One terminal row per family prevents pooled readiness from hiding a failure.",
    "gpu_runtime_rows": "PID-linked CUDA samples prove that generation used the declared accelerators.",
    "process_ownership_rows": "PID lineage prevents an unrelated process from supplying the evidence.",
    "teardown_rows": "Exit and reap evidence proves the task ended only its own process.",
    "vram_release_rows": "Both devices must recover before another family can load safely.",
    "checkpoint_rows": "Atomic per-model checkpoints preserve completed evidence after interruption.",
    "gguf_load_canary_complete_score": "Completion requires a terminal row for every exact family.",
    "gguf_runtime_ready_score": "Readiness requires three live CUDA outputs plus clean exit and memory release.",
    "random_seed": "A fixed seed makes the bounded generation controls repeatable.",
    "reproducibility_checksum": "A content digest detects later changes to evidence or configuration.",
    "gate_check_summary": "Expected and observed values make every blocked result actionable.",
    "verifier_is_oracle": "False states that execution evidence does not judge semantic correctness.",
    "verdict_class": "A closed class keeps terminal automation states unambiguous.",
    "honest_verdict": "A class-consistent prefix reports whether the runtime ran or blocked.",
}

_ALLOCATION_RE = re.compile(
    r"allocating\s+([0-9]+(?:\.[0-9]+)?)\s+MiB\s+on\s+device\s+(\d+):\s*([^\n]+)",
    re.IGNORECASE,
)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU", re.I)


def canonical_json(value: Any) -> str:
    """Return one stable JSON spelling for hashes and worker payloads."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text after explicit UTF-8 encoding."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the digest that contains the hash."""

    return sha256_text(
        canonical_json(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
    )


def configuration_ladder() -> list[JsonDict]:
    """Return the preregistered one-factor memory ladder."""

    exact = deepcopy(EXP6962_LOAD_CONFIG)
    dual_visible = deepcopy(exact)
    dual_visible.update(
        {
            "config_id": "dual_cuda_auto_split_ctx8192",
            "changed_factor": "visible_devices",
            "visible_devices": [0, 1],
        }
    )
    even_split = deepcopy(dual_visible)
    even_split.update(
        {
            "config_id": "dual_cuda_even_split_ctx8192",
            "changed_factor": "tensor_split",
            "tensor_split": [0.5, 0.5],
        }
    )
    promoted = deepcopy(even_split)
    promoted.update(
        {
            "config_id": PROMOTED_CONFIG_ID,
            "changed_factor": "n_ctx",
            "n_ctx": 16_384,
        }
    )
    return [exact, dual_visible, even_split, promoted]


def configuration_ladder_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject drift in order, single-factor changes, or the promoted floor."""

    errors: list[str] = []
    expected_ids = [row["config_id"] for row in configuration_ladder()]
    if [row.get("config_id") for row in rows] != expected_ids:
        errors.append("ladder_config_order_mismatch")
    for previous, current in zip(rows, rows[1:], strict=False):
        changed = [
            field for field in _CONFIG_COMPARE_FIELDS if previous.get(field) != current.get(field)
        ]
        if len(changed) != 1 or current.get("changed_factor") != (changed[0] if changed else None):
            errors.append("ladder_multiple_factors_changed")
    if not rows or int(rows[-1].get("n_ctx", 0) or 0) < 16_384:
        errors.append("promoted_context_too_small")
    if not rows or rows[-1].get("visible_devices") != [0, 1]:
        errors.append("promoted_not_dual_cuda")
    return list(dict.fromkeys(errors))


def resolve_model_specs(
    *,
    cached_pair_func: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the canonical pair first, then add the missing dense family."""

    pair = cached_pair_func(gpu_indices=(0, 1)) or []
    pair_paths = {str(row.get("hf_id")): str(row.get("model_path") or "") for row in pair}
    rows: list[JsonDict] = []
    for model_id in REQUIRED_MODEL_IDS:
        path = pair_paths.get(model_id) or resolver(model_id, PREFERRED_QUANT) or ""
        rows.append(
            {
                "name": model_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
                "hf_id": model_id,
                "model_path": str(path),
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "preferred_quant": PREFERRED_QUANT,
                "resolution_method": "cached_sota_pair(gpu_indices=(0, 1))"
                if model_id in pair_paths
                else "resolve_cached_gguf dense extension",
            }
        )
    return rows


def model_spec_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require the exact ordered families and concrete primary GGUF paths."""

    errors: list[str] = []
    if [row.get("hf_id") for row in rows] != list(REQUIRED_MODEL_IDS):
        errors.append("model_ids_mismatch")
    for row in rows:
        model_id = str(row.get("hf_id", ""))
        path = str(row.get("model_path", ""))
        if not path:
            errors.append(f"model_path_missing:{model_id}")
        elif "mmproj" in Path(path).name.lower() or Path(path).suffix.lower() != ".gguf":
            errors.append(f"model_path_not_primary_gguf:{model_id}")
        if row.get("gpu_indices") != [0, 1]:
            errors.append(f"dual_gpu_indices_missing:{model_id}")
        if row.get("headline_eligible") is not True:
            errors.append(f"headline_eligibility_missing:{model_id}")
    return errors


MODEL_SPECS = resolve_model_specs()


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:
    """Run one read-only host probe and preserve its raw terminal state."""

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


def gpu_inventory() -> JsonDict:  # pragma: no cover - depends on the live NVIDIA host.
    """Read both device and compute-process tables from ``nvidia-smi``."""

    device_receipt = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    process_receipt = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        )
    )
    devices: list[JsonDict] = []
    for line in device_receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "utilization_gpu_pct": int(parts[6]),
                    "temperature_c": int(parts[7]),
                }
            )
        except ValueError:
            continue
    processes: list[JsonDict] = []
    for line in process_receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            processes.append(
                {
                    "pid": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "used_memory_mb": int(parts[2]),
                    "process_name": parts[3],
                }
            )
        except ValueError:
            continue
    return {
        "query_ok": device_receipt["passed"] is True and process_receipt["passed"] is True,
        "devices": devices,
        "processes": processes,
        "raw_receipts": [device_receipt, process_receipt],
    }


def llama_cpp_probe() -> JsonDict:  # pragma: no cover - depends on the native binding.
    """Confirm that the installed binding was built with CUDA offload."""

    try:
        from llama_cpp import __version__
        from llama_cpp import llama_cpp as backend

        return {
            "importable": True,
            "gpu_offload": bool(backend.llama_supports_gpu_offload()),
            "version": str(__version__),
        }
    except Exception as exc:
        return {
            "importable": False,
            "gpu_offload": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def checkpoint_is_writable(path: Path) -> bool:
    """Prove the checkpoint directory with a same-directory temporary file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".write-probe-", dir=path.parent)
        os.close(descriptor)
        Path(temporary_name).unlink()
        return True
    except OSError:
        return False


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
) -> JsonDict:
    """Collect every load gate before a child process can start."""

    gpu = gpu_probe()
    devices = list(gpu.get("devices", []))
    processes = list(gpu.get("processes", []))
    binding = llama_probe()
    checkpoint_writable = writable_probe(checkpoint_path)
    spec_errors = model_spec_errors(model_specs)
    file_rows = [
        {
            "model_id": row.get("hf_id"),
            "path": row.get("model_path"),
            "exists": Path(str(row.get("model_path", ""))).is_file(),
        }
        for row in model_specs
    ]
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
            {str(row["model_id"]): row["exists"] for row in file_rows},
            len(file_rows) == 3 and all(row["exists"] for row in file_rows),
        ),
        _check(
            "llama_cpp_cuda_bindings",
            {"importable": True, "gpu_offload": True},
            binding,
            binding.get("importable") is True and binding.get("gpu_offload") is True,
        ),
        _check("checkpoint_writable", True, checkpoint_writable, checkpoint_writable),
    ]
    return {
        "all_passed": all(row["passed"] is True for row in checks),
        "checks": checks,
        "gpu_topology": gpu,
        "baseline_gpu_memory_rows": [
            {
                "index": row.get("index"),
                "uuid": row.get("uuid"),
                "memory_used_mb": row.get("memory_used_mb"),
                "memory_free_mb": row.get("memory_free_mb"),
            }
            for row in devices
        ],
        "llama_cpp": binding,
        "checkpoint_path": str(checkpoint_path),
        "legacy_gpu_lease_path": str(REPO_ROOT / "ops/gpu-lease.json"),
        "legacy_gpu_lease_present": (REPO_ROOT / "ops/gpu-lease.json").is_file(),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return all checks and the first failed expected-observed pair."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def _proc_stat(pid: int) -> tuple[int, int] | None:
    """Read parent PID and start ticks from one Linux process identity."""

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[3]), int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def capture_process_ownership(pid: int, parent_pid: int, command: Sequence[str]) -> JsonDict:
    """Bind a live child PID and start identity to the task controller."""

    identity = _proc_stat(pid)
    observed_parent, start_ticks = identity if identity is not None else (None, None)
    return {
        "pid": int(pid),
        "expected_parent_pid": int(parent_pid),
        "parent_pid": observed_parent,
        "pid_start_ticks": start_ticks,
        "command": list(command),
        "command_hash": sha256_text(canonical_json(list(command))),
        "owned": identity is not None and observed_parent == int(parent_pid),
    }


def parse_allocation_request(stderr_text: str) -> JsonDict | None:
    """Extract the exact failed CUDA allocation from llama.cpp diagnostics."""

    matches = _ALLOCATION_RE.findall(stderr_text)
    if not matches:
        return None
    requested, device, error = matches[-1]
    return {"requested_mib": float(requested), "device": int(device), "error": error.strip()}


def parse_offloaded_layers(stderr_text: str) -> JsonDict:
    """Extract the final offloaded layer count from llama.cpp diagnostics."""

    matches = _OFFLOAD_RE.findall(stderr_text)
    if not matches:
        return {"offloaded": 0, "total": None}
    offloaded, total = matches[-1]
    return {"offloaded": int(offloaded), "total": int(total)}


def build_vram_release_row(
    *,
    model_id: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    after_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Require both GPU readings to return within the fixed tolerance."""

    before = {int(row["index"]): int(row["memory_used_mb"]) for row in baseline_rows}
    after = {int(row["index"]): int(row["memory_used_mb"]) for row in after_rows}
    required = sorted(before)
    missing = [index for index in required if index not in after]
    deltas = {str(index): after[index] - before[index] for index in required if index in after}
    max_increase = max(deltas.values(), default=0)
    return {
        "model_id": model_id,
        "baseline_memory_used_mb": {str(key): value for key, value in before.items()},
        "after_memory_used_mb": {str(key): value for key, value in after.items()},
        "increase_mb": deltas,
        "max_increase_mb": max_increase,
        "tolerance_mb": VRAM_RELEASE_TOLERANCE_MB,
        "missing_device_indices": missing,
        "passed": not missing and max_increase <= VRAM_RELEASE_TOLERANCE_MB,
    }


def embedded_tokenizer_probe(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Load only GGUF metadata and its embedded tokenizer, never weights."""

    started = time.perf_counter()
    try:
        from llama_cpp import Llama

        tokenizer = Llama(model_path=str(model["model_path"]), vocab_only=True, verbose=False)
        tokens = tokenizer.tokenize(b"GGUF envelope tokenizer probe")
        metadata = dict(getattr(tokenizer, "metadata", {}) or {})
        close = getattr(tokenizer, "close", None)
        if callable(close):
            close()
        return {
            "model_id": model["hf_id"],
            "passed": bool(tokens),
            "token_count": len(tokens),
            "tokenizer_source": "embedded_gguf",
            "metadata": metadata,
            "metadata_hash": sha256_text(canonical_json(metadata)),
            "duration_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "model_id": model["hf_id"],
            "passed": False,
            "token_count": 0,
            "tokenizer_source": "embedded_gguf",
            "metadata": {},
            "metadata_hash": sha256_text("{}"),
            "duration_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def worker_execute(
    payload: Mapping[str, Any],
    *,
    llama_factory: Callable[..., Any] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
) -> JsonDict:
    """Load, optionally generate, close, and collect inside one child process."""

    llm: Any = None
    started_ns = clock()
    config = dict(payload["config"])
    try:
        if llama_factory is None:
            from llama_cpp import Llama

            llama_factory = Llama
        kwargs: JsonDict = {
            "model_path": str(payload["model_path"]),
            "n_ctx": int(config["n_ctx"]),
            "n_gpu_layers": int(config["n_gpu_layers"]),
            "n_batch": int(config["n_batch"]),
            "n_ubatch": int(config["n_ubatch"]),
            "main_gpu": int(config["main_gpu"]),
            "split_mode": 1,
            "use_mmap": bool(config["use_mmap"]),
            "use_mlock": bool(config["use_mlock"]),
            "seed": RANDOM_SEED,
            # Diagnostics do not change allocation. They expose the failed CUDA request.
            "verbose": True,
        }
        if config.get("tensor_split") is not None:
            kwargs["tensor_split"] = list(config["tensor_split"])
        llm = llama_factory(**kwargs)
        loaded_ns = clock()
        output = ""
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str | None = None
        live_duration_s = 0.0
        if payload.get("generation") is True:
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
            finish_reason = str(choice.get("finish_reason") or "unknown")
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            if completion_tokens <= 0 and output:
                completion_tokens = len(llm.tokenize(output.encode("utf-8"), add_bos=False))
            live_duration_s = max(0.0, (ended_ns - loaded_ns) / 1_000_000_000)
        else:
            ended_ns = loaded_ns
        return {
            "model_id": payload["model_id"],
            "config_id": config["config_id"],
            "terminal_state": "complete",
            "load_duration_s": max(0.0, (loaded_ns - started_ns) / 1_000_000_000),
            "live_duration_s": live_duration_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": completion_tokens / live_duration_s
            if live_duration_s > 0
            else 0.0,
            "output": output,
            "output_hash": sha256_text(output),
            "finish_reason": finish_reason,
            "exception_type": None,
            "exception_message": None,
            "exception_traceback": None,
            "worker_started_ns": started_ns,
            "worker_ended_ns": ended_ns,
            "garbage_collection_ran": True,
            "teardown_complete": True,
        }
    except Exception as exc:
        return {
            "model_id": payload.get("model_id"),
            "config_id": config.get("config_id"),
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
            "worker_started_ns": started_ns,
            "worker_ended_ns": None,
            "garbage_collection_ran": True,
            "teardown_complete": True,
        }
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        llm = None
        gc.collect()


def _worker_main(payload_path: Path, output_path: Path) -> int:  # pragma: no cover
    """Execute the private child protocol and publish its terminal row."""

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    row = worker_execute(payload)
    write_json_atomic(output_path, row)
    return 0 if row["terminal_state"] == "complete" else 1


def _owned_process_absent(pid: int, start_ticks: int | None) -> bool:
    identity = _proc_stat(pid)
    return identity is None or identity[1] != start_ticks


def _terminate_owned_process(pid: int, parent_pid: int, start_ticks: int | None) -> bool:
    """Signal only the exact child identity created by this controller."""

    identity = _proc_stat(pid)
    if identity is None or identity != (parent_pid, start_ticks):
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


def _wait_for_vram_release(
    baseline_rows: Sequence[Mapping[str, Any]],
    model_id: str,
    *,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
    timeout_s: float = VRAM_RELEASE_TIMEOUT_S,
) -> JsonDict:  # pragma: no cover - live GPU polling.
    deadline = time.monotonic() + timeout_s
    last = gpu_probe()
    while time.monotonic() < deadline:
        row = build_vram_release_row(
            model_id=model_id,
            baseline_rows=baseline_rows,
            after_rows=_memory_rows(last),
        )
        if row["passed"] is True:
            row["wait_duration_s"] = max(0.0, timeout_s - (deadline - time.monotonic()))
            return row
        time.sleep(1.0)
        last = gpu_probe()
    row = build_vram_release_row(
        model_id=model_id,
        baseline_rows=baseline_rows,
        after_rows=_memory_rows(last),
    )
    row["wait_duration_s"] = timeout_s
    return row


def run_attempt(
    *,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    generation: bool,
    checkpoint_dir: Path,
    gpu_probe: Callable[[], JsonDict] = gpu_inventory,
) -> JsonDict:  # pragma: no cover - live subprocess and GPU integration.
    """Run one owned child while the parent samples its GPU residency."""

    attempt_id = f"{str(model['hf_id']).replace('/', '__')}__{config['config_id']}"
    attempt_dir = checkpoint_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload_path = attempt_dir / "payload.json"
    output_path = attempt_dir / "worker.json"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    payload = {
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "config": dict(config),
        "generation": bool(generation),
    }
    write_json_atomic(payload_path, payload)
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--worker-payload",
        str(payload_path),
        "--worker-output",
        str(output_path),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in config["visible_devices"])
    baseline_gpu = gpu_probe()
    baseline_rows = _memory_rows(baseline_gpu)
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
        ownership = capture_process_ownership(process.pid, os.getpid(), command)
        start_ticks = ownership.get("pid_start_ticks")
        samples: list[JsonDict] = []
        deadline = time.monotonic() + MODEL_TIMEOUT_S
        timed_out = False
        while process.poll() is None and time.monotonic() < deadline:
            sample = gpu_probe()
            sample["monotonic_ns"] = time.monotonic_ns()
            samples.append(sample)
            time.sleep(POLL_INTERVAL_S)
        if process.poll() is None:
            timed_out = True
            if _terminate_owned_process(process.pid, os.getpid(), start_ticks):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if capture_process_ownership(process.pid, os.getpid(), command)["owned"]:
                        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    if output_path.is_file():
        worker = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        worker = {
            "model_id": model["hf_id"],
            "config_id": config["config_id"],
            "terminal_state": "failed",
            "exception_type": "TimeoutError" if timed_out else "WorkerOutputMissing",
            "exception_message": "worker_timeout" if timed_out else "worker output missing",
            "exception_traceback": None,
            "output": "",
            "output_hash": sha256_text(""),
            "live_duration_s": 0.0,
            "tokens_per_second": 0.0,
            "teardown_complete": True,
        }
    owned_samples = [
        {"monotonic_ns": sample["monotonic_ns"], **process_row, "devices": sample["devices"]}
        for sample in samples
        for process_row in sample.get("processes", [])
        if process_row.get("pid") == process.pid
    ]
    layer_row = parse_offloaded_layers(stderr_text)
    release = _wait_for_vram_release(baseline_rows, str(model["hf_id"]), gpu_probe=gpu_probe)
    absent = _owned_process_absent(process.pid, start_ticks)
    live_cuda = bool(owned_samples) and layer_row["offloaded"] > 0
    row = {
        **dict(worker),
        "model_id": model["hf_id"],
        "model_path": model["model_path"],
        "model_file_hash": sha256_file(model["model_path"]),
        "config_id": config["config_id"],
        "load_config": dict(config),
        "generation_requested": bool(generation),
        "visible_devices": list(config["visible_devices"]),
        "allocation_request": parse_allocation_request(stderr_text),
        "offloaded_layers": layer_row["offloaded"],
        "total_layers": layer_row["total"],
        "gpu_samples": owned_samples,
        "max_gpu_utilization_pct": max(
            (
                int(device.get("utilization_gpu_pct", 0) or 0)
                for sample in samples
                for device in sample.get("devices", [])
            ),
            default=0,
        ),
        "live_cuda": live_cuda,
        "process_exit_code": process.returncode,
        "owned_process_absent": absent,
        "teardown_complete": worker.get("teardown_complete") is True and absent,
        "vram_release_passed": release["passed"] is True,
        "backend_stdout_hash": sha256_text(stdout_text),
        "backend_stderr_hash": sha256_text(stderr_text),
        "backend_stderr": stderr_text,
        "process_ownership": ownership,
        "baseline_gpu_memory_rows": baseline_rows,
        "vram_release": release,
    }
    return row


def reduce_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Derive bare completion and live-runtime scores from exact family rows."""

    exact_order = [row.get("model_id") for row in rows] == list(REQUIRED_MODEL_IDS)
    terminal = exact_order and all(
        row.get("terminal_state") in {"complete", "failed", "blocked"} for row in rows
    )
    complete_score = int(terminal)
    ready = terminal and all(
        row.get("config_id") == PROMOTED_CONFIG_ID
        and row.get("terminal_state") == "complete"
        and row.get("live_cuda") is True
        and bool(str(row.get("output", "")))
        and row.get("output_hash") == sha256_text(str(row.get("output", "")))
        and row.get("process_exit_code") == 0
        and row.get("owned_process_absent") is True
        and row.get("teardown_complete") is True
        and row.get("vram_release_passed") is True
        and int(row.get("offloaded_layers", 0) or 0) > 0
        and float(row.get("live_duration_s", 0.0) or 0.0) > 0
        and float(row.get("tokens_per_second", 0.0) or 0.0) > 0
        for row in rows
    )
    return complete_score, int(ready)


def _source_hashes() -> dict[str, str | None]:
    paths = {
        "experiment_6962": EXP6962_RESULT_PATH,
        "module": Path(__file__),
        "wrapper": REPO_ROOT / "scripts/experiments/experiment_6966_gguf_load_envelope_canary.py",
        "tests": REPO_ROOT / "tests/python/test_experiment_6966_gguf_load_envelope_canary.py",
        "spec": REPO_ROOT / "openspec/capabilities/research-harnesses/spec.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    live_duration_s: float,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    source_artifact_hashes: Mapping[str, Any] | None = None,
    model_file_hashes: Mapping[str, Any] | None = None,
    reproduction_rows: Sequence[Mapping[str, Any]] = (),
    load_config_rows: Sequence[Mapping[str, Any]] | None = None,
    vocab_probe_rows: Sequence[Mapping[str, Any]] = (),
    live_generation_rows: Sequence[Mapping[str, Any]] = (),
    gpu_runtime_rows: Sequence[Mapping[str, Any]] = (),
    process_ownership_rows: Sequence[Mapping[str, Any]] = (),
    teardown_rows: Sequence[Mapping[str, Any]] = (),
    vram_release_rows: Sequence[Mapping[str, Any]] = (),
    checkpoint_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build a complete terminal artifact from measured rows only."""

    generations = [deepcopy(dict(row)) for row in live_generation_rows]
    complete_score, ready_score = reduce_scores(generations)
    preflight_passed = preconditions.get("all_passed") is True
    if not preflight_passed:
        verdict_class = "blocked"
        verdict = "blocked_gguf_load_envelope_canary"
    elif ready_score == 1:
        verdict_class = "positive"
        verdict = "complete: all three GGUF families generated on dual CUDA and released VRAM"
    elif complete_score == 1:
        verdict_class = "null"
        verdict = "complete_null_gguf_load_envelope_canary"
    else:
        verdict_class = "partial"
        verdict = "partial_gguf_load_envelope_canary"
    artifact: JsonDict = {
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": float(live_duration_s),
        "source_artifact_hashes": dict(source_artifact_hashes or {}),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "models_used": [str(row.get("hf_id")) for row in model_specs],
        "model_file_hashes": dict(model_file_hashes or {}),
        "gpu_topology": deepcopy(dict(preconditions.get("gpu_topology", {}))),
        "baseline_gpu_memory_rows": deepcopy(
            list(preconditions.get("baseline_gpu_memory_rows", []))
        ),
        "reproduction_rows": [deepcopy(dict(row)) for row in reproduction_rows],
        "load_config_rows": [
            deepcopy(dict(row)) for row in (load_config_rows or configuration_ladder())
        ],
        "vocab_probe_rows": [deepcopy(dict(row)) for row in vocab_probe_rows],
        "live_generation_rows": generations,
        "gpu_runtime_rows": [deepcopy(dict(row)) for row in gpu_runtime_rows],
        "process_ownership_rows": [deepcopy(dict(row)) for row in process_ownership_rows],
        "teardown_rows": [deepcopy(dict(row)) for row in teardown_rows],
        "vram_release_rows": [deepcopy(dict(row)) for row in vram_release_rows],
        "checkpoint_rows": [deepcopy(dict(row)) for row in checkpoint_rows],
        "gguf_load_canary_complete_score": complete_score,
        "gguf_runtime_ready_score": ready_score,
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
    """Recompute fields, gates, verdict, and checksum without trusting headlines."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(f"missing_field:{field}" for field in missing)
    if missing:
        return errors
    if set(dict(artifact.get("field_principles", {}))) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("models_used") != list(REQUIRED_MODEL_IDS):
        errors.append("models_used_mismatch")
    if [row.get("hf_id") for row in artifact.get("MODEL_SPECS", [])] != list(REQUIRED_MODEL_IDS):
        errors.append("model_specs_mismatch")
    errors.extend(configuration_ladder_errors(artifact.get("load_config_rows", [])))
    derived_complete, derived_ready = reduce_scores(artifact.get("live_generation_rows", []))
    for field, expected in (
        ("gguf_load_canary_complete_score", derived_complete),
        ("gguf_runtime_ready_score", derived_ready),
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
    if (
        float(artifact.get("duration_s", -1.0)) < 0
        or float(artifact.get("live_duration_s", -1.0)) < 0
    ):
        errors.append("duration_invalid")
    preflight_passed = artifact.get("preconditions_checked", {}).get("all_passed") is True
    verdict_class = artifact.get("verdict_class")
    verdict = str(artifact.get("honest_verdict", ""))
    if not preflight_passed:
        summary = artifact.get("gate_check_summary", {})
        if verdict_class != "blocked" or verdict != "blocked_gguf_load_envelope_canary":
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
    return errors


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Sync a complete temporary JSON file before atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def checkpoint_model_row(path: Path, manifest_hash: str, row: Mapping[str, Any]) -> JsonDict:
    """Append one terminal family row while refusing manifest or payload drift."""

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


def _model_file_hashes(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    return {str(row["hf_id"]): sha256_file(row["model_path"]) for row in model_specs}


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preflight_fn: Callable[[Sequence[Mapping[str, Any]], Path], JsonDict] | None = None,
    attempt_runner: Callable[..., JsonDict] = run_attempt,
) -> JsonDict:
    """Preflight, reproduce, test the ladder, and run three families in order."""

    started = time.perf_counter()
    specs = [deepcopy(dict(row)) for row in (model_specs or MODEL_SPECS)]
    if preflight_fn is None:
        preconditions = collect_preconditions(model_specs=specs, checkpoint_path=checkpoint_path)
    else:
        preconditions = preflight_fn(specs, checkpoint_path)
    source_hashes = _source_hashes()
    file_hashes = _model_file_hashes(specs)
    if preconditions.get("all_passed") is not True:
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            live_duration_s=0.0,
            model_specs=specs,
            preconditions=preconditions,
            source_artifact_hashes=source_hashes,
            model_file_hashes=file_hashes,
        )
        write_json_atomic(result_path, artifact)
        return artifact

    vocab_rows = [embedded_tokenizer_probe(model) for model in specs]
    if not all(row["passed"] is True for row in vocab_rows):
        checks = list(preconditions["checks"])
        checks.append(
            _check(
                "embedded_tokenizer_probes",
                {model_id: True for model_id in REQUIRED_MODEL_IDS},
                {row["model_id"]: row["passed"] for row in vocab_rows},
                False,
            )
        )
        preconditions = {**dict(preconditions), "all_passed": False, "checks": checks}
        artifact = build_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            live_duration_s=0.0,
            model_specs=specs,
            preconditions=preconditions,
            source_artifact_hashes=source_hashes,
            model_file_hashes=file_hashes,
            vocab_probe_rows=vocab_rows,
        )
        write_json_atomic(result_path, artifact)
        return artifact

    attempt_dir = checkpoint_path.parent
    ladder = configuration_ladder()
    reproduction = attempt_runner(
        model=specs[0],
        config=ladder[0],
        generation=False,
        checkpoint_dir=attempt_dir,
    )
    reproduction_rows = [reproduction]
    all_attempts = [reproduction]
    generation_rows: list[JsonDict] = []
    release_allows_next = reproduction.get("vram_release_passed") is True
    qwen_terminal: JsonDict | None = None
    if release_allows_next:
        for config in ladder[1:]:
            row = attempt_runner(
                model=specs[0],
                config=config,
                generation=config["config_id"] == PROMOTED_CONFIG_ID,
                checkpoint_dir=attempt_dir,
            )
            all_attempts.append(row)
            if row.get("vram_release_passed") is not True:
                release_allows_next = False
                qwen_terminal = row
                break
            if config["config_id"] == PROMOTED_CONFIG_ID:
                qwen_terminal = row
        if qwen_terminal is not None:
            generation_rows.append(qwen_terminal)
    for model in specs[1:]:
        if not release_allows_next:
            break
        row = attempt_runner(
            model=model,
            config=ladder[-1],
            generation=True,
            checkpoint_dir=attempt_dir,
        )
        all_attempts.append(row)
        generation_rows.append(row)
        release_allows_next = row.get("vram_release_passed") is True

    manifest_hash = sha256_text(canonical_json({"models": specs, "ladder": ladder}))
    checkpoint_rows: list[JsonDict] = []
    for row in generation_rows:
        checkpoint_rows.append(checkpoint_model_row(checkpoint_path, manifest_hash, row))
    live_duration = sum(float(row.get("live_duration_s", 0.0) or 0.0) for row in all_attempts)
    artifact = build_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        live_duration_s=live_duration,
        model_specs=specs,
        preconditions=preconditions,
        source_artifact_hashes=source_hashes,
        model_file_hashes=file_hashes,
        reproduction_rows=reproduction_rows,
        load_config_rows=ladder,
        vocab_probe_rows=vocab_rows,
        live_generation_rows=generation_rows,
        gpu_runtime_rows=[
            {
                "model_id": row.get("model_id"),
                "config_id": row.get("config_id"),
                "gpu_samples": row.get("gpu_samples", []),
                "max_gpu_utilization_pct": row.get("max_gpu_utilization_pct", 0),
                "offloaded_layers": row.get("offloaded_layers", 0),
            }
            for row in all_attempts
        ],
        process_ownership_rows=[dict(row.get("process_ownership", {})) for row in all_attempts],
        teardown_rows=[
            {
                "model_id": row.get("model_id"),
                "config_id": row.get("config_id"),
                "process_exit_code": row.get("process_exit_code"),
                "owned_process_absent": row.get("owned_process_absent"),
                "garbage_collection_ran": row.get("garbage_collection_ran"),
                "passed": row.get("teardown_complete") is True,
            }
            for row in all_attempts
        ],
        vram_release_rows=[dict(row.get("vram_release", {})) for row in all_attempts],
        checkpoint_rows=checkpoint_rows,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact_validation_failed:{errors}")
    write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - command surface.
    """Run the public canary, private worker, or artifact validator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--worker-payload", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.worker_payload is not None:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-payload")
        return _worker_main(args.worker_payload, args.worker_output)
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
                "gguf_load_canary_complete_score": artifact["gguf_load_canary_complete_score"],
                "gguf_runtime_ready_score": artifact["gguf_runtime_ready_score"],
                "honest_verdict": artifact["honest_verdict"],
                "validation_errors": errors,
            }
        )
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - exercised through the public wrapper.
    raise SystemExit(main())
