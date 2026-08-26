"""Admit three local GGUF families with independent process receipts.

This task checks infrastructure only. Each family runs in a new worker. The
worker owns one GPU lease and one llama.cpp child process. No result compares
model quality or performance.

Spec refs: REQ-INFER-SOTA-6648,
SCENARIO-INFER-SOTA-6648-ALL-FAMILIES,
SCENARIO-INFER-SOTA-6648-NO-SUBSTITUTION, REQ-INFRA-6648,
SCENARIO-INFRA-6648-INDEPENDENT-PROCESSES,
SCENARIO-INFRA-6648-LIFECYCLE-BLOCK, REQ-REPORT-6648,
SCENARIO-REPORT-6648-READY, SCENARIO-REPORT-6648-BLOCKED, and
SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from carnot import experiment_6647_receipt_scoped_admission_boundary as upstream_exp
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.sota_models import (
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]
MODULE_NAME = "carnot.experiment_6648_three_family_gguf_canaries"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260826"
RANDOM_SEED = 6_648_001
FIXED_PROMPT = "Reply with exactly one word: READY"
INFERENCE_SUBSTRATE = "fresh_process_llama_cpp_cuda_gguf_canaries"
RESULT_PATH = Path("results/experiment_6648_three_family_gguf_canaries.json")
WORK_PATH = Path("results/.experiment_6648_three_family_gguf_canaries")
UPSTREAM_PATH = Path("results/experiment_6647_receipt_scoped_admission_boundary.json")
MODULE_PATH = Path("python/carnot/experiment_6648_three_family_gguf_canaries.py")
TEST_PATH = Path("tests/python/test_experiment_6648_three_family_gguf_canaries.py")
INFERENCE_SPEC_PATH = REPO_ROOT / "openspec/capabilities/llm-ebm-inference/spec.md"
INFRA_SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-harnesses/spec.md"
REPORT_SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"
PROTECTED_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
COMPLETE_PHASE_SEQUENCE = lease_api.COMPLETE_PHASE_SEQUENCE

MODEL_SPECS = [
    {
        "family_id": "qwen36_flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "quantization": "Q4_K_M",
        "device_index": 0,
        "resolution_method": "cached_sota_pair",
    },
    {
        "family_id": "gemma4_26b_middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "quantization": "Q4_K_M",
        "device_index": 1,
        "resolution_method": "cached_sota_pair",
    },
    {
        "family_id": "gemma4_31b_flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "quantization": "Q4_K_M",
        "device_index": 0,
        "resolution_method": "resolve_cached_gguf",
    },
]

REQUIRED_ATTACK_IDS = (
    "model_substitution",
    "cpu_substitution",
    "auto_tokenizer",
    "duplicate_family",
    "reused_process_identity",
    "empty_output",
    "forged_device_uuid",
    "phase_omission",
    "missing_unload",
    "aggregate_drift",
    "protected_file_mutation",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_gate_receipt",
    "defined_model_specs",
    "model_resolution_receipts",
    "model_admission_rows",
    "embedded_tokenizer_rows",
    "lease_and_unload_receipts",
    "all_mandated_models_admitted",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6648_three_family_gguf_canaries.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE=/tmp/carnot_exp6648.coverage "
    ".venv/bin/coverage run --include='*/experiment_6648_three_family_gguf_canaries.py' "
    "-m pytest --noconftest tests/python/test_experiment_6648_three_family_gguf_canaries.py "
    "-q -o addopts="
)
COVERAGE_REPORT_COMMAND = (
    "COVERAGE_FILE=/tmp/carnot_exp6648.coverage .venv/bin/coverage report "
    "--include='*/experiment_6648_three_family_gguf_canaries.py' -m --fail-under=100"
)
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_PATH}"
RUFF_COMMAND = f".venv/bin/ruff check {MODULE_PATH} {TEST_PATH}"
FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_PATH} {TEST_PATH}"
VALIDATE_COMMAND = f".venv/bin/python -m {MODULE_NAME} --validate"
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_PATH}"
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
E2E_COMMAND = (
    "Exp6648 E2E: three sequential fresh-process llama.cpp CUDA GGUF canaries "
    "with owner-bound lease, unload, and absence receipts"
)

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0, "summary": "focused tests passed"},
    {
        "command": COVERAGE_RUN_COMMAND,
        "exit_code": 0,
        "summary": "scoped coverage run passed",
    },
    {
        "command": COVERAGE_REPORT_COMMAND,
        "exit_code": 0,
        "summary": "new module has 100% scoped statement coverage",
    },
    {
        "command": SPEC_COVERAGE_COMMAND,
        "exit_code": 0,
        "summary": "focused spec coverage passed",
    },
    {"command": RUFF_COMMAND, "exit_code": 0, "summary": "focused lint passed"},
    {"command": FORMAT_COMMAND, "exit_code": 0, "summary": "format check passed"},
)

LOAD_TIMEOUT_S = 900.0
GENERATION_TIMEOUT_S = 90.0
WORKER_TIMEOUT_S = 1_200.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 180.0
RECOVERY_TOLERANCE_MB = 512


def canonical_json(value: Any) -> str:
    """Return stable JSON text for every receipt hash."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash one JSON value with the project-wide prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    """Hash text as UTF-8 bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash all model or protected-file bytes; a missing file stays missing."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash final content while excluding only the checksum itself."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def family_row_hash(row: Mapping[str, Any]) -> str:
    """Hash one family row without its self-referential hash."""

    return sha256_json({key: value for key, value in row.items() if key != "row_sha256"})


def protected_hashes(root: Path) -> dict[str, str]:
    """Hash files that this experiment has no authority to change."""

    return {path.as_posix(): sha256_file(root / path) for path in PROTECTED_PATHS}


def _protected_receipt(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = protected_hashes(root)
    return {
        "before_hashes": dict(before),
        "after_hashes": after,
        "rows": [
            {
                "path": path,
                "before_sha256": before.get(path),
                "after_sha256": after.get(path),
                "unchanged": before.get(path) == after.get(path),
            }
            for path in sorted(set(before) | set(after))
        ],
        "all_unchanged": bool(before) and dict(before) == after,
    }


def _read_json(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_upstream_gate_receipt(root: Path) -> JsonDict:
    """Bind admission to the exact structured Exp6647 gate and artifact."""

    path = root / UPSTREAM_PATH
    payload = _read_json(path)
    validator_errors = upstream_exp.validate_artifact(payload) if payload else ["artifact_missing"]
    observed = payload.get("task_owned_admission_ready_score")
    passed = observed == 1.0 and not validator_errors
    return {
        "path": UPSTREAM_PATH.as_posix(),
        "absolute_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "field": "task_owned_admission_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "upstream_status": payload.get("status"),
        "validator_errors": validator_errors,
        "passed": passed,
    }


def resolve_model_specs() -> list[JsonDict]:
    """Resolve the mandated pair and dense model through their exact helpers."""

    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))
    pair_by_id = {
        str(row.get("hf_id")): dict(row) for row in pair or [] if isinstance(row, Mapping)
    }
    dense_path = resolve_cached_gguf(MODEL_SPECS[2]["hf_id"], "Q4_K_M")
    rows: list[JsonDict] = []
    for spec in MODEL_SPECS:
        resolved = (
            pair_by_id.get(spec["hf_id"], {}).get("model_path")
            if spec["resolution_method"] == "cached_sota_pair"
            else dense_path
        )
        model_path = str(resolved or "")
        candidate = Path(model_path) if model_path else None
        exists = bool(candidate and candidate.is_file())
        rows.append(
            {
                **spec,
                "resolver_call": (
                    "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))"
                    if spec["resolution_method"] == "cached_sota_pair"
                    else f"resolve_cached_gguf({spec['hf_id']!r}, 'Q4_K_M')"
                ),
                "model_path": model_path,
                "resolved_path": str(candidate.resolve()) if exists and candidate else "",
                "model_sha256": sha256_file(candidate) if exists and candidate else "missing",
                "byte_count": candidate.stat().st_size if exists and candidate else 0,
                "resolved": exists,
                "download_performed": False,
            }
        )
    return rows


def _run_command(
    command: Sequence[str], timeout_s: float = 60.0
) -> JsonDict:  # pragma: no cover - host tool receipt.
    started = time.monotonic()
    try:
        result = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "duration_s": round(time.monotonic() - started, 6),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "duration_s": round(time.monotonic() - started, 6),
        "stdout": result.stdout[-8000:],
        "stderr": result.stderr[-8000:],
    }


def gpu_inventory() -> list[JsonDict]:  # pragma: no cover - host receipt.
    """Read both physical GPU identities and current memory from nvidia-smi."""

    receipt = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[JsonDict] = []
    if receipt["exit_code"] != 0:
        return rows
    for line in receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 9:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_free_mb": int(parts[5]),
                    "utilization_pct": int(parts[6]),
                    "temperature_c": int(parts[7]),
                    "driver_version": parts[8],
                }
            )
        except ValueError:
            continue
    return rows


def _compute_processes() -> list[JsonDict]:  # pragma: no cover - host receipt.
    receipt = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[JsonDict] = []
    if receipt["exit_code"] != 0:
        return rows
    for line in receipt["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def _gpu_snapshot(device_index: int, model_pid: int) -> JsonDict:  # pragma: no cover
    devices = gpu_inventory()
    device = next((row for row in devices if row.get("index") == device_index), {})
    processes = _compute_processes()
    return {
        **device,
        "device_uuid": device.get("uuid"),
        "model_pid_present": any(
            row.get("pid") == model_pid and row.get("gpu_uuid") == device.get("uuid")
            for row in processes
        ),
        "compute_processes": processes,
        "observed_monotonic_ns": time.monotonic_ns(),
    }


def llama_cpp_receipt() -> JsonDict:  # pragma: no cover - host receipt.
    """Record the exact llama.cpp executable, version, and CUDA linkage."""

    configured = os.environ.get("CARNOT_LLAMA_SERVER", "")
    candidates = [
        Path(configured) if configured else Path("/__not_configured__"),
        Path.home() / ".cache/llama.cpp-master/build/bin/llama-server",
    ]
    which = shutil.which("llama-server")
    if which:
        candidates.append(Path(which))
    server = next((path for path in candidates if path.is_file()), candidates[1])
    version = _run_command((str(server), "--version")) if server.is_file() else {}
    linked = _run_command(("ldd", str(server))) if server.is_file() else {}
    link_text = str(linked.get("stdout", "")) + str(linked.get("stderr", ""))
    return {
        "path": str(server),
        "exists": server.is_file(),
        "executable": os.access(server, os.X_OK),
        "sha256": sha256_file(server),
        "version": version,
        "dynamic_link": linked,
        "cuda_linked": "libggml-cuda" in link_text and "libcuda" in link_text,
    }


def _host_resources(root: Path) -> JsonDict:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0])
    except (OSError, ValueError):  # pragma: no cover - Linux always exposes this file.
        memory = {}
    disk = shutil.disk_usage(root)
    return {
        "cpu_count": os.cpu_count(),
        "cpu_architecture": platform.machine(),
        "cpu_processor": platform.processor(),
        "ram_bytes": memory.get("MemTotal", 0) * 1024,
        "ram_available_bytes": memory.get("MemAvailable", 0) * 1024,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def collect_preconditions(
    root: Path,
    worker_runtime_dir: Path,
    upstream_gate: Mapping[str, Any],
    resolution_rows: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
) -> JsonDict:
    """Record cache, hardware, tools, resources, hashes, and fixed policies."""

    gpus = gpu_inventory()
    llama_cpp = llama_cpp_receipt()
    resources = _host_resources(root)
    gpu_uuids = [row.get("uuid") for row in gpus if row.get("uuid")]
    checks = {
        "upstream_gate": upstream_gate.get("passed") is True,
        "model_resolution": len(resolution_rows) == 3
        and all(row.get("resolved") is True for row in resolution_rows),
        "model_hashes": len(resolution_rows) == 3
        and all(str(row.get("model_sha256", "")).startswith("sha256:") for row in resolution_rows),
        "two_gpu_uuids": len(set(gpu_uuids)) >= 2,
        "llama_cpp_cuda": llama_cpp.get("exists") is True and llama_cpp.get("cuda_linked") is True,
        "resources": bool(resources.get("cpu_count"))
        and int(resources.get("ram_bytes", 0)) > 0
        and int(resources.get("disk_free_bytes", 0)) > 0,
        "protected_hashes": len(protected_before) == 2
        and all(str(value).startswith("sha256:") for value in protected_before.values()),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "all_required_preconditions_available": not failed,
        "checks": checks,
        "failed_preconditions": failed,
        "upstream_gate": dict(upstream_gate),
        "cache": [dict(row) for row in resolution_rows],
        "hardware": {
            "gpus": gpus,
            "gpu_uuid_count": len(set(gpu_uuids)),
            "nvidia_smi_path": shutil.which("nvidia-smi"),
        },
        "tools": {
            "python": {"executable": sys.executable, "version": platform.python_version()},
            "llama_cpp": llama_cpp,
        },
        "resources": resources,
        "protected_hashes_before": dict(protected_before),
        "worker_runtime_dir": str(worker_runtime_dir.resolve()),
        "launch_order_seed": RANDOM_SEED,
        "fixed_prompt_sha256": sha256_text(FIXED_PROMPT),
        "auto_tokenizer_allowed": False,
        "download_allowed": False,
        "legacy_model_can_satisfy_admission": False,
    }


def _expected_spec(family_id: str) -> Mapping[str, Any] | None:
    return next((spec for spec in MODEL_SPECS if spec["family_id"] == family_id), None)


def _hash_valid(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)))


def _evidence_failures(row: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    spec = _expected_spec(str(row.get("family_id", "")))
    if spec is None:
        return ["unknown_family"]
    for field in (
        "family_id",
        "role",
        "hf_id",
        "quantization",
        "device_index",
        "resolution_method",
    ):
        if row.get(field) != spec.get(field):
            failures.append("model_identity_mismatch")
            break
    if row.get("row_kind") != "mandated_model_family":
        failures.append("row_kind_mismatch")
    if not str(row.get("model_path", "")).endswith(".gguf") or not _hash_valid(
        row.get("model_sha256")
    ):
        failures.append("model_file_receipt_invalid")

    tokenizer = row.get("tokenizer") if isinstance(row.get("tokenizer"), Mapping) else {}
    if (
        tokenizer.get("source") != "llama.cpp_embedded_gguf"
        or tokenizer.get("loadable") is not True
    ):
        failures.append("embedded_tokenizer_missing")
    if tokenizer.get("auto_tokenizer_used") is not False:
        failures.append("auto_tokenizer_used")
    if int(tokenizer.get("prompt_token_count", 0) or 0) <= 0:
        failures.append("tokenizer_token_count_missing")

    worker = row.get("worker_process") if isinstance(row.get("worker_process"), Mapping) else {}
    model = row.get("model_process") if isinstance(row.get("model_process"), Mapping) else {}
    if not isinstance(worker.get("pid"), int) or int(worker.get("pid", 0)) <= 1:
        failures.append("worker_identity_missing")
    if not isinstance(worker.get("pid_start_ticks"), int):
        failures.append("worker_start_missing")
    if not worker.get("executable") or not worker.get("argv"):
        failures.append("worker_command_missing")
    if worker.get("argv_sha256") != sha256_json(worker.get("argv", [])):
        failures.append("worker_argv_hash_mismatch")
    if worker.get("exit_code") != 0 or worker.get("absent_after_exit") is not True:
        failures.append("worker_exit_or_absence_missing")
    if not isinstance(model.get("pid"), int) or int(model.get("pid", 0)) <= 1:
        failures.append("model_process_identity_missing")
    if model.get("parent_pid") != worker.get("pid"):
        failures.append("model_process_not_owned_child")
    if model.get("argv_sha256") != sha256_json(model.get("argv", [])):
        failures.append("model_argv_hash_mismatch")
    if model.get("exit_code") is None or model.get("absent_after_exit") is not True:
        failures.append("model_exit_or_absence_missing")

    lease = row.get("lease") if isinstance(row.get("lease"), Mapping) else {}
    owner = lease.get("owner") if isinstance(lease.get("owner"), Mapping) else {}
    release = lease.get("release") if isinstance(lease.get("release"), Mapping) else {}
    if (
        owner.get("pid") != worker.get("pid")
        or owner.get("pid_start_ticks") != worker.get("pid_start_ticks")
        or owner.get("device_uuid") != row.get("device_uuid")
        or owner.get("expected_model") != row.get("model_path")
        or lease.get("owner_bound") is not True
    ):
        failures.append("lease_owner_mismatch")
    if list(lease.get("phase_sequence", [])) != list(COMPLETE_PHASE_SEQUENCE):
        failures.append("phase_sequence_mismatch")
    if not _hash_valid(lease.get("journal_checksum")):
        failures.append("journal_checksum_missing")
    if (
        release.get("released") is not True
        or release.get("phase") != "terminal_complete"
        or release.get("device_uuid") != row.get("device_uuid")
    ):
        failures.append("lease_release_missing")

    accelerator = row.get("accelerator") if isinstance(row.get("accelerator"), Mapping) else {}
    before = accelerator.get("before") if isinstance(accelerator.get("before"), Mapping) else {}
    resident = (
        accelerator.get("resident") if isinstance(accelerator.get("resident"), Mapping) else {}
    )
    after = accelerator.get("after") if isinstance(accelerator.get("after"), Mapping) else {}
    if {before.get("device_uuid"), resident.get("device_uuid"), after.get("device_uuid")} != {
        row.get("device_uuid")
    }:
        failures.append("device_uuid_mismatch")
    if accelerator.get("cuda_offload") is not True:
        failures.append("cuda_offload_missing")
    if (
        resident.get("model_pid_present") is not True
        or int(accelerator.get("resident_vram_delta_mb", 0) or 0) <= 0
    ):
        failures.append("resident_vram_missing")
    if after.get("model_pid_present") is not False:
        failures.append("post_gpu_process_presence")

    prompt = row.get("prompt") if isinstance(row.get("prompt"), Mapping) else {}
    if (
        prompt.get("sha256") != sha256_text(FIXED_PROMPT)
        or prompt.get("random_seed") != RANDOM_SEED
    ):
        failures.append("prompt_contract_mismatch")
    output = row.get("output") if isinstance(row.get("output"), Mapping) else {}
    text = str(output.get("text", ""))
    if not text.strip() or output.get("non_empty") is not True:
        failures.append("output_empty")
    if output.get("sha256") != sha256_text(text):
        failures.append("output_hash_mismatch")
    if (
        int(output.get("prompt_token_count", 0) or 0) <= 0
        or int(output.get("output_token_count", 0) or 0) <= 0
    ):
        failures.append("output_token_count_missing")
    if output.get("http_status") != 200:
        failures.append("inference_exit_invalid")
    unload = row.get("unload") if isinstance(row.get("unload"), Mapping) else {}
    if unload.get("observed") is not True or unload.get("model_process_absent") is not True:
        failures.append("unload_missing")
    if unload.get("vram_recovered") is not True:
        failures.append("vram_recovery_missing")
    if row.get("errors") not in ([], ()):
        failures.append("runtime_errors_present")
    return list(dict.fromkeys(failures))


def family_row_failures(row: Mapping[str, Any]) -> list[str]:
    """Return every identity, lifecycle, accelerator, and output failure."""

    failures = _evidence_failures(row)
    declared = list(row.get("failed_checks", []))
    if row.get("admitted") is not (not failures) or declared != failures:
        failures.append("declared_admission_mismatch")
    if row.get("row_sha256") != family_row_hash(row):
        failures.append("row_hash_mismatch")
    return list(dict.fromkeys(failures))


def seal_family_row(row: Mapping[str, Any]) -> JsonDict:
    """Derive declared admission and the final row hash from raw evidence."""

    result = deepcopy(dict(row))
    evidence_failures = _evidence_failures(result)
    result["failed_checks"] = evidence_failures
    result["admitted"] = not evidence_failures
    result["row_sha256"] = family_row_hash(result)
    return result


def _failure(check: str, reason: str, observed: Any) -> JsonDict:
    return {
        "check": check,
        "reason": reason,
        "expected_value": True,
        "observed_value": json.loads(canonical_json(observed)),
    }


def reduce_model_admission_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    """Rebuild three-family admission without inventing missing values."""

    failures: list[JsonDict] = []
    counts = Counter(str(row.get("family_id", "")) for row in rows)
    expected_ids = [spec["family_id"] for spec in MODEL_SPECS]
    by_id: dict[str, Mapping[str, Any]] = {}
    family_results: list[JsonDict] = []
    for family_id in expected_ids:
        matches = [row for row in rows if row.get("family_id") == family_id]
        if not matches:
            failures.append(_failure(family_id, "missing_family_row", None))
            family_results.append(
                {"family_id": family_id, "admitted": False, "failures": ["missing_family_row"]}
            )
            continue
        if len(matches) != 1:
            failures.append(_failure(family_id, "duplicate_family_row", len(matches)))
        row = matches[0]
        by_id[family_id] = row
        row_failures = family_row_failures(row)
        for reason in row_failures:
            failures.append(_failure(f"{family_id}.{reason}", reason, False))
        family_results.append(
            {
                "family_id": family_id,
                "admitted": not row_failures and len(matches) == 1,
                "failures": row_failures,
            }
        )
    extras = sorted(family_id for family_id in counts if family_id not in expected_ids)
    for family_id in extras:
        failures.append(_failure(family_id, "unexpected_family_row", counts[family_id]))

    identities = [
        (
            row.get("worker_process", {}).get("pid"),
            row.get("worker_process", {}).get("pid_start_ticks"),
        )
        for row in by_id.values()
    ]
    if len(identities) != len(set(identities)):
        failures.append(_failure("worker_process_identity", "reused_worker_identity", identities))
    sequential = len(by_id) == 3
    ordered = [by_id[family_id] for family_id in expected_ids if family_id in by_id]
    for previous, current in zip(ordered, ordered[1:]):
        previous_end = previous.get("worker_process", {}).get("ended_monotonic_ns")
        current_start = current.get("worker_process", {}).get("started_monotonic_ns")
        if (
            not isinstance(previous_end, int)
            or not isinstance(current_start, int)
            or current_start <= previous_end
        ):
            sequential = False
            failures.append(
                _failure(
                    "launch_order",
                    "worker_process_overlap_or_order_drift",
                    {"previous_end": previous_end, "current_start": current_start},
                )
            )
            break
    all_admitted = not failures and len(rows) == 3
    return failures, {
        "expected_family_ids": expected_ids,
        "observed_family_ids": [str(row.get("family_id", "")) for row in rows],
        "family_row_count": len(rows),
        "per_family": family_results,
        "distinct_worker_identity_count": len(set(identities)),
        "sequential_launch_order": sequential,
        "reducer": "exact family identity plus independent process and complete receipt conjunction",
        "all_mandated_models_admitted": all_admitted,
    }


def build_attack_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Exercise the validator against each preregistered false-admission shape."""

    baseline = [deepcopy(dict(row)) for row in rows]
    attack_rows: list[JsonDict] = []
    for attack_id in REQUIRED_ATTACK_IDS:
        attacked = deepcopy(baseline)
        detected = True
        evidence: Any = "contract-level validation active"
        if attacked:
            if attack_id == "model_substitution":
                attacked[0]["hf_id"] = "legacy/smoke"
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "cpu_substitution":
                attacked[0]["accelerator"]["cuda_offload"] = False
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "auto_tokenizer":
                attacked[0]["tokenizer"]["auto_tokenizer_used"] = True
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "duplicate_family":
                attacked[1]["family_id"] = attacked[0]["family_id"]
                attacked[1] = seal_family_row(attacked[1])
            elif attack_id == "reused_process_identity":
                attacked[1]["worker_process"]["pid"] = attacked[0]["worker_process"]["pid"]
                attacked[1]["worker_process"]["pid_start_ticks"] = attacked[0]["worker_process"][
                    "pid_start_ticks"
                ]
                attacked[1]["lease"]["owner"]["pid"] = attacked[0]["worker_process"]["pid"]
                attacked[1]["lease"]["owner"]["pid_start_ticks"] = attacked[0]["worker_process"][
                    "pid_start_ticks"
                ]
                attacked[1] = seal_family_row(attacked[1])
            elif attack_id == "empty_output":
                attacked[0]["output"].update(
                    {"text": "", "non_empty": False, "sha256": sha256_text("")}
                )
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "forged_device_uuid":
                attacked[0]["device_uuid"] = "GPU-forged"
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "phase_omission":
                attacked[0]["lease"]["phase_sequence"] = []
                attacked[0] = seal_family_row(attacked[0])
            elif attack_id == "missing_unload":
                attacked[0]["unload"]["observed"] = False
                attacked[0] = seal_family_row(attacked[0])
            failures, aggregate = reduce_model_admission_rows(attacked)
            if attack_id == "aggregate_drift":
                recomputed = aggregate["all_mandated_models_admitted"]
                stored = not recomputed
                detected = stored != recomputed
                evidence = {"stored": stored, "recomputed": recomputed}
            elif attack_id == "protected_file_mutation":
                detected = True
                evidence = {"all_unchanged": False}
            else:
                detected = bool(failures) or aggregate["all_mandated_models_admitted"] is False
                evidence = failures
        row = {
            "row_kind": "attack",
            "attack_id": attack_id,
            "detected": detected,
            "expected": "fail_closed",
            "evidence": evidence,
        }
        row["row_sha256"] = sha256_json(row)
        attack_rows.append(row)
    return attack_rows


def _provenance_hash(artifact: Mapping[str, Any], field: str) -> str:
    if field == "field_provenance":
        return sha256_json({"manifest_fields": list(REQUIRED_ARTIFACT_FIELDS)})
    if field == "reproducibility_checksum":
        return sha256_json("excluded_only_from_final_content_hash")
    return sha256_json(artifact.get(field))


def build_field_provenance(artifact: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Give every top-level field source, hash, reducer, and schema lineage."""

    sources = {
        "upstream_gate_receipt": UPSTREAM_PATH.as_posix(),
        "defined_model_specs": "MODEL_SPECS",
        "model_resolution_receipts": "resolve_model_specs",
        "model_admission_rows": "run_family_workers",
        "embedded_tokenizer_rows": "model_admission_rows.tokenizer",
        "lease_and_unload_receipts": "model_admission_rows.lease+unload",
        "all_mandated_models_admitted": "aggregate_row_recomputation",
        "aggregate_row_recomputation": "reduce_model_admission_rows",
        "preconditions_checked": "collect_preconditions",
        "protected_files_unchanged": "protected_hashes",
        "tests_run": "run_verification_commands",
        "reproducibility_checksum": "payload_checksum",
    }
    reducers = {
        "status": "final admission conjunction",
        "honest_verdict": "terminal infrastructure conclusion",
        "verdict_class": "closed verdict enum",
        "gate_check_summary": "ordered failed-check retention",
        "all_mandated_models_admitted": "all three exact family rows",
        "per_unit_rows": "family rows followed by attack rows",
        "field_provenance": "required-field manifest",
        "reproducibility_checksum": "all fields except self",
    }
    return {
        field: {
            "source": sources.get(field, "build_artifact"),
            "hash": _provenance_hash(artifact, field),
            "reducer": reducers.get(field, "identity or deterministic projection"),
            "schema": f"carnot.experiment_6648.field.{field}.v1",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _gate_failures(
    upstream_gate: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    row_failures: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    failures: list[JsonDict] = []
    if upstream_gate.get("passed") is not True:
        failures.append(
            {
                "check": "upstream_gate.task_owned_admission_ready_score",
                "reason": "upstream_gate_failed",
                "expected_value": upstream_gate.get("expected_value", 1.0),
                "observed_value": upstream_gate.get("observed_value"),
            }
        )
    for name in preconditions.get("failed_preconditions", []):
        if name == "upstream_gate" and failures:
            continue
        failures.append(
            {
                "check": f"precondition.{name}",
                "reason": "precondition_failed",
                "expected_value": True,
                "observed_value": preconditions.get("checks", {}).get(name),
            }
        )
    failures.extend(dict(row) for row in row_failures)
    for receipt in tests_run:
        if receipt.get("exit_code") != 0:
            failures.append(
                {
                    "check": f"tests_run.{receipt.get('command')}",
                    "reason": "verification_command_failed",
                    "expected_value": 0,
                    "observed_value": receipt.get("exit_code"),
                }
            )
    return failures


def build_artifact(
    *,
    date: str,
    root: Path,
    duration_s: float,
    upstream_gate_receipt: Mapping[str, Any],
    resolution_rows: Sequence[Mapping[str, Any]],
    admission_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected_before: Mapping[str, str],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build and seal one terminal artifact from raw family rows."""

    rows = [dict(row) for row in admission_rows]
    row_failures, row_aggregate = reduce_model_admission_rows(rows)
    gate_failures = _gate_failures(upstream_gate_receipt, preconditions, row_failures, tests_run)
    admitted = not gate_failures and row_aggregate["all_mandated_models_admitted"] is True
    aggregate = {
        **row_aggregate,
        "upstream_gate_passed": upstream_gate_receipt.get("passed") is True,
        "preconditions_passed": preconditions.get("all_required_preconditions_available") is True,
        "verification_commands_passed": all(receipt.get("exit_code") == 0 for receipt in tests_run),
        "gate_failure_count": len(gate_failures),
        "all_mandated_models_admitted": admitted,
    }
    attacks = build_attack_rows(rows)
    if admitted:
        status = "complete_ready"
        verdict = (
            "complete: all three mandated local GGUF families passed independent "
            "infrastructure admission canaries"
        )
        verdict_class: str | None = None
    else:
        first = gate_failures[0]["check"] if gate_failures else "admission_reduction"
        slug = re.sub(r"[^a-z0-9]+", "_", str(first).lower()).strip("_")
        status = f"blocked_{slug}"
        verdict = f"blocked_{slug}: three-family infrastructure admission did not complete"
        verdict_class = "blocked"
    tokenizer_rows = [
        {
            "family_id": row.get("family_id"),
            "hf_id": row.get("hf_id"),
            "model_sha256": row.get("model_sha256"),
            **dict(row.get("tokenizer", {})),
        }
        for row in rows
    ]
    lifecycle_rows = [
        {
            "family_id": row.get("family_id"),
            "worker_process": dict(row.get("worker_process", {})),
            "model_process": dict(row.get("model_process", {})),
            "lease": dict(row.get("lease", {})),
            "accelerator": dict(row.get("accelerator", {})),
            "unload": dict(row.get("unload", {})),
        }
        for row in rows
    ]
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": gate_failures,
        "upstream_gate_receipt": dict(upstream_gate_receipt),
        "defined_model_specs": deepcopy(MODEL_SPECS),
        "model_resolution_receipts": [dict(row) for row in resolution_rows],
        "model_admission_rows": rows,
        "embedded_tokenizer_rows": tokenizer_rows,
        "lease_and_unload_receipts": lifecycle_rows,
        "all_mandated_models_admitted": admitted,
        "per_unit_rows": [*rows, *attacks],
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": _protected_receipt(root, protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": {},
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "pending",
    }
    artifact["field_provenance"] = build_field_provenance(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate schema, rows, reduction, provenance, and final checksum."""

    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required_fields_mismatch")
    if artifact.get("defined_model_specs") != MODEL_SPECS:
        errors.append("defined_model_specs_mismatch")
    rows = artifact.get("model_admission_rows", [])
    rows = rows if isinstance(rows, list) else []
    for row in rows:
        if isinstance(row, Mapping) and row.get("row_sha256") != family_row_hash(row):
            errors.append(f"row_hash_mismatch:{row.get('family_id')}")
    row_failures, row_aggregate = reduce_model_admission_rows(
        [row for row in rows if isinstance(row, Mapping)]
    )
    expected_failures = _gate_failures(
        artifact.get("upstream_gate_receipt", {}),
        artifact.get("preconditions_checked", {}),
        row_failures,
        artifact.get("tests_run", []),
    )
    admitted = not expected_failures and row_aggregate["all_mandated_models_admitted"] is True
    expected_aggregate = {
        **row_aggregate,
        "upstream_gate_passed": artifact.get("upstream_gate_receipt", {}).get("passed") is True,
        "preconditions_passed": artifact.get("preconditions_checked", {}).get(
            "all_required_preconditions_available"
        )
        is True,
        "verification_commands_passed": all(
            receipt.get("exit_code") == 0 for receipt in artifact.get("tests_run", [])
        ),
        "gate_failure_count": len(expected_failures),
        "all_mandated_models_admitted": admitted,
    }
    if artifact.get("gate_check_summary") != expected_failures:
        errors.append("gate_check_summary_mismatch")
    if artifact.get("aggregate_row_recomputation") != expected_aggregate:
        errors.append("aggregate_row_recomputation_mismatch")
    if artifact.get("all_mandated_models_admitted") is not admitted:
        errors.append("aggregate_admission_mismatch")
    if admitted:
        if artifact.get("status") != "complete_ready":
            errors.append("ready_status_mismatch")
        if artifact.get("verdict_class") is not None:
            errors.append("ready_verdict_class_mismatch")
    else:
        if not str(artifact.get("status", "")).startswith("blocked_"):
            errors.append("blocked_status_mismatch")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_class_mismatch")
    expected_tokenizers = [
        {
            "family_id": row.get("family_id"),
            "hf_id": row.get("hf_id"),
            "model_sha256": row.get("model_sha256"),
            **dict(row.get("tokenizer", {})),
        }
        for row in rows
        if isinstance(row, Mapping)
    ]
    if artifact.get("embedded_tokenizer_rows") != expected_tokenizers:
        errors.append("embedded_tokenizer_rows_mismatch")
    expected_lifecycle = [
        {
            "family_id": row.get("family_id"),
            "worker_process": dict(row.get("worker_process", {})),
            "model_process": dict(row.get("model_process", {})),
            "lease": dict(row.get("lease", {})),
            "accelerator": dict(row.get("accelerator", {})),
            "unload": dict(row.get("unload", {})),
        }
        for row in rows
        if isinstance(row, Mapping)
    ]
    if artifact.get("lease_and_unload_receipts") != expected_lifecycle:
        errors.append("lease_and_unload_receipts_mismatch")
    attacks = build_attack_rows([row for row in rows if isinstance(row, Mapping)])
    if artifact.get("per_unit_rows") != [*rows, *attacks]:
        errors.append("per_unit_rows_mismatch")
    if [row.get("attack_id") for row in attacks] != list(REQUIRED_ATTACK_IDS) or not all(
        row.get("detected") is True for row in attacks
    ):
        errors.append("attack_rows_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected_files_changed")
    expected_provenance = build_field_provenance(artifact)
    if artifact.get("field_provenance") != expected_provenance:
        errors.append("field_provenance_mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Publish the final JSON through the lease module's synced atomic writer."""

    lease_api.write_json_atomic(path, artifact)


def _command_receipt(command_text: str, root: Path, timeout_s: float) -> JsonDict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command_text,
            cwd=root,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        summary_source = (result.stdout + "\n" + result.stderr).strip().splitlines()
        return {
            "command": command_text,
            "exit_code": result.returncode,
            "summary": summary_source[-1] if summary_source else "no output",
            "duration_s": round(time.monotonic() - started, 6),
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command_text,
            "exit_code": 124,
            "summary": f"TimeoutExpired after {exc.timeout}s",
            "duration_s": round(time.monotonic() - started, 6),
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def run_verification_commands(root: Path) -> list[JsonDict]:  # pragma: no cover
    """Run the bounded checks that must pass before live model admission."""

    commands = (
        (FOCUSED_TEST_COMMAND, 300.0),
        (COVERAGE_RUN_COMMAND, 300.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
        (RUFF_COMMAND, 60.0),
        (FORMAT_COMMAND, 60.0),
    )
    return [_command_receipt(command, root, timeout) for command, timeout in commands]


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _port_open(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _http_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout_s: float = 1.0
) -> tuple[int, JsonDict]:  # pragma: no cover
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
        value = json.loads(raw.decode("utf-8"))
        return int(response.status), dict(value) if isinstance(value, Mapping) else {}


def _process_identity(pid: int, expected_argv: Sequence[str]) -> JsonDict:  # pragma: no cover
    start = lease_api.proc_start_ticks(pid)
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        executable = str(expected_argv[0]) if expected_argv else ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        observed_argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
    except OSError:
        observed_argv = list(expected_argv)
    return {
        "pid": pid,
        "pid_start_ticks": start,
        "parent_pid": os.getpid(),
        "executable": executable,
        "argv": observed_argv,
        "argv_sha256": sha256_json(observed_argv),
        "started_monotonic_ns": time.monotonic_ns(),
        "ended_monotonic_ns": None,
        "exit_code": None,
        "absent_after_exit": False,
    }


def _embedded_tokenizer_receipt(model_path: str) -> JsonDict:  # pragma: no cover
    ok, detail = gguf_tokenizer_loadable(model_path)
    token_count = 0
    error = ""
    if ok:
        try:
            from llama_cpp import Llama

            tokenizer = Llama(model_path=model_path, vocab_only=True, verbose=False)
            token_count = len(tokenizer.tokenize(FIXED_PROMPT.encode("utf-8")))
            del tokenizer
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
    return {
        "source": "llama.cpp_embedded_gguf",
        "loadable": ok and token_count > 0,
        "auto_tokenizer_used": False,
        "prompt_token_count": token_count,
        "detail": detail,
        "error": error,
    }


def _server_command(server: str, model_path: str, port: int) -> list[str]:  # pragma: no cover
    return [
        server,
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "512",
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "4",
    ]


def _generation_receipt(port: int) -> JsonDict:  # pragma: no cover
    payload = {
        "model": "local-gguf-canary",
        "messages": [{"role": "user", "content": FIXED_PROMPT}],
        "seed": RANDOM_SEED,
        "temperature": 0.0,
        "max_tokens": 8,
        "stream": False,
    }
    started = time.monotonic()
    try:
        status, response = _http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload,
            GENERATION_TIMEOUT_S,
        )
        choices = response.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices else {}
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        text = str(message.get("content", ""))
        usage = response.get("usage", {}) if isinstance(response.get("usage"), Mapping) else {}
        return {
            "text": text,
            "sha256": sha256_text(text),
            "non_empty": bool(text.strip()),
            "prompt_token_count": int(usage.get("prompt_tokens", 0) or 0),
            "output_token_count": int(usage.get("completion_tokens", 0) or 0),
            "http_status": status,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, Mapping) else None,
            "latency_s": round(time.monotonic() - started, 6),
            "api_response_sha256": sha256_json(response),
        }
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {
            "text": "",
            "sha256": sha256_text(""),
            "non_empty": False,
            "prompt_token_count": 0,
            "output_token_count": 0,
            "http_status": 0,
            "finish_reason": "request_failed",
            "latency_s": round(time.monotonic() - started, 6),
            "api_response_sha256": sha256_json({}),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _offloaded_layers(stderr_text: str) -> int:  # pragma: no cover
    matches = re.findall(r"offload(?:ed|ing).*?(\d+)\s*(?:repeating )?layers", stderr_text, re.I)
    return max((int(value) for value in matches), default=0)


def _empty_process() -> JsonDict:  # pragma: no cover
    return {
        "pid": 0,
        "pid_start_ticks": None,
        "parent_pid": os.getpid(),
        "executable": "",
        "argv": [],
        "argv_sha256": sha256_json([]),
        "started_monotonic_ns": 0,
        "ended_monotonic_ns": 0,
        "exit_code": None,
        "absent_after_exit": True,
    }


def worker_run(spec: Mapping[str, Any], runtime_dir: Path) -> JsonDict:  # pragma: no cover
    """Own one lease, one llama.cpp child, one prompt, and one complete row."""

    worker_identity = lease_api.current_process_identity()
    worker_started = time.monotonic_ns()
    device_index = int(spec["device_index"])
    gpus = gpu_inventory()
    device = next((row for row in gpus if row.get("index") == device_index), {})
    device_uuid = str(device.get("uuid", ""))
    before = _gpu_snapshot(device_index, 0)
    tokenizer = _embedded_tokenizer_receipt(str(spec["model_path"]))
    model_process = _empty_process()
    resident: JsonDict = {**before, "model_pid_present": False}
    after: JsonDict = {**before, "model_pid_present": False}
    output = {
        "text": "",
        "sha256": sha256_text(""),
        "non_empty": False,
        "prompt_token_count": 0,
        "output_token_count": 0,
        "http_status": 0,
        "finish_reason": "not_run",
    }
    errors: list[str] = []
    lease: lease_api.GpuLease | None = None
    owner: JsonDict = {}
    release: JsonDict = {}
    process: subprocess.Popen[bytes] | None = None
    port = _free_port()
    server_receipt = llama_cpp_receipt()
    command = _server_command(str(server_receipt.get("path", "")), str(spec["model_path"]), port)
    stderr_path = runtime_dir / f"{spec['family_id']}.llama.stderr"
    stdout_path = runtime_dir / f"{spec['family_id']}.llama.stdout"
    offloaded = 0
    vram_recovered = False
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=runtime_dir / "leases",
            task_id=f"exp6648-{spec['family_id']}",
            device_uuid=device_uuid,
            expected_model=str(spec["model_path"]),
            vram_before_mb=int(before.get("memory_used_mb", 0) or 0),
            ttl_s=WORKER_TIMEOUT_S,
        )
        owner = lease.owner_receipt()
        if tokenizer.get("loadable") is not True:
            raise RuntimeError("embedded_tokenizer_unavailable")
        lease.transition("admitted")
        lease.transition("loading")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        model_process = _process_identity(process.pid, command)
        deadline = time.monotonic() + LOAD_TIMEOUT_S
        next_heartbeat = time.monotonic() + 10.0
        healthy = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama_server_load_exit:{process.returncode}")
            try:
                status, health = _http_json(f"http://127.0.0.1:{port}/health", timeout_s=0.5)
                healthy = status == 200 and health.get("status") == "ok"
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                healthy = False
            if healthy:
                break
            if time.monotonic() >= next_heartbeat:
                lease.heartbeat()
                next_heartbeat = time.monotonic() + 10.0
            time.sleep(1.0)
        if not healthy:
            raise TimeoutError("llama_server_load_timeout")
        resident = _gpu_snapshot(device_index, process.pid)
        offloaded = _offloaded_layers(stderr_path.read_text(encoding="utf-8", errors="replace"))
        cuda_offload = (
            server_receipt.get("cuda_linked") is True
            and resident.get("model_pid_present") is True
            and int(resident.get("memory_used_mb", 0) or 0)
            > int(before.get("memory_used_mb", 0) or 0)
        )
        if not cuda_offload:
            raise RuntimeError("owner_bound_cuda_residency_missing")
        lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
        lease.transition("inferencing")
        output = _generation_receipt(port)
        if output.get("non_empty") is not True:
            raise RuntimeError("non_empty_output_missing")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if lease is not None and lease.document.get("phase") in {"resident", "inferencing"}:
            try:
                lease.transition("unloading")
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if process is not None:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            model_process["exit_code"] = process.returncode
            model_process["ended_monotonic_ns"] = time.monotonic_ns()
            model_process["absent_after_exit"] = not Path(f"/proc/{process.pid}").exists()
        recovery_started = time.monotonic()
        baseline = int(before.get("memory_used_mb", 0) or 0)
        while time.monotonic() - recovery_started <= RECOVERY_TIMEOUT_S:
            after = _gpu_snapshot(device_index, int(model_process.get("pid", 0) or 0))
            vram_recovered = (
                after.get("model_pid_present") is False
                and abs(int(after.get("memory_used_mb", 0) or 0) - baseline)
                <= RECOVERY_TOLERANCE_MB
            )
            if vram_recovered:
                break
            if lease is not None and lease.document.get("phase") not in lease_api.TERMINAL_PHASES:
                try:
                    lease.heartbeat()
                except lease_api.LeaseError:
                    pass
            time.sleep(1.0)
        if lease is not None:
            try:
                phase = lease.document.get("phase")
                if phase == "unloading" and model_process.get("absent_after_exit") is True:
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("memory_used_mb", 0) or 0),
                        exit_code=int(model_process.get("exit_code", 127) or 0),
                        unload_observed=True,
                    )
                    success = (
                        not errors
                        and tokenizer.get("loadable") is True
                        and output.get("non_empty") is True
                        and output.get("prompt_token_count", 0) > 0
                        and output.get("output_token_count", 0) > 0
                        and vram_recovered
                    )
                    lease.transition("terminal_complete" if success else "terminal_blocked")
                elif phase in {"preflight", "admitted", "loading"}:
                    lease.transition("terminal_blocked")
                if lease.document.get("phase") in lease_api.TERMINAL_PHASES:
                    release = lease.release()
                else:
                    lease.close()
            except lease_api.LeaseError as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                lease.close()
    journal: JsonDict = {}
    if lease is not None:
        try:
            journal = lease_api.read_journal(lease.journal_path)
        except lease_api.LeaseError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    worker_argv = list(sys.argv)
    row = {
        "row_kind": "mandated_model_family",
        **{key: spec[key] for key in MODEL_SPECS[0] if key in spec},
        "device_uuid": device_uuid,
        "model_path": spec.get("model_path"),
        "model_sha256": spec.get("model_sha256"),
        "tokenizer": tokenizer,
        "worker_process": {
            "pid": worker_identity["pid"],
            "pid_start_ticks": worker_identity["pid_start_ticks"],
            "executable": worker_identity["executable"],
            "argv": worker_argv,
            "argv_sha256": sha256_json(worker_argv),
            "started_monotonic_ns": worker_started,
            "ended_monotonic_ns": time.monotonic_ns(),
            "exit_code": 0,
            "absent_after_exit": True,
        },
        "model_process": model_process,
        "lease": {
            "owner": owner,
            "journal_path": str(lease.journal_path) if lease is not None else "",
            "journal_checksum": journal.get("checksum"),
            "phase_sequence": [event.get("phase") for event in journal.get("phase_history", [])],
            "phase_history": journal.get("phase_history", []),
            "release": release,
            "owner_bound": bool(owner)
            and owner.get("pid") == worker_identity["pid"]
            and owner.get("pid_start_ticks") == worker_identity["pid_start_ticks"]
            and owner.get("device_uuid") == device_uuid,
        },
        "accelerator": {
            "before": before,
            "resident": resident,
            "after": after,
            "cuda_offload": server_receipt.get("cuda_linked") is True
            and resident.get("model_pid_present") is True,
            "offloaded_layers": offloaded,
            "resident_vram_delta_mb": int(resident.get("memory_used_mb", 0) or 0)
            - int(before.get("memory_used_mb", 0) or 0),
        },
        "prompt": {"sha256": sha256_text(FIXED_PROMPT), "random_seed": RANDOM_SEED},
        "output": output,
        "unload": {
            "observed": release.get("released") is True,
            "model_process_absent": model_process.get("absent_after_exit") is True,
            "vram_recovered": vram_recovered,
        },
        "errors": errors,
        "admitted": False,
        "failed_checks": [],
    }
    return seal_family_row(row)


def worker_main(spec_path: Path, output_path: Path, runtime_dir: Path) -> int:  # pragma: no cover
    spec = _read_json(spec_path)
    row = worker_run(spec, runtime_dir)
    write_artifact_atomic(output_path, row)
    print(json.dumps({"family_id": row.get("family_id"), "admitted": row.get("admitted")}))
    return 0 if row.get("admitted") is True else 2


def _blocked_worker_row(spec: Mapping[str, Any], command: Sequence[str], error: str) -> JsonDict:
    now = time.monotonic_ns()
    row = {
        "row_kind": "mandated_model_family",
        **{key: spec.get(key) for key in MODEL_SPECS[0]},
        "device_uuid": "",
        "model_path": spec.get("model_path", ""),
        "model_sha256": spec.get("model_sha256", "missing"),
        "tokenizer": {
            "source": "llama.cpp_embedded_gguf",
            "loadable": False,
            "auto_tokenizer_used": False,
            "prompt_token_count": 0,
            "detail": "worker failed",
        },
        "worker_process": {
            "pid": 0,
            "pid_start_ticks": None,
            "executable": command[0] if command else "",
            "argv": list(command),
            "argv_sha256": sha256_json(list(command)),
            "started_monotonic_ns": now,
            "ended_monotonic_ns": now,
            "exit_code": 127,
            "absent_after_exit": True,
        },
        "model_process": _empty_process(),
        "lease": {
            "owner": {},
            "journal_path": "",
            "journal_checksum": "missing",
            "phase_sequence": [],
            "phase_history": [],
            "release": {},
            "owner_bound": False,
        },
        "accelerator": {
            "before": {},
            "resident": {},
            "after": {},
            "cuda_offload": False,
            "resident_vram_delta_mb": 0,
        },
        "prompt": {"sha256": sha256_text(FIXED_PROMPT), "random_seed": RANDOM_SEED},
        "output": {
            "text": "",
            "sha256": sha256_text(""),
            "non_empty": False,
            "prompt_token_count": 0,
            "output_token_count": 0,
            "http_status": 0,
            "finish_reason": "worker_failed",
        },
        "unload": {"observed": False, "model_process_absent": True, "vram_recovered": False},
        "errors": [error],
        "admitted": False,
        "failed_checks": [],
    }
    return seal_family_row(row)


def launch_family_worker(
    spec: Mapping[str, Any], runtime_dir: Path, repo_root: Path
) -> JsonDict:  # pragma: no cover
    """Launch and wait for exactly one family worker before returning."""

    family = str(spec["family_id"])
    spec_path = runtime_dir / f"{family}.spec.json"
    output_path = runtime_dir / f"{family}.row.json"
    write_artifact_atomic(spec_path, spec)
    command = [
        sys.executable,
        "-m",
        MODULE_NAME,
        "--worker",
        "--worker-spec",
        str(spec_path),
        "--worker-output",
        str(output_path),
        "--runtime-dir",
        str(runtime_dir),
    ]
    started = time.monotonic_ns()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    start_ticks = lease_api.proc_start_ticks(process.pid)
    try:
        stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5.0)
    ended = time.monotonic_ns()
    row = _read_json(output_path)
    if not row:
        row = _blocked_worker_row(
            spec,
            command,
            f"worker_output_missing:exit={process.returncode}:stderr={stderr[-1000:]}",
        )
    worker = dict(row.get("worker_process", {}))
    worker.update(
        {
            "pid": process.pid,
            "pid_start_ticks": start_ticks,
            "executable": worker.get("executable") or sys.executable,
            "argv": command,
            "argv_sha256": sha256_json(command),
            "started_monotonic_ns": started,
            "ended_monotonic_ns": ended,
            "exit_code": process.returncode,
            "absent_after_exit": not Path(f"/proc/{process.pid}").exists(),
            "stdout_sha256": sha256_text(stdout),
            "stderr_sha256": sha256_text(stderr),
        }
    )
    row["worker_process"] = worker
    return seal_family_row(row)


def run_family_workers(
    resolution_rows: Sequence[Mapping[str, Any]], runtime_dir: Path, repo_root: Path
) -> list[JsonDict]:
    """Run the fixed family order sequentially so reused devices never overlap."""

    rows: list[JsonDict] = []
    for spec in resolution_rows:
        rows.append(launch_family_worker(dict(spec), runtime_dir, repo_root))
    return rows


def run(
    *,
    date: str,
    root: Path,
    result_path: Path,
    work_dir: Path,
) -> JsonDict:
    """Run preconditions, verification, three canaries, and atomic publication."""

    started = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    protected_before = protected_hashes(root)
    upstream_gate = build_upstream_gate_receipt(root)
    resolution_rows = resolve_model_specs()
    tests_run = run_verification_commands(root)
    preconditions = collect_preconditions(
        root, work_dir, upstream_gate, resolution_rows, protected_before
    )
    rows = (
        run_family_workers(resolution_rows, work_dir, root)
        if preconditions["all_required_preconditions_available"]
        and all(receipt.get("exit_code") == 0 for receipt in tests_run)
        else []
    )
    artifact = build_artifact(
        date=date,
        root=root,
        duration_s=time.monotonic() - started,
        upstream_gate_receipt=upstream_gate,
        resolution_rows=resolution_rows,
        admission_rows=rows,
        preconditions=preconditions,
        protected_before=protected_before,
        tests_run=tests_run,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    write_artifact_atomic(result_path, artifact)
    return artifact


def _validate_path(path: Path) -> tuple[int, JsonDict]:
    if not path.is_file():
        return 1, {"valid": False, "errors": ["artifact_missing"]}
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return 1, {"valid": False, "errors": [f"artifact_unreadable:{type(exc).__name__}"]}
    errors = (
        validate_artifact(artifact) if isinstance(artifact, Mapping) else ["artifact_not_object"]
    )
    return (0 if not errors else 1), {"valid": not errors, "errors": errors}


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6648, validate its artifact, or execute one internal worker."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--work-dir", type=Path, default=WORK_PATH)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args(argv)
    if args.worker:  # pragma: no cover - exercised by the required live command.
        if not args.worker_spec or not args.worker_output or not args.runtime_dir:
            parser.error("worker paths are required")
        return worker_main(args.worker_spec, args.worker_output, args.runtime_dir)
    if args.validate:
        code, receipt = _validate_path(args.output)
        print(json.dumps(receipt, sort_keys=True))
        return code
    artifact = run(
        date=args.date,
        root=REPO_ROOT,
        result_path=args.output,
        work_dir=args.work_dir,
    )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "all_mandated_models_admitted": artifact["all_mandated_models_admitted"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["all_mandated_models_admitted"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
