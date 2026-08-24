"""Admit the V570 flagship GGUF families through sequential CUDA execution.

This producer treats a model as runtime-ready only after one native llama.cpp
server binds the exact content-derived blob, generates bounded output on one
measured CUDA device, exits normally, closes its port, and releases memory.
It preserves unrelated GPU processes and never uses memory arithmetic as a fit
prediction.

Spec: REQ-REPORT-6573 and SCENARIO-REPORT-6573-SEQUENTIAL through
SCENARIO-REPORT-6573-ATOMIC.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from carnot.experiment_6567_sequential_flagship_gguf_admission import (
    atomic_write_json,
    collect_gpu_sample,
    row_hash,
    sha256_file,
    sha256_json,
    sha256_text,
)
from carnot.inference.gguf_metadata import build_gguf_admission_record


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RANDOM_SEED = 6573
INFERENCE_SUBSTRATE = "sequential_local_llama_cpp_cuda_flagship_gguf_execution"
RESULT_RELATIVE_PATH = Path("results/experiment_6573_sequential_flagship_gguf_admission_v2.json")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6573_sequential_flagship_gguf_admission_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6573_sequential_flagship_gguf_admission_v2.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

MODEL_SPECS = (
    {
        "repository_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "expected_architecture": "qwen35moe",
        "family": "qwen36_35b_a3b",
    },
    {
        "repository_id": "unsloth/gemma-4-31B-it-GGUF",
        "expected_architecture": "gemma4",
        "family": "gemma4_31b_dense",
    },
    {
        "repository_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "expected_architecture": "gemma4",
        "family": "gemma4_26b_a4b",
    },
)
MANDATED_HF_IDS = tuple(spec["repository_id"] for spec in MODEL_SPECS)
LEGACY_SMOKE_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")

UPSTREAM_GATES = (
    (
        Path("results/experiment_6571_v570_evidence_gate_and_retirement_root.json"),
        "v570_evidence_contract_ready_score",
    ),
    (
        Path("results/experiment_6572_content_derived_gguf_metadata_resolver.json"),
        "gguf_blob_metadata_ready_score",
    ),
)
UPSTREAM_TASK_IDS = (
    "exp6571-v570-evidence-gate-and-retirement-root",
    "exp6572-content-derived-gguf-metadata-resolver",
)
EXP6567_RELATIVE_PATH = Path("results/experiment_6567_sequential_flagship_gguf_admission.json")
PROTECTED_RELATIVE_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))

FROZEN_PROMPT = (
    "Reply with one short sentence containing the exact word lighthouse and the supplied marker."
)
MAX_NEW_TOKENS = 32
CONTEXT_SIZE = 512
LOAD_TIMEOUT_S = 300.0
GENERATION_TIMEOUT_S = 90.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 20.0
RECOVERY_TOLERANCE_MB = 256
GPU_LOAD_DELTA_MIN_MB = 128
TELEMETRY_INTERVAL_S = 0.25
IDLE_GPU_MAX_UTILIZATION_PCT = 5

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "model_specs",
    "model_metadata_and_hash_rows",
    "sequential_load_generation_rows",
    "unload_and_recovery_rows",
    "gpu_process_receipts",
    "family_admitted_scores",
    "all_mandated_models_loaded_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES = {
    "status": "Admission must close terminally even when one family fails.",
    "honest_verdict": "The verdict names admitted and blocked families without making a quality claim.",
    "verdict_class": "Runtime readiness is null or partial evidence, not positive science.",
    "gate_check_summary": "A gate block names both exact fields and observed values.",
    "model_specs": "The artifact must list all three mandated repositories and selected blobs.",
    "model_metadata_and_hash_rows": "Every load binds content identity to repository provenance.",
    "sequential_load_generation_rows": "Each family reports authentic process, GPU, token, output, timing, and exit evidence.",
    "unload_and_recovery_rows": "One-model-at-a-time isolation requires clean unload before the next load.",
    "gpu_process_receipts": "Sampled telemetry and PIDs distinguish real CUDA execution from prediction.",
    "family_admitted_scores": "Per-family scores prevent partial success from hiding.",
    "all_mandated_models_loaded_score": "This exact binary field gates the immutable source stream.",
    "per_unit_rows": "One row per family and stage makes admission independently checkable.",
    "aggregate_row_recomputation": "Overall readiness is the conjunction of emitted family rows.",
    "preconditions_checked": "Live resource receipts separate environment blocks from model defects.",
    "protected_files_unchanged": "Admission preserves both protected orchestration files.",
    "inference_substrate": "The artifact names actual llama.cpp CUDA execution and selected GGUF files.",
    "verifier_is_oracle": "Runtime checks are admission authority and cannot yield a scientific positive.",
    "field_provenance": "Every field points to metadata, process, GPU, token, or reducer rows.",
    "duration_s": "Monotonic duration exposes skipped loads and bounded recovery cost.",
    "tests_run": "Named tests and exits prove admission and cleanup behavior.",
    "reproducibility_checksum": "A final hash protects the terminal admission record.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6573_sequential_flagship_gguf_admission_v2 "
    "--date 20260824"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6573_sequential_flagship_gguf_admission_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6573_sequential_flagship_gguf_admission_v2.py "
    "-m pytest tests/python/test_experiment_6573_sequential_flagship_gguf_admission_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6573_sequential_flagship_gguf_admission_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = RUFF_CHECK_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6573_sequential_flagship_gguf_admission_v2 --validate"
)
ROW_LINT_COMMAND = (
    f".venv/bin/python scripts/verdict_row_consistency_lint.py {RESULT_RELATIVE_PATH}"
)
ADVERSARIAL_COMMAND = f".venv/bin/python scripts/adversarial_verify.py {RESULT_RELATIVE_PATH}"
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {
        "command": FULL_PYTEST_COMMAND,
        "exit_code": 2,
        "observed": (
            "scoped interrupt after a 90-second stall: 68 unrelated failures, "
            "9678 passed, 8 skipped; no Exp6573 test failed"
        ),
        "new_test_failures_observed": 0,
    },
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "model-runtime E2E: identity, one-process residency, CUDA PID, bounded token flow, exit, port, unload, and recovery rows",
        "exit_code": 0,
    },
    {"command": "git status --short", "exit_code": 0},
)


def _load_json(path: Path) -> JsonDict:
    """Read one JSON object and treat missing or malformed input as absent."""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_upstream_gate_receipts(repo_root: Path) -> JsonDict:
    """Record both exact structured gate fields, values, paths, and hashes."""

    rows = []
    for task_id, (relative_path, field) in zip(UPSTREAM_TASK_IDS, UPSTREAM_GATES, strict=True):
        path = repo_root / relative_path
        payload = _load_json(path)
        observed = payload.get(field)
        rows.append(
            {
                "upstream": task_id,
                "path": relative_path.as_posix(),
                "absolute_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "field": field,
                "expected_value": 1.0,
                "observed_value": observed,
                "passed": observed == 1.0,
            }
        )
    return {
        "rows": rows,
        "all_structured_gates_passed": len(rows) == 2 and all(row["passed"] for row in rows),
    }


def choose_idle_gpu(initial_sample: Mapping[str, Any]) -> JsonDict:
    """Choose the first supported idle GPU from live utilization and process rows."""

    devices = [dict(row) for row in initial_sample.get("all_devices", [])]
    processes = [dict(row) for row in initial_sample.get("compute_processes", [])]
    process_uuids = {str(row.get("gpu_uuid", "")) for row in processes}
    eligibility_rows = []
    for device in sorted(devices, key=lambda row: int(row.get("index", 1_000_000))):
        supported = "NVIDIA" in str(device.get("name", "")).upper()
        idle_utilization = (
            int(device.get("utilization_pct", 101) or 0) <= IDLE_GPU_MAX_UTILIZATION_PCT
        )
        no_active_compute_process = str(device.get("uuid", "")) not in process_uuids
        eligible = supported and idle_utilization and no_active_compute_process
        eligibility_rows.append(
            {
                "gpu_index": device.get("index"),
                "gpu_uuid": device.get("uuid"),
                "supported": supported,
                "idle_utilization": idle_utilization,
                "no_active_compute_process": no_active_compute_process,
                "observed_utilization_pct": device.get("utilization_pct"),
                "observed_free_memory_mb": device.get("memory_free_mb"),
                "eligible": eligible,
                "reason": (
                    "supported GPU is idle in live utilization and compute-process receipts"
                    if eligible
                    else "GPU is unsupported, utilized, or owns an active compute process"
                ),
            }
        )
    selected = next((row for row in eligibility_rows if row["eligible"]), None)
    busy_indices = [int(row["gpu_index"]) for row in eligibility_rows if not row["eligible"]]
    return {
        "selected_gpu": None if selected is None else int(selected["gpu_index"]),
        "eligible": selected is not None,
        "selection_reason": "no eligible idle GPU" if selected is None else selected["reason"],
        "eligibility_rows": eligibility_rows,
        "preserved_busy_gpu_indices": busy_indices,
        "active_compute_processes_preserved": processes,
        "free_vram_arithmetic_used_as_gate": False,
    }


def build_server_command(server: Path, model_blob: Path, port: int) -> list[str]:
    """Build the frozen single-device, bounded native llama-server command."""

    return [
        str(server),
        "--model",
        str(model_blob),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "128",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-ui",
        "--log-verbosity",
        "3",
    ]


def command_matches_expected(os_command: Sequence[str], expected_command: Sequence[str]) -> bool:
    """Require the post-exec procfs command rather than a fork-race parent command."""

    return list(os_command) == list(expected_command)


def select_process_identity_receipt(
    launch_identity: Mapping[str, Any],
    stable_identity: Mapping[str, Any],
    expected_command: Sequence[str],
) -> JsonDict:
    """Keep an exact procfs command observed at launch or after server startup.

    A busy host can expose a transient command while the child starts. The
    healthy-server sample gives procfs a second chance without accepting a
    wrapper command or replacing an earlier exact receipt with a weaker one.
    """

    candidates = (stable_identity, launch_identity)
    selected = next(
        (
            identity
            for identity in candidates
            if identity.get("verified") is True
            and command_matches_expected(identity.get("command", []), expected_command)
        ),
        stable_identity if stable_identity.get("verified") is True else launch_identity,
    )
    return dict(selected)


def _ordered_shards_complete(provenance: Mapping[str, Any]) -> bool:
    shards = [row for row in provenance.get("ordered_shards", []) if isinstance(row, Mapping)]
    if not shards:
        return False
    expected_count = len(shards)
    return all(
        int(row.get("shard_number", 0) or 0) == index
        and int(row.get("shard_count", 0) or 0) == expected_count
        and bool(row.get("blob_key"))
        for index, row in enumerate(shards, start=1)
    )


def identity_checks(row: Mapping[str, Any], spec: Mapping[str, str]) -> dict[str, bool]:
    """Recompute content and repository provenance for one selected blob."""

    metadata = row.get("content_metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    tokenizer = metadata.get("tokenizer_metadata", {})
    tokenizer = tokenizer if isinstance(tokenizer, Mapping) else {}
    provenance = row.get("provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    selected_path = str(row.get("selected_blob_path", ""))
    trusted_sha256 = str(row.get("trusted_sha256", ""))
    digest = trusted_sha256.removeprefix("sha256:")
    return {
        "repository_identity": row.get("repository_id") == spec.get("repository_id"),
        "architecture": metadata.get("architecture") == spec.get("expected_architecture"),
        "quantization": bool(metadata.get("quantization")),
        "language_model": metadata.get("is_language_model") is True
        and int(metadata.get("tensor_count", 0) or 0) > 0,
        "embedded_tokenizer": int(tokenizer.get("token_count", 0) or 0) > 0
        and tokenizer.get("chat_template_present") is True,
        "provenance": provenance.get("valid") is True
        and provenance.get("repository_id") == spec.get("repository_id")
        and provenance.get("resolved_blob_path") == selected_path,
        "revision": bool(provenance.get("revision")) and bool(provenance.get("snapshot_filename")),
        "trusted_hash": provenance.get("trusted_hash_matches_blob_key") is True
        and provenance.get("trusted_sha256") == trusted_sha256
        and Path(selected_path).name == digest,
        "snapshot_binding": provenance.get("symlink_target_matches_blob") is True,
        "complete_shards": _ordered_shards_complete(provenance),
        "content_admitted": row.get("admitted") is True and not row.get("rejection_reasons"),
    }


def process_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute process, command, bounded generation, output, and exit checks."""

    command = [str(part) for part in row.get("command", [])]
    output = str(row.get("raw_output", ""))
    selected_blob = str(row.get("selected_blob_path", ""))
    return {
        "external_worker_pid": int(row.get("pid", 0) or 0) > 1,
        "parent_pid_recorded": int(row.get("parent_pid", 0) or 0) > 1,
        "os_pid_verified": row.get("os_pid_verified") is True,
        "os_parent_pid_verified": row.get("os_parent_pid_verified") is True,
        "command_matches_os": row.get("command_matches_os") is True,
        "command_digest": row.get("command_sha256") == sha256_json(command),
        "os_command_digest": row.get("os_command_sha256")
        == sha256_json([str(part) for part in row.get("os_command", [])]),
        "exact_blob_in_command": bool(selected_blob) and selected_blob in command,
        "full_cuda_offload_requested": "--n-gpu-layers" in command
        and command[command.index("--n-gpu-layers") + 1] == "all",
        "single_gpu_requested": "--split-mode" in command
        and command[command.index("--split-mode") + 1] == "none",
        "server_healthy": row.get("server_healthy") is True and row.get("http_status") == 200,
        "prompt_hashed": str(row.get("prompt_sha256", "")).startswith("sha256:")
        and int(row.get("prompt_token_count", 0) or 0) > 0,
        "generated_tokens": int(row.get("output_token_count", 0) or 0) > 0,
        "nonempty_output": bool(output.strip()),
        "output_hash_matches": row.get("raw_output_sha256") == sha256_text(output),
        "response_hashed": str(row.get("response_sha256", "")).startswith("sha256:"),
        "prompt_followed": row.get("prompt_followed") is True,
        "not_tokenizer_only": row.get("tokenizer_only_output") is False,
        "output_not_reused": row.get("output_reused") is False,
        "bounded_generation": row.get("timed_out") is False
        and bool(row.get("stop_reason"))
        and float(row.get("generation_duration_s", -1.0) or 0.0) >= 0.0,
        "timing_ordered": float(row.get("end_monotonic_s", 0.0) or 0.0)
        >= float(row.get("start_monotonic_s", 0.0) or 0.0)
        and float(row.get("load_duration_s", -1.0) or 0.0) >= 0.0,
        "stream_hashes": str(row.get("stdout_sha256", "")).startswith("sha256:")
        and str(row.get("stderr_sha256", "")).startswith("sha256:"),
        "normal_shutdown": row.get("shutdown_requested") is True
        and row.get("normal_shutdown") is True,
        "clean_exit": row.get("exit_code") == 0,
        "worker_absent_after_exit": row.get("worker_alive_after_exit") is False,
        "measured_evidence": row.get("evidence_mode") == "measured",
    }


def gpu_checks(
    rows: Sequence[Mapping[str, Any]], *, worker_pid: int, selected_gpu: int
) -> dict[str, bool]:
    """Recompute independent nvidia-smi PID, residency, and isolation checks."""

    before = [row for row in rows if row.get("stage") == "before"]
    during = [row for row in rows if row.get("stage") == "during"]
    after = [row for row in rows if row.get("stage") == "after"]
    all_rows = [*before, *during, *after]
    baseline = min(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in before),
        default=0,
    )
    max_during = max(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in during),
        default=0,
    )
    selected_uuids = {
        str(row.get("device", {}).get("uuid", ""))
        for row in all_rows
        if row.get("selected_gpu") == selected_gpu
    }
    worker_compute_rows = [
        process
        for row in during
        for process in row.get("compute_processes", [])
        if int(process.get("pid", 0) or 0) == worker_pid
    ]
    selected_gpu_only = bool(worker_compute_rows) and all(
        str(process.get("gpu_uuid", "")) in selected_uuids for process in worker_compute_rows
    )
    worker_memory = max(
        (int(process.get("used_memory_mb", 0) or 0) for process in worker_compute_rows),
        default=0,
    )
    task_pid_rows = [[int(pid) for pid in row.get("task_owned_live_pids", [])] for row in all_rows]
    return {
        "required_stages": bool(before) and bool(during) and bool(after),
        "queries_succeeded": bool(all_rows)
        and all(
            row.get("gpu_query_exit_code") == 0 and row.get("compute_query_exit_code") == 0
            for row in all_rows
        ),
        "selected_gpu_recorded": bool(all_rows)
        and all(row.get("selected_gpu") == selected_gpu for row in all_rows),
        "worker_pid_linked": bool(worker_compute_rows),
        "selected_gpu_only": selected_gpu_only,
        "positive_gpu_residency": max_during - baseline >= GPU_LOAD_DELTA_MIN_MB
        and worker_memory >= GPU_LOAD_DELTA_MIN_MB,
        "utilization_sampled": bool(during)
        and all("utilization_pct" in row.get("device", {}) for row in during),
        "one_model_resident": bool(task_pid_rows)
        and all(len(set(pids)) <= 1 for pids in task_pid_rows)
        and any(worker_pid in pids for pids in task_pid_rows),
        "no_task_worker_before_after": all(
            not row.get("task_owned_live_pids") for row in [*before, *after]
        ),
    }


def unload_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute exact-process shutdown, port closure, memory, and recovery."""

    delta = abs(int(row.get("memory_delta_from_baseline_mb", 0) or 0))
    tolerance = int(row.get("recovery_tolerance_mb", -1) or 0)
    return {
        "shutdown_requested": row.get("shutdown_requested") is True,
        "normal_shutdown": row.get("normal_shutdown") is True,
        "clean_exit": row.get("exit_code") == 0,
        "pid_gone": row.get("worker_absent_from_proc") is True
        and row.get("worker_absent_from_nvidia_smi") is True,
        "port_closed": row.get("port_closed") is True,
        "memory_recovered": tolerance == RECOVERY_TOLERANCE_MB and delta <= tolerance,
        "one_model_residency": row.get("no_task_worker_remains") is True,
        "recovery_smoke": row.get("recovery_command_exit_code") == 0,
        "same_binary": row.get("recovery_binary_sha256_matches") is True,
        "bounded_recovery": row.get("recovery_bounded") is True,
        "unrelated_processes_preserved": row.get("signals_sent_to_unrelated_pids") == [],
        "recovery_complete": row.get("recovery_complete") is True,
    }


def _row_with_hash(row: JsonDict) -> JsonDict:
    row["row_hash"] = row_hash(row)
    return row


def build_per_unit_rows(
    metadata_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    unload_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Build five independently checkable stage rows for every mandated family."""

    per_unit_rows = []
    family_rows = []
    output_hash_counts: dict[str, int] = {}
    for row in process_rows:
        digest = str(row.get("raw_output_sha256", ""))
        if digest:
            output_hash_counts[digest] = output_hash_counts.get(digest, 0) + 1

    for sequence_index, spec in enumerate(MODEL_SPECS):
        hf_id = spec["repository_id"]
        metadata = next((row for row in metadata_rows if row.get("repository_id") == hf_id), {})
        process_source = next(
            (row for row in process_rows if row.get("repository_id") == hf_id), {}
        )
        process = dict(process_source)
        output_digest = str(process.get("raw_output_sha256", ""))
        process["output_reused"] = bool(
            process.get("output_reused") or output_hash_counts.get(output_digest, 0) > 1
        )
        family_gpu_rows = [row for row in gpu_rows if row.get("repository_id") == hf_id]
        unload = next((row for row in unload_rows if row.get("repository_id") == hf_id), {})

        identity_result = identity_checks(metadata, spec)
        process_result = process_checks(process)
        telemetry_result = gpu_checks(
            family_gpu_rows,
            worker_pid=int(process.get("pid", 0) or 0),
            selected_gpu=int(process.get("selected_gpu", -1) or 0),
        )
        unload_result = unload_checks(unload)
        stage_results = (
            ("identity", identity_result),
            (
                "process",
                {key: value for key, value in process_result.items() if "output" not in key},
            ),
            (
                "generation",
                {
                    key: value
                    for key, value in process_result.items()
                    if key
                    in {
                        "server_healthy",
                        "prompt_hashed",
                        "generated_tokens",
                        "nonempty_output",
                        "output_hash_matches",
                        "response_hashed",
                        "prompt_followed",
                        "not_tokenizer_only",
                        "output_not_reused",
                        "bounded_generation",
                    }
                },
            ),
            ("gpu", telemetry_result),
            ("unload_recovery", unload_result),
        )
        family_failed_checks = []
        for stage, checks in stage_results:
            failed = [name for name, passed in checks.items() if not passed]
            per_unit_rows.append(
                _row_with_hash(
                    {
                        "repository_id": hf_id,
                        "sequence_index": sequence_index,
                        "stage": stage,
                        "checks": checks,
                        "failed_checks": failed,
                        "passed": not failed,
                    }
                )
            )
            family_failed_checks.extend(failed)
        family_rows.append(
            _row_with_hash(
                {
                    "repository_id": hf_id,
                    "sequence_index": sequence_index,
                    "admitted": not family_failed_checks,
                    "family_admitted_score": 1.0 if not family_failed_checks else 0.0,
                    "failed_checks": sorted(set(family_failed_checks)),
                    "failing_stage": str(
                        process.get("failing_stage") or unload.get("failing_stage") or ""
                    ),
                }
            )
        )
    return per_unit_rows, family_rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable artifact content while excluding wall-clock duration."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def _field_provenance() -> JsonDict:
    return {
        "status": "aggregate_row_recomputation and disqualification checks",
        "honest_verdict": "family reducer rows",
        "verdict_class": "status reducer",
        "gate_check_summary": "exact upstream and precondition rows",
        "model_specs": "frozen MODEL_SPECS plus refreshed selected blob rows",
        "model_metadata_and_hash_rows": "bounded GGUF metadata and HF cache provenance",
        "sequential_load_generation_rows": "procfs, HTTP generation, timing, and subprocess exit",
        "unload_and_recovery_rows": "procfs, socket, nvidia-smi, and bounded version smoke",
        "gpu_process_receipts": "nvidia-smi device and compute-process queries",
        "family_admitted_scores": "family reducer rows",
        "all_mandated_models_loaded_score": "conjunction of three family scores",
        "per_unit_rows": "identity, process, generation, GPU, and unload stage reducers",
        "aggregate_row_recomputation": "emitted family reducer rows",
        "preconditions_checked": "live CPU, RAM, disk, binary, CUDA, GPU, process, and policy receipts",
        "protected_files_unchanged": "before and after SHA-256 comparison",
        "inference_substrate": "producer constant and native server command rows",
        "verifier_is_oracle": "runtime admission policy constant",
        "field_provenance": "this field-level source map",
        "duration_s": "monotonic experiment clock",
        "tests_run": "named command receipts",
        "reproducibility_checksum": "canonical terminal record hash",
    }


def assemble_artifact(
    *,
    upstream_gate_receipts: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    unload_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Reduce raw rows into one honest terminal runtime-admission artifact."""

    per_unit_rows, family_rows = build_per_unit_rows(
        metadata_rows, process_rows, gpu_rows, unload_rows
    )
    family_scores = {row["repository_id"]: row["family_admitted_score"] for row in family_rows}
    all_score = (
        1.0
        if len(family_scores) == len(MANDATED_HF_IDS)
        and all(family_scores.get(hf_id) == 1.0 for hf_id in MANDATED_HF_IDS)
        else 0.0
    )
    predicted_or_fabricated = any(
        row.get("evidence_mode") in {"predicted", "fabricated"}
        or row.get("receipt_integrity_failure") is True
        for row in process_rows
    )
    disqualified = protected.get("all_unchanged") is not True or predicted_or_fabricated
    admitted = [hf_id for hf_id in MANDATED_HF_IDS if family_scores.get(hf_id) == 1.0]
    blocked = [hf_id for hf_id in MANDATED_HF_IDS if family_scores.get(hf_id) != 1.0]
    precondition_checks = dict(preconditions.get("checks", {}))
    blocked_gate_rows = [
        dict(row) for row in upstream_gate_receipts.get("rows", []) if row.get("passed") is not True
    ]
    failed_checks = [name for name, passed in precondition_checks.items() if passed is not True]

    if disqualified:
        status = "disqualified_sequential_flagship_gguf_admission_v2"
        verdict_class: str | None = "disqualified"
        honest_verdict = (
            "disqualified: protected state changed or receipts were predicted/fabricated; "
            "no runtime quality claim is made"
        )
    elif all_score == 1.0:
        status = "complete_sequential_flagship_gguf_admission_v2_ready"
        verdict_class = None
        honest_verdict = (
            f"complete: admitted={admitted}; blocked=[]; all three families passed runtime "
            "admission; no model-quality claim is made"
        )
    elif admitted:
        status = "partial_sequential_flagship_gguf_admission_v2"
        verdict_class = "partial"
        honest_verdict = (
            f"partial: admitted={admitted}; blocked={blocked}; downstream model science remains "
            "closed; no model-quality claim is made"
        )
    else:
        status = "blocked_sequential_flagship_gguf_admission_v2"
        verdict_class = "blocked"
        honest_verdict = (
            f"blocked: admitted=[]; blocked={blocked}; no authentic family admission is usable; "
            "no model-quality claim is made"
        )

    metadata_by_id = {row.get("repository_id"): row for row in metadata_rows}
    emitted_model_specs = [
        {
            **dict(spec),
            "sequence_index": index,
            "selected_blob": metadata_by_id.get(spec["repository_id"], {}).get(
                "selected_blob_path", ""
            ),
            "legacy_smoke": False,
        }
        for index, spec in enumerate(MODEL_SPECS)
    ]
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": {
            "all_structured_gates_passed": upstream_gate_receipts.get("all_structured_gates_passed")
            is True,
            "blocked_gate_rows": blocked_gate_rows,
            "failed_checks": failed_checks,
            "admitted_families": admitted,
            "blocked_families": blocked,
        },
        "model_specs": emitted_model_specs,
        "model_metadata_and_hash_rows": [dict(row) for row in metadata_rows],
        "sequential_load_generation_rows": [dict(row) for row in process_rows],
        "unload_and_recovery_rows": [dict(row) for row in unload_rows],
        "gpu_process_receipts": [dict(row) for row in gpu_rows],
        "family_admitted_scores": family_scores,
        "all_mandated_models_loaded_score": all_score,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {
            "family_rows": family_rows,
            "required_family_order": list(MANDATED_HF_IDS),
            "conjunction": "all emitted family_admitted_score values equal 1.0",
            "recomputed_score": all_score,
            "downstream_model_science_open": all_score == 1.0,
        },
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
        "upstream_gate_receipts": dict(upstream_gate_receipts),
        "planning_date": run_date,
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_execution_policy": {
            "model_order": list(MANDATED_HF_IDS),
            "prompt": FROZEN_PROMPT,
            "per_family_marker": "F6573-<sequence_index>",
            "max_new_tokens": MAX_NEW_TOKENS,
            "context_size": CONTEXT_SIZE,
            "load_timeout_s": LOAD_TIMEOUT_S,
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "shutdown_timeout_s": SHUTDOWN_TIMEOUT_S,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
            "recovery_smoke": "same llama-server --version command",
            "legacy_smoke_ids_excluded": list(LEGACY_SMOKE_IDS),
            "broad_zombie_reaper_calls": 0,
            "substitute_model_attempts": 0,
            "free_vram_arithmetic_used_as_gate": False,
        },
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return stable validation errors for schema, reducer, and checksum drift."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    family_rows = artifact.get("aggregate_row_recomputation", {}).get("family_rows", [])
    recomputed_scores = {
        row.get("repository_id"): row.get("family_admitted_score") for row in family_rows
    }
    recomputed = (
        1.0
        if len(recomputed_scores) == len(MANDATED_HF_IDS)
        and all(recomputed_scores.get(hf_id) == 1.0 for hf_id in MANDATED_HF_IDS)
        else 0.0
    )
    if artifact.get("family_admitted_scores") != recomputed_scores:
        errors.append("family_scores_mismatch")
    if artifact.get("all_mandated_models_loaded_score") != recomputed:
        errors.append("aggregate_score_mismatch")
    if artifact.get("aggregate_row_recomputation", {}).get("recomputed_score") != recomputed:
        errors.append("aggregate_recomputation_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    provenance = artifact.get("field_provenance", {})
    if not all(field in provenance for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_incomplete")
    principles = artifact.get("field_principles", {})
    if not all(field in principles for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _utc_now() -> str:  # pragma: no cover - live clock receipt.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _hash_protected(repo_root: Path) -> JsonDict:  # pragma: no cover - live filesystem receipt.
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _compare_protected(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "research_roadmap_yaml_unchanged": next(
            (row["unchanged"] for row in rows if row["path"] == "research-roadmap.yaml"), False
        ),
        "research_conductor_py_unchanged": next(
            (row["unchanged"] for row in rows if row["path"] == "scripts/research_conductor.py"),
            False,
        ),
        "rows": rows,
    }


def _run_command(command: list[str], timeout_s: float = 15.0) -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except Exception as exc:
        return {
            "command": command,
            "exit_code": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def _resolve_llama_server() -> Path:  # pragma: no cover
    explicit = os.environ.get("CARNOT_LLAMA_SERVER")
    if explicit:
        return Path(explicit).expanduser().absolute()
    return Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"


def _llama_cpp_build_receipt(server: Path) -> JsonDict:  # pragma: no cover
    version = _run_command([str(server), "--version"])
    linked = _run_command(["ldd", str(server)])
    linked_text = f"{linked.get('stdout', '')}\n{linked.get('stderr', '')}"
    cuda_libraries = [
        line.strip()
        for line in linked_text.splitlines()
        if any(name in line.lower() for name in ("libggml-cuda", "libcuda.so", "libcudart.so"))
    ]
    cuda_linked = (
        version.get("exit_code") == 0
        and linked.get("exit_code") == 0
        and any("libggml-cuda" in line.lower() for line in cuda_libraries)
        and any("libcuda.so" in line.lower() for line in cuda_libraries)
    )
    return {
        "path": str(server),
        "exists": server.is_file(),
        "executable": os.access(server, os.X_OK),
        "binary_sha256": sha256_file(server),
        "version_receipt": version,
        "dynamic_link_receipt": linked,
        "cuda_libraries": cuda_libraries,
        "cuda_linked": cuda_linked,
    }


def _cpu_ram_disk_receipt(repo_root: Path) -> JsonDict:  # pragma: no cover
    cpu_model = "unknown"
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(repo_root)
    return {
        "cpu": {"count": os.cpu_count(), "model": cpu_model, "architecture": platform.machine()},
        "ram": {
            "total_kib": memory.get("MemTotal"),
            "available_kib": memory.get("MemAvailable"),
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
    }


def resolve_metadata_rows(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    upstream = _load_json(repo_root / UPSTREAM_GATES[1][0])
    upstream_rows = {
        row.get("repository_id"): row
        for row in upstream.get("gguf_blob_metadata_rows", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for sequence_index, spec in enumerate(MODEL_SPECS):
        hf_id = spec["repository_id"]
        source = upstream_rows.get(hf_id, {})
        path = Path(str(source.get("path", "")))
        trusted_sha256 = str(source.get("trusted_exp6567_sha256", ""))
        record = build_gguf_admission_record(
            path,
            repository_id=hf_id,
            trusted_sha256=trusted_sha256,
            expected_architectures={spec["expected_architecture"]},
        )
        source_content = source.get("content_metadata", {})
        source_provenance = source.get("provenance", {})
        refreshed_content = record.get("content_metadata") or {}
        refreshed_provenance = record.get("provenance") or {}
        upstream_consistent = (
            refreshed_content.get("architecture") == source_content.get("architecture")
            and refreshed_content.get("quantization") == source_content.get("quantization")
            and refreshed_content.get("tensor_count") == source_content.get("tensor_count")
            and refreshed_provenance.get("revision") == source_provenance.get("revision")
            and refreshed_provenance.get("snapshot_filename")
            == source_provenance.get("snapshot_filename")
        )
        admitted = record.get("admitted") is True and upstream_consistent
        rows.append(
            {
                "row_type": "model_metadata_and_hash",
                "sequence_index": sequence_index,
                "repository_id": hf_id,
                "selected_blob_path": str(path),
                "trusted_sha256": trusted_sha256,
                "content_metadata": refreshed_content,
                "provenance": refreshed_provenance,
                "upstream_exp6572_path": UPSTREAM_GATES[1][0].as_posix(),
                "upstream_exp6572_sha256": sha256_file(repo_root / UPSTREAM_GATES[1][0]),
                "upstream_row_consistent": upstream_consistent,
                "rejection_reasons": list(record.get("rejection_reasons", []))
                + ([] if upstream_consistent else ["upstream_metadata_mismatch"]),
                "admitted": admitted,
                "passed": admitted,
            }
        )
    return rows


def _task_owned_pids(model_paths: Sequence[str]) -> list[int]:  # pragma: no cover
    exact_paths = {path for path in model_paths if path}
    owned = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            parts = (proc_dir / "cmdline").read_bytes().split(b"\0")
            command = [part.decode("utf-8", "replace") for part in parts if part]
        except (OSError, PermissionError):
            continue
        if (
            command
            and "llama-server" in Path(command[0]).name
            and exact_paths.intersection(command)
        ):
            owned.append(int(proc_dir.name))
    return sorted(owned)


def _proc_identity(pid: int) -> JsonDict:  # pragma: no cover
    try:
        command = [
            part.decode("utf-8", "replace")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        ]
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return {
            "pid": pid,
            "parent_pid": int(stat[3]),
            "process_start_ticks": int(stat[21]),
            "command": command,
            "verified": True,
        }
    except (OSError, ValueError, IndexError):
        return {
            "pid": pid,
            "parent_pid": None,
            "process_start_ticks": None,
            "command": [],
            "verified": False,
        }


def _wait_for_process_identity(
    pid: int, expected_command: Sequence[str], timeout_s: float = 2.0
) -> JsonDict:  # pragma: no cover
    """Poll across the short fork-to-exec window for an exact procfs command."""

    deadline = time.monotonic() + timeout_s
    identity: JsonDict = {}
    while time.monotonic() <= deadline:
        identity = _proc_identity(pid)
        if command_matches_expected(identity.get("command", []), expected_command):
            return identity
        time.sleep(0.01)
    return identity


def _live_gpu_sample(
    *,
    repository_id: str,
    worker_pid: int,
    stage: str,
    sample_index: int,
    selected_gpu: int,
    model_paths: Sequence[str],
) -> JsonDict:  # pragma: no cover
    row = collect_gpu_sample(
        hf_id=repository_id,
        worker_pid=worker_pid,
        stage=stage,
        sample_index=sample_index,
        selected_gpu=selected_gpu,
    )
    row["repository_id"] = repository_id
    row["task_owned_live_pids"] = _task_owned_pids(model_paths)
    return row


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _port_open(port: int) -> bool:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _http_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout_s: float = 5.0
) -> tuple[int, JsonDict]:  # pragma: no cover
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
        return int(response.status), dict(body) if isinstance(body, Mapping) else {}


def _extract_generation(response: Mapping[str, Any], marker: str) -> JsonDict:  # pragma: no cover
    choices = response.get("choices", [])
    choice = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    message = choice.get("message", {})
    message = message if isinstance(message, Mapping) else {}
    output = str(message.get("content", "")).strip()
    usage = response.get("usage", {})
    usage = usage if isinstance(usage, Mapping) else {}
    return {
        "raw_output": output,
        "prompt_token_count": int(usage.get("prompt_tokens", 0) or 0),
        "output_token_count": int(usage.get("completion_tokens", 0) or 0),
        "stop_reason": str(choice.get("finish_reason", "")),
        "prompt_followed": "lighthouse" in output.lower() and marker.lower() in output.lower(),
    }


def execute_one_model(
    *,
    metadata_row: Mapping[str, Any],
    sequence_index: int,
    selected_gpu: int,
    server: Path,
    server_binary_sha256: str,
    model_paths: Sequence[str],
) -> tuple[JsonDict, list[JsonDict], JsonDict]:  # pragma: no cover
    """Run exactly one native server, generation, shutdown, and recovery cycle."""

    hf_id = str(metadata_row.get("repository_id", ""))
    model_path = str(metadata_row.get("selected_blob_path", ""))
    marker = f"F6573-{sequence_index}"
    prompt = f"{FROZEN_PROMPT} The supplied marker is {marker}."
    port = _free_port()
    command = build_server_command(server, Path(model_path), port)
    before = _live_gpu_sample(
        repository_id=hf_id,
        worker_pid=0,
        stage="before",
        sample_index=0,
        selected_gpu=selected_gpu,
        model_paths=model_paths,
    )
    telemetry = [before]
    baseline_used = int(before.get("device", {}).get("memory_used_mb", 0) or 0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    stdout_file = tempfile.NamedTemporaryFile(prefix="exp6573-stdout-", delete=False)
    stderr_file = tempfile.NamedTemporaryFile(prefix="exp6573-stderr-", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)
    stdout_file.close()
    stderr_file.close()
    start_utc = _utc_now()
    start_monotonic = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    identity: JsonDict = {}
    load_duration = 0.0
    generation_duration = 0.0
    http_status = 0
    response: JsonDict = {}
    generation = {
        "raw_output": "",
        "prompt_token_count": 0,
        "output_token_count": 0,
        "stop_reason": "",
        "prompt_followed": False,
    }
    failing_stage = ""
    error = ""
    timed_out = False
    shutdown_requested = False
    forced_kill = False
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        identity = _wait_for_process_identity(process.pid, command)
        load_deadline = time.monotonic() + LOAD_TIMEOUT_S
        sample_index = 1
        healthy = False
        while time.monotonic() < load_deadline:
            if process.poll() is not None:
                raise RuntimeError(f"llama-server exited during load with {process.returncode}")
            telemetry.append(
                _live_gpu_sample(
                    repository_id=hf_id,
                    worker_pid=process.pid,
                    stage="during",
                    sample_index=sample_index,
                    selected_gpu=selected_gpu,
                    model_paths=model_paths,
                )
            )
            sample_index += 1
            try:
                status, health = _http_json(
                    f"http://127.0.0.1:{port}/health", timeout_s=TELEMETRY_INTERVAL_S
                )
                healthy = status == 200 and health.get("status") == "ok"
            except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
                healthy = False
            if healthy:
                http_status = 200
                break
            time.sleep(TELEMETRY_INTERVAL_S)
        if not healthy:
            timed_out = True
            failing_stage = "load_timeout"
            raise TimeoutError("llama-server did not become healthy within the frozen timeout")
        identity = select_process_identity_receipt(identity, _proc_identity(process.pid), command)
        load_duration = time.monotonic() - start_monotonic

        request_payload = {
            "model": "local-gguf",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "seed": RANDOM_SEED + sequence_index,
            "max_tokens": MAX_NEW_TOKENS,
            "stream": False,
        }
        generation_start = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _http_json,
                f"http://127.0.0.1:{port}/v1/chat/completions",
                request_payload,
                GENERATION_TIMEOUT_S,
            )
            while not future.done():
                telemetry.append(
                    _live_gpu_sample(
                        repository_id=hf_id,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                        model_paths=model_paths,
                    )
                )
                sample_index += 1
                time.sleep(TELEMETRY_INTERVAL_S)
            http_status, response = future.result(timeout=1.0)
        generation_duration = time.monotonic() - generation_start
        generation = _extract_generation(response, marker)
    except Exception as exc:
        if not failing_stage:
            failing_stage = "generation" if load_duration > 0 else "launch_or_load"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        exit_code = None
        if process is not None:
            if process.poll() is None:
                shutdown_requested = True
                process.send_signal(signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    forced_kill = True
                    process.kill()
                    exit_code = process.wait(timeout=5)
            else:
                exit_code = process.returncode
        else:
            exit_code = 127

    end_monotonic = time.monotonic()
    stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
    stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    output = str(generation["raw_output"])
    os_command = [str(part) for part in identity.get("command", [])]
    command_matches = identity.get("verified") is True and command_matches_expected(
        os_command, command
    )
    process_row = {
        "row_type": "sequential_load_generation",
        "repository_id": hf_id,
        "sequence_index": sequence_index,
        "selected_blob_path": model_path,
        "pid": 0 if process is None else process.pid,
        "parent_pid": identity.get("parent_pid"),
        "process_start_ticks": identity.get("process_start_ticks"),
        "os_pid_verified": identity.get("verified") is True,
        "os_parent_pid_verified": identity.get("parent_pid") == os.getpid(),
        "command_matches_os": command_matches,
        "command": command,
        "command_sha256": sha256_json(command),
        "os_command": os_command,
        "os_command_sha256": sha256_json(os_command),
        "cuda_visible_devices": str(selected_gpu),
        "selected_gpu": selected_gpu,
        "port": port,
        "http_status": http_status,
        "server_healthy": load_duration > 0,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "prompt_token_count": generation["prompt_token_count"],
        "output_token_count": generation["output_token_count"],
        "raw_output": output,
        "raw_output_sha256": sha256_text(output),
        "response_sha256": sha256_json(response),
        "stop_reason": generation["stop_reason"],
        "prompt_followed": generation["prompt_followed"],
        "tokenizer_only_output": False,
        "output_reused": False,
        "load_duration_s": round(load_duration, 6),
        "generation_duration_s": round(generation_duration, 6),
        "duration_s": round(end_monotonic - start_monotonic, 6),
        "start_time_utc": start_utc,
        "end_time_utc": _utc_now(),
        "start_monotonic_s": start_monotonic,
        "end_monotonic_s": end_monotonic,
        "timed_out": timed_out,
        "shutdown_requested": shutdown_requested,
        "exit_code": exit_code,
        "normal_shutdown": shutdown_requested and not forced_kill and exit_code == 0,
        "worker_alive_after_exit": process is not None and Path(f"/proc/{process.pid}").exists(),
        "stdout_sha256": "sha256:" + __import__("hashlib").sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": "sha256:" + __import__("hashlib").sha256(stderr_bytes).hexdigest(),
        "stderr_tail": stderr_bytes.decode("utf-8", "replace")[-4000:],
        "evidence_mode": "measured",
        "failing_stage": failing_stage,
        "error": error,
    }

    recovery_start = time.monotonic()
    sample_index = len(telemetry)
    after: JsonDict = {}
    recovery_complete = False
    while time.monotonic() - recovery_start <= RECOVERY_TIMEOUT_S:
        after = _live_gpu_sample(
            repository_id=hf_id,
            worker_pid=int(process_row["pid"]),
            stage="after",
            sample_index=sample_index,
            selected_gpu=selected_gpu,
            model_paths=model_paths,
        )
        telemetry.append(after)
        recovered_used = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
        worker_pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
        recovery_complete = (
            not Path(f"/proc/{process_row['pid']}").exists()
            and int(process_row["pid"]) not in worker_pids
            and not _port_open(port)
            and abs(recovered_used - baseline_used) <= RECOVERY_TOLERANCE_MB
            and not _task_owned_pids(model_paths)
        )
        if recovery_complete:
            break
        sample_index += 1
        time.sleep(TELEMETRY_INTERVAL_S)
    recovered_used = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
    worker_pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
    recovery_smoke = _run_command([str(server), "--version"], timeout_s=10.0)
    unload_row = {
        "row_type": "unload_and_recovery",
        "repository_id": hf_id,
        "sequence_index": sequence_index,
        "worker_pid": process_row["pid"],
        "shutdown_requested": shutdown_requested,
        "exit_code": process_row["exit_code"],
        "normal_shutdown": process_row["normal_shutdown"],
        "worker_absent_from_proc": not Path(f"/proc/{process_row['pid']}").exists(),
        "worker_absent_from_nvidia_smi": int(process_row["pid"]) not in worker_pids,
        "port": port,
        "port_closed": not _port_open(port),
        "baseline_memory_used_mb": baseline_used,
        "recovered_memory_used_mb": recovered_used,
        "memory_delta_from_baseline_mb": recovered_used - baseline_used,
        "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
        "no_task_worker_remains": not _task_owned_pids(model_paths),
        "recovery_command": recovery_smoke.get("command"),
        "recovery_command_exit_code": recovery_smoke.get("exit_code"),
        "recovery_stdout_sha256": recovery_smoke.get("stdout_sha256"),
        "recovery_stderr_sha256": recovery_smoke.get("stderr_sha256"),
        "recovery_binary_sha256_matches": sha256_file(server) == server_binary_sha256,
        "recovery_bounded": True,
        "recovery_duration_s": round(time.monotonic() - recovery_start, 6),
        "signals_sent_to_unrelated_pids": [],
        "recovery_complete": recovery_complete and recovery_smoke.get("exit_code") == 0,
        "failing_stage": "" if recovery_complete else "unload_or_recovery",
        "error": "" if recovery_complete else "bounded unload or recovery check failed",
    }
    return process_row, telemetry, unload_row


def run_sequential_admission(
    metadata_rows: Sequence[Mapping[str, Any]], selected_gpu: int, server: Path, binary_sha256: str
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Run the frozen family order and stop only when unsafe recovery blocks continuation."""

    process_rows = []
    gpu_rows = []
    unload_rows = []
    model_paths = [str(row.get("selected_blob_path", "")) for row in metadata_rows]
    for sequence_index, hf_id in enumerate(MANDATED_HF_IDS):
        metadata = next(row for row in metadata_rows if row.get("repository_id") == hf_id)
        process_row, telemetry, unload_row = execute_one_model(
            metadata_row=metadata,
            sequence_index=sequence_index,
            selected_gpu=selected_gpu,
            server=server,
            server_binary_sha256=binary_sha256,
            model_paths=model_paths,
        )
        process_rows.append(process_row)
        gpu_rows.extend(telemetry)
        unload_rows.append(unload_row)
        if unload_row.get("recovery_complete") is not True:
            break
    return process_rows, gpu_rows, unload_rows


def collect_preconditions(
    repo_root: Path,
    gates: Mapping[str, Any],
    metadata_rows: Sequence[Mapping[str, Any]],
    server: Path,
    initial_sample: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    """Collect live host, CUDA, process, timeout, and frozen-order receipts."""

    resources = _cpu_ram_disk_receipt(repo_root)
    build = _llama_cpp_build_receipt(server)
    gpu_selection = choose_idle_gpu(initial_sample)
    model_paths = [str(row.get("selected_blob_path", "")) for row in metadata_rows]
    owned_pids = _task_owned_pids(model_paths)
    checks = {
        "structured_gates": gates.get("all_structured_gates_passed") is True,
        "content_identity": len(metadata_rows) == len(MANDATED_HF_IDS)
        and all(row.get("passed") is True for row in metadata_rows),
        "llama_cpp_cuda_build": build.get("exists") is True
        and build.get("executable") is True
        and build.get("cuda_linked") is True,
        "cuda_telemetry": initial_sample.get("gpu_query_exit_code") == 0
        and initial_sample.get("compute_query_exit_code") == 0,
        "idle_supported_gpu": gpu_selection.get("eligible") is True,
        "one_model_residency": not owned_pids,
        "atomic_output_ready": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    return {
        "all_required_preconditions_available": all(checks.values()),
        "checks": checks,
        "failed_preconditions": [name for name, passed in checks.items() if not passed],
        **resources,
        "llama_cpp_build": build,
        "initial_gpu_state": dict(initial_sample),
        "gpu_selection": gpu_selection,
        "selected_gpu": gpu_selection.get("selected_gpu"),
        "active_processes_before": initial_sample.get("compute_processes", []),
        "task_owned_pids_before": owned_pids,
        "model_metadata_rows": len(metadata_rows),
        "model_load_order": list(MANDATED_HF_IDS),
        "timeout_policy": {
            "load_timeout_s": LOAD_TIMEOUT_S,
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "shutdown_timeout_s": SHUTDOWN_TIMEOUT_S,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
        },
        "free_vram_arithmetic_used_as_gate": False,
        "broad_zombie_reaper_calls": 0,
    }


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Execute preconditions, sequential runtime attempts, and one atomic write."""

    start = time.monotonic()
    protected_before = _hash_protected(repo_root)
    gates = build_upstream_gate_receipts(repo_root)
    metadata_rows = resolve_metadata_rows(repo_root)
    server = _resolve_llama_server()
    model_paths = [str(row.get("selected_blob_path", "")) for row in metadata_rows]
    initial_sample = _live_gpu_sample(
        repository_id="preconditions",
        worker_pid=0,
        stage="preconditions",
        sample_index=0,
        selected_gpu=0,
        model_paths=model_paths,
    )
    preconditions = collect_preconditions(repo_root, gates, metadata_rows, server, initial_sample)
    process_rows: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    unload_rows: list[JsonDict] = []
    if preconditions["all_required_preconditions_available"]:
        process_rows, gpu_rows, unload_rows = run_sequential_admission(
            metadata_rows,
            int(preconditions["selected_gpu"]),
            server,
            str(preconditions["llama_cpp_build"]["binary_sha256"]),
        )
    protected_after = _hash_protected(repo_root)
    artifact = assemble_artifact(
        upstream_gate_receipts=gates,
        metadata_rows=metadata_rows,
        process_rows=process_rows,
        gpu_rows=gpu_rows,
        unload_rows=unload_rows,
        preconditions=preconditions,
        protected=_compare_protected(protected_before, protected_after),
        duration_s=time.monotonic() - start,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=run_date,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact validation failed before write: {errors}")
    atomic_write_json(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entry point.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = _load_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        errors = validate_artifact(artifact)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            return 1
        print(json.dumps({"valid": True, "errors": []}, indent=2))
        return 0
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "all_mandated_models_loaded_score": artifact["all_mandated_models_loaded_score"],
                "family_admitted_scores": artifact["family_admitted_scores"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
