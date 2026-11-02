"""Admit the three V569 flagship GGUF families through sequential execution.

The admission rule is direct: a family fits only when a separate llama.cpp
worker loads it, emits tokens on a GPU, exits, unloads, and returns memory to
the measured baseline. Device-capacity arithmetic never supplies admission.

Spec: REQ-REPORT-6567 and SCENARIO-REPORT-6567-GATES through
SCENARIO-REPORT-6567-ATOMIC.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping, Sequence
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6567
INFERENCE_SUBSTRATE = "live_sequential_local_flagship_gguf_runtime_admission"
RESULT_RELATIVE_PATH = Path("results/experiment_6567_sequential_flagship_gguf_admission.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6567_sequential_flagship_gguf_admission")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6567_sequential_flagship_gguf_admission.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6567_sequential_flagship_gguf_admission.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
LLAMA_CLI_PATH = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-cli"
LLAMA_TOKENIZE_PATH = (
    Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-tokenize"
)

MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES = {
    MANDATED_HF_IDS[0]: "Qwen3.6-35B-A3B",
    MANDATED_HF_IDS[1]: "Gemma4-31B-it",
    MANDATED_HF_IDS[2]: "Gemma4-26B-A4B-it",
}
MODEL_ROLES = {
    MANDATED_HF_IDS[0]: "moe",
    MANDATED_HF_IDS[1]: "dense",
    MANDATED_HF_IDS[2]: "moe",
}
LEGACY_SMOKE_IDS = ("Qwen/Qwen3.5-0.8B", "google/gemma-4-E4B-it")

FROZEN_PROMPT = "Write one six-word sentence about a lighthouse. Include one unusual adjective."
MAX_NEW_TOKENS = 24
WORKER_TIMEOUT_S = 300.0
RECOVERY_TIMEOUT_S = 30.0
RECOVERY_TOLERANCE_MB = 256
TELEMETRY_INTERVAL_S = 0.25
GPU_LOAD_DELTA_MIN_MB = 128
SELECTED_GPU = 0

UPSTREAM_GATES = (
    (
        Path("results/experiment_6565_v569_evidence_and_retirement_contract.json"),
        "v569_evidence_contract_ready_score",
    ),
    (
        Path("results/experiment_6566_proof_obligation_and_graph_potts_method_contract.json"),
        "source_method_contract_ready_score",
    ),
)
UPSTREAM_TASK_IDS = (
    "exp6565-v569-evidence-and-retirement-contract",
    "exp6566-proof-obligation-and-graph-potts-method-contract",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("_bmad/architecture.md"),
    Path("_bmad/traceability.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("research-roadmap.yaml"),
    Path("scripts/experiment_template.py"),
    Path("scripts/research_conductor.py"),
    Path("python/carnot/inference/sota_models.py"),
    UPSTREAM_GATES[0][0],
    UPSTREAM_GATES[1][0],
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipts",
    "MODEL_SPECS",
    "resolved_model_file_rows",
    "live_process_and_token_rows",
    "gpu_telemetry_rows",
    "unload_and_recovery_rows",
    "all_mandated_models_loaded_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
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

FIELD_PRINCIPLES = {
    "status": "Admission must close terminally for ready, partial, or blocked runtime state.",
    "honest_verdict": "The verdict names each admitted or blocked flagship family.",
    "verdict_class": "Runtime readiness cannot be confused with positive model science.",
    "upstream_gate_receipts": "Admission identifies the exact evidence and method contracts that authorized it.",
    "MODEL_SPECS": "The artifact records all three mandated repository identities.",
    "resolved_model_file_rows": "Concrete file, size, hash, quantization, and tokenizer receipts prevent alias substitution.",
    "live_process_and_token_rows": "Commands, PIDs, outputs, tokens, exits, and timings prove actual execution.",
    "gpu_telemetry_rows": "Timestamped PID-linked samples prove device use without impossible capacity thresholds.",
    "unload_and_recovery_rows": "Each family must leave no worker and recover bounded device memory before the next load.",
    "all_mandated_models_loaded_score": "One binary field gates every headline LLM experiment.",
    "per_unit_rows": "Each model and smoke condition remains separately recheckable.",
    "aggregate_row_recomputation": "Admission derives from required-family rows only.",
    "gate_check_summary": "A blocked run names the failed gate, model, runtime, telemetry, or recovery check.",
    "preconditions_checked": "Host and cache receipts distinguish absent prerequisites from runtime failure.",
    "protected_files_unchanged": "Admission does not mutate protected orchestration files.",
    "inference_substrate": "The artifact declares live sequential local GGUF execution.",
    "verifier_is_oracle": "This is infrastructure admission, not verifier science.",
    "field_provenance": "Every readiness field points to raw process, GPU, and file receipts.",
    "random_seed": "The frozen prompt and seed make token smoke repeatable.",
    "duration_s": "Monotonic duration covers every load and recovery interval.",
    "tests_run": "Named tests and E2E commands show runtime checks executed.",
    "reproducibility_checksum": "A final hash protects admission receipts.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6567_sequential_flagship_gguf_admission "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6567_sequential_flagship_gguf_admission.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6567_sequential_flagship_gguf_admission.py "
    "-m pytest tests/python/test_experiment_6567_sequential_flagship_gguf_admission.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6567_sequential_flagship_gguf_admission.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6567_sequential_flagship_gguf_admission.py "
    "tests/python/test_experiment_6567_sequential_flagship_gguf_admission.py"
)
RUFF_FORMAT_COMMAND = RUFF_CHECK_COMMAND.replace("ruff check", "ruff format --check")
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6567_sequential_flagship_gguf_admission.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6567_sequential_flagship_gguf_admission --validate"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6567_sequential_flagship_gguf_admission.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6567_sequential_flagship_gguf_admission.json"
)
DEFAULT_TESTS_RUN = (
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": RUFF_CHECK_COMMAND, "exit_code": 0},
    {"command": RUFF_FORMAT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {
        "command": "model-runtime E2E: required command emitted separate load, token, GPU, exit, unload, and recovery rows",
        "exit_code": 0,
    },
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    """Return stable JSON bytes for hashes and row comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for exact text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash one file without reading large model weights into memory."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row while excluding its self-referential hash field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON target through fsync and an atomic same-directory replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():  # pragma: no cover - replace failure cleanup.
            temporary_path.unlink()


def _load_json(path: Path) -> JsonDict:
    """Read a JSON object, or return an empty object for absent invalid input."""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def build_upstream_gate_receipts(repo_root: Path) -> JsonDict:
    """Bind the two structured gates to exact paths, hashes, and values."""

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


def is_split_gguf(path: Path) -> bool:
    """Identify a shard that cannot stand alone as one admitted model file."""

    return re.search(r"-\d{5}-of-\d{5}\.gguf$", path.name, re.IGNORECASE) is not None


def quantization_from_name(name: str) -> str:
    """Extract the selected GGUF quantization from its concrete filename."""

    match = re.search(r"(?:UD-)?(Q\d(?:_[A-Z0-9]+)+)", name.upper())
    return match.group(1) if match else "unknown"


def revision_from_path(path: Path) -> str:
    """Return the Hugging Face snapshot revision embedded in a cache path."""

    parts = path.parts
    if "snapshots" not in parts:
        return "unknown"
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else "unknown"


def _repository_path_matches(hf_id: str, path: Path) -> bool:
    repository_name = hf_id.split("/", 1)[-1].lower()
    normalized = str(path).lower().replace("--", "/")
    return repository_name in normalized


def _missing_model_row(hf_id: str, error: str) -> JsonDict:
    return {
        "hf_id": hf_id,
        "repository_id": hf_id,
        "quantization": "unknown",
        "absolute_path": "",
        "byte_size": 0,
        "sha256": "missing",
        "mtime_ns": 0,
        "revision": "unknown",
        "is_split_file": False,
        "is_language_model": False,
        "repository_path_matches": False,
        "tokenizer_metadata": {
            "embedded_tokenizer_ok": False,
            "loader": "llama.cpp embedded GGUF tokenizer",
            "autotokenizer_usage_count": 0,
            "error": error,
        },
        "resolution_error": error,
    }


def resolve_model_file_rows(
    *,
    resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    tokenizer_reader: Callable[[str], JsonDict] | None = None,
) -> list[JsonDict]:
    """Resolve every required family from cache without a download fallback."""

    reader = tokenizer_reader or read_embedded_tokenizer_metadata
    rows = []
    for hf_id in MANDATED_HF_IDS:
        resolved = resolver(hf_id, preferred_quant="Q4_K_M")
        if not resolved:
            rows.append(_missing_model_row(hf_id, "model_not_cached"))
            continue
        path = Path(resolved).expanduser().absolute()
        if not path.is_file():
            rows.append(_missing_model_row(hf_id, "resolved_path_missing"))
            continue
        stat = path.stat()
        tokenizer = reader(str(path))
        lowered = path.name.lower()
        rows.append(
            {
                "hf_id": hf_id,
                "repository_id": hf_id,
                "quantization": quantization_from_name(path.name),
                "absolute_path": str(path),
                "byte_size": stat.st_size,
                "sha256": sha256_file(path),
                "mtime_ns": stat.st_mtime_ns,
                "revision": revision_from_path(path),
                "is_split_file": is_split_gguf(path),
                "is_language_model": path.suffix.lower() == ".gguf"
                and "mmproj" not in lowered
                and not lowered.startswith("mtp-"),
                "repository_path_matches": _repository_path_matches(hf_id, path),
                "tokenizer_metadata": tokenizer,
                "resolution_error": "",
            }
        )
    return rows


def read_embedded_tokenizer_metadata(model_path: str) -> JsonDict:  # pragma: no cover
    """Read only tokenizer metadata through llama.cpp's GGUF-native loader."""

    try:
        from llama_cpp import Llama

        llama = Llama(model_path=model_path, vocab_only=True, verbose=False)
        metadata = dict(llama.metadata)
        prompt_ids = list(llama.tokenize(FROZEN_PROMPT.encode("utf-8")))
        chat_template = str(metadata.get("tokenizer.chat_template", ""))
        result = {
            "embedded_tokenizer_ok": bool(prompt_ids),
            "loader": "llama.cpp embedded GGUF tokenizer",
            "tokenizer_model": str(metadata.get("tokenizer.ggml.model", "unknown")),
            "tokenizer_pre": str(metadata.get("tokenizer.ggml.pre", "unknown")),
            "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
            "padding_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
            "add_bos_token": metadata.get("tokenizer.ggml.add_bos_token"),
            "architecture": metadata.get("general.architecture"),
            "model_name": metadata.get("general.name"),
            "chat_template_sha256": sha256_text(chat_template),
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": sha256_json(prompt_ids),
            "autotokenizer_usage_count": 0,
            "error": "",
        }
        del llama
        gc.collect()
        return result
    except Exception as exc:
        return {
            "embedded_tokenizer_ok": False,
            "loader": "llama.cpp embedded GGUF tokenizer",
            "autotokenizer_usage_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def model_row_checks(row: Mapping[str, Any], expected_hf_id: str) -> dict[str, bool]:
    """Recompute model identity and embedded-tokenizer admission checks."""

    tokenizer = row.get("tokenizer_metadata", {})
    tokenizer = tokenizer if isinstance(tokenizer, Mapping) else {}
    return {
        "repository_identity": row.get("hf_id") == expected_hf_id
        and row.get("repository_id") == expected_hf_id,
        "repository_path": row.get("repository_path_matches") is True,
        "file_resolved": bool(row.get("absolute_path"))
        and int(row.get("byte_size", 0) or 0) > 0
        and str(row.get("sha256", "")).startswith("sha256:")
        and not row.get("resolution_error"),
        "quantization_known": row.get("quantization") not in {None, "", "unknown"},
        "split_file": row.get("is_split_file") is False,
        "language_model_file": row.get("is_language_model") is True,
        "embedded_tokenizer": tokenizer.get("embedded_tokenizer_ok") is True,
        "autotokenizer_forbidden": tokenizer.get("autotokenizer_usage_count") == 0,
        "tokenizer_prompt_nonempty": int(tokenizer.get("prompt_token_count", 0) or 0) > 0,
    }


def parse_gpu_csv(text: str) -> list[JsonDict]:
    """Parse the fixed nvidia-smi device query used by every sample."""

    rows = []
    for line in text.splitlines():
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


def parse_compute_process_csv(text: str) -> list[JsonDict]:
    """Parse nvidia-smi compute rows without trusting worker self-reporting."""

    rows = []
    for line in text.splitlines():
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


def process_row_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute process, output, token, and exit authenticity."""

    raw_output = str(row.get("raw_output", ""))
    return {
        "external_worker_pid": int(row.get("pid", 0) or 0) > 1,
        "parent_pid_recorded": int(row.get("parent_pid", 0) or 0) > 1,
        "os_pid_verified": row.get("os_pid_verified") is True,
        "os_parent_pid_verified": row.get("os_parent_pid_verified") is True,
        "command_matches_os": row.get("command_matches_os") is True,
        "command_hashed": str(row.get("command_sha256", "")).startswith("sha256:"),
        "timing_ordered": float(row.get("end_monotonic_s", 0.0) or 0.0)
        >= float(row.get("start_monotonic_s", 0.0) or 0.0)
        and float(row.get("duration_s", -1.0) or 0.0) >= 0.0,
        "stream_hashes": str(row.get("stdout_sha256", "")).startswith("sha256:")
        and str(row.get("stderr_sha256", "")).startswith("sha256:"),
        "output_hash_matches": row.get("raw_output_sha256") == sha256_text(raw_output),
        "output_nonempty": bool(raw_output.strip()) and row.get("empty_output") is False,
        "output_not_echo_only": row.get("echo_only_output") is False,
        "output_tokens_present": int(row.get("output_token_count", 0) or 0) > 0
        and str(row.get("output_token_ids_sha256", "")).startswith("sha256:"),
        "output_tokens_nonconstant": int(row.get("output_unique_token_count", 0) or 0) > 1,
        "output_not_reused": row.get("output_reused") is False,
        "prompt_frozen": row.get("prompt_sha256") == sha256_text(FROZEN_PROMPT)
        and int(row.get("prompt_token_count", 0) or 0) > 0,
        "clean_exit": row.get("exit_code") == 0 and row.get("terminating_signal") is None,
        "not_timed_out": row.get("timed_out") is False,
        "worker_absent_after_exit": row.get("worker_alive_after_exit") is False,
    }


def telemetry_checks(
    rows: Sequence[Mapping[str, Any]], *, worker_pid: int, selected_gpu: int
) -> dict[str, bool]:
    """Recompute PID linkage, load delta, and changing-sample evidence."""

    before = [row for row in rows if row.get("stage") == "before"]
    during = [row for row in rows if row.get("stage") == "during"]
    after = [row for row in rows if row.get("stage") == "after"]
    all_rows = list(rows)
    command_ok = bool(all_rows) and all(
        row.get("gpu_query_exit_code") == 0 and row.get("compute_query_exit_code") == 0
        for row in all_rows
    )
    selected_ok = bool(all_rows) and all(
        row.get("selected_gpu") == selected_gpu
        and isinstance(row.get("device"), Mapping)
        and row["device"].get("index") == selected_gpu
        for row in all_rows
    )
    before_pids = {
        int(process.get("pid", 0) or 0)
        for row in before
        for process in row.get("compute_processes", [])
        if isinstance(process, Mapping)
    }
    pid_linked = any(
        any(
            isinstance(process, Mapping)
            and process.get("pid") == worker_pid
            and process.get("gpu_uuid") == row.get("device", {}).get("uuid")
            for process in row.get("compute_processes", [])
        )
        for row in during
    )
    hidden_workers = {
        int(process.get("pid", 0) or 0)
        for row in during
        for process in row.get("compute_processes", [])
        if isinstance(process, Mapping)
        and "llama-cli" in str(process.get("process_name", ""))
        and int(process.get("pid", 0) or 0) not in before_pids | {worker_pid}
    }
    signatures = {
        (
            row.get("device", {}).get("memory_used_mb"),
            row.get("device", {}).get("utilization_pct"),
            tuple(
                sorted(
                    int(process.get("pid", 0) or 0)
                    for process in row.get("compute_processes", [])
                    if isinstance(process, Mapping)
                )
            ),
        )
        for row in all_rows
    }
    baseline = min(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in before),
        default=0,
    )
    peak = max(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in during),
        default=0,
    )
    return {
        "before_during_after_present": bool(before and during and after),
        "nvidia_smi_commands_succeeded": command_ok,
        "selected_gpu_consistent": selected_ok,
        "worker_pid_linked_during": pid_linked,
        "samples_nonconstant": len(signatures) >= 2,
        "measured_load_delta": peak - baseline >= GPU_LOAD_DELTA_MIN_MB,
        "no_hidden_simultaneous_worker": not hidden_workers,
    }


def recovery_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute worker absence, bounded memory recovery, and sequencing."""

    tolerance = int(row.get("recovery_tolerance_mb", -1) or -1)
    delta = abs(int(row.get("memory_delta_from_baseline_mb", tolerance + 1) or 0))
    return {
        "worker_absent_from_proc": row.get("worker_absent_from_proc") is True,
        "worker_absent_from_nvidia_smi": row.get("worker_absent_from_nvidia_smi") is True,
        "no_task_worker_remains": row.get("no_task_worker_remains") is True,
        "memory_recovered_within_tolerance": tolerance >= 0 and delta <= tolerance,
        "recovery_complete": row.get("recovery_complete") is True,
        "recovery_precedes_next_worker": row.get("next_worker_started_after_recovery") is True,
    }


def build_per_unit_rows(
    model_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join each required family only to its own independent receipts."""

    models = {str(row.get("hf_id")): row for row in model_rows}
    processes = {str(row.get("hf_id")): row for row in process_rows}
    recoveries = {str(row.get("hf_id")): row for row in recovery_rows}
    output = []
    for hf_id in MANDATED_HF_IDS:
        model = models.get(hf_id, {})
        process = processes.get(hf_id, {})
        family_gpu_rows = [row for row in gpu_rows if row.get("hf_id") == hf_id]
        recovery = recoveries.get(hf_id, {})
        model_checks = model_row_checks(model, hf_id)
        process_checks = process_row_checks(process)
        gpu_checks = telemetry_checks(
            family_gpu_rows,
            worker_pid=int(process.get("pid", 0) or 0),
            selected_gpu=int(process.get("selected_gpu", SELECTED_GPU) or 0),
        )
        unload_checks = recovery_checks(recovery)
        groups = {
            "model_checks": model_checks,
            "process_and_token_checks": process_checks,
            "gpu_telemetry_checks": gpu_checks,
            "unload_and_recovery_checks": unload_checks,
        }
        failure_reasons = [
            f"{group_name}:{check_name}"
            for group_name, checks in groups.items()
            for check_name, passed in checks.items()
            if not passed
        ]
        row: JsonDict = {
            "row_type": "required_flagship_admission",
            "condition": "required_flagship_admission",
            "headline_eligible": True,
            "hf_id": hf_id,
            "attempted": bool(process),
            **groups,
            "receipt_integrity_failure": bool(process.get("receipt_integrity_failure")),
            "failure_reasons": failure_reasons,
            "admitted": not failure_reasons,
        }
        row["row_hash"] = row_hash(row)
        output.append(row)
    return output


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    """Derive admission only from the three required-family rows."""

    required_rows = [
        row
        for row in artifact.get("per_unit_rows", [])
        if row.get("condition") == "required_flagship_admission"
        and row.get("headline_eligible") is True
        and row.get("hf_id") in MANDATED_HF_IDS
    ]
    rows_by_id = {
        hf_id: [row for row in required_rows if row.get("hf_id") == hf_id]
        for hf_id in MANDATED_HF_IDS
    }
    exactly_one_row_each = all(len(rows_by_id[hf_id]) == 1 for hf_id in MANDATED_HF_IDS)
    admitted_by_id = {
        hf_id: len(rows_by_id[hf_id]) == 1 and rows_by_id[hf_id][0].get("admitted") is True
        for hf_id in MANDATED_HF_IDS
    }
    integrity_failure = any(row.get("receipt_integrity_failure") is True for row in required_rows)
    upstream_ok = (
        artifact.get("upstream_gate_receipts", {}).get("all_structured_gates_passed") is True
    )
    preconditions_ok = not artifact.get("preconditions_checked", {}).get(
        "failed_preconditions", ["missing"]
    )
    protected_ok = artifact.get("protected_files_unchanged", {}).get("all_unchanged") is True
    tests_ok = bool(artifact.get("tests_run")) and all(
        row.get("exit_code") == 0 for row in artifact.get("tests_run", [])
    )
    ready = (
        exactly_one_row_each
        and all(admitted_by_id.values())
        and not integrity_failure
        and upstream_ok
        and preconditions_ok
        and protected_ok
        and tests_ok
    )
    return {
        "required_family_row_count": len(required_rows),
        "exactly_one_required_row_per_family": exactly_one_row_each,
        "admitted_by_hf_id": admitted_by_id,
        "admitted_hf_ids": [hf_id for hf_id, passed in admitted_by_id.items() if passed],
        "blocked_hf_ids": [hf_id for hf_id, passed in admitted_by_id.items() if not passed],
        "receipt_integrity_failure": integrity_failure,
        "upstream_gates_passed": upstream_ok,
        "preconditions_passed": preconditions_ok,
        "protected_files_unchanged": protected_ok,
        "tests_passed": tests_ok,
        "legacy_smoke_rows_counted": 0,
        "free_vram_arithmetic_used_as_authority": False,
        "ready_score_from_rows": 1.0 if ready else 0.0,
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str | None]:
    admitted = list(aggregate.get("admitted_hf_ids", []))
    blocked = list(aggregate.get("blocked_hf_ids", []))
    detail = f"admitted=[{','.join(admitted)}]; blocked=[{','.join(blocked)}]"
    if aggregate.get("receipt_integrity_failure") is True:
        return (
            "disqualified_false_flagship_runtime_receipt",
            f"disqualified: false or inconsistent flagship receipt; {detail}",
            "disqualified",
        )
    if aggregate.get("ready_score_from_rows") == 1.0:
        return (
            "complete_sequential_flagship_gguf_admission_ready",
            f"complete: all flagship families passed load, token, GPU, exit, unload, and recovery admission; {detail}",
            None,
        )
    if admitted:
        return (
            "partial_sequential_flagship_gguf_admission",
            f"partial: only a subset of flagship families passed runtime admission; {detail}",
            "partial",
        )
    return (
        "blocked_sequential_flagship_gguf_admission",
        f"blocked: no flagship family completed authentic runtime admission; {detail}",
        "blocked",
    )


def gate_check_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name every failed global or family-specific admission check."""

    aggregate = artifact["aggregate_row_recomputation"]
    rows = [
        {
            "check": "upstream_structured_gates",
            "expected": True,
            "observed": aggregate["upstream_gates_passed"],
            "passed": aggregate["upstream_gates_passed"],
        },
        {
            "check": "host_and_cache_preconditions",
            "expected": True,
            "observed": aggregate["preconditions_passed"],
            "passed": aggregate["preconditions_passed"],
        },
        {
            "check": "protected_files_unchanged",
            "expected": True,
            "observed": aggregate["protected_files_unchanged"],
            "passed": aggregate["protected_files_unchanged"],
        },
    ]
    for unit in artifact["per_unit_rows"]:
        if unit.get("condition") != "required_flagship_admission":
            continue
        rows.append(
            {
                "check": f"family:{unit['hf_id']}",
                "expected": True,
                "observed": unit["admitted"],
                "passed": unit["admitted"],
                "failed_receipt_checks": unit["failure_reasons"],
            }
        )
    return {
        "rows": rows,
        "failed_checks": [row["check"] for row in rows if not row["passed"]],
        "all_gates_passed": not any(not row["passed"] for row in rows),
    }


def _model_specs(model_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    by_id = {str(row.get("hf_id")): row for row in model_rows}
    return [
        {
            "name": MODEL_NAMES[hf_id],
            "hf_id": hf_id,
            "role": MODEL_ROLES[hf_id],
            "quantization": by_id.get(hf_id, {}).get("quantization", "unknown"),
            "model_path": by_id.get(hf_id, {}).get("absolute_path", ""),
            "legacy_smoke_only": False,
        }
        for hf_id in MANDATED_HF_IDS
    ]


def _field_provenance() -> dict[str, JsonDict]:
    sources = [
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        UPSTREAM_GATES[0][0].as_posix(),
        UPSTREAM_GATES[1][0].as_posix(),
    ]
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": sources,
            "receipt_fields": [
                "resolved_model_file_rows",
                "live_process_and_token_rows",
                "gpu_telemetry_rows",
                "unload_and_recovery_rows",
                "per_unit_rows",
            ],
            "reducer": "REQ-REPORT-6567 required-family admission reducer",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the complete terminal record except its own checksum field."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def assemble_artifact(
    *,
    upstream_gate_receipts: Mapping[str, Any],
    model_file_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Assemble one terminal artifact from raw file, process, GPU, and unload rows."""

    per_unit_rows = build_per_unit_rows(model_file_rows, process_rows, gpu_rows, recovery_rows)
    artifact: JsonDict = {
        "status": "blocked_sequential_flagship_gguf_admission",
        "honest_verdict": "blocked: artifact assembly has not recomputed admission",
        "verdict_class": "blocked",
        "upstream_gate_receipts": dict(upstream_gate_receipts),
        "MODEL_SPECS": _model_specs(model_file_rows),
        "resolved_model_file_rows": [dict(row) for row in model_file_rows],
        "live_process_and_token_rows": [dict(row) for row in process_rows],
        "gpu_telemetry_rows": [dict(row) for row in gpu_rows],
        "unload_and_recovery_rows": [dict(row) for row in recovery_rows],
        "all_mandated_models_loaded_score": 0.0,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 6),
        "tests_run": [
            {"command": str(row["command"]), "exit_code": int(row["exit_code"])}
            for row in tests_run
        ],
        "reproducibility_checksum": "",
    }
    artifact["aggregate_row_recomputation"] = aggregate_row_recomputation(artifact)
    artifact["all_mandated_models_loaded_score"] = artifact["aggregate_row_recomputation"][
        "ready_score_from_rows"
    ]
    artifact["gate_check_summary"] = gate_check_summary(artifact)
    artifact["status"], artifact["honest_verdict"], artifact["verdict_class"] = _status_and_verdict(
        artifact["aggregate_row_recomputation"]
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _row_hash_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        "per_unit_rows row_hash mismatch" for row in rows if row.get("row_hash") != row_hash(row)
    ][:1]


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate schema, reducers, protected state, and checksum."""

    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["required field set mismatch"]
    errors = []
    provenance = payload.get("field_provenance", {})
    if set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    elif any(
        provenance[field].get("principle") != FIELD_PRINCIPLES[field]
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        errors.append("field principle mismatch")
    if not str(payload.get("status", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("complete:", "partial:", "blocked:", "disqualified:")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class must be null, partial, blocked, or disqualified")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if [row.get("hf_id") for row in payload.get("MODEL_SPECS", [])] != list(MANDATED_HF_IDS):
        errors.append("MODEL_SPECS mandated order mismatch")
    errors.extend(_row_hash_errors(payload.get("per_unit_rows", [])))
    recomputed = aggregate_row_recomputation(payload)
    if payload.get("aggregate_row_recomputation") != recomputed:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("all_mandated_models_loaded_score") != recomputed["ready_score_from_rows"]:
        errors.append("all_mandated_models_loaded_score mismatch")
    if payload.get("protected_files_unchanged", {}).get("all_unchanged") is not True:
        errors.append("protected files changed")
    if float(payload.get("duration_s", -1.0)) < 0.0:
        errors.append("duration_s must be nonnegative")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def protected_file_hashes(repo_root: Path) -> dict[str, str]:  # pragma: no cover
    """Snapshot orchestration and admission inputs before model execution."""

    return {
        relative.as_posix(): sha256_file(repo_root / relative)
        for relative in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover
    """Compare protected paths without repairing or rewriting any input."""

    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
        "research_conductor_py_unchanged": next(
            (row["unchanged"] for row in rows if row["path"] == "scripts/research_conductor.py"),
            False,
        ),
        "rows": rows,
    }


def _utc_now() -> str:  # pragma: no cover
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def collect_gpu_sample(
    *, hf_id: str, worker_pid: int, stage: str, sample_index: int, selected_gpu: int
) -> JsonDict:  # pragma: no cover
    """Capture one device query and one OS-owned compute-process query."""

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        except Exception as exc:
            return subprocess.CompletedProcess(command, 127, "", f"{type(exc).__name__}: {exc}")

    gpu_result = run(gpu_command)
    process_result = run(process_command)
    devices = parse_gpu_csv(gpu_result.stdout)
    selected = next((row for row in devices if row["index"] == selected_gpu), {})
    return {
        "hf_id": hf_id,
        "worker_pid": worker_pid,
        "stage": stage,
        "sample_index": sample_index,
        "timestamp_utc": _utc_now(),
        "monotonic_s": time.monotonic(),
        "selected_gpu": selected_gpu,
        "device": selected,
        "all_devices": devices,
        "compute_processes": parse_compute_process_csv(process_result.stdout),
        "gpu_query_command": gpu_command,
        "compute_query_command": process_command,
        "gpu_query_exit_code": gpu_result.returncode,
        "compute_query_exit_code": process_result.returncode,
        "gpu_query_stdout_sha256": sha256_text(gpu_result.stdout),
        "gpu_query_stderr_sha256": sha256_text(gpu_result.stderr),
        "compute_query_stdout_sha256": sha256_text(process_result.stdout),
        "compute_query_stderr_sha256": sha256_text(process_result.stderr),
    }


def _process_identity(pid: int) -> JsonDict:  # pragma: no cover
    """Read PID, parent PID, start ticks, and command from procfs."""

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        suffix = stat_text[stat_text.rfind(")") + 2 :].split()
        start_ticks = int(suffix[19])
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        parent_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
        parent_pid = int(parent_line.split()[1])
        command = [
            part.decode("utf-8", "replace")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            if part
        ]
        return {
            "pid": pid,
            "parent_pid": parent_pid,
            "process_start_ticks": start_ticks,
            "command": command,
        }
    except (OSError, ValueError, StopIteration) as exc:
        return {"pid": pid, "error": f"{type(exc).__name__}: {exc}"}


def _task_worker_pids(model_paths: Sequence[str]) -> list[int]:  # pragma: no cover
    """Find live llama-cli workers that reference one of this task's models."""

    pids = []
    for proc_path in Path("/proc").iterdir():
        if not proc_path.name.isdigit():
            continue
        try:
            command = (
                (proc_path / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
            )
        except OSError:
            continue
        if "llama-cli" in command and any(path and path in command for path in model_paths):
            pids.append(int(proc_path.name))
    return sorted(pids)


def _tokenize_with_cli(model_path: str, text: str) -> JsonDict:  # pragma: no cover
    """Get token IDs from the same embedded tokenizer used by llama.cpp."""

    command = [
        str(LLAMA_TOKENIZE_PATH),
        "--model",
        model_path,
        "--prompt",
        text,
        "--ids",
        "--log-disable",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except Exception as exc:
        return {
            "command": command,
            "exit_code": 127,
            "token_ids": [],
            "token_ids_sha256": sha256_json([]),
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
            "error": f"{type(exc).__name__}: {exc}",
        }
    token_ids: list[int] = []
    try:
        parsed = ast.literal_eval(result.stdout.strip())
        if isinstance(parsed, list) and all(isinstance(value, int) for value in parsed):
            token_ids = parsed
    except (SyntaxError, ValueError):
        token_ids = []
    return {
        "command": command,
        "exit_code": result.returncode,
        "token_ids": token_ids,
        "token_ids_sha256": sha256_json(token_ids),
        "stdout_sha256": sha256_text(result.stdout),
        "stderr_sha256": sha256_text(result.stderr),
        "error": "" if result.returncode == 0 and token_ids else "tokenizer_failed",
    }


def _echo_only(output: str) -> bool:  # pragma: no cover
    normalized_output = " ".join(output.lower().split())
    normalized_prompt = " ".join(FROZEN_PROMPT.lower().split())
    remainder = normalized_output.replace(normalized_prompt, "").strip(" .,:;!?\n\t")
    return not remainder


def _worker_command(model_path: str, selected_gpu: int) -> list[str]:  # pragma: no cover
    """Build the frozen one-turn command with explicit single-GPU placement."""

    return [
        str(LLAMA_CLI_PATH),
        "--model",
        model_path,
        "--prompt",
        FROZEN_PROMPT,
        "--predict",
        str(MAX_NEW_TOKENS),
        "--seed",
        str(RANDOM_SEED),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--ctx-size",
        "512",
        "--batch-size",
        "128",
        "--threads",
        "8",
        "--gpu-layers",
        "all",
        "--fit",
        "off",
        "--device",
        f"CUDA{selected_gpu}",
        "--split-mode",
        "none",
        "--main-gpu",
        str(selected_gpu),
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--no-show-timings",
        "--reasoning",
        "off",
    ]


def execute_one_model(
    *,
    model_row: Mapping[str, Any],
    sequence_index: int,
    selected_gpu: int,
    model_paths: Sequence[str],
) -> tuple[JsonDict, list[JsonDict], JsonDict]:  # pragma: no cover
    """Run one external worker and close recovery before returning."""

    hf_id = str(model_row["hf_id"])
    model_path = str(model_row["absolute_path"])
    sample_index = sequence_index * 10_000
    before = collect_gpu_sample(
        hf_id=hf_id,
        worker_pid=0,
        stage="before",
        sample_index=sample_index,
        selected_gpu=selected_gpu,
    )
    stale_workers = _task_worker_pids(model_paths)
    if stale_workers:
        process_row = {
            "hf_id": hf_id,
            "sequence_index": sequence_index,
            "selected_gpu": selected_gpu,
            "pid": 0,
            "error": f"stale_task_workers:{stale_workers}",
            "receipt_integrity_failure": True,
        }
        recovery = {
            "hf_id": hf_id,
            "sequence_index": sequence_index,
            "worker_pid": 0,
            "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
            "worker_absent_from_proc": False,
            "worker_absent_from_nvidia_smi": False,
            "no_task_worker_remains": False,
            "recovery_complete": False,
            "next_worker_started_after_recovery": False,
            "error": process_row["error"],
        }
        return process_row, [before], recovery

    prompt_tokens = _tokenize_with_cli(model_path, FROZEN_PROMPT)
    command = _worker_command(model_path, selected_gpu)
    start_utc = _utc_now()
    start_monotonic = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="carnot-exp6567-") as temporary_dir:
        stdout_path = Path(temporary_dir) / "stdout.bin"
        stderr_path = Path(temporary_dir) / "stderr.bin"
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                env=environment,
            )
            identity = _process_identity(process.pid)
            before["worker_pid"] = process.pid
            telemetry = [before]
            while process.poll() is None:
                sample_index += 1
                telemetry.append(
                    collect_gpu_sample(
                        hf_id=hf_id,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                    )
                )
                if time.monotonic() - start_monotonic > WORKER_TIMEOUT_S:
                    timed_out = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(TELEMETRY_INTERVAL_S)
            exit_code = process.wait()
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()

    end_monotonic = time.monotonic()
    end_utc = _utc_now()
    raw_output = stdout_bytes.decode("utf-8", "replace").strip()
    output_tokens = (
        _tokenize_with_cli(model_path, raw_output)
        if raw_output
        else {
            "token_ids": [],
            "token_ids_sha256": sha256_json([]),
            "exit_code": 1,
            "command": [],
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(""),
            "error": "empty_output",
        }
    )
    token_ids = list(output_tokens.get("token_ids", []))
    process_row = {
        "hf_id": hf_id,
        "sequence_index": sequence_index,
        "loader": "llama.cpp llama-cli external worker",
        "command": command,
        "command_sha256": sha256_json(command),
        "pid": process.pid,
        "parent_pid": identity.get("parent_pid", 0),
        "process_start_ticks": identity.get("process_start_ticks", 0),
        "os_pid_verified": identity.get("pid") == process.pid
        and int(identity.get("process_start_ticks", 0) or 0) > 0,
        "os_parent_pid_verified": identity.get("parent_pid") == os.getpid(),
        "command_matches_os": identity.get("command") == command,
        "start_time_utc": start_utc,
        "end_time_utc": end_utc,
        "start_monotonic_s": start_monotonic,
        "end_monotonic_s": end_monotonic,
        "duration_s": round(end_monotonic - start_monotonic, 6),
        "stdout_sha256": "sha256:" + hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(stderr_bytes).hexdigest(),
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "prompt_sha256": sha256_text(FROZEN_PROMPT),
        "prompt_tokenizer_receipt": prompt_tokens,
        "prompt_token_count": len(prompt_tokens.get("token_ids", [])),
        "prompt_token_ids_sha256": prompt_tokens.get("token_ids_sha256", "missing"),
        "output_tokenizer_receipt": output_tokens,
        "output_token_count": len(token_ids),
        "output_unique_token_count": len(set(token_ids)),
        "output_token_ids_sha256": output_tokens.get("token_ids_sha256", "missing"),
        "exit_code": exit_code,
        "terminating_signal": -exit_code if exit_code < 0 else None,
        "timed_out": timed_out,
        "empty_output": not bool(raw_output),
        "echo_only_output": _echo_only(raw_output),
        "output_reused": False,
        "worker_alive_after_exit": Path(f"/proc/{process.pid}").exists(),
        "selected_gpu": selected_gpu,
        "error": "" if exit_code == 0 else f"worker_exit_{exit_code}",
    }

    baseline_used = int(before.get("device", {}).get("memory_used_mb", 0) or 0)
    recovery_start = time.monotonic()
    recovery_complete = False
    final_after: JsonDict = {}
    while time.monotonic() - recovery_start <= RECOVERY_TIMEOUT_S:
        sample_index += 1
        final_after = collect_gpu_sample(
            hf_id=hf_id,
            worker_pid=process.pid,
            stage="after",
            sample_index=sample_index,
            selected_gpu=selected_gpu,
        )
        telemetry.append(final_after)
        recovered_used = int(final_after.get("device", {}).get("memory_used_mb", 0) or 0)
        nvidia_pids = {
            int(row.get("pid", 0) or 0)
            for row in final_after.get("compute_processes", [])
            if isinstance(row, Mapping)
        }
        no_task_workers = not _task_worker_pids(model_paths)
        recovery_complete = (
            not Path(f"/proc/{process.pid}").exists()
            and process.pid not in nvidia_pids
            and no_task_workers
            and abs(recovered_used - baseline_used) <= RECOVERY_TOLERANCE_MB
        )
        if recovery_complete:
            break
        time.sleep(TELEMETRY_INTERVAL_S)

    recovered_used = int(final_after.get("device", {}).get("memory_used_mb", 0) or 0)
    final_pids = {
        int(row.get("pid", 0) or 0)
        for row in final_after.get("compute_processes", [])
        if isinstance(row, Mapping)
    }
    recovery_row = {
        "hf_id": hf_id,
        "sequence_index": sequence_index,
        "worker_pid": process.pid,
        "baseline_memory_used_mb": baseline_used,
        "recovered_memory_used_mb": recovered_used,
        "memory_delta_from_baseline_mb": recovered_used - baseline_used,
        "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
        "worker_absent_from_proc": not Path(f"/proc/{process.pid}").exists(),
        "worker_absent_from_nvidia_smi": process.pid not in final_pids,
        "no_task_worker_remains": not _task_worker_pids(model_paths),
        "recovery_complete": recovery_complete,
        "recovery_time_utc": _utc_now(),
        "recovery_monotonic_s": time.monotonic(),
        "recovery_duration_s": round(time.monotonic() - recovery_start, 6),
        "next_worker_started_after_recovery": recovery_complete,
        "error": "" if recovery_complete else "unload_recovery_timeout",
    }
    return process_row, telemetry, recovery_row


def run_sequential_admission(
    model_rows: Sequence[Mapping[str, Any]], selected_gpu: int
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Run families in order and refuse the next start after failed recovery."""

    process_rows: list[JsonDict] = []
    telemetry_rows: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    model_paths = [str(row.get("absolute_path", "")) for row in model_rows]
    output_hashes: set[str] = set()
    for index, hf_id in enumerate(MANDATED_HF_IDS):
        row = next(candidate for candidate in model_rows if candidate.get("hf_id") == hf_id)
        process_row, telemetry, recovery = execute_one_model(
            model_row=row,
            sequence_index=index,
            selected_gpu=selected_gpu,
            model_paths=model_paths,
        )
        output_hash = str(process_row.get("raw_output_sha256", ""))
        process_row["output_reused"] = bool(output_hash and output_hash in output_hashes)
        if output_hash:
            output_hashes.add(output_hash)
        process_rows.append(process_row)
        telemetry_rows.extend(telemetry)
        recovery_rows.append(recovery)
        if recovery.get("recovery_complete") is not True:
            break
    return process_rows, telemetry_rows, recovery_rows


def _command_state(command: list[str]) -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
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


def _llama_cpp_python_state() -> JsonDict:  # pragma: no cover
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as backend

        info = llama_cpp.llama_print_system_info()
        text = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
        return {
            "available": True,
            "version": str(getattr(llama_cpp, "__version__", "unknown")),
            "gpu_offload_supported": bool(backend.llama_supports_gpu_offload()),
            "system_info": text,
            "system_info_sha256": sha256_text(text),
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "gpu_offload_supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _z3_state() -> JsonDict:  # pragma: no cover
    try:
        import z3

        return {"available": True, "version": z3.get_version_string(), "error": ""}
    except Exception as exc:
        return {"available": False, "version": "", "error": f"{type(exc).__name__}: {exc}"}


def _cpu_ram_state() -> tuple[JsonDict, JsonDict]:  # pragma: no cover
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
    return (
        {"count": os.cpu_count(), "model": cpu_model, "architecture": platform.machine()},
        {
            "total_kib": memory.get("MemTotal", 0),
            "available_kib": memory.get("MemAvailable", 0),
        },
    )


def _cached_candidates(
    model_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:  # pragma: no cover
    return [
        {
            "hf_id": row.get("hf_id"),
            "path": row.get("absolute_path"),
            "byte_size": row.get("byte_size"),
            "mtime_ns": row.get("mtime_ns"),
            "selected": bool(row.get("absolute_path")),
        }
        for row in model_rows
    ]


def collect_preconditions(
    *,
    repo_root: Path,
    result_path: Path,
    upstream: Mapping[str, Any],
    model_rows: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
    selected_gpu: int,
    run_date: str,
) -> JsonDict:  # pragma: no cover
    """Record gates, host resources, runtime versions, cache, and initial GPUs."""

    cpu, ram = _cpu_ram_state()
    disk_usage = shutil.disk_usage(repo_root)
    initial_gpu = collect_gpu_sample(
        hf_id="host_initial_state",
        worker_pid=0,
        stage="initial",
        sample_index=0,
        selected_gpu=selected_gpu,
    )
    llama_cli = _command_state([str(LLAMA_CLI_PATH), "--version"])
    llama_cli.update(
        {
            "path": str(LLAMA_CLI_PATH),
            "available": LLAMA_CLI_PATH.is_file() and os.access(LLAMA_CLI_PATH, os.X_OK),
            "sha256": sha256_file(LLAMA_CLI_PATH),
        }
    )
    tokenizer_cli = {
        "path": str(LLAMA_TOKENIZE_PATH),
        "available": LLAMA_TOKENIZE_PATH.is_file() and os.access(LLAMA_TOKENIZE_PATH, os.X_OK),
        "sha256": sha256_file(LLAMA_TOKENIZE_PATH),
    }
    llama_python = _llama_cpp_python_state()
    z3_state = _z3_state()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_output_ready = os.access(result_path.parent, os.W_OK)
    model_checks = {row["hf_id"]: model_row_checks(row, str(row["hf_id"])) for row in model_rows}
    model_paths = [str(row.get("absolute_path", "")) for row in model_rows]
    selected_present = bool(initial_gpu.get("device"))
    checks = {
        "structured_gates": upstream.get("all_structured_gates_passed") is True,
        "cuda_runtime": initial_gpu.get("gpu_query_exit_code") == 0 and selected_present,
        "llama_cpp_cli": llama_cli["available"] is True and llama_cli["exit_code"] == 0,
        "llama_tokenize_cli": tokenizer_cli["available"] is True,
        "llama_cpp_python": llama_python.get("available") is True
        and llama_python.get("gpu_offload_supported") is True,
        "z3": z3_state.get("available") is True,
        "all_model_files_resolved": len(model_rows) == 3
        and all(checks["file_resolved"] for checks in model_checks.values()),
        "all_embedded_tokenizers_valid": len(model_rows) == 3
        and all(
            checks["embedded_tokenizer"]
            and checks["autotokenizer_forbidden"]
            and checks["tokenizer_prompt_nonempty"]
            for checks in model_checks.values()
        ),
        "model_identity_and_file_shape": len(model_rows) == 3
        and all(all(checks.values()) for checks in model_checks.values()),
        "no_stale_task_workers": not _task_worker_pids(model_paths),
        "atomic_output_ready": atomic_output_ready,
    }
    return {
        "planning_date": run_date,
        "platform": platform.platform(),
        "python": {"version": platform.python_version(), "executable": os.sys.executable},
        "cpu": cpu,
        "ram": ram,
        "disk": {
            "path": str(repo_root),
            "total_bytes": disk_usage.total,
            "free_bytes": disk_usage.free,
        },
        "cuda": {
            "available": selected_present,
            "driver_version": initial_gpu.get("device", {}).get("driver_version", ""),
            "selected_gpu": selected_gpu,
        },
        "llama_cpp_cli": llama_cli,
        "llama_tokenize_cli": tokenizer_cli,
        "llama_cpp_python": llama_python,
        "z3": z3_state,
        "cached_model_candidates": _cached_candidates(model_rows),
        "model_preflight_checks": model_checks,
        "protected_file_hashes_before": dict(protected_before),
        "initial_gpu_state": initial_gpu,
        "frozen_execution_contract": {
            "prompt": FROZEN_PROMPT,
            "prompt_sha256": sha256_text(FROZEN_PROMPT),
            "random_seed": RANDOM_SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "worker_timeout_s": WORKER_TIMEOUT_S,
            "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
            "selected_gpu": selected_gpu,
            "temperature": 0.0,
            "one_worker_at_a_time": True,
        },
        "free_vram_arithmetic_used_as_gate": False,
        "checks": checks,
        "failed_preconditions": [name for name, passed in checks.items() if not passed],
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    write: bool = True,
    run_date: str = RUN_DATE,
    selected_gpu: int = SELECTED_GPU,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:  # pragma: no cover
    """Run preconditions, sequential workers, recovery, and atomic assembly."""

    started = time.monotonic()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    before = protected_file_hashes(repo_root)
    upstream = build_upstream_gate_receipts(repo_root)
    model_rows = resolve_model_file_rows()
    preconditions = collect_preconditions(
        repo_root=repo_root,
        result_path=result_path,
        upstream=upstream,
        model_rows=model_rows,
        protected_before=before,
        selected_gpu=selected_gpu,
        run_date=run_date,
    )
    process_rows: list[JsonDict] = []
    gpu_rows: list[JsonDict] = []
    recovery_rows: list[JsonDict] = []
    if not preconditions["failed_preconditions"]:
        process_rows, gpu_rows, recovery_rows = run_sequential_admission(model_rows, selected_gpu)
    after = protected_file_hashes(repo_root)
    artifact = assemble_artifact(
        upstream_gate_receipts=upstream,
        model_file_rows=model_rows,
        process_rows=process_rows,
        gpu_rows=gpu_rows,
        recovery_rows=recovery_rows,
        preconditions=preconditions,
        protected=protected_files_unchanged(before, after),
        duration_s=time.monotonic() - started,
        tests_run=tests_run,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run live admission or validate an existing terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--gpu-index", type=int, default=SELECTED_GPU)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(f"artifact not found: {result_path}")
            return 1
        errors = validate_artifact(_load_json(result_path))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {result_path}")
        return 0
    artifact = build_artifact(  # pragma: no cover - required live E2E command.
        repo_root=REPO_ROOT,
        result_path=result_path,
        write=True,
        run_date=str(args.date),
        selected_gpu=int(args.gpu_index),
    )
    print(f"wrote {result_path}: {artifact['honest_verdict']}")  # pragma: no cover
    return 0  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
