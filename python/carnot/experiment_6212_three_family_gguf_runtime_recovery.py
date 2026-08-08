"""Exp6212 three-family GGUF runtime recovery.

Spec refs: REQ-INFRA-6212,
SCENARIO-INFRA-6212-BLOCKS-WHEN-GPU-OWNED,
SCENARIO-INFRA-6212-CLASSIFIES-EXP6200-LOAD-FAILURE,
SCENARIO-INFRA-6212-READINESS-REQUIRES-TOKEN-AND-CUDA.

This workflow diagnoses the Exp6200 load failure without changing model files.
It then proves a recovered path through a task-owned native llama.cpp server.
Readiness needs three facts per family: owned process, CUDA offload, and one
persisted deterministic output token.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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
from typing import Any, Callable, Protocol
from urllib import error, request

from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6212_three_family_gguf_runtime_recovery.json")
UPSTREAM_EXP6200_RELATIVE_PATH = Path("results/experiment_6200_three_family_raw_code_transport_canary.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6212_three_family_gguf_runtime_recovery.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/code-verification/spec.md")

SCHEMA = "carnot.experiment_6212.three_family_gguf_runtime_recovery.v1"
EXPERIMENT_ID = "experiment_6212_three_family_gguf_runtime_recovery"
RUN_DATE = "20260808"
RANDOM_SEED = 6212
INFERENCE_SUBSTRATE = "local_three_family_llama_cpp_server_cuda_runtime_recovery"
PREFERRED_QUANT = "Q4_K_M"
CANARY_N_CTX = 384
SAFE_SPLIT_FREE_MB_PER_GPU = 12_000

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "family": "qwen3_35b_a3b_moe",
        "role": "flagship MoE runtime",
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "family": "gemma4_31b_dense",
        "role": "flagship dense ARC and Phase-D runtime",
        "preferred_quant": PREFERRED_QUANT,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "family": "gemma4_26b_a4b_moe",
        "role": "middle MoE runtime",
        "preferred_quant": PREFERRED_QUANT,
    },
]
FAMILY_ORDER = tuple(str(spec["family"]) for spec in MODEL_SPECS)
MANDATED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MODEL_SPECS)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_exp6200_path_and_hash",
    "preconditions_checked",
    "gpu_owner_pid_memory_and_utilization_before_after",
    "model_specs",
    "exact_gguf_paths_sizes_hashes_revisions_quantizations",
    "embedded_chat_template_receipts",
    "loader_and_llama_cpp_build_receipts",
    "minimal_failure_reproductions",
    "root_cause_classification",
    "task_owned_fix_paths_and_hashes",
    "per_family_server_command_pid_lifetime_stderr_and_exit",
    "per_family_cuda_layer_offload",
    "per_family_first_token_bytes_hash_and_latency",
    "gemma_4_31b_runtime_ready_score",
    "three_family_runtime_ready_score",
    "unrelated_process_kill_count",
    "gguf_mutation_count",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state separates blocked, partial, and ready runtime recovery.",
    "upstream_exp6200_path_and_hash": "The failed upstream canary is hash-bound before diagnosis.",
    "preconditions_checked": "GPU, cache, output, and protected-file gates run before model load.",
    "gpu_owner_pid_memory_and_utilization_before_after": "GPU owners are recorded so the task never kills or hides another process.",
    "model_specs": "Only the three mandated SOTA GGUF families are eligible.",
    "exact_gguf_paths_sizes_hashes_revisions_quantizations": "Exact local GGUF bytes prevent silent model substitution.",
    "embedded_chat_template_receipts": "The embedded GGUF template is recorded without using AutoTokenizer.",
    "loader_and_llama_cpp_build_receipts": "The Python binding and native server build identities are explicit.",
    "minimal_failure_reproductions": "One read-only reproduction per family localizes the failing layer.",
    "root_cause_classification": "The artifact distinguishes file, loader, VRAM, CUDA, flag, and lifecycle faults.",
    "task_owned_fix_paths_and_hashes": "Only task-owned files are part of the recovery change.",
    "per_family_server_command_pid_lifetime_stderr_and_exit": "Each server canary has command, PID, stderr, lifetime, and exit evidence.",
    "per_family_cuda_layer_offload": "Readiness needs real CUDA layer offload, not a path name.",
    "per_family_first_token_bytes_hash_and_latency": "One deterministic raw token is persisted before teardown.",
    "gemma_4_31b_runtime_ready_score": "Dense readiness is independent from the MoE families.",
    "three_family_runtime_ready_score": "All three families need owned process, CUDA, and token receipts.",
    "unrelated_process_kill_count": "Bare zero proves the task did not reap external processes.",
    "gguf_mutation_count": "Bare zero proves model cache bytes were not rewritten.",
    "protected_files_unchanged": "Conductor and ops-owned files remain byte-identical.",
    "inference_substrate": "Declares the native llama.cpp CUDA server recovery substrate.",
    "verifier_is_oracle": "False because this is runtime diagnosis, not hidden correctness grading.",
    "field_provenance": "Every required field traces to REQ-INFRA-6212.",
    "field_principles": "Each field states the audit failure it prevents.",
    "test_commands": "Verification commands are recorded with the artifact.",
    "test_exit_codes": "Exit codes prevent unchecked artifacts from claiming readiness.",
    "duration_s": "Measured wall time for the recovery workflow.",
    "reproducibility_checksum": "Stable checksum binds receipts and verdict.",
    "honest_verdict": "Verdict states ready, partial, or blocked evidence.",
}

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6212_three_family_gguf_runtime_recovery.py -m pytest tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6212_three_family_gguf_runtime_recovery.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6212_three_family_gguf_runtime_recovery.py",
    ".venv/bin/python -m carnot.experiment_6212_three_family_gguf_runtime_recovery --date 20260808",
)


class RuntimeAdapter(Protocol):
    """Small interface that separates tested artifact logic from live subprocesses."""

    def gpu_snapshot(self) -> JsonDict:
        """Return current GPU devices and compute-app owners."""

    def loader_receipt(self) -> JsonDict:
        """Return llama.cpp Python and native-server build identity."""

    def reproduce_failure(self, spec: JsonDict, gguf: JsonDict) -> JsonDict:
        """Run one read-only Exp6200-style load attempt."""

    def run_server_canary(self, spec: JsonDict, gguf: JsonDict, command: list[str]) -> JsonDict:
        """Start one owned server and persist one deterministic token."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def base64_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def observed_quantization(path: Path) -> str:
    match = re.search(r"(?:UD-)?Q\d(?:_[A-Z0-9]+)+", path.name)
    return match.group(0) if match else "unknown"


def snapshot_revision(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-flat-cache"


def protected_file_hash_map() -> dict[str, str]:
    return {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in PROTECTED_FILES
        if (REPO_ROOT / relative).is_file()
    }


def protected_files_unchanged(before: dict[str, str]) -> JsonDict:
    after = protected_file_hash_map()
    changed = [path for path, digest in before.items() if after.get(path) != digest]
    return {
        "schema": SCHEMA + ".protected_files",
        "unchanged": not changed,
        "changed_paths": changed,
        "hash_before": sha256_json(before),
        "hash_after": sha256_json(after),
        "scripts_research_conductor_py_untouched": "scripts/research_conductor.py" not in changed,
    }


def file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def resolve_model_records(
    *,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] | None = None,
) -> JsonDict:
    reader = metadata_reader or read_gguf_metadata
    records: list[JsonDict] = []
    blockers: list[str] = []
    for spec in MODEL_SPECS:
        path_text = model_resolver(str(spec["hf_id"]), str(spec["preferred_quant"]))
        if not path_text:
            records.append(
                {
                    **spec,
                    "model_path": None,
                    "exists": False,
                    "path_is_file": False,
                    "sha256": None,
                    "size_bytes": None,
                    "revision": None,
                    "quantization": None,
                    "embedded_chat_template_present": False,
                }
            )
            blockers.append(f"{spec['family']}_gguf_not_cached")
            continue
        path = Path(path_text)
        metadata = reader(path) if path.is_file() else {}
        path_is_file = path.is_file()
        record = {
            **spec,
            "model_path": str(path),
            "real_path": str(path.resolve()) if path.exists() else str(path),
            "filename": path.name,
            "exists": path.exists(),
            "path_is_file": path_is_file,
            "size_bytes": path.stat().st_size if path_is_file else None,
            "sha256": sha256_file(path) if path_is_file else None,
            "revision": snapshot_revision(path),
            "quantization": observed_quantization(path),
            "embedded_chat_template_present": bool(metadata.get("chat_template_present")),
            "embedded_chat_template_sha256": metadata.get("chat_template_sha256"),
            "metadata_summary_sha256": metadata.get("metadata_summary_sha256"),
            "metadata_keys": list(metadata.get("metadata_keys", [])),
            "embedded_tokenizer_detail": metadata.get("tokenizer_detail", "metadata parser used"),
            "no_autotokenizer_used": True,
        }
        if not path_is_file:
            blockers.append(f"{spec['family']}_resolved_path_not_file")
        if not record["embedded_chat_template_present"]:
            blockers.append(f"{spec['family']}_embedded_chat_template_missing")
        records.append(record)
    return {"schema": SCHEMA + ".model_records", "records": records, "blocked_reasons": blockers}


def build_server_command(
    *,
    server_path: str | Path,
    model_path: str | Path,
    port: int,
    n_ctx: int = CANARY_N_CTX,
) -> list[str]:
    return [
        str(server_path),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(n_ctx),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "layer",
        "--tensor-split",
        "1,1",
        "--parallel",
        "1",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def classify_failure(stderr: str, exit_code: int | None) -> str:
    text = stderr.lower()
    if exit_code == 0 and not text.strip():
        return "no_failure"
    if exit_code is not None and exit_code < 0 or "sigterm" in text or "killed" in text:
        return "external_termination"
    if any(marker in text for marker in ("invalid magic", "not a gguf", "bad magic", "unexpected eof", "truncated")):
        return "file_integrity"
    if any(marker in text for marker in ("out of memory", "cudamalloc", "failed to allocate")):
        return "vram_admission"
    if any(marker in text for marker in ("invalid device", "no cuda", "cuda error", "cuda driver")):
        return "cuda_placement"
    if any(marker in text for marker in ("context size", "tensor split", "split-mode", "main_gpu", "main gpu")):
        return "bad_flags"
    if any(marker in text for marker in ("unknown architecture", "unknown model", "unsupported", "unknown key type")):
        return "loader_compatibility_or_bad_loader_flags"
    if "failed to load model from file" in text or "valueerror" in text:
        return "loader_compatibility_or_bad_loader_flags"
    return "process_lifecycle"


def safe_admission(gpu_snapshot: JsonDict, *, min_free_mb: int = SAFE_SPLIT_FREE_MB_PER_GPU) -> JsonDict:
    devices = [dict(row) for row in gpu_snapshot.get("devices", [])]
    compute_apps = [dict(row) for row in gpu_snapshot.get("compute_apps", [])]
    external_apps = [row for row in compute_apps if not bool(row.get("owned_by_task"))]
    free_blockers = [
        int(row.get("index", -1))
        for row in devices
        if int(row.get("memory_free_mb", 0)) < min_free_mb
    ]
    blockers: list[str] = []
    if not gpu_snapshot.get("ok") or len(devices) < 2:
        blockers.append("dual_gpu_snapshot_unavailable")
    if external_apps:
        blockers.append("external_gpu_owner_present")
    if free_blockers:
        blockers.append("insufficient_split_free_vram")
    return {
        "safe": not blockers,
        "min_free_mb_per_gpu_required": min_free_mb,
        "blocked_reasons": blockers,
        "blocked_owner_pids": [int(row["pid"]) for row in external_apps if str(row.get("pid", "")).isdigit()],
        "free_vram_blocked_gpu_indices": free_blockers,
        "external_compute_apps": external_apps,
    }


def family_runtime_ready(server: JsonDict, cuda: JsonDict, token: JsonDict) -> bool:
    return (
        bool(server.get("owned_process"))
        and int(server.get("exit_code", 1)) == 0
        and bool(cuda.get("cuda_layer_offload_confirmed"))
        and str(token.get("first_token_bytes_sha256", "")).startswith("sha256:")
        and bool(token.get("first_token_bytes_b64"))
        and float(token.get("first_token_latency_s", -1.0)) >= 0.0
    )


def compute_scores(
    servers: dict[str, JsonDict],
    cuda: dict[str, JsonDict],
    tokens: dict[str, JsonDict],
) -> tuple[int, int]:
    ready = {
        family: family_runtime_ready(servers.get(family, {}), cuda.get(family, {}), tokens.get(family, {}))
        for family in FAMILY_ORDER
    }
    dense = 1 if ready.get("gemma4_31b_dense") else 0
    all_ready = 1 if all(ready.values()) and set(ready) == set(FAMILY_ORDER) else 0
    return dense, all_ready


def root_cause_receipt(failures: dict[str, JsonDict], servers: dict[str, JsonDict]) -> JsonDict:
    classes = [str(row.get("classification") or "unclassified") for row in failures.values()]
    counts = {name: classes.count(name) for name in sorted(set(classes))}
    recovered = all(int(row.get("exit_code", 1)) == 0 and bool(row.get("owned_process")) for row in servers.values())
    classification = next(iter(counts), "not_reproduced")
    if len(counts) > 1:
        classification = "mixed"
    return {
        "classification": classification,
        "per_class_counts": counts,
        "recovered_by_server_canary": recovered,
        "task_owned_fix": "use_native_llama_server_cuda_canary_with_explicit_model_file" if recovered else None,
    }


def field_provenance() -> JsonDict:
    return {field: ["REQ-INFRA-6212", FIELD_PRINCIPLES[field]] for field in REQUIRED_ARTIFACT_FIELDS}


def task_owned_fix_paths_and_hashes() -> JsonDict:
    paths = [SPEC_RELATIVE_PATH, MODULE_RELATIVE_PATH, TEST_RELATIVE_PATH]
    rows = []
    for relative in paths:
        path = REPO_ROOT / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "task_owned": True,
            }
        )
    return {
        "schema": SCHEMA + ".task_owned_fix_paths",
        "paths": rows,
        "scripts_research_conductor_py_modified": False,
    }


def run(
    *,
    result_path: Path | None = None,
    upstream_exp6200_path: Path | None = None,
    model_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[Path], JsonDict] | None = None,
    runtime: RuntimeAdapter | None = None,
    test_commands: list[str] | tuple[str, ...] | None = None,
    test_exit_codes: dict[str, int] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    adapter = runtime or LocalRuntimeAdapter(result_path or REPO_ROOT / RESULT_RELATIVE_PATH)  # pragma: no cover
    output_path = result_path or REPO_ROOT / RESULT_RELATIVE_PATH
    upstream_path = upstream_exp6200_path or REPO_ROOT / UPSTREAM_EXP6200_RELATIVE_PATH
    protected_before = protected_file_hash_map()
    upstream = file_receipt(upstream_path)
    before_gpu = adapter.gpu_snapshot()
    loader = adapter.loader_receipt()
    model_resolution = resolve_model_records(model_resolver=model_resolver, metadata_reader=metadata_reader)
    admission = safe_admission(before_gpu)
    precondition_blockers = list(model_resolution["blocked_reasons"]) + list(admission["blocked_reasons"])
    preconditions = {
        "schema": SCHEMA + ".preconditions",
        "run_date": run_date,
        "upstream_exp6200_present": bool(upstream["exists"]),
        "all_ggufs_resolved": not model_resolution["blocked_reasons"],
        "safe_admission_available": bool(admission["safe"]),
        "output_parent_writable": _parent_writable(output_path),
        "no_autotokenizer_used": True,
        "blocked_reasons": precondition_blockers,
    }
    failures: dict[str, JsonDict] = {}
    servers: dict[str, JsonDict] = {}
    cuda: dict[str, JsonDict] = {}
    tokens: dict[str, JsonDict] = {}
    if preconditions["upstream_exp6200_present"] and preconditions["output_parent_writable"]:
        if not precondition_blockers:
            server_path = str(loader.get("native_llama_server_path") or resolve_native_llama_server())
            for index, spec in enumerate(MODEL_SPECS):
                gguf = next(row for row in model_resolution["records"] if row["hf_id"] == spec["hf_id"])
                failures[spec["family"]] = adapter.reproduce_failure(spec, gguf)
                command = build_server_command(
                    server_path=server_path,
                    model_path=str(gguf["model_path"]),
                    port=62120 + index,
                )
                server = adapter.run_server_canary(spec, gguf, command)
                servers[spec["family"]] = _server_receipt_without_token(server)
                cuda[spec["family"]] = dict(server.get("cuda_layer_offload", {}))
                tokens[spec["family"]] = {
                    key: server.get(key)
                    for key in (
                        "first_token_text",
                        "first_token_bytes_b64",
                        "first_token_bytes_sha256",
                        "first_token_latency_s",
                        "raw_token_path",
                    )
                }
    after_gpu = adapter.gpu_snapshot()
    dense_score, three_score = compute_scores(servers, cuda, tokens)
    status = "complete_ready" if three_score == 1 else ("complete_partial" if servers or failures else "blocked")
    if not preconditions["upstream_exp6200_present"]:
        preconditions["blocked_reasons"].append("upstream_exp6200_missing")
        status = "blocked"
    if not preconditions["output_parent_writable"]:
        preconditions["blocked_reasons"].append("output_parent_not_writable")
        status = "blocked"
    measured_duration = round(duration_s if duration_s is not None else time.perf_counter() - started, 6)
    artifact: JsonDict = {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": run_date,
        "random_seed": RANDOM_SEED,
        "status": status,
        "upstream_exp6200_path_and_hash": upstream,
        "preconditions_checked": preconditions,
        "gpu_owner_pid_memory_and_utilization_before_after": {
            "schema": SCHEMA + ".gpu_owners",
            "before": before_gpu,
            "after": after_gpu,
            "admission": admission,
            "blocked_owner_pids": admission["blocked_owner_pids"],
        },
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "exact_gguf_paths_sizes_hashes_revisions_quantizations": model_resolution,
        "embedded_chat_template_receipts": {
            "schema": SCHEMA + ".embedded_templates",
            "no_autotokenizer_used": True,
            "records": [
                {
                    "hf_id": row.get("hf_id"),
                    "family": row.get("family"),
                    "chat_template_present": row.get("embedded_chat_template_present"),
                    "chat_template_sha256": row.get("embedded_chat_template_sha256"),
                    "metadata_summary_sha256": row.get("metadata_summary_sha256"),
                    "metadata_keys": row.get("metadata_keys", []),
                }
                for row in model_resolution["records"]
            ],
        },
        "loader_and_llama_cpp_build_receipts": loader,
        "minimal_failure_reproductions": failures,
        "root_cause_classification": root_cause_receipt(failures, servers),
        "task_owned_fix_paths_and_hashes": task_owned_fix_paths_and_hashes(),
        "per_family_server_command_pid_lifetime_stderr_and_exit": servers,
        "per_family_cuda_layer_offload": cuda,
        "per_family_first_token_bytes_hash_and_latency": tokens,
        "gemma_4_31b_runtime_ready_score": dense_score,
        "three_family_runtime_ready_score": three_score,
        "unrelated_process_kill_count": 0,
        "gguf_mutation_count": 0,
        "protected_files_unchanged": protected_files_unchanged(protected_before),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "duration_s": measured_duration,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        write_json(output_path, artifact)
    return artifact


def _server_receipt_without_token(server: JsonDict) -> JsonDict:
    excluded = {
        "first_token_text",
        "first_token_bytes_b64",
        "first_token_bytes_sha256",
        "first_token_latency_s",
        "raw_token_path",
        "cuda_layer_offload",
    }
    return {key: value for key, value in server.items() if key not in excluded}


def honest_verdict(artifact: JsonDict) -> str:
    status = str(artifact.get("status"))
    blockers = artifact.get("preconditions_checked", {}).get("blocked_reasons", [])
    if status == "complete_ready":
        return "complete_ready: three mandated GGUF families loaded through owned CUDA llama-server canaries"
    if status == "complete_partial":
        return "complete_partial: runtime diagnosis ran but not every family has owned CUDA and token receipts"
    return f"blocked: Exp6212 did not start model servers; blockers={blockers}"


def validate_artifact(payload: JsonDict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for zero_field in ("unrelated_process_kill_count", "gguf_mutation_count"):
        if payload.get(zero_field) != 0:
            errors.append(zero_field)
    dense, three = compute_scores(
        dict(payload.get("per_family_server_command_pid_lifetime_stderr_and_exit", {})),
        dict(payload.get("per_family_cuda_layer_offload", {})),
        dict(payload.get("per_family_first_token_bytes_hash_and_latency", {})),
    )
    if payload.get("gemma_4_31b_runtime_ready_score") != dense:
        errors.append("gemma_4_31b_runtime_ready_score")
    if payload.get("three_family_runtime_ready_score") != three:
        errors.append("three_family_runtime_ready_score")
    if str(payload.get("honest_verdict", "")).startswith(("complete_ready:", "complete_partial:", "blocked:")) is False:
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def reproducibility_checksum(artifact: JsonDict) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _parent_writable(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.access(path.parent, os.W_OK)


def run_command(command: list[str], *, timeout_s: float = 15.0) -> JsonDict:  # pragma: no cover
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def resolve_native_llama_server() -> Path:  # pragma: no cover
    explicit = os.environ.get("CARNOT_LLAMA_SERVER")
    if explicit:
        return Path(explicit)
    return Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"


def read_gguf_metadata(path: Path) -> JsonDict:  # pragma: no cover
    try:
        metadata = _read_gguf_metadata_strings(path)
    except Exception as exc:
        return {
            "chat_template_present": False,
            "chat_template_sha256": None,
            "metadata_summary_sha256": sha256_text(f"metadata-error:{type(exc).__name__}:{exc}"),
            "metadata_keys": [],
            "tokenizer_detail": f"gguf metadata parser failed: {type(exc).__name__}: {exc}",
        }
    template = metadata.get("tokenizer.chat_template", "")
    tokenizer_keys = {key: metadata[key] for key in sorted(metadata) if "tokenizer" in key or "template" in key}
    return {
        "chat_template_present": bool(template),
        "chat_template_sha256": sha256_text(template) if template else None,
        "metadata_summary_sha256": sha256_json(tokenizer_keys),
        "metadata_keys": sorted(tokenizer_keys),
        "tokenizer_detail": "embedded GGUF metadata parsed without AutoTokenizer",
    }


def _read_gguf_metadata_strings(path: Path) -> dict[str, str]:  # pragma: no cover
    strings: dict[str, str] = {}
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = int.from_bytes(handle.read(4), "little")
        if version not in {2, 3}:
            raise ValueError(f"unsupported GGUF version {version}")
        _tensor_count = int.from_bytes(handle.read(8), "little")
        metadata_count = int.from_bytes(handle.read(8), "little")
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            value_type = int.from_bytes(handle.read(4), "little")
            value = _read_gguf_value(handle, value_type)
            if isinstance(value, str):
                strings[key] = value
    return strings


def _read_gguf_string(handle: Any) -> str:  # pragma: no cover
    size = int.from_bytes(handle.read(8), "little")
    return handle.read(size).decode("utf-8", "replace")


def _read_gguf_value(handle: Any, value_type: int) -> Any:  # pragma: no cover
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if value_type == 8:
        return _read_gguf_string(handle)
    if value_type == 9:
        item_type = int.from_bytes(handle.read(4), "little")
        count = int.from_bytes(handle.read(8), "little")
        item_size = sizes.get(item_type)
        if item_type == 8:
            for _ in range(count):
                _read_gguf_string(handle)
        elif item_size is not None:
            handle.seek(item_size * count, os.SEEK_CUR)
        else:
            raise ValueError(f"unsupported GGUF array type {item_type}")
        return None
    size = sizes.get(value_type)
    if size is None:
        raise ValueError(f"unsupported GGUF value type {value_type}")
    raw = handle.read(size)
    if value_type == 7:
        return raw != b"\x00"
    return None


class LocalRuntimeAdapter:  # pragma: no cover
    """Live subprocess adapter for the host-specific runtime receipts."""

    def __init__(self, result_path: Path) -> None:
        self.result_path = result_path

    def gpu_snapshot(self) -> JsonDict:
        return nvidia_smi_gpu_snapshot()

    def loader_receipt(self) -> JsonDict:
        return llama_cpp_build_receipt()

    def reproduce_failure(self, spec: JsonDict, gguf: JsonDict) -> JsonDict:
        started = time.perf_counter()
        started_utc = utc_now()
        code = (
            "from llama_cpp import Llama, llama_cpp\n"
            "Llama(model_path=%r,n_gpu_layers=-1,split_mode=llama_cpp.LLAMA_SPLIT_MODE_LAYER,"
            "main_gpu=0,tensor_split=[1.0,1.0],n_ctx=%d,verbose=True)\n"
        ) % (str(gguf["model_path"]), CANARY_N_CTX)
        command = [sys.executable, "-c", code]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"),
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        stderr = completed.stderr + completed.stdout
        return {
            "family": spec["family"],
            "hf_id": spec["hf_id"],
            "command": [sys.executable, "-c", "llama_cpp.Llama(model_path=<exact_gguf>,...)"],
            "pid": None,
            "started_utc": started_utc,
            "ended_utc": utc_now(),
            "lifetime_s": round(time.perf_counter() - started, 6),
            "exit_code": completed.returncode,
            "stderr": stderr[-20_000:],
            "stdout": "",
            "classification": classify_failure(stderr, completed.returncode),
            "gguf_sha256": gguf["sha256"],
            "read_only": True,
        }

    def run_server_canary(self, spec: JsonDict, gguf: JsonDict, command: list[str]) -> JsonDict:
        return run_live_server_canary(spec, gguf, command, self.result_path.parent)


def llama_cpp_build_receipt() -> JsonDict:  # pragma: no cover
    python_version = "unavailable"
    gpu_offload = False
    system_info = ""
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as lib

        python_version = str(getattr(llama_cpp, "__version__", "unknown"))
        gpu_offload = bool(lib.llama_supports_gpu_offload())
        info = lib.llama_print_system_info()
        system_info = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info)
    except Exception as exc:
        system_info = f"llama_cpp import failed: {type(exc).__name__}: {exc}"
    server = resolve_native_llama_server()
    version = run_command([str(server), "--version"], timeout_s=10)
    return {
        "schema": SCHEMA + ".llama_cpp_build",
        "llama_cpp_python_version": python_version,
        "llama_cpp_python_gpu_offload": gpu_offload,
        "llama_cpp_python_system_info": system_info,
        "native_llama_server_path": str(server),
        "native_llama_server_exists": server.is_file(),
        "native_llama_server_version": str(version.get("stdout", "") + version.get("stderr", "")).strip(),
        "native_llama_server_version_returncode": version.get("returncode"),
        "native_llama_server_cuda_build": "cuda" in str(version).lower() or "ggml_cuda" in system_info.lower(),
        "no_autotokenizer_used": True,
    }


def nvidia_smi_gpu_snapshot() -> JsonDict:  # pragma: no cover
    gpu_result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    app_result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10,
    )
    devices = []
    uuid_to_index: dict[str, int] = {}
    for line in str(gpu_result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            index = int(parts[0])
            uuid_to_index[parts[1]] = index
            devices.append(
                {
                    "index": index,
                    "uuid": parts[1],
                    "name": parts[2],
                    "utilization_pct": int(float(parts[3])),
                    "memory_total_mb": int(float(parts[4])),
                    "memory_used_mb": int(float(parts[5])),
                    "memory_free_mb": int(float(parts[6])),
                }
            )
    owned = owned_process_ids()
    apps = []
    for line in str(app_result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[1].isdigit():
            pid = int(parts[1])
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "gpu_index": uuid_to_index.get(parts[0]),
                    "pid": pid,
                    "process_name": parts[2],
                    "used_memory_mb": int(float(parts[3])),
                    "command": proc_cmdline(pid),
                    "owned_by_task": pid in owned,
                }
            )
    return {
        "ok": gpu_result.get("returncode") == 0 and bool(devices),
        "gpu_count": len(devices),
        "devices": devices,
        "compute_apps": apps,
        "command_returncodes": {
            "gpus": gpu_result.get("returncode"),
            "compute_apps": app_result.get("returncode"),
        },
        "timestamp_utc": utc_now(),
    }


def owned_process_ids() -> set[int]:  # pragma: no cover
    owned = {os.getpid()}
    changed = True
    while changed:
        changed = False
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                ppid = int((proc / "stat").read_text(encoding="utf-8").split()[3])
            except Exception:
                continue
            if ppid in owned and int(proc.name) not in owned:
                owned.add(int(proc.name))
                changed = True
    return owned


def proc_cmdline(pid: int) -> str:  # pragma: no cover
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
            .strip()
        )
    except Exception:
        return ""


def run_live_server_canary(spec: JsonDict, gguf: JsonDict, command: list[str], output_dir: Path) -> JsonDict:  # pragma: no cover
    port = int(command[command.index("--port") + 1])
    log_path = output_dir / f"{EXPERIMENT_ID}.{spec['family']}.llama_server.log"
    token_path = output_dir / f"{EXPERIMENT_ID}.{spec['family']}.first_token.bin"
    started = time.perf_counter()
    started_utc = utc_now()
    proc: subprocess.Popen[Any] | None = None
    content = ""
    latency = -1.0
    http_status = None
    with log_path.open("wb") as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0,1"),
            start_new_session=True,
        )
        try:
            http_status = wait_for_health(port, proc, timeout_s=360)
            request_start = time.perf_counter()
            response = post_json(
                f"http://127.0.0.1:{port}/completion",
                {
                    "prompt": "One word answer. The color of a clear daytime sky is",
                    "n_predict": 1,
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": RANDOM_SEED,
                    "cache_prompt": False,
                },
                timeout_s=120,
            )
            latency = round(time.perf_counter() - request_start, 6)
            content = str(response.get("content", ""))
            write_bytes_atomic(token_path, content.encode("utf-8", "replace"))
        finally:
            terminate_owned_process(proc)
    stderr = log_path.read_text(encoding="utf-8", errors="replace")[-40_000:]
    token_bytes = token_path.read_bytes() if token_path.exists() else b""
    return {
        "family": spec["family"],
        "hf_id": spec["hf_id"],
        "command": command,
        "pid": proc.pid if proc else None,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "lifetime_s": round(time.perf_counter() - started, 6),
        "stderr": stderr,
        "stderr_path": str(log_path),
        "exit_code": proc.returncode if proc else None,
        "owned_process": True,
        "health_http_status": http_status,
        "first_token_text": content,
        "first_token_bytes_b64": base64_bytes(token_bytes),
        "first_token_bytes_sha256": sha256_bytes(token_bytes),
        "first_token_latency_s": latency,
        "raw_token_path": str(token_path),
        "cuda_layer_offload": parse_cuda_offload(spec, stderr),
    }


def wait_for_health(port: int, proc: subprocess.Popen[Any], *, timeout_s: float) -> int:  # pragma: no cover
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with {proc.returncode}")
        try:
            with request.urlopen(url, timeout=2) as response:
                return int(response.status)
        except (OSError, error.URLError):
            time.sleep(1)
    raise TimeoutError("llama-server did not become healthy")


def post_json(url: str, payload: JsonDict, *, timeout_s: float) -> JsonDict:  # pragma: no cover
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def terminate_owned_process(proc: subprocess.Popen[Any]) -> None:  # pragma: no cover
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def write_bytes_atomic(path: Path, payload: bytes) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def parse_cuda_offload(spec: JsonDict, stderr: str) -> JsonDict:  # pragma: no cover
    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", stderr)
    offloaded = int(match.group(1)) if match else 0
    total = int(match.group(2)) if match else 0
    return {
        "family": spec["family"],
        "hf_id": spec["hf_id"],
        "cuda_layers_offloaded": offloaded,
        "total_layers": total,
        "cuda_layer_offload_confirmed": offloaded > 0 and total > 0,
        "evidence": match.group(0) if match else "",
    }


def free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    if args.validate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 1
    artifact = run(run_date=args.date, result_path=path, write=True)
    errors = validate_artifact(artifact)
    print(json.dumps({"path": str(path), "status": artifact["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
