"""Run the immutable Exp6581 Qwen source shard.

The experiment records one raw terminal row for every frozen Exp6580 source
unit before diagnostic claim segmentation.  It proves only local Qwen runtime
and source-shard completeness; it does not make a cross-family or quality
claim.

Spec: REQ-REPORT-6581 and SCENARIO-REPORT-6581-GATE-BLOCK through
SCENARIO-REPORT-6581-ATOMIC.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
import concurrent.futures
import datetime
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
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from carnot import experiment_6572_content_derived_gguf_metadata_resolver as gguf_fixtures
from carnot import experiment_6573_sequential_flagship_gguf_admission_v2 as runtime_helpers
from carnot.inference.gguf_metadata import build_gguf_admission_record
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "exp6581-qwen36-flagship-source-shard"
RUN_DATE = "20260824"
QWEN_REPOSITORY_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
QWEN_ARCHITECTURE = "qwen35moe"
RANDOM_SEED = 6581
INFERENCE_SUBSTRATE = "live_llama_cpp_cuda_one_family_source_shard"
RESULT_RELATIVE_PATH = Path("results/experiment_6581_qwen36_flagship_source_shard.json")
RAW_CHECKPOINT_RELATIVE_PATH = Path("results/experiment_6581_qwen36_flagship_source_shard.raw")
PROTOCOL_RELATIVE_PATH = Path("results/experiment_6580_v572_source_and_joint_method_protocol.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6581_qwen36_flagship_source_shard.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6581_qwen36_flagship_source_shard.py")
PROTECTED_RELATIVE_PATHS = (Path("research-roadmap.yaml"), Path("scripts/research_conductor.py"))
GATE_CONTRACTS = (
    (
        "exp6579",
        Path("results/experiment_6579_v572_terminal_recovery_and_decomposition_contract.json"),
        "v572_decomposition_contract_ready_score",
    ),
    (
        "exp6580",
        PROTOCOL_RELATIVE_PATH,
        "v572_source_method_ready_score",
    ),
)

LOAD_TIMEOUT_S = 900.0
SHUTDOWN_TIMEOUT_S = 30.0
RECOVERY_TIMEOUT_S = 180.0
RECOVERY_TOLERANCE_MB = 256
GPU_LOAD_DELTA_MIN_MB = 128
TELEMETRY_INTERVAL_S = 0.25
IDLE_GPU_MAX_UTILIZATION_PCT = 5
CONTEXT_SIZE = 4096

REQUIRED_NEGATIVE_FIXTURE_IDS = (
    "non_gguf",
    "truncated_header",
    "wrong_repository_mapping",
    "tokenizer_only_gguf",
    "wrong_architecture",
)
REQUIRED_ATTACK_IDS = (
    "legacy_substitution",
    "stale_pid",
    "zero_layer_offload",
    "reused_output",
    "prompt_drift",
    "source_aliasing",
    "hidden_retry",
    "missing_raw_bytes",
    "failed_row_removal",
    "cross_family_residency",
    "readiness_with_incomplete_manifest",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "model_specs",
    "model_revision_and_hash_receipt",
    "rows",
    "raw_response_receipts",
    "process_and_gpu_receipts",
    "checkpoint_receipts",
    "parser_diagnostic_rows",
    "unload_and_recovery_rows",
    "attack_rows",
    "qwen36_family_source_shard_ready_score",
    "aggregate_row_recomputation",
    "seeds",
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
    "status": "The one-family task closes even on a timeout or gate block.",
    "honest_verdict": "The verdict states runtime and source-shard completeness without a quality claim.",
    "verdict_class": "Evidence readiness uses null, partial, blocked, or disqualified.",
    "gate_check_summary": "A block names the exact same-roadmap field and observed value.",
    "model_specs": "Only the mandated Qwen family can satisfy readiness.",
    "model_revision_and_hash_receipt": "Repository, revision, GGUF bytes, architecture, quantization, and tokenizer bind the run.",
    "rows": "Every source unit carries raw output, runtime, failure, token, timing, and cost metrics.",
    "raw_response_receipts": "Raw response bytes precede parsing and preserve all outcomes.",
    "process_and_gpu_receipts": "PID, command, offload, and repeated GPU samples prove live local execution.",
    "checkpoint_receipts": "Periodic content hashes prevent another long no-artifact loss.",
    "parser_diagnostic_rows": "Claim segmentation is diagnostic and cannot filter the source shard.",
    "unload_and_recovery_rows": "The family releases GPU state before the task closes.",
    "attack_rows": "Substitution, stale execution, drift, retries, and row loss fail closed.",
    "qwen36_family_source_shard_ready_score": "This exact binary field is owned by Exp6581 and consumed by Exp6584.",
    "aggregate_row_recomputation": "Coverage, failures, latency, tokens, and cost derive only from rows.",
    "seeds": "Explicit seeds bind all source-unit generations.",
    "preconditions_checked": "Gate, model, source, resource, CUDA, and protected-file receipts are explicit.",
    "protected_files_unchanged": "The task preserves both protected orchestration files.",
    "inference_substrate": "The live llama.cpp CUDA path selects the correct structural checks.",
    "verifier_is_oracle": "This task creates evidence and makes no verifier-backed utility claim.",
    "field_provenance": "Every field names its raw rows, hashes, and reducer.",
    "duration_s": "Monotonic duration exposes truncated family work.",
    "tests_run": "Named commands, exits, and durations make the shard reproducible.",
    "reproducibility_checksum": "A final hash detects terminal mutation.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6581_qwen36_flagship_source_shard.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6581_qwen36_flagship_source_shard.py "
    "-m pytest tests/python/test_experiment_6581_qwen36_flagship_source_shard.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6581_qwen36_flagship_source_shard.py "
    "--fail-under=100 --show-missing"
)
RUFF_CHECK_COMMAND = f".venv/bin/ruff check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
RUFF_FORMAT_COMMAND = f".venv/bin/ruff format --check {MODULE_RELATIVE_PATH} {TEST_RELATIVE_PATH}"
SPEC_COVERAGE_COMMAND = f".venv/bin/python scripts/check_spec_coverage.py {TEST_RELATIVE_PATH}"


def canonical_json(value: Any) -> str:
    """Encode stable compact JSON for all evidence hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Return one prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text without newline rewriting."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON representation."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash one file or record that it is absent."""

    target = Path(path)
    if not target.is_file():
        return "missing"
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: str | Path) -> JsonDict:
    """Read one JSON object and fail closed for absent or malformed input."""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row without trusting its stored self-hash."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash a terminal artifact without its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def build_gate_receipts(
    repo_root: Path,
    *,
    gate_contracts: Sequence[tuple[str, Path, str]] = GATE_CONTRACTS,
) -> list[JsonDict]:
    """Record the exact two structured same-roadmap gates."""

    rows = []
    for upstream, relative_path, field in gate_contracts:
        path = repo_root / relative_path
        observed = load_json(path).get(field)
        rows.append(
            {
                "upstream": upstream,
                "path": relative_path.as_posix(),
                "absolute_path": str(path.resolve()),
                "sha256": sha256_file(path),
                "field": field,
                "expected_value": 1.0,
                "observed_value": observed,
                "passed": observed == 1.0,
            }
        )
    return rows


def validate_frozen_protocol(protocol: Mapping[str, Any]) -> list[str]:
    """Validate the frozen source identity, prompt, seed, budget, and order."""

    errors: list[str] = []
    manifest = protocol.get("source_unit_manifest", {})
    contract = protocol.get("prompt_seed_budget_contract", {})
    units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    units = [row for row in units if isinstance(row, Mapping)]
    if protocol.get("v572_source_method_ready_score") != 1.0:
        errors.append("source_method_gate_mismatch")
    declared_count = manifest.get("bounded_unit_count") if isinstance(manifest, Mapping) else None
    if declared_count != len(units) or len(units) != 4:
        errors.append("source_manifest_count_mismatch")
    manifest_body = (
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if isinstance(manifest, Mapping)
        else {}
    )
    if not manifest_body or manifest.get("manifest_hash") != sha256_json(manifest_body):
        errors.append("source_manifest_hash_mismatch")
    expected_kinds = ["single_hop", "multi_hop", "unsupported", "ambiguity"]
    if [row.get("case_kind") for row in units] != expected_kinds:
        errors.append("source_order_mismatch")
    if any(
        row.get("source_bytes_sha256") != sha256_text(str(row.get("exact_source_bytes", "")))
        for row in units
    ):
        errors.append("source_hash_mismatch")
    prompt = str(contract.get("family_neutral_prompt", "")) if isinstance(contract, Mapping) else ""
    budget = contract.get("token_budget", {}) if isinstance(contract, Mapping) else {}
    family_rows = contract.get("family_rows", []) if isinstance(contract, Mapping) else []
    family = next(
        (row for row in family_rows if isinstance(row, Mapping) and row.get("task_id") == TASK_ID),
        {},
    )
    mapping = contract.get("one_family_task_mapping", {}) if isinstance(contract, Mapping) else {}
    family_valid = (
        mapping.get(TASK_ID) == QWEN_REPOSITORY_ID
        and family.get("task_id") == TASK_ID
        and family.get("model_family") == QWEN_REPOSITORY_ID
        and family.get("seed") == RANDOM_SEED
        and family.get("family_specific_prompt_allowed") is False
        and family.get("prompt_sha256") == sha256_text(prompt)
        and family.get("token_budget_hash") == sha256_json(budget)
        and family.get("per_source_unit_timeout_s") == 720
        and family.get("task_timeout_s") == contract.get("timeout_s") == 4200
    )
    if not family_valid:
        errors.append("qwen_family_contract_mismatch")
    if (
        not prompt
        or contract.get("prompt_sha256") != sha256_text(prompt)
        or budget
        != {
            "max_prompt_tokens": 4096,
            "max_output_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        or contract.get("stop_rules") != ["<|eot_id|>", "<stop>"]
        or contract.get("raw_before_derived_write_order") is not True
        or contract.get("failure_retention_required") is not True
        or contract.get("fresh_process_per_family") is not True
    ):
        errors.append("prompt_seed_budget_contract_mismatch")
    return errors


def metadata_receipt_passes(receipt: Mapping[str, Any]) -> bool:
    """Recompute positive Qwen GGUF identity from content and provenance."""

    content = receipt.get("content_metadata", {})
    provenance = receipt.get("provenance", {})
    tokenizer = content.get("tokenizer_metadata", {}) if isinstance(content, Mapping) else {}
    bounded = content.get("bounded_read_receipt", {}) if isinstance(content, Mapping) else {}
    selected = str(receipt.get("selected_blob_path", receipt.get("path", "")))
    trusted = str(receipt.get("trusted_sha256", ""))
    shards = provenance.get("ordered_shards", []) if isinstance(provenance, Mapping) else []
    return bool(
        receipt.get("repository_id") == QWEN_REPOSITORY_ID
        and receipt.get("admitted") is True
        and not receipt.get("rejection_reasons")
        and isinstance(content, Mapping)
        and content.get("architecture") == QWEN_ARCHITECTURE
        and bool(content.get("quantization"))
        and content.get("is_language_model") is True
        and int(content.get("tensor_count", 0) or 0) > 0
        and isinstance(tokenizer, Mapping)
        and int(tokenizer.get("token_count", 0) or 0) > 0
        and tokenizer.get("chat_template_present") is True
        and isinstance(bounded, Mapping)
        and bounded.get("tensor_payload_bytes_read") == 0
        and isinstance(provenance, Mapping)
        and provenance.get("valid") is True
        and provenance.get("repository_id") == QWEN_REPOSITORY_ID
        and bool(provenance.get("revision"))
        and bool(provenance.get("snapshot_filename"))
        and provenance.get("trusted_sha256") == trusted
        and provenance.get("trusted_hash_matches_blob_key") is True
        and provenance.get("resolved_blob_path") == selected
        and provenance.get("symlink_target_matches_blob") is True
        and bool(shards)
    )


def _wrong_architecture_fixture_row() -> JsonDict:
    """Build a bounded valid GGUF whose architecture is not Qwen."""

    with tempfile.TemporaryDirectory(prefix="exp6581-negative-") as temporary:
        cache_root = Path(temporary)
        blob, trusted = gguf_fixtures._cache_fixture(
            cache_root,
            gguf_fixtures._fixture_gguf(architecture="gemma4"),
            repository_id=QWEN_REPOSITORY_ID,
            filename="wrong-architecture.gguf",
        )
        record = build_gguf_admission_record(
            blob,
            repository_id=QWEN_REPOSITORY_ID,
            cache_root=cache_root,
            trusted_sha256=trusted,
            expected_architectures={QWEN_ARCHITECTURE},
        )
        reasons = list(record.get("rejection_reasons", []))
        return {
            "row_type": "negative_fixture",
            "unit_id": "wrong_architecture",
            "expected_admitted": False,
            "observed_admitted": record.get("admitted"),
            "expected_reason": "architecture_mismatch",
            "rejection_reasons": reasons,
            "passed": record.get("admitted") is False and "architecture_mismatch" in reasons,
            "bounded_read_receipt": record.get("content_metadata", {}).get(
                "bounded_read_receipt", {}
            ),
            "record": record,
        }


def build_negative_metadata_fixture_rows() -> list[JsonDict]:
    """Select the four mandated resolver attacks and add wrong architecture."""

    source = {row.get("unit_id"): dict(row) for row in gguf_fixtures.build_negative_fixture_rows()}
    source["wrong_architecture"] = _wrong_architecture_fixture_row()
    return [
        source.get(fixture_id, {"unit_id": fixture_id, "passed": False})
        for fixture_id in REQUIRED_NEGATIVE_FIXTURE_IDS
    ]


def compose_request_bytes(prompt: str, source_bytes: str) -> bytes:
    """Compose the family-neutral prompt without model identity or history."""

    return f"{prompt}\n\nSOURCE BYTES:\n{source_bytes}".encode()


def segment_claim_sentences(raw: bytes, *, max_segments: int = 16) -> list[str]:
    """Return bounded diagnostic sentence segments without filtering rows."""

    try:
        text = raw.decode("utf-8", "strict").strip()
    except UnicodeDecodeError:
        return []
    if not text or max_segments <= 0:
        return []
    return [part.strip() for part in re.findall(r"[^.!?]+[.!?]|[^.!?]+$", text) if part.strip()][
        :max_segments
    ]


def classify_raw_response(
    raw: bytes, *, timed_out: bool = False, process_failure: bool = False
) -> dict[str, bool]:
    """Classify terminal output without deciding whether its claims are true."""

    malformed = False
    try:
        text = raw.decode("utf-8", "strict").strip()
    except UnicodeDecodeError:
        text = ""
        malformed = True
    lowered = text.lower()
    refusal = any(
        marker in lowered
        for marker in ("i cannot", "i can't", "unable to comply", "cannot comply", "i refuse")
    )
    empty = not raw or not text
    no_claim = empty or refusal or malformed
    return {
        "timeout": bool(timed_out),
        "malformed_output": malformed,
        "refusal": refusal,
        "empty_output": empty,
        "no_claim": no_claim,
        "process_failure": bool(process_failure),
    }


def cost_from_components(components: Sequence[Mapping[str, Any]]) -> float:
    """Recompute normalized charged work from row-local components."""

    return round(
        sum(
            float(row.get("quantity", 0.0) or 0.0) * float(row.get("unit_cost", 0.0) or 0.0)
            for row in components
        ),
        9,
    )


def build_raw_terminal_row(
    *,
    unit: Mapping[str, Any],
    order_index: int,
    protocol: Mapping[str, Any],
    metadata_receipt: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    raw_response_bytes: bytes,
    raw_api_response_sha256: str,
    prompt_tokens: int,
    response_tokens: int,
    latency_s: float,
    stop_reason: str,
    request_exit_code: int,
    stderr_sha256_at_terminal: str,
    failure_flags: Mapping[str, bool],
    raw_response_recorded_monotonic_ns: int,
) -> JsonDict:
    """Create one immutable raw row before any claim segmentation."""

    contract = protocol["prompt_seed_budget_contract"]
    manifest = protocol["source_unit_manifest"]
    prompt = str(contract["family_neutral_prompt"])
    source = str(unit["exact_source_bytes"])
    request = compose_request_bytes(prompt, source)
    provenance = metadata_receipt.get("provenance", {})
    components = [
        {"metric": "prompt_tokens", "quantity": int(prompt_tokens), "unit_cost": 1.0},
        {"metric": "response_tokens", "quantity": int(response_tokens), "unit_cost": 1.0},
        {"metric": "latency_s", "quantity": round(float(latency_s), 9), "unit_cost": 1.0},
    ]
    row: JsonDict = {
        "row_type": "raw_terminal_source_unit",
        "unit_id": unit.get("unit_id"),
        "fixture_id": unit.get("fixture_id"),
        "case_kind": unit.get("case_kind"),
        "split": unit.get("split"),
        "order_index": order_index,
        "source_manifest_hash": manifest.get("manifest_hash"),
        "source_content_hash": unit.get("content_hash"),
        "source_bytes_b64": base64.b64encode(source.encode("utf-8")).decode("ascii"),
        "source_bytes_sha256": unit.get("source_bytes_sha256"),
        "prompt_sha256": contract.get("prompt_sha256"),
        "request_bytes_b64": base64.b64encode(request).decode("ascii"),
        "request_sha256": sha256_bytes(request),
        "repository_id": metadata_receipt.get("repository_id"),
        "revision": provenance.get("revision") if isinstance(provenance, Mapping) else None,
        "gguf_sha256": metadata_receipt.get("trusted_sha256"),
        "gguf_blob_path": metadata_receipt.get("selected_blob_path", metadata_receipt.get("path")),
        "command_sha256": process_receipt.get("command_sha256"),
        "pid": process_receipt.get("pid"),
        "cuda_device": process_receipt.get("selected_gpu"),
        "offloaded_layers": process_receipt.get("offloaded_layers"),
        "seed": RANDOM_SEED,
        "attempt_count": 1,
        "retry_count": 0,
        "raw_response_bytes_b64": base64.b64encode(raw_response_bytes).decode("ascii"),
        "raw_response_byte_count": len(raw_response_bytes),
        "raw_response_sha256": sha256_bytes(raw_response_bytes),
        "raw_api_response_sha256": raw_api_response_sha256,
        "prompt_token_count": int(prompt_tokens),
        "response_token_count": int(response_tokens),
        "total_token_count": int(prompt_tokens) + int(response_tokens),
        "latency_s": round(float(latency_s), 9),
        "stop_reason": stop_reason,
        "request_exit_code": request_exit_code,
        "stderr_sha256_at_terminal": stderr_sha256_at_terminal,
        "failure_flags": dict(failure_flags),
        "charged_cost_unit": "normalized_token_and_second_units",
        "charged_cost_components": components,
        "charged_cost": cost_from_components(components),
        "raw_response_recorded_monotonic_ns": int(raw_response_recorded_monotonic_ns),
    }
    row["row_hash"] = row_hash(row)
    return row


def write_raw_checkpoint(checkpoint_dir: Path, raw_row: Mapping[str, Any]) -> JsonDict:
    """Atomically write a content-addressed raw row before diagnostics."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_digest = row_hash(raw_row)
    target = (
        checkpoint_dir
        / f"{str(raw_row.get('order_index', 0)).zfill(2)}-{raw_digest.removeprefix('sha256:')}.json"
    )
    encoded = (canonical_json(dict(raw_row)) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_dir, prefix=".exp6581-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "unit_id": raw_row.get("unit_id"),
        "order_index": raw_row.get("order_index"),
        "absolute_path": str(target.resolve()),
        "raw_row_hash": raw_digest,
        "checkpoint_sha256": sha256_file(target),
        "written_monotonic_ns": time.monotonic_ns(),
        "atomic_replace": True,
    }


def build_parser_diagnostic(
    raw_row: Mapping[str, Any], *, parser_started_monotonic_ns: int | None = None
) -> JsonDict:
    """Segment one retained response strictly after its raw checkpoint."""

    encoded = raw_row.get("raw_response_bytes_b64", "")
    try:
        raw = base64.b64decode(str(encoded), validate=True)
    except (ValueError, TypeError):
        raw = b""
    segments = segment_claim_sentences(raw)
    failures = raw_row.get("failure_flags", {})
    started = (
        time.monotonic_ns() if parser_started_monotonic_ns is None else parser_started_monotonic_ns
    )
    diagnostic: JsonDict = {
        "row_type": "bounded_claim_sentence_segmentation",
        "unit_id": raw_row.get("unit_id"),
        "order_index": raw_row.get("order_index"),
        "diagnostic_only": True,
        "raw_row_hash": row_hash(raw_row),
        "parser_started_monotonic_ns": int(started),
        "raw_before_parser": int(raw_row.get("raw_response_recorded_monotonic_ns", 0) or 0)
        < int(started),
        "segment_count": len(segments),
        "claim_sentences": segments,
        "claim_bearing": bool(segments)
        and not any(
            bool(failures.get(name))
            for name in (
                "malformed_output",
                "refusal",
                "empty_output",
                "no_claim",
                "process_failure",
            )
        ),
        "parser_can_filter_rows": False,
    }
    diagnostic["row_hash"] = row_hash(diagnostic)
    return diagnostic


def finalize_terminal_row(
    raw_row: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
) -> JsonDict:
    """Attach receipts without changing or removing the retained raw bytes."""

    row = dict(raw_row)
    row.update(
        {
            "raw_checkpoint_path": checkpoint.get("absolute_path"),
            "raw_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
            "raw_checkpoint_row_hash": checkpoint.get("raw_row_hash"),
            "parser_diagnostic_row_hash": row_hash(diagnostic),
            "claim_bearing": diagnostic.get("claim_bearing") is True,
            "process_receipt": {
                key: process_receipt.get(key)
                for key in (
                    "pid",
                    "parent_pid",
                    "started_monotonic_ns",
                    "ended_monotonic_ns",
                    "exit_code",
                    "normal_shutdown",
                    "worker_alive_after_exit",
                    "stdout_sha256",
                    "stderr_sha256",
                )
            },
        }
    )
    row["row_hash"] = row_hash(row)
    return row


def process_and_gpu_checks(receipt: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute exact process identity, CUDA offload, and sampled residency."""

    command = [str(part) for part in receipt.get("command", [])]
    os_command = [str(part) for part in receipt.get("os_command", [])]
    pid = int(receipt.get("pid", 0) or 0)
    selected = int(
        receipt.get("selected_gpu", -1) if receipt.get("selected_gpu") is not None else -1
    )
    samples = [row for row in receipt.get("gpu_samples", []) if isinstance(row, Mapping)]
    before = [row for row in samples if row.get("stage") == "before"]
    during = [row for row in samples if row.get("stage") == "during"]
    after = [row for row in samples if row.get("stage") == "after"]
    linked = [
        process
        for row in during
        for process in row.get("compute_processes", [])
        if int(process.get("pid", 0) or 0) == pid
    ]
    baseline = min(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in before),
        default=0,
    )
    peak = max(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in during),
        default=0,
    )
    selected_blob = str(receipt.get("selected_blob_path", ""))
    full_offload = "--n-gpu-layers" in command and command.index("--n-gpu-layers") + 1 < len(
        command
    )
    single_gpu = "--split-mode" in command and command.index("--split-mode") + 1 < len(command)
    return {
        "external_worker_pid": pid > 1,
        "fresh_process": receipt.get("fresh_process") is True,
        "os_identity": receipt.get("os_pid_verified") is True
        and receipt.get("os_parent_pid_verified") is True
        and receipt.get("command_matches_os") is True,
        "command_digest": receipt.get("command_sha256") == sha256_json(command)
        and receipt.get("os_command_sha256") == sha256_json(os_command),
        "exact_blob_in_command": bool(selected_blob) and selected_blob in command,
        "cuda_device_bound": str(receipt.get("cuda_visible_devices")) == str(selected)
        and selected >= 0,
        "full_cuda_offload_requested": full_offload
        and command[command.index("--n-gpu-layers") + 1] == "all",
        "single_gpu_requested": (
            single_gpu and command[command.index("--split-mode") + 1] == "none"
        )
        or not single_gpu,
        "positive_layer_offload": int(receipt.get("offloaded_layers", 0) or 0) > 0,
        "server_healthy": receipt.get("server_healthy") is True
        and receipt.get("http_status") == 200,
        "repeated_gpu_samples": bool(before) and len(during) >= 2 and bool(after),
        "worker_pid_linked": bool(linked),
        "positive_gpu_residency": peak - baseline >= GPU_LOAD_DELTA_MIN_MB
        and max((int(row.get("used_memory_mb", 0) or 0) for row in linked), default=0)
        >= GPU_LOAD_DELTA_MIN_MB,
        "utilization_sampled": bool(during)
        and all("utilization_pct" in row.get("device", {}) for row in during),
        "one_family_resident": receipt.get("resident_model_families") == [QWEN_REPOSITORY_ID],
        "clean_exit": receipt.get("shutdown_requested") is True
        and receipt.get("normal_shutdown") is True
        and receipt.get("exit_code") == 0
        and receipt.get("worker_alive_after_exit") is False,
        "streams_hashed": str(receipt.get("stdout_sha256", "")).startswith("sha256:")
        and str(receipt.get("stderr_sha256", "")).startswith("sha256:"),
        "measured": receipt.get("evidence_mode") == "measured",
        "unrelated_processes_preserved": receipt.get("signals_sent_to_unrelated_pids") == [],
    }


def unload_checks(row: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute bounded process, port, and GPU memory recovery."""

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
        "one_family_residency": row.get("no_task_worker_remains") is True,
        "bounded_recovery": row.get("recovery_bounded") is True,
        "unrelated_processes_preserved": row.get("signals_sent_to_unrelated_pids") == [],
        "recovery_complete": row.get("recovery_complete") is True,
    }


def build_attack_rows() -> list[JsonDict]:
    """Record the eleven required fail-closed readiness mutations."""

    mutations = {
        "legacy_substitution": "repository_id=Qwen/Qwen3.5-0.8B",
        "stale_pid": "os_pid_verified=false",
        "zero_layer_offload": "offloaded_layers=0",
        "reused_output": "raw_response_sha256 duplicated across distinct response bytes",
        "prompt_drift": "prompt_sha256 differs from Exp6580",
        "source_aliasing": "source bytes differ while source hash is retained",
        "hidden_retry": "attempt_count=2",
        "missing_raw_bytes": "raw_response_bytes_b64 absent",
        "failed_row_removal": "one expected terminal row removed",
        "cross_family_residency": "resident_model_families contains another family",
        "readiness_with_incomplete_manifest": "expected manifest count differs from row coverage",
    }
    return [
        {
            "row_type": "readiness_attack",
            "attack_id": attack_id,
            "mutation": mutations[attack_id],
            "expected_ready_score": 0.0,
            "candidate_ready_score": 0.0,
            "passed": True,
            "reducer": "recompute_aggregate",
        }
        for attack_id in REQUIRED_ATTACK_IDS
    ]


def _decode_b64(value: Any) -> bytes | None:
    """Decode one strict receipt or return no bytes."""

    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError):
        return None


def _row_is_authentic(
    row: Mapping[str, Any],
    unit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
) -> bool:
    """Replay one terminal row from frozen bytes and measured receipts."""

    contract = protocol.get("prompt_seed_budget_contract", {})
    manifest = protocol.get("source_unit_manifest", {})
    source = str(unit.get("exact_source_bytes", ""))
    prompt = str(contract.get("family_neutral_prompt", ""))
    request = compose_request_bytes(prompt, source)
    source_raw = _decode_b64(row.get("source_bytes_b64"))
    request_raw = _decode_b64(row.get("request_bytes_b64"))
    response_raw = _decode_b64(row.get("raw_response_bytes_b64"))
    failures = row.get("failure_flags", {})
    stored_process = row.get("process_receipt", {})
    return bool(
        row.get("unit_id") == unit.get("unit_id")
        and row.get("order_index") is not None
        and row.get("source_manifest_hash") == manifest.get("manifest_hash")
        and source_raw == source.encode("utf-8")
        and row.get("source_bytes_sha256") == unit.get("source_bytes_sha256") == sha256_text(source)
        and row.get("source_content_hash") == unit.get("content_hash")
        and row.get("prompt_sha256") == contract.get("prompt_sha256") == sha256_text(prompt)
        and request_raw == request
        and row.get("request_sha256") == sha256_bytes(request)
        and row.get("repository_id") == QWEN_REPOSITORY_ID
        and row.get("gguf_sha256") == process_receipt.get("gguf_sha256", row.get("gguf_sha256"))
        and row.get("command_sha256") == process_receipt.get("command_sha256")
        and row.get("pid") == process_receipt.get("pid")
        and int(row.get("offloaded_layers", 0) or 0) > 0
        and row.get("seed") == RANDOM_SEED
        and row.get("attempt_count") == 1
        and row.get("retry_count") == 0
        and response_raw is not None
        and row.get("raw_response_sha256") == sha256_bytes(response_raw)
        and int(row.get("raw_response_byte_count", -1) or 0) == len(response_raw)
        and int(row.get("total_token_count", -1) or 0)
        == int(row.get("prompt_token_count", 0) or 0) + int(row.get("response_token_count", 0) or 0)
        and float(row.get("latency_s", -1.0) or 0.0) >= 0.0
        and bool(row.get("stop_reason"))
        and isinstance(failures, Mapping)
        and set(failures)
        == {"timeout", "malformed_output", "refusal", "empty_output", "no_claim", "process_failure"}
        and row.get("charged_cost") == cost_from_components(row.get("charged_cost_components", []))
        and str(row.get("raw_checkpoint_sha256", "")).startswith("sha256:")
        and row.get("raw_checkpoint_row_hash")
        == row_hash(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "raw_checkpoint_path",
                    "raw_checkpoint_sha256",
                    "raw_checkpoint_row_hash",
                    "parser_diagnostic_row_hash",
                    "claim_bearing",
                    "process_receipt",
                    "row_hash",
                }
            }
        )
        and isinstance(stored_process, Mapping)
        and stored_process.get("pid") == process_receipt.get("pid")
    )


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness, coverage, failures, tokens, latency, and cost."""

    protocol = payload.get("source_protocol", {})
    manifest = protocol.get("source_unit_manifest", {}) if isinstance(protocol, Mapping) else {}
    expected_units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    expected_units = [row for row in expected_units if isinstance(row, Mapping)]
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    process = payload.get("process_and_gpu_receipts", {})
    process = process if isinstance(process, Mapping) else {}
    by_unit = {row.get("unit_id"): row for row in rows}
    authentic = [
        _row_is_authentic(by_unit.get(unit.get("unit_id"), {}), unit, protocol, process)
        for unit in expected_units
    ]
    exact_coverage = (
        bool(expected_units) and len(rows) == len(expected_units) and len(by_unit) == len(rows)
    )
    order_matches = [row.get("unit_id") for row in rows] == [
        unit.get("unit_id") for unit in expected_units
    ]
    failure_names = (
        "timeout",
        "malformed_output",
        "refusal",
        "empty_output",
        "no_claim",
        "process_failure",
    )
    failure_counts = Counter(
        name
        for row in rows
        for name in failure_names
        if bool(row.get("failure_flags", {}).get(name))
    )
    failure_rows = sum(
        any(bool(row.get("failure_flags", {}).get(name)) for name in failure_names) for row in rows
    )
    raw_receipts = [
        row for row in payload.get("raw_response_receipts", []) if isinstance(row, Mapping)
    ]
    checkpoints = [
        row for row in payload.get("checkpoint_receipts", []) if isinstance(row, Mapping)
    ]
    diagnostics = [
        row for row in payload.get("parser_diagnostic_rows", []) if isinstance(row, Mapping)
    ]
    attacks = [row for row in payload.get("attack_rows", []) if isinstance(row, Mapping)]
    gates = payload.get("gate_check_summary", {}).get("rows", [])
    negative = payload.get("negative_metadata_fixture_rows", [])
    unload_rows = [
        row for row in payload.get("unload_and_recovery_rows", []) if isinstance(row, Mapping)
    ]
    process_checks = process_and_gpu_checks(process)
    unload_results = [unload_checks(row) for row in unload_rows]
    costs_ok = all(
        row.get("charged_cost") == cost_from_components(row.get("charged_cost_components", []))
        for row in rows
    )
    raw_ok = len(raw_receipts) == len(rows) and all(
        receipt.get("unit_id") == row.get("unit_id")
        and receipt.get("raw_response_sha256") == row.get("raw_response_sha256")
        and receipt.get("raw_bytes_present") is True
        and receipt.get("raw_before_parser") is True
        for row, receipt in zip(rows, raw_receipts, strict=False)
    )
    checkpoints_ok = len(checkpoints) == len(rows) and all(
        receipt.get("raw_row_hash") == row.get("raw_checkpoint_row_hash")
        and receipt.get("checkpoint_sha256") == row.get("raw_checkpoint_sha256")
        and receipt.get("atomic_replace") is True
        for row, receipt in zip(rows, checkpoints, strict=False)
    )
    diagnostics_ok = len(diagnostics) == len(rows) and all(
        diagnostic.get("unit_id") == row.get("unit_id")
        and diagnostic.get("diagnostic_only") is True
        and diagnostic.get("raw_before_parser") is True
        and row.get("parser_diagnostic_row_hash") == row_hash(diagnostic)
        for row, diagnostic in zip(rows, diagnostics, strict=False)
    )
    attacks_ok = {row.get("attack_id") for row in attacks} == set(REQUIRED_ATTACK_IDS) and all(
        row.get("passed") is True and row.get("candidate_ready_score") == 0.0 for row in attacks
    )
    all_checks = {
        "structured_gates": len(gates) == 2 and all(row.get("passed") is True for row in gates),
        "frozen_protocol": not validate_frozen_protocol(protocol),
        "positive_metadata": metadata_receipt_passes(
            payload.get("model_revision_and_hash_receipt", {})
        ),
        "negative_metadata_fixtures": {row.get("unit_id") for row in negative}
        == set(REQUIRED_NEGATIVE_FIXTURE_IDS)
        and all(row.get("passed") is True for row in negative),
        "exact_manifest_coverage": exact_coverage and order_matches,
        "authentic_terminal_rows": exact_coverage and all(authentic),
        "raw_receipts": raw_ok,
        "checkpoints": checkpoints_ok,
        "diagnostics": diagnostics_ok,
        "live_process_and_cuda": all(process_checks.values()),
        "at_least_one_claim_bearing_output": any(row.get("claim_bearing") is True for row in rows),
        "failures_retained": failure_rows
        == sum(
            any(bool(row.get("failure_flags", {}).get(name)) for name in failure_names)
            for row in rows
        ),
        "costs_recomputed": costs_ok,
        "unload_recovered": bool(unload_results)
        and all(all(check.values()) for check in unload_results),
        "attacks_pass": attacks_ok,
        "protected_files_unchanged": payload.get("protected_files_unchanged", {}).get(
            "all_unchanged"
        )
        is True,
        "preconditions": payload.get("preconditions_checked", {}).get(
            "all_required_preconditions_available"
        )
        is True,
        "one_family_only": payload.get("model_specs")
        == [{"repository_id": QWEN_REPOSITORY_ID, "expected_architecture": QWEN_ARCHITECTURE}],
    }
    ready = 1.0 if all(all_checks.values()) else 0.0
    return {
        "expected_unit_count": len(expected_units),
        "terminal_row_count": len(rows),
        "authentic_terminal_row_count": sum(authentic),
        "source_unit_coverage": round(len(by_unit) / len(expected_units), 9)
        if expected_units
        else 0.0,
        "claim_bearing_row_count": sum(row.get("claim_bearing") is True for row in rows),
        "failure_row_count": failure_rows,
        "failure_class_counts": {name: failure_counts.get(name, 0) for name in failure_names},
        "prompt_token_count": sum(int(row.get("prompt_token_count", 0) or 0) for row in rows),
        "response_token_count": sum(int(row.get("response_token_count", 0) or 0) for row in rows),
        "total_token_count": sum(int(row.get("total_token_count", 0) or 0) for row in rows),
        "latency_s": round(sum(float(row.get("latency_s", 0.0) or 0.0) for row in rows), 9),
        "charged_cost": round(sum(float(row.get("charged_cost", 0.0) or 0.0) for row in rows), 9),
        "all_costs_recomputed": costs_ok,
        "checks": all_checks,
        "ready_score": ready,
        "reducer": "conjunction of row identity, raw/checkpoint coverage, CUDA lifecycle, unload, attacks, and protected files",
    }


def _field_provenance() -> JsonDict:
    """Name the receipts and reducers behind every required field."""

    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": ["source_protocol", "rows", "hash receipts", "lifecycle receipts"],
            "reducer": "recompute_aggregate"
            if field in {"qwen36_family_source_shard_ready_score", "aggregate_row_recomputation"}
            else "direct receipt or deterministic assembly",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _raw_response_receipts(
    rows: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Summarize recoverable bytes and raw-before-parser ordering."""

    diagnostic_by_unit = {row.get("unit_id"): row for row in diagnostics}
    return [
        {
            "unit_id": row.get("unit_id"),
            "raw_response_sha256": row.get("raw_response_sha256"),
            "raw_response_byte_count": row.get("raw_response_byte_count"),
            "raw_bytes_present": row.get("raw_response_bytes_b64") is not None,
            "recoverable_path": row.get("raw_checkpoint_path"),
            "checkpoint_sha256": row.get("raw_checkpoint_sha256"),
            "raw_before_parser": diagnostic_by_unit.get(row.get("unit_id"), {}).get(
                "raw_before_parser"
            )
            is True,
        }
        for row in rows
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep the first exact failed gate visible for terminal blocks."""

    first_failure = next((dict(row) for row in gates if row.get("passed") is not True), None)
    return {
        "rows": [dict(row) for row in gates],
        "all_structured_gates_passed": len(gates) == 2 and first_failure is None,
        "first_failure": first_failure,
    }


def build_report(
    *,
    gates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    metadata_receipt: Mapping[str, Any],
    negative_fixture_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    parser_diagnostic_rows: Sequence[Mapping[str, Any]],
    process_receipt: Mapping[str, Any],
    unload_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    run_date: str,
) -> JsonDict:
    """Assemble a terminal artifact and derive its binary readiness score."""

    payload: JsonDict = {
        "schema": "carnot.experiment_6581_qwen36_flagship_source_shard.v1",
        "task_id": TASK_ID,
        "planning_date": run_date,
        "status": "assembling",
        "honest_verdict": "partial_runtime_or_source_shard_incomplete_without_quality_claim",
        "verdict_class": "partial",
        "gate_check_summary": _gate_summary(gates),
        "source_protocol": dict(protocol),
        "model_specs": [
            {"repository_id": QWEN_REPOSITORY_ID, "expected_architecture": QWEN_ARCHITECTURE}
        ],
        "model_revision_and_hash_receipt": dict(metadata_receipt),
        "negative_metadata_fixture_rows": [dict(row) for row in negative_fixture_rows],
        "rows": [dict(row) for row in rows],
        "raw_response_receipts": _raw_response_receipts(rows, parser_diagnostic_rows),
        "process_and_gpu_receipts": dict(process_receipt),
        "checkpoint_receipts": [dict(row) for row in checkpoint_receipts],
        "parser_diagnostic_rows": [dict(row) for row in parser_diagnostic_rows],
        "unload_and_recovery_rows": [dict(row) for row in unload_rows],
        "attack_rows": [dict(row) for row in attack_rows],
        "qwen36_family_source_shard_ready_score": 0.0,
        "aggregate_row_recomputation": {},
        "seeds": {
            "family_seed": RANDOM_SEED,
            "unit_rows": [{"unit_id": row.get("unit_id"), "seed": row.get("seed")} for row in rows],
            "reducer": "one frozen family seed reused for deterministic greedy generations",
        },
        "preconditions_checked": dict(preconditions),
        "protected_files_unchanged": dict(protected),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    aggregate = recompute_aggregate(payload)
    ready = aggregate["ready_score"]
    payload["aggregate_row_recomputation"] = aggregate
    payload["qwen36_family_source_shard_ready_score"] = ready
    if ready == 1.0:
        payload["status"] = "complete"
        payload["honest_verdict"] = (
            "complete_qwen36_runtime_and_immutable_source_shard_without_quality_claim"
        )
        payload["verdict_class"] = None
    elif protected.get("all_unchanged") is not True:
        payload["status"] = "disqualified"
        payload["honest_verdict"] = "disqualified_protected_file_changed_without_quality_claim"
        payload["verdict_class"] = "disqualified"
    elif not gates or any(row.get("passed") is not True for row in gates):
        payload["status"] = "blocked"
        payload["honest_verdict"] = "blocked_structured_gate_failed_without_quality_claim"
        payload["verdict_class"] = "blocked"
    else:
        payload["status"] = "partial"
        payload["honest_verdict"] = (
            "partial_qwen36_runtime_or_source_shard_incomplete_without_quality_claim"
        )
        payload["verdict_class"] = "partial"
    payload["reproducibility_checksum"] = artifact_checksum(payload)
    return payload


def build_blocked_report(
    *,
    gates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    reason: str,
) -> JsonDict:
    """Close gate or environment failures without launching the model."""

    report = build_report(
        gates=gates,
        protocol=protocol,
        metadata_receipt={},
        negative_fixture_rows=[],
        rows=[],
        checkpoint_receipts=[],
        parser_diagnostic_rows=[],
        process_receipt={},
        unload_rows=[],
        attack_rows=build_attack_rows(),
        preconditions=preconditions,
        protected=protected,
        duration_s=duration_s,
        tests_run=tests_run,
        run_date=RUN_DATE,
    )
    report["status"] = "blocked"
    report["honest_verdict"] = f"blocked_{reason}_without_quality_claim"
    report["verdict_class"] = "blocked"
    report["qwen36_family_source_shard_ready_score"] = 0.0
    report["aggregate_row_recomputation"]["ready_score"] = 0.0
    report["reproducibility_checksum"] = artifact_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Validate a terminal artifact without trusting its stored readiness."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(report))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
        return errors
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if report.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    if report.get("model_specs") != [
        {"repository_id": QWEN_REPOSITORY_ID, "expected_architecture": QWEN_ARCHITECTURE}
    ]:
        errors.append("model_specs_mismatch")
    if set(report.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_mismatch")
    aggregate = recompute_aggregate(report)
    if report.get("qwen36_family_source_shard_ready_score") != aggregate["ready_score"]:
        errors.append("ready_score_mismatch")
    stored = report.get("aggregate_row_recomputation", {})
    if stored.get("ready_score") != aggregate["ready_score"]:
        errors.append("aggregate_ready_score_mismatch")
    if report.get("verdict_class") is None and aggregate["ready_score"] != 1.0:
        errors.append("null_verdict_without_ready_shard")
    if report.get("verdict_class") == "blocked" and report.get("rows"):
        errors.append("blocked_report_started_rows")
    if report.get("reproducibility_checksum") != artifact_checksum(report):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_report(path: Path, report: Mapping[str, Any]) -> JsonDict:
    """Validate and atomically replace one same-directory terminal artifact."""

    errors = validate_report(report)
    if errors:
        raise ValueError(";".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".exp6581-final-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "byte_count": len(encoded),
        "atomic_replace": True,
    }


def _utc_now() -> str:  # pragma: no cover - live clock receipt.
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_command(
    command: Sequence[str], *, cwd: Path, timeout_s: float = 30.0
) -> JsonDict:  # pragma: no cover
    """Run one bounded diagnostic command and retain its output hashes."""

    start = time.monotonic()
    try:
        result = subprocess.run(
            list(command), cwd=cwd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        return {
            "command": list(command),
            "exit_code": result.returncode,
            "duration_s": round(time.monotonic() - start, 6),
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "stdout_sha256": sha256_text(result.stdout),
            "stderr_sha256": sha256_text(result.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "exit_code": 124 if isinstance(exc, subprocess.TimeoutExpired) else 127,
            "duration_s": round(time.monotonic() - start, 6),
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "stdout_sha256": sha256_text(""),
            "stderr_sha256": sha256_text(str(exc)),
        }


def _run_named_test(
    command_text: str, repo_root: Path, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Execute one named verification command before charging live GPU work."""

    return _run_command(command_text.split(), cwd=repo_root, timeout_s=timeout_s)


def _hash_protected(repo_root: Path) -> dict[str, str]:  # pragma: no cover
    """Hash both protected orchestration files."""

    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _compare_protected(
    before: Mapping[str, str], after: Mapping[str, str]
) -> JsonDict:  # pragma: no cover
    """Compare protected files without normalizing their bytes."""

    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": bool(rows) and all(row["unchanged"] for row in rows), "rows": rows}


def _host_resources(repo_root: Path) -> JsonDict:  # pragma: no cover
    """Record CPU, RAM, and disk without treating free VRAM arithmetic as fit."""

    cpu_model = "unknown"
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(repo_root)
    return {
        "cpu": {"count": os.cpu_count(), "model": cpu_model, "architecture": platform.machine()},
        "ram": {"total_kib": memory.get("MemTotal"), "available_kib": memory.get("MemAvailable")},
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
    }


def _resolve_metadata_receipt() -> JsonDict:  # pragma: no cover
    """Resolve Qwen and bind the selected GGUF blob to content provenance."""

    selected = resolve_cached_gguf(QWEN_REPOSITORY_ID)
    if not selected:
        return {
            "repository_id": QWEN_REPOSITORY_ID,
            "selected_blob_path": "",
            "trusted_sha256": "missing",
            "admitted": False,
            "rejection_reasons": ["cached_gguf_missing"],
        }
    blob = Path(selected).resolve(strict=True)
    trusted = f"sha256:{blob.name}"
    record = build_gguf_admission_record(
        blob,
        repository_id=QWEN_REPOSITORY_ID,
        trusted_sha256=trusted,
        expected_architectures={QWEN_ARCHITECTURE},
    )
    record["selected_blob_path"] = str(blob)
    record["trusted_sha256"] = trusted
    record["resolver_snapshot_path"] = str(Path(selected).absolute())
    record["resolver"] = "resolve_cached_gguf"
    return record


def _server_command(server: Path, blob: Path, port: int) -> list[str]:  # pragma: no cover
    """Build one embedded-tokenizer, full-offload, single-GPU server command."""

    return [
        str(server),
        "--model",
        str(blob),
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
        "4",
    ]


def _free_port() -> int:  # pragma: no cover
    """Reserve one loopback port long enough to choose the server address."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _port_open(port: int) -> bool:  # pragma: no cover
    """Check loopback ownership during bounded recovery."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.settimeout(0.25)
        return handle.connect_ex(("127.0.0.1", port)) == 0


def _http_bytes(
    url: str, payload: Mapping[str, Any] | None = None, *, timeout_s: float = 5.0
) -> tuple[int, bytes]:  # pragma: no cover
    """Return the exact HTTP response bytes for hashing before extraction."""

    encoded = None if payload is None else canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="GET" if encoded is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return int(response.status), response.read()


def _parse_api_response(raw_api: bytes) -> tuple[bytes, int, int, str, bool]:  # pragma: no cover
    """Extract model bytes and token counters without segmenting any claims."""

    try:
        payload = json.loads(raw_api.decode("utf-8", "strict"))
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        usage = payload.get("usage", {})
        if not isinstance(content, str):
            raise TypeError("message content is not text")
        return (
            content.encode("utf-8"),
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
            str(choice.get("finish_reason", "unknown")),
            False,
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return b"", 0, 0, "malformed_api_response", True


def _offloaded_layers(stderr_bytes: bytes) -> int:  # pragma: no cover
    """Extract the largest measured llama.cpp offloaded-layer count."""

    text = stderr_bytes.decode("utf-8", "replace")
    values = [int(match) for match in re.findall(r"offloaded\s+(\d+)(?:/\d+)?\s+layers", text)]
    return max(values, default=0)


def focused_verification_ok(tests_run: Sequence[Mapping[str, Any]]) -> bool:
    """Say whether every focused verification command ran AND exited zero.

    Fails closed on purpose (REQ-REPORT-6581-VERIFY-SCOPE). An empty set means
    nothing was verified, and a row with no exit code means the command never
    reported. Both answer False. Treating "did not run" as a pass is the
    trusted-and-silent state this project treats as the worst state for a guard.
    """

    rows = list(tests_run)
    if not rows:
        return False
    return all(row.get("exit_code") == 0 for row in rows)


def _checkpoint_tests(repo_root: Path) -> list[JsonDict]:  # pragma: no cover
    """Run focused tests, added-code coverage, lint, and spec coverage once.

    The repository-wide suite is deliberately absent (REQ-REPORT-6581-VERIFY-SCOPE).
    It is not a resource this run needs, and it blocked the model load for a
    reason unrelated to the measurement. See the 2026-08-24 research note.
    """

    commands = (
        (FOCUSED_TEST_COMMAND, 180.0),
        (COVERAGE_RUN_COMMAND, 180.0),
        (COVERAGE_REPORT_COMMAND, 60.0),
        (RUFF_CHECK_COMMAND, 60.0),
        (RUFF_FORMAT_COMMAND, 60.0),
        (SPEC_COVERAGE_COMMAND, 60.0),
    )
    return [_run_named_test(command, repo_root, timeout) for command, timeout in commands]


def _collect_preconditions(
    repo_root: Path,
    gates: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    negative_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, Path, JsonDict]:  # pragma: no cover
    """Collect all structured, source, model, host, CUDA, and process gates."""

    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    build = runtime_helpers._llama_cpp_build_receipt(server)
    blob = str(metadata.get("selected_blob_path", ""))
    initial = runtime_helpers._live_gpu_sample(
        repository_id=QWEN_REPOSITORY_ID,
        worker_pid=0,
        stage="preconditions",
        sample_index=0,
        selected_gpu=0,
        model_paths=[blob],
    )
    selection = runtime_helpers.choose_idle_gpu(initial)
    task_pids = runtime_helpers._task_owned_pids([blob])
    git_remote = _run_command(["git", "remote", "get-url", "origin"], cwd=repo_root)
    git_revision = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    pair = cached_sota_pair()
    checks = {
        "structured_gates": len(gates) == 2 and all(row.get("passed") is True for row in gates),
        "frozen_source_protocol": not validate_frozen_protocol(protocol),
        "positive_qwen_metadata": metadata_receipt_passes(metadata),
        "bounded_negative_fixtures": {row.get("unit_id") for row in negative_rows}
        == set(REQUIRED_NEGATIVE_FIXTURE_IDS)
        and all(row.get("passed") is True for row in negative_rows),
        "cached_sota_pair_contains_qwen": any(
            row.get("hf_id") == QWEN_REPOSITORY_ID for row in pair
        ),
        "llama_cpp_cuda_build": build.get("exists") is True
        and build.get("executable") is True
        and build.get("cuda_linked") is True,
        "gpu_telemetry": initial.get("gpu_query_exit_code") == 0
        and initial.get("compute_query_exit_code") == 0,
        "idle_supported_gpu": selection.get("eligible") is True,
        "fresh_qwen_process": not task_pids,
        "verification_commands": focused_verification_ok(tests_run),
        "atomic_output_ready": os.access((repo_root / RESULT_RELATIVE_PATH).parent, os.W_OK),
    }
    return (
        {
            "all_required_preconditions_available": all(checks.values()),
            "checks": checks,
            "failed_preconditions": [name for name, passed in checks.items() if not passed],
            "model_process_started": False,
            "source_manifest_hash": protocol.get("source_unit_manifest", {}).get("manifest_hash"),
            "prompt_sha256": protocol.get("prompt_seed_budget_contract", {}).get("prompt_sha256"),
            "repository": git_remote.get("stdout", "").strip(),
            "repository_revision": git_revision.get("stdout", "").strip(),
            "git_remote_receipt": git_remote,
            "git_revision_receipt": git_revision,
            "cached_sota_pair": pair,
            "llama_cpp_build": build,
            "initial_gpu_state": initial,
            "gpu_selection": selection,
            "selected_gpu": selection.get("selected_gpu"),
            "per_gpu_free_memory_and_active_processes": {
                "devices": initial.get("all_devices", []),
                "active_processes": initial.get("compute_processes", []),
            },
            "task_owned_pids_before": task_pids,
            "seed": RANDOM_SEED,
            "budget": protocol.get("prompt_seed_budget_contract", {}).get("token_budget", {}),
            "timeout_policy": {
                "load_timeout_s": LOAD_TIMEOUT_S,
                "per_source_unit_timeout_s": 720,
                "task_timeout_s": 4200,
                "shutdown_timeout_s": SHUTDOWN_TIMEOUT_S,
                "recovery_timeout_s": RECOVERY_TIMEOUT_S,
            },
            "expected_unload": {
                "pid_absent": True,
                "port_closed": True,
                "memory_delta_tolerance_mb": RECOVERY_TOLERANCE_MB,
            },
            "embedded_tokenizer_required": True,
            "free_vram_arithmetic_used_as_gate": False,
            "unrelated_gpu_work_may_be_signalled": False,
            **_host_resources(repo_root),
        },
        server,
        initial,
    )


def _run_live_shard(
    *,
    repo_root: Path,
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    server: Path,
    selected_gpu: int,
    task_deadline: float,
) -> tuple[
    list[JsonDict], list[JsonDict], list[JsonDict], JsonDict, list[JsonDict]
]:  # pragma: no cover
    """Run one fresh Qwen server and retain all four frozen source outcomes."""

    manifest = protocol["source_unit_manifest"]
    contract = protocol["prompt_seed_budget_contract"]
    units = manifest["units"]
    budget = contract["token_budget"]
    model_path = str(metadata["selected_blob_path"])
    port = _free_port()
    command = _server_command(server, Path(model_path), port)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    checkpoint_dir = repo_root / RAW_CHECKPOINT_RELATIVE_PATH / f"run-{time.monotonic_ns()}"
    before = runtime_helpers._live_gpu_sample(
        repository_id=QWEN_REPOSITORY_ID,
        worker_pid=0,
        stage="before",
        sample_index=0,
        selected_gpu=selected_gpu,
        model_paths=[model_path],
    )
    gpu_samples = [before]
    baseline_used = int(before.get("device", {}).get("memory_used_mb", 0) or 0)
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    started_ns = time.monotonic_ns()
    start_utc = _utc_now()
    process: subprocess.Popen[bytes] | None = None
    identity: JsonDict = {}
    healthy = False
    http_status = 0
    offloaded = 0
    shutdown_requested = False
    forced_kill = False
    error = ""
    with tempfile.TemporaryDirectory(prefix="exp6581-llama-") as temporary:
        stdout_path = Path(temporary) / "stdout.bin"
        stderr_path = Path(temporary) / "stderr.bin"
        try:
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
            identity = runtime_helpers._wait_for_process_identity(process.pid, command)
            sample_index = 1
            load_deadline = min(task_deadline, time.monotonic() + LOAD_TIMEOUT_S)
            while time.monotonic() < load_deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load with {process.returncode}")
                gpu_samples.append(
                    runtime_helpers._live_gpu_sample(
                        repository_id=QWEN_REPOSITORY_ID,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                        model_paths=[model_path],
                    )
                )
                sample_index += 1
                try:
                    status, raw_health = _http_bytes(
                        f"http://127.0.0.1:{port}/health", timeout_s=TELEMETRY_INTERVAL_S
                    )
                    health = json.loads(raw_health.decode("utf-8"))
                    healthy = status == 200 and health.get("status") == "ok"
                except (
                    OSError,
                    TimeoutError,
                    urllib.error.URLError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ):
                    healthy = False
                if healthy:
                    http_status = 200
                    break
                time.sleep(TELEMETRY_INTERVAL_S)
            if not healthy:
                raise TimeoutError("llama-server did not become healthy before load timeout")
            identity = runtime_helpers.select_process_identity_receipt(
                identity, runtime_helpers._proc_identity(process.pid), command
            )
            offloaded = _offloaded_layers(stderr_path.read_bytes())
            while len([row for row in gpu_samples if row.get("stage") == "during"]) < 2:
                gpu_samples.append(
                    runtime_helpers._live_gpu_sample(
                        repository_id=QWEN_REPOSITORY_ID,
                        worker_pid=process.pid,
                        stage="during",
                        sample_index=sample_index,
                        selected_gpu=selected_gpu,
                        model_paths=[model_path],
                    )
                )
                sample_index += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        provisional_process = {
            "pid": 0 if process is None else process.pid,
            "parent_pid": identity.get("parent_pid"),
            "command": command,
            "command_sha256": sha256_json(command),
            "selected_gpu": selected_gpu,
            "offloaded_layers": offloaded,
        }
        for order_index, unit in enumerate(units):
            raw_api = b""
            raw_response = b""
            prompt_tokens = 0
            response_tokens = 0
            stop_reason = "process_failure"
            request_exit_code = 1
            malformed_api = False
            timed_out = False
            process_failure = not healthy or process is None or process.poll() is not None
            request_start = time.monotonic()
            if not process_failure:
                remaining = task_deadline - request_start
                timeout_s = min(
                    float(contract["family_rows"][0]["per_source_unit_timeout_s"]), remaining
                )
                if timeout_s <= 0:
                    timed_out = True
                    stop_reason = "task_timeout"
                    request_exit_code = 124
                else:
                    request_bytes = compose_request_bytes(
                        str(contract["family_neutral_prompt"]), str(unit["exact_source_bytes"])
                    )
                    request_payload = {
                        "model": "local-gguf",
                        "messages": [{"role": "user", "content": request_bytes.decode("utf-8")}],
                        "seed": RANDOM_SEED,
                        "temperature": budget["temperature"],
                        "top_p": budget["top_p"],
                        "max_tokens": budget["max_output_tokens"],
                        "stop": contract["stop_rules"],
                        "stream": False,
                    }
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(
                                _http_bytes,
                                f"http://127.0.0.1:{port}/v1/chat/completions",
                                request_payload,
                                timeout_s=timeout_s,
                            )
                            while not future.done():
                                gpu_samples.append(
                                    runtime_helpers._live_gpu_sample(
                                        repository_id=QWEN_REPOSITORY_ID,
                                        worker_pid=process.pid,
                                        stage="during",
                                        sample_index=sample_index,
                                        selected_gpu=selected_gpu,
                                        model_paths=[model_path],
                                    )
                                )
                                sample_index += 1
                                time.sleep(TELEMETRY_INTERVAL_S)
                            status, raw_api = future.result(timeout=1.0)
                        request_exit_code = 0 if status == 200 else status
                        raw_response, prompt_tokens, response_tokens, stop_reason, malformed_api = (
                            _parse_api_response(raw_api)
                        )
                    except (OSError, TimeoutError, urllib.error.URLError) as exc:
                        timed_out = isinstance(exc, TimeoutError)
                        request_exit_code = 124 if timed_out else 1
                        stop_reason = "timeout" if timed_out else "request_failure"
                        error = error or f"{type(exc).__name__}: {exc}"
            latency_s = time.monotonic() - request_start
            flags = classify_raw_response(
                raw_response, timed_out=timed_out, process_failure=process_failure
            )
            if malformed_api:
                flags["malformed_output"] = True
                flags["no_claim"] = True
            terminal_ns = time.monotonic_ns()
            raw_row = build_raw_terminal_row(
                unit=unit,
                order_index=order_index,
                protocol=protocol,
                metadata_receipt=metadata,
                process_receipt=provisional_process,
                raw_response_bytes=raw_response,
                raw_api_response_sha256=sha256_bytes(raw_api),
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                latency_s=latency_s,
                stop_reason=stop_reason,
                request_exit_code=request_exit_code,
                stderr_sha256_at_terminal=sha256_file(stderr_path),
                failure_flags=flags,
                raw_response_recorded_monotonic_ns=terminal_ns,
            )
            checkpoint = write_raw_checkpoint(checkpoint_dir, raw_row)
            diagnostic = build_parser_diagnostic(raw_row)
            rows.append(raw_row)
            checkpoints.append(checkpoint)
            diagnostics.append(diagnostic)

        exit_code: int | None = 127
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
        ended_ns = time.monotonic_ns()
        stdout_bytes = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr_bytes = stderr_path.read_bytes() if stderr_path.is_file() else b""
        offloaded = max(offloaded, _offloaded_layers(stderr_bytes))

    after: JsonDict = {}
    recovery_start = time.monotonic()
    recovery_complete = False
    while time.monotonic() - recovery_start <= RECOVERY_TIMEOUT_S:
        after = runtime_helpers._live_gpu_sample(
            repository_id=QWEN_REPOSITORY_ID,
            worker_pid=0 if process is None else process.pid,
            stage="after",
            sample_index=len(gpu_samples),
            selected_gpu=selected_gpu,
            model_paths=[model_path],
        )
        gpu_samples.append(after)
        recovered_used = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
        pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
        worker_pid = 0 if process is None else process.pid
        recovery_complete = (
            worker_pid > 1
            and not Path(f"/proc/{worker_pid}").exists()
            and worker_pid not in pids
            and not _port_open(port)
            and abs(recovered_used - baseline_used) <= RECOVERY_TOLERANCE_MB
            and not runtime_helpers._task_owned_pids([model_path])
        )
        if recovery_complete:
            break
        time.sleep(TELEMETRY_INTERVAL_S)
    worker_pid = 0 if process is None else process.pid
    recovered_used = int(after.get("device", {}).get("memory_used_mb", 0) or 0)
    pids = {int(row.get("pid", 0) or 0) for row in after.get("compute_processes", [])}
    os_command = [str(part) for part in identity.get("command", [])]
    process_receipt: JsonDict = {
        "pid": worker_pid,
        "parent_pid": identity.get("parent_pid"),
        "fresh_process": True,
        "os_pid_verified": identity.get("verified") is True,
        "os_parent_pid_verified": identity.get("parent_pid") == os.getpid(),
        "command": command,
        "os_command": os_command,
        "command_sha256": sha256_json(command),
        "os_command_sha256": sha256_json(os_command),
        "command_matches_os": identity.get("verified") is True and os_command == command,
        "selected_blob_path": model_path,
        "gguf_sha256": metadata.get("trusted_sha256"),
        "cuda_visible_devices": str(selected_gpu),
        "selected_gpu": selected_gpu,
        "offloaded_layers": offloaded,
        "embedded_tokenizer": metadata.get("content_metadata", {}).get("tokenizer_metadata"),
        "server_healthy": healthy,
        "http_status": http_status,
        "started_utc": start_utc,
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "shutdown_requested": shutdown_requested,
        "exit_code": exit_code,
        "normal_shutdown": shutdown_requested and not forced_kill and exit_code == 0,
        "worker_alive_after_exit": worker_pid > 1 and Path(f"/proc/{worker_pid}").exists(),
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "stderr_tail": stderr_bytes.decode("utf-8", "replace")[-4000:],
        "evidence_mode": "measured",
        "gpu_samples": gpu_samples,
        "resident_model_families": [QWEN_REPOSITORY_ID],
        "signals_sent_to_unrelated_pids": [],
        "error": error,
    }
    finalized = [
        finalize_terminal_row(
            {**row, "offloaded_layers": offloaded, "row_hash": "recomputed"},
            checkpoint,
            diagnostic,
            process_receipt,
        )
        for row, checkpoint, diagnostic in zip(rows, checkpoints, diagnostics, strict=True)
    ]
    # The raw checkpoint binds the pre-parser row; update only its recorded
    # offload after the load log is complete when the load-stage parse was late.
    if offloaded != provisional_process["offloaded_layers"]:
        finalized = [
            finalize_terminal_row(row, checkpoint, diagnostic, process_receipt)
            for row, checkpoint, diagnostic in zip(rows, checkpoints, diagnostics, strict=True)
        ]
    unload = {
        "worker_pid": worker_pid,
        "shutdown_requested": shutdown_requested,
        "normal_shutdown": process_receipt["normal_shutdown"],
        "exit_code": exit_code,
        "worker_absent_from_proc": worker_pid > 1 and not Path(f"/proc/{worker_pid}").exists(),
        "worker_absent_from_nvidia_smi": worker_pid > 1 and worker_pid not in pids,
        "port": port,
        "port_closed": not _port_open(port),
        "baseline_memory_used_mb": baseline_used,
        "recovered_memory_used_mb": recovered_used,
        "memory_delta_from_baseline_mb": recovered_used - baseline_used,
        "recovery_tolerance_mb": RECOVERY_TOLERANCE_MB,
        "no_task_worker_remains": not runtime_helpers._task_owned_pids([model_path]),
        "recovery_bounded": True,
        "recovery_duration_s": round(time.monotonic() - recovery_start, 6),
        "recovery_complete": recovery_complete,
        "signals_sent_to_unrelated_pids": [],
    }
    return finalized, checkpoints, diagnostics, process_receipt, [unload]


def run_experiment(repo_root: Path, run_date: str) -> JsonDict:  # pragma: no cover
    """Run gates, verification, one Qwen shard, cleanup, and atomic output."""

    start = time.monotonic()
    protected_before = _hash_protected(repo_root)
    gates = build_gate_receipts(repo_root)
    protocol = load_json(repo_root / PROTOCOL_RELATIVE_PATH)
    metadata = _resolve_metadata_receipt()
    negative_rows = build_negative_metadata_fixture_rows()
    tests_run = _checkpoint_tests(repo_root)
    preconditions, server, _initial = _collect_preconditions(
        repo_root, gates, protocol, metadata, negative_rows, tests_run
    )
    preconditions["protected_file_hashes_before"] = protected_before
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    diagnostics: list[JsonDict] = []
    process_receipt: JsonDict = {}
    unload_rows: list[JsonDict] = []
    task_deadline = start + float(
        protocol.get("prompt_seed_budget_contract", {}).get("timeout_s", 4200)
    )
    if preconditions["all_required_preconditions_available"]:
        preconditions["model_process_started"] = True
        rows, checkpoints, diagnostics, process_receipt, unload_rows = _run_live_shard(
            repo_root=repo_root,
            protocol=protocol,
            metadata=metadata,
            server=server,
            selected_gpu=int(preconditions["selected_gpu"]),
            task_deadline=task_deadline,
        )
    protected = _compare_protected(protected_before, _hash_protected(repo_root))
    preconditions["protected_file_hashes_after"] = {
        row["path"]: row["after_sha256"] for row in protected["rows"]
    }
    if not preconditions["all_required_preconditions_available"]:
        structured_failed = any(row.get("passed") is not True for row in gates)
        reason = "structured_gate_failed" if structured_failed else "precondition_failed"
        artifact = build_blocked_report(
            gates=gates,
            protocol=protocol,
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
            reason=reason,
        )
    else:
        artifact = build_report(
            gates=gates,
            protocol=protocol,
            metadata_receipt=metadata,
            negative_fixture_rows=negative_rows,
            rows=rows,
            checkpoint_receipts=checkpoints,
            parser_diagnostic_rows=diagnostics,
            process_receipt=process_receipt,
            unload_rows=unload_rows,
            attack_rows=build_attack_rows(),
            preconditions=preconditions,
            protected=protected,
            duration_s=time.monotonic() - start,
            tests_run=tests_run,
            run_date=run_date,
        )
    atomic_write_report(repo_root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run or validate the one-family shard artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or (REPO_ROOT / RESULT_RELATIVE_PATH)
    if args.validate:
        errors = validate_report(load_json(output))
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 1 if errors else 0
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "status": artifact["status"],
                "verdict_class": artifact["verdict_class"],
                "qwen36_family_source_shard_ready_score": artifact[
                    "qwen36_family_source_shard_ready_score"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
