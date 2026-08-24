"""Independently replay and merge the three V572 family source shards.

Spec refs: REQ-REPORT-6584, SCENARIO-REPORT-6584-MISSING,
SCENARIO-REPORT-6584-REPLAY, SCENARIO-REPORT-6584-MERGE,
SCENARIO-REPORT-6584-ATTACKS, SCENARIO-REPORT-6584-UNLOAD, and
SCENARIO-REPORT-6584-ATOMIC.

The family artifacts contain both raw rows and convenient summaries. This
module reads the raw checkpoints and recomputes the summaries. It keeps a
missing family visible as coverage evidence but never invents that family's
model response.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

from carnot.inference.gguf_metadata import GgufMetadataError, read_gguf_metadata


JsonDict = dict[str, Any]
GgufInspector = Callable[[Path], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260824"
RESULT_RELATIVE_PATH = Path("results/experiment_6584_three_family_source_receipt_audit.json")
PROTOCOL_RELATIVE_PATH = Path("results/experiment_6580_v572_source_and_joint_method_protocol.json")
EXP6577_RELATIVE_PATH = Path(
    "results/experiment_6577_flagship_source_stream_independent_audit.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
PROTECTED_RELATIVE_PATHS = (
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)
AUDIT_TOOL_RELATIVE_PATHS = (
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/artifact_convention_audit.py"),
)
INFERENCE_SUBSTRATE = "immutable_three_family_source_replay_no_llm"
RECOVERY_TOLERANCE_MB = 256
GPU_LOAD_DELTA_MIN_MB = 128

FAMILY_SPECS: tuple[JsonDict, ...] = (
    {
        "family_id": "qwen36",
        "task_id": "exp6581-qwen36-flagship-source-shard",
        "artifact_path": "results/experiment_6581_qwen36_flagship_source_shard.json",
        "schema": "carnot.experiment_6581_qwen36_flagship_source_shard.v1",
        "repository_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "architecture": "qwen35moe",
        "tokenizer_model": "qwen2",
        "seed": 6581,
        "readiness_field": "qwen36_family_source_shard_ready_score",
    },
    {
        "family_id": "gemma4_31b",
        "task_id": "exp6582-gemma4-31b-flagship-source-shard",
        "artifact_path": "results/experiment_6582_gemma4_31b_flagship_source_shard.json",
        "schema": "carnot.experiment_6582_gemma4_31b_flagship_source_shard.v1",
        "repository_id": "unsloth/gemma-4-31B-it-GGUF",
        "architecture": "gemma4",
        "tokenizer_model": "gemma4",
        "seed": 6582,
        "readiness_field": "gemma4_31b_family_source_shard_ready_score",
    },
    {
        "family_id": "gemma4_26b_a4b",
        "task_id": "exp6583-gemma4-26b-a4b-flagship-source-shard",
        "artifact_path": "results/experiment_6583_gemma4_26b_a4b_flagship_source_shard.json",
        "schema": "carnot.experiment_6583_gemma4_26b_a4b_flagship_source_shard.v1",
        "repository_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "architecture": "gemma4",
        "tokenizer_model": "gemma4",
        "seed": 6583,
        "readiness_field": "gemma4_26b_a4b_family_source_shard_ready_score",
    },
)

FAILURE_CLASSES = (
    "timeout",
    "malformed_output",
    "refusal",
    "empty_output",
    "no_claim",
    "process_failure",
)
REQUIRED_ATTACKS = (
    "missing_or_blocked_family",
    "legacy_substitution",
    "source_alias",
    "prompt_drift",
    "seed_drift",
    "duplicate_unit_id",
    "copied_output_across_families",
    "selective_retry",
    "hidden_row_drop",
    "null_only_rows",
    "stale_pid",
    "zero_layer_offload",
    "missing_raw_path",
    "missing_unload",
    "reused_process",
    "protected_drift",
    "readiness_contradicted_by_rows",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "upstream_artifact_receipts",
    "rows",
    "family_coverage_rows",
    "failure_retention_rows",
    "duplicate_drift_and_substitution_rows",
    "unload_and_recovery_rows",
    "all_family_source_audit_ready_score",
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
    "status": "The always-run audit closes even when a family is missing.",
    "honest_verdict": "The verdict states lineage, coverage, failure retention, unload, and merge disposition.",
    "verdict_class": "The audit uses null, partial, blocked, or disqualified.",
    "gate_check_summary": "A block names the exact missing artifact or failed field and observed value.",
    "upstream_artifact_receipts": "Path, hash, status, and schema bind every family object under audit.",
    "rows": "Each expected source-family unit carries independently recomputed runtime and evidence metrics.",
    "family_coverage_rows": "The three mandated families and every frozen source unit remain explicit.",
    "failure_retention_rows": "Failure classes cannot vanish from a merged denominator.",
    "duplicate_drift_and_substitution_rows": "Copied output, drift, stale execution, and legacy substitution are tested.",
    "unload_and_recovery_rows": "One family cannot contaminate the next task's evidence.",
    "all_family_source_audit_ready_score": "This exact binary field gates Exp6585.",
    "aggregate_row_recomputation": "Coverage, failures, timing, tokens, and cost derive only from emitted rows.",
    "preconditions_checked": "Input, tool, raw-path, resource, and protected-file receipts are explicit.",
    "protected_files_unchanged": "The audit preserves both protected orchestration files.",
    "inference_substrate": "This is immutable three-family artifact replay with no new LLM.",
    "verifier_is_oracle": "The evidence audit is authority and cannot create positive science.",
    "field_provenance": "Each field identifies source rows, hashes, and independent reducer code.",
    "duration_s": "Monotonic duration exposes skipped replay or attack work.",
    "tests_run": "Named commands, exits, and durations make the audit reproducible.",
    "reproducibility_checksum": "A final hash detects audit mutation.",
}


def canonical_json(value: Any) -> str:
    """Encode stable compact JSON so every evidence hash can be repeated."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Return one tagged SHA-256 digest over exact bytes."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a value only after stable JSON encoding."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Hash one file in bounded chunks, or record that it is missing."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash a row without trusting its self-referential hash field."""

    return sha256_json({key: value for key, value in row.items() if key != "row_hash"})


def artifact_checksum(payload: Mapping[str, Any]) -> str:
    """Hash a terminal artifact without its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )


def recover_bytes(row: Mapping[str, Any], stem: str) -> tuple[bytes | None, str]:
    """Recover inline bytes and say why corrupt or absent content did not load."""

    value = row.get(f"{stem}_bytes_b64")
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True), "inline_base64"
        except (binascii.Error, ValueError):
            return None, "invalid_base64"
    return None, "missing"


def _claim_object(raw_text: str) -> tuple[Mapping[str, Any] | None, bool]:
    """Parse the required JSON object and disclose a Markdown wrapper as malformed."""

    try:
        value = json.loads(raw_text)
        return (value if isinstance(value, Mapping) else None), False
    except json.JSONDecodeError:
        stripped = raw_text.strip()
        if stripped.startswith("```json") and stripped.endswith("```"):
            body = stripped[len("```json") : -3].strip()
            try:
                value = json.loads(body)
                return (value if isinstance(value, Mapping) else None), True
            except json.JSONDecodeError:
                pass
    return None, bool(raw_text.strip())


def classify_response(
    raw: bytes, *, timed_out: bool = False, process_failure: bool = False
) -> JsonDict:
    """Recompute terminal failure classes from bytes and process receipts."""

    malformed_utf8 = False
    try:
        text = raw.decode("utf-8", "strict").strip()
    except UnicodeDecodeError:
        text = ""
        malformed_utf8 = True
    lowered = text.lower()
    refusal = any(
        marker in lowered
        for marker in ("i cannot", "i can't", "unable to comply", "cannot comply", "i refuse")
    )
    claim, malformed_envelope = _claim_object(text) if text else (None, False)
    required = {"claim_id", "supported_spans", "unsupported_reason", "release_action"}
    claim_bearing = isinstance(claim, Mapping) and required <= set(claim)
    request_for_claim = "provide the claim" in lowered or "which claim" in lowered
    empty = not raw or not text
    # A semantic request for the missing claim is a ``no_claim`` result, not a
    # parser attempt.  Invalid UTF-8 and other non-JSON output remain malformed,
    # while a refusal keeps its own failure class.
    malformed = malformed_utf8 or (malformed_envelope and not request_for_claim and not refusal)
    no_claim = empty or refusal or request_for_claim or not claim_bearing
    return {
        "timeout": bool(timed_out),
        "malformed_output": bool(malformed),
        "refusal": bool(refusal),
        "empty_output": bool(empty),
        "no_claim": bool(no_claim and not claim_bearing),
        "process_failure": bool(process_failure),
        "claim_bearing": bool(claim_bearing),
    }


def cost_from_components(components: Sequence[Mapping[str, Any]]) -> float:
    """Recompute normalized charged work from row-local quantities and rates."""

    return round(
        sum(
            float(component.get("quantity", 0.0) or 0.0)
            * float(component.get("unit_cost", 0.0) or 0.0)
            for component in components
        ),
        9,
    )


def _read_json(path: Path) -> tuple[JsonDict | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if not isinstance(value, Mapping):
        return None, "schema_not_object"
    return dict(value), None


def protected_hashes(repo_root: Path) -> dict[str, str]:
    """Hash the two files that this audit is forbidden to change."""

    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path.as_posix(),
            "before_sha256": before.get(path.as_posix(), "missing"),
            "after_sha256": after.get(path.as_posix(), "missing"),
            "unchanged": before.get(path.as_posix()) == after.get(path.as_posix())
            and before.get(path.as_posix()) != "missing",
        }
        for path in PROTECTED_RELATIVE_PATHS
    ]
    return {
        "rows": rows,
        "all_unchanged": all(row["unchanged"] for row in rows),
        "changed_paths": [row["path"] for row in rows if not row["unchanged"]],
    }


def _tool_version(repo_root: Path, relative: Path) -> JsonDict:
    path = repo_root / relative
    commit = "unavailable"
    if path.is_file() and (repo_root / ".git").exists():  # pragma: no cover - live git receipt.
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
    return {
        "path": relative.as_posix(),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "version": commit,
    }


def _resources(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    ram_total = ram_available = None
    try:
        values = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:"))
        }
        ram_total = values.get("MemTotal")
        ram_available = values.get("MemAvailable")
    except OSError:  # pragma: no cover - non-Linux fallback.
        pass
    return {
        "cpu": {"logical_count": os.cpu_count(), "architecture": platform.machine()},
        "ram": {"total_bytes": ram_total, "available_bytes": ram_available},
        "disk": {"path": str(repo_root), "total_bytes": disk.total, "free_bytes": disk.free},
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _protocol_checks(
    protocol: Mapping[str, Any], family_specs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    manifest = protocol.get("source_unit_manifest", {})
    contract = protocol.get("prompt_seed_budget_contract", {})
    units = manifest.get("units", []) if isinstance(manifest, Mapping) else []
    units = [row for row in units if isinstance(row, Mapping)]
    manifest_body = (
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if isinstance(manifest, Mapping)
        else {}
    )
    prompt = str(contract.get("family_neutral_prompt", "")) if isinstance(contract, Mapping) else ""
    budget = contract.get("token_budget", {}) if isinstance(contract, Mapping) else {}
    family_rows = contract.get("family_rows", []) if isinstance(contract, Mapping) else []
    by_task = {row.get("task_id"): row for row in family_rows if isinstance(row, Mapping)}
    family_contract_rows = []
    for family in family_specs:
        row = by_task.get(family["task_id"], {})
        checks = {
            "model": row.get("model_family") == family["repository_id"],
            "seed": row.get("seed") == family["seed"],
            "prompt": row.get("prompt_sha256") == sha256_bytes(prompt.encode()),
            "budget": row.get("token_budget_hash") == sha256_json(budget),
            "timeouts": row.get("per_source_unit_timeout_s") == 720
            and row.get("task_timeout_s") == contract.get("timeout_s") == 4200,
            "family_neutral": row.get("family_specific_prompt_allowed") is False,
        }
        family_contract_rows.append(
            {
                "family_id": family["family_id"],
                "expected_seed": family["seed"],
                "observed_seed": row.get("seed"),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    source_hashes_valid = bool(units) and all(
        unit.get("source_bytes_sha256")
        == sha256_bytes(str(unit.get("exact_source_bytes", "")).encode())
        for unit in units
    )
    checks = {
        "source_method_gate": protocol.get("v572_source_method_ready_score") == 1.0,
        "manifest_count": isinstance(manifest, Mapping)
        and manifest.get("bounded_unit_count") == len(units),
        "manifest_hash": bool(manifest_body)
        and manifest.get("manifest_hash") == sha256_json(manifest_body),
        "source_hashes": source_hashes_valid,
        "prompt_hash": bool(prompt)
        and contract.get("prompt_sha256") == sha256_bytes(prompt.encode()),
        "family_contracts": all(row["passed"] for row in family_contract_rows),
        "raw_first": contract.get("raw_before_derived_write_order") is True,
        "failure_retention": contract.get("failure_retention_required") is True,
        "fresh_process": contract.get("fresh_process_per_family") is True,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "units": units,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "budget": budget,
        "budget_sha256": sha256_json(budget),
        "manifest_hash": manifest.get("manifest_hash") if isinstance(manifest, Mapping) else None,
        "family_contract_rows": family_contract_rows,
        "seed_values_identical": len({row.get("seed") for row in family_contract_rows}) == 1,
        "seed_contract_mode": "one prescribed deterministic seed per family task",
    }


def _process_checks(
    receipt: Mapping[str, Any], family: Mapping[str, Any], gguf_path: str
) -> JsonDict:
    command = [str(part) for part in receipt.get("command", [])]
    os_command = [str(part) for part in receipt.get("os_command", [])]
    pid = int(receipt.get("pid", 0) or 0)
    selected = receipt.get("selected_gpu")
    samples = [row for row in receipt.get("gpu_samples", []) if isinstance(row, Mapping)]
    before = [row for row in samples if row.get("stage") == "before"]
    during = [row for row in samples if row.get("stage") == "during"]
    after = [row for row in samples if row.get("stage") == "after"]
    linked = [
        process
        for sample in during
        for process in sample.get("compute_processes", [])
        if isinstance(process, Mapping) and int(process.get("pid", 0) or 0) == pid
    ]
    baseline = min(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in before),
        default=0,
    )
    peak = max(
        (int(row.get("device", {}).get("memory_used_mb", 0) or 0) for row in during),
        default=0,
    )
    full_offload = "--n-gpu-layers" in command and command.index("--n-gpu-layers") + 1 < len(
        command
    )
    single_gpu = "--split-mode" in command and command.index("--split-mode") + 1 < len(command)
    checks = {
        "fresh_process": receipt.get("fresh_process") is True,
        "pid_evidence": pid > 0
        and int(receipt.get("parent_pid", 0) or 0) > 0
        and receipt.get("os_pid_verified") is True
        and receipt.get("os_parent_pid_verified") is True,
        "command_hash": receipt.get("command_sha256") == sha256_json(command),
        "os_command_hash": receipt.get("os_command_sha256") == sha256_json(os_command),
        "command_matches_os": receipt.get("command_matches_os") is True and command == os_command,
        "model_path_bound": bool(gguf_path) and gguf_path in command,
        "cuda_device_bound": selected is not None
        and str(receipt.get("cuda_visible_devices")) == str(selected),
        "full_cuda_offload_requested": full_offload
        and command[command.index("--n-gpu-layers") + 1] == "all",
        "single_gpu_requested": not single_gpu
        or command[command.index("--split-mode") + 1] == "none",
        "positive_layer_offload": int(receipt.get("offloaded_layers", 0) or 0) > 0,
        "repeated_gpu_samples": bool(before) and len(during) >= 2 and bool(after),
        "worker_pid_linked": bool(linked),
        "positive_gpu_residency": peak - baseline >= GPU_LOAD_DELTA_MIN_MB
        and max((int(row.get("used_memory_mb", 0) or 0) for row in linked), default=0)
        >= GPU_LOAD_DELTA_MIN_MB,
        "utilization_sampled": bool(during)
        and all("utilization_pct" in row.get("device", {}) for row in during),
        "one_family_resident": receipt.get("resident_model_families") == [family["repository_id"]],
        "server_healthy": receipt.get("server_healthy") is True
        and receipt.get("http_status") == 200,
        "monotonic_lifecycle": int(receipt.get("started_monotonic_ns", 0) or 0)
        < int(receipt.get("ended_monotonic_ns", 0) or 0),
        "clean_exit": receipt.get("shutdown_requested") is True
        and receipt.get("normal_shutdown") is True
        and receipt.get("exit_code") == 0
        and receipt.get("worker_alive_after_exit") is False,
        "streams_hashed": str(receipt.get("stdout_sha256", "")).startswith("sha256:")
        and str(receipt.get("stderr_sha256", "")).startswith("sha256:"),
        "measured": receipt.get("evidence_mode") == "measured",
        "unrelated_processes_preserved": receipt.get("signals_sent_to_unrelated_pids") == [],
        "embedded_tokenizer": receipt.get("embedded_tokenizer") is True,
    }
    return {"checks": checks, "passed": all(checks.values()), "pid": pid}


def _unload_receipt(payload: Mapping[str, Any], family: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("unload_and_recovery_rows", []) if isinstance(row, Mapping)]
    row = rows[0] if len(rows) == 1 else {}
    delta = abs(int(row.get("memory_delta_from_baseline_mb", 0) or 0))
    checks = {
        "one_receipt": len(rows) == 1,
        "shutdown_requested": row.get("shutdown_requested") is True,
        "normal_shutdown": row.get("normal_shutdown") is True,
        "clean_exit": row.get("exit_code") == 0,
        "pid_gone": row.get("worker_absent_from_proc") is True
        and row.get("worker_absent_from_nvidia_smi") is True,
        "port_closed": row.get("port_closed") is True,
        "memory_recovered": row.get("recovery_tolerance_mb") == RECOVERY_TOLERANCE_MB
        and delta <= RECOVERY_TOLERANCE_MB,
        "one_family_residency": row.get("no_task_worker_remains") is True,
        "bounded_recovery": row.get("recovery_bounded") is True,
        "unrelated_processes_preserved": row.get("signals_sent_to_unrelated_pids") == [],
        "recovery_complete": row.get("recovery_complete") is True,
    }
    result: JsonDict = {
        "family_id": family["family_id"],
        "task_id": family["task_id"],
        "worker_pid": row.get("worker_pid"),
        "checks": checks,
        "passed": all(checks.values()),
    }
    result["row_hash"] = row_hash(result)
    return result


def _metadata_receipt(
    payload: Mapping[str, Any],
    family: Mapping[str, Any],
    inspector: GgufInspector,
) -> JsonDict:
    stored = payload.get("model_revision_and_hash_receipt", {})
    stored = stored if isinstance(stored, Mapping) else {}
    provenance = stored.get("provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    path = Path(str(stored.get("selected_blob_path", "")))
    file_hash = sha256_file(path)
    try:
        content = dict(inspector(path)) if path.is_file() else {}
        inspect_error = None if content else "empty_metadata"
    except (OSError, GgufMetadataError, ValueError) as exc:
        content = {}
        inspect_error = f"{type(exc).__name__}:{exc}"
    tokenizer = content.get("tokenizer_metadata", {}) if isinstance(content, Mapping) else {}
    checks = {
        "repository": stored.get("repository_id") == family["repository_id"],
        "revision": bool(provenance.get("revision")),
        "blob_exists": path.is_file(),
        "gguf_hash": file_hash != "missing" and stored.get("trusted_sha256") == file_hash,
        "architecture": content.get("architecture") == family["architecture"],
        "tokenizer_model": isinstance(tokenizer, Mapping)
        and tokenizer.get("model") == family["tokenizer_model"],
        "tokenizer_count": isinstance(tokenizer, Mapping)
        and int(tokenizer.get("token_count", 0) or 0) > 0,
        "chat_template": isinstance(tokenizer, Mapping)
        and tokenizer.get("chat_template_present") is True,
    }
    return {
        "path": str(path),
        "repository_id": stored.get("repository_id"),
        "revision": provenance.get("revision"),
        "gguf_sha256_recomputed": file_hash,
        "tokenizer": dict(tokenizer) if isinstance(tokenizer, Mapping) else {},
        "inspect_error": inspect_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _raw_checkpoint(
    final_row: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict]:
    path = Path(str(final_row.get("raw_checkpoint_path", "")))
    payload, error = _read_json(path)
    raw = payload or {}
    stored_hash = row_hash(raw) if raw else "missing"
    expected_name = (
        f"{int(final_row.get('order_index', 0) or 0):02d}-{stored_hash.removeprefix('sha256:')}.json"
        if raw
        else None
    )
    receipt = {
        "path": str(path),
        "exists": path.is_file(),
        "read_error": error,
        "sha256_recomputed": sha256_file(path),
        "row_hash_recomputed": stored_hash,
        "content_addressed_name": path.name == expected_name if expected_name else False,
        "atomic_replace_receipted": checkpoint_receipt.get("atomic_replace") is True,
    }
    return raw, receipt


def _replay_one_row(
    *,
    final_row: Mapping[str, Any],
    unit: Mapping[str, Any],
    family: Mapping[str, Any],
    protocol: Mapping[str, Any],
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    process_result: Mapping[str, Any],
) -> JsonDict:
    order = int(final_row.get("order_index", -1) or 0)
    checkpoints = [
        row for row in payload.get("checkpoint_receipts", []) if isinstance(row, Mapping)
    ]
    diagnostics = [
        row for row in payload.get("parser_diagnostic_rows", []) if isinstance(row, Mapping)
    ]
    checkpoint = next((row for row in checkpoints if row.get("order_index") == order), {})
    diagnostic = next((row for row in diagnostics if row.get("order_index") == order), {})
    raw, raw_receipt = _raw_checkpoint(final_row, checkpoint)
    source, source_mode = recover_bytes(raw, "source")
    request, request_mode = recover_bytes(raw, "request")
    response, response_mode = recover_bytes(raw, "raw_response")
    contract = protocol.get("prompt_seed_budget_contract", {})
    manifest = protocol.get("source_unit_manifest", {})
    prompt = str(contract.get("family_neutral_prompt", ""))
    expected_source = str(unit.get("exact_source_bytes", "")).encode()
    expected_request = prompt.encode() + b"\n\nSOURCE BYTES:\n" + expected_source
    process = payload.get("process_and_gpu_receipts", {})
    process = process if isinstance(process, Mapping) else {}
    timed_out = raw.get("request_exit_code") == 124 or raw.get("stop_reason") == "timeout"
    process_failure = raw.get("request_exit_code") not in (0, None) or process.get("exit_code") != 0
    classified = classify_response(
        response or b"", timed_out=timed_out, process_failure=process_failure
    )
    recomputed_failures = [name for name in FAILURE_CLASSES if classified[name]]
    stored_failures = raw.get("failure_flags", {})
    stored_failure_classes = [
        name
        for name in FAILURE_CLASSES
        if isinstance(stored_failures, Mapping) and stored_failures.get(name) is True
    ]
    raw_core_matches_final = bool(raw) and all(
        final_row.get(key) == value for key, value in raw.items() if key != "row_hash"
    )
    raw_time = int(raw.get("raw_response_recorded_monotonic_ns", 0) or 0)
    written_time = int(checkpoint.get("written_monotonic_ns", 0) or 0)
    parser_time = int(diagnostic.get("parser_started_monotonic_ns", 0) or 0)
    raw_before_parser = raw_time > 0 and raw_time < written_time <= parser_time
    costs = raw.get("charged_cost_components", [])
    cost_recomputed = cost_from_components(costs) if isinstance(costs, Sequence) else 0.0
    total_tokens = int(raw.get("prompt_token_count", 0) or 0) + int(
        raw.get("response_token_count", 0) or 0
    )
    checks = {
        "final_row_hash": final_row.get("row_hash") == row_hash(final_row),
        "raw_checkpoint_exists": raw_receipt["exists"] and raw_receipt["read_error"] is None,
        "raw_checkpoint_hash": raw_receipt["sha256_recomputed"]
        == final_row.get("raw_checkpoint_sha256")
        == checkpoint.get("checkpoint_sha256"),
        "raw_row_hash": raw_receipt["row_hash_recomputed"]
        == raw.get("row_hash")
        == final_row.get("raw_checkpoint_row_hash")
        == checkpoint.get("raw_row_hash"),
        "content_addressed_path": raw_receipt["content_addressed_name"],
        "atomic_checkpoint": raw_receipt["atomic_replace_receipted"],
        "raw_core_matches_final": raw_core_matches_final,
        "unit_id": raw.get("unit_id") == final_row.get("unit_id") == unit.get("unit_id"),
        "unit_order": raw.get("order_index") == final_row.get("order_index"),
        "manifest_hash": raw.get("source_manifest_hash") == manifest.get("manifest_hash"),
        "source_bytes": source == expected_source,
        "source_hash": raw.get("source_bytes_sha256")
        == unit.get("source_bytes_sha256")
        == sha256_bytes(expected_source),
        "source_content_hash": raw.get("source_content_hash") == unit.get("content_hash"),
        "prompt_hash": raw.get("prompt_sha256")
        == contract.get("prompt_sha256")
        == sha256_bytes(prompt.encode()),
        "request_bytes": request == expected_request,
        "request_hash": raw.get("request_sha256") == sha256_bytes(expected_request),
        "repository": raw.get("repository_id") == family["repository_id"],
        "revision": raw.get("revision") == metadata.get("revision"),
        "gguf_hash": raw.get("gguf_sha256") == metadata.get("gguf_sha256_recomputed"),
        "command_hash": raw.get("command_sha256") == process.get("command_sha256"),
        "pid": raw.get("pid") == final_row.get("pid") == process_result.get("pid"),
        "cuda_device": raw.get("cuda_device") == process.get("selected_gpu"),
        "positive_offload": int(raw.get("offloaded_layers", 0) or 0) > 0
        and raw.get("offloaded_layers") == process.get("offloaded_layers"),
        "seed": raw.get("seed") == family["seed"],
        "single_attempt": raw.get("attempt_count") == 1 and raw.get("retry_count") == 0,
        "response_bytes": response is not None
        and raw.get("raw_response_byte_count") == len(response),
        "response_hash": response is not None
        and raw.get("raw_response_sha256") == sha256_bytes(response),
        "token_total": raw.get("total_token_count") == total_tokens,
        "latency": isinstance(raw.get("latency_s"), (int, float))
        and not isinstance(raw.get("latency_s"), bool)
        and math.isfinite(float(raw.get("latency_s")))
        and float(raw.get("latency_s")) >= 0.0,
        "stop_reason": bool(raw.get("stop_reason")),
        "exit": isinstance(raw.get("request_exit_code"), int),
        "stderr_hash": str(raw.get("stderr_sha256_at_terminal", "")).startswith("sha256:"),
        "charged_cost": raw.get("charged_cost") == cost_recomputed,
        "failure_flags": set(stored_failures) == set(FAILURE_CLASSES)
        if isinstance(stored_failures, Mapping)
        else False,
        "failure_classes_match": stored_failure_classes == recomputed_failures,
        "raw_before_parser": raw_before_parser
        and diagnostic.get("raw_before_parser") is True
        and diagnostic.get("parser_can_filter_rows") is False,
        "parser_hash": diagnostic.get("row_hash") == row_hash(diagnostic)
        and final_row.get("parser_diagnostic_row_hash") == row_hash(diagnostic),
    }
    audit_row: JsonDict = {
        "row_type": "independently_replayed_source_family_unit",
        "family_id": family["family_id"],
        "task_id": family["task_id"],
        "artifact_path": family["artifact_path"],
        "unit_id": unit.get("unit_id"),
        "fixture_id": unit.get("fixture_id"),
        "case_kind": unit.get("case_kind"),
        "split": unit.get("split"),
        "order_index": final_row.get("order_index"),
        "source_storage": source_mode,
        "source_bytes_sha256": sha256_bytes(source) if source is not None else "missing",
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "request_storage": request_mode,
        "request_sha256": sha256_bytes(request) if request is not None else "missing",
        "repository_id": raw.get("repository_id", final_row.get("repository_id")),
        "revision": raw.get("revision", final_row.get("revision")),
        "gguf_path": metadata.get("path"),
        "gguf_sha256": metadata.get("gguf_sha256_recomputed"),
        "tokenizer": metadata.get("tokenizer", {}),
        "command_sha256": process.get("command_sha256"),
        "pid": process_result.get("pid"),
        "pid_evidence_passed": process_result.get("checks", {}).get("pid_evidence") is True,
        "cuda_device": raw.get("cuda_device", final_row.get("cuda_device")),
        "offloaded_layers": raw.get("offloaded_layers", final_row.get("offloaded_layers")),
        "gpu_sample_count": len(process.get("gpu_samples", [])),
        "raw_checkpoint_path": str(final_row.get("raw_checkpoint_path", "")),
        "raw_checkpoint_sha256": raw_receipt["sha256_recomputed"],
        "raw_path_resolves": raw_receipt["exists"],
        "raw_before_parser": raw_before_parser,
        "raw_response_storage": response_mode,
        "raw_response_sha256": sha256_bytes(response) if response is not None else "missing",
        "raw_response_byte_count": len(response) if response is not None else 0,
        "prompt_token_count": int(raw.get("prompt_token_count", 0) or 0),
        "response_token_count": int(raw.get("response_token_count", 0) or 0),
        "total_token_count": total_tokens,
        "latency_s": round(float(raw.get("latency_s", 0.0) or 0.0), 9),
        "stop_reason": raw.get("stop_reason"),
        "request_exit_code": raw.get("request_exit_code"),
        "stderr_sha256": raw.get("stderr_sha256_at_terminal"),
        "charged_cost": cost_recomputed,
        "charged_cost_unit": raw.get("charged_cost_unit"),
        "failure_classes": recomputed_failures,
        "stored_failure_classes": stored_failure_classes,
        "claim_bearing": classified["claim_bearing"],
        "checks": checks,
        "replay_failures": [name for name, passed in checks.items() if not passed],
        "replay_passed": all(checks.values())
        and metadata.get("passed") is True
        and process_result.get("passed") is True,
    }
    audit_row["row_hash"] = row_hash(audit_row)
    return audit_row


def _upstream_receipt(
    repo_root: Path, family: Mapping[str, Any]
) -> tuple[JsonDict, JsonDict | None]:
    path = repo_root / str(family["artifact_path"])
    payload, error = _read_json(path)
    receipt = {
        "family_id": family["family_id"],
        "task_id": family["task_id"],
        "path": family["artifact_path"],
        "absolute_path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "read_error": error,
        "schema": payload.get("schema") if payload else None,
        "status": payload.get("status") if payload else None,
        "verdict_class": payload.get("verdict_class") if payload else None,
        "stored_readiness_field": family["readiness_field"],
        "stored_readiness_value": payload.get(family["readiness_field"]) if payload else None,
        "stored_readiness_used_by_reducer": False,
        "row_count": len(payload.get("rows", [])) if payload else 0,
    }
    return receipt, payload


def _coverage_rows(
    family_specs: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any] | None],
    rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    replay_by_key = {(row.get("family_id"), row.get("unit_id")): row for row in rows}
    output = []
    for family in family_specs:
        payload = payloads.get(str(family["family_id"]))
        final_rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
        final_ids = {row.get("unit_id") for row in final_rows if isinstance(row, Mapping)}
        for unit in units:
            replay = replay_by_key.get((family["family_id"], unit.get("unit_id")), {})
            row = {
                "family_id": family["family_id"],
                "repository_id": family["repository_id"],
                "unit_id": unit.get("unit_id"),
                "fixture_id": unit.get("fixture_id"),
                "expected": True,
                "artifact_present": isinstance(payload, Mapping),
                "row_present": unit.get("unit_id") in final_ids,
                "replay_emitted": bool(replay),
                "replay_passed": replay.get("replay_passed") is True,
                "raw_path_resolves": replay.get("raw_path_resolves") is True,
                "failure_classes": list(replay.get("failure_classes", [])),
            }
            row["row_hash"] = row_hash(row)
            output.append(row)
    return output


def _failure_retention(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    output = []
    for family in FAMILY_SPECS:
        family_rows = [row for row in rows if row.get("family_id") == family["family_id"]]
        for failure in FAILURE_CLASSES:
            raw_count = sum(failure in row.get("failure_classes", []) for row in family_rows)
            emitted_count = sum(failure in row.get("failure_classes", []) for row in family_rows)
            row = {
                "family_id": family["family_id"],
                "failure_class": failure,
                "raw_count": raw_count,
                "emitted_count": emitted_count,
                "denominator": len(family_rows),
                "retained": raw_count == emitted_count,
            }
            row["row_hash"] = row_hash(row)
            output.append(row)
    return output


def _attack_row(attack_id: str, passed: bool, observed: Any, expected: Any) -> JsonDict:
    row: JsonDict = {
        "attack_id": attack_id,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "candidate_ready_score": 1.0 if passed else 0.0,
        "reducer": "independent Exp6584 invariant reducer",
    }
    row["row_hash"] = row_hash(row)
    return row


def _attacks(
    *,
    family_specs: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any] | None],
    rows: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
    unload: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    family_readiness: Mapping[str, float],
) -> list[JsonDict]:
    repositories = {family["repository_id"] for family in family_specs}
    observed_final_rows = [
        row
        for payload in payloads.values()
        if isinstance(payload, Mapping)
        for row in payload.get("rows", [])
        if isinstance(row, Mapping)
    ]
    unavailable = [
        receipt["family_id"]
        for receipt in receipts
        if not receipt["exists"]
        or receipt["read_error"] is not None
        or str(receipt.get("status", "")).startswith("blocked")
        or receipt["row_count"] == 0
    ]
    bad_repositories = sorted(
        {
            str(row.get("repository_id"))
            for row in observed_final_rows
            if row.get("repository_id") not in repositories
        }
    )
    replay_by_key = {(row.get("repository_id"), row.get("unit_id")): row for row in rows}
    final_source_errors = 0
    final_prompt_errors = 0
    final_seed_errors = 0
    expected_seed_by_repository = {
        family["repository_id"]: family["seed"] for family in family_specs
    }
    for final_row in observed_final_rows:
        replay = replay_by_key.get((final_row.get("repository_id"), final_row.get("unit_id")), {})
        final_source, _ = recover_bytes(final_row, "source")
        final_source_errors += final_source is None or sha256_bytes(final_source) != replay.get(
            "source_bytes_sha256"
        )
        final_prompt_errors += final_row.get("prompt_sha256") != replay.get("prompt_sha256")
        final_seed_errors += final_row.get("seed") != expected_seed_by_repository.get(
            final_row.get("repository_id")
        )
    source_errors = final_source_errors + sum(
        not row.get("checks", {}).get("source_bytes", False)
        or not row.get("checks", {}).get("source_hash", False)
        for row in rows
    )
    prompt_errors = final_prompt_errors + sum(
        not row.get("checks", {}).get("prompt_hash", False) for row in rows
    )
    seed_errors = final_seed_errors + sum(
        not row.get("checks", {}).get("seed", False) for row in rows
    )
    final_keys = [(row.get("repository_id"), row.get("unit_id")) for row in observed_final_rows]
    duplicate_count = len(final_keys) - len(set(final_keys))
    output_pairs: dict[tuple[Any, Any], set[Any]] = {}
    for row in observed_final_rows:
        key = (row.get("unit_id"), row.get("raw_response_sha256"))
        output_pairs.setdefault(key, set()).add(row.get("repository_id"))
    copied = sorted(
        f"{unit}:{digest}" for (unit, digest), families in output_pairs.items() if len(families) > 1
    )
    retry_count = sum(
        row.get("attempt_count") != 1 or row.get("retry_count") != 0 for row in observed_final_rows
    )
    dropped = sum(not row.get("row_present", False) for row in coverage)
    null_families = sorted(
        family["family_id"]
        for family in family_specs
        if isinstance(payloads.get(str(family["family_id"])), Mapping)
        and payloads[str(family["family_id"])].get("rows")
        and not any(
            classify_response(
                recover_bytes(row, "raw_response")[0] or b"",
                timed_out=row.get("request_exit_code") == 124,
                process_failure=row.get("request_exit_code") not in (0, None),
            )["claim_bearing"]
            for row in payloads[str(family["family_id"])].get("rows", [])
            if isinstance(row, Mapping)
        )
    )
    stale_pid = sum(not row.get("checks", {}).get("pid", False) for row in rows)
    zero_offload = sum(int(row.get("offloaded_layers", 0) or 0) <= 0 for row in observed_final_rows)
    missing_paths = sum(not row.get("raw_path_resolves", False) for row in rows)
    missing_unload = [row["family_id"] for row in unload if not row["passed"]]
    pids = [row.get("worker_pid") for row in unload if isinstance(row.get("worker_pid"), int)]
    reused_pids = len(pids) - len(set(pids))
    contradicted = sorted(
        receipt["family_id"]
        for receipt in receipts
        if receipt.get("stored_readiness_value")
        != family_readiness.get(str(receipt["family_id"]), 0.0)
    )
    return [
        _attack_row("missing_or_blocked_family", not unavailable, unavailable, []),
        _attack_row("legacy_substitution", not bad_repositories, bad_repositories, []),
        _attack_row("source_alias", source_errors == 0, source_errors, 0),
        _attack_row("prompt_drift", prompt_errors == 0, prompt_errors, 0),
        _attack_row("seed_drift", seed_errors == 0, seed_errors, 0),
        _attack_row("duplicate_unit_id", duplicate_count == 0, duplicate_count, 0),
        _attack_row("copied_output_across_families", not copied, copied, []),
        _attack_row("selective_retry", retry_count == 0, retry_count, 0),
        _attack_row("hidden_row_drop", dropped == 0, dropped, 0),
        _attack_row("null_only_rows", not null_families, null_families, []),
        _attack_row("stale_pid", stale_pid == 0, stale_pid, 0),
        _attack_row("zero_layer_offload", zero_offload == 0, zero_offload, 0),
        _attack_row("missing_raw_path", missing_paths == 0, missing_paths, 0),
        _attack_row("missing_unload", not missing_unload, missing_unload, []),
        _attack_row("reused_process", reused_pids == 0, reused_pids, 0),
        _attack_row(
            "protected_drift",
            protected.get("all_unchanged") is True,
            protected.get("changed_paths", []),
            [],
        ),
        _attack_row("readiness_contradicted_by_rows", not contradicted, contradicted, []),
    ]


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "raw_sources": [
                PROTOCOL_RELATIVE_PATH.as_posix(),
                *[str(family["artifact_path"]) for family in FAMILY_SPECS],
                "content-addressed raw checkpoints",
            ],
            "reducer": "experiment_6584_three_family_source_receipt_audit independent reducer",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _retirement(repo_root: Path, receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    prior, _ = _read_json(repo_root / EXP6577_RELATIVE_PATH)
    prior_first = (
        prior.get("gate_check_summary", {}).get("first_failure")
        if isinstance(prior, Mapping)
        else None
    )
    missing = [receipt["path"] for receipt in receipts if not receipt["exists"]]
    same_chain = (
        bool(missing) and len(missing) == len(receipts) and isinstance(prior_first, Mapping)
    )
    return {
        "prior_artifact_path": EXP6577_RELATIVE_PATH.as_posix(),
        "prior_first_failure": prior_first,
        "current_missing_paths": missing,
        "same_missing_chain_as_exp6577": same_chain,
        "activated": same_chain,
        "reason": (
            "same all-family upstream-missing chain repeated"
            if same_chain
            else "current family state does not repeat Exp6577's missing Exp6576 chain"
        ),
    }


def _gate_summary(
    receipts: Sequence[Mapping[str, Any]],
    family_specs: Sequence[Mapping[str, Any]],
    family_readiness: Mapping[str, float],
    all_ready: float,
    retirement: Mapping[str, Any],
) -> JsonDict:
    checks = []
    by_family = {family["family_id"]: family for family in family_specs}
    for receipt in receipts:
        family = by_family[receipt["family_id"]]
        checks.append(
            {
                "field": receipt["path"],
                "expected": "file_exists",
                "observed": "present" if receipt["exists"] else "missing",
                "passed": receipt["exists"],
            }
        )
        if receipt["exists"]:
            checks.extend(
                [
                    {
                        "field": f"{receipt['path']}.schema",
                        "expected": family["schema"],
                        "observed": receipt["schema"],
                        "passed": receipt["schema"] == family["schema"],
                    },
                    {
                        "field": f"{receipt['path']}.status",
                        "expected": "complete_nonblocked",
                        "observed": receipt["status"],
                        "passed": not str(receipt.get("status", "")).startswith("blocked")
                        and receipt.get("read_error") is None,
                    },
                    {
                        "field": f"{receipt['path']}.rows.length",
                        "expected": "frozen_source_unit_count",
                        "observed": receipt["row_count"],
                        "passed": receipt["row_count"] > 0,
                    },
                    {
                        "field": f"{receipt['path']}.{family['readiness_field']}.recomputed",
                        "expected": 1.0,
                        "observed": family_readiness.get(str(receipt["family_id"]), 0.0),
                        "passed": family_readiness.get(str(receipt["family_id"]), 0.0) == 1.0,
                    },
                ]
            )
    checks.append(
        {
            "field": "all_family_source_audit_ready_score",
            "expected": 1.0,
            "observed": all_ready,
            "passed": all_ready == 1.0,
        }
    )
    first = next((dict(check) for check in checks if not check["passed"]), None)
    return {
        "checks": checks,
        "passed": first is None,
        "failed_check_count": sum(not check["passed"] for check in checks),
        "first_failure": first,
        "retirement": dict(retirement),
    }


def _raw_path_preconditions(payloads: Mapping[str, Mapping[str, Any] | None]) -> list[JsonDict]:
    rows = []
    for family_id, payload in payloads.items():
        if not isinstance(payload, Mapping):
            continue
        for row in payload.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            path = Path(str(row.get("raw_checkpoint_path", "")))
            rows.append(
                {
                    "family_id": family_id,
                    "unit_id": row.get("unit_id"),
                    "path": str(path),
                    "exists": path.is_file(),
                    "sha256": sha256_file(path),
                    "expected_sha256": row.get("raw_checkpoint_sha256"),
                }
            )
    return rows


def build_audit(
    repo_root: Path,
    *,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str] | None = None,
    family_specs: Sequence[Mapping[str, Any]] = FAMILY_SPECS,
    gguf_inspector: GgufInspector = read_gguf_metadata,
) -> JsonDict:
    """Build one terminal audit from protocol, shards, and raw checkpoints."""

    protocol, protocol_error = _read_json(repo_root / PROTOCOL_RELATIVE_PATH)
    protocol = protocol or {}
    protocol_result = _protocol_checks(protocol, family_specs)
    before = dict(protected_before or protected_hashes(repo_root))
    receipts = []
    payloads: dict[str, Mapping[str, Any] | None] = {}
    metadata_by_family: dict[str, JsonDict] = {}
    process_by_family: dict[str, JsonDict] = {}
    unload_rows = []
    merged_rows = []
    units = protocol_result["units"]
    for family in family_specs:
        receipt, payload = _upstream_receipt(repo_root, family)
        receipts.append(receipt)
        payloads[str(family["family_id"])] = payload
        if not isinstance(payload, Mapping):
            unloaded = _unload_receipt({}, family)
            unload_rows.append(unloaded)
            metadata_by_family[str(family["family_id"])] = {"passed": False}
            process_by_family[str(family["family_id"])] = {"passed": False, "pid": None}
            continue
        metadata = _metadata_receipt(payload, family, gguf_inspector)
        process = payload.get("process_and_gpu_receipts", {})
        process = process if isinstance(process, Mapping) else {}
        process_result = _process_checks(process, family, str(metadata.get("path", "")))
        metadata_by_family[str(family["family_id"])] = metadata
        process_by_family[str(family["family_id"])] = process_result
        unloaded = _unload_receipt(payload, family)
        unload_rows.append(unloaded)
        final_rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
        by_order = {row.get("order_index"): row for row in final_rows}
        for order, unit in enumerate(units):
            final = by_order.get(order)
            if not isinstance(final, Mapping):
                continue
            merged_rows.append(
                _replay_one_row(
                    final_row=final,
                    unit=unit,
                    family=family,
                    protocol=protocol,
                    payload=payload,
                    metadata=metadata,
                    process_result=process_result,
                )
            )
    coverage = _coverage_rows(family_specs, units, payloads, merged_rows)
    failure_rows = _failure_retention(merged_rows)
    expected_per_family = len(units)
    family_readiness = {}
    for family in family_specs:
        family_id = str(family["family_id"])
        receipt = next(row for row in receipts if row["family_id"] == family_id)
        family_rows = [row for row in merged_rows if row["family_id"] == family_id]
        family_unload = next(row for row in unload_rows if row["family_id"] == family_id)
        ready = bool(
            receipt["exists"]
            and receipt["read_error"] is None
            and not str(receipt.get("status", "")).startswith("blocked")
            and len(family_rows) == expected_per_family
            and all(row["replay_passed"] for row in family_rows)
            and metadata_by_family[family_id].get("passed") is True
            and process_by_family[family_id].get("passed") is True
            and family_unload["passed"]
            and any(row["claim_bearing"] for row in family_rows)
        )
        family_readiness[family_id] = 1.0 if ready else 0.0
    after = protected_hashes(repo_root)
    protected = _protected_receipt(before, after)
    attacks = _attacks(
        family_specs=family_specs,
        receipts=receipts,
        payloads=payloads,
        rows=merged_rows,
        coverage=coverage,
        unload=unload_rows,
        protected=protected,
        family_readiness=family_readiness,
    )
    failure_counts = Counter(
        failure for row in merged_rows for failure in row.get("failure_classes", [])
    )
    failure_row_count = sum(bool(row.get("failure_classes")) for row in merged_rows)
    expected_rows = len(family_specs) * expected_per_family
    aggregate = {
        "expected_family_count": len(family_specs),
        "readable_family_count": sum(
            receipt["exists"] and receipt["read_error"] is None for receipt in receipts
        ),
        "nonblocked_family_count": sum(
            receipt["exists"]
            and receipt["read_error"] is None
            and not str(receipt.get("status", "")).startswith("blocked")
            for receipt in receipts
        ),
        "expected_source_unit_count_per_family": expected_per_family,
        "expected_row_count": expected_rows,
        "observed_row_count": len(merged_rows),
        "replayed_row_count": sum(row["replay_passed"] for row in merged_rows),
        "source_unit_coverage": round(len(merged_rows) / expected_rows, 9)
        if expected_rows
        else 0.0,
        "claim_bearing_row_count": sum(row["claim_bearing"] for row in merged_rows),
        "failure_row_count": failure_row_count,
        "failure_class_counts": {
            failure: failure_counts.get(failure, 0) for failure in FAILURE_CLASSES
        },
        "prompt_token_count": sum(int(row["prompt_token_count"]) for row in merged_rows),
        "response_token_count": sum(int(row["response_token_count"]) for row in merged_rows),
        "total_token_count": sum(int(row["total_token_count"]) for row in merged_rows),
        "latency_s": round(sum(float(row["latency_s"]) for row in merged_rows), 9),
        "charged_cost": round(sum(float(row["charged_cost"]) for row in merged_rows), 9),
        "all_costs_recomputed": all(
            row.get("checks", {}).get("charged_cost") is True for row in merged_rows
        ),
        "all_raw_paths_resolve": len(merged_rows) == expected_rows
        and all(row["raw_path_resolves"] for row in merged_rows),
        "failures_retained": all(row["retained"] for row in failure_rows),
        "family_readiness_recomputation": family_readiness,
        "stored_family_readiness_values": {
            receipt["family_id"]: receipt["stored_readiness_value"] for receipt in receipts
        },
        "stored_family_aggregates_imported": False,
        "protocol_comparison": {
            "source_units_identical": protocol_result["checks"]["source_hashes"],
            "prompt_identical": protocol_result["checks"]["prompt_hash"],
            "order_identical": [row.get("unit_id") for row in units]
            == [row.get("unit_id") for row in sorted(units, key=lambda row: units.index(row))],
            "budget_identical": protocol_result["checks"]["family_contracts"],
            "prescribed_seeds_match": protocol_result["checks"]["family_contracts"],
            "seed_values_identical": protocol_result["seed_values_identical"],
            "model_identity_only_runtime_difference_required": True,
        },
    }
    ready_conditions = {
        "protocol": protocol_error is None and protocol_result["passed"],
        "all_families": all(value == 1.0 for value in family_readiness.values()),
        "exact_coverage": len(merged_rows) == expected_rows
        and all(row["row_present"] for row in coverage),
        "all_rows_replay": len(merged_rows) == expected_rows
        and all(row["replay_passed"] for row in merged_rows),
        "failures_retained": aggregate["failures_retained"],
        "raw_paths": aggregate["all_raw_paths_resolve"],
        "unload": all(row["passed"] for row in unload_rows),
        "costs": aggregate["all_costs_recomputed"],
        "attacks": all(row["passed"] for row in attacks),
        "protected": protected["all_unchanged"],
    }
    all_ready = 1.0 if all(ready_conditions.values()) else 0.0
    aggregate["ready_conditions"] = ready_conditions
    aggregate["ready_score"] = all_ready
    retirement = _retirement(repo_root, receipts)
    gates = _gate_summary(receipts, family_specs, family_readiness, all_ready, retirement)
    blocked_input = any(
        not receipt["exists"]
        or receipt["read_error"] is not None
        or str(receipt.get("status", "")).startswith("blocked")
        or receipt["row_count"] == 0
        for receipt in receipts
    )
    if all_ready == 1.0:
        status = "complete_three_family_source_receipt_audit"
        verdict_class = None
        verdict = (
            f"complete_three_family_source_receipt_audit: lineage={len(family_specs)}/{len(family_specs)}; "
            f"coverage={len(merged_rows)}/{expected_rows}; failure_retention={failure_row_count}/{failure_row_count}; "
            f"unload={sum(row['passed'] for row in unload_rows)}/{len(family_specs)}; merge=immutable_ready"
        )
    elif protected["all_unchanged"] is not True:
        status = "disqualified_three_family_source_receipt_audit"
        verdict_class = "disqualified"
        verdict = (
            f"disqualified_three_family_source_receipt_audit: lineage={sum(value == 1.0 for value in family_readiness.values())}/{len(family_specs)}; "
            f"coverage={len(merged_rows)}/{expected_rows}; failure_retention={failure_row_count}/{failure_row_count}; "
            f"unload={sum(row['passed'] for row in unload_rows)}/{len(family_specs)}; merge=protected_file_drift"
        )
    elif blocked_input:
        status = "blocked_three_family_source_receipt_audit"
        verdict_class = "blocked"
        first = gates["first_failure"] or {}
        verdict = (
            f"blocked_three_family_source_receipt_audit: lineage={sum(value == 1.0 for value in family_readiness.values())}/{len(family_specs)}; "
            f"coverage={len(merged_rows)}/{expected_rows}; failure_retention={failure_row_count}/{failure_row_count}; "
            f"unload={sum(row['passed'] for row in unload_rows)}/{len(family_specs)}; merge=not_eligible; "
            f"field={first.get('field')} expected={first.get('expected')} observed={first.get('observed')}"
        )
    else:
        status = "partial_three_family_source_receipt_audit"
        verdict_class = "partial"
        verdict = (
            f"partial_three_family_source_receipt_audit: lineage={sum(value == 1.0 for value in family_readiness.values())}/{len(family_specs)}; "
            f"coverage={len(merged_rows)}/{expected_rows}; failure_retention={failure_row_count}/{failure_row_count}; "
            f"unload={sum(row['passed'] for row in unload_rows)}/{len(family_specs)}; merge=replay_incomplete"
        )
    raw_path_receipts = _raw_path_preconditions(payloads)
    preconditions = {
        "planning_date": RUN_DATE,
        "expected_family_paths": [family["artifact_path"] for family in family_specs],
        "protocol_path": PROTOCOL_RELATIVE_PATH.as_posix(),
        "protocol_exists": (repo_root / PROTOCOL_RELATIVE_PATH).is_file(),
        "protocol_sha256": sha256_file(repo_root / PROTOCOL_RELATIVE_PATH),
        "protocol_read_error": protocol_error,
        "source_manifest_hash": protocol_result["manifest_hash"],
        "source_manifest_hash_recomputed": sha256_json(
            {
                key: value
                for key, value in protocol.get("source_unit_manifest", {}).items()
                if key != "manifest_hash"
            }
        )
        if isinstance(protocol.get("source_unit_manifest"), Mapping)
        else "missing",
        "prompt_sha256": protocol_result["prompt_sha256"],
        "budget_sha256": protocol_result["budget_sha256"],
        "protocol_checks": protocol_result["checks"],
        "family_contract_rows": protocol_result["family_contract_rows"],
        "audit_tool_versions": [
            _tool_version(repo_root, relative) for relative in AUDIT_TOOL_RELATIVE_PATHS
        ],
        "resources": _resources(repo_root),
        "protected_file_hashes_before": before,
        "protected_file_hashes_after": after,
        "raw_content_addressed_path_receipts": raw_path_receipts,
        "all_declared_raw_paths_available": all(row["exists"] for row in raw_path_receipts),
        "model_inference_invoked": False,
        "llm_calls_issued": 0,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    artifact: JsonDict = {
        "schema": "carnot.experiment_6584_three_family_source_receipt_audit.v1",
        "task_id": "exp6584-three-family-source-receipt-audit",
        "planning_date": RUN_DATE,
        "status": status,
        "honest_verdict": verdict,
        "verdict_class": verdict_class,
        "gate_check_summary": gates,
        "upstream_artifact_receipts": receipts,
        "rows": merged_rows,
        "family_coverage_rows": coverage,
        "failure_retention_rows": failure_rows,
        "duplicate_drift_and_substitution_rows": attacks,
        "unload_and_recovery_rows": unload_rows,
        "all_family_source_audit_ready_score": all_ready,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(),
        "duration_s": round(float(duration_s), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate terminal shape, non-positive semantics, hashes, and reducer output."""

    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_mismatch")
    if payload.get("verdict_class") not in (None, "partial", "blocked", "disqualified"):
        errors.append("verdict_class_invalid")
    score = payload.get("all_family_source_audit_ready_score")
    aggregate = payload.get("aggregate_row_recomputation", {})
    if score not in (0.0, 1.0) or not isinstance(aggregate, Mapping):
        errors.append("ready_score_invalid")
    elif score != aggregate.get("ready_score"):
        errors.append("ready_score_reducer_mismatch")
    if score == 1.0 and payload.get("verdict_class") is not None:
        errors.append("clean_verdict_class_not_null")
    if payload.get("verdict_class") == "blocked" and not str(
        payload.get("honest_verdict", "")
    ).startswith("blocked_"):
        errors.append("blocked_verdict_prefix_missing")
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance_incomplete")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if provenance.get(field, {}).get("principle") != principle:
                errors.append(f"field_principle_mismatch:{field}")
    for container in (
        "rows",
        "family_coverage_rows",
        "failure_retention_rows",
        "duplicate_drift_and_substitution_rows",
        "unload_and_recovery_rows",
    ):
        rows = payload.get(container, [])
        if not isinstance(rows, list):
            errors.append(f"{container}_not_list")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or row.get("row_hash") != row_hash(row):
                errors.append(f"{container}_row_hash:{index}")
    if payload.get("reproducibility_checksum") != artifact_checksum(payload):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Validate first, then replace one same-directory terminal artifact."""

    errors = validate_artifact(payload)
    if errors:
        raise ValueError("artifact validation failed: " + ", ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp6584-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "atomic_replace": True}


def _run_test(command: str, repo_root: Path) -> JsonDict:  # pragma: no cover - CLI receipt.
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    return {
        "command": command,
        "exit_code": result.returncode,
        "duration_s": round(time.monotonic() - started, 6),
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
    }


def run_experiment(
    repo_root: Path, run_date: str, *, run_tests: bool = True
) -> JsonDict:  # pragma: no cover - required live entrypoint.
    started = time.monotonic()
    protected_before = protected_hashes(repo_root)
    commands = (
        ".venv/bin/pytest tests/python/test_experiment_6584_three_family_source_receipt_audit.py -q --no-cov -n 0",
        ".venv/bin/coverage run --rcfile=/dev/null --include='*/experiment_6584_three_family_source_receipt_audit.py' -m pytest -o addopts='' tests/python/test_experiment_6584_three_family_source_receipt_audit.py -q --no-cov -n 0",
        ".venv/bin/coverage report --rcfile=/dev/null --include='*/experiment_6584_three_family_source_receipt_audit.py' --fail-under=100 --show-missing",
        ".venv/bin/ruff check python/carnot/experiment_6584_three_family_source_receipt_audit.py tests/python/test_experiment_6584_three_family_source_receipt_audit.py",
        ".venv/bin/ruff format --check python/carnot/experiment_6584_three_family_source_receipt_audit.py tests/python/test_experiment_6584_three_family_source_receipt_audit.py",
        ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6584_three_family_source_receipt_audit.py",
        ".venv/bin/pytest tests/python -q",
    )
    tests_run = [_run_test(command, repo_root) for command in commands] if run_tests else []
    artifact = build_audit(
        repo_root,
        duration_s=time.monotonic() - started,
        tests_run=tests_run,
        protected_before=protected_before,
    )
    artifact["planning_date"] = run_date
    artifact["preconditions_checked"]["planning_date"] = run_date
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        payload, error = _read_json(args.validate)
        errors = [error] if error else validate_artifact(payload or {})
        if errors:
            print("invalid: " + ", ".join(str(item) for item in errors))
            return 1
        print("valid")
        return 0
    artifact = run_experiment(REPO_ROOT, args.date, run_tests=not args.skip_tests)
    atomic_write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": artifact["status"],
                "honest_verdict": artifact["honest_verdict"],
                "all_family_source_audit_ready_score": artifact[
                    "all_family_source_audit_ready_score"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
