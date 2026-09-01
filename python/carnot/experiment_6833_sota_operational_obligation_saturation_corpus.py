"""Build the live local SOTA operational-obligation saturation corpus.

The producer treats model text as untrusted evidence. Exp6832 exact checkers
score each retained byte sequence. Readiness measures execution completeness,
not whether a model selected the correct actions.

Spec refs: REQ-CONSTRAINT-6833 and SCENARIO-CONSTRAINT-6833-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
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

from carnot import experiment_6832_operational_obligation_saturation_fixture as fixture_api
from carnot.experiment_6812_sota_operational_handoff_corpus_v2 import (
    _gpu_inventory,
    _gpu_snapshot,
    _llama_server_path,
    _offload_layers,
    _pid_start_ticks,
)
from carnot.inference.gguf_metadata import read_gguf_metadata
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf
from carnot.gpu_lease_phase_journal import (
    GpuLease,
    journal_path_for,
    process_start_matches,
    read_journal,
    validate_journal_document,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
FIXTURE_PATH = Path("results/experiment_6832_operational_obligation_saturation_fixture.json")
MODULE_PATH = Path("python/carnot/experiment_6833_sota_operational_obligation_saturation_corpus.py")
WRAPPER_PATH = Path(
    "scripts/experiments/experiment_6833_sota_operational_obligation_saturation_corpus.py"
)
OUTPUT_PATH = Path("results/experiment_6833_sota_operational_obligation_saturation_corpus.json")
CHECKPOINT_PATH = Path(
    "results/.checkpoints/experiment_6833_sota_operational_obligation_saturation_corpus/rows.json"
)
SCHEMA = "carnot.experiment_6833.sota_operational_obligation_saturation_corpus.v1"
EXPERIMENT_ID = "experiment_6833_sota_operational_obligation_saturation_corpus"
BLOCKED_STATUS = "complete_blocked_sota_operational_saturation_corpus"
COMPLETE_STATUS = "complete_sota_operational_saturation_corpus"
PARTIAL_STATUS = "complete_partial_sota_operational_saturation_corpus"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 6833
ARMS = ("typed", "compressed")
BATCH_SIZE = 25
DISK_FREE_FLOOR_BYTES = 1_073_741_824
VRAM_RECOVERY_TOLERANCE_MB = 512

PLANNED_MODELS: tuple[JsonDict, ...] = (
    {
        "family_id": "qwen36",
        "hub_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "expected_sha256": "sha256:ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",
    },
    {
        "family_id": "gemma31",
        "hub_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "flagship_dense",
        "filename": "gemma-4-31B-it-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "expected_sha256": "sha256:9fdf3dc8b0384830b4402d151388c140bd8eb2abf8d60588d8224231198254a1",
    },
    {
        "family_id": "gemma26",
        "hub_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe",
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "quantization": "Q4_K_M",
        "expected_sha256": "sha256:34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35",
    },
)

DECODE_SETTINGS: JsonDict = {
    "context_size": 4096,
    "max_output_tokens": 384,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
    "stop_rules": ["</s>", "<|im_end|>", "<|endoftext|>"],
    "retry_budget": 0,
    "repair_budget": 0,
    "answer_feedback": False,
    "grammar": None,
    "request_timeout_s": 240,
    "load_timeout_s": 360,
    "teardown_timeout_s": 90,
}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "The versioned name fixes the artifact contract.",
    "experiment_id": "The stable name identifies the producer.",
    "run_date": "The supplied date identifies this execution.",
    "status": "The terminal state separates complete, partial, and blocked work.",
    "field_principles": "Each top-level field states why it exists.",
    "preconditions_checked": "Observed gates explain whether live work was admissible.",
    "inference_substrate": "Only live local model inference supports this corpus.",
    "duration_s": "The full monotonic task duration makes omitted work visible.",
    "phase_clocks": "Phase and batch clocks expose where task time was spent.",
    "random_seed": "The fixed seed binds every matched generation request.",
    "reproducibility_checksum": "The hash binds fixture, models, code, commands, rows, and output.",
    "MODEL_SPECS": "Resolved model records bind exact files and embedded prompt machinery.",
    "models_used": "Only models with complete corpus receipts appear here.",
    "fixture_receipt": "The exact Exp6832 file and checker hashes bind scoring truth.",
    "code_receipts": "Producer and wrapper hashes identify the executing code.",
    "process_receipts": "Owned process, CUDA, token, lease, and teardown evidence stays model-local.",
    "accelerator_samples": "Task-window GPU samples bind model residency and recovery.",
    "checkpoint_manifest": "Durable row identities and hashes support exact resume.",
    "per_unit_rows": "Each model, scenario, and arm unit retains bytes and exact scores.",
    "row_coverage": "Expected, observed, duplicate, and missing identities control completeness.",
    "budget_parity": "Matched request settings keep both prompt arms comparable.",
    "exact_scores": "Exp6832 checks summarize obligations and joint outcomes.",
    "descriptive_aggregates": "Non-inferential summaries support the independent audit.",
    "operational_saturation_corpus_ready": "Completeness and authenticity control downstream use.",
    "gate_check_summary": "The summary names the first failed gate and exact values.",
    "verifier_is_oracle": "False keeps deterministic external checks separate from model generation.",
    "verdict_class": "A closed class prevents readiness from becoming an inferential claim.",
    "honest_verdict": "The complete prefix exposes the terminal outcome.",
}
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}


def canonical_bytes(value: Any) -> bytes:
    """Encode stable JSON bytes for hashes and durable files."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the repository's explicit SHA-256 form."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a large model without loading it into memory."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def row_identity(model_id: str, scenario_id: str, arm: str) -> str:
    """Name the exact unit so restart cannot silently alter its denominator."""

    return f"{model_id}|{scenario_id}|{arm}"


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    """Hash the manifest without its self-referential checksum."""

    return sha256_bytes(
        canonical_bytes({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    )


def build_manifest(fixture: Mapping[str, Any]) -> JsonDict:
    """Freeze all 900 identities before any local model starts."""

    scenarios = fixture.get("scenarios") or []
    expected = [
        row_identity(str(model["hub_id"]), str(scenario["scenario_id"]), arm)
        for model in PLANNED_MODELS
        for scenario in scenarios
        for arm in ARMS
    ]
    manifest: JsonDict = {
        "fixture_schema": fixture.get("schema"),
        "fixture_reproducibility_checksum": fixture.get("reproducibility_checksum"),
        "fixture_scenario_hashes": [scenario.get("scenario_hash") for scenario in scenarios],
        "model_ids": [model["hub_id"] for model in PLANNED_MODELS],
        "arms": list(ARMS),
        "random_seed": RANDOM_SEED,
        "decode_settings": deepcopy(DECODE_SETTINGS),
        "expected_row_count": len(expected),
        "expected_row_ids": expected,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = manifest_checksum(manifest)
    return manifest


def _revision_from_path(path: Path) -> str:
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "local-unversioned"


def _read_metadata_string(path: Path, metadata: Mapping[str, Any], key: str) -> str:
    """Read one bounded GGUF string using offsets proven by the shared parser."""

    source = (
        metadata.get("field_provenance", {}).get("metadata_keys", {}).get(key, {})
        if isinstance(metadata.get("field_provenance"), Mapping)
        else {}
    )
    offset = source.get("value_offset") if isinstance(source, Mapping) else None
    if not isinstance(offset, int) or source.get("value_type") != "string":
        return ""
    with path.open("rb") as handle:
        handle.seek(offset)
        length_raw = handle.read(8)
        if len(length_raw) != 8:
            return ""
        length = int.from_bytes(length_raw, "little")
        if length < 1 or length > 16 * 1024 * 1024:
            return ""
        raw = handle.read(length)
    try:
        return raw.decode("utf-8") if len(raw) == length else ""
    except UnicodeDecodeError:
        return ""


def read_model_metadata(path: str | Path) -> JsonDict:
    """Capture tokenizer fields and the exact embedded chat template."""

    candidate = Path(path)
    metadata = read_gguf_metadata(candidate)
    template = _read_metadata_string(candidate, metadata, "tokenizer.chat_template")
    return {
        **metadata,
        "chat_template": template,
        "chat_template_sha256": sha256_bytes(template.encode()),
    }


def resolve_model_specs(
    *,
    pair_resolver: Callable[..., list[dict] | None] = cached_sota_pair,
    single_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
    metadata_reader: Callable[[str | Path], JsonDict] = read_model_metadata,
    file_hasher: Callable[[str | Path], str] = sha256_file,
) -> list[JsonDict]:
    """Call the shared pair resolver and then resolve the required middle model."""

    pair = pair_resolver(gpu_indices=(0, 1), model_indices=(0, 2)) or []
    paths = {
        str(row.get("hf_id")): str(row.get("model_path") or "")
        for row in pair
        if isinstance(row, Mapping)
    }
    middle_id = str(PLANNED_MODELS[2]["hub_id"])
    paths[middle_id] = str(single_resolver(middle_id, "Q4_K_M") or "")
    records: list[JsonDict] = []
    for planned in PLANNED_MODELS:
        path = Path(paths.get(str(planned["hub_id"]), ""))
        metadata: JsonDict = {}
        if path.is_file():
            try:
                metadata = metadata_reader(path)
            except (OSError, ValueError):
                metadata = {}
        records.append(
            {
                **deepcopy(planned),
                "revision": _revision_from_path(path) if path.is_file() else "missing",
                "model_path": str(path.absolute()) if path.is_file() else "",
                "model_sha256": file_hasher(path) if path.is_file() else "missing",
                "model_size_bytes": path.stat().st_size if path.is_file() else 0,
                "gguf_metadata": metadata,
                "decode_settings": deepcopy(DECODE_SETTINGS),
            }
        )
    return records


def model_record_errors(record: Mapping[str, Any], planned: Mapping[str, Any]) -> list[str]:
    """Reject substitution and missing embedded prompt machinery."""

    errors = [key for key, value in planned.items() if record.get(key) != value]
    if record.get("model_sha256") != planned.get("expected_sha256"):
        errors.append("model_sha256")
    path = Path(str(record.get("model_path") or ""))
    if path.name != planned.get("filename"):
        errors.append("filename")
    metadata = record.get("gguf_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("is_language_model") is not True:
        errors.append("language_model_metadata")
        metadata = {}
    tokenizer = metadata.get("tokenizer_metadata")
    if not isinstance(tokenizer, Mapping) or int(tokenizer.get("token_count") or 0) < 1:
        errors.append("embedded_tokenizer")
    if not metadata.get("chat_template") or not tokenizer.get("chat_template_present"):
        errors.append("embedded_chat_template")
    if record.get("decode_settings") != DECODE_SETTINGS:
        errors.append("decode_settings")
    return list(dict.fromkeys(errors))


def request_payload(prompt: str, seed: int, *, max_tokens: int) -> JsonDict:
    """Build one matched chat request without grammar, repair, or feedback."""

    return {
        "model": "local-gguf",
        "messages": [{"role": "user", "content": prompt}],
        "seed": int(seed),
        "temperature": DECODE_SETTINGS["temperature"],
        "top_p": DECODE_SETTINGS["top_p"],
        "top_k": DECODE_SETTINGS["top_k"],
        "repeat_penalty": DECODE_SETTINGS["repeat_penalty"],
        "max_tokens": int(max_tokens),
        "stop": deepcopy(DECODE_SETTINGS["stop_rules"]),
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _row_checksum(row: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_bytes({key: value for key, value in row.items() if key != "row_sha256"})
    )


def build_scored_row(
    *,
    scenario: Mapping[str, Any],
    arm: str,
    prompt: bytes,
    raw_output: bytes,
    model: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
    prompt_tokens: int,
    generated_tokens: int,
    latency_s: float,
    checker_sha256: str,
) -> JsonDict:
    """Retain exact bytes and score once through the frozen Exp6832 checker."""

    joint = fixture_api.check_joint(scenario, raw_output)
    parsed_fields = (
        {"selected_action_ids": joint["selected_action_ids"]} if joint.get("parsed") else None
    )
    identity = row_identity(str(model["hub_id"]), str(scenario["scenario_id"]), arm)
    row: JsonDict = {
        "row_id": identity,
        "model_id": model["hub_id"],
        "model_revision": model["revision"],
        "model_filename": model["filename"],
        "model_sha256": model["model_sha256"],
        "scenario_id": scenario["scenario_id"],
        "scenario_hash": scenario["scenario_hash"],
        "obligation_count": scenario["obligation_count"],
        "arm": arm,
        "random_seed": RANDOM_SEED,
        "decode_settings": deepcopy(DECODE_SETTINGS),
        "prompt_bytes_b64": base64.b64encode(prompt).decode("ascii"),
        "prompt_byte_length": len(prompt),
        "prompt_sha256": sha256_bytes(prompt),
        "raw_output_bytes_b64": base64.b64encode(raw_output).decode("ascii"),
        "raw_output_byte_length": len(raw_output),
        "raw_output_sha256": sha256_bytes(raw_output),
        "parsed_fields": parsed_fields,
        "parse_status": {"parsed": bool(joint.get("parsed")), "error": joint.get("parse_error")},
        "obligation_results": deepcopy(joint.get("obligation_checks") or {}),
        "joint_result": deepcopy(joint),
        "token_counts": {
            "prompt": int(prompt_tokens),
            "generated": int(generated_tokens),
            "allowed_generated": DECODE_SETTINGS["max_output_tokens"],
        },
        "latency_s": round(float(latency_s), 6),
        "process_identity": {
            "session_id": process_receipt["session_id"],
            "pid": process_receipt["pid"],
            "pid_start_ticks": process_receipt["pid_start_ticks"],
            "port": process_receipt["port"],
            "physical_gpu_uuid": process_receipt["physical_gpu_uuid"],
        },
        "checker_sha256": checker_sha256,
        "scoring_authority": "Exp6832 exact external checkers",
        "repair_attempts": 0,
        "content_retry_attempts": 0,
        "row_sha256": "",
    }
    row["row_sha256"] = _row_checksum(row)
    return row


def validate_row(
    row: Mapping[str, Any],
    scenario: Mapping[str, Any],
    model: Mapping[str, Any],
    process_receipt: Mapping[str, Any],
) -> list[str]:
    """Recompute byte, identity, process, budget, score, and row hashes."""

    errors: list[str] = []
    try:
        prompt = base64.b64decode(str(row.get("prompt_bytes_b64") or ""), validate=True)
        raw = base64.b64decode(str(row.get("raw_output_bytes_b64") or ""), validate=True)
    except (ValueError, TypeError):
        return ["invalid row base64"]
    expected_id = row_identity(
        str(model["hub_id"]), str(scenario["scenario_id"]), str(row.get("arm"))
    )
    if row.get("row_id") != expected_id:
        errors.append("row identity mismatch")
    if row.get("model_sha256") != model.get("model_sha256"):
        errors.append("row model mismatch")
    if row.get("scenario_hash") != scenario.get("scenario_hash"):
        errors.append("row scenario mismatch")
    if row.get("prompt_sha256") != sha256_bytes(prompt) or row.get("prompt_byte_length") != len(
        prompt
    ):
        errors.append("prompt byte receipt mismatch")
    if row.get("raw_output_sha256") != sha256_bytes(raw) or row.get(
        "raw_output_byte_length"
    ) != len(raw):
        errors.append("output byte receipt mismatch")
    if row.get("decode_settings") != DECODE_SETTINGS:
        errors.append("row budget mismatch")
    process = row.get("process_identity") or {}
    if process.get("session_id") != process_receipt.get("session_id"):
        errors.append("row process mismatch")
    rescored = fixture_api.check_joint(scenario, raw)
    if row.get("joint_result") != rescored or row.get("obligation_results") != rescored.get(
        "obligation_checks"
    ):
        errors.append("exact score mismatch")
    if row.get("row_sha256") != _row_checksum(row):
        errors.append("row checksum mismatch")
    return errors


class RowCheckpoint:
    """Store completed rows atomically and reject any changed completed row."""

    def __init__(self, path: Path, manifest_sha256: str) -> None:
        self.path = path
        self.manifest_sha256 = manifest_sha256
        self.rows: list[JsonDict] = []
        self.process_receipts: list[JsonDict] = []
        if path.is_file():
            document = json.loads(path.read_bytes())
            if document.get("manifest_sha256") != manifest_sha256:
                raise ValueError("checkpoint manifest hash mismatch")
            rows = document.get("rows")
            if not isinstance(rows, list):
                raise ValueError("checkpoint rows invalid")
            for row in rows:
                if not isinstance(row, dict) or row.get("row_sha256") != _row_checksum(row):
                    raise ValueError("checkpoint row hash mismatch")
            identities = [row["row_id"] for row in rows]
            if len(identities) != len(set(identities)):
                raise ValueError("checkpoint duplicate row identity")
            self.rows = deepcopy(rows)
            receipts = document.get("process_receipts", [])
            if not isinstance(receipts, list) or any(
                not isinstance(receipt, dict) for receipt in receipts
            ):
                raise ValueError("checkpoint process receipts invalid")
            sessions = [receipt.get("session_id") for receipt in receipts]
            if len(sessions) != len(set(sessions)):
                raise ValueError("checkpoint duplicate process receipt")
            self.process_receipts = deepcopy(receipts)

    @property
    def completed_ids(self) -> set[str]:
        return {str(row["row_id"]) for row in self.rows}

    def missing_ids(self, expected: Sequence[str]) -> list[str]:
        complete = self.completed_ids
        return [identity for identity in expected if identity not in complete]

    def _persist(self) -> str:
        """Publish rows and process receipts in one durable document."""

        payload = {
            "schema": "carnot.experiment_6833.row_checkpoint.v1",
            "manifest_sha256": self.manifest_sha256,
            "rows": self.rows,
            "process_receipts": self.process_receipts,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return sha256_bytes(self.path.read_bytes())

    def append_batch(self, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
        existing = {str(row["row_id"]): row for row in self.rows}
        accepted: list[str] = []
        duplicates: list[str] = []
        for row_value in rows:
            row = deepcopy(dict(row_value))
            identity = str(row["row_id"])
            if row.get("row_sha256") != _row_checksum(row):
                raise ValueError("checkpoint row hash mismatch")
            if identity in existing:
                if existing[identity] != row:
                    raise ValueError("checkpoint completed row changed")
                duplicates.append(identity)
                continue
            existing[identity] = row
            self.rows.append(row)
            accepted.append(identity)
        return {
            "accepted_row_ids": accepted,
            "duplicate_row_ids": duplicates,
            "completed_row_count": len(self.rows),
            "checkpoint_sha256": self._persist(),
            "durable": True,
        }

    def record_process_receipt(self, receipt_value: Mapping[str, Any]) -> JsonDict:
        """Publish a model receipt after teardown without changing completed rows."""

        receipt = deepcopy(dict(receipt_value))
        session_id = str(receipt.get("session_id") or "")
        if not session_id:
            raise ValueError("checkpoint process receipt has no session")
        existing = {
            str(candidate.get("session_id")): candidate for candidate in self.process_receipts
        }
        if session_id in existing:
            if existing[session_id] != receipt:
                raise ValueError("checkpoint process receipt changed")
            return {
                "session_id": session_id,
                "duplicate": True,
                "checkpoint_sha256": sha256_bytes(self.path.read_bytes()),
                "durable": True,
            }
        self.process_receipts.append(receipt)
        return {
            "session_id": session_id,
            "duplicate": False,
            "checkpoint_sha256": self._persist(),
            "durable": True,
        }

    def upsert_process_receipt(self, receipt_value: Mapping[str, Any]) -> JsonDict:
        """Persist active evidence, but never let one process become another."""

        receipt = deepcopy(dict(receipt_value))
        session_id = str(receipt.get("session_id") or "")
        if not session_id:
            raise ValueError("checkpoint process receipt has no session")
        identity_fields = (
            "purpose",
            "model_id",
            "model_sha256",
            "session_id",
            "command",
            "pid",
            "process_start_time",
            "pid_start_ticks",
            "port",
            "physical_gpu_uuid",
            "visible_devices",
        )
        for index, current in enumerate(self.process_receipts):
            if str(current.get("session_id")) != session_id:
                continue
            if current == receipt:
                return {
                    "session_id": session_id,
                    "duplicate": True,
                    "updated": False,
                    "checkpoint_sha256": sha256_bytes(self.path.read_bytes()),
                    "durable": True,
                }
            if any(current.get(field) != receipt.get(field) for field in identity_fields):
                raise ValueError("checkpoint process identity changed")
            if current.get("authentic") is True:
                raise ValueError("checkpoint completed process receipt changed")
            self.process_receipts[index] = receipt
            return {
                "session_id": session_id,
                "duplicate": False,
                "updated": True,
                "checkpoint_sha256": self._persist(),
                "durable": True,
            }
        self.process_receipts.append(receipt)
        return {
            "session_id": session_id,
            "duplicate": False,
            "updated": False,
            "checkpoint_sha256": self._persist(),
            "durable": True,
        }


def build_recovered_process_receipt(
    *,
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    server: Path,
    journal: Mapping[str, Any],
    server_log: bytes,
    server_log_path: Path,
    lease_release: Mapping[str, Any],
    device_index: int,
    process_start_time: str,
    teardown_duration_s: float,
    process_absent_after_exit: bool,
    gpu_memory_recovered: bool,
) -> JsonDict:
    """Recover a stopped process receipt only from matching durable evidence."""

    if not rows:
        raise ValueError("recovery rows missing")
    journal_errors = validate_journal_document(journal, check_freshness=False)
    if journal_errors:
        raise ValueError("lease journal invalid: " + ",".join(journal_errors))
    first_process = rows[0].get("process_identity") or {}
    session_id = str(first_process.get("session_id") or "")
    row_processes = [row.get("process_identity") or {} for row in rows]
    if any(process != first_process for process in row_processes):
        raise ValueError("recovery row process identities differ")
    if any(row.get("model_id") != model.get("hub_id") for row in rows):
        raise ValueError("recovery row model mismatch")
    family_id = str(model.get("family_id") or "")
    expected_task = f"exp6833-corpus-{family_id}"
    if (
        journal.get("task_id") != expected_task
        or journal.get("expected_model") != model.get("hub_id")
        or journal.get("device_uuid") != first_process.get("physical_gpu_uuid")
    ):
        raise ValueError("lease journal identity mismatch")
    port = int(first_process.get("port") or 0)
    model_path = str(model.get("model_path") or "").encode()
    log_identity = (
        model_path in server_log
        and b"CUDA0" in server_log
        and f"127.0.0.1:{port}".encode() in server_log
    )
    offloaded_layers, total_layers = _offload_layers(server_log)
    completed_requests = server_log.count(b"done request: POST /v1/chat/completions")
    if not log_identity or offloaded_layers <= 0 or completed_requests < len(rows):
        raise ValueError("server log identity mismatch")
    raw_outputs: list[bytes] = []
    try:
        raw_outputs = [
            base64.b64decode(str(row["raw_output_bytes_b64"]), validate=True) for row in rows
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("recovery row output invalid") from exc
    if not raw_outputs[0] or not raw_outputs[-1]:
        raise ValueError("recovery token evidence missing")
    lease_released = lease_release.get("released") is True
    authentic = bool(
        session_id
        and int(first_process.get("pid") or 0) > 1
        and int(first_process.get("pid_start_ticks") or 0) > 0
        and lease_released
        and process_absent_after_exit
        and gpu_memory_recovered
    )
    return {
        "purpose": "corpus",
        "model_id": model["hub_id"],
        "model_sha256": model["model_sha256"],
        "session_id": session_id,
        "command": _server_command(server, model, port),
        "pid": int(first_process["pid"]),
        "process_start_time": process_start_time,
        "pid_start_ticks": int(first_process["pid_start_ticks"]),
        "port": port,
        "physical_gpu_uuid": first_process["physical_gpu_uuid"],
        "visible_devices": [int(device_index)],
        "first_token_b64": base64.b64encode(raw_outputs[0][:1]).decode("ascii"),
        "final_token_b64": base64.b64encode(raw_outputs[-1][-1:]).decode("ascii"),
        "token_receipt_semantics": "first_and_final_retained_output_bytes",
        "lease_owned": True,
        "lease_released": lease_released,
        "cuda_offload": True,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "process_exit_code": None,
        "process_absent_after_exit": bool(process_absent_after_exit),
        "teardown_complete": bool(process_absent_after_exit and gpu_memory_recovered),
        "teardown_mode": "stale_lease_recovery",
        "teardown_duration_s": round(float(teardown_duration_s), 6),
        "duration_s": None,
        "error": None,
        "receipt_status": "recovered_complete",
        "receipt_recovery": {
            "journal_sha256": sha256_bytes(canonical_bytes(journal)),
            "server_log_path": str(server_log_path),
            "server_log_sha256": sha256_bytes(server_log),
            "completed_request_count": completed_requests,
            "retained_row_count": len(rows),
            "lease_release": deepcopy(dict(lease_release)),
            "process_exit_code_observed": False,
        },
        "authentic": authentic,
    }


def validate_cross_model_isolation(
    rows: Sequence[Mapping[str, Any]], receipts: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Require one corpus process identity for every row's declared model."""

    errors: list[str] = []
    by_session = {
        str(receipt.get("session_id")): receipt
        for receipt in receipts
        if receipt.get("purpose", "corpus") == "corpus"
    }
    corpus_receipts = [
        receipt for receipt in receipts if receipt.get("purpose", "corpus") == "corpus"
    ]
    if len(by_session) != len(corpus_receipts):
        errors.append("model receipts share a process session")
    for row in rows:
        session_id = str((row.get("process_identity") or {}).get("session_id") or "")
        receipt = by_session.get(session_id)
        if receipt is None or receipt.get("model_id") != row.get("model_id"):
            errors.append("row process does not match its model receipt")
            break
    return errors


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Re-score retained output and bind every row to its exact process."""

    scenarios = {str(row["scenario_id"]): row for row in fixture.get("scenarios") or []}
    models_by_id = {str(row["hub_id"]): row for row in models}
    receipts_by_session = {
        str(row.get("session_id")): row
        for row in receipts
        if row.get("purpose", "corpus") == "corpus"
    }
    failures: list[JsonDict] = []
    for row in rows:
        scenario = scenarios.get(str(row.get("scenario_id")))
        model = models_by_id.get(str(row.get("model_id")))
        receipt = receipts_by_session.get(
            str((row.get("process_identity") or {}).get("session_id") or "")
        )
        if scenario is None or model is None or receipt is None:
            failures.append({"row_id": row.get("row_id"), "errors": ["row source missing"]})
            continue
        errors = validate_row(row, scenario, model, receipt)
        if errors:
            failures.append({"row_id": row.get("row_id"), "errors": errors})
    return failures


def terminate_owned_process(process: Any, *, timeout_s: float) -> int | None:
    """Stop only the process object created by this task."""

    if process is None:
        return None
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            return int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            process.kill()
            return int(process.wait(timeout=10))
    return process.returncode


def _score_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fields = {field: {"passed": 0, "checked": 0} for field in fixture_api.OBLIGATION_FIELDS}
    obligation_passed = 0
    obligation_checked = 0
    for row in rows:
        for obligation in (row.get("obligation_results") or {}).values():
            obligation_checked += 1
            obligation_passed += int(obligation.get("passed") is True)
            for field, passed in (obligation.get("fields") or {}).items():
                fields[field]["checked"] += 1
                fields[field]["passed"] += int(passed is True)
    return {
        "row_count": len(rows),
        "parsed": sum((row.get("parse_status") or {}).get("parsed") is True for row in rows),
        "joint_passed": sum((row.get("joint_result") or {}).get("passed") is True for row in rows),
        "joint_failed": sum(
            (row.get("joint_result") or {}).get("passed") is not True for row in rows
        ),
        "obligation_checked": obligation_checked,
        "obligation_passed": obligation_passed,
        "field_results": fields,
        "scoring_authority": "Exp6832 exact external checkers",
    }


def _descriptive_aggregates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    cells: JsonDict = {}
    for model in PLANNED_MODELS:
        model_id = str(model["hub_id"])
        cells[model_id] = {}
        for arm in ARMS:
            arm_rows = [
                row for row in rows if row.get("model_id") == model_id and row.get("arm") == arm
            ]
            cells[model_id][arm] = {
                "rows": len(arm_rows),
                "parsed": sum(
                    (row.get("parse_status") or {}).get("parsed") is True for row in arm_rows
                ),
                "joint_passed": sum(
                    (row.get("joint_result") or {}).get("passed") is True for row in arm_rows
                ),
                "by_obligation_count": {
                    str(count): {
                        "rows": len(
                            [row for row in arm_rows if row.get("obligation_count") == count]
                        ),
                        "joint_passed": sum(
                            (row.get("joint_result") or {}).get("passed") is True
                            for row in arm_rows
                            if row.get("obligation_count") == count
                        ),
                    }
                    for count in fixture_api.OBLIGATION_COUNTS
                },
            }
    return {"authority": "descriptive_only_exp6834_owns_inference", "cells": cells}


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "fixture_receipt": artifact.get("fixture_receipt"),
        "MODEL_SPECS": artifact.get("MODEL_SPECS"),
        "code_receipts": artifact.get("code_receipts"),
        "process_receipts": artifact.get("process_receipts"),
        "per_unit_rows": artifact.get("per_unit_rows"),
    }
    return sha256_bytes(canonical_bytes(payload))


def _base_artifact(
    *,
    run_date: str,
    models: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_clocks: Mapping[str, Any],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": BLOCKED_STATUS,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "phase_clocks": deepcopy(dict(phase_clocks)),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "MODEL_SPECS": deepcopy(list(models)),
        "models_used": [],
        "fixture_receipt": {},
        "code_receipts": {},
        "process_receipts": [],
        "accelerator_samples": [],
        "checkpoint_manifest": {},
        "per_unit_rows": [],
        "row_coverage": {"expected": 900, "observed_unique": 0, "duplicates": [], "missing": []},
        "budget_parity": {"passed": False, "settings": deepcopy(DECODE_SETTINGS)},
        "exact_scores": _score_summary([]),
        "descriptive_aggregates": _descriptive_aggregates([]),
        "operational_saturation_corpus_ready": False,
        "gate_check_summary": {
            "passed": False,
            "failed_check": None,
            "expected": None,
            "observed": None,
        },
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": BLOCKED_STATUS,
    }


def build_blocked_artifact(
    *,
    run_date: str,
    failed_check: str,
    expected: Any,
    observed: Any,
    preconditions: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_clocks: Mapping[str, Any],
    process_receipts: Sequence[Mapping[str, Any]] = (),
    accelerator_samples: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the required terminal block without headline rows."""

    artifact = _base_artifact(
        run_date=run_date, models=models, duration_s=duration_s, phase_clocks=phase_clocks
    )
    artifact.update(
        {
            "preconditions_checked": deepcopy(list(preconditions)),
            "process_receipts": deepcopy(list(process_receipts)),
            "accelerator_samples": deepcopy(list(accelerator_samples)),
            "gate_check_summary": {
                "passed": False,
                "failed_check": failed_check,
                "expected": expected,
                "observed": observed,
            },
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def assemble_artifact(
    *,
    run_date: str,
    fixture: Mapping[str, Any],
    manifest: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    process_receipts: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    accelerator_samples: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_clocks: Mapping[str, Any],
    code_receipts: Mapping[str, Any] | None = None,
    fixture_file_sha256: str | None = None,
) -> JsonDict:
    """Derive readiness from row and process evidence, never from accuracy."""

    expected_ids = list(manifest["expected_row_ids"])
    observed_ids = [str(row.get("row_id")) for row in rows]
    unique_ids = set(observed_ids)
    duplicates = sorted({identity for identity in observed_ids if observed_ids.count(identity) > 1})
    missing = [identity for identity in expected_ids if identity not in unique_ids]
    corpus_receipts = [
        receipt for receipt in process_receipts if receipt.get("purpose", "corpus") == "corpus"
    ]
    budget_passed = all(row.get("decode_settings") == DECODE_SETTINGS for row in rows)
    process_complete = (
        {receipt.get("model_id") for receipt in corpus_receipts}
        == {model["hub_id"] for model in PLANNED_MODELS}
        and all(receipt.get("authentic") is True for receipt in corpus_receipts)
        and all(receipt.get("teardown_complete") is True for receipt in corpus_receipts)
    )
    isolation_errors = validate_cross_model_isolation(rows, process_receipts)
    row_validation_errors = validate_rows(rows, fixture, models, process_receipts)
    checkpoint_complete = bool(checkpoint_receipts) and all(
        receipt.get("durable") is True for receipt in checkpoint_receipts
    )
    coverage_complete = (
        len(rows) == 900 and len(unique_ids) == 900 and not duplicates and not missing
    )
    ready = (
        coverage_complete
        and budget_passed
        and process_complete
        and checkpoint_complete
        and not isolation_errors
        and not row_validation_errors
    )
    artifact = _base_artifact(
        run_date=run_date, models=models, duration_s=duration_s, phase_clocks=phase_clocks
    )
    exact_scores = _score_summary(rows)
    artifact.update(
        {
            "status": COMPLETE_STATUS if ready else PARTIAL_STATUS,
            "preconditions_checked": deepcopy(list(preconditions)),
            "models_used": [
                model["hub_id"]
                for model in PLANNED_MODELS
                if any(
                    receipt.get("model_id") == model["hub_id"] and receipt.get("authentic") is True
                    for receipt in corpus_receipts
                )
            ],
            "fixture_receipt": {
                "path": str(FIXTURE_PATH),
                "schema": fixture.get("schema"),
                "file_sha256": fixture_file_sha256 or sha256_bytes(canonical_bytes(fixture)),
                "reproducibility_checksum": fixture.get("reproducibility_checksum"),
                "checker_sha256": (
                    fixture.get("implementation_hashes", {}).get("module", {}).get("sha256")
                ),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            "code_receipts": deepcopy(dict(code_receipts or {})),
            "process_receipts": deepcopy(list(process_receipts)),
            "accelerator_samples": deepcopy(list(accelerator_samples)),
            "checkpoint_manifest": {
                "manifest_sha256": manifest["manifest_sha256"],
                "completed_row_count": len(unique_ids),
                "completed_row_ids": sorted(unique_ids),
                "completed_row_hashes": {str(row["row_id"]): row["row_sha256"] for row in rows},
                "batch_receipts": deepcopy(list(checkpoint_receipts)),
                "valid": checkpoint_complete,
            },
            "per_unit_rows": deepcopy(list(rows)),
            "row_coverage": {
                "expected": len(expected_ids),
                "expected_identities": expected_ids,
                "observed_unique": len(unique_ids),
                "observed_identities": sorted(unique_ids),
                "duplicates": duplicates,
                "missing": missing,
            },
            "budget_parity": {
                "passed": budget_passed,
                "settings": deepcopy(DECODE_SETTINGS),
                "tokens_allowed_equal": budget_passed,
            },
            "exact_scores": exact_scores,
            "descriptive_aggregates": _descriptive_aggregates(rows),
            "operational_saturation_corpus_ready": ready,
            "gate_check_summary": {
                "passed": ready,
                "failed_check": None if ready else "corpus_readiness",
                "expected": {
                    "rows": 900,
                    "authentic_processes": 3,
                    "budget_parity": True,
                    "checkpoint_complete": True,
                    "cross_model_isolation": True,
                    "row_validation_errors": 0,
                },
                "observed": {
                    "rows": len(unique_ids),
                    "authentic_processes": len(
                        {
                            receipt.get("model_id")
                            for receipt in corpus_receipts
                            if receipt.get("authentic") is True
                        }
                    ),
                    "budget_parity": budget_passed,
                    "checkpoint_complete": checkpoint_complete,
                    "cross_model_isolation": not isolation_errors,
                    "row_validation_errors": row_validation_errors,
                },
            },
            "verdict_class": (
                "positive"
                if ready and exact_scores["joint_passed"] > 0
                else "null"
                if ready
                else "partial"
            ),
            "honest_verdict": (
                "complete_positive_sota_operational_saturation_corpus"
                if ready and exact_scores["joint_passed"] > 0
                else "complete_null_sota_operational_saturation_corpus"
                if ready
                else PARTIAL_STATUS
            ),
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate terminal schema, checksum, closed values, and readiness claims."""

    errors: list[str] = []
    if set(artifact) != set(FIELD_PRINCIPLES):
        errors.append("top-level field set mismatch")
    if set(artifact.get("field_principles") or {}) != set(artifact):
        errors.append("field principles mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("honest verdict prefix mismatch")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    blocked = artifact.get("status") == BLOCKED_STATUS
    if blocked:
        if (
            artifact.get("per_unit_rows") != []
            or artifact.get("operational_saturation_corpus_ready") is not False
        ):
            errors.append("blocked artifact contains headline rows")
        gate = artifact.get("gate_check_summary") or {}
        if gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("blocked gate receipt mismatch")
        if (
            artifact.get("verdict_class") != "blocked"
            or artifact.get("honest_verdict") != BLOCKED_STATUS
        ):
            errors.append("blocked verdict mismatch")
        return errors
    rows = artifact.get("per_unit_rows") or []
    coverage = artifact.get("row_coverage") or {}
    receipts = [
        receipt
        for receipt in (artifact.get("process_receipts") or [])
        if receipt.get("purpose", "corpus") == "corpus"
    ]
    try:
        fixture = json.loads((REPO_ROOT / FIXTURE_PATH).read_bytes())
        row_validation_errors = validate_rows(
            rows,
            fixture,
            artifact.get("MODEL_SPECS") or [],
            artifact.get("process_receipts") or [],
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        row_validation_errors = [{"row_id": None, "errors": ["fixture unavailable"]}]
    derived_ready = (
        len(rows) == 900
        and coverage.get("observed_unique") == 900
        and not coverage.get("duplicates")
        and not coverage.get("missing")
        and (artifact.get("budget_parity") or {}).get("passed") is True
        and (artifact.get("checkpoint_manifest") or {}).get("valid") is True
        and {receipt.get("model_id") for receipt in receipts}
        == {model["hub_id"] for model in PLANNED_MODELS}
        and all(receipt.get("authentic") is True for receipt in receipts)
        and not validate_cross_model_isolation(rows, artifact.get("process_receipts") or [])
        and not row_validation_errors
    )
    if artifact.get("operational_saturation_corpus_ready") is not derived_ready:
        errors.append("readiness is not evidence-derived")
    if artifact.get("exact_scores") != _score_summary(rows):
        errors.append("exact scores are not row-derived")
    if (artifact.get("gate_check_summary") or {}).get("passed") is not derived_ready:
        errors.append("gate summary mismatch")
    return errors


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace the selected artifact after validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class LivePhaseError(RuntimeError):
    """Retain a named live gate and all evidence collected before failure."""

    def __init__(self, check: str, observed: Any, receipt: Mapping[str, Any]) -> None:
        super().__init__(str(observed))
        self.check = check
        self.observed = observed
        self.receipt = deepcopy(dict(receipt))


def _free_port() -> int:  # pragma: no cover - live socket boundary.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _server_command(
    server: Path, model: Mapping[str, Any], port: int
) -> list[str]:  # pragma: no cover
    return [
        str(server),
        "--model",
        str(model["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(DECODE_SETTINGS["context_size"]),
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


def _health(port: int) -> bool:  # pragma: no cover - live HTTP boundary.
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.75) as response:  # noqa: S310
            value = json.loads(response.read())
            return response.status == 200 and value.get("status") == "ok"
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _stream_generation(
    port: int, prompt: bytes, *, max_tokens: int
) -> JsonDict:  # pragma: no cover
    payload = request_payload(prompt.decode("utf-8"), RANDOM_SEED, max_tokens=max_tokens)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=canonical_bytes(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    output_parts: list[str] = []
    raw_api: list[bytes] = []
    first = b""
    final = b""
    usage: Mapping[str, Any] = {}
    finish_reason = "unknown"
    with urllib.request.urlopen(  # noqa: S310
        request, timeout=float(DECODE_SETTINGS["request_timeout_s"])
    ) as response:
        while True:
            line = response.readline()
            if not line:
                break
            raw_api.append(line)
            stripped = line.strip()
            if not stripped.startswith(b"data:"):
                continue
            data = stripped[5:].strip()
            if data == b"[DONE]":
                continue
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            choice = (chunk.get("choices") or [{}])[0]
            piece = (choice.get("delta") or {}).get("content")
            if isinstance(piece, str) and piece:
                encoded = piece.encode()
                first = first or encoded
                final = encoded
                output_parts.append(piece)
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    raw_output = "".join(output_parts).encode()
    return {
        "raw_output": raw_output,
        "raw_api_sha256": sha256_bytes(b"".join(raw_api)),
        "first_token_b64": base64.b64encode(first).decode("ascii"),
        "final_token_b64": base64.b64encode(final).decode("ascii"),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "generated_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": round(time.monotonic() - started, 6),
        "finish_reason": finish_reason,
    }


def _probe_lease(device: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live lease boundary.
    try:
        lease = GpuLease.acquire(
            runtime_dir=Path("/tmp/carnot-gpu-leases"),
            task_id="exp6833-preflight",
            device_uuid=str(device["uuid"]),
            expected_model="preflight",
            vram_before_mb=int(device["memory_used_mb"]),
            ttl_s=60,
        )
        owner = lease.owner_receipt()
        lease.transition("terminal_blocked")
        release = lease.release()
        return {"available": True, "owner": owner, "release": release}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _process_start_time_from_ticks(start_ticks: int) -> str:  # pragma: no cover
    """Convert the kernel's boot-relative process time to an ISO timestamp."""

    boot_seconds = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines()
        if line.startswith("btime ")
    )
    clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    timestamp = boot_seconds + int(start_ticks) / clock_ticks
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _find_interrupted_server_log(
    *, model: Mapping[str, Any], port: int, temp_root: Path
) -> tuple[Path, bytes]:  # pragma: no cover
    """Find one orphaned task log that names the exact model and port."""

    model_path = str(model.get("model_path") or "").encode()
    port_text = f"127.0.0.1:{int(port)}".encode()
    matches: list[tuple[Path, bytes]] = []
    for path in temp_root.glob("exp6833-llama-*/stderr.bin"):
        try:
            value = path.read_bytes()
        except OSError:
            continue
        if model_path in value and port_text in value:
            matches.append((path, value))
    if len(matches) != 1:
        raise ValueError(f"interrupted server log count: {len(matches)}")
    return matches[0]


def recover_checkpoint_process_receipts(  # pragma: no cover - live restart boundary.
    *,
    checkpoint: RowCheckpoint,
    models: Sequence[Mapping[str, Any]],
    server: Path,
    runtime_dir: Path = Path("/tmp/carnot-gpu-leases"),
    temp_root: Path | None = None,
) -> list[JsonDict]:
    """Complete interrupted receipts from task-owned journal and log evidence."""

    models_by_id = {str(model["hub_id"]): model for model in models}
    receipts_by_session = {
        str(receipt.get("session_id")): receipt for receipt in checkpoint.process_receipts
    }
    row_sessions = {
        str((row.get("process_identity") or {}).get("session_id") or "") for row in checkpoint.rows
    }
    incomplete_sessions = sorted(
        session
        for session in row_sessions
        if session and receipts_by_session.get(session, {}).get("authentic") is not True
    )
    recovered_receipts: list[JsonDict] = []
    inventory = _gpu_inventory()
    devices_by_uuid = {str(device["uuid"]): device for device in inventory}
    for session_id in incomplete_sessions:
        rows = [
            row
            for row in checkpoint.rows
            if str((row.get("process_identity") or {}).get("session_id") or "") == session_id
        ]
        model_ids = {str(row.get("model_id") or "") for row in rows}
        if len(model_ids) != 1 or next(iter(model_ids)) not in models_by_id:
            raise ValueError("interrupted receipt model identity mismatch")
        model = models_by_id[next(iter(model_ids))]
        process = rows[0]["process_identity"]
        process_pid = int(process["pid"])
        process_ticks = int(process["pid_start_ticks"])
        if process_start_matches(process_pid, process_ticks):
            raise ValueError("interrupted server process is still live")
        device_uuid = str(process["physical_gpu_uuid"])
        device = devices_by_uuid.get(device_uuid)
        if device is None:
            raise ValueError("interrupted receipt GPU is unavailable")
        journal_path = journal_path_for(runtime_dir, device_uuid)
        journal = read_journal(journal_path)
        owner = journal["owner"]
        if process_start_matches(int(owner["pid"]), int(owner["pid_start_ticks"])):
            raise ValueError("interrupted lease owner is still live")
        current_sample = _gpu_snapshot(int(device["index"]), process_pid)
        memory_recovered = (
            abs(
                int(current_sample.get("memory_used_mb", 0) or 0)
                - int((journal.get("vram_mb") or {}).get("before", 0) or 0)
            )
            <= VRAM_RECOVERY_TOLERANCE_MB
        )
        log_path, server_log = _find_interrupted_server_log(
            model=model,
            port=int(process["port"]),
            temp_root=temp_root or Path(tempfile.gettempdir()),
        )
        recovery_started = time.monotonic()
        lease = GpuLease.acquire(
            runtime_dir=runtime_dir,
            task_id=f"exp6833-recover-{model['family_id']}",
            device_uuid=device_uuid,
            expected_model=str(model["hub_id"]),
            vram_before_mb=int(current_sample.get("memory_used_mb", 0) or 0),
            ttl_s=60,
        )
        owner_receipt = lease.owner_receipt()
        recovery = owner_receipt.get("recovery") or {}
        if (
            recovery.get("performed") is not True
            or recovery.get("previous_checksum") != journal.get("checksum")
        ):
            lease.transition("terminal_blocked")
            lease.release()
            raise ValueError("stale lease recovery receipt mismatch")
        lease.transition("terminal_blocked")
        release = lease.release()
        release["recovery"] = recovery
        receipt = build_recovered_process_receipt(
            rows=rows,
            model=model,
            server=server,
            journal=journal,
            server_log=server_log,
            server_log_path=log_path,
            lease_release=release,
            device_index=int(device["index"]),
            process_start_time=_process_start_time_from_ticks(process_ticks),
            teardown_duration_s=time.monotonic() - recovery_started,
            process_absent_after_exit=not Path(f"/proc/{process_pid}").exists(),
            gpu_memory_recovered=memory_recovered,
        )
        if receipt["authentic"] is not True:
            raise ValueError("recovered process receipt is incomplete")
        checkpoint.upsert_process_receipt(receipt)
        recovered_receipts.append(receipt)
    return recovered_receipts


def collect_preconditions(  # pragma: no cover - live host boundary.
    root: Path, fixture: Mapping[str, Any], models: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], JsonDict | None, Path]:
    checks: list[JsonDict] = [
        {
            "check": "operational_saturation_fixture_ready",
            "expected": True,
            "observed": fixture.get("operational_saturation_fixture_ready"),
            "passed": fixture.get("operational_saturation_fixture_ready") is True,
        }
    ]
    for model, planned in zip(models, PLANNED_MODELS, strict=True):
        errors = model_record_errors(model, planned)
        checks.append(
            {
                "check": f"model_identity_and_metadata:{planned['hub_id']}",
                "expected": [],
                "observed": errors,
                "passed": not errors,
            }
        )
    server = _llama_server_path()
    try:
        version = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, timeout=20, check=False
        )
        server_observed: Any = {
            "returncode": version.returncode,
            "stdout": version.stdout.strip(),
            "stderr": version.stderr.strip(),
        }
        server_ready = server.is_file() and os.access(server, os.X_OK) and version.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        server_observed = f"{type(exc).__name__}: {exc}"
        server_ready = False
    checks.append(
        {
            "check": "llama_cpp_executable",
            "expected": 0,
            "observed": server_observed,
            "passed": server_ready,
        }
    )
    devices = _gpu_inventory()
    device = max(devices, key=lambda row: int(row["memory_free_mb"]), default=None)
    required_mb = max(
        (int(model.get("model_size_bytes") or 0) // 1024**2 + 2048 for model in models), default=0
    )
    checks.append(
        {
            "check": "cuda_and_vram",
            "expected": {"device_count_at_least": 1, "free_vram_mb_at_least": required_mb},
            "observed": {"devices": devices, "selected": device},
            "passed": device is not None and int(device["memory_free_mb"]) >= required_mb,
        }
    )
    lease = (
        _probe_lease(device) if device is not None else {"available": False, "error": "no device"}
    )
    checks.append(
        {
            "check": "exclusive_task_owned_gpu_lease",
            "expected": True,
            "observed": lease,
            "passed": lease.get("available") is True,
        }
    )
    disk_free = shutil.disk_usage(root).free
    checks.append(
        {
            "check": "free_disk_bytes",
            "expected": {"at_least": DISK_FREE_FLOOR_BYTES},
            "observed": disk_free,
            "passed": disk_free >= DISK_FREE_FLOOR_BYTES,
        }
    )
    ports = [_free_port() for _ in range(6)]
    checks.append(
        {
            "check": "free_task_ports",
            "expected": 6,
            "observed": ports,
            "passed": len(set(ports)) == 6,
        }
    )
    return checks, deepcopy(device), server


def run_live_phase(  # pragma: no cover - required local CUDA E2E.
    *,
    root: Path,
    server: Path,
    device: Mapping[str, Any],
    model: Mapping[str, Any],
    purpose: str,
    work: Sequence[tuple[Mapping[str, Any], str]],
    checkpoint: RowCheckpoint | None,
    checker_sha256: str,
) -> tuple[list[JsonDict], JsonDict, list[JsonDict], list[JsonDict], JsonDict]:
    port = _free_port()
    command = _server_command(server, model, port)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device["index"])
    visible_devices = [int(device["index"])]
    process: subprocess.Popen[bytes] | None = None
    lease: Any = None
    phase_start = time.monotonic()
    process_started_at = ""
    process_start_ticks = 0
    offloaded_layers = 0
    total_layers: int | None = None
    baseline = _gpu_snapshot(int(device["index"]), 0)
    resident: JsonDict = {}
    after: JsonDict = {}
    samples = [deepcopy(baseline)]
    rows: list[JsonDict] = []
    checkpoint_receipts: list[JsonDict] = []
    batch_clocks: JsonDict = {}
    first_token = ""
    final_token = ""
    exit_code: int | None = None
    teardown_complete = False
    teardown_duration_s = 0.0
    error = ""
    lease_owner: JsonDict | None = None
    lease_release: JsonDict | None = None
    active_receipt: JsonDict | None = None
    server_stderr_sha256 = ""
    server_stdout_sha256 = ""
    with tempfile.TemporaryDirectory(prefix="exp6833-llama-") as temporary:
        stderr_path = Path(temporary) / "stderr.bin"
        stdout_path = Path(temporary) / "stdout.bin"
        try:
            lease = GpuLease.acquire(
                runtime_dir=Path("/tmp/carnot-gpu-leases"),
                task_id=f"exp6833-{purpose}-{model['family_id']}",
                device_uuid=str(device["uuid"]),
                expected_model=str(model["hub_id"]),
                vram_before_mb=int(baseline.get("memory_used_mb", 0) or 0),
                ttl_s=1800,
            )
            lease_owner = lease.owner_receipt()
            lease.transition("admitted")
            lease.transition("loading")
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                )
            process_started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            process_start_ticks = _pid_start_ticks(process.pid)
            deadline = time.monotonic() + float(DECODE_SETTINGS["load_timeout_s"])
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"llama-server exited during load: {process.returncode}")
                if _health(port):
                    break
                time.sleep(0.5)
            else:
                raise TimeoutError("llama-server load timeout")
            for _ in range(80):
                offloaded_layers, total_layers = _offload_layers(stderr_path.read_bytes())
                resident = _gpu_snapshot(int(device["index"]), process.pid)
                if offloaded_layers > 0 and resident.get("model_pid_present") is True:
                    break
                time.sleep(0.25)
            if offloaded_layers <= 0 or resident.get("model_pid_present") is not True:
                raise RuntimeError("CUDA offload receipt missing")
            samples.append(deepcopy(resident))
            lease.transition("resident", vram_mb=int(resident.get("memory_used_mb", 0) or 0))
            lease.transition("inferencing")
            active_receipt = {
                "purpose": purpose,
                "model_id": model["hub_id"],
                "model_sha256": model["model_sha256"],
                "session_id": f"exp6833-{purpose}-{model['family_id']}-{process_start_ticks}",
                "command": command,
                "pid": process.pid,
                "process_start_time": process_started_at,
                "pid_start_ticks": process_start_ticks,
                "port": port,
                "physical_gpu_uuid": device["uuid"],
                "visible_devices": visible_devices,
                "first_token_b64": "",
                "final_token_b64": "",
                "lease_owned": True,
                "lease_released": False,
                "cuda_offload": True,
                "offloaded_layers": offloaded_layers,
                "total_layers": total_layers,
                "process_exit_code": None,
                "process_absent_after_exit": False,
                "teardown_complete": False,
                "teardown_duration_s": 0.0,
                "duration_s": 0.0,
                "error": "active",
                "receipt_status": "active",
                "server_log_path": str(stderr_path),
                "authentic": False,
            }
            if checkpoint is not None and purpose == "corpus":
                checkpoint.upsert_process_receipt(active_receipt)
            pending_batch: list[JsonDict] = []
            for index, (scenario, arm) in enumerate(work):
                batch_started = time.monotonic()
                prompt = str(scenario["prompts"][arm]).encode()
                generated = _stream_generation(
                    port,
                    prompt,
                    max_tokens=1
                    if purpose == "canary"
                    else int(DECODE_SETTINGS["max_output_tokens"]),
                )
                first_token = first_token or str(generated["first_token_b64"])
                final_token = str(generated["final_token_b64"])
                if not first_token:
                    raise RuntimeError("first token missing")
                if purpose == "corpus":
                    row = build_scored_row(
                        scenario=scenario,
                        arm=arm,
                        prompt=prompt,
                        raw_output=generated["raw_output"],
                        model=model,
                        process_receipt={
                            "session_id": f"exp6833-{purpose}-{model['family_id']}-{process_start_ticks}",
                            "pid": process.pid,
                            "pid_start_ticks": process_start_ticks,
                            "port": port,
                            "physical_gpu_uuid": device["uuid"],
                        },
                        prompt_tokens=generated["prompt_tokens"],
                        generated_tokens=generated["generated_tokens"],
                        latency_s=generated["latency_s"],
                        checker_sha256=checker_sha256,
                    )
                    rows.append(row)
                    pending_batch.append(row)
                    if checkpoint is not None and (
                        len(pending_batch) >= BATCH_SIZE or index == len(work) - 1
                    ):
                        if active_receipt is not None:
                            active_receipt.update(
                                {
                                    "first_token_b64": first_token,
                                    "final_token_b64": final_token,
                                    "duration_s": round(time.monotonic() - phase_start, 6),
                                    "last_checkpoint_row_id": row["row_id"],
                                }
                            )
                            checkpoint.upsert_process_receipt(active_receipt)
                        receipt = checkpoint.append_batch(pending_batch)
                        receipt["model_id"] = model["hub_id"]
                        receipt["batch_index"] = len(checkpoint_receipts)
                        checkpoint_receipts.append(receipt)
                        pending_batch = []
                batch_clocks[f"batch_{index // BATCH_SIZE:03d}"] = round(
                    batch_clocks.get(f"batch_{index // BATCH_SIZE:03d}", 0.0)
                    + time.monotonic()
                    - batch_started,
                    6,
                )
                if (index + 1) % BATCH_SIZE == 0:
                    lease.heartbeat()
                    samples.append(_gpu_snapshot(int(device["index"]), process.pid))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            teardown_started = time.monotonic()
            if lease is not None:
                try:
                    if lease.document.get("phase") in {"resident", "inferencing"}:
                        lease.transition("unloading")
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            exit_code = terminate_owned_process(
                process, timeout_s=float(DECODE_SETTINGS["teardown_timeout_s"])
            )
            pid = process.pid if process is not None else 0
            deadline = time.monotonic() + float(DECODE_SETTINGS["teardown_timeout_s"])
            while time.monotonic() < deadline:
                after = _gpu_snapshot(int(device["index"]), pid)
                recovered = (
                    abs(
                        int(after.get("memory_used_mb", 0) or 0)
                        - int(baseline.get("memory_used_mb", 0) or 0)
                    )
                    <= VRAM_RECOVERY_TOLERANCE_MB
                )
                if (
                    not Path(f"/proc/{pid}").exists()
                    and after.get("model_pid_present") is False
                    and recovered
                ):
                    teardown_complete = True
                    break
                time.sleep(0.5)
            samples.append(deepcopy(after))
            if lease is not None:
                try:
                    phase = str(lease.document.get("phase"))
                    if phase == "loading":
                        lease.transition("terminal_blocked")
                    elif phase == "unloading":
                        lease.transition(
                            "validating",
                            vram_mb=int(after.get("memory_used_mb", 0) or 0),
                            exit_code=int(exit_code if exit_code is not None else 1),
                            unload_observed=not Path(f"/proc/{pid}").exists(),
                        )
                        lease.transition(
                            "terminal_complete"
                            if teardown_complete and not error
                            else "terminal_blocked"
                        )
                    lease_release = lease.release()
                except Exception as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            teardown_duration_s = time.monotonic() - teardown_started
            if stderr_path.is_file():
                server_stderr_sha256 = sha256_file(stderr_path)
            if stdout_path.is_file():
                server_stdout_sha256 = sha256_file(stdout_path)
    receipt: JsonDict = {
        "purpose": purpose,
        "model_id": model["hub_id"],
        "model_sha256": model["model_sha256"],
        "session_id": f"exp6833-{purpose}-{model['family_id']}-{process_start_ticks}",
        "command": command,
        "pid": process.pid if process is not None else 0,
        "process_start_time": process_started_at,
        "pid_start_ticks": process_start_ticks,
        "port": port,
        "physical_gpu_uuid": device["uuid"],
        "visible_devices": visible_devices,
        "first_token_b64": first_token,
        "final_token_b64": final_token,
        "lease_owned": lease_owner is not None,
        "lease_released": lease_release is not None and lease_release.get("released") is True,
        "cuda_offload": offloaded_layers > 0,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "process_exit_code": exit_code,
        "process_absent_after_exit": process is not None
        and not Path(f"/proc/{process.pid}").exists(),
        "teardown_complete": teardown_complete,
        "teardown_duration_s": round(teardown_duration_s, 6),
        "duration_s": round(time.monotonic() - phase_start, 6),
        "error": error or None,
        "phase_complete": not error,
        "receipt_status": "complete" if not error else "interrupted_complete_evidence",
        "server_stderr_sha256": server_stderr_sha256,
        "server_stdout_sha256": server_stdout_sha256,
    }
    receipt["authentic"] = bool(
        receipt["pid"] > 1
        and receipt["pid_start_ticks"] > 0
        and receipt["first_token_b64"]
        and receipt["final_token_b64"]
        and receipt["lease_owned"]
        and receipt["lease_released"]
        and receipt["cuda_offload"]
        and receipt["process_absent_after_exit"]
        and receipt["teardown_complete"]
    )
    if checkpoint is not None and purpose == "corpus":
        checkpoint.upsert_process_receipt(receipt)
    if not receipt["authentic"] or error:
        raise LivePhaseError(
            f"{purpose}_phase:{model['hub_id']}", error or "incomplete receipt", receipt
        )
    return rows, receipt, checkpoint_receipts, samples, batch_clocks


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in checks if row.get("passed") is not True), None)


def _code_receipts(root: Path) -> JsonDict:  # pragma: no cover - live file boundary.
    return {
        "module": {"path": str(MODULE_PATH), "sha256": sha256_file(root / MODULE_PATH)},
        "wrapper": {"path": str(WRAPPER_PATH), "sha256": sha256_file(root / WRAPPER_PATH)},
    }


def run(run_date: str, root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - terminal CUDA E2E.
    started = time.monotonic()
    clocks: JsonDict = {}
    preflight_started = time.monotonic()
    models = resolve_model_specs()
    try:
        fixture_bytes = (root / FIXTURE_PATH).read_bytes()
        fixture = json.loads(fixture_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        checks = [
            {
                "check": "operational_saturation_fixture",
                "expected": "readable Exp6832 JSON artifact",
                "observed": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        ]
        clocks["preflight"] = round(time.monotonic() - preflight_started, 6)
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check="operational_saturation_fixture",
            expected="readable Exp6832 JSON artifact",
            observed=checks[0]["observed"],
            preconditions=checks,
            models=models,
            duration_s=time.monotonic() - started,
            phase_clocks=clocks,
        )
        _write_json(root / OUTPUT_PATH, artifact)
        return artifact
    manifest = build_manifest(fixture)
    checkpoint_started = time.monotonic()
    try:
        checkpoint = RowCheckpoint(root / CHECKPOINT_PATH, manifest["manifest_sha256"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        clocks["checkpoint_load"] = round(time.monotonic() - checkpoint_started, 6)
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check="checkpoint_manifest_identity",
            expected=manifest["manifest_sha256"],
            observed=observed,
            preconditions=[
                {
                    "check": "checkpoint_manifest_identity",
                    "expected": manifest["manifest_sha256"],
                    "observed": observed,
                    "passed": False,
                }
            ],
            models=models,
            duration_s=time.monotonic() - started,
            phase_clocks=clocks,
        )
        _write_json(root / OUTPUT_PATH, artifact)
        return artifact
    clocks["checkpoint_load"] = round(time.monotonic() - checkpoint_started, 6)
    recovery_started = time.monotonic()
    try:
        recovered_receipts = recover_checkpoint_process_receipts(
            checkpoint=checkpoint,
            models=models,
            server=_llama_server_path(),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        observed = f"{type(exc).__name__}: {exc}"
        clocks["checkpoint_receipt_recovery"] = round(
            time.monotonic() - recovery_started, 6
        )
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check="checkpoint_process_receipt_recovery",
            expected="checksummed task-owned lease and server-log evidence",
            observed=observed,
            preconditions=[
                {
                    "check": "checkpoint_process_receipt_recovery",
                    "expected": "checksummed task-owned lease and server-log evidence",
                    "observed": observed,
                    "passed": False,
                }
            ],
            models=models,
            duration_s=time.monotonic() - started,
            phase_clocks=clocks,
        )
        _write_json(root / OUTPUT_PATH, artifact)
        return artifact
    clocks["checkpoint_receipt_recovery"] = round(time.monotonic() - recovery_started, 6)
    checks, device, server = collect_preconditions(root, fixture, models)
    checks.append(
        {
            "check": "checkpoint_process_receipt_recovery",
            "expected": "all retained row sessions have authentic receipts",
            "observed": {
                "recovered_session_ids": [
                    receipt["session_id"] for receipt in recovered_receipts
                ],
                "retained_row_count": len(checkpoint.rows),
            },
            "passed": True,
        }
    )
    clocks["preflight"] = round(time.monotonic() - preflight_started, 6)
    failed = _first_failed(checks)
    if failed is not None or device is None:
        failed = failed or {
            "check": "cuda_and_vram",
            "expected": "eligible device",
            "observed": None,
        }
        artifact = build_blocked_artifact(
            run_date=run_date,
            failed_check=str(failed["check"]),
            expected=failed.get("expected"),
            observed=failed.get("observed"),
            preconditions=checks,
            models=models,
            duration_s=time.monotonic() - started,
            phase_clocks=clocks,
        )
        _write_json(root / OUTPUT_PATH, artifact)
        return artifact
    process_receipts: list[JsonDict] = []
    accelerator_samples: list[JsonDict] = []
    canary_scenario = fixture["scenarios"][0]
    for model in models:
        phase_started = time.monotonic()
        try:
            _, receipt, _, samples, _ = run_live_phase(
                root=root,
                server=server,
                device=device,
                model=model,
                purpose="canary",
                work=[(canary_scenario, ARMS[0])],
                checkpoint=None,
                checker_sha256=str(fixture["implementation_hashes"]["module"]["sha256"]),
            )
            process_receipts.append(receipt)
            accelerator_samples.extend(samples)
            clocks[f"teardown:canary:{model['hub_id']}"] = receipt["teardown_duration_s"]
            checks.append(
                {
                    "check": f"live_one_token_canary:{model['hub_id']}",
                    "expected": True,
                    "observed": receipt,
                    "passed": True,
                }
            )
            clocks[f"canary:{model['hub_id']}"] = round(time.monotonic() - phase_started, 6)
        except LivePhaseError as exc:
            process_receipts.append(exc.receipt)
            checks.append(
                {
                    "check": exc.check,
                    "expected": "authentic one-token CUDA canary",
                    "observed": exc.observed,
                    "passed": False,
                }
            )
            artifact = build_blocked_artifact(
                run_date=run_date,
                failed_check=exc.check,
                expected="authentic one-token CUDA canary",
                observed=exc.observed,
                preconditions=checks,
                models=models,
                duration_s=time.monotonic() - started,
                phase_clocks=clocks,
                process_receipts=process_receipts,
                accelerator_samples=accelerator_samples,
            )
            _write_json(root / OUTPUT_PATH, artifact)
            return artifact
    all_rows = deepcopy(checkpoint.rows)
    checkpoint_receipts: list[JsonDict] = []
    completed = checkpoint.completed_ids
    process_receipts.extend(deepcopy(checkpoint.process_receipts))
    for model in models:
        work = [
            (scenario, arm)
            for scenario in fixture["scenarios"]
            for arm in ARMS
            if row_identity(str(model["hub_id"]), str(scenario["scenario_id"]), arm)
            not in completed
        ]
        if not work:
            clocks[f"model:{model['hub_id']}"] = 0.0
            continue
        phase_started = time.monotonic()
        try:
            new_rows, receipt, publishes, samples, batches = run_live_phase(
                root=root,
                server=server,
                device=device,
                model=model,
                purpose="corpus",
                work=work,
                checkpoint=checkpoint,
                checker_sha256=str(fixture["implementation_hashes"]["module"]["sha256"]),
            )
            all_rows.extend(new_rows)
            process_receipts.append(receipt)
            checkpoint_receipts.extend(publishes)
            checkpoint_receipts.append(checkpoint.record_process_receipt(receipt))
            accelerator_samples.extend(samples)
            clocks[f"model:{model['hub_id']}"] = round(time.monotonic() - phase_started, 6)
            clocks[f"teardown:corpus:{model['hub_id']}"] = receipt["teardown_duration_s"]
            for name, duration in batches.items():
                clocks[f"batch:{model['family_id']}:{name}"] = duration
            completed.update(row["row_id"] for row in new_rows)
        except LivePhaseError as exc:
            process_receipts.append(exc.receipt)
            artifact = build_blocked_artifact(
                run_date=run_date,
                failed_check=exc.check,
                expected="complete authentic local llama.cpp CUDA corpus phase",
                observed=exc.observed,
                preconditions=checks,
                models=models,
                duration_s=time.monotonic() - started,
                phase_clocks=clocks,
                process_receipts=process_receipts,
                accelerator_samples=accelerator_samples,
            )
            _write_json(root / OUTPUT_PATH, artifact)
            return artifact
    scoring_started = time.monotonic()
    all_rows = deepcopy(checkpoint.rows)
    clocks["scoring"] = round(time.monotonic() - scoring_started, 6)
    artifact = assemble_artifact(
        run_date=run_date,
        fixture=fixture,
        manifest=manifest,
        models=models,
        rows=all_rows,
        process_receipts=process_receipts,
        preconditions=checks,
        accelerator_samples=accelerator_samples,
        checkpoint_receipts=checkpoint_receipts or [{"durable": True, "resumed_complete": True}],
        duration_s=time.monotonic() - started,
        phase_clocks=clocks,
        code_receipts=_code_receipts(root),
        fixture_file_sha256=sha256_bytes(fixture_bytes),
    )
    verify_started = time.monotonic()
    errors = validate_artifact(artifact)
    clocks["verify"] = round(time.monotonic() - verify_started, 6)
    artifact["phase_clocks"] = clocks
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    if errors:
        raise ValueError("; ".join(errors))
    write_started = time.monotonic()
    _write_json(root / OUTPUT_PATH, artifact)
    artifact["phase_clocks"]["write"] = round(time.monotonic() - write_started, 6)
    _write_json(root / OUTPUT_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin command boundary.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    arguments = parser.parse_args(argv)
    if re.fullmatch(r"\d{8}", arguments.date) is None:
        parser.error("--date must use YYYYMMDD")
    datetime.strptime(arguments.date, "%Y%m%d")
    artifact = run(arguments.date)
    print(json.dumps({"status": artifact["status"], "honest_verdict": artifact["honest_verdict"]}))
    return 0 if artifact["operational_saturation_corpus_ready"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
