"""Capture blind symbolic traces with the mandated Qwen3.8 GGUF.

The model worker reads only Exp7180's three-field generation view. It records
transport and parsing evidence, but it never reads labels or scores correctness.

Spec refs: REQ-VERIFY-7181 and SCENARIO-VERIFY-7181-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any, Iterator
from urllib import error, request

from carnot import experiment_7160_v631_qwen38_lease_diagnosis as lease_preflight
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_7150_v628_grounding_preflight import _gpu_snapshot
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    LiveProcessOps,
    NativeLlamaServerSupervisor,
    canonical_json,
    cleanup_recorded_identity,
    read_process_identity,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_current_model
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260910"
TASK_ID = "experiment_7181_v633_qwen38_symbolic_traces"
RANDOM_SEED = 7_181_202_609_10
RESULT_PATH = Path("results/experiment_7181_v633_qwen38_symbolic_traces.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7181_v633_qwen38_symbolic_traces")
RAW_DIR = Path("results/raw/experiment_7181_v633_qwen38_symbolic_traces")
RAW_MANIFEST_NAME = "raw_manifest.json"
GENERATION_VIEW_PATH = Path(
    "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
)
FIXTURE_ARTIFACT_PATH = Path("results/experiment_7180_v633_symbolic_edit_fixture.json")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7181_v633_qwen38_symbolic_traces.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7181_v633_qwen38_symbolic_traces.py")
TEST_PATH = Path("tests/python/test_experiment_7181_v633_qwen38_symbolic_traces.py")
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]
PINNED_FIXTURE_SHA256 = "sha256:a76492f304e684d75ea25037261c90bd0367fa54819bb867b38fe2d96a00398a"
PINNED_GENERATION_SHA256 = "sha256:6927314707d5075ef7b48e469db14f8a0c67d54caf73de05a9c8c9d082cd628d"
EXP7180_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "inference_substrate": "exact_source_fixture_construction",
    "inference_substrate_class": "cpu_exact_solver_or_simulator",
    "execution_venue": "host",
    "fixture_ready_score": 1,
    "random_seed": 7_180_202_609_10,
    "reproducibility_checksum": (
        "sha256:72d4a6c940dc52825f0eac30cb623df727bd7f45e362dc14d8a611fa5536bec5"
    ),
    "verifier_is_oracle": False,
    "verdict_class": "positive",
    "honest_verdict": "complete_positive_symbolic_edit_fixture_ready_no_live_verifier_result",
    "generation_view_path": GENERATION_VIEW_PATH.as_posix(),
}
OUTPUT_TOKEN_BUDGET = 192
CANARY_TOKEN_BUDGET = 32
REQUEST_CAP_S = 120.0
MEASUREMENT_CAP_S = 2_100.0
TOTAL_CAP_S = 3_600.0
CONTEXT_TOKEN_BUDGET = 4_096
CHECKPOINT_CADENCE = 8
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": RANDOM_SEED & 0x7FFFFFFF,
    "cache_prompt": False,
    "max_tokens": OUTPUT_TOKEN_BUDGET,
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "trace_capture_complete_score",
    "MODEL_SPECS",
    "model_specs",
    "raw_manifest",
    "gpu_receipts",
    "phase_spans",
    "runner_receipt",
    "completion_rows",
)
FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use model_full_generation when the declared work runs; use blocked_no_run only before any qualifying work.",
    "trace_capture_complete_score": "One means complete scheduled capture and authentic runtime evidence, regardless of correctness.",
    "MODEL_SPECS": "The mandated Qwen3.8 GGUF identifies the real generator.",
    "model_specs": "Resolved path, revision, quantization, and hash identify the loaded bytes.",
    "raw_manifest": "Raw requests and completions permit independent replay.",
    "gpu_receipts": "Task-linked device and process measurements authenticate CUDA execution.",
    "phase_spans": "Monotonic spans separate loading, inference, tests, and cleanup.",
    "runner_receipt": "Record model_count=1 and the selected runner; two devices are not two models.",
    "completion_rows": "Every unit retains text, token budget, errors, and parse status.",
}

RESPONSE_FIELDS = {
    "direct_decision",
    "claim_tuple",
    "evidence_tuple",
    "source_start",
    "source_end",
    "missing_fields",
}
FORBIDDEN_WORKER_FIELDS = frozenset(
    {
        "split",
        "variant",
        "edit_label",
        "support_label",
        "expected_answer",
        "expected_response",
        "authority",
        "support_hash",
        "source_artifact_hashes",
    }
)
JSON_RESPONSE_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(RESPONSE_FIELDS),
    "properties": {
        "direct_decision": {
            "type": "string",
            "enum": ["supported", "unsupported", "abstain"],
        },
        "claim_tuple": {"$ref": "#/$defs/tuple"},
        "evidence_tuple": {"anyOf": [{"$ref": "#/$defs/tuple"}, {"type": "null"}]},
        "source_start": {"type": ["integer", "null"], "minimum": 0},
        "source_end": {"type": ["integer", "null"], "minimum": 0},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "tuple": {
            "type": "object",
            "additionalProperties": False,
            "required": ["subject", "relation", "object", "polarity", "quantity", "unit"],
            "properties": {
                "subject": {"type": "string"},
                "relation": {"type": "string"},
                "object": {"type": "string"},
                "polarity": {"type": "string", "enum": ["positive", "negative"]},
                "quantity": {"type": ["number", "null"]},
                "unit": {"type": ["string", "null"]},
            },
        }
    },
}


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so whitespace changes stay visible."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one stable JSON spelling for cross-process receipts."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash an ordinary source file without normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None = None,
    field: str | None = None,
) -> JsonDict:
    """Keep both sides and the source field for one reproducible gate."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every evidence field except the checksum that contains the hash."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def _prompt_evidence(prompt: str) -> str:
    """Return the exact evidence substring used to validate model spans."""

    try:
        return prompt.split("Evidence:\n", 1)[1].split("\n\nClaim:", 1)[0]
    except IndexError:  # pragma: no cover - schedule validation rejects this first.
        return ""


def build_schedule(generation_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze the public three-field rows without opening any private sidecar."""

    schedule: list[JsonDict] = []
    expected_keys = {"unit_id", "text", "response_schema"}
    for index, source in enumerate(generation_rows):
        if set(source) != expected_keys:
            raise ValueError(f"worker_view_shape:{index}")
        model_input = deepcopy(dict(source))
        prompt = str(model_input["text"])
        schedule.append(
            {
                "row_order": index,
                "unit_id": str(model_input["unit_id"]),
                "model_input": model_input,
                "model_input_sha256": sha256_json(model_input),
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "response_schema": deepcopy(model_input["response_schema"]),
                "response_schema_sha256": sha256_json(model_input["response_schema"]),
                "decoding_parameters": deepcopy(DECODING_PARAMETERS),
                "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
                "output_token_budget": OUTPUT_TOKEN_BUDGET,
            }
        )
    errors = schedule_errors(schedule)
    if errors:
        raise ValueError("schedule_invalid:" + ",".join(errors))
    return schedule


def schedule_errors(schedule: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject count, order, hash, schema, or worker blinding drift."""

    errors: list[str] = []
    if len(schedule) != 192:
        errors.append("schedule_row_count_mismatch")
    if [row.get("row_order") for row in schedule] != list(range(len(schedule))):
        errors.append("schedule_order_mismatch")
    if len({row.get("unit_id") for row in schedule}) != len(schedule):
        errors.append("schedule_unit_id_duplicate")
    for index, row in enumerate(schedule):
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping):  # pragma: no cover - defensive cold check.
            errors.append(f"row_{index}:model_input_invalid")
            continue
        if set(model_input) != {"unit_id", "text", "response_schema"}:
            errors.append(f"row_{index}:worker_view_shape")
        if FORBIDDEN_WORKER_FIELDS.intersection(model_input):
            errors.append(f"row_{index}:private_field_exposed")
        if row.get("unit_id") != model_input.get("unit_id"):
            errors.append(f"row_{index}:unit_id_mismatch")
        if row.get("model_input_sha256") != sha256_json(model_input):
            errors.append(f"row_{index}:model_input_hash_mismatch")
        prompt = str(row.get("prompt", ""))
        if prompt != model_input.get("text") or row.get("prompt_sha256") != sha256_text(prompt):
            errors.append(f"row_{index}:prompt_hash_mismatch")
        schema = model_input.get("response_schema")
        if row.get("response_schema") != schema or row.get("response_schema_sha256") != sha256_json(
            schema
        ):
            errors.append(f"row_{index}:response_schema_mismatch")
        if row.get("decoding_parameters") != DECODING_PARAMETERS or row.get(
            "decoding_parameters_sha256"
        ) != sha256_json(DECODING_PARAMETERS):
            errors.append(f"row_{index}:decoding_parameters_mismatch")
        if row.get("output_token_budget") != OUTPUT_TOKEN_BUDGET:
            errors.append(f"row_{index}:token_budget_mismatch")
    return list(dict.fromkeys(errors))


def _tuple_valid(value: Any) -> bool:
    """Check tuple fields without filling values the model omitted."""

    if not isinstance(value, Mapping):
        return False
    required = {"subject", "relation", "object", "polarity", "quantity", "unit"}
    return (
        set(value) == required
        and all(isinstance(value[field], str) for field in ("subject", "relation", "object"))
        and value["polarity"] in {"positive", "negative"}
        and (
            value["quantity"] is None
            or isinstance(value["quantity"], (int, float))
            and not isinstance(value["quantity"], bool)
        )
        and (value["unit"] is None or isinstance(value["unit"], str))
    )


def parse_structured_output(raw_output: str, prompt: str) -> JsonDict:
    """Parse one exact JSON object and never repair partial model text."""

    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return {"parse_status": "failed", "parse_error": "invalid_json", "parsed": None}
    if not isinstance(value, dict):  # pragma: no cover - exercised by cold validation inputs.
        return {"parse_status": "failed", "parse_error": "root_not_object", "parsed": None}
    if set(value) != RESPONSE_FIELDS:
        return {"parse_status": "failed", "parse_error": "field_set_mismatch", "parsed": None}
    if value["direct_decision"] not in {"supported", "unsupported", "abstain"}:
        return {
            "parse_status": "failed",
            "parse_error": "direct_decision_invalid",
            "parsed": None,
        }
    if not _tuple_valid(value["claim_tuple"]):
        return {"parse_status": "failed", "parse_error": "claim_tuple_invalid", "parsed": None}
    if value["evidence_tuple"] is not None and not _tuple_valid(value["evidence_tuple"]):
        return {
            "parse_status": "failed",
            "parse_error": "evidence_tuple_invalid",
            "parsed": None,
        }
    if not isinstance(value["missing_fields"], list) or not all(
        isinstance(field, str) for field in value["missing_fields"]
    ):
        return {"parse_status": "failed", "parse_error": "missing_fields_invalid", "parsed": None}
    start = value["source_start"]
    end = value["source_end"]
    evidence = _prompt_evidence(prompt)
    null_span = start is None and end is None
    integer_span = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start <= end <= len(evidence)
    )
    if not (null_span or integer_span):
        return {"parse_status": "failed", "parse_error": "source_span_invalid", "parsed": None}
    return {"parse_status": "valid", "parse_error": None, "parsed": value}


def build_completion_row(
    schedule_row: Mapping[str, Any],
    response: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> JsonDict:
    """Conserve exact request, completion, parse, timing, and ownership data."""

    prompt = str(schedule_row.get("prompt", ""))
    raw_output = str(response.get("raw_output", ""))
    raw_response = deepcopy(response.get("raw_response", {}))
    raw_request = deepcopy(response.get("raw_request", {"prompt": prompt}))
    parsed = parse_structured_output(raw_output, prompt)
    structured = parsed["parsed"] if isinstance(parsed["parsed"], Mapping) else {}
    request_error = response.get("error")
    completion_tokens = int(response.get("completion_tokens", 0) or 0)
    finish_reason = response.get("finish_reason")
    return {
        "row_order": schedule_row.get("row_order"),
        "unit_id": schedule_row.get("unit_id"),
        "terminal_state": "request_error" if request_error else "response",
        "prompt": prompt,
        "prompt_bytes_b64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        "prompt_sha256": sha256_text(prompt),
        "raw_request": raw_request,
        "raw_request_sha256": sha256_json(raw_request),
        "raw_output": raw_output,
        "completion_bytes_b64": base64.b64encode(raw_output.encode("utf-8")).decode("ascii"),
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response": raw_response,
        "raw_response_sha256": sha256_json(raw_response),
        "output_token_budget": schedule_row.get("output_token_budget"),
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "truncated": finish_reason in {"length", "max_tokens"}
        or completion_tokens >= int(schedule_row.get("output_token_budget", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "request_error": request_error,
        "parse_status": parsed["parse_status"],
        "parse_error": parsed["parse_error"],
        "direct_decision": structured.get("direct_decision"),
        "extracted_tuples": {
            "claim_tuple": deepcopy(structured.get("claim_tuple")),
            "evidence_tuple": deepcopy(structured.get("evidence_tuple")),
        },
        "source_span": {
            "start": structured.get("source_start"),
            "end": structured.get("source_end"),
        },
        "missing_fields": deepcopy(structured.get("missing_fields")),
        "server_pid": resource_receipt.get("server_pid"),
        "server_pid_start_ticks": resource_receipt.get("server_pid_start_ticks"),
        "gpu_uuid": resource_receipt.get("gpu_uuid"),
        "lease_id": resource_receipt.get("lease_id"),
    }


def completion_row_errors(row: Mapping[str, Any], schedule_row: Mapping[str, Any]) -> list[str]:
    """Recompute raw byte hashes and parser state for one terminal row."""

    errors: list[str] = []
    if row.get("row_order") != schedule_row.get("row_order"):
        errors.append("row_order_mismatch")
    if row.get("unit_id") != schedule_row.get("unit_id"):
        errors.append("unit_id_mismatch")
    prompt = str(row.get("prompt", ""))
    if prompt != schedule_row.get("prompt"):
        errors.append("prompt_mismatch")
    if row.get("prompt_sha256") != sha256_text(prompt):
        errors.append("prompt_hash_mismatch")
    if row.get("prompt_bytes_b64") != base64.b64encode(prompt.encode("utf-8")).decode("ascii"):
        errors.append("prompt_bytes_mismatch")
    raw_output = str(row.get("raw_output", ""))
    if row.get("raw_output_sha256") != sha256_text(raw_output):
        errors.append("raw_output_hash_mismatch")
    if row.get("completion_bytes_b64") != base64.b64encode(raw_output.encode("utf-8")).decode(
        "ascii"
    ):
        errors.append("completion_bytes_mismatch")
    if row.get("raw_request_sha256") != sha256_json(row.get("raw_request", {})):
        errors.append("raw_request_hash_mismatch")
    if row.get("raw_response_sha256") != sha256_json(row.get("raw_response", {})):
        errors.append("raw_response_hash_mismatch")
    reparsed = parse_structured_output(raw_output, prompt)
    if row.get("parse_status") != reparsed["parse_status"]:
        errors.append("parse_status_mismatch")
    if row.get("parse_error") != reparsed["parse_error"]:
        errors.append("parse_error_mismatch")
    if row.get("terminal_state") not in {"response", "request_error"}:
        errors.append("terminal_state_invalid")
    if not isinstance(row.get("latency_s"), (int, float)) or float(row.get("latency_s", -1)) < 0:
        errors.append("latency_invalid")
    return errors


def checkpoint_identity(
    schedule: Sequence[Mapping[str, Any]], generation_view_sha256: str, model_sha256: str
) -> JsonDict:
    """Bind resume to the schedule, public bytes, model, and decoding contract."""

    return {
        "schedule_sha256": sha256_json(list(schedule)),
        "generation_view_sha256": generation_view_sha256,
        "model_sha256": model_sha256,
        "model_specs_sha256": sha256_json(MODEL_SPECS),
        "response_schema_sha256": sha256_json(JSON_RESPONSE_SCHEMA),
        "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
        "random_seed": RANDOM_SEED,
    }


def write_checkpoint(
    path: Path, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Atomically write one complete eight-row checkpoint with row hashes."""

    if not rows or len(rows) % CHECKPOINT_CADENCE:
        raise ValueError("checkpoint_row_cadence")
    payload = {
        "schema": "carnot.exp7181.checkpoint.v1",
        "identity": deepcopy(dict(identity)),
        "row_count": len(rows),
        "row_hashes": [sha256_json(row) for row in rows],
        "rows": [deepcopy(dict(row)) for row in rows],
    }
    atomic_write_json(path, payload, allow_override=False, sort_keys=True)
    return payload


def resume_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> list[JsonDict]:
    """Reject a checkpoint if any frozen identity or row byte hash changed."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint_identity_mismatch")
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) % CHECKPOINT_CADENCE
        or payload.get("row_count") != len(rows)
    ):
        raise ValueError("checkpoint_row_cadence")
    if payload.get("row_hashes") != [sha256_json(row) for row in rows]:
        raise ValueError("checkpoint_row_hash_mismatch")
    return [deepcopy(dict(row)) for row in rows]


def cleanup_owned_process(
    recorded: JsonDict,
    current_identity: Callable[[int], JsonDict],
    process_ops: Any,
    contract: JsonDict,
) -> JsonDict:
    """Use the shipped identity guard so PID reuse cannot redirect cleanup."""

    return cleanup_recorded_identity(recorded, current_identity, process_ops, contract=contract)


def classify_terminal(
    checks: Sequence[Mapping[str, Any]],
    canary: Mapping[str, Any] | None,
    completion_rows: Sequence[Mapping[str, Any]],
    *,
    provenance_ok: bool,
) -> JsonDict:
    """Classify only the model work that produced real terminal receipts."""

    rows = list(completion_rows)
    any_failed_gate = any(row.get("passed") is not True for row in checks)
    if canary is None and not rows and any_failed_gate:
        return {"status": "blocked", "inference_substrate_class": "blocked_no_run", "score": 0}
    if not rows:
        return {
            "status": "partial",
            "inference_substrate_class": "model_bounded_generation",
            "score": 0,
        }
    complete = (
        len(rows) == 192
        and all(row.get("terminal_state") in {"response", "request_error"} for row in rows)
        and provenance_ok
        and not any_failed_gate
    )
    return {
        "status": "complete" if complete else "partial",
        "inference_substrate_class": "model_full_generation",
        "score": int(complete),
    }


def _source_artifact_hashes(
    root: Path, *, raw_manifest: Path | None = None, model_sha256: str | None = None
) -> JsonDict:
    """Bind public inputs, code, tests, and the raw replay manifest."""

    paths = {
        "exp7180_artifact": root / FIXTURE_ARTIFACT_PATH,
        "generation_view": root / GENERATION_VIEW_PATH,
        "constraint_spec": root / SPEC_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "focused_tests": root / TEST_PATH,
        "research_program": root / "research-program.md",
        "research_references": root / "research-references.md",
        "exclusion_manifest": root / "ops/exclusion_manifest.yaml",
        "e2e_test_plan": root / "ops/e2e-test-plan.md",
        "sota_models": root / "python/carnot/inference/sota_models.py",
        "native_supervisor": root / "python/carnot/inference/llama_server_supervisor.py",
        "gpu_lease": root / "python/carnot/gpu_lease_phase_journal.py",
    }
    result = {
        name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()
    }
    manifest = raw_manifest or root / RAW_DIR / RAW_MANIFEST_NAME
    result["raw_manifest"] = sha256_file(manifest) if manifest.is_file() else "missing"
    result["model_gguf"] = model_sha256 or "missing"
    result["decoding_contract"] = sha256_json(DECODING_PARAMETERS)
    return result


def base_artifact(run_date: str, *, root: Path | None = None) -> JsonDict:
    """Create every required field before a fallible resource check."""

    repository = root or Path(__file__).resolve().parents[2]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": "preflight_pending_native_llama_cpp",
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": _source_artifact_hashes(repository),
        "rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_row(
            "trace_capture_complete",
            True,
            False,
            False,
            upstream="experiment_7181",
            field="trace_capture_complete_score",
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_symbolic_trace_capture",
        "inference_substrate_class": "blocked_no_run",
        "trace_capture_complete_score": 0,
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "raw_manifest": {},
        "gpu_receipts": {},
        "phase_spans": [],
        "runner_receipt": {"model_count": 1, "runner": None},
        "completion_rows": [],
        "canary_receipt": None,
        "study_question": (
            "Can Qwen3.8 produce one blind structured symbolic trace for every frozen Exp7180 row?"
        ),
        "scope_answer": "No terminal model measurement is available yet.",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _load_artifact(value: Mapping[str, Any] | str | Path | object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, (str, Path)):
        return None
    try:
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def validate_artifact(
    value: Mapping[str, Any] | str | Path | object, *, check_source_hashes: bool = True
) -> list[str]:
    """Cold-check terminal class, schedule, raw rows, sources, and checksum."""

    artifact = _load_artifact(value)
    if artifact is None:  # pragma: no cover - CLI validation failure path.
        return ["artifact_unreadable"]
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("required_fields_missing:" + ",".join(missing))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("model_specs_mandate_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or float(artifact.get("duration_s", -1)) < 0
    ):
        errors.append("duration_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")

    checks = list(artifact.get("preconditions_checked", []))
    failed = next((row for row in checks if row.get("passed") is not True), None)
    status = artifact.get("status")
    if status == "blocked":
        if failed is None:
            errors.append("blocked_failed_gate_missing")
        elif artifact.get("gate_check_summary") != failed:
            errors.append("blocked_gate_summary_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("trace_capture_complete_score") != 0:
            errors.append("blocked_readiness_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_mismatch")
        return list(dict.fromkeys(errors))

    manifest = artifact.get("raw_manifest", {})
    schedule = list(manifest.get("schedule", [])) if isinstance(manifest, Mapping) else []
    errors.extend(schedule_errors(schedule))
    rows = list(artifact.get("completion_rows", []))
    if artifact.get("rows") != rows:
        errors.append("rows_projection_mismatch")
    for index, (row, sealed) in enumerate(zip(rows, schedule, strict=False)):
        errors.extend(
            f"row_{index}:{row_error}" for row_error in completion_row_errors(row, sealed)
        )
    provenance_ok = bool(dict(artifact.get("gpu_receipts", {})).get("provenance_ok"))
    expected = classify_terminal(
        checks, artifact.get("canary_receipt"), rows, provenance_ok=provenance_ok
    )
    if artifact.get("inference_substrate_class") != expected["inference_substrate_class"]:
        errors.append("terminal_substrate_class_mismatch")
    expected_score = int(
        expected["score"] == 1 and len(rows) == len(schedule) == 192 and not errors
    )
    if artifact.get("trace_capture_complete_score") != expected_score:
        errors.append("trace_capture_complete_score_mismatch")
    if expected_score and (status != "complete" or artifact.get("verdict_class") != "positive"):
        errors.append("complete_terminal_state_mismatch")
    if status == "complete" and not expected_score:
        errors.append("forged_complete_status")
    runner = artifact.get("runner_receipt", {})
    if not isinstance(runner, Mapping) or runner.get("model_count") != 1:
        errors.append("runner_model_count_mismatch")

    if check_source_hashes and isinstance(artifact.get("source_artifact_hashes"), Mapping):
        root = find_repo_root()
        manifest_path = Path(str(manifest.get("path", root / RAW_DIR / RAW_MANIFEST_NAME)))
        model_value = artifact.get("model_specs", {})
        if isinstance(model_value, Sequence) and not isinstance(model_value, (str, bytes)):
            model_hash = model_value[0].get("sha256") if model_value else None
        elif isinstance(model_value, Mapping):
            model_hash = model_value.get("sha256")
        else:  # pragma: no cover - schema error is reported by source mismatch.
            model_hash = None
        expected_sources = _source_artifact_hashes(
            root,
            raw_manifest=manifest_path,
            model_sha256=str(model_hash) if model_hash else None,
        )
        if artifact.get("source_artifact_hashes") != expected_sources:
            errors.append("source_artifact_hashes_mismatch")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Print one compact phase record and flush it immediately."""

    print(
        canonical_json({"experiment": 7181, "phase": phase, "event": event, **fields}), flush=True
    )


@contextmanager
def _phase_span(spans: list[JsonDict], phase: int, name: str) -> Iterator[None]:  # pragma: no cover
    """Measure one named phase with monotonic time and visible boundaries."""

    started = time.monotonic()
    _progress(phase, "phase_start", name=name)
    try:
        yield
    finally:
        ended = time.monotonic()
        spans.append(
            {
                "phase": phase,
                "name": name,
                "start_monotonic_s": started,
                "end_monotonic_s": ended,
                "duration_s": ended - started,
            }
        )
        _progress(phase, "phase_end", name=name, duration_s=round(ended - started, 6))


@contextmanager
def _heartbeat(
    phase: int, operation: str, completed: Callable[[], int], total: int
) -> Iterator[None]:  # pragma: no cover
    """Report progress outside a blocking native call at least every minute."""

    stop = threading.Event()
    started = time.monotonic()

    def emit() -> None:
        while not stop.wait(45.0):
            _progress(
                phase,
                "heartbeat",
                operation=operation,
                elapsed_s=round(time.monotonic() - started, 3),
                completed_units=completed(),
                total_units=total,
            )

    thread = threading.Thread(target=emit, name=f"exp7181-heartbeat-{phase}", daemon=True)
    thread.start()
    _progress(
        phase,
        "heartbeat_start",
        operation=operation,
        elapsed_s=0.0,
        completed_units=completed(),
        total_units=total,
    )
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        _progress(
            phase,
            "heartbeat_end",
            operation=operation,
            elapsed_s=round(time.monotonic() - started, 3),
            completed_units=completed(),
            total_units=total,
        )


@contextmanager
def _stream_server_log(
    supervisor: NativeLlamaServerSupervisor,
) -> Iterator[None]:  # pragma: no cover
    """Stream newly written native-server output while the child is active."""

    stop = threading.Event()

    def stream() -> None:
        offset = 0
        while not stop.wait(0.25):
            path = supervisor.log_path
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    print(f"exp7181 native-server {line.rstrip()}", flush=True)
                offset = handle.tell()

    thread = threading.Thread(target=stream, name="exp7181-native-log", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _write_running_shell(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Keep fallible startup state below checkpoints, never at the result path."""

    atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def _load_generation_rows(path: Path) -> list[JsonDict]:  # pragma: no cover
    """Read exactly one public JSONL input for the generation worker."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _content_addressed_hash(path: Path) -> str | None:  # pragma: no cover
    """Use the immutable cache object's digest without scanning model tensors."""

    try:
        target_name = path.resolve(strict=True).name.lower()
    except OSError:
        return None
    return "sha256:" + target_name if re.fullmatch(r"[0-9a-f]{64}", target_name) else None


def _collect_preflight(
    root: Path,
    run_date: str,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Check exact public bytes, native runtime, cache, storage, and idle GPUs."""

    checks: list[JsonDict] = []
    context: JsonDict = {}
    checks.append(gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE, field="run_date"))
    required = {
        "fixture_artifact": root / FIXTURE_ARTIFACT_PATH,
        "generation_view": root / GENERATION_VIEW_PATH,
        "constraint_spec": root / SPEC_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "focused_tests": root / TEST_PATH,
        "research_program": root / "research-program.md",
        "research_references": root / "research-references.md",
        "exclusion_manifest": root / "ops/exclusion_manifest.yaml",
        "e2e_test_plan": root / "ops/e2e-test-plan.md",
    }
    source_state = {name: path.is_file() for name, path in required.items()}
    spec_has_req = (
        "REQ-VERIFY-7181" in required["constraint_spec"].read_text(encoding="utf-8")
        if required["constraint_spec"].is_file()
        else False
    )
    source_state["spec_has_req"] = spec_has_req
    checks.append(
        gate_row(
            "required_source_bytes",
            {name: True for name in source_state},
            source_state,
            all(source_state.values()),
            upstream="repository",
            field="required_paths",
        )
    )
    fixture_hash = (
        sha256_file(required["fixture_artifact"])
        if required["fixture_artifact"].is_file()
        else "missing"
    )
    generation_hash = (
        sha256_file(required["generation_view"])
        if required["generation_view"].is_file()
        else "missing"
    )
    observed_hashes = {
        "fixture_artifact": fixture_hash,
        "generation_view": generation_hash,
    }
    expected_hashes = {
        "fixture_artifact": PINNED_FIXTURE_SHA256,
        "generation_view": PINNED_GENERATION_SHA256,
    }
    checks.append(
        gate_row(
            "same_milestone_input_hashes",
            expected_hashes,
            observed_hashes,
            observed_hashes == expected_hashes,
            upstream="experiment_7180",
            field="source_bytes",
        )
    )
    try:
        fixture = json.loads(required["fixture_artifact"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        fixture = {}
    observed_fields = {field: fixture.get(field) for field in EXP7180_EXPECTED_FIELDS}
    checks.append(
        gate_row(
            "same_milestone_gate_fields",
            EXP7180_EXPECTED_FIELDS,
            observed_fields,
            observed_fields == EXP7180_EXPECTED_FIELDS,
            upstream="experiment_7180",
            field="terminal_gate_fields",
        )
    )
    sidecar_receipt = dict(fixture.get("sidecar_hashes", {}))
    observed_public_receipt = {
        "generation_view_row_count": sidecar_receipt.get("generation_view_row_count"),
        "generation_view_sha256": sidecar_receipt.get("generation_view_sha256"),
        "label_mutation_preserves_generation_view_bytes": sidecar_receipt.get(
            "label_mutation_preserves_generation_view_bytes"
        ),
    }
    expected_public_receipt = {
        "generation_view_row_count": 192,
        "generation_view_sha256": PINNED_GENERATION_SHA256,
        "label_mutation_preserves_generation_view_bytes": True,
    }
    checks.append(
        gate_row(
            "public_generation_receipt",
            expected_public_receipt,
            observed_public_receipt,
            observed_public_receipt == expected_public_receipt,
            upstream="experiment_7180",
            field="sidecar_hashes",
        )
    )
    schedule: list[JsonDict] = []
    try:
        schedule = build_schedule(_load_generation_rows(required["generation_view"]))
        schedule_problem = schedule_errors(schedule)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        schedule_problem = [f"{type(exc).__name__}:{exc}"]
    checks.append(
        gate_row(
            "blind_192_row_schedule",
            {"row_count": 192, "errors": []},
            {"row_count": len(schedule), "errors": schedule_problem},
            not schedule_problem,
            upstream="experiment_7180_generation_view",
            field="rows",
        )
    )
    storage = {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_writable": checkpoint_dir.is_dir() and os.access(checkpoint_dir, os.W_OK),
        "raw_dir": str(raw_dir),
        "raw_writable": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
        "result_parent_writable": (root / RESULT_PATH).parent.is_dir()
        and os.access((root / RESULT_PATH).parent, os.W_OK),
    }
    checks.append(
        gate_row(
            "writable_output_storage",
            {
                **storage,
                "checkpoint_writable": True,
                "raw_writable": True,
                "result_parent_writable": True,
            },
            storage,
            storage["checkpoint_writable"]
            and storage["raw_writable"]
            and storage["result_parent_writable"],
            upstream="host_filesystem",
            field="output_paths",
        )
    )

    resolved = cached_current_model(gpu_index=0, preferred_quant=QUANTIZATION)
    model_path = Path(str(resolved.get("model_path"))) if resolved else None
    model_exists = bool(model_path and model_path.is_file())
    model_hash = _content_addressed_hash(model_path) if model_path and model_exists else None
    metadata = read_gguf_metadata(model_path) if model_path and model_exists else {}
    model_spec = {
        "hf_id": resolved.get("hf_id") if resolved else None,
        "quantization": QUANTIZATION,
        "path": str(model_path.resolve()) if model_path and model_exists else None,
        "revision": snapshot_revision(model_path) if model_path and model_exists else None,
        "bytes": model_path.stat().st_size if model_path and model_exists else None,
        "sha256": model_hash,
        "hash_source": "content_addressed_cache_target" if model_hash else None,
        "embedded_chat_template": metadata.get("chat_template_present") is True,
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "auto_tokenizer_used": False,
    }
    model_ok = bool(
        model_exists
        and resolved
        and resolved.get("hf_id") == QWEN_MODEL_ID
        and QUANTIZATION.lower() in model_path.name.lower()
        and model_hash
        and metadata.get("chat_template_present") is True
    )
    checks.append(
        gate_row(
            "mandated_gguf_cache_and_chat_template",
            {
                "hf_id": QWEN_MODEL_ID,
                "quantization": QUANTIZATION,
                "exists": True,
                "content_addressed": True,
                "embedded_chat_template": True,
                "auto_tokenizer_used": False,
            },
            {
                "hf_id": model_spec["hf_id"],
                "quantization": model_spec["quantization"],
                "exists": model_exists,
                "content_addressed": bool(model_hash),
                "embedded_chat_template": model_spec["embedded_chat_template"],
                "auto_tokenizer_used": False,
            },
            model_ok,
            upstream="cached_current_model",
            field="model_path",
        )
    )
    server = resolve_native_llama_server()
    _progress(3, "phase_start", name="native_cuda_runner")
    _progress(3, "subprocess_start", operation="native_runner_capabilities", path=str(server))
    runner_rows = lease_preflight.collect_runner_capabilities(server)
    _progress(3, "subprocess_end", operation="native_runner_capabilities")
    runner_errors = lease_preflight.runner_capability_errors(runner_rows)
    runner = deepcopy(runner_rows[0]) if runner_rows else {}
    checks.append(
        gate_row(
            "native_cuda_runtime",
            [],
            runner_errors,
            not runner_errors,
            upstream="native_llama_server",
            field="cuda_linkage_and_capabilities",
        )
    )
    _progress(3, "phase_end", name="native_cuda_runner", passed=not runner_errors)
    _progress(4, "phase_start", name="live_gpu_and_lease_inventory")
    _progress(4, "subprocess_start", operation="gpu_and_lease_inventory")
    process_rows, query_receipts = lease_preflight.collect_gpu_process_rows()
    lease_rows = lease_preflight.scan_lease_rows(lease_preflight.LEASE_RUNTIME_DIR, process_rows)
    classified = lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = lease_preflight.readiness_decision(
        classified,
        lease_rows,
        [
            {
                "repository": QWEN_MODEL_ID,
                "filename": model_path.name if model_path else None,
                "path": str(model_path) if model_path else None,
                "real_path": str(model_path.resolve()) if model_exists and model_path else None,
                "revision": model_spec["revision"],
                "bytes": model_spec["bytes"],
                "sha256": model_hash,
                "hash_source": model_spec["hash_source"],
                "weights_opened": False,
                "valid": model_ok,
            }
        ],
        runner_rows,
    )
    _progress(4, "subprocess_end", operation="gpu_and_lease_inventory")
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    checks.append(
        gate_row(
            "gpu_inventory_queries",
            True,
            query_ok,
            query_ok,
            upstream="nvidia-smi",
            field="returncodes",
        )
    )
    available = list(decision.get("available_gpu_uuids", []))
    idle_observed = {
        "available_gpu_uuids": available,
        "conflicting_processes": decision.get("conflicting_processes", []),
        "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
    }
    checks.append(
        gate_row(
            "idle_task_ownable_rtx_3090",
            {"minimum_count": 1, "no_compute_process": True, "no_conflicting_lease": True},
            idle_observed,
            bool(available),
            upstream="live_gpu_and_lease_inventory",
            field="available_gpu_uuids",
        )
    )
    context.update(
        {
            "schedule": schedule,
            "generation_view_sha256": generation_hash,
            "model_spec": model_spec,
            "model_path": model_path,
            "server_path": server,
            "runner": runner,
            "process_rows": classified,
            "lease_rows": lease_rows,
            "query_receipts": query_receipts,
            "available_gpu_uuids": available,
        }
    )
    _progress(4, "phase_end", name="live_gpu_and_lease_inventory", passed=bool(available))
    return checks, schedule, context


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(server: Path, model: Path, port: int) -> list[str]:  # pragma: no cover
    """Use one leased GPU and the chat template embedded in the GGUF."""

    return [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_TOKEN_BUDGET),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
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


def _wait_for_health(
    supervisor: NativeLlamaServerSupervisor, port: int, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Poll bounded health while the external heartbeat remains active."""

    started = time.monotonic()
    deadline = started + timeout_s
    last_error = "not_started"
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        if supervisor.proc and supervisor.proc.poll() is not None:
            return {
                "ok": False,
                "classification": "early_exit",
                "attempts": attempts,
                "duration_s": time.monotonic() - started,
            }
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as response:
                return {
                    "ok": response.status == 200,
                    "classification": "healthy",
                    "attempts": attempts,
                    "duration_s": time.monotonic() - started,
                }
        except (OSError, error.URLError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
        time.sleep(1.0)
    return {
        "ok": False,
        "classification": "deadline_expired",
        "last_error": last_error,
        "attempts": attempts,
        "duration_s": time.monotonic() - started,
    }


def _chat_request(
    port: int,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    schema: JsonDict | None,
) -> JsonDict:  # pragma: no cover
    """Make one bounded native request and retain exact request and response data."""

    payload: JsonDict = {
        "messages": [
            {
                "role": "system",
                "content": "Return only one JSON object. Use only the supplied evidence and claim.",
            },
            {"role": "user", "content": prompt},
        ],
        **DECODING_PARAMETERS,
        "max_tokens": int(max_tokens),
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "symbolic_trace", "schema": schema},
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with request.urlopen(http_request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    choice = dict(choices[0])
    message = dict(choice.get("message") or {})
    usage = dict(body.get("usage") or {})
    raw_output = str(message.get("content") or message.get("reasoning_content") or "")
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
        "raw_output": raw_output,
        "raw_response": body,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.monotonic() - started,
        "finish_reason": choice.get("finish_reason"),
        "error": None,
    }


def _request_or_error(
    port: int,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
    schema: JsonDict | None,
) -> JsonDict:  # pragma: no cover
    """Convert every request exception into a real terminal error receipt."""

    started = time.monotonic()
    try:
        return _chat_request(
            port,
            prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            schema=schema,
        )
    except Exception as exc:  # noqa: BLE001 - the exact runtime failure is retained.
        return {
            "raw_request": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "schema_sha256": sha256_json(schema) if schema else None,
            },
            "raw_output": "",
            "raw_response": {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": time.monotonic() - started,
            "finish_reason": None,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _selected_device(context: Mapping[str, Any], gpu_uuid: str) -> JsonDict:  # pragma: no cover
    """Return the idle inventory row for the selected leased GPU."""

    return next(
        deepcopy(dict(row)) for row in context["process_rows"] if row.get("gpu_uuid") == gpu_uuid
    )


def _gpu_provenance(
    snapshot: Mapping[str, Any], identity: Mapping[str, Any], gpu_uuid: str, server_log: str
) -> JsonDict:  # pragma: no cover
    """Bind CUDA log markers to VRAM held by the task-owned server PID."""

    pid = int(identity.get("pid", -1))
    apps = [
        deepcopy(dict(app))
        for app in snapshot.get("compute_apps", [])
        if app.get("gpu_uuid") == gpu_uuid
        and int(app.get("pid", -2)) == pid
        and app.get("owned_by_task") is True
        and int(app.get("used_memory_mb", 0) or 0) > 0
    ]
    layer_match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", server_log)
    cuda_log = bool(layer_match) or any(
        marker in server_log.lower()
        for marker in ("ggml_cuda", "cuda0 buffer", "cuda found", "cuda : archs")
    )
    return {
        "server_pid": pid,
        "server_pid_start_ticks": identity.get("start_time_ticks"),
        "gpu_uuid": gpu_uuid,
        "owned_compute_apps": apps,
        "task_owned_vram_mb": sum(int(app["used_memory_mb"]) for app in apps),
        "cuda_log_evidence": cuda_log,
        "logged_layer_receipt": layer_match.group(0) if layer_match else None,
        "provenance_ok": bool(apps and cuda_log and identity.get("owned_by_task") is True),
    }


def _write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    status: str,
) -> JsonDict:  # pragma: no cover
    """Publish the public schedule and exact per-request replay paths."""

    manifest = {
        "schema": "carnot.exp7181.raw_manifest.v1",
        "status": status,
        "path": str(raw_dir / RAW_MANIFEST_NAME),
        "worker_input_paths": [str(GENERATION_VIEW_PATH)],
        "forbidden_input_open_count": 0,
        "schedule": [deepcopy(dict(row)) for row in schedule],
        "schedule_sha256": sha256_json(list(schedule)),
        "checkpoint_identity": deepcopy(dict(identity)),
        "checkpoint_receipts": [deepcopy(dict(row)) for row in checkpoint_rows],
        "raw_rows": [
            {
                "row_order": row.get("row_order"),
                "unit_id": row.get("unit_id"),
                "path": str(raw_dir / f"row_{int(row.get('row_order', 0)):03d}.json"),
                "prompt_sha256": row.get("prompt_sha256"),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "terminal_state": row.get("terminal_state"),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(raw_dir / RAW_MANIFEST_NAME, manifest, allow_override=False, sort_keys=True)
    return manifest


def _release_lease(
    lease: lease_api.GpuLease,
    cleanup: Mapping[str, Any],
    after: Mapping[str, Any],
    identity: Mapping[str, Any],
    complete: bool,
) -> JsonDict:  # pragma: no cover
    """Close the lease only after owned cleanup and unload evidence are durable."""

    pid = identity.get("pid")
    vram_released = not any(
        app.get("pid") == pid and int(app.get("used_memory_mb", 0) or 0) > 0
        for app in after.get("compute_apps", [])
    )
    phase = str(lease.document.get("phase"))
    if phase in {"resident", "inferencing"}:
        lease.transition("unloading")
        phase = "unloading"
    if phase == "unloading":
        device_after = next(
            (
                device
                for device in after.get("devices", [])
                if device.get("uuid") == lease.device_uuid
            ),
            {},
        )
        lease.transition(
            "validating",
            vram_mb=int(device_after.get("memory_used_mb", 0) or 0),
            exit_code=0,
            unload_observed=cleanup.get("leak_free") is True and vram_released,
        )
        phase = "validating"
    if phase == "validating":
        lease.transition("terminal_complete" if complete else "terminal_blocked")
    elif phase in {"preflight", "admitted", "loading"}:
        lease.transition("terminal_blocked")
    return lease.release()


def _live_capture(
    context: Mapping[str, Any],
    checkpoint_dir: Path,
    raw_dir: Path,
    spans: list[JsonDict],
    total_started: float,
) -> JsonDict:  # pragma: no cover
    """Acquire one GPU, run canary and schedule, then close only owned state."""

    schedule = list(context["schedule"])
    gpu_uuid = str(context["available_gpu_uuids"][0])
    device = _selected_device(context, gpu_uuid)
    gpu_index = int(device["gpu_index"])
    model = Path(str(context["model_path"]))
    model_spec = deepcopy(dict(context["model_spec"]))
    port = _free_port()
    command = _server_command(Path(context["server_path"]), model, port)
    contract = supervisor_contract(
        outer_deadline_s=TOTAL_CAP_S,
        health_timeout_s=600.0,
        token_timeout_s=REQUEST_CAP_S,
        cleanup_grace_s=30.0,
        kill_after_cleanup_timeout_s=10.0,
        retry_budget=0,
        endurance_interval_s=0.0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    lease: lease_api.GpuLease | None = None
    identity: JsonDict = {}
    canary: JsonDict | None = None
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    snapshots: list[JsonDict] = []
    provenance: JsonDict = {"provenance_ok": False}
    cleanup: JsonDict = {"action": "not_started", "leak_free": True}
    lease_release: JsonDict = {"released": False}
    runtime_error: str | None = None
    load_duration = 0.0
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    checkpoint_id = checkpoint_identity(
        schedule, str(context["generation_view_sha256"]), str(model_spec["sha256"])
    )
    try:
        with _phase_span(spans, 5, "task_owned_lease_and_model_load"):
            snapshots.append(_gpu_snapshot("before_model_load", phase=5))
            lease = lease_api.GpuLease.acquire(
                runtime_dir=lease_preflight.LEASE_RUNTIME_DIR,
                task_id=TASK_ID,
                device_uuid=gpu_uuid,
                expected_model=str(model),
                vram_before_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
                ttl_s=900.0,
            )
            lease.transition("admitted")
            lease.transition("loading")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            _progress(5, "subprocess_start", operation="native_llama_server", command=command)
            _progress(5, "model_load_start", model=QWEN_MODEL_ID, gpu_uuid=gpu_uuid)
            load_started = time.monotonic()
            identity = supervisor.launch()
            with _stream_server_log(supervisor), _heartbeat(5, "native_model_load", lambda: 0, 1):
                health = _wait_for_health(supervisor, port, 600.0)
            load_duration = time.monotonic() - load_started
            _progress(
                5,
                "model_load_end",
                model=QWEN_MODEL_ID,
                gpu_uuid=gpu_uuid,
                health=health.get("ok"),
                duration_s=round(load_duration, 6),
            )
            if health.get("ok") is not True:
                raise RuntimeError(f"model_health_failed:{health.get('classification')}")
            resident = _gpu_snapshot("model_resident", phase=5)
            snapshots.append(resident)
            provenance = _gpu_provenance(resident, identity, gpu_uuid, supervisor.stderr_tail())
            if provenance["provenance_ok"] is not True:
                raise RuntimeError("cuda_placement_or_owned_vram_unconfirmed")
            lease.transition("resident", vram_mb=int(provenance["task_owned_vram_mb"]))
            lease.transition("inferencing")

        resource = {
            "server_pid": identity.get("pid"),
            "server_pid_start_ticks": identity.get("start_time_ticks"),
            "gpu_uuid": gpu_uuid,
            "lease_id": lease.lease_id,
        }
        with _phase_span(spans, 6, "bounded_canary_generation"):
            _progress(6, "generation_start", operation="canary", max_tokens=CANARY_TOKEN_BUDGET)
            with _heartbeat(6, "canary_generation", lambda: 0, 1):
                canary_response = _request_or_error(
                    port,
                    "Return one JSON object with the single field canary_ok set to true.",
                    max_tokens=CANARY_TOKEN_BUDGET,
                    timeout_s=REQUEST_CAP_S,
                    schema={
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["canary_ok"],
                        "properties": {"canary_ok": {"type": "boolean"}},
                    },
                )
            canary = {
                **canary_response,
                "terminal_state": "request_error" if canary_response.get("error") else "response",
                "max_tokens": CANARY_TOKEN_BUDGET,
                "model_bounded_generation": True,
            }
            _progress(
                6,
                "generation_end",
                operation="canary",
                terminal_state=canary["terminal_state"],
                duration_s=round(float(canary["latency_s"]), 6),
            )
            if canary["terminal_state"] != "response":
                raise RuntimeError(f"canary_failed:{canary.get('error')}")

        latest = checkpoint_dir / "checkpoint_latest.json"
        if latest.is_file():
            rows = resume_checkpoint(latest, checkpoint_id)
            _progress(7, "checkpoint_resume", completed_units=len(rows), total_units=192)
        measurement_started = time.monotonic()
        with _phase_span(spans, 7, "frozen_192_row_generation"):
            while len(rows) < len(schedule):
                elapsed_measurement = time.monotonic() - measurement_started
                elapsed_total = time.monotonic() - total_started
                remaining = min(
                    MEASUREMENT_CAP_S - elapsed_measurement,
                    TOTAL_CAP_S - elapsed_total,
                )
                if remaining <= 0.25:
                    runtime_error = "measurement_or_total_execution_cap_reached"
                    break
                index = len(rows)
                sealed = schedule[index]
                _progress(
                    7,
                    "generation_start",
                    operation="scheduled_row",
                    row_order=index,
                    unit_id=sealed["unit_id"],
                    completed_units=len(rows),
                    total_units=len(schedule),
                )
                with _heartbeat(7, "scheduled_generation", lambda: len(rows), len(schedule)):
                    response = _request_or_error(
                        port,
                        str(sealed["prompt"]),
                        max_tokens=OUTPUT_TOKEN_BUDGET,
                        timeout_s=min(REQUEST_CAP_S, max(0.25, remaining)),
                        schema=JSON_RESPONSE_SCHEMA,
                    )
                row = build_completion_row(sealed, response, resource)
                rows.append(row)
                atomic_write_json(
                    raw_dir / f"row_{index:03d}.json",
                    {"schedule": deepcopy(dict(sealed)), "completion": row},
                    allow_override=False,
                    sort_keys=True,
                )
                _progress(
                    7,
                    "generation_end",
                    operation="scheduled_row",
                    row_order=index,
                    unit_id=sealed["unit_id"],
                    terminal_state=row["terminal_state"],
                    parse_status=row["parse_status"],
                    completion_tokens=row["completion_tokens"],
                    duration_s=round(float(row["latency_s"]), 6),
                )
                if len(rows) % CHECKPOINT_CADENCE == 0:
                    path = checkpoint_dir / f"checkpoint_{len(rows):03d}.json"
                    _progress(7, "checkpoint_write_start", path=str(path), rows=len(rows))
                    payload = write_checkpoint(path, checkpoint_id, rows)
                    write_checkpoint(latest, checkpoint_id, rows)
                    checkpoints.append(
                        {
                            "path": str(path),
                            "row_count": len(rows),
                            "checkpoint_sha256": sha256_json(payload),
                            "row_hashes": deepcopy(payload["row_hashes"]),
                        }
                    )
                    lease.heartbeat()
                    _progress(7, "checkpoint_write_end", path=str(path), rows=len(rows))
    except Exception as exc:  # noqa: BLE001 - retain real progress and classify it below.
        runtime_error = f"{type(exc).__name__}:{exc}"
        _progress(7, "runtime_error", error=runtime_error, completed_units=len(rows))
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        with _phase_span(spans, 8, "owned_cleanup"):
            _progress(
                8,
                "cleanup_start",
                pid=identity.get("pid"),
                lease_id=getattr(lease, "lease_id", None),
            )
            cleanup = supervisor.cleanup()
            if supervisor.proc is not None:
                try:
                    supervisor.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            after = _gpu_snapshot("after_cleanup", phase=8)
            snapshots.append(after)
            if lease is not None:
                try:
                    lease_release = _release_lease(
                        lease,
                        cleanup,
                        after,
                        identity,
                        len(rows) == 192 and provenance.get("provenance_ok") is True,
                    )
                except lease_api.LeaseError as exc:
                    lease_release = {
                        "released": False,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                    lease.close()
            _progress(
                8,
                "cleanup_end",
                pid=identity.get("pid"),
                process_released=cleanup.get("leak_free"),
                lease_released=lease_release.get("released"),
            )
            _progress(
                5,
                "subprocess_end",
                operation="native_llama_server",
                returncode=supervisor.proc.poll() if supervisor.proc else None,
            )
    provenance_ok = bool(
        provenance.get("provenance_ok")
        and cleanup.get("leak_free") is True
        and lease_release.get("released") is True
    )
    return {
        "canary": canary,
        "rows": rows,
        "checkpoints": checkpoints,
        "checkpoint_identity": checkpoint_id,
        "gpu_receipts": {
            "preflight_process_rows": deepcopy(context["process_rows"]),
            "preflight_lease_rows": deepcopy(context["lease_rows"]),
            "inventory_query_receipts": deepcopy(context["query_receipts"]),
            "lease_owner": lease.owner_receipt() if lease is not None else None,
            "server_identity": deepcopy(identity),
            "runtime_provenance": provenance,
            "snapshots": snapshots,
            "cleanup": cleanup,
            "lease_release": lease_release,
            "provenance_ok": provenance_ok,
        },
        "runner_receipt": {
            "model_count": 1,
            "runner": "native_llama.cpp_server",
            "selected_runner": deepcopy(context["runner"]),
            "command": command,
            "load_duration_s": load_duration,
            "request_cap_s": REQUEST_CAP_S,
            "measurement_cap_s": MEASUREMENT_CAP_S,
            "total_execution_cap_s": TOTAL_CAP_S,
            "repair_call_count": 0,
        },
        "runtime_error": runtime_error,
    }


def _finish(
    artifact: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    capture: Mapping[str, Any] | None,
    raw_dir: Path,
    duration_s: float,
) -> JsonDict:  # pragma: no cover
    """Build one honest terminal artifact from observed work and cleanup."""

    result = deepcopy(dict(artifact))
    rows = [deepcopy(dict(row)) for row in (capture or {}).get("rows", [])]
    canary = deepcopy((capture or {}).get("canary"))
    gpu_receipts = deepcopy((capture or {}).get("gpu_receipts", {}))
    provenance_ok = bool(gpu_receipts.get("provenance_ok"))
    terminal = classify_terminal(checks, canary, rows, provenance_ok=provenance_ok)
    failed = next((deepcopy(dict(row)) for row in checks if row.get("passed") is not True), None)
    if capture and capture.get("runtime_error") and canary is None and not rows:
        failed = gate_row(
            "live_runtime_before_generation",
            "qualifying_generation_receipt",
            capture["runtime_error"],
            False,
            upstream="native_llama_server",
            field="runtime_error",
        )
        checks = [*checks, failed]
        terminal = classify_terminal(checks, canary, rows, provenance_ok=provenance_ok)
    if terminal["status"] == "blocked":
        verdict_class = "blocked"
        honest = f"blocked_{failed['check'] if failed else 'external_precondition'}"
        substrate = "preflight_only_no_model_invocation"
        gate = failed or gate_row(
            "external_precondition", True, False, False, upstream="experiment_7181"
        )
        scope_answer = "No model trace measurement ran because an external prerequisite failed."
    elif terminal["score"] == 1:
        verdict_class = "positive"
        honest = "complete_positive_transport_capture_no_correctness_claim"
        substrate = "native_llama_cpp_qwen38_symbolic_trace_generation"
        gate = gate_row(
            "trace_capture_complete",
            True,
            True,
            True,
            upstream="experiment_7181",
            field="trace_capture_complete_score",
        )
        scope_answer = (
            "Qwen3.8 returned a terminal completion or request-error receipt for all 192 blind rows. "
            "This establishes trace transport only; no correctness score was computed."
        )
    else:
        verdict_class = "partial"
        honest = (
            "partial_canary_only_bounded_generation"
            if not rows
            else "partial_incomplete_resumable_symbolic_trace_capture"
        )
        substrate = (
            "native_llama_cpp_qwen38_bounded_canary"
            if not rows
            else "native_llama_cpp_qwen38_symbolic_trace_generation"
        )
        gate = gate_row(
            "trace_capture_complete",
            192,
            len(rows),
            False,
            upstream="experiment_7181",
            field="completion_rows",
        )
        scope_answer = (
            f"Qwen3.8 produced {len(rows)} of 192 scheduled terminal receipts. "
            "The incomplete owned work remains resumable and supports no complete-capture claim."
        )
    model_spec = deepcopy(context.get("model_spec", {}))
    manifest = _write_raw_manifest(
        raw_dir,
        context.get("schedule", []),
        rows,
        (capture or {}).get("checkpoints", []),
        (capture or {}).get(
            "checkpoint_identity",
            checkpoint_identity(
                context.get("schedule", []),
                str(context.get("generation_view_sha256", "missing")),
                str(model_spec.get("sha256", "missing")),
            ),
        ),
        status=terminal["status"],
    )
    result.update(
        {
            "status": terminal["status"],
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "inference_substrate": substrate,
            "inference_substrate_class": terminal["inference_substrate_class"],
            "duration_s": float(duration_s),
            "rows": rows,
            "completion_rows": rows,
            "trace_capture_complete_score": terminal["score"],
            "model_specs": [model_spec] if model_spec else [],
            "raw_manifest": manifest,
            "gpu_receipts": gpu_receipts,
            "runner_receipt": deepcopy(
                (capture or {}).get(
                    "runner_receipt",
                    {
                        "model_count": 1,
                        "runner": "native_llama.cpp_server_preflight",
                        "selected_runner": deepcopy(context.get("runner", {})),
                    },
                )
            ),
            "gate_check_summary": gate,
            "verdict_class": verdict_class,
            "honest_verdict": honest,
            "canary_receipt": canary,
            "scope_answer": scope_answer,
        }
    )
    result["source_artifact_hashes"] = _source_artifact_hashes(
        find_repo_root(),
        raw_manifest=raw_dir / RAW_MANIFEST_NAME,
        model_sha256=str(model_spec.get("sha256")) if model_spec.get("sha256") else None,
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def run_experiment(
    *, root: Path, run_date: str, result_path: Path, checkpoint_dir: Path, raw_dir: Path
) -> JsonDict:  # pragma: no cover
    """Run one finite attempt and preserve a blocked or measured terminal result."""

    total_started = time.monotonic()
    spans: list[JsonDict] = []
    artifact = base_artifact(run_date, root=root)
    artifact["phase_spans"] = spans
    with _phase_span(spans, 0, "checkpoint_shell_before_checks"):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shell_path = checkpoint_dir / "running_shell.json"
        _progress(0, "checkpoint_write_start", path=str(shell_path))
        _write_running_shell(shell_path, artifact)
        _progress(0, "checkpoint_write_end", path=str(shell_path))
        _write_raw_manifest(raw_dir, [], [], [], {}, status="running_unsealed")

    with _phase_span(spans, 1, "source_and_same_milestone_contracts"):
        _progress(1, "benchmark_start", operation="preflight_contracts")
        checks, schedule, context = _collect_preflight(root, run_date, checkpoint_dir, raw_dir)
        _progress(
            1,
            "benchmark_end",
            operation="preflight_contracts",
            passed=all(row.get("passed") is True for row in checks),
        )
    context["schedule"] = schedule
    artifact["model_specs"] = [deepcopy(context.get("model_spec", {}))]
    artifact["runner_receipt"] = {
        "model_count": 1,
        "runner": "native_llama.cpp_server_preflight",
        "selected_runner": deepcopy(context.get("runner", {})),
    }
    with _phase_span(spans, 2, "blind_schedule_sealed"):
        _progress(
            2,
            "benchmark_start",
            operation="schedule_validation",
            completed_units=0,
            total_units=192,
        )
        schedule_problem = schedule_errors(schedule) if schedule else ["schedule_missing"]
        _progress(
            2,
            "benchmark_end",
            operation="schedule_validation",
            completed_units=len(schedule),
            total_units=192,
            errors=schedule_problem,
        )
    all_ready = bool(checks) and all(row.get("passed") is True for row in checks)
    capture: JsonDict | None = None
    if all_ready:
        capture = _live_capture(context, checkpoint_dir, raw_dir, spans, total_started)
    result = _finish(
        artifact,
        checks,
        context,
        capture,
        raw_dir,
        time.monotonic() - total_started,
    )
    result["phase_spans"] = deepcopy(spans)
    result["reproducibility_checksum"] = artifact_checksum(result)
    phase_nine_started = time.monotonic()
    _progress(9, "phase_start", name="terminal_validation_and_atomic_write")
    _progress(9, "validation_start", operation="cold_artifact_validation")
    result["phase_spans"] = deepcopy(spans)
    result["reproducibility_checksum"] = artifact_checksum(result)
    errors = validate_artifact(result, check_source_hashes=True)
    _progress(9, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    phase_nine_ended = time.monotonic()
    result["phase_spans"] = [
        *deepcopy(spans),
        {
            "phase": 9,
            "name": "terminal_validation_and_write_preparation",
            "start_monotonic_s": phase_nine_started,
            "end_monotonic_s": phase_nine_ended,
            "duration_s": phase_nine_ended - phase_nine_started,
        },
    ]
    result["reproducibility_checksum"] = artifact_checksum(result)
    _progress(9, "artifact_write_start", path=str(result_path), status=result["status"])
    atomic_write_json(result_path, result, allow_override=False, sort_keys=True)
    _progress(9, "artifact_write_end", path=str(result_path), status=result["status"])
    _progress(
        9,
        "phase_end",
        name="terminal_validation_and_atomic_write",
        duration_s=round(time.monotonic() - phase_nine_started, 6),
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    _progress(0, "phase_start", name="entrypoint_before_checks")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    root = find_repo_root()
    if args.validate is not None:
        _progress(9, "validation_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(9, "validation_end", path=str(args.validate), errors=errors)
        return int(bool(errors))
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint_dir = (
        args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_experiment(
        root=root,
        run_date=args.date,
        result_path=result_path,
        checkpoint_dir=checkpoint_dir,
        raw_dir=raw_dir,
    )
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "trace_capture_complete_score": result["trace_capture_complete_score"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
