"""Capture separated Qwen source, claim, and direct typed outputs.

The model reads only Exp7195's public rows. This task records transport,
parsing, and cost evidence. A later independent task decides verifier value.

Spec refs: REQ-VERIFY-7196 and SCENARIO-VERIFY-7196-*.
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

import yaml

from carnot import experiment_7181_v633_qwen38_symbolic_traces as shipped_runtime
from carnot import experiment_7195_v634_typed_grounding as typed_fixture
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    canonical_json,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_current_model
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260910"
TASK_ID = "experiment_7196_v634_qwen_atomic_capture"
RANDOM_SEED = 7_196_202_609_10
RESULT_PATH = Path("results/experiment_7196_v634_qwen_atomic_capture.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7196_v634_qwen_atomic_capture")
RAW_DIR = Path("results/raw/experiment_7196_v634_qwen_atomic_capture")
RAW_MANIFEST_NAME = "raw_manifest.json"
UPSTREAM_PATH = Path("results/experiment_7195_v634_typed_grounding.json")
PUBLIC_VIEW_PATH = Path("results/experiment_7195_v634_typed_grounding_public.jsonl")
AUTHORITY_PATH = Path("results/experiment_7195_v634_typed_grounding_authority.jsonl")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7196_v634_qwen_atomic_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7196_v634_qwen_atomic_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7196_v634_qwen_atomic_capture.py")

QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]
PINNED_UPSTREAM_SHA256 = "sha256:be16bf10a37010e12a3c4851c313c0d834a391ab2040be9201efdce9221054da"
PINNED_PUBLIC_SHA256 = "sha256:ecf5decc1ab53233c3ee011b38508d5e5e10bb3b30ebf6e7c0265397d939150b"
PINNED_CONTRACT_SHA256 = "sha256:26a1c7678a9fc9f3f89e79c1d8f5cefc9a9cb15c046cd3c9b878de9b18c67e08"
UPSTREAM_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "typed_executor_ready_score": 1,
    "verdict_class": "circular_positive",
    "honest_verdict": "complete_circular_positive_typed_executor_ready_no_independent_value_claim",
    "public_view_path": PUBLIC_VIEW_PATH.as_posix(),
    "authority_sidecar_path": AUTHORITY_PATH.as_posix(),
    "reproducibility_checksum": (
        "sha256:653251a6e721daf041e0f22fbf2edefed819ff61fe9787dfd75b724cbe5b93cc"
    ),
}

SOURCE_TOKEN_BUDGET = 128
CLAIM_TOKEN_BUDGET = 64
DIRECT_TOKEN_BUDGET = 16
CANARY_TOKEN_BUDGET = 8
REQUEST_CAP_S = 60.0
CAPTURE_CAP_S = 2_400.0
TOTAL_CAP_S = 3_300.0
CONTEXT_TOKEN_BUDGET = 4_096
CHECKPOINT_CADENCE = 24
LOGICAL_RECEIPT_COUNT = 576
EXPECTED_COLD_REQUEST_COUNT = 481
EXPECTED_SOURCE_CACHE_HITS = 95
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": RANDOM_SEED & 0x7FFFFFFF,
    "cache_prompt": False,
}

TYPED_JSON_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entity_bindings", "relations", "missing_fields"],
    "properties": {
        "entity_bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity_id", "surface", "source_start", "source_end"],
                "properties": {
                    "entity_id": {"type": "string"},
                    "surface": {"type": "string"},
                    "source_start": {"type": "integer", "minimum": 0},
                    "source_end": {"type": "integer", "minimum": 0},
                },
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "subject_id",
                    "operator",
                    "object_id",
                    "polarity",
                    "source_start",
                    "source_end",
                ],
                "properties": {
                    "subject_id": {"type": "string"},
                    "operator": {"type": "string"},
                    "object_id": {"type": "string"},
                    "polarity": {"type": "string", "enum": ["positive", "negative"]},
                    "source_start": {"type": "integer", "minimum": 0},
                    "source_end": {"type": "integer", "minimum": 0},
                },
            },
        },
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
}
DIRECT_JSON_SCHEMA: JsonDict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {"decision": {"type": "string", "enum": ["supported", "unsupported", "abstain"]}},
}

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260910, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when verification uses the same correctness authority; separate implementations alone do not remove circularity.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings, including nulls; blocked_* for external blocks. Never promote infrastructure readiness as scientific benefit.",
    "atomic_capture_complete_score": "Complete capture retains unsuccessful calls for downstream analysis.",
    "MODEL_SPECS": "Include unsloth/Qwen3.8-27B-GGUF through the resolved GGUF path.",
    "model_specs": "Record the loaded model and embedded tokenizer/template hashes.",
    "completion_rows": "Keep every source, claim and direct call with raw bytes and errors.",
    "phase_spans": "Separate setup, load, generation, parsing and teardown costs.",
    "gpu_receipts": "Samples must overlap actual generation and identify owned processes.",
    "runner_receipt": "Record one model, actual replicas and lease lifecycle.",
    "raw_manifest": "Hashes bind each response to its frozen input and prompt.",
    "inference_mode": "live_gpu requires measured CUDA execution in the task window.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def sha256_text(value: str) -> str:
    """Hash exact UTF-8 text so prompt whitespace remains evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes without newline or JSON normalization."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one stable JSON spelling for cross-process receipts."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a normal source file without normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all result fields except the checksum that contains the hash."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None = None,
    field: str | None = None,
) -> JsonDict:
    """Keep both sides of one prerequisite instead of hiding a failure."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def first_failed_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Return the first real gate failure so terminal reasons stay stable."""

    return next((deepcopy(dict(row)) for row in rows if row.get("passed") is not True), None)


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) >= {"principle", "value"}:
        return value["value"]
    return value


def _manifest_hits(value: Any, wanted: frozenset[str]) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"experiment_id", "experiment_ids"}:
                candidates = child if isinstance(child, list) else [child]
                hits.update(str(item) for item in candidates if str(item) in wanted)
            hits.update(_manifest_hits(child, wanted))
    elif isinstance(value, list):
        for child in value:
            hits.update(_manifest_hits(child, wanted))
    return hits


def upstream_gate_rows(
    upstream: Mapping[str, Any],
    public_bytes: bytes,
    *,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Check quarantine before terminal fields can authorize consumption."""

    flagged = bool(_unwrap(upstream.get("flagged_adversarial", False)))
    hits = sorted(_manifest_hits(exclusion_manifest, frozenset({"7195"})))
    observed_fields = {field: upstream.get(field) for field in UPSTREAM_EXPECTED_FIELDS}
    manifest = dict(upstream.get("fixture_manifest") or {})
    sidecars = dict(manifest.get("sidecar_hashes") or {})
    contract = dict(manifest.get("frozen_generation_contract") or {})
    old_values = sorted(
        {
            row.get("old_grounding_value_score")
            for row in upstream.get("error_decomposition_rows", [])
            if isinstance(row, Mapping)
        }
    )
    return [
        gate_row(
            "upstream_structured_quarantine",
            False,
            flagged,
            not flagged,
            upstream=UPSTREAM_PATH.as_posix(),
            field="flagged_adversarial",
        ),
        gate_row(
            "upstream_manifest_quarantine",
            [],
            hits,
            not hits,
            upstream="ops/exclusion_manifest.yaml",
            field="excluded_upstream_ids",
        ),
        gate_row(
            "upstream_terminal_gate_fields",
            UPSTREAM_EXPECTED_FIELDS,
            observed_fields,
            observed_fields == UPSTREAM_EXPECTED_FIELDS,
            upstream=UPSTREAM_PATH.as_posix(),
            field="terminal_gate_fields",
        ),
        gate_row(
            "known_failed_value_preserved",
            [0],
            old_values,
            old_values == [0],
            upstream=UPSTREAM_PATH.as_posix(),
            field="error_decomposition_rows.old_grounding_value_score",
        ),
        gate_row(
            "frozen_generation_contract",
            PINNED_CONTRACT_SHA256,
            contract.get("contract_sha256"),
            contract.get("contract_sha256") == PINNED_CONTRACT_SHA256,
            upstream=UPSTREAM_PATH.as_posix(),
            field="fixture_manifest.frozen_generation_contract.contract_sha256",
        ),
        gate_row(
            "public_view_receipt",
            {"row_count": 192, "sha256": PINNED_PUBLIC_SHA256},
            {
                "row_count": sidecars.get("public_view_row_count"),
                "sha256": sha256_bytes(public_bytes),
            },
            sidecars.get("public_view_row_count") == 192
            and sidecars.get("public_view_sha256") == PINNED_PUBLIC_SHA256
            and sha256_bytes(public_bytes) == PINNED_PUBLIC_SHA256,
            upstream=PUBLIC_VIEW_PATH.as_posix(),
            field="fixture_manifest.sidecar_hashes.public_view_sha256",
        ),
    ]


def _schema_for(call_type: str) -> JsonDict:
    return deepcopy(DIRECT_JSON_SCHEMA if call_type == "direct" else TYPED_JSON_SCHEMA)


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[JsonDict]:
    """Freeze three isolated logical calls for each public comparison row."""

    atomic = dict(contract.get("atomic_prompts") or {})
    schedule: list[JsonDict] = []
    first_source_call: dict[str, str] = {}
    for row_order, public in enumerate(public_rows):
        if set(public) != {"unit_id", "source_text", "claim_text"}:
            raise ValueError(f"public_view_shape:{row_order}")
        for call_type in ("source", "claim", "direct"):
            prompt_contract = dict(atomic.get(call_type) or {})
            visible = list(prompt_contract.get("visible_fields") or [])
            model_input = {field: public[field] for field in visible}
            template = str(prompt_contract.get("template", ""))
            prompt = template.format(**model_input)
            call_id = f"{row_order:03d}:{call_type}"
            input_hash = sha256_json(model_input)
            cache_key = sha256_text(str(public["source_text"])) if call_type == "source" else None
            reused_from = first_source_call.get(str(cache_key)) if cache_key else None
            if cache_key and reused_from is None:
                first_source_call[cache_key] = call_id
            schema = _schema_for(call_type)
            schedule.append(
                {
                    "logical_order": len(schedule),
                    "row_order": row_order,
                    "unit_id": str(public["unit_id"]),
                    "call_id": call_id,
                    "call_type": call_type,
                    "model_input": deepcopy(model_input),
                    "input_sha256": input_hash,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_template_sha256": sha256_text(template),
                    "response_schema": schema,
                    "response_schema_sha256": sha256_json(schema),
                    "output_token_budget": int(prompt_contract.get("max_tokens", 0)),
                    "decoding_parameters": deepcopy(DECODING_PARAMETERS),
                    "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
                    "cache_key": cache_key,
                    "cold_request_expected": reused_from is None,
                    "cache_hit_expected": reused_from is not None,
                    "reuse_from_call_id": reused_from,
                }
            )
    errors = schedule_errors(schedule, public_rows, contract)
    if errors:
        raise ValueError("schedule_invalid:" + ",".join(errors))
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> list[str]:
    """Reject count, order, isolation, budget, hash, or cache drift."""

    errors: list[str] = []
    atomic = dict(contract.get("atomic_prompts") or {})
    expected_visible = {
        "source": ["source_text"],
        "claim": ["claim_text"],
        "direct": ["source_text", "claim_text"],
    }
    expected_budgets = {"source": 128, "claim": 64, "direct": 16}
    if len(public_rows) != 192:
        errors.append("public_row_count_mismatch")
    if len(schedule) != LOGICAL_RECEIPT_COUNT:
        errors.append("logical_call_count_mismatch")
    seen_sources: dict[str, str] = {}
    for index, row in enumerate(schedule):
        if row.get("logical_order") != index:
            errors.append(f"call_{index}:logical_order")
        call_type = str(row.get("call_type"))
        row_order = int(row.get("row_order", -1))
        public = public_rows[row_order] if 0 <= row_order < len(public_rows) else {}
        visible = expected_visible.get(call_type, [])
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping) or set(model_input) != set(visible):
            errors.append(f"call_{index}:model_input_shape")
            continue
        expected_input = {field: public.get(field) for field in visible}
        if dict(model_input) != expected_input or row.get("input_sha256") != sha256_json(
            model_input
        ):
            errors.append(f"call_{index}:input_hash")
        prompt_contract = dict(atomic.get(call_type) or {})
        template = str(prompt_contract.get("template", ""))
        prompt = template.format(**expected_input)
        if row.get("prompt") != prompt or row.get("prompt_sha256") != sha256_text(prompt):
            errors.append(f"call_{index}:prompt_hash")
        if row.get("prompt_template_sha256") != sha256_text(template):
            errors.append(f"call_{index}:template_hash")
        if row.get("output_token_budget") != expected_budgets.get(call_type):
            errors.append(f"call_{index}:token_budget")
        if row.get("response_schema") != _schema_for(call_type) or row.get(
            "response_schema_sha256"
        ) != sha256_json(_schema_for(call_type)):
            errors.append(f"call_{index}:schema")
        if row.get("decoding_parameters") != DECODING_PARAMETERS or row.get(
            "decoding_parameters_sha256"
        ) != sha256_json(DECODING_PARAMETERS):
            errors.append(f"call_{index}:decoding")
        if call_type == "source":
            cache_key = sha256_text(str(public.get("source_text", "")))
            prior = seen_sources.get(cache_key)
            if row.get("cache_key") != cache_key:
                errors.append(f"call_{index}:cache_key")
            if bool(row.get("cache_hit_expected")) != (prior is not None):
                errors.append(f"call_{index}:cache_hit")
            if row.get("reuse_from_call_id") != prior:
                errors.append(f"call_{index}:cache_origin")
            seen_sources.setdefault(cache_key, str(row.get("call_id")))
        elif row.get("cache_key") is not None or row.get("cache_hit_expected"):
            errors.append(f"call_{index}:non_source_cache")
    if (
        sum(bool(row.get("cold_request_expected")) for row in schedule)
        != EXPECTED_COLD_REQUEST_COUNT
    ):
        errors.append("cold_request_count_mismatch")
    if sum(bool(row.get("cache_hit_expected")) for row in schedule) != EXPECTED_SOURCE_CACHE_HITS:
        errors.append("source_cache_hit_count_mismatch")
    return list(dict.fromkeys(errors))


def frozen_schedule_replay_errors(
    root: Path, observed_schedule: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Rebuild the schedule from pinned public bytes and reject manifest drift."""

    try:
        upstream = json.loads((root / UPSTREAM_PATH).read_text(encoding="utf-8"))
        public_bytes = (root / PUBLIC_VIEW_PATH).read_bytes()
        public_rows = _load_public_rows(public_bytes)
        contract = dict(upstream["fixture_manifest"]["frozen_generation_contract"])
        expected_schedule = build_schedule(public_rows, contract)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        return [f"frozen_schedule_unavailable:{type(exc).__name__}"]
    if list(observed_schedule) != expected_schedule:
        return ["frozen_schedule_mismatch"]
    return []


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _typed_shape_valid(value: Mapping[str, Any]) -> bool:
    if set(value) != {"entity_bindings", "relations", "missing_fields"}:
        return False
    bindings = value["entity_bindings"]
    relations = value["relations"]
    missing = value["missing_fields"]
    if not isinstance(bindings, list) or not isinstance(relations, list):
        return False
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        return False
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "entity_id",
            "surface",
            "source_start",
            "source_end",
        }:
            return False
        if not isinstance(binding["entity_id"], str) or not isinstance(binding["surface"], str):
            return False
        if not _is_integer(binding["source_start"]) or not _is_integer(binding["source_end"]):
            return False
        if not 0 <= binding["source_start"] <= binding["source_end"]:
            return False
    for relation in relations:
        if not isinstance(relation, Mapping) or set(relation) != {
            "subject_id",
            "operator",
            "object_id",
            "polarity",
            "source_start",
            "source_end",
        }:
            return False
        if not all(
            isinstance(relation[field], str) for field in ("subject_id", "operator", "object_id")
        ):
            return False
        if relation["polarity"] not in {"positive", "negative"}:
            return False
        if not _is_integer(relation["source_start"]) or not _is_integer(relation["source_end"]):
            return False
        if not 0 <= relation["source_start"] <= relation["source_end"]:
            return False
    return True


def parse_output(call_type: str, raw_output: str) -> JsonDict:
    """Parse exactly one JSON object without repair or semantic completion."""

    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return {
            "parse_status": "failed",
            "parse_error": "invalid_json",
            "parsed": None,
            "unknown": False,
            "abstention": False,
        }
    if not isinstance(value, dict):
        return {
            "parse_status": "failed",
            "parse_error": "root_not_object",
            "parsed": None,
            "unknown": False,
            "abstention": False,
        }
    if call_type == "direct":
        if set(value) != {"decision"} or value.get("decision") not in {
            "supported",
            "unsupported",
            "abstain",
        }:
            return {
                "parse_status": "failed",
                "parse_error": "direct_decision_invalid",
                "parsed": None,
                "unknown": False,
                "abstention": False,
            }
        unknown = value["decision"] == "abstain"
    elif not _typed_shape_valid(value):
        return {
            "parse_status": "failed",
            "parse_error": "typed_output_invalid",
            "parsed": None,
            "unknown": False,
            "abstention": False,
        }
    else:
        unknown = bool(value["missing_fields"] or not value["relations"])
    return {
        "parse_status": "valid",
        "parse_error": None,
        "parsed": value,
        "unknown": unknown,
        "abstention": unknown,
    }


def _bytes_from_b64_or_json(encoded: Any, value: Any) -> bytes:
    if not isinstance(encoded, str):
        return canonical_json(value).encode("utf-8")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return canonical_json(value).encode("utf-8")


def build_completion_row(
    schedule_row: Mapping[str, Any],
    response: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
) -> JsonDict:
    """Conserve one cold request, response, parse result, and owner identity."""

    prompt = str(schedule_row.get("prompt", ""))
    raw_output = str(response.get("raw_output", ""))
    raw_request = deepcopy(response.get("raw_request", {}))
    raw_response = deepcopy(response.get("raw_response", {}))
    request_bytes = _bytes_from_b64_or_json(response.get("raw_request_bytes_b64"), raw_request)
    response_bytes = _bytes_from_b64_or_json(response.get("raw_response_bytes_b64"), raw_response)
    parsed = parse_output(str(schedule_row.get("call_type")), raw_output)
    completion_tokens = int(response.get("completion_tokens", 0) or 0)
    budget = int(schedule_row.get("output_token_budget", 0) or 0)
    finish_reason = response.get("finish_reason")
    request_error = response.get("error")
    return {
        "logical_order": schedule_row.get("logical_order"),
        "row_order": schedule_row.get("row_order"),
        "unit_id": schedule_row.get("unit_id"),
        "call_id": schedule_row.get("call_id"),
        "call_type": schedule_row.get("call_type"),
        "terminal_state": "request_error" if request_error else "response",
        "input_sha256": schedule_row.get("input_sha256"),
        "prompt": prompt,
        "prompt_bytes_b64": base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        "prompt_sha256": sha256_text(prompt),
        "prompt_template_sha256": schedule_row.get("prompt_template_sha256"),
        "raw_request": raw_request,
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_request_sha256": sha256_bytes(request_bytes),
        "raw_output": raw_output,
        "raw_output_bytes_b64": base64.b64encode(raw_output.encode("utf-8")).decode("ascii"),
        "raw_output_sha256": sha256_text(raw_output),
        "raw_response": raw_response,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_response_sha256": sha256_bytes(response_bytes),
        "output_token_budget": budget,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "truncated": finish_reason in {"length", "max_tokens"} or completion_tokens >= budget,
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "request_error": request_error,
        "parse_status": parsed["parse_status"],
        "parse_error": parsed["parse_error"],
        "parsed_output": deepcopy(parsed["parsed"]),
        "unknown": parsed["unknown"],
        "abstention": parsed["abstention"],
        "cold_request": True,
        "cache_hit": False,
        "cache_key": schedule_row.get("cache_key"),
        "reuse_from_call_id": None,
        "request_ordinal": response.get("request_ordinal"),
        "server_pid": resource_receipt.get("server_pid"),
        "server_pid_start_ticks": resource_receipt.get("server_pid_start_ticks"),
        "gpu_uuid": resource_receipt.get("gpu_uuid"),
        "lease_id": resource_receipt.get("lease_id"),
    }


def build_cache_hit_row(schedule_row: Mapping[str, Any], cold_row: Mapping[str, Any]) -> JsonDict:
    """Project one exact source result without pretending a request ran."""

    if schedule_row.get("call_type") != "source" or cold_row.get("call_type") != "source":
        raise ValueError("cache_source_only")
    if schedule_row.get("input_sha256") != cold_row.get("input_sha256"):
        raise ValueError("cache_input_hash_mismatch")
    row = deepcopy(dict(cold_row))
    row.update(
        {
            "logical_order": schedule_row.get("logical_order"),
            "row_order": schedule_row.get("row_order"),
            "unit_id": schedule_row.get("unit_id"),
            "call_id": schedule_row.get("call_id"),
            "input_sha256": schedule_row.get("input_sha256"),
            "prompt": schedule_row.get("prompt"),
            "prompt_bytes_b64": base64.b64encode(
                str(schedule_row.get("prompt", "")).encode("utf-8")
            ).decode("ascii"),
            "prompt_sha256": schedule_row.get("prompt_sha256"),
            "prompt_template_sha256": schedule_row.get("prompt_template_sha256"),
            "cold_request": False,
            "cache_hit": True,
            "cache_key": schedule_row.get("cache_key"),
            "reuse_from_call_id": cold_row.get("call_id"),
            "request_ordinal": None,
            "latency_s": 0.0,
        }
    )
    return row


def completion_row_errors(row: Mapping[str, Any], schedule_row: Mapping[str, Any]) -> list[str]:
    """Recompute exact bytes, parsing, isolation, and cache state."""

    errors: list[str] = []
    for field in ("logical_order", "row_order", "unit_id", "call_id", "call_type", "input_sha256"):
        if row.get(field) != schedule_row.get(field):
            errors.append(f"{field}_mismatch")
    prompt = str(row.get("prompt", ""))
    if prompt != schedule_row.get("prompt") or row.get("prompt_sha256") != sha256_text(prompt):
        errors.append("prompt_hash_mismatch")
    if row.get("prompt_bytes_b64") != base64.b64encode(prompt.encode("utf-8")).decode("ascii"):
        errors.append("prompt_bytes_mismatch")
    raw_output = str(row.get("raw_output", ""))
    if row.get("raw_output_sha256") != sha256_text(raw_output):
        errors.append("raw_output_hash_mismatch")
    if row.get("raw_output_bytes_b64") != base64.b64encode(raw_output.encode("utf-8")).decode(
        "ascii"
    ):
        errors.append("raw_output_bytes_mismatch")
    request_bytes = _bytes_from_b64_or_json(
        row.get("raw_request_bytes_b64"), row.get("raw_request")
    )
    response_bytes = _bytes_from_b64_or_json(
        row.get("raw_response_bytes_b64"), row.get("raw_response")
    )
    if row.get("raw_request_sha256") != sha256_bytes(request_bytes):
        errors.append("raw_request_hash_mismatch")
    if row.get("raw_response_sha256") != sha256_bytes(response_bytes):
        errors.append("raw_response_hash_mismatch")
    parsed = parse_output(str(row.get("call_type")), raw_output)
    for field in ("parse_status", "parse_error", "parsed_output", "unknown", "abstention"):
        expected = parsed["parsed"] if field == "parsed_output" else parsed[field]
        if row.get(field) != expected:
            errors.append(f"{field}_mismatch")
    cache_expected = bool(schedule_row.get("cache_hit_expected"))
    if (
        bool(row.get("cache_hit")) != cache_expected
        or bool(row.get("cold_request")) == cache_expected
    ):
        errors.append("cache_state_mismatch")
    if cache_expected and row.get("reuse_from_call_id") != schedule_row.get("reuse_from_call_id"):
        errors.append("cache_origin_mismatch")
    if row.get("output_token_budget") != schedule_row.get("output_token_budget"):
        errors.append("token_budget_mismatch")
    if row.get("terminal_state") not in {"response", "request_error"}:
        errors.append("terminal_state_invalid")
    return list(dict.fromkeys(errors))


def checkpoint_identity(
    schedule: Sequence[Mapping[str, Any]], public_sha256: str, model_sha256: str
) -> JsonDict:
    """Bind resume to public inputs, prompts, model, and decoding."""

    return {
        "schema": "carnot.exp7196.checkpoint_identity.v1",
        "schedule_sha256": sha256_json(list(schedule)),
        "public_view_sha256": public_sha256,
        "model_sha256": model_sha256,
        "contract_sha256": PINNED_CONTRACT_SHA256,
        "decoding_parameters_sha256": sha256_json(DECODING_PARAMETERS),
        "random_seed": RANDOM_SEED,
    }


def write_checkpoint(
    path: Path, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Publish complete receipts atomically before later model calls."""

    payload = {
        "schema": "carnot.exp7196.checkpoint.v1",
        "identity": deepcopy(dict(identity)),
        "row_count": len(rows),
        "rows": [deepcopy(dict(row)) for row in rows],
        "row_hashes": [sha256_json(row) for row in rows],
    }
    atomic_write_json(path, payload, allow_override=False, sort_keys=True)
    return payload


def resume_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> list[JsonDict]:
    """Reject changed input, model, contract, or terminal row bytes."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint_identity_mismatch")
    rows = list(value.get("rows") or [])
    if value.get("row_count") != len(rows):
        raise ValueError("checkpoint_row_count_mismatch")
    if value.get("row_hashes") != [sha256_json(row) for row in rows]:
        raise ValueError("checkpoint_row_hash_mismatch")
    return [deepcopy(dict(row)) for row in rows]


def summarize_capture(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count failures on fixed denominators and separate cold from reused cost."""

    logical = list(rows)
    per_type: JsonDict = {}
    for call_type in ("source", "claim", "direct"):
        selected = [row for row in logical if row.get("call_type") == call_type]
        per_type[call_type] = {
            "denominator": 192,
            "completed_count": len(selected),
            "invalid_count": sum(row.get("parse_status") != "valid" for row in selected),
            "truncated_count": sum(bool(row.get("truncated")) for row in selected),
            "unknown_count": sum(bool(row.get("unknown")) for row in selected),
            "request_error_count": sum(bool(row.get("request_error")) for row in selected),
        }
    cold = [row for row in logical if row.get("cold_request") is True]
    prompt_tokens = sum(int(row.get("prompt_tokens", 0) or 0) for row in cold)
    completion_tokens = sum(int(row.get("completion_tokens", 0) or 0) for row in cold)
    latency = sum(float(row.get("latency_s", 0.0) or 0.0) for row in cold)
    return {
        "logical_receipt_count": len(logical),
        "cold_request_count": len(cold),
        "source_cache_hit_count": sum(bool(row.get("cache_hit")) for row in logical),
        "per_call_type": per_type,
        "cold_costs": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_s": latency,
        },
        "amortized_costs": {
            "cold_requests_per_public_row": len(cold) / 192,
            "prompt_tokens_per_public_row": prompt_tokens / 192,
            "completion_tokens_per_public_row": completion_tokens / 192,
            "latency_s_per_public_row": latency / 192,
        },
    }


def classify_terminal(
    checks: Sequence[Mapping[str, Any]],
    model_loaded: bool,
    canary: Mapping[str, Any] | None,
    completion_rows: Sequence[Mapping[str, Any]],
    provenance_ok: bool,
    duration_s: float,
) -> JsonDict:
    """Classify measured work without padding a duration or discarding errors."""

    failed = first_failed_gate(checks)
    rows = list(completion_rows)
    if failed is not None and not model_loaded and canary is None and not rows:
        return {
            "status": "blocked",
            "inference_substrate_class": "blocked_no_run",
            "duration_floor_s": 0.0,
            "duration_floor_met": True,
            "score": 0,
            "verdict_class": "blocked",
        }
    if canary is None and not rows:
        substrate_class = "model_load_no_generation"
        floor = 2.0
        status = "partial"
        verdict = "partial" if duration_s >= floor else "disqualified"
        return {
            "status": status,
            "inference_substrate_class": substrate_class,
            "duration_floor_s": floor,
            "duration_floor_met": duration_s >= floor,
            "score": 0,
            "verdict_class": verdict,
        }
    if not rows:
        floor = 10.0
        return {
            "status": "partial",
            "inference_substrate_class": "model_bounded_generation",
            "duration_floor_s": floor,
            "duration_floor_met": duration_s >= floor,
            "score": 0,
            "verdict_class": "partial" if duration_s >= floor else "disqualified",
        }
    floor = 60.0
    complete = len(rows) == LOGICAL_RECEIPT_COUNT and all(
        row.get("terminal_state") in {"response", "request_error"} for row in rows
    )
    parse_poor = any(
        row.get("parse_status") != "valid"
        or bool(row.get("truncated"))
        or bool(row.get("request_error"))
        for row in rows
    )
    evidence_ok = complete and provenance_ok and failed is None and duration_s >= floor
    verdict = (
        ("null" if parse_poor else "positive")
        if evidence_ok
        else ("disqualified" if complete else "partial")
    )
    return {
        "status": "complete" if complete else "partial",
        "inference_substrate_class": "model_full_generation",
        "duration_floor_s": floor,
        "duration_floor_met": duration_s >= floor,
        "score": int(evidence_ok),
        "verdict_class": verdict,
    }


def build_result_rows(completion_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project three call receipts into one comparison row without scoring truth."""

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in completion_rows:
        grouped.setdefault(int(row.get("row_order", -1)), []).append(row)
    result: list[JsonDict] = []
    for row_order in sorted(grouped):
        calls = grouped[row_order]
        errors = [
            f"{row.get('call_type')}:{row.get('request_error') or row.get('parse_error')}"
            for row in calls
            if row.get("request_error") or row.get("parse_error")
        ]
        result.append(
            {
                "unit_id": calls[0].get("unit_id"),
                "row_order": row_order,
                "arm": "atomic_separate_capture",
                "seed": RANDOM_SEED,
                "metric": int(
                    len(calls) == 3
                    and all(
                        row.get("terminal_state") in {"response", "request_error"} for row in calls
                    )
                ),
                "error": errors or None,
                "abstention": any(bool(row.get("abstention")) for row in calls),
                "call_ids": [row.get("call_id") for row in calls],
            }
        )
    return result


def _source_artifact_hashes(
    root: Path, *, raw_manifest: Path | None = None, model_sha256: str | None = None
) -> JsonDict:
    """Bind public inputs, prompts, implementation, and runtime contracts."""

    paths = {
        "exp7195_artifact": root / UPSTREAM_PATH,
        "exp7195_public_view": root / PUBLIC_VIEW_PATH,
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
    hashes = {
        name: sha256_file(path) if path.is_file() else "missing" for name, path in paths.items()
    }
    manifest_path = raw_manifest or root / RAW_DIR / RAW_MANIFEST_NAME
    hashes.update(
        {
            "raw_manifest": sha256_file(manifest_path) if manifest_path.is_file() else "missing",
            "model_gguf": model_sha256 or "missing",
            "frozen_generation_contract": PINNED_CONTRACT_SHA256,
            "decoding_contract": sha256_json(DECODING_PARAMETERS),
            "source_prompt_template": sha256_text(
                "Extract typed entities and relations from SOURCE only. SOURCE:\n{source_text}"
            ),
            "claim_prompt_template": sha256_text(
                "Extract one typed relation from CLAIM only. CLAIM:\n{claim_text}"
            ),
            "direct_prompt_template": sha256_text(
                "Judge support from SOURCE and CLAIM only. SOURCE:\n{source_text}\nCLAIM:\n{claim_text}"
            ),
        }
    )
    return hashes


def base_artifact(run_date: str, *, root: Path | None = None) -> JsonDict:
    """Create every required field before a fallible resource check."""

    repository = root or Path(__file__).resolve().parents[2]
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "preflight_pending_native_llama_cpp",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "source_artifact_hashes": _source_artifact_hashes(repository),
        "rows": [],
        "sample_size_budget": {
            "planned_rows": 192,
            "completed_rows": 0,
            "independent_units": 192,
            "planned_logical_receipts": LOGICAL_RECEIPT_COUNT,
            "completed_logical_receipts": 0,
            "planned_cold_requests": EXPECTED_COLD_REQUEST_COUNT,
            "completed_cold_requests": 0,
            "planned_source_cache_hits": EXPECTED_SOURCE_CACHE_HITS,
            "completed_source_cache_hits": 0,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_row(
            "atomic_capture_complete",
            LOGICAL_RECEIPT_COUNT,
            0,
            False,
            upstream="experiment_7196",
            field="completion_rows",
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_atomic_capture",
        "atomic_capture_complete_score": 0,
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "completion_rows": [],
        "phase_spans": [],
        "gpu_receipts": {},
        "runner_receipt": {"model_count": 1, "replica_count": 0, "runner": None},
        "raw_manifest": {},
        "inference_mode": "not_run",
        "canary_receipt": None,
        "cost_summary": summarize_capture([]),
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
    """Cold-check terminal state, raw rows, costs, ownership, and hashes."""

    artifact = _load_artifact(value)
    if artifact is None:
        return ["artifact_unreadable"]
    if artifact.get("status") == "running":
        return ["status_not_terminal"]
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
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    checks = list(artifact.get("preconditions_checked") or [])
    failed = first_failed_gate(checks)
    if artifact.get("status") == "blocked":
        if failed is None:
            errors.append("blocked_failed_gate_missing")
        elif artifact.get("gate_check_summary") != failed:
            errors.append("blocked_gate_summary_mismatch")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("atomic_capture_complete_score") != 0:
            errors.append("blocked_score_mismatch")
        if artifact.get("verdict_class") != "blocked":
            errors.append("blocked_verdict_mismatch")
        return list(dict.fromkeys(errors))

    manifest = artifact.get("raw_manifest")
    schedule = list(manifest.get("schedule") or []) if isinstance(manifest, Mapping) else []
    rows = list(artifact.get("completion_rows") or [])
    if len(schedule) != LOGICAL_RECEIPT_COUNT:
        errors.append("schedule_missing_or_incomplete")
    if artifact.get("rows") != build_result_rows(rows):
        errors.append("result_rows_mismatch")
    for index, (row, sealed) in enumerate(zip(rows, schedule, strict=False)):
        errors.extend(f"row_{index}:{problem}" for problem in completion_row_errors(row, sealed))
    summary = summarize_capture(rows)
    if artifact.get("cost_summary") != summary:
        errors.append("cost_summary_mismatch")
    gpu = dict(artifact.get("gpu_receipts") or {})
    runner = dict(artifact.get("runner_receipt") or {})
    expected = classify_terminal(
        checks,
        bool(runner.get("model_loaded")),
        artifact.get("canary_receipt"),
        rows,
        bool(gpu.get("provenance_ok")),
        float(artifact.get("duration_s", 0.0) or 0.0),
    )
    for field, expected_value in (
        ("status", expected["status"]),
        ("inference_substrate_class", expected["inference_substrate_class"]),
        ("verdict_class", expected["verdict_class"]),
        ("atomic_capture_complete_score", expected["score"]),
    ):
        if artifact.get(field) != expected_value:
            errors.append(f"{field}_mismatch")
    if runner.get("model_count") != 1 or runner.get("runner") == "DualGPURunner":
        errors.append("runner_model_count_mismatch")
    if artifact.get("status") == "complete" and artifact.get("inference_mode") != "live_gpu":
        errors.append("complete_inference_mode_mismatch")
    if check_source_hashes and isinstance(artifact.get("source_artifact_hashes"), Mapping):
        root = find_repo_root()
        errors.extend(
            f"schedule_replay:{problem}"
            for problem in frozen_schedule_replay_errors(root, schedule)
        )
        manifest_path = Path(str(manifest.get("path", root / RAW_DIR / RAW_MANIFEST_NAME)))
        model_specs = list(artifact.get("model_specs") or [])
        model_hash = model_specs[0].get("sha256") if model_specs else None
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
        canonical_json({"experiment": 7196, "phase": phase, "event": event, **fields}), flush=True
    )


@contextmanager
def _phase_span(spans: list[JsonDict], phase: int, name: str) -> Iterator[None]:  # pragma: no cover
    """Measure one phase with a monotonic clock and visible boundaries."""

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
    """Keep visible progress outside a blocking native operation."""

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

    thread = threading.Thread(target=emit, name=f"exp7196-heartbeat-{phase}", daemon=True)
    thread.start()
    _progress(
        phase,
        "heartbeat_start",
        operation=operation,
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
    """Stream the owned child log while llama.cpp is active."""

    stop = threading.Event()

    def stream() -> None:
        offset = 0
        while not stop.wait(0.25):
            if not supervisor.log_path.is_file():
                continue
            with supervisor.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    print(f"exp7196 native-server {line.rstrip()}", flush=True)
                offset = handle.tell()

    thread = threading.Thread(target=stream, name="exp7196-native-log", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _content_addressed_hash(path: Path) -> str | None:  # pragma: no cover
    """Use the immutable cache object's digest without scanning model tensors."""

    try:
        target_name = path.resolve(strict=True).name.lower()
    except OSError:
        return None
    return "sha256:" + target_name if re.fullmatch(r"[0-9a-f]{64}", target_name) else None


def _record_check(
    checks: list[JsonDict], row_factory: Callable[[], JsonDict]
) -> JsonDict:  # pragma: no cover
    """Print before and after every precondition, including failures."""

    _progress(1, "check_start", completed_checks=len(checks))
    row = row_factory()
    checks.append(row)
    _progress(1, "check_end", check=row["check"], passed=row["passed"])
    return row


def _load_public_rows(public_bytes: bytes) -> list[JsonDict]:  # pragma: no cover
    return [json.loads(line) for line in public_bytes.splitlines()]


def _runner_supports(runner: Mapping[str, Any], flag: str) -> bool:  # pragma: no cover
    receipts = list(runner.get("command_receipts") or [])
    help_text = ""
    if len(receipts) >= 2:
        help_text = f"{receipts[1].get('stdout', '')}\n{receipts[1].get('stderr', '')}"
    return flag in help_text


def _collect_preflight(
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Check bytes, quarantine, tools, cache, outputs, and idle GPUs."""

    checks: list[JsonDict] = []
    context: JsonDict = {}
    _progress(1, "check_start", check="run_date")
    checks.append(gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE, field="run_date"))
    _progress(1, "check_end", check="run_date", passed=checks[-1]["passed"])
    required = {
        "upstream_artifact": root / UPSTREAM_PATH,
        "public_view": root / PUBLIC_VIEW_PATH,
        "constraint_spec": root / SPEC_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "focused_tests": root / TEST_PATH,
        "research_program": root / "research-program.md",
        "research_references": root / "research-references.md",
        "exclusion_manifest": root / "ops/exclusion_manifest.yaml",
        "e2e_test_plan": root / "ops/e2e-test-plan.md",
    }
    _progress(1, "check_start", check="required_source_bytes")
    source_state = {
        name: path.is_file() and os.access(path, os.R_OK) for name, path in required.items()
    }
    source_state["spec_has_req"] = bool(
        source_state["constraint_spec"]
        and "REQ-VERIFY-7196" in required["constraint_spec"].read_text(encoding="utf-8")
    )
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
    _progress(1, "check_end", check="required_source_bytes", passed=checks[-1]["passed"])
    if not all(source_state.values()):
        return checks, [], context

    _progress(1, "check_start", check="exact_upstream_source_bytes")
    observed_hashes = {
        "artifact": sha256_file(required["upstream_artifact"]),
        "public_view": sha256_file(required["public_view"]),
    }
    expected_hashes = {"artifact": PINNED_UPSTREAM_SHA256, "public_view": PINNED_PUBLIC_SHA256}
    checks.append(
        gate_row(
            "exact_upstream_source_bytes",
            expected_hashes,
            observed_hashes,
            observed_hashes == expected_hashes,
            upstream="experiment_7195",
            field="source_bytes",
        )
    )
    _progress(1, "check_end", check="exact_upstream_source_bytes", passed=checks[-1]["passed"])
    if checks[-1]["passed"] is not True:
        return checks, [], context

    _progress(1, "check_start", check="upstream_parse")
    public_bytes = required["public_view"].read_bytes()
    try:
        upstream = json.loads(required["upstream_artifact"].read_text(encoding="utf-8"))
        manifest_data = yaml.safe_load(required["exclusion_manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "upstream_parse",
                "valid_json_and_yaml",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="experiment_7195",
                field="source_documents",
            )
        )
        _progress(1, "check_end", check="upstream_parse", passed=False)
        return checks, [], context
    checks.append(
        gate_row(
            "upstream_parse",
            "valid_json_and_yaml",
            "valid_json_and_yaml",
            True,
            upstream="experiment_7195",
            field="source_documents",
        )
    )
    _progress(1, "check_end", check="upstream_parse", passed=True)
    for upstream_row in upstream_gate_rows(
        upstream, public_bytes, exclusion_manifest=manifest_data
    ):
        _progress(1, "check_start", check=upstream_row["check"])
        checks.append(upstream_row)
        _progress(1, "check_end", check=upstream_row["check"], passed=upstream_row["passed"])
        if upstream_row["passed"] is not True:
            return checks, [], context

    contract = dict(upstream["fixture_manifest"]["frozen_generation_contract"])
    _progress(1, "check_start", check="blind_atomic_schedule")
    try:
        public_rows = _load_public_rows(public_bytes)
        schedule = build_schedule(public_rows, contract)
        schedule_problem = schedule_errors(schedule, public_rows, contract)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        public_rows = []
        schedule = []
        schedule_problem = [f"{type(exc).__name__}:{exc}"]
    checks.append(
        gate_row(
            "blind_atomic_schedule",
            {"logical_calls": 576, "cold_requests": 481, "cache_hits": 95, "errors": []},
            {
                "logical_calls": len(schedule),
                "cold_requests": sum(bool(row.get("cold_request_expected")) for row in schedule),
                "cache_hits": sum(bool(row.get("cache_hit_expected")) for row in schedule),
                "errors": schedule_problem,
            },
            not schedule_problem,
            upstream=PUBLIC_VIEW_PATH.as_posix(),
            field="rows",
        )
    )
    _progress(1, "check_end", check="blind_atomic_schedule", passed=checks[-1]["passed"])

    _progress(1, "check_start", check="writable_output_storage")
    storage = {
        "result_parent": result_path.parent.is_dir() and os.access(result_path.parent, os.W_OK),
        "checkpoint_dir": checkpoint_dir.is_dir() and os.access(checkpoint_dir, os.W_OK),
        "raw_dir": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
    }
    checks.append(
        gate_row(
            "writable_output_storage",
            {name: True for name in storage},
            storage,
            all(storage.values()),
            upstream="host_filesystem",
            field="output_paths",
        )
    )
    _progress(1, "check_end", check="writable_output_storage", passed=checks[-1]["passed"])

    _progress(1, "check_start", check="force_live_environment")
    live_value = os.environ.get("CARNOT_FORCE_LIVE")
    checks.append(
        gate_row(
            "force_live_environment",
            "1",
            live_value,
            live_value == "1",
            upstream="process_environment",
            field="CARNOT_FORCE_LIVE",
        )
    )
    _progress(1, "check_end", check="force_live_environment", passed=checks[-1]["passed"])

    _progress(1, "check_start", check="mandated_gguf_cache_and_embedded_metadata")
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
        "embedded_tokenizer_metadata_sha256": metadata.get("metadata_summary_sha256"),
        "embedded_chat_template_sha256": metadata.get("chat_template_sha256"),
        "embedded_chat_template_present": metadata.get("chat_template_present") is True,
        "tokenizer_detail": metadata.get("tokenizer_detail"),
        "auto_tokenizer_used": False,
    }
    model_ok = bool(
        model_exists
        and resolved
        and resolved.get("hf_id") == QWEN_MODEL_ID
        and model_path
        and QUANTIZATION.lower() in model_path.name.lower()
        and model_hash
        and metadata.get("metadata_summary_sha256")
        and metadata.get("chat_template_present") is True
    )
    checks.append(
        gate_row(
            "mandated_gguf_cache_and_embedded_metadata",
            {
                "hf_id": QWEN_MODEL_ID,
                "quantization": QUANTIZATION,
                "exists": True,
                "content_addressed": True,
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
                "auto_tokenizer_used": False,
            },
            {
                "hf_id": model_spec["hf_id"],
                "quantization": model_spec["quantization"],
                "exists": model_exists,
                "content_addressed": bool(model_hash),
                "embedded_tokenizer": bool(model_spec["embedded_tokenizer_metadata_sha256"]),
                "embedded_chat_template": model_spec["embedded_chat_template_present"],
                "auto_tokenizer_used": False,
            },
            model_ok,
            upstream="cached_current_model",
            field="model_path",
        )
    )
    _progress(
        1,
        "check_end",
        check="mandated_gguf_cache_and_embedded_metadata",
        passed=checks[-1]["passed"],
    )

    server = resolve_native_llama_server()
    _progress(2, "phase_start", name="native_runner_capability_subprocesses")
    _progress(2, "subprocess_start", operation="runner_version_help_linkage", path=str(server))
    runner_rows = shipped_runtime.lease_preflight.collect_runner_capabilities(server)
    _progress(2, "subprocess_end", operation="runner_version_help_linkage")
    runner_errors = shipped_runtime.lease_preflight.runner_capability_errors(runner_rows)
    runner = deepcopy(runner_rows[0]) if runner_rows else {}
    non_thinking_supported = _runner_supports(runner, "--reasoning")
    _progress(1, "check_start", check="native_cuda_runtime")
    checks.append(
        gate_row(
            "native_cuda_runtime",
            {"errors": [], "json_grammar": True, "non_thinking_supported": True},
            {
                "errors": runner_errors,
                "json_grammar": runner.get("grammar_or_json_output") is True,
                "non_thinking_supported": non_thinking_supported,
            },
            not runner_errors
            and runner.get("grammar_or_json_output") is True
            and non_thinking_supported,
            upstream="native_llama_server",
            field="cuda_linkage_and_capabilities",
        )
    )
    _progress(1, "check_end", check="native_cuda_runtime", passed=checks[-1]["passed"])
    _progress(2, "phase_end", name="native_runner_capability_subprocesses")

    _progress(3, "phase_start", name="read_only_gpu_and_lease_conflict_check")
    _progress(3, "subprocess_start", operation="gpu_and_lease_inventory")
    process_rows, query_receipts = shipped_runtime.lease_preflight.collect_gpu_process_rows()
    lease_rows = shipped_runtime.lease_preflight.scan_lease_rows(
        shipped_runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = shipped_runtime.lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = shipped_runtime.lease_preflight.readiness_decision(
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
    _progress(3, "subprocess_end", operation="gpu_and_lease_inventory")
    query_ok = all(receipt.get("returncode") == 0 for receipt in query_receipts)
    _progress(1, "check_start", check="gpu_inventory_queries")
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
    _progress(1, "check_end", check="gpu_inventory_queries", passed=checks[-1]["passed"])
    available = list(decision.get("available_gpu_uuids") or [])
    _progress(1, "check_start", check="idle_task_ownable_rtx_3090")
    checks.append(
        gate_row(
            "idle_task_ownable_rtx_3090",
            {"minimum_count": 1, "no_unowned_compute": True, "no_conflicting_lease": True},
            {
                "available_gpu_uuids": available,
                "conflicting_processes": decision.get("conflicting_processes", []),
                "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
            },
            bool(available),
            upstream="live_gpu_and_lease_inventory",
            field="available_gpu_uuids",
        )
    )
    _progress(1, "check_end", check="idle_task_ownable_rtx_3090", passed=checks[-1]["passed"])
    _progress(3, "phase_end", name="read_only_gpu_and_lease_conflict_check")
    context.update(
        {
            "upstream": upstream,
            "contract": contract,
            "public_rows": public_rows,
            "public_view_sha256": sha256_bytes(public_bytes),
            "schedule": schedule,
            "model_spec": model_spec,
            "model_path": model_path,
            "server_path": server,
            "runner": runner,
            "non_thinking_supported": non_thinking_supported,
            "process_rows": classified,
            "lease_rows": lease_rows,
            "query_receipts": query_receipts,
            "available_gpu_uuids": available,
        }
    )
    return checks, schedule, context


def _free_port() -> int:  # pragma: no cover
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(
    server: Path, model: Path, port: int, *, non_thinking_supported: bool
) -> list[str]:  # pragma: no cover
    """Use one GPU and the chat template embedded in the GGUF."""

    command = [
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
    ]
    if non_thinking_supported:
        command.extend(["--reasoning", "off"])
    return [*command, "--no-webui", "--log-verbosity", "3"]


def _wait_for_health(
    supervisor: NativeLlamaServerSupervisor, port: int, timeout_s: float
) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    deadline = started + timeout_s
    last_error = "not_started"
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        if supervisor.proc and supervisor.proc.poll() is not None:
            return {"ok": False, "classification": "early_exit", "attempts": attempts}
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


def _request_payload(sealed: Mapping[str, Any]) -> tuple[JsonDict, bytes]:  # pragma: no cover
    payload: JsonDict = {
        "messages": [
            {
                "role": "system",
                "content": "Return only one JSON object that matches the supplied syntax.",
            },
            {"role": "user", "content": str(sealed["prompt"])},
        ],
        **DECODING_PARAMETERS,
        "max_tokens": int(sealed["output_token_budget"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"atomic_{sealed['call_type']}",
                "schema": deepcopy(sealed["response_schema"]),
            },
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload, encoded


def _chat_request(
    port: int, sealed: Mapping[str, Any], *, timeout_s: float, request_ordinal: int
) -> JsonDict:  # pragma: no cover
    payload, encoded = _request_payload(sealed)
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with request.urlopen(http_request, timeout=timeout_s) as response:
        response_bytes = response.read()
    body = json.loads(response_bytes.decode("utf-8"))
    choice = dict(list(body.get("choices") or [{}])[0])
    message = dict(choice.get("message") or {})
    usage = dict(body.get("usage") or {})
    return {
        "raw_request": payload,
        "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
        "raw_output": str(message.get("content") or message.get("reasoning_content") or ""),
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_s": time.monotonic() - started,
        "finish_reason": choice.get("finish_reason"),
        "error": None,
        "request_ordinal": request_ordinal,
    }


def _request_or_error(
    port: int, sealed: Mapping[str, Any], *, timeout_s: float, request_ordinal: int
) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    payload, encoded = _request_payload(sealed)
    try:
        return _chat_request(port, sealed, timeout_s=timeout_s, request_ordinal=request_ordinal)
    except Exception as exc:  # noqa: BLE001 - exact failure is part of the terminal receipt.
        return {
            "raw_request": payload,
            "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "raw_output": "",
            "raw_response": {},
            "raw_response_bytes_b64": base64.b64encode(b"").decode("ascii"),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_s": time.monotonic() - started,
            "finish_reason": None,
            "error": f"{type(exc).__name__}:{exc}",
            "request_ordinal": request_ordinal,
        }


def _generation_overlap_sample(
    holder: list[JsonDict], identity: Mapping[str, Any], gpu_uuid: str, call_id: str
) -> threading.Thread:  # pragma: no cover
    """Sample GPU state concurrently so provenance overlaps generation."""

    def sample() -> None:
        started = time.monotonic()
        snapshot = shipped_runtime._gpu_snapshot("during_generation", phase=7)
        ended = time.monotonic()
        pid = int(identity.get("pid", -1))
        owned = [
            deepcopy(dict(app))
            for app in snapshot.get("compute_apps", [])
            if int(app.get("pid", -2)) == pid
            and app.get("owned_by_task") is True
            and int(app.get("used_memory_mb", 0) or 0) > 0
        ]
        holder.append(
            {
                "call_id": call_id,
                "gpu_uuid": gpu_uuid,
                "server_pid": pid,
                "sample_start_monotonic_s": started,
                "sample_end_monotonic_s": ended,
                "owned_compute_apps": owned,
                "task_owned_vram_mb": sum(int(app["used_memory_mb"]) for app in owned),
            }
        )

    thread = threading.Thread(target=sample, name=f"exp7196-gpu-sample-{call_id}", daemon=True)
    thread.start()
    return thread


def _write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    status: str,
) -> JsonDict:  # pragma: no cover
    manifest = {
        "schema": "carnot.exp7196.raw_manifest.v1",
        "status": status,
        "path": str(raw_dir / RAW_MANIFEST_NAME),
        "worker_input_paths": [PUBLIC_VIEW_PATH.as_posix()],
        "forbidden_input_paths": [AUTHORITY_PATH.as_posix()],
        "forbidden_input_open_count": 0,
        "schedule": [deepcopy(dict(row)) for row in schedule],
        "schedule_sha256": sha256_json(list(schedule)),
        "checkpoint_identity": deepcopy(dict(identity)),
        "checkpoint_receipts": [deepcopy(dict(row)) for row in checkpoint_rows],
        "raw_rows": [
            {
                "logical_order": row.get("logical_order"),
                "unit_id": row.get("unit_id"),
                "call_id": row.get("call_id"),
                "call_type": row.get("call_type"),
                "path": str(raw_dir / f"call_{int(row.get('logical_order', 0)):03d}.json"),
                "row_sha256": sha256_json(row),
                "raw_output_sha256": row.get("raw_output_sha256"),
                "terminal_state": row.get("terminal_state"),
                "cache_hit": row.get("cache_hit"),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(raw_dir / RAW_MANIFEST_NAME, manifest, allow_override=False, sort_keys=True)
    return manifest


def _live_capture(
    context: Mapping[str, Any],
    checkpoint_dir: Path,
    raw_dir: Path,
    spans: list[JsonDict],
) -> JsonDict:  # pragma: no cover
    """Acquire one GPU, run all logical calls, and close only owned state."""

    schedule = list(context["schedule"])
    gpu_uuid = str(context["available_gpu_uuids"][0])
    device = shipped_runtime._selected_device(context, gpu_uuid)
    gpu_index = int(device["gpu_index"])
    model = Path(str(context["model_path"]))
    model_spec = deepcopy(dict(context["model_spec"]))
    port = _free_port()
    command = _server_command(
        Path(context["server_path"]),
        model,
        port,
        non_thinking_supported=bool(context["non_thinking_supported"]),
    )
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
    model_loaded = False
    canary: JsonDict | None = None
    rows: list[JsonDict] = []
    checkpoints: list[JsonDict] = []
    snapshots: list[JsonDict] = []
    overlap_samples: list[JsonDict] = []
    provenance: JsonDict = {"provenance_ok": False}
    cleanup: JsonDict = {"action": "not_started", "leak_free": True}
    lease_release: JsonDict = {"released": False}
    runtime_error: str | None = None
    load_duration = 0.0
    native_generation_s = 0.0
    parse_persist_s = 0.0
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    checkpoint_id = checkpoint_identity(
        schedule, str(context["public_view_sha256"]), str(model_spec["sha256"])
    )
    capture_started = 0.0
    try:
        with _phase_span(spans, 5, "task_owned_lease_and_model_load"):
            snapshots.append(shipped_runtime._gpu_snapshot("before_model_load", phase=5))
            lease = lease_api.GpuLease.acquire(
                runtime_dir=shipped_runtime.lease_preflight.LEASE_RUNTIME_DIR,
                task_id=TASK_ID,
                device_uuid=gpu_uuid,
                expected_model=str(model),
                vram_before_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
                ttl_s=900.0,
            )
            lease.transition("admitted")
            lease.transition("loading")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            _progress(5, "subprocess_start", operation="owned_native_llama_server", command=command)
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
            model_loaded = True
            resident = shipped_runtime._gpu_snapshot("model_resident", phase=5)
            snapshots.append(resident)
            provenance = shipped_runtime._gpu_provenance(
                resident, identity, gpu_uuid, supervisor.stderr_tail()
            )
            if provenance.get("provenance_ok") is not True:
                raise RuntimeError("cuda_placement_or_owned_vram_unconfirmed")
            lease.transition("resident", vram_mb=int(provenance["task_owned_vram_mb"]))
            lease.transition("inferencing")

        resource = {
            "server_pid": identity.get("pid"),
            "server_pid_start_ticks": identity.get("start_time_ticks"),
            "gpu_uuid": gpu_uuid,
            "lease_id": lease.lease_id,
        }
        with _phase_span(spans, 6, "eight_token_bounded_canary"):
            canary_sealed = {
                "call_type": "canary",
                "prompt": "Return one JSON object with canary_ok set to true.",
                "output_token_budget": CANARY_TOKEN_BUDGET,
                "response_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["canary_ok"],
                    "properties": {"canary_ok": {"type": "boolean"}},
                },
            }
            _progress(6, "generation_start", operation="canary", max_tokens=8)
            with _heartbeat(6, "canary_generation", lambda: 0, 1):
                canary_response = _request_or_error(
                    port,
                    canary_sealed,
                    timeout_s=REQUEST_CAP_S,
                    request_ordinal=-1,
                )
            canary = {
                **canary_response,
                "terminal_state": "request_error" if canary_response.get("error") else "response",
                "max_tokens": CANARY_TOKEN_BUDGET,
                "model_bounded_generation": True,
            }
            atomic_write_json(raw_dir / "canary.json", canary, allow_override=False, sort_keys=True)
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
            _progress(7, "checkpoint_resume", completed_units=len(rows), total_units=576)
        by_call_id = {str(row["call_id"]): row for row in rows}
        request_ordinal = sum(row.get("cold_request") is True for row in rows)
        capture_started = time.monotonic()
        last_loop_report = capture_started
        with _phase_span(spans, 7, "atomic_192_row_capture"):
            while len(rows) < len(schedule):
                elapsed = time.monotonic() - capture_started
                remaining = CAPTURE_CAP_S - elapsed
                if remaining <= 0.25:
                    runtime_error = "capture_cap_reached"
                    break
                index = len(rows)
                sealed = schedule[index]
                if time.monotonic() - last_loop_report >= 45.0:
                    _progress(
                        7,
                        "loop_progress",
                        completed_units=index,
                        total_units=576,
                        completed_public_rows=index // 3,
                        elapsed_s=round(elapsed, 3),
                    )
                    last_loop_report = time.monotonic()
                parse_started = time.monotonic()
                if sealed["cache_hit_expected"]:
                    cold = by_call_id[str(sealed["reuse_from_call_id"])]
                    row = build_cache_hit_row(sealed, cold)
                    _progress(
                        7,
                        "cache_hit",
                        call_id=sealed["call_id"],
                        reuse_from=sealed["reuse_from_call_id"],
                        completed_units=index,
                        total_units=576,
                    )
                else:
                    lease.heartbeat()
                    _progress(
                        7,
                        "generation_start",
                        operation="atomic_call",
                        call_id=sealed["call_id"],
                        call_type=sealed["call_type"],
                        completed_units=index,
                        total_units=576,
                    )
                    sample_thread = None
                    sample_offset = len(overlap_samples)
                    if request_ordinal % CHECKPOINT_CADENCE == 0:
                        sample_thread = _generation_overlap_sample(
                            overlap_samples, identity, gpu_uuid, str(sealed["call_id"])
                        )
                    with _heartbeat(7, "atomic_generation", lambda: len(rows), 576):
                        response = _request_or_error(
                            port,
                            sealed,
                            timeout_s=min(REQUEST_CAP_S, max(0.25, remaining)),
                            request_ordinal=request_ordinal,
                        )
                    request_ended = time.monotonic()
                    native_generation_s += float(response.get("latency_s", 0.0) or 0.0)
                    if sample_thread is not None:
                        sample_thread.join(timeout=10.0)
                        for sample in overlap_samples[sample_offset:]:
                            sample["request_end_monotonic_s"] = request_ended
                            sample["request_interval_overlap"] = (
                                sample["sample_start_monotonic_s"] <= request_ended
                            )
                    row = build_completion_row(sealed, response, resource)
                    request_ordinal += 1
                    _progress(
                        7,
                        "generation_end",
                        operation="atomic_call",
                        call_id=sealed["call_id"],
                        terminal_state=row["terminal_state"],
                        parse_status=row["parse_status"],
                        completion_tokens=row["completion_tokens"],
                        duration_s=round(float(row["latency_s"]), 6),
                    )
                rows.append(row)
                by_call_id[str(row["call_id"])] = row
                atomic_write_json(
                    raw_dir / f"call_{index:03d}.json",
                    {"schedule": deepcopy(dict(sealed)), "completion": row},
                    allow_override=False,
                    sort_keys=True,
                )
                latest_payload = write_checkpoint(latest, checkpoint_id, rows)
                if len(rows) % CHECKPOINT_CADENCE == 0:
                    path = checkpoint_dir / f"checkpoint_{len(rows):03d}.json"
                    _progress(7, "checkpoint_write_start", path=str(path), rows=len(rows))
                    payload = write_checkpoint(path, checkpoint_id, rows)
                    checkpoints.append(
                        {
                            "path": str(path),
                            "row_count": len(rows),
                            "checkpoint_sha256": sha256_json(payload),
                            "row_hashes_sha256": sha256_json(payload["row_hashes"]),
                        }
                    )
                    _progress(7, "checkpoint_write_end", path=str(path), rows=len(rows))
                else:
                    _ = latest_payload
                parse_persist_s += time.monotonic() - parse_started - float(row["latency_s"])
    except Exception as exc:  # noqa: BLE001 - preserve real partial work and classify below.
        runtime_error = f"{type(exc).__name__}:{exc}"
        _progress(7, "runtime_error", error=runtime_error, completed_units=len(rows))
    finally:
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
        with _phase_span(spans, 8, "owned_teardown"):
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
            after = shipped_runtime._gpu_snapshot("after_cleanup", phase=8)
            snapshots.append(after)
            overlap_ok = any(
                sample.get("request_interval_overlap") is True
                and bool(sample.get("owned_compute_apps"))
                for sample in overlap_samples
            )
            if lease is not None:
                try:
                    lease_release = shipped_runtime._release_lease(
                        lease,
                        cleanup,
                        after,
                        identity,
                        len(rows) == 576 and provenance.get("provenance_ok") is True and overlap_ok,
                    )
                except lease_api.LeaseError as exc:
                    lease_release = {"released": False, "error": f"{type(exc).__name__}:{exc}"}
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
                operation="owned_native_llama_server",
                returncode=supervisor.proc.poll() if supervisor.proc else None,
            )
    overlap_ok = any(
        sample.get("request_interval_overlap") is True and bool(sample.get("owned_compute_apps"))
        for sample in overlap_samples
    )
    provenance_ok = bool(
        provenance.get("provenance_ok")
        and overlap_ok
        and cleanup.get("leak_free") is True
        and lease_release.get("released") is True
    )
    spans.extend(
        [
            {
                "phase": 7,
                "name": "native_generation_calls_aggregate",
                "duration_s": native_generation_s,
                "measurement": "sum_of_monotonic_request_intervals",
            },
            {
                "phase": 7,
                "name": "parsing_checkpoint_and_raw_persistence_aggregate",
                "duration_s": max(0.0, parse_persist_s),
                "measurement": "sum_of_monotonic_post_request_intervals",
            },
        ]
    )
    return {
        "model_loaded": model_loaded,
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
            "generation_overlap_samples": overlap_samples,
            "generation_overlap_ok": overlap_ok,
            "snapshots": snapshots,
            "cleanup": cleanup,
            "lease_release": lease_release,
            "provenance_ok": provenance_ok,
        },
        "runner_receipt": {
            "model_count": 1,
            "replica_count": 1 if identity else 0,
            "runner": "native_llama.cpp_server",
            "dual_gpu_runner_used": False,
            "selected_runner": deepcopy(context["runner"]),
            "command": command,
            "model_loaded": model_loaded,
            "load_duration_s": load_duration,
            "request_cap_s": REQUEST_CAP_S,
            "capture_cap_s": CAPTURE_CAP_S,
            "canary_token_budget": CANARY_TOKEN_BUDGET,
            "source_token_budget": SOURCE_TOKEN_BUDGET,
            "claim_token_budget": CLAIM_TOKEN_BUDGET,
            "direct_token_budget": DIRECT_TOKEN_BUDGET,
            "non_thinking_supported": bool(context["non_thinking_supported"]),
            "non_thinking_enabled": "--reasoning" in command,
            "decoding_parameters": deepcopy(DECODING_PARAMETERS),
            "parser_retry_count": 0,
            "regenerated_missing_row_count": 0,
            "lease_lifecycle": deepcopy(lease.document.get("phase_history", [])) if lease else [],
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
    """Build an honest terminal result from observed work and cleanup."""

    result = deepcopy(dict(artifact))
    rows = [deepcopy(dict(row)) for row in (capture or {}).get("rows", [])]
    canary = deepcopy((capture or {}).get("canary"))
    gpu_receipts = deepcopy((capture or {}).get("gpu_receipts", {}))
    runner_receipt = deepcopy(
        (capture or {}).get(
            "runner_receipt",
            {
                "model_count": 1,
                "replica_count": 0,
                "runner": "native_llama.cpp_server_preflight",
                "model_loaded": False,
                "selected_runner": deepcopy(context.get("runner", {})),
            },
        )
    )
    terminal = classify_terminal(
        checks,
        bool(runner_receipt.get("model_loaded")),
        canary,
        rows,
        bool(gpu_receipts.get("provenance_ok")),
        duration_s,
    )
    failed = first_failed_gate(checks)
    if terminal["status"] == "blocked":
        gate = failed or gate_row(
            "external_precondition", True, False, False, upstream="experiment_7196"
        )
        honest = f"blocked_{gate['check']}"
        substrate = "preflight_only_no_model_invocation"
        inference_mode = "not_run"
    elif terminal["status"] == "complete" and terminal["score"] == 1:
        gate = gate_row(
            "atomic_capture_complete",
            LOGICAL_RECEIPT_COUNT,
            len(rows),
            True,
            upstream="experiment_7196",
            field="completion_rows",
        )
        honest = (
            "complete_null_atomic_capture_parse_poor_bank_available_for_independent_audit"
            if terminal["verdict_class"] == "null"
            else "complete_positive_atomic_transport_capture_no_verifier_value_claim"
        )
        substrate = "live_llm_inference"
        inference_mode = "live_gpu"
    elif terminal["status"] == "complete":
        gate = gate_row(
            "authentic_complete_capture",
            {"receipts": 576, "provenance": True, "duration_floor_met": True},
            {
                "receipts": len(rows),
                "provenance": bool(gpu_receipts.get("provenance_ok")),
                "duration_floor_met": terminal["duration_floor_met"],
            },
            False,
            upstream="experiment_7196",
            field="capture_evidence",
        )
        honest = "complete_disqualified_atomic_capture_provenance_or_duration_failed"
        substrate = "live_llm_inference"
        inference_mode = "live_gpu" if gpu_receipts.get("runtime_provenance") else "not_verified"
    else:
        gate = gate_row(
            "atomic_capture_complete",
            LOGICAL_RECEIPT_COUNT,
            len(rows),
            False,
            upstream="experiment_7196",
            field="completion_rows",
        )
        honest = (
            "partial_incomplete_resumable_atomic_capture"
            if rows
            else (
                "partial_canary_only_bounded_generation"
                if canary is not None
                else "partial_model_load_without_generation"
            )
        )
        substrate = (
            "live_llm_inference"
            if canary is not None or rows
            else "native_llama_cpp_model_load_without_generation"
        )
        inference_mode = (
            "live_gpu" if bool(gpu_receipts.get("runtime_provenance")) else "not_verified"
        )
    model_spec = deepcopy(dict(context.get("model_spec", {})))
    manifest = _write_raw_manifest(
        raw_dir,
        context.get("schedule", []),
        rows,
        (capture or {}).get("checkpoints", []),
        (capture or {}).get(
            "checkpoint_identity",
            checkpoint_identity(
                context.get("schedule", []),
                str(context.get("public_view_sha256", "missing")),
                str(model_spec.get("sha256", "missing")),
            ),
        ),
        status=terminal["status"],
    )
    summary = summarize_capture(rows)
    comparison_rows = build_result_rows(rows)
    result.update(
        {
            "status": terminal["status"],
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "inference_substrate": substrate,
            "inference_substrate_class": terminal["inference_substrate_class"],
            "inference_mode": inference_mode,
            "duration_s": float(duration_s),
            "rows": comparison_rows,
            "sample_size_budget": {
                "planned_rows": 192,
                "completed_rows": len(comparison_rows),
                "independent_units": 192,
                "planned_logical_receipts": LOGICAL_RECEIPT_COUNT,
                "completed_logical_receipts": len(rows),
                "planned_cold_requests": EXPECTED_COLD_REQUEST_COUNT,
                "completed_cold_requests": summary["cold_request_count"],
                "planned_source_cache_hits": EXPECTED_SOURCE_CACHE_HITS,
                "completed_source_cache_hits": summary["source_cache_hit_count"],
                "exclusions": [],
            },
            "gate_check_summary": gate,
            "verdict_class": terminal["verdict_class"],
            "honest_verdict": honest,
            "atomic_capture_complete_score": terminal["score"],
            "model_specs": [model_spec] if model_spec else [],
            "completion_rows": rows,
            "gpu_receipts": gpu_receipts,
            "runner_receipt": runner_receipt,
            "raw_manifest": manifest,
            "canary_receipt": canary,
            "cost_summary": summary,
            "runtime_error": (capture or {}).get("runtime_error"),
            "duration_floor_s": terminal["duration_floor_s"],
            "duration_floor_met": terminal["duration_floor_met"],
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
    """Run one finite attempt and preserve blocked or measured evidence."""

    total_started = time.monotonic()
    spans: list[JsonDict] = []
    artifact = base_artifact(run_date, root=root)
    artifact["phase_spans"] = spans
    with _phase_span(spans, 0, "schema_complete_checkpoint_before_checks"):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        shell_path = checkpoint_dir / "running_shell.json"
        _progress(0, "checkpoint_write_start", path=str(shell_path))
        atomic_write_json(shell_path, artifact, allow_override=False, sort_keys=True)
        _progress(0, "checkpoint_write_end", path=str(shell_path))
        _write_raw_manifest(raw_dir, [], [], [], {}, status="running_unsealed")

    with _phase_span(spans, 1, "preconditions_and_frozen_schedule"):
        _progress(1, "benchmark_start", operation="preflight_contracts")
        checks, schedule, context = _collect_preflight(
            root, run_date, result_path, checkpoint_dir, raw_dir
        )
        _progress(
            1,
            "benchmark_end",
            operation="preflight_contracts",
            passed=bool(checks) and all(row.get("passed") is True for row in checks),
        )
    context["schedule"] = schedule
    capture: JsonDict | None = None
    if checks and all(row.get("passed") is True for row in checks):
        capture = _live_capture(context, checkpoint_dir, raw_dir, spans)

    terminal_build_started = time.monotonic()
    result = _finish(
        artifact,
        checks,
        context,
        capture,
        raw_dir,
        time.monotonic() - total_started,
    )
    result["phase_spans"] = deepcopy(spans)
    result["phase_spans"].append(
        {
            "phase": 9,
            "name": "terminal_artifact_build",
            "start_monotonic_s": terminal_build_started,
            "end_monotonic_s": time.monotonic(),
            "duration_s": time.monotonic() - terminal_build_started,
        }
    )
    result["reproducibility_checksum"] = artifact_checksum(result)
    _progress(9, "phase_start", name="terminal_validation_and_atomic_write")
    _progress(9, "validation_start", operation="cold_artifact_validation")
    errors = validate_artifact(result, check_source_hashes=True)
    _progress(9, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(9, "artifact_write_start", path=str(result_path), status=result["status"])
    atomic_write_json(result_path, result, allow_override=False, sort_keys=True)
    _progress(9, "artifact_write_end", path=str(result_path), status=result["status"])
    _progress(9, "phase_end", name="terminal_validation_and_atomic_write")
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
                "atomic_capture_complete_score": result["atomic_capture_complete_score"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
