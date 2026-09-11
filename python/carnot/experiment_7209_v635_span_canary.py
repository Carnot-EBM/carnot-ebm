"""Run the bounded Qwen3.8 source-span extraction canary.

The model receives only eight public canary inputs. Two grammar arms produce
separate source and claim relations. Exact public compilation and the shipped
executor run only after raw model output is durable.

Spec refs: REQ-VERIFY-7209 and SCENARIO-VERIFY-7209-*.
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
import platform
import re
import socket
import subprocess
import threading
import time
from typing import Any, Iterator
from urllib import error, request

import yaml

from carnot import experiment_7181_v633_qwen38_symbolic_traces as shipped_runtime
from carnot import experiment_7196_v634_qwen_atomic_capture as shipped_capture
from carnot import experiment_7208_v635_span_fixture as fixture
from carnot import gpu_lease_phase_journal as lease_api
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    supervisor_contract,
)
from carnot.inference.sota_models import cached_current_model
from carnot.paths import repo_root as find_repo_root
from carnot.verify.experiment_7195_source_relation_executor import (
    EntityBinding,
    TypedRelation,
    execute_relation,
)


JsonDict = dict[str, Any]
Tokenize = Callable[[bytes], Sequence[int]]

RUN_DATE = "20260911"
TASK_ID = "experiment_7209_v635_span_canary"
RANDOM_SEED = 7_209_001
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]

RESULT_PATH = Path("results/experiment_7209_v635_span_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7209")
RAW_DIR = Path("results/raw/experiment_7209")
UPSTREAM_PATH = Path("results/experiment_7208_v635_span_fixture.json")
PUBLIC_PATH = Path("results/fixtures/experiment_7208/public.jsonl")
AUTHORITY_PATH = Path("results/fixtures/experiment_7208/authority.jsonl")
FIXTURE_MANIFEST_PATH = Path("results/fixtures/experiment_7208/manifest.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7209_v635_span_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7209_v635_span_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7209_v635_span_canary.py")

PINNED_UPSTREAM_SHA256 = "sha256:f500faf3d5dc18cc4fd348b2000eed752a67625cee2b2b4050146e239004299a"
PINNED_PUBLIC_SHA256 = "sha256:e063654faf39d76e9dbfdf63df01119871def3c0c0ad853dfea98b6c3fbbbe0a"
PINNED_AUTHORITY_SHA256 = "sha256:569bd663b08ada100dd41c04d52ff530b55ef6649c331380cb624301ff2f0615"
PINNED_MANIFEST_SHA256 = "sha256:e276c46df1094985494d7e60991cae178ca2fb1f75490285d6c7b5f3e00a4c2b"

UPSTREAM_EXPECTED_FIELDS: JsonDict = {
    "status": "complete",
    "run_date": RUN_DATE,
    "verdict_class": "circular_positive",
    "honest_verdict": "complete_circular_positive_span_fixture_ready_no_verifier_value_claim",
    "span_fixture_ready_score": 1,
    "public_view_path": PUBLIC_PATH.as_posix(),
    "authority_sidecar_path": AUTHORITY_PATH.as_posix(),
}

SOURCE_TOKEN_BUDGET = 384
CLAIM_TOKEN_BUDGET = 128
CONTEXT_TOKEN_BUDGET = 8192
MODEL_LOAD_CAP_S = 240.0
REQUEST_CAP_S = 60.0
LIVE_WINDOW_CAP_S = 900.0
TOKEN_BUDGETS = {"source": SOURCE_TOKEN_BUDGET, "claim": CLAIM_TOKEN_BUDGET}
DECODING_PARAMETERS: JsonDict = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": RANDOM_SEED,
    "cache_prompt": False,
}
PROMPT_TEMPLATES = {
    "source": (
        "Extract all stated relations from the source. Use UTF-8 byte offsets copied "
        "from this source. Return only the required JSON object.\nSOURCE:\n{input_text}"
    ),
    "claim": (
        "Extract the one stated relation from the claim. Use UTF-8 byte offsets copied "
        "from this claim. Return only the required JSON object.\nCLAIM:\n{input_text}"
    ),
}

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "inference_mode": "Use live_gpu only when task-owned CUDA generation occurred.",
    "execution_venue": "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host.",
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": "Keep unit_id, arm, seed, metric, error and abstention for each comparison; do not replace numeric rows with a task roster.",
    "sample_size_budget": "Retain planned, attempted, completed, censored and independent-unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "Same correctness authority remains circular even with a separate implementation.",
    "verdict_class": "Use exactly positive | circular_positive | null | blocked | disqualified | partial; only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. Readiness is not scientific value.",
    "span_canary_ready_score": "Only complete, nontruncated and semantically usable canary extraction unlocks the live panel.",
    "canary_rows": "Every attempted call retains raw output, parsed relation and failure reason.",
    "frozen_decoding_contract": "The larger run uses exactly the canary-qualified settings.",
    "token_budget_receipt": "Measured serialized sizes distinguish representation fit from an invented duration claim.",
    "MODEL_SPECS": "Include the mandated Qwen3.8 GGUF for every LLM call.",
    "model_invoked": "Record whether generation actually occurred.",
    "phase_spans": "Separate loading, prefill, generation, parsing and verification costs.",
    "gpu_receipts": "Task-owned CUDA evidence must overlap actual model work.",
    "model_identity_receipt": "Record repository ID, revision, GGUF hash and actual runtime.",
    "runner_receipt": "Model count and execution runner explain compute allocation.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def canonical_json(value: Any) -> str:
    """Use one compact JSON spelling for stable byte receipts."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes without normalizing their contents."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash the canonical JSON bytes for one structured receipt."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks without changing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind terminal evidence while excluding duration and the hash itself."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def unwrap_principle(value: Any) -> Any:
    """Unwrap only the exact principle/value contract shape."""

    if isinstance(value, Mapping) and set(value) == {"principle", "value"}:
        return value["value"]
    return value


def is_quarantined(artifact: Mapping[str, Any]) -> bool:
    """Reject explicit structured quarantine without trusting mapping truthiness."""

    for field in ("flagged_adversarial", "quarantined", "fabricated"):
        observed = unwrap_principle(artifact.get(field))
        if observed is True or (
            isinstance(observed, str) and observed.lower() in {"true", "quarantined"}
        ):
            return True
    return False


def _manifest_hits(value: Any, wanted: set[str]) -> bool:
    """Find an excluded upstream identifier in nested manifest data."""

    if isinstance(value, Mapping):
        return any(_manifest_hits(item, wanted) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_hits(item, wanted) for item in value)
    return isinstance(value, str) and value in wanted


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str | None,
    field: str,
) -> JsonDict:
    """Keep both sides of one gate so a block remains diagnosable."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def gate_summary(failure: Mapping[str, Any] | None) -> JsonDict:
    """Project the first failed gate into the stable terminal shape."""

    if failure is None:
        return {
            "passed": True,
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": "all_required_checks_pass",
            "observed_value": "all_required_checks_pass",
        }
    return {
        "passed": False,
        "failed_check": failure["check"],
        "upstream": failure["upstream"],
        "field": failure["field"],
        "expected_value": failure["expected_value"],
        "observed_value": failure["observed_value"],
    }


def load_yaml(path: Path) -> Any:
    """Read one YAML prerequisite after its exact bytes are available for hashing."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read object-only JSONL rows without repairing malformed data."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("jsonl_row_not_object")
        rows.append(value)
    return rows


def load_canary_split(
    public_path: Path, authority_path: Path
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Select eight supported canary rows and never expose other labels downstream."""

    authority = [
        row
        for row in _read_jsonl(authority_path)
        if row.get("split") == "canary" and row.get("variant") == "supported"
    ]
    wanted = [str(row["unit_id"]) for row in authority]
    public_source = _read_jsonl(public_path)
    if any("unit_id" not in row for row in public_source):
        raise ValueError("canary_supported_denominator")
    public_by_id = {str(row["unit_id"]): row for row in public_source}
    public = [deepcopy(public_by_id[unit_id]) for unit_id in wanted if unit_id in public_by_id]
    if len(authority) != 8 or len(public) != 8 or len(set(wanted)) != 8:
        raise ValueError("canary_supported_denominator")
    return public, authority


def _call_prompt(call_type: str, input_text: str) -> str:
    """Render one frozen prompt from only the public text for that call."""

    return PROMPT_TEMPLATES[call_type].format(input_text=input_text)


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 32 cold calls without placing evaluator fields in model work."""

    if len(public_rows) != 8 or len(authority_rows) != 8:
        raise ValueError("canary_supported_denominator")
    authority_ids = [str(row["unit_id"]) for row in authority_rows]
    if any(
        row.get("split") != "canary" or row.get("variant") != "supported" for row in authority_rows
    ):
        raise ValueError("non_canary_authority")
    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    if set(public_by_id) != set(authority_ids):
        raise ValueError("public_authority_identity")

    schedule: list[JsonDict] = []
    for unit_id in authority_ids:
        public = public_by_id[unit_id]
        for arm in ("grammar_only", "reference"):
            for call_type, field in (("source", "source_text"), ("claim", "claim_text")):
                input_text = str(public[field])
                input_bytes = input_text.encode("utf-8")
                grammar = fixture.compile_grammar(input_bytes, call_type, arm)
                prompt = _call_prompt(call_type, input_text)
                order = len(schedule)
                call_id = sha256_json(
                    {
                        "unit_id": unit_id,
                        "arm": arm,
                        "call_type": call_type,
                        "seed": RANDOM_SEED,
                    }
                )
                schedule.append(
                    {
                        "call_order": order,
                        "call_id": call_id,
                        "unit_id": unit_id,
                        "arm": arm,
                        "call_type": call_type,
                        "seed": RANDOM_SEED,
                        "model_input": {field: input_text},
                        "input_text": input_text,
                        "input_sha256": sha256_bytes(input_bytes),
                        "prompt": prompt,
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                        "grammar": grammar["grammar"],
                        "grammar_sha256": grammar["grammar_sha256"],
                        "reference_grammar_sha256": fixture.compile_grammar(
                            input_bytes, call_type, "reference"
                        )["grammar_sha256"],
                        "output_token_budget": TOKEN_BUDGETS[call_type],
                        "context_token_budget": CONTEXT_TOKEN_BUDGET,
                        "decoding_parameters": deepcopy(DECODING_PARAMETERS),
                        "cold_request": True,
                    }
                )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Rebuild the frozen schedule and name every changed request property."""

    try:
        expected = build_schedule(public_rows, authority_rows)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != len(expected):
        errors.append("schedule_count")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field in wanted:
            if observed.get(field) != wanted[field]:
                errors.append(f"call_{index}:{field}")
        extra = set(observed) - set(wanted)
        if extra:
            errors.append(f"call_{index}:extra_fields")
    return errors


def measure_token_budgets(schedule: Sequence[Mapping[str, Any]], tokenizer: Tokenize) -> JsonDict:
    """Measure unique expected forms with the embedded tokenizer before generation."""

    forms: dict[str, dict[str, bytes]] = {"source": {}, "claim": {}}
    for row in schedule:
        call_type = str(row["call_type"])
        unit_id = str(row["unit_id"])
        if unit_id in forms[call_type]:
            continue
        value = fixture.extract_public_completion(str(row["input_text"]).encode("utf-8"), call_type)
        forms[call_type][unit_id] = fixture.canonical_json(value).encode("utf-8")
    receipt: JsonDict = {
        "measurement_status": "measured_embedded_gguf_tokenizer",
        "headroom_fraction": 0.2,
        "budget_increased_after_failure": False,
    }
    all_fit = True
    for call_type in ("source", "claim"):
        values = list(forms[call_type].values())
        token_counts = [len(tokenizer(value)) for value in values]
        byte_counts = [len(value) for value in values]
        budget = TOKEN_BUDGETS[call_type]
        fits = bool(token_counts) and max(token_counts) * 1.2 <= budget
        receipt[call_type] = {
            "measured_form_count": len(values),
            "minimum_serialized_tokens": min(token_counts),
            "maximum_serialized_tokens": max(token_counts),
            "minimum_serialized_bytes": min(byte_counts),
            "maximum_serialized_bytes": max(byte_counts),
            "budget": budget,
            "required_budget_with_20_percent_headroom": max(token_counts) * 1.2,
            "fits_with_20_percent_headroom": fits,
            "serialized_form_hashes": [sha256_bytes(value) for value in values],
        }
        all_fit = all_fit and fits
    receipt["all_forms_fit_with_20_percent_headroom"] = all_fit
    return receipt


def upstream_gate_rows(
    upstream: Mapping[str, Any],
    upstream_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    manifest_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate producer fields, exact bytes, quarantine, and historical nulls."""

    hashes = {
        "artifact": sha256_bytes(upstream_bytes),
        "public": sha256_bytes(public_bytes),
        "authority": sha256_bytes(authority_bytes),
        "manifest": sha256_bytes(manifest_bytes),
    }
    expected_hashes = {
        "artifact": PINNED_UPSTREAM_SHA256,
        "public": PINNED_PUBLIC_SHA256,
        "authority": PINNED_AUTHORITY_SHA256,
        "manifest": PINNED_MANIFEST_SHA256,
    }
    rows = [
        gate_row(
            "exact_upstream_bytes",
            expected_hashes,
            hashes,
            hashes == expected_hashes,
            upstream="experiment_7208",
            field="source_bytes",
        )
    ]
    quarantined = is_quarantined(upstream)
    rows.append(
        gate_row(
            "structured_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="experiment_7208",
            field="flagged_adversarial|quarantined|fabricated",
        )
    )
    excluded = _manifest_hits(
        exclusion_manifest,
        {
            "Exp7208",
            "experiment_7208_v635_span_fixture",
            "results/experiment_7208_v635_span_fixture.json",
        },
    )
    rows.append(
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="experiment_7208",
            field="experiment_ids",
        )
    )
    observed_fields = {
        field: unwrap_principle(upstream.get(field)) for field in UPSTREAM_EXPECTED_FIELDS
    }
    rows.append(
        gate_row(
            "producer_gate_fields",
            UPSTREAM_EXPECTED_FIELDS,
            observed_fields,
            observed_fields == UPSTREAM_EXPECTED_FIELDS,
            upstream="experiment_7208",
            field="terminal_fields",
        )
    )
    checksum_ok = upstream.get("reproducibility_checksum") == fixture.artifact_checksum(upstream)
    rows.append(
        gate_row(
            "upstream_authentication",
            True,
            checksum_ok,
            checksum_ok,
            upstream="experiment_7208",
            field="reproducibility_checksum",
        )
    )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError:
        manifest = {}
    source_hashes = upstream.get("source_artifact_hashes", {})
    sidecar_observed = {
        "artifact_public": source_hashes.get("public_view"),
        "artifact_authority": source_hashes.get("authority_sidecar"),
        "artifact_manifest": source_hashes.get("fixture_manifest"),
        "manifest_public": manifest.get("public_view_sha256"),
        "manifest_authority": manifest.get("authority_sidecar_sha256"),
        "manifest_schema": manifest.get("schema"),
    }
    sidecar_expected = {
        "artifact_public": PINNED_PUBLIC_SHA256,
        "artifact_authority": PINNED_AUTHORITY_SHA256,
        "artifact_manifest": PINNED_MANIFEST_SHA256,
        "manifest_public": PINNED_PUBLIC_SHA256,
        "manifest_authority": PINNED_AUTHORITY_SHA256,
        "manifest_schema": "carnot.exp7208.span_fixture.v1",
    }
    rows.append(
        gate_row(
            "sidecar_authentication",
            sidecar_expected,
            sidecar_observed,
            sidecar_observed == sidecar_expected,
            upstream="experiment_7208",
            field="sidecar_hashes",
        )
    )
    history = upstream.get("upstream_history", {})
    preserved = {
        "exp7197_grounding_value_score": history.get("exp7197_grounding_value_score"),
        "known_failed_value_promoted_to_readiness": history.get(
            "known_failed_value_promoted_to_readiness"
        ),
    }
    expected_preserved = {
        "exp7197_grounding_value_score": 0,
        "known_failed_value_promoted_to_readiness": False,
    }
    rows.append(
        gate_row(
            "known_failed_value_preserved",
            expected_preserved,
            preserved,
            preserved == expected_preserved,
            upstream="experiment_7208",
            field="upstream_history",
        )
    )
    return rows


def _relation_view(relation: Mapping[str, Any]) -> JsonDict:
    """Remove compiler-added surface names before authority comparison."""

    return {field: relation.get(field) for field in fixture.RELATION_FIELDS}


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Conserve one request, raw response, parse, and public-reference result."""

    raw = str(response.get("raw_completion") or "")
    raw_response = deepcopy(dict(response.get("raw_response") or {}))
    choices = list(raw_response.get("choices") or [])
    message = dict(choices[0].get("message") or {}) if choices else {}
    reasoning = str(message.get("reasoning_content") or "")
    reasoning_marker = bool(re.search(r"</?think(?:\s|>)", raw, flags=re.IGNORECASE))
    reasoning_present = bool(reasoning or reasoning_marker)
    finish_reason = response.get("finish_reason")
    truncated = finish_reason in {"length", "max_tokens"}
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("completion_not_object")
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}:{exc}"
    compiled: JsonDict = {"outcome": "unknown", "relations": [], "errors": ["not_parsed"]}
    if parse_error is None:
        compiled = fixture.compile_completion(
            str(sealed["input_text"]).encode("utf-8"),
            parsed,
            str(sealed["call_type"]),
            str(sealed["reference_grammar_sha256"]),
        )
    compile_errors = list(compiled.get("errors") or [])
    exact_reference_valid = bool(
        parse_error is None and not compile_errors and compiled.get("outcome") == "known"
    )
    failures: list[str] = []
    if response.get("error"):
        failures.append(f"transport_error:{response['error']}")
    if truncated:
        failures.append("truncated")
    if reasoning_present:
        failures.append("reasoning_present")
    if parse_error is not None:
        failures.append("json_parse_error")
    elif compile_errors:
        failures.extend(f"compile:{item}" for item in compile_errors)
    elif compiled.get("outcome") != "known":
        failures.append("explicit_unknown")
    terminal = "complete" if not failures else "failed"
    response_bytes = base64.b64decode(str(response.get("raw_response_bytes_b64") or ""))
    return {
        "call_order": sealed["call_order"],
        "call_id": sealed["call_id"],
        "unit_id": sealed["unit_id"],
        "arm": sealed["arm"],
        "call_type": sealed["call_type"],
        "seed": sealed["seed"],
        "input_text": sealed["input_text"],
        "input_sha256": sealed["input_sha256"],
        "prompt": sealed["prompt"],
        "prompt_sha256": sealed["prompt_sha256"],
        "request_payload": deepcopy(dict(response.get("raw_request") or {})),
        "request_payload_sha256": sha256_bytes(
            base64.b64decode(str(response.get("raw_request_bytes_b64") or ""))
        ),
        "grammar": sealed["grammar"],
        "grammar_sha256": sealed["grammar_sha256"],
        "raw_response": raw_response,
        "raw_response_sha256": sha256_bytes(response_bytes),
        "raw_completion": raw,
        "raw_completion_sha256": sha256_bytes(raw.encode("utf-8")),
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "finish_reason": finish_reason,
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "transport_error": response.get("error"),
        "terminal_state": terminal,
        "parse_valid": parse_error is None,
        "parse_error": parse_error,
        "parsed_completion": deepcopy(parsed),
        "compiled_completion": compiled,
        "compile_errors": compile_errors,
        "exact_reference_valid": exact_reference_valid,
        "truncated": truncated,
        "reasoning_content": reasoning,
        "reasoning_marker_present": reasoning_marker,
        "reasoning_disabled_observed": not reasoning_present,
        "failure_reasons": failures,
        "cuda_offload": deepcopy(dict(resource)),
    }


def _sentence_ranges(text: bytes) -> list[tuple[int, int]]:
    """Return half-open sentence byte ranges for executor evidence spans."""

    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(rb"[.!?]", text):
        end = match.end()
        while start < end and text[start : start + 1].isspace():
            start += 1
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def _execute_compiled_pair(
    source_text: str, source: Mapping[str, Any], claim: Mapping[str, Any]
) -> JsonDict:
    """Convert compiled public references into the shipped typed executor input."""

    if source.get("outcome") != "known" or claim.get("outcome") != "known":
        return {"decision": "unknown", "abstention": True, "errors": ["unusable_completion"]}
    source_bytes = source_text.encode("utf-8")
    source_relations = list(source.get("relations") or [])
    claim_relations = list(claim.get("relations") or [])
    if not source_relations or len(claim_relations) != 1:
        return {"decision": "unknown", "abstention": True, "errors": ["relation_count"]}
    bindings_by_surface: dict[str, EntityBinding] = {}
    for relation in source_relations:
        for prefix in ("subject", "object"):
            surface = str(relation[f"{prefix}_surface"])
            if surface not in bindings_by_surface:
                bindings_by_surface[surface] = EntityBinding(
                    surface,
                    surface,
                    int(relation[f"{prefix}_start"]),
                    int(relation[f"{prefix}_end"]),
                )
    ranges = _sentence_ranges(source_bytes)
    typed_source = []
    for relation in source_relations:
        sentence = int(relation["sentence_index"])
        if not 0 <= sentence < len(ranges):
            return {"decision": "unknown", "abstention": True, "errors": ["sentence_index"]}
        left, right = ranges[sentence]
        typed_source.append(
            TypedRelation(
                str(relation["subject_surface"]),
                str(relation["predicate"]),
                str(relation["object_surface"]),
                str(relation["polarity"]),
                left,
                right,
            )
        )
    claim_relation = claim_relations[0]
    typed_claim = TypedRelation(
        str(claim_relation["subject_surface"]),
        str(claim_relation["predicate"]),
        str(claim_relation["object_surface"]),
        str(claim_relation["polarity"]),
        0,
        max(1, len(source_bytes)),
    )
    result = execute_relation(
        source_bytes,
        tuple(bindings_by_surface.values()),
        tuple(typed_source),
        typed_claim,
    )
    return {
        "decision": result.decision,
        "abstention": result.abstention,
        "errors": list(result.uncertainty_reasons),
    }


def score_completion_pairs(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Compare references separately, then run each complete pair through the executor."""

    calls = {
        (str(row["unit_id"]), str(row["arm"]), str(row["call_type"])): row
        for row in completion_rows
    }
    schedule_by_key = {
        (str(row["unit_id"]), str(row["arm"]), str(row["call_type"])): row for row in schedule
    }
    authority = {str(row["unit_id"]): row for row in authority_rows}
    output: list[JsonDict] = []
    for private in authority_rows:
        unit_id = str(private["unit_id"])
        for arm in ("grammar_only", "reference"):
            source = calls.get((unit_id, arm, "source"))
            claim = calls.get((unit_id, arm, "claim"))
            source_call = schedule_by_key.get((unit_id, arm, "source"))
            claim_call = schedule_by_key.get((unit_id, arm, "claim"))
            source_compiled = dict(source.get("compiled_completion") or {}) if source else {}
            claim_compiled = dict(claim.get("compiled_completion") or {}) if claim else {}
            expected_source = (
                fixture.extract_public_completion(
                    str(source_call["input_text"]).encode("utf-8"), "source"
                )
                if source_call
                else {"relations": []}
            )
            expected_claim = dict(authority[unit_id].get("claim_relation") or {})
            observed_source = [_relation_view(row) for row in source_compiled.get("relations", [])]
            observed_claim = [_relation_view(row) for row in claim_compiled.get("relations", [])]
            source_agreement = observed_source == expected_source.get("relations", [])
            claim_agreement = observed_claim == [expected_claim]
            executor_invoked = bool(source and claim and source_call and claim_call)
            execution = (
                _execute_compiled_pair(
                    str(source_call["input_text"]), source_compiled, claim_compiled
                )
                if executor_invoked
                else {"decision": "unknown", "abstention": True, "errors": ["missing_call"]}
            )
            expected_decision = str(private["expected_decision"])
            passed = bool(
                source_agreement and claim_agreement and execution["decision"] == expected_decision
            )
            output.append(
                {
                    "unit_id": unit_id,
                    "arm": arm,
                    "seed": RANDOM_SEED,
                    "metric": int(passed),
                    "error": None if passed else "canary_relation_or_decision_disagreement",
                    "abstention": bool(execution["abstention"]),
                    "prediction": execution["decision"],
                    "expected_prediction": expected_decision,
                    "source_relation_agreement": source_agreement,
                    "claim_relation_agreement": claim_agreement,
                    "executor_invoked": executor_invoked,
                    "executor_errors": execution["errors"],
                }
            )
    return output


def readiness_receipt(
    completion_rows: Sequence[Mapping[str, Any]], comparison_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Apply fixed 16-call and eight-interpretation readiness denominators."""

    reference_calls = [row for row in completion_rows if row.get("arm") == "reference"]
    usable = [
        row
        for row in reference_calls
        if row.get("terminal_state") == "complete"
        and row.get("parse_valid") is True
        and row.get("exact_reference_valid") is True
        and row.get("truncated") is False
        and row.get("reasoning_disabled_observed") is True
    ]
    reference_comparisons = [row for row in comparison_rows if row.get("arm") == "reference"]
    correct = sum(row.get("metric") == 1 for row in reference_comparisons)
    ready = (
        len(reference_calls) == 16
        and len(usable) == 16
        and len(reference_comparisons) == 8
        and correct >= 7
    )
    return {
        "reference_call_denominator": 16,
        "attempted_reference_calls": len(reference_calls),
        "complete_parse_valid_exact_reference_calls": len(usable),
        "truncated_reference_calls": sum(row.get("truncated") is True for row in reference_calls),
        "reasoning_present_reference_calls": sum(
            row.get("reasoning_disabled_observed") is False for row in reference_calls
        ),
        "combined_interpretation_denominator": 8,
        "attempted_combined_interpretations": len(reference_comparisons),
        "correct_combined_interpretations": correct,
        "minimum_correct_combined_interpretations": 7,
        "span_canary_ready_score": int(ready),
    }


def _frozen_decoding_contract() -> JsonDict:
    """Return the one canary contract eligible for a later held-out run."""

    return {
        "context_tokens": CONTEXT_TOKEN_BUDGET,
        "source_tokens": SOURCE_TOKEN_BUDGET,
        "claim_tokens": CLAIM_TOKEN_BUDGET,
        "decoding_parameters": deepcopy(DECODING_PARAMETERS),
        "prompt_templates": deepcopy(PROMPT_TEMPLATES),
        "prompt_template_hashes": {
            key: sha256_bytes(value.encode("utf-8")) for key, value in PROMPT_TEMPLATES.items()
        },
        "arms": ["grammar_only", "reference"],
        "separate_source_and_claim_calls": True,
        "call_count": 32,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "request_cap_s": REQUEST_CAP_S,
        "live_window_cap_s": LIVE_WINDOW_CAP_S,
        "non_thinking_required": True,
        "rerun_variants": 0,
        "held_out_run_authorized": False,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before any external prerequisite can fail."""

    return {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_calls": 32,
            "attempted_calls": 0,
            "completed_calls": 0,
            "censored_calls": 32,
            "planned_comparisons": 16,
            "completed_comparisons": 0,
            "independent_units": 8,
            "representation_arms": 2,
            "calls_per_unit_arm": 2,
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_exp7209_running_preconditions",
        "span_canary_ready_score": 0,
        "canary_rows": [],
        "frozen_decoding_contract": _frozen_decoding_contract(),
        "token_budget_receipt": {"measurement_status": "not_measured"},
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "phase_spans": [],
        "gpu_receipts": {},
        "model_identity_receipt": {},
        "runner_receipt": {
            "model_count": 1,
            "runner": "native_llama.cpp_server",
            "dual_gpu_runner_used": False,
        },
        "readiness_receipt": {},
        "scope_statement": "bounded canary generation only; no held-out verifier value claim",
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish an external block without turning absence into a measured null."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(failure)
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = f"blocked_exp7209_{failure['check'] if failure else 'unknown'}"
    artifact["span_canary_ready_score"] = 0
    artifact["duration_s"] = duration_s
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finalize_measured_artifact(
    artifact: JsonDict,
    completion_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    token_receipt: Mapping[str, Any],
    *,
    duration_s: float,
    live_evidence: bool,
) -> JsonDict:
    """Finish a ready or null canary from measured rows without padding time."""

    artifact["canary_rows"] = [deepcopy(dict(row)) for row in completion_rows]
    artifact["rows"] = [deepcopy(dict(row)) for row in comparison_rows]
    artifact["token_budget_receipt"] = deepcopy(dict(token_receipt))
    receipt = readiness_receipt(completion_rows, comparison_rows)
    artifact["readiness_receipt"] = receipt
    ready = int(receipt["span_canary_ready_score"] == 1 and live_evidence)
    artifact["span_canary_ready_score"] = ready
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": len(completion_rows),
            "completed_calls": sum(
                row.get("terminal_state") == "complete" for row in completion_rows
            ),
            "censored_calls": 32 - len(completion_rows),
            "completed_comparisons": len(comparison_rows),
        }
    )
    artifact["status"] = "complete"
    artifact["model_invoked"] = bool(live_evidence)
    if live_evidence:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
        artifact["gpu_receipts"].setdefault("provenance_ok", True)
    else:
        artifact["inference_substrate"] = "embedded_gguf_tokenizer_measurement_no_generation"
        artifact["inference_substrate_class"] = "model_load_no_generation"
        artifact["inference_mode"] = "not_invoked"
    artifact["verdict_class"] = "circular_positive" if ready else "null"
    artifact["honest_verdict"] = (
        "complete_circular_positive_span_canary_ready_no_held_out_value_claim"
        if ready
        else "complete_null_span_canary_not_ready_no_held_out_value_claim"
    )
    artifact["duration_s"] = duration_s
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal shape, fixed denominators, and reproducibility hash."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("MODEL_SPECS")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_identity")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping):
        errors.append("sample_size_budget")
    else:
        expected_counts = {
            "planned_calls": 32,
            "attempted_calls": len(value.get("canary_rows", [])),
            "completed_calls": sum(
                row.get("terminal_state") == "complete" for row in value.get("canary_rows", [])
            ),
            "censored_calls": 32 - len(value.get("canary_rows", [])),
            "planned_comparisons": 16,
            "completed_comparisons": len(value.get("rows", [])),
            "independent_units": 8,
        }
        if any(budget.get(field) != expected for field, expected in expected_counts.items()):
            errors.append("sample_size_budget")
    status = value.get("status")
    if status == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or value.get("span_canary_ready_score") != 0
            or not isinstance(summary, Mapping)
            or summary.get("passed") is not False
        ):
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    if status != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    token_receipt = value.get("token_budget_receipt")
    if not isinstance(token_receipt, Mapping) or token_receipt.get("measurement_status") != (
        "measured_embedded_gguf_tokenizer"
    ):
        errors.append("token_budget_receipt")
    ready_receipt = readiness_receipt(value.get("canary_rows", []), value.get("rows", []))
    live = value.get("model_invoked") is True
    expected_ready = int(ready_receipt["span_canary_ready_score"] == 1 and live)
    if value.get("readiness_receipt") != ready_receipt:
        errors.append("readiness_receipt")
    if value.get("span_canary_ready_score") != expected_ready:
        errors.append("span_canary_ready_score")
    if expected_ready:
        if value.get("verdict_class") != "circular_positive":
            errors.append("verdict_class")
    elif value.get("verdict_class") != "null":
        errors.append("verdict_class")
    if live and (
        value.get("inference_substrate") != "live_llm_inference"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
        or not isinstance(value.get("gpu_receipts"), Mapping)
        or value["gpu_receipts"].get("provenance_ok") is not True
    ):
        errors.append("live_inference_provenance")
    return list(dict.fromkeys(errors))


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Freeze prompts, grammars, settings, model identity, and raw row hashes."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema": "carnot.exp7209.raw_manifest.v1",
        "status": "complete" if len(completion_rows) == 32 else "partial",
        "raw_row_count": len(completion_rows),
        "worker_input_paths": [PUBLIC_PATH.as_posix()],
        "authority_path_opened_by_model_worker": False,
        "development_or_test_label_read_count": 0,
        "schedule": [deepcopy(dict(row)) for row in schedule],
        "schedule_sha256": sha256_json(list(schedule)),
        "prompt_templates": deepcopy(PROMPT_TEMPLATES),
        "prompt_templates_sha256": sha256_json(PROMPT_TEMPLATES),
        "decoding_contract": _frozen_decoding_contract(),
        "grammar_generator_path": str(MODULE_PATH),
        "grammar_generator_sha256": sha256_file(Path(__file__)),
        "model_identity": deepcopy(dict(model_identity)),
        "raw_rows": [
            {
                "call_order": row.get("call_order"),
                "call_id": row.get("call_id"),
                "unit_id": row.get("unit_id"),
                "arm": row.get("arm"),
                "call_type": row.get("call_type"),
                "raw_completion_sha256": row.get("raw_completion_sha256"),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "grammar_sha256": row.get("grammar_sha256"),
                "terminal_state": row.get("terminal_state"),
                "row_sha256": sha256_json(row),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(
        raw_dir / "schedule.json",
        {"schedule": list(schedule)},
        allow_override=False,
        sort_keys=True,
    )
    atomic_write_json(
        raw_dir / "frozen_contract.json",
        {"contract": _frozen_decoding_contract(), "model_identity": dict(model_identity)},
        allow_override=False,
        sort_keys=True,
    )
    atomic_write_json(raw_dir / "raw_manifest.json", frozen, allow_override=False, sort_keys=True)
    return frozen


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Flush observed state at every boundary and long-operation heartbeat."""

    print(
        canonical_json({"experiment": 7209, "phase": phase, "event": event, **fields}), flush=True
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
    """Report observed time and completed units outside blocking native calls."""

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

    thread = threading.Thread(target=emit, name=f"exp7209-heartbeat-{phase}", daemon=True)
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
    """Stream new lines from only the task-owned native child log."""

    stop = threading.Event()

    def stream() -> None:
        offset = 0
        while not stop.wait(0.25):
            if not supervisor.log_path.is_file():
                continue
            with supervisor.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                for line in handle:
                    print(f"exp7209 native-server {line.rstrip()}", flush=True)
                offset = handle.tell()

    thread = threading.Thread(target=stream, name="exp7209-native-log", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _content_addressed_hash(path: Path) -> str | None:  # pragma: no cover
    """Use the immutable cache target digest without scanning model tensors."""

    try:
        name = path.resolve(strict=True).name.lower()
    except OSError:
        return None
    return f"sha256:{name}" if re.fullmatch(r"[0-9a-f]{64}", name) else None


def _runner_supports(runner: Mapping[str, Any], flag: str) -> bool:  # pragma: no cover
    """Check one option against the retained native help receipt."""

    receipts = list(runner.get("command_receipts") or [])
    text = ""
    if len(receipts) >= 2:
        text = f"{receipts[1].get('stdout', '')}\n{receipts[1].get('stderr', '')}"
    return flag in text


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Bind every repository source and sealed upstream used by this canary."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "research_references": Path("research-references.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "constraint_spec": SPEC_PATH,
        "upstream_artifact": UPSTREAM_PATH,
        "public_view": PUBLIC_PATH,
        "authority_sidecar": AUTHORITY_PATH,
        "fixture_manifest": FIXTURE_MANIFEST_PATH,
        "exp7208_module": Path("python/carnot/experiment_7208_v635_span_fixture.py"),
        "exp7195_executor": Path(
            "python/carnot/verify/experiment_7195_source_relation_executor.py"
        ),
        "sota_models": Path("python/carnot/inference/sota_models.py"),
        "llama_server_supervisor": Path("python/carnot/inference/llama_server_supervisor.py"),
        "gpu_lease_journal": Path("python/carnot/gpu_lease_phase_journal.py"),
        "module": MODULE_PATH,
        "entrypoint": WRAPPER_PATH,
        "focused_tests": TEST_PATH,
    }
    return {
        name: sha256_file(root / path) if (root / path).is_file() else "missing"
        for name, path in paths.items()
    }


def _collect_preflight(
    root: Path, run_date: str, result_path: Path, checkpoint_dir: Path, raw_dir: Path
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:  # pragma: no cover
    """Check exact sources, native tools, cache, and idle GPUs without mutation."""

    checks: list[JsonDict] = []
    context: JsonDict = {}

    def record(row: JsonDict) -> None:
        _progress(1, "check_start", check=row["check"])
        checks.append(row)
        _progress(1, "check_end", check=row["check"], passed=row["passed"])

    record(
        gate_row(
            "run_date", RUN_DATE, run_date, run_date == RUN_DATE, upstream=None, field="run_date"
        )
    )
    required = {
        "upstream": root / UPSTREAM_PATH,
        "public": root / PUBLIC_PATH,
        "authority": root / AUTHORITY_PATH,
        "manifest": root / FIXTURE_MANIFEST_PATH,
        "exclusion": root / EXCLUSION_PATH,
        "spec": root / SPEC_PATH,
        "module": root / MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "tests": root / TEST_PATH,
        "adversarial_tool": root / "scripts/adversarial_verify.py",
        "row_lint_tool": root / "scripts/verdict_row_consistency_lint.py",
        "spec_coverage_tool": root / "scripts/check_spec_coverage.py",
    }
    observed_sources = {
        name: path.is_file() and os.access(path, os.R_OK) for name, path in required.items()
    }
    observed_sources["spec_has_req"] = bool(
        observed_sources["spec"]
        and "REQ-VERIFY-7209" in required["spec"].read_text(encoding="utf-8")
    )
    record(
        gate_row(
            "required_source_bytes",
            {name: True for name in observed_sources},
            observed_sources,
            all(observed_sources.values()),
            upstream="repository",
            field="required_paths",
        )
    )
    storage = {
        "result_parent": result_path.parent.is_dir() and os.access(result_path.parent, os.W_OK),
        "checkpoint_dir": checkpoint_dir.is_dir() and os.access(checkpoint_dir, os.W_OK),
        "raw_dir": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
    }
    record(
        gate_row(
            "writable_output_storage",
            {name: True for name in storage},
            storage,
            all(storage.values()),
            upstream="host_filesystem",
            field="output_paths",
        )
    )
    if not all(observed_sources.values()):
        return checks, [], [], context
    try:
        upstream_bytes = required["upstream"].read_bytes()
        public_bytes = required["public"].read_bytes()
        authority_bytes = required["authority"].read_bytes()
        manifest_bytes = required["manifest"].read_bytes()
        upstream = json.loads(upstream_bytes)
        exclusion = load_yaml(required["exclusion"])
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        record(
            gate_row(
                "upstream_parse",
                "valid_json_jsonl_and_yaml",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="experiment_7208",
                field="source_documents",
            )
        )
        return checks, [], [], context
    for row in upstream_gate_rows(
        upstream, upstream_bytes, public_bytes, authority_bytes, manifest_bytes, exclusion
    ):
        record(row)
    if any(row["passed"] is not True for row in checks):
        return checks, [], [], context
    try:
        public_rows, authority_rows = load_canary_split(required["public"], required["authority"])
        schedule = build_schedule(public_rows, authority_rows)
        schedule_problem = schedule_errors(schedule, public_rows, authority_rows)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        public_rows, authority_rows, schedule = [], [], []
        schedule_problem = [f"{type(exc).__name__}:{exc}"]
    record(
        gate_row(
            "blind_canary_schedule",
            {"calls": 32, "units": 8, "errors": []},
            {
                "calls": len(schedule),
                "units": len({row.get("unit_id") for row in schedule}),
                "errors": schedule_problem,
            },
            not schedule_problem,
            upstream="experiment_7208_canary_split",
            field="schedule",
        )
    )
    grammar_problem = [
        problem for row in schedule for problem in fixture.grammar_errors(str(row["grammar"]))
    ]
    record(
        gate_row(
            "grammar_payloads",
            {"count": 32, "errors": []},
            {"count": len(schedule), "errors": grammar_problem},
            len(schedule) == 32 and not grammar_problem,
            upstream="experiment_7208_compiler",
            field="gbnf",
        )
    )
    live_value = os.environ.get("CARNOT_FORCE_LIVE")
    record(
        gate_row(
            "force_live_environment",
            "1",
            live_value,
            live_value == "1",
            upstream="process_environment",
            field="CARNOT_FORCE_LIVE",
        )
    )
    _progress(2, "model_resolution_start", model=QWEN_MODEL_ID)
    resolved = cached_current_model(gpu_index=0, preferred_quant=QUANTIZATION)
    model_path = Path(str(resolved.get("model_path"))) if resolved else None
    model_exists = bool(model_path and model_path.is_file())
    model_hash = _content_addressed_hash(model_path) if model_path and model_exists else None
    metadata = read_gguf_metadata(model_path) if model_path and model_exists else {}
    model_identity = {
        "hf_id": resolved.get("hf_id") if resolved else None,
        "quantization": QUANTIZATION,
        "gguf_path": str(model_path.resolve()) if model_path and model_exists else None,
        "revision": snapshot_revision(model_path) if model_path and model_exists else None,
        "gguf_bytes": model_path.stat().st_size if model_path and model_exists else None,
        "gguf_sha256": model_hash,
        "gguf_hash_source": "content_addressed_cache_target" if model_hash else None,
        "embedded_tokenizer_sha256": metadata.get("metadata_summary_sha256"),
        "embedded_chat_template_sha256": metadata.get("chat_template_sha256"),
        "embedded_chat_template_present": metadata.get("chat_template_present") is True,
        "auto_tokenizer_used": False,
        "runtime": "native_llama.cpp_server",
    }
    model_ok = bool(
        resolved
        and resolved.get("hf_id") == QWEN_MODEL_ID
        and model_exists
        and model_path
        and "q4_k_m" in model_path.name.lower()
        and model_hash
        and model_identity["embedded_tokenizer_sha256"]
        and model_identity["embedded_chat_template_present"]
    )
    record(
        gate_row(
            "mandated_gguf_cache_and_embedded_contract",
            {
                "hf_id": QWEN_MODEL_ID,
                "quantization": QUANTIZATION,
                "content_addressed": True,
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            {
                "hf_id": model_identity["hf_id"],
                "quantization": model_identity["quantization"],
                "content_addressed": bool(model_hash),
                "embedded_tokenizer": bool(model_identity["embedded_tokenizer_sha256"]),
                "embedded_chat_template": model_identity["embedded_chat_template_present"],
            },
            model_ok,
            upstream="cached_current_model",
            field="model_identity",
        )
    )
    _progress(2, "model_resolution_end", model=QWEN_MODEL_ID, passed=model_ok)
    tokenizer_loader = fixture.resolve_tokenizer_loader()
    record(
        gate_row(
            "embedded_tokenizer_tool",
            True,
            tokenizer_loader is not None,
            tokenizer_loader is not None,
            upstream="llama_cpp",
            field="Llama",
        )
    )
    server = resolve_native_llama_server()
    _progress(3, "subprocess_start", operation="runner_version_help_linkage", path=str(server))
    runner_rows = shipped_runtime.lease_preflight.collect_runner_capabilities(server)
    _progress(3, "subprocess_end", operation="runner_version_help_linkage")
    runner_errors = shipped_runtime.lease_preflight.runner_capability_errors(runner_rows)
    runner = deepcopy(runner_rows[0]) if runner_rows else {}
    non_thinking = _runner_supports(runner, "--reasoning")
    record(
        gate_row(
            "native_cuda_runtime",
            {"errors": [], "gbnf": True, "non_thinking": True},
            {
                "errors": runner_errors,
                "gbnf": runner.get("grammar_or_json_output") is True,
                "non_thinking": non_thinking,
            },
            not runner_errors and runner.get("grammar_or_json_output") is True and non_thinking,
            upstream="native_llama_server",
            field="cuda_linkage_and_capabilities",
        )
    )
    _progress(4, "subprocess_start", operation="read_only_gpu_and_lease_inventory")
    process_rows, query_receipts = shipped_runtime.lease_preflight.collect_gpu_process_rows()
    lease_rows = shipped_runtime.lease_preflight.scan_lease_rows(
        shipped_runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = shipped_runtime.lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    cache_rows = [
        {
            "repository": QWEN_MODEL_ID,
            "filename": model_path.name if model_path else None,
            "path": str(model_path) if model_path else None,
            "real_path": str(model_path.resolve()) if model_exists and model_path else None,
            "revision": model_identity["revision"],
            "bytes": model_identity["gguf_bytes"],
            "sha256": model_hash,
            "hash_source": model_identity["gguf_hash_source"],
            "weights_opened": False,
            "valid": model_ok,
        }
    ]
    decision = shipped_runtime.lease_preflight.readiness_decision(
        classified, lease_rows, cache_rows, runner_rows
    )
    _progress(4, "subprocess_end", operation="read_only_gpu_and_lease_inventory")
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    record(
        gate_row(
            "gpu_inventory_queries",
            True,
            query_ok,
            query_ok,
            upstream="nvidia-smi",
            field="returncodes",
        )
    )
    available = list(decision.get("available_gpu_uuids") or [])
    record(
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
    context = {
        "schedule": schedule,
        "public_rows": public_rows,
        "authority_rows": authority_rows,
        "model_path": model_path,
        "model_identity": model_identity,
        "tokenizer_loader": tokenizer_loader,
        "server_path": server,
        "runner": runner,
        "runner_rows": runner_rows,
        "non_thinking_supported": non_thinking,
        "process_rows": classified,
        "lease_rows": lease_rows,
        "query_receipts": query_receipts,
        "available_gpu_uuids": available,
    }
    return checks, public_rows, authority_rows, context


def _load_embedded_tokenizer(
    model_path: Path, loader: Callable[..., Any]
) -> tuple[Any | None, Tokenize | None, JsonDict]:  # pragma: no cover
    """Load only the GGUF vocabulary and expose its exact token counter."""

    try:
        owner = loader(model_path=str(model_path), vocab_only=True, n_ctx=8, verbose=False)
    except Exception as exc:  # noqa: BLE001 - the external loader failure is evidence.
        return (
            None,
            None,
            {
                "embedded_tokenizer_available": False,
                "error": f"{type(exc).__name__}:{exc}",
            },
        )

    def tokenize(value: bytes) -> Sequence[int]:
        return owner.tokenize(value, add_bos=False)

    return (
        owner,
        tokenize,
        {
            "embedded_tokenizer_available": True,
            "model_path": str(model_path),
            "vocab_only": True,
            "auto_tokenizer_used": False,
        },
    )


def _free_port() -> int:  # pragma: no cover
    """Ask the kernel for one currently free loopback port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(server: Path, model: Path, port: int) -> list[str]:  # pragma: no cover
    """Use one GPU, the embedded chat template, and supported non-thinking mode."""

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
    """Poll only the owned server and stop at the fixed model-load deadline."""

    started = time.monotonic()
    attempts = 0
    last_error = "not_started"
    while time.monotonic() - started < timeout_s:
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
        "attempts": attempts,
        "last_error": last_error,
        "duration_s": time.monotonic() - started,
    }


def _request_payload(sealed: Mapping[str, Any]) -> tuple[JsonDict, bytes]:  # pragma: no cover
    """Build the actual llama.cpp chat request from the sealed public call."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "Return only one JSON object accepted by the supplied grammar.",
            },
            {"role": "user", "content": str(sealed["prompt"])},
        ],
        **DECODING_PARAMETERS,
        "max_tokens": int(sealed["output_token_budget"]),
        "stream": False,
        "grammar": str(sealed["grammar"]),
    }
    encoded = canonical_json(payload).encode("utf-8")
    return payload, encoded


def _chat_request(
    port: int, sealed: Mapping[str, Any], *, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Run one bounded native call and retain exact request and response bytes."""

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
        "raw_response": body,
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "raw_completion": str(message.get("content") or ""),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "finish_reason": choice.get("finish_reason"),
        "latency_s": time.monotonic() - started,
        "error": None,
    }


def _request_or_error(
    port: int, sealed: Mapping[str, Any], *, timeout_s: float
) -> JsonDict:  # pragma: no cover
    """Preserve transport faults as rows instead of dropping their denominator."""

    started = time.monotonic()
    payload, encoded = _request_payload(sealed)
    try:
        return _chat_request(port, sealed, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - exact transport failure is evidence.
        return {
            "raw_request": payload,
            "raw_request_bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "raw_response": {},
            "raw_response_bytes_b64": "",
            "raw_completion": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": None,
            "latency_s": time.monotonic() - started,
            "error": f"{type(exc).__name__}:{exc}",
        }


def _generation_overlap_sample(
    holder: list[JsonDict], identity: Mapping[str, Any], gpu_uuid: str, call_id: str
) -> threading.Thread:  # pragma: no cover
    """Sample owned VRAM concurrently with one native generation request."""

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

    thread = threading.Thread(target=sample, name=f"exp7209-gpu-{call_id[-8:]}", daemon=True)
    thread.start()
    return thread


def _transport_fault_key(error_text: Any) -> str | None:  # pragma: no cover
    """Reduce a transport error to a stable class for deterministic-stop logic."""

    if not error_text:
        return None
    return str(error_text).split(":", 1)[0]


def _live_capture(
    context: Mapping[str, Any], checkpoint_dir: Path, raw_dir: Path, spans: list[JsonDict]
) -> JsonDict:  # pragma: no cover
    """Own one GPU and native server for the fixed 32-call live window."""

    schedule = list(context["schedule"])
    gpu_uuid = str(context["available_gpu_uuids"][0])
    device = shipped_runtime._selected_device(context, gpu_uuid)
    gpu_index = int(device["gpu_index"])
    model = Path(str(context["model_path"]))
    port = _free_port()
    command = _server_command(Path(context["server_path"]), model, port)
    contract = supervisor_contract(
        outer_deadline_s=LIVE_WINDOW_CAP_S,
        health_timeout_s=MODEL_LOAD_CAP_S,
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
    rows: list[JsonDict] = []
    samples: list[JsonDict] = []
    snapshots: list[JsonDict] = []
    provenance: JsonDict = {"provenance_ok": False}
    cleanup: JsonDict = {"action": "not_started", "leak_free": True}
    lease_release: JsonDict = {"released": False}
    runtime_error: str | None = None
    model_loaded = False
    live_started = time.monotonic()
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    prefill_s = 0.0
    generation_s = 0.0
    parsing_s = 0.0
    try:
        with _phase_span(spans, 5, "task_owned_lease_and_model_load"):
            snapshots.append(shipped_runtime._gpu_snapshot("before_model_load", phase=5))
            lease = lease_api.GpuLease.acquire(
                runtime_dir=shipped_runtime.lease_preflight.LEASE_RUNTIME_DIR,
                task_id=TASK_ID,
                device_uuid=gpu_uuid,
                expected_model=str(model),
                vram_before_mb=int(device.get("gpu_memory_used_mb", 0) or 0),
                ttl_s=LIVE_WINDOW_CAP_S,
            )
            lease.transition("admitted")
            lease.transition("loading")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            _progress(5, "subprocess_start", operation="owned_native_llama_server", command=command)
            _progress(5, "model_load_start", model=QWEN_MODEL_ID, gpu_uuid=gpu_uuid)
            identity = supervisor.launch()
            load_started = time.monotonic()
            with _stream_server_log(supervisor), _heartbeat(5, "native_model_load", lambda: 0, 1):
                health = _wait_for_health(supervisor, port, MODEL_LOAD_CAP_S)
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

        previous_fault: str | None = None
        repeated_faults = 0
        with _phase_span(spans, 6, "fixed_32_call_span_canary"):
            for sealed in schedule:
                remaining = LIVE_WINDOW_CAP_S - (time.monotonic() - live_started)
                if remaining <= 0.25:
                    runtime_error = "live_window_cap_reached"
                    break
                lease.heartbeat()
                _progress(
                    6,
                    "generation_start",
                    call_id=sealed["call_id"],
                    arm=sealed["arm"],
                    call_type=sealed["call_type"],
                    completed_units=len(rows),
                    total_units=32,
                )
                sample_offset = len(samples)
                sample_thread = _generation_overlap_sample(
                    samples, identity, gpu_uuid, str(sealed["call_id"])
                )
                with _heartbeat(6, "bounded_generation", lambda: len(rows), 32):
                    response = _request_or_error(
                        port,
                        sealed,
                        timeout_s=min(REQUEST_CAP_S, max(0.25, remaining)),
                    )
                request_ended = time.monotonic()
                sample_thread.join(timeout=10.0)
                new_samples = samples[sample_offset:]
                for sample in new_samples:
                    sample["request_end_monotonic_s"] = request_ended
                    sample["request_interval_overlap"] = (
                        sample["sample_start_monotonic_s"] <= request_ended
                    )
                sample = new_samples[0] if new_samples else {}
                resource = {
                    "server_pid": identity.get("pid"),
                    "server_pid_start_ticks": identity.get("start_time_ticks"),
                    "gpu_uuid": gpu_uuid,
                    "lease_id": lease.lease_id,
                    "cuda_offload_confirmed": provenance.get("provenance_ok") is True,
                    "gpu_sample_sha256": sha256_json(sample) if sample else None,
                }
                parse_started = time.monotonic()
                row = build_completion_row(sealed, response, resource)
                parsing_s += time.monotonic() - parse_started
                timings = dict(response.get("raw_response", {}).get("timings") or {})
                prefill_s += float(timings.get("prompt_ms", 0.0) or 0.0) / 1000.0
                generation_s += (
                    float(
                        timings.get("predicted_ms", float(response.get("latency_s", 0.0)) * 1000.0)
                        or 0.0
                    )
                    / 1000.0
                )
                rows.append(row)
                atomic_write_json(
                    raw_dir / f"call_{int(sealed['call_order']):02d}.json",
                    {"schedule": dict(sealed), "completion": row},
                    allow_override=False,
                    sort_keys=True,
                )
                atomic_write_json(
                    checkpoint_dir / "checkpoint_latest.json",
                    {
                        "row_count": len(rows),
                        "rows": rows,
                        "schedule_sha256": sha256_json(schedule),
                    },
                    allow_override=False,
                    sort_keys=True,
                )
                _progress(
                    6,
                    "generation_end",
                    call_id=sealed["call_id"],
                    terminal_state=row["terminal_state"],
                    completion_tokens=row["completion_tokens"],
                    duration_s=round(row["latency_s"], 6),
                    completed_units=len(rows),
                    total_units=32,
                )
                fault = _transport_fault_key(response.get("error"))
                if fault and fault == previous_fault:
                    repeated_faults += 1
                elif fault:
                    repeated_faults = 1
                else:
                    repeated_faults = 0
                previous_fault = fault
                if repeated_faults >= 2:
                    runtime_error = f"repeated_deterministic_transport_fault:{fault}"
                    break
    except Exception as exc:  # noqa: BLE001 - partial live evidence must survive.
        runtime_error = f"{type(exc).__name__}:{exc}"
        _progress(6, "runtime_error", error=runtime_error, completed_units=len(rows))
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
            if lease is not None:
                try:
                    lease_release = shipped_runtime._release_lease(
                        lease,
                        cleanup,
                        after,
                        identity,
                        len(rows) == 32 and runtime_error is None,
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
    overlap_count = sum(
        sample.get("request_interval_overlap") is True and bool(sample.get("owned_compute_apps"))
        for sample in samples
    )
    provenance_ok = bool(
        model_loaded
        and provenance.get("provenance_ok") is True
        and overlap_count == len(rows)
        and len(rows) > 0
        and cleanup.get("leak_free") is True
        and lease_release.get("released") is True
    )
    spans.extend(
        [
            {"phase": 6, "name": "prefill_aggregate", "duration_s": prefill_s},
            {"phase": 6, "name": "generation_aggregate", "duration_s": generation_s},
            {"phase": 7, "name": "parsing_aggregate", "duration_s": parsing_s},
        ]
    )
    return {
        "rows": rows,
        "runtime_error": runtime_error,
        "model_loaded": model_loaded,
        "model_invoked": bool(rows),
        "gpu_receipts": {
            "preflight_process_rows": deepcopy(context["process_rows"]),
            "preflight_lease_rows": deepcopy(context["lease_rows"]),
            "inventory_query_receipts": deepcopy(context["query_receipts"]),
            "server_identity": deepcopy(identity),
            "runtime_provenance": provenance,
            "generation_overlap_samples": samples,
            "generation_overlap_count": overlap_count,
            "generation_overlap_denominator": len(rows),
            "generation_overlap_ok": overlap_count == len(rows) and len(rows) > 0,
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
            "command": command,
            "context_tokens": CONTEXT_TOKEN_BUDGET,
            "request_cap_s": REQUEST_CAP_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
            "live_window_cap_s": LIVE_WINDOW_CAP_S,
            "non_thinking_supported": context["non_thinking_supported"],
            "non_thinking_enabled": "--reasoning" in command and "off" in command,
            "decoding_parameters": deepcopy(DECODING_PARAMETERS),
        },
    }


def _write_checkpoint(
    path: Path, artifact: Mapping[str, Any], started: float
) -> None:  # pragma: no cover
    """Write nonterminal progress only below the checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _write_terminal(
    artifact: JsonDict, result_path: Path, checkpoint_path: Path, started: float
) -> JsonDict:  # pragma: no cover
    """Cold-check once, then atomically publish the stable terminal artifact."""

    validation_started = time.monotonic()
    _progress(9, "validation_start", path=str(result_path))
    artifact["duration_s"] = time.monotonic() - started
    artifact["phase_spans"].append(
        {
            "phase": 9,
            "name": "cold_artifact_verification",
            "duration_s": time.monotonic() - validation_started,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    _progress(9, "validation_end", errors=errors)
    if errors:
        raise ValueError(f"invalid Exp7209 artifact: {errors}")
    _progress(10, "write_start", path=str(result_path))
    _write_checkpoint(checkpoint_path, artifact, started)
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(10, "write_end", path=str(result_path))
    return artifact


def run_experiment(
    root: Path | None = None, run_date: str = RUN_DATE
) -> JsonDict:  # pragma: no cover
    """Run one finite canary attempt and preserve a valid terminal outcome."""

    _progress(0, "start", detail="print before checking any prerequisite")
    started = time.monotonic()
    repo = root or find_repo_root(start=__file__)
    result_path = repo / RESULT_PATH
    checkpoint_dir = repo / CHECKPOINT_DIR
    raw_dir = repo / RAW_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "terminal_candidate.json"
    artifact = base_artifact(run_date)
    _write_checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    spans: list[JsonDict] = artifact["phase_spans"]
    with _phase_span(spans, 1, "preconditions_and_upstream_authentication"):
        checks, public_rows, authority_rows, context = _collect_preflight(
            repo, run_date, result_path, checkpoint_dir, raw_dir
        )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is not None:
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        return _write_terminal(artifact, result_path, checkpoint_path, started)

    schedule = list(context["schedule"])
    owner = None
    tokenizer = None
    with _phase_span(spans, 2, "embedded_tokenizer_load_and_size_gate"):
        _progress(2, "model_load_start", operation="embedded_tokenizer_vocab_only")
        owner, tokenizer, tokenizer_load = _load_embedded_tokenizer(
            Path(context["model_path"]), context["tokenizer_loader"]
        )
        _progress(
            2,
            "model_load_end",
            operation="embedded_tokenizer_vocab_only",
            available=tokenizer_load["embedded_tokenizer_available"],
        )
        if tokenizer is None:
            token_receipt = {"measurement_status": "tokenizer_unavailable", **tokenizer_load}
        else:
            token_receipt = measure_token_budgets(schedule, tokenizer)
            token_receipt["tokenizer_load_receipt"] = tokenizer_load
    if owner is not None:
        owner.close()
    artifact["token_budget_receipt"] = token_receipt
    tokenizer_ok = tokenizer is not None
    tokenizer_check = gate_row(
        "embedded_tokenizer_load",
        True,
        tokenizer_ok,
        tokenizer_ok,
        upstream="cached_qwen_gguf",
        field="embedded_tokenizer",
    )
    checks.append(tokenizer_check)
    if not tokenizer_ok:
        artifact["preconditions_checked"] = checks
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        return _write_terminal(artifact, result_path, checkpoint_path, started)
    fit = token_receipt["all_forms_fit_with_20_percent_headroom"] is True
    fit_check = gate_row(
        "representation_size_fit",
        True,
        fit,
        fit,
        upstream="embedded_qwen_tokenizer",
        field="token_budget_receipt",
    )
    checks.append(fit_check)
    artifact["preconditions_checked"] = checks
    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    artifact["model_identity_receipt"]["token_budget_receipt_sha256"] = sha256_json(token_receipt)
    if not fit:
        artifact["gate_check_summary"] = gate_summary(fit_check)
        finalize_measured_artifact(
            artifact,
            [],
            [],
            token_receipt,
            duration_s=time.monotonic() - started,
            live_evidence=False,
        )
        return _write_terminal(artifact, result_path, checkpoint_path, started)

    with _phase_span(spans, 3, "freeze_prompts_grammars_settings_and_hashes"):
        _progress(3, "write_start", path=str(raw_dir))
        atomic_write_json(
            raw_dir / "grammar_contract.json",
            {
                "schedule_grammar_hashes": [row["grammar_sha256"] for row in schedule],
                "tuple_schema_sha256": sha256_json(fixture.TUPLE_SCHEMA),
                "generator_sha256": sha256_file(repo / MODULE_PATH),
            },
            allow_override=False,
            sort_keys=True,
        )
        _progress(3, "write_end", path=str(raw_dir))

    capture = _live_capture(context, checkpoint_dir, raw_dir, spans)
    completion_rows = list(capture["rows"])
    with _phase_span(spans, 7, "compiler_executor_and_canary_authority_verification"):
        _progress(7, "verification_start", completed_calls=len(completion_rows), total_calls=32)
        comparison_rows = score_completion_pairs(schedule, completion_rows, authority_rows)
        _progress(
            7, "verification_end", completed_comparisons=len(comparison_rows), total_comparisons=16
        )
    artifact["gpu_receipts"] = capture["gpu_receipts"]
    artifact["runner_receipt"] = capture["runner_receipt"]
    artifact["model_invoked"] = capture["model_invoked"]
    external_runtime_failure = bool(
        capture["runtime_error"]
        or len(completion_rows) != 32
        or capture["gpu_receipts"].get("provenance_ok") is not True
    )
    if external_runtime_failure:
        runtime_check = gate_row(
            "live_runtime_completion_and_cuda_provenance",
            {"calls": 32, "runtime_error": None, "provenance_ok": True},
            {
                "calls": len(completion_rows),
                "runtime_error": capture["runtime_error"],
                "provenance_ok": capture["gpu_receipts"].get("provenance_ok"),
            },
            False,
            upstream="owned_native_llama_server",
            field="live_capture",
        )
        checks.append(runtime_check)
        artifact["canary_rows"] = completion_rows
        artifact["rows"] = comparison_rows
        artifact["sample_size_budget"].update(
            {
                "attempted_calls": len(completion_rows),
                "completed_calls": sum(
                    row["terminal_state"] == "complete" for row in completion_rows
                ),
                "censored_calls": 32 - len(completion_rows),
                "completed_comparisons": len(comparison_rows),
            }
        )
        artifact["model_invoked"] = bool(completion_rows)
        if completion_rows:
            artifact["inference_substrate"] = "live_llm_inference"
            artifact["inference_substrate_class"] = "model_bounded_generation"
            artifact["inference_mode"] = "live_gpu"
        finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
    else:
        finalize_measured_artifact(
            artifact,
            completion_rows,
            comparison_rows,
            token_receipt,
            duration_s=time.monotonic() - started,
            live_evidence=True,
        )
    manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, artifact["model_identity_receipt"]
    )
    artifact["raw_manifest"] = manifest
    artifact["source_artifact_hashes"]["raw_manifest"] = sha256_file(raw_dir / "raw_manifest.json")
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _write_terminal(artifact, result_path, checkpoint_path, started)


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the experiment contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the canary and accept any cold-valid terminal outcome."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    args = parser.parse_args(argv)
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7209] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7209] terminal verdict={artifact['honest_verdict']} "
        f"score={artifact['span_canary_ready_score']}",
        flush=True,
    )
    return 0
