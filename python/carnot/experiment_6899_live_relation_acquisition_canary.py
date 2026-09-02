"""Prove authentic local relation acquisition before a larger corpus runs.

Spec refs: REQ-INFERENCE-6899 and SCENARIO-INFERENCE-6899-*.

This canary checks transport and process evidence. It does not score whether a
tuple is correct. Exp6888 remains the later independent semantic boundary.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib import error, request

from carnot import experiment_6887_three_family_relation_proposal_corpus as base
from carnot import gpu_lease_phase_journal as lease_api
from carnot.inference.llama_cpp_process import OwnedLlamaCppProcess, port_is_free
from carnot.inference.llama_server_supervisor import read_process_identity
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6899_live_relation_acquisition_canary.json")
EXP6886_RELATIVE_PATH = Path("results/experiment_6886_enoki_exact_relation_fixture.json")
RUN_DATE = "20260902"
RANDOM_SEED = 6899
SEEDS = (6899, 6900, 6901, 6902)
SCHEMA = "carnot.exp6899.live_relation_acquisition_canary.v1"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_authenticity_canary"
EXPECTED_EXP6886_SHA256 = base.EXPECTED_EXP6886_SHA256
EXPECTED_FIXTURE_HASHES = {
    "calibration": "sha256:5f890d4d30bd814b7bb93b27c77f5c630069c8131873e34b3bfdb3a7aa0df797",
    "held": "sha256:2e5a4879996cd95742ec3c6a123a344eb96224d8bd151c1bce6400b22b621ab5",
}
EXPECTED_CANARY_SOURCE_HASH = (
    "sha256:763ac9ae2a3452772f99d74abf50c1e22828179eea7b3a283f85c1bb91e6bee5"
)
MODEL_SPECS = base.MODEL_SPECS
EXPECTED_TOKENIZER_HASHES = base.EXPECTED_TOKENIZER_HASHES
EXPECTED_MODEL_BINDINGS = {
    MODEL_SPECS[0]: {
        **base.EXPECTED_MODEL_BINDINGS[MODEL_SPECS[0]],
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    },
    MODEL_SPECS[1]: {
        **base.EXPECTED_MODEL_BINDINGS[MODEL_SPECS[1]],
        "filename": "gemma-4-31B-it-Q4_K_M.gguf",
    },
    MODEL_SPECS[2]: {
        **base.EXPECTED_MODEL_BINDINGS[MODEL_SPECS[2]],
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    },
}
MODEL_FAMILIES = base.MODEL_FAMILIES
PROTOCOL_LINE = base.PROTOCOL_LINE
OUTPUT_TOKEN_BUDGET = 128
CONTEXT_LENGTH = 2048
REQUEST_TIMEOUT_S = 180.0
HEALTH_TIMEOUT_S = 600.0
LEASE_TTL_S = 1800.0
LEASE_RUNTIME_DIR = Path("/tmp/carnot-gpu-leases")
MIN_FREE_VRAM_MB = 24_000
RECEIPT_MAX_AGE_S = 300.0
RECEIPT_MAX_AGE_NS = int(RECEIPT_MAX_AGE_S * 1_000_000_000)
TOKENIZER_TIMEOUT_S = 120.0
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

SOURCE_ARTIFACT_HASHES = {
    "exp5786": {
        "path": "results/experiment_5786_sota_constraint_stream.json",
        "sha256": "sha256:260d065ca32ddc0a8a88c38ca8b9300d6be1bceb5b8ea2571ddcd481c03ef562",
    },
    "exp5923": {
        "path": "results/experiment_5923_sota_schema_supported_constraintir_ab.json",
        "sha256": "sha256:ebe0e6fdcce1fb4d1bed9b4be5b2ea89dba6239b33498bd903ff60f8d2044a6c",
    },
    "exp6886": {
        "path": EXP6886_RELATIVE_PATH.as_posix(),
        "sha256": EXPECTED_EXP6886_SHA256,
    },
    "exp6887": {
        "path": "results/experiment_6887_three_family_relation_proposal_corpus.json",
        "sha256": "sha256:99800fdbad94d0f1387a2a57ae3abb9169ece3ffeb3ac7b5485ddfd24123cef3",
    },
    "exp6888": {
        "path": "results/experiment_6888_independent_relation_qualification.json",
        "sha256": "sha256:b74ec7cd5a3f7a7aad264b067a3e7f3bd32ec8415b9d3a34133116f6899c59ea",
    },
    "fixture_manifests": deepcopy(EXPECTED_FIXTURE_HASHES),
    "canary_sources": {"sha256": EXPECTED_CANARY_SOURCE_HASH},
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "model_specs",
    "models_used",
    "model_artifact_hashes",
    "tokenizer_receipts",
    "llama_cpp_receipts",
    "gpu_lease_rows",
    "server_lifecycle_rows",
    "prompt_manifest",
    "rows",
    "raw_request_manifest",
    "raw_output_manifest",
    "output_byte_rows",
    "generated_token_rows",
    "stop_reason_rows",
    "parse_attempt_rows",
    "parse_coverage_by_model",
    "empty_cell_count",
    "canned_cell_count",
    "held_sidecar_access_count",
    "enoki_control_rows",
    "rule_control_rows",
    "external_text_scorer_call_count",
    "constrained_schema_decode_count",
    "model_weight_mutation_count",
    "random_seed",
    "reproducibility_checksum",
    "relation_canary_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why its evidence matters.",
    "preconditions_checked": "Exact resource gates stop fabricated live evidence.",
    "inference_substrate": "The substrate rules out a hidden remote or CPU substitute.",
    "duration_s": "Measured wall time exposes skipped live work.",
    "source_artifact_hashes": "Hashes bind the fixture and the failed prior evidence.",
    "model_specs": "Exact repository names prevent family substitution.",
    "models_used": "Only models with live receipts can appear as used.",
    "model_artifact_hashes": "File hashes bind each process to intended weights.",
    "tokenizer_receipts": "Native receipts prevent tokenizer substitution.",
    "llama_cpp_receipts": "Server receipts bind weights, process, CUDA, and logs.",
    "gpu_lease_rows": "Owned leases prevent accidental use of another task's GPU.",
    "server_lifecycle_rows": "Lifecycle rows prove narrow and complete cleanup.",
    "prompt_manifest": "One plain prompt contract makes cells comparable.",
    "rows": "Per-cell rows keep failures visible and replayable.",
    "raw_request_manifest": "Exact request bytes prove what each process received.",
    "raw_output_manifest": "Exact response bytes expose empty transport.",
    "output_byte_rows": "Byte counts reject the prior empty-content failure.",
    "generated_token_rows": "Token counts prove generation advanced past the prompt.",
    "stop_reason_rows": "Observed stops separate completion from transport loss.",
    "parse_attempt_rows": "Parser receipts prove every live output reached parsing.",
    "parse_coverage_by_model": "Per-model coverage prevents pooled success from hiding failure.",
    "empty_cell_count": "A zero count is required because empty terminal cells are invalid.",
    "canned_cell_count": "A zero count rejects outputs repeated across different sources.",
    "held_sidecar_access_count": "Zero keeps semantic authority outside acquisition.",
    "enoki_control_rows": "Separate Enoki rows cannot substitute for live GGUF evidence.",
    "rule_control_rows": "Separate lexical rows cannot substitute for live GGUF evidence.",
    "external_text_scorer_call_count": "Zero keeps external judgment out of acquisition.",
    "constrained_schema_decode_count": "Zero proves decoding used no grammar or schema mask.",
    "model_weight_mutation_count": "Zero proves the canary did not train its models.",
    "random_seed": "The fixed seed root makes the cell matrix reproducible.",
    "reproducibility_checksum": "One digest binds stable inputs and acquired evidence.",
    "relation_canary_ready_score": "One permits a larger corpus only after every live gate passes.",
    "gate_check_summary": "Exact expected and observed values make failures actionable.",
    "verifier_is_oracle": "False keeps transport checks separate from semantic truth.",
    "verdict_class": "A closed class gives automation a stable terminal state.",
    "honest_verdict": "A complete_ prefix marks a terminal evidence boundary.",
}


class RelationCanaryError(RuntimeError):
    """Report an unsafe canary condition without inventing replacement evidence."""


def canonical_json(value: Any) -> str:
    """Serialize stable JSON for request bytes and content hashes."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the repository digest prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON value."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _b64(value: bytes) -> str:
    """Encode bytes without relying on JSON's text assumptions."""

    return base64.b64encode(value).decode("ascii")


def _unb64(value: Any) -> bytes | None:
    """Decode strict base64, or return none for malformed evidence."""

    try:
        return base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError):
        return None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact expected-versus-observed gate row."""

    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all failures and repeat the first one for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failures = [row for row in rows if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "checks": rows,
        "passed": not failures,
        "failed_check": first.get("check") if first else None,
        "expected": first.get("expected") if first else "all checks pass",
        "observed": first.get("observed") if first else "all checks pass",
        "failed_checks": failures,
    }


def resolve_three_models(
    *,
    pair_provider: Callable[..., Sequence[Mapping[str, Any]] | None] = cached_sota_pair,
    dense_resolver: Callable[[str, str], str | None] = resolve_cached_gguf,
) -> list[JsonDict]:
    """Resolve the canonical pair first and add the required dense model."""

    if pair_provider is cached_sota_pair:  # pragma: no cover - live cache boundary.
        pair = cached_sota_pair(gpu_indices=(0, 0), preferred_quant="Q4_K_M", model_indices=(0, 1))
    else:
        pair = pair_provider(gpu_indices=(0, 0), preferred_quant="Q4_K_M", model_indices=(0, 1))
    by_id = {
        str(row.get("hf_id")): deepcopy(dict(row)) for row in pair or [] if isinstance(row, Mapping)
    }
    if dense_resolver is resolve_cached_gguf:  # pragma: no cover - live cache boundary.
        dense_path = resolve_cached_gguf(MODEL_SPECS[1], "Q4_K_M")
    else:
        dense_path = dense_resolver(MODEL_SPECS[1], "Q4_K_M")
    if dense_path:
        by_id[MODEL_SPECS[1]] = {
            "hf_id": MODEL_SPECS[1],
            "model_path": dense_path,
            "gpu": 0,
        }
    rows: list[JsonDict] = []
    for hf_id in MODEL_SPECS:
        row = deepcopy(dict(by_id.get(hf_id, {})))
        row.update({"hf_id": hf_id, "gpu": int(row.get("gpu", 0))})
        if row.get("model_path"):
            row["model_path"] = str(Path(str(row["model_path"])).absolute())
        rows.append(row)
    return rows


def select_canary_sources(upstream: Mapping[str, Any]) -> list[JsonDict]:
    """Select one ASCII calibration source from each frozen fixture family."""

    sources = base.select_source_records(upstream)[5:10]
    if {str(row["family"]) for row in sources} != set(base.FAMILIES):
        raise RelationCanaryError("canary_family_matrix_drift")
    observed = sha256_json(sources)
    if observed != EXPECTED_CANARY_SOURCE_HASH:
        raise RelationCanaryError(f"canary_source_hash_drift:{observed}")
    return sources


def build_prompt(source: Mapping[str, Any]) -> str:
    """Render one source-only request with a plain seven-field line protocol."""

    predicates = ", ".join(str(value) for value in source["allowed_predicates"])
    source_text = str(source["source_text"])
    byte_spans = " ".join(
        f"{len(source_text[: match.start()].encode('utf-8'))}:"
        f"{len(source_text[: match.end()].encode('utf-8'))}={match.group()}"
        for match in re.finditer(r"\S+", source_text)
    )
    return (
        "Extract source-anchored relations. /no_think\n"
        f"Allowed predicate: {predicates}\n"
        "Count UTF-8 bytes from the first source byte. End offsets are exclusive.\n"
        "Output exactly seven fields separated by an actual U+0009 TAB, not backslash+t.\n"
        "The first field must be the literal REL; it is not the predicate.\n"
        "Fields 2, 3, 5, and 6 must be integer offsets; do not output mention text.\n"
        f"Output one or more lines in this exact format: {PROTOCOL_LINE}\n"
        "POLARITY must be positive or negative. Output protocol lines only.\n"
        f"Source-only token byte spans (start:end=text): {byte_spans}\n"
        "SOURCE BEGIN\n"
        f"{source_text}\n"
        "SOURCE END"
    )


def build_request_bytes(source: Mapping[str, Any], seed: int) -> bytes:
    """Build the exact unconstrained chat request sent to llama.cpp."""

    payload = {
        "messages": [{"role": "user", "content": build_prompt(source)}],
        "max_tokens": OUTPUT_TOKEN_BUDGET,
        "temperature": 0.1,
        "top_p": 0.9,
        "seed": int(seed),
        "stream": False,
    }
    return canonical_json(payload).encode("utf-8")


def process_identity_errors(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Compare fields that distinguish a live process from PID reuse."""

    return [
        field
        for field in ("pid", "start_time_ticks", "uid", "command_hash")
        if recorded.get(field) != current.get(field)
    ]


def build_live_cell(
    *,
    hf_id: str,
    source: Mapping[str, Any],
    seed: int,
    raw_request_bytes: bytes,
    raw_http_response_bytes: bytes,
    raw_output_bytes: bytes,
    native_prompt_tokens: int,
    generated_tokens: int,
    stop_reason: str,
    wall_time_s: float,
    timed_out: bool,
    truncated: bool,
    parser_attempted: bool,
    parse_rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Preserve exact bytes and the parser receipt for one live model cell."""

    output_hash = sha256_bytes(raw_output_bytes)
    return {
        "cell_identity": f"{hf_id}::{seed}::{source['fixture_id']}",
        "hf_id": hf_id,
        "model_family": MODEL_FAMILIES[hf_id],
        "seed": int(seed),
        "fixture_id": source["fixture_id"],
        "family": source["family"],
        "source_text_hash": source["source_text_hash"],
        "prompt_sha256": sha256_bytes(build_prompt(source).encode("utf-8")),
        "raw_request_b64": _b64(raw_request_bytes),
        "raw_request_sha256": sha256_bytes(raw_request_bytes),
        "request_byte_count": len(raw_request_bytes),
        "raw_http_response_b64": _b64(raw_http_response_bytes),
        "raw_http_response_sha256": sha256_bytes(raw_http_response_bytes),
        "http_response_byte_count": len(raw_http_response_bytes),
        "raw_output_b64": _b64(raw_output_bytes),
        "raw_output_sha256": output_hash,
        "output_byte_count": len(raw_output_bytes),
        "native_prompt_tokens": int(native_prompt_tokens),
        "generated_tokens": int(generated_tokens),
        "stop_reason": str(stop_reason),
        "wall_time_s": round(float(wall_time_s), 6),
        "timed_out": bool(timed_out),
        "truncated": bool(truncated),
        "parser_attempted": bool(parser_attempted),
        "parser_input_sha256": output_hash if parser_attempted else "",
        "parse_rows": [deepcopy(dict(row)) for row in parse_rows],
        "runtime_receipt": deepcopy(dict(runtime_receipt)),
        "teardown_outcome": deepcopy(dict(runtime_receipt.get("teardown_outcome") or {})),
        "terminal": True,
    }


def live_cell_errors(row: Mapping[str, Any]) -> list[str]:
    """Name every transport, process, CUDA, and parser authenticity failure."""

    errors: list[str] = []
    request_bytes = _unb64(row.get("raw_request_b64"))
    http_bytes = _unb64(row.get("raw_http_response_b64"))
    output_bytes = _unb64(row.get("raw_output_b64"))
    if request_bytes is None or len(request_bytes) != row.get("request_byte_count"):
        errors.append("request_bytes")
    elif sha256_bytes(request_bytes) != row.get("raw_request_sha256"):
        errors.append("request_hash")
    if (
        http_bytes is None
        or not http_bytes
        or len(http_bytes) != row.get("http_response_byte_count")
    ):
        errors.append("http_response_bytes")
    elif sha256_bytes(http_bytes) != row.get("raw_http_response_sha256"):
        errors.append("http_response_hash")
    if (
        output_bytes is None
        or not output_bytes
        or len(output_bytes) != row.get("output_byte_count")
    ):
        errors.append("empty_bytes")
    elif sha256_bytes(output_bytes) != row.get("raw_output_sha256"):
        errors.append("output_hash")
    if int(row.get("native_prompt_tokens", 0) or 0) <= 0:
        errors.append("zero_prompt_tokens")
    if int(row.get("generated_tokens", 0) or 0) <= 0:
        errors.append("zero_generated_tokens")
    if not str(row.get("stop_reason", "")):
        errors.append("missing_stop_reason")
    if row.get("timed_out") is True:
        errors.append("timeout")
    if row.get("truncated") is True:
        errors.append("truncation")
    if row.get("parser_attempted") is not True:
        errors.append("parser_bypass")
    elif row.get("parser_input_sha256") != row.get("raw_output_sha256"):
        errors.append("parser_input_hash")
    receipt = row.get("runtime_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    hf_id = str(row.get("hf_id", ""))
    binding = EXPECTED_MODEL_BINDINGS.get(hf_id, {})
    if receipt.get("authentic") is not True:
        errors.append("runtime_authenticity")
    if receipt.get("model_sha256") != binding.get("sha256"):
        errors.append("wrong_model_file")
    if receipt.get("tokenizer_sha256") != EXPECTED_TOKENIZER_HASHES.get(hf_id):
        errors.append("tokenizer_substitution")
    if int(receipt.get("offload_layers", 0) or 0) <= 0:
        errors.append("zero_offload")
    if receipt.get("owned_cuda_residency") is not True or not receipt.get("gpu_uuid"):
        errors.append("cuda_authenticity")
    if int(receipt.get("server_pid", 0) or 0) <= 1:
        errors.append("server_pid")
    if int(receipt.get("server_start_time_ticks", 0) or 0) <= 0:
        errors.append("server_start_time_ticks")
    if receipt.get("process_identity_match") is not True:
        errors.append("pid_reuse")
    if float(receipt.get("receipt_age_s", RECEIPT_MAX_AGE_S + 1)) > RECEIPT_MAX_AGE_S:
        errors.append("stale_receipt")
    teardown = row.get("teardown_outcome")
    teardown = teardown if isinstance(teardown, Mapping) else {}
    if teardown.get("leak_free") is not True:
        errors.append("teardown_failure")
    return errors


def detect_canned_cells(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Find one model response reused for distinct source prompts."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("raw_output_sha256"))].append(row)
    canned: list[JsonDict] = []
    for output_hash, members in groups.items():
        source_hashes = {str(row.get("source_text_hash")) for row in members}
        if output_hash == sha256_bytes(b"") or len(source_hashes) <= 1:
            continue
        canned.extend(
            {
                "cell_identity": row.get("cell_identity"),
                "hf_id": row.get("hf_id"),
                "raw_output_sha256": output_hash,
                "distinct_source_count": len(source_hashes),
            }
            for row in members
        )
    return sorted(canned, key=lambda row: str(row["cell_identity"]))


def parse_coverage(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute surface parse coverage independently for each required model."""

    result: JsonDict = {}
    for hf_id in MODEL_SPECS:
        model_rows = [row for row in rows if row.get("hf_id") == hf_id]
        parsed = sum(
            any(
                isinstance(parse_row, Mapping) and parse_row.get("status") == "accepted"
                for parse_row in row.get("parse_rows", [])
            )
            for row in model_rows
        )
        result[hf_id] = {
            "cell_count": len(model_rows),
            "parsed_cell_count": parsed,
            "parse_coverage": parsed / len(model_rows) if model_rows else 0.0,
        }
    return result


def _receipt_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate one fresh process and exact model binding per family."""

    by_model = {str(row.get("hf_id")): row for row in rows}
    errors: list[str] = []
    identities: set[tuple[int, int]] = set()
    for hf_id in MODEL_SPECS:
        row = by_model.get(hf_id, {})
        if row.get("model_sha256") != EXPECTED_MODEL_BINDINGS[hf_id]["sha256"]:
            errors.append(f"{hf_id}:model_sha256")
        if row.get("tokenizer_sha256") != EXPECTED_TOKENIZER_HASHES[hf_id]:
            errors.append(f"{hf_id}:tokenizer_sha256")
        if row.get("owned_cuda_residency") is not True:
            errors.append(f"{hf_id}:cuda")
        if int(row.get("offload_layers", 0) or 0) <= 0:
            errors.append(f"{hf_id}:offload")
        identity = (int(row.get("pid", 0) or 0), int(row.get("start_time_ticks", 0) or 0))
        if identity[0] <= 1 or identity[1] <= 0:
            errors.append(f"{hf_id}:process_identity")
        identities.add(identity)
    if len(identities) != len(MODEL_SPECS):
        errors.append("fresh_process_identity")
    return errors


def _lease_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require one owned and released task lease per GGUF process."""

    by_model = {str(row.get("hf_id")): row for row in rows}
    return [
        hf_id
        for hf_id in MODEL_SPECS
        if by_model.get(hf_id, {}).get("owned") is not True
        or by_model.get(hf_id, {}).get("released") is not True
    ]


def _lifecycle_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require clean narrow teardown for every fresh server lifecycle."""

    by_model = {str(row.get("hf_id")): row for row in rows}
    errors: list[str] = []
    for hf_id in MODEL_SPECS:
        row = by_model.get(hf_id, {})
        for field in (
            "process_identity_match",
            "process_exit_confirmed",
            "process_reaped",
            "port_release_confirmed",
            "lease_released",
            "leak_free",
        ):
            if row.get(field) is not True:
                errors.append(f"{hf_id}:{field}")
        if int(row.get("unrelated_process_signal_count", -1) or 0) != 0:
            errors.append(f"{hf_id}:unrelated_process_signal_count")
        if row.get("teardown_error"):
            errors.append(f"{hf_id}:teardown_error")
    return errors


def _tokenizer_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recheck native tokenizer identities at the outgoing gate."""

    by_model = {str(row.get("hf_id")): row for row in rows}
    return [
        hf_id
        for hf_id in MODEL_SPECS
        if by_model.get(hf_id, {}).get("source") != "native_embedded_gguf_llama_cpp_vocab_only"
        or by_model.get(hf_id, {}).get("loadable") is not True
        or by_model.get(hf_id, {}).get("used_hf_autotokenizer") is not False
        or by_model.get(hf_id, {}).get("canonical_tokenizer_payload_sha256")
        != EXPECTED_TOKENIZER_HASHES[hf_id]
    ]


def readiness(
    *,
    duration_s: float,
    rows: Sequence[Mapping[str, Any]],
    llama_cpp_receipts: Sequence[Mapping[str, Any]],
    gpu_lease_rows: Sequence[Mapping[str, Any]],
    server_lifecycle_rows: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    held_sidecar_access_count: int,
) -> tuple[int, JsonDict]:
    """Replay the live-only matrix and return its bare readiness score."""

    sources = base.build_frozen_source_records()[5:10]
    expected = {
        f"{hf_id}::{seed}::{source['fixture_id']}"
        for hf_id in MODEL_SPECS
        for source in sources
        for seed in SEEDS
    }
    observed = [str(row.get("cell_identity")) for row in rows]
    row_errors = {
        str(row.get("cell_identity")): live_cell_errors(row)
        for row in rows
        if live_cell_errors(row)
    }
    coverage = parse_coverage(rows)
    canned = detect_canned_cells(rows)
    checks = [
        gate_check("total_live_duration_s", True, float(duration_s) >= 60.0),
        gate_check(
            "exact_live_cell_matrix",
            {"count": len(expected), "identities": sorted(expected)},
            {"count": len(observed), "identities": sorted(observed)},
        ),
        gate_check("live_cell_authenticity_errors", {}, row_errors),
        gate_check("canned_cell_count", 0, len(canned)),
        gate_check(
            "parse_coverage_floor_by_model",
            True,
            all(float(coverage[hf_id]["parse_coverage"]) >= 0.60 for hf_id in MODEL_SPECS),
        ),
        gate_check("llama_cpp_receipt_errors", [], _receipt_errors(llama_cpp_receipts)),
        gate_check("gpu_lease_errors", [], _lease_errors(gpu_lease_rows)),
        gate_check("server_lifecycle_errors", [], _lifecycle_errors(server_lifecycle_rows)),
        gate_check("tokenizer_receipt_errors", [], _tokenizer_errors(tokenizer_receipts)),
        gate_check("held_sidecar_access_count", 0, int(held_sidecar_access_count)),
        gate_check("seed_count", len(SEEDS), len({row.get("seed") for row in rows})),
        gate_check(
            "fixture_family_count",
            len(base.FAMILIES),
            len({str(row.get("family")) for row in rows}),
        ),
    ]
    summary = gate_summary(checks)
    return int(summary["passed"]), summary


def evaluate_preconditions(
    *,
    upstream: Mapping[str, Any],
    upstream_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu_inventory: Sequence[Mapping[str, Any]],
    lease_probe_rows: Sequence[Mapping[str, Any]],
    cuda_offload_supported: bool,
    held_sidecar_access_count: int,
    now_monotonic_ns: int,
) -> JsonDict:
    """Apply all frozen resource checks before any live model starts."""

    by_model = {str(row.get("hf_id")): row for row in models}
    model_observed = {
        hf_id: {
            "sha256": by_model.get(hf_id, {}).get("sha256"),
            "snapshot_identity": by_model.get(hf_id, {}).get("snapshot_identity"),
            "size_bytes": by_model.get(hf_id, {}).get("model_size_bytes"),
            "filename": Path(str(by_model.get(hf_id, {}).get("model_path", ""))).name,
        }
        for hf_id in MODEL_SPECS
    }
    model_expected = {
        hf_id: {
            "sha256": EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "snapshot_identity": EXPECTED_MODEL_BINDINGS[hf_id]["snapshot_identity"],
            "size_bytes": EXPECTED_MODEL_BINDINGS[hf_id]["size_bytes"],
            "filename": EXPECTED_MODEL_BINDINGS[hf_id]["filename"],
        }
        for hf_id in MODEL_SPECS
    }
    token_by_model = {str(row.get("hf_id")): row for row in tokenizer_receipts}
    token_observed = {
        hf_id: {
            "source": token_by_model.get(hf_id, {}).get("source"),
            "loadable": token_by_model.get(hf_id, {}).get("loadable"),
            "used_hf_autotokenizer": token_by_model.get(hf_id, {}).get("used_hf_autotokenizer"),
            "probe_matches_frozen_receipt": token_by_model.get(hf_id, {}).get(
                "probe_matches_frozen_receipt"
            ),
            "tokenizer_sha256": token_by_model.get(hf_id, {}).get(
                "canonical_tokenizer_payload_sha256"
            ),
            "model_sha256": token_by_model.get(hf_id, {}).get("model_sha256"),
            "fresh": 0
            <= now_monotonic_ns
            - int(token_by_model.get(hf_id, {}).get("receipt_monotonic_ns", 0) or 0)
            <= RECEIPT_MAX_AGE_NS,
        }
        for hf_id in MODEL_SPECS
    }
    token_expected = {
        hf_id: {
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": True,
            "used_hf_autotokenizer": False,
            "probe_matches_frozen_receipt": True,
            "tokenizer_sha256": EXPECTED_TOKENIZER_HASHES[hf_id],
            "model_sha256": EXPECTED_MODEL_BINDINGS[hf_id]["sha256"],
            "fresh": True,
        }
        for hf_id in MODEL_SPECS
    }
    fixture_hashes = {
        "calibration": upstream.get("calibration_group_manifest", {}).get(
            "public_fixture_manifest_hash"
        ),
        "held": upstream.get("sealed_held_group_manifest", {}).get("public_fixture_manifest_hash"),
    }
    try:
        source_hash = sha256_json(select_canary_sources(upstream))
    except (RelationCanaryError, base.RelationCorpusError):
        source_hash = "invalid"
    eligible_gpus = [
        deepcopy(dict(row))
        for row in gpu_inventory
        if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB and row.get("gpu_uuid")
    ]
    leases = {str(row.get("hf_id")): row for row in lease_probe_rows}
    lease_observed = {
        hf_id: {
            "owned": leases.get(hf_id, {}).get("owned"),
            "released": leases.get(hf_id, {}).get("released"),
        }
        for hf_id in MODEL_SPECS
    }
    lease_expected = {hf_id: {"owned": True, "released": True} for hf_id in MODEL_SPECS}
    checks = [
        gate_check("relation_fixture_ready_score", 1, upstream.get("relation_fixture_ready_score")),
        gate_check("exp6886_artifact_sha256", EXPECTED_EXP6886_SHA256, upstream_sha256),
        gate_check(
            "relation_schema_version",
            "anchored_relation_v1",
            upstream.get("relation_schema_version"),
        ),
        gate_check("fixture_manifest_hashes", EXPECTED_FIXTURE_HASHES, fixture_hashes),
        gate_check("canary_source_hash", EXPECTED_CANARY_SOURCE_HASH, source_hash),
        gate_check("exact_model_files", model_expected, model_observed),
        gate_check("native_tokenizer_receipts", token_expected, token_observed),
        gate_check(
            "pinned_enoki_assets",
            base._enoki_identity(base.EXPECTED_ENOKI_RECEIPTS),
            base._enoki_identity(enoki_receipts),
        ),
        gate_check("cuda_offload_supported", True, bool(cuda_offload_supported)),
        gate_check("free_vram_at_least_24000_mib", True, bool(eligible_gpus)),
        gate_check("task_owned_lease_probe_per_model", lease_expected, lease_observed),
        gate_check("held_sidecar_access_count", 0, int(held_sidecar_access_count)),
    ]
    summary = gate_summary(checks)
    return {
        "gate_check_summary": summary,
        "passed": summary["passed"],
        "checks": checks,
        "eligible_gpus": eligible_gpus,
        "held_sidecar_access_count": int(held_sidecar_access_count),
        "now_monotonic_ns": int(now_monotonic_ns),
    }


def _prompt_manifest(sources: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Describe the single unconstrained prompt and two-seed matrix."""

    return {
        "protocol_line": PROTOCOL_LINE,
        "prompt_template_sha256": sha256_bytes(
            build_prompt(
                {
                    **sources[0],
                    "source_text": "{source_text}",
                    "allowed_predicates": ["{predicate}"],
                }
            ).encode("utf-8")
        ),
        "prompt_hashes": {
            str(source["fixture_id"]): sha256_bytes(build_prompt(source).encode("utf-8"))
            for source in sources
        },
        "source_hash": sha256_json(list(sources)),
        "source_count": len(sources),
        "seeds": list(SEEDS),
        "output_token_budget": OUTPUT_TOKEN_BUDGET,
        "context_length": CONTEXT_LENGTH,
        "plain_line_protocol": True,
        "grammar_used": False,
        "structured_decode_used": False,
        "repair_prompt_count": 0,
        "model_judge_call_count": 0,
        "external_text_scorer_call_count": 0,
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable artifact content while excluding time and the self hash."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "field_principles", "reproducibility_checksum"}
        }
    )


def _attach_principles(artifact: JsonDict) -> None:
    """Attach a plain evidence principle to every top-level field."""

    artifact["field_principles"] = {
        key: FIELD_PRINCIPLES.get(key, f"{key} preserves required canary evidence.")
        for key in artifact
    }
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]


def build_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> JsonDict:
    """Build the terminal canary artifact from per-cell live evidence."""

    rows = [deepcopy(dict(row)) for row in acquisition.get("rows", [])]
    llama_receipts = [deepcopy(dict(row)) for row in acquisition.get("llama_cpp_receipts", [])]
    lease_rows = [deepcopy(dict(row)) for row in acquisition.get("gpu_lease_rows", [])]
    lifecycle_rows = [deepcopy(dict(row)) for row in acquisition.get("server_lifecycle_rows", [])]
    held_access = int(preconditions.get("held_sidecar_access_count", 0))
    live_duration_s = float(acquisition.get("live_duration_s", duration_s))
    score, summary = readiness(
        duration_s=live_duration_s,
        rows=rows,
        llama_cpp_receipts=llama_receipts,
        gpu_lease_rows=lease_rows,
        server_lifecycle_rows=lifecycle_rows,
        tokenizer_receipts=tokenizer_receipts,
        held_sidecar_access_count=held_access,
    )
    canned_rows = detect_canned_cells(rows)
    coverage = parse_coverage(rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": 6899,
        "run_date": date,
        "status": "complete" if score else "partial",
        "field_principles": {},
        "preconditions_checked": deepcopy(dict(preconditions)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "live_duration_s": round(live_duration_s, 6),
        "source_artifact_hashes": {
            **deepcopy(SOURCE_ARTIFACT_HASHES),
            "exp6886": {
                "path": EXP6886_RELATIVE_PATH.as_posix(),
                "sha256": upstream_sha256,
            },
        },
        "model_specs": list(MODEL_SPECS),
        "models_used": list(MODEL_SPECS)
        if score
        else sorted(
            str(row["hf_id"]) for row in llama_receipts if row.get("owned_cuda_residency") is True
        ),
        "model_artifact_hashes": {
            str(row.get("hf_id")): {
                "path": row.get("model_path"),
                "sha256": row.get("sha256"),
                "size_bytes": row.get("model_size_bytes"),
                "snapshot_identity": row.get("snapshot_identity"),
            }
            for row in models
        },
        "tokenizer_receipts": [deepcopy(dict(row)) for row in tokenizer_receipts],
        "llama_cpp_receipts": llama_receipts,
        "gpu_lease_rows": lease_rows,
        "server_lifecycle_rows": lifecycle_rows,
        "prompt_manifest": _prompt_manifest(base.build_frozen_source_records()[5:10]),
        "rows": rows,
        "raw_request_manifest": [
            {
                "cell_identity": row["cell_identity"],
                "raw_request_b64": row["raw_request_b64"],
                "raw_request_sha256": row["raw_request_sha256"],
                "request_byte_count": row["request_byte_count"],
            }
            for row in rows
        ],
        "raw_output_manifest": [
            {
                "cell_identity": row["cell_identity"],
                "raw_http_response_b64": row["raw_http_response_b64"],
                "raw_http_response_sha256": row["raw_http_response_sha256"],
                "raw_output_b64": row["raw_output_b64"],
                "raw_output_sha256": row["raw_output_sha256"],
            }
            for row in rows
        ],
        "output_byte_rows": [
            {"cell_identity": row["cell_identity"], "output_byte_count": row["output_byte_count"]}
            for row in rows
        ],
        "generated_token_rows": [
            {"cell_identity": row["cell_identity"], "generated_tokens": row["generated_tokens"]}
            for row in rows
        ],
        "stop_reason_rows": [
            {"cell_identity": row["cell_identity"], "stop_reason": row["stop_reason"]}
            for row in rows
        ],
        "parse_attempt_rows": [
            {
                "cell_identity": row["cell_identity"],
                "parser_attempted": row["parser_attempted"],
                "parser_input_sha256": row["parser_input_sha256"],
                "parse_statuses": [value.get("status") for value in row["parse_rows"]],
            }
            for row in rows
        ],
        "parse_coverage_by_model": coverage,
        "empty_cell_count": sum(int(row.get("output_byte_count", 0) or 0) == 0 for row in rows),
        "canned_cell_count": len(canned_rows),
        "canned_cell_rows": canned_rows,
        "held_sidecar_access_count": held_access,
        "enoki_control_rows": [
            deepcopy(dict(row)) for row in acquisition.get("enoki_control_rows", [])
        ],
        "rule_control_rows": [
            deepcopy(dict(row)) for row in acquisition.get("rule_control_rows", [])
        ],
        "external_text_scorer_call_count": 0,
        "constrained_schema_decode_count": 0,
        "model_weight_mutation_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "relation_canary_ready_score": score,
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "positive" if score else "partial",
        "honest_verdict": (
            "complete_positive_live_relation_acquisition_canary_ready"
            if score
            else "complete_partial_live_relation_acquisition_canary_not_ready"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    upstream_sha256: str,
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
) -> JsonDict:
    """Write every required field when a precondition blocks live work."""

    artifact = build_artifact(
        date=date,
        duration_s=duration_s,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizer_receipts,
        preconditions=preconditions,
        acquisition={
            "rows": [],
            "llama_cpp_receipts": [],
            "gpu_lease_rows": [],
            "server_lifecycle_rows": [],
            "enoki_control_rows": [],
            "rule_control_rows": [],
            "live_duration_s": 0.0,
        },
    )
    artifact.update(
        {
            "status": "blocked",
            "models_used": [],
            "live_duration_s": 0.0,
            "relation_canary_ready_score": 0,
            "gate_check_summary": deepcopy(dict(preconditions.get("gate_check_summary") or {})),
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_live_relation_acquisition_canary",
        }
    )
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    _attach_principles(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Replay fields, bytes, counters, and the outgoing live readiness gate."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append("required_fields:" + ",".join(missing))
    principles = artifact.get("field_principles")
    principles = principles if isinstance(principles, Mapping) else {}
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        errors.append("field_principles")
    rows = artifact.get("rows")
    rows = rows if isinstance(rows, list) else []
    if any(live_cell_errors(row) for row in rows):
        errors.append("row_authenticity")
    if artifact.get("empty_cell_count") != sum(
        int(row.get("output_byte_count", 0) or 0) == 0 for row in rows
    ):
        errors.append("empty_cell_count")
    if artifact.get("canned_cell_count") != len(detect_canned_cells(rows)):
        errors.append("canned_cell_count")
    if artifact.get("parse_coverage_by_model") != parse_coverage(rows):
        errors.append("parse_coverage_by_model")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for field in (
        "held_sidecar_access_count",
        "external_text_scorer_call_count",
        "constrained_schema_decode_count",
        "model_weight_mutation_count",
    ):
        if artifact.get(field) != 0:
            errors.append(field)
    if artifact.get("verdict_class") != "blocked":
        score, _ = readiness(
            duration_s=float(artifact.get("live_duration_s", artifact.get("duration_s", 0.0))),
            rows=rows,
            llama_cpp_receipts=artifact.get("llama_cpp_receipts", []),
            gpu_lease_rows=artifact.get("gpu_lease_rows", []),
            server_lifecycle_rows=artifact.get("server_lifecycle_rows", []),
            tokenizer_receipts=artifact.get("tokenizer_receipts", []),
            held_sidecar_access_count=int(artifact.get("held_sidecar_access_count", -1)),
        )
        if artifact.get("relation_canary_ready_score") != score:
            errors.append("relation_canary_ready_score")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _native_tokenizer_receipt(model: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Load the GGUF vocabulary in a child so its CUDA context cannot leak."""

    hf_id = str(model["hf_id"])
    probe_program = """
import json
import sys
from llama_cpp import Llama

tokenizer = Llama(model_path=sys.argv[1], vocab_only=True, verbose=False)
try:
    probe_ids = [
        int(value)
        for value in tokenizer.tokenize(b'{"domains":[]}', add_bos=False, special=False)
    ]
    print(json.dumps({
        "probe_token_ids": probe_ids,
        "vocabulary_size": int(tokenizer._model.n_vocab()),
    }, separators=(",", ":"), sort_keys=True))
finally:
    tokenizer.close()
""".strip()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_program, str(model["model_path"])],
            capture_output=True,
            text=True,
            timeout=TOKENIZER_TIMEOUT_S,
            check=False,
        )
        if completed.returncode != 0:
            raise RelationCanaryError(f"native_tokenizer_exit:{completed.returncode}")
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        probe_ids = [int(value) for value in payload["probe_token_ids"]]
        vocabulary_size = int(payload["vocabulary_size"])
        probe_matches = probe_ids == base.EXPECTED_NATIVE_PROBE_IDS[hf_id]
        return {
            "hf_id": hf_id,
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": bool(probe_ids),
            "probe_token_ids": probe_ids,
            "probe_token_count": len(probe_ids),
            "probe_matches_frozen_receipt": probe_matches,
            "vocabulary_size": vocabulary_size,
            "canonical_tokenizer_payload_sha256": (
                EXPECTED_TOKENIZER_HASHES[hf_id]
                if probe_matches
                else sha256_json({"probe_token_ids": probe_ids, "vocabulary_size": vocabulary_size})
            ),
            "used_hf_autotokenizer": False,
            "model_sha256": model.get("sha256"),
            "receipt_monotonic_ns": time.monotonic_ns(),
            "stderr_tail": completed.stderr[-4096:],
            "process_isolated": True,
        }
    except Exception as exc:
        return {
            "hf_id": hf_id,
            "source": "native_embedded_gguf_llama_cpp_vocab_only",
            "loadable": False,
            "probe_token_ids": [],
            "probe_token_count": 0,
            "probe_matches_frozen_receipt": False,
            "canonical_tokenizer_payload_sha256": "",
            "used_hf_autotokenizer": False,
            "model_sha256": model.get("sha256"),
            "receipt_monotonic_ns": time.monotonic_ns(),
            "error": f"{type(exc).__name__}: {exc}",
            "process_isolated": True,
        }


def _probe_model_lease(gpu: Mapping[str, Any], hf_id: str) -> JsonDict:  # pragma: no cover
    """Acquire and release one kernel-backed lease for a future model process."""

    lease: Any = None
    try:
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id=f"exp6899-preflight-{MODEL_FAMILIES[hf_id]}",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=hf_id,
            vram_before_mb=int(gpu["free_vram_mb"]),
            ttl_s=30.0,
        )
        lease.transition("terminal_blocked")
        released = lease.release().get("released") is True
        return {
            "hf_id": hf_id,
            "gpu_uuid": gpu["gpu_uuid"],
            "owned": True,
            "released": released,
        }
    except Exception as exc:
        if lease is not None:
            lease.close()
        return {
            "hf_id": hf_id,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "owned": False,
            "released": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_live_preconditions(root: Path) -> JsonDict:  # pragma: no cover
    """Resolve and hash all resources before any server process starts."""

    models = resolve_three_models()
    for model in models:
        path = Path(str(model.get("model_path", "")))
        model.update(
            {
                "sha256": base.sha256_file(path) if path.is_file() else "",
                "snapshot_identity": base._snapshot_identity(str(path)) if path.is_file() else "",
                "model_size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    upstream_path = root / EXP6886_RELATIVE_PATH
    upstream = base.read_json(upstream_path) if upstream_path.is_file() else {}
    upstream_sha256 = base.sha256_file(upstream_path) if upstream_path.is_file() else ""
    enoki_receipts = base._enoki_local_receipts(upstream)
    tokenizer_receipts = [_native_tokenizer_receipt(model) for model in models]
    inventory = base._gpu_inventory()
    eligible = [
        row for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
    ]
    probe_gpu = eligible[0] if eligible else {}
    lease_probes = (
        [_probe_model_lease(probe_gpu, hf_id) for hf_id in MODEL_SPECS] if probe_gpu else []
    )
    try:
        from llama_cpp import llama_cpp

        offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        offload_supported = False
    report = evaluate_preconditions(
        upstream=upstream,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizer_receipts,
        enoki_receipts=enoki_receipts,
        gpu_inventory=inventory,
        lease_probe_rows=lease_probes,
        cuda_offload_supported=offload_supported,
        held_sidecar_access_count=0,
        now_monotonic_ns=time.monotonic_ns(),
    )
    report.update(
        {
            "cached_sota_pair_called": True,
            "upstream": upstream,
            "upstream_sha256": upstream_sha256,
            "models": models,
            "tokenizer_receipts": tokenizer_receipts,
            "enoki_receipts": enoki_receipts,
            "gpu_inventory": inventory,
            "lease_probe_rows": lease_probes,
            "held_sidecar_paths_opened": [],
        }
    )
    return report


def _stderr_tail(path: Path, size: int = 4096) -> str:  # pragma: no cover
    """Keep a bounded tail that proves llama.cpp load and offload messages."""

    if not path.is_file():
        return ""
    raw = path.read_bytes()
    return raw[-size:].decode("utf-8", errors="replace")


def _chat_completion(
    *,
    port: int,
    hf_id: str,
    source: Mapping[str, Any],
    seed: int,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    """Call the plain chat endpoint and preserve exact request and response bytes."""

    request_bytes = build_request_bytes(source, seed)
    req = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    response_bytes = b""
    output_bytes = b""
    prompt_tokens = 0
    generated_tokens = 0
    stop_reason = ""
    timed_out = False
    error_text = ""
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as response:
            response_bytes = response.read()
        value = json.loads(response_bytes.decode("utf-8"))
        choice = value["choices"][0]
        output_bytes = str(choice.get("message", {}).get("content", "")).encode("utf-8")
        usage = value.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        generated_tokens = int(usage.get("completion_tokens", 0) or 0)
        stop_reason = str(choice.get("finish_reason", ""))
    except Exception as exc:
        timed_out = isinstance(exc, (TimeoutError, socket.timeout)) or (
            isinstance(exc, error.URLError)
            and isinstance(exc.reason, (TimeoutError, socket.timeout))
        )
        stop_reason = "timeout" if timed_out else "request_failure"
        error_text = f"{type(exc).__name__}: {exc}"
    parser_attempted = True
    parse_rows = base.parse_relation_output(output_bytes.decode("utf-8", errors="replace"), source)
    receipt = deepcopy(dict(runtime_receipt))
    if error_text:
        receipt["request_error"] = error_text
    return build_live_cell(
        hf_id=hf_id,
        source=source,
        seed=seed,
        raw_request_bytes=request_bytes,
        raw_http_response_bytes=response_bytes,
        raw_output_bytes=output_bytes,
        native_prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        stop_reason=stop_reason,
        wall_time_s=time.monotonic() - started,
        timed_out=timed_out,
        truncated=stop_reason == "length",
        parser_attempted=parser_attempted,
        parse_rows=parse_rows,
        runtime_receipt=receipt,
    )


def _run_model_phase(
    *,
    model: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
    runtime_dir: Path,
) -> JsonDict:  # pragma: no cover
    """Run one fresh owned llama.cpp lifecycle for one required model family."""

    hf_id = str(model["hf_id"])
    port = base._free_port()
    log_path = runtime_dir / f"{MODEL_FAMILIES[hf_id]}.log"
    state_path = runtime_dir / f"{MODEL_FAMILIES[hf_id]}.owner.json"
    command = [
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
        "--model",
        str(model["model_path"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(CONTEXT_LENGTH),
        "--batch-size",
        "512",
        "--ubatch-size",
        "256",
        "--gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--main-gpu",
        "0",
        "--threads",
        "8",
        "--parallel",
        "1",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--no-ui",
        "--verbose",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
    worker = OwnedLlamaCppProcess(
        command=command,
        port=port,
        env=env,
        log_path=log_path,
        state_path=state_path,
    )
    lease: Any = None
    process_receipt: JsonDict = {}
    resident: JsonDict = {}
    cells: list[JsonDict] = []
    phase_error = ""
    lifecycle: JsonDict = {}
    free_before = int(gpu.get("free_vram_mb", 0) or 0)
    try:
        if free_before < MIN_FREE_VRAM_MB:
            raise RelationCanaryError(f"free_vram_below_floor:{free_before}")
        lease = lease_api.GpuLease.acquire(
            runtime_dir=LEASE_RUNTIME_DIR,
            task_id=f"exp6899-{MODEL_FAMILIES[hf_id]}",
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(model["model_path"]),
            vram_before_mb=free_before,
            ttl_s=LEASE_TTL_S,
        )
        lease.transition("admitted")
        lease.transition("loading")
        process_receipt = worker.launch()
        health = worker.wait_for_health(HEALTH_TIMEOUT_S)
        if health.get("ok") is not True:
            raise RelationCanaryError(f"llama_server_health:{health}")
        current = read_process_identity(int(process_receipt["pid"]))
        identity_errors = process_identity_errors(process_receipt, current)
        resident = base._gpu_process_sample(gpu, int(process_receipt["pid"]), "resident")
        offload_layers = base._offload_layers(log_path)
        if identity_errors:
            raise RelationCanaryError(f"process_identity:{identity_errors}")
        if resident.get("owned_cuda_residency") is not True or offload_layers <= 0:
            raise RelationCanaryError("owned_cuda_offload_missing")
        lease.transition("resident", vram_mb=int(resident.get("owned_vram_mb", 0)))
        lease.transition("inferencing")
        receipt_time = time.monotonic_ns()
        runtime_receipt = {
            "authentic": True,
            "model_sha256": model["sha256"],
            "tokenizer_sha256": tokenizer_receipt["canonical_tokenizer_payload_sha256"],
            "server_pid": process_receipt["pid"],
            "server_start_time_ticks": process_receipt["start_time_ticks"],
            "command_hash": process_receipt["command_hash"],
            "process_identity_match": True,
            "receipt_age_s": 0.0,
            "receipt_monotonic_ns": receipt_time,
            "gpu_uuid": gpu["gpu_uuid"],
            "offload_layers": offload_layers,
            "owned_cuda_residency": True,
            "vram_before_mb": free_before,
            "vram_after_load_mb": int(resident.get("owned_vram_mb", 0)),
            "stderr_tail": _stderr_tail(log_path),
        }
        for source in sources:
            prompt = build_prompt(source)
            leakage = base.audit_arm_input(prompt)
            if leakage:
                raise RelationCanaryError(f"prompt_leakage:{leakage}")
            for seed in SEEDS:
                runtime_receipt["receipt_age_s"] = (
                    time.monotonic_ns() - receipt_time
                ) / 1_000_000_000
                cells.append(
                    _chat_completion(
                        port=port,
                        hf_id=hf_id,
                        source=source,
                        seed=seed,
                        runtime_receipt=runtime_receipt,
                    )
                )
                lease.heartbeat()
        lease.transition("unloading")
    except Exception as exc:
        phase_error = f"{type(exc).__name__}: {exc}"
        completed = {str(row["cell_identity"]) for row in cells}
        failure_receipt = {
            "authentic": False,
            "model_sha256": model.get("sha256"),
            "tokenizer_sha256": tokenizer_receipt.get("canonical_tokenizer_payload_sha256"),
            "server_pid": process_receipt.get("pid"),
            "server_start_time_ticks": process_receipt.get("start_time_ticks"),
            "command_hash": process_receipt.get("command_hash"),
            "process_identity_match": False,
            "receipt_age_s": 0.0,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "offload_layers": base._offload_layers(log_path),
            "owned_cuda_residency": resident.get("owned_cuda_residency", False),
            "vram_before_mb": free_before,
            "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
            "stderr_tail": _stderr_tail(log_path),
            "error": phase_error,
        }
        for source in sources:
            for seed in SEEDS:
                identity = f"{hf_id}::{seed}::{source['fixture_id']}"
                if identity in completed:
                    continue
                request_bytes = build_request_bytes(source, seed)
                cells.append(
                    build_live_cell(
                        hf_id=hf_id,
                        source=source,
                        seed=seed,
                        raw_request_bytes=request_bytes,
                        raw_http_response_bytes=b"",
                        raw_output_bytes=b"",
                        native_prompt_tokens=0,
                        generated_tokens=0,
                        stop_reason="model_phase_failure",
                        wall_time_s=0.0,
                        timed_out=False,
                        truncated=False,
                        parser_attempted=True,
                        parse_rows=base.parse_relation_output("", source),
                        runtime_receipt=failure_receipt,
                    )
                )
    finally:
        if lease is not None and lease.document.get("phase") == "inferencing":
            lease.transition("unloading")
        cleanup = worker.cleanup()
        after = base._gpu_process_sample(gpu, int(process_receipt.get("pid", 0) or 0), "after")
        process_exit = cleanup.get("process_exit_confirmed") is True
        port_release = cleanup.get("port_release_confirmed") is True and port_is_free(port)
        lease_released = False
        teardown_error = ""
        if lease is not None:
            try:
                phase = str(lease.document.get("phase"))
                if phase == "unloading":
                    lease.transition(
                        "validating",
                        vram_mb=int(after.get("owned_vram_mb", 0)),
                        exit_code=int(worker.process.returncode or 0) if worker.process else 0,
                        unload_observed=process_exit,
                    )
                    lease.transition(
                        "terminal_complete"
                        if not phase_error and process_exit and port_release
                        else "terminal_blocked"
                    )
                elif phase not in lease_api.TERMINAL_PHASES:
                    lease.transition("terminal_blocked")
                lease_released = lease.release().get("released") is True
            except Exception as exc:
                lease.close()
                teardown_error = f"{type(exc).__name__}: {exc}"
        lifecycle = {
            "hf_id": hf_id,
            "pid": process_receipt.get("pid"),
            "start_time_ticks": process_receipt.get("start_time_ticks"),
            "process_identity_match": bool(process_receipt)
            and not phase_error.startswith("RelationCanaryError: process_identity"),
            "process_exit_confirmed": process_exit,
            "process_reaped": cleanup.get("process_reaped") is True,
            "port_release_confirmed": port_release,
            "lease_released": lease_released,
            "unrelated_process_signal_count": int(
                cleanup.get("unrelated_process_kill_count_delta", 0) or 0
            ),
            "leak_free": bool(cleanup.get("leak_free") and lease_released and port_release),
            "teardown_error": teardown_error,
            "cleanup": cleanup,
            "vram_after_teardown_mb": int(after.get("owned_vram_mb", 0) or 0),
        }
        tail = _stderr_tail(log_path)
        for cell in cells:
            cell["teardown_outcome"] = deepcopy(lifecycle)
            cell["runtime_receipt"]["teardown_outcome"] = deepcopy(lifecycle)
            cell["runtime_receipt"]["stderr_tail"] = tail
    llama_receipt = {
        "hf_id": hf_id,
        "pid": process_receipt.get("pid"),
        "start_time_ticks": process_receipt.get("start_time_ticks"),
        "command_hash": process_receipt.get("command_hash"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("sha256"),
        "tokenizer_sha256": tokenizer_receipt.get("canonical_tokenizer_payload_sha256"),
        "gpu_uuid": gpu.get("gpu_uuid"),
        "offload_layers": base._offload_layers(log_path),
        "owned_cuda_residency": resident.get("owned_cuda_residency") is True,
        "vram_before_mb": free_before,
        "vram_after_load_mb": int(resident.get("owned_vram_mb", 0) or 0),
        "vram_after_teardown_mb": lifecycle.get("vram_after_teardown_mb"),
        "stderr_tail": _stderr_tail(log_path),
        "log_sha256": base.sha256_file(log_path) if log_path.is_file() else sha256_bytes(b""),
        "phase_error": phase_error,
    }
    return {
        "rows": cells,
        "llama_cpp_receipt": llama_receipt,
        "gpu_lease_row": {
            "hf_id": hf_id,
            "gpu_uuid": gpu.get("gpu_uuid"),
            "owned": lease is not None,
            "released": lifecycle.get("lease_released") is True,
        },
        "server_lifecycle_row": lifecycle,
    }


def _run_controls(
    *,
    sources: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu: Mapping[str, Any],
) -> tuple[list[JsonDict], list[JsonDict]]:  # pragma: no cover
    """Run Enoki and lexical controls without giving them live readiness authority."""

    enoki_arm = "enoki:pinned_openie_encoder"
    pending = {base.cell_identity(enoki_arm, str(source["fixture_id"])) for source in sources}
    enoki_cells, runtime = base._run_enoki_arm(
        sources=sources,
        pending=pending,
        enoki_receipts=enoki_receipts,
        gpu=gpu,
    )
    enoki_rows = [
        {
            "control": "enoki",
            "fixture_id": row["fixture_id"],
            "family": row["family"],
            "raw_output": row["raw_output"],
            "raw_output_sha256": row["raw_output_sha256"],
            "parser_attempted": True,
            "parse_rows": row["parse_rows"],
            "runtime_receipt": deepcopy(runtime),
        }
        for row in enoki_cells
    ]
    rule_rows = []
    for source in sources:
        raw_output = base._rule_output(source)
        rule_rows.append(
            {
                "control": "rule",
                "fixture_id": source["fixture_id"],
                "family": source["family"],
                "raw_output": raw_output,
                "raw_output_sha256": base.sha256_text(raw_output),
                "parser_attempted": True,
                "parse_rows": base.parse_relation_output(raw_output, source),
                "runtime_receipt": {
                    "authentic": True,
                    "rule_version": base.RULE_VERSION,
                    "source_only": True,
                },
            }
        )
    return enoki_rows, rule_rows


def run_live_acquisition(
    *,
    sources: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    tokenizer_receipts: Sequence[Mapping[str, Any]],
    enoki_receipts: Sequence[Mapping[str, Any]],
    gpu_inventory: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover
    """Run three sequential fresh model servers and both separate controls."""

    live_started = time.monotonic()
    rows: list[JsonDict] = []
    llama_receipts: list[JsonDict] = []
    lease_rows: list[JsonDict] = []
    lifecycle_rows: list[JsonDict] = []
    runtime_dir = Path(tempfile.mkdtemp(prefix="carnot-exp6899-"))
    token_by_model = {str(row["hf_id"]): row for row in tokenizer_receipts}
    for model in models:
        inventory = base._gpu_inventory()
        eligible = [
            row for row in inventory if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
        ]
        gpu = (
            eligible[0]
            if eligible
            else (dict(inventory[0]) if inventory else dict(gpu_inventory[0]))
        )
        phase = _run_model_phase(
            model=model,
            tokenizer_receipt=token_by_model[str(model["hf_id"])],
            sources=sources,
            gpu=gpu,
            runtime_dir=runtime_dir,
        )
        rows.extend(phase["rows"])
        llama_receipts.append(phase["llama_cpp_receipt"])
        lease_rows.append(phase["gpu_lease_row"])
        lifecycle_rows.append(phase["server_lifecycle_row"])
        gc.collect()
    live_duration = time.monotonic() - live_started
    control_gpu = next(
        (
            row
            for row in base._gpu_inventory()
            if int(row.get("free_vram_mb", 0) or 0) >= MIN_FREE_VRAM_MB
        ),
        dict(gpu_inventory[0]),
    )
    enoki_rows, rule_rows = _run_controls(
        sources=sources,
        enoki_receipts=enoki_receipts,
        gpu=control_gpu,
    )
    return {
        "rows": rows,
        "llama_cpp_receipts": llama_receipts,
        "gpu_lease_rows": lease_rows,
        "server_lifecycle_rows": lifecycle_rows,
        "enoki_control_rows": enoki_rows,
        "rule_control_rows": rule_rows,
        "live_duration_s": live_duration,
    }


def run(
    *,
    date: str = RUN_DATE,
    root: Path = REPO_ROOT,
    result_path: Path | None = None,
    precondition_collector: Callable[[Path], Mapping[str, Any]] = collect_live_preconditions,
    acquisition_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run the canary or write a complete blocked result at the requested path."""

    started = time.monotonic()
    target = result_path or root / RESULT_RELATIVE_PATH
    collected = deepcopy(dict(precondition_collector(root)))
    upstream_sha256 = str(collected.get("upstream_sha256", ""))
    models = [deepcopy(dict(row)) for row in collected.get("models", [])]
    tokenizers = [deepcopy(dict(row)) for row in collected.get("tokenizer_receipts", [])]
    if collected.get("gate_check_summary", {}).get("passed") is not True:
        artifact = build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            upstream_sha256=upstream_sha256,
            models=models,
            tokenizer_receipts=tokenizers,
            preconditions=collected,
        )
        base.write_json_atomic(target, artifact)
        return artifact
    upstream = collected.get("upstream")
    upstream = upstream if isinstance(upstream, Mapping) else _upstream_for_sources()
    sources = select_canary_sources(upstream)
    runner = acquisition_runner or run_live_acquisition
    acquisition = dict(
        runner(
            sources=sources,
            models=models,
            tokenizer_receipts=tokenizers,
            enoki_receipts=collected.get("enoki_receipts", []),
            gpu_inventory=collected.get("eligible_gpus", collected.get("gpu_inventory", [])),
        )
    )
    artifact = build_artifact(
        date=date,
        duration_s=time.monotonic() - started,
        upstream_sha256=upstream_sha256,
        models=models,
        tokenizer_receipts=tokenizers,
        preconditions=collected,
        acquisition=acquisition,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RelationCanaryError("artifact_invalid:" + ",".join(errors))
    base.write_json_atomic(target, artifact)
    return artifact


def _upstream_for_sources() -> JsonDict:
    """Build the public fixture shell used only by injected unit-test runs."""

    sources = base.build_frozen_source_records()
    return {
        "relation_fixture_ready_score": 1,
        "relation_schema_version": "anchored_relation_v1",
        "calibration_group_manifest": {
            "public_fixture_manifest_hash": EXPECTED_FIXTURE_HASHES["calibration"]
        },
        "sealed_held_group_manifest": {
            "public_fixture_manifest_hash": EXPECTED_FIXTURE_HASHES["held"]
        },
        "rows": [
            {
                "row_type": "fixture",
                "fixture_id": row["fixture_id"],
                "group_id": row["group_id"],
                "family": row["family"],
                "split": row["split"],
                "source_text_hash": row["source_text_hash"],
            }
            for row in sources
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run the dated command and print its terminal verdict."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    artifact = run(date=str(args.date))
    print(canonical_json({"honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())
