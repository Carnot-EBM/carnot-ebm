"""Qualify the repaired direct comparator transport with live bounded calls.

The canary uses eight development prompts. It compares grammar-constrained and
unconstrained native requests, then reconstructs every result from raw bytes.
Decision accuracy is descriptive because the canary tests measurement fidelity.

Spec refs: REQ-VERIFY-7277 and SCENARIO-VERIFY-7277-*.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time
from typing import Any, Callable
from urllib import error, request

from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7275_v640_semantic_replay as replay
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root
from scripts.experiment_template import cached_sota_pair


JsonDict = dict[str, Any]

RUN_DATE = "20260913"
MILESTONE = "2026.09.640"
EXPERIMENT_ID = "exp7277-comparator-canary"
TASK_ID = "experiment_7277_v640_comparator_canary"
SCHEMA = "carnot.exp7277.v640_comparator_canary.v1"
RANDOM_SEED = 727_720_260_913
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

PLANNED_UNITS = 8
PLANNED_CALLS = 16
OUTPUT_TOKEN_BUDGET = 128
MODEL_LOAD_CAP_S = 600.0
CANARY_CAP_S = 900.0
ARMS = ("constrained", "unconstrained")

UPSTREAM_ARTIFACT_PATH = Path("results/experiment_7275_v640_semantic_replay.json")
UPSTREAM_CONTRACT_PATH = Path("results/raw/experiment_7275/comparator_contract.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7277_v640_comparator_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7277_v640_comparator_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7277_v640_comparator_canary.py")
RESULT_PATH = Path("results/experiment_7277_v640_comparator_canary.json")
RAW_DIR = Path("results/raw/experiment_7277")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7277_v640_comparator_canary.json")

PINNED_INPUT_HASHES = {
    UPSTREAM_ARTIFACT_PATH: (
        "sha256:10fa70a3f63846fe3a149f53a4a39fcedee6cfaf7163f709d418602c63dd3885"
    ),
    UPSTREAM_CONTRACT_PATH: (
        "sha256:119ebc8154212fd2f5b5373bacb66723f765f299bf932939127263fa9ae7820a"
    ),
}

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "independent_raw_replay",
    "adversarial_verify",
    "verdict_row_consistency",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the result and retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked for terminal evidence; keep unfinished work in separate checkpoints.",
    "run_date": "Use 20260913 and actual UTC start and end times.",
    "field_principles": "Store explanations here; consumer values remain ordinary top-level fields.",
    "preconditions_checked": "Record actual input hashes, authority separation, resource ownership, and failures.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical identities in hashed sidecars.",
    "model_invoked": "Derive from actual calls, including failed or unusable generation.",
    "invocation_counts": "Separate attempted and completed loads and generation from usable answers.",
    "inference_substrate": "Use the recognized literal for actual computation, not an invented task label.",
    "inference_substrate_class": "Use bounded generation, load-only, or the correct no-model class; never pad duration.",
    "inference_mode": "Claim live_gpu only after task-owned CUDA generation starts.",
    "execution_venue": "Host orchestration is host; identify real device execution separately.",
    "duration_s": "Measure monotonic elapsed time and disjoint phase spans.",
    "random_seed": "Freeze independent-unit seeds before observing results.",
    "reproducibility_checksum": "Bind code, configuration, input manifests, and raw evidence.",
    "source_artifact_hashes": "Preserve exact input identity, retirement, and quarantine status.",
    "rows": "Keep each unit, arm, seed, error, abstention, cost, metric, and censoring state.",
    "sample_size_budget": "Record planned, attempted, completed, and censored units and the stopping rule.",
    "acceptance_gate_results": "Each criterion records expected, observed, passed, and principle; separate completeness and value.",
    "gate_check_summary": "For blocked work name upstream, exact field or check, observed value, and expected value.",
    "verifier_is_oracle": "Expose shared verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_; retain the measured finding.",
    "verdict_class": "Use the closed verdict set. Oracle evidence cannot be positive, and only unfinished work is partial.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "comparator_canary_ready_score": "One means fixed transport and replay obligations pass, independent of decision accuracy.",
    "raw_call_manifest": "Retain all sixteen authentic calls, including invalid or truncated replies.",
    "request_grammar_rows": "Bind requested grammar and settings to exact native request bytes.",
    "runner_receipt": "Retain model identity, owned server process, GPU samples, launch settings, and invocation counts.",
    "semantic_accuracy_rows": "Report descriptive development accuracy without making a held-out claim.",
    "timestamps": "Retain actual UTC start and end observations.",
    "phase_spans": "Keep measured phase costs disjoint and monotonic.",
    "model_identity_receipt": "Pin the current model revision, content hash, tokenizer, and embedded chat template.",
    "gpu_receipts": "Bind live calls to one owned RTX 3090 lease and task process.",
    "canary_rows": "Retain the producer reduction before independent replay.",
    "replay_discrepancies": "A byte, request, response, or parser mismatch must remain visible.",
    "readiness_receipt": "Keep transport readiness separate from descriptive correctness.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "request_response_joins": "All scheduled requests need authentic response bytes for independent reduction.",
    "constrained_grammar": "Every constrained request must forward the sealed grammar exactly.",
    "replay_discrepancies": "Independent reconstruction must find no byte or parser drift.",
    "constrained_full_decisions": "Every constrained output must parse with the requested full decision vocabulary.",
    "current_model_identity": "The result must identify the exact current GGUF and embedded template.",
    "owned_cuda_provenance": "Model work must overlap one task-owned RTX 3090 process and lease.",
    "native_runner_receipt": "One sequential native server must retain properties and measured KV headroom.",
    "raw_call_manifest": "All sixteen calls, including failures, must remain addressable by hash.",
    "focused_validation": "Every fixed focused test, lint, replay, and artifact check must pass.",
}

DEVELOPMENT_PROMPTS: tuple[JsonDict, ...] = (
    {
        "unit_id": "dev_supported_founder",
        "source": "Amina founded Cedar Labs in 2019.",
        "claim": "Amina founded Cedar Labs.",
        "expected_decision": "supported",
        "boundary": "ordinary",
    },
    {
        "unit_id": "dev_contradicted_year",
        "source": "The North Bridge opened in 1984.",
        "claim": "The North Bridge opened in 1994.",
        "expected_decision": "contradicted",
        "boundary": "ordinary",
    },
    {
        "unit_id": "dev_unknown_revenue",
        "source": "Orchid Works is based in Lima.",
        "claim": "Orchid Works earned ten million dollars in 2025.",
        "expected_decision": "unknown",
        "boundary": "missing_support",
    },
    {
        "unit_id": "dev_supported_duplicate",
        "source": "Mira met Jo at noon. Later, Mira thanked Jo for the map.",
        "claim": "Mira thanked Jo.",
        "expected_decision": "supported",
        "boundary": "duplicate_mention",
    },
    {
        "unit_id": "dev_contradicted_duplicate",
        "source": "Ravi called Nia twice. Nia answered the second call.",
        "claim": "Nia answered neither call.",
        "expected_decision": "contradicted",
        "boundary": "duplicate_mention",
    },
    {
        "unit_id": "dev_unknown_duplicate",
        "source": "Lee saw Pat near the station. Lee later saw Pat downtown.",
        "claim": "Pat owned the station.",
        "expected_decision": "unknown",
        "boundary": "duplicate_mention",
    },
    {
        "unit_id": "dev_supported_unicode",
        "source": "Zoë mailed the café receipt to Émile.",
        "claim": "Émile received a café receipt from Zoë.",
        "expected_decision": "supported",
        "boundary": "unicode",
    },
    {
        "unit_id": "dev_contradicted_direction",
        "source": "The blue key opens the west door, not the east door.",
        "claim": "The blue key opens the east door.",
        "expected_decision": "contradicted",
        "boundary": "negated_relation",
    },
)
DEVELOPMENT_AUTHORITY = {
    str(row["unit_id"]): str(row["expected_decision"]) for row in DEVELOPMENT_PROMPTS
}


def canonical_json(value: Any) -> str:
    """Use the exact compact JSON spelling used by the native request path."""

    return replay.canonical_json(value)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with the repository's prefixed SHA-256 spelling."""

    return replay.sha256_bytes(value)


def sha256_file(path: Path) -> str:
    """Hash one file without normalizing its evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str,
    field: str,
) -> JsonDict:
    """Retain enough detail to diagnose one failed precondition."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed prerequisite without hiding later checks."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    if failure is None:
        return {
            "failed_check": None,
            "upstream": None,
            "field": None,
            "expected_value": None,
            "observed_value": None,
        }
    return {
        "failed_check": failure.get("check"),
        "upstream": failure.get("upstream"),
        "field": failure.get("field"),
        "expected_value": failure.get("expected_value"),
        "observed_value": failure.get("observed_value"),
    }


def development_prompts() -> list[JsonDict]:
    """Return a fresh copy so sealed development units cannot be mutated globally."""

    return deepcopy(list(DEVELOPMENT_PROMPTS))


def build_schedule(
    contract: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]] | None = None
) -> list[JsonDict]:
    """Freeze the paired 16-call schedule without putting labels in model inputs."""

    units = list(prompts or development_prompts())
    seeds = list(contract.get("draw_seeds") or [])
    if len(seeds) != 2:
        raise ValueError("comparator contract must define two fixed draw seeds")
    template = str(contract.get("prompt_template") or "")
    grammar = str(contract.get("grammar") or "")
    schedule: list[JsonDict] = []
    for unit_index, unit in enumerate(units):
        prompt = template.format(source=unit["source"], claim=unit["claim"])
        for arm_index, arm in enumerate(ARMS):
            active_grammar = grammar if arm == "constrained" else ""
            seed = int(seeds[arm_index])
            schedule.append(
                {
                    "call_order": len(schedule),
                    "call_id": f"exp7277:{unit['unit_id']}:{arm}",
                    "unit_id": str(unit["unit_id"]),
                    "unit_index": unit_index,
                    "arm": arm,
                    "call_type": "direct_judgment",
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "grammar": active_grammar,
                    "grammar_sha256": sha256_bytes(active_grammar.encode("utf-8")),
                    "grammar_requested": arm == "constrained",
                    "output_token_budget": OUTPUT_TOKEN_BUDGET,
                    "seed": seed,
                    "decoding_parameters": {
                        "temperature": 0.0,
                        "top_k": 1,
                        "top_p": 1.0,
                        "seed": seed,
                        "cache_prompt": False,
                    },
                    "retry_budget": 0,
                    "development_only": True,
                    "held_out_eligible": False,
                }
            )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[str]:
    """Reject denominator, pairing, grammar, budget, seed, or authority drift."""

    errors: list[str] = []
    if len(schedule) != PLANNED_CALLS:
        errors.append("call_denominator")
    units = {str(row.get("unit_id")) for row in schedule}
    if len(units) != PLANNED_UNITS:
        errors.append("unit_denominator")
    grammar = str(contract.get("grammar") or "")
    seeds = list(contract.get("draw_seeds") or [])
    for unit_id in sorted(units):
        pair = [row for row in schedule if row.get("unit_id") == unit_id]
        if [row.get("arm") for row in pair] != list(ARMS):
            errors.append(f"{unit_id}:arm_pair")
            continue
        if pair[0].get("prompt") != pair[1].get("prompt"):
            errors.append(f"{unit_id}:prompt_pair")
    for index, row in enumerate(schedule):
        arm = row.get("arm")
        expected_grammar = grammar if arm == "constrained" else ""
        expected_seed = seeds[0 if arm == "constrained" else 1] if len(seeds) == 2 else None
        checks = {
            "call_order": row.get("call_order") == index,
            "grammar": row.get("grammar") == expected_grammar,
            "grammar_requested": row.get("grammar_requested") is (arm == "constrained"),
            "token_budget": row.get("output_token_budget") == OUTPUT_TOKEN_BUDGET,
            "seed": row.get("seed") == expected_seed,
            "decoding_seed": dict(row.get("decoding_parameters") or {}).get("seed")
            == expected_seed,
            "temperature": dict(row.get("decoding_parameters") or {}).get("temperature") == 0.0,
            "retry_budget": row.get("retry_budget") == 0,
            "development_only": row.get("development_only") is True,
            "held_out": row.get("held_out_eligible") is False,
            "authority_hidden": "expected_decision" not in row,
        }
        errors.extend(f"call_{index}:{name}" for name, passed in checks.items() if not passed)
    return list(dict.fromkeys(errors))


def _decode_b64(value: Any) -> bytes:
    """Decode retained bytes strictly so malformed evidence fails closed."""

    if not isinstance(value, str):
        raise ValueError("base64_type")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64_invalid") from exc


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Preserve one native exchange and apply the sealed full-decision parser."""

    expected_payload, expected_request = replay.build_native_request(
        str(sealed["prompt"]),
        str(sealed["grammar"]),
        int(sealed["seed"]),
        OUTPUT_TOKEN_BUDGET,
    )
    raw_request_b64 = str(response.get("raw_request_bytes_b64") or "")
    try:
        actual_request = _decode_b64(raw_request_b64)
    except ValueError:
        actual_request = b""
    raw_completion = str(response.get("raw_completion") or "")
    finish_reason = str(response.get("finish_reason") or "")
    transport_complete = response.get("error") is None and bool(
        response.get("raw_response_bytes_b64")
    )
    reduced = replay._reduce_direct_fixture(
        expected_request,
        actual_request,
        raw_completion,
        finish_reason,
    )
    if not transport_complete:
        reduced = {"accepted": False, "decision": None, "classification": "transport_error"}
    error_value = response.get("error") or (
        None if reduced["accepted"] else reduced["classification"]
    )
    row: JsonDict = {
        "call_order": int(sealed["call_order"]),
        "call_id": str(sealed["call_id"]),
        "unit_id": str(sealed["unit_id"]),
        "arm": str(sealed["arm"]),
        "seed": int(sealed["seed"]),
        "metric": int(reduced["accepted"]),
        "error": error_value,
        "abstention": reduced.get("decision") == "unknown" or not reduced["accepted"],
        "censored": False,
        "terminal_state": "complete" if transport_complete else "failed",
        "transport_complete": transport_complete,
        "parse_valid": reduced["accepted"] is True,
        "usable": reduced["accepted"] is True,
        "decision": reduced.get("decision"),
        "parser_classification": reduced["classification"],
        "raw_completion": raw_completion,
        "raw_completion_sha256": sha256_bytes(raw_completion.encode("utf-8")),
        "raw_request": deepcopy(dict(response.get("raw_request") or {})),
        "actual_parameters": deepcopy(dict(response.get("raw_request") or {})),
        "raw_request_bytes_b64": raw_request_b64,
        "request_bytes_sha256": sha256_bytes(actual_request),
        "expected_request_bytes_sha256": sha256_bytes(expected_request),
        "raw_response": deepcopy(dict(response.get("raw_response") or {})),
        "raw_response_bytes_b64": str(response.get("raw_response_bytes_b64") or ""),
        "response_bytes_sha256": sha256_bytes(
            _decode_b64(str(response.get("raw_response_bytes_b64") or ""))
            if response.get("raw_response_bytes_b64")
            else b""
        ),
        "grammar_sha256": str(sealed["grammar_sha256"]),
        "grammar_requested": sealed["grammar_requested"] is True,
        "output_token_budget": OUTPUT_TOKEN_BUDGET,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "finish_reason": response.get("finish_reason"),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "request_started_at_utc": response.get("started_at_utc"),
        "response_observed_at_utc": response.get("completed_at_utc") or utc_now(),
        "resource": deepcopy(dict(resource)),
        "expected_payload_sha256": sha256_bytes(canonical_json(expected_payload).encode("utf-8")),
    }
    row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
    return row


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:
    """Reconstruct all calls from bytes without issuing another model request."""

    rows: list[JsonDict] = []
    errors: list[str] = []
    if len(schedule) != len(retained_rows):
        errors.append("replay_denominator")
    for index, sealed in enumerate(schedule):
        if index >= len(retained_rows):
            errors.append(f"call_{index}:missing_completion")
            continue
        retained = retained_rows[index]
        try:
            request_bytes = _decode_b64(retained.get("raw_request_bytes_b64"))
            response_bytes = _decode_b64(retained.get("raw_response_bytes_b64"))
            response_body, response_content = replay._response_content(response_bytes)
        except ValueError as exc:
            errors.append(f"call_{index}:{exc}")
            continue
        expected_payload, expected_request = replay.build_native_request(
            str(sealed["prompt"]),
            str(sealed["grammar"]),
            int(sealed["seed"]),
            OUTPUT_TOKEN_BUDGET,
        )
        if request_bytes != expected_request:
            errors.append(f"call_{index}:request_bytes")
        if retained.get("actual_parameters") != expected_payload:
            errors.append(f"call_{index}:actual_parameters")
        if retained.get("raw_response") != response_body:
            errors.append(f"call_{index}:raw_response")
        if retained.get("raw_completion") != response_content:
            errors.append(f"call_{index}:raw_completion")
        reduced = replay._reduce_direct_fixture(
            expected_request,
            request_bytes,
            response_content,
            str(retained.get("finish_reason") or ""),
        )
        if (
            retained.get("parse_valid") is not (reduced["accepted"] is True)
            or retained.get("decision") != reduced.get("decision")
            or retained.get("parser_classification") != reduced["classification"]
        ):
            errors.append(f"call_{index}:parser_reduction")
        parse_error = None if reduced["accepted"] else reduced["classification"]
        rows.append(
            {
                "call_order": index,
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "arm": sealed["arm"],
                "seed": sealed["seed"],
                "metric": int(reduced["accepted"]),
                "error": parse_error,
                "abstention": reduced.get("decision") == "unknown" or not reduced["accepted"],
                "censored": False,
                "request_response_joined": True,
                "request_bytes_sha256": sha256_bytes(request_bytes),
                "response_bytes_sha256": sha256_bytes(response_bytes),
                "parse_valid": reduced["accepted"] is True,
                "usable": reduced["accepted"] is True,
                "decision": reduced.get("decision"),
                "parser_classification": reduced["classification"],
                "finish_reason": retained.get("finish_reason"),
                "prompt_tokens": int(retained.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(retained.get("completion_tokens", 0) or 0),
                "latency_s": float(retained.get("latency_s", 0.0) or 0.0),
            }
        )
    return rows, list(dict.fromkeys(errors))


def request_grammar_rows(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind each requested grammar to the native request that carried it."""

    rows: list[JsonDict] = []
    for sealed, retained in zip(schedule, retained_rows, strict=False):
        expected = str(sealed["grammar"])
        actual = str(dict(retained.get("actual_parameters") or {}).get("grammar") or "")
        constrained = sealed.get("arm") == "constrained"
        rows.append(
            {
                "call_order": sealed["call_order"],
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "arm": sealed["arm"],
                "seed": sealed["seed"],
                "max_tokens": dict(retained.get("actual_parameters") or {}).get("max_tokens"),
                "temperature": dict(retained.get("actual_parameters") or {}).get("temperature"),
                "requested_grammar_sha256": sha256_bytes(expected.encode("utf-8")),
                "actual_grammar_sha256": sha256_bytes(actual.encode("utf-8")),
                "grammar_active": bool(actual),
                "grammar_forwarded_exactly": constrained and bool(expected) and actual == expected,
                "request_bytes_sha256": retained.get("request_bytes_sha256"),
            }
        )
    return rows


def semantic_accuracy_rows(replay_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Grade development rows descriptively without changing transport readiness."""

    return [
        {
            "unit_id": row.get("unit_id"),
            "arm": row.get("arm"),
            "seed": row.get("seed"),
            "expected_decision": DEVELOPMENT_AUTHORITY.get(str(row.get("unit_id"))),
            "observed_decision": row.get("decision"),
            "parse_valid": row.get("parse_valid") is True,
            "correct": row.get("parse_valid") is True
            and row.get("decision") == DEVELOPMENT_AUTHORITY.get(str(row.get("unit_id"))),
            "metric": "development_direct_decision_accuracy",
            "development_only": True,
            "held_out_claim": False,
        }
        for row in replay_rows
    ]


def _identity_complete(identity: Mapping[str, Any]) -> bool:
    """Require the current content-addressed GGUF and embedded template identity."""

    required = (
        "gguf_path",
        "revision",
        "gguf_sha256",
        "embedded_tokenizer_sha256",
        "embedded_chat_template_sha256",
    )
    return bool(
        identity.get("hf_id") == MODEL_ID
        and identity.get("quantization") == QUANTIZATION
        and all(identity.get(field) for field in required)
        and identity.get("embedded_chat_template_present") is True
        and identity.get("auto_tokenizer_used") is False
        and identity.get("runtime") == "native_llama.cpp_server"
    )


def _runner_complete(runner: Mapping[str, Any]) -> bool:
    """Require one sequential server plus process, properties, and KV evidence."""

    identity = dict(runner.get("server_identity") or {})
    headroom = dict(runner.get("kv_headroom") or {})
    return bool(
        runner.get("model_count") == 1
        and runner.get("replica_count") == 1
        and runner.get("runner") == "native_llama.cpp_server"
        and runner.get("dual_gpu_runner_used") is False
        and runner.get("command")
        and runner.get("server_props")
        and identity.get("pid")
        and identity.get("start_time_ticks")
        and headroom.get("measured") is True
        and isinstance(headroom.get("headroom_mb"), (int, float))
        and headroom["headroom_mb"] > 0
        and runner.get("model_revision")
        and runner.get("model_sha256")
    )


def readiness_receipt(
    replay_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    grammar_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
    runner_receipt: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> JsonDict:
    """Compute transport readiness without consulting development correctness."""

    constrained = [row for row in replay_rows if row.get("arm") == "constrained"]
    constrained_grammar = [row for row in grammar_rows if row.get("arm") == "constrained"]
    observed = {
        "request_response_joins": sum(
            row.get("request_response_joined") is True for row in replay_rows
        ),
        "constrained_grammar": sum(
            row.get("grammar_forwarded_exactly") is True for row in constrained_grammar
        ),
        "replay_discrepancies": len(replay_errors),
        "constrained_full_decisions": sum(
            row.get("parse_valid") is True and row.get("decision") in replay.DIRECT_DECISIONS
            for row in constrained
        ),
        "current_model_identity": _identity_complete(model_identity),
        "owned_cuda_provenance": gpu_receipts.get("provenance_ok") is True,
        "native_runner_receipt": _runner_complete(runner_receipt),
        "raw_call_manifest": raw_manifest.get("status") == "complete"
        and raw_manifest.get("raw_call_count") == PLANNED_CALLS,
    }
    expected = {
        "request_response_joins": PLANNED_CALLS,
        "constrained_grammar": PLANNED_UNITS,
        "replay_discrepancies": 0,
        "constrained_full_decisions": PLANNED_UNITS,
        "current_model_identity": True,
        "owned_cuda_provenance": True,
        "native_runner_receipt": True,
        "raw_call_manifest": True,
    }
    criteria = [
        {
            "criterion": name,
            "expected": expected[name],
            "observed": value,
            "passed": value == expected[name],
            "principle": GATE_PRINCIPLES[name],
        }
        for name, value in observed.items()
    ]
    return {
        "criteria": criteria,
        "comparator_canary_ready_score": int(all(row["passed"] for row in criteria)),
        "accuracy_consulted": False,
    }


def _validations_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require every named focused validation once and require each to pass."""

    names = [row.get("name") for row in receipts]
    return sorted(names) == sorted(REQUIRED_VALIDATION_NAMES) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before any fallible prerequisite."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": PLANNED_UNITS,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": PLANNED_UNITS,
            "planned_calls": PLANNED_CALLS,
            "attempted_calls": 0,
            "completed_calls": 0,
            "censored_calls": PLANNED_CALLS,
            "stopping_rule": "attempt each frozen call once; no retry, retuning, or accuracy stop",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_comparator_canary_preconditions_unfinished",
        "verdict_class": "partial",
        "validation_receipts": [],
        "comparator_canary_ready_score": 0,
        "raw_call_manifest": {},
        "request_grammar_rows": [],
        "runner_receipt": {
            "model_count": 1,
            "replica_count": 0,
            "runner": "native_llama.cpp_server",
            "dual_gpu_runner_used": False,
        },
        "semantic_accuracy_rows": [],
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "phase_spans": [],
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "canary_rows": [],
        "replay_discrepancies": [],
        "readiness_receipt": {},
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish an external pre-launch block without inventing model work."""

    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["model_invoked"] = False
    artifact["invocation_counts"] = deepcopy(ZERO_INVOCATION_COUNTS)
    artifact["inference_substrate"] = "blocked_before_qualifying_computation"
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["inference_mode"] = "not_invoked"
    artifact["verdict_class"] = "blocked"
    failure = artifact["gate_check_summary"].get("failed_check") or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_comparator_canary_{failure}"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finalize_measured_artifact(
    artifact: JsonDict,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    grammar_rows: Sequence[Mapping[str, Any]],
    accuracy_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish a measured ready or null canary without using accuracy as a gate."""

    artifact["canary_rows"] = [deepcopy(dict(row)) for row in completion_rows]
    artifact["rows"] = [deepcopy(dict(row)) for row in replay_rows]
    artifact["replay_discrepancies"] = list(replay_errors)
    artifact["request_grammar_rows"] = [deepcopy(dict(row)) for row in grammar_rows]
    artifact["semantic_accuracy_rows"] = [deepcopy(dict(row)) for row in accuracy_rows]
    runner_counts = dict(artifact.get("runner_receipt", {}).get("invocation_counts") or {})
    if runner_counts:
        artifact["invocation_counts"] = runner_counts
    receipt = readiness_receipt(
        replay_rows,
        replay_errors,
        grammar_rows,
        artifact["model_identity_receipt"],
        artifact["gpu_receipts"],
        artifact["runner_receipt"],
        artifact["raw_call_manifest"],
    )
    artifact["readiness_receipt"] = receipt
    artifact["comparator_canary_ready_score"] = receipt["comparator_canary_ready_score"]
    validation_ok = _validations_complete(artifact["validation_receipts"])
    artifact["acceptance_gate_results"] = deepcopy(receipt["criteria"]) + [
        {
            "criterion": "focused_validation",
            "expected": True,
            "observed": validation_ok,
            "passed": validation_ok,
            "principle": GATE_PRINCIPLES["focused_validation"],
        }
    ]
    attempted_units = len({row.get("unit_id") for row in completion_rows})
    completed_units = len(
        {row.get("unit_id") for row in replay_rows if row.get("request_response_joined") is True}
    )
    artifact["sample_size_budget"].update(
        {
            "attempted_units": attempted_units,
            "completed_units": completed_units,
            "censored_units": PLANNED_UNITS - completed_units,
            "attempted_calls": len(completion_rows),
            "completed_calls": sum(
                row.get("request_response_joined") is True for row in replay_rows
            ),
            "censored_calls": PLANNED_CALLS - len(replay_rows),
        }
    )
    counts = dict(artifact.get("invocation_counts") or {})
    artifact["model_invoked"] = int(counts.get("generation_calls_attempted", 0) or 0) > 0
    if artifact["model_invoked"]:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
    elif int(counts.get("model_loads_completed", 0) or 0) > 0:
        artifact["inference_substrate"] = "model_load_no_generation"
        artifact["inference_substrate_class"] = "model_load_no_generation"
        artifact["inference_mode"] = "not_invoked"
    artifact["status"] = "complete"
    ready = artifact["comparator_canary_ready_score"] == 1
    artifact["verdict_class"] = "circular_positive" if ready else "null"
    artifact["honest_verdict"] = (
        "complete_circular_positive_comparator_canary_ready_measurement_only"
        if ready
        else "complete_null_comparator_canary_not_ready_no_retuning"
    )
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal shape, denominators, provenance, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    checks = (
        (value.get("schema") == SCHEMA, "schema"),
        (value.get("experiment_id") == EXPERIMENT_ID, "experiment_id"),
        (value.get("milestone") == MILESTONE, "milestone"),
        (value.get("run_date") == RUN_DATE, "run_date"),
        (value.get("field_principles") == FIELD_PRINCIPLES, "field_principles"),
        (value.get("MODEL_SPECS") == MODEL_SPECS, "MODEL_SPECS"),
        (value.get("random_seed") == RANDOM_SEED, "random_seed"),
        (value.get("execution_venue") == "host", "execution_venue"),
        (value.get("verifier_is_oracle") is True, "verifier_is_oracle"),
        (isinstance(value.get("source_artifact_hashes"), Mapping), "source_artifact_hashes"),
        (isinstance(value.get("validation_receipts"), list), "validation_receipts"),
        (isinstance(value.get("runner_receipt"), Mapping), "runner_receipt"),
        (isinstance(value.get("raw_call_manifest"), Mapping), "raw_call_manifest"),
        (isinstance(value.get("request_grammar_rows"), list), "request_grammar_rows"),
        (isinstance(value.get("semantic_accuracy_rows"), list), "semantic_accuracy_rows"),
    )
    errors.extend(name for passed, name in checks if not passed)
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    counts = value.get("invocation_counts")
    if not isinstance(counts, Mapping) or any(
        field not in counts for field in ZERO_INVOCATION_COUNTS
    ):
        errors.append("invocation_counts")
    status = value.get("status")
    if status == "blocked":
        summary = value.get("gate_check_summary")
        blocked_ok = bool(
            value.get("verdict_class") == "blocked"
            and str(value.get("honest_verdict", "")).startswith("blocked_")
            and value.get("model_invoked") is False
            and value.get("invocation_counts") == ZERO_INVOCATION_COUNTS
            and value.get("inference_substrate_class") == "blocked_no_run"
            and isinstance(summary, Mapping)
            and summary.get("failed_check")
        )
        if not blocked_ok:
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    if status != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    if not isinstance(counts, Mapping):
        return list(dict.fromkeys(errors))
    if (
        value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
    ):
        errors.append("live_inference_substrate")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping):
        errors.append("sample_size_budget")
    else:
        expected_budget = {
            "planned_units": PLANNED_UNITS,
            "attempted_units": PLANNED_UNITS,
            "completed_units": PLANNED_UNITS,
            "censored_units": 0,
            "planned_calls": PLANNED_CALLS,
            "attempted_calls": PLANNED_CALLS,
            "completed_calls": PLANNED_CALLS,
            "censored_calls": 0,
        }
        if any(budget.get(key) != expected for key, expected in expected_budget.items()):
            errors.append("sample_size_budget")
    if len(value.get("rows") or []) != PLANNED_CALLS:
        errors.append("rows")
    if len(value.get("canary_rows") or []) != PLANNED_CALLS:
        errors.append("canary_rows")
    if len(value.get("request_grammar_rows") or []) != PLANNED_CALLS:
        errors.append("request_grammar_rows")
    if len(value.get("semantic_accuracy_rows") or []) != PLANNED_CALLS:
        errors.append("semantic_accuracy_rows")
    receipt = readiness_receipt(
        value.get("rows") or [],
        value.get("replay_discrepancies") or [],
        value.get("request_grammar_rows") or [],
        value.get("model_identity_receipt") or {},
        value.get("gpu_receipts") or {},
        value.get("runner_receipt") or {},
        value.get("raw_call_manifest") or {},
    )
    if value.get("readiness_receipt") != receipt:
        errors.append("readiness_receipt")
    expected_ready = receipt["comparator_canary_ready_score"]
    if value.get("comparator_canary_ready_score") != expected_ready:
        errors.append("comparator_canary_ready_score")
    if expected_ready == 1 and value.get("verdict_class") != "circular_positive":
        errors.append("verdict_class")
    if expected_ready == 0 and value.get("verdict_class") != "null":
        errors.append("verdict_class")
    if value.get("verdict_class") == "positive":
        errors.append("oracle_positive")
    if not _validations_complete(value.get("validation_receipts") or []):
        errors.append("validation_receipts")
    if counts.get("generation_calls_attempted") != PLANNED_CALLS:
        errors.append("invocation_counts")
    if dict(value.get("runner_receipt") or {}).get("invocation_counts") != counts:
        errors.append("runner_invocation_counts")
    return list(dict.fromkeys(errors))


def _manifest_lists_experiment(value: Any, experiment_id: int) -> bool:
    """Find an exact experiment ID without substring matches in unrelated prose."""

    if isinstance(value, Mapping):
        if value.get("experiment_id") == experiment_id:
            return True
        ids = value.get("experiment_ids")
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
            if experiment_id in ids:
                return True
        return any(_manifest_lists_experiment(item, experiment_id) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_manifest_lists_experiment(item, experiment_id) for item in value)
    return False


def _contract_checksum(contract: Mapping[str, Any]) -> str:
    """Rebuild the sealed checksum without including the checksum field itself."""

    value = {key: item for key, item in contract.items() if key != "contract_sha256"}
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def authenticate_inputs(
    root: Path,
    *,
    expected_hashes: Mapping[Path, str] | None = None,
    upstream_validator: Callable[[object], list[str]] = replay.validate_artifact,
    exclusion_manifest: Any | None = None,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate Exp7275, its contract, current code, and exclusion state."""

    checks: list[JsonDict] = []
    wanted = dict(expected_hashes or PINNED_INPUT_HASHES)
    for relative, expected in wanted.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            gate_row(
                "authenticated_input",
                expected,
                observed,
                observed == expected,
                upstream=relative.as_posix(),
                field="sha256",
            )
        )
    upstream_path = root / UPSTREAM_ARTIFACT_PATH
    contract_path = root / UPSTREAM_CONTRACT_PATH
    if not upstream_path.is_file() or not contract_path.is_file():
        return checks, {}
    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(
            gate_row(
                "input_parse",
                "valid_json",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="exp7275-semantic-replay",
                field="json",
            )
        )
        return checks, {}
    terminal_errors = upstream_validator(upstream)
    checks.extend(
        [
            gate_row(
                "upstream_terminal_contract",
                [],
                terminal_errors,
                not terminal_errors,
                upstream="exp7275-semantic-replay",
                field="terminal_schema_and_checksum",
            ),
            gate_row(
                "upstream_status",
                "complete",
                upstream.get("status"),
                upstream.get("status") == "complete",
                upstream="exp7275-semantic-replay",
                field="status",
            ),
            gate_row(
                "semantic_replay_ready",
                1,
                upstream.get("semantic_replay_ready_score"),
                upstream.get("semantic_replay_ready_score") == 1,
                upstream="exp7275-semantic-replay",
                field="semantic_replay_ready_score",
            ),
            gate_row(
                "upstream_quarantine",
                False,
                bool(upstream.get("quarantined") or upstream.get("flagged_adversarial")),
                not bool(upstream.get("quarantined") or upstream.get("flagged_adversarial")),
                upstream="exp7275-semantic-replay",
                field="quarantined_or_flagged_adversarial",
            ),
        ]
    )
    contract_receipt = dict(upstream.get("comparator_contract_path") or {})
    checks.append(
        gate_row(
            "sealed_contract_link",
            {
                "path": UPSTREAM_CONTRACT_PATH.as_posix(),
                "sha256": sha256_file(contract_path),
            },
            {"path": contract_receipt.get("path"), "sha256": contract_receipt.get("sha256")},
            contract_receipt.get("path") == UPSTREAM_CONTRACT_PATH.as_posix()
            and contract_receipt.get("sha256") == sha256_file(contract_path),
            upstream="exp7275-semantic-replay",
            field="comparator_contract_path",
        )
    )
    contract_expected = {
        "schema": "carnot.exp7275.comparator_contract.v1",
        "measurement_kind": "direct_comparator_measurement_repair",
        "decision_values": list(replay.DIRECT_DECISIONS),
        "draw_count": 2,
        "public_only_prompts": True,
        "grammar_forwarding": "exact_requested_bytes",
        "tie_rule": "unknown",
        "finite_id_answer_channel": False,
        "schema_supported_semantic_reprompt": False,
        "mention_changed": False,
    }
    contract_observed = {
        "schema": contract.get("schema"),
        "measurement_kind": contract.get("measurement_kind"),
        "decision_values": contract.get("decision_values"),
        "draw_count": contract.get("draw_count"),
        "public_only_prompts": contract.get("public_only_prompts"),
        "grammar_forwarding": contract.get("grammar_forwarding"),
        "tie_rule": contract.get("tie_rule"),
        "finite_id_answer_channel": contract.get("finite_id_answer_channel"),
        "schema_supported_semantic_reprompt": contract.get("schema_supported_semantic_reprompt"),
        "mention_changed": dict(contract.get("mention_method") or {}).get("changed"),
    }
    contract_ok = bool(
        contract_observed == contract_expected
        and contract.get("grammar") == replay.DIRECT_GRAMMAR
        and contract.get("grammar_sha256") == sha256_bytes(replay.DIRECT_GRAMMAR.encode("utf-8"))
        and contract.get("contract_sha256") == _contract_checksum(contract)
    )
    checks.append(
        gate_row(
            "sealed_comparator_contract",
            contract_expected,
            contract_observed,
            contract_ok,
            upstream=UPSTREAM_CONTRACT_PATH.as_posix(),
            field="contract_fields_and_checksum",
        )
    )
    source_hashes = dict(upstream.get("source_artifact_hashes") or {})
    for relative, alias in (
        (replay.MODULE_PATH, "module"),
        (replay.WRAPPER_PATH, "entrypoint"),
    ):
        path = root / relative
        recorded = source_hashes.get(relative.as_posix())
        if isinstance(recorded, Mapping):
            expected = recorded.get("sha256")
        else:
            expected = source_hashes.get(alias)
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            gate_row(
                "upstream_code_identity",
                expected,
                observed,
                isinstance(expected, str) and observed == expected,
                upstream=relative.as_posix(),
                field="sha256",
            )
        )
    manifest = exclusion_manifest
    if manifest is None:
        exclusion_path = root / EXCLUSION_PATH
        manifest = replay.prior.load_yaml(exclusion_path) if exclusion_path.is_file() else {}
    excluded = any(_manifest_lists_experiment(manifest, value) for value in (7275, 7277))
    checks.append(
        gate_row(
            "retirement_and_quarantine",
            False,
            excluded,
            not excluded,
            upstream=EXCLUSION_PATH.as_posix(),
            field="experiment_id",
        )
    )
    return checks, contract


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live inventory.
    """Hash each source and authenticated input used by this invocation."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        Path("scripts/experiment_template.py"),
        UPSTREAM_ARTIFACT_PATH,
        UPSTREAM_CONTRACT_PATH,
        replay.MODULE_PATH,
        replay.WRAPPER_PATH,
        Path("python/carnot/experiment_7209_v635_span_canary.py"),
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        path.as_posix(): {
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
            "retired": False,
            "quarantined": False,
        }
        for path in paths
    }


def _content_hash(path: Path) -> str | None:  # pragma: no cover - host cache.
    """Read the immutable cache blob name instead of scanning 16 GB of tensors."""

    return live_runtime._content_addressed_hash(path)


def collect_live_preflight(
    root: Path,
    contract: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - host resource boundary.
    """Resolve the cached model, native runtime, output paths, and idle GPU."""

    checks: list[JsonDict] = []
    context: JsonDict = {}
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
    storage = {
        "result_parent_writable": os.access(result_path.parent, os.W_OK),
        "checkpoint_parent_writable": os.access(checkpoint_path.parent, os.W_OK),
        "raw_dir_writable": os.access(raw_dir, os.W_OK),
    }
    checks.append(
        gate_row(
            "output_storage",
            {key: True for key in storage},
            storage,
            all(storage.values()),
            upstream="host_filesystem",
            field="output_paths",
        )
    )
    _progress(3, "model_resolution_start", model=MODEL_ID)
    pair = cached_sota_pair(preferred_quant=QUANTIZATION)
    selected = next((row for row in pair or [] if row.get("hf_id") == MODEL_ID), None)
    model_path = Path(str(selected.get("model_path"))) if selected else None
    model_exists = bool(model_path and model_path.is_file())
    model_hash = _content_hash(model_path) if model_path and model_exists else None
    metadata = live_runtime.read_gguf_metadata(model_path) if model_path and model_exists else {}
    identity = {
        "hf_id": selected.get("hf_id") if selected else None,
        "quantization": QUANTIZATION,
        "gguf_path": str(model_path.resolve()) if model_path and model_exists else None,
        "revision": live_runtime.snapshot_revision(model_path)
        if model_path and model_exists
        else None,
        "gguf_bytes": model_path.stat().st_size if model_path and model_exists else None,
        "gguf_sha256": model_hash,
        "gguf_hash_source": "content_addressed_cache_target" if model_hash else None,
        "embedded_tokenizer_sha256": metadata.get("metadata_summary_sha256"),
        "embedded_chat_template_sha256": metadata.get("chat_template_sha256"),
        "embedded_chat_template_present": metadata.get("chat_template_present") is True,
        "auto_tokenizer_used": False,
        "runtime": "native_llama.cpp_server",
        "resolver": "scripts.experiment_template.cached_sota_pair",
        "resolved_pair_count": len(pair or []),
    }
    model_ok = bool(
        selected
        and model_exists
        and model_path
        and "q4_k_m" in model_path.name.lower()
        and model_hash
        and identity["revision"]
        and identity["embedded_tokenizer_sha256"]
        and identity["embedded_chat_template_present"]
        and identity["embedded_chat_template_sha256"]
        == dict(contract.get("embedded_chat_template") or {}).get("sha256")
    )
    checks.append(
        gate_row(
            "cached_current_model_identity",
            {
                "hf_id": MODEL_ID,
                "quantization": QUANTIZATION,
                "content_addressed": True,
                "embedded_chat_template_matches": True,
            },
            {
                "hf_id": identity["hf_id"],
                "quantization": identity["quantization"],
                "content_addressed": bool(model_hash),
                "embedded_chat_template_matches": identity["embedded_chat_template_sha256"]
                == dict(contract.get("embedded_chat_template") or {}).get("sha256"),
            },
            model_ok,
            upstream="scripts.experiment_template.cached_sota_pair",
            field="model_path_revision_hash_and_template",
        )
    )
    _progress(3, "model_resolution_end", model=MODEL_ID, passed=model_ok)
    tokenizer_loader = live_runtime.fixture.resolve_tokenizer_loader()
    checks.append(
        gate_row(
            "embedded_gguf_tokenizer_transport",
            True,
            tokenizer_loader is not None,
            tokenizer_loader is not None,
            upstream="llama_cpp",
            field="embedded_tokenizer_loader",
        )
    )
    server = live_runtime.resolve_native_llama_server()
    _progress(4, "subprocess_start", operation="native_runner_capabilities", path=str(server))
    runner_rows = live_runtime.shipped_runtime.lease_preflight.collect_runner_capabilities(server)
    _progress(4, "subprocess_end", operation="native_runner_capabilities")
    runner_errors = live_runtime.shipped_runtime.lease_preflight.runner_capability_errors(
        runner_rows
    )
    runner = deepcopy(runner_rows[0]) if runner_rows else {}
    non_thinking = live_runtime._runner_supports(runner, "--reasoning")
    checks.append(
        gate_row(
            "native_cuda_runtime",
            {"errors": [], "grammar": True, "non_thinking": True},
            {
                "errors": runner_errors,
                "grammar": runner.get("grammar_or_json_output") is True,
                "non_thinking": non_thinking,
            },
            not runner_errors and runner.get("grammar_or_json_output") is True and non_thinking,
            upstream="native_llama_server",
            field="cuda_linkage_and_capabilities",
        )
    )
    _progress(5, "subprocess_start", operation="gpu_and_lease_inventory")
    process_rows, query_receipts = (
        live_runtime.shipped_runtime.lease_preflight.collect_gpu_process_rows()
    )
    lease_rows = live_runtime.shipped_runtime.lease_preflight.scan_lease_rows(
        live_runtime.shipped_runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = live_runtime.shipped_runtime.lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    cache_rows = [
        {
            "repository": MODEL_ID,
            "filename": model_path.name if model_path else None,
            "path": str(model_path) if model_path else None,
            "real_path": str(model_path.resolve()) if model_path and model_exists else None,
            "revision": identity["revision"],
            "bytes": identity["gguf_bytes"],
            "sha256": model_hash,
            "hash_source": identity["gguf_hash_source"],
            "weights_opened": False,
            "valid": model_ok,
        }
    ]
    decision = live_runtime.shipped_runtime.lease_preflight.readiness_decision(
        classified, lease_rows, cache_rows, runner_rows
    )
    _progress(5, "subprocess_end", operation="gpu_and_lease_inventory")
    available = list(decision.get("available_gpu_uuids") or [])
    selected_device = next(
        (
            dict(row)
            for row in classified
            if row.get("gpu_uuid") in available and "RTX 3090" in str(row.get("gpu_name"))
        ),
        {},
    )
    model_mb = (
        float(identity["gguf_bytes"]) / (1024.0 * 1024.0) if identity.get("gguf_bytes") else 0.0
    )
    free_mb = float(selected_device.get("gpu_memory_free_mb", 0) or 0)
    preload_headroom = free_mb - model_mb
    resource_ok = bool(
        available
        and selected_device
        and all(row.get("returncode") == 0 for row in query_receipts)
        and preload_headroom > 1024.0
    )
    checks.append(
        gate_row(
            "idle_task_ownable_rtx_3090_with_kv_headroom",
            {
                "minimum_count": 1,
                "no_conflicting_compute": True,
                "preload_headroom_mb_gt": 1024,
            },
            {
                "available_gpu_uuids": available,
                "gpu_name": selected_device.get("gpu_name"),
                "free_vram_mb": free_mb,
                "model_file_mb": model_mb,
                "preload_headroom_mb": preload_headroom,
                "conflicting_processes": decision.get("conflicting_processes", []),
                "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
            },
            resource_ok,
            upstream="live_gpu_and_lease_inventory",
            field="available_gpu_uuids_and_kv_headroom",
        )
    )
    context = {
        "schedule": list(schedule),
        "model_path": model_path,
        "model_identity": identity,
        "tokenizer_loader": tokenizer_loader,
        "server_path": server,
        "runner": runner,
        "runner_rows": runner_rows,
        "non_thinking_supported": non_thinking,
        "process_rows": classified,
        "lease_rows": lease_rows,
        "query_receipts": query_receipts,
        "available_gpu_uuids": available,
        "task_id": TASK_ID,
        "request_cap_s": CANARY_CAP_S,
        "live_window_cap_s": CANARY_CAP_S,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "completion_builder": build_completion_row,
        "preload_kv_headroom": {
            "measured": resource_ok,
            "gpu_uuid": selected_device.get("gpu_uuid"),
            "free_vram_mb": free_mb,
            "model_file_mb": model_mb,
            "headroom_mb": preload_headroom,
        },
    }
    return checks, context


def _fetch_server_props(port: int) -> JsonDict:  # pragma: no cover - live server.
    """Retain the native server property bytes while the owned process is live."""

    _progress(6, "subprocess_start", operation="native_server_props", port=port)
    started = time.monotonic()
    try:
        with request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10.0) as response:
            response_bytes = response.read()
        decoded = json.loads(response_bytes.decode("utf-8"))
        receipt = {
            "available": isinstance(decoded, Mapping),
            "http_status": 200,
            "properties": decoded,
            "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
            "raw_response_sha256": sha256_bytes(response_bytes),
            "duration_s": time.monotonic() - started,
            "error": None,
        }
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        receipt = {
            "available": False,
            "http_status": None,
            "properties": {},
            "raw_response_bytes_b64": "",
            "raw_response_sha256": None,
            "duration_s": time.monotonic() - started,
            "error": f"{type(exc).__name__}:{exc}",
        }
    _progress(6, "subprocess_end", operation="native_server_props", passed=receipt["available"])
    return receipt


def capture_live(
    context: Mapping[str, Any], checkpoint_dir: Path, raw_dir: Path, spans: list[JsonDict]
) -> JsonDict:  # pragma: no cover - live native GPU boundary.
    """Reuse the shipped finite capture and observe properties before teardown."""

    observed: JsonDict = {}
    original_wait = live_runtime._wait_for_health

    def wait_with_properties(supervisor: Any, port: int, timeout_s: float) -> JsonDict:
        health = original_wait(supervisor, port, timeout_s)
        if health.get("ok") is True:
            observed["server_props"] = _fetch_server_props(port)
        return health

    live_runtime._wait_for_health = wait_with_properties
    try:
        capture = live_runtime._live_capture(context, checkpoint_dir, raw_dir, spans)
    finally:
        live_runtime._wait_for_health = original_wait
    snapshots = list(capture.get("gpu_receipts", {}).get("snapshots") or [])
    resident = next(
        (row for row in snapshots if row.get("phase") == "model_resident"),
        {},
    )
    gpu_uuid = str((context.get("available_gpu_uuids") or [""])[0])
    device = next(
        (row for row in resident.get("devices", []) if row.get("uuid") == gpu_uuid),
        {},
    )
    resident_headroom = float(device.get("memory_free_mb", 0) or 0)
    kv_headroom = {
        "measured": bool(device and resident_headroom > 0),
        "measurement_point": "after_model_and_configured_kv_cache_load",
        "gpu_uuid": gpu_uuid,
        "context_tokens": live_runtime.CONTEXT_TOKEN_BUDGET,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "headroom_mb": resident_headroom,
        "preload_estimate": deepcopy(dict(context.get("preload_kv_headroom") or {})),
    }
    identity = deepcopy(dict(capture.get("gpu_receipts", {}).get("server_identity") or {}))
    capture["runner_receipt"].update(
        {
            "server_props": deepcopy(observed.get("server_props") or {}),
            "kv_headroom": kv_headroom,
            "server_identity": identity,
            "model_revision": dict(context["model_identity"])["revision"],
            "model_sha256": dict(context["model_identity"])["gguf_sha256"],
        }
    )
    return capture


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    """Write new raw evidence or require an existing byte-identical document."""

    expected = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing raw evidence differs: {path}")
        return
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def seal_development_inputs(raw_dir: Path, contract: Mapping[str, Any]) -> JsonDict:
    """Seal public prompts and a separate development-only authority sidecar."""

    prompts = development_prompts()
    public = [
        {
            "unit_id": row["unit_id"],
            "source": row["source"],
            "claim": row["claim"],
            "boundary": row["boundary"],
            "development_only": True,
            "held_out_eligible": False,
        }
        for row in prompts
    ]
    authority = [
        {
            "unit_id": row["unit_id"],
            "expected_decision": row["expected_decision"],
            "development_only": True,
            "model_worker_input": False,
        }
        for row in prompts
    ]
    public_path = raw_dir / "development_prompts.json"
    authority_path = raw_dir / "development_authority.json"
    _write_or_match(public_path, {"prompts": public})
    _write_or_match(authority_path, {"authority": authority})
    return {
        "development_prompt_count": len(public),
        "decision_vocabulary_covered": sorted({row["expected_decision"] for row in prompts}),
        "duplicate_boundary_count": sum(row["boundary"] == "duplicate_mention" for row in prompts),
        "held_out_eligible_count": 0,
        "model_worker_authority_read_count": 0,
        "public_path": str(public_path),
        "public_sha256": sha256_file(public_path),
        "authority_path": str(authority_path),
        "authority_sha256": sha256_file(authority_path),
        "contract_sha256": contract.get("contract_sha256"),
    }


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Seal all authentic request and response rows, including failures."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    schedule_value = {"schedule": [deepcopy(dict(row)) for row in schedule]}
    _write_or_match(raw_dir / "schedule.json", schedule_value)
    calls: list[JsonDict] = []
    for index, (sealed, completion) in enumerate(zip(schedule, completion_rows, strict=False)):
        value = {"schedule": deepcopy(dict(sealed)), "completion": deepcopy(dict(completion))}
        path = raw_dir / f"call_{index:02d}.json"
        _write_or_match(path, value)
        calls.append(
            {
                "call_order": index,
                "call_id": sealed["call_id"],
                "arm": sealed["arm"],
                "path": str(path),
                "sha256": sha256_file(path),
                "request_bytes_sha256": completion.get("request_bytes_sha256"),
                "response_bytes_sha256": completion.get("response_bytes_sha256"),
                "terminal_state": completion.get("terminal_state"),
            }
        )
    payload = {
        "schema": "carnot.exp7277.raw_call_manifest.v1",
        "status": "complete" if len(calls) == PLANNED_CALLS else "partial",
        "raw_call_count": len(calls),
        "schedule_sha256": sha256_bytes(canonical_json(list(schedule)).encode("utf-8")),
        "model_identity": deepcopy(dict(model_identity)),
        "authority_path_opened_by_model_worker": False,
        "calls": calls,
    }
    manifest_path = raw_dir / "raw_call_manifest.json"
    _write_or_match(manifest_path, payload)
    return {
        **payload,
        "path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def independent_replay_from_raw(raw_dir: Path) -> tuple[list[JsonDict], list[str]]:
    """Replay task-owned call files without reading the producer artifact."""

    try:
        schedule = list(json.loads((raw_dir / "schedule.json").read_text())["schedule"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return [], [f"schedule:{type(exc).__name__}:{exc}"]
    completions: list[JsonDict] = []
    errors: list[str] = []
    for index, sealed in enumerate(schedule):
        path = raw_dir / f"call_{index:02d}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schedule") != sealed:
                errors.append(f"call_{index}:schedule")
            completions.append(dict(value["completion"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"call_{index}:{type(exc).__name__}:{exc}")
    replayed, replay_errors = independent_replay(schedule, completions)
    return replayed, errors + replay_errors


def validation_commands(root: Path, raw_dir: Path) -> list[tuple[str, list[str]]]:
    """Return only the fixed focused validations allowed by the task."""

    python = str(root / ".venv/bin/python")
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7209_v635_span_canary.py",
        "tests/python/test_experiment_7264_v639_mention_canary.py",
        "tests/python/test_experiment_7265_v639_mention_heldout.py",
        "tests/python/test_experiment_7275_v640_semantic_replay.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    coverage_file = "/tmp/.coverage-exp7277-v640"
    candidate = (root / RAW_CANDIDATE_PATH).as_posix()
    return [
        (
            "focused_pytest",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7277-focused",
                test,
                "-q",
            ],
        ),
        (
            "affected_suites",
            [
                python,
                "-u",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7277-affected",
                *affected,
                "-q",
            ],
        ),
        (
            "scoped_coverage",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--basetemp=/tmp/exp7277-coverage",
                test,
                "-q",
            ],
        ),
        (
            "scoped_coverage_report",
            [
                python,
                "-u",
                "-m",
                "coverage",
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [python, "-u", "-m", "ruff", "check", *changed]),
        ("ruff_format", [python, "-u", "-m", "ruff", "format", "--check", *changed]),
        ("mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        (
            "scoped_spec_coverage",
            [python, "-u", "scripts/check_spec_coverage.py", test, *affected],
        ),
        (
            "independent_raw_replay",
            [
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--replay-raw",
                raw_dir.as_posix(),
            ],
        ),
        ("adversarial_verify", [python, "-u", "scripts/adversarial_verify.py", candidate]),
        (
            "verdict_row_consistency",
            [python, "-u", "scripts/verdict_row_consistency_lint.py", candidate],
        ),
    ]


def _progress(phase: int, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary and long-operation observation."""

    print(f"[exp7277] {canonical_json({'phase': phase, 'event': event, **details})}", flush=True)


def _run_validations(root: Path, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream each focused subprocess and retain exact receipts."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts: list[JsonDict] = []
    commands = validation_commands(root, raw_dir)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(
            10,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(commands),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - fixed local argv only.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        assert process.stdout is not None
        with live_runtime._heartbeat(10, name, lambda: index - 1, len(commands)):
            for line in process.stdout:
                lines.append(line)
                print(f"[exp7277:{name}] {line.rstrip()}", flush=True)
            returncode = process.wait()
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipts.append(
            {
                "name": name,
                "command": shlex.join(command),
                "exit_code": returncode,
                "passed": returncode == 0,
                "timed_out": False,
                "duration_s": time.monotonic() - started,
                "log_path": log_path.relative_to(root).as_posix(),
                "log_sha256": sha256_file(log_path),
            }
        )
        _progress(
            10,
            "subprocess_end",
            operation=name,
            exit_code=returncode,
            completed_units=index,
            total_units=len(commands),
        )
    return receipts


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover
    """Expose pending validation in the raw candidate without inventing success."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7277/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write unfinished current work only to the task checkpoint path."""

    value = deepcopy(dict(artifact))
    if value.get("status") not in {"complete", "blocked"}:
        value["status"] = "partial"
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_comparator_canary_unfinished"
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=True, sort_keys=True)


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    validation_runner: Callable[[Path, Path], list[JsonDict]] = _run_validations,
    capture_runner: Callable[
        [Mapping[str, Any], Path, Path, list[JsonDict]], JsonDict
    ] = capture_live,
) -> JsonDict:  # pragma: no cover - required native entrypoint exercises this path.
    """Authenticate, capture, replay, validate, and atomically publish the canary."""

    repo = root or find_repo_root(start=__file__)
    result_path = repo / RESULT_PATH
    raw_dir = repo / RAW_DIR
    checkpoint_path = repo / CHECKPOINT_PATH
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact = base_artifact(run_date)
    spans: list[JsonDict] = []
    _progress(0, "phase_start", operation="startup_and_output_authentication")
    _checkpoint(checkpoint_path, artifact)
    _progress(0, "phase_end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase = time.monotonic()
    _progress(1, "phase_start", operation="authenticate_exp7275")
    checks, contract = authenticate_inputs(repo)
    spans.append({"phase": "authenticate_exp7275", "duration_s": time.monotonic() - phase})
    _progress(1, "phase_end", failed=sum(row["passed"] is not True for row in checks))
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    if any(row["passed"] is not True for row in checks):
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        _progress(12, "write_start", path=str(result_path))
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(12, "write_end", path=str(result_path))
        return blocked

    phase = time.monotonic()
    _progress(2, "phase_start", operation="seal_development_prompts_and_schedule")
    prompt_receipt = seal_development_inputs(raw_dir, contract)
    schedule = build_schedule(contract)
    problems = schedule_errors(schedule, contract)
    _write_or_match(raw_dir / "schedule.json", {"schedule": schedule})
    schedule_check = gate_row(
        "sealed_development_schedule",
        {
            "units": PLANNED_UNITS,
            "calls": PLANNED_CALLS,
            "errors": [],
            "held_out_eligible": 0,
        },
        {
            "units": prompt_receipt["development_prompt_count"],
            "calls": len(schedule),
            "errors": problems,
            "held_out_eligible": prompt_receipt["held_out_eligible_count"],
        },
        not problems and prompt_receipt["held_out_eligible_count"] == 0,
        upstream="exp7277-development-prompts",
        field="schedule",
    )
    checks.append(schedule_check)
    artifact["preconditions_checked"] = checks
    spans.append({"phase": "seal_development_inputs", "duration_s": time.monotonic() - phase})
    _checkpoint(checkpoint_path, artifact)
    _progress(2, "phase_end", units=PLANNED_UNITS, calls=len(schedule), errors=problems)
    if not schedule_check["passed"]:
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        return blocked

    phase = time.monotonic()
    _progress(3, "phase_start", operation="live_resource_preflight")
    resource_checks, context = collect_live_preflight(
        repo, contract, schedule, result_path, checkpoint_path, raw_dir
    )
    checks.extend(resource_checks)
    artifact["preconditions_checked"] = checks
    spans.append({"phase": "live_resource_preflight", "duration_s": time.monotonic() - phase})
    _progress(3, "phase_end", failed=sum(row["passed"] is not True for row in checks))
    if any(row["passed"] is not True for row in checks):
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        _progress(12, "write_start", path=str(result_path))
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(12, "write_end", path=str(result_path))
        return blocked

    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    _progress(6, "benchmark_start", operation="native_load_and_fixed_16_calls")
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    capture = capture_runner(context, checkpoint_path.parent, raw_dir, spans)
    completions = list(capture.get("rows") or [])
    counts = {
        "model_loads_attempted": 1,
        "model_loads_completed": int(capture.get("model_loaded") is True),
        "generation_calls_attempted": len(completions),
        "generation_calls_completed": sum(
            row.get("transport_complete") is True for row in completions
        ),
        "usable_answers": sum(row.get("usable") is True for row in completions),
    }
    artifact["invocation_counts"] = counts
    artifact["model_invoked"] = capture.get("model_invoked") is True
    artifact["gpu_receipts"] = deepcopy(dict(capture.get("gpu_receipts") or {}))
    artifact["runner_receipt"] = deepcopy(dict(capture.get("runner_receipt") or {}))
    artifact["runner_receipt"]["invocation_counts"] = deepcopy(counts)
    _progress(
        6,
        "benchmark_end",
        operation="native_load_and_fixed_16_calls",
        completed_units=len(completions),
        total_units=PLANNED_CALLS,
        runtime_error=capture.get("runtime_error"),
    )

    phase = time.monotonic()
    _progress(8, "benchmark_start", operation="independent_raw_reduction")
    replay_rows, replay_errors = independent_replay(schedule, completions)
    grammar_rows = request_grammar_rows(schedule, completions)
    accuracy_rows = semantic_accuracy_rows(replay_rows)
    spans.append({"phase": "independent_raw_reduction", "duration_s": time.monotonic() - phase})
    _progress(
        8,
        "benchmark_end",
        operation="independent_raw_reduction",
        completed_units=len(replay_rows),
        discrepancies=len(replay_errors),
    )
    manifest = write_raw_manifest(
        raw_dir, schedule, completions, artifact["model_identity_receipt"]
    )
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"][(RAW_DIR / "raw_call_manifest.json").as_posix()] = {
        "sha256": manifest["manifest_sha256"],
        "retired": False,
        "quarantined": False,
    }
    for row in manifest["calls"]:
        artifact["source_artifact_hashes"][f"raw_call_{row['call_order']:02d}"] = {
            "sha256": row["sha256"],
            "retired": False,
            "quarantined": False,
        }
    artifact["phase_spans"] = spans
    artifact["validation_receipts"] = _pending_receipts()
    candidate = finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        grammar_rows,
        accuracy_rows,
        duration_s=time.monotonic() - started,
    )
    _progress(9, "write_start", path=str(repo / RAW_CANDIDATE_PATH))
    atomic_write_json(repo / RAW_CANDIDATE_PATH, candidate, allow_override=True, sort_keys=True)
    _progress(9, "write_end", path=str(repo / RAW_CANDIDATE_PATH))
    _checkpoint(checkpoint_path, candidate)

    phase = time.monotonic()
    _progress(10, "phase_start", operation="focused_validation")
    receipts = validation_runner(repo, raw_dir)
    spans.append({"phase": "focused_validation", "duration_s": time.monotonic() - phase})
    _progress(
        10,
        "phase_end",
        operation="focused_validation",
        passed=sum(row.get("passed") is True for row in receipts),
        total=len(receipts),
    )
    artifact["validation_receipts"] = receipts
    artifact["phase_spans"] = spans
    terminal = finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        grammar_rows,
        accuracy_rows,
        duration_s=time.monotonic() - started,
    )
    if not _validations_complete(receipts):
        terminal["status"] = "partial"
        terminal["verdict_class"] = "partial"
        terminal["honest_verdict"] = "partial_comparator_canary_validation_failed"
        terminal["reproducibility_checksum"] = artifact_checksum(terminal)
        _checkpoint(checkpoint_path, terminal)
        raise RuntimeError("focused validation failed; terminal artifact not published")
    terminal_errors = validate_artifact(terminal)
    if terminal_errors:
        _checkpoint(checkpoint_path, terminal)
        raise ValueError(f"invalid Exp7277 artifact: {terminal_errors}")
    atomic_write_json(repo / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(12, "write_start", path=str(result_path))
    atomic_write_json(result_path, terminal, allow_override=False, sort_keys=True)
    _progress(12, "write_end", path=str(result_path))
    return terminal


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V640 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the live canary or replay task-owned raw bytes without generation."""

    print("[exp7277] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    if args.replay_raw is not None:
        _progress(1, "benchmark_start", operation="independent_raw_replay")
        rows, errors = independent_replay_from_raw(args.replay_raw)
        _progress(
            1,
            "benchmark_end",
            operation="independent_raw_replay",
            completed_units=len(rows),
            discrepancies=len(errors),
        )
        if errors or len(rows) != PLANNED_CALLS:
            print(canonical_json({"errors": errors, "rows": len(rows)}), flush=True)
            return 1
        return 0
    run_experiment(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
