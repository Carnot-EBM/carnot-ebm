"""Run the V641 live source-reuse canary on the mandated local GGUF.

The module reuses the shipped mention-pointer prompts, compiler, typed executor,
native llama.cpp loader, and GPU lease. It adds only the small development
schedule and reducer needed to test the Exp7291 cache and cost contract.

Spec refs: REQ-VERIFY-7292 and SCENARIO-VERIFY-7292-*.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import time
from typing import Any

import yaml

from carnot import experiment_7237_v637_mention_canary as extraction
from carnot import experiment_7265_v639_mention_heldout as heldout
from carnot import experiment_7277_v640_comparator_canary as shipped_canary
from carnot import experiment_7291_v641_reuse_fixture as fixture
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260914"
MILESTONE = "2026.09.641"
EXPERIMENT_ID = "exp7292-reuse-canary"
TASK_ID = "experiment_7292_v641_reuse_canary"
SCHEMA = "carnot.exp7292.v641_reuse_canary.v1"
RANDOM_SEED = 729_220_260_914
EVALUATION_SEED = 729_230_260_914
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

ARMS = ("warm_prefix_direct", "fresh_verifier", "versioned_reuse_verifier")
PLANNED_GROUPS = 2
PLANNED_UNITS = 8
PLANNED_CALLS = 44
OUTPUT_TOKEN_BUDGET = 128
MODEL_LOAD_CAP_S = 600.0
LIVE_WINDOW_CAP_S = 900.0
REQUEST_CAP_S = 90.0

UPSTREAM_PATH = Path("results/experiment_7291_v641_reuse_fixture.json")
UPSTREAM_MANIFEST_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/manifest.json")
UPSTREAM_ANALYSIS_PATH = Path(
    "results/raw/experiment_7291_v641_reuse_fixture/analysis-contract.json"
)
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7292_v641_reuse_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7292_v641_reuse_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7292_v641_reuse_canary.py")
RESULT_PATH = Path("results/experiment_7292_v641_reuse_canary.json")
RAW_DIR = Path("results/raw/experiment_7292_v641_reuse_canary")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7292_v641_reuse_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7292_v641_reuse_canary")

PINNED_UPSTREAM_SHA256 = "sha256:ecf1888d12bb8eac85b66ccde765d5bf5ca1a5358422b23590585024a629a263"
PINNED_MANIFEST_SHA256 = "sha256:fafc05f40627d55fd27c83c8e5be8fb710d9837f6ba15b3138fc01bbdb4b5a87"
PINNED_ANALYSIS_SHA256 = "sha256:d51e430f657d91dc1c6d37d9a4aba758fb74ff8cea85603384177d812f8f2f18"

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}

REQUIRED_VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "full_python_suite",
    "scoped_coverage",
    "scoped_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "independent_raw_reducer",
    "adversarial_verify",
    "verdict_row_consistency",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit/arm/seed with metric, cost, error, abstention and censoring; no aggregate-only claim.",
    "sample_size_budget": "Planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness/value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked_* verdict names upstream/check, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_ or complete:; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed efficacy gates forbid positive. Only own unfinished work is partial; unchanged external failure is terminal blocked.",
    "validation_receipts": "Command, exit code, elapsed time and log hash; preserve actual failures.",
    "reuse_canary_ready_score": "One only for valid actual-model transport, revision invalidation, comparator and cost accounting.",
    "runtime_model_identity": "Exact mandated GGUF, tokenizer and runtime configuration hashes.",
    "per_call_rows": "Attempted/completed/failed calls, real token work, cache state, phases and outputs.",
    "warm_prefix_receipt": "Actual backend cache capability and observed input-token reuse for each arm.",
    "canary_control_rows": "Fresh/reused semantic agreement, changed-source invalidation and correct comparator evaluation.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "call_accounting": "Every planned call must have one retained terminal row.",
    "raw_replay": "Raw native bytes must rebuild all retained parser outcomes.",
    "actual_model_transport": "The exact model must return every bounded request through the owned CUDA server.",
    "runtime_model_identity": "Model, quantization, tokenizer, and GGUF hashes must be present.",
    "owned_gpu_provenance": "Each generation must overlap the task-owned native process.",
    "comparator_decisions": "Both direct draws must produce a genuine frozen-rule decision for every unit.",
    "source_version_invalidation": "Each group must replace version one before later claims use version two.",
    "fresh_reuse_semantic_parity": "Fresh and reused compiled constraints must serialize identically.",
    "native_cache_evidence": "Each response must report cached tokens or a measured zero.",
    "fair_warm_comparator": "All three arms must use the same enabled native prompt-cache policy.",
}

DIRECT_PROMPT = heldout.DIRECT_PROMPT
DIRECT_GRAMMAR = heldout._direct_grammar()
LIVE_MODEL_CONFIGURATION = {
    "hf_id": MODEL_ID,
    "quantization": QUANTIZATION,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "cache_prompt": True,
    "max_output_tokens": OUTPUT_TOKEN_BUDGET,
    "runtime": "native_llama.cpp_server",
}

canonical_json = fixture.canonical_json
sha256_bytes = fixture.sha256_bytes
sha256_file = fixture.sha256_file
live_runtime = shipped_canary.live_runtime


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence without making local clock readings reproducible."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_bytes(canonical_json(stable).encode())


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str,
    field: str,
) -> JsonDict:
    """Record the exact comparison behind one fail-closed decision."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure while keeping every check in the artifact."""

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


def select_development_groups(public: Mapping[str, Any]) -> list[JsonDict]:
    """Select two public groups and move their fixed revision after claim two."""

    groups = list(public.get("development_groups") or [])
    if len(groups) < PLANNED_GROUPS:
        raise ValueError("development_group_denominator")
    selected: list[JsonDict] = []
    for original in groups[:PLANNED_GROUPS]:
        group = deepcopy(dict(original))
        claims = list(group.get("claims") or [])
        if len(claims) != 8:
            raise ValueError("development_claim_denominator")
        chosen = [deepcopy(dict(claims[index])) for index in (0, 1, 4, 5)]
        for index, claim in enumerate(chosen, start=1):
            claim["chronology_index"] = index
            claim["source_version"] = 1 if index <= 2 else 2
        group["claims"] = chosen
        group["revision_before_claim"] = 3
        selected.append(group)
    if any(
        forbidden in canonical_json(selected)
        for forbidden in ("expected_decision", "gold_span", "construction_label")
    ):
        raise ValueError("authority_field_in_public_groups")
    return selected


def _source_for(group: Mapping[str, Any], version: int) -> JsonDict:
    """Return the one public source document named by a claim."""

    matches = [
        row for row in group.get("source_versions", []) if row.get("source_version") == version
    ]
    if len(matches) != 1:
        raise ValueError("source_version_identity")
    return deepcopy(dict(matches[0]))


def _append_call(
    schedule: list[JsonDict],
    *,
    group: Mapping[str, Any],
    claim: Mapping[str, Any],
    comparison_arm: str,
    call_type: str,
    document: Mapping[str, Any],
    draw: int | None = None,
) -> None:
    """Append one public request with the fixed warm policy and token ceiling."""

    order = len(schedule)
    seed = RANDOM_SEED + order
    if call_type == "direct":
        source = document["source"]
        claim_document = document["claim"]
        prompt = DIRECT_PROMPT.format(source=source["text"], claim=claim_document["text"])
        grammar = deepcopy(DIRECT_GRAMMAR)
        transport_arm = "direct_judge"
        model_input = {"source": source["text"], "claim": claim_document["text"]}
    else:
        prompt = extraction._prompt("mention_pointer", document, call_type)
        grammar = extraction.compile_grammar("mention_pointer", document, call_type)
        transport_arm = "mention_pointer"
        model_input = {"text": document["text"], "mentions": document["mentions"]}
    settings = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": seed,
        "cache_prompt": True,
    }
    identity = {
        "task": TASK_ID,
        "group_id": group["group_id"],
        "unit_id": claim["unit_id"],
        "comparison_arm": comparison_arm,
        "call_type": call_type,
        "draw": draw,
        "seed": seed,
    }
    schedule.append(
        {
            "call_order": order,
            "call_id": sha256_bytes(canonical_json(identity).encode()),
            "unit_id": claim["unit_id"],
            "group_id": group["group_id"],
            "source_id": group["source_id"],
            "source_version": claim["source_version"],
            "arm": transport_arm,
            "comparison_arm": comparison_arm,
            "call_type": call_type,
            "draw": draw,
            "seed": seed,
            "document": deepcopy(dict(document)),
            "model_input": model_input,
            "input_sha256": sha256_bytes(canonical_json(model_input).encode()),
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            **grammar,
            "output_token_budget": OUTPUT_TOKEN_BUDGET,
            "context_token_budget": extraction.CONTEXT_TOKEN_BUDGET,
            "request_timeout_s": REQUEST_CAP_S,
            "decoding_parameters": settings,
            "retry_budget": 0,
            "development_only": True,
            "held_out_eligible": False,
            "warm_cache_policy": "cache_prompt_true",
        }
    )


def build_schedule(groups: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze sixteen direct, sixteen fresh, and twelve reuse calls."""

    if len(groups) != PLANNED_GROUPS:
        raise ValueError("canary_group_denominator")
    schedule: list[JsonDict] = []
    for group in groups:
        claims = list(group.get("claims") or [])
        if len(claims) != 4:
            raise ValueError("canary_claim_denominator")
        for claim in claims:
            source = _source_for(group, int(claim["source_version"]))["document"]
            document = {"source": source, "claim": claim["claim"]}
            for draw in (1, 2):
                _append_call(
                    schedule,
                    group=group,
                    claim=claim,
                    comparison_arm="warm_prefix_direct",
                    call_type="direct",
                    document=document,
                    draw=draw,
                )
        for claim in claims:
            source = _source_for(group, int(claim["source_version"]))["document"]
            _append_call(
                schedule,
                group=group,
                claim=claim,
                comparison_arm="fresh_verifier",
                call_type="source",
                document=source,
            )
            _append_call(
                schedule,
                group=group,
                claim=claim,
                comparison_arm="fresh_verifier",
                call_type="claim",
                document=claim["claim"],
            )
        for version in (1, 2):
            version_claims = [row for row in claims if row.get("source_version") == version]
            if len(version_claims) != 2:
                raise ValueError("revision_claim_denominator")
            _append_call(
                schedule,
                group=group,
                claim=version_claims[0],
                comparison_arm="versioned_reuse_verifier",
                call_type="source",
                document=_source_for(group, version)["document"],
            )
            for claim in version_claims:
                _append_call(
                    schedule,
                    group=group,
                    claim=claim,
                    comparison_arm="versioned_reuse_verifier",
                    call_type="claim",
                    document=claim["claim"],
                )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Rebuild the sealed public schedule and report every changed field."""

    try:
        expected = build_schedule(groups)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != PLANNED_CALLS:
        errors.append("call_denominator")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field, expected_value in wanted.items():
            if observed.get(field) != expected_value:
                errors.append(f"call_{index}:{field}")
        if set(observed) - set(wanted):
            errors.append(f"call_{index}:extra_fields")
    counts = {arm: sum(row.get("comparison_arm") == arm for row in schedule) for arm in ARMS}
    if counts != {
        "warm_prefix_direct": 16,
        "fresh_verifier": 16,
        "versioned_reuse_verifier": 12,
    }:
        errors.append("arm_call_denominators")
    return list(dict.fromkeys(errors))


def _native_cache_receipt(response: Mapping[str, Any]) -> JsonDict:
    """Read llama.cpp cache counters without inferring unreported reuse."""

    raw_response = response.get("raw_response")
    body = dict(raw_response) if isinstance(raw_response, Mapping) else {}
    usage = dict(body.get("usage") or {})
    details = dict(usage.get("prompt_tokens_details") or {})
    timings = dict(body.get("timings") or {})
    cached_present = "cached_tokens" in details
    cache_n_present = "cache_n" in timings
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    cache_n = int(timings.get("cache_n", 0) or 0)
    evidence = cached_present or cache_n_present
    return {
        "cache_policy": "cache_prompt_true",
        "evidence_present": evidence,
        "cached_tokens": cached_tokens,
        "cache_n": cache_n,
        "measured_absence": evidence and cached_tokens == 0 and cache_n == 0,
        "cached_tokens_field_present": cached_present,
        "cache_n_field_present": cache_n_present,
    }


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Keep the shipped parser row and attach cache and comparison identity."""

    row = heldout.build_completion_row(sealed, response, resource)
    row.update(
        {
            "transport_arm": row["arm"],
            "arm": sealed["comparison_arm"],
            "group_id": sealed["group_id"],
            "source_id": sealed["source_id"],
            "source_version": sealed["source_version"],
            "draw": sealed.get("draw"),
            "model_response_error": response.get("error"),
            "native_cache_receipt": _native_cache_receipt(response),
        }
    )
    row["row_sha256"] = heldout.capture._row_hash(row)
    return row


def _decode_b64(value: Any, label: str) -> bytes:
    """Decode retained bytes strictly so corrupt evidence cannot be repaired."""

    if not isinstance(value, str):
        raise ValueError(label)
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(label) from exc


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:
    """Rebuild every row from native bytes and preserve transport failures."""

    replayed: list[JsonDict] = []
    errors: list[str] = []
    if len(schedule) != len(retained_rows):
        errors.append("replay_denominator")
    for index, sealed in enumerate(schedule):
        if index >= len(retained_rows):
            errors.append(f"call_{index}:missing_row")
            continue
        retained = retained_rows[index]
        try:
            request_bytes = _decode_b64(retained.get("raw_request_bytes_b64"), "raw_request_bytes")
            response_value = retained.get("raw_response_bytes_b64")
            response_bytes = (
                _decode_b64(response_value, "raw_response_bytes") if response_value else b""
            )
            raw_response = json.loads(response_bytes) if response_bytes else {}
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            errors.append(f"call_{index}:{exc}")
            continue
        raw_request = retained.get("actual_parameters")
        if request_bytes != canonical_json(raw_request).encode():
            errors.append(f"call_{index}:request_bytes")
        response = {
            "raw_request": deepcopy(dict(raw_request or {})),
            "raw_request_bytes_b64": retained.get("raw_request_bytes_b64", ""),
            "raw_response": raw_response,
            "raw_response_bytes_b64": retained.get("raw_response_bytes_b64", ""),
            "raw_completion": retained.get("raw_completion", ""),
            "prompt_tokens": retained.get("prompt_tokens", 0),
            "completion_tokens": retained.get("completion_tokens", 0),
            "finish_reason": retained.get("finish_reason"),
            "latency_s": retained.get("latency_s", 0.0),
            "error": retained.get("model_response_error"),
            "started_at_utc": retained.get("request_started_at_utc"),
            "completed_at_utc": retained.get("response_observed_at_utc"),
        }
        resource = {
            "server_pid": retained.get("server_pid"),
            "server_pid_start_ticks": retained.get("server_pid_start_ticks"),
            "gpu_uuid": retained.get("gpu_uuid"),
            "lease_id": retained.get("lease_id"),
            "cuda_offload_confirmed": retained.get("cuda_offload_confirmed") is True,
        }
        rebuilt = build_completion_row(sealed, response, resource)
        replayed.append(rebuilt)
        if rebuilt != retained:
            errors.append(f"call_{index}:replay_mismatch")
    return replayed, errors


def _compiled(row: Mapping[str, Any] | None) -> JsonDict:
    """Return one compiled result or an explicit missing-call failure."""

    if not row or not isinstance(row.get("compiled_completion"), Mapping):
        return {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
    return deepcopy(dict(row["compiled_completion"]))


def _decision(
    source_document: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
    claim_row: Mapping[str, Any] | None,
) -> JsonDict:
    """Execute two actual compiled outputs without repairing either response."""

    if not source_row or not claim_row:
        return {"decision": "unknown", "abstention": True, "errors": ["missing_call"]}
    compiled_source = _compiled(source_row)
    compiled_claim = _compiled(claim_row)
    if source_row.get("usable") is not True or claim_row.get("usable") is not True:
        return {"decision": "unknown", "abstention": True, "errors": ["unusable_extraction"]}
    return fixture.pointer._execute_compiled_pair(source_document, compiled_source, compiled_claim)


def _cost(call_rows: Sequence[Mapping[str, Any] | None]) -> JsonDict:
    """Add only observed token, latency, failure, and native cache work."""

    present = [row for row in call_rows if row]
    return {
        "attempted_calls": len(present),
        "completed_calls": sum(row.get("transport_complete") is True for row in present),
        "failed_calls": sum(row.get("transport_complete") is not True for row in present),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0) or 0) for row in present),
        "completion_tokens": sum(int(row.get("completion_tokens", 0) or 0) for row in present),
        "cached_tokens": sum(
            int(dict(row.get("native_cache_receipt") or {}).get("cached_tokens", 0) or 0)
            for row in present
        ),
        "latency_s": sum(float(row.get("latency_s", 0.0) or 0.0) for row in present),
    }


def reduce_canary(
    groups: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce three arms and compare fresh versus reused serialized constraints."""

    by_key = {
        (
            str(row.get("group_id")),
            str(row.get("unit_id")),
            str(row.get("arm")),
            str(row.get("call_type")),
            row.get("draw"),
        ): row
        for row in completion_rows
    }
    output: list[JsonDict] = []
    controls: list[JsonDict] = []
    invalidations = 0
    served_stale = 0
    for group in groups:
        group_id = str(group["group_id"])
        cache = fixture.SourceCompilationCache(enabled=True, capacity=2)
        reuse_sources: dict[int, Mapping[str, Any] | None] = {}
        reuse_compiled: dict[int, JsonDict] = {}
        for version in (1, 2):
            version_claims = [
                row for row in group["claims"] if int(row["source_version"]) == version
            ]
            source_row = by_key.get(
                (
                    group_id,
                    str(version_claims[0]["unit_id"]),
                    "versioned_reuse_verifier",
                    "source",
                    None,
                )
            )
            reuse_sources[version] = source_row
            if source_row and isinstance(source_row.get("parsed_completion"), Mapping):
                provenance = {
                    "source_id": group["source_id"],
                    "source_version": version,
                    "document": _source_for(group, version)["document"],
                    "completion": source_row["parsed_completion"],
                    "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                    "model_configuration": deepcopy(LIVE_MODEL_CONFIGURATION),
                }
                cached = cache.compile_or_get(provenance)
                invalidations += int(cached.get("invalidated_entries", 0) or 0)
                if cached.get("ok") is True:
                    reuse_compiled[version] = deepcopy(dict(cached["compiled_source"]))
        served_stale += cache.served_stale_constraints
        for claim in group["claims"]:
            unit_id = str(claim["unit_id"])
            version = int(claim["source_version"])
            source_document = _source_for(group, version)["document"]
            direct = [
                by_key.get((group_id, unit_id, "warm_prefix_direct", "direct", draw))
                for draw in (1, 2)
            ]
            draws = [
                str(dict(row.get("compiled_completion") or {}).get("decision", "unknown"))
                if row and row.get("usable") is True
                else "unknown"
                for row in direct
            ]
            direct_prediction = fixture.reduce_two_draws(draws[0], draws[1])
            direct_complete = all(row and row.get("transport_complete") is True for row in direct)
            output.append(
                {
                    "group_id": group_id,
                    "unit_id": unit_id,
                    "source_version": version,
                    "arm": "warm_prefix_direct",
                    "seed": claim["seed"],
                    "prediction": direct_prediction,
                    "draws": draws,
                    "metric": int(direct_prediction in {"supported", "contradicted", "unknown"}),
                    "error": None if direct_complete else "direct_call_incomplete",
                    "abstention": direct_prediction == "unknown",
                    "censored": not direct_complete,
                    "cost": _cost(direct),
                }
            )
            fresh_source = by_key.get((group_id, unit_id, "fresh_verifier", "source", None))
            fresh_claim = by_key.get((group_id, unit_id, "fresh_verifier", "claim", None))
            fresh_decision = _decision(source_document, fresh_source, fresh_claim)
            fresh_complete = bool(
                fresh_source
                and fresh_claim
                and fresh_source.get("transport_complete") is True
                and fresh_claim.get("transport_complete") is True
            )
            fresh_source_compiled = _compiled(fresh_source)
            fresh_claim_compiled = _compiled(fresh_claim)
            output.append(
                {
                    "group_id": group_id,
                    "unit_id": unit_id,
                    "source_version": version,
                    "arm": "fresh_verifier",
                    "seed": claim["seed"],
                    "prediction": fresh_decision["decision"],
                    "metric": int(not fresh_decision.get("errors")),
                    "error": None
                    if fresh_complete and not fresh_decision.get("errors")
                    else ";".join(fresh_decision.get("errors") or ["fresh_call_incomplete"]),
                    "abstention": fresh_decision.get("abstention") is True,
                    "censored": not fresh_complete,
                    "cost": _cost([fresh_source, fresh_claim]),
                    "serialized_source_constraints": canonical_json(fresh_source_compiled),
                    "serialized_claim_constraints": canonical_json(fresh_claim_compiled),
                }
            )
            reuse_source = reuse_sources.get(version)
            reuse_claim = by_key.get((group_id, unit_id, "versioned_reuse_verifier", "claim", None))
            reuse_source_view = deepcopy(dict(reuse_source or {}))
            if version in reuse_compiled:
                reuse_source_view["compiled_completion"] = reuse_compiled[version]
            reuse_decision = _decision(source_document, reuse_source_view, reuse_claim)
            reuse_complete = bool(
                reuse_source
                and reuse_claim
                and reuse_source.get("transport_complete") is True
                and reuse_claim.get("transport_complete") is True
            )
            reuse_source_compiled = _compiled(reuse_source_view)
            reuse_claim_compiled = _compiled(reuse_claim)
            parity = bool(
                canonical_json(fresh_source_compiled) == canonical_json(reuse_source_compiled)
                and canonical_json(fresh_claim_compiled) == canonical_json(reuse_claim_compiled)
            )
            errors = list(reuse_decision.get("errors") or [])
            if not parity:
                errors.append("fresh_reuse_serialization_mismatch")
            output.append(
                {
                    "group_id": group_id,
                    "unit_id": unit_id,
                    "source_version": version,
                    "arm": "versioned_reuse_verifier",
                    "seed": claim["seed"],
                    "prediction": reuse_decision["decision"],
                    "metric": int(not errors),
                    "error": None if reuse_complete and not errors else ";".join(errors),
                    "abstention": reuse_decision.get("abstention") is True,
                    "censored": not reuse_complete,
                    "cost": _cost([reuse_source, reuse_claim]),
                    "serialized_source_constraints": canonical_json(reuse_source_compiled),
                    "serialized_claim_constraints": canonical_json(reuse_claim_compiled),
                    "fresh_reuse_serialization_equal": parity,
                }
            )
            controls.append(
                {
                    "group_id": group_id,
                    "unit_id": unit_id,
                    "source_version": version,
                    "fresh_prediction": fresh_decision["decision"],
                    "reuse_prediction": reuse_decision["decision"],
                    "fresh_reuse_serialization_equal": parity,
                    "source_revision_active": version == 2,
                    "stale_constraints_served": 0,
                    "comparator_draws": draws,
                    "comparator_prediction": direct_prediction,
                    "comparator_rule_applied": True,
                }
            )
    return {
        "rows": output,
        "canary_control_rows": controls,
        "replayable_call_count": len(completion_rows),
        "source_version_invalidations": invalidations,
        "served_stale_constraints": served_stale,
        "fresh_reuse_serialization_mismatches": sum(
            row["fresh_reuse_serialization_equal"] is not True for row in controls
        ),
    }


def warm_prefix_receipt(
    schedule: Sequence[Mapping[str, Any]], completion_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Summarize actual native cache counters under one equal warm policy."""

    by_id = {str(row.get("call_id")): row for row in completion_rows}
    arm_rows: list[JsonDict] = []
    policies: set[str] = set()
    evidence_count = 0
    total_cached = 0
    for arm in ARMS:
        sealed_rows = [row for row in schedule if row.get("comparison_arm") == arm]
        observed = [by_id.get(str(row["call_id"])) for row in sealed_rows]
        present = [row for row in observed if row]
        receipts = [dict(row.get("native_cache_receipt") or {}) for row in present]
        policies.update(str(row.get("cache_policy")) for row in receipts)
        arm_evidence = sum(row.get("evidence_present") is True for row in receipts)
        cached = sum(int(row.get("cached_tokens", 0) or 0) for row in receipts)
        evidence_count += arm_evidence
        total_cached += cached
        arm_rows.append(
            {
                "arm": arm,
                "planned_calls": len(sealed_rows),
                "observed_calls": len(present),
                "cache_evidence_calls": arm_evidence,
                "cached_tokens": cached,
                "cache_n": sum(int(row.get("cache_n", 0) or 0) for row in receipts),
                "explicit_measured_absence": bool(
                    len(present) == len(sealed_rows)
                    and arm_evidence == len(present)
                    and cached == 0
                    and all(int(row.get("cache_n", 0) or 0) == 0 for row in receipts)
                ),
            }
        )
    schedule_policy_ok = all(
        dict(row.get("decoding_parameters") or {}).get("cache_prompt") is True
        and row.get("warm_cache_policy") == "cache_prompt_true"
        for row in schedule
    )
    evidence_complete = len(completion_rows) == len(schedule) and evidence_count == len(schedule)
    fair = schedule_policy_ok and policies == {"cache_prompt_true"}
    return {
        "backend": "native_llama.cpp_server",
        "cache_capability_requested": "cache_prompt_true",
        "arms": arm_rows,
        "native_cache_evidence_complete": evidence_complete,
        "fair_warm_comparator": fair,
        "total_cached_tokens": total_cached,
        "explicit_measured_absence": evidence_complete and total_cached == 0,
        "cost_claim_allowed": evidence_complete and fair,
    }


def acceptance_gates(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    reduced: Mapping[str, Any],
    warm: Mapping[str, Any],
    *,
    model_identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply transport readiness gates without consulting development labels."""

    comparative_rows = list(reduced.get("rows") or [])
    direct = [row for row in comparative_rows if row.get("arm") == "warm_prefix_direct"]
    observed = {
        "call_accounting": len(completion_rows) == len(schedule) == PLANNED_CALLS,
        "raw_replay": not replay_errors and len(completion_rows) == PLANNED_CALLS,
        "actual_model_transport": len(completion_rows) == PLANNED_CALLS
        and all(row.get("transport_complete") is True for row in completion_rows),
        "runtime_model_identity": bool(
            model_identity.get("gguf_sha256")
            and model_identity.get("embedded_tokenizer_sha256")
            and model_identity.get("hf_id", MODEL_ID) == MODEL_ID
        ),
        "owned_gpu_provenance": gpu_receipts.get("provenance_ok") is True,
        "comparator_decisions": len(direct) == PLANNED_UNITS
        and all(
            row.get("prediction") in {"supported", "contradicted", "unknown"}
            and len(row.get("draws") or []) == 2
            for row in direct
        ),
        "source_version_invalidation": reduced.get("source_version_invalidations") == PLANNED_GROUPS
        and reduced.get("served_stale_constraints") == 0,
        "fresh_reuse_semantic_parity": reduced.get("fresh_reuse_serialization_mismatches") == 0,
        "native_cache_evidence": warm.get("native_cache_evidence_complete") is True,
        "fair_warm_comparator": warm.get("fair_warm_comparator") is True,
    }
    expected: JsonDict = {
        "call_accounting": PLANNED_CALLS,
        "raw_replay": [],
        "actual_model_transport": PLANNED_CALLS,
        "runtime_model_identity": MODEL_ID,
        "owned_gpu_provenance": True,
        "comparator_decisions": PLANNED_UNITS,
        "source_version_invalidation": PLANNED_GROUPS,
        "fresh_reuse_semantic_parity": 0,
        "native_cache_evidence": PLANNED_CALLS,
        "fair_warm_comparator": "cache_prompt_true_for_all_arms",
    }
    actual: JsonDict = {
        "call_accounting": len(completion_rows),
        "raw_replay": list(replay_errors),
        "actual_model_transport": sum(
            row.get("transport_complete") is True for row in completion_rows
        ),
        "runtime_model_identity": model_identity.get("hf_id", MODEL_ID)
        if model_identity.get("gguf_sha256")
        else None,
        "owned_gpu_provenance": gpu_receipts.get("provenance_ok") is True,
        "comparator_decisions": len(direct),
        "source_version_invalidation": reduced.get("source_version_invalidations"),
        "fresh_reuse_semantic_parity": reduced.get("fresh_reuse_serialization_mismatches"),
        "native_cache_evidence": sum(
            int(row.get("cache_evidence_calls", 0) or 0) for row in warm.get("arms", [])
        ),
        "fair_warm_comparator": warm.get("fair_warm_comparator"),
    }
    return [
        {
            "criterion": name,
            "expected": expected[name],
            "observed": actual[name],
            "passed": passed,
            "principle": GATE_PRINCIPLES[name],
        }
        for name, passed in observed.items()
    ]


def ready_score(gates: Sequence[Mapping[str, Any]]) -> int:
    """Return one only when every preregistered readiness gate passes."""

    return int(
        {str(row.get("criterion")) for row in gates} == set(GATE_PRINCIPLES)
        and all(row.get("passed") is True for row in gates)
    )


def classify_inference(counts: Mapping[str, Any]) -> JsonDict:
    """Derive the substrate only from attempted load and generation events."""

    loads = int(counts.get("model_loads_attempted", 0) or 0)
    generations = int(counts.get("generation_calls_attempted", 0) or 0)
    if generations:
        return {
            "model_invoked": True,
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_bounded_generation",
            "inference_mode": "live_gpu",
        }
    if loads:
        return {
            "model_invoked": True,
            "inference_substrate": "model_load_no_generation",
            "inference_substrate_class": "model_load_no_generation",
            "inference_mode": "not_invoked",
        }
    return {
        "model_invoked": False,
        "inference_substrate": "blocked_before_qualifying_computation",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_invoked",
    }


def authenticate_inputs(
    root: Path,
    *,
    expected_upstream_sha256: str = PINNED_UPSTREAM_SHA256,
) -> tuple[list[JsonDict], JsonDict]:
    """Authenticate Exp7291 and public manifests without opening its labels."""

    paths = {
        "upstream": root / UPSTREAM_PATH,
        "manifest": root / UPSTREAM_MANIFEST_PATH,
        "analysis": root / UPSTREAM_ANALYSIS_PATH,
        "exclusion": root / EXCLUSION_PATH,
    }
    hashes = {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}
    checks = [
        gate_row(
            "upstream_artifact_hash",
            expected_upstream_sha256,
            hashes["upstream"],
            hashes["upstream"] == expected_upstream_sha256,
            upstream=UPSTREAM_PATH.as_posix(),
            field="sha256",
        ),
        gate_row(
            "development_manifest_hash",
            PINNED_MANIFEST_SHA256,
            hashes["manifest"],
            hashes["manifest"] == PINNED_MANIFEST_SHA256,
            upstream=UPSTREAM_MANIFEST_PATH.as_posix(),
            field="sha256",
        ),
        gate_row(
            "analysis_contract_hash",
            PINNED_ANALYSIS_SHA256,
            hashes["analysis"],
            hashes["analysis"] == PINNED_ANALYSIS_SHA256,
            upstream=UPSTREAM_ANALYSIS_PATH.as_posix(),
            field="sha256",
        ),
    ]
    if not all(path.is_file() for path in paths.values()):
        return checks, {}
    try:
        upstream = json.loads(paths["upstream"].read_text(encoding="utf-8"))
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        analysis = json.loads(paths["analysis"].read_text(encoding="utf-8"))
        exclusions = yaml.safe_load(paths["exclusion"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        checks.append(
            gate_row(
                "input_parse",
                "valid_json_and_yaml",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="declared_inputs",
                field="serialization",
            )
        )
        return checks, {}
    upstream_errors = fixture.validate_artifact(upstream, root=root)
    checks.extend(
        [
            gate_row(
                "upstream_terminal_schema",
                [],
                upstream_errors,
                not upstream_errors,
                upstream=EXPERIMENT_ID,
                field="terminal_schema_and_checksum",
            ),
            gate_row(
                "upstream_status",
                "complete",
                upstream.get("status"),
                upstream.get("status") == "complete",
                upstream="exp7291-reuse-fixture",
                field="status",
            ),
            gate_row(
                "reuse_fixture_ready",
                1,
                upstream.get("reuse_fixture_ready_score"),
                upstream.get("reuse_fixture_ready_score") == 1,
                upstream="exp7291-reuse-fixture",
                field="reuse_fixture_ready_score",
            ),
            gate_row(
                "upstream_quarantine",
                False,
                bool(upstream.get("quarantined") or upstream.get("flagged_adversarial")),
                not bool(upstream.get("quarantined") or upstream.get("flagged_adversarial")),
                upstream="exp7291-reuse-fixture",
                field="quarantined_or_flagged_adversarial",
            ),
            gate_row(
                "public_authority_boundary",
                {
                    "authority_fields_present": False,
                    "development_groups": 8,
                    "evaluation_labels_read": False,
                },
                {
                    "authority_fields_present": manifest.get("authority_fields_present"),
                    "development_groups": len(manifest.get("development_groups") or []),
                    "evaluation_labels_read": False,
                },
                manifest.get("authority_fields_present") is False
                and len(manifest.get("development_groups") or []) == 8,
                upstream=UPSTREAM_MANIFEST_PATH.as_posix(),
                field="development_groups_and_authority_boundary",
            ),
            gate_row(
                "frozen_analysis_contract",
                True,
                analysis.get("frozen_before_predictions"),
                analysis.get("frozen_before_predictions") is True,
                upstream=UPSTREAM_ANALYSIS_PATH.as_posix(),
                field="frozen_before_predictions",
            ),
        ]
    )
    excluded = fixture._manifest_lists_experiment(
        exclusions, EXPERIMENT_ID
    ) or fixture._manifest_lists_experiment(exclusions, "exp7291-reuse-fixture")
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
    return checks, {
        "upstream": upstream,
        "development_manifest": manifest,
        "analysis_contract": analysis,
        "authority_path_opened": False,
    }


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live inventory.
    """Hash every current source and declared upstream used by the run."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        UPSTREAM_PATH,
        UPSTREAM_MANIFEST_PATH,
        UPSTREAM_ANALYSIS_PATH,
        fixture.MODULE_PATH,
        extraction.MODULE_PATH,
        heldout.MODULE_PATH,
        shipped_canary.MODULE_PATH,
        Path("python/carnot/experiment_7209_v635_span_canary.py"),
        Path("python/carnot/inference/llama_server_supervisor.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        Path("scripts/check_spec_coverage.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    return {
        path.as_posix(): {
            "sha256": sha256_file(root / path) if (root / path).is_file() else "missing",
            "terminal_class": "input",
            "retired": False,
            "quarantined": False,
        }
        for path in paths
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before fallible host checks."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
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
        "random_seed": {
            "development": RANDOM_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "evaluation_observed": False,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_groups": PLANNED_GROUPS,
            "planned_units": PLANNED_UNITS,
            "planned_calls": PLANNED_CALLS,
            "attempted_calls": 0,
            "completed_calls": 0,
            "failed_calls": 0,
            "censored_calls": PLANNED_CALLS,
            "output_token_ceiling_per_call": OUTPUT_TOKEN_BUDGET,
            "generation_call_ceiling": PLANNED_CALLS,
            "live_window_ceiling_s_including_load": LIVE_WINDOW_CAP_S,
            "stopping_rule": "attempt the frozen 44 calls once; stop at 900 seconds; never retry, repair, or tune",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7292_running_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "reuse_canary_ready_score": 0,
        "runtime_model_identity": {},
        "per_call_rows": [],
        "warm_prefix_receipt": {},
        "canary_control_rows": [],
        "phase_spans": [],
        "runner_receipt": {},
        "gpu_receipts": {},
        "raw_call_manifest": {},
        "replay_discrepancies": [],
        "pilot_limitations": {
            "measurement_readiness_only": True,
            "efficacy_claimed": False,
            "rare_error_safety_certified": False,
            "held_out_evaluation_opened": False,
        },
    }


def finalize_blocked_artifact(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish an external block and preserve any real invocation boundary."""

    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["reuse_canary_ready_score"] = 0
    classification = classify_inference(artifact.get("invocation_counts") or {})
    artifact.update(classification)
    failure = artifact["gate_check_summary"].get("failed_check") or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_exp7292_{failure}"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validations_accounted(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require one terminal receipt for each fixed validation command."""

    return {str(row.get("name")) for row in receipts} == set(REQUIRED_VALIDATION_NAMES) and all(
        row.get("exit_code") is not None for row in receipts
    )


def _validations_pass(receipts: Sequence[Mapping[str, Any]], *, include_full: bool) -> bool:
    """Require each scoped command and optionally the repository-wide suite."""

    required = set(REQUIRED_VALIDATION_NAMES)
    if not include_full:
        required.remove("full_python_suite")
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code") == 0
        for name in required
    )


def finalize_measured_artifact(
    artifact: JsonDict,
    completion_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    reduced: Mapping[str, Any],
    warm: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish one measurement-readiness result without an efficacy claim."""

    artifact["per_call_rows"] = deepcopy(list(completion_rows))
    artifact["rows"] = deepcopy(list(reduced.get("rows") or []))
    artifact["canary_control_rows"] = deepcopy(list(reduced.get("canary_control_rows") or []))
    artifact["warm_prefix_receipt"] = deepcopy(dict(warm))
    artifact["replay_discrepancies"] = list(replay_errors)
    artifact["acceptance_gate_results"] = deepcopy(list(gates))
    artifact["reuse_canary_ready_score"] = ready_score(gates)
    counts = dict(artifact.get("invocation_counts") or ZERO_INVOCATION_COUNTS)
    artifact.update(classify_inference(counts))
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": int(counts.get("generation_calls_attempted", 0) or 0),
            "completed_calls": int(counts.get("generation_calls_completed", 0) or 0),
            "failed_calls": int(counts.get("generation_calls_failed", 0) or 0),
            "censored_calls": PLANNED_CALLS - len(completion_rows),
            "completed_units": len(
                {row.get("unit_id") for row in replay_rows if row.get("transport_complete") is True}
            ),
        }
    )
    artifact["status"] = "complete"
    scoped_ok = _validations_pass(artifact.get("validation_receipts") or [], include_full=False)
    full_ok = _validations_pass(artifact.get("validation_receipts") or [], include_full=True)
    if not scoped_ok and _validations_accounted(artifact.get("validation_receipts") or []):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_reuse_canary_scoped_validation_failed"
    elif (
        scoped_ok
        and not full_ok
        and _validations_accounted(artifact.get("validation_receipts") or [])
    ):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = (
            "complete_disqualified_reuse_canary_ready_but_repository_full_suite_failed"
        )
    elif artifact["reuse_canary_ready_score"] == 1:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = (
            "complete_circular_positive_reuse_canary_ready_measurement_only"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_null_reuse_canary_not_ready_no_retuning"
    artifact["gate_check_summary"] = None
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object, *, require_validations: bool = True) -> list[str]:
    """Cold-check terminal identity, event classes, denominators, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "execution_venue": "host",
        "verifier_is_oracle": True,
    }
    errors.extend(field for field, wanted in expected.items() if value.get(field) != wanted)
    counts = value.get("invocation_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(ZERO_INVOCATION_COUNTS)
        or any(int(item or 0) < 0 for item in counts.values())
    ):
        errors.append("invocation_counts")
    if isinstance(counts, Mapping):
        classification = classify_inference(counts)
        if any(value.get(field) != observed for field, observed in classification.items()):
            errors.append("inference_classification")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or not str(value.get("honest_verdict", "")).startswith("blocked_")
            or value.get("reuse_canary_ready_score") != 0
            or not isinstance(summary, Mapping)
            or not summary.get("failed_check")
        ):
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list) or value.get("reuse_canary_ready_score") != ready_score(gates):
        errors.append("reuse_canary_ready_score")
    if len(value.get("per_call_rows") or []) > PLANNED_CALLS:
        errors.append("per_call_rows")
    if len(value.get("rows") or []) != PLANNED_UNITS * len(ARMS):
        errors.append("rows")
    if len(value.get("canary_control_rows") or []) != PLANNED_UNITS:
        errors.append("canary_control_rows")
    allowed = {"circular_positive", "null", "disqualified"}
    if value.get("verdict_class") not in allowed or value.get("verdict_class") == "positive":
        errors.append("verdict_class")
    if require_validations and not _validations_accounted(value.get("validation_receipts") or []):
        errors.append("validation_receipts")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and slow-call boundary for the outer monitor."""

    print(
        f"[exp7292] {canonical_json({'phase': phase, 'event': event, **details})}",
        flush=True,
    )


def _write_or_match(path: Path, value: Any) -> None:  # pragma: no cover
    """Write immutable raw evidence once or require byte-identical resume data."""

    expected = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing raw evidence differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def _checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write unfinished state only to the declared checkpoint path."""

    value = deepcopy(dict(artifact))
    if value.get("status") not in {"complete", "blocked"}:
        value["status"] = "partial"
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_exp7292_unfinished_checkpoint_only"
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=True, sort_keys=True)


def _transport_contract() -> JsonDict:  # pragma: no cover - local cache boundary.
    """Bind the preflight template check to the exact cached GGUF metadata."""

    _progress(3, "model_resolution_start", model=MODEL_ID)
    pair = shipped_canary.cached_sota_pair(preferred_quant=QUANTIZATION)
    selected = next((row for row in pair or [] if row.get("hf_id") == MODEL_ID), None)
    path = Path(str(selected.get("model_path"))) if selected else None
    metadata = live_runtime.read_gguf_metadata(path) if path and path.is_file() else {}
    template_hash = metadata.get("chat_template_sha256")
    _progress(3, "model_resolution_end", model=MODEL_ID, resolved=bool(template_hash))
    return {"embedded_chat_template": {"sha256": template_hash}}


def collect_live_preflight(
    root: Path,
    schedule: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - host resource boundary.
    """Adapt the shipped exact-model, native-runtime, and lease preflight."""

    checks, context = shipped_canary.collect_live_preflight(
        root,
        _transport_contract(),
        schedule,
        result_path,
        checkpoint_path,
        raw_dir,
    )
    context.update(
        {
            "schedule": list(schedule),
            "task_id": TASK_ID,
            "request_cap_s": REQUEST_CAP_S,
            "live_window_cap_s": LIVE_WINDOW_CAP_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
            "completion_builder": build_completion_row,
        }
    )
    return checks, context


def capture_live(
    context: Mapping[str, Any], checkpoint_dir: Path, raw_dir: Path, spans: list[JsonDict]
) -> JsonDict:  # pragma: no cover - owned native GPU boundary.
    """Run the fixed schedule through the shipped loader and lease owner."""

    return shipped_canary.capture_live(context, checkpoint_dir, raw_dir, spans)


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - durable evidence boundary.
    """Index every authentic call file and its exact evidence hash."""

    calls = []
    for index, row in enumerate(completion_rows):
        path = raw_dir / f"call_{index:02d}.json"
        calls.append(
            {
                "call_order": index,
                "call_id": row.get("call_id"),
                "path": path.as_posix(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "terminal_state": row.get("terminal_state"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "cached_tokens": dict(row.get("native_cache_receipt") or {}).get("cached_tokens"),
            }
        )
    manifest = {
        "schema": "carnot.exp7292.raw_call_manifest.v1",
        "status": "complete" if len(calls) == PLANNED_CALLS else "complete_with_censoring",
        "planned_calls": PLANNED_CALLS,
        "retained_calls": len(calls),
        "schedule_sha256": sha256_bytes(canonical_json(list(schedule)).encode()),
        "model_identity": deepcopy(dict(model_identity)),
        "authority_path_opened_by_model_worker": False,
        "evaluation_labels_read": False,
        "calls": calls,
    }
    path = raw_dir / "raw-call-manifest.json"
    _write_or_match(path, manifest)
    return {**manifest, "path": path.as_posix(), "sha256": sha256_file(path)}


def independent_replay_from_raw(raw_dir: Path) -> tuple[JsonDict, list[str]]:
    """Rebuild calls and comparative rows from task-owned raw files only."""

    errors: list[str] = []
    try:
        schedule = json.loads((raw_dir / "schedule.json").read_text(encoding="utf-8"))["schedule"]
        groups = json.loads((raw_dir / "development-manifest.json").read_text(encoding="utf-8"))[
            "groups"
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {}, [f"manifest:{type(exc).__name__}:{exc}"]
    retained = []
    for index in range(len(schedule)):
        try:
            value = json.loads((raw_dir / f"call_{index:02d}.json").read_text(encoding="utf-8"))
            if value.get("schedule") != schedule[index]:
                errors.append(f"call_{index}:schedule")
            retained.append(value["completion"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"call_{index}:{type(exc).__name__}:{exc}")
    replayed, replay_errors = independent_replay(schedule, retained)
    reduced = reduce_canary(groups, schedule, replayed)
    return reduced, errors + replay_errors


def validation_commands(
    root: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover - fixed local commands.
    """Return the focused, full-suite, coverage, lint, replay, and artifact checks."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7277_v640_comparator_canary.py",
        "tests/python/test_experiment_7278_v640_source_measurement.py",
        "tests/python/test_experiment_7291_v641_reuse_fixture.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    coverage_file = "/tmp/.coverage-exp7292-v641"
    candidate = (root / RAW_CANDIDATE_PATH).as_posix()
    common = ["-o", "addopts=", "-n", "0"]
    return [
        (
            "focused_pytest",
            [pytest, *common, "--basetemp=/tmp/exp7292-focused", test, "-q"],
        ),
        (
            "affected_suites",
            [pytest, *common, "--basetemp=/tmp/exp7292-affected", *affected, "-q"],
        ),
        (
            "full_python_suite",
            [
                pytest,
                *common,
                "--basetemp=/tmp/exp7292-full-python",
                "tests/python",
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
                *common,
                "--basetemp=/tmp/exp7292-coverage",
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
        ("changed_module_mypy", [python, "-u", "-m", "mypy", MODULE_PATH.as_posix()]),
        ("scoped_spec_coverage", [python, "-u", "scripts/check_spec_coverage.py", test]),
        (
            "independent_raw_reducer",
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


def _run_validations(root: Path, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream each subprocess and retain its exit code, time, and log hash."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    commands = validation_commands(root, raw_dir)
    receipts: list[JsonDict] = []
    for index, (name, command) in enumerate(commands, start=1):
        _progress(
            10,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(commands),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - fixed repository-local argv.
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
                print(f"[exp7292:{name}] {line.rstrip()}", flush=True)
            exit_code = process.wait()
        log_path = validation_dir / f"{name}.log"
        log_path.write_text("".join(lines), encoding="utf-8")
        receipts.append(
            {
                "name": name,
                "command": shlex.join(command),
                "exit_code": exit_code,
                "passed": exit_code == 0,
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
            exit_code=exit_code,
            completed_units=index,
            total_units=len(commands),
        )
    return receipts


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover
    """Expose pending validation without creating success-shaped evidence."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"{RAW_DIR.as_posix()}/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    validation_runner: Callable[[Path, Path], list[JsonDict]] = _run_validations,
    capture_runner: Callable[
        [Mapping[str, Any], Path, Path, list[JsonDict]], JsonDict
    ] = capture_live,
) -> JsonDict:  # pragma: no cover - required native entrypoint.
    """Authenticate, capture, replay, validate, and atomically publish the canary."""

    repo = root or find_repo_root(start=__file__)
    result_path = repo / RESULT_PATH
    raw_dir = repo / RAW_DIR
    checkpoint_path = repo / CHECKPOINT_PATH
    checkpoint_dir = repo / CHECKPOINT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact = base_artifact(run_date)
    spans: list[JsonDict] = []

    _progress(0, "phase_start", operation="startup_and_output_authentication")
    _checkpoint(checkpoint_path, artifact)
    output_checks = [
        gate_row(
            "declared_output_absent",
            False,
            result_path.exists(),
            not result_path.exists(),
            upstream="declared_output",
            field=RESULT_PATH.as_posix(),
        )
    ]
    _progress(0, "phase_end", output_exists=result_path.exists())
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if not validate_artifact(existing):
            return existing
        blocked = finalize_blocked_artifact(
            artifact, output_checks, duration_s=time.monotonic() - started
        )
        _checkpoint(checkpoint_path, blocked)
        raise ValueError("declared output exists but is not a valid Exp7292 artifact")

    phase = time.monotonic()
    _progress(1, "phase_start", operation="authenticate_exp7291")
    checks, inputs = authenticate_inputs(repo)
    checks = output_checks + checks
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    spans.append({"phase": "authenticate_inputs", "duration_s": time.monotonic() - phase})
    _progress(1, "phase_end", failed=sum(row["passed"] is not True for row in checks))
    if any(row["passed"] is not True for row in checks):
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        _progress(12, "write_start", path=str(result_path))
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(12, "write_end", path=str(result_path))
        return blocked

    phase = time.monotonic()
    _progress(2, "phase_start", operation="seal_public_development_schedule")
    groups = select_development_groups(inputs["development_manifest"])
    schedule = build_schedule(groups)
    problems = schedule_errors(schedule, groups)
    schedule_check = gate_row(
        "frozen_canary_schedule",
        {"groups": PLANNED_GROUPS, "units": PLANNED_UNITS, "calls": PLANNED_CALLS, "errors": []},
        {
            "groups": len(groups),
            "units": sum(len(group["claims"]) for group in groups),
            "calls": len(schedule),
            "errors": problems,
        },
        not problems,
        upstream="exp7292-public-development-manifest",
        field="schedule",
    )
    checks.append(schedule_check)
    _write_or_match(raw_dir / "development-manifest.json", {"groups": groups})
    _write_or_match(raw_dir / "schedule.json", {"schedule": schedule})
    spans.append({"phase": "seal_schedule", "duration_s": time.monotonic() - phase})
    artifact["preconditions_checked"] = checks
    _checkpoint(checkpoint_path, artifact)
    _progress(2, "phase_end", groups=len(groups), calls=len(schedule), errors=problems)
    if not schedule_check["passed"]:
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        return blocked

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase = time.monotonic()
    _progress(3, "phase_start", operation="live_resource_preflight")
    resource_checks, context = collect_live_preflight(
        repo, schedule, result_path, checkpoint_path, raw_dir
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

    artifact["runtime_model_identity"] = deepcopy(context["model_identity"])
    phase = time.monotonic()
    _progress(6, "benchmark_start", operation="native_load_and_fixed_44_calls")
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    runtime_spans: list[JsonDict] = []
    capture = capture_runner(context, checkpoint_dir, raw_dir, runtime_spans)
    completion_rows = list(capture.get("rows") or [])
    loaded = capture.get("model_loaded") is True
    counts = {
        "model_loads_attempted": 1,
        "model_loads_completed": int(loaded),
        "model_loads_failed": int(not loaded),
        "model_loads_in_flight": 0,
        "generation_calls_attempted": len(completion_rows),
        "generation_calls_completed": sum(
            row.get("transport_complete") is True for row in completion_rows
        ),
        "generation_calls_failed": sum(
            row.get("transport_complete") is not True for row in completion_rows
        ),
        "generation_calls_in_flight": 0,
        "usable_answers": sum(row.get("usable") is True for row in completion_rows),
    }
    artifact["invocation_counts"] = counts
    artifact.update(classify_inference(counts))
    artifact["gpu_receipts"] = deepcopy(dict(capture.get("gpu_receipts") or {}))
    artifact["runner_receipt"] = deepcopy(dict(capture.get("runner_receipt") or {}))
    artifact["runner_receipt"].update(
        {
            "invocation_counts": deepcopy(counts),
            "runtime_phase_spans": runtime_spans,
            "runtime_error": capture.get("runtime_error"),
        }
    )
    spans.append({"phase": "native_capture", "duration_s": time.monotonic() - phase})
    _progress(
        6,
        "benchmark_end",
        operation="native_load_and_fixed_44_calls",
        completed_units=len(completion_rows),
        total_units=PLANNED_CALLS,
        runtime_error=capture.get("runtime_error"),
    )

    phase = time.monotonic()
    _progress(8, "benchmark_start", operation="independent_raw_reduction")
    replay_rows, replay_errors = independent_replay(schedule, completion_rows)
    reduced = reduce_canary(groups, schedule, replay_rows)
    warm = warm_prefix_receipt(schedule, replay_rows)
    gates = acceptance_gates(
        schedule,
        replay_rows,
        replay_errors,
        reduced,
        warm,
        model_identity=artifact["runtime_model_identity"],
        gpu_receipts=artifact["gpu_receipts"],
    )
    manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, artifact["runtime_model_identity"]
    )
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"][manifest["path"]] = {
        "sha256": manifest["sha256"],
        "terminal_class": manifest["status"],
        "retired": False,
        "quarantined": False,
    }
    spans.append({"phase": "independent_reduction", "duration_s": time.monotonic() - phase})
    artifact["phase_spans"] = spans
    artifact["validation_receipts"] = _pending_receipts()
    candidate = finalize_measured_artifact(
        artifact,
        completion_rows,
        replay_rows,
        replay_errors,
        reduced,
        warm,
        gates,
        duration_s=time.monotonic() - started,
    )
    _progress(
        8,
        "benchmark_end",
        operation="independent_raw_reduction",
        replayed=len(replay_rows),
        discrepancies=len(replay_errors),
        ready=candidate["reuse_canary_ready_score"],
    )
    _progress(9, "write_start", path=str(repo / RAW_CANDIDATE_PATH))
    atomic_write_json(repo / RAW_CANDIDATE_PATH, candidate, allow_override=True, sort_keys=True)
    _progress(9, "write_end", path=str(repo / RAW_CANDIDATE_PATH))
    _checkpoint(checkpoint_path, candidate)

    phase = time.monotonic()
    _progress(10, "phase_start", operation="focused_validation")
    receipts = validation_runner(repo, raw_dir)
    spans.append({"phase": "validation", "duration_s": time.monotonic() - phase})
    artifact["validation_receipts"] = receipts
    artifact["phase_spans"] = spans
    terminal = finalize_measured_artifact(
        artifact,
        completion_rows,
        replay_rows,
        replay_errors,
        reduced,
        warm,
        gates,
        duration_s=time.monotonic() - started,
    )
    _progress(
        10,
        "phase_end",
        passed=sum(row.get("passed") is True for row in receipts),
        total=len(receipts),
    )
    if not _validations_pass(receipts, include_full=False):
        _checkpoint(checkpoint_path, terminal)
        raise RuntimeError("scoped validation failed; terminal artifact not published")
    errors = validate_artifact(terminal)
    if errors:
        _checkpoint(checkpoint_path, terminal)
        raise ValueError(f"invalid Exp7292 terminal artifact: {errors}")
    atomic_write_json(repo / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(12, "write_start", path=str(result_path))
    atomic_write_json(result_path, terminal, allow_override=False, sort_keys=True)
    _progress(12, "write_end", path=str(result_path))
    return terminal


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V641 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the live canary or replay only task-owned raw evidence."""

    print("[exp7292] startup", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    if args.replay_raw is not None:
        _progress(1, "benchmark_start", operation="independent_raw_replay")
        reduced, errors = independent_replay_from_raw(args.replay_raw)
        _progress(
            1,
            "benchmark_end",
            operation="independent_raw_replay",
            comparative_rows=len(reduced.get("rows") or []),
            discrepancies=len(errors),
        )
        if errors or len(reduced.get("rows") or []) != PLANNED_UNITS * len(ARMS):
            print(canonical_json({"errors": errors}), flush=True)
            return 1
        return 0
    run_experiment(run_date=args.date)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
