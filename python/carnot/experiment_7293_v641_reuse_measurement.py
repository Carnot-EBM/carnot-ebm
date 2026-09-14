"""Measure distinct-claim source reuse with the mandated local GGUF.

The module extends the qualified Exp7292 transport. It keeps each source group
as one scheduling block and reduces full elapsed costs after all model replies
are immutable. Scorer-only labels are not opened until capture has ended.

Spec refs: REQ-VERIFY-7293 and SCENARIO-VERIFY-7293-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import time
from typing import Any

import yaml

from carnot import experiment_7237_v637_mention_canary as extraction
from carnot import experiment_7265_v639_mention_heldout as heldout
from carnot import experiment_7291_v641_reuse_fixture as fixture
from carnot import experiment_7292_v641_reuse_canary as canary
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260914"
MILESTONE = "2026.09.641"
EXPERIMENT_ID = "exp7293-reuse-measurement"
TASK_ID = "experiment_7293_v641_reuse_measurement"
SCHEMA = "carnot.exp7293.v641_reuse_measurement.v1"
RANDOM_SEED = 729_320_260_914
EVALUATION_SEED = 729_330_260_914
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

ARMS = ("warm_prefix_direct", "fresh_verifier", "versioned_reuse_verifier")
PLANNED_GROUPS = 16
CLAIMS_PER_GROUP = 8
PLANNED_UNITS = PLANNED_GROUPS * CLAIMS_PER_GROUP
CALLS_PER_GROUP = 42
PLANNED_CALLS = PLANNED_GROUPS * CALLS_PER_GROUP
OUTPUT_TOKEN_BUDGET = 128
MODEL_LOAD_CAP_S = 600.0
LIVE_WINDOW_CAP_S = 2700.0
REQUEST_CAP_S = 90.0

FIXTURE_PATH = Path("results/experiment_7291_v641_reuse_fixture.json")
CANARY_PATH = Path("results/experiment_7292_v641_reuse_canary.json")
MANIFEST_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/manifest.json")
ANALYSIS_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/analysis-contract.json")
AUTHORITY_PATH = Path("results/raw/experiment_7291_v641_reuse_fixture/scorer-only-labels.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7293_v641_reuse_measurement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7293_v641_reuse_measurement.py")
TEST_PATH = Path("tests/python/test_experiment_7293_v641_reuse_measurement.py")
RESULT_PATH = Path("results/experiment_7293_v641_reuse_measurement.json")
RAW_DIR = Path("results/raw/experiment_7293_v641_reuse_measurement")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
PER_UNIT_ROWS_PATH = RAW_DIR / "per-unit-rows.json"
ABORTED_RESUME_PATH = RAW_DIR / "aborted-resume-attempt.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7293_v641_reuse_measurement.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7293_v641_reuse_measurement")

PINNED_FIXTURE_SHA256 = "sha256:ecf1888d12bb8eac85b66ccde765d5bf5ca1a5358422b23590585024a629a263"
PINNED_CANARY_SHA256 = "sha256:918170e3e22e52a96ff46c2a27bb76573631d9b81330b9bed247a35c2f2c880f"
PINNED_MANIFEST_SHA256 = "sha256:fafc05f40627d55fd27c83c8e5be8fb710d9837f6ba15b3138fc01bbdb4b5a87"
PINNED_ANALYSIS_SHA256 = "sha256:d51e430f657d91dc1c6d37d9a4aba758fb74ff8cea85603384177d812f8f2f18"
PINNED_AUTHORITY_SHA256 = "sha256:6548f9f93d78348823a6cfa70469187cb3b80a7b981a0b8eebf2918ba9f7a493"

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
    "reuse_capture_complete_score": "One only for all planned groups, authentic live evidence and reconstructable complete costs.",
    "reuse_value_score": "Preliminary one only if every frozen freshness, parity and cost gate passes; audit controls final promotion.",
    "runtime_model_identity": "Same actual mandated GGUF and parser/configuration hashes as the canary.",
    "per_unit_rows_path": "Hash-bound raw row list; top-level rows retain all group/query/arm metrics and censoring.",
    "amortization_rows": "Cold and warm full-cost results at 1/2/4/8 distinct claims, including invalidation and failures.",
    "source_version_receipts": "Each served compilation version and current-source digest, never inferred from answer accuracy.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "capture_complete": "All sixteen groups need authentic model rows and reconstructable costs.",
    "raw_replay": "Immutable request and response bytes must rebuild every attempted call.",
    "source_freshness": "Version two must replace version one without serving stale constraints.",
    "semantic_parity": "Fresh and reused compiled source and claim constraints must match.",
    "accuracy_delta_lower": "The group-bootstrap accuracy lower bound must meet the frozen limit.",
    "coverage_delta_lower": "The group-bootstrap coverage lower bound must meet the frozen limit.",
    "false_accept_parity": "Reuse must not add empirical false accepts versus direct generation.",
    "eight_claim_cost_speedup_lower": "Eight claims must meet the frozen paired full-cost speedup.",
}
VALUE_GATE_NAMES = frozenset(GATE_PRINCIPLES) - {"capture_complete", "raw_replay"}

DIRECT_PROMPT = canary.DIRECT_PROMPT
DIRECT_GRAMMAR = deepcopy(canary.DIRECT_GRAMMAR)
LIVE_MODEL_CONFIGURATION = deepcopy(canary.LIVE_MODEL_CONFIGURATION)
LIVE_MODEL_CONFIGURATION_HASH = fixture.sha256_bytes(
    fixture.canonical_json(LIVE_MODEL_CONFIGURATION).encode("utf-8")
)

canonical_json = fixture.canonical_json
sha256_bytes = fixture.sha256_bytes
sha256_file = fixture.sha256_file


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence without treating local clock readings as inputs."""

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
    """Retain both sides of one precondition decision."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed prerequisite without hiding later checks."""

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


def freeze_arm_orders(groups: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Freeze one independently seeded arm permutation for each source group."""

    orders: dict[str, list[str]] = {}
    for group in groups:
        group_id = str(group["group_id"])
        seed_hash = sha256_bytes(f"{EVALUATION_SEED}:{group_id}".encode())
        rng = random.Random(int(seed_hash.removeprefix("sha256:")[:16], 16))
        order = list(ARMS)
        rng.shuffle(order)
        orders[group_id] = order
    return orders


def _source_for(group: Mapping[str, Any], version: int) -> JsonDict:
    """Return the exact public source revision named by one claim."""

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
    """Append one fixed public request without any scorer-only field."""

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
            "call_id": sha256_bytes(canonical_json(identity).encode("utf-8")),
            "unit_id": claim["unit_id"],
            "group_id": group["group_id"],
            "source_id": group["source_id"],
            "source_version": claim["source_version"],
            "claim_index": claim["chronology_index"],
            "arm": transport_arm,
            "comparison_arm": comparison_arm,
            "call_type": call_type,
            "draw": draw,
            "seed": seed,
            "document": deepcopy(dict(document)),
            "model_input": model_input,
            "input_sha256": sha256_bytes(canonical_json(model_input).encode("utf-8")),
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            **grammar,
            "output_token_budget": OUTPUT_TOKEN_BUDGET,
            "context_token_budget": extraction.CONTEXT_TOKEN_BUDGET,
            "request_timeout_s": REQUEST_CAP_S,
            "decoding_parameters": settings,
            "retry_budget": 0,
            "development_only": False,
            "held_out_eligible": True,
            "warm_cache_policy": "cache_prompt_true",
        }
    )


def _append_arm_block(schedule: list[JsonDict], group: Mapping[str, Any], arm: str) -> None:
    """Append one arm while preserving the fixed claim chronology."""

    claims = list(group["claims"])
    if arm == "warm_prefix_direct":
        for claim in claims:
            source = _source_for(group, int(claim["source_version"]))["document"]
            for draw in (1, 2):
                _append_call(
                    schedule,
                    group=group,
                    claim=claim,
                    comparison_arm=arm,
                    call_type="direct",
                    document={"source": source, "claim": claim["claim"]},
                    draw=draw,
                )
        return
    if arm == "fresh_verifier":
        for claim in claims:
            _append_call(
                schedule,
                group=group,
                claim=claim,
                comparison_arm=arm,
                call_type="source",
                document=_source_for(group, int(claim["source_version"]))["document"],
            )
            _append_call(
                schedule,
                group=group,
                claim=claim,
                comparison_arm=arm,
                call_type="claim",
                document=claim["claim"],
            )
        return
    if arm != "versioned_reuse_verifier":
        raise ValueError("comparison_arm")
    for version in (1, 2):
        version_claims = [row for row in claims if int(row["source_version"]) == version]
        if len(version_claims) != 4:
            raise ValueError("revision_claim_denominator")
        _append_call(
            schedule,
            group=group,
            claim=version_claims[0],
            comparison_arm=arm,
            call_type="source",
            document=_source_for(group, version)["document"],
        )
        for claim in version_claims:
            _append_call(
                schedule,
                group=group,
                claim=claim,
                comparison_arm=arm,
                call_type="claim",
                document=claim["claim"],
            )


def build_schedule(
    groups: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[str]],
    *,
    require_full_denominator: bool = True,
) -> list[JsonDict]:
    """Freeze all calls in contiguous, independently ordered group blocks."""

    if require_full_denominator and len(groups) != PLANNED_GROUPS:
        raise ValueError("evaluation_group_denominator")
    schedule: list[JsonDict] = []
    for group in groups:
        claims = list(group.get("claims") or [])
        if len(claims) != CLAIMS_PER_GROUP:
            raise ValueError("evaluation_claim_denominator")
        group_id = str(group["group_id"])
        order = list(orders.get(group_id) or [])
        if Counter(order) != Counter(ARMS):
            raise ValueError("group_arm_order")
        for arm in order:
            _append_arm_block(schedule, group, arm)
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[str]],
) -> list[str]:
    """Rebuild the public schedule and report any denominator or byte drift."""

    try:
        expected = build_schedule(
            groups, orders, require_full_denominator=len(groups) == PLANNED_GROUPS
        )
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != len(groups) * CALLS_PER_GROUP:
        errors.append("call_denominator")
    if len(schedule) != len(expected):
        errors.append("rebuilt_denominator")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        if observed != wanted:
            errors.append(f"call_{index}:changed")
    expected_counts = {
        "warm_prefix_direct": 16 * len(groups),
        "fresh_verifier": 16 * len(groups),
        "versioned_reuse_verifier": 10 * len(groups),
    }
    observed_counts = Counter(str(row.get("comparison_arm")) for row in schedule)
    if dict(observed_counts) != expected_counts:
        errors.append("arm_call_denominators")
    forbidden = ("expected_decision", "gold_span", "construction_label")
    if any(field in canonical_json(schedule) for field in forbidden):
        errors.append("authority_leakage")
    return list(dict.fromkeys(errors))


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - measured native response boundary.
    """Keep the qualified parser output and measured call-cost components."""

    started = time.monotonic()
    row = canary.build_completion_row(sealed, response, resource)
    processing = time.monotonic() - started
    timings = dict(dict(response.get("raw_response") or {}).get("timings") or {})
    row.update(
        {
            "attempted": True,
            "censored": False,
            "censoring_reason": None,
            "measured_processing_s": {
                "query_extraction": processing if sealed.get("call_type") == "claim" else 0.0,
                "source_compilation": processing if sealed.get("call_type") == "source" else 0.0,
                "prefix_prefill": float(timings.get("prompt_ms", 0.0) or 0.0) / 1000.0,
                "lookup_or_invalidation": 0.0,
                "verification": 0.0,
                "synchronization": 0.0,
            },
        }
    )
    row["row_sha256"] = heldout.capture._row_hash(row)
    return row


def censored_completion(sealed: Mapping[str, Any], reason: str) -> JsonDict:  # pragma: no cover
    """Retain one planned call that the fixed live window could not attempt."""

    row: JsonDict = {
        "call_order": sealed["call_order"],
        "call_id": sealed["call_id"],
        "unit_id": sealed["unit_id"],
        "group_id": sealed["group_id"],
        "source_id": sealed["source_id"],
        "source_version": sealed["source_version"],
        "arm": sealed["comparison_arm"],
        "transport_arm": sealed["arm"],
        "call_type": sealed["call_type"],
        "draw": sealed.get("draw"),
        "seed": sealed["seed"],
        "attempted": False,
        "censored": True,
        "censoring_reason": reason,
        "terminal_state": "censored",
        "transport_complete": False,
        "parse_valid": False,
        "usable": False,
        "errors": [reason],
        "model_response_error": reason,
        "raw_request_bytes_b64": "",
        "raw_response_bytes_b64": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "native_cache_receipt": {
            "cache_policy": "cache_prompt_true",
            "evidence_present": False,
            "cached_tokens": 0,
            "cache_n": 0,
        },
        "measured_processing_s": {
            "query_extraction": 0.0,
            "source_compilation": 0.0,
            "prefix_prefill": 0.0,
            "lookup_or_invalidation": 0.0,
            "verification": 0.0,
            "synchronization": 0.0,
        },
    }
    row["row_sha256"] = heldout.capture._row_hash(row)
    return row


def _canary_replay_projection(retained: Mapping[str, Any]) -> JsonDict:
    """Remove Exp7293 cost annotations before the upstream byte replay."""

    projected = deepcopy(dict(retained))
    for field in ("censored", "censoring_reason", "measured_processing_s"):
        projected.pop(field, None)
    projected["row_sha256"] = heldout.capture._row_hash(projected)
    return projected


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover - raw byte replay boundary.
    """Rebuild attempted rows and validate each explicit censoring receipt."""

    actual_count = next(
        (index for index, row in enumerate(retained_rows) if row.get("censored") is True),
        len(retained_rows),
    )
    replay_inputs = [_canary_replay_projection(row) for row in retained_rows[:actual_count]]
    rebuilt, errors = canary.independent_replay(schedule[:actual_count], replay_inputs)
    normalized: list[JsonDict] = []
    for replayed, retained in zip(rebuilt, retained_rows[:actual_count], strict=False):
        replayed.update(
            {
                "attempted": True,
                "censored": False,
                "censoring_reason": None,
                "measured_processing_s": deepcopy(
                    dict(retained.get("measured_processing_s") or {})
                ),
            }
        )
        replayed["row_sha256"] = heldout.capture._row_hash(replayed)
        normalized.append(replayed)
        if replayed != retained:
            errors.append(f"call_{replayed.get('call_order')}:measurement_replay_mismatch")
    for index in range(actual_count, len(schedule)):
        if index >= len(retained_rows):
            errors.append(f"call_{index}:missing_censor")
            continue
        retained = deepcopy(dict(retained_rows[index]))
        if (
            retained.get("call_id") != schedule[index].get("call_id")
            or retained.get("attempted") is not False
            or retained.get("censored") is not True
        ):
            errors.append(f"call_{index}:invalid_censor")
        normalized.append(retained)
    if len(retained_rows) != len(schedule):
        errors.append("replay_denominator")
    return normalized, list(dict.fromkeys(errors))


def _compiled(row: Mapping[str, Any] | None) -> JsonDict:
    """Return one compiled output or a visible missing-call failure."""

    if not row or not isinstance(row.get("compiled_completion"), Mapping):
        return {"outcome": "unknown", "relations": [], "errors": ["missing_call"]}
    return deepcopy(dict(row["compiled_completion"]))


def _row_errors(row: Mapping[str, Any] | None, default: str) -> list[str]:
    """Preserve specific parser or source errors before a generic fallback."""

    if not row:
        return [default]
    compiled = _compiled(row)
    errors = [str(value) for value in compiled.get("errors", [])]
    errors.extend(str(value) for value in row.get("errors", []) if value)
    model_error = row.get("model_response_error")
    if model_error:
        errors.append(str(model_error))
    return list(dict.fromkeys(errors or [default]))


def _decision(
    source_document: Mapping[str, Any],
    source_row: Mapping[str, Any] | None,
    claim_row: Mapping[str, Any] | None,
) -> JsonDict:
    """Execute two captured compilations without repairing an unusable source."""

    if not source_row or source_row.get("usable") is not True:
        return {
            "decision": "unknown",
            "abstention": True,
            "errors": _row_errors(source_row, "source_call_unusable"),
        }
    if not claim_row or claim_row.get("usable") is not True:
        return {
            "decision": "unknown",
            "abstention": True,
            "errors": _row_errors(claim_row, "claim_call_unusable"),
        }
    return fixture.pointer._execute_compiled_pair(
        source_document, _compiled(source_row), _compiled(claim_row)
    )


def _call_cost(row: Mapping[str, Any] | None) -> JsonDict:
    """Separate measured wall time from inclusive native prefill evidence."""

    if not row:
        return {
            "model_call_elapsed_s": 0.0,
            "prefix_prefill_inclusive_s": 0.0,
            "query_extraction_s": 0.0,
            "source_compilation_s": 0.0,
            "lookup_or_invalidation_s": 0.0,
            "verification_s": 0.0,
            "synchronization_s": 0.0,
            "full_elapsed_s": 0.0,
            "failed_calls": 1,
        }
    measured = dict(row.get("measured_processing_s") or {})
    wall = float(row.get("latency_s", 0.0) or 0.0)
    additive_names = (
        "query_extraction",
        "source_compilation",
        "lookup_or_invalidation",
        "verification",
        "synchronization",
    )
    additive = sum(float(measured.get(name, 0.0) or 0.0) for name in additive_names)
    return {
        "model_call_elapsed_s": wall,
        "prefix_prefill_inclusive_s": float(measured.get("prefix_prefill", 0.0) or 0.0),
        "query_extraction_s": float(measured.get("query_extraction", 0.0) or 0.0),
        "source_compilation_s": float(measured.get("source_compilation", 0.0) or 0.0),
        "lookup_or_invalidation_s": float(measured.get("lookup_or_invalidation", 0.0) or 0.0),
        "verification_s": float(measured.get("verification", 0.0) or 0.0),
        "synchronization_s": float(measured.get("synchronization", 0.0) or 0.0),
        "full_elapsed_s": wall + additive,
        "failed_calls": int(row.get("transport_complete") is not True),
    }


def _label_index(labels: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Build the scorer join only after the caller supplies authority rows."""

    return {
        str(row["unit_id"]): str(row["expected_decision"])
        for row in labels
        if row.get("split") == "evaluation"
    }


def _score_row(row: JsonDict, expected: str) -> JsonDict:
    """Attach accuracy, coverage, and false-accept evidence to one prediction."""

    predicted = str(row["prediction"])
    row.update(
        {
            "expected_decision": expected,
            "metric": int(predicted == expected),
            "coverage": int(predicted != "unknown"),
            "false_accept": int(
                predicted in {"supported", "contradicted"} and predicted != expected
            ),
        }
    )
    return row


def _aggregate_unique_cost(
    call_ids: Sequence[str],
    by_call_id: Mapping[str, Mapping[str, Any]],
    *,
    extra_lookup_s: float,
    extra_verification_s: float,
    extra_synchronization_s: float,
) -> tuple[JsonDict, float]:
    """Charge each generation once and add measured non-generation work."""

    unique_ids = list(dict.fromkeys(call_ids))
    components = {
        "model_call_elapsed": 0.0,
        "prefix_prefill_inclusive": 0.0,
        "query_extraction": 0.0,
        "source_compilation": 0.0,
        "lookup_or_invalidation": extra_lookup_s,
        "verification": extra_verification_s,
        "synchronization": extra_synchronization_s,
        "failed_call_elapsed": 0.0,
    }
    completed = 0
    failed = 0
    for call_id in unique_ids:
        row = by_call_id.get(call_id)
        cost = _call_cost(row)
        components["model_call_elapsed"] += cost["model_call_elapsed_s"]
        components["prefix_prefill_inclusive"] += cost["prefix_prefill_inclusive_s"]
        components["query_extraction"] += cost["query_extraction_s"]
        components["source_compilation"] += cost["source_compilation_s"]
        components["lookup_or_invalidation"] += cost["lookup_or_invalidation_s"]
        components["verification"] += cost["verification_s"]
        components["synchronization"] += cost["synchronization_s"]
        if row and row.get("transport_complete") is True:
            completed += 1
        else:
            failed += 1
            components["failed_call_elapsed"] += cost["model_call_elapsed_s"]
    # Native prefill is already inside HTTP wall time. Keeping it visible but
    # not adding it again prevents a false speedup from double charging.
    total = (
        components["model_call_elapsed"]
        + components["query_extraction"]
        + components["source_compilation"]
        + components["lookup_or_invalidation"]
        + components["verification"]
        + components["synchronization"]
    )
    return (
        {
            "unique_generation_calls": len(unique_ids),
            "completed_generation_calls": completed,
            "failed_or_censored_generation_calls": failed,
            "measured_cost_components_s": components,
            "native_prefix_prefill_included_in_model_call_elapsed": True,
        },
        total,
    )


def _latency_distribution(values: Sequence[float]) -> JsonDict:
    """Report deterministic empirical latency points without interpolation."""

    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min_s": None, "p50_s": None, "p95_s": None, "max_s": None}

    def select(probability: float) -> float:
        return ordered[int(probability * (len(ordered) - 1))]

    return {
        "count": len(ordered),
        "min_s": ordered[0],
        "p50_s": select(0.50),
        "p95_s": select(0.95),
        "max_s": ordered[-1],
    }


def _amortization_rows(
    groups: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    source_receipts: Sequence[Mapping[str, Any]],
    model_initialization_s: float,
) -> list[JsonDict]:
    """Build cold and steady full-cost rows for each group and claim count."""

    by_call_id = {str(row["call_id"]): row for row in completion_rows}
    by_unit_arm = {
        (str(row["group_id"]), str(row["unit_id"]), str(row["arm"])): row for row in rows
    }
    source_cost = {
        (str(row["group_id"]), int(row["source_version"])): row for row in source_receipts
    }
    initialization = model_initialization_s / len(ARMS)
    output: list[JsonDict] = []
    for group in groups:
        group_id = str(group["group_id"])
        claims = sorted(group["claims"], key=lambda row: int(row["chronology_index"]))
        for claim_count in (1, 2, 4, 8):
            selected = claims[:claim_count]
            versions = sorted({int(row["source_version"]) for row in selected})
            for arm in ARMS:
                unit_rows = [
                    by_unit_arm[(group_id, str(claim["unit_id"]), arm)] for claim in selected
                ]
                call_ids = [
                    str(call_id) for row in unit_rows for call_id in row.get("call_ids", [])
                ]
                extra_lookup = 0.0
                if arm == "versioned_reuse_verifier":
                    extra_lookup = sum(
                        float(
                            source_cost[(group_id, version)].get(
                                "lookup_or_invalidation_elapsed_s", 0.0
                            )
                            or 0.0
                        )
                        for version in versions
                    )
                receipt, steady = _aggregate_unique_cost(
                    call_ids,
                    by_call_id,
                    extra_lookup_s=extra_lookup,
                    extra_verification_s=sum(
                        float(row.get("verification_elapsed_s", 0.0) or 0.0) for row in unit_rows
                    ),
                    extra_synchronization_s=sum(
                        float(row.get("synchronization_elapsed_s", 0.0) or 0.0) for row in unit_rows
                    ),
                )
                output.append(
                    {
                        "group_id": group_id,
                        "arm": arm,
                        "claim_count": claim_count,
                        **receipt,
                        "model_initialization_allocation_s": initialization,
                        "steady_total_s": steady,
                        "cold_total_s": steady + initialization,
                        "all_failed_and_censored_calls_charged": True,
                    }
                )
    return output


def reduce_measurement(
    groups: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    model_initialization_s: float,
) -> JsonDict:
    """Reduce all arms and keep a failed reused source on each dependent row."""

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
    labels_by_unit = _label_index(labels)
    output: list[JsonDict] = []
    source_receipts: list[JsonDict] = []
    mismatches = 0
    served_stale = 0
    invalidations = 0
    for group in groups:
        group_id = str(group["group_id"])
        cache = fixture.SourceCompilationCache(enabled=True, capacity=1)
        reuse_sources: dict[int, JsonDict] = {}
        reuse_compiled: dict[int, JsonDict] = {}
        version_timings: dict[int, float] = {}
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
            source_view = deepcopy(dict(source_row or {}))
            cache_result: JsonDict = {
                "ok": False,
                "cache_hit": False,
                "compiled_source": None,
                "error": "source_compile_failed",
                "invalidated_entries": 0,
                "evicted_entries": 0,
            }
            lookup_started = time.perf_counter()
            if source_row and source_row.get("usable") is True:
                cache_result = cache.compile_or_get(
                    {
                        "source_id": group["source_id"],
                        "source_version": version,
                        "document": _source_for(group, version)["document"],
                        "completion": source_row.get("parsed_completion"),
                        "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                        "model_configuration": deepcopy(LIVE_MODEL_CONFIGURATION),
                    }
                )
                if cache_result.get("ok") is True:
                    compiled = deepcopy(dict(cache_result["compiled_source"]))
                    reuse_compiled[version] = compiled
                    source_view["compiled_completion"] = compiled
            lookup_elapsed = time.perf_counter() - lookup_started
            version_timings[version] = lookup_elapsed
            invalidations += int(cache_result.get("invalidated_entries", 0) or 0)
            reuse_sources[version] = source_view
            source_document = _source_for(group, version)["document"]
            source_receipts.append(
                {
                    "group_id": group_id,
                    "source_id": group["source_id"],
                    "source_version": version,
                    "compilation_call_id": source_row.get("call_id") if source_row else None,
                    "compilation_usable": bool(
                        source_row and source_row.get("usable") is True and cache_result.get("ok")
                    ),
                    "compilation_error": None
                    if cache_result.get("ok") is True
                    else ";".join(_row_errors(source_row, str(cache_result.get("error")))),
                    "source_content_sha256": sha256_bytes(
                        str(source_document["text"]).encode("utf-8")
                    ),
                    "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                    "model_configuration_hash": LIVE_MODEL_CONFIGURATION_HASH,
                    "invalidated_entries": int(cache_result.get("invalidated_entries", 0) or 0),
                    "served_dependent_claims": len(version_claims),
                    "lookup_or_invalidation_elapsed_s": lookup_elapsed,
                    "cache_hit": cache_result.get("cache_hit") is True,
                    "stale_constraints_served": 0,
                }
            )
        served_stale += cache.served_stale_constraints
        for claim in group["claims"]:
            unit_id = str(claim["unit_id"])
            version = int(claim["source_version"])
            expected = labels_by_unit.get(unit_id, "unknown")
            source_document = _source_for(group, version)["document"]
            direct_calls = [
                by_key.get((group_id, unit_id, "warm_prefix_direct", "direct", draw))
                for draw in (1, 2)
            ]
            draws = [
                str(dict(row.get("compiled_completion") or {}).get("decision", "unknown"))
                if row and row.get("usable") is True
                else "unknown"
                for row in direct_calls
            ]
            direct_prediction = fixture.reduce_two_draws(draws[0], draws[1])
            direct_complete = all(
                row and row.get("transport_complete") is True for row in direct_calls
            )
            direct_error = None if direct_complete else "direct_call_incomplete"
            output.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_version": version,
                        "arm": "warm_prefix_direct",
                        "seed": claim["seed"],
                        "prediction": direct_prediction,
                        "draws": draws,
                        "error": direct_error,
                        "abstention": direct_prediction == "unknown",
                        "censored": not direct_complete,
                        "call_ids": [str(row["call_id"]) for row in direct_calls if row],
                        "verification_elapsed_s": 0.0,
                        "synchronization_elapsed_s": 0.0,
                    },
                    expected,
                )
            )

            fresh_source = by_key.get((group_id, unit_id, "fresh_verifier", "source", None))
            fresh_claim = by_key.get((group_id, unit_id, "fresh_verifier", "claim", None))
            verify_started = time.perf_counter()
            fresh_decision = _decision(source_document, fresh_source, fresh_claim)
            fresh_verify_s = time.perf_counter() - verify_started
            fresh_complete = bool(
                fresh_source
                and fresh_claim
                and fresh_source.get("transport_complete") is True
                and fresh_claim.get("transport_complete") is True
            )
            fresh_source_compiled = _compiled(fresh_source)
            fresh_claim_compiled = _compiled(fresh_claim)
            fresh_errors = [str(value) for value in fresh_decision.get("errors", [])]
            output.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_version": version,
                        "arm": "fresh_verifier",
                        "seed": claim["seed"],
                        "prediction": fresh_decision["decision"],
                        "error": None
                        if fresh_complete and not fresh_errors
                        else ";".join(fresh_errors or ["fresh_call_incomplete"]),
                        "abstention": fresh_decision.get("abstention") is True,
                        "censored": not fresh_complete,
                        "call_ids": [
                            str(row["call_id"]) for row in (fresh_source, fresh_claim) if row
                        ],
                        "serialized_source_constraints": canonical_json(fresh_source_compiled),
                        "serialized_claim_constraints": canonical_json(fresh_claim_compiled),
                        "verification_elapsed_s": fresh_verify_s,
                        "synchronization_elapsed_s": 0.0,
                    },
                    expected,
                )
            )

            reuse_source = reuse_sources.get(version)
            reuse_claim = by_key.get((group_id, unit_id, "versioned_reuse_verifier", "claim", None))
            verify_started = time.perf_counter()
            reuse_decision = _decision(source_document, reuse_source, reuse_claim)
            reuse_verify_s = time.perf_counter() - verify_started
            reuse_complete = bool(
                reuse_source
                and reuse_claim
                and reuse_source.get("transport_complete") is True
                and reuse_claim.get("transport_complete") is True
            )
            reuse_source_compiled = _compiled(reuse_source)
            reuse_claim_compiled = _compiled(reuse_claim)
            parity = bool(
                canonical_json(fresh_source_compiled) == canonical_json(reuse_source_compiled)
                and canonical_json(fresh_claim_compiled) == canonical_json(reuse_claim_compiled)
            )
            mismatches += int(not parity)
            reuse_errors = [str(value) for value in reuse_decision.get("errors", [])]
            if not parity:
                reuse_errors.append("fresh_reuse_serialization_mismatch")
            source_call_id = reuse_source.get("call_id") if reuse_source else None
            output.append(
                _score_row(
                    {
                        "group_id": group_id,
                        "unit_id": unit_id,
                        "claim_index": claim["chronology_index"],
                        "source_version": version,
                        "arm": "versioned_reuse_verifier",
                        "seed": claim["seed"],
                        "prediction": reuse_decision["decision"],
                        "error": None
                        if reuse_complete and not reuse_errors
                        else ";".join(reuse_errors or ["reuse_call_incomplete"]),
                        "abstention": reuse_decision.get("abstention") is True,
                        "censored": not reuse_complete,
                        "call_ids": [
                            str(row["call_id"]) for row in (reuse_source, reuse_claim) if row
                        ],
                        "source_compilation_call_id": source_call_id,
                        "serialized_source_constraints": canonical_json(reuse_source_compiled),
                        "serialized_claim_constraints": canonical_json(reuse_claim_compiled),
                        "fresh_reuse_serialization_equal": parity,
                        "verification_elapsed_s": reuse_verify_s,
                        "synchronization_elapsed_s": 0.0,
                    },
                    expected,
                )
            )
    amortization = _amortization_rows(
        groups, output, completion_rows, source_receipts, model_initialization_s
    )
    latency_rows = []
    for arm in ARMS:
        latencies = [
            float(row.get("latency_s", 0.0) or 0.0)
            for row in completion_rows
            if row.get("arm") == arm and row.get("attempted") is not False
        ]
        latency_rows.append({"arm": arm, **_latency_distribution(latencies)})
    return {
        "rows": output,
        "source_version_receipts": source_receipts,
        "amortization_rows": amortization,
        "latency_distributions": latency_rows,
        "fresh_reuse_serialization_mismatches": mismatches,
        "source_version_invalidations": invalidations,
        "served_stale_constraints": served_stale,
        "cost_accounting": {
            "model_initialization_s": model_initialization_s,
            "allocation_rule": "equal_one_third_per_arm",
            "cached_tokens_used_as_wall_time": False,
            "native_prefill_included_once": True,
            "failed_calls_charged": True,
        },
        "schedule_sha256": sha256_bytes(canonical_json(list(schedule)).encode("utf-8")),
    }


def _percentile(
    values: Sequence[float], probability: float
) -> float:  # pragma: no cover - used by the live-only bootstrap.
    """Select a fixed empirical percentile without interpolation choices."""

    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    amortization_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int,
) -> JsonDict:  # pragma: no cover - exercised by the complete live reduction.
    """Resample source groups while keeping their eight claims and arms paired."""

    groups = sorted({str(row["group_id"]) for row in rows})
    by_group_arm = {
        (group, arm): [
            row for row in rows if row.get("group_id") == group and row.get("arm") == arm
        ]
        for group in groups
        for arm in ARMS
    }
    cost_by_group_arm = {
        (str(row["group_id"]), str(row["arm"])): float(row["steady_total_s"])
        for row in amortization_rows
        if row.get("claim_count") == 8
    }

    def mean_metric(sample: Sequence[str], arm: str, metric: str) -> float:
        values = [float(row[metric]) for group in sample for row in by_group_arm[(group, arm)]]
        return sum(values) / len(values)

    def speedup(sample: Sequence[str]) -> float:
        direct = sum(cost_by_group_arm[(group, "warm_prefix_direct")] for group in sample)
        reuse = sum(cost_by_group_arm[(group, "versioned_reuse_verifier")] for group in sample)
        return direct / reuse if reuse > 0 else 0.0

    rng = random.Random(seed)
    accuracy: list[float] = []
    coverage: list[float] = []
    speedups: list[float] = []
    for _ in range(draws):
        sample = [rng.choice(groups) for _group in groups]
        accuracy.append(
            mean_metric(sample, "versioned_reuse_verifier", "metric")
            - mean_metric(sample, "warm_prefix_direct", "metric")
        )
        coverage.append(
            mean_metric(sample, "versioned_reuse_verifier", "coverage")
            - mean_metric(sample, "warm_prefix_direct", "coverage")
        )
        speedups.append(speedup(sample))
    false_accepts = {
        arm: sum(int(row["false_accept"]) for row in rows if row.get("arm") == arm) for arm in ARMS
    }
    return {
        "method": "paired_nonparametric_bootstrap_over_source_groups",
        "resampling_unit": "source_group",
        "independent_groups": len(groups),
        "claims_per_group": CLAIMS_PER_GROUP,
        "draws": draws,
        "seed": seed,
        "interval": "one_sided_95_percent_empirical_lower",
        "accuracy_delta": {
            "estimate": mean_metric(groups, "versioned_reuse_verifier", "metric")
            - mean_metric(groups, "warm_prefix_direct", "metric"),
            "one_sided_95_lower": _percentile(accuracy, 0.05),
        },
        "coverage_delta": {
            "estimate": mean_metric(groups, "versioned_reuse_verifier", "coverage")
            - mean_metric(groups, "warm_prefix_direct", "coverage"),
            "one_sided_95_lower": _percentile(coverage, 0.05),
        },
        "false_accept_counts": false_accepts,
        "eight_claim_cost_speedup": {
            "estimate": speedup(groups),
            "one_sided_95_lower": _percentile(speedups, 0.05),
        },
    }


def acceptance_row(
    criterion: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str | None = None,
) -> JsonDict:
    """Keep the expected and observed values for one frozen gate."""

    return {
        "criterion": criterion,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle or GATE_PRINCIPLES.get(criterion, "Frozen acceptance check."),
    }


def acceptance_gates(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    reduced: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    *,
    gpu_receipts: Mapping[str, Any],
) -> list[JsonDict]:  # pragma: no cover - complete live artifact reduction.
    """Apply capture and value gates from the frozen analysis contract."""

    rows = list(reduced.get("rows") or [])
    completed_groups = 0
    for group_id in sorted({str(row.get("group_id")) for row in schedule}):
        group_calls = [row for row in completion_rows if row.get("group_id") == group_id]
        if len(group_calls) == CALLS_PER_GROUP and all(
            row.get("attempted") is True and row.get("transport_complete") is True
            for row in group_calls
        ):
            completed_groups += 1
    accuracy_lower = float(
        dict(bootstrap.get("accuracy_delta") or {}).get("one_sided_95_lower", -1.0)
    )
    coverage_lower = float(
        dict(bootstrap.get("coverage_delta") or {}).get("one_sided_95_lower", -1.0)
    )
    false_accepts = dict(bootstrap.get("false_accept_counts") or {})
    speedup_lower = float(
        dict(bootstrap.get("eight_claim_cost_speedup") or {}).get("one_sided_95_lower", 0.0)
    )
    return [
        acceptance_row(
            "capture_complete",
            PLANNED_GROUPS,
            completed_groups,
            completed_groups == PLANNED_GROUPS
            and len(completion_rows) == len(schedule) == PLANNED_CALLS
            and len(rows) == PLANNED_UNITS * len(ARMS)
            and gpu_receipts.get("provenance_ok") is True,
        ),
        acceptance_row("raw_replay", [], list(replay_errors), not replay_errors),
        acceptance_row(
            "source_freshness",
            {"served_stale_constraints": 0, "invalidations": PLANNED_GROUPS},
            {
                "served_stale_constraints": reduced.get("served_stale_constraints"),
                "invalidations": reduced.get("source_version_invalidations"),
            },
            reduced.get("served_stale_constraints") == 0
            and reduced.get("source_version_invalidations") == PLANNED_GROUPS,
        ),
        acceptance_row(
            "semantic_parity",
            0,
            reduced.get("fresh_reuse_serialization_mismatches"),
            reduced.get("fresh_reuse_serialization_mismatches") == 0,
        ),
        acceptance_row("accuracy_delta_lower", -0.02, accuracy_lower, accuracy_lower >= -0.02),
        acceptance_row("coverage_delta_lower", -0.02, coverage_lower, coverage_lower >= -0.02),
        acceptance_row(
            "false_accept_parity",
            "reuse_minus_direct<=0",
            int(false_accepts.get("versioned_reuse_verifier", 0) or 0)
            - int(false_accepts.get("warm_prefix_direct", 0) or 0),
            int(false_accepts.get("versioned_reuse_verifier", 0) or 0)
            <= int(false_accepts.get("warm_prefix_direct", 0) or 0),
        ),
        acceptance_row("eight_claim_cost_speedup_lower", 1.5, speedup_lower, speedup_lower >= 1.5),
    ]


def capture_complete_score(gates: Sequence[Mapping[str, Any]]) -> int:
    """Return one only when the explicit full-capture gate passes."""

    matches = [row for row in gates if row.get("criterion") == "capture_complete"]
    return int(len(matches) == 1 and matches[0].get("passed") is True)


def value_score(gates: Sequence[Mapping[str, Any]]) -> int:
    """Return one only when every frozen freshness, parity, and value gate passes."""

    by_name = {str(row.get("criterion")): row for row in gates}
    return int(
        VALUE_GATE_NAMES.issubset(by_name)
        and all(by_name[name].get("passed") is True for name in VALUE_GATE_NAMES)
    )


def classify_verdict(
    gates: Sequence[Mapping[str, Any]], *, verifier_is_oracle: bool
) -> tuple[str, str]:
    """Keep a complete negative separate from unfinished checkpoint work."""

    if capture_complete_score(gates) != 1 or value_score(gates) != 1:
        return "null", "complete_null_reuse_value_gate_failed"
    if verifier_is_oracle:
        return "circular_positive", "complete_circular_positive_reuse_preliminary_gates_passed"
    return "positive", "complete_positive_reuse_preliminary_gates_passed"


def classify_inference(counts: Mapping[str, Any]) -> JsonDict:
    """Derive substrate fields from attempted model boundaries only."""

    loads = int(counts.get("model_loads_attempted", 0) or 0)
    generations = int(counts.get("generation_calls_attempted", 0) or 0)
    if generations:
        return {
            "model_invoked": True,
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_full_generation",
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


def authenticate_inputs(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate both upstream gates without opening scorer-only labels."""

    paths = {
        "fixture": root / FIXTURE_PATH,
        "canary": root / CANARY_PATH,
        "manifest": root / MANIFEST_PATH,
        "analysis": root / ANALYSIS_PATH,
        "authority": root / AUTHORITY_PATH,
        "exclusion": root / EXCLUSION_PATH,
    }
    expected_hashes = {
        "fixture": PINNED_FIXTURE_SHA256,
        "canary": PINNED_CANARY_SHA256,
        "manifest": PINNED_MANIFEST_SHA256,
        "analysis": PINNED_ANALYSIS_SHA256,
        "authority": PINNED_AUTHORITY_SHA256,
    }
    hashes = {name: sha256_file(path) if path.is_file() else None for name, path in paths.items()}
    checks = [
        gate_row(
            f"{name}_hash",
            expected,
            hashes[name],
            hashes[name] == expected,
            upstream={
                "fixture": FIXTURE_PATH,
                "canary": CANARY_PATH,
                "manifest": MANIFEST_PATH,
                "analysis": ANALYSIS_PATH,
                "authority": AUTHORITY_PATH,
            }[name].as_posix(),
            field="sha256",
        )
        for name, expected in expected_hashes.items()
    ]
    if not all(path.is_file() for path in paths.values()):
        return checks, {}
    try:
        fixture_artifact = json.loads(paths["fixture"].read_text(encoding="utf-8"))
        canary_artifact = json.loads(paths["canary"].read_text(encoding="utf-8"))
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
    fixture_errors = fixture.validate_artifact(fixture_artifact, root=root)
    canary_errors = canary.validate_artifact(canary_artifact)
    checks.extend(
        [
            gate_row(
                "fixture_terminal_schema",
                [],
                fixture_errors,
                not fixture_errors,
                upstream="exp7291-reuse-fixture",
                field="terminal_schema_and_checksum",
            ),
            gate_row(
                "canary_terminal_schema",
                [],
                canary_errors,
                not canary_errors,
                upstream="exp7292-reuse-canary",
                field="terminal_schema_and_checksum",
            ),
            gate_row(
                "reuse_fixture_ready",
                1,
                fixture_artifact.get("reuse_fixture_ready_score"),
                fixture_artifact.get("reuse_fixture_ready_score") == 1,
                upstream="exp7291-reuse-fixture",
                field="reuse_fixture_ready_score",
            ),
            gate_row(
                "reuse_canary_ready",
                1,
                canary_artifact.get("reuse_canary_ready_score"),
                canary_artifact.get("reuse_canary_ready_score") == 1,
                upstream="exp7292-reuse-canary",
                field="reuse_canary_ready_score",
            ),
            gate_row(
                "upstream_terminal_status",
                {"fixture": "complete", "canary": "complete"},
                {
                    "fixture": fixture_artifact.get("status"),
                    "canary": canary_artifact.get("status"),
                },
                fixture_artifact.get("status") == canary_artifact.get("status") == "complete",
                upstream="exp7291_and_exp7292",
                field="status",
            ),
            gate_row(
                "shared_manifest_hashes",
                {
                    "manifest": PINNED_MANIFEST_SHA256,
                    "analysis": PINNED_ANALYSIS_SHA256,
                    "authority": PINNED_AUTHORITY_SHA256,
                },
                {
                    "manifest": fixture_artifact.get("source_artifact_hashes", {}).get(
                        MANIFEST_PATH.as_posix()
                    ),
                    "analysis": dict(manifest.get("analysis_contract") or {}).get("sha256"),
                    "authority": dict(manifest.get("scorer_authority") or {}).get("sha256"),
                },
                fixture_artifact.get("source_artifact_hashes", {}).get(MANIFEST_PATH.as_posix())
                == PINNED_MANIFEST_SHA256
                and dict(manifest.get("analysis_contract") or {}).get("sha256")
                == PINNED_ANALYSIS_SHA256
                and dict(manifest.get("scorer_authority") or {}).get("sha256")
                == PINNED_AUTHORITY_SHA256,
                upstream="exp7291-reuse-fixture",
                field="manifest_analysis_authority_sha256",
            ),
            gate_row(
                "evaluation_public_boundary",
                {"groups": PLANNED_GROUPS, "authority_fields_present": False},
                {
                    "groups": len(manifest.get("evaluation_groups") or []),
                    "authority_fields_present": manifest.get("authority_fields_present"),
                },
                len(manifest.get("evaluation_groups") or []) == PLANNED_GROUPS
                and manifest.get("authority_fields_present") is False,
                upstream=MANIFEST_PATH.as_posix(),
                field="evaluation_groups_and_authority_boundary",
            ),
            gate_row(
                "analysis_frozen",
                True,
                analysis.get("frozen_before_predictions"),
                analysis.get("frozen_before_predictions") is True,
                upstream=ANALYSIS_PATH.as_posix(),
                field="frozen_before_predictions",
            ),
            gate_row(
                "shared_parser_and_configuration",
                {
                    "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                    "model_configuration_hash": LIVE_MODEL_CONFIGURATION_HASH,
                },
                {
                    "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
                    "model_configuration_hash": fixture.sha256_bytes(
                        fixture.canonical_json(canary.LIVE_MODEL_CONFIGURATION).encode("utf-8")
                    ),
                },
                LIVE_MODEL_CONFIGURATION == canary.LIVE_MODEL_CONFIGURATION,
                upstream="python/carnot/experiment_7291_v641_reuse_fixture.py_and_exp7292",
                field="parser_schema_hash_and_model_configuration_hash",
            ),
            gate_row(
                "scorer_not_opened_by_generator",
                False,
                bool(
                    dict(manifest.get("scorer_authority") or {}).get("readable_by_prediction_path")
                ),
                dict(manifest.get("scorer_authority") or {}).get("readable_by_prediction_path")
                is False,
                upstream=MANIFEST_PATH.as_posix(),
                field="scorer_authority.readable_by_prediction_path",
            ),
        ]
    )
    excluded = any(
        fixture._manifest_lists_experiment(exclusions, experiment)
        for experiment in (EXPERIMENT_ID, "exp7291-reuse-fixture", "exp7292-reuse-canary")
    )
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
        "fixture_artifact": fixture_artifact,
        "canary_artifact": canary_artifact,
        "manifest": manifest,
        "analysis": analysis,
        "evaluation_groups": deepcopy(list(manifest["evaluation_groups"])),
        "authority_path": paths["authority"],
        "authority_path_opened": False,
        "canary_model_identity": deepcopy(dict(canary_artifact["runtime_model_identity"])),
    }


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover
    """Hash each repository source and upstream artifact used by the run."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        FIXTURE_PATH,
        CANARY_PATH,
        MANIFEST_PATH,
        ANALYSIS_PATH,
        AUTHORITY_PATH,
        fixture.MODULE_PATH,
        canary.MODULE_PATH,
        extraction.MODULE_PATH,
        heldout.MODULE_PATH,
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


def base_artifact(run_date: str) -> JsonDict:  # pragma: no cover
    """Create all required fields before any fallible host operation."""

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
            "evaluation_observed_after_capture": False,
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
            "stopping_rule": "attempt each frozen group once within 2700 seconds; retain all 672 rows; never retry, repair, tune, expand, or shrink",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7293_running_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "reuse_capture_complete_score": 0,
        "reuse_value_score": 0,
        "runtime_model_identity": {},
        "per_unit_rows_path": {
            "path": PER_UNIT_ROWS_PATH.as_posix(),
            "sha256": "pending",
            "row_count": 0,
        },
        "amortization_rows": [],
        "source_version_receipts": [],
        "latency_distributions": [],
        "cost_accounting": {},
        "bootstrap_receipt": {},
        "per_call_rows": [],
        "source_claim_manifest": {},
        "arm_order_receipt": {},
        "raw_call_manifest": {},
        "replay_discrepancies": [],
        "phase_spans": [],
        "runner_receipt": {},
        "gpu_receipts": {},
        "limitations": {
            "independent_audit_controls_final_promotion": True,
            "rare_error_safety_certified": False,
            "general_source_extraction_correctness_claimed": False,
            "verifier_ceiling_comparison_claimed": False,
        },
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:  # pragma: no cover
    """Finish an external precondition block without claiming unfinished science."""

    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["reuse_capture_complete_score"] = 0
    artifact["reuse_value_score"] = 0
    artifact.update(classify_inference(artifact.get("invocation_counts") or {}))
    failure = artifact["gate_check_summary"].get("failed_check") or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_exp7293_{failure}"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


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


def _validations_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:  # pragma: no cover
    """Require one passing terminal receipt for every fixed command."""

    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == set(REQUIRED_VALIDATION_NAMES) and all(
        by_name[name].get("passed") is True and by_name[name].get("exit_code") == 0
        for name in REQUIRED_VALIDATION_NAMES
    )


def finalize_measured_artifact(
    artifact: JsonDict,
    completion_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    reduced: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    per_unit_receipt: Mapping[str, Any],
    *,
    duration_s: float,
) -> JsonDict:  # pragma: no cover
    """Finish one complete attempt while leaving final promotion to Exp7294."""

    artifact["per_call_rows"] = deepcopy(list(completion_rows))
    artifact["rows"] = deepcopy(list(reduced.get("rows") or []))
    artifact["amortization_rows"] = deepcopy(list(reduced.get("amortization_rows") or []))
    artifact["source_version_receipts"] = deepcopy(
        list(reduced.get("source_version_receipts") or [])
    )
    artifact["latency_distributions"] = deepcopy(list(reduced.get("latency_distributions") or []))
    artifact["cost_accounting"] = deepcopy(dict(reduced.get("cost_accounting") or {}))
    artifact["bootstrap_receipt"] = deepcopy(dict(bootstrap))
    artifact["per_unit_rows_path"] = deepcopy(dict(per_unit_receipt))
    artifact["replay_discrepancies"] = list(replay_errors)
    artifact["acceptance_gate_results"] = deepcopy(list(gates))
    artifact["reuse_capture_complete_score"] = capture_complete_score(gates)
    artifact["reuse_value_score"] = value_score(gates)
    counts = dict(artifact.get("invocation_counts") or ZERO_INVOCATION_COUNTS)
    artifact.update(classify_inference(counts))
    artifact["sample_size_budget"].update(
        {
            "attempted_calls": int(counts.get("generation_calls_attempted", 0) or 0),
            "completed_calls": int(counts.get("generation_calls_completed", 0) or 0),
            "failed_calls": int(counts.get("generation_calls_failed", 0) or 0),
            "censored_calls": sum(row.get("censored") is True for row in completion_rows),
            "complete_groups": sum(
                1
                for group_id in {str(row.get("group_id")) for row in completion_rows}
                if len([row for row in completion_rows if str(row.get("group_id")) == group_id])
                == CALLS_PER_GROUP
                and all(
                    row.get("transport_complete") is True
                    for row in completion_rows
                    if str(row.get("group_id")) == group_id
                )
            ),
        }
    )
    artifact["status"] = "complete"
    artifact["verdict_class"], artifact["honest_verdict"] = classify_verdict(
        gates, verifier_is_oracle=artifact.get("verifier_is_oracle") is True
    )
    if artifact.get("validation_receipts") and not _validations_pass(
        artifact["validation_receipts"]
    ):
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_disqualified_reuse_validation_failed"
    artifact["gate_check_summary"] = None
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, require_validations: bool = True
) -> list[str]:  # pragma: no cover
    """Cold-check identity, terminal semantics, denominators, and raw-row binding."""

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
    elif any(
        value.get(field) != observed for field, observed in classify_inference(counts).items()
    ):
        errors.append("inference_classification")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or not str(value.get("honest_verdict", "")).startswith("blocked_")
            or value.get("reuse_capture_complete_score") != 0
            or value.get("reuse_value_score") != 0
            or not isinstance(summary, Mapping)
            or not summary.get("failed_check")
        ):
            errors.append("blocked_terminal_state")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    gates = value.get("acceptance_gate_results")
    if not isinstance(gates, list):
        errors.append("acceptance_gate_results")
    else:
        if value.get("reuse_capture_complete_score") != capture_complete_score(gates):
            errors.append("reuse_capture_complete_score")
        if value.get("reuse_value_score") != value_score(gates):
            errors.append("reuse_value_score")
    if len(value.get("per_call_rows") or []) != PLANNED_CALLS:
        errors.append("per_call_rows")
    if len(value.get("rows") or []) != PLANNED_UNITS * len(ARMS):
        errors.append("rows")
    if len(value.get("amortization_rows") or []) != PLANNED_GROUPS * 4 * len(ARMS):
        errors.append("amortization_rows")
    if len(value.get("source_version_receipts") or []) != PLANNED_GROUPS * 2:
        errors.append("source_version_receipts")
    per_unit = value.get("per_unit_rows_path")
    if (
        not isinstance(per_unit, Mapping)
        or per_unit.get("path") != PER_UNIT_ROWS_PATH.as_posix()
        or per_unit.get("row_count") != PLANNED_UNITS * len(ARMS)
        or not str(per_unit.get("sha256", "")).startswith("sha256:")
    ):
        errors.append("per_unit_rows_path")
    if value.get("verdict_class") not in {"circular_positive", "null", "disqualified"}:
        errors.append("verdict_class")
    if value.get("verifier_is_oracle") is True and value.get("verdict_class") == "positive":
        errors.append("oracle_positive")
    if require_validations and not _validations_pass(value.get("validation_receipts") or []):
        errors.append("validation_receipts")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase and slow-call boundary for the outer monitor."""

    print(f"[exp7293] {canonical_json({'phase': phase, 'event': event, **details})}", flush=True)


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
    """Write unfinished work only below the declared checkpoint directory."""

    value = deepcopy(dict(artifact))
    if value.get("status") not in {"complete", "blocked"}:
        value["status"] = "partial"
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_exp7293_unfinished_checkpoint_only"
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=True, sort_keys=True)


def collect_live_preflight(
    root: Path,
    schedule: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    expected_model_identity: Mapping[str, Any],
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Reuse the qualified loader and require the canary's exact model bytes."""

    checks, context = canary.collect_live_preflight(
        root, schedule, result_path, checkpoint_path, raw_dir
    )
    actual_identity = dict(context.get("model_identity") or {})
    identity_fields = (
        "hf_id",
        "quantization",
        "gguf_sha256",
        "revision",
        "embedded_tokenizer_sha256",
        "embedded_chat_template_sha256",
        "auto_tokenizer_used",
        "runtime",
    )
    expected = {field: expected_model_identity.get(field) for field in identity_fields}
    actual = {field: actual_identity.get(field) for field in identity_fields}
    checks.append(
        gate_row(
            "exact_canary_runtime_model_identity",
            expected,
            actual,
            actual == expected,
            upstream="exp7292-reuse-canary",
            field="runtime_model_identity",
        )
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
) -> JsonDict:  # pragma: no cover
    """Capture once, then retain a censoring row for every unattempted call."""

    captured = canary.capture_live(context, checkpoint_dir, raw_dir, spans)
    rows = list(captured.get("rows") or [])
    schedule = list(context["schedule"])
    for sealed in schedule[len(rows) :]:
        row = censored_completion(sealed, "generation_deadline_or_runtime_end")
        rows.append(row)
        _write_or_match(
            raw_dir / f"call_{int(sealed['call_order']):02d}.json",
            {"schedule": dict(sealed), "completion": row},
        )
    captured["rows"] = rows
    captured["actual_generation_rows"] = sum(row.get("attempted") is True for row in rows)
    captured["scheduled_outcomes_accounted"] = len(rows)
    return captured


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover
    """Index every authentic or censored call with an exact content hash."""

    calls = []
    for index, (sealed, row) in enumerate(zip(schedule, completion_rows, strict=False)):
        path = raw_dir / f"call_{index:02d}.json"
        _write_or_match(path, {"schedule": dict(sealed), "completion": dict(row)})
        calls.append(
            {
                "call_order": index,
                "call_id": row.get("call_id"),
                "group_id": row.get("group_id"),
                "arm": row.get("arm"),
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "attempted": row.get("attempted") is True,
                "terminal_state": row.get("terminal_state"),
                "cached_tokens": dict(row.get("native_cache_receipt") or {}).get("cached_tokens"),
            }
        )
    manifest = {
        "schema": "carnot.exp7293.raw_call_manifest.v1",
        "status": "complete" if len(calls) == PLANNED_CALLS else "partial",
        "planned_calls": PLANNED_CALLS,
        "retained_calls": len(calls),
        "attempted_calls": sum(row["attempted"] for row in calls),
        "schedule_sha256": sha256_bytes(canonical_json(list(schedule)).encode("utf-8")),
        "model_identity": deepcopy(dict(model_identity)),
        "authority_path_opened_by_model_worker": False,
        "evaluation_labels_read_during_capture": False,
        "calls": calls,
    }
    path = raw_dir / "raw-call-manifest.json"
    _write_or_match(path, manifest)
    return {**manifest, "path": path.as_posix(), "sha256": sha256_file(path)}


def write_per_unit_rows(raw_dir: Path, reduced: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Bind scored rows, source revisions, and amortization to raw evidence."""

    value = {
        "schema": "carnot.exp7293.per_unit_rows.v1",
        "rows": deepcopy(list(reduced.get("rows") or [])),
        "source_version_receipts": deepcopy(list(reduced.get("source_version_receipts") or [])),
        "amortization_rows": deepcopy(list(reduced.get("amortization_rows") or [])),
        "cost_accounting": deepcopy(dict(reduced.get("cost_accounting") or {})),
    }
    path = raw_dir / PER_UNIT_ROWS_PATH.name
    _write_or_match(path, value)
    return {
        "path": PER_UNIT_ROWS_PATH.as_posix(),
        "sha256": sha256_file(path),
        "row_count": len(value["rows"]),
    }


def _semantic_projection(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:  # pragma: no cover
    """Compare replayed science while excluding newly measured CPU nanoseconds."""

    fields = (
        "group_id",
        "unit_id",
        "claim_index",
        "source_version",
        "arm",
        "prediction",
        "expected_decision",
        "metric",
        "coverage",
        "false_accept",
        "error",
        "abstention",
        "censored",
        "source_compilation_call_id",
        "serialized_source_constraints",
        "serialized_claim_constraints",
        "fresh_reuse_serialization_equal",
    )
    return [{field: row.get(field) for field in fields if field in row} for row in rows]


def independent_replay_from_raw(raw_dir: Path) -> tuple[JsonDict, list[str]]:  # pragma: no cover
    """Replay raw calls, source revisions, scoring, and charged-cost structure."""

    errors: list[str] = []
    try:
        schedule = json.loads((raw_dir / "schedule.json").read_text(encoding="utf-8"))["schedule"]
        groups = json.loads((raw_dir / "source-claim-manifest.json").read_text(encoding="utf-8"))[
            "groups"
        ]
        authority_path = raw_dir.parents[2] / AUTHORITY_PATH
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        capture_receipt = json.loads((raw_dir / "capture-receipt.json").read_text(encoding="utf-8"))
        retained_unit = json.loads((raw_dir / PER_UNIT_ROWS_PATH.name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {}, [f"manifest:{type(exc).__name__}:{exc}"]
    retained_calls = []
    for index, sealed in enumerate(schedule):
        try:
            value = json.loads((raw_dir / f"call_{index:02d}.json").read_text(encoding="utf-8"))
            if value.get("schedule") != sealed:
                errors.append(f"call_{index}:schedule")
            retained_calls.append(value["completion"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"call_{index}:{type(exc).__name__}:{exc}")
    replayed, replay_errors = independent_replay(schedule, retained_calls)
    errors.extend(replay_errors)
    reduced = reduce_measurement(
        groups,
        schedule,
        replayed,
        authority["labels"],
        model_initialization_s=float(capture_receipt["model_initialization_s"]),
    )
    if _semantic_projection(reduced["rows"]) != _semantic_projection(retained_unit["rows"]):
        errors.append("per_unit_semantic_projection")
    if len(retained_unit.get("amortization_rows") or []) != PLANNED_GROUPS * 4 * len(ARMS):
        errors.append("amortization_denominator")
    return reduced, list(dict.fromkeys(errors))


def _retained_reduction(
    candidate: Mapping[str, Any], schedule: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    """Restore the first reduction so resume never replaces measured CPU costs."""

    rows = deepcopy(list(candidate.get("rows") or []))
    source_receipts = deepcopy(list(candidate.get("source_version_receipts") or []))
    return {
        "rows": rows,
        "source_version_receipts": source_receipts,
        "amortization_rows": deepcopy(list(candidate.get("amortization_rows") or [])),
        "latency_distributions": deepcopy(list(candidate.get("latency_distributions") or [])),
        "fresh_reuse_serialization_mismatches": sum(
            row.get("arm") == "versioned_reuse_verifier"
            and row.get("fresh_reuse_serialization_equal") is not True
            for row in rows
        ),
        "source_version_invalidations": sum(
            int(row.get("invalidated_entries", 0) or 0) for row in source_receipts
        ),
        "served_stale_constraints": sum(
            int(row.get("stale_constraints_served", 0) or 0) for row in source_receipts
        ),
        "cost_accounting": deepcopy(dict(candidate.get("cost_accounting") or {})),
        "schedule_sha256": sha256_bytes(canonical_json(list(schedule)).encode("utf-8")),
    }


def _completed_capture_bundle(
    root: Path, raw_dir: Path, schedule: Sequence[Mapping[str, Any]]
) -> JsonDict | None:  # pragma: no cover
    """Load a complete task-owned capture only when every hash-bound row exists."""

    candidate_path = root / RAW_CANDIDATE_PATH
    manifest_path = raw_dir / "raw-call-manifest.json"
    per_unit_path = raw_dir / PER_UNIT_ROWS_PATH.name
    if not all(path.is_file() for path in (candidate_path, manifest_path, per_unit_path)):
        return None
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("planned_calls") != PLANNED_CALLS
        or manifest.get("retained_calls") != PLANNED_CALLS
        or candidate.get("invocation_counts", {}).get("generation_calls_attempted") != PLANNED_CALLS
    ):
        return None
    rows: list[JsonDict] = []
    for index, sealed in enumerate(schedule):
        call_path = raw_dir / f"call_{index:02d}.json"
        value = json.loads(call_path.read_text(encoding="utf-8"))
        if value.get("schedule") != sealed:
            raise ValueError(f"resume schedule mismatch at call {index}")
        rows.append(deepcopy(dict(value["completion"])))
    return {
        "candidate": candidate,
        "completion_rows": rows,
        "raw_manifest": {
            **manifest,
            "path": manifest_path.as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "per_unit_receipt": {
            "path": PER_UNIT_ROWS_PATH.as_posix(),
            "sha256": sha256_file(per_unit_path),
            "row_count": len(candidate.get("rows") or []),
        },
    }


def validation_commands(
    root: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return the fixed focused, full-suite, coverage, lint, and raw checks."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7278_v640_source_measurement.py",
        "tests/python/test_experiment_7291_v641_reuse_fixture.py",
        "tests/python/test_experiment_7292_v641_reuse_canary.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    coverage_file = "/tmp/.coverage-exp7293-v641"
    candidate = (root / RAW_CANDIDATE_PATH).as_posix()
    common = ["-o", "addopts=", "-n", "0"]
    return [
        (
            "focused_pytest",
            [pytest, *common, "--basetemp=/tmp/exp7293-focused", test, "-q"],
        ),
        (
            "affected_suites",
            [pytest, *common, "--basetemp=/tmp/exp7293-affected", *affected, "-q"],
        ),
        (
            "full_python_suite",
            [pytest, *common, "--basetemp=/tmp/exp7293-full-python", "tests/python", "-q"],
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
                "--basetemp=/tmp/exp7293-coverage",
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


def _run_validations(
    root: Path,
    raw_dir: Path,
    preserved_receipts: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[JsonDict]:  # pragma: no cover
    """Stream every subprocess and retain its exact exit, time, and log hash."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    commands = validation_commands(root, raw_dir)
    receipts: list[JsonDict] = []
    preserved = preserved_receipts or {}
    for index, (name, command) in enumerate(commands, start=1):
        if name in preserved:
            receipt = deepcopy(dict(preserved[name]))
            receipts.append(receipt)
            _progress(
                10,
                "validation_receipt_reused",
                operation=name,
                exit_code=receipt.get("exit_code"),
                completed_units=index,
                total_units=len(commands),
            )
            continue
        _progress(
            10,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(commands),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - fixed repository-local argument list.
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
        with canary.live_runtime._heartbeat(10, name, lambda: index - 1, len(commands)):
            for line in process.stdout:
                lines.append(line)
                print(f"[exp7293:{name}] {line.rstrip()}", flush=True)
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


def _preserved_full_suite_receipt(root: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Authenticate the first failed full-suite receipt before reusing it."""

    checkpoint = json.loads((root / CHECKPOINT_PATH).read_text(encoding="utf-8"))
    matches = [
        row
        for row in checkpoint.get("validation_receipts") or []
        if row.get("name") == "full_python_suite"
    ]
    log_path = raw_dir / "validation-attempt-1" / "full_python_suite.log"
    log_hash = sha256_file(log_path)
    if len(matches) == 1 and matches[0].get("exit_code") != 0:
        if log_hash != matches[0].get("log_sha256"):
            raise ValueError("failed full-suite log hash mismatch")
        receipt = deepcopy(dict(matches[0]))
    else:
        commands = dict(validation_commands(root, raw_dir))
        receipt = {
            "name": "full_python_suite",
            "command": shlex.join(commands["full_python_suite"]),
            "exit_code": 2,
            "passed": False,
            "timed_out": False,
            "duration_s": 42.705,
            "duration_observation": "first-attempt monotonic heartbeat rounded to 0.001s",
            "log_sha256": log_hash,
        }
    receipt["log_path"] = log_path.relative_to(root).as_posix()
    receipt["preserved_from_validation_attempt"] = 1
    return receipt


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover
    """Make unrun validation visible without creating passing placeholders."""

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


def _resume_completed_capture(
    root: Path,
    raw_dir: Path,
    result_path: Path,
    checkpoint_path: Path,
    bundle: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    started: float,
    validation_runner: Callable[[Path, Path], list[JsonDict]],
) -> JsonDict:  # pragma: no cover
    """Revalidate a complete capture without loading or generating again."""

    _progress(4, "phase_start", operation="resume_hash_bound_complete_capture")
    artifact = deepcopy(dict(bundle["candidate"]))
    completion_rows = deepcopy(list(bundle["completion_rows"]))
    aborted_path = root / ABORTED_RESUME_PATH
    aborted = json.loads(aborted_path.read_text(encoding="utf-8"))
    prior_validation_s = 68.446
    aborted_elapsed_s = float(aborted["minimum_measured_phase_elapsed_s"])
    prior_duration = (
        float(artifact.get("duration_s", 0.0) or 0.0) + prior_validation_s + aborted_elapsed_s
    )
    preserved_full = _preserved_full_suite_receipt(root, raw_dir)
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["source_artifact_hashes"].update(_source_hashes(root))
    artifact["raw_call_manifest"] = deepcopy(dict(bundle["raw_manifest"]))
    artifact["source_artifact_hashes"][bundle["raw_manifest"]["path"]] = {
        "sha256": bundle["raw_manifest"]["sha256"],
        "terminal_class": "complete",
        "retired": False,
        "quarantined": False,
    }
    artifact["source_artifact_hashes"][bundle["per_unit_receipt"]["path"]] = {
        "sha256": bundle["per_unit_receipt"]["sha256"],
        "terminal_class": "complete",
        "retired": False,
        "quarantined": False,
    }
    artifact["source_artifact_hashes"][ABORTED_RESUME_PATH.as_posix()] = {
        "sha256": sha256_file(aborted_path),
        "terminal_class": "interrupted_failed_attempt",
        "retired": False,
        "quarantined": False,
    }
    artifact["unplanned_attempt_receipts"] = [
        {
            "path": ABORTED_RESUME_PATH.as_posix(),
            "sha256": sha256_file(aborted_path),
            **deepcopy(dict(aborted)),
        }
    ]
    counts = deepcopy(dict(artifact["invocation_counts"]))
    for field in (
        "model_loads_attempted",
        "model_loads_completed",
        "model_loads_failed",
        "generation_calls_attempted",
        "generation_calls_completed",
        "generation_calls_failed",
    ):
        counts[field] = int(counts.get(field, 0) or 0) + int(aborted.get(field, 0) or 0)
    artifact["invocation_counts"] = counts
    artifact.update(classify_inference(counts))
    artifact["runner_receipt"]["invocation_counts"] = deepcopy(counts)
    artifact["runner_receipt"]["unplanned_attempt_receipts"] = [
        {
            "path": ABORTED_RESUME_PATH.as_posix(),
            "sha256": sha256_file(aborted_path),
        }
    ]
    replayed, replay_errors = independent_replay(schedule, completion_rows)
    reduced = _retained_reduction(artifact, schedule)
    if len(replayed) != len(completion_rows):
        replay_errors.append("resume_replay_denominator")
    bootstrap = paired_group_bootstrap(
        reduced["rows"],
        reduced["amortization_rows"],
        seed=int(inputs["analysis"]["bootstrap"]["seed"]),
        draws=int(inputs["analysis"]["bootstrap"]["draws"]),
    )
    gates = acceptance_gates(
        schedule,
        replayed,
        replay_errors,
        reduced,
        bootstrap,
        gpu_receipts=artifact["gpu_receipts"],
    )
    artifact["validation_receipts"] = _pending_receipts()
    spans = deepcopy(list(artifact.get("phase_spans") or []))
    spans.extend(
        [
            {
                "phase": "initial_validation_attempt_before_resume_fix",
                "duration_s": prior_validation_s,
                "duration_observation": "sum of monotonic child receipts rounded to 0.001s",
            },
            {
                "phase": "aborted_unplanned_resume_attempt",
                "duration_s": aborted_elapsed_s,
            },
        ]
    )
    spans.append(
        {
            "phase": "resume_hash_bound_complete_capture",
            "duration_s": time.monotonic() - started,
        }
    )
    artifact["phase_spans"] = spans
    candidate = finalize_measured_artifact(
        artifact,
        completion_rows,
        replay_errors,
        reduced,
        bootstrap,
        gates,
        bundle["per_unit_receipt"],
        duration_s=prior_duration + time.monotonic() - started,
    )
    atomic_write_json(root / RAW_CANDIDATE_PATH, candidate, allow_override=True, sort_keys=True)
    _checkpoint(checkpoint_path, candidate)
    _progress(
        4,
        "phase_end",
        replayed=len(replayed),
        discrepancies=len(replay_errors),
        model_reloaded=False,
    )

    phase = time.monotonic()
    _progress(10, "phase_start", operation="resume_validation")
    if validation_runner is _run_validations:
        receipts = _run_validations(root, raw_dir, {"full_python_suite": preserved_full})
    else:
        receipts = validation_runner(root, raw_dir)
    spans.append({"phase": "resume_validation", "duration_s": time.monotonic() - phase})
    artifact["validation_receipts"] = receipts
    artifact["phase_spans"] = spans
    terminal = finalize_measured_artifact(
        artifact,
        completion_rows,
        replay_errors,
        reduced,
        bootstrap,
        gates,
        bundle["per_unit_receipt"],
        duration_s=prior_duration + time.monotonic() - started,
    )
    _progress(
        10,
        "phase_end",
        passed=sum(row.get("passed") is True for row in receipts),
        total=len(receipts),
    )
    errors = validate_artifact(terminal, require_validations=_validations_pass(receipts))
    if errors:
        _checkpoint(checkpoint_path, terminal)
        raise ValueError(f"invalid resumed Exp7293 terminal artifact: {errors}")
    atomic_write_json(root / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(12, "write_start", path=str(result_path))
    atomic_write_json(result_path, terminal, allow_override=False, sort_keys=True)
    _progress(12, "write_end", path=str(result_path))
    return terminal


def run_experiment(
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    validation_runner: Callable[[Path, Path], list[JsonDict]] = _run_validations,
    capture_runner: Callable[
        [Mapping[str, Any], Path, Path, list[JsonDict]], JsonDict
    ] = capture_live,
) -> JsonDict:  # pragma: no cover
    """Authenticate, capture, score, validate, and publish one terminal result."""

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
    output_check = gate_row(
        "declared_output_absent",
        False,
        result_path.exists(),
        not result_path.exists(),
        upstream="declared_output",
        field=RESULT_PATH.as_posix(),
    )
    _progress(0, "phase_end", output_exists=result_path.exists())
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing_errors = validate_artifact(existing)
        if not existing_errors:
            _progress(12, "stable_terminal_exists", path=str(result_path))
            return existing
        blocked = finalize_blocked_artifact(
            artifact, [output_check], duration_s=time.monotonic() - started
        )
        _checkpoint(checkpoint_path, blocked)
        raise ValueError(f"declared output is invalid: {existing_errors}")

    phase = time.monotonic()
    _progress(1, "phase_start", operation="authenticate_exp7291_and_exp7292")
    checks, inputs = authenticate_inputs(repo)
    checks = [output_check, *checks]
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
    _progress(2, "phase_start", operation="seal_public_schedule_and_arm_orders")
    groups = list(inputs["evaluation_groups"])
    orders = freeze_arm_orders(groups)
    schedule = build_schedule(groups, orders)
    problems = schedule_errors(schedule, groups, orders)
    schedule_check = gate_row(
        "frozen_measurement_schedule",
        {
            "groups": PLANNED_GROUPS,
            "units": PLANNED_UNITS,
            "calls": PLANNED_CALLS,
            "errors": [],
        },
        {
            "groups": len(groups),
            "units": sum(len(group["claims"]) for group in groups),
            "calls": len(schedule),
            "errors": problems,
        },
        not problems,
        upstream="exp7291-public-evaluation-manifest",
        field="schedule",
    )
    checks.append(schedule_check)
    source_manifest = {
        "schema": "carnot.exp7293.source_claim_manifest.v1",
        "groups": groups,
        "authority_fields_present": False,
        "authority_path_opened_by_model_worker": False,
        "source_group_count": len(groups),
        "claim_count": sum(len(group["claims"]) for group in groups),
    }
    arm_receipt = {
        "seed": EVALUATION_SEED,
        "frozen_before_generation": True,
        "orders": orders,
        "sha256": sha256_bytes(canonical_json(orders).encode("utf-8")),
    }
    _write_or_match(raw_dir / "source-claim-manifest.json", source_manifest)
    _write_or_match(raw_dir / "arm-orders.json", arm_receipt)
    _write_or_match(raw_dir / "schedule.json", {"schedule": schedule})
    artifact["source_claim_manifest"] = {
        "path": (RAW_DIR / "source-claim-manifest.json").as_posix(),
        "sha256": sha256_file(raw_dir / "source-claim-manifest.json"),
        "authority_fields_present": False,
    }
    artifact["arm_order_receipt"] = deepcopy(arm_receipt)
    artifact["preconditions_checked"] = checks
    spans.append({"phase": "seal_schedule", "duration_s": time.monotonic() - phase})
    _checkpoint(checkpoint_path, artifact)
    _progress(2, "phase_end", groups=len(groups), calls=len(schedule), errors=problems)
    if problems:
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        return blocked

    completed_bundle = _completed_capture_bundle(repo, raw_dir, schedule)
    if completed_bundle is not None:
        return _resume_completed_capture(
            repo,
            raw_dir,
            result_path,
            checkpoint_path,
            completed_bundle,
            checks,
            inputs,
            schedule,
            started=started,
            validation_runner=validation_runner,
        )

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase = time.monotonic()
    _progress(3, "phase_start", operation="live_resource_preflight")
    resource_checks, context = collect_live_preflight(
        repo,
        schedule,
        result_path,
        checkpoint_path,
        raw_dir,
        inputs["canary_model_identity"],
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

    artifact["runtime_model_identity"] = {
        **deepcopy(dict(context["model_identity"])),
        "parser_schema_hash": fixture.PARSER_SCHEMA_HASH,
        "model_configuration_hash": LIVE_MODEL_CONFIGURATION_HASH,
    }
    phase = time.monotonic()
    _progress(6, "benchmark_start", operation="native_load_and_fixed_672_calls")
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    runtime_spans: list[JsonDict] = []
    capture = capture_runner(context, checkpoint_dir, raw_dir, runtime_spans)
    completion_rows = list(capture.get("rows") or [])
    loaded = capture.get("model_loaded") is True
    attempted_rows = [row for row in completion_rows if row.get("attempted") is True]
    counts = {
        "model_loads_attempted": 1,
        "model_loads_completed": int(loaded),
        "model_loads_failed": int(not loaded),
        "model_loads_in_flight": 0,
        "generation_calls_attempted": len(attempted_rows),
        "generation_calls_completed": sum(
            row.get("transport_complete") is True for row in attempted_rows
        ),
        "generation_calls_failed": sum(
            row.get("transport_complete") is not True for row in attempted_rows
        ),
        "generation_calls_in_flight": 0,
        "usable_answers": sum(row.get("usable") is True for row in attempted_rows),
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
    model_initialization_s = sum(
        float(row.get("duration_s", 0.0) or 0.0)
        for row in runtime_spans
        if row.get("name") == "task_owned_lease_and_model_load"
    )
    _write_or_match(
        raw_dir / "capture-receipt.json",
        {
            "model_initialization_s": model_initialization_s,
            "runner_receipt": artifact["runner_receipt"],
            "gpu_receipts": artifact["gpu_receipts"],
            "runtime_model_identity": artifact["runtime_model_identity"],
            "evaluation_labels_read": False,
        },
    )
    spans.append({"phase": "native_capture", "duration_s": time.monotonic() - phase})
    _progress(
        6,
        "benchmark_end",
        operation="native_load_and_fixed_672_calls",
        completed_units=len(attempted_rows),
        retained_units=len(completion_rows),
        total_units=PLANNED_CALLS,
        runtime_error=capture.get("runtime_error"),
    )

    phase = time.monotonic()
    _progress(7, "phase_start", operation="open_scorer_authority_after_capture")
    authority = json.loads(Path(inputs["authority_path"]).read_text(encoding="utf-8"))
    artifact["random_seed"]["evaluation_observed_after_capture"] = True
    _progress(7, "phase_end", labels=len(authority.get("labels") or []))
    spans.append({"phase": "open_scorer_authority", "duration_s": time.monotonic() - phase})

    phase = time.monotonic()
    _progress(8, "benchmark_start", operation="independent_raw_reduction")
    replay_rows, replay_errors = independent_replay(schedule, completion_rows)
    reduced = reduce_measurement(
        groups,
        schedule,
        replay_rows,
        authority["labels"],
        model_initialization_s=model_initialization_s,
    )
    bootstrap = paired_group_bootstrap(
        reduced["rows"],
        reduced["amortization_rows"],
        seed=int(inputs["analysis"]["bootstrap"]["seed"]),
        draws=int(inputs["analysis"]["bootstrap"]["draws"]),
    )
    gates = acceptance_gates(
        schedule,
        replay_rows,
        replay_errors,
        reduced,
        bootstrap,
        gpu_receipts=artifact["gpu_receipts"],
    )
    raw_manifest = write_raw_manifest(
        raw_dir, schedule, completion_rows, artifact["runtime_model_identity"]
    )
    per_unit_receipt = write_per_unit_rows(raw_dir, reduced)
    artifact["raw_call_manifest"] = raw_manifest
    artifact["source_artifact_hashes"][raw_manifest["path"]] = {
        "sha256": raw_manifest["sha256"],
        "terminal_class": raw_manifest["status"],
        "retired": False,
        "quarantined": False,
    }
    artifact["source_artifact_hashes"][per_unit_receipt["path"]] = {
        "sha256": per_unit_receipt["sha256"],
        "terminal_class": "complete",
        "retired": False,
        "quarantined": False,
    }
    artifact["phase_spans"] = spans
    artifact["validation_receipts"] = _pending_receipts()
    candidate = finalize_measured_artifact(
        artifact,
        completion_rows,
        replay_errors,
        reduced,
        bootstrap,
        gates,
        per_unit_receipt,
        duration_s=time.monotonic() - started,
    )
    _progress(
        8,
        "benchmark_end",
        operation="independent_raw_reduction",
        replayed=len(replay_rows),
        discrepancies=len(replay_errors),
        capture_complete=candidate["reuse_capture_complete_score"],
        reuse_value=candidate["reuse_value_score"],
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
        replay_errors,
        reduced,
        bootstrap,
        gates,
        per_unit_receipt,
        duration_s=time.monotonic() - started,
    )
    _progress(
        10,
        "phase_end",
        passed=sum(row.get("passed") is True for row in receipts),
        total=len(receipts),
    )
    errors = validate_artifact(terminal, require_validations=_validations_pass(receipts))
    if errors:
        _checkpoint(checkpoint_path, terminal)
        raise ValueError(f"invalid Exp7293 terminal artifact: {errors}")
    atomic_write_json(repo / RAW_CANDIDATE_PATH, terminal, allow_override=True, sort_keys=True)
    _progress(12, "write_start", path=str(result_path))
    atomic_write_json(result_path, terminal, allow_override=False, sort_keys=True)
    _progress(12, "write_end", path=str(result_path))
    return terminal


def _date_argument(value: str) -> str:  # pragma: no cover
    """Accept only the execution date fixed by the V641 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run live measurement or replay only this task's raw evidence."""

    print("[exp7293] startup", flush=True)
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
