"""Measure source interventions with unchanged mention extraction and direct judgment.

The module keeps public generation inputs separate from private grading. It
reuses the shipped mention compiler, repaired direct parser, and native GPU
lease runtime.

Spec refs: REQ-VERIFY-7278 and SCENARIO-VERIFY-7278-*.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import time
from typing import Any, Callable

from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7236_v637_mention_fixture as mention_fixture
from carnot import experiment_7237_v637_mention_canary as extraction
from carnot import experiment_7238_v637_mention_capture as capture
from carnot import experiment_7275_v640_semantic_replay as semantic_replay
from carnot import experiment_7277_v640_comparator_canary as comparator_canary
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260913"
MILESTONE = "2026.09.640"
EXPERIMENT_ID = "exp7278-source-measurement"
TASK_ID = "experiment_7278_v640_source_measurement"
SCHEMA = "carnot.exp7278.v640_source_measurement.v1"
RANDOM_SEED = 727_820_260_913
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

PLANNED_BASE_GROUPS = 16
PLANNED_UNITS = 64
PLANNED_CALLS = 256
PLANNED_RESULT_ROWS = 128
OUTPUT_TOKEN_BUDGET = 128
REQUEST_CAP_S = 90.0
MODEL_LOAD_CAP_S = 600.0
GENERATION_CAP_S = 2400.0
CONDITIONS = (
    "supported",
    "relation_reversal",
    "insufficient_evidence",
    "consistent_entity_renaming",
)
ARMS = ("mention_pointer", "direct_self_consistency")

UPSTREAM_REPLAY_PATH = Path("results/experiment_7275_v640_semantic_replay.json")
UPSTREAM_CANARY_PATH = Path("results/experiment_7277_v640_comparator_canary.json")
UPSTREAM_CONTRACT_PATH = Path("results/raw/experiment_7275/comparator_contract.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7278_v640_source_measurement.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7278_v640_source_measurement.py")
TEST_PATH = Path("tests/python/test_experiment_7278_v640_source_measurement.py")
RESULT_PATH = Path("results/experiment_7278_v640_source_measurement.json")
RAW_DIR = Path("results/raw/experiment_7278")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7278_v640_source_measurement.json")
PUBLIC_MANIFEST_PATH = RAW_DIR / "public_manifest.json"
PRIVATE_AUTHORITY_PATH = RAW_DIR / "private_authority_manifest.json"

PINNED_INPUT_HASHES = {
    UPSTREAM_REPLAY_PATH: "sha256:10fa70a3f63846fe3a149f53a4a39fcedee6cfaf7163f709d418602c63dd3885",
    UPSTREAM_CANARY_PATH: "sha256:f5f7b6b4c7e5c8a0fd1d1312d8b3cb247d3ec0943df2197614a28c6c2cd25a93",
    UPSTREAM_CONTRACT_PATH: "sha256:119ebc8154212fd2f5b5373bacb66723f765f299bf932939127263fa9ae7820a",
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
    "inference_substrate_class": "Use the actual full-generation, load-only, or no-model class; never pad duration.",
    "inference_mode": "Use live_gpu only after actual generation begins.",
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
    "verdict_class": "Use the closed verdict set; oracle evidence cannot be positive and only unfinished work is partial.",
    "validation_receipts": "Retain command, exit code, timing, and log hash; do not hide failures.",
    "source_capture_complete_score": "Complete fixed-denominator authenticated measurement with zero replay disagreement, independent of efficacy.",
    "source_value_score": "Descriptive primary value gate; final independent confirmation belongs to the audit.",
    "raw_call_manifest": "Keep one authentic outcome or explicit missing or censored record for each scheduled call.",
    "source_fidelity_rows": "Keep all 64 units for every arm and separate fidelity from final decision.",
    "runner_receipt": "Retain current model identity, transport settings, GPU ownership, and server ownership.",
    "source_intervention_rows": "Retain paired renaming, reversal, missing-source, and shuffled-source controls.",
    "public_manifest_path": "Point to the frozen public input manifest with no private labels.",
    "private_authority_manifest_path": "Point to evaluator-only authority excluded from generation.",
    "representation_metric_rows": "Report representation validity and decision outcomes separately for each arm.",
    "token_latency_rows": "Retain per-call tokens, latency, errors, and censoring without filtering.",
    "selection_receipt": "Bind the fresh roster, seeds, arm order, and shuffle before inference.",
    "replay_rows": "Independently reconstruct every raw outcome before private scoring.",
    "replay_discrepancies": "Keep every request, response, parser, or roster mismatch visible.",
    "model_identity_receipt": "Pin the current GGUF revision, content hash, tokenizer, and chat template.",
    "gpu_receipts": "Bind generation to the owned server process, GPU UUID, lease, samples, and cleanup.",
    "feasibility_projection": "Project the fixed 256-call cost from canary timing before model launch.",
    "timestamps": "Retain actual UTC start and end observations.",
    "phase_spans": "Retain measured disjoint phase durations.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

AUTHORITY_ONLY_FIELDS = {
    "condition",
    "expected_decision",
    "gold_source_completion",
    "gold_claim_completion",
    "authority_parser",
    "game_source",
}

GATE_PRINCIPLES = {
    "outcome_accounting": "All 256 scheduled outcomes must remain in the fixed denominator.",
    "authentic_joins": "Each outcome must have authentic bytes or an explicit authenticated censoring record.",
    "replay_discrepancies": "Independent reduction must reproduce every retained outcome without mismatch.",
    "authority_separation": "No private label, game source, or authority parser may enter a model request.",
    "equal_budget": "Both arms receive the same 128-token maximum and time limit.",
    "source_value": "The unchanged pointer arm must exceed the direct self-consistency comparator to pass the descriptive value gate.",
    "focused_validation": "Every fixed focused checker must pass before publication.",
}

canonical_json = mention_fixture.canonical_json
sha256_bytes = mention_fixture.sha256_bytes
request_payload = live_runtime._request_payload


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without parsing durable evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def sha256_json(value: Any) -> str:
    """Hash one value with the stable JSON spelling used by native requests."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def gate_row(
    check: str,
    expected: Any,
    observed: Any,
    passed: bool,
    *,
    upstream: str,
    field: str,
) -> JsonDict:
    """Keep the exact comparison that supports one precondition decision."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
        "upstream": upstream,
        "field": field,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project the first failed prerequisite without hiding later failures."""

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


def _unit_id(group_index: int, condition_index: int) -> str:
    """Create fresh opaque IDs that do not reveal a condition name."""

    value = f"{RANDOM_SEED}:fresh-source:{group_index}:{condition_index}"
    return f"n-{hashlib.sha256(value.encode()).hexdigest()[:20]}"


def _names(group_index: int, renamed: bool) -> tuple[str, str, str]:
    """Create deterministic disjoint surfaces for original and renamed controls."""

    prefix = "R" if renamed else "K"
    start = group_index * 3
    return tuple(prefix + mention_fixture._letters(start + offset) for offset in range(3))  # type: ignore[return-value]


def build_source_fixture() -> tuple[JsonDict, JsonDict]:
    """Build 16 fresh groups while keeping expected decisions private."""

    public_rows: list[JsonDict] = []
    authority_rows: list[JsonDict] = []
    predicates = ("precedes", "starts before", "ends before", "occurs before")
    for group_index in range(PLANNED_BASE_GROUPS):
        predicate = predicates[group_index % len(predicates)]
        for condition_index, condition in enumerate(CONDITIONS):
            renamed = condition == "consistent_entity_renaming"
            subject, obj, missing = _names(group_index, renamed)
            source_text = f"{subject} {predicate} {obj}."
            if condition == "relation_reversal":
                claim_text = f"{obj} {predicate} {subject}."
                expected = "contradicted"
                claim_relation = (obj, predicate, subject, "positive", 0)
            elif condition == "insufficient_evidence":
                claim_text = f"{subject} {predicate} {missing}."
                expected = "unknown"
                claim_relation = (subject, predicate, missing, "positive", 0)
            else:
                claim_text = f"{subject} {predicate} {obj}."
                expected = "supported"
                claim_relation = (subject, predicate, obj, "positive", 0)
            unit_id = _unit_id(group_index, condition_index)
            source = mention_fixture._document(f"{unit_id}-source", source_text)
            claim = mention_fixture._document(f"{unit_id}-claim", claim_text)
            public_rows.append(
                {
                    "unit_id": unit_id,
                    "base_group_id": f"g-{group_index:02d}",
                    "source": source,
                    "claim": claim,
                }
            )
            authority_rows.append(
                {
                    "unit_id": unit_id,
                    "base_group_id": f"g-{group_index:02d}",
                    "condition": condition,
                    "expected_decision": expected,
                    "gold_source_completion": mention_fixture._gold_completion(
                        source, [(subject, predicate, obj, "positive", 0)]
                    ),
                    "gold_claim_completion": mention_fixture._gold_completion(
                        claim, [claim_relation]
                    ),
                    "model_worker_input": False,
                    "authority_parser": "mention_fixture.execute_pointer_pair",
                    "game_source": None,
                }
            )
    public = {
        "schema": "carnot.exp7278.public_source_measurement.v1",
        "fresh_source_groups": PLANNED_BASE_GROUPS,
        "unit_count": len(public_rows),
        "private_fields_present": False,
        "rows": public_rows,
    }
    authority = {
        "schema": "carnot.exp7278.private_source_authority.v1",
        "random_seed": RANDOM_SEED,
        "unit_count": len(authority_rows),
        "condition_counts": dict(
            sorted(Counter(row["condition"] for row in authority_rows).items())
        ),
        "model_worker_input": False,
        "rows": authority_rows,
    }
    return public, authority


def _paired_arm_orders() -> list[tuple[str, str]]:
    """Freeze paired arm order before any generated answer exists."""

    generator = random.Random(RANDOM_SEED)
    return [
        ARMS if generator.getrandbits(1) == 0 else tuple(reversed(ARMS))
        for _ in range(PLANNED_UNITS)
    ]


def _source_shuffle_permutation(unit_ids: Sequence[str]) -> dict[str, str]:
    """Freeze a deranged source mapping for controls that need no new call."""

    shuffled = list(unit_ids)
    random.Random(RANDOM_SEED + 1).shuffle(shuffled)
    rotated = shuffled[1:] + shuffled[:1]
    return dict(zip(shuffled, rotated, strict=True))


def _append_call(
    schedule: list[JsonDict],
    *,
    unit: Mapping[str, Any],
    arm: str,
    call_type: str,
    draw_index: int | None,
) -> None:
    """Append one public call with equal budgets and an independent fixed seed."""

    order = len(schedule)
    seed = RANDOM_SEED + 10_000 + order
    if arm == "mention_pointer":
        document = deepcopy(dict(unit[call_type]))
        prompt = extraction._prompt("mention_pointer", document, call_type)
        grammar = extraction.compile_grammar("mention_pointer", document, call_type)["grammar"]
    else:
        document = {"source": deepcopy(unit["source"]), "claim": deepcopy(unit["claim"])}
        prompt = semantic_replay.DIRECT_PROMPT.format(
            source=unit["source"]["text"], claim=unit["claim"]["text"]
        )
        grammar = semantic_replay.DIRECT_GRAMMAR
    schedule.append(
        {
            "call_order": order,
            "call_id": f"exp7278:{unit['unit_id']}:{arm}:{call_type}:{draw_index}",
            "unit_id": str(unit["unit_id"]),
            "base_group_id": str(unit["base_group_id"]),
            "arm": arm,
            "call_type": call_type,
            "draw_index": draw_index,
            "document": document,
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "grammar": grammar,
            "grammar_sha256": sha256_bytes(grammar.encode("utf-8")),
            "grammar_requested": True,
            "output_token_budget": OUTPUT_TOKEN_BUDGET,
            "request_timeout_s": REQUEST_CAP_S,
            "seed": seed,
            "decoding_parameters": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": seed,
                "cache_prompt": False,
            },
            "retry_budget": 0,
            "adaptive_stopping": False,
        }
    )


def build_schedule(public_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze exactly four equal-budget calls for each fresh public unit."""

    if len(public_rows) != PLANNED_UNITS:
        raise ValueError("schedule requires 64 public units")
    unit_ids = [str(row.get("unit_id")) for row in public_rows]
    if len(set(unit_ids)) != PLANNED_UNITS:
        raise ValueError("schedule requires unique public units")
    schedule: list[JsonDict] = []
    for unit, arm_order in zip(public_rows, _paired_arm_orders(), strict=True):
        for arm in arm_order:
            if arm == "mention_pointer":
                _append_call(
                    schedule,
                    unit=unit,
                    arm=arm,
                    call_type="source",
                    draw_index=None,
                )
                _append_call(
                    schedule,
                    unit=unit,
                    arm=arm,
                    call_type="claim",
                    draw_index=None,
                )
            else:
                for draw_index in range(2):
                    _append_call(
                        schedule,
                        unit=unit,
                        arm=arm,
                        call_type="direct",
                        draw_index=draw_index,
                    )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], public_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Name denominator, public-byte, order, budget, or authority leakage changes."""

    try:
        expected = build_schedule(public_rows)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != PLANNED_CALLS:
        errors.append("call_denominator")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field, expected_value in wanted.items():
            if observed.get(field) != expected_value:
                errors.append(f"call_{index}:{field}")
        if set(observed) & AUTHORITY_ONLY_FIELDS:
            errors.append(f"call_{index}:authority_leakage")
    return list(dict.fromkeys(errors))


def selection_receipt(
    public: Mapping[str, Any],
    authority: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Bind the roster and controls before inference without exposing authority."""

    public_ids = [str(row["unit_id"]) for row in public["rows"]]
    authority_ids = [str(row["unit_id"]) for row in authority["rows"]]
    exposed_ids = {
        mention_fixture._unit_id("held_out", index)
        for index in range(mention_fixture.SPLIT_COUNTS["held_out"])
    }
    schedule_text = canonical_json(list(schedule))
    return {
        "roster_frozen_before_inference": True,
        "public_roster_sha256": sha256_json(public_ids),
        "public_manifest_sha256": sha256_json(public),
        "private_authority_sha256": sha256_json(authority),
        "schedule_sha256": sha256_json(list(schedule)),
        "selected_unit_count": len(public_ids),
        "public_private_ids_match": public_ids == authority_ids,
        "authority_fields_in_schedule": sum(
            any(field in row for field in AUTHORITY_ONLY_FIELDS) for row in schedule
        ),
        "private_terms_in_schedule": [
            term
            for term in ("authority_parser", "expected_decision", "game_source")
            if term in schedule_text
        ],
        "exp7265_held_out_overlap": sorted(set(public_ids) & exposed_ids),
        "paired_arm_orders": [list(value) for value in _paired_arm_orders()],
        "unit_seeds": [int(row["seed"]) for row in schedule],
        "source_shuffle_permutation": _source_shuffle_permutation(public_ids),
        "adaptive_stopping": False,
        "held_out_rows_used_for_training_or_tuning": False,
    }


def _decode_b64(value: Any) -> bytes:
    """Decode retained transport bytes strictly so corruption stays visible."""

    if not isinstance(value, str):
        raise ValueError("base64_type")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64_invalid") from exc


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Preserve one model outcome with the shipped arm-specific parser."""

    if sealed.get("arm") == "mention_pointer":
        row = capture.build_completion_row(sealed, response, resource)
    else:
        row = comparator_canary.build_completion_row(sealed, response, resource)
    transport_error = response.get("error")
    row.update(
        {
            "call_type": sealed["call_type"],
            "draw_index": sealed.get("draw_index"),
            "attempted": True,
            "censored": False,
            "timeout": bool(transport_error and "timeout" in str(transport_error).lower()),
            "transport_error": transport_error,
            "schedule_row_sha256": sha256_json(dict(sealed)),
        }
    )
    row["row_sha256"] = sha256_json(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )
    return row


def censored_completion(sealed: Mapping[str, Any], reason: str) -> JsonDict:
    """Account for a deadline-missing call without inventing response bytes."""

    row = {
        "call_order": sealed["call_order"],
        "call_id": sealed["call_id"],
        "unit_id": sealed["unit_id"],
        "arm": sealed["arm"],
        "call_type": sealed["call_type"],
        "draw_index": sealed.get("draw_index"),
        "seed": sealed["seed"],
        "attempted": False,
        "censored": True,
        "censoring_reason": reason,
        "terminal_state": "censored",
        "transport_complete": False,
        "parse_valid": False,
        "usable": False,
        "abstention": True,
        "error": reason,
        "raw_request_bytes_b64": "",
        "raw_response_bytes_b64": "",
        "request_bytes_sha256": None,
        "response_bytes_sha256": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "schedule_row_sha256": sha256_json(dict(sealed)),
    }
    row["row_sha256"] = sha256_json(row)
    return row


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:
    """Rebuild every retained parser outcome from raw bytes without inference."""

    rows: list[JsonDict] = []
    errors: list[str] = []
    if len(schedule) != len(retained_rows):
        errors.append("replay_denominator")
    for index, sealed in enumerate(schedule):
        if index >= len(retained_rows):
            errors.append(f"call_{index}:missing_outcome")
            continue
        retained = retained_rows[index]
        if retained.get("call_id") != sealed.get("call_id"):
            errors.append(f"call_{index}:call_id")
        if retained.get("schedule_row_sha256") != sha256_json(dict(sealed)):
            errors.append(f"call_{index}:schedule_row")
        if retained.get("censored") is True:
            valid_censor = bool(
                retained.get("attempted") is False
                and retained.get("terminal_state") == "censored"
                and retained.get("censoring_reason")
            )
            if not valid_censor:
                errors.append(f"call_{index}:censoring_record")
            rows.append(
                {
                    "call_order": index,
                    "call_id": sealed["call_id"],
                    "unit_id": sealed["unit_id"],
                    "arm": sealed["arm"],
                    "call_type": sealed["call_type"],
                    "draw_index": sealed.get("draw_index"),
                    "seed": sealed["seed"],
                    "request_response_joined": False,
                    "explicit_censoring_authenticated": valid_censor,
                    "censored": True,
                    "parse_valid": False,
                    "usable": False,
                    "decision": None,
                    "compiled_completion": None,
                    "error": retained.get("censoring_reason"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": 0.0,
                }
            )
            continue
        try:
            request_bytes = _decode_b64(retained.get("raw_request_bytes_b64"))
            response_bytes = _decode_b64(retained.get("raw_response_bytes_b64"))
            response_body, content = semantic_replay._response_content(response_bytes)
        except ValueError as exc:
            errors.append(f"call_{index}:{exc}")
            continue
        expected_payload, expected_request = request_payload(sealed)
        if request_bytes != expected_request:
            errors.append(f"call_{index}:request_bytes")
        if retained.get("actual_parameters") != expected_payload:
            errors.append(f"call_{index}:actual_parameters")
        if "raw_response" in retained and retained.get("raw_response") != response_body:
            errors.append(f"call_{index}:raw_response")
        if retained.get("raw_completion") != content:
            errors.append(f"call_{index}:raw_completion")
        if sealed.get("arm") == "mention_pointer":
            replay_input = dict(retained)
            replay_input["call_id"] = sealed["call_id"]
            rebuilt = capture.replay_completion_rows([sealed], [replay_input])[0]
            decision = None
            compiled = deepcopy(rebuilt.get("compiled_completion"))
            parser_classification = None
        else:
            reduced = semantic_replay._reduce_direct_fixture(
                expected_request,
                request_bytes,
                content,
                str(retained.get("finish_reason") or ""),
            )
            rebuilt = {
                "parse_valid": reduced["accepted"] is True,
                "usable": reduced["accepted"] is True,
                "decision": reduced.get("decision"),
                "parser_classification": reduced["classification"],
            }
            decision = reduced.get("decision")
            compiled = None
            parser_classification = reduced["classification"]
        for field in ("parse_valid", "usable"):
            if retained.get(field) is not rebuilt.get(field):
                errors.append(f"call_{index}:{field}")
        if sealed.get("arm") == "mention_pointer" and retained.get(
            "compiled_completion"
        ) != rebuilt.get("compiled_completion"):
            errors.append(f"call_{index}:compiled_completion")
        if sealed.get("arm") == "direct_self_consistency":
            for field in ("decision", "parser_classification"):
                if retained.get(field) != rebuilt.get(field):
                    errors.append(f"call_{index}:{field}")
        rows.append(
            {
                "call_order": index,
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "arm": sealed["arm"],
                "call_type": sealed["call_type"],
                "draw_index": sealed.get("draw_index"),
                "seed": sealed["seed"],
                "request_response_joined": request_bytes == expected_request,
                "explicit_censoring_authenticated": False,
                "censored": False,
                "parse_valid": rebuilt.get("parse_valid") is True,
                "usable": rebuilt.get("usable") is True,
                "decision": decision,
                "compiled_completion": compiled,
                "parser_classification": parser_classification,
                "error": retained.get("error"),
                "request_bytes_sha256": sha256_bytes(request_bytes),
                "response_bytes_sha256": sha256_bytes(response_bytes),
                "prompt_tokens": int(retained.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(retained.get("completion_tokens", 0) or 0),
                "latency_s": float(retained.get("latency_s", 0.0) or 0.0),
            }
        )
    return rows, list(dict.fromkeys(errors))


def _decision_from_pointer(
    public: Mapping[str, Any], source_row: Mapping[str, Any], claim_row: Mapping[str, Any]
) -> JsonDict:
    """Execute captured public pointers without repairing an invalid extraction."""

    source_completion = source_row.get("compiled_completion")
    claim_completion = claim_row.get("compiled_completion")
    if not isinstance(source_completion, Mapping) or not isinstance(claim_completion, Mapping):
        return {"decision": "unknown", "errors": ["missing_or_invalid_extraction"]}
    return mention_fixture._execute_compiled_pair(
        public["source"], source_completion, claim_completion
    )


def _direct_decision(draws: Sequence[Mapping[str, Any]]) -> tuple[str, str, bool]:
    """Apply the frozen two-draw rule and keep the first draw as one-shot."""

    values = [
        str(row.get("decision"))
        if row.get("parse_valid") is True
        and row.get("decision") in semantic_replay.DIRECT_DECISIONS
        else "unknown"
        for row in draws
    ]
    while len(values) < 2:
        values.append("unknown")
    tied = values[0] != values[1]
    return ("unknown" if tied else values[0], values[0], tied)


def _false_accept(expected: str, predicted: str) -> bool:
    """Count a confident wrong decision separately from an abstention."""

    return predicted != "unknown" and predicted != expected


def reduce_measurement(
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Grade both arms and derive source controls from the same captured rows."""

    del completion_rows  # Raw cost evidence remains in replay_rows and the call manifest.
    public_by_id = {str(row["unit_id"]): row for row in public_rows}
    authority_by_id = {str(row["unit_id"]): row for row in authority_rows}
    calls_by_unit: dict[str, list[Mapping[str, Any]]] = {}
    for row in replay_rows:
        calls_by_unit.setdefault(str(row["unit_id"]), []).append(row)
    result_rows: list[JsonDict] = []
    fidelity_rows: list[JsonDict] = []
    pointer_predictions: dict[str, str] = {}
    for public in public_rows:
        unit_id = str(public["unit_id"])
        hidden = authority_by_id[unit_id]
        calls = calls_by_unit.get(unit_id, [])
        source = next(
            (
                row
                for row in calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "source"
            ),
            {},
        )
        claim = next(
            (
                row
                for row in calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "claim"
            ),
            {},
        )
        executed = _decision_from_pointer(public, source, claim)
        pointer_prediction = str(executed.get("decision") or "unknown")
        pointer_predictions[unit_id] = pointer_prediction
        gold_source = mention_fixture.compile_pointer_completion(
            public["source"], hidden["gold_source_completion"], "source"
        )
        gold_claim = mention_fixture.compile_pointer_completion(
            public["claim"], hidden["gold_claim_completion"], "claim"
        )
        source_fidelity = source.get("compiled_completion") == gold_source
        claim_fidelity = claim.get("compiled_completion") == gold_claim
        expected = str(hidden["expected_decision"])
        pointer_row = {
            "unit_id": unit_id,
            "base_group_id": hidden["base_group_id"],
            "condition": hidden["condition"],
            "arm": "mention_pointer",
            "seed": [source.get("seed"), claim.get("seed")],
            "prediction": pointer_prediction,
            "one_shot_decision": None,
            "self_consistency_tie": None,
            "expected_decision": expected,
            "representation_valid": bool(
                source.get("parse_valid") is True and claim.get("parse_valid") is True
            ),
            "source_fidelity": source_fidelity,
            "claim_fidelity": claim_fidelity,
            "decision_correct": pointer_prediction == expected,
            "unknown": pointer_prediction == "unknown",
            "false_accept": _false_accept(expected, pointer_prediction),
            "errors": list(executed.get("errors") or []),
            "censored": bool(source.get("censored") or claim.get("censored")),
        }
        result_rows.append(pointer_row)
        fidelity_rows.append(deepcopy(pointer_row))
        direct_calls = sorted(
            (row for row in calls if row.get("arm") == "direct_self_consistency"),
            key=lambda row: int(row.get("draw_index", 0) or 0),
        )
        direct_prediction, one_shot, tied = _direct_decision(direct_calls)
        direct_row = {
            "unit_id": unit_id,
            "base_group_id": hidden["base_group_id"],
            "condition": hidden["condition"],
            "arm": "direct_self_consistency",
            "seed": [row.get("seed") for row in direct_calls],
            "prediction": direct_prediction,
            "one_shot_decision": one_shot,
            "self_consistency_tie": tied,
            "expected_decision": expected,
            "representation_valid": len(direct_calls) == 2
            and all(row.get("parse_valid") is True for row in direct_calls),
            "source_fidelity": None,
            "claim_fidelity": None,
            "decision_correct": direct_prediction == expected,
            "unknown": direct_prediction == "unknown",
            "false_accept": _false_accept(expected, direct_prediction),
            "errors": [str(row["error"]) for row in direct_calls if row.get("error")],
            "censored": any(row.get("censored") is True for row in direct_calls),
        }
        result_rows.append(direct_row)
        fidelity_rows.append(deepcopy(direct_row))

    metrics: list[JsonDict] = []
    for arm in ARMS:
        arm_rows = [row for row in result_rows if row["arm"] == arm]
        metrics.append(
            {
                "arm": arm,
                "units": len(arm_rows),
                "representation_valid": sum(row["representation_valid"] for row in arm_rows),
                "source_fidelity": (
                    sum(row["source_fidelity"] is True for row in arm_rows)
                    if arm == "mention_pointer"
                    else None
                ),
                "claim_fidelity": (
                    sum(row["claim_fidelity"] is True for row in arm_rows)
                    if arm == "mention_pointer"
                    else None
                ),
                "answer_correct": sum(row["decision_correct"] for row in arm_rows),
                "unknown": sum(row["unknown"] for row in arm_rows),
                "false_accept": sum(row["false_accept"] for row in arm_rows),
                "accuracy": sum(row["decision_correct"] for row in arm_rows)
                / max(1, len(arm_rows)),
            }
        )

    by_group_condition = {
        (str(row["base_group_id"]), str(row["condition"])): row for row in authority_rows
    }
    result_by_unit_arm = {(str(row["unit_id"]), str(row["arm"])): row for row in result_rows}
    intervention_rows: list[JsonDict] = []
    for group_index in range(PLANNED_BASE_GROUPS):
        group_id = f"g-{group_index:02d}"
        baseline = by_group_condition[(group_id, "supported")]
        baseline_id = str(baseline["unit_id"])
        for condition, intervention in (
            ("consistent_entity_renaming", "consistent_entity_renaming"),
            ("relation_reversal", "relation_reversal"),
            ("insufficient_evidence", "missing_source_evidence"),
        ):
            target = by_group_condition[(group_id, condition)]
            target_id = str(target["unit_id"])
            intervention_rows.append(
                {
                    "base_group_id": group_id,
                    "intervention": intervention,
                    "baseline_unit_id": baseline_id,
                    "target_unit_id": target_id,
                    "expected_baseline": baseline["expected_decision"],
                    "expected_target": target["expected_decision"],
                    "arm_predictions": {
                        arm: {
                            "baseline": result_by_unit_arm[(baseline_id, arm)]["prediction"],
                            "target": result_by_unit_arm[(target_id, arm)]["prediction"],
                        }
                        for arm in ARMS
                    },
                    "additional_model_calls": 0,
                }
            )
        permutation = _source_shuffle_permutation([str(row["unit_id"]) for row in public_rows])
        source_id = permutation[baseline_id]
        source_public = public_by_id[source_id]
        source_calls = calls_by_unit.get(source_id, [])
        baseline_calls = calls_by_unit.get(baseline_id, [])
        source_capture = next(
            (
                row
                for row in source_calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "source"
            ),
            {},
        )
        claim_capture = next(
            (
                row
                for row in baseline_calls
                if row.get("arm") == "mention_pointer" and row.get("call_type") == "claim"
            ),
            {},
        )
        shuffled_public = {
            "source": source_public["source"],
            "claim": public_by_id[baseline_id]["claim"],
        }
        shuffled_prediction = _decision_from_pointer(
            shuffled_public, source_capture, claim_capture
        ).get("decision", "unknown")
        source_hidden = authority_by_id[source_id]
        expected_shuffled = mention_fixture.execute_pointer_pair(
            shuffled_public["source"],
            shuffled_public["claim"],
            source_hidden["gold_source_completion"],
            baseline["gold_claim_completion"],
        )["decision"]
        intervention_rows.append(
            {
                "base_group_id": group_id,
                "intervention": "source_shuffle",
                "baseline_unit_id": baseline_id,
                "source_unit_id": source_id,
                "expected_baseline": baseline["expected_decision"],
                "expected_target": expected_shuffled,
                "arm_predictions": {
                    "mention_pointer": {
                        "baseline": pointer_predictions[baseline_id],
                        "target": shuffled_prediction,
                    },
                    "direct_self_consistency": {"baseline": None, "target": None},
                },
                "same_captured_source_extraction": True,
                "additional_model_calls": 0,
            }
        )

    token_latency_rows = [
        {
            "call_order": row.get("call_order"),
            "call_id": row.get("call_id"),
            "unit_id": row.get("unit_id"),
            "arm": row.get("arm"),
            "call_type": row.get("call_type"),
            "draw_index": row.get("draw_index"),
            "prompt_tokens": row.get("prompt_tokens", 0),
            "completion_tokens": row.get("completion_tokens", 0),
            "latency_s": row.get("latency_s", 0.0),
            "error": row.get("error"),
            "censored": row.get("censored") is True,
        }
        for row in replay_rows
    ]
    metric_by_arm = {row["arm"]: row for row in metrics}
    source_value_score = int(
        metric_by_arm["mention_pointer"]["accuracy"]
        > metric_by_arm["direct_self_consistency"]["accuracy"]
    )
    return {
        "rows": result_rows,
        "source_fidelity_rows": fidelity_rows,
        "source_intervention_rows": intervention_rows,
        "representation_metric_rows": metrics,
        "token_latency_rows": token_latency_rows,
        "source_value_score": source_value_score,
        "source_value_observation": {
            "mention_pointer_accuracy": metric_by_arm["mention_pointer"]["accuracy"],
            "direct_self_consistency_accuracy": metric_by_arm["direct_self_consistency"][
                "accuracy"
            ],
            "accuracy_delta": metric_by_arm["mention_pointer"]["accuracy"]
            - metric_by_arm["direct_self_consistency"]["accuracy"],
            "direction_frozen_before_inference": "mention_pointer_strictly_greater",
        },
    }


def completeness_receipt(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    replay_errors: Sequence[str],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Measure capture accounting without consulting semantic correctness."""

    accounted = sum(
        row.get("request_response_joined") is True
        or row.get("explicit_censoring_authenticated") is True
        for row in replay_rows
    )
    authentic = accounted == PLANNED_CALLS
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    roster_ok = len(public_ids) == PLANNED_UNITS and public_ids == authority_ids
    complete = int(
        len(schedule) == PLANNED_CALLS
        and len(completion_rows) == PLANNED_CALLS
        and len(replay_rows) == PLANNED_CALLS
        and accounted == PLANNED_CALLS
        and not replay_errors
        and roster_ok
    )
    return {
        "planned_outcomes": PLANNED_CALLS,
        "outcomes_accounted": accounted,
        "authentic_or_explicit_censored_outcomes": accounted,
        "authentic_joins_complete": authentic,
        "public_private_roster_join": roster_ok,
        "replay_mismatch_count": len(replay_errors),
        "source_capture_complete_score": complete,
        "semantic_correctness_consulted": False,
    }


def _validations_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each focused validation once and require every command to pass."""

    names = [row.get("name") for row in receipts]
    return sorted(names) == sorted(REQUIRED_VALIDATION_NAMES) and all(
        row.get("passed") is True and row.get("exit_code") == 0 for row in receipts
    )


def base_artifact(run_date: str) -> JsonDict:
    """Create every required field before the first fallible precondition."""

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
            "planned_base_groups": PLANNED_BASE_GROUPS,
            "planned_units": PLANNED_UNITS,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": PLANNED_UNITS,
            "planned_calls": PLANNED_CALLS,
            "attempted_calls": 0,
            "completed_calls": 0,
            "censored_calls": PLANNED_CALLS,
            "stopping_rule": "attempt all 256 frozen calls once until the 2400 second deadline; no adaptive stop",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary([]),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_source_measurement_unfinished",
        "verdict_class": "partial",
        "validation_receipts": [],
        "source_capture_complete_score": 0,
        "source_value_score": 0,
        "raw_call_manifest": {},
        "source_fidelity_rows": [],
        "runner_receipt": {
            "model_count": 1,
            "replica_count": 0,
            "runner": "native_llama.cpp_server",
            "dual_gpu_runner_used": False,
        },
        "source_intervention_rows": [],
        "public_manifest_path": {},
        "private_authority_manifest_path": {},
        "representation_metric_rows": [],
        "token_latency_rows": [],
        "selection_receipt": {},
        "replay_rows": [],
        "replay_discrepancies": [],
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "feasibility_projection": {},
        "timestamps": {"started_at_utc": utc_now(), "completed_at_utc": None},
        "phase_spans": [],
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], *, duration_s: float
) -> JsonDict:
    """Finish an external pre-launch block without claiming partial task work."""

    artifact["status"] = "blocked"
    artifact["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    artifact["gate_check_summary"] = gate_summary(checks)
    artifact["MODEL_SPECS"] = deepcopy(MODEL_SPECS)
    artifact["model_invoked"] = False
    artifact["invocation_counts"] = deepcopy(ZERO_INVOCATION_COUNTS)
    artifact["inference_substrate"] = "blocked_before_qualifying_computation"
    artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["inference_mode"] = "not_invoked"
    artifact["source_capture_complete_score"] = 0
    artifact["source_value_score"] = 0
    artifact["verdict_class"] = "blocked"
    failure = artifact["gate_check_summary"].get("failed_check") or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_source_measurement_{failure}"
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
    reduced: Mapping[str, Any],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish complete measurement evidence while keeping value gates separate."""

    public_rows = list(artifact.get("_public_rows") or [])
    authority_rows = list(artifact.get("_authority_rows") or [])
    if not public_rows or not authority_rows:
        public, authority = build_source_fixture()
        public_rows = list(public["rows"])
        authority_rows = list(authority["rows"])
    receipt = completeness_receipt(
        schedule,
        completion_rows,
        replay_rows,
        replay_errors,
        public_rows,
        authority_rows,
    )
    artifact.pop("_public_rows", None)
    artifact.pop("_authority_rows", None)
    artifact["rows"] = deepcopy(list(reduced["rows"]))
    artifact["source_fidelity_rows"] = deepcopy(list(reduced["source_fidelity_rows"]))
    artifact["source_intervention_rows"] = deepcopy(list(reduced["source_intervention_rows"]))
    artifact["representation_metric_rows"] = deepcopy(list(reduced["representation_metric_rows"]))
    artifact["token_latency_rows"] = deepcopy(list(reduced["token_latency_rows"]))
    artifact["replay_rows"] = deepcopy(list(replay_rows))
    artifact["replay_discrepancies"] = list(replay_errors)
    artifact["source_capture_complete_score"] = receipt["source_capture_complete_score"]
    artifact["source_value_score"] = int(reduced["source_value_score"])
    artifact["source_value_observation"] = deepcopy(reduced["source_value_observation"])
    counts = dict(artifact.get("invocation_counts") or {})
    artifact["model_invoked"] = int(counts.get("generation_calls_attempted", 0) or 0) > 0
    if artifact["model_invoked"]:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_full_generation"
        artifact["inference_mode"] = "live_gpu"
    elif int(counts.get("model_loads_attempted", 0) or 0) > 0:
        artifact["inference_substrate"] = "live_llm_inference"
        artifact["inference_substrate_class"] = "model_load_only"
        artifact["inference_mode"] = "load_only"
    attempted_units = len(
        {str(row.get("unit_id")) for row in completion_rows if row.get("attempted") is True}
    )
    completed_units = len(
        {
            str(row.get("unit_id"))
            for row in replay_rows
            if row.get("request_response_joined") is True
        }
    )
    artifact["sample_size_budget"].update(
        {
            "attempted_units": attempted_units,
            "completed_units": completed_units,
            "censored_units": PLANNED_UNITS - completed_units,
            "attempted_calls": sum(row.get("attempted") is True for row in completion_rows),
            "completed_calls": sum(
                row.get("request_response_joined") is True for row in replay_rows
            ),
            "censored_calls": sum(row.get("censored") is True for row in completion_rows),
        }
    )
    selection = dict(artifact.get("selection_receipt") or {})
    authority_separation = bool(
        selection.get("authority_fields_in_schedule") == 0
        and not selection.get("private_terms_in_schedule")
        and not selection.get("exp7265_held_out_overlap")
    )
    equal_budget = bool(
        len(schedule) == PLANNED_CALLS
        and all(
            row.get("output_token_budget") == OUTPUT_TOKEN_BUDGET
            and row.get("request_timeout_s") == REQUEST_CAP_S
            for row in schedule
        )
    )
    observed = {
        "outcome_accounting": receipt["outcomes_accounted"],
        "authentic_joins": receipt["authentic_joins_complete"],
        "replay_discrepancies": len(replay_errors),
        "authority_separation": authority_separation,
        "equal_budget": equal_budget,
        "source_value": artifact["source_value_score"],
        "focused_validation": _validations_complete(artifact["validation_receipts"]),
    }
    expected = {
        "outcome_accounting": PLANNED_CALLS,
        "authentic_joins": True,
        "replay_discrepancies": 0,
        "authority_separation": True,
        "equal_budget": True,
        "source_value": 1,
        "focused_validation": True,
    }
    artifact["acceptance_gate_results"] = [
        {
            "criterion": name,
            "expected": expected[name],
            "observed": value,
            "passed": value == expected[name],
            "principle": GATE_PRINCIPLES[name],
        }
        for name, value in observed.items()
    ]
    artifact["status"] = "complete"
    if artifact["source_value_score"] == 1:
        artifact["honest_verdict"] = (
            "complete_circular_positive_mention_pointer_exceeds_equal_budget_direct"
        )
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["honest_verdict"] = (
            "complete_null_mention_pointer_does_not_exceed_equal_budget_direct"
        )
        artifact["verdict_class"] = "null"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, terminal semantics, denominators, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in value:
            return [f"missing_required_field:{field}"]
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema")
    if (value.get("experiment_id"), value.get("milestone"), value.get("run_date")) != (
        EXPERIMENT_ID,
        MILESTONE,
        RUN_DATE,
    ):
        errors.append("identity")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("MODEL_SPECS") != MODEL_SPECS or value.get("execution_venue") != "host":
        errors.append("execution_identity")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration < 0:
        errors.append("duration_s")
    status = value.get("status")
    if status == "blocked":
        if (
            value.get("verdict_class") != "blocked"
            or not str(value.get("honest_verdict", "")).startswith("blocked_")
            or value.get("invocation_counts") != ZERO_INVOCATION_COUNTS
            or value.get("model_invoked") is not False
            or not dict(value.get("gate_check_summary") or {}).get("failed_check")
            or value.get("source_capture_complete_score") != 0
        ):
            errors.append("blocked_terminal_state")
    elif status == "complete":
        budget = dict(value.get("sample_size_budget") or {})
        if (
            len(value.get("rows") or []) != PLANNED_RESULT_ROWS
            or len(value.get("source_fidelity_rows") or []) != PLANNED_RESULT_ROWS
            or len(value.get("source_intervention_rows") or []) != PLANNED_UNITS
            or len(value.get("replay_rows") or []) != PLANNED_CALLS
            or budget.get("planned_calls") != PLANNED_CALLS
            or budget.get("attempted_calls", 0) + budget.get("censored_calls", 0) != PLANNED_CALLS
            or value.get("source_capture_complete_score") != 1
        ):
            errors.append("denominators")
        if value.get("model_invoked") is not True or (
            value.get("inference_substrate"),
            value.get("inference_substrate_class"),
            value.get("inference_mode"),
        ) != ("live_llm_inference", "model_full_generation", "live_gpu"):
            errors.append("live_execution_contract")
        if not _validations_complete(value.get("validation_receipts") or []):
            errors.append("validation_receipts")
        if value.get("source_value_score") == 1:
            if value.get("verdict_class") != "circular_positive" or not str(
                value.get("honest_verdict", "")
            ).startswith("complete_circular_positive_"):
                errors.append("value_verdict")
        elif value.get("verdict_class") != "null" or not str(
            value.get("honest_verdict", "")
        ).startswith("complete_null_"):
            errors.append("null_verdict")
        gates = value.get("acceptance_gate_results")
        if not isinstance(gates, list) or len(gates) != len(GATE_PRINCIPLES):
            errors.append("acceptance_gate_results")
        if dict(value.get("raw_call_manifest") or {}).get("raw_call_count") != PLANNED_CALLS:
            errors.append("raw_call_manifest")
    else:
        errors.append("status")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    return errors


def _manifest_lists_experiment(value: Any, experiment_id: int) -> bool:
    """Find exact experiment identifiers without matching unrelated prose."""

    return semantic_replay._manifest_lists_experiment(value, experiment_id)


def feasibility_projection(canary: Mapping[str, Any]) -> JsonDict:
    """Project the fixed call budget from measured canary latency before launch."""

    latencies = [
        float(row.get("latency_s", 0.0) or 0.0)
        for row in canary.get("rows", [])
        if isinstance(row, Mapping)
    ]
    total = sum(latencies)
    mean = total / len(latencies) if latencies else 0.0
    projected = mean * PLANNED_CALLS
    return {
        "source": UPSTREAM_CANARY_PATH.as_posix(),
        "calibration_calls": len(latencies),
        "calibration_total_latency_s": total,
        "mean_call_latency_s": mean,
        "planned_calls": PLANNED_CALLS,
        "projected_generation_s": projected,
        "generation_cap_s": GENERATION_CAP_S,
        "projected_within_cap": bool(latencies and projected <= GENERATION_CAP_S),
        "projection_recorded_before_model_launch": True,
        "stopping_rule_changed_by_projection": False,
    }


def authenticate_inputs(
    root: Path,
    *,
    expected_hashes: Mapping[Path, str] | None = None,
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - filesystem boundary.
    """Authenticate both upstream gates, exact bytes, exclusion state, and outputs."""

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
    required = [
        root / UPSTREAM_REPLAY_PATH,
        root / UPSTREAM_CANARY_PATH,
        root / UPSTREAM_CONTRACT_PATH,
    ]
    if not all(path.is_file() for path in required):
        return checks, {}, {}
    try:
        replay_artifact = json.loads(required[0].read_text(encoding="utf-8"))
        canary_artifact = json.loads(required[1].read_text(encoding="utf-8"))
        contract = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(
            gate_row(
                "input_parse",
                "valid_json",
                f"{type(exc).__name__}:{exc}",
                False,
                upstream="exp7275_and_exp7277",
                field="json",
            )
        )
        return checks, {}, {}
    replay_errors = semantic_replay.validate_artifact(replay_artifact)
    canary_errors = comparator_canary.validate_artifact(canary_artifact)
    checks.extend(
        [
            gate_row(
                "semantic_replay_terminal",
                {"errors": [], "ready": 1},
                {
                    "errors": replay_errors,
                    "ready": replay_artifact.get("semantic_replay_ready_score"),
                },
                not replay_errors and replay_artifact.get("semantic_replay_ready_score") == 1,
                upstream="exp7275-semantic-replay",
                field="terminal_schema_and_semantic_replay_ready_score",
            ),
            gate_row(
                "comparator_canary_terminal",
                {"errors": [], "ready": 1},
                {
                    "errors": canary_errors,
                    "ready": canary_artifact.get("comparator_canary_ready_score"),
                },
                not canary_errors and canary_artifact.get("comparator_canary_ready_score") == 1,
                upstream="exp7277-comparator-canary",
                field="terminal_schema_and_comparator_canary_ready_score",
            ),
        ]
    )
    quarantined = any(
        bool(value.get("quarantined") or value.get("flagged_adversarial"))
        for value in (replay_artifact, canary_artifact)
    )
    checks.append(
        gate_row(
            "upstream_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="exp7275_and_exp7277",
            field="quarantined_or_flagged_adversarial",
        )
    )
    exclusion_path = root / EXCLUSION_PATH
    manifest = semantic_replay.prior.load_yaml(exclusion_path) if exclusion_path.is_file() else {}
    excluded = any(_manifest_lists_experiment(manifest, value) for value in (7275, 7277, 7278))
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
    output_state = {
        "result_absent": not (root / RESULT_PATH).exists(),
        "result_parent_writable": os.access(root / RESULT_PATH.parent, os.W_OK),
        "raw_parent_writable": os.access(root / RAW_DIR.parent, os.W_OK),
        "checkpoint_parent_writable": os.access(root / CHECKPOINT_PATH.parent, os.W_OK),
    }
    checks.append(
        gate_row(
            "authenticated_output_paths",
            {key: True for key in output_state},
            output_state,
            all(output_state.values()),
            upstream="host_filesystem",
            field="task_owned_output_paths",
        )
    )
    return checks, contract, feasibility_projection(canary_artifact)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live inventory.
    """Hash every source and external prerequisite used by this invocation."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        SPEC_PATH,
        UPSTREAM_REPLAY_PATH,
        UPSTREAM_CANARY_PATH,
        UPSTREAM_CONTRACT_PATH,
        mention_fixture.MODULE_PATH,
        capture.MODULE_PATH,
        semantic_replay.MODULE_PATH,
        comparator_canary.MODULE_PATH,
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


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    """Write new evidence once or require exact existing bytes for safe resume."""

    expected = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"existing raw evidence differs: {path}")
        return
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def seal_inputs(
    raw_dir: Path,
    public: Mapping[str, Any],
    authority: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Write public and evaluator-only manifests before any model work."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    public_path = raw_dir / "public_manifest.json"
    authority_path = raw_dir / "private_authority_manifest.json"
    schedule_path = raw_dir / "schedule.json"
    _write_or_match(public_path, public)
    _write_or_match(authority_path, authority)
    _write_or_match(schedule_path, {"schedule": list(schedule)})
    return {
        "public_manifest_path": {
            "path": str(public_path),
            "sha256": sha256_file(public_path),
            "private_fields_present": False,
        },
        "private_authority_manifest_path": {
            "path": str(authority_path),
            "sha256": sha256_file(authority_path),
            "model_worker_input": False,
        },
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
    }


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Retain one authentic or explicit censored record for every planned call."""

    calls: list[JsonDict] = []
    for index, (sealed, completion) in enumerate(zip(schedule, completion_rows, strict=True)):
        path = raw_dir / f"call_{index:02d}.json"
        _write_or_match(path, {"schedule": dict(sealed), "completion": dict(completion)})
        calls.append(
            {
                "call_order": index,
                "call_id": sealed["call_id"],
                "unit_id": sealed["unit_id"],
                "arm": sealed["arm"],
                "path": str(path),
                "sha256": sha256_file(path),
                "terminal_state": completion.get("terminal_state"),
                "attempted": completion.get("attempted") is True,
                "censored": completion.get("censored") is True,
                "request_bytes_sha256": completion.get("request_bytes_sha256"),
                "response_bytes_sha256": completion.get("response_bytes_sha256"),
            }
        )
    manifest = {
        "schema": "carnot.exp7278.raw_call_manifest.v1",
        "status": "complete" if len(calls) == PLANNED_CALLS else "partial",
        "raw_call_count": len(calls),
        "planned_call_count": PLANNED_CALLS,
        "schedule_sha256": sha256_json(list(schedule)),
        "model_identity": deepcopy(dict(model_identity)),
        "authority_path_opened_by_model_worker": False,
        "calls": calls,
    }
    path = raw_dir / "raw_call_manifest.json"
    _write_or_match(path, manifest)
    return {**manifest, "path": str(path), "manifest_sha256": sha256_file(path)}


def independent_replay_from_raw(
    raw_dir: Path,
) -> tuple[list[JsonDict], list[str]]:  # pragma: no cover - CLI replay boundary.
    """Replay task raw files and private grading without reading the producer result."""

    errors: list[str] = []
    try:
        public = json.loads((raw_dir / "public_manifest.json").read_text(encoding="utf-8"))
        authority = json.loads(
            (raw_dir / "private_authority_manifest.json").read_text(encoding="utf-8")
        )
        schedule = list(json.loads((raw_dir / "schedule.json").read_text())["schedule"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return [], [f"manifest:{type(exc).__name__}:{exc}"]
    completions: list[JsonDict] = []
    for index, sealed in enumerate(schedule):
        path = raw_dir / f"call_{index:02d}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("schedule") != sealed:
                errors.append(f"call_{index}:schedule")
            completions.append(dict(value["completion"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"call_{index}:{type(exc).__name__}:{exc}")
    replay_rows, replay_errors = independent_replay(schedule, completions)
    errors.extend(replay_errors)
    if not errors:
        reduced = reduce_measurement(
            public["rows"], authority["rows"], schedule, completions, replay_rows
        )
        candidate_path = raw_dir / "measured-terminal-candidate.json"
        if candidate_path.is_file():
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            for field in (
                "rows",
                "source_fidelity_rows",
                "source_intervention_rows",
                "representation_metric_rows",
                "token_latency_rows",
            ):
                if candidate.get(field) != reduced.get(field):
                    errors.append(f"candidate:{field}")
    return replay_rows, list(dict.fromkeys(errors))


def collect_live_preflight(
    root: Path,
    contract: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - GPU boundary.
    """Reuse the qualified cache, tokenizer, server, and idle-GPU checks."""

    checks, context = comparator_canary.collect_live_preflight(
        root,
        contract,
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
            "live_window_cap_s": GENERATION_CAP_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
            "completion_builder": build_completion_row,
        }
    )
    return checks, context


def capture_live(
    context: Mapping[str, Any],
    checkpoint_dir: Path,
    raw_dir: Path,
    spans: list[JsonDict],
) -> JsonDict:  # pragma: no cover - live native model boundary.
    """Run the fixed schedule once and convert deadline omissions to censoring."""

    original_fault_key = live_runtime._transport_fault_key
    live_runtime._transport_fault_key = lambda _value: None
    try:
        captured = comparator_canary.capture_live(context, checkpoint_dir, raw_dir, spans)
    finally:
        live_runtime._transport_fault_key = original_fault_key
    rows = list(captured.get("rows") or [])
    schedule = list(context["schedule"])
    for sealed in schedule[len(rows) :]:
        rows.append(censored_completion(sealed, "generation_deadline_or_runtime_end"))
    captured["rows"] = rows
    captured["scheduled_outcomes_accounted"] = len(rows)
    captured["actual_generation_rows"] = sum(row.get("attempted") is True for row in rows)
    return captured


def validation_commands(root: Path, raw_dir: Path) -> list[tuple[str, list[str]]]:
    """Return the fixed focused checks and never a repository-wide pytest run."""

    python = str(root / ".venv/bin/python")
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7236_v637_mention_fixture.py",
        "tests/python/test_experiment_7238_v637_mention_capture.py",
        "tests/python/test_experiment_7265_v639_mention_heldout.py",
        "tests/python/test_experiment_7275_v640_semantic_replay.py",
        "tests/python/test_experiment_7277_v640_comparator_canary.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
    coverage_file = "/tmp/.coverage-exp7278-v640"
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
                "--basetemp=/tmp/exp7278-focused",
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
                "--basetemp=/tmp/exp7278-affected",
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
                "--basetemp=/tmp/exp7278-coverage",
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
    """Flush every phase and long-operation boundary for external monitoring."""

    print(f"[exp7278] {canonical_json({'phase': phase, 'event': event, **details})}", flush=True)


def _run_validations(
    root: Path, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover - subprocess boundary.
    """Stream focused validation output and preserve exact failure receipts."""

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
                print(f"[exp7278:{name}] {line.rstrip()}", flush=True)
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


def _pending_receipts() -> list[JsonDict]:  # pragma: no cover - runtime candidate.
    """Make pending validation explicit instead of inventing successful checks."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7278/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _checkpoint(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover
    """Write unfinished current work only below the checkpoint directory."""

    value = deepcopy(dict(artifact))
    value.pop("_public_rows", None)
    value.pop("_authority_rows", None)
    if value.get("status") not in {"complete", "blocked"}:
        value["status"] = "partial"
        value["verdict_class"] = "partial"
        value["honest_verdict"] = "partial_source_measurement_unfinished"
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
) -> JsonDict:  # pragma: no cover - required live entrypoint.
    """Authenticate, seal, capture, replay, validate, and publish one result."""

    repo = root or find_repo_root(start=__file__)
    result_path = repo / RESULT_PATH
    raw_dir = repo / RAW_DIR
    checkpoint_path = repo / CHECKPOINT_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifact = base_artifact(run_date)
    spans: list[JsonDict] = []

    _progress(0, "phase_start", operation="startup_and_output_authentication")
    _checkpoint(checkpoint_path, artifact)
    _progress(0, "phase_end", checkpoint=str(checkpoint_path), result=str(result_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase = time.monotonic()
    _progress(1, "phase_start", operation="authenticate_exp7275_and_exp7277")
    checks, contract, projection = authenticate_inputs(repo)
    spans.append({"phase": "authenticate_upstreams", "duration_s": time.monotonic() - phase})
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    artifact["feasibility_projection"] = projection
    _progress(1, "phase_end", failed=sum(row.get("passed") is not True for row in checks))
    if any(row.get("passed") is not True for row in checks):
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        _progress(12, "write_start", path=str(result_path))
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(12, "write_end", path=str(result_path))
        return blocked

    phase = time.monotonic()
    _progress(2, "phase_start", operation="seal_fresh_public_and_private_rosters")
    public, authority = build_source_fixture()
    schedule = build_schedule(public["rows"])
    schedule_problems = schedule_errors(schedule, public["rows"])
    selection = selection_receipt(public, authority, schedule)
    seal = seal_inputs(raw_dir, public, authority, schedule)
    schedule_check = gate_row(
        "sealed_fresh_schedule",
        {
            "base_groups": PLANNED_BASE_GROUPS,
            "units": PLANNED_UNITS,
            "calls": PLANNED_CALLS,
            "errors": [],
            "authority_leaks": 0,
            "exp7265_overlap": [],
        },
        {
            "base_groups": public["fresh_source_groups"],
            "units": public["unit_count"],
            "calls": len(schedule),
            "errors": schedule_problems,
            "authority_leaks": selection["authority_fields_in_schedule"],
            "exp7265_overlap": selection["exp7265_held_out_overlap"],
        },
        not schedule_problems
        and selection["authority_fields_in_schedule"] == 0
        and not selection["exp7265_held_out_overlap"],
        upstream="exp7278-fresh-source-fixture",
        field="sealed_roster_schedule_and_authority_separation",
    )
    checks.append(schedule_check)
    artifact["preconditions_checked"] = checks
    artifact["selection_receipt"] = selection
    artifact["public_manifest_path"] = seal["public_manifest_path"]
    artifact["private_authority_manifest_path"] = seal["private_authority_manifest_path"]
    artifact["source_intervention_basis"] = {
        "arxiv_2607_00895": "Use localized source changes and exact evidence spans across structured public inputs.",
        "arxiv_2607_17047": "Separate solver hardness from surface sensitivity with proof-preserving entity renaming.",
        "papers_used_for_training_or_tuning": False,
    }
    spans.append({"phase": "seal_fresh_inputs", "duration_s": time.monotonic() - phase})
    _checkpoint(checkpoint_path, artifact)
    _progress(
        2,
        "phase_end",
        base_groups=PLANNED_BASE_GROUPS,
        units=PLANNED_UNITS,
        calls=len(schedule),
        roster_sha256=selection["public_roster_sha256"],
        errors=schedule_problems,
    )
    if not schedule_check["passed"]:
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        return blocked

    phase = time.monotonic()
    _progress(
        3,
        "phase_start",
        operation="live_resource_preflight",
        feasibility_projection_s=projection.get("projected_generation_s"),
    )
    resource_checks, context = collect_live_preflight(
        repo, contract, schedule, result_path, checkpoint_path, raw_dir
    )
    checks.extend(resource_checks)
    artifact["preconditions_checked"] = checks
    spans.append({"phase": "live_resource_preflight", "duration_s": time.monotonic() - phase})
    _progress(3, "phase_end", failed=sum(row.get("passed") is not True for row in checks))
    if any(row.get("passed") is not True for row in checks):
        artifact["phase_spans"] = spans
        blocked = finalize_blocked_artifact(artifact, checks, duration_s=time.monotonic() - started)
        _progress(12, "write_start", path=str(result_path))
        atomic_write_json(result_path, blocked, allow_override=False, sort_keys=True)
        _progress(12, "write_end", path=str(result_path))
        return blocked

    artifact["model_identity_receipt"] = deepcopy(dict(context["model_identity"]))
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    _progress(6, "benchmark_start", operation="native_load_and_fixed_256_calls")
    captured = capture_runner(context, checkpoint_path.parent, raw_dir, spans)
    completions = list(captured.get("rows") or [])
    counts = {
        "model_loads_attempted": 1,
        "model_loads_completed": int(captured.get("model_loaded") is True),
        "generation_calls_attempted": sum(row.get("attempted") is True for row in completions),
        "generation_calls_completed": sum(
            row.get("transport_complete") is True for row in completions
        ),
        "usable_answers": sum(row.get("usable") is True for row in completions),
    }
    artifact["invocation_counts"] = counts
    artifact["model_invoked"] = counts["generation_calls_attempted"] > 0
    artifact["gpu_receipts"] = deepcopy(dict(captured.get("gpu_receipts") or {}))
    artifact["runner_receipt"] = deepcopy(dict(captured.get("runner_receipt") or {}))
    artifact["runner_receipt"]["invocation_counts"] = deepcopy(counts)
    _progress(
        6,
        "benchmark_end",
        operation="native_load_and_fixed_256_calls",
        attempted_calls=counts["generation_calls_attempted"],
        accounted_outcomes=len(completions),
        total_units=PLANNED_CALLS,
        runtime_error=captured.get("runtime_error"),
    )
    if not artifact["model_invoked"]:
        artifact["phase_spans"] = spans
        _checkpoint(checkpoint_path, artifact)
        raise RuntimeError("model load or generation failed; partial evidence retained")

    phase = time.monotonic()
    _progress(8, "benchmark_start", operation="independent_raw_reduction")
    manifest = write_raw_manifest(
        raw_dir, schedule, completions, artifact["model_identity_receipt"]
    )
    replay_rows, replay_errors = independent_replay(schedule, completions)
    reduced = reduce_measurement(
        public["rows"], authority["rows"], schedule, completions, replay_rows
    )
    spans.append({"phase": "independent_raw_reduction", "duration_s": time.monotonic() - phase})
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"][(RAW_DIR / "raw_call_manifest.json").as_posix()] = {
        "sha256": manifest["manifest_sha256"],
        "retired": False,
        "quarantined": False,
    }
    artifact["source_artifact_hashes"][PUBLIC_MANIFEST_PATH.as_posix()] = {
        "sha256": seal["public_manifest_path"]["sha256"],
        "retired": False,
        "quarantined": False,
    }
    artifact["source_artifact_hashes"][PRIVATE_AUTHORITY_PATH.as_posix()] = {
        "sha256": seal["private_authority_manifest_path"]["sha256"],
        "retired": False,
        "quarantined": False,
    }
    for row in manifest["calls"]:
        artifact["source_artifact_hashes"][f"raw_call_{row['call_order']:03d}"] = {
            "sha256": row["sha256"],
            "retired": False,
            "quarantined": False,
        }
    artifact["phase_spans"] = spans
    artifact["validation_receipts"] = _pending_receipts()
    artifact["_public_rows"] = public["rows"]
    artifact["_authority_rows"] = authority["rows"]
    candidate = finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        reduced,
        duration_s=time.monotonic() - started,
    )
    _progress(
        8,
        "benchmark_end",
        operation="independent_raw_reduction",
        completed_units=len(replay_rows),
        discrepancies=len(replay_errors),
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
        passed=sum(row.get("passed") is True for row in receipts),
        total=len(receipts),
    )
    artifact["validation_receipts"] = receipts
    artifact["phase_spans"] = spans
    artifact["_public_rows"] = public["rows"]
    artifact["_authority_rows"] = authority["rows"]
    terminal = finalize_measured_artifact(
        artifact,
        schedule,
        completions,
        replay_rows,
        replay_errors,
        reduced,
        duration_s=time.monotonic() - started,
    )
    if not _validations_complete(receipts):
        terminal["status"] = "partial"
        terminal["verdict_class"] = "partial"
        terminal["honest_verdict"] = "partial_source_measurement_validation_failed"
        terminal["reproducibility_checksum"] = artifact_checksum(terminal)
        _checkpoint(checkpoint_path, terminal)
        raise RuntimeError("focused validation failed; terminal artifact not published")
    terminal_errors = validate_artifact(terminal)
    if terminal_errors:
        _checkpoint(checkpoint_path, terminal)
        raise ValueError(f"invalid Exp7278 artifact: {terminal_errors}")
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
    """Run live measurement or independently replay task-owned raw evidence."""

    print("[exp7278] startup", flush=True)
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
    artifact = run_experiment(run_date=args.date)
    print(
        f"[exp7278] terminal verdict={artifact['honest_verdict']} complete={artifact['source_capture_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
