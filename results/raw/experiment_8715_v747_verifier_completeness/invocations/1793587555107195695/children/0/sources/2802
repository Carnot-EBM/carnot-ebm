"""Repeat the fixed mention canary with corrected compute-floor evidence.

The module reuses the V637 public scheduler, parsers, and semantic scorer. It
adds the V639 upstream gate, current artifact schema, raw replay, and validation
receipts. The live path still owns one native llama.cpp server and GPU lease.

Spec refs: REQ-VERIFY-7264 and SCENARIO-VERIFY-7264-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7237_v637_mention_canary as prior
from carnot import experiment_7261_v639_compute_contract as compute_contract
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260913"
MILESTONE = "2026.09.639"
EXPERIMENT_ID = "exp7264-mention-canary"
TASK_ID = "experiment_7264_v639_mention_canary"
SCHEMA = "carnot.exp7264.v639_mention_canary.v1"
RANDOM_SEED = prior.RANDOM_SEED
MODEL_ID = prior.QWEN_MODEL_ID
QUANTIZATION = prior.QUANTIZATION
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

ARMS = prior.ARMS
TOKEN_BUDGETS = prior.TOKEN_BUDGETS
AUTHORITY_ONLY_FIELDS = prior.AUTHORITY_ONLY_FIELDS
MODEL_LOAD_CAP_S = prior.MODEL_LOAD_CAP_S
REQUEST_CAP_S = prior.REQUEST_CAP_S
INFERENCE_DEADLINE_S = prior.INFERENCE_DEADLINE_S

RESULT_PATH = Path("results/experiment_7264_v639_mention_canary.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7264")
RAW_DIR = Path("results/raw/experiment_7264")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
COMPUTE_PATH = Path("results/experiment_7261_v639_compute_contract.json")
FIXTURE_PATH = prior.UPSTREAM_PATH
PUBLIC_PATH = prior.PUBLIC_PATH
AUTHORITY_PATH = prior.AUTHORITY_PATH
EXCLUSION_PATH = prior.EXCLUSION_PATH
SPEC_PATH = prior.SPEC_PATH
MODULE_PATH = Path("python/carnot/experiment_7264_v639_mention_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7264_v639_mention_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7264_v639_mention_canary.py")

PINNED_COMPUTE_SHA256 = "sha256:c134e91939ff2207dcfe7ba168bc010c37e8a0a3e2c1344b0a83a02dec2b3c8d"

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
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted/completed loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute: full generation 60s, bounded generation 10s, load-only 2s; never pad time.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked_* name the upstream, exact field/check, observed value and expected value.",
    "verifier_is_oracle": "Expose shared evaluator/verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements, blocked_* for external absence, and explain the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed scientific gates forbid positive. Only incomplete own work is partial; unchanged external blocks are blocked.",
    "validation_receipts": "Record actual command, exit code and log hash; preserve failures and never suppress checks.",
    "mention_canary_ready_score": "One means original quality/provenance gates passed for scaling; no historical quarantine is erased.",
    "usable_unit_counts": "Successful transport and JSON parsing alone do not constitute faithful extraction.",
    "frozen_heldout_settings": "Bind per-arm budgets, seeds, templates and abstention rules before evaluation.",
    "raw_call_manifest": "Each request/reply and actual parameter set is independently replayable.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "authenticated_provenance": "Every model row must join authenticated code, model, process, GPU, and request evidence.",
    "fixed_schedule_complete": "The fixed denominator must retain every planned call.",
    "pointer_complete_parse_units_at_least_7": "At least seven units must return two complete syntactic pointer replies.",
    "pointer_usable_units_at_least_7": "At least seven units must resolve both replies into usable evidence.",
    "pointer_semantic_correct_units_at_least_6": "At least six units must preserve source, claim, direction, and decision meaning.",
    "negative_control_false_accepts_zero": "A reversed relation must never become accepted support.",
    "scoped_validation": "The measured candidate must pass every fixed focused validation command.",
}

canonical_json = prior.canonical_json
sha256_bytes = prior.sha256_bytes
sha256_file = prior.sha256_file
sha256_json = prior.sha256_json
load_yaml = prior.load_yaml
gate_row = prior.gate_row
gate_summary = prior.gate_summary
load_calibration_manifests = prior.load_calibration_manifests
unit_document = prior.unit_document
build_schedule = prior.build_schedule
schedule_errors = prior.schedule_errors
selection_receipt = prior.selection_receipt
render_gold_completion = prior.render_gold_completion
score_semantics = prior.score_semantics
measure_token_budgets = prior.measure_token_budgets
request_payload = prior._request_payload


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable inputs and raw evidence without process-local clocks."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key
        not in {
            "duration_s",
            "timestamps",
            "phase_spans",
            "reproducibility_checksum",
        }
    }
    return sha256_json(stable)


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Add one actual observation time to the shipped lossless row reducer."""

    row = prior.build_completion_row(sealed, response, resource)
    row["request_started_at_utc"] = response.get("started_at_utc")
    row["response_observed_at_utc"] = response.get("completed_at_utc") or utc_now()
    return row


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:
    """Rebuild rows from retained bytes and report any immutable-byte drift."""

    errors: list[str] = []
    try:
        replayed = prior.replay_completion_rows(schedule, retained_rows)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], [f"replay_error:{type(exc).__name__}:{exc}"]
    for index, (rebuilt, retained) in enumerate(zip(replayed, retained_rows, strict=False)):
        rebuilt["request_started_at_utc"] = retained.get("request_started_at_utc")
        rebuilt["response_observed_at_utc"] = retained.get("response_observed_at_utc")
        if rebuilt != retained:
            errors.append(f"call_{index}:replay_mismatch")
    if len(replayed) != len(retained_rows) or len(schedule) != len(retained_rows):
        errors.append("replay_denominator")
    return replayed, errors


def upstream_gate_rows(
    compute: Mapping[str, Any],
    compute_bytes: bytes,
    fixture: Mapping[str, Any],
    fixture_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate the repaired compute contract and exact fixture evidence."""

    compute_hash = sha256_bytes(compute_bytes)
    checks = [
        gate_row(
            "compute_contract_exact_bytes",
            PINNED_COMPUTE_SHA256,
            compute_hash,
            compute_hash == PINNED_COMPUTE_SHA256,
            upstream="exp7261-compute-contract",
            field="artifact_sha256",
        ),
        gate_row(
            "compute_contract_ready",
            1,
            compute.get("compute_contract_ready_score"),
            compute.get("compute_contract_ready_score") == 1,
            upstream="exp7261-compute-contract",
            field="compute_contract_ready_score",
        ),
        gate_row(
            "compute_contract_quarantine",
            False,
            live_runtime.is_quarantined(compute),
            not live_runtime.is_quarantined(compute),
            upstream="exp7261-compute-contract",
            field="structured_quarantine",
        ),
        gate_row(
            "compute_contract_checksum",
            True,
            compute.get("reproducibility_checksum")
            == compute_contract.reproducibility_checksum(compute),
            compute.get("reproducibility_checksum")
            == compute_contract.reproducibility_checksum(compute),
            upstream="exp7261-compute-contract",
            field="reproducibility_checksum",
        ),
        gate_row(
            "compute_contract_exclusion_manifest",
            False,
            live_runtime._manifest_hits(
                exclusion_manifest,
                {"exp7261-compute-contract", "experiment_7261_v639_compute_contract"},
            ),
            not live_runtime._manifest_hits(
                exclusion_manifest,
                {"exp7261-compute-contract", "experiment_7261_v639_compute_contract"},
            ),
            upstream="ops/exclusion_manifest.yaml",
            field="experiment_ids",
        ),
    ]
    checks.extend(
        prior.upstream_gate_rows(
            fixture,
            fixture_bytes,
            public_bytes,
            authority_bytes,
            public_bytes,
            exclusion_manifest,
        )
    )
    return checks


def readiness_receipt(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    *,
    provenance_errors: Sequence[str],
) -> JsonDict:
    """Apply the original gate and attach one principle to each criterion."""

    receipt = prior.readiness_receipt(
        schedule,
        completion_rows,
        semantic_rows,
        provenance_errors=provenance_errors,
    )
    receipt["criteria"] = [
        {
            "criterion": row["criterion"],
            "expected": True,
            "observed": row["actual_value"],
            "passed": row["passed"],
            "principle": GATE_PRINCIPLES[row["criterion"]],
        }
        for row in receipt["criteria"]
    ]
    return receipt


def usable_unit_counts(
    completion_rows: Sequence[Mapping[str, Any]], semantic_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Keep parse, usable, and faithful unit counts separate for every arm."""

    result: JsonDict = {}
    for arm in ARMS:
        calls = [row for row in completion_rows if row.get("arm") == arm]
        unit_ids = sorted({str(row.get("unit_id")) for row in calls})
        parse_complete = 0
        usable = 0
        for unit_id in unit_ids:
            pair = [row for row in calls if str(row.get("unit_id")) == unit_id]
            parse_complete += int(
                len(pair) == 2
                and {row.get("call_type") for row in pair} == {"source", "claim"}
                and all(
                    row.get("transport_complete") is True
                    and row.get("parse_valid") is True
                    and row.get("truncated") is False
                    for row in pair
                )
            )
            usable += int(len(pair) == 2 and all(row.get("usable") is True for row in pair))
        semantic = [row for row in semantic_rows if row.get("arm") == arm]
        result[arm] = {
            "unit_denominator": 8,
            "parse_complete": parse_complete,
            "usable": usable,
            "semantic_fidelity": sum(row.get("fully_correct") is True for row in semantic),
            "unknown_or_abstained": sum(row.get("abstention") is True for row in semantic),
        }
    return result


def frozen_heldout_settings() -> JsonDict:
    """Freeze Exp7265 budgets, seeds, templates, and abstention behavior now."""

    return {
        "target_experiment": "exp7265-mention-heldout",
        "model": deepcopy(MODEL_SPECS[0]),
        "arms": {
            "mention_pointer": {
                "call_pattern": [384, 128],
                "call_types": ["source", "claim"],
                "prompt_sha256": sha256_json(prior.POINTER_PROMPTS),
            },
            "explicit_schema_offset_control": {
                "call_pattern": [384, 128],
                "call_types": ["source", "claim"],
                "prompt_sha256": sha256_json(prior.EXPLICIT_PROMPTS),
            },
            "direct_judge": {
                "call_pattern": [512],
                "call_types": ["judge"],
                "prompt_contract": "one bounded source-and-claim decision call",
            },
        },
        "seed_base": 7_265_202_609_13,
        "seed_rule": "seed_base_plus_public_call_order",
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "retry_malformed": False,
        "abstention_rule": "unknown_or_unusable_is_not_faithful_evidence",
        "frozen_before_evaluation": True,
    }


def base_artifact(run_date: str) -> JsonDict:
    """Create a schema-complete checkpoint before any fallible prerequisite."""

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
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "not_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": {},
        "rows": [],
        "raw_rows": [],
        "schedule": [],
        "sample_size_budget": {
            "planned_independent_units": 8,
            "planned_arms": 3,
            "planned_calls": 48,
            "planned_semantic_rows": 24,
            "attempted_calls": 0,
            "completed_calls": 0,
            "censored_calls": 48,
            "completed_semantic_rows": 0,
            "stopping_rule": "run the fixed 48 calls once; do not tune or retry on outcomes",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7264_running_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "mention_canary_ready_score": 0,
        "usable_unit_counts": {},
        "frozen_heldout_settings": frozen_heldout_settings(),
        "raw_call_manifest": {},
        "selection_receipt": {},
        "readiness_receipt": {},
        "runner_receipt": {},
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "phase_spans": [],
        "token_budget_receipt": {},
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish one external block without inventing model work or a partial result."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["mention_canary_ready_score"] = 0
    artifact["gate_check_summary"] = gate_summary(failure)
    check = str(failure.get("check", "unknown")) if failure else "unknown"
    artifact["honest_verdict"] = f"blocked_exp7264_{check}"
    counts = artifact.get("invocation_counts") or ZERO_INVOCATION_COUNTS
    if counts.get("generation_calls_attempted", 0):
        artifact["inference_substrate"] = "live_llm_inference_local_gguf_sota"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
    elif counts.get("model_loads_completed", 0) and not counts.get("generation_calls_attempted", 0):
        artifact["inference_substrate"] = "model_load_no_generation"
        artifact["inference_substrate_class"] = "model_load_no_generation"
    elif not counts.get("model_loads_attempted", 0):
        artifact["inference_substrate"] = "blocked_no_run"
        artifact["inference_substrate_class"] = "blocked_no_run"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validation_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each named command exactly once and require every command to pass."""

    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == set(REQUIRED_VALIDATION_NAMES) and all(
        by_name[name].get("passed") is True and by_name[name].get("exit_code") == 0
        for name in REQUIRED_VALIDATION_NAMES
    )


def finalize_measured_artifact(
    artifact: JsonDict,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Finish one complete scale-gate measurement without claiming semantic value."""

    identity = artifact.get("model_identity_receipt")
    gpu = artifact.get("gpu_receipts")
    provenance_errors = prior._identity_errors(
        identity if isinstance(identity, Mapping) else {},
        gpu if isinstance(gpu, Mapping) else {},
        completion_rows,
    )
    if isinstance(identity, dict):
        identity["identity_errors"] = provenance_errors
    readiness = readiness_receipt(
        schedule,
        completion_rows,
        semantic_rows,
        provenance_errors=provenance_errors,
    )
    validation_ok = _validation_complete(validation_receipts)
    validation_gate = {
        "criterion": "scoped_validation",
        "expected": list(REQUIRED_VALIDATION_NAMES),
        "observed": [
            str(row.get("name"))
            for row in validation_receipts
            if row.get("passed") is True and row.get("exit_code") == 0
        ],
        "passed": validation_ok,
        "principle": GATE_PRINCIPLES["scoped_validation"],
    }
    transport = sum(row.get("transport_complete") is True for row in completion_rows)
    usable = sum(row.get("usable") is True for row in completion_rows)
    attempted_loads = max(
        1,
        int((artifact.get("invocation_counts") or {}).get("model_loads_attempted", 0)),
    )
    completed_loads = max(
        1,
        int((artifact.get("invocation_counts") or {}).get("model_loads_completed", 0)),
    )
    ready = readiness["mention_canary_ready_score"] == 1 and validation_ok
    artifact.update(
        {
            "status": "complete",
            "model_invoked": len(completion_rows) > 0,
            "invocation_counts": {
                "model_loads_attempted": attempted_loads,
                "model_loads_completed": completed_loads,
                "generation_calls_attempted": len(completion_rows),
                "generation_calls_completed": transport,
                "usable_answers": usable,
            },
            "inference_substrate": "live_llm_inference_local_gguf_sota",
            "inference_substrate_class": "model_bounded_generation",
            "inference_mode": "live_gpu",
            "duration_s": duration_s,
            "schedule": deepcopy(list(schedule)),
            "raw_rows": deepcopy(list(completion_rows)),
            "rows": deepcopy(list(semantic_rows)),
            "sample_size_budget": {
                "planned_independent_units": 8,
                "planned_arms": 3,
                "planned_calls": 48,
                "planned_semantic_rows": 24,
                "attempted_calls": len(completion_rows),
                "completed_calls": transport,
                "censored_calls": 48 - len(completion_rows),
                "completed_semantic_rows": len(semantic_rows),
                "stopping_rule": "run the fixed 48 calls once; do not tune or retry on outcomes",
            },
            "acceptance_gate_results": [*deepcopy(readiness["criteria"]), validation_gate],
            "gate_check_summary": gate_summary(None),
            "honest_verdict": (
                "complete_circular_positive_mention_canary_ready_scale_gate_only"
                if ready
                else "complete_null_mention_canary_not_ready_no_semantic_value_claim"
            ),
            "verdict_class": "circular_positive" if ready else "null",
            "validation_receipts": deepcopy(list(validation_receipts)),
            "mention_canary_ready_score": int(ready),
            "usable_unit_counts": usable_unit_counts(completion_rows, semantic_rows),
            "readiness_receipt": readiness,
            "raw_call_manifest": {
                "schedule_sha256": sha256_json(list(schedule)),
                "raw_row_count": len(completion_rows),
            },
        }
    )
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check required identity, denominators, gates, provenance, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        value.get("schema") != SCHEMA
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("milestone") != MILESTONE
    ):
        errors.append("identity")
    if value.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("MODEL_SPECS")
    if value.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed")
    if value.get("execution_venue") != "host" or not value.get("execution_host"):
        errors.append("execution_venue")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    if value.get("frozen_heldout_settings") != frozen_heldout_settings():
        errors.append("frozen_heldout_settings")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if value.get("verdict_class") != "blocked" or value.get("mention_canary_ready_score") != 0:
            errors.append("blocked_terminal_state")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        if value.get("inference_substrate") not in {
            "blocked_no_run",
            "model_load_no_generation",
            "live_llm_inference_local_gguf_sota",
        }:
            errors.append("blocked_substrate")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))

    schedule = value.get("schedule")
    raw_rows = value.get("raw_rows")
    rows = value.get("rows")
    if (
        not isinstance(schedule, list)
        or not isinstance(raw_rows, list)
        or not isinstance(rows, list)
    ):
        errors.append("rows")
        return list(dict.fromkeys(errors))
    _, replay_errors = independent_replay(schedule, raw_rows)
    if replay_errors:
        errors.append("raw_replay")
    if len(schedule) != 48 or len(raw_rows) != 48 or len(rows) != 24:
        errors.append("rows")
    identity = value.get("model_identity_receipt")
    gpu = value.get("gpu_receipts")
    provenance_errors = prior._identity_errors(
        identity if isinstance(identity, Mapping) else {},
        gpu if isinstance(gpu, Mapping) else {},
        raw_rows,
    )
    readiness = readiness_receipt(
        schedule,
        raw_rows,
        rows,
        provenance_errors=provenance_errors,
    )
    receipts = value.get("validation_receipts")
    if not isinstance(receipts, list) or {
        str(row.get("name")) for row in receipts if isinstance(row, Mapping)
    } != set(REQUIRED_VALIDATION_NAMES):
        errors.append("validation_receipts")
        validation_ok = False
    else:
        validation_ok = _validation_complete(receipts)
    expected_ready = int(readiness["mention_canary_ready_score"] == 1 and validation_ok)
    expected_class = "circular_positive" if expected_ready else "null"
    if value.get("mention_canary_ready_score") != expected_ready:
        errors.append("mention_canary_ready_score")
    if value.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    expected_criteria = [
        *readiness["criteria"],
        {
            "criterion": "scoped_validation",
            "expected": list(REQUIRED_VALIDATION_NAMES),
            "observed": [
                str(row.get("name"))
                for row in receipts or []
                if isinstance(row, Mapping)
                and row.get("passed") is True
                and row.get("exit_code") == 0
            ],
            "passed": validation_ok,
            "principle": GATE_PRINCIPLES["scoped_validation"],
        },
    ]
    if value.get("acceptance_gate_results") != expected_criteria:
        errors.append("acceptance_gate_results")
    counts = value.get("invocation_counts")
    transport = sum(row.get("transport_complete") is True for row in raw_rows)
    usable = sum(row.get("usable") is True for row in raw_rows)
    if not isinstance(counts, Mapping) or (
        counts.get("model_loads_attempted"),
        counts.get("model_loads_completed"),
        counts.get("generation_calls_attempted"),
        counts.get("generation_calls_completed"),
        counts.get("usable_answers"),
    ) != (1, 1, len(raw_rows), transport, usable):
        errors.append("invocation_counts")
    budget = value.get("sample_size_budget")
    if not isinstance(budget, Mapping) or (
        budget.get("planned_calls"),
        budget.get("attempted_calls"),
        budget.get("completed_calls"),
        budget.get("censored_calls"),
        budget.get("completed_semantic_rows"),
    ) != (48, len(raw_rows), transport, 48 - len(raw_rows), len(rows)):
        errors.append("sample_size_budget")
    if value.get("usable_unit_counts") != usable_unit_counts(raw_rows, rows):
        errors.append("usable_unit_counts")
    manifest = value.get("raw_call_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("schedule_sha256") != sha256_json(schedule)
        or manifest.get("raw_row_count") != len(raw_rows)
    ):
        errors.append("raw_call_manifest")
    if (
        value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference_local_gguf_sota"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
        or not isinstance(gpu, Mapping)
        or gpu.get("provenance_ok") is not True
        or provenance_errors
    ):
        errors.append("live_inference_provenance")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration < 10.0:
        errors.append("bounded_generation_duration_floor")
    return list(dict.fromkeys(errors))


def write_raw_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:
    """Seal real request, reply, parameter, token, time, and model evidence."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "carnot.exp7264.raw_calls.v1",
        "status": "complete" if len(completion_rows) == 48 else "partial",
        "created_at_utc": utc_now(),
        "model": deepcopy(dict(model_identity)),
        "raw_row_count": len(completion_rows),
        "schedule_sha256": sha256_json(list(schedule)),
        "authority_path_opened_by_model_worker": False,
        "retry_count": 0,
        "rows": [
            {
                "call_order": row.get("call_order"),
                "call_id": row.get("call_id"),
                "unit_id": row.get("unit_id"),
                "arm": row.get("arm"),
                "call_type": row.get("call_type"),
                "seed": row.get("seed"),
                "prompt": row.get("prompt"),
                "prompt_sha256": row.get("prompt_sha256"),
                "grammar_sha256": row.get("grammar_sha256"),
                "actual_parameters": deepcopy(row.get("actual_parameters")),
                "raw_request_bytes_b64": row.get("raw_request_bytes_b64"),
                "request_bytes_sha256": row.get("request_bytes_sha256"),
                "raw_response_bytes_b64": row.get("raw_response_bytes_b64"),
                "response_bytes_sha256": row.get("response_bytes_sha256"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "finish_reason": row.get("finish_reason"),
                "latency_s": row.get("latency_s"),
                "request_started_at_utc": row.get("request_started_at_utc"),
                "response_observed_at_utc": row.get("response_observed_at_utc"),
                "transport_complete": row.get("transport_complete"),
                "parse_valid": row.get("parse_valid"),
                "usable": row.get("usable"),
                "row_sha256": sha256_json(row),
            }
            for row in completion_rows
        ],
    }
    atomic_write_json(
        raw_dir / "raw_call_manifest.json",
        manifest,
        allow_override=False,
        sort_keys=True,
    )
    return manifest


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live inventory.
    """Hash every exact repository input used by this invocation."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "verification_spec": SPEC_PATH,
        "compute_contract": COMPUTE_PATH,
        "fixture_artifact": FIXTURE_PATH,
        "public_manifest": PUBLIC_PATH,
        "authority_manifest": AUTHORITY_PATH,
        "prior_canary_artifact": Path("results/experiment_7237_v637_mention_canary.json"),
        "prior_canary_module": Path("python/carnot/experiment_7237_v637_mention_canary.py"),
        "live_runtime": Path("python/carnot/experiment_7209_v635_span_canary.py"),
        "sota_models": Path("python/carnot/inference/sota_models.py"),
        "experiment_template": Path("scripts/experiment_template.py"),
        "module": MODULE_PATH,
        "entrypoint": WRAPPER_PATH,
        "focused_tests": TEST_PATH,
    }
    return {
        name: sha256_file(root / path) if (root / path).is_file() else "missing"
        for name, path in paths.items()
    }


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    """Flush each real phase and blocking-call observation."""

    print(
        canonical_json({"experiment": 7264, "phase": phase, "event": event, **fields}),
        flush=True,
    )


def _checkpoint(
    path: Path, artifact: Mapping[str, Any], started: float
) -> None:  # pragma: no cover
    """Write unfinished state only below the task checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _pending_validation_receipts() -> list[JsonDict]:  # pragma: no cover
    """Represent validation as pending so the measured candidate stays null."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7264/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _collect_preflight(  # pragma: no cover - live host boundary.
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Check Exp7261 first, then reuse the shipped native resource preflight."""

    compute_path = root / COMPUTE_PATH
    fixture_path = root / FIXTURE_PATH
    required = {
        "compute_contract": compute_path,
        "fixture_artifact": fixture_path,
        "public_manifest": root / PUBLIC_PATH,
        "authority_manifest": root / AUTHORITY_PATH,
        "exclusion_manifest": root / EXCLUSION_PATH,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        failed = gate_row(
            "required_upstream_paths",
            [],
            missing,
            False,
            upstream="repository",
            field="required_paths",
        )
        return [failed], [], [], {}

    compute_bytes = compute_path.read_bytes()
    fixture_bytes = fixture_path.read_bytes()
    public_bytes = (root / PUBLIC_PATH).read_bytes()
    authority_bytes = (root / AUTHORITY_PATH).read_bytes()
    try:
        compute = json.loads(compute_bytes)
        fixture = json.loads(fixture_bytes)
        exclusion = load_yaml(root / EXCLUSION_PATH)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        failed = gate_row(
            "upstream_parse",
            "valid_json_and_yaml",
            f"{type(exc).__name__}:{exc}",
            False,
            upstream="exp7261_and_exp7236",
            field="source_documents",
        )
        return [failed], [], [], {}
    initial = upstream_gate_rows(
        compute,
        compute_bytes,
        fixture,
        fixture_bytes,
        public_bytes,
        authority_bytes,
        exclusion,
    )
    failure = next((row for row in initial if row.get("passed") is not True), None)
    if failure:
        return initial, [], [], {}

    def fixture_gate(
        observed_fixture: Mapping[str, Any],
        observed_fixture_bytes: bytes,
        observed_public_bytes: bytes,
        observed_authority_bytes: bytes,
        _manifest_bytes: bytes,
        observed_exclusion: Any,
    ) -> list[JsonDict]:
        return upstream_gate_rows(
            compute,
            compute_bytes,
            observed_fixture,
            observed_fixture_bytes,
            observed_public_bytes,
            observed_authority_bytes,
            observed_exclusion,
        )

    checks, public_rows, authority_rows, context = live_runtime._collect_preflight(
        root,
        run_date,
        result_path,
        checkpoint_dir,
        raw_dir,
        contract={
            "run_date": RUN_DATE,
            "upstream_path": FIXTURE_PATH,
            "public_path": PUBLIC_PATH,
            "authority_path": AUTHORITY_PATH,
            "manifest_path": PUBLIC_PATH,
            "spec_path": SPEC_PATH,
            "module_path": MODULE_PATH,
            "wrapper_path": WRAPPER_PATH,
            "test_path": TEST_PATH,
            "spec_req": "REQ-VERIFY-7264",
            "expected_calls": 48,
            "expected_units": 8,
            "task_id": TASK_ID,
            "upstream_id": "experiment_7236",
            "split_id": "experiment_7236_calibration",
            "upstream_gate_rows": fixture_gate,
            "load_split": load_calibration_manifests,
            "build_schedule": build_schedule,
            "schedule_errors": schedule_errors,
            "request_cap_s": REQUEST_CAP_S,
            "live_window_cap_s": INFERENCE_DEADLINE_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
        },
    )
    if context.get("schedule"):
        context["selection_receipt"] = selection_receipt(
            public_rows, authority_rows, context["schedule"]
        )
        context["completion_builder"] = build_completion_row
    return checks, public_rows, authority_rows, context


def independent_replay_from_raw(root: Path, raw_dir: Path) -> list[str]:  # pragma: no cover
    """Replay each task-owned call file without issuing a model request."""

    try:
        schedule_value = json.loads((raw_dir / "schedule.json").read_text(encoding="utf-8"))
        schedule = list(schedule_value["schedule"])
        retained = []
        for index in range(48):
            value = json.loads((raw_dir / f"call_{index:02d}.json").read_text(encoding="utf-8"))
            if value.get("schedule") != schedule[index]:
                return [f"call_{index}:schedule"]
            retained.append(value["completion"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"raw_read:{type(exc).__name__}:{exc}"]
    public_rows, authority_rows = load_calibration_manifests(
        root / PUBLIC_PATH, root / AUTHORITY_PATH
    )
    errors = schedule_errors(schedule, public_rows, authority_rows)
    replayed, replay_errors = independent_replay(schedule, retained)
    errors.extend(replay_errors)
    semantics = score_semantics(schedule, replayed, public_rows, authority_rows)
    if len(semantics) != 24:
        errors.append("semantic_denominator")
    return errors


def _validation_commands(
    root: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return the fixed focused commands that can promote the null candidate."""

    python = str(root / ".venv/bin/python")
    coverage_file = "/tmp/.coverage-exp7264-v639"
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7237_v637_mention_canary.py",
        "tests/python/test_adversarial_verify_substrate_class_20260905.py",
        "tests/python/test_substrate_class_cutover_20260907.py",
    ]
    changed = [MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), test]
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
                "--basetemp=/tmp/exp7264-v639-focused",
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
                "--basetemp=/tmp/exp7264-v639-affected",
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
                "--basetemp=/tmp/exp7264-v639-coverage",
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
            [
                python,
                "-u",
                "scripts/check_spec_coverage.py",
                test,
                *affected,
            ],
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


def _run_validations(root: Path, raw_dir: Path) -> list[JsonDict]:  # pragma: no cover
    """Stream each fixed subprocess and retain its exact output hash and exit code."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[JsonDict] = []
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    for index, (name, command) in enumerate(_validation_commands(root, raw_dir), start=1):
        _progress(
            9,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(REQUIRED_VALIDATION_NAMES),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - command is a fixed local argv list.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        with live_runtime._heartbeat(9, name, lambda: index - 1, len(REQUIRED_VALIDATION_NAMES)):
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                print(f"exp7264 {name} {line.rstrip()}", flush=True)
            returncode = process.wait()
        output = "".join(lines)
        log_path = validation_dir / f"{name}.log"
        log_path.write_text(output, encoding="utf-8")
        receipt = {
            "name": name,
            "command": shlex.join(command),
            "exit_code": returncode,
            "passed": returncode == 0,
            "timed_out": False,
            "duration_s": time.monotonic() - started,
            "log_path": log_path.relative_to(root).as_posix(),
            "log_sha256": sha256_file(log_path),
        }
        receipts.append(receipt)
        _progress(
            9,
            "subprocess_end",
            operation=name,
            exit_code=returncode,
            completed_units=index,
            total_units=len(REQUIRED_VALIDATION_NAMES),
        )
    return receipts


def _publish_terminal(  # pragma: no cover - live atomic publication.
    artifact: JsonDict, result_path: Path, checkpoint_path: Path, started: float
) -> JsonDict:
    """Cold-check the final value before one atomic terminal write."""

    _progress(10, "validation_start", path=str(result_path))
    artifact["duration_s"] = time.monotonic() - started
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact)
    _progress(10, "validation_end", errors=errors)
    if errors:
        raise ValueError(f"invalid Exp7264 artifact: {errors}")
    _checkpoint(checkpoint_path, artifact, started)
    _progress(11, "write_start", path=str(result_path))
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(11, "write_end", path=str(result_path))
    return artifact


def run_experiment(  # pragma: no cover - real native GPU E2E.
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
) -> JsonDict:
    """Run one fixed live capture or publish one exact external block."""

    _progress(0, "start", detail="checkpoint before every prerequisite")
    started = time.monotonic()
    repo = root or find_repo_root(start=__file__)
    destination = output_root or repo
    result_path = destination / RESULT_PATH
    checkpoint_dir = destination / CHECKPOINT_DIR
    raw_dir = destination / RAW_DIR
    checkpoint_path = checkpoint_dir / "terminal_candidate.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = base_artifact(run_date)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", checkpoint=str(checkpoint_path))

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    phase_started = time.monotonic()
    _progress(1, "phase_start", operation="authenticated_preflight")
    checks, public_rows, authority_rows, context = _collect_preflight(
        repo, run_date, result_path, checkpoint_dir, raw_dir
    )
    artifact["phase_spans"].append(
        {
            "phase": 1,
            "name": "authenticated_preflight",
            "duration_s": time.monotonic() - phase_started,
        }
    )
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = _source_hashes(repo)
    failure = next((row for row in checks if row.get("passed") is not True), None)
    _progress(1, "phase_end", passed=failure is None)
    if failure is not None:
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _publish_terminal(artifact, result_path, checkpoint_path, started)

    schedule = list(context["schedule"])
    artifact["schedule"] = deepcopy(schedule)
    artifact["selection_receipt"] = deepcopy(context["selection_receipt"])
    _progress(2, "phase_start", operation="freeze_48_public_requests")
    atomic_write_json(
        raw_dir / "schedule.json",
        {"schedule": schedule},
        allow_override=False,
        sort_keys=True,
    )
    _checkpoint(checkpoint_path, artifact, started)
    _progress(2, "phase_end", calls=len(schedule), units=8, arms=3)

    _progress(3, "model_load_start", operation="embedded_tokenizer_vocab_only")
    tokenizer_started = time.monotonic()
    owner, tokenizer, tokenizer_load = live_runtime._load_embedded_tokenizer(
        Path(context["model_path"]), context["tokenizer_loader"]
    )
    artifact["phase_spans"].append(
        {
            "phase": 3,
            "name": "embedded_tokenizer_load",
            "duration_s": time.monotonic() - tokenizer_started,
        }
    )
    _progress(
        3,
        "model_load_end",
        operation="embedded_tokenizer_vocab_only",
        available=tokenizer_load["embedded_tokenizer_available"],
    )
    if tokenizer is None:
        check = gate_row(
            "embedded_tokenizer_load",
            True,
            False,
            False,
            upstream="cached_qwen_gguf",
            field="embedded_tokenizer",
        )
        checks.append(check)
        artifact["preconditions_checked"] = checks
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _publish_terminal(artifact, result_path, checkpoint_path, started)
    artifact["token_budget_receipt"] = measure_token_budgets(schedule, tokenizer)
    owner.close()
    artifact["model_identity_receipt"] = deepcopy(context["model_identity"])
    artifact["model_identity_receipt"]["tokenizer_load_receipt"] = tokenizer_load

    _progress(4, "benchmark_start", operation="native_load_and_fixed_48_calls")
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    capture = live_runtime._live_capture(context, checkpoint_dir, raw_dir, artifact["phase_spans"])
    completion_rows = list(capture["rows"])
    artifact["invocation_counts"] = {
        "model_loads_attempted": 1,
        "model_loads_completed": int(capture["model_loaded"] is True),
        "generation_calls_attempted": len(completion_rows),
        "generation_calls_completed": sum(
            row.get("transport_complete") is True for row in completion_rows
        ),
        "usable_answers": sum(row.get("usable") is True for row in completion_rows),
    }
    artifact["model_invoked"] = capture["model_invoked"] is True
    _progress(
        4,
        "benchmark_end",
        completed_units=len(completion_rows),
        total_units=48,
        runtime_error=capture["runtime_error"],
    )

    _progress(5, "benchmark_start", operation="independent_semantic_reduction")
    replayed, replay_errors = independent_replay(schedule, completion_rows)
    semantic_rows = score_semantics(schedule, replayed, public_rows, authority_rows)
    _progress(
        5,
        "benchmark_end",
        semantic_rows=len(semantic_rows),
        replay_errors=replay_errors,
    )
    artifact["gpu_receipts"] = deepcopy(capture["gpu_receipts"])
    artifact["runner_receipt"] = deepcopy(capture["runner_receipt"])
    artifact["runner_receipt"].update(
        {
            "model_loaded": capture["model_loaded"] is True,
            "cleanup_ok": artifact["gpu_receipts"].get("cleanup", {}).get("leak_free") is True,
            "reducer_errors": replay_errors,
        }
    )
    manifest = write_raw_manifest(raw_dir, schedule, replayed, artifact["model_identity_receipt"])
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"]["raw_call_manifest"] = sha256_file(
        raw_dir / "raw_call_manifest.json"
    )
    for path in sorted(raw_dir.glob("call_*.json")):
        artifact["source_artifact_hashes"][path.stem] = sha256_file(path)

    transport = sum(row.get("transport_complete") is True for row in replayed)
    runtime_failed = bool(
        capture["runtime_error"]
        or replay_errors
        or len(replayed) != 48
        or transport != 48
        or capture["gpu_receipts"].get("provenance_ok") is not True
    )
    if runtime_failed:
        check = gate_row(
            "live_runtime_completion_and_cuda_provenance",
            {"rows": 48, "transport": 48, "runtime_error": None, "provenance_ok": True},
            {
                "rows": len(replayed),
                "transport": transport,
                "runtime_error": capture["runtime_error"],
                "provenance_ok": capture["gpu_receipts"].get("provenance_ok"),
                "replay_errors": replay_errors,
            },
            False,
            upstream="owned_native_llama_server",
            field="live_capture",
        )
        checks.append(check)
        artifact["preconditions_checked"] = checks
        artifact["schedule"] = deepcopy(schedule)
        artifact["raw_rows"] = deepcopy(replayed)
        artifact["rows"] = deepcopy(semantic_rows)
        finalize_blocked_artifact(artifact, checks, time.monotonic() - started)
        return _publish_terminal(artifact, result_path, checkpoint_path, started)

    finalize_measured_artifact(
        artifact,
        schedule,
        replayed,
        semantic_rows,
        duration_s=time.monotonic() - started,
        validation_receipts=_pending_validation_receipts(),
    )
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"] = _source_hashes(repo) | {
        "raw_call_manifest": sha256_file(raw_dir / "raw_call_manifest.json"),
        **{path.stem: sha256_file(path) for path in sorted(raw_dir.glob("call_*.json"))},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(8, "write_start", path=str(destination / RAW_CANDIDATE_PATH))
    atomic_write_json(
        destination / RAW_CANDIDATE_PATH,
        artifact,
        allow_override=False,
        sort_keys=True,
    )
    _progress(8, "write_end", path=str(destination / RAW_CANDIDATE_PATH))

    receipts = _run_validations(repo, raw_dir)
    finalize_measured_artifact(
        artifact,
        schedule,
        replayed,
        semantic_rows,
        duration_s=time.monotonic() - started,
        validation_receipts=receipts,
    )
    artifact["raw_call_manifest"] = manifest
    artifact["source_artifact_hashes"] = _source_hashes(repo) | {
        "raw_call_manifest": sha256_file(raw_dir / "raw_call_manifest.json"),
        **{path.stem: sha256_file(path) for path in sorted(raw_dir.glob("call_*.json"))},
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return _publish_terminal(artifact, result_path, checkpoint_path, started)


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V639 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canary or perform its independent no-generation raw replay."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    if args.replay_raw is not None:
        root = find_repo_root(start=__file__)
        errors = independent_replay_from_raw(root, args.replay_raw)
        print(canonical_json({"replay_errors": errors}), flush=True)
        return int(bool(errors))
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7264] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7264] terminal verdict={artifact['honest_verdict']} "
        f"ready={artifact['mention_canary_ready_score']}",
        flush=True,
    )
    return 0
