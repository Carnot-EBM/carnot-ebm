"""Measure the unchanged held-out mention question with the clean V639 canary.

The module keeps model-visible work separate from private grading. It reuses
the shipped extraction compiler, exact executor, and native llama.cpp runtime.

Spec refs: REQ-VERIFY-7265 and SCENARIO-VERIFY-7265-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys
import time
from typing import Any, Iterator

from carnot import experiment_7209_v635_span_canary as live_runtime
from carnot import experiment_7236_v637_mention_fixture as mention_fixture
from carnot import experiment_7237_v637_mention_canary as extraction
from carnot import experiment_7238_v637_mention_capture as capture
from carnot import experiment_7264_v639_mention_canary as clean_canary
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import utc_now
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]

RUN_DATE = "20260913"
MILESTONE = "2026.09.639"
EXPERIMENT_ID = "exp7265-mention-heldout"
TASK_ID = "experiment_7265_v639_mention_heldout"
SCHEMA = "carnot.exp7265.v639_mention_heldout.v1"
RANDOM_SEED = 726_520_260_913
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

ARMS = ("mention_pointer", "explicit_schema_offset_control", "direct_judge")
REPRESENTATION_ARMS = ARMS[:2]
TOKEN_BUDGETS = {"source": 384, "claim": 128, "direct": 512}
CONTEXT_TOKEN_BUDGET = 8192
MODEL_LOAD_CAP_S = 240.0
REQUEST_CAP_S = 90.0
INFERENCE_DEADLINE_S = 2400.0
PLANNED_UNITS = 64
PLANNED_CALLS = 320
PLANNED_ROWS = 192

RESULT_PATH = Path("results/experiment_7265_v639_mention_heldout.json")
CHECKPOINT_DIR = Path("results/checkpoints/experiment_7265")
RAW_DIR = Path("results/raw/experiment_7265")
CAPTURE_MANIFEST_PATH = RAW_DIR / "raw_call_manifest.json"
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
UPSTREAM_PATH = Path("results/experiment_7264_v639_mention_canary.json")
FIXTURE_PATH = Path("results/experiment_7236_v637_mention_fixture.json")
PUBLIC_PATH = Path("results/raw/experiment_7236/public_manifest.json")
AUTHORITY_PATH = Path("results/raw/experiment_7236/authority_manifest.json")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
MODULE_PATH = Path("python/carnot/experiment_7265_v639_mention_heldout.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7265_v639_mention_heldout.py")
TEST_PATH = Path("tests/python/test_experiment_7265_v639_mention_heldout.py")

PINNED_CANARY_SHA256 = "sha256:ff251124a7709c97bc301e10211d70721cce722f1cb4bec6a20f5f4955f6bef8"
PINNED_FIXTURE_SHA256 = capture.PINNED_FIXTURE_SHA256
PINNED_PUBLIC_SHA256 = capture.PINNED_PUBLIC_SHA256
PINNED_AUTHORITY_SHA256 = capture.PINNED_AUTHORITY_SHA256

AUTHORITY_ONLY_FIELDS = extraction.AUTHORITY_ONLY_FIELDS
DIRECT_PROMPT = (
    "Judge the claim from the source. Return decision a when the claim is supported, "
    "b when it is contradicted, and c when the source is insufficient. Return only "
    "one JSON object with the decision key.\nSOURCE:\n{source}\nCLAIM:\n{claim}"
)
canonical_json = mention_fixture.canonical_json
sha256_bytes = mention_fixture.sha256_bytes
sha256_file = mention_fixture.sha256_file
sha256_json = capture.sha256_json
load_yaml = live_runtime.load_yaml
gate_row = live_runtime.gate_row
gate_summary = live_runtime.gate_summary
request_payload = live_runtime._request_payload
load_held_out_manifests = capture.load_held_out_manifests
score_semantics = capture.score_semantics
_shipped_identity_errors = capture._identity_errors

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
    "mention_capture_complete_score": "One means the full frozen denominator is accounted, regardless of accuracy.",
    "raw_call_manifest": "Tie all 320 planned calls to raw bytes, errors or explicit censoring.",
    "source_fidelity_rows": "Separate source relation fidelity from claim fidelity and constraint execution.",
    "token_cost_rows": "Matched access and budgets expose extra compute in each arm.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

GATE_PRINCIPLES = {
    "authenticated_provenance": "Every call joins its request bytes, fixed seed, model, server, GPU, and lease.",
    "fixed_schedule": "All 320 public calls stay in the frozen denominator.",
    "terminal_accounting": "Every planned call has one authentic terminal outcome.",
    "independent_rows": "Private grading retains all 192 arm-unit rows.",
    "semantic_exactness": "This strict diagnostic reports quality but does not define measurement completion.",
    "scoped_validation": "Every required focused checker must pass before terminal publication.",
}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash durable evidence while excluding process-local clock observations."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "timestamps", "phase_spans", "reproducibility_checksum"}
    }
    return sha256_json(stable)


def _paired_orders() -> list[tuple[str, str]]:
    """Freeze paired extraction-arm order without using private outcomes."""

    generator = random.Random(RANDOM_SEED)
    return [
        REPRESENTATION_ARMS
        if generator.getrandbits(1) == 0
        else tuple(reversed(REPRESENTATION_ARMS))
        for _ in range(PLANNED_UNITS)
    ]


def _direct_grammar() -> JsonDict:
    """Use neutral output symbols so public grammar checks cannot see label words."""

    grammar = (
        'root ::= known\nknown ::= "{\\"decision\\":" decision "}"\n'
        'decision ::= "\\"a\\"" | "\\"b\\"" | "\\"c\\""\n'
    )
    digest = sha256_bytes(grammar.encode())
    return {
        "grammar": grammar,
        "grammar_sha256": digest,
        "reference_grammar_sha256": digest,
    }


def _append_schedule_row(
    schedule: list[JsonDict],
    *,
    unit_id: str,
    arm: str,
    call_type: str,
    document: Mapping[str, Any],
    model_input: Mapping[str, Any],
    prompt: str,
    grammar: Mapping[str, Any],
) -> None:
    """Append one public call and bind its identity to the frozen seed."""

    order = len(schedule)
    seed = RANDOM_SEED + order
    schedule.append(
        {
            "call_order": order,
            "call_id": sha256_json(
                {
                    "task": TASK_ID,
                    "unit_id": unit_id,
                    "arm": arm,
                    "call_type": call_type,
                    "seed": seed,
                }
            ),
            "unit_id": unit_id,
            "arm": arm,
            "call_type": call_type,
            "seed": seed,
            "document": deepcopy(dict(document)),
            "model_input": deepcopy(dict(model_input)),
            "input_sha256": sha256_json(model_input),
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode()),
            **deepcopy(dict(grammar)),
            "output_token_budget": TOKEN_BUDGETS[call_type],
            "context_token_budget": CONTEXT_TOKEN_BUDGET,
            "request_timeout_s": REQUEST_CAP_S,
            "decoding_parameters": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": seed,
                "cache_prompt": False,
            },
            "cold_request": True,
        }
    )


def build_schedule(
    public_rows: Sequence[Mapping[str, Any]], authority_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Freeze 320 model-visible calls without using private labels for selection."""

    if len(public_rows) != PLANNED_UNITS or len(authority_rows) != PLANNED_UNITS:
        raise ValueError("held_out_denominator")
    public_ids = [str(row.get("unit_id")) for row in public_rows]
    authority_ids = [str(row.get("unit_id")) for row in authority_rows]
    if (
        any(row.get("split") != "held_out" for row in [*public_rows, *authority_rows])
        or len(set(public_ids)) != PLANNED_UNITS
        or public_ids != authority_ids
    ):
        raise ValueError("held_out_identity")
    schedule: list[JsonDict] = []
    for public, arm_order in zip(public_rows, _paired_orders(), strict=True):
        unit_id = str(public["unit_id"])
        for arm in arm_order:
            for call_type in ("source", "claim"):
                document = extraction.unit_document(public, call_type)
                _append_schedule_row(
                    schedule,
                    unit_id=unit_id,
                    arm=arm,
                    call_type=call_type,
                    document=document,
                    model_input={
                        "text": document["text"],
                        "mentions": document["mentions"] if arm == "mention_pointer" else None,
                    },
                    prompt=extraction._prompt(arm, document, call_type),
                    grammar=extraction.compile_grammar(arm, document, call_type),
                )
        source = extraction.unit_document(public, "source")
        claim = extraction.unit_document(public, "claim")
        _append_schedule_row(
            schedule,
            unit_id=unit_id,
            arm="direct_judge",
            call_type="direct",
            document={"source": source, "claim": claim},
            model_input={"source": source["text"], "claim": claim["text"]},
            prompt=DIRECT_PROMPT.format(source=source["text"], claim=claim["text"]),
            grammar=_direct_grammar(),
        )
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]],
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Name any change from the public schedule rebuilt from sealed inputs."""

    try:
        expected = build_schedule(public_rows, authority_rows)
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive input.
        return [f"schedule_rebuild:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    if len(schedule) != len(expected):
        errors.append("schedule_count")
    for index, (observed, wanted) in enumerate(zip(schedule, expected, strict=False)):
        for field, value in wanted.items():
            if observed.get(field) != value:
                errors.append(f"call_{index}:{field}")
        if set(observed) - set(wanted):  # pragma: no cover - defensive input.
            errors.append(f"call_{index}:extra_fields")
    return errors


def selection_receipt(
    public_rows: Sequence[Mapping[str, Any]],
    authority_rows: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Seal roster, phrase split, vocabulary split, and private hashes before calls."""

    return {
        "selection_frozen_before_inference": True,
        "selected_unit_ids": [str(row["unit_id"]) for row in public_rows],
        "selected_unit_count": len(public_rows),
        "condition_counts": dict(
            sorted(Counter(str(row["condition_key"]) for row in authority_rows).items())
        ),
        "phrase_split_sha256": sha256_json([row.get("relation_phrase") for row in authority_rows]),
        "vocabulary_split_sha256": sha256_json(
            [row.get("entity_vocabulary") for row in authority_rows]
        ),
        "public_rows_sha256": sha256_json(list(public_rows)),
        "independent_authority_sha256": sha256_json(list(authority_rows)),
        "schedule_sha256": sha256_json(list(schedule)),
        "authority_fields_in_model_schedule": sum(
            bool(set(row) & AUTHORITY_ONLY_FIELDS) for row in schedule
        ),
        "paired_arm_orders": [list(order) for order in _paired_orders()],
        "retry_allowed": False,
        "held_out_bytes_used_for_tuning": False,
    }


def upstream_gate_rows(
    canary: Mapping[str, Any],
    canary_bytes: bytes,
    public_bytes: bytes,
    authority_bytes: bytes,
    fixture_bytes: bytes,
    exclusion_manifest: Any,
) -> list[JsonDict]:
    """Authenticate the clean canary and unchanged held-out source bytes."""

    observed_hashes = {
        "canary": sha256_bytes(canary_bytes),
        "fixture": sha256_bytes(fixture_bytes),
        "public": sha256_bytes(public_bytes),
        "authority": sha256_bytes(authority_bytes),
    }
    expected_hashes = {
        "canary": PINNED_CANARY_SHA256,
        "fixture": PINNED_FIXTURE_SHA256,
        "public": PINNED_PUBLIC_SHA256,
        "authority": PINNED_AUTHORITY_SHA256,
    }
    try:
        fixture = json.loads(fixture_bytes)
        public = json.loads(public_bytes)
        authority = json.loads(authority_bytes)
    except json.JSONDecodeError:  # pragma: no cover - preflight records this as blocked.
        fixture, public, authority = {}, {}, {}
    ready = canary.get("mention_canary_ready_score")
    upstream_valid = clean_canary.validate_artifact(canary) == []
    quarantined = any(
        live_runtime.is_quarantined(value)
        for value in (canary, fixture)
        if isinstance(value, Mapping)
    )
    excluded = live_runtime._manifest_hits(
        exclusion_manifest,
        {EXPERIMENT_ID, "exp7264-mention-canary", UPSTREAM_PATH.as_posix()},
    )
    counts = {
        "public_units": len(public.get("rows", [])),
        "authority_units": len(authority.get("rows", [])),
        "public_held_out": sum(row.get("split") == "held_out" for row in public.get("rows", [])),
        "authority_held_out": sum(
            row.get("split") == "held_out" for row in authority.get("rows", [])
        ),
    }
    expected_counts = {
        "public_units": 72,
        "authority_units": 72,
        "public_held_out": 64,
        "authority_held_out": 64,
    }
    frozen = canary.get("frozen_heldout_settings") or {}
    no_tuning = bool(
        frozen.get("target_experiment") == EXPERIMENT_ID
        and frozen.get("seed_base") == RANDOM_SEED
        and frozen.get("frozen_before_evaluation") is True
    )
    return [
        gate_row(
            "exact_upstream_bytes",
            expected_hashes,
            observed_hashes,
            observed_hashes == expected_hashes,
            upstream="exp7264_and_exp7236",
            field="artifact_and_manifest_sha256",
        ),
        gate_row(
            "mention_canary_ready",
            1,
            ready,
            ready == 1,
            upstream="exp7264-mention-canary",
            field="mention_canary_ready_score",
        ),
        gate_row(
            "canary_cold_validation",
            True,
            upstream_valid,
            upstream_valid,
            upstream="exp7264-mention-canary",
            field="reproducibility_checksum_and_terminal_schema",
        ),
        gate_row(
            "structured_quarantine",
            False,
            quarantined,
            not quarantined,
            upstream="exp7264_and_exp7236",
            field="flagged_adversarial|quarantined|fabricated",
        ),
        gate_row(
            "exclusion_manifest",
            False,
            excluded,
            not excluded,
            upstream="ops/exclusion_manifest.yaml",
            field="experiment_ids",
        ),
        gate_row(
            "held_out_roster",
            expected_counts,
            counts,
            counts == expected_counts,
            upstream="experiment_7236",
            field="manifest_unit_counts",
        ),
        gate_row(
            "held_out_not_used_for_tuning",
            True,
            no_tuning,
            no_tuning,
            upstream="exp7264-mention-canary",
            field="frozen_heldout_settings",
        ),
    ]


def build_completion_row(
    sealed: Mapping[str, Any], response: Mapping[str, Any], resource: Mapping[str, Any]
) -> JsonDict:
    """Preserve the shipped lossless reduction plus actual UTC call times."""

    row = capture.build_completion_row(sealed, response, resource)
    row = _normalize_direct_row(row)
    row["request_started_at_utc"] = response.get("started_at_utc")
    row["response_observed_at_utc"] = response.get("completed_at_utc") or utc_now()
    row["row_sha256"] = capture._row_hash(row)
    return row


def _normalize_direct_row(row: JsonDict) -> JsonDict:
    """Map neutral direct symbols after the authentic response bytes are retained."""

    if row.get("arm") != "direct_judge":
        return row
    parsed = row.get("parsed_completion")
    if not isinstance(parsed, Mapping):
        return row
    mapping = {"a": "supported", "b": "contradicted", "c": "unknown"}
    decision = parsed.get("decision")
    if decision not in mapping:
        return row
    normalized = {"decision": mapping[str(decision)]}
    row["parsed_completion"] = normalized
    row["compiled_completion"] = deepcopy(normalized)
    row["parse_valid"] = True
    row["explicit_unknown"] = normalized["decision"] == "unknown"
    row["usable"] = bool(
        row.get("transport_complete")
        and not row.get("truncated")
        and row.get("request_bytes_match")
        and row.get("seed_join_valid")
    )
    row["errors"] = [error for error in row.get("errors", []) if "shape" not in str(error)]
    return row


def independent_replay(
    schedule: Sequence[Mapping[str, Any]], retained_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[JsonDict], list[str]]:
    """Rebuild model rows from retained bytes without issuing another call."""

    try:
        replayed = [
            _normalize_direct_row(row)
            for row in capture.replay_completion_rows(schedule, retained_rows)
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
        return [], [f"replay_error:{type(exc).__name__}:{exc}"]
    errors: list[str] = []
    for index, (rebuilt, retained) in enumerate(zip(replayed, retained_rows, strict=False)):
        rebuilt["request_started_at_utc"] = retained.get("request_started_at_utc")
        rebuilt["response_observed_at_utc"] = retained.get("response_observed_at_utc")
        rebuilt["row_sha256"] = capture._row_hash(rebuilt)
        if rebuilt != retained:
            errors.append(f"call_{index}:replay_mismatch")
    if len(replayed) != len(retained_rows) or len(schedule) != len(retained_rows):
        errors.append("replay_denominator")
    return replayed, errors


def _offset_view(row: Mapping[str, Any] | None) -> list[JsonDict]:
    """Project compiler offsets and relation meaning without repairing missing data."""

    if not row or not isinstance(row.get("compiled_completion"), Mapping):
        return []
    relations = row["compiled_completion"].get("relations") or []
    return [
        {
            "subject_start": relation.get("subject_start"),
            "subject_end": relation.get("subject_end"),
            "object_start": relation.get("object_start"),
            "object_end": relation.get("object_end"),
            "predicate": relation.get("predicate"),
            "polarity": relation.get("polarity"),
        }
        for relation in relations
        if isinstance(relation, Mapping)
    ]


def source_fidelity_rows(
    semantic_rows: Sequence[Mapping[str, Any]], completion_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Keep representation fidelity separate from final executor correctness."""

    calls = {str(row.get("call_id")): row for row in completion_rows}
    projected = []
    for row in semantic_rows:
        provenance = row.get("final_response_provenance") or {}
        source = calls.get(str(provenance.get("source_call_id")))
        claim = calls.get(str(provenance.get("claim_call_id")))
        direct = calls.get(str(provenance.get("direct_call_id")))
        errors = [
            *list((source or {}).get("errors") or []),
            *list((claim or {}).get("errors") or []),
            *list((direct or {}).get("errors") or []),
        ]
        projected.append(
            {
                "unit_id": row.get("unit_id"),
                "arm": row.get("arm"),
                "condition": row.get("condition"),
                "source_offsets": _offset_view(source),
                "claim_offsets": _offset_view(claim),
                "source_fidelity": row.get("source_fidelity"),
                "claim_fidelity": row.get("claim_fidelity"),
                "relation_and_polarity": {
                    "source": _offset_view(source),
                    "claim": _offset_view(claim),
                },
                "sufficiency": row.get("prediction") != "unknown",
                "exact_energy": int(not bool(row.get("fully_correct"))),
                "accepted_errors": sum(
                    bool(row.get(field)) for field in ("false_accept", "missing_output_penalty")
                ),
                "coverage": bool(row.get("representation_valid")),
                "direct_decision_correct": (
                    row.get("decision_correct") if row.get("arm") == "direct_judge" else None
                ),
                "decision_correct": row.get("decision_correct"),
                "errors": list(dict.fromkeys(str(error) for error in errors)),
            }
        )
    return projected


def token_cost_rows(completion_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose matched prompt, output, time, timeout, and terminal costs per call."""

    return capture.decoding_cost_rows(completion_rows)


def frozen_capture_settings() -> JsonDict:
    """Return the Exp7264 configuration that was frozen before held-out work."""

    return {
        "model": deepcopy(MODEL_SPECS[0]),
        "arms": list(ARMS),
        "representation_token_budgets": {"source": 384, "claim": 128},
        "direct_token_budget": 512,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed_base": RANDOM_SEED,
        "retry_malformed": False,
        "model_load_cap_s": MODEL_LOAD_CAP_S,
        "request_cap_s": REQUEST_CAP_S,
        "inference_deadline_s": INFERENCE_DEADLINE_S,
        "pointer_prompt_sha256": clean_canary.frozen_heldout_settings()["arms"]["mention_pointer"][
            "prompt_sha256"
        ],
        "explicit_prompt_sha256": clean_canary.frozen_heldout_settings()["arms"][
            "explicit_schema_offset_control"
        ]["prompt_sha256"],
        "direct_prompt_sha256": sha256_bytes(DIRECT_PROMPT.encode()),
        "paired_arm_orders": [list(order) for order in _paired_orders()],
        "stopping_rule": "attempt all 320 fixed calls once within 2400 seconds",
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
        "source_fidelity_rows": [],
        "token_cost_rows": [],
        "sample_size_budget": {
            "planned_independent_units": PLANNED_UNITS,
            "planned_arms": len(ARMS),
            "planned_calls": PLANNED_CALLS,
            "planned_semantic_rows": PLANNED_ROWS,
            "attempted_calls": 0,
            "completed_calls": 0,
            "censored_calls": PLANNED_CALLS,
            "completed_semantic_rows": 0,
            "stopping_rule": "attempt every fixed call once within 2400 seconds; retain every outcome",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(None),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_exp7265_running_checkpoint_only",
        "verdict_class": "partial",
        "validation_receipts": [],
        "mention_capture_complete_score": 0,
        "raw_call_manifest": {},
        "selection_receipt": {},
        "completeness_receipt": {},
        "frozen_capture_settings": frozen_capture_settings(),
        "runner_receipt": {},
        "model_identity_receipt": {},
        "gpu_receipts": {},
        "phase_spans": [],
        "token_budget_receipt": {},
        "feasibility_projection": {},
    }


def finalize_blocked_artifact(
    artifact: JsonDict, checks: Sequence[Mapping[str, Any]], duration_s: float
) -> JsonDict:
    """Finish an external block without reporting task work as partially complete."""

    failure = next((row for row in checks if row.get("passed") is not True), None)
    artifact["status"] = "blocked"
    artifact["verdict_class"] = "blocked"
    artifact["mention_capture_complete_score"] = 0
    artifact["gate_check_summary"] = gate_summary(failure)
    check = str(failure.get("check", "unknown")) if failure else "unknown"
    artifact["honest_verdict"] = f"blocked_exp7265_{check}"
    counts = artifact.get("invocation_counts") or ZERO_INVOCATION_COUNTS
    if counts.get("generation_calls_attempted", 0):  # pragma: no cover - live fault.
        artifact["inference_substrate"] = "live_llm_inference_local_gguf_sota"
        artifact["inference_substrate_class"] = "model_bounded_generation"
        artifact["inference_mode"] = "live_gpu"
    elif counts.get("model_loads_completed", 0):  # pragma: no cover - live load-only fault.
        artifact["inference_substrate"] = "model_load_no_generation"
        artifact["inference_substrate_class"] = "model_load_no_generation"
    artifact["duration_s"] = duration_s
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _validation_complete(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require each named focused command once and require every command to pass."""

    by_name = {str(row.get("name")): row for row in receipts}
    return set(by_name) == set(REQUIRED_VALIDATION_NAMES) and all(
        by_name[name].get("passed") is True and by_name[name].get("exit_code") == 0
        for name in REQUIRED_VALIDATION_NAMES
    )


def _criteria(
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    provenance_errors: Sequence[str],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], int, bool]:
    """Separate denominator completion from the strict semantic diagnostic."""

    receipt = capture.completeness_receipt(
        schedule, completion_rows, provenance_errors=provenance_errors
    )
    validation_ok = _validation_complete(validation_receipts)
    exact = len(semantic_rows) == PLANNED_ROWS and all(
        row.get("fully_correct") is True for row in semantic_rows
    )
    criteria = [
        {
            "criterion": "authenticated_provenance",
            "expected": True,
            "observed": not provenance_errors,
            "passed": not provenance_errors,
            "principle": GATE_PRINCIPLES["authenticated_provenance"],
        },
        {
            "criterion": "fixed_schedule",
            "expected": PLANNED_CALLS,
            "observed": len(schedule),
            "passed": len(schedule) == PLANNED_CALLS,
            "principle": GATE_PRINCIPLES["fixed_schedule"],
        },
        {
            "criterion": "terminal_accounting",
            "expected": PLANNED_CALLS,
            "observed": receipt["authentic_terminal_outcomes"],
            "passed": receipt["authentic_terminal_outcomes"] == PLANNED_CALLS,
            "principle": GATE_PRINCIPLES["terminal_accounting"],
        },
        {
            "criterion": "independent_rows",
            "expected": PLANNED_ROWS,
            "observed": len(semantic_rows),
            "passed": len(semantic_rows) == PLANNED_ROWS,
            "principle": GATE_PRINCIPLES["independent_rows"],
        },
        {
            "criterion": "semantic_exactness",
            "expected": PLANNED_ROWS,
            "observed": sum(row.get("fully_correct") is True for row in semantic_rows),
            "passed": exact,
            "principle": GATE_PRINCIPLES["semantic_exactness"],
        },
        {
            "criterion": "scoped_validation",
            "expected": list(REQUIRED_VALIDATION_NAMES),
            "observed": [
                str(row.get("name"))
                for row in validation_receipts
                if row.get("passed") is True and row.get("exit_code") == 0
            ],
            "passed": validation_ok,
            "principle": GATE_PRINCIPLES["scoped_validation"],
        },
    ]
    complete = int(
        receipt["mention_capture_complete_score"] == 1
        and len(semantic_rows) == PLANNED_ROWS
        and validation_ok
    )
    return criteria, complete, exact


def finalize_measured_artifact(
    artifact: JsonDict,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    provenance_errors: Sequence[str],
    validation_receipts: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Finish a full measurement and keep scientific quality separate from completion."""

    receipt = capture.completeness_receipt(
        schedule, completion_rows, provenance_errors=provenance_errors
    )
    criteria, complete, exact = _criteria(
        schedule,
        completion_rows,
        semantic_rows,
        provenance_errors,
        validation_receipts,
    )
    transport = int(receipt["transport_completed_calls"])
    usable = sum(row.get("usable") is True for row in completion_rows)
    artifact.update(
        {
            "status": "complete",
            "model_invoked": len(completion_rows) > 0,
            "invocation_counts": {
                "model_loads_attempted": 1,
                "model_loads_completed": 1,
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
            "source_fidelity_rows": source_fidelity_rows(semantic_rows, completion_rows),
            "token_cost_rows": token_cost_rows(completion_rows),
            "sample_size_budget": {
                "planned_independent_units": PLANNED_UNITS,
                "planned_arms": len(ARMS),
                "planned_calls": PLANNED_CALLS,
                "planned_semantic_rows": PLANNED_ROWS,
                "attempted_calls": len(completion_rows),
                "completed_calls": transport,
                "censored_calls": PLANNED_CALLS - len(completion_rows),
                "completed_semantic_rows": len(semantic_rows),
                "stopping_rule": "attempt every fixed call once within 2400 seconds; retain every outcome",
            },
            "acceptance_gate_results": criteria,
            "gate_check_summary": gate_summary(None),
            "honest_verdict": (
                "complete_circular_positive_heldout_semantics_exact_oracle_graded"
                if complete and exact
                else "complete_null_heldout_semantics_not_exact_or_validation_incomplete"
            ),
            "verdict_class": "circular_positive" if complete and exact else "null",
            "validation_receipts": deepcopy(list(validation_receipts)),
            "mention_capture_complete_score": complete,
            "raw_call_manifest": {
                "path": CAPTURE_MANIFEST_PATH.as_posix(),
                "schedule_sha256": sha256_json(list(schedule)),
                "raw_row_count": len(completion_rows),
            },
            "completeness_receipt": receipt,
        }
    )
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check terminal shape, fixed denominators, provenance, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping"]
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in value]
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if value.get("schema") != SCHEMA:
        errors.append("schema")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("milestone") != MILESTONE:
        errors.append("identity")
    if value.get("run_date") != RUN_DATE or value.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("run_contract")
    if value.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if value.get("random_seed") != RANDOM_SEED or value.get("execution_venue") != "host":
        errors.append("execution_contract")
    if value.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle")
    duration = value.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum")
    if value.get("status") == "blocked":
        summary = value.get("gate_check_summary")
        if (
            value.get("verdict_class") != "blocked"
            or value.get("mention_capture_complete_score") != 0
        ):
            errors.append("blocked_terminal_state")
        if not isinstance(summary, Mapping) or summary.get("passed") is not False:
            errors.append("gate_check_summary")
        return list(dict.fromkeys(errors))
    if value.get("status") != "complete":
        errors.append("status")
        return list(dict.fromkeys(errors))
    schedule = value.get("schedule")
    raw = value.get("raw_rows")
    rows = value.get("rows")
    fidelity = value.get("source_fidelity_rows")
    costs = value.get("token_cost_rows")
    if not all(isinstance(item, list) for item in (schedule, raw, rows, fidelity, costs)):
        errors.append("rows")
        return list(dict.fromkeys(errors))
    identity = value.get("model_identity_receipt")
    provenance_errors = (
        list(identity.get("identity_errors") or []) if isinstance(identity, Mapping) else []
    )
    criteria, complete, exact = _criteria(
        schedule,
        raw,
        rows,
        provenance_errors,
        value.get("validation_receipts") or [],
    )
    if len(schedule) != PLANNED_CALLS or len(raw) != PLANNED_CALLS or len(rows) != PLANNED_ROWS:
        errors.append("rows")
    if fidelity != source_fidelity_rows(rows, raw) or costs != token_cost_rows(raw):
        errors.append("derived_rows")
    if value.get("acceptance_gate_results") != criteria:
        errors.append("acceptance_gate_results")
    if value.get("mention_capture_complete_score") != complete:
        errors.append("mention_capture_complete_score")
    expected_class = "circular_positive" if complete and exact else "null"
    if value.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    counts = value.get("invocation_counts")
    transport = sum(row.get("transport_complete") is True for row in raw)
    usable = sum(row.get("usable") is True for row in raw)
    if not isinstance(counts, Mapping) or (
        counts.get("model_loads_attempted"),
        counts.get("model_loads_completed"),
        counts.get("generation_calls_attempted"),
        counts.get("generation_calls_completed"),
        counts.get("usable_answers"),
    ) != (1, 1, len(raw), transport, usable):
        errors.append("invocation_counts")
    if (
        value.get("model_invoked") is not True
        or value.get("inference_substrate") != "live_llm_inference_local_gguf_sota"
        or value.get("inference_substrate_class") != "model_bounded_generation"
        or value.get("inference_mode") != "live_gpu"
    ):
        errors.append("live_inference_provenance")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration < 10:
        errors.append("bounded_generation_duration_floor")
    manifest = value.get("raw_call_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("schedule_sha256") != sha256_json(schedule)
        or manifest.get("raw_row_count") != len(raw)
    ):
        errors.append("raw_call_manifest")
    return list(dict.fromkeys(errors))


def write_capture_manifest(
    raw_dir: Path,
    schedule: Sequence[Mapping[str, Any]],
    completion_rows: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live artifact writer.
    """Bind each raw outcome to its sealed request and authenticated model."""

    manifest = {
        "schema": "carnot.exp7265.raw_calls.v1",
        "status": "complete" if len(completion_rows) == PLANNED_CALLS else "partial",
        "created_at_utc": utc_now(),
        "model": deepcopy(dict(model_identity)),
        "scheduled_calls": PLANNED_CALLS,
        "attempted_calls": len(completion_rows),
        "schedule_sha256": sha256_json(list(schedule)),
        "authority_path_opened_by_model_worker": False,
        "retry_count": 0,
        "rows": deepcopy(list(completion_rows)),
    }
    atomic_write_json(
        raw_dir / "raw_call_manifest.json", manifest, allow_override=False, sort_keys=True
    )
    # The shipped finite runner reads this compatibility sidecar after capture.
    atomic_write_json(raw_dir / "manifest.json", manifest, allow_override=False, sort_keys=True)
    return manifest


def checkpoint_identity(
    schedule: Sequence[Mapping[str, Any]],
    public_sha256: str,
    authority_sha256: str,
    model_sha256: str,
) -> JsonDict:  # pragma: no cover - live resume boundary.
    """Bind resume permission to model, settings, manifests, and schedule."""

    return {
        "schema": "carnot.exp7265.resume.v1",
        "schedule_sha256": sha256_json(list(schedule)),
        "public_sha256": public_sha256,
        "authority_sha256": authority_sha256,
        "model_sha256": model_sha256,
        "settings_sha256": sha256_json(frozen_capture_settings()),
    }


def feasibility_projection(canary: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Project held-out generation time from all measured canary call latencies."""

    raw = list(canary.get("raw_rows") or [])
    measured = sum(float(row.get("latency_s", 0) or 0) for row in raw)
    per_call = measured / len(raw) if raw else 0.0
    projected = per_call * PLANNED_CALLS
    return {
        "source": UPSTREAM_PATH.as_posix(),
        "measured_calls": len(raw),
        "measured_generation_s": measured,
        "measured_generation_s_per_call": per_call,
        "projected_generation_s": projected,
        "startup_allowance_s": MODEL_LOAD_CAP_S,
        "projected_total_s": projected + MODEL_LOAD_CAP_S,
        "inference_budget_s": INFERENCE_DEADLINE_S,
        "projected_feasible": bool(
            len(raw) == 48 and projected + MODEL_LOAD_CAP_S <= INFERENCE_DEADLINE_S
        ),
    }


def _identity_errors(
    identity: Mapping[str, Any],
    gpu_receipts: Mapping[str, Any],
    completion_rows: Sequence[Mapping[str, Any]],
) -> list[str]:  # pragma: no cover - live provenance reducer.
    """Use the shipped provenance checker for the same model and row contract."""

    return _shipped_identity_errors(identity, gpu_receipts, completion_rows)


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - live inventory.
    """Hash every source, sealed input, and checker used by this invocation."""

    paths = {
        "agents": Path("AGENTS.md"),
        "claude": Path("CLAUDE.md"),
        "codex": Path("CODEX.md"),
        "research_program": Path("research-program.md"),
        "exclusion_manifest": EXCLUSION_PATH,
        "e2e_test_plan": Path("ops/e2e-test-plan.md"),
        "verification_spec": SPEC_PATH,
        "clean_canary": UPSTREAM_PATH,
        "fixture_artifact": FIXTURE_PATH,
        "public_manifest": PUBLIC_PATH,
        "authority_manifest": AUTHORITY_PATH,
        "capture_reducer": Path("python/carnot/experiment_7238_v637_mention_capture.py"),
        "live_runtime": Path("python/carnot/experiment_7209_v635_span_canary.py"),
        "typed_executor": Path("python/carnot/verify/experiment_7195_source_relation_executor.py"),
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
    """Flush each phase and long-operation observation for external monitoring."""

    print(
        canonical_json({"experiment": 7265, "phase": phase, "event": event, **fields}),
        flush=True,
    )


def _collect_preflight(  # pragma: no cover - live host boundary.
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_dir: Path,
    raw_dir: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
    """Authenticate Exp7264, then reuse cached-model and GPU ownership checks."""

    checks, public, authority, context = live_runtime._collect_preflight(
        root,
        run_date,
        result_path,
        checkpoint_dir,
        raw_dir,
        contract={
            "run_date": RUN_DATE,
            "upstream_path": UPSTREAM_PATH,
            "public_path": PUBLIC_PATH,
            "authority_path": AUTHORITY_PATH,
            "manifest_path": FIXTURE_PATH,
            "spec_path": SPEC_PATH,
            "module_path": MODULE_PATH,
            "wrapper_path": WRAPPER_PATH,
            "test_path": TEST_PATH,
            "spec_req": "REQ-VERIFY-7265",
            "expected_calls": PLANNED_CALLS,
            "expected_units": PLANNED_UNITS,
            "task_id": TASK_ID,
            "upstream_id": "exp7264-mention-canary",
            "split_id": "experiment_7236_held_out",
            "upstream_gate_rows": upstream_gate_rows,
            "load_split": load_held_out_manifests,
            "build_schedule": build_schedule,
            "schedule_errors": schedule_errors,
            "request_cap_s": REQUEST_CAP_S,
            "live_window_cap_s": INFERENCE_DEADLINE_S,
            "model_load_cap_s": MODEL_LOAD_CAP_S,
        },
    )
    if context.get("schedule"):
        context["selection_receipt"] = selection_receipt(public, authority, context["schedule"])
        context["completion_builder"] = build_completion_row
    return checks, public, authority, context


def independent_replay_from_raw(root: Path, raw_dir: Path) -> list[str]:  # pragma: no cover
    """Replay all task-owned raw call files and the private semantic reduction."""

    try:
        schedule = list(json.loads((raw_dir / "schedule.json").read_text())["schedule"])
        retained = [
            json.loads((raw_dir / f"call_{index:02d}.json").read_text())["completion"]
            for index in range(PLANNED_CALLS)
        ]
        public, authority = load_held_out_manifests(root / PUBLIC_PATH, root / AUTHORITY_PATH)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [f"raw_read:{type(exc).__name__}:{exc}"]
    errors = schedule_errors(schedule, public, authority)
    replayed, replay_errors = independent_replay(schedule, retained)
    errors.extend(replay_errors)
    rows = score_semantics(schedule, replayed, public, authority)
    if len(rows) != PLANNED_ROWS:
        errors.append("semantic_denominator")
    candidate = raw_dir / RAW_CANDIDATE_PATH.name
    if candidate.is_file():
        value = json.loads(candidate.read_text())
        if value.get("rows") != rows:
            errors.append("independent_semantic_reduction")
    return errors


def _validation_commands(
    root: Path, raw_dir: Path
) -> list[tuple[str, list[str]]]:  # pragma: no cover
    """Return the fixed focused commands that can publish the measured result."""

    python = str(root / ".venv/bin/python")
    coverage_file = "/tmp/.coverage-exp7265-v639"
    test = TEST_PATH.as_posix()
    affected = [
        "tests/python/test_experiment_7238_v637_mention_capture.py",
        "tests/python/test_experiment_7264_v639_mention_canary.py",
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
                "--basetemp=/tmp/exp7265-focused",
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
                "--basetemp=/tmp/exp7265-affected",
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
                "--basetemp=/tmp/exp7265-coverage",
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
        ("scoped_spec_coverage", [python, "-u", "scripts/check_spec_coverage.py", test, *affected]),
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
    """Stream each checker and retain its exact exit status and output hash."""

    validation_dir = raw_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = f"{root / 'python'}:{root}"
    receipts = []
    commands = _validation_commands(root, raw_dir)
    for index, (name, command) in enumerate(commands, start=1):
        _progress(
            9,
            "subprocess_start",
            operation=name,
            completed_units=index - 1,
            total_units=len(commands),
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - fixed local argv.
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        with live_runtime._heartbeat(9, name, lambda: index - 1, len(commands)):
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                print(f"exp7265 {name} {line.rstrip()}", flush=True)
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
            9,
            "subprocess_end",
            operation=name,
            exit_code=returncode,
            completed_units=index,
            total_units=len(commands),
        )
    return receipts


def _pending_validation_receipts() -> list[JsonDict]:  # pragma: no cover
    """Keep the raw candidate null until every checker has executed."""

    return [
        {
            "name": name,
            "command": "pending",
            "exit_code": None,
            "passed": False,
            "timed_out": False,
            "duration_s": 0.0,
            "log_path": f"results/raw/experiment_7265/validation/{name}.log",
            "log_sha256": "pending",
        }
        for name in REQUIRED_VALIDATION_NAMES
    ]


def _checkpoint(
    path: Path, artifact: Mapping[str, Any], started: float
) -> None:  # pragma: no cover
    """Write unfinished work only below the task checkpoint directory."""

    value = deepcopy(dict(artifact))
    value["duration_s"] = time.monotonic() - started
    value["reproducibility_checksum"] = artifact_checksum(value)
    atomic_write_json(path, value, allow_override=False, sort_keys=True)


def _terminal(  # pragma: no cover - live atomic publication.
    artifact: JsonDict, result_path: Path, checkpoint_path: Path, started: float
) -> JsonDict:
    """Validate the raw candidate, run focused checks, then publish atomically."""

    root = find_repo_root(start=__file__)
    raw_dir = root / RAW_DIR
    if artifact.get("status") == "complete":
        provenance = list(
            (artifact.get("model_identity_receipt") or {}).get("identity_errors") or []
        )
        finalize_measured_artifact(
            artifact,
            artifact["schedule"],
            artifact["raw_rows"],
            artifact["rows"],
            duration_s=time.monotonic() - started,
            provenance_errors=provenance,
            validation_receipts=_pending_validation_receipts(),
        )
        artifact["source_artifact_hashes"] = _source_hashes(root) | {
            "raw_call_manifest": sha256_file(raw_dir / "raw_call_manifest.json"),
            **{path.stem: sha256_file(path) for path in sorted(raw_dir.glob("call_*.json"))},
        }
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        _progress(8, "write_start", path=str(root / RAW_CANDIDATE_PATH))
        atomic_write_json(root / RAW_CANDIDATE_PATH, artifact, allow_override=False, sort_keys=True)
        _progress(8, "write_end", path=str(root / RAW_CANDIDATE_PATH))
        receipts = _run_validations(root, raw_dir)
        finalize_measured_artifact(
            artifact,
            artifact["schedule"],
            artifact["raw_rows"],
            artifact["rows"],
            duration_s=time.monotonic() - started,
            provenance_errors=provenance,
            validation_receipts=receipts,
        )
    artifact["duration_s"] = time.monotonic() - started
    artifact["timestamps"]["completed_at_utc"] = utc_now()
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _progress(10, "validation_start", path=str(result_path))
    errors = validate_artifact(artifact)
    _progress(10, "validation_end", errors=errors)
    if errors:
        raise ValueError(f"invalid Exp7265 artifact: {errors}")
    _checkpoint(checkpoint_path, artifact, started)
    _progress(11, "write_start", path=str(result_path))
    atomic_write_json(result_path, artifact, allow_override=False, sort_keys=True)
    _progress(11, "write_end", path=str(result_path))
    return artifact


@contextmanager
def _patched_capture() -> Iterator[None]:  # pragma: no cover - live reuse adapter.
    """Give the shipped finite runner this task's paths and pure contracts."""

    replacements = {
        "RUN_DATE": RUN_DATE,
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "TASK_ID": TASK_ID,
        "RANDOM_SEED": RANDOM_SEED,
        "MODEL_SPECS": MODEL_SPECS,
        "INFERENCE_DEADLINE_S": INFERENCE_DEADLINE_S,
        "RESULT_PATH": RESULT_PATH,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "RAW_DIR": RAW_DIR,
        "CAPTURE_MANIFEST_PATH": CAPTURE_MANIFEST_PATH,
        "UPSTREAM_PATH": UPSTREAM_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "artifact_checksum": artifact_checksum,
        "base_artifact": base_artifact,
        "finalize_blocked_artifact": finalize_blocked_artifact,
        "finalize_measured_artifact": finalize_measured_artifact,
        "validate_artifact": validate_artifact,
        "frozen_capture_settings": frozen_capture_settings,
        "checkpoint_identity": checkpoint_identity,
        "write_capture_manifest": write_capture_manifest,
        "feasibility_projection": feasibility_projection,
        "_identity_errors": _identity_errors,
        "_source_hashes": _source_hashes,
        "_collect_preflight": _collect_preflight,
        "_progress": _progress,
        "_terminal": _terminal,
    }
    original = {name: getattr(capture, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(capture, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(capture, name, value)


def run_experiment(  # pragma: no cover - native GPU E2E.
    root: Path | None = None,
    run_date: str = RUN_DATE,
    *,
    output_root: Path | None = None,
) -> JsonDict:
    """Run the finite shipped capture with the authenticated Exp7265 contract."""

    os.environ["CARNOT_FORCE_LIVE"] = "1"
    with _patched_capture():
        return capture.run_experiment(root=root, run_date=run_date, output_root=output_root)


def _date_argument(value: str) -> str:
    """Accept only the execution date fixed by the V639 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run capture or replay raw evidence without a second model invocation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--replay-raw", type=Path)
    args = parser.parse_args(argv)
    if args.replay_raw is not None:
        errors = independent_replay_from_raw(find_repo_root(start=__file__), args.replay_raw)
        print(canonical_json({"replay_errors": errors}), flush=True)
        return int(bool(errors))
    artifact = run_experiment(run_date=args.date)
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7265] invalid artifact: {errors}", flush=True)
        return 1
    print(
        f"[exp7265] terminal verdict={artifact['honest_verdict']} "
        f"complete={artifact['mention_capture_complete_score']}",
        flush=True,
    )
    return 0
