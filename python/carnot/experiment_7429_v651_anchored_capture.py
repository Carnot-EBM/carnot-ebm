"""Capture the frozen Exp7416 paired extraction schedule after ownership repair.

The module reuses Exp7416 for corpus selection, prompts, transport rows, and
parsing. New code checks the Exp7422 repair, gates measured work with four
development calls, persists content-addressed evidence, and independently
reduces claim-preservation endpoints.

Spec refs: REQ-VERIFY-7429 and SCENARIO-VERIFY-7429-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as native_runtime
from carnot import experiment_7400_v649_assignment_canary as canary
from carnot import experiment_7416_v650_anchored_extraction as frozen
from carnot import experiment_7422_v651_runtime_ownership as ownership
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
PHASE = 3
EXPERIMENT_ID = "exp7429-v651-anchored-capture"
TASK_ID = "experiment_7429_v651_anchored_capture"
SCHEMA = "carnot.exp7429.v651.anchored_capture.v1"
MODEL_ID = frozen.MODEL_ID
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = frozen.QUANTIZATION
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
RANDOM_SEED = frozen.RANDOM_SEED
MAX_GENERATED_TOKENS = frozen.MAX_GENERATED_TOKENS
MAX_DEVELOPMENT_CALLS = 4
DEVELOPMENT_TOKEN_BUDGET = 64
DEVELOPMENT_USABLE_MINIMUM = 3
MODEL_LOAD_TIMEOUT_S = frozen.MODEL_LOAD_TIMEOUT_S
CAPTURE_TIMEOUT_S = 2400.0
VALIDATION_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = frozen.REQUEST_TIMEOUT_S
LEASE_WAIT_TIMEOUT_S = 120.0

MODULE_PATH = Path("python/carnot/experiment_7429_v651_anchored_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7429_v651_anchored_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7429_v651_anchored_capture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_7429_v651_anchored_capture.json")
RAW_DIR = Path("results/raw/experiment_7429_v651_anchored_capture")
EXP7416_PATH = Path("results/experiment_7416_v650_anchored_extraction.json")
EXP7422_PATH = Path("results/experiment_7422_v651_runtime_ownership.json")
EXPECTED_EXP7416_SHA256 = "sha256:a67c9e94f7b3a23d53a5b27dfaa6975912e6f10ae9df5d32ecd1677c1fac98ce"
EXPECTED_EXP7422_SHA256 = "sha256:5217e73f74804a5020d884afbd58ad410d5b49cf107249a751a55deeccbee31a"
EXPECTED_SCHEDULE_SHA256 = "sha256:10e85d829654699a659f2269087344673a64441ccb991bb4d15d3989f519edeb"

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Use a versioned plain top-level schema with experiment, milestone, and terminal status.",
    "run_date": "Use 20260919 with actual UTC boundaries and monotonic phase timing.",
    "preconditions_checked": "Record exact resource, path, corpus, protocol, schedule, and ownership observations before dependent work.",
    "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for every current LLM attempt; detailed resolved facts stay in model_specs.",
    "model_invoked": "Set true only after actual current model load or generation work is attempted.",
    "invocation_counts": "Reduce actual owned current load and generation attempts, completions, failures, cancellations, and in-flight work.",
    "inference_substrate": "Use a truthful string and keep device, model, and runner facts in inference_substrate_details.",
    "inference_substrate_class": "Use model_bounded_generation, model_load_no_generation, or no_model_load from current event counts.",
    "execution_venue": "Use the closed host value; CPU, CUDA, and board identities are separate details.",
    "duration_s": "Measure current work without padding and separate scientific, validation, and cold-reader durations.",
    "phase_spans": "Keep real phase times, flushed boundaries, completed units, and checkpoint references.",
    "random_seed": "Freeze selection, arm order, development, sampling, and resampling seeds.",
    "reproducibility_checksum": "Bind code, protocol, inputs, raw rows, current events, reductions, and validation scope.",
    "source_artifact_hashes": "Hash exact input and shard bytes and preserve historical flags without importing historical calls.",
    "rows": "Keep every measured case, arm, seed, condition, and terminal disposition.",
    "sample_size_budget": "Keep planned, attempted, completed, failed, censored, and unstarted counts with fixed independent groups and stop rules.",
    "acceptance_gate_results": "Keep completion, evidence, validation, safety, and scientific value checks separate.",
    "gate_check_summary": "Name exact upstream, path, check, field, operator, expected value, and observed value.",
    "verifier_is_oracle": "Use true where constructed source fixtures define endpoint correctness; positive fixture claims are circular.",
    "honest_verdict": "Start completed findings with complete_ and unchanged unavailable prerequisites with blocked_.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings because flagged evidence cannot supply scientific readiness.",
    "validation_receipts": "Retain exact scoped arguments, environments, exits, durations, names, and hashed logs.",
    "field_principles": "Explain field intent separately and keep gate values as ordinary scalars.",
    "promotion_score": "Keep zero because capture changes no rollout, publication, defaults, or generator weights.",
    "extraction_capture_complete_score": "Set one for a valid fully accounted paired capture independently of semantic gain.",
    "extraction_value_score": "Set one only for qualified claim preservation under the frozen paired comparison.",
    "extraction_rows": "Keep all 48 cases by two arms, including failed, censored, truncated, and unstarted rows.",
    "runner_receipt": "Record the current native server PID, start tick, lease, model hash, runtime flags, and decoding parameters.",
    "usable_output_count": "Count raw-derived usable non-truncated measured replies, not transport successes.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)

sha256_file = frozen.sha256_file
canonical_hash = frozen.canonical_hash
atomic_json = frozen.atomic_json
load_object = frozen.load_object
artifact_checksum = frozen.artifact_checksum
gate_row = frozen.gate_row
gate_check_summary = frozen.gate_check_summary
zero_counts = frozen.zero_counts
substrate_class_from_counts = frozen.substrate_class_from_counts


def utc_now() -> str:  # pragma: no cover - actual wall-clock evidence.
    """Return one current UTC boundary while elapsed time stays monotonic."""

    return frozen.utc_now()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase, long operation, heartbeat, and checkpoint boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7429] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def build_frozen_schedule(cases: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reuse the Exp7416 builder so prompts and arm order cannot drift."""

    return frozen.build_schedule(cases)


def schedule_identity(schedule: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete schedule rather than a selected field subset."""

    return canonical_hash(list(schedule))


def runtime_ownership_gate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Check the three exact Exp7422 operands that authorize a new capture."""

    fields = (
        ("runtime_ownership_ready", "runtime_ownership_ready_score", "==", 1),
        (
            "runtime_ownership_verdict_eligible",
            "verdict_class",
            "in",
            ["positive", "circular_positive", "null"],
        ),
        ("runtime_ownership_unflagged", "flagged_adversarial", "==", False),
    )
    return [
        gate_row(
            check,
            EXP7422_PATH.as_posix(),
            field,
            operator,
            expected,
            artifact.get(field),
            principle="Only the completed ownership repair can authorize current model work.",
        )
        for check, field, operator, expected in fields
    ]


def historical_zero_attempt_gate(artifact: Mapping[str, Any]) -> JsonDict:
    """Reduce the historical invocation ledger to one provenance-scoped scalar."""

    counts = artifact.get("invocation_counts") or {}
    attempted = int(counts.get("model_loads_attempted", 0) or 0) + int(
        counts.get("generation_calls_attempted", 0) or 0
    )
    return gate_row(
        "exp7416_zero_attempts",
        EXP7416_PATH.as_posix(),
        "historical_total_attempted_invocations",
        "==",
        0,
        attempted,
        principle="A measured predecessor would make this a replacement cohort.",
    )


def build_development_schedule(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze four parser-readiness calls without changing measured calls."""

    if len(schedule) != 96:
        raise ValueError("measured_schedule_count")
    rows: list[JsonDict] = []
    for index, value in enumerate(schedule[:MAX_DEVELOPMENT_CALLS]):
        row = deepcopy(dict(value))
        row.update(
            {
                "call_index": index,
                "call_id": f"development:{value['call_id']}",
                "request_id": f"development:{value['request_id']}",
                "capture_phase": "development",
                "max_new_tokens": DEVELOPMENT_TOKEN_BUDGET,
            }
        )
        rows.append(row)
    return rows


def _usable(row: Mapping[str, Any]) -> bool:
    """Require a parsed nonempty reply whose native stop reason is not truncation."""

    return bool(
        row.get("terminal_state") == "response"
        and row.get("finish_reason") not in {"length", "max_tokens"}
        and row.get("parse_valid") is True
        and int(row.get("triple_count", 0) or 0) > 0
        and str(row.get("raw_reply") or "").strip()
    )


def reduce_development_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Open measured capture only after three of four fixed replies are usable."""

    values = [dict(row) for row in rows]
    usable = sum(_usable(row) for row in values)
    attempted = sum(row.get("attempted") is True for row in values)
    completed = sum(row.get("terminal_state") == "response" for row in values)
    opened = len(values) == MAX_DEVELOPMENT_CALLS and usable >= DEVELOPMENT_USABLE_MINIMUM
    return {
        "planned": MAX_DEVELOPMENT_CALLS,
        "attempted": attempted,
        "completed": completed,
        "usable_output_count": usable,
        "required_usable_output_count": DEVELOPMENT_USABLE_MINIMUM,
        "capture_open": opened,
        "terminal_class": "ready" if opened else "null",
    }


def write_content_addressed_shard(root: Path, phase: str, value: Mapping[str, Any]) -> JsonDict:
    """Write immutable JSON bytes under their digest and return a byte receipt."""

    encoded = (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    relative = Path(phase) / f"sha256-{digest}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("content_address_collision")
    else:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return {
        "path": relative.as_posix(),
        "sha256": f"sha256:{digest}",
        "bytes": len(encoded),
        "phase": phase,
    }


def _argument_text(value: Any) -> str:
    """Read text from either free strings or anchored argument objects."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("text"), str):
        return str(value["text"])
    return ""


def independent_semantic_endpoints(row: Mapping[str, Any]) -> JsonDict:
    """Use constructed authority only for direction and visible marker retention."""

    empty = {
        "relation_direction_valid": None,
        "negation_retained": None,
        "quantifier_retained": None,
    }
    if row.get("corpus") != "constructed_challenge" or row.get("parse_valid") is not True:
        return empty
    triples = row.get("decoded_triples")
    source = row.get("source_relation")
    if not isinstance(triples, list) or not isinstance(source, list) or len(source) != 3:
        direction: bool | None = None
    else:
        expected_subject = str(source[0]).strip().casefold()
        expected_relation = str(source[1]).strip().casefold()
        expected_object = str(source[2]).strip().casefold()
        direction = any(
            isinstance(triple, Mapping)
            and _argument_text(triple.get("subject")).strip().casefold() == expected_subject
            and str(triple.get("relation") or "").strip().casefold() == expected_relation
            and _argument_text(triple.get("object")).strip().casefold() == expected_object
            for triple in triples
        )
    answer = str(row.get("answer") or "").casefold()
    rendered = json.dumps(triples, sort_keys=True).casefold()
    negation: bool | None = None
    if row.get("family") == "negation":
        markers = {word for word in ("not", "no", "never") if word in answer.split()}
        negation = all(word in rendered for word in markers) if markers else None
    quantifier: bool | None = None
    if row.get("family") in {"time_qualifier", "count_mismatch", "unit_mismatch"}:
        tokens = frozen._WORD.findall(answer)
        markers = {
            token
            for token in tokens
            if token.isdigit() or token in {"kg", "kilogram", "kilograms", "pound", "pounds"}
        }
        quantifier = all(token in rendered for token in markers) if markers else None
    return {
        "relation_direction_valid": direction,
        "negation_retained": negation,
        "quantifier_retained": quantifier,
    }


def _metric(values: Sequence[Any]) -> JsonDict:
    """Reduce one Boolean endpoint while retaining every assigned call."""

    known = [value for value in values if isinstance(value, bool)]
    passed = sum(value is True for value in known)
    return {
        "numerator": passed,
        "denominator": len(values),
        "known": len(known),
        "unknown": len(values) - len(known),
        "rate_all_assigned": passed / len(values) if values else None,
        "rate_known": passed / len(known) if known else None,
    }


_DERIVED_FIELDS = {
    "parse_valid",
    "parse_error",
    "decoded_triples",
    "triple_count",
    "coverage_valid",
    "argument_anchoring_valid",
    "qualifier_anchoring_valid",
    "argument_spans",
    "qualifier_spans",
    "semantic_judgment",
    "retry_count",
    "repair_attempted",
    "teacher_span_overlap",
    "qualifier_retained",
    "relation_direction_valid",
    "negation_retained",
    "quantifier_retained",
    "metric",
    "metric_value",
    "cost",
}


def _reparse(row: Mapping[str, Any]) -> JsonDict:
    """Recompute all parser and semantic fields from immutable transport evidence."""

    raw = {key: deepcopy(value) for key, value in row.items() if key not in _DERIVED_FIELDS}
    parsed = frozen.parse_capture_row(raw)
    parsed.update(independent_semantic_endpoints(parsed))
    return parsed


def _value_score(metrics: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> int:
    """Require known non-degradation across every declared preservation endpoint."""

    free = metrics.get("free") or {}
    anchored = metrics.get("anchored") or {}
    names = (
        "json_parse_validity",
        "claim_coverage",
        "relation_direction",
        "negation_retention",
        "quantifier_retention",
    )
    for name in names:
        free_row = free.get(name) or {}
        anchored_row = anchored.get(name) or {}
        if name in {"relation_direction", "negation_retention", "quantifier_retention"} and (
            int(free_row.get("known", 0) or 0) == 0 or int(anchored_row.get("known", 0) or 0) == 0
        ):
            return 0
        if float(anchored_row.get("rate_all_assigned", 0.0) or 0.0) < float(
            free_row.get("rate_all_assigned", 0.0) or 0.0
        ):
            return 0
    return 1


def reduce_capture(
    rows: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Independently reparse all 96 rows and reduce both arms without dropping failures."""

    if len(rows) != 96 or len(cases) != 48:
        raise ValueError("fixed_panel_shape")
    parsed = [_reparse(row) for row in rows]
    inherited = frozen.reduce_extractions(parsed, cases)
    arm_metrics: JsonDict = {}
    for arm in ("free", "anchored"):
        arm_rows = [row for row in parsed if row.get("arm") == arm]
        arm_metrics[arm] = {
            "json_parse_validity": _metric([row.get("parse_valid") for row in arm_rows]),
            "claim_coverage": _metric([row.get("coverage_valid") for row in arm_rows]),
            "relation_direction": _metric(
                [row.get("relation_direction_valid") for row in arm_rows]
            ),
            "negation_retention": _metric([row.get("negation_retained") for row in arm_rows]),
            "quantifier_retention": _metric([row.get("quantifier_retained") for row in arm_rows]),
        }
    usable = sum(_usable(row) for row in parsed)
    return {
        **inherited,
        "extraction_rows": parsed,
        "arm_metrics": arm_metrics,
        "usable_output_count": usable,
        "extraction_value_score": _value_score(arm_metrics),
        "official_scope": {
            "case_count": 24,
            "call_count": 48,
            "authority": "machine_annotation_not_truth",
        },
        "constructed_scope": {
            "case_count": 24,
            "call_count": 48,
            "independent_pair_count": 8,
            "authority": "constructed_exact_fixture",
        },
    }


def _base_artifact() -> JsonDict:
    """Return the complete ordinary-field shape shared by all dispositions."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "run_date": RUN_DATE,
        "status": "blocked_unstarted",
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": zero_counts(),
        "current_invocation_events": [],
        "current_run_id": None,
        "current_owner_pid": None,
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_substrate": "no_model_load",
        "inference_substrate_details": {},
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "duration_s": 0.0,
        "scientific_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "cold_start_validation_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "selection": RANDOM_SEED,
            "arm_order": RANDOM_SEED,
            "development": RANDOM_SEED,
            "sampling": RANDOM_SEED,
            "resampling": RANDOM_SEED,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 96,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 96,
            "independent_groups": 48,
            "case_count": 48,
            "official_case_count": 24,
            "constructed_case_count": 24,
            "independent_constructed_pair_count": 8,
            "measured_calls_per_case": 2,
            "maximum_new_tokens_per_call": MAX_GENERATED_TOKENS,
            "development_call_limit": MAX_DEVELOPMENT_CALLS,
            "development_token_limit": DEVELOPMENT_TOKEN_BUDGET,
            "development_usable_minimum": DEVELOPMENT_USABLE_MINIMUM,
            "development_calls_used": 0,
            "capture_timeout_s": CAPTURE_TIMEOUT_S,
            "validation_timeout_s": VALIDATION_TIMEOUT_S,
            "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
            "stop_rule": "four_fixed_development_calls_then_96_dispositions_or_capture_deadline_without_replacement",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_unstarted",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "unrelated_broad_suite_observations": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "extraction_capture_complete_score": 0,
        "extraction_value_score": None,
        "extraction_rows": [],
        "runner_receipt": None,
        "usable_output_count": 0,
        "development_rows": [],
        "development_gate": {},
        "raw_capture_manifest": [],
        "arm_metrics": {},
        "official_scope": {},
        "constructed_scope": {},
    }


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Publish unchanged external absence without inventing current model work."""

    artifact = _base_artifact()
    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary.get('check') or 'precondition'}",
            "verdict_class": "blocked",
            "extraction_value_score": None,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _fixture_reply(call: Mapping[str, Any]) -> str:
    """Create one deterministic parser fixture for terminal artifact tests."""

    answer = str(call["answer"])
    subject = answer.split()[0]
    object_text = answer.rstrip(".").split()[-1]
    if call["arm"] == "free":
        triple: JsonDict = {
            "subject": subject,
            "relation": "mentions",
            "object": object_text,
            "qualifiers": [],
        }
    else:
        object_start = answer.rfind(object_text)
        triple = {
            "subject": {"text": subject, "span": [0, len(subject)]},
            "relation": "mentions",
            "object": {
                "text": object_text,
                "span": [object_start, object_start + len(object_text)],
            },
            "qualifiers": [],
        }
    return json.dumps({"triples": [triple]}, sort_keys=True)


def _fixture_row(call: Mapping[str, Any]) -> JsonDict:
    """Build one immutable fixture row through the shipped raw reader."""

    reply = _fixture_reply(call)
    response = {
        "raw_request": {"messages": [{"role": "user", "content": call["prompt"]}]},
        "raw_response": {"choices": [{"message": {"content": reply}}]},
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": "response",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "latency_s": 0.2,
    }
    return frozen.build_raw_capture_row(call, response, {})


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic complete artifact through production reducers."""

    cases = frozen._fixture_cases()
    schedule = build_frozen_schedule(cases)
    raw_rows = [_fixture_row(call) for call in schedule]
    reduced = reduce_capture(raw_rows, cases)
    development_schedule = build_development_schedule(schedule)
    development_rows = [_reparse(_fixture_row(call)) for call in development_schedule]
    development = reduce_development_gate(development_rows)
    counts = zero_counts()
    counts.update(
        {
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 100,
            "generation_calls_completed": 100,
        }
    )
    receipts = [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    model_spec = {
        "hf_id": MODEL_ID,
        "revision": "fixture-revision",
        "quantization": QUANTIZATION,
        "sha256": "sha256:" + "1" * 64,
        "native_chat_template": True,
        "runtime_flags": {"n_gpu_layers": "all", "split_mode": "none"},
        "decoding": {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED,
            "retry_budget": 0,
        },
    }
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_extraction_capture",
            "started_at_utc": "2026-09-19T00:00:00Z",
            "completed_at_utc": "2026-09-19T00:00:20Z",
            "model_specs": [model_spec],
            "model_invoked": True,
            "invocation_counts": counts,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_details": {"gpu_name": "NVIDIA GeForce RTX 3090"},
            "inference_substrate_class": "model_bounded_generation",
            "duration_s": 20.0,
            "scientific_duration_s": 12.0,
            "validation_duration_s": 8.0,
            "cold_start_validation_duration_s": 1.0,
            "source_artifact_hashes": {"fixture": {"sha256": canonical_hash("fixture")}},
            "rows": deepcopy(reduced["extraction_rows"]),
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": 96,
                "completed": 96,
                "failed": 0,
                "censored": 0,
                "unstarted": 0,
                "development_calls_used": 4,
            },
            "acceptance_gate_results": [
                gate_row(
                    "capture_accounted",
                    EXPERIMENT_ID,
                    "planned_call_count",
                    "==",
                    96,
                    96,
                    principle="Every planned measured call remains accountable.",
                    category="completion",
                )
            ],
            "honest_verdict": "complete_null_anchored_capture_no_qualified_value_gain",
            "verdict_class": "null",
            "validation_receipts": receipts,
            "repository_health": {
                "status": "healthy",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": False,
            },
            "extraction_capture_complete_score": 1,
            "runner_receipt": {
                "owned_by_task": True,
                "pid": 1234,
                "pid_start_ticks": 5678,
                "lease_id": "fixture-lease",
                "lease_released": True,
                "all_layers_offloaded": True,
                "model_sha256": model_spec["sha256"],
                "model_revision": model_spec["revision"],
                "runtime_flags": model_spec["runtime_flags"],
                "decoding": model_spec["decoding"],
            },
            "development_rows": development_rows,
            "development_gate": development,
            "raw_capture_manifest": frozen._raw_manifest(reduced["extraction_rows"]),
            **reduced,
        }
    )
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    artifact["gate_check_summary"] = gate_check_summary(artifact["acceptance_gate_results"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _model_spec_errors(value: Mapping[str, Any]) -> list[str]:
    """Require the resolved identity, template, flags, and decoding contract."""

    specs = value.get("model_specs") or []
    if value.get("model_invoked") is not True:
        return []
    if not isinstance(specs, list) or len(specs) != 1 or not isinstance(specs[0], Mapping):
        return ["resolved_model_spec_missing"]
    spec = specs[0]
    required = {
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "native_chat_template": True,
    }
    errors = [
        f"resolved_model_spec_mismatch:{field}"
        for field, expected in required.items()
        if spec.get(field) != expected
    ]
    for field in ("revision", "sha256", "runtime_flags", "decoding"):
        if not spec.get(field):
            errors.append(f"resolved_model_spec_missing:{field}")
    decoding = spec.get("decoding") or {}
    if decoding.get("max_new_tokens") != 384 or decoding.get("retry_budget") != 0:
        errors.append("resolved_model_decoding_mismatch")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, require_terminal: bool = False, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check identity, raw rows, reductions, current counts, and terminal receipts."""

    del root
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "execution_venue": "host",
        "promotion_score": 0,
        "verifier_is_oracle": True,
        "MODEL_SPECS": MODEL_SPECS,
    }
    errors.extend(
        f"declaration_mismatch:{field}"
        for field, expected_value in expected.items()
        if value.get(field) != expected_value
    )
    if set(value.get("field_principles") or {}) != REQUIRED_FIELDS:
        errors.append("field_principles_mismatch")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    counts = value.get("invocation_counts") or {}
    if value.get("inference_substrate_class") != substrate_class_from_counts(counts):
        errors.append("substrate_class_mismatch")
    invoked = (
        int(counts.get("model_loads_attempted", 0) or 0)
        + int(counts.get("generation_calls_attempted", 0) or 0)
        > 0
    )
    if value.get("model_invoked") is not invoked:
        errors.append("model_invoked_mismatch")
    errors.extend(_model_spec_errors(value))
    rows = value.get("extraction_rows") or []
    if value.get("rows") != rows:
        errors.append("rows_alias_mismatch")
    if value.get("verdict_class") == "blocked":
        if rows or value.get("extraction_capture_complete_score") != 0:
            errors.append("blocked_rows_or_score_invalid")
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
    elif not isinstance(rows, list) or len(rows) != 96:
        errors.append("extraction_row_count_mismatch")
    else:
        if all(isinstance(row, Mapping) for row in rows):
            cases_by_id: dict[str, JsonDict] = {}
            for row in rows:
                case_id = str(row.get("case_id") or "")
                cases_by_id.setdefault(case_id, deepcopy(dict(row)))
            cases = list(cases_by_id.values())
            if len(cases) != 48:
                errors.append("case_identity_mismatch")
            else:
                try:
                    reduced = reduce_capture(rows, cases)
                except (TypeError, ValueError) as exc:
                    errors.append(f"independent_reduction_failed:{type(exc).__name__}:{exc}")
                else:
                    comparisons = {
                        "usable_output_count": "usable_output_count_mismatch",
                        "arm_metrics": "arm_metrics_mismatch",
                        "extraction_value_score": "extraction_value_score_mismatch",
                        "official_scope": "official_scope_mismatch",
                        "constructed_scope": "constructed_scope_mismatch",
                    }
                    for field, error in comparisons.items():
                        if value.get(field) != reduced.get(field):
                            errors.append(error)
                    expected_budget = {
                        "planned": 96,
                        "attempted": reduced["attempted_call_count"],
                        "completed": reduced["completed_call_count"],
                        "failed": reduced["failed_call_count"],
                        "censored": reduced["censored_call_count"],
                        "unstarted": reduced["unstarted_call_count"],
                    }
                    budget = value.get("sample_size_budget") or {}
                    if any(
                        budget.get(key) != observed for key, observed in expected_budget.items()
                    ):
                        errors.append("sample_size_budget_mismatch")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"row_shape:{index}")
                continue
            errors.extend(frozen._hash_matches(row, index))
            if row.get("persisted_before_parse") is not True:
                errors.append(f"raw_not_persisted_before_parse:{index}")
    development_rows = value.get("development_rows") or []
    if value.get("verdict_class") != "blocked":
        if len(development_rows) != 4:
            errors.append("development_row_count_mismatch")
        elif value.get("development_gate") != reduce_development_gate(development_rows):
            errors.append("development_gate_mismatch")
    receipts = value.get("validation_receipts") or []
    if require_terminal and value.get("verdict_class") != "blocked":
        names = {
            row.get("name")
            for row in receipts
            if row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
        }
        missing = set((*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)) - names
        if missing:
            errors.append("required_validation_receipts_missing:" + ",".join(sorted(missing)))
        runner = value.get("runner_receipt") or {}
        expected_complete = int(
            not missing
            and len(rows) == 96
            and value.get("flagged_adversarial") is False
            and dict(value.get("development_gate") or {}).get("capture_open") is True
            and runner.get("all_layers_offloaded") is True
        )
        if value.get("extraction_capture_complete_score") != expected_complete:
            errors.append("extraction_capture_complete_score_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Reparse immutable replies and compare every stored independent endpoint."""

    return validate_artifact(artifact, require_terminal=False, root=root)


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate Exp7412, the frozen Exp7416 question, and Exp7422 readiness."""

    checks, context = frozen.collect_preconditions(root)
    required = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, EXP7416_PATH, EXP7422_PATH)
    for relative in required:
        path = root / relative
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None,
                principle="Required bytes must exist before dependent work.",
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-VERIFY-7429",
            "REQ-VERIFY-7429" if "REQ-VERIFY-7429" in spec else None,
            principle="Implementation starts only after the capability requirement exists.",
        )
    )
    exp7416 = load_object(root / EXP7416_PATH)
    exp7422 = load_object(root / EXP7422_PATH)
    checks.extend(
        [
            gate_row(
                "exp7416_artifact_hash",
                EXP7416_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_EXP7416_SHA256,
                sha256_file(root / EXP7416_PATH) if (root / EXP7416_PATH).is_file() else None,
                principle="The original zero-attempt result must remain immutable.",
            ),
            gate_row(
                "exp7416_identity",
                EXP7416_PATH.as_posix(),
                "experiment_id",
                "==",
                "exp7416-anchored-extraction",
                exp7416.get("experiment_id"),
                principle="Only the original unmeasured extraction question may be resumed.",
            ),
            historical_zero_attempt_gate(exp7416),
            gate_row(
                "exp7422_artifact_hash",
                EXP7422_PATH.as_posix(),
                "sha256",
                "==",
                EXPECTED_EXP7422_SHA256,
                sha256_file(root / EXP7422_PATH) if (root / EXP7422_PATH).is_file() else None,
                principle="The ownership repair bytes must remain exact.",
            ),
            *runtime_ownership_gate_rows(exp7422),
            gate_row(
                "frozen_exp7416_schedule",
                "authenticated_exp7412_predictor_views",
                "schedule_sha256",
                "==",
                EXPECTED_SCHEDULE_SHA256,
                schedule_identity(context.get("schedule") or []),
                principle="The rerun must answer the exact previously unmeasured question.",
            ),
        ]
    )
    exclusion = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7429" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            "==",
            False,
            excluded,
            principle="A retired unchanged task cannot run automatically.",
        )
    )
    context.update(
        {
            "exp7416": exp7416,
            "exp7422": exp7422,
            "exp7416_sha256": EXPECTED_EXP7416_SHA256,
            "exp7422_sha256": EXPECTED_EXP7422_SHA256,
        }
    )
    return checks, context


def _runtime_preconditions(
    root: Path, context: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover - host and GPU dependent.
    """Reuse model preflight and apply the repaired capacity predicate."""

    inherited = frozen._runtime_preconditions(root, context, started)
    checks = [
        row
        for row in inherited
        if row.get("check") not in {"one_owned_rtx3090_slot", "bounded_lease_wait"}
    ]
    capacity = ownership.reduce_capacity_row(
        "exp7429-host",
        context.get("process_rows") or [],
        context.get("lease_rows") or [],
        context.get("gpu_query_receipts") or [],
    )
    context["available_gpu_uuids"] = list(capacity["available_gpu_uuids"])
    context["capacity_receipt"] = capacity
    checks.extend(
        [
            gate_row(
                "repaired_gpu_inventory_query",
                EXP7422_PATH.as_posix(),
                "query_ok",
                "==",
                True,
                capacity["query_ok"],
                principle="Inventory transport and capacity are independent facts.",
            ),
            gate_row(
                "one_owned_rtx3090_slot",
                EXP7422_PATH.as_posix(),
                "available_rtx3090_slots",
                ">=",
                1,
                capacity["available_capacity"],
                principle="At least one available device is capacity, not yet ownership.",
            ),
            gate_row(
                "bounded_lease_wait",
                "runtime_policy",
                "lease_wait_timeout_s",
                "==",
                LEASE_WAIT_TIMEOUT_S,
                LEASE_WAIT_TIMEOUT_S,
                principle="Lease acquisition cannot consume an unbounded task window.",
            ),
        ]
    )
    model_spec = dict(context.get("model_spec") or {})
    model_spec["runtime_flags"] = {
        "n_gpu_layers": "all",
        "split_mode": "none",
        "parallel": 1,
        "fit": "off",
        "offline": True,
        "jinja": True,
        "reasoning": "off",
    }
    decoding = dict(model_spec.get("decoding") or {})
    decoding.update(
        {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED,
            "repair_attempts": 0,
            "retry_budget": 0,
        }
    )
    model_spec["decoding"] = decoding
    context["model_spec"] = model_spec
    return checks


class _DevelopmentGateClosed(RuntimeError):  # pragma: no cover - live boundary.
    """Stop the shipped loop after a completed but unusable development gate."""


class _ShardRecorder(canary.InvocationEventRecorder):  # pragma: no cover - live evidence.
    """Flush each owned invocation event before the native loop continues."""

    def __init__(self, *args: Any, shard_root: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.shard_root = shard_root
        self.shards: list[JsonDict] = []
        self.development_rows: list[JsonDict] = []

    def _append(self, call_id: str, operation: str, state: str) -> None:
        super()._append(call_id, operation, state)
        self.shards.append(
            write_content_addressed_shard(self.shard_root, "events", self.events[-1])
        )

    def __call__(self, phase: str, event: str, **details: Any) -> None:
        super().__call__(phase, event, **details)
        if (phase, event) == ("generation", "after_call") and details.get("completed") == 4:
            gate = reduce_development_gate(self.development_rows)
            if gate["capture_open"] is not True:
                raise _DevelopmentGateClosed(
                    f"usable_development_outputs:{gate['usable_output_count']}_of_4"
                )


@contextmanager
def _shared_runtime_settings(
    recorder: _ShardRecorder,
    response_shards: list[JsonDict],
    raw_root: Path,
) -> Iterator[None]:  # pragma: no cover - live globals are restored in finally.
    """Apply fixed decoding and raw-first callbacks to one shipped native server."""

    original_chat = native_runtime._chat_request
    call_counter = 0

    def chat(port: int, prompt: str, timeout_s: float) -> JsonDict:
        nonlocal call_counter
        old_tokens = native_runtime.MAX_GENERATED_TOKENS
        native_runtime.MAX_GENERATED_TOKENS = (
            DEVELOPMENT_TOKEN_BUDGET
            if call_counter < MAX_DEVELOPMENT_CALLS
            else MAX_GENERATED_TOKENS
        )
        try:
            return original_chat(port, prompt, timeout_s)
        finally:
            call_counter += 1
            native_runtime.MAX_GENERATED_TOKENS = old_tokens

    def build_row(
        *,
        call_index: int,
        request: Mapping[str, Any],
        prompt: str,
        response: Mapping[str, Any],
        runtime_identity: Mapping[str, Any],
    ) -> JsonDict:
        schedule = deepcopy(dict(request))
        schedule["call_index"] = call_index
        schedule["prompt"] = prompt
        row = frozen.build_raw_capture_row(schedule, response, runtime_identity)
        response_shards.append(write_content_addressed_shard(raw_root, "responses", row))
        atomic_json(
            raw_root / "capture_checkpoint.json",
            {
                "status": "nonterminal_resumable",
                "completed_response_units": len(response_shards),
                "latest_response_shard": response_shards[-1],
            },
        )
        if request.get("capture_phase") == "development":
            recorder.development_rows.append(_reparse(row))
        return row

    values = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "INFERENCE_WINDOW_TIMEOUT_S": CAPTURE_TIMEOUT_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": RANDOM_SEED,
            "evaluation": RANDOM_SEED,
            "resampling": RANDOM_SEED,
        },
        "render_public_prompt": lambda request: str(request["prompt"]),
        "build_call_row": build_row,
        "progress": recorder,
        "_chat_request": chat,
    }
    previous = {name: getattr(native_runtime, name) for name in values}
    try:
        for name, value in values.items():
            setattr(native_runtime, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(native_runtime, name, value)


def _normalize_raw_row(schedule: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
    """Bind a native response or filler disposition to the exact scheduled unit."""

    if response.get("raw_request_sha256"):
        row = deepcopy(dict(response))
        row.update(deepcopy(dict(schedule)))
        return row
    return frozen.build_raw_capture_row(
        schedule,
        response,
        response.get("runtime_identity_receipt") or {},
    )


def _capture_current(
    context: Mapping[str, Any], raw_dir: Path, started: float
) -> JsonDict:  # pragma: no cover - live model work.
    """Run four development calls and the frozen 96 calls through one owned server."""

    recorder = _ShardRecorder(
        f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}",
        os.getpid(),
        started,
        shard_root=raw_dir,
    )
    measured_schedule = deepcopy(list(context["schedule"]))
    development_schedule = build_development_schedule(measured_schedule)
    combined = [*development_schedule, *measured_schedule]
    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = combined
    response_shards: list[JsonDict] = []
    with _shared_runtime_settings(recorder, response_shards, raw_dir):
        capture = native_runtime._live_capture(runtime_context, raw_dir / "native")
    recorder.close(capture)
    native_rows = list(capture.get("rows") or [])
    development_raw = [
        _normalize_raw_row(schedule, native_rows[index] if index < len(native_rows) else {})
        for index, schedule in enumerate(development_schedule)
    ]
    measured_raw = [
        _normalize_raw_row(
            schedule,
            native_rows[index + MAX_DEVELOPMENT_CALLS]
            if index + MAX_DEVELOPMENT_CALLS < len(native_rows)
            else {},
        )
        for index, schedule in enumerate(measured_schedule)
    ]
    development_rows = [_reparse(row) for row in development_raw]
    measured_rows = [_reparse(row) for row in measured_raw]
    final_shards = [
        write_content_addressed_shard(raw_dir, "measured", row) for row in measured_rows
    ]
    capture.update(
        {
            "development_rows": development_rows,
            "development_gate": reduce_development_gate(development_rows),
            "rows": measured_rows,
            "current_run_id": recorder.run_id,
            "current_owner_pid": recorder.owner_pid,
            "current_invocation_events": recorder.events,
            "event_shards": recorder.shards,
            "response_shards": response_shards,
            "measured_shards": final_shards,
        }
    )
    atomic_json(
        raw_dir / "capture_checkpoint.json",
        {
            "status": "capture_complete_validation_pending",
            "development_gate": capture["development_gate"],
            "measured_dispositions": len(measured_rows),
            "event_shards": recorder.shards,
            "response_shards": response_shards,
            "measured_shards": final_shards,
        },
    )
    return capture


def _span(
    spans: list[JsonDict],
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str | None = None,
) -> None:  # pragma: no cover - actual timing evidence.
    """Close one real phase with its monotonic boundaries and checkpoint."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "start_s": round(phase_started - run_started, 6),
            "end_s": round(ended - run_started, 6),
            "duration_s": round(ended - phase_started, 6),
            "completed_units": completed_units,
            "heartbeat_times_s": [],
            "checkpoint_times_s": [round(ended - run_started, 6)],
            "checkpoint_at_utc": utc_now(),
            "checkpoint": checkpoint,
        }
    )


def _run_affected_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - subprocess orchestration.
    """Run the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7429-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        return [], {"passed": False, "plan_errors": plan_errors}
    planned = [
        PlannedCommand(spec=command, category="required_affected_validation", required=True)
        for command in commands
    ]
    progress(started, "validation", "before_affected_commands", count=len(planned))
    receipts = run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected", heartbeat_s=60.0
    )
    progress(started, "validation", "after_affected_commands", count=len(receipts))
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return receipts, {**reduced, "plan_errors": []}


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build cold replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7429_v651_anchored_capture import independent_reduce_artifact;"
        "value=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "errors=independent_reduce_artifact(value);"
        "print(json.dumps({'errors':errors},sort_keys=True),flush=True);"
        "raise SystemExit(bool(errors))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "measured_candidate",
        ),
        CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "measured_candidate",
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful receipt for every declared command."""

    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is not True
            for row in receipts
        )
        == 1
        for name in names
    )


def _source_hashes(
    root: Path, context: Mapping[str, Any], capture: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - measured source evidence.
    """Bind current sources, prerequisites, model, runner, events, and raw rows."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        frozen.MODULE_PATH,
        frozen.UPSTREAM_PATH,
        frozen.SOURCE_ARTIFACT_PATH,
        frozen.SOURCE_MANIFEST_PATH,
        frozen.CHALLENGE_PATH,
        EXP7416_PATH,
        EXP7422_PATH,
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7400_v649_assignment_canary.py"),
        Path("python/carnot/experiment_7422_v651_runtime_ownership.py"),
        Path("python/carnot/inference/sota_models.py"),
    )
    hashes: JsonDict = {}
    for relative in paths:
        path = root / relative
        if path.is_file():
            original_flag = None
            if relative in {frozen.UPSTREAM_PATH, EXP7416_PATH, EXP7422_PATH}:
                original_flag = load_object(path).get("flagged_adversarial")
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "original_flagged_adversarial": original_flag,
            }
    model_spec = dict(context.get("model_spec") or {})
    model_path = Path(str(model_spec.get("path") or ""))
    if model_path.is_file():
        hashes["cached_model"] = {
            "path": str(model_path),
            "sha256": model_spec.get("sha256"),
            "original_flagged_adversarial": None,
        }
    hashes["current_event_shards"] = deepcopy(list(capture.get("event_shards") or []))
    hashes["current_response_shards"] = deepcopy(list(capture.get("response_shards") or []))
    hashes["measured_row_shards"] = deepcopy(list(capture.get("measured_shards") or []))
    return hashes


def _runner_receipt(
    capture: Mapping[str, Any], context: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - measured runtime evidence.
    """Combine owned native identity, lease, offload, model, and cleanup facts."""

    identity = dict(capture.get("runtime_identity") or {})
    provenance = dict(capture.get("gpu_receipts") or {}).get("provenance") or {}
    model_spec = dict(context.get("model_spec") or {})
    offload = canary.build_offload_receipt(
        identity, provenance, int(model_spec.get("model_block_count", 0) or 0)
    )
    release = dict(capture.get("gpu_receipts") or {}).get("lease_release") or {}
    cleanup = dict(capture.get("gpu_receipts") or {}).get("cleanup") or {}
    return {
        **deepcopy(identity),
        **offload,
        "runner_pid": identity.get("pid"),
        "runner_pid_start_ticks": identity.get("start_time_ticks"),
        "runner_path": str(context.get("server_path") or ""),
        "runner_sha256": context.get("server_sha256"),
        "model_sha256": model_spec.get("sha256"),
        "model_revision": model_spec.get("revision"),
        "quantization": model_spec.get("quantization"),
        "runtime_flags": deepcopy(model_spec.get("runtime_flags") or {}),
        "decoding": deepcopy(model_spec.get("decoding") or {}),
        "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
        "lease_released": release.get("released") is True,
        "lease_release": deepcopy(release),
        "cleanup": deepcopy(cleanup),
    }


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live orchestration.
    """Authenticate, capture, reduce, validate, and atomically publish once."""

    started = time.monotonic()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start")
    checks, context = collect_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date",
            "execution_contract",
            "run_date",
            "==",
            RUN_DATE,
            run_date,
            principle="The fixed protocol uses one declared execution date.",
        ),
    )
    _span(spans, "preconditions_static", phase_started, started, len(checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(checks)
        artifact.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    phase_started = time.monotonic()
    progress(started, "preconditions", "before_runtime_checks")
    runtime_checks = _runtime_preconditions(root, context, started)
    checks.extend(runtime_checks)
    progress(
        started,
        "preconditions",
        "after_runtime_checks",
        passed=all(row["passed"] for row in checks),
    )
    _span(spans, "preconditions_runtime", phase_started, started, len(runtime_checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(checks)
        artifact.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
                "model_specs": [deepcopy(dict(context.get("model_spec") or {}))],
                "inference_substrate_details": deepcopy(
                    dict(context.get("capacity_receipt") or {})
                ),
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    phase_started = time.monotonic()
    scientific_started = phase_started
    progress(started, "generation", "before_model_load", planned_development=4, planned_measured=96)
    capture = _capture_current(context, raw_dir / "owned_runtime", started)
    progress(
        started,
        "generation",
        "after_model_load_and_generation",
        development_usable=capture["development_gate"]["usable_output_count"],
        measured_dispositions=len(capture["rows"]),
    )
    _span(
        spans,
        "model_load_and_generation",
        phase_started,
        started,
        len(capture["development_rows"]) + len(capture["rows"]),
        (raw_dir / "owned_runtime/capture_checkpoint.json").as_posix(),
    )
    scientific_duration = time.monotonic() - scientific_started

    phase_started = time.monotonic()
    progress(started, "reduction", "start", planned=96)
    reduced = reduce_capture(capture["rows"], context["cases"])
    current = canary.reduce_current_events(
        capture["current_invocation_events"],
        run_id=capture["current_run_id"],
        owner_pid=capture["current_owner_pid"],
    )
    substrate_class = substrate_class_from_counts(current["invocation_counts"])
    substrate_name = {
        "model_bounded_generation": INFERENCE_SUBSTRATE,
        "model_load_no_generation": "owned_native_cuda_llama_cpp_model_load_no_generation",
        "no_model_load": "no_model_load",
    }[substrate_class]
    runner = _runner_receipt(capture, context)
    progress(started, "reduction", "complete", completed=96, usable=reduced["usable_output_count"])
    _span(spans, "reduction", phase_started, started, 96)

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    affected_ok = bool(affected.get("passed"))
    development_open = capture["development_gate"]["capture_open"] is True
    completion_observed = int(
        development_open
        and len(reduced["extraction_rows"]) == 96
        and current["model_invoked"] is True
        and runner["all_layers_offloaded"] is True
        and affected_ok
    )
    gates = [
        gate_row(
            "development_capture_open",
            EXPERIMENT_ID,
            "usable_development_outputs",
            ">=",
            DEVELOPMENT_USABLE_MINIMUM,
            capture["development_gate"]["usable_output_count"],
            principle="Three usable non-truncated development outputs are required.",
            category="completion",
        ),
        gate_row(
            "all_planned_dispositions",
            EXPERIMENT_ID,
            "planned_call_count",
            "==",
            96,
            reduced["planned_call_count"],
            principle="Every measured call remains accountable.",
            category="completion",
        ),
        gate_row(
            "cuda_all_layer_offload",
            "runner_receipt",
            "all_layers_offloaded",
            "==",
            True,
            runner["all_layers_offloaded"],
            principle="CPU fallback cannot supply current model evidence.",
            category="evidence",
        ),
        gate_row(
            "affected_validation",
            "validation_receipts",
            "required_checks_passed",
            "==",
            True,
            affected_ok,
            principle="Genuine affected failures disqualify the capture.",
            category="validation",
        ),
        gate_row(
            "qualified_extraction_value",
            EXPERIMENT_ID,
            "extraction_value_score",
            "==",
            1,
            reduced["extraction_value_score"],
            principle="Scientific value requires qualified preservation under the frozen comparison.",
            category="scientific_value",
        ),
    ]
    artifact = _base_artifact()
    artifact.update(
        {
            "status": (
                "complete_capture_awaiting_terminal_readers"
                if development_open
                else "complete_development_gate_closed"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "model_specs": [deepcopy(dict(context["model_spec"]))],
            **current,
            "current_invocation_events": deepcopy(capture["current_invocation_events"]),
            "current_run_id": capture["current_run_id"],
            "current_owner_pid": capture["current_owner_pid"],
            "inference_substrate": substrate_name,
            "inference_substrate_details": {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "runner": str(context.get("server_path") or ""),
                "gpu_uuid": dict(capture.get("runtime_identity") or {}).get("gpu_uuid"),
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            "inference_substrate_class": substrate_class,
            "scientific_duration_s": round(scientific_duration, 6),
            "phase_spans": spans,
            "rows": deepcopy(reduced["extraction_rows"]),
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": reduced["attempted_call_count"],
                "completed": reduced["completed_call_count"],
                "failed": reduced["failed_call_count"],
                "censored": reduced["censored_call_count"],
                "unstarted": reduced["unstarted_call_count"],
                "development_calls_used": capture["development_gate"]["attempted"],
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary([*checks, *gates]),
            "honest_verdict": (
                "complete_null_anchored_capture_no_qualified_value_gain"
                if development_open
                else "complete_null_development_output_insufficient_capture_not_opened"
            ),
            "verdict_class": "null",
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected_ok else "required_checks_failed",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": not affected_ok,
                "affected_reduction": affected,
            },
            "extraction_capture_complete_score": completion_observed,
            "runner_receipt": runner,
            "development_rows": deepcopy(capture["development_rows"]),
            "development_gate": deepcopy(capture["development_gate"]),
            "raw_capture_manifest": frozen._raw_manifest(reduced["extraction_rows"]),
            **reduced,
        }
    )
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    artifact["source_artifact_hashes"] = _source_hashes(root, context, capture)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate, artifact)

    cold_started = time.monotonic()
    progress(started, "validation", "before_terminal_commands", candidate=candidate)
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec=spec,
                category="safety" if spec.name == "adversarial_verify" else "completion",
                required=True,
            )
            for spec in _terminal_commands(root, candidate)
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(started, "validation", "after_terminal_commands", count=len(terminal_receipts))
    cold_duration = time.monotonic() - cold_started
    terminal_ok = _receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    terminal_gate = gate_row(
        "terminal_readers",
        "validation_receipts",
        "terminal_checks_passed",
        "==",
        True,
        terminal_ok and independent_ok and not flagged,
        principle="Fresh replay, independent reduction, and both strict readers must pass.",
        category="validation",
    )
    gates.append(terminal_gate)
    _span(
        spans,
        "validation",
        validation_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    final_complete = int(completion_observed and terminal_ok and independent_ok and not flagged)
    disqualified = not affected_ok or not terminal_ok or not independent_ok or flagged
    if disqualified:
        artifact.update(
            {
                "status": "complete_required_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_anchored_capture_required_check_failed",
            }
        )
    elif development_open:
        artifact["status"] = "complete_extraction_capture"
        if artifact["extraction_value_score"] == 1:
            artifact["honest_verdict"] = "complete_circular_positive_qualified_claim_preservation"
            artifact["verdict_class"] = "circular_positive"
    artifact["flagged_adversarial"] = flagged
    artifact["extraction_capture_complete_score"] = final_complete
    artifact["validation_receipts"] = [*affected_receipts, *terminal_receipts]
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = gate_check_summary([*checks, *gates])
    artifact["phase_spans"] = spans
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["cold_start_validation_duration_s"] = round(cold_duration, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, require_terminal=True, root=root)
    if errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "extraction_capture_complete_score": 0,
                "internal_validation_errors": errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress(started, "write", "before_atomic_publish", artifact=output)
    atomic_json(output, artifact)
    progress(
        started,
        "write",
        "after_atomic_publish",
        artifact=output,
        verdict=artifact["honest_verdict"],
    )
    return artifact


def _date_argument(value: str) -> str:
    """Reject execution outside the fixed V651 date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run live capture or cold-validate one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = load_object(args.validate)
        errors = list(dict.fromkeys(independent_reduce_artifact(value, root=REPO_ROOT)))
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "extraction_capture_complete_score": result["extraction_capture_complete_score"],
                "extraction_value_score": result["extraction_value_score"],
                "usable_output_count": result["usable_output_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
