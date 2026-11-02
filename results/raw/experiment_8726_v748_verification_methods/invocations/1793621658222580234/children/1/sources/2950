"""Run one paired live test of the sealed V652 claim-span protocol.

The module treats the span and verbatim schemas as two representations of the
same extraction task. It keeps real-paragraph diagnostics mechanical. Only the
sealed constructed controls have exact qualifier authority.

Spec refs: REQ-VERIFY-7442 and SCENARIO-VERIFY-7442-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as native_runtime
from carnot import experiment_7400_v649_assignment_canary as canary
from carnot import experiment_7422_v651_runtime_ownership as ownership
from carnot import experiment_7437_v652_span_protocol as protocol
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7442-v652-span-capture"
TASK_ID = "experiment_7442_v652_span_capture"
SCHEMA = "carnot.exp7442.v652.span_capture.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

RESULT_PATH = Path("results/experiment_7442_v652_span_capture.json")
RAW_DIR = Path("results/raw/experiment_7442_v652_span_capture")
MODULE_PATH = Path("python/carnot/experiment_7442_v652_span_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7442_v652_span_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7442_v652_span_capture.py")
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
PROTOCOL_PATH = Path("results/experiment_7437_v652_span_protocol.json")
EXPECTED_PROTOCOL_SHA256 = "sha256:7130813047bef0bfc977e5d7231db79abf3f5e8a408dd510551ca3e65f40ae1d"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
INFERENCE_SUBSTRATE = "owned_native_cuda_llama_cpp_fixed_span_generation"
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 6_527_442
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 45.0
MODEL_LOAD_TIMEOUT_S = 600.0
AGGREGATE_LIVE_BUDGET_S = 1500.0
LEASE_WAIT_TIMEOUT_S = 120.0
DEVELOPMENT_PARAGRAPHS = 4
DEVELOPMENT_CALLS = 8
DEVELOPMENT_USABLE_MINIMUM = 3
SEALED_EVALUATION_UNITS = 48
EVALUATION_CALLS = 96
MAX_GENERATION_CALLS = 104
BOOTSTRAP_DRAWS = 10_000

AFFECTED_CHECK_NAMES = REQUIRED_CHECK_NAMES
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

FIELD_PRINCIPLES = {
    "schema": "Use a versioned plain top-level schema with experiment identity, milestone, and terminal status.",
    "run_date": "Use 20260920 and record actual UTC start, end, and monotonic duration.",
    "preconditions_checked": "Name each observed resource, path, identity, and prerequisite before dependent work.",
    "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for every current model call.",
    "model_invoked": "Keep actual current model attempts separate from archived or scripted evidence.",
    "invocation_counts": "Reconcile current loads and generations across every terminal disposition.",
    "inference_substrate": "Use a truthful string and keep device facts in inference_substrate_details.",
    "inference_substrate_class": "Declare model_bounded_generation for the fixed 256-token extraction calls.",
    "execution_venue": "Use host and record CPU, CUDA, and external-device identity separately.",
    "duration_s": "Measure current work and separate model, computation, cold-start, and validation time.",
    "phase_spans": "Bind phase times, progress boundaries, completed units, and checkpoints.",
    "random_seed": "Freeze arm order and paired bootstrap resampling before outcomes are read.",
    "reproducibility_checksum": "Bind code, protocol, input bytes, raw rows, and exact validation scope.",
    "source_artifact_hashes": "Preserve source identity, original classes, and flags through typed byte receipts.",
    "rows": "Keep every evaluation unit, arm, seed, condition, failure, and unstarted disposition.",
    "sample_size_budget": "Separate planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep validity, completion, evidence, safety, and benefit checks separate.",
    "gate_check_summary": "Name exact upstream, path, check, field, operands, and observed failure.",
    "verifier_is_oracle": "Use true only for the sealed constructed exact-string qualifier controls.",
    "honest_verdict": "Use complete_ for finished work and blocked_ only for unchanged external absence.",
    "verdict_class": "Use positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical findings because flagged science cannot supply readiness.",
    "validation_receipts": "Retain exact scoped commands, environments, exits, durations, and hashed logs.",
    "field_principles": "Explain field intent separately while gate values remain ordinary scalars.",
    "promotion_score": "Keep zero because this milestone changes no rollout, publication, defaults, or weights.",
    "span_capture_complete_score": "Set one only for a valid fully accounted 96-call evaluation panel.",
    "span_value_score": "Require positive paired completion CI, no constructed qualifier loss, and lower span token cost.",
    "development_rows": "Retain all eight canary calls even when the evaluation panel stays unopened.",
    "extraction_rows": "Keep literal spans, parser outcome, token cost, and disposition for all 96 evaluation calls.",
    "semantic_pair_rows": "Keep constructed qualifier authority separate from real-paragraph unknown semantics.",
    "raw_capture_manifest": "Hash current request and response bytes before parser interpretation.",
    "runner_receipt": "Bind model file, native runtime, actual CUDA offload, owned PID, and lease identity.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


canonical_hash = protocol.canonical_hash
sha256_file = protocol.sha256_file
atomic_json = protocol.atomic_json


def utc_now() -> str:  # pragma: no cover - current wall-clock evidence.
    """Return one actual UTC boundary while elapsed time uses a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase, long operation, model boundary, and checkpoint."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7442] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Hash the complete artifact without recursively hashing its checksum."""

    copied = deepcopy(dict(value))
    copied.pop("reproducibility_checksum", None)
    return canonical_hash(copied)


def _load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str = EXPERIMENT_ID,
    path: str = RESULT_PATH.as_posix(),
    field: str | None = None,
) -> JsonDict:
    """Keep one gate's exact operands and decision rule visible."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
        "upstream": upstream,
        "path": path,
        "field": field or check,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failure while retaining the full gate table."""

    failed = [dict(row) for row in gates if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row.get("check")) for row in failed],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("path") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "field": first.get("field") if first else "gate_check_summary",
        "operator": first.get("operator") if first else "==",
        "expected": deepcopy(first.get("expected")) if first else True,
        "observed": deepcopy(first.get("observed")) if first else True,
        "passed": not failed,
    }


def protocol_gate_rows(value: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate the four Exp7437 operands that permit model work."""

    fields = (
        ("protocol_identity", "experiment_id", "==", "exp7437-v652-span-protocol"),
        ("span_protocol_ready", "span_protocol_ready_score", "==", 1),
        (
            "span_protocol_verdict_eligible",
            "verdict_class",
            "in",
            ["null", "positive", "circular_positive"],
        ),
        ("span_protocol_unflagged", "flagged_adversarial", "==", False),
    )
    rows: list[JsonDict] = []
    for check, field, operator, expected in fields:
        observed = value.get(field)
        passed = observed == expected if operator == "==" else observed in expected
        rows.append(
            _gate(
                check,
                "precondition",
                operator,
                expected,
                observed,
                passed,
                "Only the exact ready, eligible, and unflagged span protocol can authorize capture.",
                upstream=PROTOCOL_PATH.as_posix(),
                path=PROTOCOL_PATH.as_posix(),
                field=field,
            )
        )
    return rows


def _prompt(arm: str, paragraph: str) -> str:
    """Use the exact frozen Exp7437 prompt renderer for each representation."""

    return protocol._prompt(arm, paragraph)


def _schedule_row(
    source: Mapping[str, Any], *, arm: str, arm_order: int, pair_id: str, call_id: str
) -> JsonDict:
    """Project one sealed paragraph into one fixed-budget current call."""

    paragraph = str(source["paragraph"])
    prompt = _prompt(arm, paragraph)
    return {
        "call_id": call_id,
        "request_id": call_id,
        "pair_id": pair_id,
        "unit_id": str(source.get("unit_id") or pair_id),
        "case_index": int(source.get("case_index", 0) or 0),
        "group_id": str(source.get("group_id") or ""),
        "response_id": str(source.get("response_id") or ""),
        "condition": str(source.get("condition") or "ragtruth_unchanged_response"),
        "capture_phase": str(source.get("capture_phase") or "evaluation"),
        "arm": arm,
        "arm_order": arm_order,
        "seed": RANDOM_SEED,
        "paragraph": paragraph,
        "paragraph_sha256": str(source["paragraph_sha256"]),
        "clipped": bool(source.get("clipped")),
        "complete_response_coverage_eligible": bool(
            source.get("complete_response_coverage_eligible")
        ),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "generation_count": 1,
        "grammar_mask": False,
        "parser_retry_count": 0,
        "prompt": prompt,
        "prompt_sha256": canonical_hash(prompt),
    }


def build_development_schedule(development: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Freeze four development paragraphs through both arms for eight calls."""

    if len(development) != DEVELOPMENT_PARAGRAPHS:
        raise ValueError("development_paragraph_count")
    rows: list[JsonDict] = []
    for index, source_value in enumerate(development):
        source = dict(source_value)
        source.update(
            {
                "case_index": index,
                "condition": "sealed_development_paragraph",
                "capture_phase": "development",
            }
        )
        first = "span" if index % 2 == 0 else "verbatim"
        arms = (first, "verbatim" if first == "span" else "span")
        pair_id = f"development-{index:02d}"
        for arm_order, arm in enumerate(arms):
            rows.append(
                _schedule_row(
                    source,
                    arm=arm,
                    arm_order=arm_order,
                    pair_id=pair_id,
                    call_id=f"{pair_id}-{arm}",
                )
            )
    return rows


def build_evaluation_schedule(sealed: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Run every sealed unit through both arms while preserving its first-arm order."""

    if len(sealed) != SEALED_EVALUATION_UNITS:
        raise ValueError("sealed_schedule_count")
    rows: list[JsonDict] = []
    for index, source_value in enumerate(sealed):
        source = dict(source_value)
        source["capture_phase"] = "evaluation"
        first = str(source.get("arm") or "")
        if first not in protocol.ARMS:
            raise ValueError("sealed_schedule_arm")
        arms = (first, "verbatim" if first == "span" else "span")
        pair_id = f"paired-{index:02d}-{source.get('unit_id')}"
        for arm_order, arm in enumerate(arms):
            rows.append(
                _schedule_row(
                    source,
                    arm=arm,
                    arm_order=arm_order,
                    pair_id=pair_id,
                    call_id=f"evaluation-{index:02d}-{arm}",
                )
            )
    identity = canonical_hash(rows)
    for row in rows:
        row["evaluation_schedule_sha256"] = identity
    return rows


def _transport_hash(value: Any) -> str:
    """Hash the exact canonical transport bytes used by the local server client."""

    encoded = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _nonspace_coverage(paragraph: str, spans: Sequence[Sequence[int]]) -> float:
    """Measure literal character coverage without treating it as semantic truth."""

    eligible = {index for index, char in enumerate(paragraph) if not char.isspace()}
    covered = {
        index for start, end in spans for index in range(int(start), int(end)) if index in eligible
    }
    return len(covered) / len(eligible) if eligible else 1.0


def build_capture_row(schedule: Mapping[str, Any], response: Mapping[str, Any]) -> JsonDict:
    """Bind immutable transport evidence, then parse one terminal disposition."""

    row = deepcopy(dict(schedule))
    raw_request = deepcopy(response.get("raw_request") or {})
    raw_response = deepcopy(response.get("raw_response") or {})
    raw_reply = str(response.get("raw_reply") or "")
    attempted = response.get("attempted") is True
    terminal_state = str(
        response.get("terminal_state") or ("unstarted" if not attempted else "failed")
    )
    finish_reason = response.get("finish_reason")
    raw = {
        "raw_request": raw_request,
        "raw_response": raw_response,
        "raw_reply": raw_reply,
        "raw_request_sha256": _transport_hash(raw_request),
        "raw_response_sha256": _transport_hash(raw_response),
        "raw_reply_sha256": canonical_hash(raw_reply),
        "persisted_before_parse": True,
        "attempted": attempted,
        "terminal_state": terminal_state,
        "finish_reason": finish_reason,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "output_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "error": response.get("error"),
        "censored": bool(response.get("censored")),
        "runtime_identity_receipt": deepcopy(response.get("runtime_identity_receipt") or {}),
    }
    row.update(raw)
    if not attempted:
        parsed = protocol._base_parse("unstarted")
        disposition = "unstarted"
    elif terminal_state != "response":
        parsed = protocol._base_parse("transport_failure")
        disposition = "cancelled" if terminal_state == "cancelled" else "failed"
    elif finish_reason in {"length", "max_tokens"}:
        parsed = protocol._base_parse("truncated_output")
        disposition = "truncated"
    else:
        parsed = protocol.parse_claim_output(
            raw_reply,
            arm=str(row["arm"]),
            paragraph=str(row["paragraph"]),
        )
        disposition = "completed" if parsed["parse_valid"] else "malformed"
    spans = parsed.get("claim_spans") or []
    claims = parsed.get("claims") or []
    whole = bool(
        parsed.get("parse_valid")
        and claims
        and all(str(claim).rstrip().endswith((".", "?", "!", ";")) for claim in claims)
    )
    row.update(parsed)
    row.update(
        {
            # The shared native loop reports these generic parser fields after
            # persisting each response.  Keep them beside the span-specific
            # parser result so a valid raw response cannot become a runtime
            # failure at the adapter boundary.
            "parse_status": "valid" if parsed.get("parse_valid") is True else "invalid",
            "parse_errors": []
            if parsed.get("parse_valid") is True
            else [str(parsed.get("disposition") or disposition)],
            "disposition": disposition,
            "completed_valid_output": bool(
                disposition == "completed"
                and parsed.get("parse_valid")
                and claims
                and parsed.get("literal_span_reconstruction") is True
            ),
            "nonempty_output": bool(raw_reply.strip()),
            "whole_proposition_coverage": whole,
            "covered_nonspace_fraction": _nonspace_coverage(str(row["paragraph"]), spans),
            "qualifier_retention": None,
            "semantic_truth": "unknown"
            if row.get("condition") == "ragtruth_unchanged_response"
            else "not_assigned",
            "retry_count": 0,
            "repair_attempted": False,
        }
    )
    return row


def reconcile_terminal_events(
    events: Sequence[Mapping[str, Any]],
    terminal_rows: Sequence[Mapping[str, Any]],
    *,
    monotonic_ns: int,
) -> list[JsonDict]:
    """Close only unfinished generations backed by persisted terminal rows.

    This is an accounting repair, not another invocation.  The appended event
    retains the source call identifier so an auditor can trace why the already
    persisted native response makes the generation terminal.
    """

    copied = [deepcopy(dict(row)) for row in events]
    attempt_times = {
        row.get("call_id"): int(row.get("monotonic_ns", 0) or 0)
        for row in copied
        if row.get("operation") == "generation" and row.get("state") == "attempted"
    }
    # A recovery process can run after a host reboot. Its monotonic clock then
    # starts below the persisted attempt time. Keep that bad derived shard as
    # raw evidence, but replace it in the reduced ledger with a valid ordering.
    reconciled = [
        row
        for row in copied
        if not (
            row.get("reconciliation_basis") == "persisted_terminal_raw_response"
            and int(row.get("monotonic_ns", 0) or 0) <= attempt_times.get(row.get("call_id"), -1)
        )
    ]
    attempted = [
        row
        for row in reconciled
        if row.get("operation") == "generation" and row.get("state") == "attempted"
    ]
    terminal_ids = {
        row.get("call_id")
        for row in reconciled
        if row.get("operation") == "generation" and row.get("state") != "attempted"
    }
    available = [
        dict(row)
        for row in terminal_rows
        if row.get("terminal_state") in {"response", "request_error", "cancelled"}
    ]
    next_monotonic_ns = max(
        monotonic_ns,
        max((int(row.get("monotonic_ns", 0) or 0) for row in reconciled), default=0) + 1,
    )
    for index, attempt in enumerate(attempted):
        if attempt.get("call_id") in terminal_ids or index >= len(available):
            continue
        source = available[index]
        state = "completed" if source.get("terminal_state") == "response" else "failed"
        reconciled.append(
            {
                **{
                    key: attempt[key]
                    for key in ("scope", "transport", "run_id", "owner_pid", "call_id")
                },
                "operation": "generation",
                "state": state,
                "monotonic_ns": next_monotonic_ns + index,
                "reconciled_from_call_id": source.get("call_id"),
                "reconciliation_basis": "persisted_terminal_raw_response",
            }
        )
    return reconciled


def unstarted_evaluation_rows(schedule: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Materialize every unopened evaluation call instead of dropping it."""

    return [
        build_capture_row(
            row,
            {
                "raw_request": {},
                "raw_response": {},
                "raw_reply": "",
                "attempted": False,
                "terminal_state": "unstarted",
            },
        )
        for row in schedule
    ]


def capture_row_errors(row: Mapping[str, Any]) -> list[str]:
    """Recompute exact request, response, reply, and paragraph byte identities."""

    errors: list[str] = []
    expected = {
        "raw_request_sha256": _transport_hash(row.get("raw_request") or {}),
        "raw_response_sha256": _transport_hash(row.get("raw_response") or {}),
        "raw_reply_sha256": canonical_hash(str(row.get("raw_reply") or "")),
        "paragraph_sha256": "sha256:"
        + hashlib.sha256(str(row.get("paragraph") or "").encode("utf-8")).hexdigest(),
    }
    labels = {
        "raw_request_sha256": "raw_request_hash_mismatch",
        "raw_response_sha256": "raw_response_hash_mismatch",
        "raw_reply_sha256": "raw_reply_hash_mismatch",
        "paragraph_sha256": "paragraph_hash_mismatch",
    }
    for field, observed in expected.items():
        if row.get(field) != observed:
            errors.append(labels[field])
    if row.get("persisted_before_parse") is not True:
        errors.append("raw_not_persisted_before_parse")
    return errors


def _usable(row: Mapping[str, Any]) -> bool:
    """Require complete, nonempty, parse-valid, exact reconstruction."""

    return bool(
        row.get("terminal_state") == "response"
        and row.get("finish_reason") not in {"length", "max_tokens"}
        and row.get("parse_valid") is True
        and row.get("completed_valid_output") is True
        and row.get("literal_span_reconstruction") is True
    )


def reduce_development_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Open evaluation only when each arm supplies three usable canary calls."""

    values = [dict(row) for row in rows]
    by_arm = {
        arm: sum(_usable(row) for row in values if row.get("arm") == arm) for arm in protocol.ARMS
    }
    shape_ok = len(values) == DEVELOPMENT_CALLS and all(
        sum(row.get("arm") == arm for row in values) == DEVELOPMENT_PARAGRAPHS
        for arm in protocol.ARMS
    )
    opened = shape_ok and all(by_arm[arm] >= DEVELOPMENT_USABLE_MINIMUM for arm in protocol.ARMS)
    return {
        "planned": DEVELOPMENT_CALLS,
        "attempted": sum(row.get("attempted") is True for row in values),
        "completed": sum(row.get("terminal_state") == "response" for row in values),
        "usable_by_arm": by_arm,
        "required_usable_per_arm": DEVELOPMENT_USABLE_MINIMUM,
        "capture_open": opened,
        "terminal_class": "ready" if opened else "null",
    }


def _metric(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    """Reduce one Boolean endpoint while retaining unknown and assigned calls."""

    values = [row.get(field) for row in rows]
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


def _paired_bootstrap(differences: Sequence[float]) -> JsonDict:
    """Compute the frozen paired percentile interval with one declared seed."""

    if not differences:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "pairs": 0}
    estimate = sum(differences) / len(differences)
    rng = random.Random(RANDOM_SEED)
    draws = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "estimate": estimate,
        "ci95_low": draws[int(0.025 * (BOOTSTRAP_DRAWS - 1))],
        "ci95_high": draws[int(0.975 * (BOOTSTRAP_DRAWS - 1))],
        "pairs": len(differences),
        "draws": BOOTSTRAP_DRAWS,
        "seed": RANDOM_SEED,
    }


def _semantic_pairs(controls: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Reduce only the twelve sealed constructed qualifier pairs as exact checks."""

    exact_controls = [row for row in controls if row.get("scope") == "constructed_exact_check"]
    if not exact_controls:
        exact_controls = protocol.reduce_constructed_pairs(protocol.constructed_qualifier_pairs())
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in exact_controls:
        if row.get("scope") == "constructed_exact_check":
            groups[str(row.get("pair_id"))].append(row)
    rows: list[JsonDict] = []
    for pair_id in sorted(groups):
        arm_rows = {str(row.get("arm")): row for row in groups[pair_id]}
        if set(arm_rows) != set(protocol.ARMS):
            continue
        span = arm_rows["span"].get("qualifier_retention") is True
        verbatim = arm_rows["verbatim"].get("qualifier_retention") is True
        rows.append(
            {
                "pair_id": pair_id,
                "family": arm_rows["span"].get("family"),
                "authority": "constructed_exact_string",
                "span_qualifier_retained": span,
                "verbatim_qualifier_retained": verbatim,
                "paired_delta": int(span) - int(verbatim),
            }
        )
    return rows


def _reparse(row: Mapping[str, Any]) -> JsonDict:
    """Rebuild parser fields from immutable transport evidence and schedule fields."""

    schedule_fields = {
        key: deepcopy(value)
        for key, value in row.items()
        if key
        not in {
            "disposition",
            "syntax_valid",
            "parse_valid",
            "claims",
            "claim_spans",
            "literal_span_reconstruction",
            "qualifier_retention",
            "missing_modifiers",
            "retry_count",
            "repair_attempted",
            "completed_valid_output",
            "nonempty_output",
            "whole_proposition_coverage",
            "covered_nonspace_fraction",
            "semantic_truth",
            "output_tokens",
        }
    }
    response = {
        "raw_request": deepcopy(row.get("raw_request") or {}),
        "raw_response": deepcopy(row.get("raw_response") or {}),
        "raw_reply": str(row.get("raw_reply") or ""),
        "attempted": row.get("attempted") is True,
        "terminal_state": row.get("terminal_state"),
        "finish_reason": row.get("finish_reason"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "latency_s": row.get("latency_s"),
        "error": row.get("error"),
        "censored": row.get("censored"),
        "runtime_identity_receipt": deepcopy(row.get("runtime_identity_receipt") or {}),
    }
    return build_capture_row(schedule_fields, response)


def reduce_evaluation(
    rows: Sequence[Mapping[str, Any]], controls: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Reparse all 96 calls and compute the frozen paired representation gates."""

    if len(rows) != EVALUATION_CALLS:
        raise ValueError("evaluation_row_count")
    parsed = [_reparse(row) for row in rows]
    pairs: dict[str, list[JsonDict]] = defaultdict(list)
    for row in parsed:
        pairs[str(row.get("pair_id"))].append(row)
    if len(pairs) != SEALED_EVALUATION_UNITS or any(
        len(pair) != 2 or {row.get("arm") for row in pair} != set(protocol.ARMS)
        for pair in pairs.values()
    ):
        raise ValueError("evaluation_pair_shape")
    differences: list[float] = []
    token_differences: list[float] = []
    for pair in pairs.values():
        arms = {str(row["arm"]): row for row in pair}
        differences.append(
            float(bool(arms["span"]["completed_valid_output"]))
            - float(bool(arms["verbatim"]["completed_valid_output"]))
        )
        if arms["span"]["attempted"] and arms["verbatim"]["attempted"]:
            token_differences.append(
                float(arms["span"]["completion_tokens"])
                - float(arms["verbatim"]["completion_tokens"])
            )
    completion = _paired_bootstrap(differences)
    output_delta = sum(token_differences) / len(token_differences) if token_differences else None
    semantic = _semantic_pairs(controls)
    qualifier_delta = (
        sum(float(row["paired_delta"]) for row in semantic) / len(semantic) if semantic else None
    )
    value = int(
        completion["ci95_low"] is not None
        and float(completion["ci95_low"]) > 0.0
        and qualifier_delta is not None
        and qualifier_delta >= 0.0
        and output_delta is not None
        and output_delta < 0.0
    )
    arm_metrics = {
        arm: {
            "completed_valid_output": _metric(
                [row for row in parsed if row["arm"] == arm], "completed_valid_output"
            ),
            "literal_span_reconstruction": _metric(
                [row for row in parsed if row["arm"] == arm],
                "literal_span_reconstruction",
            ),
            "whole_proposition_coverage": _metric(
                [row for row in parsed if row["arm"] == arm],
                "whole_proposition_coverage",
            ),
        }
        for arm in protocol.ARMS
    }
    counts = {
        "planned": EVALUATION_CALLS,
        "attempted": sum(row["attempted"] is True for row in parsed),
        "completed": sum(row["terminal_state"] == "response" for row in parsed),
        "failed": sum(row["disposition"] in {"failed", "malformed", "truncated"} for row in parsed),
        "censored": sum(bool(row["censored"]) for row in parsed),
        "unstarted": sum(row["attempted"] is not True for row in parsed),
    }
    return {
        "extraction_rows": parsed,
        "sample_counts": counts,
        "arm_metrics": arm_metrics,
        "paired_completion_advantage": completion,
        "paired_output_token_delta": output_delta,
        "constructed_qualifier_delta": qualifier_delta,
        "semantic_pair_rows": semantic,
        "span_value_score": value,
        "real_paragraph_semantic_scope": "unknown_mechanical_omission_and_retention_only",
    }


def _base_artifact() -> JsonDict:
    """Return one complete plain-field shape for every terminal disposition."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": "blocked_unstarted",
        "run_date": RUN_DATE,
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "current_run_id": None,
        "current_owner_pid": None,
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_substrate": "no_model_load",
        "inference_substrate_details": {},
        "inference_substrate_class": "no_model_load",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "model_duration_s": 0.0,
        "computation_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "cold_start_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {
            "arm_order": protocol.RANDOM_SEED,
            "sampling": RANDOM_SEED,
            "resampling": RANDOM_SEED,
            "fitting": None,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": EVALUATION_CALLS,
            "attempted": 0,
            "completed": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": EVALUATION_CALLS,
            "independent_paired_units": SEALED_EVALUATION_UNITS,
            "development_planned": DEVELOPMENT_CALLS,
            "development_attempted": 0,
            "maximum_generation_calls": MAX_GENERATION_CALLS,
            "maximum_new_tokens_per_call": MAX_NEW_TOKENS,
            "per_call_ceiling_s": REQUEST_TIMEOUT_S,
            "aggregate_live_budget_s": AGGREGATE_LIVE_BUDGET_S,
            "lease_wait_timeout_s": LEASE_WAIT_TIMEOUT_S,
            "stop_rule": "eight fixed canary calls; open only with three usable per arm; then account for all 96 evaluation calls without retry",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_unstarted",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "span_capture_complete_score": 0,
        "span_value_score": 0,
        "development_rows": [],
        "development_gate": {},
        "extraction_rows": [],
        "semantic_pair_rows": [],
        "raw_capture_manifest": [],
        "runner_receipt": None,
        "producer_runtime_error": None,
        "arm_metrics": {},
        "paired_completion_advantage": {},
        "paired_output_token_delta": None,
        "constructed_qualifier_delta": None,
        "real_paragraph_semantic_scope": "unknown_mechanical_omission_and_retention_only",
        "protocol_receipt": {},
        "validation_manifest": {
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "affected_checks": list(AFFECTED_CHECK_NAMES),
            "terminal_checks": list(TERMINAL_CHECK_NAMES),
        },
    }


def build_blocked_artifact(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Publish exact external absence with zero dependent model work."""

    gates = [deepcopy(dict(row)) for row in checks]
    summary = _gate_summary(gates)
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "blocked_precondition",
            "started_at_utc": "2026-09-20T00:00:00Z",
            "completed_at_utc": "2026-09-20T00:00:00Z",
            "preconditions_checked": gates,
            "acceptance_gate_results": gates,
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary.get('check') or 'precondition'}",
            "verdict_class": "blocked",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _raw_manifest(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project each raw call into one stable transport byte receipt."""

    return [
        {
            "call_id": row.get("call_id"),
            "pair_id": row.get("pair_id"),
            "arm": row.get("arm"),
            "disposition": row.get("disposition"),
            "raw_request_sha256": row.get("raw_request_sha256"),
            "raw_response_sha256": row.get("raw_response_sha256"),
            "raw_reply_sha256": row.get("raw_reply_sha256"),
            "persisted_before_parse": row.get("persisted_before_parse"),
        }
        for row in rows
    ]


def required_receipt_errors(receipts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Require one passing terminal receipt for every affected and cold check."""

    errors: list[str] = []
    for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES):
        matches = [row for row in receipts if row.get("name") == name]
        if not matches:
            errors.append(f"required_validation_receipt_missing:{name}")
        elif len(matches) > 1:
            errors.append(f"required_validation_receipt_duplicate:{name}")
        elif not (
            matches[0].get("passed") is True
            and matches[0].get("exit_code") == 0
            and matches[0].get("timed_out") is not True
        ):
            errors.append(f"required_validation_receipt_failed:{name}")
    return errors


def _fixture_response(row: Mapping[str, Any], *, tokens: int) -> JsonDict:
    """Create one deterministic transport row for cold artifact tests."""

    paragraph = str(row["paragraph"])
    end = paragraph.find(".") + 1
    if end <= 0:
        end = len(paragraph)
    reply = (
        json.dumps({"claims": [[0, end]]})
        if row["arm"] == "span"
        else json.dumps({"claims": [paragraph[:end]]})
    )
    request = {
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_NEW_TOKENS,
    }
    response = {
        "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": tokens},
    }
    return {
        "raw_request": request,
        "raw_response": response,
        "raw_reply": reply,
        "attempted": True,
        "terminal_state": "response",
        "finish_reason": "stop",
        "prompt_tokens": 20,
        "completion_tokens": tokens,
        "latency_s": 0.1,
        "runtime_identity_receipt": {"pid": 1234, "start_time_ticks": 5678},
    }


def _fixture_schedules() -> tuple[list[JsonDict], list[JsonDict]]:
    """Read the sealed producer sidecars for deterministic current fixtures."""

    value = _load_object(REPO_ROOT / PROTOCOL_PATH)
    manifest = dict(value.get("span_protocol_manifest") or {})
    panel = _load_object(REPO_ROOT / str(dict(manifest.get("panel") or {}).get("path") or ""))
    sealed = _load_object(REPO_ROOT / str(dict(manifest.get("schedule") or {}).get("path") or ""))
    return (
        build_development_schedule(panel.get("development") or []),
        build_evaluation_schedule(sealed.get("rows") or []),
    )


def build_fixture_artifact() -> JsonDict:
    """Build a complete replayable artifact without loading the model."""

    development_schedule, evaluation_schedule = _fixture_schedules()
    development_rows = [
        build_capture_row(row, _fixture_response(row, tokens=6)) for row in development_schedule
    ]
    evaluation_rows = [
        build_capture_row(row, _fixture_response(row, tokens=6 if row["arm"] == "span" else 8))
        for row in evaluation_schedule
    ]
    reduced = reduce_evaluation(evaluation_rows, protocol.parser_control_rows())
    events: list[JsonDict] = []
    run_id = f"{EXPERIMENT_ID}:fixture"
    owner_pid = 1234

    def event(call_id: str, operation: str, state: str, tick: int) -> JsonDict:
        return {
            "scope": "current",
            "transport": "owned_runtime",
            "run_id": run_id,
            "owner_pid": owner_pid,
            "call_id": call_id,
            "operation": operation,
            "state": state,
            "monotonic_ns": tick,
        }

    events.extend(
        [
            event("model-load", "model_load", "attempted", 1),
            event("model-load", "model_load", "completed", 2),
        ]
    )
    for index in range(MAX_GENERATION_CALLS):
        events.extend(
            [
                event(f"generation-{index}", "generation", "attempted", 3 + index * 2),
                event(f"generation-{index}", "generation", "completed", 4 + index * 2),
            ]
        )
    receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.0,
        }
        for name in (*AFFECTED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    ]
    model_spec = {
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "path": "/external/cache/Qwen3.8-27B-Q4_K_M.gguf",
        "revision": "fixture-revision",
        "bytes": 1,
        "sha256": "sha256:" + "1" * 64,
        "native_tokenizer": "embedded_gguf",
        "native_chat_template": True,
        "chat_template_sha256": "sha256:" + "2" * 64,
        "model_block_count": 65,
        "runtime_flags": {
            "n_gpu_layers": "all",
            "split_mode": "none",
            "parallel": 1,
            "fit": "off",
            "offline": True,
            "jinja": True,
            "reasoning": "off",
        },
        "decoding": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED,
            "retry_budget": 0,
        },
    }
    protocol_value = _load_object(REPO_ROOT / PROTOCOL_PATH)
    runner = {
        "owned_by_task": True,
        "runner_pid": 1234,
        "runner_pid_start_ticks": 5678,
        "server_build": "fixture-b9606",
        "runner_sha256": "sha256:" + "3" * 64,
        "model_sha256": model_spec["sha256"],
        "model_revision": model_spec["revision"],
        "gpu_uuid": "GPU-fixture",
        "requested_gpu_layers": "all",
        "actual_offloaded_layers": 65,
        "total_model_layers": 65,
        "all_layers_offloaded": True,
        "lease_id": "fixture-lease",
        "lease_released": True,
        "request_sha256": canonical_hash(
            [row["raw_request_sha256"] for row in [*development_rows, *evaluation_rows]]
        ),
        "response_sha256": canonical_hash(
            [row["raw_response_sha256"] for row in [*development_rows, *evaluation_rows]]
        ),
    }
    context = {
        "model_spec": model_spec,
        "protocol": protocol_value,
        "protocol_manifest": protocol_value.get("span_protocol_manifest") or {},
        "evaluation_schedule": evaluation_schedule,
        "source_hashes": {
            PROTOCOL_PATH.as_posix(): {
                "path": PROTOCOL_PATH.as_posix(),
                "sha256": EXPECTED_PROTOCOL_SHA256,
                "original_verdict_class": "null",
                "original_flagged_adversarial": False,
            }
        },
    }
    capture = {
        "development_rows": development_rows,
        "development_gate": reduce_development_gate(development_rows),
        "rows": evaluation_rows,
        "current_invocation_events": events,
        "current_run_id": run_id,
        "current_owner_pid": owner_pid,
        "event_shards": [],
        "response_shards": [],
        "evaluation_shards": [],
    }
    return _measured_artifact(
        context=context,
        checks=protocol_gate_rows(protocol_value),
        capture=capture,
        reduced=reduced,
        runner=runner,
        receipts=receipts,
        affected_ok=True,
        phase_spans=[
            {
                "phase": "model_load_and_generation",
                "start_s": 0.0,
                "end_s": 12.0,
                "duration_s": 12.0,
                "completed_units": MAX_GENERATION_CALLS,
                "checkpoint": "fixture",
            }
        ],
        started_at="2026-09-20T00:00:00Z",
        duration_s=20.0,
        model_duration_s=12.0,
        computation_duration_s=1.0,
        validation_duration_s=7.0,
        cold_start_duration_s=1.0,
        require_terminal=True,
        flagged_adversarial=False,
    )


def _model_spec_errors(value: Mapping[str, Any]) -> list[str]:
    """Require the resolved Q4_K_M model, embedded metadata, and fixed decoding."""

    if value.get("model_invoked") is not True:
        return []
    specs = value.get("model_specs") or []
    if not isinstance(specs, list) or len(specs) != 1 or not isinstance(specs[0], Mapping):
        return ["resolved_model_spec_missing"]
    spec = specs[0]
    errors: list[str] = []
    expected = {
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "native_tokenizer": "embedded_gguf",
        "native_chat_template": True,
    }
    for field, wanted in expected.items():
        if spec.get(field) != wanted:
            errors.append(f"resolved_model_spec_mismatch:{field}")
    for field in ("path", "revision", "sha256", "chat_template_sha256", "runtime_flags"):
        if not spec.get(field):
            errors.append(f"resolved_model_spec_missing:{field}")
    decoding = spec.get("decoding") or {}
    if decoding.get("max_new_tokens") != MAX_NEW_TOKENS or decoding.get("retry_budget") != 0:
        errors.append("resolved_model_decoding_mismatch")
    return errors


def validate_artifact(
    value: Mapping[str, Any], *, require_terminal: bool, root: Path = REPO_ROOT
) -> list[str]:
    """Cold-check identity, raw rows, current provenance, gates, and receipts."""

    del root
    errors = [f"missing_field:{field}" for field in REQUIRED_FIELDS if field not in value]
    expected = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "MODEL_SPECS": MODEL_SPECS,
        "execution_venue": EXECUTION_VENUE,
        "promotion_score": 0,
        "verifier_is_oracle": True,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            errors.append(f"declaration_mismatch:{field}")
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
    attempted = int(counts.get("model_loads_attempted", 0) or 0) + int(
        counts.get("generation_calls_attempted", 0) or 0
    )
    if value.get("model_invoked") is not (attempted > 0):
        errors.append("model_invoked_mismatch")
    if value.get("verdict_class") == "blocked":
        if attempted or value.get("rows") or value.get("span_capture_complete_score") != 0:
            errors.append("blocked_model_or_rows_invalid")
        if not str(value.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("blocked_verdict_prefix_invalid")
        if value.get("reproducibility_checksum") != artifact_checksum(value):
            errors.append("reproducibility_checksum_mismatch")
        return list(dict.fromkeys(errors))

    errors.extend(_model_spec_errors(value))
    events = value.get("current_invocation_events") or []
    try:
        current = canary.reduce_current_events(
            events,
            run_id=str(value.get("current_run_id")),
            owner_pid=int(value.get("current_owner_pid")),
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"current_event_reduction_failed:{type(exc).__name__}:{exc}")
    else:
        for field in ("model_invoked", "invocation_counts", "event_count", "event_sha256"):
            if value.get(field) != current.get(field):
                errors.append(f"current_receipt_mismatch:{field}")
    development = value.get("development_rows") or []
    if len(development) != DEVELOPMENT_CALLS:
        errors.append("development_row_count_mismatch")
    elif value.get("development_gate") != reduce_development_gate(development):
        errors.append("development_gate_mismatch")
    rows = value.get("extraction_rows") or []
    if value.get("rows") != rows:
        errors.append("rows_alias_mismatch")
    if not isinstance(rows, list) or len(rows) != EVALUATION_CALLS:
        errors.append("extraction_row_count_mismatch")
        reduced = None
    else:
        row_errors = [
            f"row_{index}:{error}"
            for index, row in enumerate(rows)
            for error in capture_row_errors(row)
        ]
        errors.extend(row_errors)
        try:
            reduced = reduce_evaluation(rows, protocol.parser_control_rows())
        except (TypeError, ValueError) as exc:
            errors.append(f"independent_reduction_failed:{type(exc).__name__}:{exc}")
            reduced = None
    if reduced is not None:
        comparisons = (
            "extraction_rows",
            "arm_metrics",
            "paired_completion_advantage",
            "paired_output_token_delta",
            "constructed_qualifier_delta",
            "semantic_pair_rows",
            "span_value_score",
            "real_paragraph_semantic_scope",
        )
        for field in comparisons:
            if value.get(field) != reduced.get(field):
                errors.append(f"{field}_mismatch")
        budget = value.get("sample_size_budget") or {}
        if any(
            budget.get(field) != observed for field, observed in reduced["sample_counts"].items()
        ):
            errors.append("sample_size_budget_mismatch")
        if value.get("raw_capture_manifest") != _raw_manifest(reduced["extraction_rows"]):
            errors.append("raw_capture_manifest_mismatch")
    receipts = value.get("validation_receipts") or []
    receipt_errors = required_receipt_errors(receipts) if require_terminal else []
    receipt_failure_is_terminal = (
        value.get("verdict_class") == "disqualified"
        and value.get("span_capture_complete_score") == 0
    )
    errors.extend(
        error
        for error in receipt_errors
        if not (
            error.startswith("required_validation_receipt_failed:") and receipt_failure_is_terminal
        )
    )
    runner = value.get("runner_receipt") or {}
    terminal_dispositions = len(rows) == EVALUATION_CALLS and all(
        isinstance(row, Mapping) and row.get("disposition") != "unstarted" for row in rows
    )
    expected_complete = int(
        require_terminal
        and not receipt_errors
        and terminal_dispositions
        and runner.get("all_layers_offloaded") is True
        and runner.get("lease_released") is True
        and value.get("flagged_adversarial") is False
    )
    if value.get("span_capture_complete_score") != expected_complete:
        errors.append("span_capture_complete_score_mismatch")
    if reduced is not None and value.get("span_value_score") != reduced["span_value_score"]:
        errors.append("span_value_score_mismatch")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def independent_reduce_artifact(
    value: Mapping[str, Any], *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Recompute every parser, paired endpoint, counter, and terminal gate."""

    return validate_artifact(value, require_terminal=require_terminal, root=root)


def date_argument(value: str) -> str:
    """Reject an execution date outside the fixed V652 boundary."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def _read_bound_json(root: Path, reference: Mapping[str, Any]) -> JsonDict:
    """Load one protocol sidecar only when its exact bytes still match."""

    path = Path(str(reference.get("path") or ""))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file() or sha256_file(resolved) != reference.get("sha256"):
        raise ValueError(f"sidecar_hash_mismatch:{reference.get('path')}")
    value = _load_object(resolved)
    if not value:
        raise ValueError(f"sidecar_invalid:{reference.get('path')}")
    return value


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate exact sources, protocol bytes, sidecars, flags, and schedules."""

    required = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7422_v651_runtime_ownership.py"),
        Path("results/experiment_7422_v651_runtime_ownership.json"),
        Path("python/carnot/experiment_7429_v651_anchored_capture.py"),
        Path("results/experiment_7429_v651_anchored_capture.json"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/experiment_7437_v652_span_protocol.py"),
        PROTOCOL_PATH,
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    checks: list[JsonDict] = []
    for relative in required:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            _gate(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "==",
                "readable_nonempty_bytes",
                observed,
                observed == "readable_nonempty_bytes",
                "Required source bytes must exist before dependent work.",
                upstream=relative.as_posix(),
                path=relative.as_posix(),
                field="bytes",
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _gate(
            "driving_requirement",
            "precondition",
            "==",
            "REQ-VERIFY-7442",
            "REQ-VERIFY-7442" if "REQ-VERIFY-7442" in spec_text else None,
            "REQ-VERIFY-7442" in spec_text,
            "Implementation and model work require an existing capability requirement.",
            upstream=SPEC_PATH.as_posix(),
            path=SPEC_PATH.as_posix(),
            field="REQ-*",
        )
    )
    protocol_path = root / PROTOCOL_PATH
    value = _load_object(protocol_path)
    observed_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    checks.append(
        _gate(
            "span_protocol_artifact_hash",
            "precondition",
            "==",
            EXPECTED_PROTOCOL_SHA256,
            observed_hash,
            observed_hash == EXPECTED_PROTOCOL_SHA256,
            "A changed same-milestone producer cannot silently authorize model work.",
            upstream=PROTOCOL_PATH.as_posix(),
            path=PROTOCOL_PATH.as_posix(),
            field="sha256",
        )
    )
    checks.extend(protocol_gate_rows(value))
    protocol_errors = (
        protocol.independent_reduce_artifact(value, root=root, require_terminal=True)
        if value
        else ["protocol_missing"]
    )
    checks.append(
        _gate(
            "span_protocol_cold_replay",
            "precondition",
            "==",
            [],
            protocol_errors,
            not protocol_errors,
            "Protocol summaries cannot replace its independently replayed sidecars.",
            upstream=PROTOCOL_PATH.as_posix(),
            path=PROTOCOL_PATH.as_posix(),
            field="independent_reduce_artifact",
        )
    )
    manifest = dict(value.get("span_protocol_manifest") or {})
    try:
        panel = _read_bound_json(root, dict(manifest.get("panel") or {}))
        sealed = _read_bound_json(root, dict(manifest.get("schedule") or {}))
        evaluator = _read_bound_json(root, dict(manifest.get("evaluator") or {}))
        development_schedule = build_development_schedule(panel.get("development") or [])
        evaluation_schedule = build_evaluation_schedule(sealed.get("rows") or [])
    except (KeyError, TypeError, ValueError) as exc:
        panel = {}
        sealed = {}
        evaluator = {}
        development_schedule = []
        evaluation_schedule = []
        sidecar_error: str | None = f"{type(exc).__name__}:{exc}"
    else:
        sidecar_error = None
    checks.append(
        _gate(
            "bound_protocol_sidecars",
            "precondition",
            "==",
            {"error": None, "development_calls": 8, "evaluation_calls": 96},
            {
                "error": sidecar_error,
                "development_calls": len(development_schedule),
                "evaluation_calls": len(evaluation_schedule),
            },
            sidecar_error is None
            and len(development_schedule) == DEVELOPMENT_CALLS
            and len(evaluation_schedule) == EVALUATION_CALLS,
            "Panel, schedule, evaluator, and protocol bytes must remain exact.",
            upstream=PROTOCOL_PATH.as_posix(),
            path=PROTOCOL_PATH.as_posix(),
            field="span_protocol_manifest",
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7442" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        _gate(
            "current_task_not_quarantined",
            "precondition",
            "==",
            False,
            excluded,
            not excluded,
            "A quarantined task cannot reacquire scarce model capacity.",
            upstream="ops/exclusion_manifest.yaml",
            path="ops/exclusion_manifest.yaml",
            field=EXPERIMENT_ID,
        )
    )
    source_hashes: JsonDict = {}
    for relative in required:
        path = root / relative
        if path.is_file():
            original: JsonDict = {}
            if relative.suffix == ".json":
                source = _load_object(path)
                original = {
                    "original_verdict_class": source.get("verdict_class"),
                    "original_flagged_adversarial": source.get("flagged_adversarial"),
                }
            source_hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                **original,
            }
    return checks, {
        "protocol": value,
        "protocol_manifest": manifest,
        "panel": panel,
        "sealed_schedule": sealed,
        "evaluator": evaluator,
        "development_schedule": development_schedule,
        "evaluation_schedule": evaluation_schedule,
        "parser_controls": deepcopy(value.get("parser_control_rows") or []),
        "source_hashes": source_hashes,
    }


def _runtime_preconditions(
    root: Path, context: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover - host and GPU dependent.
    """Resolve native CUDA, cached Q4_K_M bytes, free capacity, and lease policy."""

    inherited = canary._runtime_preconditions(root, context, started)
    checks = [
        row
        for row in inherited
        if row.get("check") not in {"one_owned_rtx3090_slot", "bounded_lease_wait"}
    ]
    capacity = ownership.reduce_capacity_row(
        "exp7442-host",
        context.get("process_rows") or [],
        context.get("lease_rows") or [],
        context.get("gpu_query_receipts") or [],
    )
    context["available_gpu_uuids"] = list(capacity.get("available_gpu_uuids") or [])
    context["capacity_receipt"] = capacity
    checks.extend(
        [
            _gate(
                "repaired_gpu_inventory_query",
                "precondition",
                "==",
                True,
                capacity.get("query_ok"),
                capacity.get("query_ok") is True,
                "Inventory transport and current free capacity are separate facts.",
                upstream="nvidia-smi_and_gpu_lease_journal",
                path="host_runtime",
                field="query_ok",
            ),
            _gate(
                "one_free_rtx3090_slot",
                "precondition",
                ">=",
                1,
                capacity.get("available_capacity"),
                int(capacity.get("available_capacity", 0) or 0) >= 1,
                "One available device is required before task-owned lease acquisition.",
                upstream="nvidia-smi_and_gpu_lease_journal",
                path="host_runtime",
                field="available_rtx3090_slots",
            ),
            _gate(
                "bounded_lease_wait",
                "precondition",
                "==",
                LEASE_WAIT_TIMEOUT_S,
                LEASE_WAIT_TIMEOUT_S,
                True,
                "Lease contention cannot consume more than 120 seconds.",
                upstream="runtime_policy",
                path="runtime_policy",
                field="lease_wait_timeout_s",
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
    model_spec["decoding"] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED,
        "retry_budget": 0,
    }
    context["model_spec"] = model_spec
    return checks


def write_content_addressed_shard(root: Path, phase: str, value: Mapping[str, Any]) -> JsonDict:
    """Persist immutable evidence under its hash and return one byte receipt."""

    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


class _DevelopmentGateClosed(RuntimeError):  # pragma: no cover - live boundary.
    """Stop the owned server loop after a completed low-yield canary."""


class _ShardRecorder(canary.InvocationEventRecorder):  # pragma: no cover - live evidence.
    """Flush invocation events and enforce the frozen two-arm canary boundary."""

    def __init__(self, *args: Any, shard_root: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.shard_root = shard_root
        self.event_shards: list[JsonDict] = []
        self.response_shards: list[JsonDict] = []
        self.development_rows: list[JsonDict] = []

    def _append(self, call_id: str, operation: str, state: str) -> None:
        super()._append(call_id, operation, state)
        self.event_shards.append(
            write_content_addressed_shard(self.shard_root, "events", self.events[-1])
        )

    def __call__(self, phase: str, event: str, **details: Any) -> None:
        super().__call__(phase, event, **details)
        if (phase, event) == ("generation", "after_call") and details.get("completed") == 8:
            gate = reduce_development_gate(self.development_rows)
            if gate["capture_open"] is not True:
                raise _DevelopmentGateClosed(
                    "usable_development_outputs:"
                    + json.dumps(gate["usable_by_arm"], sort_keys=True)
                )


@contextmanager
def _shared_runtime_settings(
    recorder: _ShardRecorder, raw_root: Path
) -> Iterator[None]:  # pragma: no cover - restored native globals.
    """Apply the fixed budget and raw-first callback to one owned server."""

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
        transport = {
            "call_id": schedule.get("call_id"),
            "raw_request": deepcopy(response.get("raw_request") or {}),
            "raw_response": deepcopy(response.get("raw_response") or {}),
            "raw_reply": str(response.get("raw_reply") or ""),
            "attempted": True,
            "terminal_state": "response" if not response.get("error") else "request_error",
            "finish_reason": response.get("finish_reason"),
            "prompt_tokens": response.get("prompt_tokens"),
            "completion_tokens": response.get("completion_tokens"),
            "latency_s": response.get("latency_s"),
            "error": response.get("error"),
            "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        }
        recorder.response_shards.append(
            write_content_addressed_shard(raw_root, "responses", transport)
        )
        row = build_capture_row(schedule, transport)
        if schedule.get("capture_phase") == "development":
            recorder.development_rows.append(row)
        atomic_json(
            raw_root / "capture_checkpoint.json",
            {
                "status": "nonterminal_resumable",
                "completed_units": len(recorder.response_shards),
                "latest_response_shard": recorder.response_shards[-1],
            },
        )
        return row

    values = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "MAX_GENERATED_TOKENS": MAX_NEW_TOKENS,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "INFERENCE_WINDOW_TIMEOUT_S": AGGREGATE_LIVE_BUDGET_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": RANDOM_SEED,
            "evaluation": RANDOM_SEED,
            "resampling": RANDOM_SEED,
        },
        "render_public_prompt": lambda request: str(request["prompt"]),
        "build_call_row": build_row,
        "progress": recorder,
    }
    previous = {name: getattr(native_runtime, name) for name in values}
    try:
        for name, value in values.items():
            setattr(native_runtime, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(native_runtime, name, value)


def _normalize_runtime_row(
    schedule: Mapping[str, Any], response: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover
    """Bind native responses and post-failure fillers to the exact schedule."""

    if response.get("raw_request_sha256") and response.get("pair_id") == schedule.get("pair_id"):
        return deepcopy(dict(response))
    transport = {
        "raw_request": deepcopy(response.get("raw_request") or {}),
        "raw_response": deepcopy(response.get("raw_response") or {}),
        "raw_reply": str(response.get("raw_reply") or ""),
        "attempted": response.get("attempted") is True,
        "terminal_state": response.get("terminal_state") or "unstarted",
        "finish_reason": response.get("finish_reason"),
        "prompt_tokens": response.get("prompt_tokens"),
        "completion_tokens": response.get("completion_tokens"),
        "latency_s": response.get("latency_s"),
        "error": response.get("error"),
        "censored": response.get("censored"),
        "runtime_identity_receipt": deepcopy(response.get("runtime_identity_receipt") or {}),
    }
    return build_capture_row(schedule, transport)


def _capture_current(
    context: Mapping[str, Any], raw_dir: Path, started: float
) -> JsonDict:  # pragma: no cover - live model work.
    """Run eight canary calls and, when opened, all 96 evaluation calls."""

    recorder = _ShardRecorder(
        f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}",
        os.getpid(),
        started,
        shard_root=raw_dir,
    )
    development = deepcopy(list(context["development_schedule"]))
    evaluation = deepcopy(list(context["evaluation_schedule"]))
    combined = [*development, *evaluation]
    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = combined
    with _shared_runtime_settings(recorder, raw_dir):
        capture = native_runtime._live_capture(runtime_context, raw_dir / "native")
    recorder.close(capture)
    native_rows = list(capture.get("rows") or [])
    normalized = [
        _normalize_runtime_row(
            schedule,
            native_rows[index] if index < len(native_rows) else {},
        )
        for index, schedule in enumerate(combined)
    ]
    development_rows = normalized[:DEVELOPMENT_CALLS]
    gate = reduce_development_gate(development_rows)
    evaluation_rows = (
        normalized[DEVELOPMENT_CALLS:]
        if gate["capture_open"]
        else unstarted_evaluation_rows(evaluation)
    )
    row_shards = [
        write_content_addressed_shard(raw_dir, "evaluation", row) for row in evaluation_rows
    ]
    capture.update(
        {
            "development_rows": development_rows,
            "development_gate": gate,
            "rows": evaluation_rows,
            "current_run_id": recorder.run_id,
            "current_owner_pid": recorder.owner_pid,
            "current_invocation_events": recorder.events,
            "event_shards": recorder.event_shards,
            "response_shards": recorder.response_shards,
            "evaluation_shards": row_shards,
        }
    )
    atomic_json(
        raw_dir / "capture_checkpoint.json",
        {
            "status": "capture_complete_validation_pending",
            "development_gate": gate,
            "evaluation_dispositions": len(evaluation_rows),
            "event_shards": recorder.event_shards,
            "response_shards": recorder.response_shards,
            "evaluation_shards": row_shards,
        },
    )
    return capture


def _runner_receipt(capture: Mapping[str, Any], context: Mapping[str, Any]) -> JsonDict:
    """Bind native build, owned process, model bytes, CUDA offload, and lease release."""

    identity = dict(capture.get("runtime_identity") or {})
    gpu = dict(capture.get("gpu_receipts") or {})
    provenance = dict(gpu.get("provenance") or {})
    model_spec = dict(context.get("model_spec") or {})
    offload = canary.build_offload_receipt(
        identity, provenance, int(model_spec.get("model_block_count", 0) or 0)
    )
    release = dict(gpu.get("lease_release") or {})
    all_rows = [
        *list(capture.get("development_rows") or []),
        *list(capture.get("rows") or []),
    ]
    return {
        **deepcopy(identity),
        **offload,
        "runner_pid": identity.get("pid"),
        "runner_pid_start_ticks": identity.get("start_time_ticks"),
        "server_build": identity.get("server_build") or identity.get("build"),
        "runner_path": str(context.get("server_path") or ""),
        "runner_sha256": context.get("server_sha256"),
        "model_sha256": model_spec.get("sha256"),
        "model_revision": model_spec.get("revision"),
        "gpu_uuid": identity.get("gpu_uuid"),
        "lease_id": identity.get("lease_id"),
        "lease_released": release.get("released") is True,
        "lease_release": release,
        "cleanup": deepcopy(gpu.get("cleanup") or {}),
        "request_sha256": canonical_hash([row.get("raw_request_sha256") for row in all_rows]),
        "response_sha256": canonical_hash([row.get("raw_response_sha256") for row in all_rows]),
        "runtime_flags": deepcopy(model_spec.get("runtime_flags") or {}),
        "decoding": deepcopy(model_spec.get("decoding") or {}),
    }


def _byte_receipt(path: Path, relative_to: Path, phase: str) -> JsonDict:
    """Describe one already-persisted recovery input without rewriting it."""

    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "phase": phase,
    }


def _find_owned_lease_journal(lease_id: str) -> tuple[Path | None, JsonDict]:
    """Find the immutable task lease journal matching the raw runtime identity."""

    lease_root = Path("/tmp/carnot-gpu-leases")
    for path in sorted(lease_root.glob("*.journal.json")):
        value = _load_object(path)
        if value.get("lease_id") == lease_id:
            return path, value
    return None, {}


def _recovery_model_spec(root: Path, identity: Mapping[str, Any]) -> JsonDict:
    """Recover metadata for the exact current model hash from an authenticated source."""

    historical = _load_object(root / "results/experiment_7429_v651_anchored_capture.json")
    wanted_hash = identity.get("model_sha256")
    matches = [
        dict(row)
        for row in historical.get("model_specs") or []
        if isinstance(row, Mapping) and row.get("sha256") == wanted_hash
    ]
    spec = deepcopy(matches[0]) if len(matches) == 1 else {}
    spec.update(
        {
            "hf_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "path": identity.get("model_path"),
            "revision": identity.get("model_revision"),
            "bytes": identity.get("model_bytes"),
            "sha256": wanted_hash,
            "native_tokenizer": "embedded_gguf",
            "native_chat_template": True,
            "runtime_flags": {
                "n_gpu_layers": "all",
                "split_mode": "none",
                "parallel": 1,
                "fit": "off",
                "offline": True,
                "jinja": True,
                "reasoning": "off",
            },
            "decoding": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": 1,
                "top_p": 1.0,
                "seed": RANDOM_SEED,
                "retry_budget": 0,
            },
        }
    )
    return spec


def _recover_owned_capture(
    root: Path, context: JsonDict, raw_root: Path
) -> tuple[JsonDict, JsonDict, list[JsonDict]]:  # pragma: no cover - real recovery bytes.
    """Rebuild the failed live attempt solely from its owned persisted evidence."""

    native_dir = raw_root / "native"
    call_paths = sorted(native_dir.glob("call_*.json"))
    if len(call_paths) != 1:
        raise ValueError(f"recovery_native_call_count:{len(call_paths)}")
    native_row = _load_object(call_paths[0])
    identity = dict(native_row.get("runtime_identity_receipt") or {})
    lease_path, lease = _find_owned_lease_journal(str(identity.get("lease_id") or ""))
    log_paths = sorted(native_dir.glob("llama_server_*.log"))
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace") if log_paths else ""
    event_paths = sorted((raw_root / "events").glob("*.json"))
    events = sorted(
        (_load_object(path) for path in event_paths),
        key=lambda row: int(row.get("monotonic_ns", 0) or 0),
    )
    events = reconcile_terminal_events(
        events,
        [native_row],
        monotonic_ns=time.monotonic_ns(),
    )
    if len(events) > len(event_paths):
        write_content_addressed_shard(raw_root, "events", events[-1])
    event_paths = sorted((raw_root / "events").glob("*.json"))

    development_schedule = list(context["development_schedule"])
    if native_row.get("call_id") != development_schedule[0].get("call_id"):
        raise ValueError("recovery_call_schedule_mismatch")
    development_rows = [build_capture_row(development_schedule[0], native_row)]
    development_rows.extend(
        build_capture_row(
            row,
            {
                "attempted": False,
                "terminal_state": "unstarted",
                "raw_request": {},
                "raw_response": {},
                "raw_reply": "",
            },
        )
        for row in development_schedule[1:]
    )
    evaluation_rows = unstarted_evaluation_rows(context["evaluation_schedule"])
    evaluation_shards = [
        write_content_addressed_shard(raw_root, "evaluation", row) for row in evaluation_rows
    ]
    response_paths = sorted((raw_root / "responses").glob("*.json"))
    capture = {
        "development_rows": development_rows,
        "development_gate": reduce_development_gate(development_rows),
        "rows": evaluation_rows,
        "current_invocation_events": events,
        "current_run_id": events[0].get("run_id") if events else None,
        "current_owner_pid": events[0].get("owner_pid") if events else None,
        "event_shards": [_byte_receipt(path, raw_root, "events") for path in event_paths],
        "response_shards": [_byte_receipt(path, raw_root, "responses") for path in response_paths],
        "evaluation_shards": evaluation_shards,
        "recovery_timing": {
            "started_at_utc": datetime.fromtimestamp(
                min(path.stat().st_mtime for path in event_paths), UTC
            )
            .isoformat()
            .replace("+00:00", "Z"),
            "basis": "earliest_persisted_invocation_event_mtime",
        },
        "runtime_identity": {
            **identity,
            "server_build": dict(native_row.get("raw_response") or {}).get("system_fingerprint"),
        },
        "gpu_receipts": {
            "provenance": {
                "provenance_ok": identity.get("cuda_provenance_ok") is True,
                "cuda_log_evidence": "CUDA0   : NVIDIA GeForce RTX 3090" in log_text,
                "task_owned_vram_mb": identity.get("task_owned_vram_mb"),
            },
            "lease_release": {
                "released": lease.get("released") is True,
                "lease_id": lease.get("lease_id"),
                "device_uuid": lease.get("device_uuid"),
                "pid": dict(lease.get("owner") or {}).get("pid"),
                "pid_start_ticks": dict(lease.get("owner") or {}).get("pid_start_ticks"),
                "phase": lease.get("phase"),
                "checksum": lease.get("checksum"),
                "signals_sent": dict(lease.get("recovery") or {}).get("signals_sent") or [],
            },
            "cleanup": {
                "bounded": True,
                "leak_free": lease.get("released") is True
                and dict(lease.get("unload_evidence") or {}).get("observed") is True,
                "exit_code": dict(lease.get("exit_evidence") or {}).get("exit_code"),
                "unrelated_process_kill_count_delta": 0,
            },
        },
    }
    model_spec = _recovery_model_spec(root, identity)
    context["model_spec"] = model_spec
    context["server_path"] = list(identity.get("command") or [""])[0]
    server = Path(str(context["server_path"]))
    context["server_sha256"] = sha256_file(server) if server.is_file() else None
    runner = _runner_receipt(capture, context)
    raw_request = dict(native_row.get("raw_request") or {})
    lease_owner = dict(lease.get("owner") or {})
    checks = [
        _gate(
            "recovered_native_terminal_response",
            "evidence",
            "==",
            "response",
            native_row.get("terminal_state"),
            native_row.get("terminal_state") == "response",
            "Only a persisted terminal native response may close the unfinished event.",
            upstream="owned_runtime",
            path=call_paths[0].relative_to(root).as_posix(),
            field="terminal_state",
        ),
        _gate(
            "recovered_owned_process_identity",
            "evidence",
            "==",
            {
                "owner_pid": capture["current_owner_pid"],
                "lease_id": identity.get("lease_id"),
                "owned_by_task": True,
            },
            {
                "owner_pid": lease_owner.get("pid"),
                "lease_id": lease.get("lease_id"),
                "owned_by_task": identity.get("owned_by_task"),
            },
            lease_owner.get("pid") == capture["current_owner_pid"]
            and lease.get("lease_id") == identity.get("lease_id")
            and identity.get("owned_by_task") is True,
            "The raw response, event ledger, and kernel lease must name one owner.",
            upstream="gpu_lease_journal",
            path=str(lease_path) if lease_path else "/tmp/carnot-gpu-leases",
            field="owner",
        ),
        _gate(
            "recovered_exact_model_identity",
            "evidence",
            "==",
            {"hf_id": MODEL_ID, "quantization": QUANTIZATION},
            {
                "hf_id": model_spec.get("hf_id"),
                "quantization": identity.get("quantization"),
            },
            model_spec.get("hf_id") == MODEL_ID
            and identity.get("quantization") == QUANTIZATION
            and model_spec.get("sha256") == identity.get("model_sha256"),
            "Recovery must retain the exact current cached GGUF identity.",
            upstream="owned_runtime",
            path=call_paths[0].relative_to(root).as_posix(),
            field="runtime_identity_receipt.model_sha256",
        ),
        _gate(
            "recovered_fixed_generation_budget",
            "evidence",
            "==",
            MAX_NEW_TOKENS,
            raw_request.get("max_tokens"),
            raw_request.get("max_tokens") == MAX_NEW_TOKENS,
            "The persisted request must retain the fixed extraction budget.",
            upstream="owned_runtime",
            path=call_paths[0].relative_to(root).as_posix(),
            field="raw_request.max_tokens",
        ),
        _gate(
            "recovered_cuda_offload",
            "evidence",
            "==",
            True,
            runner.get("all_layers_offloaded"),
            runner.get("all_layers_offloaded") is True,
            "Current CUDA log, owned VRAM, and all-layer command must agree.",
            upstream="owned_runtime",
            path=log_paths[0].relative_to(root).as_posix() if log_paths else "missing",
            field="all_layers_offloaded",
        ),
        _gate(
            "recovered_owned_lease_released",
            "safety",
            "==",
            True,
            runner.get("lease_released"),
            runner.get("lease_released") is True,
            "Recovery cannot leave the owned server or lease resident.",
            upstream="gpu_lease_journal",
            path=str(lease_path) if lease_path else "/tmp/carnot-gpu-leases",
            field="released",
        ),
    ]
    if lease_path:
        context["source_hashes"]["current_lease_journal"] = {
            "path": str(lease_path),
            "sha256": sha256_file(lease_path),
        }
    context["source_hashes"]["current_native_call"] = {
        "path": call_paths[0].relative_to(root).as_posix(),
        "sha256": sha256_file(call_paths[0]),
    }
    if log_paths:
        context["source_hashes"]["current_native_log"] = {
            "path": log_paths[0].relative_to(root).as_posix(),
            "sha256": sha256_file(log_paths[0]),
        }
    return capture, runner, checks


def _span(
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
    checkpoint: str,
) -> JsonDict:  # pragma: no cover - real monotonic evidence.
    """Close one disjoint phase with a checkpoint and completed-unit count."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": round(phase_started - run_started, 6),
        "end_s": round(ended - run_started, 6),
        "duration_s": round(ended - phase_started, 6),
        "completed_units": completed_units,
        "checkpoint": checkpoint,
        "checkpoint_at_utc": utc_now(),
    }


def _run_affected_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - subprocess orchestration.
    """Run the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7442-validation-", dir="/tmp"))
    commands = build_command_plan(root, VALIDATION_MANIFEST, private)
    plan_errors = validate_command_plan(root, VALIDATION_MANIFEST, commands)
    if plan_errors:
        return [], {"passed": False, "plan_errors": plan_errors}
    progress(started, "affected_validation", "before_subprocesses", planned=len(commands))
    receipts = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    progress(
        started,
        "affected_validation",
        "after_subprocesses",
        completed_units=len(receipts),
        passed=reduced.get("passed"),
    )
    return receipts, {**reduced, "plan_errors": []}


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build fresh replay, independent reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7442_v652_span_capture import independent_reduce_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=independent_reduce_artifact(v,root=pathlib.Path.cwd(),require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);raise SystemExit(bool(e))"
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


def _measured_artifact(
    *,
    context: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    reduced: Mapping[str, Any],
    runner: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    affected_ok: bool,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at: str,
    duration_s: float,
    model_duration_s: float,
    computation_duration_s: float,
    validation_duration_s: float,
    cold_start_duration_s: float,
    require_terminal: bool,
    flagged_adversarial: bool,
    producer_runtime_error: str | None = None,
) -> JsonDict:
    """Build one candidate or terminal artifact from raw measured evidence."""

    current = canary.reduce_current_events(
        capture.get("current_invocation_events") or [],
        run_id=str(capture.get("current_run_id")),
        owner_pid=int(capture.get("current_owner_pid")),
    )
    development_gate = dict(capture.get("development_gate") or {})
    receipt_errors = required_receipt_errors(receipts) if require_terminal else []
    all_dispositions = len(reduced["extraction_rows"]) == EVALUATION_CALLS and all(
        row.get("disposition") != "unstarted" for row in reduced["extraction_rows"]
    )
    complete = int(
        require_terminal
        and affected_ok
        and not receipt_errors
        and development_gate.get("capture_open") is True
        and all_dispositions
        and runner.get("all_layers_offloaded") is True
        and runner.get("lease_released") is True
        and not flagged_adversarial
        and producer_runtime_error is None
    )
    gates = [
        *[deepcopy(dict(row)) for row in checks],
        _gate(
            "producer_runtime_integrity",
            "validation",
            "==",
            None,
            producer_runtime_error,
            producer_runtime_error is None,
            "A producer adapter exception disqualifies even hash-bound partial evidence.",
            upstream="owned_runtime",
            path="owned_runtime/capture_checkpoint.json",
            field="producer_runtime_error",
        ),
        _gate(
            "development_gate_open",
            "completion",
            "==",
            True,
            development_gate.get("capture_open"),
            development_gate.get("capture_open") is True,
            "Each arm needs three complete, nonempty, exact canary outputs.",
        ),
        _gate(
            "all_evaluation_dispositions",
            "completion",
            "==",
            EVALUATION_CALLS,
            sum(row.get("disposition") != "unstarted" for row in reduced["extraction_rows"]),
            all_dispositions,
            "All 96 evaluation calls must keep one valid terminal disposition.",
        ),
        _gate(
            "cuda_all_layer_offload",
            "evidence",
            "==",
            True,
            runner.get("all_layers_offloaded"),
            runner.get("all_layers_offloaded") is True,
            "CPU fallback cannot supply current representation evidence.",
            upstream="runner_receipt",
            path="runner_receipt",
            field="all_layers_offloaded",
        ),
        _gate(
            "affected_validation",
            "validation",
            "==",
            True,
            affected_ok,
            affected_ok,
            "Every frozen affected-file command must pass.",
            upstream="validation_receipts",
            path="validation_receipts",
            field="affected_validation",
        ),
        _gate(
            "terminal_validation",
            "validation",
            "==",
            True,
            not receipt_errors and not flagged_adversarial if require_terminal else None,
            require_terminal and not receipt_errors and not flagged_adversarial,
            "Fresh replay, independent reduction, adversarial verification, and strict rows must pass.",
            upstream="validation_receipts",
            path="validation_receipts",
            field="terminal_validation",
        ),
        _gate(
            "span_representation_value",
            "benefit",
            "==",
            1,
            reduced["span_value_score"],
            reduced["span_value_score"] == 1,
            "Value needs a positive paired CI, no qualifier loss, and lower output-token cost.",
        ),
    ]
    disqualified = producer_runtime_error is not None or (
        require_terminal and (not affected_ok or bool(receipt_errors) or flagged_adversarial)
    )
    if producer_runtime_error is not None:
        status = "complete_producer_runtime_failed"
        verdict_class = "disqualified"
        verdict = "complete_disqualified_span_capture_producer_runtime_error"
    elif disqualified:
        status = "complete_required_validation_failed"
        verdict_class = "disqualified"
        verdict = "complete_disqualified_span_capture_required_check_failed"
    elif development_gate.get("capture_open") is not True:
        status = "complete_development_gate_closed"
        verdict_class = "null"
        verdict = "complete_null_span_capture_development_gate_closed"
    elif reduced["span_value_score"] == 1:
        status = "complete_span_capture"
        verdict_class = "circular_positive"
        verdict = "complete_circular_positive_span_representation_value"
    else:
        status = "complete_span_capture"
        verdict_class = "null"
        verdict = "complete_null_span_capture_no_paired_representation_value"
    artifact = _base_artifact()
    source_hashes = deepcopy(dict(context.get("source_hashes") or {}))
    source_hashes.update(
        {
            "current_event_shards": deepcopy(list(capture.get("event_shards") or [])),
            "current_response_shards": deepcopy(list(capture.get("response_shards") or [])),
            "current_evaluation_shards": deepcopy(list(capture.get("evaluation_shards") or [])),
        }
    )
    artifact.update(
        {
            "status": status,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "model_specs": [deepcopy(dict(context.get("model_spec") or {}))],
            **current,
            "current_invocation_events": deepcopy(
                list(capture.get("current_invocation_events") or [])
            ),
            "current_run_id": capture.get("current_run_id"),
            "current_owner_pid": capture.get("current_owner_pid"),
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_details": {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "device": runner.get("gpu_uuid"),
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "cuda": runner.get("all_layers_offloaded") is True,
                "cpu": os.uname().machine,
                "external_device": None,
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "duration_s": round(duration_s, 6),
            "model_duration_s": round(model_duration_s, 6),
            "computation_duration_s": round(computation_duration_s, 6),
            "validation_duration_s": round(validation_duration_s, 6),
            "cold_start_duration_s": round(cold_start_duration_s, 6),
            "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
            "source_artifact_hashes": source_hashes,
            "rows": deepcopy(reduced["extraction_rows"]),
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                **deepcopy(dict(reduced["sample_counts"])),
                "development_attempted": development_gate.get("attempted", 0),
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": _gate_summary(gates),
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "flagged_adversarial": flagged_adversarial,
            "validation_receipts": [deepcopy(dict(row)) for row in receipts],
            "span_capture_complete_score": complete,
            "development_rows": deepcopy(list(capture.get("development_rows") or [])),
            "development_gate": development_gate,
            "raw_capture_manifest": _raw_manifest(reduced["extraction_rows"]),
            "runner_receipt": deepcopy(dict(runner)),
            "producer_runtime_error": producer_runtime_error,
            "protocol_receipt": {
                "path": PROTOCOL_PATH.as_posix(),
                "sha256": EXPECTED_PROTOCOL_SHA256,
                "manifest_hash": dict(context.get("protocol_manifest") or {}).get("manifest_hash"),
                "schedule_sha256": (
                    list(context.get("evaluation_schedule") or [{}])[0].get(
                        "evaluation_schedule_sha256"
                    )
                    if context.get("evaluation_schedule")
                    else None
                ),
                "original_verdict_class": dict(context.get("protocol") or {}).get("verdict_class"),
                "original_flagged_adversarial": dict(context.get("protocol") or {}).get(
                    "flagged_adversarial"
                ),
            },
            **deepcopy(dict(reduced)),
        }
    )
    artifact["rows"] = deepcopy(artifact["extraction_rows"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _owned_run_start(runner: Mapping[str, Any]) -> tuple[float, str]:
    """Recover the original owner start on this still-running host boot."""

    ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
    parent = dict(runner.get("parent_identity") or {})
    ticks = int(parent.get("start_time_ticks", 0) or 0)
    if ticks <= 0:
        raise ValueError("recovery_owner_start_ticks_missing")
    boot_line = next(
        (line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")),
        "",
    )
    if not boot_line:
        raise ValueError("recovery_boot_time_missing")
    boot_utc = int(boot_line.split()[1])
    monotonic_start = ticks / ticks_per_second
    uptime_s = float(Path("/proc/uptime").read_text().split()[0])
    if monotonic_start > uptime_s:
        raise ValueError("recovery_owner_boot_mismatch")
    wall_start = datetime.fromtimestamp(boot_utc + monotonic_start, UTC)
    return monotonic_start, wall_start.isoformat().replace("+00:00", "Z")


def _recover_and_publish(
    *,
    root: Path,
    output: Path,
    raw_dir: Path,
    checks: list[JsonDict],
    context: JsonDict,
    progress_started: float,
) -> JsonDict:  # pragma: no cover - one owned failure recovery.
    """Finish the existing one-shot attempt without loading or calling a model."""

    progress(progress_started, "recovery", "before_raw_replay", model_calls_started=0)
    recovery_started = time.monotonic()
    capture, runner, recovery_checks = _recover_owned_capture(
        root, context, raw_dir / "owned_runtime"
    )
    checks.extend(recovery_checks)
    events = list(capture.get("current_invocation_events") or [])
    load_ticks = [
        int(row.get("monotonic_ns", 0) or 0)
        for row in events
        if row.get("operation") == "model_load"
    ]
    development_rows = list(capture.get("development_rows") or [])
    response_latency = sum(float(row.get("latency_s", 0.0) or 0.0) for row in development_rows)
    model_duration = (
        (max(load_ticks) - min(load_ticks)) / 1_000_000_000 + response_latency
        if load_ticks
        else response_latency
    )
    try:
        original_started, started_at = _owned_run_start(runner)
    except ValueError as exc:
        original_started = recovery_started - model_duration
        started_at = str(dict(capture.get("recovery_timing") or {}).get("started_at_utc"))
        checks.append(
            _gate(
                "recovery_monotonic_continuity",
                "evidence",
                "==",
                True,
                False,
                False,
                "A reboot breaks monotonic continuity, so active phase durations replace wall time.",
                upstream="owned_runtime",
                path="/proc/uptime",
                field=str(exc),
            )
        )
        monotonic_continuity = False
    else:
        monotonic_continuity = True
    model_start_s = (
        min(load_ticks) / 1_000_000_000 - original_started
        if load_ticks and monotonic_continuity
        else 0.0
    )
    model_end_s = model_start_s + model_duration
    spans: list[JsonDict] = [
        {
            "phase": "model_load_and_generation",
            "start_s": round(model_start_s, 6),
            "end_s": round(model_end_s, 6),
            "duration_s": round(model_duration, 6),
            "completed_units": 1,
            "checkpoint": (raw_dir / "owned_runtime/native/call_00.json")
            .relative_to(root)
            .as_posix(),
            "checkpoint_at_utc": datetime.fromtimestamp(
                int(dict(development_rows[0].get("raw_response") or {}).get("created", 0) or 0),
                UTC,
            )
            .isoformat()
            .replace("+00:00", "Z"),
        }
    ]
    reduction_started = time.monotonic()
    reduced = reduce_evaluation(capture["rows"], context.get("parser_controls") or [])
    computation_duration = time.monotonic() - reduction_started
    spans.append(
        _span(
            "producer_failure_reconciliation",
            recovery_started,
            original_started,
            1,
            (raw_dir / "owned_runtime/capture_checkpoint.json").relative_to(root).as_posix(),
        )
    )
    progress(
        progress_started,
        "recovery",
        "after_raw_replay",
        completed_units=1,
        evaluation_unstarted=reduced["sample_counts"]["unstarted"],
    )

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, progress_started)
    affected_ok = bool(affected.get("passed"))
    spans.append(
        _span(
            "affected_validation",
            validation_started,
            original_started,
            len(affected_receipts),
            "affected_receipts_complete",
        )
    )
    producer_error = "KeyError:'parse_status'"
    candidate = _measured_artifact(
        context=context,
        checks=checks,
        capture=capture,
        reduced=reduced,
        runner=runner,
        receipts=affected_receipts,
        affected_ok=affected_ok,
        phase_spans=spans,
        started_at=started_at,
        duration_s=time.monotonic() - original_started,
        model_duration_s=model_duration,
        computation_duration_s=computation_duration,
        validation_duration_s=time.monotonic() - validation_started,
        cold_start_duration_s=0.0,
        require_terminal=False,
        flagged_adversarial=False,
        producer_runtime_error=producer_error,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_started = time.monotonic()
    terminal_specs = _terminal_commands(root, candidate_path)
    progress(
        progress_started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(terminal_specs),
    )
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec,
                "safety" if spec.name == "adversarial_verify" else "completion",
                True,
            )
            for spec in terminal_specs
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    all_receipts = [*affected_receipts, *terminal_receipts]
    terminal_errors = required_receipt_errors(all_receipts)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    progress(
        progress_started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=not terminal_errors and not flagged,
    )
    spans.append(
        _span(
            "terminal_validation",
            terminal_started,
            original_started,
            len(terminal_receipts),
            "terminal_receipts_complete",
        )
    )
    cold_duration = sum(
        float(row.get("duration_s", 0.0) or 0.0)
        for row in terminal_receipts
        if row.get("name") in {"declared_entrypoint_cold_replay", "independent_cold_reducer"}
    )
    final = _measured_artifact(
        context=context,
        checks=checks,
        capture=capture,
        reduced=reduced,
        runner=runner,
        receipts=all_receipts,
        affected_ok=affected_ok,
        phase_spans=spans,
        started_at=started_at,
        duration_s=time.monotonic() - original_started,
        model_duration_s=model_duration,
        computation_duration_s=computation_duration,
        validation_duration_s=time.monotonic() - validation_started,
        cold_start_duration_s=cold_duration,
        require_terminal=True,
        flagged_adversarial=flagged,
        producer_runtime_error=producer_error,
    )
    errors = independent_reduce_artifact(final, root=root, require_terminal=True)
    if errors:
        final.update(
            {
                "status": "complete_internal_validation_failed",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "verdict_class": "disqualified",
                "flagged_adversarial": True,
                "span_capture_complete_score": 0,
                "internal_validation_errors": errors,
            }
        )
        final["reproducibility_checksum"] = artifact_checksum(final)
    progress(progress_started, "write", "before_atomic_publish", path=output)
    atomic_json(output, final)
    progress(
        progress_started,
        "write",
        "after_atomic_publish",
        completed_units=1,
        verdict=final["honest_verdict"],
    )
    return final


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live capability E2E.
    """Authenticate, capture once, validate, reduce independently, and publish."""

    started = time.monotonic()
    started_at = utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "start", completed_units=0)
    checks, context = collect_preconditions(root)
    checks.insert(
        0,
        _gate(
            "run_date",
            "precondition",
            "==",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            "The sealed experiment runs on its declared date.",
            upstream="execution_contract",
            path="execution_contract",
            field="run_date",
        ),
    )
    spans.append(
        _span("preconditions_static", phase_started, started, len(checks), "static_authenticated")
    )
    progress(
        started,
        "preconditions",
        "static_complete",
        completed_units=len(checks),
        passed=all(row.get("passed") is True for row in checks),
    )
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(checks)
        blocked.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
                "source_artifact_hashes": deepcopy(context.get("source_hashes") or {}),
            }
        )
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        progress(started, "write", "before_atomic_publish", path=output)
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_publish", completed_units=1)
        return blocked

    recovery_call = raw_dir / "owned_runtime/native/call_00.json"
    if recovery_call.is_file() and not output.exists():
        progress(
            started,
            "recovery",
            "owned_checkpoint_detected",
            path=recovery_call.relative_to(root),
            additional_model_calls=0,
        )
        return _recover_and_publish(
            root=root,
            output=output,
            raw_dir=raw_dir,
            checks=checks,
            context=context,
            progress_started=started,
        )

    phase_started = time.monotonic()
    progress(started, "preconditions", "before_runtime_checks", completed_units=len(checks))
    runtime_checks = _runtime_preconditions(root, context, started)
    checks.extend(runtime_checks)
    progress(
        started,
        "preconditions",
        "after_runtime_checks",
        completed_units=len(runtime_checks),
        passed=all(row.get("passed") is True for row in runtime_checks),
    )
    spans.append(
        _span(
            "preconditions_runtime",
            phase_started,
            started,
            len(runtime_checks),
            "runtime_authenticated",
        )
    )
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        blocked = build_blocked_artifact(checks)
        blocked.update(
            {
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "duration_s": round(time.monotonic() - started, 6),
                "phase_spans": spans,
                "model_specs": [deepcopy(dict(context.get("model_spec") or {}))],
                "inference_substrate_details": deepcopy(
                    dict(context.get("capacity_receipt") or {})
                ),
                "source_artifact_hashes": deepcopy(context.get("source_hashes") or {}),
            }
        )
        blocked["reproducibility_checksum"] = artifact_checksum(blocked)
        progress(started, "write", "before_atomic_publish", path=output)
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_publish", completed_units=1)
        return blocked

    phase_started = time.monotonic()
    progress(
        started,
        "generation",
        "before_model_load",
        completed_units=0,
        planned_development=DEVELOPMENT_CALLS,
        planned_evaluation=EVALUATION_CALLS,
    )
    capture = _capture_current(context, raw_dir / "owned_runtime", started)
    model_duration = time.monotonic() - phase_started
    progress(
        started,
        "generation",
        "after_model_load_and_generation",
        completed_units=len(capture.get("development_rows") or []) + len(capture.get("rows") or []),
        canary_open=dict(capture.get("development_gate") or {}).get("capture_open"),
    )
    spans.append(
        _span(
            "model_load_and_generation",
            phase_started,
            started,
            len(capture.get("development_rows") or []) + len(capture.get("rows") or []),
            (raw_dir / "owned_runtime/capture_checkpoint.json").as_posix(),
        )
    )

    phase_started = time.monotonic()
    progress(started, "reduction", "start", completed_units=0)
    reduced = reduce_evaluation(capture.get("rows") or [], context.get("parser_controls") or [])
    runner = _runner_receipt(capture, context)
    computation_duration = time.monotonic() - phase_started
    progress(
        started,
        "reduction",
        "complete",
        completed_units=len(reduced["extraction_rows"]),
        span_value_score=reduced["span_value_score"],
    )
    spans.append(
        _span(
            "independent_reduction",
            phase_started,
            started,
            EVALUATION_CALLS,
            "raw_rows_reduced",
        )
    )

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    affected_ok = bool(affected.get("passed"))
    spans.append(
        _span(
            "affected_validation",
            validation_started,
            started,
            len(affected_receipts),
            "affected_receipts_complete",
        )
    )
    candidate = _measured_artifact(
        context=context,
        checks=checks,
        capture=capture,
        reduced=reduced,
        runner=runner,
        receipts=affected_receipts,
        affected_ok=affected_ok,
        phase_spans=spans,
        started_at=started_at,
        duration_s=time.monotonic() - started,
        model_duration_s=model_duration,
        computation_duration_s=computation_duration,
        validation_duration_s=time.monotonic() - validation_started,
        cold_start_duration_s=0.0,
        require_terminal=False,
        flagged_adversarial=False,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)

    terminal_started = time.monotonic()
    terminal_specs = _terminal_commands(root, candidate_path)
    if tuple(spec.name for spec in terminal_specs) != TERMINAL_CHECK_NAMES:
        raise RuntimeError("terminal_command_name_drift")
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned=len(terminal_specs),
    )
    terminal_receipts = run_categorized_commands(
        root,
        [
            PlannedCommand(
                spec,
                "safety" if spec.name == "adversarial_verify" else "completion",
                True,
            )
            for spec in terminal_specs
        ],
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_errors = required_receipt_errors([*affected_receipts, *terminal_receipts])
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal_receipts),
        passed=not terminal_errors and not flagged,
    )
    spans.append(
        _span(
            "terminal_validation",
            terminal_started,
            started,
            len(terminal_receipts),
            "terminal_receipts_complete",
        )
    )
    cold_duration = sum(
        float(row.get("duration_s", 0.0) or 0.0)
        for row in terminal_receipts
        if row.get("name") in {"declared_entrypoint_cold_replay", "independent_cold_reducer"}
    )
    final = _measured_artifact(
        context=context,
        checks=checks,
        capture=capture,
        reduced=reduced,
        runner=runner,
        receipts=[*affected_receipts, *terminal_receipts],
        affected_ok=affected_ok,
        phase_spans=spans,
        started_at=started_at,
        duration_s=time.monotonic() - started,
        model_duration_s=model_duration,
        computation_duration_s=computation_duration,
        validation_duration_s=time.monotonic() - validation_started,
        cold_start_duration_s=cold_duration,
        require_terminal=True,
        flagged_adversarial=flagged,
    )
    errors = independent_reduce_artifact(final, root=root, require_terminal=True)
    if errors:
        final.update(
            {
                "status": "complete_internal_validation_failed",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "verdict_class": "disqualified",
                "flagged_adversarial": True,
                "span_capture_complete_score": 0,
                "internal_validation_errors": errors,
            }
        )
        final["reproducibility_checksum"] = artifact_checksum(final)
    progress(started, "write", "before_atomic_publish", path=output)
    atomic_json(output, final)
    progress(
        started,
        "write",
        "after_atomic_publish",
        completed_units=1,
        verdict=final["honest_verdict"],
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the one-shot capture or cold-replay one measured candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress(started, "entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = _load_object(args.validate)
        names = {
            row.get("name")
            for row in value.get("validation_receipts") or []
            if isinstance(row, Mapping)
        }
        require_terminal = set(TERMINAL_CHECK_NAMES).issubset(names)
        errors = independent_reduce_artifact(
            value, root=REPO_ROOT, require_terminal=require_terminal
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        json.dumps(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "honest_verdict": result["honest_verdict"],
                "span_capture_complete_score": result["span_capture_complete_score"],
                "span_value_score": result["span_value_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
