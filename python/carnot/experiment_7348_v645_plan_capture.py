"""Capture current local-model plans for the sealed V645 public panel.

The model sees public requests and the public plan schema only. Private rules
open after raw replies are sealed, so source errors stay separate from oracle
failures.

Spec refs: REQ-CL-7348 and SCENARIO-CL-7348-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
import time
from typing import Any, Callable, Iterator

from carnot import experiment_7344_v645_executor_fixture as fixture_mod
from carnot import experiment_7347_v645_plan_canary as canary_mod
from carnot.experiment_7330_v644_public_learner import (
    canonical_bytes,
    sha256_json,
    validate_public_request,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260916"
MILESTONE = "2026.09.645"
EXPERIMENT_ID = "exp7348-plan-capture"
SCHEMA = "carnot.exp7348.v645_plan_capture.v1"
TASK_ID = "experiment_7348_v645_plan_capture"
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": QWEN_MODEL_ID, "quantization": QUANTIZATION}]
FIXTURE_PATH = Path("results/experiment_7344_v645_executor_fixture.json")
CANARY_PATH = Path("results/experiment_7347_v645_plan_canary.json")
PUBLIC_MANIFEST_PATH = REPO_ROOT / (
    "results/raw/experiment_7344_v645_executor_fixture/public/public_manifest.json"
)
PRIVATE_EXECUTOR_PATH = Path("scripts/experiments/experiment_7330_v644_private_executor.py")
MODULE_PATH = Path("python/carnot/experiment_7348_v645_plan_capture.py")
CANARY_MODULE_PATH = Path("python/carnot/experiment_7347_v645_plan_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7348_v645_plan_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7348_v645_plan_capture.py")
CANARY_TEST_PATH = Path("tests/python/test_experiment_7347_v645_plan_canary.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
RESULT_PATH = Path("results/experiment_7348_v645_plan_capture.json")
RAW_DIR = Path("results/raw/experiment_7348_v645_plan_capture")
SCHEDULE_PATH = RAW_DIR / "schedule_manifest.json"
CANDIDATE_MANIFEST_PATH = RAW_DIR / "candidate_manifest.json"
EVALUATION_PATH = RAW_DIR / "private_evaluation.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7348_v645_plan_capture.json")
RANDOM_SEED = {"development": 7_348_101, "evaluation": 7_348_201, "resampling": 7_348_301}
PLANNED_CALLS = 128
MAX_GENERATED_TOKENS = 256
MODEL_LOAD_TIMEOUT_S = 600.0
GENERATION_TIMEOUT_S = 2_100.0
REQUEST_TIMEOUT_S = 300.0
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input and resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; LLM tasks include unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked result names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
    "honest_verdict": "Completed work starts complete_; external absence starts blocked_ with its failed check.",
    "verdict_class": "Use positive, circular_positive, null, blocked, disqualified, or partial. Only unfinished own work is partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "plan_capture_complete_score": "Complete accounting includes failures; it does not assert usable proposals.",
    "candidate_manifest_path": "Consumers use sealed exact candidates with source-call hashes.",
    "source_fidelity_rows": "Record all public fields and paired renaming outcomes.",
    "generation_cost_rows": "Carry costs into every downstream arm without calling replay a live run.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


def sha256_text(value: str) -> str:
    """Hash exact text so a changed prompt or reply cannot keep its identity."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:  # pragma: no cover - live evidence helper.
    """Hash exact file bytes without normalizing durable evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:  # pragma: no cover - wall-clock evidence only.
    """Record real UTC boundaries while durations use a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary and long-operation boundary immediately."""

    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[exp7348] phase={phase} event={event} {suffix}".rstrip(), flush=True)


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Retain both sides of one prerequisite or acceptance decision."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Name the first failed check without hiding its exact observed value."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "check_count": len(checks),
            "failed_check_count": 0,
            "failed_check": None,
            "upstream": EXPERIMENT_ID,
            "artifact_field": "plan_capture_complete_score",
            "expected_value": 1,
            "observed_value": 1,
            "passed": True,
        }
    return {
        "check_count": len(checks),
        "failed_check_count": sum(row.get("passed") is not True for row in checks),
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "artifact_field": failed.get("artifact_field"),
        "expected_value": deepcopy(failed.get("expected_value")),
        "observed_value": deepcopy(failed.get("observed_value")),
        "passed": False,
    }


def dependency_gate_rows(fixture: Mapping[str, Any], canary: Mapping[str, Any]) -> list[JsonDict]:
    """Reject blocked, partial, disqualified, or quarantined same-milestone inputs."""

    fixture_expected = {
        "status": "complete",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "executor_fixture_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "circular_positive",
    }
    fixture_observed = {key: fixture.get(key) for key in fixture_expected}
    fixture_quarantined = "quarantin" in str(fixture.get("honest_verdict", "")).lower()
    canary_expected = {
        "status": "complete",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "plan_transport_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "positive",
        "model_invoked": True,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": "model_bounded_generation",
    }
    canary_observed = {key: canary.get(key) for key in canary_expected}
    canary_quarantined = "quarantin" in str(canary.get("honest_verdict", "")).lower()
    return [
        gate_row(
            "executor_fixture_ready",
            "exp7344-executor-fixture",
            "executor_fixture_ready_score",
            {**fixture_expected, "quarantined": False},
            {**fixture_observed, "quarantined": fixture_quarantined},
            fixture_observed == fixture_expected and not fixture_quarantined,
            "The public panel must come from the current qualified executor fixture.",
        ),
        gate_row(
            "plan_canary_ready",
            "exp7347-plan-canary",
            "plan_transport_ready_score",
            {**canary_expected, "quarantined": False},
            {**canary_observed, "quarantined": canary_quarantined},
            canary_observed == canary_expected and not canary_quarantined,
            "The same local model and owned CUDA transport must pass before batch capture.",
        ),
    ]


def build_schedule(public_manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze two calls for every original request and renamed twin."""

    panel = public_manifest.get("live_proposal_panel")
    if not isinstance(panel, list) or len(panel) != 32:
        raise ValueError("live_proposal_panel_count")
    schedule: list[JsonDict] = []
    seen_pairs: set[str] = set()
    for pair_index, pair_value in enumerate(panel):
        if not isinstance(pair_value, Mapping):
            raise ValueError("live_proposal_pair")
        pair = dict(pair_value)
        panel_id = str(pair.get("panel_id", ""))
        if not panel_id or panel_id in seen_pairs:
            raise ValueError("panel_identity")
        seen_pairs.add(panel_id)
        order = (
            ("original", "twin")
            if pair.get("presentation_order") == "original_first"
            else ("twin", "original")
        )
        rename = dict(pair.get("renaming_map") or {})
        for candidate_index in range(2):
            for side in order:
                request_value = pair.get(side)
                if not isinstance(request_value, Mapping):
                    raise ValueError("public_request")
                public_request = deepcopy(dict(request_value))
                validate_public_request(public_request)
                prompt = canary_mod.render_public_prompt(public_request)
                call_index = len(schedule)
                request_id = str(public_request["request_id"])
                row: JsonDict = {
                    "call_index": call_index,
                    "call_id": f"capture-{call_index:03d}",
                    "pair_index": pair_index,
                    "panel_id": panel_id,
                    "stream_id": pair.get("stream_id"),
                    "cohort": pair.get("cohort"),
                    "warmup": pair.get("warmup"),
                    "presentation_order": pair.get("presentation_order"),
                    "pair_side": side,
                    "candidate_index": candidate_index,
                    "request_id": request_id,
                    "public_request": public_request,
                    "public_request_sha256": sha256_json(public_request),
                    "renaming_map": rename,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "seed": RANDOM_SEED["development"] + call_index,
                    "max_generated_tokens": MAX_GENERATED_TOKENS,
                }
                schedule.append(row)
    if len(schedule) != PLANNED_CALLS:
        raise ValueError("scheduled_call_count")
    return schedule


def schedule_errors(
    schedule: Sequence[Mapping[str, Any]], public_manifest: Mapping[str, Any]
) -> list[str]:
    """Rebuild the frozen schedule and name any byte or budget drift."""

    try:
        expected = build_schedule(public_manifest)
    except (KeyError, TypeError, ValueError) as error:
        return [f"schedule_source_invalid:{type(error).__name__}:{error}"]
    errors: list[str] = []
    if list(schedule) != expected:
        errors.append("schedule_rebuild_mismatch")
    if len(schedule) != PLANNED_CALLS:
        errors.append("scheduled_call_count")
    if len({str(row.get("call_id")) for row in schedule}) != len(schedule):
        errors.append("call_identity")
    if any(row.get("max_generated_tokens") != MAX_GENERATED_TOKENS for row in schedule):
        errors.append("token_budget")
    counts = Counter(str(row.get("request_id")) for row in schedule)
    if len(counts) != 64 or set(counts.values()) != {2}:
        errors.append("request_candidate_allocation")
    return errors


def build_call_row(
    *,
    schedule_row: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> JsonDict:
    """Retain one raw response under its frozen schedule identity."""

    request = dict(schedule_row["public_request"])
    raw_reply = str(response.get("raw_reply") or "")
    parsed = canary_mod.decode_public_plan(raw_reply, request)
    error_text = response.get("error")
    terminal_state = "request_error" if error_text else "response"
    return {
        **deepcopy(dict(schedule_row)),
        "raw_reply": raw_reply,
        "raw_reply_sha256": sha256_text(raw_reply),
        "raw_response": deepcopy(dict(response.get("raw_response") or {})),
        "parse_status": parsed["parse_status"],
        "parse_errors": deepcopy(parsed["parse_errors"]),
        "decoded_plan": deepcopy(parsed["plan"]),
        "attempted": True,
        "terminal_state": terminal_state,
        "error": error_text,
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "finish_reason": response.get("finish_reason"),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": terminal_state != "response",
    }


def censored_call_row(
    schedule_row: Mapping[str, Any], runtime_identity: Mapping[str, Any], reason: str
) -> JsonDict:
    """Give one unstarted scheduled call an explicit terminal disposition."""

    return {
        **deepcopy(dict(schedule_row)),
        "raw_reply": "",
        "raw_reply_sha256": sha256_text(""),
        "raw_response": {},
        "parse_status": "invalid",
        "parse_errors": ["cancelled_before_attempt"],
        "decoded_plan": None,
        "attempted": False,
        "terminal_state": "cancelled",
        "error": reason,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_s": 0.0,
        "finish_reason": None,
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": True,
    }


def _raw_source_fidelity(row: Mapping[str, Any]) -> JsonDict:
    """Classify public-field fidelity without consulting hidden rules."""

    request = dict(row.get("public_request") or {})
    activities = list(request.get("activities") or [])
    try:
        value = json.loads(str(row.get("raw_reply") or ""))
    except (json.JSONDecodeError, TypeError):
        value = None
    object_value = value if isinstance(value, dict) else {}
    assignments_value = object_value.get("assignments")
    assignments = assignments_value if isinstance(assignments_value, dict) else {}
    identity = object_value.get("request_id") == request.get("request_id")
    assignment_fields = set(assignments) == set(activities)
    typed = assignment_fields and all(
        isinstance(assignments[name], int) and not isinstance(assignments[name], bool)
        for name in activities
    )
    window = typed and all(
        assignments[name] in request["allowed_starts"][name] for name in activities
    )
    duration = typed and all(
        assignments[name] + int(request["durations"][name]) <= int(request["horizon"])
        for name in activities
    )
    weight = (
        typed
        and set(request.get("weights", {})) == set(activities)
        and all(
            isinstance(request["weights"][name], int) and request["weights"][name] > 0
            for name in activities
        )
    )
    return {
        "identity_fidelity": identity,
        "assignment_field_fidelity": assignment_fields,
        "assignment_type_fidelity": typed,
        "window_fidelity": window,
        "duration_fidelity": duration,
        "weight_fidelity": weight,
    }


def _mapped_twin_assignments(row: Mapping[str, Any]) -> JsonDict | None:
    """Map a valid twin plan back to original identifiers for paired comparison."""

    plan = row.get("decoded_plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("assignments"), Mapping):
        return None
    assignments = dict(plan["assignments"])
    if row.get("pair_side") == "original":
        return assignments
    inverse = {str(value): str(key) for key, value in dict(row.get("renaming_map") or {}).items()}
    return {inverse.get(str(name), str(name)): value for name, value in assignments.items()}


def reduce_calls(
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    evaluator_rows: Sequence[Mapping[str, Any]],
    *,
    load_receipt: Mapping[str, Any],
    model_load_duration_s: float,
) -> JsonDict:
    """Reduce fixed-denominator capture, fidelity, paired behavior, and cost."""

    schedule_by_id = {str(row.get("call_id")): row for row in schedule}
    rows_by_id = {str(row.get("call_id")): row for row in rows}
    evaluator_by_id = {str(row.get("call_id")): row for row in evaluator_rows}
    terminal_states = {"response", "request_error", "cancelled"}
    row_identity_sound = len(rows) == len(rows_by_id) == len(schedule) == PLANNED_CALLS
    for call_id, schedule_row in schedule_by_id.items():
        row = rows_by_id.get(call_id)
        row_identity_sound = bool(
            row_identity_sound
            and isinstance(row, Mapping)
            and row.get("call_index") == schedule_row.get("call_index")
            and row.get("request_id") == schedule_row.get("request_id")
            and row.get("prompt_sha256") == schedule_row.get("prompt_sha256")
            and row.get("public_request_sha256") == schedule_row.get("public_request_sha256")
            and row.get("max_generated_tokens") == MAX_GENERATED_TOKENS
            and row.get("terminal_state") in terminal_states
            and row.get("raw_reply_sha256") == sha256_text(str(row.get("raw_reply") or ""))
        )

    source_rows: list[JsonDict] = []
    cost_rows: list[JsonDict] = []
    load_share = float(model_load_duration_s) / PLANNED_CALLS
    for row_value in rows:
        row = dict(row_value)
        evaluation = dict(evaluator_by_id.get(str(row.get("call_id"))) or {})
        fidelity = _raw_source_fidelity(row)
        source_rows.append(
            {
                "call_id": row.get("call_id"),
                "panel_id": row.get("panel_id"),
                "candidate_index": row.get("candidate_index"),
                "pair_side": row.get("pair_side"),
                "request_id": row.get("request_id"),
                "parse_status": row.get("parse_status"),
                "parse_errors": deepcopy(row.get("parse_errors")),
                **fidelity,
                "hidden_rule_evaluated": evaluation.get("evaluated") is True,
                "hidden_rule_accepted": evaluation.get("hidden_rule_accepted"),
                "source_failure": row.get("parse_status") != "valid",
                "censored": row.get("censored") is True,
            }
        )
        prompt_tokens = int(row.get("prompt_tokens", 0) or 0)
        completion_tokens = int(row.get("completion_tokens", 0) or 0)
        latency = float(row.get("latency_s", 0.0) or 0.0)
        cost_rows.append(
            {
                "call_id": row.get("call_id"),
                "attempted": row.get("attempted") is True,
                "censored": row.get("censored") is True,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "generation_duration_s": latency,
                "allocated_model_load_s": load_share,
                "cold_accounting_duration_s": load_share + latency,
                "warm_accounting_duration_s": latency,
                "current_model_generation_calls": int(row.get("attempted") is True),
                "downstream_cpu_replay_model_calls": 0,
            }
        )

    pair_groups: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("panel_id")), int(row.get("candidate_index", -1)))
        pair_groups[key][str(row.get("pair_side"))] = row
    pair_rows: list[JsonDict] = []
    for (panel_id, candidate_index), group in sorted(pair_groups.items()):
        original = group.get("original")
        twin = group.get("twin")
        original_plan = _mapped_twin_assignments(original or {})
        twin_plan = _mapped_twin_assignments(twin or {})
        both_valid = bool(
            original is not None
            and twin is not None
            and original.get("parse_status") == "valid"
            and twin.get("parse_status") == "valid"
        )
        pair_rows.append(
            {
                "panel_id": panel_id,
                "candidate_index": candidate_index,
                "paired_observation_count": 1,
                "original_call_id": original.get("call_id") if original else None,
                "twin_call_id": twin.get("call_id") if twin else None,
                "both_source_valid": both_valid,
                "renamed_assignments_match": (original_plan == twin_plan if both_valid else None),
                "original_hidden_rule_accepted": evaluator_by_id.get(
                    str(original.get("call_id")) if original else "", {}
                ).get("hidden_rule_accepted"),
                "twin_hidden_rule_accepted": evaluator_by_id.get(
                    str(twin.get("call_id")) if twin else "", {}
                ).get("hidden_rule_accepted"),
            }
        )

    attempted = [row for row in rows if row.get("attempted") is True]
    completed = [row for row in attempted if row.get("terminal_state") == "response"]
    failed = [row for row in attempted if row.get("terminal_state") == "request_error"]
    cancelled = [row for row in rows if row.get("terminal_state") == "cancelled"]
    censored = [row for row in rows if row.get("censored") is True]
    usable = [row for row in rows if row.get("parse_status") == "valid"]
    source_failures = [
        row
        for row in rows
        if row.get("terminal_state") == "response" and row.get("parse_status") != "valid"
    ]
    hidden_failures = [row for row in evaluator_rows if row.get("hidden_rule_accepted") is False]
    invocation_counts = {
        "model_loads_attempted": int(load_receipt.get("attempted") is True),
        "model_loads_completed": int(load_receipt.get("completed") is True),
        "model_loads_failed": int(load_receipt.get("failed") is True),
        "model_loads_cancelled": int(load_receipt.get("cancelled") is True),
        "model_loads_in_flight": int(load_receipt.get("in_flight") is True),
        "generation_calls_attempted": len(attempted),
        "generation_calls_completed": len(completed),
        "generation_calls_failed": len(failed),
        "generation_calls_cancelled": len(cancelled),
        "generation_calls_in_flight": 0,
    }
    budget = {
        "planned_units": PLANNED_CALLS,
        "attempted_units": len(attempted),
        "completed_units": len(completed),
        "failed_units": len(failed),
        "cancelled_units": len(cancelled),
        "censored_units": len(censored),
    }
    return {
        "plan_capture_complete_score": int(row_identity_sound),
        "usable_candidate_count": len(usable),
        "source_parse_failure_count": len(source_failures),
        "hidden_rule_failure_count": len(hidden_failures),
        "source_fidelity_rows": source_rows,
        "renamed_pair_rows": pair_rows,
        "generation_cost_rows": cost_rows,
        "invocation_counts": invocation_counts,
        "sample_size_budget": budget,
    }


def checkpoint_identity(
    schedule: Sequence[Mapping[str, Any]], source_manifest_sha256: str, model_sha256: str
) -> JsonDict:
    """Bind resume state to schedule, source bytes, model bytes, and decoding limits."""

    return {
        "schedule_sha256": sha256_json(list(schedule)),
        "source_manifest_sha256": source_manifest_sha256,
        "model_sha256": model_sha256,
        "max_generated_tokens": MAX_GENERATED_TOKENS,
        "generation_timeout_s": GENERATION_TIMEOUT_S,
    }


def write_checkpoint(
    path: Path, identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Atomically persist each captured row with a byte-level row hash."""

    payload_rows = [deepcopy(dict(row)) for row in rows]
    _atomic_json(
        path,
        {
            "schema": "carnot.exp7348.checkpoint.v1",
            "identity": deepcopy(dict(identity)),
            "rows": payload_rows,
            "row_hashes": [sha256_json(row) for row in payload_rows],
        },
    )


def resume_checkpoint(path: Path, expected_identity: Mapping[str, Any]) -> list[JsonDict]:
    """Resume only when frozen identity and every completed row still match."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint_identity")
    rows = value.get("rows")
    hashes = value.get("row_hashes")
    if not isinstance(rows, list) or not isinstance(hashes, list) or len(rows) != len(hashes):
        raise ValueError("checkpoint_rows")
    if [sha256_json(row) for row in rows] != hashes:
        raise ValueError("checkpoint_row_hash")
    return [deepcopy(dict(row)) for row in rows]


def classify_terminal(
    *,
    complete_score: int,
    usable_candidate_count: int,
    required_checks_passed: bool,
    flagged_adversarial: bool,
) -> JsonDict:
    """Keep accounting completion separate from usefulness and promotion."""

    if not required_checks_passed or flagged_adversarial:
        return {
            "status": "complete",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_plan_capture_required_check_failed",
            "plan_capture_complete_score": 0,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    if complete_score != 1:
        return {
            "status": "complete",
            "verdict_class": "null",
            "honest_verdict": "complete_null_plan_capture_incomplete_accounting",
            "plan_capture_complete_score": 0,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    if usable_candidate_count == 0:
        return {
            "status": "complete",
            "verdict_class": "null",
            "honest_verdict": "complete_null_plan_capture_zero_usable_candidates",
            "plan_capture_complete_score": 1,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    return {
        "status": "complete",
        "verdict_class": "circular_positive",
        "honest_verdict": f"complete_circular_positive_plan_capture_{usable_candidate_count}_usable_candidates",
        "plan_capture_complete_score": 1,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
    }


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reload raw rows and rebuild every count, pair, cost, and fidelity result."""

    manifest = artifact.get("raw_call_manifest")
    if not isinstance(manifest, Mapping):
        return ["raw_call_manifest_unavailable"]
    schedule = manifest.get("schedule")
    calls = manifest.get("calls")
    if not isinstance(schedule, list) or not isinstance(calls, list):
        return ["raw_call_rows_unavailable"]
    errors: list[str] = []
    if artifact.get("schedule") != schedule:
        errors.append("schedule_manifest_mismatch")
    if artifact.get("rows") != calls:
        errors.append("rows_manifest_mismatch")
    reduced = reduce_calls(
        schedule,
        calls,
        list(artifact.get("evaluator_rows") or []),
        load_receipt=dict(artifact.get("load_receipt") or {}),
        model_load_duration_s=float(artifact.get("model_load_duration_s", 0.0) or 0.0),
    )
    for field in (
        "plan_capture_complete_score",
        "usable_candidate_count",
        "source_parse_failure_count",
        "hidden_rule_failure_count",
        "source_fidelity_rows",
        "renamed_pair_rows",
        "generation_cost_rows",
        "invocation_counts",
        "sample_size_budget",
    ):
        if artifact.get(field) != reduced[field]:
            errors.append(f"{field}_mismatch")
    return errors


def artifact_checksum(artifact: Mapping[str, Any]) -> str:  # pragma: no cover
    """Hash terminal evidence without hashing the checksum into itself."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one complete JSON object by replacing a local temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_object(path: Path) -> JsonDict:  # pragma: no cover
    """Read one required object and reject a non-object before dependent work."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_object:{path}")
    return value


def _phase_close(
    spans: list[JsonDict], name: str, started: float, units: int, checkpoint: str | None = None
) -> None:  # pragma: no cover
    """Close one measured phase and retain its durable boundary."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_monotonic_s": started,
            "end_monotonic_s": ended,
            "duration_s": ended - started,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
    )


def _base_artifact(run_date: str, started_at: str) -> JsonDict:  # pragma: no cover
    """Create every required field before an external check can fail."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": {
            "model_loads_attempted": 0,
            "model_loads_completed": 0,
            "model_loads_failed": 0,
            "model_loads_cancelled": 0,
            "model_loads_in_flight": 0,
            "generation_calls_attempted": 0,
            "generation_calls_completed": 0,
            "generation_calls_failed": 0,
            "generation_calls_cancelled": 0,
            "generation_calls_in_flight": 0,
        },
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "rows": [],
        "schedule": [],
        "sample_size_budget": {
            "planned_units": PLANNED_CALLS,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "cancelled_units": PLANNED_CALLS,
            "censored_units": PLANNED_CALLS,
            "max_generated_tokens_per_unit": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "stopping_rule": "128 fixed calls or the fixed generation deadline; no replacement or output-conditioned retry",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_not_started",
        "verdict_class": "blocked",
        "flagged_adversarial": True,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "historical_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "plan_capture_complete_score": 0,
        "usable_candidate_count": 0,
        "source_parse_failure_count": 0,
        "hidden_rule_failure_count": 0,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
        "candidate_manifest_path": str(REPO_ROOT / CANDIDATE_MANIFEST_PATH),
        "raw_call_manifest": {"schedule": [], "calls": []},
        "evaluator_rows": [],
        "source_fidelity_rows": [],
        "renamed_pair_rows": [],
        "generation_cost_rows": [],
        "load_receipt": {
            "attempted": False,
            "completed": False,
            "failed": False,
            "cancelled": False,
            "in_flight": False,
            "error": None,
        },
        "model_load_duration_s": 0.0,
        "runtime_identity_receipt": {},
        "gpu_receipts": {},
        "evaluation_receipt": {},
    }


def _collect_preconditions(
    root: Path, run_date: str
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Authenticate exact producers, source bytes, model, runner, and CUDA capacity."""

    progress("preconditions", "start")
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    checks: list[JsonDict] = []
    context: JsonDict = {}
    checks.append(
        gate_row(
            "run_date",
            "execution_contract",
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            "The capture is tied to the V645 execution date.",
        )
    )
    required = {
        "fixture": root / FIXTURE_PATH,
        "canary": root / CANARY_PATH,
        "public_manifest": PUBLIC_MANIFEST_PATH,
        "module": root / MODULE_PATH,
        "canary_module": root / CANARY_MODULE_PATH,
        "entrypoint": root / WRAPPER_PATH,
        "tests": root / TEST_PATH,
        "canary_tests": root / CANARY_TEST_PATH,
        "spec": root / SPEC_PATH,
        "private_executor": root / PRIVATE_EXECUTOR_PATH,
        "validation_runner": root / "python/carnot/reporting/experiment_7303_validation_scope.py",
        "e2e_plan": root / "ops/e2e-test-plan.md",
        "exclusion_manifest": root / "ops/exclusion_manifest.yaml",
        "research_program": root / "research-program.md",
        "research_references": root / "research-references.md",
    }
    observed_paths = {name: path.is_file() for name, path in required.items()}
    observed_paths["spec_has_req"] = bool(
        required["spec"].is_file() and "REQ-CL-7348" in required["spec"].read_text(encoding="utf-8")
    )
    checks.append(
        gate_row(
            "required_source_paths",
            "repository",
            "preconditions_checked",
            {name: True for name in observed_paths},
            observed_paths,
            all(observed_paths.values()),
            "All exact inputs and the driving requirement must exist before model work.",
        )
    )
    fixture: JsonDict = {}
    canary: JsonDict = {}
    public_manifest: JsonDict = {}
    if all(required[name].is_file() for name in ("fixture", "canary", "public_manifest")):
        fixture = _load_object(required["fixture"])
        canary = _load_object(required["canary"])
        public_manifest = _load_object(required["public_manifest"])
    checks.extend(dependency_gate_rows(fixture, canary))
    declared_fixture_hash = dict(canary.get("source_artifact_hashes") or {}).get(str(FIXTURE_PATH))
    observed_fixture_hash = (
        sha256_file(required["fixture"]) if required["fixture"].is_file() else None
    )
    checks.append(
        gate_row(
            "canary_fixture_bytes",
            "exp7347-plan-canary",
            "source_artifact_hashes",
            declared_fixture_hash,
            observed_fixture_hash,
            bool(declared_fixture_hash and declared_fixture_hash == observed_fixture_hash),
            "The canary and capture must use the same exact fixture artifact.",
        )
    )
    declared_public_hash = dict(fixture.get("source_artifact_hashes") or {}).get(
        str(required["public_manifest"])
    )
    observed_public_hash = (
        sha256_file(required["public_manifest"]) if required["public_manifest"].is_file() else None
    )
    checks.append(
        gate_row(
            "fixture_public_manifest_bytes",
            "exp7344-executor-fixture",
            "source_artifact_hashes",
            declared_public_hash,
            observed_public_hash,
            bool(declared_public_hash and declared_public_hash == observed_public_hash),
            "Generation must use the exact public bytes sealed by the fixture.",
        )
    )
    schedule: list[JsonDict] = []
    schedule_issue: str | None = None
    try:
        schedule = build_schedule(public_manifest)
    except (KeyError, TypeError, ValueError) as error:
        schedule_issue = f"{type(error).__name__}:{error}"
    checks.append(
        gate_row(
            "frozen_128_call_schedule",
            "public_manifest",
            "sample_size_budget",
            {"scheduled_calls": PLANNED_CALLS, "errors": []},
            {
                "scheduled_calls": len(schedule),
                "errors": [schedule_issue]
                if schedule_issue
                else schedule_errors(schedule, public_manifest),
            },
            len(schedule) == PLANNED_CALLS
            and schedule_issue is None
            and not schedule_errors(schedule, public_manifest),
            "All requests, twins, candidates, prompts, seeds, and token budgets freeze before generation.",
        )
    )
    private_path_value = dict(fixture.get("raw_evidence_paths") or {}).get("private_manifest")
    private_path = Path(str(private_path_value)) if private_path_value else Path("/missing")
    declared_private_hash = dict(fixture.get("source_artifact_hashes") or {}).get(str(private_path))
    observed_private_hash = sha256_file(private_path) if private_path.is_file() else None
    checks.append(
        gate_row(
            "private_evaluator_bytes_sealed",
            "exp7344-executor-fixture",
            "source_artifact_hashes",
            declared_private_hash,
            observed_private_hash,
            bool(declared_private_hash and declared_private_hash == observed_private_hash),
            "Private authority is authenticated now but remains unopened until raw replies are sealed.",
        )
    )

    progress("preconditions", "before_runtime_preflight")
    runtime_checks, runtime_context = canary_mod._collect_preconditions(root, run_date)
    progress("preconditions", "after_runtime_preflight")
    checks.extend(
        gate_row(
            f"canary_runtime_{row['check']}",
            str(row.get("upstream")),
            str(row.get("artifact_field")),
            row.get("expected_value"),
            row.get("observed_value"),
            row.get("passed") is True,
            str(row.get("principle")),
        )
        for row in runtime_checks
        if row.get("check")
        in {
            "cached_current_model",
            "native_cuda_runner",
            "cuda_inventory_and_owned_capacity",
            "force_live_mode",
        }
    )
    model_spec = dict(runtime_context.get("model_spec") or {})
    canary_model_spec = next(iter(canary.get("model_specs") or []), {})
    same_model = all(
        model_spec.get(field) == canary_model_spec.get(field)
        for field in ("hf_id", "quantization", "bytes", "sha256")
    )
    checks.append(
        gate_row(
            "same_verified_model_identity",
            "exp7347-plan-canary",
            "MODEL_SPECS",
            {
                field: canary_model_spec.get(field)
                for field in ("hf_id", "quantization", "bytes", "sha256")
            },
            {
                field: model_spec.get(field)
                for field in ("hf_id", "quantization", "bytes", "sha256")
            },
            same_model,
            "The batch runner must serve the same verified current model bytes as the canary.",
        )
    )
    context.update(runtime_context)
    context.update(
        {
            "fixture": fixture,
            "canary": canary,
            "public_manifest": public_manifest,
            "private_manifest_path": private_path,
            "schedule": schedule,
        }
    )
    progress("preconditions", "complete", passed=all(row.get("passed") is True for row in checks))
    return checks, context


@contextmanager
def _batch_canary_settings() -> Iterator[None]:  # pragma: no cover
    """Temporarily apply this experiment's fixed limits to the reused runner."""

    names = {
        "TASK_ID": TASK_ID,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "INFERENCE_WINDOW_TIMEOUT_S": GENERATION_TIMEOUT_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "RANDOM_SEED": RANDOM_SEED,
    }
    previous = {name: getattr(canary_mod, name) for name in names}
    try:
        for name, value in names.items():
            setattr(canary_mod, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(canary_mod, name, value)


def _capture_schedule(context: Mapping[str, Any], raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Reuse the owned CUDA runner and bind each reply to its frozen call row."""

    schedule = [deepcopy(dict(row)) for row in context["schedule"]]
    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = [deepcopy(row["public_request"]) for row in schedule]
    with _batch_canary_settings():
        capture = canary_mod._live_capture(runtime_context, raw_dir / "calls")
    runtime_rows = list(capture.get("rows") or [])
    rows: list[JsonDict] = []
    for index, schedule_row in enumerate(schedule):
        if index < len(runtime_rows):
            runtime_row = dict(runtime_rows[index])
            response = {
                "raw_reply": runtime_row.get("raw_reply"),
                "raw_response": runtime_row.get("raw_response"),
                "prompt_tokens": runtime_row.get("prompt_tokens"),
                "completion_tokens": runtime_row.get("completion_tokens"),
                "latency_s": runtime_row.get("latency_s"),
                "finish_reason": runtime_row.get("finish_reason"),
                "error": runtime_row.get("error"),
            }
            if runtime_row.get("terminal_state") == "cancelled":
                row = censored_call_row(
                    schedule_row,
                    dict(runtime_row.get("runtime_identity_receipt") or {}),
                    str(runtime_row.get("error") or "generation_deadline"),
                )
            else:
                row = build_call_row(
                    schedule_row=schedule_row,
                    response=response,
                    runtime_identity=dict(runtime_row.get("runtime_identity_receipt") or {}),
                )
        else:
            row = censored_call_row(
                schedule_row,
                dict(capture.get("runtime_identity") or {}),
                str(capture.get("runtime_error") or "generation_deadline"),
            )
        rows.append(row)
        _atomic_json(raw_dir / "calls" / f"call_{index:03d}.json", row)
        identity = checkpoint_identity(
            schedule,
            sha256_file(PUBLIC_MANIFEST_PATH),
            str(dict(context["model_spec"])["sha256"]),
        )
        write_checkpoint(REPO_ROOT / CHECKPOINT_PATH, identity, rows)
        progress(
            "generation",
            "checkpoint",
            completed=index + 1,
            total=PLANNED_CALLS,
            elapsed_s=round(sum(float(item.get("latency_s", 0.0) or 0.0) for item in rows), 3),
        )
    capture["rows"] = rows
    return capture


def _score_sealed_calls(
    raw_manifest_path: Path,
    public_manifest_path: Path,
    private_manifest_path: Path,
    evaluator: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], bool]
    | None = None,
) -> JsonDict:  # pragma: no cover
    """Open private rules only after the raw manifest exists and has a stable hash."""

    if not raw_manifest_path.is_file():
        raise ValueError("raw_manifest_not_sealed")
    raw_sha256 = sha256_file(raw_manifest_path)
    raw = _load_object(raw_manifest_path)
    public = _load_object(public_manifest_path)
    private = _load_object(private_manifest_path)
    if private.get("public_manifest_sha256") != sha256_file(public_manifest_path):
        raise ValueError("private_public_manifest_mismatch")
    if evaluator is None:
        namespace = runpy.run_path(
            str(REPO_ROOT / PRIVATE_EXECUTOR_PATH), run_name="exp7348_oracle"
        )
        checker = namespace.get("_check")
        if not callable(checker):
            raise ValueError("private_executor_missing_check")
        evaluator = checker
    public_index: dict[str, JsonDict] = {}
    for pair in public["live_proposal_panel"]:
        for side in ("original", "twin"):
            request = deepcopy(dict(pair[side]))
            public_index[str(request["request_id"])] = request
    private_index = dict(private["evaluator_records"])
    evaluation_rows: list[JsonDict] = []
    for row in raw["calls"]:
        call_id = str(row["call_id"])
        request_id = str(row["request_id"])
        if row.get("parse_status") != "valid" or not isinstance(row.get("decoded_plan"), Mapping):
            evaluation_rows.append(
                {
                    "call_id": call_id,
                    "request_id": request_id,
                    "evaluated": False,
                    "hidden_rule_accepted": None,
                    "reason": "source_parse_failure_or_censoring",
                }
            )
            continue
        request = public_index[request_id]
        private_record = dict(private_index[request_id])
        accepted = evaluator(
            request, dict(row["decoded_plan"]), dict(private_record["private_rules"])
        )
        evaluation_rows.append(
            {
                "call_id": call_id,
                "request_id": request_id,
                "evaluated": True,
                "hidden_rule_accepted": bool(accepted),
                "reason": "qualified_private_executor",
            }
        )
    return {
        "schema": "carnot.exp7348.private_evaluation.v1",
        "raw_manifest_path": str(raw_manifest_path),
        "raw_manifest_sha256_before_private_open": raw_sha256,
        "public_manifest_sha256": sha256_file(public_manifest_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "private_executor_path": str(REPO_ROOT / PRIVATE_EXECUTOR_PATH),
        "private_executor_sha256": sha256_file(REPO_ROOT / PRIVATE_EXECUTOR_PATH),
        "verifier_is_oracle": True,
        "rows": evaluation_rows,
        "evaluated_call_count": sum(row["evaluated"] is True for row in evaluation_rows),
    }


def _source_hashes(
    root: Path, context: Mapping[str, Any], raw_manifest: Path, evaluation_path: Path
) -> JsonDict:  # pragma: no cover
    """Authenticate exact producers, code, model, evaluator, and raw evidence."""

    paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        MODULE_PATH,
        CANARY_MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        CANARY_TEST_PATH,
        FIXTURE_PATH,
        CANARY_PATH,
        PRIVATE_EXECUTOR_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    )
    hashes = {str(path): sha256_file(root / path) for path in paths}
    hashes[str(PUBLIC_MANIFEST_PATH)] = sha256_file(PUBLIC_MANIFEST_PATH)
    private_path = Path(str(context["private_manifest_path"]))
    hashes[str(private_path)] = sha256_file(private_path)
    model_path = Path(str(context["model_path"])).resolve()
    hashes[str(model_path)] = str(dict(context["model_spec"])["sha256"])
    hashes[str(raw_manifest)] = sha256_file(raw_manifest)
    hashes[str(evaluation_path)] = sha256_file(evaluation_path)
    return hashes


def run_affected_validation(root: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Run the shipped explicit-scope checks with private temporary parents."""

    basetemp = Path("/tmp/carnot-exp7348-v645-scoped")
    coverage_file = Path("/tmp/carnot-exp7348-v645-coverage/.coverage")
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    return run_scoped_validation(
        root,
        test_paths=[str(TEST_PATH), str(CANARY_TEST_PATH)],
        changed_modules=[str(MODULE_PATH), str(CANARY_MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=raw_dir / "validation/scoped",
        historical_failures=[],
    )


def run_terminal_validation(
    root: Path, candidate: Path, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Cold-reduce the candidate and run both strict terminal readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib;"
        "from carnot.experiment_7348_v645_plan_capture import independent_reduce;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "e=independent_reduce(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    commands = [
        CommandSpec("independent_reducer", (python, "-u", "-c", reducer), "candidate"),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful receipt for each named current check."""

    return all(
        sum(
            row.get("name") == name and row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
        )
        == 1
        for name in names
    )


def _acceptance_gates(
    checks: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    scoped_ok: bool,
    terminal_ok: bool,
    flagged: bool,
) -> dict[str, JsonDict]:  # pragma: no cover
    """Apply completion and verification gates without turning accounting into value."""

    return {
        "preconditions": {
            "expected": True,
            "observed": all(row.get("passed") is True for row in checks),
            "passed": all(row.get("passed") is True for row in checks),
            "principle": "Every producer and runtime check must pass before model loading.",
        },
        "fixed_denominator_complete": {
            "expected": 1,
            "observed": reduced.get("plan_capture_complete_score"),
            "passed": reduced.get("plan_capture_complete_score") == 1,
            "principle": "Each scheduled call needs one authentic terminal disposition.",
        },
        "affected_validation": {
            "expected": True,
            "observed": scoped_ok,
            "passed": scoped_ok,
            "principle": "Changed modules and their exact tests must pass current scoped checks.",
        },
        "terminal_validation": {
            "expected": True,
            "observed": terminal_ok,
            "passed": terminal_ok,
            "principle": "Cold reduction and both strict terminal readers must pass.",
        },
        "adversarial_clear": {
            "expected": False,
            "observed": flagged,
            "passed": not flagged,
            "principle": "A critical adversarial finding prevents promotion.",
        },
    }


def validate_artifact(artifact: object) -> list[str]:  # pragma: no cover
    """Cold-check terminal identity, denominators, scores, and evidence hashes."""

    if not isinstance(artifact, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    errors.extend(f"missing_required_field:{field}" for field in missing)
    if errors:
        return errors
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("lifecycle_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    counts = dict(artifact.get("invocation_counts") or {})
    if artifact.get("model_invoked") is not (counts.get("model_loads_attempted", 0) > 0):
        errors.append("model_invoked_mismatch")
    if artifact.get("status") == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            errors.append("blocked_verdict_invalid")
    else:
        errors.extend(independent_reduce(artifact))
        if artifact.get("inference_substrate_class") not in {
            "model_bounded_generation",
            "model_load_no_generation",
        }:
            errors.append("complete_substrate_invalid")
        if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
            errors.append("complete_verdict_invalid")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and any(
        artifact.get(field) != 0
        for field in ("plan_capture_complete_score", "value_ready_score", "promotion_ready_score")
    ):
        errors.append("failed_scores_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _write_blocked(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    output: Path,
    started: float,
) -> JsonDict:  # pragma: no cover
    """Write one honest external block without success-shaped model evidence."""

    summary = gate_check_summary(checks)
    artifact.update(
        {
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "completed_at_utc": _utc_now(),
            "duration_s": time.monotonic() - started,
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary.get('failed_check') or 'unknown_precondition'}",
            "flagged_adversarial": True,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(output, artifact)
    return artifact


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover
    """Authenticate, capture, seal, evaluate, validate, and publish once."""

    started = time.monotonic()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact = _base_artifact(run_date, _utc_now())
    checks, context = _collect_preconditions(root, run_date)
    if any(row.get("passed") is not True for row in checks):
        progress("write", "blocked_start")
        result = _write_blocked(artifact, checks, output, started)
        progress("write", "blocked_complete", artifact=output)
        return result

    schedule = [deepcopy(dict(row)) for row in context["schedule"]]
    schedule_manifest = {
        "schema": "carnot.exp7348.schedule.v1",
        "sealed_before_generation": True,
        "schedule": schedule,
        "schedule_sha256": sha256_json(schedule),
    }
    _atomic_json(root / SCHEDULE_PATH, schedule_manifest)
    progress("generation", "before_owned_runner", planned_calls=PLANNED_CALLS)
    capture = _capture_schedule(context, raw_dir)
    progress("generation", "after_owned_runner", captured_calls=len(capture["rows"]))
    raw_manifest: JsonDict = {
        "schema": "carnot.exp7348.candidate_manifest.v1",
        "producer": EXPERIMENT_ID,
        "sealed_before_private_evaluation": True,
        "schedule": schedule,
        "calls": deepcopy(capture["rows"]),
    }
    raw_manifest["manifest_sha256"] = sha256_json(raw_manifest)
    _atomic_json(root / CANDIDATE_MANIFEST_PATH, raw_manifest)

    evaluation_started = time.monotonic()
    progress("evaluation", "before_private_evaluator", raw_manifest=root / CANDIDATE_MANIFEST_PATH)
    evaluation = _score_sealed_calls(
        root / CANDIDATE_MANIFEST_PATH,
        PUBLIC_MANIFEST_PATH,
        Path(str(context["private_manifest_path"])),
    )
    _atomic_json(root / EVALUATION_PATH, evaluation)
    progress("evaluation", "after_private_evaluator", evaluated=evaluation["evaluated_call_count"])
    _phase_close(
        capture["phase_spans"],
        "evaluation",
        evaluation_started,
        len(evaluation["rows"]),
        str(root / EVALUATION_PATH),
    )
    load_span = next(
        (row for row in capture["phase_spans"] if row.get("phase") == "model_load"), {}
    )
    load_duration = float(load_span.get("duration_s", 0.0) or 0.0)
    reduced = reduce_calls(
        schedule,
        capture["rows"],
        evaluation["rows"],
        load_receipt=dict(capture["load_receipt"]),
        model_load_duration_s=load_duration,
    )

    test_started = time.monotonic()
    progress("tests", "before_scoped_validation")
    validation = run_affected_validation(root, raw_dir)
    progress("tests", "after_scoped_validation", passed=validation.get("required_checks_passed"))
    scoped_receipts = list(validation.get("validation_receipts") or [])
    scoped_ok = bool(
        validation.get("required_checks_passed")
        and _receipts_pass(scoped_receipts, REQUIRED_CHECK_NAMES)
    )
    _phase_close(capture["phase_spans"], "tests", test_started, len(scoped_receipts))

    authentic_generation = bool(
        reduced["invocation_counts"]["generation_calls_attempted"] > 0
        and dict(capture.get("gpu_receipts") or {}).get("provenance", {}).get("provenance_ok")
        is True
    )
    substrate_class = (
        "model_bounded_generation" if authentic_generation else "model_load_no_generation"
    )
    artifact.update(
        {
            "status": "complete",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "MODEL_SPECS": deepcopy(MODEL_SPECS),
            "model_specs": [deepcopy(dict(context["model_spec"]))],
            "model_invoked": capture["load_receipt"].get("attempted") is True,
            "invocation_counts": deepcopy(reduced["invocation_counts"]),
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": substrate_class,
            "phase_spans": deepcopy(capture["phase_spans"]),
            "rows": deepcopy(capture["rows"]),
            "schedule": schedule,
            "sample_size_budget": {
                **deepcopy(reduced["sample_size_budget"]),
                "max_generated_tokens_per_unit": MAX_GENERATED_TOKENS,
                "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
                "generation_timeout_s": GENERATION_TIMEOUT_S,
                "stopping_rule": "128 fixed calls or the fixed generation deadline; no replacement or output-conditioned retry",
            },
            "candidate_manifest_path": str(root / CANDIDATE_MANIFEST_PATH),
            "raw_call_manifest": raw_manifest,
            "evaluator_rows": deepcopy(evaluation["rows"]),
            "evaluation_receipt": evaluation,
            "source_fidelity_rows": deepcopy(reduced["source_fidelity_rows"]),
            "renamed_pair_rows": deepcopy(reduced["renamed_pair_rows"]),
            "generation_cost_rows": deepcopy(reduced["generation_cost_rows"]),
            "plan_capture_complete_score": reduced["plan_capture_complete_score"],
            "usable_candidate_count": reduced["usable_candidate_count"],
            "source_parse_failure_count": reduced["source_parse_failure_count"],
            "hidden_rule_failure_count": reduced["hidden_rule_failure_count"],
            "load_receipt": deepcopy(capture["load_receipt"]),
            "model_load_duration_s": load_duration,
            "runtime_identity_receipt": deepcopy(capture["runtime_identity"]),
            "gpu_receipts": deepcopy(capture["gpu_receipts"]),
            "validation_receipts": scoped_receipts,
            "repository_health": deepcopy(validation.get("repository_health") or {}),
            "completed_at_utc": _utc_now(),
            "duration_s": time.monotonic() - started,
            "flagged_adversarial": False,
        }
    )
    artifact.update(
        classify_terminal(
            complete_score=reduced["plan_capture_complete_score"],
            usable_candidate_count=reduced["usable_candidate_count"],
            required_checks_passed=scoped_ok,
            flagged_adversarial=False,
        )
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, reduced, scoped_ok, False, False
    )
    artifact["gate_check_summary"] = gate_check_summary(checks)
    artifact["source_artifact_hashes"] = _source_hashes(
        root, context, root / CANDIDATE_MANIFEST_PATH, root / EVALUATION_PATH
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "terminal_candidate.json"
    _atomic_json(candidate, artifact)

    terminal_started = time.monotonic()
    progress("tests", "before_terminal_validation", candidate=candidate)
    terminal_receipts = run_terminal_validation(root, candidate, raw_dir)
    progress("tests", "after_terminal_validation")
    terminal_ok = _receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    artifact["validation_receipts"] = [*scoped_receipts, *terminal_receipts]
    _phase_close(
        artifact["phase_spans"],
        "terminal_validation",
        terminal_started,
        len(terminal_receipts),
    )
    independent_ok = not independent_reduce(artifact)
    adversarial_receipt = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = not bool(adversarial_receipt.get("passed"))
    artifact.update(
        classify_terminal(
            complete_score=reduced["plan_capture_complete_score"],
            usable_candidate_count=reduced["usable_candidate_count"],
            required_checks_passed=scoped_ok and terminal_ok and independent_ok,
            flagged_adversarial=flagged,
        )
    )
    artifact["flagged_adversarial"] = flagged
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, reduced, scoped_ok, terminal_ok and independent_ok, flagged
    )
    gate_rows = [
        gate_row(
            name,
            EXPERIMENT_ID,
            "acceptance_gate_results",
            row["expected"],
            row["observed"],
            row["passed"],
            row["principle"],
        )
        for name, row in artifact["acceptance_gate_results"].items()
    ]
    artifact["gate_check_summary"] = gate_check_summary([*checks, *gate_rows])
    write_started = time.monotonic()
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = time.monotonic() - started
    _phase_close(artifact["phase_spans"], "write", write_started, 1, str(output))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        artifact.update(
            {
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "plan_capture_complete_score": 0,
                "value_ready_score": 0,
                "promotion_ready_score": 0,
                "internal_validation_errors": validation_errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress("write", "start", artifact=output)
    _atomic_json(output, artifact)
    progress("write", "complete", artifact=output, verdict=artifact["honest_verdict"])
    return artifact


def _date_argument(value: str) -> str:  # pragma: no cover
    """Reject execution outside the fixed V645 date."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Run the live capture or cold-check one task-owned candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    progress("entrypoint", "start", date=args.date)
    if args.validate is not None:
        value = _load_object(args.validate)
        errors = validate_artifact(value)
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "plan_capture_complete_score": result["plan_capture_complete_score"],
                "usable_candidate_count": result["usable_candidate_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
