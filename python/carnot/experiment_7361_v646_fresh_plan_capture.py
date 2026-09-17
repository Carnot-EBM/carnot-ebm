"""Capture fresh Qwen plans while keeping public fidelity separate.

The module reuses the owned GGUF runtime, sealed private evaluator, and tested
schedule reducer. It adds only the V646 schedule, public-source evidence, and
terminal record assembly needed by this experiment.

Spec refs: REQ-REPORT-7361 and SCENARIO-REPORT-7361-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as canary_mod
from carnot import experiment_7348_v645_plan_capture as prior_capture
from carnot import experiment_7359_v646_capture_reducer as capture_reducer
from carnot.experiment_7330_v644_public_learner import canonical_bytes, sha256_json
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7361-fresh-plan-capture"
SCHEMA = "carnot.exp7361.v646_fresh_plan_capture.v1"
TASK_ID = "experiment_7361_v646_fresh_plan_capture"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]

REDUCER_PATH = Path("results/experiment_7359_v646_capture_reducer.json")
FIXTURE_PATH = Path("results/experiment_7360_v646_learning_fixture.json")
PUBLIC_MANIFEST_PATH = Path(
    "results/raw/experiment_7360_v646_learning_fixture/public/public_manifest.json"
)
PRIVATE_MANIFEST_PATH = Path(
    "results/raw/experiment_7360_v646_learning_fixture/evaluator/evaluator_private_manifest.json"
)
MODULE_PATH = Path("python/carnot/experiment_7361_v646_fresh_plan_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7361_v646_fresh_plan_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7361_v646_fresh_plan_capture.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7361_v646_fresh_plan_capture.json")
RAW_DIR = Path("results/raw/experiment_7361_v646_fresh_plan_capture")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7361_v646_fresh_plan_capture.json")

RANDOM_SEED = {"development": 7_360_607, "evaluation": 7_360_211, "resampling": 7_360_307}
MAX_GENERATED_TOKENS = 256
CANARY_CALLS = 4
EVALUATION_CALLS = 128
MODEL_LOAD_TIMEOUT_S = 600.0
CANARY_TIMEOUT_S = 600.0
GENERATION_TIMEOUT_S = 1_800.0
REQUEST_TIMEOUT_S = 300.0
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Version this record and retain ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and affected validation; never a success-shaped placeholder.",
    "run_date": "Use 20260917 and actual UTC timestamps.",
    "preconditions_checked": "Exact input, resource and required-field checks before dependent work.",
    "MODEL_SPECS": "Actual intended identities; every LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, even failure.",
    "invocation_counts": "Attempted/completed/failed/cancelled/in-flight calls; separate current from historical.",
    "inference_substrate": "Actual computation, with historical inference in explicitly labeled hash-bound sidecars.",
    "inference_substrate_class": "Actual closed duration class; no duration padding.",
    "execution_venue": "Host CPU or owned CUDA runtime as measured; no new board execution in V646.",
    "duration_s": "Measured monotonic elapsed, never synthetic elapsed or sleep to pass a floor.",
    "phase_spans": "Disjoint measured load, generation, evaluation, validation and write spans.",
    "random_seed": "Frozen development/evaluation/resampling seeds; null if truly inapplicable.",
    "reproducibility_checksum": "Bind exact code, settings, evaluator, inputs and raw evidence.",
    "source_artifact_hashes": "Exact producer paths and immutable byte hashes; preserve original classes/flags.",
    "rows": "Every comparative unit/arm/metric/cost/failure/censoring disposition, not only pooled means.",
    "sample_size_budget": "Frozen planned/attempted/completed/censored units and stopping rules.",
    "acceptance_gate_results": "Expected, observed and passed separately for required validation, safety and scientific value; never mark failed value as successful.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, expected and observed value, including missing paths.",
    "verifier_is_oracle": "True when the evaluator defines truth; independent code alone cannot remove circularity.",
    "honest_verdict": "Precise free-text terminal outcome; distinguish accounting, null science and unavailable work.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished retryable OWN work is partial; external unchanged absence is blocked.",
    "flagged_adversarial": "Current independent verification state; critical findings set true and prevent promotion.",
    "validation_receipts": "Exact command/scope/return code/elapsed/log hash for every required and diagnostic check, including failures.",
    "repository_health": "Dated unrelated failures kept separately from affected required validation.",
    "field_principles": "Explain each field without wrapping scalar gates or ordinary dictionaries.",
    "plan_capture_complete_score": "One only for complete authenticated fresh schedule and independent reduction; output quality is separate.",
    "source_fidelity_rows": "Each source request and candidate with entity/quantity/order evidence and parse/executor status.",
    "call_manifest": "Canary and evaluation identities, actual owned runtime, raw paths/hashes and immutable terminal call states.",
    "usable_proposal_count": "Actual schema-usable proposals; zero is a valid measured null, not invented model failure.",
    "runtime_receipts": "Lease identity, model hash, model-count/runner/device records and joined task-window GPU telemetry.",
}
REQUIRED_FIELDS = frozenset(
    {
        *FIELD_PRINCIPLES,
        "experiment_id",
        "milestone",
        "phase",
        "started_at_utc",
        "completed_at_utc",
        "model_specs",
        "value_ready_score",
        "promotion_ready_score",
        "schedule",
        "evaluator_rows",
        "generation_cost_rows",
        "renamed_pair_rows",
        "candidate_manifest_path",
        "raw_call_manifest",
        "internal_validation_errors",
    }
)


def sha256_text(value: str) -> str:
    """Hash exact text so parser behavior cannot change candidate identity."""

    return capture_reducer.sha256_text(value)


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without normalizing the durable evidence."""

    return capture_reducer.sha256_file(path)


def _utc_now() -> str:  # pragma: no cover - wall-clock evidence.
    """Record real UTC while all durations use the monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each boundary so model and validation work stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[exp7361] phase={phase} event={event} {suffix}".rstrip(), flush=True)


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep both sides of one decision so a block identifies its cause."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure and preserve every expected and observed value."""

    failed = [dict(row) for row in checks if row.get("passed") is not True]
    first = failed[0] if failed else None
    return {
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "artifact_field": first.get("artifact_field") if first else "plan_capture_complete_score",
        "expected_value": deepcopy(first.get("expected_value")) if first else 1,
        "observed_value": deepcopy(first.get("observed_value")) if first else 1,
        "passed": not failed,
    }


def dependency_gate_rows(reducer: Mapping[str, Any], fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Apply the six exact YAML gates and reject quarantine language."""

    definitions = (
        (
            "exp7359-capture-reducer",
            reducer,
            {
                "experiment_id": "exp7359-capture-reducer",
                "milestone": MILESTONE,
                "run_date": RUN_DATE,
                "capture_reducer_ready_score": 1,
                "verdict_class": {"positive", "circular_positive", "null"},
                "flagged_adversarial": False,
            },
        ),
        (
            "exp7360-learning-fixture",
            fixture,
            {
                "experiment_id": "exp7360-learning-fixture",
                "milestone": MILESTONE,
                "run_date": RUN_DATE,
                "learning_fixture_ready_score": 1,
                "verdict_class": {"positive", "circular_positive", "null"},
                "flagged_adversarial": False,
            },
        ),
    )
    rows: list[JsonDict] = []
    for upstream, artifact, expected_fields in definitions:
        for field, expected in expected_fields.items():
            observed = artifact.get(field)
            passed = observed in expected if isinstance(expected, set) else observed == expected
            rows.append(
                gate_row(
                    f"{upstream}_{field}",
                    upstream,
                    field,
                    sorted(expected) if isinstance(expected, set) else expected,
                    observed,
                    passed,
                    "Only an eligible terminal same-milestone producer may authorize model work.",
                )
            )
        terminal_text = " ".join(
            str(artifact.get(field, "")) for field in ("status", "honest_verdict")
        ).lower()
        ineligible = any(word in terminal_text for word in ("quarantin", "blocked", "partial"))
        ineligible = ineligible or artifact.get("verdict_class") in {
            "blocked",
            "partial",
            "disqualified",
        }
        rows.append(
            gate_row(
                f"{upstream}_eligible_terminal_state",
                upstream,
                "status",
                "terminal_not_quarantined_blocked_partial_or_disqualified",
                terminal_text,
                not ineligible,
                "Quarantined or unfinished evidence cannot authorize dependent work.",
            )
        )
    return rows


def build_canary_schedule(public_manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze the four sealed development requests before any model outcome."""

    requests = public_manifest.get("development_canary")
    if not isinstance(requests, list) or len(requests) != CANARY_CALLS:
        raise ValueError("development_canary_count")
    schedule: list[JsonDict] = []
    for index, value in enumerate(requests):
        if not isinstance(value, Mapping):
            raise ValueError("development_canary_request")
        request = deepcopy(dict(value))
        canary_mod.decode_public_plan(
            canonical_json(
                {
                    "request_id": request.get("request_id"),
                    "assignments": {
                        name: request["allowed_starts"][name][0]
                        for name in request.get("activities", [])
                    },
                }
            ),
            request,
        )
        prompt = canary_mod.render_public_prompt(request)
        schedule.append(
            {
                "cohort_type": "development_canary",
                "call_index": index,
                "call_id": f"canary-{index:03d}",
                "candidate_id": f"canary-candidate-{index:03d}",
                "request_id": request["request_id"],
                "public_request": request,
                "public_request_sha256": sha256_json(request),
                "renaming_map": {},
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "seed": RANDOM_SEED["development"] + index,
                "max_generated_tokens": MAX_GENERATED_TOKENS,
            }
        )
    if len({row["request_id"] for row in schedule}) != CANARY_CALLS:
        raise ValueError("development_canary_identity")
    return schedule


def build_evaluation_schedule(public_manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze two candidates for every original and renamed sealed request."""

    panel = public_manifest.get("live_proposal_panel")
    if not isinstance(panel, list) or len(panel) != 32:
        raise ValueError("live_proposal_panel_count")
    schedule: list[JsonDict] = []
    for pair_index, value in enumerate(panel):
        if not isinstance(value, Mapping):
            raise ValueError("live_proposal_pair")
        pair = dict(value)
        order = (
            ("original", "twin")
            if pair.get("presentation_order") == "original_first"
            else ("twin", "original")
        )
        for candidate_index in range(2):
            for side in order:
                request_value = pair.get(side)
                if not isinstance(request_value, Mapping):
                    raise ValueError("public_request")
                request = deepcopy(dict(request_value))
                prompt = canary_mod.render_public_prompt(request)
                call_index = len(schedule)
                schedule.append(
                    {
                        "cohort_type": "evaluation",
                        "call_index": call_index,
                        "call_id": f"evaluation-{call_index:03d}",
                        "candidate_id": (
                            f"{pair.get('panel_id')}:{side}:candidate-{candidate_index}"
                        ),
                        "pair_index": pair_index,
                        "panel_id": pair.get("panel_id"),
                        "stream_id": pair.get("stream_id"),
                        "cohort": pair.get("cohort"),
                        "warmup": pair.get("warmup"),
                        "presentation_order": pair.get("presentation_order"),
                        "pair_side": side,
                        "candidate_index": candidate_index,
                        "request_id": request["request_id"],
                        "public_request": request,
                        "public_request_sha256": sha256_json(request),
                        "renaming_map": deepcopy(dict(pair.get("renaming_map") or {})),
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "seed": RANDOM_SEED["evaluation"] + call_index,
                        "max_generated_tokens": MAX_GENERATED_TOKENS,
                    }
                )
    if len(schedule) != EVALUATION_CALLS:  # pragma: no cover - loop shape proves this count.
        raise ValueError("evaluation_call_count")
    return schedule


def schedule_errors(
    canary: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    public_manifest: Mapping[str, Any],
) -> list[str]:
    """Rebuild both schedules so settings cannot drift after outcomes exist."""

    errors: list[str] = []
    try:
        expected_canary = build_canary_schedule(public_manifest)
        expected_evaluation = build_evaluation_schedule(public_manifest)
    except (KeyError, TypeError, ValueError) as error:
        return [f"schedule_source_invalid:{type(error).__name__}:{error}"]
    if list(canary) != expected_canary:
        errors.append("canary_schedule_rebuild_mismatch")
    if list(evaluation) != expected_evaluation:
        errors.append("evaluation_schedule_rebuild_mismatch")
    all_rows = [*canary, *evaluation]
    if any(row.get("max_generated_tokens") != MAX_GENERATED_TOKENS for row in all_rows):
        errors.append("token_budget")
    if len({str(row.get("call_id")) for row in all_rows}) != CANARY_CALLS + EVALUATION_CALLS:
        errors.append("call_identity")
    return errors


def source_fidelity(row: Mapping[str, Any]) -> JsonDict:
    """Score only public entities, quantities, and order from exact raw text."""

    request = dict(row.get("public_request") or {})
    activities = [str(value) for value in request.get("activities") or []]
    try:
        value = json.loads(str(row.get("raw_reply") or ""))
    except (json.JSONDecodeError, TypeError):
        value = None
    candidate = value if isinstance(value, dict) else {}
    assignments_value = candidate.get("assignments")
    assignments = assignments_value if isinstance(assignments_value, dict) else {}
    entity = set(assignments) == set(activities)
    typed = entity and all(
        isinstance(assignments[name], int) and not isinstance(assignments[name], bool)
        for name in activities
    )
    quantity = typed and all(
        assignments[name] in request.get("allowed_starts", {}).get(name, [])
        and assignments[name] + int(request.get("durations", {}).get(name, 0))
        <= int(request.get("horizon", -1))
        for name in activities
    )
    ordering = list(assignments) == activities
    identity = candidate.get("request_id") == request.get("request_id")
    parser_valid = row.get("parse_status") == "valid"
    schema_valid = bool(
        parser_valid
        and identity
        and entity
        and typed
        and set(candidate)
        == {
            "request_id",
            "assignments",
        }
    )
    return {
        "parser_valid": parser_valid,
        "schema_valid": schema_valid,
        "request_identity_fidelity": identity,
        "entity_fidelity": entity,
        "quantity_fidelity": quantity,
        "ordering_fidelity": ordering,
        "public_semantic_correct": bool(identity and entity and quantity and ordering),
        "expected_entities": activities,
        "observed_entities": list(assignments),
        "quantity_evidence": {
            name: {
                "observed": assignments.get(name),
                "allowed_starts": deepcopy(request.get("allowed_starts", {}).get(name)),
                "duration": request.get("durations", {}).get(name),
                "horizon": request.get("horizon"),
            }
            for name in activities
        },
    }


def reduce_evaluation(
    schedule: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    candidate_bytes: Mapping[str, bytes],
    *,
    expected_schedule_sha256: str,
    evaluator_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Use the tested reducer, then require agreement and no unstarted calls."""

    primary = capture_reducer.reduce_capture(
        schedule,
        calls,
        candidate_bytes,
        expected_schedule_sha256=expected_schedule_sha256,
        evaluator_rows=evaluator_rows,
        cost_rows=cost_rows,
    )
    independent = capture_reducer.independent_reduce_capture(
        schedule,
        calls,
        candidate_bytes,
        expected_schedule_sha256=expected_schedule_sha256,
    )
    agreement = all(
        primary.get(field) == independent.get(field)
        for field in ("capture_complete_score", "errors", "reduced_budget")
    )
    cancelled = any(row.get("terminal_state") == "cancelled" for row in calls)
    complete = int(
        primary.get("capture_complete_score") == 1
        and independent.get("capture_complete_score") == 1
        and agreement
        and not cancelled
    )
    evaluator_by_id = {str(row.get("call_id")): row for row in evaluator_rows}
    fidelity_rows = []
    for call in calls:
        evidence = source_fidelity(call)
        evaluation = evaluator_by_id.get(str(call.get("call_id")), {})
        fidelity_rows.append(
            {
                "call_id": call.get("call_id"),
                "candidate_id": call.get("candidate_id"),
                "request_id": call.get("request_id"),
                "panel_id": call.get("panel_id"),
                "candidate_index": call.get("candidate_index"),
                "pair_side": call.get("pair_side"),
                "parse_status": call.get("parse_status"),
                "parse_errors": deepcopy(call.get("parse_errors") or []),
                **evidence,
                "executor_evaluated": evaluation.get("evaluated") is True,
                "executor_valid": evaluation.get("hidden_rule_accepted"),
                "censored": call.get("terminal_state") == "cancelled",
            }
        )
    return {
        "plan_capture_complete_score": complete,
        "usable_proposal_count": sum(row.get("parse_status") == "valid" for row in calls),
        "public_semantic_correct_count": sum(
            row["public_semantic_correct"] for row in fidelity_rows
        ),
        "executor_valid_count": sum(
            row.get("hidden_rule_accepted") is True for row in evaluator_rows
        ),
        "primary_independent_agreement": agreement,
        "reducer_errors": deepcopy(primary.get("errors") or []),
        "sample_size_budget": deepcopy(primary["reduced_budget"]),
        "rows": deepcopy(primary["call_rows"]),
        "renamed_pair_rows": deepcopy(primary["identifier_twin_rows"]),
        "semantic_failure_counts": deepcopy(primary["semantic_failure_counts"]),
        "source_fidelity_rows": fidelity_rows,
        "primary_reduction": primary,
        "independent_reduction": independent,
    }


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete record without recursively hashing its checksum."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Publish one complete canonical object through an atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
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
    """Return an empty object when a prerequisite is missing or malformed."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _phase_close(
    spans: list[JsonDict],
    phase: str,
    started: float,
    units: int,
    checkpoint: str | None = None,
    *,
    cohort: str | None = None,
) -> None:  # pragma: no cover
    """Close one real monotonic interval and name its durable boundary."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "cohort": cohort,
            "start_monotonic_s": started,
            "end_monotonic_s": ended,
            "duration_s": ended - started,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
    )


def base_artifact(run_date: str, started_at: str, root: Path = REPO_ROOT) -> JsonDict:
    """Create the complete blocked shape before any fallible dependency check."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 2,
        "status": "blocked_not_started",
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
            "historical_model_loads": 0,
            "historical_generation_calls": 0,
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
            "canary_planned_units": CANARY_CALLS,
            "evaluation_planned_units": EVALUATION_CALLS,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "cancelled_units": EVALUATION_CALLS,
            "censored_units": EVALUATION_CALLS,
            "max_generated_tokens_per_call": MAX_GENERATED_TOKENS,
            "generation_timeout_s": GENERATION_TIMEOUT_S,
            "stopping_rule": "four canary calls, then 128 fixed evaluation calls or the generation deadline; no retries",
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
        "usable_proposal_count": 0,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
        "source_fidelity_rows": [],
        "renamed_pair_rows": [],
        "evaluator_rows": [],
        "generation_cost_rows": [],
        "call_manifest": {
            "schema": "carnot.exp7361.call_manifest.v1",
            "canary_calls": [],
            "evaluation_calls": [],
        },
        "runtime_receipts": {},
        "candidate_manifest_path": str(root / RAW_DIR / "proposal_manifest.json"),
        "raw_call_manifest": {"schedule": [], "calls": []},
        "internal_validation_errors": [],
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object,
    *,
    allow_preterminal: bool = False,
    root: Path = REPO_ROOT,
) -> list[str]:
    """Reject identity, reduction, checksum, and terminal-state drift."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors = [
        f"missing_required_field:{field}" for field in sorted(REQUIRED_FIELDS - set(artifact))
    ]
    if errors:
        return errors
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("lifecycle_invalid")
    specs = artifact.get("MODEL_SPECS")
    if not isinstance(specs, list) or not specs or specs[0].get("hf_id") != MODEL_ID:
        errors.append("model_identity_invalid")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if not allow_preterminal and artifact.get("status") != "blocked_not_started":
        errors.extend(independent_reduce_artifact(artifact, root=root))
        if (
            artifact.get("model_invoked") is True
            and artifact.get("inference_substrate_class") != "model_bounded_generation"
        ):
            errors.append("substrate_class_invalid")
        if artifact.get("verdict_class") in {"blocked", "disqualified"} and any(
            artifact.get(field) != 0
            for field in (
                "plan_capture_complete_score",
                "value_ready_score",
                "promotion_ready_score",
            )
        ):
            errors.append("failed_scores_invalid")
    return list(dict.fromkeys(errors))


def _source_path_gate(
    fixture: Mapping[str, Any], key: str, expected_path: Path, root: Path
) -> JsonDict:  # pragma: no cover
    """Check one fixture-declared path and its exact producer hash."""

    declared = dict(fixture.get("raw_evidence_paths") or {}).get(key)
    resolved = Path(str(declared)).resolve() if declared else Path("/missing")
    expected = (root / expected_path).resolve()
    hashes = dict(fixture.get("source_artifact_hashes") or {})
    declared_hash = hashes.get(str(resolved)) or hashes.get(str(expected))
    observed_hash = sha256_file(expected) if expected.is_file() else None
    observed = {
        "path": str(resolved),
        "exists": expected.is_file(),
        "sha256": observed_hash,
    }
    wanted = {"path": str(expected), "exists": True, "sha256": declared_hash}
    return gate_row(
        f"fixture_{key}_bytes",
        "exp7360-learning-fixture",
        "source_artifact_hashes",
        wanted,
        observed,
        bool(declared_hash and observed == wanted),
        "The capture must consume the exact sealed fixture bytes.",
    )


def collect_preconditions(
    root: Path, run_date: str
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - live filesystem and GPU checks.
    """Authenticate producers and sealed bytes before the reusable GPU preflight."""

    progress("preconditions", "start")
    checks = [
        gate_row(
            "run_date",
            "execution_contract",
            "run_date",
            RUN_DATE,
            run_date,
            run_date == RUN_DATE,
            "This capture is tied to the fixed V646 execution date.",
        )
    ]
    required = (
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
        REDUCER_PATH,
        FIXTURE_PATH,
        PUBLIC_MANIFEST_PATH,
        PRIVATE_MANIFEST_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/experiment_7347_v645_plan_canary.py"),
        Path("python/carnot/experiment_7348_v645_plan_capture.py"),
        Path("python/carnot/experiment_7359_v646_capture_reducer.py"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
    )
    availability = {path.as_posix(): (root / path).is_file() for path in required}
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    availability["REQ-REPORT-7361"] = "REQ-REPORT-7361" in spec
    checks.append(
        gate_row(
            "required_source_paths",
            "repository",
            "preconditions_checked",
            {key: True for key in availability},
            availability,
            all(availability.values()),
            "All declared bytes and the driving requirement must exist before model work.",
        )
    )
    reducer = _load_object(root / REDUCER_PATH)
    fixture = _load_object(root / FIXTURE_PATH)
    checks.extend(dependency_gate_rows(reducer, fixture))
    checks.append(_source_path_gate(fixture, "public_manifest", PUBLIC_MANIFEST_PATH, root))
    checks.append(_source_path_gate(fixture, "private_manifest", PRIVATE_MANIFEST_PATH, root))
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7361" in exclusion or "exp7361" in exclusion.lower()
    checks.append(
        gate_row(
            "current_task_not_excluded",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
            not excluded,
            "An excluded task must stop before it acquires a runtime lease.",
        )
    )
    context: JsonDict = {"reducer": reducer, "fixture": fixture}
    if any(row.get("passed") is not True for row in checks):
        progress("preconditions", "complete", passed=False)
        return checks, context

    public_manifest = _load_object(root / PUBLIC_MANIFEST_PATH)
    private_manifest = _load_object(root / PRIVATE_MANIFEST_PATH)
    canary_schedule = build_canary_schedule(public_manifest)
    evaluation_schedule = build_evaluation_schedule(public_manifest)
    schedule_issues = schedule_errors(canary_schedule, evaluation_schedule, public_manifest)
    checks.append(
        gate_row(
            "frozen_schedule",
            "exp7360-learning-fixture",
            "sample_size_budget",
            {"canary": CANARY_CALLS, "evaluation": EVALUATION_CALLS, "errors": []},
            {
                "canary": len(canary_schedule),
                "evaluation": len(evaluation_schedule),
                "errors": schedule_issues,
            },
            not schedule_issues,
            "Every prompt, request, candidate, order, seed, and token cap freezes before inference.",
        )
    )
    private_public_hash = private_manifest.get("public_manifest_sha256")
    actual_public_hash = sha256_file(root / PUBLIC_MANIFEST_PATH)
    checks.append(
        gate_row(
            "private_public_binding",
            "exp7360-learning-fixture",
            "public_manifest_sha256",
            actual_public_hash,
            private_public_hash,
            private_public_hash == actual_public_hash,
            "Private labels must be sealed against the exact public request bytes.",
        )
    )
    if any(row.get("passed") is not True for row in checks):
        progress("preconditions", "complete", passed=False)
        return checks, context

    progress("preconditions", "before_runtime_preflight")
    old_checks, runtime = canary_mod._collect_preconditions(root, canary_mod.RUN_DATE)
    progress("preconditions", "after_runtime_preflight")
    for row in old_checks:
        if row.get("check") in {
            "cached_current_model",
            "native_cuda_runner",
            "cuda_inventory_and_owned_capacity",
            "force_live_mode",
        }:
            checks.append(
                gate_row(
                    f"runtime_{row['check']}",
                    str(row.get("upstream")),
                    str(row.get("artifact_field")),
                    row.get("expected_value"),
                    row.get("observed_value"),
                    row.get("passed") is True,
                    str(row.get("principle")),
                )
            )
    model_spec = deepcopy(dict(runtime.get("model_spec") or {}))
    model_spec["runtime_settings"] = {
        "max_generated_tokens": MAX_GENERATED_TOKENS,
        "canary_timeout_s": CANARY_TIMEOUT_S,
        "generation_timeout_s": GENERATION_TIMEOUT_S,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "parallel_requests": 1,
        "development_seed": RANDOM_SEED["development"],
        "evaluation_seed": RANDOM_SEED["evaluation"],
    }
    context.update(runtime)
    context.update(
        {
            "public_manifest": public_manifest,
            "private_manifest": private_manifest,
            "canary_schedule": canary_schedule,
            "evaluation_schedule": evaluation_schedule,
            "model_spec": model_spec,
        }
    )
    progress("preconditions", "complete", passed=all(row.get("passed") is True for row in checks))
    return checks, context


@contextmanager
def _runtime_settings(seed: int, timeout_s: float) -> Iterator[None]:  # pragma: no cover
    """Apply one frozen cohort configuration to the shared owned runner."""

    values = {
        "TASK_ID": TASK_ID,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "INFERENCE_WINDOW_TIMEOUT_S": timeout_s,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": seed,
            "evaluation": RANDOM_SEED["evaluation"],
            "resampling": RANDOM_SEED["resampling"],
        },
    }
    previous = {name: getattr(canary_mod, name) for name in values}
    try:
        for name, value in values.items():
            setattr(canary_mod, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(canary_mod, name, value)


def _bind_runtime_rows(
    schedule: Sequence[Mapping[str, Any]], capture: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover
    """Join shared-runner replies to the schedule identities frozen earlier."""

    runtime_rows = list(capture.get("rows") or [])
    rows: list[JsonDict] = []
    for index, schedule_value in enumerate(schedule):
        schedule_row = dict(schedule_value)
        if index >= len(runtime_rows):
            rows.append(
                prior_capture.censored_call_row(
                    schedule_row,
                    dict(capture.get("runtime_identity") or {}),
                    str(capture.get("runtime_error") or "generation_deadline"),
                )
            )
            continue
        runtime_row = dict(runtime_rows[index])
        if runtime_row.get("terminal_state") == "cancelled":
            rows.append(
                prior_capture.censored_call_row(
                    schedule_row,
                    dict(runtime_row.get("runtime_identity_receipt") or {}),
                    str(runtime_row.get("error") or "generation_deadline"),
                )
            )
            continue
        response = {
            "raw_reply": runtime_row.get("raw_reply"),
            "raw_response": runtime_row.get("raw_response"),
            "prompt_tokens": runtime_row.get("prompt_tokens"),
            "completion_tokens": runtime_row.get("completion_tokens"),
            "latency_s": runtime_row.get("latency_s"),
            "finish_reason": runtime_row.get("finish_reason"),
            "error": runtime_row.get("error"),
        }
        rows.append(
            prior_capture.build_call_row(
                schedule_row=schedule_row,
                response=response,
                runtime_identity=dict(runtime_row.get("runtime_identity_receipt") or {}),
            )
        )
    return rows


def _capture_cohort(
    context: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    raw_dir: Path,
    *,
    seed: int,
    timeout_s: float,
) -> JsonDict:  # pragma: no cover - live model work.
    """Run one fixed cohort through the existing leased GGUF runtime."""

    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = [
        deepcopy(dict(row["public_request"])) for row in schedule
    ]
    with _runtime_settings(seed, timeout_s):
        capture = canary_mod._live_capture(runtime_context, raw_dir / "runtime")
    rows = _bind_runtime_rows(schedule, capture)
    for index, row in enumerate(rows):
        _atomic_json(raw_dir / f"call_{index:03d}.json", row)
        progress(
            "generation",
            "checkpoint",
            cohort=row.get("cohort_type"),
            completed=index + 1,
            total=len(rows),
        )
    capture["rows"] = rows
    return capture


def _canary_readiness(capture: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    """Require four owned responses and four exact public schemas."""

    rows = list(capture.get("rows") or [])
    reduced = canary_mod.reduce_raw_calls(
        rows, load_receipt=dict(capture.get("load_receipt") or {})
    )
    parse_failures = [str(row.get("call_id")) for row in rows if row.get("parse_status") != "valid"]
    return {
        "transport_ready_score": reduced["plan_transport_ready_score"],
        "schema_ready_score": int(len(rows) == CANARY_CALLS and not parse_failures),
        "parse_failure_call_ids": parse_failures,
        "passed": reduced["plan_transport_ready_score"] == 1 and not parse_failures,
    }


def _generation_cost_rows(
    calls: Sequence[Mapping[str, Any]], load_duration_s: float
) -> list[JsonDict]:  # pragma: no cover
    """Allocate load cost without treating later CPU replay as model work."""

    share = float(load_duration_s) / len(calls) if calls else 0.0
    return [
        {
            "call_id": row.get("call_id"),
            "candidate_id": row.get("candidate_id"),
            "attempted": row.get("attempted") is True,
            "terminal_state": row.get("terminal_state"),
            "prompt_tokens": int(row.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(row.get("completion_tokens", 0) or 0),
            "generation_duration_s": float(row.get("latency_s", 0.0) or 0.0),
            "allocated_model_load_s": share,
            "current_model_generation_calls": int(row.get("attempted") is True),
            "historical_model_generation_calls": 0,
        }
        for row in calls
    ]


def _load_duration(capture: Mapping[str, Any]) -> float:  # pragma: no cover
    """Sum only measured model-load spans for one cohort."""

    return sum(
        float(row.get("duration_s", 0.0) or 0.0)
        for row in capture.get("phase_spans") or []
        if row.get("phase") == "model_load"
    )


def _invocation_counts(captures: Sequence[Mapping[str, Any]]) -> JsonDict:  # pragma: no cover
    """Count every current attempt while leaving historical counts at zero."""

    loads = [dict(capture.get("load_receipt") or {}) for capture in captures]
    calls = [row for capture in captures for row in capture.get("rows") or []]
    states = [str(row.get("terminal_state")) for row in calls]
    return {
        "model_loads_attempted": sum(row.get("attempted") is True for row in loads),
        "model_loads_completed": sum(row.get("completed") is True for row in loads),
        "model_loads_failed": sum(row.get("failed") is True for row in loads),
        "model_loads_cancelled": sum(row.get("cancelled") is True for row in loads),
        "model_loads_in_flight": sum(row.get("in_flight") is True for row in loads),
        "generation_calls_attempted": sum(row.get("attempted") is True for row in calls),
        "generation_calls_completed": states.count("response"),
        "generation_calls_failed": states.count("request_error"),
        "generation_calls_cancelled": states.count("cancelled"),
        "generation_calls_in_flight": states.count("in_flight"),
        "historical_model_loads": 0,
        "historical_generation_calls": 0,
    }


def _manifest_rows(
    rows: Sequence[Mapping[str, Any]], raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Bind each call identity and runtime receipt to its exact candidate file."""

    result = []
    for index, row in enumerate(rows):
        path = raw_dir / f"call_{index:03d}.json"
        result.append(
            {
                "call_id": row.get("call_id"),
                "candidate_id": row.get("candidate_id"),
                "request_id": row.get("request_id"),
                "cohort_type": row.get("cohort_type"),
                "terminal_state": row.get("terminal_state"),
                "attempted": row.get("attempted") is True,
                "raw_path": str(path),
                "raw_sha256": sha256_file(path),
                "raw_reply_sha256": row.get("raw_reply_sha256"),
                "runtime_identity_receipt": deepcopy(row.get("runtime_identity_receipt") or {}),
            }
        )
    return result


def classify_terminal(
    *,
    complete_score: int,
    usable_count: int,
    semantic_count: int,
    required_checks_passed: bool,
    flagged_adversarial: bool,
) -> JsonDict:  # pragma: no cover
    """Keep accounting, value, and promotion as independent decisions."""

    if not required_checks_passed or flagged_adversarial:
        return {
            "status": "complete_fresh_plan_capture_disqualified",
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_fresh_plan_capture_required_check_failed",
            "plan_capture_complete_score": 0,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    if complete_score != 1:
        return {
            "status": "partial_fresh_plan_capture_censored",
            "verdict_class": "partial",
            "honest_verdict": "partial_fresh_plan_capture_generation_budget_censored",
            "plan_capture_complete_score": 0,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    if usable_count == 0 or semantic_count == 0:
        return {
            "status": "complete_fresh_plan_capture_null",
            "verdict_class": "null",
            "honest_verdict": "complete_null_fresh_plan_capture_no_publicly_faithful_proposal",
            "plan_capture_complete_score": 1,
            "value_ready_score": 0,
            "promotion_ready_score": 0,
        }
    return {
        "status": "complete_fresh_plan_capture",
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_fresh_plan_capture_with_public_fidelity",
        "plan_capture_complete_score": 1,
        "value_ready_score": 0,
        "promotion_ready_score": 0,
    }


def _acceptance_gates(
    checks: Sequence[Mapping[str, Any]],
    canary: Mapping[str, Any],
    reduced: Mapping[str, Any],
    scoped_ok: bool,
    terminal_ok: bool,
    flagged: bool,
) -> dict[str, JsonDict]:  # pragma: no cover
    """Report science separately so accounting cannot authorize promotion."""

    return {
        "preconditions": {
            "expected": True,
            "observed": all(row.get("passed") is True for row in checks),
            "passed": all(row.get("passed") is True for row in checks),
            "principle": "Exact producers, bytes, model, runner, and device must pass first.",
        },
        "canary_readiness": {
            "expected": True,
            "observed": canary.get("passed") is True,
            "passed": canary.get("passed") is True,
            "principle": "All four development calls must have owned transport and exact schema.",
        },
        "capture_completion": {
            "expected": 1,
            "observed": reduced.get("plan_capture_complete_score"),
            "passed": reduced.get("plan_capture_complete_score") == 1,
            "principle": "Every planned evaluation call needs an authenticated non-cancelled disposition.",
        },
        "scientific_value": {
            "expected": ">=1 publicly faithful proposal",
            "observed": reduced.get("public_semantic_correct_count", 0),
            "passed": int(reduced.get("public_semantic_correct_count", 0) or 0) >= 1,
            "principle": "A value miss is null science and does not rewrite capture accounting.",
        },
        "affected_validation": {
            "expected": True,
            "observed": scoped_ok,
            "passed": scoped_ok,
            "principle": "Only explicit changed files and their tests gate this task.",
        },
        "terminal_validation": {
            "expected": True,
            "observed": terminal_ok,
            "passed": terminal_ok,
            "principle": "Cold reduction and both strict readers must pass.",
        },
        "adversarial_clear": {
            "expected": False,
            "observed": flagged,
            "passed": not flagged,
            "principle": "A current critical finding prevents readiness and promotion.",
        },
        "promotion": {
            "expected": 1,
            "observed": 0,
            "passed": False,
            "principle": "This capture supplies proposals; it never authorizes promotion.",
        },
    }


def run_affected_validation(root: Path, raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Run the fixed Exp7303 command set over this module and test only."""

    basetemp = Path("/tmp/carnot-exp7361-v646-scoped")
    coverage_file = Path("/tmp/carnot-exp7361-v646-coverage/.coverage")
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    return run_scoped_validation(
        root,
        test_paths=[TEST_PATH.as_posix()],
        changed_modules=[MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=basetemp,
        coverage_file=coverage_file,
        log_dir=raw_dir / "validation/affected",
        historical_failures=[],
    )


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful real receipt for every named command."""

    return all(
        sum(
            row.get("name") == name and row.get("passed") is True and row.get("exit_code") == 0
            for row in receipts
        )
        == 1
        for name in names
    )


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:  # pragma: no cover
    """Reload candidate bytes and reproduce the stored evaluation reduction."""

    schedule = artifact.get("schedule")
    manifest = artifact.get("raw_call_manifest")
    calls = manifest.get("calls") if isinstance(manifest, Mapping) else None
    if not isinstance(schedule, list) or not isinstance(calls, list):
        return ["raw_evidence_unavailable"]
    if not schedule and not calls and artifact.get("status") == "blocked_not_started":
        return []
    call_manifest = dict(artifact.get("call_manifest") or {})
    entries = list(call_manifest.get("evaluation_calls") or [])
    candidate_bytes: dict[str, bytes] = {}
    errors: list[str] = []
    for entry in entries:
        path = Path(str(entry.get("raw_path")))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"candidate_path_missing:{entry.get('call_id')}")
            continue
        blob = path.read_bytes()
        if sha256_file(path) != entry.get("raw_sha256"):
            errors.append(f"candidate_hash_mismatch:{entry.get('call_id')}")
        candidate_bytes[str(entry.get("call_id"))] = blob
    if errors:
        return errors
    reduced = reduce_evaluation(
        schedule,
        calls,
        candidate_bytes,
        expected_schedule_sha256=sha256_json(schedule),
        evaluator_rows=list(artifact.get("evaluator_rows") or []),
        cost_rows=list(artifact.get("generation_cost_rows") or []),
    )
    comparisons = {
        "plan_capture_complete_score": reduced["plan_capture_complete_score"],
        "usable_proposal_count": reduced["usable_proposal_count"],
        "source_fidelity_rows": reduced["source_fidelity_rows"],
        "renamed_pair_rows": reduced["renamed_pair_rows"],
    }
    for field, expected in comparisons.items():
        if artifact.get(field) != expected:
            errors.append(f"{field}_mismatch")
    observed_budget = dict(artifact.get("sample_size_budget") or {})
    if any(
        observed_budget.get(key) != value for key, value in reduced["sample_size_budget"].items()
    ):
        errors.append("sample_size_budget_mismatch")
    return errors


def run_terminal_validation(
    root: Path, candidate: Path, raw_dir: Path
) -> list[JsonDict]:  # pragma: no cover
    """Cold-reduce the candidate and run both strict artifact readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib;"
        "from carnot.experiment_7361_v646_fresh_plan_capture import independent_reduce_artifact;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "e=independent_reduce_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
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


def _source_hashes(
    root: Path,
    context: Mapping[str, Any],
    proposal_manifest: Path,
    evaluation_path: Path,
) -> JsonDict:  # pragma: no cover
    """Bind producers, code, model, evaluator, and all sealed manifests."""

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
        REDUCER_PATH,
        FIXTURE_PATH,
        PUBLIC_MANIFEST_PATH,
        PRIVATE_MANIFEST_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7359_v646_capture_reducer.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
    )
    hashes = {path.as_posix(): sha256_file(root / path) for path in paths}
    model_spec = dict(context.get("model_spec") or {})
    if model_spec.get("path") and model_spec.get("sha256"):
        hashes[str(model_spec["path"])] = str(model_spec["sha256"])
    hashes[str(proposal_manifest)] = sha256_file(proposal_manifest)
    hashes[str(evaluation_path)] = sha256_file(evaluation_path)
    return hashes


def _write_blocked(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    output: Path,
    started: float,
) -> JsonDict:  # pragma: no cover
    """Publish external absence without model-shaped success fields."""

    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition_failed",
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


def _blocked_after_canary(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    capture: Mapping[str, Any],
    readiness: Mapping[str, Any],
    raw_dir: Path,
    output: Path,
    started: float,
) -> JsonDict:  # pragma: no cover
    """Retain attempted model work when canary readiness blocks evaluation."""

    rows = list(capture.get("rows") or [])
    manifest_rows = _manifest_rows(rows, raw_dir / "canary")
    artifact.update(
        {
            "status": "complete_fresh_plan_capture_disqualified",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "MODEL_SPECS": [
                {
                    "hf_id": MODEL_ID,
                    "quantization": QUANTIZATION,
                    "file_sha256": dict(capture.get("runtime_identity") or {}).get("model_sha256"),
                }
            ],
            "model_invoked": dict(capture.get("load_receipt") or {}).get("attempted") is True,
            "invocation_counts": _invocation_counts([capture]),
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_bounded_generation",
            "rows": rows,
            "call_manifest": {
                "schema": "carnot.exp7361.call_manifest.v1",
                "canary_calls": manifest_rows,
                "evaluation_calls": [],
                "canary_readiness": deepcopy(dict(readiness)),
            },
            "runtime_receipts": {"canary": deepcopy(dict(capture))},
            "completed_at_utc": _utc_now(),
            "duration_s": time.monotonic() - started,
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_canary_readiness_failed_evaluation_not_started",
            "flagged_adversarial": True,
            "gate_check_summary": gate_check_summary(
                [
                    *checks,
                    gate_row(
                        "canary_readiness",
                        EXPERIMENT_ID,
                        "call_manifest",
                        True,
                        readiness.get("passed"),
                        False,
                        "Evaluation requires four owned schema-valid canary calls.",
                    ),
                ]
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(output, artifact)
    return artifact


def run_experiment(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    output_path: Path | None = None,
) -> JsonDict:  # pragma: no cover - end-to-end live orchestration.
    """Authenticate, capture, reduce, validate, and atomically publish once."""

    started = time.monotonic()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact = base_artifact(run_date, _utc_now(), root)
    checks, context = collect_preconditions(root, run_date)
    if any(row.get("passed") is not True for row in checks):
        progress("write", "before_blocked_artifact")
        result = _write_blocked(artifact, checks, output, started)
        progress("write", "after_blocked_artifact", artifact=output)
        return result

    canary_schedule = list(context["canary_schedule"])
    evaluation_schedule = list(context["evaluation_schedule"])
    schedule_record = {
        "schema": "carnot.exp7361.schedule.v1",
        "sealed_before_generation": True,
        "canary_schedule": canary_schedule,
        "evaluation_schedule": evaluation_schedule,
        "canary_schedule_sha256": sha256_json(canary_schedule),
        "evaluation_schedule_sha256": sha256_json(evaluation_schedule),
    }
    _atomic_json(raw_dir / "schedule_manifest.json", schedule_record)

    progress("canary", "before_model_load_and_generation", planned=CANARY_CALLS)
    canary_capture = _capture_cohort(
        context,
        canary_schedule,
        raw_dir / "canary",
        seed=RANDOM_SEED["development"],
        timeout_s=CANARY_TIMEOUT_S,
    )
    progress("canary", "after_model_load_and_generation", completed=len(canary_capture["rows"]))
    readiness = _canary_readiness(canary_capture)
    if readiness["passed"] is not True:
        progress("evaluation", "blocked_by_canary", diagnostic=canonical_json(readiness))
        return _blocked_after_canary(
            artifact, checks, canary_capture, readiness, raw_dir, output, started
        )

    progress("evaluation_generation", "before_model_load_and_generation", planned=EVALUATION_CALLS)
    evaluation_capture = _capture_cohort(
        context,
        evaluation_schedule,
        raw_dir / "evaluation",
        seed=RANDOM_SEED["evaluation"],
        timeout_s=GENERATION_TIMEOUT_S,
    )
    progress(
        "evaluation_generation",
        "after_model_load_and_generation",
        completed=len(evaluation_capture["rows"]),
    )
    evaluation_calls = list(evaluation_capture["rows"])
    proposal_manifest: JsonDict = {
        "schema": "carnot.exp7361.proposal_manifest.v1",
        "producer": EXPERIMENT_ID,
        "sealed_before_private_evaluation": True,
        "schedule": evaluation_schedule,
        "calls": evaluation_calls,
    }
    proposal_manifest["manifest_sha256"] = sha256_json(proposal_manifest)
    proposal_path = raw_dir / "proposal_manifest.json"
    _atomic_json(proposal_path, proposal_manifest)

    evaluator_started = time.monotonic()
    progress("private_evaluation", "before_evaluator", manifest=proposal_path)
    evaluation = prior_capture._score_sealed_calls(
        proposal_path,
        root / PUBLIC_MANIFEST_PATH,
        root / PRIVATE_MANIFEST_PATH,
    )
    evaluation_path = raw_dir / "private_evaluation.json"
    _atomic_json(evaluation_path, evaluation)
    progress("private_evaluation", "after_evaluator", evaluated=evaluation["evaluated_call_count"])

    spans: list[JsonDict] = []
    for cohort, capture in (
        ("development_canary", canary_capture),
        ("evaluation", evaluation_capture),
    ):
        for span in capture.get("phase_spans") or []:
            spans.append({**deepcopy(dict(span)), "cohort": cohort})
    _phase_close(
        spans,
        "evaluation",
        evaluator_started,
        len(evaluation["rows"]),
        str(evaluation_path),
        cohort="private_executor",
    )

    evaluation_entries = _manifest_rows(evaluation_calls, raw_dir / "evaluation")
    candidate_bytes = {
        str(entry["call_id"]): Path(str(entry["raw_path"])).read_bytes()
        for entry in evaluation_entries
    }
    cost_rows = _generation_cost_rows(evaluation_calls, _load_duration(evaluation_capture))
    reduced = reduce_evaluation(
        evaluation_schedule,
        evaluation_calls,
        candidate_bytes,
        expected_schedule_sha256=sha256_json(evaluation_schedule),
        evaluator_rows=list(evaluation["rows"]),
        cost_rows=cost_rows,
    )

    validation_started = time.monotonic()
    progress("validation", "before_affected_commands")
    validation = run_affected_validation(root, raw_dir)
    progress("validation", "after_affected_commands", passed=validation["required_checks_passed"])
    scoped_receipts = list(validation.get("validation_receipts") or [])
    scoped_ok = bool(
        validation.get("required_checks_passed")
        and _receipts_pass(scoped_receipts, REQUIRED_CHECK_NAMES)
    )
    _phase_close(spans, "validation", validation_started, len(scoped_receipts))

    captures = [canary_capture, evaluation_capture]
    counts = _invocation_counts(captures)
    actual_model = dict(context["model_spec"])
    artifact.update(
        {
            "status": "complete_fresh_plan_capture",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "MODEL_SPECS": [
                {
                    "hf_id": MODEL_ID,
                    "quantization": QUANTIZATION,
                    "filename": Path(str(actual_model["path"])).name,
                    "file_sha256": actual_model["sha256"],
                    "file_bytes": actual_model["bytes"],
                }
            ],
            "model_specs": [actual_model],
            "model_invoked": counts["model_loads_attempted"] > 0,
            "invocation_counts": counts,
            "inference_substrate": "live_llm_inference",
            "inference_substrate_class": "model_bounded_generation",
            "phase_spans": spans,
            "rows": [*deepcopy(canary_capture["rows"]), *deepcopy(evaluation_calls)],
            "schedule": deepcopy(evaluation_schedule),
            "sample_size_budget": {
                **deepcopy(reduced["sample_size_budget"]),
                "canary_planned_units": CANARY_CALLS,
                "canary_attempted_units": sum(
                    row.get("attempted") is True for row in canary_capture["rows"]
                ),
                "evaluation_planned_units": EVALUATION_CALLS,
                "max_generated_tokens_per_call": MAX_GENERATED_TOKENS,
                "generation_timeout_s": GENERATION_TIMEOUT_S,
                "stopping_rule": "four canary calls, then 128 fixed evaluation calls or the generation deadline; no retries",
            },
            "candidate_manifest_path": str(proposal_path),
            "raw_call_manifest": proposal_manifest,
            "evaluator_rows": deepcopy(evaluation["rows"]),
            "generation_cost_rows": cost_rows,
            "source_fidelity_rows": deepcopy(reduced["source_fidelity_rows"]),
            "renamed_pair_rows": deepcopy(reduced["renamed_pair_rows"]),
            "plan_capture_complete_score": reduced["plan_capture_complete_score"],
            "usable_proposal_count": reduced["usable_proposal_count"],
            "public_semantic_correct_count": reduced["public_semantic_correct_count"],
            "executor_valid_count": reduced["executor_valid_count"],
            "primary_independent_agreement": reduced["primary_independent_agreement"],
            "reducer_errors": deepcopy(reduced["reducer_errors"]),
            "call_manifest": {
                "schema": "carnot.exp7361.call_manifest.v1",
                "canary_schedule_sha256": sha256_json(canary_schedule),
                "evaluation_schedule_sha256": sha256_json(evaluation_schedule),
                "canary_calls": _manifest_rows(canary_capture["rows"], raw_dir / "canary"),
                "evaluation_calls": evaluation_entries,
                "canary_readiness": readiness,
            },
            "runtime_receipts": {
                "model_count": 1,
                "runner": deepcopy(context.get("runner") or {}),
                "device_preflight": deepcopy(context.get("query_receipts") or []),
                "canary": {
                    "load_receipt": deepcopy(canary_capture["load_receipt"]),
                    "runtime_identity": deepcopy(canary_capture["runtime_identity"]),
                    "served_model": deepcopy(canary_capture["served_model_receipt"]),
                    "gpu_telemetry": deepcopy(canary_capture["gpu_receipts"]),
                },
                "evaluation": {
                    "load_receipt": deepcopy(evaluation_capture["load_receipt"]),
                    "runtime_identity": deepcopy(evaluation_capture["runtime_identity"]),
                    "served_model": deepcopy(evaluation_capture["served_model_receipt"]),
                    "gpu_telemetry": deepcopy(evaluation_capture["gpu_receipts"]),
                },
            },
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
            usable_count=reduced["usable_proposal_count"],
            semantic_count=reduced["public_semantic_correct_count"],
            required_checks_passed=scoped_ok,
            flagged_adversarial=False,
        )
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, readiness, reduced, scoped_ok, False, False
    )
    artifact["gate_check_summary"] = gate_check_summary(checks)
    artifact["source_artifact_hashes"] = _source_hashes(
        root, context, proposal_path, evaluation_path
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "terminal_candidate.json"
    _atomic_json(candidate, artifact)

    terminal_started = time.monotonic()
    progress("terminal_validation", "before_commands", candidate=candidate)
    terminal_receipts = run_terminal_validation(root, candidate, raw_dir)
    progress("terminal_validation", "after_commands")
    terminal_ok = _receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    artifact["validation_receipts"] = [*scoped_receipts, *terminal_receipts]
    _phase_close(spans, "validation", terminal_started, len(terminal_receipts), cohort="terminal")
    artifact.update(
        classify_terminal(
            complete_score=reduced["plan_capture_complete_score"],
            usable_count=reduced["usable_proposal_count"],
            semantic_count=reduced["public_semantic_correct_count"],
            required_checks_passed=scoped_ok and terminal_ok and independent_ok,
            flagged_adversarial=flagged,
        )
    )
    artifact["flagged_adversarial"] = flagged
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks,
        readiness,
        reduced,
        scoped_ok,
        terminal_ok and independent_ok,
        flagged,
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
        if name != "promotion"
    ]
    artifact["gate_check_summary"] = gate_check_summary([*checks, *gate_rows])
    write_started = time.monotonic()
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = time.monotonic() - started
    _phase_close(spans, "write", write_started, 1, str(output))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validation_errors = validate_artifact(artifact, root=root)
    if validation_errors:
        artifact.update(
            {
                "status": "complete_fresh_plan_capture_disqualified",
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
    progress("write", "before_atomic_publish", artifact=output)
    _atomic_json(output, artifact)
    progress("write", "after_atomic_publish", verdict=artifact["honest_verdict"])
    return artifact


def _date_argument(value: str) -> str:
    """Reject accidental execution outside the fixed V646 date."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the live capture or cold-check one task-owned candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    progress("entrypoint", "start", date=args.date)
    if args.validate is not None:
        errors = validate_artifact(_load_object(args.validate), allow_preterminal=True)
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
                "usable_proposal_count": result["usable_proposal_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
