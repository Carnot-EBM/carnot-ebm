"""Qualify bounded Qwen transport for partial Boolean assignments.

The model sees four sealed development formulas. It returns ordinary JSON
literal lists. Transport readiness stays separate from formula correctness.

Spec refs: REQ-CL-7372 and SCENARIO-CL-7372-*.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as runtime
from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.inference.llama_server_supervisor import canonical_json
from carnot.learning.implication_memory import FormulaVersion
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
)
from carnot.resolvers.gguf_cache import GGUFCacheResolver


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
PHASE = 2
EXPERIMENT_ID = "exp7372-v647-qwen-canary"
TASK_ID = "experiment_7372_v647_qwen_canary"
SCHEMA = "carnot.exp7372.v647.qwen_canary.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [MODEL_ID]
MAX_GENERATED_TOKENS = 128
MODEL_LOAD_TIMEOUT_S = 600.0
INFERENCE_WINDOW_TIMEOUT_S = 900.0
REQUEST_TIMEOUT_S = 180.0
RANDOM_SEED = {"experiment": 7_372_101, "resampling": 7_372_301}

MODULE_PATH = Path("python/carnot/experiment_7372_v647_qwen_canary.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7372_v647_qwen_canary.py")
TEST_PATH = Path("tests/python/test_experiment_7372_v647_qwen_canary.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
PRODUCER_PATH = REPO_ROOT / "results/experiment_7371_v647_proof_boundary.json"
DEVELOPMENT_FORMULAS_PATH = (
    REPO_ROOT / "results/raw/experiment_7371_v647_proof_boundary/development_formulas.json"
)
RESULT_PATH = Path("results/experiment_7372_v647_qwen_canary.json")
RAW_DIR = Path("results/raw/experiment_7372_v647_qwen_canary")
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_replay",
    "independent_reducer",
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
    "schema": "Use a versioned schema with ordinary experiment_id and milestone fields.",
    "status": "Write a terminal status only after actual work and required validation.",
    "run_date": "Use 20260917 and retain actual UTC start and completion times.",
    "preconditions_checked": "Record exact paths, producer identity, hashes, class, and resources before model work.",
    "MODEL_SPECS": "Name only unsloth/Qwen3.8-27B-GGUF and its exact cached settings.",
    "model_invoked": "Set true after any current model load or generation attempt, including failure.",
    "invocation_counts": "Separate attempted, complete, failed, cancelled, and in-flight model operations.",
    "inference_substrate": "Record actual owned native CUDA work and label historical evidence separately.",
    "inference_substrate_class": "Use the closed class that matches actual bounded generation without padding.",
    "execution_venue": "Record the measured host CUDA device; make no board-execution claim.",
    "duration_s": "Measure monotonic duration and never add sleeps to satisfy a class floor.",
    "phase_spans": "Retain measured read, build, load, generation, evaluation, validation, and write spans.",
    "random_seed": "Freeze experiment and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, formulas, protocol, and exact raw rows.",
    "source_artifact_hashes": "Hash exact producer, formula, model, server, code, and raw evidence bytes.",
    "rows": "Retain every call outcome, metric, cost, failure, and censoring disposition.",
    "sample_size_budget": "Predeclare four calls, token limits, stopping rules, and remaining work.",
    "acceptance_gate_results": "Keep expected, observed, and passed values separate for every gate.",
    "gate_check_summary": "Name the first failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Set true only when the evaluator defines truth; independent code alone is insufficient.",
    "honest_verdict": "Use complete_ for finished work and blocked_ for an unavailable prerequisite.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Set true only for a current critical independent finding.",
    "validation_receipts": "Retain command argv, environment, scope, exit, duration, and log hash for each check.",
    "repository_health": "Keep unrelated dated failures separate from current affected checks.",
    "field_principles": "Explain required fields without wrapping ordinary dictionaries or numeric scores.",
    "promotion_score": "Keep zero because this canary authorizes no rollout or publication.",
    "qwen_assignment_transport_ready_score": "Set one only for the fixed owned four-call and three-usable contract.",
    "runner_receipt": "Bind PID, start tick, lease, GGUF, server hash, CUDA device, and offload evidence.",
    "raw_call_rows": "Retain request and response paths, hashes, parsing, finish, token, and timing data.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(FIELD_PRINCIPLES)


def sha256_text(value: str) -> str:
    """Hash exact text so a response mutation changes the evidence identity."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes so producer and raw-call identities cannot drift."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the terminal record without recursively hashing its checksum."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(value))


def _utc_now() -> str:  # pragma: no cover - wall-clock evidence.
    """Record real UTC boundaries while durations use a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each boundary so long model and subprocess work stays visible."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(f"[exp7372] phase={phase} event={event}" + (f" {suffix}" if suffix else ""), flush=True)


def _span(
    spans: list[JsonDict], phase: str, phase_started: float, run_started: float, units: int
) -> None:  # pragma: no cover
    """Close one measured interval relative to the experiment start."""

    ended = time.monotonic()
    spans.append(
        {
            "phase": phase,
            "start_elapsed_s": round(phase_started - run_started, 6),
            "end_elapsed_s": round(ended - run_started, 6),
            "duration_s": round(ended - phase_started, 6),
            "completed_units": units,
            "checkpoint_at_utc": _utc_now(),
        }
    )


def _load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def schedule_row(formula_row: Mapping[str, Any], index: int) -> JsonDict:
    """Convert one sealed development formula into one deterministic request."""

    formula = deepcopy(dict(formula_row.get("formula") or {}))
    formula_id = str(formula_row.get("formula_id") or "")
    if not formula_id or not formula:
        raise ValueError("development_formula_shape")
    request = {
        "request_id": f"exp7372-canary-{index:02d}-{formula_id}",
        "formula_id": formula_id,
        "family": formula_row.get("family"),
        "n_vars": formula_row.get("n_vars"),
        "formula": formula,
        "cohort": "development_canary",
        "call_index": index,
    }
    request["prompt"] = render_assignment_prompt(request)
    return request


def build_canary_schedule(fixture: Mapping[str, Any]) -> list[JsonDict]:
    """Freeze the first four sealed development formulas before model outcomes."""

    formulas = fixture.get("development_formulas")
    if not isinstance(formulas, list):
        raise ValueError("development_formulas")
    if len(formulas) < 4:
        raise ValueError("development_formula_count")
    selected = [dict(row) for row in formulas[:4] if isinstance(row, Mapping)]
    identities = [row.get("formula_id") for row in selected]
    if len(selected) != 4 or len(set(identities)) != 4 or any(not value for value in identities):
        raise ValueError("development_formula_identity")
    return [schedule_row(row, index) for index, row in enumerate(selected)]


def render_assignment_prompt(request: Mapping[str, Any]) -> str:
    """Expose only exact formula bytes and the unrestricted assignment schema."""

    schema = {"assignments": ["signed_integer_literal", "signed_integer_literal"]}
    formula = {
        "formula_id": request.get("formula_id"),
        "version": (request.get("formula") or {}).get("version"),
        "n_vars": request.get("n_vars"),
        "source_hash": (request.get("formula") or {}).get("source_hash"),
        "clauses": (request.get("formula") or {}).get("raw_clause_order"),
    }
    return (
        "Return exactly one JSON object and no other text. "
        "The assignments list must contain two to four distinct nonzero signed integer literals. "
        "Do not return an answer label, explanation, markdown, or a full assignment unless it has at most four literals.\n"
        f"OUTPUT_SCHEMA={canonical_json(schema)}\n"
        f"FORMULA={canonical_json(formula)}"
    )


def decode_assignment(raw_reply: str, n_vars: int) -> JsonDict:
    """Parse one exact assignment object without repair or schema guessing."""

    invalid = {
        "parse_status": "invalid",
        "assignments": None,
        "schema_valid": False,
        "variable_references_valid": False,
        "response_fidelity_valid": False,
    }
    try:
        value = json.loads(raw_reply)
    except (json.JSONDecodeError, TypeError):
        return {**invalid, "parse_errors": ["json_object"]}
    if not isinstance(value, dict):
        return {**invalid, "parse_errors": ["json_object"]}
    if set(value) != {"assignments"}:
        return {**invalid, "parse_errors": ["top_level_fields"]}
    assignments = value["assignments"]
    if not isinstance(assignments, list):
        return {**invalid, "parse_errors": ["assignments_list"]}
    if not 2 <= len(assignments) <= 4:
        return {**invalid, "parse_errors": ["assignment_count"]}
    if any(isinstance(literal, bool) or not isinstance(literal, int) for literal in assignments):
        return {**invalid, "parse_errors": ["literal_type"]}
    if any(literal == 0 or abs(literal) > n_vars for literal in assignments):
        return {
            **invalid,
            "parse_errors": ["variable_reference"],
            "schema_valid": True,
        }
    if len({abs(literal) for literal in assignments}) != len(assignments):
        return {
            **invalid,
            "parse_errors": ["duplicate_variable"],
            "schema_valid": True,
        }
    return {
        "parse_status": "valid",
        "parse_errors": [],
        "assignments": list(assignments),
        "schema_valid": True,
        "variable_references_valid": True,
        "response_fidelity_valid": value["assignments"] == assignments,
    }


def formula_extendible(formula: Mapping[str, Any], assignments: Sequence[int]) -> bool:
    """Check whether one partial assignment extends to a full satisfying assignment."""

    raw_clauses = formula.get("raw_clause_order")
    clauses = raw_clauses if isinstance(raw_clauses, list) else []
    exact = FormulaVersion.from_clauses(str(formula["version"]), int(formula["n_vars"]), clauses)
    satisfiable, _assignment = exact.solve(tuple(assignments))
    return satisfiable


def _native_request_payload(prompt: str) -> JsonDict:
    """Rebuild the exact fixed request sent by the shared native runner."""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED["experiment"],
        "cache_prompt": False,
        "max_tokens": MAX_GENERATED_TOKENS,
    }


def build_call_row(
    schedule: Mapping[str, Any], response: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> JsonDict:
    """Retain one model disposition and evaluate parsing separately from SAT."""

    raw_reply = str(response.get("raw_reply") or "")
    parsed = decode_assignment(raw_reply, int(schedule["n_vars"]))
    assignments = parsed.get("assignments")
    formula_valid = (
        formula_extendible(dict(schedule["formula"]), assignments)
        if isinstance(assignments, list)
        else None
    )
    error_text = response.get("error")
    terminal_state = "request_error" if error_text else "response"
    raw_request = deepcopy(
        dict(response.get("raw_request") or _native_request_payload(str(schedule["prompt"])))
    )
    request_bytes = canonical_json(raw_request).encode("utf-8")
    return {
        "call_index": int(schedule["call_index"]),
        "request_id": schedule["request_id"],
        "formula_id": schedule["formula_id"],
        "formula_version": dict(schedule["formula"]).get("version"),
        "formula_source_hash": dict(schedule["formula"]).get("source_hash"),
        "n_vars": int(schedule["n_vars"]),
        "prompt": schedule["prompt"],
        "prompt_sha256": sha256_text(str(schedule["prompt"])),
        "raw_request": raw_request,
        "raw_request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_reply": raw_reply,
        "raw_reply_sha256": sha256_text(raw_reply),
        "raw_response": deepcopy(dict(response.get("raw_response") or {})),
        "parse_status": parsed["parse_status"],
        "parse_errors": deepcopy(parsed["parse_errors"]),
        "schema_valid": parsed["schema_valid"],
        "variable_references_valid": parsed["variable_references_valid"],
        "response_fidelity_valid": parsed["response_fidelity_valid"],
        "decoded_assignments": deepcopy(assignments),
        "asserted_literals_match_raw_response": parsed["response_fidelity_valid"],
        "usable_proposal": parsed["parse_status"] == "valid",
        "formula_valid": formula_valid,
        "attempted": True,
        "terminal_state": terminal_state,
        "error": error_text,
        "finish_reason": response.get("finish_reason"),
        "truncated": response.get("finish_reason") in {"length", "max_tokens"},
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "max_new_tokens": MAX_GENERATED_TOKENS,
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": terminal_state != "response",
        "metric": "partial_assignment_transport",
        "metric_value": int(parsed["parse_status"] == "valid"),
        "cost": {
            "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(response.get("completion_tokens", 0) or 0),
            "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        },
    }


def _identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    """Select fields that prove all calls came from one owned instance."""

    return tuple(
        identity.get(field)
        for field in (
            "pid",
            "start_time_ticks",
            "lease_id",
            "model_sha256",
            "server_executable_sha256",
            "gpu_uuid",
        )
    )


def _identity_sound(row: Mapping[str, Any]) -> bool:
    """Require ownership, CUDA proof, hashes, lease, and served-model agreement."""

    identity = row.get("runtime_identity_receipt")
    raw_response = row.get("raw_response")
    if not isinstance(identity, Mapping) or not isinstance(raw_response, Mapping):
        return False
    return bool(
        identity.get("owned_by_task") is True
        and identity.get("cuda_provenance_ok") is True
        and identity.get("pid") is not None
        and identity.get("start_time_ticks") is not None
        and identity.get("lease_id")
        and identity.get("gpu_uuid")
        and identity.get("model_sha256")
        and identity.get("server_executable_sha256")
        and runtime._served_identity_sound(identity, raw_response)
    )


def reduce_raw_calls(
    rows: Sequence[Mapping[str, Any]], load_receipt: Mapping[str, Any]
) -> JsonDict:
    """Reduce owned transport while leaving mathematical correctness separate."""

    attempted = [row for row in rows if row.get("attempted") is True]
    responses = [row for row in attempted if row.get("terminal_state") == "response"]
    failed = [row for row in attempted if row.get("terminal_state") == "request_error"]
    cancelled = [row for row in rows if row.get("terminal_state") == "cancelled"]
    in_flight = [row for row in rows if row.get("terminal_state") == "in_flight"]
    identities = [
        row.get("runtime_identity_receipt")
        for row in rows
        if isinstance(row.get("runtime_identity_receipt"), Mapping)
    ]
    identity_consistent = bool(
        len(rows) == 4
        and len(identities) == 4
        and all(_identity_sound(row) for row in rows)
        and len({_identity_key(identity) for identity in identities}) == 1
    )
    usable = sum(row.get("usable_proposal") is True for row in rows)
    valid = sum(row.get("formula_valid") is True for row in rows)
    complete_dispositions = len(responses) == 4 and not failed and not cancelled and not in_flight
    ready = int(
        load_receipt.get("completed") is True
        and identity_consistent
        and complete_dispositions
        and usable >= 3
    )
    return {
        "qwen_assignment_transport_ready_score": ready,
        "usable_proposal_count": usable,
        "formula_valid_proposal_count": valid,
        "runtime_identity_consistent": identity_consistent,
        "complete_owned_runtime_disposition_count": len(responses),
        "invocation_counts": {
            "model_loads_attempted": int(load_receipt.get("attempted") is True),
            "model_loads_completed": int(load_receipt.get("completed") is True),
            "model_loads_failed": int(load_receipt.get("failed") is True),
            "model_loads_cancelled": int(load_receipt.get("cancelled") is True),
            "model_loads_in_flight": int(load_receipt.get("in_flight") is True),
            "generation_calls_attempted": len(attempted),
            "generation_calls_completed": len(responses),
            "generation_calls_failed": len(failed),
            "generation_calls_cancelled": len(cancelled),
            "generation_calls_in_flight": len(in_flight),
            "historical_model_loads": 0,
            "historical_generation_calls": 0,
        },
    }


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep both sides of one gate so a blocked result stays exact."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate while retaining the complete failure count."""

    failures = [row for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "check_count": len(checks),
        "failed_check_count": len(failures),
        "failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "artifact_field": first.get("artifact_field")
        if first
        else "qwen_assignment_transport_ready_score",
        "expected_value": deepcopy(first.get("expected_value")) if first else 1,
        "observed_value": deepcopy(first.get("observed_value")) if first else 1,
        "passed": not failures,
    }


def producer_gate_rows(
    producer: Mapping[str, Any],
    *,
    producer_path: Path,
    producer_hash: str | None,
    formulas_hash: str | None,
    excluded: bool,
) -> list[JsonDict]:
    """Authenticate the exact proof-boundary producer before any model work."""

    declared_hash = dict(producer.get("source_artifact_hashes") or {}).get(
        "results/raw/experiment_7371_v647_proof_boundary/development_formulas.json"
    )
    fields: tuple[tuple[str, Any, Any], ...] = (
        ("experiment_id", "exp7371-v647-proof-boundary", producer.get("experiment_id")),
        ("milestone", MILESTONE, producer.get("milestone")),
        ("proof_boundary_ready_score", 1, producer.get("proof_boundary_ready_score")),
        (
            "verdict_class",
            ["positive", "circular_positive", "null"],
            producer.get("verdict_class"),
        ),
        ("flagged_adversarial", False, producer.get("flagged_adversarial")),
        ("status", "complete_*", producer.get("status")),
        ("quarantined", False, excluded),
        ("development_formulas_sha256", declared_hash, formulas_hash),
    )
    rows: list[JsonDict] = []
    for field, expected, observed in fields:
        if field == "verdict_class":
            passed = observed in expected
        elif field == "status":
            passed = isinstance(observed, str) and observed.startswith("complete")
        else:
            passed = observed == expected and expected is not None
        rows.append(
            gate_row(
                f"producer_{field}",
                producer_path.as_posix(),
                field,
                expected,
                observed,
                passed,
            )
        )
    rows.insert(
        0,
        gate_row(
            "producer_path",
            producer_path.as_posix(),
            "path",
            "readable_nonempty_json_object",
            "readable_nonempty_json_object" if producer else None,
            bool(producer and producer_hash),
        ),
    )
    rows.insert(
        1,
        gate_row(
            "producer_sha256",
            producer_path.as_posix(),
            "sha256",
            "sha256:<64 hex>",
            producer_hash,
            bool(
                producer_hash and producer_hash.startswith("sha256:") and len(producer_hash) == 71
            ),
        ),
    )
    return rows


def collect_dependency_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Check source and producer bytes without touching a GPU or loading a model."""

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
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/resolvers/gguf_cache.py"),
        Path("python/carnot/experiment_7361_v646_fresh_plan_capture.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        PRODUCER_PATH.relative_to(REPO_ROOT),
        DEVELOPMENT_FORMULAS_PATH.relative_to(REPO_ROOT),
    )
    checks: list[JsonDict] = []
    for relative in required:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                available,
            )
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7372",
            "REQ-CL-7372" if "REQ-CL-7372" in spec_text else None,
            "REQ-CL-7372" in spec_text,
        )
    )
    producer_path = root / PRODUCER_PATH.relative_to(REPO_ROOT)
    formulas_path = root / DEVELOPMENT_FORMULAS_PATH.relative_to(REPO_ROOT)
    producer = _load_object(producer_path)
    fixture = _load_object(formulas_path)
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    excluded = "experiment_id: 7372" in exclusion or "exp7372-v647-qwen-canary" in exclusion
    checks.extend(
        producer_gate_rows(
            producer,
            producer_path=producer_path,
            producer_hash=sha256_file(producer_path) if producer_path.is_file() else None,
            formulas_hash=sha256_file(formulas_path) if formulas_path.is_file() else None,
            excluded=excluded,
        )
    )
    schedule: list[JsonDict] = []
    selection_error: str | None = None
    try:
        schedule = build_canary_schedule(fixture)
    except (KeyError, TypeError, ValueError) as exc:
        selection_error = f"{type(exc).__name__}:{exc}"
    checks.append(
        gate_row(
            "four_sealed_development_formulas",
            formulas_path.as_posix(),
            "development_formulas",
            {"count": 4, "selection_error": None},
            {"count": len(schedule), "selection_error": selection_error},
            len(schedule) == 4 and selection_error is None,
        )
    )
    return checks, {"producer": producer, "fixture": fixture, "schedule": schedule}


def _runtime_preconditions(root: Path, context: JsonDict) -> list[JsonDict]:  # pragma: no cover
    """Resolve cached model, native server, and one unclaimed CUDA device."""

    checks: list[JsonDict] = []
    checks.append(
        gate_row(
            "force_live_mode",
            "process_environment",
            "CARNOT_FORCE_LIVE",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
            os.environ.get("CARNOT_FORCE_LIVE") == "1",
        )
    )
    model_path = GGUFCacheResolver().resolve(MODEL_ID, MODEL_FILENAME)
    exists = bool(model_path and model_path.is_file())
    model_hash = runtime._content_addressed_hash(model_path) if exists and model_path else None
    metadata = read_gguf_metadata(model_path) if exists and model_path else {}
    tokenizer_ok, tokenizer_detail = runtime.cpu_embedded_tokenizer_check(
        str(model_path) if model_path else ""
    )
    model_spec = {
        "hf_id": MODEL_ID,
        "filename": MODEL_FILENAME,
        "quantization": QUANTIZATION,
        "path": str(model_path.resolve()) if exists and model_path else None,
        "revision": snapshot_revision(model_path) if exists and model_path else None,
        "bytes": model_path.stat().st_size if exists and model_path else None,
        "sha256": model_hash,
        "native_tokenizer": "embedded_gguf",
        "embedded_tokenizer_loadable": tokenizer_ok,
        "embedded_tokenizer_detail": tokenizer_detail,
        "native_chat_template": metadata.get("chat_template_present") is True,
        "chat_template_sha256": metadata.get("chat_template_sha256"),
        "decoding": {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": RANDOM_SEED["experiment"],
            "repair_attempts": 0,
            "retry_budget": 0,
        },
    }
    model_ok = bool(
        exists
        and model_hash
        and model_path
        and QUANTIZATION.lower() in model_path.name.lower()
        and tokenizer_ok
        and model_spec["native_chat_template"]
    )
    checks.append(
        gate_row(
            "cached_qwen_gguf",
            "local_huggingface_cache",
            "MODEL_SPECS",
            {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "tokenizer": True,
                "chat_template": True,
            },
            {
                "model": MODEL_ID if exists else None,
                "quantization": QUANTIZATION if exists else None,
                "tokenizer": tokenizer_ok,
                "chat_template": model_spec["native_chat_template"],
            },
            model_ok,
        )
    )
    server = Path(resolve_native_llama_server())
    server_hash = sha256_file(server) if server.is_file() else None
    progress("preconditions", "before_subprocess", operation="native_runner_capabilities")
    runner_rows = runtime.lease_preflight.collect_runner_capabilities(server)
    progress("preconditions", "after_subprocess", operation="native_runner_capabilities")
    runner_errors = runtime.lease_preflight.runner_capability_errors(runner_rows)
    checks.append(
        gate_row(
            "native_cuda_runner",
            server.as_posix(),
            "runner_receipt",
            {"errors": [], "server_sha256": "sha256:<64 hex>"},
            {"errors": runner_errors, "server_sha256": server_hash},
            not runner_errors and bool(server_hash),
        )
    )
    progress("preconditions", "before_subprocess", operation="gpu_inventory")
    process_rows, query_receipts = runtime.lease_preflight.collect_gpu_process_rows()
    lease_rows = runtime.lease_preflight.scan_lease_rows(
        runtime.lease_preflight.LEASE_RUNTIME_DIR, process_rows
    )
    classified = runtime.lease_preflight.classify_process_rows(
        process_rows, lease_rows, current_task_id=TASK_ID
    )
    decision = runtime.lease_preflight.readiness_decision(
        classified,
        lease_rows,
        [
            {
                "repository": MODEL_ID,
                "filename": MODEL_FILENAME,
                "path": str(model_path) if model_path else None,
                "real_path": str(model_path.resolve()) if exists and model_path else None,
                "revision": model_spec["revision"],
                "bytes": model_spec["bytes"],
                "sha256": model_hash,
                "hash_source": "content_addressed_cache_target" if model_hash else None,
                "weights_opened": False,
                "valid": model_ok,
            }
        ],
        runner_rows,
    )
    progress("preconditions", "after_subprocess", operation="gpu_inventory")
    query_ok = all(row.get("returncode") == 0 for row in query_receipts)
    available = list(decision.get("available_gpu_uuids", []))
    checks.append(
        gate_row(
            "cuda_capacity_without_displacement",
            "nvidia-smi_and_gpu_lease_journal",
            "runner_receipt",
            {"query_ok": True, "minimum_available_gpu_count": 1, "conflicts": []},
            {
                "query_ok": query_ok,
                "available_gpu_uuids": available,
                "conflicts": decision.get("conflicting_processes", []),
                "conflicting_lease_ids": decision.get("conflicting_lease_ids", []),
            },
            query_ok and bool(available),
        )
    )
    context.update(
        {
            "model_path": model_path,
            "model_spec": model_spec,
            "server_path": server,
            "server_sha256": server_hash,
            "runner_capabilities": deepcopy(runner_rows),
            "process_rows": classified,
            "lease_rows": lease_rows,
            "gpu_query_receipts": query_receipts,
            "available_gpu_uuids": available,
        }
    )
    return checks


@contextmanager
def _shared_runtime_settings() -> Iterator[None]:  # pragma: no cover
    """Apply this canary's fixed schedule to the shipped owned runner."""

    values = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "INFERENCE_WINDOW_TIMEOUT_S": INFERENCE_WINDOW_TIMEOUT_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": RANDOM_SEED["experiment"],
            "evaluation": RANDOM_SEED["experiment"],
            "resampling": RANDOM_SEED["resampling"],
        },
        "render_public_prompt": render_assignment_prompt,
        "progress": progress,
    }
    previous = {name: getattr(runtime, name) for name in values}
    try:
        for name, value in values.items():
            setattr(runtime, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(runtime, name, value)


def _capture(context: Mapping[str, Any], raw_dir: Path) -> JsonDict:  # pragma: no cover
    """Run all four requests through one task-owned native model instance."""

    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = deepcopy(list(context["schedule"]))
    with _shared_runtime_settings():
        capture = runtime._live_capture(runtime_context, raw_dir / "owned_runtime")
    identity = deepcopy(dict(capture.get("runtime_identity") or {}))
    identity["server_executable_path"] = str(context.get("server_path") or "")
    identity["server_executable_sha256"] = context.get("server_sha256")
    identity["native_tokenizer"] = dict(context.get("model_spec") or {}).get("native_tokenizer")
    identity["native_chat_template"] = dict(context.get("model_spec") or {}).get(
        "native_chat_template"
    )
    runtime_rows = list(capture.get("rows") or [])
    rows: list[JsonDict] = []
    for index, schedule in enumerate(context["schedule"]):
        source = dict(runtime_rows[index]) if index < len(runtime_rows) else {}
        if source.get("terminal_state") == "cancelled":
            row = {
                **build_call_row(
                    schedule,
                    {
                        "raw_reply": "",
                        "raw_response": source.get("raw_response") or {},
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                        "finish_reason": None,
                        "error": source.get("error") or capture.get("runtime_error"),
                    },
                    identity,
                ),
                "attempted": False,
                "terminal_state": "cancelled",
                "censored": True,
            }
        else:
            row = build_call_row(
                schedule,
                {
                    "raw_request": _native_request_payload(str(schedule["prompt"])),
                    "raw_reply": source.get("raw_reply"),
                    "raw_response": source.get("raw_response") or {},
                    "prompt_tokens": source.get("prompt_tokens"),
                    "completion_tokens": source.get("completion_tokens"),
                    "latency_s": source.get("latency_s"),
                    "finish_reason": source.get("finish_reason"),
                    "error": source.get("error"),
                },
                identity,
            )
        rows.append(row)
        progress(
            "generation",
            "checkpoint",
            completed=index + 1,
            total=4,
            terminal_state=row["terminal_state"],
            parse_status=row["parse_status"],
        )
    capture["rows"] = rows
    capture["runtime_identity"] = identity
    return capture


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Derive the exact affected command set through the Exp7358 boundary."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad, missing, duplicate, or path-invalid affected commands."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _run_affected_validation(
    root: Path, raw_dir: Path
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Execute the Exp7358 plan through the Exp7303 streaming runner."""

    private = Path("/tmp/carnot-exp7372-v647-validation")
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        return [], {"passed": False, "plan_errors": plan_errors}
    planned = [
        PlannedCommand(spec=command, category="required_affected_validation", required=True)
        for command in commands
    ]
    receipts = run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected", heartbeat_s=60.0
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, receipts)
    return receipts, {**reduced, "plan_errors": []}


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:  # pragma: no cover
    """Build the cold replay and independent strict-reader commands."""

    python = str(root / ".venv/bin/python")
    replay = (
        "import json,pathlib;"
        "from carnot.experiment_7372_v647_qwen_canary import independent_reduce_artifact;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "e=independent_reduce_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    return [
        CommandSpec(
            "declared_entrypoint_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate",
        ),
        CommandSpec("independent_reducer", (python, "-u", "-c", replay), "candidate"),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate",
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
            "candidate",
        ),
    ]


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful real receipt for each named check."""

    return all(
        sum(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is False
            for row in receipts
        )
        == 1
        for name in names
    )


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Reload exact call files and rebuild transport and formula counts."""

    rows = artifact.get("rows")
    manifests = artifact.get("raw_call_rows")
    if not isinstance(rows, list) or not isinstance(manifests, list) or len(manifests) != 4:
        return ["raw_evidence_unavailable"]
    errors: list[str] = []
    loaded: list[JsonDict] = []
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            errors.append(f"raw_call_manifest_invalid:{index}")
            continue
        path = Path(str(manifest.get("raw_path") or ""))
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append(f"raw_call_path_missing:{index}")
            continue
        if sha256_file(path) != manifest.get("raw_sha256"):
            errors.append(f"raw_call_hash_mismatch:{index}")
            continue
        value = _load_object(path)
        loaded.append(value)
        expected = {
            key: deepcopy(value)
            for key, value in manifest.items()
            if key not in {"raw_path", "raw_sha256"}
        }
        if value != expected or index >= len(rows) or value != rows[index]:
            errors.append(f"raw_call_row_mismatch:{index}")
    if errors:
        return errors
    reduced = reduce_raw_calls(loaded, dict(artifact.get("load_receipt") or {}))
    for field in (
        "qwen_assignment_transport_ready_score",
        "usable_proposal_count",
        "formula_valid_proposal_count",
        "runtime_identity_consistent",
        "complete_owned_runtime_disposition_count",
        "invocation_counts",
    ):
        observed = artifact.get(field)
        if (
            field == "qwen_assignment_transport_ready_score"
            and artifact.get("verdict_class") == "disqualified"
        ):
            observed = artifact.get("observed_qwen_assignment_transport_ready_score")
        if observed != reduced[field]:
            errors.append(f"{field}_mismatch")
    return errors


def _acceptance_gates(
    preconditions: Sequence[Mapping[str, Any]],
    reduced: Mapping[str, Any],
    affected_ok: bool,
    terminal_ok: bool,
    flagged: bool,
    measured_duration_s: float,
) -> JsonDict:  # pragma: no cover
    """Expose prerequisites, transport, timing, checks, and promotion separately."""

    usable = int(reduced.get("usable_proposal_count", 0) or 0)
    gates = {
        "preconditions": (True, all(row.get("passed") is True for row in preconditions), True),
        "four_owned_runtime_dispositions": (
            4,
            reduced.get("complete_owned_runtime_disposition_count"),
            reduced.get("complete_owned_runtime_disposition_count") == 4,
        ),
        "usable_assignment_proposals": (">=3", usable, usable >= 3),
        "assignment_transport": (
            1,
            reduced.get("qwen_assignment_transport_ready_score"),
            reduced.get("qwen_assignment_transport_ready_score") == 1,
        ),
        "bounded_generation_floor": (">=10.0", measured_duration_s, measured_duration_s >= 10.0),
        "affected_validation": (True, affected_ok, affected_ok),
        "terminal_validation": (True, terminal_ok, terminal_ok),
        "adversarial_clear": (False, flagged, not flagged),
        "promotion": (1, 0, False),
    }
    principles = {
        "preconditions": "All exact producer and resource checks must pass before generation.",
        "four_owned_runtime_dispositions": "All four calls must finish on the one owned runtime.",
        "usable_assignment_proposals": "At least three responses must pass schema, variable, and fidelity checks.",
        "assignment_transport": "Formula validity is measured separately and does not define transport.",
        "bounded_generation_floor": "The actual bounded model workload must meet its class floor without padding.",
        "affected_validation": "Every command in the exact Exp7358 affected plan must pass.",
        "terminal_validation": "Cold replay, independent reduction, and both strict readers must pass.",
        "adversarial_clear": "A current critical independent finding prevents readiness.",
        "promotion": "This transport canary never authorizes rollout or publication.",
    }
    return {
        name: {
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principles[name],
        }
        for name, (expected, observed, passed) in gates.items()
    }


def _zero_counts() -> JsonDict:
    """Provide an explicit no-inference counter set for blocked output."""

    return {
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
    }


def _base_artifact(run_date: str, started_at: str) -> JsonDict:  # pragma: no cover
    """Create a complete blocked shape before any external check can fail."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_not_started",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_settings": {},
        "model_invoked": False,
        "invocation_counts": _zero_counts(),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 4,
            "remaining_units": 4,
            "max_new_tokens_per_unit": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "inference_window_timeout_s": INFERENCE_WINDOW_TIMEOUT_S,
            "stopping_rule": "four terminal calls or the fixed deadline; no repair, retry, or replacement",
        },
        "acceptance_gate_results": {},
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_not_started",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_checked",
            "historical_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "promotion_score": 0,
        "qwen_assignment_transport_ready_score": 0,
        "observed_qwen_assignment_transport_ready_score": 0,
        "usable_proposal_count": 0,
        "formula_valid_proposal_count": 0,
        "runtime_identity_consistent": False,
        "complete_owned_runtime_disposition_count": 0,
        "runner_receipt": {},
        "raw_call_rows": [],
        "load_receipt": {
            "attempted": False,
            "completed": False,
            "failed": False,
            "cancelled": False,
            "in_flight": False,
            "error": None,
        },
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "cold_artifact_replay"],
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
    }


def _source_hashes(
    root: Path, context: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    """Bind instructions, producers, code, model, server, and raw calls."""

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
        PRODUCER_PATH.relative_to(REPO_ROOT),
        DEVELOPMENT_FORMULAS_PATH.relative_to(REPO_ROOT),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/resolvers/gguf_cache.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
    )
    hashes = {path.as_posix(): sha256_file(root / path) for path in paths}
    model_spec = dict(context.get("model_spec") or {})
    if model_spec.get("path") and model_spec.get("sha256"):
        hashes[str(model_spec["path"])] = str(model_spec["sha256"])
    if context.get("server_path") and context.get("server_sha256"):
        hashes[str(context["server_path"])] = str(context["server_sha256"])
    for row in raw_rows:
        hashes[str(row["raw_path"])] = str(row["raw_sha256"])
    return hashes


def validate_artifact(value: object) -> list[str]:  # pragma: no cover - cold CLI path.
    """Reject identity, raw evidence, score, class, and checksum drift."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(value))
    errors.extend(f"missing_required_field:{field}" for field in missing)
    if errors:
        return errors
    if value.get("schema") != SCHEMA or value.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if value.get("milestone") != MILESTONE or value.get("run_date") != RUN_DATE:
        errors.append("lifecycle_invalid")
    if value.get("MODEL_SPECS") != MODEL_SPECS or value.get("promotion_score") != 0:
        errors.append("model_or_promotion_invalid")
    if value.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    counts = dict(value.get("invocation_counts") or {})
    if bool(value.get("model_invoked")) != (counts.get("model_loads_attempted", 0) > 0):
        errors.append("model_invoked_mismatch")
    if value.get("model_invoked"):
        errors.extend(independent_reduce_artifact(value))
        if value.get("inference_substrate_class") != "model_bounded_generation":
            errors.append("substrate_class_invalid")
        if float(value.get("duration_s", 0.0) or 0.0) < 10.0:
            errors.append("bounded_generation_duration_floor")
    elif value.get("verdict_class") == "blocked":
        if value.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
    if (
        value.get("verdict_class") in {"blocked", "disqualified"}
        and value.get("qwen_assignment_transport_ready_score") != 0
    ):
        errors.append("failed_readiness_invalid")
    if (
        value.get("verdict_class") in {"positive", "circular_positive"}
        and value.get("qwen_assignment_transport_ready_score") != 1
    ):
        errors.append("positive_readiness_invalid")
    if value.get("reproducibility_checksum") != artifact_checksum(value):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:  # pragma: no cover
    """Use the shipped fsync and rename helper for one complete JSON object."""

    runtime._atomic_json(path, value)


def _write_blocked(
    artifact: JsonDict,
    checks: Sequence[Mapping[str, Any]],
    output: Path,
    started: float,
) -> JsonDict:  # pragma: no cover
    """Publish external absence without a success-shaped model record."""

    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition_failed",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "completed_at_utc": _utc_now(),
            "duration_s": round(time.monotonic() - started, 6),
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary.get('failed_check') or 'unknown_precondition'}",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    _atomic_json(output, artifact)
    return artifact


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - live orchestration.
    """Authenticate, capture, reduce, validate, and atomically publish once."""

    started = time.monotonic()
    started_at = _utc_now()
    output = output_path or root / RESULT_PATH
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifact = _base_artifact(run_date, started_at)
    spans: list[JsonDict] = []

    read_started = time.monotonic()
    progress("preconditions", "start")
    checks, context = collect_dependency_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date", "execution_contract", "run_date", RUN_DATE, run_date, run_date == RUN_DATE
        ),
    )
    _span(spans, "read", read_started, started, len(checks))
    if any(row.get("passed") is not True for row in checks):
        artifact["phase_spans"] = spans
        return _write_blocked(artifact, checks, output, started)

    build_started = time.monotonic()
    progress("preconditions", "runtime_start")
    runtime_checks = _runtime_preconditions(root, context)
    checks.extend(runtime_checks)
    progress("preconditions", "runtime_complete", passed=all(row["passed"] for row in checks))
    _span(spans, "build", build_started, started, len(runtime_checks))
    if any(row.get("passed") is not True for row in checks):
        artifact["phase_spans"] = spans
        return _write_blocked(artifact, checks, output, started)

    inference_started = time.monotonic()
    progress("inference", "before_model_load_and_generation", calls=4)
    capture = _capture(context, raw_dir)
    progress("inference", "after_model_load_and_generation", calls=len(capture["rows"]))
    inference_duration = time.monotonic() - inference_started
    for runtime_span in capture.get("phase_spans") or []:
        spans.append(deepcopy(dict(runtime_span)))

    evaluation_started = time.monotonic()
    raw_rows: list[JsonDict] = []
    rows = list(capture["rows"])
    for index, row in enumerate(rows):
        raw_path = raw_dir / f"call_{index:02d}.json"
        _atomic_json(raw_path, row)
        raw_rows.append(
            {
                **deepcopy(dict(row)),
                "raw_path": raw_path.relative_to(root).as_posix(),
                "raw_sha256": sha256_file(raw_path),
            }
        )
    reduced = reduce_raw_calls(rows, dict(capture["load_receipt"]))
    _span(spans, "evaluate", evaluation_started, started, len(rows))

    validate_started = time.monotonic()
    progress("validation", "before_affected_commands")
    affected_receipts, affected = _run_affected_validation(root, raw_dir)
    affected_ok = bool(affected.get("passed"))
    progress("validation", "after_affected_commands", passed=affected_ok)

    artifact.update(
        {
            "status": "complete_assignment_transport_measured",
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "model_settings": deepcopy(context["model_spec"]),
            "model_invoked": capture["load_receipt"].get("attempted") is True,
            "invocation_counts": deepcopy(reduced["invocation_counts"]),
            "inference_substrate": "owned_native_cuda_llama_cpp",
            "inference_substrate_class": "model_bounded_generation",
            "execution_venue": "host",
            "phase_spans": spans,
            "rows": deepcopy(rows),
            "raw_call_rows": raw_rows,
            "load_receipt": deepcopy(capture["load_receipt"]),
            "runner_receipt": {
                **deepcopy(dict(capture.get("runtime_identity") or {})),
                "gpu_provenance": deepcopy(
                    dict(capture.get("gpu_receipts") or {}).get("provenance") or {}
                ),
                "cleanup": deepcopy(dict(capture.get("gpu_receipts") or {}).get("cleanup") or {}),
                "lease_release": deepcopy(
                    dict(capture.get("gpu_receipts") or {}).get("lease_release") or {}
                ),
                "host": os.uname().nodename,
                "board_execution": False,
                "single_model_instance": True,
            },
            "qwen_assignment_transport_ready_score": int(
                reduced["qwen_assignment_transport_ready_score"]
            ),
            "observed_qwen_assignment_transport_ready_score": int(
                reduced["qwen_assignment_transport_ready_score"]
            ),
            "usable_proposal_count": int(reduced["usable_proposal_count"]),
            "formula_valid_proposal_count": int(reduced["formula_valid_proposal_count"]),
            "runtime_identity_consistent": bool(reduced["runtime_identity_consistent"]),
            "complete_owned_runtime_disposition_count": int(
                reduced["complete_owned_runtime_disposition_count"]
            ),
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected_ok else "required_checks_failed",
                "historical_failures": [],
                "affects_required_checks": not affected_ok,
                "affected_reduction": affected,
            },
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted_units": reduced["invocation_counts"]["generation_calls_attempted"],
                "completed_units": reduced["invocation_counts"]["generation_calls_completed"],
                "censored_units": sum(row.get("censored") is True for row in rows),
                "remaining_units": 0,
            },
        }
    )
    artifact["source_artifact_hashes"] = _source_hashes(root, context, raw_rows)
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["flagged_adversarial"] = False
    provisional_ready = int(reduced["qwen_assignment_transport_ready_score"])
    artifact["verdict_class"] = "circular_positive" if provisional_ready else "null"
    artifact["honest_verdict"] = (
        f"complete_circular_positive_qwen_assignment_transport_ready_{artifact['usable_proposal_count']}_of_4_usable"
        if provisional_ready
        else f"complete_null_qwen_assignment_transport_not_ready_{artifact['usable_proposal_count']}_of_4_usable"
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, reduced, affected_ok, False, False, inference_duration
    )
    artifact["gate_check_summary"] = gate_check_summary(checks)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured-terminal-candidate.json"
    _atomic_json(candidate, artifact)

    progress("validation", "before_terminal_commands", candidate=candidate)
    terminal_receipts = run_commands(
        root, _terminal_commands(root, candidate), log_dir=raw_dir / "validation/terminal"
    )
    progress("validation", "after_terminal_commands")
    terminal_ok = _receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    _span(
        spans,
        "validate",
        validate_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    artifact["phase_spans"] = spans
    artifact["validation_receipts"] = [*affected_receipts, *terminal_receipts]
    artifact["flagged_adversarial"] = flagged
    disqualified = not (affected_ok and terminal_ok and independent_ok and not flagged)
    if disqualified:
        artifact.update(
            {
                "status": "complete_required_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_qwen_canary_required_check_failed",
                "qwen_assignment_transport_ready_score": 0,
            }
        )
    else:
        artifact["status"] = (
            "complete_assignment_transport_ready"
            if provisional_ready
            else "complete_assignment_transport_null"
        )
        artifact["verdict_class"] = "circular_positive" if provisional_ready else "null"
    artifact["acceptance_gate_results"] = _acceptance_gates(
        checks, reduced, affected_ok, terminal_ok and independent_ok, flagged, inference_duration
    )
    gate_rows = [
        gate_row(
            name,
            EXPERIMENT_ID,
            "acceptance_gate_results",
            value["expected"],
            value["observed"],
            value["passed"],
        )
        for name, value in artifact["acceptance_gate_results"].items()
        if name != "promotion"
    ]
    artifact["gate_check_summary"] = gate_check_summary([*checks, *gate_rows])
    write_started = time.monotonic()
    artifact["completed_at_utc"] = _utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    _span(spans, "write", write_started, started, 1)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "qwen_assignment_transport_ready_score": 0,
                "internal_validation_errors": validation_errors,
            }
        )
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    progress("write", "start", artifact=output)
    _atomic_json(output, artifact)
    progress("write", "complete", artifact=output, verdict=artifact["honest_verdict"])
    return artifact


def _date_argument(value: str) -> str:
    """Reject accidental execution outside the fixed V647 date."""

    if value != RUN_DATE:
        raise ValueError(f"date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run live capture or validate one task-owned candidate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    progress("entrypoint", "start", date=args.date)
    if args.validate is not None:
        errors = validate_artifact(_load_object(args.validate))
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "qwen_assignment_transport_ready_score": result[
                    "qwen_assignment_transport_ready_score"
                ],
                "usable_proposal_count": result["usable_proposal_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
