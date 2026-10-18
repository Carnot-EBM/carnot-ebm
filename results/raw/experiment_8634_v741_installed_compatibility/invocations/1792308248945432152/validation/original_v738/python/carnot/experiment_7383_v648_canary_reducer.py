"""Repair assignment receipt reduction from immutable historical evidence.

This module replays four old model calls and two proof producers. It does not
load a model or turn historical transport evidence into current science.

Spec refs: REQ-REPORT-7383 and SCENARIO-REPORT-7383-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot import experiment_7370_v647_proof_memory as proof_memory
from carnot import experiment_7371_v647_proof_boundary as proof_boundary
from carnot.experiment_7372_v647_qwen_canary import (
    decode_assignment,
    formula_extendible,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    atomic_json,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.648"
PHASE = 1
EXPERIMENT_ID = "exp7383-canary-reducer"
SCHEMA = "carnot.exp7383.v648.canary_reducer.v1"
EXPECTED_DECODING_SEED = 7_372_101

MODULE_PATH = Path("python/carnot/experiment_7383_v648_canary_reducer.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7383_v648_canary_reducer.py")
TEST_PATH = Path("tests/python/test_experiment_7383_v648_canary_reducer.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7383_v648_canary_reducer.json")
RAW_DIR = Path("results/raw/experiment_7383_v648_canary_reducer")

CANARY_PATH = Path("results/experiment_7372_v647_qwen_canary.json")
CANARY_CANDIDATE_PATH = Path(
    "results/raw/experiment_7372_v647_qwen_canary/measured-terminal-candidate.json"
)
CANARY_RAW_DIR = Path("results/raw/experiment_7372_v647_qwen_canary")
CANARY_CALL_PATHS = tuple(CANARY_RAW_DIR / f"call_{index:02d}.json" for index in range(4))
NATIVE_CALL_PATHS = tuple(
    CANARY_RAW_DIR / f"owned_runtime/call_{index:02d}.json" for index in range(4)
)
NATIVE_LOG_PATH = CANARY_RAW_DIR / "owned_runtime/llama_server_2853297611839229850.log"
HISTORICAL_TERMINAL_LOG_PATHS = (
    CANARY_RAW_DIR / "validation/terminal/00_declared_entrypoint_replay.log",
    CANARY_RAW_DIR / "validation/terminal/01_independent_reducer.log",
    CANARY_RAW_DIR / "validation/terminal/02_adversarial_verify.log",
    CANARY_RAW_DIR / "validation/terminal/03_verdict_row_consistency_strict.log",
)
PROOF_MEMORY_PATH = Path("results/experiment_7370_v647_proof_memory.json")
PROOF_BOUNDARY_PATH = Path("results/experiment_7371_v647_proof_boundary.json")
SEALED_MANIFEST_PATH = Path("data/v647_implication_stream_manifest.json")
DEVELOPMENT_FORMULAS_PATH = Path(
    "results/raw/experiment_7371_v647_proof_boundary/development_formulas.json"
)
PROOF_BOUNDARY_RAW_PATHS = (
    DEVELOPMENT_FORMULAS_PATH,
    Path("results/raw/experiment_7371_v647_proof_boundary/evaluation_streams.json"),
    Path("results/raw/experiment_7371_v647_proof_boundary/live_proposal_requests.json"),
    Path("results/raw/experiment_7371_v647_proof_boundary/authority_attacks.json"),
    Path("results/raw/experiment_7371_v647_proof_boundary/synthetic_evidence.json"),
)

SOURCE_CODE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7372_v647_qwen_canary.py"),
    Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
    Path("python/carnot/experiment_7370_v647_proof_memory.py"),
    Path("python/carnot/learning/implication_memory.py"),
    SPEC_PATH,
    Path("openspec/capabilities/constraint-verification/spec.md"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
IMMUTABLE_INPUT_PATHS = (
    CANARY_PATH,
    CANARY_CANDIDATE_PATH,
    *CANARY_CALL_PATHS,
    *NATIVE_CALL_PATHS,
    NATIVE_LOG_PATH,
    *HISTORICAL_TERMINAL_LOG_PATHS,
    PROOF_MEMORY_PATH,
    PROOF_BOUNDARY_PATH,
    SEALED_MANIFEST_PATH,
    *PROOF_BOUNDARY_RAW_PATHS,
)
INPUT_PATHS = tuple(dict.fromkeys((*SOURCE_CODE_PATHS, *IMMUTABLE_INPUT_PATHS)))

ZERO_INVOCATION_COUNTS = {
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
}
RANDOM_SEED = {
    "experiment": 7_383_202_609_18,
    "historical_decoding": EXPECTED_DECODING_SEED,
    "historical_resampling": 7_372_301,
}
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

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Versioned schema with ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and required validation; no success-shaped bootstrap artifact.",
    "run_date": "Use 20260918 plus actual start/end UTC timestamps.",
    "preconditions_checked": "Exact paths, producer identity/hash/class and resource checks before dependent work.",
    "MODEL_SPECS": "Use an empty list because this task performs no current LLM work.",
    "model_invoked": "False because this task attempts no current model load or generation.",
    "invocation_counts": "Keep all current LLM counts at zero; historical calls stay separate.",
    "inference_substrate": "Record actual host CPU aggregation and keep historical CUDA provenance labeled.",
    "inference_substrate_class": "Use aggregation for the actual current computation without padding.",
    "execution_venue": "Use the closed host string and keep device detail in inference_substrate.",
    "duration_s": "Measure monotonic duration and never invent time or sleep to meet a floor.",
    "phase_spans": "Measure read, build, load, generate, evaluate, validate, and write boundaries.",
    "random_seed": "Freeze current and retained historical seeds for reproducible reduction.",
    "reproducibility_checksum": "Bind exact code, settings, sources, raw rows, and decisions.",
    "source_artifact_hashes": "Preserve exact producer and immutable evidence byte hashes.",
    "rows": "Retain each historical call outcome, metric, cost, failure, and censoring disposition.",
    "sample_size_budget": "Predeclare four replay units and stop without retry or replacement.",
    "acceptance_gate_results": "Keep expected, observed, operator, and passed separate for every gate.",
    "gate_check_summary": "Name the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "True because formal SAT evaluation defines assignment truth.",
    "honest_verdict": "Use complete_ for finished scope and blocked_ for unavailable external input.",
    "verdict_class": "Use the closed verdict class without turning completion into efficacy.",
    "flagged_adversarial": "A critical current finding excludes this producer from readiness.",
    "validation_receipts": "Retain executed argv, environment, scope, exit, duration, and log hash.",
    "repository_health": "Keep unrelated dated observations separate from required affected checks.",
    "field_principles": "Explain every output field without wrapping ordinary values.",
    "promotion_score": "Always zero because this task authorizes no rollout, weight change, or publication.",
    "assignment_reducer_ready_score": "One only after corrected reduction, four authentic calls, and terminal readers agree.",
    "proof_boundary_replay_ready_score": "One only while original proof checks and sealed manifest bytes remain valid.",
    "historical_model_inputs": "Keep original model, code, raw hashes, verdict, and flag outside current inference.",
    "discrepancy_rows": "Show original and independently reduced values with explicit comparison operators.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)


def utc_now() -> str:  # pragma: no cover - wall-clock evidence.
    """Return one real UTC boundary while durations use a monotonic clock."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Print each phase and long-operation boundary immediately."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7383] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_text(value: str) -> str:
    """Hash exact text so any response mutation changes its identity."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    """Hash the canonical compact JSON used by the historical request runner."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes without loading a large evidence file at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_object(path: Path) -> JsonDict:
    """Load one JSON object and return an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete record without recursively hashing the checksum field."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return hash_json(value)


def compare(operator: str, observed: Any, expected: Any) -> bool:
    """Apply one explicit closed comparison instead of inferring it from prose."""

    if operator == "==":
        return observed == expected
    if operator == ">=":
        return isinstance(observed, (int, float)) and observed >= expected
    if operator == "is":
        return observed is expected
    raise ValueError(f"unsupported_comparison_operator:{operator}")


def _gate(
    check: str,
    expected: Any,
    operator: str,
    observed: Any,
    *,
    category: str,
    principle: str,
    upstream: str = EXPERIMENT_ID,
    artifact_field: str | None = None,
) -> JsonDict:
    """Keep both operands and the operator beside every gate decision."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field or check,
        "expected": deepcopy(expected),
        "operator": operator,
        "observed": deepcopy(observed),
        "passed": compare(operator, observed, expected),
        "principle": principle,
    }


def build_receipt_gate_rows(
    *,
    usable_count: int,
    promotion_score: int,
    execution_venue: object,
    verifier_is_oracle: bool,
    verdict_class: str,
) -> list[JsonDict]:
    """Build the four gates implicated by the historical receipt defects."""

    oracle_expected = "circular_positive" if verifier_is_oracle else verdict_class
    return [
        _gate(
            "usable_assignment_proposals",
            3,
            ">=",
            usable_count,
            category="completion",
            artifact_field="usable_proposal_count",
            principle="At least three usable responses pass; a fourth response cannot fail the floor.",
        ),
        _gate(
            "promotion_forbidden",
            0,
            "==",
            promotion_score,
            category="promotion",
            artifact_field="promotion_score",
            principle="A diagnostic receipt never authorizes rollout or publication.",
        ),
        _gate(
            "execution_venue_closed",
            "host",
            "==",
            execution_venue,
            category="safety",
            artifact_field="execution_venue",
            principle="The closed venue field remains machine-comparable.",
        ),
        _gate(
            "oracle_verdict_class",
            oracle_expected,
            "==",
            verdict_class,
            category="safety",
            artifact_field="verdict_class",
            principle="An oracle-positive claim carries its circularity in the verdict class.",
        ),
    ]


def collect_preconditions(repo_root: Path) -> tuple[list[JsonDict], dict[str, str]]:
    """Authenticate every required source before any dependent reduction."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "artifact_field": "bytes",
                "expected_value": "readable_nonempty_bytes",
                "observed_value": "readable_nonempty_bytes" if available else "missing_or_empty",
                "passed": available,
                "blocking": True,
            }
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    requirement_present = "REQ-REPORT-7383" in spec_text
    checks.append(
        {
            "check": "driving_requirement",
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-*",
            "expected_value": "REQ-REPORT-7383",
            "observed_value": "REQ-REPORT-7383" if requirement_present else "missing",
            "passed": requirement_present,
            "blocking": True,
        }
    )

    canary = _load_object(root / CANARY_PATH)
    expected_canary = {
        "experiment_id": "exp7372-v647-qwen-canary",
        "verdict_class": "disqualified",
        "flagged_adversarial": True,
        "inference_substrate_class": "model_bounded_generation",
    }
    observed_canary = {field: canary.get(field) for field in expected_canary}
    checks.append(
        {
            "check": "historical_canary_quarantine",
            "upstream": CANARY_PATH.as_posix(),
            "artifact_field": "identity/class/flag/substrate",
            "expected_value": expected_canary,
            "observed_value": observed_canary if canary else "missing",
            "passed": observed_canary == expected_canary,
            "blocking": True,
            "historical_only": True,
            "authorizes_current_readiness": False,
        }
    )

    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    retired = "experiment_id: 7383" in exclusion_text or "experiment_id: exp7383" in exclusion_text
    checks.append(
        {
            "check": "current_task_not_retired",
            "upstream": "ops/exclusion_manifest.yaml",
            "artifact_field": EXPERIMENT_ID,
            "expected_value": False,
            "observed_value": retired,
            "passed": not retired,
            "blocking": True,
        }
    )
    return checks, hashes


def load_assignment_evidence(repo_root: Path) -> tuple[list[JsonDict], list[JsonDict]]:
    """Load the four immutable calls and their public formula definitions."""

    root = repo_root.resolve()
    calls = [_load_object(root / path) for path in CANARY_CALL_PATHS]
    fixture = _load_object(root / DEVELOPMENT_FORMULAS_PATH)
    formulas = fixture.get("development_formulas")
    return calls, [dict(row) for row in formulas or [] if isinstance(row, Mapping)]


def _response_content(row: Mapping[str, Any]) -> object:
    """Read the exact assistant content from one native response envelope."""

    try:
        return row["raw_response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _runtime_identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    """Select immutable fields that identify one owned native server."""

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


def _runtime_identity_valid(row: Mapping[str, Any]) -> bool:
    """Check ownership and served-model identity without trusting old readiness."""

    identity = row.get("runtime_identity_receipt")
    response = row.get("raw_response")
    if not isinstance(identity, Mapping) or not isinstance(response, Mapping):
        return False
    required = (
        "pid",
        "start_time_ticks",
        "lease_id",
        "model_sha256",
        "server_executable_sha256",
        "gpu_uuid",
    )
    return bool(
        identity.get("owned_by_task") is True
        and identity.get("owned_by_current_uid") is True
        and identity.get("cuda_provenance_ok") is True
        and all(identity.get(field) for field in required)
        and response.get("model") == identity.get("served_model")
    )


def reduce_assignment_receipts(
    calls: Sequence[Mapping[str, Any]], formulas: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Independently reduce four raw calls with integrity and SAT kept separate."""

    formulas_by_id = {str(row.get("formula_id")): dict(row) for row in formulas}
    errors: list[str] = []
    rows: list[JsonDict] = []
    identities: list[tuple[Any, ...]] = []
    if len(calls) != 4:
        errors.append("raw_call_count")
    for index, source in enumerate(calls):
        row = dict(source)
        row_errors: list[str] = []
        raw_request = row.get("raw_request")
        if not isinstance(raw_request, Mapping):
            row_errors.append(f"raw_request_missing:{index}")
            raw_request = {}
        request_hash_valid = hash_json(raw_request) == row.get("raw_request_sha256")
        if not request_hash_valid:
            row_errors.append(f"request_hash_mismatch:{index}")
        request_bytes = json.dumps(raw_request, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        request_bytes_valid = base64.b64encode(request_bytes).decode("ascii") == row.get(
            "raw_request_bytes_b64"
        )
        if not request_bytes_valid:
            row_errors.append(f"request_bytes_mismatch:{index}")
        seed_valid = raw_request.get("seed") == EXPECTED_DECODING_SEED
        if not seed_valid:
            row_errors.append(f"decoding_seed_mismatch:{index}")

        raw_reply = row.get("raw_reply")
        response_hash_valid = isinstance(raw_reply, str) and sha256_text(raw_reply) == row.get(
            "raw_reply_sha256"
        )
        if not response_hash_valid:
            row_errors.append(f"response_hash_mismatch:{index}")
        n_vars = row.get("n_vars")
        parsed = (
            decode_assignment(raw_reply, n_vars)
            if isinstance(n_vars, int)
            else {
                "parse_status": "invalid",
                "parse_errors": ["n_vars"],
                "assignments": None,
                "schema_valid": False,
                "variable_references_valid": False,
                "response_fidelity_valid": False,
            }
        )
        if parsed.get("variable_references_valid") is not True:
            row_errors.append(f"variable_reference_invalid:{index}")
        response_fidelity = bool(
            parsed.get("parse_status") == "valid"
            and _response_content(row) == raw_reply
            and row.get("decoded_assignments") == parsed.get("assignments")
        )
        if not response_fidelity:
            row_errors.append(f"response_fidelity_invalid:{index}")

        formula_row = formulas_by_id.get(str(row.get("formula_id")))
        formula = dict(formula_row.get("formula") or {}) if formula_row else {}
        formula_identity_valid = bool(
            formula
            and formula.get("version") == row.get("formula_version")
            and formula.get("source_hash") == row.get("formula_source_hash")
            and formula.get("n_vars") == n_vars
        )
        if not formula_identity_valid:
            row_errors.append(f"formula_identity_invalid:{index}")
        assignments = parsed.get("assignments")
        sat_extendible = bool(
            formula_identity_valid
            and isinstance(assignments, list)
            and formula_extendible(formula, assignments)
        )

        truncated = (
            row.get("finish_reason") in {"length", "max_tokens"} or row.get("truncated") is True
        )
        if truncated:
            row_errors.append(f"truncated_output:{index}")
        runtime_present = isinstance(row.get("runtime_identity_receipt"), Mapping)
        if not runtime_present:
            row_errors.append(f"runtime_identity_missing:{index}")
        runtime_valid = _runtime_identity_valid(row)
        if runtime_present and not runtime_valid:
            row_errors.append(f"runtime_identity_invalid:{index}")
        if runtime_valid:
            identities.append(_runtime_identity_key(row["runtime_identity_receipt"]))

        usable = bool(
            request_hash_valid
            and request_bytes_valid
            and seed_valid
            and response_hash_valid
            and parsed.get("parse_status") == "valid"
            and response_fidelity
            and formula_identity_valid
            and not truncated
            and row.get("terminal_state") == "response"
        )
        claimed_fields = {
            "schema_valid": parsed.get("schema_valid"),
            "variable_references_valid": parsed.get("variable_references_valid"),
            "response_fidelity_valid": response_fidelity,
            "usable_proposal": usable,
            "formula_valid": sat_extendible,
            "truncated": truncated,
        }
        for field, reduced_value in claimed_fields.items():
            if row.get(field) != reduced_value:
                row_errors.append(f"stored_{field}_mismatch:{index}")
        errors.extend(row_errors)
        rows.append(
            {
                "unit_id": row.get("request_id"),
                "call_index": index,
                "formula_id": row.get("formula_id"),
                "historical": True,
                "current_model_invocation": False,
                "request_sha256": row.get("raw_request_sha256"),
                "response_sha256": row.get("raw_reply_sha256"),
                "decoding_seed": raw_request.get("seed"),
                "request_hash_valid": request_hash_valid,
                "request_bytes_valid": request_bytes_valid,
                "response_hash_valid": response_hash_valid,
                "response_fidelity_valid": response_fidelity,
                "runtime_identity_valid": runtime_valid,
                "schema_valid": parsed.get("schema_valid") is True,
                "variable_references_valid": parsed.get("variable_references_valid") is True,
                "usable_proposal": usable,
                "sat_extendible": sat_extendible,
                "truncated": truncated,
                "metric": "historical_assignment_transport_usable",
                "metric_value": int(usable),
                "cost": deepcopy(row.get("cost") or {}),
                "failure": row_errors or None,
                "censored": row.get("terminal_state") != "response",
            }
        )

    identity_consistent = len(identities) == 4 and len(set(identities)) == 1
    if not identity_consistent:
        errors.append("runtime_identity_inconsistent")
    usable_count = sum(row["usable_proposal"] for row in rows)
    fidelity_count = sum(row["response_fidelity_valid"] for row in rows)
    sat_count = sum(row["sat_extendible"] for row in rows)
    complete_count = sum(not row["censored"] for row in rows)
    transport_ready = int(
        len(rows) == 4
        and usable_count >= 3
        and complete_count == 4
        and identity_consistent
        and not errors
    )
    return {
        "rows": rows,
        "errors": list(dict.fromkeys(errors)),
        "usable_proposal_count": usable_count,
        "response_fidelity_count": fidelity_count,
        "sat_extendible_count": sat_count,
        "complete_response_count": complete_count,
        "runtime_identity_consistent": identity_consistent,
        "historical_transport_ready_score": transport_ready,
        "assignment_reducer_ready_score": transport_ready,
    }


def replay_proof_boundary(repo_root: Path) -> JsonDict:
    """Replay shipped proof reducers and authenticate the sealed public bytes."""

    root = repo_root.resolve()
    memory_artifact = _load_object(root / PROOF_MEMORY_PATH)
    boundary_artifact = _load_object(root / PROOF_BOUNDARY_PATH)
    manifest = _load_object(root / SEALED_MANIFEST_PATH)
    errors: list[str] = []
    if not memory_artifact:
        errors.append("proof_memory_artifact_missing")
    if not boundary_artifact:
        errors.append("proof_boundary_artifact_missing")
    if not manifest:
        errors.append("sealed_manifest_missing")

    memory_validation = proof_memory.validate_artifact(memory_artifact) if memory_artifact else []
    boundary_validation = (
        proof_boundary.validate_artifact(boundary_artifact) if boundary_artifact else []
    )
    boundary_cold = (
        proof_boundary.cold_reload_errors(boundary_artifact, root) if boundary_artifact else []
    )
    protocol_errors = proof_boundary.validate_protocol(manifest) if manifest else []
    errors.extend(f"proof_memory:{error}" for error in memory_validation)
    errors.extend(f"proof_boundary:{error}" for error in boundary_validation)
    errors.extend(f"proof_boundary_cold:{error}" for error in boundary_cold)
    errors.extend(f"sealed_manifest:{error}" for error in protocol_errors)

    memory_reduction = proof_memory.independent_reduce(memory_artifact) if memory_artifact else {}
    boundary_reduction = (
        proof_boundary.independent_reduce(boundary_artifact) if boundary_artifact else {}
    )
    manifest_hash = sha256_file(root / SEALED_MANIFEST_PATH) if manifest else None
    if boundary_artifact.get("frozen_protocol_sha256") != manifest_hash:
        errors.append("frozen_protocol_hash_mismatch")
    for relative in PROOF_BOUNDARY_RAW_PATHS:
        expected = boundary_artifact.get("source_artifact_hashes", {}).get(relative.as_posix())
        observed = sha256_file(root / relative) if (root / relative).is_file() else None
        if expected != observed:
            errors.append(f"proof_sidecar_hash_mismatch:{relative.as_posix()}")
    memory_hash = sha256_file(root / PROOF_MEMORY_PATH) if memory_artifact else None
    recorded_memory_hash = boundary_artifact.get("source_artifact_hashes", {}).get(
        str((root / PROOF_MEMORY_PATH).resolve())
    )
    if recorded_memory_hash != memory_hash:
        errors.append("proof_memory_provenance_hash_mismatch")

    memory_ready = int(
        not memory_validation and memory_reduction.get("proof_fixture_ready_score") == 1
    )
    boundary_ready = int(
        not errors
        and memory_ready == 1
        and boundary_reduction.get("proof_boundary_ready_score") == 1
    )
    return {
        "errors": list(dict.fromkeys(errors)),
        "proof_memory_replay_ready_score": memory_ready,
        "proof_boundary_replay_ready_score": boundary_ready,
        "manifest_path": SEALED_MANIFEST_PATH.as_posix(),
        "manifest_sha256": manifest_hash,
        "proof_memory_artifact_sha256": memory_hash,
        "proof_boundary_artifact_sha256": sha256_file(root / PROOF_BOUNDARY_PATH)
        if boundary_artifact
        else None,
        "proof_memory_reduction": memory_reduction,
        "proof_boundary_reduction": boundary_reduction,
        "protocol_error_count": len(protocol_errors),
        "cold_reload_error_count": len(boundary_cold),
    }


def historical_model_inputs(repo_root: Path) -> JsonDict:
    """Preserve the original model evidence without counting it as current work."""

    root = repo_root.resolve()
    canary = _load_object(root / CANARY_PATH)
    candidate = _load_object(root / CANARY_CANDIDATE_PATH)
    runtime_receipts = [
        row.get("runtime_identity_receipt")
        for row in canary.get("rows", [])
        if isinstance(row, Mapping)
    ]
    runtime_hashes = [hash_json(row) for row in runtime_receipts if isinstance(row, Mapping)]
    return {
        "label": "historical_exp7372_disqualified_not_current_inference",
        "artifact_path": CANARY_PATH.as_posix(),
        "artifact_sha256": sha256_file(root / CANARY_PATH),
        "candidate_path": CANARY_CANDIDATE_PATH.as_posix(),
        "candidate_sha256": sha256_file(root / CANARY_CANDIDATE_PATH),
        "historical_experiment_id": canary.get("experiment_id"),
        "historical_verdict": canary.get("honest_verdict"),
        "historical_verdict_class": canary.get("verdict_class"),
        "historical_flagged_adversarial": canary.get("flagged_adversarial"),
        "historical_inference_substrate_class": canary.get("inference_substrate_class"),
        "historical_model_specs": deepcopy(canary.get("MODEL_SPECS") or []),
        "historical_invocation_counts": deepcopy(canary.get("invocation_counts") or {}),
        "historical_execution_venue": deepcopy(candidate.get("execution_venue")),
        "native_load_receipt": deepcopy(canary.get("load_receipt") or {}),
        "native_load_receipt_sha256": hash_json(canary.get("load_receipt") or {}),
        "native_runner_receipt": deepcopy(canary.get("runner_receipt") or {}),
        "native_runner_receipt_sha256": hash_json(canary.get("runner_receipt") or {}),
        "runtime_identity_receipt_sha256": runtime_hashes,
        "raw_call_hashes": {
            path.as_posix(): sha256_file(root / path) for path in CANARY_CALL_PATHS
        },
        "native_transport_hashes": {
            path.as_posix(): sha256_file(root / path)
            for path in (*NATIVE_CALL_PATHS, NATIVE_LOG_PATH)
        },
        "terminal_validator_log_hashes": {
            path.as_posix(): sha256_file(root / path) for path in HISTORICAL_TERMINAL_LOG_PATHS
        },
        "historical_only": True,
        "counted_as_current": False,
        "authorizes_current_readiness": False,
    }


def discrepancy_rows(repo_root: Path, reduced: Mapping[str, Any]) -> list[JsonDict]:
    """Reproduce each known Exp7372 defect with its corrected operator."""

    root = repo_root.resolve()
    terminal = _load_object(root / CANARY_PATH)
    candidate = _load_object(root / CANARY_CANDIDATE_PATH)
    old_usable = terminal.get("acceptance_gate_results", {}).get("usable_assignment_proposals", {})
    old_promotion = terminal.get("acceptance_gate_results", {}).get("promotion", {})
    observed_usable = reduced.get("usable_proposal_count")
    return [
        {
            "discrepancy_id": "usable_threshold_equality",
            "original_field": "acceptance_gate_results.usable_assignment_proposals",
            "original_expected": old_usable.get("expected"),
            "original_operator": "==",
            "exact_observed_value": old_usable.get("observed"),
            "original_passed": old_usable.get("passed"),
            "corrected_expected": 3,
            "corrected_operator": ">=",
            "independently_reduced_value": observed_usable,
            "corrected_passed": compare(">=", observed_usable, 3),
        },
        {
            "discrepancy_id": "forbidden_promotion_target",
            "original_field": "acceptance_gate_results.promotion",
            "original_expected": old_promotion.get("expected"),
            "original_operator": "==",
            "exact_observed_value": old_promotion.get("observed"),
            "original_passed": old_promotion.get("passed"),
            "corrected_expected": 0,
            "corrected_operator": "==",
            "independently_reduced_value": 0,
            "corrected_passed": True,
        },
        {
            "discrepancy_id": "execution_venue_not_closed_string",
            "original_field": "execution_venue",
            "original_expected": "host",
            "original_operator": "==",
            "exact_observed_value": deepcopy(candidate.get("execution_venue")),
            "corrected_expected": "host",
            "corrected_operator": "==",
            "independently_reduced_value": "host",
            "corrected_passed": True,
        },
        {
            "discrepancy_id": "oracle_positive_class",
            "original_field": "verdict_class",
            "original_expected": "circular_positive",
            "original_operator": "==",
            "exact_observed_value": candidate.get("verdict_class"),
            "corrected_expected": "circular_positive",
            "corrected_operator": "==",
            "independently_reduced_value": "circular_positive",
            "corrected_passed": True,
        },
        {
            "discrepancy_id": "internal_readiness_disagreement",
            "original_field": "qwen_assignment_transport_ready_score",
            "original_expected": 1,
            "original_operator": "==",
            "exact_observed_value": terminal.get("qwen_assignment_transport_ready_score"),
            "corrected_expected": 1,
            "corrected_operator": "==",
            "independently_reduced_value": reduced.get("historical_transport_ready_score"),
            "corrected_passed": reduced.get("historical_transport_ready_score") == 1,
        },
    ]


def build_validation_plan(repo_root: Path, private_root: Path) -> list[CommandSpec]:
    """Derive the exact fixed affected plan through the Exp7358 helper."""

    return build_command_plan(repo_root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(repo_root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad, duplicate, missing, or path-invalid validation commands."""

    errors = validate_command_plan(repo_root, VALIDATION_MANIFEST, commands)
    counts = Counter(command.name for command in commands)
    if counts != Counter(REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if "full_python_suite" in counts:
        errors.append("full_python_suite_forbidden")
    return list(dict.fromkeys(errors))


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful real receipt for each declared check."""

    counts = Counter(row.get("name") for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _acceptance_gates(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[JsonDict]:
    """Reduce current validation and historical diagnosis as separate gates."""

    reduction = artifact.get("assignment_reduction") or {}
    proof = artifact.get("proof_boundary_replay") or {}
    receipts = artifact.get("validation_receipts") or []
    preconditions = artifact.get("preconditions_checked") or []
    historical = artifact.get("historical_model_inputs") or {}
    gates = [
        _gate(
            "required_preconditions",
            True,
            "is",
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            category="safety",
            artifact_field="preconditions_checked",
            principle="Missing immutable evidence stops dependent reduction.",
        ),
        _gate(
            "historical_canary_stays_disqualified",
            {"verdict_class": "disqualified", "flagged_adversarial": True},
            "==",
            {
                "verdict_class": historical.get("historical_verdict_class"),
                "flagged_adversarial": historical.get("historical_flagged_adversarial"),
            },
            category="safety",
            artifact_field="historical_model_inputs",
            principle="Reducer repair cannot rehabilitate its quarantined input.",
            upstream=CANARY_PATH.as_posix(),
        ),
        _gate(
            "four_raw_calls",
            4,
            "==",
            len(artifact.get("rows") or []),
            category="completion",
            artifact_field="rows",
            principle="Every historical request and response must be replayed exactly once.",
        ),
        _gate(
            "usable_assignment_proposals",
            3,
            ">=",
            reduction.get("usable_proposal_count"),
            category="completion",
            artifact_field="assignment_reduction.usable_proposal_count",
            principle="At least three usable responses pass; four cannot fail an equality accident.",
        ),
        _gate(
            "response_fidelity",
            4,
            "==",
            reduction.get("response_fidelity_count"),
            category="safety",
            artifact_field="assignment_reduction.response_fidelity_count",
            principle="The decoded literals must match the exact raw response bytes.",
        ),
        _gate(
            "sat_extendibility_reported_separately",
            0,
            "==",
            reduction.get("sat_extendible_count"),
            category="scientific_efficacy",
            artifact_field="assignment_reduction.sat_extendible_count",
            principle="Transport success does not hide that no historical assignment extends to SAT.",
        ),
        _gate(
            "proof_boundary_replay",
            1,
            "==",
            proof.get("proof_boundary_replay_ready_score"),
            category="completion",
            artifact_field="proof_boundary_replay_ready_score",
            principle="All original proof checks and sealed bytes must remain valid.",
            upstream=PROOF_BOUNDARY_PATH.as_posix(),
        ),
        _gate(
            "current_model_invocations",
            {"MODEL_SPECS": [], "model_invoked": False, "counts": ZERO_INVOCATION_COUNTS},
            "==",
            {
                "MODEL_SPECS": artifact.get("MODEL_SPECS"),
                "model_invoked": artifact.get("model_invoked"),
                "counts": artifact.get("invocation_counts"),
            },
            category="safety",
            artifact_field="MODEL_SPECS/model_invoked/invocation_counts",
            principle="Historical calls never become current model work.",
        ),
        _gate(
            "execution_venue_closed",
            "host",
            "==",
            artifact.get("execution_venue"),
            category="safety",
            artifact_field="execution_venue",
            principle="The current aggregation venue uses the closed host value.",
        ),
        _gate(
            "promotion_forbidden",
            0,
            "==",
            artifact.get("promotion_score"),
            category="promotion",
            artifact_field="promotion_score",
            principle="This diagnosis never authorizes an external action.",
        ),
        _gate(
            "required_affected_validation",
            True,
            "is",
            _receipts_pass(receipts, REQUIRED_CHECK_NAMES),
            category="required_validation",
            artifact_field="validation_receipts",
            principle="Every scoped affected check must execute once and pass.",
        ),
        _gate(
            "current_scientific_benefit",
            None,
            "is",
            None,
            category="scientific_efficacy",
            artifact_field="scientific_value_score",
            principle="This reducer diagnosis measures no current model benefit.",
        ),
    ]
    if require_terminal:
        gates.append(
            _gate(
                "terminal_validation",
                True,
                "is",
                _receipts_pass(receipts, TERMINAL_CHECK_NAMES),
                category="required_validation",
                artifact_field="validation_receipts",
                principle="Declared replay, cold reduction, and both strict readers must pass.",
            )
        )
    return gates


def _gate_summary(gates: Sequence[Mapping[str, Any]], ready_score: int) -> JsonDict:
    """Name the first failed gate without hiding later failures."""

    failures = [dict(row) for row in gates if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "failed_check": first.get("check") if first else None,
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "artifact_field": first.get("artifact_field")
        if first
        else "assignment_reducer_ready_score",
        "expected_value": deepcopy(first.get("expected")) if first else 1,
        "observed_value": deepcopy(first.get("observed")) if first else ready_score,
        "failed_check_count": len(failures),
        "passed": not failures,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every top-level field and retain prompt-supplied principles."""

    principles = {
        field: "This field keeps one part of the reducer receipt auditable." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _artifact_state(
    artifact: Mapping[str, Any], *, require_terminal: bool
) -> tuple[str, str, str, int, int]:
    """Classify completion without turning reducer readiness into science value."""

    preconditions = artifact.get("preconditions_checked") or []
    if not preconditions or any(row.get("passed") is not True for row in preconditions):
        return "blocked", "blocked_required_immutable_input", "blocked", 0, 0
    if not require_terminal:
        return (
            "candidate_pending_terminal_validation",
            "partial_candidate_pending_terminal_validation",
            "partial",
            0,
            0,
        )
    assignment = artifact.get("assignment_reduction") or {}
    proof = artifact.get("proof_boundary_replay") or {}
    receipts = artifact.get("validation_receipts") or []
    current_declarations_valid = bool(
        artifact.get("MODEL_SPECS") == []
        and artifact.get("model_invoked") is False
        and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and artifact.get("inference_substrate_class") == "aggregation"
        and artifact.get("execution_venue") == "host"
        and artifact.get("promotion_score") == 0
    )
    ready = bool(
        assignment.get("assignment_reducer_ready_score") == 1
        and not assignment.get("errors")
        and proof.get("proof_boundary_replay_ready_score") == 1
        and not proof.get("errors")
        and current_declarations_valid
        and _receipts_pass(receipts, REQUIRED_CHECK_NAMES)
        and _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
        and artifact.get("flagged_adversarial") is False
    )
    if not ready:
        return (
            "complete_required_validation_failed",
            "complete_disqualified_reducer_or_validation_defect",
            "disqualified",
            0,
            0,
        )
    return (
        "complete_assignment_reducer_ready_historical_canary_disqualified",
        "complete_null_assignment_reducer_ready_no_current_scientific_benefit",
        "null",
        1,
        1,
    )


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    assignment: Mapping[str, Any],
    proof: Mapping[str, Any],
    historical: Mapping[str, Any],
    discrepancies: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    run_date: str,
    started_at: str,
    completed_at: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    require_terminal: bool,
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one complete candidate or terminal record from reduced evidence."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_llm_invocation_receipts": [],
        "small_ebm_training": {
            "performed": False,
            "training_steps": 0,
            "gibbs_updates": 0,
            "note": "No Gibbs fitting is needed for immutable receipt aggregation.",
        },
        "inference_substrate": "host_cpu_aggregation_from_immutable_upstream_artifacts",
        "inference_substrate_details": {
            "device": platform.processor() or platform.machine() or "unknown_cpu",
            "python": platform.python_version(),
            "jax_platform": "cpu",
            "jax_work_performed": False,
            "native_cuda_work_performed": False,
            "resource_lease": "host_cpu_no_exclusive_accelerator",
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes),
        "rows": deepcopy(assignment.get("rows") or []),
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": len(assignment.get("rows") or []),
            "completed_units": assignment.get("complete_response_count", 0),
            "censored_units": sum(
                bool(row.get("censored")) for row in assignment.get("rows") or []
            ),
            "unstarted_units": max(0, 4 - len(assignment.get("rows") or [])),
            "remaining_work": 0 if len(assignment.get("rows") or []) == 4 else 4,
            "stopping_rule": "Replay each of four immutable calls once; do not retry, replace, or generate.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_reducer",
        "verdict_class": "partial",
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "required_check_names": list(REQUIRED_CHECK_NAMES),
        "terminal_check_names": list(TERMINAL_CHECK_NAMES),
        "repository_health": {
            "status": "affected_scope_only",
            "as_of": "2026-09-18",
            "affects_required_checks": False,
            "unrelated_findings": [],
        },
        "field_principles": {},
        "promotion_score": 0,
        "scientific_value_score": 0,
        "assignment_reducer_ready_score": 0,
        "proof_boundary_replay_ready_score": 0,
        "historical_model_inputs": deepcopy(dict(historical)),
        "discrepancy_rows": [deepcopy(dict(row)) for row in discrepancies],
        "assignment_reduction": deepcopy(dict(assignment)),
        "proof_boundary_replay": deepcopy(dict(proof)),
        "fresh_canary_required": {
            "experiment_id": "exp7388-proposal-capture",
            "required_calls": 4,
            "satisfied_by_this_receipt": False,
        },
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": list(TERMINAL_CHECK_NAMES[:2]),
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "external_publication_authorized": False,
    }
    status, honest, verdict, assignment_ready, proof_ready = _artifact_state(
        artifact, require_terminal=require_terminal
    )
    artifact.update(
        status=status,
        honest_verdict=honest,
        verdict_class=verdict,
        assignment_reducer_ready_score=assignment_ready,
        proof_boundary_replay_ready_score=proof_ready,
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(
        artifact, require_terminal=require_terminal
    )
    artifact["gate_check_summary"] = _gate_summary(
        artifact["acceptance_gate_results"], assignment_ready
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(repo_root: Path, receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a deterministic terminal fixture from the real immutable inputs."""

    checks, hashes = collect_preconditions(repo_root)
    calls, formulas = load_assignment_evidence(repo_root)
    assignment = reduce_assignment_receipts(calls, formulas)
    proof = replay_proof_boundary(repo_root)
    historical = historical_model_inputs(repo_root)
    return build_artifact(
        preconditions=checks,
        source_hashes=hashes,
        assignment=assignment,
        proof=proof,
        historical=historical,
        discrepancies=discrepancy_rows(repo_root, assignment),
        receipts=receipts,
        run_date=RUN_DATE,
        started_at="2026-09-18T00:00:00Z",
        completed_at="2026-09-18T00:00:01Z",
        duration_s=1.0,
        phase_spans=[],
        require_terminal=True,
    )


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    *,
    run_date: str,
    started_at: str,
    duration_s: float,
    source_hashes: Mapping[str, str] | None = None,
) -> JsonDict:
    """Publish external absence as blocked without success-shaped evidence."""

    failed = next(
        (dict(row) for row in preconditions if row.get("passed") is not True),
        {
            "check": "unknown_precondition",
            "upstream": EXPERIMENT_ID,
            "artifact_field": "unknown",
            "expected_value": True,
            "observed_value": "missing",
        },
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "host_cpu_aggregation_precondition_checks_only",
        "inference_substrate_details": {
            "device": platform.machine(),
            "resource_lease": "none",
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": float(duration_s),
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_hashes or {}),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 4,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 4,
            "unstarted_units": 4,
            "remaining_work": 4,
            "stopping_rule": "Stop before dependent work when immutable input is unavailable.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {
            "failed_check": failed.get("check"),
            "upstream": failed.get("upstream"),
            "artifact_field": failed.get("artifact_field"),
            "expected_value": failed.get("expected_value"),
            "observed_value": failed.get("observed_value", "missing"),
            "failed_check_count": sum(row.get("passed") is not True for row in preconditions),
            "passed": False,
        },
        "verifier_is_oracle": True,
        "honest_verdict": f"blocked_{failed.get('check', 'required_input')}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_evaluated",
            "as_of": "2026-09-18",
            "affects_required_checks": False,
        },
        "field_principles": {},
        "promotion_score": 0,
        "assignment_reducer_ready_score": 0,
        "proof_boundary_replay_ready_score": 0,
        "historical_model_inputs": {},
        "discrepancy_rows": [],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check exact inputs, reductions, gates, state, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    errors.extend(
        f"missing_required_field:{field}"
        for field in sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    )
    if errors:
        return errors
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if not (
        artifact.get("MODEL_SPECS") == []
        and artifact.get("model_invoked") is False
        and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS
        and artifact.get("inference_substrate_class") == "aggregation"
        and artifact.get("execution_venue") == "host"
        and artifact.get("promotion_score") == 0
    ):
        errors.append("current_execution_declaration_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")

    for relative, expected in (artifact.get("source_artifact_hashes") or {}).items():
        path = Path(str(relative))
        if not path.is_absolute():
            path = root / path
        observed = sha256_file(path) if path.is_file() else None
        if observed != expected:
            errors.append(f"source_hash_mismatch:{relative}")

    if artifact.get("verdict_class") != "blocked":
        calls, formulas = load_assignment_evidence(root)
        assignment = reduce_assignment_receipts(calls, formulas)
        if artifact.get("assignment_reduction") != assignment:
            errors.append("assignment_reduction_mismatch")
        if artifact.get("rows") != assignment["rows"]:
            errors.append("assignment_rows_mismatch")
        proof = replay_proof_boundary(root)
        if artifact.get("proof_boundary_replay") != proof:
            errors.append("proof_boundary_replay_mismatch")
        historical = historical_model_inputs(root)
        if artifact.get("historical_model_inputs") != historical:
            errors.append("historical_model_inputs_mismatch")
        if artifact.get("discrepancy_rows") != discrepancy_rows(root, assignment):
            errors.append("discrepancy_rows_mismatch")

        expected_gates = _acceptance_gates(artifact, require_terminal=require_terminal)
        if artifact.get("acceptance_gate_results") != expected_gates:
            errors.append("acceptance_gate_results_mismatch")
        state = _artifact_state(artifact, require_terminal=require_terminal)
        expected_state = {
            "status": state[0],
            "honest_verdict": state[1],
            "verdict_class": state[2],
            "assignment_reducer_ready_score": state[3],
            "proof_boundary_replay_ready_score": state[4],
        }
        if any(artifact.get(field) != expected for field, expected in expected_state.items()):
            errors.append("terminal_state_mismatch")
        if artifact.get("assignment_reducer_ready_score") != state[3]:
            errors.append("assignment_reducer_ready_score_mismatch")
        if artifact.get("proof_boundary_replay_ready_score") != state[4]:
            errors.append("proof_boundary_replay_ready_score_mismatch")
        expected_summary = _gate_summary(expected_gates, state[3])
        if artifact.get("gate_check_summary") != expected_summary:
            errors.append("gate_check_summary_mismatch")
    elif any(
        artifact.get(field) != 0
        for field in (
            "assignment_reducer_ready_score",
            "proof_boundary_replay_ready_score",
            "promotion_score",
        )
    ):
        errors.append("blocked_scores_nonzero")

    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _terminal_commands(repo_root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build cold replay and strict-reader commands for the measured candidate."""

    python = str(repo_root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7383_v648_canary_reducer import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v,require_terminal=False);"
        "print(e,flush=True);raise SystemExit(bool(e))"
    )
    commands = (
        (
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
            "completion",
        ),
        (
            "independent_reducer",
            (python, "-u", "-c", reducer_code, str(candidate)),
            "completion",
        ),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "safety",
        ),
    )
    return [
        PlannedCommand(CommandSpec(name, argv, "measured_candidate"), category, True)
        for name, argv, category in commands
    ]


def _span(
    phase: str, phase_started: float, run_started: float, completed_units: int
) -> JsonDict:  # pragma: no cover - measured orchestration.
    """Close one measured phase and retain its real checkpoint boundary."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_elapsed_s": phase_started - run_started,
        "end_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": completed_units,
        "checkpoint_at_utc": utc_now(),
    }


def run_experiment(
    repo_root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - exercised through the declared entrypoint.
    """Authenticate, replay, validate, and atomically publish one receipt."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    raw_dir = root / RAW_DIR
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "read", "start")
    preconditions, source_hashes = collect_preconditions(root)
    spans.append(_span("read", phase_started, started, len(preconditions)))
    progress(started, "read", "end", passed=all(row["passed"] for row in preconditions))
    if not all(row["passed"] for row in preconditions):
        blocked = build_blocked_artifact(
            preconditions,
            run_date=run_date,
            started_at=started_at,
            duration_s=time.monotonic() - started,
            source_hashes=source_hashes,
        )
        progress(started, "write", "before_atomic_blocked", path=output)
        atomic_json(output, blocked)
        progress(started, "write", "after_atomic_blocked", path=output)
        return blocked

    phase_started = time.monotonic()
    progress(started, "build", "start")
    calls, formulas = load_assignment_evidence(root)
    historical = historical_model_inputs(root)
    spans.append(_span("build", phase_started, started, len(calls)))
    progress(started, "build", "end", calls=len(calls))

    for no_work_phase in ("load", "generate"):
        phase_started = time.monotonic()
        progress(started, no_work_phase, "before_no_current_model_work")
        spans.append(_span(no_work_phase, phase_started, started, 0))
        progress(started, no_work_phase, "after_no_current_model_work", model_invoked=False)

    phase_started = time.monotonic()
    progress(started, "evaluate", "start")
    assignment = reduce_assignment_receipts(calls, formulas)
    proof = replay_proof_boundary(root)
    discrepancies = discrepancy_rows(root, assignment)
    spans.append(_span("evaluate", phase_started, started, len(calls)))
    progress(
        started,
        "evaluate",
        "end",
        usable=assignment["usable_proposal_count"],
        sat=assignment["sat_extendible_count"],
        proof_ready=proof["proof_boundary_replay_ready_score"],
    )

    private_root = Path(tempfile.mkdtemp(prefix="exp7383-validation-", dir="/tmp"))
    phase_started = time.monotonic()
    progress(started, "validate", "before_affected_subprocesses")
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    affected_receipts: list[JsonDict] = []
    if not plan_errors:
        affected_receipts = run_categorized_commands(
            root,
            [PlannedCommand(command, "required_validation", True) for command in commands],
            log_dir=raw_dir / "validation/affected",
        )
    affected = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected_receipts)
    spans.append(_span("validate_affected", phase_started, started, len(affected_receipts)))
    progress(
        started,
        "validate",
        "after_affected_subprocesses",
        passed=affected["passed"] and not plan_errors,
    )

    phase_started = time.monotonic()
    progress(started, "write", "before_candidate")
    candidate = build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        assignment=assignment,
        proof=proof,
        historical=historical,
        discrepancies=discrepancies,
        receipts=affected_receipts,
        run_date=run_date,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        require_terminal=False,
    )
    candidate_path = raw_dir / "measured-terminal-candidate.json"
    atomic_json(candidate_path, candidate)
    spans.append(_span("write_candidate", phase_started, started, 1))
    progress(started, "write", "after_candidate", path=candidate_path)

    phase_started = time.monotonic()
    progress(started, "validate", "before_terminal_subprocesses")
    terminal_receipts = run_categorized_commands(
        root,
        _terminal_commands(root, candidate_path),
        log_dir=raw_dir / "validation/terminal",
    )
    spans.append(_span("validate_terminal", phase_started, started, len(terminal_receipts)))
    critical = any("[CRITICAL]" in str(row.get("output_tail") or "") for row in terminal_receipts)
    progress(
        started,
        "validate",
        "after_terminal_subprocesses",
        passed=_receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES),
        critical=critical,
    )

    phase_started = time.monotonic()
    progress(started, "write", "before_atomic_terminal", path=output)
    final = build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        assignment=assignment,
        proof=proof,
        historical=historical,
        discrepancies=discrepancies,
        receipts=[*affected_receipts, *terminal_receipts],
        run_date=run_date,
        started_at=started_at,
        completed_at=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=[*spans, _span("write", phase_started, started, 1)],
        require_terminal=True,
        flagged_adversarial=critical,
    )
    validation_errors = validate_artifact(final, root=root, require_terminal=True)
    if validation_errors:
        raise RuntimeError(f"terminal_artifact_invalid:{validation_errors}")
    atomic_json(candidate_path, final)
    atomic_json(output, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse the thin entrypoint and its cold-validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entrypoint E2E.
    """Run aggregation or cold-validate one measured candidate."""

    args = parse_args(argv)
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.validate is not None:
        value = _load_object(args.validate)
        errors = validate_artifact(value, root=REPO_ROOT, require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
