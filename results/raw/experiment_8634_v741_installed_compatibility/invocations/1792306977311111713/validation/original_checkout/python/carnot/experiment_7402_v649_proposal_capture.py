"""Capture the unchanged 64-call prospective Boolean proposal panel.

The model supplies literal proposals. Exact 2-SAT checks describe those
proposals but do not choose a favorable candidate or redefine model hardness.

Spec refs: REQ-REPORT-7402 and SCENARIO-REPORT-7402-*.
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
import tempfile
import time
from typing import Any, Iterator

from carnot import experiment_7347_v645_plan_canary as runtime
from carnot import experiment_7371_v647_proof_boundary as proof_boundary
from carnot import experiment_7372_v647_qwen_canary as assignment_parser
from carnot import experiment_7400_v649_assignment_canary as canary
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
from carnot.reporting import current_work_receipt
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260918"
MILESTONE = "2026.09.649"
PHASE = 3
EXPERIMENT_ID = "exp7402-v649-proposal-capture"
TASK_ID = "experiment_7402_v649_proposal_capture"
SCHEMA = "carnot.exp7402.v649.proposal_capture.v1"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
MAX_GENERATED_TOKENS = 256
MODEL_LOAD_TIMEOUT_S = 600.0
MODEL_WORK_TIMEOUT_S = 2400.0
REQUEST_TIMEOUT_S = 180.0
VALIDATION_RESERVE_S = 900.0
RANDOM_SEED = {
    "experiment": 7_371_401,
    "resampling": 7_371_307,
}

MODULE_PATH = Path("python/carnot/experiment_7402_v649_proposal_capture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7402_v649_proposal_capture.py")
TEST_PATH = Path("tests/python/test_experiment_7402_v649_proposal_capture.py")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7402_v649_proposal_capture.json")
RAW_DIR = Path("results/raw/experiment_7402_v649_proposal_capture")
PRODUCER_PATH = Path("results/experiment_7400_v649_assignment_canary.json")
PROTOCOL_PATH = Path("data/v647_implication_stream_manifest.json")
EXPECTED_PRODUCER_SHA256 = "sha256:ab4604e692aaedfe1afed4764b61ad324fc81901bc5aafd6378b76b918460c62"
EXPECTED_PROTOCOL_SHA256 = "sha256:6c0a312a8e34da96c2f2308b84c503bd8d064f59a8bc78caa41cf2b187d95fab"
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
    "schema": "Use a versioned schema with ordinary identity fields; publish terminal state only after measured work and checks.",
    "run_date": "Use 20260918 and retain actual UTC start and completion timestamps.",
    "preconditions_checked": "Authenticate exact inputs, eligibility, cache, runtime, device, and entrypoint before dependent work.",
    "MODEL_SPECS": "Name the current Qwen GGUF used for fresh LLM work; keep small training separate.",
    "model_invoked": "Set true after any attempted current real model load or generation, including failure.",
    "invocation_counts": "Derive owned current load and generation dispositions from events, never historical evidence.",
    "inference_substrate": "Describe current work with a string and keep device and software facts in details.",
    "inference_substrate_class": "Use model_bounded_generation for the fixed short-output panel without runtime padding.",
    "execution_venue": "Use the closed host string instead of a hostname or device mapping.",
    "duration_s": "Measure current task time monotonically and retain scientific and validation spans separately.",
    "phase_spans": "Retain actual phase boundaries, heartbeat checkpoints, and completed units.",
    "random_seed": "Freeze experiment and resampling seeds before observing outputs.",
    "reproducibility_checksum": "Bind current code, settings, input hashes, event ledger, selections, and raw rows.",
    "source_artifact_hashes": "Hash exact canary, protocol, code, model, runner, sidecar, and raw evidence bytes.",
    "rows": "Keep every planned candidate, metric contribution, cost, failure, and censoring disposition.",
    "sample_size_budget": "Predeclare planned, attempted, completed, censored, unstarted, limits, stop rule, and independent groups.",
    "acceptance_gate_results": "Keep prerequisite, completion, evidence, validation, safety, and efficacy checks separate.",
    "gate_check_summary": "Every blocked result names upstream path, check, field, operator, expected value, and observed value.",
    "verifier_is_oracle": "Mark true because exact SAT defines semantic correctness while selection remains syntax-first.",
    "honest_verdict": "Use complete_ for finished work and blocked_ for an unchanged missing prerequisite.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Preserve critical verifier findings and never use invalid science for readiness.",
    "validation_receipts": "Retain exact argv, scoped environment, check name, exit, duration, and hashed log, including failures.",
    "repository_health": "Keep unrelated broad-suite observations separate while affected failures remain disqualifying.",
    "field_principles": "Explain fields without wrapping numeric gates or ordinary dictionaries.",
    "promotion_score": "Keep zero; capture cannot roll out, update weights, publish, or submit externally.",
    "candidate_capture_complete_score": "Require complete authentic dispositions, usable syntax-first selections, sound runtime evidence, and validation.",
    "candidate_rows": "Retain all 64 calls and link them to all 32 exact selected-request dispositions.",
    "complexity_rows": "Describe source size, solver work, tokens, parse, and semantics without treating solver time as model hardness.",
}
REQUIRED_FIELDS = frozenset(FIELD_PRINCIPLES)


def utc_now() -> str:  # pragma: no cover - real wall-clock boundary.
    """Return one actual UTC boundary while elapsed time stays monotonic."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase, long-call, heartbeat, and checkpoint boundaries."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7402] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so source and raw evidence cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for prompts, events, selections, and artifacts."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON value with fsync and a local atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Return one JSON object, or an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact without recursively hashing its checksum."""

    value = deepcopy(dict(artifact))
    value.pop("reproducibility_checksum", None)
    return canonical_hash(value)


def compare(operator: str, observed: Any, expected: Any) -> bool:
    """Apply one declared gate operator without interpreting prose."""

    if operator == "==":
        return observed == expected
    if operator == "in":
        return observed in expected
    if operator == ">=":
        return observed >= expected
    raise ValueError(f"unsupported_operator:{operator}")


def gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    operator: str,
    expected: Any,
    observed: Any,
    *,
    category: str = "precondition",
) -> JsonDict:
    """Retain both operands and an independently computed gate result."""

    try:
        passed = compare(operator, observed, expected)
    except (TypeError, ValueError):
        passed = False
    return {
        "category": category,
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "operator": operator,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": passed,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first exact failure while retaining every failed check."""

    failures = [dict(row) for row in checks if row.get("passed") is not True]
    first = failures[0] if failures else None
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": [row.get("check") for row in failures],
        "upstream": first.get("upstream") if first else EXPERIMENT_ID,
        "path": first.get("upstream") if first else RESULT_PATH.as_posix(),
        "check": first.get("check") if first else "all_required_checks",
        "artifact_field": first.get("artifact_field") if first else "gate_check_summary",
        "operator": first.get("operator") if first else "==",
        "expected_value": deepcopy(first.get("expected_value")) if first else True,
        "observed_value": deepcopy(first.get("observed_value")) if first else True,
        "passed": not failures,
    }


def producer_gate_rows(producer: Mapping[str, Any], *, root: Path) -> list[JsonDict]:
    """Authenticate the exact eligible Exp7400 canary before model work."""

    fields: tuple[tuple[str, str, str, Any], ...] = (
        ("producer_experiment_id", "experiment_id", "==", "exp7400-v649-assignment-canary"),
        ("producer_milestone", "milestone", "==", MILESTONE),
        ("assignment_transport_ready", "qwen_assignment_transport_ready_score", "==", 1),
        (
            "producer_verdict_class",
            "verdict_class",
            "in",
            ["positive", "circular_positive", "null"],
        ),
        ("producer_adversarial_flag", "flagged_adversarial", "==", False),
    )
    rows = [
        gate_row(check, PRODUCER_PATH.as_posix(), field, operator, expected, producer.get(field))
        for check, field, operator, expected in fields
    ]
    producer_path = root / PRODUCER_PATH
    rows.append(
        gate_row(
            "exact_canary_artifact_hash",
            PRODUCER_PATH.as_posix(),
            "sha256",
            "==",
            EXPECTED_PRODUCER_SHA256,
            sha256_file(producer_path) if producer_path.is_file() else None,
        )
    )
    replay_errors = canary.validate_artifact(producer, root=root, require_terminal=True)
    rows.append(
        gate_row(
            "canary_cold_validation",
            PRODUCER_PATH.as_posix(),
            "validate_artifact.errors",
            "==",
            [],
            replay_errors,
        )
    )
    return rows


def build_capture_schedule(manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Expand the frozen eight-by-four-by-two panel without reading outcomes."""

    streams = manifest.get("live_proposal_streams")
    if not isinstance(streams, list) or len(streams) != 8:
        raise ValueError("stream_count")
    live_seed = dict(manifest.get("random_seed") or {}).get("live_protocol")
    if not isinstance(live_seed, int):
        raise ValueError("live_protocol_seed")
    schedule: list[JsonDict] = []
    request_group_index = 0
    for stream_index, stream_value in enumerate(streams):
        if not isinstance(stream_value, Mapping):
            raise ValueError("stream_shape")
        stream = dict(stream_value)
        requests = stream.get("requests")
        versions_value = stream.get("versions")
        if not isinstance(requests, list) or len(requests) != 4:
            raise ValueError("request_count")
        if not isinstance(versions_value, list):
            raise ValueError("version_shape")
        versions = {
            str(row.get("version")): deepcopy(dict(row))
            for row in versions_value
            if isinstance(row, Mapping)
        }
        for request_index, request_value in enumerate(requests):
            if not isinstance(request_value, Mapping):
                raise ValueError("request_shape")
            request = dict(request_value)
            if request.get("proposal_count") != 2:
                raise ValueError("proposal_count")
            version = versions.get(str(request.get("formula_version")))
            if version is None or version.get("source_hash") != request.get("formula_source_hash"):
                raise ValueError("formula_identity")
            for candidate_index in range(2):
                call_index = len(schedule)
                request_id = str(request.get("request_id") or "")
                schedule.append(
                    {
                        "call_index": call_index,
                        "call_id": f"{request_id}:candidate-{candidate_index}",
                        "stream_index": stream_index,
                        "stream_id": stream.get("stream_id"),
                        "family": stream.get("family"),
                        "formula_id": stream.get("formula_id"),
                        "n_vars": int(stream.get("n_vars", 0)),
                        "stream_seed": stream.get("seed"),
                        "request_group_index": request_group_index,
                        "request_index": request_index,
                        "request_id": request_id,
                        "query_type": request.get("split"),
                        "split": request.get("split"),
                        "formula_version": request.get("formula_version"),
                        "formula_source_hash": request.get("formula_source_hash"),
                        "formula": version,
                        "prompt": request.get("prompt"),
                        "candidate_index": candidate_index,
                        "inference_seed": live_seed,
                        "max_new_tokens": MAX_GENERATED_TOKENS,
                    }
                )
            request_group_index += 1
    if len(schedule) != 64 or len({row["call_id"] for row in schedule}) != 64:
        raise ValueError("call_identity")
    expected_splits = ["warm_up", "future", "future", "version_change"]
    if any(
        [row.get("split") for row in stream["requests"]] != expected_splits for stream in streams
    ):
        raise ValueError("request_schedule")
    return schedule


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict]:
    """Check exact sources, canary eligibility, and protocol identity first."""

    required = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/experiment_7372_v647_qwen_canary.py"),
        Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("openspec/capabilities/constraint-verification/spec.md"),
        SPEC_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        PRODUCER_PATH,
        PROTOCOL_PATH,
    )
    checks: list[JsonDict] = []
    for relative in required:
        path = root / relative
        observed = "readable_nonempty_bytes" if path.is_file() and path.stat().st_size else None
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "==",
                "readable_nonempty_bytes",
                observed,
            )
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "==",
            "REQ-REPORT-7402",
            "REQ-REPORT-7402" if "REQ-REPORT-7402" in spec else None,
        )
    )
    producer = load_object(root / PRODUCER_PATH)
    checks.extend(producer_gate_rows(producer, root=root))
    protocol = load_object(root / PROTOCOL_PATH)
    protocol_path = root / PROTOCOL_PATH
    protocol_hash = sha256_file(protocol_path) if protocol_path.is_file() else None
    checks.append(
        gate_row(
            "exact_protocol_hash",
            PROTOCOL_PATH.as_posix(),
            "sha256",
            "==",
            EXPECTED_PROTOCOL_SHA256,
            protocol_hash,
        )
    )
    checks.append(
        gate_row(
            "independent_protocol_validation",
            PROTOCOL_PATH.as_posix(),
            "validate_protocol.errors",
            "==",
            [],
            proof_boundary.validate_protocol(protocol),
        )
    )
    exclusion = (
        (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
        if (root / "ops/exclusion_manifest.yaml").is_file()
        else ""
    )
    excluded = "experiment_id: 7402" in exclusion or EXPERIMENT_ID in exclusion
    checks.append(
        gate_row(
            "not_excluded",
            "ops/exclusion_manifest.yaml",
            "experiment_id",
            "==",
            False,
            excluded,
        )
    )
    schedule: list[JsonDict] = []
    selection_error: str | None = None
    try:
        schedule = build_capture_schedule(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        selection_error = f"{type(exc).__name__}:{exc}"
    checks.append(
        gate_row(
            "frozen_proposal_panel",
            PROTOCOL_PATH.as_posix(),
            "live_proposal_streams",
            "==",
            {"calls": 64, "requests": 32, "selection_error": None},
            {
                "calls": len(schedule),
                "requests": len({row["request_id"] for row in schedule}),
                "selection_error": selection_error,
            },
        )
    )
    return checks, {
        "producer": producer,
        "producer_sha256": sha256_file(root / PRODUCER_PATH)
        if (root / PRODUCER_PATH).is_file()
        else None,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "schedule": schedule,
    }


def _native_request_payload(prompt: str, seed: int) -> JsonDict:
    """Rebuild the exact fixed native request for byte-level evidence."""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": seed,
        "cache_prompt": False,
        "max_tokens": MAX_GENERATED_TOKENS,
    }


def build_candidate_row(
    schedule: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> JsonDict:
    """Retain exact candidate bytes and measure syntax and SAT separately."""

    raw_reply = str(response.get("raw_reply") or "")
    parsed = assignment_parser.decode_assignment(raw_reply, int(schedule["n_vars"]))
    assignments = parsed.get("assignments")
    terminal_state = str(
        response.get("terminal_state") or ("request_error" if response.get("error") else "response")
    )
    attempted = response.get("attempted") is not False
    solver_duration_ns: int | None = None
    semantic_extendible: bool | None = None
    if parsed["parse_status"] == "valid" and isinstance(assignments, list):
        formula = dict(schedule["formula"])
        exact = FormulaVersion.from_clauses(
            str(formula["version"]),
            int(formula["n_vars"]),
            formula["raw_clause_order"],
        )
        solver_started = time.perf_counter_ns()
        semantic_extendible, _witness = exact.solve(assignments)
        solver_duration_ns = time.perf_counter_ns() - solver_started
    raw_request = deepcopy(
        dict(
            response.get("raw_request")
            or _native_request_payload(str(schedule["prompt"]), int(schedule["inference_seed"]))
        )
    )
    raw_response = deepcopy(dict(response.get("raw_response") or {}))
    request_bytes = canonical_json(raw_request).encode("utf-8")
    response_bytes = canonical_json(raw_response).encode("utf-8")
    formula = dict(schedule["formula"])
    return {
        **deepcopy(dict(schedule)),
        "prompt_sha256": canonical_hash(str(schedule["prompt"])),
        "raw_request": raw_request,
        "raw_request_sha256": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
        "raw_request_bytes_b64": base64.b64encode(request_bytes).decode("ascii"),
        "raw_reply": raw_reply,
        "raw_reply_sha256": canonical_hash(raw_reply),
        "raw_reply_bytes_b64": base64.b64encode(raw_reply.encode("utf-8")).decode("ascii"),
        "raw_response": raw_response,
        "raw_response_sha256": "sha256:" + hashlib.sha256(response_bytes).hexdigest(),
        "raw_response_bytes_b64": base64.b64encode(response_bytes).decode("ascii"),
        "parse_status": parsed["parse_status"],
        "parse_errors": deepcopy(parsed["parse_errors"]),
        "schema_valid": parsed["schema_valid"],
        "variable_references_valid": parsed["variable_references_valid"],
        "response_fidelity_valid": parsed["response_fidelity_valid"],
        "source_literal_fidelity": parsed["parse_status"] == "valid",
        "decoded_assignments": deepcopy(assignments),
        "semantic_extendible": semantic_extendible,
        "exact_solver_duration_ns": solver_duration_ns,
        "exact_solver_clause_count": len(formula["raw_clause_order"]),
        "attempted": attempted,
        "terminal_state": terminal_state,
        "error": response.get("error"),
        "finish_reason": response.get("finish_reason"),
        "truncated": response.get("finish_reason") in {"length", "max_tokens"},
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "runtime_identity_receipt": deepcopy(dict(runtime_identity)),
        "censored": attempted and terminal_state != "response",
        "selected_for_request": False,
        "selection_rank": None,
        "equal_later_comparator_cost_share": 0.5,
        "metric": "literal_proposal_capture",
        "metric_value": int(parsed["parse_status"] == "valid"),
        "cost": {
            "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(response.get("completion_tokens", 0) or 0),
            "latency_s": float(response.get("latency_s", 0.0) or 0.0),
            "later_comparator_share": 0.5,
        },
    }


def reduce_panel(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Apply syntax-first selection and emit unfiltered complexity strata."""

    candidates = [deepcopy(dict(row)) for row in rows]
    grouped: dict[int, list[JsonDict]] = {}
    for row in candidates:
        grouped.setdefault(int(row["request_group_index"]), []).append(row)
    dispositions: list[JsonDict] = []
    for group_index in sorted(grouped):
        group = sorted(grouped[group_index], key=lambda row: int(row["candidate_index"]))
        selected = next(
            (
                row
                for row in group
                if row.get("terminal_state") == "response" and row.get("schema_valid") is True
            ),
            None,
        )
        if selected is not None:
            selected["selected_for_request"] = True
            selected["selection_rank"] = 1
        source = group[0]
        dispositions.append(
            {
                "request_group_index": group_index,
                "stream_id": source.get("stream_id"),
                "request_id": source.get("request_id"),
                "query_type": source.get("query_type"),
                "candidate_call_ids": [row.get("call_id") for row in group],
                "candidate_reply_sha256": [row.get("raw_reply_sha256") for row in group],
                "selection_policy": "first_schema_valid_candidate_in_registered_order",
                "selection_status": (
                    "selected_first_schema_valid" if selected else "no_schema_valid_candidate"
                ),
                "selected_candidate_index": (
                    selected.get("candidate_index") if selected is not None else None
                ),
                "selected_call_id": selected.get("call_id") if selected is not None else None,
                "selected_source_literal_fidelity": (
                    selected.get("source_literal_fidelity") if selected is not None else None
                ),
                "selected_semantic_extendible": (
                    selected.get("semantic_extendible") if selected is not None else None
                ),
                "both_candidates_charged": len(group) == 2,
                "candidate_cost_shares": [
                    row.get("equal_later_comparator_cost_share") for row in group
                ],
            }
        )
    completed = sum(row.get("terminal_state") == "response" for row in candidates)
    attempted = sum(row.get("attempted") is True for row in candidates)
    censored = sum(
        row.get("attempted") is True and row.get("terminal_state") != "response"
        for row in candidates
    )
    unstarted = sum(row.get("attempted") is not True for row in candidates)
    selected_count = sum(row["selected_candidate_index"] is not None for row in dispositions)
    faithful_count = sum(row["selected_source_literal_fidelity"] is True for row in dispositions)
    semantic_count = sum(row["semantic_extendible"] is True for row in candidates)
    complexity = [
        {
            "call_index": row.get("call_index"),
            "call_id": row.get("call_id"),
            "stream_id": row.get("stream_id"),
            "request_id": row.get("request_id"),
            "candidate_index": row.get("candidate_index"),
            "family": row.get("family"),
            "query_type": row.get("query_type"),
            "formula_version": row.get("formula_version"),
            "original_variable_count": row.get("n_vars"),
            "original_clause_count": row.get("exact_solver_clause_count"),
            "exact_solver_duration_ns": row.get("exact_solver_duration_ns"),
            "exact_solver_assumption_count": len(row.get("decoded_assignments") or []),
            "prompt_tokens": row.get("prompt_tokens"),
            "completion_tokens": row.get("completion_tokens"),
            "parse_status": row.get("parse_status"),
            "source_literal_fidelity": row.get("source_literal_fidelity"),
            "semantic_extendible": row.get("semantic_extendible"),
            "terminal_state": row.get("terminal_state"),
            "solver_effort_is_model_hardness": False,
            "post_hoc_filtered": False,
        }
        for row in candidates
    ]
    observed_score = int(
        len(candidates) == 64
        and len(dispositions) == 32
        and attempted == 64
        and completed + censored + unstarted == 64
        and unstarted == 0
        and selected_count == 32
        and faithful_count == 32
    )
    return {
        "candidate_rows": candidates,
        "request_dispositions": dispositions,
        "complexity_rows": complexity,
        "planned_call_count": 64,
        "attempted_call_count": attempted,
        "completed_call_count": completed,
        "censored_call_count": censored,
        "unstarted_call_count": unstarted,
        "selected_request_count": selected_count,
        "source_faithful_selected_request_count": faithful_count,
        "semantic_extendible_candidate_count": semantic_count,
        "candidate_capture_observed_score": observed_score,
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Freeze the exact Exp7358 affected list before subprocess execution."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad, duplicate, missing, or path-invalid affected commands."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require exactly one successful real receipt for each named command."""

    return all(
        len(matched := [row for row in receipts if row.get("name") == name]) == 1
        and matched[0].get("passed") is True
        and matched[0].get("exit_code") == 0
        and matched[0].get("timed_out") is False
        for name in names
    )


def zero_counts() -> JsonDict:
    """Return explicit zero current load and generation dispositions."""

    return deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)


def _acceptance_gates(
    *,
    preconditions_ok: bool,
    reduced: Mapping[str, Any],
    events_ok: bool,
    offload_ok: bool,
    affected_ok: bool,
    terminal_state: str,
    flagged: bool,
) -> list[JsonDict]:
    """Keep completion, evidence, validation, safety, and promotion separate."""

    values = (
        ("prerequisites", "completion", "==", True, preconditions_ok),
        (
            "all_calls_accounted",
            "completion",
            "==",
            64,
            reduced.get("completed_call_count", 0)
            + reduced.get("censored_call_count", 0)
            + reduced.get("unstarted_call_count", 0),
        ),
        ("all_calls_attempted", "completion", "==", 64, reduced.get("attempted_call_count")),
        ("all_requests_selected", "efficacy", "==", 32, reduced.get("selected_request_count")),
        (
            "all_selections_source_faithful",
            "efficacy",
            "==",
            32,
            reduced.get("source_faithful_selected_request_count"),
        ),
        ("qualified_event_reduction", "evidence", "==", True, events_ok),
        ("owned_all_layer_cuda_runtime", "evidence", "==", True, offload_ok),
        ("affected_validation", "validation", "==", True, affected_ok),
        ("terminal_readers", "validation", "in", ["not_yet_executed", "passed"], terminal_state),
        ("adversarial_clean", "safety", "==", False, flagged),
        ("promotion_disabled", "safety", "==", 0, 0),
    )
    return [
        gate_row(
            check,
            EXPERIMENT_ID,
            check,
            operator,
            expected,
            observed,
            category=category,
        )
        for check, category, operator, expected, observed in values
    ]


def _base_artifact() -> JsonDict:
    """Create one schema-complete no-run shape for blocked publication."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_not_started",
        "run_date": RUN_DATE,
        "started_at_utc": None,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_settings": {},
        "model_invoked": False,
        "invocation_counts": zero_counts(),
        "current_invocation_events": [],
        "current_run_id": None,
        "current_owner_pid": None,
        "event_count": 0,
        "event_sha256": canonical_hash([]),
        "inference_mode": "live_gpu",
        "inference_substrate": "blocked_no_run",
        "inference_substrate_details": {},
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": 0.0,
        "scientific_duration_s": 0.0,
        "validation_duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": None,
        "source_artifact_hashes": {},
        "historical_receipt_sidecars": [],
        "rows": [],
        "candidate_rows": [],
        "request_dispositions": [],
        "complexity_rows": [],
        "raw_call_rows": [],
        "sample_size_budget": {
            "planned": 64,
            "attempted": 0,
            "completed": 0,
            "censored": 0,
            "unstarted": 64,
            "max_new_tokens_per_unit": MAX_GENERATED_TOKENS,
            "model_load_timeout_s": MODEL_LOAD_TIMEOUT_S,
            "model_work_timeout_s": MODEL_WORK_TIMEOUT_S,
            "validation_reserve_s": VALIDATION_RESERVE_S,
            "stop_rule": "attempt the 64 registered calls once; no retry, repair, replacement, tuning, or larger budget",
            "effective_independent_group_count": 8,
            "request_group_count": 32,
            "candidates_per_request": 2,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_not_started",
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
        "candidate_capture_complete_score": 0,
        "candidate_capture_observed_score": 0,
        "planned_call_count": 64,
        "attempted_call_count": 0,
        "completed_call_count": 0,
        "censored_call_count": 0,
        "unstarted_call_count": 64,
        "selected_request_count": 0,
        "source_faithful_selected_request_count": 0,
        "semantic_extendible_candidate_count": 0,
        "owned_runtime_receipt": {},
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "fresh_process_cold_replay"],
        "downstream_memory_efficacy_measured": False,
        "learning_claim": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_changed": False,
    }


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    *,
    started_at: str | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    """Publish unchanged missing inputs as blocked with zero current calls."""

    artifact = _base_artifact()
    summary = gate_check_summary(checks)
    artifact.update(
        {
            "status": "blocked_precondition_failed",
            "started_at_utc": started_at,
            "completed_at_utc": started_at,
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "duration_s": duration_s,
            "gate_check_summary": summary,
            "honest_verdict": f"blocked_{summary['check']}",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    reduced: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    current = canary.reduce_current_events(events, run_id="run-test", owner_pid=7402)
    gates = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=True,
        offload_ok=True,
        affected_ok=True,
        terminal_state="passed",
        flagged=False,
    )
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_candidate_capture_ready",
            "started_at_utc": "2026-09-18T00:00:00Z",
            "completed_at_utc": "2026-09-18T00:00:12Z",
            "preconditions_checked": [
                gate_row("fixture", "unit_test", "fixture", "==", True, True)
            ],
            "model_settings": {"max_new_tokens": 256, "seed": RANDOM_SEED["experiment"]},
            **current,
            "current_invocation_events": [deepcopy(dict(row)) for row in events],
            "current_run_id": "run-test",
            "current_owner_pid": 7402,
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {"gpu": "RTX 3090"},
            "inference_substrate_class": "model_bounded_generation",
            "duration_s": 12.0,
            "scientific_duration_s": 10.5,
            "validation_duration_s": 1.5,
            "phase_spans": [
                {
                    "phase": "generation",
                    "start_s": 0.0,
                    "end_s": 10.5,
                    "heartbeat_times_s": [],
                    "checkpoint_times_s": [10.5],
                }
            ],
            "rows": deepcopy(reduced["candidate_rows"]),
            **deepcopy(dict(reduced)),
            "raw_call_rows": [deepcopy(dict(row)) for row in raw_rows],
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": reduced["attempted_call_count"],
                "completed": reduced["completed_call_count"],
                "censored": reduced["censored_call_count"],
                "unstarted": reduced["unstarted_call_count"],
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary(gates),
            "honest_verdict": "complete_circular_positive_candidate_capture_ready_32_of_32",
            "verdict_class": "circular_positive",
            "validation_receipts": [deepcopy(dict(row)) for row in receipts],
            "repository_health": {
                "status": "healthy",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": False,
            },
            "candidate_capture_complete_score": 1,
            "owned_runtime_receipt": {
                "offload": {"all_layers_offloaded": True},
            },
        }
    )
    return artifact


def _raw_rows(artifact: Mapping[str, Any], *, root: Path) -> tuple[list[JsonDict], list[str]]:
    """Reload exact raw candidates and report path, hash, or row drift."""

    manifests = artifact.get("raw_call_rows")
    embedded = artifact.get("candidate_rows")
    if not isinstance(manifests, list) or not isinstance(embedded, list) or not manifests:
        return [], ["raw_evidence_unavailable"]
    rows: list[JsonDict] = []
    errors: list[str] = []
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            errors.append(f"raw_call_manifest_invalid:{index}")
            continue
        path = Path(str(manifest.get("path") or ""))
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            errors.append(f"raw_call_path_missing:{index}")
            continue
        if sha256_file(resolved) != manifest.get("sha256"):
            errors.append(f"raw_call_hash_mismatch:{index}")
            continue
        row = load_object(resolved)
        rows.append(row)
        if index >= len(embedded) or row != embedded[index]:
            errors.append(f"raw_call_row_mismatch:{index}")
    return rows, errors


def independent_reduce_artifact(
    artifact: Mapping[str, Any], *, root: Path = REPO_ROOT
) -> list[str]:
    """Reload raw rows and recompute selection, complexity, calls, and score."""

    rows, errors = _raw_rows(artifact, root=root)
    if errors:
        return errors
    reduced = reduce_panel(rows)
    events = artifact.get("current_invocation_events")
    if not isinstance(events, list):
        return ["current_invocation_events_invalid"]
    try:
        current = canary.reduce_current_events(
            events,
            run_id=str(artifact.get("current_run_id")),
            owner_pid=int(artifact.get("current_owner_pid")),
        )
    except (TypeError, ValueError) as exc:
        return [f"current_event_reduction_failed:{exc}"]
    comparisons = {
        "candidate_rows": reduced["candidate_rows"],
        "rows": reduced["candidate_rows"],
        "request_dispositions": reduced["request_dispositions"],
        "complexity_rows": reduced["complexity_rows"],
        "planned_call_count": reduced["planned_call_count"],
        "attempted_call_count": reduced["attempted_call_count"],
        "completed_call_count": reduced["completed_call_count"],
        "censored_call_count": reduced["censored_call_count"],
        "unstarted_call_count": reduced["unstarted_call_count"],
        "selected_request_count": reduced["selected_request_count"],
        "source_faithful_selected_request_count": reduced["source_faithful_selected_request_count"],
        "semantic_extendible_candidate_count": reduced["semantic_extendible_candidate_count"],
        "candidate_capture_observed_score": reduced["candidate_capture_observed_score"],
        "invocation_counts": current["invocation_counts"],
        "model_invoked": current["model_invoked"],
        "event_count": current["event_count"],
        "event_sha256": current["event_sha256"],
    }
    for field, expected in comparisons.items():
        if artifact.get(field) != expected:
            errors.append(f"{field}_mismatch")
    expected_score = (
        0
        if artifact.get("verdict_class") in {"blocked", "disqualified", "null"}
        else reduced["candidate_capture_observed_score"]
    )
    if artifact.get("candidate_capture_complete_score") != expected_score:
        errors.append("candidate_capture_complete_score_mismatch")
    return list(dict.fromkeys(errors))


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, reductions, declarations, budgets, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        return [f"missing_required_field:{field}" for field in missing]
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS or artifact.get("promotion_score") != 0:
        errors.append("model_contract_invalid")
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
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or budget.get("planned") != (
        int(budget.get("completed", 0) or 0)
        + int(budget.get("censored", 0) or 0)
        + int(budget.get("unstarted", 0) or 0)
    ):
        errors.append("sample_budget_accounting_invalid")
    if artifact.get("model_invoked"):
        if artifact.get("inference_substrate_class") != "model_bounded_generation":
            errors.append("substrate_class_invalid")
        if float(artifact.get("duration_s", 0.0) or 0.0) < 10.0:
            errors.append("bounded_generation_duration_floor")
        errors.extend(independent_reduce_artifact(artifact, root=root))
    elif artifact.get("verdict_class") == "blocked":
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_invalid")
    else:
        errors.append("completed_artifact_without_model_attempt")
    if require_terminal and not receipts_pass(
        list(artifact.get("validation_receipts") or []),
        (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES),
    ):
        errors.append("required_validation_receipts_invalid")
    if artifact.get("verdict_class") in {"blocked", "disqualified", "null"}:
        if artifact.get("candidate_capture_complete_score") != 0:
            errors.append("failed_readiness_invalid")
    elif artifact.get("verdict_class") in {"positive", "circular_positive"}:
        if artifact.get("candidate_capture_complete_score") != 1:
            errors.append("positive_readiness_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def rtx3090_slot_gate_rows(*, query_ok: bool, available_slot_count: int) -> list[JsonDict]:
    """Require at least one free slot without rejecting additional free GPUs."""

    return [
        gate_row(
            "gpu_inventory_query",
            "nvidia-smi_and_gpu_lease_journal",
            "query_ok",
            "==",
            True,
            query_ok,
        ),
        gate_row(
            "one_owned_rtx3090_slot",
            "nvidia-smi_and_gpu_lease_journal",
            "available_rtx3090_slots",
            ">=",
            1,
            available_slot_count,
        ),
    ]


def _runtime_preconditions(
    root: Path, context: JsonDict, started: float
) -> list[JsonDict]:  # pragma: no cover - host and GPU dependent.
    """Reuse the qualified canary preflight and apply this panel's budget."""

    inherited = canary._runtime_preconditions(root, context, started)
    checks = [row for row in inherited if row.get("check") != "one_owned_rtx3090_slot"]
    query_ok = all(row.get("returncode") == 0 for row in context.get("gpu_query_receipts", []))
    checks.extend(
        rtx3090_slot_gate_rows(
            query_ok=query_ok,
            available_slot_count=len(context.get("available_gpu_uuids", [])),
        )
    )
    model_spec = dict(context.get("model_spec") or {})
    decoding = dict(model_spec.get("decoding") or {})
    decoding.update(
        {
            "max_new_tokens": MAX_GENERATED_TOKENS,
            "seed": RANDOM_SEED["experiment"],
            "repair_attempts": 0,
            "retry_budget": 0,
        }
    )
    model_spec["decoding"] = decoding
    context["model_spec"] = model_spec
    return checks


@contextmanager
def _shared_runtime_settings(
    recorder: canary.InvocationEventRecorder,
) -> Iterator[None]:  # pragma: no cover - live runtime mutation is restored.
    """Apply the frozen panel settings to the shipped owned native runner."""

    values = {
        "TASK_ID": TASK_ID,
        "RUN_DATE": RUN_DATE,
        "MAX_GENERATED_TOKENS": MAX_GENERATED_TOKENS,
        "MODEL_LOAD_TIMEOUT_S": MODEL_LOAD_TIMEOUT_S,
        "INFERENCE_WINDOW_TIMEOUT_S": MODEL_WORK_TIMEOUT_S,
        "REQUEST_TIMEOUT_S": REQUEST_TIMEOUT_S,
        "RANDOM_SEED": {
            "development": RANDOM_SEED["experiment"],
            "evaluation": RANDOM_SEED["experiment"],
            "resampling": RANDOM_SEED["resampling"],
        },
        "render_public_prompt": lambda request: str(request["prompt"]),
        "progress": recorder,
    }
    previous = {name: getattr(runtime, name) for name in values}
    try:
        for name, value in values.items():
            setattr(runtime, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(runtime, name, value)


def _capture_current(
    context: Mapping[str, Any], raw_dir: Path, started: float
) -> JsonDict:  # pragma: no cover - live model work.
    """Run the 64 registered calls through one task-owned model instance."""

    recorder = canary.InvocationEventRecorder(
        f"{EXPERIMENT_ID}:{os.getpid()}:{time.monotonic_ns()}", os.getpid(), started
    )
    runtime_context = deepcopy(dict(context))
    runtime_context["selected_requests"] = deepcopy(list(context["schedule"]))
    with _shared_runtime_settings(recorder):
        capture = runtime._live_capture(runtime_context, raw_dir / "native")
    recorder.close(capture)
    identity = deepcopy(dict(capture.get("runtime_identity") or {}))
    identity.update(
        {
            "server_executable_path": str(context.get("server_path") or ""),
            "server_executable_sha256": context.get("server_sha256"),
            "native_tokenizer": dict(context.get("model_spec") or {}).get("native_tokenizer"),
            "native_chat_template": dict(context.get("model_spec") or {}).get(
                "native_chat_template"
            ),
        }
    )
    runtime_rows = list(capture.get("rows") or [])
    rows: list[JsonDict] = []
    for index, schedule in enumerate(context["schedule"]):
        source = dict(runtime_rows[index]) if index < len(runtime_rows) else {}
        rows.append(build_candidate_row(schedule, source, identity))
        progress(
            started,
            "generation",
            "checkpoint",
            completed=index + 1,
            total=64,
            terminal_state=rows[-1]["terminal_state"],
        )
    capture.update(
        {
            "rows": rows,
            "runtime_identity": identity,
            "current_run_id": recorder.run_id,
            "current_owner_pid": recorder.owner_pid,
            "current_invocation_events": recorder.events,
        }
    )
    return capture


def _span(
    spans: list[JsonDict],
    phase: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
) -> None:  # pragma: no cover - real timing evidence.
    """Close one actual phase with monotonic and UTC checkpoints."""

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
        }
    )


def _run_affected_validation(
    root: Path, raw_dir: Path, started: float
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover - subprocess orchestration.
    """Execute the frozen Exp7358 plan through the streaming Exp7303 runner."""

    private = Path(tempfile.mkdtemp(prefix="exp7402-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private)
    plan_errors = validate_validation_plan(root, commands)
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
    """Build fresh-process replay and unchanged strict-reader commands."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7402_v649_proposal_capture import independent_reduce_artifact;"
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


def _source_hashes(
    root: Path,
    context: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    historical_sidecar: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - measured source evidence.
    """Bind exact sources, inputs, model, runner, sidecar, and raw calls."""

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
        PRODUCER_PATH,
        PROTOCOL_PATH,
        Path("python/carnot/experiment_7400_v649_assignment_canary.py"),
        Path("python/carnot/experiment_7372_v647_qwen_canary.py"),
        Path("python/carnot/experiment_7371_v647_proof_boundary.py"),
        Path("python/carnot/experiment_7358_v646_validation_contract.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
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
        hashes[str(row["path"])] = str(row["sha256"])
    hashes[str(historical_sidecar["path"])] = str(historical_sidecar["sha256"])
    return hashes


def run_experiment(
    *, root: Path = REPO_ROOT, run_date: str = RUN_DATE, output_path: Path | None = None
) -> JsonDict:  # pragma: no cover - bounded live orchestration.
    """Authenticate, capture, validate, cold-replay, and atomically publish."""

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
        0, gate_row("run_date", "execution_contract", "run_date", "==", RUN_DATE, run_date)
    )
    _span(spans, "preconditions_static", phase_started, started, len(checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks, started_at=started_at, duration_s=round(time.monotonic() - started, 6)
        )
        artifact["phase_spans"] = spans
        artifact["completed_at_utc"] = utc_now()
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    phase_started = time.monotonic()
    progress(started, "preconditions", "runtime_start")
    runtime_checks = _runtime_preconditions(root, context, started)
    checks.extend(runtime_checks)
    progress(
        started, "preconditions", "runtime_complete", passed=all(row["passed"] for row in checks)
    )
    _span(spans, "preconditions_runtime", phase_started, started, len(runtime_checks))
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks, started_at=started_at, duration_s=round(time.monotonic() - started, 6)
        )
        artifact["phase_spans"] = spans
        artifact["model_settings"] = deepcopy(context.get("model_spec") or {})
        artifact["completed_at_utc"] = utc_now()
        artifact["reproducibility_checksum"] = artifact_checksum(artifact)
        atomic_json(output, artifact)
        return artifact

    historical_sidecar = current_work_receipt.write_immutable_sidecar(
        raw_dir / "historical_exp7400_canary.json",
        scope="historical_model_receipts",
        payload={
            "artifact_path": PRODUCER_PATH.as_posix(),
            "artifact_sha256": context["producer_sha256"],
            "original_verdict_class": context["producer"].get("verdict_class"),
            "original_flagged_adversarial": context["producer"].get("flagged_adversarial"),
            "qwen_assignment_transport_ready_score": context["producer"].get(
                "qwen_assignment_transport_ready_score"
            ),
            "authorizes_current_inference_counts": False,
        },
        root=root,
    )

    phase_started = time.monotonic()
    scientific_started = phase_started
    progress(started, "generation", "before_model_load_and_generation", planned_calls=64)
    capture = _capture_current(context, raw_dir / "owned_runtime", started)
    progress(
        started,
        "generation",
        "after_model_load_and_generation",
        completed_calls=len(capture.get("rows") or []),
    )
    _span(spans, "model_load_and_generation", phase_started, started, len(capture["rows"]))
    scientific_duration = time.monotonic() - scientific_started

    phase_started = time.monotonic()
    reduced = reduce_panel(capture["rows"])
    raw_rows: list[JsonDict] = []
    for index, row in enumerate(reduced["candidate_rows"]):
        path = raw_dir / f"call_{index:02d}.json"
        atomic_json(path, row)
        raw_rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
        progress(started, "reduction", "checkpoint", completed=index + 1, total=64)
    current = canary.reduce_current_events(
        capture["current_invocation_events"],
        run_id=capture["current_run_id"],
        owner_pid=capture["current_owner_pid"],
    )
    provenance = dict(capture.get("gpu_receipts") or {}).get("provenance") or {}
    offload = canary.build_offload_receipt(
        dict(capture.get("runtime_identity") or {}),
        provenance,
        int(context["model_spec"]["model_block_count"]),
    )
    _span(spans, "reduction", phase_started, started, len(raw_rows))

    validation_started = time.monotonic()
    affected_receipts, affected = _run_affected_validation(root, raw_dir, started)
    affected_ok = bool(affected.get("passed"))
    observed_ready = int(
        reduced["candidate_capture_observed_score"] == 1
        and offload["all_layers_offloaded"] is True
        and current["model_invoked"] is True
        and affected_ok
    )
    gates = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=True,
        offload_ok=offload["all_layers_offloaded"] is True,
        affected_ok=affected_ok,
        terminal_state="not_yet_executed",
        flagged=False,
    )
    artifact = _base_artifact()
    artifact.update(
        {
            "status": "complete_candidate_awaiting_terminal_readers",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "preconditions_checked": [deepcopy(dict(row)) for row in checks],
            "model_settings": deepcopy(context["model_spec"]),
            **current,
            "current_invocation_events": deepcopy(capture["current_invocation_events"]),
            "current_run_id": capture["current_run_id"],
            "current_owner_pid": capture["current_owner_pid"],
            "inference_substrate": "owned_native_cuda_llama_cpp_bounded_generation",
            "inference_substrate_details": {
                "model": MODEL_ID,
                "quantization": QUANTIZATION,
                "runner": str(context["server_path"]),
                "gpu_uuid": dict(capture.get("runtime_identity") or {}).get("gpu_uuid"),
                "gpu_name": "NVIDIA GeForce RTX 3090",
                "embedded_tokenizer": True,
                "embedded_chat_template": True,
            },
            "inference_substrate_class": "model_bounded_generation",
            "scientific_duration_s": round(scientific_duration, 6),
            "phase_spans": spans,
            "historical_receipt_sidecars": [historical_sidecar],
            "rows": deepcopy(reduced["candidate_rows"]),
            **deepcopy(reduced),
            "raw_call_rows": raw_rows,
            "sample_size_budget": {
                **artifact["sample_size_budget"],
                "attempted": reduced["attempted_call_count"],
                "completed": reduced["completed_call_count"],
                "censored": reduced["censored_call_count"],
                "unstarted": reduced["unstarted_call_count"],
            },
            "acceptance_gate_results": gates,
            "gate_check_summary": gate_check_summary([*checks, *gates]),
            "honest_verdict": (
                "complete_circular_positive_candidate_capture_ready_32_of_32"
                if observed_ready
                else f"complete_null_candidate_capture_low_yield_{reduced['source_faithful_selected_request_count']}_of_32"
            ),
            "verdict_class": "circular_positive" if observed_ready else "null",
            "validation_receipts": affected_receipts,
            "repository_health": {
                "status": "healthy" if affected_ok else "required_checks_failed",
                "unrelated_broad_suite_observations": [],
                "affects_required_checks": not affected_ok,
                "affected_reduction": affected,
            },
            "candidate_capture_complete_score": observed_ready,
            "owned_runtime_receipt": {
                **deepcopy(dict(capture.get("runtime_identity") or {})),
                "model_load": deepcopy(capture.get("load_receipt") or {}),
                "gpu_provenance": deepcopy(provenance),
                "offload": offload,
                "cleanup": deepcopy(dict(capture.get("gpu_receipts") or {}).get("cleanup") or {}),
                "lease_release": deepcopy(
                    dict(capture.get("gpu_receipts") or {}).get("lease_release") or {}
                ),
            },
        }
    )
    artifact["source_artifact_hashes"] = _source_hashes(root, context, raw_rows, historical_sidecar)
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    candidate = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate, artifact)

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
    terminal_ok = receipts_pass(terminal_receipts, TERMINAL_CHECK_NAMES)
    independent_ok = not independent_reduce_artifact(artifact, root=root)
    adversarial = next(
        (row for row in terminal_receipts if row.get("name") == "adversarial_verify"), {}
    )
    flagged = adversarial.get("passed") is not True
    _span(
        spans,
        "validation",
        validation_started,
        started,
        len(affected_receipts) + len(terminal_receipts),
    )
    final_ready = int(observed_ready and terminal_ok and independent_ok and not flagged)
    disqualified = not affected_ok or not terminal_ok or not independent_ok or flagged
    if disqualified:
        artifact.update(
            {
                "status": "complete_required_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_candidate_capture_required_check_failed",
                "candidate_capture_complete_score": 0,
            }
        )
    elif final_ready:
        artifact["status"] = "complete_candidate_capture_ready"
        artifact["verdict_class"] = "circular_positive"
    else:
        artifact["status"] = "complete_candidate_capture_null"
        artifact["verdict_class"] = "null"
    artifact["flagged_adversarial"] = flagged
    artifact["candidate_capture_complete_score"] = final_ready
    artifact["validation_receipts"] = [*affected_receipts, *terminal_receipts]
    artifact["acceptance_gate_results"] = _acceptance_gates(
        preconditions_ok=True,
        reduced=reduced,
        events_ok=independent_ok,
        offload_ok=offload["all_layers_offloaded"] is True,
        affected_ok=affected_ok,
        terminal_state="passed" if terminal_ok else "failed",
        flagged=flagged,
    )
    artifact["gate_check_summary"] = gate_check_summary(
        [*checks, *artifact["acceptance_gate_results"]]
    )
    artifact["phase_spans"] = spans
    artifact["completed_at_utc"] = utc_now()
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["validation_duration_s"] = round(time.monotonic() - validation_started, 6)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        artifact.update(
            {
                "status": "complete_internal_validation_failed",
                "verdict_class": "disqualified",
                "honest_verdict": "complete_disqualified_internal_artifact_validation_failed",
                "flagged_adversarial": True,
                "candidate_capture_complete_score": 0,
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
    """Reject execution outside the fixed V649 date."""

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
        errors = validate_artifact(
            load_object(args.validate), root=REPO_ROOT, require_terminal=False
        )
        print(canonical_json({"errors": errors}), flush=True)
        return int(bool(errors))
    result = run_experiment(root=REPO_ROOT, run_date=args.date)
    print(
        canonical_json(
            {
                "artifact": str(REPO_ROOT / RESULT_PATH),
                "status": result["status"],
                "honest_verdict": result["honest_verdict"],
                "candidate_capture_complete_score": result["candidate_capture_complete_score"],
                "selected_request_count": result["selected_request_count"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
