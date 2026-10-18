"""Run a two-episode live ARC reachability sentinel.

This experiment qualifies first-action and generation-callback reachability. It
does not estimate efficacy. The live child uses the submitted agent factory,
the shared request budget, and the shared invocation boundary. This module adds
only durable observation capture and an independent terminal reduction.

Spec refs: REQ-ARC-WMTE-7431 and SCENARIO-ARC-WMTE-7431-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.request

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7376_v647_arc_outcomes as shipped_live
from carnot.agentic.arc_inference_boundary import (
    BOUNDARY_LEDGER_ENV,
    InvocationBoundaryLedger,
    reduce_boundary_events,
)
from carnot.agentic.arc_request_budget import (
    EpisodeRequestBudget,
    RequestReservation,
    attach_request_budget,
)
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
PHASE = 3
EXPERIMENT_ID = "exp7431-v651-arc-live-sentinel"
TASK_ID = "experiment_7431_v651_arc_live_sentinel"
SCHEMA = "carnot.exp7431.v651.arc_live_sentinel.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
RANDOM_SEED = 7_431_651
ACTION_LIMIT = 64
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_LIMIT_S = 240.0
AGGREGATE_LIVE_LIMIT_S = 900.0
MODEL_LOAD_LIMIT_S = 600.0

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
EXP7422_PATH = Path("results/experiment_7422_v651_runtime_ownership.json")
EXP7424_PATH = Path("results/experiment_7424_v651_arc_receipt_boundary.json")
RESULT_PATH = Path("results/experiment_7431_v651_arc_live_sentinel.json")
RAW_DIR = Path("results/raw/experiment_7431_v651_arc_live_sentinel")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7431_v651_arc_live_sentinel.json")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7431_v651_arc_live_sentinel.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7431_v651_arc_live_sentinel.py")
TEST_PATH = Path("tests/python/test_experiment_7431_v651_arc_live_sentinel.py")

REQUIRED_E2E = ("e2e_009", "e2e_010", "e2e_009_llm_off_environment")
REQUIRED_TERMINAL = (
    "fresh_process_cold_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
TERMINAL_DISPOSITIONS = {
    "complete",
    "complete_error",
    "failed",
    "censored_timeout",
    "censored_no_first_action",
    "unstarted",
}
WITHHELD_INPUTS = (
    "per_game_adapter",
    "banked_solution",
    "hand_model",
    "hand_solver",
    "saved_engine",
    "lookup_solver",
    "replay_route",
    "hidden_game_source",
    "offline_ground_truth_bfs",
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7411_v650_arc_call_budget.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_request_budget.py"),
    Path("python/carnot/agentic/arc_inference_boundary.py"),
    Path("python/carnot/experiment_7406_v649_arc_generalization.py"),
    Path("python/carnot/experiment_7263_v639_arc_live.py"),
    Path("scripts/arc_loop_solve.py"),
    REGISTRY_PATH,
    EXP7422_PATH,
    EXP7424_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a truthful phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7431] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so evidence drift changes the identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a complete JSON object after flushing it to storage."""

    current_work_receipt.atomic_json(path, value)


def load_object(path: Path) -> JsonDict:
    """Load one JSON object, or return an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum value itself."""

    copied = deepcopy(dict(value))
    copied["reproducibility_checksum"] = ""
    return canonical_hash(copied)


def _compare(operator: str, expected: Any, observed: Any) -> bool:
    if operator == "==":
        return observed == expected
    if operator == "in":
        return observed in expected
    if operator == ">=":
        return observed is not None and observed >= expected
    raise ValueError(f"unsupported operator: {operator}")


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str,
    artifact_field: str,
    principle: str,
    operator: str = "==",
) -> JsonDict:
    """Retain one exact check with its source and interpretation."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": _compare(operator, expected, observed),
        "upstream": upstream,
        "artifact_field": artifact_field,
        "principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every failed gate and name the first exact failure."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": failures,
        "first_failure": failures[0] if failures else None,
    }


def _source_record(path: Path, *, role: str) -> JsonDict:
    record: JsonDict = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "role": role,
    }
    if path.suffix == ".json":
        value = load_object(path)
        record["original_flags"] = {
            key: value.get(key) for key in ("status", "verdict_class", "flagged_adversarial")
        }
    return record


def collect_preconditions(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate exact inputs and both repaired upstream boundaries."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "precondition",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=relative.as_posix(),
                artifact_field="bytes",
                principle="Dependent work starts only after exact source bytes exist.",
            )
        )
        if available:
            role = (
                "structured_prerequisite"
                if relative in {EXP7422_PATH, EXP7424_PATH}
                else "current_input"
            )
            hashes[relative.as_posix()] = _source_record(path, role=role)

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        gate_row(
            "driving_requirement",
            "precondition",
            True,
            "REQ-ARC-WMTE-7431" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7431",
            principle="The capability contract must exist before runtime work.",
        )
    )
    prerequisites = (
        (EXP7422_PATH, "runtime_ownership_ready_score"),
        (EXP7424_PATH, "arc_receipt_boundary_ready_score"),
    )
    for path, score_field in prerequisites:
        value = load_object(root / path)
        for field, expected, operator in (
            (score_field, 1, "=="),
            ("verdict_class", ["positive", "circular_positive", "null"], "in"),
            ("flagged_adversarial", False, "=="),
        ):
            checks.append(
                gate_row(
                    f"{path.stem}.{field}",
                    "precondition",
                    expected,
                    value.get(field),
                    upstream=path.as_posix(),
                    artifact_field=field,
                    operator=operator,
                    principle="Only a ready, eligible, unflagged repair can authorize live work.",
                )
            )
    checks.append(
        gate_row(
            "force_live",
            "precondition",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
            upstream="environment",
            artifact_field="CARNOT_FORCE_LIVE",
            principle="The sentinel must never fall back to simulated inference.",
        )
    )
    try:
        exclusion = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
        registry = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
        parse_ok = isinstance(exclusion, Mapping) and isinstance(registry, Mapping)
    except (OSError, yaml.YAMLError):
        exclusion, registry, parse_ok = {}, {}, False
    checks.append(
        gate_row(
            "manifest_and_registry_parse",
            "precondition",
            True,
            parse_ok,
            upstream=f"{EXCLUSION_PATH.as_posix()}|{REGISTRY_PATH.as_posix()}",
            artifact_field="valid_yaml_objects",
            principle="Selection and quarantine checks need readable structured metadata.",
        )
    )
    excluded_text = json.dumps(exclusion, sort_keys=True)
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "precondition",
            False,
            EXPERIMENT_ID in excluded_text or '"experiment_id": 7431' in excluded_text,
            upstream=EXCLUSION_PATH.as_posix(),
            artifact_field=EXPERIMENT_ID,
            principle="A retired task cannot start new model work.",
        )
    )
    return checks, hashes, dict(registry) if isinstance(registry, Mapping) else {}


def freeze_two_game_rotation(registry: Mapping[str, Any]) -> JsonDict:
    """Take the first two games from the shipped label-blind rotation."""

    from carnot.agentic.arc_game_adapters import adaptered_games

    source = shipped_live.freeze_panel(registry, adaptered_games=set(adaptered_games()))
    rows = [deepcopy(dict(row)) for row in source.get("game_rows", [])[:2]]
    games = [str(row["game"]) for row in rows]
    return {
        "passed": source.get("passed") is True and len(rows) == 2,
        "games": games,
        "game_rows": rows,
        "selection_basis": source.get("selection_basis"),
        "label_blind": True,
        "registry_prechecked": True,
        "selection_used_current_outcomes": False,
        "game_source_read": False,
        "offline_ground_truth_search_used": False,
        "policy_received_registry_data": False,
    }


def build_schedule(games: Sequence[str]) -> list[JsonDict]:
    """Seal one seed for each of exactly two selected games."""

    return [
        {
            "episode_id": f"{game}:seed-{RANDOM_SEED}",
            "game": str(game),
            "seed": RANDOM_SEED,
            "execution_order": index,
            "sentinel": index == 0,
            "action_limit": ACTION_LIMIT,
            "request_limit": REQUEST_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "episode_limit_s": EPISODE_LIMIT_S,
            "adapter_disabled": True,
            "banked_solution_disabled": True,
            "hand_model_disabled": True,
            "hand_solver_disabled": True,
            "saved_engine_disabled": True,
            "lookup_solver_disabled": True,
            "hidden_game_source_disabled": True,
            "offline_ground_truth_bfs_disabled": True,
            "withheld_inputs": list(WITHHELD_INPUTS),
        }
        for index, game in enumerate(games[:2])
    ]


def reduce_current_invocations(
    events: Sequence[Mapping[str, Any]], *, child_terminal: bool
) -> JsonDict:
    """Reduce raw shared-boundary events without using projected episode rows."""

    raw = reduce_boundary_events(events)
    raw_counts = dict(raw.get("invocation_counts") or {})
    counts = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
    for key in counts:
        if key.endswith("_cancelled"):
            continue
        counts[key] = int(raw_counts.get(key) or 0)
    if child_terminal:
        for prefix in ("model_loads", "generation_calls"):
            in_flight_key = f"{prefix}_in_flight"
            counts[f"{prefix}_cancelled"] = counts[in_flight_key]
            counts[in_flight_key] = 0
    attempted = counts["model_loads_attempted"] + counts["generation_calls_attempted"]
    generated = counts["generation_calls_attempted"] > 0
    loaded = counts["model_loads_attempted"] > 0
    substrate_class = (
        "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if loaded
        else "no_model_load"
    )
    return {
        "model_invoked": attempted > 0,
        "invocation_counts": counts,
        "inference_substrate_class": substrate_class,
        "call_rows": deepcopy(list(raw.get("call_rows") or [])),
        "errors": deepcopy(list(raw.get("errors") or [])),
        "duplicate_event_count": int(raw.get("duplicate_event_count") or 0),
    }


def reduce_request_budget_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check permit-to-request-to-terminal chains and excess dispatch."""

    per_episode: Counter[str] = Counter()
    terminal = 0
    dispatched = 0
    permit_to_terminal = 0
    unterminated = 0
    ordering_errors = 0
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        per_episode[episode_id] += 1
        disposition = row.get("disposition")
        is_terminal = disposition in {"completed", "failed", "cancelled"}
        terminal += int(is_terminal)
        dispatch = row.get("request_dispatched") is True
        dispatched += int(dispatch)
        permit_to_terminal += int(is_terminal)
        unterminated += int(not is_terminal)
        reserved = row.get("reserved_monotonic")
        request_started = row.get("request_started_monotonic")
        if (
            dispatch
            and isinstance(reserved, (int, float))
            and isinstance(request_started, (int, float))
            and request_started < reserved
        ):
            ordering_errors += 1
    excess = sum(max(0, count - REQUEST_LIMIT) for count in per_episode.values())
    return {
        "permits_acquired": len(rows),
        "requests_dispatched": dispatched,
        "terminal_permits": terminal,
        "permit_to_terminal_chains": permit_to_terminal,
        "unterminated_permits": unterminated,
        "excess_dispatches": excess,
        "ordering_errors": ordering_errors,
        "zero_excess_dispatch": excess == 0,
        "accounting_valid": len(rows) == terminal + unterminated,
    }


def _unstarted_row(schedule: Mapping[str, Any]) -> JsonDict:
    return {
        **deepcopy(dict(schedule)),
        "disposition": "unstarted",
        "policy_entry": None,
        "first_action": None,
        "first_action_latency_s": None,
        "first_observation": None,
        "actions": 0,
        "start_level": None,
        "max_level": None,
        "level_progress": 0,
        "progress_score": 0,
        "tool_feedback_consumed": 0,
        "supervisor": {"fired": 0, "consumed": 0},
        "elapsed_s": 0.0,
        "request_budget_receipt": None,
        "solve_credit": 0,
    }


def reduce_episode_panel(
    schedule: Sequence[Mapping[str, Any]], episode_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Account for each scheduled unit and enforce the sentinel stop rule."""

    by_id = {
        str(row.get("episode_id")): deepcopy(dict(row))
        for row in episode_rows
        if isinstance(row, Mapping) and row.get("episode_id")
    }
    rows: list[JsonDict] = []
    sentinel_reached = False
    sentinel_started = False
    for index, sealed in enumerate(schedule):
        episode_id = str(sealed.get("episode_id"))
        observed = by_id.get(episode_id)
        if index == 0:
            sentinel_started = observed is not None
            sentinel_reached = bool(observed and observed.get("first_action"))
        if observed is None or (index > 0 and not sentinel_reached):
            row = _unstarted_row(sealed)
        else:
            row = {**deepcopy(dict(sealed)), **observed}
            row.setdefault("solve_credit", 0)
        row["progress_score"] = int(row.get("level_progress") or 0)
        rows.append(row)
    dispositions = Counter(str(row.get("disposition")) for row in rows)
    completed = sum(row.get("disposition") in {"complete", "complete_error"} for row in rows)
    failed = dispositions["failed"]
    censored = sum(str(row.get("disposition", "")).startswith("censored_") for row in rows)
    unstarted = dispositions["unstarted"]
    attempted = len(rows) - unstarted
    terminal_valid = (
        len(rows) == 2
        and sentinel_started
        and all(row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows)
        and (sentinel_reached or rows[1].get("disposition") == "unstarted")
    )
    budget = {
        "planned_units": 2,
        "attempted_units": attempted,
        "completed_units": completed,
        "failed_units": failed,
        "censored_units": censored,
        "unstarted_units": unstarted,
        "independent_groups": ["public_adapter_withheld_development_proxy"],
        "stopping_rule": "Stop after two terminal episodes, or after sentinel failure before first action.",
        "request_limit_per_episode": REQUEST_LIMIT,
        "action_limit_per_episode": ACTION_LIMIT,
        "episode_limit_s": EPISODE_LIMIT_S,
        "aggregate_live_limit_s": AGGREGATE_LIVE_LIMIT_S,
    }
    return {
        "per_game_results": rows,
        "sample_size_budget": budget,
        "sentinel_started": sentinel_started,
        "sentinel_reached_first_action": sentinel_reached,
        "terminal_dispositions_valid": terminal_valid,
        "arc_sentinel_capture_complete_score": int(terminal_valid),
    }


def _request_rows(episodes: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for episode in episodes:
        receipt = episode.get("request_budget_receipt")
        if not isinstance(receipt, Mapping):
            continue
        transport = {
            int(row.get("call_index") or 0): dict(row)
            for row in episode.get("server_request_rows", [])
            if isinstance(row, Mapping)
        }
        for row in receipt.get("callback_rows", []):
            if not isinstance(row, Mapping):
                continue
            item = deepcopy(dict(row))
            request = transport.get(int(item.get("reservation_index") or 0), {})
            item.update(
                {
                    "request_dispatched": request.get(
                        "request_dispatched", item.get("request_dispatched")
                    )
                    is True,
                    "response_observed": request.get(
                        "response_observed", item.get("response_observed")
                    )
                    is True,
                    "request_started_monotonic": request.get("request_started_monotonic"),
                    "request_sha256": request.get("request_sha256"),
                    "response_sha256": request.get("response_sha256"),
                    "transport_error": request.get("error"),
                }
            )
            rows.append(item)
    return rows


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _field_principles(keys: Sequence[str]) -> dict[str, str]:
    """Explain field intent without wrapping machine-readable scalars."""

    specific = {
        "schema": "A versioned plain schema binds the experiment, milestone, and terminal status.",
        "run_date": "The fixed run date stays separate from actual UTC boundaries.",
        "preconditions_checked": "Exact paths, identities, and observed values precede dependent work.",
        "MODEL_SPECS": "Every current LLM attempt uses the mandated Qwen repository.",
        "model_invoked": "True means an owned current load or generation attempt occurred.",
        "invocation_counts": "Only persisted current boundary events supply load and generation counts.",
        "inference_substrate": "The string describes actual host-orchestrated native model work.",
        "inference_substrate_class": "The class follows attempted work and never follows padding.",
        "execution_venue": "Host is the closed venue; CUDA identity remains in details.",
        "duration_s": "Monotonic current work is split into validation, model, and cold-reader phases.",
        "phase_spans": "Real boundaries expose completed units and checkpoint paths.",
        "random_seed": "Frozen selection and episode seeds make the small panel repeatable.",
        "reproducibility_checksum": "The digest binds code, inputs, raw rows, and validation scope.",
        "source_artifact_hashes": "Exact input hashes preserve original upstream flags.",
        "rows": "Every scheduled unit remains visible, including censored and unstarted units.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted counts stay separate.",
        "acceptance_gate_results": "Each check states category, operator, expected, observed, pass, and principle.",
        "gate_check_summary": "A blocked result names the exact upstream field and observed value.",
        "verifier_is_oracle": "True states that the public environment defines progress observations.",
        "honest_verdict": "A complete finding is limited to sentinel reachability.",
        "verdict_class": "Null means two episodes do not establish comparative benefit.",
        "flagged_adversarial": "Critical verifier findings cannot supply readiness.",
        "validation_receipts": "Exact commands, environments, exits, durations, and hashed logs remain auditable.",
        "promotion_score": "Zero forbids rollout, publication, or generator-weight updates.",
        "arc_sentinel_capture_complete_score": "One means both scheduled units have valid terminal dispositions.",
        "per_game_results": "Each game records withheld inputs, permits, actions, latency, progress, and cost.",
        "solve_provenance": "Live self-discovery labels only the actual scored-policy runtime path.",
        "request_budget_rows": "Every permit is joined to dispatch and terminal transport observations.",
        "supervisor_outcomes": "Firing and consumption observations carry no efficacy interpretation.",
        "live_efficacy_score": "Zero prevents a two-episode reachability check from becoming a gain claim.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable current evidence.")
        for key in keys
    }


def _duration_breakdown(spans: Sequence[Mapping[str, Any]]) -> JsonDict:
    totals: Counter[str] = Counter()
    for row in spans:
        duration = float(row.get("duration_s") or 0.0)
        phase = str(row.get("phase") or "other")
        if "validation" in phase or "e2e" in phase:
            totals["validation_s"] += duration
        elif "live" in phase or "model" in phase:
            totals["model_s"] += duration
        elif "terminal" in phase or "cold" in phase:
            totals["cold_start_s"] += duration
        else:
            totals["other_s"] += duration
    return {
        "validation_s": totals["validation_s"],
        "model_s": totals["model_s"],
        "cold_start_s": totals["cold_start_s"],
        "other_s": totals["other_s"],
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one independently reducible terminal record."""

    panel = reduce_episode_panel(schedule, episode_rows)
    current = reduce_current_invocations(boundary_events, child_terminal=True)
    request_rows = _request_rows(panel["per_game_results"])
    request_reduction = reduce_request_budget_rows(request_rows)
    required = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E)
    validation_ok = _receipts_pass(validation_receipts, required)
    terminal_ok = not require_terminal or _receipts_pass(validation_receipts, REQUIRED_TERMINAL)
    preconditions_ok = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            preconditions_ok,
            upstream="preconditions_checked",
            artifact_field="all_inputs_and_resources",
            principle="Unavailable inputs stop dependent work.",
        ),
        gate_row(
            "label_blind_two_game_schedule",
            "method",
            True,
            selection.get("passed") is True and len(schedule) == 2,
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="selection_receipt",
            principle="Game choice must precede outcomes and expose no labels to policy.",
        ),
        gate_row(
            "current_invocation_ledger",
            "provenance",
            [],
            current["errors"],
            upstream=BOUNDARY_PATH.as_posix(),
            artifact_field="current_invocation_events",
            principle="Only valid current boundary events can supply model counts.",
        ),
        gate_row(
            "request_budget_safety",
            "safety",
            True,
            request_reduction["accounting_valid"]
            and request_reduction["zero_excess_dispatch"]
            and request_reduction["unterminated_permits"] == 0
            and request_reduction["ordering_errors"] == 0,
            upstream="request_budget_rows",
            artifact_field="permit_request_terminal_chain",
            principle="Every callback consumes a permit before dispatch and becomes terminal.",
        ),
        gate_row(
            "terminal_episode_dispositions",
            "completion",
            True,
            panel["terminal_dispositions_valid"],
            upstream="per_game_results",
            artifact_field="two_scheduled_units",
            principle="A failed sentinel still leaves the second unit explicitly unstarted.",
        ),
        gate_row(
            "required_validation",
            "required_validation",
            True,
            validation_ok and terminal_ok,
            upstream="validation_receipts",
            artifact_field="affected_e2e_and_terminal_checks",
            principle="Affected failures disqualify the candidate.",
        ),
        gate_row(
            "automatic_promotion",
            "promotion",
            0,
            0,
            upstream="protocol",
            artifact_field="promotion_score",
            principle="A reachability sentinel cannot authorize rollout or publication.",
        ),
    ]
    ready = all(row["passed"] for row in gates)
    attempted = current["model_invoked"]
    generated = current["invocation_counts"]["generation_calls_attempted"] > 0
    substrate = current["inference_substrate_class"]
    runtime_path_used = any(row.get("policy_entry") for row in panel["per_game_results"])
    status = (
        "complete_null_arc_live_sentinel" if ready else "complete_disqualified_required_evidence"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(MODEL_SPECS if attempted else []),
        "model_specs": [deepcopy(dict(row)) for row in model_specs] if attempted else [],
        "model_invoked": attempted,
        "invocation_counts": deepcopy(current["invocation_counts"]),
        "current_invocation_events": [deepcopy(dict(row)) for row in boundary_events],
        "current_invocation_call_rows": deepcopy(current["call_rows"]),
        "inference_substrate": (
            "owned_native_cuda_llama_cpp_bounded_arc_generation"
            if generated
            else "owned_native_cuda_llama_cpp_model_load_no_generation"
            if attempted
            else "no_model_load"
        ),
        "inference_substrate_details": {
            "host": platform.node(),
            "python": platform.python_version(),
            "model": MODEL_ID if attempted else None,
            "quantization": QUANTIZATION if attempted else None,
            "runner": "LocalGGUFProposer_native_llama.cpp" if attempted else None,
            **deepcopy(dict(runtime_receipt)),
        },
        "inference_substrate_class": substrate,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "duration_breakdown_s": _duration_breakdown(phase_spans),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "selection": RANDOM_SEED,
            "episodes": [RANDOM_SEED, RANDOM_SEED],
            "sampling": RANDOM_SEED,
            "resampling": None,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "schedule_rows": [deepcopy(dict(row)) for row in schedule],
        "selection_receipt": deepcopy(dict(selection)),
        "rows": deepcopy(panel["per_game_results"]),
        "sample_size_budget": deepcopy(panel["sample_size_budget"]),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_null_arc_live_sentinel_reachability_no_efficacy_claim"
            if ready
            else "complete_disqualified_required_evidence"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_check_names": [*required, *(REQUIRED_TERMINAL if require_terminal else ())],
        "repository_health": {
            "status": "scoped_only",
            "unrelated_failures": [],
            "affects_required_checks": not ready,
        },
        "field_principles": {},
        "promotion_score": 0,
        "arc_sentinel_capture_complete_score": panel["arc_sentinel_capture_complete_score"],
        "per_game_results": deepcopy(panel["per_game_results"]),
        "solve_provenance": (
            "live_agent_self_discovery" if runtime_path_used else "uncredited_no_runtime"
        ),
        "solve_credit": 0,
        "new_level_credit": 0,
        "new_level_reproduction_required": any(
            int(row.get("level_progress") or 0) > 0 for row in panel["per_game_results"]
        ),
        "public_development_generalization_proxy": True,
        "hidden_leaderboard_claimed": False,
        "request_budget_rows": request_rows,
        "request_budget_reduction": request_reduction,
        "supervisor_outcomes": [
            {
                "episode_id": row.get("episode_id"),
                **deepcopy(dict(row.get("supervisor") or {"fired": 0, "consumed": 0})),
                "efficacy_inference": False,
            }
            for row in panel["per_game_results"]
        ],
        "live_efficacy_score": 0,
        "paired_control_present": False,
        "treatment_effect_claimed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "supervisor_arm_added": False,
        "supervisor_fitted": False,
        "solve_registry_changed": False,
        "research_conductor_changed": False,
        "small_ebm_training": {"performed": False},
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a deterministic terminal fixture through production reducers."""

    selection = {
        "passed": True,
        "games": [row["game"] for row in schedule],
        "label_blind": True,
        "registry_prechecked": True,
    }
    preconditions = [
        gate_row(
            "fixture",
            "precondition",
            True,
            True,
            upstream="test",
            artifact_field="fixture",
            principle="The test fixture is explicit.",
        )
    ]
    return build_terminal_artifact(
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:00:12+00:00",
        duration_s=12.0,
        phase_spans=[{"phase": "live", "duration_s": 10.0, "completed_units": 2}],
        preconditions=preconditions,
        source_hashes={},
        schedule=schedule,
        selection=selection,
        episode_rows=episode_rows,
        boundary_events=boundary_events,
        runtime_receipt={"child_terminal": True},
        model_specs=[
            {
                "hf_id": MODEL_ID,
                "quantization": QUANTIZATION,
                "model_filename": MODEL_FILENAME,
                "model_path": f"/cache/{MODEL_FILENAME}",
                "revision": "fixture-revision",
                "sha256": "sha256:" + "c" * 64,
                "flags": {"n_gpu_layers": 999, "use_chat_template": True},
                "chat_template": "embedded",
            }
        ],
        validation_receipts=validation_receipts,
        require_terminal=True,
    )


def independent_reduce(value: Mapping[str, Any]) -> JsonDict:
    """Recompute the panel, model counters, and request chains from raw rows."""

    schedule = value.get("schedule_rows") if isinstance(value.get("schedule_rows"), list) else []
    rows = value.get("per_game_results") if isinstance(value.get("per_game_results"), list) else []
    events = (
        value.get("current_invocation_events")
        if isinstance(value.get("current_invocation_events"), list)
        else []
    )
    panel = reduce_episode_panel(schedule, rows)
    current = reduce_current_invocations(events, child_terminal=True)
    requests = value.get("request_budget_rows")
    requests = requests if isinstance(requests, list) else []
    return {
        "arc_sentinel_capture_complete_score": panel["arc_sentinel_capture_complete_score"],
        "sample_size_budget": panel["sample_size_budget"],
        "invocation_counts": current["invocation_counts"],
        "model_invoked": current["model_invoked"],
        "inference_substrate_class": current["inference_substrate_class"],
        "invocation_errors": current["errors"],
        "request_budget_reduction": reduce_request_budget_rows(requests),
    }


def independent_reduce_file(path: Path) -> JsonDict:
    """Cold-load one candidate and compare declared independent reductions."""

    artifact = load_object(path)
    reduced = independent_reduce(artifact)
    return {
        **reduced,
        "matches_declared": (
            reduced["arc_sentinel_capture_complete_score"]
            == artifact.get("arc_sentinel_capture_complete_score")
            and reduced["sample_size_budget"] == artifact.get("sample_size_budget")
            and reduced["invocation_counts"] == artifact.get("invocation_counts")
            and reduced["model_invoked"] is artifact.get("model_invoked")
            and reduced["inference_substrate_class"] == artifact.get("inference_substrate_class")
            and reduced["request_budget_reduction"] == artifact.get("request_budget_reduction")
        ),
    }


def validate_artifact(
    value: Mapping[str, Any] | Path, *, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, current events, rows, scores, receipts, and checksum."""

    artifact = load_object(value) if isinstance(value, Path) else deepcopy(dict(value))
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if artifact.get("verdict_class") == "blocked":
        counts = artifact.get("invocation_counts")
        if (
            artifact.get("model_invoked") is not False
            or artifact.get("MODEL_SPECS") != []
            or counts != current_work_receipt.ZERO_INVOCATION_COUNTS
            or artifact.get("inference_substrate_class") != "no_model_load"
        ):
            errors.append("blocked_model_work_invalid")
        if (
            artifact.get("promotion_score") != 0
            or artifact.get("live_efficacy_score") != 0
            or artifact.get("arc_sentinel_capture_complete_score") != 0
        ):
            errors.append("nonpromotion_scores_invalid")
        if artifact.get("execution_venue") != EXECUTION_VENUE:
            errors.append("execution_venue_invalid")
        duration = artifact.get("duration_s")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            errors.append("duration_invalid")
        if not str(artifact.get("honest_verdict") or "").startswith("blocked_"):
            errors.append("verdict_invalid")
        if (
            artifact.get("solve_credit") != 0
            or artifact.get("hidden_leaderboard_claimed") is not False
        ):
            errors.append("solve_claim_invalid")
        summary = artifact.get("gate_check_summary")
        if not isinstance(summary, Mapping) or summary.get("first_failure") is None:
            errors.append("blocked_gate_summary_invalid")
        principles = artifact.get("field_principles")
        if not isinstance(principles, Mapping) or set(principles) != set(artifact):
            errors.append("field_principles_incomplete")
        if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
            errors.append("checksum_mismatch")
        return list(dict.fromkeys(errors))
    reduced = independent_reduce(artifact)
    if reduced["invocation_counts"] != artifact.get("invocation_counts"):
        errors.append("invocation_reduction_mismatch")
    if reduced["model_invoked"] is not artifact.get("model_invoked"):
        errors.append("model_invoked_mismatch")
    if reduced["inference_substrate_class"] != artifact.get("inference_substrate_class"):
        errors.append("substrate_class_mismatch")
    if reduced["invocation_errors"]:
        errors.append("invocation_events_invalid")
    expected_specs = MODEL_SPECS if reduced["model_invoked"] else []
    if artifact.get("MODEL_SPECS") != expected_specs:
        errors.append("MODEL_SPECS_invalid")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_invalid")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        errors.append("duration_invalid")
    elif (reduced["inference_substrate_class"] == "model_bounded_generation" and duration < 10) or (
        reduced["inference_substrate_class"] == "model_load_no_generation" and duration < 2
    ):
        errors.append("duration_floor_invalid")
    if reduced["sample_size_budget"] != artifact.get("sample_size_budget"):
        errors.append("sample_size_budget_mismatch")
    if reduced["arc_sentinel_capture_complete_score"] != artifact.get(
        "arc_sentinel_capture_complete_score"
    ):
        errors.append("capture_score_mismatch")
    if reduced["request_budget_reduction"] != artifact.get("request_budget_reduction"):
        errors.append("request_budget_reduction_mismatch")
    if artifact.get("promotion_score") != 0 or artifact.get("live_efficacy_score") != 0:
        errors.append("nonpromotion_scores_invalid")
    if artifact.get("solve_credit") != 0 or artifact.get("hidden_leaderboard_claimed") is not False:
        errors.append("solve_claim_invalid")
    if artifact.get("verdict_class") not in {"null", "blocked", "disqualified"}:
        errors.append("verdict_invalid")
    if artifact.get("verdict_class") == "null" and not str(
        artifact.get("honest_verdict") or ""
    ).startswith("complete_"):
        errors.append("verdict_invalid")
    required = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E)
    if require_terminal:
        required = (*required, *REQUIRED_TERMINAL)
    if artifact.get("verdict_class") != "blocked" and not _receipts_pass(
        artifact.get("validation_receipts") or [], required
    ):
        errors.append("validation_receipts_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7303 command set through the Exp7358 planner."""

    return validation_contract.build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject command expansion and private-path drift before subprocess work."""

    return validation_contract.validate_command_plan(root, VALIDATION_MANIFEST, commands)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Reuse the shipped ARC E2E-009, E2E-010, and private environment smoke."""

    return shipped_live.e2e_command_specs(root, private)


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh-process replay and the two strict artifact readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (
                python,
                "-u",
                str(root / WRAPPER_PATH),
                "--replay",
                str(candidate),
            ),
            "cold replay of exact measured candidate",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact measured candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "exact measured candidate",
        ),
    ]


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), sort_keys=True, default=str).encode() + b"\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[JsonDict]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return []
    rows: list[JsonDict] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


class DurableEpisodeRequestBudget(EpisodeRequestBudget):
    """Persist permit acquisition and terminal state around the shared budget."""

    def __init__(self, *args: Any, event_path: Path, **kwargs: Any) -> None:
        self.event_path = event_path
        super().__init__(*args, **kwargs)

    def reserve(self, *, branch: str, request_id: str | None = None) -> RequestReservation:
        reservation = super().reserve(branch=branch, request_id=request_id)
        row = next(
            item
            for item in self.receipt()["callback_rows"]
            if item["request_id"] == reservation.request_id
        )
        _append_jsonl(
            self.event_path,
            {
                "event": "permit_acquired",
                "episode_id": self.episode_id,
                "request_id": reservation.request_id,
                "reservation_index": row["reservation_index"],
                "monotonic": row["reserved_monotonic"],
            },
        )
        return reservation

    def _terminal(self, request_id: str, disposition: str) -> None:
        super()._terminal(request_id, disposition)
        _append_jsonl(
            self.event_path,
            {
                "event": "permit_terminal",
                "episode_id": self.episode_id,
                "request_id": request_id,
                "disposition": disposition,
                "monotonic": time.monotonic(),
            },
        )

    def _fail(self, request_id: str, error: BaseException | str) -> None:
        super()._fail(request_id, error)
        row = next(
            item for item in self.receipt()["callback_rows"] if item["request_id"] == request_id
        )
        _append_jsonl(
            self.event_path,
            {
                "event": "permit_terminal",
                "episode_id": self.episode_id,
                "request_id": request_id,
                "disposition": row["disposition"],
                "error": row["error"],
                "monotonic": time.monotonic(),
            },
        )

    def cancel(self, reason: str = "episode_cancelled") -> None:
        """Persist every permit that cancellation moves to a terminal state."""

        before = {
            str(row["request_id"]): str(row["disposition"])
            for row in self.receipt()["callback_rows"]
        }
        super().cancel(reason)
        for row in self.receipt()["callback_rows"]:
            request_id = str(row["request_id"])
            if before.get(request_id) == "in_flight" and row["disposition"] == "cancelled":
                _append_jsonl(
                    self.event_path,
                    {
                        "event": "permit_terminal",
                        "episode_id": self.episode_id,
                        "request_id": request_id,
                        "disposition": "cancelled",
                        "cancel_reason": row["cancel_reason"],
                        "monotonic": row["terminal_monotonic"],
                    },
                )


class DurableRequestCapture:
    """Persist request and response bytes around the real server call."""

    def __init__(
        self,
        root: Path,
        event_path: Path,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ValueError("max_new_tokens must be an integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.root = root
        self.event_path = event_path
        self.max_new_tokens = max_new_tokens
        self.original: Any = None
        self.episode_id = "bootstrap"
        self.indices: Counter[str] = Counter()

    def install(self) -> None:
        self.original = urllib.request.urlopen
        urllib.request.urlopen = self._open

    def restore(self) -> None:
        if self.original is not None:
            urllib.request.urlopen = self.original

    def begin_episode(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def _open(self, request: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(request, urllib.request.Request) or not str(request.full_url).endswith(
            ("/completion", "/v1/chat/completions", "/v1/completions")
        ):
            return self.original(request, *args, **kwargs)
        call_index = self.indices[self.episode_id]
        self.indices[self.episode_id] += 1
        body = bytes(request.data or b"")
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}
        requested = int(payload.get("max_tokens") or payload.get("n_predict") or 0)
        if requested > self.max_new_tokens:
            raise RuntimeError(
                f"requested token budget exceeded: {requested}>{self.max_new_tokens}"
            )
        directory = self.root / self.episode_id.replace(":", "__") / "requests"
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / f"{call_index:02d}_request.json"
        response_path = directory / f"{call_index:02d}_response.json"
        request_path.write_bytes(body)
        started = time.monotonic()
        request_sha = "sha256:" + hashlib.sha256(body).hexdigest()
        base = {
            "episode_id": self.episode_id,
            "call_index": call_index,
            "request_dispatched": True,
            "request_started_monotonic": started,
            "request_sha256": request_sha,
            "request_path": str(request_path),
            "requested_max_tokens": requested,
            "url": str(request.full_url),
        }
        _append_jsonl(self.event_path, {"event": "server_request", **base})
        try:
            response = self.original(request, *args, **kwargs)
            response_bytes = response.read()
            response.close()
            response_path.write_bytes(response_bytes)
            response_sha = "sha256:" + hashlib.sha256(response_bytes).hexdigest()
            _append_jsonl(
                self.event_path,
                {
                    "event": "server_response",
                    **base,
                    "response_observed": True,
                    "response_sha256": response_sha,
                    "response_path": str(response_path),
                    "elapsed_s": time.monotonic() - started,
                },
            )
            return io.BytesIO(response_bytes)
        except BaseException as exc:
            _append_jsonl(
                self.event_path,
                {
                    "event": "server_error",
                    **base,
                    "response_observed": False,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                    "elapsed_s": time.monotonic() - started,
                },
            )
            raise


class EpisodeTimeout(Exception):
    """Stop one policy episode at its fixed wall-time boundary."""


def _level(frame: Any) -> int:
    value = getattr(frame, "frame", frame)
    levels = getattr(value, "levels", None)
    return int(levels[-1]) if isinstance(levels, list) and levels else 0


def _frame_hash(frame: Any) -> str:
    value = getattr(frame, "frame", frame)
    public = getattr(value, "frame", value)
    return canonical_hash(public)


def _transport_rows(events: Sequence[Mapping[str, Any]], episode_id: str) -> list[JsonDict]:
    rows: dict[int, JsonDict] = {}
    for event in events:
        if event.get("episode_id") != episode_id or event.get("event") not in {
            "server_request",
            "server_response",
            "server_error",
        }:
            continue
        index = int(event.get("call_index") or 0)
        rows.setdefault(index, {}).update(deepcopy(dict(event)))
    return [rows[index] for index in sorted(rows)]


def _run_policy_episode(  # pragma: no cover - real ARC environment and policy integration.
    schedule: Mapping[str, Any], proposer: Any, capture: DurableRequestCapture, event_path: Path
) -> JsonDict:
    """Run the submitted adapter on a real public environment through Agent methods."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import make_carnot_agent
    from carnot.agentic import arc_executable_world_model as e3

    episode_id = str(schedule["episode_id"])
    game = str(schedule["game"])
    episode_dir = event_path.parent / episode_id.replace(":", "__")
    episode_dir.mkdir(parents=True, exist_ok=True)
    old_e3_dir = e3.E3_DIR
    e3.E3_DIR = episode_dir / "fresh_e3"
    capture.begin_episode(episode_id)
    budget = DurableEpisodeRequestBudget(
        episode_id,
        limit=REQUEST_LIMIT,
        deadline_s=EPISODE_LIMIT_S,
        event_path=event_path,
    )
    attach_request_budget(proposer, budget)

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
    agent = agent_type(game_id=game)
    policy = agent._policy
    entered = time.monotonic()
    policy_entry = {
        "factory": "make_carnot_agent",
        "policy_class": type(policy).__name__,
        "choose_action_path": True,
        "is_done_path": True,
        "adapter_disabled": True,
        "denied_paths": list(WITHHELD_INPUTS),
    }
    _append_jsonl(event_path, {"event": "policy_entry", "episode_id": episode_id, **policy_entry})
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest: Any = None
    actions = 0
    first_action: JsonDict | None = None
    first_observation: JsonDict | None = None
    start_level: int | None = None
    max_level = 0
    disposition = "complete"
    error: str | None = None

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout(f"episode exceeded {EPISODE_LIMIT_S}s")

    previous = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EPISODE_LIMIT_S)
    try:
        for action_index in range(ACTION_LIMIT):
            if agent.is_done(frames, latest):
                break
            action = agent.choose_action(frames, latest)
            name = str(getattr(action, "name", action))
            data_value = getattr(action, "action_data", None)
            data = data_value.model_dump() if hasattr(data_value, "model_dump") else None
            if isinstance(data, Mapping):
                data = {key: value for key, value in data.items() if key != "game_id"}
            if name == "RESET":
                latest = env.reset()
            else:
                if first_action is None:
                    first_action = {
                        "action": name,
                        "data": deepcopy(data),
                        "action_index": action_index,
                        "elapsed_s": time.monotonic() - entered,
                    }
                    _append_jsonl(
                        event_path,
                        {"event": "first_action", "episode_id": episode_id, **first_action},
                    )
                latest = env.step(action if isinstance(action, GameAction) else action, data=data)
                actions += 1
            level = _level(latest)
            if start_level is None:
                start_level = level
                max_level = level
            max_level = max(max_level, level)
            if first_action is not None and first_observation is None:
                first_observation = {
                    "level": level,
                    "frame_sha256": _frame_hash(latest),
                    "elapsed_s": time.monotonic() - entered,
                }
                _append_jsonl(
                    event_path,
                    {
                        "event": "first_observation",
                        "episode_id": episode_id,
                        **first_observation,
                    },
                )
            frames.append(latest)
            _append_jsonl(
                event_path,
                {
                    "event": "action_observation",
                    "episode_id": episode_id,
                    "action_index": action_index,
                    "action": name,
                    "level": level,
                    "elapsed_s": time.monotonic() - entered,
                },
            )
    except EpisodeTimeout as exc:
        disposition = "censored_timeout" if first_action is not None else "censored_no_first_action"
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        disposition = "complete_error" if first_action is not None else "censored_no_first_action"
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        budget.cancel("episode_terminal")
        e3.E3_DIR = old_e3_dir
    supervisor = (
        policy.trajectory_supervisor_diagnostics()
        if hasattr(policy, "trajectory_supervisor_diagnostics")
        else {}
    )
    redirects = list(supervisor.get("redirects") or supervisor.get("would_have_redirects") or [])
    attempts = [
        dict(row) for row in getattr(policy, "induction_attempts", []) if isinstance(row, Mapping)
    ]
    tool_feedback = sum(
        int(bool((row.get("tool_loop") or {}).get("tool_gap_events"))) for row in attempts
    )
    request_events = _read_jsonl(event_path)
    row = {
        **deepcopy(dict(schedule)),
        "disposition": disposition,
        "policy_entry": policy_entry,
        "first_action": first_action,
        "first_action_latency_s": first_action.get("elapsed_s") if first_action else None,
        "first_observation": first_observation,
        "actions": actions,
        "start_level": start_level,
        "max_level": max_level if start_level is not None else None,
        "level_progress": max(0, max_level - start_level) if start_level is not None else 0,
        "tool_feedback_consumed": tool_feedback,
        "supervisor": {
            "fired": len(redirects),
            "consumed": sum(row.get("applied") is True for row in redirects),
            "mode": supervisor.get("mode"),
            "arms_enabled": deepcopy(supervisor.get("arms_enabled") or []),
        },
        "elapsed_s": time.monotonic() - entered,
        "request_budget_receipt": budget.receipt(),
        "server_request_rows": _transport_rows(request_events, episode_id),
        "induction_attempt_count": len(attempts),
        "solve_credit": 0,
        "error": error,
    }
    _append_jsonl(
        event_path,
        {
            "event": "episode_terminal",
            "episode_id": episode_id,
            "disposition": disposition,
            "actions": actions,
            "elapsed_s": row["elapsed_s"],
        },
    )
    return row


def session_environment(
    base: Mapping[str, str], *, gpu_index: int, port: int, raw_dir: Path
) -> dict[str, str]:
    """Build the adapter-withheld native CUDA child environment."""

    env = dict(base)
    for key in (
        "CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED",
        "CARNOT_ARC_PLAYBOOK_RETRIEVAL",
        "CARNOT_ARC_RUN_LOCAL_ADAPTATION",
        "CARNOT_ARC_CROSS_LEVEL_ENGINE_CARRY",
        "CARNOT_ARC_SUPPLY_WIN_TRANSITION",
        "CARNOT_ARC_STRUCTURED_NAV",
        "CARNOT_ARC_SGE_CANDIDATE_ROUTER",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{REPO_ROOT / 'python'}:{REPO_ROOT}",
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_LLM_BACKEND": "llamacpp",
            "CARNOT_ARC_GGUF_PATH": env.get("CARNOT_ARC_GGUF_PATH", ""),
            "CARNOT_ARC_PROPOSER_PORT": str(port),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(MAX_NEW_TOKENS),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(int(EPISODE_LIMIT_S)),
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(REQUEST_LIMIT),
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(REQUEST_LIMIT),
            "CARNOT_ARC_LLAMA_SERVER_PARALLEL": "1",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu_index),
            "CARNOT_ARC_RANDOM_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
            "CARNOT_ARC_MTP": "0",
            "CARNOT_ARC_ACTION_PROVENANCE": "1",
            "CARNOT_ARC_ACTION_PROVENANCE_DIR": str(raw_dir / "action_provenance"),
            "CARNOT_ARC_SERVER_LOG_DIR": str(raw_dir / "server_logs"),
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            BOUNDARY_LEDGER_ENV: str(REPO_ROOT / BOUNDARY_PATH),
        }
    )
    return env


def _absolute_model_path(path: str | Path) -> str:
    """Keep the snapshot symlink path so the shipped revision reader can authenticate it."""

    return str(Path(path).expanduser().absolute())


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live child.
    """Load one owned Qwen server and run the two real Agent episodes."""

    started = time.monotonic()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    schedule = load_object(Path(args.schedule_path)).get("rows") or []
    random.seed(RANDOM_SEED)
    capture = DurableRequestCapture(raw_dir, event_path)
    proposer: Any = None
    rows: list[JsonDict] = []
    session: JsonDict = {
        "child_pid": os.getpid(),
        "model_loaded": False,
        "model_invoked": False,
        "episodes": rows,
        "runtime_receipt": {},
        "error": None,
    }
    try:
        capture.install()
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        progress(started, "model_load", "before", model_path=args.model_path)
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.8-27B",
            model_path=_absolute_model_path(args.model_path),
            port=int(args.port),
            mtp=False,
            kv_quant="q8_0",
            use_chat_template=True,
            n_gpu_layers=999,
            n_ctx=49_152,
            max_tokens=MAX_NEW_TOKENS,
            timeout=int(EPISODE_LIMIT_S),
            tries=1,
        )
        proposer.model_repository = MODEL_ID
        proposer.model_revision = str(args.model_revision)
        proposer.requested_model_filename = MODEL_FILENAME
        proposer.requested_model_path = _absolute_model_path(args.model_path)
        if not proposer._ensure_server():
            raise RuntimeError("owned native CUDA llama-server failed to start")
        session["model_loaded"] = True
        progress(started, "model_load", "after", server_pid=getattr(proposer._proc, "pid", None))
        session["runtime_receipt"] = {
            "child_pid": os.getpid(),
            "server_pid": getattr(proposer._proc, "pid", None),
            "native_binary": proposer.last_launch_argv[0] if proposer.last_launch_argv else None,
            "server_command": list(proposer.last_launch_argv),
            "n_gpu_layers": 999,
            "n_ctx": 49_152,
            "kv_quantization": "q8_0",
            "use_chat_template": True,
            "chat_template": "embedded_model_template",
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": REQUEST_LIMIT,
        }
        atomic_json(
            Path(args.checkpoint_path),
            {"stage": "model_loaded", "model_loaded": True, "completed_units": 0},
        )
        for index, schedule_row in enumerate(schedule):
            if index > 0 and (not rows or not rows[0].get("first_action")):
                break
            progress(
                started,
                "episode",
                "before_benchmark",
                episode_id=schedule_row["episode_id"],
                completed_units=index,
            )
            row = _run_policy_episode(schedule_row, proposer, capture, event_path)
            rows.append(row)
            atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            atomic_json(
                Path(args.checkpoint_path),
                {
                    "stage": "episodes",
                    "model_loaded": True,
                    "completed_units": len(rows),
                    "total_units": len(schedule),
                },
            )
            progress(
                started,
                "episode",
                "after_benchmark",
                episode_id=schedule_row["episode_id"],
                completed_units=len(rows),
                disposition=row["disposition"],
            )
        session["model_invoked"] = bool(
            InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
        )
    except BaseException as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"[:500]
        progress(started, "live_child", "error", error=session["error"])
    finally:
        capture.restore()
        progress(started, "model_unload", "before")
        if proposer is not None:
            proposer.stop()
        progress(started, "model_unload", "after")
        session["duration_s"] = time.monotonic() - started
        atomic_json(Path(args.session_path), session)
        atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "child_terminal",
                "model_loaded": session["model_loaded"],
                "completed_units": len(rows),
                "terminal_child": True,
            },
        )
    return 0


def _free_port() -> int:  # pragma: no cover - operating-system socket allocation.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _observe_loaded_lease(lease: Any, checkpoint: Mapping[str, Any], resident: bool) -> bool:
    """Advance the owned lease when the child durably reports model residency."""

    if checkpoint.get("model_loaded") is True and not resident:
        lease.transition("resident", vram_mb=0)
        lease.transition("inferencing")
        return True
    return resident


def run_child_with_lease(
    *, resources: Mapping[str, Any], schedule_path: Path, started: float
) -> JsonDict:  # pragma: no cover - owned process boundary.
    """Acquire one fresh GPU lease and supervise only the owned child group."""

    from carnot.gpu_lease_phase_journal import GpuLease

    gpu = dict(resources["gpu"])
    lease = GpuLease.acquire(
        runtime_dir=REPO_ROOT / RAW_DIR / "gpu_lease",
        task_id=TASK_ID,
        device_uuid=str(gpu.get("uuid")),
        expected_model=str(resources["model_path"]),
        vram_before_mb=int(gpu.get("total_memory_mb") or 0) - int(gpu.get("free_memory_mb") or 0),
        ttl_s=90,
    )
    lease.transition("admitted")
    lease.transition("loading")
    port = _free_port()
    model_spec = dict(resources.get("model_spec") or {})
    revision = str(model_spec.get("revision") or model_spec.get("model_revision") or "unknown")
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / WRAPPER_PATH),
        "--role",
        "live-session",
        "--date",
        RUN_DATE,
        "--model-path",
        str(resources["model_path"]),
        "--model-hash",
        str(resources["model_hash"]),
        "--model-revision",
        revision,
        "--gpu-index",
        str(gpu["index"]),
        "--port",
        str(port),
        "--schedule-path",
        str(schedule_path),
        "--raw-dir",
        str(REPO_ROOT / RAW_DIR),
        "--checkpoint-path",
        str(REPO_ROOT / CHECKPOINT_PATH),
        "--session-path",
        str(REPO_ROOT / SESSION_PATH),
    ]
    env = session_environment(
        os.environ, gpu_index=int(gpu["index"]), port=port, raw_dir=REPO_ROOT / RAW_DIR
    )
    env["CARNOT_ARC_GGUF_PATH"] = str(resources["model_path"])
    env["CARNOT_LLAMA_SERVER"] = str(resources["server"])
    progress(started, "live_subprocess", "before", command=" ".join(command))
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
    child_started = time.monotonic()
    next_heartbeat = child_started
    timed_out = False
    resident = False
    while process.poll() is None:
        now = time.monotonic()
        checkpoint = load_object(REPO_ROOT / CHECKPOINT_PATH)
        resident = _observe_loaded_lease(lease, checkpoint, resident)
        if now - child_started >= AGGREGATE_LIVE_LIMIT_S:
            timed_out = True
            break
        if now >= next_heartbeat:
            lease.heartbeat()
            progress(
                started,
                "live_subprocess",
                "heartbeat",
                completed_units=int(checkpoint.get("completed_units") or 0),
                model_loaded=bool(checkpoint.get("model_loaded")),
                pending_operation=checkpoint.get("stage", "child_startup"),
            )
            next_heartbeat = now + 45.0
        time.sleep(0.5)
    signals_sent: list[str] = []
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        signals_sent.append("SIGTERM:owned_process_group")
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            signals_sent.append("SIGKILL:owned_process_group")
            process.wait(timeout=10)
    progress(
        started,
        "live_subprocess",
        "after",
        returncode=process.returncode,
        timed_out=timed_out,
    )
    session = load_object(REPO_ROOT / SESSION_PATH)
    if not session:
        session = {
            "child_pid": process.pid,
            "model_loaded": False,
            "model_invoked": bool(_read_jsonl(REPO_ROOT / BOUNDARY_PATH)),
            "episodes": list(
                load_object(REPO_ROOT / RAW_DIR / "episode_rows.json").get("rows") or []
            ),
            "error": "live_child_did_not_write_session",
        }
    resident = _observe_loaded_lease(lease, session, resident)
    if session.get("model_loaded"):
        lease.transition("unloading")
        lease.transition(
            "validating", vram_mb=0, exit_code=int(process.returncode or 0), unload_observed=True
        )
        lease.transition("terminal_complete")
    else:
        lease.transition("terminal_blocked")
    release = lease.release()
    runtime = dict(session.get("runtime_receipt") or {})
    runtime.update(
        {
            "gpu_uuid": gpu.get("uuid"),
            "gpu_index": gpu.get("index"),
            "gpu_name": gpu.get("name"),
            "lease_owner": lease.owner_receipt(),
            "lease_release": release,
            "fresh_lease": True,
            "signals_sent": signals_sent,
            "timed_out": timed_out,
            "child_returncode": process.returncode,
            "child_terminal": True,
        }
    )
    session["runtime_receipt"] = runtime
    session["timed_out"] = timed_out
    atomic_json(REPO_ROOT / SESSION_PATH, session)
    return session


def _phase(
    spans: list[JsonDict],
    name: str,
    phase_start: float,
    run_start: float,
    units: int,
    checkpoint: str | None = None,
) -> None:
    now = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": round(phase_start - run_start, 6),
            "end_s": round(now - run_start, 6),
            "duration_s": round(now - phase_start, 6),
            "completed_units": units,
            "checkpoint": checkpoint,
            "ended_at_utc": utc_now(),
        }
    )


def _runtime_preconditions(  # pragma: no cover - native CUDA and model-cache integration.
    root: Path, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Reuse shipped Qwen, tokenizer, native CUDA, and idle-device readers."""

    checks, hashes, resources = shipped_live.runtime_preconditions(
        root, gpu_wait_s=120.0, started=started
    )
    normalized = [
        {
            **dict(row),
            "category": row.get("category", "precondition"),
            "operator": row.get("operator", "=="),
            "principle": row.get(
                "principle", "The shipped reader authenticates this runtime resource."
            ),
        }
        for row in checks
    ]
    from carnot.agentic.arc_eval_provenance import huggingface_snapshot_revision

    spec = dict(resources.get("model_spec") or {})
    revision = huggingface_snapshot_revision(str(resources.get("model_path") or ""), MODEL_ID)
    spec.update(
        {
            "hf_id": MODEL_ID,
            "quantization": QUANTIZATION,
            "model_filename": MODEL_FILENAME,
            "revision": revision or spec.get("revision") or spec.get("model_revision"),
            "sha256": resources.get("model_hash"),
            "flags": {
                "n_gpu_layers": 999,
                "n_ctx": 49_152,
                "kv_quantization": "q8_0",
                "parallel": 1,
                "mtp": False,
                "offline": True,
                "use_chat_template": True,
            },
            "chat_template": "embedded_model_template",
            "decoding": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "request_limit_per_episode": REQUEST_LIMIT,
                "retry_budget": 0,
                "alternative_generator": False,
            },
        }
    )
    resources["model_spec"] = spec
    return normalized, hashes, resources


def _blocked_artifact(
    *,
    started_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]] = (),
    selection: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Publish external absence as blocked without fake model work."""

    rows = [_unstarted_row(row) for row in schedule]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "blocked_external_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "current_invocation_call_rows": [],
        "inference_substrate": "no_model_load",
        "inference_substrate_details": {},
        "inference_substrate_class": "no_model_load",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": round(max(duration_s, 0.000001), 6),
        "duration_breakdown_s": _duration_breakdown(spans),
        "phase_spans": [deepcopy(dict(row)) for row in spans],
        "random_seed": {
            "selection": RANDOM_SEED,
            "episodes": [RANDOM_SEED, RANDOM_SEED],
            "sampling": RANDOM_SEED,
            "resampling": None,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(hashes)),
        "schedule_rows": [deepcopy(dict(row)) for row in schedule],
        "selection_receipt": deepcopy(dict(selection or {})),
        "rows": rows,
        "sample_size_budget": {
            "planned_units": 2,
            "attempted_units": 0,
            "completed_units": 0,
            "failed_units": 0,
            "censored_units": 0,
            "unstarted_units": 2,
            "independent_groups": ["public_adapter_withheld_development_proxy"],
            "stopping_rule": "External precondition failure stops before model work.",
            "request_limit_per_episode": REQUEST_LIMIT,
            "action_limit_per_episode": ACTION_LIMIT,
            "episode_limit_s": EPISODE_LIMIT_S,
            "aggregate_live_limit_s": AGGREGATE_LIVE_LIMIT_S,
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "required_check_names": [],
        "repository_health": {
            "status": "scoped_only",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": {},
        "promotion_score": 0,
        "arc_sentinel_capture_complete_score": 0,
        "per_game_results": rows,
        "solve_provenance": "uncredited_no_runtime",
        "solve_credit": 0,
        "new_level_credit": 0,
        "new_level_reproduction_required": False,
        "public_development_generalization_proxy": True,
        "hidden_leaderboard_claimed": False,
        "request_budget_rows": [],
        "request_budget_reduction": reduce_request_budget_rows([]),
        "supervisor_outcomes": [],
        "live_efficacy_score": 0,
        "paired_control_present": False,
        "treatment_effect_claimed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "supervisor_arm_added": False,
        "supervisor_fitted": False,
        "solve_registry_changed": False,
        "research_conductor_changed": False,
        "small_ebm_training": {"performed": False},
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def finalize_existing_candidate(
    root: Path,
) -> JsonDict:  # pragma: no cover - recovery orchestration.
    """Revalidate a preserved measured candidate after a producer-only repair."""

    started = time.monotonic()
    candidate_path = root / TERMINAL_CANDIDATE_PATH
    candidate = load_object(candidate_path)
    errors = validate_artifact(candidate, require_terminal=False)
    if errors:
        raise RuntimeError(f"preserved measured candidate invalid: {errors}")
    progress(started, "finalize", "before_affected_validation")
    private = Path(tempfile.mkdtemp(prefix="exp7431-finalize-", dir="/tmp"))
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation command plan drift: {plan_errors}")
    affected = validation_contract.run_categorized_commands(
        root,
        [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
        log_dir=root / RAW_DIR / "validation/finalize_affected",
        heartbeat_s=60.0,
    )
    prior_receipts = [
        deepcopy(dict(row))
        for row in candidate.get("validation_receipts", [])
        if row.get("name") not in {*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_TERMINAL}
    ]
    receipts = [*affected, *prior_receipts]
    static_checks, current_hashes, _ = collect_preconditions(root)
    if not all(row.get("passed") is True for row in static_checks):
        raise RuntimeError("current static preconditions failed during candidate finalization")
    source_hashes = deepcopy(dict(candidate.get("source_artifact_hashes") or {}))
    source_hashes.update(current_hashes)
    spans = [deepcopy(dict(row)) for row in candidate.get("phase_spans", [])]
    _phase(spans, "producer_repair_validation", started, started, len(affected))
    duration_s = float(candidate.get("duration_s") or 0.0) + (time.monotonic() - started)
    upgraded = build_terminal_artifact(
        started_at_utc=str(candidate["started_at_utc"]),
        ended_at_utc=utc_now(),
        duration_s=duration_s,
        phase_spans=spans,
        preconditions=candidate["preconditions_checked"],
        source_hashes=source_hashes,
        schedule=candidate["schedule_rows"],
        selection=candidate["selection_receipt"],
        episode_rows=candidate["per_game_results"],
        boundary_events=candidate["current_invocation_events"],
        runtime_receipt=candidate["inference_substrate_details"],
        model_specs=candidate["model_specs"],
        validation_receipts=receipts,
        require_terminal=False,
    )
    upgraded_errors = validate_artifact(upgraded, require_terminal=False)
    if upgraded_errors:
        raise RuntimeError(f"upgraded measured candidate invalid: {upgraded_errors}")
    atomic_json(candidate_path, upgraded)
    terminal_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", candidate=candidate_path)
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, candidate_path),
        log_dir=root / RAW_DIR / "validation/finalize_terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    _phase(spans, "terminal_cold_validation", terminal_started, started, len(terminal))
    artifact = build_terminal_artifact(
        started_at_utc=str(candidate["started_at_utc"]),
        ended_at_utc=utc_now(),
        duration_s=float(candidate.get("duration_s") or 0.0) + (time.monotonic() - started),
        phase_spans=spans,
        preconditions=candidate["preconditions_checked"],
        source_hashes=source_hashes,
        schedule=candidate["schedule_rows"],
        selection=candidate["selection_receipt"],
        episode_rows=candidate["per_game_results"],
        boundary_events=candidate["current_invocation_events"],
        runtime_receipt=candidate["inference_substrate_details"],
        model_specs=candidate["model_specs"],
        validation_receipts=receipts,
        require_terminal=True,
    )
    final_errors = validate_artifact(artifact, require_terminal=True)
    if final_errors:
        raise RuntimeError(f"terminal artifact invalid: {final_errors}")
    atomic_json(root / RESULT_PATH, artifact)
    progress(started, "finalize", "terminal_published", path=RESULT_PATH)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Run prechecks, scoped validation, live work, cold readers, and publish."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before_static")
    checks, hashes, registry = collect_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date",
            "precondition",
            RUN_DATE,
            run_date,
            upstream="command_line",
            artifact_field="--date",
            principle="The protocol uses the declared execution date.",
        ),
    )
    _phase(spans, "preconditions_static", phase_start, started, len(checks))
    progress(started, "preconditions", "after_static", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=[],
        )
        atomic_json(root / RESULT_PATH, artifact)
        progress(started, "write", "terminal_blocked", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    progress(started, "selection", "before_registry_precheck")
    selection = freeze_two_game_rotation(registry)
    schedule = build_schedule(selection["games"])
    atomic_json(root / SCHEDULE_PATH, {"selection_receipt": selection, "rows": schedule})
    checks.append(
        gate_row(
            "label_blind_rotation",
            "precondition",
            True,
            selection["passed"] and len(schedule) == 2,
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="two_selected_games",
            principle="The registry is used only for label-blind selection and duplicate-credit precheck.",
        )
    )
    _phase(spans, "selection", phase_start, started, len(schedule), SCHEDULE_PATH.as_posix())
    progress(started, "selection", "after_registry_precheck", games=selection["games"])
    if not selection["passed"] or len(schedule) != 2:
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=[],
            schedule=schedule,
            selection=selection,
        )
        atomic_json(root / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7431-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses")
    plan = build_validation_plan(root, private / "scoped")
    plan_errors = validate_validation_plan(root, plan)
    if plan_errors:
        raise RuntimeError(f"validation command plan drift: {plan_errors}")
    receipts.extend(
        validation_contract.run_categorized_commands(
            root,
            [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
            log_dir=root / RAW_DIR / "validation/affected",
            heartbeat_s=60.0,
        )
    )
    progress(started, "validation", "after_affected_subprocesses", completed_units=len(receipts))
    progress(started, "e2e", "before_subprocesses")
    e2e = validation_scope.run_commands(
        root,
        e2e_command_specs(root, private / "e2e"),
        log_dir=root / RAW_DIR / "validation/e2e",
        heartbeat_s=60.0,
    )
    receipts.extend(e2e)
    progress(started, "e2e", "after_subprocesses", completed_units=len(e2e))
    _phase(spans, "affected_validation_and_e2e", phase_start, started, len(receipts))
    if not _receipts_pass(receipts, (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E)):
        artifact = build_terminal_artifact(
            started_at_utc=started_at,
            ended_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
            phase_spans=spans,
            preconditions=checks,
            source_hashes=hashes,
            schedule=schedule,
            selection=selection,
            episode_rows=[],
            boundary_events=[],
            runtime_receipt={},
            model_specs=[],
            validation_receipts=receipts,
            require_terminal=False,
        )
        atomic_json(root / RESULT_PATH, artifact)
        return artifact

    phase_start = time.monotonic()
    progress(started, "runtime_preconditions", "before_model_cuda_lease_checks")
    runtime_checks, runtime_hashes, resources = _runtime_preconditions(root, started)
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    _phase(spans, "runtime_preconditions", phase_start, started, len(runtime_checks))
    progress(
        started,
        "runtime_preconditions",
        "after_model_cuda_lease_checks",
        passed=all(row.get("passed") is True for row in checks),
    )
    if not all(row.get("passed") is True for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=receipts,
            schedule=schedule,
            selection=selection,
        )
        atomic_json(root / RESULT_PATH, artifact)
        progress(started, "write", "terminal_blocked", path=RESULT_PATH)
        return artifact

    for path in (
        root / BOUNDARY_PATH,
        root / RUNTIME_EVENT_PATH,
        root / SESSION_PATH,
        root / CHECKPOINT_PATH,
        root / TERMINAL_CANDIDATE_PATH,
        root / RAW_DIR / "episode_rows.json",
    ):
        path.unlink(missing_ok=True)
    phase_start = time.monotonic()
    progress(started, "live", "before_model_load_generation_benchmark", planned_units=2)
    session = run_child_with_lease(
        resources=resources, schedule_path=root / SCHEDULE_PATH, started=started
    )
    progress(
        started,
        "live",
        "after_model_load_generation_benchmark",
        completed_units=len(session.get("episodes") or []),
    )
    _phase(
        spans,
        "live_model_and_episodes",
        phase_start,
        started,
        len(session.get("episodes") or []),
        SESSION_PATH.as_posix(),
    )

    phase_start = time.monotonic()
    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    episodes = [dict(row) for row in session.get("episodes", []) if isinstance(row, Mapping)]
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        schedule=schedule,
        selection=selection,
        episode_rows=episodes,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"measured candidate invalid: {candidate_errors}")
    atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)
    _phase(
        spans,
        "independent_reduction",
        phase_start,
        started,
        len(candidate["rows"]),
        TERMINAL_CANDIDATE_PATH.as_posix(),
    )

    phase_start = time.monotonic()
    progress(
        started, "terminal_validation", "before_subprocesses", candidate=TERMINAL_CANDIDATE_PATH
    )
    terminal = validation_scope.run_commands(
        root,
        terminal_command_specs(root, root / TERMINAL_CANDIDATE_PATH),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))
    _phase(spans, "terminal_cold_validation", phase_start, started, len(terminal))

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        schedule=schedule,
        selection=selection,
        episode_rows=episodes,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=True,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal artifact invalid: {errors}")
    progress(started, "write", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "write",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        capture=artifact["arc_sentinel_capture_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed public command and private child/replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    args = parser.parse_args(argv)
    if args.replay is None and args.date is None:
        parser.error("--date is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Run the host experiment, private live child, or cold replay."""

    args = parse_args(argv)
    if args.replay is not None:
        reduced = independent_reduce_file(args.replay)
        errors = validate_artifact(args.replay, require_terminal=False)
        print(
            json.dumps({"reduced": reduced, "validation_errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    if args.finalize_existing:
        artifact = finalize_existing_candidate(REPO_ROOT)
        return 0 if artifact.get("verdict_class") == "null" else 1
    if args.role == "live-session":
        return run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "ACTION_LIMIT",
    "AGGREGATE_LIVE_LIMIT_S",
    "EPISODE_LIMIT_S",
    "EXECUTION_VENUE",
    "INFERENCE_SUBSTRATE_CLASS",
    "MAX_NEW_TOKENS",
    "MODEL_FILENAME",
    "MODEL_ID",
    "MODEL_SPECS",
    "REPO_ROOT",
    "REQUEST_LIMIT",
    "REQUIRED_E2E",
    "REQUIRED_TERMINAL",
    "SPEC_PATH",
    "VALIDATION_MANIFEST",
    "artifact_checksum",
    "atomic_json",
    "build_artifact_for_test",
    "build_schedule",
    "build_validation_plan",
    "independent_reduce_file",
    "main",
    "parse_args",
    "reduce_current_invocations",
    "reduce_episode_panel",
    "reduce_request_budget_rows",
    "validate_artifact",
    "validate_validation_plan",
    "validation_scope",
]
