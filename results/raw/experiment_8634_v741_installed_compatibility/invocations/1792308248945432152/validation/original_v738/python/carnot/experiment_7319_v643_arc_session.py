"""Measure one authority-backed live ARC selfparse session.

The module reuses the shipped live policy, native llama.cpp runner, GPU lease,
request capture, and scoped validator. It adds the new authority dependency,
target rotation, causal tool-use receipt, and cumulative induction accounting.

Spec refs: REQ-ARC-WMTE-7319 and SCENARIO-ARC-WMTE-7319-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7305_v642_arc_selfparse as prior
from carnot import experiment_7318_v643_arc_authority as authority
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260915"
MILESTONE = "2026.09.643"
EXPERIMENT_ID = "exp7319-arc-session"
SCHEMA = "carnot.experiment_7319.v643.arc_session.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
TARGET_ROTATION = ("r11l", "re86")
TARGET_GAME = "re86"
DEVELOPMENT_SEED = 7_319_202_609_15
EVALUATION_SEED = 17_319_202_609_15
ACTION_LIMIT = 192
COMPLETION_LIMIT = 2
GENERATED_TOKEN_LIMIT = 4096
TOKENS_PER_CALL = GENERATED_TOKEN_LIMIT // COMPLETION_LIMIT
SESSION_LIMIT_S = 3000
MODEL_LOAD_LIMIT_S = 600

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP7318_PATH = Path("results/experiment_7318_v643_arc_authority.json")
EXP7305_PATH = Path("results/experiment_7305_v642_arc_selfparse.json")
HISTORICAL_LEDGER_PATH = Path(
    "results/raw/experiment_7305_v642_arc_selfparse/cumulative_inductions.json"
)
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RESULT_PATH = Path("results/experiment_7319_v643_arc_session.json")
RAW_DIR = Path("results/raw/experiment_7319_v643_arc_session")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7319_v643_arc_session.json")
BOUNDARY_PATH = RAW_DIR / "receipt_events.jsonl"
TOOL_EVENT_PATH = RAW_DIR / "tool_events.jsonl"
SESSION_PATH = RAW_DIR / "live_session.json"
RAW_ROW_PATH = RAW_DIR / "independent_reduction_input.json"
CUMULATIVE_PATH = RAW_DIR / "cumulative_inductions.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7319_v643_arc_session.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7319_v643_arc_session.py")
TEST_PATH = Path("tests/python/test_experiment_7319_v643_arc_session.py")
HISTORICAL_REPOSITORY_HEALTH_LOG = Path(
    "results/raw/experiment_7289_v641_arc_boundary/validation/04_full_python_suite.log"
)

REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
ZERO_INVOCATION_COUNTS = deepcopy(prior.ZERO_INVOCATION_COUNTS)

FIELD_PRINCIPLES = {
    "schema": "Version this artifact; keep ordinary top-level experiment_id and milestone.",
    "status": "Write terminal output only after current work and required validation.",
    "experiment_id": "Identify this bounded session with one stable ordinary value.",
    "milestone": "Bind this result to milestone 2026.09.643.",
    "run_date": "Use 20260915; preserve actual UTC timestamps and monotonic phase spans.",
    "started_at_utc": "Record when the current invocation began.",
    "ended_at_utc": "Record when terminal construction ended.",
    "preconditions_checked": "Record input identities, availability, and the exact failed check.",
    "MODEL_SPECS": "Current executable identities only; every LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any actual attempted load or generation, including unusable results.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation using the recognized substrate literal.",
    "inference_substrate_class": "Use the recognized class that matches the actual model work.",
    "inference_mode": "Use live_gpu only when task-owned CUDA execution overlaps current work.",
    "execution_venue": "Use host for this milestone; historical board work is not current board execution.",
    "execution_host": "Record the host separately from the venue.",
    "duration_s": "Measure real elapsed time; never sleep or inflate counts to satisfy a floor.",
    "phase_spans": "Record disjoint elapsed spans, units, checkpoint boundaries, and pending operations.",
    "random_seed": "Seal independent development and evaluation seeds before observing results.",
    "reproducibility_checksum": "Bind code, public inputs, private evaluator identity, settings, and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers; historical diagnostic evidence cannot authorize readiness.",
    "rows": "Emit every comparative unit and arm with metrics, costs, failures, abstentions, and censoring.",
    "sample_size_budget": "Record planned, attempted, complete, and censored counts and the frozen stopping rule.",
    "acceptance_gate_results": "Each check has expected, observed, passed, and a short principle.",
    "gate_check_summary": "Every block names upstream, check, artifact field, expected value, and observed value.",
    "verifier_is_oracle": "Declare shared executor authority; true forbids a positive scientific class.",
    "honest_verdict": "Completed findings start complete_; external absence starts blocked_. State the actual result.",
    "verdict_class": "Use the closed verdict enum; only unfinished own work is partial.",
    "validation_receipts": "Keep exact commands, scopes, exits, elapsed times, log hashes, and failures.",
    "repository_health": "Preserve dated unrelated failures as observations, never as passed required checks.",
    "field_principles": "Explain why each field exists without wrapping executable scalars or ordinary dictionaries.",
    "arc_session_complete_score": "One means the bounded attempt is fully accounted, even when its result is null.",
    "arc_tool_use_score": "One requires result-to-later-request-to-policy-to-action evidence.",
    "solve_provenance": "Use live_agent_self_discovery for current live attempts.",
    "per_game_results": "Record target, disabled adapters, actions, errors, changed-cell metrics, and censoring.",
    "tool_use_chain": "Link raw tool result, subsequent request, consumed policy plan, and observed action.",
    "cumulative_induction_ledger": "Hash and deduplicate ten historical inductions before adding actual new ones.",
    "runner_receipt": "Record exact Qwen identity, lease, PID/start tick, GPU use, and all decode limits.",
    "selection_receipt": "Freeze the least recently measured registered target before current outcomes.",
    "official_score": "Remain unset because this local case study is not an official evaluation.",
    "new_solve_claimed": "Remain false because cleared public levels receive no new credit.",
    "registry_modified": "Remain false because this task cannot add public solve credit.",
    "production_default_changed": "Remain false because this session does not change submitted defaults.",
    "population_improvement_claimed": "Remain false because one game is a case study.",
    "claim_scope": "State that one clustered session has no population inference.",
}


authority_artifact_checksum = authority.artifact_checksum
prior_artifact_checksum = prior.artifact_checksum
prior_session_environment = prior.session_environment
sha256_file = prior.sha256_file
atomic_write = prior.atomic_write


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def gate_check(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    principle: str = "Reject unsafe evidence before it can affect the current result.",
) -> JsonDict:
    return authority.gate_check(check, upstream, artifact_field, expected, observed, principle)


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    return authority.gate_summary(checks)


def _quarantined(value: Mapping[str, Any]) -> bool:
    return bool(value.get("flagged_adversarial") or value.get("quarantined"))


def check_dependency(artifact: Mapping[str, Any] | None) -> JsonDict:
    """Return the first unsafe authority state before reading its readiness score."""

    upstream = EXP7318_PATH.as_posix()
    if artifact is None:
        return gate_check("exp7318_dependency", upstream, "artifact", "available", "missing")
    if _quarantined(artifact):
        return gate_check("exp7318_dependency", upstream, "quarantine_state", False, True)
    verdict = artifact.get("verdict_class")
    for rejected in ("disqualified", "blocked", "partial"):
        if verdict == rejected:
            return gate_check(
                "exp7318_dependency",
                upstream,
                "verdict_class",
                f"not_{rejected}",
                verdict,
            )
    if artifact.get("status") != "complete":
        return gate_check(
            "exp7318_dependency", upstream, "status", "complete", artifact.get("status")
        )
    if artifact.get("arc_authority_ready_score") != 1:
        return gate_check(
            "exp7318_dependency",
            upstream,
            "arc_authority_ready_score",
            1,
            artifact.get("arc_authority_ready_score"),
        )
    checksum = artifact.get("reproducibility_checksum")
    observed: Any = "missing"
    if isinstance(checksum, str):
        observed = checksum == authority_artifact_checksum(artifact)
    return gate_check(
        "exp7318_dependency",
        upstream,
        "reproducibility_checksum_valid",
        True,
        observed,
    )


def authenticate_caller_hashes(
    root: Path,
    artifact: Mapping[str, Any],
    *,
    hasher: Callable[[Path], str] = sha256_file,
) -> list[JsonDict]:
    """Compare the scored caller path with the exact Exp7318 receipts."""

    live = artifact.get("live_entrypoint_receipt")
    hashes = live.get("file_hashes") if isinstance(live, Mapping) else None
    if not isinstance(hashes, Mapping) or not hashes:
        return [
            gate_check(
                "exp7318_caller_hash",
                EXP7318_PATH.as_posix(),
                "live_entrypoint_receipt.file_hashes",
                "mapping",
                "missing",
            )
        ]
    caller_roles = ("live_launcher", "provenance_construction", "scored_caller")
    selected = [hashes.get(role) for role in caller_roles if role in hashes]
    if not selected:
        return [
            gate_check(
                "exp7318_caller_hash",
                EXP7318_PATH.as_posix(),
                "live_entrypoint_receipt.caller_hashes",
                "mapping",
                "missing",
            )
        ]
    rows: list[JsonDict] = []
    for receipt in selected:
        if not isinstance(receipt, Mapping):
            rows.append(
                gate_check(
                    "exp7318_caller_hash",
                    EXP7318_PATH.as_posix(),
                    "file_hash_receipt",
                    "mapping",
                    type(receipt).__name__,
                )
            )
            continue
        relative = str(receipt.get("path") or "")
        path = root / relative
        observed = hasher(path) if relative and path.is_file() else "missing"
        rows.append(
            gate_check(
                "exp7318_caller_hash",
                relative or EXP7318_PATH.as_posix(),
                "sha256",
                receipt.get("sha256"),
                observed,
            )
        )
    return rows


def freeze_target(
    registry: Mapping[str, Any],
    *,
    adaptered_games: set[str] | frozenset[str],
    previous_target: str,
) -> JsonDict:
    """Freeze the target not used by the immediately preceding measured session."""

    indexed = {
        str(row.get("game")): dict(row)
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    available = [game for game in TARGET_ROTATION if game in indexed]
    target = next((game for game in available if game != previous_target), None)
    selected = indexed.get(target or "", {})
    passed = bool(
        len(available) == len(TARGET_ROTATION)
        and target == TARGET_GAME
        and target in adaptered_games
    )
    return {
        "passed": passed,
        "target": target,
        "rotation": list(TARGET_ROTATION),
        "seed": EVALUATION_SEED,
        "selection_basis": "frozen_least_recently_measured_rotation",
        "least_recently_measured_basis": {
            game: "exp7305" if game == previous_target else None for game in TARGET_ROTATION
        },
        "previous_measured_target": previous_target,
        "registry_prechecked": len(available) == len(TARGET_ROTATION),
        "registry_levels_before_attempt": int(selected.get("levels_reproduced") or 0),
        "registry_reproducibility": selected.get("reproducibility"),
        "adapter_available_but_withheld": target in adaptered_games,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "outcomes_seen_before_freeze": False,
        "registered_levels_are_transfer_only": True,
        "game_source_read": False,
        "offline_ground_truth_search_used": False,
        "hand_built_game_model_used": False,
    }


def session_environment(
    base_env: Mapping[str, str],
    *,
    episode_dir: Path,
    gpu_index: int,
    port: int,
    boundary_path: Path | None = None,
    arm: str = "direct_selfparse",
) -> dict[str, str]:
    """Reuse the authenticated selfparse environment with only the sealed seed changed."""

    environment = prior_session_environment(
        base_env,
        episode_dir=episode_dir,
        gpu_index=gpu_index,
        port=port,
        boundary_path=boundary_path,
        arm=arm,
    )
    environment["CARNOT_ARC_RANDOM_SEED"] = str(EVALUATION_SEED)
    environment["CARNOT_ARC_GENERATOR_SEED"] = str(EVALUATION_SEED)
    return environment


def _request_with_result(
    requests: Sequence[Mapping[str, Any]], turn: int, bounded_response: str
) -> str | None:
    for request in requests:
        if int(request.get("call_index", -1) or -1) <= turn:
            continue
        path = str(request.get("request_path") or "")
        if prior._request_contains(path, bounded_response):
            return path
    return None


def reduce_tool_use_chain(
    episode: Mapping[str, Any], tool_events: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Require the runtime result, later prompt, policy plan, and action links."""

    requests = [row for row in episode.get("raw_request_manifest", []) if isinstance(row, Mapping)]
    inductions = [row for row in episode.get("induction_rows", []) if isinstance(row, Mapping)]
    consumptions = [
        row for row in episode.get("policy_consumption_rows", []) if isinstance(row, Mapping)
    ]
    actions = [row for row in episode.get("action_rows", []) if isinstance(row, Mapping)]
    rows: list[JsonDict] = []
    for event_index, event in enumerate(tool_events):
        induction_index = int(event.get("induction_index", event_index) or 0)
        turn = int(event.get("turn", 0) or 0)
        dispatch = event.get("dispatch_result")
        successful = isinstance(dispatch, Mapping) and dispatch.get("ok") is True
        bounded = str(event.get("bounded_response") or "")
        later_path = (
            _request_with_result(requests, turn, bounded) if successful and bounded else None
        )
        planned = bool(
            later_path
            and induction_index < len(inductions)
            and inductions[induction_index].get("planned") is True
        )
        consumption = next(
            (
                dict(row)
                for row in consumptions
                if row.get("attempt_index") == induction_index
                and row.get("policy_action_executed") is True
            ),
            None,
        )
        if not planned:
            consumption = None
        action = next(
            (
                dict(row)
                for row in actions
                if consumption is not None
                and int(row.get("i", -1) or -1) >= int(consumption.get("action_index", 0) or 0)
            ),
            None,
        )
        absent: list[str] = []
        if not successful:
            absent.append("runtime_result")
        if later_path is None:
            absent.append("later_request")
        if not planned:
            absent.append("installed_plan")
        if consumption is None:
            absent.append("policy_consumption")
        if action is None:
            absent.append("environment_action")
        rows.append(
            {
                "event_index": event_index,
                "induction_index": induction_index,
                "tool_name": event.get("parsed_tool"),
                "tool_arguments": deepcopy(event.get("parsed_arguments")),
                "runtime_result": deepcopy(dispatch),
                "later_request_path": later_path,
                "induction_installed_plan": planned,
                "policy_consumption": consumption,
                "environment_action": action,
                "absent_links": absent,
                "actual_exception": event.get("exception"),
                "passed": not absent,
            }
        )
    return {
        "arc_tool_use_score": int(any(row["passed"] for row in rows)),
        "successful_tool_results": sum("runtime_result" not in row["absent_links"] for row in rows),
        "results_in_later_requests": sum(
            "later_request" not in row["absent_links"] for row in rows
        ),
        "policy_consumed_results": sum(
            "policy_consumption" not in row["absent_links"] for row in rows
        ),
        "subsequent_environment_actions": sum(
            "environment_action" not in row["absent_links"] for row in rows
        ),
        "rows": rows,
    }


def _current_induction_rows(episode: Mapping[str, Any]) -> tuple[list[JsonDict], int]:
    projected: list[JsonDict] = []
    censored = 0
    episode_id = str(episode.get("episode_id") or "")
    for index, raw in enumerate(episode.get("induction_rows", [])):
        if not isinstance(raw, Mapping):
            censored += 1
            continue
        row = deepcopy(dict(raw))
        explicit = str(row.get("induction_id") or "")
        explicit_authentic = bool(
            HASH_RE.fullmatch(explicit)
            and row.get("engaged") is True
            and row.get("source_authenticated") is True
        )
        gap = row.get("tool_gap")
        gap_authentic = bool(
            isinstance(gap, Mapping)
            and gap.get("selfparse", True) is True
            and int(gap.get("tool_calls_total", 0) or 0) > 0
            and gap.get("terminated_by")
        )
        if not explicit_authentic and not gap_authentic:
            censored += 1
            continue
        if not explicit_authentic:
            identity = {
                "episode_id": episode_id,
                "attempt_index": int(row.get("attempt_index", index) or index),
                "reason": row.get("reason"),
                "started_at": row.get("started_at"),
                "tool_gap": gap,
            }
            explicit = "sha256:" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        projected.append(
            {
                "induction_id": explicit,
                "source_session_id": episode_id,
                "source_authenticated": True,
                "engaged": True,
                "attempt_index": int(row.get("attempt_index", index) or index),
                "reason": row.get("reason"),
                "content_hash_basis": "sealed_current_attempt_identity",
            }
        )
    return projected, censored


def build_cumulative_ledger(
    output_path: Path,
    upstream_path: Path,
    historical_path: Path,
    episode: Mapping[str, Any],
) -> JsonDict:
    """Authenticate Exp7305's ten hashes and add only distinct current rows."""

    try:
        upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        upstream = {}
        historical = {}
    upstream_checksum = upstream.get("reproducibility_checksum")
    ledger_ref = upstream.get("cumulative_induction_ledger")
    ledger_ref = ledger_ref if isinstance(ledger_ref, Mapping) else {}
    historical_rows = historical.get("rows")
    historical_rows = historical_rows if isinstance(historical_rows, list) else []
    historical_ids = [
        str(row.get("induction_id") or "") for row in historical_rows if isinstance(row, Mapping)
    ]
    historical_authenticated = bool(
        upstream.get("status") == "complete"
        and upstream.get("verdict_class") not in {"blocked", "partial", "disqualified"}
        and not _quarantined(upstream)
        and isinstance(upstream_checksum, str)
        and upstream_checksum == prior_artifact_checksum(upstream)
        and ledger_ref.get("sha256") == sha256_file(historical_path)
        and ledger_ref.get("authenticated_unique_inductions") == 10
        and historical.get("authenticated_unique_inductions") == 10
        and len(historical_ids) == 10
        and len(set(historical_ids)) == 10
        and all(HASH_RE.fullmatch(identity) for identity in historical_ids)
        and all(
            isinstance(row, Mapping)
            and row.get("engaged") is True
            and row.get("source_authenticated") is True
            for row in historical_rows
        )
    )
    retained = [deepcopy(dict(row)) for row in historical_rows] if historical_authenticated else []
    by_id = {str(row["induction_id"]): row for row in retained}
    source_receipts = historical.get("source_receipts")
    source_receipts = source_receipts if isinstance(source_receipts, list) else []
    historical_source_count = sum(
        int(row.get("accepted_inductions", 0) or 0)
        for row in source_receipts
        if isinstance(row, Mapping)
    )
    current, censored = _current_induction_rows(episode)
    current_duplicates = 0
    new_count = 0
    for row in current:
        identity = str(row["induction_id"])
        if identity in by_id:
            current_duplicates += 1
            continue
        by_id[identity] = row
        retained.append(row)
        new_count += 1
    sidecar: JsonDict = {
        "schema": "carnot.experiment_7319.cumulative_induction_ledger.v1",
        "historical_source_path": str(historical_path),
        "historical_source_sha256": (
            sha256_file(historical_path) if historical_path.is_file() else "missing"
        ),
        "historical_ledger_authenticated": historical_authenticated,
        "historical_authenticated_count": len(historical_ids) if historical_authenticated else 0,
        "historical_duplicate_count": max(0, historical_source_count - len(historical_ids))
        if historical_authenticated
        else 0,
        "new_authentic_count": new_count,
        "current_duplicate_count": current_duplicates,
        "censored_attempt_count": censored,
        "cumulative_total": len(retained),
        "rows": retained,
    }
    atomic_write(output_path, sidecar)
    return {
        **sidecar,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
    }


def _episode_accounted(episode: Mapping[str, Any]) -> bool:
    return bool(
        episode.get("disposition") in {"complete", "censored_timeout"}
        and episode.get("censored") in {True, False}
        and episode.get("fresh_store") is True
        and int(episode.get("action_count") or 0) <= ACTION_LIMIT
        and int(episode.get("generation_calls_completed") or 0) <= COMPLETION_LIMIT
        and int(episode.get("generated_tokens") or 0) <= GENERATED_TOKEN_LIMIT
        and episode.get("adapter_disabled") is True
        and episode.get("banked_solution_disabled") is True
    )


def reduce_current_session(
    episode: Mapping[str, Any],
    boundary: Mapping[str, Any],
    tool_events: Sequence[Mapping[str, Any]],
) -> JsonDict:
    counts = boundary.get("invocation_counts")
    counts = (
        deepcopy(dict(counts)) if isinstance(counts, Mapping) else deepcopy(ZERO_INVOCATION_COUNTS)
    )
    chain = reduce_tool_use_chain(episode, tool_events)
    boundary_authenticated = bool(
        boundary.get("activity_known") is True
        and boundary.get("disqualified") is False
        and boundary.get("model_invoked") is True
        and int(counts.get("model_loads_attempted", 0) or 0) > 0
    )
    complete = int(boundary_authenticated and _episode_accounted(episode))
    tool_score = int(complete == 1 and chain["arc_tool_use_score"] == 1)
    return {
        "arc_session_complete_score": complete,
        "arc_tool_use_score": tool_score,
        "verdict_class": "circular_positive" if tool_score else "null",
        "invocation_counts": counts,
        "tool_use_chain": chain,
        "episode_accounted": _episode_accounted(episode),
        "boundary_evidence_authenticated": boundary_authenticated,
    }


def _model_specs(boundary: Mapping[str, Any], runner: Mapping[str, Any]) -> list[JsonDict]:
    call_rows = boundary.get("call_rows")
    identities = [
        row.get("model_identity")
        for row in call_rows
        if isinstance(call_rows, list) and isinstance(row, Mapping)
        if isinstance(row.get("model_identity"), Mapping)
    ]
    identity = dict(identities[0]) if identities else {}
    runner_identity = runner.get("model_identity")
    runner_identity = runner_identity if isinstance(runner_identity, Mapping) else {}
    return [
        {
            "hf_id": identity.get("model_repository") or runner_identity.get("hf_id") or MODEL_ID,
            "quantization": QUANTIZATION,
            "model_filename": identity.get("model_filename")
            or runner_identity.get("model_filename"),
            "model_revision": identity.get("model_revision") or runner_identity.get("revision"),
            "model_path": identity.get("model_path") or runner_identity.get("model_path"),
            "model_file_hash": runner_identity.get("model_hash"),
        }
    ]


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    required = validation_scope.reduce_required_checks(receipts)
    extra = [
        row
        for row in receipts
        if row.get("name")
        in {
            "cold_replay_current_receipts",
            "terminal_candidate_adversarial_verify",
            "terminal_candidate_row_consistency_strict",
        }
    ]
    return bool(
        required["required_checks_passed"]
        and all(row.get("passed") is True and row.get("exit_code") == 0 for row in extra)
    )


def _acceptance_results(
    reduction: Mapping[str, Any],
    cumulative: Mapping[str, Any],
    runner: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    values = (
        (
            "bounded_session_accounted",
            1,
            reduction["arc_session_complete_score"],
            "Scientific success is separate from complete attempt accounting.",
        ),
        (
            "historical_inductions_authenticated",
            10,
            cumulative.get("historical_authenticated_count"),
            "The current total must retain the ten content-hash identities.",
        ),
        (
            "one_task_owned_cuda_runner",
            True,
            bool(runner.get("task_linked_cuda_execution") and runner.get("model_count") == 1),
            "A current live claim needs one task-owned GPU and one current model.",
        ),
        (
            "affected_scoped_validation",
            True,
            _validation_passed(validation_receipts),
            "Any failing affected check disqualifies the current result.",
        ),
    )
    return [
        {
            "check": check,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
            "principle": principle,
        }
        for check, expected, observed, principle in values
    ]


def _field_principles(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        key: FIELD_PRINCIPLES.get(key, "Keep this value explicit so the record stays auditable.")
        for key in artifact
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    episode: Mapping[str, Any],
    boundary: Mapping[str, Any],
    runner: Mapping[str, Any],
    tool_events: Sequence[Mapping[str, Any]],
    cumulative_ledger: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    repository_health: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    reduction = reduce_current_session(episode, boundary, tool_events)
    counts = reduction["invocation_counts"]
    generation_attempted = int(counts.get("generation_calls_attempted", 0) or 0) > 0
    model_invoked = boundary.get("model_invoked") is True
    substrate = (
        "model_full_generation"
        if generation_attempted
        else "model_load_no_generation"
        if model_invoked
        else "aggregation"
    )
    validation_passed = _validation_passed(validation_receipts)
    raw_session_score = int(reduction["arc_session_complete_score"])
    raw_tool_score = int(reduction["arc_tool_use_score"])
    session_score = raw_session_score if validation_passed else 0
    tool_score = raw_tool_score if validation_passed else 0
    verdict_class = reduction["verdict_class"] if validation_passed else "disqualified"
    changed_metrics = episode.get("changed_cell_prediction_metrics")
    if not isinstance(changed_metrics, Mapping):
        changed_metrics = {
            "learned_accuracy": episode.get("heldout_accuracy"),
            "identity_accuracy": episode.get("identity_baseline_accuracy"),
            "same_changed_cells": (
                episode.get("heldout_accuracy") is not None
                and episode.get("identity_baseline_accuracy") is not None
            ),
        }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": _model_specs(boundary, runner) if model_invoked else [],
        "model_invoked": model_invoked,
        "invocation_counts": deepcopy(counts),
        "inference_substrate": substrate,
        "inference_substrate_class": substrate,
        "inference_mode": (
            "live_gpu" if runner.get("task_linked_cuda_execution") is True else "not_verified"
        ),
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [
            {
                "unit_id": episode.get("episode_id"),
                "arm": "direct_selfparse",
                "seed": episode.get("seed"),
                "metrics": {
                    "levels": int(episode.get("levels") or 0),
                    "arc_tool_use_score": tool_score,
                    "changed_cell_predictions": deepcopy(dict(changed_metrics)),
                },
                "costs": {
                    "actions": int(episode.get("action_count") or 0),
                    "generated_tokens": int(episode.get("generated_tokens") or 0),
                    "prompt_tokens": int(
                        (episode.get("compute_cost") or {}).get("prompt_tokens", 0)
                        if isinstance(episode.get("compute_cost"), Mapping)
                        else 0
                    ),
                    "wall_s": (
                        (episode.get("compute_cost") or {}).get("wall_s")
                        if isinstance(episode.get("compute_cost"), Mapping)
                        else None
                    ),
                },
                "failures": [str(episode.get("error"))] if episode.get("error") else [],
                "abstentions": int(tool_score == 0),
                "censored": bool(episode.get("censored")),
            }
        ],
        "sample_size_budget": {
            "planned_units": 1,
            "attempted_units": 1,
            "complete_units": int(episode.get("disposition") == "complete"),
            "censored_units": int(bool(episode.get("censored"))),
            "action_limit": ACTION_LIMIT,
            "completion_limit": COMPLETION_LIMIT,
            "generated_token_limit": GENERATED_TOKEN_LIMIT,
            "session_limit_s_including_load": SESSION_LIMIT_S,
            "model_load_limit_s": MODEL_LOAD_LIMIT_S,
            "stopping_rule": "one frozen target and seed; stop at the first inherited limit",
            "outcome_based_extension": False,
        },
        "acceptance_gate_results": _acceptance_results(
            reduction, cumulative_ledger, runner, validation_receipts
        ),
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": (
            "complete_disqualified_affected_validation_failed"
            if not validation_passed
            else "complete_circular_positive_runtime_tool_result_consumed_before_policy_action"
            if tool_score
            else "complete_null_no_runtime_tool_result_to_policy_action_chain"
        ),
        "verdict_class": verdict_class,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "arc_session_complete_score": session_score,
        "arc_tool_use_score": tool_score,
        "tool_use_chain": deepcopy(reduction["tool_use_chain"]),
        "solve_provenance": "live_agent_self_discovery",
        "per_game_results": [deepcopy(dict(episode))],
        "cumulative_induction_ledger": deepcopy(dict(cumulative_ledger)),
        "runner_receipt": deepcopy(dict(runner)),
        "selection_receipt": deepcopy(dict(selection)),
        "official_score": None,
        "new_solve_claimed": False,
        "registry_modified": False,
        "production_default_changed": False,
        "population_improvement_claimed": False,
        "claim_scope": "one adapter-withheld public-game case study; no population inference",
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:
    summary = gate_summary(preconditions)
    first = summary.get("first_failure") or {
        "upstream": EXPERIMENT_ID,
        "check": "unknown_external_precondition",
        "artifact_field": "state",
        "expected": "available",
        "observed": "unknown",
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": "blocked",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [],
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_mode": "not_invoked",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "random_seed": {
            "development": DEVELOPMENT_SEED,
            "independent_evaluation": EVALUATION_SEED,
            "sealed_before_outcomes": True,
        },
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 1,
            "attempted_units": 0,
            "complete_units": 0,
            "censored_units": 1,
            "stopping_rule": "block before model work when an external prerequisite fails",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": (
            "blocked_external_precondition:"
            f"{first['upstream']}:{first['check']}:{first['artifact_field']}:"
            f"expected={first['expected']!r}:observed={first['observed']!r}"
        ),
        "verdict_class": "blocked",
        "validation_receipts": [],
        "repository_health": _repository_health(),
        "arc_session_complete_score": 0,
        "arc_tool_use_score": 0,
        "tool_use_chain": {"arc_tool_use_score": 0, "rows": []},
        "solve_provenance": "not_started_external_block",
        "per_game_results": [],
        "cumulative_induction_ledger": {
            "historical_authenticated_count": 0,
            "new_authentic_count": 0,
            "cumulative_total": 0,
        },
        "runner_receipt": {},
        "selection_receipt": {},
        "official_score": None,
        "new_solve_claimed": False,
        "registry_modified": False,
        "production_default_changed": False,
        "population_improvement_claimed": False,
        "claim_scope": "no live session started because an external prerequisite failed",
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(artifact)
    artifact["field_principles"]["field_principles"] = FIELD_PRINCIPLES["field_principles"]
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any]) -> list[str]:
    """Cold-check terminal identity, accounting, safety scores, and checksum."""

    artifact = dict(value)
    errors: list[str] = []
    if (artifact.get("schema"), artifact.get("experiment_id"), artifact.get("milestone")) != (
        SCHEMA,
        EXPERIMENT_ID,
        MILESTONE,
    ):
        errors.append("identity_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_not_terminal")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    prefix = "blocked_" if artifact.get("status") == "blocked" else "complete_"
    if not str(artifact.get("honest_verdict") or "").startswith(prefix):
        errors.append("honest_verdict_prefix_invalid")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("official_score") is not None:
        errors.append("official_score_must_be_null")
    if any(
        artifact.get(field) is not False
        for field in (
            "new_solve_claimed",
            "registry_modified",
            "production_default_changed",
            "population_improvement_claimed",
        )
    ):
        errors.append("forbidden_claim_or_state_change")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("status") == "blocked":
        if artifact.get("model_invoked") is not False:
            errors.append("blocked_model_invoked")
        if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
            errors.append("blocked_invocation_counts_invalid")
        if (
            artifact.get("arc_session_complete_score") != 0
            or artifact.get("arc_tool_use_score") != 0
        ):
            errors.append("blocked_scores_nonzero")
        if artifact.get("gate_check_summary", {}).get("first_failure") is None:
            errors.append("blocked_failure_summary_missing")
        return errors
    per_game = artifact.get("per_game_results")
    episode = per_game[0] if isinstance(per_game, list) and len(per_game) == 1 else {}
    if not isinstance(episode, Mapping) or not _episode_accounted(episode):
        errors.append("episode_accounting_invalid")
    validation_passed = _validation_passed(
        artifact.get("validation_receipts", [])
        if isinstance(artifact.get("validation_receipts"), list)
        else []
    )
    if artifact.get("verdict_class") == "disqualified":
        if (
            artifact.get("arc_session_complete_score") != 0
            or artifact.get("arc_tool_use_score") != 0
        ):
            errors.append("disqualified_scores_nonzero")
    elif not validation_passed:
        errors.append("non_disqualified_validation_failure")
    cumulative = artifact.get("cumulative_induction_ledger")
    if not isinstance(cumulative, Mapping) or (
        cumulative.get("historical_ledger_authenticated") is not True
        or cumulative.get("historical_authenticated_count") != 10
    ):
        errors.append("historical_induction_ledger_invalid")
    if artifact.get("model_invoked") is True:
        specs = artifact.get("MODEL_SPECS")
        if not isinstance(specs, list) or len(specs) != 1 or specs[0].get("hf_id") != MODEL_ID:
            errors.append("current_model_spec_invalid")
        substrate = artifact.get("inference_substrate_class")
        floor = 60 if substrate == "model_full_generation" else 2
        if float(artifact.get("duration_s", 0) or 0) < floor:
            errors.append("substrate_duration_floor_failed")
        if substrate == "model_full_generation" and artifact.get("inference_mode") != "live_gpu":
            errors.append("live_gpu_receipt_missing")
    return errors


def independent_reduce(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload["episode"]
    boundary = payload["boundary"]
    tool_events = payload["tool_events"]
    reduction = reduce_current_session(episode, boundary, tool_events)
    actions = [row for row in episode.get("action_rows", []) if isinstance(row, Mapping)]
    counts = boundary.get("invocation_counts", {})
    completed = int(counts.get("generation_calls_completed", 0) or 0)
    episode_completed = int(episode.get("generation_calls_completed", 0) or 0)
    action_count = int(episode.get("action_count", 0) or 0)
    action_receipts_valid = all("i" in row and "action" in row for row in actions)
    return {
        **reduction,
        "action_receipts_replayed": len(actions),
        "generation_receipts_replayed": completed,
        "cold_replay_passed": bool(
            reduction["arc_session_complete_score"] == 1
            and completed == episode_completed
            and action_receipts_valid
            and len(actions) <= action_count
        ),
    }


def _repository_health() -> JsonDict:
    path = REPO_ROOT / HISTORICAL_REPOSITORY_HEALTH_LOG
    return validation_scope.build_repository_health(
        [
            {
                "observed_at": "2026-09-14",
                "source_experiment": "exp7289-arc-boundary",
                "command": ".venv/bin/pytest -o addopts= tests/python -q --no-cov -n 0",
                "exit_code": 2,
                "resolved": False,
                "log_path": HISTORICAL_REPOSITORY_HEALTH_LOG.as_posix(),
                "log_sha256": sha256_file(path) if path.is_file() else None,
            }
        ]
    )


def run_scoped_validation(private_root: Path) -> JsonDict:  # pragma: no cover - subprocesses.
    private_root.mkdir(parents=True, exist_ok=True)
    return validation_scope.run_scoped_validation(
        REPO_ROOT,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=private_root,
        coverage_file=private_root / ".coverage",
        log_dir=REPO_ROOT / RAW_DIR / "validation",
        historical_failures=_repository_health()["historical_failures"],
    )


def _utc_now() -> str:  # pragma: no cover - integration timestamp.
    return datetime.now(UTC).isoformat()


def _progress(started: float, phase: str, event: str, **fields: Any) -> None:  # pragma: no cover
    print(
        json.dumps(
            {
                "phase": phase,
                "event": event,
                "elapsed_s": round(time.monotonic() - started, 3),
                **fields,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _load_json(path: Path) -> JsonDict | None:  # pragma: no cover - integration input.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _span(phase: str, start: float, origin: float, **fields: Any) -> JsonDict:  # pragma: no cover
    end = time.monotonic()
    return {
        "phase": phase,
        "start_elapsed_s": round(start - origin, 6),
        "end_elapsed_s": round(end - origin, 6),
        "duration_s": round(end - start, 6),
        **fields,
    }


def collect_preconditions(
    root: Path, *, gpu_wait_s: float = 120.0
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - live preflight.
    started = time.monotonic()
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    required = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("research-references.md"),
        EXCLUSION_PATH,
        Path("ops/e2e-test-plan.md"),
        REGISTRY_PATH,
        SPEC_PATH,
        EXP7318_PATH,
        EXP7305_PATH,
        HISTORICAL_LEDGER_PATH,
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        Path("python/carnot/experiment_7305_v642_arc_selfparse.py"),
        Path("python/carnot/experiment_7318_v643_arc_authority.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_eval_provenance.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("scripts/arc_loop_solve.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in required:
        path = root / relative
        checks.append(
            gate_check("required_input", relative.as_posix(), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes[relative.as_posix()] = {
                "sha256": sha256_file(path),
                "role": "current_source_or_input",
                "authorizes_readiness": relative == EXP7318_PATH,
            }
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_check(
            "driving_capability",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7319",
            True,
            "REQ-ARC-WMTE-7319" in spec,
        )
    )
    dependency = _load_json(root / EXP7318_PATH)
    dependency_gate = check_dependency(dependency)
    checks.append(dependency_gate)
    if dependency is not None:
        checks.extend(authenticate_caller_hashes(root, dependency))
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
    checks.append(
        gate_check(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "current_task_rejected",
            False,
            prior._manifest_rejects(manifest, EXPERIMENT_ID),
        )
    )
    historical = build_cumulative_ledger(
        root / CUMULATIVE_PATH,
        root / EXP7305_PATH,
        root / HISTORICAL_LEDGER_PATH,
        {"induction_rows": []},
    )
    checks.append(
        gate_check(
            "historical_induction_ledger",
            HISTORICAL_LEDGER_PATH.as_posix(),
            "historical_authenticated_count",
            10,
            historical["historical_authenticated_count"],
        )
    )
    for relative in (RESULT_PATH.parent, RAW_DIR, CHECKPOINT_PATH.parent):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        checks.append(
            gate_check(
                "output_path",
                relative.as_posix(),
                "owner_and_writable",
                True,
                path.is_dir() and os.access(path, os.W_OK) and path.stat().st_uid == os.getuid(),
            )
        )
    deadline = time.monotonic() + max(0.0, gpu_wait_s)
    idle: list[JsonDict] = []
    while True:
        idle = [
            row
            for row in prior._gpu_inventory()
            if not row.get("compute_apps") and int(row.get("free_memory_mb") or 0) >= 20_000
        ]
        if idle or time.monotonic() >= deadline:
            break
        _progress(
            started,
            "gpu_reservation",
            "pending_idle_owned_gpu",
            remaining_s=round(max(0.0, deadline - time.monotonic()), 1),
        )
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    gpu = idle[0] if idle else None
    checks.append(
        gate_check("gpu_reservation", "nvidia-smi", "idle_gpu_with_20GB", True, gpu is not None)
    )
    from carnot.inference.sota_models import cached_current_model, gguf_tokenizer_loadable

    model = cached_current_model(
        gpu_index=int(gpu.get("index", 0)) if gpu else 0,
        preferred_quant=QUANTIZATION,
    )
    model_path = Path(str(model.get("model_path"))) if model else None
    model_ok = bool(
        model
        and model.get("hf_id") == MODEL_ID
        and model_path
        and model_path.is_file()
        and QUANTIZATION in model_path.name
    )
    checks.append(gate_check("cached_current_model", MODEL_ID, "Q4_K_M_path", True, model_ok))
    _progress(started, "embedded_tokenizer", "before_model_tokenizer_load")
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(model_path) if model_ok else None)
    _progress(
        started,
        "embedded_tokenizer",
        "after_model_tokenizer_load",
        passed=tokenizer_ok,
    )
    checks.append(
        gate_check(
            "embedded_tokenizer",
            str(model_path),
            "loadable",
            True,
            tokenizer_ok,
            tokenizer_detail,
        )
    )
    server_candidates = (
        os.environ.get("CARNOT_LLAMA_SERVER"),
        str(Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"),
    )
    server = next(
        (Path(value) for value in server_candidates if value and Path(value).is_file()), None
    )
    checks.append(
        gate_check("native_runtime", "llama.cpp", "server_binary", True, server is not None)
    )
    model_hash = None
    if model_ok and model_path is not None:
        _progress(started, "model_hash", "before_cached_model_hash", path=str(model_path))
        model_hash = sha256_file(model_path)
        hashes[str(model_path)] = {
            "sha256": model_hash,
            "role": "current_model",
            "authorizes_readiness": True,
        }
        _progress(started, "model_hash", "after_cached_model_hash", sha256=model_hash)
    if server is not None:
        hashes[str(server)] = {
            "sha256": sha256_file(server),
            "role": "native_runtime_binary",
            "authorizes_readiness": True,
        }
    return (
        checks,
        hashes,
        {
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "model_spec": deepcopy(model) if model else {},
            "server": str(server) if server else None,
            "gpu": deepcopy(gpu) if gpu else None,
            "resolved_MODEL_SPECS": [deepcopy(model)] if model_ok and model else [],
        },
    )


@contextmanager
def _configured_runtime() -> Any:  # pragma: no cover - live integration.
    updates = {
        "RUN_DATE": RUN_DATE,
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MODEL_SPECS": MODEL_SPECS,
        "TARGET_GAME": TARGET_GAME,
        "DEVELOPMENT_SEED": DEVELOPMENT_SEED,
        "EVALUATION_SEED": EVALUATION_SEED,
        "ACTION_LIMIT": ACTION_LIMIT,
        "COMPLETION_LIMIT": COMPLETION_LIMIT,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "TOKENS_PER_CALL": TOKENS_PER_CALL,
        "SESSION_LIMIT_S": SESSION_LIMIT_S,
        "MODEL_LOAD_LIMIT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "TOOL_EVENT_PATH": TOOL_EVENT_PATH,
        "SESSION_PATH": SESSION_PATH,
        "RAW_ROW_PATH": RAW_ROW_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "session_environment": session_environment,
    }
    old = {name: getattr(prior, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(prior, name, value)
        with prior._configured_runtime() as reused:
            yield reused
    finally:
        for name, value in old.items():
            setattr(prior, name, value)


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live child.
    with _configured_runtime():
        return int(prior.run_live_session(args))


def _censored_episode(session: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live reduction.
    episodes = session.get("episodes")
    if isinstance(episodes, list) and episodes and isinstance(episodes[0], Mapping):
        row = deepcopy(dict(episodes[0]))
        row.setdefault("fresh_store", True)
        row.setdefault("adapter_disabled", True)
        row.setdefault("banked_solution_disabled", True)
        return row
    requests = session.get("requests") if isinstance(session.get("requests"), list) else []
    return {
        "episode_id": f"{TARGET_GAME}:direct_selfparse",
        "game": TARGET_GAME,
        "seed": EVALUATION_SEED,
        "arm": "direct_selfparse",
        "disposition": "censored_timeout" if session.get("timed_out") else "complete",
        "censored": bool(session.get("timed_out")),
        "model_loaded": bool(session.get("model_loaded")),
        "model_invoked": bool(session.get("model_invoked")),
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": sum(
            isinstance(row, Mapping) and row.get("transport_completed") is True for row in requests
        ),
        "generated_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in requests if isinstance(row, Mapping)
        ),
        "action_count": 0,
        "action_limit": ACTION_LIMIT,
        "levels": 0,
        "adapter_disabled": True,
        "banked_solution_disabled": True,
        "fresh_store": True,
        "induction_rows": [],
        "policy_consumption_rows": [],
        "raw_request_manifest": deepcopy(requests),
        "action_rows": [],
        "error": session.get("error") or "session_ended_before_episode_row",
    }


def _finalize_runner_receipt(
    session: Mapping[str, Any], resources: Mapping[str, Any], episode: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - live receipt.
    runtime = deepcopy(dict(session.get("runtime_receipt") or {}))
    model_spec = session.get("model_spec")
    model_spec = model_spec if isinstance(model_spec, Mapping) else {}
    native = Path(str(runtime.get("native_binary") or ""))
    model_path = Path(str(resources.get("model_path") or ""))
    requests = [row for row in session.get("requests", []) if isinstance(row, Mapping)]
    runtime.update(
        {
            "model_count": 1,
            "runner_choice": "one_owned_gpu_one_native_llama_cpp_server",
            "dual_model_speedup_claimed": False,
            "model_identity": {
                "hf_id": MODEL_ID,
                "quantization": QUANTIZATION,
                "model_path": str(model_path),
                "model_filename": model_path.name,
                "model_hash": resources.get("model_hash"),
                "revision": model_spec.get("revision")
                or (resources.get("model_spec") or {}).get("revision"),
            },
            "native_binary_hash": sha256_file(native) if native.is_file() else None,
            "actual_request_settings": [
                deepcopy(dict(row.get("decoding_parameters") or {})) for row in requests
            ],
            "token_counts": {
                "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in requests),
                "generated_tokens": sum(int(row.get("completion_tokens") or 0) for row in requests),
            },
            "task_window_utilization": {
                "wall_fraction": min(1.0, float(session.get("duration_s") or 0) / SESSION_LIMIT_S),
                "action_fraction": int(episode.get("action_count") or 0) / ACTION_LIMIT,
                "completion_fraction": int(episode.get("generation_calls_completed") or 0)
                / COMPLETION_LIMIT,
                "generated_token_fraction": int(episode.get("generated_tokens") or 0)
                / GENERATED_TOKEN_LIMIT,
            },
            "child_phase_spans": deepcopy(session.get("phase_spans") or []),
        }
    )
    return runtime


def _run_receipt_commands(
    commands: Sequence[validation_scope.CommandSpec], log_dir: Path
) -> list[JsonDict]:  # pragma: no cover - subprocess validation.
    return validation_scope.run_commands(
        REPO_ROOT,
        commands,
        log_dir=log_dir,
        heartbeat_s=60.0,
    )


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - live orchestration.
    started = time.monotonic()
    started_utc = _utc_now()
    phase_spans: list[JsonDict] = []
    _progress(started, "startup", "entrypoint_and_paths_authenticated", run_date=args.date)
    phase_start = time.monotonic()
    _progress(started, "preconditions", "begin")
    checks, hashes, resources = collect_preconditions(REPO_ROOT, gpu_wait_s=args.gpu_wait_s)
    phase_spans.append(
        _span(
            "preconditions",
            phase_start,
            started,
            units=len(checks),
            checkpoint_boundary="pre_model",
            pending_operations=[],
        )
    )
    _progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in checks),
        units=len(checks),
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
            model_specs=resources.get("resolved_MODEL_SPECS", []),
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"blocked artifact validation failed: {errors}")
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        _progress(started, "publication", "terminal_blocked_artifact_written", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    _progress(started, "target_freeze", "begin")
    registry = yaml.safe_load((REPO_ROOT / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    from carnot.agentic.arc_game_adapters import adaptered_games

    selection = freeze_target(
        registry,
        adaptered_games=set(adaptered_games()),
        previous_target="r11l",
    )
    target_gate = gate_check(
        "registry_precheck_and_freeze",
        REGISTRY_PATH.as_posix(),
        "selected_target",
        TARGET_GAME,
        selection.get("target") if selection.get("passed") else "unavailable",
    )
    checks.append(target_gate)
    atomic_write(REPO_ROOT / RAW_DIR / "frozen_target.json", selection)
    phase_spans.append(
        _span(
            "target_freeze_and_history",
            phase_start,
            started,
            units=1,
            checkpoint_boundary="frozen_before_outcome",
            pending_operations=[],
        )
    )
    _progress(
        started,
        "target_freeze",
        "end",
        passed=target_gate["passed"],
        target=selection.get("target"),
    )
    if not target_gate["passed"]:
        artifact = build_blocked_artifact(
            started_at_utc=started_utc,
            ended_at_utc=_utc_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
            model_specs=resources.get("resolved_MODEL_SPECS", []),
        )
        atomic_write(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    schedule = [
        {
            "episode_id": f"{TARGET_GAME}:direct_selfparse",
            "game": TARGET_GAME,
            "arm": "direct_selfparse",
            "seed": EVALUATION_SEED,
            "execution_order": 0,
            "adapter_disabled": True,
            "banked_solution_disabled": True,
        }
    ]
    schedule_path = REPO_ROOT / RAW_DIR / "frozen_schedule.json"
    atomic_write(schedule_path, {"selection_receipt": selection, "rows": schedule})
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {"status": "running", "phase": "live_window", "completed_units": 0},
    )
    _progress(
        started,
        "live_window",
        "before_model_load_generation_benchmark",
        cap_s=SESSION_LIMIT_S,
    )
    phase_start = time.monotonic()
    with _configured_runtime() as reused:
        session = reused.run_child_with_lease(
            resources=resources,
            schedule_path=schedule_path,
            raw_dir=REPO_ROOT / RAW_DIR,
            checkpoint_path=REPO_ROOT / CHECKPOINT_PATH,
            session_path=REPO_ROOT / SESSION_PATH,
            remaining_s=SESSION_LIMIT_S,
        )
    phase_spans.append(
        _span(
            "live_window",
            phase_start,
            started,
            units=1,
            checkpoint_boundary="live_child_terminal",
            pending_operations=[],
        )
    )
    _progress(
        started,
        "live_window",
        "after_model_load_generation_benchmark",
        model_invoked=session.get("model_invoked"),
        completed_units=len(session.get("episodes") or []),
    )
    episode = _censored_episode(session)
    boundary = prior.reduce_boundary_ledger(REPO_ROOT / BOUNDARY_PATH)
    tool_events = prior._read_tool_events(REPO_ROOT / TOOL_EVENT_PATH)
    cumulative = build_cumulative_ledger(
        REPO_ROOT / CUMULATIVE_PATH,
        REPO_ROOT / EXP7305_PATH,
        REPO_ROOT / HISTORICAL_LEDGER_PATH,
        episode,
    )
    runner = _finalize_runner_receipt(session, resources, episode)
    raw_input = {"episode": episode, "boundary": boundary, "tool_events": tool_events}
    atomic_write(REPO_ROOT / RAW_ROW_PATH, raw_input)
    for path in sorted(item for item in (REPO_ROOT / RAW_DIR).rglob("*") if item.is_file()):
        hashes[str(path.relative_to(REPO_ROOT))] = {
            "sha256": sha256_file(path),
            "role": "current_raw_sidecar",
            "authorizes_readiness": False,
        }

    private = Path(tempfile.mkdtemp(prefix="exp7319-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    _progress(started, "validation", "before_scoped_subprocesses")
    validation = run_scoped_validation(private)
    receipts = list(validation["validation_receipts"])
    cold_command = validation_scope.CommandSpec(
        "cold_replay_current_receipts",
        (
            str(REPO_ROOT / ".venv/bin/python"),
            "-u",
            str(REPO_ROOT / WRAPPER_PATH),
            "--reduce-raw",
            str(REPO_ROOT / RAW_ROW_PATH),
        ),
        "current_invocation_and_action_receipts",
    )
    receipts.extend(_run_receipt_commands([cold_command], REPO_ROOT / RAW_DIR / "cold-replay"))
    phase_spans.append(
        _span(
            "scoped_validation_and_cold_replay",
            phase_start,
            started,
            units=len(receipts),
            checkpoint_boundary="measured_candidate_ready",
            pending_operations=[],
        )
    )
    _progress(
        started,
        "validation",
        "after_scoped_subprocesses",
        passed=_validation_passed(receipts),
        units=len(receipts),
    )
    candidate = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner=runner,
        tool_events=tool_events,
        cumulative_ledger={key: value for key, value in cumulative.items() if key != "rows"},
        validation_receipts=receipts,
        repository_health=validation["repository_health"],
        phase_spans=phase_spans,
    )
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, candidate)
    phase_start = time.monotonic()
    terminal_commands = [
        validation_scope.CommandSpec(
            "terminal_candidate_adversarial_verify",
            (
                str(REPO_ROOT / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                str(REPO_ROOT / TERMINAL_CANDIDATE_PATH),
            ),
            "measured_terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "terminal_candidate_row_consistency_strict",
            (
                str(REPO_ROOT / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(REPO_ROOT / TERMINAL_CANDIDATE_PATH),
            ),
            "measured_terminal_candidate",
        ),
    ]
    _progress(started, "terminal_validation", "before_subprocesses", units=2)
    receipts.extend(
        _run_receipt_commands(terminal_commands, REPO_ROOT / RAW_DIR / "terminal-validation")
    )
    phase_spans.append(
        _span(
            "terminal_validation",
            phase_start,
            started,
            units=2,
            checkpoint_boundary="terminal_checks_complete",
            pending_operations=[],
        )
    )
    _progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        passed=_validation_passed(receipts),
        units=2,
    )
    artifact = build_terminal_artifact(
        started_at_utc=started_utc,
        ended_at_utc=_utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        episode=episode,
        boundary=boundary,
        runner=runner,
        tool_events=tool_events,
        cumulative_ledger={key: value for key, value in cumulative.items() if key != "rows"},
        validation_receipts=receipts,
        repository_health=validation["repository_health"],
        phase_spans=phase_spans,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    _progress(started, "publication", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_write(REPO_ROOT / TERMINAL_CANDIDATE_PATH, artifact)
    atomic_write(REPO_ROOT / RESULT_PATH, artifact)
    atomic_write(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "phase": "terminal_published",
            "completed_units": 1,
            "result_path": RESULT_PATH.as_posix(),
        },
    )
    _progress(
        started,
        "publication",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        arc_session_complete_score=artifact["arc_session_complete_score"],
        arc_tool_use_score=artifact["arc_tool_use_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover - CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-hash")
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--schedule-path", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--session-path", type=Path, default=SESSION_PATH)
    parser.add_argument("--gpu-wait-s", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    started = time.monotonic()
    _progress(started, "startup", "entrypoint")
    args = parse_args(argv)
    if args.reduce_raw is not None:
        print(json.dumps(independent_reduce(args.reduce_raw), sort_keys=True), flush=True)
        return 0
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.role == "live-session":
        return run_live_session(args)
    run_experiment(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
