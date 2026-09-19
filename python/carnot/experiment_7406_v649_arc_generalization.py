"""Run the checkpoint-qualified bounded ARC generalization panel.

The host path composes the qualified episode journal with the existing native
CUDA ARC runner. It keeps solution data outside the scored policy process and
reduces every sealed game-seed unit, including censored and unstarted units.

Spec refs: REQ-ARC-WMTE-7406 and SCENARIO-ARC-WMTE-7406-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7319_v643_arc_session as induction_base
from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7376_v647_arc_outcomes as live_base
from carnot import experiment_7398_v649_arc_checkpoint as checkpoint_base
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.649"
EXPERIMENT_ID = "exp7406-arc-generalization"
SCHEMA = "carnot.exp7406.v649_arc_generalization.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
TARGET_GAMES = ("bp35", "cn04", "dc22")
EPISODE_SEEDS = (7_406_202_609_19, 17_406_202_609_19)
RANDOM_SEED = 27_406_202_609_19
ACTION_LIMIT = 128
MODEL_CALL_LIMIT = 2
MAX_NEW_TOKENS = 256
GENERATED_TOKEN_LIMIT = MODEL_CALL_LIMIT * MAX_NEW_TOKENS
EPISODE_WORK_LIMIT_S = 240
TOTAL_EPISODE_WORK_LIMIT_S = 1800
MODEL_LOAD_LIMIT_S = 600
CURATED_ARMS = live_base.CURATED_ARMS
WITHHELD_INPUTS = (
    "registry_solution_data",
    "per_game_adapter",
    "game_source",
    "saved_engine",
    "checkpoint",
    "banked_solution",
    "hand_solver",
    "replay_route",
    "offline_ground_truth_search",
)

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
SUPERVISOR_LEDGER_PATH = Path("ops/arc_supervisor_refinement_ledger.json")
EXP7398_PATH = Path("results/experiment_7398_v649_arc_checkpoint.json")
EXP7305_PATH = Path("results/experiment_7305_v642_arc_selfparse.json")
HISTORICAL_INDUCTION_PATH = Path(
    "results/raw/experiment_7305_v642_arc_selfparse/cumulative_inductions.json"
)
HISTORICAL_REFERENCE_PATHS = {
    game: Path(f"results/arc_loop_solve_{game}.json") for game in TARGET_GAMES
}
RESULT_PATH = Path("results/experiment_7406_v649_arc_generalization.json")
RAW_DIR = Path("results/raw/experiment_7406_v649_arc_generalization")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7406_v649_arc_generalization.json")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "receipt_events.jsonl"
TOOL_EVENT_PATH = RAW_DIR / "tool_events.jsonl"
RAW_PANEL_PATH = RAW_DIR / "independent_reduction_input.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
EPISODE_CHECKPOINT_DIR = RAW_DIR / "episode_checkpoints"
CUMULATIVE_INDUCTION_PATH = RAW_DIR / "cumulative_inductions.json"
MODULE_PATH = Path("python/carnot/experiment_7406_v649_arc_generalization.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7406_v649_arc_generalization.py")
TEST_PATH = Path("tests/python/test_experiment_7406_v649_arc_generalization.py")

EXP7398_CODE_PATHS = (
    Path("python/carnot/experiment_7398_v649_arc_checkpoint.py"),
    Path("scripts/experiments/experiment_7398_v649_arc_checkpoint.py"),
    Path("tests/python/test_experiment_7398_v649_arc_checkpoint.py"),
)
REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_offline_smoke")
REQUIRED_TERMINAL_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
ZERO_INVOCATION_COUNTS = deepcopy(live_base.ZERO_INVOCATION_COUNTS)
CheckpointIntegrityError = checkpoint_base.CheckpointIntegrityError
HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7376_v647_arc_outcomes.py"),
    Path("python/carnot/experiment_7384_v648_arc_invocation_boundary.py"),
    Path("python/carnot/experiment_7398_v649_arc_checkpoint.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("scripts/arc_loop_solve.py"),
    REGISTRY_PATH,
    SUPERVISOR_LEDGER_PATH,
    Path("docs/research-notes/avo-adaptation-for-local-generator-2026-08-21.md"),
    Path("openspec/capabilities/research-reporting/spec.md"),
    SPEC_PATH,
    EXP7398_PATH,
    EXP7305_PATH,
    HISTORICAL_INDUCTION_PATH,
    *HISTORICAL_REFERENCE_PATHS.values(),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)


def utc_now() -> str:
    """Return one aware UTC timestamp for an observed boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print a flushed phase or long-operation boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7406] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes for schedules, rows, and terminal identity."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact bytes without reading a large model into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Flush a complete JSON object before one same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_object(path: Path) -> JsonDict:
    """Load one JSON object, or return an empty object for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the checksum slot itself."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def gate_row(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    upstream: str = "current_experiment",
    artifact_field: str | None = None,
    operator: str = "==",
) -> JsonDict:
    """Retain one exact comparison, including a missing observed value."""

    if operator == "==":
        passed = observed == expected
    elif operator == "is":
        passed = observed is expected
    elif operator == "in":
        passed = observed in expected
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field or check,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all failed gates and expose the first exact failure."""

    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "failed_checks": failed,
        "first_failure": failed[0] if failed else None,
    }


def check_checkpoint_dependency(
    artifact: Mapping[str, Any] | None,
    code_paths: Mapping[str, Path],
) -> JsonDict:
    """Authenticate Exp7398's structured fields and its three recorded code hashes."""

    value = dict(artifact or {})
    upstream = EXP7398_PATH.as_posix()
    checks = [
        gate_row(
            "exp7398_identity",
            "precondition",
            "exp7398-arc-checkpoint",
            value.get("experiment_id"),
            upstream=upstream,
            artifact_field="experiment_id",
        ),
        gate_row(
            "exp7398_checkpoint_ready",
            "precondition",
            1,
            value.get("arc_checkpoint_ready_score"),
            upstream=upstream,
            artifact_field="arc_checkpoint_ready_score",
        ),
        gate_row(
            "exp7398_verdict_eligible",
            "precondition",
            ["positive", "circular_positive", "null"],
            value.get("verdict_class"),
            upstream=upstream,
            artifact_field="verdict_class",
            operator="in",
        ),
        gate_row(
            "exp7398_adversarial_clear",
            "precondition",
            False,
            value.get("flagged_adversarial"),
            upstream=upstream,
            artifact_field="flagged_adversarial",
        ),
    ]
    recorded = value.get("source_artifact_hashes")
    recorded = dict(recorded) if isinstance(recorded, Mapping) else {}
    for relative in EXP7398_CODE_PATHS:
        key = relative.as_posix()
        path = code_paths.get(key)
        expected = (recorded.get(key) or {}).get("sha256")
        observed = sha256_file(path) if path is not None and path.is_file() else None
        required_expected = (
            expected if HASH_RE.fullmatch(str(expected or "")) else "recorded_sha256"
        )
        checks.append(
            gate_row(
                f"exp7398_code_hash:{key}",
                "precondition",
                required_expected,
                observed,
                upstream=upstream,
                artifact_field=f"source_artifact_hashes.{key}.sha256",
            )
        )
    return {
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "gate_check_summary": gate_summary(checks),
    }


def freeze_panel(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    """Reuse the predeclared three-game rotation without reading current outcomes."""

    selection = live_base.freeze_panel(registry, adaptered_games=adaptered_games)
    selection["label_blind"] = True
    selection["registry_use"] = "duplicate_credit_precheck_only"
    selection["policy_received_registry_data"] = False
    return selection


def build_schedule(games: Sequence[str]) -> list[JsonDict]:
    """Seal two fixed seeds for each of the three predeclared games."""

    rows = live_base.build_schedule(games, EPISODE_SEEDS)
    for index, row in enumerate(rows):
        row.update(
            {
                "execution_order": index,
                "sentinel": index == 0,
                "episode_work_limit_s": EPISODE_WORK_LIMIT_S,
                "action_limit": ACTION_LIMIT,
                "completion_limit": MODEL_CALL_LIMIT,
                "max_new_tokens_per_call": MAX_NEW_TOKENS,
                "generated_token_limit": GENERATED_TOKEN_LIMIT,
                "withheld_inputs": list(WITHHELD_INPUTS),
                "registry_solution_data_disabled": True,
                "game_source_disabled": True,
                "checkpoint_input_disabled": True,
                "off_path_engines_disabled": True,
            }
        )
    return rows


def session_environment(
    base_env: Mapping[str, str],
    *,
    arm: str,
    episode_dir: Path,
    gpu_index: int,
    port: int,
    boundary_path: Path | None = None,
) -> dict[str, str]:
    """Keep the curated supervisor and impose the fixed generation limits."""

    if arm not in {"curated_supervisor", "current_feedback"}:
        raise ValueError(f"unknown live arm: {arm}")
    episode_id = episode_dir.name.replace("__", ":")
    if _LIVE_STATE is not None and _LIVE_STATE.abort_after_sentinel:
        first_id = next(iter(_LIVE_STATE.schedule))
        if episode_id not in {first_id, "bootstrap"}:
            raise SentinelAbort("sentinel failed before later episode start")
    env = live_base.session_environment(
        base_env,
        arm=arm,
        episode_dir=episode_dir,
        gpu_index=gpu_index,
        port=port,
        boundary_path=boundary_path,
    )
    env.pop("CARNOT_7376_EPISODE_ID", None)
    env.update(
        {
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(MODEL_CALL_LIMIT),
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(MODEL_CALL_LIMIT),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(MAX_NEW_TOKENS),
            "CARNOT_7406_EPISODE_ID": episode_id,
        }
    )
    for key in (
        "CARNOT_ARC_SUPERVISOR_ORDER",
        "CARNOT_ARC_SUPERVISOR_ORDER_HASH",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
    ):
        env.pop(key, None)
    return env


def _checkpoint_path(directory: Path, episode: Mapping[str, Any]) -> Path:
    return directory / checkpoint_base._checkpoint_name(str(episode["episode_id"]))


def read_episode_events(directory: Path, episode: Mapping[str, Any]) -> JsonDict | None:
    """Read one journal through the qualified Exp7398 integrity checker."""

    return checkpoint_base.read_episode_checkpoint(_checkpoint_path(directory, episode), episode)


def append_episode_event(
    directory: Path,
    episode: Mapping[str, Any],
    kind: str,
    detail: Mapping[str, Any],
) -> JsonDict:
    """Append one hash-chained event using the next durable kind ordinal."""

    path = _checkpoint_path(directory, episode)
    journal = checkpoint_base.read_episode_checkpoint(path, episode)
    ordinal = sum(row.get("kind") == kind for row in (journal or {}).get("events", []))
    safe_detail = json.loads(json.dumps(dict(detail), default=str))
    return checkpoint_base._append_event(path, episode, kind, ordinal, safe_detail)


def _events(journal: Mapping[str, Any] | None, kind: str) -> list[JsonDict]:
    return [
        deepcopy(dict(row))
        for row in (journal or {}).get("events", [])
        if isinstance(row, Mapping) and row.get("kind") == kind
    ]


def _sentinel_passed(journal: Mapping[str, Any] | None) -> bool:
    policies = _events(journal, "policy_entered")
    requests = _events(journal, "request_attempted")
    actions = [
        row
        for row in _events(journal, "environment_action_completed")
        if row.get("detail", {}).get("action") not in {None, "RESET"}
    ]
    return bool(policies and requests and actions)


def project_tool_dispatch_results(
    induction_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Project observed tool events while keeping no-fire demand unknown."""

    projected: list[JsonDict] = []
    for index, raw in enumerate(induction_rows):
        row = dict(raw)
        tool_loop = row.get("tool_loop")
        tool_loop = dict(tool_loop) if isinstance(tool_loop, Mapping) else {}
        event_rows = tool_loop.get("event_rows") or row.get("tool_event_rows") or []
        events = [dict(item) for item in event_rows if isinstance(item, Mapping)]
        if not events:
            projected.append(
                {
                    "attempt_index": int(row.get("attempt_index", index) or index),
                    "tool_firing_observed": False,
                    "tool_demand_interpretation": "unknown_no_firing_observed",
                    "invented_attempt": False,
                    "parsed_tool": None,
                    "dispatch_result": None,
                }
            )
            continue
        for event in events:
            projected.append(
                {
                    "attempt_index": int(row.get("attempt_index", index) or index),
                    "tool_firing_observed": True,
                    "tool_demand_interpretation": "observed_runtime_tool_firing",
                    "invented_attempt": False,
                    "parsed_tool": event.get("parsed_tool"),
                    "parsed_arguments": deepcopy(event.get("parsed_arguments")),
                    "dispatch_result": deepcopy(event.get("dispatch_result")),
                    "bounded_response": event.get("bounded_response"),
                }
            )
    return projected


def _episode_row(
    episode: Mapping[str, Any],
    journal: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    *,
    sentinel_failed: bool,
    sentinel_attempted: bool,
) -> JsonDict:
    events = [deepcopy(dict(row)) for row in (journal or {}).get("events", [])]
    policies = _events(journal, "policy_entered")
    requests = _events(journal, "request_attempted")
    generations = _events(journal, "generation_completed")
    actions = [
        row
        for row in _events(journal, "environment_action_completed")
        if row.get("detail", {}).get("action") not in {None, "RESET"}
    ]
    inductions = [dict(row.get("detail") or {}) for row in _events(journal, "induction_attempt")]
    terminals = _events(journal, "episode_terminal")
    terminal = dict(terminals[-1].get("detail") or {}) if terminals else {}
    attempted = bool(events) or bool(episode.get("sentinel") and sentinel_attempted)
    if not attempted:
        disposition = "unstarted"
    elif episode.get("sentinel") and sentinel_failed:
        disposition = "censored_sentinel"
    elif terminals:
        disposition = str(terminal.get("disposition") or "complete_error")
    else:
        disposition = "censored_timeout"
    censored = disposition.startswith("censored_")
    completed = disposition in {"complete", "complete_error"}
    levels = [int(row.get("detail", {}).get("level") or 0) for row in actions]
    first_action_detail = dict(actions[0].get("detail") or {}) if actions else {}
    start_level = int(first_action_detail.get("start_level", levels[0] if levels else 0) or 0)
    progress_event = next(
        (row for row in actions if int(row.get("detail", {}).get("level") or 0) > start_level),
        None,
    )
    progress_detail = dict(progress_event.get("detail") or {}) if progress_event else {}
    policy_detail = dict(policies[0].get("detail") or {}) if policies else {}
    generation_details = [dict(row.get("detail") or {}) for row in generations]
    supervisor = terminal.get("trajectory_supervisor")
    supervisor = dict(supervisor) if isinstance(supervisor, Mapping) else {}
    redirects = [
        {
            **deepcopy(dict(row)),
            "causal_interpretation": "descriptive_association_only",
        }
        for row in supervisor.get("redirects", [])
        if isinstance(row, Mapping)
    ]
    tool_rows = terminal.get("tool_dispatch_result_rows")
    tool_rows = (
        [dict(row) for row in tool_rows if isinstance(row, Mapping)]
        if isinstance(tool_rows, list)
        else project_tool_dispatch_results(inductions)
    )
    journal_path = _checkpoint_path(Path("."), episode).name
    return {
        "episode_id": episode.get("episode_id"),
        "game": episode.get("game"),
        "seed": episode.get("seed"),
        "execution_order": episode.get("execution_order"),
        "sentinel": episode.get("sentinel") is True,
        "disposition": disposition,
        "completed": completed,
        "censored": censored,
        "censoring_reason": disposition if censored else None,
        "policy_entered": bool(policies),
        "policy_class": policy_detail.get("policy_class"),
        "factory": policy_detail.get("factory"),
        "denied_paths": deepcopy(policy_detail.get("denied_paths") or []),
        "action_limit": ACTION_LIMIT,
        "actions_observed": len(actions),
        "actions_to_progress": int(progress_detail.get("action_index")) if progress_event else None,
        "actions_to_progress_censored": bool(attempted and progress_event is None),
        "actions_to_progress_upper_bound": len(actions)
        if attempted and not progress_event
        else None,
        "elapsed_s_to_progress": float(progress_detail.get("elapsed_s"))
        if progress_event
        else None,
        "elapsed_s_to_progress_censored": bool(attempted and progress_event is None),
        "elapsed_s_to_progress_upper_bound": (
            max(
                [float((row.get("detail") or {}).get("elapsed_s") or 0.0) for row in events]
                or [0.0]
            )
            if attempted and not progress_event
            else None
        ),
        "elapsed_s_observed": max(
            [float((row.get("detail") or {}).get("elapsed_s") or 0.0) for row in events] or [0.0]
        ),
        "level_start": start_level,
        "level_end": levels[-1] if levels else 0,
        "level_progress": max(levels, default=0) - start_level,
        "completion_limit": MODEL_CALL_LIMIT,
        "max_new_tokens_per_call": MAX_NEW_TOKENS,
        "generated_token_limit": GENERATED_TOKEN_LIMIT,
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": len(generations),
        "generated_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in generation_details
        ),
        "raw_generation_receipts": generation_details,
        "tool_dispatch_result_rows": tool_rows,
        "resumed_policy_actions": deepcopy(terminal.get("resumed_policy_actions") or []),
        "induction_attempt_rows": inductions,
        "induction_attempt_count": len(inductions),
        "trajectory_supervisor": supervisor or None,
        "redirect_rows": redirects,
        "redirect_count": len(redirects),
        "supervisor_outcome": (
            "no_redirect_fired_unknown_demand" if not redirects else "redirects_observed"
        ),
        "costs": {
            "wall_s": float(terminal.get("elapsed_s") or 0.0),
            "cpu_time_s": terminal.get("cpu_time_s"),
            "gpu_time_s": terminal.get("gpu_time_s"),
            "gpu_time_measurement": terminal.get("gpu_time_measurement", "unavailable"),
            "environment_actions": len(actions),
            "generation_calls": len(requests),
            "generated_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in generation_details
            ),
        },
        "failure": terminal.get("error"),
        "historical_reference": deepcopy(dict(reference or {})),
        "historical_comparison_interpretation": "noncausal_comparator_only",
        "registry_levels_before_attempt": episode.get("registry_levels_before_attempt"),
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_credit": False,
        "credit_decision": "generalization_observation_no_new_solve_credit",
        "registry_duplicate_precheck": True,
        "reproduction_required_for_new_credit": True,
        "reproduction_performed": False,
        "durable_event_count": len(events),
        "durable_event_kinds": [row.get("kind") for row in events],
        "durable_events": events,
        "journal_file": journal_path,
        "journal_checksum": (journal or {}).get("journal_checksum"),
    }


def _row_event_errors(row: Mapping[str, Any]) -> list[str]:
    events = [dict(item) for item in row.get("durable_events", []) if isinstance(item, Mapping)]
    counts = Counter(str(item.get("kind")) for item in events)
    errors: list[str] = []
    if int(row.get("durable_event_count") or 0) != len(events):
        errors.append(f"{row.get('episode_id')}:durable_event_count")
    if int(row.get("generation_calls_attempted") or 0) != counts["request_attempted"]:
        errors.append(f"{row.get('episode_id')}:generation_calls_attempted")
    if int(row.get("generation_calls_completed") or 0) != counts["generation_completed"]:
        errors.append(f"{row.get('episode_id')}:generation_calls_completed")
    action_count = sum(
        item.get("kind") == "environment_action_completed"
        and (item.get("detail") or {}).get("action") not in {None, "RESET"}
        for item in events
    )
    if int(row.get("actions_observed") or 0) != action_count:
        errors.append(f"{row.get('episode_id')}:actions_observed")
    if int(row.get("induction_attempt_count") or 0) != counts["induction_attempt"]:
        errors.append(f"{row.get('episode_id')}:induction_attempt_count")
    return errors


def reduce_durable_panel(
    schedule: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    historical_references: Mapping[str, Mapping[str, Any]],
    *,
    sentinel_attempted: bool = False,
) -> JsonDict:
    """Reduce all six sealed identities directly from hash-chained journals."""

    journals: dict[str, JsonDict | None] = {}
    integrity_failures: list[str] = []
    for episode in schedule:
        episode_id = str(episode.get("episode_id"))
        try:
            journals[episode_id] = read_episode_events(checkpoint_dir, episode)
        except CheckpointIntegrityError as error:
            journals[episode_id] = None
            integrity_failures.append(f"{episode_id}:{error}")
    sentinel = dict(schedule[0]) if schedule else {}
    sentinel_journal = journals.get(str(sentinel.get("episode_id")))
    sentinel_passed = _sentinel_passed(sentinel_journal)
    sentinel_started = bool(sentinel_journal) or sentinel_attempted
    sentinel_failed = sentinel_started and not sentinel_passed
    rows = [
        _episode_row(
            episode,
            journals.get(str(episode.get("episode_id"))),
            historical_references.get(str(episode.get("game"))),
            sentinel_failed=bool(index == 0 and sentinel_failed),
            sentinel_attempted=sentinel_attempted,
        )
        for index, episode in enumerate(schedule)
    ]
    accounting_failures = list(integrity_failures)
    expected_ids = [str(row.get("episode_id")) for row in schedule]
    if len(schedule) != 6 or len(set(expected_ids)) != 6:
        accounting_failures.append("six_episode_schedule_identity")
    if sentinel_failed and any(row["disposition"] != "unstarted" for row in rows[1:]):
        accounting_failures.append("later_episode_started_after_failed_sentinel")
    dispositions = Counter(str(row["disposition"]) for row in rows)
    completed = sum(bool(row["completed"]) for row in rows)
    censored = sum(bool(row["censored"]) for row in rows)
    unstarted = dispositions["unstarted"]
    attempted = len(rows) - unstarted
    if len(rows) != completed + censored + unstarted:
        accounting_failures.append("planned_disposition_equality")
    if attempted != completed + censored:
        accounting_failures.append("attempted_disposition_equality")
    authenticity_failures: list[JsonDict] = []
    for episode, row in zip(schedule, rows, strict=True):
        if row["disposition"] == "unstarted":
            continue
        checks = {
            "identity": all(
                row.get(key) == episode.get(key) for key in ("episode_id", "game", "seed")
            ),
            "factory": row.get("factory") == "make_carnot_agent",
            "policy": row.get("policy_class") == "E3AgentPolicy",
            "path_denial": set(WITHHELD_INPUTS) <= set(row.get("denied_paths") or []),
            "action_budget": int(row.get("actions_observed") or 0) <= ACTION_LIMIT,
            "call_budget": int(row.get("generation_calls_attempted") or 0) <= MODEL_CALL_LIMIT,
            "token_budget": int(row.get("generated_tokens") or 0) <= GENERATED_TOKEN_LIMIT,
            "episode_time_budget": float(row.get("elapsed_s_observed") or 0.0)
            <= EPISODE_WORK_LIMIT_S + 1.0,
            "event_equalities": not _row_event_errors(row),
        }
        for check, passed in checks.items():
            if not passed:
                authenticity_failures.append(
                    {"episode_id": row.get("episode_id"), "check": check, "observed": False}
                )
    capture = int(
        len(rows) == 6
        and not accounting_failures
        and not authenticity_failures
        and (sentinel_passed or sentinel_failed)
    )
    return {
        "arc_generalization_capture_complete_score": capture,
        "planned_units": len(schedule),
        "attempted_units": attempted,
        "completed_units": completed,
        "censored_units": censored,
        "unstarted_units": unstarted,
        "sentinel_attempted": sentinel_started,
        "sentinel_passed": sentinel_passed,
        "sentinel_failed": sentinel_failed,
        "progress_episode_count": sum(int(row.get("level_progress") or 0) > 0 for row in rows),
        "redirect_firing_count": sum(int(row.get("redirect_count") or 0) for row in rows),
        "current_induction_count": sum(
            int(row.get("induction_attempt_count") or 0) for row in rows
        ),
        "accounting_failures": accounting_failures,
        "authenticity_failures": authenticity_failures,
        "per_game_results": rows,
        "causal_interpretation": "descriptive_association_only",
    }


def build_cumulative_induction_ledger(
    historical: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Retain authentic history and append only observed distinct current attempts."""

    historical_rows = [
        deepcopy(dict(row)) for row in historical.get("rows", []) if isinstance(row, Mapping)
    ]
    historical_authenticated = bool(
        historical.get("historical_ledger_authenticated") is True
        and all(
            HASH_RE.fullmatch(str(row.get("induction_id") or ""))
            and row.get("source_authenticated") is True
            and row.get("engaged") is True
            for row in historical_rows
        )
    )
    historical_count = len(historical_rows) if historical_authenticated else 0
    retained = deepcopy(historical_rows) if historical_authenticated else []
    identities = {str(row["induction_id"]) for row in retained}
    current_valid: list[JsonDict] = []
    censored = 0
    observed_attempt_count = 0
    for episode in episodes:
        raw_attempts = list(episode.get("induction_attempt_rows", []))
        observed_attempt_count += len(raw_attempts)
        derived, episode_censored = induction_base._current_induction_rows(
            {
                "episode_id": episode.get("episode_id"),
                "induction_rows": raw_attempts,
            }
        )
        censored += episode_censored
        for raw in derived:
            row = deepcopy(dict(raw))
            row.setdefault("source_session_id", episode.get("episode_id"))
            current_valid.append(row)
    duplicates = 0
    new_count = 0
    for row in current_valid:
        identity = str(row["induction_id"])
        if identity in identities:
            duplicates += 1
            continue
        identities.add(identity)
        retained.append(row)
        new_count += 1
    return {
        "schema": "carnot.exp7406.cumulative_induction_ledger.v1",
        "historical_ledger_authenticated": historical_authenticated,
        "historical_authenticated_count": historical_count,
        "current_authenticated_count": len(current_valid),
        "current_observed_attempt_count": observed_attempt_count,
        "current_new_count": new_count,
        "current_duplicate_count": duplicates,
        "current_censored_count": censored,
        "cumulative_total": len(retained),
        "no_firing_interpretation": "unknown_tool_demand_not_absent_demand",
        "rows": retained,
    }


def load_historical_references(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    """Project noncausal counts without retaining any historical solution action."""

    references: dict[str, JsonDict] = {}
    for game, path in paths.items():
        value = load_object(path)
        solution = value.get("solution")
        action_count = len(solution) if isinstance(solution, list) else value.get("moves")
        references[str(game)] = {
            "game": str(game),
            "comparator_class": "hand_tuned_historical_noncausal",
            "causal_use": False,
            "actions": action_count,
            "levels": value.get("reproduced_levels", value.get("reached_level")),
            "solve_provenance": value.get("solve_provenance", "development_proxy"),
            "mode": value.get("mode"),
            "source_path": str(path),
            "source_sha256": sha256_file(path) if path.is_file() else None,
        }
    return references


def affected_manifest() -> validation_contract.AffectedManifest:
    """Name the exact current test, module, and thin entrypoint."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7358 plan through the Exp7303 runner."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any expansion or drift from the frozen affected manifest."""

    return validation_contract.validate_command_plan(root, affected_manifest(), commands)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Build E2E-009, E2E-010, and the private LLM-off environment smoke."""

    private.mkdir(parents=True, exist_ok=True)
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    return [
        validation_scope.CommandSpec(
            "e2e_009",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e009'}",
                "tests/python/test_arc_induction_state_persistence.py",
                "-q",
            ),
            "E2E-009 cross-call ARC induction memory",
        ),
        validation_scope.CommandSpec(
            "e2e_010",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={private / 'e2e010'}",
                "tests/python/test_arc_tool_grammar_transport.py",
                "-q",
            ),
            "E2E-010 local grammar tool transport",
        ),
        validation_scope.CommandSpec(
            "e2e_offline_smoke",
            (
                "/usr/bin/env",
                "CARNOT_ARC_DISABLE_INDUCTION=1",
                python,
                "-u",
                "scripts/arc_loop_solve.py",
                "--mechanism",
                "e3",
                "--game",
                "r11l",
                "--max-actions",
                "12",
                "--output",
                str(private / "r11l-twelve-action-smoke.json"),
            ),
            "private LLM-off environment smoke",
        ),
    ]


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build the cold reducer and unchanged strict terminal readers."""

    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                "-c",
                (
                    "import json,sys; "
                    "from carnot.experiment_7406_v649_arc_generalization import independent_reduce_file; "
                    "r=independent_reduce_file(sys.argv[1]); print(json.dumps(r,sort_keys=True)); "
                    "raise SystemExit(0 if r['matches_declared'] else 1)"
                ),
                str(candidate),
            ),
            "cold artifact and embedded-event reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "unchanged adversarial verifier",
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
            "strict verdict-row consistency",
        ),
    ]


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
    """Explain ordinary fields without changing their executable values."""

    specific = {
        "schema": "A versioned schema lets strict readers reject incompatible records.",
        "run_date": "The fixed execution date retains actual UTC start and end timestamps.",
        "preconditions_checked": "Exact input hashes and eligibility checks precede dependent work.",
        "MODEL_SPECS": "The current live work names only the mandated Qwen model.",
        "model_invoked": "Any attempted current load makes this true, even after failure.",
        "invocation_counts": "Current attempted and terminal calls come only from durable receipts.",
        "inference_substrate": "This string describes the actual current owned model work.",
        "inference_substrate_details": "Device and software details remain outside the substrate string.",
        "inference_substrate_class": "The class follows actual bounded generation or load-only work.",
        "execution_venue": "The closed host value does not contain device details.",
        "duration_s": "Measured monotonic duration is never padded.",
        "phase_spans": "Measured boundaries expose validation, load, generation, and write costs.",
        "random_seed": "The experiment and two episode seeds were fixed before outcomes.",
        "reproducibility_checksum": "The checksum binds configuration, inputs, code, and raw rows.",
        "source_artifact_hashes": "Exact byte hashes retain source and producer identity.",
        "rows": "All six sealed units retain metrics, costs, and disposition.",
        "sample_size_budget": "Planned, attempted, complete, censored, and unstarted counts stay distinct.",
        "acceptance_gate_results": "Gate category, operator, expected, observed, and pass state stay explicit.",
        "gate_check_summary": "Every failure retains its upstream path, field, and missing value.",
        "verifier_is_oracle": "The deployed environment shares the correctness oracle.",
        "honest_verdict": "Complete and blocked prefixes distinguish findings from absent inputs.",
        "verdict_class": "The closed class prevents efficacy from leaking through accounting.",
        "flagged_adversarial": "A critical terminal-reader finding prevents readiness.",
        "validation_receipts": "Exact commands, environments, exits, durations, and log hashes support audit.",
        "repository_health": "Unrelated repository health stays outside affected validation.",
        "field_principles": "This map explains every ordinary top-level field.",
        "promotion_score": "Zero forbids rollout, publication, or weight changes.",
        "arc_generalization_capture_complete_score": "One is complete durable accounting, not efficacy.",
        "per_game_results": "Each game and seed reports progress censoring and noncausal comparison.",
        "solve_provenance": "Only current live attempts use live-agent self-discovery provenance.",
        "owned_runtime_receipt": "Model, PID, lease, and CUDA offload facts bind the owned runtime.",
    }
    return {
        key: specific.get(key, f"The {key} field retains directly auditable experiment evidence.")
        for key in keys
    }


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    panel: Mapping[str, Any],
    cumulative_ledger: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    invocation_counts: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a null, disqualified, or bounded observation from raw evidence."""

    counts = {**ZERO_INVOCATION_COUNTS, **deepcopy(dict(invocation_counts))}
    model_invoked = int(counts.get("model_loads_attempted") or 0) > 0
    generated = int(counts.get("generation_calls_attempted") or 0) > 0
    affected_passed = _receipts_pass(validation_receipts, validation_scope.REQUIRED_CHECK_NAMES)
    e2e_passed = _receipts_pass(validation_receipts, REQUIRED_E2E_NAMES)
    terminal_passed = _receipts_pass(validation_receipts, REQUIRED_TERMINAL_NAMES)
    adversarial = next(
        (row for row in validation_receipts if row.get("name") == "adversarial_verify"), None
    )
    flagged = bool(adversarial is not None and adversarial.get("passed") is not True)
    gates = [
        gate_row(
            "preconditions",
            "precondition",
            True,
            bool(preconditions) and all(row.get("passed") is True for row in preconditions),
            upstream="preconditions_checked",
            artifact_field="all_preconditions_passed",
        ),
        gate_row(
            "durable_panel_accounting",
            "completion",
            1,
            int(panel.get("arc_generalization_capture_complete_score") or 0),
            upstream=RAW_PANEL_PATH.as_posix(),
            artifact_field="arc_generalization_capture_complete_score",
        ),
        gate_row(
            "owned_native_cuda_runtime",
            "safety",
            True,
            runtime_receipt.get("task_linked_cuda_execution") is True,
            upstream="owned_runtime_receipt",
            artifact_field="task_linked_cuda_execution",
        ),
        gate_row(
            "bounded_generation_attempted",
            "completion",
            True,
            int(counts.get("generation_calls_attempted") or 0) > 0,
            upstream="invocation_counts",
            artifact_field="generation_calls_attempted>0",
        ),
        gate_row(
            "historical_induction_boundary",
            "safety",
            True,
            cumulative_ledger.get("historical_ledger_authenticated") is True,
            upstream=HISTORICAL_INDUCTION_PATH.as_posix(),
            artifact_field="historical_ledger_authenticated",
        ),
        gate_row(
            "affected_validation",
            "required_validation",
            True,
            affected_passed,
            upstream="validation_receipts",
            artifact_field="Exp7358_Exp7303_required_checks",
        ),
        gate_row(
            "e2e_plumbing",
            "required_validation",
            True,
            e2e_passed,
            upstream="validation_receipts",
            artifact_field="E2E-009/E2E-010/offline_smoke",
        ),
        gate_row(
            "terminal_readers",
            "safety",
            True,
            terminal_passed,
            upstream="validation_receipts",
            artifact_field="independent/adversarial/strict",
        ),
        gate_row(
            "automatic_promotion",
            "promotion",
            0,
            0,
            upstream="protocol",
            artifact_field="promotion_score",
        ),
    ]
    required_ok = all(row["passed"] for row in gates[:-1]) and not flagged
    capture_score = int(required_ok)
    progress_count = int(panel.get("progress_episode_count") or 0)
    if required_ok:
        verdict_class = "null"
        honest_verdict = (
            "complete_null_bounded_generalization_observation"
            if progress_count
            else "complete_null_bounded_panel_no_progress"
        )
    else:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_required_evidence"
    rows = deepcopy(list(panel.get("per_game_results") or []))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": honest_verdict,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": list(MODEL_SPECS) if model_invoked else [],
        "resolved_model_specs": [deepcopy(dict(row)) for row in model_specs],
        "model_invoked": model_invoked,
        "invocation_counts": counts,
        "inference_substrate": (
            "owned_native_cuda_llama_cpp" if model_invoked else "no_current_model_invocation"
        ),
        "inference_substrate_details": {
            "device": runtime_receipt.get("gpu_name"),
            "gpu_uuid": runtime_receipt.get("gpu_uuid"),
            "software": runtime_receipt.get("runner", "native llama.cpp"),
            "quantization": QUANTIZATION,
            "context_tokens": 49_152,
            "output_tokens_per_call": MAX_NEW_TOKENS,
            "context_independent_of_output_limit": True,
        },
        "inference_substrate_class": (
            "model_bounded_generation"
            if generated
            else "model_load_no_generation"
            if model_invoked
            else "no_model_load"
        ),
        "inference_mode": (
            "live_gpu"
            if runtime_receipt.get("task_linked_cuda_execution") is True
            else "not_verified"
        ),
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "small_ebm_training": {"performed": False, "kind": "none"},
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "experiment": RANDOM_SEED,
            "episodes": list(EPISODE_SEEDS),
            "resampling": None,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": rows,
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": panel.get("attempted_units"),
            "completed_units": panel.get("completed_units"),
            "censored_units": panel.get("censored_units"),
            "unstarted_units": panel.get("unstarted_units"),
            "games": 3,
            "seeds_per_game": 2,
            "action_limit_per_episode": ACTION_LIMIT,
            "model_call_limit_per_episode": MODEL_CALL_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "episode_work_limit_s": EPISODE_WORK_LIMIT_S,
            "total_episode_work_limit_s": TOTAL_EPISODE_WORK_LIMIT_S,
            "stopping_rule": (
                "stop after six sealed dispositions, total episode cap, or a failed sentinel"
            ),
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_check_names": [
            *validation_scope.REQUIRED_CHECK_NAMES,
            *REQUIRED_E2E_NAMES,
            *REQUIRED_TERMINAL_NAMES,
        ],
        "repository_health": {
            "status": "not_assessed_by_scoped_experiment",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "promotion_score": 0,
        "arc_generalization_capture_complete_score": capture_score,
        "per_game_results": rows,
        "selection_receipt": deepcopy(dict(selection)),
        "cumulative_tool_demand_ledger": deepcopy(dict(cumulative_ledger)),
        "historical_induction_count": cumulative_ledger.get("historical_authenticated_count"),
        "current_induction_count": cumulative_ledger.get("current_authenticated_count"),
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_claimed": False,
        "reproduced_levels": [],
        "official_score": None,
        "owned_runtime_receipt": deepcopy(dict(runtime_receipt)),
        "raw_evidence_receipt": {
            "schedule_hash": canonical_hash(
                [row.get("episode_id") for row in panel.get("per_game_results", [])]
            ),
            "row_hash": canonical_hash(rows),
            "accounting_failures": deepcopy(panel.get("accounting_failures") or []),
            "authenticity_failures": deepcopy(panel.get("authenticity_failures") or []),
            "sentinel_passed": panel.get("sentinel_passed"),
        },
        "redirect_progress_interpretation": "descriptive_association_only",
        "hidden_game_capacity_claimed": False,
        "absent_tool_demand_claimed": False,
        "production_defaults_changed": False,
        "supervisor_ordering_changed": False,
        "generator_weights_changed": False,
        "solve_registry_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles([*artifact, "field_principles"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> JsonDict:
    """Publish an external precondition block without fake current work."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_external_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "resolved_model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_details": {},
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "small_ebm_training": {"performed": False, "kind": "none"},
        "phase_spans": [],
        "random_seed": {
            "experiment": RANDOM_SEED,
            "episodes": list(EPISODE_SEEDS),
            "resampling": None,
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 0,
            "unstarted_units": 6,
            "stopping_rule": "external gate failure stops before dependent work",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": gate_summary(preconditions),
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "required_check_names": [],
        "repository_health": {
            "status": "not_evaluated",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "promotion_score": 0,
        "arc_generalization_capture_complete_score": 0,
        "per_game_results": [],
        "selection_receipt": {},
        "cumulative_tool_demand_ledger": {},
        "historical_induction_count": 0,
        "current_induction_count": 0,
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_claimed": False,
        "reproduced_levels": [],
        "official_score": None,
        "owned_runtime_receipt": {},
        "raw_evidence_receipt": {},
        "redirect_progress_interpretation": "not_measured",
        "hidden_game_capacity_claimed": False,
        "absent_tool_demand_claimed": False,
        "production_defaults_changed": False,
        "supervisor_ordering_changed": False,
        "generator_weights_changed": False,
        "solve_registry_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
    }
    artifact["field_principles"] = _field_principles([*artifact, "field_principles"])
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, budgets, durable event counts, and safety fields."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (artifact.get("schema"), artifact.get("experiment_id"), artifact.get("milestone")) != (
        SCHEMA,
        EXPERIMENT_ID,
        MILESTONE,
    ):
        errors.append("identity")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    verdict = artifact.get("verdict_class")
    if verdict not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    blocked = verdict == "blocked"
    expected_prefix = "blocked_" if blocked else "complete_"
    if not str(artifact.get("honest_verdict") or "").startswith(expected_prefix):
        errors.append("honest_verdict")
    counts = artifact.get("invocation_counts")
    counts = dict(counts) if isinstance(counts, Mapping) else {}
    invoked = int(counts.get("model_loads_attempted") or 0) > 0
    generated = int(counts.get("generation_calls_attempted") or 0) > 0
    if artifact.get("model_invoked") is not invoked:
        errors.append("model_invoked")
    expected_specs = MODEL_SPECS if invoked else []
    if artifact.get("MODEL_SPECS") != expected_specs:
        errors.append("MODEL_SPECS")
    expected_class = (
        "blocked_no_run"
        if blocked
        else "model_bounded_generation"
        if generated
        else "model_load_no_generation"
        if invoked
        else "no_model_load"
    )
    if artifact.get("inference_substrate_class") != expected_class:
        errors.append("inference_substrate_class")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue")
    if generated and float(artifact.get("duration_s") or 0.0) < 10.0:
        errors.append("duration_s")
    if invoked and not generated and float(artifact.get("duration_s") or 0.0) < 2.0:
        errors.append("duration_s")
    budget = artifact.get("sample_size_budget")
    budget = dict(budget) if isinstance(budget, Mapping) else {}
    planned = int(budget.get("planned_units") or 0)
    completed = int(budget.get("completed_units") or 0)
    censored = int(budget.get("censored_units") or 0)
    unstarted = int(budget.get("unstarted_units") or 0)
    attempted = int(budget.get("attempted_units") or 0)
    if (
        planned != 6
        or planned != completed + censored + unstarted
        or attempted != completed + censored
    ):
        errors.append("sample_size_budget")
    rows = artifact.get("rows") or []
    if blocked:
        if rows or invoked or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
            errors.append("blocked_invocations_or_rows")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_check_summary")
    else:
        if len(rows) != 6:
            errors.append("rows")
        for row in rows:
            if isinstance(row, Mapping):
                errors.extend(_row_event_errors(row))
        if verdict == "null" and not _receipts_pass(
            artifact.get("validation_receipts") or [],
            (
                *validation_scope.REQUIRED_CHECK_NAMES,
                *REQUIRED_E2E_NAMES,
                *REQUIRED_TERMINAL_NAMES,
            ),
        ):
            errors.append("validation_receipts")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score")
    if artifact.get("new_solve_claimed") is not False:
        errors.append("new_solve_claimed")
    if artifact.get("official_score") is not None:
        errors.append("official_score")
    if artifact.get("supervisor_ordering_changed") is not False:
        errors.append("supervisor_ordering_changed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def independent_reduce_file(path: str | Path) -> JsonDict:
    """Cold-load the candidate and recompute accounting from embedded events."""

    artifact = load_object(Path(path))
    errors = validate_artifact(artifact)
    row_errors = [
        error
        for row in artifact.get("rows", [])
        if isinstance(row, Mapping)
        for error in _row_event_errors(row)
    ]
    budget = dict(artifact.get("sample_size_budget") or {})
    accounting_exact = int(budget.get("planned_units") or 0) == sum(
        int(budget.get(key) or 0)
        for key in ("completed_units", "censored_units", "unstarted_units")
    ) and int(budget.get("attempted_units") or 0) == sum(
        int(budget.get(key) or 0) for key in ("completed_units", "censored_units")
    )
    gates = artifact.get("acceptance_gate_results") or []
    reduced = int(
        not errors
        and not row_errors
        and accounting_exact
        and all(row.get("passed") is True for row in gates)
    )
    declared = int(artifact.get("arc_generalization_capture_complete_score") or 0)
    return {
        "declared_arc_generalization_capture_complete_score": declared,
        "reduced_arc_generalization_capture_complete_score": reduced,
        "accounting_exact": accounting_exact,
        "row_event_errors": row_errors,
        "validation_errors": errors,
        "matches_declared": declared == reduced,
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
        record.update(
            {
                "producer_experiment_id": value.get("experiment_id"),
                "producer_status": value.get("status"),
                "producer_verdict_class": value.get("verdict_class"),
                "producer_flagged_adversarial": value.get("flagged_adversarial"),
            }
        )
    return record


def _manifest_rejects(value: Any, experiment_id: str) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("experiment_id")) in {experiment_id, "7406"}:
            return True
        return any(_manifest_rejects(item, experiment_id) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_rejects(item, experiment_id) for item in value)
    return False


def _historical_induction_payload(root: Path) -> JsonDict:
    upstream = load_object(root / EXP7305_PATH)
    historical = load_object(root / HISTORICAL_INDUCTION_PATH)
    ledger_ref = upstream.get("cumulative_induction_ledger")
    ledger_ref = dict(ledger_ref) if isinstance(ledger_ref, Mapping) else {}
    rows = [dict(row) for row in historical.get("rows", []) if isinstance(row, Mapping)]
    authenticated = bool(
        upstream.get("status") == "complete"
        and upstream.get("verdict_class") not in {"blocked", "partial", "disqualified"}
        and upstream.get("flagged_adversarial") is not True
        and isinstance(upstream.get("reproducibility_checksum"), str)
        and upstream.get("reproducibility_checksum")
        == induction_base.prior_artifact_checksum(upstream)
        and ledger_ref.get("sha256")
        == (
            sha256_file(root / HISTORICAL_INDUCTION_PATH)
            if (root / HISTORICAL_INDUCTION_PATH).is_file()
            else None
        )
        and len(rows) == 10
        and ledger_ref.get("authenticated_unique_inductions") == 10
        and historical.get("authenticated_unique_inductions") == 10
    )
    return {
        "historical_ledger_authenticated": authenticated,
        "rows": rows,
        "source_path": HISTORICAL_INDUCTION_PATH.as_posix(),
    }


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict, dict[str, JsonDict], JsonDict]:
    """Authenticate all static sources and the structured Exp7398 gate."""

    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_row(
                f"required_input:{relative.as_posix()}",
                "precondition",
                True,
                available,
                upstream=relative.as_posix(),
                artifact_field="readable_nonempty_bytes",
            )
        )
        if available:
            role = (
                "checkpoint_dependency"
                if relative == EXP7398_PATH
                else "historical_induction_evidence"
                if relative in {EXP7305_PATH, HISTORICAL_INDUCTION_PATH}
                else "historical_noncausal_comparator"
                if relative in HISTORICAL_REFERENCE_PATHS.values()
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
            "### REQ-ARC-WMTE-7406:" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7406",
        )
    )
    upstream = load_object(root / EXP7398_PATH)
    dependency = check_checkpoint_dependency(
        upstream,
        {path.as_posix(): root / path for path in EXP7398_CODE_PATHS},
    )
    checks.extend(dependency["checks"])
    manifest_parse_ok = True
    try:
        manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        manifest = {}
        manifest_parse_ok = False
    checks.append(
        gate_row(
            "exclusion_manifest_parse",
            "precondition",
            True,
            manifest_parse_ok,
            upstream=EXCLUSION_PATH.as_posix(),
            artifact_field="valid_yaml",
        )
    )
    checks.append(
        gate_row(
            "current_task_not_quarantined",
            "precondition",
            False,
            _manifest_rejects(manifest, EXPERIMENT_ID),
            upstream=EXCLUSION_PATH.as_posix(),
            artifact_field=EXPERIMENT_ID,
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
        )
    )
    registry_parse_ok = True
    try:
        registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        registry_value = {}
        registry_parse_ok = False
    checks.append(
        gate_row(
            "solve_registry_parse",
            "precondition",
            True,
            registry_parse_ok,
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="valid_yaml",
        )
    )
    references = load_historical_references(
        {game: root / path for game, path in HISTORICAL_REFERENCE_PATHS.items()}
    )
    historical = _historical_induction_payload(root)
    checks.append(
        gate_row(
            "historical_induction_ledger",
            "precondition",
            True,
            historical["historical_ledger_authenticated"],
            upstream=HISTORICAL_INDUCTION_PATH.as_posix(),
            artifact_field="historical_ledger_authenticated",
        )
    )
    return checks, hashes, dict(registry_value), references, historical


def runtime_preconditions(
    root: Path, *, gpu_wait_s: float, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover - hardware integration.
    """Reuse the shipped current-model, tokenizer, CUDA, and lease preflight."""

    checks, hashes, resources = live_base.runtime_preconditions(
        root, gpu_wait_s=gpu_wait_s, started=started
    )
    normalized = [
        {
            **dict(row),
            "category": row.get("category", "precondition"),
            "operator": row.get("operator", "=="),
        }
        for row in checks
    ]
    return normalized, hashes, resources


@contextmanager
def _configured_runtime() -> Any:  # pragma: no cover - live process integration.
    """Point the shipped v647 runtime at the v649 paths and fixed bounds."""

    updates = {
        "RUN_DATE": RUN_DATE,
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "RANDOM_SEED": RANDOM_SEED,
        "EPISODE_SEEDS": EPISODE_SEEDS,
        "ACTION_LIMIT": ACTION_LIMIT,
        "MODEL_CALL_LIMIT": MODEL_CALL_LIMIT,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "EPISODE_WORK_LIMIT_S": TOTAL_EPISODE_WORK_LIMIT_S,
        "MODEL_LOAD_LIMIT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "SCHEDULE_PATH": SCHEDULE_PATH,
        "SESSION_PATH": SESSION_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "RAW_PANEL_PATH": RAW_PANEL_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "session_environment": session_environment,
    }
    old = {name: getattr(live_base, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(live_base, name, value)
        with live_base._configured_runtime() as reused:
            yield reused
    finally:
        for name, value in old.items():
            setattr(live_base, name, value)


class EpisodeWorkTimeout(Exception):
    """Stop one episode at its declared wall-time boundary."""


class SentinelAbort(BaseException):
    """Stop the child before any later episode starts after sentinel failure."""


class _LiveState:  # pragma: no cover - live process integration.
    def __init__(self, schedule: Sequence[Mapping[str, Any]], raw_dir: Path) -> None:
        self.schedule = {str(row["episode_id"]): dict(row) for row in schedule}
        self.raw_dir = raw_dir
        self.checkpoint_dir = raw_dir / "episode_checkpoints"
        self.started = time.monotonic()
        self.active_episode_id: str | None = None
        self.last_move: tuple[Any, Any] = (None, None)
        self.abort_after_sentinel = False
        self.timed_out: set[str] = set()
        self.cpu_started: dict[str, float] = {}
        self.episode_started: dict[str, float] = {}

    def episode(self, episode_id: str | None = None) -> JsonDict:
        identity = episode_id or self.active_episode_id or ""
        return self.schedule[identity]

    def emit(self, kind: str, detail: Mapping[str, Any]) -> None:
        episode = self.episode()
        episode_started = self.episode_started.get(str(episode["episode_id"]), self.started)
        append_episode_event(
            self.checkpoint_dir,
            episode,
            kind,
            {**dict(detail), "elapsed_s": round(time.monotonic() - episode_started, 6)},
        )

    def elapsed(self, episode_id: str) -> float:
        """Return monotonic work time for one active episode."""

        return round(time.monotonic() - self.episode_started.get(episode_id, self.started), 6)

    def sentinel_passed(self) -> bool:
        first = next(iter(self.schedule.values()))
        return _sentinel_passed(read_episode_events(self.checkpoint_dir, first))


_LIVE_STATE: _LiveState | None = None


class _DurableToolSink(list[JsonDict]):  # pragma: no cover - live process integration.
    """Append each observed tool dispatch before the child can lose its summary."""

    def __init__(self, path: Path, state: _LiveState) -> None:
        super().__init__()
        self.path = path
        self.state = state

    def append(self, value: JsonDict) -> None:
        row = {"episode_id": self.state.active_episode_id, **deepcopy(dict(value))}
        super().append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, _canonical_bytes(row) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_jsonl(path: Path) -> list[JsonDict]:  # pragma: no cover - live evidence I/O.
    rows: list[JsonDict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _raw_generation(request: Mapping[str, Any]) -> str | None:  # pragma: no cover
    path_text = request.get("response_path")
    if not path_text:
        return None
    value = load_object(Path(str(path_text)))
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    choice = dict(choices[0])
    message = choice.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(choice.get("text") or "")


def _postprocess_live_session(
    state: _LiveState, session_path: Path, tool_path: Path
) -> JsonDict:  # pragma: no cover - live process integration.
    session = load_object(session_path)
    episodes = [dict(row) for row in session.get("episodes", []) if isinstance(row, Mapping)]
    by_id = {str(row.get("episode_id")): row for row in episodes}
    tool_rows = _read_jsonl(tool_path)
    for episode_id, row in by_id.items():
        episode = state.schedule.get(episode_id)
        if episode is None:
            continue
        journal = read_episode_events(state.checkpoint_dir, episode)
        existing_generation_indices = {
            int(event.get("detail", {}).get("call_index") or 0)
            for event in _events(journal, "generation_completed")
        }
        for request in row.get("raw_request_manifest", []):
            if not isinstance(request, Mapping) or request.get("transport_completed") is not True:
                continue
            call_index = int(request.get("call_index") or 0)
            if call_index in existing_generation_indices:
                continue
            append_episode_event(
                state.checkpoint_dir,
                episode,
                "generation_completed",
                {
                    **dict(request),
                    "raw_generation": _raw_generation(request),
                    "elapsed_s": state.elapsed(episode_id),
                },
            )
        existing_inductions = len(
            _events(read_episode_events(state.checkpoint_dir, episode), "induction_attempt")
        )
        attempts = [
            dict(item) for item in row.get("induction_rows", []) if isinstance(item, Mapping)
        ]
        current_tools = [item for item in tool_rows if item.get("episode_id") == episode_id]
        for index, attempt in enumerate(attempts[existing_inductions:], start=existing_inductions):
            payload = deepcopy(attempt)
            payload.setdefault("attempt_index", index)
            payload["tool_event_rows"] = deepcopy(
                [
                    item
                    for item in current_tools
                    if item.get("attempt_index", item.get("induction_index")) in {None, index}
                ]
            )
            append_episode_event(
                state.checkpoint_dir,
                episode,
                "induction_attempt",
                {**payload, "elapsed_s": state.elapsed(episode_id)},
            )
        durable_attempts = [
            dict(item.get("detail") or {})
            for item in _events(
                read_episode_events(state.checkpoint_dir, episode), "induction_attempt"
            )
        ]
        journal = read_episode_events(state.checkpoint_dir, episode)
        sentinel_failed = bool(episode.get("sentinel") and not _sentinel_passed(journal))
        if sentinel_failed:
            row["disposition"] = "censored_sentinel"
            row["censored"] = True
            row["censoring_reason"] = "sentinel_missing_policy_request_or_action"
        elif episode_id in state.timed_out:
            row["disposition"] = "censored_timeout"
            row["censored"] = True
            row["censoring_reason"] = "episode_work_limit"
        terminal_detail = {
            "disposition": row.get("disposition", "complete_error"),
            "error": row.get("error"),
            "cpu_time_s": round(
                max(
                    0.0,
                    time.process_time() - state.cpu_started.get(episode_id, time.process_time()),
                ),
                6,
            ),
            "gpu_time_s": None,
            "gpu_time_measurement": "unavailable_native_runtime_did_not_export_per_episode_gpu_time",
            "trajectory_supervisor": deepcopy(row.get("trajectory_supervisor")),
            "tool_dispatch_result_rows": project_tool_dispatch_results(durable_attempts),
            "resumed_policy_actions": deepcopy(row.get("policy_consumption_rows") or []),
            "elapsed_s": state.elapsed(episode_id),
        }
        if not _events(read_episode_events(state.checkpoint_dir, episode), "episode_terminal"):
            append_episode_event(state.checkpoint_dir, episode, "episode_terminal", terminal_detail)
    session["episodes"] = list(by_id.values())
    atomic_json(session_path, session)
    return session


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live CUDA child.
    """Instrument the reused scored child with per-episode durable events."""

    global _LIVE_STATE
    schedule_payload = load_object(Path(args.schedule_path))
    schedule = [dict(row) for row in schedule_payload.get("rows", []) if isinstance(row, Mapping)]
    state = _LiveState(schedule, Path(args.raw_dir))
    _LIVE_STATE = state
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import arc_leaderboard_eval
    from carnot import experiment_7234_v637_arc_scored_dryrun as scored
    from carnot import experiment_7263_v639_arc_live as live_runner
    from carnot.agentic import arc_induction_tool_loop as tool_loop

    original_build = scored.build_disposable_submitted_policy
    original_open = live_runner.RequestCapture._open
    original_step = arc_leaderboard_eval.ProgressWriter.step
    original_run_game = arc_leaderboard_eval.run_game
    original_project = live_runner._episode_result
    original_induce = tool_loop.induce_with_tool_loop
    tool_sink = _DurableToolSink(Path(args.raw_dir) / TOOL_EVENT_PATH.name, state)

    def build_policy(game: str, proposer: Any) -> tuple[Any, JsonDict]:
        policy, factory = original_build(game, proposer)
        episode_id = os.environ.get("CARNOT_7406_EPISODE_ID", "")
        state.active_episode_id = episode_id
        state.episode_started[episode_id] = time.monotonic()
        state.cpu_started[episode_id] = time.process_time()
        state.emit(
            "policy_entered",
            {
                "factory": factory.get("factory"),
                "policy_class": type(policy).__name__,
                "denied_paths": list(WITHHELD_INPUTS),
                "factory_receipt": factory,
            },
        )
        original_next = policy.next_move

        def recorded_next(frames: Any, latest: Any) -> tuple[Any, Any]:
            move = original_next(frames, latest)
            state.last_move = move
            state.emit(
                "policy_action_requested",
                {"action": move[0], "data": move[1]},
            )
            return move

        policy.next_move = recorded_next
        return policy, factory

    def captured_open(capture: Any, request: Any, *call_args: Any, **kwargs: Any) -> Any:
        import urllib.request

        if isinstance(request, urllib.request.Request) and str(request.full_url).endswith(
            ("/completion", "/v1/chat/completions", "/v1/completions")
        ):
            body = bytes(request.data or b"")
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {}
            state.emit(
                "request_attempted",
                {
                    "call_index": len(capture.episode_rows(capture.episode_id)),
                    "request_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                    "requested_max_tokens": payload.get("max_tokens", payload.get("n_predict")),
                },
            )
        return original_open(capture, request, *call_args, **kwargs)

    def progress_step(writer: Any, **kwargs: Any) -> None:
        original_step(writer, **kwargs)
        action, data = state.last_move
        if action is not None:
            state.emit(
                "environment_action_completed",
                {
                    "action_index": kwargs.get("actions"),
                    "action": action,
                    "data": data,
                    "level": kwargs.get("level"),
                    "start_level": kwargs.get("start_level"),
                    "loop_index": kwargs.get("loop_index"),
                },
            )
            state.last_move = (None, None)

    def bounded_run_game(*call_args: Any, **kwargs: Any) -> JsonDict:
        def alarm_handler(_signum: int, _frame: Any) -> None:
            raise EpisodeWorkTimeout(f"episode exceeded {EPISODE_WORK_LIMIT_S}s")

        previous = signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, EPISODE_WORK_LIMIT_S)
        try:
            return original_run_game(*call_args, **kwargs)
        except EpisodeWorkTimeout:
            if state.active_episode_id:
                state.timed_out.add(state.active_episode_id)
            if (
                state.active_episode_id == next(iter(state.schedule))
                and not state.sentinel_passed()
            ):
                state.abort_after_sentinel = True
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    def project_episode(*call_args: Any, **kwargs: Any) -> JsonDict:
        row = original_project(*call_args, **kwargs)
        policy = kwargs.get("policy")
        if policy is not None and hasattr(policy, "trajectory_supervisor_diagnostics"):
            row["trajectory_supervisor"] = deepcopy(policy.trajectory_supervisor_diagnostics())
        first_id = next(iter(state.schedule))
        if str(row.get("episode_id")) == first_id and not state.sentinel_passed():
            row.update(
                {
                    "disposition": "censored_sentinel",
                    "censored": True,
                    "censoring_reason": "sentinel_missing_policy_request_or_action",
                }
            )
            state.abort_after_sentinel = True
        return row

    def instrumented_induce(*call_args: Any, **kwargs: Any) -> Any:
        kwargs["tool_event_sink"] = tool_sink
        return original_induce(*call_args, **kwargs)

    scored.build_disposable_submitted_policy = build_policy
    live_runner.RequestCapture._open = captured_open
    arc_leaderboard_eval.ProgressWriter.step = progress_step
    arc_leaderboard_eval.run_game = bounded_run_game
    live_runner._episode_result = project_episode
    tool_loop.induce_with_tool_loop = instrumented_induce
    try:
        try:
            with _configured_runtime() as reused:
                reused.run_live_session(args)
        except SentinelAbort:
            pass
        _postprocess_live_session(state, Path(args.session_path), tool_sink.path)
        return 0
    finally:
        tool_loop.induce_with_tool_loop = original_induce
        live_runner._episode_result = original_project
        arc_leaderboard_eval.run_game = original_run_game
        arc_leaderboard_eval.ProgressWriter.step = original_step
        live_runner.RequestCapture._open = original_open
        scored.build_disposable_submitted_policy = original_build
        _LIVE_STATE = None


def _invocation_counts(session: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce owned current calls from durable per-episode events."""

    attempted = sum(int(row.get("generation_calls_attempted") or 0) for row in rows)
    completed = sum(int(row.get("generation_calls_completed") or 0) for row in rows)
    requests = [
        dict(request)
        for row in rows
        for request in row.get("raw_generation_receipts", row.get("raw_request_manifest", []))
        if isinstance(request, Mapping)
    ]
    timed_out = bool(session.get("timed_out"))
    loaded = bool(session.get("model_loaded"))
    incomplete = max(0, attempted - completed)
    cancelled = incomplete if timed_out or any(row.get("censored") is True for row in rows) else 0
    failed = max(0, incomplete - cancelled)
    return {
        "model_loads_attempted": 1,
        "model_loads_completed": int(loaded),
        "model_loads_failed": int(not loaded and not timed_out),
        "model_loads_cancelled": int(not loaded and timed_out),
        "model_loads_in_flight": 0,
        "generation_calls_attempted": attempted,
        "generation_calls_completed": completed,
        "generation_calls_failed": failed,
        "generation_calls_cancelled": cancelled,
        "generation_calls_in_flight": 0,
        "usable_answers": sum(request.get("usable_answer") is True for request in requests),
    }


def _phase(name: str, phase_start: float, run_start: float, units: int) -> JsonDict:
    now = time.monotonic()
    return {
        "phase": name,
        "started_elapsed_s": round(phase_start - run_start, 6),
        "ended_elapsed_s": round(now - run_start, 6),
        "duration_s": round(now - phase_start, 6),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def _hash_raw_evidence(root: Path, hashes: JsonDict) -> None:
    for path in sorted(item for item in (root / RAW_DIR).rglob("*") if item.is_file()):
        if path == root / TERMINAL_CANDIDATE_PATH:
            continue
        hashes[path.relative_to(root).as_posix()] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "current_raw_evidence",
        }


def _write_raw_panel(
    path: Path, schedule: Sequence[Mapping[str, Any]], panel: Mapping[str, Any]
) -> None:
    atomic_json(
        path,
        {
            "schema": "carnot.exp7406.raw_panel.v1",
            "schedule": [deepcopy(dict(row)) for row in schedule],
            "per_game_results": deepcopy(list(panel.get("per_game_results") or [])),
            "accounting": {
                key: panel.get(key)
                for key in (
                    "planned_units",
                    "attempted_units",
                    "completed_units",
                    "censored_units",
                    "unstarted_units",
                )
            },
            "accounting_failures": deepcopy(panel.get("accounting_failures") or []),
            "authenticity_failures": deepcopy(panel.get("authenticity_failures") or []),
        },
    )


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover - CLI orchestration.
    """Run preflight, required checks, live panel, cold readers, and publication."""

    started = time.monotonic()
    started_at = utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=args.date)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before_static_checks")
    checks, hashes, registry, references, historical = collect_preconditions(REPO_ROOT)
    phases.append(_phase("read", phase_start, started, len(checks)))
    progress(
        started,
        "preconditions",
        "after_static_checks",
        passed=all(row["passed"] for row in checks),
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            preconditions=checks,
            source_hashes=hashes,
            started_at_utc=started_at,
            ended_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
        )
        errors = validate_artifact(artifact)
        if errors:
            raise RuntimeError(f"blocked artifact validation failed: {errors}")
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        progress(started, "write", "terminal_blocked_artifact_written", path=RESULT_PATH)
        return artifact

    from carnot.agentic.arc_game_adapters import adaptered_games

    phase_start = time.monotonic()
    progress(started, "selection", "before_label_blind_freeze")
    selection = freeze_panel(registry, adaptered_games=set(adaptered_games()))
    checks.append(
        gate_row(
            "label_blind_rotation",
            "precondition",
            True,
            selection.get("passed") is True,
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="three_eligible_games",
        )
    )
    schedule = build_schedule(selection.get("games") or [])
    for row in schedule:
        registry_row = next(item for item in selection["game_rows"] if item["game"] == row["game"])
        row["registry_levels_before_attempt"] = registry_row["registry_levels_before_attempt"]
    atomic_json(REPO_ROOT / SCHEDULE_PATH, {"selection_receipt": selection, "rows": schedule})
    phases.append(_phase("freeze", phase_start, started, len(schedule)))
    progress(started, "selection", "after_label_blind_freeze", games=selection.get("games"))
    if not selection.get("passed") or len(schedule) != 6:
        artifact = build_blocked_artifact(
            preconditions=checks,
            source_hashes=hashes,
            started_at_utc=started_at,
            ended_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    phase_start = time.monotonic()
    progress(started, "runtime_preconditions", "before_cache_runtime_device_checks")
    runtime_checks, runtime_hashes, resources = runtime_preconditions(
        REPO_ROOT, gpu_wait_s=args.gpu_wait_s, started=started
    )
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    phases.append(_phase("runtime_preconditions", phase_start, started, len(runtime_checks)))
    progress(
        started,
        "runtime_preconditions",
        "after_cache_runtime_device_checks",
        passed=all(row.get("passed") is True for row in checks),
    )
    if not all(row.get("passed") is True for row in checks):
        artifact = build_blocked_artifact(
            preconditions=checks,
            source_hashes=hashes,
            started_at_utc=started_at,
            ended_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
        )
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7406-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses")
    plan = build_validation_plan(REPO_ROOT, private / "scoped")
    plan_errors = validate_validation_plan(REPO_ROOT, plan)
    if plan_errors:
        raise RuntimeError(f"validation plan drift: {plan_errors}")
    scoped = validation_contract.run_categorized_commands(
        REPO_ROOT,
        [validation_contract.PlannedCommand(row, "required_validation", True) for row in plan],
        log_dir=REPO_ROOT / RAW_DIR / "validation/scoped",
        heartbeat_s=60.0,
    )
    progress(started, "validation", "after_affected_subprocesses", completed_units=len(scoped))
    progress(started, "e2e", "before_subprocesses")
    e2e = validation_scope.run_commands(
        REPO_ROOT,
        e2e_command_specs(REPO_ROOT, private / "e2e"),
        log_dir=REPO_ROOT / RAW_DIR / "validation/e2e",
        heartbeat_s=60.0,
    )
    progress(started, "e2e", "after_subprocesses", completed_units=len(e2e))
    receipts = [*scoped, *e2e]
    phases.append(_phase("validate_before_live", phase_start, started, len(receipts)))

    if not _receipts_pass(receipts, (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E_NAMES)):
        empty_panel = reduce_durable_panel(schedule, REPO_ROOT / EPISODE_CHECKPOINT_DIR, references)
        artifact = build_terminal_artifact(
            started_at_utc=started_at,
            ended_at_utc=utc_now(),
            duration_s=time.monotonic() - started,
            preconditions=checks,
            source_hashes=hashes,
            selection=selection,
            panel=empty_panel,
            cumulative_ledger=build_cumulative_induction_ledger(historical, []),
            model_specs=[],
            runtime_receipt={},
            invocation_counts=ZERO_INVOCATION_COUNTS,
            validation_receipts=receipts,
            phase_spans=phases,
        )
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        return artifact

    phase_start = time.monotonic()
    progress(
        started,
        "live_panel",
        "before_model_load_generation_benchmark",
        planned_units=6,
        total_episode_work_limit_s=TOTAL_EPISODE_WORK_LIMIT_S,
    )
    with _configured_runtime() as reused:
        session = reused.run_child_with_lease(
            resources=resources,
            schedule_path=REPO_ROOT / SCHEDULE_PATH,
            raw_dir=REPO_ROOT / RAW_DIR,
            checkpoint_path=REPO_ROOT / CHECKPOINT_PATH,
            session_path=REPO_ROOT / SESSION_PATH,
            remaining_s=TOTAL_EPISODE_WORK_LIMIT_S,
        )
    progress(
        started,
        "live_panel",
        "after_model_load_generation_benchmark",
        child_rows=len(session.get("episodes") or []),
    )
    phases.append(_phase("load_generate", phase_start, started, len(session.get("episodes") or [])))

    phase_start = time.monotonic()
    panel = reduce_durable_panel(
        schedule,
        REPO_ROOT / EPISODE_CHECKPOINT_DIR,
        references,
        sentinel_attempted=bool(session.get("model_loaded") or session.get("model_invoked")),
    )
    cumulative = build_cumulative_induction_ledger(historical, panel["per_game_results"])
    atomic_json(REPO_ROOT / CUMULATIVE_INDUCTION_PATH, cumulative)
    _write_raw_panel(REPO_ROOT / RAW_PANEL_PATH, schedule, panel)
    phases.append(_phase("evaluate", phase_start, started, len(panel["per_game_results"])))
    progress(
        started,
        "evaluation",
        "after_durable_reduction",
        capture=panel["arc_generalization_capture_complete_score"],
        progress_episodes=panel["progress_episode_count"],
    )
    _hash_raw_evidence(REPO_ROOT, hashes)
    counts = _invocation_counts(session, panel["per_game_results"])
    runtime = deepcopy(dict(session.get("runtime_receipt") or {}))
    runtime.update(
        {
            "authenticated_model": deepcopy(resources.get("model_spec")),
            "episode_ids": [row["episode_id"] for row in panel["per_game_results"]],
            "single_model_instance": True,
            "board_execution": False,
        }
    )
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        panel=panel,
        cumulative_ledger=cumulative,
        model_specs=[resources["model_spec"]],
        runtime_receipt=runtime,
        invocation_counts=counts,
        validation_receipts=receipts,
        phase_spans=phases,
    )
    atomic_json(REPO_ROOT / TERMINAL_CANDIDATE_PATH, candidate)

    phase_start = time.monotonic()
    progress(started, "terminal_validation", "before_cold_and_strict_subprocesses")
    terminal = validation_scope.run_commands(
        REPO_ROOT,
        terminal_command_specs(REPO_ROOT, REPO_ROOT / TERMINAL_CANDIDATE_PATH),
        log_dir=REPO_ROOT / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    phases.append(_phase("terminal_validation", phase_start, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_cold_and_strict_subprocesses",
        completed_units=len(terminal),
    )

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        panel=panel,
        cumulative_ledger=cumulative,
        model_specs=[resources["model_spec"]],
        runtime_receipt=runtime,
        invocation_counts=counts,
        validation_receipts=receipts,
        phase_spans=[*phases, _phase("write", time.monotonic(), started, 1)],
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    progress(started, "write", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_json(REPO_ROOT / TERMINAL_CANDIDATE_PATH, artifact)
    atomic_json(REPO_ROOT / RESULT_PATH, artifact)
    progress(
        started,
        "write",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        capture=artifact["arc_generalization_capture_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed date and private live-child arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, choices=[RUN_DATE])
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI handoff.
    """Run the requested host or private child role."""

    args = parse_args(argv)
    if args.role == "live-session":
        return run_live_session(args)
    artifact = run_experiment(args)
    return 0 if str(artifact.get("status", "")).startswith(("complete_", "blocked_")) else 1
