"""Capture six bounded live ARC supervisor outcomes without tuning the policy.

The live path reuses the shipped GPU lease, llama.cpp server, scored policy
factory, and ARC game loop. This module adds the frozen six-episode schedule,
outcome receipts, support reduction, scoped validation, and terminal artifact.

Spec refs: REQ-ARC-WMTE-7376 and SCENARIO-ARC-WMTE-7376-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7305_v642_arc_selfparse as live_base
from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
EXPERIMENT_ID = "exp7376-arc-outcomes"
SCHEMA = "carnot.experiment_7376.v647.arc_outcomes.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_SPECS = [MODEL_ID]
QUANTIZATION = "Q4_K_M"
TARGET_GAMES = ("bp35", "cn04", "dc22")
EPISODE_SEEDS = (7_376_202_609_17, 17_376_202_609_17)
RANDOM_SEED = 27_376_202_609_17
ACTION_LIMIT = 128
MODEL_CALL_LIMIT = 2
MAX_NEW_TOKENS = 256
GENERATED_TOKEN_LIMIT = MODEL_CALL_LIMIT * MAX_NEW_TOKENS
MODEL_LOAD_LIMIT_S = 600
EPISODE_WORK_LIMIT_S = 1800
CURATED_ARMS = (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)
WITHHELD_INPUTS = [
    "game_adapter",
    "banked_solution",
    "hand_solver",
    "saved_engine",
    "replay_route",
]

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
LEDGER_PATH = Path("ops/arc_supervisor_refinement_ledger.json")
EXP7358_PATH = Path("results/experiment_7358_v646_validation_contract.json")
EXP7365_PATH = Path("results/experiment_7365_v646_supervisor_support.json")
EXP7366_PATH = Path("results/experiment_7366_supervisor_live.json")
RESULT_PATH = Path("results/experiment_7376_v647_arc_outcomes.json")
RAW_DIR = Path("results/raw/experiment_7376_v647_arc_outcomes")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7376_v647_arc_outcomes.json")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "receipt_events.jsonl"
RAW_PANEL_PATH = RAW_DIR / "independent_reduction_input.json"
TERMINAL_CANDIDATE_PATH = RAW_DIR / "measured_terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7376_v647_arc_outcomes.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7376_v647_arc_outcomes.py")
TEST_PATH = Path("tests/python/test_experiment_7376_v647_arc_outcomes.py")

REQUIRED_E2E_NAMES = ("e2e_009", "e2e_010", "e2e_009_llm_off_environment")
REQUIRED_TERMINAL_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
REQUIRED_VALIDATION_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_NAMES = REQUIRED_TERMINAL_NAMES
ZERO_INVOCATION_COUNTS = deepcopy(live_base.ZERO_INVOCATION_COUNTS)
_PRIOR_SESSION_ENVIRONMENT = live_base.session_environment

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    REGISTRY_PATH,
    LEDGER_PATH,
    SPEC_PATH,
    Path("scripts/experiment_template.py"),
    Path("scripts/arc_loop_solve.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    Path("python/carnot/agentic/arc_supervisor_refinement.py"),
    Path("python/carnot/experiment_7365_v646_supervisor_support.py"),
    EXP7358_PATH,
    EXP7365_PATH,
    EXP7366_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with ordinary top-level experiment and milestone fields.",
    "status": "Publish a terminal state only after actual work and required validation.",
    "run_date": "Use 20260917 with actual UTC start and end timestamps.",
    "preconditions_checked": "Record exact paths, producer identity, hashes, classes, and resources before use.",
    "MODEL_SPECS": "Name only the mandated Qwen model used by current inference.",
    "model_invoked": "Set true for any attempted current load or generation, including failure.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight work.",
    "inference_substrate": "Describe actual task-owned native CUDA computation.",
    "inference_substrate_class": "Use the bounded-generation class and never pad measured time.",
    "execution_venue": "Record host orchestration and owned CUDA details in the runner receipt.",
    "duration_s": "Use measured monotonic duration without sleeps for a duration floor.",
    "phase_spans": "Retain measured read, build, load, generate, evaluate, validate, and write spans.",
    "random_seed": "Freeze experiment, episode, and reduction seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, protocol, sources, and raw rows.",
    "source_artifact_hashes": "Retain exact producer paths, byte hashes, verdicts, and flags.",
    "rows": "Keep every episode metric, cost, failure, and censoring disposition.",
    "sample_size_budget": "State planned, attempted, completed, censored units and fixed stopping rules.",
    "acceptance_gate_results": "Separate expected, observed, and passed values for each gate.",
    "gate_check_summary": "Name every failed check and keep the first exact expected and observed value.",
    "verifier_is_oracle": "Declare shared evaluator authority; observations do not establish causal effect.",
    "honest_verdict": "Use complete_ for finished work and blocked_ for unavailable prerequisites.",
    "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
    "flagged_adversarial": "Set true only for a critical independent finding.",
    "validation_receipts": "Keep command arguments, environment, scope, exit, duration, and log hash.",
    "repository_health": "Keep unrelated dated failures separate from required affected checks.",
    "field_principles": "Explain each ordinary field without wrapping its value.",
    "promotion_score": "Remain zero because observations do not authorize supervisor changes.",
    "arc_outcome_capture_complete_score": "Score complete disposition and evidence accounting, not efficacy.",
    "supervisor_support_ready_score": "Require the unchanged per-arm and leave-one-game-out support floor.",
    "per_game_results": "Keep all six action and model budgets, outcomes, failures, and censoring.",
    "solve_provenance": "Use live_agent_self_discovery for any incidental level observation.",
    "runner_receipt": "Bind the owned model, runtime, process, GPU, and lease to this run.",
    "generalization_activity": "Use adapter-withheld scored-policy episodes with curated arms unchanged.",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large model bytes do not enter memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Return one stable content identity for JSON-shaped evidence."""

    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind a terminal artifact while excluding only its checksum slot."""

    payload = deepcopy(dict(value))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one complete JSON document and replace the target atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> JsonDict | None:
    """Load an object or return none for missing and malformed external bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def utc_now() -> str:  # pragma: no cover - wall-clock integration boundary.
    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush a truthful phase boundary for the outer silence watchdog."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7376] phase={phase} event={event} "
        f"elapsed_s={max(0.0, time.monotonic() - started):.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def gate_check(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Record one exact comparison before dependent work uses its value."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


# Public spelling used by the spec-linked tests and artifact builders.
gate = gate_check


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every failed gate and identify the first without hiding later failures."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    return {
        "all_passed": not failed,
        "failed_count": len(failed),
        "first_failure": failed[0] if failed else None,
        "failed_checks": failed,
    }


def freeze_panel(
    registry: Mapping[str, Any], *, adaptered_games: set[str] | frozenset[str]
) -> JsonDict:
    """Freeze the three predeclared public proxies from registry metadata only."""

    indexed = {
        str(row.get("game")): dict(row)
        for row in registry.get("games", [])
        if isinstance(row, Mapping) and row.get("game")
    }
    rows: list[JsonDict] = []
    for rank, game in enumerate(TARGET_GAMES):
        source = indexed.get(game, {})
        rows.append(
            {
                "game": game,
                "eligible": bool(source) and game in adaptered_games,
                "selection_rank": rank,
                "mechanic_class": source.get("mechanic_class"),
                "registry_levels_before_attempt": int(source.get("levels_reproduced") or 0),
                "registry_reproducibility": source.get("reproducibility"),
                "adapter_available_but_withheld": game in adaptered_games,
                "adapter_disabled": True,
                "banked_solution_disabled": True,
                "hand_solver_disabled": True,
                "saved_engine_disabled": True,
                "replay_route_disabled": True,
            }
        )
    failures = [
        {
            "game": row["game"],
            "check": "registry_and_adapter_eligibility",
            "expected": True,
            "observed": row["eligible"],
        }
        for row in rows
        if not row["eligible"]
    ]
    return {
        "passed": len(rows) == 3 and all(row["eligible"] for row in rows),
        "games": [row["game"] for row in rows],
        "game_rows": rows,
        "selection_basis": "predeclared_exp7365_disjoint_evaluation_panel_prefix",
        "registry_prechecked": True,
        "selection_used_current_outcomes": False,
        "outcomes_seen_before_freeze": False,
        "game_source_read": False,
        "offline_ground_truth_search_used": False,
        "failures": failures,
    }


def build_schedule(games: Sequence[str], seeds: Sequence[int] = EPISODE_SEEDS) -> list[JsonDict]:
    """Create exactly two fresh bounded episodes for each frozen game."""

    rows: list[JsonDict] = []
    for game in games:
        for seed in seeds:
            rows.append(
                {
                    "episode_id": f"{game}:seed-{int(seed)}",
                    "game": str(game),
                    "seed": int(seed),
                    "arm": "curated_supervisor",
                    "execution_order": len(rows),
                    "action_limit": ACTION_LIMIT,
                    "completion_limit": MODEL_CALL_LIMIT,
                    "max_new_tokens_per_call": MAX_NEW_TOKENS,
                    "generated_token_limit": GENERATED_TOKEN_LIMIT,
                    "fresh_store": True,
                    "withheld_inputs": list(WITHHELD_INPUTS),
                    "adapter_disabled": True,
                    "banked_solution_disabled": True,
                    "hand_solver_disabled": True,
                    "saved_engine_disabled": True,
                    "replay_route_disabled": True,
                }
            )
    return rows


def _episode_seed(episode_dir: Path) -> int:
    marker = "seed-"
    text = episode_dir.name
    return int(text.split(marker, 1)[1]) if marker in text else EPISODE_SEEDS[0]


def session_environment(
    base_env: Mapping[str, str],
    *,
    arm: str,
    episode_dir: Path,
    gpu_index: int,
    port: int,
    boundary_path: Path | None = None,
) -> dict[str, str]:
    """Enable the existing applied supervisor while keeping its curated policy."""

    if arm not in {"curated_supervisor", "current_feedback"}:
        raise ValueError(f"unknown live arm: {arm}")
    env = _PRIOR_SESSION_ENVIRONMENT(
        base_env,
        episode_dir=episode_dir,
        gpu_index=gpu_index,
        port=port,
        boundary_path=boundary_path,
        arm="direct_selfparse",
    )
    for key in (
        "CARNOT_ARC_SELFPARSE_RESULT_RESUME",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
        "CARNOT_ARC_SUPERVISOR_TOOL_LOOP_REINDUCTION",
        "CARNOT_ARC_SUPERVISOR_ORDER",
        "CARNOT_ARC_SUPERVISOR_ORDER_HASH",
    ):
        env.pop(key, None)
    seed = _episode_seed(episode_dir)
    env.update(
        {
            "CARNOT_FORCE_LIVE": "1",
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR": "1",
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": "120",
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(MODEL_CALL_LIMIT),
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(MODEL_CALL_LIMIT),
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(MAX_NEW_TOKENS),
            "CARNOT_ARC_RANDOM_SEED": str(seed),
            "CARNOT_ARC_GENERATOR_SEED": str(seed),
            "CARNOT_7376_EPISODE_ID": episode_dir.name.replace("__", ":"),
        }
    )
    return env


def _runtime_event_id(*, receipt_id: str, game: str, seed: Any, action_index: Any, arm: str) -> str:
    return canonical_hash(
        {
            "receipt_id": receipt_id,
            "game": game,
            "seed": seed,
            "action_index": action_index,
            "arm": arm,
        }
    )


def redirect_rows_for_episode(episode: Mapping[str, Any]) -> list[JsonDict]:
    """Project only observed applied redirects into action-linked outcome rows."""

    receipt = episode.get("trajectory_supervisor")
    if not isinstance(receipt, Mapping):
        return []
    mode = receipt.get("mode") or episode.get("supervisor_mode")
    if mode != "applied":
        return []
    redirects = [dict(row) for row in receipt.get("redirects", []) if isinstance(row, Mapping)]
    receipt_id = canonical_hash(
        {
            "episode_id": episode.get("episode_id"),
            "supervisor": receipt,
            "starting_policy": episode.get("starting_policy"),
        }
    )
    terminal_action = int(episode.get("action_count") or 0)
    rows: list[JsonDict] = []
    for index, redirect in enumerate(redirects):
        arm = str(redirect.get("arm"))
        if arm not in CURATED_ARMS or not isinstance(redirect.get("resolved_by_levelup"), bool):
            continue
        trigger = int(redirect.get("action_index") or 0)
        observed_resolved = redirect["resolved_by_levelup"] is True
        episode_censored = bool(episode.get("censored"))
        resolved = None if episode_censored else observed_resolved
        actions_to_levelup = redirect.get("actions_to_levelup")
        if observed_resolved and isinstance(actions_to_levelup, int):
            later_end = trigger + actions_to_levelup
            censoring = None
        else:
            later_end = terminal_action
            censoring = (
                "episode_censored_before_terminal_outcome"
                if episode_censored
                else "right_censored_episode_end"
            )
        competing = [
            {
                "arm": other.get("arm"),
                "action_index": other.get("action_index"),
            }
            for other in redirects[index + 1 :]
            if int(other.get("action_index") or 0) <= later_end
        ]
        rows.append(
            {
                "runtime_event_id": _runtime_event_id(
                    receipt_id=receipt_id,
                    game=str(episode.get("game")),
                    seed=episode.get("seed"),
                    action_index=trigger,
                    arm=arm,
                ),
                "receipt_id": receipt_id,
                "episode_id": episode.get("episode_id"),
                "game": episode.get("game"),
                "seed": episode.get("seed"),
                "trigger_action": trigger,
                "selected_arm": arm,
                "later_action_range": [trigger + 1, later_end],
                "level_before": redirect.get("level"),
                "level_progress": int(observed_resolved) if not episode_censored else None,
                "resolved_by_levelup": resolved,
                "actions_to_levelup": actions_to_levelup,
                "competing_redirects": competing,
                "censoring": censoring,
                "censoring_reason": censoring,
                "censored": episode_censored,
                "outcome_observed": not episode_censored,
                "mode": "applied",
                "evidence_class": "authenticated_applied_outcome",
                "causal_interpretation": "descriptive_association_only",
                "source_engine_provenance_sha256": (
                    episode.get("source_engine_provenance") or {}
                ).get("sha256"),
                "starting_policy_sha256": (episode.get("starting_policy") or {}).get("sha256"),
                "ending_policy_sha256": (episode.get("ending_policy") or {}).get("sha256"),
            }
        )
    return rows


def extract_new_outcomes(episodes: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Extract authenticated action-linked rows without inventing missing outcomes."""

    return [row for episode in episodes for row in redirect_rows_for_episode(episode)]


def historical_event_rows(ledger: Mapping[str, Any]) -> list[JsonDict]:
    """Convert durable applied development receipts to the same event identity shape."""

    entries = ledger.get("entries")
    if not isinstance(entries, Mapping):
        return []
    rows: list[JsonDict] = []
    for key, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping) or raw_entry.get("mode") != "applied":
            continue
        entry = dict(raw_entry)
        receipt_id = str(entry.get("receipt_id") or key)
        for index, redirect in enumerate(entry.get("redirects", [])):
            if not isinstance(redirect, Mapping):
                continue
            arm = str(redirect.get("arm"))
            outcome = redirect.get("resolved_by_levelup")
            if arm not in CURATED_ARMS or not isinstance(outcome, bool):
                continue
            rows.append(
                {
                    "runtime_event_id": f"historical:{key}:{index}",
                    "receipt_id": receipt_id,
                    "episode_id": receipt_id,
                    "game": entry.get("game"),
                    "seed": entry.get("seed"),
                    "trigger_action": redirect.get("action_index"),
                    "selected_arm": arm,
                    "resolved_by_levelup": outcome,
                    "actions_to_levelup": redirect.get("actions_to_levelup"),
                    "censoring": None if outcome else "historical_episode_end",
                    "censored": False,
                    "outcome_observed": True,
                    "mode": "applied",
                    "historical_only": True,
                    "evidence_class": "authenticated_applied_outcome",
                    "causal_interpretation": "descriptive_association_only",
                    "source": entry.get("source"),
                }
            )
    return rows


historical_support_rows = historical_event_rows


def join_event_rows(
    historical_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Join by runtime identity and retain the first authenticated occurrence once."""

    unique: dict[str, JsonDict] = {}
    duplicates = 0
    for raw in (*historical_rows, *new_rows):
        row = dict(raw)
        identity = row.get("runtime_event_id")
        if not isinstance(identity, str) or not identity:
            continue
        if identity in unique:
            duplicates += 1
            continue
        unique[identity] = deepcopy(row)
    return {"rows": list(unique.values()), "duplicates_dropped": duplicates}


def reduce_support(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str] = CURATED_ARMS
) -> JsonDict:
    """Recompute the fixed per-arm and leave-one-game-out support threshold."""

    deduplicated = join_event_rows([], rows)
    eligible = [
        dict(row)
        for row in deduplicated["rows"]
        if row.get("evidence_class") in {None, "authenticated_applied_outcome"}
        and row.get("mode", "applied") == "applied"
        and row.get("outcome_observed", True) is True
        and row.get("censored", False) is not True
        and row.get("selected_arm") in arms
        and isinstance(row.get("resolved_by_levelup"), bool)
        and isinstance(row.get("runtime_event_id"), str)
        and bool(row.get("game"))
    ]
    games = sorted({str(row.get("game")) for row in eligible if row.get("game")})

    def arm_summary(subset: Sequence[Mapping[str, Any]], arm: str) -> JsonDict:
        selected = [row for row in subset if row.get("selected_arm") == arm]
        return {
            "outcome_bearing_decisions": len(selected),
            "supported_decision_count": len(selected),
            "development_games": sorted(
                {str(row.get("game")) for row in selected if row.get("game")}
            ),
            "development_game_count": len(
                {str(row.get("game")) for row in selected if row.get("game")}
            ),
            "resolved_by_levelup_count": sum(
                row.get("resolved_by_levelup") is True for row in selected
            ),
        }

    per_arm = {arm: arm_summary(eligible, arm) for arm in arms}
    aggregate_passed = all(
        row["outcome_bearing_decisions"] >= 10 and row["development_game_count"] >= 3
        for row in per_arm.values()
    )
    loo: list[JsonDict] = []
    for held_out in games:
        subset = [row for row in eligible if str(row.get("game")) != held_out]
        summaries = {arm: arm_summary(subset, arm) for arm in arms}
        passed = all(
            row["outcome_bearing_decisions"] >= 10 and row["development_game_count"] >= 3
            for row in summaries.values()
        )
        loo.append(
            {
                "held_out_game": held_out,
                "expected_minimum_per_arm": 10,
                "expected_minimum_games_per_arm": 3,
                "per_arm": summaries,
                "passed": passed,
            }
        )
    shortfalls = [
        {
            "arm": arm,
            "expected_decisions": 10,
            "observed_decisions": row["outcome_bearing_decisions"],
            "expected_games": 3,
            "observed_games": row["development_game_count"],
        }
        for arm, row in per_arm.items()
        if row["outcome_bearing_decisions"] < 10 or row["development_game_count"] < 3
    ]
    result = {
        "supervisor_support_ready_score": int(
            aggregate_passed and bool(loo) and all(row["passed"] for row in loo)
        ),
        "per_arm_support": per_arm,
        "leave_one_game_out_rows": loo,
        "shortfalls": shortfalls,
        "eligible_runtime_event_count": len(eligible),
        "development_game_count": len(games),
        "duplicate_runtime_event_count": deduplicated["duplicates_dropped"],
        "duplicate_event_count": deduplicated["duplicates_dropped"],
        "ineligible_event_count": len(deduplicated["rows"]) - len(eligible),
        "causal_interpretation": "descriptive_association_only",
    }
    result["per_arm"] = deepcopy(per_arm)
    return result


def _episode_summary(
    episode: Mapping[str, Any], redirect_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    receipt = episode.get("trajectory_supervisor")
    receipt = dict(receipt) if isinstance(receipt, Mapping) else {}
    action_count = int(episode.get("action_count") or 0)
    calls_attempted = int(episode.get("generation_calls_attempted") or 0)
    generated_tokens = int(episode.get("generated_tokens") or 0)
    return {
        "episode_id": episode.get("episode_id"),
        "game": episode.get("game"),
        "seed": episode.get("seed"),
        "disposition": episode.get("disposition"),
        "censored": bool(episode.get("censored")),
        "censoring_reason": episode.get("censoring_reason"),
        "action_budget": ACTION_LIMIT,
        "actions_observed": action_count,
        "model_call_budget": MODEL_CALL_LIMIT,
        "model_calls_attempted": calls_attempted,
        "model_calls_completed": int(episode.get("generation_calls_completed") or 0),
        "max_new_tokens_per_call": MAX_NEW_TOKENS,
        "generated_tokens": generated_tokens,
        "level_progress": int(episode.get("levels") or 0),
        "redirect_count": len(redirect_rows),
        "no_firing": not redirect_rows,
        "all_arms_exhausted": bool(
            receipt.get("stagnations_unredirected")
            and set(receipt.get("arms_used") or []) >= set(CURATED_ARMS)
        ),
        "supervisor_mode": receipt.get("mode"),
        "supervisor_observe_errors": receipt.get("observe_errors"),
        "starting_policy_sha256": (episode.get("starting_policy") or {}).get("sha256"),
        "ending_policy_sha256": (episode.get("ending_policy") or {}).get("sha256"),
        "source_engine_provenance_sha256": (episode.get("source_engine_provenance") or {}).get(
            "sha256"
        ),
        "model_invocation_receipt_count": len(episode.get("raw_request_manifest") or []),
        "tool_result_to_later_action_receipts": deepcopy(
            episode.get("tool_result_to_later_action_receipts") or []
        ),
        "failure": episode.get("error"),
        "cost": deepcopy(episode.get("compute_cost") or {}),
        "registered_levels_before_attempt": episode.get("registry_levels_before_attempt"),
        "new_solve_credit": False,
    }


def reduce_episode_accounting(
    schedule: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Check all six sealed dispositions while allowing a complete no-fire ledger."""

    sealed = {str(row.get("episode_id")): dict(row) for row in schedule}
    observed = {str(row.get("episode_id")): dict(row) for row in episodes}
    failures: list[JsonDict] = []
    per_game: list[JsonDict] = []
    all_redirects: list[JsonDict] = []
    if len(sealed) != 6 or len(observed) != 6 or set(sealed) != set(observed):
        failures.append(
            {
                "check": "six_episode_identity",
                "expected": sorted(sealed),
                "observed": sorted(observed),
            }
        )
    for episode_id, planned in sealed.items():
        episode = observed.get(episode_id)
        if episode is None:
            continue
        receipt = episode.get("trajectory_supervisor")
        receipt = dict(receipt) if isinstance(receipt, Mapping) else {}
        mode = receipt.get("mode") or episode.get("supervisor_mode")
        checks = {
            "identity": all(
                episode.get(key) == planned.get(key) for key in ("episode_id", "game", "seed")
            ),
            "action_budget": int(episode.get("action_count") or 0) <= ACTION_LIMIT,
            "model_call_budget": int(episode.get("generation_calls_attempted") or 0)
            <= MODEL_CALL_LIMIT,
            "token_budget": int(episode.get("generated_tokens") or 0) <= GENERATED_TOKEN_LIMIT,
            "withheld_inputs": all(
                episode.get(key) is True
                for key in (
                    "adapter_disabled",
                    "banked_solution_disabled",
                    "hand_solver_disabled",
                    "saved_engine_disabled",
                    "replay_route_disabled",
                )
            ),
            "fresh_store": episode.get("fresh_store") is True,
            "scored_policy": (
                (episode.get("factory_receipt") or {}).get("factory") == "make_carnot_agent"
                and (episode.get("factory_receipt") or {}).get("policy_class") == "E3AgentPolicy"
            ),
            "curated_applied_supervisor": (
                mode == "applied" and list(receipt.get("arms_enabled") or []) == list(CURATED_ARMS)
            ),
            "terminal_disposition": episode.get("disposition")
            in {"complete", "complete_error", "censored_timeout", "censored_budget"},
        }
        for name, passed in checks.items():
            if not passed:
                failures.append({"episode_id": episode_id, "check": name, "observed": False})
        rows = redirect_rows_for_episode(episode)
        all_redirects.extend(rows)
        per_game.append(_episode_summary(episode, rows))
    return {
        "arc_outcome_capture_complete_score": int(not failures),
        "planned_units": len(sealed),
        "attempted_units": len(observed),
        "completed_units": sum(row.get("censored") is not True for row in observed.values()),
        "censored_units": sum(row.get("censored") is True for row in observed.values()),
        "redirect_event_count": len(all_redirects),
        "no_firing_episode_count": sum(row["no_firing"] for row in per_game),
        "all_arms_exhausted_episode_count": sum(row["all_arms_exhausted"] for row in per_game),
        "accounting_failures": failures,
        "authenticity_failures": deepcopy(failures),
        "per_game_results": per_game,
        "redirect_rows": all_redirects,
    }


def reduce_raw_panel(
    payload: Mapping[str, Any], *, historical_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Independently reduce episode dispositions, redirects, and support."""

    schedule = [dict(row) for row in payload.get("schedule", []) if isinstance(row, Mapping)]
    episodes = [dict(row) for row in payload.get("episodes", []) if isinstance(row, Mapping)]
    expected_ids = [str(row.get("episode_id")) for row in schedule]
    observed = {str(row.get("episode_id")): row for row in episodes}
    failures: list[str] = []
    if len(schedule) != 6 or len(episodes) != 6 or set(expected_ids) != set(observed):
        failures.append("planned_episode_disposition_mismatch")

    redirect_rows: list[JsonDict] = []
    per_game: list[JsonDict] = []
    authenticity_failures: list[JsonDict] = []
    for sealed in schedule:
        episode_id = str(sealed.get("episode_id"))
        episode = observed.get(episode_id)
        if episode is None:
            continue
        episode_redirects = redirect_rows_for_episode(episode)
        redirect_rows.extend(episode_redirects)
        per_game.append(_episode_summary(episode, episode_redirects))
        receipt = episode.get("trajectory_supervisor")
        receipt = receipt if isinstance(receipt, Mapping) else {}
        checks = {
            "identity_matches_schedule": all(
                episode.get(key) == sealed.get(key) for key in ("episode_id", "game", "seed")
            ),
            "budgets_match": (
                episode.get("action_limit") == ACTION_LIMIT
                and episode.get("completion_limit") == MODEL_CALL_LIMIT
                and episode.get("generated_token_limit") == GENERATED_TOKEN_LIMIT
                and episode.get("max_new_tokens_per_call") == MAX_NEW_TOKENS
                and int(episode.get("action_count") or 0) <= ACTION_LIMIT
                and int(episode.get("generation_calls_attempted") or 0) <= MODEL_CALL_LIMIT
                and int(episode.get("generated_tokens") or 0) <= GENERATED_TOKEN_LIMIT
            ),
            "withheld_inputs": all(
                episode.get(key) is True
                for key in (
                    "adapter_disabled",
                    "banked_solution_disabled",
                    "hand_solver_disabled",
                    "saved_engine_disabled",
                    "replay_route_disabled",
                )
            ),
            "factory_is_scored_e3": (
                (episode.get("factory_receipt") or {}).get("factory") == "make_carnot_agent"
                and (episode.get("factory_receipt") or {}).get("policy_class") == "E3AgentPolicy"
            ),
            "curated_applied_supervisor": (
                receipt.get("mode") == "applied"
                and list(receipt.get("arms_enabled") or []) == list(CURATED_ARMS)
            ),
            "policy_hashes_present": all(
                isinstance((episode.get(field) or {}).get("sha256"), str)
                for field in ("starting_policy", "ending_policy", "source_engine_provenance")
            ),
            "disposition_terminal": episode.get("disposition")
            in {
                "complete",
                "complete_error",
                "censored_timeout",
                "censored_budget",
            },
        }
        for check, passed in checks.items():
            if not passed:
                authenticity_failures.append(
                    {"episode_id": episode_id, "check": check, "observed": False}
                )

    joined = join_event_rows(historical_rows, redirect_rows)
    support = reduce_support(joined["rows"], CURATED_ARMS)
    complete = not failures and not authenticity_failures
    return {
        "arc_outcome_capture_complete_score": int(complete),
        "supervisor_support_ready_score": int(
            complete and support["supervisor_support_ready_score"] == 1
        ),
        "planned_units": len(schedule),
        "attempted_units": len(episodes),
        "completed_units": sum(row.get("censored") is not True for row in episodes),
        "censored_units": sum(row.get("censored") is True for row in episodes),
        "no_firing_episode_count": sum(row["no_firing"] for row in per_game),
        "all_arms_exhausted_episode_count": sum(row["all_arms_exhausted"] for row in per_game),
        "accounting_failures": failures,
        "authenticity_failures": authenticity_failures,
        "per_game_results": per_game,
        "redirect_rows": redirect_rows,
        "joined_event_rows": joined["rows"],
        "join_duplicates_dropped": joined["duplicates_dropped"],
        **support,
    }


def independent_reduce(path: Path) -> JsonDict:
    """Reload raw bytes and independently reduce accounting and support evidence."""

    payload = load_json(path) or {}
    schedule = [dict(row) for row in payload.get("schedule", []) if isinstance(row, Mapping)]
    episodes = [dict(row) for row in payload.get("episodes", []) if isinstance(row, Mapping)]
    historical = [
        dict(row)
        for row in (payload.get("historical_rows") or payload.get("historical_event_rows") or [])
        if isinstance(row, Mapping)
    ]
    accounting = reduce_episode_accounting(schedule, episodes)
    joined = join_event_rows(historical, accounting["redirect_rows"])
    return {
        "episode_accounting": accounting,
        "support": reduce_support(joined["rows"]),
        "joined_rows": joined["rows"],
        "duplicates_dropped": joined["duplicates_dropped"],
    }


def affected_manifest() -> validation_contract.AffectedManifest:
    """Name the exact changed files for the Experiment 7358 command planner."""

    return validation_contract.AffectedManifest(
        experiment_id=EXPERIMENT_ID,
        test_paths=(TEST_PATH.as_posix(),),
        changed_modules=(MODULE_PATH.as_posix(),),
        static_paths=(WRAPPER_PATH.as_posix(),),
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the exact Experiment 7358 affected command plan."""

    return validation_contract.build_command_plan(root, affected_manifest(), private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject any drift from the Experiment 7358 scoped command contract."""

    return validation_contract.validate_command_plan(root, affected_manifest(), commands)


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("exit_code", 0) == 0
        and by_name[name].get("timed_out") is not True
        for name in names
    )


def _acceptance_gates(
    preconditions: Sequence[Mapping[str, Any]],
    reduction: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    invocation_counts: Mapping[str, Any],
) -> list[JsonDict]:
    required_names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *REQUIRED_E2E_NAMES,
        *REQUIRED_TERMINAL_NAMES,
    )
    adversarial = next(
        (row for row in validation_receipts if row.get("name") == "adversarial_verify"), None
    )
    return [
        gate_check(
            "preconditions",
            "current_run",
            "all_preconditions_passed",
            True,
            all(row.get("passed") is True for row in preconditions),
        ),
        gate_check(
            "episode_accounting",
            RAW_PANEL_PATH.as_posix(),
            "arc_outcome_capture_complete_score",
            1,
            int(reduction.get("arc_outcome_capture_complete_score") or 0),
        ),
        gate_check(
            "native_cuda_runtime",
            "runner_receipt",
            "task_linked_cuda_execution",
            True,
            runtime_receipt.get("task_linked_cuda_execution") is True,
        ),
        gate_check(
            "bounded_generation_attempted",
            "invocation_counts",
            "generation_calls_attempted>0",
            True,
            int(invocation_counts.get("generation_calls_attempted") or 0) > 0,
        ),
        gate_check(
            "required_validation",
            "validation_receipts",
            "all_required_commands_passed",
            True,
            _receipts_pass(validation_receipts, required_names),
        ),
        gate_check(
            "adversarial_clear",
            "adversarial_verify",
            "critical_finding",
            False,
            bool(adversarial is not None and adversarial.get("passed") is not True),
        ),
        gate_check(
            "support_floor",
            "joined_runtime_events",
            "supervisor_support_ready_score",
            1,
            int(reduction.get("supervisor_support_ready_score") or 0),
        ),
        gate_check("promotion", "protocol", "promotion_score", 1, 0),
    ]


def _principles(artifact: Mapping[str, Any]) -> JsonDict:
    return {
        key: FIELD_PRINCIPLES.get(
            key,
            "Retain this ordinary field as direct evidence for independent recomputation.",
        )
        for key in artifact
    }


def build_terminal_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    raw_panel: Mapping[str, Any],
    reduction: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    invocation_counts: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    repository_health: Mapping[str, Any],
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build a complete, null, or disqualified terminal artifact from raw evidence."""

    counts = {**ZERO_INVOCATION_COUNTS, **deepcopy(dict(invocation_counts))}
    model_invoked = int(counts.get("model_loads_attempted") or 0) > 0
    generated = int(counts.get("generation_calls_attempted") or 0) > 0
    gates = _acceptance_gates(
        preconditions, reduction, validation_receipts, runtime_receipt, counts
    )
    required_ok = all(row["passed"] for row in gates[:6])
    flagged = gates[5]["observed"] is True
    support_ready = int(
        required_ok and not flagged and reduction.get("supervisor_support_ready_score") == 1
    )
    if required_ok:
        status = (
            "complete_supervisor_support_floor_observed"
            if support_ready
            else "complete_null_insufficient_supervisor_support"
        )
        verdict_class = "null"
        honest_verdict = status
    else:
        status = "complete_disqualified_required_evidence"
        verdict_class = "disqualified"
        honest_verdict = status
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": list(MODEL_SPECS),
        "resolved_model_specs": [deepcopy(dict(row)) for row in model_specs],
        "model_invoked": model_invoked,
        "invocation_counts": counts,
        "inference_substrate": (
            "owned_native_cuda_llama_cpp" if model_invoked else "blocked_no_run"
        ),
        "inference_substrate_class": (
            "model_bounded_generation"
            if generated
            else "model_load_no_generation"
            if model_invoked
            else "blocked_no_run"
        ),
        "inference_mode": (
            "live_gpu"
            if runtime_receipt.get("task_linked_cuda_execution") is True
            else "not_verified"
        ),
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "experiment": RANDOM_SEED,
            "episodes": list(EPISODE_SEEDS),
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(reduction.get("per_game_results") or []),
        "redirect_outcome_rows": deepcopy(reduction.get("redirect_rows") or []),
        "joined_support_rows": deepcopy(reduction.get("joined_event_rows") or []),
        "per_arm_support": deepcopy(reduction.get("per_arm_support") or {}),
        "leave_one_game_out_rows": deepcopy(reduction.get("leave_one_game_out_rows") or []),
        "support_shortfalls": deepcopy(reduction.get("shortfalls") or []),
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": reduction.get("attempted_units"),
            "completed_units": reduction.get("completed_units"),
            "censored_units": reduction.get("censored_units"),
            "games": 3,
            "seeds_per_game": 2,
            "action_limit_per_episode": ACTION_LIMIT,
            "model_call_limit_per_episode": MODEL_CALL_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
            "aggregate_episode_work_budget_s": EPISODE_WORK_LIMIT_S,
            "stopping_rule": "six sealed dispositions or the first fixed action, call, token, or elapsed ceiling",
            "remaining_work": 0 if reduction.get("planned_units") == 6 else 6,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": flagged,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "promotion_score": 0,
        "scientific_value_score": 0,
        "arc_outcome_capture_complete_score": int(
            required_ok and bool(reduction.get("arc_outcome_capture_complete_score"))
        ),
        "supervisor_support_ready_score": support_ready,
        "per_game_results": deepcopy(reduction.get("per_game_results") or []),
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_claimed": False,
        "reproduced_levels": [],
        "official_score": None,
        "runner_receipt": deepcopy(dict(runtime_receipt)),
        "selection_receipt": deepcopy(dict(selection)),
        "raw_evidence_receipt": {
            "raw_panel_sha256": canonical_hash(raw_panel),
            "episode_count": len(raw_panel.get("episodes", [])),
            "redirect_count": len(reduction.get("redirect_rows") or []),
            "join_duplicates_dropped": reduction.get("join_duplicates_dropped"),
            "accounting_failures": deepcopy(reduction.get("accounting_failures") or []),
            "authenticity_failures": deepcopy(reduction.get("authenticity_failures") or []),
        },
        "generalization_activity": {
            "kind": "adapter_withheld_scored_policy_runtime",
            "public_games_are_generalization_proxy": True,
            "hidden_leaderboard_evidence": False,
            "curated_arms_unchanged": True,
            "offline_fitted_selector_used": False,
            "priority_changed": False,
            "autonomous_arm_invention": False,
        },
        "production_defaults_changed": False,
        "solve_registry_changed": False,
        "submitted_policy_changed": False,
        "field_principles": {},
    }
    artifact["field_principles"] = _principles(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selection: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    historical_rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    invocation_counts: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    repository_health: Mapping[str, Any],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
) -> JsonDict:
    """Build the public artifact from independently reducible schedule and rows."""

    accounting = reduce_episode_accounting(schedule, episodes)
    joined = join_event_rows(historical_rows, accounting["redirect_rows"])
    support = reduce_support(joined["rows"])
    reduction = {
        **accounting,
        **support,
        "joined_event_rows": joined["rows"],
        "join_duplicates_dropped": joined["duplicates_dropped"],
        "supervisor_support_ready_score": int(
            accounting["arc_outcome_capture_complete_score"] == 1
            and support["supervisor_support_ready_score"] == 1
        ),
    }
    raw_panel = {
        "schedule": [deepcopy(dict(row)) for row in schedule],
        "episodes": [deepcopy(dict(row)) for row in episodes],
        "historical_rows": [deepcopy(dict(row)) for row in historical_rows],
    }
    return build_terminal_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        selection=selection,
        raw_panel=raw_panel,
        reduction=reduction,
        runtime_receipt=runtime_receipt,
        model_specs=model_specs,
        invocation_counts=invocation_counts,
        validation_receipts=validation_receipts,
        repository_health=repository_health,
        started_at_utc=started_at_utc,
        ended_at_utc=completed_at_utc,
        duration_s=duration_s,
        phase_spans=phase_spans,
    )


def build_blocked_artifact(
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    started_at_utc: str,
    duration_s: float,
    ended_at_utc: str | None = None,
    completed_at_utc: str | None = None,
) -> JsonDict:
    """Publish external absence as terminal blocked work without fake inference."""

    summary = gate_summary(preconditions)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "status": "blocked_external_precondition",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc or completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": list(MODEL_SPECS),
        "resolved_model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "inference_mode": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": socket.gethostname(),
        "duration_s": round(max(float(duration_s), 0.000001), 6),
        "phase_spans": [],
        "random_seed": {
            "experiment": RANDOM_SEED,
            "episodes": list(EPISODE_SEEDS),
            "sealed_before_outcomes": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": [],
        "redirect_outcome_rows": [],
        "joined_support_rows": [],
        "per_arm_support": {},
        "leave_one_game_out_rows": [],
        "support_shortfalls": [],
        "sample_size_budget": {
            "planned_units": 6,
            "attempted_units": 0,
            "completed_units": 0,
            "censored_units": 0,
            "remaining_work": 6,
            "stopping_rule": "block before dependent work on the first unavailable prerequisite",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "honest_verdict": "blocked_external_precondition",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {
            "status": "not_evaluated",
            "historical_failures": [],
            "affects_required_checks": False,
        },
        "promotion_score": 0,
        "scientific_value_score": 0,
        "arc_outcome_capture_complete_score": 0,
        "supervisor_support_ready_score": 0,
        "per_game_results": [],
        "solve_provenance": "live_agent_self_discovery",
        "new_solve_claimed": False,
        "reproduced_levels": [],
        "official_score": None,
        "runner_receipt": {},
        "selection_receipt": {},
        "raw_evidence_receipt": {},
        "generalization_activity": {
            "kind": "adapter_withheld_scored_policy_runtime",
            "completed": False,
        },
        "production_defaults_changed": False,
        "solve_registry_changed": False,
        "submitted_policy_changed": False,
        "field_principles": {},
    }
    artifact["field_principles"] = _principles(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, safety scores, duration class, fields, and checksum."""

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
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    blocked = artifact.get("verdict_class") == "blocked"
    expected_prefix = "blocked_" if blocked else "complete_"
    if not str(artifact.get("honest_verdict") or "").startswith(expected_prefix):
        errors.append("honest_verdict")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("MODEL_SPECS")
    counts = artifact.get("invocation_counts") or {}
    if blocked:
        if artifact.get("model_invoked") is not False or counts != ZERO_INVOCATION_COUNTS:
            errors.append("blocked_invocations")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate")
        if (artifact.get("gate_check_summary") or {}).get("first_failure") is None:
            errors.append("blocked_gate_check_summary")
    else:
        invoked = int(counts.get("model_loads_attempted") or 0) > 0
        generated = int(counts.get("generation_calls_attempted") or 0) > 0
        expected_class = "model_bounded_generation" if generated else "model_load_no_generation"
        if artifact.get("model_invoked") is not invoked:
            errors.append("model_invoked")
        if artifact.get("inference_substrate_class") != expected_class:
            errors.append("inference_substrate_class")
        floor = 10.0 if generated else 2.0
        if float(artifact.get("duration_s") or 0) < floor:
            errors.append("duration_s")
        if (
            require_validation
            and not _receipts_pass(
                artifact.get("validation_receipts") or [],
                (
                    *validation_scope.REQUIRED_CHECK_NAMES,
                    *REQUIRED_E2E_NAMES,
                    *REQUIRED_TERMINAL_NAMES,
                ),
            )
            and artifact.get("verdict_class") != "disqualified"
        ):
            errors.append("validation_receipts")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score")
    if artifact.get("verdict_class") in {"blocked", "disqualified"} and (
        artifact.get("supervisor_support_ready_score") != 0
        or artifact.get("scientific_value_score") != 0
    ):
        errors.append("unsafe_readiness")
    if artifact.get("production_defaults_changed") is not False:
        errors.append("production_defaults_changed")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum")
    return list(dict.fromkeys(errors))


def _source_record(path: Path, *, role: str) -> JsonDict:  # pragma: no cover
    record: JsonDict = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "role": role,
        "path": path.as_posix(),
    }
    if path.suffix == ".json":
        value = load_json(path) or {}
        record.update(
            {
                "producer_experiment_id": value.get("experiment_id") or value.get("experiment"),
                "producer_status": value.get("status"),
                "producer_verdict_class": value.get("verdict_class"),
                "producer_flagged_adversarial": value.get("flagged_adversarial"),
                "producer_inference_substrate_class": value.get("inference_substrate_class"),
            }
        )
    return record


def collect_preconditions(
    root: Path,
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:  # pragma: no cover
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            gate_check(
                "required_input",
                relative.as_posix(),
                "readable_nonempty_bytes",
                True,
                available,
            )
        )
        if available:
            role = (
                "historical_diagnostic_evidence"
                if relative in {EXP7365_PATH, EXP7366_PATH}
                else "validation_contract"
                if relative == EXP7358_PATH
                else "current_input"
            )
            hashes[relative.as_posix()] = _source_record(path, role=role)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    checks.append(
        gate_check(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7376",
            True,
            "### REQ-ARC-WMTE-7376:" in spec,
        )
    )
    manifest_text = (
        (root / EXCLUSION_PATH).read_text(encoding="utf-8")
        if (root / EXCLUSION_PATH).is_file()
        else ""
    )
    checks.append(
        gate_check(
            "exclusion_manifest",
            EXCLUSION_PATH.as_posix(),
            "exp7376_retired",
            False,
            "exp7376" in manifest_text.lower(),
        )
    )
    checks.append(
        gate_check(
            "force_live",
            "environment",
            "CARNOT_FORCE_LIVE",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE"),
        )
    )
    registry = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    ledger = load_json(root / LEDGER_PATH) or {}
    return checks, hashes, dict(registry), ledger


def runtime_preconditions(
    root: Path, *, gpu_wait_s: float, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:  # pragma: no cover
    checks: list[JsonDict] = []
    hashes: JsonDict = {}
    deadline = time.monotonic() + max(0.0, gpu_wait_s)
    idle: list[JsonDict] = []
    while True:
        idle = [
            row
            for row in live_base._gpu_inventory()
            if not row.get("compute_apps") and int(row.get("free_memory_mb") or 0) >= 20_000
        ]
        if idle or time.monotonic() >= deadline:
            break
        progress(
            started,
            "runtime_preconditions",
            "pending_idle_gpu",
            remaining_s=round(max(0.0, deadline - time.monotonic()), 1),
        )
        time.sleep(min(30.0, max(0.0, deadline - time.monotonic())))
    gpu = idle[0] if idle else None
    checks.append(
        gate_check("owned_gpu", "nvidia-smi", "idle_gpu_with_20GB", True, gpu is not None)
    )

    from carnot.inference.sota_models import cached_current_model, gguf_tokenizer_loadable
    from llama_cpp import llama_cpp

    native_cuda = bool(llama_cpp.llama_supports_gpu_offload())
    checks.append(
        gate_check(
            "native_cuda_offload", "llama_cpp", "llama_supports_gpu_offload", True, native_cuda
        )
    )
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
    checks.append(gate_check("model_cache", MODEL_ID, "Q4_K_M_path", True, model_ok))
    progress(started, "runtime_preconditions", "before_tokenizer_load")
    tokenizer_ok, tokenizer_detail = gguf_tokenizer_loadable(str(model_path) if model_ok else None)
    progress(
        started,
        "runtime_preconditions",
        "after_tokenizer_load",
        passed=tokenizer_ok,
    )
    checks.append(
        gate_check(
            "embedded_tokenizer",
            str(model_path),
            "loadable",
            True,
            tokenizer_ok,
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
        progress(started, "runtime_preconditions", "before_model_hash", path=model_path)
        model_hash = sha256_file(model_path)
        progress(started, "runtime_preconditions", "after_model_hash", sha256=model_hash)
        hashes[str(model_path)] = {
            "sha256": model_hash,
            "bytes": model_path.stat().st_size,
            "role": "current_model",
            "authenticated_native_cuda": native_cuda,
        }
    if server is not None:
        hashes[str(server)] = {
            "sha256": sha256_file(server),
            "bytes": server.stat().st_size,
            "role": "native_runtime_binary",
        }
    resolved = {
        **deepcopy(dict(model or {})),
        "hf_id": MODEL_ID,
        "quantization": QUANTIZATION,
        "model_path": str(model_path) if model_path else None,
        "sha256": model_hash,
        "bytes": model_path.stat().st_size if model_ok and model_path else None,
        "runtime_settings": {
            "runner": "LocalGGUFProposer_native_llama.cpp",
            "context_tokens": 49_152,
            "kv_quantization": "q8_0",
            "offload_layers_requested": 999,
            "model_call_limit_per_episode": MODEL_CALL_LIMIT,
            "max_new_tokens_per_call": MAX_NEW_TOKENS,
        },
        "tokenizer_detail": tokenizer_detail,
    }
    return (
        checks,
        hashes,
        {
            "gpu": gpu,
            "model_path": str(model_path) if model_ok and model_path else None,
            "model_hash": model_hash,
            "model_spec": resolved,
            "server": str(server) if server else None,
        },
    )


def _policy_receipt(policy: Any, *, phase: str) -> JsonDict:  # pragma: no cover
    from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG

    supervisor = policy.trajectory_supervisor_diagnostics()
    attempts = [
        dict(row) for row in getattr(policy, "induction_attempts", []) if isinstance(row, Mapping)
    ]
    engine_hashes = sorted(
        {
            str(value)
            for row in attempts
            for key, value in row.items()
            if "engine" in str(key) and "sha" in str(key) and isinstance(value, str)
        }
    )
    payload = {
        "phase": phase,
        "policy_class": type(policy).__name__,
        "policy_module": type(policy).__module__,
        "submitted_config_sha256": canonical_hash(SUBMITTED_AGENT_CONFIG),
        "trajectory_supervisor": supervisor,
        "induction_attempt_count": len(attempts),
        "engine_hashes": engine_hashes,
        "plan_length": len(getattr(policy, "plan", []) or []),
        "phase_state": getattr(policy, "phase", None),
    }
    return {**payload, "sha256": canonical_hash(payload)}


@contextmanager
def _configured_runtime() -> Any:  # pragma: no cover
    from carnot import experiment_7263_v639_arc_live as live

    base_updates = {
        "RUN_DATE": RUN_DATE,
        "MILESTONE": MILESTONE,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "MODEL_SPECS": [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}],
        "TARGET_GAME": "three_game_panel",
        "DEVELOPMENT_SEED": RANDOM_SEED,
        "EVALUATION_SEED": EPISODE_SEEDS[0],
        "ACTION_LIMIT": ACTION_LIMIT,
        "COMPLETION_LIMIT": MODEL_CALL_LIMIT,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "TOKENS_PER_CALL": MAX_NEW_TOKENS,
        "SESSION_LIMIT_S": EPISODE_WORK_LIMIT_S,
        "MODEL_LOAD_LIMIT_S": MODEL_LOAD_LIMIT_S,
        "RESULT_PATH": RESULT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "TOOL_EVENT_PATH": RAW_DIR / "tool_events.jsonl",
        "SESSION_PATH": SESSION_PATH,
        "RAW_ROW_PATH": RAW_PANEL_PATH,
        "TERMINAL_CANDIDATE_PATH": TERMINAL_CANDIDATE_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "session_environment": session_environment,
    }
    live_updates = {
        "ACTION_LIMIT": ACTION_LIMIT,
        "COMPLETION_LIMIT": MODEL_CALL_LIMIT,
        "GENERATED_TOKEN_LIMIT": GENERATED_TOKEN_LIMIT,
        "TOKENS_PER_CALL": MAX_NEW_TOKENS,
        "LIVE_WINDOW_S": EPISODE_WORK_LIMIT_S,
        "STARTUP_TIMEOUT_S": MODEL_LOAD_LIMIT_S,
    }
    old_base = {name: getattr(live_base, name) for name in base_updates}
    old_live = {name: getattr(live, name) for name in live_updates}
    try:
        for name, value in base_updates.items():
            setattr(live_base, name, value)
        for name, value in live_updates.items():
            setattr(live, name, value)
        with live_base._configured_runtime() as reused:
            yield reused
    finally:
        for name, value in old_live.items():
            setattr(live, name, value)
        for name, value in old_base.items():
            setattr(live_base, name, value)


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover
    """Add supervisor and policy receipts around the reused shipped live child."""

    from carnot import experiment_7263_v639_arc_live as live
    from carnot import experiment_7234_v637_arc_scored_dryrun as scored

    original_build = scored.build_disposable_submitted_policy
    original_project = live._episode_result

    def build_policy(game: str, proposer: Any) -> tuple[Any, JsonDict]:
        policy, factory = original_build(game, proposer)
        policy._exp7376_starting_policy = _policy_receipt(policy, phase="before_first_action")
        return policy, factory

    def project_episode(*call_args: Any, **kwargs: Any) -> JsonDict:
        row = original_project(*call_args, **kwargs)
        policy = kwargs["policy"]
        ending = _policy_receipt(policy, phase="after_terminal_action")
        attempts = [
            dict(item)
            for item in getattr(policy, "induction_attempts", [])
            if isinstance(item, Mapping)
        ]
        provenance = {
            "policy_source_sha256": sha256_file(
                REPO_ROOT / "python/carnot/agentic/arc_competition_agent.py"
            ),
            "supervisor_source_sha256": sha256_file(
                REPO_ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"
            ),
            "induction_attempts_sha256": canonical_hash(attempts),
            "engine_hashes": ending["engine_hashes"],
        }
        provenance["sha256"] = canonical_hash(provenance)
        row.update(
            {
                "seed": _episode_seed(
                    Path(str(row.get("episode_id", "seed-0")).replace(":", "__"))
                ),
                "max_new_tokens_per_call": MAX_NEW_TOKENS,
                "hand_solver_disabled": True,
                "saved_engine_disabled": True,
                "replay_route_disabled": True,
                "starting_policy": deepcopy(policy._exp7376_starting_policy),
                "ending_policy": ending,
                "trajectory_supervisor": deepcopy(ending["trajectory_supervisor"]),
                "source_engine_provenance": provenance,
                "tool_result_to_later_action_receipts": [],
            }
        )
        return row

    scored.build_disposable_submitted_policy = build_policy
    live._episode_result = project_episode
    try:
        with _configured_runtime() as reused:
            return int(reused.run_live_session(args))
    finally:
        live._episode_result = original_project
        scored.build_disposable_submitted_policy = original_build


def _complete_episode_rows(
    schedule: Sequence[Mapping[str, Any]], session: Mapping[str, Any]
) -> list[JsonDict]:  # pragma: no cover
    existing = {
        str(row.get("episode_id")): deepcopy(dict(row))
        for row in session.get("episodes", [])
        if isinstance(row, Mapping)
    }
    rows: list[JsonDict] = []
    for sealed in schedule:
        episode_id = str(sealed["episode_id"])
        row = existing.get(
            episode_id,
            {
                **deepcopy(dict(sealed)),
                "disposition": "censored_timeout",
                "censored": True,
                "censoring_reason": "aggregate_episode_work_timeout",
                "model_invoked": False,
                "model_loaded": bool(session.get("model_loaded")),
                "generation_calls_attempted": 0,
                "generation_calls_completed": 0,
                "generated_tokens": 0,
                "action_count": 0,
                "levels": 0,
                "trajectory_supervisor": None,
                "starting_policy": {},
                "ending_policy": {},
                "source_engine_provenance": {},
                "tool_result_to_later_action_receipts": [],
                "raw_request_manifest": [],
                "factory_receipt": {
                    "factory": "make_carnot_agent",
                    "policy_class": "E3AgentPolicy",
                    "adapter_disabled": True,
                },
                "error": "aggregate_episode_work_timeout",
            },
        )
        row.update(
            {
                "episode_id": episode_id,
                "game": sealed["game"],
                "seed": sealed["seed"],
                "arm": "curated_supervisor",
                "action_limit": ACTION_LIMIT,
                "completion_limit": MODEL_CALL_LIMIT,
                "generated_token_limit": GENERATED_TOKEN_LIMIT,
                "max_new_tokens_per_call": MAX_NEW_TOKENS,
                "fresh_store": True,
                "adapter_disabled": True,
                "banked_solution_disabled": True,
                "hand_solver_disabled": True,
                "saved_engine_disabled": True,
                "replay_route_disabled": True,
            }
        )
        if row.get("error") and row.get("disposition") == "complete":
            row["disposition"] = "complete_error"
        rows.append(row)
    return rows


def _invocation_counts(
    session: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover
    requests = [
        dict(request)
        for episode in episodes
        for request in episode.get("raw_request_manifest", [])
        if isinstance(request, Mapping)
    ]
    timed_out = bool(session.get("timed_out"))
    loaded = bool(session.get("model_loaded"))
    return {
        "model_loads_attempted": 1,
        "model_loads_completed": int(loaded),
        "model_loads_failed": int(not loaded and not timed_out),
        "model_loads_cancelled": int(not loaded and timed_out),
        "model_loads_in_flight": 0,
        "generation_calls_attempted": len(requests),
        "generation_calls_completed": sum(
            row.get("transport_completed") is True for row in requests
        ),
        "generation_calls_failed": sum(bool(row.get("error")) for row in requests),
        "generation_calls_cancelled": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": sum(row.get("usable_answer") is True for row in requests),
    }


def _phase(
    name: str, phase_start: float, run_start: float, units: int
) -> JsonDict:  # pragma: no cover
    return {
        "phase": name,
        "started_elapsed_s": round(phase_start - run_start, 6),
        "duration_s": round(time.monotonic() - phase_start, 6),
        "completed_units": units,
        "checkpoint_at_utc": utc_now(),
    }


def e2e_command_specs(
    root: Path, private: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    for parent in (private / "e2e009", private / "e2e010"):
        parent.mkdir(parents=True, exist_ok=True)
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
            "E2E-009",
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
            "E2E-010",
        ),
        validation_scope.CommandSpec(
            "e2e_009_llm_off_environment",
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
                str(private / "e2e009-llm-off.json"),
            ),
            "E2E-009 LLM-off real environment smoke",
        ),
    ]


def terminal_command_specs(
    root: Path, candidate: Path, raw_panel: Path
) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    python = str(root / ".venv/bin/python")
    return [
        validation_scope.CommandSpec(
            "independent_reducer",
            (
                python,
                "-u",
                str(root / WRAPPER_PATH),
                "--reduce-raw",
                str(raw_panel),
                "--ledger",
                str(root / LEDGER_PATH),
            ),
            "candidate raw evidence",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "measured terminal candidate",
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
            "measured terminal candidate",
        ),
    ]


def _run_scoped_validation(root: Path, private: Path) -> list[JsonDict]:  # pragma: no cover
    manifest = affected_manifest()
    commands = validation_contract.build_command_plan(root, manifest, private)
    plan_errors = validation_contract.validate_command_plan(root, manifest, commands)
    if plan_errors:
        return [
            {
                "name": "validation_command_plan",
                "passed": False,
                "exit_code": 1,
                "timed_out": False,
                "scope": "affected_manifest",
                "command": "not_started",
                "command_argv": [],
                "duration_s": 0.0,
                "log_path": None,
                "log_sha256": None,
                "output_tail": json.dumps(plan_errors),
            }
        ]
    return validation_scope.run_commands(
        root,
        commands,
        log_dir=root / RAW_DIR / "validation/scoped",
        heartbeat_s=60.0,
    )


def _hash_raw_evidence(root: Path, hashes: JsonDict) -> None:  # pragma: no cover
    for path in sorted(item for item in (root / RAW_DIR).rglob("*") if item.is_file()):
        hashes[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": "current_raw_evidence",
        }


def run_experiment(args: argparse.Namespace) -> JsonDict:  # pragma: no cover
    started = time.monotonic()
    started_at = utc_now()
    phases: list[JsonDict] = []
    progress(started, "startup", "entrypoint_authenticated", run_date=args.date)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before_static_checks")
    checks, hashes, registry, ledger = collect_preconditions(REPO_ROOT)
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
        errors = validate_artifact(artifact, require_validation=False)
        if errors:
            raise RuntimeError(f"blocked artifact validation failed: {errors}")
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        progress(started, "write", "terminal_blocked_artifact_written", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    progress(started, "runtime_preconditions", "before_resource_checks")
    runtime_checks, runtime_hashes, resources = runtime_preconditions(
        REPO_ROOT, gpu_wait_s=args.gpu_wait_s, started=started
    )
    checks.extend(runtime_checks)
    hashes.update(runtime_hashes)
    phases.append(_phase("build", phase_start, started, len(runtime_checks)))
    progress(
        started,
        "runtime_preconditions",
        "after_resource_checks",
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
        atomic_json(REPO_ROOT / RESULT_PATH, artifact)
        progress(started, "write", "terminal_runtime_block_written", path=RESULT_PATH)
        return artifact

    from carnot.agentic.arc_game_adapters import adaptered_games

    phase_start = time.monotonic()
    progress(started, "selection", "before_registry_precheck")
    selection = freeze_panel(registry, adaptered_games=set(adaptered_games()))
    selection_check = gate_check(
        "registry_precheck",
        REGISTRY_PATH.as_posix(),
        "three_eligible_adapter_withheld_games",
        True,
        selection["passed"],
    )
    checks.append(selection_check)
    schedule = build_schedule(selection["games"], EPISODE_SEEDS)
    atomic_json(REPO_ROOT / SCHEDULE_PATH, {"selection_receipt": selection, "rows": schedule})
    phases.append(_phase("build", phase_start, started, len(schedule)))
    progress(
        started,
        "selection",
        "after_registry_precheck",
        games=selection["games"],
        passed=selection["passed"],
    )
    if not selection["passed"]:
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
    progress(
        started,
        "live_episodes",
        "before_model_load_generation_benchmark",
        planned_units=6,
    )
    with _configured_runtime() as reused:
        session = reused.run_child_with_lease(
            resources=resources,
            schedule_path=REPO_ROOT / SCHEDULE_PATH,
            raw_dir=REPO_ROOT / RAW_DIR,
            checkpoint_path=REPO_ROOT / CHECKPOINT_PATH,
            session_path=REPO_ROOT / SESSION_PATH,
            remaining_s=EPISODE_WORK_LIMIT_S,
        )
    episodes = _complete_episode_rows(schedule, session)
    phases.append(_phase("load_generate", phase_start, started, len(episodes)))
    progress(
        started,
        "live_episodes",
        "after_model_load_generation_benchmark",
        completed_units=len(episodes),
    )

    phase_start = time.monotonic()
    historical = historical_event_rows(ledger)
    raw_panel = {
        "schema": "carnot.exp7376.raw_panel.v1",
        "schedule": schedule,
        "episodes": episodes,
        "historical_event_rows": historical,
        "historical_ledger_sha256": hashes[LEDGER_PATH.as_posix()]["sha256"],
    }
    atomic_json(REPO_ROOT / RAW_PANEL_PATH, raw_panel)
    reloaded = load_json(REPO_ROOT / RAW_PANEL_PATH) or {}
    reduction = reduce_raw_panel(
        reloaded,
        historical_rows=[
            dict(row)
            for row in reloaded.get("historical_event_rows", [])
            if isinstance(row, Mapping)
        ],
    )
    phases.append(_phase("evaluate", phase_start, started, len(episodes)))
    progress(
        started,
        "evaluation",
        "independent_raw_reduction_complete",
        capture=reduction["arc_outcome_capture_complete_score"],
        support=reduction["supervisor_support_ready_score"],
        redirects=len(reduction["redirect_rows"]),
    )
    _hash_raw_evidence(REPO_ROOT, hashes)

    private = Path(tempfile.mkdtemp(prefix="exp7376-validation-", dir="/tmp"))
    phase_start = time.monotonic()
    progress(started, "validation", "before_scoped_commands")
    scoped = _run_scoped_validation(REPO_ROOT, private / "scoped")
    progress(started, "validation", "after_scoped_commands", completed_units=len(scoped))
    progress(started, "e2e", "before_scoped_e2e")
    e2e = validation_scope.run_commands(
        REPO_ROOT,
        e2e_command_specs(REPO_ROOT, private),
        log_dir=REPO_ROOT / RAW_DIR / "validation/e2e",
        heartbeat_s=60.0,
    )
    progress(started, "e2e", "after_scoped_e2e", completed_units=len(e2e))
    receipts = [*scoped, *e2e]
    phases.append(_phase("validate", phase_start, started, len(receipts)))

    counts = _invocation_counts(session, episodes)
    runtime = deepcopy(dict(session.get("runtime_receipt") or {}))
    runtime.update(
        {
            "authenticated_model": deepcopy(resources["model_spec"]),
            "episode_ids": [row["episode_id"] for row in episodes],
            "raw_panel_sha256": canonical_hash(raw_panel),
            "single_model_instance": True,
            "board_execution": False,
        }
    )
    repository_health = validation_scope.build_repository_health([])
    candidate = build_terminal_artifact(
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        raw_panel=raw_panel,
        reduction=reduction,
        runtime_receipt=runtime,
        model_specs=[resources["model_spec"]],
        invocation_counts=counts,
        validation_receipts=receipts,
        repository_health=repository_health,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phases,
    )
    atomic_json(REPO_ROOT / TERMINAL_CANDIDATE_PATH, candidate)

    phase_start = time.monotonic()
    progress(started, "terminal_validation", "before_independent_and_strict_checks")
    terminal = validation_scope.run_commands(
        REPO_ROOT,
        terminal_command_specs(
            REPO_ROOT, REPO_ROOT / TERMINAL_CANDIDATE_PATH, REPO_ROOT / RAW_PANEL_PATH
        ),
        log_dir=REPO_ROOT / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    receipts.extend(terminal)
    phases.append(_phase("validate", phase_start, started, len(terminal)))
    progress(
        started,
        "terminal_validation",
        "after_independent_and_strict_checks",
        completed_units=len(terminal),
    )

    artifact = build_terminal_artifact(
        preconditions=checks,
        source_hashes=hashes,
        selection=selection,
        raw_panel=raw_panel,
        reduction=reduction,
        runtime_receipt=runtime,
        model_specs=[resources["model_spec"]],
        invocation_counts=counts,
        validation_receipts=receipts,
        repository_health=repository_health,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=phases,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"terminal artifact validation failed: {errors}")
    progress(started, "write", "before_atomic_terminal_write", path=RESULT_PATH)
    atomic_json(REPO_ROOT / TERMINAL_CANDIDATE_PATH, artifact)
    atomic_json(REPO_ROOT / RESULT_PATH, artifact)
    atomic_json(
        REPO_ROOT / CHECKPOINT_PATH,
        {
            "status": "complete",
            "completed_units": 6,
            "result_path": RESULT_PATH.as_posix(),
            "arc_outcome_capture_complete_score": artifact["arc_outcome_capture_complete_score"],
            "supervisor_support_ready_score": artifact["supervisor_support_ready_score"],
        },
    )
    progress(started, "write", "after_atomic_terminal_write", path=RESULT_PATH)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--reduce-raw", type=Path)
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / LEDGER_PATH)
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI integration.
    started = time.monotonic()
    progress(started, "startup", "entrypoint")
    args = parse_args(argv)
    if args.reduce_raw is not None:
        raw = load_json(args.reduce_raw) or {}
        ledger = load_json(args.ledger) or {}
        historical = [
            dict(row)
            for row in raw.get("historical_event_rows", historical_event_rows(ledger))
            if isinstance(row, Mapping)
        ]
        print(
            json.dumps(reduce_raw_panel(raw, historical_rows=historical), sort_keys=True),
            flush=True,
        )
        return 0
    if args.date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    if args.role == "live-session":
        return run_live_session(args)
    run_experiment(args)
    return 0
