"""Reduce archived ARC supervisor exposure without inventing an arm effect.

The experiment reads two completed live-policy rows from Experiment 7431. It
does not run a game or a model. Historical callbacks stay in hash-bound
sidecars so the current aggregation receipt remains zero-call provenance.

Spec refs: REQ-ARC-WMTE-7444 and SCENARIO-ARC-WMTE-7444-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    ARM_TOOL_LOOP_REINDUCTION,
)
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.652"
PHASE = 3
EXPERIMENT_ID = "exp7444-v652-arc-supervisor-evidence"
SCHEMA = "carnot.exp7444.v652.arc_supervisor_evidence.v1"
RESULT_PATH = Path("results/experiment_7444_v652_arc_supervisor_evidence.json")
RAW_DIR = Path("results/raw/experiment_7444_v652_arc_supervisor_evidence")
MODULE_PATH = Path("python/carnot/experiment_7444_v652_arc_supervisor_evidence.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7444_v652_arc_supervisor_evidence.py")
TEST_PATH = Path("tests/python/test_experiment_7444_v652_arc_supervisor_evidence.py")
SHARED_TEST_PATH = Path("tests/python/test_arc_supervisor_refinement.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
SOURCE_PATH = Path("results/experiment_7431_v651_arc_live_sentinel.json")
SOURCE_ROWS_PATH = Path("results/raw/experiment_7431_v651_arc_live_sentinel/episode_rows.json")
SOURCE_EVENTS_PATH = Path("results/raw/experiment_7431_v651_arc_live_sentinel/runtime_events.jsonl")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
NOTE_PATH = Path("docs/research-notes/v652-arc-supervisor-evidence.md")

CURATED_ARMS = tuple(ARM_ORDER)
DEFAULT_ENABLED_ARMS = (
    ARM_DROP_GOAL_BIAS,
    ARM_ALLOW_REINDUCTION,
    ARM_FORCE_DIVERSITY,
)
SHIPPED_FIRING_THRESHOLD = 120
ZERO_CURRENT_INVOCATIONS = deepcopy(current_work_receipt.ZERO_INVOCATION_COUNTS)
EXPECTED_GAMES = ("bp35", "cn04")
TERMINAL_CHECK_NAMES = (
    "fresh_process_cold_replay",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), SHARED_TEST_PATH.as_posix()),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def utc_now() -> str:
    """Return an aware UTC timestamp for a measured boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each boundary so a long validation child never looks silent."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7444] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _gate(
    check: str,
    category: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep one gate's comparison plain and independently readable."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "operator": "==",
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": bool(passed),
        "principle": principle,
    }


def _load_json(path: Path) -> JsonDict:
    """Read one JSON object and return an empty object for malformed bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Read existing event rows while rejecting malformed lines explicitly."""

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


def authenticate_upstream(source: Mapping[str, Any]) -> list[JsonDict]:
    """Authenticate the exact V651 identity, flags, and two episode rows."""

    rows = [row for row in source.get("per_game_results", []) if isinstance(row, Mapping)]
    outcomes = [row for row in source.get("supervisor_outcomes", []) if isinstance(row, Mapping)]
    games = sorted(str(row.get("game")) for row in rows)
    checks = [
        _gate(
            "source_experiment_id",
            "precondition",
            SOURCE_PATH.as_posix(),
            "experiment_id",
            "exp7431-v651-arc-live-sentinel",
            source.get("experiment_id"),
            source.get("experiment_id") == "exp7431-v651-arc-live-sentinel",
            "Only the named live sentinel can supply this archived evidence.",
        ),
        _gate(
            "source_milestone",
            "precondition",
            SOURCE_PATH.as_posix(),
            "milestone",
            "2026.09.651",
            source.get("milestone"),
            source.get("milestone") == "2026.09.651",
            "The milestone identity prevents a similarly named artifact from entering.",
        ),
        _gate(
            "source_verdict_class",
            "precondition",
            SOURCE_PATH.as_posix(),
            "verdict_class",
            "null",
            source.get("verdict_class"),
            source.get("verdict_class") == "null",
            "The source was a complete reachability null, not efficacy evidence.",
        ),
        _gate(
            "source_adversarial_flag",
            "precondition",
            SOURCE_PATH.as_posix(),
            "flagged_adversarial",
            False,
            source.get("flagged_adversarial"),
            source.get("flagged_adversarial") is False,
            "Quarantined evidence cannot support a current interpretation.",
        ),
        _gate(
            "source_status",
            "precondition",
            SOURCE_PATH.as_posix(),
            "status",
            "complete_*",
            source.get("status"),
            str(source.get("status") or "").startswith("complete_"),
            "Only a terminal producer row can be reduced as archived evidence.",
        ),
        _gate(
            "two_expected_games",
            "precondition",
            SOURCE_PATH.as_posix(),
            "per_game_results.games",
            sorted(EXPECTED_GAMES),
            games,
            games == sorted(EXPECTED_GAMES),
            "The task is limited to the two authenticated V651 episodes.",
        ),
        _gate(
            "two_supervisor_outcomes",
            "precondition",
            SOURCE_PATH.as_posix(),
            "supervisor_outcomes.episode_id",
            sorted(str(row.get("episode_id")) for row in rows),
            sorted(str(row.get("episode_id")) for row in outcomes),
            len(outcomes) == 2
            and sorted(str(row.get("episode_id")) for row in outcomes)
            == sorted(str(row.get("episode_id")) for row in rows),
            "Every episode needs its matching supervisor summary.",
        ),
    ]
    flags_ok = len(rows) == 2 and all(
        row.get("disposition") == "complete"
        and row.get("actions") == 62
        and row.get("adapter_disabled") is True
        and row.get("banked_solution_disabled") is True
        and row.get("saved_engine_disabled") is True
        and row.get("level_progress") == 0
        for row in rows
    )
    checks.append(
        _gate(
            "episode_original_flags",
            "precondition",
            SOURCE_PATH.as_posix(),
            "per_game_results.flags",
            "complete, actions=62, help disabled, level_progress=0",
            [
                {
                    "game": row.get("game"),
                    "disposition": row.get("disposition"),
                    "actions": row.get("actions"),
                    "adapter_disabled": row.get("adapter_disabled"),
                    "banked_solution_disabled": row.get("banked_solution_disabled"),
                    "saved_engine_disabled": row.get("saved_engine_disabled"),
                    "level_progress": row.get("level_progress"),
                }
                for row in rows
            ],
            flags_ok,
            "Original withholding and outcome flags must survive the aggregation.",
        )
    )
    return checks


def registry_precheck(registry: Mapping[str, Any]) -> list[JsonDict]:
    """Confirm the two public targets are already full clears before reduction."""

    games = {
        str(row.get("game")): row for row in registry.get("games", []) if isinstance(row, Mapping)
    }
    checks: list[JsonDict] = []
    for game in EXPECTED_GAMES:
        row = games.get(game, {})
        observed = {
            "present": game in games,
            "full_game_clear": row.get("full_game_clear"),
            "levels_reproduced": row.get("levels_reproduced"),
        }
        passed = (
            game in games
            and row.get("full_game_clear") is True
            and isinstance(row.get("levels_reproduced"), int)
            and int(row["levels_reproduced"]) > 0
        )
        checks.append(
            _gate(
                f"registry_precheck_{game}",
                "precondition",
                REGISTRY_PATH.as_posix(),
                f"games[{game}]",
                {"present": True, "full_game_clear": True, "levels_reproduced": ">0"},
                observed,
                passed,
                "A solved public target is an observation source, never a new solve target.",
            )
        )
    return checks


def _redirect_rows(supervisor: Mapping[str, Any]) -> list[JsonDict]:
    """Read applied or shadow trigger rows without treating shadow as applied."""

    value = supervisor.get("redirects")
    if not isinstance(value, list):
        value = supervisor.get("would_have_redirects")
    return [dict(row) for row in value or [] if isinstance(row, Mapping)]


def _window_rows(supervisor: Mapping[str, Any]) -> list[JsonDict]:
    """Return only recorded window rows; a missing list stays empty and unknown."""

    value = supervisor.get("unredirected_windows")
    return [dict(row) for row in value or [] if isinstance(row, Mapping)]


def _level_changes(observations: Sequence[Mapping[str, Any]], start_level: int) -> JsonDict:
    """Separate transient increases and decreases from terminal banked progress."""

    levels = [
        int(row["level"])
        for row in observations
        if row.get("event") == "action_observation"
        and isinstance(row.get("level"), int)
        and not isinstance(row.get("level"), bool)
    ]
    previous = start_level
    increases = 0
    decreases = 0
    for level in levels:
        increases += int(level > previous)
        decreases += int(level < previous)
        previous = level
    peak = max([start_level, *levels])
    return {
        "level_increase_events": increases,
        "level_decrease_events": decreases,
        "peak_level": peak,
        "terminal_observed_level": levels[-1] if levels else start_level,
        "transient_level_progress": max(0, peak - start_level),
    }


def _next_prerequisite(disabled_arms: Sequence[str]) -> JsonDict:
    """Name the exact exposure needed before effect or new-arm inference."""

    return {
        "for_arm_effect": (
            "Run an adapter-withheld live comparison at the shipped 120-action stagnation "
            "threshold; record one curated-arm firing, whether it was applied, and the later "
            "transient and banked outcome."
        ),
        "for_new_arm": (
            "Enable every curated arm through an explicit experiment opt-in, observe all four "
            "arms fire on one level stretch, then record a later exhausted stagnation window."
        ),
        "requires_arm_enablement": list(disabled_arms),
        "threshold_actions": SHIPPED_FIRING_THRESHOLD,
        "generation_budget_change_required": False,
        "live_policy_changed_by_this_task": False,
    }


def reduce_episode(
    episode: Mapping[str, Any],
    supervisor_sidecar: Mapping[str, Any] | None,
    observations: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce one archived episode and preserve unknown window details."""

    summary = episode.get("supervisor")
    supervisor = dict(supervisor_sidecar or (summary if isinstance(summary, Mapping) else {}))
    redirects = _redirect_rows(supervisor)
    windows = _window_rows(supervisor)
    detailed = "window" in supervisor or "actions_observed" in supervisor
    enabled = [str(arm) for arm in supervisor.get("arms_enabled", []) if isinstance(arm, str)]
    fired_by_arm = Counter(
        str(row.get("arm")) for row in redirects if row.get("arm") in CURATED_ARMS
    )
    aggregate_fired = supervisor.get("fired")
    fired = len(redirects) if redirects else int(aggregate_fired or 0)
    consumed = int(supervisor.get("consumed") or 0)
    mode = str(supervisor.get("mode") or "unknown")
    disabled = [arm for arm in CURATED_ARMS if arm not in enabled]
    actions_observed = int(supervisor.get("actions_observed") or episode.get("actions") or 0)
    threshold = int(supervisor.get("window") or SHIPPED_FIRING_THRESHOLD)
    if actions_observed < threshold:
        threshold_reached: bool | None = False
    elif redirects or windows:
        threshold_reached = True
    else:
        threshold_reached = None
    eligible_windows = len(redirects) + len(windows) if detailed else None
    fired_arms = {arm for arm, count in fired_by_arm.items() if count > 0}
    all_fired = set(CURATED_ARMS).issubset(fired_arms)
    exhausted = sum(
        set(CURATED_ARMS).issubset(set(row.get("arms_used") or []))
        and set(CURATED_ARMS).issubset(fired_arms)
        for row in windows
    )
    banked = int(episode.get("level_progress") or 0)
    if fired == 0:
        effect = "none_no_firing"
    elif mode == "shadow" or consumed == 0:
        effect = "none_shadow_trigger_not_intervention"
    elif banked > 0:
        effect = "measured_progress_after_intervention"
    else:
        effect = "measured_failed_intervention"
    arm_rows = []
    for arm in CURATED_ARMS:
        arm_fired: int | None
        if redirects or fired == 0:
            arm_fired = int(fired_by_arm[arm])
        else:
            arm_fired = None
        arm_rows.append(
            {
                "arm": arm,
                "enabled": arm in enabled,
                "fired": arm_fired,
                "consumed": None if arm_fired is None else int(mode != "shadow" and arm_fired > 0),
                "effect_evidence": (
                    "none"
                    if not arm_fired or mode == "shadow"
                    else "measured_progress"
                    if banked > 0
                    else "measured_failed_intervention"
                ),
            }
        )
    request = episode.get("request_budget_receipt")
    request_receipt = dict(request) if isinstance(request, Mapping) else {}
    level_changes = _level_changes(observations, int(episode.get("start_level") or 0))
    return {
        "unit_id": str(episode.get("episode_id") or episode.get("game") or "unknown"),
        "source_group": "v651_live_scored_policy_adapter_withheld",
        "condition": "archived_shadow_supervisor",
        "game": episode.get("game"),
        "seed": episode.get("seed"),
        "disposition": episode.get("disposition"),
        "actions": episode.get("actions"),
        "adapter_disabled": episode.get("adapter_disabled"),
        "banked_solution_disabled": episode.get("banked_solution_disabled"),
        "saved_engine_disabled": episode.get("saved_engine_disabled"),
        "callbacks_attempted": int(request_receipt.get("attempted") or 0),
        "completions_received": int(request_receipt.get("completed") or 0),
        "callbacks_failed": int(request_receipt.get("failed") or 0),
        "callbacks_cancelled": int(request_receipt.get("cancelled") or 0),
        "callbacks_in_flight": int(request_receipt.get("in_flight") or 0),
        "tool_feedback_consumed": int(episode.get("tool_feedback_consumed") or 0),
        "supervisor_mode": mode,
        "enabled_arms": enabled,
        "disabled_arms": disabled,
        "supervisor_firings": fired,
        "consumed_redirects": consumed,
        "arm_rows": arm_rows,
        "arm_effect_evidence": effect,
        "measured_failed_intervention": effect == "measured_failed_intervention",
        "shipped_firing_threshold": threshold,
        "firing_threshold_reached": threshold_reached,
        "eligible_stagnation_windows": eligible_windows,
        "window_evidence_status": (
            "recorded_timestamped_window_receipt"
            if detailed
            else "unknown_missing_timestamped_window_receipt"
        ),
        "all_curated_arms_fired": all_fired,
        "all_arms_exhausted_windows": exhausted,
        "new_arm_evidence": bool(all_fired and exhausted > 0),
        "banked_progress": banked,
        **level_changes,
        "supervisor_helped_banked_progress": bool(consumed > 0 and banked > 0),
        "solve_credit": 0,
        "new_level_credit": 0,
        "automatic_policy_change": False,
        "next_live_prerequisite": _next_prerequisite(disabled),
        "verdict_class": "null",
    }


def reduce_private_ledger(
    episodes: Sequence[Mapping[str, Any]],
    supervisor_sidecars: Mapping[str, Mapping[str, Any]],
    observations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    """Reduce the private two-episode ledger without writing the shared ledger."""

    rows = [
        reduce_episode(
            episode,
            supervisor_sidecars.get(str(episode.get("episode_id"))),
            observations.get(str(episode.get("episode_id")), ()),
        )
        for episode in episodes
    ]
    firings = sum(int(row["supervisor_firings"]) for row in rows)
    new_arm = any(row["new_arm_evidence"] for row in rows)
    recommendations = (
        [
            {
                "action": "consider_human_curated_new_arm",
                "basis": "all_four_curated_arms_fired_then_stagnation_continued",
                "automatic_change": False,
            }
        ]
        if new_arm
        else []
    )
    return {
        "schema": "carnot.exp7444.private_supervisor_ledger.v1",
        "episode_count": len(rows),
        "rows": rows,
        "callbacks_attempted": sum(int(row["callbacks_attempted"]) for row in rows),
        "completions_received": sum(int(row["completions_received"]) for row in rows),
        "tool_feedback_consumed": sum(int(row["tool_feedback_consumed"]) for row in rows),
        "supervisor_firings": firings,
        "consumed_redirects": sum(int(row["consumed_redirects"]) for row in rows),
        "measured_failed_interventions": sum(
            int(row["measured_failed_intervention"]) for row in rows
        ),
        "transient_level_increases": sum(int(row["level_increase_events"]) for row in rows),
        "banked_progress": sum(int(row["banked_progress"]) for row in rows),
        "new_arm_evidence": new_arm,
        "supervisor_recommendations": recommendations,
        "outcome_evidence_count": sum(int(row["consumed_redirects"] > 0) for row in rows),
    }


def archived_episode_payload(
    episode: Mapping[str, Any],
    supervisor_outcome: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    source_path: Path,
) -> JsonDict:
    """Build a typed archive payload that binds exact source bytes and fields."""

    return {
        "schema": "carnot.exp7444.archived_episode.v1",
        "scope": "historical_model_receipts",
        "source": {
            "path": source_path.as_posix(),
            "sha256": current_work_receipt.sha256_file(source_path),
        },
        "episode_receipt": deepcopy(dict(episode)),
        "trajectory_supervisor_receipt": deepcopy(dict(supervisor_outcome)),
        "trajectory_window_receipt": None,
        "trajectory_window_receipt_status": "unknown_not_preserved_by_v651_terminal_row",
        "timestamped_observations": [deepcopy(dict(row)) for row in observations],
    }


def current_aggregation_receipt(
    *,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    sidecar_references: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build the zero-model receipt for this host aggregation."""

    return current_work_receipt.build_current_work_receipt(
        run_id=EXPERIMENT_ID,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={
            "cpu": platform.processor() or platform.machine(),
            "cuda_used": False,
            "external_device_used": False,
            "model_load_performed": False,
        },
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_monotonic_ns,
        ended_monotonic_ns=ended_monotonic_ns,
        sidecar_references=sidecar_references,
        phase_spans=phase_spans,
        small_ebm_training={"performed": False},
    )


def _sample_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all unit dispositions explicit even for this fixed two-row audit."""

    counts = Counter(str(row.get("disposition")) for row in rows)
    return {
        "planned_units": 2,
        "attempted_units": len(rows),
        "completed_units": counts["complete"],
        "failed_units": sum(count for key, count in counts.items() if "error" in key),
        "censored_units": sum(count for key, count in counts.items() if "censored" in key),
        "unstarted_units": 2 - len(rows),
        "independent_groups": ["archived_v651_public_adapter_withheld_development_proxy"],
        "stopping_rule": "Reduce exactly the two authenticated V651 episodes; do not accrue new runs.",
    }


def _field_principles() -> JsonDict:
    """Explain required top-level fields without wrapping their scalar values."""

    return {
        "schema": "A versioned plain schema keeps terminal identity machine-readable.",
        "run_date": "The requested date is separate from measured UTC boundaries.",
        "preconditions_checked": "Exact resources and observed values stop fabricated inputs.",
        "MODEL_SPECS": "An empty list states that this aggregation invoked no current LLM.",
        "model_invoked": "Current attempts remain distinct from archived callbacks.",
        "invocation_counts": "Attempted and terminal states reconcile current model work.",
        "inference_substrate": "The string names the work while details name devices.",
        "inference_substrate_class": "Aggregation uses the matching duration floor.",
        "execution_venue": "Host execution is separate from CPU, CUDA, and device details.",
        "duration_s": "Measured current time separates aggregation and validation cost.",
        "phase_spans": "Phase boundaries bind progress events and completed units.",
        "random_seed": "Null is correct because no fitting, sampling, or resampling occurred.",
        "reproducibility_checksum": "The hash binds protocol, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Byte hashes preserve source identity and original flags.",
        "rows": "Per-game rows preserve failures, exposures, and unknown evidence.",
        "sample_size_budget": "Disposition counts prevent two rows from implying a larger study.",
        "acceptance_gate_results": "Validity gates remain distinct from a benefit gate.",
        "gate_check_summary": "A failed prerequisite names its exact missing or changed value.",
        "verifier_is_oracle": "The deployed environment level signal is the scoring authority.",
        "honest_verdict": "A complete null reports no firing evidence without inventing failure.",
        "verdict_class": "Null is terminal science; partial is reserved for owned unfinished work.",
        "flagged_adversarial": "Critical findings quarantine science from readiness claims.",
        "validation_receipts": "Exact commands, exits, durations, and log hashes remain auditable.",
        "field_principles": "Field intent stays separate from plain gate scalars.",
        "promotion_score": "This milestone authorizes no rollout or policy change.",
        "per_game_results": "Two dispositions preserve mode, exposures, actions, and progress.",
        "supervisor_recommendations": "An empty outcome ledger yields no policy change.",
        "solve_provenance": "The label describes archived origin and grants no current credit.",
        "solve_credit": "No game ran and no level was reproduced by this task.",
        "next_live_prerequisite": "The next comparison needs an observed firing and outcome.",
    }


def gate_check_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed prerequisite while keeping every later gate visible."""

    failed = [row for row in gates if row.get("passed") is not True]
    if not failed:
        return {"passed": True, "failed_count": 0, "first_failure": None}
    first = failed[0]
    return {
        "passed": False,
        "failed_count": len(failed),
        "first_failure": {
            key: deepcopy(first.get(key))
            for key in ("upstream", "check", "artifact_field", "expected", "observed")
        },
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind stable evidence and exact validation scope, excluding run timing."""

    payload = {
        "schema": artifact.get("schema"),
        "experiment_id": artifact.get("experiment_id"),
        "milestone": artifact.get("milestone"),
        "source_artifact_hashes": artifact.get("source_artifact_hashes"),
        "rows": artifact.get("rows"),
        "receipt_sidecars": artifact.get("receipt_sidecars"),
        "affected_file_manifest": artifact.get("affected_file_manifest"),
        "required_check_names": artifact.get("required_check_names"),
        "protocol": "REQ-ARC-WMTE-7444",
    }
    return validation_contract.canonical_hash(payload)


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one passing, non-timeout, zero-exit receipt per named command."""

    return all(
        len(rows := [row for row in receipts if row.get("name") == name]) == 1
        and rows[0].get("passed") is True
        and rows[0].get("exit_code") == 0
        and rows[0].get("timed_out") is False
        for name in names
    )


def _build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    current_receipt: Mapping[str, Any],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    ended_at_utc: str,
    phase_spans: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one terminal-shaped artifact from already reduced evidence."""

    ledger = reduce_private_ledger([], {}, {})
    ledger_rows = [deepcopy(dict(row)) for row in rows]
    ledger.update(
        {
            "episode_count": len(ledger_rows),
            "rows": ledger_rows,
            "callbacks_attempted": sum(int(row["callbacks_attempted"]) for row in ledger_rows),
            "completions_received": sum(int(row["completions_received"]) for row in ledger_rows),
            "tool_feedback_consumed": sum(
                int(row["tool_feedback_consumed"]) for row in ledger_rows
            ),
            "supervisor_firings": sum(int(row["supervisor_firings"]) for row in ledger_rows),
            "consumed_redirects": sum(int(row["consumed_redirects"]) for row in ledger_rows),
            "measured_failed_interventions": sum(
                int(row["measured_failed_intervention"]) for row in ledger_rows
            ),
            "transient_level_increases": sum(
                int(row["level_increase_events"]) for row in ledger_rows
            ),
            "banked_progress": sum(int(row["banked_progress"]) for row in ledger_rows),
            "new_arm_evidence": any(row["new_arm_evidence"] for row in ledger_rows),
        }
    )
    ledger["supervisor_recommendations"] = (
        [
            {
                "action": "consider_human_curated_new_arm",
                "basis": "all_four_curated_arms_fired_then_stagnation_continued",
                "automatic_change": False,
            }
        ]
        if ledger["new_arm_evidence"]
        else []
    )
    ledger["outcome_evidence_count"] = sum(
        int(row["consumed_redirects"] > 0) for row in ledger_rows
    )
    affected_ok = _required_receipts_pass(
        validation_receipts, validation_scope.REQUIRED_CHECK_NAMES
    )
    terminal_ok = not require_terminal or _required_receipts_pass(
        validation_receipts, TERMINAL_CHECK_NAMES
    )
    gates = [deepcopy(dict(row)) for row in preconditions]
    gates.extend(
        [
            _gate(
                "two_episode_reduction",
                "validity",
                "private_supervisor_ledger",
                "episode_count",
                2,
                ledger["episode_count"],
                ledger["episode_count"] == 2,
                "The result must preserve both scheduled V651 units.",
            ),
            _gate(
                "no_arm_effect_evidence",
                "benefit",
                "private_supervisor_ledger",
                "supervisor_firings",
                0,
                ledger["supervisor_firings"],
                ledger["supervisor_firings"] == 0,
                "Zero firings support no arm-effect or failed-intervention claim.",
            ),
            _gate(
                "avo_boundary",
                "safety",
                "private_supervisor_ledger",
                "supervisor_recommendations",
                [],
                ledger["supervisor_recommendations"],
                ledger["supervisor_recommendations"] == [],
                "An empty outcome ledger cannot promote, retire, or generate an arm.",
            ),
            _gate(
                "affected_validation",
                "validation",
                "validation_receipts",
                "required_affected_checks",
                True,
                affected_ok,
                affected_ok,
                "Only the frozen affected scope determines implementation validity.",
            ),
            _gate(
                "terminal_readers",
                "validation",
                "validation_receipts",
                "required_terminal_checks",
                True,
                terminal_ok,
                terminal_ok,
                "Cold reduction and unchanged readers must accept the candidate.",
            ),
        ]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "complete_null_no_supervisor_firings_no_policy_change",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        **deepcopy(dict(current_receipt)),
        "duration_breakdown_s": {
            str(row.get("phase")): round(
                float(row.get("end_s") or 0.0) - float(row.get("start_s") or 0.0), 6
            )
            for row in phase_spans
        },
        "random_seed": None,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": ledger_rows,
        "per_game_results": ledger_rows,
        "private_supervisor_ledger": ledger,
        "sample_size_budget": _sample_budget(ledger_rows),
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_check_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": "complete_null_no_supervisor_firings_no_arm_effect_evidence",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_check_names": [
            *validation_scope.REQUIRED_CHECK_NAMES,
            *TERMINAL_CHECK_NAMES,
        ],
        "affected_file_manifest": {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
        "repository_health": {
            "status": "scoped_only",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
        "field_principles": _field_principles(),
        "promotion_score": 0,
        "supervisor_evidence_complete_score": int(all(row.get("passed") is True for row in gates)),
        "supervisor_recommendations": deepcopy(ledger["supervisor_recommendations"]),
        "next_live_prerequisite": _next_prerequisite(
            sorted({arm for row in ledger_rows for arm in row["disabled_arms"]})
        ),
        "solve_provenance": "live_agent_self_discovery_archived_attempt_origin_only",
        "solve_credit": 0,
        "new_level_credit": 0,
        "new_level_reproduction_required": False,
        "current_solve_claimed": False,
        "generator_weights_changed": False,
        "production_defaults_changed": False,
        "supervisor_arm_added": False,
        "supervisor_order_changed": False,
        "solve_registry_changed": False,
        "research_conductor_changed": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_fixture(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build a deterministic terminal-shaped object for private reducer tests."""

    receipt = current_aggregation_receipt(
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        sidecar_references=[],
        phase_spans=[],
    )
    receipt["current_owner_pid"] = os.getpid()
    return _build_artifact(
        rows=rows,
        current_receipt=receipt,
        preconditions=[],
        source_hashes={},
        validation_receipts=[],
        started_at_utc="2026-09-20T00:00:00+00:00",
        ended_at_utc="2026-09-20T00:00:00+00:00",
        phase_spans=[],
        require_terminal=False,
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute terminal claims from raw per-game rows only."""

    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    firings = sum(int(row.get("supervisor_firings") or 0) for row in rows)
    consumed = sum(int(row.get("consumed_redirects") or 0) for row in rows)
    banked = sum(int(row.get("banked_progress") or 0) for row in rows)
    failed = sum(int(row.get("measured_failed_intervention") is True) for row in rows)
    new_arm = any(row.get("new_arm_evidence") is True for row in rows)
    recommendations = artifact.get("supervisor_recommendations")
    declared = artifact.get("private_supervisor_ledger")
    declared_ledger = dict(declared) if isinstance(declared, Mapping) else {}
    matches = (
        len(rows) == 2
        and declared_ledger.get("episode_count") == len(rows)
        and declared_ledger.get("supervisor_firings") == firings
        and declared_ledger.get("consumed_redirects") == consumed
        and declared_ledger.get("banked_progress") == banked
        and declared_ledger.get("measured_failed_interventions") == failed
        and declared_ledger.get("new_arm_evidence") is new_arm
        and (recommendations == [] if firings == 0 else True)
    )
    return {
        "episode_count": len(rows),
        "supervisor_firings": firings,
        "consumed_redirects": consumed,
        "banked_progress": banked,
        "measured_failed_interventions": failed,
        "new_arm_evidence": new_arm,
        "matches_declared": matches,
    }


def validate_artifact(
    artifact: Mapping[str, Any], *, root: Path, require_terminal: bool
) -> list[str]:
    """Cold-check identity, provenance, rows, AVO boundaries, and checksum."""

    errors: list[str] = []
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id_mismatch")
    if artifact.get("milestone") != MILESTONE or artifact.get("run_date") != RUN_DATE:
        errors.append("run_identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("current_model_boundary_invalid")
    if artifact.get("invocation_counts") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    receipt_errors = current_work_receipt.validate_current_work_receipt(artifact, root=root)
    errors.extend(f"current_receipt:{item}" for item in receipt_errors)
    reduced = independent_reduce(artifact)
    if reduced["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    if reduced["supervisor_firings"] == 0 and artifact.get("supervisor_recommendations") != []:
        errors.append("zero_firing_recommendation_forbidden")
    if any(
        artifact.get(field) != 0
        for field in ("promotion_score", "solve_credit", "new_level_credit")
    ):
        errors.append("credit_or_promotion_nonzero")
    if artifact.get("production_defaults_changed") is not False:
        errors.append("production_default_change_forbidden")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    if require_terminal and not _required_receipts_pass(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_receipts_invalid")
    return list(dict.fromkeys(errors))


def _phase(
    spans: list[JsonDict],
    name: str,
    phase_started: float,
    run_started: float,
    completed_units: int,
) -> None:
    """Append one monotonic phase span with its finished-unit checkpoint."""

    spans.append(
        {
            "phase": name,
            "start_s": round(phase_started - run_started, 6),
            "end_s": round(time.monotonic() - run_started, 6),
            "completed_units": completed_units,
        }
    )


def _source_hash_row(path: Path, role: str, **extra: Any) -> JsonDict:
    """Hash one exact input and retain its role plus original flags."""

    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": current_work_receipt.sha256_file(path),
        "role": role,
        **deepcopy(extra),
    }


def _write_note(path: Path) -> None:
    """Write the short research interpretation in plain technical language."""

    text = """# V652 ARC supervisor evidence

Experiment 7444 reads the two completed Experiment 7431 episodes. It does not
run a game or invoke a model. Both archived episodes used the scored policy with
per-game adapters and saved solutions disabled. Each recorded 62 actions, zero
banked progress, shadow supervisor mode, and zero supervisor firings.

The shipped firing threshold is 120 stagnant actions. The two 62-action rows
cannot reach that threshold. Experiment 7431 preserved a supervisor summary but
did not preserve detailed timestamped window rows in its terminal episode row.
The missing window contents therefore remain unknown. No window was invented.

Zero firings provide no evidence that an arm helped or failed. No arm is
promoted or retired. The fourth curated arm, `tool_loop_reinduction`, was not
enabled. A new arm is not justified. That conclusion would require all four
curated arms to fire on one level stretch and a later recorded exhausted window.

The next live comparison must keep adapters withheld, use the shipped 120-action
threshold, record an actual curated-arm firing, state whether it was applied,
and retain the later transient and banked outcome. The live policy remains
unchanged.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _terminal_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build fresh-process readers for the exact measured candidate."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "fresh_process_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact measured candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact measured candidate",
            300.0,
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
            300.0,
        ),
    ]


def _blocked_artifact(
    checks: Sequence[Mapping[str, Any]], started_at: str, started_ns: int
) -> JsonDict:  # pragma: no cover - exercised only when an external input disappears.
    """Publish missing external evidence as blocked without fabricated rows."""

    ended_ns = time.monotonic_ns()
    receipt = current_aggregation_receipt(
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=[],
        phase_spans=[],
    )
    artifact = _build_artifact(
        rows=[],
        current_receipt=receipt,
        preconditions=checks,
        source_hashes={},
        validation_receipts=[],
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        phase_spans=[],
        require_terminal=False,
    )
    artifact.update(
        {
            "status": "blocked_missing_external_prerequisite",
            "honest_verdict": "blocked_missing_external_prerequisite",
            "verdict_class": "blocked",
            "supervisor_evidence_complete_score": 0,
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - CLI orchestration.
    """Authenticate, reduce, validate, and atomically publish the V652 artifact."""

    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    progress(started, "preconditions", "start", completed_units=0)
    required_paths = (
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        SPEC_PATH,
        SOURCE_PATH,
        SOURCE_ROWS_PATH,
        SOURCE_EVENTS_PATH,
        REGISTRY_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        SHARED_TEST_PATH,
    )
    path_checks = [
        _gate(
            f"path_{path.as_posix()}",
            "precondition",
            path.as_posix(),
            "path",
            "readable_file",
            "readable_file" if (root / path).is_file() else "missing",
            (root / path).is_file(),
            "Dependent work starts only after its exact source exists.",
        )
        for path in required_paths
    ]
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    path_checks.append(
        _gate(
            "capability_requirement",
            "precondition",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7444",
            "present",
            "present" if "REQ-ARC-WMTE-7444" in spec_text else "missing",
            "REQ-ARC-WMTE-7444" in spec_text,
            "Implementation needs a requirement and scenarios before behavior changes.",
        )
    )
    source = _load_json(root / SOURCE_PATH)
    path_checks.extend(authenticate_upstream(source))
    try:
        registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        registry_value = {}
    registry = dict(registry_value) if isinstance(registry_value, Mapping) else {}
    path_checks.extend(registry_precheck(registry))
    excluded_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = "experiment_id: 7444" in excluded_text
    path_checks.append(
        _gate(
            "task_not_excluded",
            "precondition",
            "ops/exclusion_manifest.yaml",
            "experiment_id: 7444",
            False,
            excluded,
            not excluded,
            "A retired task must not silently run again.",
        )
    )
    _phase(spans, "preconditions", started, started, len(path_checks))
    if not all(row["passed"] for row in path_checks):
        artifact = _blocked_artifact(path_checks, started_at, started_ns)
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        progress(started, "preconditions", "blocked", completed_units=len(path_checks))
        return artifact

    phase_started = time.monotonic()
    progress(started, "sidecars", "start", completed_units=0)
    source_rows_value = _load_json(root / SOURCE_ROWS_PATH)
    episode_rows = [
        dict(row) for row in source_rows_value.get("rows", []) if isinstance(row, Mapping)
    ]
    outcomes = {
        str(row.get("episode_id")): dict(row)
        for row in source.get("supervisor_outcomes", [])
        if isinstance(row, Mapping)
    }
    events = _read_jsonl(root / SOURCE_EVENTS_PATH)
    references: list[JsonDict] = []
    observations: dict[str, list[JsonDict]] = {}
    for index, episode in enumerate(episode_rows):
        episode_id = str(episode.get("episode_id"))
        episode_events = [row for row in events if row.get("episode_id") == episode_id]
        observations[episode_id] = episode_events
        payload = archived_episode_payload(
            episode,
            outcomes.get(episode_id, {}),
            episode_events,
            source_path=root / SOURCE_PATH,
        )
        sidecar_path = root / RAW_DIR / "sidecars" / f"v651_{episode.get('game')}.json"
        references.append(
            current_work_receipt.write_immutable_sidecar(
                sidecar_path,
                scope="historical_model_receipts",
                payload=payload,
                root=root,
            )
        )
        progress(
            started,
            "sidecars",
            "unit_complete",
            completed_units=index + 1,
            game=episode.get("game"),
        )
    _phase(spans, "sidecars", phase_started, started, len(references))

    phase_started = time.monotonic()
    progress(started, "reduction", "start", completed_units=0)
    ledger = reduce_private_ledger(episode_rows, {}, observations)
    reduced_rows = ledger["rows"]
    per_game_hashes: dict[str, JsonDict] = {}
    for index, row in enumerate(reduced_rows):
        game = str(row["game"])
        per_game_path = root / RAW_DIR / "per_game" / f"{game}.json"
        current_work_receipt.atomic_json(
            per_game_path,
            {
                "schema": "carnot.exp7444.per_game.v1",
                "experiment_id": EXPERIMENT_ID,
                "source_artifact": SOURCE_PATH.as_posix(),
                "row": row,
            },
        )
        per_game_hashes[per_game_path.relative_to(root).as_posix()] = _source_hash_row(
            per_game_path, "per_game_reduction"
        )
        progress(
            started,
            "reduction",
            "unit_complete",
            completed_units=index + 1,
            game=game,
        )
    _write_note(root / NOTE_PATH)
    _phase(spans, "reduction", phase_started, started, len(reduced_rows))

    source_hashes: dict[str, JsonDict] = {}
    for path, role in (
        (SOURCE_PATH, "archived_terminal_input"),
        (SOURCE_ROWS_PATH, "archived_episode_rows"),
        (SOURCE_EVENTS_PATH, "archived_timestamped_events"),
        (REGISTRY_PATH, "registry_precheck"),
        (SPEC_PATH, "capability_spec"),
        (MODULE_PATH, "current_code"),
        (WRAPPER_PATH, "current_entrypoint"),
        (TEST_PATH, "current_tests"),
        (SHARED_TEST_PATH, "affected_shared_tests"),
        (NOTE_PATH, "research_note"),
    ):
        extra: JsonDict = {}
        if path == SOURCE_PATH:
            extra["original_flags"] = {
                "status": source.get("status"),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
            }
        source_hashes[path.as_posix()] = _source_hash_row(root / path, role, **extra)
    source_hashes.update(per_game_hashes)

    phase_started = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses", completed_units=0)
    private_root = Path(tempfile.mkdtemp(prefix="exp7444-validation-", dir="/tmp"))
    plan = validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, plan)
    if plan_errors:
        raise RuntimeError(f"validation command plan drift: {plan_errors}")
    affected = validation_contract.run_categorized_commands(
        root,
        [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in plan
        ],
        log_dir=root / RAW_DIR / "validation/affected",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "validation",
        "after_affected_subprocesses",
        completed_units=len(affected),
    )
    _phase(spans, "affected_validation", phase_started, started, len(affected))

    ended_ns = time.monotonic_ns()
    current_receipt = current_aggregation_receipt(
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
        sidecar_references=references,
        phase_spans=spans,
    )
    candidate = _build_artifact(
        rows=reduced_rows,
        current_receipt=current_receipt,
        preconditions=path_checks,
        source_hashes=source_hashes,
        validation_receipts=affected,
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        phase_spans=spans,
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"measured candidate invalid: {candidate_errors}")
    candidate_path = root / RAW_DIR / "measured_terminal_candidate.json"
    current_work_receipt.atomic_json(candidate_path, candidate)

    phase_started = time.monotonic()
    progress(started, "terminal_validation", "before_subprocesses", completed_units=0)
    terminal = validation_scope.run_commands(
        root,
        _terminal_specs(root, candidate_path),
        log_dir=root / RAW_DIR / "validation/terminal",
        heartbeat_s=60.0,
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )
    _phase(spans, "terminal_validation", phase_started, started, len(terminal))
    final_receipt = current_aggregation_receipt(
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        sidecar_references=references,
        phase_spans=spans,
    )
    artifact = _build_artifact(
        rows=reduced_rows,
        current_receipt=final_receipt,
        preconditions=path_checks,
        source_hashes=source_hashes,
        validation_receipts=[*affected, *terminal],
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        phase_spans=spans,
        require_terminal=True,
    )
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal artifact invalid: {errors}")
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(started, "publish", "terminal_published", completed_units=2, path=RESULT_PATH)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public entrypoint's run date or read-only replay path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the aggregation or independently replay one measured candidate."""

    args = parse_args(argv)
    if args.replay is not None:
        artifact = _load_json(args.replay)
        reduced = independent_reduce(artifact)
        errors = validate_artifact(artifact, root=REPO_ROOT, require_terminal=False)
        print(
            json.dumps({"reduced": reduced, "validation_errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors))
    if args.date != RUN_DATE:  # pragma: no cover - guarded CLI misuse.
        raise SystemExit(f"run date must be {RUN_DATE}")
    run_experiment(REPO_ROOT, args.date)  # pragma: no cover - CLI orchestration.
    return 0  # pragma: no cover - reached only by the public orchestration path.


if __name__ == "__main__":  # pragma: no cover - thin module CLI.
    raise SystemExit(main())
