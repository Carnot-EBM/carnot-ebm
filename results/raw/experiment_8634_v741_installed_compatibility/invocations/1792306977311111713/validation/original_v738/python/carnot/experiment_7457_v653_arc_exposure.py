"""Measure paired ARC supervisor exposure at the shipped 120-action boundary.

The experiment keeps the production policy unchanged. It runs the same scored
policy twice per game and seed: once with the supervisor in shadow mode and
once with its existing opt-in application flag. Synthetic hook qualification
proves the boundary before the live rows start, but never enters live metrics.

Spec refs: REQ-ARC-WMTE-7457 and SCENARIO-ARC-WMTE-7457-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot import experiment_7376_v647_arc_outcomes as shipped_live
from carnot import experiment_7431_v651_arc_live_sentinel as live_support
from carnot.agentic import arc_competition_agent as competition_agent
from carnot.agentic.arc_inference_boundary import BOUNDARY_LEDGER_ENV, InvocationBoundaryLedger
from carnot.agentic.arc_request_budget import attach_request_budget
from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER, enabled_arms
from carnot.reporting import current_work_receipt
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
PHASE = 3
EXPERIMENT_ID = "exp7457-v653-arc-exposure"
TASK_ID = "experiment_7457_v653_arc_exposure"
SCHEMA = "carnot.exp7457.v653.arc_exposure.v1"

MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
MODEL_SPECS = [MODEL_ID]
INFERENCE_SUBSTRATE_CLASS = "model_bounded_generation"
EXECUTION_VENUE = "host"
SUPERVISOR_THRESHOLD = 120
ACTION_LIMIT = 180
EPISODE_LIMIT_S = 240.0
AGGREGATE_LIVE_LIMIT_S = 2400.0
MODEL_LOAD_LIMIT_S = 600.0
REQUEST_LIMIT = 2
MAX_NEW_TOKENS = 256
EPISODE_SEEDS = (7_457_001, 7_457_002)
EXPECTED_GAMES = ("bp35", "cn04")
CURATED_ARMS = tuple(ARM_ORDER)
# The fourth shipped arm is deliberately default-off. This tuple captures the
# arms that the live comparison can fire without changing that production gate.
ENABLED_ARMS = tuple(arm for arm in CURATED_ARMS if arm in enabled_arms())

SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PATH = Path("ops/e2e-test-plan.md")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
UPSTREAM_PATH = Path("results/experiment_7448_v653_capture_lifecycle.json")
RESULT_PATH = Path("results/experiment_7457_v653_arc_exposure.json")
RAW_DIR = Path("results/raw/experiment_7457_v653_arc_exposure")
SCHEDULE_PATH = RAW_DIR / "frozen_schedule.json"
SESSION_PATH = RAW_DIR / "live_session.json"
BOUNDARY_PATH = RAW_DIR / "current_invocation_events.jsonl"
RUNTIME_EVENT_PATH = RAW_DIR / "runtime_events.jsonl"
ACTION_PATH = RAW_DIR / "live_action_rows.jsonl"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7457_v653_arc_exposure.json")
TERMINAL_CANDIDATE_PATH = RAW_DIR / "terminal_candidate.json"
MODULE_PATH = Path("python/carnot/experiment_7457_v653_arc_exposure.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7457_v653_arc_exposure.py")
TEST_PATH = Path("tests/python/test_experiment_7457_v653_arc_exposure.py")
SHARED_TEST_PATHS = (
    Path("tests/python/test_arc_trajectory_supervisor.py"),
    Path("tests/python/test_arc_request_budget.py"),
    Path("tests/python/test_arc_induction_state_persistence.py"),
    Path("tests/python/test_arc_tool_grammar_transport.py"),
)

REQUIRED_E2E = ("e2e_009", "e2e_010", "e2e_009_llm_off_environment")
REQUIRED_TERMINAL = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
TERMINAL_DISPOSITIONS = {
    "complete",
    "complete_error",
    "failed",
    "censored_timeout",
    "censored_aggregate_limit",
    "censored_no_first_action",
    "unstarted",
}
WITHHELD_INPUTS = (
    "per_game_adapter",
    "stored_engine",
    "banked_solution",
    "saved_route",
    "hand_model",
    "hand_solver",
    "hidden_game_source",
    "offline_ground_truth_bfs",
)

VALIDATION_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(), *(path.as_posix() for path in SHARED_TEST_PATHS)),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    E2E_PATH,
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7431_v651_arc_live_sentinel.py"),
    Path("python/carnot/experiment_7444_v652_arc_supervisor_evidence.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_request_budget.py"),
    REGISTRY_PATH,
    UPSTREAM_PATH,
    SPEC_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
    *SHARED_TEST_PATHS,
)

FIELD_PRINCIPLES = {
    "schema": "Use a versioned schema with the exact experiment, milestone, and terminal state.",
    "run_date": "Use 20260920 and retain real UTC and monotonic clock identities.",
    "preconditions_checked": "Name exact paths, identities, and observed prerequisite values.",
    "MODEL_SPECS": "Name Qwen3.8-27B-GGUF only when current model work was attempted.",
    "model_invoked": "Separate an attempted current model call from synthetic hook events.",
    "invocation_counts": "Balance attempted, completed, failed, cancelled, and in-flight calls.",
    "inference_substrate": "Describe current host inference without importing historical evidence.",
    "inference_substrate_class": "Use bounded generation, load-only, or no-load from raw events.",
    "execution_venue": "Use host and keep the CUDA device identity in a typed detail field.",
    "duration_s": "Measure current work and never pad a duration floor.",
    "phase_spans": "Bind timings, progress boundaries, and completed-unit checkpoints.",
    "random_seed": "Freeze both episode seeds and explain the absent resampling seed.",
    "reproducibility_checksum": "Bind code, protocol, immutable inputs, rows, and validation scope.",
    "source_artifact_hashes": "Preserve exact bytes and upstream flags without rehabilitation.",
    "rows": "Keep every live disposition, including censored and unstarted episodes.",
    "sample_size_budget": "Separate all planned, attempted, completed, failed, censored, and unstarted units.",
    "acceptance_gate_results": "Keep validity and benefit gates typed with exact operands.",
    "gate_check_summary": "Name every failed gate and the first exact blocking field.",
    "verifier_is_oracle": "True because the public environment supplies the level authority.",
    "honest_verdict": "Use a complete finding only after current work reaches a terminal state.",
    "verdict_class": "Use the closed terminal verdict vocabulary.",
    "flagged_adversarial": "Preserve critical findings because flagged evidence cannot supply readiness.",
    "validation_receipts": "Record exact commands, environments, exits, durations, and log hashes.",
    "field_principles": "Explain field intent separately from machine-readable values.",
    "promotion_score": "Remain zero because this pilot authorizes no rollout.",
    "arc_exposure_complete_score": "Require eight dispositions and separate valid synthetic evidence.",
    "per_game_results": "Pair shadow and applied rows by game and seed with exposure denominators.",
    "solve_provenance": "Use self-discovery only for a live level whose own trace reproduced.",
    "new_level_credit": "Remain zero because both public development games are already registered.",
    "supervisor_exposure_rows": "Name the 120-action threshold and whether each redirect applied.",
    "arm_value_score": "Remain zero because eight exposure episodes cannot promote an arm.",
}


def utc_now() -> str:
    """Return one aware UTC boundary for the durable experiment record."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase and long-operation boundary with current progress."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7457] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON so any terminal-field change moves the identity."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash exact file bytes in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to an empty mapping."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def artifact_checksum(value: Mapping[str, Any]) -> str:
    """Bind every terminal field except the self-referential checksum."""

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
    """Keep one gate as bare values plus its interpretation."""

    return {
        "check": check,
        "category": category,
        "op": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": _compare(operator, expected, observed),
        "upstream": upstream,
        "path": upstream,
        "field": artifact_field,
        "artifact_field": artifact_field,
        "principle": principle,
    }


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all failures and expose the first exact blocker."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "all_passed": not failures,
        "failed_count": len(failures),
        "failed_checks": failures,
        "first_failure": failures[0] if failures else None,
    }


def _source_record(
    path: Path, role: str, original_flags: Mapping[str, Any] | None = None
) -> JsonDict:
    row: JsonDict = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "role": role,
    }
    if original_flags is not None:
        row["original_flags"] = deepcopy(dict(original_flags))
    return row


def collect_preconditions(
    root: Path, *, force_live: str | None = None
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Authenticate exact inputs, Exp7448 fields, and the solve registry."""

    gates: list[JsonDict] = []
    hashes: JsonDict = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        gates.append(
            gate_row(
                f"source_bytes:{relative.as_posix()}",
                "validity",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
                upstream=relative.as_posix(),
                artifact_field="bytes",
                principle="Dependent work starts only after the exact input bytes exist.",
            )
        )
        if available:
            hashes[relative.as_posix()] = _source_record(path, "current_input")

    upstream = load_object(root / UPSTREAM_PATH)
    original_flags = {
        "status": upstream.get("status"),
        "verdict_class": upstream.get("verdict_class"),
        "flagged_adversarial": upstream.get("flagged_adversarial"),
        "capture_lifecycle_ready_score": upstream.get("capture_lifecycle_ready_score"),
    }
    if (root / UPSTREAM_PATH).is_file():
        hashes[UPSTREAM_PATH.as_posix()] = _source_record(
            root / UPSTREAM_PATH, "structured_prerequisite", original_flags
        )
    for field, expected, operator in (
        ("capture_lifecycle_ready_score", 1, "=="),
        ("verdict_class", ["null", "positive"], "in"),
        ("flagged_adversarial", False, "=="),
    ):
        gates.append(
            gate_row(
                f"exp7448-capture-lifecycle.{field}",
                "validity",
                expected,
                upstream.get(field),
                upstream=UPSTREAM_PATH.as_posix(),
                artifact_field=field,
                operator=operator,
                principle="Only the qualified unflagged capture lifecycle can authorize live work.",
            )
        )

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    gates.append(
        gate_row(
            "driving_requirement",
            "validity",
            True,
            "REQ-ARC-WMTE-7457" in spec_text,
            upstream=SPEC_PATH.as_posix(),
            artifact_field="REQ-ARC-WMTE-7457",
            principle="The requirement and scenarios must exist before implementation runs.",
        )
    )
    gates.append(
        gate_row(
            "force_live",
            "validity",
            "1",
            os.environ.get("CARNOT_FORCE_LIVE") if force_live is None else force_live,
            upstream="environment",
            artifact_field="CARNOT_FORCE_LIVE",
            principle="The comparison must not fall back to simulated inference.",
        )
    )
    try:
        registry_value = yaml.safe_load((root / REGISTRY_PATH).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        registry_value = {}
    registry = dict(registry_value) if isinstance(registry_value, Mapping) else {}
    gates.append(
        gate_row(
            "registry_parse",
            "validity",
            True,
            bool(registry),
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="yaml_object",
            principle="The duplicate-credit precheck needs structured registry bytes.",
        )
    )
    excluded = "experiment_id: 7457" in (
        (root / EXCLUSION_PATH).read_text(encoding="utf-8")
        if (root / EXCLUSION_PATH).is_file()
        else ""
    )
    gates.append(
        gate_row(
            "task_not_excluded",
            "validity",
            False,
            excluded,
            upstream=EXCLUSION_PATH.as_posix(),
            artifact_field="experiment_id: 7457",
            principle="An excluded task cannot start new model work.",
        )
    )
    return gates, hashes, registry


def registry_precheck(registry: Mapping[str, Any]) -> JsonDict:
    """Confirm both public development games already have full clears."""

    rows = {
        str(row.get("game")): row for row in registry.get("games", []) if isinstance(row, Mapping)
    }
    solved = {
        game: bool(
            rows.get(game, {}).get("full_game_clear") is True
            and int(rows.get(game, {}).get("levels_reproduced") or 0) > 0
        )
        for game in EXPECTED_GAMES
    }
    return {
        "passed": all(solved.values()),
        "games": list(EXPECTED_GAMES),
        "already_solved": solved,
        "full_game_clear": {
            game: rows.get(game, {}).get("full_game_clear") for game in EXPECTED_GAMES
        },
        "levels_reproduced": {
            game: rows.get(game, {}).get("levels_reproduced") for game in EXPECTED_GAMES
        },
        "registry_used_for_policy": False,
        "new_credit_allowed": False,
    }


def build_schedule(games: Sequence[str]) -> list[JsonDict]:
    """Seal two games, two seeds, and interleaved shadow/applied pairs."""

    rows: list[JsonDict] = []
    for seed in EPISODE_SEEDS:
        for game in tuple(games)[:2]:
            pair_id = f"{game}:seed-{seed}"
            for condition in ("shadow", "applied"):
                rows.append(
                    {
                        "episode_id": f"{pair_id}:{condition}",
                        "pair_id": pair_id,
                        "game": str(game),
                        "seed": seed,
                        "condition": condition,
                        "execution_order": len(rows),
                        "action_limit": ACTION_LIMIT,
                        "episode_limit_s": EPISODE_LIMIT_S,
                        "request_limit": REQUEST_LIMIT,
                        "max_new_tokens_per_call": MAX_NEW_TOKENS,
                        "supervisor_threshold": SUPERVISOR_THRESHOLD,
                        "adapter_disabled": True,
                        "stored_engines_disabled": True,
                        "banked_solution_disabled": True,
                        "saved_route_disabled": True,
                        "hidden_game_source_disabled": True,
                        "offline_ground_truth_bfs_disabled": True,
                        "withheld_inputs": list(WITHHELD_INPUTS),
                    }
                )
    return rows


def qualify_scripted_policy_hooks() -> list[JsonDict]:
    """Drive the actual policy hook at 119 and 120 stagnant observations."""

    prior_apply = os.environ.get("CARNOT_ARC_TRAJECTORY_SUPERVISOR")
    prior_window = os.environ.get("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW")
    prior_tool = os.environ.get("CARNOT_ARC_SUPERVISOR_TOOL_ARM")
    prior_level = competition_agent._level_of
    rows: list[JsonDict] = []
    try:
        os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = str(SUPERVISOR_THRESHOLD)
        os.environ.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)
        competition_agent._level_of = lambda frame: int(frame.levels_completed)
        for condition in ("shadow", "applied"):
            if condition == "applied":
                os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR"] = "1"
            else:
                os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
            policy = competition_agent.E3AgentPolicy(
                "bp35", proposer=object(), target_levels=2, value_head=None
            )
            policy.induced = True
            frame = SimpleNamespace(levels_completed=0)
            for _ in range(SUPERVISOR_THRESHOLD - 1):
                policy._maybe_supervise_trajectory(frame)
            before = policy.trajectory_supervisor_diagnostics()
            before_rows = list(before.get("redirects") or before.get("would_have_redirects") or [])
            policy._maybe_supervise_trajectory(frame)
            after = policy.trajectory_supervisor_diagnostics()
            redirects = list(after.get("redirects") or after.get("would_have_redirects") or [])
            rows.append(
                {
                    "evidence_class": "synthetic_policy_hook_qualification",
                    "excluded_from_live_metrics": True,
                    "condition": condition,
                    "threshold": SUPERVISOR_THRESHOLD,
                    "actions_before_boundary": SUPERVISOR_THRESHOLD - 1,
                    "redirects_before_boundary": len(before_rows),
                    "actions_at_boundary": int(after.get("actions_observed") or 0),
                    "proposed_redirect": redirects[0]["arm"] if redirects else None,
                    "redirect_applied": condition == "applied" and bool(redirects),
                    "arm_order": list(CURATED_ARMS),
                    "arms_enabled": list(after.get("arms_enabled") or []),
                    "passed": len(before_rows) == 0 and len(redirects) == 1,
                }
            )
    finally:
        competition_agent._level_of = prior_level
        for key, value in (
            ("CARNOT_ARC_TRAJECTORY_SUPERVISOR", prior_apply),
            ("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", prior_window),
            ("CARNOT_ARC_SUPERVISOR_TOOL_ARM", prior_tool),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return rows


def new_arm_evidence(receipt: Mapping[str, Any]) -> bool:
    """Require all enabled firings and a later exhausted window on one stretch."""

    enabled = tuple(str(arm) for arm in receipt.get("arms_enabled", []) if isinstance(arm, str))
    redirects = [dict(row) for row in receipt.get("redirects", []) if isinstance(row, Mapping)]
    windows = [
        dict(row) for row in receipt.get("unredirected_windows", []) if isinstance(row, Mapping)
    ]
    for window in windows:
        stretch = window.get("stretch_level")
        fired = {
            str(row.get("arm"))
            for row in redirects
            if row.get("stretch_level", row.get("level")) == stretch
            and int(row.get("action_index") or 0) < int(window.get("action_index") or 0)
        }
        used = {str(arm) for arm in window.get("arms_used", [])}
        window_enabled = tuple(window.get("arms_enabled") or enabled)
        if enabled and set(enabled).issubset(fired) and set(window_enabled).issubset(used):
            return True
    return False


def _unstarted_row(schedule: Mapping[str, Any]) -> JsonDict:
    """Represent one fixed panel unit that aggregate stopping prevented."""

    return {
        **deepcopy(dict(schedule)),
        "disposition": "unstarted",
        "action_count": 0,
        "threshold_reached": False,
        "remaining_exposure_gap": SUPERVISOR_THRESHOLD,
        "effect_evidence": "no_run",
        "intervention_null": False,
        "arm_firings": {arm: 0 for arm in CURATED_ARMS},
        "arm_applications": {arm: 0 for arm in CURATED_ARMS},
        "peak_level": None,
        "terminal_level": None,
        "banked_progress": 0,
        "actions_after_redirect": [],
        "generation_calls": 0,
        "elapsed_s": 0.0,
        "action_rows": [],
        "request_budget_receipt": None,
        "solve_provenance": "unstarted",
        "new_level_credit": 0,
        "new_arm_evidence": False,
    }


def reduce_episode(schedule: Mapping[str, Any], observed: Mapping[str, Any]) -> JsonDict:
    """Reduce one live episode without turning missing exposure into a null."""

    row = {**deepcopy(dict(schedule)), **deepcopy(dict(observed))}
    action_rows = [dict(item) for item in row.get("action_rows", []) if isinstance(item, Mapping)]
    action_count = int(row.get("action_count") or len(action_rows))
    receipt = dict(row.get("supervisor_receipt") or {})
    condition = str(row.get("condition"))
    redirect_key = "redirects" if condition == "applied" else "would_have_redirects"
    redirects = [dict(item) for item in receipt.get(redirect_key, []) if isinstance(item, Mapping)]
    firings = Counter(str(item.get("arm")) for item in redirects)
    applications = Counter(
        str(item.get("arm"))
        for item in redirects
        if condition == "applied" or item.get("applied") is True
    )
    plateau = int(action_rows[-1].get("plateau_counter") or 0) if action_rows else 0
    remaining = SUPERVISOR_THRESHOLD - plateau if plateau else SUPERVISOR_THRESHOLD
    threshold_reached = bool(
        row.get("threshold_reached")
        or redirects
        or any(item.get("eligible_window") is True for item in action_rows)
    )
    disposition = str(row.get("disposition") or "failed")
    completed = disposition == "complete"
    effect_evidence = (
        "paired_intervention_outcome"
        if condition == "applied" and applications and completed
        else "shadow_counterfactual"
        if condition == "shadow" and redirects and completed
        else "exposure_without_eligible_arm"
        if completed and threshold_reached
        else "exposure_limited"
        if disposition.startswith("censored") or not threshold_reached
        else "no_applied_redirect"
    )
    request_receipt = row.get("request_budget_receipt")
    calls = (
        int(request_receipt.get("attempted") or 0) if isinstance(request_receipt, Mapping) else 0
    )
    reduced = {
        **row,
        "action_count": action_count,
        "threshold_reached": threshold_reached,
        "remaining_exposure_gap": remaining,
        "effect_evidence": effect_evidence,
        "intervention_null": False,
        "arm_firings": {arm: firings[arm] for arm in CURATED_ARMS},
        "arm_applications": {arm: applications[arm] for arm in CURATED_ARMS},
        "actions_after_redirect": [
            {
                "arm": item.get("arm"),
                "action_index": item.get("action_index"),
                "actions_after_redirect": max(
                    0, action_count - int(item.get("action_index") or action_count)
                ),
            }
            for item in redirects
        ],
        "generation_calls": calls,
        "new_arm_evidence": new_arm_evidence({**receipt, "redirects": redirects}),
        "new_level_credit": 0,
    }
    return reduced


def reduce_panel(
    schedule: Sequence[Mapping[str, Any]], episode_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Account for every fixed unit and form four game-seed pairs."""

    by_id = {
        str(row.get("episode_id")): dict(row)
        for row in episode_rows
        if isinstance(row, Mapping) and row.get("episode_id")
    }
    rows = [
        reduce_episode(sealed, by_id[str(sealed["episode_id"])])
        if str(sealed["episode_id"]) in by_id
        else _unstarted_row(sealed)
        for sealed in schedule
    ]
    dispositions = Counter(str(row.get("disposition")) for row in rows)
    completed = dispositions["complete"]
    failed = dispositions["failed"] + dispositions["complete_error"]
    censored = sum(
        key.startswith("censored_") for key in dispositions for _ in range(dispositions[key])
    )
    unstarted = dispositions["unstarted"]
    attempted = len(rows) - unstarted
    pairs: list[JsonDict] = []
    for pair_id in dict.fromkeys(str(row.get("pair_id")) for row in rows):
        pair_rows = [row for row in rows if row.get("pair_id") == pair_id]
        shadow = next((row for row in pair_rows if row.get("condition") == "shadow"), None)
        applied = next((row for row in pair_rows if row.get("condition") == "applied"), None)
        pair_complete = bool(
            shadow
            and applied
            and shadow.get("disposition") == applied.get("disposition") == "complete"
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "game": pair_rows[0].get("game") if pair_rows else None,
                "seed": pair_rows[0].get("seed") if pair_rows else None,
                "shadow_episode_id": shadow.get("episode_id") if shadow else None,
                "applied_episode_id": applied.get("episode_id") if applied else None,
                "shadow_disposition": shadow.get("disposition") if shadow else None,
                "applied_disposition": applied.get("disposition") if applied else None,
                "pair_complete": pair_complete,
                "shadow_threshold_reached": bool(shadow and shadow.get("threshold_reached")),
                "applied_threshold_reached": bool(applied and applied.get("threshold_reached")),
                "shadow_banked_progress": int(shadow.get("banked_progress") or 0) if shadow else 0,
                "applied_banked_progress": int(applied.get("banked_progress") or 0)
                if applied
                else 0,
                "right_censored": bool(
                    not pair_complete
                    or (shadow and str(shadow.get("disposition", "")).startswith("censored_"))
                    or (applied and str(applied.get("disposition", "")).startswith("censored_"))
                ),
            }
        )
    budget = {
        "planned_units": len(schedule),
        "attempted_units": attempted,
        "completed_units": completed,
        "failed_units": failed,
        "censored_units": censored,
        "unstarted_units": unstarted,
        "fixed_stop_rule": True,
    }
    all_dispositions = len(rows) == len(schedule) == 8 and all(
        row.get("disposition") in TERMINAL_DISPOSITIONS for row in rows
    )
    return {
        "rows": rows,
        "per_game_results": pairs,
        "sample_size_budget": budget,
        "all_dispositions_present": all_dispositions,
        "scientific_null_eligible_pairs": sum(pair["pair_complete"] for pair in pairs),
        "any_new_arm_evidence": any(row.get("new_arm_evidence") is True for row in rows),
    }


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    by_name = {str(row.get("name")): row for row in receipts}
    return all(
        name in by_name
        and by_name[name].get("passed") is True
        and by_name[name].get("timed_out") is not True
        and by_name[name].get("exit_code") == 0
        for name in names
    )


def _request_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    result: list[JsonDict] = []
    for episode in rows:
        receipt = episode.get("request_budget_receipt")
        if not isinstance(receipt, Mapping):
            continue
        for callback in receipt.get("callback_rows", []):
            if isinstance(callback, Mapping):
                result.append({"episode_id": episode.get("episode_id"), **dict(callback)})
    return result


def _field_principles(keys: Sequence[str]) -> JsonDict:
    return {
        key: FIELD_PRINCIPLES.get(
            key,
            "Retain this field as typed experiment evidence for independent reduction.",
        )
        for key in keys
    }


def _duration_breakdown(spans: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {str(row.get("phase")): round(float(row.get("duration_s") or 0.0), 6) for row in spans}


def build_terminal_artifact(
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    episode_rows: Sequence[Mapping[str, Any]],
    scripted_rows: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    require_terminal: bool,
) -> JsonDict:
    """Build one terminal-shaped artifact only from raw durable inputs."""

    panel = reduce_panel(schedule, episode_rows)
    current = live_support.reduce_current_invocations(boundary_events, child_terminal=True)
    requests = _request_rows(panel["rows"])
    request_reduction = live_support.reduce_request_budget_rows(requests)
    scripted_valid = (
        len(scripted_rows) == 2
        and all(row.get("passed") is True for row in scripted_rows)
        and all(row.get("excluded_from_live_metrics") is True for row in scripted_rows)
    )
    affected_names = validation_scope.REQUIRED_CHECK_NAMES
    base_checks_pass = all(row.get("passed") is True for row in preconditions)
    affected_pass = _receipts_pass(validation_receipts, (*affected_names, *REQUIRED_E2E))
    terminal_pass = _receipts_pass(validation_receipts, REQUIRED_TERMINAL)
    complete_score = int(
        base_checks_pass
        and affected_pass
        and (terminal_pass if require_terminal else False)
        and panel["all_dispositions_present"]
        and scripted_valid
        and not current["errors"]
        and request_reduction["zero_excess_dispatch"]
        and request_reduction["unterminated_permits"] == 0
    )
    model_invoked = bool(current["model_invoked"])
    class_name = str(current["inference_substrate_class"])
    blocked = not model_invoked
    status = (
        "blocked_no_run"
        if blocked
        else "complete_bounded_supervisor_exposure"
        if require_terminal and complete_score
        else "measured_candidate_pending_terminal_readers"
    )
    verdict_class = "blocked" if blocked else "null"
    honest_verdict = (
        "blocked_no_run"
        if blocked
        else "complete_null_bounded_exposure_no_arm_promotion"
        if require_terminal and complete_score
        else "complete_null_measured_candidate_pending_terminal_readers"
    )
    gates = [deepcopy(dict(row)) for row in preconditions]
    gates.extend(
        [
            gate_row(
                "eight_episode_dispositions",
                "validity",
                True,
                panel["all_dispositions_present"],
                upstream="rows",
                artifact_field="all_dispositions_present",
                principle="All fixed units need an explicit terminal disposition.",
            ),
            gate_row(
                "synthetic_live_separation",
                "validity",
                True,
                scripted_valid,
                upstream="scripted_qualification_rows",
                artifact_field="excluded_from_live_metrics",
                principle="Synthetic hook evidence cannot enter live outcome metrics.",
            ),
            gate_row(
                "no_automatic_arm_promotion",
                "benefit",
                0,
                0,
                upstream="bounded_protocol",
                artifact_field="arm_value_score",
                principle="Eight exposure episodes do not authorize arm promotion.",
            ),
        ]
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
        "boot_identity": (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
            if Path("/proc/sys/kernel/random/boot_id").is_file()
            else "unavailable"
        ),
        "clock_segment": "single_host_process_tree",
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": deepcopy(MODEL_SPECS if model_invoked else []),
        "model_specs": [deepcopy(dict(row)) for row in model_specs] if model_invoked else [],
        "model_invoked": model_invoked,
        "invocation_counts": current["invocation_counts"],
        "current_invocation_events": [deepcopy(dict(row)) for row in boundary_events],
        "current_invocation_call_rows": current["call_rows"],
        "inference_substrate": (
            "owned_native_cuda_llama_cpp_qwen3.8_27b_gguf" if model_invoked else "no_model_load"
        ),
        "inference_substrate_class": class_name,
        "inference_substrate_details": deepcopy(dict(runtime_receipt)),
        "execution_venue": EXECUTION_VENUE,
        "duration_s": round(max(duration_s, 0.000001), 6),
        "duration_breakdown_s": _duration_breakdown(phase_spans),
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": {
            "fit": None,
            "projection": None,
            "stream": list(EPISODE_SEEDS),
            "episodes": list(EPISODE_SEEDS),
            "resampling": None,
            "explanation": "The fixed paired panel uses two stream seeds and no resampling.",
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "schedule_rows": [deepcopy(dict(row)) for row in schedule],
        "rows": panel["rows"],
        "sample_size_budget": {
            **panel["sample_size_budget"],
            "independent_groups": ["bp35", "cn04"],
            "stopping_rule": "Run eight fixed episodes once; censor at 180 actions, 240 seconds, or 2400 aggregate seconds.",
            "request_limit_per_episode": REQUEST_LIMIT,
            "action_limit_per_episode": ACTION_LIMIT,
            "episode_limit_s": EPISODE_LIMIT_S,
            "aggregate_live_limit_s": AGGREGATE_LIVE_LIMIT_S,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": gate_summary(gates),
        "verifier_is_oracle": True,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "required_check_names": [
            *affected_names,
            *REQUIRED_E2E,
            *(REQUIRED_TERMINAL if require_terminal else ()),
        ],
        "field_principles": {},
        "promotion_score": 0,
        "arc_exposure_complete_score": complete_score,
        "per_game_results": panel["per_game_results"],
        "solve_provenance": [
            {
                "episode_id": row.get("episode_id"),
                "value": row.get("solve_provenance", "no_level_reached"),
            }
            for row in panel["rows"]
        ],
        "new_level_credit": 0,
        "supervisor_exposure_rows": [
            {
                "episode_id": row.get("episode_id"),
                "condition": row.get("condition"),
                "threshold": SUPERVISOR_THRESHOLD,
                "threshold_reached": row.get("threshold_reached"),
                "remaining_exposure_gap": row.get("remaining_exposure_gap"),
                "arm_firings": row.get("arm_firings"),
                "arm_applications": row.get("arm_applications"),
            }
            for row in panel["rows"]
        ],
        "arm_value_score": 0,
        "scripted_qualification_rows": [deepcopy(dict(row)) for row in scripted_rows],
        "scripted_rows_in_live_metrics": False,
        "supervisor_threshold": SUPERVISOR_THRESHOLD,
        "supervisor_arm_order": list(CURATED_ARMS),
        "supervisor_enabled_arms": list(ENABLED_ARMS),
        "new_arm_proposed": False,
        "new_arm_evidence_present": panel["any_new_arm_evidence"],
        "automatic_arm_promotion": False,
        "request_budget_rows": requests,
        "request_budget_reduction": request_reduction,
        "paired_control_present": True,
        "treatment_effect_claimed": False,
        "production_defaults_changed": False,
        "supervisor_threshold_changed": False,
        "supervisor_arm_order_changed": False,
        "model_budget_increased": False,
        "solve_registry_changed": False,
        "research_conductor_changed": False,
        "public_development_generalization_proxy": True,
        "hidden_leaderboard_claimed": False,
        "game_source_read": False,
        "ground_truth_bfs_used": False,
        "repository_health": {
            "status": "scoped_only",
            "unrelated_failures": [],
            "affects_required_checks": False,
        },
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute panel, current calls, request chains, and protected constants."""

    schedule = (
        artifact.get("schedule_rows") if isinstance(artifact.get("schedule_rows"), list) else []
    )
    rows = artifact.get("rows") if isinstance(artifact.get("rows"), list) else []
    panel = reduce_panel(schedule, rows)
    events = (
        artifact.get("current_invocation_events")
        if isinstance(artifact.get("current_invocation_events"), list)
        else []
    )
    current = live_support.reduce_current_invocations(events, child_terminal=True)
    request_rows = _request_rows(panel["rows"])
    requests = live_support.reduce_request_budget_rows(request_rows)
    declared_budget = artifact.get("sample_size_budget") or {}
    budget_matches = all(
        declared_budget.get(key) == value for key, value in panel["sample_size_budget"].items()
    )
    matches = (
        panel["rows"] == rows
        and panel["per_game_results"] == artifact.get("per_game_results")
        and budget_matches
        and current["invocation_counts"] == artifact.get("invocation_counts")
        and current["model_invoked"] is artifact.get("model_invoked")
        and current["inference_substrate_class"] == artifact.get("inference_substrate_class")
        and requests == artifact.get("request_budget_reduction")
        and artifact.get("supervisor_threshold") == SUPERVISOR_THRESHOLD
        and artifact.get("supervisor_arm_order") == list(CURATED_ARMS)
        and artifact.get("promotion_score") == artifact.get("arm_value_score") == 0
        and artifact.get("new_level_credit") == 0
    )
    return {
        "matches_declared": matches,
        "sample_size_budget": panel["sample_size_budget"],
        "per_game_results": panel["per_game_results"],
        "invocation_counts": current["invocation_counts"],
        "model_invoked": current["model_invoked"],
        "inference_substrate_class": current["inference_substrate_class"],
        "invocation_errors": current["errors"],
        "request_budget_reduction": requests,
        "all_dispositions_present": panel["all_dispositions_present"],
    }


def independent_reduce_file(path: Path) -> JsonDict:
    """Cold-load the exact candidate and independently recompute its claims."""

    return independent_reduce(load_object(path))


def validate_artifact(
    value: Mapping[str, Any] | Path, *, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, safety constants, and receipts."""

    artifact = load_object(value) if isinstance(value, Path) else deepcopy(dict(value))
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("supervisor_threshold") != SUPERVISOR_THRESHOLD:
        errors.append("supervisor_threshold_mismatch")
    if artifact.get("supervisor_arm_order") != list(CURATED_ARMS):
        errors.append("supervisor_arm_order_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    reduced = independent_reduce(artifact)
    if reduced["matches_declared"] is not True:
        errors.append("independent_reduction_mismatch")
    if reduced["invocation_errors"]:
        errors.append("invocation_events_invalid")
    expected_specs = MODEL_SPECS if reduced["model_invoked"] else []
    if artifact.get("MODEL_SPECS") != expected_specs:
        errors.append("MODEL_SPECS_mismatch")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
        errors.append("duration_invalid")
    elif (reduced["inference_substrate_class"] == "model_bounded_generation" and duration < 10) or (
        reduced["inference_substrate_class"] == "model_load_no_generation" and duration < 2
    ):
        errors.append("duration_floor_invalid")
    if any(
        artifact.get(field) != 0
        for field in ("promotion_score", "arm_value_score", "new_level_credit")
    ):
        errors.append("credit_or_promotion_nonzero")
    if artifact.get("new_arm_proposed") is not False:
        errors.append("new_arm_proposal_forbidden")
    if any(
        artifact.get(field) is not False
        for field in (
            "production_defaults_changed",
            "supervisor_threshold_changed",
            "supervisor_arm_order_changed",
            "model_budget_increased",
            "solve_registry_changed",
            "research_conductor_changed",
        )
    ):
        errors.append("protected_default_changed")
    if artifact.get("scripted_rows_in_live_metrics") is not False:
        errors.append("synthetic_live_separation_invalid")
    required = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E)
    if require_terminal:
        required = (*required, *REQUIRED_TERMINAL)
    if (
        artifact.get("verdict_class") != "blocked"
        and require_terminal
        and not _receipts_pass(artifact.get("validation_receipts") or [], required)
    ):
        errors.append("validation_receipts_invalid")
    if artifact.get("verdict_class") not in {"null", "blocked", "disqualified"}:
        errors.append("verdict_class_invalid")
    if artifact.get("verdict_class") == "null" and not str(
        artifact.get("honest_verdict") or ""
    ).startswith("complete_"):
        errors.append("honest_verdict_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _fixture_event(call: str, operation: str, state: str, tick: int) -> JsonDict:
    row: JsonDict = {
        "schema": "carnot.arc_inference_boundary_event.v1",
        "event_id": f"sha256:{call}-{state}",
        "call_id": call,
        "operation": operation,
        "state": state,
        "recorded_monotonic_ns": tick,
        "started_monotonic_ns": tick if state == "attempted" else tick - 1,
        "owner_pid": 41,
        "child_pid": 42,
        "model_identity": {
            "model_repository": MODEL_ID,
            "model_filename": MODEL_FILENAME,
            "model_revision": "fixture-revision",
            "model_path": f"/cache/{MODEL_FILENAME}",
        },
    }
    if state in {"completed", "failed"}:
        row["ended_monotonic_ns"] = tick
    return row


def build_artifact_for_test() -> JsonDict:
    """Build a deterministic eight-row artifact through production reducers."""

    schedule = build_schedule(EXPECTED_GAMES)
    episode_rows: list[JsonDict] = []
    for sealed in schedule:
        action_rows = [
            {
                "episode_id": sealed["episode_id"],
                "action_index": index + 1,
                "monotonic_s": float(index + 1),
                "level": 0,
                "plateau_counter": (index + 1) % SUPERVISOR_THRESHOLD,
                "eligible_window": index + 1 == SUPERVISOR_THRESHOLD,
                "proposed_redirect": None,
                "applied_redirect": False,
                "callback_outcome": "not_called",
                "later_progress": False,
            }
            for index in range(ACTION_LIMIT)
        ]
        episode_rows.append(
            {
                **sealed,
                "disposition": "complete",
                "action_count": ACTION_LIMIT,
                "threshold_reached": True,
                "peak_level": 0,
                "terminal_level": 0,
                "banked_progress": 0,
                "supervisor_receipt": {
                    "mode": sealed["condition"],
                    "window": SUPERVISOR_THRESHOLD,
                    "arms_enabled": list(ENABLED_ARMS),
                    (
                        "redirects" if sealed["condition"] == "applied" else "would_have_redirects"
                    ): [],
                    "unredirected_windows": [],
                },
                "action_rows": action_rows,
                "request_budget_receipt": {
                    "attempted": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "in_flight": 0,
                    "callback_rows": [],
                },
                "elapsed_s": 1.0,
                "solve_provenance": "no_level_reached",
                "new_level_credit": 0,
            }
        )
    events = [
        _fixture_event("load", "model_load", "attempted", 1),
        _fixture_event("load", "model_load", "completed", 2),
        _fixture_event("generation", "generation", "attempted", 3),
        _fixture_event("generation", "generation", "completed", 4),
    ]
    scripted = [
        {
            "condition": condition,
            "evidence_class": "synthetic_policy_hook_qualification",
            "excluded_from_live_metrics": True,
            "passed": True,
        }
        for condition in ("shadow", "applied")
    ]
    return build_terminal_artifact(
        started_at_utc="2026-09-20T00:00:00Z",
        ended_at_utc="2026-09-20T00:00:12Z",
        duration_s=12.0,
        phase_spans=[{"phase": "fixture", "duration_s": 12.0, "completed_units": 8}],
        preconditions=[
            gate_row(
                "fixture",
                "validity",
                True,
                True,
                upstream="test",
                artifact_field="fixture",
                principle="The deterministic fixture is explicit.",
            )
        ],
        source_hashes={},
        schedule=schedule,
        episode_rows=episode_rows,
        scripted_rows=scripted,
        boundary_events=events,
        runtime_receipt={"child_terminal": True},
        model_specs=[{"hf_id": MODEL_ID}],
        validation_receipts=[],
        require_terminal=False,
    )


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the fixed Exp7303 commands through the Exp7358 planner."""

    return validation_contract.build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject scope expansion and private temporary-path drift."""

    return validation_contract.validate_command_plan(root, VALIDATION_MANIFEST, commands)


def e2e_command_specs(root: Path, private: Path) -> list[validation_scope.CommandSpec]:
    """Reuse shipped E2E-009, E2E-010, and their private environment smoke."""

    return shipped_live.e2e_command_specs(root, private)


def terminal_command_specs(root: Path, candidate: Path) -> list[validation_scope.CommandSpec]:
    """Build cold replay, independent reduction, and both unchanged readers."""

    python = str(root / ".venv/bin/python")
    wrapper = str(root / WRAPPER_PATH)
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--replay", str(candidate)),
            "exact candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", wrapper, "--replay", str(candidate), "--reduce-only"),
            "exact candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact candidate",
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
            "exact candidate",
            300.0,
        ),
    ]


class SupervisorProbe:
    """Observe the actual supervisor call without changing its decision."""

    def __init__(self, supervisor: Any, applies: bool) -> None:
        self.supervisor = supervisor
        self.applies = applies
        self.last: JsonDict = {}

    def observe(self, snapshot: Any) -> Any:
        """Delegate one real observation and retain its boundary details."""

        before = int(getattr(self.supervisor, "_actions_since_progress", 0))
        redirect = self.supervisor.observe(snapshot)
        after = int(getattr(self.supervisor, "_actions_since_progress", 0))
        self.last = {
            "level": int(snapshot.level),
            "plateau_before": before,
            "plateau_counter": after,
            "eligible_window": before + 1 >= int(self.supervisor.window),
            "proposed_redirect": redirect.arm if redirect is not None else None,
            "applied_redirect": bool(redirect is not None and self.applies),
        }
        return redirect

    def receipt(self) -> JsonDict:
        """Return the unchanged shipped receipt."""

        return dict(self.supervisor.receipt())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.supervisor, name)


class EpisodeTimeout(Exception):
    """Stop one policy episode at its fixed wall-clock ceiling."""


def _reproduce_trace(  # pragma: no cover - real public-environment replay integration.
    game: str, trace: Sequence[Mapping[str, Any]], target_level: int
) -> JsonDict:
    """Replay only the agent's own actions on a fresh public environment."""

    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit

    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    latest: Any = None
    peak = 0
    error: str | None = None
    try:
        for row in trace:
            action_name = str(row.get("action"))
            data = row.get("data")
            if action_name == "RESET":
                latest = env.reset()
            else:
                action = getattr(GameAction, action_name, action_name)
                latest = env.step(action, data=data)
            peak = max(peak, live_support._level(latest))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
    return {
        "attempted": True,
        "trace_length": len(trace),
        "target_level": target_level,
        "reproduced_peak_level": peak,
        "passed": error is None and peak >= target_level,
        "error": error,
    }


def _run_policy_episode(  # pragma: no cover - real public environment and model integration.
    schedule: Mapping[str, Any], proposer: Any, capture: Any, event_path: Path, action_path: Path
) -> JsonDict:
    """Run one actual scored policy episode and persist every action observation."""

    from arcengine import GameAction
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import make_carnot_agent

    episode_id = str(schedule["episode_id"])
    game = str(schedule["game"])
    seed = int(schedule["seed"])
    condition = str(schedule["condition"])
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    if condition == "applied":
        os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_TRAJECTORY_SUPERVISOR", None)
    os.environ["CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW"] = str(SUPERVISOR_THRESHOLD)
    os.environ.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)

    episode_dir = event_path.parent / "episodes" / episode_id.replace(":", "__")
    episode_dir.mkdir(parents=True, exist_ok=True)
    old_e3_dir = e3.E3_DIR
    e3.E3_DIR = episode_dir / "fresh_e3"
    capture.begin_episode(episode_id)
    budget = live_support.DurableEpisodeRequestBudget(
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
    probe = SupervisorProbe(policy._trajectory_supervisor, condition == "applied")
    policy._trajectory_supervisor = probe
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest: Any = None
    action_rows: list[JsonDict] = []
    trace: list[JsonDict] = []
    entered = time.monotonic()
    start_level: int | None = None
    peak_level = 0
    terminal_level = 0
    disposition = "complete"
    error: str | None = None

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout(f"episode exceeded {EPISODE_LIMIT_S}s")

    previous_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, EPISODE_LIMIT_S)
    try:
        for action_index in range(ACTION_LIMIT):
            if agent.is_done(frames, latest):
                break
            calls_before = int(budget.receipt()["attempted"])
            action = agent.choose_action(frames, latest)
            calls_after_receipt = budget.receipt()
            calls_after = int(calls_after_receipt["attempted"])
            callback_outcome = "not_called"
            if calls_after > calls_before:
                callback_rows = list(calls_after_receipt.get("callback_rows") or [])
                callback_outcome = str(callback_rows[-1].get("disposition") or "in_flight")
            name = str(getattr(action, "name", action))
            data_value = getattr(action, "action_data", None)
            data = data_value.model_dump() if hasattr(data_value, "model_dump") else None
            if isinstance(data, Mapping):
                data = {key: value for key, value in data.items() if key != "game_id"}
            if name == "RESET":
                latest = env.reset()
            else:
                latest = env.step(action if isinstance(action, GameAction) else action, data=data)
            level = live_support._level(latest)
            if start_level is None:
                start_level = level
            peak_level = max(peak_level, level)
            terminal_level = level
            trace.append({"action": name, "data": deepcopy(data)})
            detail = dict(probe.last)
            row = {
                "episode_id": episode_id,
                "action_index": action_index + 1,
                "action": name,
                "data": deepcopy(data),
                "monotonic_s": round(time.monotonic() - entered, 6),
                "level": level,
                "plateau_counter": int(detail.get("plateau_counter") or 0),
                "eligible_window": bool(detail.get("eligible_window")),
                "proposed_redirect": detail.get("proposed_redirect"),
                "applied_redirect": bool(detail.get("applied_redirect")),
                "callback_outcome": callback_outcome,
                "later_progress": False,
            }
            action_rows.append(row)
            live_support._append_jsonl(action_path, row)
            frames.append(latest)
    except EpisodeTimeout as exc:
        disposition = "censored_timeout"
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        disposition = "complete_error" if action_rows else "censored_no_first_action"
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        budget.cancel("episode_terminal")
        e3.E3_DIR = old_e3_dir

    for row in action_rows:
        row["later_progress"] = any(
            int(later.get("level") or 0) > int(row.get("level") or 0)
            for later in action_rows[int(row["action_index"]) :]
        )
    supervisor = policy.trajectory_supervisor_diagnostics()
    reached = start_level is not None and peak_level > start_level
    reproduction = (
        _reproduce_trace(game, trace, peak_level)
        if reached
        else {"attempted": False, "passed": False, "reason": "no_level_reached"}
    )
    solve_provenance = (
        "live_agent_self_discovery"
        if reached and reproduction.get("passed") is True
        else "unreproduced_live_progress"
        if reached
        else "no_level_reached"
    )
    return {
        **deepcopy(dict(schedule)),
        "disposition": disposition,
        "policy_entry": {
            "factory": "make_carnot_agent",
            "policy_class": type(policy).__name__,
            "choose_action_path": True,
            "is_done_path": True,
            "withheld_inputs": list(WITHHELD_INPUTS),
        },
        "action_count": len(action_rows),
        "threshold_reached": any(row["eligible_window"] for row in action_rows),
        "start_level": start_level,
        "peak_level": peak_level if start_level is not None else None,
        "terminal_level": terminal_level if start_level is not None else None,
        "banked_progress": max(0, terminal_level - start_level) if start_level is not None else 0,
        "action_rows": action_rows,
        "supervisor_receipt": supervisor,
        "request_budget_receipt": budget.receipt(),
        "server_request_rows": live_support._transport_rows(
            live_support._read_jsonl(event_path), episode_id
        ),
        "elapsed_s": round(time.monotonic() - entered, 6),
        "solve_provenance": solve_provenance,
        "trace_reproduction": reproduction,
        "new_level_credit": 0,
        "error": error,
    }


def session_environment(
    base: Mapping[str, str], *, gpu_index: int, port: int, raw_dir: Path
) -> dict[str, str]:  # pragma: no cover - native child environment.
    """Build the shipped adapter-withheld environment for this owned child."""

    env = live_support.session_environment(base, gpu_index=gpu_index, port=port, raw_dir=raw_dir)
    env.update(
        {
            "CARNOT_ARC_INDUCE_MAX_TOKENS": str(MAX_NEW_TOKENS),
            "CARNOT_ARC_INDUCE_TIMEOUT": str(int(EPISODE_LIMIT_S)),
            "CARNOT_ARC_MAX_REFINEMENT_ROUNDS": str(REQUEST_LIMIT),
            "CARNOT_ARC_INDUCE_TOOL_TURNS": str(REQUEST_LIMIT),
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": str(SUPERVISOR_THRESHOLD),
            BOUNDARY_LEDGER_ENV: str(REPO_ROOT / BOUNDARY_PATH),
        }
    )
    env.pop("CARNOT_ARC_SUPERVISOR_TOOL_ARM", None)
    return env


def run_live_session(args: argparse.Namespace) -> int:  # pragma: no cover - live child.
    """Load one owned Qwen server and run the eight fixed episodes."""

    started = time.monotonic()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    event_path = raw_dir / RUNTIME_EVENT_PATH.name
    action_path = raw_dir / ACTION_PATH.name
    schedule = load_object(Path(args.schedule_path)).get("rows") or []
    capture = live_support.DurableRequestCapture(raw_dir, event_path)
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
            model_path=live_support._absolute_model_path(args.model_path),
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
        proposer.requested_model_path = live_support._absolute_model_path(args.model_path)
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
            "mtp": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "request_limit_per_episode": REQUEST_LIMIT,
        }
        current_work_receipt.atomic_json(
            Path(args.checkpoint_path),
            {"stage": "model_loaded", "model_loaded": True, "completed_units": 0},
        )
        for index, sealed in enumerate(schedule):
            progress(
                started,
                "episode",
                "before_benchmark",
                episode_id=sealed["episode_id"],
                completed_units=index,
            )
            row = _run_policy_episode(sealed, proposer, capture, event_path, action_path)
            rows.append(row)
            current_work_receipt.atomic_json(raw_dir / "episode_rows.json", {"rows": rows})
            current_work_receipt.atomic_json(
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
                episode_id=sealed["episode_id"],
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
        current_work_receipt.atomic_json(Path(args.session_path), session)
        current_work_receipt.atomic_json(
            Path(args.checkpoint_path),
            {
                "stage": "child_terminal",
                "model_loaded": session["model_loaded"],
                "completed_units": len(rows),
                "terminal_child": True,
            },
        )
    return 0


def run_child_with_lease(
    *, resources: Mapping[str, Any], schedule_path: Path, started: float
) -> JsonDict:  # pragma: no cover - owned live process boundary.
    """Acquire one GPU lease and supervise only this experiment's child."""

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
    port = live_support._free_port()
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
        resident = live_support._observe_loaded_lease(lease, checkpoint, resident)
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
            "model_invoked": bool(
                InvocationBoundaryLedger(REPO_ROOT / BOUNDARY_PATH).read_events()
            ),
            "episodes": list(
                load_object(REPO_ROOT / RAW_DIR / "episode_rows.json").get("rows") or []
            ),
            "error": "live_child_did_not_write_session",
        }
    resident = live_support._observe_loaded_lease(lease, session, resident)
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
    current_work_receipt.atomic_json(REPO_ROOT / SESSION_PATH, session)
    return session


def _phase(
    spans: list[JsonDict],
    name: str,
    phase_start: float,
    run_start: float,
    completed_units: int,
    checkpoint: str | None = None,
) -> None:
    now = time.monotonic()
    spans.append(
        {
            "phase": name,
            "start_s": round(phase_start - run_start, 6),
            "end_s": round(now - run_start, 6),
            "duration_s": round(now - phase_start, 6),
            "completed_units": completed_units,
            "checkpoint": checkpoint,
            "ended_at_utc": utc_now(),
        }
    )


def _runtime_preconditions(  # pragma: no cover - native model and CUDA checks.
    root: Path, started: float
) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    """Resolve the current cached model through the shipped live reader."""

    checks, hashes, resources = live_support._runtime_preconditions(root, started)
    return [dict(row) for row in checks], dict(hashes), dict(resources)


def _blocked_artifact(
    *,
    started_at: str,
    duration_s: float,
    spans: Sequence[Mapping[str, Any]],
    checks: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    scripted_rows: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]] = (),
    session: Mapping[str, Any] | None = None,
) -> JsonDict:  # pragma: no cover - external absence path.
    """Publish an exact blocked state without invented generation work."""

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=duration_s,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=hashes,
        schedule=schedule,
        episode_rows=list((session or {}).get("episodes") or []),
        scripted_rows=scripted_rows,
        boundary_events=boundary_events,
        runtime_receipt=(session or {}).get("runtime_receipt") or {},
        model_specs=[],
        validation_receipts=receipts,
        require_terminal=False,
    )
    current = live_support.reduce_current_invocations(boundary_events, child_terminal=True)
    if current["model_invoked"]:
        artifact["status"] = (
            "model_load_no_generation"
            if current["inference_substrate_class"] == "model_load_no_generation"
            else "blocked_owned_live_failure"
        )
        artifact["honest_verdict"] = artifact["status"]
        artifact["verdict_class"] = "blocked"
    else:
        artifact["status"] = "blocked_no_run"
        artifact["honest_verdict"] = "blocked_no_run"
        artifact["verdict_class"] = "blocked"
    artifact["arc_exposure_complete_score"] = 0
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - orchestration.
    """Run prechecks, scoped validation, live episodes, readers, and publish."""

    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    receipts: list[JsonDict] = []
    progress(started, "startup", "begin", run_date=run_date)

    phase_start = time.monotonic()
    progress(started, "preconditions", "before")
    checks, hashes, registry = collect_preconditions(root)
    checks.insert(
        0,
        gate_row(
            "run_date",
            "validity",
            RUN_DATE,
            run_date,
            upstream="command_line",
            artifact_field="--date",
            principle="The protocol runs under the declared date.",
        ),
    )
    registry_receipt = registry_precheck(registry)
    checks.append(
        gate_row(
            "registry_public_games_already_solved",
            "validity",
            True,
            registry_receipt["passed"],
            upstream=REGISTRY_PATH.as_posix(),
            artifact_field="bp35|cn04.full_game_clear",
            principle="These public development games cannot receive new solve credit.",
        )
    )
    schedule = build_schedule(EXPECTED_GAMES)
    current_work_receipt.atomic_json(
        root / SCHEDULE_PATH,
        {"registry_precheck": registry_receipt, "rows": schedule},
    )
    _phase(spans, "preconditions", phase_start, started, len(checks), SCHEDULE_PATH.as_posix())
    progress(started, "preconditions", "after", passed=all(row["passed"] for row in checks))
    if not all(row["passed"] for row in checks):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=receipts,
            schedule=schedule,
            scripted_rows=[],
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        progress(started, "publish", "terminal_blocked", path=RESULT_PATH)
        return artifact

    phase_start = time.monotonic()
    progress(started, "scripted_qualification", "before")
    scripted_rows = qualify_scripted_policy_hooks()
    current_work_receipt.atomic_json(
        root / RAW_DIR / "scripted_qualification.json", {"rows": scripted_rows}
    )
    _phase(spans, "scripted_qualification", phase_start, started, len(scripted_rows))
    progress(
        started,
        "scripted_qualification",
        "after",
        passed=all(row["passed"] for row in scripted_rows),
    )
    if not all(row["passed"] for row in scripted_rows):
        checks.append(
            gate_row(
                "scripted_policy_hooks",
                "validity",
                True,
                False,
                upstream="actual_E3AgentPolicy_hooks",
                artifact_field="passed",
                principle="Live work needs a qualified threshold and consumption boundary.",
            )
        )
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=receipts,
            schedule=schedule,
            scripted_rows=scripted_rows,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        return artifact

    private = Path(tempfile.mkdtemp(prefix="exp7457-validation-", dir="/tmp"))
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
    required_before_live = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_E2E)
    if not _receipts_pass(receipts, required_before_live):
        artifact = _blocked_artifact(
            started_at=started_at,
            duration_s=time.monotonic() - started,
            spans=spans,
            checks=checks,
            hashes=hashes,
            receipts=receipts,
            schedule=schedule,
            scripted_rows=scripted_rows,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
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
            scripted_rows=scripted_rows,
        )
        current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
        progress(started, "publish", "terminal_blocked_no_run", path=RESULT_PATH)
        return artifact

    for path in (
        root / BOUNDARY_PATH,
        root / RUNTIME_EVENT_PATH,
        root / ACTION_PATH,
        root / SESSION_PATH,
        root / CHECKPOINT_PATH,
        root / TERMINAL_CANDIDATE_PATH,
        root / RAW_DIR / "episode_rows.json",
    ):
        path.unlink(missing_ok=True)
    phase_start = time.monotonic()
    progress(started, "live", "before_model_load_generation_benchmark", planned_units=8)
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

    boundary_events = InvocationBoundaryLedger(root / BOUNDARY_PATH).read_events()
    episodes = [dict(row) for row in session.get("episodes", []) if isinstance(row, Mapping)]
    source_hashes = deepcopy(hashes)
    for relative, role in (
        (SCHEDULE_PATH, "frozen_protocol"),
        (RAW_DIR / "scripted_qualification.json", "synthetic_policy_hook_evidence"),
        (MODULE_PATH, "producer_code"),
        (WRAPPER_PATH, "declared_entrypoint"),
        (TEST_PATH, "new_behavior_tests"),
    ):
        path = root / relative
        if path.is_file():
            source_hashes[relative.as_posix()] = _source_record(path, role)

    phase_start = time.monotonic()
    candidate = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        schedule=schedule,
        episode_rows=episodes,
        scripted_rows=scripted_rows,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=False,
    )
    candidate_errors = validate_artifact(candidate, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"measured candidate invalid: {candidate_errors}")
    current_work_receipt.atomic_json(root / TERMINAL_CANDIDATE_PATH, candidate)
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
    _phase(spans, "terminal_validation", phase_start, started, len(terminal))

    artifact = build_terminal_artifact(
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        preconditions=checks,
        source_hashes=source_hashes,
        schedule=schedule,
        episode_rows=episodes,
        scripted_rows=scripted_rows,
        boundary_events=boundary_events,
        runtime_receipt=session.get("runtime_receipt") or {},
        model_specs=[resources["model_spec"]],
        validation_receipts=receipts,
        require_terminal=True,
    )
    errors = validate_artifact(artifact, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal artifact invalid: {errors}")
    progress(started, "publish", "before_atomic_terminal_write", path=RESULT_PATH)
    current_work_receipt.atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "after_atomic_terminal_write",
        path=RESULT_PATH,
        exposure=artifact["arc_exposure_complete_score"],
    )
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public experiment, private child, and cold replay roles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=[RUN_DATE])
    parser.add_argument("--role", choices=("experiment", "live-session"), default="experiment")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--reduce-only", action="store_true")
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
    """Run the host experiment, owned live child, or cold replay."""

    args = parse_args(argv)
    if args.replay is not None:
        reduced = independent_reduce_file(args.replay)
        errors = [] if args.reduce_only else validate_artifact(args.replay, require_terminal=False)
        print(
            json.dumps({"reduced": reduced, "validation_errors": errors}, sort_keys=True),
            flush=True,
        )
        return int(bool(errors) or reduced.get("matches_declared") is not True)
    if args.role == "live-session":
        return run_live_session(args)
    artifact = run_experiment(REPO_ROOT, str(args.date))
    return 0 if artifact.get("verdict_class") in {"null", "blocked"} else 1


__all__ = [
    "ACTION_LIMIT",
    "AGGREGATE_LIVE_LIMIT_S",
    "CURATED_ARMS",
    "ENABLED_ARMS",
    "EPISODE_LIMIT_S",
    "EPISODE_SEEDS",
    "EXECUTION_VENUE",
    "INFERENCE_SUBSTRATE_CLASS",
    "MAX_NEW_TOKENS",
    "MODEL_SPECS",
    "REQUEST_LIMIT",
    "SPEC_PATH",
    "SUPERVISOR_THRESHOLD",
    "UPSTREAM_PATH",
    "build_artifact_for_test",
    "build_schedule",
    "collect_preconditions",
    "independent_reduce",
    "main",
    "new_arm_evidence",
    "qualify_scripted_policy_hooks",
    "reduce_episode",
    "reduce_panel",
    "registry_precheck",
    "validate_artifact",
]
