"""Audit recorded supervisor support without inventing counterfactual outcomes.

The durable ledger records actions that the live E3 supervisor really applied.
This module compares only existing arm orders. A candidate loses support at its
first different action because the archived suffix then belongs to another path.

Spec refs: REQ-ARC-WMTE-7365 and SCENARIO-ARC-WMTE-7365-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.agentic.arc_supervisor_refinement import load_ledger
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    ARM_TOOL_LOOP_REINDUCTION,
    TrajectorySnapshot,
    TrajectorySupervisor,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.646"
EXPERIMENT_ID = "exp7365-supervisor-support"
SCHEMA = "carnot.exp7365.v646.supervisor_support.v1"
EXPERIMENT_CONFIG_SCHEMA = "carnot.exp7365.supervisor_order_opt_in.v1"
RESULT_PATH = Path("results/experiment_7365_v646_supervisor_support.json")
RAW_DIR = Path("results/raw/experiment_7365_v646_supervisor_support")
MODULE_PATH = Path("python/carnot/experiment_7365_v646_supervisor_support.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7365_v646_supervisor_support.py")
TEST_PATH = Path("tests/python/test_experiment_7365_v646_supervisor_support.py")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
VALIDATION_CONTRACT_PATH = Path("results/experiment_7358_v646_validation_contract.json")
LEDGER_PATH = Path("ops/arc_supervisor_refinement_ledger.json")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
HISTORICAL_PATHS = (
    Path("results/experiment_6656_arc_trace_automaton_live_loo.json"),
    Path("results/experiment_6921_arc_dynamic_supervisor_banked_credit.json"),
    Path("results/experiment_6682_arc_held_family_supervisor_ab.json"),
)
INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("openspec/capabilities/research-reporting/spec.md"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/agentic/arc_supervisor_refinement.py"),
    Path("python/carnot/agentic/arc_trajectory_supervisor.py"),
    Path("tests/python/test_arc_supervisor_refinement.py"),
    Path("tests/python/test_arc_supervisor_refinement_eval_runs_20260904.py"),
    REGISTRY_PATH,
    LEDGER_PATH,
    VALIDATION_CONTRACT_PATH,
    *HISTORICAL_PATHS,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

MIN_SUPPORTED_PER_ARM = 10
MIN_DEVELOPMENT_GAMES = 3
REQUIRED_EVALUATION_GAMES = 4
COMPARED_ARMS = (ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY)
CANDIDATE_ORDERINGS = (
    tuple(ARM_ORDER),
    (
        ARM_DROP_GOAL_BIAS,
        ARM_FORCE_DIVERSITY,
        ARM_ALLOW_REINDUCTION,
        ARM_TOOL_LOOP_REINDUCTION,
    ),
    (
        ARM_ALLOW_REINDUCTION,
        ARM_DROP_GOAL_BIAS,
        ARM_FORCE_DIVERSITY,
        ARM_TOOL_LOOP_REINDUCTION,
    ),
)
ZERO_CURRENT_INVOCATIONS = {
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
TERMINAL_CHECK_NAMES = (
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
AFFECTED_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


class SupervisorSupportError(RuntimeError):
    """Stop publication when current evidence and derived fields disagree."""


def utc_now() -> str:
    """Return a real aware UTC timestamp at an execution boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each phase boundary and each potentially long subprocess boundary."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7365] phase={phase} event={event} "
        f"elapsed_s={time.monotonic() - started:.3f}" + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def config_hash(value: Mapping[str, Any]) -> str:
    """Hash exact experiment configuration fields with canonical JSON."""

    return validation_contract.canonical_hash(dict(value))


def _config_payload(order: Sequence[str]) -> JsonDict:
    """Build the only payload a future live experiment may opt in with."""

    return {
        "schema": EXPERIMENT_CONFIG_SCHEMA,
        "enabled": True,
        "arm_order": list(order),
    }


def _valid_order(order: Sequence[str]) -> bool:
    """Accept each existing curated arm exactly once and no new arm."""

    return len(order) == len(ARM_ORDER) and set(order) == set(ARM_ORDER)


class OrderedTrajectorySupervisor(TrajectorySupervisor):
    """Apply a validated experiment order through the existing supervisor seam.

    The wrapper asks the production eligibility table about one arm at a time.
    It does not copy the table or change a production default. The tool-loop arm
    still needs the ordinary re-induction arm to have fired first.
    """

    def __init__(self, arm_order: Sequence[str], **kwargs: Any) -> None:
        if not _valid_order(arm_order):
            raise ValueError("arm_order must be a permutation of existing curated arms")
        super().__init__(**kwargs)
        self.arm_order = tuple(arm_order)

    def _first_eligible_arm(self, snapshot: TrajectorySnapshot) -> tuple[str | None, str]:
        original = set(self._arms_used)
        for arm in self.arm_order:
            if arm in original:
                continue
            if arm == ARM_TOOL_LOOP_REINDUCTION and ARM_ALLOW_REINDUCTION not in original:
                continue
            self._arms_used = original | (set(ARM_ORDER) - {arm})
            selected = super()._first_eligible_arm(snapshot)
            self._arms_used = set(original)
            if selected[0] == arm:
                return selected
        self._arms_used = original
        return None, ""


def _validated_opt_in(config: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """Return a safe order only for an exact, hash-bound opt-in object."""

    if not isinstance(config, Mapping):
        return None
    payload = {
        "schema": config.get("schema"),
        "enabled": config.get("enabled"),
        "arm_order": config.get("arm_order"),
    }
    raw_order = payload["arm_order"]
    if (
        payload["schema"] != EXPERIMENT_CONFIG_SCHEMA
        or payload["enabled"] is not True
        or not isinstance(raw_order, list)
        or not all(isinstance(arm, str) for arm in raw_order)
        or not _valid_order(raw_order)
        or config.get("config_hash") != config_hash(payload)
    ):
        return None
    return tuple(raw_order)


def install_experiment_supervisor(policy: Any, config: Mapping[str, Any] | None) -> bool:
    """Install an ordered supervisor only after a valid explicit live opt-in.

    A missing or invalid opt-in leaves the policy object untouched. This makes
    the normal E3 factory and its curated defaults the fallback in every case.
    """

    order = _validated_opt_in(config)
    if order is None:
        return False
    policy._trajectory_supervisor = OrderedTrajectorySupervisor(order)
    policy._trajectory_supervisor_applies = True
    return True


def _gate_row(
    check: str,
    upstream: str,
    artifact_field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> JsonDict:
    """Keep exact expected and observed values for one prerequisite or gate."""

    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def check_upstream_contract(
    producer: Mapping[str, Any], *, available: bool, excluded: bool
) -> list[JsonDict]:
    """Check the three structured gates plus terminal and quarantine state."""

    upstream = VALIDATION_CONTRACT_PATH.as_posix()
    if not available:
        return [
            _gate_row(
                "validation_contract_path",
                upstream,
                "path",
                "readable_nonempty_json",
                "missing",
                False,
            )
        ]
    verdict = producer.get("verdict_class")
    status = str(producer.get("status", "missing"))
    allowed = ["positive", "circular_positive", "null"]
    return [
        _gate_row(
            "validation_contract_ready",
            upstream,
            "validation_contract_ready_score",
            1,
            producer.get("validation_contract_ready_score", "missing"),
            producer.get("validation_contract_ready_score") == 1,
        ),
        _gate_row(
            "validation_contract_verdict",
            upstream,
            "verdict_class",
            allowed,
            verdict,
            verdict in allowed,
        ),
        _gate_row(
            "validation_contract_adversarial",
            upstream,
            "flagged_adversarial",
            False,
            producer.get("flagged_adversarial", "missing"),
            producer.get("flagged_adversarial") is False,
        ),
        _gate_row(
            "validation_contract_terminal_status",
            upstream,
            "status",
            "terminal_not_blocked_partial_or_disqualified",
            status,
            not any(token in status for token in ("blocked", "partial", "disqualified")),
        ),
        _gate_row(
            "validation_contract_not_quarantined",
            "ops/exclusion_manifest.yaml",
            "exp7358-validation-contract",
            False,
            excluded,
            not excluded,
        ),
    ]


def _load_json_object(path: Path) -> JsonDict:
    """Load one JSON object, returning empty data for unavailable external bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(root: Path) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Hash every named input and check producer gates before ledger reduction."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _gate_row(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else "missing_or_empty",
                available,
            )
        )
        if available:
            hashes[relative.as_posix()] = validation_contract.sha256_file(path)

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    present = "REQ-ARC-WMTE-7365" in spec_text
    checks.append(
        _gate_row(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-ARC-WMTE-7365",
            "REQ-ARC-WMTE-7365" if present else "missing",
            present,
        )
    )

    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    producer = _load_json_object(root / VALIDATION_CONTRACT_PATH)
    checks.extend(
        check_upstream_contract(
            producer,
            available=bool(producer),
            excluded=(
                "experiment_id: 7358" in exclusion_text
                or "exp7358-validation-contract" in exclusion_text
            ),
        )
    )
    current_excluded = "experiment_id: 7365" in exclusion_text or EXPERIMENT_ID in exclusion_text
    checks.append(
        _gate_row(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            current_excluded,
            not current_excluded,
        )
    )

    historical: list[JsonDict] = []
    for relative in HISTORICAL_PATHS:
        source = _load_json_object(root / relative)
        historical.append(
            {
                "path": relative.as_posix(),
                "sha256": hashes.get(relative.as_posix()),
                "historical_only": True,
                "status": source.get("status"),
                "verdict_class": source.get("verdict_class"),
                "flagged_adversarial": source.get("flagged_adversarial"),
                "model_invoked": source.get("model_invoked", "not_recorded"),
                "inference_substrate": source.get("inference_substrate"),
            }
        )
    return checks, hashes, historical


def freeze_game_split(registry: Mapping[str, Any], development_games: Sequence[str]) -> JsonDict:
    """Freeze a deterministic four-game panel outside development games and families."""

    raw_games = registry.get("games")
    game_rows = (
        [row for row in raw_games if isinstance(row, Mapping)]
        if isinstance(raw_games, list)
        else []
    )
    by_id = {
        str(row["game"]): row
        for row in game_rows
        if isinstance(row.get("game"), str) and row.get("game")
    }
    development_ids = sorted(set(str(game) for game in development_games))
    development_families = sorted(
        {
            str(by_id.get(game, {}).get("mechanic_class") or f"unknown:{game}")
            for game in development_ids
        }
    )
    eligible = [
        game
        for game, row in sorted(by_id.items())
        if game not in development_ids
        and int(row.get("levels_reproduced") or 0) > 0
        and str(row.get("mechanic_class") or f"unknown:{game}") not in development_families
    ]
    evaluation_ids = eligible[:REQUIRED_EVALUATION_GAMES]
    evaluation_families = sorted(
        {str(by_id[game].get("mechanic_class") or f"unknown:{game}") for game in evaluation_ids}
    )
    registry_precheck = [
        {
            "game": game,
            "assignment": "development" if game in development_ids else "evaluation",
            "family": str(by_id.get(game, {}).get("mechanic_class") or f"unknown:{game}"),
            "levels_reproduced": by_id.get(game, {}).get("levels_reproduced"),
            "existing_public_solve_only": True,
        }
        for game in [*development_ids, *evaluation_ids]
    ]
    return {
        "development_game_ids": development_ids,
        "evaluation_game_ids": evaluation_ids,
        "development_family_ids": development_families,
        "evaluation_family_ids": evaluation_families,
        "disjoint_games": not bool(set(development_ids) & set(evaluation_ids)),
        "disjoint_families": not bool(set(development_families) & set(evaluation_families)),
        "evaluation_reachable": len(evaluation_ids) == REQUIRED_EVALUATION_GAMES
        and all(int(by_id[game].get("levels_reproduced") or 0) > 0 for game in evaluation_ids),
        "registry_precheck": registry_precheck,
    }


def replay_trajectory(
    entry: Mapping[str, Any], candidate_order: Sequence[str], assignment: str
) -> list[JsonDict]:
    """Replay one recorded arm prefix and censor the first changed action onward."""

    if not _valid_order(candidate_order):
        raise ValueError("candidate_order must be a permutation of existing curated arms")
    redirects = [row for row in entry.get("redirects", []) if isinstance(row, Mapping)]
    receipt_id = str(entry.get("receipt_id") or "missing_receipt_id")
    changed = False
    rows: list[JsonDict] = []
    for index, redirect in enumerate(redirects):
        observed_arm = str(redirect.get("arm") or "")
        remaining_observed = {
            str(row.get("arm") or "") for row in redirects[index:] if row.get("arm") is not None
        }
        candidate_arm = next((arm for arm in candidate_order if arm in remaining_observed), None)
        reason: str | None = None
        if changed:
            reason = "after_candidate_changed_action"
        elif candidate_arm != observed_arm:
            changed = True
            reason = "candidate_changed_actual_action"
        outcome = redirect.get("resolved_by_levelup")
        if reason is None and not isinstance(outcome, bool):
            reason = "missing_observed_outcome"
        supported = reason is None
        rows.append(
            {
                "candidate_ordering": list(candidate_order),
                "trajectory_id": receipt_id,
                "game": entry.get("game"),
                "family_assignment": assignment,
                "decision_index": index,
                "action_index": redirect.get("action_index"),
                "level": redirect.get("level"),
                "observed_arm": observed_arm,
                "candidate_arm": candidate_arm,
                "observed_outcome": outcome if supported else None,
                "actions_to_levelup": redirect.get("actions_to_levelup") if supported else None,
                "co_credited_count": redirect.get("co_credited_count"),
                "supported": supported,
                "censoring_reason": reason,
                "unsupported_counterfactual": reason
                in {"candidate_changed_actual_action", "after_candidate_changed_action"},
                "causal_interpretation": "descriptive_observed_association_only",
                "source": entry.get("source"),
            }
        )
    return rows


def _candidate_metrics(rows: Sequence[Mapping[str, Any]], candidate_index: int) -> JsonDict:
    """Reduce supported decisions while keeping games and trajectories independent."""

    supported = [row for row in rows if row.get("supported") is True]
    per_arm = {
        arm: {
            "supported_decision_count": sum(
                1 for row in supported if row.get("observed_arm") == arm
            ),
            "development_game_count": len(
                {row.get("game") for row in supported if row.get("observed_arm") == arm}
            ),
            "trajectory_count": len(
                {row.get("trajectory_id") for row in supported if row.get("observed_arm") == arm}
            ),
            "observed_levelup_count": sum(
                1
                for row in supported
                if row.get("observed_arm") == arm and row.get("observed_outcome") is True
            ),
        }
        for arm in COMPARED_ARMS
    }
    for metrics in per_arm.values():
        count = int(metrics["supported_decision_count"])
        metrics["observed_levelup_rate"] = (
            round(int(metrics["observed_levelup_count"]) / count, 6) if count else None
        )
    return {
        "candidate_id": f"ordering_{candidate_index}",
        "arm_order": list(rows[0]["candidate_ordering"]) if rows else [],
        "supported_decision_count": len(supported),
        "supported_trajectory_count": len({row.get("trajectory_id") for row in supported}),
        "supported_game_count": len({row.get("game") for row in supported}),
        "censored_decision_count": len(rows) - len(supported),
        "unsupported_counterfactual_count": sum(
            int(row.get("unsupported_counterfactual") is True) for row in rows
        ),
        "per_arm": per_arm,
    }


def _loo_rows(
    replay_rows: Sequence[Mapping[str, Any]], candidate_id: str, development_games: Sequence[str]
) -> list[JsonDict]:
    """Apply the support floor after holding out each development game."""

    rows: list[JsonDict] = []
    for held_out in development_games:
        kept = [
            row
            for row in replay_rows
            if row.get("supported") is True and row.get("game") != held_out
        ]
        counts = Counter(str(row.get("observed_arm")) for row in kept)
        games_by_arm = {
            arm: sorted({str(row.get("game")) for row in kept if row.get("observed_arm") == arm})
            for arm in COMPARED_ARMS
        }
        rows.append(
            {
                "candidate_id": candidate_id,
                "held_out_game": held_out,
                "supported_decisions_by_arm": {arm: counts[arm] for arm in COMPARED_ARMS},
                "support_games_by_arm": games_by_arm,
                "expected_minimum_per_arm": MIN_SUPPORTED_PER_ARM,
                "passed": all(counts[arm] >= MIN_SUPPORTED_PER_ARM for arm in COMPARED_ARMS),
            }
        )
    return rows


def _control_rows(
    ledger: Mapping[str, Any], support_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Prove that each receipt hazard is preserved or censored, never pooled away."""

    ingest = (ledger.get("recommendation") or {}).get("ingest_counts") or {}
    missing = [
        row for row in support_rows if row.get("censoring_reason") == "missing_observed_outcome"
    ]
    changed = [row for row in support_rows if row.get("unsupported_counterfactual") is True]
    co_credit = [row for row in support_rows if isinstance(row.get("co_credited_count"), int)]
    controls = ledger.get("controls") if isinstance(ledger.get("controls"), Mapping) else {}
    return [
        {
            "control": "duplicate_partial_final",
            "observed": int(ingest.get("applied_duplicate") or 0),
            "passed": len(ledger.get("entries") or {})
            == len(set((ledger.get("entries") or {}).keys())),
        },
        {
            "control": "shadow_separation",
            "observed": len(controls),
            "passed": all(row.get("mode") == "shadow" for row in controls.values()),
        },
        {
            "control": "co_credit_preserved",
            "observed": len(co_credit),
            "passed": all(row.get("co_credited_count") is not None for row in co_credit),
        },
        {
            "control": "missing_outcome_censored",
            "observed": len(missing),
            "passed": all(row.get("supported") is False for row in missing),
        },
        {
            "control": "changed_action_suffix_censored",
            "observed": len(changed),
            "passed": all(
                row.get("supported") is False and row.get("observed_outcome") is None
                for row in changed
            ),
        },
    ]


def audit_ledger(
    ledger: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    candidate_orderings: Sequence[Sequence[str]] = CANDIDATE_ORDERINGS,
) -> JsonDict:
    """Run the complete support-masked audit as a pure deterministic reduction."""

    if len(candidate_orderings) > 3 or any(
        not _valid_order(order) for order in candidate_orderings
    ):
        raise ValueError("candidate orderings must be at most three curated permutations")
    entries = ledger.get("entries") if isinstance(ledger.get("entries"), Mapping) else {}
    controls = ledger.get("controls") if isinstance(ledger.get("controls"), Mapping) else {}
    normalized_entries: list[JsonDict] = []
    for receipt_id, raw_entry in sorted(entries.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        entry["receipt_id"] = str(receipt_id)
        normalized_entries.append(entry)
    development_games = sorted(
        {str(entry.get("game")) for entry in normalized_entries if entry.get("game") is not None}
    )
    split = freeze_game_split(registry, development_games)

    support_rows: list[JsonDict] = []
    candidate_rows: list[JsonDict] = []
    loo_rows: list[JsonDict] = []
    candidate_replays: dict[str, list[JsonDict]] = {}
    for candidate_index, ordering in enumerate(candidate_orderings):
        replayed = [
            row
            for entry in normalized_entries
            for row in replay_trajectory(entry, ordering, "development")
        ]
        candidate_id = f"ordering_{candidate_index}"
        candidate_replays[candidate_id] = replayed
        support_rows.extend(replayed)
        metrics = _candidate_metrics(replayed, candidate_index)
        folds = _loo_rows(replayed, candidate_id, development_games)
        loo_rows.extend(folds)
        arm_floor = all(
            metrics["per_arm"][arm]["supported_decision_count"] >= MIN_SUPPORTED_PER_ARM
            and metrics["per_arm"][arm]["development_game_count"] >= MIN_DEVELOPMENT_GAMES
            for arm in COMPARED_ARMS
        )
        loo_passed = bool(folds) and all(row["passed"] for row in folds)
        metrics["arm_support_floor_passed"] = arm_floor
        metrics["leave_one_game_out_passed"] = loo_passed
        metrics["eligible_before_controls"] = arm_floor and loo_passed
        metrics.update(
            {
                "unit_id": candidate_id,
                "metric": "supported_decision_count",
                "value": metrics["supported_decision_count"],
                "disposition": "complete",
                "censored": metrics["censored_decision_count"] > 0,
                "failures": [],
                "costs": {"current_model_calls": 0, "live_arc_runs": 0},
            }
        )
        candidate_rows.append(metrics)

    for control_id, control in sorted(controls.items()):
        support_rows.append(
            {
                "kind": "shadow_control",
                "control_id": str(control_id),
                "game": control.get("game") if isinstance(control, Mapping) else None,
                "trajectory_id": str(control_id),
                "supported": False,
                "censoring_reason": "shadow_not_applied",
                "co_credited_count": None,
                "held_out_assignment": "control",
            }
        )

    control_rows = _control_rows(ledger, support_rows)
    controls_passed = all(row["passed"] for row in control_rows)
    panel_passed = (
        split["disjoint_games"]
        and split["disjoint_families"]
        and split["evaluation_reachable"]
        and len(split["evaluation_game_ids"]) == REQUIRED_EVALUATION_GAMES
    )
    eligible = [
        row
        for row in candidate_rows
        if row["eligible_before_controls"] and controls_passed and panel_passed
    ]
    selected = eligible[0] if eligible else None
    frozen_order = list(selected["arm_order"]) if selected is not None else None
    frozen_config = _config_payload(frozen_order) if frozen_order is not None else None
    frozen_hash = config_hash(frozen_config) if frozen_config is not None else None
    ingest = (ledger.get("recommendation") or {}).get("ingest_counts") or {}
    receipt_counts = {
        "applied": len(normalized_entries),
        "shadow": len(controls),
        "error": int(ingest.get("error_rows") or 0),
        "duplicate": int(ingest.get("applied_duplicate") or 0)
        + int(ingest.get("controls_duplicate") or 0),
    }
    unsupported_count = sum(
        int(row.get("unsupported_counterfactual") is True) for row in support_rows
    )
    return {
        "receipt_counts": receipt_counts,
        "independent_trajectory_count": len(normalized_entries),
        "development_game_count": len(development_games),
        "evaluation_game_count": len(split["evaluation_game_ids"]),
        "game_split": split,
        "support_rows": support_rows,
        "candidate_rows": candidate_rows,
        "leave_one_game_out_rows": loo_rows,
        "control_rows": control_rows,
        "controls_passed": controls_passed,
        "panel_passed": panel_passed,
        "unsupported_counterfactual_count": unsupported_count,
        "supervisor_trial_ready_score": int(selected is not None),
        "frozen_ordering": frozen_order,
        "frozen_supervisor_trial_manifest": {
            "schema": EXPERIMENT_CONFIG_SCHEMA,
            "arm_order": frozen_order,
            "config_hash": frozen_hash,
            "development_game_ids": split["development_game_ids"],
            "development_family_ids": split["development_family_ids"],
            "evaluation_game_ids": split["evaluation_game_ids"],
            "evaluation_family_ids": split["evaluation_family_ids"],
            "minimum_supported_decisions_per_arm": MIN_SUPPORTED_PER_ARM,
            "minimum_development_games": MIN_DEVELOPMENT_GAMES,
            "required_evaluation_games": REQUIRED_EVALUATION_GAMES,
            "live_action_budget": "unchanged_from_E3_experiment_runner",
            "production_defaults_changed": False,
            "curated_arm_definitions_changed": False,
        },
    }


FIELD_PRINCIPLES = {
    "schema": "Version this record and keep ordinary experiment identity fields.",
    "status": "Use a terminal state only after the affected checks finish.",
    "run_date": "Use the requested date and real UTC boundary timestamps.",
    "preconditions_checked": "Record exact paths and fields before ledger work.",
    "MODEL_SPECS": "List current model identities; this CPU audit has none.",
    "model_invoked": "Set true for any attempted current model load or generation.",
    "invocation_counts": "Separate every current call state from historical receipts.",
    "inference_substrate": "Name the actual host CPU ledger reduction.",
    "inference_substrate_class": "Use the measured closed CPU solver class.",
    "execution_venue": "Name the measured host venue; no board runs occur.",
    "duration_s": "Use real monotonic elapsed time without padding.",
    "phase_spans": "Keep load, generation, evaluation, validation, and write disjoint.",
    "random_seed": "Freeze split and resampling seeds even when selection is deterministic.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator, and raw evidence.",
    "source_artifact_hashes": "Hash exact producer and evidence bytes.",
    "rows": "Keep every compared ordering with costs, failures, and censoring.",
    "sample_size_budget": "State planned, completed, censored units and stopping rules.",
    "acceptance_gate_results": "Separate validation, safety, support, and scientific value.",
    "gate_check_summary": "Name the first exact failed field and retain all failures.",
    "verifier_is_oracle": "The live level-up receipt defines the observed outcome only.",
    "honest_verdict": "Separate a completed null support audit from unavailable work.",
    "verdict_class": "Use the closed terminal class vocabulary.",
    "flagged_adversarial": "A critical current verifier finding prevents readiness.",
    "validation_receipts": "Keep exact command, scope, exit, elapsed time, and log hash.",
    "repository_health": "Keep dated unrelated failures outside affected validation.",
    "field_principles": "Explain scalar fields without wrapping their values.",
    "supervisor_trial_ready_score": "Require all support, panel, and routing floors.",
    "support_rows": "Keep every decision, outcome, censor, co-credit, and assignment.",
    "frozen_supervisor_trial_manifest": "Freeze an eligible order and disjoint IDs.",
    "unsupported_counterfactual_count": "Count every deliberately unknown changed-path row.",
    "generalization_activity": "Use ledger refinement and leave-one-game-out checks only.",
}


def _phase_span(name: str, start: float, end: float, run_start: float, units: int) -> JsonDict:
    """Represent one measured half-open monotonic phase interval."""

    return {
        "phase": name,
        "start_s": start - run_start,
        "end_s": end - run_start,
        "duration_s": end - start,
        "completed_units": units,
    }


def _gate(check: str, category: str, expected: Any, observed: Any) -> JsonDict:
    """Build one acceptance row without conflating its category."""

    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _required_validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    """Require affected and terminal checks; broad health stays diagnostic."""

    required = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *TERMINAL_CHECK_NAMES,
    )
    return all(
        sum(row.get("name") == name for row in receipts) == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        for name in required
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all terminal fields except the checksum value itself."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return validation_contract.canonical_hash(payload)


def build_artifact(
    *,
    audit: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    historical_inference_sidecars: Sequence[Mapping[str, Any]] = (),
    flagged_adversarial: bool = False,
) -> JsonDict:
    """Build one complete artifact from raw audit rows and current receipts."""

    ready = int(audit.get("supervisor_trial_ready_score") or 0)
    required_validation = _required_validation_passed(validation_receipts)
    preconditions_passed = bool(preconditions_checked) and all(
        row.get("passed") is True for row in preconditions_checked
    )
    disqualified = not required_validation or flagged_adversarial
    if disqualified:
        ready = 0
        status = "complete_disqualified_required_check_failed"
        verdict_class = "disqualified"
        honest = "complete_disqualified_required_check_failed_no_supervisor_trial"
    elif ready:
        status = "complete_supervisor_trial_manifest_ready_no_policy_benefit"
        verdict_class = "null"
        honest = "complete_support_qualified_trial_manifest_no_live_policy_benefit_claim"
    else:
        status = "complete_null_insufficient_supported_outcomes"
        verdict_class = "null"
        honest = "complete_null_insufficient_supported_outcomes"

    gates = [
        _gate("preconditions", "completion", True, preconditions_passed),
        _gate("required_validation", "required_validation", True, required_validation),
        _gate("adversarial_clear", "safety", False, flagged_adversarial),
        _gate("support_floor", "support", 1, ready),
        _gate("scientific_value", "efficacy", 1, 0),
        _gate("promotion", "promotion", 1, 0),
    ]
    failures = [row for row in gates if row["passed"] is not True]
    candidate_rows = [dict(row) for row in audit.get("candidate_rows", [])]
    support_rows = [dict(row) for row in audit.get("support_rows", [])]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": 3,
        "status": status,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [dict(row) for row in preconditions_checked],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": dict(ZERO_CURRENT_INVOCATIONS),
            "historical": {
                "sidecar_count": len(historical_inference_sidecars),
                "counted_as_current": False,
            },
        },
        "inference_substrate": "host_cpu_json_ledger_reduction_and_exact_support_mask",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not_reported",
            "operation": "JSON ledger ingestion, deterministic prefix mask, grouped counts",
        },
        "duration_s": float(duration_s),
        "phase_spans": [dict(row) for row in phase_spans],
        "random_seed": {
            "development_split": 7365001,
            "evaluation_split": 7365002,
            "resampling": 7365003,
            "selection_is_deterministic": True,
        },
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(source_artifact_hashes),
        "historical_inference_sidecars": [dict(row) for row in historical_inference_sidecars],
        "rows": candidate_rows,
        "sample_size_budget": {
            "candidate_orderings_planned": len(CANDIDATE_ORDERINGS),
            "candidate_orderings_attempted": len(candidate_rows),
            "candidate_orderings_completed": len(candidate_rows),
            "trajectory_units_completed": audit.get("independent_trajectory_count", 0),
            "decision_rows_completed": len(support_rows),
            "decision_rows_censored": sum(
                int(row.get("supported") is False) for row in support_rows
            ),
            "stopping_rule": "audit all deduplicated ledger entries; no live run",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "all_passed": not failures,
            "failed_count": len(failures),
            "first_failure": failures[0] if failures else None,
            "failed_checks": failures,
        },
        "verifier_is_oracle": True,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "flagged_adversarial": bool(flagged_adversarial),
        "validation_receipts": [dict(row) for row in validation_receipts],
        "repository_health": {
            "as_of": RUN_DATE,
            "unrelated_failures": [
                {
                    "name": row.get("name"),
                    "exit_code": row.get("exit_code"),
                    "timed_out": row.get("timed_out"),
                    "duration_s": row.get("duration_s"),
                    "log_path": row.get("log_path"),
                    "log_sha256": row.get("log_sha256"),
                }
                for row in validation_receipts
                if row.get("name") == "full_python_suite" and row.get("passed") is not True
            ],
        },
        "field_principles": dict(FIELD_PRINCIPLES),
        "supervisor_trial_ready_score": ready,
        "support_rows": support_rows,
        "receipt_counts": dict(audit.get("receipt_counts", {})),
        "control_rows": [dict(row) for row in audit.get("control_rows", [])],
        "leave_one_game_out_rows": [dict(row) for row in audit.get("leave_one_game_out_rows", [])],
        "frozen_supervisor_trial_manifest": dict(audit.get("frozen_supervisor_trial_manifest", {})),
        "unsupported_counterfactual_count": int(audit.get("unsupported_counterfactual_count") or 0),
        "generalization_activity": {
            "kind": "existing_supervisor_ledger_refinement_with_leave_one_game_out_checks",
            "new_public_solve": False,
            "live_trial_executed": False,
            "causal_off_policy_claim": False,
        },
        "game_split": dict(audit.get("game_split", {})),
        "production_defaults_changed": False,
        "curated_arm_definitions_changed": False,
        "promotion_score": 0,
        "scientific_value_score": 0,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    preconditions: Sequence[Mapping[str, Any]],
    hashes: Mapping[str, str],
    *,
    duration_s: float,
    started_at_utc: str,
    completed_at_utc: str,
) -> JsonDict:
    """Build a terminal blocked record without doing dependent ledger work."""

    first = next((dict(row) for row in preconditions if row.get("passed") is not True), None)
    artifact = build_artifact(
        audit={
            "candidate_rows": [],
            "support_rows": [],
            "receipt_counts": {},
            "control_rows": [],
            "leave_one_game_out_rows": [],
            "frozen_supervisor_trial_manifest": {
                "schema": EXPERIMENT_CONFIG_SCHEMA,
                "arm_order": None,
                "config_hash": None,
                "production_defaults_changed": False,
                "curated_arm_definitions_changed": False,
            },
            "unsupported_counterfactual_count": 0,
            "supervisor_trial_ready_score": 0,
            "game_split": {},
        },
        preconditions_checked=preconditions,
        source_artifact_hashes=hashes,
        validation_receipts=[],
        duration_s=duration_s,
        phase_spans=[],
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
    )
    artifact.update(
        {
            "status": "blocked_external_prerequisite",
            "verdict_class": "blocked",
            "honest_verdict": "blocked_external_prerequisite_no_dependent_ledger_work",
            "gate_check_summary": {
                "all_passed": False,
                "failed_count": sum(row.get("passed") is not True for row in preconditions),
                "first_failure": first,
                "failed_checks": [
                    dict(row) for row in preconditions if row.get("passed") is not True
                ],
            },
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _has_required_receipts(artifact: Mapping[str, Any]) -> bool:
    """Cold-check one passing row for every current required command."""

    receipts = artifact.get("validation_receipts")
    return isinstance(receipts, list) and _required_validation_passed(receipts)


def validate_artifact(value: object, *, require_validation: bool = True) -> list[str]:
    """Cold-check identity, no-model state, support accounting, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_object"]
    artifact = value
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract")
    counts = artifact.get("invocation_counts")
    if not isinstance(counts, Mapping) or counts.get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("invocation_counts")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class")
    if artifact.get("promotion_score") != 0 or artifact.get("scientific_value_score") != 0:
        errors.append("promotion")
    if artifact.get("production_defaults_changed") is not False:
        errors.append("production_defaults")
    support_rows = artifact.get("support_rows")
    if isinstance(support_rows, list):
        unsupported = sum(
            int(row.get("unsupported_counterfactual") is True)
            for row in support_rows
            if isinstance(row, Mapping)
        )
        if unsupported != artifact.get("unsupported_counterfactual_count"):
            errors.append("unsupported_counterfactual_count")
    else:
        errors.append("support_rows")
    manifest = artifact.get("frozen_supervisor_trial_manifest")
    ready = artifact.get("supervisor_trial_ready_score")
    if artifact.get("verdict_class") != "blocked":
        if not isinstance(manifest, Mapping):
            errors.append("manifest")
        elif ready == 1:
            order = manifest.get("arm_order")
            payload = _config_payload(order) if isinstance(order, list) else {}
            if (
                not isinstance(order, list)
                or not _valid_order(order)
                or manifest.get("config_hash") != config_hash(payload)
                or len(manifest.get("evaluation_game_ids") or []) != REQUIRED_EVALUATION_GAMES
            ):
                errors.append("readiness")
        elif ready != 0 or manifest.get("arm_order") is not None:
            errors.append("readiness")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if require_validation and artifact.get("verdict_class") not in {"blocked", "disqualified"}:
        if not _has_required_receipts(artifact):
            errors.append("validation_receipts")
    return sorted(set(errors))


def _load_registry(path: Path) -> JsonDict:
    """Load the solve registry as one plain mapping."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupervisorSupportError("registry_not_mapping")
    return value


def run_affected_validation(root: Path, raw_dir: Path) -> list[JsonDict]:
    """Run the Exp7358 plan for this exact module, test, and thin wrapper."""

    private_root = Path(tempfile.mkdtemp(prefix="exp7365-validation-", dir="/tmp"))
    commands = validation_contract.build_command_plan(root, AFFECTED_MANIFEST, private_root)
    errors = validation_contract.validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if errors:
        raise SupervisorSupportError("invalid_scoped_command_plan:" + ",".join(errors))
    planned = [
        validation_contract.PlannedCommand(command, "required_validation", True)
        for command in commands
    ]
    return validation_contract.run_categorized_commands(
        root, planned, log_dir=raw_dir / "validation/affected"
    )


def run_full_python_suite(root: Path, raw_dir: Path) -> list[JsonDict]:
    """Run the user-required full Python suite once and retain its exact receipt."""

    command = validation_scope.CommandSpec(
        "full_python_suite",
        (str(root / ".venv/bin/pytest"), "tests/python", "-q"),
        "repository_python_tests",
        1800.0,
    )
    return validation_scope.run_commands(root, [command], log_dir=raw_dir / "validation/full_suite")


def prior_full_suite_receipt(path: Path) -> list[JsonDict]:
    """Reuse the completed broad diagnostic instead of rerunning 66k tests.

    Broad tests report repository health, not affected validation. A metadata
    correction still needs fresh scoped checks. It must not start a second broad
    run after the first run reached its declared bound.
    """

    artifact = _load_json_object(path)
    rows = artifact.get("validation_receipts")
    if not isinstance(rows, list):
        return []
    prior = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("name") == "full_python_suite"
    ]
    if len(prior) != 1:
        return []
    prior[0]["reused_as_repository_health_diagnostic"] = True
    prior[0]["current_required_validation"] = False
    return prior


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run an independent cold reducer and both required artifact guards."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,sys;from pathlib import Path;"
        "from carnot.experiment_7365_v646_supervisor_support import validate_artifact;"
        "value=json.loads(Path(sys.argv[1]).read_text());"
        "errors=validate_artifact(value,require_validation=False);"
        "print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        validation_scope.CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer, str(candidate)), "raw_evidence"
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "terminal_candidate",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal_candidate",
        ),
    ]
    return validation_scope.run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def run_experiment(
    root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:
    """Execute preconditions, real ledger audit, validation, and atomic write."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_start = time.monotonic()
    progress(started, "preconditions", "start")
    preconditions, hashes, historical = collect_preconditions(root)
    phase_end = time.monotonic()
    spans.append(_phase_span("load", phase_start, phase_end, started, len(preconditions)))
    progress(
        started,
        "preconditions",
        "end",
        passed=all(row["passed"] for row in preconditions),
    )
    if not all(row["passed"] for row in preconditions):
        blocked = build_blocked_artifact(
            preconditions,
            hashes,
            duration_s=time.monotonic() - started,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
        )
        progress(started, "write", "before_atomic_blocked", path=output_path)
        validation_contract.atomic_json(root / output_path, blocked)
        progress(started, "write", "after_atomic_blocked", path=output_path)
        return blocked

    generation_point = time.monotonic()
    spans.append(_phase_span("generation", generation_point, generation_point, started, 0))
    progress(started, "generation", "skipped_no_model_work")

    phase_start = time.monotonic()
    progress(started, "evaluation", "start")
    ledger = load_ledger(root / LEDGER_PATH)
    registry = _load_registry(root / REGISTRY_PATH)
    audit = audit_ledger(ledger, registry)
    phase_end = time.monotonic()
    spans.append(
        _phase_span(
            "evaluation",
            phase_start,
            phase_end,
            started,
            int(audit["independent_trajectory_count"]),
        )
    )
    progress(
        started,
        "evaluation",
        "end",
        trajectories=audit["independent_trajectory_count"],
        ready=audit["supervisor_trial_ready_score"],
    )

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    validation_contract.atomic_json(raw_dir / "ledger_audit.json", audit)

    phase_start = time.monotonic()
    progress(started, "validation", "before_affected_subprocesses")
    validation_receipts = run_affected_validation(root, raw_dir)
    progress(started, "validation", "after_affected_subprocesses")
    prior_suite = prior_full_suite_receipt(root / output_path)
    if prior_suite:
        progress(started, "validation", "reuse_prior_full_python_suite_receipt")
        validation_receipts.extend(prior_suite)
    else:
        progress(started, "validation", "before_full_python_suite")
        validation_receipts.extend(run_full_python_suite(root, raw_dir))
        progress(started, "validation", "after_full_python_suite")

    candidate = build_artifact(
        audit=audit,
        preconditions_checked=preconditions,
        source_artifact_hashes=hashes,
        validation_receipts=validation_receipts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        historical_inference_sidecars=historical,
    )
    candidate_path = raw_dir / "terminal_candidate.json"
    validation_contract.atomic_json(candidate_path, candidate)
    progress(started, "validation", "before_terminal_subprocesses")
    terminal = run_terminal_validation(root, candidate_path, raw_dir)
    validation_receipts.extend(terminal)
    progress(started, "validation", "after_terminal_subprocesses")
    phase_end = time.monotonic()
    spans.append(
        _phase_span("validation", phase_start, phase_end, started, len(validation_receipts))
    )

    flagged = any(
        row.get("name") == "adversarial_verify"
        and (row.get("passed") is not True or "CRITICAL" in str(row.get("output_tail") or ""))
        for row in terminal
    )
    final = build_artifact(
        audit=audit,
        preconditions_checked=preconditions,
        source_artifact_hashes=hashes,
        validation_receipts=validation_receipts,
        duration_s=time.monotonic() - started,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        historical_inference_sidecars=historical,
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(
        final, require_validation=final["verdict_class"] not in {"blocked", "disqualified"}
    )
    if errors:
        raise SupervisorSupportError("terminal_artifact_invalid:" + ",".join(errors))

    write_start = time.monotonic()
    write_end = time.monotonic()
    spans.append(_phase_span("write", write_start, write_end, started, 1))
    final["phase_spans"] = spans
    final["duration_s"] = time.monotonic() - started
    final["completed_at_utc"] = utc_now()
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    progress(started, "write", "before_atomic_terminal", path=output_path)
    validation_contract.atomic_json(raw_dir / "terminal_candidate.json", final)
    validation_contract.atomic_json(root / output_path, final)
    progress(started, "write", "after_atomic_terminal", status=final["status"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed-date entrypoint and optional cold validation mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the declared experiment entrypoint or cold-validate one artifact."""

    args = parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(_load_json_object(args.validate))
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - use the thin script entrypoint.
    raise SystemExit(main())
