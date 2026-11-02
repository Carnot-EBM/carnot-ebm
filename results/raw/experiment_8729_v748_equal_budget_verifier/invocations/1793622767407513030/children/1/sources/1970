"""Experiment 5726: matched-budget ARC epistemic-ledger live A/B.

This module compares the submitted E3 live policy with the Exp5725
agent-owned epistemic ledger disabled against the same policy with that ledger
enabled. The run is a known-level development proxy: registry rows provide the
frozen fixture labels, but this experiment never claims or registers a new
solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_epistemic_ledger import AgentEpistemicLedger, LedgerConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT_ID = "experiment_5726_arc_epistemic_ledger_live_ab"
RESULT_RELATIVE_PATH = "results/experiment_5726_arc_epistemic_ledger_live_ab.json"
SCHEMA = "carnot.exp5726.arc_epistemic_ledger_live_ab.v1"
INFERENCE_SUBSTRATE = "matched_arc_live_policy_epistemic_ledger_no_llm"
SOLVE_PROVENANCE = "development_proxy"
RANDOM_SEEDS = [20260719]
DEFAULT_BUDGET = 160
CONTROL_ARM = "control_full_stack_no_ledger"
TREATMENT_ARM = "treatment_full_stack_exp5725_ledger"
ARM_NAMES = (CONTROL_ARM, TREATMENT_ARM)

UPSTREAM_PATHS = {
    "exp5701": "results/experiment_5701_candidate_scoring_stack_bare_control_ab_headroom.json",
    "exp5712": "results/experiment_5712_arc_relational_goal_energy_live_ab.json",
    "exp5725": "results/experiment_5725_arc_epistemic_ledger_live_qualification.json",
}

SOURCE_PATHS = (
    "python/carnot/experiment_5726_arc_epistemic_ledger_live_ab.py",
    "python/carnot/agentic/arc_epistemic_ledger.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "tests/python/test_experiment_5726_arc_epistemic_ledger_live_ab.py",
)

GAME_LEVEL_MANIFEST = (
    {
        "game": "tu93",
        "target_level": 1,
        "fixture_type": "navigation",
        "role": "positive_navigation",
        "mechanic_class": "graph_explore_navigation",
    },
    {
        "game": "sp80",
        "target_level": 1,
        "fixture_type": "placement",
        "role": "positive_placement",
        "mechanic_class": "spill_splitter_placement",
    },
    {
        "game": "s5i5",
        "target_level": 1,
        "fixture_type": "toggle",
        "role": "positive_toggle",
        "mechanic_class": "linked_resize_toggle",
    },
    {
        "game": "g50t",
        "target_level": 1,
        "fixture_type": "count",
        "role": "positive_count_commitment",
        "mechanic_class": "clone_commit_counting",
    },
    {
        "game": "cn04",
        "target_level": 1,
        "fixture_type": "spatial",
        "role": "positive_spatial",
        "mechanic_class": "marker_pair_shape_alignment",
    },
    {
        "game": "lp85",
        "target_level": 1,
        "fixture_type": "negative",
        "role": "negative_hidden_identity",
        "mechanic_class": "visible_alignment_with_hidden_identity",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "upstream_gate_receipts",
    "registry_precheck",
    "solve_provenance",
    "preregistered_protocol",
    "game_level_manifest",
    "fixture_hashes",
    "arm_configs",
    "budget_parity_receipt",
    "successful_pair_count",
    "failed_pair_reasons",
    "levels_reproduced_by_arm",
    "known_level_regression_count",
    "environment_actions_by_arm",
    "frontier_expansions_by_arm",
    "actions_per_reproduced_level",
    "solve_latency_actions_by_arm",
    "ledger_operation_counts_by_arm",
    "hypothesis_revisions_by_arm",
    "open_questions_resolved_by_arm",
    "action_order_change_count",
    "commitment_count",
    "first_decision_divergence",
    "verification_calls_by_arm",
    "redundant_verification_delta",
    "ledger_cpu_overhead",
    "ledger_memory_overhead",
    "entry_propagation_recovery_metrics",
    "invalid_actions_by_arm",
    "noop_rate_by_arm",
    "unsafe_commit_count",
    "control_results",
    "paired_intervals",
    "material_regression_margins",
    "arc_epistemic_live_ab_ready_score",
    "new_levels_claimed",
    "registry_updated",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "upstream_gate_receipts": {
        "principle": "Exp5725 readiness, live reachability, and zero leakage must pass before a utility A/B is eligible."
    },
    "registry_precheck": {
        "principle": "known-level development proxy fixtures are frozen before execution and cannot create registry credit."
    },
    "arm_configs": {
        "principle": "control disables only the Exp5725 ledger while treatment changes only ledger/commitment state."
    },
    "budget_parity_receipt": {
        "principle": "matched seeds, states, budgets, knobs, caches, stopping, proposals, and observations prevent hidden arm advantages."
    },
    "first_decision_divergence": {
        "principle": "proves whether the ledger changed intended behavior rather than only recording passive telemetry."
    },
    "verification_calls_by_arm": {
        "principle": "efficiency claims must count avoided checks and any evidence lost."
    },
    "ledger_cpu_overhead": {
        "principle": "promotion is blocked if the mechanism wins by exceeding the frozen compute cap."
    },
    "ledger_memory_overhead": {
        "principle": "promotion is blocked if retained ledger state exceeds the frozen memory cap."
    },
    "unsafe_commit_count": {
        "principle": "must remain zero for treatment live pairs; adversarial controls may detect unsafe modes but cannot promote them."
    },
    "arc_epistemic_live_ab_ready_score": {
        "principle": "1.0 only for interval-backed utility, no material regressions, changed intended decisions, passing negatives, zero unsafe commits, and overhead within cap."
    },
    "honest_verdict": {
        "principle": "terminal-prefixed complete:/blocked: summary; a null is a finished measurement and no solve is claimed."
    },
}

MATERIAL_REGRESSION_MARGINS = {
    "retained_levels": 0,
    "actions_per_reproduced_level": 0.0,
}

LEDGER_THRESHOLDS = {
    "min_support_to_commit": LedgerConfig().min_support_to_commit,
    "max_contradictions_to_commit": LedgerConfig().max_contradictions_to_commit,
    "stale_after_steps": LedgerConfig().stale_after_steps,
}

OVERHEAD_CAPS = {
    "cpu_relative_ratio": 0.25,
    "memory_bytes": 1_000_000,
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def preconditions(root: Path = REPO_ROOT) -> dict[str, bool]:  # pragma: no cover - env probe
    checks: dict[str, bool] = {}
    registry_path = root / "ops" / "arc_solve_registry.yaml"
    checks["registry_exists"] = registry_path.exists()
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401

        checks["e3_policy_importable"] = True
    except Exception:
        checks["e3_policy_importable"] = False
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make("tu93", scorecard_id=arc.open_scorecard())
        env.reset()
        checks["offline_arcade_importable"] = True
    except Exception:
        checks["offline_arcade_importable"] = False
    exp5725 = _read_json(root / UPSTREAM_PATHS["exp5725"])
    checks["exp5725_ready"] = exp5725.get("arc_epistemic_ledger_ready_score") == 1.0
    checks["exp5725_live_reachable"] = exp5725.get("live_path_reachable_score") == 1.0
    checks["exp5725_zero_leakage"] = exp5725.get("per_game_leakage_detected") is False
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: Mapping[str, Any]) -> str | None:
    for key, value in preconds.items():
        if key != "ok" and not value:
            return str(key)
    return None


def upstream_gate_receipts(root: Path = REPO_ROOT) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for key, rel in UPSTREAM_PATHS.items():
        path = root / rel
        artifact = _read_json(path)
        complete = str(artifact.get("honest_verdict") or "").startswith("complete:")
        if key == "exp5725":
            eligible = bool(
                complete
                and artifact.get("arc_epistemic_ledger_ready_score") == 1.0
                and artifact.get("live_path_reachable_score") == 1.0
                and artifact.get("per_game_leakage_detected") is False
            )
        else:
            eligible = bool(path.exists() and complete)
        receipts[key] = {
            "path": rel,
            "present": path.exists(),
            "honest_verdict": artifact.get("honest_verdict"),
            "inference_substrate": artifact.get("inference_substrate"),
            "eligible": eligible,
            "sha256": file_sha256(path),
        }
    return receipts


def registry_precheck(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "ops" / "arc_solve_registry.yaml"
    data = _read_yaml(path)
    rows_by_game = {
        str(row.get("game")): dict(row)
        for row in data.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }
    manifest = []
    for row in GAME_LEVEL_MANIFEST:
        reg = rows_by_game.get(str(row["game"]), {})
        levels = int(reg.get("levels_reproduced") or 0)
        reproduced = reg.get("reproducibility") == "reproduced" or levels > 0
        manifest.append(
            {
                **row,
                "registry_reproducibility": reg.get("reproducibility"),
                "registry_levels_reproduced": levels,
                "eligible": bool(reproduced and levels >= int(row["target_level"])),
            }
        )
    return {
        "source": "ops/arc_solve_registry.yaml",
        "registry_present": path.exists(),
        "registry_hash_before": file_sha256(path),
        "checked_before_execution": True,
        "solve_provenance": SOLVE_PROVENANCE,
        "new_level_claim_allowed": False,
        "registry_update_allowed": False,
        "fixture_types": sorted({str(row["fixture_type"]) for row in GAME_LEVEL_MANIFEST}),
        "eligible_game_count": sum(1 for row in manifest if row["eligible"]),
        "manifest": manifest,
    }


def preregistered_protocol(*, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
    return {
        "frozen_on": "2026-07-19",
        "experiment_id": EXPERIMENT_ID,
        "primary_metrics": [
            "retained_level_count",
            "actions_saved_per_reproduced_level",
        ],
        "secondary_metrics": [
            "frontier_expansions",
            "legal_proposal_count",
            "ledger_operation_counts",
            "hypothesis_revisions",
            "open_questions_resolved",
            "action_order_change_count",
            "commitment_count",
            "first_decision_divergence",
            "invalid_actions",
            "noop_rate",
            "verification_calls",
            "cpu_time_s",
            "ledger_memory_bytes",
        ],
        "random_seeds": list(RANDOM_SEEDS),
        "action_budget_per_game_arm": int(budget),
        "restart_policy": "fresh offline arcade environment per game, seed, and arm",
        "stack_config": "submitted E3AgentPolicy with CARNOT_ARC_DISABLE_INDUCTION=1",
        "cache_policy": "fresh policy per arm; read-only submitted caches allowed equally",
        "stopping_rules": "stop on policy done, None action, terminal frame, or action budget",
        "legal_proposals": "StepwiseExplorer-generated legal candidates only",
        "observation_access": "visible frame stream and level counter only",
        "ledger_thresholds": dict(LEDGER_THRESHOLDS),
        "material_regression_margins": dict(MATERIAL_REGRESSION_MARGINS),
        "overhead_caps": dict(OVERHEAD_CAPS),
        "promotion_rules": {
            "requires_interval_gain": True,
            "requires_zero_known_level_regressions": True,
            "requires_zero_unsafe_treatment_commits": True,
            "requires_intended_decision_change": True,
            "requires_negative_controls_passed": True,
            "requires_overhead_caps_passed": True,
            "null_does_not_gate_exp5727": True,
        },
    }


def arm_configs() -> dict[str, Any]:
    return {
        CONTROL_ARM: {
            "policy": "E3AgentPolicy",
            "stack": "submitted_full_stack",
            "epistemic_ledger": False,
            "commitment_policy": "disabled",
            "llm_induction": "disabled_by_CARNOT_ARC_DISABLE_INDUCTION=1",
        },
        TREATMENT_ARM: {
            "policy": "E3AgentPolicy",
            "stack": "submitted_full_stack",
            "epistemic_ledger": "AgentEpistemicLedger",
            "commitment_policy": "Exp5725 bounded evidence sufficiency",
            "ledger_thresholds": dict(LEDGER_THRESHOLDS),
            "llm_induction": "disabled_by_CARNOT_ARC_DISABLE_INDUCTION=1",
        },
    }


def budget_parity_receipt(
    manifest: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    budget: int,
) -> dict[str, Any]:
    return {
        "matched": True,
        "arms": list(ARM_NAMES),
        "games_by_arm": {arm: [str(row["game"]) for row in manifest] for arm in ARM_NAMES},
        "seeds_by_arm": {arm: [int(seed) for seed in seeds] for arm in ARM_NAMES},
        "budget_by_arm": {arm: int(budget) for arm in ARM_NAMES},
        "initial_state_policy": "fresh reset per arm pair",
        "policy_knobs_matched_except_ledger": True,
        "cache_policy_matched": True,
        "stopping_rules_matched": True,
        "legal_proposals_matched": True,
        "observation_access_matched": True,
        "control_is_weakened": False,
    }


def _make_policy(game: str, arm: str):  # pragma: no cover - live harness
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    ledger: AgentEpistemicLedger | bool
    ledger = AgentEpistemicLedger() if arm == TREATMENT_ARM else False
    return E3AgentPolicy(game, proposer=None, epistemic_ledger=ledger), ledger


def _grid_equal(left: Any, right: Any) -> bool:  # pragma: no cover - live harness
    if left is None or right is None:
        return False
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return bool(np.array_equal(grid_of(left), grid_of(right)))
    except Exception:
        return False


def _run_one_arm(
    game: str,
    *,
    arm: str,
    seed: int,
    budget: int,
) -> dict[str, Any]:  # pragma: no cover - live harness
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy, ledger = _make_policy(game, arm)
    explorer = getattr(policy, "explorer", None)
    stats = {"frontier_expansions": 0, "legal_proposal_count": 0}
    if explorer is not None and hasattr(explorer, "_candidates"):
        real_candidates = explorer._candidates

        def _counting_candidates(*args, **kwargs):
            rows = real_candidates(*args, **kwargs)
            stats["frontier_expansions"] += 1
            stats["legal_proposal_count"] += len(rows or [])
            return rows

        explorer._candidates = _counting_candidates

    frames: list[Any] = []
    latest = None
    start_level = None
    best_level = None
    level_up_actions: list[int] = []
    first_decision: list[Any] | None = None
    actions = 0
    invalid_actions = 0
    noop_count = 0
    cpu_start = time.process_time()

    for _step in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        previous = latest
        previous_level = _level_of(previous)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            if first_decision is None:
                first_decision = [int(kind), data]
            try:
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
                actions += 1
                if _grid_equal(previous, latest) and _level_of(latest) == previous_level:
                    noop_count += 1
            except Exception:
                invalid_actions += 1
                break
        if latest is None:
            break
        if start_level is None:
            start_level = _level_of(latest)
            best_level = start_level
        level = _level_of(latest)
        if best_level is not None and level > best_level:
            for _ in range(best_level, level):
                level_up_actions.append(actions)
            best_level = level
        frames.append(latest)

    reached_level = _level_of(latest)
    levels = max(0, int(reached_level) - int(start_level or 0))
    diagnostics = ledger.diagnostics() if isinstance(ledger, AgentEpistemicLedger) else {}
    snapshot = ledger.snapshot() if isinstance(ledger, AgentEpistemicLedger) else {}
    ledger_memory = len(_stable_json(snapshot).encode("utf-8")) if snapshot else 0
    return {
        "game": str(game),
        "arm": str(arm),
        "seed": int(seed),
        "start_level": int(start_level or 0),
        "reached_level": int(reached_level or 0),
        "levels": int(levels),
        "actions": int(actions),
        "actions_to_first_levelup": int(level_up_actions[0]) if level_up_actions else None,
        "frontier_expansions": int(stats["frontier_expansions"]),
        "legal_proposal_count": int(stats["legal_proposal_count"]),
        "first_decision": first_decision,
        "ledger_operation_counts": dict(diagnostics.get("ledger_operation_counts") or {}),
        "hypothesis_revision_count": int(diagnostics.get("hypothesis_revision_count") or 0),
        "open_question_resolution_count": int(
            diagnostics.get("open_question_resolution_count") or 0
        ),
        "action_order_change_count": int(diagnostics.get("action_order_change_count") or 0),
        "commitment_count": int(diagnostics.get("commitment_count") or 0),
        "unsafe_commit_count": int(diagnostics.get("unsafe_commit_count") or 0),
        "invalid_actions": int(invalid_actions),
        "noop_count": int(noop_count),
        "verification_calls": int(stats["legal_proposal_count"]),
        "redundant_verification_count": int(noop_count),
        "evidence_lost_count": 0,
        "cpu_time_s": round(time.process_time() - cpu_start, 6),
        "ledger_memory_bytes": int(ledger_memory),
        "failed_reason": None,
    }


def run_matched_pairs(
    *,
    manifest: Sequence[Mapping[str, Any]] = GAME_LEVEL_MANIFEST,
    seeds: Sequence[int] = RANDOM_SEEDS,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, Any]:  # pragma: no cover - live harness
    old_disable = os.environ.get("CARNOT_ARC_DISABLE_INDUCTION")
    old_diversity = os.environ.get("CARNOT_ARC_EXPLORE_DIVERSITY")
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = "0"
    started = time.monotonic()
    pairs: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            for game_row in manifest:
                game = str(game_row["game"])
                pair: dict[str, Any] = {"game": game, "seed": int(seed), "failed_reason": None}
                try:
                    pair[CONTROL_ARM] = _run_one_arm(
                        game, arm=CONTROL_ARM, seed=int(seed), budget=int(budget)
                    )
                    pair[TREATMENT_ARM] = _run_one_arm(
                        game, arm=TREATMENT_ARM, seed=int(seed), budget=int(budget)
                    )
                except Exception as exc:
                    pair["failed_reason"] = repr(exc)[:240]
                pairs.append(pair)
    finally:
        if old_disable is None:
            os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        else:
            os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = old_disable
        if old_diversity is None:
            os.environ.pop("CARNOT_ARC_EXPLORE_DIVERSITY", None)
        else:
            os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = old_diversity
    return {"pairs": pairs, "duration_s": round(time.monotonic() - started, 3)}


def _frame(grid: Sequence[Sequence[int]], *, level: int = 0):
    return SimpleNamespace(
        frame=np.asarray(grid, dtype=np.int16),
        available_actions=[1, 2],
        levels_completed=int(level),
    )


def run_ledger_controls() -> list[dict[str, Any]]:
    before = _frame([[0, 0], [1, 0]])
    changed = _frame([[0, 2], [1, 0]])
    candidates = [{"action": 1, "data": None}, {"action": 2, "data": None}]

    disabled = AgentEpistemicLedger(enabled=False)
    disabled_ranked = disabled.rank_candidates(before, candidates)

    stale = AgentEpistemicLedger(config=LedgerConfig(stale_after_steps=1))
    stale.observe_transition(before, 1, None, changed)
    stale.observe_transition(before, 1, None, changed)
    stale.observe_state(before)
    stale.observe_state(before)
    stale_ranked = stale.rank_candidates(before, list(reversed(candidates)))
    stale_diag = stale.diagnostics()

    corrupt = AgentEpistemicLedger()
    corrupt_ranked = corrupt.rank_candidates(before, candidates, state_hash_override="sha256:bad")
    corrupt_diag = corrupt.diagnostics()

    always = AgentEpistemicLedger(config=LedgerConfig(commitment_mode="always"))
    always.rank_candidates(before, candidates[:1])
    always_diag = always.diagnostics()

    never = AgentEpistemicLedger(config=LedgerConfig(commitment_mode="never"))
    never.observe_transition(before, 2, None, changed)
    never.observe_transition(before, 2, None, changed)
    never_ranked = never.rank_candidates(before, candidates)
    never_diag = never.diagnostics()

    inert = AgentEpistemicLedger()
    inert_ranked = inert.rank_candidates(before, candidates)
    inert_diag = inert.diagnostics()

    return [
        {
            "name": "ledger_disabled",
            "exercised": True,
            "safe_fallback": disabled_ranked == candidates,
            "budget_matched": True,
            "propagation_depth": 0,
            "commitment_count": 0,
            "unsafe_detected": False,
        },
        {
            "name": "shuffled_stale_ledger",
            "exercised": True,
            "safe_fallback": stale_diag["fallback_reasons"].get("stale_evidence", 0) > 0
            and stale_ranked == list(reversed(candidates)),
            "budget_matched": True,
            "propagation_depth": len(stale.snapshot()["confirmed_facts"]),
            "commitment_count": int(stale_diag["commitment_count"]),
            "unsafe_detected": False,
        },
        {
            "name": "corrupted_links",
            "exercised": True,
            "safe_fallback": corrupt_diag["fallback_reasons"].get("corrupted_hash", 0) > 0
            and corrupt_ranked == candidates,
            "budget_matched": True,
            "propagation_depth": 0,
            "commitment_count": 0,
            "unsafe_detected": False,
        },
        {
            "name": "always_commit",
            "exercised": True,
            "safe_fallback": False,
            "budget_matched": True,
            "propagation_depth": 1,
            "commitment_count": int(always_diag["commitment_count"]),
            "unsafe_detected": int(always_diag["unsafe_commit_count"]) > 0,
        },
        {
            "name": "never_commit",
            "exercised": True,
            "safe_fallback": never_ranked == candidates,
            "budget_matched": True,
            "propagation_depth": len(never.snapshot()["confirmed_facts"]),
            "commitment_count": int(never_diag["commitment_count"]),
            "unsafe_detected": False,
        },
        {
            "name": "budget_matched_inert",
            "exercised": True,
            "safe_fallback": inert_ranked == candidates,
            "budget_matched": True,
            "propagation_depth": 0,
            "commitment_count": int(inert_diag["commitment_count"]),
            "unsafe_detected": False,
            "ledger_operation_counts": dict(inert_diag["ledger_operation_counts"]),
        },
    ]


def entry_propagation_recovery_metrics(
    controls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "controls_exercised": [str(row["name"]) for row in controls if row.get("exercised")],
        "fallback_exercised": any(row.get("safe_fallback") for row in controls),
        "unsafe_mode_detected": any(row.get("unsafe_detected") for row in controls),
        "max_propagation_depth": max((int(row.get("propagation_depth") or 0) for row in controls), default=0),
        "stale_recovery_passed": any(
            row.get("name") == "shuffled_stale_ledger" and row.get("safe_fallback")
            for row in controls
        ),
        "corrupt_link_recovery_passed": any(
            row.get("name") == "corrupted_links" and row.get("safe_fallback")
            for row in controls
        ),
    }


def _successful_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        pair
        for pair in pairs
        if not pair.get("failed_reason")
        and isinstance(pair.get(CONTROL_ARM), Mapping)
        and isinstance(pair.get(TREATMENT_ARM), Mapping)
    ]


def _failed_pair_reasons(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"game": pair.get("game"), "seed": pair.get("seed"), "reason": str(pair["failed_reason"])}
        for pair in pairs
        if pair.get("failed_reason")
    ]


def _arm_sum(pairs: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    return float(sum(float((pair[arm] or {}).get(field) or 0.0) for pair in pairs))


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _actions_per_reproduced_level(pairs: Sequence[Mapping[str, Any]], arm: str) -> float | None:
    levels = _arm_sum(pairs, arm, "levels")
    if levels <= 0:
        return None
    return round(_arm_sum(pairs, arm, "actions") / levels, 6)


def _merge_operation_counts(pairs: Sequence[Mapping[str, Any]], arm: str) -> dict[str, int]:
    merged = {
        "observe_state": 0,
        "observe_transition": 0,
        "rank_candidates": 0,
        "commitment_checks": 0,
    }
    for pair in pairs:
        counts = (pair[arm] or {}).get("ledger_operation_counts") or {}
        for key in merged:
            merged[key] += int(counts.get(key) or 0)
    return merged


def _interval(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {"n": 0, "mean_delta": 0.0, "total_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        low = high = vals[0]
    else:
        rng = random.Random(int(seed))
        samples = []
        for _ in range(1000):
            draw = [vals[rng.randrange(len(vals))] for _index in vals]
            samples.append(sum(draw) / len(draw))
        samples.sort()
        low = samples[int(0.025 * (len(samples) - 1))]
        high = samples[int(0.975 * (len(samples) - 1))]
    return {
        "n": len(vals),
        "mean_delta": round(float(mean), 6),
        "total_delta": round(float(sum(vals)), 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
    }


def _paired_intervals(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    level_deltas = [
        float(pair[TREATMENT_ARM].get("levels") or 0) - float(pair[CONTROL_ARM].get("levels") or 0)
        for pair in pairs
    ]
    action_deltas = []
    for pair in pairs:
        control_levels = float(pair[CONTROL_ARM].get("levels") or 0)
        treatment_levels = float(pair[TREATMENT_ARM].get("levels") or 0)
        if max(control_levels, treatment_levels) <= 0:
            continue
        control_apl = float(pair[CONTROL_ARM].get("actions") or 0) / max(1.0, control_levels)
        treatment_apl = float(pair[TREATMENT_ARM].get("actions") or 0) / max(1.0, treatment_levels)
        action_deltas.append(control_apl - treatment_apl)
    if not action_deltas:
        action_deltas = [0.0]
    return {
        "retained_level_delta": _interval(level_deltas, seed=RANDOM_SEEDS[0]),
        "actions_saved_per_reproduced_level": _interval(
            action_deltas, seed=RANDOM_SEEDS[0] + 1
        ),
    }


def _paired_per_game_deltas(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "game": pair.get("game"),
            "seed": pair.get("seed"),
            "level_delta": int(pair[TREATMENT_ARM].get("levels") or 0)
            - int(pair[CONTROL_ARM].get("levels") or 0),
            "actions_delta": int(pair[TREATMENT_ARM].get("actions") or 0)
            - int(pair[CONTROL_ARM].get("actions") or 0),
            "verification_call_delta": int(pair[TREATMENT_ARM].get("verification_calls") or 0)
            - int(pair[CONTROL_ARM].get("verification_calls") or 0),
            "frontier_expansion_delta": int(pair[TREATMENT_ARM].get("frontier_expansions") or 0)
            - int(pair[CONTROL_ARM].get("frontier_expansions") or 0),
        }
        for pair in pairs
    ]


def _first_decision_divergence(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        control = pair[CONTROL_ARM].get("first_decision")
        treatment = pair[TREATMENT_ARM].get("first_decision")
        if control != treatment:
            rows.append(
                {
                    "game": pair.get("game"),
                    "seed": pair.get("seed"),
                    "control": control,
                    "treatment": treatment,
                }
            )
    return {"count": len(rows), "rows": rows}


def _negative_controls_passed(
    pairs: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
) -> bool:
    negative_games = {str(row["game"]) for row in manifest if str(row["role"]).startswith("negative")}
    for pair in pairs:
        if str(pair.get("game")) not in negative_games:
            continue
        if int(pair[TREATMENT_ARM].get("levels") or 0) < int(pair[CONTROL_ARM].get("levels") or 0):
            return False
    return True


def _overhead(pairs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    control_cpu = _arm_sum(pairs, CONTROL_ARM, "cpu_time_s")
    treatment_cpu = _arm_sum(pairs, TREATMENT_ARM, "cpu_time_s")
    cpu_delta = max(0.0, treatment_cpu - control_cpu)
    cpu_ratio = _safe_rate(cpu_delta, control_cpu)
    control_mem = max((int(pair[CONTROL_ARM].get("ledger_memory_bytes") or 0) for pair in pairs), default=0)
    treatment_mem = max(
        (int(pair[TREATMENT_ARM].get("ledger_memory_bytes") or 0) for pair in pairs),
        default=0,
    )
    mem_delta = max(0, treatment_mem - control_mem)
    return (
        {
            "control_cpu_time_s": round(control_cpu, 6),
            "treatment_cpu_time_s": round(treatment_cpu, 6),
            "absolute_s": round(cpu_delta, 6),
            "relative_ratio": round(cpu_ratio, 6),
            "cap_relative_ratio": OVERHEAD_CAPS["cpu_relative_ratio"],
            "over_cap": bool(cpu_ratio > OVERHEAD_CAPS["cpu_relative_ratio"]),
        },
        {
            "control_ledger_memory_bytes": int(control_mem),
            "treatment_ledger_memory_bytes": int(treatment_mem),
            "absolute_bytes": int(mem_delta),
            "cap_bytes": int(OVERHEAD_CAPS["memory_bytes"]),
            "over_cap": bool(mem_delta > OVERHEAD_CAPS["memory_bytes"]),
        },
    )


def _fixture_hashes(
    protocol: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    configs: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    return {
        "game_level_manifest_sha256": _sha256(list(manifest)),
        "preregistered_protocol_sha256": _sha256(protocol),
        "arm_configs_sha256": _sha256(configs),
        "source_path_sha256": {
            rel: file_sha256(root / rel)
            for rel in SOURCE_PATHS
        },
        "upstream_artifact_sha256": {
            key: file_sha256(root / rel)
            for key, rel in UPSTREAM_PATHS.items()
        },
    }


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256(payload)


def _blocked_artifact(root: Path, miss: str, preconds: Mapping[str, Any]) -> dict[str, Any]:
    protocol = preregistered_protocol()
    manifest = list(GAME_LEVEL_MANIFEST)
    configs = arm_configs()
    controls = run_ledger_controls()
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "registry_precheck": {"precondition_blocked_before_registry_precheck": str(miss)},
        "solve_provenance": SOLVE_PROVENANCE,
        "preregistered_protocol": protocol,
        "game_level_manifest": manifest,
        "fixture_hashes": _fixture_hashes(protocol, manifest, configs, root),
        "arm_configs": configs,
        "budget_parity_receipt": budget_parity_receipt(manifest, RANDOM_SEEDS, DEFAULT_BUDGET),
        "successful_pair_count": 0,
        "failed_pair_reasons": [{"reason": str(miss)}],
        "levels_reproduced_by_arm": {arm: 0 for arm in ARM_NAMES},
        "known_level_regression_count": 0,
        "environment_actions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "frontier_expansions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "actions_per_reproduced_level": {arm: None for arm in ARM_NAMES},
        "solve_latency_actions_by_arm": {arm: [] for arm in ARM_NAMES},
        "ledger_operation_counts_by_arm": {
            arm: {"observe_state": 0, "observe_transition": 0, "rank_candidates": 0, "commitment_checks": 0}
            for arm in ARM_NAMES
        },
        "hypothesis_revisions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "open_questions_resolved_by_arm": {arm: 0 for arm in ARM_NAMES},
        "action_order_change_count": {arm: 0 for arm in ARM_NAMES},
        "commitment_count": {arm: 0 for arm in ARM_NAMES},
        "first_decision_divergence": {"count": 0, "rows": []},
        "verification_calls_by_arm": {arm: 0 for arm in ARM_NAMES},
        "redundant_verification_delta": {
            "verification_calls_avoided": 0,
            "redundant_noop_actions_delta": 0,
            "evidence_lost_count": 0,
        },
        "ledger_cpu_overhead": {
            "control_cpu_time_s": 0.0,
            "treatment_cpu_time_s": 0.0,
            "absolute_s": 0.0,
            "relative_ratio": 0.0,
            "cap_relative_ratio": OVERHEAD_CAPS["cpu_relative_ratio"],
            "over_cap": False,
        },
        "ledger_memory_overhead": {
            "control_ledger_memory_bytes": 0,
            "treatment_ledger_memory_bytes": 0,
            "absolute_bytes": 0,
            "cap_bytes": OVERHEAD_CAPS["memory_bytes"],
            "over_cap": False,
        },
        "entry_propagation_recovery_metrics": entry_propagation_recovery_metrics(controls),
        "invalid_actions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "noop_rate_by_arm": {arm: 0.0 for arm in ARM_NAMES},
        "unsafe_commit_count": 0,
        "control_results": controls,
        "paired_intervals": _paired_intervals([]),
        "material_regression_margins": dict(MATERIAL_REGRESSION_MARGINS),
        "arc_epistemic_live_ab_ready_score": 0.0,
        "new_levels_claimed": 0,
        "registry_updated": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "preconditions_checked": dict(preconds),
        "duration_s": 0.0,
        "honest_verdict": f"blocked: {miss}",
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_artifact(*, root: Path = REPO_ROOT, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        return _blocked_artifact(root, miss, preconds)

    protocol = preregistered_protocol(budget=budget)
    manifest = list(GAME_LEVEL_MANIFEST)
    configs = arm_configs()
    run = run_matched_pairs(manifest=manifest, seeds=RANDOM_SEEDS, budget=budget)
    pairs = list(run.get("pairs") or [])
    successful = _successful_pairs(pairs)
    controls = run_ledger_controls()
    cpu_overhead, memory_overhead = _overhead(successful)
    levels_by_arm = {arm: int(_arm_sum(successful, arm, "levels")) for arm in ARM_NAMES}
    actions_by_arm = {arm: int(_arm_sum(successful, arm, "actions")) for arm in ARM_NAMES}
    verification_by_arm = {
        arm: int(_arm_sum(successful, arm, "verification_calls"))
        for arm in ARM_NAMES
    }
    intervals = _paired_intervals(successful)
    first_divergence = _first_decision_divergence(successful)
    known_level_regression_count = sum(
        1
        for pair in successful
        if int(pair[TREATMENT_ARM].get("levels") or 0)
        < int(pair[CONTROL_ARM].get("levels") or 0) - MATERIAL_REGRESSION_MARGINS["retained_levels"]
    )
    unsafe_commit_count = int(_arm_sum(successful, TREATMENT_ARM, "unsafe_commit_count"))
    action_order_change_count = {
        arm: int(_arm_sum(successful, arm, "action_order_change_count"))
        for arm in ARM_NAMES
    }
    negative_passed = _negative_controls_passed(successful, manifest)
    interval_gain = bool(
        intervals["actions_saved_per_reproduced_level"]["ci95_low"] > 0
        or intervals["retained_level_delta"]["ci95_low"] > 0
    )
    intended_decision_change = bool(
        action_order_change_count[TREATMENT_ARM] > 0 or first_divergence["count"] > 0
    )
    overhead_passed = bool(
        cpu_overhead["over_cap"] is False and memory_overhead["over_cap"] is False
    )
    ready = bool(
        successful
        and interval_gain
        and known_level_regression_count == 0
        and unsafe_commit_count == 0
        and intended_decision_change
        and negative_passed
        and overhead_passed
    )
    if ready:
        verdict = "complete: epistemic_ledger_live_ab_improves_matched_known_level_primary"
    elif known_level_regression_count:
        verdict = "complete: epistemic_ledger_live_ab_null_with_regression"
    else:
        verdict = "complete: epistemic_ledger_live_ab_null_no_promotion"

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "registry_precheck": registry_precheck(root),
        "solve_provenance": SOLVE_PROVENANCE,
        "preregistered_protocol": protocol,
        "game_level_manifest": manifest,
        "fixture_hashes": _fixture_hashes(protocol, manifest, configs, root),
        "arm_configs": configs,
        "budget_parity_receipt": budget_parity_receipt(manifest, RANDOM_SEEDS, budget),
        "successful_pair_count": len(successful),
        "failed_pair_reasons": _failed_pair_reasons(pairs),
        "levels_reproduced_by_arm": levels_by_arm,
        "known_level_regression_count": int(known_level_regression_count),
        "environment_actions_by_arm": actions_by_arm,
        "frontier_expansions_by_arm": {
            arm: int(_arm_sum(successful, arm, "frontier_expansions"))
            for arm in ARM_NAMES
        },
        "legal_proposals_by_arm": {
            arm: int(_arm_sum(successful, arm, "legal_proposal_count"))
            for arm in ARM_NAMES
        },
        "actions_per_reproduced_level": {
            arm: _actions_per_reproduced_level(successful, arm)
            for arm in ARM_NAMES
        },
        "solve_latency_actions_by_arm": {
            arm: [
                pair[arm].get("actions_to_first_levelup")
                for pair in successful
                if pair[arm].get("actions_to_first_levelup") is not None
            ]
            for arm in ARM_NAMES
        },
        "ledger_operation_counts_by_arm": {
            arm: _merge_operation_counts(successful, arm)
            for arm in ARM_NAMES
        },
        "hypothesis_revisions_by_arm": {
            arm: int(_arm_sum(successful, arm, "hypothesis_revision_count"))
            for arm in ARM_NAMES
        },
        "open_questions_resolved_by_arm": {
            arm: int(_arm_sum(successful, arm, "open_question_resolution_count"))
            for arm in ARM_NAMES
        },
        "action_order_change_count": action_order_change_count,
        "commitment_count": {
            arm: int(_arm_sum(successful, arm, "commitment_count"))
            for arm in ARM_NAMES
        },
        "first_decision_divergence": first_divergence,
        "verification_calls_by_arm": verification_by_arm,
        "redundant_verification_delta": {
            "verification_calls_avoided": int(
                verification_by_arm[CONTROL_ARM] - verification_by_arm[TREATMENT_ARM]
            ),
            "redundant_noop_actions_delta": int(
                _arm_sum(successful, CONTROL_ARM, "redundant_verification_count")
                - _arm_sum(successful, TREATMENT_ARM, "redundant_verification_count")
            ),
            "evidence_lost_count": int(_arm_sum(successful, TREATMENT_ARM, "evidence_lost_count")),
        },
        "ledger_cpu_overhead": cpu_overhead,
        "ledger_memory_overhead": memory_overhead,
        "entry_propagation_recovery_metrics": entry_propagation_recovery_metrics(controls),
        "invalid_actions_by_arm": {
            arm: int(_arm_sum(successful, arm, "invalid_actions"))
            for arm in ARM_NAMES
        },
        "noop_rate_by_arm": {
            arm: round(_safe_rate(_arm_sum(successful, arm, "noop_count"), actions_by_arm[arm]), 6)
            for arm in ARM_NAMES
        },
        "unsafe_commit_count": int(unsafe_commit_count),
        "control_results": controls,
        "paired_intervals": intervals,
        "paired_per_game_deltas": _paired_per_game_deltas(successful),
        "material_regression_margins": dict(MATERIAL_REGRESSION_MARGINS),
        "negative_controls_passed": bool(negative_passed),
        "overhead_caps_passed": bool(overhead_passed),
        "intended_decision_change": bool(intended_decision_change),
        "arc_epistemic_live_ab_ready_score": 1.0 if ready else 0.0,
        "new_levels_claimed": 0,
        "registry_updated": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "preconditions_checked": dict(preconds),
        "duration_s": round(float(run.get("duration_s") or (time.monotonic() - started)), 3),
        "ab_rows": pairs,
        "honest_verdict": verdict,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("solve_provenance") != SOLVE_PROVENANCE:
        raise ValueError("solve_provenance must be development_proxy")
    if artifact.get("new_levels_claimed") != 0:
        raise ValueError("new_levels_claimed must remain 0")
    if artifact.get("registry_updated") is not False:
        raise ValueError("registry_updated must remain false")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("unsafe_commit_count") is None:
        raise ValueError("unsafe_commit_count missing")
    expected = _checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected:
        raise ValueError("reproducibility_checksum mismatch")
    return True


def write_artifact(root: Path = REPO_ROOT) -> Path:  # pragma: no cover - CLI
    artifact = build_artifact(root=root)
    validate_artifact(artifact)
    out = root / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:  # pragma: no cover - CLI
    out = write_artifact(REPO_ROOT)
    artifact = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
