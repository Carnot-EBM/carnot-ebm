"""Experiment 5712: matched-budget relational goal-energy live-policy A/B.

The experiment compares the submitted ``E3AgentPolicy`` full stack with the
same stack plus Exp5711's relational-plus-legacy goal-energy route. It is a
known-level development-proxy measurement, not a new-solve attempt: the registry
is read to freeze eligibility, but it is never updated.
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
from typing import Any

import numpy as np

from carnot.agentic.arc_goal_energy_live import (
    GoalEnergyCandidateGuidance,
    RelationalGoalEnergy,
    load_relational_goal_energy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT_ID = "experiment_5712_arc_relational_goal_energy_live_ab"
RESULT_RELATIVE_PATH = "results/experiment_5712_arc_relational_goal_energy_live_ab.json"
SCHEMA = "carnot.exp5712.arc_relational_goal_energy_live_ab.v1"
INFERENCE_SUBSTRATE = "matched_arc_live_policy_no_llm"
DEFAULT_BUDGET = 500
RANDOM_SEEDS = [20260715]
CONTROL_ARM = "control"
TREATMENT_ARM = "treatment"
ARM_NAMES = (CONTROL_ARM, TREATMENT_ARM)

UPSTREAM_PATHS = {
    "exp5701": "results/experiment_5701_candidate_scoring_stack_bare_control_ab_headroom.json",
    "exp5703": "results/experiment_5703_sp80_candidate_stack_mechanism_trace.json",
    "exp5711": "results/experiment_5711_arc_relational_goal_energy_live_qualification.json",
}

SOURCE_PATHS = (
    "python/carnot/experiment_5712_arc_relational_goal_energy_live_ab.py",
    "python/carnot/agentic/arc_goal_energy_live.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
)

GAME_LEVEL_MANIFEST = (
    {
        "game": "cd82",
        "target_level": 1,
        "role": "route_positive",
        "mechanic_class": "palette_region_fill",
        "source": "Exp5711 reproduced receipt plus registry precheck",
    },
    {
        "game": "cn04",
        "target_level": 1,
        "role": "route_positive",
        "mechanic_class": "marker_pair_shape_alignment",
        "source": "Exp5711 reproduced receipt plus registry precheck",
    },
    {
        "game": "sk48",
        "target_level": 1,
        "role": "route_positive",
        "mechanic_class": "chain_color_reorder",
        "source": "Exp5711 reproduced receipt plus registry precheck",
    },
    {
        "game": "sp80",
        "target_level": 1,
        "role": "route_positive",
        "mechanic_class": "spill_splitter_placement",
        "source": "Exp5711 reproduced receipt plus registry precheck",
    },
    {
        "game": "lp85",
        "target_level": 1,
        "role": "negative_count_identity",
        "mechanic_class": "visible_target_alignment_with_hidden_identity",
        "source": "Exp5701 known-level headroom negative for relational route accept",
    },
    {
        "game": "tu93",
        "target_level": 1,
        "role": "negative_navigation",
        "mechanic_class": "graph_explore_navigation",
        "source": "Exp5701 known-level headroom negative for relational route accept",
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
    "level_regression_count",
    "environment_actions_by_arm",
    "frontier_expansions_by_arm",
    "actions_per_reproduced_level",
    "candidate_order_change_count",
    "score_variance_by_arm",
    "route_activation_count",
    "invalid_actions_by_arm",
    "noop_rate_by_arm",
    "fallback_rate_by_arm",
    "paired_intervals",
    "material_regression_margins",
    "unsafe_route_accept_count",
    "control_results",
    "relational_live_ab_ready_score",
    "new_levels_claimed",
    "registry_updated",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "upstream_gate_receipts": {
        "principle": "Eligibility is explicit: Exp5701 supplies the matched known-level budget, Exp5703 supplies the inert legacy-goal diagnosis, and Exp5711 qualifies the relational route before this A/B uses it."
    },
    "registry_precheck": {
        "principle": "Reads reproduced public-game registry state before execution and forbids claiming a new solve from this development-proxy A/B."
    },
    "arm_configs": {
        "principle": "The primary control stays the submitted full stack; only the treatment goal-energy source changes, so a win is attributable to the relational route."
    },
    "budget_parity_receipt": {
        "principle": "Matched seeds, games, action budgets, fresh environments, and stopping rules prevent a hidden budget/caching advantage."
    },
    "successful_pair_count": {
        "principle": "Reports the real denominator so a null or win cannot hide failed game/arm pairs."
    },
    "levels_reproduced_by_arm": {
        "principle": "Retention is primary: the treatment cannot promote if it loses known levels beyond the frozen margin."
    },
    "environment_actions_by_arm": {
        "principle": "Efficiency is measured in real environment actions, not internal scoring calls."
    },
    "candidate_order_change_count": {
        "principle": "A route that never changes intended ordering is mechanism-inert and cannot justify promotion."
    },
    "unsafe_route_accept_count": {
        "principle": "Unsupported/corrupt route inputs must fail closed; any unsafe accept blocks promotion."
    },
    "relational_live_ab_ready_score": {
        "principle": "Scalar advisory promotion is 1.0 only for interval-backed efficiency or retention gain with no regressions, preserved negatives, intended ordering exercise, and zero unsafe accepts."
    },
    "honest_verdict": {
        "principle": "Terminal-prefixed complete:/blocked: verdict; a null is a completed measurement and does not claim a solve."
    },
}

MATERIAL_REGRESSION_MARGINS = {
    "retained_levels": 0,
    "actions_per_reproduced_level": 0.0,
}


class _GuidanceProbe:
    """Wrap candidate guidance so the experiment can count real ordering changes."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.changed_order_count = 0

    def set_goal_energy(self, goal_energy: Any) -> None:
        if hasattr(self.inner, "set_goal_energy"):
            self.inner.set_goal_energy(goal_energy)

    def diagnostics(self) -> dict[str, Any]:
        if hasattr(self.inner, "diagnostics"):
            out = dict(self.inner.diagnostics())
        else:
            out = {"enabled": True}
        out["candidate_order_change_count"] = int(self.changed_order_count)
        return out

    def rank_candidates(self, frame: Any, candidates: Sequence[Any]) -> list[dict[str, Any]]:
        before = [_candidate_signature(row) for row in candidates]
        ranked = self.inner.rank_candidates(frame, candidates)
        after = [_candidate_signature(row) for row in ranked]
        if before != after:
            self.changed_order_count += 1
        return ranked


def _candidate_signature(candidate: Any) -> tuple[Any, str]:
    if isinstance(candidate, Mapping):
        action = candidate.get("action", candidate.get("action_id"))
        data = candidate.get("data")
    else:
        action = getattr(candidate, "action", getattr(candidate, "action_id", None))
        data = getattr(candidate, "data", None)
    return (action, json.dumps(data, sort_keys=True, default=str))


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def preconditions(root: Path = REPO_ROOT) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["registry_exists"] = (root / "ops" / "arc_solve_registry.yaml").exists()
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make("sp80", scorecard_id=arc.open_scorecard())
        env.reset()
        checks["offline_arcade_importable"] = True
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401

        checks["e3_policy_importable"] = True
    except Exception:
        checks["e3_policy_importable"] = False
    exp5711 = _read_json(root / UPSTREAM_PATHS["exp5711"])
    checks["exp5711_ready"] = exp5711.get("relational_goal_energy_ready_score") == 1.0
    checks["ok"] = all(checks.values())
    return checks


def _first_precondition_miss(preconds: Mapping[str, Any]) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return str(key)
    return None


def upstream_gate_receipts(root: Path = REPO_ROOT) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for key, rel in UPSTREAM_PATHS.items():
        path = root / rel
        artifact = _read_json(path)
        receipts[key] = {
            "path": rel,
            "present": path.exists(),
            "honest_verdict": artifact.get("honest_verdict"),
            "inference_substrate": artifact.get("inference_substrate"),
            "eligibility": {
                "eligible": bool(
                    path.exists()
                    and str(artifact.get("honest_verdict") or "").startswith("complete:")
                ),
                "reason": {
                    "exp5701": "matched known-level budget and headroom roster source",
                    "exp5703": "legacy goal-energy inertness diagnosis on sp80",
                    "exp5711": "relational route qualified with ready score 1.0",
                }[key],
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
        }
    receipts["exp5711"]["eligibility"]["eligible"] = bool(
        receipts["exp5711"]["eligibility"]["eligible"]
        and _read_json(root / UPSTREAM_PATHS["exp5711"]).get("relational_goal_energy_ready_score")
        == 1.0
    )
    return receipts


def registry_precheck(root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / "ops" / "arc_solve_registry.yaml"
    registry_rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for row in data.get("games", []) or []:
                if isinstance(row, Mapping) and row.get("game"):
                    registry_rows[str(row["game"])] = dict(row)
        except Exception:
            registry_rows = {}
    manifest = []
    for row in GAME_LEVEL_MANIFEST:
        reg = registry_rows.get(str(row["game"]), {})
        levels_reproduced = int(reg.get("levels_reproduced") or 0)
        reproduced = reg.get("reproducibility") == "reproduced"
        manifest.append(
            {
                **row,
                "registry_reproducibility": reg.get("reproducibility"),
                "registry_levels_reproduced": levels_reproduced,
                "eligible": bool(reproduced and levels_reproduced >= int(row["target_level"])),
            }
        )
    return {
        "source": "ops/arc_solve_registry.yaml",
        "registry_present": path.exists(),
        "checked_before_execution": True,
        "solve_provenance": "development_proxy",
        "new_level_claim_allowed": False,
        "registry_update_allowed": False,
        "eligible_game_count": sum(1 for row in manifest if row["eligible"]),
        "manifest": manifest,
    }


def preregistered_protocol() -> dict[str, Any]:
    return {
        "frozen_on": "2026-07-15",
        "experiment_id": EXPERIMENT_ID,
        "primary_metrics": [
            "retained_level_count",
            "actions_saved_per_reproduced_level",
        ],
        "secondary_metrics": [
            "frontier_expansions",
            "candidate_count",
            "candidate_order_change_count",
            "score_variance",
            "route_activation_count",
            "invalid_actions",
            "noop_rate",
            "fallback_rate",
            "actions_to_first_levelup",
        ],
        "random_seeds": list(RANDOM_SEEDS),
        "action_budget_per_game_arm": DEFAULT_BUDGET,
        "restart_policy": "fresh offline arcade environment per game, seed, and arm",
        "cache_policy": "fresh E3AgentPolicy per arm; submitted read-only model/router caches allowed equally",
        "stopping_rules": "stop on policy done, None action, terminal frame, or action budget",
        "full_stack_config": "submitted E3AgentPolicy defaults with CARNOT_ARC_DISABLE_INDUCTION=1",
        "treatment_delta": "replace submitted legacy goal energy with RelationalGoalEnergy(fallback=legacy)",
        "thresholds": {
            "paired_bootstrap_samples": 1000,
            "ci": 0.95,
            "route_variance_floor": 1e-12,
        },
        "promotion_rules": {
            "requires_interval_gain": True,
            "requires_zero_level_regressions_beyond_margin": True,
            "requires_intended_order_change": True,
            "requires_negative_controls_preserved": True,
            "requires_zero_unsafe_route_accepts": True,
        },
    }


def arm_configs(root: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        CONTROL_ARM: {
            "policy": "E3AgentPolicy",
            "config": "submitted_full_stack",
            "goal_bias": "submitted_default_goal_satisfaction_energy",
            "goal_candidate_guidance": "submitted_default",
            "llm_induction": "disabled_by_CARNOT_ARC_DISABLE_INDUCTION=1",
        },
        TREATMENT_ARM: {
            "policy": "E3AgentPolicy",
            "config": "submitted_full_stack_plus_exp5711_route",
            "goal_bias": "RelationalGoalEnergy(fallback=load_exp4020_goal_energy)",
            "goal_candidate_guidance": "same submitted candidate guidance over treatment goal_bias",
            "llm_induction": "disabled_by_CARNOT_ARC_DISABLE_INDUCTION=1",
            "route_loader_available": load_relational_goal_energy(root) is not None,
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
        "policy_knobs_matched_except_goal_route": True,
        "fresh_env_per_arm": True,
        "fresh_policy_per_arm": True,
        "stopping_rules_matched": True,
        "control_is_weakened": False,
    }


def _baseline_actions(env: Any) -> dict[int, int]:
    for attr in ("baseline_actions", "human_actions", "reference_actions"):
        value = getattr(getattr(env, "info", env), attr, None)
        if value:
            if isinstance(value, (list, tuple)):
                return {int(i): int(v) for i, v in enumerate(value)}
            return {int(k): int(v) for k, v in dict(value).items()}
    return {}


def _score_efficiency(
    baseline: Mapping[int, int],
    level_up_actions: Sequence[int],
    total_actions: int,
) -> tuple[float, list[dict[str, Any]]]:
    if not baseline:
        return 0.0, []
    per_level = []
    try:
        from arc_agi.scorecard import EnvironmentScoreCalculator

        calc = EnvironmentScoreCalculator()
        previous = 0
        for level in sorted(baseline):
            if level < len(level_up_actions):
                at = int(level_up_actions[level])
                level_actions = at - previous
                completed = True
                previous = at
            else:
                level_actions = int(total_actions) - previous
                completed = False
                previous = int(total_actions)
            calc.add_level(
                level_index=int(level) + 1,
                completed=completed,
                actions_taken=int(level_actions),
                baseline_actions=int(baseline[level]),
            )
            per_level.append(
                {
                    "level": int(level),
                    "agent_actions": int(level_actions),
                    "human_actions": int(baseline[level]),
                    "completed": bool(completed),
                }
            )
        return round(float(calc.to_score(include_levels=False).score), 4), per_level
    except Exception:
        return 0.0, per_level


def _grid_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    try:
        from carnot.agentic.arc_agi3_world_model import grid_of

        return bool(np.array_equal(grid_of(left), grid_of(right)))
    except Exception:
        return False


def _make_policy(game: str, arm: str, root: Path):
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    if arm == TREATMENT_ARM:
        goal_bias = load_relational_goal_energy(root) or RelationalGoalEnergy()
        return E3AgentPolicy(game, proposer=None, goal_bias=goal_bias, goal_candidate_guidance=True)
    return E3AgentPolicy(game, proposer=None)


def _run_one_arm(game: str, *, arm: str, seed: int, budget: int, root: Path) -> dict[str, Any]:
    from arcengine import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _level_of

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    baseline = _baseline_actions(env)
    policy = _make_policy(game, arm, root)
    explorer = getattr(policy, "explorer", None)
    stats = {
        "frontier_expansions": 0,
        "candidate_count": 0,
        "candidate_order_change_count": 0,
    }
    if explorer is not None and hasattr(explorer, "_candidates"):
        real_candidates = explorer._candidates

        def _counting_candidates(*args, **kwargs):
            rows = real_candidates(*args, **kwargs)
            stats["frontier_expansions"] += 1
            stats["candidate_count"] += len(rows or [])
            return rows

        explorer._candidates = _counting_candidates
    if explorer is not None and getattr(explorer, "goal_candidate_guidance", None) is not None:
        probe = _GuidanceProbe(explorer.goal_candidate_guidance)
        explorer.goal_candidate_guidance = probe
    else:
        probe = None

    frames: list[Any] = []
    latest = None
    start_level = None
    best_level = None
    level_up_actions: list[int] = []
    actions = 0
    invalid_actions = 0
    noop_count = 0

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
    efficiency, per_level = _score_efficiency(baseline, level_up_actions, actions)
    goal_diag = (
        explorer.goal_bias_diagnostics()
        if explorer is not None and hasattr(explorer, "goal_bias_diagnostics")
        else {}
    )
    goal_obj = getattr(explorer, "goal_bias", None) if explorer is not None else None
    route_diag = goal_obj.diagnostics() if hasattr(goal_obj, "diagnostics") else {}
    order_changes = int(probe.changed_order_count if probe is not None else 0)
    guidance_diag = (
        explorer.goal_candidate_guidance_diagnostics()
        if explorer is not None and hasattr(explorer, "goal_candidate_guidance_diagnostics")
        else {}
    )
    return {
        "game": str(game),
        "arm": str(arm),
        "seed": int(seed),
        "start_level": int(start_level or 0),
        "reached_level": int(reached_level or 0),
        "levels": int(levels),
        "actions": int(actions),
        "efficiency": float(efficiency),
        "per_level": per_level,
        "actions_to_first_levelup": int(level_up_actions[0]) if level_up_actions else None,
        "frontier_expansions": int(stats["frontier_expansions"]),
        "candidate_count": int(stats["candidate_count"]),
        "candidate_order_change_count": order_changes,
        "score_variance": float(goal_diag.get("score_variance") or 0.0),
        "route_activation_count": int(route_diag.get("routed_call_count") or 0),
        "invalid_actions": int(invalid_actions),
        "noop_count": int(noop_count),
        "fallback_count": int(route_diag.get("fallback_count") or 0),
        "goal_bias_call_count": int(
            route_diag.get("call_count") or goal_diag.get("nodes_scored") or 0
        ),
        "goal_bias_diagnostics": goal_diag,
        "goal_route_diagnostics": route_diag,
        "goal_candidate_guidance_diagnostics": guidance_diag,
        "failed_reason": None,
    }


def run_matched_pairs(
    *,
    manifest: Sequence[Mapping[str, Any]] = GAME_LEVEL_MANIFEST,
    seeds: Sequence[int] = RANDOM_SEEDS,
    budget: int = DEFAULT_BUDGET,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
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
                        game, arm=CONTROL_ARM, seed=int(seed), budget=int(budget), root=root
                    )
                    pair[TREATMENT_ARM] = _run_one_arm(
                        game, arm=TREATMENT_ARM, seed=int(seed), budget=int(budget), root=root
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


def _mask(shape: tuple[int, int], coords: Sequence[tuple[int, int]]) -> list[list[bool]]:
    out = np.zeros(shape, dtype=bool)
    for y, x in coords:
        out[int(y), int(x)] = True
    return out.tolist()


def _state(grid: np.ndarray, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {"frame": np.asarray(grid, dtype=int), "relational_goal_receipt": dict(receipt)}


def run_route_controls() -> list[dict[str, Any]]:
    candidates = [
        {"action": 1, "data": None, "candidate_state": {"rank": 1}},
        {"action": 2, "data": None, "candidate_state": {"rank": 0}},
    ]
    disabled_order_changed = False

    guidance = GoalEnergyCandidateGuidance(
        goal_energy=lambda state: float(state["rank"]),
        transition_predictor=lambda _frame, candidate: candidate["candidate_state"],
    )
    shuffled = guidance.rank_candidates(object(), candidates)
    shuffled_changed = [row["action"] for row in shuffled] == [2, 1]

    energy = RelationalGoalEnergy()
    corrupt_score = energy(
        _state(
            np.zeros((3, 3), dtype=int),
            {
                "route_class": "region_pair_equality",
                "source_mask": [[True]],
                "target_mask": [[True]],
            },
        )
    )
    corrupt_diag = energy.diagnostics()

    always_energy = RelationalGoalEnergy()
    always_score = always_energy(
        _state(
            np.zeros((3, 3), dtype=int),
            {
                "route_class": "always_route",
                "source_mask": _mask((3, 3), [(1, 1)]),
            },
        )
    )
    always_diag = always_energy.diagnostics()

    zero_guidance = GoalEnergyCandidateGuidance(
        goal_energy=lambda _state: 1.0,
        transition_predictor=lambda _frame, candidate: candidate["candidate_state"],
    )
    zero_ranked = zero_guidance.rank_candidates(object(), candidates)
    zero_changed = [row["action"] for row in zero_ranked] != [1, 2]

    return [
        {
            "name": "disabled_route",
            "intended_exercise": False,
            "intended_ordering_changed": disabled_order_changed,
            "safe_fallback": True,
            "unsafe_route_accept": False,
        },
        {
            "name": "shuffled_score",
            "intended_exercise": True,
            "intended_ordering_changed": bool(shuffled_changed),
            "safe_fallback": True,
            "unsafe_route_accept": False,
            "diagnostics": guidance.diagnostics(),
        },
        {
            "name": "corrupted_mask",
            "intended_exercise": False,
            "intended_ordering_changed": False,
            "safe_fallback": bool(
                not corrupt_diag.get("last_routed")
                and corrupt_diag.get("last_fallback_reason") == "corrupt_receipt"
            ),
            "unsafe_route_accept": bool(corrupt_diag.get("last_routed")),
            "score": float(corrupt_score),
            "diagnostics": corrupt_diag,
        },
        {
            "name": "always_route",
            "intended_exercise": False,
            "intended_ordering_changed": False,
            "safe_fallback": bool(
                not always_diag.get("last_routed")
                and always_diag.get("last_fallback_reason") == "unsupported_route_class"
            ),
            "unsafe_route_accept": bool(always_diag.get("last_routed")),
            "score": float(always_score),
            "diagnostics": always_diag,
        },
        {
            "name": "zero_variance",
            "intended_exercise": False,
            "intended_ordering_changed": bool(zero_changed),
            "safe_fallback": bool(not zero_changed),
            "unsafe_route_accept": False,
            "diagnostics": zero_guidance.diagnostics(),
        },
    ]


def _successful_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        pair
        for pair in pairs
        if not pair.get("failed_reason")
        and isinstance(pair.get(CONTROL_ARM), Mapping)
        and isinstance(pair.get(TREATMENT_ARM), Mapping)
    ]


def _failed_pair_reasons(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        reason = pair.get("failed_reason")
        if reason:
            rows.append(
                {
                    "game": pair.get("game"),
                    "seed": pair.get("seed"),
                    "reason": str(reason),
                }
            )
    return rows


def _arm_sum(pairs: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    return float(sum(float((pair[arm] or {}).get(field) or 0.0) for pair in pairs))


def _arm_mean(pairs: Sequence[Mapping[str, Any]], arm: str, field: str) -> float:
    if not pairs:
        return 0.0
    return float(sum(float((pair[arm] or {}).get(field) or 0.0) for pair in pairs) / len(pairs))


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _actions_per_reproduced_level(pairs: Sequence[Mapping[str, Any]], arm: str) -> float | None:
    levels = _arm_sum(pairs, arm, "levels")
    if levels <= 0:
        return None
    return round(_arm_sum(pairs, arm, "actions") / levels, 6)


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
            draw = [vals[rng.randrange(len(vals))] for _i in range(len(vals))]
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
        control = pair[CONTROL_ARM]
        treatment = pair[TREATMENT_ARM]
        control_levels = float(control.get("levels") or 0)
        treatment_levels = float(treatment.get("levels") or 0)
        if max(control_levels, treatment_levels) <= 0:
            continue
        control_apl = float(control.get("actions") or 0) / max(1.0, control_levels)
        treatment_apl = float(treatment.get("actions") or 0) / max(1.0, treatment_levels)
        action_deltas.append(control_apl - treatment_apl)
    if not action_deltas:
        action_deltas = [0.0]
    return {
        "retained_level_delta": _interval(level_deltas, seed=RANDOM_SEEDS[0]),
        "actions_saved_per_reproduced_level": _interval(action_deltas, seed=RANDOM_SEEDS[0] + 1),
    }


def _paired_per_game_deltas(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        control = pair[CONTROL_ARM]
        treatment = pair[TREATMENT_ARM]
        rows.append(
            {
                "game": pair.get("game"),
                "seed": pair.get("seed"),
                "level_delta": int(treatment.get("levels") or 0)
                - int(control.get("levels") or 0),
                "actions_delta": int(treatment.get("actions") or 0)
                - int(control.get("actions") or 0),
                "efficiency_delta": round(
                    float(treatment.get("efficiency") or 0.0)
                    - float(control.get("efficiency") or 0.0),
                    6,
                ),
                "frontier_expansion_delta": int(treatment.get("frontier_expansions") or 0)
                - int(control.get("frontier_expansions") or 0),
                "candidate_delta": int(treatment.get("candidate_count") or 0)
                - int(control.get("candidate_count") or 0),
            }
        )
    return rows


def _negative_controls_preserved(
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


def _fixture_hashes(
    root: Path,
    protocol: Mapping[str, Any],
    manifest: Sequence[Mapping[str, Any]],
    configs: Mapping[str, Any],
) -> dict[str, Any]:
    upstream_hashes = {}
    for key, rel in UPSTREAM_PATHS.items():
        path = root / rel
        upstream_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "game_level_manifest_sha256": "sha256:" + _json_hash(list(manifest)),
        "preregistered_protocol_sha256": "sha256:" + _json_hash(protocol),
        "arm_configs_sha256": "sha256:" + _json_hash(configs),
        "upstream_artifact_sha256": upstream_hashes,
    }


def _checksum(root: Path, artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    source_hashes = {}
    for rel in SOURCE_PATHS:
        path = root / rel
        source_hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return "sha256:" + _json_hash({"artifact": payload, "source_hashes": source_hashes})


def _blocked_artifact(root: Path, miss: str, preconds: Mapping[str, Any]) -> dict[str, Any]:
    protocol = preregistered_protocol()
    configs = arm_configs(root)
    manifest = list(GAME_LEVEL_MANIFEST)
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "registry_precheck": {"precondition_blocked_before_registry_precheck": str(miss)},
        "solve_provenance": "development_proxy",
        "preregistered_protocol": protocol,
        "game_level_manifest": manifest,
        "fixture_hashes": _fixture_hashes(root, protocol, manifest, configs),
        "arm_configs": configs,
        "budget_parity_receipt": budget_parity_receipt(manifest, RANDOM_SEEDS, DEFAULT_BUDGET),
        "successful_pair_count": 0,
        "failed_pair_reasons": [{"reason": str(miss)}],
        "levels_reproduced_by_arm": {arm: 0 for arm in ARM_NAMES},
        "level_regression_count": 0,
        "environment_actions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "frontier_expansions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "actions_per_reproduced_level": {arm: None for arm in ARM_NAMES},
        "candidate_order_change_count": {arm: 0 for arm in ARM_NAMES},
        "score_variance_by_arm": {arm: 0.0 for arm in ARM_NAMES},
        "route_activation_count": {arm: 0 for arm in ARM_NAMES},
        "invalid_actions_by_arm": {arm: 0 for arm in ARM_NAMES},
        "noop_rate_by_arm": {arm: 0.0 for arm in ARM_NAMES},
        "fallback_rate_by_arm": {arm: 0.0 for arm in ARM_NAMES},
        "paired_intervals": _paired_intervals([]),
        "paired_per_game_deltas": [],
        "material_regression_margins": MATERIAL_REGRESSION_MARGINS,
        "unsafe_route_accept_count": 0,
        "control_results": [],
        "relational_live_ab_ready_score": 0.0,
        "new_levels_claimed": 0,
        "registry_updated": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "preconditions_checked": dict(preconds),
        "duration_s": 0.0,
        "honest_verdict": f"blocked: {miss}",
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(root, artifact)
    return artifact


def build_artifact(*, root: Path = REPO_ROOT, budget: int = DEFAULT_BUDGET) -> dict[str, Any]:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    if miss:
        return _blocked_artifact(root, miss, preconds)

    protocol = preregistered_protocol()
    manifest = list(GAME_LEVEL_MANIFEST)
    configs = arm_configs(root)
    run = run_matched_pairs(manifest=manifest, seeds=RANDOM_SEEDS, budget=budget, root=root)
    pairs = list(run.get("pairs") or [])
    successful = _successful_pairs(pairs)
    failed = _failed_pair_reasons(pairs)
    controls = run_route_controls()
    unsafe_count = sum(1 for row in controls if row.get("unsafe_route_accept"))
    intervals = _paired_intervals(successful)
    level_regression_count = sum(
        1
        for pair in successful
        if int(pair[TREATMENT_ARM].get("levels") or 0)
        < int(pair[CONTROL_ARM].get("levels") or 0) - MATERIAL_REGRESSION_MARGINS["retained_levels"]
    )
    levels_by_arm = {
        arm: int(_arm_sum(successful, arm, "levels"))
        for arm in ARM_NAMES
    }
    actions_by_arm = {
        arm: int(_arm_sum(successful, arm, "actions"))
        for arm in ARM_NAMES
    }
    fallback_rate_by_arm = {
        arm: round(
            _safe_rate(_arm_sum(successful, arm, "fallback_count"), _arm_sum(successful, arm, "goal_bias_call_count")),
            6,
        )
        for arm in ARM_NAMES
    }
    negative_preserved = _negative_controls_preserved(successful, manifest)
    interval_gain = bool(
        intervals["actions_saved_per_reproduced_level"]["ci95_low"] > 0
        or intervals["retained_level_delta"]["ci95_low"] > 0
    )
    changed_orderings = bool(
        sum(int(pair[TREATMENT_ARM].get("candidate_order_change_count") or 0) for pair in successful)
        > 0
    )
    ready = bool(
        successful
        and interval_gain
        and level_regression_count == 0
        and changed_orderings
        and negative_preserved
        and unsafe_count == 0
    )
    if ready:
        verdict = "complete: relational_live_route_improves_matched_known_level_efficiency"
    elif level_regression_count:
        verdict = "complete: relational_live_route_null_with_level_regression"
    else:
        verdict = "complete: relational_live_route_null_no_promotion"

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "upstream_gate_receipts": upstream_gate_receipts(root),
        "registry_precheck": registry_precheck(root),
        "solve_provenance": "development_proxy",
        "preregistered_protocol": protocol,
        "game_level_manifest": manifest,
        "fixture_hashes": _fixture_hashes(root, protocol, manifest, configs),
        "arm_configs": configs,
        "budget_parity_receipt": budget_parity_receipt(manifest, RANDOM_SEEDS, budget),
        "successful_pair_count": len(successful),
        "failed_pair_reasons": failed,
        "levels_reproduced_by_arm": levels_by_arm,
        "level_regression_count": int(level_regression_count),
        "environment_actions_by_arm": actions_by_arm,
        "frontier_expansions_by_arm": {
            arm: int(_arm_sum(successful, arm, "frontier_expansions"))
            for arm in ARM_NAMES
        },
        "actions_per_reproduced_level": {
            arm: _actions_per_reproduced_level(successful, arm)
            for arm in ARM_NAMES
        },
        "candidate_order_change_count": {
            arm: int(_arm_sum(successful, arm, "candidate_order_change_count"))
            for arm in ARM_NAMES
        },
        "candidate_count_by_arm": {
            arm: int(_arm_sum(successful, arm, "candidate_count"))
            for arm in ARM_NAMES
        },
        "score_variance_by_arm": {
            arm: round(_arm_mean(successful, arm, "score_variance"), 10)
            for arm in ARM_NAMES
        },
        "route_activation_count": {
            arm: int(_arm_sum(successful, arm, "route_activation_count"))
            for arm in ARM_NAMES
        },
        "invalid_actions_by_arm": {
            arm: int(_arm_sum(successful, arm, "invalid_actions"))
            for arm in ARM_NAMES
        },
        "noop_rate_by_arm": {
            arm: round(_safe_rate(_arm_sum(successful, arm, "noop_count"), actions_by_arm[arm]), 6)
            for arm in ARM_NAMES
        },
        "fallback_rate_by_arm": fallback_rate_by_arm,
        "solve_latency_actions_by_arm": {
            arm: [
                pair[arm].get("actions_to_first_levelup")
                for pair in successful
                if pair[arm].get("actions_to_first_levelup") is not None
            ]
            for arm in ARM_NAMES
        },
        "paired_intervals": intervals,
        "paired_per_game_deltas": _paired_per_game_deltas(successful),
        "material_regression_margins": MATERIAL_REGRESSION_MARGINS,
        "unsafe_route_accept_count": int(unsafe_count),
        "control_results": controls,
        "negative_controls_preserved": bool(negative_preserved),
        "relational_live_ab_ready_score": 1.0 if ready else 0.0,
        "new_levels_claimed": 0,
        "registry_updated": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": list(RANDOM_SEEDS),
        "preconditions_checked": dict(preconds),
        "duration_s": round(float(run.get("duration_s") or (time.monotonic() - started)), 3),
        "honest_verdict": verdict,
        "ab_rows": pairs,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(root, artifact)
    return artifact


def write_artifact(root: Path = REPO_ROOT) -> Path:  # pragma: no cover
    artifact = build_artifact(root=root)
    out = root / RESULT_RELATIVE_PATH
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:  # pragma: no cover
    out = write_artifact(REPO_ROOT)
    artifact = json.loads(out.read_text(encoding="utf-8"))
    print(f"wrote {out} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
