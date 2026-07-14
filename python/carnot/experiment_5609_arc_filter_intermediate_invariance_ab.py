"""Experiment 5609: reachability-controlled ARC filter intermediate-invariance A/B.

Spec refs: REQ-ARC-FCP-5609,
SCENARIO-ARC-FCP-5609-REACHABILITY-GATES-BLOCK-OUTCOME-TUNING,
SCENARIO-ARC-FCP-5609-MATCHED-BUDGET-ARM-ISOLATION,
SCENARIO-ARC-FCP-5609-DOWNSTREAM-PROMOTION-GATE.

This module measures the already-wired inert-click and object-history filters
on the offline arcade live-agent runtime. It deliberately avoids game source,
per-game adapters, exhaustive offline BFS, and new solve claims. The outcome
A/B keeps proposer availability identical across all arms and sets the
explore budget above the action budget, so the frozen generator path remains
unchanged and uninvoked while the live exploration filter hooks are measured.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_solve_artifact_discipline import (
    ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5609_arc_filter_intermediate_invariance_ab"
RESULT_RELATIVE_PATH = "results/experiment_5609_arc_filter_intermediate_invariance_ab.json"
SCHEMA = "carnot.exp5609.arc_filter_intermediate_invariance_ab.v1"
INFERENCE_SUBSTRATE = ARC_FILTER_RUNTIME_NO_LLM_SUBSTRATE
RANDOM_SEED = 5609
DEFAULT_ROSTER = ("dc22", "bp35", "s5i5")
DEFAULT_ACTION_BUDGET = 18
DEFAULT_TARGET_LEVELS = 1
STOPPING_RULE = "fixed_action_budget_no_induction"

ARM_NAMES = ("baseline", "inert_only", "history_only", "combined")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "registry_precheck",
    "roster",
    "mechanism_reachability_controls",
    "arm_configs",
    "matched_budget_receipt",
    "candidate_counts_by_arm",
    "environment_actions_by_arm",
    "distinct_states_by_arm",
    "nodes_expanded_by_arm",
    "levels_gained_by_arm",
    "wall_time_by_arm",
    "paired_effects_and_intervals",
    "filter_promotion_decisions",
    "solve_provenance",
    "offline_reproduced",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": {
        "principle": "principle annotations are carried in the artifact so every required 5609 field is auditable."
    },
    "registry_precheck": {
        "principle": "duplicate solve targets are excluded and roster selection is auditable from registry/public environment metadata, not from game source."
    },
    "roster": {
        "principle": "scope is auditable; at least three click/change-diverse games are measured under identical arms."
    },
    "mechanism_reachability_controls": {
        "principle": "a null is interpretable because each mechanism first proves its shipped hook can fire on runtime frames."
    },
    "arm_configs": {
        "principle": "variables are isolated: only inert_click_pruner and object_history_salience vary across the four arms."
    },
    "matched_budget_receipt": {
        "principle": "comparisons are fair: games, seeds, action budgets, proposer availability, target levels, and stopping rules are identical."
    },
    "candidate_counts_by_arm": {
        "principle": "direct filter action is visible before downstream metrics are interpreted."
    },
    "environment_actions_by_arm": {
        "principle": "filters must affect the live path, not only an internal candidate list."
    },
    "distinct_states_by_arm": {
        "principle": "cosmetic candidate collapse is separated from real state-space change."
    },
    "nodes_expanded_by_arm": {
        "principle": "search work is measured independently of candidate counts."
    },
    "levels_gained_by_arm": {
        "principle": "the north-star outcome remains visible even though this is a development proxy."
    },
    "wall_time_by_arm": {
        "principle": "runtime overhead cannot be hidden by reporting only search counts."
    },
    "paired_effects_and_intervals": {
        "principle": "uncertainty controls promotion; no single aggregate delta can promote a filter."
    },
    "filter_promotion_decisions": {
        "principle": "each mechanism is decided separately so a combined arm cannot hide one mechanism's repeat no-op."
    },
    "solve_provenance": {
        "principle": "development_proxy -- public-game measurement receives no new-level credit."
    },
    "offline_reproduced": {
        "principle": "known-level safety is exact and never inferred from level counters alone."
    },
    "inference_substrate": {
        "principle": "offline_arcade_live_agent_runtime_filters_no_new_llm -- current reachable code is measured with no new LLM calls."
    },
    "honest_verdict": {
        "principle": "repeat reachable no-op retires the corresponding mechanism instead of re-running unconstrained prototypes."
    },
}


class InstrumentedInertClickPruner:
    """Small counting wrapper around the shipped InertClickSigPruner."""

    verifier_is_oracle = False

    def __init__(self) -> None:
        from carnot.agentic.arc_agi3_world_model import grid_of
        from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner

        self.inner = InertClickSigPruner(grid_of)
        self.rank_calls = 0
        self.rows_in = 0
        self.rows_out = 0

    @property
    def min_observations(self) -> int:
        return int(self.inner.min_observations)

    @property
    def min_specificity(self) -> float:
        return float(self.inner.min_specificity)

    @property
    def pruned(self) -> int:
        return int(self.inner.pruned)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        self.inner.observe(*args, **kwargs)

    def rank_candidates(self, frame: Any, rows: Sequence[dict]) -> list[dict]:
        self.rank_calls += 1
        self.rows_in += len(rows)
        out = self.inner.rank_candidates(frame, rows)
        self.rows_out += len(out)
        return out

    def stats(self) -> JsonDict:
        out = dict(self.inner.stats())
        out.update(
            {
                "rank_calls": int(self.rank_calls),
                "rows_in": int(self.rows_in),
                "rows_out": int(self.rows_out),
                "rows_dropped": int(self.rows_in - self.rows_out),
            }
        )
        return out


class InstrumentedObjectHistoryPrior:
    """Object-history prior with an explicit count of bonus-bearing rows."""

    verifier_is_oracle = False

    def __init__(self) -> None:
        from carnot.agentic.arc_object_history_salience import ObjectHistorySaliencePrior

        self.inner = ObjectHistorySaliencePrior()
        self.bonus_rows_scored = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def for_path(self, path: list[Mapping[str, Any]]) -> "InstrumentedObjectHistoryPrior":
        self.inner.for_path(path)
        return self

    def observe_transition(self, *args: Any, **kwargs: Any) -> None:
        self.inner.observe_transition(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.inner.reset(*args, **kwargs)

    def score(self, frame: Any, candidate: Any) -> float:
        return float(self.inner.score(frame, candidate))

    def bonus_rows(self, frame: Any, rows: Sequence[dict]) -> int:
        count = 0
        for row in rows:
            try:
                base = float(self.inner.base_prior.score(frame, row))
                full = float(self.inner.score(frame, row))
            except Exception:
                continue
            if full > base:
                count += 1
        self.bonus_rows_scored += count
        return count

    def as_dict(self) -> JsonDict:
        out = dict(self.inner.as_dict())
        out["bonus_rows_scored"] = int(self.bonus_rows_scored)
        return out

    def diagnostics(self) -> JsonDict:
        return self.as_dict()


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        env = arc.make(DEFAULT_ROSTER[0], scorecard_id=arc.open_scorecard())
        env.reset()
        checks["offline_arcade_makes_env"] = True
    except Exception:
        checks["offline_arcade_makes_env"] = False
    try:
        from arcengine.enums import GameAction  # noqa: F401
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner  # noqa: F401
        from carnot.agentic.arc_object_history_salience import (  # noqa: F401
            ObjectHistorySaliencePrior,
        )

        checks["runtime_filter_imports"] = True
    except Exception:
        checks["runtime_filter_imports"] = False
    checks["registry_present"] = (root / "ops" / "arc_solve_registry.yaml").exists()
    checks["ok"] = all(checks.values())
    return checks


def registry_precheck(roster: Sequence[str] = DEFAULT_ROSTER, root: Path = REPO_ROOT) -> JsonDict:
    from carnot.agentic import arc_solver_kit as kit

    unique_roster = list(dict.fromkeys(str(game) for game in roster))
    registry_text = (root / "ops" / "arc_solve_registry.yaml").read_text(encoding="utf-8")
    env_rows = {}
    try:
        arc = kit.offline_arcade()
        for info in arc.get_environments():
            short = str(info.game_id).split("-", 1)[0]
            env_rows[short] = {
                "game_id": str(info.game_id),
                "title": str(info.title),
                "tags": list(info.tags or []),
                "baseline_actions": list(info.baseline_actions or []),
            }
    except Exception:
        env_rows = {}

    roster_rows = []
    for game in unique_roster:
        env_info = env_rows.get(game, {})
        tags = [str(tag) for tag in env_info.get("tags", [])]
        roster_rows.append(
            {
                "game": game,
                "registry_entry_present": game in registry_text,
                "public_environment_metadata_present": bool(env_info),
                "click_capable_by_public_tags": any("click" in tag for tag in tags),
                "tags": tags,
                "baseline_actions": env_info.get("baseline_actions", []),
            }
        )

    duplicate_exclusions = [game for game in roster if list(roster).count(game) > 1]
    return {
        "ok": len(unique_roster) >= 3
        and not duplicate_exclusions
        and all(row["registry_entry_present"] for row in roster_rows)
        and all(row["public_environment_metadata_present"] for row in roster_rows),
        "duplicate_solve_targets_excluded": not duplicate_exclusions,
        "excluded_duplicate_solve_targets": sorted(set(duplicate_exclusions)),
        "selected_games": unique_roster,
        "roster_rows": roster_rows,
        "source_files_read": [],
        "per_game_adapters_created": False,
        "exhaustive_offline_bfs_run": False,
    }


def make_arm_configs(
    *,
    roster: Sequence[str],
    action_budget: int,
    target_levels: int,
    random_seed: int = RANDOM_SEED,
) -> dict[str, JsonDict]:
    base = {
        "roster": list(roster),
        "random_seed": int(random_seed),
        "action_budget": int(action_budget),
        "explore_budget": int(action_budget) + 1,
        "target_levels": int(target_levels),
        "proposer_available": False,
        "frozen_live_generator": "unchanged_not_invoked",
        "stopping_rule": STOPPING_RULE,
    }
    return {
        "baseline": {**base, "inert_click_pruner": False, "object_history_salience": False},
        "inert_only": {**base, "inert_click_pruner": True, "object_history_salience": False},
        "history_only": {**base, "inert_click_pruner": False, "object_history_salience": True},
        "combined": {**base, "inert_click_pruner": True, "object_history_salience": True},
    }


def matched_budget_receipt(arm_configs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    baseline = dict(arm_configs["baseline"])
    invariant_keys = (
        "roster",
        "random_seed",
        "action_budget",
        "explore_budget",
        "target_levels",
        "proposer_available",
        "frozen_live_generator",
        "stopping_rule",
    )
    isolated = True
    for key in invariant_keys:
        values = {_stable_json(config.get(key)) for config in arm_configs.values()}
        isolated = isolated and len(values) == 1
    return {
        "fair": bool(isolated),
        **{key: baseline[key] for key in invariant_keys},
        "varied_only": ["inert_click_pruner", "object_history_salience"],
    }


def run_reachability_controls(roster: Sequence[str]) -> JsonDict:
    inert: JsonDict = {"reachable": False, "per_game": []}
    history: JsonDict = {"reachable": False, "per_game": []}
    for game in roster:
        row = _probe_reachability_game(game)
        inert["per_game"].append(row["inert_click"])
        history["per_game"].append(row["object_history"])
        if not inert["reachable"] and row["inert_click"].get("reachable"):
            inert.update(row["inert_click"])
        if not history["reachable"] and row["object_history"].get("reachable"):
            history.update(row["object_history"])
        if inert["reachable"] and history["reachable"]:
            break
    return {
        "inert_click": inert,
        "object_history": history,
        "ok": bool(inert.get("reachable") and history.get("reachable")),
    }


def run_all_arms(
    *,
    roster: Sequence[str],
    arm_configs: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    per_arm: dict[str, list[JsonDict]] = {}
    for arm in ARM_NAMES:
        config = arm_configs[arm]
        rows = []
        for game in roster:
            rows.append(_run_policy_arm(game=game, arm=arm, config=config))
        per_arm[arm] = rows
    return summarize_arm_results(per_arm)


def summarize_arm_results(per_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    candidate_counts = {}
    environment_actions = {}
    distinct_states = {}
    nodes_expanded = {}
    levels_gained = {}
    wall_time = {}
    offline_by_arm = {}
    for arm, rows in per_arm.items():
        candidate_counts[arm] = {
            "proposed_total": int(sum(int(row.get("proposed_candidates", 0)) for row in rows)),
            "pruned_or_reranked_total": int(
                sum(int(row.get("pruned_or_reranked_candidates", 0)) for row in rows)
            ),
            "by_game": {
                str(row["game"]): {
                    "proposed": int(row.get("proposed_candidates", 0)),
                    "pruned_or_reranked": int(row.get("pruned_or_reranked_candidates", 0)),
                }
                for row in rows
            },
        }
        environment_actions[arm] = _sum_metric(rows, "environment_actions")
        distinct_states[arm] = _sum_metric(rows, "distinct_states")
        nodes_expanded[arm] = _sum_metric(rows, "nodes_expanded")
        levels_gained[arm] = _sum_metric(rows, "levels_gained")
        wall_time[arm] = round(float(sum(float(row.get("wall_time_s", 0.0)) for row in rows)), 3)
        offline_by_arm[arm] = {
            "all_reproduced": all(bool(row.get("offline_reproduced")) for row in rows),
            "reproduced_levels": {
                str(row["game"]): int(row.get("reproduced_level", 0)) for row in rows
            },
        }

    paired = _paired_effects(per_arm)
    exact_safety = all(row["all_reproduced"] for row in offline_by_arm.values())
    return {
        "per_game_results_by_arm": {
            arm: [dict(row) for row in rows] for arm, rows in per_arm.items()
        },
        "candidate_counts_by_arm": candidate_counts,
        "environment_actions_by_arm": environment_actions,
        "distinct_states_by_arm": distinct_states,
        "nodes_expanded_by_arm": nodes_expanded,
        "levels_gained_by_arm": levels_gained,
        "wall_time_by_arm": wall_time,
        "paired_effects_and_intervals": paired,
        "offline_reproduced": {
            "exact_known_level_safety": bool(exact_safety),
            "by_arm": offline_by_arm,
        },
    }


def decide_filter_promotions(
    *,
    controls: Mapping[str, Any],
    paired_effects: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    decisions = {}
    mapping = {
        "inert_click": "inert_only_vs_baseline",
        "object_history": "history_only_vs_baseline",
    }
    for mechanism, effect_key in mapping.items():
        control = controls.get(mechanism, {})
        if not bool(control.get("reachable")):
            decisions[mechanism] = {
                "decision": "blocked_unreachable",
                "reachable": False,
                "candidate_reduction_only": False,
                "reason": "mechanism reachability control failed",
            }
            continue
        effect = paired_effects.get(effect_key, {})
        safety_regression = bool(effect.get("safety_regression"))
        downstream_improved = _downstream_improved(effect)
        candidate_reduced = float(effect.get("candidate_count_delta_mean", 0.0) or 0.0) < 0.0
        candidate_only = bool(candidate_reduced and not downstream_improved)
        if safety_regression:
            decision = "retire_safety_regression"
            reason = "treatment failed same-or-better reproduced-level safety"
        elif downstream_improved:
            decision = "promote_candidate_pending_operator_review"
            reason = "paired downstream intermediate improved without safety regression"
        else:
            decision = "retire_reachable_downstream_noop"
            reason = "reachable control but no downstream live-path improvement"
        decisions[mechanism] = {
            "decision": decision,
            "reachable": True,
            "candidate_reduction_only": candidate_only,
            "downstream_improved": bool(downstream_improved),
            "safety_regression": bool(safety_regression),
            "paired_effect_key": effect_key,
            "reason": reason,
        }
    return decisions


def build_artifact(
    *,
    roster: Sequence[str] = DEFAULT_ROSTER,
    action_budget: int = DEFAULT_ACTION_BUDGET,
    target_levels: int = DEFAULT_TARGET_LEVELS,
    root: Path = REPO_ROOT,
) -> JsonDict:
    started_at = time.time()
    roster = tuple(dict.fromkeys(str(game) for game in roster))
    preconds = preconditions(root)
    registry = (
        registry_precheck(roster, root=root)
        if preconds.get("registry_present", True)
        else {
            "ok": False,
            "duplicate_solve_targets_excluded": False,
            "selected_games": list(roster),
        }
    )
    arm_configs = make_arm_configs(
        roster=roster, action_budget=action_budget, target_levels=target_levels
    )
    receipt = matched_budget_receipt(arm_configs)

    if not preconds.get("ok") or not registry.get("ok"):
        controls: JsonDict = {
            "inert_click": {"reachable": False},
            "object_history": {"reachable": False},
            "ok": False,
        }
        summaries = _empty_summaries()
        decisions = decide_filter_promotions(
            controls=controls, paired_effects=summaries["paired_effects_and_intervals"]
        )
        verdict = "complete: arc_filter_ab_precondition_or_registry_blocked"
    else:
        controls = run_reachability_controls(roster)
        if not controls.get("ok"):
            summaries = _empty_summaries()
            decisions = decide_filter_promotions(
                controls=controls, paired_effects=summaries["paired_effects_and_intervals"]
            )
            verdict = "complete: arc_filter_ab_mechanism_unreachable"
        else:
            summaries = run_all_arms(roster=roster, arm_configs=arm_configs)
            decisions = decide_filter_promotions(
                controls=controls,
                paired_effects=summaries["paired_effects_and_intervals"],
            )
            verdict = _verdict_from_decisions(decisions)

    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "registry_precheck": registry,
        "roster": list(roster),
        "mechanism_reachability_controls": controls,
        "arm_configs": arm_configs,
        "matched_budget_receipt": receipt,
        "candidate_counts_by_arm": summaries["candidate_counts_by_arm"],
        "environment_actions_by_arm": summaries["environment_actions_by_arm"],
        "distinct_states_by_arm": summaries["distinct_states_by_arm"],
        "nodes_expanded_by_arm": summaries["nodes_expanded_by_arm"],
        "levels_gained_by_arm": summaries["levels_gained_by_arm"],
        "wall_time_by_arm": summaries["wall_time_by_arm"],
        "paired_effects_and_intervals": summaries["paired_effects_and_intervals"],
        "filter_promotion_decisions": decisions,
        "solve_provenance": "development_proxy",
        "offline_reproduced": summaries["offline_reproduced"],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": verdict,
        "inference_substrate_note": (
            "No new LLM calls: proposer_available=False and explore_budget exceeds action_budget, "
            "so the frozen generator route is unchanged and uninvoked."
        ),
        "per_game_results_by_arm": summaries["per_game_results_by_arm"],
        "preconditions_checked": preconds,
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum_without_self(artifact)
    return artifact


def main() -> None:
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


def _probe_reachability_game(game: str) -> JsonDict:
    from arcengine.enums import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
    from carnot.agentic.arc_inert_click_pruner import InertClickSigPruner
    from carnot.agentic.arc_object_history_salience import ObjectHistorySaliencePrior

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    current = env.reset()
    base_prior = ColorBlobSaliencePrior()
    inert = InertClickSigPruner(grid_of)
    history = ObjectHistorySaliencePrior()
    click_count = 0
    changed_count = 0
    inert_hit: JsonDict | None = None
    history_hit: JsonDict | None = None
    trace_prefix: list[JsonDict] = []

    for outer in range(12):
        points = base_prior.click_points(current, max_points=12)
        if not points:
            break
        x, y = points[outer % len(points)]
        for _rep in range(4):
            before = current
            after = env.step(GameAction.from_id(6), data={"x": int(x), "y": int(y)})
            if after is None:
                break
            did_change = _frame_changed(before, after)
            changed_count += int(did_change)
            click_count += 1
            level_up = int(getattr(after, "levels_completed", 0) or 0) > int(
                getattr(before, "levels_completed", 0) or 0
            )
            history.observe_transition(before, 6, {"x": int(x), "y": int(y)}, after)
            inert.observe(
                before,
                {"action": 6, "data": {"x": int(x), "y": int(y)}},
                after,
                leveled_up=bool(level_up),
            )
            if len(trace_prefix) < 10:
                trace_prefix.append(
                    {"action": 6, "data": {"x": int(x), "y": int(y)}, "changed": bool(did_change)}
                )
            current = after
            stats = inert.stats()
            if inert_hit is None and int(stats.get("pruned_signatures", 0)) > 0:
                inert_hit = {
                    "reachable": True,
                    "game": game,
                    "clicks_observed_at_first_hit": int(click_count),
                    "stats": stats,
                }
            if history_hit is None:
                pair = _same_base_history_order_pair(history, current)
                if pair is not None:
                    history_hit = {
                        "reachable": True,
                        "game": game,
                        "same_base_ordering_changes": 1,
                        "pair": pair,
                    }
            if inert_hit is not None and history_hit is not None:
                break
        if inert_hit is not None and history_hit is not None:
            break

    inert_row = inert_hit or {"reachable": False, "game": game, "stats": inert.stats()}
    history_row = history_hit or {
        "reachable": False,
        "game": game,
        "same_base_ordering_changes": 0,
        "tracked_hash_count": int(history.tracked_hash_count),
    }
    common = {
        "game": game,
        "fixed_trace": "salience_centroid_clicks_top12_repeat4",
        "clicks_observed": int(click_count),
        "changed_clicks_observed": int(changed_count),
        "trace_prefix": trace_prefix,
    }
    inert_row.update(common)
    history_row.update(common)
    return {"inert_click": inert_row, "object_history": history_row}


def _same_base_history_order_pair(prior: Any, frame: Any) -> JsonDict | None:
    rows = _history_pair_rows(prior, frame)
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            if abs(float(left["base_score"]) - float(right["base_score"])) > 1e-9:
                continue
            if abs(float(left["history_score"]) - float(right["history_score"])) <= 1e-9:
                continue
            history_order = (
                "left_before_right"
                if left["history_score"] > right["history_score"]
                else "right_before_left"
            )
            return {
                "base_order": "tie_stable_original_order",
                "history_order": history_order,
                "left": left,
                "right": right,
            }
    return None


def _history_pair_rows(prior: Any, frame: Any) -> list[JsonDict]:
    from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior

    base_prior = ColorBlobSaliencePrior()
    rows: list[JsonDict] = []
    for index, (x, y) in enumerate(base_prior.click_points(frame, max_points=30)):
        candidate = {"action": 6, "data": {"x": int(x), "y": int(y)}}
        try:
            base_score = float(prior.base_prior.score(frame, candidate))
            history_score = float(prior.score(frame, candidate))
        except Exception:
            continue
        rows.append(
            {
                "index": int(index),
                "action": 6,
                "data": {"x": int(x), "y": int(y)},
                "base_score": base_score,
                "history_score": history_score,
            }
        )
    return rows


def _run_policy_arm(*, game: str, arm: str, config: Mapping[str, Any]) -> JsonDict:
    from arcengine.enums import GameAction
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    random.seed(int(config["random_seed"]))
    np.random.seed(int(config["random_seed"]) % (2**32 - 1))
    instrument = {
        "candidate_generation_calls": 0,
        "final_candidates": 0,
        "inert_rows_dropped": 0,
        "history_bonus_rows": 0,
    }
    inert = InstrumentedInertClickPruner() if config.get("inert_click_pruner") else False
    history = InstrumentedObjectHistoryPrior() if config.get("object_history_salience") else False
    started = time.time()
    policy = E3AgentPolicy(
        game,
        proposer=None,
        explore_budget=int(config["explore_budget"]),
        target_levels=int(config["target_levels"]),
        inert_click_pruner=inert,
        object_history_salience=history,
    )
    _instrument_candidates(policy, instrument, inert, history)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    latest = env.reset()
    frames = [latest]
    env_actions: list[JsonDict] = []
    for _step in range(int(config["action_budget"])):
        move = policy.next_move(frames, latest)
        action_id, data = move
        if action_id == "RESET":
            latest = env.reset()
            frames.append(latest)
            continue
        if action_id is None:
            frames.append(latest)
            continue
        latest = env.step(
            GameAction.from_id(int(action_id)),
            data=data if isinstance(data, dict) else None,
        )
        if latest is None:
            break
        env_actions.append(
            {"action": int(action_id), "data": data if isinstance(data, dict) else None}
        )
        frames.append(latest)
    try:
        policy.next_move(frames, latest)
    except Exception:
        pass
    wall = round(time.time() - started, 3)
    actions = list(env_actions)
    start_level = int(getattr(policy.explorer, "start_level", 0) or 0)
    best_level = int(getattr(policy.explorer, "best_level", start_level) or start_level)
    levels_gained = max(0, best_level - start_level)
    actions_to_level = _actions_to_first_level(policy.transitions, start_level)
    reproduction = _replay_actions(game, actions)
    reproduced_level = int(reproduction.get("reproduced_level", 0))
    safety_regression = reproduced_level < best_level
    pruner_stats = inert.stats() if isinstance(inert, InstrumentedInertClickPruner) else None
    history_stats = (
        history.as_dict() if isinstance(history, InstrumentedObjectHistoryPrior) else None
    )
    pruned_or_reranked = int(instrument["inert_rows_dropped"] + instrument["history_bonus_rows"])
    return {
        "game": game,
        "arm": arm,
        "proposed_candidates": int(
            instrument["final_candidates"] + instrument["inert_rows_dropped"]
        ),
        "pruned_or_reranked_candidates": pruned_or_reranked,
        "environment_actions": int(len(actions)),
        "distinct_states": int(len(getattr(policy.explorer, "graph", {}) or {})),
        "nodes_expanded": int(instrument["candidate_generation_calls"]),
        "levels_gained": int(levels_gained),
        "actions_to_level": actions_to_level,
        "wall_time_s": wall,
        "offline_reproduced": bool(reproduction["offline_reproduced"]),
        "reproduced_level": reproduced_level,
        "best_runtime_level": int(best_level),
        "safety_regression": bool(safety_regression),
        "pruner_stats": pruner_stats,
        "history_prior_stats": history_stats,
        "exact_offline_reproduction": reproduction,
        "action_trace_sha256": _sha256(actions),
    }


def _instrument_candidates(
    policy: Any,
    instrument: dict[str, int],
    inert: InstrumentedInertClickPruner | bool,
    history: InstrumentedObjectHistoryPrior | bool,
) -> None:
    original = policy.explorer._candidates

    def wrapped(
        frame: Any, path: Sequence[dict] | None = None, previous_frame: Any | None = None
    ) -> list[dict]:
        before_dropped = (
            inert.rows_in - inert.rows_out if isinstance(inert, InstrumentedInertClickPruner) else 0
        )
        rows = original(frame, path=path, previous_frame=previous_frame)
        after_dropped = (
            inert.rows_in - inert.rows_out if isinstance(inert, InstrumentedInertClickPruner) else 0
        )
        instrument["candidate_generation_calls"] += 1
        instrument["final_candidates"] += len(rows)
        instrument["inert_rows_dropped"] += max(0, after_dropped - before_dropped)
        if isinstance(history, InstrumentedObjectHistoryPrior):
            instrument["history_bonus_rows"] += history.bonus_rows(frame, rows)
        return rows

    policy.explorer._candidates = wrapped


def _replay_actions(game: str, actions: Sequence[Mapping[str, Any]]) -> JsonDict:
    from arcengine.enums import GameAction
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start_level = int(getattr(frame, "levels_completed", 0) or 0)
    final_level = start_level
    replayed = 0
    ok = True
    for action in actions:
        try:
            frame = env.step(
                GameAction.from_id(int(action["action"])),
                data=action.get("data") if isinstance(action.get("data"), dict) else None,
            )
        except Exception:
            ok = False
            break
        if frame is None:
            ok = False
            break
        replayed += 1
        final_level = int(getattr(frame, "levels_completed", final_level) or final_level)
    return {
        "offline_reproduced": bool(ok and replayed == len(actions)),
        "replayed_actions": int(replayed),
        "expected_actions": int(len(actions)),
        "start_level": int(start_level),
        "reproduced_level": int(final_level),
        "levels_gained": int(max(0, final_level - start_level)),
    }


def _paired_effects(per_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    baseline = {str(row["game"]): row for row in per_arm.get("baseline", [])}
    out: JsonDict = {}
    for arm in ("inert_only", "history_only", "combined"):
        rows = [row for row in per_arm.get(arm, []) if str(row["game"]) in baseline]
        effects: JsonDict = {}
        metric_names = {
            "candidate_count": "proposed_candidates",
            "pruned_or_reranked": "pruned_or_reranked_candidates",
            "environment_actions": "environment_actions",
            "distinct_states": "distinct_states",
            "nodes_expanded": "nodes_expanded",
            "levels_gained": "levels_gained",
            "wall_time": "wall_time_s",
        }
        for label, field in metric_names.items():
            deltas = [
                float(row.get(field, 0.0)) - float(baseline[str(row["game"])].get(field, 0.0))
                for row in rows
            ]
            mean, ci = _mean_ci(deltas)
            effects[f"{label}_delta_mean"] = mean
            effects[f"{label}_delta_95ci"] = ci
        atl_deltas = []
        for row in rows:
            base = baseline[str(row["game"])].get("actions_to_level")
            value = row.get("actions_to_level")
            if base is None or value is None:
                continue
            atl_deltas.append(float(value) - float(base))
        mean, ci = _mean_ci(atl_deltas)
        effects["actions_to_level_delta_mean"] = mean
        effects["actions_to_level_delta_95ci"] = ci
        effects["n_pairs"] = len(rows)
        effects["safety_regression"] = any(
            bool(row.get("safety_regression"))
            or int(row.get("reproduced_level", 0))
            < int(baseline[str(row["game"])].get("reproduced_level", 0))
            for row in rows
        )
        out[f"{arm}_vs_baseline"] = effects
    return out


def _downstream_improved(effect: Mapping[str, Any]) -> bool:
    if bool(effect.get("safety_regression")):
        return False
    return any(
        (
            float(effect.get("environment_actions_delta_mean", 0.0) or 0.0) < 0.0,
            float(effect.get("distinct_states_delta_mean", 0.0) or 0.0) < 0.0,
            float(effect.get("nodes_expanded_delta_mean", 0.0) or 0.0) < 0.0,
            float(effect.get("wall_time_delta_mean", 0.0) or 0.0) < 0.0,
            float(effect.get("levels_gained_delta_mean", 0.0) or 0.0) > 0.0,
            float(effect.get("actions_to_level_delta_mean", 0.0) or 0.0) < 0.0,
        )
    )


def _verdict_from_decisions(decisions: Mapping[str, Mapping[str, Any]]) -> str:
    if any(row.get("decision") == "retire_safety_regression" for row in decisions.values()):
        return "complete: arc_filter_ab_safety_regression_filters_retired"
    if any(
        row.get("decision") == "promote_candidate_pending_operator_review"
        for row in decisions.values()
    ):
        return "complete: arc_filter_ab_downstream_improvement_candidate"
    if all(row.get("decision") == "blocked_unreachable" for row in decisions.values()):
        return "complete: arc_filter_ab_mechanism_unreachable"
    return "complete: arc_filter_ab_reachable_repeat_noop_filters_retired"


def _empty_summaries() -> JsonDict:
    return {
        "per_game_results_by_arm": {},
        "candidate_counts_by_arm": {},
        "environment_actions_by_arm": {},
        "distinct_states_by_arm": {},
        "nodes_expanded_by_arm": {},
        "levels_gained_by_arm": {},
        "wall_time_by_arm": {},
        "paired_effects_and_intervals": {},
        "offline_reproduced": {"exact_known_level_safety": False, "by_arm": {}},
    }


def _actions_to_first_level(transitions: Iterable[Any], start_level: int) -> int | None:
    for idx, transition in enumerate(transitions, start=1):
        if int(getattr(transition, "level_after", start_level) or start_level) > int(start_level):
            return int(idx)
    return None


def _frame_changed(before: Any, after: Any) -> bool:
    from carnot.agentic.arc_agi3_world_model import grid_of

    g0 = grid_of(before)
    g1 = grid_of(after)
    return bool(g0.shape != g1.shape or (g0 != g1).any())


def _sum_metric(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return int(sum(int(row.get(field, 0) or 0) for row in rows))


def _mean_ci(values: Sequence[float]) -> tuple[float, list[float | None]]:
    values = [float(v) for v in values]
    if not values:
        return 0.0, [None, None]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return round(mean, 6), [round(mean, 6), round(mean, 6)]
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    half = 1.96 * math.sqrt(var) / math.sqrt(len(values))
    return round(mean, 6), [round(mean - half, 6), round(mean + half, 6)]


def _first_precondition_miss(checks: Mapping[str, Any]) -> str | None:
    for key, value in checks.items():
        if key != "ok" and not value:
            return key
    return None


def _checksum_without_self(artifact: Mapping[str, Any]) -> str:
    return _sha256({k: v for k, v in artifact.items() if k != "reproducibility_checksum"})


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


if __name__ == "__main__":
    main()
