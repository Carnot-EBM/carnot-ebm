"""Exp 4021: heuristic search over a verifier-certified ARC-AGI-3 world model.

Spec refs: REQ-PHASE4-030, SCENARIO-PHASE4-030.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
RESULT_NAME = "experiment_4021_heuristic_search_over_verified_wm.json"
RANDOM_SEED = 4021
TARGET_GAME = "r11l"
TARGET_LEVEL = 4
PRIOR_WALL_LEVEL = 3
MAX_EXPANSIONS = 50000
INFERENCE_SUBSTRATE = (
    "offline_arc_agi3_verified_env_copy_simulator_exp4020_goal_predicate_"
    "coded_heuristic_mpc_replanning"
)

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_goal_predicate_separation import compile_goal_predicate  # noqa: E402
from carnot.agentic.arc_heuristic_search_over_verified_wm import (  # noqa: E402
    DEFAULT_HEURISTIC_NAME,
    SearchResult,
    artifact_schema_errors,
    best_first_search,
    build_search_artifact,
    coded_goal_distance_heuristic,
    write_artifact,
)
from experiment_3964_r11l_incremental_l2 import _levels_completed  # noqa: E402
from experiment_3992_incremental_levels_verifier_validated import (  # noqa: E402
    _build_safe_path_moves,
    _execute_moves,
)
from experiment_4014_break_level_wall_explore_first import (  # noqa: E402
    _advance_r11l_to_l3,
    _load_offline_arcade,
    _select_game_id,
)


@dataclass(frozen=True)
class R11LRunOutcome:
    """Result of the live r11l L4 wall attempt before JSON normalization."""

    search: SearchResult
    real_env_confirmed: bool
    levels_completed_after: int
    executed_actions: int
    per_step_replans: int
    diagnosis: str


def _sprite_center(sprite: Any) -> tuple[int, int]:
    return (
        int(sprite.y) + int(getattr(sprite, "height", 1)) // 2,
        int(sprite.x) + int(getattr(sprite, "width", 1)) // 2,
    )


def _r11l_state_features(env: Any, frame: Any, *, target_level: int = TARGET_LEVEL) -> dict[str, Any]:
    groups = getattr(env._game, "kacotwgjcyq", {}) or {}
    total = 0
    unsatisfied = 0
    manhattan = 0
    for data in groups.values():
        composite = data.get("roduyfsmiznvg")
        target = data.get("gosubdcyegamj")
        if not composite or not target:
            continue
        total += 1
        if not composite.collides_with(target):
            unsatisfied += 1
            cy, cx = _sprite_center(composite)
            ty, tx = _sprite_center(target)
            manhattan += abs(cy - ty) + abs(cx - tx)
    levels = _levels_completed(frame, env)
    return {
        "game_family": TARGET_GAME,
        "level": int(min(target_level, levels + 1)),
        "levels_completed": int(levels),
        "total_targets": int(total),
        "satisfied_targets": int(max(0, total - unsatisfied)),
        "unsatisfied_targets": int(unsatisfied),
        "manhattan_to_target": int(manhattan),
    }


def _load_goal_predicate() -> Any:
    path = REPO / "results" / "experiment_4020_goal_induction_separation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    code = str(payload.get("goal_predicate_code") or "")
    if not code:
        raise RuntimeError("exp4020 goal_predicate_code is empty")
    return compile_goal_predicate(code)


class R11LVerifiedMacroWorldModel:
    """Verifier simulator that expands r11l L4 by safe-composite macro actions.

    Each successor is produced on a deep-copied environment, so search can spend
    simulated actions without mutating the live environment.  The only real-env
    mutation happens after a plan has been selected.
    """

    def __init__(self, env: Any, game_action: Any, goal_predicate: Any, *, target_level: int) -> None:
        self.game_action = game_action
        self.goal_predicate = goal_predicate
        self.target_level = int(target_level)
        self._envs: dict[str, Any] = {}
        self._counter = 0
        self.start_state = self._register(copy.deepcopy(env), None)

    def _register(self, env: Any, frame: Any) -> dict[str, Any]:
        state = _r11l_state_features(env, frame, target_level=self.target_level)
        state_id = f"s{self._counter}"
        self._counter += 1
        state["state_id"] = state_id
        self._envs[state_id] = env
        return state

    def is_goal(self, state: dict[str, Any]) -> bool:
        return int(state.get("levels_completed", 0)) >= self.target_level or bool(self.goal_predicate(state))

    def next_states(self, state: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if self.is_goal(state):
            return []
        env = self._envs[str(state["state_id"])]
        try:
            moves, predicted_after = _build_safe_path_moves(env, self.game_action, self.target_level)
        except Exception as exc:
            state["simulator_bottleneck"] = f"{type(exc).__name__}: {exc}"
            return []
        if not moves:
            state["simulator_bottleneck"] = "no_safe_composite_path_macro_generated"
            return []

        sim_env = copy.deepcopy(env)
        frame, click_actions, _, _, _ = _execute_moves(sim_env, moves, self.game_action, self.target_level)
        child = self._register(sim_env, frame)
        macro = {
            "kind": "r11l_safe_composite_path_macro",
            "moves": moves,
            "predicted_levels_after": int(predicted_after),
            "simulated_levels_after": int(_levels_completed(frame, sim_env)),
            "simulated_click_actions": int(click_actions),
            "heuristic_after": coded_goal_distance_heuristic(child),
        }
        return [(macro, child)]


def _execute_first_mpc_macro(env: Any, game_action: Any, plan: list[Any], *, target_level: int) -> tuple[bool, int, int]:
    executed = 0
    frame = None
    for macro in plan[:1]:
        moves = list(macro.get("moves", [])) if isinstance(macro, dict) else []
        if not moves:
            continue
        frame, click_actions, _, _, _ = _execute_moves(env, moves, game_action, target_level)
        executed += int(click_actions)
        if _levels_completed(frame, env) >= target_level:
            return True, _levels_completed(frame, env), executed
    return False, _levels_completed(frame, env), executed


def _run_r11l_wall_search(*, max_expansions: int = MAX_EXPANSIONS, budget: int = 160) -> R11LRunOutcome:
    from arcengine.enums import GameAction

    arc = _load_offline_arcade()
    game_id = _select_game_id(arc, TARGET_GAME)
    env = arc.make(game_id)
    frame, _, _ = _advance_r11l_to_l3(env, GameAction, budget)
    if _levels_completed(frame, env) < PRIOR_WALL_LEVEL:
        return R11LRunOutcome(
            search=SearchResult(
                solved=False,
                actions=[],
                nodes_expanded=0,
                final_state=_r11l_state_features(env, frame),
                bottleneck="banked_r11l_L3_replay_failed",
                max_expansions=max_expansions,
            ),
            real_env_confirmed=False,
            levels_completed_after=_levels_completed(frame, env),
            executed_actions=0,
            per_step_replans=0,
            diagnosis="could not re-establish the prior r11l L3 wall state before search",
        )

    predicate = _load_goal_predicate()
    confirmed = False
    levels_after = _levels_completed(frame, env)
    executed_actions = 0
    replans = 0
    total_nodes = 0
    executed_plan: list[Any] = []
    search = SearchResult(
        solved=False,
        actions=[],
        nodes_expanded=0,
        final_state=_r11l_state_features(env, frame),
        bottleneck="not_started",
        max_expansions=max_expansions,
    )
    while levels_after < TARGET_LEVEL and replans < 8 and total_nodes < max_expansions:
        remaining_expansions = max(1, max_expansions - total_nodes)
        model = R11LVerifiedMacroWorldModel(env, GameAction, predicate, target_level=TARGET_LEVEL)
        search = best_first_search(
            model.start_state,
            next_states=model.next_states,
            is_goal=model.is_goal,
            heuristic=coded_goal_distance_heuristic,
            max_expansions=remaining_expansions,
        )
        total_nodes += search.nodes_expanded
        if not search.solved or not search.actions:
            search = SearchResult(
                solved=False,
                actions=executed_plan,
                nodes_expanded=total_nodes,
                final_state=search.final_state,
                bottleneck=search.bottleneck or "frontier_exhausted",
                max_expansions=max_expansions,
            )
            break
        replans += 1
        executed_plan.append(search.actions[0])
        confirmed, levels_after, click_actions = _execute_first_mpc_macro(
            env,
            GameAction,
            search.actions,
            target_level=TARGET_LEVEL,
        )
        executed_actions += click_actions
        if confirmed:
            search = SearchResult(
                solved=True,
                actions=executed_plan,
                nodes_expanded=total_nodes,
                final_state=_r11l_state_features(env, None),
                bottleneck="",
                max_expansions=max_expansions,
            )
            break
    if not confirmed and search.solved:
        search = SearchResult(
            solved=True,
            actions=executed_plan or search.actions,
            nodes_expanded=total_nodes,
            final_state=search.final_state,
            bottleneck="real_env_confirmation_failed",
            max_expansions=max_expansions,
        )
    elif not confirmed and total_nodes >= max_expansions:
        search = SearchResult(
            solved=False,
            actions=executed_plan,
            nodes_expanded=total_nodes,
            final_state=search.final_state,
            bottleneck="expansion_bound_exhausted",
            max_expansions=max_expansions,
        )
    elif not confirmed and replans >= 8:
        search = SearchResult(
            solved=False,
            actions=executed_plan,
            nodes_expanded=total_nodes,
            final_state=search.final_state,
            bottleneck="mpc_replan_limit_reached",
            max_expansions=max_expansions,
        )

    diagnosis = (
        "search found a verifier-simulated L4 plan and the real env levels_completed counter confirmed it"
        if confirmed
        else f"bounded search did not advance beyond r11l L4: {search.bottleneck or 'unconfirmed_plan'}"
    )
    return R11LRunOutcome(
        search=search,
        real_env_confirmed=bool(confirmed),
        levels_completed_after=int(levels_after),
        executed_actions=int(executed_actions),
        per_step_replans=int(replans),
        diagnosis=diagnosis,
    )


def run(*, max_expansions: int = MAX_EXPANSIONS, write: bool = True) -> dict[str, Any]:
    started = time.time()
    try:
        outcome = _run_r11l_wall_search(max_expansions=max_expansions)
    except Exception as exc:
        outcome = R11LRunOutcome(
            search=SearchResult(
                solved=False,
                actions=[],
                nodes_expanded=0,
                final_state={},
                bottleneck=f"precondition_or_runtime_blocked_{type(exc).__name__}",
                max_expansions=max_expansions,
            ),
            real_env_confirmed=False,
            levels_completed_after=PRIOR_WALL_LEVEL,
            executed_actions=0,
            per_step_replans=0,
            diagnosis=f"search could not run on the offline r11l wall: {type(exc).__name__}: {exc}",
        )

    artifact = build_search_artifact(
        outcome.search,
        game=TARGET_GAME,
        target_level=TARGET_LEVEL,
        prior_level=PRIOR_WALL_LEVEL,
        real_env_confirmed=outcome.real_env_confirmed,
        heuristic_used=DEFAULT_HEURISTIC_NAME,
        inference_substrate=INFERENCE_SUBSTRATE,
        duration_s=time.time() - started,
        random_seed=RANDOM_SEED,
        diagnosis=outcome.diagnosis,
    )
    artifact["levels_completed_after"] = int(outcome.levels_completed_after)
    artifact["executed_real_env_actions"] = int(outcome.executed_actions)
    artifact["per_step_mpc_replans"] = int(outcome.per_step_replans)
    artifact["prior_single_step_stall"] = "experiment_4014 r11l L4 single-action candidate held"
    artifact["model_reuse_note"] = "reused offline verified env-copy simulator plus Exp 4020 sandboxed is_goal; no new induction"

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, REPO / "results" / RESULT_NAME)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-expansions", type=int, default=MAX_EXPANSIONS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(max_expansions=args.max_expansions, write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - CLI exercised by the experiment command
    main()
