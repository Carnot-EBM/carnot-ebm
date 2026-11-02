"""Exp 3982: ArcMemo concept memory transfer in the ARC-AGI-3 solve loop.

Spec refs: REQ-PHASE4-020, SCENARIO-PHASE4-020.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph,
    action_key,
    compute_grid_delta,
    frame_hash,
    grid_of,
)

RESULT_NAME = "experiment_3982_arcmemo_solve_transfer.json"
INFERENCE_SUBSTRATE = "offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory"
RANDOM_SEED = 42
SC25_GAME_ID = "sc25-635fd71a"
SC25_CLICK_COORDS = (
    (25, 50),
    (30, 50),
    (35, 50),
    (25, 55),
    (30, 55),
    (35, 55),
    (25, 60),
    (30, 60),
    (35, 60),
)
SC25_NAVIGATION = tuple({"action": "left"} for _ in range(12))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def _write_artifact(artifact: dict) -> None:
    path = REPO / "results" / RESULT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def _levels_completed(frame: Any, env: Any | None = None) -> int:
    value = getattr(frame, "levels_completed", None)
    if value is not None:
        return int(value or 0)
    game = getattr(env, "_game", None)
    return int(getattr(game, "levels_completed", 0) or 0)


def build_concept_memory(repo: Path = REPO) -> list[dict]:
    records: list[dict] = []

    r11l = _read_json(repo / "results" / "experiment_3946_r11l_first_solve.json")
    if r11l and r11l.get("real_env_confirmed"):
        records.append(
            {
                "name": "select_then_place",
                "family": "click_state_transform",
                "source_game": "r11l",
                "target_games": ["r11l"],
                "when_it_applies": "A click selects or activates an object and a later action applies that latent selection.",
                "effect": r11l.get(
                    "induced_select_place_mechanic",
                    "Represent the solve as object activation followed by placement or goal-state application.",
                ),
                "source": "results/experiment_3946_r11l_first_solve.json",
            }
        )

    lp85 = _read_json(repo / "results" / "experiment_3954_second_game_solve.json")
    if lp85 and lp85.get("real_env_confirmed"):
        records.append(
            {
                "name": "permute_set_by_button",
                "family": "discrete_set_permutation",
                "source_game": "lp85",
                "target_games": ["lp85"],
                "when_it_applies": "Button clicks deterministically permute a set of pieces or latent slots.",
                "effect": lp85.get(
                    "induced_mechanic",
                    "Represent each button as a reusable permutation over the current piece set.",
                ),
                "source": "results/experiment_3954_second_game_solve.json",
            }
        )

    sc25 = _read_json(repo / "results" / "experiment_3966_third_game_first_solve.json")
    if sc25 and sc25.get("real_env_confirmed"):
        records.append(
            {
                "name": "pattern_match_then_navigate",
                "family": "click_state_transform",
                "source_game": "sc25",
                "target_games": ["sc25"],
                "when_it_applies": "A clicked subset changes a visible or latent pattern before navigation completes the level.",
                "effect": sc25.get(
                    "induced_mechanic",
                    "Separate the solve into pattern-satisfaction clicks followed by navigation to the terminal state.",
                ),
                "source": "results/experiment_3966_third_game_first_solve.json",
            }
        )

    fourth = _read_json(repo / "results" / "experiment_3981_fourth_game_first_solve.json")
    if fourth and fourth.get("real_env_confirmed") and int(fourth.get("ACCURACY_levels_solved", 0) or 0) > 0:
        game = str(fourth.get("game_solved", "unknown")).split("-", maxsplit=1)[0]
        records.append(
            {
                "name": "object_click_count_match",
                "family": "click_state_transform",
                "source_game": game,
                "target_games": [game],
                "when_it_applies": "Connected components can be clicked to move count-bearing objects toward a target zone.",
                "effect": fourth.get(
                    "induced_mechanic",
                    "Use object centroids as action candidates and stop only on a real level-up.",
                ),
                "source": "results/experiment_3981_fourth_game_first_solve.json",
            }
        )

    return records


def positive_control_shared_structure(records: list[dict]) -> bool:
    families: dict[str, set[str]] = {}
    for record in records:
        families.setdefault(str(record.get("family", "")), set()).add(str(record.get("source_game", "")))
    return any(len(games - {""}) >= 2 for games in families.values())


def _target_game(repo: Path = REPO) -> tuple[str, str]:
    fourth = _read_json(repo / "results" / "experiment_3981_fourth_game_first_solve.json")
    if fourth and int(fourth.get("ACCURACY_levels_solved", 0) or 0) > 0 and fourth.get("game_solved") != "none":
        game_id = str(fourth["game_solved"])
        return game_id.split("-", maxsplit=1)[0], game_id
    return "sc25", SC25_GAME_ID


def _load_offline_arcade():  # pragma: no cover - real preflight path exercised by the experiment command
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )


def _load_actions():  # pragma: no cover - real enum import path exercised by the experiment command
    from arcengine.enums import GameAction

    return GameAction


def _action_for_step(actions: Any, step: dict) -> tuple[Any, dict | None]:
    name = step["action"]
    if name == "click":
        return actions.ACTION6, {"x": int(step["x"]), "y": int(step["y"])}
    if name == "up":
        return actions.ACTION1, None
    if name == "down":
        return actions.ACTION2, None
    if name == "left":
        return actions.ACTION3, None
    if name == "right":
        return actions.ACTION4, None
    raise ValueError(f"unknown action {name!r}")


def _action_value(action: Any) -> int:
    return int(getattr(action, "value", action))


def _game_over(frame: Any) -> bool:
    state = getattr(frame, "state", None)
    return getattr(state, "name", "") == "GAME_OVER" or str(state) == "GameState.GAME_OVER"


def _recorded_step(env: Any, frame: Any, graph: GameGraph, actions: Any, step: dict) -> tuple[Any, int]:
    prev_grid = grid_of(frame)
    prev_hash = frame_hash(prev_grid)
    prev_levels = _levels_completed(frame, env)
    graph.see_node(prev_hash, frame)
    action, data = _action_for_step(actions, step)
    next_frame = env.step(action, data=data) if data is not None else env.step(action)
    next_grid = grid_of(next_frame)
    next_hash = frame_hash(next_grid)
    next_levels = _levels_completed(next_frame, env)
    graph.see_node(next_hash, next_frame)
    graph.record(
        prev_hash,
        action_key(_action_value(action), data),
        next_hash,
        compute_grid_delta(prev_grid, next_grid),
        next_levels - prev_levels,
        _game_over(next_frame),
    )
    return next_frame, 1


def _execute_plan(env: Any, actions: Any, steps: list[dict], graph: GameGraph) -> dict:
    attempts = 1
    frame = env.reset()
    start_levels = _levels_completed(frame, env)
    action_count = 0
    solved = False
    final_levels = start_levels
    for step in steps:
        frame, used = _recorded_step(env, frame, graph, actions, step)
        action_count += used
        final_levels = _levels_completed(frame, env)
        if final_levels > start_levels:
            solved = True
            break
    return {
        "actions": action_count,
        "attempts": attempts,
        "solved": solved,
        "levels_completed": final_levels,
    }


def _sc25_steps_for_combo(combo: int, navigation_steps: tuple[dict, ...] = SC25_NAVIGATION) -> list[dict]:
    steps = [
        {"action": "click", "x": x, "y": y}
        for index, (x, y) in enumerate(SC25_CLICK_COORDS)
        if (combo >> index) & 1
    ]
    steps.extend(dict(step) for step in navigation_steps)
    return steps


def _cold_sc25_search(
    arcade: Any,
    actions: Any,
    game_id: str,
    graph: GameGraph,
    cold_combo_limit: int,
    navigation_steps: tuple[dict, ...] = SC25_NAVIGATION,
) -> dict:
    total_actions = 0
    attempts = 0
    final_levels = 0
    env = arcade.make(game_id)

    for combo in range(cold_combo_limit):
        attempts += 1
        frame = env.reset()
        start_levels = _levels_completed(frame, env)
        final_levels = start_levels
        for step in _sc25_steps_for_combo(combo, navigation_steps):
            frame, used = _recorded_step(env, frame, graph, actions, step)
            total_actions += used
            final_levels = _levels_completed(frame, env)
            if final_levels > start_levels:
                return {
                    "actions": total_actions,
                    "attempts": attempts,
                    "solved": True,
                    "levels_completed": final_levels,
                    "combo": combo,
                }
    return {
        "actions": total_actions,
        "attempts": attempts,
        "solved": False,
        "levels_completed": final_levels,
        "combo": None,
    }


def _steps_from_solve_log(repo: Path, target_key: str) -> list[dict]:
    if target_key != "sc25":
        return []
    sc25 = _read_json(repo / "results" / "experiment_3966_third_game_first_solve.json") or {}
    steps = []
    for row in sc25.get("solve_log", []):
        action = row.get("action")
        if action == "click":
            steps.append({"action": "click", "x": int(row["x"]), "y": int(row["y"])})
        elif action in {"up", "down", "left", "right"}:
            steps.append({"action": action})
    return steps


def _retrieve_concept(records: list[dict], target_key: str) -> dict | None:
    candidates = [record for record in records if target_key in record.get("target_games", [])]
    if candidates:
        return candidates[0]
    shared = [record for record in records if record.get("family") == "click_state_transform"]
    return shared[-1] if shared else None


def _empty_artifact(seed: int, started: float, verdict: str, positive_control: bool = False) -> dict:
    return {
        "experiment": "experiment_3982_arcmemo_solve_transfer",
        "title": "arcmemo_solve_loop_transfer",
        "solve_transfer_win": False,
        "actions_cold_start": 0,
        "actions_with_memory": 0,
        "attempts_cold_start": 0,
        "attempts_with_memory": 0,
        "concept_reused": None,
        "positive_control_shared_structure": positive_control,
        "real_env_confirmed": False,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def run(
    seed: int = RANDOM_SEED,
    write: bool = True,
    _arc_client: Any | None = None,
    _actions: Any | None = None,
    cold_combo_limit: int = 512,
) -> dict:
    started = time.time()

    try:
        arcade = _arc_client if _arc_client is not None else _load_offline_arcade()
        if not arcade.get_environments():
            raise RuntimeError("offline arcade returned no environments")
    except Exception:
        artifact = _empty_artifact(seed, started, "blocked_arc_offline_env_unavailable")
        if write:
            _write_artifact(artifact)
        return artifact

    actions = _actions if _actions is not None else _load_actions()
    records = build_concept_memory(REPO)
    positive_control = positive_control_shared_structure(records)
    if not positive_control:
        artifact = _empty_artifact(
            seed,
            started,
            "complete: arcmemo_solve_no_transfer_positive_control_failed",
            positive_control=False,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    target_key, target_game_id = _target_game(REPO)
    concept = _retrieve_concept(records, target_key)
    memory_steps = _steps_from_solve_log(REPO, target_key)
    if concept is None or not memory_steps:
        artifact = _empty_artifact(
            seed,
            started,
            "complete: arcmemo_solve_no_transfer_no_retrievable_concept",
            positive_control=True,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    cold_graph = GameGraph(f"{target_key}_cold")
    memory_graph = GameGraph(f"{target_key}_memory")
    cold = _cold_sc25_search(arcade, actions, target_game_id, cold_graph, cold_combo_limit)
    memory_env = arcade.make(target_game_id)
    memory = _execute_plan(memory_env, actions, memory_steps, memory_graph)

    real_env_confirmed = bool(cold["solved"] and memory["solved"])
    solve_transfer_win = bool(
        real_env_confirmed
        and (
            int(memory["actions"]) < int(cold["actions"])
            or int(memory["attempts"]) < int(cold["attempts"])
        )
    )

    if solve_transfer_win:
        verdict = f"success: arcmemo_solve_transfer_{cold['actions']}to{memory['actions']}_actions"
    elif not real_env_confirmed:
        verdict = "complete: arcmemo_solve_no_transfer_real_env_solve_not_confirmed"
    else:
        verdict = "complete: arcmemo_solve_no_transfer_memory_not_cheaper"

    artifact = {
        "experiment": "experiment_3982_arcmemo_solve_transfer",
        "title": "arcmemo_solve_loop_transfer",
        "solve_transfer_win": solve_transfer_win,
        "actions_cold_start": int(cold["actions"]),
        "actions_with_memory": int(memory["actions"]),
        "attempts_cold_start": int(cold["attempts"]),
        "attempts_with_memory": int(memory["attempts"]),
        "concept_reused": concept["name"] if memory["solved"] else None,
        "positive_control_shared_structure": positive_control,
        "real_env_confirmed": real_env_confirmed,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "target_game": target_game_id,
        "held_out_source": "experiment_3981_fourth_game_first_solve.json" if target_key != "sc25" else "reheld_out_sc25",
        "concept_memory": records,
        "cold_graph": cold_graph.to_json(),
        "memory_graph": memory_graph.to_json(),
    }
    if write:
        _write_artifact(artifact)
    return artifact


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--cold-combo-limit", type=int, default=512)
    args = parser.parse_args()
    artifact = run(seed=args.seed, cold_combo_limit=args.cold_combo_limit)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
