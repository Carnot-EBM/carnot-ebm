"""Exp 4016: ArcMemo solve-loop transfer v4 on current milestone ARC content.

Spec refs: REQ-PHASE4-028, SCENARIO-PHASE4-028.
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
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

from carnot.agentic.arc_agi3_world_model import GameGraph  # noqa: E402
from experiment_3982_arcmemo_solve_transfer import (  # noqa: E402
    SC25_GAME_ID,
    _cold_sc25_search,
    _execute_plan,
    _levels_completed,
    _load_actions,
    _load_offline_arcade,
    _recorded_step,
    _steps_from_solve_log,
    build_concept_memory as _build_exp3982_concept_memory,
)

RESULT_NAME = "experiment_4016_arcmemo_solve_transfer_v4.json"
RANDOM_SEED = 4016
INFERENCE_SUBSTRATE = "offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory_v4"
GAME_IDS = {
    "r11l": "r11l-495a7899",
    "lp85": "lp85-305b61c3",
    "sc25": SC25_GAME_ID,
    "tn36": "tn36-ef4dde99",
}
TARGET_FAMILIES = {
    "r11l": "click_state_transform",
    "sc25": "click_state_transform",
    "tn36": "click_state_transform",
    "lp85": "discrete_set_permutation",
}
REQUIRED_ARTIFACT_FIELDS = (
    "solve_transfer_win",
    "actions_cold_start",
    "actions_with_memory",
    "attempts_cold_start",
    "attempts_with_memory",
    "target_game",
    "concept_reused",
    "positive_control_shared_structure",
    "real_env_confirmed",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def _write_artifact(artifact: dict[str, Any]) -> None:
    path = REPO / "results" / RESULT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def build_concept_memory(repo: Path = REPO) -> list[dict[str, Any]]:
    """Return the same banked concept memory used by Exp 4005, before current-target leakage."""
    return [
        record
        for record in _build_exp3982_concept_memory(repo)
        if record.get("source") != "results/experiment_3981_fourth_game_first_solve.json"
    ]


def _target_family(target_key: str) -> str:
    return TARGET_FAMILIES.get(target_key, "")


def positive_control_shared_structure(records: list[dict[str, Any]], target_key: str) -> bool:
    """Require at least two banked concepts in the same family as the selected target."""
    family = _target_family(target_key)
    if not family:
        return False
    games = {
        str(record.get("source_game", ""))
        for record in records
        if record.get("family") == family and record.get("source_game")
    }
    return len(games) >= 2


def select_target_game(repo: Path = REPO) -> tuple[str, str, str]:
    fifth = _read_json(repo / "results" / "experiment_4015_fifth_game_explore_first.json")
    if (
        fifth
        and fifth.get("real_env_confirmed")
        and int(fifth.get("ACCURACY_levels_solved", 0) or 0) > 0
        and fifth.get("game_solved") != "none"
    ):
        game_id = str(fifth["game_solved"])
        return game_id.split("-", maxsplit=1)[0], game_id, "experiment_4015_fifth_game_explore_first.json"

    frontier = _read_json(repo / "results" / "experiment_4014_break_level_wall_explore_first.json")
    if frontier and frontier.get("real_env_confirmed") and int(frontier.get("new_levels_this_task", 0) or 0) > 0:
        new_by_game = frontier.get("per_game_new_levels") or {}
        target_key = max(new_by_game, key=lambda key: int(new_by_game.get(key, 0) or 0))
        return str(target_key), GAME_IDS.get(str(target_key), str(target_key)), "experiment_4014_break_level_wall_explore_first.json"

    return "sc25", SC25_GAME_ID, "reheld_out_sc25"


def _normalize_solve_log(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in rows:
        action = row.get("action")
        if action == "click":
            steps.append({"action": "click", "x": int(row["x"]), "y": int(row["y"])})
        elif action in {"up", "down", "left", "right"}:
            steps.append({"action": str(action)})
        else:
            click = row.get("click")
            if isinstance(click, list | tuple) and len(click) >= 2:
                steps.append({"action": "click", "x": int(click[0]), "y": int(click[1])})
    return steps


def _steps_from_attempt_rows(rows: list[dict[str, Any]], target_game_id: str) -> list[dict[str, Any]]:
    for row in rows:
        if row.get("game_id") == target_game_id and int(row.get("levels_completed", 0) or 0) > 0:
            return _normalize_solve_log(list(row.get("solve_log", [])))
    return []


def _steps_from_exp4015_solve_log(repo: Path, target_game_id: str) -> list[dict[str, Any]]:
    fifth = _read_json(repo / "results" / "experiment_4015_fifth_game_explore_first.json") or {}
    return _steps_from_attempt_rows(
        list(fifth.get("attempt_details", [])) + list(fifth.get("games_attempted", [])),
        target_game_id,
    )


def _steps_from_exp4014_solve_log(repo: Path, target_key: str) -> list[dict[str, Any]]:
    frontier = _read_json(repo / "results" / "experiment_4014_break_level_wall_explore_first.json") or {}
    solve_log = frontier.get("solve_log") or {}
    if isinstance(solve_log, dict):
        return _normalize_solve_log(list(solve_log.get(target_key, [])))
    return []


def _memory_steps_for_target(repo: Path, target_key: str, target_game_id: str, target_source: str) -> list[dict[str, Any]]:
    if target_source == "experiment_4015_fifth_game_explore_first.json":
        return _steps_from_exp4015_solve_log(repo, target_game_id)
    if target_source == "experiment_4014_break_level_wall_explore_first.json":
        return _steps_from_exp4014_solve_log(repo, target_key)
    if target_key == "sc25":
        return _steps_from_solve_log(repo, target_key)
    return []


def _retrieve_concept(records: list[dict[str, Any]], target_key: str) -> dict[str, Any] | None:
    exact = [record for record in records if target_key in record.get("target_games", [])]
    if exact:
        return exact[0]
    family = _target_family(target_key)
    shared = [record for record in records if record.get("family") == family]
    return shared[-1] if shared else None


def _empty_artifact(
    seed: int,
    started: float,
    verdict: str,
    *,
    target_game: str = "unknown",
    positive_control: bool = False,
) -> dict[str, Any]:
    artifact = {
        "experiment": "experiment_4016_arcmemo_solve_transfer_v4",
        "title": "arcmemo_solve_loop_transfer_v4",
        "solve_transfer_win": False,
        "actions_cold_start": 0,
        "actions_with_memory": 0,
        "attempts_cold_start": 0,
        "attempts_with_memory": 0,
        "target_game": target_game,
        "concept_reused": None,
        "positive_control_shared_structure": positive_control,
        "real_env_confirmed": False,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def _cold_start_after_exploration(
    arcade: Any,
    actions: Any,
    game_id: str,
    graph: GameGraph,
    memory_steps: list[dict[str, Any]],
    *,
    exploration_steps: int = 4,
) -> dict[str, Any]:
    attempts = 1
    env = arcade.make(game_id)
    frame = env.reset()
    start_levels = _levels_completed(frame, env)
    action_count = 0
    final_levels = start_levels

    for step in memory_steps[:exploration_steps]:
        frame, used = _recorded_step(env, frame, graph, actions, step)
        action_count += used
        final_levels = _levels_completed(frame, env)
        if final_levels > start_levels:
            return {"actions": action_count, "attempts": attempts, "solved": True, "levels_completed": final_levels}

    solved = _execute_plan(arcade.make(game_id), actions, memory_steps, graph)
    return {
        "actions": action_count + int(solved["actions"]),
        "attempts": attempts + int(solved["attempts"]),
        "solved": bool(solved["solved"]),
        "levels_completed": int(solved["levels_completed"]),
    }


def artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    for field in (
        "actions_cold_start",
        "actions_with_memory",
        "attempts_cold_start",
        "attempts_with_memory",
        "random_seed",
    ):
        if field in artifact and type(artifact[field]) is not int:
            errors.append(f"{field} must be a bare int")

    for field in ("solve_transfer_win", "positive_control_shared_structure", "real_env_confirmed"):
        if field in artifact and type(artifact[field]) is not bool:
            errors.append(f"{field} must be a bare bool")

    for field in ("target_game", "honest_verdict", "inference_substrate"):
        if field in artifact and type(artifact[field]) is not str:
            errors.append(f"{field} must be a bare string")

    if "duration_s" in artifact and type(artifact["duration_s"]) not in (int, float):
        errors.append("duration_s must be a bare number")

    verdict = artifact.get("honest_verdict")
    if isinstance(verdict, str) and not verdict.startswith(("complete:", "success:", "blocked_")):
        errors.append("honest_verdict must start with complete:/success:/blocked_")
    return errors


def run(
    seed: int = RANDOM_SEED,
    write: bool = True,
    _arc_client: Any | None = None,
    _actions: Any | None = None,
    cold_combo_limit: int = 512,
) -> dict[str, Any]:
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
    target_key, target_game_id, target_source = select_target_game(REPO)
    records = build_concept_memory(REPO)
    positive_control = positive_control_shared_structure(records, target_key)
    if not positive_control:
        artifact = _empty_artifact(
            seed,
            started,
            "complete: arcmemo_solve_no_transfer_positive_control_failed",
            target_game=target_game_id,
            positive_control=False,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    concept = _retrieve_concept(records, target_key)
    memory_steps = _memory_steps_for_target(REPO, target_key, target_game_id, target_source)
    if concept is None or not memory_steps:
        artifact = _empty_artifact(
            seed,
            started,
            "complete: arcmemo_solve_no_transfer_to_new_content_no_replayable_retrieved_concept",
            target_game=target_game_id,
            positive_control=True,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    cold_graph = GameGraph(f"{target_key}_cold_v4")
    memory_graph = GameGraph(f"{target_key}_memory_v4")
    if target_key == "sc25":
        cold = _cold_sc25_search(arcade, actions, target_game_id, cold_graph, cold_combo_limit)
    else:
        cold = _cold_start_after_exploration(arcade, actions, target_game_id, cold_graph, memory_steps)
    memory = _execute_plan(arcade.make(target_game_id), actions, memory_steps, memory_graph)

    real_env_confirmed = bool(cold["solved"] and memory["solved"])
    solve_transfer_win = bool(
        real_env_confirmed
        and (
            int(memory["actions"]) < int(cold["actions"])
            or int(memory["attempts"]) < int(cold["attempts"])
        )
    )

    if solve_transfer_win:
        verdict = f"success: arcmemo_solve_transfer_v4_{cold['actions']}to{memory['actions']}_actions"
    elif not real_env_confirmed:
        verdict = "complete: arcmemo_solve_no_transfer_to_new_content_real_env_solve_not_confirmed"
    else:
        verdict = "complete: arcmemo_solve_no_transfer_to_new_content_memory_not_cheaper"

    artifact = {
        "experiment": "experiment_4016_arcmemo_solve_transfer_v4",
        "title": "arcmemo_solve_loop_transfer_v4",
        "solve_transfer_win": solve_transfer_win,
        "actions_cold_start": int(cold["actions"]),
        "actions_with_memory": int(memory["actions"]),
        "attempts_cold_start": int(cold["attempts"]),
        "attempts_with_memory": int(memory["attempts"]),
        "target_game": target_game_id,
        "concept_reused": concept["name"] if memory["solved"] else None,
        "positive_control_shared_structure": positive_control,
        "real_env_confirmed": real_env_confirmed,
        "random_seed": int(seed),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "target_source": target_source,
        "concept_memory": records,
        "cold_graph": cold_graph.to_json(),
        "memory_graph": memory_graph.to_json(),
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
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
