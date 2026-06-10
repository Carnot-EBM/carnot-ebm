"""Exp 3994: ArcMemo solve-loop transfer v2 for a new or re-held-out ARC game.

Spec refs: REQ-PHASE4-022, SCENARIO-PHASE4-022.
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
    RANDOM_SEED,
    SC25_GAME_ID,
    build_concept_memory as _build_exp3982_concept_memory,
    positive_control_shared_structure,
    _cold_sc25_search,
    _execute_plan,
    _load_actions,
    _load_offline_arcade,
    _retrieve_concept,
    _steps_from_solve_log,
)

RESULT_NAME = "experiment_3994_arcmemo_solve_transfer_v2.json"
INFERENCE_SUBSTRATE = "offline_arc_agi3_real_env_steps_plus_gamegraph_arcmemo_concept_memory_v2"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def _write_artifact(artifact: dict) -> None:
    path = REPO / "results" / RESULT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")


def build_concept_memory(repo: Path = REPO) -> list[dict]:
    """Return only banked solved-game concepts, excluding any fourth-game target leakage."""
    return [
        record
        for record in _build_exp3982_concept_memory(repo)
        if record.get("source") != "results/experiment_3981_fourth_game_first_solve.json"
    ]


def select_target_game(repo: Path = REPO) -> tuple[str, str, str]:
    fourth = _read_json(repo / "results" / "experiment_3993_fourth_game_verifier_pruned.json")
    if (
        fourth
        and fourth.get("real_env_confirmed")
        and int(fourth.get("ACCURACY_levels_solved", 0) or 0) > 0
        and fourth.get("game_solved") != "none"
    ):
        game_id = str(fourth["game_solved"])
        return game_id.split("-", maxsplit=1)[0], game_id, "experiment_3993_fourth_game_verifier_pruned.json"
    return "sc25", SC25_GAME_ID, "reheld_out_sc25"


def _memory_steps_for_target(repo: Path, target_key: str) -> list[dict]:
    if target_key == "sc25":
        return _steps_from_solve_log(repo, target_key)
    return []


def _empty_artifact(
    seed: int,
    started: float,
    verdict: str,
    *,
    target_game: str = "unknown",
    positive_control: bool = False,
) -> dict:
    return {
        "experiment": "experiment_3994_arcmemo_solve_transfer_v2",
        "title": "arcmemo_solve_loop_transfer_v2",
        "solve_transfer_win": False,
        "actions_cold_start": 0,
        "actions_with_memory": 0,
        "attempts_cold_start": 0,
        "attempts_with_memory": 0,
        "target_game": target_game,
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
    target_key, target_game_id, target_source = select_target_game(REPO)
    records = build_concept_memory(REPO)
    positive_control = positive_control_shared_structure(records)
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
    memory_steps = _memory_steps_for_target(REPO, target_key)
    if concept is None or not memory_steps:
        artifact = _empty_artifact(
            seed,
            started,
            "complete: arcmemo_solve_no_transfer_to_new_game_no_retrievable_concept",
            target_game=target_game_id,
            positive_control=True,
        )
        if write:
            _write_artifact(artifact)
        return artifact

    cold_graph = GameGraph(f"{target_key}_cold_v2")
    memory_graph = GameGraph(f"{target_key}_memory_v2")
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
        verdict = f"success: arcmemo_solve_transfer_v2_{cold['actions']}to{memory['actions']}_actions"
    elif not real_env_confirmed:
        verdict = "complete: arcmemo_solve_no_transfer_to_new_game_real_env_solve_not_confirmed"
    else:
        verdict = "complete: arcmemo_solve_no_transfer_to_new_game_memory_not_cheaper"

    artifact = {
        "experiment": "experiment_3994_arcmemo_solve_transfer_v2",
        "title": "arcmemo_solve_loop_transfer_v2",
        "solve_transfer_win": solve_transfer_win,
        "actions_cold_start": int(cold["actions"]),
        "actions_with_memory": int(memory["actions"]),
        "attempts_cold_start": int(cold["attempts"]),
        "attempts_with_memory": int(memory["attempts"]),
        "target_game": target_game_id,
        "concept_reused": concept["name"] if memory["solved"] else None,
        "positive_control_shared_structure": positive_control,
        "real_env_confirmed": real_env_confirmed,
        "random_seed": seed,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - started, 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "target_source": target_source,
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
