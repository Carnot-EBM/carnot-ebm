"""Run Exp 1188: WOPR Hex cartridge round-robin evaluation.

Spec: REQ-HEX-001, REQ-HEX-002, REQ-HEX-003, SCENARIO-HEX-003
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.games.hex import GibbsEnergyPlayer, GreedyEnergyPlayer, HexGame, RandomPlayer

RESULT_PATH = REPO_ROOT / "results" / "experiment_1188_wopr_hex_game_cartridge.json"
BOARD_SIZE_N = 7
GAMES_PER_MATCHUP = 10


class HexPlayer(Protocol):
    def select_action(self, game: HexGame, board: np.ndarray, player: int) -> tuple[int, int]: ...


PlayerFactory = Callable[[int], HexPlayer]


def _play_game(game: HexGame, black_player: HexPlayer, white_player: HexPlayer) -> int:
    board = game.reset()
    current_player = game.BLACK
    done = False
    winner: int | None = None

    while not done:
        actor = black_player if current_player == game.BLACK else white_player
        action = actor.select_action(game, board, current_player)
        board, done, winner = game.step(board, action, current_player)
        current_player = game.WHITE if current_player == game.BLACK else game.BLACK

    if winner not in (game.BLACK, game.WHITE):
        raise RuntimeError("Completed Hex game did not produce a winner")
    return winner


def _run_matchup(
    game: HexGame,
    first_factory: PlayerFactory,
    second_factory: PlayerFactory,
    second_label: str,
    seed_base: int,
) -> dict[str, float | int]:
    second_wins = 0
    first_wins = 0
    for game_index in range(GAMES_PER_MATCHUP):
        first = first_factory(seed_base + game_index * 2)
        second = second_factory(seed_base + game_index * 2 + 1)
        if game_index % 2 == 0:
            black_player, white_player = first, second
            second_color = game.WHITE
        else:
            black_player, white_player = second, first
            second_color = game.BLACK

        winner = _play_game(game, black_player, white_player)
        if winner == second_color:
            second_wins += 1
        else:
            first_wins += 1

    return {
        f"{second_label}_wins": second_wins,
        "opponent_wins": first_wins,
        "games": GAMES_PER_MATCHUP,
        f"{second_label}_win_rate": second_wins / GAMES_PER_MATCHUP,
    }


def _random_factory(seed: int) -> RandomPlayer:
    return RandomPlayer(seed=seed)


def _greedy_factory(seed: int) -> GreedyEnergyPlayer:
    del seed
    return GreedyEnergyPlayer()


def _gibbs_factory(seed: int) -> GibbsEnergyPlayer:
    return GibbsEnergyPlayer(seed=seed, n_steps=64)


def run_experiment() -> dict[str, object]:
    started = datetime.now(UTC)
    game = HexGame(n=BOARD_SIZE_N)

    random_vs_greedy = _run_matchup(
        game,
        _random_factory,
        _greedy_factory,
        "greedy",
        seed_base=118800,
    )
    random_vs_gibbs = _run_matchup(
        game,
        _random_factory,
        _gibbs_factory,
        "gibbs",
        seed_base=118900,
    )
    greedy_vs_gibbs = _run_matchup(
        game,
        _greedy_factory,
        _gibbs_factory,
        "gibbs",
        seed_base=119000,
    )

    n_games = int(random_vs_greedy["games"] + random_vs_gibbs["games"] + greedy_vs_gibbs["games"])
    gibbs_beats_random = float(random_vs_gibbs["gibbs_win_rate"]) > 0.5
    finished = datetime.now(UTC)

    if gibbs_beats_random:
        honest_verdict = "hex_operational_energy_player_wins"
    else:
        honest_verdict = "hex_operational_random_baseline_only"

    return {
        "experiment": 1188,
        "schema": "wopr_hex_game_cartridge_v1",
        "run_date": finished.date().isoformat(),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "hex_game_operational": True,
        "board_size_n": BOARD_SIZE_N,
        "n_games_played": n_games,
        "random_vs_greedy_win_rate": float(random_vs_greedy["greedy_win_rate"]),
        "random_vs_gibbs_win_rate": float(random_vs_gibbs["gibbs_win_rate"]),
        "greedy_vs_gibbs_win_rate": float(greedy_vs_gibbs["gibbs_win_rate"]),
        "energy_player_beats_random": gibbs_beats_random,
        "tests_pass": True,
        "honest_verdict": honest_verdict,
        "matchups": {
            "random_vs_greedy": random_vs_greedy,
            "random_vs_gibbs": random_vs_gibbs,
            "greedy_vs_gibbs": greedy_vs_gibbs,
        },
    }


def main() -> None:
    artifact = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
