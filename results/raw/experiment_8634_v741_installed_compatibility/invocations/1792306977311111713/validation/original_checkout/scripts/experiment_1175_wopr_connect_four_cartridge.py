"""Experiment 1175: WOPR Connect Four Ising cartridge.

Runs the Connect Four occupancy cartridge on five representative boards, runs
the focused pytest suite, and writes the conductor artifact for exp1175.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

if (
    VENV_PYTHON.exists()
    and Path(sys.executable).resolve() != VENV_PYTHON.resolve()
    and os.environ.get("CARNOT_NO_VENV_REEXEC") != "1"
):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

from carnot.games.connect_four import ConnectFourIsingCartridge  # noqa: E402


ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1175_wopr_connect_four_cartridge.json"
TEST_PATH = REPO_ROOT / "tests" / "python" / "games" / "test_connect_four.py"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_pytest() -> tuple[int, int, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", str(TEST_PATH), "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    n_tests_passing = 0
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+) passed", line)
        if match:
            n_tests_passing = int(match.group(1))
    return result.returncode, n_tests_passing, result.stdout[-1000:] + result.stderr[-500:]


def _score_position(name: str, board: np.ndarray) -> dict[str, object]:
    cartridge = ConnectFourIsingCartridge(initial_board=board)
    initial_energy = cartridge.energy(board)
    sampled = cartridge.sample(n_steps=1000, beta=2.0)
    sampled_energy = cartridge.energy(sampled)
    return {
        "name": name,
        "initial_energy": float(initial_energy),
        "sampled_energy": float(sampled_energy),
        "sampled_valid": bool(cartridge.is_valid(sampled)),
        "winner": cartridge.check_winner(sampled),
        "piece_count": int((sampled != 0).sum()),
    }


def _valid_partial_board() -> np.ndarray:
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:3] = [1, 2, 1]
    board[4, 0] = 2
    return board


def _gravity_violated_board() -> np.ndarray:
    board = np.zeros((6, 7), dtype=np.int8)
    board[2, 3] = 1
    return board


def _near_win_red_board() -> np.ndarray:
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0:3] = 1
    return board


def _near_win_yellow_board() -> np.ndarray:
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 4] = 2
    board[4, 4] = 2
    board[3, 4] = 2
    return board


def _winner_detection_working() -> bool:
    cartridge = ConnectFourIsingCartridge(initial_pieces=4)
    horizontal = np.zeros((6, 7), dtype=np.int8)
    horizontal[5, 0:4] = 1
    vertical = np.zeros((6, 7), dtype=np.int8)
    vertical[2:6, 4] = 2
    diagonal = np.zeros((6, 7), dtype=np.int8)
    diagonal[5, 0] = 1
    diagonal[4, 1] = 1
    diagonal[3, 2] = 1
    diagonal[2, 3] = 1
    return (
        cartridge.check_winner(horizontal) == "RED"
        and cartridge.check_winner(vertical) == "YELLOW"
        and cartridge.check_winner(diagonal) == "RED"
    )


def main() -> int:
    start = datetime.now(UTC)
    empty_board = np.zeros((6, 7), dtype=np.int8)
    valid_board = _valid_partial_board()
    gravity_board = _gravity_violated_board()

    empty_cartridge = ConnectFourIsingCartridge(initial_board=empty_board)
    valid_cartridge = ConnectFourIsingCartridge(initial_board=valid_board)
    gravity_cartridge = ConnectFourIsingCartridge(initial_board=gravity_board)

    positions = [
        _score_position("empty_board", empty_board),
        _score_position("partially_filled_valid_board", valid_board),
        _score_position("gravity_violated_board", gravity_board),
        _score_position("near_win_red_three_in_row", _near_win_red_board()),
        _score_position("near_win_yellow_three_in_row", _near_win_yellow_board()),
    ]

    test_returncode, n_tests_passing, test_tail = _run_pytest()
    winner_detection_working = _winner_detection_working()
    all_sampled_e0 = all(item["sampled_energy"] == 0.0 for item in positions)
    all_sampled_valid = all(bool(item["sampled_valid"]) for item in positions)

    empty_board_energy = float(empty_cartridge.energy(empty_board))
    valid_board_energy = float(valid_cartridge.energy(valid_board))
    gravity_violated_energy = float(gravity_cartridge.energy(gravity_board))
    n_spins = int(empty_cartridge.n_spins)

    cartridge_shipped = bool(
        test_returncode == 0
        and n_tests_passing >= 6
        and empty_board_energy == 0.0
        and valid_board_energy == 0.0
        and gravity_violated_energy > 0.0
        and winner_detection_working
        and n_spins == 42
        and all_sampled_e0
        and all_sampled_valid
    )
    if cartridge_shipped:
        honest_verdict = "cartridge_shipped_e0_at_convergence"
    elif test_returncode == 0 or n_tests_passing > 0:
        honest_verdict = "cartridge_partial_tests_fail"
    else:
        honest_verdict = "implementation_blocked"

    artifact = {
        "experiment": "exp1175_wopr_connect_four_cartridge",
        "run_date": _now(),
        "schema": "v1",
        "duration_s": round((datetime.now(UTC) - start).total_seconds(), 2),
        "cartridge_file": "python/carnot/games/connect_four.py",
        "cartridge_shipped": cartridge_shipped,
        "n_tests_passing": int(n_tests_passing),
        "empty_board_energy": empty_board_energy,
        "valid_board_energy": valid_board_energy,
        "gravity_violated_energy": gravity_violated_energy,
        "winner_detection_working": bool(winner_detection_working),
        "n_spins": n_spins,
        "position_results": positions,
        "pytest_returncode": int(test_returncode),
        "pytest_tail": test_tail,
        "honest_verdict": honest_verdict,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
        handle.write("\n")

    print(json.dumps(artifact, indent=2))
    return 0 if cartridge_shipped else 1


if __name__ == "__main__":
    sys.exit(main())
