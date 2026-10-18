"""Experiment 1097: WOPR N-Queens cartridge.

Validates the 8x8 N-Queens WOPR cartridge, runs the focused cartridge tests,
and writes the milestone artifact expected by the conductor.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


REPO_ROOT = _repo_root()
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")

if (
    os.path.exists(VENV_PYTHON)
    and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON)
    and os.environ.get("CARNOT_NO_VENV_REEXEC") != "1"
):
    os.execv(VENV_PYTHON, [VENV_PYTHON, *sys.argv])


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


start_dt = datetime.now(UTC)
start_ts = _now()

CARTRIDGE_PATH = os.path.join(REPO_ROOT, "spaces", "wopr-games", "games", "nqueens.py")
TEST_PATH = os.path.join(REPO_ROOT, "tests", "python", "test_nqueens_cartridge.py")
ARTIFACT_PATH = os.path.join(REPO_ROOT, "results", "experiment_1097_wopr_nqueens_cartridge.json")

cartridge_file_written = os.path.exists(CARTRIDGE_PATH)
print(f"[1] Cartridge file exists: {cartridge_file_written} ({CARTRIDGE_PATH})")

sys.path.insert(0, os.path.join(REPO_ROOT, "spaces", "wopr-games"))

final_energy = -1.0
n_iterations_to_solution = 0

try:
    from games.nqueens import BOARD_SIZE, N_SPINS, NQueensGame

    game = NQueensGame(seed=17)
    steps = game.carnot_solve(max_iterations=50000)
    if steps:
        final_energy = float(steps[-1].energy)
        n_iterations_to_solution = int(steps[-1].iteration + 1)
    else:
        state = game.initial_state()
        final_energy = float(game.energy(state))
        n_iterations_to_solution = 0

    print(f"[2] Game solved: final_energy={final_energy}, iterations={n_iterations_to_solution}")
except Exception as exc:  # noqa: BLE001
    BOARD_SIZE = 8
    N_SPINS = 64
    print(f"[2] Game run error: {exc}")


test_result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        TEST_PATH,
        "-v",
    ],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
)

tests_passing = 0
for line in test_result.stdout.splitlines():
    match = re.search(r"(\d+) passed", line)
    if match:
        tests_passing = int(match.group(1))

print(f"[3] Tests: {tests_passing} passing (returncode={test_result.returncode})")
if test_result.stdout:
    print(test_result.stdout[-1000:])
if test_result.stderr:
    print(test_result.stderr[-500:])

if cartridge_file_written and final_energy == 0.0 and tests_passing >= 5:
    honest_verdict = "cartridge_shipped"
elif cartridge_file_written and final_energy > 0.0:
    honest_verdict = "cartridge_partial_energy_nonzero"
else:
    honest_verdict = "failed"

duration_s = (datetime.now(UTC) - start_dt).total_seconds()

artifact = {
    "experiment": "exp1097_wopr_nqueens_cartridge",
    "run_date": start_ts,
    "schema": "v1",
    "duration_s": round(duration_s, 2),
    "cartridge_file_written": bool(cartridge_file_written),
    "n_queens": int(BOARD_SIZE),
    "n_spins": int(N_SPINS),
    "final_energy": float(final_energy),
    "n_iterations_to_solution": int(n_iterations_to_solution),
    "tests_passing": int(tests_passing),
    "honest_verdict": honest_verdict,
}

os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)
    f.write("\n")

print(f"[4] Honest verdict: {honest_verdict}")
print(f"[5] Artifact written: {ARTIFACT_PATH}")
print(json.dumps(artifact, indent=2))

sys.exit(0 if honest_verdict == "cartridge_shipped" else 1)
