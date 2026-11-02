#!/usr/bin/env python3
"""Experiment 1141: WOPR Slitherlink rescue cartridge.

Validates the Slitherlink WOPR cartridge on the canonical 3x3 diamond puzzle,
runs the focused cartridge tests, and writes the conductor artifact.
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

CARTRIDGE_REL = "spaces/wopr-games/games/slitherlink.py"
CARTRIDGE_PATH = os.path.join(REPO_ROOT, *CARTRIDGE_REL.split("/"))
TEST_PATH = os.path.join(REPO_ROOT, "spaces", "wopr-games", "tests", "test_slitherlink.py")
ARTIFACT_PATH = os.path.join(REPO_ROOT, "results", "experiment_1141_wopr_slitherlink_rescue.json")

cartridge_file_written = os.path.exists(CARTRIDGE_PATH)
final_energy = float("inf")
n_iterations_to_convergence = 0
n_spins = 0
registered_in_app = False
runtime_error = ""

print(f"[1] Cartridge file exists: {cartridge_file_written} ({CARTRIDGE_REL})")

try:
    sys.path.insert(0, os.path.join(REPO_ROOT, "spaces", "wopr-games"))
    from games import ALL_GAMES
    from games.slitherlink import CANONICAL_SLITHERLINK_PUZZLE, SlitherinkCartridge

    cartridge = SlitherinkCartridge(CANONICAL_SLITHERLINK_PUZZLE, (3, 3))
    spins = cartridge.sample(n_steps=5000)
    final_energy = cartridge.energy(spins)
    n_iterations_to_convergence = cartridge.last_iterations_to_convergence
    n_spins = cartridge.n_spins
    registered_in_app = any(game.name == "SLITHERLINK" for game in ALL_GAMES)
    print(
        "[2] Canonical puzzle sampled: "
        f"final_energy={final_energy}, iterations={n_iterations_to_convergence}"
    )
except Exception as exc:  # noqa: BLE001
    runtime_error = str(exc)
    print(f"[2] Canonical puzzle run error: {runtime_error}")

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
    check=False,
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

slitherlink_cartridge_shipped = final_energy == 0.0
if runtime_error:
    honest_verdict = "blocked_runtime_error"
elif final_energy == 0.0:
    honest_verdict = "e0_achieved"
elif final_energy < 0.1:
    honest_verdict = "e_near_zero_below_01"
else:
    honest_verdict = "e_above_threshold"

duration_s = (datetime.now(UTC) - start_dt).total_seconds()

artifact = {
    "experiment": "exp1141_wopr_slitherlink_rescue",
    "run_date": start_ts,
    "schema": "v1",
    "duration_s": round(duration_s, 2),
    "cartridge_file": CARTRIDGE_REL,
    "cartridge_file_written": bool(cartridge_file_written),
    "n_spins": int(n_spins),
    "canonical_puzzle_e_at_convergence": float(final_energy),
    "n_iterations_to_convergence": int(n_iterations_to_convergence),
    "slitherlink_cartridge_shipped": bool(slitherlink_cartridge_shipped),
    "tests_passing": int(tests_passing),
    "registered_in_app": bool(registered_in_app),
    "honest_verdict": honest_verdict,
}

os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
with open(ARTIFACT_PATH, "w", encoding="utf-8") as artifact_file:
    json.dump(artifact, artifact_file, indent=2)
    artifact_file.write("\n")

print(f"[4] Honest verdict: {honest_verdict}")
print(f"[5] Artifact written: {ARTIFACT_PATH}")
print(json.dumps(artifact, indent=2))

sys.exit(0 if honest_verdict == "e0_achieved" and tests_passing == 5 else 1)
