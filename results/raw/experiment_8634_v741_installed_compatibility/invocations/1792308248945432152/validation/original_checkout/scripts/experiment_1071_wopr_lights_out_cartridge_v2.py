"""Experiment 1071: WOPR Lights Out Cartridge with Ising Solver (v2).

Blocked in milestone .82 by Codex config error (now fixed).
Implements the Lights Out WOPR cartridge using ParallelIsingSampler for
ground-state search, validates E=0 is reached, and runs unit tests.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, UTC

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


start_ts = _now()
start_dt = datetime.now(UTC)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARTRIDGE_PATH = os.path.join(REPO_ROOT, "spaces", "wopr-games", "games", "lights_out.py")
TEST_PATH = os.path.join(REPO_ROOT, "tests", "python", "test_lights_out_cartridge.py")
ARTIFACT_PATH = os.path.join(
    REPO_ROOT, "results", "experiment_1071_wopr_lights_out_cartridge_v2.json"
)

# ---------------------------------------------------------------------------
# Step 1: verify cartridge file exists
# ---------------------------------------------------------------------------

cartridge_file_written = os.path.exists(CARTRIDGE_PATH)
print(f"[1] Cartridge file exists: {cartridge_file_written} ({CARTRIDGE_PATH})")

# ---------------------------------------------------------------------------
# Step 2: run the game and capture final_energy + ising flag
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(REPO_ROOT, "spaces", "wopr-games"))

final_energy: float = -1.0
ising_solver_used: bool = False

try:
    from games.lights_out import LightsOutGame

    game = LightsOutGame(seed=17)
    steps = game.carnot_solve(max_iterations=10000)
    final_energy = steps[-1].energy if steps else game.energy(game.initial_state())
    ising_solver_used = game.ising_solver_used
    print(f"[2] Game solved: final_energy={final_energy}, ising_used={ising_solver_used}")
    print(f"    Steps taken: {len(steps)}")
except Exception as exc:
    print(f"[2] Game run error: {exc}")

# ---------------------------------------------------------------------------
# Step 3: run the unit tests
# ---------------------------------------------------------------------------

test_result = subprocess.run(
    [sys.executable, "-m", "pytest", TEST_PATH, "-v", "--no-cov", "-q"],
    capture_output=True,
    text=True,
    cwd=REPO_ROOT,
)

tests_passing = 0
import re

for line in test_result.stdout.splitlines():
    m = re.search(r"(\d+) passed", line)
    if m:
        tests_passing = int(m.group(1))

print(f"[3] Tests: {tests_passing}/4 passing (returncode={test_result.returncode})")
if test_result.stdout:
    print(test_result.stdout[-800:])
if test_result.stderr:
    print(test_result.stderr[-400:])

# ---------------------------------------------------------------------------
# Step 4: determine honest verdict
# ---------------------------------------------------------------------------

if cartridge_file_written and final_energy == 0.0 and tests_passing >= 4:
    honest_verdict = "cartridge_shipped"
elif cartridge_file_written and final_energy is not None and final_energy > 0:
    honest_verdict = "cartridge_partial_energy_nonzero"
else:
    honest_verdict = "failed"

print(f"[4] Honest verdict: {honest_verdict}")

# ---------------------------------------------------------------------------
# Step 5: write artifact
# ---------------------------------------------------------------------------

duration_s = (datetime.now(UTC) - start_dt).total_seconds()

artifact = {
    "experiment": "exp1071_wopr_lights_out_cartridge_v2",
    "run_date": start_ts,
    "schema": "v1",
    "duration_s": round(duration_s, 2),
    "cartridge_file_written": cartridge_file_written,
    "ising_solver_used": ising_solver_used,
    "final_energy": float(final_energy),
    "max_iterations_tested": 10000,
    "tests_passing": tests_passing,
    "honest_verdict": honest_verdict,
}

os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
with open(ARTIFACT_PATH, "w") as f:
    json.dump(artifact, f, indent=2)

print(f"\n[5] Artifact written: {ARTIFACT_PATH}")
print(json.dumps(artifact, indent=2))

sys.exit(0 if honest_verdict == "cartridge_shipped" else 1)
