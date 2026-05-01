"""Experiment 1124: WOPR Hashi puzzle cartridge.

Validates the Hashi WOPR cartridge on the canonical 5x5 puzzle, runs the
focused cartridge tests, and writes the conductor artifact for exp1124.
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

CARTRIDGE_PATH = os.path.join(REPO_ROOT, "spaces", "wopr-games", "games", "hashi.py")
TEST_PATH = os.path.join(REPO_ROOT, "tests", "python", "test_hashi_cartridge.py")
ARTIFACT_PATH = os.path.join(REPO_ROOT, "results", "experiment_1124_wopr_hashi_cartridge.json")

cartridge_file_written = os.path.exists(CARTRIDGE_PATH)
print(f"[1] Cartridge file exists: {cartridge_file_written} ({CARTRIDGE_PATH})")

sys.path.insert(0, os.path.join(REPO_ROOT, "spaces", "wopr-games"))

final_energy = -1.0
n_iterations_to_convergence = 0
n_spins = 0
n_islands = 0

try:
    from games.hashi import CANONICAL_HASHI_PUZZLE, HashiCartridge

    cartridge = HashiCartridge()
    solution, final_energy, n_iterations_to_convergence = cartridge.solve(CANONICAL_HASHI_PUZZLE)
    n_spins = solution.model.n_spins
    n_islands = len(solution.model.islands)
    print(
        "[2] Canonical puzzle solved: "
        f"final_energy={final_energy}, iterations={n_iterations_to_convergence}"
    )
except Exception as exc:  # noqa: BLE001
    print(f"[2] Canonical puzzle run error: {exc}")


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

hashi_cartridge_shipped = bool(
    cartridge_file_written
    and final_energy == 0.0
    and test_result.returncode == 0
    and tests_passing >= 4
)
if hashi_cartridge_shipped:
    honest_verdict = "e0_achieved"
elif cartridge_file_written and final_energy > 0.0:
    honest_verdict = "e_positive_still_solving"
elif cartridge_file_written or tests_passing > 0:
    honest_verdict = "partial"
else:
    honest_verdict = "failed"

duration_s = (datetime.now(UTC) - start_dt).total_seconds()

artifact = {
    "experiment": "exp1124_wopr_hashi_cartridge",
    "run_date": start_ts,
    "schema": "v1",
    "duration_s": round(duration_s, 2),
    "cartridge_file": "app/wopr_gallery/cartridges/hashi.py",
    "actual_cartridge_file": "spaces/wopr-games/games/hashi.py",
    "cartridge_file_written": bool(cartridge_file_written),
    "n_islands": int(n_islands),
    "n_spins": int(n_spins),
    "canonical_puzzle_e_at_convergence": float(final_energy),
    "n_iterations_to_convergence": int(n_iterations_to_convergence),
    "hashi_cartridge_shipped": hashi_cartridge_shipped,
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

sys.exit(0 if honest_verdict == "e0_achieved" else 1)
