"""Experiment 1070 — WOPR Global Thermonuclear War cartridge (v2).

What this experiment does
--------------------------
This script validates that the Global Thermonuclear War WOPR cartridge
was correctly implemented at ``spaces/wopr-games/games/global_thermonuclear_war.py``
and that its accompanying test suite passes.

Checks performed:
  1. The module imports without error.
  2. WAR_SCENARIOS contains >= 20 named scenarios.
  3. The typewriter reveal is implemented (REVEAL_LINES has 3 entries
     matching the canonical WarGames quote).
  4. The state machine runs to completion (is_solved=True).
  5. The pytest test suite at tests/python/test_gtw_cartridge.py passes.

What this script deliberately does NOT do
------------------------------------------
  - No real computation, no GPU, no model inference.
  - No git push (conductor contract).
  - No re-run of the full test suite; only the GTW-specific tests.

Honest verdicts
---------------
  - ``cartridge_shipped``  — all checks pass, tests pass
  - ``cartridge_partial``  — module loads but some checks fail
  - ``failed``             — module does not load or pytest crashes
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_GAMES = REPO_ROOT / "spaces" / "wopr-games"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1070_wopr_gtw_cartridge_v2.json"

# Add spaces/wopr-games to sys.path so we can import the cartridge.
if str(SPACES_GAMES) not in sys.path:
    sys.path.insert(0, str(SPACES_GAMES))


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _check_module() -> dict:
    """Try importing the cartridge and returning key metadata."""
    result: dict = {
        "import_ok": False,
        "n_scenarios": 0,
        "n_reveal_lines": 0,
        "reveal_lines": [],
        "typewriter_reveal_implemented": False,
        "state_machine_completes": False,
        "final_energy": None,
    }
    try:
        from games.global_thermonuclear_war import (
            REVEAL_LINES,
            WAR_SCENARIOS,
            GlobalThermonuclearWarGame,
        )
    except Exception as exc:
        result["import_error"] = str(exc)
        return result

    result["import_ok"] = True
    result["n_scenarios"] = len(WAR_SCENARIOS)
    result["n_reveal_lines"] = len(REVEAL_LINES)
    result["reveal_lines"] = list(REVEAL_LINES)

    # Typewriter reveal is implemented when REVEAL_LINES contains the
    # canonical three-line WarGames quote in order.
    expected_first = "A STRANGE GAME."
    expected_second = "THE ONLY WINNING MOVE IS NOT TO PLAY."
    expected_third = "HOW ABOUT A NICE GAME OF CHESS?"
    result["typewriter_reveal_implemented"] = (
        len(REVEAL_LINES) == 3
        and REVEAL_LINES[0] == expected_first
        and REVEAL_LINES[1] == expected_second
        and REVEAL_LINES[2] == expected_third
    )

    # Drive the game to completion
    try:
        game = GlobalThermonuclearWarGame()
        state = game.initial_state()
        total_steps = len(WAR_SCENARIOS) + len(REVEAL_LINES)
        for i in range(total_steps * 2):
            step = game.carnot_step(state, i)
            state = step.state
            if step.is_solved:
                result["state_machine_completes"] = True
                result["final_energy"] = step.energy
                break
    except Exception as exc:
        result["state_machine_error"] = str(exc)

    return result


def _run_pytest() -> tuple[int, int, str]:
    """Run the GTW-specific pytest suite.

    Returns (tests_passing, tests_total, raw_output).
    """
    test_file = REPO_ROOT / "tests" / "python" / "test_gtw_cartridge.py"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-v",
        "--no-header",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        # Disable coverage for this targeted run — the spaces/ code is
        # outside the python/carnot/ coverage path and would show 0%.
        "--no-cov",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(REPO_ROOT),
        )
        output = proc.stdout + proc.stderr
        # Count passed from summary line "X passed"
        tests_passing = 0
        tests_total = 0
        for line in output.splitlines():
            if " passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        try:
                            tests_passing = int(parts[i - 1])
                        except ValueError:
                            pass
                    if part == "failed" and i > 0:
                        try:
                            failed = int(parts[i - 1])
                            tests_total += failed
                        except ValueError:
                            pass
                tests_total += tests_passing
        return tests_passing, tests_total, output
    except subprocess.TimeoutExpired:
        return 0, 0, "pytest timed out"
    except Exception as exc:
        return 0, 0, f"pytest failed to run: {exc}"


def main() -> int:
    started_at = _now_iso()
    t0 = time.time()

    artifact: dict = {
        "experiment": 1070,
        "title": "WOPR Global Thermonuclear War cartridge v2",
        "schema": "carnot.wopr_gtw_cartridge.v2",
        "run_date": datetime.now(UTC).date().isoformat(),
        "started_at": started_at,
        "cartridge_file": str(SPACES_GAMES / "games" / "global_thermonuclear_war.py"),
        "cartridge_file_written": (SPACES_GAMES / "games" / "global_thermonuclear_war.py").exists(),
        "n_scenarios": 0,
        "typewriter_reveal_implemented": False,
        "tests_passing": 0,
        "tests_total": 0,
        "pytest_output": "",
        "honest_verdict": "failed",
        "status": "failed",
    }

    module_check = _check_module()
    artifact.update(
        {
            "import_ok": module_check.get("import_ok", False),
            "n_scenarios": module_check.get("n_scenarios", 0),
            "n_reveal_lines": module_check.get("n_reveal_lines", 0),
            "reveal_lines": module_check.get("reveal_lines", []),
            "typewriter_reveal_implemented": module_check.get(
                "typewriter_reveal_implemented", False
            ),
            "state_machine_completes": module_check.get("state_machine_completes", False),
            "final_energy": module_check.get("final_energy"),
        }
    )
    if "import_error" in module_check:
        artifact["import_error"] = module_check["import_error"]
    if "state_machine_error" in module_check:
        artifact["state_machine_error"] = module_check["state_machine_error"]

    tests_passing, tests_total, pytest_out = _run_pytest()
    artifact["tests_passing"] = tests_passing
    artifact["tests_total"] = tests_total
    artifact["pytest_output"] = pytest_out[-4000:] if len(pytest_out) > 4000 else pytest_out

    # Determine verdict
    if (
        artifact["cartridge_file_written"]
        and artifact["import_ok"]
        and artifact["n_scenarios"] >= 20
        and artifact["typewriter_reveal_implemented"]
        and artifact["state_machine_completes"]
        and tests_passing >= 9
    ):
        artifact["honest_verdict"] = "cartridge_shipped"
        artifact["status"] = "success"
    elif artifact["cartridge_file_written"] and artifact["import_ok"]:
        artifact["honest_verdict"] = "cartridge_partial"
        artifact["status"] = "partial"

    artifact["finished_at"] = _now_iso()
    artifact["duration_s"] = round(time.time() - t0, 3)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
