"""Experiment 1059 — WOPR Games HuggingFace Space (Sudoku v1).

What this experiment does
-------------------------

It is a *packaging and verification* experiment, not a research one.
The Sudoku-as-energy-minimisation cartridge has existed under
``spaces/wopr-games/`` for several days; this script audits that it
is shippable by checking five concrete things:

  1. The Gradio Space's entry point (``spaces/wopr-games/app.py``)
     imports cleanly. Import-time errors would mean a broken Space.
  2. The Sudoku cartridge solves the demo puzzle to E=0 within a
     reasonable iteration budget (with simulated-annealing restarts
     to escape local minima — the within-row Metropolis sampler
     plateaus otherwise).
  3. The four WarGames-flavoured easter eggs the task spec demands
     all return the expected canonical responses. We hard-code the
     expected substrings so a future drift in ``EASTER_EGGS`` is
     caught here, not by a confused visitor.
  4. The HuggingFace ``HF_TOKEN`` environment variable is or is not
     set (deploy is gated on it; we never deploy without explicit
     human authorisation).
  5. A standardised artifact is written to
     ``results/experiment_1059_wopr_spaces_sudoku_v1.json`` so the
     conductor's reconciliation step can pick it up.

Why this lives in ``scripts/`` and not ``tests/``
-------------------------------------------------

The artifact contract (``REQUIRED_RESULT_FIELDS`` from
``scripts.experiment_template``) is the conductor's currency: it
expects a JSON deliverable with ``experiment``, ``schema``, dates,
duration, status, etc. Tests check correctness; an experiment
script is the *deliverable producer*.

The hard truth this script reports
----------------------------------

The Sudoku Metropolis sampler ships a *within-row swap* move set,
which can only fix column and box violations. On real puzzles it
plateaus at a non-zero energy — typically E≈2-6 — and needs
repeated restarts to find E=0. We check that restarts do reach
E=0 within ~120 s wall-clock; if the budget were tighter we would
need to swap in a stronger move-set or a different sampler. This
is a known limitation noted in the change proposal at
``openspec/change-proposals/huggingface-spaces-sudoku-demo.md``.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACES_DIR = REPO_ROOT / "spaces" / "wopr-games"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1059_wopr_spaces_sudoku_v1.json"


# Required easter-egg checks: (input phrase, substring that MUST appear in the
# response). Substrings are used rather than equality so future flavour-text
# polish doesn't break the check, while still catching outright drift.
REQUIRED_EASTER_EGGS: list[tuple[str, str]] = [
    ("LIST GAMES", "AVAILABLE GAMES"),
    ("GLOBAL THERMONUCLEAR WAR", "NICE GAME OF CHESS"),
    ("HOW ABOUT A NICE GAME OF CHESS", "SUDOKU"),
    ("GREETINGS PROFESSOR FALKEN", "SHALL WE PLAY A GAME"),
]


def _check_app_import() -> tuple[bool, str | None]:
    """Import the Space's app.py from a fresh interpreter path.

    Returns (ok, error_message). On success ``error_message`` is None.
    """
    sys.path.insert(0, str(SPACES_DIR))
    try:
        # Force a fresh import in case the module was already loaded with
        # a different sys.path during interactive testing.
        for mod in ("app", "wopr_shell", "games", "games.sudoku"):
            if mod in sys.modules:
                del sys.modules[mod]
        importlib.import_module("app")
        return True, None
    except Exception as exc:  # noqa: BLE001 — we report all import failures
        return False, f"{type(exc).__name__}: {exc}"


def _check_easter_eggs() -> tuple[int, list[dict]]:
    """Verify the four required easter eggs respond correctly.

    Returns (count_passed, per_egg_details).
    """
    sys.path.insert(0, str(SPACES_DIR))
    if "wopr_shell" in sys.modules:
        del sys.modules["wopr_shell"]
    from wopr_shell import respond_to_terminal_input  # type: ignore

    passed = 0
    details: list[dict] = []
    for query, expected_substr in REQUIRED_EASTER_EGGS:
        response = respond_to_terminal_input(query)
        ok = expected_substr.upper() in response.upper()
        details.append(
            {
                "query": query,
                "expected_substr": expected_substr,
                "response": response,
                "passed": ok,
            }
        )
        if ok:
            passed += 1
    return passed, details


def _solve_sudoku_with_restarts(
    max_iterations: int = 200_000,
    plateau_limit: int = 5_000,
    initial_seed: int = 7,
    wall_clock_budget_s: float = 120.0,
) -> dict:
    """Run the Sudoku cartridge with simulated-annealing restarts until E=0
    or the budgets are exhausted.

    The within-row Metropolis sampler in ``games.sudoku`` plateaus on real
    puzzles. To honestly demonstrate "energy reaches zero" we wrap it in
    the same restart-on-plateau pattern a deployed Space would use.
    """
    sys.path.insert(0, str(SPACES_DIR))
    if "games.sudoku" in sys.modules:
        del sys.modules["games.sudoku"]
    from games.sudoku import SudokuGame  # type: ignore

    game = SudokuGame(seed=initial_seed)
    state = game.initial_state()
    initial_energy = game.energy(state)
    best_energy = initial_energy
    plateau_count = 0
    restarts = 0
    solved_iter: int | None = None
    total_iters = 0

    t0 = time.time()
    for i in range(max_iterations):
        step = game.carnot_step(state, i)
        state = step.state
        total_iters = i
        if step.energy < best_energy:
            best_energy = step.energy
            plateau_count = 0
        else:
            plateau_count += 1
        if step.is_solved:
            solved_iter = i
            break
        if plateau_count > plateau_limit:
            restarts += 1
            game._rng = random.Random(initial_seed + restarts)
            state = game.initial_state()
            plateau_count = 0
        if time.time() - t0 > wall_clock_budget_s:
            break

    elapsed = time.time() - t0
    final_energy = game.energy(state)
    return {
        "initial_energy": float(initial_energy),
        "final_energy": float(final_energy),
        "solved_iter": solved_iter,
        "restarts": restarts,
        "total_iters": total_iters,
        "wall_clock_s": round(elapsed, 3),
        "reaches_zero": final_energy == 0.0,
    }


def _hf_token_present() -> bool:
    """Whether an HF deployment token is available in the environment."""
    for var in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(var):
            return True
    return False


def _verdict(
    code_complete: bool,
    deployed: bool,
    egg_count: int,
    energy_zero: bool,
) -> str:
    """Map the five outcomes to one of the four declared honest verdicts."""
    if not code_complete:
        return "failed"
    if not energy_zero:
        return "partial_no_solver"
    if deployed:
        return "space_deployed_full_demo"
    if egg_count >= 4 and energy_zero:
        return "code_complete_deploy_pending"
    return "partial_no_solver"


def main() -> int:
    started_at = datetime.now(UTC)
    t0 = time.time()

    # 1. Import check
    import_ok, import_err = _check_app_import()

    # 2. Easter eggs
    egg_count, egg_details = _check_easter_eggs()

    # 3. Sudoku solver reaches E=0
    solver_summary = _solve_sudoku_with_restarts()

    # 4. Token / deploy gate. We never auto-deploy in CI; the conductor
    #    should not be making external pushes. Deploy stays human-only.
    deploy_token_present = _hf_token_present()
    space_deployed = False
    space_url = None
    deploy_note = (
        "HF_TOKEN present but auto-deploy is disabled for this experiment "
        "to avoid surprising the operator with a public-facing push. "
        "Run `huggingface-cli upload Carnot-EBM/wopr-games spaces/wopr-games/ .` "
        "manually to ship."
        if deploy_token_present
        else "HF_TOKEN absent. Ship locally via "
        "`cd spaces/wopr-games && python app.py` for verification."
    )

    code_complete = import_ok and egg_count >= 4 and solver_summary["reaches_zero"]
    local_test_passed = import_ok and solver_summary["reaches_zero"]
    honest_verdict = _verdict(
        code_complete=code_complete,
        deployed=space_deployed,
        egg_count=egg_count,
        energy_zero=solver_summary["reaches_zero"],
    )

    finished_at = datetime.now(UTC)
    artifact = {
        "experiment": 1059,
        "title": "WOPR Games HuggingFace Space — Sudoku v1",
        "schema": "carnot.wopr_spaces_sudoku.v1",
        "run_date": started_at.date().isoformat(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_s": round(time.time() - t0, 3),
        "status": "success" if code_complete else "partial",
        "honest_verdict": honest_verdict,
        # Required task fields
        "space_code_complete": code_complete,
        "space_deployed": space_deployed,
        "space_url": space_url,
        "easter_eggs_implemented": egg_count,
        "sudoku_solver_energy_reaches_zero": solver_summary["reaches_zero"],
        "local_test_passed": local_test_passed,
        # Diagnostic fields
        "import_ok": import_ok,
        "import_error": import_err,
        "easter_egg_details": egg_details,
        "solver_summary": solver_summary,
        "deploy_token_present": deploy_token_present,
        "deploy_note": deploy_note,
        "spaces_dir": str(SPACES_DIR.relative_to(REPO_ROOT)),
        "decision_class": "verify",
        "cost_usd": 0.0,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {RESULT_PATH}")
    print(f"honest_verdict: {honest_verdict}")
    print(f"easter_eggs_implemented: {egg_count}/4 required")
    print(
        f"sudoku reaches zero: {solver_summary['reaches_zero']} "
        f"(restarts={solver_summary['restarts']}, "
        f"iters={solver_summary['total_iters']}, "
        f"t={solver_summary['wall_clock_s']}s)"
    )
    return 0 if code_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
