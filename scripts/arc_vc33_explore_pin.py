#!/usr/bin/env python3
"""Live pin for vc33's `actions_to_first_solve`, and for WHAT decides it.

WHY THIS IS A SCRIPT AND NOT A PYTEST TEST. It drives the real offline arcade, which needs
`environment_files/<game>/` -- gitignored, ~75 KB of game source per game, absent from a clean
checkout. A pytest test asserting these numbers would go red on any fresh clone; a test that
cannot run in a clean checkout is a worse failure mode than the skip it would replace, and
skips are forbidden (CLAUDE.md "Tests Must Run and Assert"). The asset-independent half of the
same finding IS pinned in the suite, at
`tests/python/test_arc_live_asset_arm_confound_2026_07_30.py`.

WHAT IT PINS (2026-07-30 measurement, seed 1, LLM-free -- the proposer is a stub that raises if
it is ever invoked, so the window is provably free of generation):

  * with the live action-effect scorer DISABLED, vc33 completes level 1 in exactly 10 actions
    with ZERO inductions. Zero inductions is the load-bearing half: it is what proves the
    number cannot be moved by any goal-predicate / world-model-induction change, which is what
    a 2026-07-30 pre-flight wrongly concluded had regressed it.

  * with the shipped configuration AND the untracked live assets present
    (`results/experiment_4629_live_frame_change_cnn.pt` + `data/arc_transition_corpus/*.npz`),
    the same episode takes 15 actions. The +5 is the scorer's cost on this game -- a real,
    pre-existing production effect (both assets predate the commits that were blamed for it),
    NOT a regression introduced by any commit.

If the assets are absent (a fresh clone, or a `git worktree`), the second pin cannot be
measured and the script says so and exits 3 -- it never silently reports the first number as
if it were production behaviour, which is precisely the substitution that produced the false
attribution.

THE CELL THAT BREAKS THE CONFOUND, recorded here so the pinned record contains it. Running each
commit in ONE tree, with the two untracked assets symlinked in and then removed
(`actions_to_first_solve` / action-trace sha256 prefix):

    commit                          assets PRESENT      assets ABSENT
    8441055c0 (pre-flight's base)   15  (no trace)      10  (no trace)
    aa8a38e31 (f9a458e87's parent)  15  19ca5e74        10  e693e8c5
    f9a458e87 (the accused commit)  15  19ca5e74        10  e693e8c5
    HEAD (6fc2bd17b)                15  19ca5e74        10  e693e8c5

The commit axis is inert in BOTH asset conditions -- which is what a commit-only or asset-only
sweep could not have shown. This script deliberately does NOT grow an `--at-commit` mode to
reproduce those cells: doing so would put worktree creation and asset symlinking inside a pin
whose whole value is being simple enough to trust. The reproduction driver is
`scratchpad/f35_cell.py` in the session that measured them; the numbers are recorded above and
in the correction artifact.

Usage:  python scripts/arc_vc33_explore_pin.py [--game vc33] [--seed 1]
Exit:   0 both pins hold; 1 a pin failed; 3 assets absent (second pin unmeasurable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# Expected values, measured 2026-07-30 on seed 1. Changing one of these is a deliberate act:
# say WHY in the commit, with the replacement measurement.
EXPECTED_NO_SCORER_ACTIONS = 10
EXPECTED_WITH_SCORER_ACTIONS = 15


class _StubProposer:
    """Any CALL is a hard failure. That is the proof this window is LLM-free rather than the
    assumption that it is -- the explore stream runs before any induction on vc33."""

    def __init__(self) -> None:
        self.no_think_prefix = None
        self.max_tokens = None
        self.tries = None
        self.include_playbook_exemplars = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("proposer invoked: this window is NOT llm-free")

    def propose(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("proposer.propose invoked: this window is NOT llm-free")


def _run(game: str, seed: int, budget: int, *, scorer_enabled: bool) -> dict[str, Any]:
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca

    saved = aca.SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED
    try:
        # Toggled in-process only. The shipped default on disk is never edited -- flipping a
        # SUBMITTED_* default is an operator decision, not a measurement's side effect.
        aca.SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED = bool(scorer_enabled)
        # The policy is handed an ANONYMIZED id (the held-out / hidden-game simulation) while
        # the ENV runs the real game, matching the probe this pin reproduces.
        anon = "hg" + hashlib.sha256(f"{game}|heldout".encode()).hexdigest()[:6]
        res = atp.run_bounded_progress(
            game,
            "frozen_gemma_pin",
            proposer=_StubProposer(),
            seed=seed,
            budget=budget,
            max_inductions=1,
            wall_s=900.0,
            explore_budget=24,
            policy_game_id=anon,
        )
    finally:
        aca.SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED = saved
    return res.to_row(include_events=True, include_trace=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="vc33")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    # A per-run private engine store: the canonical `results/arc_e3/` is EVIDENCE and is never
    # written by a measurement. Set BEFORE the first import -- E3_DIR resolves at import time.
    tmp = tempfile.mkdtemp(prefix="arc_vc33_pin_")
    os.environ["CARNOT_ARC_E3_DIR"] = tmp
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(REPO / "python"))

    from carnot.agentic.arc_frame_change_predictor import (
        LIVE_CNN_CHECKPOINT_RELATIVE_PATH,
        TRANSITION_CORPUS_RELATIVE_DIR,
        load_live_action_effect_scorer,
    )

    report: dict[str, Any] = {"game": args.game, "seed": args.seed, "repo": str(REPO)}

    off = _run(args.game, args.seed, EXPECTED_NO_SCORER_ACTIONS + 1, scorer_enabled=False)
    report["scorer_off"] = {
        "actions_to_first_solve": off.get("actions_to_first_solve"),
        "levels_gained": off.get("levels_gained"),
        "n_inductions": off.get("n_inductions"),
    }
    failures: list[str] = []
    if off.get("actions_to_first_solve") != EXPECTED_NO_SCORER_ACTIONS:
        failures.append(
            f"scorer-off actions_to_first_solve = {off.get('actions_to_first_solve')!r}, "
            f"expected {EXPECTED_NO_SCORER_ACTIONS}"
        )
    if off.get("n_inductions") != 0:
        failures.append(
            f"scorer-off n_inductions = {off.get('n_inductions')!r}, expected 0 -- the pin is "
            "only meaningful while no induction runs inside the window"
        )

    scorer = load_live_action_effect_scorer(root=REPO)
    report["live_assets_present"] = {
        "cnn_checkpoint": (REPO / LIVE_CNN_CHECKPOINT_RELATIVE_PATH).exists(),
        "transition_corpus_shards": len(list((REPO / TRANSITION_CORPUS_RELATIVE_DIR).glob("*.npz")))
        if (REPO / TRANSITION_CORPUS_RELATIVE_DIR).is_dir()
        else 0,
        "scorer_loaded": scorer is not None,
    }
    if scorer is None:
        report["verdict"] = "assets_absent_second_pin_unmeasurable"
        report["failures"] = failures
        print(json.dumps(report, indent=2))
        return 1 if failures else 3

    on = _run(args.game, args.seed, EXPECTED_WITH_SCORER_ACTIONS + 1, scorer_enabled=True)
    report["scorer_on"] = {
        "actions_to_first_solve": on.get("actions_to_first_solve"),
        "levels_gained": on.get("levels_gained"),
        "n_inductions": on.get("n_inductions"),
    }
    if on.get("actions_to_first_solve") != EXPECTED_WITH_SCORER_ACTIONS:
        failures.append(
            f"scorer-on actions_to_first_solve = {on.get('actions_to_first_solve')!r}, "
            f"expected {EXPECTED_WITH_SCORER_ACTIONS}"
        )

    report["failures"] = failures
    report["verdict"] = "pins_hold" if not failures else "pin_failed"
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
