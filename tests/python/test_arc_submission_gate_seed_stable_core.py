"""Seed-stable CORE membership for the ARC submission gate.

REQ: arc-action-efficiency-lever-gate / SCENARIO: seed-stable-core-membership

WHY THIS EXISTS (2026-07-24). `measure()`'s 2026-06-20 comment states the intent that a knife-edge
marginal solve flipping from order-perturbation noise must NOT count as a regression (the A1/A2
lesson: a 5%-recall prune that removed ~nothing still reshuffled a chaotic ~7800-action trajectory
and dropped m0r0). The redesign delivered the COUNT half of that intent -- a lever can no longer
fail merely by solving a different same-size set. It did not deliver the other half: `_verdict`
hard-FAILs on any `core - cur_solved`, and CORE was "whatever the baseline solved in ONE run at ONE
budget", so a chaotic near-budget solve can enter CORE from a single lucky run and then veto any
lever that merely reshuffles search order.

The pre-existing stability machinery does not cover this. `select_headroom_budget` /
`stable_vs_1_5x` check the BUDGET axis (does the solved SET hold at 1.5x budget) to pick a budget;
raising the budget only ever helps a marginal solve, so it cannot detect instability along the
SEED / SEARCH-ORDER axis -- the axis a lever actually perturbs.

THE FIX TIGHTENS CORE RATHER THAN WEAKENING CONTAINMENT. Weakening containment would re-open the
hole the 2026-06-20 redesign closed (a lever trading CORE solves for fringe ones -- A2 swapped 3
core for 2 fringe). Instead a game earns veto power only by solving REPRODUCIBLY across runs.

CRITICALLY, THIS CHANGES NO VERDICT ON ITS OWN. It is backward compatible: a baseline with no
`core_stable_games` key -- every baseline written before 2026-07-24 -- behaves exactly as before.
Whether m0r0 specifically is seed-stable is an EMPIRICAL question that requires a multi-seed
baseline measurement. Excluding it by fiat would be the "relax the cap to admit a favoured result"
anti-pattern CLAUDE.md forbids, and would contradict the deliberate, load-bearing assertion in
test_arc_submission_gate_verdict.py that dropping m0r0 must FAIL. That test still passes unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "arc_local_submission_gate.py"
_spec = importlib.util.spec_from_file_location("arc_local_submission_gate", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

_METRIC = {"field": "actions", "definition": "total_actions_on_solved_games"}
_GAMES = ["lp85", "m0r0", "sp80", "vc33", "cd82"]


def _baseline(core_stable_games=None) -> dict:
    base = {
        "games": list(_GAMES),
        "action_metric": dict(_METRIC),
        "solved_count": 4,
        "solved_games": ["lp85", "m0r0", "sp80", "vc33"],
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
    }
    if core_stable_games is not None:
        base["core_stable_games"] = list(core_stable_games)
    return base


def _cur(solved: dict[str, int]) -> dict:
    return {
        "games": list(_GAMES),
        "action_metric": dict(_METRIC),
        "solved_count": len(solved),
        "solved_games": sorted(solved),
        "actions_by_game": dict(solved),
        "median_actions_on_solved": (sorted(solved.values())[len(solved) // 2] if solved else None),
    }


def test_stable_core_is_the_intersection_across_runs():
    """A game earns CORE only by solving in EVERY run; a 2-of-3 flake does not."""
    runs = [
        {"solved_games": ["lp85", "m0r0", "sp80", "vc33"]},
        {"solved_games": ["lp85", "sp80", "vc33"]},  # m0r0 flaked
        {"solved_games": ["lp85", "m0r0", "sp80", "vc33"]},
    ]
    assert gate.stable_core_from_runs(runs) == {"lp85", "sp80", "vc33"}


def test_stable_core_handles_single_run_and_empty():
    """One run -> that run's solves. No runs -> empty (never crash the gate)."""
    assert gate.stable_core_from_runs([{"solved_games": ["lp85", "m0r0"]}]) == {"lp85", "m0r0"}
    assert gate.stable_core_from_runs([]) == set()


def test_stable_core_reads_per_game_rows_when_solved_games_absent():
    """Falls back to per_game rows, matching _solved_set's contract."""
    runs = [
        {"per_game": [{"game": "lp85", "solved": True}, {"game": "m0r0", "solved": True}]},
        {"per_game": [{"game": "lp85", "solved": True}, {"game": "m0r0", "solved": False}]},
    ]
    assert gate.stable_core_from_runs(runs) == {"lp85"}


def test_backward_compatible_baseline_without_stable_core_still_fails_on_m0r0():
    """MUST-STILL-FAIL CONTROL 1 (backward compatibility).

    A baseline with no `core_stable_games` key -- i.e. every baseline written before this change --
    keeps the historical veto exactly. This is the same scenario as
    test_arc_submission_gate_verdict.py::test_a1_frame_change_prune_fails_lost_core_m0r0.
    """
    ok, msg = gate._verdict(_cur({"lp85": 7905, "sp80": 7766, "vc33": 7726}), _baseline())
    assert ok is False
    assert "m0r0" in msg and "CORE" in msg


def test_dropping_a_seed_stable_core_game_still_fails():
    """MUST-STILL-FAIL CONTROL 2: the relaxation must not go soft.

    Even with a seed-stable core recorded, losing a game that IS stable is still a hard regression.
    """
    base = _baseline(core_stable_games=["lp85", "sp80", "vc33"])
    ok, msg = gate._verdict(_cur({"m0r0": 7789, "sp80": 7724, "vc33": 7731}), base)  # lost lp85
    assert ok is False
    assert "lp85" in msg and "CORE" in msg


def test_trading_core_for_fringe_still_fails_under_stable_core():
    """MUST-STILL-FAIL CONTROL 3: the A2 hole stays closed.

    Swapping stable CORE solves for fringe ones must still FAIL -- the property the 2026-06-20
    redesign exists to protect. Fringe gains are a bonus, never netted against a core loss.
    """
    base = _baseline(core_stable_games=["lp85", "sp80", "vc33"])
    ok, msg = gate._verdict(_cur({"lp85": 7792, "cd82": 4000}), base)  # lost sp80+vc33, gained cd82
    assert ok is False
    assert "CORE" in msg


def test_unstable_game_loses_veto_power_but_stable_containment_holds():
    """The actual fix: a measured-unstable game cannot veto, while stable containment is intact.

    m0r0 excluded from `core_stable_games` (measured flaky across seeds) -> a lever that preserves
    every STABLE core solve is no longer hard-vetoed by m0r0's coin-flip.
    """
    base = _baseline(core_stable_games=["lp85", "sp80", "vc33"])
    ok, msg = gate._verdict(_cur({"lp85": 7792, "sp80": 7724, "vc33": 7731}), base)
    assert "lost CORE solves" not in msg
    assert ok is True or "efficiency" in msg.lower()
