"""Regression fixtures for the ARC submission gate's CORE set-containment verdict.

REQ: arc-action-efficiency-lever-gate / SCENARIO: noise-robust-solve-rate-preserving-verdict

WHY THIS EXISTS (2026-06-20): the gate's OLD verdict compared raw solved-COUNT, so a score-lever
that merely reordered the chaotic ~7800-action near-budget search and flipped one knife-edge solve
4<->3 FAILed automatically -- even when it removed nothing of merit (A1 frame-change prune had no-op
recall 0.053 yet dropped m0r0 from solved@7789 to unsolved@7803; A2 imitation prior same signature).
The CORE set-containment verdict fixes this WITHOUT going soft: it must still FAIL a config that
trades CORE solves for fringe ones. These four fixtures are the load-bearing proof of that property.
The relaxation is NOT accepted unless all four assert.

CORE (the games the verified baseline solves) = {lp85, m0r0, sp80, vc33}, baseline actions
lp85=7792, m0r0=7789, sp80=7724, vc33=7731 (median 7760).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "arc_local_submission_gate.py"
_spec = importlib.util.spec_from_file_location("arc_local_submission_gate", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _baseline() -> dict:
    """The verified CORE baseline: 4 core games solved at ~7724-7792 actions."""
    return {
        "solved_count": 4,
        "solved_games": ["lp85", "m0r0", "sp80", "vc33"],
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
    }


def _cur(solved: dict[str, int]) -> dict:
    """Build a treatment measurement dict from a {game: actions} map."""
    return {
        "solved_count": len(solved),
        "solved_games": sorted(solved),
        "actions_by_game": dict(solved),
        "median_actions_on_solved": (sorted(solved.values())[len(solved) // 2] if solved else None),
    }


def test_a1_frame_change_prune_fails_lost_core_m0r0():
    """A1: dropped m0r0 (a CORE solve) while reordering -> MUST FAIL (lost CORE solve)."""
    # A1 with_prune actuals: lp85@7905, sp80@7766, vc33@7726 solved; m0r0 NOT solved.
    ok, msg = gate._verdict(_cur({"lp85": 7905, "sp80": 7766, "vc33": 7726}), _baseline())
    assert ok is False
    assert "m0r0" in msg and "CORE" in msg


def test_a2_imitation_prior_fails_core_traded_for_fringe():
    """A2-style: trade 3 CORE solves for 2 fringe solves -> MUST FAIL (lost CORE), bonus NOT netted."""
    # vc33 (core) kept faster + two FRINGE games solved, but lp85/sp80/m0r0 (core) lost.
    ok, msg = gate._verdict(_cur({"vc33": 7733, "su15": 7700, "ls20": 7710}), _baseline())
    assert ok is False
    assert "lost CORE" in msg
    # the fringe gains must NOT rescue it -- a count/intersection metric would wrongly PASS this.


def test_positive_core_faster_passes_improved():
    """All CORE solves preserved AND median actions cut -> PASS (IMPROVED)."""
    ok, msg = gate._verdict(
        _cur({"lp85": 4000, "m0r0": 4100, "sp80": 3900, "vc33": 3950}), _baseline())
    assert ok is True
    assert "IMPROVED" in msg


def test_neutral_core_same_passes_non_inferior():
    """CORE preserved, actions unchanged -> PASS (non-inferior); the deterministic-rerun control."""
    ok, msg = gate._verdict(
        _cur({"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731}), _baseline())
    assert ok is True
    assert "non-inferior" in msg


def test_bonus_solve_reported_but_core_required():
    """A new fringe solve is a reported BONUS on top of a preserved CORE -> PASS with BONUS noted."""
    ok, msg = gate._verdict(
        _cur({"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731, "ft09": 7600}), _baseline())
    assert ok is True
    assert "BONUS" in msg and "ft09" in msg


def test_legacy_baseline_without_core_keys_falls_back_to_count():
    """An old baseline JSON (no solved_games/actions_by_game) -> legacy count check, still works."""
    legacy_base = {"solved_count": 4, "median_actions_on_solved": 7760.0,
                   "per_game": [{"game": "lp85", "solved": True, "actions": 7792},
                                {"game": "m0r0", "solved": True, "actions": 7789},
                                {"game": "sp80", "solved": True, "actions": 7724},
                                {"game": "vc33", "solved": True, "actions": 7731}]}
    # per_game present -> CORE reconstructed -> losing m0r0 still FAILs.
    ok, msg = gate._verdict(_cur({"lp85": 7905, "sp80": 7766, "vc33": 7726}), legacy_base)
    assert ok is False and "m0r0" in msg
