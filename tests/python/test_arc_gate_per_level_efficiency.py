"""The submission gate judges configs on PER-LEVEL efficiency (the real leaderboard metric).

REQ: arc-action-efficiency-per-level-gate / SCENARIO: per-level-efficiency-primary-verdict

WHY (2026-06-20): the real ARC-AGI-3 score is per-level sum(min(human/agent_per_level,1)^2), NOT total
actions (docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md). The gate used to
judge median TOTAL actions, which scored an efficient-but-over-running solve at ~0 (lp85 solved L1 in 20
actions == human-class, but ran to 7792 -> old metric said 0). The gate now uses per-level efficiency as
the PRIMARY metric (higher = better), with median-actions as a wall-clock fallback only when efficiency
data is absent.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "arc_local_submission_gate.py"
_spec = importlib.util.spec_from_file_location("arc_local_submission_gate_eff", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

CORE = ["lp85", "m0r0", "sp80", "vc33"]


def _m(solved_eff: dict[str, float], solved_actions: dict[str, int] | None = None) -> dict:
    """Build a measurement with per-game per-level efficiency (+ actions)."""
    solved_actions = solved_actions or {g: 5000 for g in solved_eff}
    rows = [{"game": g, "solved": True, "actions": solved_actions.get(g),
             "efficiency": solved_eff[g]} for g in solved_eff]
    return {
        "games": gate.GATE_GAMES, "action_metric": dict(gate.CANONICAL_ACTION_METRIC),
        "solved_count": len(solved_eff), "solved_games": sorted(solved_eff),
        "actions_by_game": dict(solved_actions),
        "efficiency_by_game": dict(solved_eff),
        "per_game": rows,
    }


def _baseline() -> dict:
    # baseline: lp85 efficient (0.72), others near-0 (the real measured CORE state)
    return _m({"lp85": 0.72, "m0r0": 0.0001, "sp80": 0.0, "vc33": 0.0},
              {"lp85": 20, "m0r0": 3891, "sp80": 7218, "vc33": 1758})


def test_efficiency_improvement_passes():
    """A config that RAISES CORE per-level efficiency -> PASS (IMPROVED)."""
    cur = _m({"lp85": 0.72, "m0r0": 0.5, "sp80": 0.0, "vc33": 0.0})  # m0r0 got efficient
    ok, msg = gate._verdict(cur, _baseline())
    assert ok is True and "IMPROVED" in msg and "efficiency" in msg


def test_efficiency_drop_fails():
    """A config that LOWERS CORE per-level efficiency (e.g. lp85 regressed) -> FAIL, even if it solves
    the same games and uses fewer total actions."""
    cur = _m({"lp85": 0.05, "m0r0": 0.0001, "sp80": 0.0, "vc33": 0.0})  # lp85 efficiency cratered
    ok, msg = gate._verdict(cur, _baseline())
    assert ok is False and "efficiency" in msg


def test_efficiency_is_primary_over_total_actions():
    """KEY: fewer TOTAL actions but LOWER per-level efficiency must FAIL -- the lp85 lesson. Old gate
    (median actions) would PASS this; the per-level gate must FAIL it."""
    # cur uses far fewer actions everywhere but lp85's per-level efficiency dropped 0.72 -> 0.05
    cur = _m({"lp85": 0.05, "m0r0": 0.0001, "sp80": 0.0, "vc33": 0.0},
             {"lp85": 80, "m0r0": 100, "sp80": 200, "vc33": 90})  # all far fewer actions
    ok, msg = gate._verdict(cur, _baseline())
    assert ok is False and "efficiency" in msg


def test_lost_core_solve_still_fails():
    """CORE solve-set containment is preserved -- losing a CORE solve fails before efficiency is checked."""
    cur = _m({"lp85": 0.72, "sp80": 0.0, "vc33": 0.0})  # m0r0 lost
    ok, msg = gate._verdict(cur, _baseline())
    assert ok is False and "m0r0" in msg


def test_sf2_missing_cur_efficiency_hard_fails():
    """SF-2 (adversarial review): if the baseline has per-level efficiency but the treatment emits none
    (broken eval), HARD FAIL -- do NOT silently fall back to the retired total-actions metric (which
    could PASS a real efficiency regression)."""
    cur = {"solved_count": 4, "solved_games": CORE,
           "action_metric": dict(gate.CANONICAL_ACTION_METRIC),
           "actions_by_game": {g: 50 for g in CORE}}  # no efficiency_by_game
    ok, msg = gate._verdict(cur, _baseline())
    assert ok is False and "could not measure per-level efficiency" in msg


def test_mf1_efficiency_baseline_passes_canonical_guard():
    """MF-1 (adversarial review): an efficiency-bearing baseline must PASS validate_canonical_baseline
    (the retired total-actions 7760 control must NOT block it). Re-baselining legitimately moves total
    actions, so the guard now checks per-level efficiency presence, not the 7760 control."""
    eff_base = {"games": list(gate.CANONICAL_GAME_SET),
                "action_metric": dict(gate.CANONICAL_ACTION_METRIC),
                "solved_games": list(CORE),
                "actions_by_game": {"lp85": 20, "m0r0": 3891, "sp80": 7218, "vc33": 1758},
                "efficiency_by_game": {"lp85": 2.0069, "m0r0": 0.0, "sp80": 0.0, "vc33": 0.0}}
    v = gate.validate_canonical_baseline(eff_base)
    assert v["ok"] is True and v["errors"] == []
    # but a baseline missing efficiency for a CORE game must still FAIL the guard
    bad = {**eff_base, "efficiency_by_game": {"lp85": 2.0069}}
    assert gate.validate_canonical_baseline(bad)["ok"] is False


def test_fallback_to_median_actions_when_no_efficiency():
    """Legacy baseline/fixtures WITHOUT efficiency data -> fall back to the median-actions wall-clock
    proxy (so B2's fixtures and an un-re-baselined gate keep working)."""
    base = {"solved_count": 4, "solved_games": CORE,
            "action_metric": dict(gate.CANONICAL_ACTION_METRIC),
            "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731}}
    cur = {"solved_count": 4, "solved_games": CORE,
           "action_metric": dict(gate.CANONICAL_ACTION_METRIC),
           "actions_by_game": {"lp85": 4000, "m0r0": 4100, "sp80": 3900, "vc33": 3950}}
    ok, msg = gate._verdict(cur, base)
    assert ok is True and "median actions" in msg  # used the fallback, not efficiency
