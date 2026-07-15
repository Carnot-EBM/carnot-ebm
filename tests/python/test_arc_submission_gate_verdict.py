"""Regression fixtures for the ARC submission gate's CORE set-containment verdict.

REQ: arc-action-efficiency-lever-gate / SCENARIO: noise-robust-solve-rate-preserving-verdict
REQ-ARC-FCP-4518 / SCENARIO-ARC-FCP-4518: canonical fixed-metric dashboard guard

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
import json
import sys
from pathlib import Path

_GATE = Path(__file__).resolve().parents[2] / "scripts" / "kaggle" / "arc_local_submission_gate.py"
_spec = importlib.util.spec_from_file_location("arc_local_submission_gate", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _baseline() -> dict:
    """The verified CORE baseline: 4 core games solved at ~7724-7792 actions."""
    return {
        "games": ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"],
        "action_metric": {"field": "actions", "definition": "total_actions_on_solved_games"},
        "solved_count": 4,
        "solved_games": ["lp85", "m0r0", "sp80", "vc33"],
        "actions_by_game": {"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731},
        "median_actions_on_solved": 7760.0,
    }


def _cur(solved: dict[str, int]) -> dict:
    """Build a treatment measurement dict from a {game: actions} map."""
    return {
        "games": ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"],
        "action_metric": {"field": "actions", "definition": "total_actions_on_solved_games"},
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
        _cur({"lp85": 4000, "m0r0": 4100, "sp80": 3900, "vc33": 3950}), _baseline()
    )
    assert ok is True
    assert "IMPROVED" in msg


def test_neutral_core_same_passes_non_inferior():
    """CORE preserved, actions unchanged -> PASS (non-inferior); the deterministic-rerun control."""
    ok, msg = gate._verdict(
        _cur({"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731}), _baseline()
    )
    assert ok is True
    assert "non-inferior" in msg


def test_bonus_solve_reported_but_core_required():
    """A new fringe solve is a reported BONUS on top of a preserved CORE -> PASS with BONUS noted."""
    ok, msg = gate._verdict(
        _cur({"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731, "ft09": 7600}), _baseline()
    )
    assert ok is True
    assert "BONUS" in msg and "ft09" in msg


def test_legacy_baseline_without_core_keys_falls_back_to_count():
    """An old baseline JSON (no solved_games/actions_by_game) -> legacy count check, still works."""
    legacy_base = {
        "solved_count": 4,
        "median_actions_on_solved": 7760.0,
        "per_game": [
            {"game": "lp85", "solved": True, "actions": 7792},
            {"game": "m0r0", "solved": True, "actions": 7789},
            {"game": "sp80", "solved": True, "actions": 7724},
            {"game": "vc33", "solved": True, "actions": 7731},
        ],
    }
    # per_game present -> CORE reconstructed -> losing m0r0 still FAILs.
    ok, msg = gate._verdict(_cur({"lp85": 7905, "sp80": 7766, "vc33": 7726}), legacy_base)
    assert ok is False and "m0r0" in msg


def test_req_arc_fcp_4518_baseline_and_game_set_guard():
    """REQ-ARC-FCP-4518: the 8-game set and 7760 baseline cannot move silently."""
    validation = gate.validate_canonical_baseline(_baseline())

    assert validation["ok"] is True
    assert validation["canonical_game_set"] == list(gate.CANONICAL_GAME_SET)
    assert validation["canonical_baseline_median_actions"] == 7760.0
    assert validation["core_games"] == ["lp85", "m0r0", "sp80", "vc33"]

    moved = _baseline()
    moved["median_actions_on_solved"] = 7761.0
    moved_errors = gate.validate_canonical_baseline(moved)["errors"]
    assert any("7760" in error for error in moved_errors)

    cherry_picked = _baseline()
    cherry_picked["games"] = ["lp85", "m0r0", "sp80", "vc33"]
    cherry_errors = gate.validate_canonical_baseline(cherry_picked)["errors"]
    assert any("fixed 8-game" in error for error in cherry_errors)


def test_req_arc_fcp_4518_rejects_mismatched_action_metric_field():
    """REQ-ARC-FCP-4518: actions-to-first-level-up cannot be compared to total actions."""
    cur = _cur({"lp85": 2984, "m0r0": 2984, "sp80": 2984, "vc33": 2984})
    cur["action_metric"] = {
        "field": "actions_to_first_levelup",
        "definition": "stop_at_first_levelup",
    }

    ok, msg = gate._verdict(cur, _baseline())

    assert ok is False
    assert "action metric mismatch" in msg
    assert "actions_to_first_levelup" in msg
    assert "actions" in msg


def test_req_arc_fcp_4518_dashboard_row_reports_uniform_lever_delta():
    """REQ-ARC-FCP-4518: every A-lever gets the same median/core/bonus row."""
    cur = _cur({"lp85": 7000, "m0r0": 7100, "sp80": 7200, "vc33": 7300, "ft09": 7400})

    row = gate.dashboard_row(cur, _baseline(), lever="A1_frame_change_prune")

    assert row["lever"] == "A1_frame_change_prune"
    assert row["metric_action_field"] == "actions"
    assert row["median_actions_on_core"] == 7150.0
    assert row["baseline_median_actions_on_core"] == 7760.0
    assert row["actions_saved_vs_baseline"] == 610.0
    assert row["core_solves_preserved"] is True
    assert row["bonus_solves"] == ["ft09"]
    assert row["verdict_pass"] is True


def test_req_arc_fcp_4518_positive_control_proves_metric_can_detect_reduction():
    """REQ-ARC-FCP-4518: the canonical harness has a positive-control self-test."""
    control = gate.positive_control(_baseline())

    assert control["passed"] is True
    assert control["dashboard_row"]["core_solves_preserved"] is True
    assert control["dashboard_row"]["median_actions_on_core"] < 7760.0
    assert "IMPROVED" in control["dashboard_row"]["verdict"]


def test_req_arc_fcp_4518_headroom_budget_selection_uses_solved_set_plateau():
    """REQ-ARC-FCP-4518: B* is the first budget stable against its 1.5x comparison."""
    by_budget = {
        8000: {"solved_games": ["lp85", "m0r0", "sp80", "vc33"]},
        12000: {"solved_games": ["lp85", "m0r0", "sp80", "vc33"]},
        18000: {"solved_games": ["lp85", "m0r0", "sp80", "vc33", "ft09"]},
        24000: {"solved_games": ["lp85", "m0r0", "sp80", "vc33", "ft09"]},
        36000: {"solved_games": ["lp85", "m0r0", "sp80", "vc33", "ft09"]},
    }

    selected, rows = gate.select_headroom_budget(by_budget, candidates=(8000, 12000, 24000))

    assert selected == 8000
    assert rows[0]["budget"] == 8000
    assert rows[0]["comparison_budget"] == 12000
    assert rows[0]["stable_vs_1_5x"] is True
    assert rows[1]["stable_vs_1_5x"] is False


def test_req_arc_fcp_4518_cli_exposes_lever_and_canonical_budget_default():
    """REQ-ARC-FCP-4518: `--lever` attributes rows without changing the metric."""
    parser = gate._build_parser()

    args = parser.parse_args(["--lever", "A6"])

    assert args.lever == "A6"
    assert args.budget == gate.DEFAULT_BUDGET


def _efficiency_baseline() -> dict:
    baseline = _baseline()
    baseline["efficiency_by_game"] = {
        "lp85": gate.CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR,
        "m0r0": 0.0003,
        "sp80": 0.0001,
        "vc33": 0.0001,
    }
    baseline["core_efficiency"] = 2.0074
    return baseline


def test_req_arc_fcp_4527_measure_tracks_per_game_score_and_nav_fields(monkeypatch):
    """REQ-ARC-FCP-4527: measure() promotes score and nav diagnostics into CI-guarded maps."""

    def fake_measure_game(game: str, _policy: str, _budget: int, _cap: int) -> dict:
        solved = game in gate.CANONICAL_CORE_GAMES
        return {
            "game": game,
            "timed_out": False,
            "solved": solved,
            "actions": 100 if solved else 999,
            "efficiency": 1.25 if game == "lp85" else (0.25 if solved else 0.0),
            "levels": 2 if game == "lp85" else (1 if solved else 0),
            "deepest_level_reached": 2 if game == "lp85" else (1 if solved else 0),
            "reset_replay_steps": 7 if game == "lp85" else 3,
            "forward_walk_hit_rate": 0.5 if game == "lp85" else 0.0,
        }

    monkeypatch.setattr(gate, "_measure_game", fake_measure_game)

    measurement = gate.measure("fixture", budget=123, cap=4)

    assert measurement["deepest_level_by_game"]["lp85"] == 2
    assert measurement["per_level_efficiency_by_game"]["lp85"] == 1.25
    assert measurement["navigation_by_game"]["lp85"] == {
        "reset_replay_steps": 7,
        "forward_walk_hit_rate": 0.5,
    }
    lp85 = next(row for row in measurement["per_game"] if row["game"] == "lp85")
    assert lp85["deepest_level_reached"] == 2
    assert lp85["per_level_efficiency"] == 1.25
    assert lp85["reset_replay_steps"] == 7
    assert lp85["forward_walk_hit_rate"] == 0.5


def test_req_arc_fcp_4527_nav_regression_warns_without_changing_core_verdict():
    """SCENARIO-ARC-FCP-4527: higher replay tax at equal actions is WARN, not a score fail."""

    base = _baseline()
    base["navigation_by_game"] = {
        game: {"reset_replay_steps": 10, "forward_walk_hit_rate": 0.25}
        for game in gate.CANONICAL_CORE_GAMES
    }
    cur = _cur({"lp85": 7792, "m0r0": 7789, "sp80": 7724, "vc33": 7731})
    cur["navigation_by_game"] = {
        game: {"reset_replay_steps": 10, "forward_walk_hit_rate": 0.25}
        for game in gate.CANONICAL_CORE_GAMES
    }
    cur["navigation_by_game"]["lp85"] = {
        "reset_replay_steps": 25,
        "forward_walk_hit_rate": 0.10,
    }

    row = gate.dashboard_row(cur, base, lever="nav_regression_fixture")

    assert row["verdict_pass"] is True
    assert "non-inferior" in row["verdict"]
    assert row["nav_regression_warning"].startswith("WARN:")
    assert "lp85" in row["nav_regression_warning"]
    assert row["navigation_by_game"]["lp85"]["reset_replay_steps"] == 25


def test_req_arc_fcp_4527_validate_baseline_rejects_deflated_lp85_efficiency():
    """REQ-ARC-FCP-4527: canonical baseline validation pins the lp85 efficiency floor."""

    assert gate.validate_canonical_baseline(_efficiency_baseline())["ok"] is True

    deflated = _efficiency_baseline()
    deflated["efficiency_by_game"] = dict(deflated["efficiency_by_game"])
    deflated["efficiency_by_game"]["lp85"] = gate.CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR - 0.0001

    validation = gate.validate_canonical_baseline(deflated)

    assert validation["ok"] is False
    assert any("lp85" in error and "floor" in error for error in validation["errors"])


def test_req_arc_fcp_4527_update_baseline_rejects_invalid_candidate(monkeypatch, tmp_path, capsys):
    """REQ-ARC-FCP-4527: --update-baseline validates before persisting a new baseline."""

    candidate = _efficiency_baseline()
    candidate["efficiency_by_game"] = dict(candidate["efficiency_by_game"])
    candidate["efficiency_by_game"]["lp85"] = 1.0
    candidate["per_game"] = [
        {
            "game": game,
            "solved": True,
            "actions": actions,
            "efficiency": candidate["efficiency_by_game"][game],
        }
        for game, actions in candidate["actions_by_game"].items()
    ]
    target = tmp_path / "arc-submission-baseline.json"

    monkeypatch.setattr(gate, "BASELINE", target)
    monkeypatch.setattr(gate, "measure", lambda _policy, _budget, _cap: candidate)
    monkeypatch.setattr(sys, "argv", ["arc_local_submission_gate.py", "--update-baseline"])

    rc = gate.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert not target.exists()
    assert "canonical baseline guard failed" in captured.out


def test_req_arc_fcp_4527_update_baseline_accepts_bonus_solves_outside_core(
    monkeypatch, tmp_path, capsys
):
    """2026-07-15 incident: a real measurement that solves the 4 CORE games PLUS bonus games
    outside CANONICAL_CORE_GAMES (e.g. cd82/ft09/su15) must NOT be refused by the cherry-pick
    guard. Before the fix, `--update-baseline` persisted the FULL solved_games (core + bonus) as
    the new baseline's core-identity field, and validate_canonical_baseline correctly rejected
    that (7 games != the canonical 4) -- but this then blocked a genuinely improved measurement
    from ever becoming the new baseline. The fix scopes the persisted solved_games to the
    intersection with CANONICAL_CORE_GAMES, so bonus solves are recorded elsewhere but never
    block the update."""

    candidate = _efficiency_baseline()
    candidate["solved_games"] = sorted({*gate.CANONICAL_CORE_GAMES, "cd82", "ft09", "su15"})
    candidate["solved_count"] = len(candidate["solved_games"])
    candidate["actions_by_game"] = {
        **candidate["actions_by_game"],
        "cd82": 7800,
        "ft09": 7801,
        "su15": 7802,
    }
    target = tmp_path / "arc-submission-baseline.json"

    monkeypatch.setattr(gate, "BASELINE", target)
    monkeypatch.setattr(gate, "measure", lambda _policy, _budget, _cap: candidate)
    monkeypatch.setattr(sys, "argv", ["arc_local_submission_gate.py", "--update-baseline"])

    rc = gate.main()
    captured = capsys.readouterr()

    assert rc == 0, captured.out
    assert target.exists()
    persisted = json.loads(target.read_text())
    assert sorted(persisted["solved_games"]) == sorted(gate.CANONICAL_CORE_GAMES)
    assert "baseline UPDATED" in captured.out
