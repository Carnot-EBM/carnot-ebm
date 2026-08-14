"""REQ-ARC-BENCH-6267 / REQ-ARC-FLAG-LEDGER-6268.

The ARC loop had generation and no selection: 101 `CARNOT_ARC_*` flags, 13 of 16 recent tasks
ending `ready_no_solve_claim` or `default_off`, and three "holdout" experiments whose metric sets
share zero keys. `arc_bench.py` produces one comparable number; `arc_flag_ledger.py` selects on it.

These tests cover the DECISION logic, not the arcade. Running a game needs the offline arcade and
takes seconds per game; the part that can silently be wrong is the rule that decides what turns
on, so that is what is pinned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_bench  # noqa: E402
import arc_flag_ledger as ledger  # noqa: E402


def _row(game, levels, actions, error=None):
    return {
        "game": game,
        "levels_cleared": levels,
        "actions_spent": actions,
        "solution_len": 0,
        "wall_s": 0.1,
        "error": error,
        "adapter_used": False,
    }


def _report(rows):
    return {"per_game_rows": rows, **arc_bench.aggregate(rows)}


# --------------------------------------------------------------------------- aggregate


def test_aggregate_reports_errors_instead_of_dropping_them():
    """A benchmark that silently drops its hard cases reports a rising average while getting worse."""
    agg = arc_bench.aggregate([_row("a", 1, 100), _row("b", 0, 500), _row("c", 0, 0, "boom")])

    assert agg["games_run"] == 3
    assert agg["games_errored"] == 1
    assert agg["total_levels_cleared"] == 1
    assert agg["clear_rate"] == 0.5, "the errored game is excluded from the rate, not counted clean"


def test_actions_per_level_is_none_not_a_sentinel_when_nothing_cleared():
    """ "Infinitely inefficient" and "did not run" are different facts.

    A sentinel like 0 or a huge float would be averaged into a later comparison as if it meant
    something.
    """
    agg = arc_bench.aggregate([_row("a", 0, 900), _row("b", 0, 100)])

    assert agg["total_levels_cleared"] == 0
    assert agg["actions_per_level_cleared"] is None


# --------------------------------------------------------------------------- rotation


def test_rotation_advances_so_successive_runs_cover_the_roster(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")
    monkeypatch.setattr(arc_bench, "ROTATION", tmp_path / "rot.json")
    games = [f"g{i}" for i in range(10)]

    first = arc_bench.select(games, 4)
    second = arc_bench.select(games, 4)

    assert first == ["g0", "g1", "g2", "g3"]
    assert second == ["g4", "g5", "g6", "g7"], "a fixed offset would re-measure the same head slice"


def test_rotation_resets_when_the_roster_changes(tmp_path, monkeypatch):
    """The QA-audit `units_signature` lesson, applied here.

    An offset persisted against a roster that has since changed silently skips the newly-added
    games -- sometimes for many runs. Re-measuring some games is cheap; a new game never being
    benchmarked is the failure that matters, so a changed roster resolves the offset to 0.
    """
    monkeypatch.setattr(arc_bench, "ROTATION", tmp_path / "rot.json")
    arc_bench.select([f"g{i}" for i in range(10)], 4)  # offset -> 4

    picked = arc_bench.select([f"h{i}" for i in range(10)], 4)  # different roster

    assert picked == ["h0", "h1", "h2", "h3"], "a changed roster must restart from the beginning"


def test_rotation_survives_a_corrupt_state_file(tmp_path, monkeypatch):
    rot = tmp_path / "rot.json"
    rot.write_text("{not json")
    monkeypatch.setattr(arc_bench, "ROTATION", rot)

    assert arc_bench.select([f"g{i}" for i in range(6)], 2) == ["g0", "g1"]


# --------------------------------------------------------------------------- the regression gate


def test_promotion_is_refused_when_any_single_game_regresses():
    """THE load-bearing test. An aggregate win that costs one game is the ka59 failure repeating.

    The ARC engine store overwrote unconditionally and destroyed ka59 from 1.0 to 0.0; retention
    was worth p=4.9e-5. Here the arm clears three NEW games and loses one. Total levels go UP. It
    must still be refused, and the lost game must be named.
    """
    base = _report([_row("keep", 1, 100), _row("a", 0, 100), _row("b", 0, 100), _row("c", 0, 100)])
    arm = _report([_row("keep", 0, 100), _row("a", 1, 100), _row("b", 1, 100), _row("c", 1, 100)])

    cmp = ledger.compare(base, arm)
    ok, why = ledger.verdict(cmp)

    assert cmp["arm_total_levels"] > cmp["baseline_total_levels"], "the aggregate really is better"
    assert not ok, "and it is still refused"
    assert "REFUSED" in why and "keep" in why, "the game that broke must be named"


def test_promotion_needs_more_than_one_improved_game():
    base = _report([_row("a", 0, 100), _row("b", 0, 100)])
    arm = _report([_row("a", 1, 100), _row("b", 0, 100)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert not ok
    assert "HOLD" in why and "coincidence" in why


def test_two_improved_games_and_no_regression_promotes():
    base = _report([_row("a", 0, 100), _row("b", 0, 100), _row("c", 1, 100)])
    arm = _report([_row("a", 1, 100), _row("b", 1, 100), _row("c", 1, 100)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert ok and "PROMOTE" in why


def test_same_levels_but_strictly_cheaper_promotes_because_the_score_squares_efficiency():
    base = _report([_row("a", 1, 9000), _row("b", 1, 9000)])
    arm = _report([_row("a", 1, 400), _row("b", 1, 500)])

    ok, why = ledger.verdict(ledger.compare(base, arm))

    assert ok and "fewer actions" in why


def test_cheaper_on_one_game_and_costlier_on_another_does_not_promote():
    """Mixed efficiency is not an improvement, it is a trade nobody authorised."""
    base = _report([_row("a", 1, 9000), _row("b", 1, 100)])
    arm = _report([_row("a", 1, 400), _row("b", 1, 5000)])

    ok, _ = ledger.verdict(ledger.compare(base, arm))

    assert not ok


def test_a_game_that_errored_on_either_side_is_excluded_from_the_comparison():
    """An error is missing data. Reading it as a regression would refuse promotions for a flake.

    It is still visible: `aggregate` counts it in `games_errored`, so it cannot be lost entirely.
    """
    base = _report([_row("a", 1, 100), _row("b", 0, 100), _row("c", 0, 100)])
    arm = _report([_row("a", 0, 0, "timeout"), _row("b", 1, 100), _row("c", 1, 100)])

    cmp = ledger.compare(base, arm)
    ok, _ = ledger.verdict(cmp)

    assert cmp["levels_regressed"] == [], "the errored game is not a regression"
    assert ok, "the two genuine improvements still count"


# --------------------------------------------------------------------------- ledger safety


def test_promote_refuses_a_flag_that_was_never_measured(tmp_path, monkeypatch, capsys):
    """Promotion reads recorded evidence and never measures implicitly.

    A promote that quietly runs its own measurement is a promote nobody reviewed.
    """
    monkeypatch.setattr(ledger, "LEDGER", tmp_path / "l.yaml")

    rc = ledger.cmd_promote("CARNOT_ARC_NEVER_TESTED")

    assert rc == 1
    assert "no ledger entry" in capsys.readouterr().out


def test_promote_refuses_a_flag_whose_last_measurement_said_no(tmp_path, monkeypatch, capsys):
    import yaml

    lp = tmp_path / "l.yaml"
    lp.write_text(
        yaml.safe_dump(
            {
                "flags": {
                    "CARNOT_ARC_X": {
                        "state": "unevaluated",
                        "promotable": False,
                        "evidence": [{"verdict": "REFUSED: regressed on ['ka59']."}],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(ledger, "LEDGER", lp)

    rc = ledger.cmd_promote("CARNOT_ARC_X")

    assert rc == 1
    out = capsys.readouterr().out
    assert "REFUSING" in out and "ka59" in out, "must repeat the reason, not just decline"


def test_a_corrupt_ledger_raises_instead_of_starting_a_fresh_one(tmp_path, monkeypatch):
    """Silently replacing an unreadable ledger would erase every promotion and its evidence."""
    import pytest

    lp = tmp_path / "l.yaml"
    lp.write_text("{[: not yaml at all\n  - ][")
    monkeypatch.setattr(ledger, "LEDGER", lp)

    with pytest.raises(SystemExit):
        ledger.load()


def test_discover_reads_flags_from_source_not_a_hardcoded_list():
    """A hand-maintained watch-list drifts narrower than the thing it watches.

    That is this project's most-repeated defect, so the flag set is derived from the agent source.
    """
    flags = ledger.discover_flags()

    assert len(flags) > 50, f"expected the real agent flag surface, got {len(flags)}"
    assert all(f.startswith("CARNOT_ARC_") for f in flags)


def test_bench_report_always_carries_the_held_out_caveat():
    """The benchmark runs on PUBLIC games. Nobody may quote it as a hidden-game result."""
    assert "PUBLIC" in arc_bench.CAVEAT
    assert "not a hidden-game result" in arc_bench.CAVEAT


def test_roster_comes_from_the_registry_and_holds_the_public_games():
    games = arc_bench.roster()

    assert len(games) >= 25
    for known in ("ls20", "ka59", "dc22", "vc33"):
        assert known in games


def test_run_game_reports_an_error_row_rather_than_raising(monkeypatch):
    """A benchmark that dies on one game loses the whole sweep, including the games that passed."""
    import carnot.agentic.arc_solver_kit as kit

    def boom():
        raise RuntimeError("arcade unavailable")

    monkeypatch.setattr(kit, "offline_arcade", boom)

    row = arc_bench.run_game("ls20")

    assert row["error"] and "arcade unavailable" in row["error"]
    assert row["levels_cleared"] == 0
    assert row["game"] == "ls20"


def test_measured_report_round_trips_as_json():
    """The report is consumed by another process, so it must serialise."""
    rows = [_row("a", 1, 100), _row("b", 0, 200, "err")]
    report = {"schema": arc_bench.SCHEMA, "per_game_rows": rows, **arc_bench.aggregate(rows)}

    assert json.loads(json.dumps(report))["schema"] == "carnot.arc_bench.v1"
