"""Spec: REQ-ARC-WMTE-6850, SCENARIO-ARC-WMTE-6850-A, SCENARIO-ARC-WMTE-6850-B

A multi-game ARC eval banks each game as it finishes.

INCIDENT 2026-09-01. `scripts/arc_leaderboard_eval.py` wrote its one artifact after the whole
per-game loop. A two-game run at roughly 5.4 hours per game (measured on cd82, 19,543s) therefore
produced nothing for about 11 hours, and a crash at hour 10 lost both games -- including every
trajectory-supervisor receipt they had accrued, which is the evidence such a run exists to
gather. The death receipt (REQ-INFRA-6830) records the stage and the game names, not arm
outcomes, so it does not cover the gap.

The tests drive the real `main()` with a stubbed `run_game`, because the defect is in the
CONTROL FLOW around the loop; a test of a pure helper would not have seen it. The writer itself
is stubbed so nothing here can touch `results/`, which is evidence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as ale  # noqa: E402


def _fake_row(game: str) -> dict[str, Any]:
    return {
        "game": game,
        "levels": 1,
        "reached": 1,
        "actions": 100,
        "efficiency": 0.5,
        "gap": None,
        "reset_replay_steps": 0,
        "forward_walk_hit_rate": 0.0,
    }


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, Any]]]:
    """Record every write in order, and let none of them reach the real results directory."""
    captured: list[tuple[Path, dict[str, Any]]] = []
    monkeypatch.setattr(
        ale, "_write_json_atomic", lambda out, payload: captured.append((out, json.loads(payload)))
    )
    monkeypatch.setattr(ale, "_build_policy", lambda kind, game: object())
    return captured


@pytest.fixture
def unlinked(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record partial-file removals without deleting anything real."""
    gone: list[Path] = []
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: gone.append(self))
    return gone


def _expected_partial(tag: str) -> Path:
    return ale.REPO / "results" / "arc_leaderboard_eval_runs" / f"{tag}-{os.getpid()}.partial.json"


def test_each_game_is_banked_before_the_next_one_starts(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-A: when game two begins, game one is already on disk."""
    order: list[str] = []

    def _run_game(game: str, policy: Any, **kw: Any) -> dict[str, Any]:
        if game == "gameB":
            banked = [p for _, p in writes if p.get("complete") is False]
            assert banked, "game A was not banked before game B started"
            assert banked[-1]["games_completed"] == 1
            assert banked[-1]["games_planned"] == 2
            assert [r["game"] for r in banked[-1]["per_game"]] == ["gameA"]
        order.append(game)
        return _fake_row(game)

    monkeypatch.setattr(ale, "run_game", _run_game)
    # --only FILTERS the claimed set, so ids absent from CLAIMED select nothing and the run
    # succeeds vacuously in seconds. That exact trap wasted a real batch on 2026-08-31.
    monkeypatch.setattr(ale, "CLAIMED", ["gameA", "gameB"])
    monkeypatch.setattr(sys, "argv", ["arc_leaderboard_eval.py", "--only", "gameA,gameB"])
    assert ale.main() == 0
    assert order == ["gameA", "gameB"]


def test_a_crash_between_games_leaves_the_finished_game_banked(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-B: the case the change exists for. Kill it mid-run; game one survives."""

    def _run_game(game: str, policy: Any, **kw: Any) -> dict[str, Any]:
        if game == "gameB":
            raise KeyboardInterrupt("simulated death at hour 10")
        return _fake_row(game)

    monkeypatch.setattr(ale, "run_game", _run_game)
    # --only FILTERS the claimed set, so ids absent from CLAIMED select nothing and the run
    # succeeds vacuously in seconds. That exact trap wasted a real batch on 2026-08-31.
    monkeypatch.setattr(ale, "CLAIMED", ["gameA", "gameB"])
    monkeypatch.setattr(sys, "argv", ["arc_leaderboard_eval.py", "--only", "gameA,gameB"])
    with pytest.raises(KeyboardInterrupt):
        ale.main()

    assert len(writes) == 1, "the finished game was lost"
    out, payload = writes[0]
    assert out == _expected_partial("gameA-gameB")
    assert payload["games_completed"] == 1
    assert payload["live_levels"] == 1
    # A partial must never read as a finished result. Its verdict carries no terminal prefix,
    # so the reconciler's classifier cannot mistake it for a completed run.
    assert payload["complete"] is False
    assert not payload["honest_verdict"].startswith(("complete", "success", "passed", "shipped"))


def test_a_partial_never_lands_on_the_tracked_sweep_file(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial on the tracked path would replace a full sweep with an in-progress one.

    That is the record loss commit f2b82c89a6's 25-game sweep suffered. A FULL sweep (no
    --only) must still keep its partials in the gitignored run-scoped directory.
    """
    monkeypatch.setattr(ale, "run_game", lambda game, policy, **kw: _fake_row(game))
    monkeypatch.setattr(ale, "CLAIMED", ["gameA", "gameB"])
    monkeypatch.setattr(sys, "argv", ["arc_leaderboard_eval.py"])
    ale.main()

    tracked = ale.REPO / "results" / "arc_leaderboard_eval.json"
    partials = [(out, p) for out, p in writes if p.get("complete") is False]
    assert partials, "no partial was written"
    for out, _ in partials:
        assert out != tracked
        assert out.parent.name == "arc_leaderboard_eval_runs"
    # the FINAL write is the one allowed to claim the tracked path on a full sweep
    assert writes[-1][0] == tracked
    assert writes[-1][1]["complete"] is True


def test_the_partial_is_removed_once_the_real_record_exists(
    writes: list[tuple[Path, dict[str, Any]]],
    unlinked: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash insurance, not a second record. A stale partial beside a finished run misleads."""
    monkeypatch.setattr(ale, "run_game", lambda game, policy, **kw: _fake_row(game))
    # --only FILTERS the claimed set, so ids absent from CLAIMED select nothing and the run
    # succeeds vacuously in seconds. That exact trap wasted a real batch on 2026-08-31.
    monkeypatch.setattr(ale, "CLAIMED", ["gameA", "gameB"])
    monkeypatch.setattr(sys, "argv", ["arc_leaderboard_eval.py", "--only", "gameA,gameB"])
    assert ale.main() == 0
    assert _expected_partial("gameA-gameB") in unlinked
