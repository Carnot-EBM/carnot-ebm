"""REQ-ARC-WMTE-7735 and REQ-REPORT-7735 mechanism checks."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic.arc_go_explore import GoExploreReplayArchive, _coarse_cell
from carnot.agentic.arc_competition_agent import make_carnot_agent
from carnot.experiment_7735_v673_arc_organic_visits import cold_reduce, fixture_episode


def frame(color: int) -> SimpleNamespace:
    return SimpleNamespace(
        frame=np.asarray([[color, 0], [0, 0]]),
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5, 6],
        state="NOT_FINISHED",
    )


def test_scenario_arc_wmte_7735_provenance_and_reload() -> None:
    archive = GoExploreReplayArchive(bins=2, selector="organic_visits")
    path = [{"action": 2, "data": None}]
    key = _coarse_cell(frame(3).frame, 0, bins=2)
    archive.observe(frame(3), path, provenance="organic")
    for _ in range(3):
        archive.observe(frame(3), path, provenance="reset")
        archive.observe(frame(3), path, provenance="replay")
    assert archive._cells[key]["seen"] == 7
    assert archive._cells[key]["organic_seen"] == 1
    assert archive._cells[key]["reset_seen"] == 3
    assert archive._cells[key]["replay_seen"] == 3
    restored = GoExploreReplayArchive.from_snapshot(archive.snapshot())
    assert restored.snapshot() == archive.snapshot()
    with pytest.raises(ValueError, match="unsupported_archive_snapshot"):
        GoExploreReplayArchive.from_snapshot({"version": 0})


def test_scenario_arc_wmte_7735_selector_uses_organic_only() -> None:
    archive = GoExploreReplayArchive(bins=2, selector="organic_visits")
    for color, visits, organic, replay in ((2, 2, 5, 0), (3, 2, 1, 20)):
        path = [{"action": color, "data": None}]
        archive.observe(frame(color), path)
        for _ in range(organic - 1):
            archive.observe(frame(color), path)
        for _ in range(replay):
            archive.observe(frame(color), path, provenance="replay")
        archive._cells[_coarse_cell(frame(color).frame, 0, bins=2)]["visits"] = visits
    assert archive.select_prefix() == [{"action": 2, "data": None}]


def test_scenario_arc_wmte_7735_scored_wrapper_opt_in_and_off_rng_parity() -> None:
    class Base:
        def __init__(self, game_id: str = "unknown") -> None:
            self.game_id = game_id

    default = make_carnot_agent(Base, proposer=None)()
    explicit_off = make_carnot_agent(Base, proposer=None, organic_visits=False)()
    assert default._policy.explorer.go_explore_archive is None
    assert explicit_off._policy.explorer.go_explore_archive is None
    random.seed(7735)
    before = random.getstate()
    for index in range(300):
        current = frame(index % 7 + 1)
        assert default._policy.explorer.next_move(
            [], current
        ) == explicit_off._policy.explorer.next_move([], current)
        assert random.getstate() == before
    enabled = make_carnot_agent(Base, proposer=None, organic_visits=True)()
    assert enabled._policy.explorer.go_explore_archive is not None
    assert enabled._policy.explorer.go_explore_archive.selector is not None


def test_scenario_report_7735_cold_reduction_rejects_changed_count() -> None:
    row = fixture_episode("fixture", 0, "organic")
    assert cold_reduce([row])["actions"] == row["actions_charged"]
    bad = dict(row, actions_charged=row["actions_charged"] + 1)
    with pytest.raises(ValueError, match="raw_action_count_mismatch"):
        cold_reduce([bad])
