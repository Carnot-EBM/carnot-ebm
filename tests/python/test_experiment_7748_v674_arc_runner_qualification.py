"""REQ-ARC-WMTE-7748 and REQ-REPORT-7748 runner qualification."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic.arc_go_explore import _coarse_cell
from carnot.experiment_7748_v674_arc_runner_qualification import (
    QualificationArchive,
    cold_reduce,
    make_agent_factory,
    run_qualification_episode,
)
from carnot.experiment_7708_v671_arc_generalization_runner import _FixtureArcade


def frame(color: int) -> SimpleNamespace:
    return SimpleNamespace(
        frame=np.asarray([[color, 0], [0, 0]], dtype=np.int16),
        levels_completed=0,
        available_actions=[1, 2, 3, 4, 5, 6],
        state="NOT_FINISHED",
    )


def test_scenario_arc_wmte_7748_distinct_and_reload() -> None:
    """SCENARIO-ARC-WMTE-7748-DISTINCT: replay sightings change only total rank."""
    selections = {}
    for arm in ("total", "organic"):
        events: list[dict] = []
        archive = QualificationArchive(arm, events, bins=2, selector="organic_visits")
        assert archive.select_prefix() == []
        for color in (2, 3):
            archive.observe(frame(color), [{"action": color, "data": None}])
        for _ in range(4):
            archive.observe(frame(2), [{"action": 2, "data": None}], provenance="replay")
        archive.observe(frame(3), [{"action": 3, "data": None}])
        for color in (2, 3):
            archive._cells[_coarse_cell(frame(color).frame, 0, bins=2)]["visits"] = 2
        restored = QualificationArchive.restore(archive.snapshot(), arm, events)
        assert restored.snapshot() == archive.snapshot()
        assert restored.bins == 2
        selections[arm] = restored.select_prefix()
        assert [event["provenance"] for event in events[:2]] == ["organic", "organic"]
        assert events[-1]["event"] == "selection"
    assert selections["total"] == [{"action": 2, "data": None}]
    assert selections["organic"] == [{"action": 3, "data": None}]
    single = QualificationArchive("total", [], bins=2, selector="organic_visits")
    single.observe(frame(2), [{"action": 2, "data": None}])
    assert single.select_prefix() == [{"action": 2, "data": None}]


def test_scenario_arc_wmte_7748_parity_and_shared_configuration() -> None:
    """SCENARIO-ARC-WMTE-7748-PARITY: off stays exact and enabled arms match."""

    class Base:
        def __init__(self, game_id: str = "fixture") -> None:
            self.game_id = game_id

    off_a = make_agent_factory("off", [])(Base, proposer=None)()
    off_b = make_agent_factory("off", [])(Base, proposer=None)()
    random.seed(7748)
    for index in range(300):
        before = random.getstate()
        current = frame(index % 7 + 1)
        assert off_a._policy.explorer.next_move([], current) == off_b._policy.explorer.next_move(
            [], current
        )
        assert random.getstate() == before
    assert off_a._policy.explorer.go_explore_archive is None
    for arm in ("total", "organic"):
        agent = make_agent_factory(arm, [])(Base, proposer=None)()
        explorer = agent._policy.explorer
        assert explorer._go_explore_organic_visits is True
        assert explorer.go_explore_archive.selector == "organic_visits"
        assert explorer.go_explore_archive.bins == 6
        assert explorer._go_explore_replay_actions_charged == 0
    with pytest.raises(ValueError, match="unknown_arm"):
        make_agent_factory("wrong", [])


def test_scenario_report_7748_custody_fixture_and_cold_reject() -> None:
    """SCENARIO-REPORT-7748-CUSTODY: every row and action remains recomputable."""
    rows = []
    for seed in range(10):
        rows.append(run_qualification_episode("fixture", seed, "off", _FixtureArcade(), 3))
    assert all(row["policy_entry"]["policy_class"] == "E3AgentPolicy" for row in rows)
    assert cold_reduce(rows, len(rows))["completed"] == 10
    with pytest.raises(ValueError, match="schedule_row_count"):
        cold_reduce(rows, 11)
    bad = [dict(row) for row in rows]
    bad[0]["actions_charged"] += 1
    with pytest.raises(ValueError, match="action_count"):
        cold_reduce(bad, len(bad))
    bad[0] = dict(rows[0], new_solve_credit=True)
    with pytest.raises(ValueError, match="duplicate_solve_credit"):
        cold_reduce(bad, len(bad))
