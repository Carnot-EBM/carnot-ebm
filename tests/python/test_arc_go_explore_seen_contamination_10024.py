"""REQ-ARC-WMTE-10024: archive replays do not create organic sightings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic import arc_competition_agent as comp
from carnot.agentic.arc_go_explore import GoExploreReplayArchive, _coarse_cell


REPO = Path(__file__).resolve().parents[2]
PREFIX = [{"action": 2, "data": None}, {"action": 3, "data": None}]


def _frame(color: int, actions: tuple[int, ...] = (2, 3)) -> SimpleNamespace:
    return SimpleNamespace(
        frame=np.asarray([[color, 0], [0, 0]], dtype=np.int16),
        levels_completed=0,
        available_actions=list(actions),
    )


def _explorer(archive: GoExploreReplayArchive | None, **kwargs: object) -> comp.StepwiseExplorer:
    return comp.StepwiseExplorer(
        go_explore_archive=archive,
        auto_hud_mask=False,
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        **kwargs,
    )


def _seeded_replay(archive: GoExploreReplayArchive) -> tuple[comp.StepwiseExplorer, list]:
    root, step1, target, organic = [_frame(color) for color in (1, 2, 3, 4)]
    if not archive._cells:
        archive.observe(target, PREFIX)
    explorer = _explorer(archive)
    explorer._ingest(root)
    assert explorer.root is not None
    explorer.graph[explorer.root]["untested"] = []
    return explorer, [root, step1, target, organic]


def test_scenario_arc_wmte_10024_two_step_replay_landing_and_next_organic() -> None:
    """SCENARIO-ARC-WMTE-10024-REPLAY-LANDING: last pop precedes landing ingest."""

    archive = GoExploreReplayArchive(bins=2)
    explorer, (root, step1, target, organic) = _seeded_replay(archive)
    assert explorer.next_move([], root) == ("RESET", None)
    assert explorer._go_explore_replay_active is True
    baseline = archive.diagnostics()["observations"]

    # _ingest runs before the next _serve: pending has 2, then 1, then 0 items.
    # The empty queue at target must clear the flag only after suppressing that frame.
    for frame, expected_action, remaining in (
        (root, (2, None), 1),
        (step1, (3, None), 0),
        (target, (2, None), 0),
    ):
        assert explorer.next_move([], frame) == expected_action
        assert len(explorer.pending) == remaining
        assert archive.diagnostics()["observations"] == baseline
    assert explorer._go_explore_replay_active is False
    assert archive._cells[_coarse_cell(target.frame, 0, bins=2)]["seen"] == 1

    explorer.next_move([], organic)
    assert archive.diagnostics()["observations"] == baseline + 1
    assert _coarse_cell(organic.frame, 0, bins=2) in archive._cells


def test_scenario_arc_wmte_10024_repeated_returns_do_not_inflate_seen() -> None:
    """SCENARIO-ARC-WMTE-10024-ORGANIC-SEEN: own replay cannot add momentum."""

    archive = GoExploreReplayArchive(bins=2)
    target = _frame(3)
    archive.observe(target, PREFIX)
    target_key = _coarse_cell(target.frame, 0, bins=2)
    for _ in range(3):
        explorer, (root, step1, target, _) = _seeded_replay(archive)
        assert explorer.next_move([], root) == ("RESET", None)
        assert explorer.next_move([], root) == (2, None)
        assert explorer.next_move([], step1) == (3, None)
        explorer.next_move([], target)
        assert explorer._go_explore_replay_active is False
    assert archive.diagnostics()["selected_prefixes"] == 3
    assert archive._cells[target_key]["seen"] == 1


def test_scenario_arc_wmte_10024_ordinary_frontier_pending_still_observed() -> None:
    """SCENARIO-ARC-WMTE-10024-NONREPLAY-AND-OFF: frontier queue stays organic."""

    archive = GoExploreReplayArchive(bins=2)
    explorer = _explorer(archive, search_mode="best_first")
    assert explorer.next_move([], _frame(1)) == (2, None)
    assert explorer.next_move([], _frame(2)) == ("RESET", None)
    assert explorer._prov_branch == "frontier.navigate"
    assert len(explorer.pending) == 1
    assert explorer._go_explore_replay_active is False
    assert archive.diagnostics()["observations"] == 2

    assert explorer.next_move([], _frame(1)) == (3, None)
    assert explorer._prov_branch == "pending_drain"
    assert explorer.pending == []
    assert archive.diagnostics()["observations"] == 3
    explorer.next_move([], _frame(3))
    assert explorer._go_explore_replay_active is False
    assert archive.diagnostics()["observations"] == 4


def _main_module() -> ModuleType:
    source = subprocess.run(
        ["git", "show", "main:python/carnot/agentic/arc_competition_agent.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    module = ModuleType("carnot.agentic._main_arc_competition_agent_10024")
    module.__file__ = str(REPO / "python/carnot/agentic/arc_competition_agent.py")
    module.__package__ = "carnot.agentic"
    sys.modules[module.__name__] = module
    exec(compile(source, "main:arc_competition_agent.py", "exec"), module.__dict__)
    return module


def _decision_bytes(explorer: Any, actions: list[tuple]) -> bytes:
    graph = explorer.graph
    state = {
        "actions": actions,
        "root": explorer.root,
        "cur": explorer.cur,
        "pending": explorer.pending,
        "awaiting_action": (explorer.awaiting or {}).get("action"),
        "branch": explorer._prov_branch,
        "serve_kind": explorer._prov_serve_kind,
        "graph": {
            key: {"path": node["path"], "untested": node["untested"]}
            for key, node in sorted(graph.items())
        },
    }
    return json.dumps(state, sort_keys=True, separators=(",", ":")).encode()


def test_scenario_arc_wmte_10024_archive_off_matches_main_100_fixtures() -> None:
    """SCENARIO-ARC-WMTE-10024-NONREPLAY-AND-OFF: 100 byte-parity fixtures."""

    main = _main_module()
    assert comp.SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED is False
    assert main.SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED is False
    for seed in range(100):
        kwargs = {
            "go_explore_archive": None,
            "auto_hud_mask": False,
            "online_discriminative": False,
            "navigation_cost_tiebreak": False,
            "search_mode": "best_first" if seed % 2 else "depth_first_ride",
        }
        current = comp.StepwiseExplorer(**kwargs)
        original = main.StepwiseExplorer(**kwargs)
        actions = (2,) if seed % 3 == 0 else (2, 3)
        first = _frame(1 + seed, actions)
        second = _frame(102 + seed, actions)
        current_actions = [current.next_move([], first), current.next_move([], second)]
        original_actions = [original.next_move([], first), original.next_move([], second)]
        assert current._go_explore_replay_active is False
        assert _decision_bytes(current, current_actions) == _decision_bytes(
            original, original_actions
        ), seed
