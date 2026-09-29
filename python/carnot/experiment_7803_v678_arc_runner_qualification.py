"""Scored ARC runner controls for REQ-ARC-WMTE-7803 and REQ-REPORT-7803."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic.arc_go_explore import _coarse_cell
from carnot.experiment_7748_v674_arc_runner_qualification import QualificationArchive
from carnot.experiment_7763_v675_arc_runner_qualification import ARMS, GAMES, SEEDS, schedule_rows
from carnot.experiment_7776_v676_arc_runner_qualification import gate_decision


def freeze_panel(path: Path, *, expected_games: Sequence[str] = GAMES) -> dict[str, Any]:
    """Use historical bytes as the panel source so current roadmap edits cannot move it."""
    panel: dict[str, Any] = json.loads(path.read_text())
    expected = {
        "games": list(expected_games),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "controls": {
            "spacing_fresh_actions": 20,
            "replay_cap": 400,
            "max_prefix": 30,
            "cell_bins": 6,
            "max_cells": 256,
        },
        "rows": schedule_rows(),
        "max_actions_charged": 2000,
        "max_seconds_per_episode": 75,
    }
    if any(panel.get(key) != value for key, value in expected.items()):
        raise ValueError("panel_changed")
    return panel


def classify_runner(
    receipts: Sequence[Mapping[str, Any]],
    required: Sequence[str],
    sdk_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    """Open readiness only for every current check and both real three-arm probes."""
    expected = {(game, arm) for game in ("r11l", "cd82") for arm in ARMS}
    observed = [(row.get("game"), row.get("arm")) for row in sdk_rows]
    sdk_ok = len(observed) == len(expected) and set(observed) == expected
    sdk_ok = sdk_ok and all(
        row.get("error") is None
        and row.get("new_solve_credit") is False
        and row.get("counts", {}).get("sdk_transitions", 0) > 0
        and row.get("policy_entry", {}).get("policy_class") == "E3AgentPolicy"
        and all(
            action.get("induction_attempt_count") == 0
            and action.get("actual_observation", {}).get("frame_sha256")
            for action in row.get("actions", [])
        )
        for row in sdk_rows
    )
    return gate_decision(receipts, required, sdk_ok=sdk_ok)


def positive_selector_fixture() -> dict[str, Any]:
    """Expose a replay-only rank change without treating known fixture truth as a solve."""

    def frame(color: int) -> SimpleNamespace:
        return SimpleNamespace(
            frame=np.asarray([[color, 0], [0, 0]], dtype=np.int16),
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5, 6],
            state="NOT_FINISHED",
        )

    evidence: dict[str, Any] = {"off_archive": None}
    for arm in ("total", "organic"):
        events: list[dict[str, Any]] = []
        archive = QualificationArchive(arm, events, bins=2, selector="organic_visits")
        for color in (2, 3):
            archive.observe(frame(color), [{"action": color, "data": None}])
        cell = _coarse_cell(frame(2).frame, 0, bins=2)
        before = archive._cells[cell]["organic_seen"]
        for _ in range(4):
            archive.observe(frame(2), [{"action": 2, "data": None}], provenance="replay")
        after = archive._cells[cell]["organic_seen"]
        replay = archive._cells[cell]["replay_seen"]
        archive.observe(frame(3), [{"action": 3, "data": None}])
        for color in (2, 3):
            archive._cells[_coarse_cell(frame(color).frame, 0, bins=2)]["visits"] = 2
        evidence[f"{arm}_prefix"] = archive.select_prefix()
        evidence[f"{arm}_events"] = events
        evidence["organic_seen_before_replay"] = before
        evidence["organic_seen_after_replay"] = after
        evidence["replay_seen_after_replay"] = replay
    return evidence
