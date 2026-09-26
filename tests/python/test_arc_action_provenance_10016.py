"""REQ-ARC-WMTE-10016: resulting-frame provenance on the scored policy."""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_competition_agent import E3AgentPolicy


class Frame:
    def __init__(self, level: int = 0) -> None:
        self.frame = [np.zeros((8, 8), dtype=int).tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _drive(monkeypatch, tmp_path, *, record: bool):
    """SCENARIO-ARC-WMTE-10016-IDENTITY: replay identical scripted frames."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE_DIR", str(tmp_path))
    if record:
        monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    else:
        monkeypatch.delenv("CARNOT_ARC_ACTION_PROVENANCE", raising=False)
    policy = E3AgentPolicy("unknown", proposer=None, explore_budget=100)
    latest = None
    frames = []
    actions = []
    for i in range(16):
        move = policy.next_move(frames, latest)
        actions.append(move)
        latest = Frame(1 if i >= 9 else 0)
        frames.append(latest)
    policy.is_done(frames, latest)
    recorder = policy.action_provenance()
    return actions, [] if recorder is None else recorder.rows


def test_provenance_passive_and_real_counter(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-10016-PROVENANCE: only real frames credit progress."""
    off, _ = _drive(monkeypatch, tmp_path, record=False)
    on, rows = _drive(monkeypatch, tmp_path, record=True)
    assert off == on
    assert len(rows) == len(on)
    assert all(
        "top_branch" in row
        and "phase_before" in row
        and "explorer_branch" in row
        and "explorer_serve_kind" in row
        and "plan_step" in row
        for row in rows
    )
    assert [row["levels_completed"] for row in rows[:9]] == [0] * 9
    assert rows[9]["levels_completed"] == 1
    assert rows[-1]["levels_completed"] == 1
    assert all(row["plan_step"] is False for row in rows)


def test_plan_action_is_marked_as_plan_step(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-10016-PROVENANCE: a consumed plan is explicit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE", "1")
    monkeypatch.setenv("CARNOT_ARC_ACTION_PROVENANCE_DIR", str(tmp_path))
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    policy = E3AgentPolicy("unknown", proposer=None, explore_budget=100)
    policy.plan = [{"action": 1, "data": None}]
    policy.pi = 0
    policy.phase = "execute"
    policy.induced = True
    frame = Frame()
    assert policy.next_move([frame], frame) == (1, None)
    row = policy.action_provenance().rows[-1]
    assert row["top_branch"] == "execute.plan_step"
    assert row["plan_step"] is True
    assert row["levels_completed"] is None
    policy.observe_action_outcome(Frame(1))
    assert row["levels_completed"] == 1
    assert row["level_after"] == 1
    policy.observe_action_outcome(Frame(2))
    assert row["levels_completed"] == 1
