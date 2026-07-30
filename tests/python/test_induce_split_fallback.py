"""Separate-focused-goal-call fallback for the L2 reinduction.

Origin: 2026-06-25 capture diagnostic (proto_l2_capture / proto_l2_fix_finder). On complex real L2
prompts the combined engine+is_level_complete induction fails: the model rambles its analysis into
engine() comments, exhausts max_tokens, and never writes is_level_complete (missing def). A budget
bump does NOT help (the model rambles more). The fix: when the combined call fails, induce each
function in its OWN focused call (engine-only + a focused goal-only call with the win exemplar) so
the engine ramble cannot starve the goal. The combined happy path is unchanged (no regression).

Spec refs: SCENARIO-ARC-WMTE-6044-WIN-STATE-IS-A-TRANSITION-NOT-A-FRAME -- the focused goal-only
call is the prompt whose win-state polarity that scenario corrects, and the assertions below pin the
corrected wording. (These tests predate the requirement and carried NO spec reference at all; the
`spec-coverage` hook is staged-file-scoped, so the gap only surfaced when this file was next touched.)
"""

from __future__ import annotations

import ast
import json
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as awm
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, Transition

pytestmark = pytest.mark.memory_watchdog_skip

_ENGINE = "import numpy as np\ndef engine(grid, action, data):\n    return np.asarray(grid)\n"
_GOAL = "import numpy as np\ndef is_level_complete(grid):\n    return False\n"
_BOTH = (
    "import numpy as np\n"
    "def engine(grid, action, data):\n    return np.asarray(grid)\n"
    "def is_level_complete(grid):\n    return False\n"
)


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._b = payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def read(self, *_a: object) -> bytes:
        return self._b


def _proposer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> LocalGGUFProposer:
    monkeypatch.delenv("CARNOT_ARC_CODEONLY_INDUCE", raising=False)
    monkeypatch.setattr(awm, "E3_DIR", tmp_path)
    p = LocalGGUFProposer(
        repo_substr="X",
        model_path="/x.gguf",
        port=59998,
        no_think_prefix="/no_think\n",
        max_tokens=128,
        tries=1,
    )
    monkeypatch.setattr(p, "_ensure_server", lambda: True)
    return p


def _seq_urlopen(monkeypatch: pytest.MonkeyPatch, contents: list[str]) -> list:
    """Return /completion responses in order; capture each request body."""
    bodies: list = []
    it = iter(contents)

    def fake(req, timeout=None):  # noqa: ANN001
        bodies.append(json.loads(req.data.decode()))
        return _FakeResp(json.dumps({"content": next(it)}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return bodies


def _trans() -> list[Transition]:
    return [
        Transition(
            grid=np.zeros((2, 2), dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.ones((2, 2), dtype=np.int16),
            level_before=1,
            level_after=1,
        )
    ]


def test_combined_success_uses_single_call_no_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Happy path: the combined call yields both defs -> ONE call, no split (no regression)."""
    p = _proposer(monkeypatch, tmp_path)
    bodies = _seq_urlopen(monkeypatch, [_BOTH])
    ok, msg = p.induce("g", _trans(), 1)
    assert ok is True
    assert "split" not in msg
    assert len(bodies) == 1
    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert "def engine" in wm and "def is_level_complete" in wm


def test_combined_failure_falls_back_to_focused_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The captured failure mode: combined yields engine but NO is_level_complete -> fall back to a
    focused engine call + a focused goal call (with the win exemplar) -> a valid combined world model."""
    p = _proposer(monkeypatch, tmp_path)
    # combined call (returns engine-only -> missing is_level_complete -> fails), then engine-only ok,
    # then goal-only ok.
    bodies = _seq_urlopen(monkeypatch, [_ENGINE, _ENGINE, _GOAL])
    ok, msg = p.induce("g", _trans(), 1, previous_level_complete_grid=np.array([[1, 0], [0, 1]]))
    assert ok is True
    assert "split induce" in msg
    assert len(bodies) == 3  # combined + focused engine + focused goal
    # The focused goal call still carries the boundary grid -- but CORRECTED 2026-07-29 it is no
    # longer labelled a "WIN STATE". That grid is captured after the level counter incremented, so it
    # is the CURRENT level's opening board; `_goal_only_prompt` used to tell the model
    # "is_level_complete must return True here", which is false and also contradicts
    # `_goal_satisfiability_check`'s root-true rejection. Assert the corrected polarity, and that the
    # grid is still present (the object vocabulary is the useful part).
    goal_prompt = bodies[2]["prompt"]
    assert "WIN STATE" not in goal_prompt
    assert "board at the START of the current level" in goal_prompt
    assert "is_level_complete must return False here" in goal_prompt
    # all three are code-only (directive + stop) even though required differs per call
    for b in bodies:
        assert b.get("stop") == ["```"]
        assert b["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    wm = (tmp_path / "g" / "world_model.py").read_text()
    assert "def engine" in wm and "def is_level_complete" in wm
    ast.parse(wm)


def test_split_engine_failure_reports_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """If even the focused engine call cannot produce a valid engine, induce reports failure (no
    fabricated world model)."""
    p = _proposer(monkeypatch, tmp_path)
    # combined fails (no goal), focused engine also fails (prose, no def engine)
    bodies = _seq_urlopen(monkeypatch, [_ENGINE, "just prose, no code here", _GOAL])
    ok, msg = p.induce("g", _trans(), 1, previous_level_complete_grid=np.array([[1, 0]]))
    assert ok is False
    assert "engine failed" in msg
    assert not (tmp_path / "g" / "world_model.py").exists()
    assert len(bodies) == 2  # combined + engine (goal never reached)
