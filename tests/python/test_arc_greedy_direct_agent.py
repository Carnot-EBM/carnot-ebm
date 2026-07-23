"""Tests for the Duck-Harness-style greedy-direct agent's pure parse/decide logic
(python/carnot/agentic/arc_greedy_direct_agent.py, 2026-07-23, operator: match the leaderboard leaders).

Covers the two error-prone pieces without a GPU: the primed-sequence parser and the
tool-inspection-then-commit orientation loop (scripted proposer). The full env loop
(run_greedy_direct) is validated separately by the real single-game smoke.

Spec: REQ-ARC-WMTE-5829, SCENARIO-ARC-WMTE-5829-COMMIT-WITHIN-BUDGET,
SCENARIO-ARC-WMTE-5829-CLICK-COORD-MAPPING
(openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic import arc_greedy_direct_agent as gda
from carnot.agentic.arc_greedy_direct_agent import (
    _parse_sequence,
    _render_objects,
    decide_sequence,
)


class TestParseSequence:
    def test_single_action(self):
        seq = _parse_sequence('{"a":4}]', max_seq=5, avail=[1, 2, 3, 4, 6])
        assert seq == [{"a": 4}]

    def test_sequence_of_moves(self):
        seq = _parse_sequence('{"a":4},{"a":4},{"a":3}]', max_seq=5, avail=[1, 2, 3, 4, 6])
        assert seq == [{"a": 4}, {"a": 4}, {"a": 3}]

    def test_click_keeps_logical_coords(self):
        seq = _parse_sequence('{"a":6,"x":3,"y":5}]', max_seq=5, avail=[1, 6])
        assert seq == [{"a": 6, "x": 3, "y": 5}]

    def test_caps_at_max_seq(self):
        seq = _parse_sequence(",".join(['{"a":1}'] * 10) + "]", max_seq=3, avail=[1])
        assert len(seq) == 3

    def test_drops_unavailable_actions(self):
        seq = _parse_sequence('{"a":5},{"a":1}]', max_seq=5, avail=[1, 2])  # 5 not available
        assert seq == [{"a": 1}]

    def test_click_without_coords_dropped(self):
        seq = _parse_sequence('{"a":6}]', max_seq=5, avail=[1, 6])  # click needs x,y
        assert seq == []

    def test_salvages_missing_trailing_bracket(self):
        seq = _parse_sequence('{"a":2},{"a":2}', max_seq=5, avail=[2])  # no closing ]
        assert seq == [{"a": 2}, {"a": 2}]

    def test_garbage_returns_empty(self):
        assert _parse_sequence("<think> reasoning leaked", max_seq=5, avail=[1]) == []

    def test_loose_flat_click_recovered(self):
        # gemma-4-31B emitted a bare "6, 18, 41" after the prime (first smoke, 2026-07-23);
        # coerce a flat 3-number list into a=6,x=18,y=41 rather than wasting a retry.
        seq = _parse_sequence("6, 18, 41]", max_seq=5, avail=[1, 6])
        assert seq == [{"a": 6, "x": 18, "y": 41}]

    def test_loose_flat_single_move_recovered(self):
        seq = _parse_sequence("4]", max_seq=5, avail=[4])
        assert seq == [{"a": 4}]

    def test_loose_two_numbers_is_ambiguous_dropped(self):
        # 2 numbers can't be [a] or [a,x,y] -- do not guess.
        assert _parse_sequence("6, 18]", max_seq=5, avail=[6]) == []


class TestObjectPerception:
    """v2: winner-style object segmentation view (REQ-ARC-WMTE-5829)."""

    def test_render_objects_lists_discrete_regions(self):
        grid = np.full((16, 16), 5, dtype=int)  # 5 = dominant background (excluded)
        grid[2:4, 3:5] = 7  # one small object
        grid[10:12, 10:13] = 9  # another
        text, objects = _render_objects(grid)
        assert len(objects) == 2
        colors = {o["color"] for o in objects}
        assert colors == {7, 9}
        assert "#0" in text and "color=" in text

    def test_parse_resolves_object_reference_to_centroid(self):
        objects = [
            {"id": 0, "color": 7, "row": 2, "col": 4, "size": 4},
            {"id": 1, "color": 9, "row": 11, "col": 11, "size": 6},
        ]
        seq = _parse_sequence('{"a":6,"obj":1}]', max_seq=5, avail=[1, 6], objects=objects)
        assert seq == [{"a": 6, "x": 11, "y": 11}]  # object 1's (col,row) centroid

    def test_parse_drops_unknown_object_id(self):
        objects = [{"id": 0, "color": 7, "row": 2, "col": 4, "size": 4}]
        assert _parse_sequence('{"a":6,"obj":9}]', max_seq=5, avail=[6], objects=objects) == []

    def test_decide_sequence_objects_mode_commits_object_click(self, monkeypatch):
        grid = np.full((16, 16), 5, dtype=int)
        grid[2:4, 3:5] = 7
        proposer = _FakeProposer(["commit", '{"a":6,"obj":0}]'])
        _patch(monkeypatch, proposer)
        seq, transcript = decide_sequence(
            grid, [], [1, 6], proposer, max_turns=4, max_seq=5, perception="objects"
        )
        # object 0 is the color-7 region centered near row 2-3, col 3-4
        assert len(seq) == 1 and seq[0]["a"] == 6 and "x" in seq[0] and "y" in seq[0]


class _FakeProposer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.timeout = 30

    def _ensure_server(self):
        return True

    def _url(self):
        return "http://fake"


def _patch(monkeypatch, proposer):
    def fake_complete(p, prompt, *, max_tokens, stop, seed=None):
        del p, prompt, max_tokens, stop, seed
        idx = proposer.calls
        proposer.calls += 1
        if idx >= len(proposer.responses):
            return False, "no more scripted responses"
        return True, proposer.responses[idx]

    monkeypatch.setattr(gda, "_complete", fake_complete)


class TestDecideSequence:
    def test_commits_immediately_when_model_gives_action(self, monkeypatch):
        # turn 0: the tool-line completion is NOT a TOOL, so it re-prompts primed -> the commit
        proposer = _FakeProposer(["ACTION: ...", '{"a":4},{"a":4}]'])
        _patch(monkeypatch, proposer)
        seq, transcript = decide_sequence(
            np.zeros((8, 8), dtype=int), [], [1, 2, 3, 4, 6], proposer, max_turns=4, max_seq=5
        )
        assert seq == [{"a": 4}, {"a": 4}]
        assert any("COMMIT" in t for t in transcript)

    def test_tool_call_then_commit(self, monkeypatch):
        # turn 0: a TOOL line (honored, dispatched); turn 1: not-a-tool -> primed commit
        proposer = _FakeProposer(["TOOL: inspect_cell 1 1", "commit now", '{"a":1}]'])
        _patch(monkeypatch, proposer)
        seq, transcript = decide_sequence(
            np.zeros((8, 8), dtype=int), [], [1, 6], proposer, max_turns=4, max_seq=5
        )
        assert seq == [{"a": 1}]
        assert any("TOOL inspect_cell" in t for t in transcript)

    def test_repeated_tool_short_circuited(self, monkeypatch):
        # A non-tool line triggers a SECOND (primed) completion that turn; a tool line consumes only
        # one. So: turn0 tool (1), turn1 same tool -> short-circuit (1), turn2 non-tool (1) + primed
        # commit (1) = 4 scripted responses total.
        proposer = _FakeProposer(
            ["TOOL: count_color 3", "TOOL: count_color 3", "commit now", '{"a":2}]']
        )
        _patch(monkeypatch, proposer)
        seq, transcript = decide_sequence(
            np.zeros((8, 8), dtype=int), [], [1, 2], proposer, max_turns=6, max_seq=5
        )
        assert seq == [{"a": 2}]
        assert any("already asked" in t for t in transcript)

    def test_exhausts_turns_without_parseable_action_returns_empty(self, monkeypatch):
        proposer = _FakeProposer(["junk", "junk", "junk", "junk", "junk", "junk", "junk", "junk"])
        _patch(monkeypatch, proposer)
        seq, _ = decide_sequence(
            np.zeros((5, 5), dtype=int), [], [1], proposer, max_turns=3, max_seq=5
        )
        assert seq == []
