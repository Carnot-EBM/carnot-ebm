"""Tests for the tool-calling orientation + multi-step lookahead search
(operator directive, 2026-07-23, following up on REQ-ARC-WMTE-5827's GAP-ARC-REACTIVE-FILTER-MYOPIC
diagnosis): up to 12 tool-calling turns per decision (inspect, reason, then propose a RANKED
candidate set), real multi-step lookahead by reusing arc_solver_kit.OfflineSolver's best-first
search rather than a new search algorithm.

Spec: REQ-ARC-WMTE-5828, SCENARIO-ARC-WMTE-5828-TOOL-LOOP-COMMITS-WITHIN-BUDGET,
SCENARIO-ARC-WMTE-5828-DEDUP-PREVENTS-REPEAT-QUERY,
SCENARIO-ARC-WMTE-5828-WARMUP-SENTINEL-NEVER-COLLIDES-WITH-GENUINE-CHOICE
(openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_tool_loop_lookahead import (
    MAX_CANDIDATES,
    ToolLoopLookaheadSession,
    _candidates_from_payload,
    _parse_turn,
    _ToolDispatch,
    run_tool_loop,
)


class _FakeProposer:
    """A scripted proposer: replays a fixed sequence of raw completion texts, one per call.
    Mirrors the real _completion() request shape closely enough that run_tool_loop never
    touches the network -- these tests need no GPU."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.timeout = 30

    def _ensure_server(self) -> bool:
        return True

    def _url(self) -> str:
        return "http://fake"


def _patch_completion(monkeypatch, proposer: _FakeProposer) -> None:
    import carnot.agentic.arc_tool_loop_lookahead as mod

    def fake_completion(p, prompt, *, max_tokens):
        del p, prompt, max_tokens
        idx = proposer.calls
        proposer.calls += 1
        if idx >= len(proposer.responses):
            return False, "no more scripted responses"
        return True, proposer.responses[idx]

    monkeypatch.setattr(mod, "_completion", fake_completion)


class TestParseTurn:
    def test_parses_tool_call(self):
        kind, payload = _parse_turn("TOOL: inspect_cell 3 4")
        assert kind == "tool"
        assert payload == ("inspect_cell", "3 4")

    def test_parses_action_array(self):
        kind, payload = _parse_turn('ACTION: [{"a":6,"x":1,"y":2,"confidence":0.8}]')
        assert kind == "action"
        assert payload == [{"a": 6, "x": 1, "y": 2, "confidence": 0.8}]

    def test_rejects_action_object_not_array(self):
        # the contract requires a JSON ARRAY (a ranked candidate set), even for one candidate
        kind, _ = _parse_turn('ACTION: {"a":6,"x":1,"y":2,"confidence":0.8}')
        assert kind == "unparseable"

    def test_unparseable_garbage(self):
        kind, _ = _parse_turn("<think>\nsome reasoning leaked through\n</think>")
        assert kind == "unparseable"

    def test_empty_text_is_unparseable(self):
        kind, _ = _parse_turn("")
        assert kind == "unparseable"


class TestCandidatesFromPayload:
    def test_parses_valid_candidates(self):
        payload = [{"a": 6, "x": 1, "y": 2, "confidence": 0.9}, {"a": 1, "confidence": 0.3}]
        cands = _candidates_from_payload(payload)
        assert len(cands) == 2
        assert cands[0].action_id == 6
        assert cands[0].data == {"x": 1, "y": 2}
        assert cands[1].action_id == 1
        assert cands[1].data is None

    def test_caps_at_max_candidates(self):
        payload = [{"a": 1, "confidence": 0.5} for _ in range(MAX_CANDIDATES + 5)]
        cands = _candidates_from_payload(payload)
        assert len(cands) == MAX_CANDIDATES

    def test_clamps_confidence_to_unit_range(self):
        payload = [{"a": 1, "confidence": 5.0}, {"a": 2, "confidence": -3.0}]
        cands = _candidates_from_payload(payload)
        assert cands[0].confidence == 1.0
        assert cands[1].confidence == 0.0

    def test_skips_items_without_action_field(self):
        payload = [{"confidence": 0.5}, {"a": 3, "confidence": 0.5}]
        cands = _candidates_from_payload(payload)
        assert len(cands) == 1
        assert cands[0].action_id == 3

    def test_action6_without_xy_has_no_data(self):
        cands = _candidates_from_payload([{"a": 6, "confidence": 0.5}])
        assert cands[0].data is None


class TestToolDispatch:
    def test_inspect_cell_reports_color(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 3] = 7
        dispatch = _ToolDispatch(grid, [], [1, 6])
        assert dispatch.call("inspect_cell", "2 3") == "7"

    def test_inspect_cell_out_of_bounds(self):
        grid = np.zeros((5, 5), dtype=int)
        dispatch = _ToolDispatch(grid, [], [1])
        assert "out of bounds" in dispatch.call("inspect_cell", "99 99")

    def test_count_color(self):
        grid = np.zeros((4, 4), dtype=int)
        grid[0, 0] = grid[1, 1] = 3
        dispatch = _ToolDispatch(grid, [], [1])
        assert dispatch.call("count_color", "3") == "2"

    def test_inspect_history_empty(self):
        dispatch = _ToolDispatch(np.zeros((3, 3), dtype=int), [], [1])
        assert dispatch.call("inspect_history", "") == "(no history yet)"

    def test_unknown_tool_reports_error(self):
        dispatch = _ToolDispatch(np.zeros((3, 3), dtype=int), [], [1])
        assert "unknown tool" in dispatch.call("nonexistent_tool", "")


class TestRunToolLoopScripted:
    """Real control-flow tests using a scripted (non-network) proposer -- no GPU needed."""

    def test_commits_immediately_when_model_answers_directly(self, monkeypatch):
        proposer = _FakeProposer(['ACTION: [{"a":6,"x":1,"y":2,"confidence":0.9}]'])
        _patch_completion(monkeypatch, proposer)
        outcome = run_tool_loop(np.zeros((5, 5), dtype=int), [], [1, 6], proposer, max_turns=12)
        assert outcome.ok is True
        assert outcome.turns_used == 1
        assert outcome.candidates[0].action_id == 6
        assert outcome.candidates[0].confidence == 0.9

    def test_tool_call_then_commit(self, monkeypatch):
        proposer = _FakeProposer(
            [
                "TOOL: inspect_frame",
                'ACTION: [{"a":1,"confidence":0.6}]',
            ]
        )
        _patch_completion(monkeypatch, proposer)
        outcome = run_tool_loop(np.zeros((5, 5), dtype=int), [], [1], proposer, max_turns=12)
        assert outcome.ok is True
        assert outcome.turns_used == 2
        assert any("TOOL inspect_frame" in line for line in outcome.transcript)

    def test_repeated_tool_call_is_short_circuited_without_a_new_completion(self, monkeypatch):
        proposer = _FakeProposer(
            [
                "TOOL: inspect_cell 1 1",
                "TOOL: inspect_cell 1 1",  # exact repeat -- should be short-circuited locally
                'ACTION: [{"a":1,"confidence":0.5}]',
            ]
        )
        _patch_completion(monkeypatch, proposer)
        outcome = run_tool_loop(np.zeros((5, 5), dtype=int), [], [1], proposer, max_turns=12)
        assert outcome.ok is True
        assert any("already asked" in line for line in outcome.transcript)

    def test_exhausts_budget_without_commit_returns_not_ok(self, monkeypatch):
        proposer = _FakeProposer(["TOOL: inspect_frame"] * 3)
        _patch_completion(monkeypatch, proposer)
        outcome = run_tool_loop(np.zeros((5, 5), dtype=int), [], [1], proposer, max_turns=3)
        assert outcome.ok is False
        assert outcome.candidates == []
        assert outcome.turns_used == 3

    def test_unparseable_completion_consumes_a_turn_but_continues(self, monkeypatch):
        proposer = _FakeProposer(
            [
                "<think>garbage that never resolves</think>",
                'ACTION: [{"a":2,"confidence":0.4}]',
            ]
        )
        _patch_completion(monkeypatch, proposer)
        outcome = run_tool_loop(np.zeros((5, 5), dtype=int), [], [1, 2], proposer, max_turns=12)
        assert outcome.ok is True
        assert outcome.turns_used == 2


class TestWarmupSentinel:
    def test_warmup_label_is_not_valid_json(self):
        import json

        with __import__("pytest").raises(Exception):
            json.loads(ToolLoopLookaheadSession.WARMUP_LABEL)

    def test_warmup_label_cannot_collide_with_a_genuine_json_action_label(self):
        from carnot.agentic.arc_game_adapters import _json_action_label

        # every genuine candidate label is valid JSON; the sentinel deliberately is not,
        # so a `label == WARMUP_LABEL` check in apply() can never misfire on a real choice
        # (the exact bug this test guards -- see the class's own WARMUP_LABEL docstring).
        for action_id in range(8):
            assert _json_action_label(action_id) != ToolLoopLookaheadSession.WARMUP_LABEL
        assert _json_action_label(1, {"x": 1, "y": 1}) != ToolLoopLookaheadSession.WARMUP_LABEL


class _Frame:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame


class TestActionLabelsFallbackPadding:
    """Regression coverage for a real bug found via a direct trace (not assumed): when the tool
    loop's only proposed candidate turns out to be a no-op, its resulting state hashes IDENTICAL
    to the parent and is correctly never pushed to the search frontier -- but with no OTHER
    candidate offered, the search then has nothing left to explore and dies after one node,
    regardless of the turn/node budgets. action_labels() must always pad a thin candidate set
    with real structured fallback alternatives so a single bad guess can't starve the search."""

    def test_pads_a_single_tool_loop_candidate_with_fallbacks(self, monkeypatch):
        # rich_action_candidates is imported inside action_labels() via a LOCAL import, so it
        # must be patched at its source module -- patching it on arc_tool_loop_lookahead itself
        # would have no effect (the local import re-reads the source module at call time).
        import carnot.agentic.arc_graph_explore as graph_explore_mod

        proposer = _FakeProposer(['ACTION: [{"a":6,"x":0,"y":0,"confidence":0.9}]'])
        _patch_completion(monkeypatch, proposer)

        fake_fallback = type("FakeCand", (), {"action_id": 1, "data": None})()
        monkeypatch.setattr(
            graph_explore_mod, "rich_action_candidates", lambda frame: [fake_fallback]
        )

        session = ToolLoopLookaheadSession(proposer, max_turns=12)
        labels = session.action_labels(None, _Frame(np.zeros((8, 8), dtype=int)), ())

        assert len(labels) >= 2  # the tool-loop candidate PLUS at least one fallback
        assert '{"action":6,"data":{"x":0,"y":0}}' in labels
        assert '{"action":1}' in labels

    def test_does_not_pad_when_tool_loop_already_offers_multiple_candidates(self, monkeypatch):
        import carnot.agentic.arc_graph_explore as graph_explore_mod

        proposer = _FakeProposer(
            ['ACTION: [{"a":6,"x":0,"y":0,"confidence":0.9},{"a":1,"confidence":0.4}]']
        )
        _patch_completion(monkeypatch, proposer)

        calls = []
        monkeypatch.setattr(
            graph_explore_mod,
            "rich_action_candidates",
            lambda frame: calls.append(1) or [],
        )

        session = ToolLoopLookaheadSession(proposer, max_turns=12)
        labels = session.action_labels(None, _Frame(np.zeros((8, 8), dtype=int)), ())

        assert len(labels) == 2
        assert not calls  # the fallback path is only consulted when < 2 genuine candidates exist
