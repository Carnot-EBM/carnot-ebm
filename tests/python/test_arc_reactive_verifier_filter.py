"""Tests for the verifier-filtered reactive loop (operator-directed architectural pivot,
2026-07-22): propose one action at a time, filter with a real verifier, no upfront symbolic
world-model induction.

Spec: REQ-ARC-WMTE-5827, SCENARIO-ARC-WMTE-5827-DEAD-END-FILTER-REJECTS-KNOWN-NOOP,
SCENARIO-ARC-WMTE-5827-FRAME-CHANGE-SCORER-RANKS-SURVIVORS
(openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

from carnot.agentic.arc_reactive_verifier_filter import (
    _ProposedAction,
    _dead_end_key,
    _filter_candidates,
)


class _FakeScorer:
    """Deterministic candidate_score: prefers higher x, so ranking is verifiable."""

    def candidate_score(self, frame, candidate):
        del frame
        if candidate.data and "x" in candidate.data:
            return float(candidate.data["x"])
        return float(candidate.action_id)


class TestDeadEndKey:
    def test_click_action_includes_xy(self):
        assert _dead_end_key("stateA", 6, {"x": 3, "y": 4}) == ("stateA", 6, 3, 4)

    def test_keyboard_action_has_none_xy(self):
        assert _dead_end_key("stateA", 1, None) == ("stateA", 1, None, None)

    def test_different_states_are_distinct_keys(self):
        assert _dead_end_key("stateA", 6, {"x": 3, "y": 4}) != _dead_end_key(
            "stateB", 6, {"x": 3, "y": 4}
        )


class TestFilterCandidates:
    def test_rejects_known_dead_end(self):
        dead_end_keys = {("s0", 6, 3, 4)}
        proposed = [{"action": 6, "data": {"x": 3, "y": 4}}]
        chosen, rejections, scored = _filter_candidates(
            proposed,
            state_key="s0",
            dead_end_keys=dead_end_keys,
            frame=None,
            frame_change_scorer=None,
        )
        assert chosen is None
        assert rejections == 1
        assert scored == 0

    def test_accepts_non_dead_end_with_no_scorer(self):
        proposed = [{"action": 6, "data": {"x": 3, "y": 4}}]
        chosen, rejections, scored = _filter_candidates(
            proposed, state_key="s0", dead_end_keys=set(), frame=None, frame_change_scorer=None
        )
        assert isinstance(chosen, _ProposedAction)
        assert chosen.action_id == 6
        assert chosen.data == {"x": 3, "y": 4}
        assert rejections == 0
        assert scored == 0  # no scorer supplied -- first survivor taken, not scored

    def test_frame_change_scorer_picks_highest_scoring_survivor(self):
        proposed = [
            {"action": 6, "data": {"x": 1, "y": 0}},
            {"action": 6, "data": {"x": 9, "y": 0}},
            {"action": 6, "data": {"x": 5, "y": 0}},
        ]
        chosen, rejections, scored = _filter_candidates(
            proposed,
            state_key="s0",
            dead_end_keys=set(),
            frame="fake_frame",
            frame_change_scorer=_FakeScorer(),
        )
        assert chosen.data == {"x": 9, "y": 0}  # highest-x wins per _FakeScorer
        assert rejections == 0
        assert scored == 3

    def test_dead_end_filter_runs_before_scoring(self):
        """The best-scoring candidate is a known dead-end -- it must be rejected, not chosen,
        even though it would win on frame-change score alone."""
        dead_end_keys = {("s0", 6, 9, 0)}
        proposed = [
            {"action": 6, "data": {"x": 1, "y": 0}},
            {"action": 6, "data": {"x": 9, "y": 0}},  # dead end, would otherwise win
        ]
        chosen, rejections, scored = _filter_candidates(
            proposed,
            state_key="s0",
            dead_end_keys=dead_end_keys,
            frame="fake_frame",
            frame_change_scorer=_FakeScorer(),
        )
        assert chosen.data == {"x": 1, "y": 0}
        assert rejections == 1
        assert scored == 1

    def test_all_candidates_rejected_returns_none(self):
        dead_end_keys = {("s0", 6, 1, 1), ("s0", 3, None, None)}
        proposed = [{"action": 6, "data": {"x": 1, "y": 1}}, {"action": 3}]
        chosen, rejections, scored = _filter_candidates(
            proposed,
            state_key="s0",
            dead_end_keys=dead_end_keys,
            frame=None,
            frame_change_scorer=_FakeScorer(),
        )
        assert chosen is None
        assert rejections == 2
        assert scored == 0

    def test_scorer_exception_treated_as_zero_score_not_a_crash(self):
        class _BrokenScorer:
            def candidate_score(self, frame, candidate):
                raise RuntimeError("model unavailable")

        proposed = [{"action": 6, "data": {"x": 1, "y": 1}}]
        chosen, rejections, scored = _filter_candidates(
            proposed,
            state_key="s0",
            dead_end_keys=set(),
            frame=None,
            frame_change_scorer=_BrokenScorer(),
        )
        assert chosen is not None  # still picks a candidate despite the scorer failing
        assert rejections == 0
        assert scored == 1

    def test_ignores_proposal_missing_action_field(self):
        proposed = [{"not_action": 6}, {"action": 3}]
        chosen, rejections, scored = _filter_candidates(
            proposed, state_key="s0", dead_end_keys=set(), frame=None, frame_change_scorer=None
        )
        assert chosen.action_id == 3
        assert rejections == 0
