"""Tests for the goal-hypothesis verifier (python/carnot/agentic/arc_goal_verifier.py).

The winner-recipe reproduction (2026-07-23) showed gemma-4-31B induces plausible-but-WRONG goals and
pursues them to completion without ever winning. The GoalVerifier falsifies a goal against the ONLY
ground-truth signal a hidden-game agent has -- the observable level counter -- and feeds rejected goals
back so the model must hypothesize a different win condition.

Spec: REQ-ARC-WMTE-5830, SCENARIO-ARC-WMTE-5830-FALSIFY-ON-NO-LEVELUP,
SCENARIO-ARC-WMTE-5830-SUPPORT-ON-LEVELUP
(openspec/capabilities/arc-world-model-trust-energy/spec.md).
"""

from __future__ import annotations

from carnot.agentic.arc_goal_verifier import (
    GoalVerifier,
    extract_goal,
    progress_indicates_completion,
)


class TestExtractGoal:
    def test_parses_goal_line(self):
        notes = "RULES: click toggles\nGOAL: Fill column 63 with 15s\nPROGRESS: done"
        assert extract_goal(notes) == "Fill column 63 with 15s"

    def test_case_insensitive(self):
        assert extract_goal("goal: set all cells to 1") == "set all cells to 1"

    def test_no_goal_returns_empty(self):
        assert extract_goal("RULES: click toggles\nPROGRESS: none") == ""

    def test_empty_notes(self):
        assert extract_goal("") == ""


class TestGoalVerifier:
    def test_falsifies_a_pursued_goal_with_no_levelup(self):
        gv = GoalVerifier(goal_patience=5, min_activity=3)
        gv.set_goal("fill column 63")
        for _ in range(6):  # >= patience, all frame-changing, no level-up
            gv.observe(frame_changed=True, leveled_up=False)
        assert gv.verdict() == "falsified"
        assert gv.maybe_falsify() is True
        assert "fill column 63" in gv.falsified
        assert "fill column 63" in gv.feedback()

    def test_supports_a_goal_that_produced_a_levelup(self):
        gv = GoalVerifier(goal_patience=5, min_activity=3)
        gv.set_goal("reach the exit")
        gv.observe(frame_changed=True, leveled_up=False)
        gv.observe(frame_changed=True, leveled_up=True)  # a level-up while pursuing this goal
        assert gv.verdict() == "supported"
        assert gv.maybe_falsify() is False
        assert "reach the exit" in gv.supported

    def test_pending_until_enough_activity(self):
        gv = GoalVerifier(goal_patience=10, min_activity=5)
        gv.set_goal("some goal")
        for _ in range(3):
            gv.observe(frame_changed=True, leveled_up=False)
        assert gv.verdict() == "pending"

    def test_does_not_falsify_an_unpursued_goal_no_activity(self):
        # a goal the agent never actually acted on (all no-ops) is not evidence against the goal
        gv = GoalVerifier(goal_patience=5, min_activity=3)
        gv.set_goal("some goal")
        for _ in range(20):
            gv.observe(frame_changed=False, leveled_up=False)
        assert gv.verdict() == "pending"  # activity floor not met

    def test_new_goal_resets_counters_and_counts_switch(self):
        gv = GoalVerifier(goal_patience=5, min_activity=3)
        gv.set_goal("goal A")
        for _ in range(6):
            gv.observe(frame_changed=True, leveled_up=False)
        gv.maybe_falsify()
        gv.set_goal("goal B")  # a genuinely new goal
        assert gv.actions_on_goal == 0
        assert gv.goal_switches == 2
        assert gv.verdict() == "pending"  # goal B is fresh

    def test_same_goal_does_not_reset(self):
        gv = GoalVerifier()
        gv.set_goal("goal A")
        gv.observe(frame_changed=True, leveled_up=False)
        gv.set_goal("goal A")  # unchanged
        assert gv.actions_on_goal == 1
        assert gv.goal_switches == 1

    def test_empty_feedback_when_nothing_falsified(self):
        assert GoalVerifier().feedback() == ""

    def test_stats_shape(self):
        gv = GoalVerifier(goal_patience=2, min_activity=1)
        gv.set_goal("g")
        gv.observe(frame_changed=True, leveled_up=False)
        gv.observe(frame_changed=True, leveled_up=False)
        gv.maybe_falsify()
        s = gv.stats()
        assert s["falsified_goals"] == ["g"] and s["goal_switches"] == 1


class TestCompletionFalsification:
    def test_progress_completion_detected(self):
        assert progress_indicates_completion("PROGRESS: column 63 is filled with 15s") is True
        assert progress_indicates_completion("PROGRESS: all grid cells are state 1") is True

    def test_progress_incomplete_not_detected(self):
        assert progress_indicates_completion("PROGRESS: clicked a few cells, exploring") is False
        assert progress_indicates_completion("RULES: click toggles") is False

    def test_falsify_on_reported_completion(self):
        gv = GoalVerifier()
        gv.set_goal("fill column 63")
        gv.observe(frame_changed=True, leveled_up=False)  # real activity, no level-up
        assert gv.falsify_on_reported_completion("PROGRESS: column 63 is filled") is True
        assert "fill column 63" in gv.falsified

    def test_no_completion_falsify_if_leveled_up(self):
        gv = GoalVerifier()
        gv.set_goal("reach exit")
        gv.observe(frame_changed=True, leveled_up=True)
        assert gv.falsify_on_reported_completion("PROGRESS: reached and completed") is False
