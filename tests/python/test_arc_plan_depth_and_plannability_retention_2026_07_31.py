"""REQ-ARC-WMTE-6035 / REQ-ARC-FCP-5699-15 -- the coupled search horizon, and plannability as a
retention TIEBREAK.

TWO CHANGES LANDED 2026-07-31, tested together because they were measured together and because
the second is only meaningful given the first.

  (1) `plan_in_model` and `_goal_satisfiability_check` read their depth cap from ONE resolver,
      `arc_executable_world_model.plan_max_depth_default()`, defaulting to 80 rather than 40.
  (2) `execute_bounded_llm_reinduction` breaks a retention TIE on plannability -- among rounds
      whose held-out dynamics signal is not worse than the incumbent's, prefer one whose goal is
      reachable.

WHAT THE MEASUREMENT SAID, since these tests encode it. Re-running the shipped gate and planner
over 48 frozen induction candidates while sweeping ONLY `max_depth`:

    max_depth   clears dynamics   plan found   BOTH
           40                 9            2      0
           61                 9            6      3
          200                 9            6      3

The empty intersection at the shipped default was a horizon artifact, not evidence that the two
criteria conflict. And it did not fully dissolve: at the corrected horizon 9 candidates still
disagree, including 3 that are dynamics-perfect and unplannable -- which is what (2) is for.

"TIEBREAK" AND NOT "PRIMARY KEY", and this file said the wrong one for a while. The first draft
of (2) ranked plannability ABOVE dynamics outright; `test_arc_engine_retention_best_round.py::
test_a_planned_return_reports_its_own_round_not_the_retained_one` -- which already encoded the
opposite decision -- failed, and it was right. Retention exists to preserve the best DYNAMICS
model, so a reachable goal must not rescue a materially worse engine. The measured conflict is
among SIX tn36 candidates ALL at held-out accuracy 1.0, of which only THREE plan: that is a tie,
not a deficit, and only tie-breaking is supported by it.

SCENARIO-ARC-WMTE-6035-PLANNABILITY-TIEBREAK
SCENARIO-ARC-FCP-5699-15-COUPLED-DEPTH
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as reind


# --------------------------------------------------------------------------------------------
# A world where the goal sits at a KNOWN depth, on a PATH -- the shape the real corpus had.
#
# Deliberately mirrors the measured tn36 mechanic rather than inventing a convenient one: every
# action fills the next cell, so all actions from a state collapse to ONE distinct successor and
# the deduped search tree is a path. That is why `max_nodes` was never the binding constraint
# there and `max_depth` was; a branching fixture would test a different thing.
# --------------------------------------------------------------------------------------------
def _path_world(goal_depth: int, width: int = 100):
    def engine(grid, action, data=None):
        g = np.asarray(grid).copy()
        filled = int((g[0] == 1).sum())
        if filled < width:
            g[0, filled] = 1
        return g

    def is_level_complete(grid):
        return bool(int((np.asarray(grid)[0] == 1).sum()) >= goal_depth)

    start = np.zeros((1, width), dtype=int)
    return engine, is_level_complete, start


class TestCoupledDepthResolver:
    def test_default_is_80_not_the_old_40(self, monkeypatch):
        monkeypatch.delenv("CARNOT_ARC_PLAN_MAX_DEPTH", raising=False)
        assert e3.plan_max_depth_default() == 80

    def test_env_override_restores_the_previous_behaviour_exactly(self, monkeypatch):
        """The change must be reversible without editing source -- that is what makes it an A/B."""
        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "40")
        assert e3.plan_max_depth_default() == 40

    @pytest.mark.parametrize("bad", ["", "   ", "abc", "0", "-5", "1.5"])
    def test_malformed_or_nonpositive_override_falls_back_to_the_default(self, monkeypatch, bad):
        """Fail-safe direction is the DEFAULT, never 0. A 0 horizon would silently disable the
        planner entirely -- every search would discard its root unexpanded and return None, which
        reads downstream as 'no plan exists' rather than 'the knob was malformed'."""
        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", bad)
        assert e3.plan_max_depth_default() == 80

    def test_the_gate_and_the_planner_resolve_to_the_SAME_number(self, monkeypatch):
        """THE soundness property. The gate vetoes goals it cannot reach within its cap and
        justifies that veto by asserting the planner is bounded identically. If these two ever
        disagree the gate is either certifying goals the planner will fail on, or vetoing goals
        the planner could reach. Asserted at three values so a hardcoded literal cannot pass."""
        for value in ("40", "80", "137"):
            monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", value)
            engine, goal, start = _path_world(goal_depth=int(value) + 5)
            check = reind._goal_satisfiability_check(engine=engine, goal=goal, start_grid=start)
            assert check["max_depth"] == int(value) == e3.plan_max_depth_default()

    def test_goal_at_depth_61_is_unreachable_at_40_and_reachable_at_80(self, monkeypatch):
        """The measured tn36 case, reduced to a fixture: a 61-action goal against the old cap."""
        engine, goal, start = _path_world(goal_depth=61)

        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "40")
        diag_40: dict = {}
        assert e3.plan_in_model(engine, goal, start, diagnostics=diag_40) is None
        assert diag_40["termination_reason"] == "depth_capped"

        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "80")
        diag_80: dict = {}
        plan = e3.plan_in_model(engine, goal, start, diagnostics=diag_80)
        assert plan is not None and len(plan) == 61
        assert diag_80["termination_reason"] == "plan_found"

    def test_gate_agrees_with_planner_at_both_horizons(self, monkeypatch):
        """Not a restatement of the previous test: it checks the GATE flips in lockstep with the
        PLANNER. The measured corpus showed exactly this lockstep, and it is the property that
        makes a gate veto trustworthy."""
        engine, goal, start = _path_world(goal_depth=61)

        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "40")
        capped = reind._goal_satisfiability_check(engine=engine, goal=goal, start_grid=start)
        assert capped["satisfiable"] is False
        assert capped["counterexample"]["kind"] == "goal_unreached_within_depth"

        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "80")
        assert (
            reind._goal_satisfiability_check(engine=engine, goal=goal, start_grid=start)[
                "satisfiable"
            ]
            is True
        )

    def test_an_explicit_caller_argument_still_wins_over_the_default(self, monkeypatch):
        """Every existing test and diagnostic that pins a horizon must keep pinning it; the
        resolver supplies a default, it does not seize control."""
        monkeypatch.setenv("CARNOT_ARC_PLAN_MAX_DEPTH", "80")
        engine, goal, start = _path_world(goal_depth=61)
        assert e3.plan_in_model(engine, goal, start, max_depth=40) is None
        check = reind._goal_satisfiability_check(
            engine=engine, goal=goal, start_grid=start, max_depth=40
        )
        assert check["max_depth"] == 40 and check["satisfiable"] is False

    def test_max_nodes_was_not_widened(self, monkeypatch):
        """This change is a HORIZON correction, not a budget relaxation. If a future edit raises
        the node budget under cover of this one, the depth finding stops supporting it."""
        monkeypatch.delenv("CARNOT_ARC_GOAL_GATE_MAX_NODES", raising=False)
        # Goal deliberately BEYOND the horizon: a satisfiable check short-circuits and returns
        # only {satisfiable, first_true_depth, reachable_grids_evaluated, counterexample}, so the
        # budget fields exist to be asserted on only when the search actually ran to a bound.
        engine, goal, start = _path_world(goal_depth=500, width=600)
        check = reind._goal_satisfiability_check(engine=engine, goal=goal, start_grid=start)
        assert check["satisfiable"] is False
        assert check["max_nodes"] == 20000

    def test_factored_subgoal_planner_inherits_rather_than_pinning_40(self):
        """It used to declare its own literal 40, which would have silently kept this path on the
        old horizon after the change -- the drift the shared resolver exists to prevent."""
        import inspect

        sig = inspect.signature(e3.plan_factored_subgoal_sequence)
        assert sig.parameters["max_depth"].default is None


class TestPlannabilityPromotion:
    """Asserts the decision table of the SHIPPED `should_promote_on_plannability`.

    `_promote` below is a thin alias, NOT a reimplementation, and that distinction is the whole
    reason this class is trustworthy. The first version of these tests carried a private copy of
    the predicate; mutation testing then showed the suite stayed GREEN when the shipped
    expression was inverted (plannability made to LOSE to dynamics) and when its `retention_on`
    guard was deleted. Two tests that could not fail. The rule now lives in exactly one place and
    these call it, so deleting or inverting any conjunct fails here.
    """

    @staticmethod
    def _promote(**kw):
        """Thin pass-through with EQUAL signals as the default, because a tie is the measured
        case: the six conflicting tn36 candidates all score held-out accuracy 1.0. Any test about
        the dynamics floor passes the signals explicitly."""
        kw.setdefault("retention_signal", 1.0)
        kw.setdefault("best_signal", 1.0)
        return reind.should_promote_on_plannability(**kw)

    def test_a_plannable_round_displaces_an_unplannable_incumbent_ON_EQUAL_DYNAMICS(self):
        """The measured case: dynamics cannot separate them, so plannability decides."""
        assert self._promote(
            retention_on=True,
            round_goal_satisfiable=True,
            best_goal_satisfiable=False,
            is_best=False,
        )

    def test_plannability_does_NOT_override_a_dynamics_DEFICIT(self):
        """THE CORRECTION. The first draft made plannability the primary key, so a weak mover
        with a reachable goal displaced a good engine with an unreachable one. That is a plan to
        nowhere, and `test_arc_engine_retention_best_round.py::
        test_a_planned_return_reports_its_own_round_not_the_retained_one` had already decided
        against it. Nothing measured supports overriding a deficit -- only breaking a tie."""
        assert not self._promote(
            retention_on=True,
            round_goal_satisfiable=True,
            best_goal_satisfiable=False,
            is_best=False,
            retention_signal=0.30,
            best_signal=0.90,
        )

    def test_a_plannable_round_with_BETTER_dynamics_is_promoted(self):
        """The floor is `>=`, not equality: strictly better dynamics AND plannable must win. (It
        would usually win on the ordinary path anyway; asserted so a future edit cannot narrow
        the floor to `==` and silently drop this case.)"""
        assert self._promote(
            retention_on=True,
            round_goal_satisfiable=True,
            best_goal_satisfiable=False,
            is_best=False,
            retention_signal=0.95,
            best_signal=0.40,
        )

    def test_an_unplannable_round_never_displaces_a_plannable_incumbent(self):
        """Monotonicity. Once a plannable engine is retained, no dynamics score can dislodge it."""
        assert not self._promote(
            retention_on=True,
            round_goal_satisfiable=False,
            best_goal_satisfiable=True,
            is_best=False,
            retention_signal=1.0,
            best_signal=0.0,
        )

    def test_dynamics_decides_between_two_plannable_rounds(self):
        """With both satisfiable the promotion stands down and the pre-existing dynamics
        comparison decides -- which is what keeps a trivially satisfiable (degenerate) predicate
        from beating a genuinely good engine."""
        assert not self._promote(
            retention_on=True,
            round_goal_satisfiable=True,
            best_goal_satisfiable=True,
            is_best=False,
        )

    def test_it_does_not_fire_when_the_round_already_won_on_dynamics(self):
        """No double-application: `is_best` already true means the normal path did the work."""
        assert not self._promote(
            retention_on=True,
            round_goal_satisfiable=True,
            best_goal_satisfiable=False,
            is_best=True,
        )

    def test_disabling_retention_reproduces_the_prior_behaviour_exactly(self):
        """The env A/B switch must still be a true A/B, or the measured comparison is not
        recoverable."""
        assert not self._promote(
            retention_on=False,
            round_goal_satisfiable=True,
            best_goal_satisfiable=False,
            is_best=False,
        )

    def test_promotion_is_wired_and_records_itself_in_the_round(self):
        """A promotion that leaves no trace is unauditable: a reader of the artifact could not
        tell whether the retained engine won on dynamics or on plannability."""
        src = reind.__file__.replace(".pyc", ".py") if reind.__file__ else ""
        with open(src) as fh:
            body = fh.read()
        assert "_promote_on_plannability" in body
        assert "retained_by_plannability_promotion" in body
        assert "best_goal_satisfiable" in body

    def test_rejected_rounds_cannot_be_promoted(self):
        """Promotion widens WHICH accepted engine is kept, never WHETHER a rejected one is. A
        round failing the held-out verifier `continue`s before the goal check, so the promotion
        is unreachable for it -- asserted on the source order, since that ordering IS the
        guarantee."""
        src = reind.__file__.replace(".pyc", ".py")
        with open(src) as fh:
            body = fh.read()
        reject = body.index('row["skipped"] = "heldout_transition_verification_failed"')
        # Anchor on the CALL SITE, not the bare name: `should_promote_on_plannability` also
        # appears at module level as the definition, which sits ABOVE the reject-and-continue and
        # would make this assertion pass for the wrong reason.
        promote = body.index("_promote_on_plannability = should_promote_on_plannability(")
        assert reject < promote, "the reject-and-continue must precede the promotion"


def test_env_knob_is_absent_in_production_so_the_default_is_what_ships():
    """A default nobody reaches is not a default. If this variable is set in the environment the
    suite runs in, every measurement above is describing an override rather than shipped
    behaviour."""
    assert os.environ.get("CARNOT_ARC_PLAN_MAX_DEPTH") is None
