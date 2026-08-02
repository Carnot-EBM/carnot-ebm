"""Tests for the action-provenance CENSUS analysis -- the layer that turns episodes into
the headline number.

WHY THESE TESTS AND NOT OTHERS. The census driver's expensive half (spawn a subprocess,
load a 31B GGUF, step the offline arcade) is not what can silently be WRONG -- it either
runs or it visibly does not. The cheap half is where a wrong answer would look right:

  * the level-up attribution has a real off-by-one in it. A row's `level_before` and
    `level_after` are BOTH read off the frame the policy is looking at when it chooses, so
    a level-up appears as an increase BETWEEN consecutive rows and the action that caused
    it is the PREVIOUS row's. Reading `level_after` as "the level after this action" is the
    obvious interpretation and it is wrong; if the attribution were off by one the artifact
    would name the wrong branch as the one that banks levels, which is the single most
    load-bearing claim in the whole exercise.
  * "missing is not zero" is a policy, and a policy that is not tested is a comment. A
    crashed cell folded in as a zero manufactures exactly the null the measurement was
    looking for.
  * the wall-truncation floor was PRE-REGISTERED before the first cell landed. A
    pre-registration that the code does not actually implement is worse than none.

Each test therefore drives the real `analyse_cell` / `aggregate` against a hand-built
episode whose correct answer is known by construction. No GPU, no network, no subprocess,
and nothing written outside the test's own memory.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "scripts"))

from arc_action_provenance_census import (  # noqa: E402
    WALL_TRUNCATION_MIN_BUDGET_FRACTION,
    aggregate,
    analyse_cell,
)


def _row(i, top, *, level=0, explorer=None, serve=None, plan_epoch=0, plan_len=0):
    """One provenance row in the shape the instrument actually emits."""
    return {
        "i": i,
        "game": "zz00",
        "action": 1,
        "top_branch": top,
        "explorer_branch": explorer,
        "explorer_serve_kind": serve,
        "level_before": level,
        "level_after": level,
        "plan_epoch": plan_epoch,
        "plan_len": plan_len,
        "plan_present": top.startswith(("execute", "induce.plan")),
        "explorer_explored_out": False,
        "explorer_graph_nodes": 3,
        "explorer_cur_depth": 1,
    }


def _cell(rows, *, levels_gained=0, events=None, timed_out=False, error=None):
    by_top: dict[str, int] = {}
    by_exp: dict[str, int] = {}
    by_serve: dict[str, int] = {}
    for r in rows:
        by_top[r["top_branch"]] = by_top.get(r["top_branch"], 0) + 1
        if r.get("explorer_branch"):
            by_exp[r["explorer_branch"]] = by_exp.get(r["explorer_branch"], 0) + 1
        if r.get("explorer_serve_kind"):
            by_serve[r["explorer_serve_kind"]] = by_serve.get(r["explorer_serve_kind"], 0) + 1
    return {
        "game": "zz00",
        "replicate": 0,
        "missing_observation": bool(error),
        "error": error,
        "wall_s_measured": 12.0,
        "provenance": {
            "rows": rows,
            "summary": {
                "by_top_branch": by_top,
                "by_explorer_branch": by_exp,
                "by_serve_kind": by_serve,
                "explorer_fraction": None,
                "navigation_or_replay_fraction": None,
                "new_information_expansion_fraction": None,
                "new_information_expansions": 0,
                "navigation_or_replay_actions": 0,
                "plans_abandoned": 0,
                "plans_consumed_fully": 0,
                "recorder_errors": [],
            },
        },
        "result_row": {
            "levels_gained": levels_gained,
            "timed_out": timed_out,
            "error": None,
            "induction_events": events or [],
        },
    }


def _induction(*, produced=True, trusted=True, skipped="", planned=True, heldout=1.0):
    """One induction event in the shape `result_row.induction_events` actually carries."""
    return {
        "reason": "stall",
        "skipped": skipped,
        "planned": planned,
        "plan_length": 5,
        "refinement_rounds": [
            {
                "proposer_ok": produced,
                "accepted_by_heldout_verifier": trusted,
                "heldout_accuracy": heldout,
                "trust_energy": -3.2,
                "goal_predicate_satisfiable": True,
            }
        ],
    }


def test_plan_derived_numerator_excludes_the_reset_spent_replaying_a_plan():
    """The headline numerator counts actions the plan CHOSE, not actions spent because of
    one. The RESET that precedes a replay-from-root is real budget the pipeline caused, and
    it is reported -- separately. Folding it in would inflate the pipeline's share by one
    action per plan, which on these episode lengths is a visible fraction."""
    rows = [
        _row(0, "explore.explorer", explorer="bootstrap_reset"),
        _row(1, "induce.plan_needs_reset", plan_epoch=1, plan_len=2),
        _row(2, "execute.plan_step", plan_epoch=1, plan_len=2),
        _row(3, "execute.plan_step", plan_epoch=1, plan_len=2),
    ]
    a = analyse_cell(_cell(rows, events=[_induction()]), budget=4)
    assert a["n_plan_derived"] == 2, "only the two execute.plan_step actions were plan-chosen"
    assert a["n_reset_for_plan_replay"] == 1
    assert a["plan_derived_fraction"] == pytest.approx(0.5)


def test_level_up_is_attributed_to_the_previous_row_not_the_row_it_appears_on():
    """The off-by-one that would name the wrong branch as the level-banker.

    The level increments on row 3 -- meaning the frame row 3 observes is already at the new
    level -- so the action that banked it is row 2's. Here row 2 is a plan step and row 3 is
    an explorer action, so a correct implementation credits the PLAN branch and an off-by-one
    credits the EXPLORER. The two answers point at opposite conclusions."""
    rows = [
        _row(0, "explore.explorer", level=0, explorer="pending_drain", serve="navigation"),
        _row(1, "explore.explorer", level=0, explorer="pending_drain", serve="probe"),
        _row(2, "execute.plan_step", level=0, plan_epoch=1, plan_len=1),
        _row(3, "explore.explorer", level=1, explorer="pending_drain", serve="navigation"),
    ]
    a = analyse_cell(_cell(rows, levels_gained=1, events=[_induction()]), budget=4)
    assert a["n_level_ups_seen_in_rows"] == 1
    ev = a["level_up_events"][0]
    assert ev["observed_at_row"] == 3
    assert ev["causing_action_row"] == 2
    assert ev["causing_action_branch"] == "execute.plan_step"
    assert a["level_ups_from_plan_branch"] == 1


def test_level_up_credited_to_the_explorer_when_the_explorer_emitted_the_banking_action():
    """The mirror image of the previous test, so a hardcoded 'plan' answer cannot pass both.

    Same shape, but the action immediately before the level increment is an explorer probe.
    The credit must follow the branch, not the test author's expectation."""
    rows = [
        _row(0, "execute.plan_step", level=0, plan_epoch=1, plan_len=1),
        _row(1, "explore.explorer", level=0, explorer="frontier.pop_untested"),
        _row(2, "explore.explorer", level=1, explorer="pending_drain", serve="navigation"),
    ]
    a = analyse_cell(_cell(rows, levels_gained=1, events=[_induction()]), budget=3)
    ev = a["level_up_events"][0]
    assert ev["causing_action_row"] == 1
    assert ev["causing_action_branch"] == "explore.explorer"
    assert ev["causing_action_explorer_branch"] == "frontier.pop_untested"
    assert a["level_ups_from_plan_branch"] == 0


def test_level_up_before_the_pipelines_first_action_is_flagged_as_such():
    """The ordering claim, which is stronger than branch attribution and needs no model.

    Branch attribution says "an explorer action banked this"; someone can still argue the
    plan set up the state that made it possible. A level banked STRICTLY BEFORE the pipeline
    emitted its first action admits no such argument -- the pipeline had not acted yet. Both
    directions are asserted so the field cannot be a constant."""
    rows = [
        _row(0, "explore.explorer", level=0, explorer="depth_ride.pop_untested"),
        _row(1, "explore.explorer", level=1, explorer="pending_drain"),  # bank at row 0
        _row(2, "execute.plan_step", level=1, plan_epoch=1, plan_len=2),
        _row(3, "execute.plan_step", level=1, plan_epoch=1, plan_len=2),
        _row(4, "explore.explorer", level=2, explorer="pending_drain"),  # bank at row 3
    ]
    a = analyse_cell(_cell(rows, levels_gained=2, events=[_induction()]), budget=5)
    assert a["actions_before_first_plan_step"] == 2
    evs = a["level_up_events"]
    assert len(evs) == 2
    assert evs[0]["causing_action_row"] == 0 and evs[0]["before_first_plan_action"] is True
    assert evs[1]["causing_action_row"] == 3 and evs[1]["before_first_plan_action"] is False
    assert a["level_ups_strictly_before_the_pipelines_first_action"] == 1


def test_level_up_in_an_episode_with_no_plan_action_at_all_is_counted_separately():
    """When the pipeline never emitted an action, `before_first_plan_action` has no defined
    value (there is no 'first plan action' to be before). It must be None rather than True,
    and the episode's banks must be counted under the explicit no-plan-anywhere field --
    otherwise the strongest-sounding number silently absorbs a different, weaker case."""
    rows = [
        _row(0, "explore.explorer", level=0, explorer="depth_ride.pop_untested"),
        _row(1, "explore.explorer", level=1, explorer="pending_drain"),
    ]
    a = analyse_cell(_cell(rows, levels_gained=1), budget=2)
    assert a["actions_before_first_plan_step"] is None
    assert a["level_up_events"][0]["before_first_plan_action"] is None
    assert a["level_ups_strictly_before_the_pipelines_first_action"] == 0
    assert a["level_ups_with_no_plan_action_anywhere_in_the_episode"] == 1


@pytest.mark.parametrize(
    ("events", "plan_rows", "banked", "expected"),
    [
        ([], 0, 0, "no_induction_ever_fired"),
        (
            [_induction(produced=False, trusted=False, planned=False)],
            0,
            0,
            "induction_fired_but_no_engine_produced",
        ),
        (
            [_induction(trusted=False, skipped="trust_below_threshold", planned=False)],
            0,
            0,
            "engine_produced_but_never_trusted_by_heldout_verifier",
        ),
        (
            [_induction(skipped="degenerate_goal_predicate", planned=False)],
            0,
            0,
            "engine_trusted_but_no_plan_ever_executed",
        ),
        ([_induction()], 2, 0, "plan_executed_but_level_not_gained"),
        ([_induction()], 2, 1, "banked_level"),
    ],
)
def test_where_it_is_lost_names_the_first_stage_that_failed(events, plan_rows, banked, expected):
    """The labels must PARTITION the episodes -- one terminal label each, assigned by the
    FIRST stage that failed. Overlapping labels would let the same episode be counted under
    two different diagnoses and make the tally meaningless as a place to attack."""
    rows = [_row(0, "explore.explorer", explorer="bootstrap_reset")]
    rows += [
        _row(1 + k, "execute.plan_step", plan_epoch=1, plan_len=plan_rows) for k in range(plan_rows)
    ]
    a = analyse_cell(_cell(rows, levels_gained=banked, events=events), budget=10)
    assert a["where_it_is_lost"] == expected


def test_a_crashed_cell_is_missing_not_a_zero():
    """MISSING IS NOT ZERO. A cell that never ran must contribute nothing to the numerator
    AND nothing to the denominator; folding it in as 0/0-actions would manufacture the null
    the measurement exists to test."""
    a = analyse_cell(
        {"game": "zz00", "replicate": 1, "missing_observation": True, "error": "worker_exit_1"},
        budget=400,
    )
    assert a["missing_observation"] is True
    assert "n_plan_derived" not in a, "a missing cell must not report a plan-derived count"
    agg = aggregate([a], budget=400)
    entry = agg["per_game"][0]
    assert entry["replicates_observed"] == 0
    assert entry["missing_observations"] == [{"replicate": 1, "error": "worker_exit_1"}]
    assert "plan_derived_fraction" not in entry


def test_wall_truncated_episode_below_the_prereg_floor_becomes_missing():
    """The PRE-REGISTERED truncation rule, exercised on both sides of its own threshold.

    An episode cut off by the wall clock having spent less than the floor of its action
    budget is a value never observed. Above the floor it is a real, if short, observation.
    Testing only one side would leave the threshold itself unverified."""
    rows = [
        _row(i, "explore.explorer", explorer="pending_drain", serve="navigation") for i in range(10)
    ]
    budget = int(10 / (WALL_TRUNCATION_MIN_BUDGET_FRACTION / 2))  # consumed fraction < floor

    truncated = analyse_cell(_cell(rows, timed_out=True), budget=budget)
    assert truncated["missing_observation"] is True
    assert truncated["wall_truncated_below_prereg_floor"] is True

    # Same episode, same wall-clock truncation, but it spent MOST of its budget: observed.
    kept = analyse_cell(_cell(rows, timed_out=True), budget=11)
    assert kept["missing_observation"] is False
    assert kept["actions_recorded"] == 10

    # And an episode that simply ENDED EARLY without timing out is always an observation:
    # that is the agent's own behaviour, not a measurement failure.
    short = analyse_cell(_cell(rows, timed_out=False), budget=budget)
    assert short["missing_observation"] is False


def test_unknown_branch_labels_are_surfaced_rather_than_bucketed():
    """If the policy grows a decision path the closed vocabulary does not know, the
    accounting has silently mis-attributed actions. That must be visible in the artifact,
    not swallowed into an 'other' bucket that nobody reads."""
    rows = [_row(0, "explore.explorer", explorer="bootstrap_reset"), _row(1, "brand.new.branch")]
    a = analyse_cell(_cell(rows), budget=2)
    assert a["unknown_top_branches"] == ["brand.new.branch"]


def test_pooled_aggregation_weights_by_actions_not_by_episode():
    """Shares are pooled over the SUM of observed budgets. A mean-of-fractions would let a
    12-action episode outvote a 400-action one, which on a per-ACTION question is the wrong
    denominator."""
    long_ep = analyse_cell(
        _cell([_row(i, "explore.explorer", explorer="pending_drain") for i in range(100)]),
        budget=100,
    )
    short_ep = analyse_cell(
        _cell(
            [_row(i, "execute.plan_step", plan_epoch=1, plan_len=4) for i in range(4)],
            events=[_induction()],
        ),
        budget=4,
    )
    agg = aggregate([long_ep, short_ep], budget=400)
    entry = agg["per_game"][0]
    assert entry["pooled_actions"] == 104
    assert entry["pooled_by_top_branch"]["explore.explorer"] == 100
    # abs=1e-6 because the aggregate rounds shares to 6 decimal places on the way out, so
    # the artifact carries 0.038462 rather than the unrounded 0.0384615...
    assert entry["pooled_by_top_branch_share"]["execute.plan_step"] == pytest.approx(
        4 / 104, abs=1e-6
    )
    # The naive mean-of-fractions would have been (0.0 + 1.0)/2 = 0.5 -- an order of
    # magnitude off. Asserted explicitly so a refactor back to it fails here.
    assert entry["pooled_by_top_branch_share"]["execute.plan_step"] < 0.05
