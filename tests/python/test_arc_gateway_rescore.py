"""Regression tests for the gateway-accurate ARC re-scorer.

Spec: REQ-HARNESS-5940 (gateway-charged action accounting for ARC score claims),
SCENARIO-HARNESS-5940-1 (a reset before a level-up costs score),
SCENARIO-HARNESS-5940-2 (the post-solve tail is free),
SCENARIO-HARNESS-5940-3 (a whole-run reset count cannot determine the
correction), SCENARIO-HARNESS-5940-4 (greedy understates the worst case at the
115 cap), SCENARIO-HARNESS-5940-5 (a zeroed baseline channel is refused, not
scored).

Each test pins a fact that was ACTUALLY measured or an incident that ACTUALLY
occurred on 2026-07-26, not a synthetic happy path (CLAUDE.md QA-Layer
Authenticity Discipline: "write the regression test for the exact
incident/counterexample that motivated the check").

Facts pinned here:
  * the installed gateway CHARGES a RESET an action (scorecard.py:701-704), so
    resets taken BEFORE a level-up inflate that level's squared denominator;
  * the POST-SOLVE TAIL costs exactly nothing, resets included
    (scorecard.py:178-183) -- the settled 2026-07-26 charge model;
  * driving `EnvironmentScoreCalculator` directly agrees with driving the FULL
    `Scorecard`/`Card`/`from_scorecard` chain -- neither is a paraphrase;
  * greedy-on-marginals is WRONG inside the 115 cap's flat region (the real
    sp80 seed-20260726 cell), which is why the bound uses an exact DP;
  * a missing per-level human baseline scores 0.0, never 1.0.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("arc_agi.scorecard", reason="installed arc_agi scorer is the oracle")

from arc_gateway_rescore import (  # noqa: E402
    _level_score,
    crosscheck_post_solve_tail_is_free,
    crosscheck_reset_charge,
    gateway_score_via_calculator,
    gateway_score_via_full_chain,
    greedy_worst_case,
    rescore_row,
    worst_case_allocation,
)


def test_reset_is_charged_an_action_by_the_full_installed_chain():
    """THE DEFECT. 5 resets before a 10-action level-up must lower the score."""
    got = crosscheck_reset_charge()
    assert got["reset_is_charged"] is True
    # level 1: 10 charged -> (10/10)^2*100 = 100, weight 1, 2 levels -> 100/3
    assert got["no_resets_score"] == pytest.approx(100.0 / 3.0, abs=1e-4)
    # with 5 resets: 15 charged -> (10/15)^2*100 = 44.444, /3
    assert got["five_resets_before_levelup_score"] == pytest.approx(
        ((10 / 15) ** 2 * 100) / 3.0, abs=1e-4
    )
    assert got["five_resets_before_levelup_score"] < got["no_resets_score"]


def test_post_solve_tail_costs_exactly_nothing_resets_included():
    """The settled charge model: an incomplete level scores 0.0 whatever it is charged."""
    got = crosscheck_post_solve_tail_is_free()
    assert got["tail_is_free"] is True
    assert got["tail_0_score"] == pytest.approx(got["tail_530_frames_incl_30_resets_score"])


def test_calculator_path_agrees_with_full_scorecard_chain():
    """Neither path may be a paraphrase of the other; they must agree numerically."""
    baselines = [7, 18, 44]
    level_offline = [15, 42]  # the real vc33 b4000 per-level costs
    tail = 3828
    via_calc, _ = gateway_score_via_calculator(baselines, level_offline, tail)

    kinds: list[str] = []
    levels: list[int] = []
    lvl = 0
    for cost in level_offline:
        kinds.extend(["ACTION"] * cost)
        levels.extend([lvl] * cost)
        lvl += 1
        levels[-1] = lvl
    kinds.extend(["ACTION"] * tail)
    levels.extend([lvl] * tail)
    via_chain = gateway_score_via_full_chain("vc33", baselines, kinds, levels)
    assert via_calc == pytest.approx(via_chain, abs=1e-9)


def test_greedy_is_wrong_inside_the_115_cap_and_dp_is_not():
    """The real sp80 seed-20260726 cell: a SUPERHUMAN level 1 sits at the 115 cap.

    27 agent actions vs a human baseline of 39 gives a raw (39/27)^2*100 = 208.6,
    capped to 115. Inside that flat region the marginal gain of one more reset is
    ZERO, so greedy refuses to allocate and reports the UNCHANGED score. The DP
    spends through the flat region and finds the real worst case. This is the
    counterexample that motivated using a DP for the bound.
    """
    baselines = [39, 58, 25, 148, 96, 152]
    level_offline = [27]
    assert ((39 / 27) ** 2) * 100 > 115, "precondition: the cap must be active"

    dp_score, dp_alloc = worst_case_allocation(baselines, level_offline, 50, 373)
    greedy_score, greedy_alloc = greedy_worst_case(baselines, level_offline, 50, 373)

    assert greedy_alloc == [0], "greedy sees a zero marginal and allocates nothing"
    assert dp_alloc == [50], "the DP spends the whole reset budget"
    assert dp_score < greedy_score, "the DP must find a strictly worse (lower) score"
    unchanged, _ = gateway_score_via_calculator(baselines, level_offline, 373)
    assert greedy_score == pytest.approx(unchanged), "greedy returns the uncorrected score"


def test_dp_never_reports_a_higher_score_than_greedy():
    """The DP is a minimiser; it can never be beaten by the greedy cross-check."""
    baselines = [7, 18, 44, 61, 131]
    for level_offline in ([15, 42], [15, 42, 300], [5], [5, 5, 5, 5]):
        for n_resets in (0, 1, 13, 115):
            dp, _ = worst_case_allocation(baselines, level_offline, n_resets, 100)
            gr, _ = greedy_worst_case(baselines, level_offline, n_resets, 100)
            assert dp <= gr + 1e-9


def test_zero_resets_leaves_the_score_exactly_unchanged():
    """The bound's BEST case must be an identity, not an approximation."""
    baselines = [7, 18, 44]
    offline, _ = gateway_score_via_calculator(baselines, [15, 42], 3828)
    worst, alloc = worst_case_allocation(baselines, [15, 42], 0, 3828)
    assert alloc == [0, 0]
    assert worst == pytest.approx(offline)


def test_missing_baseline_scores_zero_never_one():
    """A zero/absent baseline is 0.0 in the real scorer -- the inverse was a gameable hole."""
    assert _level_score(0, 10) == 0.0
    score, _ = gateway_score_via_calculator([0, 0], [10], 5)
    assert score == 0.0


def test_charged_actions_cannot_be_zero_divided():
    """A completed level charged 0 actions scores 0.0 rather than raising."""
    assert _level_score(10, 0) == 0.0


def test_rescore_row_flags_a_row_with_no_reset_count_as_unrescorable():
    """The 1713 cptb cells + 375 budget-sweep rows: convention declared, count absent.

    They must be reported as a coverage hole, never silently scored as if the
    reset count were zero (which would read as a clean null).
    """
    row = {
        "game": "r11l",
        "actions": 1956,
        "efficiency": 0.0,
        "action_count_convention": "resets_excluded_run_game_native",
        # no n_resets
    }
    res = rescore_row(row)
    assert res.rescorable is False
    assert res.reason == "no_reset_count_recorded"


def test_rescore_row_flags_missing_human_baselines():
    """A dead baseline channel must be REFUSED, not scored as agreement.

    A prior agent read `getattr(env, "baseline_actions")` when the field lives on
    `env.info`; both charge models then summed to 0.0 and the comparison read as
    "no difference". A row with zeroed baselines must be unrescorable.
    """
    row = {
        "game": "vc33",
        "actions": 400,
        "n_resets": 13,
        "level_up_actions": [15],
        "per_level": [{"level": 0, "agent_actions": 15, "human_actions": 0}],
    }
    res = rescore_row(row)
    assert res.rescorable is False
    assert res.reason == "no_human_baselines_in_row"


def test_rescore_row_reproduces_the_recorded_vc33_b400_cell():
    """A real recorded row: vc33 / seed 20260724 / b400, recorded efficiency 2.0897."""
    row = {
        "game": "vc33",
        "seed": 20260724,
        "budget": 400,
        "actions": 387,
        "n_frames": 400,
        "n_resets": 13,
        "efficiency": 2.0897,
        "level_up_actions": [15, 57],
        "per_level": [
            {"level": 0, "agent_actions": 15, "human_actions": 7, "completed": True},
            {"level": 1, "agent_actions": 42, "human_actions": 18, "completed": True},
            {"level": 2, "agent_actions": 330, "human_actions": 44, "completed": False},
            {"level": 3, "agent_actions": 0, "human_actions": 61, "completed": False},
            {"level": 4, "agent_actions": 0, "human_actions": 131, "completed": False},
            {"level": 5, "agent_actions": 0, "human_actions": 34, "completed": False},
            {"level": 6, "agent_actions": 0, "human_actions": 152, "completed": False},
        ],
    }
    res = rescore_row(row)
    assert res.rescorable is True
    assert res.n_levels_completed == 2
    assert res.level_offline == [15, 42]
    assert res.offline_score == pytest.approx(2.0897, abs=1e-3)
    # the worst case must be strictly worse, and the delta positive (optimism)
    assert res.worst_score < res.offline_score
    assert res.delta_worst > 0


def test_exact_attribution_path_uses_per_level_resets_when_present():
    """With `resets_before_levelups` recorded, the score is EXACT, not bounded.

    Pins the measured vc33 seed-20260726 cell: 12 whole-run resets, of which 1
    precede level 1 and 12 precede level 2 (cumulative), giving a gateway score
    strictly between the offline score and the worst case.
    """
    row = {
        "game": "vc33",
        "seed": 20260726,
        "budget": 400,
        "actions": 388,
        "n_frames": 400,
        "n_resets": 12,
        "level_up_actions": [15, 60],
        "resets_before_levelups": [1, 12],
        "per_level": [
            {"level": 0, "agent_actions": 15, "human_actions": 7, "completed": True},
            {"level": 1, "agent_actions": 45, "human_actions": 18, "completed": True},
            {"level": 2, "agent_actions": 328, "human_actions": 44, "completed": False},
            {"level": 3, "agent_actions": 0, "human_actions": 61, "completed": False},
            {"level": 4, "agent_actions": 0, "human_actions": 131, "completed": False},
            {"level": 5, "agent_actions": 0, "human_actions": 34, "completed": False},
            {"level": 6, "agent_actions": 0, "human_actions": 152, "completed": False},
        ],
    }
    res = rescore_row(row)
    assert res.exact_score is not None
    assert res.exact_resets_before_levelups == [1, 12]
    # exact must sit inside the bound [worst, offline] -- that is what a bound means
    assert res.worst_score <= res.exact_score <= res.offline_score
    # and it must be strictly below the offline score: resets DID precede level-ups
    assert res.exact_score < res.offline_score


def test_frames_equals_offline_actions_plus_resets_on_every_recorded_row():
    """The identity the whole correction rests on, asserted over the real corpus.

    Verified 2026-07-26 over all 1401 persisted early-stop rows with ZERO
    exceptions. If this ever fails, `n_frames` is no longer the whole-run
    gateway-charged total and the bound's arithmetic is invalid.
    """
    import glob
    import json

    files = [
        f
        for f in glob.glob(os.path.join(REPO, "results/early_stop_sweep_20260726/rows_*.json"))
        if "exact_attribution" not in f
    ]
    assert files, "the recorded corpus must be present for this identity check"
    checked = 0
    for f in files:
        for r in json.loads(open(f).read()).get("rows") or []:
            if r.get("n_frames") is None or r.get("n_resets") is None:
                continue
            assert int(r["n_frames"]) == int(r["actions"] or 0) + int(r["n_resets"]), (
                f"identity broken in {os.path.basename(f)} for {r.get('game')}"
            )
            checked += 1
    assert checked > 1000, f"expected the full corpus, only checked {checked} rows"
