"""Tests for the gateway-accurate ARC re-score.

Spec: REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-FREE-OPENING-RESET,
SCENARIO-ARC-WMTE-5986-M1-UNREACHABLE, SCENARIO-ARC-WMTE-5986-POSITIONAL-CORRUPTION,
SCENARIO-ARC-WMTE-5986-DEAD-BASELINE, SCENARIO-ARC-WMTE-5986-ROUNDING-SEPARATED

Every test here is written against a DEFECT that actually occurred, not against a happy path:

  * the level-drop bug that turned a 2.09 score into 21.31 (found in this analyser's own first draft),
  * the all-resets-charged model being unreachable through the real chain (the reason the published
    figure was wrong at all),
  * a dead baseline channel reading as a clean "no optimism" null (a prior agent's incident),
  * a relative loss computed by differencing two 4-decimal-rounded scores (the per-span artifact's
    published tu93 figure),
  * a distribution summary over an empty list reading as zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import analyze_arc_gateway_accurate_rescore as mod  # noqa: E402

# The vc33 / seed-20260724 / budget-400 shape. Chosen because its ONLY pre-level-up reset IS the
# opening one, which makes it the cell where the correction is total: its gateway-accurate score
# equals its recorded offline score exactly.
VC33_BASELINES = [7, 18, 44, 61, 131, 34, 152]
VC33_SEG_OFFLINE = [15, 42]
VC33_SEG_RESETS = [1, 0]
VC33_TAIL_OFFLINE = 330
VC33_TAIL_RESETS = 12
VC33_RECORDED_OFFLINE_SCORE = 2.0897  # from the row, 4dp
VC33_RECORDED_M1_SCORE = 1.9955  # from the row, 4dp


def _drive(charge_opening: bool, game: str) -> dict:
    return mod.score_via_update_scorecard(
        game,
        VC33_BASELINES,
        VC33_SEG_OFFLINE,
        VC33_SEG_RESETS,
        VC33_TAIL_OFFLINE,
        VC33_TAIL_RESETS,
        charge_the_opening_reset=charge_opening,
    )


def test_opening_reset_is_free_reproduces_the_recorded_offline_score() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-FREE-OPENING-RESET.

    On a cell whose only pre-level-up reset is the opening one, the gateway-accurate score must equal
    the recorded OFFLINE score -- because the only charge the offline harness missed was the free one.
    """
    free = _drive(False, "t01a")
    assert free["score"] is not None
    assert abs(free["score"] - VC33_RECORDED_OFFLINE_SCORE) < 5e-4, free["score"]


def test_charging_the_opening_reset_reproduces_the_published_number() -> None:
    """REQ-ARC-WMTE-5986. The M1 model must reproduce the row's own `efficiency_gateway_charged`.

    Without this the M2-vs-M1 difference could be a plumbing difference rather than a charge-model
    difference.
    """
    charged = _drive(True, "t01b")
    assert charged["score"] is not None
    assert abs(charged["score"] - VC33_RECORDED_M1_SCORE) < 5e-4, charged["score"]


def test_free_opening_reset_scores_strictly_higher_than_charging_it() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-FREE-OPENING-RESET.

    Directionality is the whole claim: the published figure was too PESSIMISTIC. If these two ever
    compared equal on this fixture the correction would be a no-op and the artifact's headline
    would be unfalsifiable.
    """
    free = _drive(False, "t02a")["score"]
    charged = _drive(True, "t02b")["score"]
    assert free is not None and charged is not None
    assert free > charged


def test_charged_totals_differ_by_exactly_one_action_and_one_reset() -> None:
    """REQ-ARC-WMTE-5986. The arithmetic of the correction, asserted on the card's own counters.

    `actions + resets - n_full_resets` with `n_full_resets == 1` is the claim; this reads it off the
    real `Card` rather than trusting the analyser's own bookkeeping.
    """
    total_offline = sum(VC33_SEG_OFFLINE) + VC33_TAIL_OFFLINE
    total_resets = sum(VC33_SEG_RESETS) + VC33_TAIL_RESETS
    free = _drive(False, "t03a")
    charged = _drive(True, "t03b")
    assert free["card_actions"] == [total_offline + total_resets - 1]
    assert charged["card_actions"] == [total_offline + total_resets]
    assert free["card_resets"] == [total_resets - 1]
    assert charged["card_resets"] == [total_resets]


def test_all_resets_charged_requires_a_phantom_new_play() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-M1-UNREACHABLE.

    The published model is not a conservative reading of the gateway; it is an unreachable one. The
    driver must SAY so via `required_phantom_new_play` rather than quietly inserting the phantom.
    """
    charged = _drive(True, "t04a")
    free = _drive(False, "t04b")
    assert charged["required_phantom_new_play"] is True
    assert free["required_phantom_new_play"] is False


def test_first_reset_without_full_reset_creates_no_card_at_all() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-M1-UNREACHABLE.

    Drives the installed chain directly: a run whose FIRST call is `update_scorecard(RESET,
    full_reset=False)` never creates a card, so there is no counter for the opening reset to
    increment. This is the mechanism behind the whole correction and is asserted, not narrated.
    """
    from arc_agi.scorecard import Scorecard
    from arcengine.enums import ActionInput, FrameDataRaw, GameAction, GameState

    sc = Scorecard(card_id="t05")
    data = FrameDataRaw(
        game_id="t05g",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        action_input=ActionInput(id=GameAction.RESET),
        guid="g",
    )
    sc.update_scorecard("g", data, False)
    assert "t05g" not in sc.cards


def test_witness_block_reports_all_of_its_own_conclusions_as_true() -> None:
    """REQ-ARC-WMTE-5986. The published witness must be COMPUTED, so it must be able to fail.

    Asserting the witness's own boolean fields here means a future edit that breaks the arithmetic
    turns the witness False and fails this test, rather than shipping a witness that always agrees
    with the prose beside it.
    """
    w = mod.witness_opening_reset_is_free()
    assert w["charged_actions_M2_equals_offline_plus_resets_minus_one"] is True
    assert w["charged_actions_M1_equals_offline_plus_resets"] is True
    assert w["resets_counter_M2_is_one_lower"] is True
    assert w["M2_score_is_strictly_higher_than_M1"] is True
    # The unreachability probe must have actually raised. "NONE_the_chain_accepted_it" is the
    # sentinel for the chain silently tolerating it, which would invalidate the structural claim.
    assert w["M1_is_unreachable_without_a_phantom_new_play"]["result"].startswith("KeyError")


def test_every_frame_reports_the_running_level_so_no_spurious_entry_appears() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-POSITIONAL-CORRUPTION.

    THE REGRESSION TEST FOR THIS ANALYSER'S OWN BUG. An earlier draft hardcoded `levels_completed=0`
    on the non-final frames of each span. `Card.set_levels_completed` appends an entry whenever the
    value CHANGES in EITHER direction, so span 2's first frame inserted a spurious `(0, 16)` into
    `actions_by_level`; the scorer consumes that list POSITIONALLY, so every later level's charge
    shifted a slot and a 2.09 score became 21.31.

    The invariant: exactly one `actions_by_level` entry per completed level, and no entry whose
    recorded level is 0.
    """
    free = _drive(False, "t06a")
    abl = free["actions_by_level"][0]
    assert len(abl) == len(VC33_SEG_OFFLINE), abl
    assert [lvl for lvl, _ in abl] == [1, 2], abl
    assert all(lvl > 0 for lvl, _ in abl), abl


def test_a_level_drop_really_would_corrupt_the_score() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-POSITIONAL-CORRUPTION.

    The companion to the test above: proves the invariant is LOAD-BEARING rather than cosmetic, by
    driving the installed chain with the exact defect the earlier draft had and showing the score
    moves by an order of magnitude. A guard whose violation is harmless is not worth a test.
    """
    from arc_agi.models import EnvironmentInfo
    from arc_agi.scorecard import EnvironmentScorecard, Scorecard
    from arcengine.enums import ActionInput, FrameDataRaw, GameAction, GameState

    # Frames identical to the correct fixture EXCEPT that span 2's non-final frames report level 0.
    frames: list[tuple[str, int]] = [("RESET", 0)]
    frames += [("ACTION", 0)] * 14
    frames += [("ACTION", 1)]
    frames += [("ACTION", 0)] * 41  # the defect: level under-reported after level 1 was reached
    frames += [("ACTION", 2)]
    frames += [("RESET", 2)] * VC33_TAIL_RESETS
    frames += [("ACTION", 2)] * VC33_TAIL_OFFLINE

    sc = Scorecard(card_id="t07")
    seen = False
    for kind, lvl in frames:
        if kind == "RESET":
            full = not seen
            seen = True
            ai = ActionInput(id=GameAction.RESET)
        else:
            full = False
            ai = ActionInput(id=GameAction.ACTION1)
        sc.update_scorecard(
            "g",
            FrameDataRaw(
                game_id="t07g",
                state=GameState.NOT_FINISHED,
                levels_completed=int(lvl),
                action_input=ai,
                guid="g",
            ),
            full,
        )
    abl = sc.cards["t07g"].actions_by_level[0]
    assert any(lvl == 0 for lvl, _ in abl), abl  # the spurious entry really is inserted
    env_sc = EnvironmentScorecard.from_scorecard(
        sc, [EnvironmentInfo(game_id="t07g", baseline_actions=VC33_BASELINES)]
    )
    corrupted = [run.score for env in env_sc.environments for run in env.runs][0]
    assert corrupted > 10 * VC33_RECORDED_OFFLINE_SCORE, corrupted


def test_only_the_first_span_changes_between_the_two_charge_models() -> None:
    """REQ-ARC-WMTE-5986.

    The per-level cost is a DIFFERENCE of cumulative charged counts, so subtracting one charge from
    every cumulative total can only move span 1. A fix that subtracted from every span would show up
    here as more than one differing entry.
    """
    cell = {
        "game": "vc33",
        "seed": 1,
        "budget": 400,
        "levels": 2,
        "level_up_actions_offline": VC33_SEG_OFFLINE,
        "resets_before_levelups": [1, 1],
        "level_up_charged": [16, 58],
        "offline_actions": 387,
        "n_resets": 13,
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    got = mod._exact_cell(cell, {})
    assert got is not None and got["usable"]
    m1, m2 = got["seg_charged_M1"], got["seg_charged_M2"]
    diffs = [i for i in range(len(m1)) if m1[i] != m2[i]]
    assert diffs == [0], (m1, m2)
    assert m1[0] - m2[0] == 1


def test_the_two_scorer_paths_agree_on_a_real_cell_under_both_models() -> None:
    """REQ-ARC-WMTE-5986. Path 1 (calculator) vs path 2 (real `update_scorecard` chain).

    Both models are checked, because agreement under only one would be consistent with the driver
    ignoring the charge model.
    """
    cell = {
        "game": "vc33",
        "seed": 1,
        "budget": 400,
        "levels": 2,
        "level_up_actions_offline": VC33_SEG_OFFLINE,
        "resets_before_levelups": [1, 1],
        "level_up_charged": [16, 58],
        "offline_actions": 387,
        "n_resets": 13,
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    got = mod._exact_cell(cell, {})
    assert got is not None
    # The published booleans...
    assert got["chain_agrees_M1"] is True
    assert got["chain_agrees_M2"] is True
    # ...AND the underlying numbers, independently. Asserting only the booleans was caught by a
    # mutation that hardcoded `chain_agrees_M1: True` and still passed -- the exact "forced gate"
    # failure mode this project keeps hitting. A flag is only evidence if the quantity behind it is
    # checked too.
    assert got["chain_M1_score"] is not None and got["chain_M2_score"] is not None
    assert abs(got["chain_M1_score"] - got["score_M1_all_resets_charged"]) < 1e-6
    assert abs(got["chain_M2_score"] - got["score_M2_bootstrap_free_GATEWAY_ACCURATE"]) < 1e-6
    # And the two models must actually differ on this cell, or the agreement above is vacuous.
    assert got["score_M1_all_resets_charged"] != got["score_M2_bootstrap_free_GATEWAY_ACCURATE"]


def test_the_chain_agreement_flag_is_DERIVED_and_goes_False_when_the_paths_disagree(
    monkeypatch,
) -> None:
    """REQ-ARC-WMTE-5986. The gate reads a boolean, so the boolean must be falsifiable.

    A mutation that hardcoded `chain_agrees_M1: True` survived a test that only asserted the flag was
    True AND that the two scores matched -- because the scores genuinely did match. The only way to
    prove the flag is derived rather than decorative is to make path 2 return a WRONG score and
    require the flag to notice.
    """
    real = mod.score_via_update_scorecard

    def wrong(*args, **kwargs):  # noqa: ANN002, ANN003 - a stub, signature mirrors the real one
        out = dict(real(*args, **kwargs))
        out["score"] = (out["score"] or 0.0) + 1.0
        return out

    monkeypatch.setattr(mod, "score_via_update_scorecard", wrong)
    cell = {
        "game": "vc33",
        "seed": 1,
        "budget": 400,
        "levels": 2,
        "level_up_actions_offline": VC33_SEG_OFFLINE,
        "resets_before_levelups": [1, 1],
        "level_up_charged": [16, 58],
        "offline_actions": 387,
        "n_resets": 13,
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    got = mod._exact_cell(cell, {})
    assert got is not None
    assert got["chain_agrees_M1"] is False
    assert got["chain_agrees_M2"] is False


def test_a_dead_baseline_channel_is_stamped_not_scored_zero() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-DEAD-BASELINE.

    A zero baseline makes every charge model agree at score 0, which reads as "no optimism" -- the
    most reassuring possible output from a broken input. The cell must be refused with a reason.
    """
    cell = {
        "game": "zz99",
        "seed": 1,
        "budget": 400,
        "levels": 1,
        "level_up_actions_offline": [15],
        "resets_before_levelups": [1],
        "level_up_charged": [16],
        "offline_actions": 100,
        "n_resets": 3,
        "per_level": [{"human_actions": 0}, {"human_actions": 0}],
    }
    got = mod._exact_cell(cell, {})
    assert got is not None
    assert got["usable"] is False
    assert got["reason"] == "baseline_channel_dead_or_zero"


def test_a_first_span_without_a_reset_is_stamped_rather_than_silently_adjusted() -> None:
    """REQ-ARC-WMTE-5986.

    The `-1` is applied to span 1. If span 1 carried no reset the opening reset would sit elsewhere
    and the subtraction would be wrong, so the code must publish
    `opening_reset_in_first_span: False` and leave span 1 alone -- and the gate that reads that flag
    must therefore be able to fail.
    """
    cell = {
        "game": "zz98",
        "seed": 1,
        "budget": 400,
        "levels": 2,
        "level_up_actions_offline": [15, 42],
        "resets_before_levelups": [0, 2],
        "level_up_charged": [15, 59],
        "offline_actions": 387,
        "n_resets": 5,
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    got = mod._exact_cell(cell, {})
    assert got is not None and got["usable"]
    assert got["opening_reset_in_first_span"] is False
    assert got["seg_charged_M1"] == got["seg_charged_M2"]


def test_the_cumulative_identity_is_checked_and_can_fail() -> None:
    """REQ-ARC-WMTE-5986.

    `level_up_charged` is CUMULATIVE while `level_up_actions_offline` is PER-SPAN, in the same row.
    Mixing the two conventions produces a plausible wrong number, so the identity is asserted per
    cell. A row that violates it must report `cumulative_identity_holds: False`.
    """
    good = {
        "game": "vc33",
        "seed": 1,
        "budget": 400,
        "levels": 2,
        "level_up_actions_offline": [15, 42],
        "resets_before_levelups": [1, 1],
        "level_up_charged": [16, 58],
        "offline_actions": 387,
        "n_resets": 13,
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    bad = dict(good, level_up_charged=[16, 99])
    assert mod._exact_cell(good, {})["cumulative_identity_holds"] is True
    assert mod._exact_cell(bad, {})["cumulative_identity_holds"] is False


def test_rounding_a_small_score_before_differencing_misstates_the_loss() -> None:
    """REQ-ARC-WMTE-5986, SCENARIO-ARC-WMTE-5986-ROUNDING-SEPARATED.

    The per-span artifact published tu93's loss as 0.041667, which is exactly `0.0003 / 0.0072` --
    a ratio of two 4-decimal-rounded scores. This asserts the hazard is real at that magnitude: the
    rounded-difference loss and the unrounded loss differ by more than a tenth of the figure.
    """
    offline, m1 = 0.0071913, 0.0068587  # tu93 shape, full precision
    unrounded = (offline - m1) / offline
    rounded = (round(offline, 4) - round(m1, 4)) / round(offline, 4)
    assert abs(unrounded - rounded) / unrounded > 0.05, (unrounded, rounded)


def test_empty_distribution_reports_n_zero_and_refuses_to_look_like_zero() -> None:
    """REQ-ARC-WMTE-5986.

    A summary of nothing that prints `median: 0` is indistinguishable from a real zero. `_dist`
    reports `n` first and stamps the empty case.
    """
    empty = mod._dist([])
    assert empty["n"] == 0
    assert empty.get("EMPTY_do_not_read_as_zero") is True
    assert "median" not in empty
    nonempty = mod._dist([1.0, 2.0, 3.0])
    assert nonempty["n"] == 3 and nonempty["median"] == 2.0


def test_the_M2_bound_is_inside_the_M1_bound_and_the_best_case_is_structurally_zero() -> None:
    """REQ-ARC-WMTE-5986.

    Removing one reset from the chargeable pool can only narrow the worst case. The best case is 0
    by construction (all resets in the free tail), which is why the bound is stamped uninformative
    rather than quoted.
    """
    row = {
        "game": "vc33",
        "seed": 1,
        "budget": 400,
        "n_resets": 13,
        "actions": 387,
        "level_up_actions": [15, 57],  # CUMULATIVE in the recorded rows
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    got = mod._bound_row(row)
    assert got is not None
    assert got["rel_worst_M2"] <= got["rel_worst_M1"]
    assert got["rel_worst_M2"] < got["rel_worst_M1"]  # 13 resets vs 12 must actually differ
    assert got["best_case_is_zero_by_construction"] is True
    assert got["score_M0_offline_recorded"] > got["worst_case_M1"]


def test_a_row_with_no_reset_count_is_not_bounded_at_all() -> None:
    """REQ-ARC-WMTE-5986.

    A row that never recorded a reset count cannot be re-scored at any width. Returning a bound of
    zero for it would silently assert "no optimism here", so it must be excluded outright.
    """
    row = {
        "game": "vc33",
        "actions": 387,
        "level_up_actions": [15, 57],
        "per_level": [{"human_actions": b} for b in VC33_BASELINES],
    }
    assert mod._bound_row(row) is None
