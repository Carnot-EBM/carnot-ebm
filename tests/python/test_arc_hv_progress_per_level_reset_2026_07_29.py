"""REQ-ARC-WMTE-6045: `hv_progress` must reset its baseline at every level-up.

THE INCIDENT. `run_bounded_progress` tracked `best_hv` as a running minimum over the WHOLE run
and never reset it when the agent advanced a level, so a later level's board distance was scored
against the FIRST level's starting distance. The frame returned by the step that COMPLETES a
level is already the NEXT level's opening board, so the verifier is never read at a solved
state, the global minimum stays pinned at `start_hv`, and the run scores 0.0.

vc33 is the exemplar and it is committed evidence: 2 banked levels -- the most of any cell in
the 24-cell 2026-07-29 retention corpus -- `start_hv == best_hv == 18.0`, `hv_progress == 0.0`.
The best run in the corpus recorded as having made no progress at all. Both tests below that
touch real data read `results/` READ-ONLY; historical artifacts are never rewritten.
"""

from __future__ import annotations

import json
import os

import pytest

from carnot.agentic.arc_actions_to_progress import per_level_hv_progress

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETENTION_CELLS = os.path.join(REPO, "results", "arc_engine_retention_20260729", "cells")


# ---- SCENARIO-ARC-WMTE-6045-1: a completed level scores 1.0, not the stale global minimum ----
def test_completed_levels_score_one_and_final_level_scores_against_its_own_start() -> None:
    """The vc33 shape: two levels completed, ended on a third with no measured improvement.

    Under the old global-minimum rule this run scored 0.0. The completed levels are credited by
    the LEVEL COUNTER (the win oracle), so their 1.0 does not depend on the verifier at all.
    """
    hv_per_level = {
        0: {"start": 18.0, "best": 18.0},  # completed -> 1.0 regardless of these readings
        1: {"start": 25.0, "best": 25.0},  # completed -> 1.0
        2: {"start": 30.0, "best": 30.0},  # still in progress, no improvement -> 0.0
    }
    per_level, best = per_level_hv_progress(hv_per_level, [0, 1, 2], True)
    assert per_level == {"0": 1.0, "1": 1.0, "2": 0.0}
    assert best == pytest.approx(1.0)


def test_final_level_partial_progress_is_scored_against_its_own_opening_board() -> None:
    """A level-2 board that closed half its own distance must read 0.5, not level 1's fraction."""
    hv_per_level = {0: {"start": 100.0, "best": 40.0}, 1: {"start": 20.0, "best": 10.0}}
    per_level, best = per_level_hv_progress(hv_per_level, [0, 1], True)
    # Level 0 completed -> 1.0. Level 1 closed 10 of its own 20 -> 0.5. NOT (100-10)/100 = 0.9,
    # which is what a never-reset global baseline would have produced.
    assert per_level == {"0": 1.0, "1": 0.5}
    assert best == pytest.approx(1.0)


def test_single_level_run_is_unchanged_by_the_fix() -> None:
    """A run that never levels up must produce exactly the old number -- no silent drift.

    This is the regression guard: the fix must only change runs that actually levelled up.
    """
    per_level, best = per_level_hv_progress({0: {"start": 40.0, "best": 20.0}}, [0], True)
    assert per_level == {"0": 0.5}
    assert best == pytest.approx(0.5)


# ---- SCENARIO-ARC-WMTE-6045-2: the measurability guard applies to the final level only -------
def test_immovable_verifier_omits_the_final_level_but_still_credits_completed_ones() -> None:
    """An unmeasurable verifier cannot evidence PARTIAL progress -- but a level-up is a level-up.

    Recording the final level as 0.0 here is what let instrument floors be averaged into means.
    Dropping the COMPLETED level's 1.0 as well would be the opposite error: that credit rests on
    the level counter, not on the verifier.
    """
    hv_per_level = {0: {"start": 0.0, "best": 0.0}, 1: {"start": 0.0, "best": 0.0}}
    per_level, best = per_level_hv_progress(hv_per_level, [0, 1], False)
    assert per_level == {"0": 1.0}
    assert best == pytest.approx(1.0)


def test_no_readings_at_all_yields_no_entries_and_no_best() -> None:
    per_level, best = per_level_hv_progress({}, [], True)
    assert per_level == {}
    assert best is None


def test_measurable_none_is_treated_as_permissive_not_as_false() -> None:
    """`hv_progress_measurable` is None on legacy rows; None must not silently drop the level.

    The guard is `is not False` on purpose: an UNKNOWN measurability is not evidence of an
    immovable verifier, and treating it as one would delete real observations.
    """
    per_level, best = per_level_hv_progress({0: {"start": 10.0, "best": 4.0}}, [0], None)
    assert per_level == {"0": 0.6}
    assert best == pytest.approx(0.6)


def test_zero_and_small_start_do_not_divide_by_zero() -> None:
    """`max(abs(start), 1.0)` is the denominator, so a 0.0 start cannot blow up."""
    per_level, _ = per_level_hv_progress({0: {"start": 0.0, "best": 0.0}}, [0], True)
    assert per_level == {"0": 0.0}
    per_level, _ = per_level_hv_progress({0: {"start": 0.5, "best": 0.0}}, [0], True)
    assert per_level == {"0": 0.5}


def test_negative_progress_is_floored_at_zero() -> None:
    """`best` should never exceed `start`, but if it does the metric must not go negative."""
    per_level, _ = per_level_hv_progress({0: {"start": 10.0, "best": 12.0}}, [0], True)
    assert per_level == {"0": 0.0}


def test_a_level_seen_without_a_reading_is_skipped_not_crashed() -> None:
    per_level, best = per_level_hv_progress({0: {"start": 8.0, "best": 2.0}}, [0, 1], True)
    # Level 1 was entered but the verifier never returned a value for it. Level 0 is therefore
    # not the final level and is credited 1.0; level 1 contributes nothing.
    assert per_level == {"0": 1.0}
    assert best == pytest.approx(1.0)


# ---- SCENARIO-ARC-WMTE-6045-3: the committed vc33 cell is the regression case ---------------
def test_committed_vc33_cell_shows_the_defect_this_fix_addresses() -> None:
    """Pin the historical defect to real committed data so the fix has a documented origin.

    Read-only. This asserts the OLD artifact still shows the bad number (it must -- never-prune
    forbids rewriting it) and that the fix, applied to that run's shape, produces 1.0 instead.
    """
    path = os.path.join(RETENTION_CELLS, "ret0__vc33__s1.json")
    assert os.path.exists(path), f"committed evidence missing: {path}"
    with open(path) as fh:
        res = json.load(fh)["result"]

    # The historical record, unmodified: 2 levels gained yet 0.0 progress.
    assert res["levels_gained"] == 2
    assert res["start_hv"] == pytest.approx(18.0)
    assert res["best_hv"] == pytest.approx(18.0)
    assert res["hv_progress"] == pytest.approx(0.0)

    # The same run under the per-level rule: levels 0 and 1 were completed, so at least one
    # level scores 1.0 no matter what the verifier read.
    n_levels = res["levels_gained"] + 1
    seen = list(range(n_levels))
    per_level, best = per_level_hv_progress(
        {lvl: {"start": 18.0, "best": 18.0} for lvl in seen}, seen, True
    )
    assert best == pytest.approx(1.0)
    assert per_level == {"0": 1.0, "1": 1.0, "2": 0.0}


# ---- SCENARIO-ARC-WMTE-6045-4: a mid-run RESET must not mis-credit a level ------------------
def test_a_mid_run_reset_that_lowers_the_level_does_not_mis_credit() -> None:
    """`E3AgentPolicy` can emit a RESET, which drops the frame's level back down.

    `hv_level_seen` records levels in FIRST-ENTRY order, so a RESET back to an already-seen level
    does not re-append -- which means the last entry is the HIGHEST level ever entered, not the
    level the run happened to end on. That is the correct thing to key on: every level BELOW the
    maximum was necessarily completed (you cannot reach level k without completing k-1), and the
    maximum is the only one that was not. So the credit stays right even though the run's final
    frame is on a lower level.

    Pinned because it is non-obvious and a "fix" that keyed on the last OBSERVED level instead
    would start crediting a level the agent never completed.
    """
    # Entered 0, levelled up to 1, then RESET back to 0 and finished there.
    per_level, best = per_level_hv_progress(
        {0: {"start": 10.0, "best": 2.0}, 1: {"start": 20.0, "best": 20.0}}, [0, 1], True
    )
    # Level 0 WAS completed (the agent reached level 1), so 1.0 is correct. Level 1 is the
    # highest entered and was not completed, so the verifier scores it: no improvement -> 0.0.
    assert per_level == {"0": 1.0, "1": 0.0}
    assert best == pytest.approx(1.0)


def test_a_skipped_level_does_not_break_the_credit_rule() -> None:
    """If the counter jumps 0 -> 2, level 1 is never observed and simply contributes nothing."""
    per_level, best = per_level_hv_progress(
        {0: {"start": 10.0, "best": 10.0}, 2: {"start": 5.0, "best": 5.0}}, [0, 2], True
    )
    assert per_level == {"0": 1.0, "2": 0.0}
    assert best == pytest.approx(1.0)


# ---- SCENARIO-ARC-WMTE-6045-5: credit comes from the LEVEL COUNTER, not from readings --------
def test_a_completed_level_with_no_verifier_reading_is_still_credited() -> None:
    """The gap the first version of this fix left open (found 2026-07-30 review).

    `hv_level_seen` is appended ONLY inside the branch where a hand-verifier reading was obtained.
    So a level the agent entered, COMPLETED and left without the verifier ever returning a value
    was invisible to this function and earned nothing -- while the docstring claimed the 1.0 was
    awarded "independent of whether the verifier is measurable at all". The claim was true only
    once at least one reading existed at that level.

    Here the agent went 0 -> 1 -> 2 and the verifier read at levels 0 and 2 but never at 1. Level 1
    was nonetheless completed: the counter left it. It must be credited 1.0.
    """
    hv_per_level = {0: {"start": 40.0, "best": 30.0}, 2: {"start": 12.0, "best": 6.0}}
    per_level, best = per_level_hv_progress(hv_per_level, [0, 2], True, levels_entered=[0, 1, 2])
    assert per_level["1"] == 1.0, "a completed level must not need a verifier reading to count"
    assert per_level["0"] == 1.0
    # Level 2 is the one the run ended on, so it is the only verifier-scored entry: 6/12 = 0.5.
    assert per_level["2"] == pytest.approx(0.5)
    assert best == pytest.approx(1.0)


def test_levels_entered_defaults_to_hv_level_seen_so_old_callers_are_unchanged() -> None:
    """The new parameter is additive: omitting it reproduces the previous behaviour exactly."""
    hv_per_level = {0: {"start": 8.0, "best": 2.0}, 1: {"start": 20.0, "best": 15.0}}
    assert per_level_hv_progress(hv_per_level, [0, 1], True) == per_level_hv_progress(
        hv_per_level, [0, 1], True, levels_entered=[0, 1]
    )


def test_credit_survives_an_unmeasurable_verifier_on_a_completed_level() -> None:
    """An immovable verifier must not erase a level the counter says was completed.

    `hv_measurable=False` suppresses the FINAL level's partial score (an immovable verifier cannot
    evidence partial progress) but must not touch completed levels, whose 1.0 never depended on
    the verifier in the first place.
    """
    per_level, best = per_level_hv_progress(
        {0: {"start": 8.0, "best": 8.0}}, [0], False, levels_entered=[0, 1, 2]
    )
    assert per_level == {"0": 1.0, "1": 1.0}
    assert "2" not in per_level  # final level, unmeasurable -> omitted, not recorded as 0.0
    assert best == pytest.approx(1.0)
