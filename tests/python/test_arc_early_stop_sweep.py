"""Tests for the early-stop grace sweep and its analyser.

REQ-ARC-WMTE-5983 / SCENARIO-ARC-WMTE-5983-DEAD-FLAG-IS-DETECTED-NOT-NULLED
REQ-ARC-WMTE-5983 / SCENARIO-ARC-WMTE-5983-EMPTY-PASS-REGION-IS-RESTATED-NOT-RUN
REQ-ARC-WMTE-5983 / SCENARIO-ARC-WMTE-5983-SAFETY-WITNESS-IS-NON-EMPTY

Each test targets a DEFECT CLASS this measurement could have shipped, not a happy path:

  * an arm whose swept parameter silently failed to apply (the dead-flag trap) reading as a clean
    null instead of as uninstrumented;
  * a safety gate passing over a movable set that is empty, i.e. arithmetically forced;
  * a score gate passing because no cell's level-up vector could move, reported as evidence;
  * a one-sided test hiding a reversal, and a "not significant" that no support could have rejected;
  * an any-seed union comparison, which shows a control failing against itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_arc_early_stop_sweep as esa  # noqa: E402

# The real `baselines_for` instantiates an offline-arcade game env to read its human baselines. That
# is correct for a measurement run and wrong for a unit test: it costs hundreds of MB per game and
# makes the synthetic game ids ("aa11") log errors. Stubbed for every test here; the two tests that
# are specifically ABOUT the baselines override this with their own monkeypatch.
_STUB_BASELINES = [20, 40, 60, 80, 100, 120, 140, 160]


@pytest.fixture(autouse=True)
def _stub_baselines(monkeypatch):
    monkeypatch.setattr(esa, "baselines_for", lambda game: list(_STUB_BASELINES))
    esa._BASELINE_CACHE.clear()
    yield
    esa._BASELINE_CACHE.clear()


def _row(game, seed, budget, grace, levels, actions, eff, lua, *, stopped=False, wall=1.0):
    """One synthetic sweep row. `lua` is the level-up checkpoint vector the scorer differences."""
    return {
        "game": game,
        "seed": seed,
        "budget": budget,
        "early_stop_grace": grace,
        "early_stop_grace_requested": grace,
        "early_stop_grace_applied": True,
        "ran": True,
        "levels": levels,
        "actions": actions,
        "efficiency": eff,
        "level_up_actions": list(lua),
        "inter_levelup_gaps": [lua[i] - lua[i - 1] for i in range(1, len(lua))],
        "actions_after_last_levelup": (actions - lua[-1]) if lua else actions,
        "reached_any_level": bool(lua),
        "early_stopped": stopped,
        "n_resets": 0,
        "n_frames": actions,
        "wall_s": wall,
        "gated_flags": {f"f{i}": True for i in range(8)},
    }


# --------------------------------------------------------------------------------------------
def test_never_firing_arm_is_reported_uninstrumented_not_as_a_clean_null():
    """THE DEAD-FLAG TRAP. `SUBMITTED_EARLY_STOP_GRACE` is read nowhere, so a wiring mistake makes a
    treatment arm byte-identical to the control. Without this check that arm reports "no
    regression, no score loss" -- a clean, meaningless pass."""
    rows = [
        _row("vc33", 1, 400, None, 2, 300, 2.0, [15, 57]),
        _row("vc33", 1, 400, 150, 2, 300, 2.0, [15, 57], stopped=False),  # never fired
    ]
    c = esa.instrumentation_census(rows)
    assert c["every_treatment_arm_fired_somewhere"] is False
    assert c["uninstrumented_arms"] == ["150"]
    assert c["control_never_fired"] is True

    rows[1]["early_stopped"] = True
    c2 = esa.instrumentation_census(rows)
    assert c2["every_treatment_arm_fired_somewhere"] is True
    assert c2["uninstrumented_arms"] == []


def test_parameter_that_failed_to_apply_is_visible_in_the_census():
    """An arm whose attribute-set silently failed reads back None, not the requested value."""
    rows = [
        _row("vc33", 1, 400, None, 1, 300, 1.0, [15]),
        {
            **_row("vc33", 1, 400, 150, 1, 300, 1.0, [15]),
            "early_stop_grace": None,
            "early_stop_grace_applied": False,
        },
    ]
    assert esa.instrumentation_census(rows)["grace_applied_all_rows"] is False


def test_safety_gate_over_an_empty_movable_set_is_uninterpretable_not_a_pass():
    """SCENARIO-ARC-WMTE-5983-SAFETY-WITNESS-IS-NON-EMPTY. Every cell tops out at ONE level, so no
    cell HAS a later level-up to forgo and no arm could have regressed. This is exp4524's defect:
    'levels preserved' was arithmetically forced there because both arms topped out at L1."""
    rows = []
    for g in ("aa11", "bb22"):
        rows.append(_row(g, 1, 400, None, 1, 300, 1.0, [15]))
        rows.append(_row(g, 1, 400, 150, 1, 170, 1.0, [15], stopped=True))
    out = esa.analyse_condition(rows, 400)["arms"]["150"]["gate_safety"]
    assert out["witness_movable_cells"] == 0
    assert out["verdict"] == "UNINTERPRETABLE_EMPTY_MOVABLE_SET_NO_CELL_COULD_REGRESS"
    assert out["interpretable"] is False


def test_safety_gate_with_a_real_movable_set_passes_and_names_the_at_risk_cells():
    """Same gate, non-empty movable set: one cell reaches L2 with a 42-action gap, which grace=150
    clears and grace=20 does not. The witness must distinguish the two.

    NOTE the grace=150 verdict: movable but NOT at risk, so its pass is arithmetically FORCED and is
    stamped `PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK`, not a plain pass. See the dedicated test below.
    """
    rows = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57]),
        _row("vc33", 1, 400, 150, 2, 210, 2.09, [15, 57], stopped=True),
        _row("vc33", 1, 400, 20, 1, 40, 0.5, [15], stopped=True),
    ]
    cond = esa.analyse_condition(rows, 400)["arms"]
    ok = cond["150"]["gate_safety"]
    assert ok["witness_movable_cells"] == 1 and ok["witness_at_risk_cells"] == 0
    assert ok["verdict"] == "PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK"
    assert ok["passed"] is True and ok["falsifiable"] is False
    bad = cond["20"]["gate_safety"]
    assert bad["witness_at_risk_cells"] == 1, "a 42-action gap must be AT RISK under grace=20"
    assert bad["verdict"] == "FAIL_LEVEL_REGRESSION"
    assert bad["regressing_games"] == ["vc33"]


def test_a_safety_PASS_with_no_at_risk_cell_is_stamped_UNFALSIFIABLE_not_a_clean_pass():
    """FAILURE MODE #2 IN ITS PASS DIRECTION (adversarial review, 2026-07-26).

    `movable > 0` is the right guard for a FAIL -- some cell had to be ABLE to regress. It is the
    WRONG guard for a PASS: for "this grace is SAFE" to have been falsifiable, some cell must have
    been AT RISK. With at_risk == 0 the pass says only "the grace exceeds every observed gap", which
    is the DEFINITION of in-sample safety for a fixed grace, not evidence for it. The live sweep hit
    this exactly: grace 1300 was reported as "the only safe firing point" at b2000 with at_risk == 0
    -- and it regresses 6 cells at b4000.

    Both branches are asserted so the stamp cannot be a constant.
    """
    # BY CONSTRUCTION: the only gap (42 actions) is far below the grace, so nothing was at risk.
    forced = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57]),
        _row("vc33", 1, 400, 200, 2, 260, 2.09, [15, 57], stopped=True),
    ]
    g = esa.analyse_condition(forced, 400)["arms"]["200"]["gate_safety"]
    assert g["passed"] is True
    assert g["falsifiable"] is False
    assert g["verdict"] == "PASS_UNFALSIFIABLE_NO_CELL_WAS_AT_RISK"

    # FALSIFIABLE: the gap (240 actions) EXCEEDS the grace, so the level COULD have been lost -- and
    # was not. That is a pass carrying information.
    real = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 255]),
        _row("vc33", 1, 400, 200, 2, 300, 2.09, [15, 255], stopped=True),
    ]
    g2 = esa.analyse_condition(real, 400)["arms"]["200"]["gate_safety"]
    assert g2["witness_at_risk_cells"] == 1
    assert g2["passed"] is True and g2["falsifiable"] is True
    assert g2["verdict"] == "PASS_AT_RISK_CELLS_SURVIVED"


def test_the_safety_gate_reports_the_reachable_p_floor_at_its_own_at_risk_support():
    """A safety verdict decided by 2 at-risk cells cannot reach p<=0.05 whatever those cells did: the
    smallest two-sided p at n=2 is 0.5. Reporting the verdict without the floor invites reading a
    single-observation existence claim as an estimate (failure mode #4's cousin)."""
    rows = [_row("vc33", s, 400, None, 2, 390, 2.09, [15, 255]) for s in (1, 2)] + [
        _row("vc33", s, 400, 200, 2, 300, 2.09, [15, 255], stopped=True) for s in (1, 2)
    ]
    p = esa.analyse_condition(rows, 400)["arms"]["200"]["gate_safety"]["witness_support_power"]
    assert p["n_at_risk_cells"] == 2
    assert p["p_min_reachable_two_sided_at_this_support"] == 0.5
    assert p["any_p_below_0.05_reachable"] is False


def test_the_safe_firing_window_is_computed_so_a_grid_that_misses_it_is_visible():
    """FAILURE MODE #1 IN A NEW DRESS (adversarial review, 2026-07-26). The b400 decision gate
    reported "no firing grace value is safe", but the tested grid (50/100/150/200/400) contained no
    point strictly between the largest at-risk gap (340.2 frames) and the largest post-level-up tail
    (372.3 frames) -- so the gate could only ever have said "none of the values TESTED". Both bounds
    come from the CONTROL arm, so grid adequacy is checkable before any treatment arm runs."""
    rows = [
        # movable cell: gap of 240 sets the SAFE lower bound
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 255]),
        _row("vc33", 1, 400, 100, 1, 120, 0.5, [15], stopped=True),
        # a longer tail sets the FIRING upper bound: level-up at 30, run to 390 -> tail 360
        _row("sp80", 1, 400, None, 1, 390, 4.76, [30]),
        _row("sp80", 1, 400, 100, 1, 140, 4.76, [30], stopped=True),
    ]
    w = esa.analyse_condition(rows, 400)["safe_firing_window"]
    assert w["lower_bound_frames_exclusive"] == 240.0
    assert w["upper_bound_frames_exclusive"] == 360.0
    assert w["window_is_non_empty"] is True
    # The only tested value (100) is BELOW the safe bound, so the grid misses the window entirely.
    assert w["tested_graces"] == [100]
    assert w["tested_graces_inside_the_window"] == []
    assert w["grid_spans_the_window"] is False
    assert w["unsafe_by_construction_tested_graces"] == [100]

    # Add a point inside the window and the grid becomes adequate.
    rows += [
        _row("vc33", 1, 400, 300, 2, 320, 2.09, [15, 255], stopped=True),
        _row("sp80", 1, 400, 300, 1, 340, 4.76, [30], stopped=True),
    ]
    w2 = esa.analyse_condition(rows, 400)["safe_firing_window"]
    assert w2["tested_graces_inside_the_window"] == [300]
    assert w2["grid_spans_the_window"] is True


def test_score_concentration_is_reported_at_the_same_level_as_the_score_delta():
    """FINDING 5 (adversarial review, 2026-07-26). Score deltas are SUMS, and at b400 four of 75
    cells carry 91.35% of the corpus efficiency total (one cell alone 52.58%). A delta framed as a
    corpus effect is, at that concentration, a statement about two or three cells. This must travel
    WITH the delta, not sit in a residual."""
    rows = [
        _row("big1", 1, 400, None, 1, 390, 8.0, [30]),
        _row("big1", 1, 400, 100, 1, 140, 8.0, [30], stopped=True),
        _row("small", 1, 400, None, 1, 390, 1.0, [30]),
        _row("small", 1, 400, 100, 1, 140, 1.0, [30], stopped=True),
        _row("zero", 1, 400, None, 0, 390, 0.0, []),
        _row("zero", 1, 400, 100, 0, 390, 0.0, []),
    ]
    sc = esa.analyse_condition(rows, 400)["score_concentration"]
    assert sc["control_efficiency_sum"] == 9.0
    assert sc["n_control_cells"] == 3
    assert sc["n_cells_with_nonzero_score"] == 2
    assert sc["top_1_share_pct"] == pytest.approx(88.89, abs=0.01)
    assert sc["max_single_cell_efficiency"] == 8.0
    assert sc["top_cells"][0]["game"] == "big1"


def test_duplicate_analysis_rows_are_deduped_and_the_count_is_reported():
    """A follow-up arm file re-running the SAME cells would double-count them in every corpus-level
    DENOMINATOR while the matched comparison kept one row -- silently halving percentages. Adding
    such a file is the obvious move (this sweep did it to probe the safe window), so the trap is
    live. Dedupe must also REPORT, because a silent dedupe is its own hidden decision."""
    a = _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57])
    b = dict(a)
    kept, rep = esa.dedupe_rows([a, b, _row("vc33", 1, 400, 100, 1, 120, 0.5, [15], stopped=True)])
    assert len(kept) == 2
    assert rep["n_duplicate_rows_dropped"] == 1
    assert rep["duplicates_are_consistent"] is True

    # A duplicate that DISAGREES is a determinism failure and must be surfaced, not silently resolved.
    c = dict(a)
    c["levels"] = 1
    _, rep2 = esa.dedupe_rows([a, c])
    assert rep2["n_duplicates_disagreeing_on_an_outcome"] == 1
    assert rep2["duplicates_are_consistent"] is False
    assert rep2["disagreeing_detail"][0]["diff"]["levels"] == [2, 1]


def test_the_wall_saving_is_split_into_measured_jitter_and_real_saving_with_no_threshold():
    """Wall clock is this mechanism's ENTIRE case and the noisiest thing measured, so an
    unattributable wall saving is the most quotable wrong number available. The live grace-350 arm at
    b400 is exactly that shape: 21 actions saved (0.072%) against a MEASURED 4.08s / 2.42% wall
    saving.

    THE TEST NEEDS NO CHOSEN THRESHOLD because the design supplies its own control: a cell where both
    arms took the SAME number of actions did identical work (LLM-off is deterministic in the seed --
    the reproduction check verifies it), so its wall difference is jitter with a TRUE value of zero.
    On the live data 74 of 75 cells did identical work and contribute +3.19s of spurious "saving",
    against 0.89s on the one cell that actually stopped early -- 78% noise. Three branches asserted so
    the stamp cannot be a constant."""
    # NOISE: the firing cell saves a little; a second, UNCHANGED cell contributes more "saving" than it.
    noisy = [
        _row("vc33", 1, 400, None, 1, 390, 1.0, [30], wall=10.0),
        _row("vc33", 1, 400, 350, 1, 380, 1.0, [30], stopped=True, wall=9.5),
        _row("sp80", 1, 400, None, 1, 390, 1.0, [30], wall=10.0),  # identical work in both arms
        _row("sp80", 1, 400, 350, 1, 390, 1.0, [30], stopped=False, wall=8.0),
    ]
    w = esa.analyse_condition(noisy, 400)["arms"]["350"]["benefit_actions_and_wall"][
        "wall_saving_attribution"
    ]
    assert w["n_cells_doing_identical_work"] == 1
    assert w["wall_s_jitter_on_identical_work_cells"] == 2.0
    assert w["wall_s_saved_on_cells_whose_actions_changed"] == 0.5
    assert w["jitter_share_of_measured_saving_pct"] == 80.0
    assert w["noise_dominated"] is True

    # REAL: the saving lives on the cell whose action count actually changed; no jitter elsewhere.
    real = [
        _row("vc33", 1, 400, None, 1, 400, 1.0, [30], wall=10.0),
        _row("vc33", 1, 400, 100, 1, 200, 1.0, [30], stopped=True, wall=5.0),
        _row("sp80", 1, 400, None, 1, 390, 1.0, [30], wall=10.0),
        _row("sp80", 1, 400, 100, 1, 390, 1.0, [30], stopped=False, wall=10.0),
    ]
    w2 = esa.analyse_condition(real, 400)["arms"]["100"]["benefit_actions_and_wall"][
        "wall_saving_attribution"
    ]
    assert w2["wall_s_jitter_on_identical_work_cells"] == 0.0
    assert w2["wall_s_saved_on_cells_whose_actions_changed"] == 5.0
    assert w2["noise_dominated"] is False
    # The heuristic ratio is still reported, and labelled as a heuristic in its own field name.
    assert w2["measured_over_attributable_ratio_heuristic"] == pytest.approx(1.0, abs=0.02)

    # An arm that changed NO cell's actions has a real part of exactly 0: every second is jitter.
    inert = [
        _row("vc33", 1, 400, None, 1, 390, 1.0, [30], wall=10.0),
        _row("vc33", 1, 400, 400, 1, 390, 1.0, [30], stopped=False, wall=9.0),
    ]
    w3 = esa.analyse_condition(inert, 400)["arms"]["400"]["benefit_actions_and_wall"][
        "wall_saving_attribution"
    ]
    assert w3["actions_saved"] == 0
    assert w3["wall_s_saved_on_cells_whose_actions_changed"] == 0.0
    assert w3["jitter_share_of_measured_saving_pct"] == 100.0
    assert w3["noise_dominated"] is True


def test_the_contention_control_keeps_every_process_and_never_reports_one_arbitrarily():
    """FOUND WHILE RECONCILING A 0.4% DISCREPANCY (2026-07-26): the changelog reported 80.1s of
    concurrent wall time and a rebuild produced 79.76s.

    Cause: the N concurrent processes each ran the SAME cells, and the old code collapsed them with
    `{key(r): r for r in conc_rows}` -- last write wins. So (a) the published wall figure was ONE
    ARBITRARY process, chosen by command-line ORDER (the three real processes measured 80.08 / 81.10 /
    79.76s, inflation 1.553 / 1.573 / 1.547), and (b) `outcomes_identical` was checked against that
    one surviving process, leaving N-1 processes' outcomes never compared to serial at all.

    This test pins both halves: order must not change the answer, and a mismatch in ANY process must
    fail the gate."""
    ser = [_row("g1", 1, 400, None, 1, 100, 1.0, [30], wall=10.0)]
    p1 = [{**_row("g1", 1, 400, None, 1, 100, 1.0, [30], wall=15.0), "_source": "p1"}]
    p2 = [{**_row("g1", 1, 400, None, 1, 100, 1.0, [30], wall=25.0), "_source": "p2"}]

    c = esa.contention_check(ser, p1 + p2)
    assert c["n_concurrent_processes"] == 2
    assert c["concurrent_wall_s_mean"] == 20.0
    assert (c["concurrent_wall_s_min"], c["concurrent_wall_s_max"]) == (15.0, 25.0)
    assert c["wall_inflation_factor_mean"] == 2.0
    assert c["outcomes_identical"] is True

    # ORDER INDEPENDENCE -- the whole point. The old implementation returned 25.0 here and 15.0 above.
    c_rev = esa.contention_check(ser, p2 + p1)
    assert c_rev["concurrent_wall_s_mean"] == c["concurrent_wall_s_mean"]
    assert c_rev["wall_inflation_factor_mean"] == c["wall_inflation_factor_mean"]

    # A MISMATCH IN THE PROCESS THAT IS *NOT* LAST must still fail the gate. Under last-write-wins
    # this exact case passed: p1's regression was overwritten by p2's clean row.
    p1_bad = [{**p1[0], "levels": 0}]
    c_bad = esa.contention_check(ser, p1_bad + p2)
    assert c_bad["outcomes_identical"] is False
    assert c_bad["n_processes_with_a_mismatch"] == 1
    assert c_bad["mismatches"][0]["process"] == "p1"


def test_an_unfireable_arm_below_the_budget_is_inert_not_a_wiring_alarm():
    """The census's old inertness rule was `grace >= budget`, which is sufficient but not necessary.
    A grace BELOW the budget that exceeds every control cell's post-level-up tail also cannot fire,
    and calling it WIRING SUSPECT stamps the whole artifact uninterpretable. This is why grace 380 was
    deliberately NOT put on the b400 grid: it is below the budget and provably unfireable."""
    rows = [
        _row("vc33", 1, 400, None, 1, 390, 1.0, [30]),  # tail = 360 actions
        _row("vc33", 1, 400, 380, 1, 390, 1.0, [30], stopped=False),  # cannot fire: 360 < 380
        _row("vc33", 1, 400, 100, 1, 140, 1.0, [30], stopped=True),
    ]
    c = esa.instrumentation_census(rows)
    assert c["control_cells_whose_tail_could_close_this_grace"]["380"] == 0
    assert "380" in c["inert_by_construction_arms"]
    assert c["uninstrumented_arms_wiring_suspect"] == []
    assert c["every_firing_capable_arm_fired"] is True, (
        "an unfireable arm must not poison the artifact's verdict"
    )

    # But an arm that COULD have fired and did not is still a wiring alarm.
    rows[1]["early_stop_grace"] = 200  # tail 360 > 200, so it could have closed
    rows[1]["early_stop_grace_requested"] = 200
    c2 = esa.instrumentation_census(rows)
    assert c2["control_cells_whose_tail_could_close_this_grace"]["200"] == 1
    assert c2["uninstrumented_arms_wiring_suspect"] == ["200"]
    assert c2["every_firing_capable_arm_fired"] is False


def test_score_gate_passing_with_no_moved_levelup_vector_is_stamped_structurally_frozen():
    """A score comparison over cells whose checkpoint vectors are identical is FROZEN by
    construction -- the score is a function of that vector, so equal scores are not evidence. It
    must not be reported as a plain pass."""
    rows = [
        _row("vc33", 1, 400, None, 1, 390, 1.5, [15]),
        _row("vc33", 1, 400, 150, 1, 170, 1.5, [15], stopped=True),
    ]
    g = esa.analyse_condition(rows, 400)["arms"]["150"]["gate_score_non_inferiority_authoritative"]
    assert g["witness_cells_whose_levelup_vector_moved"] == 0
    assert g["verdict"] == "PASS_STRUCTURALLY_FROZEN_NO_CELL_COULD_MOVE"
    assert g["interpretable"] is False


def test_only_per_seed_matched_cells_are_compared_and_unmatched_are_reported():
    """AN ANY-SEED UNION COMPARISON shows a control failing against itself. A cell present for the
    control but not the arm must be DROPPED and COUNTED, never folded into the control sum."""
    rows = [
        _row("vc33", 1, 400, None, 1, 300, 1.0, [15]),
        _row("vc33", 1, 400, 150, 1, 200, 1.0, [15], stopped=True),
        _row("vc33", 2, 400, None, 1, 300, 9.0, [15]),  # seed 2 control only -- no arm cell
    ]
    arm = esa.analyse_condition(rows, 400)["arms"]["150"]
    assert arm["n_matched_cells"] == 1
    assert arm["n_unmatched_cells_dropped"] == 1
    g = arm["gate_score_non_inferiority_authoritative"]
    assert g["control_efficiency_sum"] == 1.0, (
        "the unmatched seed-2 control must NOT enter the control sum -- that is the union "
        "comparison that makes a control fail against itself"
    )


def test_sign_test_reports_both_tails_and_the_reachable_floor():
    """A ONE-SIDED test makes a REVERSAL read as 'no effect'. And at small support no p below the
    floor is reachable, so 'not significant' would carry no information."""
    rev = esa.sign_test_both_tails([-1.0] * 8)
    assert rev["direction_favoured"] == "negative"
    assert rev["p_one_sided_less"] < 0.01 < rev["p_one_sided_greater"]
    assert rev["p_two_sided"] < 0.05

    tiny = esa.sign_test_both_tails([1.0, 1.0])
    assert tiny["p_min_reachable_two_sided"] == 0.5
    assert tiny["can_reject_at_0.05"] is False, (
        "at n=2 no outcome could reject at 0.05; reporting p alone would hide that"
    )
    zeros = esa.sign_test_both_tails([0.0, 0.0, 0.0])
    assert zeros["n_nonzero"] == 0 and zeros["direction_favoured"] == "tie"


def test_the_two_charge_models_disagree_only_about_the_tail():
    """The resolved model bills the tail to a ZERO-scoring incomplete level; the refuted one bills
    it to the last COMPLETED level. They must therefore be identical when there is no tail and
    diverge when there is -- which is the whole reason the gate had to be restated."""
    base = [20, 40, 60]
    no_tail = (
        esa.score_actions_to_level([15, 55], 55, base),
        esa.score_total_action_charge([15, 55], 55, base),
    )
    assert no_tail[0] == no_tail[1]
    with_tail_auth = esa.score_actions_to_level([15, 55], 4000, base)
    with_tail_refuted = esa.score_total_action_charge([15, 55], 4000, base)
    assert with_tail_auth == no_tail[0], (
        "the AUTHORITATIVE model must be INVARIANT to the post-solve tail -- this is the fact that "
        "empties the pass region of an 'efficiency must improve' gate"
    )
    assert with_tail_refuted < with_tail_auth


def test_reproduction_check_detects_a_nondeterministic_cell():
    """A sweep whose cells are not reproducible cannot support a per-cell matched comparison."""
    a = [_row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57])]
    same = [_row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57])]
    assert esa.reproduction_check(a, same)["deterministic"] is True
    drift = [_row("vc33", 1, 400, None, 1, 390, 0.5, [15])]
    r = esa.reproduction_check(a, drift)
    assert r["deterministic"] is False and r["n_mismatched"] == 1


def test_contention_check_reports_inflation_and_catches_an_outcome_change():
    """The benefit headline is a wall-clock claim, so the measurement condition must be stated:
    inflation is expected, an OUTCOME change is not."""
    s = [_row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57], wall=1.0)]
    c = [_row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57], wall=1.72)]
    out = esa.contention_check(s, c)
    # Reported per process and as a mean/range since 2026-07-26 (a single collapsed row made the
    # figure depend on argument order); with ONE process the mean IS that process.
    assert out["n_concurrent_processes"] == 1
    assert out["wall_inflation_factor_mean"] == 1.72 and out["outcomes_identical"] is True
    assert out["per_process"][0]["wall_inflation_factor"] == 1.72
    bad = [_row("vc33", 1, 400, None, 1, 390, 0.5, [15], wall=1.72)]
    assert esa.contention_check(s, bad)["outcomes_identical"] is False


def test_subset_condition_is_flagged_as_not_corpus_level():
    """A condition scoped to level-reaching games has SUMS that are not corpus claims. Mislabelling
    it would let a subset's savings be read as a 25-game result."""
    rows = [
        _row("vc33", 1, 4000, None, 2, 3900, 2.09, [15, 57]),
        _row("vc33", 1, 4000, 400, 2, 500, 2.09, [15, 57], stopped=True),
    ]
    cond = esa.analyse_condition(rows, 4000)
    assert cond["is_full_25_game_corpus"] is False
    assert cond["score_sums_are_corpus_level"] is False
    assert "NOT corpus-level" in cond["scope_note"]


def test_inert_by_construction_arm_is_not_confused_with_a_wiring_failure():
    """A grace >= the budget CANNOT close inside a loop that runs at most `budget` iterations, so
    that arm is a deliberate INERT control and its equality with the control is a determinism check.
    A grace BELOW the budget that still never fired is the shape a dead-flag wiring failure takes.
    Conflating the two would either suppress a real result or hide a real defect."""
    rows = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57]),
        _row("vc33", 1, 400, 400, 2, 390, 2.09, [15, 57], stopped=False),  # inert: grace == budget
        _row("vc33", 1, 400, 100, 2, 160, 2.09, [15, 57], stopped=False),  # could fire, did not
    ]
    c = esa.instrumentation_census(rows)
    assert c["inert_by_construction_arms"] == ["400"]
    assert c["uninstrumented_arms_wiring_suspect"] == ["100"]
    assert c["every_firing_capable_arm_fired"] is False
    assert c["every_treatment_arm_fired_somewhere"] is False, (
        "the coarse flag still reports both; the verdict must key on the wiring-suspect set"
    )

    rows[2]["early_stopped"] = True
    c2 = esa.instrumentation_census(rows)
    assert c2["every_firing_capable_arm_fired"] is True
    assert c2["inert_by_construction_arms"] == ["400"]


def test_sensitivity_model_is_stamped_untrustworthy_when_the_baselines_are_missing(monkeypatch):
    """THE DEAD-DIAGNOSTIC DEFECT, reproduced. The first draft read the baselines off the wrong
    attribute, so `baselines_for` returned [] for every game, BOTH charge models summed to 0.0, and
    the sensitivity check reported 'no difference' -- a dead channel dressed as a null result. The
    cross-check against the row's own installed-scorer `efficiency` is what makes that visible."""
    rows = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57]),
        _row("vc33", 1, 400, 150, 2, 210, 2.09, [15, 57], stopped=True),
    ]
    monkeypatch.setattr(esa, "baselines_for", lambda game: [])
    esa._BASELINE_CACHE.clear()
    s = esa.analyse_condition(rows, 400)["arms"]["150"]["sensitivity_refuted_total_action_charge"]
    assert s["control_sum"] == 0.0 and s["arm_sum"] == 0.0
    assert (
        s["authoritative_reimplementation_crosscheck"]["cells_with_nonempty_human_baselines"] == 0
    )
    assert s["trustworthy"] is False, (
        "a sensitivity model computed from no baselines must be stamped untrustworthy, never "
        "reported as 'the alternative shows no difference'"
    )


def test_crosscheck_catches_a_recomputation_that_disagrees_with_the_installed_scorer(monkeypatch):
    """If the reimplemented authoritative model disagrees with the row's `efficiency`, either the
    persisted checkpoint vector or the baselines are wrong -- and every derived number beside them,
    including the sensitivity model, is untrustworthy."""
    rows = [
        _row("vc33", 1, 400, None, 2, 390, 99.0, [15, 57]),  # efficiency cannot be 99 here
        _row("vc33", 1, 400, 150, 2, 210, 99.0, [15, 57], stopped=True),
    ]
    monkeypatch.setattr(esa, "baselines_for", lambda game: [20, 40, 60])
    esa._BASELINE_CACHE.clear()
    x = esa.analyse_condition(rows, 400)["arms"]["150"]["sensitivity_refuted_total_action_charge"][
        "authoritative_reimplementation_crosscheck"
    ]
    assert x["matches_installed_scorer_on_every_cell"] is False
    assert x["n_mismatched"] == 2


def test_mechanism_reach_bounds_the_saving_by_the_cells_the_window_can_arm_on():
    """A saving claim above the reachable action fraction is impossible by construction: the window
    arms only after the first level-up, so never-levelling cells are out of reach however long their
    tails are. Reporting the tail fraction of WON cells as the available saving is the overstatement
    this bound prevents."""
    rows = [
        # two cells that never level up -- 2000 actions the mechanism can never touch
        _row("aa11", 1, 400, None, 0, 1000, 0.0, []),
        _row("aa11", 1, 400, 150, 0, 1000, 0.0, [], stopped=False),
        _row("bb22", 1, 400, None, 0, 1000, 0.0, []),
        _row("bb22", 1, 400, 150, 0, 1000, 0.0, [], stopped=False),
        # one that does -- 400 actions, of which the tail is reachable
        _row("vc33", 1, 400, None, 2, 400, 2.09, [15, 57]),
        _row("vc33", 1, 400, 150, 2, 210, 2.09, [15, 57], stopped=True),
    ]
    reach = esa.analyse_condition(rows, 400)["mechanism_reach"]
    assert reach["cells_unreachable_levels_eq_0"] == 2
    assert reach["cells_where_window_can_arm_levels_ge_1"] == 1
    assert reach["control_actions_total"] == 2400
    assert reach["control_actions_on_reachable_cells"] == 400
    assert reach["reachable_fraction_of_actions"] == round(400 / 2400, 4)


def test_inert_arms_supply_a_wall_clock_noise_floor_and_a_determinism_witness(tmp_path):
    """An arm whose grace exceeds its budget CANNOT fire, so its true saving is exactly zero on
    every axis. Zero measured ACTION saving is then a determinism witness, and its measured WALL
    saving is the noise floor a wall-clock benefit claim must clear -- otherwise a 4% wall saving
    cannot be told from timing jitter, and wall clock is the only axis this mechanism can win on."""
    rows = [
        _row("vc33", 1, 400, None, 2, 390, 2.09, [15, 57], wall=10.0),
        # inert: grace == budget, so identical actions but jittered wall
        _row("vc33", 1, 400, 400, 2, 390, 2.09, [15, 57], stopped=False, wall=10.4),
        _row("vc33", 1, 400, 150, 2, 210, 2.09, [15, 57], stopped=True, wall=6.0),
    ]
    src = tmp_path / "rows.json"
    src.write_text(__import__("json").dumps({"rows": rows, "flag_parity_vs_live_globals": {}}))
    loaded = esa.load_rows([str(src)])
    cond = {"b400": esa.analyse_condition(loaded, 400)}
    summary = [
        {
            "condition": "b400",
            "grace": arm["grace"],
            "cells_early_stopped": arm["n_cells_early_stopped"],
            "actions_saved_pct": arm["benefit_actions_and_wall"]["total_actions_saved_pct"],
            "wall_saved_pct": arm["benefit_actions_and_wall"]["total_wall_s_saved_pct"],
        }
        for arm in cond["b400"]["arms"].values()
    ]
    inert = [
        s
        for s in summary
        if s["cells_early_stopped"] == 0
        and s["grace"] is not None
        and s["grace"] >= cond[s["condition"]]["budget"]
    ]
    assert [s["grace"] for s in inert] == [400]
    assert inert[0]["actions_saved_pct"] == 0.0, (
        "an inert arm must save exactly zero actions -- a non-zero value would mean the run is not "
        "a deterministic function of the seed, invalidating every per-cell matched comparison"
    )
    assert inert[0]["wall_saved_pct"] < 0, (
        "the jittered inert wall is noise, and it can be negative"
    )


def test_a_regression_above_the_largest_action_gap_is_explained_in_frames_not_left_a_mystery():
    """THE r11l @ b4000 CASE, reproduced. The window counts FRAMES; the persisted gaps are ACTIONS.
    A cell with a 2775-ACTION gap and 219 resets has a ~2936-FRAME gap, so grace=2800 loses the
    level even though 2800 > 2775. Without the frames conversion on the regressing cell, that
    regression looks impossible and invites the wrong diagnosis."""
    ctrl = _row("r11l", 1, 4000, None, 2, 3781, 0.0053, [761, 3536])
    ctrl["n_frames"] = 4000  # 219 resets inflate frames above actions
    ctrl["n_resets"] = 219
    arm = _row("r11l", 1, 4000, 2800, 1, 3447, 0.004, [761], stopped=True)
    arm["n_frames"] = 3634
    arm["n_resets"] = 187
    s = esa.analyse_condition([ctrl, arm], 4000)["arms"]["2800"]["gate_safety"]
    assert s["verdict"] == "FAIL_LEVEL_REGRESSION"
    reg = s["regressing_cells"][0]
    assert reg["control_inter_levelup_gaps"] == [2775]
    assert reg["control_gaps_estimated_in_frames"][0] > 2800, (
        "the ACTION gap is below the grace but the FRAME gap is above it -- that is the whole "
        "explanation for this regression and it must be in the record"
    )
    assert reg["grace_frames"] == 2800 and reg["control_n_resets"] == 219
    assert s["witness_at_risk_cells"] == 1, (
        "the frames-inflated at-risk witness must flag this cell"
    )


def test_the_artifact_registers_itself_under_the_correct_analyser(tmp_path, monkeypatch):
    """The freshness index's entire value is naming the code to RE-RUN when an artifact drifts. The
    shared `register_analyzed_artifact` helper defaults to its OWN __file__, so an importing analyser
    that forgets the `analyzer=` argument registers under the WRONG name and sends a future reader
    chasing the wrong rebuild command."""
    sys.path.insert(0, str(REPO / "scripts"))
    import analyze_scored_path_lever_ab as sibling

    idx = tmp_path / "index.json"
    monkeypatch.setattr(sibling, "ANALYZED_ARTIFACT_INDEX", idx)
    target = REPO / "results" / "outer_loop_arc_early_stop_grace_sweep_20260726.json"
    if not target.exists():  # keep the test meaningful without depending on a run having happened
        target = REPO / "README.md"

    sibling.register_analyzed_artifact(target, analyzer=Path(esa.__file__).resolve())
    entry = __import__("json").loads(idx.read_text())[str(target.resolve().relative_to(REPO))]
    assert entry["analyzer"] == "scripts/analyze_arc_early_stop_sweep.py"

    sibling.register_analyzed_artifact(target)  # the defaulted, wrong-name path
    entry2 = __import__("json").loads(idx.read_text())[str(target.resolve().relative_to(REPO))]
    assert entry2["analyzer"] == "scripts/analyze_scored_path_lever_ab.py", (
        "the default is retained for backwards compatibility, which is exactly why an importing "
        "analyser must pass analyzer= explicitly"
    )


# ---------------------------------------------------------------------------------------------
# REQ-ARC-WMTE-5983 / SCENARIO: analyser-declares-aggregation-substrate-and-true-measurement-wall
#
# WHY THESE EXIST (2026-07-26). The shipped artifact declared
# `inference_substrate: offline_arcade_live_agent_runtime_self_discovery_no_llm` alongside
# `duration_s: 7.884` -- the ANALYSER's runtime -- while the live measurement behind it took
# 9126s. A reader sanity-checking "is this run plausibly real?" would have read the wrong clock
# and seen a live agent apparently stepping 1401 ARC cells in 8 seconds. That is the
# DURATION_TOO_SHORT failure mode reached from the opposite direction: not a fabricated run
# hiding behind a short duration, but a real run whose honest cost was invisible.
# ---------------------------------------------------------------------------------------------

_ARTIFACT_PATH = REPO / "results" / "outer_loop_arc_early_stop_grace_sweep_20260726.json"


@pytest.fixture
def artifact():
    if not _ARTIFACT_PATH.exists():
        pytest.fail(f"the sweep artifact is missing: {_ARTIFACT_PATH}")
    return json.loads(_ARTIFACT_PATH.read_text())


def test_substrate_is_aggregation_not_live_agent(artifact):
    """The analyser steps no env; declaring the live-agent substrate misattributes its runtime."""
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_true_measurement_wall_is_published_and_dwarfs_the_analyser_duration(artifact):
    """The number a reader needs must be PRESENT, not derivable only from the raw row files."""
    mw = artifact["measurement_wall_s"]
    assert mw["total_s"] > 0, "a zero total means the elapsed read is a dead channel"
    # The whole point: these two clocks differ by orders of magnitude and must not be confused.
    assert mw["total_s"] > 100 * artifact["duration_s"]
    assert mw["n_cells"] >= 1000, "every arm's rows must be counted, not just --rows"


def test_every_row_file_reports_its_own_elapsed_so_no_fallback_is_in_use(artifact):
    """A silent fallback to summed per-cell wall undercounts by ~25% (9126s vs 6861s measured)."""
    mw = artifact["measurement_wall_s"]
    assert mw["all_files_report_their_own_elapsed"] is True
    assert mw["files_using_fallback_basis"] == []


def test_measurement_wall_prefers_file_elapsed_over_summed_cell_wall(tmp_path):
    """THE UNDERCOUNT DEFECT, pinned directly on the helper.

    Per-cell `wall_s` omits per-cell setup, so summing it is systematically low. The helper must
    prefer the driving process's own clock and SAY which basis it used.
    """
    p = tmp_path / "rows_x.json"
    p.write_text(json.dumps({"elapsed_s": 100.0, "rows": [{"wall_s": 1.0}, {"wall_s": 2.0}]}))
    got = esa.measurement_wall_clock([str(p)])
    assert got["total_s"] == 100.0, "must not fall back to the 3.0s cell sum when elapsed exists"
    assert got["per_file"][0]["basis"] == "file_elapsed_s"


def test_measurement_wall_falls_back_and_discloses_when_elapsed_is_absent(tmp_path):
    """An older row file without `elapsed_s` must still be counted -- and flagged, not hidden."""
    p = tmp_path / "rows_old.json"
    p.write_text(json.dumps({"rows": [{"wall_s": 1.5}, {"wall_s": 2.5}]}))
    got = esa.measurement_wall_clock([str(p)])
    assert got["total_s"] == 4.0
    assert got["per_file"][0]["basis"] == "summed_cell_wall_s_fallback"
    assert got["files_using_fallback_basis"] == [str(p)]
    assert got["all_files_report_their_own_elapsed"] is False


def test_measurement_wall_survives_an_unreadable_file_without_crashing(tmp_path):
    """The analyser must never die on a truncated row file mid-sweep; it must report the gap."""
    bad = tmp_path / "rows_trunc.json"
    bad.write_text('{"rows": [')
    got = esa.measurement_wall_clock([str(bad)])
    assert got["per_file"][0]["basis"] == "unreadable"
    assert got["total_s"] == 0.0
