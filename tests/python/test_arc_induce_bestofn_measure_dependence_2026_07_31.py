"""Regression tests for the nine review findings against the Phase-1 / Phase-2 induce artifacts.

Spec: REQ-ARC-WMTE-4664 (the L2 goal-predicate satisfiability gate whose depth horizon is the
subject of the Phase-2 correction), REQ-ARC-WMTE-6051 (the duplicate-state key that makes the
node budget non-binding).

WHAT THESE PROTECT, and why each is a REGRESSION rather than a nicety. Every test below
reproduces a specific claim that was stated in artifact prose and was WRONG or UNGROUNDED at the
time it shipped. The failure mode in each case is the same shape: a number or a set that the
underlying data does not support, in a sentence confident enough that no reader would re-derive
it. None of these would have been caught by the existing suite -- the Phase-2 depth test asserted
`n_candidates_newly_disproved_by_more_depth >= 1` and never once asked at WHAT DEPTH, which is
precisely how "at depth 61 the sweep also disproves 3" survived while being false at 61.

  1. VACUOUS CRITERION-(i) PASSES. lp85's held-out split is 18 rows of which 0 change, so an
     engine predicting "nothing changes" scores accuracy 1.0 on it. Three lp85 candidates passed
     criterion (i) that way -- a third of the 9 stall-path passers -- inflating the headline (i)
     yield from a change-gradable 0.25 to 0.40. `heldout_can_grade_change: false` WAS in the
     artifact; no sentence anyone would read mentioned it. `i_dynamics_strict` does not filter
     them either: its no-changing-rows fallback asks only for zero hallucinated no-ops, which
     excludes a wholly inert engine and nothing else, so calling it "strict" overpromised.

  2. THE SIGN OF THE ANTI-SELECTIVITY CLAIM IS MEASURE-DEPENDENT. "Selecting on dynamics is
     anti-selective for plannability" holds under this harness's out-of-sample (i) bar (0 of 9
     selected are plannable vs 2 of 22 rejected) and REVERSES under the shipped trust gate the
     live pipeline actually runs (1 of 7 vs 1 of 24). The two bars disagree numerically on 20 of
     31 scored stall candidates. Under the shipped gate ft09 k1 is in BOTH the selected and the
     plannable set, so the "empty intersection" is a property of the harsher bar, not of the
     candidates. Both directions rest on the same 2 plannable candidates.

  3. A MEASUREMENT FAILURE IS NOT A CRITERION FAILURE. ft09 candidate 5 produced an engine that
     did not terminate inside the validation bound (`gate_timeout`). Scoring it False recorded a
     MISSING OBSERVATION as a zero in every denominator. `unrunnable:*` is genuinely different --
     no engine exists -- and stays a False.

  4. DISPROOFS LAND AT DEPTH 70, NOT 61. At 61 the three disproved candidates are still
     `goal_unreached_within_depth`. The bidirectionality of the sweep is the strongest evidence
     that this is a horizon fix rather than gate-widening, so it must be quoted at the depth where
     it is true rather than borrowed as corroboration for the depth-61 result.

  5. NODE COUNTS AND CANDIDATE SETS MUST NAME THEIR MEMBERS. The accuracy-1.0 trio {k0,k2,k6}
     expands 2226 nodes; 2229 belongs to k4 at accuracy 0.588. The full plannable set at depth 61
     is {k0,k1,k2,k4,k6} -- k1 was already plannable at the shipped depth 40 and is the LOW
     accuracy member, which is exactly the case that weakens the anti-selectivity argument, so
     omitting it from a selection-rule argument is not a rounding error.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
BON_DIR = REPO / "results" / "arc_induce_bestofn_20260731"
BON_ARTIFACT = REPO / "results" / "outer_loop_arc_induce_bestofn_20260731.json"
P2_ARTIFACT = REPO / "results" / "outer_loop_arc_induce_phase2_20260731.json"


@pytest.fixture(scope="module")
def bon():
    return json.loads(BON_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def scored():
    return json.loads((BON_DIR / "bestofn_scored.json").read_text())


@pytest.fixture(scope="module")
def phase2():
    return json.loads(P2_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def stall_scored(scored):
    """The stall-path candidates the harness actually managed to score."""
    post = set(scored["postbank_games"])
    return [
        c for c in scored["candidates"] if c["game"] not in post and c.get("score_status") == "ok"
    ]


# --------------------------------------------------------------------------------------------
# 1. The vacuous criterion-(i) passes
# --------------------------------------------------------------------------------------------


def test_lp85_holdout_cannot_grade_change_at_all(scored):
    """The premise of the whole finding: lp85's held-out split has no changing transition."""
    lp85 = scored["splits"]["lp85"]
    assert lp85["n_heldout"] > 0, "lp85 does hold rows out; the problem is what is in them"
    assert lp85["heldout_n_changing"] == 0
    assert lp85["heldout_can_grade_change"] is False


def test_lp85_criterion_i_passers_are_earned_on_a_split_that_cannot_falsify_them(scored):
    """Three lp85 candidates pass (i) by predicting 'nothing changes' where nothing changes."""
    lp85 = [c for c in scored["candidates"] if c["game"] == "lp85"]
    passers = [c for c in lp85 if c["criteria"]["i_dynamics"] is True]
    assert len(passers) == 3
    for c in passers:
        assert c["heldout_accuracy"] == 1.0
        # Not one held-out CHANGING transition was predicted, because there were none to predict.
        assert c["heldout_n_changing"] == 0
        assert c["heldout_n_changes_correct"] == 0


def test_strict_bar_does_not_catch_the_vacuity_and_must_not_be_relied_on(scored):
    """`i_dynamics_strict` agrees with the plain bar on lp85 -- it is not a vacuity filter.

    This is the finding's sharpest point: the artifact carried a bar NAMED strict that a reader
    would reasonably assume excluded identity-function passes, and it does not. If someone later
    'simplifies' the change-gradable criterion away on the grounds that strict already covers it,
    this test fails."""
    lp85 = [c for c in scored["candidates"] if c["game"] == "lp85"]
    for c in lp85:
        assert c["criteria"]["i_dynamics_strict"] == c["criteria"]["i_dynamics"]
    assert sum(1 for c in lp85 if c["criteria"]["i_dynamics_strict"] is True) == 3


def test_change_gradable_criterion_excludes_the_whole_game_at_every_N(scored):
    """Gradability is a property of the SPLIT, not of the candidate row.

    Reading `heldout_n_changing` off the row instead lets lp85's two syntax-error candidates back
    in as genuine Falses (they carry None because they were never scored), which flipped lp85's
    per-game value from None to False at N=8 only and moved the criterion's yield from a flat
    0.25 to 0.2 for a reason having nothing to do with change prediction."""
    for c in scored["candidates"]:
        if not scored["splits"][c["game"]]["heldout_can_grade_change"]:
            assert c["criteria"]["i_dynamics_change_gradable"] is None, (
                f"{c['game']} k{c['candidate']} ({c.get('score_status')}) leaked into a "
                "criterion whose split cannot grade change"
            )


def test_change_gradable_yield_is_flat_across_N_and_lower_than_the_headline(bon):
    ys = bon["yields_stall_path"]
    seen = []
    for n in (1, 4, 8):
        block = ys.get(f"N{n}")
        if not isinstance(block, dict) or "status" in block:
            continue
        grad = block["i_dynamics_change_gradable"]
        seen.append(grad["yield"])
        # Four of the five stall games can grade change; lp85 cannot.
        assert grad["n_measured_games"] == 4
    assert seen and len(set(seen)) == 1, f"change-gradable yield drifts with N: {seen}"
    assert seen[0] == 0.25
    assert bon["yield_by_criterion_and_N"]["N8"]["i_dynamics"] == 0.4


def test_the_vacuity_is_named_in_prose_not_only_in_a_nested_field(bon):
    """A caveat that lives only in `splits.lp85.heldout_can_grade_change` is not a disclosure.

    The original artifact carried that field and no prose mentioned it, which is how the inflated
    0.40 became the headline number. Both the affected game and the corrected value must appear
    in the sentence a reader actually reads."""
    headline = bon["headline"]
    assert "lp85" in headline
    assert "0.25" in headline
    assert bon["acceptance_gates"]["criterion_i_vacuity_is_disclosed_not_merely_recorded"]["passed"]
    grad = bon["criterion_i_change_gradability"]
    assert grad["headline_i_yield_all_games"] == 0.4
    assert grad["i_yield_change_gradable_games_only"] == 0.25
    assert grad["marginal_change_gradable_only"]["n_pass"] == 6


def test_discounting_lp85_runs_against_the_conclusion_being_drawn(scored):
    """Direction-of-bias check: excluding lp85 leaves the passers entirely tn36's dead engines.

    Stated so the disclosure cannot be read as hedging. If the change-gradable passers were ever
    to include a plannable candidate, the artifact's framing would need revisiting -- and this
    test would fail."""
    grad_pass = [
        c for c in scored["candidates"] if c["criteria"]["i_dynamics_change_gradable"] is True
    ]
    assert {c["game"] for c in grad_pass} == {"tn36"}
    assert not any(c["criteria"]["iii_plan_found_unconditional"] for c in grad_pass)


# --------------------------------------------------------------------------------------------
# 2. The sign of the anti-selectivity claim depends on the measure
# --------------------------------------------------------------------------------------------


def test_the_two_dynamics_bars_disagree_on_most_scored_candidates(stall_scored):
    """20 of 31. The out-of-sample bar and the shipped trust gate are different measurements."""
    differ = [
        c
        for c in stall_scored
        if c.get("heldout_accuracy") is None
        or c.get("shipped_gate_heldout_accuracy") is None
        or abs(float(c["heldout_accuracy"]) - float(c["shipped_gate_heldout_accuracy"])) > 1e-9
    ]
    assert len(stall_scored) == 31
    assert len(differ) == 20


def test_anti_selectivity_reverses_under_the_shipped_trust_gate(stall_scored):
    """The finding in one assertion: same candidates, same plannable set, opposite sign."""

    def rate(selector):
        sel = [c for c in stall_scored if selector(c) is True]
        rest = [c for c in stall_scored if selector(c) is not True]
        p_sel = sum(1 for c in sel if c["criteria"]["iii_plan_found_unconditional"])
        p_rest = sum(1 for c in rest if c["criteria"]["iii_plan_found_unconditional"])
        return (p_sel, len(sel), p_rest, len(rest))

    by_i = rate(lambda c: c["criteria"]["i_dynamics"])
    by_gate = rate(lambda c: c.get("shipped_gate_passes"))
    assert by_i == (0, 9, 2, 22), by_i
    assert by_gate == (1, 7, 1, 24), by_gate
    # Anti-selective one way, positively selective the other.
    assert by_i[0] / by_i[1] < by_i[2] / by_i[3]
    assert by_gate[0] / by_gate[1] > by_gate[2] / by_gate[3]


def test_ft09_k1_is_in_both_sets_under_the_shipped_gate(stall_scored):
    """So the 'empty intersection' is specific to the harsher bar, not a fact about candidates."""
    k1 = next(c for c in stall_scored if c["game"] == "ft09" and c["candidate"] == 1)
    assert k1["criteria"]["i_dynamics"] is False  # out-of-sample: 0.8
    assert k1["shipped_gate_passes"] is True  # shipped gate: 1.0
    assert k1["criteria"]["iii_plan_found_unconditional"] is True
    assert k1["heldout_accuracy"] == 0.8
    assert k1["shipped_gate_heldout_accuracy"] == 1.0


def test_both_2x2s_are_reported_not_only_the_one_supporting_the_headline(bon):
    md = bon["dynamics_vs_plannability"]["stall"]["MEASURE_DEPENDENCE"]
    assert md["by_out_of_sample_criterion_i"]["direction"] == "anti_selective"
    assert md["by_shipped_trust_gate"]["direction"] == "positively_selective"
    assert md["n_candidates_where_the_two_accuracies_differ"] == 20


def test_headline_and_verdict_do_not_assert_an_unqualified_direction(bon):
    """The verdict token itself must not encode a direction the data does not fix."""
    assert "measure_dependent" in bon["honest_verdict"]
    assert "anti_selects" not in bon["honest_verdict"]
    headline = bon["headline"]
    assert "DEPENDS ON WHICH DYNAMICS BAR" in headline
    assert "the sign reverses" in headline
    assert "DIRECTION is undetermined" in headline


# --------------------------------------------------------------------------------------------
# 3. gate_timeout is undetermined, not a zero
# --------------------------------------------------------------------------------------------


def test_gate_timeout_candidate_is_none_on_every_criterion(scored):
    """An engine that existed and was never measured must not be scored as a failure."""
    timeouts = [c for c in scored["candidates"] if c.get("score_status") == "gate_timeout"]
    assert len(timeouts) == 1
    (t,) = timeouts
    assert (t["game"], t["candidate"]) == ("ft09", 5)
    assert t["validation_timed_out"] is True
    for name, value in t["criteria"].items():
        assert value is None, f"{name} recorded a missing observation as {value!r}"


def test_unrunnable_candidates_remain_genuine_zeros(scored):
    """The complementary half: no engine exists, so False is the honest verdict.

    Without this, a well-meaning widening of the undetermined class would quietly delete the 8
    broken-code candidates from the denominator and inflate every yield."""
    broken = [
        c for c in scored["candidates"] if str(c.get("score_status", "")).startswith("unrunnable")
    ]
    assert len(broken) == 8
    for c in broken:
        assert c["criteria"]["i_dynamics"] is False
        assert c["criteria"]["iii_plan_found_unconditional"] is False


def test_denominators_exclude_the_undetermined_candidate(bon):
    marg = bon["marginal_per_candidate_rate"]
    assert marg["i_dynamics"]["n"] == 39, "40 candidates minus the one never measured"
    assert marg["iii_plan_found_unconditional"]["n"] == 39
    disp = bon["candidate_disposition"]["stall"]
    assert disp["n_candidates"] == 40
    assert disp["n_scored"] == 31
    assert disp["n_genuine_zero_no_engine_produced"] == 8
    assert disp["n_undetermined_excluded_from_all_denominators"] == 1


# --------------------------------------------------------------------------------------------
# 4/6. Wilson intervals on the headline yield
# --------------------------------------------------------------------------------------------


def test_headline_yield_carries_an_interval_that_admits_how_wide_it_is(bon):
    """0.4 is 2 of 5 games. Reported bare it reads as a rate; the interval spans 0.12-0.77."""
    block = bon["yields_stall_path"]["N8"]["iii_plan_found_unconditional"]
    assert block["n_pass"] == 2
    assert block["n_measured_games"] == 5
    lo, hi = block["yield_wilson95"]
    assert lo < 0.2 and hi > 0.7
    assert "Wilson" in bon["headline"]


def test_underpowered_note_covers_the_yield_delta_not_just_the_pooled_means(bon):
    """The caveat was originally attached only to the pooled means, understating it."""
    stall = bon["dynamics_vs_plannability"]["stall"]
    note = stall["the_yield_deltas_are_underpowered_too"]
    assert "2 successes across 5 games" in note
    assert "accept_first_over_same_N" in note, "must name the control that isolates selection"
    # And that control must actually still be zero, or the attribution claim is stale.
    for n in (1, 4, 8):
        af = bon["yields_stall_path"][f"N{n}"]["accept_first_over_same_N"]["yield"]
        assert af["iii_plan_found_unconditional"] == 0.0


# --------------------------------------------------------------------------------------------
# 5. Phase 2: the depth of the disproofs, the node count, and the plannable sets
# --------------------------------------------------------------------------------------------


def test_disproofs_land_at_depth_70_and_not_at_61(phase2):
    """The claim that shipped said 'the same sweep at depth 61 also disproves 3'. It does not."""
    sweep = phase2["shipped_depth_sweep"]
    assert sweep["depth_at_which_the_disproofs_first_land"] == 70
    assert sweep["candidates_newly_disproved"] == [3, 5, 7]
    for row in sweep["rows"]:
        if row["candidate"] in (3, 5, 7):
            assert row["at_every_depth"]["61"]["gate_kind"] == "goal_unreached_within_depth"
            assert row["at_every_depth"]["70"]["gate_kind"] == "degenerate_goal_predicate"
    assert "depth 70" in phase2["headline"]


def test_every_swept_depth_is_reported_not_only_the_three_prose_discusses(phase2):
    """The row originally carried 40/61/100 only -- which is HOW the depth-70 fact went missing.

    A row that omits measured depths cannot contradict a claim made about them, so the omission
    was not merely incomplete reporting, it disabled the check."""
    depths = set(phase2["shipped_depth_sweep"]["depths_swept"])
    assert depths == {40, 50, 61, 70, 100}
    for row in phase2["shipped_depth_sweep"]["rows"]:
        assert {int(k) for k in row["at_every_depth"]} == depths


def test_all_three_disproved_candidates_have_heldout_accuracy_1_0(phase2):
    """The prose named k5/k7 and omitted k3, undercounting its own anti-selectivity evidence."""
    assert phase2["shipped_depth_sweep"]["candidates_disproved_despite_heldout_accuracy_1_0"] == [
        3,
        5,
        7,
    ]
    item4 = phase2["what_it_would_take_to_bank_a_level"]["on_tn36_specifically"][3]
    assert "k3, k5, k7" in item4


def test_node_count_quoted_for_the_trio_is_the_trio_s_own(phase2):
    """2229 is k4 at accuracy 0.588; the accuracy-1.0 trio expands 2226.

    The sibling session's independent replication quoted 2226 for exactly this set, so the max
    is also the number a replication cannot reproduce."""
    sweep = phase2["shipped_depth_sweep"]
    assert sweep["plan_nodes_expanded_for_the_accuracy_1_0_trio"] == 2226
    assert sweep["max_plan_nodes_expanded_at_depth_61"] == 2229
    for row in sweep["rows"]:
        if row["candidate"] in (0, 2, 6):
            assert row["at_depth_61"]["plan_nodes_expanded"] == 2226
        if row["candidate"] == 4:
            assert row["at_depth_61"]["plan_nodes_expanded"] == 2229
    assert "2226" in phase2["headline"]
    assert "2229" not in phase2["headline"]


def test_plannable_set_argument_ranges_over_the_full_set_including_k1(phase2):
    """A selection rule ranges over everything available to it, and k1 is the low-accuracy member.

    Quoting only the newly-plannable slice {k0,k2,k4,k6} drops the one case that weakens the
    anti-selectivity argument being made in the same sentence."""
    sweep = phase2["shipped_depth_sweep"]
    assert sweep["candidates_newly_plannable_at_depth_61"] == [0, 2, 4, 6]
    assert sweep["candidates_plannable_at_depth_61_full_set"] == [0, 1, 2, 4, 6]
    item4 = phase2["what_it_would_take_to_bank_a_level"]["on_tn36_specifically"][3]
    assert "{k0, k1, k2, k4, k6}" in item4
    assert "already plannable at the shipped depth 40" in item4


# --------------------------------------------------------------------------------------------
# 7. The build witness says what it tests
# --------------------------------------------------------------------------------------------


def test_exe_check_is_named_after_the_negative_test_it_performs(bon):
    """`'build-hip' not in exe` under the name `..._is_cuda_build` asserted more than it tested."""
    observed = bon["acceptance_gates"]["generator_proven_cuda_build_and_model"]["observed"]
    assert "server_exe_is_not_hip_build" in observed
    assert "server_exe_is_cuda_build" not in observed
    assert observed["server_exe_is_not_hip_build"] is True
    assert "build-hip" in observed["server_exe"] or "build-hip" not in observed["server_exe"]
    gate = bon["acceptance_gates"]["generator_proven_cuda_build_and_model"]
    assert "NEGATIVE path heuristic" in gate["what_the_exe_check_actually_tests"]


def test_cuda_claim_still_rests_on_independent_per_pid_vram_evidence(bon):
    """The rename must not weaken the substantive proof, which was never the path heuristic."""
    observed = bon["acceptance_gates"]["generator_proven_cuda_build_and_model"]["observed"]
    rows = observed["vram_rows_mine"]
    assert rows, "per-PID nvidia-smi residency is what actually proves the CUDA bind"
    assert any("21434" in r for r in rows)
    assert bon["acceptance_gate_passed"] is True
