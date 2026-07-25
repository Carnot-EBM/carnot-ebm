"""REQ-ARC-WMTE-5960 harness wiring: the HUD gate's decision predicates and its attribution.

Spec: REQ-ARC-WMTE-5960,
SCENARIO-ARC-WMTE-5960-GATE-REQUIRES-EVERY-SEED,
SCENARIO-ARC-WMTE-5960-GATE-REQUIRES-AN-ARMED-SAFETY-AXIS,
SCENARIO-ARC-WMTE-5960-GATE-REQUIRES-EVERY-AFFECTED-GAME-MEASURED,
SCENARIO-ARC-WMTE-5960-ATTRIBUTION-COVERS-EVERY-MEASURED-GAME,
SCENARIO-ARC-WMTE-5960-ARTIFACT-IDENTITY.

WHY THIS FILE EXISTS. Every test here is a REGRESSION for a specific defect an adversarial
review found in the first implementation of this gate, each of which had already produced a
wrong conclusion in a real artifact:

  * `passed` was `any_pass` across all HUD arms, so it stamped PASSED on arm G -- the one arm
    with the collapse guard DISABLED and therefore an all-null `safety` block.
  * The gain axis was `any(new_wins)` over seeds, contradicting the gate's own docstring ("PER
    SEED ... not an any-seed union") and letting a one-seed gain decide a flag flip.
  * Nothing required the repair-affected games to be measured. The repair changes the mask on
    6 of 25 public games; the smoke measured ONE of them and the gate still reported a pass, so
    both halves -- including the regression clause -- were evaluated over games where the
    intervention is a no-op.
  * Aliasing attribution iterated only games where a guard-armed arm LOST a win, so it was
    structurally blind to ar25's proven collapses, and its empty output was read as "0
    regressions attributable to the repair".
  * The artifact declared `experiment_id: 5836` and a frontier-discipline title, so the HUD
    result was undiscoverable by requirement or by experiment id.

All of it is pure analysis over row dicts -- no env, no agent run.
"""

from __future__ import annotations

import pytest

from carnot import experiment_5836_frontier_discipline_ab as ab

# Importing the harness pulls arc_competition_agent's chain (torch / jax, ~650MB RSS). One-off
# IMPORT footprint, not a per-test leak -- same marker and same reason as the sibling harness
# test file. Every test below still runs and still asserts.
pytestmark = pytest.mark.memory_watchdog_skip


_CLEAN_GUARD = {
    "collapse_refusals": 0,
    "keys_with_multiple_successors": 3,
    "non_deterministic_keys_excluded_by_control": 0,
    "uncontrolled_branchings_declined": 0,
    "control_live": True,
    "keys_with_repeated_unmasked_antecedent": 2,
    "control_had_power_on_any_key": True,
    "proven_collapses": 0,
    "unproven_masked_branchings": 0,
    "globally_revoked": False,
    "split_budget_exceeded": False,
    "attribution": {
        "branchings_differing_in_repair_added_region": 0,
        "branchings_differing_in_already_shipped_region": 0,
        "branchings_differing_outside_the_mask": 0,
        "attribution_unavailable": 0,
        "regions_supplied": True,
    },
}


def _row(
    arm,
    game,
    seed,
    levels,
    *,
    condition="real",
    digest="d_shipped",
    stage2="admitted",
    guard=None,
    stages_disabled=(),
    stage2_armed=True,
):
    """One measured cell, instrumented the way `run_cell` instruments it."""

    hud = {
        "hud_mask_stage2_confirm_enabled": stage2_armed,
        "hud_mask_safety_stages_explicitly_disabled": list(stages_disabled),
        "collapse_guard": (dict(_CLEAN_GUARD) if guard is None else guard),
        "stage2": {"stage2_verdict": stage2},
    }
    if guard is False:  # explicitly UNARMED guard (arm G's shape)
        hud["collapse_guard"] = None
    return {
        "arm": arm,
        "game": game,
        "seed": seed,
        "condition": condition,
        "ran": True,
        "levels": levels,
        "hud_mask_resolved": digest is not None,
        "hud_mask_cell_count": 64 if digest else 0,
        "hud_mask_digest": digest,
        "hud_mask_stage2_verdict": stage2,
        "node_inflation": 0.2,
        "hud_mask": hud,
    }


def _delta(changed=("gA",), inert=("gB",)):
    return {
        "games_where_mask_changed": list(changed),
        "games_where_mask_is_inert": list(inert),
        "per_game": {},
        "total_cells_dropped_by_the_repair": 0,
    }


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------


def test_only_the_fully_guarded_arm_is_a_flip_candidate():
    """G ships Stage 1 bare and G2 has no Stage 2, so neither can be the flip candidate."""

    assert ab.HUD_MASK_FLIP_CANDIDATE_ARMS == ("G3",)
    assert set(ab.HUD_MASK_FLIP_CANDIDATE_ARMS) <= set(ab.HUD_MASK_ARMS)
    assert ab.ARMS["G"]["kwargs"]["hud_mask_collapse_guard"] is False
    assert ab.ARMS["G"]["kwargs"]["hud_mask_stage2_confirm"] is False
    assert ab.ARMS["G2"]["kwargs"]["hud_mask_collapse_guard"] is True
    assert ab.ARMS["G2"]["kwargs"]["hud_mask_stage2_confirm"] is False
    assert ab.ARMS["G3"]["kwargs"]["hud_mask_collapse_guard"] is True
    assert ab.ARMS["G3"]["kwargs"]["hud_mask_stage2_confirm"] is True


def test_the_flip_candidate_differs_from_its_matched_control_only_by_the_hud_mechanism():
    """G3 minus B2 must be ONLY the HUD stages, or the A/B cannot attribute a delta."""

    control = ab.ARMS[ab.HUD_MASK_CONTROL_ARM]["kwargs"]
    candidate = ab.ARMS["G3"]["kwargs"]
    extra = {k: v for k, v in candidate.items() if control.get(k) != v}
    assert set(extra) == {
        "edge_bar_hud_mask",
        "hud_mask_collapse_guard",
        "hud_mask_stage2_confirm",
    }


# ---------------------------------------------------------------------------
# The gain axis: every seed, not any seed
# ---------------------------------------------------------------------------


def test_gate_fails_a_gain_on_only_one_of_three_seeds():
    """REGRESSION for the any-seed disjunction (fatal-adjacent: it decides the flag flip).

    The previous implementation computed `gained = any(s['new_wins'] for s in per_seed)`, so a
    treatment that gains one game on ONE seed and is flat on the others PASSED -- demonstrated
    directly with synthetic rows before the fix. The docstring meanwhile asserted "PER SEED ...
    not an any-seed union".
    """

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("B2", "gB", seed, 0, digest=None))
        # The treatment gains gA on seed 1 ONLY.
        rows.append(_row("G3", "gA", seed, 1 if seed == 1 else 0, digest="d_new"))
        rows.append(_row("G3", "gB", seed, 0, digest="d_new"))
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA", "gB"), inert=()))
    arm = gate["per_arm"]["G3"]
    assert arm["any_seed_gained"] is True, "the honest description of a one-seed gain"
    assert arm["all_seeds_gained"] is False
    assert arm["passed"] is False
    assert "did_not_gain_on_every_seed" in arm["gate_blockers"]
    assert gate["passed"] is False


def test_gate_passes_only_when_every_seed_gains():
    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("G3", "gA", seed, 1, digest="d_new"))
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA",), inert=()))
    arm = gate["per_arm"]["G3"]
    assert arm["all_seeds_gained"] is True
    assert arm["passed"] is True
    assert arm["gate_blockers"] == []
    assert gate["passed"] is True
    assert gate["verdict"] == "passed"


def test_gate_fails_a_regression_on_any_single_seed():
    """The regression half stays `any()` -- the conservative direction, deliberately."""

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("B2", "gLoss", seed, 1, digest=None))
        rows.append(_row("G3", "gA", seed, 1, digest="d_new"))
        # gLoss is given back on seed 3 only.
        rows.append(_row("G3", "gLoss", seed, 0 if seed == 3 else 1, digest=None))
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA",), inert=("gLoss",)))
    arm = gate["per_arm"]["G3"]
    assert arm["any_seed_regressed"] is True
    assert arm["passed"] is False
    assert "regressed_on_at_least_one_seed" in arm["gate_blockers"]


# ---------------------------------------------------------------------------
# The safety axis as a conjunct
# ---------------------------------------------------------------------------


def test_gate_refuses_to_certify_an_arm_whose_safety_axis_is_unmeasured():
    """REGRESSION for the fatal finding: `any_pass` certified the arm with the guard OFF.

    Arm G's `safety` block in the real artifact was `guard_armed: false`, `collapse_refusals:
    null`, `keys_with_multiple_successors: null`, `control_live_on_all_cells: false` -- while the
    artifact reported `acceptance_gate_hud_mask_passed: True` on the strength of that arm.
    """

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(
            _row(
                "G",
                "gA",
                seed,
                1,
                digest="d_new",
                guard=False,
                stage2=None,
                stage2_armed=False,
                stages_disabled=("collapse_guard", "stage2_confirm"),
            )
        )
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA",), inert=()))
    arm = gate["per_arm"]["G"]
    assert arm["all_seeds_gained"] is True, "arm G really did gain on every seed"
    assert arm["safety"]["guard_armed"] is False
    assert arm["flip_eligible"] is False
    assert arm["role"] == "mechanism_isolation_only"
    assert arm["passed"] is False
    assert "not_a_flip_candidate_arm" in arm["gate_blockers"]
    assert "safety_axis_unmeasured" in arm["gate_blockers"]
    # And the gate as a whole must NOT report a pass on the strength of that arm.
    assert gate["passed"] is False


def test_gate_refuses_a_flip_candidate_whose_mask_was_revoked_at_runtime():
    """A mask withdrawn mid-run corrupted 97.7% of the graph; a revoked cell is not a pass."""

    revoked = dict(_CLEAN_GUARD, globally_revoked=True)
    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("G3", "gA", seed, 1, digest="d_new", guard=revoked))
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA",), inert=()))
    arm = gate["per_arm"]["G3"]
    assert arm["safety"]["globally_revoked_cells"]
    assert arm["passed"] is False
    assert "mask_was_revoked_at_runtime" in arm["gate_blockers"]


def test_safety_reports_control_power_separately_from_control_liveness():
    """`control_live` says an antecedent was supplied; POWER says it could have exonerated."""

    powerless = dict(
        _CLEAN_GUARD,
        collapse_refusals=2,
        keys_with_repeated_unmasked_antecedent=0,
        control_had_power_on_any_key=False,
        proven_collapses=0,
        unproven_masked_branchings=2,
    )
    rows = [_row("G3", "gA", 1, 1, digest="d_new", guard=powerless)]
    safety = ab._hud_arm_safety(rows, "G3", "real")
    assert safety["control_live_on_all_cells"] is True
    assert safety["keys_with_repeated_unmasked_antecedent"] == 0
    assert safety["proven_collapses"] == 0
    assert safety["unproven_masked_branchings"] == 2
    assert safety["refusals_are_all_proven"] is False
    assert safety["control_liveness_is_not_control_power"] is True


# ---------------------------------------------------------------------------
# Mechanism coverage: the repair-affected games must actually be measured
# ---------------------------------------------------------------------------


def test_gate_refuses_a_pass_when_a_repair_affected_game_was_not_measured():
    """REGRESSION for the fatal 3-game smoke.

    The repair changes the mask on 6 of 25 public games (measured: ar25, cn04, lp85, r11l, sc25,
    tn36). The smoke measured r11l, lf52 and tu93 -- and on lf52/tu93 the old and new masks are
    byte-identical, so the newly-added cells were exercised on ONE of the six while the gate
    reported a pass.
    """

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("G3", "gA", seed, 1, digest="d_new"))
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gA", "gUnmeasured"), inert=()))
    arm = gate["per_arm"]["G3"]
    assert arm["all_seeds_gained"] is True
    assert arm["unmeasured_repair_affected_games"] == ["gUnmeasured"]
    assert arm["repair_affected_games_measured"] is False
    assert arm["passed"] is False
    assert "repair_affected_games_unmeasured" in arm["gate_blockers"]
    assert gate["passed"] is False
    assert gate["mechanism_coverage"]["repair_affected_games"] == ["gA", "gUnmeasured"]


def test_gate_refuses_a_pass_when_the_affected_game_set_is_unknown():
    """No mask-delta table means the coverage question is UNANSWERED, which is not a pass."""

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "gA", seed, 0, digest=None))
        rows.append(_row("G3", "gA", seed, 1, digest="d_new"))
    gate = ab.hud_mask_gate(rows, mask_delta=None)
    assert gate["mechanism_coverage"]["mask_delta_available"] is False
    assert gate["per_arm"]["G3"]["passed"] is False
    assert gate["passed"] is False


def test_mechanism_activity_names_the_inert_games_by_digest():
    """A game where treatment and control mask the SAME cells is a measurement of nothing.

    Compared by DIGEST, not by cell count: two masks of equal size can occupy different cells.
    """

    rows = [
        _row("B2", "gChanged", 1, 0, digest="d_shipped"),
        _row("B2", "gInert", 1, 1, digest="d_same"),
        _row("G3", "gChanged", 1, 1, digest="d_repaired"),
        _row("G3", "gInert", 1, 1, digest="d_same"),
    ]
    mech = ab._hud_arm_mechanism_active(rows, "G3", "real")
    assert mech["games_where_mask_changed_vs_control"] == ["gChanged"]
    assert mech["games_where_mask_is_inert_vs_control"] == ["gInert"]
    assert mech["active"] is True


def test_equal_cell_counts_with_different_cells_are_not_inert():
    """The count comparison would have exonerated a repair that MOVED the mask."""

    rows = [
        _row("B2", "gMoved", 1, 1, digest="d_left"),
        _row("G3", "gMoved", 1, 1, digest="d_right"),
    ]
    for row in rows:
        row["hud_mask_cell_count"] = 64  # identical COUNT, different CELLS
    mech = ab._hud_arm_mechanism_active(rows, "G3", "real")
    assert mech["games_where_mask_changed_vs_control"] == ["gMoved"]
    assert mech["games_where_mask_is_inert_vs_control"] == []


# ---------------------------------------------------------------------------
# Attribution: whose mask is aliasing
# ---------------------------------------------------------------------------


def test_attribution_sees_a_guard_refusal_on_a_game_where_no_win_was_lost():
    """REGRESSION for the blind window that missed ar25 entirely.

    The previous implementation iterated `lost_wins` only, so a game where the guard fired but no
    control-held win was lost contributed NOTHING -- and its empty
    `regressions_attributable_to_the_REPAIR_widening_the_mask` was reported as "0 regressions
    attributable to the repair".
    """

    aliasing = dict(
        _CLEAN_GUARD,
        collapse_refusals=4,
        keys_with_multiple_successors=4,
        proven_collapses=1,
        unproven_masked_branchings=3,
        attribution=dict(
            _CLEAN_GUARD["attribution"],
            branchings_differing_in_repair_added_region=4,
        ),
    )
    rows = [
        # Neither arm wins gAlias, so the win/loss window is empty for it.
        _row("B2", "gAlias", 1, 0, digest="d_shipped"),
        _row("G3", "gAlias", 1, 0, digest="d_repaired", guard=aliasing),
    ]
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gAlias",), inert=()))
    attribution = gate["aliasing_attribution"]
    assert attribution["guard_armed_arms_losing_a_control_win_where_the_mask_WIDENED"] == []
    evidence = attribution["per_game_guard_evidence"]
    assert [e["game"] for e in evidence] == ["gAlias"]
    assert evidence[0]["refusals"] == 4
    assert evidence[0]["mask_changed_by_the_repair_on_this_game"] is True
    assert attribution["branchings_where_antecedents_differ_in_the_REPAIR_ADDED_region"] == 4
    assert attribution["proven_collapses_total"] == 1
    assert attribution["unproven_masked_branchings_total"] == 3


def test_attribution_demotes_the_shipped_mask_claim_to_a_hypothesis():
    """ "The shipped mask collapses provably-distinct states" is a hypothesis with a confound."""

    unproven = dict(
        _CLEAN_GUARD,
        collapse_refusals=2,
        keys_with_multiple_successors=2,
        keys_with_repeated_unmasked_antecedent=0,
        control_had_power_on_any_key=False,
        proven_collapses=0,
        unproven_masked_branchings=2,
    )
    rows = [
        _row("B2", "gShipped", 1, 1, digest="d_same"),
        _row("G3", "gShipped", 1, 0, digest="d_same", guard=unproven),
    ]
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=(), inert=("gShipped",)))
    hypothesis = gate["aliasing_attribution"]["shipped_mask_aliasing_hypothesis"]
    assert hypothesis["supported_by_lost_wins_on_unchanged_masks"] is True
    assert hypothesis["proven_collapses"] == 0
    assert hypothesis["unproven_branchings"] == 2
    assert hypothesis["is_a_proof"] is False
    assert "unrendered hidden state" in hypothesis["named_confound"]


def test_attribution_compares_masks_by_digest_not_by_cell_count():
    rows = [
        _row("B2", "gMoved", 1, 1, digest="d_left"),
        _row("G3", "gMoved", 1, 0, digest="d_right", guard=dict(_CLEAN_GUARD)),
    ]
    for row in rows:
        row["hud_mask_cell_count"] = 64
    gate = ab.hud_mask_gate(rows, mask_delta=_delta(changed=("gMoved",), inert=()))
    attribution = gate["aliasing_attribution"]
    assert attribution["guard_armed_arms_losing_a_control_win_where_the_mask_WIDENED"]
    assert attribution["guard_armed_arms_losing_a_control_win_where_the_mask_is_UNCHANGED"] == []


# ---------------------------------------------------------------------------
# The witness, the verdict, and artifact identity
# ---------------------------------------------------------------------------


def test_hud_gate_pass_region_is_provably_non_empty_including_the_safety_conjuncts():
    """The exp5835 guard, extended so a tightened SAFETY predicate turns the witness False."""

    witness = ab._hud_mask_gate_pass_region_witness()
    assert witness["passes"] is True
    assert witness["gained_on_every_seed"] is True
    assert witness["regressed"] is False
    assert witness["safety_conjuncts_satisfiable"] is True
    assert witness["arm"] in ab.HUD_MASK_FLIP_CANDIDATE_ARMS
    assert ab.hud_mask_gate([])["pass_region_nonempty"] is True


def test_verdict_names_the_hud_result_rather_than_a_different_mechanism():
    """REGRESSION: the artifact carrying the HUD result said "uninterpretable" about the graft."""

    scope = {"full_declared_spec": True, "n_games": 25, "budget": 2000}
    cap = {"available": False}
    passed = {
        "per_arm": {"G3": {"measured": True, "gate_blockers": []}},
        "passed": True,
        "mechanism_coverage": {"repair_affected_games": ["gA"]},
    }
    verdict = ab.verdict_for(scope, cap, positive_control_ran=True, error_rate=0.0, hud_gate=passed)
    assert "hud_detector_gate_passed_on_flip_candidate_arm" in verdict

    failed = {
        "per_arm": {
            "G3": {
                "measured": True,
                "gate_blockers": ["regressed_on_at_least_one_seed"],
                "unmeasured_repair_affected_games": [],
            }
        },
        "passed": False,
        "mechanism_coverage": {"repair_affected_games": ["gA"]},
    }
    verdict = ab.verdict_for(scope, cap, positive_control_ran=True, error_rate=0.0, hud_gate=failed)
    assert "hud_detector_gate_failed_regressed_on_at_least_one_seed" in verdict

    # Even with NO positive control the HUD result must still appear: the HUD gate compares
    # against arm B2, a matched control measured in the same run, and does not depend on arm E.
    verdict = ab.verdict_for(
        scope, cap, positive_control_ran=False, error_rate=0.0, hud_gate=failed
    )
    assert "no_positive_control" in verdict
    assert "hud_detector_gate_failed" in verdict


def test_verdict_says_undecided_when_affected_games_were_not_measured():
    scope = {"full_declared_spec": False, "n_games": 3, "budget": 2000}
    undecided = {
        "per_arm": {
            "G3": {
                "measured": True,
                "gate_blockers": ["repair_affected_games_unmeasured"],
                "unmeasured_repair_affected_games": ["ar25", "cn04"],
            }
        },
        "passed": False,
        "mechanism_coverage": {"repair_affected_games": ["ar25", "cn04", "r11l"]},
    }
    verdict = ab.verdict_for(
        scope, {"available": False}, positive_control_ran=True, error_rate=0.0, hud_gate=undecided
    )
    assert "hud_detector_gate_undecided_2_of_3_repair_affected_games_unmeasured" in verdict


def test_verdict_says_mechanism_isolation_only_when_no_flip_candidate_ran():
    scope = {"full_declared_spec": False, "n_games": 3, "budget": 2000}
    isolation = {
        "per_arm": {"G": {"measured": True, "gate_blockers": ["not_a_flip_candidate_arm"]}},
        "passed": False,
        "mechanism_coverage": {"repair_affected_games": []},
    }
    verdict = ab.verdict_for(
        scope, {"available": False}, positive_control_ran=True, error_rate=0.0, hud_gate=isolation
    )
    assert "hud_detector_mechanism_isolation_only_no_flip_candidate_arm_measured" in verdict


def test_verdict_is_unchanged_when_no_hud_arm_ran():
    """The HUD tail must be empty on a non-HUD run, so no published verdict string moves."""

    scope = {"full_declared_spec": True, "n_games": 25, "budget": 2000}
    verdict = ab.verdict_for(
        scope, {"available": False}, positive_control_ran=True, error_rate=0.0, hud_gate=None
    )
    assert verdict == "complete_frontier_discipline_ab_measured"


def test_artifact_identity_constants_name_this_requirement():
    """REGRESSION: a HUD run wrote `experiment_id: 5836` with a frontier-discipline title."""

    assert ab.HUD_MASK_EXPERIMENT_ID == 5960
    assert len({ab.EXPERIMENT_ID, ab.CLICK_PIXEL_EXPERIMENT_ID, ab.HUD_MASK_EXPERIMENT_ID}) == 3


def test_the_affected_game_set_is_a_property_of_the_detector_not_of_the_run_scope():
    """REGRESSION: narrowing the run must not narrow the requirement.

    `run()` computes the mask-delta table over ALL_GAMES, never over the games the caller chose.
    Computed over the run's scope instead, a 3-game smoke saw "1 affected game (r11l), measured"
    and the coverage conjunct passed trivially -- which is the SAME defect the conjunct was added
    to prevent, just moved one level up. Verified directly: the corrected 3-game smoke reports
    all six affected games and blocks on the five it did not measure.
    """

    import inspect

    source = inspect.getsource(ab.run)
    assert "hud_mask_delta_table(ALL_GAMES)" in source
    assert "hud_mask_delta_table(games)" not in source


def test_a_narrow_run_cannot_pass_the_coverage_conjunct():
    """The behavioural half of the same guard, on rows rather than on source."""

    rows = []
    for seed in (1, 2, 3):
        rows.append(_row("B2", "r11l", seed, 0, digest=None))
        rows.append(_row("G3", "r11l", seed, 1, digest="d_new"))
    # The detector's affected set is the full six, regardless of what this run measured.
    delta = _delta(
        changed=("ar25", "cn04", "lp85", "r11l", "sc25", "tn36"),
        inert=("lf52", "tu93"),
    )
    gate = ab.hud_mask_gate(rows, mask_delta=delta)
    arm = gate["per_arm"]["G3"]
    assert arm["all_seeds_gained"] is True
    assert arm["unmeasured_repair_affected_games"] == ["ar25", "cn04", "lp85", "sc25", "tn36"]
    assert arm["passed"] is False
    assert gate["passed"] is False
