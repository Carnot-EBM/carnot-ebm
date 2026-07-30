"""Tests for adversarial_verify.check_arc_outer_loop_solve -- the ARC live-agent self-solve guard.

Spec: REQ-ARC-WMTE-6049 (the solve-claim matcher SHALL be word/token-boundary aware).

The ARC-AGI-3 deliverable is the LIVE agent self-discovering hidden-game solves from its OWN attempts +
runtime RE. This guard flags ARC solve artifacts that represent outer-loop / off-path / duplicate solving.
CRITICAL sub-checks fire only once provenance is DECLARED, so artifacts predating the contract get at most
a WARN (no retroactive quarantine). See CLAUDE.md "ARC Live-Path Reachability Discipline".
"""

import scripts.adversarial_verify as av


def _flags(d: dict) -> list:
    flags: list = []
    av.check_arc_outer_loop_solve(d, flags)
    return flags


def _kinds(d: dict, severity: str) -> list:
    return [
        f.kind for f in _flags(d) if f.severity == severity and f.kind == "ARC_OUTER_LOOP_SOLVE"
    ]


def _solve(**kw) -> dict:
    base = {"offline_reproduced": True, "game": "qqzz_unregistered", "reproduced_levels": 1}
    base.update(kw)
    return base


def test_non_solve_artifact_is_untouched():
    # a verifier-eval ARC artifact (no offline_reproduced / no level) is not a solve claim -> no flag
    assert _flags({"experiment": "arc_oracle_distinct", "auroc": 0.9}) == []
    assert _flags({"offline_reproduced": False, "game": "tu93"}) == []


def test_missing_provenance_is_warn_not_critical():
    # predates the contract -> surfaced, not quarantined
    assert _kinds(_solve(), "critical") == []
    assert _kinds(_solve(), "warn") == ["ARC_OUTER_LOOP_SOLVE"]


def test_outer_loop_re_provenance_is_critical():
    assert _kinds(_solve(solve_provenance="outer_loop_re"), "critical") == ["ARC_OUTER_LOOP_SOLVE"]


def test_unknown_provenance_is_critical():
    assert _kinds(_solve(solve_provenance="hand_wavy"), "critical") == ["ARC_OUTER_LOOP_SOLVE"]


def test_live_agent_self_discovery_on_new_game_is_clean():
    # an unregistered game, no outer-loop inputs -> the legit self-discovery case, no flags
    assert _flags(_solve(solve_provenance="live_agent_self_discovery")) == []


def test_self_discovery_but_outer_loop_inputs_is_critical():
    d = _solve(solve_provenance="live_agent_self_discovery", offline_ground_truth_bfs=True)
    assert _kinds(d, "critical") == ["ARC_OUTER_LOOP_SOLVE"]


def test_development_proxy_resolve_is_allowed():
    # the offline dev twin re-running an already-solved game is a proxy, not a duplicate solve claim
    d = _solve(solve_provenance="development_proxy", game="tu93", reproduced_levels=3)
    assert _kinds(d, "critical") == []


def test_offline_calibration_solve_is_critical_even_without_solve_shape():
    # the 2nd-recurrence incident: a calibration experiment makes a prose tu93-L3 solve claim WITHOUT the
    # structural offline_reproduced+game+level shape -> the calibration detector must still CRITICAL it
    d = {
        "experiment": "experiment_hazard_l3_calibration",
        "honest_verdict": "success: facing_aware_omni_lethal_zone_CLEAN_winpath_unpruned_on_tu93_L3",
        "bfs_nodes": 22,
    }
    assert _kinds(d, "critical") == ["ARC_OUTER_LOOP_SOLVE"]


def test_offline_calibration_detector_keys_on_name_not_negated_prose():
    # an HONEST artifact whose methodology says it does NOT use offline BFS must NOT be false-flagged
    d = {
        "experiment": "arc_hazard_prune_ab_tu93",
        "honest_verdict": "success: hazard_move_pruner_preserving_the_solve",
        "methodology_note": "fits from the search's own deaths, NO offline ground-truth BFS",
        "game": "tu93",
    }
    assert _kinds(d, "critical") == []


def test_declared_outer_loop_input_triggers_calibration_critical():
    d = {
        "experiment": "arc_solve_x",
        "game": "tu93",
        "honest_verdict": "solved L3",
        "offline_ground_truth_bfs": True,
    }
    assert _kinds(d, "critical") == ["ARC_OUTER_LOOP_SOLVE"]


def test_duplicate_self_discovery_against_registry_is_critical():
    # claiming a fresh self-discovery of tu93 at a level the registry already records -> no new capability
    d = _solve(solve_provenance="live_agent_self_discovery", game="tu93", reproduced_levels=2)
    reg = av._arc_registry_level("tu93")
    if reg is None:
        # registry not loadable in this env -> the duplication sub-check is skipped by design
        assert _kinds(d, "critical") == []
    else:
        assert reg >= 2
        assert "ARC_OUTER_LOOP_SOLVE" in _kinds(d, "critical")


# ---- 2026-07-30: the solve-claim regex must be word-boundary aware -------------------------
def test_unresolved_is_not_a_solve_claim() -> None:
    """`solv` matched inside UNRELATED LONGER WORDS whose meaning is the OPPOSITE of a solve.

    The incident: a cost artifact whose verdict read
    `..._so_affordability_at_110_cells_is_unresolved_and_the_recommendation_is_cheaper_search...`
    was CRITICAL-flagged as an ARC game-solve claim. The only trigger was the `solv` inside
    "unre-SOLV-ed", combined with `used_env_source: True` (legitimately true -- the offline arcade
    reads environment_files). The artifact claimed no level at all.

    This is the bug class CLAUDE.md "QA-Layer Authenticity Discipline" exists for: a substring
    match with no boundary awareness, quarantining honest work. Origin bug #3 of that discipline
    was the same shape (`"diffusiongemma_met" in verdict` matching inside `meta_tensor`).
    """
    d = {
        "experiment": "outer_loop_arc_plan_affordability_corrected_20260730",
        "honest_verdict": (
            "complete_search_budget_threshold_is_137347_engine_calls_and_per_attempt_cost_is_"
            "engine_dependent_so_affordability_at_110_cells_is_unresolved_and_the_recommendation_"
            "is_cheaper_search_not_a_bigger_compute_budget"
        ),
        "used_env_source": True,
        "solve_provenance": "development_proxy",
    }
    assert _kinds(d, "critical") == []
    assert _kinds(d, "warn") == []


def test_the_regex_still_catches_every_real_solve_phrasing() -> None:
    """The narrowing must not blunt the guard -- pinned in both directions.

    A negative lookbehind for a letter (rather than `\\b`) is required because this project's
    verdicts are underscore-joined and `_` is a word character: `\\bsolv` would MISS
    `complete_self_solve_...`, i.e. the exact strings the guard exists to catch.
    """
    for phrasing in (
        "complete_self_solve_tu93_l3",
        "solved tu93 level 3",
        "solving the hidden game",
        "arc_solver_kit reproduction",
        "winpath found",
        "lethal rung calibrated",
        "level-up banked",
        "levelup",
        "reached L3",
    ):
        assert av._ARC_GAME_SOLVE_CLAIM_RE.search(phrasing), phrasing

    for phrasing in (
        "affordability is unresolved",
        "the question is now resolved",
        "the precipitate did not dissolve",
        "resolver returned None",
    ):
        assert not av._ARC_GAME_SOLVE_CLAIM_RE.search(phrasing), phrasing


def test_a_real_calibration_solve_is_still_critical() -> None:
    """CONTROL: the incident this check was built for must still fire.

    Proves the regex narrowing above did not disable the guard -- an artifact that declares an
    outer-loop input AND makes a genuine solve claim is still quarantined.
    """
    d = {
        "experiment": "outer_loop_tu93_hazard_calibration",
        "honest_verdict": "complete_solved_tu93_l3_via_exhaustive_bfs",
        "offline_ground_truth_bfs": True,
        "solve_provenance": "development_proxy",
    }
    assert _kinds(d, "critical") == ["ARC_OUTER_LOOP_SOLVE"]
