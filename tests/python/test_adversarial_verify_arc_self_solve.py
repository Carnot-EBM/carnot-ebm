"""Tests for adversarial_verify.check_arc_outer_loop_solve -- the ARC live-agent self-solve guard.

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
