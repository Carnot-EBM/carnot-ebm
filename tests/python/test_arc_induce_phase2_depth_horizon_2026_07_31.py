"""Phase-2 regression tests: the depth-horizon diagnosis and the decoupling null.

Spec: REQ-ARC-WMTE-4664 (the L2 goal-predicate satisfiability gate whose `max_depth=40` horizon
this phase shows was the binding constraint, and whose UNDECIDED verdict these tests keep
distinguishable from a disproof), REQ-ARC-WMTE-6051 (the duplicate-state key that collapses
tn36's 32 changing root actions to a single successor, which is why the node budget was never
binding and only the depth cap could stop the search).

WHAT THESE PROTECT. Phase 2 corrected the interpretation of Phase 1's headline -- criterion
(iii) was reported as an "empty intersection" when it was in fact a SEARCH-HORIZON artifact of
the shipped `max_depth=40`. Two pieces of logic carry that correction, and both have a failure
mode that would silently reverse the conclusion:

  * `classify_rollout` separates `inert_at_root` (the engine never moved) from
    `disproved_at_engine_fixed_point` (the engine ran to a standstill with the goal false). The
    worker reports BOTH as `engine_predicts_no_change_from_any_action`; only `depth_reached`
    tells them apart. Collapsing them merges 13 broken engines with 8 genuine goal disproofs and
    would make "a missing dynamics model is the leading cause" unstateable.
  * `balanced_accuracy` must return None -- not 0.0, not 1.0 -- when one class is absent. lp85's
    held-out tape is 100% no-op, so the degenerate "nothing ever changes" engine scores 1.0 on
    plain accuracy. Reporting that as a decoupling success is exactly the trap the phase exists
    to avoid.

The remaining tests assert the artifact's load-bearing numbers still trace to the recorded raw
data, so an edit to the harness that changes the story cannot land silently.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
PHASE2 = REPO / "results" / "arc_induce_phase2_20260731"
ARTIFACT = REPO / "results" / "outer_loop_arc_induce_phase2_20260731.json"
SHIPPED_MAX_DEPTH = 40


def _analyze():
    spec = importlib.util.spec_from_file_location("p2_analyze", PHASE2 / "harness" / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyze():
    return _analyze()


@pytest.fixture(scope="module")
def artifact():
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def raw():
    return json.loads((PHASE2 / "phase2_raw.json").read_text())


# --------------------------------------------------------------------------------------------
# classify_rollout: the inert / disproved distinction
# --------------------------------------------------------------------------------------------


def test_inert_at_root_and_fixed_point_disproof_are_not_merged(analyze):
    """Same worker status, opposite meanings -- `depth_reached` is the only discriminator."""
    common = {
        "worker_status": "ok",
        "rollout_status": "engine_predicts_no_change_from_any_action",
        "goal_first_true_depth": None,
    }
    assert analyze.classify_rollout(depth_reached=0, **common) == "inert_at_root"
    assert analyze.classify_rollout(depth_reached=61, **common) == "disproved_at_engine_fixed_point"
    assert analyze.classify_rollout(depth_reached=1, **common) == "disproved_at_engine_fixed_point"


def test_goal_depth_is_classified_against_the_shipped_horizon(analyze):
    """A goal at depth 41 is reachable-but-invisible; at 40 it is inside what the gate can see."""
    common = {"worker_status": "ok", "rollout_status": "goal_reached", "depth_reached": 99}
    assert (
        analyze.classify_rollout(goal_first_true_depth=SHIPPED_MAX_DEPTH, **common)
        == "reachable_within_shipped_depth"
    )
    assert (
        analyze.classify_rollout(goal_first_true_depth=SHIPPED_MAX_DEPTH + 1, **common)
        == "reachable_BEYOND_shipped_depth"
    )
    # The tn36 case that carries the phase.
    assert (
        analyze.classify_rollout(goal_first_true_depth=61, **common)
        == "reachable_BEYOND_shipped_depth"
    )


def test_unscored_candidates_are_never_folded_into_a_substantive_class(analyze):
    """A truncation is a missing observation, not a negative result."""
    for status in ("timeout", "unrunnable:SyntaxError", "no_engine"):
        cls = analyze.classify_rollout(
            worker_status=status,
            rollout_status=None,
            depth_reached=None,
            goal_first_true_depth=None,
        )
        assert cls.startswith("not_scored:"), cls


# --------------------------------------------------------------------------------------------
# balanced_accuracy: the degenerate-predictor trap
# --------------------------------------------------------------------------------------------


def test_balanced_accuracy_is_undefined_when_a_class_is_absent(analyze):
    """lp85 is 100% no-op; the constant 'nothing changes' engine must not score as a success."""
    all_noop_constant_engine = {"tp": 0, "fp": 0, "tn": 18, "fn": 0}
    assert analyze.balanced_accuracy(all_noop_constant_engine) is None
    all_changing = {"tp": 17, "fp": 0, "tn": 0, "fn": 0}
    assert analyze.balanced_accuracy(all_changing) is None


def test_balanced_accuracy_scores_a_constant_predictor_at_chance(analyze):
    """With both classes present, predicting one class always must come out at exactly 0.5."""
    assert analyze.balanced_accuracy({"tp": 0, "fp": 0, "tn": 13, "fn": 1}) == 0.5
    assert analyze.balanced_accuracy({"tp": 4, "fp": 16, "tn": 0, "fn": 0}) == 0.5
    # A genuinely informative predictor must exceed it.
    assert analyze.balanced_accuracy({"tp": 4, "fp": 0, "tn": 16, "fn": 0}) == 1.0


# --------------------------------------------------------------------------------------------
# the artifact's load-bearing claims still trace to the raw record
# --------------------------------------------------------------------------------------------


def test_depth_sweep_moved_only_the_horizon_and_produced_disproofs_too(artifact):
    """A horizon change that only ever admits passes would be threshold relaxation in disguise."""
    sweep = artifact["shipped_depth_sweep"]
    assert sweep["n_candidates_newly_plannable_at_depth_61"] == 4
    # The bidirectional evidence: more depth also DISPROVED goals that 40 had left undecided.
    assert sweep["n_candidates_newly_disproved_by_more_depth"] >= 1
    # The node budget was never the binding constraint.
    assert sweep["max_plan_nodes_expanded_at_depth_61"] < 20000
    assert sweep["node_budget_utilisation_at_depth_61"] < 0.25


def test_depth_probe_reproduces_the_40_to_61_flip_for_every_claimed_candidate(artifact):
    """Each candidate claimed as newly plannable must actually fail at 40 and plan at 61."""
    rows = artifact["shipped_depth_sweep"]["rows"]
    flipped = [
        r
        for r in rows
        if r["at_depth_61"]["plan_found"] and not r["at_shipped_depth_40"]["plan_found"]
    ]
    assert (
        len(flipped) == artifact["shipped_depth_sweep"]["n_candidates_newly_plannable_at_depth_61"]
    )
    for r in flipped:
        assert r["at_shipped_depth_40"]["gate_kind"] == "goal_unreached_within_depth"
        assert r["at_depth_61"]["gate_kind"] == "satisfiable"
        assert r["at_depth_61"]["plan_length"] == 61


def test_taxonomy_totals_match_the_raw_stall_candidates(artifact, raw):
    tax = artifact["why_the_stall_path_never_clears_the_goal_gate"]["taxonomy"]
    n_stall_raw = sum(1 for r in raw["results"] if r["game"] != "vc33")
    assert sum(tax.values()) == n_stall_raw == artifact["n_stall_candidates_scored"]


def test_decoupling_null_declares_its_missing_positive_control(artifact):
    """A null claim without a positive control must say so -- CLAUDE.md FALSE_NEGATIVE_RISK."""
    dec = artifact["decoupling_dynamics_without_a_certified_goal"]
    assert "NOT SUPPORTED" in dec["verdict"]
    assert dec["false_negative_risk_declared"]
    assert dec["balanced_accuracy_all_median"] == 0.5
    # The structural reason the corpus cannot test it: headroom and capability never co-occur.
    headroom = dec["headroom_by_game"]
    working_engine_games = {"tn36", "tu93"}
    for g in working_engine_games:
        assert headroom[g]["noop_fraction"] == 0.0, g
    assert set(dec["gradeable_games"]).isdisjoint(working_engine_games)


def test_artifact_claims_no_solve_and_no_game_played(artifact):
    """Phase 2 banked nothing and played nothing; the artifact must not imply otherwise."""
    assert artifact["solve_provenance"] == "not_a_solve_artifact"
    assert artifact["acceptance_gates"]["no_arc_game_played"]["passed"] is True
    assert "offline_reproduced" not in artifact
    assert artifact["model_specs"]["invoked"] is False
    assert artifact["what_it_would_take_to_bank_a_level"]["not_verified_here"]
