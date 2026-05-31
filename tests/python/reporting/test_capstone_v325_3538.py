"""Tests for the Capstone v325 aggregation module (Depth-Over-Breadth XI).

References:
  REQ-CAPSTONE-325: The .325 capstone must aggregate upstream artifacts, skip
    flagged_adversarial ones (exp3528 TAUTOLOGY) per the fabrication gate,
    derive G1-G4 gate status from the gate synthesis (exp3537), report P0.1
    Route-1 Sudoku discriminating-tier as POSITIVE (SA-restarts=1.0 vs
    single-SA=0.733), Route-2 as an informative NEGATIVE (headroom present,
    flip_count=3, delta=-0.032), aggregation positive promoted at n>=80
    (exp3532), and self-learning rule deployed (exp3533).  random_seed must
    be 20260531 (NOT the experiment number 3538) per the exp3503 tautology fix.

SCENARIO-CAP325-001: exp3528 flagged_adversarial → p0_1_route1_gc_flagged=True.
SCENARIO-CAP325-002: exp3528 excluded → gc headline numbers are None.
SCENARIO-CAP325-003: exp3529 CLEAN → sudoku_energy_power_visible=True.
SCENARIO-CAP325-004: exp3529 CLEAN → sudoku solve_rate=1.0 vs single-SA≈0.733.
SCENARIO-CAP325-005: exp3531 CLEAN → p0_1_route2_corpus_had_headroom=True.
SCENARIO-CAP325-006: exp3531 CLEAN → p0_1_route2_flip_count=3 (non-degenerate).
SCENARIO-CAP325-007: exp3531 CLEAN → p0_1_route2_delta negative (informative neg).
SCENARIO-CAP325-008: exp3532 CLEAN → aggregation_positive_promoted=True.
SCENARIO-CAP325-009: exp3532 CLEAN → aggregation_mean_auroc ≈ 0.9234.
SCENARIO-CAP325-010: exp3533 CLEAN → self_learning_collapse_prevented=True.
SCENARIO-CAP325-011: exp3533 CLEAN → self_learning_quality_maintained=False.
SCENARIO-CAP325-012: exp3534 CLEAN → g2=False, g2_package_auroc=0.9131.
SCENARIO-CAP325-013: gate synthesis → unmet_gates=['G2'], G1/G3/G4 True.
SCENARIO-CAP325-014: depth_forcing_function_can_relax=True from gate synthesis.
SCENARIO-CAP325-015: p0_1_has_clean_defensible_verdict=True when route1+route2 clean.
SCENARIO-CAP325-016: p0_1_has_clean_defensible_verdict=False when gate synthesis absent.
SCENARIO-CAP325-017: all required schema fields present.
SCENARIO-CAP325-018: honest_verdict has terminal prefix.
SCENARIO-CAP325-019: capstone_v325_ready=True.
SCENARIO-CAP325-020: random_seed=20260531, NOT experiment number 3538.
SCENARIO-CAP325-021: reproducibility_checksum deterministic and 64 hex chars.
SCENARIO-CAP325-022: paper_v6_safe_claims non-empty and contains 0.9131.
SCENARIO-CAP325-023: paper_v6_forbidden_claims include exp3528 TAUTOLOGY reference.
SCENARIO-CAP325-024: paper_v6_forbidden_claims forbid Route-2 positive claim.
SCENARIO-CAP325-025: exp3528 marked SKIPPED_flagged_adversarial in upstreams.
SCENARIO-CAP325-026: paper_v6_safe_claims reference Sudoku discriminating tier.
SCENARIO-CAP325-027: paper_v6_safe_claims reference Route-2 informative negative.
SCENARIO-CAP325-028: paper_v6_safe_claims reference aggregation n>=80 result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v325_3538 as cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, filename: str, data: Any) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Minimal upstream fixture factories
# ---------------------------------------------------------------------------

def _exp3528_flagged() -> dict:
    """Route-1 graph coloring — FLAGGED adversarial (TAUTOLOGY)."""
    return {
        "experiment": 3528,
        "schema": "carnot.kona_p01_gate.graph_coloring_headroom.v1",
        "honest_verdict": (
            "complete: p01_energy_beats_strong_nonAR_baseline_on_hard_graph_"
            "coloring_solve_rate_1_0_vs_strong_0_956"
        ),
        "inference_substrate": "ising_energy_optimization_cpu",
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "severity": "critical",
             "detail": "calibration_vanilla_descent_solve_rate == vanilla_descent_solve_rate_hard_tier"},
            {"kind": "TAUTOLOGY", "severity": "critical",
             "detail": "pt_mean_swap_rate == pt_swap_acceptance_rate"},
        ],
        "vanilla_descent_solve_rate": 0.2,
        "strong_baseline_solve_rate": 0.9555,
        "solve_rate": 1.0,
        "energy_beats_strong_baseline": True,
        "duration_s": 9.38,
    }


def _exp3529_clean() -> dict:
    """Route-1 Sudoku discriminating tier — CLEAN POSITIVE."""
    return {
        "experiment": 3529,
        "schema": "carnot.kona_p01_gate.v4",
        "honest_verdict": (
            "complete: p01_sudoku_energy_power_visible_on_discriminating_tier_"
            "solve_rate_1_00_vs_single_sa_0_73"
        ),
        "inference_substrate": "ising_energy_optimization_cpu",
        "energy_power_gradient_present": True,
        "solve_rate": 1.0,
        "discrete_sa_single_solve_rate": 0.7333333333333333,
        "ar_greedy_solve_rate": 0.0,
        "n_puzzles": 45,
        "duration_s": 1370.4,
    }


def _exp3530_build() -> dict:
    """Route-2 corpus build — oracle did not exceed SC in this corpus."""
    return {
        "experiment": 3530,
        "honest_verdict": "complete: route2_corpus_build_oracle_lte_sc_insufficient_headroom",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "oracle_exceeds_sc": False,
        "corpus_has_headroom": False,
        "duration_s": 0.5,
    }


def _exp3531_clean() -> dict:
    """Route-2 fair test — CLEAN informative negative."""
    return {
        "experiment": 3531,
        "honest_verdict": (
            "complete: energy_does_not_beat_sc_even_with_headroom_route2_"
            "selection_premise_bounded_informative_negative"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "corpus_oracle_exceeds_sc": True,
        "selectable_headroom": 0.0108,
        "reranker_makes_distinct_selections": True,
        "flip_count_best_vs_sc": 3,
        "flips_correct_best": 0,
        "flips_incorrect_best": 3,
        "delta_best_vs_self_consistency": -0.032258,
        "self_consistency_accuracy": 0.505376,
        "duration_s": 0.786,
    }


def _exp3532_clean() -> dict:
    """Aggregation positive promoted — CLEAN."""
    return {
        "experiment": 3532,
        "honest_verdict": (
            "complete: step_to_final_aggregation_replicates_n93_multiseed_"
            "auroc_09234_ci_08991_09478_promotable_secondary_headline"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "mean_final_correctness_auroc": 0.923444,
        "final_correctness_auroc_ci95": [0.899056, 0.947832],
        "shuffle_control_collapses": True,
        "n_problems": 93,
        "n_seeds": 5,
        "duration_s": 3.856,
    }


def _exp3533_clean() -> dict:
    """Self-learning rule deployed — CLEAN."""
    return {
        "experiment": 3533,
        "honest_verdict": (
            "complete: conservative_default_beta_prevents_collapse_but_"
            "over_regularizes_quality_drops_needs_tuning"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "collapse_detected_deploy_arm": False,
        "collapse_detected_control_beta0": True,
        "quality_maintained": False,
        "deployed_alpha_t_margin": 4.776,
        "duration_s": 1.0,
    }


def _exp3534_clean() -> dict:
    """G2 regression verify — CLEAN."""
    return {
        "experiment": 3534,
        "honest_verdict": (
            "complete: fover_g2_package_regression_clean_external_ask_"
            "current_g2_operator_gated"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "g2_met": False,
        "package_reproduced_auroc": 0.9131,
        "package_auroc_within_ci": True,
        "package_sha256": "521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be",
        "package_cid": "QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c",
        "external_run_pending": True,
        "external_ask_workflow_path": ".github/workflows/fover-g2-repro.yml",
        "duration_s": 4.48,
    }


def _exp3537_gate_synthesis() -> dict:
    """Gate synthesis — G1/G3/G4 met; G2 pending; depth can relax."""
    return {
        "experiment": 3537,
        "honest_verdict": (
            "complete: g1_g3_g4_met_g2_pending_p01_sudoku_energy_power_visible_"
            "graph_coloring_flagged_skipped_route2_informative_negative_"
            "headroom_present_depth_relax=yes"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": True,
        "g2": False,
        "g3": True,
        "g4": True,
        "unmet_gates": ["G2"],
        "p01_has_clean_defensible_verdict": True,
        "p01_sudoku_energy_power_visible": True,
        "p01_route2_corpus_had_headroom_exp3530": False,
        "p01_route2_fair_verdict": (
            "complete: energy_does_not_beat_sc_even_with_headroom_route2_"
            "selection_premise_bounded_informative_negative"
        ),
        "p01_route2_corpus_had_headroom": True,
        "p01_route2_flip_count": 3,
        "p01_route2_delta": -0.032258,
        "aggregation_positive_promoted": "mean_auroc=0.9234, CI=[0.899056, 0.947832]",
        "self_learning_deployed_rule": (
            "complete: conservative_default_beta_prevents_collapse_but_"
            "over_regularizes_quality_drops_needs_tuning"
        ),
        "depth_forcing_function_can_relax": True,
        "duration_s": 0.071,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Minimal valid .325 upstream artifacts in a temp directory."""
    _write(tmp_path, "experiment_3528_graph_coloring.json", _exp3528_flagged())
    _write(tmp_path, "experiment_3529_sudoku.json", _exp3529_clean())
    _write(tmp_path, "experiment_3530_route2_build.json", _exp3530_build())
    _write(tmp_path, "experiment_3531_route2_fair_test.json", _exp3531_clean())
    _write(tmp_path, "experiment_3532_aggregation.json", _exp3532_clean())
    _write(tmp_path, "experiment_3533_self_learning.json", _exp3533_clean())
    _write(tmp_path, "experiment_3534_g2_regression.json", _exp3534_clean())
    _write(tmp_path, "experiment_3537_gate_synthesis.json", _exp3537_gate_synthesis())
    return tmp_path


# ---------------------------------------------------------------------------
# REQ-CAPSTONE-325 / SCENARIO-CAP325-* tests
# ---------------------------------------------------------------------------

def test_exp3528_flagged_gc_flagged_true(results_dir: Path) -> None:
    """SCENARIO-CAP325-001: exp3528 flagged_adversarial → p0_1_route1_gc_flagged=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_gc_flagged"] is True


def test_exp3528_excluded_gc_headline_numbers_none(results_dir: Path) -> None:
    """SCENARIO-CAP325-002: exp3528 flagged → gc headline numbers are None."""
    result = cap.run_capstone(results_dir)
    # exp3528 is in _FLAGGED_THIS_MILESTONE so headline GC fields should be None
    assert result["p0_1_route1_graph_coloring_verdict"] is None
    assert result["p0_1_route1_gc_headroom_preserved"] is None
    assert result["p0_1_route1_gc_beats_strong_baseline"] is None


def test_exp3529_sudoku_energy_power_visible(results_dir: Path) -> None:
    """SCENARIO-CAP325-003: exp3529 CLEAN → sudoku_energy_power_visible=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_sudoku_energy_power_visible"] is True


def test_exp3529_sudoku_solve_rates(results_dir: Path) -> None:
    """SCENARIO-CAP325-004: exp3529 CLEAN → solve_rate=1.0, single-SA≈0.733."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_sudoku_solve_rate"] == pytest.approx(1.0, abs=1e-6)
    assert result["p0_1_route1_sudoku_single_sa_baseline"] == pytest.approx(
        0.733, abs=0.01
    )


def test_exp3531_route2_corpus_had_headroom(results_dir: Path) -> None:
    """SCENARIO-CAP325-005: exp3531 CLEAN → p0_1_route2_corpus_had_headroom=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route2_corpus_had_headroom"] is True


def test_exp3531_route2_flip_count_nonzero(results_dir: Path) -> None:
    """SCENARIO-CAP325-006: exp3531 CLEAN → p0_1_route2_flip_count=3 (non-degenerate)."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route2_flip_count"] == 3
    assert result["p0_1_route2_flip_count"] > 0


def test_exp3531_route2_delta_negative(results_dir: Path) -> None:
    """SCENARIO-CAP325-007: exp3531 CLEAN → p0_1_route2_delta negative (informative)."""
    result = cap.run_capstone(results_dir)
    delta = result["p0_1_route2_delta"]
    assert delta is not None
    assert delta < 0, f"Route-2 delta should be negative (informative neg), got {delta}"


def test_exp3532_aggregation_positive_promoted(results_dir: Path) -> None:
    """SCENARIO-CAP325-008: exp3532 CLEAN → aggregation_positive_promoted=True."""
    result = cap.run_capstone(results_dir)
    assert result["aggregation_positive_promoted"] is True


def test_exp3532_aggregation_mean_auroc(results_dir: Path) -> None:
    """SCENARIO-CAP325-009: exp3532 CLEAN → aggregation_mean_auroc ≈ 0.9234."""
    result = cap.run_capstone(results_dir)
    assert result["aggregation_mean_auroc"] is not None
    assert result["aggregation_mean_auroc"] == pytest.approx(0.9234, abs=0.001)


def test_exp3533_collapse_prevented(results_dir: Path) -> None:
    """SCENARIO-CAP325-010: exp3533 CLEAN → self_learning_collapse_prevented=True."""
    result = cap.run_capstone(results_dir)
    assert result["self_learning_collapse_prevented"] is True


def test_exp3533_quality_not_maintained(results_dir: Path) -> None:
    """SCENARIO-CAP325-011: exp3533 CLEAN → self_learning_quality_maintained=False."""
    result = cap.run_capstone(results_dir)
    assert result["self_learning_quality_maintained"] is False


def test_exp3534_g2_regression_clean(results_dir: Path) -> None:
    """SCENARIO-CAP325-012: exp3534 CLEAN → g2=False, g2_package_auroc=0.9131."""
    result = cap.run_capstone(results_dir)
    assert result["g2"] is False
    assert result["g2_package_regression_auroc"] == pytest.approx(0.9131, abs=1e-4)
    assert result["g2_package_auroc_in_ci"] is True
    assert result["g2_external_run_pending"] is True


def test_gate_synthesis_g_gates(results_dir: Path) -> None:
    """SCENARIO-CAP325-013: gate synthesis → G1/G3/G4 met, G2 not met."""
    result = cap.run_capstone(results_dir)
    assert result["g1"] is True
    assert result["g2"] is False
    assert result["g3"] is True
    assert result["g4"] is True
    assert "G2" in result["unmet_gates"]


def test_depth_can_relax_from_gate_synthesis(results_dir: Path) -> None:
    """SCENARIO-CAP325-014: depth_forcing_function_can_relax=True from gate synthesis."""
    result = cap.run_capstone(results_dir)
    assert result["depth_forcing_function_can_relax"] is True


def test_p01_clean_defensible_verdict(results_dir: Path) -> None:
    """SCENARIO-CAP325-015: p0_1_has_clean_defensible_verdict=True when route1+route2 clean."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_has_clean_defensible_verdict"] is True


def test_p01_not_defensible_without_gate_synthesis(tmp_path: Path) -> None:
    """SCENARIO-CAP325-016: p0_1_has_clean_defensible_verdict=False when gate absent."""
    # Only write exp3529 (no gate synthesis, no route2 evidence)
    _write(tmp_path, "experiment_3529_sudoku.json", _exp3529_clean())
    result = cap.run_capstone(tmp_path)
    # Without gate synthesis or route2 headroom+flip, defensible=False
    assert result["p0_1_has_clean_defensible_verdict"] is False


def test_required_schema_fields_present(results_dir: Path) -> None:
    """SCENARIO-CAP325-017: all required schema fields present."""
    result = cap.run_capstone(results_dir)
    required = [
        "schema", "experiment", "milestone", "inference_substrate",
        "g1", "g2", "g3", "g4", "unmet_gates", "paper_ready",
        "p0_1_status", "p0_1_has_clean_defensible_verdict",
        "p0_1_route1_gc_flagged",
        "p0_1_route1_graph_coloring_verdict",
        "p0_1_route1_gc_headroom_preserved",
        "p0_1_route1_gc_beats_strong_baseline",
        "p0_1_route1_sudoku_verdict",
        "p0_1_route1_sudoku_energy_power_visible",
        "p0_1_route1_sudoku_solve_rate",
        "p0_1_route1_sudoku_single_sa_baseline",
        "p0_1_route2_corpus_had_headroom_exp3530",
        "p0_1_route2_verdict",
        "p0_1_route2_corpus_had_headroom",
        "p0_1_route2_flip_count",
        "p0_1_route2_delta",
        "aggregation_positive_promoted",
        "aggregation_mean_auroc",
        "aggregation_ci95",
        "self_learning_verdict",
        "self_learning_collapse_prevented",
        "self_learning_quality_maintained",
        "g2_package_status", "g2_package_regression_auroc",
        "g2_package_auroc_in_ci",
        "depth_forcing_function_can_relax",
        "key_finding", "top_forward_gap",
        "paper_v6_safe_claims", "paper_v6_forbidden_claims",
        "capstone_v325_ready", "honest_verdict",
        "reproducibility_checksum", "experiments_completed",
        "upstreams", "random_seed",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field!r}"


def test_honest_verdict_has_terminal_prefix(results_dir: Path) -> None:
    """SCENARIO-CAP325-018: honest_verdict starts with a terminal prefix."""
    result = cap.run_capstone(results_dir)
    v = result["honest_verdict"]
    prefixes = ("complete:", "complete_", "success:", "success_",
                "passed:", "passed_", "shipped:", "shipped_")
    assert any(v.startswith(p) for p in prefixes), (
        f"honest_verdict missing terminal prefix: {v!r}"
    )


def test_capstone_v325_ready_true(results_dir: Path) -> None:
    """SCENARIO-CAP325-019: capstone_v325_ready=True."""
    result = cap.run_capstone(results_dir)
    assert result["capstone_v325_ready"] is True


def test_random_seed_is_fixed_not_experiment_number(results_dir: Path) -> None:
    """SCENARIO-CAP325-020: random_seed=20260531, NOT experiment number 3538."""
    result = cap.run_capstone(results_dir)
    assert result["random_seed"] == 20260531
    assert result["random_seed"] != result["experiment"], (
        "random_seed must NOT equal experiment number (tautology fix)"
    )


def test_reproducibility_checksum_deterministic_and_hex(results_dir: Path) -> None:
    """SCENARIO-CAP325-021: checksum is deterministic and 64 hex chars."""
    r1 = cap.run_capstone(results_dir)
    r2 = cap.run_capstone(results_dir)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]
    assert len(r1["reproducibility_checksum"]) == 64
    assert all(c in "0123456789abcdef" for c in r1["reproducibility_checksum"])


def test_paper_v6_safe_claims_contain_headline_auroc(results_dir: Path) -> None:
    """SCENARIO-CAP325-022: paper_v6_safe_claims non-empty and contains 0.9131."""
    result = cap.run_capstone(results_dir)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "0.9131" in safe_str


def test_paper_v6_forbidden_claims_reference_exp3528_tautology(results_dir: Path) -> None:
    """SCENARIO-CAP325-023: forbidden_claims include exp3528 TAUTOLOGY reference."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    assert "3528" in forbidden_str or "TAUTOLOGY" in forbidden_str.upper() or \
           "graph" in forbidden_str.lower()


def test_paper_v6_forbidden_claims_forbid_route2_positive(results_dir: Path) -> None:
    """SCENARIO-CAP325-024: forbidden_claims forbid 'energy beats SC' Route-2 claim."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    # Must warn against claiming energy beats SC when exp3531 was negative
    assert "beats sc" in forbidden_str.lower() or \
           "route-2" in forbidden_str.lower() or \
           "route 2" in forbidden_str.lower() or \
           "3531" in forbidden_str


def test_exp3528_skipped_in_upstreams(results_dir: Path) -> None:
    """SCENARIO-CAP325-025: exp3528 marked SKIPPED_flagged_adversarial in upstreams."""
    result = cap.run_capstone(results_dir)
    assert "SKIPPED_flagged_adversarial" in result["upstreams"]["exp3528"]
    # exp3529 (clean positive) must NOT be marked as skipped
    assert "SKIPPED" not in result["upstreams"]["exp3529"]


def test_paper_v6_safe_claims_reference_sudoku_discriminating_tier(results_dir: Path) -> None:
    """SCENARIO-CAP325-026: paper_v6_safe_claims reference Sudoku discriminating tier."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "sudoku" in safe_str.lower() or "discriminating" in safe_str.lower()
    assert "1.0" in safe_str or "solve_rate" in safe_str.lower()


def test_paper_v6_safe_claims_reference_route2_informative_negative(results_dir: Path) -> None:
    """SCENARIO-CAP325-027: paper_v6_safe_claims reference Route-2 informative negative."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "negative" in safe_str.lower() or "route" in safe_str.lower()
    assert "headroom" in safe_str.lower() or "flip" in safe_str.lower()


def test_paper_v6_safe_claims_reference_aggregation_promotion(results_dir: Path) -> None:
    """SCENARIO-CAP325-028: paper_v6_safe_claims reference aggregation n>=80 result."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "0.9234" in safe_str or "aggregat" in safe_str.lower()
