"""Tests for the Capstone v323 aggregation module.

References:
  REQ-CAPSTONE-323: The .323 capstone must aggregate upstream artifacts,
    skip flagged_adversarial ones (exp3507 TAUTOLOGY, exp3508 adversarial)
    per the fabrication gate, derive G1-G4 gate status from unflagged
    primary experiments, report P0.1 Route 1 as POSITIVE (Sudoku
    solve_rate=1.0 vs AR=0.0), and emit paper_v6_safe_claims honouring
    the Paper-v6 Narrowing Discipline.  random_seed must be 20260531
    (NOT the experiment number 3514) per the exp3503 tautology fix.

SCENARIO-CAP323-001: P0.1 Route 1 positive (exp3505) → p0_1_route1_blocked=False.
SCENARIO-CAP323-002: P0.1 Route 1 solve_rate=1.0 from exp3505.
SCENARIO-CAP323-003: P0.1 Route 2 flagged (exp3507) → p0_1_route2_blocked=True,
                     delta excluded from headline.
SCENARIO-CAP323-004: P0.1 has clean verdict when Route 1 positive.
SCENARIO-CAP323-005: exp3507 flagged → p0_1_route2_flagged=True, delta=None.
SCENARIO-CAP323-006: exp3508 flagged → step_to_final_gap_closed_fraction=None.
SCENARIO-CAP323-007: FR-11 deployment finding present when exp3509 clean.
SCENARIO-CAP323-008: FR-11 deployed_law_prevents_collapse=False from exp3509.
SCENARIO-CAP323-009: G2 regression clean from exp3510, g2=False.
SCENARIO-CAP323-010: depth_forcing_function_can_relax=True when Route 1 positive
                     and gate synthesis confirms relax.
SCENARIO-CAP323-011: All required schema fields present.
SCENARIO-CAP323-012: honest_verdict has terminal prefix.
SCENARIO-CAP323-013: capstone_v323_ready=True.
SCENARIO-CAP323-014: reproducibility_checksum is deterministic and 64 hex chars.
SCENARIO-CAP323-015: paper_v6_safe_claims non-empty and contains 0.9131.
SCENARIO-CAP323-016: paper_v6_forbidden_claims references retracted claims.
SCENARIO-CAP323-017: exp3507 and exp3508 marked as SKIPPED_flagged_adversarial in upstreams.
SCENARIO-CAP323-018: unmet_gates contains 'G2'.
SCENARIO-CAP323-019: random_seed is 20260531, NOT 3514.
SCENARIO-CAP323-020: paper_v6_safe_claims contain Sudoku positive result claim.
SCENARIO-CAP323-021: paper_v6_safe_claims warn against expired forbidden phrasings.
SCENARIO-CAP323-022: depth_forcing_function_can_relax=False when exp3513 absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v323_3514 as cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, filename: str, data: Any) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


def _exp3505_positive() -> dict:
    """Route 1 Sudoku optimizer-ladder — CLEAN POSITIVE result."""
    return {
        "experiment": 3505,
        "schema": "carnot.kona_p01_gate.v2",
        "honest_verdict": "complete: energy_global_inference_solves_sudoku_p01_datapoint_positive",
        "inference_substrate": "ising_energy_optimization_cpu",
        "solve_rate": 1.0,
        "easy_tier_solve_rate": 1.0,
        "ar_baseline_solve_rate": 0.0,
        "encoding_validity_E0_reasserted": {
            "total_energy": 0.0,
            "is_valid": True,
        },
        "n_puzzles": 21,
        "duration_s": 416.4,
    }


def _exp3507_flagged() -> dict:
    """Route 2 in-band — FLAGGED adversarial (TAUTOLOGY)."""
    return {
        "experiment": 3507,
        "honest_verdict": (
            "complete: process_energy_does_not_change_selections"
            "_selection_premise_refuted_on_this_substrate"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": True,
        "level3_sc": 0.653061,
        "self_consistency_accuracy": 0.653061,
        "process_energy_argmin_accuracy": 0.653061,
        "optimal_aggregation_accuracy": 0.653061,
        "delta_optimal_vs_self_consistency": 0.0,
        "flip_count_optimal_vs_sc": 0,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
        "duration_s": 2.553,
    }


def _exp3508_flagged() -> dict:
    """Step-to-final gap — FLAGGED adversarial."""
    return {
        "experiment": 3508,
        "honest_verdict": "complete: step_to_final_aggregation_recovers_correctness_signal_gap_closed_97pct",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": True,
        "gap_closed_fraction": 0.9665,
        "duration_s": 3.877,
    }


def _exp3509_clean() -> dict:
    """FR-11 beta-law deployment — CLEAN, deployed_law_prevents_collapse=False."""
    return {
        "experiment": 3509,
        "honest_verdict": (
            "complete: beta_min_lambda_min_law_does_not_generalize_to_deployment"
            "_use_conservative_default"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "deployed_law_prevents_collapse": False,
        "recommended_phase5_rule": (
            "Deploy beta=f(lambda_min): beta = 1.8461 * lambda_min + (-0.3001), "
            "clip to [0, 0.5]. Safety margin: add 0.10. Deployment NOT VALIDATED."
        ),
        "duration_s": 1.0,
    }


def _exp3510_clean() -> dict:
    """G2 regression + external ask — CLEAN."""
    return {
        "experiment": 3510,
        "honest_verdict": (
            "complete: fover_g2_package_regression_clean_external_ask_current_g2_operator_gated"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "g2_met": False,
        "fover_auroc": 0.9131,
        "auroc_within_ci": True,
        "package_sha256": "521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be",
        "package_cid": "QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c",
        "external_run_pending": True,
        "external_ask_workflow": ".github/workflows/fover-g2-repro.yml",
        "duration_s": 4.68,
    }


def _exp3513_gate_synthesis() -> dict:
    """Gate synthesis — G1/G3/G4 met, G2 pending, depth can relax."""
    return {
        "experiment": 3513,
        "honest_verdict": (
            "complete: g1_g3_g4_met_g2_pending_p01_route1_positive_sudoku_solves"
            "_route2_flagged_skipped"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": True,
        "g2": False,
        "g3": True,
        "g4": True,
        "unmet_gates": ["G2"],
        "p01_has_clean_verdict": True,
        "depth_forcing_function_can_relax": True,
        "duration_s": 0.062,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Minimal valid .323 upstream artifacts in a temp directory."""
    _write(tmp_path, "experiment_3505_p01_sudoku.json", _exp3505_positive())
    _write(tmp_path, "experiment_3507_p01_inband.json", _exp3507_flagged())
    _write(tmp_path, "experiment_3508_gap.json", _exp3508_flagged())
    _write(tmp_path, "experiment_3509_fr11.json", _exp3509_clean())
    _write(tmp_path, "experiment_3510_g2.json", _exp3510_clean())
    _write(tmp_path, "experiment_3513_gate_synthesis.json", _exp3513_gate_synthesis())
    return tmp_path


# ---------------------------------------------------------------------------
# REQ-CAPSTONE-323 / SCENARIO-CAP323-* tests
# ---------------------------------------------------------------------------

def test_p01_route1_not_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP323-001: exp3505 positive verdict → p0_1_route1_blocked=False."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_blocked"] is False


def test_p01_route1_solve_rate(results_dir: Path) -> None:
    """SCENARIO-CAP323-002: solve_rate=1.0 from exp3505."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_solve_rate"] == pytest.approx(1.0, abs=1e-6)
    assert result["p0_1_route1_ar_baseline_solve_rate"] == pytest.approx(0.0, abs=1e-6)


def test_p01_route2_blocked_because_flagged(results_dir: Path) -> None:
    """SCENARIO-CAP323-003: exp3507 flagged → p0_1_route2_blocked=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route2_blocked"] is True


def test_p01_has_clean_verdict_route1_positive(results_dir: Path) -> None:
    """SCENARIO-CAP323-004: Route 1 positive → p0_1_has_clean_verdict=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_has_clean_verdict"] is True


def test_p01_route2_flagged_delta_excluded(results_dir: Path) -> None:
    """SCENARIO-CAP323-005: exp3507 flagged → p0_1_route2_flagged=True, delta=None."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route2_flagged"] is True
    assert result["p0_1_route2_delta"] is None
    assert result["p0_1_route2_flip_count"] is None


def test_step_to_final_gap_excluded_when_flagged(results_dir: Path) -> None:
    """SCENARIO-CAP323-006: exp3508 flagged → step_to_final_gap_closed_fraction=None."""
    result = cap.run_capstone(results_dir)
    assert result["step_to_final_gap_flagged"] is True
    assert result["step_to_final_gap_closed_fraction"] is None


def test_fr11_deployment_verdict_present(results_dir: Path) -> None:
    """SCENARIO-CAP323-007: exp3509 clean → FR-11 deployment verdict populated."""
    result = cap.run_capstone(results_dir)
    assert result["fr11_deployment_verdict"] is not None
    assert "complete:" in result["fr11_deployment_verdict"]
    assert result["fr11_recommended_phase5_rule"] is not None


def test_fr11_deployed_law_does_not_prevent_collapse(results_dir: Path) -> None:
    """SCENARIO-CAP323-008: deployed_law_prevents_collapse=False from exp3509."""
    result = cap.run_capstone(results_dir)
    assert result["fr11_beta_law_deployment_validated"] is False


def test_g2_regression_clean_and_g2_not_met(results_dir: Path) -> None:
    """SCENARIO-CAP323-009: exp3510 clean → g2_met=False, regression info populated."""
    result = cap.run_capstone(results_dir)
    assert result["g2"] is False
    assert result["g2_package_auroc_in_ci"] is True
    assert result["g2_external_run_pending"] is True
    assert "521ecbc" in result["g2_package_sha256"]


def test_depth_can_relax_when_route1_positive(results_dir: Path) -> None:
    """SCENARIO-CAP323-010: Route 1 positive + gate synthesis confirms relax → can_relax=True."""
    result = cap.run_capstone(results_dir)
    assert result["depth_forcing_function_can_relax"] is True


def test_required_schema_fields_present(results_dir: Path) -> None:
    """SCENARIO-CAP323-011: all required schema fields present."""
    result = cap.run_capstone(results_dir)
    required = [
        "schema", "experiment", "milestone", "inference_substrate",
        "g1", "g2", "g3", "g4", "unmet_gates", "paper_ready",
        "p0_1_status", "p0_1_has_clean_verdict",
        "p0_1_route1_blocked", "p0_1_route1_solve_rate",
        "p0_1_route2_blocked", "p0_1_route2_flagged",
        "step_to_final_gap_closed_fraction", "step_to_final_gap_flagged",
        "key_finding",
        "fr11_beta_law_deployment_validated", "fr11_deployment_verdict",
        "g2_package_status", "g2_package_regression_auroc",
        "depth_forcing_function_can_relax", "top_forward_gap",
        "paper_v6_safe_claims", "paper_v6_forbidden_claims",
        "capstone_v323_ready", "honest_verdict",
        "reproducibility_checksum", "experiments_completed",
        "upstreams", "random_seed",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_honest_verdict_has_terminal_prefix(results_dir: Path) -> None:
    """SCENARIO-CAP323-012: honest_verdict starts with a terminal prefix."""
    result = cap.run_capstone(results_dir)
    v = result["honest_verdict"]
    prefixes = ("complete:", "complete_", "success:", "success_",
                "passed:", "passed_", "shipped:", "shipped_")
    assert any(v.startswith(p) for p in prefixes), (
        f"honest_verdict missing terminal prefix: {v!r}"
    )


def test_capstone_v323_ready_true(results_dir: Path) -> None:
    """SCENARIO-CAP323-013: capstone_v323_ready=True."""
    result = cap.run_capstone(results_dir)
    assert result["capstone_v323_ready"] is True


def test_reproducibility_checksum_deterministic_and_hex(results_dir: Path) -> None:
    """SCENARIO-CAP323-014: checksum is deterministic and 64 hex chars."""
    r1 = cap.run_capstone(results_dir)
    r2 = cap.run_capstone(results_dir)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]
    assert len(r1["reproducibility_checksum"]) == 64
    assert all(c in "0123456789abcdef" for c in r1["reproducibility_checksum"])


def test_paper_v6_safe_claims_contain_headline_auroc(results_dir: Path) -> None:
    """SCENARIO-CAP323-015: paper_v6_safe_claims non-empty and contains 0.9131."""
    result = cap.run_capstone(results_dir)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "0.9131" in safe_str


def test_paper_v6_forbidden_claims_references_retractions(results_dir: Path) -> None:
    """SCENARIO-CAP323-016: forbidden_claims references retracted items."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    assert "thermalization" in forbidden_str.lower() or "0.9857" in forbidden_str
    assert "hardware speedup" in forbidden_str.lower() or "kv260" in forbidden_str.lower()


def test_flagged_exp3507_exp3508_skipped_in_upstreams(results_dir: Path) -> None:
    """SCENARIO-CAP323-017: exp3507 and exp3508 marked SKIPPED_flagged in upstreams."""
    result = cap.run_capstone(results_dir)
    assert "SKIPPED_flagged_adversarial" in result["upstreams"]["exp3507"]
    assert "SKIPPED_flagged_adversarial" in result["upstreams"]["exp3508"]
    # exp3505 (clean positive) must NOT be marked as skipped
    assert "SKIPPED" not in result["upstreams"]["exp3505"]


def test_unmet_gates_contains_g2_only(results_dir: Path) -> None:
    """SCENARIO-CAP323-018: unmet_gates contains exactly 'G2'."""
    result = cap.run_capstone(results_dir)
    assert "G2" in result["unmet_gates"]
    assert len(result["unmet_gates"]) == 1


def test_random_seed_is_fixed_value_not_experiment_number(results_dir: Path) -> None:
    """SCENARIO-CAP323-019: random_seed is 20260531, NOT the experiment number 3514."""
    result = cap.run_capstone(results_dir)
    assert result["random_seed"] == 20260531
    assert result["random_seed"] != result["experiment"], (
        "random_seed must NOT equal experiment number (tautology fix)"
    )


def test_paper_v6_safe_claims_contain_sudoku_positive(results_dir: Path) -> None:
    """SCENARIO-CAP323-020: paper_v6_safe_claims contain Sudoku positive result."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "solve_rate" in safe_str or "sudoku" in safe_str.lower()
    assert "1.0" in safe_str or "positive" in safe_str.lower()


def test_paper_v6_forbidden_claims_warn_about_exp3507_tautology(results_dir: Path) -> None:
    """SCENARIO-CAP323-021: forbidden_claims warn against citing flagged exp3507 delta."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    assert "3507" in forbidden_str or "tautology" in forbidden_str.lower()


def test_depth_cannot_relax_when_gate_synthesis_absent(tmp_path: Path) -> None:
    """SCENARIO-CAP323-022: depth_can_relax=False when exp3513 (gate synthesis) absent."""
    # Only write exp3505 (positive) but NO gate synthesis
    _write(tmp_path, "experiment_3505_p01.json", _exp3505_positive())
    _write(tmp_path, "experiment_3510_g2.json", _exp3510_clean())
    result = cap.run_capstone(tmp_path)
    # Without gate synthesis confirming relax, depth_can_relax should be False
    assert result["depth_forcing_function_can_relax"] is False
