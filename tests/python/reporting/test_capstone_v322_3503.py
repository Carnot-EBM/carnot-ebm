"""Tests for the Capstone v322 aggregation module.

References:
  REQ-CAPSTONE-322: The .322 capstone must aggregate upstream artifacts,
    skip flagged_adversarial ones (exp3502 TAUTOLOGY) per the fabrication gate,
    derive G1-G4 gate status from unflagged primary experiments, report both
    P0.1 routes as BLOCKED, and emit paper_v6_safe_claims honouring the
    Paper-v6 Narrowing Discipline.

SCENARIO-CAP322-001: P0.1 Route 1 blocked (exp3494) → p0_1_route1_blocked=True.
SCENARIO-CAP322-002: P0.1 Route 2 blocked (exp3495) → p0_1_route2_blocked=True.
SCENARIO-CAP322-003: Both routes blocked → p0_1_has_clean_verdict=False.
SCENARIO-CAP322-004: exp3497 CLEAN → calibration diagnosis populated with mathaware AUROC.
SCENARIO-CAP322-005: exp3497 flagged → calibration fields null.
SCENARIO-CAP322-006: exp3498 CLEAN → FR-11 law and R² populated.
SCENARIO-CAP322-007: exp3498 missing → FR-11 law null.
SCENARIO-CAP322-008: exp3499 CLEAN → G2 regression clean, g2_met=False.
SCENARIO-CAP322-009: depth_forcing_function_can_relax=False when P0.1 blocked.
SCENARIO-CAP322-010: All required schema fields present.
SCENARIO-CAP322-011: honest_verdict has terminal prefix.
SCENARIO-CAP322-012: capstone_v322_ready=True.
SCENARIO-CAP322-013: reproducibility_checksum is deterministic and 64 hex chars.
SCENARIO-CAP322-014: paper_v6_safe_claims non-empty and contains 0.9131.
SCENARIO-CAP322-015: paper_v6_forbidden_claims references retracted claims.
SCENARIO-CAP322-016: exp3502 flagged numbers not aggregated as headlines.
SCENARIO-CAP322-017: G2 gate=False when exp3499 g2_met=False.
SCENARIO-CAP322-018: unmet_gates contains 'G2'.
SCENARIO-CAP322-019: PolarFire reachable from exp3501.
SCENARIO-CAP322-020: paper_v6_safe_claims contain beta_min law claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v322_3503 as cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, filename: str, data: Any) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


def _exp3494_blocked() -> dict:
    return {
        "experiment": 3494,
        "schema": "carnot.kona_p01_gate.v1",
        "honest_verdict": "complete: blocked_kona_failure_is_representational_not_optimizer",
        "inference_substrate": "ising_energy_optimization_cpu",
        "easy_tier_solve_rate": 0.0,
        "solve_rate": None,
        "encoding_validity_E0": {
            "total_energy": 0.0,
            "is_valid": True,
        },
        "duration_s": 180.3,
    }


def _exp3495_blocked() -> dict:
    return {
        "experiment": 3495,
        "honest_verdict": "complete: blocked_contested_subset_too_small_n=21",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "contested_subset_n": 21,
        "delta_optimal_vs_self_consistency": None,
        "flip_count_optimal_vs_sc": None,
        "duration_s": 0.116,
    }


def _exp3497_clean() -> dict:
    return {
        "experiment": 3497,
        "honest_verdict": (
            "complete: mathaware_recalibration_recovers_correctness_signal"
            "_domain_shift_was_the_cause"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": False,
        "mathaware_recalibrated_correctness_auroc": 0.624931,
        "step_vs_final_auroc_gap": 0.13795,
        "process_energy_correctness_auroc": 0.60102,
        "acceptance_gate_g0_distinct_pipelines": {"passed": True},
        "n_candidates_heldout": 288,
        "duration_s": 1.512,
    }


def _exp3498_clean() -> dict:
    return {
        "experiment": 3498,
        "honest_verdict": (
            "complete: beta_min_predictable_from_lambda_min_phase5_deployment_law_established"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "flagged_adversarial": False,
        "beta_min_lambda_min_fit": {
            "slope": 1.846114,
            "intercept": -0.300069,
            "r_squared": 0.988610,
        },
        "recommended_phase5_rule": (
            "Phase-5 deployment rule: beta_min = -0.3001 + 1.8461 * lambda_min "
            "(R²=0.989). beta=0 sufficient when lambda_min ≤ 0.1625."
        ),
        "law_holds_out_of_sample": True,
        "duration_s": 1.0,
    }


def _exp3499_clean() -> dict:
    return {
        "experiment": 3499,
        "honest_verdict": (
            "complete: fover_g2_package_regression_clean_external_ask_ready_g2_operator_gated"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "package_reproduced_auroc": 0.9131,
        "package_auroc_within_ci": True,
        "package_sha256": "521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be",
        "package_cid": "QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c",
        "g2_met": False,
        "external_run_pending": True,
        "duration_s": 3.15,
    }


def _exp3501_reachable() -> dict:
    return {
        "experiment": 3501,
        "honest_verdict": "complete: polarfire reachable and continuity confirmed deflagged",
        "inference_substrate": "hardware_smoke",
        "polarfire_ssh_reachable": True,
        "duration_s": 5.0,
    }


def _exp3502_flagged() -> dict:
    """Gate synthesis flagged for TAUTOLOGY (experiment==random_seed by construction)."""
    return {
        "experiment": 3502,
        "honest_verdict": "complete: g1_g3_g4_met_g2_pending_p01_both_routes_blocked",
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
        "g1": True,
        "g2": False,
        "g3": True,
        "g4": True,
        "unmet_gates": ["G2"],
        "p01_has_clean_verdict": False,
        "duration_s": 0.108,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Minimal valid .322 upstream artifacts in a temp directory."""
    _write(tmp_path, "experiment_3494_p01_sudoku.json", _exp3494_blocked())
    _write(tmp_path, "experiment_3495_p01_inband.json", _exp3495_blocked())
    _write(tmp_path, "experiment_3497_calibration.json", _exp3497_clean())
    _write(tmp_path, "experiment_3498_fr11_law.json", _exp3498_clean())
    _write(tmp_path, "experiment_3499_g2_regression.json", _exp3499_clean())
    _write(tmp_path, "experiment_3501_polarfire.json", _exp3501_reachable())
    _write(tmp_path, "experiment_3502_gate_synthesis.json", _exp3502_flagged())
    return tmp_path


# ---------------------------------------------------------------------------
# REQ-CAPSTONE-322 / SCENARIO-CAP322-* tests
# ---------------------------------------------------------------------------

def test_p01_route1_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP322-001: exp3494 blocked verdict → p0_1_route1_blocked=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route1_blocked"] is True


def test_p01_route2_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP322-002: exp3495 blocked verdict → p0_1_route2_blocked=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_route2_blocked"] is True


def test_p01_no_clean_verdict_when_both_routes_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP322-003: both routes blocked → p0_1_has_clean_verdict=False."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_has_clean_verdict"] is False


def test_calibration_diagnosis_populated_when_clean(results_dir: Path) -> None:
    """SCENARIO-CAP322-004: exp3497 clean → mathaware AUROC and gap populated."""
    result = cap.run_capstone(results_dir)
    assert result["cal_v5_flagged"] is False
    assert result["cal_v5_mathaware_auroc"] == pytest.approx(0.624931, rel=1e-4)
    assert result["cal_v5_step_vs_final_auroc_gap"] == pytest.approx(0.13795, rel=1e-3)
    assert result["cal_v5_gate_g0_distinct_pipelines_passed"] is True


def test_calibration_fields_null_when_exp3497_flagged(tmp_path: Path) -> None:
    """SCENARIO-CAP322-005: exp3497 flagged → calibration fields null."""
    flagged = dict(_exp3497_clean())
    flagged["flagged_adversarial"] = True
    _write(tmp_path, "experiment_3497_cal.json", flagged)
    _write(tmp_path, "experiment_3499_g2.json", _exp3499_clean())
    result = cap.run_capstone(tmp_path)
    assert result["cal_v5_mathaware_auroc"] is None
    assert result["cal_v5_step_vs_final_auroc_gap"] is None


def test_fr11_law_populated_when_exp3498_clean(results_dir: Path) -> None:
    """SCENARIO-CAP322-006: exp3498 clean → FR-11 law and R² populated."""
    result = cap.run_capstone(results_dir)
    assert result["fr11_beta_min_lambda_min_law"] is not None
    assert "beta_min" in result["fr11_beta_min_lambda_min_law"]
    assert result["fr11_r2"] == pytest.approx(0.988610, rel=1e-3)
    assert result["fr11_law_holds_out_of_sample"] is True


def test_fr11_law_null_when_exp3498_missing(tmp_path: Path) -> None:
    """SCENARIO-CAP322-007: exp3498 missing → fr11_beta_min_lambda_min_law=None."""
    _write(tmp_path, "experiment_3499_g2.json", _exp3499_clean())
    result = cap.run_capstone(tmp_path)
    assert result["fr11_beta_min_lambda_min_law"] is None
    assert result["fr11_r2"] is None


def test_g2_regression_clean_and_g2_not_met(results_dir: Path) -> None:
    """SCENARIO-CAP322-008: exp3499 clean → regression clean but g2=False."""
    result = cap.run_capstone(results_dir)
    assert result["g2"] is False
    assert result["g2_package_regression_auroc"] == pytest.approx(0.9131, abs=1e-4)
    assert result["g2_package_auroc_in_ci"] is True
    assert result["g2_external_run_pending"] is True
    assert "521ecbc" in result["g2_package_sha256"]


def test_depth_cannot_relax_when_p01_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP322-009: both P0.1 routes blocked → depth_can_relax=False."""
    result = cap.run_capstone(results_dir)
    assert result["depth_forcing_function_can_relax"] is False


def test_required_schema_fields_present(results_dir: Path) -> None:
    """SCENARIO-CAP322-010: all required schema fields present."""
    result = cap.run_capstone(results_dir)
    required = [
        "schema", "experiment", "milestone", "inference_substrate",
        "g1", "g2", "g3", "g4", "unmet_gates", "paper_ready",
        "p0_1_status", "p0_1_has_clean_verdict",
        "p0_1_route1_blocked", "p0_1_route2_blocked",
        "key_finding", "calibration_diagnosis",
        "fr11_beta_min_lambda_min_law", "fr11_r2",
        "g2_package_status", "g2_package_regression_auroc",
        "depth_forcing_function_can_relax", "top_forward_gap",
        "paper_v6_safe_claims", "paper_v6_forbidden_claims",
        "capstone_v322_ready", "honest_verdict",
        "reproducibility_checksum", "experiments_completed",
        "upstreams", "random_seed",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_honest_verdict_has_terminal_prefix(results_dir: Path) -> None:
    """SCENARIO-CAP322-011: honest_verdict starts with a terminal prefix."""
    result = cap.run_capstone(results_dir)
    v = result["honest_verdict"]
    prefixes = ("complete:", "complete_", "success:", "success_",
                "passed:", "passed_", "shipped:", "shipped_")
    assert any(v.startswith(p) for p in prefixes), (
        f"honest_verdict missing terminal prefix: {v!r}"
    )


def test_capstone_v322_ready_true(results_dir: Path) -> None:
    """SCENARIO-CAP322-012: capstone_v322_ready=True."""
    result = cap.run_capstone(results_dir)
    assert result["capstone_v322_ready"] is True


def test_reproducibility_checksum_deterministic_and_hex(results_dir: Path) -> None:
    """SCENARIO-CAP322-013: checksum is deterministic and 64 hex chars."""
    r1 = cap.run_capstone(results_dir)
    r2 = cap.run_capstone(results_dir)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]
    assert len(r1["reproducibility_checksum"]) == 64
    assert all(c in "0123456789abcdef" for c in r1["reproducibility_checksum"])


def test_paper_v6_safe_claims_contain_headline_auroc(results_dir: Path) -> None:
    """SCENARIO-CAP322-014: paper_v6_safe_claims non-empty and contains 0.9131."""
    result = cap.run_capstone(results_dir)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "0.9131" in safe_str


def test_paper_v6_forbidden_claims_references_retractions(results_dir: Path) -> None:
    """SCENARIO-CAP322-015: forbidden_claims references retracted items."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    assert "thermalization" in forbidden_str.lower() or "0.9857" in forbidden_str
    assert "hardware speedup" in forbidden_str.lower() or "kv260" in forbidden_str.lower()


def test_flagged_exp3502_not_aggregated_as_headline(results_dir: Path) -> None:
    """SCENARIO-CAP322-016: exp3502 flagged for TAUTOLOGY; gate status from primary experiments."""
    result = cap.run_capstone(results_dir)
    # Gate status must come from primary experiments, not from the flagged synthesis
    # G1 should still be True (derived from stable known state)
    assert result["g1"] is True
    # upstreams should mark exp3502 as skipped
    assert "SKIPPED_flagged_adversarial" in result["upstreams"]["exp3502"]
    # The gate_synthesis_note should mention the TAUTOLOGY false-positive
    assert "TAUTOLOGY" in result["gate_synthesis_note"]


def test_g2_gate_false_when_g2_met_is_false(results_dir: Path) -> None:
    """SCENARIO-CAP322-017: g2_met=False in exp3499 → result g2=False."""
    result = cap.run_capstone(results_dir)
    assert result["g2"] is False


def test_unmet_gates_contains_g2(results_dir: Path) -> None:
    """SCENARIO-CAP322-018: unmet_gates contains 'G2'."""
    result = cap.run_capstone(results_dir)
    assert "G2" in result["unmet_gates"]
    assert len(result["unmet_gates"]) == 1  # only G2 unmet


def test_polarfire_reachable_from_exp3501(results_dir: Path) -> None:
    """SCENARIO-CAP322-019: PolarFire reachable when exp3501 reports ssh_reachable=True."""
    result = cap.run_capstone(results_dir)
    assert result["polarfire_reachable"] is True


def test_paper_v6_safe_claims_contain_beta_min_law(results_dir: Path) -> None:
    """SCENARIO-CAP322-020: paper_v6_safe_claims contain the FR-11 beta_min law claim."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "beta_min" in safe_str or "lambda_min" in safe_str
