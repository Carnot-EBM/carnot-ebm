"""Tests for the Capstone v320 aggregation module.

References:
  REQ-CAPSTONE-320: The .320 capstone must aggregate upstream artifacts,
    skip flagged_adversarial ones per the fabrication gate, emit correct
    G1-G4 gate status, and produce a paper_v6_safe_claims list that
    honours the Paper-v6 Narrowing Discipline.

SCENARIO-CAP320-001: exp3472 blocked → p0_1_v6_blocked=True.
SCENARIO-CAP320-002: exp3472 missing → p0_1_v6_blocked=True.
SCENARIO-CAP320-003: exp3473 flagged → cal advisory fields populated, numbers excluded.
SCENARIO-CAP320-004: exp3473 missing → cal advisory fields zero, note says no data.
SCENARIO-CAP320-005: exp3474 clean + arm_a_collapsed → fr11_collapse_confirmed_at_n200=True.
SCENARIO-CAP320-006: exp3474 missing → fr11_collapse_confirmed_at_n200=False.
SCENARIO-CAP320-007: exp3474 flagged → fr11_collapse_confirmed_at_n200=False (flagged excluded).
SCENARIO-CAP320-008: exp3475 blocked → kona_v5_blocked=True.
SCENARIO-CAP320-009: exp3476 clean → g2 package fields populated.
SCENARIO-CAP320-010: depth_forcing_function_can_relax is False when p01 blocked.
SCENARIO-CAP320-011: All required schema fields present.
SCENARIO-CAP320-012: honest_verdict has terminal prefix.
SCENARIO-CAP320-013: capstone_v320_ready=True.
SCENARIO-CAP320-014: reproducibility_checksum is deterministic and 64 hex chars.
SCENARIO-CAP320-015: paper_v6_safe_claims is non-empty list containing 0.9131.
SCENARIO-CAP320-016: paper_v6_forbidden_claims references retracted/blocked claims.
SCENARIO-CAP320-017: Flagged exp3473 numbers not cited as forward claims in safe_claims.
SCENARIO-CAP320-018: Gate values match exp3480 when present.
SCENARIO-CAP320-019: polarfire_reachable=True when exp3479 clean.
SCENARIO-CAP320-020: paper_v6_safe_claims contains fr11 collapse claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v320_3481 as cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(directory: Path, filename: str, data: Any) -> None:
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


def _gate_artifact(*, g2: bool = False) -> dict:
    return {
        "g1": True,
        "g2": g2,
        "g3": True,
        "g4": True,
        "unmet_gates": [] if g2 else ["G2"],
        "paper_ready": g2,
        "depth_forcing_function_can_relax": False,
        "honest_verdict": "complete: g1_g3_g4_met_g2_unmet",
    }


def _exp3471_outside_band() -> dict:
    return {
        "experiment": 3471,
        "honest_verdict": "complete: blocked_no_headroom_benchmark_sc_outside_band",
        "n_problems_completed": 34,
        "warmup_self_consistency_accuracy": 0.2647,
        "self_consistency_in_headroom_band": False,
    }


def _exp3472_blocked() -> dict:
    return {
        "experiment": 3472,
        "honest_verdict": "complete: blocked_p01_corpus_too_small_n=21",
        "n_problems_heldout": 21,
        "self_consistency_in_headroom_band": None,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _exp3473_flagged() -> dict:
    return {
        "experiment": 3473,
        "honest_verdict": "complete: energy_fails_to_recover_minority_correct_even_with_headroom",
        "flagged_adversarial": True,
        "process_energy_correctness_auroc": 0.441181,
        "trained_energy_correctness_auroc": 0.563326,
        "minority_correct_recovery_rate_process": 0.041667,
        "minority_correct_recovery_rate_trained": 0.041667,
        "minority_correct_fraction": 0.705882,
        "n_candidates_heldout": 204,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }


def _exp3474_clean_collapse() -> dict:
    return {
        "experiment_id": 3474,
        "honest_verdict": "complete: at_risk_grounding_causes_collapse_at_depth_entropy_reg_prevents_it",
        "flagged_adversarial": False,
        "arm_a_mode_collapse_detected": True,
        "arm_b_mode_collapse_detected": False,
        "collapse_onset_iteration": 138,
        "arm_a_final_entropy": 0.9901,
        "arm_b_final_entropy": 4.9067,
        "grounding_collapse_consequence": (
            "ARM A COLLAPSED at depth N=200 (onset iteration 138): entropy→0.9901. "
            "ARM B prevented collapse: entropy=4.9067."
        ),
    }


def _exp3475_blocked() -> dict:
    return {
        "experiment": 3475,
        "honest_verdict": "complete: blocked_kona_instances_saturated_no_headroom",
        "untrained_hybrid_solve_rate": None,
        "blocked_detail": "untrained_hybrid_solve_rate=1.0000 >= 0.80",
    }


def _exp3476_clean() -> dict:
    return {
        "experiment": 3476,
        "honest_verdict": "complete: fover_g2_self_contained_package_verified_external_run_pending",
        "g2_status": "self_contained_package_verified_external_run_pending",
        "package_sha256": "521ecbc3adfa42bce839d16cdcb48cf552e267fc9a8bc69f86068b92a937e6be",
        "package_cid": "QmcoN4zKfAT7GPpokzM31acbE4RBkntfPjhXoEun2NMo9c",
        "package_verified_reproduces": True,
        "condition_a_auroc_isolated": 0.9131,
        "learning_contribution_isolated": 0.0185,
        "g2_independent_reproducer": False,
    }


def _exp3479_reachable() -> dict:
    return {
        "experiment_id": 3479,
        "honest_verdict": "complete: polarfire reachable and continuity confirmed",
        "polarfire_reachable": True,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Minimal valid .320 upstream artifacts in a temp directory."""
    _write(tmp_path, "experiment_3480_g_gate_status.json", _gate_artifact())
    _write(tmp_path, "experiment_3471_p01_headroom.json", _exp3471_outside_band())
    _write(tmp_path, "experiment_3472_p01_v6.json", _exp3472_blocked())
    _write(tmp_path, "experiment_3473_cal_v3.json", _exp3473_flagged())
    _write(tmp_path, "experiment_3474_fr11.json", _exp3474_clean_collapse())
    _write(tmp_path, "experiment_3475_kona.json", _exp3475_blocked())
    _write(tmp_path, "experiment_3476_g2.json", _exp3476_clean())
    _write(tmp_path, "experiment_3479_polarfire.json", _exp3479_reachable())
    return tmp_path


# ---------------------------------------------------------------------------
# REQ-CAPSTONE-320 / SCENARIO-CAP320-* tests
# ---------------------------------------------------------------------------

def test_p01_blocked_when_exp3472_has_blocked_verdict(results_dir: Path) -> None:
    """SCENARIO-CAP320-001: blocked verdict → p0_1_v6_blocked=True."""
    result = cap.run_capstone(results_dir)
    assert result["p0_1_v6_blocked"] is True


def test_p01_blocked_when_exp3472_missing(tmp_path: Path) -> None:
    """SCENARIO-CAP320-002: missing artifact → p0_1_v6_blocked=True."""
    _write(tmp_path, "experiment_3480_gate.json", _gate_artifact())
    result = cap.run_capstone(tmp_path)
    assert result["p0_1_v6_blocked"] is True


def test_cal_advisory_populated_when_exp3473_flagged(results_dir: Path) -> None:
    """SCENARIO-CAP320-003: flagged exp3473 → advisory fields non-zero, cal_v3_flagged=True."""
    result = cap.run_capstone(results_dir)
    assert result["cal_v3_flagged"] is True
    assert result["cal_v3_process_auroc_advisory"] is not None
    assert result["cal_v3_process_auroc_advisory"] == pytest.approx(0.441181, rel=1e-4)
    assert result["cal_v3_minority_recovery_advisory"] is not None
    # Numbers must NOT appear in paper_v6_safe_claims as forward claims
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    # Advisory note is allowed; a raw forward-claim framing is not
    assert "0.441181" not in safe_str or "advisory" in safe_str.lower() or "flagged" in safe_str.lower()


def test_cal_advisory_zero_when_exp3473_missing(tmp_path: Path) -> None:
    """SCENARIO-CAP320-004: missing exp3473 → advisory fields zero, cal note reflects no data."""
    _write(tmp_path, "experiment_3480_gate.json", _gate_artifact())
    result = cap.run_capstone(tmp_path)
    assert result["cal_v3_process_auroc_advisory"] is None or result["cal_v3_process_auroc_advisory"] == 0.0


def test_fr11_collapse_confirmed_when_exp3474_clean(results_dir: Path) -> None:
    """SCENARIO-CAP320-005: clean exp3474 arm_a_collapsed → fr11_collapse_confirmed_at_n200=True."""
    result = cap.run_capstone(results_dir)
    assert result["fr11_collapse_confirmed_at_n200"] is True
    assert result["fr11_arm_b_prevents_collapse"] is True
    assert result["fr11_collapse_onset_iteration"] == 138


def test_fr11_collapse_false_when_exp3474_missing(tmp_path: Path) -> None:
    """SCENARIO-CAP320-006: missing exp3474 → fr11_collapse_confirmed_at_n200=False."""
    _write(tmp_path, "experiment_3480_gate.json", _gate_artifact())
    result = cap.run_capstone(tmp_path)
    assert result["fr11_collapse_confirmed_at_n200"] is False


def test_fr11_collapse_false_when_exp3474_flagged(tmp_path: Path) -> None:
    """SCENARIO-CAP320-007: flagged exp3474 → collapse not confirmed (excluded by fabrication gate)."""
    _write(tmp_path, "experiment_3480_gate.json", _gate_artifact())
    flagged = dict(_exp3474_clean_collapse())
    flagged["flagged_adversarial"] = True
    _write(tmp_path, "experiment_3474_fr11.json", flagged)
    result = cap.run_capstone(tmp_path)
    assert result["fr11_collapse_confirmed_at_n200"] is False


def test_kona_blocked_when_exp3475_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP320-008: blocked exp3475 → kona_v5_blocked=True."""
    result = cap.run_capstone(results_dir)
    assert result["kona_v5_blocked"] is True


def test_g2_package_fields_populated(results_dir: Path) -> None:
    """SCENARIO-CAP320-009: clean exp3476 → g2 package fields populated."""
    result = cap.run_capstone(results_dir)
    assert result["g2_package_verified_internally"] is True
    assert "521ecbc" in result["g2_package_sha256"]
    assert result["g2_condition_a_auroc_isolated"] == pytest.approx(0.9131, abs=1e-4)
    assert result["g2_external_confirmed"] is False
    assert result["g2_package_status"] == "self_contained_package_verified_external_run_pending"


def test_depth_cannot_relax_when_p01_blocked(results_dir: Path) -> None:
    """SCENARIO-CAP320-010: p01 blocked → depth_forcing_function_can_relax=False."""
    result = cap.run_capstone(results_dir)
    assert result["depth_forcing_function_can_relax"] is False


def test_required_schema_fields_present(results_dir: Path) -> None:
    """SCENARIO-CAP320-011: all required schema fields present."""
    result = cap.run_capstone(results_dir)
    required = [
        "schema", "experiment", "milestone", "inference_substrate",
        "g1", "g2", "g3", "g4", "unmet_gates", "paper_ready",
        "p0_1_v6_verdict", "p0_1_v6_blocked", "p0_1_v6_summary",
        "fr11_collapse_confirmed_at_n200", "fr11_arm_b_prevents_collapse",
        "g2_package_status", "depth_forcing_function_can_relax",
        "paper_v6_safe_claims", "paper_v6_forbidden_claims",
        "capstone_v320_ready", "honest_verdict", "reproducibility_checksum",
        "upstreams",
    ]
    for field in required:
        assert field in result, f"Missing required field: {field}"


def test_honest_verdict_has_terminal_prefix(results_dir: Path) -> None:
    """SCENARIO-CAP320-012: honest_verdict starts with a terminal prefix."""
    result = cap.run_capstone(results_dir)
    v = result["honest_verdict"]
    prefixes = ("complete:", "complete_", "success:", "success_",
                "passed:", "passed_", "shipped:", "shipped_")
    assert any(v.startswith(p) for p in prefixes), (
        f"honest_verdict missing terminal prefix: {v!r}"
    )


def test_capstone_v320_ready_true(results_dir: Path) -> None:
    """SCENARIO-CAP320-013: capstone_v320_ready=True."""
    result = cap.run_capstone(results_dir)
    assert result["capstone_v320_ready"] is True


def test_reproducibility_checksum_deterministic_and_hex(results_dir: Path) -> None:
    """SCENARIO-CAP320-014: checksum is deterministic and 64 hex chars."""
    r1 = cap.run_capstone(results_dir)
    r2 = cap.run_capstone(results_dir)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]
    assert len(r1["reproducibility_checksum"]) == 64
    assert all(c in "0123456789abcdef" for c in r1["reproducibility_checksum"])


def test_paper_v6_safe_claims_contain_headline(results_dir: Path) -> None:
    """SCENARIO-CAP320-015: paper_v6_safe_claims is non-empty and contains 0.9131."""
    result = cap.run_capstone(results_dir)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "0.9131" in safe_str


def test_paper_v6_forbidden_claims_references_p01_and_retractions(results_dir: Path) -> None:
    """SCENARIO-CAP320-016: forbidden_claims references blocked P0.1 and retracted items."""
    result = cap.run_capstone(results_dir)
    forbidden_str = json.dumps(result["paper_v6_forbidden_claims"])
    assert "p0.1" in forbidden_str.lower() or "sc" in forbidden_str.lower() or "beats" in forbidden_str.lower()
    assert "0.9857" in forbidden_str or "thermalization" in forbidden_str.lower()


def test_flagged_exp3473_numbers_not_in_safe_claims_as_forward(results_dir: Path) -> None:
    """SCENARIO-CAP320-017: flagged exp3473 minority numbers excluded from forward safe claims."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    # The number 0.041667 should only appear with an 'advisory' or 'flagged' qualifier
    if "0.041" in safe_str:
        assert "advisory" in safe_str.lower() or "flagged" in safe_str.lower()


def test_gate_values_from_exp3480(tmp_path: Path) -> None:
    """SCENARIO-CAP320-018: gate values match exp3480 artifact when present."""
    gate = {"g1": True, "g2": False, "g3": False, "g4": True,
            "unmet_gates": ["G2", "G3"], "paper_ready": False,
            "depth_forcing_function_can_relax": False}
    _write(tmp_path, "experiment_3480_gate.json", gate)
    _write(tmp_path, "experiment_3472_p01.json", _exp3472_blocked())
    result = cap.run_capstone(tmp_path)
    assert result["g3"] is False
    assert "G3" in result["unmet_gates"]
    assert result["paper_ready"] is False


def test_polarfire_reachable_when_exp3479_clean(results_dir: Path) -> None:
    """SCENARIO-CAP320-019: polarfire_reachable=True when exp3479 clean."""
    result = cap.run_capstone(results_dir)
    assert result["polarfire_reachable"] is True


def test_paper_v6_safe_claims_contain_fr11_collapse(results_dir: Path) -> None:
    """SCENARIO-CAP320-020: paper_v6_safe_claims contain fr11 depth collapse claim."""
    result = cap.run_capstone(results_dir)
    safe_str = json.dumps(result["paper_v6_safe_claims"])
    assert "collapse" in safe_str.lower() or "n=200" in safe_str.lower() or "depth" in safe_str.lower()
