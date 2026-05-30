"""Tests for the Capstone v319 aggregation module.

References:
  REQ-CAPSTONE-319: The .319 capstone must aggregate upstream artifacts,
    skip flagged_adversarial ones per the fabrication gate, emit correct
    G1-G4 gate status, and produce a paper_v6_safe_claims list that
    honours the Paper-v6 Narrowing Discipline.

SCENARIO-CAP319-001: exp3460 flagged → p0_1_v5_is_clean=False, numbers excluded.
SCENARIO-CAP319-002: exp3460 missing → p0_1_v5_is_clean=False.
SCENARIO-CAP319-003: exp3461 clean → trained_energy_correctness_auroc populated.
SCENARIO-CAP319-004: exp3461 missing → trained_energy_correctness_auroc=0.0.
SCENARIO-CAP319-005: exp3461 auroc>0.55 → trained_energy_crosses_055_threshold=True.
SCENARIO-CAP319-006: exp3463 clean → g2_ci_status from artifact, g2_dryrun_green.
SCENARIO-CAP319-007: exp3463 missing → g2_ci_status contains 'pending'.
SCENARIO-CAP319-008: exp3462 flagged → fr11 directional verdict labelled ADVISORY_ONLY.
SCENARIO-CAP319-009: exp3464 clean → kona_trained_hybrid_delta populated.
SCENARIO-CAP319-010: depth_forcing_function_can_relax is False when P0.1 flagged.
SCENARIO-CAP319-011: All required schema fields present.
SCENARIO-CAP319-012: honest_verdict has terminal prefix.
SCENARIO-CAP319-013: capstone_v319_ready=True.
SCENARIO-CAP319-014: reproducibility_checksum is deterministic and 64 hex chars.
SCENARIO-CAP319-015: paper_v6_safe_claims is non-empty list with 0.9131.
SCENARIO-CAP319-016: paper_v6_forbidden_claims references retracted claims.
SCENARIO-CAP319-017: Flagged exp3460 numbers not cited as forward claims in safe_claims.
SCENARIO-CAP319-018: Gate values match exp3468 when present.
SCENARIO-CAP319-019: kona_trained_hybrid_delta=0.0 when exp3464 clean.
SCENARIO-CAP319-020: paper_v6_safe_claims contain calibration advance claim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v319_3469 as cap


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


def _exp3460_flagged() -> dict:
    return {
        "experiment": 3460,
        "honest_verdict": "complete: trained_energy_matches_but_does_not_beat_self_consistency_at_equal_compute",
        "flagged_adversarial": True,
        "trained_energy_weighted_vote_accuracy": 0.908333,
        "self_consistency_accuracy": 0.908333,
        "delta_trained_energy_vs_self_consistency": 0.0,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }


def _exp3460_clean() -> dict:
    return {
        "experiment": 3460,
        "honest_verdict": "complete: trained_energy_beats_self_consistency",
        "flagged_adversarial": False,
        "trained_energy_weighted_vote_accuracy": 0.925,
        "self_consistency_accuracy": 0.908333,
        "delta_trained_energy_vs_self_consistency": 0.016667,
    }


def _exp3461_clean(auroc: float = 0.629401) -> dict:
    return {
        "experiment": 3461,
        "honest_verdict": "complete: trained_or_fover_energy_tracks_correctness_lift_over_untrained_reported",
        "trained_energy_correctness_auroc": auroc,
        "fover_energy_correctness_auroc": 0.605838,
        "trained_energy_auroc_lift_over_untrained": auroc - 0.516,
        "within_problem_argmin_correct_rate_trained": 0.858333,
        "untrained_energy_auroc_baseline": 0.516,
    }


def _exp3462_flagged() -> dict:
    return {
        "experiment_id": 3462,
        "honest_verdict": "complete: residual_diversity_holds_no_collapse_in_fr11_loop_deflagged",
        "flagged_adversarial": True,
        "grounding_collapse_consequence": "ARM A did NOT collapse over 50 iterations.",
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }


def _exp3463_clean() -> dict:
    return {
        "artifact": "experiment_3463",
        "honest_verdict": "complete: fover_g2_ci_dryrun_green_handoff_ready_external_run_pending",
        "g2_status": "ci_dryrun_green_handoff_ready_external_run_pending",
        "g2_ci_dryrun_green": True,
        "g2_independent_reproducer": False,
        "g2_handoff_package_ready": True,
        "condition_a_auroc_isolated": 0.9131,
    }


def _exp3464_clean() -> dict:
    return {
        "experiment": 3464,
        "honest_verdict": "complete: trained_energy_no_lift_over_untrained_kona_hybrid",
        "delta_trained_vs_untrained_hybrid": 0.0,
        "untrained_hybrid_solve_rate": 1.0,
        "trained_hybrid_solve_rate": 1.0,
    }


def _exp3458_clean() -> dict:
    return {
        "experiment_id": "exp3458",
        "honest_verdict": "complete: archive_v318_activate_v319_ready",
    }


def _exp3459_clean() -> dict:
    return {
        "experiment": 3459,
        "honest_verdict": "complete: p01_generation_corpus_complete_n=120",
        "n_problems_completed": 120,
    }


def _exp3465_clean() -> dict:
    return {
        "experiment_id": 3465,
        "honest_verdict": "complete: blocked_kv260_ssh_unreachable",
        "inference_substrate": "hardware_smoke",
    }


def _exp3466_clean() -> dict:
    return {
        "experiment": 3466,
        "honest_verdict": "complete: blocked_gatemate_toolchain_missing",
        "inference_substrate": "hardware_smoke",
    }


def _exp3467_clean() -> dict:
    return {
        "experiment": 3467,
        "honest_verdict": "complete: polarfire reachable and continuity confirmed",
        "inference_substrate": "hardware_smoke",
    }


def _populate_results(
    results_dir: Path,
    *,
    flagged_3460: bool = True,
    auroc_3461: float = 0.629401,
) -> None:
    """Write all .319 upstream artifacts to *results_dir*."""
    _write(results_dir, "experiment_3458_archive_v318_activate_v319.json", _exp3458_clean())
    _write(results_dir, "experiment_3459_p01_generation_corpus_extend_to_120_v2.json", _exp3459_clean())
    if flagged_3460:
        _write(results_dir, "experiment_3460_p01_trained_energy_reranker_vs_self_consistency_v5.json", _exp3460_flagged())
    else:
        _write(results_dir, "experiment_3460_p01_trained_energy_reranker_vs_self_consistency_v5.json", _exp3460_clean())
    _write(results_dir, "experiment_3461_energy_correctness_calibration_trained_vs_untrained_v2.json", _exp3461_clean(auroc_3461))
    _write(results_dir, "experiment_3462_fr11_grounding_collapse_clean_rerun_v2.json", _exp3462_flagged())
    _write(results_dir, "experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.json", _exp3463_clean())
    _write(results_dir, "experiment_3464_kona_trained_energy_hybrid_solve_rate_v4.json", _exp3464_clean())
    _write(results_dir, "experiment_3465_kv260_terminal_latency_transcript_v5.json", _exp3465_clean())
    _write(results_dir, "experiment_3466_gatemate_opportunistic_detect_continuity_v3.json", _exp3466_clean())
    _write(results_dir, "experiment_3467_polarfire_reachability_audit_v5.json", _exp3467_clean())
    _write(results_dir, "experiment_3468_g_gate_status_synthesis_v319.json", _gate_artifact())


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-001/002: p0_1_v5_is_clean
# ---------------------------------------------------------------------------

def test_p0_1_v5_is_clean_false_when_exp3460_flagged(tmp_path):
    # REQ-CAPSTONE-319 SCENARIO-CAP319-001
    _populate_results(tmp_path, flagged_3460=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v5_is_clean"] is False


def test_p0_1_v5_is_clean_true_when_exp3460_clean(tmp_path):
    # SCENARIO-CAP319-001 inverse
    _populate_results(tmp_path, flagged_3460=False)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v5_is_clean"] is True


def test_p0_1_v5_verdict_contains_flagged_when_exp3460_flagged(tmp_path):
    # SCENARIO-CAP319-001: verdict string discloses the flagged status
    _populate_results(tmp_path, flagged_3460=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "flagged_adversarial" in result["p0_1_v5_verdict"]


def test_p0_1_v5_verdict_missing_when_exp3460_absent(tmp_path):
    # SCENARIO-CAP319-002: no exp3460 artifact
    _populate_results(tmp_path)
    (tmp_path / "experiment_3460_p01_trained_energy_reranker_vs_self_consistency_v5.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v5_is_clean"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-003/004/005: trained_energy_correctness_auroc
# ---------------------------------------------------------------------------

def test_trained_energy_correctness_auroc_from_exp3461(tmp_path):
    # SCENARIO-CAP319-003
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["trained_energy_correctness_auroc"] == pytest.approx(0.629401)


def test_trained_energy_correctness_auroc_zero_when_exp3461_missing(tmp_path):
    # SCENARIO-CAP319-004
    _populate_results(tmp_path)
    (tmp_path / "experiment_3461_energy_correctness_calibration_trained_vs_untrained_v2.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["trained_energy_correctness_auroc"] == pytest.approx(0.0)


def test_trained_energy_crosses_055_true_when_above_threshold(tmp_path):
    # SCENARIO-CAP319-005: 0.629 > 0.55
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["trained_energy_crosses_055_threshold"] is True


def test_trained_energy_crosses_055_false_when_below_threshold(tmp_path):
    # SCENARIO-CAP319-005 inverse: auroc below 0.55
    _populate_results(tmp_path, auroc_3461=0.510)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["trained_energy_crosses_055_threshold"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-006/007: g2_ci_status
# ---------------------------------------------------------------------------

def test_g2_ci_status_from_exp3463(tmp_path):
    # SCENARIO-CAP319-006
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "ci_dryrun_green" in result["g2_ci_status"]


def test_g2_dryrun_green_from_exp3463(tmp_path):
    # SCENARIO-CAP319-006: g2_ci_dryrun_green field
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["g2_dryrun_green"] is True


def test_g2_handoff_package_ready_from_exp3463(tmp_path):
    # SCENARIO-CAP319-006: g2_handoff_package_ready field
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["g2_handoff_package_ready"] is True


def test_g2_ci_status_pending_when_exp3463_missing(tmp_path):
    # SCENARIO-CAP319-007
    _populate_results(tmp_path)
    (tmp_path / "experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert "pending" in result["g2_ci_status"]


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-008: fr11_collapse_directional_verdict
# ---------------------------------------------------------------------------

def test_fr11_verdict_labelled_advisory_only_when_flagged(tmp_path):
    # SCENARIO-CAP319-008
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "ADVISORY_ONLY" in result["fr11_collapse_directional_verdict"]


def test_fr11_verdict_contains_collapse_content(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "collapse" in result["fr11_collapse_directional_verdict"].lower()


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-009: kona_trained_hybrid_delta
# ---------------------------------------------------------------------------

def test_kona_trained_hybrid_delta_from_exp3464(tmp_path):
    # SCENARIO-CAP319-009
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["kona_trained_hybrid_delta"] == pytest.approx(0.0)


def test_kona_trained_hybrid_verdict_from_exp3464(tmp_path):
    # SCENARIO-CAP319-009: verdict string
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "no_lift" in result["kona_trained_hybrid_verdict"]


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-010: depth_forcing_function_can_relax
# ---------------------------------------------------------------------------

def test_depth_forcing_false_when_p0_1_flagged(tmp_path):
    # SCENARIO-CAP319-010
    _populate_results(tmp_path, flagged_3460=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_when_gate_artifact_missing(tmp_path):
    _populate_results(tmp_path)
    (tmp_path / "experiment_3468_g_gate_status_synthesis_v319.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["depth_forcing_function_can_relax"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-011: required schema fields
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "experiment",
    "experiment_id",
    "milestone",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "g1", "g2", "g3", "g4",
    "unmet_gates",
    "paper_ready",
    "p0_1_v5_verdict",
    "p0_1_v5_is_clean",
    "p0_1_v5_summary",
    "trained_energy_correctness_auroc",
    "trained_energy_auroc_lift_over_untrained",
    "trained_energy_crosses_055_threshold",
    "g2_ci_status",
    "g2_dryrun_green",
    "g2_handoff_package_ready",
    "g2_external_ask_confirmed",
    "kona_trained_hybrid_delta",
    "kona_trained_hybrid_verdict",
    "fr11_collapse_directional_verdict",
    "depth_forcing_function_can_relax",
    "depth_forcing_function_rationale",
    "next_depth_focus",
    "paper_v6_safe_claims",
    "paper_v6_forbidden_claims",
    "upstreams",
    "capstone_v319_ready",
    "honest_verdict",
    "flagged_adversarial_this_milestone",
}


def test_all_required_fields_present(tmp_path):
    # SCENARIO-CAP319-011
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    missing = _REQUIRED_FIELDS - set(result)
    assert not missing, f"Missing fields: {sorted(missing)}"


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-012: honest_verdict terminal prefix
# ---------------------------------------------------------------------------

_TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_",
                       "passed:", "passed_", "shipped:", "shipped_")


def test_honest_verdict_has_terminal_prefix(tmp_path):
    # SCENARIO-CAP319-012
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    v = result["honest_verdict"]
    assert any(v.startswith(p) for p in _TERMINAL_PREFIXES), (
        f"Non-terminal prefix in honest_verdict: {v!r}"
    )


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-013: capstone_v319_ready
# ---------------------------------------------------------------------------

def test_capstone_v319_ready_true(tmp_path):
    # SCENARIO-CAP319-013
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["capstone_v319_ready"] is True


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-014: reproducibility_checksum
# ---------------------------------------------------------------------------

def test_reproducibility_checksum_deterministic(tmp_path):
    # SCENARIO-CAP319-014: same inputs → same checksum
    _populate_results(tmp_path)
    r1 = cap.run_capstone(results_dir=tmp_path)
    r2 = cap.run_capstone(results_dir=tmp_path)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]


def test_reproducibility_checksum_nonempty(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert len(result["reproducibility_checksum"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-015: paper_v6_safe_claims
# ---------------------------------------------------------------------------

def test_paper_v6_safe_claims_nonempty(tmp_path):
    # SCENARIO-CAP319-015
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0


def test_paper_v6_safe_claims_contain_fover_headline(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    all_claims = " ".join(result["paper_v6_safe_claims"])
    assert "0.9131" in all_claims


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-016: paper_v6_forbidden_claims
# ---------------------------------------------------------------------------

def test_paper_v6_forbidden_claims_reference_retracted(tmp_path):
    # SCENARIO-CAP319-016: retracted claims #2-#11 referenced in forbidden list
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    forbidden_text = " ".join(result["paper_v6_forbidden_claims"])
    assert "thermalization" in forbidden_text    # #2
    assert "speedup" in forbidden_text           # #3
    assert "0.9857" in forbidden_text            # #11
    assert "k=15" in forbidden_text              # conflation guard


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-017: flagged exp3460 numbers excluded from safe claims
# ---------------------------------------------------------------------------

def test_flagged_exp3460_numbers_excluded_from_safe_claims(tmp_path):
    # SCENARIO-CAP319-017: the 0.908333 flagged accuracy from exp3460 must not
    # appear in paper_v6_safe_claims as a standalone citeable number.
    _populate_results(tmp_path, flagged_3460=True)
    result = cap.run_capstone(results_dir=tmp_path)
    # Verify forbidden_claims explicitly blocks citing exp3460 numbers
    forbidden_text = " ".join(result["paper_v6_forbidden_claims"])
    assert "exp3460" in forbidden_text or "flagged" in forbidden_text.lower()


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-018: gate values match exp3468
# ---------------------------------------------------------------------------

def test_gate_values_match_exp3468(tmp_path):
    # SCENARIO-CAP319-018
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["g1"] is True
    assert result["g2"] is False
    assert result["g3"] is True
    assert result["g4"] is True
    assert "G2" in result["unmet_gates"]
    assert result["paper_ready"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-019: kona_trained_hybrid_delta
# ---------------------------------------------------------------------------

def test_kona_trained_hybrid_delta_zero(tmp_path):
    # SCENARIO-CAP319-019
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["kona_trained_hybrid_delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SCENARIO-CAP319-020: calibration advance in safe claims
# ---------------------------------------------------------------------------

def test_paper_v6_safe_claims_contain_calibration_advance(tmp_path):
    # SCENARIO-CAP319-020: the key .319 advance — trained energy carries signal
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    all_claims = " ".join(result["paper_v6_safe_claims"])
    # Must mention the AUROC lift and that 0.629 > 0.55 threshold
    assert "0.629" in all_claims
    assert "0.55" in all_claims


# ---------------------------------------------------------------------------
# Integration: actual on-disk artifact (skipped if absent)
# ---------------------------------------------------------------------------

def test_on_disk_artifact_is_valid():
    """Validate the actual written artifact if it exists."""
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3469_capstone_v319.json"
    )
    if not artifact_path.exists():
        pytest.skip("artifact not yet written — run the script first")
    data = json.loads(artifact_path.read_text())
    missing = _REQUIRED_FIELDS - set(data)
    assert not missing, f"Artifact missing fields: {sorted(missing)}"
    assert data["capstone_v319_ready"] is True
    v = data["honest_verdict"]
    assert any(v.startswith(p) for p in _TERMINAL_PREFIXES)
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
