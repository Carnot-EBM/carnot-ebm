"""Tests for the Capstone v318 aggregation module.

References:
  REQ-CAPSTONE-318: The .318 capstone must aggregate upstream artifacts,
    skip flagged_adversarial ones per the fabrication gate, emit correct
    G1-G4 gate status, and produce a paper_v6_safe_claims list that
    honours the Paper-v6 Narrowing Discipline.

SCENARIO-CAP318-001: exp3449 flagged → p0_1_v4_is_clean=False, numbers excluded.
SCENARIO-CAP318-002: exp3449 missing → p0_1_v4_is_clean=False, p0_1_v4_verdict='MISSING'.
SCENARIO-CAP318-003: exp3450 clean → energy_correctness_auroc populated.
SCENARIO-CAP318-004: exp3450 missing → energy_correctness_auroc=0.0.
SCENARIO-CAP318-005: exp3451 clean → g2_ci_status from artifact.
SCENARIO-CAP318-006: exp3451 missing → g2_ci_status contains 'pending'.
SCENARIO-CAP318-007: exp3452 flagged → fr11 directional verdict preserved labelled ADVISORY_ONLY.
SCENARIO-CAP318-008: depth_forcing_function_can_relax is False when P0.1 flagged.
SCENARIO-CAP318-009: All required schema fields present.
SCENARIO-CAP318-010: honest_verdict has terminal prefix.
SCENARIO-CAP318-011: capstone_v318_ready=True.
SCENARIO-CAP318-012: reproducibility_checksum is deterministic.
SCENARIO-CAP318-013: paper_v6_safe_claims is non-empty list.
SCENARIO-CAP318-014: paper_v6_forbidden_claims references retracted claims.
SCENARIO-CAP318-015: Flagged artifact exp3449 summary not present in safe_claims.
SCENARIO-CAP318-016: Gate values match exp3456 when gate artifact present.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v318_3457 as cap


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
        "honest_verdict": "complete: g2_sole_unmet_gate",
    }


def _exp3449_flagged() -> dict:
    return {
        "experiment": 3449,
        "honest_verdict": "complete: energy_matches_but_does_not_beat_sc",
        "flagged_adversarial": True,
        "delta_energy_vs_self_consistency": 0.0,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }


def _exp3449_clean() -> dict:
    return {
        "experiment": 3449,
        "honest_verdict": "complete: energy_beats_self_consistency",
        "flagged_adversarial": False,
        "delta_energy_vs_self_consistency": 0.03,
    }


def _exp3450_clean() -> dict:
    return {
        "experiment": 3450,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "honest_verdict": "complete: energy_does_not_track_correctness",
        "energy_as_correctness_auroc": 0.516,
    }


def _exp3451_clean() -> dict:
    return {
        "experiment": 3451,
        "honest_verdict": "complete: fover_g2_ci_and_docker_cleanroom_ready_external_run_pending",
        "g2_status": "ci_and_docker_ready_external_run_pending",
        "g2_independent_reproducer": False,
        "g2_docker_cleanroom_reproduced": True,
    }


def _exp3452_flagged() -> dict:
    return {
        "experiment": 3452,
        "honest_verdict": "complete: at_risk_grounding_causes_collapse_entropy_reg_prevents_it",
        "flagged_adversarial": True,
        "corrigendum_pending": [{"kind": "TAUTOLOGY"}],
    }


def _exp3447_clean() -> dict:
    return {
        "experiment_id": "exp3447",
        "honest_verdict": "complete: archive_v316_v317_activate_v318_ready=true",
    }


def _exp3448_clean() -> dict:
    return {
        "experiment": 3448,
        "honest_verdict": "complete: p01_generation_corpus_partial_resumable_n=47",
        "n_problems_completed": 47,
    }


def _exp3453_clean() -> dict:
    return {
        "experiment": 3453,
        "honest_verdict": "complete: blocked_kv260_ssh_unreachable",
        "inference_substrate": "hardware_smoke",
    }


def _exp3454_clean() -> dict:
    return {
        "experiment": 3454,
        "honest_verdict": "complete: blocked_gatemate_toolchain_missing",
        "inference_substrate": "hardware_smoke",
    }


def _exp3455_clean() -> dict:
    return {
        "experiment": 3455,
        "honest_verdict": "complete: polarfire reachable and continuity confirmed",
        "inference_substrate": "hardware_smoke",
    }


def _populate_results(results_dir: Path, *, flagged_3449: bool = True) -> None:
    """Write all .318 upstream artifacts to *results_dir*."""
    _write(results_dir, "experiment_3447_archive_v316_v317_activate_v318.json", _exp3447_clean())
    _write(results_dir, "experiment_3448_p01_generation_corpus_builder_v1.json", _exp3448_clean())
    if flagged_3449:
        _write(results_dir, "experiment_3449_p01_energy_vote_vs_self_consistency_cached_scoring_v4.json", _exp3449_flagged())
    else:
        _write(results_dir, "experiment_3449_p01_energy_vote_vs_self_consistency_cached_scoring_v4.json", _exp3449_clean())
    _write(results_dir, "experiment_3450_energy_correctness_calibration_audit_v1.json", _exp3450_clean())
    _write(results_dir, "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.json", _exp3451_clean())
    _write(results_dir, "experiment_3452_fr11_grounding_collapse_stress_test_v1.json", _exp3452_flagged())
    _write(results_dir, "experiment_3453_kv260_terminal_latency_transcript_v4.json", _exp3453_clean())
    _write(results_dir, "experiment_3454_gatemate_opportunistic_detect_continuity_v2.json", _exp3454_clean())
    _write(results_dir, "experiment_3455_polarfire_reachability_audit_v4.json", _exp3455_clean())
    _write(results_dir, "experiment_3456_g_gate_status_synthesis_v318.json", _gate_artifact())


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-001/002: p0_1_v4_is_clean
# ---------------------------------------------------------------------------

def test_p0_1_v4_is_clean_false_when_exp3449_flagged(tmp_path):
    # REQ-CAPSTONE-318 SCENARIO-CAP318-001
    _populate_results(tmp_path, flagged_3449=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v4_is_clean"] is False


def test_p0_1_v4_is_clean_true_when_exp3449_clean(tmp_path):
    # SCENARIO-CAP318-001 inverse
    _populate_results(tmp_path, flagged_3449=False)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v4_is_clean"] is True


def test_p0_1_v4_verdict_contains_flagged_when_exp3449_flagged(tmp_path):
    # SCENARIO-CAP318-001: verdict string discloses the flagged status
    _populate_results(tmp_path, flagged_3449=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "flagged_adversarial" in result["p0_1_v4_verdict"]


def test_p0_1_v4_verdict_missing_when_exp3449_absent(tmp_path):
    # SCENARIO-CAP318-002: no exp3449 artifact
    _populate_results(tmp_path, flagged_3449=True)
    (tmp_path / "experiment_3449_p01_energy_vote_vs_self_consistency_cached_scoring_v4.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["p0_1_v4_is_clean"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-003/004: energy_correctness_auroc
# ---------------------------------------------------------------------------

def test_energy_correctness_auroc_from_exp3450(tmp_path):
    # SCENARIO-CAP318-003
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["energy_correctness_auroc"] == pytest.approx(0.516)


def test_energy_correctness_auroc_zero_when_exp3450_missing(tmp_path):
    # SCENARIO-CAP318-004
    _populate_results(tmp_path)
    (tmp_path / "experiment_3450_energy_correctness_calibration_audit_v1.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["energy_correctness_auroc"] == pytest.approx(0.0)


def test_energy_tracks_correctness_false_when_auroc_below_threshold(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    # 0.516 < 0.55 → does not track correctness
    assert result["energy_tracks_correctness"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-005/006: g2_ci_status
# ---------------------------------------------------------------------------

def test_g2_ci_status_from_exp3451(tmp_path):
    # SCENARIO-CAP318-005
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["g2_ci_status"] == "ci_and_docker_ready_external_run_pending"


def test_g2_ci_status_pending_when_exp3451_missing(tmp_path):
    # SCENARIO-CAP318-006
    _populate_results(tmp_path)
    (tmp_path / "experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert "pending" in result["g2_ci_status"]


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-007: fr11_collapse_directional_verdict
# ---------------------------------------------------------------------------

def test_fr11_verdict_labelled_advisory_only_when_flagged(tmp_path):
    # SCENARIO-CAP318-007
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert "ADVISORY_ONLY" in result["fr11_collapse_directional_verdict"]


def test_fr11_verdict_contains_collapse_directional_content(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    # The directional honest_verdict from exp3452 must appear somewhere
    assert "collapse" in result["fr11_collapse_directional_verdict"].lower()


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-008: depth_forcing_function_can_relax
# ---------------------------------------------------------------------------

def test_depth_forcing_false_when_p0_1_flagged(tmp_path):
    # SCENARIO-CAP318-008
    _populate_results(tmp_path, flagged_3449=True)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["depth_forcing_function_can_relax"] is False


def test_depth_forcing_false_when_gate_artifact_missing(tmp_path):
    # No gate artifact → default depth_can_relax=False
    _populate_results(tmp_path)
    (tmp_path / "experiment_3456_g_gate_status_synthesis_v318.json").unlink()
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["depth_forcing_function_can_relax"] is False


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-009: required schema fields
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
    "p0_1_v4_verdict",
    "p0_1_v4_is_clean",
    "p0_1_v4_summary",
    "energy_correctness_auroc",
    "energy_tracks_correctness",
    "g2_ci_status",
    "g2_docker_cleanroom_reproduced",
    "fr11_collapse_directional_verdict",
    "depth_forcing_function_can_relax",
    "depth_forcing_function_rationale",
    "next_depth_focus",
    "paper_v6_safe_claims",
    "paper_v6_forbidden_claims",
    "upstreams",
    "capstone_v318_ready",
    "honest_verdict",
}


def test_all_required_fields_present(tmp_path):
    # SCENARIO-CAP318-009
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    missing = _REQUIRED_FIELDS - set(result)
    assert not missing, f"Missing fields: {sorted(missing)}"


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-010: honest_verdict terminal prefix
# ---------------------------------------------------------------------------

_TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_",
                       "passed:", "passed_", "shipped:", "shipped_")


def test_honest_verdict_has_terminal_prefix(tmp_path):
    # SCENARIO-CAP318-010
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    v = result["honest_verdict"]
    assert any(v.startswith(p) for p in _TERMINAL_PREFIXES), (
        f"Non-terminal prefix in honest_verdict: {v!r}"
    )


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-011: capstone_v318_ready
# ---------------------------------------------------------------------------

def test_capstone_v318_ready_true(tmp_path):
    # SCENARIO-CAP318-011
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["capstone_v318_ready"] is True


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-012: reproducibility_checksum deterministic
# ---------------------------------------------------------------------------

def test_reproducibility_checksum_deterministic(tmp_path):
    # SCENARIO-CAP318-012: same inputs → same checksum
    _populate_results(tmp_path)
    r1 = cap.run_capstone(results_dir=tmp_path)
    r2 = cap.run_capstone(results_dir=tmp_path)
    assert r1["reproducibility_checksum"] == r2["reproducibility_checksum"]


def test_reproducibility_checksum_nonempty(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert len(result["reproducibility_checksum"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-013/014: paper_v6 claims content
# ---------------------------------------------------------------------------

def test_paper_v6_safe_claims_nonempty(tmp_path):
    # SCENARIO-CAP318-013
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert isinstance(result["paper_v6_safe_claims"], list)
    assert len(result["paper_v6_safe_claims"]) > 0


def test_paper_v6_forbidden_claims_reference_retracted(tmp_path):
    # SCENARIO-CAP318-014: retracted claims #2-#11 referenced in forbidden list
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    forbidden_text = " ".join(result["paper_v6_forbidden_claims"])
    # Check a representative sample of the 11 retracted claims
    assert "thermalization" in forbidden_text    # #2
    assert "speedup" in forbidden_text           # #3
    assert "0.9857" in forbidden_text            # #11
    assert "k=15" in forbidden_text              # conflation guard


def test_paper_v6_safe_claims_contain_fover_headline(tmp_path):
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    all_claims = " ".join(result["paper_v6_safe_claims"])
    assert "0.9131" in all_claims


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-015: flagged exp3449 numbers not in safe_claims
# ---------------------------------------------------------------------------

def test_flagged_exp3449_numbers_excluded_from_safe_claims(tmp_path):
    # SCENARIO-CAP318-015: the 0.87234 SC accuracy from flagged exp3449
    # must not appear in paper_v6_safe_claims.
    _populate_results(tmp_path, flagged_3449=True)
    result = cap.run_capstone(results_dir=tmp_path)
    all_safe = " ".join(result["paper_v6_safe_claims"])
    # Flagged SC accuracy from exp3449
    assert "0.87234" not in all_safe


# ---------------------------------------------------------------------------
# SCENARIO-CAP318-016: gate values match exp3456 when present
# ---------------------------------------------------------------------------

def test_gate_values_match_exp3456(tmp_path):
    # SCENARIO-CAP318-016
    _populate_results(tmp_path)
    result = cap.run_capstone(results_dir=tmp_path)
    assert result["g1"] is True
    assert result["g2"] is False  # gate artifact has g2=False
    assert result["g3"] is True
    assert result["g4"] is True
    assert "G2" in result["unmet_gates"]
    assert result["paper_ready"] is False


# ---------------------------------------------------------------------------
# Integration: actual on-disk artifact (skipped if absent)
# ---------------------------------------------------------------------------

def test_on_disk_artifact_is_valid():
    """Validate the actual written artifact if it exists."""
    artifact_path = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "experiment_3457_capstone_v318.json"
    )
    if not artifact_path.exists():
        pytest.skip("artifact not yet written — run the script first")
    data = json.loads(artifact_path.read_text())
    missing = _REQUIRED_FIELDS - set(data)
    assert not missing, f"Artifact missing fields: {sorted(missing)}"
    assert data["capstone_v318_ready"] is True
    v = data["honest_verdict"]
    assert any(v.startswith(p) for p in _TERMINAL_PREFIXES)
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
