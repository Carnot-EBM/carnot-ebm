"""Tests for exp3504 — Archive v322 and Activate v323.

Covers REQ-RETRO-010 (SCENARIO-RETRO-010): milestone archive artifacts must
capture the key finding (.322 architecture held — P0.1 produced honest scientific
diagnoses on both routes for the first time), carry the correct publication gate
status, record the flagged tautology artifacts, and emit a `complete:` prefixed
honest_verdict with random_seed != experiment_number.
"""

from __future__ import annotations

import json
import time

import pytest

import scripts.experiment_3504_archive_v322_activate_v323 as module

# ---------------------------------------------------------------------------
# Shared fake upstream for isolated unit tests (no filesystem reads)
# ---------------------------------------------------------------------------

FAKE_UPSTREAM: dict[str, dict] = {
    "exp3494": {
        "honest_verdict": "complete: blocked_kona_failure_is_representational_not_optimizer",
        "encoding_validity_E0": {"is_valid": True, "total_energy": 0.0},
        "easy_tier_solve_rate": 0.0,
        "inference_substrate": "ising_energy_optimization_cpu",
    },
    "exp3495": {
        "honest_verdict": "complete: blocked_contested_subset_too_small_n=21",
        "contested_subset_n": 21,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3496": {
        "honest_verdict": "complete: blocked_no_in_band_split_found_sc_outside_band",
        "n_problems_completed": 27,
        "per_level_probe_sc": {"3": 0.5, "4": 0.625},
        "inference_substrate": "live_llm_inference",
    },
    "exp3497": {
        "honest_verdict": "complete: mathaware_recalibration_recovers_correctness_signal",
        "step_vs_final_auroc_gap": 0.13795,
        "mathaware_recalibrated_correctness_auroc": 0.624931,
        "process_energy_correctness_auroc": 0.60102,
        "flagged_adversarial": False,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3498": {
        "honest_verdict": "complete: beta_min_predictable_from_lambda_min",
        "recommended_phase5_rule": "beta_min = -0.3001 + 1.8461 * lambda_min (R²=0.989)",
        "beta_min_lambda_min_fit": {"r_squared": 0.9886},
        "law_holds_out_of_sample": True,
        "flagged_adversarial": False,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3499": {
        "honest_verdict": "complete: fover_g2_package_regression_clean_external_ask_ready",
        "package_reproduced_auroc": 0.9131,
        "package_auroc_within_ci": True,
        "g2_met": False,
        "external_run_pending": True,
        "package_sha256": "deadbeef",
        "package_cid": "QmFake",
        "flagged_adversarial": False,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    },
    "exp3500": {
        "honest_verdict": "complete: blocked_kv260_ssh_unreachable",
        "kv260_terminal_state_reached": False,
        "inference_substrate": "hardware_smoke",
    },
    "exp3501": {
        "honest_verdict": "complete: polarfire reachable and continuity confirmed deflagged",
        "inference_substrate": "hardware_smoke",
    },
    "exp3502": {
        "honest_verdict": "complete: g1_g3_g4_met_g2_pending_p01_both_routes_blocked",
        "flagged_adversarial": True,
        "random_seed": 3502,
        "inference_substrate": "aggregation_from_upstream_artifacts",
    },
    "exp3503": {
        "honest_verdict": "complete: capstone_v322_ready=true",
        "flagged_adversarial": True,
        "random_seed": 3503,
        "upstreams": {
            "exp3502": "SKIPPED_flagged_adversarial (directional: complete: g1/g3/g4 met)"
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
    },
}


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: verdict prefix
# ---------------------------------------------------------------------------


def test_honest_verdict_starts_with_complete() -> None:
    """REQ-RETRO-010 SCENARIO-RETRO-010: honest_verdict must start with 'complete:'."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["honest_verdict"].startswith("complete:"), (
        f"honest_verdict must start with 'complete:', got: {artifact['honest_verdict']!r}"
    )


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: archive ready flag
# ---------------------------------------------------------------------------


def test_archive_v323_ready_flag() -> None:
    """REQ-RETRO-010: archive_v322_activate_v323_ready must be True."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["archive_v322_activate_v323_ready"] is True


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: random seed is NOT the experiment number
# ---------------------------------------------------------------------------


def test_random_seed_is_not_experiment_number() -> None:
    """REQ-RETRO-010: random_seed must NOT equal the experiment number (avoids TAUTOLOGY flag)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["random_seed"] != artifact["experiment"], (
        f"random_seed={artifact['random_seed']} must not equal "
        f"experiment={artifact['experiment']} — that triggers the adversarial_verify "
        "TAUTOLOGY flag that affected exp3502/3503."
    )
    assert artifact["random_seed"] == module.RANDOM_SEED


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: .322 architecture held
# ---------------------------------------------------------------------------


def test_p01_architecture_held() -> None:
    """REQ-RETRO-010: p01_architecture_held must be True (.322 first honest-science milestone)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_architecture_held"] is True
    assert artifact["p01_first_honest_science_milestone"] is True


def test_p01_route1_encoding_valid() -> None:
    """REQ-RETRO-010: Route 1 encoding must be recorded as valid (E=0)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_route1_encoding_valid"] is True
    assert artifact["p01_route1_solve_rate"] == pytest.approx(0.0)


def test_p01_route2_contested_n_recorded() -> None:
    """REQ-RETRO-010: Route 2 contested_n must be recorded as 21."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_route2_contested_n"] == 21


def test_p01_has_clean_verdict_false() -> None:
    """REQ-RETRO-010: p01_has_clean_verdict must be False (both routes blocked)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["p01_has_clean_verdict"] is False


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: clean positives propagated
# ---------------------------------------------------------------------------


def test_clean_positives_cal_step_gap() -> None:
    """REQ-RETRO-010: exp3497 step_vs_final_auroc_gap must propagate."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    cal = artifact["clean_positives"]["exp3497_mathaware_calibration"]
    assert cal["step_vs_final_auroc_gap"] == pytest.approx(0.13795)
    assert cal["mathaware_recalibrated_correctness_auroc"] == pytest.approx(0.624931)
    assert cal["flagged"] is False


def test_clean_positives_fr11_law() -> None:
    """REQ-RETRO-010: exp3498 FR-11 beta_min law must propagate."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    fr11 = artifact["clean_positives"]["exp3498_fr11_beta_min_law"]
    assert fr11["r_squared"] == pytest.approx(0.9886)
    assert fr11["law_holds_out_of_sample"] is True
    assert fr11["flagged"] is False


def test_clean_positives_g2_regression() -> None:
    """REQ-RETRO-010: exp3499 G2 regression status must propagate."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    g2 = artifact["clean_positives"]["exp3499_g2_regression_verify"]
    assert g2["package_auroc"] == pytest.approx(0.9131)
    assert g2["package_auroc_within_ci"] is True
    assert g2["g2_met"] is False
    assert g2["external_run_pending"] is True
    assert g2["flagged"] is False


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: flagged artifacts recorded
# ---------------------------------------------------------------------------


def test_flagged_artifacts_listed() -> None:
    """REQ-RETRO-010: exp3502 and exp3503 must be listed as flagged for TAUTOLOGY."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert 3502 in artifact["flagged_artifacts_this_milestone"]
    assert 3503 in artifact["flagged_artifacts_this_milestone"]


def test_flagged_artifacts_note_present() -> None:
    """REQ-RETRO-010: note must explain the tautology is a trivial bug, not fabrication."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    note = artifact["flagged_artifacts_note"]
    assert "TAUTOLOGY" in note
    assert "trivial" in note.lower()
    assert "fabrication" in note.lower()


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: publication gate status
# ---------------------------------------------------------------------------


def test_g2_status_propagated() -> None:
    """REQ-RETRO-010: G2 must be False (external run pending)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gate = artifact["publication_gate_status"]
    assert gate["G2_independent_reproducer"] is False
    assert gate["G2_external_run_pending"] is True
    assert gate["G2_package_auroc"] == pytest.approx(0.9131)
    assert gate["paper_ready"] is False
    assert "G2" in gate["unmet_gates"]
    assert gate["sole_unmet_gate"] == "G2"


def test_g1_g3_g4_met() -> None:
    """REQ-RETRO-010: G1, G3, G4 must be True."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gate = artifact["publication_gate_status"]
    assert gate["G1_headline_measured"] is True
    assert gate["G3_prose_narrowing_clean"] is True
    assert gate["G4_numbers_trace_to_artifacts"] is True


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: forward gap
# ---------------------------------------------------------------------------


def test_forward_gap_mentions_combinatorial_optimizer() -> None:
    """REQ-RETRO-010: forward_gap_top must name real combinatorial optimizers."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gap = artifact["forward_gap_top"]
    assert "SA" in gap or "simulated annealing" in gap.lower() or "QUBO" in gap
    assert "seed tautology" in gap.lower() or "tautology" in gap.lower()


def test_forward_gap_mentions_corpus() -> None:
    """REQ-RETRO-010: forward_gap_top must reference the in-band level-3 corpus."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    gap = artifact["forward_gap_top"]
    assert "level-3" in gap or "level 3" in gap.lower() or "in-band" in gap.lower()


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: schema + milestone fields
# ---------------------------------------------------------------------------


def test_schema_field() -> None:
    """REQ-RETRO-010: schema must be carnot.operational_retro.v66."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["schema"] == "carnot.operational_retro.v66"


def test_milestone_fields() -> None:
    """REQ-RETRO-010: milestone_archived and milestone_activated must be set."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["milestone_archived"] == "2026.05.322"
    assert artifact["milestone_activated"] == "2026.05.323"


def test_inference_substrate() -> None:
    """REQ-RETRO-010: inference_substrate must be aggregation_from_upstream_artifacts."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_duration_positive() -> None:
    """REQ-RETRO-010: duration_s must be positive (floored at 0.001)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic() - 0.5)
    assert artifact["duration_s"] > 0


def test_reproducibility_checksum_present() -> None:
    """REQ-RETRO-010: reproducibility_checksum must be a non-empty hex string."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) > 0


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: experiments_completed
# ---------------------------------------------------------------------------


def test_experiments_completed_all_ten() -> None:
    """REQ-RETRO-010: experiments_completed must cover all 10 .322 experiments."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    ids = {e["id"] for e in artifact["experiments_completed"]}
    # exp3493 (prior archive), exp3494–exp3503
    expected = {3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503}
    assert expected.issubset(ids), f"missing IDs: {expected - ids}"


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: field_provenance
# ---------------------------------------------------------------------------


def test_field_provenance_required_keys() -> None:
    """REQ-RETRO-010: field_provenance must annotate key fields."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    provenance = artifact["field_provenance"]
    required = {
        "honest_verdict",
        "inference_substrate",
        "archive_v322_activate_v323_ready",
        "publication_gate_status",
        "random_seed",
        "duration_s",
        "reproducibility_checksum",
        "cited_upstream_artifacts",
    }
    for key in required:
        assert key in provenance, f"field_provenance missing key: {key!r}"


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: cited upstream artifacts
# ---------------------------------------------------------------------------


def test_cited_upstream_artifacts_all_ten() -> None:
    """REQ-RETRO-010: cited_upstream_artifacts must list all ten upstream sources."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    ids = {a["experiment_id"] for a in artifact["cited_upstream_artifacts"]}
    expected = {
        "exp3494", "exp3495", "exp3496", "exp3497", "exp3498",
        "exp3499", "exp3500", "exp3501", "exp3502", "exp3503",
    }
    assert ids == expected


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: depth forcing function still active
# ---------------------------------------------------------------------------


def test_depth_forcing_function_active() -> None:
    """REQ-RETRO-010: depth_forcing_function_active must be True (P0.1 still open)."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    assert artifact["depth_forcing_function_active"] is True


# ---------------------------------------------------------------------------
# SCENARIO-RETRO-010: JSON serializable
# ---------------------------------------------------------------------------


def test_json_serializable() -> None:
    """REQ-RETRO-010: the full artifact must be JSON-serializable."""
    artifact = module._build_retro(FAKE_UPSTREAM, time.monotonic())
    serialized = json.dumps(artifact)  # must not raise
    roundtripped = json.loads(serialized)
    assert roundtripped["honest_verdict"].startswith("complete:")
