"""Tests for EBM-CoT Trajectory Verifier."""

from __future__ import annotations

from carnot.pipeline.ebm_cot_trajectory import EBMCoTTrajectoryVerifier


def test_ebm_cot_trajectory_verifier_init():
    """Test initialization of verifier."""
    specs = [{"name": "Mock"}]
    verifier = EBMCoTTrajectoryVerifier(specs)
    assert verifier.specs == specs
    
    verifier_empty = EBMCoTTrajectoryVerifier()
    assert verifier_empty.specs == []


def test_score_step():
    """Test scoring logic mock."""
    verifier = EBMCoTTrajectoryVerifier()
    assert verifier.score_step("") == 10.0
    # "hello" has len 5, so 10.0 / (5 + 1.0) = 1.666...
    assert round(verifier.score_step("hello"), 2) == 1.67


def test_verify_trajectory_success():
    """Test a successful trajectory without energy spikes."""
    verifier = EBMCoTTrajectoryVerifier()
    # Mock scores will be calculated from length.
    # We want energies to be non-increasing or mildly increasing.
    # lengths: 10, 10, 10 -> same lengths => same energy, no spike.
    states = [
        "1234567890",
        "1234567890",
        "1234567890"
    ]
    res = verifier.verify_trajectory(states)
    assert not res["rejected"]
    assert not res["early_commitment_detected"]
    assert len(res["energies"]) == 3


def test_verify_trajectory_spike():
    """Test a trajectory that gets rejected due to an energy spike."""
    verifier = EBMCoTTrajectoryVerifier()
    # "A very long sentence that has low energy" -> len ~ 40 -> energy ~ 0.24
    # "Short" -> len 5 -> energy 1.67. This is a spike of ~1.4.
    # Wait, the spike threshold is 1.5 in the class `energy > energies[-1] + 1.5`.
    # Let's make it short enough:
    # "Very very very long sentence that represents high confidence" (len 60) -> energy 0.16
    # "x" (len 1) -> energy 5.0 -> Spike!
    states = [
        "Very very very long sentence that represents high confidence",
        "x",
        "y"
    ]
    res = verifier.verify_trajectory(states)
    assert res["rejected"]
    assert res["early_commitment_detected"]
    # The verifier aborts evaluation early on a detected spike (the compute-savings
    # design — see `states_saved`): it appends the spiking step's energy then breaks,
    # so the spike at step 1 yields exactly 2 calibrated energies, not 3.
    assert len(res["energies"]) == 2
    assert res["states_evaluated"] == 2
    assert res["states_saved"] == 1
