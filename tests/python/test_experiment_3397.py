"""Tests for EBM-CoT Live Benchmark AUROC (Exp 3397)."""

from carnot.pipeline.ebm_cot_trajectory import EBMCoTTrajectoryVerifier

def test_step_wise_energy_calibration_req_verify_3397():
    """Test step-wise energy calibration as required by REQ-VERIFY-3397."""
    verifier = EBMCoTTrajectoryVerifier()
    
    # Test empty list
    assert verifier.apply_step_wise_energy_calibration([]) == []
    
    raw_energies = [1.0, 2.0, 3.0]
    calibrated = verifier.apply_step_wise_energy_calibration(raw_energies)
    
    assert len(calibrated) == len(raw_energies)
    assert calibrated[0] == 1.0
    assert calibrated[1] == 0.7 * 2.0 + 0.3 * 1.0
    assert calibrated[2] == 0.7 * 3.0 + 0.3 * calibrated[1]

def test_verify_trajectory_with_calibration():
    """Test trajectory verification with calibration enabled."""
    verifier = EBMCoTTrajectoryVerifier()
    
    # Generate some strings that would cause raw scores to spike
    # raw_scores ~ 10 / (len + 1)
    # len=9 -> 1.0
    # len=1 -> 5.0 (Spike of 4.0!)
    states = [
        "123456789",  # raw: 1.0
        "x",          # raw: 5.0
    ]
    # Calibrated:
    # [0]: 1.0
    # [1]: 0.7 * 5.0 + 0.3 * 1.0 = 3.5 + 0.3 = 3.8
    # Spike = 3.8 - 1.0 = 2.8 > 1.5, so it should reject.
    res = verifier.verify_trajectory(states)
    
    assert res["rejected"]
    assert res["early_commitment_detected"]
    assert len(res["energies"]) == 2
