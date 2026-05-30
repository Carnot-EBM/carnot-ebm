"""Tests for EBM-CoT Scaling and Compute Savings (Exp 3411)."""

from carnot.pipeline.ebm_cot_trajectory import EBMCoTTrajectoryVerifier

def test_compute_savings_req_verify_3411():
    """Test compute time savings calculation as required by REQ-VERIFY-3411."""
    verifier = EBMCoTTrajectoryVerifier()
    
    # 5 states, spikes on 3rd state (index 2)
    # raw_scores ~ 10 / (len + 1)
    # len=9 -> 1.0
    # len=1 -> 5.0 (Spike of 4.0!)
    states = [
        "123456789",  # raw: 1.0, cal: 1.0
        "123456789",  # raw: 1.0, cal: 1.0
        "x",          # raw: 5.0, cal: 0.7*5.0 + 0.3*1.0 = 3.8 -> SPIKE!
        "y",
        "z"
    ]
    
    res = verifier.verify_trajectory(states)
    
    assert res["rejected"]
    assert res["early_commitment_detected"]
    
    # Check that we compute stats about states saved
    assert "total_states" in res
    assert "states_evaluated" in res
    assert "states_saved" in res
    
    assert res["total_states"] == 5
    assert res["states_evaluated"] == 3
    assert res["states_saved"] == 2
