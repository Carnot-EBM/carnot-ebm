import pytest
from carnot.verify.interwhen_monitor import (
    evaluate_constraint_satisfaction,
    evaluate_evidence_presence,
    evaluate_unsupported_commitment,
    score_trajectory,
    MonitorTrajectoryFeatures
)

def test_interwhen_monitor_signals():
    # REQ-VERIFY-3332
    state1 = "I am thinking about constraintA and constraintB."
    assert evaluate_constraint_satisfaction(state1, ["constraintA", "constraintB", "constraintC"]) == 2/3
    
    assert evaluate_constraint_satisfaction(state1, []) == 1.0

    state2 = "Because of this, we know that..."
    assert evaluate_evidence_presence(state2) > 0.0
    
    state3 = "The answer is obviously 42."
    assert evaluate_unsupported_commitment(state3) == 1.0

def test_trajectory_scoring():
    # REQ-VERIFY-3332
    states = [
        "Let's start by looking at constraintA.",
        "Since constraintA holds, it implies constraintB.",
        "Therefore, the answer is safe."
    ]
    features = score_trajectory(states, ["constraintA", "constraintB"])
    assert len(features.constraint_satisfaction_trend) == 3
    assert features.evidence_presence_trend[1] > 0
    assert features.unsupported_commitment_trend[2] > 0
