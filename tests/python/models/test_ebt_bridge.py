"""Tests for EBTBridge.

Spec: REQ-INFER-018, SCENARIO-INFER-018-001
"""

import math

from carnot.models.ebt_bridge import EBTBridge


class MockModel:
    def __init__(self, logprob_dict):
        self.logprob_dict = logprob_dict

    def get_sequence_logprob(self, text: str) -> float:
        return self.logprob_dict.get(text, -math.inf)


def test_sequence_energy():
    """SCENARIO-INFER-018-001: Sequence Energy Calculation."""
    mock_model = MockModel({
        "highly likely sequence": -1.0,
        "unlikely sequence": -10.0
    })
    
    bridge = EBTBridge(mock_model)
    
    energy_high_prob = bridge.sequence_energy("highly likely sequence")
    energy_low_prob = bridge.sequence_energy("unlikely sequence")
    
    assert isinstance(energy_high_prob, float)
    assert energy_high_prob < energy_low_prob
