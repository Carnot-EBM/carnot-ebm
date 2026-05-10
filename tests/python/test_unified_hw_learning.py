"""Test for Unified HW Verification Loop.

Spec traces: REQ-HW-LEARN-101, SCENARIO-HW-LEARN-101
"""

from carnot.pipeline.unified_hw_learning import UnifiedHWVerificationLoop
from carnot.pipeline.verification_loop import Violation
from carnot.training.online_updater import OnlineUpdater
from carnot.models.cikan_verifier import CIKAN

class MockPYNQCIKAN(CIKAN):
    """Mock for PYNQ-based CIKAN verifier."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upload_count = 0
        
    def upload_weights(self):
        self.upload_count += 1

def test_unified_hw_verification_loop():
    """SCENARIO-HW-LEARN-101: Wire PYNQ-based CIKAN verifier into the VerificationLoop."""
    
    cikan = MockPYNQCIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="sgd", learning_rate=0.01)
    
    loop = UnifiedHWVerificationLoop(cikan, updater)
    
    stream = [
        Violation(features=[0.9, 0.9], label=0.0),
        Violation(features=[0.1, 0.1], label=1.0),
    ]
    
    loop.run(stream)
    
    assert loop.n_processed == 2
    assert loop.n_updated == 2
    assert loop.n_uploaded == 2
    assert cikan.upload_count == 2
