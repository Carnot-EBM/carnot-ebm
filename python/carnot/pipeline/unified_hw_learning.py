"""Unified Hardware Learning Verification Loop.

Spec: REQ-HW-LEARN-101
"""

from typing import Any, Sequence
from carnot.pipeline.verification_loop import VerificationLoop, Violation
from carnot.training.online_updater import OnlineUpdater

class UnifiedHWVerificationLoop(VerificationLoop):
    """Extends VerificationLoop to handle HW PYNQ updates."""
    
    def __init__(self, cikan: Any, updater: OnlineUpdater):
        super().__init__(cikan, updater)
        self.n_uploaded = 0

    def run(self, stream: Sequence[Violation]) -> None:
        """Process a stream, update, and upload weights to FPGA on violations."""
        for item in stream:
            self.n_processed += 1
            
            # Trigger fine-tuning step
            self.updater.step(self.cikan, item.features, item.label)
            self.n_updated += 1
            
            # Re-upload weights to the FPGA
            if hasattr(self.cikan, "upload_weights"):
                self.cikan.upload_weights()
                self.n_uploaded += 1
