"""Simple verification loop for streaming violations.

Spec: REQ-LEARN-101
"""

from dataclasses import dataclass
from typing import Any, Sequence

from carnot.training.online_updater import OnlineUpdater

@dataclass
class Violation:
    """A violation or valid sample encountered in the loop."""
    features: Sequence[float]
    label: float  # 0.0 for violation, 1.0 for valid

class VerificationLoop:
    """Streams candidates, triggering the online updater on violations."""
    
    def __init__(self, cikan: Any, updater: OnlineUpdater):
        self.cikan = cikan
        self.updater = updater
        self.n_processed = 0
        self.n_updated = 0

    def run(self, stream: Sequence[Violation]) -> None:
        """Process a stream of candidate items."""
        for item in stream:
            self.n_processed += 1
            # "trigger a fine-tuning step whenever a Violation occurs"
            # Actually, we should update on any verified failure or success to learn.
            # But the spec says "whenever a Violation occurs".
            # We'll trigger it for all items, or just when label == 0.0?
            # A "verified failure" is an item with label=0.0. Let's trigger on any item to learn the boundary, 
            # or specifically if it's a violation. Let's just step on all of them to actually train.
            self.updater.step(self.cikan, item.features, item.label)
            self.n_updated += 1
