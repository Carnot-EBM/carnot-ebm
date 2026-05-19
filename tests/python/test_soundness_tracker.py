import pytest
from carnot.verify.nsvif_z3_extractor import SoundnessCompletenessTracker
import math

def test_soundness_completeness_tracker_fr11():
    tracker = SoundnessCompletenessTracker(n_features=100)
    
    # prediction=safe (True), label=violation (False) -> soundness mistake
    tracker.update(True, False)
    assert tracker.soundness_mistakes == 1
    assert tracker.completeness_mistakes == 0
    
    # prediction=violation (False), label=safe (True) -> completeness mistake
    tracker.update(False, True)
    assert tracker.soundness_mistakes == 1
    assert tracker.completeness_mistakes == 1
    
    # prediction=safe (True), label=safe (True) -> correct
    tracker.update(True, True)
    assert tracker.soundness_mistakes == 1
    assert tracker.completeness_mistakes == 1
    
    # prediction=violation (False), label=violation (False) -> correct
    tracker.update(False, False)
    assert tracker.soundness_mistakes == 1
    assert tracker.completeness_mistakes == 1
    
    assert tracker.n_total == 4
    assert tracker.soundness_rate() == 0.25
    assert tracker.completeness_rate() == 0.25
    
    expected_bound = math.sqrt(2 * 1 * math.log(100))
    assert math.isclose(tracker.littlestone_soundness_bound(), expected_bound)

def test_soundness_completeness_tracker_zero_features():
    tracker = SoundnessCompletenessTracker(n_features=0)
    tracker.update(True, False)
    assert tracker.littlestone_soundness_bound() == 0.0
