import pytest
from carnot.verify.tier0z_temporal_causal import TemporalCausalConsistencyVerifier

def test_temporal_causal_consistency_verifier():
    verifier = TemporalCausalConsistencyVerifier()
    
    # Empty or single sentence -> 0
    assert verifier.score("question", "") == 0.0
    assert verifier.score("q", "Just one sentence.") == 0.0
    
    # Consistent causality
    # "A happened. Therefore B happened." -> no backward temporal conflict -> 0 violations
    consistent_text = "The system failed. Therefore it crashed."
    assert verifier.score("q", consistent_text) == 0.0
    
    # Contradiction: forward causality but temporal backward
    # "Therefore" (causal forward) but "previously" (temporal backward)
    contradiction_text = "The system failed. Therefore it crashed previously."
    assert verifier.score("q", contradiction_text) > 0.0
    
    # Temporal sequence contradiction
    # "First" should be at the beginning
    first_at_end = "Sentence one. Sentence two. Sentence three. First, this happened."
    assert verifier.score("q", first_at_end) > 0.0
    
    # "Finally" should not have many sentences after it
    finally_early = "Finally, we conclude. But wait there is more. And more. And more."
    assert verifier.score("q", finally_early) > 0.0
    
    # Paradox: after and before in close proximity
    paradox = "It happened after the event but before the event."
    assert verifier.score("q", paradox) > 0.0
