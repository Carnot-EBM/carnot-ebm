import pytest
from carnot.pipeline.nco_decoder import NCOConstraintDecoder

def test_nco_decoder_basic():
    patterns = ["12 + 7 = 20", "20 / 3 = 7"]
    decoder = NCOConstraintDecoder(patterns)
    
    # Should not match
    tokens = ["12", " +", " 7", " =", " 19"]
    res = decoder.decode(tokens)
    assert res["nco_rejection_score"] == 0
    assert res["n_patterns_fired"] == 0
    assert "INITIAL" in res["wfsa_states_visited"]
    
    # Should match first pattern
    tokens = ["12", " +", " 7", " =", " 20"]
    res = decoder.decode(tokens)
    assert res["nco_rejection_score"] == 1
    assert res["n_patterns_fired"] == 1
    assert "REJECTED" in res["wfsa_states_visited"]
    assert "TRACKING_0" in res["wfsa_states_visited"]
