import pytest
from carnot.verify.hierarchical_logcons import HierarchicalLogConsVerifier

def test_hierarchical_logcons_valid_structure():
    verifier = HierarchicalLogConsVerifier()
    entry = {"prompt": "test", "response_text": "text (with) [brackets] <valid>."}
    entry["semantic_energy_score"] = 0.0
    entry["laab_score"] = 0.0
    result = verifier.verify(entry)
    assert result["structural_valid"] is True

def test_hierarchical_logcons_invalid_structure():
    verifier = HierarchicalLogConsVerifier()
    entry = {"prompt": "test", "response_text": "text (with [invalid> brackets]"}
    entry["semantic_energy_score"] = 0.0
    entry["laab_score"] = 0.0
    result = verifier.verify(entry)
    assert result["structural_valid"] is False
    assert result["logcons_score"] >= 0.3

def test_hierarchical_logcons_z3_hierarchy_violation():
    verifier = HierarchicalLogConsVerifier()
    # Level 1 claimed, Level 2 required => 1 >= 2 is False (unsat)
    entry = {"prompt": "required_level=2", "response_text": "claimed_level=1"}
    result = verifier.verify(entry)
    assert result["z3_encoding_used"] is True
    assert result["hierarchy_violation"] is True
    assert result["logcons_score"] == 1.0
    
def test_hierarchical_logcons_z3_hierarchy_valid():
    verifier = HierarchicalLogConsVerifier()
    # Level 2 claimed, Level 1 required => 2 >= 1 is True (sat)
    entry = {"prompt": "required_level=1", "response_text": "claimed_level=2"}
    result = verifier.verify(entry)
    assert result["z3_encoding_used"] is True
    assert result["hierarchy_violation"] is False
