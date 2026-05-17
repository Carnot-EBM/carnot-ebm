"""Tests for CARM pipeline module."""

from carnot.pipeline.carm import CARM

def test_carm_retrieve_domains():
    """Test retrieving domains from natural language prompts. REQ-CARM"""
    carm = CARM()
    domains = carm.retrieve_domains("Please calculate the sum of 4 and 5")
    assert "arithmetic" in domains
    
    domains = carm.retrieve_domains("Write a python function that loops")
    assert "code" in domains
    
    domains = carm.retrieve_domains("If A is true then B must be true")
    assert "logic" in domains

def test_carm_retrieve_constraint_types():
    """Test retrieving constraint types from natural language prompts. REQ-CARM"""
    carm = CARM()
    types = carm.retrieve_constraint_types("If A then B")
    assert "implication" in types
    
    types = carm.retrieve_constraint_types("Return type should be int")
    assert "return_type" in types
