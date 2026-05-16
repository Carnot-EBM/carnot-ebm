"""Tests for the Recursive Logic Subsystem (RLS) verifier stub.

Spec: REQ-EBT-203, SCENARIO-EBT-203
"""
from carnot.models.boltzmann.rls_verifier import verify_trace

def test_verify_trace_inconsistent():
    # SCENARIO-EBT-203
    trace = ["The sky is blue.", "This is a contradiction.", "Therefore, false."]
    energy = verify_trace(trace)
    assert energy > 10.0

def test_verify_trace_consistent():
    # SCENARIO-EBT-203
    trace = ["The sky is blue.", "Thus, we can see it."]
    energy = verify_trace(trace)
    assert energy == 0.0
