"""Tests for SCG-MEM structural enforcer."""
import pytest
from carnot.memory.scg_adapter import ScgAdapter

def test_scg_adapter_valid_trace():
    """
    Test that a valid trace passes the SCG-MEM structural enforcer.
    Traces to: REQ-LEARN-1680, SCENARIO-LEARN-1680
    """
    adapter = ScgAdapter()
    valid_trace = {
        "trace_id": "tr-123",
        "memory_embedding": [0.1, 0.2, -0.5],
        "cognitive_context": "test context",
        "utility_score": 0.95
    }
    assert adapter.enforce_schema(valid_trace) is True

def test_scg_adapter_invalid_trace_missing_field():
    """
    Test that an invalid trace (missing required field) is rejected.
    Traces to: REQ-LEARN-1680, SCENARIO-LEARN-1680
    """
    adapter = ScgAdapter()
    invalid_trace = {
        "trace_id": "tr-124",
        "memory_embedding": [0.1, 0.2, -0.5]
        # missing cognitive_context
    }
    assert adapter.enforce_schema(invalid_trace) is False

def test_scg_adapter_invalid_trace_wrong_type():
    """
    Test that an invalid trace (wrong type) is rejected.
    Traces to: REQ-LEARN-1680, SCENARIO-LEARN-1680
    """
    adapter = ScgAdapter()
    invalid_trace = {
        "trace_id": "tr-125",
        "memory_embedding": "not an array",
        "cognitive_context": "test context"
    }
    assert adapter.enforce_schema(invalid_trace) is False

def test_scg_adapter_process_embeddings():
    """
    Test processing a list of traces.
    Traces to: REQ-LEARN-1680, SCENARIO-LEARN-1680
    """
    adapter = ScgAdapter()
    traces = [
        {"trace_id": "tr-1", "memory_embedding": [0.1], "cognitive_context": "ctx"},
        {"trace_id": "tr-2", "memory_embedding": "invalid", "cognitive_context": "ctx"},
        {"trace_id": "tr-3", "memory_embedding": [0.2], "cognitive_context": "ctx"}
    ]
    valid_traces = adapter.process_embeddings(traces)
    assert len(valid_traces) == 2
    assert valid_traces[0]["trace_id"] == "tr-1"
    assert valid_traces[1]["trace_id"] == "tr-3"
