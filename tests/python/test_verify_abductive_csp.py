import pytest
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_abductive_csp_layer_initialization():
    pipeline = VerifyRepairPipeline(enable_abductive_csp=True)
    assert pipeline.abductive_csp_layer is not None

def test_abductive_csp_formulate_graph():
    pipeline = VerifyRepairPipeline(enable_abductive_csp=True)
    traces = ["A implies B", "B is true", "Therefore A is true"]
    graph = pipeline.abductive_csp_layer.formulate_graph(traces)
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) == 3

def test_abductive_csp_verify_coherence():
    pipeline = VerifyRepairPipeline(enable_abductive_csp=True)
    traces = ["A implies B", "A is true", "Therefore B is true"]
    result = pipeline.abductive_csp_layer.verify_coherence(traces)
    assert result["is_coherent"] is True
    assert result["energy"] == 0.0

def test_abductive_csp_verify_incoherence():
    pipeline = VerifyRepairPipeline(enable_abductive_csp=True)
    traces = ["A implies B", "A is true", "Therefore B is false"]
    result = pipeline.abductive_csp_layer.verify_coherence(traces)
    assert result["is_coherent"] is False
    assert result["energy"] > 0.0
